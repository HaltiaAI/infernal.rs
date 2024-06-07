use std::{
    borrow::Cow,
    cell::{self, RefCell},
    fs::File,
    io::{Read, Seek, SeekFrom},
    ops::Deref,
    rc::Rc,
};

use memmap2::Mmap;
use thiserror::Error;
use tracing::trace;

use crate::{buf::Buf, BufId, ComputeGraph, Optimizer, Tensor, TensorSrc};

pub use metal::{ComputeError, Config, Metal};

mod metal;
mod paging;

pub trait Backend {
    fn add_buf(&mut self, buf: BackendBuf) -> Result<(), MemoryLimitTooLow>;

    #[inline(always)]
    fn get_tensor_buf(&self, t: &Tensor) -> Result<&BackendBuf, DanglingTensor> {
        self.get_buf(t.buf_id).ok_or_else(|| DanglingTensor::new(t))
    }

    fn get_buf(&self, buf_id: BufId) -> Option<&BackendBuf>;

    fn mem_limit(&self) -> usize;

    fn set_mem_limit(&mut self, mem_limit: usize) -> Result<(), MemoryLimitTooLow>;

    fn compute_graph<O: Optimizer>(
        &mut self,
        graph: &ComputeGraph<O>,
        ctx: &impl TensorSrc,
    ) -> Result<(), ComputeError>;
}

// TODO
pub struct MemLimit {
    pub ram: usize,
    pub vram: usize,
}

#[derive(Clone, Debug, Error)]
#[error(
    "The size needed by the backend ({needed}) exceeds the configured \
    max memory size ({limit}) and the backend cannot execute the compute \
    graph. You need to increase the max memory size to {needed} to \
    accomodate the operation. You can do this via \
    `Backend::set_mem_limit`."
)]
pub struct MemoryLimitTooLow {
    pub needed: usize,
    pub limit: usize,
}

#[derive(Clone, Debug, Error)]
#[error("tensor '{0}' buffer {1:?} not in page cache")]
pub struct DanglingTensor(pub String, pub BufId);

impl DanglingTensor {
    pub(crate) fn new(t: &Tensor) -> Self {
        Self(t.name.to_string(), t.buf_id)
    }
}

#[derive(Debug, Error)]
#[error("{0} device not found")]
pub struct DeviceNotFound(&'static str);

#[doc(hidden)]
pub trait IntoBackendBufs {
    type Data: Into<BackendData>;
    type Iter: Iterator<Item = (BufId, Self::Data)>;

    fn into_backend_bufs(self) -> Self::Iter;
}

pub struct BackendBuf {
    id: BufId,
    name: Cow<'static, str>,
    data: BackendData,
}

#[doc(hidden)]
pub enum BackendData {
    Mmap(Mmap),
    Owned(Buf),
    Shared(Rc<RefCell<Buf>>),
    Paged(PagedBuf),
    PagedMmap(PagedMmap),
}

impl From<Rc<RefCell<Buf>>> for BackendBuf {
    #[inline(always)]
    fn from(buf: Rc<RefCell<Buf>>) -> Self {
        let id = BufId::from_addr(&buf.borrow());
        Self {
            id,
            name: String::new().into(),
            data: BackendData::Shared(buf),
        }
    }
}

impl BackendBuf {
    pub(crate) fn new(id: BufId, ctx_name: &str, data: BackendData) -> Self {
        Self {
            id,
            name: format!("{ctx_name}#{}", id.idx()).into(),
            data,
        }
    }

    pub fn with_name(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        self.name = name.into();
        self
    }

    #[inline(always)]
    pub fn id(&self) -> BufId {
        self.id
    }

    // Returns the byte size of the buffer.
    //
    // # Important
    //
    // This function potentially dynamically borrows self if this is a shared buffer! Therefore,
    // user must make sure to only call this function if the buffer is not already borrowed.
    #[inline(always)]
    pub(crate) fn size(&self) -> usize {
        use BackendData::*;
        match &self.data {
            Mmap(buf) => buf.len(),
            Owned(buf) => buf.len(),
            Shared(buf) => RefCell::borrow(buf).len(),
            Paged(page) => page.size,
            PagedMmap(page) => page.size,
        }
    }

    #[inline(always)]
    pub fn borrow(&self) -> BufRef {
        use BackendData::*;
        match &self.data {
            Mmap(buf) => BufRef {
                id: self.id,
                inner: BufRefInner::Slice(buf),
            },
            Owned(buf) => BufRef {
                id: self.id,
                inner: BufRefInner::Slice(buf),
            },
            Shared(buf) => BufRef {
                id: self.id,
                // TODO: return errors here
                inner: BufRefInner::Rc(
                    buf.try_borrow()
                        .expect("backend cannot mutate buffer while it's being read"),
                ),
            },
            Paged(buf) => BufRef {
                id: self.id,
                inner: BufRefInner::Slice(buf.data.as_ref().expect("paged buffer not loaded")),
            },
            PagedMmap(buf) => BufRef {
                id: self.id,
                inner: BufRefInner::Slice(buf.mmap.as_ref().expect("paged mmap not loaded")),
            },
        }
    }

    #[inline(always)]
    pub fn as_mmap(&self) -> Option<&Mmap> {
        match &self.data {
            BackendData::Mmap(buf) => Some(buf),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn as_shared(&self) -> Option<&Rc<RefCell<Buf>>> {
        match &self.data {
            BackendData::Shared(buf) => Some(buf),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn is_offloadable(&self) -> bool {
        matches!(self.data, BackendData::Paged(_) | BackendData::PagedMmap(_))
    }

    pub fn status(&self) -> BufStatus {
        match &self.data {
            BackendData::Paged(buf) => {
                if buf.data.is_some() {
                    BufStatus::Loaded
                } else {
                    BufStatus::Offloaded
                }
            }
            BackendData::PagedMmap(buf) => {
                if buf.mmap.is_some() {
                    BufStatus::Loaded
                } else {
                    BufStatus::Offloaded
                }
            }
            _ => BufStatus::Loaded,
        }
    }

    /// Pages in the buffer if it's paged out.
    pub(crate) fn load(&mut self) -> std::io::Result<()> {
        match &mut self.data {
            BackendData::Paged(buf) => buf.load(),
            BackendData::PagedMmap(buf) => buf.load(),
            _ => Ok(()),
        }
    }

    /// Pages out the buffer if it's an offloadable buffer that is paged in. The returned status
    /// tells us whether the buffer was paged out.
    // TODO: consider ability to offload write-tensors in which case we'd need to write to disk and
    // return an io::Result here
    pub(crate) fn offload(&mut self) -> BufStatus {
        match &mut self.data {
            BackendData::Paged(buf) => {
                buf.offload();
                BufStatus::Offloaded
            }
            BackendData::PagedMmap(buf) => {
                buf.offload();
                BufStatus::Offloaded
            }
            _ => BufStatus::Loaded,
        }
    }
}

pub struct BufRef<'a> {
    id: BufId,
    inner: BufRefInner<'a>,
}

enum BufRefInner<'a> {
    Slice(&'a [u8]),
    Rc(cell::Ref<'a, Buf>),
}

impl BufRef<'_> {
    #[inline(always)]
    pub fn len(&self) -> usize {
        match &self.inner {
            BufRefInner::Slice(buf) => buf.len(),
            BufRefInner::Rc(buf) => buf.len(),
        }
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        match &self.inner {
            BufRefInner::Slice(buf) => buf.is_empty(),
            BufRefInner::Rc(buf) => buf.is_empty(),
        }
    }

    pub fn tensor_data(&self, tensor: &Tensor) -> &[u8] {
        assert_eq!(self.id, tensor.buf_id);
        let offs = tensor.data_offs;
        let size = tensor.size();
        match &self.inner {
            BufRefInner::Slice(buf) => &buf[offs..offs + size],
            BufRefInner::Rc(buf) => &buf[offs..offs + size],
        }
    }
}

impl Deref for BufRef<'_> {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        match &self.inner {
            BufRefInner::Slice(buf) => buf,
            BufRefInner::Rc(buf) => buf,
        }
    }
}

impl From<Mmap> for BackendData {
    #[inline(always)]
    fn from(buf: Mmap) -> Self {
        Self::Mmap(buf)
    }
}
impl From<Buf> for BackendData {
    #[inline(always)]
    fn from(buf: Buf) -> Self {
        Self::Owned(buf)
    }
}
impl From<Rc<RefCell<Buf>>> for BackendData {
    #[inline(always)]
    fn from(buf: Rc<RefCell<Buf>>) -> Self {
        Self::Shared(buf)
    }
}
impl From<PagedBuf> for BackendData {
    #[inline(always)]
    fn from(buf: PagedBuf) -> Self {
        Self::Paged(buf)
    }
}
impl From<PagedMmap> for BackendData {
    #[inline(always)]
    fn from(buf: PagedMmap) -> Self {
        Self::PagedMmap(buf)
    }
}

pub struct PagedBuf {
    pub(crate) file: File,
    pub(crate) offs: usize,
    pub(crate) size: usize,
    pub(crate) data: Option<Vec<u8>>,
}

impl PagedBuf {
    pub(crate) fn load(&mut self) -> std::io::Result<()> {
        if self.data.is_none() {
            trace!("loading buffer from disk");
            self.file.seek(SeekFrom::Start(self.offs as u64))?;
            let mut data = vec![0; self.size];
            self.file.read_exact(&mut data)?;
            self.data = Some(data);
        }
        Ok(())
    }

    pub(crate) fn offload(&mut self) {
        if self.data.is_some() {
            trace!("offloading buffer from RAM");
            self.data = None;
        }
    }
}

pub struct PagedMmap {
    // TODO: we may want to keep around the path instead of the file so that hibernation of app or
    // doesn't break things. not sure if an actual issue so not important rn. the only issue with
    // that is potential latency for reopening the same fd
    pub(crate) file: File,
    /// The absolute offset into the file where the data starts.
    pub(crate) offs: usize,
    /// The size of the page. TODO: It needs to be page aligned to the system size.
    pub(crate) size: usize,
    pub(crate) mmap: Option<Mmap>,
}

impl PagedMmap {
    pub(crate) fn load(&mut self) -> std::io::Result<()> {
        if self.mmap.is_none() {
            trace!("loading buffer from disk");
            let mmap = crate::alloc::mmap::mmap(&self.file, Some(self.offs), Some(self.size))?;
            self.mmap = Some(mmap);
        }
        Ok(())
    }

    pub(crate) fn offload(&mut self) {
        if self.mmap.is_some() {
            trace!("offloading mmap from RAM");
            self.mmap = None;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufStatus {
    Loaded,
    Offloaded,
}

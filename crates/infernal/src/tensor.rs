use core::fmt;
use std::{
    borrow::Cow,
    mem,
    ops::{Deref, Index, IndexMut},
};

use crate::{
    backend::Backend, buf::BufId, builder::RawParts, op::Op, GetCtxId, Layout, TensorNotFound,
    TensorSrc,
};

// TODO: add newtype
pub type CtxId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TensorId {
    // TODO: consider making types u16 -- unlikely to have more tensors than that per context anyway
    ctx: u32,
    idx: u32,
}

impl fmt::Display for TensorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}#{}]", self.ctx, self.idx)
    }
}

impl TensorId {
    pub(crate) const fn new(ctx: CtxId, idx: u32) -> Self {
        Self { ctx, idx }
    }

    #[inline(always)]
    pub const fn ctx(&self) -> CtxId {
        self.ctx
    }

    #[inline(always)]
    pub const fn idx(&self) -> usize {
        self.idx as usize
    }
}

/// Read-only list of tensors and their metadata.
pub struct TensorCtx {
    pub(crate) id: CtxId,
    // TODO: impl Display with id + name
    pub(crate) tensors: Vec<Tensor>,
    pub(crate) name: Cow<'static, str>,
}

impl fmt::Debug for TensorCtx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorCtx")
            .field("id", &self.id)
            .field("tensors", &self.tensors.len())
            .field("name", &self.name)
            .finish()
    }
}

impl fmt::Display for TensorCtx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{{}:{}}}", self.id, self.name)
    }
}

impl TensorCtx {
    #[inline(always)]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[inline(always)]
    pub fn get(&self, id: TensorId) -> Option<&Tensor> {
        if id.ctx() == self.id {
            self.tensors.get(id.idx())
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn get_mut(&mut self, id: TensorId) -> Option<&mut Tensor> {
        if id.ctx() == self.id {
            self.tensors.get_mut(id.idx())
        } else {
            None
        }
    }

    // TODO: probably remove in favor of a better API
    pub fn into_raw_parts(self) -> RawParts {
        RawParts {
            id: self.id,
            tensors: self.tensors,
            name: Some(self.name),
            auto_op_names: Default::default(),
        }
    }
}

impl GetCtxId for TensorCtx {
    #[inline(always)]
    fn id(&self) -> CtxId {
        self.id
    }
}

impl TensorSrc for TensorCtx {
    #[inline(always)]
    fn get_tensor(&self, id: TensorId) -> Result<&Tensor, TensorNotFound> {
        self.get(id).ok_or(TensorNotFound(id))
    }
}

impl Index<TensorId> for TensorCtx {
    type Output = Tensor;

    #[inline(always)]
    fn index(&self, id: TensorId) -> &Self::Output {
        assert_eq!(id.ctx(), self.id, "tensor does not belong to this context");
        assert!(id.idx() < self.tensors.len(), "tensor index out of bounds");
        &self.tensors[id.idx()]
    }
}

impl IndexMut<TensorId> for TensorCtx {
    #[inline(always)]
    fn index_mut(&mut self, id: TensorId) -> &mut Self::Output {
        assert_eq!(id.ctx(), self.id, "tensor does not belong to this context");
        assert!(id.idx() < self.tensors.len(), "tensor index out of bounds");
        &mut self.tensors[id.idx()]
    }
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub(crate) id: TensorId,
    pub(crate) layout: Layout,
    pub(crate) op: Op,
    /// The relative offset into the data buffer referred to by `buf_id`.
    pub(crate) data_offs: usize,
    /// The unique ID of the buffer that contains the data for this tensor.
    pub(crate) buf_id: BufId,
    pub(crate) name: Cow<'static, str>,
    /// If this tensor is a view of another tensor, this is the ID and buffer offset of the original tensor.
    pub(crate) view: Option<View>,
    // #[cfg(not(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios"))))]
    pub(crate) loc: MemoryLocation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct View {
    /// The id of the source tensor on which this view is based.
    pub origin: TensorId,
    /// The offset into the source tensor's data buffer. This means that [`Tensor::data_offs`] is
    /// the view source tensor's data offset plus this value.
    pub offs: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryLocation {
    /// Memory is allocated on the host RAM.
    ///
    /// On Apple Silicon this is the unified memory. On other platforms this is the CPU RAM.
    Ram,
    /// Memory is allocated on the GPU vRAM.
    ///
    /// This option is only available on platforms without unified memory.
    Vram,
    // Split, TODO
}

impl Default for MemoryLocation {
    #[inline(always)]
    fn default() -> Self {
        Self::Ram
    }
}

impl Deref for Tensor {
    type Target = Layout;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.layout
    }
}

impl From<&'_ mut Tensor> for TensorId {
    #[inline(always)]
    fn from(t: &'_ mut Tensor) -> Self {
        t.id
    }
}

impl Tensor {
    #[inline(always)]
    pub fn id(&self) -> TensorId {
        self.id
    }

    #[inline(always)]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    #[inline(always)]
    pub fn op(&self) -> &Op {
        &self.op
    }

    #[inline(always)]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[inline(always)]
    pub fn buf_id(&self) -> BufId {
        self.buf_id
    }

    #[inline(always)]
    pub fn data_offs(&self) -> usize {
        self.data_offs
    }

    #[inline(always)]
    pub fn view(&self) -> Option<&View> {
        self.view.as_ref()
    }

    #[inline(always)]
    pub fn is_view(&self) -> bool {
        self.view.is_some()
    }

    #[inline(always)]
    pub fn set_name(&mut self, name: impl Into<Cow<'static, str>>) -> &mut Self {
        self.name = name.into();
        self
    }

    // TODO: these shouldn't be available after TensorBuilder::finish. we can solve this by tagging
    // the tensor with a phantomtype as WIP/done, and exposing methods accordingly, but feels a bit
    // heavy handed, plus it wouldn't allow moving to the tensor to another context without reallocs
    // (transmuting might work since it would be a ZST, but it's hacky).
    #[inline(always)]
    pub fn set_loc(&mut self, loc: MemoryLocation) -> &mut Self {
        self.loc = loc;
        self
    }

    #[inline(always)]
    pub fn offload(&mut self, offload: bool) -> &mut Self {
        if offload {
            self.set_loc(MemoryLocation::Vram)
        } else {
            self
        }
    }

    #[inline(always)]
    pub(crate) fn reassign_buf(&mut self, buf_id: BufId, data_offs: usize) {
        self.buf_id = buf_id;
        self.data_offs = data_offs;
    }

    /// Two different tensors may be backed by the exact same buffer, e.g. if one tensor is
    /// a transpose of another, or a different dimensional view.
    #[inline(always)]
    pub fn has_same_buf(&self, other: &Self) -> bool {
        self.buf_id == other.buf_id && self.data_offs == other.data_offs
    }

    #[inline(always)]
    pub fn elements<'b, T>(&self, backend: &impl Backend) -> &'b [T] {
        let buf = backend.get_tensor_buf(self).unwrap().borrow();
        let data = buf.tensor_data(self);
        // Check if T is the same as the tensor's type.
        self.assert_type_match::<T>();
        // SAFETY: We know at this point that the data is of type T.
        // TODO: should we check alignment?
        // TODO: check if tensor is contiguous and return error if not
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, self.layout.total_elems()) }
    }

    pub(crate) fn assert_type_match<T>(&self) {
        // This means only F32 and I32 are supported for now (and later I8, I16).
        let type_name = std::any::type_name::<T>();
        assert_eq!(
            type_name,
            self.typ().name(),
            "cannot cast tensor data to type '{}', tensor is of type '{:?}'",
            type_name,
            self.typ()
        );
        // Sanity check.
        assert_eq!(
            self.typ().size(),
            mem::size_of::<T>(),
            "tensor type size mismatch while casting tensor to '{}', tensor is of type '{:?}'",
            type_name,
            self.typ()
        );
    }

    /// Visits self, then all parents of this node tensor.
    #[inline(always)]
    pub(crate) fn visit_nodes(&self, src: &impl TensorSrc, mut f: impl FnMut(&Tensor)) {
        f(self);
        for p in self.op.parents() {
            let p = src
                .get_tensor(p)
                .expect("parent tensor must be present in source");
            f(p);
        }
    }

    /// Visits self, then all parents of this node tensor.
    #[inline(always)]
    pub(crate) fn try_visit_nodes<E>(
        &self,
        src: &impl TensorSrc,
        mut f: impl FnMut(&Tensor) -> Result<(), E>,
    ) -> Result<(), E> {
        f(self)?;
        for p in self.op.parents() {
            let p = src
                .get_tensor(p)
                .expect("parent tensor must be present in source");
            f(p)?;
        }
        Ok(())
    }

    pub(crate) fn loc(&self) -> MemoryLocation {
        self.loc
    }
}

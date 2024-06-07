use std::{
    borrow::Cow,
    cell::RefCell,
    collections::HashMap,
    fs::File,
    io, mem,
    ops::{Deref, Index, IndexMut},
};

use thiserror::Error;

use crate::{
    alloc::{
        bump::BumpAlloc,
        graph::{GraphAlloc, GraphArgs},
        measure::AllocMeasure,
        mmap::{MappingArgs, MmapAlloc},
        AllocSize, Oom, ReadWriteAlloc, TensorAlloc,
    },
    backend::{Backend, BackendBuf, IntoBackendBufs, MemoryLimitTooLow},
    buf::{BufId, SliceCast},
    op::Op,
    tensor::{CtxId, MemoryLocation, View},
    ComputeGraph, GetCtxId, Layout, Tensor, TensorCtx, TensorId, TensorNotFound, TensorSrc, Type,
};

/// Convenience type for the most used tensor builder.
pub type GraphBuilder<'a, B, S> = TensorBuilder<&'a mut GraphAlloc<B>, S>;

#[derive(Debug)]
pub struct TensorOptions {
    layout: Layout,
    name: Option<Cow<'static, str>>,
    loc: Option<MemoryLocation>,
    // mmap_offs: Option<usize>,
    prealloc: bool,
}

impl TensorOptions {
    #[inline(always)]
    pub fn new(layout: Layout) -> Self {
        Self {
            layout,
            name: None,
            loc: None,
            // mmap_offs: None,
            prealloc: false,
        }
    }
}

impl From<&'_ Layout> for usize {
    #[inline(always)]
    fn from(layout: &Layout) -> Self {
        layout.size()
    }
}

impl From<&'_ Layout> for GraphArgs {
    fn from(layout: &Layout) -> Self {
        Self {
            size: layout.size(),
            prealloc: false,
        }
    }
}

impl From<&'_ TensorOptions> for usize {
    #[inline(always)]
    fn from(to: &TensorOptions) -> Self {
        to.layout.size()
    }
}

impl From<&'_ TensorOptions> for GraphArgs {
    fn from(to: &TensorOptions) -> Self {
        Self {
            size: to.layout.size(),
            prealloc: to.prealloc,
        }
    }
}

pub trait TensorOptionsExt {
    fn with_name(self, name: impl Into<Cow<'static, str>>) -> TensorOptions;
    fn with_loc(self, location: MemoryLocation) -> TensorOptions;
    fn with_type(self, typ: Type) -> TensorOptions;
    fn prealloc(self) -> TensorOptions;
}

impl<T> TensorOptionsExt for T
where
    T: Into<TensorOptions>,
{
    #[inline(always)]
    fn with_name(self, name: impl Into<Cow<'static, str>>) -> TensorOptions {
        let mut this = self.into();
        this.name = Some(name.into());
        this
    }

    #[inline(always)]
    fn with_loc(self, loc: MemoryLocation) -> TensorOptions {
        let mut this = self.into();
        this.loc = Some(loc);
        this
    }

    #[inline(always)]
    fn with_type(self, typ: Type) -> TensorOptions {
        let mut this = self.into();
        this.layout = Layout::new(this.layout.raw_dims(), typ);
        this
    }

    /// This is only relevant when using the graph allocator.
    #[inline(always)]
    fn prealloc(self) -> TensorOptions {
        let mut this = self.into();
        this.prealloc = true;
        this
    }
}

impl<T: Into<Layout>> From<T> for TensorOptions {
    #[inline(always)]
    fn from(value: T) -> Self {
        Self::new(value.into())
    }
}

#[derive(Clone, Copy, Debug)]
pub enum AutoOpName {
    /// Do not auto name operations.
    ///
    /// Use this for the best performance and avoiding heap allocations.
    Disabled,
    /// Auto names tensors based on the op and the source tensor IDs. Does not require [`TensorCtx`]
    /// of sources to be present.
    Simple,
    /// Note that expanding source tensor names is only available if a [`TensorCtx`] is available for
    /// all source tensors. Otherwise, the various methods to create tensor ops will panic.
    Expanded {
        /// If enabled, includes the tensor ID in the name for each parent tensor.
        ///
        /// This is quite noisy so only useful in specific scenarios.
        with_id: bool,
    },
}

impl Default for AutoOpName {
    #[inline(always)]
    fn default() -> Self {
        Self::Disabled
    }
}

impl AutoOpName {
    pub fn expand() -> Self {
        Self::Expanded { with_id: false }
    }
}

#[derive(Default)]
pub struct NoSrc;

impl TensorSrc for NoSrc {
    #[inline(always)]
    fn get_tensor(&self, id: TensorId) -> Result<&Tensor, TensorNotFound> {
        Err(TensorNotFound(id))
    }
}

/// Tensor builder used for building tensors from the mmap'd weights.
pub type WeightsBuilder = TensorBuilder<MmapAlloc, NoSrc>;

#[derive(Default)]
pub struct TensorBuilder<A, S: TensorSrc = NoSrc> {
    id: u32,
    pub(crate) alloc: A,
    src: S,
    pub(crate) tensors: Vec<Tensor>,
    name: Option<Cow<'static, str>>,
    pub(crate) auto_op_names: AutoOpName,
}

#[derive(Default)]
pub struct RawParts {
    pub(crate) id: u32,
    pub(crate) tensors: Vec<Tensor>,
    pub(crate) name: Option<Cow<'static, str>>,
    pub(crate) auto_op_names: AutoOpName,
}

// --- mmap alloc ---
impl TensorBuilder<MmapAlloc, NoSrc> {
    #[inline(always)]
    pub fn mmap(file: File, data_offs: usize) -> io::Result<Self> {
        Ok(Self::new(MmapAlloc::map(file, data_offs)?))
    }
}
impl<S: TensorSrc> TensorBuilder<MmapAlloc, S> {
    #[inline(always)]
    pub fn mmap_with_src(file: File, data_offs: usize, src: S) -> io::Result<Self> {
        Ok(Self::new_with_src(MmapAlloc::map(file, data_offs)?, src))
    }

    pub fn new_mapped_tensor(
        &mut self,
        t: impl Into<TensorOptions>,
        offs: usize,
    ) -> Result<TensorId, Oom> {
        let t = t.into();
        let (buf_id, data_offs) = self.mmap_tensor_buf(t.layout, offs, t.prealloc)?;
        Ok(self.new_tensor_impl(t.layout, Op::Const, t.name, buf_id, data_offs, None, t.loc))
    }

    fn mmap_tensor_buf(
        &mut self,
        layout: Layout,
        offs: usize,
        prealloc: bool,
    ) -> Result<(BufId, usize), Oom> {
        let buf = self.alloc.alloc(MappingArgs {
            size: layout.size(),
            offs,
            prealloc,
        })?;
        Ok((buf.id, buf.offs))
    }
}

// --- bump alloc ---
impl TensorBuilder<BumpAlloc, NoSrc> {
    #[inline(always)]
    pub fn prealloc(size: usize) -> Result<Self, Oom> {
        Ok(Self::new(BumpAlloc::new(size)?))
    }
}
impl<S: TensorSrc> TensorBuilder<BumpAlloc, S> {
    #[inline(always)]
    pub fn prealloc_with_src(size: usize, src: S) -> Result<Self, Oom> {
        Ok(Self::new_with_src(BumpAlloc::new(size)?, src))
    }
}
impl<'buf, S: TensorSrc + Default> TensorBuilder<BumpAlloc<&'buf mut [u8]>, S> {
    /// Reuses an existing buffer to build new tensors.
    ///
    /// One use case for this is to rebuild the compute graph in an ML inferencing loop while
    /// keeping the existing buffer for efficiency, without changing any of the anciliary structures
    /// like the [compute graph](crate::graph::ComputeGraph) or mapped backend buffers.
    /// This possible because a tensor context always reuses tensor ids, so the previous references
    /// in the compute graph will remain valid. Thus, when rebuilding, only the underlying tensor
    /// metadata changes.
    ///
    /// Note that this also means that not creating the same tensors on the second pass but reusing
    /// the existing compute graph and backend buffers will result in
    /// a [`TensorNotFound`](super::TensorNotFound) error. However, there are *no* panics -- it's up
    /// to the user to rebuild their graphs correctly but the library will handle it gracefully if
    /// not.
    #[inline(always)]
    pub fn reuse(buf: &'buf mut [u8]) -> Self {
        Self::new(BumpAlloc::with_buf(buf))
    }
}
impl<'buf, S: TensorSrc> TensorBuilder<BumpAlloc<&'buf mut [u8]>, S> {
    /// See documentation for [`Self::reuse`].
    #[inline(always)]
    pub fn reuse_with_src(buf: &'buf mut [u8], src: S) -> Self {
        Self::new_with_src(BumpAlloc::with_buf(buf), src)
    }
}

// --- measure alloc ---
impl TensorBuilder<AllocMeasure, NoSrc> {
    #[inline(always)]
    pub fn measure() -> Self {
        Self::new(AllocMeasure::default())
    }
}
impl<S: TensorSrc> TensorBuilder<AllocMeasure, S> {
    #[inline(always)]
    pub fn measure_with_src(src: S) -> Self {
        Self::new_with_src(AllocMeasure::default(), src)
    }
}

// --- graph alloc ---
impl<'a, B: Deref<Target = [u8]>, S: TensorSrc> TensorBuilder<&'a mut GraphAlloc<B>, S> {
    #[inline(always)]
    pub fn graph_alloc_with_src(alloc: &'a mut GraphAlloc<B>, src: S) -> Self {
        Self::new_with_src(alloc, src)
    }

    #[inline(always)]
    pub fn alloc_graph(&mut self, graph: &ComputeGraph, mem_limit: usize) -> Result<usize, Oom> {
        self.alloc
            .alloc(graph, mem_limit, &mut self.tensors, self.id)
    }
}

// --- read-write ---
impl<A: ReadWriteAlloc, S: TensorSrc> TensorBuilder<A, S> {
    pub fn set_at_1d<T: Copy>(&mut self, t: TensorId, col: usize, value: T) {
        assert_eq!(t.ctx(), self.id);
        let t = &self.tensors[t.idx()];
        t.assert_type_match::<T>();
        let data = &mut self
            .alloc
            .get_mut_slice(t.buf_id, t.data_offs, t.size())
            .unwrap();
        let cols = &mut unsafe { mem::transmute::<_, &mut [T]>(&mut data[..]) }[..t.cols()];
        cols[col] = value;
    }

    #[inline(always)]
    // #[deprecated(note = "use `fill` instead")]
    pub fn fill_1d_f32(&mut self, t: TensorId, value: f32) {
        self.fill(t, value);
    }

    #[inline(always)]
    // #[deprecated(note = "use `fill` instead")]
    pub fn fill_2d_f32(&mut self, t: TensorId, value: f32) {
        self.fill(t, value);
    }

    #[inline(always)]
    pub fn fill<T: Copy>(&mut self, t: TensorId, v: T) {
        self.fill_with(t, || v)
    }

    pub fn fill_with<T: Copy>(&mut self, t: TensorId, mut f: impl FnMut() -> T) {
        assert_eq!(t.ctx(), self.id);
        let t = &self.tensors[t.idx()];
        t.assert_type_match::<T>();

        let all_rows = t.total_rows();
        let cols = t.cols();
        let row_stride = t.row_stride();
        // SAFETY: Tensor is created by this builder so we know that the buffer id is valid.
        let data = &mut self
            .alloc
            .get_mut_slice(t.buf_id, t.data_offs, t.size())
            .unwrap();
        for row in 0..all_rows {
            let row_start = row * row_stride;
            // SAFETY: we know that the data is T due to the type check above.
            let row = &mut data[row_start..].cast_mut()[..cols];
            for col in row {
                *col = f();
            }
        }
    }

    /// Copies data from the tensor to the given output array `to`, starting at `from_elem`, which
    /// is in terms of `T` and not the byte offset.
    pub fn copy_data_to<T: Copy>(&self, from: TensorId, from_elem: usize, to: &mut [T]) {
        assert_eq!(from.ctx(), self.id);
        let t = &self.tensors[from.idx()];
        t.assert_type_match::<T>();

        let from = self
            .alloc
            .get_slice(t.buf_id, t.data_offs, t.size())
            .expect("source tensor buffer must be present in its builder");
        Self::assert_type_alignment::<T>(from);
        // SAFETY: This is safe because above we've verified the tensor's type is the same as the
        // target type and that the buffer length is a multiple of the target type size.
        let from: &[T] = from.cast();
        let from = &from[from_elem..from_elem + to.len()];
        assert!(
            to.len() >= from.len(),
            "destination buffer is too small: from={}, to={}",
            from.len(),
            to.len()
        );
        to.copy_from_slice(from);
    }

    /// Copies data into the tensor from the given input array `from`.
    pub fn copy_data_from<T: Copy>(&mut self, to: TensorId, from: &[T]) {
        assert_eq!(to.ctx(), self.id);
        let t = &self.tensors[to.idx()];
        t.assert_type_match::<T>();

        let to = self
            .alloc
            .get_mut_slice(t.buf_id, t.data_offs, t.size())
            .expect("source tensor buffer must be present in its builder");
        Self::assert_type_alignment::<T>(to);

        // SAFETY: This is safe because above we've verified the tensor's type is the same as the
        // target type and that the buffer length is a multiple of the target type size.
        unsafe {
            to.as_mut_ptr()
                .cast::<T>()
                .copy_from(from.as_ptr(), from.len());
        }
        // let to: &mut [T] = unsafe { mem::transmute(to) };
        // assert!(to.len() >= from.len(), "destination buffer is too small");
        // to[..from.len()].copy_from_slice(from)
    }

    fn assert_type_alignment<T: Copy>(buf: &[u8]) {
        assert_eq!(
            buf.len() % mem::size_of::<T>(),
            0,
            "source buffer length must be a multiple of the target type size"
        );
    }
}

fn next_ctx_id() -> CtxId {
    thread_local! {
        static CONTEXT_ID: RefCell<u32> = const { RefCell::new(0) };
    }
    CONTEXT_ID.with(|id| {
        let mut id = id.borrow_mut();
        let prev = *id;
        *id += 1;
        prev
    })
}

impl<A: TensorAlloc, S: TensorSrc + Default> TensorBuilder<A, S> {
    pub fn new(alloc: A) -> Self {
        Self {
            id: next_ctx_id(),
            alloc,
            name: Default::default(),
            src: Default::default(),
            tensors: Default::default(),
            auto_op_names: AutoOpName::Disabled,
        }
    }
}

impl<A: TensorAlloc, S: TensorSrc + GetCtxId> TensorBuilder<A, HashMap<CtxId, S>> {
    /// Inserts a single [`TensorSrc`] into the source map.
    #[inline(always)]
    pub fn with_src(mut self, src: S) -> Self {
        self.src.insert(src.id(), src);
        self
    }
}

impl<A: TensorAlloc, S: TensorSrc> TensorBuilder<A, S> {
    pub fn new_with_src(alloc: A, src: S) -> Self {
        Self {
            id: next_ctx_id(),
            alloc,
            name: Default::default(),
            src,
            tensors: Default::default(),
            auto_op_names: AutoOpName::Disabled,
        }
    }

    pub fn alloc(&self) -> &A {
        &self.alloc
    }

    #[doc(hidden)] // Only used by the test suite.
    pub fn alloc_mut(&mut self) -> &mut A {
        &mut self.alloc
    }

    /// Sets the name of the tensor context.
    ///
    /// This name is used by the Metal backend to name buffers associated with this context. This is
    /// useful for debugging.
    #[inline(always)]
    pub fn with_name(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Enable automatic naming of operation tensors.
    ///
    /// For example, if two const tensors x and y are added together, the resulting tensor is named
    /// `(x + y)`. This is useful for debugging.
    #[inline(always)]
    pub fn with_auto_op_names(mut self, a: AutoOpName) -> Self {
        self.auto_op_names = a;
        self
    }

    /// Reuse some parts of a previously used tensor context to reuse allocated memory.
    #[inline(always)]
    pub fn with_raw_parts(mut self, parts: RawParts) -> Self {
        self.id = parts.id;
        self.tensors = parts.tensors;
        self.name = parts.name;
        self.auto_op_names = parts.auto_op_names;
        self
    }

    /// Reuse the allocated memory of another now unused tensor context.
    #[inline(always)]
    pub fn with_reused_memory<A2: TensorAlloc, S2: TensorSrc>(
        mut self,
        other: TensorBuilder<A2, S2>,
    ) -> Self {
        self.tensors = other.tensors;
        self.tensors.clear();
        self
    }

    #[inline(always)]
    pub fn data_alloc_size(&self) -> usize
    where
        A: AllocSize,
    {
        self.alloc.alloc_size()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_tensor_impl(
        &mut self,
        layout: Layout,
        op: Op,
        name: Option<Cow<'static, str>>,
        buf_id: BufId,
        data_offs: usize,
        view: Option<View>,
        loc: Option<MemoryLocation>,
    ) -> TensorId {
        // NB this has to come first not to mess up the tensor ID assignment.
        let id = TensorId::new(self.id, self.tensors.len() as u32);
        self.tensors.push(Tensor {
            id,
            layout,
            op,
            buf_id,
            data_offs,
            name: name.unwrap_or_default(),
            view,
            loc: loc.unwrap_or_default(),
        });
        id
    }

    #[inline(always)]
    pub fn get(&self, id: TensorId) -> Option<&Tensor> {
        if id.ctx() == self.id {
            self.tensors.get(id.idx())
        } else {
            self.src.get_tensor(id).ok()
        }
    }

    #[inline(always)]
    pub fn get_mut(&mut self, id: TensorId) -> Option<&mut Tensor> {
        if id.ctx() == self.id {
            self.tensors.get_mut(id.idx())
        } else {
            // NB: Can't return a mutable tensor from a read-only source.
            None
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Finishes building the tensors by loading the data onto the backend and returning the
    /// finalized tensor context.
    pub fn finish(
        mut self,
        // TODO: use generic Backend trait once done
        backend: &mut impl Backend,
        // TODO: make result generic from Backend
    ) -> Result<(TensorCtx, S), BackendBufError>
    where
        A: IntoBackendBufs,
    {
        let name = if let Some(name) = self.name {
            name
        } else {
            Cow::Owned(format!("ctx_{}", self.id))
        };

        self.alloc.finalize_tensors(&mut self.tensors)?;

        for (id, data) in self.alloc.into_backend_bufs() {
            let buf = BackendBuf::new(id, &name, data.into());
            backend.add_buf(buf)?;
        }

        Ok((
            TensorCtx {
                id: self.id,
                tensors: self.tensors,
                name,
            },
            self.src,
        ))
    }

    /// Similar to [`Self::finish`] but does not load the data onto the backend, because the underlying buffer
    /// is a reference that is manually loaded into the backend.
    pub fn finish_discard_buf(self) -> TensorCtx {
        let name = if let Some(name) = self.name {
            name
        } else {
            Cow::Owned(format!("ctx_{}", self.id))
        };

        TensorCtx {
            id: self.id,
            tensors: self.tensors,
            name,
        }
    }

    /// Turns the self into a tensor builder with no source. This can help resolve lifetime issues
    /// because the tensor builder references other contexts and builders rather than owns them.
    pub fn into_no_src(self) -> TensorBuilder<A, NoSrc> {
        TensorBuilder {
            id: self.id,
            alloc: self.alloc,
            src: NoSrc,
            tensors: self.tensors,
            name: self.name,
            auto_op_names: self.auto_op_names,
        }
    }

    pub fn into_raw_parts(mut self) -> RawParts {
        self.tensors.clear();
        RawParts {
            id: self.id,
            tensors: self.tensors,
            name: self.name,
            auto_op_names: self.auto_op_names,
        }
    }
}

#[derive(Debug, Error)]
pub enum BackendBufError {
    #[error(transparent)]
    MemoryLimitTooLow(#[from] MemoryLimitTooLow),
    #[error("buffer IO error: {0}")]
    Io(#[from] io::Error),
}

impl<A: TensorAlloc, S: TensorSrc> TensorBuilder<A, S>
where
    A::Arg: for<'a> From<&'a TensorOptions>,
{
    #[inline(always)]
    pub fn new_tensor(&mut self, t: impl Into<TensorOptions>) -> Result<TensorId, Oom> {
        let t = t.into();
        let buf = self.alloc.alloc(A::Arg::from(&t))?;
        Ok(self.new_tensor_impl(t.layout, Op::Const, t.name, buf.id, buf.offs, None, t.loc))
    }
}

impl<A: TensorAlloc, S: TensorSrc> TensorSrc for TensorBuilder<A, S> {
    #[inline(always)]
    fn get_tensor(&self, id: TensorId) -> Result<&Tensor, TensorNotFound> {
        self.get(id).ok_or(TensorNotFound(id))
    }
}

impl<A, S: TensorSrc> GetCtxId for TensorBuilder<A, S> {
    #[inline(always)]
    fn id(&self) -> CtxId {
        self.id
    }
}

impl<A: TensorAlloc, S: TensorSrc> Index<TensorId> for TensorBuilder<A, S> {
    type Output = Tensor;

    #[inline(always)]
    fn index(&self, id: TensorId) -> &Self::Output {
        self.get(id).unwrap()
    }
}

impl<A: TensorAlloc, S: TensorSrc> IndexMut<TensorId> for TensorBuilder<A, S> {
    #[inline(always)]
    fn index_mut(&mut self, id: TensorId) -> &mut Self::Output {
        self.get_mut(id).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::{buf::SliceCast, D1, D2, D3, D4};

    use super::*;

    pub const KB: usize = 1024 * 1024;

    #[test]
    fn fill_1d() {
        let mut b = TensorBuilder::prealloc(KB).unwrap();
        let t = b.new_tensor(D1::from(3).with_type(Type::I32)).unwrap();
        b.fill(t, 1);
        let t = &b[t];
        let data = b
            .alloc
            .get_slice(t.buf_id, t.data_offs, t.size())
            .unwrap()
            .cast::<i32>();
        assert_eq!(data, &[1_i32, 1, 1]);
    }

    #[test]
    fn fill_2d() {
        let mut b = TensorBuilder::prealloc(KB).unwrap();
        let t = b
            .new_tensor(D2 { rows: 4, cols: 8 }.with_type(Type::I32))
            .unwrap();
        b.fill(t, 9);
        let t = &b[t];
        let n = t.total_elems() as i32;
        assert_eq!(n, 4 * 8);
        let data = b
            .alloc
            .get_slice(t.buf_id, t.data_offs, t.size())
            .unwrap()
            .cast::<i32>();
        let expected = (0..n).map(|_| 9).collect::<Vec<_>>();
        assert_eq!(data, expected);
    }

    #[test]
    fn fill_with_2d() {
        let mut b = TensorBuilder::prealloc(KB).unwrap();
        let t = b
            .new_tensor(D2 { rows: 4, cols: 8 }.with_type(Type::I32))
            .unwrap();
        let mut iter = 0..;
        b.fill_with(t, || iter.next().unwrap());
        let t = &b[t];
        let n = t.total_elems() as i32;
        let data = b
            .alloc
            .get_slice(t.buf_id, t.data_offs, t.size())
            .unwrap()
            .cast::<i32>();
        let expected = (0..n).collect::<Vec<_>>();
        assert_eq!(data, expected);
    }

    #[test]
    fn fill_3d() {
        let mut b = TensorBuilder::prealloc(KB).unwrap();
        let t = b
            .new_tensor(
                D3 {
                    depth: 2,
                    rows: 4,
                    cols: 8,
                }
                .with_type(Type::I32),
            )
            .unwrap();
        b.fill(t, 9);
        let t = &b[t];
        let n = t.total_elems() as i32;
        assert_eq!(n, 2 * 4 * 8);
        let data = b
            .alloc
            .get_slice(t.buf_id, t.data_offs, t.size())
            .unwrap()
            .cast::<i32>();
        let expected = (0..n).map(|_| 9).collect::<Vec<_>>();
        assert_eq!(data, expected);
    }

    #[test]
    fn fill_4d() {
        let mut b = TensorBuilder::prealloc(KB).unwrap();
        let t = b
            .new_tensor(
                D4 {
                    batches: 3,
                    depth: 2,
                    rows: 4,
                    cols: 8,
                }
                .with_type(Type::I32),
            )
            .unwrap();
        b.fill(t, 9);
        let t = &b[t];
        let n = t.total_elems() as i32;
        assert_eq!(n, 3 * 2 * 4 * 8);
        let data = b
            .alloc
            .get_slice(t.buf_id, t.data_offs, t.size())
            .unwrap()
            .cast::<i32>();
        let expected = (0..n).map(|_| 9).collect::<Vec<_>>();
        assert_eq!(data, expected);
    }

    #[test]
    fn regression_tensor_option_with_size() {
        let n = 512;
        let typ = Type::F16;
        let value = D1::from(n).with_type(typ).with_name("cache_k");
        let expected = Layout::new_1d(n, Type::F16);
        assert_eq!(value.layout.size(), expected.size());
    }
}

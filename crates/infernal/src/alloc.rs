use std::io;

use thiserror::Error;

pub mod bump;
pub mod graph;
pub mod measure;
pub mod mmap;
mod split;

pub use bump::BumpAlloc;
pub use graph::{GraphAlloc, RawParts};
pub use measure::AllocMeasure;
pub use mmap::MmapAlloc;
pub use split::*;

use crate::{BufId, Tensor};

/// A trait for allocating (or pretending to allocate) tensor data.
///
/// To use the buffers, you must implement `IntoBackendBufs` for your allocator. Some allocators
/// don't implement it as they work on buffer references and thus don't need to explicitly give out
/// their buffer to the backend.
pub trait TensorAlloc {
    type Arg;

    fn alloc(&mut self, arg: Self::Arg) -> Result<AllocResult, Oom>;

    #[allow(unused_variables)]
    fn finalize_tensors(&mut self, tensors: &mut [Tensor]) -> io::Result<()> {
        Ok(())
    }
}

impl<T: TensorAlloc> TensorAlloc for &mut T {
    type Arg = T::Arg;

    #[inline(always)]
    fn alloc(&mut self, arg: Self::Arg) -> Result<AllocResult, Oom> {
        (**self).alloc(arg)
    }
}

pub trait ReadWriteAlloc: TensorAlloc {
    fn get_slice(&self, id: BufId, offs: usize, len: usize) -> Option<&[u8]>;
    fn get_mut_slice(&mut self, id: BufId, offs: usize, len: usize) -> Option<&mut [u8]>;
}

impl<T: ReadWriteAlloc> ReadWriteAlloc for &mut T {
    #[inline(always)]
    fn get_slice(&self, id: BufId, offs: usize, len: usize) -> Option<&[u8]> {
        (**self).get_slice(id, offs, len)
    }

    #[inline(always)]
    fn get_mut_slice(&mut self, id: BufId, offs: usize, len: usize) -> Option<&mut [u8]> {
        (**self).get_mut_slice(id, offs, len)
    }
}

#[derive(Debug)]
pub struct AllocResult {
    pub id: BufId,
    pub offs: usize,
}

#[derive(Clone, Copy, Debug, Error)]
#[error("OUT OF MEMORY: requested {requested} bytes from {from} out of {free} free bytes ({capacity} total)")]
pub struct Oom {
    pub from: &'static str,
    pub requested: usize,
    pub free: usize,
    pub capacity: usize,
}

pub trait AllocSize {
    fn alloc_size(&self) -> usize;
}
impl<T: AllocSize> AllocSize for &T {
    #[inline(always)]
    fn alloc_size(&self) -> usize {
        (**self).alloc_size()
    }
}

use crate::{
    alloc::{AllocResult, AllocSize, Oom, ReadWriteAlloc, TensorAlloc},
    buf::BufId,
};

/// A type that can be used in place of an allocator that only measures how much allocation would
/// occur but does not in itself allocate anything.
#[derive(Default)]
pub struct AllocMeasure {
    size: usize,
}

impl TensorAlloc for AllocMeasure {
    type Arg = usize;

    #[inline(always)]
    fn alloc(&mut self, size: usize) -> Result<AllocResult, Oom> {
        self.size += size;
        Ok(AllocResult {
            id: BufId::unallocated(),
            offs: 0,
        })
    }
}

impl AllocSize for AllocMeasure {
    #[inline(always)]
    fn alloc_size(&self) -> usize {
        self.size
    }
}

// We implement this because `AllocMeasure` is often used in place of `ReadWriteAlloc`s, which is
// the whole point: we want to be able to drop in replace the allocator to measure usage without
// making the API more complicated to use.
impl ReadWriteAlloc for AllocMeasure {
    #[inline(always)]
    fn get_slice(&self, _id: BufId, _offs: usize, _len: usize) -> Option<&[u8]> {
        None
    }

    #[inline(always)]
    fn get_mut_slice(&mut self, _id: BufId, _offs: usize, _len: usize) -> Option<&mut [u8]> {
        None
    }
}

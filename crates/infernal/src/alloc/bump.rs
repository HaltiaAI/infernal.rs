use std::{iter, ops::DerefMut};

use tracing::trace;

use crate::{
    alloc::{AllocResult, AllocSize, Oom, ReadWriteAlloc, TensorAlloc},
    backend::{self, IntoBackendBufs},
    buf::{Buf, BufId},
    utils::Bytes,
};

/// A fast and simple allocator that allocates a single buffer and returns slices of it.
///
/// This is the fastest in-process allocator that you can use. It's appropriate when the size of the
/// data is known ahead of time and, besides using an mmap allocator, should be used whenever
/// possible.
pub struct BumpAlloc<B: DerefMut<Target = [u8]> = Buf> {
    id: BufId,
    buf: B,
    free_offs: usize,
}

impl BumpAlloc {
    #[inline(always)]
    pub fn new(size: usize) -> Result<Self, Oom> {
        Ok(Self::with_buf(Buf::new(size)?))
    }
}

impl<B: DerefMut<Target = [u8]>> BumpAlloc<B> {
    #[inline(always)]
    pub fn with_buf(buf: B) -> Self {
        Self {
            id: BufId::from_addr(&buf),
            buf,
            free_offs: 0,
        }
    }
}

impl<B: DerefMut<Target = [u8]>> TensorAlloc for BumpAlloc<B> {
    type Arg = usize;

    #[inline(always)]
    fn alloc(&mut self, size: usize) -> Result<AllocResult, Oom> {
        trace!(
            size = %Bytes(size),
            total_so_far = %Bytes(self.alloc_size()),
            capacity = %Bytes(self.buf.len()),
            free = %Bytes(self.buf.len() - self.alloc_size()),
            "alloc"
        );
        if self.free_offs + size > self.buf.as_ref().len() {
            return Err(Oom {
                from: "bump allocator",
                requested: size,
                free: self.buf.as_ref().len() - self.free_offs,
                capacity: self.buf.as_ref().len(),
            });
        }

        let offs = self.free_offs;
        self.free_offs += size;

        Ok(AllocResult { id: self.id, offs })
    }
}

impl<B: DerefMut<Target = [u8]>> AllocSize for BumpAlloc<B> {
    #[inline(always)]
    fn alloc_size(&self) -> usize {
        self.free_offs
    }
}

impl<B: DerefMut<Target = [u8]>> ReadWriteAlloc for BumpAlloc<B> {
    #[inline(always)]
    fn get_slice(&self, id: BufId, offs: usize, len: usize) -> Option<&[u8]> {
        // TODO: error handling for slice out of bounds
        if id == self.id {
            Some(&self.buf[offs..offs + len])
        } else {
            None
        }
    }

    #[inline(always)]
    fn get_mut_slice(&mut self, id: BufId, offs: usize, len: usize) -> Option<&mut [u8]> {
        // TODO: error handling for slice out of bounds
        if id == self.id {
            Some(&mut self.buf[offs..offs + len])
        } else {
            None
        }
    }
}

impl<B: DerefMut<Target = [u8]>> IntoBackendBufs for BumpAlloc<B>
where
    backend::BackendData: From<B>,
{
    type Data = B;
    type Iter = iter::Once<(BufId, Self::Data)>;

    #[inline(always)]
    fn into_backend_bufs(self) -> Self::Iter {
        iter::once((self.id, self.buf))
    }
}

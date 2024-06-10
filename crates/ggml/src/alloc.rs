use std::ptr::NonNull;

use crate::{buffer::PageAlignedBuf, sys, ComputeGraph, InitError, ManagedStorageTag, Tensor};

#[derive(Debug)]
pub struct Allocator<S>(pub(crate) NonNull<sys::ggml_allocr>, S);

impl Allocator<PageAlignedBuf> {
    #[inline(always)]
    pub fn new(mut buf: PageAlignedBuf, align: usize) -> Result<Self, InitError> {
        let allocr = unsafe { sys::ggml_allocr_new(buf.as_mut_ptr().cast(), buf.len(), align) };
        Ok(Self(NonNull::new(allocr).ok_or(InitError)?, buf))
    }

    pub fn page_aligned_data(&self) -> &[u8] {
        &self.1
    }
}

impl Allocator<ManagedStorageTag> {
    #[inline(always)]
    pub fn new_measure(align: usize) -> Result<Self, InitError> {
        let allocr = unsafe { sys::ggml_allocr_new_measure(align) };
        Ok(Self(NonNull::new(allocr).ok_or(InitError)?, ManagedStorageTag))
    }
}

impl<S> Allocator<S> {
    pub fn is_measure(&self) -> bool {
        unsafe { sys::ggml_allocr_is_measure(self.0.as_ptr()) }
    }

    pub fn reset(&mut self) {
        unsafe {
            sys::ggml_allocr_reset(self.0.as_ptr());
        }
    }

    #[inline(always)]
    pub fn alloc_graph(&mut self, gf: &mut ComputeGraph) -> usize {
        unsafe { sys::ggml_allocr_alloc_graph(self.0.as_ptr(), gf.0.as_mut_ptr()) }
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    #[inline(always)]
    pub fn alloc(&mut self, t: &mut Tensor) {
        unsafe {
            sys::ggml_allocr_alloc(self.0.as_mut(), t.p());
        }
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    #[inline(always)]
    pub fn set_parse_seq(&mut self, list: &[i32]) {
        unsafe { sys::ggml_allocr_set_parse_seq(self.0.as_mut(), list.as_ptr(), list.len() as i32) }
    }
}

impl<S> Drop for Allocator<S> {
    fn drop(&mut self) {
        unsafe {
            sys::ggml_allocr_free(self.0.as_mut());
        }
    }
}

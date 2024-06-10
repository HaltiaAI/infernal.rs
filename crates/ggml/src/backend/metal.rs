use core::slice;
use std::{ffi::CString, mem, ptr::NonNull};

use crate::{sys, Allocator, Backend, BufferError, ComputeGraph, InitError, Tensor};

// TODO: feature gate
pub struct Metal(pub(crate) NonNull<sys::ggml_metal_context>);

impl std::fmt::Debug for Metal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Metal").finish()
    }
}

impl Backend for Metal {
    #[inline(always)]
    fn new() -> Result<Self, InitError>
    where
        Self: Sized,
    {
        unsafe {
            let ctx = sys::ggml_metal_init(1);
            Ok(Self(NonNull::new(ctx).ok_or(InitError)?))
        }
    }

    #[inline(always)]
    fn optimize_graph(&mut self, cg: &mut ComputeGraph) {
        self.find_concurrency(cg);
    }

    #[inline(always)]
    fn optimize_allocator<S>(&mut self, alloc: &mut Allocator<S>) {
        alloc.set_parse_seq(self.get_concur_list());
    }

    #[inline(always)]
    fn add_buffer(
        &mut self,
        buf: &[u8],
        name: impl AsRef<[u8]>,
        max_size: Option<usize>,
    ) -> Result<(), BufferError> {
        let name = name.as_ref();
        let name = CString::new(name).expect("buffer name");
        unsafe {
            // SAFETY:
            //
            // According to the ggml_metal_add_buffer docs:
            // > you don't need to keep the host memory buffer allocated as it is never
            // > accessed by Metal
            //
            // This is not actually true because the host data buffer is used in the `ggml_metal_get_buffer` call to match the sought for tensor.
            // However, while the function signature specifies a mutable pointer in classic
            // C fashion, it's never actually mutated (which makes sense, because in the LLama code
            // for example, the read-only mmap buffer is added the same way -- so it cannot be
            // modified), so we can safetly transmute it to a fake mutable pointer.
            //
            // This is obviously not amazing but it is what it is until we get rid of the
            // C code.
            if sys::ggml_metal_add_buffer(
                self.0.as_mut(),
                name.as_ptr(),
                mem::transmute(buf.as_ptr()),
                buf.len(),
                max_size.unwrap_or_default(),
            ) {
                Ok(())
            } else {
                Err(BufferError)
            }
        }
    }

    #[inline(always)]
    fn compute_graph(&mut self, cg: &mut ComputeGraph, n_threads: usize) {
        unsafe {
            sys::ggml_metal_set_n_cb(self.0.as_mut(), n_threads as i32);
            sys::ggml_metal_graph_compute(self.0.as_mut(), cg.0.as_mut_ptr());
        }
    }

    #[inline(always)]
    fn get_result_tensor(&mut self, cg: &mut ComputeGraph) -> Tensor {
        let t = super::get_result_tensor(cg);
        unsafe {
            sys::ggml_metal_get_tensor(self.0.as_mut(), t.p());
        }
        t
    }

    #[inline(always)]
    fn get_embd_tensor(&mut self, cg: &mut ComputeGraph) -> Tensor {
        let t = super::get_embd_tensor(cg);
        unsafe {
            sys::ggml_metal_get_tensor(self.0.as_mut(), t.p());
        }
        t
    }
}

impl Metal {
    #[inline(always)]
    pub fn find_concurrency(&mut self, cg: &mut ComputeGraph) {
        unsafe {
            sys::ggml_metal_graph_find_concurrency(self.p(), cg.0.as_mut_ptr(), false);
        }
    }

    // TODO: check where this was needed
    // pub fn concur_list(&self) -> &[i32] {
    //     unsafe {
    //         let ptr =
    //             sys::ggml_metal_get_concur_list(mem::transmute::<_, *mut _>(self.0.as_ptr()));
    //         let len = GGML_MAX_CONCUR as usize;
    //     }
    // }

    fn get_concur_list(&self) -> &[i32] {
        unsafe {
            let list = sys::ggml_metal_get_concur_list(self.p());
            let n = sys::ggml_metal_if_optimized(self.p());
            slice::from_raw_parts_mut(list, n as usize)
        }
    }

    pub(crate) fn p(&self) -> *mut sys::ggml_metal_context {
        self.0.as_ptr()
    }
}

impl Drop for Metal {
    fn drop(&mut self) {
        unsafe { sys::ggml_metal_free(self.0.as_ptr()) }
    }
}

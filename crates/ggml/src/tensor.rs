use core::{fmt, slice};
use std::{mem, ptr::NonNull};

use crate::{pad, sys, Allocator, Backend, ComputeGraph, Dims, TensorLocation, TensorType};

// Cloning the tensor is fine, despite its storing a pointer because the pointer is not released
// when a tensor instance is dropped. This is because the tensor's memory is managed by Context and
// arbitrary number of copies can point into its internal memory pool, which will be cleaned up when
// the context goes out of scope.
//
// FIXME: bound this lifetime by the lifetime of the context that produced the tensor for
// correctness. until then: yolo
#[derive(Debug, Clone)]
pub struct Tensor(pub(crate) NonNull<sys::ggml_tensor>);

impl Tensor {
    #[inline(always)]
    pub(crate) fn from_raw(ptr: *mut sys::ggml_tensor) -> Self {
        // TODO: error handling here instead
        Self(NonNull::new(ptr).expect("failed to allocate tensor"))
    }

    #[inline(always)]
    pub fn total_elems(&self) -> usize {
        unsafe { sys::ggml_nelements(self.0.as_ptr()) as usize }
    }

    #[inline(always)]
    pub fn nbytes(&self) -> usize {
        unsafe { sys::ggml_nbytes(self.0.as_ptr()) }
    }

    #[inline(always)]
    pub fn nbytes_padded(&self) -> usize {
        pad(self.nbytes() as u64, sys::GGML_MEM_ALIGN as usize) as usize
    }

    #[inline(always)]
    pub fn elem_size(&self) -> u32 {
        unsafe { sys::ggml_element_size(self.0.as_ptr()) as u32 }
    }

    #[inline(always)]
    pub fn is_transposed(&self) -> bool {
        unsafe { sys::ggml_is_transposed(self.0.as_ptr()) }
    }

    #[inline(always)]
    pub fn data_mut(&mut self) -> &mut [u8] {
        unsafe {
            let data = self.0.as_mut().data;
            slice::from_raw_parts_mut(data as _, self.nbytes())
        }
    }

    #[inline(always)]
    pub fn set_data(&mut self, data: *const u8) {
        unsafe {
            self.0.as_mut().data = data as _;
        }
    }

    #[inline(always)]
    pub fn location(&self) -> TensorLocation {
        unsafe {
            self.0
                .as_ref()
                .backend
                .try_into()
                .expect("invalid backend type")
        }
    }

    #[inline(always)]
    pub fn set_zero(&mut self) {
        unsafe {
            sys::ggml_set_zero(self.0.as_mut());
        }
    }

    #[inline(always)]
    pub fn set_f32(&mut self, v: f32) {
        unsafe {
            sys::ggml_set_f32(self.0.as_mut(), v);
        }
    }

    #[inline(always)]
    pub fn set_i32_1d_at(&mut self, col: usize, v: i32) {
        unsafe {
            sys::ggml_set_i32_1d(self.0.as_mut(), col as i32, v);
        }
    }

    #[inline(always)]
    pub fn set_f32_1d_at(&mut self, col: usize, v: f32) {
        unsafe {
            sys::ggml_set_f32_1d(self.0.as_mut(), col as i32, v);
        }
    }

    #[inline(always)]
    pub fn set_name(&mut self, name: impl AsRef<[u8]>) {
        let name = name.as_ref();
        unsafe {
            let this = self.0.as_mut();
            let name_len = name.len().min(this.name.len() - 1);
            this.name[..name_len].copy_from_slice(mem::transmute(&name[..name_len]));
            this.name[name_len] = 0; // Zero-terminate c-string.
        }
    }

    #[inline(always)]
    pub fn set_location(&mut self, loc: TensorLocation) {
        unsafe {
            let this = self.0.as_mut();
            this.backend = loc as sys::ggml_backend;
        }
    }

    #[inline(always)]
    pub fn quantize_q4_0(&mut self, src: &Self, n: u32, k: u32, hist: &mut [i64]) {
        unsafe {
            sys::ggml_quantize_q4_0(
                src.0.as_ref().data as _,
                self.0.as_mut().data,
                n as _,
                k as _,
                hist.as_mut_ptr(),
            );
        }
    }

    #[inline(always)]
    pub fn build_forward_cgraph(&mut self) -> ComputeGraph {
        unsafe {
            let cg = sys::ggml_build_forward(self.0.as_mut());
            ComputeGraph(cg.into())
        }
    }

    #[inline(always)]
    pub fn name(&self) -> &str {
        unsafe {
            let name = &self.0.as_ref().name;
            let len = name.iter().position(|b| *b == 0).unwrap_or(name.len());
            let s = slice::from_raw_parts(name.as_ptr(), len);
            std::str::from_utf8(mem::transmute(s)).expect("tensor name")
        }
    }

    #[inline(always)]
    pub fn typ(&self) -> TensorType {
        unsafe {
            // SAFETY: This can't be wrong because the type is set when the tensor is created.
            self.0.as_ref().type_.try_into().unwrap()
        }
    }

    #[inline(always)]
    pub fn dim0(&self) -> i64 {
        unsafe { self.0.as_ref().ne[0] }
    }

    #[inline(always)]
    pub fn dim1(&self) -> i64 {
        unsafe { self.0.as_ref().ne[1] }
    }

    #[inline(always)]
    pub fn dims(&self) -> Dims {
        unsafe {
            // TODO: verify safety of transmute
            Dims::from(mem::transmute::<_, &[u64]>(
                &self.0.as_ref().ne[..self.0.as_ref().n_dims as usize],
            ))
        }
    }

    #[inline(always)]
    pub fn stride(&self) -> &[usize; 4] {
        unsafe { &self.0.as_ref().nb }
    }

    #[inline(always)]
    pub fn has_dims(&self, expected: &[u32]) -> bool {
        unsafe {
            let this = self.0.as_ref();
            if this.n_dims as usize != expected.len() {
                return false;
            }
            #[allow(clippy::needless_range_loop)] // stfu
            for i in 0..expected.len() {
                if this.ne[i] as u64 != expected[i] as u64 {
                    return false;
                }
            }
        }
        true
    }

    #[inline(always)]
    pub fn alloc<S>(mut self, alloc: &mut Allocator<S>) -> Self {
        alloc.alloc(&mut self);
        self
    }

    #[inline(always)]
    pub fn with_name(mut self, name: impl AsRef<[u8]>) -> Self {
        self.set_name(name);
        self
    }

    // TODO: document safety or maybe make entire fn unsafe
    #[inline(always)]
    pub fn copy_data_to<T>(&self, offset: usize, dest: &mut [T]) {
        unsafe {
            dest.as_mut_ptr()
                .copy_from(self.0.as_ref().data.cast::<T>().add(offset), dest.len());
        }
    }

    // TODO: document safety
    #[inline(always)]
    pub fn copy_data_from<T>(&mut self, src: &[T]) {
        unsafe {
            self.0
                .as_mut()
                .data
                .cast::<T>()
                .copy_from(src.as_ptr() as _, src.len());
        }
    }

    pub(crate) fn p(&self) -> *mut sys::ggml_tensor {
        self.0.as_ptr()
    }

    #[inline(always)]
    pub fn offload_to(mut self, b: &mut impl Backend) -> Self {
        b.offload_tensor(&mut self);
        self
    }

    /// Convenience method that only offloads to the accelerator if `offload` is true. This makes
    /// callsites a lot more DRY and thus less error prone.
    #[inline(always)]
    pub fn maybe_offload_to(self, offload: bool, b: &mut impl Backend) -> Self {
        if offload {
            self.offload_to(b)
        } else {
            self
        }
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}[{}x{}]", self.typ(), self.dim0(), self.dim1())
    }
}

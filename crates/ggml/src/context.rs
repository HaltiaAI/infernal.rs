use std::{mem, ptr::NonNull};

use memmap2::Mmap;

use crate::{
    buffer::PageAlignedBuf, gguf::TensorInfo, sys, ComputeGraph, InitError, InternalAlloc,
    ManagedStorageTag, Tensor, TensorType,
};

pub type MmapContext = Context<Mmap>;
pub type PageAlignedContext = Context<PageAlignedBuf>;
pub type ManagedContext = Context<ManagedStorageTag>;

#[derive(Debug)]
pub struct Context<S>(pub(crate) NonNull<sys::ggml_context>, S);

impl Context<ManagedStorageTag> {
    /// Creates a new context using the managed memory pool for the tensor data.
    #[inline(always)]
    pub fn new_managed(mem_size: usize) -> Result<Self, InitError> {
        let init_params = sys::ggml_init_params {
            mem_size,
            mem_buffer: std::ptr::null_mut(),
            no_alloc: false,
        };
        let ctx = unsafe { sys::ggml_init(init_params) };
        Ok(Self(NonNull::new(ctx).ok_or(InitError)?, ManagedStorageTag))
    }
}

impl<'a> Context<&'a mut PageAlignedBuf> {
    #[inline(always)]
    pub fn with_page_aligend_ref(
        buf: &'a mut PageAlignedBuf,
        no_alloc: bool,
    ) -> Result<Self, InitError> {
        let init_params = sys::ggml_init_params {
            mem_size: buf.len(),
            mem_buffer: buf.as_mut_ptr().cast(),
            no_alloc,
        };
        let ctx = unsafe { sys::ggml_init(init_params) };
        Ok(Self(NonNull::new(ctx).ok_or(InitError)?, buf))
    }

    #[inline(always)]
    pub fn page_aligned_data(&self) -> &[u8] {
        self.1
    }
}

impl Context<PageAlignedBuf> {
    #[inline(always)]
    pub fn with_page_aligend(mut buf: PageAlignedBuf, no_alloc: bool) -> Result<Self, InitError> {
        let init_params = sys::ggml_init_params {
            mem_size: buf.len(),
            mem_buffer: buf.as_mut_ptr().cast(),
            no_alloc,
        };
        let ctx = unsafe { sys::ggml_init(init_params) };
        Ok(Self(NonNull::new(ctx).ok_or(InitError)?, buf))
    }

    #[inline(always)]
    pub fn page_aligned_data(&self) -> &[u8] {
        &self.1
    }
}

impl Context<Mmap> {
    /// Creates a new context using the provided mapping for the tensor data.
    ///
    /// Note that in this case there are two memory types used by ggml:
    /// 1) The memory indicated by `ctx_size` is the internal memory used by ggml to allocate the
    ///    tensors without their data.
    /// 2) The mmap, which is usually a model's weightg mapped into memory and accessed directly by
    ///    ggml when performing matrix computations.
    #[inline(always)]
    pub fn with_mmap(ctx_size: usize, mmap: Mmap) -> Result<Self, InitError> {
        let init_params = sys::ggml_init_params {
            mem_size: ctx_size,
            mem_buffer: std::ptr::null_mut(),
            no_alloc: true,
        };
        let ctx = unsafe { sys::ggml_init(init_params) };
        Ok(Self(NonNull::new(ctx).ok_or(InitError)?, mmap))
    }

    /// Creates a tensor allocated into the buffer provided in the constructor.
    #[inline(always)]
    pub fn create_tensor(&self, data_offset: u64, info: &TensorInfo) -> Tensor {
        let ptr = unsafe {
            sys::ggml_new_tensor(
                self.p(),
                info.typ as u32,
                info.dims.n() as i32,
                mem::transmute(info.dims.as_ptr()),
            )
        };
        let mut t = Tensor::from_raw(ptr).with_name(&info.name);
        let data = unsafe {
            self.1
                .as_ptr()
                .offset((data_offset + info.offset as u64) as isize)
        };
        t.set_data(data);
        t
    }

    #[inline(always)]
    pub fn mmap_data(&self) -> &[u8] {
        &self.1
    }
}

impl<'a> Context<&'a Mmap> {
    /// Creates a new context using the provided mapping for the tensor data.
    ///
    /// Note that in this case there are two memory types used by ggml:
    /// 1) The memory indicated by `ctx_size` is the internal memory used by ggml to allocate the
    ///    tensors without their data.
    /// 2) The mmap, which is usually a model's weightg mapped into memory and accessed directly by
    ///    ggml when performing matrix computations.
    #[inline(always)]
    pub fn with_mmap_ref(ctx_size: usize, mmap: &'a Mmap) -> Result<Self, InitError> {
        let init_params = sys::ggml_init_params {
            mem_size: ctx_size,
            mem_buffer: std::ptr::null_mut(),
            no_alloc: true,
        };
        let ctx = unsafe { sys::ggml_init(init_params) };
        Ok(Self(NonNull::new(ctx).ok_or(InitError)?, mmap))
    }

    /// Creates a tensor allocated into the buffer provided in the constructor.
    #[inline(always)]
    pub fn create_tensor(&self, data_offset: u64, info: &TensorInfo) -> Tensor {
        let ptr = unsafe {
            sys::ggml_new_tensor(
                self.p(),
                info.typ as u32,
                info.dims.n() as i32,
                mem::transmute(info.dims.as_ptr()),
            )
        };
        let mut t = Tensor::from_raw(ptr).with_name(&info.name);
        let data = unsafe {
            self.1
                .as_ptr()
                .offset((data_offset + info.offset as u64) as isize)
        };
        t.set_data(data);
        t
    }

    #[inline(always)]
    pub fn mmap_data(&self) -> &[u8] {
        self.1
    }
}

impl<S: InternalAlloc> Context<S> {
    pub fn create_1d_tensor(&self, typ: TensorType, cols: u32) -> Tensor {
        let ptr = unsafe { sys::ggml_new_tensor_1d(self.p(), typ as sys::ggml_type, cols as i64) };
        Tensor::from_raw(ptr)
    }

    #[inline(always)]
    pub fn create_2d_tensor(&self, typ: TensorType, cols: u32, rows: u32) -> Tensor {
        let ptr = unsafe {
            sys::ggml_new_tensor_2d(self.p(), typ as sys::ggml_type, cols as i64, rows as i64)
        };
        Tensor::from_raw(ptr)
    }

    #[inline(always)]
    pub fn create_3d_tensor(&self, typ: TensorType, cols: u32, rows: u32, depth: u32) -> Tensor {
        let ptr = unsafe {
            sys::ggml_new_tensor_3d(
                self.p(),
                typ as sys::ggml_type,
                cols as i64,
                rows as i64,
                depth as i64,
            )
        };
        Tensor::from_raw(ptr)
    }

    #[inline(always)]
    pub fn create_4d_tensor(
        &self,
        typ: TensorType,
        cols: u32,
        rows: u32,
        depth: u32,
        batches: u32,
    ) -> Tensor {
        let ptr = unsafe {
            sys::ggml_new_tensor_4d(
                self.p(),
                typ as sys::ggml_type,
                cols as i64,
                rows as i64,
                depth as i64,
                batches as i64,
            )
        };
        Tensor::from_raw(ptr)
    }

    #[inline(always)]
    pub fn internal_data(&self) -> &[u8] {
        unsafe {
            let ptr = mem::transmute(sys::ggml_get_mem_buffer(self.p()));
            let len = sys::ggml_get_mem_size(self.p());
            std::slice::from_raw_parts(ptr, len)
        }
    }
}

impl<S> Context<S> {
    #[inline(always)]
    pub fn set_no_alloc(&self, no_alloc: bool) {
        unsafe { sys::ggml_set_no_alloc(self.p(), no_alloc) }
    }

    #[inline(always)]
    pub fn max_tensor_size(&self) -> usize {
        unsafe { sys::ggml_get_max_tensor_size(self.p()) }
    }

    #[inline(always)]
    pub fn mul(&mut self, a: &Tensor, b: &Tensor) -> Tensor {
        unsafe {
            let r = sys::ggml_mul(self.0.as_mut(), a.p(), b.p());
            Tensor::from_raw(r)
        }
    }

    #[inline(always)]
    pub fn mul_mat(&mut self, a: &Tensor, b: &Tensor) -> Tensor {
        unsafe {
            let r = sys::ggml_mul_mat(self.0.as_mut(), a.p(), b.p());
            Tensor::from_raw(r)
        }
    }

    #[inline(always)]
    pub fn add(&mut self, a: &Tensor, b: &Tensor) -> Tensor {
        unsafe {
            let r = sys::ggml_add(self.0.as_mut(), a.p(), b.p());
            Tensor::from_raw(r)
        }
    }

    #[inline(always)]
    pub fn silu(&mut self, a: &Tensor) -> Tensor {
        unsafe {
            let t = sys::ggml_silu(self.p(), a.p());
            Tensor::from_raw(t)
        }
    }

    #[inline(always)]
    pub fn relu(&mut self, a: &Tensor) -> Tensor {
        unsafe {
            let t = sys::ggml_relu(self.p(), a.p());
            Tensor::from_raw(t)
        }
    }

    // TODO: this is not implemented in the current checked out commit of ggml
    // #[inline(always)]
    // pub fn im2col(
    //     &mut self,
    //     a: &Tensor,
    //     b: &Tensor,
    //     stride: i32,
    //     padding: i32,
    //     dilation: i32,
    // ) -> Tensor {
    //     unsafe {
    //         let t = sys::ggml_im2col(self.p(), a.p(), b.p(), stride, padding, dilation);
    //         Tensor::from_raw(t)
    //     }
    // }

    #[inline(always)]
    pub fn conv_1d(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        stride: i32,
        padding: i32,
        dilation: i32,
    ) -> Tensor {
        unsafe {
            let t = sys::ggml_conv_1d(self.p(), a.p(), b.p(), stride, padding, dilation);
            Tensor::from_raw(t)
        }
    }

    #[inline(always)]
    pub fn conv_1d_ph(&mut self, a: &Tensor, b: &Tensor, stride: i32, dilation: i32) -> Tensor {
        unsafe {
            let t = sys::ggml_conv_1d_ph(self.p(), a.p(), b.p(), stride, dilation);
            Tensor::from_raw(t)
        }
    }

    #[inline(always)]
    pub fn rms_norm(&mut self, a: &Tensor, eps: f32) -> Tensor {
        unsafe {
            let t = sys::ggml_rms_norm(self.p(), a.p(), eps);
            Tensor::from_raw(t)
        }
    }

    // Returns view(a).
    #[inline(always)]
    pub fn scale_inplace(&mut self, a: &Tensor, scalar: &Tensor) -> Tensor {
        unsafe {
            let t = sys::ggml_scale_inplace(self.p(), a.p(), scalar.0.as_ptr());
            Tensor::from_raw(t)
        }
    }

    /// Sets elements above the diagonal to -INF.
    #[inline(always)]
    pub fn diag_mask_inf_inplace(&mut self, a: &Tensor, n_past: u32) -> Tensor {
        unsafe {
            let t = sys::ggml_diag_mask_inf_inplace(self.p(), a.p(), n_past as i32);
            Tensor::from_raw(t)
        }
    }

    // Returns view(a).
    #[inline(always)]
    pub fn softmax_inplace(&mut self, a: &Tensor) -> Tensor {
        unsafe {
            let t = sys::ggml_soft_max_inplace(self.p(), a.p());
            Tensor::from_raw(t)
        }
    }

    #[inline(always)]
    pub fn view_1d(&self, a: &Tensor, dim0: u32, offset: u32) -> Tensor {
        unsafe {
            let t = sys::ggml_view_1d(self.p(), a.p(), dim0 as i64, offset as usize);
            Tensor::from_raw(t)
        }
    }

    #[inline(always)]
    pub fn view_2d(
        &self,
        a: &Tensor,
        dim0: u32,
        dim1: u32,
        row_stride: u32,
        offset: u32,
    ) -> Tensor {
        unsafe {
            let t = sys::ggml_view_2d(
                self.p(),
                a.p(),
                dim0 as i64,
                dim1 as i64,
                row_stride as usize,
                offset as usize,
            );
            Tensor::from_raw(t)
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn view_3d(
        &self,
        a: &Tensor,
        dim0: u32,
        dim1: u32,
        dim2: u32,
        row_stride: u32,
        slice_stride: u32,
        offset: u32,
    ) -> Tensor {
        unsafe {
            let t = sys::ggml_view_3d(
                self.p(),
                a.p(),
                dim0 as i64,
                dim1 as i64,
                dim2 as i64,
                row_stride as usize,
                slice_stride as usize,
                offset as usize,
            );
            Tensor::from_raw(t)
        }
    }

    #[inline(always)]
    pub fn reshape_2d(&self, a: &Tensor, dim0: u32, dim1: u32) -> Tensor {
        unsafe {
            let t = sys::ggml_reshape_2d(self.p(), a.p(), dim0 as i64, dim1 as i64);
            Tensor::from_raw(t)
        }
    }

    #[inline(always)]
    pub fn reshape_3d(&self, a: &Tensor, dim0: u32, dim1: u32, dim2: u32) -> Tensor {
        unsafe {
            let t = sys::ggml_reshape_3d(self.p(), a.p(), dim0 as i64, dim1 as i64, dim2 as i64);
            Tensor::from_raw(t)
        }
    }

    #[inline(always)]
    pub fn transpose(&self, a: &Tensor) -> Tensor {
        unsafe {
            let t = sys::ggml_transpose(self.p(), a.p());
            Tensor::from_raw(t)
        }
    }

    #[inline(always)]
    pub fn permute(&self, a: &Tensor, axis0: i32, axis1: i32, axis2: i32, axis3: i32) -> Tensor {
        unsafe {
            let t = sys::ggml_permute(self.p(), a.p(), axis0, axis1, axis2, axis3);
            Tensor::from_raw(t)
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn rope_custom_inplace(
        &self,
        a: &Tensor,
        n_past: u32,
        n_dims: u32,
        mode: i32,
        n_ctx: u32,
        freq_base: f32,
        freq_scale: f32,
    ) -> Tensor {
        unsafe {
            let t = sys::ggml_rope_custom_inplace(
                self.p(),
                a.p(),
                n_past as i32,
                n_dims as i32,
                mode,
                n_ctx as i32,
                freq_base,
                freq_scale,
            );
            Tensor::from_raw(t)
        }
    }

    #[inline(always)]
    pub fn copy(&self, src: &Tensor, dest: Tensor) -> Tensor {
        unsafe {
            let t = sys::ggml_cpy(self.p(), src.0.as_ptr(), dest.0.as_ptr());
            Tensor::from_raw(t)
        }
    }

    #[inline(always)]
    pub fn get_rows(&self, a: &Tensor, idx: &Tensor) -> Tensor {
        unsafe {
            let r = sys::ggml_get_rows(self.p(), a.p(), idx.p());
            Tensor::from_raw(r)
        }
    }

    #[inline(always)]
    pub fn create_graph(&mut self) -> ComputeGraph {
        unsafe {
            let cg = sys::ggml_new_graph(self.p());
            ComputeGraph(
                NonNull::new(cg)
                    .expect("ggml_new_graph failed to allocate object")
                    .into(),
            )
        }
    }

    pub(crate) fn p(&self) -> *mut sys::ggml_context {
        self.0.as_ptr()
    }
}

impl<S> Drop for Context<S> {
    fn drop(&mut self) {
        unsafe { sys::ggml_free(self.p()) }
    }
}

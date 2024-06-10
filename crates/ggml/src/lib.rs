use std::ptr::NonNull;

use thiserror::Error;

use crate::buffer::PageAlignedBuf;

pub use crate::{
    alloc::Allocator,
    backend::*,
    cgraph::ComputeGraph,
    context::{Context, ManagedContext, MmapContext, PageAlignedContext},
    tensor::Tensor,
};
pub use gguf::{self, Dims, TensorType};
pub use llama_cpp_sys::ggml::{self as sys, GGML_MAX_DIMS};

pub mod alloc;
pub mod backend;
pub mod buffer;
pub mod cgraph;
pub mod context;
pub mod tensor;

/// Specifies where the computation should take place.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(C)]
pub enum TensorLocation {
    /// The tensors are stored on the host and the compute graph is executed on the CPU or,
    /// in case of Apple Silicon, the GPU (since it uses unified memory between CPU and GPU).
    Ram = 0,
    /// Tensors are stored and executed on a discrete GPU.
    Vram = 10,
    /// The compute graph is split between CPU and discrete GPU. Value not applicable on non-Apple
    /// Silicon platforms.
    Split = 20,
}

impl TryFrom<sys::ggml_backend> for TensorLocation {
    type Error = ();
    fn try_from(value: sys::ggml_backend) -> Result<Self, Self::Error> {
        Ok(match value {
            sys::ggml_backend_GGML_BACKEND_CPU => Self::Ram,
            sys::ggml_backend_GGML_BACKEND_GPU => Self::Vram,
            sys::ggml_backend_GGML_BACKEND_GPU_SPLIT => Self::Split,
            _ => return Err(()),
        })
    }
}

#[derive(Debug, Error)]
#[error("failed to allocate object")]
pub struct InitError;

pub(crate) enum OwnedOrPtr<T> {
    Owned(T),
    Ptr(NonNull<T>),
}

impl<T> From<T> for OwnedOrPtr<T> {
    fn from(value: T) -> Self {
        Self::Owned(value)
    }
}

impl<T> From<NonNull<T>> for OwnedOrPtr<T> {
    fn from(value: NonNull<T>) -> Self {
        Self::Ptr(value)
    }
}

impl<T> OwnedOrPtr<T> {
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        match self {
            Self::Owned(t) => t as *mut _,
            Self::Ptr(t) => t.as_ptr(),
        }
    }
}

pub fn pad(x: u64, alignment: usize) -> u64 {
    let a = alignment as u64;
    (x + a - 1) & !(a - 1)
}

pub trait InternalAlloc {}
pub struct ManagedStorageTag;
impl InternalAlloc for PageAlignedBuf {}
impl<'a> InternalAlloc for &'a mut PageAlignedBuf {}
impl InternalAlloc for ManagedStorageTag {}

#[cfg(test)]
mod tests {
    use core::{fmt, slice};
    use std::{
        borrow::Cow,
        ffi::{c_char, CStr, CString},
        fs::File,
        mem,
        os::unix::ffi::OsStrExt,
        path::PathBuf,
    };

    use gguf::{kv::Value, Context};
    use libc::c_void;
    use llama_cpp_sys::ggml as sys;

    #[test]
    fn match_ggml() {
        let (our_ctx, _file, path) = get_context();

        unsafe {
            let path_cstr = CString::new(path.as_os_str().as_bytes()).unwrap();
            let init_params = sys::gguf_init_params {
                no_alloc: true,
                ctx: std::ptr::null_mut(),
            };
            let sys_ctx = sys::gguf_init_from_file(path_cstr.as_ptr(), init_params);

            let n_kv = sys::gguf_get_n_kv(sys_ctx);
            let n_tensors = sys::gguf_get_n_tensors(sys_ctx);
            let alignment = sys::gguf_get_alignment(sys_ctx);

            assert!(n_kv as u64 == our_ctx.header.n_kv);
            assert!(n_tensors as u64 == our_ctx.header.n_tensors);
            assert!(alignment == our_ctx.alignment);

            // Compare kv.
            for (i, kv) in our_ctx.kv.iter().enumerate() {
                let i = i as i32;
                assert!(i < n_kv);
                let key = cstr_to_string(sys::gguf_get_key(sys_ctx, i));
                assert_eq!(key.as_bytes(), kv.key);
                assert_kv_val(sys_ctx, i, &kv.value, None);
            }

            // Compare tensor infos.
            //
            // NB: We don't compare the tensor data here because in our usecase the tensor data is
            // never loaded at this stage, but later mmaped into program address space.
            //
            // NB: ggml doesn't expose getters for the following tensor fields: n_dims, ne, typ.
            // Unfortuantely for now this means the test is incomplete. We may consider forking
            // ggml, but that adds to maintenance overhead. For now I say not to do it as long
            // as higher level modules using the gguf loader here work fine.
            for (i, t) in our_ctx.tensors.iter().enumerate() {
                let i = i as i32;
                assert!(i < n_tensors);
                let offset = sys::gguf_get_tensor_offset(sys_ctx, i);
                assert_eq!(offset as u64, t.offset as _);
                let name = cstr_to_string(sys::gguf_get_tensor_name(sys_ctx, i));
                assert_eq!(name.as_bytes(), t.name);
            }
        }
    }

    unsafe fn assert_kv_val(
        sys_ctx: *mut sys::gguf_context,
        i: i32,
        value: &Value,
        arr_elem: Option<(sys::gguf_type, i32, *const c_void)>,
    ) {
        assert_eq!(mem::size_of::<GgufStr>(), 16);
        match value {
            Value::Uint8(v) => {
                if let Some((arr_type, offs, data)) = arr_elem {
                    assert_eq!(arr_type, sys::gguf_type_GGUF_TYPE_UINT8);
                    assert_arr_elem(data, offs, v);
                } else {
                    let sys_val = sys::gguf_get_val_u8(sys_ctx, i);
                    assert_eq!(sys_val, *v);
                }
            }
            Value::Int8(v) => {
                if let Some((arr_type, offs, data)) = arr_elem {
                    assert_eq!(arr_type, sys::gguf_type_GGUF_TYPE_INT8);
                    assert_arr_elem(data, offs, v);
                } else {
                    let sys_val = sys::gguf_get_val_i8(sys_ctx, i);
                    assert_eq!(sys_val, *v);
                }
            }
            Value::Uint16(v) => {
                if let Some((arr_type, offs, data)) = arr_elem {
                    assert_eq!(arr_type, sys::gguf_type_GGUF_TYPE_UINT16);
                    assert_arr_elem(data, offs, v);
                } else {
                    let sys_val = sys::gguf_get_val_u16(sys_ctx, i);
                    assert_eq!(sys_val, *v);
                }
            }
            Value::Int16(v) => {
                if let Some((arr_type, offs, data)) = arr_elem {
                    assert_eq!(arr_type, sys::gguf_type_GGUF_TYPE_INT16);
                    assert_arr_elem(data, offs, v);
                } else {
                    let sys_val = sys::gguf_get_val_i16(sys_ctx, i);
                    assert_eq!(sys_val, *v);
                }
            }
            Value::Uint32(v) => {
                if let Some((arr_type, offs, data)) = arr_elem {
                    assert_eq!(arr_type, sys::gguf_type_GGUF_TYPE_UINT32);
                    assert_arr_elem(data, offs, v);
                } else {
                    let sys_val = sys::gguf_get_val_u32(sys_ctx, i);
                    assert_eq!(sys_val, *v);
                }
            }
            Value::Int32(v) => {
                if let Some((arr_type, offs, data)) = arr_elem {
                    assert_eq!(arr_type, sys::gguf_type_GGUF_TYPE_INT32);
                    assert_arr_elem(data, offs, v);
                } else {
                    let sys_val = sys::gguf_get_val_i32(sys_ctx, i);
                    assert_eq!(sys_val, *v);
                }
            }
            Value::Float32(v) => {
                if let Some((arr_type, offs, data)) = arr_elem {
                    assert_eq!(arr_type, sys::gguf_type_GGUF_TYPE_FLOAT32);
                    assert_arr_elem(data, offs, v);
                } else {
                    let sys_val = sys::gguf_get_val_f32(sys_ctx, i);
                    assert_eq!(sys_val, *v);
                }
            }
            Value::Bool(v) => {
                if let Some((arr_type, offs, data)) = arr_elem {
                    assert_eq!(arr_type, sys::gguf_type_GGUF_TYPE_BOOL);
                    assert_arr_elem(data, offs, v);
                } else {
                    let sys_val = sys::gguf_get_val_bool(sys_ctx, i);
                    assert_eq!(sys_val, *v);
                }
            }
            Value::String(v) => {
                if let Some((arr_type, offs, data)) = arr_elem {
                    assert_eq!(arr_type, sys::gguf_type_GGUF_TYPE_STRING);
                    let el_ptr = mem::transmute::<_, *const GgufStr>(data).offset(offs as isize);
                    let el = slice::from_raw_parts((*el_ptr).data, (*el_ptr).n as usize);
                    assert_eq!(el, v);
                } else {
                    let sys_val = cstr_to_string(sys::gguf_get_val_str(sys_ctx, i));
                    assert_eq!(sys_val.as_bytes(), *v);
                }
            }
            Value::Array(v) => {
                if arr_elem.is_some() {
                    panic!("gguf doesn't support nested arrays");
                }
                let arr_n = sys::gguf_get_arr_n(sys_ctx, i);
                let arr_type = sys::gguf_get_arr_type(sys_ctx, i);
                let arr_data = sys::gguf_get_arr_data(sys_ctx, i);
                for (offs, v) in v.iter().enumerate() {
                    assert!((offs as i32) < arr_n);
                    assert_kv_val(sys_ctx, i, v, Some((arr_type, offs as i32, arr_data)));
                }
            }
            Value::Uint64(v) => {
                if let Some((arr_type, offs, data)) = arr_elem {
                    assert_eq!(arr_type, sys::gguf_type_GGUF_TYPE_UINT64);
                    assert_arr_elem(data, offs, v);
                } else {
                    let sys_val = sys::gguf_get_val_u64(sys_ctx, i);
                    assert_eq!(sys_val, *v);
                }
            }
            Value::Int64(v) => {
                if let Some((arr_type, offs, data)) = arr_elem {
                    assert_eq!(arr_type, sys::gguf_type_GGUF_TYPE_INT64);
                    assert_arr_elem(data, offs, v);
                } else {
                    let sys_val = sys::gguf_get_val_i64(sys_ctx, i);
                    assert_eq!(sys_val, *v);
                }
            }
            Value::Float64(v) => {
                if let Some((arr_type, offs, data)) = arr_elem {
                    assert_eq!(arr_type, sys::gguf_type_GGUF_TYPE_FLOAT64);
                    assert_arr_elem(data, offs, v);
                } else {
                    let sys_val = sys::gguf_get_val_f64(sys_ctx, i);
                    assert_eq!(sys_val, *v);
                }
            }
        }
    }

    unsafe fn assert_arr_elem<T: PartialEq + fmt::Debug>(data: *const c_void, offs: i32, v: &T) {
        let elemptr = mem::transmute::<_, *const T>(data).offset(offs as isize);
        let elem = &*elemptr;
        assert_eq!(elem, v);
    }

    // NB: We need to replicate this here as it's not exposed by the ggml API. Note that this could
    // be changed in upstream in which case our test here would break/
    #[repr(C)]
    struct GgufStr {
        n: u64,
        data: *const u8,
    }

    fn get_context() -> (Context, File, PathBuf) {
        let path = dirs::home_dir()
            .expect("$HOME must be set")
            .join(".config/haltia/orca_mini_v3_7b.Q3_K_S.gguf");
        assert!(
            path.exists(),
            "~/.config/haltia/orca_mini_v3_7b.Q3_K_S.gguf is required for this test to run"
        );
        let file = File::open(&path).unwrap();
        let (ctx, file) = Context::load(file).unwrap();
        (ctx, file, path)
    }

    #[test]
    fn q2k_block_size_same_as_ggml() {
        // TODO:
        // assert_eq!(
        //     mem::size_of::<gguf::k_quant::BlockQ3_K>(),
        //     mem::size_of::<sys::k_quants::block_q3_k>()
        // );
    }

    #[allow(unused)]
    pub unsafe fn cstr_to_string(s: *const c_char) -> Cow<'static, str> {
        assert!(!s.is_null(), "char pointer must be valid");
        unsafe { CStr::from_ptr(s) }.to_string_lossy()
    }
}

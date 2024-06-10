use std::{ffi::CStr, mem};

use cfg_if::cfg_if;
use thiserror::Error;

use crate::{Allocator, ComputeGraph, InitError, Tensor};

cfg_if! {
    if #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))] {
        pub mod metal;
        pub use self::metal::Metal;
        pub type DefaultBackend = Metal;
    } else if #[cfg(feature = "wip-cuda")] {
        pub mod cuda;
        pub use self::cuda::Cuda;
        pub type DefaultBackend = Cuda;
    }
}

#[derive(Debug, Error)]
#[error("failed to add buffer to accelerator backend")]
pub struct BufferError;

pub trait Backend {
    /// Creates a new accelerator backend without a specific compute graph or associated allocator.
    ///
    /// The graph and allocator have to be later fine tuned for this backend via the
    /// `Self::optimize_graph` and `Self::optimize_allocator` methods, if applicable.
    fn new() -> Result<Self, InitError>
    where
        Self: Sized;

    fn compute_graph(&mut self, cg: &mut ComputeGraph, n_threads: usize);

    fn get_result_tensor(&mut self, cg: &mut ComputeGraph) -> Tensor;
    fn get_embd_tensor(&mut self, cg: &mut ComputeGraph) -> Tensor;

    // === Optional methods ===

    /// Offloads the tensor to the backend device's memory.
    ///
    /// In case of Apple Silicon this is not defined because the architecture uses unified memory
    /// and as such the GPU accesses system RAM directly. This is only relevant in case of CUDA and
    /// other discrete GPU frameworks.
    #[allow(unused_variables)]
    fn offload_tensor(&mut self, tensor: &mut Tensor) {}

    #[allow(unused_variables)]
    fn add_buffer(
        &mut self,
        buf: &[u8],
        name: impl AsRef<[u8]>,
        max_size: Option<usize>,
    ) -> Result<(), BufferError> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn set_scratch_size(&mut self, size: usize) {}

    #[allow(unused_variables)]
    fn optimize_graph(&mut self, cg: &mut ComputeGraph) {}

    #[allow(unused_variables)]
    fn optimize_allocator<S>(&mut self, alloc: &mut Allocator<S>) {}
}

#[inline(always)]
fn get_result_tensor(cg: &mut ComputeGraph) -> Tensor {
    get_output_tensor(cg, 1, "result_output")
}

#[inline(always)]
fn get_embd_tensor(cg: &mut crate::ComputeGraph) -> Tensor {
    get_output_tensor(cg, 2, "result_norm")
}

/// Default implementation of extracting an output.
#[inline(always)]
fn get_output_tensor(cg: &mut crate::ComputeGraph, rev_idx: usize, name: &'static str) -> Tensor {
    unsafe {
        let cg = &*cg.0.as_mut_ptr();
        let t = cg.nodes[cg.n_nodes as usize - rev_idx];
        assert!(!t.is_null(), "{name} tensor is null");
        assert!(
            (*t).name.starts_with(mem::transmute(name.as_bytes())),
            "graph compute {name} output not in expected tensor, got: {:?}",
            CStr::from_ptr((*t).name.as_ptr()),
        );
        Tensor::from_raw(t)
    }
}

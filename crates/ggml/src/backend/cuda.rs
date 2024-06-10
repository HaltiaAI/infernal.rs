use crate::{sys, Backend};

pub struct Cuda {
    /// Reusable buffer for graph plan construction.
    work_buf: Vec<u8>,
}

impl Backend for Cuda {
    fn new() -> Result<Self, crate::InitError>
    where
        Self: Sized,
    {
        Ok(Self {
            work_buf: Vec::new(),
        })
    }

    fn offload_tensor(&mut self, tensor: &mut crate::Tensor) {
        unsafe {
            sys::ggml_cuda_assign_buffers_no_alloc(tensor.p());
        }
    }

    fn set_scratch_size(&mut self, size: usize) {
        unsafe {
            sys::ggml_cuda_set_scratch_size(size);
        }
    }

    fn compute_graph(&mut self, cg: &mut crate::ComputeGraph, n_threads: usize) {
        // NB: ggml-cuda doesn't have a `ggml_cuda_compute_graph` function, instead, the computation
        // is initiated in `ggml_compute_graph` which calls out to `ggml_compute_forward`, which
        // conditionally first calls out to `ggml_cuda_compute_forward` (evaluating the forward
        // graph's operations) and then if that couldn't compute, it does the compute on the cpu.
        unsafe {
            let mut plan = sys::ggml_graph_plan(cg.0.as_mut_ptr(), n_threads as i32);
            if plan.work_size > 0 {
                self.work_buf.resize(plan.work_size as usize, 0);
                plan.work_data = self.work_buf.as_mut_ptr();
            }
            sys::ggml_graph_compute(cg.p(), &mut plan);
        }
    }

    fn get_result_tensor(&mut self, cg: &mut crate::ComputeGraph) -> crate::Tensor {
        super::get_result_tensor(cg)
    }

    fn get_embd_tensor(&mut self, cg: &mut crate::ComputeGraph) -> crate::Tensor {
        super::get_embd_tensor(cg)
    }
}

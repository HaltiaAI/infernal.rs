use crate::{sys, OwnedOrPtr, Tensor};

// TODO: maybe use same generic pattern as above
pub struct ComputeGraph(pub(crate) OwnedOrPtr<sys::ggml_cgraph>);

impl ComputeGraph {
    #[allow(clippy::needless_pass_by_ref_mut)]
    #[inline(always)]
    pub fn build_forward_expand(&mut self, t: &mut Tensor) {
        unsafe { sys::ggml_build_forward_expand(self.0.as_mut_ptr(), t.p()) }
    }

    #[inline(always)]
    pub fn run_on_cpu(&mut self, buf: &mut Vec<u8>, n_threads: usize) {
        unsafe {
            let mut plan = sys::ggml_graph_plan(self.0.as_mut_ptr(), n_threads as _);
            if plan.work_size > 0 {
                buf.resize(plan.work_size, 0);
                plan.work_data = buf.as_mut_ptr();
            }
            sys::ggml_graph_compute(self.0.as_mut_ptr(), &mut plan);
        }
    }
}

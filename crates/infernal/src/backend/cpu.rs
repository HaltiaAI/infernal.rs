use crate::buf::Buf;
use crate::BufId;
use crate::ComputeGraph;
use crate::Tensor;
use crate::TensorSrc;
use crate::Optimizer;
use crate::Op;

use super::Backend;
use super::BackendBuf;
use super::MemoryLimitTooLow;
use super::DanglingTensor;
use super::ComputeError;
use super::paging::Device;
use super::paging::DeviceInfo;
use super::paging;

use std::sync::Arc;

pub struct CpuBackend {
    page_cache: paging::PageCache<CpuDevice>,
}

impl CpuBackend {
    fn borrow2(&self, 
        ctx: &impl TensorSrc,
        t0: &Tensor, 
        t1: &Tensor
    ) -> Result<(&[u8], &mut [u8]), ComputeError> {
        let [buf_a, buf_dst] = self.page_cache.get2(t0, t1)?;

        let a = unsafe {
            let offset = t0.data_offs();
            let size = t0.size();
            let view = &buf_a.buf.borrow()[offset..offset + size];

            &*(view as *const _)
        };

        let dst = unsafe {
            let offset = t1.data_offs();
            let size = t1.size();
            let view = &mut buf_dst.buf.as_shared_mut().expect("Data might not be mapped in")[offset..offset + size];

            &mut *(view as *mut _)
        };

        Ok((a, dst))
    }

    fn dispatch_op<'scope, 'env, 'this: 'scope>(
        &'this self,
        ctx: &impl TensorSrc,
        dst: &Tensor,
    ) -> Result<(), ComputeError> {
        match dst.op {
            Op::Const => return Ok(()),
            Op::Unary { op, a } => {
                let a = ctx.get_tensor(a)?;
                let (buf_a, buf_dst) = self.borrow2(ctx, a, dst)?;

                match op {
                    _ => todo!(),
                }
            }
            _ => return Ok(())
        }

    }
}

impl Backend for CpuBackend {
    fn add_buf(&mut self, buf: BackendBuf) -> Result<(), MemoryLimitTooLow> {
        self.page_cache.insert(buf)
    }

    fn get_tensor_buf(&self, tensor: &Tensor) -> Result<&BackendBuf, DanglingTensor> {
        self.page_cache.get(tensor.buf_id).ok_or_else(|| DanglingTensor::new(tensor))
    }

    fn get_buf(&self, buf_id: BufId) -> Option<&BackendBuf> {
        self.page_cache.get(buf_id)
    }

    fn mem_limit(&self) -> usize {
        usize::MAX
    }

    fn set_mem_limit(&mut self, mem_limit: usize) -> Result<(), MemoryLimitTooLow> {
        Ok(())
    }

    fn compute_graph<O: Optimizer>(&mut self, graph: &ComputeGraph<O>, ctx: &impl TensorSrc) -> Result<(), ComputeError> {
        let exec_plan = graph 
            .exec_plan(self.mem_limit())
            .ok_or_else(|| ComputeError::MissingExecutionPlan(self.mem_limit()))?;

        if exec_plan.levels.is_empty() {
            return Err(ComputeError::EmptyExecutionPlan);
        }

        let mut new_vram_use = 0;

        for (level_idx, level) in exec_plan.levels.iter().enumerate() {
            let is_last_level = level_idx + 1 == exec_plan.levels.len();
            for (i, node_idx) in level.iter().enumerate() {
                let is_last_node = is_last_level && i + 1 == level.len();

                let node_id = graph.nodes()[*node_idx];
                let dst = ctx.get_tensor(node_id)?;

                if dst.op.is_noop() {
                    continue;
                }

                dst.visit_nodes(ctx, |n| {
                    new_vram_use += self.page_cache.stage(&CpuDevice, n.buf_id);
                });

                self.dispatch_op(ctx, dst)?;
            }
        }

        Ok(())
    }
}

pub struct CpuDevice;

impl Device for CpuDevice {
    type Buf = ();

    fn info(&self) -> DeviceInfo {
        DeviceInfo {
            mem_limit: usize::MAX,
            max_buf_size: usize::MAX,
        }
    }

    fn new_buf(&self, _: &BackendBuf) -> std::io::Result<Self::Buf> {
        Ok(())
    }

    fn free_buf(&self, _: Self::Buf) -> std::io::Result<()> {
        Ok(())
    }

    fn has_noncopy_bufs(&self) -> bool {
        true
    }
}

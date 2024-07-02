use crate::{BufId, ComputeGraph, Op, Optimizer, Tensor, TensorSrc};

use super::{
    paging,
    paging::{Device, DeviceInfo},
    Backend, BackendBuf, ComputeError, DanglingTensor, MemoryLimitTooLow,
};

pub struct Cpu {
    page_cache: paging::PageCache<CpuDevice>,
}

impl Cpu {
    /// Method borrows two tensors backing memory store. `t0` is borrowed immutably and is used as
    /// the src tensor, whule `t1` is borrowed mutably and is used as the `dst` tensor.
    ///
    /// # Safety
    /// Callers must ensure that `t0` and `t1` don't overlap if they point to the same buffer and
    /// they must ensure that `t0` and `t1` live and are not unloaded for as long as the returned
    /// slices live.
    fn borrow2<'t>(
        &self,
        t0: &'t Tensor,
        t1: &'t Tensor,
    ) -> Result<(&'t [u8], &'t mut [u8]), ComputeError> {
        let [buf_a, buf_dst] = self.page_cache.get2(t0, t1)?;

        debug_assert!(
            t0.buf_id() != t1.buf_id()
                || (t0.data_offs() + t0.size() <= t1.data_offs()
                    || t1.data_offs() + t1.size() <= t0.data_offs()),
            "t0 and t1 have the same buf_id but overlap"
        );

        let a = unsafe {
            let offset = t0.data_offs();
            let size = t0.size();
            let view = &buf_a.buf.borrow()[offset..offset + size];

            &*(view as *const _)
        };

        let dst = unsafe {
            let offset = t1.data_offs();
            let size = t1.size();
            let view = &mut buf_dst
                .buf
                .as_shared_mut()
                .expect("Data might not be mapped in")[offset..offset + size];

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
                let (buf_a, buf_dst) = self.borrow2(a, dst)?;

                match op {
                    _ => todo!(),
                }
            }
            _ => return Ok(()),
        }
    }
}

impl Backend for Cpu {
    fn add_buf(&mut self, buf: BackendBuf) -> Result<(), MemoryLimitTooLow> {
        self.page_cache.insert(buf)
    }

    fn get_tensor_buf(&self, tensor: &Tensor) -> Result<&BackendBuf, DanglingTensor> {
        self.page_cache
            .get(tensor.buf_id)
            .ok_or_else(|| DanglingTensor::new(tensor))
    }

    fn get_buf(&self, buf_id: BufId) -> Option<&BackendBuf> {
        self.page_cache.get(buf_id)
    }

    fn mem_limit(&self) -> usize {
        self.page_cache.mem_limit()
    }

    fn set_mem_limit(&mut self, mem_limit: usize) -> Result<(), MemoryLimitTooLow> {
        self.page_cache.set_mem_limit(mem_limit)
    }

    fn compute_graph<O: Optimizer>(
        &mut self,
        graph: &ComputeGraph<O>,
        ctx: &impl TensorSrc,
    ) -> Result<(), ComputeError> {
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
        let info = sysinfo::System::new_all();

        DeviceInfo {
            mem_limit: info.available_memory() as _,
            max_buf_size: info.available_memory() as _,
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

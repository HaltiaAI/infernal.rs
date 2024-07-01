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

use std::thread::scope;
use std::thread::Scope;
use std::thread::ScopedJoinHandle;
use std::sync::Arc;
use std::sync::Mutex;

pub struct CpuBackend {
    page_cache: paging::PageCache<CpuDevice>,
}

impl CpuBackend {
    fn dispatch_op<'scope, 'env, 'this: 'scope>(
        &'this self,
        ctx: &impl TensorSrc,
        dst: &Tensor,
        scope: &'scope Scope<'scope, 'env>
    ) -> Result<Option<ScopedJoinHandle<'scope, ()>>, ComputeError> {
        match dst.op {
            Op::Const => return Ok(None),
            Op::Unary { op, a } => {
                let a = ctx.get_tensor(a)?;
                let [buf_a, buf_dst] = self.page_cache.get2(a, dst)?;

                return Ok(Some(scope.spawn(move || {
                    let [buf_a, buf_dst] = [buf_a, buf_dst];

                    println!("op: {:?}", op);
                })));
            }
            _ => return Ok(None)
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

            // NOTE: To avoid double borrows we must stage all the buffers before executing them
            for node_idx in level {
                let node_id = graph.nodes()[*node_idx];
                let dst = ctx.get_tensor(node_id).unwrap();

                // NB: This is actually an important check because we want to avoid paging buffers
                // for noops, which would be done before the op is dispatched.
                if dst.op.is_noop() {
                    continue;
                }

                dst.visit_nodes(ctx, |n| {
                    new_vram_use += self.page_cache.stage(&CpuDevice, n.buf_id);
                });
            }

            scope(|s| {
                let mut threads = Vec::with_capacity(level.len());

                for (i, node_idx) in level.iter().enumerate() {
                    let is_last_node = is_last_level && i + 1 == level.len();

                    let node_id = graph.nodes()[*node_idx];
                    let dst = ctx.get_tensor(node_id).unwrap();

                    if dst.op.is_noop() {
                        continue;
                    }

                    self.dispatch_op(ctx, dst, s).unwrap().map(|thread| threads.push(thread));
                }

                for thread in threads.drain(..) {
                    thread.join().unwrap();
                }
            });
        }

        Ok(())
    }
}

pub struct CpuDevice;

impl Device for CpuDevice {
    type Buf = Arc<Mutex<Buf>>;

    fn info(&self) -> DeviceInfo {
        DeviceInfo {
            mem_limit: usize::MAX,
            max_buf_size: usize::MAX,
        }
    }

    fn new_buf(&self, buf: &BackendBuf) -> std::io::Result<Self::Buf> {
        return Ok(buf.as_shared().unwrap().clone());
    }

    fn free_buf(&self, buf: Self::Buf) -> std::io::Result<()> {
        Ok(())
    }

    fn has_noncopy_bufs(&self) -> bool {
        true
    }
}

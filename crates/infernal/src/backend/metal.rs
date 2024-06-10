#![allow(unused)]

use std::time::Instant;

use metal::{
    CommandBufferRef, CommandQueue, CompileOptions, ComputeCommandEncoderRef,
    ComputePipelineDescriptor, ComputePipelineState, Device, Library, MTLDispatchType,
    MTLResourceOptions,
};
use objc::rc::autoreleasepool;
use thiserror::Error;
use tracing::{debug, error, trace, warn};

use crate::{
    backend::{
        metal::kernel::Kernel,
        paging::{self, PageCache},
        Backend, BackendBuf, DanglingTensor, DeviceNotFound, MemoryLimitTooLow,
    },
    buf::BufId,
    graph::{ComputeGraph, Optimizer},
    op::{BinaryOp, UnaryOp},
    tensor::Tensor,
    utils::Bytes,
    Op, TensorNotFound, TensorSrc,
};

mod kernel;

pub(crate) const PROGRAM: &str = include_str!("program.metal");

pub struct Metal {
    device: Device,
    cmd_queue: CommandQueue,
    _lib: Library,
    page_cache: PageCache<Device>,
    pipelines: Pipelines,
}

#[derive(Debug, Default)]
pub struct Config {
    /// This is the maximum number of bytes that the backend can use for buffers at any given time.
    ///
    /// Since Darwin imposes a maximum RAM use limit on Metal, this value must be less than that
    /// value. The exact limit is not documented and varies by device, but a safe rule of thumb is
    /// to assume that a bit less than half of the total system RAM is usable by Metal. This is true
    /// on iPhone devices and on smaller Macs, however, when the system RAM increases so does this
    /// limt. On high-end Macbooks, the limit may be as high as 70%.
    ///
    /// Note, however, that this does not actually mean the total RAM used by your app will respect
    /// this limit. This simply imposes a limit on the total buffer size mapped into Metal. Whether
    /// the buffers actually reside in RAM depends on the paging policy configured for the backend.
    /// See [`PageCache`] for more details.
    ///
    /// For now this value is not verified by the backend, so it is up to the user to ensure that
    /// things work correctly. TODO: add verification.
    ///
    /// Further, this value can be set to something significantly lower than the abovel imit if the
    /// use case requires a low memory footprint. As noted, the actual RAM use is only limited if
    /// the paging system used unloads unused buffers from memory. In that case do note that this
    /// probably impacts performance as buffers would need to be constantly reloaded from disk.
    pub mem_limit: Option<usize>,
    // TODO: add paging policy, something like:
    // pub paging_system: impl PagingSystem,
}

impl Backend for Metal {
    fn add_buf(&mut self, buf: BackendBuf) -> Result<(), MemoryLimitTooLow> {
        self.page_cache.insert(buf)
    }

    #[inline(always)]
    fn get_buf(&self, buf_id: BufId) -> Option<&BackendBuf> {
        self.page_cache.get(buf_id)
    }

    /// Returns the maximum number of bytes that the backend can use for buffers at a time.
    ///
    /// This can affect the execution plan. See more in [`ComputeGraph::exec_plan`].
    #[inline(always)]
    fn mem_limit(&self) -> usize {
        self.page_cache.mem_limit()
    }

    fn set_mem_limit(&mut self, mem_limit: usize) -> Result<(), MemoryLimitTooLow> {
        self.page_cache.set_mem_limit(mem_limit)
    }

    #[inline(always)]
    fn compute_graph<O: Optimizer>(
        &mut self,
        graph: &ComputeGraph<O>,
        ctx: &impl TensorSrc,
    ) -> Result<(), ComputeError> {
        let mut res = Ok(());
        autoreleasepool(|| {
            self.compute_graph_impl(graph, ctx).unwrap_or_else(|e| {
                res = Err(e);
            });
        });
        res
    }
}

impl Metal {
    pub fn new() -> Result<Self, DeviceNotFound> {
        Self::with_config(Config::default())
    }

    pub fn with_config(conf: Config) -> Result<Self, DeviceNotFound> {
        let device = Device::system_default().ok_or(DeviceNotFound("Metal"))?;
        let options = CompileOptions::new();
        let lib = device.new_library_with_source(PROGRAM, &options).unwrap();
        let cmd_queue = device.new_command_queue();
        let pipelines = Pipelines::new(&device, &lib);

        let device_mem_limit = device.recommended_max_working_set_size() as usize;
        let mem_limit = if let Some(mem_limit) = conf.mem_limit {
            if mem_limit > device_mem_limit {
                warn!(
                    mem_limit = %Bytes(mem_limit),
                    device_mem_limit = %Bytes(device_mem_limit),
                    "configured memory limit exceeds device max buffer size, using device limit"
                );
                device_mem_limit
            } else {
                mem_limit
            }
        } else {
            device_mem_limit
        };

        warn!(
            device = device.name(),
            all_devices = ?Device::all(),
            ?options,
            has_unified_memory = device.has_unified_memory(),
            recommended_max_working_set_size = %Bytes(device.recommended_max_working_set_size() as usize),
            // max_transfer_rate = if device.max_transfer_rate() > 0 {
            //     Bytes(device.max_transfer_rate() as usize).to_string()
            // } else {
            //     "built-in GPU".to_string()
            // },
            mem_limit = %Bytes(mem_limit),
            max_buffer_size = %Bytes(device.max_buffer_length() as usize),
            "initializing backend"
        );

        let page_cache = PageCache::new(mem_limit, &device);
        Ok(Self {
            device,
            cmd_queue,
            _lib: lib,
            page_cache,
            pipelines,
        })
    }

    fn compute_graph_impl<O: Optimizer>(
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

        let mut exec_unit = ExecUnit::new(&self.cmd_queue);

        // TODO(perf): impl parallel command encoding. although this does not have much of an effect
        // on typical llms as the concurrency is typically very low (max 3).
        for (level_idx, level) in exec_plan.levels.iter().enumerate() {
            let is_last_level = level_idx + 1 == exec_plan.levels.len();
            debug!(
                level_idx,
                is_last_level,
                n_nodes = level.len(),
                "preparing graph level for execution"
            );
            for (i, node_idx) in level.iter().enumerate() {
                let is_last_node = is_last_level && i + 1 == level.len();

                let node_id = graph.nodes()[*node_idx];
                let dst = ctx.get_tensor(node_id)?;

                // NB: This is actually an important check because we want to avoid paging buffers
                // for noops, which would be done before the op is dispatched.
                if dst.op.is_noop() {
                    continue;
                }

                debug!(level_idx, node_idx, %dst.name, "scheduling node");

                // TODO: move out paging logic to own function, but currently there are lifetime
                // issues with &self captured by the exec unit and the paging routine requiring &mut
                // self as well.
                // easy to solve by moving execunit into self, but then execunit would be
                // self-referential, so doesn't work. sigh.
                let start = Instant::now();

                // Stage this node's buffers, which means we don't commit to executing it in this
                // exec unit as its buffers may cause us to exceed the memory limit.
                let mut new_vram_use = 0;
                dst.visit_nodes(ctx, |n| {
                    new_vram_use += self.page_cache.stage(&self.device, n.buf_id);
                });

                // If this op tensor's buffers would cause us to exceed the memory limit,
                // try to offload unused past buffers from the page cache.
                // TODO: if the memory limit is hit without being able to execute this buffer, unmap
                // old buffers and try again. for now we only care about metal mem usage, later on
                // we can have a 2 tiered total ram + metal ram limit system. for now kiss me.
                self.page_cache.free_space(&self.device)?;

                // If total expected memory usage still exceeds limit after trimming the metal
                // cache, meaning we couldn't offload enough pages, execute nodes in the exec unit
                // up to but not including this op. This node is executed in the next exec unit.
                if self.page_cache.expected_vram_use() > self.mem_limit() {
                    debug!(
                        mem_limit = %Bytes(self.mem_limit()),
                        expected_vram_use = %Bytes(self.page_cache.expected_vram_use()),
                        cur_vram_use = %Bytes(self.page_cache.vram_use()),
                        new_vram_use = %Bytes(new_vram_use),
                        "reached max memory use, committing command buffer"
                    );

                    if exec_unit.queue_len == 0 {
                        error!(
                            new_vram_use = %Bytes(new_vram_use),
                            expected_vram_use = %Bytes(self.page_cache.expected_vram_use()),
                            mem_limit = %Bytes(self.mem_limit()),
                            "did not schedule any nodes, memory limit too low"
                        );
                        // This can happen if the op size is larger than the max memory size. In this
                        // case the user needs to update the max memory size.
                        return Err(MemoryLimitTooLow {
                            needed: self.page_cache.expected_vram_use(),
                            limit: self.mem_limit(),
                        }
                        .into());
                    }

                    // Execute currently queued nodes
                    exec_unit.run();

                    // NB don't create a new command buffer if this is the last node because that would
                    // cause a crash due to not having encoded any commands before releasing the encoder.
                    if !is_last_node {
                        exec_unit = ExecUnit::new(&self.cmd_queue);
                    }

                    // Need to clear execution queue to make room for the next level's buffers.
                    // NB: We don't clear the staging queue here because that's going to be executed
                    // right away.
                    self.page_cache.clear_exec_queue();
                }

                // Page in buffers now that we know they fit in memory.
                dst.try_visit_nodes(ctx, |n| self.page_cache.load(&self.device, n.buf_id))?;
                let paging_time = start.elapsed();

                exec_unit.queue_len += 1;

                let start = Instant::now();
                self.dispatch_op(ctx, dst, &mut exec_unit)?;
                let dispatch_time = start.elapsed();

                trace!(level_idx, node_idx, %dst.name, ?paging_time, ?dispatch_time, "scheduled node");
            }

            // This only happens if the graph execution plan is borked. If it's simply a problem
            // of fitting the tensor in the configured memory limit, that's properly handled above.
            debug_assert!(
                exec_unit.queue_len > 0,
                "Did not encode any commands to buffer. This is a bug, please report."
            );

            // We don't need to issue a barrier for the last level, there are no commands after
            // it before waiting for existing ones to be executed.
            if !is_last_level {
                exec_unit.issue_barrier();
            }
        }

        exec_unit.run();
        // Need to clear all queues for the upcoming execution plan.
        self.page_cache.clear_staging_queue();
        self.page_cache.clear_exec_queue();

        Ok(())
    }

    fn dispatch_op(
        &self,
        ctx: &impl TensorSrc,
        dst: &Tensor,
        exec: &mut ExecUnit,
    ) -> Result<(), ComputeError> {
        match dst.op {
            // This shouldn't happen as const tensors aren't in the concur list, but
            // keeping here as a sanity check.
            Op::Const => return Ok(()),
            Op::Unary { op, a } => {
                trace!(?op, %a, "unary op");
                let a = ctx.get_tensor(a)?;
                trace!(
                    ?op,
                    a = %a.name,
                    %a.layout,
                    dst = %dst.name,
                    %dst.layout,
                );
                let [buf_a, buf_dst] = self.page_cache.get2(a, dst)?;

                let encoder = &exec.encoder;
                let dst = (dst, buf_dst).into();
                let src = [(a, buf_a).into()];

                use UnaryOp::*;
                match op {
                    // Noop tensors on the other hand _are_ in the concur list, but we don't need to
                    // do anything to them. They are just here to glue things together.
                    Noop(n) => {
                        trace!(%n, "noop");
                        return Ok(());
                    }
                    Rope(args) => {
                        trace!(?args, "rope");
                        kernel::RopeOld::new(self, dst, src, args).encode(encoder);
                    }
                    RmsNorm { eps } => {
                        trace!(%eps, "rms norm");
                        kernel::RmsNorm::new(self, dst, src, eps).encode(encoder);
                    }
                    DiagMaskInf { n_past } => {
                        trace!(%n_past, "diag mask inf");
                        kernel::DiagMaskInf::new(self, dst, src, n_past).encode(encoder);
                    }
                    Softmax { scale, .. } => {
                        trace!(scale, "softmax");
                        // TODO: we don't support masking for now
                        // NB: Softmax is a binary operation but the kernel supports setting a custom mask which
                        // we're not using. If the mask is not set, it reuses the first buffer as the second buffer.
                        let src = [src[0], src[0]];
                        kernel::Softmax::new(self, dst, src, scale).encode(encoder);
                    }
                    Silu => {
                        trace!("silu");
                        kernel::Silu::new(self, dst, src).encode(encoder);
                    }
                    Relu => {
                        trace!("relu");
                        kernel::Relu::new(self, dst, src).encode(encoder);
                    }
                }
            }
            Op::Binary { op, a, b } => {
                trace!(?op, %a, %b, "binary op");
                let a = ctx.get_tensor(a)?;
                let b = ctx.get_tensor(b)?;
                trace!(
                    ?op,
                    a = %a.name,
                    %a.layout,
                    b = %b.name,
                    %b.layout,
                    dst = %dst.name,
                    %dst.layout,
                );
                let [buf_a, buf_b, buf_dst] = self.page_cache.get3(a, b, dst)?;

                let dst = (dst, buf_dst).into();
                let src = [(a, buf_a).into(), (b, buf_b).into()];

                let encoder = &exec.encoder;
                use BinaryOp::*;
                match op {
                    Matmul => {
                        if kernel::use_mm_kernel_old(self, &src) {
                            trace!("matmul: using MM kernel");
                            kernel::MulMMOld::new(self, dst, src).encode(encoder);
                        } else {
                            trace!("matmul: using MV kernel");
                            kernel::MulMVOld::new(self, dst, src).encode(encoder);
                        }
                    }
                    Add => {
                        kernel::AddOld::new(self, dst, src).encode(encoder);
                        // kernel::BinArith::new(
                        //     &self.pipelines.add,
                        //     &self.pipelines.add_row,
                        //     dst,
                        //     src,
                        // )
                        // .encode(encoder);
                    }
                    Mul => {
                        kernel::MulOld::new(self, dst, src).encode(encoder);
                    }
                    Div => {
                        kernel::BinArith::new(
                            &self.pipelines.div,
                            &self.pipelines.div_row,
                            dst,
                            src,
                        )
                        .encode(encoder);
                    }
                    Scale => {
                        // Scale behaves a little differently. Ideally we should just be accepting
                        // a single f32 and make this an unary operation, however the scalar may be
                        // calculated lazily as part of the compute graph and thus needs to be
                        // a tensor.
                        // TODO: error handling here
                        let scale = src[1].tensor.elements::<f32>(self)[0];
                        let src = [src[0]];
                        kernel::Scale::new(self, dst, src, scale).encode(encoder);
                    }
                    Im2Col(v) => {
                        kernel::Im2Col::new(self, dst, src, v).encode(encoder);
                    }
                    GetRows => {
                        kernel::GetRows::new(self, dst, src).encode(encoder);
                    }
                    Copy => {
                        trace!(from = %a.id, original_to = %b.id, "copy op");
                        // NB: Copy is a binary operation because both parent nodes are needed by
                        // the graph optimizer but the kernel only cares about the source tensor to perform
                        // the copy.
                        let src = [src[0]];
                        kernel::Copy::new(self, dst, src).encode(encoder);
                    }
                }
            }
        }
        Ok(())
    }
}

impl paging::Device for Device {
    type Buf = metal::Buffer;

    fn info(&self) -> paging::DeviceInfo {
        paging::DeviceInfo {
            max_buf_size: self.max_buffer_length() as usize,
            mem_limit: self.recommended_max_working_set_size() as usize,
        }
    }

    fn new_buf(&self, buf: &[u8]) -> std::io::Result<Self::Buf> {
        let aligned_size = self.align(buf.len());

        // SAFETY:
        // Since we have ownership of the original storage and we only deallocate at the end of this
        // type's lifetime, giving out a pointer to it to Metal is safe.
        let mtl_buf = self.new_buffer_with_bytes_no_copy(
            buf.as_ptr() as _,
            aligned_size as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        Ok(mtl_buf)
    }

    fn free_buf(&self, _: Self::Buf) -> std::io::Result<()> {
        // This is a noop on Metal
        Ok(())
    }

    fn has_noncopy_bufs(&self) -> bool {
        self.has_unified_memory()
    }

    fn align(&self, size: usize) -> usize {
        let page_size = crate::buf::get_page_size();
        let aligned_size = crate::buf::align(size, page_size);
        debug_assert!(aligned_size % page_size == 0);
        aligned_size
    }
}

#[derive(Debug, Error)]
pub enum ComputeError {
    #[error(transparent)]
    DanglingTensor(#[from] DanglingTensor),
    #[error(transparent)]
    TensorNotFound(#[from] TensorNotFound),
    #[error("graph does not have an execution plan for {0} byte memory limit")]
    MissingExecutionPlan(usize),
    #[error("graph execution plan is empty, cannot execute")]
    EmptyExecutionPlan,
    #[error(transparent)]
    MemoryLimitTooLow(#[from] MemoryLimitTooLow),
    #[error("error paging buffer in/out: {0}")]
    Paging(#[from] std::io::Error),
}

/// Helper type for encoding and executing Metal commands.
struct ExecUnit<'a> {
    cmd_buf: &'a CommandBufferRef,
    encoder: &'a ComputeCommandEncoderRef,
    // Currently queued nodes.
    queue_len: usize,
    // Used for avoiding double comitting.
    done: bool,
}

impl<'a> ExecUnit<'a> {
    fn new(cmd_queue: &'a CommandQueue) -> Self {
        let cmd_buf = cmd_queue.new_command_buffer();
        cmd_buf.enqueue();
        let encoder =
            cmd_buf.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
        Self {
            encoder,
            cmd_buf,
            queue_len: 0,
            done: false,
        }
    }

    fn run(&mut self) -> bool {
        if !self.done && self.queue_len > 0 {
            debug!(queue_len = self.queue_len, "comitting command buffer");

            let start = Instant::now();

            self.encoder.end_encoding();
            let encoding_time = start.elapsed();
            self.cmd_buf.commit();
            let commit_time = start.elapsed() - encoding_time;
            self.cmd_buf.wait_until_completed();

            debug!(
                ?encoding_time,
                ?commit_time,
                total_time = ?start.elapsed(),
                queue_len = self.queue_len,
                "comitted command buffer"
            );

            self.done = true;
            self.queue_len = 0;

            true
        } else {
            false
        }
    }

    fn issue_barrier(&mut self) {
        debug!(queue_len = self.queue_len, "issuing barrier");

        // TODO: send PR to metal-rs
        #[allow(non_upper_case_globals)]
        const MTLBarrierScopeBuffers: u64 = 1 << 0;
        // TODO: send PR to objc to use $crate::sel
        use objc::*; // needed for sel! macro msg_send! is accessing
        unsafe { msg_send![self.encoder, memoryBarrierWithScope: MTLBarrierScopeBuffers] }
    }
}

// We implement drop in case the program short circtuis before the command buffer is committed.
impl<'a> Drop for ExecUnit<'a> {
    fn drop(&mut self) {
        if !self.run() && !self.done {
            // We need to do this because it may be that we can't commit the command buffer (e.g.
            // due to an error) and the encoder is dropped without calling endEncoding, which causes
            // a SIGABORT.
            self.encoder.end_encoding();
        }
    }
}

/// Helper macro that takes the name of a kernel and implements the Pipelines type, populating its
/// fields with the kernel (minus the "kernel_" prefix) and a constructor to instantiate all the
/// pipelines.
macro_rules! pipelines {
    ($($name:ident),* $(,)?) => {
        #[allow(dead_code, non_snake_case)]
        struct Pipelines {
            $($name: ComputePipelineState),*
        }

        impl Pipelines {
            fn new(device: &Device, lib: &Library) -> Self {
                Self {
                    $($name: Self::load_pipeline(device, lib,  concat!("kernel_", stringify!($name)))),*
                }
            }

            fn load_pipeline(device: &Device, lib: &Library, name: &str) -> ComputePipelineState {
            // TODO: return error instead of panic
                let kernel = lib
                    .get_function(name, None)
                    .unwrap_or_else(|e| {
                        panic!("kernel '{name}' missing from Metal library: {e}");
                    });
                let desc = ComputePipelineDescriptor::new();
                // TODO: experiment with this optimization:
                // desc.set_thread_group_size_is_multiple_of_thread_execution_width(true);
                desc.set_compute_function(Some(&kernel));
                let pipeline = device
                    // .new_compute_pipeline_state_with_function(&kernel)
                    .new_compute_pipeline_state(&desc)
                    .unwrap();
                debug!(
                    pipeline = name,
                    max_total_threads_per_threadgroup = pipeline.max_total_threads_per_threadgroup(),
                    thread_execution_width = pipeline.thread_execution_width(),
                    "loaded pipeline",
                );
                pipeline
            }
        }
    };
}

pipelines! {
    // -- binary ops --

    // matmul: matrix-matrix
    mul_mm_f32_f32,
    mul_mm_f16_f32,
    mul_mm_q2_K_f32,
    mul_mm_q3_K_f32,
    mul_mm_q4_K_f32,
    mul_mm_q5_K_f32,
    mul_mm_q6_K_f32,
    // older version: 0a5eebb
    mul_mm_f16_f32_old,
    mul_mm_q2_K_f32_old,
    mul_mm_q3_K_f32_old,
    mul_mm_q4_K_f32_old,
    mul_mm_q5_K_f32_old,
    mul_mm_q6_K_f32_old,

    // matmul: matrix-vector
    mul_mv_f32_f32,
    mul_mv_f16_f32_1row,
    mul_mv_f16_f32_l4,
    mul_mv_f16_f32,
    mul_mv_f16_f16,
    mul_mv_q2_K_f32,
    mul_mv_q3_K_f32,
    mul_mv_q4_K_f32,
    mul_mv_q5_K_f32,
    mul_mv_q6_K_f32,
    // older version: 0a5eebb
    mul_mat_f16_f32_1row_old,
    mul_mat_f16_f32_l4_old,
    mul_mat_f16_f32_old,
    mul_mat_q2_K_f32_old,
    mul_mat_q3_K_f32_old,
    mul_mat_q4_K_f32_old,
    mul_mat_q5_K_f32_old,
    mul_mat_q6_K_f32_old,

    add,
    add_row,
    // older version: 0a5eebb
    add_old,
    add_row_old,

    mul,
    mul_row,
    // older version: 0a5eebb
    mul_old,
    mul_row_old,

    div,
    div_row,

    im2col_f16,

    get_rows_f32,
    get_rows_f16,
    // get_rows_q4_0,
    // get_rows_q4_1,
    // get_rows_q5_0,
    // get_rows_q5_1,
    // get_rows_q8_0,
    get_rows_q2_K,
    get_rows_q3_K,
    get_rows_q4_K,
    get_rows_q5_K,
    get_rows_q6_K,


    // -- unary ops --

    cpy_f32_f32,
    cpy_f32_f16,
    cpy_f16_f16,

    rms_norm,

    // rope_f32,
    // rope_f16,
    // older version: 0a5eebb
    rope_old,

    scale_4,
    scale,

    diag_mask_inf,
    diag_mask_inf_8,

    soft_max,
    soft_max_4,

    silu,
    relu,
}

#[cfg(test)]
mod tests {
    use crate::{Layout, TensorBuilder, Type};

    use super::*;

    #[ignore]
    #[test]
    fn mem_limit_too_low() -> anyhow::Result<()> {
        let layout = Layout::new_2d(128, 128, Type::F32);
        let buf_size = 3 * layout.size();
        let mut backend = Metal::with_config(Config {
            mem_limit: Some(buf_size - 1),
        })?;

        let mut tb = TensorBuilder::prealloc(1024 * 1024)?;
        let x = tb.new_tensor(layout)?;
        let y = tb.new_tensor(layout)?;
        tb.fill_2d_f32(x, 1.3);
        tb.fill_2d_f32(y, 9.5);
        let res = tb.matmul(x, y)?;

        let (ctx, _) = tb.finish(&mut backend)?;

        let mut gf = ComputeGraph::default();
        gf.expand_forward(res, &ctx)?;
        gf.find_exec_plan(buf_size - 1, &ctx)?;
        dbg!(gf.exec_plan(buf_size - 1));
        let res = backend.compute_graph(&gf, &ctx);

        println!("{res:?}");
        assert!(matches!(
            res,
            Err(ComputeError::MemoryLimitTooLow(MemoryLimitTooLow {
                needed,
                limit
            }))
            if needed == buf_size && limit == buf_size - 1
        ));

        Ok(())
    }
}

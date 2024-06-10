use metal::{ComputePipelineState, MTLSize};
use tracing::trace;

use crate::{
    args,
    backend::{
        metal::kernel::{Arg, BinarySrc, Kernel, TensorBuf, A, B},
        Metal,
    },
    unsupported_type,
    Type::*,
};

/// Find the break-even point where the matrix-matrix kernel becomes
/// more efficient compared to the matrix-vector kernel.
pub(crate) fn use_mm_kernel_old(metal: &Metal, src: &BinarySrc) -> bool {
    let a = src[A].tensor;
    let b = src[B].tensor;

    // For now the matrix-matrix multiplication kernel only works on
    // A14+/M1+ SoCs AMD GPU and older A-chips will reuse matrix-vector
    // multiplication kernel.
    metal.device.supports_family(metal::MTLGPUFamily::Apple7)
        && a.is_contiguous()
        && b.is_contiguous()
        && b.typ() == F32
        && a.cols() % 32 == 0
        && b.rows() > 1
}

pub(crate) struct MulMMOld<'a> {
    pipeline: &'a ComputePipelineState,
    src: BinarySrc<'a>,
    dst: TensorBuf<'a>,

    gqa: u32,
}

impl<'a> MulMMOld<'a> {
    pub fn new(metal: &'a Metal, dst: TensorBuf<'a>, src: BinarySrc<'a>) -> Self {
        let a = src[A].tensor;
        let b = src[B].tensor;

        let [a_cols, _, a_depth, a_batches] = a.dims;

        let [b_cols, _, b_depth, b_batches] = b.dims;

        // These need to be verified in the op builder.
        debug_assert_eq!(a_cols, b_cols);
        debug_assert_eq!(a_batches, b_batches);
        debug_assert_eq!(b.typ(), F32);

        let pipeline = match a.typ() {
            F16 => {
                trace!("mul_mm_f16_f32_old");
                &metal.pipelines.mul_mm_f16_f32_old
            }
            Q2_K => {
                trace!("mul_mm_q2_K_f32_old");
                &metal.pipelines.mul_mm_q2_K_f32_old
            }
            Q3_K => {
                trace!("mul_mm_q3_K_f32_old");
                &metal.pipelines.mul_mm_q3_K_f32_old
            }
            Q4_K => {
                trace!("mul_mm_q4_K_f32_old");
                &metal.pipelines.mul_mm_q4_K_f32_old
            }
            Q5_K => {
                trace!("mul_mm_q5_K_f32_old");
                &metal.pipelines.mul_mm_q5_K_f32_old
            }
            Q6_K => {
                trace!("mul_mm_q6_K_f32_old");
                &metal.pipelines.mul_mm_q6_K_f32_old
            }
            t => unsupported_type!(t, matmul),
        };

        Self {
            pipeline,
            dst,
            src,
            gqa: (b_depth / a_depth) as u32,
        }
    }
}

impl<'a> Kernel<2> for MulMMOld<'a> {
    type Args<'arg> = [Arg<'arg>; 8]
    where
        Self: 'arg;

    fn pipeline(&self) -> &ComputePipelineState {
        self.pipeline
    }
    fn src(&self) -> &BinarySrc<'a> {
        &self.src
    }
    fn dst(&self) -> &TensorBuf<'a> {
        &self.dst
    }

    fn args(&self) -> Self::Args<'_> {
        let [a_cols, _, a_depth, _] = &self.src[A].tensor.dims;
        let [_, a_row_stride, a_depth_stride, _] = &self.src[A].tensor.stride;

        let [_, _, b_depth, _] = &self.src[B].tensor.dims;

        let [dst_cols, dst_rows, _, _] = &self.dst.tensor.dims;

        args![
            // A - dimensions
            a_cols,
            a_depth, //
            // A - stride
            a_row_stride,
            a_depth_stride, //
            // B - dimensions
            b_depth, //
            // dst - dimensions
            dst_cols,
            dst_rows, //
            // others
            &self.gqa,
        ]
    }

    fn threadgroup_mem(&self) -> Option<usize> {
        Some(8192)
    }

    fn threadgroups(&self) -> MTLSize {
        let [_, a_rows, _, _] = self.src[A].tensor.dims;
        let [_, b_rows, b_depth, _] = self.src[B].tensor.dims;
        let width = (b_rows as u64 + 31) / 32;
        let height = (a_rows as u64 + 63) / 64;
        MTLSize {
            width: (b_rows as u64 + 31) / 32,
            height: (a_rows as u64 + 63) / 64,
            depth: b_depth as u64,
        }
    }

    fn threads_per_threadgroup(&self) -> MTLSize {
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        }
    }
}

pub(crate) struct MulMVOld<'a> {
    pipeline: &'a ComputePipelineState,
    src: BinarySrc<'a>,
    dst: TensorBuf<'a>,

    gqa: usize,

    tg: MTLSize,
    nth0: usize,
    nth1: usize,
}

impl<'a> MulMVOld<'a> {
    pub fn new(metal: &'a Metal, dst: TensorBuf<'a>, src: BinarySrc<'a>) -> Self {
        let [a_d0, a_d1, a_d2, _] = src[A].tensor.dims;

        let [_, b_d1, b_d2, _] = src[B].tensor.dims;

        let mut nth0 = 32;
        let mut nth1 = 1;

        if src[B].tensor.typ() != F32 {
            unsupported_type!(src[B].tensor.typ(), matmul);
        }
        let (pipeline, tg) = match src[A].tensor.typ() {
            F16 => {
                let mut n_rows = 1;
                let pipeline = if b_d1 * b_d2 < 4 {
                    trace!("using mul_mv_f16_f32_1row_old");
                    &metal.pipelines.mul_mat_f16_f32_1row_old
                } else if a_d0 >= 128 && a_d1 >= 8 && a_d0 % 4 == 0 {
                    n_rows = b_d1;
                    trace!("using mul_mat_f16_f32_l4_old");
                    &metal.pipelines.mul_mat_f16_f32_l4_old
                } else {
                    trace!("using mul_mat_f16_f32_old");
                    n_rows = 4;
                    &metal.pipelines.mul_mat_f16_f32_old
                };
                let ny = (b_d1 + n_rows - 1) / n_rows;
                (
                    pipeline,
                    MTLSize {
                        width: a_d1 as u64,
                        height: ny as u64,
                        depth: b_d2 as u64,
                    },
                )
            }
            Q2_K | Q3_K | Q4_K | Q5_K | Q6_K => {
                nth0 = 2;
                nth1 = 32;
                let mut width = (a_d1 + 3) / 4;
                let pipeline = match src[A].tensor.typ() {
                    Q2_K => {
                        trace!("using mul_mat_q2_K_f32_old");
                        width = (a_d1 + 7) / 8;
                        &metal.pipelines.mul_mat_q2_K_f32_old
                    }
                    Q3_K => {
                        trace!("using mul_mat_q3_K_f32_old");
                        &metal.pipelines.mul_mat_q3_K_f32_old
                    }
                    Q4_K => {
                        trace!("using mul_mat_q4_K_f32_old");
                        nth0 = 4;
                        nth1 = 8;
                        &metal.pipelines.mul_mat_q4_K_f32_old
                    }
                    Q5_K => {
                        trace!("using mul_mat_q5_K_f32_old");
                        &metal.pipelines.mul_mat_q5_K_f32_old
                    }
                    Q6_K => {
                        trace!("using mul_mat_q6_K_f32_old");
                        width = (a_d1 + 1) / 2;
                        &metal.pipelines.mul_mat_q6_K_f32_old
                    }
                    _ => unreachable!(),
                };
                (
                    pipeline,
                    MTLSize {
                        width: width as u64,
                        height: b_d1 as u64,
                        depth: b_d2 as u64,
                    },
                )
            }
            t => unsupported_type!(t, matmul),
        };

        Self {
            pipeline,
            dst,
            src,

            gqa: b_d2 / a_d2,
            nth0,
            nth1,
            tg,
        }
    }
}

impl<'a> Kernel<2> for MulMVOld<'a> {
    type Args<'arg> = [Arg<'arg>; 15]
    where
        Self: 'arg;

    fn pipeline(&self) -> &ComputePipelineState {
        self.pipeline
    }
    fn src(&self) -> &BinarySrc<'a> {
        &self.src
    }
    fn dst(&self) -> &TensorBuf<'a> {
        &self.dst
    }

    fn args(&self) -> Self::Args<'_> {
        let [a_d0, a_d1, a_d2, _] = &self.src[A].tensor.dims;
        let [a_s0, a_s1, a_s2, _] = &self.src[A].tensor.stride;

        let [b_d0, b_d1, b_d2, _] = &self.src[B].tensor.dims;
        let [b_s0, b_s1, b_s2, _] = &self.src[B].tensor.stride;

        let [d0, d1, _, _] = &self.dst.tensor.dims;

        args![
            // A
            a_d0, a_d1, a_d2, //
            a_s0, a_s1, a_s2, //
            // B
            b_d0, b_d1, b_d2, //
            b_s0, b_s1, b_s2, //
            // dst
            d0, d1, //
            // others
            &self.gqa
        ]
    }

    fn threadgroup_mem(&self) -> Option<usize> {
        None
    }

    fn threadgroups(&self) -> MTLSize {
        self.tg
    }

    fn threads_per_threadgroup(&self) -> MTLSize {
        MTLSize {
            width: self.nth0 as u64,
            height: self.nth1 as u64,
            depth: 1,
        }
    }
}

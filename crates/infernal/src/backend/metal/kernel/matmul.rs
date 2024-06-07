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
pub(crate) fn use_mm_kernel(metal: &Metal, src: &BinarySrc) -> bool {
    let a = src[A].tensor;
    let b = src[B].tensor;

    // For now the matrix-matrix multiplication kernel only works on
    // A14+/M1+ SoCs AMD GPU and older A-chips will reuse matrix-vector
    // multiplication kernel.
    metal.device.supports_family(metal::MTLGPUFamily::Apple7)
        && !a.is_transposed()
        && !b.is_transposed()
        && b.typ() == F32
        && a.cols() % 32 == 0
        && a.cols() >= 64
        && (b.rows() > 1 || (a.typ().is_quantized() && b.depth() > 1))
}

// pub(crate) struct MulMM_Old<'a> {
//     pipeline: &'a ComputePipelineState,
//     src: BinarySrc<'a>,
//     dst: TensorBuf<'a>,

//     r2: usize,
//     r3: usize,
// }

pub(crate) struct MulMM<'a> {
    pipeline: &'a ComputePipelineState,
    src: BinarySrc<'a>,
    dst: TensorBuf<'a>,

    r2: usize,
    r3: usize,
}

impl<'a> MulMM<'a> {
    pub fn new(metal: &'a Metal, dst: TensorBuf<'a>, src: BinarySrc<'a>) -> Self {
        /*debug_*/
        assert_eq!(src[B].tensor.typ(), F32);
        let pipeline = match src[A].tensor.typ() {
            F32 => &metal.pipelines.mul_mm_f32_f32,
            F16 => &metal.pipelines.mul_mm_f16_f32,
            Q2_K => &metal.pipelines.mul_mm_q2_K_f32,
            Q3_K => &metal.pipelines.mul_mm_q3_K_f32,
            Q4_K => &metal.pipelines.mul_mm_q4_K_f32,
            Q5_K => &metal.pipelines.mul_mm_q5_K_f32,
            Q6_K => &metal.pipelines.mul_mm_q6_K_f32,
            t => unsupported_type!(t, matmul),
        };

        let a = src[A].tensor;
        let b = src[B].tensor;
        let r2 = b.depth() / a.depth();
        let r3 = b.batches() / a.batches();

        Self {
            pipeline,
            dst,
            src,
            r2,
            r3,
        }
    }
}

impl<'a> Kernel<2> for MulMM<'a> {
    type Args<'arg> = [Arg<'arg>; 12]
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
        let [a_d0, _, a_d2, _] = &self.src[A].tensor.dims;
        let [_, a_s1, a_s2, _] = &self.src[A].tensor.stride;

        let [_, _, b_d2, _] = &self.src[B].tensor.dims;
        let [b_s0, b_s1, b_s2, _] = &self.src[B].tensor.stride;

        let [d0, d1, _, _] = &self.dst.tensor.dims;

        // // TODO
        // args_v2! {
        //     a: {
        //         dims: d00, d02,
        //         stride: s01, s02,
        //     },
        //     y: {
        //         dims: d12, s10,
        //         stride: s11, s12,
        //     },
        //     dst: {
        //         dims: d0, d1,
        //         stride:
        //     },
        //     others: &self.r2, &self.r3,
        // }

        args![
            // A - dimensions
            a_d0, a_d2, //
            // A - stride
            a_s1, a_s2, //
            // B - dimensions
            b_d2, //
            // B - stride
            b_s0, b_s1, b_s2, //
            // dst - dimensions
            d0, d1, //
            // others
            &self.r2, &self.r3
        ]
    }

    fn threadgroup_mem(&self) -> Option<usize> {
        Some(8192)
    }

    fn threadgroups(&self) -> MTLSize {
        let [_, a_rows, _, _] = self.src[A].tensor.dims;
        let [_, b_rows, b_depth, b_batches] = self.src[B].tensor.dims;
        MTLSize {
            width: (b_rows as u64 + 31) / 32,
            height: (a_rows as u64 + 63) / 64,
            depth: (b_depth * b_batches) as u64,
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

pub(crate) struct MulMV<'a> {
    pipeline: &'a ComputePipelineState,
    src: BinarySrc<'a>,
    dst: TensorBuf<'a>,

    r2: usize,
    r3: usize,
    tg: MTLSize,
    nth0: usize,
    nth1: usize,
}

impl<'a> MulMV<'a> {
    pub fn new(metal: &'a Metal, dst: TensorBuf<'a>, src: BinarySrc<'a>) -> Self {
        // TODO: maybe clearer?
        // let a_dims = &src[A].tensor.dims;
        // let b_dims = &src[B].tensor.dims;

        // let r2 = b_dims[2] / a_dims[2];
        // let r3 = b_dims[3] / a_dims[3];

        let [a_d0, a_d1, a_d2, a_d3] = src[A].tensor.dims;

        let [_, b_d1, b_d2, b_d3] = src[B].tensor.dims;

        let mut nth0 = 32;
        let mut nth1 = 1;
        let (pipeline, tg) = match src[A].tensor.typ() {
            F32 => {
                trace!("using mul_mv_f32_f32");
                let n_rows = 4;
                let ny = (b_d1 + n_rows - 1) / n_rows;
                (
                    &metal.pipelines.mul_mv_f32_f32,
                    MTLSize {
                        width: a_d1 as u64,
                        height: ny as u64,
                        depth: (b_d2 * b_d3) as u64,
                    },
                )
            }
            F16 => {
                let mut n_rows = 1;
                let pipeline = if src[B].tensor.typ() == F32 {
                    if b_d1 * b_d2 < 4 {
                        trace!("using mul_mv_f16_f32_1row");
                        &metal.pipelines.mul_mv_f16_f32_1row
                    } else if a_d0 >= 128 && a_d1 >= 8 && a_d0 % 4 == 0 {
                        n_rows = b_d1;
                        trace!("using mul_mv_f16_f32_l4");
                        &metal.pipelines.mul_mv_f16_f32_l4
                    } else {
                        trace!("using mul_mv_f16_f32");
                        n_rows = 4;
                        &metal.pipelines.mul_mv_f16_f32
                    }
                } else if src[B].tensor.typ() == F16 {
                    n_rows = 4;
                    trace!("using mul_mv_f16_f16");
                    &metal.pipelines.mul_mv_f16_f16
                } else {
                    unsupported_type!(src[B].tensor.typ(), matmul);
                };
                let ny = (b_d1 + n_rows - 1) / n_rows;
                (
                    pipeline,
                    MTLSize {
                        width: a_d1 as u64,
                        height: ny as u64,
                        depth: (b_d2 * b_d3) as u64,
                    },
                )
            }
            Q2_K | Q3_K | Q4_K | Q5_K | Q6_K => {
                nth0 = 2;
                nth1 = 32;
                let mut width = (a_d1 + 3) / 4;
                let pipeline = match src[A].tensor.typ() {
                    Q2_K => {
                        trace!("using mul_mv_q2_K_f32");
                        width = (a_d1 + 7) / 8;
                        &metal.pipelines.mul_mv_q2_K_f32
                    }
                    Q3_K => {
                        trace!("using mul_mv_q3_K_f32");
                        &metal.pipelines.mul_mv_q3_K_f32
                    }
                    Q4_K => {
                        trace!("using mul_mv_q4_K_f32");
                        nth0 = 4;
                        nth1 = 8;
                        &metal.pipelines.mul_mv_q4_K_f32
                    }
                    Q5_K => {
                        trace!("using mul_mv_q5_K_f32");
                        &metal.pipelines.mul_mv_q5_K_f32
                    }
                    Q6_K => {
                        trace!("using mul_mv_q6_K_f32");
                        width = (a_d1 + 1) / 2;
                        &metal.pipelines.mul_mv_q6_K_f32
                    }
                    t => unsupported_type!(t, matmul),
                };
                (
                    pipeline,
                    MTLSize {
                        width: width as u64,
                        height: b_d1 as u64,
                        depth: (b_d2 * b_d3) as u64,
                    },
                )
            }
            t => unsupported_type!(t, matmul),
        };

        let r2 = b_d2 / a_d2;
        let r3 = b_d3 / a_d3;

        Self {
            pipeline,
            dst,
            src,

            r2,
            r3,
            nth0,
            nth1,
            tg,
        }
    }
}

impl<'a> Kernel<2> for MulMV<'a> {
    type Args<'arg> = [Arg<'arg>; 16]
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
            &self.r2, &self.r3,
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

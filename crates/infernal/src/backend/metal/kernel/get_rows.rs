use metal::{ComputePipelineState, MTLSize};

use crate::{
    args,
    backend::{
        metal::{
            kernel::{Arg, BinarySrc, Kernel, TensorBuf, A, B},
            Pipeline,
        },
        Metal,
    },
    unsupported_type, Type,
};

pub(crate) struct GetRows<'a> {
    pipeline: &'a Pipeline,
    src: BinarySrc<'a>,
    dst: TensorBuf<'a>,
}

impl<'a> GetRows<'a> {
    pub fn new(metal: &'a Metal, dst: TensorBuf<'a>, src: BinarySrc<'a>) -> Self {
        let pipeline = match src[A].tensor.typ() {
            Type::F32 => &metal.pipelines.get_rows_f32,
            Type::F16 => &metal.pipelines.get_rows_f16,
            Type::Q2_K => &metal.pipelines.get_rows_q2_K,
            Type::Q3_K => &metal.pipelines.get_rows_q3_K,
            Type::Q4_K => &metal.pipelines.get_rows_q4_K,
            Type::Q5_K => &metal.pipelines.get_rows_q5_K,
            Type::Q6_K => &metal.pipelines.get_rows_q6_K,
            t => unsupported_type!(t, matmul),
        };
        Self { pipeline, src, dst }
    }
}

impl<'a> Kernel<2> for GetRows<'a> {
    type Args<'arg> = [Arg<'arg>; 8]
    where
        Self: 'arg;

    fn pipeline(&self) -> &Pipeline {
        self.pipeline
    }
    fn src(&self) -> &BinarySrc<'a> {
        &self.src
    }
    fn dst(&self) -> &TensorBuf<'a> {
        &self.dst
    }

    fn args(&self) -> Self::Args<'_> {
        let [a_d0, ..] = &self.src[A].tensor.dims;
        let [_, a_s1, a_s2, _] = &self.src[A].tensor.stride;

        let [b_d0, ..] = &self.src[B].tensor.dims;
        let [b_s0, b_s1, ..] = &self.src[B].tensor.stride;

        let [_, s1, s2, _] = &self.dst.tensor.stride;

        args![
            // A
            a_d0, //
            a_s1, a_s2, //
            // B
            b_d0, //
            b_s0, b_s1, //
            // dst
            s1, s2,
        ]
    }

    fn threadgroup_mem(&self) -> Option<usize> {
        None
    }

    fn threadgroups(&self) -> MTLSize {
        let [b_d0, b_d1, ..] = self.src[B].tensor.dims;
        MTLSize {
            width: b_d0 as u64,
            height: b_d1 as u64,
            depth: 1,
        }
    }

    fn threads_per_threadgroup(&self) -> MTLSize {
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        }
    }
}

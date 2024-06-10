use metal::{ComputePipelineState, MTLSize};

use crate::{
    args,
    backend::metal::{
        kernel::{Arg, Kernel, TensorBuf, UnarySrc, A},
        Metal,
    },
    unsupported_type, Type,
};

pub(crate) struct Copy<'a> {
    pipeline: &'a ComputePipelineState,
    dst: TensorBuf<'a>,
    src: UnarySrc<'a>,
}

impl<'a> Copy<'a> {
    pub fn new(metal: &'a Metal, dst: TensorBuf<'a>, src: UnarySrc<'a>) -> Self {
        let pipeline = match src[A].tensor.typ() {
            Type::F32 => match dst.tensor.typ() {
                Type::F32 => &metal.pipelines.cpy_f32_f32,
                Type::F16 => &metal.pipelines.cpy_f32_f16,
                t => unsupported_type!(t, copy),
            },
            Type::F16 => match dst.tensor.typ() {
                Type::F32 => unimplemented!("copy from f16 to f32 not supported"),
                Type::F16 => &metal.pipelines.cpy_f16_f16,
                t => unsupported_type!(t, copy),
            },
            t => unsupported_type!(t, copy),
        };
        Self { pipeline, dst, src }
    }
}

impl<'a> Kernel<1> for Copy<'a> {
    type Args<'arg> = [Arg<'arg>; 16]
    where
        Self: 'arg;

    fn pipeline(&self) -> &ComputePipelineState {
        self.pipeline
    }
    fn src(&self) -> &UnarySrc<'a> {
        &self.src
    }
    fn dst(&self) -> &TensorBuf<'a> {
        &self.dst
    }

    fn args(&self) -> Self::Args<'_> {
        let [d00, d01, d02, d03] = &self.src[A].tensor.dims;
        let [s00, s01, s02, s03] = &self.src[A].tensor.stride;

        let [d0, d1, d2, d3] = &self.dst.tensor.dims;
        let [s0, s1, s2, s3] = &self.dst.tensor.stride;

        args![
            // X
            d00, d01, d02, d03, //
            s00, s01, s02, s03, //
            // dst
            d0, d1, d2, d3, //
            s0, s1, s2, s3, //
        ]
    }

    fn threadgroup_mem(&self) -> Option<usize> {
        None
    }

    fn threadgroups(&self) -> MTLSize {
        let [_, d01, d02, d03] = self.src[A].tensor.dims;
        MTLSize {
            width: d01 as u64,
            height: d02 as u64,
            depth: d03 as u64,
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

use metal::{ComputePipelineState, MTLSize};

use crate::{
    args,
    backend::{
        metal::kernel::{Arg, Kernel, TensorBuf, UnarySrc, A},
        Metal,
    },
    op,
};

pub(crate) struct RopeOld<'a> {
    pipeline: &'a ComputePipelineState,
    src: UnarySrc<'a>,
    dst: TensorBuf<'a>,
    args: op::RopeExpanded,
}

impl<'a> RopeOld<'a> {
    pub fn new(
        metal: &'a Metal,
        dst: TensorBuf<'a>,
        src: UnarySrc<'a>,
        args: op::RopeExpanded,
    ) -> Self {
        // let pipeline = match src[A].tensor.typ() {
        //     Type::F32 => &metal.pipelines.rope_f32,
        //     Type::F16 => &metal.pipelines.rope_f16,
        // };
        // TODO: switch to the new kernel if we need f16 support
        let pipeline = &metal.pipelines.rope_old;
        Self {
            pipeline,
            dst,
            src,
            args,
        }
    }
}

impl<'a> Kernel<1> for RopeOld<'a> {
    type Args<'arg> = [Arg<'arg>; 21]
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
        let [d00, d01, d02, d03] = &self.src[A].tensor.layout.dims;
        let [s00, s01, s02, s03] = &self.src[A].tensor.layout.stride;

        let [d0, d1, d2, d3] = &self.dst.tensor.layout.dims;
        let [s0, s1, s2, s3] = &self.dst.tensor.layout.stride;

        let op::RopeCustom {
            freq_base,
            freq_scale,
        } = &self.args.custom;

        args![
            // x - dimensions
            d00,
            d01,
            d02,
            d03,
            // x - stride
            s00,
            s01,
            s02,
            s03,
            // dst - dimensions
            d0,
            d1,
            d2,
            d3,
            // dst - stride
            s0,
            s1,
            s2,
            s3,
            // others: the kernel takes these as i32 for some reason
            &self.args.n_past,
            &self.args.n_dims,
            &self.args.mode,
            // customized args
            freq_base,
            freq_scale,
        ]
    }

    fn threadgroup_mem(&self) -> Option<usize> {
        None
    }

    fn threadgroups(&self) -> MTLSize {
        let [_, d01, d02, d03] = self.src[A].tensor.layout.dims;
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

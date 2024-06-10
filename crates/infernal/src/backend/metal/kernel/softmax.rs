use metal::{ComputePipelineState, MTLSize};

use crate::{
    args,
    backend::{
        metal::kernel::{Arg, BinarySrc, Kernel, TensorBuf, A},
        Metal,
    },
};

pub(crate) struct Softmax<'a> {
    pipeline: &'a ComputePipelineState,
    src: BinarySrc<'a>,
    dst: TensorBuf<'a>,
    scale: f32,
    nth: usize,
}

impl<'a> Softmax<'a> {
    pub fn new(metal: &'a Metal, dst: TensorBuf<'a>, src: BinarySrc<'a>, scale: f32) -> Self {
        let a = src[A].tensor;

        let mut nth = 32;
        let pipeline = if a.cols() % 4 == 0 {
            let cols = a.cols() / 4;
            while nth < cols && nth < 256 {
                nth *= 2;
            }
            &metal.pipelines.soft_max_4
        } else {
            while nth < a.cols() && nth < 1024 {
                nth *= 2;
            }
            &metal.pipelines.soft_max
        };

        Self {
            pipeline,
            dst,
            src,
            scale,
            nth,
        }
    }
}

impl<'a> Kernel<2> for Softmax<'a> {
    type Args<'arg> = [Arg<'arg>; 4]
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
        let [d00, d01, d02, _] = &self.src[A].tensor.dims;

        args![
            // X - dimensions
            d00,
            d01,
            d02, //
            // other
            &self.scale,
        ]
    }

    fn threadgroup_mem(&self) -> Option<usize> {
        Some(32 * std::mem::size_of::<f32>())
    }

    fn threadgroups(&self) -> MTLSize {
        let [_, d01, d02, d03] = self.src[A].tensor.dims;
        MTLSize {
            width: (d01 * d02 * d03) as u64,
            height: 1,
            depth: 1,
        }
    }

    fn threads_per_threadgroup(&self) -> MTLSize {
        MTLSize {
            width: self.nth as u64,
            height: 1,
            depth: 1,
        }
    }
}

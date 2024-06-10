use metal::{ComputePipelineState, MTLSize};

use crate::{
    args,
    backend::{
        metal::kernel::{Arg, Kernel, TensorBuf, UnarySrc, A},
        Metal,
    },
};

pub(crate) struct Scale<'a> {
    pipeline: &'a ComputePipelineState,
    // NB: Scale is a binary operation but the second operand has already been extracted in the form
    // of scale.
    src: UnarySrc<'a>,
    dst: TensorBuf<'a>,
    scale: f32,
    n: usize,
}

impl<'a> Scale<'a> {
    pub fn new(metal: &'a Metal, dst: TensorBuf<'a>, src: UnarySrc<'a>, scale: f32) -> Self {
        debug_assert!(src[A].tensor.is_contiguous());

        let mut n = dst.tensor.total_elems();
        let pipeline = if n % 4 == 0 {
            n /= 4;
            &metal.pipelines.scale_4
        } else {
            &metal.pipelines.scale
        };
        Self {
            pipeline,
            src,
            dst,
            scale,
            n,
        }
    }
}

impl<'a> Kernel<1> for Scale<'a> {
    type Args<'arg> = [Arg<'arg>; 1]
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
        args![&self.scale]
    }

    fn threadgroup_mem(&self) -> Option<usize> {
        None
    }

    fn threadgroups(&self) -> MTLSize {
        MTLSize {
            width: self.n as u64,
            height: 1,
            depth: 1,
        }
    }

    fn threads_per_threadgroup(&self) -> MTLSize {
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        }
    }
}

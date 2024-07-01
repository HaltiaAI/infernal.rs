use metal::{ComputePipelineState, MTLSize};

use crate::backend::{
    metal::kernel::{Arg, Kernel, TensorBuf, UnarySrc},
    Metal,
};

pub(crate) struct Silu<'a> {
    pipeline: &'a crate::backend::metal::Pipeline,
    src: UnarySrc<'a>,
    dst: TensorBuf<'a>,
}

impl<'a> Silu<'a> {
    pub fn new(metal: &'a Metal, dst: TensorBuf<'a>, src: UnarySrc<'a>) -> Self {
        Self {
            pipeline: &metal.pipelines.silu,
            dst,
            src,
        }
    }
}

impl<'a> Kernel<1> for Silu<'a> {
    type Args<'arg> = [Arg<'arg>; 0]
    where
        Self: 'arg;

    fn pipeline(&self) -> &crate::backend::metal::Pipeline {
        self.pipeline
    }
    fn src(&self) -> &UnarySrc<'a> {
        &self.src
    }
    fn dst(&self) -> &TensorBuf<'a> {
        &self.dst
    }

    fn args(&self) -> Self::Args<'_> {
        []
    }

    fn threadgroup_mem(&self) -> Option<usize> {
        None
    }

    fn threadgroups(&self) -> MTLSize {
        let n = self.dst.tensor.total_elems();
        debug_assert_eq!(n % 4, 0);
        MTLSize {
            width: (n / 4) as u64,
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

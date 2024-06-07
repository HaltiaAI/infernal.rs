use metal::{ComputePipelineState, MTLSize};

use crate::backend::{
    metal::kernel::{Arg, Kernel, TensorBuf, UnarySrc},
    Metal,
};

pub(crate) struct Relu<'a> {
    pipeline: &'a ComputePipelineState,
    src: UnarySrc<'a>,
    dst: TensorBuf<'a>,
}

impl<'a> Relu<'a> {
    pub fn new(metal: &'a Metal, dst: TensorBuf<'a>, src: UnarySrc<'a>) -> Self {
        Self {
            pipeline: &metal.pipelines.relu,
            dst,
            src,
        }
    }
}

impl<'a> Kernel<1> for Relu<'a> {
    type Args<'arg> = [Arg<'arg>; 0]
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
        []
    }

    fn threadgroup_mem(&self) -> Option<usize> {
        None
    }

    fn threadgroups(&self) -> MTLSize {
        let n = self.dst.tensor.total_elems();
        MTLSize {
            width: n as u64,
            height: 1,
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

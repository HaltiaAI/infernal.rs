use std::mem;

use metal::{ComputePipelineState, MTLSize};

use crate::{
    args,
    backend::{
        metal::kernel::{Arg, Kernel, TensorBuf, UnarySrc, A},
        Metal,
    },
};

pub(crate) struct RmsNorm<'a> {
    pipeline: &'a ComputePipelineState,
    src: UnarySrc<'a>,
    dst: TensorBuf<'a>,
    eps: f32,
    nth: usize,
}

impl<'a> RmsNorm<'a> {
    pub fn new(metal: &'a Metal, dst: TensorBuf<'a>, src: UnarySrc<'a>, eps: f32) -> Self {
        let a = src[A].tensor;

        debug_assert_eq!(a.cols() % 4, 0);

        let mut nth = 32;
        while nth < a.cols() && nth < 1024 {
            nth *= 2;
        }
        // TODO: Currently checked out ggml uses nth=512 while we go up to 1024. This seems to cause
        // a tiny difference in results if the float values in the tensor are small enough.

        Self {
            pipeline: &metal.pipelines.rms_norm,
            dst,
            src,
            eps,
            nth,
        }
    }
}

impl<'a> Kernel<1> for RmsNorm<'a> {
    type Args<'arg> = [Arg<'arg>; 3]
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
        let a = &self.src[A].tensor;

        let [a_d0, ..] = &a.layout.dims;
        let [_, a_s1, ..] = &a.layout.stride;

        args![
            // A - dimensions
            a_d0, //
            // A - stride
            a_s1, //
            // others:
            &self.eps,
        ]
    }

    fn threadgroup_mem(&self) -> Option<usize> {
        Some(32 * mem::size_of::<f32>())
    }

    fn threadgroups(&self) -> MTLSize {
        let n_rows = self.src[A].tensor.total_rows();
        MTLSize {
            width: n_rows as u64,
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

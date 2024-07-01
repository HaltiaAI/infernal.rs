use metal::{ComputePipelineState, MTLSize};

use crate::{
    args,
    backend::{
        metal::{
            kernel::{Arg, Kernel, TensorBuf, UnarySrc, A},
            Pipeline,
        },
        Metal,
    },
};

pub(crate) struct DiagMaskInf<'a> {
    pipeline: &'a Pipeline,
    src: UnarySrc<'a>,
    dst: TensorBuf<'a>,
    n_past: usize,
    is_8_mult: bool,
}

impl<'a> DiagMaskInf<'a> {
    pub fn new(metal: &'a Metal, dst: TensorBuf<'a>, src: UnarySrc<'a>, n_past: usize) -> Self {
        let is_8_mult = src[A].tensor.cols() % 8 == 0;
        let pipeline = if is_8_mult {
            &metal.pipelines.diag_mask_inf_8
        } else {
            &metal.pipelines.diag_mask_inf
        };
        Self {
            pipeline,
            dst,
            src,
            n_past,
            is_8_mult,
        }
    }
}

impl<'a> Kernel<1> for DiagMaskInf<'a> {
    type Args<'arg> = [Arg<'arg>; 3]
    where
        Self: 'arg;

    fn pipeline(&self) -> &Pipeline {
        self.pipeline
    }
    fn src(&self) -> &UnarySrc<'a> {
        &self.src
    }
    fn dst(&self) -> &TensorBuf<'a> {
        &self.dst
    }

    fn args(&self) -> Self::Args<'_> {
        let [d00, d01, ..] = &self.src[A].tensor.layout.dims;

        args![
            // A
            d00,
            d01, //
            // other
            &self.n_past,
        ]
    }

    fn threadgroup_mem(&self) -> Option<usize> {
        None
    }

    fn threadgroups(&self) -> MTLSize {
        let [d00, d01, d02, _] = self.src[A].tensor.layout.dims;
        if self.is_8_mult {
            MTLSize {
                width: (d00 * d01 * d02 / 8) as u64,
                height: 1,
                depth: 1,
            }
        } else {
            MTLSize {
                width: d00 as u64,
                height: d01 as u64,
                depth: d02 as u64,
            }
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

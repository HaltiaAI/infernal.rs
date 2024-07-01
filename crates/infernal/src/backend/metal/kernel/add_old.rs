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
};

pub(crate) struct AddOld<'a> {
    pipeline: &'a Pipeline,
    dst: TensorBuf<'a>,
    src: BinarySrc<'a>,
    stride: u64,
}

impl<'a> AddOld<'a> {
    pub fn new(metal: &'a Metal, dst: TensorBuf<'a>, src: BinarySrc<'a>) -> Self {
        let a = src[A].tensor;
        let b = src[B].tensor;
        debug_assert!(b.is_contiguous());
        debug_assert!(a.cols() % 4 == 0);

        let stride = a.cols() / 4;
        let bcast = b.total_elems() == b.cols();
        let pipeline = if bcast {
            &metal.pipelines.add_row_old
        } else {
            &metal.pipelines.add_old
        };

        Self {
            pipeline,
            dst,
            src,
            stride: stride as u64,
        }
    }
}

impl<'a> Kernel<2> for AddOld<'a> {
    type Args<'arg> = [Arg<'arg>; 1]
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
        args![&self.stride]
    }

    fn threadgroup_mem(&self) -> Option<usize> {
        None
    }

    fn threadgroups(&self) -> MTLSize {
        let n = self.dst.tensor.total_elems() / 4;
        MTLSize {
            width: n as u64,
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

use metal::MTLSize;

use crate::{
    args,
    backend::metal::{
        kernel::{Arg, BinarySrc, Kernel, TensorBuf, A, B},
        Pipeline,
    },
};

pub(crate) struct BinArith<'a> {
    pipeline: &'a Pipeline,
    dst: TensorBuf<'a>,
    src: BinarySrc<'a>,
    stride: u64,
    offs: usize,
    broadcast_row: bool,
}

impl<'a> BinArith<'a> {
    pub fn new(
        pipeline: &'a Pipeline,
        pipeline_row: &'a Pipeline,
        dst: TensorBuf<'a>,
        src: BinarySrc<'a>,
    ) -> Self {
        let a = src[A].tensor;
        let b = src[B].tensor;

        let broadcast_row;
        let stride;
        let pipeline = if b.total_elems() == b.cols()
            && b.is_contiguous()
            && a.cols() % 4 == 0
            && b.cols() % 4 == 0
        {
            // TODO: should this be in the condition?
            assert!(a.is_contiguous());
            assert_eq!(b.rows(), 1);
            stride = a.cols() / 4;
            broadcast_row = true;
            pipeline_row
        } else {
            stride = a.cols();
            broadcast_row = false;
            pipeline
        };

        // let mut stride = src[A].tensor.dims[0];
        // let is_row_4 = |t: &crate::Tensor| t.n() == 1 && t.is_contiguous() && t.dims[0] % 4 == 0;

        // let pipeline = if is_row_4(src[A].tensor) && is_row_4(src[B].tensor) {
        //     stride /= 4;
        //     pipeline_row
        // } else {
        //     pipeline
        // };

        Self {
            pipeline,
            dst,
            src,
            stride: stride as u64,
            offs: 0,
            broadcast_row,
        }
    }
}

impl<'a> Kernel<2> for BinArith<'a> {
    type Args<'arg> = [Arg<'arg>; 26]
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
        let [a_cols, a_rows, a_depth, a_batches] = &self.src[A].tensor.dims;
        let [a_col_stride, a_row_stride, a_depth_stride, a_batch_stride] =
            &self.src[A].tensor.stride;

        let [b_d0, b_d1, b_d2, b_d3] = &self.src[B].tensor.dims;
        let [b_s0, b_s1, b_s2, b_s3] = &self.src[B].tensor.stride;

        let [d0, d1, d2, d3] = &self.dst.tensor.dims;
        let [s0, s1, s2, s3] = &self.dst.tensor.stride;

        args![
            // A - dims
            a_cols,
            a_rows,
            a_depth,
            a_batches, //
            // A - stride
            a_col_stride,
            a_row_stride,
            a_depth_stride,
            a_batch_stride, //
            // B - dims
            b_d0,
            b_d1,
            b_d2,
            b_d3, //
            // B - stride
            b_s0,
            b_s1,
            b_s2,
            b_s3, //
            // dst - dims
            d0,
            d1,
            d2,
            d3, //
            // dst - stride
            s0,
            s1,
            s2,
            s3, //
            // others
            &self.offs,
            &self.stride,
        ]
    }

    fn threadgroup_mem(&self) -> Option<usize> {
        None
    }

    fn threadgroups(&self) -> MTLSize {
        if self.broadcast_row {
            let n = self.dst.tensor.total_elems() / 4;
            MTLSize {
                width: n as u64,
                height: 1,
                depth: 1,
            }
        } else {
            let [_, a_rows, a_depth, a_batches] = self.src[A].tensor.dims;
            MTLSize {
                width: a_rows as u64,
                height: a_depth as u64,
                depth: a_batches as u64,
            }
        }
    }

    fn threads_per_threadgroup(&self) -> MTLSize {
        if self.broadcast_row {
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            }
        } else {
            let [dst_cols, ..] = self.dst.tensor.dims;
            let n = self
                .pipeline
                .state
                .max_total_threads_per_threadgroup()
                .min(dst_cols as u64);
            MTLSize {
                width: n,
                height: 1,
                depth: 1,
            }
        }
    }
}

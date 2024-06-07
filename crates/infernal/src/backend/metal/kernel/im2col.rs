use metal::{ComputePipelineState, MTLSize};

use crate::{
    args,
    backend::{
        metal::kernel::{Arg, BinarySrc, Kernel, TensorBuf, A, B},
        Metal,
    },
    op, unsupported_type, Type,
};

pub(crate) struct Im2Col<'a> {
    pipeline: &'a ComputePipelineState,
    src: BinarySrc<'a>,
    dst: TensorBuf<'a>,
    args: op::Im2Col,
    vals: Values,
}

impl<'a> Im2Col<'a> {
    pub fn new(metal: &'a Metal, dst: TensorBuf<'a>, src: BinarySrc<'a>, args: op::Im2Col) -> Self {
        let pipeline = match src[A].tensor.typ() {
            Type::F32 => unimplemented!("im2col f32 not supported"),
            Type::F16 => &metal.pipelines.im2col_f16,
            t => unsupported_type!(t, im2col),
        };

        // TODO: verify
        let is_2d = args.dilation1 == 1;
        let values = if is_2d {
            Values::for_2d(&dst, &src)
        } else {
            Values::for_1d(&dst, &src)
        };

        Self {
            pipeline,
            dst,
            src,
            args,
            vals: values,
        }
    }
}

struct Values {
    n: i32,
    ic: i32,
    ih: i32,

    oh: i32,

    offs0: i32,
    offs1: i32,

    common: CommonValues,
}

/// Values common to both 1D and 2D operations.
struct CommonValues {
    iw: i32,
    kw: i32,
    ow: i32,
    chw: i32,
}

// This is dogshit.
impl Values {
    fn for_1d(dst: &TensorBuf, src: &BinarySrc) -> Self {
        let n = src[B].tensor.dims[2];
        let ic = src[B].tensor.dims[1];
        let ih = 1;

        let kh = 1;

        let oh = 1;

        let offs0 = src[B].tensor.dims[2] / 4;
        let offs1 = src[B].tensor.dims[1] / 4;

        Self {
            n: n as i32,
            ic: ic as i32,
            ih,
            oh,
            offs0: offs0 as i32,
            offs1: offs1 as i32,
            common: CommonValues::new(dst, src, ic, kh),
        }
    }

    fn for_2d(dst: &TensorBuf, src: &BinarySrc) -> Self {
        let n = src[B].tensor.dims[3];
        let ic = src[B].tensor.dims[2];
        let ih = src[B].tensor.dims[1];

        let kh = src[A].tensor.dims[1];

        let oh = dst.tensor.dims[2];

        let offs0 = src[B].tensor.dims[3] / 4;
        let offs1 = src[B].tensor.dims[2] / 4;

        Self {
            n: n as i32,
            ic: ic as i32,
            ih: ih as i32,
            oh: oh as i32,
            offs0: offs0 as i32,
            offs1: offs1 as i32,
            common: CommonValues::new(dst, src, ic, kh),
        }
    }
}

impl CommonValues {
    fn new(dst: &TensorBuf, src: &BinarySrc, ic: usize, kh: usize) -> Self {
        let iw = src[B].tensor.dims[0];
        let kw = src[A].tensor.dims[0];
        let ow = dst.tensor.dims[1];
        let chw = ic * kh * kw;

        Self {
            iw: iw as i32,
            kw: kw as i32,
            ow: ow as i32,
            chw: chw as i32,
        }
    }
}

impl<'a> Kernel<2> for Im2Col<'a> {
    type Args<'arg> = [Arg<'arg>; 11]
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
        args![
            // vals
            &self.vals.offs0,
            &self.vals.offs1,
            &self.vals.common.iw,
            &self.vals.ih,
            &self.vals.common.chw,
            // args
            &self.args.stride0,
            &self.args.stride1,
            &self.args.padding0,
            &self.args.padding1,
            &self.args.dilation0,
            &self.args.dilation1,
        ]
    }

    fn threadgroup_mem(&self) -> Option<usize> {
        None
    }

    fn threadgroups(&self) -> MTLSize {
        MTLSize {
            width: self.vals.ic as u64,
            height: self.vals.oh as u64,
            depth: self.vals.common.ow as u64,
        }
    }

    fn threads_per_threadgroup(&self) -> MTLSize {
        MTLSize {
            width: self.vals.n as u64,
            height: self.vals.oh as u64,
            depth: self.vals.common.kw as u64,
        }
    }
}

// const bool is_2D = ((const int32_t *)(dst->op_params))[6] == 1;

// const int32_t N = src1->ne[is_2D ? 3 : 2];
// const int32_t IC = src1->ne[is_2D ? 2 : 1];
// const int32_t IH = is_2D ? src1->ne[1] : 1;
// const int32_t IW = src1->ne[0];

// const int32_t KH = is_2D ? src0->ne[1] : 1;
// const int32_t KW = src0->ne[0];

// const int32_t OH = is_2D ? dst->ne[2] : 1;
// const int32_t OW = dst->ne[1];

// const int32_t CHW = IC * KH * KW;

// const int32_t ofs0 = src1->nb[is_2D ? 3 : 2] / 4;
// const int32_t ofs1 = src1->nb[is_2D ? 2 : 1] / 4;

// switch (src0->type) {
// case GGML_TYPE_F32:
//   GGML_ASSERT(false && "not implemented");
//   break;
// case GGML_TYPE_F16:
//   [encoder setComputePipelineState:ctx->pipeline_im2col_f16];
//   break;
// default:
//   GGML_ASSERT(false);
// };

// [encoder setBuffer:id_src1 offset:offs_src1 atIndex:0];
// [encoder setBuffer:id_dst offset:offs_dst atIndex:1];
//
// [encoder setBytes:&ofs0 length:sizeof(int32_t) atIndex:2];
// [encoder setBytes:&ofs1 length:sizeof(int32_t) atIndex:3];
// [encoder setBytes:&IW length:sizeof(int32_t) atIndex:4];
// [encoder setBytes:&IH length:sizeof(int32_t) atIndex:5];
// [encoder setBytes:&CHW length:sizeof(int32_t) atIndex:6];
// [encoder setBytes:&s0 length:sizeof(int32_t) atIndex:7];
// [encoder setBytes:&s1 length:sizeof(int32_t) atIndex:8];
// [encoder setBytes:&p0 length:sizeof(int32_t) atIndex:9];
// [encoder setBytes:&p1 length:sizeof(int32_t) atIndex:10];
// [encoder setBytes:&d0 length:sizeof(int32_t) atIndex:11];
// [encoder setBytes:&d1 length:sizeof(int32_t) atIndex:12];

// [encoder dispatchThreadgroups:MTLSizeMake(IC, OH, OW)
//         threadsPerThreadgroup:MTLSizeMake(N, KH, KW)];

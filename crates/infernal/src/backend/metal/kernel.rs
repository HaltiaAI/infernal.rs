use std::mem;

use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};

use crate::tensor::Tensor;

pub(crate) use add_old::AddOld;
mod add_old;
pub(crate) use bin_arith::BinArith;
mod bin_arith;
pub(crate) use copy::Copy;
mod copy;
pub(crate) use diag_mask_inf::DiagMaskInf;
mod diag_mask_inf;
pub(crate) use im2col::Im2Col;
mod im2col;
pub(crate) use get_rows::GetRows;
mod get_rows;
pub(crate) use matmul::{use_mm_kernel, MulMM, MulMV};
mod matmul;
pub(crate) use matmul_old::{use_mm_kernel_old, MulMMOld, MulMVOld};
mod matmul_old;
pub(crate) use mul_old::MulOld;
mod mul_old;
pub(crate) use relu::Relu;
mod relu;
pub(crate) use rms_norm::RmsNorm;
mod rms_norm;
pub(crate) use rope_old::RopeOld;
mod rope_old;
pub(crate) use scale::Scale;
mod scale;
pub(crate) use silu::Silu;
mod silu;
pub(crate) use softmax::Softmax;
mod softmax;

// Mnemonics for the source buffer indices. They are always in this order.
pub(crate) const A: usize = 0;
pub(crate) const B: usize = 1;
// pub(crate) const C: usize = 2;

pub(crate) type UnarySrc<'a> = Src<'a, 1>;
pub(crate) type BinarySrc<'a> = Src<'a, 2>;
pub(crate) type Src<'a, const N: usize> = [TensorBuf<'a>; N];

#[derive(Clone, Copy)]
pub(crate) struct TensorBuf<'a> {
    pub tensor: &'a Tensor,
    pub buf: &'a Buffer,
}

impl<'a> From<(&'a Tensor, &'a Buffer)> for TensorBuf<'a> {
    fn from((tensor, buf): (&'a Tensor, &'a Buffer)) -> Self {
        Self { tensor, buf }
    }
}

// TODO: DRY up further
// pub(crate) struct KernelImpl<'a, K: KernelOp, const N: usize> {
//     pipeline: &'a ComputePipelineState,
//     src: Src<'a, N>,
//     dst: TensorBuf<'a>,
//     kernel_op: K,
// }

// pub(crate) trait KernelOp {
//     type Iter<'arg>: Iterator<Item = &'arg Arg<'arg>>
//     where
//         Self: 'arg;

//     fn pipeline(&self) -> &ComputePipelineState;
//     fn args<'this>(&'this self) -> Self::Iter<'this>;
//     fn threads_per_threadgroup(&self) -> MTLSize;
// }

pub(crate) trait Kernel<const N_SRC: usize> {
    type Args<'arg>: IntoIterator<Item = Arg<'arg>>
    where
        Self: 'arg;

    fn encode(&self, encoder: &ComputeCommandEncoderRef) {
        encoder.set_compute_pipeline_state(self.pipeline());

        // Encode source buffers first, then the destination buffer. Always in this order.
        for (i, src) in self.src().iter().enumerate() {
            encoder.set_buffer(i as u64, Some(src.buf), src.tensor.data_offs as u64);
        }
        encoder.set_buffer(
            N_SRC as u64,
            Some(self.dst().buf),
            self.dst().tensor.data_offs as u64,
        );

        for (i, arg) in self.args().into_iter().enumerate() {
            // Account for the first n buffers encoded.
            let i = i + N_SRC + 1;
            encoder.set_bytes(i as u64, arg.size, arg.ptr);
        }

        if let Some(size) = self.threadgroup_mem() {
            // This is always encoded as the zeroth index. Not sure how this works given that the
            // first buffer is also encoded at the same index. Maybe different namespaces.
            encoder.set_threadgroup_memory_length(0, size as u64);
        };
        encoder.dispatch_thread_groups(self.threadgroups(), self.threads_per_threadgroup());
    }

    fn pipeline(&self) -> &ComputePipelineState;
    fn src(&self) -> &Src<'_, N_SRC>;
    fn dst(&self) -> &TensorBuf<'_>;
    fn args(&self) -> Self::Args<'_>;
    fn threadgroup_mem(&self) -> Option<usize>;
    fn threadgroups(&self) -> MTLSize;
    fn threads_per_threadgroup(&self) -> MTLSize;
}

#[macro_export]
macro_rules! args {
    [$($arg:expr),* $(,)?] => {
        [$(Arg::new($arg)),*]
    };
}

// Helper type to represent arbitrary arguments to be encoded into Metal.
//
// Note that it is safe to pass a raw pointer over the FFI layer because the lifetime of the pointer
// is constrained by the original lifetime of the reference, and this type is not used anywhere
// else, hence there is no chance of misusing the underlying pointer.
pub(crate) struct Arg<'a> {
    size: u64,
    ptr: *const core::ffi::c_void,
    _ref: std::marker::PhantomData<&'a ()>,
}

impl<'a> Arg<'a> {
    fn new<T>(val: &'a T) -> Self {
        Self {
            size: mem::size_of::<T>() as u64,
            ptr: val as *const T as *const core::ffi::c_void,
            _ref: Default::default(),
        }
    }
}

#[macro_export]
macro_rules! unsupported_type {
    ($ty:expr, $kernel:ident) => {
        unreachable!(
            concat!(
                "Unsupported type: {}; crate::op::TensorBuilder::",
                stringify!($kernel),
                " needs to be updated"
            ),
            $ty.name()
        )
    };
}

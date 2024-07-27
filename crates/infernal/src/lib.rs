use std::collections::HashMap;

use thiserror::Error;

use crate::{op::Op, tensor::CtxId};

pub use crate::{
    alloc::{BumpAlloc, GraphAlloc, MmapAlloc},
    backend::Backend,
    buf::{Buf, BufId},
    builder::{GraphBuilder, TensorBuilder},
    graph::{ComputeGraph, ConcurOptimizer, ExecPlan, Optimizer},
    layout::{Dims, Layout, Type, D1, D2, D3, D4},
    op::{Dup, Inplace},
    tensor::{Tensor, TensorCtx, TensorId},
};

pub mod alloc;
pub mod backend;
pub mod buf;
pub mod builder;
pub mod graph;
pub mod layout;
pub mod op;
pub mod quant;
pub mod tensor;
#[cfg(test)]
pub(crate) mod test_utils;
pub mod utils;

#[derive(Clone, Copy, Debug, Error)]
#[error("tensor {0} not found")]
pub struct TensorNotFound(pub TensorId);

pub trait GetCtxId {
    fn id(&self) -> CtxId;
}

/// Anything that stores the underlying tensor data.
///
/// Usually either a [`TensorBuilder`], a single [`TensorCtx`], or a list of such contexts or
/// builders.
pub trait TensorSrc {
    fn get_tensor(&self, id: TensorId) -> Result<&Tensor, TensorNotFound>;
}

impl<S1: TensorSrc, S2: TensorSrc> TensorSrc for (S1, S2) {
    #[inline(always)]
    fn get_tensor(&self, id: TensorId) -> Result<&Tensor, TensorNotFound> {
        let (s1, s2) = self;
        s1.get_tensor(id).or_else(|_| s2.get_tensor(id))
    }
}

impl<S1: TensorSrc, S2: TensorSrc, S3: TensorSrc> TensorSrc for (S1, S2, S3) {
    #[inline(always)]
    fn get_tensor(&self, id: TensorId) -> Result<&Tensor, TensorNotFound> {
        let (s1, s2, s3) = self;
        s1.get_tensor(id)
            .or_else(|_| s2.get_tensor(id))
            .or_else(|_| s3.get_tensor(id))
    }
}

impl<S: TensorSrc> TensorSrc for HashMap<CtxId, S> {
    #[inline(always)]
    fn get_tensor(&self, id: TensorId) -> Result<&Tensor, TensorNotFound> {
        self.get(&id.ctx())
            .ok_or(TensorNotFound(id))?
            .get_tensor(id)
    }
}

impl<S: TensorSrc, const N: usize> TensorSrc for [S; N] {
    #[inline(always)]
    fn get_tensor(&self, id: TensorId) -> Result<&Tensor, TensorNotFound> {
        for src in self {
            if let Ok(tensor) = src.get_tensor(id) {
                return Ok(tensor);
            }
        }
        Err(TensorNotFound(id))
    }
}

impl<T: TensorSrc> TensorSrc for &T {
    #[inline(always)]
    fn get_tensor(&self, id: TensorId) -> Result<&Tensor, TensorNotFound> {
        (*self).get_tensor(id)
    }
}

impl<T: GetCtxId> GetCtxId for &T {
    #[inline(always)]
    fn id(&self) -> CtxId {
        (*self).id()
    }
}

use std::ops::DerefMut;

use gguf::FileType;
use infernal::{
    alloc::{Oom, SplitPoint},
    ComputeGraph, GraphBuilder, TensorId, TensorSrc,
};
use thiserror::Error;

use crate::{
    config::Config,
    kv_cache::KvCache,
    loader::{LoadError, TensorLoader},
    tokenizer::Tokenizer,
    Hyperparams, ParamSize, Params, TokenId, Vocab,
};

mod llama2;
mod runner;

pub use llama2::*;
pub use runner::*;

/// Correponds to the [`gguf::KvKey::GeneralArchitecture`] setting's possible values.
/// Later this will be generalized.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, strum::Display, strum::EnumString,
)]
#[strum(serialize_all = "lowercase")]
pub enum Arch {
    Llama,
}

/// Represents the stateless model housing the model weights and the compute graph logic.
// TODO: ModelGraph? we may want to support non-graph based models as well and compile them on the
// spot like torch/MLX. maybe split up trait into multiple subtraits
pub trait Model: Sized {
    // TODO: needed?
    const ARCH: Arch;

    type Tokenizer: Tokenizer;

    /// Construct a tokenizer given the vocabulary.
    fn new_tokenizer(vocab: Vocab) -> Self::Tokenizer;

    /// The constructor for the model.
    fn load(tl: &mut TensorLoader, ctx: LoadArgs<'_>) -> Result<Self, LoadError>;

    // TODO: consider removing
    fn optimal_split_points(&self) -> Vec<SplitPoint<TensorId>> {
        Vec::new()
    }

    fn common(&self) -> &Common;

    /// Builds the compute graph.
    ///
    /// Does not execute any matrix operations, it simply prepares the graph for later execution.
    /// Deending on the graph input, it either measures how large the compute buffer would be or
    /// actually constructs the graph and allocates it into the buffer.
    fn build_graph<B: DerefMut<Target = [u8]>, S: TensorSrc>(
        &self,
        ctx: GraphArgs<'_, '_, B, S>,
        input: GraphInput<'_>,
    ) -> Result<(), GraphError>;
}

/// Common model metadata.
#[derive(Debug)]
pub struct Common {
    pub param_size: Option<ParamSize>,
    // pub arch: Arch,
    pub ftype: Option<FileType>,
    pub name: Option<String>,
    pub hparams: Hyperparams,
}

/// Arguments passed to the model loader.
pub struct LoadArgs<'a> {
    pub params: &'a Params,
    pub hparams: Hyperparams,
    pub ftype: Option<FileType>,
    pub name: Option<String>,
}

/// Common contextual arguments passed to the model graph builder.
pub struct GraphArgs<'a, 'alloc, B: DerefMut<Target = [u8]>, S: TensorSrc> {
    pub kv_cache: &'a KvCache,
    pub compute_tb: &'a mut GraphBuilder<'alloc, B, S>,
    pub past_len: usize,
}

pub type GraphInput<'a> = Input<GraphOp<'a, TokenId>, GraphOp<'a, f32>>;
pub type EvalInput<'a> = Input<&'a [TokenId], &'a [f32]>;

/// Whether the input matrix is for token IDs or the embedding floats directly.
pub enum Input<T, E> {
    Tokens(T),
    Embd(E),
}

impl EvalInput<'_> {
    #[inline(always)]
    pub fn len(&self) -> usize {
        match self {
            Self::Tokens(tokens) => tokens.len(),
            Self::Embd(tokens) => tokens.len(),
        }
    }

    #[must_use]
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a> From<&'a [TokenId]> for EvalInput<'a> {
    #[inline(always)]
    fn from(tokens: &'a [TokenId]) -> Self {
        Self::Tokens(tokens)
    }
}

/// There are two types of graph builder operations: one is to measure the graph's memory usage and
/// the other is to pre-evaluate the graph with a given input.
pub enum GraphOp<'a, T> {
    Measure {
        seq_len: usize,
        graph: &'a mut ComputeGraph,
    },
    PreEval(&'a [T]),
}

impl<T> GraphOp<'_, T> {
    #[inline(always)]
    pub fn len(&self) -> usize {
        match self {
            Self::Measure { seq_len: n, .. } => *n,
            Self::PreEval(buf) => buf.len(),
        }
    }

    #[must_use]
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Error)]
pub enum GraphError {
    #[error(transparent)]
    TensorAlloc(#[from] Oom),
    #[error(transparent)]
    TensorNotFound(#[from] infernal::TensorNotFound),
}

impl From<&'_ Config> for crate::Params {
    fn from(c: &'_ Config) -> Self {
        Self {
            seed: c.runner.seed,
            max_ctx_len: c.runner.window_len as _,
            batch_len: c.runner.batch_len as _,
            n_gpu_layers: c.runner.gpu_layers,
            rope_freq_base: c.runner.rope_freq_base,
            rope_freq_scale: c.runner.rope_freq_scale,
            f16_kv: c.runner.f16_kv,
        }
    }
}

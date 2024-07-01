#![allow(dead_code)]

use std::{borrow::Cow, collections::HashMap};

use gguf::TokenType;
use infernal::Type;
use thiserror::Error;

use crate::tokenizer::SpecialTokens;

pub use self::{llm::*, loader::*, model::*};
pub use infernal::backend;

pub mod config;
pub mod kv_cache;
mod llm;
mod loader;
pub mod model;
pub mod sample;
pub mod template;

#[cfg(test)]
pub(crate) mod test_utils;
pub mod tokenizer;

/// A token returned in a single inference step.
#[derive(Debug)]
pub enum TokenText {
    /// Valid utf-8 string.
    String(String),
    /// Partial utf-8 string.
    Bytes(Vec<u8>),
    /// Single utf-8 byte, part of a glyph.
    Byte(u8),
}

impl TokenText {
    /// Converts to raw bytes.
    ///
    /// Note that this may not be a valid UTF-8 string. For that, use
    /// [`Utf8Constructor`](Utf8Constructor).
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::String(s) => s.as_bytes(),
            Self::Bytes(b) => b,
            Self::Byte(b) => std::slice::from_ref(b),
        }
    }
}

/// Global error containing all error types LLM inference may return.
#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Load(#[from] LoadError),
    #[error(transparent)]
    Init(#[from] InitError),
    #[error(transparent)]
    Eval(#[from] EvalError),
}

/// Low-level LLM configuration options.
#[derive(Debug)]
pub struct Params {
    /// None means random.
    pub seed: Option<u32>,
    pub max_ctx_len: usize,
    pub batch_len: usize,
    /// None means offloading everything.
    pub n_gpu_layers: Option<usize>,
    // pub tensor_split: Vec<f32>,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub f16_kv: bool,
    // pub logits_all: bool,
    // pub vocab_only: bool,
    // pub use_mlock: bool,
    // pub embedding: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            seed: Some(0xFFFFFFFF),
            max_ctx_len: 512,
            batch_len: 512,
            // n_gqa: 1,
            // rms_norm_eps: 5e-6,
            n_gpu_layers: None,
            // tensor_split: Default::default(),
            rope_freq_base: 10_000.,
            rope_freq_scale: 1.,
            f16_kv: true,
            // logits_all: false,
            // vocab_only: false,
            // use_mlock: Default::default(),
            // embedding: false,
        }
    }
}

/// The hyperparameters defined by the model.
///
/// This is read from the model's container file and should not be changed by the user, except where
/// noted otherwise.
#[derive(Debug)]
pub struct Hyperparams {
    pub n_vocab: usize,
    pub max_train_ctx_len: usize,
    pub max_ctx_len: usize,
    pub embd_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub n_layers: usize,
    pub n_rot: usize,
    pub ff_hidden_dim: usize,

    // This is both norm_rms_eps and norm_eps, former used by Llama, latter by Falcon. Since they don't
    // overlap, we can just use one field.
    pub f_norm_eps: f32,

    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
}

impl Hyperparams {
    fn from_partial(partial: PartialHyperparams, params: &Params) -> Self {
        Self {
            n_vocab: partial.n_vocab,
            max_train_ctx_len: partial.max_train_ctx_len,
            max_ctx_len: params.max_ctx_len,
            embd_dim: partial.embd_dim,
            n_heads: partial.n_heads,
            n_kv_heads: partial.n_kv_heads,
            n_layers: partial.n_layers,
            n_rot: partial.n_rot,
            ff_hidden_dim: partial.ff_hidden_dim,

            f_norm_eps: partial.f_norm_eps,

            // TODO: move out? seems like should be able to keep in params
            rope_freq_base: params.rope_freq_base,
            rope_freq_scale: params.rope_freq_scale,
        }
    }

    #[inline(always)]
    pub fn gqa_embd_dim(&self) -> usize {
        self.embd_dim / self.n_gqa()
    }

    #[inline(always)]
    pub fn head_embd_dim(&self) -> usize {
        self.embd_dim / self.n_heads
    }

    #[inline(always)]
    pub fn n_gqa(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    #[inline(always)]
    pub fn kv_size(&self, typ: Type) -> usize {
        2 * self.gqa_embd_dim() * self.max_ctx_len * self.n_layers * typ.size()
    }
}

/// Intermediary type used for constructing the hyperparameters.
#[derive(Debug)]
struct PartialHyperparams {
    n_vocab: usize,
    max_train_ctx_len: usize,
    embd_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    n_layers: usize,
    n_rot: usize,
    ff_hidden_dim: usize,

    f_norm_eps: f32,
}

/// Common model sizes (denotes number of weights).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ParamSize {
    _1B,
    _3B,
    _7B,
    _13B,
    _15B,
    _30B,
    _34B,
    _40B,
    _65B,
    _70B,
}

// TODO: verify that this cannot be negative! other than the invalid value used in cpp
pub type TokenId = i32;
// TODO: maybe string?
// TODO#perf: use smallvec possibly to avoid many small allocs
// TODO#perf: or possibly use Bytes to intern token. pointless to have multiple copies
pub type Token = Vec<u8>;

#[derive(Clone, Copy, Debug)]
pub enum VocabType {
    Spm,
    Bpe,
}

// TODO: move into loader since this is metadata specific to file loading otherwise not used
pub struct Vocab {
    // TODO: token here should be a string?
    pub token_to_id: HashMap<Token, TokenId>,
    pub id_to_token: Vec<TokenData>,
    pub bpe_ranks: HashMap<(String, String), i32>,
    pub typ: VocabType,
    pub special: SpecialTokens,
}

#[derive(Clone, Debug)]
pub struct TokenData {
    pub text: Token,
    pub typ: TokenType,
    pub score: f32,
}

impl Vocab {
    pub fn with_capacity(typ: VocabType, special: SpecialTokens, n: usize) -> Self {
        Self {
            token_to_id: HashMap::with_capacity(n),
            id_to_token: Vec::with_capacity(n),
            bpe_ranks: HashMap::new(),
            typ,
            special,
        }
    }

    pub fn insert(&mut self, id: TokenId, text: Token, typ: TokenType, score: f32) {
        self.token_to_id.insert(text.clone(), id);
        // The id is implicit in the array index of this element.
        self.id_to_token.push(TokenData { text, typ, score });
    }

    // TODO: still needed?
    // pub fn find_bpe_rank(&self, token_left: &str, token_right: &str) -> Option<i32> {
    //     let token_left = token_left.replace(' ', "u{0120}").replace('\n', "u{010A}");
    //     let token_right = token_right.replace(' ', "u{0120}").replace('\n', "u{010A}");
    //     self.bpe_ranks.get(&(token_left, token_right)).cloned()
    // }
}

/// Common LLM tensor classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorCategory {
    TokenEmbd,
    PosEmbd,
    Output,
    OutputNorm,
    RopeFreqs,
    AttnQ,
    AttnK,
    AttnV,
    AttnQkv,
    AttnOut,
    AttnNorm,
    AttnNorm2,
    AttnRotEmbd,
    FfnGate,
    FfnDown,
    FfnUp,
    FfnNorm,
}

// TODO: generic. maybe should be part of Model?
impl TensorCategory {
    pub fn to_string(
        &self,
        arch: Arch,
        suffix: Option<&str>,
        layer: Option<usize>,
    ) -> Cow<'static, str> {
        let s = match arch {
            Arch::Llama => match (self, layer) {
                (Self::TokenEmbd, _) => "token_embd".into(),
                (Self::OutputNorm, _) => "output_norm".into(),
                (Self::Output, _) => "output".into(),
                (Self::RopeFreqs, _) => "rope_freqs".into(),
                (Self::AttnNorm, Some(layer)) => format!("blk.{layer}.attn_norm").into(),
                (Self::AttnQ, Some(layer)) => format!("blk.{layer}.attn_q").into(),
                (Self::AttnK, Some(layer)) => format!("blk.{layer}.attn_k").into(),
                (Self::AttnV, Some(layer)) => format!("blk.{layer}.attn_v").into(),
                (Self::AttnOut, Some(layer)) => format!("blk.{layer}.attn_output").into(),
                (Self::AttnRotEmbd, Some(layer)) => format!("blk.{layer}.attn_rot_embd").into(),
                (Self::FfnNorm, Some(layer)) => format!("blk.{layer}.ffn_norm").into(),
                (Self::FfnGate, Some(layer)) => format!("blk.{layer}.ffn_gate").into(),
                (Self::FfnDown, Some(layer)) => format!("blk.{layer}.ffn_down").into(),
                (Self::FfnUp, Some(layer)) => format!("blk.{layer}.ffn_up").into(),
                t => unreachable!("architecture doesn't have this tensor: {t:?}"),
            },
        };
        // TODO#perf: this is not the most efficient, we could be allocating+formatting the string
        // in one go. refactor
        if let Some(suffix) = suffix {
            format!("{}.{}", s, suffix).into()
        } else {
            s
        }
    }
}

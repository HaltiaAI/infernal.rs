use infernal::{
    alloc::Oom, builder::TensorOptionsExt, Backend, TensorBuilder, TensorCtx, TensorId, Type, D1,
};
use thiserror::Error;
use tracing::info;

use crate::Hyperparams;

/// Key-value cache for storing previous attention results.
#[derive(Debug)]
pub struct KvCache {
    pub(crate) typ: Type,
    pub(crate) k: TensorId,
    pub(crate) v: TensorId,
    // NB: This is only exposed for the debugger.
    #[doc(hidden)]
    pub ctx: TensorCtx,
    /// Number of tokens in cache. This is incremented with each call to `Context::eval`.
    // TODO: this is not being used, I think?
    pub(crate) n_tokens: usize,
}

#[derive(Debug, Error)]
pub enum KvCacheInitError {
    #[error("error allocating KV cache buffer: {0}")]
    BufAlloc(#[from] Oom),
    #[error("error while building KV cache tensor context: {0}")]
    BackendBuffer(#[from] infernal::builder::BackendBufError),
}

impl KvCache {
    /// Initializes a new key-value cache with a size fitting for the model's hyperparameters.
    pub(crate) fn new(
        backend: &mut impl Backend,
        hparams: &Hyperparams,
        typ: Type,
        max_ctx_len: usize,
    ) -> Result<Self, KvCacheInitError> {
        // TODO: make this generic, this is llama2 specific
        let n_elems = hparams.gqa_embd_dim() * hparams.n_layers * max_ctx_len;
        // +2 MB for margin
        let margin = 2 * 1024 * 1024;
        let mem_size = 2 * n_elems * typ.size() + margin;

        info!(mem_size, n_elems, max_ctx_len, "initializing KV cache");

        let mut tb = TensorBuilder::prealloc(mem_size)?.with_name("kv_cache");

        let k = tb.new_tensor(D1::from(n_elems).with_type(typ).with_name("cache_k"))?;
        let v = tb.new_tensor(D1::from(n_elems).with_type(typ).with_name("cache_v"))?;

        Ok(Self {
            typ,
            k,
            v,
            ctx: tb.finish(backend)?.0,
            n_tokens: 0,
        })
    }
}

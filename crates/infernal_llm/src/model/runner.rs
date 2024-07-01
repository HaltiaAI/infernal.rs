use std::{
    cell::RefCell,
    fmt::Debug,
    rc::Rc,
    time::{Duration, Instant},
};

use infernal::{
    alloc::{self, Oom},
    backend::BackendBuf,
    buf::Buf,
    builder::{AutoOpName, NoSrc},
    Backend, ComputeGraph, GetCtxId, GraphAlloc, TensorBuilder, TensorCtx, TensorNotFound, Type,
};
use thiserror::Error;
use tracing::{debug, info, trace};

use crate::{
    kv_cache::{KvCache, KvCacheInitError},
    load,
    model::{EvalInput, GraphArgs, GraphError, GraphInput, GraphOp, Model},
    sample::{self, Sample, Sampler},
    tokenizer::Tokenizer,
    Config, LoadError, Params, TensorLoader, TokenId, TokenText,
};

/// Handles low-level model related operations like building the graph and launching the backend
/// computation and holding state such as the cache, logits, etc. Does not manage context window,
/// which is handled by [`Llm`](crate::Llm).
///
/// The default implementation of [`DefaultModelRunner`] is based on Infernal but it is possible to
/// plug in different ML backends by implementing this trait and passing it to the
/// [`Llm`](crate::Llm) type.
pub trait ModelRunner<M: Model> {
    type Error: std::error::Error;

    fn load(config: &Config, backend: &mut impl Backend) -> Result<Self, Self::Error>
    where
        Self: Sized;

    /// Tokenizes the input string with the model's tokenizer into the output buffer.
    fn tokenize(
        &mut self,
        input: &str,
        // TODO: what is this? is this tokenizer dependent and as such should be handled by each
        // tokenizer impl?
        add_bos: bool,
        output: &mut Vec<TokenId>,
    ) -> Result<(), Self::Error>;

    /// Evaluates the input, which may be either the initial prompt or previously decoded tokens.
    /// The tokens may be in the form of token IDs or embeddings.
    fn eval<'a>(
        &mut self,
        input: impl Into<EvalInput<'a>>,
        past_len: usize,
        backend: &mut impl Backend,
    ) -> Result<(), Self::Error>;

    fn sample_next(&mut self, context_window: &[TokenId]) -> Option<TokenId>;

    fn token_to_string(&self, token: TokenId) -> Option<TokenText>;

    fn token_bos(&self) -> TokenId;

    fn token_eos(&self) -> TokenId;
    fn stats(&self) -> &Stats;
    fn clear_stats(&mut self);
    fn vocab_len(&self) -> usize;
    fn window_len(&self) -> usize;
    fn logits(&self) -> &[f32];
    fn clear_logits(&mut self);

    fn model(&self) -> &M;
}

#[derive(Debug, Error)]
pub enum InitError {
    #[error("error loading model: {0}")]
    Load(#[from] LoadError),
    // TODO
    // #[error("error initializing backend: {0}")]
    // Backend(#[from] infernal::backend::DeviceNotFound),
    #[error("error allocating compute buffer: {0}")]
    ComputeBufAlloc(#[from] Oom),
    #[error("error initializing KV cache: {0}")]
    KvCache(KvCacheInitError),
    #[error("error measuring compute graph size: {0}")]
    ComputeGraphMeasure(#[from] GraphError),
    #[error("error optimizing compute graph: {0}")]
    ComputeGraphOpt(#[from] TensorNotFound),
    #[error("error while building {0} tensor context: {1}")]
    BackendBuffer(&'static str, infernal::builder::BackendBufError),
    #[error("error while splitting weights: {0}")]
    WeightSplit(#[from] infernal::alloc::SplitError),
}

#[derive(Debug, Error)]
pub enum EvalError {
    #[error("error building compute graph: {0}")]
    Build(#[from] GraphError),
    // This shouldn't occur because we measure the graph size before building it, so the allocator
    // should always have space for the whole graph.
    // TODO: consider unwrapping instead
    #[error("error allocating graph tensors: {0}")]
    Alloc(#[from] Oom),
    #[error("error computing graph: {0}")]
    Compute(#[from] infernal::backend::ComputeError),
}

/// Default Infernal based model runner implementation.
pub struct DefaultModelRunner<M: Model, S: Sample = sample::Greedy> {
    model: M,
    tokenizer: M::Tokenizer,
    kv_cache: KvCache,

    has_evaluated_once: bool,

    logits: Vec<f32>,
    embd: Vec<f32>,

    weights_ctx: TensorCtx,
    /// The computation graph stores the execution plan. Calculated once at startup, can be reused
    /// as long as the shape of the graph does not change.
    graph: ComputeGraph,
    /// The compute buffer used to allocate intermediate tensor data in.
    ///
    /// A clone of this is stored in backend, and one here so we can re-allocate tensors, because the
    /// tensor sizes change with each inference step.
    ///
    /// To mutate this, you need to ensure that the backend is not actively referencing it.
    compute_buf: Rc<RefCell<Buf>>,

    /// Parts of the compute graph tensor builder that can be reused for each eval to save memory
    /// allocations.
    compute_tb_parts: infernal::builder::RawParts,
    alloc_parts: alloc::graph::RawParts,

    // Only using greedy sampler for now, but can change later.
    sampler: Sampler<S>,

    stats: Stats,
}

// TODO: remove this in favor of infernal_llm::context::Stats once this crate is merged into
// infernal_llm
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    /// This value is either the load time at first startup or the combined first startup and first eval times.
    ///
    /// Because the mmap'd page is not actually loaded into the program's address space when the
    /// mmapping is created (kernel lazy optimization), the first actual load of the model occurs
    /// during the first call to `Context::eval` when the program page faults on the uninitialized mapping.
    /// Therefore after construction but before the first eval, this value only reflects the time it
    /// takes to open the file, calling mmap, and initialize other structures.
    ///
    /// Once there has been at least one evaluation of the compute graph, the load time is adjusted
    /// for the first page fault. Thus, this value is necessarily an approximation.
    pub load_time: Duration,
    pub start_time: Duration,

    pub sample_time: Duration,
    pub eval_time: Duration,
    pub prompt_eval_time: Duration,

    pub n_sample: usize,
    pub n_eval: usize,
    pub n_prompt_eval: usize,
}

impl<M: Model> DefaultModelRunner<M> {
    /// Creates a new runner with the given loaded model and adds the model weights to the provided
    /// backend.
    pub fn new(
        model: M,
        params: Params,
        tokenizer: M::Tokenizer,
        loader: TensorLoader,
        backend: &mut impl Backend,
    ) -> Result<Self, InitError> {
        info!(
            params.n_gpu_layers,
            params.max_ctx_len, params.batch_len, "initializing haltia backend"
        );
        debug!(?params);

        let mut weights_tb = loader.finish()?;

        let hparams = &model.common().hparams;

        let kv_type = if params.f16_kv { Type::F16 } else { Type::F32 };
        let kv_cache = KvCache::new(backend, hparams, kv_type, params.max_ctx_len)
            .map_err(InitError::KvCache)?;
        let kv_cache_size = kv_cache.ctx[kv_cache.k].size() + kv_cache.ctx[kv_cache.v].size();
        info!(
            kv_cache_size,
            params.f16_kv, params.max_ctx_len, "allocated KV cache"
        );

        const TENSOR_ALIGN: usize = 32;

        // Build worst case graph to estimate allocation size.
        let start = Instant::now();
        let seq_len = hparams.max_ctx_len.min(params.batch_len);
        let past_len = hparams.max_ctx_len - seq_len;
        let mut alloc = GraphAlloc::measure();
        let mut compute_tb =
            TensorBuilder::graph_alloc_with_src(&mut alloc, (&weights_tb, &kv_cache.ctx))
                .with_auto_op_names(AutoOpName::expand());
        let mut graph = ComputeGraph::default();

        let build_start = Instant::now();
        model
            .build_graph(
                GraphArgs {
                    kv_cache: &kv_cache,
                    compute_tb: &mut compute_tb,
                    past_len,
                },
                GraphInput::Tokens(GraphOp::Measure {
                    seq_len,
                    graph: &mut graph,
                }),
            )
            .map_err(InitError::ComputeGraphMeasure)?;
        let build_time = build_start.elapsed();

        // And find the graph execution plan. Only need to do this once at startup.
        debug_assert!(!graph.nodes().is_empty());
        graph.find_exec_plan(backend.mem_limit(), &compute_tb)?;
        let exec_plan = graph.exec_plan(backend.mem_limit()).unwrap();

        info!(
            time = ?start.elapsed(),
            ?build_time,
            n_tensors = compute_tb.len(),
            graph.nodes = graph.nodes().len(),
            graph.leaf_nodes = graph.leaf_nodes().len(),
            plan.levels = exec_plan.levels.len(),
            plan.max_level_nodes = exec_plan.max_level_nodes,
            plan.max_level_size = exec_plan.max_level_size,
            "optimized compute graph"
        );

        // This won't allocate anything, just measure how much would be needed for this graph.
        let start = Instant::now();
        let compute_buf_size = compute_tb.alloc_graph(&graph, backend.mem_limit())?;
        // TODO: TENSOR_ALIGN needed?
        let compute_buf_size = compute_buf_size + TENSOR_ALIGN;
        info!(
            time = ?start.elapsed(),
            compute_buf_size = compute_buf_size,
            compute_buf_size_mb = compute_buf_size / 1024 / 1024,
            "measured worst case compute buffer size"
        );
        let compute_buf = Rc::new(RefCell::new(Buf::new(compute_buf_size)?));
        backend
            .add_buf(BackendBuf::from(compute_buf.clone()).with_name("compute"))
            .map_err(|e| {
                InitError::BackendBuffer(
                    "compute",
                    infernal::builder::BackendBufError::MemoryLimitTooLow(e),
                )
            })?;

        let compute_tb = compute_tb.into_no_src();

        let split_points = model.optimal_split_points();
        if !split_points.is_empty() {
            weights_tb.split_at_tensors(backend.mem_limit(), split_points)?;
        }

        let weights_ctx = weights_tb
            .finish(backend)
            .map_err(|e| InitError::BackendBuffer("weights", e))?
            .0;

        let compute_tb_parts = compute_tb.into_raw_parts();
        let max_ctx_len = hparams.max_ctx_len;
        let n_vocab = hparams.n_vocab;

        Ok(Self {
            model,
            tokenizer,
            has_evaluated_once: false,
            kv_cache,
            logits: vec![0f32; max_ctx_len],
            embd: Vec::new(),
            graph,
            compute_buf,
            compute_tb_parts,
            alloc_parts: alloc.into_raw_parts(),
            weights_ctx,
            sampler: Sampler::new(sample::Greedy, n_vocab),
            stats: Default::default(),
        })
    }

    #[inline(always)]
    pub fn compute_graph(&self) -> &ComputeGraph {
        &self.graph
    }

    #[inline(always)]
    pub fn model(&self) -> &M {
        &self.model
    }

    #[inline(always)]
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    // NB: This is only exposed for the debugger.
    #[doc(hidden)]
    pub fn rebuild_compute_graph(
        &mut self,
        input: &EvalInput<'_>,
        past_len: usize,
        backend: &impl Backend,
    ) -> Result<TensorCtx, EvalError> {
        // TODO#perf: after saturating the context window and thus tokens being of equal length, can
        // we keep the matrices the same size, avoiding the rebuild here?
        // TODO#perf: measure overhead here
        let start = Instant::now();
        let mut compute_tb = (*self.compute_buf).borrow_mut();
        let mut alloc =
            GraphAlloc::new(&mut compute_tb).with_raw_parts(std::mem::take(&mut self.alloc_parts));
        let mut compute_tb = TensorBuilder::graph_alloc_with_src(
            &mut alloc,
            [&self.weights_ctx, &self.kv_cache.ctx],
        )
        .with_raw_parts(std::mem::take(&mut self.compute_tb_parts));
        self.model.build_graph(
            GraphArgs {
                kv_cache: &self.kv_cache,
                compute_tb: &mut compute_tb,
                past_len,
            },
            match input {
                EvalInput::Tokens(tokens) => GraphInput::Tokens(GraphOp::PreEval(tokens)),
                EvalInput::Embd(embd) => GraphInput::Embd(GraphOp::PreEval(embd)),
            },
        )?;
        compute_tb.alloc_graph(&self.graph, backend.mem_limit())?;
        debug!(
            time = ?start.elapsed(),
            n_nodes = self.graph.nodes().len(),
            n_leaf_nodes = self.graph.leaf_nodes().len(),
            n_tensors = compute_tb.len(),
            "rebuilt compute graph"
        );
        let compute_ctx = compute_tb.finish_discard_buf();
        self.alloc_parts = alloc.into_raw_parts();
        Ok(compute_ctx)
    }

    // NB: This is only exposed for the debugger.
    #[doc(hidden)]
    pub fn tensor_src<'s: 'c, 'c>(&'s self, comp_ctx: &'c TensorCtx) -> [&'c TensorCtx; 3] {
        [&self.weights_ctx, &self.kv_cache.ctx, comp_ctx]
    }

    /// Extracts logits from the compute context.
    fn extract_logits(&mut self, n: usize) {
        let output = self.graph.output().expect("missing output tensor");
        let hparams = &self.model.common().hparams;
        let output_offs = hparams.n_vocab * (n - 1);
        self.logits.resize_with(hparams.n_vocab, Default::default);

        // NB: We need to re-instantiate comp_tb because the comp_buf is borrowed by the backend
        // TODO: think of a clearer api for copying out tensor data when the builder is no longer
        // available
        let mut comp_buf = (*self.compute_buf).borrow_mut();
        let comp_tb = TensorBuilder::<_, NoSrc>::reuse(&mut comp_buf)
            .with_raw_parts(std::mem::take(&mut self.compute_tb_parts));
        trace!(
            output_tensor = %output,
            compute_ctx = %comp_tb.id(),
            "extracting logits from compute context",
        );
        comp_tb.copy_data_to(output, output_offs, &mut self.logits);

        // TODO
        // if let Input::Embd(_) = input {
        //     // let embd_result = self.backend.get_embd_tensor(&mut self.graph);
        //     let n_embd = self.model.common().hparams.n_embd as usize;
        //     // Extract embeddings if requested.
        //     self.embd.resize_with(n_embd, Default::default);
        //     embd_result.copy_data_to(offset, &mut self.embd);
        // }

        self.compute_tb_parts = comp_tb.into_raw_parts();
    }
}

impl<M: Model + Debug> ModelRunner<M> for DefaultModelRunner<M> {
    type Error = crate::Error;

    #[inline(always)]
    fn load(config: &Config, backend: &mut impl Backend) -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        let params = config.into();
        let (model, tokenizer, tl) = load(&config.model.path, &params)?;
        Ok(Self::new(model, params, tokenizer, tl, backend)?)
    }

    #[inline(always)]
    fn tokenize(
        &mut self,
        input: &str,
        _add_bos: bool, // TODO: can we remove?
        output: &mut Vec<TokenId>,
    ) -> Result<(), Self::Error> {
        self.tokenizer.tokenize(input, output);
        Ok(())
    }

    fn eval<'a>(
        &mut self,
        input: impl Into<EvalInput<'a>>,
        past_len: usize,
        backend: &mut impl Backend,
    ) -> Result<(), Self::Error> {
        let start = Instant::now();

        let input = input.into();

        // Rebuild the compute graph for this particular iteration. This needs to be done because as
        // the prior tokens are fed into the model, `tokens` grows and thus the shape of the
        // matrices changes with it.
        // TODO(perf): if input is the same length as the previous input, we can avoid rebuilding
        // the graph. otoh this is very fast so not a huge win.
        let compute_ctx = self.rebuild_compute_graph(&input, past_len, backend)?;

        let compute_start = Instant::now();
        backend
            .compute_graph(
                &self.graph,
                &[&self.weights_ctx, &self.kv_cache.ctx, &compute_ctx],
            )
            .map_err(EvalError::Compute)?;

        let compute_time = compute_start.elapsed();
        let total_time = start.elapsed();
        self.compute_tb_parts = compute_ctx.into_raw_parts();

        trace!(
            ?total_time,
            graph_build_time = ?(total_time - compute_time),
            ?compute_time,
            n_tokens = input.len(),
            past_len,
            "evaluated compute graph"
        );

        self.extract_logits(input.len());

        // Update KV cache token count.
        self.kv_cache.n_tokens = past_len + input.len();

        if !self.has_evaluated_once {
            self.has_evaluated_once = true;
            self.stats.load_time += total_time;
        }

        // We measure prompt evaluation (usually input longer than 1) and auto-regressive single
        // token evaluation separately.
        if input.len() == 1 {
            self.stats.eval_time += total_time;
            self.stats.n_eval += 1;
        } else {
            self.stats.prompt_eval_time += total_time;
            self.stats.n_prompt_eval += input.len();
        }

        Ok(())
    }

    #[inline(always)]
    fn sample_next(&mut self, context_window: &[TokenId]) -> Option<TokenId> {
        let start = Instant::now();

        let token = self.sampler.sample_next(&self.logits, context_window)?;

        self.stats.sample_time += start.elapsed();
        self.stats.n_sample += 1;

        Some(token)
    }

    #[inline(always)]
    fn token_to_string(&self, token: TokenId) -> Option<TokenText> {
        self.tokenizer.token_to_string(token)
    }

    #[inline(always)]
    fn token_bos(&self) -> TokenId {
        self.tokenizer.special().bos
    }

    #[inline(always)]
    fn token_eos(&self) -> TokenId {
        self.tokenizer.special().eos
    }

    #[inline(always)]
    fn stats(&self) -> &Stats {
        &self.stats
    }

    #[inline(always)]
    fn clear_stats(&mut self) {
        // Preserve start and load across resets since they won't change. Only reset inference
        // related timings.
        self.stats = Stats {
            start_time: self.stats.start_time,
            load_time: self.stats.load_time,
            ..Default::default()
        };
    }

    #[inline(always)]
    fn vocab_len(&self) -> usize {
        self.tokenizer.vocab_len()
    }

    #[inline(always)]
    fn window_len(&self) -> usize {
        self.model().common().hparams.max_ctx_len as _
    }

    #[inline(always)]
    fn logits(&self) -> &[f32] {
        &self.logits
    }

    #[inline(always)]
    fn clear_logits(&mut self) {
        self.logits.clear();
    }

    fn model(&self) -> &M {
        &self.model
    }
}

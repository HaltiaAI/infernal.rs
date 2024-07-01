use infernal::Backend;
use tracing::{debug, info, trace, warn};

pub use crate::{
    config::{Config, LoraConfig, ModelConfig, RunnerConfig, RuntimeConfig},
    model::{ModelRunner, Stats},
    template::Template,
    TokenId, TokenText,
};
use crate::{DefaultModelRunner, Model};

// From: "Mistral 7B" https://arxiv.org/pdf/2310.06825.pdf
// TODO: include in the per model hard-coded config.
pub const DEFAULT_SYS_PROMPT: &str = "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.";

/// The LLM may spit out partial utf-8 glyphs which are not fit for consumption by the higher class
/// societies of our code base. This struct is a simple state machine that keeps track of the bytes and
/// returns a utf-8 string when it's ready. Already valid utf-8 strings are just passed through.
#[derive(Default)]
pub struct Utf8Constructor {
    buf: Vec<u8>,
}

impl Utf8Constructor {
    /// Returns a utf-8 string if the bytes are ready, otherwise None.
    #[inline(always)]
    pub fn get(&mut self, tt: TokenText) -> Option<String> {
        match tt {
            TokenText::String(token) => {
                if self.buf.is_empty() {
                    return Some(token);
                } else {
                    self.buf.extend(token.as_bytes());
                }
            }
            TokenText::Bytes(bytes) => {
                self.buf.extend(&bytes);
            }
            TokenText::Byte(byte) => {
                self.buf.push(byte);
            }
        }
        if let Ok(s) = std::str::from_utf8(&self.buf) {
            let s = s.to_string();
            self.buf.clear();
            Some(s)
        } else {
            // not ready yet
            None
        }
    }
}

/// The main LLM inferencing type.
///
/// Holds state between inferencing calls, handles recomputing the context window if the max size is
/// reached, etc.
pub struct Llm<M: Model, R: ModelRunner<M> = DefaultModelRunner<M>> {
    // TODO: remove
    template: Template,
    config: Config,
    model_runner: R,
    max_context_len: usize, // n_ctx in llama.cpp
    state: SessionState,
    // Model is stored in the runner, but need to use generic param.
    _model: std::marker::PhantomData<M>,
}

impl<M: Model, R: ModelRunner<M>> std::fmt::Debug for Llm<M, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Llm")
            .field("template", &self.template)
            .field("config", &self.config)
            .field("max_context_len", &self.max_context_len)
            .field("state", &self.state)
            .finish()
    }
}

#[derive(Debug)]
struct SessionState {
    embd: Vec<TokenId>,
    ctx_window: Vec<TokenId>,
    neg_prompt_len: usize,
    past_len: usize, // n_past in llama.cpp
    // Tokens buffer reused between calls to avoid unnecessary allocations.
    tok_buf: Vec<TokenId>,
}

impl SessionState {
    fn clear_embd(&mut self) {
        self.embd.clear();
    }
}

impl<M: Model, R: ModelRunner<M>> Llm<M, R> {
    /// Creates a new LLM instance.
    ///
    /// Tokenizes the system prompt but does not evaluate it. That happens on first call to [`Self::infer_next`]
    pub fn new(
        config: Config,
        template: Template,
        negative_prompt: &str,
        backend: &mut impl Backend,
    ) -> Result<Self, R::Error> {
        info!(path = ?config.model.path, "loading model");
        debug!(?config);
        let model_runner = R::load(&config, backend)?;
        Self::with_runner(config, template, negative_prompt, model_runner)
    }

    // Constructor for custom model runner.
    pub fn with_runner(
        config: Config,
        template: Template,
        negative_prompt: &str,
        mut model_runner: R,
    ) -> Result<Self, R::Error> {
        debug!(?config);

        let max_context_len = model_runner.window_len();

        let mut ctx_window = Vec::with_capacity(1024);
        let mut embd = Vec::with_capacity(1024);
        let mut neg_prompt_tokens = Vec::new();
        let negative_prompt = template.system_prompt(negative_prompt);
        if negative_prompt.is_empty() {
            warn!("negative prompt is empty");
        } else {
            model_runner.tokenize(&negative_prompt, false, &mut neg_prompt_tokens)?;
        }

        if neg_prompt_tokens.len() > max_context_len / 4 {
            warn!(
                prompt = ?negative_prompt,
                "Negative prompt token length of {} is too long and may \
                lead to performance and inference quality issues. Recommended token length: {}",
                neg_prompt_tokens.len(),
                max_context_len / 4
            );
        }

        let neg_prompt_len = neg_prompt_tokens.len();
        embd.extend(&neg_prompt_tokens);
        ctx_window.extend(&neg_prompt_tokens);

        Ok(Self {
            template,
            config,

            model_runner,
            max_context_len,
            state: SessionState {
                neg_prompt_len,
                embd,
                ctx_window,
                past_len: 0,
                tok_buf: neg_prompt_tokens,
            },
            _model: std::marker::PhantomData,
        })
    }

    /// Evaluates a single BOS token to warm up model on backend -- pulls kernels and data into
    /// memory.
    ///
    /// Recommended to use this at application startup so that first actual computation by user does
    /// not cause a hiccup.
    pub fn warmup(&mut self, backend: &mut impl Backend) -> Result<&mut Self, R::Error> {
        self.feed_tokens(&[self.model_runner.token_bos()]);
        self.eval(backend)?;
        self.clear_stats();
        Ok(self)
    }

    /// Overwrites the system prompt and clears internal state (since it needs to be recomputed for
    /// the new system prompt).
    pub fn set_system_prompt(&mut self, system_prompt: &str) -> Result<&mut Self, R::Error> {
        let system_prompt = self.template.system_prompt(system_prompt);
        let mut neg_prompt_tokens = Vec::new();
        self.model_runner
            .tokenize(&system_prompt, false, &mut neg_prompt_tokens)?;
        neg_prompt_tokens.push(self.model_runner.token_eos());

        if neg_prompt_tokens.len() > self.max_context_len / 4 {
            warn!(
                "Negative prompt token length of {} is too long and may \
            lead to performance and inference quality issues. Recommended token length: {} {system_prompt:?}",
                neg_prompt_tokens.len(),
                self.max_context_len / 4
            );
        }

        // We clear state, but this has a negative effect of removing previous messages as well.
        // This can be resolved by doing loading up the messages again.
        self.clear();
        self.state.past_len = 1; // TODO is this correct?

        self.state.embd.extend(&neg_prompt_tokens);
        self.state.ctx_window.extend(&neg_prompt_tokens);
        self.state.neg_prompt_len = neg_prompt_tokens.len();

        Ok(self)
    }

    /// Applites the template, tokenizes and feeds the input into the context window.
    pub fn feed_prompt(&mut self, prompt: &str) -> Result<&mut Self, R::Error> {
        let prompt = self.template.prompt(prompt);
        self.model_runner
            .tokenize(&prompt, false, &mut self.state.tok_buf)?;
        self.feed_tokens_impl();
        Ok(self)
    }

    /// Feeds the prompt without templating into the context window.
    pub fn feed_prompt_raw(&mut self, prompt: &str) -> Result<&mut Self, R::Error> {
        self.model_runner
            .tokenize(prompt, false, &mut self.state.tok_buf)?;
        self.feed_tokens_impl();
        Ok(self)
    }

    // TODO remove after removing template
    pub fn feed_user_prompt(&mut self, prompt: &str) -> Result<&mut Self, R::Error> {
        let prompt = self.template.user_prompt(prompt);
        self.model_runner
            .tokenize(&prompt, false, &mut self.state.tok_buf)?;
        self.feed_tokens_impl();
        Ok(self)
    }

    pub fn feed_system_prompt(&mut self, prompt: &str) -> Result<&mut Self, R::Error> {
        let prompt = self.template.system_prompt(prompt);
        self.model_runner
            .tokenize(&prompt, false, &mut self.state.tok_buf)?;
        self.feed_tokens_impl();
        Ok(self)
    }

    /// Directly feed tokenized input into context window.
    pub fn feed_tokens(&mut self, tokens: &[TokenId]) -> usize {
        // TODO: DRY up but lifetime issues
        self.state.tok_buf.clear();
        self.state.tok_buf.extend(tokens);
        self.feed_tokens_impl()
    }

    fn feed_tokens_impl(&mut self) -> usize {
        let tokens = &self.state.tok_buf;
        if tokens.len() > (self.max_context_len / 2) + self.state.ctx_window.len() {
            panic!(
                "Prompt being truncated as the length of {} is greater \
            than half the context length of {}",
                tokens.len(),
                1024
            );
        }

        self.state.embd.extend(tokens);
        self.state.ctx_window.extend(tokens);

        if self.state.past_len + self.state.embd.len() > self.max_context_len {
            self.context_swap();
        }

        self.state.ctx_window.len()
    }

    /// If the context window exceeds the max length, clear half of it so that new prompts can be
    /// included in it. The logits are recomputed during the next inference call.
    fn context_swap(&mut self) {
        info!(
            "Context size of {} exceeded, clearing context.",
            self.max_context_len
        );
        self.state.past_len = 1;

        let half = (self.state.neg_prompt_len + self.state.ctx_window.len()) / 2;

        // Drain half for now.
        self.state.ctx_window.drain(self.state.neg_prompt_len..half);

        // Recompute the logits for the new context window.
        self.state.embd.clear();
        self.state.embd.extend_from_slice(&self.state.ctx_window);

        if self.state.embd.len() > 1024 {
            let rem = self.state.embd.len() - 1024;
            self.state.embd.drain(self.state.neg_prompt_len..rem);
        }
    }

    /// Runs the inference loop, calling the callback for each new token, until no more tokens are
    /// generated.
    pub fn infer_loop(
        &mut self,
        backend: &mut impl Backend,
        mut token_callback: impl FnMut(String),
    ) -> Result<&mut Self, R::Error> {
        let mut utf8_ctor = Utf8Constructor::default();
        while let Some(tok) = self.infer_next(backend)? {
            if let Some(tok) = utf8_ctor.get(tok) {
                token_callback(tok);
            }
        }

        Ok(self)
    }

    /// Infers a single token and returns it, or None if no more tokens could be generated. If None
    /// is returned, the user should feed a new prompt before calling infer again.
    pub fn infer_next(
        &mut self,
        backend: &mut impl Backend,
    ) -> Result<Option<TokenText>, R::Error> {
        if self.state.past_len + self.state.embd.len() > self.max_context_len {
            self.context_swap();
        }

        // Use all generated tokens as context for next token
        self.eval(backend)?;

        // The embeddings are cleared after every iteration. This is because the embeddings must only
        // contain tokens that have not been seen: so for the user prompt it's a list of
        // all tokens in the prompt, but subsequent inference steps evaluate only a single
        // token, based on the previous token evaluated by the LLM. This is because the LLM
        // generates only a single token regardless of the embedding size, but subsequent tokens
        // need the previous tokens as context, hence going one by one.
        self.state.clear_embd();

        // Sample the next token
        let Some(new_token_id) = self.model_runner.sample_next(&self.state.ctx_window) else {
            return Ok(None);
        };
        trace!(new_token_id, "sampled next token");

        if !self.config.runner.ignore_eos && new_token_id == self.model_runner.token_eos() {
            trace!("[END OF TEXT]");
            self.state.clear_embd();
            return Ok(None);
        }

        // push this new token for next evaluation
        self.state.embd.push(new_token_id);

        if self.state.ctx_window.len() >= self.config.runner.window_len {
            self.state.ctx_window.remove(0);
        }
        self.state.ctx_window.push(new_token_id);

        // FIXME: this shouldn't happen but sometimes shortcircuits inference e.g. in benching.
        // we're getting a UserDefined token as last and not handling it apparently. what mean?
        let Some(token_text) = self.model_runner.token_to_string(new_token_id) else {
            return Ok(None);
        };

        Ok(Some(token_text))
    }

    /// Converts the context window tokens into strings and returns it as one string.
    pub fn dump_context_window(&self) -> String {
        let mut window = String::new();
        let mut ctor = Utf8Constructor::default();
        for token in self.state.ctx_window.iter() {
            if let Some(tok) = self.model_runner.token_to_string(*token) {
                // if let TokenText::Byte(byte) = &tok {
                //     if *byte == 10 {
                //         window.push('\n');
                //     }
                // }
                if let Some(tok) = ctor.get(tok) {
                    window.push_str(&tok);
                }
            }
        }
        window
    }

    /// Evaluates pending tokens in the context window in batches.
    fn eval(&mut self, backend: &mut impl Backend) -> Result<(), R::Error> {
        // TODO: Batching may be fixed, but needs to be tested.
        let batch_len = self.config.runner.batch_len;
        let n_batches = (self.state.embd.len() - 1) / batch_len + 1;
        for i in 0..n_batches {
            let batch_start = i * batch_len;
            let batch_end = {
                if self.state.embd.len() < batch_start + batch_len {
                    self.state.embd.len()
                } else {
                    batch_start + batch_len
                }
            };
            let tokens = &self.state.embd[batch_start..batch_end];
            self.model_runner
                .eval(tokens, self.state.past_len, backend)?;
            self.state.past_len += tokens.len();
        }

        Ok(())
    }

    /// Applies the template and tokenizes the input, returning the tokens.
    pub fn tokenize(
        &mut self,
        role: impl AsRef<str>,
        input: impl AsRef<str>,
    ) -> Result<Vec<TokenId>, R::Error> {
        let role = role.as_ref();
        let prompt = self.template.custom_prompt(role, input.as_ref());
        self.model_runner
            .tokenize(&prompt, false, &mut self.state.tok_buf)?;
        Ok(self.state.tok_buf.clone())
    }

    pub fn context_window(&self) -> &[TokenId] {
        &self.state.ctx_window
    }

    /// Clears computed logits, context window, embeddings, etc.
    ///
    /// Note, this won't recompute, which is done at the next inference.
    pub fn clear(&mut self) {
        self.state.embd.clear();
        self.state.ctx_window.clear();
        self.state.past_len = 0; // TODO: set to 1?
        self.model_runner.clear_logits();
    }

    pub fn template(&self) -> &Template {
        &self.template
    }

    pub fn stats(&self) -> &Stats {
        self.model_runner.stats()
    }

    /// Stats are not cleared in [`Self::clear`] so a separate method is provided.
    pub fn clear_stats(&mut self) {
        self.model_runner.clear_stats()
    }
}

#[cfg(test)]
mod test {
    use crate::Llama2;

    use super::*;

    // There was a bug in the `Llama::token_to_string` method that caused a crash when encountering
    // an emoji output.
    #[test]
    fn regression_emoji_crash() {
        let path = std::env::var("MODEL_PATH").unwrap();
        let mut backend = infernal::backend::Metal::new().unwrap();
        let mut llm = Llm::<Llama2>::new(
            Config {
                runtime: Default::default(),
                runner: Default::default(),
                model: ModelConfig::new(path),
                lora: Default::default(),
            },
            Template::new_mistral_openorca(),
            "",
            &mut backend,
        )
        .unwrap();

        llm.feed_prompt("Can you give me 10 emojis?").unwrap();
        let mut utf8_ctor = Utf8Constructor::default();
        llm.infer_loop(&mut backend, |token| {
            let Some(token) = utf8_ctor.get(token) else {
                return;
            };
            print!("{token}");
        })
        .unwrap();
        println!();
    }
}

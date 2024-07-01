use core::fmt;
use std::{
    str,
    time::{Duration, Instant},
};

use infernal_llm::{Llama2, Llm, Template, Utf8Constructor};
use thiserror::Error;
use tracing::{debug, info, trace};

use crate::{
    sample::{TokSample, TokSamples},
    ConfigBase,
};

pub use infernal_llm::Config as LlmConfig;

#[derive(Clone, Debug)]
pub struct Config {
    pub base: ConfigBase,
    pub llm: LlmConfig,
    pub n_tokens: Option<usize>,
    /// Only consider samples that have at least this many tokens evaluated.
    pub min_tokens: Option<usize>,
    // TODO: params for iphone specific slowdown evasion
}

impl Config {
    pub fn n_tokens(&self) -> usize {
        self.n_tokens.unwrap_or(N_TOKENS)
    }
}

pub const N_TOKENS: usize = 1024;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    BackendInit(#[from] infernal::backend::DeviceNotFound),
    #[error(transparent)]
    Llm(#[from] infernal_llm::Error),
    #[error("config error: {0}")]
    InvalidConfig(&'static str),
}

/// Stores measurement samples and provides methods for calculating statistics for tokens per
/// second.
#[derive(Default, Debug)]
pub struct Stats {
    pub init_time: Duration,
    pub total_time: Duration,
    pub sample: TokSamples,
    pub feed: TokSamples,
    pub infer: TokSamples,
}

impl Stats {
    pub fn print(&self, runs: usize, w: &mut impl std::io::Write) -> Result<(), std::io::Error> {
        super::print_stats_header(w, runs, &self.init_time, &self.total_time)?;
        let max_widths = Self::determine_max_widths(&[&self.sample, &self.feed, &self.infer]);
        Self::print_header(w, max_widths)?;
        Self::print_sample(w, "sampling", &self.sample, max_widths)?;
        Self::print_sample(w, "prompt eval", &self.feed, max_widths)?;
        Self::print_sample(w, "inference", &self.infer, max_widths)?;
        Ok(())
    }

    const PREFIX_PAD: usize = 16;

    // This function determines the max width required for each column.
    fn determine_max_widths(samples: &[&TokSamples]) -> (usize, usize, usize, usize) {
        let mut max_avg = 0;
        let mut max_mean = 0;
        let mut max_var = 0;
        let mut max_stdev = 0;

        fn width(s: impl fmt::Display) -> usize {
            format!("{:.4}", s).len()
        }

        for sample in samples {
            let avg = sample.avg();
            let mean = sample.mean();
            let var = sample.var();
            let stdev = sample.stdev();

            max_avg = max_avg
                .max(width(avg.tok_per_s))
                .max(width(avg.tokens))
                .max(width(avg.time_secs));
            max_mean = max_mean
                .max(width(mean.tok_per_s))
                .max(width(mean.tokens))
                .max(width(mean.time_secs));
            max_var = max_var
                .max(width(var.tok_per_s))
                .max(width(var.tokens))
                .max(width(var.time_secs));
            max_stdev = max_stdev
                .max(width(stdev.tok_per_s))
                .max(width(stdev.tokens))
                .max(width(stdev.time_secs));
        }

        (max_avg, max_mean, max_var, max_stdev)
    }

    fn print_header(
        w: &mut impl std::io::Write,
        (max_avg, max_mean, max_var, max_stdev): (usize, usize, usize, usize),
    ) -> Result<(), std::io::Error> {
        let prefix_pad = Self::PREFIX_PAD;
        writeln!(
            w,
            "{:prefix_pad$}{:max_avg$.4} | {:max_mean$.4} | {:max_var$.4} | {:max_stdev$}",
            "", "avg", "mean", "var", "stdev",
        )?;
        writeln!(
            w,
            "------------------------------------------------------------"
        )?;
        Ok(())
    }

    // Updated print_sample to take the widths into account
    fn print_sample(
        w: &mut impl std::io::Write,
        name: &str,
        sample: &TokSamples,
        (max_avg, max_mean, max_var, max_stdev): (usize, usize, usize, usize),
    ) -> Result<(), std::io::Error> {
        let avg = sample.avg();
        let mean = sample.mean();
        let var = sample.var();
        let stdev = sample.stdev();
        writeln!(w, "[[{}]]", name)?;
        // For tok/s
        let tokps_pad = Self::PREFIX_PAD - "tok/s:".len();
        writeln!(
            w,
            "tok/s:{:tokps_pad$}{:max_avg$.4} | {:max_mean$.4} | {:max_var$.4} | {:max_stdev$.4}",
            "", avg.tok_per_s, mean.tok_per_s, var.tok_per_s, stdev.tok_per_s,
        )?;
        // For tokens
        let tokens_pad = Self::PREFIX_PAD - "tokens:".len();
        writeln!(
            w,
            "tokens:{:tokens_pad$}{:max_avg$.4} | {:max_mean$.4} | {:max_var$.4} | {:max_stdev$.4}",
            "", avg.tokens, mean.tokens, var.tokens, stdev.tokens,
        )?;
        // For seconds
        let secs_pad = Self::PREFIX_PAD - "seconds:".len();
        writeln!(
            w,
            "seconds:{:secs_pad$}{:max_avg$.4} | {:max_mean$.4} | {:max_var$.4} | {:max_stdev$.4}",
            "", avg.time_secs, mean.time_secs, var.time_secs, stdev.time_secs,
        )?;
        writeln!(
            w,
            "------------------------------------------------------------"
        )?;
        Ok(())
    }
}

pub struct Bench {
    // TODO: make backend configurable (only supporting Metal rn)
    backend: infernal::backend::Metal,
    // TODO: make model configurable (only supporting llama2 rn)
    llm: Llm<Llama2>,
    config: Config,
    stats: Stats,
    prompt: String,
}

impl Bench {
    pub fn new(config: Config) -> Result<Self, Error> {
        let mut stats = Stats::default();

        // TODO: make backend configurable based on platform or CLI flags
        let mut backend = infernal::backend::Metal::new()?;

        // Instantiate LLM based on config and measure load time.
        info!(?config.llm.model.path, path_exists = config.llm.model.path.exists(), "initializing model");
        let start = Instant::now();
        let llm = Llm::new(
            config.llm.clone(),
            Template::new_mistral_openorca(),
            "", // Do not use any system prompt for benching.
            &mut backend,
        )?;
        stats.init_time = start.elapsed();

        let prompt = config
            .llm
            .model
            .prompt
            .clone()
            .take()
            .ok_or_else(|| Error::InvalidConfig("prompt must be set"))?;

        Ok(Self {
            backend,
            llm,
            config,
            stats,
            prompt,
        })
    }
}

impl crate::Benchmark for Bench {
    type Stats = Stats;
    type Error = Error;

    fn config(&self) -> &ConfigBase {
        &self.config.base
    }

    fn verify_prestart(&mut self) -> Result<(), Self::Error> {
        let n_tokens = self.config.n_tokens();

        if n_tokens == 0 {
            return Err(Error::InvalidConfig("n_tokens must be > 0"));
        }
        if self.prompt.is_empty() {
            return Err(Error::InvalidConfig("prompt must not be empty"));
        }
        if self.prompt.len() >= self.config.llm.runner.window_len {
            return Err(Error::InvalidConfig(
                "prompt must be shorter than context length",
            ));
        }

        Ok(())
    }

    fn warmup_step(&mut self, _i: usize) -> Result<(), Self::Error> {
        self.llm.feed_prompt(&self.prompt)?;
        loop {
            if self.llm.infer_next(&mut self.backend)?.is_none() {
                break;
            }
        }
        Ok(())
    }

    fn bench_step(&mut self, _i: usize) -> Result<(), Self::Error> {
        self.llm.feed_prompt(&self.prompt)?;
        let mut n_toks = 0;
        // We output in one at the end to not interfere with per token log output when debugging.
        let mut output = String::new(); // TODO#perf: store in self for reuse, and prealloc buffer
        let mut utf8_ctor = Utf8Constructor::default();
        while let Some(tok) = self.llm.infer_next(&mut self.backend)? {
            if self.config.base.verbose {
                if let Some(tok) = utf8_ctor.get(tok) {
                    output += &tok;
                    debug!(tok);
                }
            }
            n_toks += 1;
            if n_toks >= self.config.n_tokens() {
                break;
            }
        }
        if n_toks == 0 {
            panic!("model did not generate any tokens");
        }
        if !output.is_empty() {
            println!();
            println!("{output}");
        }

        let stats = self.llm.stats();
        if self.config.base.verbose {
            trace!(?stats);
        }

        // Only consider this sample if it reached the minimum tokens.
        if self
            .config
            .min_tokens
            .map(|min| stats.n_eval > min)
            .unwrap_or(true)
        {
            self.stats
                .sample
                .insert(TokSample::new(stats.n_sample, stats.sample_time));
            if self.config.base.verbose {
                info!(
                    "sample tok/s = {:?}",
                    TokSample::new(stats.n_sample, stats.sample_time).tok_per_s()
                );
            }
            self.stats
                .feed
                .insert(TokSample::new(stats.n_prompt_eval, stats.prompt_eval_time));
            if self.config.base.verbose {
                info!(
                    "p_eval tok/s = {:?}",
                    TokSample::new(stats.n_prompt_eval, stats.prompt_eval_time).tok_per_s()
                );
            }
            self.stats
                .infer
                .insert(TokSample::new(stats.n_eval, stats.eval_time));
            if self.config.base.verbose {
                info!(
                    "eval tok/s = {:?}",
                    TokSample::new(stats.n_eval, stats.eval_time).tok_per_s()
                );
            }
        }

        // Reset all timings for the next round. We do this instead of just taking the cumulative
        // timings because we want to calculate statistics such as mean, standard deviation, etc,
        // which is only possible if we have multiple samples.
        self.llm.clear_stats();

        Ok(())
    }

    fn into_stats(mut self, total_time: Duration) -> Self::Stats {
        self.stats.total_time = total_time;
        self.stats
    }
}

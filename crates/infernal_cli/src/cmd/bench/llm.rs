use std::path::PathBuf;

use clap::Args;
use infernal_bench::{infernal_llm, ConfigBase};
use serde::Deserialize;

#[derive(Args, Debug)]
pub struct Options {
    /// LLM configuration absolute path
    #[clap(long)]
    pub llm_config_path: PathBuf,

    /// Max inference steps to run. Must be smaller than n_ctx.
    #[clap(short, long)]
    pub n_tokens: Option<usize>,

    /// Only consider samples that have at least this many tokens evaluated.
    #[clap(long)]
    pub min_tokens: Option<usize>,
}

pub(crate) fn run(base: ConfigBase, opt: Options) -> anyhow::Result<()> {
    println!(
        "Reading LLM benchmark config from {}",
        opt.llm_config_path.display()
    );

    let llm_config = std::fs::read_to_string(opt.llm_config_path)?;
    let llm_config: SimpleLlmConfig = toml::from_str(&llm_config)?;

    let config = infernal_bench::llm::Config {
        base,
        llm: llm_config.into(),
        n_tokens: opt.n_tokens,
        min_tokens: opt.min_tokens,
    };
    let reps = config.base.reps();

    println!("Running LLM benchmark...");

    let bench = infernal_bench::llm::Bench::new(config)?;
    let stats = infernal_bench::run(bench)?;

    println!("Measurements completed.");
    stats.print(reps, &mut std::io::stdout())?;

    Ok(())
}

/// Simplified config only containing the configuration switches that we care about for benchmarking
/// and which maps to the full config.
#[derive(Debug, Deserialize)]
struct SimpleLlmConfig {
    path: PathBuf,
    seed: Option<u32>,
    n_ctx: usize,
    n_batch: usize,
    n_threads: Option<usize>,
    n_gpu_layers: Option<usize>,
    prompt: String,
}

impl From<SimpleLlmConfig> for infernal_llm::Config {
    fn from(value: SimpleLlmConfig) -> Self {
        Self {
            runtime: infernal_llm::RuntimeConfig {
                threads: value.n_threads.unwrap_or_else(|| num_cpus::get()),
            },
            runner: infernal_llm::RunnerConfig {
                seed: value.seed,
                window_len: value.n_ctx,
                batch_len: value.n_batch,
                gpu_layers: value.n_gpu_layers,
                ignore_eos: true,
                ..Default::default()
            },
            model: {
                let mut m = infernal_llm::ModelConfig::new(value.path);
                m.prompt = Some(value.prompt);
                m
            },
            lora: Default::default(),
        }
    }
}

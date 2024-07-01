use std::time::Duration;

use clap::{Args, Subcommand};
use infernal_bench::ConfigBase;

pub mod llm;
pub mod matmul;

#[derive(Debug, Args)]
pub struct Options {
    /// How many times to run prompt feed + eval before benching.
    #[clap(long)]
    pub warmup_rounds: Option<usize>,

    /// How many samples to collect.
    #[clap(long)]
    pub reps: Option<usize>,

    /// How much time to delay between bench iteration steps.
    #[clap(long)]
    pub step_delay_ms: Option<u64>,

    #[command(subcommand)]
    pub cmd: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    Llm(llm::Options),
    Matmul(matmul::Options),
}

pub fn run(opt: Options, verbose: bool) -> anyhow::Result<()> {
    let base = ConfigBase {
        warmup_rounds: opt.warmup_rounds,
        reps: opt.reps,
        step_delay: opt.step_delay_ms.map(Duration::from_millis),
        verbose,
    };

    match opt.cmd {
        Commands::Llm(opt) => llm::run(base, opt),
        Commands::Matmul(opt) => matmul::run(base, opt),
    }
}

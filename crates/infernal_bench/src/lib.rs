use std::time::{Duration, Instant};

use tracing::{debug, trace};

pub use infernal_llm;

pub mod llm;
pub mod matmul;
pub mod sample;

#[derive(Clone, Debug, Default)]
pub struct ConfigBase {
    pub warmup_rounds: Option<usize>,
    pub reps: Option<usize>,
    pub step_delay: Option<Duration>,
    pub verbose: bool,
}

impl ConfigBase {
    pub fn warmup_rounds(&self) -> usize {
        self.warmup_rounds.unwrap_or(defaults::WARMUP_ROUNDS)
    }

    pub fn reps(&self) -> usize {
        self.reps.unwrap_or(defaults::REPS)
    }
}

pub mod defaults {
    pub const WARMUP_ROUNDS: usize = 1;
    pub const REPS: usize = 10;
}

pub trait Benchmark {
    type Stats;
    type Error: std::error::Error;

    fn config(&self) -> &ConfigBase;

    fn verify_prestart(&mut self) -> Result<(), Self::Error>;
    fn warmup_step(&mut self, i: usize) -> Result<(), Self::Error>;
    fn bench_step(&mut self, i: usize) -> Result<(), Self::Error>;

    fn into_stats(self, total_time: Duration) -> Self::Stats;
}

pub fn run<B: Benchmark>(mut bench: B) -> Result<B::Stats, B::Error> {
    let start = Instant::now();
    let config = bench.config();
    let warmup_rounds = config.warmup_rounds();
    let reps = config.reps();
    let step_delay = config.step_delay;
    debug!(?step_delay, warmup_rounds, reps, "running bench");

    trace!("verifying benchmark");
    bench.verify_prestart()?;

    trace!(warmup_rounds, "warmup");
    for round in 0..warmup_rounds {
        bench.warmup_step(round)?;
    }

    if let Some(step_delay) = step_delay {
        std::thread::sleep(step_delay);
    }

    trace!(reps, "beginning testing");
    for round in 0..reps {
        trace!(round, "running bench step");
        bench.bench_step(round)?;
        if let Some(step_delay) = step_delay {
            // This is to avoid being throttled on the iPhone.
            std::thread::sleep(step_delay);
        }
    }

    Ok(bench.into_stats(start.elapsed()))
}

pub(crate) fn print_stats_header(
    w: &mut impl std::io::Write,
    runs: usize,
    init_time: &Duration,
    total_time: &Duration,
) -> Result<(), std::io::Error> {
    writeln!(
        w,
        "DONE. {runs} run{runs_plural} in {:?} (init_time={:?})",
        total_time,
        init_time,
        runs_plural = if runs > 1 { "s" } else { "" },
    )
}

#[test]
fn debug() {
    let config = matmul::Config {
        base: ConfigBase {
            warmup_rounds: Some(1),
            reps: Some(1),
            ..Default::default()
        },
        n_threads: 8,
        // processor: matmul::Processor::Cpu,
        quant: matmul::Quantization::F16,
        sizex: None,
        sizey: None,
        sizez: None,
    };
    let backend = infernal::backend::Metal::new().unwrap();
    let bench = matmul::Bench::new(config, backend).unwrap();
    let stats = run(bench).unwrap();
    println!("{:#?}", stats);
}

use std::time::{Duration, Instant};

use clap::ValueEnum;
use infernal::{builder::TensorOptionsExt, Backend, ComputeGraph, TensorCtx, D2};
use thiserror::Error;
use tracing::info;

use crate::{sample::TimeSamples, ConfigBase};

pub use infernal::Type;

// TODO
// #[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
// pub enum Processor {
//     Cpu,
//     Gpu,
// }

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Quantization {
    F16,
    Q3_K,
    Q4_K,
}

impl std::fmt::Display for Quantization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_possible_value()
            .expect("no values are skipped")
            .get_name()
            .fmt(f)
    }
}

impl From<Quantization> for Type {
    fn from(value: Quantization) -> Self {
        match value {
            Quantization::F16 => Self::F16,
            Quantization::Q4_K => Self::Q4_K,
            Quantization::Q3_K => Self::Q3_K,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub base: ConfigBase,
    pub n_threads: usize,
    // pub processor: Processor,
    pub quant: Quantization,
    pub sizex: Option<usize>,
    pub sizey: Option<usize>,
    pub sizez: Option<usize>,
}

impl Config {
    pub fn sizex(&self) -> usize {
        self.sizex.unwrap_or(defaults::SIZEX)
    }

    pub fn sizey(&self) -> usize {
        self.sizey.unwrap_or(defaults::SIZEY)
    }

    pub fn sizez(&self) -> usize {
        self.sizez.unwrap_or(defaults::SIZEZ)
    }

    pub fn ctx_size(&self) -> usize {
        let sizex = self.sizex() as f32;
        let sizey = self.sizey() as f32;
        let sizez = self.sizez() as f32;
        let quant_size = Type::from(self.quant).size() as f32;
        let f32_size = Type::F32.size() as f32;

        let mut total = 0.0;
        // xy
        total += sizex * sizey * quant_size;
        // yz
        total += sizey * sizez * f32_size;
        // zx
        total += sizez * sizex * quant_size;
        // xz
        total += sizex * sizez * f32_size;
        // zz
        total += sizex * sizez * f32_size;

        total.ceil() as usize
    }
}

pub mod defaults {
    // All matrices with these sizes take up about 1.6GB memory.
    pub const SIZEX: usize = 11008;
    pub const SIZEY: usize = 4096;
    pub const SIZEZ: usize = 1024;
}

/// Stores measurement samples and provides methods for calculating statistics for tokens per
/// second.
#[derive(Default, Debug)]
pub struct Stats {
    pub init_time: Duration,
    pub total_time: Duration,
    pub timings: TimeSamples,
    // pub q3_k: Samples,
    // pub q4_k: Samples,
    // pub q4: Samples,
    // pub f16: Samples,
    // pub f32: Samples,
}

impl Stats {
    pub fn print(&self, runs: usize, w: &mut impl std::io::Write) -> Result<(), std::io::Error> {
        super::print_stats_header(w, runs, &self.init_time, &self.total_time)?;
        writeln!(
            w,
            "- timings: [avg={avg} mean={mean} var={var} stdev={stdev}]",
            avg = self.timings.avg(),
            mean = self.timings.mean(),
            var = self.timings.var(),
            stdev = self.timings.stdev(),
        )?;
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Oom(#[from] infernal::alloc::Oom),
    #[error(transparent)]
    BackendBuf(#[from] infernal::builder::BackendBufError),
    #[error(transparent)]
    TensorNotFound(#[from] infernal::TensorNotFound),
}

pub struct Bench<B: Backend> {
    config: Config,
    stats: Stats,
    graph: ComputeGraph,
    ctx: TensorCtx,
    backend: B,
}

impl<B: Backend> Bench<B> {
    pub fn new(config: Config, mut backend: B) -> Result<Self, Error> {
        let start = Instant::now();

        let ctx_size = config.ctx_size();
        info!(ctx_size, "allocating memory");

        let sizex = config.sizex();
        let sizey = config.sizey();
        let sizez = config.sizez();
        let quant = Type::from(config.quant);

        let mut builder = infernal::TensorBuilder::prealloc(ctx_size)?;

        // NB: ggml stores matrices in row-major order, so columns are specified first.
        // NB: The RHS of a matmul has to be F32 in ggml for some reason.
        let xy = builder.new_tensor(
            D2 {
                rows: sizex,
                cols: sizey,
            }
            .with_type(quant)
            .with_name("xy"),
        )?;
        if !quant.is_quantized() {
            builder.fill(xy, 1.3_f32);
        }

        let yz = builder.new_tensor(
            D2 {
                rows: sizez,
                cols: sizey,
            }
            .with_type(Type::F32)
            .with_name("yz"),
        )?;
        builder.fill(yz, 1.5_f32);

        let zx = builder.new_tensor(
            D2 {
                rows: sizez,
                cols: sizex,
            }
            .with_type(quant)
            .with_name("zx"),
        )?;
        if !quant.is_quantized() {
            builder.fill(zx, 2.1_f32);
        }

        let xz = builder.matmul(xy, yz)?;
        let zz = builder.matmul(zx, xz)?;

        info!("matmul chain: {xy}x{yz} => {xz}; {zx}x{xz} => {zz}");

        let (ctx, _) = builder.finish(&mut backend)?;
        let output = zz;
        let mut graph = ComputeGraph::default();
        graph.expand_forward(output, &ctx)?;
        graph.find_exec_plan(backend.mem_limit(), &ctx)?;

        let stats = Stats {
            init_time: start.elapsed(),
            ..Default::default()
        };

        Ok(Self {
            config,
            stats,
            graph,
            ctx,
            backend,
        })
    }

    fn step(&mut self) {
        let start = Instant::now();
        self.backend.compute_graph(&self.graph, &self.ctx).unwrap();
        self.stats.timings.insert(start.elapsed().into())
    }
}

impl<B: Backend> crate::Benchmark for Bench<B> {
    type Stats = Stats;
    type Error = Error; // TODO

    fn config(&self) -> &ConfigBase {
        &self.config.base
    }

    fn verify_prestart(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn warmup_step(&mut self, _i: usize) -> Result<(), Self::Error> {
        self.step();
        Ok(())
    }

    fn bench_step(&mut self, _i: usize) -> Result<(), Self::Error> {
        self.step();
        Ok(())
    }

    fn into_stats(mut self, total_time: Duration) -> Self::Stats {
        self.stats.total_time = total_time;
        self.stats
    }
}

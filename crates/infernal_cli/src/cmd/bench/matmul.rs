use clap::Args;
use infernal_bench::{matmul::Quantization, ConfigBase};

#[derive(Args, Debug)]
pub struct Options {
    /// The number of threads to use for computing the compute graph.
    #[clap(long)]
    pub n_threads: Option<usize>,

    // #[clap(long)]
    // pub processor: Processor,
    #[clap(long, default_value_t = Quantization::Q3_K)]
    pub quant: Quantization,

    /// TODO: doc
    #[clap(long)]
    pub sizex: Option<usize>,

    /// TODO: doc
    #[clap(long)]
    pub sizey: Option<usize>,

    /// TODO: doc
    #[clap(long)]
    pub sizez: Option<usize>,

    #[clap(long, default_value = "false")]
    pub all_tensors: bool,
    // TODO: other args, for now defaults will do
}

pub(crate) fn run(base: ConfigBase, opt: Options) -> anyhow::Result<()> {
    let config = infernal_bench::matmul::Config {
        base,
        n_threads: opt.n_threads.unwrap_or_else(num_cpus::get),
        // processor: opt.processor,
        sizex: opt.sizex,
        sizey: opt.sizey,
        sizez: opt.sizez,
        quant: opt.quant,
    };
    let reps = config.base.reps();

    println!("Running matmul benchmark...");

    let backend = infernal::backend::Metal::new()?;
    let bench = infernal_bench::matmul::Bench::new(config, backend)?;
    let stats = infernal_bench::run(bench)?;

    println!("Measurements completed.");
    stats.print(reps, &mut std::io::stdout())?;

    Ok(())
}

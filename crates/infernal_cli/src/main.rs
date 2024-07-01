use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, EnvFilter};

use crate::cmd::Commands;

mod cmd;

/// Infernal CLI
#[derive(Parser, Debug)]
#[command(name = "infernal_cli", about)]
struct Cli {
    /// Enable debug logging
    #[clap(short = 'v', long)]
    verbose: bool,

    /// Disable ANSI color codes in output (useful for piping to files)
    #[clap(long, default_value = "false")]
    noansi: bool,

    #[command(subcommand)]
    cmd: Commands,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if cli.verbose {
        let subscriber = tracing_subscriber::registry()
            .with(EnvFilter::from_default_env())
            .with(
                tracing_subscriber::fmt::layer()
                    .with_ansi(!cli.noansi)
                    .without_time()
                    .with_target(false)
                    .with_writer(std::io::stdout),
            );
        let _ = tracing::subscriber::set_global_default(subscriber);
    }

    match cli.cmd {
        Commands::Bench(cmd) => cmd::bench::run(cmd, cli.verbose),
    }
}

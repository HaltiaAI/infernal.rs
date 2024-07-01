use clap::Subcommand;

pub mod bench;
//pub mod chat; TODO

#[derive(Debug, Subcommand)]
pub enum Commands {
    Bench(bench::Options),
}

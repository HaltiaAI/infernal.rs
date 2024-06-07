use std::{fs::File, io::Write};

use infernal::graph::inspect::export_debug;
use llm::{arch::EvalInput, tokenizer::Tokenizer, Model};

const INPUT: &str = "why is the sky blue?";

fn main() -> anyhow::Result<()> {
    init_tracing();

    let path = dirs::home_dir()
        .expect("$HOME must be set")
        .join(".config/haltia/default.gguf");

    let params = llm::Params::default();
    let (model, tl) = llm::load_from_file::<llm::arch::Llama>(path, &params)?;
    println!("Loaded model with default parameters.");

    let mut ctx = llm::Context::<llm::arch::Llama>::new(model, params, tl)?;

    let mut tokens = Vec::new();
    let n_past = 0;
    ctx.model_mut().tokenizer_mut().tokenize(INPUT, &mut tokens);
    println!(
        "Initializing LLM with tokens={tokens:?} (n={}).",
        tokens.len()
    );
    ctx.eval(tokens.as_slice(), n_past, 8)?;
    println!(
        "Initialized LLM with tokens={tokens:?} (n={}).",
        tokens.len()
    );

    let comp_ctx = ctx.rebuild_compute_graph(&EvalInput::Tokens(&tokens), n_past)?;
    let tensor_ctx = ctx.tensor_src(&comp_ctx);
    let graph = infernal::graph::inspect::InfernalGraph::new(
        ctx.compute_graph(),
        &tensor_ctx,
        ctx.backend(),
    );

    println!(
        "Exporting infernal debug graph of {} nodes...",
        ctx.compute_graph().nodes().len()
    );

    // TODO: specify the output file path as a CLI arg
    let mut file = File::create("infernal-graph.json")?;
    let mut writer = std::io::BufWriter::new(&mut file);
    export_debug(&graph, &mut writer)?;

    writer.flush()?;

    println!("Successfully exported infernal debug graph.");

    Ok(())
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
}

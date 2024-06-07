use std::{fs::File, io::BufReader, path::Path};

use infernal::graph::inspect::{compare, GraphHeader, Node};
use serde::Deserialize;

fn main() -> anyhow::Result<()> {
    let infernal_path = Path::new("infernal-graph.json");
    println!(
        "Opening file: {infernal_path:?} (exists={})",
        infernal_path.exists()
    );
    // let infernal_data = std::fs::read(infernal_path)?;
    let infernal_reader = BufReader::new(File::open(infernal_path)?);

    let ggml_path = Path::new("ggml-graph.json");
    println!(
        "Opening file: {ggml_path:?} (exists={})",
        ggml_path.exists()
    );
    // let ggml_data = std::fs::read(ggml_path)?;
    let ggml_reader = BufReader::new(File::open(ggml_path)?);

    // infernal
    // -----
    let mut infernal_deser = serde_json::Deserializer::from_reader(infernal_reader);
    // let infernal_header = GraphHeader {
    //     n_nodes: infernal_deser.deserialize_usize()?,
    //     n_leaf_nodes: infernal_deser.deserialize_usize()?,
    // };
    println!("Deserializing infernal graph...");
    let infernal_header = GraphHeader::deserialize(&mut infernal_deser)?;
    let infernal_nodes = infernal_deser.into_iter::<Node>();
    // let actual_nodes: Vec<Node> = rmp_serde::from_read(&mut infernal_reader)?;
    // let infernal_graph: Vec<Node> = rmp_serde::from_slice(&infernal_data)?;
    // let infernal_graph: Vec<Node> = serde_json::from_slice(&infernal_data)?;

    // GGML
    // -----
    let mut ggml_deser = serde_json::Deserializer::from_reader(ggml_reader);
    println!("Deserializing ggml graph...");
    let ggml_header = GraphHeader::deserialize(&mut ggml_deser)?;
    // let ggml_header = GraphHeader {
    //     n_nodes: ggml_deser.deserialize_usize()?,
    //     n_leaf_nodes: ggml_deser.deserialize_usize()?,
    // };
    let ggml_nodes = ggml_deser.into_iter::<Node>();
    // let expected_nodes: Vec<Node> = rmp_serde::from_read(&mut ggml_reader)?;
    // let ggml_graph: Vec<Node> = rmp_serde::from_slice(&ggml_data)?;
    // let ggml_graph: Vec<Node> = serde_json::from_slice(&ggml_data)?;

    println!("Comparing graphs...");
    println!("------------------------------------------------");
    let cmp = compare(
        infernal_header,
        infernal_nodes,
        ggml_header,
        ggml_nodes,
        |node, idx| println!("{idx}. Node '{}' matches expectations.", node.name),
    )?;
    println!("------------------------------------------------");
    println!("{cmp}");

    Ok(())
}

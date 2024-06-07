use std::{fs::File, io::BufReader, path::Path};

use infernal::graph::inspect::{GraphHeader, Node};
use serde::Deserialize;

fn main() -> anyhow::Result<()> {
    let path = Path::new("infernal-graph.json");
    println!("Opening file: {path:?} (exists={})", path.exists());
    // let infernal_data = std::fs::read(infernal_path)?;
    let reader = BufReader::new(File::open(path)?);

    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };

    let mut total_diff = 0;
    let mut deser = serde_json::Deserializer::from_reader(reader);
    println!("Deserializing infernal graph...");
    let _ = GraphHeader::deserialize(&mut deser)?;
    let mut n_nodes = 0;
    let mut n_weights = 0;
    for node in deser.into_iter::<Node>() {
        let node = node?;
        for parent in node.parents {
            if parent.info.name.ends_with(".weight") {
                let size = node.info.layout.size();
                let size_diff = size % page_size;
                println!(
                    "Node '{name}' size: {size} (align diff: {size_diff})",
                    name = node.info.name,
                    size = size,
                    size_diff = size_diff
                );
                total_diff += size_diff;
                n_weights += 1;
            }
        }
        n_nodes += 1;
    }

    println!(
        "Total align diff for {n_weights}/{n_nodes} weights: {total_diff} bytes ({} MiB)",
        total_diff as f32 / 1024. / 1024.
    );

    Ok(())
}

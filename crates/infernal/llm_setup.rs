use std::{cell::RefCell, fs::File, rc::Rc};

use infernal::{
    alloc::graph::GraphAlloc, buf::Buf, builder::TensorOptionsExt as _, Backend as _, ComputeGraph,
    GetCtxId, Layout, TensorBuilder, Type::*, D1,
};

/// This example showcases how you would set up the various tensor contexts and tensors for a more
/// complicated use case, such as building up an LLM. It explains more advanced aspects of this
/// library.
fn main() -> anyhow::Result<()> {
    let n_ctx = 512;

    let mut backend = infernal::backend::Metal::new()?;
    // Doesn't load mmap, doesn't create buffers, buffer id is assigned later during finish when
    // mmap is materialized.
    // TODO: support mmap
    // let mut weights_tb = TensorBuilder::mmap("llama.gguf").with_name("weights");
    let mut weights_tb = TensorBuilder::mmap(File::open("llama.weights")?, 0)?.with_name("weights");
    let embd_w =
        weights_tb.new_mapped_tensor(Layout::new_2d(n_ctx, n_ctx, F32).with_name("embd_L"), 0)?;
    // TODO: alternative API - which is better?
    // let embd_l = weights_tb.new_tensor(
    //     Layout::new_2d(n_ctx, n_ctx, F32)
    //         .with_name("embd_L")
    //         .with_mmap_offs(2048),
    // )?;
    // .. map all tensors
    // NOTE: do *not* finish weights_tb yet! We need to optimize the graph first to finalize the
    // mmap buffer.

    // Build worst case graph
    let mut alloc = GraphAlloc::measure();
    let mut compute_tb = TensorBuilder::graph_alloc_with_src(&mut alloc, &weights_tb);
    let input_tokens = compute_tb.new_tensor(Layout::new_1d(n_ctx, F32).with_name("inp_tokens"))?;
    // .. create all tensors
    let output_norm = compute_tb.new_tensor(Layout::new_1d(n_ctx, F32).with_name("output_norm"))?;

    // Optimize graph. We want to keep the graph as it won't change later on.
    let mut graph = ComputeGraph::default();
    graph.expand_forward(output_norm, &compute_tb).unwrap();
    let ios_mem_limit = 4 * 1024 * 1024 * 1024; // 4 GB
    graph.find_exec_plan(ios_mem_limit, &compute_tb)?;
    let exec_plan = graph.exec_plan(ios_mem_limit).unwrap();
    println!("execution plan: {exec_plan:#?}");

    let worst_case_compute_size = compute_tb.alloc_graph(&graph, ios_mem_limit)?;

    // This compute builder is not finalized here, since it was used merely for measurement.
    // As an optimization, we want to reuse some of the internal parts for when building up the
    // actual compute graph below.
    let compute_tb_parts = compute_tb.into_raw_parts();

    // Find splitting point for mmap, at or below 80% of file size. Only available for mmap based
    // builder. This is optional. Later on we can add interesting optimal splitting strategies here.
    // TODO
    // weights_tb.find_split_points(
    //     exec_plan,
    //     &[
    //         SplitPoint {
    //             threshold: 0.2,
    //             pin: false,
    //         },
    //         SplitPoint {
    //             threshold: 0.8,
    //             pin: true,
    //         },
    //     ],
    // );
    // This also marks the first buffer as PinBuf which means backend won't unmap it ever from RAM.
    // weights_tb.split_mmap(split_points);
    // Now we can finish this. This is when the buf ids for the tensors are allocated, when the mmap
    // is loaded in finish.
    let (weights_ctx, _) = weights_tb.finish(&mut backend)?;

    let kv_tb = TensorBuilder::prealloc(1024 * 1024)?.with_name("kv");
    // .. create all necessary tensors
    let (kv_ctx, _) = kv_tb.finish(&mut backend)?;

    // Later on during inferencing:
    // Keep reusing the same compute buffer housing the intermediate nodes. It's allocated to the
    // worst case size so most of the time e.g. during auto-regressive inferencing it won't use most
    // of the buffer. That's fine.
    let compute_buf = Rc::new(RefCell::new(Buf::new(worst_case_compute_size)?));
    // We use an Rc and RefCell here so we can reuse the buffer later on.
    // backend.add_buf(Rc::clone(&compute_buf), "compute")?;

    // Later on in the inferencingb loop, this is how you can reuse the buffer:
    let mut compute_buf = compute_buf.borrow_mut();
    let mut alloc = GraphAlloc::new(&mut compute_buf).with_raw_parts(alloc.into_raw_parts());
    let mut compute_tb = TensorBuilder::graph_alloc_with_src(&mut alloc, [&weights_ctx, &kv_ctx])
        .with_raw_parts(compute_tb_parts)
        .with_name("compute");

    let tokens = vec![0, 1, 2, 3, 4];
    // ... build up the same compute graph as above
    let input_tokens = compute_tb.new_tensor(
        D1 { cols: n_ctx }
            .with_type(I32)
            .with_name("inp_tokens")
            // This tensor is preallocated so we can copy data into it before allocating the entire
            // graph.
            .prealloc(),
    )?;
    compute_tb.copy_data_from(input_tokens, &tokens);
    // .. create all tensors
    let output_tensor =
        compute_tb.new_tensor(Layout::new_1d(n_ctx, F32).with_name("output_norm"))?;

    // Need to reallocate the graph to recompute tensor sizes due to the changed embd width.
    compute_tb.alloc_graph(&graph, ios_mem_limit);

    Ok(())
}

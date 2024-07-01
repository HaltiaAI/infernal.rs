use infernal::{
    builder::TensorOptionsExt as _, Backend as _, ComputeGraph, Layout, TensorBuilder, Type, D2,
};

/// Performs a simple matrix multiplication chain, showcasing Infernal's basic primitives.
fn main() -> anyhow::Result<()> {
    let mut backend = infernal::backend::Metal::new()?;

    // A tensor builder is used to create tensors and tensor operations. It's a temporary
    // structure that, when building is finished, converts tensors into a finalized read-only
    // graph.
    let buf_size = 1024 * 1024;
    let mut tb = TensorBuilder::prealloc(buf_size)?;

    let layout = Layout::new(D2 { cols: 12, rows: 12 }, Type::F32);
    let x = tb.new_tensor(layout)?;
    let y = tb.new_tensor(layout.with_name("y"))?;
    let z = tb.new_tensor(layout.with_name("z"))?;
    tb.fill(x, 1.3_f32);
    tb.fill(y, 9.5_f32);
    tb.fill(z, 8.3_f32);

    // An operation creates a new tensor, but no computation takes place until later. You can
    // chain as many of these lazy ops as you want, they will be computed later.
    let xy = tb.matmul(x, y)?;
    let output = tb.matmul(xy, z)?;

    // Finishing the build means the resulting tensor context becomes read-only from here
    // on, and the builder loads the tensor data buffer onto the accelerator backend.
    let (ctx, _) = tb.finish(&mut backend)?;

    // Proof that tensor is lazy: data is still all zero.
    assert!(ctx[output]
        .elements::<f32>(&backend)
        .iter()
        .all(|&x| x == 0.0));

    // A computation graph is used to plan how to compute the final output tensor and to find
    // optimizations using what amounts to an intermediate representation.
    let mut graph = ComputeGraph::default();
    // The graph needs to be expanded from the last output node.
    graph.expand_forward(output, &ctx)?;
    // Further, an execution plan is needed so that the backend knows how to calculate the graph.
    // These two steps have to be done once only, assuming the graph doesn't change later on.
    graph.find_exec_plan(backend.mem_limit(), &ctx)?;

    // Important! This is where all tensors are computed and materialized.
    backend.compute_graph(&graph, &ctx)?;

    // All tensors are now materialized and can be read from.
    let n_rows = layout.rows() as f32;
    assert_eq!(ctx[xy].elements::<f32>(&backend)[0], 1.3 * 9.5 * n_rows);
    assert_eq!(
        ctx[output].elements::<f32>(&backend)[0],
        1.3 * 9.5 * n_rows * 8.3 * n_rows
    );

    Ok(())
}

use infernal::{
    builder::{AutoOpName, NoSrc, TensorOptionsExt as _},
    Backend as _, ComputeGraph, Layout, TensorBuilder, Type,
};

/// Performs a simple matrix multiplication chain, showcasing Infernal's basic primitives.
fn main() -> anyhow::Result<()> {
    // The same layout is used for all input tensors.
    let layout = Layout::new_2d(12, 12, Type::F32);
    let mut backend = infernal::backend::Metal::new()?;

    // A tensor builder is used to create tensors and tensor operations. It's a temporary
    // structure that, when building is finished, converts tensors into a finalized read-only
    // graph.
    let buf_size = 1024 * 1024;
    let mut tb = TensorBuilder::<_, NoSrc>::prealloc(buf_size)?
        // Naming the builder is optional, but useful for debugging as it associates tensor
        // bufferes more clearly with a given context, which is useful when an app has multiple
        // contexts.
        .with_name("matmul_example")
        // This is useful for debugging: settings this will automatically name operations with
        // their operand (source) tensors.
        .with_auto_op_names(AutoOpName::Expanded { with_id: false });

    // When creating a tensor, the builder will only return the tensor id and not the actual
    // tensor. Tensor data is stored in its TensorCtx (see below).
    // To retrieve the `Tensor`, call `tb.get_tensor(id)` or the same method on the tensor
    // context.
    // This is an optimization to avoid copying heavy objects (or using expensive arc pointer
    // clones), since tensors are created in the hot loop of an LLM inference engine. Regular
    // pointers don't work well with Rust's ownership model.
    let x = tb.new_tensor(layout)?;
    // You can also set optional information like the tensor name. Names are useful for
    // debugging e.g. when printing the compute graph. It uses CoW under the hood, therefore
    // there is no overhead for static strings or when no name is set.
    let y = tb.new_tensor(layout.with_name("y"))?;
    let z = tb.new_tensor(layout.with_name("z"))?;
    // Fill all tensor elements with the same value for simplicity.
    tb.fill_2d_f32(x, 1.3);
    tb.fill_2d_f32(y, 9.5);
    tb.fill_2d_f32(y, 8.3);

    // An operation creates a new tensor, but no copmutation takes place until later. You can
    // chain as many of these lazy ops as you want, they will all be computed in a single batch.
    let intermediate = tb.matmul(x, y)?;
    let result = tb.matmul(intermediate, z)?;

    // Finishing the build means the resulting tensor context becomes read-only from here
    // on, and the builder loads the tensor data buffer onto the accelerator backend.
    let (ctx, _) = tb.finish(&mut backend)?;

    // A computation graph is used to plan how to compute the final output tensor and to find
    // optimizations using what amounts to an intermediate representation.
    let mut gf = ComputeGraph::default();
    // The graph needs to be expanded from the last output node -- in this case the `result` tensor.
    gf.expand_forward(result, &ctx)?;
    // Further, an execution plan is needed so that the backend knows how to calculate the graph.
    // This has to be done manually here, by the user, because in most applications the computation
    // graph is not expected to change. Therefore, to increase performance, it is calculated once at
    // startup and reused throughout the remainder of the application.
    gf.find_exec_plan(backend.mem_limit(), &ctx)?;
    // The backend then walks this graph using the optimal strategy found in the execution plan.
    backend.compute_graph(&gf, &ctx)?;

    // The output tensor `res` is now materialized and can be read from.
    let res = &ctx[result];
    // We can then inspect the elements. Note that this is only supported for f16 and f32 types as
    // we need to specify the type to cast to. Later on quantized blocks may be queried as well.
    assert_eq!(
        res.elements::<f32>(&backend)[0],
        1.3 * 9.5 * res.rows() as f32
    );

    Ok(())
}

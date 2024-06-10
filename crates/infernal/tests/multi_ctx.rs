//! Tests for using multiple [`TensorCtx`](infernal::TensorCtx)'s for operations. This more closely
//! resembles a real world use case.

use infernal::{Backend as _, ComputeGraph, GetCtxId, Layout, TensorBuilder, Type::*};

use crate::common::{assert_elements, init_tracing, KB};

mod common;

#[test]
fn multi_ctx_add() -> anyhow::Result<()> {
    init_tracing();

    let layout = Layout::new_2d(12, 12, F32);
    let mut backend = infernal::backend::Metal::new()?;

    let (src1_ctx, x) = {
        let mut tb = TensorBuilder::prealloc(KB)?.with_name("src1");
        let x = tb.new_tensor(layout)?;
        tb.fill_2d_f32(x, 1.3);
        let (ctx, _) = tb.finish(&mut backend)?;
        (ctx, x)
    };
    println!("src1 id: {}", src1_ctx.id());

    let (src2_ctx, y) = {
        let mut tb = TensorBuilder::prealloc(KB)?.with_name("src2");
        let y = tb.new_tensor(layout)?;
        tb.fill_2d_f32(y, 9.5);
        let (ctx, _) = tb.finish(&mut backend)?;
        (ctx, y)
    };
    println!("src2 id: {}", src2_ctx.id());

    let mut tb = TensorBuilder::prealloc_with_src(KB, Default::default())?
        .with_name("compute")
        .with_src(&src1_ctx)
        .with_src(&src2_ctx);
    let output = tb.add(x, y)?;

    let expected = tb.new_tensor(layout)?;
    tb.fill_2d_f32(expected, 1.3 + 9.5);

    let mut graph = ComputeGraph::default();
    graph.expand_forward(output, &tb)?;
    graph.find_exec_plan(backend.mem_limit(), &tb)?;

    let (compute_ctx, _) = tb.finish(&mut backend)?;
    println!("compute id: {}", compute_ctx.id());

    backend.compute_graph(&graph, &[&src1_ctx, &src2_ctx, &compute_ctx])?;

    assert_elements(&compute_ctx, output, expected, &backend);

    Ok(())
}

#[test]
fn multi_ctx_chain_add_matmul() -> anyhow::Result<()> {
    let layout = Layout::new_2d(12, 12, F32);
    let mut backend = infernal::backend::Metal::new()?;

    let (src1_ctx, x) = {
        let mut tb = TensorBuilder::prealloc(KB)?.with_name("src1");
        let x = tb.new_tensor(layout)?;
        tb.fill_2d_f32(x, 1.3);
        let (ctx, _) = tb.finish(&mut backend)?;
        (ctx, x)
    };

    let (src2_ctx, y) = {
        let mut tb = TensorBuilder::prealloc(KB)?.with_name("src2");
        let y = tb.new_tensor(layout)?;
        tb.fill_2d_f32(y, 9.5);
        let (ctx, _) = tb.finish(&mut backend)?;
        (ctx, y)
    };

    let mut tb = TensorBuilder::prealloc_with_src(KB, Default::default())?
        .with_src(&src1_ctx)
        .with_src(&src2_ctx)
        .with_name("compute");
    let z = tb.add(x, y)?;
    let res = tb.matmul(z, y)?;

    let expected = tb.new_tensor(layout)?;
    tb.fill_2d_f32(expected, (1.3 + 9.5) * 9.5 * 12.0);

    let mut gf = ComputeGraph::default();
    gf.expand_forward(res, &tb)?;
    gf.find_exec_plan(backend.mem_limit(), &tb)?;

    let (compute_ctx, _) = tb.finish(&mut backend)?;

    backend.compute_graph(&gf, &[&src1_ctx, &src2_ctx, &compute_ctx])?;

    assert_elements(&compute_ctx, res, expected, &backend);

    Ok(())
}

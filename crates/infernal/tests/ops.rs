//! Tests for individual operations.

use infernal::{
    builder::{AutoOpName, TensorOptionsExt as _},
    Backend as _, ComputeGraph, Layout, TensorBuilder,
    Type::*,
};

use crate::common::{assert_elements, init_tracing, KB};

mod common;

#[test]
fn square_matrices_matmul() -> anyhow::Result<()> {
    init_tracing();

    let layout = Layout::new_2d(12, 12, F32);
    let mut backend = infernal::backend::Metal::new()?;

    let mut tb = TensorBuilder::prealloc(KB)?
        .with_name("matmul_even")
        .with_auto_op_names(AutoOpName::Expanded { with_id: false });

    let x = tb.new_tensor(layout.with_name("x"))?;
    let y = tb.new_tensor(layout.with_name("y"))?;
    tb.fill_2d_f32(x, 1.3);
    tb.fill_2d_f32(y, 9.5);

    let expected = tb.new_tensor(layout)?;
    tb.fill_2d_f32(expected, 1.3 * 9.5 * 12.0);

    let res = tb.matmul(x, y)?;

    let (ctx, _) = tb.finish(&mut backend)?;

    let mut gf = ComputeGraph::default();
    gf.expand_forward(res, &ctx)?;
    gf.find_exec_plan(backend.mem_limit(), &ctx)?;
    backend.compute_graph(&gf, &ctx)?;

    assert_elements(&ctx, res, expected, &backend);

    Ok(())
}

#[test]
fn uneven_matrices_matmul() -> anyhow::Result<()> {
    init_tracing();

    let mut backend = infernal::backend::Metal::new()?;

    let mut tb = TensorBuilder::prealloc(KB)?
        .with_name("matmul_even")
        .with_auto_op_names(AutoOpName::Expanded { with_id: false });

    // t!(tb =>
    //     { x = [15, 8] = 1.3 },
    //     { y = [12, 8] = 9.5 },
    //     { expected = [15, 12] = 1.3 * 9.5 * 8. },
    // );

    let x = tb.new_tensor(Layout::new_2d(12, 8, F32).with_name("x"))?;
    tb.fill_2d_f32(x, 1.3);
    common::print_tensor_meta(&tb[x]);
    // NB: The ggml kernel requires that we pass the rhs of the matmul as a transposed matrix. Not
    // sure why! I want to undo this but will require changing the kernel probably (or maybe just
    // transposing the layout before dispatching the kernel).
    let y = tb.new_tensor(Layout::new_2d(15, 8, F32).with_name("y"))?;
    tb.fill_2d_f32(y, 9.5);
    common::print_tensor_meta(&tb[y]);

    let expected = tb.new_tensor(Layout::new_2d(15, 12, F32))?;
    tb.fill_2d_f32(expected, 1.3 * 9.5 * 8.);

    let res = tb.matmul(x, y)?;

    let (ctx, _) = tb.finish(&mut backend)?;

    let mut gf = ComputeGraph::default();
    gf.expand_forward(res, &ctx)?;
    gf.find_exec_plan(backend.mem_limit(), &ctx)?;
    backend.compute_graph(&gf, &ctx)?;

    common::print_tensor_meta(&ctx[res]);

    assert_elements(&ctx, res, expected, &backend);

    Ok(())
}

#[test]
fn add() -> anyhow::Result<()> {
    init_tracing();

    let layout = Layout::new_2d(12, 12, F32);
    let mut backend = infernal::backend::Metal::new()?;

    let mut tb = TensorBuilder::prealloc(KB)?.with_name("add_even");
    let x = tb.new_tensor(layout)?;
    let y = tb.new_tensor(layout)?;
    tb.fill_2d_f32(x, 1.3);
    tb.fill_2d_f32(y, 9.5);

    let expected = tb.new_tensor(layout)?;
    tb.fill_2d_f32(expected, 1.3 + 9.5);

    let res = tb.add(x, y)?;

    let (ctx, _) = tb.finish(&mut backend)?;

    let mut gf = ComputeGraph::default();
    gf.expand_forward(res, &ctx)?;
    gf.find_exec_plan(backend.mem_limit(), &ctx)?;
    dbg!(gf.exec_plan(backend.mem_limit()));
    backend.compute_graph(&gf, &ctx)?;

    assert_elements(&ctx, res, expected, &backend);

    Ok(())
}

//! Tests that we can create arbitrary operation chains that are batch executed in one go later.

use infernal::{Backend as _, ComputeGraph, Layout, TensorBuilder, Type::*};

use crate::common::{assert_elements, init_tracing, KB};

mod common;

#[test]
fn chain_add_matmul() -> anyhow::Result<()> {
    init_tracing();

    let layout = Layout::new_2d(12, 12, F32);
    let mut backend = infernal::backend::Metal::new()?;

    let mut tb = TensorBuilder::prealloc(KB)?.with_name("add_matmul_chain");
    let x = tb.new_tensor(layout)?;
    let y = tb.new_tensor(layout)?;
    tb.fill(x, 1.3);
    tb.fill(y, 9.5);

    let expected = tb.new_tensor(layout)?;
    tb.fill(expected, (1.3 + 9.5) * 9.5 * 12.0);

    let z = tb.add(x, y)?;
    let res = tb.matmul(z, y)?;

    let (ctx, _) = tb.finish(&mut backend)?;

    let mut gf = ComputeGraph::default();
    gf.expand_forward(res, &ctx)?;
    gf.find_exec_plan(backend.mem_limit(), &ctx)?;
    backend.compute_graph(&gf, &ctx)?;

    assert_elements(&ctx, res, expected, &backend);

    Ok(())
}

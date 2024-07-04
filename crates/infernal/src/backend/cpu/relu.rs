use crate::buf::SliceCast;

pub(crate) fn relu(src: &[u8], dst: &mut [u8]) {
    let src = src.cast::<f32>();
    let dst = dst.cast_mut::<f32>();

    for (src_val, dst_val) in src.iter().zip(dst.iter_mut()) {
        *dst_val = if *src_val > 0.0 { *src_val } else { 0.0 };
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        backend::cpu::Cpu,
        builder::{TensorBuilder, TensorOptionsExt},
        layout::{Type, D2},
        Backend, ComputeGraph,
    };
    use approx::assert_relative_eq;

    #[test]
    fn test_relu_cpu() -> anyhow::Result<()> {
        let mut backend = Cpu::new();
        let mut tb = TensorBuilder::prealloc(256 * 256)?;

        let input_data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];

        let input = tb.new_tensor(D2 { rows: 1, cols: 5 }.with_type(Type::F32))?;

        tb.copy_data_from(input, &input_data);

        let relu = tb.relu(input)?;

        let (ctx, _) = tb.finish(&mut backend)?;
        let mut graph = ComputeGraph::default();
        graph.expand_forward(relu, &ctx)?;
        graph.find_exec_plan(backend.mem_limit(), &ctx)?;
        backend.compute_graph(&graph, &ctx)?;

        let output = ctx[relu].elements::<f32>(&backend);
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];

        for (a, b) in output.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }

        Ok(())
    }
}

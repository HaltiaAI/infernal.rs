use crate::{buf::SliceCast, Tensor, Type};

pub(crate) fn rms_norm(src_tensor: &Tensor, src: &[u8], dst: &mut [u8], eps: f32) {
    debug_assert!(src_tensor.typ() == Type::F32);

    let src = src.cast::<f32>();
    let dst = dst.cast_mut::<f32>();

    let rows = src_tensor.rows();
    let cols = src_tensor.cols();

    // rms_norm(x) = x / sqrt(sum(x^2) / n + eps)
    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let row_src = &src[start..end];
        let row_dst = &mut dst[start..end];

        // Calculate sum
        let mut sum = 0.0;
        for val in row_src.iter() {
            sum += val.powi(2)
        }

        // Normalize and apply eps
        sum = (sum / cols as f32 + eps).sqrt();

        for (i, &val) in row_src.iter().enumerate() {
            row_dst[i] = val / sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::{
        backend::Backend,
        builder::{TensorBuilder, TensorOptionsExt},
        layout::{Type, D2},
    };
    use approx::assert_relative_eq;

    #[test]
    fn rms_norm_cpu() -> anyhow::Result<()> {
        let mut backend = Cpu::new();
        let mut tb = TensorBuilder::prealloc(1024 * 1024)?;

        // Test case 1: Simple 1D softmax
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input = tb.new_tensor(D2 { rows: 1, cols: 4 }.with_type(Type::F32))?;
        tb.copy_data_from(input, &input_data);

        let rms = tb.rms_norm(input, 1.0)?;

        let (ctx, _) = tb.finish(&mut backend)?;
        let mut graph = ComputeGraph::default();
        graph.expand_forward(rms, &ctx)?;
        graph.find_exec_plan(backend.mem_limit(), &ctx)?;
        backend.compute_graph(&graph, &ctx)?;

        let result = ctx[rms].elements::<f32>(&backend);
        let expected = vec![0.34299716, 0.6859943, 1.0289915, 1.3719887];

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }

        Ok(())
    }

    #[test]
    fn rms_norm_2d_cpu() -> anyhow::Result<()> {
        let mut backend = Cpu::new();
        let mut tb = TensorBuilder::prealloc(1024 * 1024)?;

        // Test case 2: 2D softmax
        let mut tb = TensorBuilder::prealloc(1024 * 1024)?;
        let input_data = vec![1.0f32, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 6.0];
        let input = tb.new_tensor(D2 { rows: 2, cols: 4 }.with_type(Type::F32))?;
        tb.copy_data_from(input, &input_data);

        let rms = tb.rms_norm(input, 1.0)?;

        let (ctx, _) = tb.finish(&mut backend)?;

        let mut graph = ComputeGraph::default();
        graph.expand_forward(rms, &ctx)?;
        graph.find_exec_plan(backend.mem_limit(), &ctx)?;
        backend.compute_graph(&graph, &ctx)?;

        let result = ctx[rms].elements::<f32>(&backend);
        let expected = vec![
            0.38490018, 0.76980036, 1.1547005, 1.1547005, 0.73960024, 0.9245003, 1.1094004,
            1.1094004,
        ];

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }

        Ok(())
    }
}

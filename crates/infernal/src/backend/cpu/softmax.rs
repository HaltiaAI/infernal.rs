use crate::{buf::SliceCast, Tensor, Type};

pub(crate) fn softmax(src_tensor: &Tensor, src: &[u8], dst: &mut [u8], scale: f32) {
    debug_assert!(src_tensor.typ() == Type::F32);

    let src = src.cast::<f32>();
    let dst = dst.cast_mut::<f32>();

    let rows = src_tensor.rows();
    let cols = src_tensor.cols();

    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let row_src = &src[start..end];
        let row_dst = &mut dst[start..end];

        // Find max value in the row
        let max_val = row_src.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Calculate exp and sum
        let mut sum = 0.0;
        for (i, &val) in row_src.iter().enumerate() {
            let exp_val = ((val - max_val) * scale).exp();
            row_dst[i] = exp_val;
            sum += exp_val;
        }

        // Normalize
        for val in row_dst.iter_mut() {
            *val /= sum;
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

    fn create_backend() -> impl Backend {
        cfg_if::cfg_if! {
            if #[cfg(feature = "test-cpu-with-metal")] {
                crate::backend::Metal::new().unwrap()
            } else {
                Cpu::new()
            }
        }
    }

    #[test]
    fn softmax_1d() -> anyhow::Result<()> {
        let mut backend = create_backend();
        let mut tb = TensorBuilder::prealloc(1024 * 1024)?;

        // Test case 1: Simple 1D softmax
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input = tb.new_tensor(D2 { rows: 1, cols: 4 }.with_type(Type::F32))?;

        tb.copy_data_from(input, &input_data);

        let softmax = tb.softmax(input)?;

        let (ctx, _) = tb.finish(&mut backend)?;
        let mut graph = ComputeGraph::default();
        graph.expand_forward(softmax, &ctx)?;
        graph.find_exec_plan(backend.mem_limit(), &ctx).unwrap();
        backend.compute_graph(&graph, &ctx)?;

        let result = ctx[softmax].elements::<f32>(&backend);
        let expected = vec![0.0320586, 0.08714432, 0.23688282, 0.64391426];

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }

        Ok(())
    }

    #[test]
    fn softmax_2d() -> anyhow::Result<()> {
        let mut backend = create_backend();
        let mut tb = TensorBuilder::prealloc(1024 * 1024)?;

        // Test case 2: 2D softmax
        let mut tb = TensorBuilder::prealloc(1024 * 1024)?;
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = tb.new_tensor(D2 { rows: 2, cols: 3 }.with_type(Type::F32))?;

        tb.copy_data_from(input, &input_data);

        let softmax = tb.softmax(input)?;

        let (ctx, _) = tb.finish(&mut backend)?;

        let mut graph = ComputeGraph::default();
        graph.expand_forward(softmax, &ctx)?;
        graph.find_exec_plan(backend.mem_limit(), &ctx)?;
        backend.compute_graph(&graph, &ctx)?;

        let result = ctx[softmax].elements::<f32>(&backend);
        let expected = vec![
            0.090030566,
            0.24472848,
            0.665241,
            0.090030566,
            0.24472848,
            0.665241,
        ];

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }

        Ok(())
    }
}

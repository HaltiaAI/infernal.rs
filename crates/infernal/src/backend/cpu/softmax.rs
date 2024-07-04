use crate::{buf::SliceCast, Tensor};

pub(crate) fn softmax(input_tensor: &Tensor, input: &[u8], output: &mut [u8], scale: f32) {
    let input = input.cast::<f32>();
    let output = output.cast_mut::<f32>();

    let rows = input_tensor.rows();
    let cols = input_tensor.cols();

    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let row_input = &input[start..end];
        let row_output = &mut output[start..end];

        // Find max value in the row
        let max_val = row_input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Calculate exp and sum
        let mut sum = 0.0;
        for (i, &val) in row_input.iter().enumerate() {
            let exp_val = ((val - max_val) * scale).exp();
            row_output[i] = exp_val;
            sum += exp_val;
        }

        // Normalize
        for val in row_output.iter_mut() {
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

    fn create_cpu_backend() -> Cpu {
        Cpu::new()
    }

    #[test]
    fn softmax_cpu() {
        let mut backend = create_cpu_backend();
        let mut tb = TensorBuilder::prealloc(1024 * 1024).unwrap();

        // Test case 1: Simple 1D softmax
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input = tb
            .new_tensor(D2 { rows: 1, cols: 4 }.with_type(Type::F32))
            .unwrap();
        tb.copy_data_from(input, &input_data);

        let softmax = tb.softmax(input).unwrap();

        let (ctx, _) = tb.finish(&mut backend).unwrap();
        let mut graph = ComputeGraph::default();
        graph.expand_forward(softmax, &ctx).unwrap();
        graph.find_exec_plan(backend.mem_limit(), &ctx).unwrap();
        backend.compute_graph(&graph, &ctx).unwrap();

        let result = ctx[softmax].elements::<f32>(&backend);
        let expected = vec![0.0320586, 0.08714432, 0.23688282, 0.64391426];

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }

        // Test case 2: 2D softmax
        let mut tb = TensorBuilder::prealloc(1024 * 1024).unwrap();
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = tb
            .new_tensor(D2 { rows: 2, cols: 3 }.with_type(Type::F32))
            .unwrap();
        tb.copy_data_from(input, &input_data);

        let softmax = tb.softmax(input).unwrap();

        let (ctx, _) = tb.finish(&mut backend).unwrap();

        let mut graph = ComputeGraph::default();
        graph.expand_forward(softmax, &ctx).unwrap();
        graph.find_exec_plan(backend.mem_limit(), &ctx).unwrap();
        backend.compute_graph(&graph, &ctx).unwrap();

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
    }
}

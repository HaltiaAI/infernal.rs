use crate::buf::SliceCast;

pub(crate) fn silu(src: &[u8], dst: &mut [u8]) {
    let src = src.cast::<f32>();
    let dst = dst.cast_mut::<f32>();

    // f(x) = x * sigmoid(x), where sigmoid(x) = 1 / (1 + exp(-x))
    for (src_val, dst_val) in src.iter().zip(dst.iter_mut()) {
        *dst_val = src_val * (1. / (1. + (-src_val).exp()));
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
    fn test_silu_cpu() -> anyhow::Result<()> {
        let mut backend = create_backend();
        let mut tb = TensorBuilder::prealloc(1024 * 1024)?;

        let input_data = vec![-2.0f32, -1.0, 0.0, 1.0];

        let input = tb.new_tensor(D2 { rows: 1, cols: 4 }.with_type(Type::F32))?;

        tb.copy_data_from(input, &input_data);

        let silu = tb.silu(input)?;

        let (ctx, _) = tb.finish(&mut backend)?;

        let graph = ComputeGraph::forward_expand(backend.mem_limit(), silu, &ctx)?;
        backend.compute_graph(&graph, &ctx)?;

        let output = ctx[silu].elements::<f32>(&backend);
        let expected = vec![-0.23840587, -0.26894143, 0.0, 0.7310586];

        for (a, b) in output.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }

        Ok(())
    }
}

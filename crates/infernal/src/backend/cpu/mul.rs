use crate::{buf::SliceCast, Tensor};

pub(crate) fn mul(
    src0_tensor: &Tensor,
    src1_tensor: &Tensor,
    dst_tensor: &Tensor,
    src0: &[u8],
    src1: &[u8],
    dst: &mut [u8],
) {
    let src0 = src0.cast::<f32>();
    let src1 = src1.cast::<f32>();
    let dst = dst.cast_mut::<f32>();

    let [src0_cols, src0_rows, src0_depth, _] = &src0_tensor.dims;
    let [src1_cols, src1_rows, src1_depth, _] = &src1_tensor.dims;
    let [dst_cols, dst_rows, dst_depth, _] = &dst_tensor.dims;

    for i in 0..*dst_cols {
        for j in 0..*dst_rows {
            for k in 0..*dst_depth {
                let a_idx = (i % src0_cols) * (src0_rows * src0_depth)
                    + (j % src0_rows) * src0_depth
                    + (k % src0_depth);

                let b_idx = (i % src1_cols) * (src1_rows * src1_depth)
                    + (j % src1_rows) * src1_depth
                    + (k % src1_depth);

                let out_idx = i * (dst_rows * dst_depth) + j * dst_depth + k;

                dst[out_idx] = src0[a_idx] * src1[b_idx];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::cpu::Cpu,
        builder::*,
        layout::{Type, D1, D2, D3},
        Backend, ComputeGraph,
    };

    #[test]
    fn mul_cpu_1d() -> anyhow::Result<()> {
        let mut backend = Cpu::new();
        let mut tb = TensorBuilder::prealloc(256 * 256)?;

        let src1 = vec![1.0f32, 1., 1., 1.];
        let src2 = vec![2., 1., 2., 1.];

        let src1_t = tb.new_tensor(D1 { cols: 4 }.with_type(Type::F32))?;
        let src2_t = tb.new_tensor(D1 { cols: 4 }.with_type(Type::F32))?;

        tb.copy_data_from::<f32>(src1_t, &src1);
        tb.copy_data_from::<f32>(src2_t, &src2);

        let dst = tb.mul(src1_t, src2_t)?;

        let (ctx, _) = tb.finish(&mut backend)?;
        let graph = ComputeGraph::forward_expand(backend.mem_limit(), dst, &ctx)?;
        backend.compute_graph(&graph, &ctx)?;

        let expected = vec![2., 1., 2., 1.];
        let result = ctx[dst].elements::<f32>(&backend);

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn mul_cpu_2d() -> anyhow::Result<()> {
        let mut backend = Cpu::new();
        let mut tb = TensorBuilder::prealloc(256 * 256)?;

        let src1 = vec![1., 1., 1., 1., 1., 1.];
        let src2 = vec![2., 1., 2., 1., 2., 3.];

        let src1_t = tb.new_tensor(D2 { rows: 2, cols: 3 }.with_type(Type::F32))?;
        let src2_t = tb.new_tensor(D2 { rows: 2, cols: 3 }.with_type(Type::F32))?;

        tb.copy_data_from::<f32>(src1_t, &src1);
        tb.copy_data_from::<f32>(src2_t, &src2);

        let dst = tb.mul(src1_t, src2_t)?;

        let (ctx, _) = tb.finish(&mut backend)?;
        let graph = ComputeGraph::forward_expand(backend.mem_limit(), dst, &ctx)?;
        backend.compute_graph(&graph, &ctx)?;

        let expected = vec![2., 1., 2., 1., 2., 3.];
        let result = ctx[dst].elements::<f32>(&backend);

        assert_eq!(result, expected);

        Ok(())
    }
}

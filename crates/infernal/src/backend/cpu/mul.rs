use crate::buf::SliceCast;
use crate::Tensor;

pub(crate) fn mul(src1_tensor: &Tensor, src2_tensor: &Tensor, dst_tensor: &Tensor, src1: &[u8], src2: &[u8], dst: &mut [u8]) {
    let src1 = src1.cast::<f32>();
    let src2 = src2.cast::<f32>();
    let dst = dst.cast_mut::<f32>();

    let [src1_cols, src1_rows, src1_depth, _] = &src1_tensor.dims;
    let [src2_cols, src2_rows, src2_depth, _] = &src2_tensor.dims;
    let [dst_cols, dst_rows, dst_depth, _] = &dst_tensor.dims;

    for i in 0..*dst_cols {
        for j in 0..*dst_rows {
            for k in 0..*dst_depth {
                let a_idx = (i % src1_cols) * (src1_rows * src1_depth) +
                    (j % src1_rows) * src1_depth +
                    (k % src1_depth);

                let b_idx = (i % src2_cols) * (src2_rows * src2_depth) +
                    (j % src2_rows) * src2_depth +
                    (k % src2_depth);

                let out_idx = i * (dst_rows * dst_depth) + j * dst_depth + k;

                dst[out_idx] = src1[a_idx] * src2[b_idx];
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
        Backend, ComputeGraph
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
        let mut graph = ComputeGraph::default();
        graph.expand_forward(dst, &ctx)?;
        graph.find_exec_plan(backend.mem_limit(), &ctx)?;
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
        let mut graph = ComputeGraph::default();
        graph.expand_forward(dst, &ctx)?;
        graph.find_exec_plan(backend.mem_limit(), &ctx)?;
        backend.compute_graph(&graph, &ctx)?;
        
        let expected = vec![2., 1., 2., 1., 2., 3.];
        let result = ctx[dst].elements::<f32>(&backend);

        assert_eq!(result, expected);

        Ok(())
    }
}

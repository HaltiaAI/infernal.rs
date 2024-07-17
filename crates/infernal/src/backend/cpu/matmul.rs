use crate::buf::SliceCast;
use crate::Tensor;

use std::ops::Mul;

pub(crate) fn matmul<TIn: Copy, TTIn: Copy, TOut: Copy>(
    src1_tensor: &Tensor,
    src2_tensor: &Tensor,
    dst_tensor: &Tensor,
    src1: &[u8],
    src2: &[u8],
    dst: &mut [u8],
)
where
    TIn: Mul<TTIn>,
    TOut: core::iter::Sum<<TIn as Mul<TTIn>>::Output>,
{
    let tin_size = std::mem::size_of::<TIn>();
    let ttin_size = std::mem::size_of::<TTIn>();
    let tout_size = std::mem::size_of::<TOut>();

    let src1 = src1.cast::<TIn>();
    let src2 = src2.cast::<TTIn>();
    let dst = dst.cast_mut::<TOut>();

    let max_depth = src1_tensor.depth().max(src2_tensor.depth());

    for depth in 0..max_depth {
        for r2 in 0..src2_tensor.rows() {
            for r1 in 0..src1_tensor.rows() {
                let bc_ab = if src2_tensor.depth() > src1_tensor.depth() {
                    depth / (src2_tensor.depth() / src1_tensor.depth())
                } else {
                    depth
                };

                let bc_ba = if src1_tensor.depth() > src2_tensor.depth() {
                    depth / (src1_tensor.depth() / src2_tensor.depth())
                } else {
                    depth
                };

                let t1_offs = (r1 * src1_tensor.row_stride() + bc_ab * src1_tensor.depth_stride()) / tin_size;
                let t2_offs = (r2 * src2_tensor.row_stride() + bc_ba * src2_tensor.depth_stride()) / ttin_size;

                let row1 = &src1[t1_offs..t1_offs + src1_tensor.cols()];
                let row2 = &src2[t2_offs..t2_offs + src2_tensor.cols()];

                let sum = row1.iter().zip(row2.iter()).map(|(&x, &y)| x * y).sum::<TOut>();

                let dst_idx = depth * dst_tensor.depth_stride()
                    + r2 * dst_tensor.row_stride()
                    + r1 * dst_tensor.col_stride();

                dst[dst_idx / 4] = sum;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        backend::cpu::Cpu,
        backend::Metal,
        builder::*,
        layout::{Type, D1, D2, D3},
        Backend, ComputeGraph
    };

    #[test]
    fn matmul_cpu_2d() -> anyhow::Result<()> {
        let mut backend = Cpu::new();
        // let mut backend = Metal::new()?;
        let mut tb = TensorBuilder::prealloc(256 * 256)?;

        let src1 = vec![2., 3., 2., 3.];
        let src2 = vec![2., 2., 2., 2.];

        let src1_t = tb.new_tensor(D2 { rows: 2, cols: 2 }.with_type(Type::F32))?;
        let src2_t = tb.new_tensor(D2 { rows: 2, cols: 2 }.with_type(Type::F32))?;

        tb.copy_data_from::<f32>(src1_t, &src1);
        tb.copy_data_from::<f32>(src2_t, &src2);

        let dst = tb.matmul(src1_t, src2_t)?;

        let (ctx, _) = tb.finish(&mut backend)?;
        let mut graph = ComputeGraph::default();
        graph.expand_forward(dst, &ctx)?;
        graph.find_exec_plan(backend.mem_limit(), &ctx)?;
        backend.compute_graph(&graph, &ctx)?;
        
        let expected = vec![10., 10., 10., 10.];
        let result = ctx[dst].elements::<f32>(&backend);

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn matmul_cpu_3d() -> anyhow::Result<()> {
        let mut backend = Cpu::new();
        //let mut backend = Metal::new()?;
        let mut tb = TensorBuilder::prealloc(256 * 256)?;

        let src1 = vec![2., 3., 2., 3., 4., 2., 3., 2.];
        let src2 = vec![2., 2., 12.5, 2., 2., 16.7, 2., 2.];

        let src1_t = tb.new_tensor(D3 { depth: 2, rows: 2, cols: 2 }.with_type(Type::F32))?;
        let src2_t = tb.new_tensor(D2 { rows: 4, cols: 2 }.with_type(Type::F32))?;

        tb.copy_data_from::<f32>(src1_t, &src1);
        tb.copy_data_from::<f32>(src2_t, &src2);

        let dst = tb.matmul(src1_t, src2_t)?;

        let (ctx, _) = tb.finish(&mut backend)?;
        let mut graph = ComputeGraph::default();
        graph.expand_forward(dst, &ctx)?;
        graph.find_exec_plan(backend.mem_limit(), &ctx)?;
        backend.compute_graph(&graph, &ctx)?;
        
        let expected = vec![10.0, 10.0, 31.0, 31.0, 54.100002, 54.100002, 10.0, 10.0];
        let result = ctx[dst].elements::<f32>(&backend);

        assert_eq!(result, expected);

        Ok(())
    }
}

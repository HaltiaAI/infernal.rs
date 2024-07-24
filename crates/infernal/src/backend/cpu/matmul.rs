use crate::{buf::SliceCast, Tensor};

use std::ops::Mul;

pub(crate) fn matmul<TIn: Copy, TTIn: Copy, TOut: Copy>(
    src0_tensor: &Tensor,
    src1_tensor: &Tensor,
    dst_tensor: &Tensor,
    src0: &[u8],
    src1: &[u8],
    dst: &mut [u8],
) where
    TIn: Mul<TTIn>,
    TOut: core::iter::Sum<<TIn as Mul<TTIn>>::Output>,
{
    let tin_size = std::mem::size_of::<TIn>();
    let ttin_size = std::mem::size_of::<TTIn>();
    let tout_size = std::mem::size_of::<TOut>();

    let src0 = src0.cast::<TIn>();
    let src1 = src1.cast::<TTIn>();
    let dst = dst.cast_mut::<TOut>();

    let max_depth = src0_tensor.depth().max(src1_tensor.depth());

    for depth in 0..max_depth {
        for r2 in 0..src1_tensor.rows() {
            for r1 in 0..src0_tensor.rows() {
                let bc_ab = if src1_tensor.depth() > src0_tensor.depth() {
                    depth / (src1_tensor.depth() / src0_tensor.depth())
                } else {
                    depth
                };

                let bc_ba = if src0_tensor.depth() > src1_tensor.depth() {
                    depth / (src0_tensor.depth() / src1_tensor.depth())
                } else {
                    depth
                };

                let t1_offs =
                    (r1 * src0_tensor.row_stride() + bc_ab * src0_tensor.depth_stride()) / tin_size;
                let t2_offs = (r2 * src1_tensor.row_stride() + bc_ba * src1_tensor.depth_stride())
                    / ttin_size;

                let row1 = &src0[t1_offs..t1_offs + src0_tensor.cols()];
                let row2 = &src1[t2_offs..t2_offs + src1_tensor.cols()];

                let sum = row1
                    .iter()
                    .zip(row2.iter())
                    .map(|(&x, &y)| x * y)
                    .sum::<TOut>();

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
        backend::{cpu::Cpu, Metal},
        builder::*,
        layout::{Type, D1, D2, D3},
        Backend, ComputeGraph,
    };

    #[test]
    fn matmul_cpu_2d() -> anyhow::Result<()> {
        let mut backend = Cpu::new();
        // let mut backend = Metal::new()?;
        let mut tb = TensorBuilder::prealloc(256 * 256)?;

        let src0 = vec![2., 3., 2., 3.];
        let src1 = vec![2., 2., 2., 2.];

        let src0_t = tb.new_tensor(D2 { rows: 2, cols: 2 }.with_type(Type::F32))?;
        let src1_t = tb.new_tensor(D2 { rows: 2, cols: 2 }.with_type(Type::F32))?;

        tb.copy_data_from::<f32>(src0_t, &src0);
        tb.copy_data_from::<f32>(src1_t, &src1);

        let dst = tb.matmul(src0_t, src1_t)?;

        let (ctx, _) = tb.finish(&mut backend)?;
        let graph = ComputeGraph::forward_expand(backend.mem_limit(), dst, &ctx)?;
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

        let src0 = vec![2., 3., 2., 3., 4., 2., 3., 2.];
        let src1 = vec![2., 2., 12.5, 2., 2., 16.7, 2., 2.];

        let src0_t = tb.new_tensor(
            D3 {
                depth: 2,
                rows: 2,
                cols: 2,
            }
            .with_type(Type::F32),
        )?;
        let src1_t = tb.new_tensor(D2 { rows: 4, cols: 2 }.with_type(Type::F32))?;

        tb.copy_data_from::<f32>(src0_t, &src0);
        tb.copy_data_from::<f32>(src1_t, &src1);

        let dst = tb.matmul(src0_t, src1_t)?;

        let (ctx, _) = tb.finish(&mut backend)?;
        let graph = ComputeGraph::forward_expand(backend.mem_limit(), dst, &ctx)?;
        backend.compute_graph(&graph, &ctx)?;

        let expected = vec![10.0, 10.0, 31.0, 31.0, 54.100002, 54.100002, 10.0, 10.0];
        let result = ctx[dst].elements::<f32>(&backend);

        assert_eq!(result, expected);

        Ok(())
    }
}

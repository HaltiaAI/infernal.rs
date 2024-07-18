use crate::{buf::SliceCast, Tensor};

pub(crate) fn get_rows(
    src_tensor: &Tensor,
    dst_tensor: &Tensor,
    src: &[u8],
    index: &[u8],
    dst: &mut [u8],
) {
    let src = src.cast::<f32>();
    let index = index.cast::<i32>();
    let dst = dst.cast_mut::<f32>();

    for (i, r) in index.iter().enumerate() {
        let x = *r as usize * src_tensor.row_stride() / core::mem::size_of::<f32>();
        let y = i * dst_tensor.row_stride() / core::mem::size_of::<f32>();

        for col in 0..src_tensor.cols() {
            dst[y + col] = src[x + col];
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        backend::cpu::Cpu,
        builder::*,
        layout::{Type, D1, D2},
        Backend, ComputeGraph,
    };

    #[test]
    fn get_rows_cpu() -> anyhow::Result<()> {
        let mut backend = Cpu::new();
        let mut tb = TensorBuilder::prealloc(256 * 256)?;

        let src0 = (1..11).map(|x| x as f32).collect::<Vec<_>>();
        let src1 = vec![1, 2, 4];

        let src0_t = tb.new_tensor(D2 { rows: 5, cols: 2 }.with_type(Type::F32))?;
        let src1_t = tb.new_tensor(D1 { cols: 3 }.with_type(Type::I32))?;

        tb.copy_data_from::<f32>(src0_t, &src0);
        tb.copy_data_from::<i32>(src1_t, &src1);

        let dst = tb.get_rows(src0_t, src1_t)?;

        let (ctx, _) = tb.finish(&mut backend)?;
        let graph = ComputeGraph::forward_expand(backend.mem_limit(), dst, &ctx)?;
        backend.compute_graph(&graph, &ctx)?;

        let expected = vec![3., 4., 5., 6., 9., 10.];
        let result = ctx[dst].elements::<f32>(&backend);

        assert_eq!(result, expected);

        Ok(())
    }
}

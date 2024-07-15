use crate::buf::SliceCast;
use crate::Tensor;

pub(crate) fn get_rows(src1_tensor: &Tensor, dst_tensor: &Tensor, src1: &[u8], src2: &[u8], dst: &mut [u8]) {
    let src1 = src1.cast::<f32>();
    let src2 = src2.cast::<i32>();
    let dst = dst.cast_mut::<f32>();

    for (i, r) in src2.iter().enumerate() {
        let x = *r as usize * src1_tensor.row_stride() / core::mem::size_of::<f32>();
        let y = i * dst_tensor.row_stride() / core::mem::size_of::<f32>();

        for col in 0..src1_tensor.cols() {
            dst[y + col] = src1[x + col];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::cpu::Cpu,
        backend::Metal,
        builder::*,
        layout::{Type, D1, D2, D3},
        Backend, ComputeGraph
    };

    #[test]
    fn get_rows_cpu() -> anyhow::Result<()> {
        let mut backend = Cpu::new();
        let mut tb = TensorBuilder::prealloc(256 * 256)?;

        let src1 = (1..11).map(|x| x as f32).collect::<Vec<_>>();
        let src2 = vec![1, 2, 4];

        let src1_t = tb.new_tensor(D2 { rows: 5, cols: 2 }.with_type(Type::F32))?;
        let src2_t = tb.new_tensor(D1 { cols: 3 }.with_type(Type::I32))?;

        tb.copy_data_from::<f32>(src1_t, &src1);
        tb.copy_data_from::<i32>(src2_t, &src2);

        let dst = tb.get_rows(src1_t, src2_t)?;

        let (ctx, _) = tb.finish(&mut backend)?;
        let mut graph = ComputeGraph::default();
        graph.expand_forward(dst, &ctx)?;
        graph.find_exec_plan(backend.mem_limit(), &ctx)?;
        backend.compute_graph(&graph, &ctx)?;
        
        let expected = vec![3., 4., 5., 6., 9., 10.];
        let result = ctx[dst].elements::<f32>(&backend);

        assert_eq!(result, expected);

        Ok(())
    }
}


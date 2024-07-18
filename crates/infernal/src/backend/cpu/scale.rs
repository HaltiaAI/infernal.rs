use crate::buf::SliceCast;

pub(crate) fn scale(src: &[u8], scale: &[u8], dst: &mut [u8]) {
    let src = src.cast::<f32>();
    let [scale, ..] = scale.cast::<f32>() else {
        unreachable!("Scalar tensor cannot be empty.");
    };

    let dst = dst.cast_mut::<f32>();

    for (idx, el) in src.iter().enumerate() {
        dst[idx] = el * scale;
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        backend::cpu::Cpu,
        builder::*,
        layout::{Type, D1, D2, D3},
        Backend, ComputeGraph,
    };

    #[test]
    fn scale_cpu() -> anyhow::Result<()> {
        let mut backend = Cpu::new();
        let mut tb = TensorBuilder::prealloc(256 * 256)?;

        let src1 = (1..5).map(|x| x as f32).collect::<Vec<_>>();
        let src2 = vec![2.];

        let src1_t = tb.new_tensor(D1 { cols: 4 }.with_type(Type::F32))?;
        let src2_t = tb.new_tensor(D1 { cols: 1 }.with_type(Type::F32))?;

        tb.copy_data_from::<f32>(src1_t, &src1);
        tb.copy_data_from::<f32>(src2_t, &src2);

        let dst = tb.scale(src1_t, src2_t)?;

        let (ctx, _) = tb.finish(&mut backend)?;
        let graph = ComputeGraph::forward_expand(backend.mem_limit(), dst, &ctx)?;
        backend.compute_graph(&graph, &ctx)?;

        let expected = vec![2., 4., 6., 8.];
        let result = ctx[dst].elements::<f32>(&backend);

        assert_eq!(result, expected);

        Ok(())
    }
}

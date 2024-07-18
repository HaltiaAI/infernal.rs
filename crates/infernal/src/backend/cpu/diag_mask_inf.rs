use crate::{buf::SliceCast, Tensor, Type};

pub(crate) fn diag_mask_inf(src_tensor: &Tensor, src: &[u8], dst: &mut [u8], past_len: usize) {
    debug_assert!(src_tensor.typ() == Type::F32);

    let src = src.cast::<f32>();
    let dst = dst.cast_mut::<f32>();

    let rows = src_tensor.rows();
    let cols = src_tensor.cols();

    for row in 0..rows {
        for col in 0..cols {
            let i = row * cols + col;

            if col > past_len + row % rows {
                dst[i] = f32::NEG_INFINITY;
            } else {
                dst[i] = src[i];
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{backend::cpu::Cpu, builder::*, *};

    fn diag_mask_inf(backend: &mut impl Backend) -> anyhow::Result<()> {
        let layout = Layout::new_2d(8, 8, Type::F32);

        let mut tb = TensorBuilder::<_, NoSrc>::prealloc(8 * 4 * 4 * 8)?
            .with_name("diag_mask_test")
            .with_auto_op_names(AutoOpName::Expanded { with_id: false });

        let tensor = tb.new_tensor(layout)?;
        tb.fill_2d_f32(tensor, 12.3);

        let result = tb.diag_mask_inf(tensor, 0)?;

        let (ctx, _) = tb.finish(backend)?;

        let graph = ComputeGraph::forward_expand(backend.mem_limit(), result, &ctx)?;
        backend.compute_graph(&graph, &ctx)?;

        let res = &ctx[result];
        let elems = res.elements::<f32>(&*backend);

        let inf = f32::NEG_INFINITY;

        #[rustfmt::skip]
        let expected = [
            12.3,  inf,  inf,  inf,  inf,  inf,  inf,  inf,
            12.3, 12.3,  inf,  inf,  inf,  inf,  inf,  inf,
            12.3, 12.3, 12.3,  inf,  inf,  inf,  inf,  inf,
            12.3, 12.3, 12.3, 12.3,  inf,  inf,  inf,  inf,
            12.3, 12.3, 12.3, 12.3, 12.3,  inf,  inf,  inf,
            12.3, 12.3, 12.3, 12.3, 12.3, 12.3,  inf,  inf,
            12.3, 12.3, 12.3, 12.3, 12.3, 12.3, 12.3,  inf,
            12.3, 12.3, 12.3, 12.3, 12.3, 12.3, 12.3, 12.3,
        ];

        assert_eq!(elems, expected);

        Ok(())
    }

    #[test]
    fn test_cpu_diagmask() -> anyhow::Result<()> {
        #[cfg(feature = "test-cpu-with-metal")]
        let mut backend = crate::backend::Metal::new().unwrap();

        #[cfg(not(feature = "test-cpu-with-metal"))]
        let mut backend = Cpu::new();

        diag_mask_inf(&mut backend)
    }

    #[test]
    fn diag_mask_inf_n_past() -> anyhow::Result<()> {
        let mut backend = Cpu::new();
        let layout = Layout::new_2d(8, 8, Type::F32);

        let mut tb = TensorBuilder::<_, NoSrc>::prealloc(8 * 4 * 4 * 8)?
            .with_name("diag_mask_test")
            .with_auto_op_names(AutoOpName::Expanded { with_id: false });

        let tensor = tb.new_tensor(layout)?;
        tb.fill_2d_f32(tensor, 12.3);

        let result = tb.diag_mask_inf(tensor, 2)?;

        let (ctx, _) = tb.finish(&mut backend)?;

        let graph = ComputeGraph::forward_expand(backend.mem_limit(), result, &ctx)?;
        backend.compute_graph(&graph, &ctx)?;

        let res = &ctx[result];
        let elems = res.elements::<f32>(&backend);

        let inf = f32::NEG_INFINITY;

        #[rustfmt::skip]
        let expected = [
            12.3, 12.3, 12.3,  inf,  inf,  inf,  inf,  inf,
            12.3, 12.3, 12.3, 12.3,  inf,  inf,  inf,  inf,
            12.3, 12.3, 12.3, 12.3, 12.3,  inf,  inf,  inf,
            12.3, 12.3, 12.3, 12.3, 12.3, 12.3,  inf,  inf,
            12.3, 12.3, 12.3, 12.3, 12.3, 12.3, 12.3,  inf,
            12.3, 12.3, 12.3, 12.3, 12.3, 12.3, 12.3, 12.3,
            12.3, 12.3, 12.3, 12.3, 12.3, 12.3, 12.3, 12.3,
            12.3, 12.3, 12.3, 12.3, 12.3, 12.3, 12.3, 12.3
        ];

        assert_eq!(elems, expected);

        Ok(())
    }
}

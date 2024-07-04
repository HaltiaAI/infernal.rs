use crate::{buf::SliceCast, Tensor};

pub(crate) fn diag_mask_inf(input_tensor: &Tensor, input: &[u8], output: &mut [u8], past_len: usize) {
    let input = input.cast::<f32>();
    let output = output.cast_mut::<f32>();

    let rows = input_tensor.rows();
    let cols = input_tensor.cols();

    for row in 0..rows {
        for col in 0..cols {
            let i = row * cols + col;

            if col > past_len + row % rows {
                output[i] = f32::NEG_INFINITY;
            } else {
                output[i] = input[i];
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{backend::cpu::Cpu, builder::*, *};

    fn diag_mask_inf(backend: &mut impl Backend) {
        let layout = Layout::new_2d(8, 8, Type::F32);

        let mut tb = TensorBuilder::<_, NoSrc>::prealloc(8 * 4 * 4 * 8)
            .unwrap()
            .with_name("diag_mask_test")
            .with_auto_op_names(AutoOpName::Expanded { with_id: false });

        let tensor = tb.new_tensor(layout).unwrap();
        tb.fill_2d_f32(tensor, 12.3);

        let result = tb.diag_mask_inf(tensor, 0).unwrap();

        let (ctx, _) = tb.finish(backend).unwrap();

        let mut gf = ComputeGraph::default();
        gf.expand_forward(result, &ctx).unwrap();
        gf.find_exec_plan(backend.mem_limit(), &ctx).unwrap();
        backend.compute_graph(&gf, &ctx).unwrap();

        let res = &ctx[result];
        let elems = res.elements::<f32>(&*backend);

        let inf = f32::NEG_INFINITY;
        let expected = [
            12.3, inf, inf, inf, inf, inf, inf, inf, 12.3, 12.3, inf, inf, inf, inf, inf, inf,
            12.3, 12.3, 12.3, inf, inf, inf, inf, inf, 12.3, 12.3, 12.3, 12.3, inf, inf, inf, inf,
            12.3, 12.3, 12.3, 12.3, 12.3, inf, inf, inf, 12.3, 12.3, 12.3, 12.3, 12.3, 12.3, inf,
            inf, 12.3, 12.3, 12.3, 12.3, 12.3, 12.3, 12.3, inf, 12.3, 12.3, 12.3, 12.3, 12.3, 12.3,
            12.3, 12.3,
        ];

        assert_eq!(elems, expected);
    }

    #[test]
    fn test_cpu_diagmask() {
        let mut backend = Cpu::new();

        diag_mask_inf(&mut backend);
    }
}

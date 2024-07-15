use crate::buf::SliceCast;
use crate::Tensor;

trait FloatExt: Sized {
    #[inline]
    fn size() -> usize {
        std::mem::size_of::<Self>()
    }
}

trait FloatCast<T>: FloatExt {
    #[inline]
    fn cast(self) -> T;
}

impl FloatExt for f32 {}
impl FloatExt for f16 {}

impl FloatCast<f32> for f16 {
    #[inline]
    fn cast(self) -> f32 {
        self as _
    }
}

impl FloatCast<f32> for f32 {
    #[inline]
    fn cast(self) -> f32 {
        self as _
    }
}

pub(crate) fn cpy(src_tensor: &Tensor, dst_tensor: &Tensor, src: &[u8], dst: &mut [u8]) {
    let src = src.cast::<f32>();
    let dst = dst.cast_mut::<f32>();

    let [scols, srows, sdepths, _] = &src_tensor.dims;
    let [nb00, nb01, nb02, nb03] = &src_tensor.stride;
    let [cols, rows, ..] = &dst_tensor.dims;
    let [nb10, nb11, nb12, nb13] = &dst_tensor.stride;

    let total_elements: usize = src_tensor.dims.iter().product();

    for batch in 0..src_tensor.batches() {
        for depth in 0..src_tensor.depth() {
            for row in 0..src_tensor.rows() {
                let n = batch * sdepths * srows * scols + depth * srows * scols + row * scols;

                let i3 = n / total_elements;

                let i2 = (n - i3 * total_elements) / (cols * rows);
                let i1 = (n - i3 * total_elements - i2 * cols * rows) / cols;
                let i0 = n - i3 * total_elements - i2 * cols * rows - i1 * cols;

                let dst_offs = (i3 * nb13 + i2 * nb12 + i1 * nb11 + i0 * nb10) / std::mem::size_of::<f32>();

                for col in 0..src_tensor.cols() {
                    let src_idx = (batch * nb03 + depth * nb02 + row * nb01 + col * nb00) / std::mem::size_of::<f32>();

                    dst[dst_offs + col] = src[src_idx];
                } 
            }
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
    fn cpy_1d() -> anyhow::Result<()> {
        let mut backend = Cpu::new();
        let mut tb = TensorBuilder::prealloc(256 * 256)?;

        let src1 = vec![1.0f32, 0.97, 1., 1.];

        let src1_t = tb.new_tensor(D1 { cols: 4 }.with_type(Type::F32))?;
        let src2_t = tb.new_tensor(D1 { cols: 4 }.with_type(Type::F32))?;

        tb.copy_data_from::<f32>(src1_t, &src1);

        let dst = tb.copy(src1_t, src2_t);

        let (ctx, _) = tb.finish(&mut backend)?;
        let mut graph = ComputeGraph::default();

        graph.expand_forward(dst, &ctx)?;
        graph.find_exec_plan(backend.mem_limit(), &ctx)?;
        backend.compute_graph(&graph, &ctx)?;
        
        let result = ctx[dst].elements::<f32>(&backend);

        assert_eq!(result, src1);
Ok(()) }

    #[test]
    fn cpy_2d() -> anyhow::Result<()> {
        let mut backend = Cpu::new();
        let mut tb = TensorBuilder::prealloc(256 * 256)?;

        let src1 = vec![1., 2., 3., 4., 5., 6.];

        let src1_t = tb.new_tensor(D2 { rows: 2, cols: 3 }.with_type(Type::F32))?;
        let src2_t = tb.new_tensor(D2 { rows: 2, cols: 3 }.with_type(Type::F32))?;

        tb.copy_data_from::<f32>(src1_t, &src1);

        let dst = tb.copy(src1_t, src2_t);

        let (ctx, _) = tb.finish(&mut backend)?;
        let mut graph = ComputeGraph::default();
        graph.expand_forward(dst, &ctx)?;
        graph.find_exec_plan(backend.mem_limit(), &ctx)?;
        backend.compute_graph(&graph, &ctx)?;
        
        let result = ctx[dst].elements::<f32>(&backend);

        assert_eq!(result, src1);

        Ok(())
    }

    #[test]
    fn cpy_2d_diff_dims() -> anyhow::Result<()> {
        let mut backend = Cpu::new();
        //let mut backend = Metal::new()?;
        let mut tb = TensorBuilder::prealloc(256 * 256)?;

        let src1 = vec![1., 2., 3., 4., 5., 6.];

        let src1_t = tb.new_tensor(D2 { rows: 2, cols: 3 }.with_type(Type::F32))?;
        let src2_t = tb.new_tensor(D2 { rows: 3, cols: 2 }.with_type(Type::F32))?;

        tb.copy_data_from::<f32>(src1_t, &src1);

        let dst = tb.copy(src1_t, src2_t);

        let (ctx, _) = tb.finish(&mut backend)?;
        let mut graph = ComputeGraph::default();
        graph.expand_forward(dst, &ctx)?;
        graph.find_exec_plan(backend.mem_limit(), &ctx)?;
        backend.compute_graph(&graph, &ctx)?;
        
        let result = ctx[dst].elements::<f32>(&backend);

        assert_eq!(result, src1);

        Ok(())
    }
}

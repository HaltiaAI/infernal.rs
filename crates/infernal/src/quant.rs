use std::borrow::Cow;

use crate::{
    alloc::{Oom, ReadWriteAlloc},
    op::{Op, UnaryOp},
    Layout, TensorBuilder, TensorId, TensorSrc, Type,
};

impl<A: ReadWriteAlloc, S: TensorSrc> TensorBuilder<A, S>
where
    A::Arg: for<'a> From<&'a Layout>,
{
    pub fn quantize_cpu(&mut self, src_id: TensorId, quant: Type) -> Result<TensorId, Oom> {
        let src = &self[src_id];
        assert_eq!(src.typ(), Type::F32);

        let loc = src.loc();
        let n_elems = src.total_elems();

        let dst_layout = Layout::new_raw_dims(src.layout.raw_dims(), quant);
        let dst_name = format!("{} ({quant:?})", src.name());
        let dst_buf = self.alloc.alloc(A::Arg::from(&dst_layout))?;
        let res_id = self.new_tensor_impl(
            dst_layout,
            Op::Const,
            Some(Cow::from(dst_name)),
            dst_buf.id,
            dst_buf.offs,
            None,
            Some(loc),
        );

        unsafe {
            llama_cpp_sys::ggml::ggml_quantize_chunk(
                gguf::TensorType::from(quant) as u32,
                self.alloc
                    .get_slice(
                        self[src_id].buf_id(),
                        self[src_id].data_offs(),
                        self[src_id].size(),
                    )
                    .unwrap()
                    .as_ptr() as *const f32,
                self.alloc
                    .get_mut_slice(dst_buf.id, dst_buf.offs, dst_layout.size())
                    .unwrap()
                    .as_ptr() as _,
                0,
                n_elems as i32,
                std::ptr::null_mut(),
            );
        }

        Ok(res_id)
    }

    pub fn dequantize_cpu(&mut self, src_id: TensorId) -> Result<TensorId, Oom> {
        let src = &self[src_id];
        assert!(!matches!(src.typ(), Type::F32 | Type::F16 | Type::I32));

        let loc = src.loc();
        let n_elems = src.total_elems();

        let dst_layout = Layout::new_raw_dims(src.layout.raw_dims(), Type::F32);
        let dst_name = format!("{} ({:?})", src.name(), Type::F32);
        let dst_buf = self.alloc.alloc(A::Arg::from(&dst_layout))?;
        let res_id = self.new_tensor_impl(
            dst_layout,
            Op::Unary {
                op: UnaryOp::Dequant,
                a: src_id,
            },
            Some(Cow::from(dst_name)),
            dst_buf.id,
            dst_buf.offs,
            None,
            Some(loc),
        );

        Ok(res_id)
    }
}

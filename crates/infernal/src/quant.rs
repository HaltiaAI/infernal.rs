use std::borrow::Cow;

use crate::{
    alloc::{Oom, ReadWriteAlloc},
    backend::{
        dequantize, dequantize_blk_q3_k, dequantize_blk_q4_k, dequantize_blk_q5_k,
        dequantize_blk_q6_k,
    },
    builder::{TensorOptions, TensorOptionsExt},
    layout::D1,
    op::{Op, UnaryOp},
    Layout, TensorBuilder, TensorId, TensorSrc, Type,
};

use gguf::k_quant::{BlockQ3_K, BlockQ4_K, BlockQ5_K, BlockQ6_K};

impl<A: ReadWriteAlloc, S: TensorSrc> TensorBuilder<A, S>
where
    A::Arg: for<'a> From<&'a Layout>,
    A::Arg: for<'a> From<&'a TensorOptions>,
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

    /// Dequantizes tensor with id `src_id` in a deffered manner. The tensor must not be of type
    /// `F32`, `F16` or `I32`.
    ///
    /// The tensor is dequantized with the dequantization routines provided by the backend.
    pub fn dequantize(&mut self, src_id: TensorId) -> Result<TensorId, Oom> {
        let src = &self[src_id];
        assert!(!matches!(src.typ(), Type::F32 | Type::F16 | Type::I32));

        let loc = src.loc();

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

    /// Dequantizes tensor with id `src_id` immediately using the CPU implementation of dequant.
    /// The tensor must not be of type `F32`, `F16` or `I32`
    pub fn dequantize_cpu(&mut self, src_id: TensorId) -> Result<TensorId, Oom> {
        let src = &self[src_id];
        assert!(!matches!(src.typ(), Type::F32 | Type::F16 | Type::I32));

        let loc = src.loc();

        let dst_layout = Layout::new_raw_dims(src.layout.raw_dims(), Type::F32);
        let dst_name = format!("{} ({:?})", src.name(), Type::F32);
        let dst_buf = self.alloc.alloc(A::Arg::from(&dst_layout))?;
        let dst_id = self.new_tensor_impl(
            dst_layout,
            Op::Const,
            Some(Cow::from(dst_name)),
            dst_buf.id,
            dst_buf.offs,
            None,
            Some(loc),
        );

        let src = self
            .alloc
            .get_slice(
                self[src_id].buf_id(),
                self[src_id].data_offs(),
                self[src_id].size(),
            )
            .unwrap()
            .as_ptr() as *const u8;

        let dst = self
            .alloc
            .get_mut_slice(dst_buf.id, dst_buf.offs, dst_layout.size())
            .unwrap()
            .as_mut_ptr();

        // SAFETY: Get around borrow checker. The src and destinations are never overlapping here.
        let src = unsafe { core::slice::from_raw_parts(src, self[src_id].size()) };
        let dst = unsafe { core::slice::from_raw_parts_mut(dst, dst_layout.size()) };

        let dst_tensor = &self[dst_id];

        match self[src_id].typ() {
            Type::Q6_K => {
                dequantize::<BlockQ6_K, _>(dst_tensor, src, dst, dequantize_blk_q6_k);
            }
            Type::Q5_K => {
                dequantize::<BlockQ5_K, _>(dst_tensor, src, dst, dequantize_blk_q5_k);
            }
            Type::Q4_K => {
                dequantize::<BlockQ4_K, _>(dst_tensor, src, dst, dequantize_blk_q4_k);
            }
            Type::Q3_K => {
                dequantize::<BlockQ3_K, _>(dst_tensor, src, dst, dequantize_blk_q3_k);
            }
            _ => todo!(),
        }

        Ok(dst_id)
    }

    pub fn diffsq(&mut self, a: TensorId, b: TensorId) -> Result<TensorId, Oom> {
        let mul = self.new_tensor(
            D1 {
                cols: self[a].cols(),
            }
            .with_type(Type::F32),
        )?;
        self.fill_1d_f32(mul, -1.0f32);
        let inv_input = self.mul(a, mul)?;
        let diff = self.add(b, inv_input)?;

        // Square differences
        let diffsq = self.mul(diff, diff)?;

        Ok(diffsq)
    }
}

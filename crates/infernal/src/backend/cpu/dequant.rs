use crate::{buf::SliceCast, Tensor};
use gguf::k_quant::{BlockQ3_K, BlockQ4_K, BlockQ5_K, BlockQ6_K, BLK};
use std::{arch::asm, iter};

#[inline]
unsafe fn f16_to_f32(i: u16) -> f32 {
    let result: f32;

    asm!(
        "fcvt {0:s}, {1:h}",
        out(vreg) result,
        in(vreg) i,
        options(pure, nomem, nostack, preserves_flags)
    );

    result
}

/// Dequantize a q3_k block.
pub(super) fn dequantize_blk_q3_k(block: &BlockQ3_K, dst: &mut [f32]) {
    let kmask1: u32 = 0b0011_0000_0011_0000_0011_0000_0011;
    let kmask2: u32 = 0b1111_0000_1111_0000_1111_0000_1111;

    let mut aux: [u32; 4] = [0; 4];
    let scales: [u32; 3] = unsafe { core::mem::transmute(block.scales.clone()) };
    aux[..3].copy_from_slice(&scales);

    let block_scale: f32 = unsafe { f16_to_f32(block.d) };

    // Unpack the local scales. The scales are quantized to 6 bits and stored packed in a total of
    // 96 bits of space.
    //
    // Unpacking them as u32's is simpler as it requires less instructions and only two masks.
    //
    // scale: 2 high bits, 4 low bits = 6 bit scale
    // high bits are stored packed in block.scales[2].
    // low bits are stored in block.scales[0] and block.scales[1]
    let scale_hi = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((scale_hi >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((scale_hi >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((scale_hi >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((scale_hi >> 2) & kmask1) << 4);

    // Scales are normalized between -32 and 31, each scale, scales 16 quantized weights
    let mut scale_factors = aux
        .cast::<i8>()
        .iter()
        .flat_map(|local_scale| iter::repeat(block_scale * (local_scale - 32) as f32).take(16));

    let mut dst_idx = 0;
    let mut hi_mask = 1;

    // We iterate over quantized weights in chunks of 32 elements
    for quants in block.qs.chunks_exact(32) {
        for shift in 0..4 {
            // Every chunk of 16 weights is scaled by one, shared, scale factor
            // For the sake of code readability and simplicity as a reference implementation
            // we repeat each scale factor and zip it with the quant table.
            for ((quant_lo, quant_hi), scale_fac) in quants
                .iter()
                .zip(block.hmask.iter())
                .zip(&mut scale_factors)
            {
                // 4th high bit
                let offset = if quant_hi & hi_mask == 0 { 4 } else { 0 };

                // quant_lo is actually multiple weights, or more precisely their low 3-bits.
                // On each iteration we shift the weight by `shift * 2` to the right and extract
                // the last 3 bits.
                //
                // Since the 4th bit is stored separately, we might compensate for it.
                let weight = ((quant_lo >> (shift << 1)) & 3) as i8 - offset;

                // 4-bit quantized weight is scaled back to an f32 using the block scale factor
                dst[dst_idx] = scale_fac * weight as f32;

                dst_idx += 1;
            }

            hi_mask <<= 1;
        }
    }
}

#[inline]
fn get_scale(j: usize, scales: &[u8]) -> (f32, f32) {
    if j < 4 {
        let d = scales[j] & 63;
        let m = scales[j + 4] & 63;

        (d as f32, m as f32)
    } else {
        let d = (scales[j + 4] & 0xf) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);

        (d as f32, m as f32)
    }
}

pub(super) fn dequantize_blk_q4_k(block: &BlockQ4_K, dst: &mut [f32]) {
    let global_scale = unsafe { f16_to_f32(block.d) };
    let scale_min = unsafe { f16_to_f32(block.dmin) };

    let mut q_offs = 0;
    let mut dst_offs = 0;

    for batch in 0..(BLK / 64) {
        let scale_offs = batch * 2;
        let (scale1, m1) = get_scale(scale_offs as usize, &block.scales);
        let (scale2, m2) = get_scale(scale_offs as usize + 1, &block.scales);

        let d1 = scale1 * global_scale;
        let d2 = scale2 * global_scale;

        let m1 = m1 * scale_min;
        let m2 = m2 * scale_min;

        for l in 0..32 {
            dst[dst_offs + l] = d1 * (block.qs[q_offs + l] & 0xf) as f32 - m1;
            dst[dst_offs + l + 32] = d2 * (block.qs[q_offs + l] >> 4) as f32 - m2;
        }

        q_offs += 32;
        dst_offs += 64;
    }
}

pub(super) fn dequantize_blk_q5_k(block: &BlockQ5_K, dst: &mut [f32]) {
    let global_scale = unsafe { f16_to_f32(block.d) };
    let scale_min = unsafe { f16_to_f32(block.dmin) };

    let mut q_offs = 0;
    let mut dst_offs = 0;
    let mut u1 = 1u8;
    let mut u2 = 2u8;

    for batch in 0..(BLK / 64) {
        let scale_offs = batch * 2;
        let (scale1, m1) = get_scale(scale_offs as usize, &block.scales);
        let (scale2, m2) = get_scale(scale_offs as usize + 1, &block.scales);

        let d1 = scale1 * global_scale;
        let d2 = scale2 * global_scale;

        let m1 = m1 * scale_min;
        let m2 = m2 * scale_min;

        for l in 0..32 {
            let high_bit_offs = if block.qh[l] & u1 != 0 { 16. } else { 0. };

            dst[dst_offs + l] = d1 * ((block.qs[q_offs + l] & 0xF) as f32 + high_bit_offs) - m1;

            let high_bit_offs = if block.qh[l] & u2 != 0 { 16. } else { 0. };

            dst[dst_offs + l + 32] = d2 * ((block.qs[q_offs + l] >> 4) as f32 + high_bit_offs) - m2;
        }

        dst_offs += 64;
        q_offs += 32;
        u1 <<= 2;
        u2 <<= 2;
    }
}

/// Dequantizes a Q6_K quantized block. Arguably a very easy dequantization scheme because you can
/// cleanly store 3 u6's in a 24bit int, making it very easy to extract the high and low weights
/// with simple masks and bitshifts.
///
/// The quantization scheme uses one master scale to scale the scales for each quantized weight.
/// The master scale is shared across the whole block, while the local scales are used to scale 4
/// weights.
pub(super) fn dequantize_blk_q6_k(block: &BlockQ6_K, dst: &mut [f32]) {
    let ql = &block.ql;
    let qh = &block.qh;
    let sc = &block.scales;

    let d = unsafe { f16_to_f32(block.d) };

    for batch in 0..(BLK / 128) {
        for l in 0..32 {
            let is = l / 16;
            // Each weight's lowest 4 bits are packed into ql while the topmost 2 bits are packed
            // in qh.
            //
            // |--------------------|
            //   ql: 0100 1001
            //       q3_l q1_l
            //       q4_l q2_l
            //
            //   qh: 10 01 01 01
            //       q4 q3 q2 q1
            // |--------------------|
            //
            // 4 weights packed in 24 bits (2 low weights @ 4-bits per weight, 1 high weight @
            //   2-bits per weight)
            //
            //  32 is substracted from the qweights so that the weights are between -32 and 32
            let dst_offs = (batch * 128) + l;
            let ql_offs = (batch * 64) + l;
            let qh_offs = (batch * 32) + l;
            let sc_offs = (batch * 8) + is;

            let q1: i8 =
                ((ql[ql_offs + 0] & 0xF) | ((qh[qh_offs] >> 0) & 3) << 4).wrapping_sub(32) as i8;
            let q2: i8 =
                ((ql[ql_offs + 32] & 0xF) | ((qh[qh_offs] >> 2) & 3) << 4).wrapping_sub(32) as i8;
            let q3: i8 =
                ((ql[ql_offs + 0] >> 4) | ((qh[qh_offs] >> 4) & 3) << 4).wrapping_sub(32) as i8;
            let q4: i8 =
                ((ql[ql_offs + 32] >> 4) | ((qh[qh_offs] >> 6) & 3) << 4).wrapping_sub(32) as i8;

            dst[dst_offs + 0] = d * sc[sc_offs + 0] as f32 * q1 as f32;
            dst[dst_offs + 32] = d * sc[sc_offs + 2] as f32 * q2 as f32;
            dst[dst_offs + 64] = d * sc[sc_offs + 4] as f32 * q3 as f32;
            dst[dst_offs + 96] = d * sc[sc_offs + 6] as f32 * q4 as f32;
        }
    }
}

pub fn dequantize<'a, B, F>(dst_tensor: &Tensor, src: &[u8], dst: &'a mut [u8], dequantize_fn: F)
where
    for<'b> F: Fn(&'b B, &'b mut [f32]),
{
    debug_assert_eq!(dst_tensor.total_elems() % BLK, 0);
    let dst = dst.cast_mut::<f32>();

    let blocks = dst_tensor.total_elems() / BLK;
    let blocks = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const B, blocks) };

    let mut block_iter = blocks.chunks_exact(dst_tensor.cols() / BLK);

    for row in 0..dst_tensor.rows() {
        let row_offs = row * dst_tensor.cols();
        let row_end = row_offs + dst_tensor.cols();

        let row_data = &mut dst[row_offs..row_end];

        let blocks = block_iter
            .next()
            .expect("Expected block, but there are none left to be consumed");

        // A block always contains 256 elements.
        for (block_idx, block) in blocks.iter().enumerate() {
            let dst_offs = block_idx * 256;
            let dst_end = dst_offs + 256;

            let dst = &mut row_data[dst_offs..dst_end];

            dequantize_fn(block, dst);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        backend::{cpu::Cpu, Metal},
        builder::TensorOptionsExt,
        layout::D1,
        Backend, ComputeGraph, Layout, TensorBuilder, Type,
    };
    use approx::*;

    #[test]
    fn dequant_q3k() -> anyhow::Result<()> {
        // 1 quant block is 256 elements, so make sure to test at least 2 blocks.
        let layout = Layout::new_2d(2, 512, Type::F32);
        let mut backend = Cpu::new();
        // let mut backend = Metal::new()?;

        let mut tb = TensorBuilder::prealloc(1024 * 1024)?.with_name("add_even");
        let x = tb.new_tensor(layout)?;
        let mut current = 0.0f32;
        tb.fill_with(x, || {
            current += 1.;
            current
        });

        let quant = tb.quantize_cpu(x, Type::Q3_K)?;
        let dequant = tb.dequantize_cpu(quant)?;

        // Dequant against gpu
        // make sure to uncomment the right backend
        /*
        let rows = tb.new_tensor(D1 { cols: 2 }.with_type(Type::I32))?;
        tb.copy_data_from(rows, &[0, 1]);
        let dequant = tb.get_rows(quant, rows)?;
        */

        // Square differences
        let diffsq = tb.diffsq(x, dequant)?;

        let (ctx, _) = tb.finish(&mut backend)?;
        let graph = ComputeGraph::forward_expand(backend.mem_limit(), diffsq, &ctx)?;
        backend.compute_graph(&graph, &ctx)?;

        //dbg!(ctx[dequant].elements::<f32>(&backend));

        let rms = ctx[diffsq].clone();
        let count = rms.total_elems();

        let sum = rms.elements::<f32>(&backend).iter().sum::<f32>();

        // CPU:   61.621677
        // Metal: 61.78904
        let error = sum / count as f32;

        assert_relative_eq!(error, 61.78904, max_relative = 0.01);

        Ok(())
    }

    #[test]
    fn dequant_q4k() -> anyhow::Result<()> {
        // 1 quant block is 256 elements, so make sure to test at least 2 blocks.
        let layout = Layout::new_2d(2, 512, Type::F32);
        let mut backend = Cpu::new();
        // let mut backend = Metal::new()?;

        let mut tb = TensorBuilder::prealloc(1024 * 1024)?.with_name("add_even");
        let x = tb.new_tensor(layout)?;
        let mut current = 0.0f32;
        tb.fill_with(x, || {
            current += 1.;
            current
        });

        let quant = tb.quantize_cpu(x, Type::Q4_K)?;
        let dequant = tb.dequantize_cpu(quant)?;

        // Dequant against gpu
        // make sure to uncomment the right backend
        /*
        let rows = tb.new_tensor(D1 { cols: 2 }.with_type(Type::I32))?;
        tb.copy_data_from(rows, &[0, 1]);
        let dequant = tb.get_rows(quant, rows)?;
        */

        // Square differences
        let diffsq = tb.diffsq(x, dequant)?;

        let (ctx, _) = tb.finish(&mut backend)?;
        let graph = ComputeGraph::forward_expand(backend.mem_limit(), diffsq, &ctx)?;
        backend.compute_graph(&graph, &ctx)?;

        // dbg!(ctx[dequant].elements::<f32>(&backend));

        let rms = ctx[diffsq].clone();
        let count = rms.total_elems();

        let sum = rms.elements::<f32>(&backend).iter().sum::<f32>();

        // CPU:   61.527101
        // Metal: 61.52101
        let error = sum / count as f32;

        assert_relative_eq!(error, 61.52101, max_relative = 0.01);

        Ok(())
    }

    #[test]
    fn dequant_q5k() -> anyhow::Result<()> {
        // 1 quant block is 256 elements, so make sure to test at least 2 blocks.
        let layout = Layout::new_2d(2, 512, Type::F32);
        let mut backend = Cpu::new();
        // let mut backend = Metal::new()?;

        let mut tb = TensorBuilder::prealloc(1024 * 1024)?.with_name("add_even");
        let x = tb.new_tensor(layout)?;
        let mut current = 0.0f32;
        tb.fill_with(x, || {
            current += 1.;
            current
        });

        let quant = tb.quantize_cpu(x, Type::Q5_K)?;
        let dequant = tb.dequantize_cpu(quant)?;

        // Dequant against gpu
        // make sure to uncomment the right backend
        /*
        let rows = tb.new_tensor(D1 { cols: 2 }.with_type(Type::I32))?;
        tb.copy_data_from(rows, &[0, 1]);
        let dequant = tb.get_rows(quant, rows)?;
        */

        // Square differences
        let diffsq = tb.diffsq(x, dequant)?;

        let (ctx, _) = tb.finish(&mut backend)?;
        let graph = ComputeGraph::forward_expand(backend.mem_limit(), diffsq, &ctx)?;
        backend.compute_graph(&graph, &ctx)?;

        // dbg!(ctx[dequant].elements::<f32>(&backend));

        let rms = ctx[diffsq].clone();
        let count = rms.total_elems();

        let sum = rms.elements::<f32>(&backend).iter().sum::<f32>();

        // CPU:   28.865108
        // Metal: 28.86510
        let error = sum / count as f32;

        assert_relative_eq!(error, 28.865108, max_relative = 0.01);

        Ok(())
    }

    #[test]
    fn dequant_q6k() -> anyhow::Result<()> {
        // 1 quant block is 256 elements, so make sure to test at least 2 blocks.
        let layout = Layout::new_2d(2, 512, Type::F32);
        let mut backend = Cpu::new();
        // let mut backend = Metal::new()?;

        let input = (0..1024).map(|x| x as f32).collect::<Vec<_>>();

        let mut tb = TensorBuilder::prealloc(1024 * 1024)?.with_name("add_even");
        let x = tb.new_tensor(layout)?;
        tb.copy_data_from(x, &input);

        let quant = tb.quantize_cpu(x, Type::Q6_K)?;

        let dequant = tb.dequantize_cpu(quant)?;

        // Dequant against gpu
        // make sure to uncomment the right backend
        /*
        let rows = tb.new_tensor(D1 { cols: 2 }.with_type(Type::I32))?;
        tb.copy_data_from(rows, &[0, 1]);
        let dequant = tb.get_rows(quant, rows)?;
        */

        // Square differences
        let diffsq = tb.diffsq(x, dequant)?;

        let (ctx, _) = tb.finish(&mut backend)?;
        let graph = ComputeGraph::forward_expand(backend.mem_limit(), diffsq, &ctx)?;
        backend.compute_graph(&graph, &ctx)?;

        let rms = ctx[diffsq].clone();
        let count = rms.total_elems();

        let sum = rms.elements::<f32>(&backend).iter().sum::<f32>();

        // CPU:   15.227578
        // Metal: 15.228573
        let error = sum / count as f32;

        assert_relative_eq!(error, 15.228573, max_relative = 0.01);

        Ok(())
    }
}

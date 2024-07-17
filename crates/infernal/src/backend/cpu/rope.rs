use crate::{
    buf::SliceCast,
    op::{RopeCustom, RopeExpanded},
    Tensor,
};

use std::f32::consts::FRAC_2_PI;

#[inline]
fn rope_yarn_ramp(low: f32, high: f32, i0: f32) -> f32 {
    let y = (i0 / 2. - low) / 0.001f32.max(high - low);
    1.0 - 1.0f32.min(0.0f32.max(y))
}

#[inline]
fn rope_yarn_corr_factor(n_dims: i32, n_ctx_orig: i32, n_rot: f32, base: f32) -> f32 {
    let factor = (n_ctx_orig as f32 / (n_rot * FRAC_2_PI)).ln() / (2. * base.ln());
    n_dims as f32 * factor
}

#[inline]
fn rope_yarn_corr_dims(
    n_dims: i32,
    n_ctx_orig: i32,
    freq_base: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> [f32; 2] {
    let start = rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base)
        .floor()
        .max(0.0);
    let end = rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)
        .ceil()
        .min(n_dims as f32 - 1.0);

    [start, end]
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
#[inline]
fn rope_yarn(
    theta_extrap: f32,
    freq_scale: f32,
    corr_dims: [f32; 2],
    i0: f32,
    ext_factor: f32,
    mut mscale: f32,
) -> (f32, f32) {
    let theta_interp = freq_scale * theta_extrap;
    let mut theta = theta_interp;

    if ext_factor != 0.0 {
        let ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1. - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1. + 0.1 * (1. / freq_scale).ln();
    }

    let cos_theta = theta.cos() * mscale;
    let sin_theta = theta.sin() * mscale;

    (cos_theta, sin_theta)
}

pub(crate) fn rope(
    src_tensor: &Tensor,
    dst_tensor: &Tensor,
    src: &[u8],
    dst: &mut [u8],
    params: RopeExpanded,
) {
    let RopeCustom {
        freq_base,
        freq_scale,
    } = params.custom;
    let RopeExpanded { n_past, n_dims, .. } = params;

    let src = src.cast::<f32>();
    let dst = dst.cast_mut::<f32>();

    //i3
    for batch in 0..src_tensor.batches() {
        // i2
        for depth in 0..src_tensor.depth() {
            let p = n_past + depth as i32;
            let theta_0 = freq_scale * p as f32;
            let inv_ndims = -1. / n_dims as f32;

            // i1
            for row in 0..src_tensor.rows() {
                for col in (0..src_tensor.cols()).step_by(2) {
                    let theta = theta_0 * freq_base.powf(inv_ndims * col as f32);
                    let cos_theta = theta.cos();
                    let sin_theta = theta.sin();

                    let src_offs = batch * src_tensor.batch_stride()
                        + depth * src_tensor.depth_stride()
                        + row * src_tensor.row_stride()
                        + col * src_tensor.col_stride();
                    let src_offs = src_offs / 4;

                    let dst_offs = batch * dst_tensor.batch_stride()
                        + depth * dst_tensor.depth_stride()
                        + row * dst_tensor.row_stride()
                        + col * dst_tensor.col_stride();
                    let dst_offs = dst_offs / 4;

                    let x0 = src[src_offs];
                    let x1 = src[src_offs + 1];

                    dst[dst_offs] = x0 * cos_theta - x1 * sin_theta;
                    dst[dst_offs + 1] = x0 * sin_theta + x1 * cos_theta;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::{cpu::Cpu, Metal},
        builder::*,
        layout::{Type, D2},
        op::Rope,
        Backend, ComputeGraph,
    };
    use approx::assert_relative_eq;

    #[test]
    fn rope_cpu_norm_property() -> anyhow::Result<()> {
        let mut backend = Cpu::new();
        //let mut backend = Metal::new()?;
        let mut tb = TensorBuilder::prealloc(128 * 1024 * 1024)?;

        let src1_t = tb.new_tensor(D2 { cols: 4, rows: 4 }.with_type(Type::F32))?;
        let src1 = (0..16).map(|x| x as f32).collect::<Vec<_>>();

        tb.copy_data_from::<f32>(src1_t, &src1);

        let r0_args = Rope {
            n_past: 4,
            n_dims: 4,
            n_ctx: 16,
            mode: 0,
            custom: None,
        };
        let r0 = tb.rope(src1_t, r0_args)?;

        let (ctx, _) = tb.finish(&mut backend)?;
        let mut graph = ComputeGraph::default();
        graph.expand_forward(r0, &ctx)?;
        graph.find_exec_plan(backend.mem_limit(), &ctx)?;
        backend.compute_graph(&graph, &ctx)?;

        let result = ctx[r0].elements::<f32>(&backend);

        // Property 1: The norm of each row should be preserved
        for i in 0..4 {
            let input_norm: f32 = src1[i * 4..(i + 1) * 4]
                .iter()
                .map(|&x| x * x)
                .sum::<f32>()
                .sqrt();
            let result_norm: f32 = result[i * 4..(i + 1) * 4]
                .iter()
                .map(|&x| x * x)
                .sum::<f32>()
                .sqrt();
            assert_relative_eq!(input_norm, result_norm, epsilon = 1e-5);
        }

        Ok(())
    }

    #[test]
    fn rope_cpu_dot_property() -> anyhow::Result<()> {
        let mut backend = Cpu::new();
        //let mut backend = Metal::new()?;
        let mut tb = TensorBuilder::prealloc(128 * 1024 * 1024)?;

        let src1_t = tb.new_tensor(D2 { cols: 4, rows: 4 }.with_type(Type::F32))?;
        let src1 = (0..16).map(|x| x as f32).collect::<Vec<_>>();

        tb.copy_data_from::<f32>(src1_t, &src1);

        let r0_args = Rope {
            n_past: 4,
            n_dims: 4,
            n_ctx: 16,
            mode: 0,
            custom: None,
        };
        let r0 = tb.rope(src1_t, r0_args)?;

        let (ctx, _) = tb.finish(&mut backend)?;
        let mut graph = ComputeGraph::default();
        graph.expand_forward(r0, &ctx)?;
        graph.find_exec_plan(backend.mem_limit(), &ctx)?;
        backend.compute_graph(&graph, &ctx)?;

        let result = ctx[r0].elements::<f32>(&backend);

        // Property 2: The dot product between the first half and second half of each row should be zero
        for i in 0..4 {
            let dot_product: f32 = result[i * 4..i * 4 + 4]
                .iter()
                .zip(result[i * 4 + 4..(i + 1) * 4].iter())
                .map(|(&a, &b)| a * b)
                .sum();

            assert_relative_eq!(dot_product, 0.0, epsilon = 1e-5);
        }

        Ok(())
    }

    #[test]
    fn rope_cpu_invariant() -> anyhow::Result<()> {
        let mut backend = Cpu::new();
        //let mut backend = Metal::new()?;
        let mut tb = TensorBuilder::prealloc(128 * 1024 * 1024)?;

        let input = tb
            .new_tensor(D2 { rows: 2, cols: 4 }.with_type(Type::F32))
            .unwrap();
        tb.fill_with(input, || rand::random::<f32>());

        let rope_args = Rope {
            n_past: 0,
            n_dims: 4,
            n_ctx: 32,
            mode: 0,
            custom: None,
        };

        // Apply RoPE twice with different n_past
        let result1 = tb.rope(input, rope_args).unwrap();
        let result2 = tb
            .rope(
                result1,
                Rope {
                    n_past: 1,
                    ..rope_args
                },
            )
            .unwrap();

        // Apply RoPE once with n_past = 1
        let result3 = tb
            .rope(
                input,
                Rope {
                    n_past: 1,
                    ..rope_args
                },
            )
            .unwrap();

        let (ctx, _) = tb.finish(&mut backend)?;
        let mut graph = ComputeGraph::default();
        graph.expand_forward(result2, &ctx)?;
        graph.expand_forward(result3, &ctx)?;
        graph.find_exec_plan(backend.mem_limit(), &ctx)?;
        backend.compute_graph(&graph, &ctx)?;

        // The results should be the same
        let data2 = ctx[result2].elements::<f32>(&backend);
        let data3 = ctx[result3].elements::<f32>(&backend);

        for (v2, v3) in data2.iter().zip(data3.iter()) {
            assert_relative_eq!(v2, v3, epsilon = 1e-5);
        }

        Ok(())
    }
}

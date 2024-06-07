//! Bulk of the operations are tested against ggml, since that was the original inspiration for
//! infernal.
#![allow(unused)]

use std::{fs::File, mem::MaybeUninit, path::Path, time::Duration};

use anyhow::{bail, Context};
use ggml::Backend as _;
use gguf::Dims;
use infernal::{
    alloc::{BumpAlloc, ReadWriteAlloc, SplitAlloc as _},
    buf::SliceCast,
    builder::{AutoOpName, TensorOptionsExt as _, WeightsBuilder},
    op::{Rope, RopeCustom},
    Backend as _, BufId, ComputeGraph, Layout, TensorBuilder, TensorId,
    Type::*,
    D1, D2, D3, D4,
};
use tracing::info;

use crate::common::{init_tracing, MB};

mod common;

// TODO: add missing tests:
// - matmul_mv: this is the most important to check
// - transpose

#[test]
fn ggml_matmul_mm() -> anyhow::Result<()> {
    // Example from LLama:
    // blk.0.attn_k.weight=[Q3_K|1024x4096|1760x110]
    // b=attention_norm_0=[F32|68x4096|16384x4]
    // tmpk=[F32|68x1024|4096x4]
    let b_dims = D2 {
        rows: 68,
        cols: 4096,
    };
    let b_val = 9.5;
    // Load q3_k weights from a model file.
    compare_with_weights(
        "matmul_mm_q3k_f32",
        ["blk.0.attn_k.weight"],
        [],
        |tb, [weight], []| {
            let b = tb.new_tensor(Layout::new(b_dims, F32).with_name("b"))?;
            tb.fill(b, b_val);
            let output = tb.matmul(weight, b)?;
            Ok(output)
        },
        |ctx, [weight], []| {
            let mut b = ctx.create_2d_tensor(
                ggml::TensorType::F32,
                b_dims.cols as u32,
                b_dims.rows as u32,
            );
            b.set_f32(b_val);
            ctx.mul_mat(&weight, &b)
        },
    )
}

#[test]
fn regression_ggml_matmul_mm() -> anyhow::Result<()> {
    // FIXME:
    // using MM for tmpv:
    // blk.0.attn_v.weight=[Q5_K|1024x4096|2816x176]
    // attention_norm_0=[F32|16x4096|16384x4]
    // tmpv=[F32|16x1024|4096x4]
    compare_with_weights(
        "regression_matmul_mm_q5k_f32",
        ["blk.0.attn_v.weight"],
        [(
            D2 {
                rows: 16,
                cols: 4096,
            }
            .into(),
            Path::new("tests/data/lvl_002_node_0002_dst_attention_norm_0"),
        )],
        |tb, [attn_v], [attn_norm]| {
            let output = tb.matmul(attn_v, attn_norm)?;
            Ok(output)
        },
        |ctx, [attn_v], [attn_norm]| ctx.mul_mat(&attn_v, &attn_norm),
    )
}

// TODO: ggml does not implement f32 on lhs
// #[test]
// fn ggml_matmul_mv() -> anyhow::Result<()> {
//     let a_dims = D2 {
//         rows: 68,
//         cols: 4096,
//     };
//     let b_dims = D1 { cols: 4096 };
//     let a_val = 156.3;
//     let b_val = 9.5;

//     compare(
//         "matmul_mv",
//         |tb| {
//             let a = tb.new_tensor(Layout::new(a_dims, F32).with_name("a"))?;
//             let b = tb.new_tensor(Layout::new(b_dims, F32).with_name("b"))?;
//             tb.fill(a, a_val);
//             tb.fill(b, b_val);
//             let output = tb.matmul(a, b)?;
//             Ok(output)
//         },
//         |ctx| {
//             let mut a = ctx.create_2d_tensor(
//                 ggml::TensorType::F32,
//                 a_dims.cols as u32,
//                 a_dims.rows as u32,
//             );
//             a.set_f32(a_val);
//             let mut b = ctx.create_1d_tensor(ggml::TensorType::F32, b_dims.cols as u32);
//             b.set_f32(b_val);
//             ctx.mul_mat(&a, &b)
//         },
//     )
// }

#[test]
fn ggml_mul_broadcast() -> anyhow::Result<()> {
    // a=rms_norm_0 a.layout=[F32|68x4096|16384x4]
    // b=blk.1.attn_norm.weight b.layout=[F32|4096|4]
    // dst=attention_norm_0 dst.layout=[F32|68x4096|16384x4]
    let a_dims = D2 {
        rows: 68,
        cols: 4096,
    };
    let b_dims = D1 { cols: 4096 };
    let a_val = 156.3;
    let b_val = 9.5;

    compare(
        "mul_broadcast",
        |tb| {
            let a = tb.new_tensor(Layout::new(a_dims, F32).with_name("a"))?;
            let b = tb.new_tensor(Layout::new(b_dims, F32).with_name("b"))?;
            tb.fill(a, a_val);
            tb.fill(b, b_val);
            let output = tb.mul(a, b)?;
            Ok(output)
        },
        |ctx| {
            let mut a = ctx.create_2d_tensor(
                ggml::TensorType::F32,
                a_dims.cols as u32,
                a_dims.rows as u32,
            );
            a.set_f32(a_val);
            let mut b = ctx.create_1d_tensor(ggml::TensorType::F32, b_dims.cols as u32);
            b.set_f32(b_val);
            ctx.mul(&a, &b)
        },
    )
}

#[test]
fn ggml_mul() -> anyhow::Result<()> {
    // a=silu a.layout=[F32|247x14336|57344x4]
    // b=result_w3 b.layout=[F32|247x14336|57344x4]
    // dst=silu_x_result_w3 dst.layout=[F32|247x14336|57344x4]
    let dims = D2 {
        rows: 247,
        cols: 14336,
    };
    let a_val = 156.3;
    let b_val = 9.5;

    compare(
        "mul",
        |tb| {
            let a = tb.new_tensor(Layout::new(dims, F32).with_name("a"))?;
            let b = tb.new_tensor(Layout::new(dims, F32).with_name("b"))?;
            tb.fill(a, a_val);
            tb.fill(b, b_val);
            let output = tb.mul(a, b)?;
            Ok(output)
        },
        |ctx| {
            let mut a =
                ctx.create_2d_tensor(ggml::TensorType::F32, dims.cols as u32, dims.rows as u32);
            a.set_f32(a_val);
            let mut b =
                ctx.create_2d_tensor(ggml::TensorType::F32, dims.cols as u32, dims.rows as u32);
            b.set_f32(b_val);
            ctx.mul(&a, &b)
        },
    )
}

#[test]
fn ggml_add() -> anyhow::Result<()> {
    let dims = D2 {
        rows: 5 * 32,
        cols: 10 * 32,
    };
    let a_val = 1.3;
    let b_val = 9.5;

    let ggml = |ctx: &mut ggml::ManagedContext| {
        let mut a = ctx.create_2d_tensor(ggml::TensorType::F32, dims.cols as u32, dims.rows as u32);
        a.set_f32(a_val);
        let mut b = ctx.create_2d_tensor(ggml::TensorType::F32, dims.cols as u32, dims.rows as u32);
        b.set_f32(b_val);
        ctx.add(&a, &b)
    };

    compare(
        "add",
        |tb| {
            let a = tb.new_tensor(Layout::new(dims, F32).with_name("a"))?;
            let b = tb.new_tensor(Layout::new(dims, F32).with_name("b"))?;
            tb.fill(a, a_val);
            tb.fill(b, b_val);
            let output = tb.add(a, b)?;
            Ok(output)
        },
        ggml,
    )?;

    // NB: We do 1 iteration only because inplace op modifies source tensor and would invalidate
    // further results.
    compare_with_n_runs(
        "add_inplace",
        1,
        |tb| {
            let a = tb.new_tensor(Layout::new(dims, F32).with_name("a"))?;
            let b = tb.new_tensor(Layout::new(dims, F32).with_name("b"))?;
            tb.fill(a, a_val);
            tb.fill(b, b_val);
            let output = tb.add_inplace(a, b);
            Ok(output)
        },
        ggml,
    )
}

#[test]
fn ggml_rms_norm() -> anyhow::Result<()> {
    // [/haltia-core/lib/llm/src/arch/llama.rs:249] f_norm_eps = 1e-5
    let f_norm_eps = 1e-5;

    let rows = 5 * 32;
    let cols = 10 * 32;
    let a_val = 1.3;

    let ggml = |ctx: &mut ggml::ManagedContext| {
        let mut a = ctx.create_2d_tensor(ggml::TensorType::F32, cols as u32, rows as u32);
        a.set_f32(a_val);
        ctx.rms_norm(&a, f_norm_eps)
    };

    compare(
        "rms_norm",
        |tb| {
            let a = tb.new_tensor(Layout::new_2d(rows, cols, F32).with_name("a"))?;
            tb.fill(a, a_val);
            let output = tb.rms_norm(a, f_norm_eps)?;
            Ok(output)
        },
        ggml,
    )?;

    compare_with_n_runs(
        "rms_norm_inplace",
        1,
        |tb| {
            let a = tb.new_tensor(Layout::new_2d(rows, cols, F32).with_name("a"))?;
            tb.fill(a, a_val);
            let output = tb.rms_norm_inplace(a, f_norm_eps);
            Ok(output)
        },
        ggml,
    )
}

#[test]
fn ggml_rope() -> anyhow::Result<()> {
    // [/haltia-core/lib/llm/src/arch/llama.rs:249] rope_freq_base = 10000.0
    // [/haltia-core/lib/llm/src/arch/llama.rs:249] rope_freq_scale = 1.0
    // [/haltia-core/lib/llm/src/arch/llama.rs:252] n_past = 221
    // [/haltia-core/lib/llm/src/arch/llama.rs:252] n_embd_head = 128
    let n_past = 221;
    let n_dims = 128;
    let mode = 0;
    let n_ctx = 0;
    let freq_base = 10_000.;
    let freq_scale = 1.;
    let rope = Rope {
        n_past,
        n_dims,
        mode,
        n_ctx,
        custom: Some(RopeCustom {
            freq_base,
            freq_scale,
        }),
    };

    let rows = 5 * 32;
    let cols = 10 * 32;
    let a_val = 1.3;

    let ggml = |ctx: &mut ggml::ManagedContext| {
        let mut a = ctx.create_2d_tensor(ggml::TensorType::F32, cols as u32, rows as u32);
        a.set_f32(a_val);
        ctx.rope_custom_inplace(
            &a,
            n_past as u32,
            n_dims as u32,
            mode as i32,
            n_ctx as u32,
            freq_base,
            freq_scale,
        )
    };

    compare_with_n_runs(
        "rope",
        1,
        |tb| {
            let a = tb.new_tensor(Layout::new_2d(rows, cols, F32).with_name("a"))?;
            tb.fill(a, a_val);
            let output = tb.rope(a, rope)?;
            Ok(output)
        },
        ggml,
    )?;

    compare_with_n_runs(
        "rope_inplace",
        1,
        |tb| {
            let a = tb.new_tensor(Layout::new_2d(rows, cols, F32).with_name("a"))?;
            tb.fill(a, a_val);
            let output = tb.rope_inplace(a, rope);
            Ok(output)
        },
        ggml,
    )
}

#[test]
fn ggml_silu() -> anyhow::Result<()> {
    let rows = 5 * 32;
    let cols = 10 * 32;
    let a_val = 1.3;

    let ggml = |ctx: &mut ggml::ManagedContext| {
        let mut a = ctx.create_2d_tensor(ggml::TensorType::F32, cols as u32, rows as u32);
        a.set_f32(a_val);
        ctx.silu(&a)
    };

    compare(
        "silu",
        |tb| {
            let a = tb.new_tensor(Layout::new_2d(rows, cols, F32).with_name("a"))?;
            tb.fill(a, a_val);
            let output = tb.silu(a)?;
            Ok(output)
        },
        ggml,
    )?;

    compare_with_n_runs(
        "silu_inplace",
        1,
        |tb| {
            let a = tb.new_tensor(Layout::new_2d(rows, cols, F32).with_name("a"))?;
            tb.fill(a, a_val);
            let output = tb.silu_inplace(a);
            Ok(output)
        },
        ggml,
    )
}

#[test]
fn ggml_copy() -> anyhow::Result<()> {
    let rows = 5 * 32;
    let cols = 10 * 32;
    let a_val = 1.3;

    compare(
        "copy",
        |tb| {
            let a = tb.new_tensor(Layout::new_2d(rows, cols, F32).with_name("a"))?;
            tb.fill(a, a_val);
            let b = tb.new_tensor(Layout::new_2d(rows, cols, F32).with_name("a"))?;
            let output = tb.copy(a, b);
            Ok(output)
        },
        |ctx| {
            let mut a = ctx.create_2d_tensor(ggml::TensorType::F32, cols as u32, rows as u32);
            a.set_f32(a_val);
            let b = ctx.create_2d_tensor(ggml::TensorType::F32, cols as u32, rows as u32);
            ctx.copy(&a, b)
        },
    )
}

#[test]
fn ggml_view_1d() -> anyhow::Result<()> {
    let rows = 5 * 32;
    let cols = 10 * 32;
    let offs = 3;
    let new_cols = rows * cols - offs;

    compare(
        "view_1d",
        |tb| {
            let a = tb.new_tensor(D2 { rows, cols }.with_name("a"))?;
            let mut val_iter = 0..;
            tb.fill_with(a, || val_iter.next().unwrap() as f32);
            let output = tb.view_1d(a, new_cols, offs);
            Ok(output)
        },
        |ctx| {
            let a = ctx.create_2d_tensor(ggml::TensorType::F32, cols as u32, rows as u32);
            let mut a_view = ctx.view_1d(&a, (cols * rows) as u32, 0);
            for i in 0..cols * rows {
                a_view.set_f32_1d_at(i, i as f32);
            }
            ctx.view_1d(&a, new_cols as u32, offs as u32)
        },
    )
}

#[test]
fn ggml_nested_view_1d() -> anyhow::Result<()> {
    let rows = 5 * 32;
    let cols = 10 * 32;
    let offs0 = 3;
    let new_cols0 = rows * cols - offs0;
    let offs1 = 2 * offs0;
    let new_cols1 = rows - offs1;

    compare(
        "nested_view_1d",
        |tb| {
            let a = tb.new_tensor(D2 { rows, cols }.with_name("a"))?;
            let mut val_iter = 0..;
            tb.fill_with(a, || val_iter.next().unwrap() as f32);
            let view0 = tb.view_1d(a, new_cols0, offs0);
            let output = tb.view_1d(view0, new_cols1, offs1);
            Ok(output)
        },
        |ctx| {
            let a = ctx.create_2d_tensor(ggml::TensorType::F32, cols as u32, rows as u32);
            let mut a_view = ctx.view_1d(&a, (cols * rows) as u32, 0);
            for i in 0..cols * rows {
                a_view.set_f32_1d_at(i, i as f32);
            }
            let view0 = ctx.view_1d(&a, new_cols0 as u32, offs0 as u32);
            ctx.view_1d(&view0, new_cols1 as u32, offs1 as u32)
        },
    )
}

#[test]
fn ggml_view_2d() -> anyhow::Result<()> {
    let depth = 2 * 32;
    let rows = 5 * 32;
    let cols = 10 * 32;
    let offs = 3;
    let new_rows = 2 * rows;
    let new_cols = cols - offs;
    let new_row_stride = 32;

    compare(
        "view_2d",
        |tb| {
            let a = tb.new_tensor(D3 { depth, rows, cols }.with_name("a"))?;
            let mut val_iter = 0..;
            tb.fill_with(a, || val_iter.next().unwrap() as f32);
            let output = tb.view_2d(
                a,
                D2 {
                    rows: new_rows,
                    cols: new_cols,
                },
                D2 {
                    rows: new_row_stride,
                    ..Default::default()
                },
                offs,
            );
            Ok(output)
        },
        |ctx| {
            let a = ctx.create_3d_tensor(
                ggml::TensorType::F32,
                cols as u32,
                rows as u32,
                depth as u32,
            );
            let mut a_view = ctx.view_1d(&a, (depth * cols * rows) as u32, 0);
            for i in 0..depth * cols * rows {
                a_view.set_f32_1d_at(i, i as f32);
            }
            ctx.view_2d(
                &a,
                new_cols as u32,
                new_rows as u32,
                new_row_stride as u32,
                offs as u32,
            )
        },
    )
}

#[test]
fn ggml_view_3d() -> anyhow::Result<()> {
    let batches = 3;
    let depth = 32;
    let rows = 2 * 32;
    let cols = 3 * 32;
    let offs = 3;
    let new_depth = depth;
    let new_rows = rows;
    let new_cols = cols - offs;
    let new_depth_stride;
    let new_row_stride;
    {
        let layout = Layout::from(D3 {
            depth: new_depth,
            rows: new_rows,
            cols: new_cols,
        });
        new_depth_stride = layout.depth_stride();
        new_row_stride = layout.row_stride();
    }

    compare(
        "view_3d",
        |tb| {
            let a = tb.new_tensor(
                D4 {
                    batches,
                    depth,
                    rows,
                    cols,
                }
                .with_name("a"),
            )?;
            let mut val_iter = 0..;
            tb.fill_with(a, || val_iter.next().unwrap() as f32);
            let output = tb.view_3d(
                a,
                D3 {
                    depth: new_depth,
                    rows: new_rows,
                    cols: new_cols,
                },
                D3 {
                    depth: new_depth_stride,
                    rows: new_row_stride,
                    ..Default::default()
                },
                offs,
            );
            Ok(output)
        },
        |ctx| {
            let a = ctx.create_4d_tensor(
                ggml::TensorType::F32,
                cols as u32,
                rows as u32,
                depth as u32,
                batches as u32,
            );
            let mut a_view = ctx.view_1d(&a, (depth * cols * rows) as u32, 0);
            for i in 0..batches * depth * cols * rows {
                a_view.set_f32_1d_at(i, i as f32);
            }
            ctx.view_3d(
                &a,
                new_cols as u32,
                new_rows as u32,
                new_depth as u32,
                new_row_stride as u32,
                new_depth_stride as u32,
                offs as u32,
            )
        },
    )
}

#[test]
fn ggml_diag_mask_inf() -> anyhow::Result<()> {
    let rows = 100;
    let cols = rows;
    let n_past = 1;
    let a_val = 1.3;

    let ggml = |ctx: &mut ggml::ManagedContext| {
        let mut a = ctx.create_2d_tensor(ggml::TensorType::F32, cols as u32, rows as u32);
        a.set_f32(a_val);
        ctx.diag_mask_inf_inplace(&a, n_past as u32)
    };

    compare(
        "diag_mask_inf",
        |tb| {
            let a = tb.new_tensor(Layout::new_2d(rows, cols, F32).with_name("a"))?;
            tb.fill(a, a_val);
            let output = tb.diag_mask_inf(a, n_past)?;
            Ok(output)
        },
        ggml,
    )?;

    compare_with_n_runs(
        "diag_mask_inf_inplace",
        1,
        |tb| {
            let a = tb.new_tensor(Layout::new_2d(rows, cols, F32).with_name("a"))?;
            tb.fill(a, a_val);
            let output = tb.diag_mask_inf_inplace(a, n_past);
            Ok(output)
        },
        ggml,
    )
}

#[test]
fn ggml_softmax() -> anyhow::Result<()> {
    let rows = 5 * 32;
    let cols = 10 * 32;
    let a_val = 1.3;

    let ggml = |ctx: &mut ggml::ManagedContext| {
        let mut a = ctx.create_2d_tensor(ggml::TensorType::F32, cols as u32, rows as u32);
        a.set_f32(a_val);
        ctx.softmax_inplace(&a)
    };

    compare(
        "softmax",
        |tb| {
            let a = tb.new_tensor(Layout::new_2d(rows, cols, F32).with_name("a"))?;
            tb.fill(a, a_val);
            let output = tb.softmax(a)?;
            Ok(output)
        },
        ggml,
    )?;

    compare_with_n_runs(
        "softmax_inplace",
        1,
        |tb| {
            let a = tb.new_tensor(Layout::new_2d(rows, cols, F32).with_name("a"))?;
            tb.fill(a, a_val);
            let output = tb.softmax_inplace(a);
            Ok(output)
        },
        ggml,
    )
}

#[test]
fn ggml_scale() -> anyhow::Result<()> {
    // Example LLama output:
    // scale_inplace: scaled=[222, 1, 32, 1], scalar=[1, 1, 1, 1]
    let cols = 10 * 32;
    let a_val = 1.3;
    let scale = 0.5;

    let ggml = |ctx: &mut ggml::ManagedContext| {
        let mut a = ctx.create_1d_tensor(ggml::TensorType::F32, cols as u32);
        a.set_f32(a_val);
        let mut scalar = ctx.create_1d_tensor(ggml::TensorType::F32, 1);
        scalar.set_f32(scale);
        ctx.scale_inplace(&a, &scalar)
    };

    compare_with_n_runs(
        "scale",
        1,
        |tb| {
            let a = tb.new_tensor(Layout::new_1d(cols, F32).with_name("a"))?;
            tb.fill_1d_f32(a, a_val);
            let scalar = tb.new_tensor(Layout::new_1d(1, F32).with_name("scalar"))?;
            tb.fill_1d_f32(scalar, scale);
            // let output = tb.scale_inplace(a, scalar);
            let output = tb.scale(a, scalar)?;
            Ok(output)
        },
        ggml,
    )?;

    compare_with_n_runs(
        "scale_inplace",
        1,
        |tb| {
            let a = tb.new_tensor(Layout::new_1d(cols, F32).with_name("a"))?;
            tb.fill_1d_f32(a, a_val);
            let scalar = tb.new_tensor(Layout::new_1d(1, F32).with_name("scalar"))?;
            tb.fill_1d_f32(scalar, scale);
            let output = tb.scale_inplace(a, scalar);
            Ok(output)
        },
        ggml,
    )
}

#[test]
fn ggml_get_rows() -> anyhow::Result<()> {
    // a=token_embd.weight a.layout=[Q3_K|32002x4096|1760x110]
    // b=inp_tokens b.layout=[I32|68|4]
    // dst=layer_inp_0 dst.layout=[F32|68x4096|16384x4]
    let tokens = [1, 6312, 28709, 1526, 13, 5845, 460, 368, 28804];
    let b_dims = D1 { cols: tokens.len() };
    compare_with_weights(
        "get_rows_2d",
        ["token_embd.weight"],
        [],
        |tb, [weight], []| {
            let idx = tb.new_tensor(Layout::new(b_dims, I32).with_name("tokens"))?;
            let mut iter = tokens.iter().copied();
            tb.fill_with(idx, || iter.next().unwrap());
            let output = tb.get_rows(weight, idx)?;
            Ok(output)
        },
        |ctx, [weight], []| {
            let mut idx = ctx.create_1d_tensor(ggml::TensorType::I32, tokens.len() as u32);
            for (i, tok) in tokens.iter().enumerate() {
                idx.set_i32_1d_at(i, *tok);
            }
            ctx.get_rows(&weight, &idx)
        },
    )
}

#[test]
fn ggml_permute() -> anyhow::Result<()> {
    // a=Qcur a.layout=[F32|247x32x128|16384x512x4]
    // dst=Q dst.layout=[F32|32x247x128|512x16384x4]
    let dims = D3 {
        depth: 247,
        rows: 32,
        cols: 128,
    };
    let axes = [0, 2, 1, 3]; // taken from arch::llama
    compare(
        "permute_3d",
        |tb| {
            let a = tb.new_tensor(Layout::new(dims, F32).with_name("a"))?;
            let mut iter = 0..tb[a].total_elems();
            tb.fill_with(a, || iter.next().unwrap() as f32);
            let output = tb.permute(a, axes);
            Ok(output)
        },
        |ctx| {
            let mut a = ctx.create_3d_tensor(
                ggml::TensorType::F32,
                dims.cols as u32,
                dims.rows as u32,
                dims.depth as u32,
            );
            let n = a.total_elems();
            for i in 0..n {
                a.set_f32_1d_at(i, i as f32);
            }
            ctx.permute(
                &a,
                axes[0] as i32,
                axes[1] as i32,
                axes[2] as i32,
                axes[3] as i32,
            )
        },
    )
}

#[test]
fn ggml_transpose() -> anyhow::Result<()> {
    // a=tmpv (reshaped) a.layout=[F32|247x1024|4096x4]
    // dst=Vcur dst.layout=[F32|1024x247|4x4096]
    let dims = D2 {
        rows: 247,
        cols: 1024,
    };
    compare(
        "tranpose_2d",
        |tb| {
            let a = tb.new_tensor(Layout::new(dims, F32).with_name("a"))?;
            let mut iter = 0..tb[a].total_elems();
            tb.fill_with(a, || iter.next().unwrap() as f32);
            let output = tb.transpose(a);
            Ok(output)
        },
        |ctx| {
            let mut a =
                ctx.create_2d_tensor(ggml::TensorType::F32, dims.cols as u32, dims.rows as u32);
            let data: &mut [f32] = a.data_mut().cast_mut();
            for (i, d) in data.iter_mut().enumerate() {
                *d = i as f32;
            }
            ctx.transpose(&a)
        },
    )
}

#[test]
fn ggml_reshape() -> anyhow::Result<()> {
    // a=tmpv a.layout=[F32|247x1024|4096x4]
    // dst=tmpv (reshaped) dst.layout=[F32|247x1024|4096x4]
    let dims = D2 {
        rows: 247,
        cols: 1024,
    };
    let reshape = D2 {
        rows: 247,
        cols: 4096 / (32 / 8), // 1024
    };
    compare(
        "reshape_2d",
        |tb| {
            let a = tb.new_tensor(Layout::new(dims, F32).with_name("a"))?;
            let mut iter = 0..tb[a].total_elems();
            tb.fill_with(a, || iter.next().unwrap() as f32);
            let output = tb.reshape(a, reshape);
            Ok(output)
        },
        |ctx| {
            let mut a =
                ctx.create_2d_tensor(ggml::TensorType::F32, dims.cols as u32, dims.rows as u32);
            let n = a.total_elems();
            for i in 0..n {
                a.set_f32_1d_at(i, i as f32);
            }
            ctx.reshape_2d(&a, reshape.cols as u32, reshape.rows as u32)
        },
    )?;

    // a=tmpk a.layout=[F32|247x1024|4096x4]
    // dst=tmpk (reshaped) dst.layout=[F32|247x8x128|4096x512x4]
    // n=247 n_embd=4096, n_ctx=512, n_head=32, n_head_kv=8, n_layer=32
    let reshape = D3 {
        depth: 247,
        rows: 8,
        cols: 4096 / 32, // 128
    };
    compare(
        "reshape_3d",
        |tb| {
            let a = tb.new_tensor(Layout::new(dims, F32).with_name("a"))?;
            let mut iter = 0..tb[a].total_elems();
            tb.fill_with(a, || iter.next().unwrap() as f32);
            let output = tb.reshape(a, reshape);
            Ok(output)
        },
        |ctx| {
            let mut a =
                ctx.create_2d_tensor(ggml::TensorType::F32, dims.cols as u32, dims.rows as u32);
            let n = a.total_elems();
            for i in 0..n {
                a.set_f32_1d_at(i, i as f32);
            }
            ctx.reshape_3d(
                &a,
                reshape.cols as u32,
                reshape.rows as u32,
                reshape.depth as u32,
            )
        },
    )
}

// TODO: not impl'd
// #[test]
// fn ggml_conv_1d() -> anyhow::Result<()> {
//     let rows = 5 * 32;
//     let cols = 10 * 32;
//     let a_val = 1.3;

//     compare(
//         "conv_1d",
//         |tb| {
//             let a = tb.new_tensor(Layout::new_2d(rows, cols, F32).with_name("a"))?;
//             tb.fill_2d_tensor(a, a_val);
//             let output = tb.conv_1d(a);
//             Ok(output)
//         },
//         |ctx| {
//             let mut a = ctx.create_2d_tensor(ggml::TensorType::F32, cols as u32, rows as u32);
//             a.set_f32(a_val);
//             ctx.conv_1d(&a)
//         },
//     )
// }

// TODO: not implemented in ggml bindings
// #[test]
// fn ggml_relu() -> anyhow::Result<()> {
//     let rows = 5 * 32;
//     let cols = 10 * 32;
//     let a_val = 1.3;

//     compare(
//         "rms_norm",
//         |tb| {
//             let a = tb.new_tensor(Layout::new_2d(rows, cols, F32).with_name("a"))?;
//             tb.fill_2d_tensor(a, a_val);
//             let output = tb.relu(a)?;
//             Ok(output)
//         },
//         |ctx| {
//             let mut a = ctx.create_2d_tensor(ggml::TensorType::F32, cols as u32, rows as u32);
//             a.set_f32(a_val);
//             ctx.relu(&a)
//         },
//     )
// }

fn compare<I, G>(
    test_case: &'static str,
    infernal_factory: I,
    ggml_factory: G,
) -> anyhow::Result<()>
where
    I: FnOnce(&mut TensorBuilder<BumpAlloc>) -> anyhow::Result<TensorId>,
    G: FnOnce(&mut ggml::Context<ggml::ManagedStorageTag>) -> ggml::Tensor,
{
    compare_with_n_runs(test_case, N_RUNS, infernal_factory, ggml_factory)
}

fn compare_with_n_runs<I, G>(
    test_case: &'static str,
    n_runs: u32,
    infernal_factory: I,
    ggml_factory: G,
) -> anyhow::Result<()>
where
    I: FnOnce(&mut TensorBuilder<BumpAlloc>) -> anyhow::Result<TensorId>,
    G: FnOnce(&mut ggml::Context<ggml::ManagedStorageTag>) -> ggml::Tensor,
{
    init_tracing();

    let mut infernal_backend = infernal::backend::Metal::new()?;

    let mut tb = TensorBuilder::prealloc(10 * MB)?
        .with_name(test_case)
        .with_auto_op_names(AutoOpName::Expanded { with_id: false });

    let infernal_output = infernal_factory(&mut tb)?;
    let (infernal_ctx, _) = tb.finish(&mut infernal_backend)?;

    let mut infernal_graph = ComputeGraph::default();
    infernal_graph.expand_forward(infernal_output, &infernal_ctx)?;
    infernal_graph.find_exec_plan(infernal_backend.mem_limit(), &infernal_ctx)?;

    let mut infernal_time = Duration::default();
    for _ in 0..n_runs {
        let start = std::time::Instant::now();
        infernal_backend.compute_graph(&infernal_graph, &infernal_ctx)?;
        let time = start.elapsed();
        info!(time = ?time, "computed infernal graph");
        infernal_time += time;
    }
    let infernal_avg_time = infernal_time / n_runs;

    // ggml

    {
        let mut ggml_ctx = ggml::Context::new_managed(10 * MB)?;
        let mut ggml_output = ggml_factory(&mut ggml_ctx).with_name("result_output");
        let mut ggml_graph = ggml_ctx.create_graph();
        ggml_graph.build_forward_expand(&mut ggml_output);

        let mut ggml_backend = ggml::Metal::new()?;
        ggml_backend.add_buffer(ggml_ctx.internal_data(), "ctx", None)?;
        ggml_backend.optimize_graph(&mut ggml_graph);

        let mut ggml_time = Duration::default();
        for _ in 0..n_runs {
            let start = std::time::Instant::now();
            ggml_backend.compute_graph(&mut ggml_graph, 1);
            let time = start.elapsed();
            info!(time = ?time, "computed ggml graph");
            ggml_time += time;
        }
        let ggml_avg_time = ggml_time / n_runs;

        compare_results(
            test_case,
            CmpArgs {
                time: infernal_time,
                avg_time: infernal_avg_time,
                backend: &infernal_backend,
                ctx: &infernal_ctx,
                output: infernal_output,
            },
            CmpArgs {
                time: ggml_time,
                avg_time: ggml_avg_time,
                backend: &ggml_backend,
                ctx: &(),
                output: ggml_output,
            },
        );
    }

    Ok(())
}

fn ggml_to_layout(t: &ggml::Tensor) -> Layout {
    let dims = t.dims();
    let mut dims_arr = [1; 4];
    dims_arr[..dims.n()].copy_from_slice(dims.as_slice());
    Layout::from_raw(dims.n(), dims_arr, *t.stride(), t.typ().try_into().unwrap())
}

fn compare_with_weights<I, G, const N_WEIGHTS: usize, const N_DATA: usize>(
    test_case: &'static str,
    weights: [&str; N_WEIGHTS],
    data: [(Dims, &Path); N_DATA],
    infernal_factory: I,
    ggml_factory: G,
) -> anyhow::Result<()>
where
    I: FnOnce(
        &mut TensorBuilder<BumpAlloc, &WeightsBuilder>,
        [TensorId; N_WEIGHTS],
        [TensorId; N_DATA],
    ) -> anyhow::Result<TensorId>,
    G: FnOnce(
        &mut ggml::Context<ggml::ManagedStorageTag>,
        [ggml::Tensor; N_WEIGHTS],
        [ggml::Tensor; N_DATA],
    ) -> ggml::Tensor,
{
    init_tracing();

    let path = dirs::home_dir()
        .expect("$HOME must be set")
        .join(".config/haltia/default.gguf");

    // Load mmap and the specified weights.

    let file = File::open(&path)?;
    let (gguf, file) = gguf::Context::load(file)?;
    let mut weights_tb =
        WeightsBuilder::mmap(file, gguf.data_offset)?.with_name(path.to_str().unwrap().to_string());

    let mut infernal_backend = infernal::backend::Metal::new()?;

    let mut infernal_weight_ids: [MaybeUninit<TensorId>; N_WEIGHTS] =
        unsafe { MaybeUninit::uninit().assume_init() };
    let mut weight_infos = Vec::with_capacity(weights.len());
    let mut infernal_mmap_id = BufId::default();
    for (weight, infernal_weight_id) in weights.iter().zip(&mut infernal_weight_ids) {
        let info = gguf
            .tensors
            .iter()
            .find(|t| t.name == weight.as_bytes())
            .context(format!("weight tensor '{weight}' not present in model"))?;
        let layout = Layout::new(info.dims, info.typ.try_into()?);
        let tensor_id = weights_tb.new_mapped_tensor(
            layout.with_name(std::str::from_utf8(&info.name)?.to_string()),
            info.offset,
        )?;
        infernal_mmap_id = weights_tb[tensor_id].buf_id();
        infernal_weight_id.write(tensor_id);
        weight_infos.push(info.clone());
    }
    let infernal_weight_ids =
        unsafe { std::mem::transmute_copy::<_, [TensorId; N_WEIGHTS]>(&infernal_weight_ids) };

    // infernal

    let mut tb = TensorBuilder::prealloc_with_src(10 * MB, &weights_tb)?
        .with_name(test_case)
        .with_auto_op_names(AutoOpName::Expanded { with_id: false });

    // Load specified tensor data if any.
    // Don't forget to add buffer to the backend for ggml.
    let mut infernal_data_ids: [MaybeUninit<TensorId>; N_DATA] =
        unsafe { MaybeUninit::uninit().assume_init() };
    let mut data_bufs = Vec::with_capacity(N_DATA);
    for (i, (dims, path)) in data.iter().enumerate() {
        let data = std::fs::read(path)?;
        let elems: &[f32] = data[..].cast();

        let id =
            tb.new_tensor(Layout::new(*dims, F32).with_name(path.to_string_lossy().to_string()))?;
        debug_assert_eq!(tb[id].total_elems(), elems.len());
        let mut elems_iter = elems.iter().copied();
        tb.fill_with(id, || elems_iter.next().unwrap() as f32);
        debug_assert!(elems_iter.next().is_none());

        // Sanity check:
        let tensor_buf = tb
            .alloc()
            .get_slice(tb[id].buf_id(), tb[id].data_offs(), tb[id].size())
            .unwrap();
        assert_eq!(tensor_buf, data);

        infernal_data_ids[i].write(id);
        data_bufs.push(data);
    }
    let infernal_data_ids =
        unsafe { std::mem::transmute_copy::<_, [TensorId; N_DATA]>(&infernal_data_ids) };

    let infernal_output = infernal_factory(&mut tb, infernal_weight_ids, infernal_data_ids)?;
    info!(n_tensors = tb.len(), "allocated infernal tensors");
    let (infernal_ctx, _) = tb.finish(&mut infernal_backend)?;

    info!(n_tensors = weights_tb.len(), "allocated infernal weights");
    let (weights_ctx, ..) = weights_tb.finish(&mut infernal_backend)?;

    let mut infernal_graph = ComputeGraph::default();
    infernal_graph.expand_forward(infernal_output, &[&weights_ctx, &infernal_ctx])?;
    infernal_graph.find_exec_plan(infernal_backend.mem_limit(), &[&weights_ctx, &infernal_ctx])?;
    info!("built infernal graph");

    const N_RUNS: u32 = 1;

    let mut infernal_time = Duration::default();
    for _ in 0..N_RUNS {
        let start = std::time::Instant::now();
        infernal_backend.compute_graph(&infernal_graph, &[&weights_ctx, &infernal_ctx])?;
        let time = start.elapsed();
        info!(time = ?time, "computed infernal graph");
        infernal_time += time;
    }
    let infernal_avg_time = infernal_time / N_RUNS;

    let buf = infernal_backend.get_buf(infernal_mmap_id).unwrap();
    let mmap = buf.as_mmap().unwrap();

    // ggml

    {
        let weights_ctx = ggml::Context::with_mmap_ref(MB, mmap)?;
        let mut ctx = ggml::Context::new_managed(10 * MB)?;

        let mut ggml_backend = ggml::Metal::new()?;
        ggml_backend.add_buffer(weights_ctx.mmap_data(), "ctx", None)?;
        ggml_backend.add_buffer(ctx.internal_data(), "ctx", None)?;

        let mut ggml_weights: [MaybeUninit<ggml::Tensor>; N_WEIGHTS] =
            unsafe { MaybeUninit::uninit().assume_init() };
        for (info, weight) in weight_infos.iter().zip(&mut ggml_weights) {
            let tensor = weights_ctx.create_tensor(gguf.data_offset as u64, info);
            weight.write(tensor);
        }
        let ggml_weights =
            unsafe { std::mem::transmute_copy::<_, [ggml::Tensor; N_WEIGHTS]>(&ggml_weights) };
        info!(n_tensors = ggml_weights.len(), "allocated ggml weights");

        let mut ggml_data_tensors: [MaybeUninit<ggml::Tensor>; N_DATA] =
            unsafe { MaybeUninit::uninit().assume_init() };
        for (i, ((dims, path), data)) in data.iter().zip(&data_bufs).enumerate() {
            let mut tensor = match dims.n() {
                4 => ctx.create_4d_tensor(
                    ggml::TensorType::F32,
                    dims.cols() as u32,
                    dims.rows() as u32,
                    dims.depth() as u32,
                    dims.batches() as u32,
                ),
                3 => ctx.create_3d_tensor(
                    ggml::TensorType::F32,
                    dims.cols() as u32,
                    dims.rows() as u32,
                    dims.depth() as u32,
                ),
                2 => ctx.create_2d_tensor(
                    ggml::TensorType::F32,
                    dims.cols() as u32,
                    dims.rows() as u32,
                ),
                1 => ctx.create_1d_tensor(ggml::TensorType::F32, dims.cols() as u32),
                _ => bail!("unsupported dims: {:?}", dims),
            }
            .with_name(path.to_string_lossy().as_bytes());

            let elems: &[f32] = data[..].cast();
            debug_assert_eq!(tensor.total_elems(), elems.len());

            tensor.set_data(data.as_ptr() as *const _);
            ggml_backend.add_buffer(data, path.to_string_lossy().as_bytes(), None)?;

            ggml_data_tensors[i].write(tensor);
        }
        let ggml_data_tensors =
            unsafe { std::mem::transmute_copy::<_, [ggml::Tensor; N_DATA]>(&ggml_data_tensors) };

        let mut ggml_output =
            ggml_factory(&mut ctx, ggml_weights, ggml_data_tensors).with_name("result_output");
        let mut ggml_graph = ctx.create_graph();
        ggml_graph.build_forward_expand(&mut ggml_output);
        ggml_backend.optimize_graph(&mut ggml_graph);
        info!("built ggml graph");

        let mut ggml_time = Duration::default();
        for _ in 0..N_RUNS {
            let start = std::time::Instant::now();
            ggml_backend.compute_graph(&mut ggml_graph, 1);
            let time = start.elapsed();
            info!(time = ?time, "computed ggml graph");
            ggml_time += time;
        }
        let ggml_avg_time = ggml_time / N_RUNS;

        compare_results(
            test_case,
            CmpArgs {
                time: infernal_time,
                avg_time: infernal_avg_time,
                backend: &infernal_backend,
                ctx: &infernal_ctx,
                output: infernal_output,
            },
            CmpArgs {
                time: ggml_time,
                avg_time: ggml_avg_time,
                backend: &ggml_backend,
                ctx: &(),
                output: ggml_output,
            },
        );
    }

    Ok(())
}

struct CmpArgs<'a, B, C, T> {
    time: Duration,
    avg_time: Duration,
    backend: &'a B,
    ctx: &'a C,
    output: T,
}

fn compare_results(
    test_case: &'static str,
    infernal: CmpArgs<'_, infernal::backend::Metal, infernal::TensorCtx, TensorId>,
    ggml: CmpArgs<'_, ggml::Metal, (), ggml::Tensor>,
    // infernal_time: Duration,
    // infernal_avg_time: Duration,
    // ggml_time: Duration,
    // ggml_avg_time: Duration,
    // infernal_result: TensorId,
    // infernal_backend: &infernal::backend::Metal,
    // ggml_result: ggml::Tensor,
    // ggml_backend: ggml::Metal,
) {
    assert_eq!(ggml.output.typ(), ggml::TensorType::F32);
    let mut ggml_elems = vec![0.0_f32; ggml.output.total_elems()];
    ggml.output.copy_data_to(0, &mut ggml_elems);

    let infernal_elems = infernal.ctx[infernal.output].elements::<f32>(infernal.backend);

    println!(
        "[{test_case}] size (bytes): infernal={} ggml={}",
        infernal.ctx[infernal.output].size(),
        ggml.output.nbytes(),
    );
    println!(
        "[{test_case}] elements: infernal={} ggml={}",
        infernal.ctx[infernal.output].total_elems(),
        ggml.output.total_elems(),
    );
    println!(
        "[{test_case}] layout: infernal={} ggml={}",
        infernal.ctx[infernal.output].layout(),
        ggml_to_layout(&ggml.output)
    );
    assert_eq!(
        infernal.ctx[infernal.output].layout(),
        &ggml_to_layout(&ggml.output)
    );
    println!(
        "[{test_case}] raw dims: infernal={:?} ggml={:?}",
        infernal.ctx[infernal.output].layout().dims_array(),
        ggml.output.dims().as_array(),
    );
    println!(
        "[{test_case}] raw stride: infernal={:?} ggml={:?}",
        infernal.ctx[infernal.output].layout().stride_array(),
        ggml.output.stride(),
    );

    let diff = infernal.time.as_secs_f64() - ggml.time.as_secs_f64();
    let mult = infernal.time.as_secs_f64() / ggml.time.as_secs_f64();
    println!(
        "[{test_case}] infernal={infernal_time:?} <> ggml={ggml_time:?} (diff={diff:.2?}ms => {mult:.2}X)",
        infernal_time = infernal.time,
        ggml_time = ggml.time
    );

    let diff = (infernal.time.as_secs_f64() - ggml.time.as_secs_f64()) * 1000.;
    let mult = infernal.time.as_secs_f64() / ggml.time.as_secs_f64();
    println!(
        "[{test_case}] infernal={infernal_avg_time:?}(S={infernal_time:?}) <> ggml={ggml_avg_time:?}(S={ggml_time:?}) (diff={diff:.2?}ms => {mult:.2}X)",
        infernal_avg_time = infernal.avg_time,
        infernal_time = infernal.time,
        ggml_avg_time = ggml.avg_time,
        ggml_time = ggml.time
    );

    compare_elements(test_case, infernal_elems, &ggml_elems);
}

fn compare_elements(test_case: &'static str, infernal_elems: &[f32], ggml_elems: &[f32]) {
    assert_eq!(infernal_elems.len(), ggml_elems.len());
    let n = ggml_elems.len();
    const M: usize = 50;
    for i in 0..n {
        if i < M {
            if i == 0 {
                println!("[{test_case}] first {M} elements:");
            }
            println!(
                "\t[{test_case}][{i}]: {} <> {}",
                infernal_elems[i], ggml_elems[i]
            );
        }
        if i + M == n {
            println!("[{test_case}] last {M} elements:");
        }
        if i + M >= n {
            println!(
                "\t[{test_case}][{i}]: {} <> {}",
                infernal_elems[i], ggml_elems[i]
            );
        }

        if (infernal_elems[i].is_nan() && ggml_elems[i].is_nan())
            || (infernal_elems[i].is_infinite() && ggml_elems[i].is_infinite())
        {
            continue;
        }
        if (infernal_elems[i] - ggml_elems[i]).abs() >= f32::EPSILON {
            // If there is a mismatch, print next N elements to help debugging.
            for j in i..(i + M).min(n) {
                println!(
                    "\t[{test_case}][{j}]: {} <> {}",
                    infernal_elems[j], ggml_elems[j]
                );
            }
            assert!(
                (infernal_elems[i] - ggml_elems[i]).abs() < f32::EPSILON,
                "[{test_case}] element {i}/{total} mismatch: infernal={got}, ggml={exp}",
                got = infernal_elems[i],
                exp = ggml_elems[i],
                total = ggml_elems.len(),
            );
        }
    }
}

const N_RUNS: u32 = 10;

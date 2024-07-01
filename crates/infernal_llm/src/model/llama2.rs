use std::ops::DerefMut;

use infernal::{
    alloc::{Oom, Pos, SplitPoint},
    builder::TensorOptionsExt,
    op::{Rope, RopeCustom},
    tensor::MemoryLocation,
    ComputeGraph, GraphBuilder, Layout, TensorId, TensorSrc, Type, D1, D2, D3,
};

use crate::{
    loader::{LoadError, TensorLoader},
    model::{Arch, Common, GraphArgs, GraphError, GraphInput, GraphOp, Input, LoadArgs, Model},
    tokenizer::spm::{Spm, SPACE_RPLCMNT},
    Hyperparams, ParamSize, TensorCategory, Vocab,
};

#[derive(Debug)]
pub struct Llama2 {
    common: Common,

    tok_embd: TensorId,
    output_norm: TensorId,
    output: TensorId,
    layers: Vec<Layer>,

    // TODO: consider making it an input to GraphArgs
    pub(crate) n_gpu_layers: Option<usize>,
}

#[derive(Debug)]
struct Layer {
    // -- pre-normalization --
    attn_norm: TensorId,

    // -- self-attention --
    wq: TensorId,
    wk: TensorId,
    wv: TensorId,
    wo: TensorId,

    // -- feed-forward --
    ffn_norm: TensorId,
    w1: TensorId, // ffn_gate
    w2: TensorId, // ffn_down
    w3: TensorId, // ffn_up
}

impl Model for Llama2 {
    const ARCH: Arch = Arch::Llama;

    type Tokenizer = Spm;

    fn new_tokenizer(vocab: Vocab) -> Self::Tokenizer {
        Spm::new(vocab)
    }

    #[inline(always)]
    fn common(&self) -> &Common {
        &self.common
    }

    fn optimal_split_points(&self) -> Vec<SplitPoint<TensorId>> {
        // We want to swap in and out the first and last tensors, to minimize paging churn. For
        // this, we need to have a pair of weights of roughly the same size, so that they can be
        // swapped out while preserving the memory limit. Because the two weights are not the same
        // size, the memory limit needs to account for the difference.
        //
        // The page sizes are the following:
        //
        // {id=buf[0#0]}:GPU buffer created size=56,328,192    => 56 MB
        // {id=buf[0#1]}:GPU buffer created size=3,354,361,856 => 3.35 GB
        // {id=buf[0#2]}:GPU buffer created size=107,577,344   => 107 MB
        //
        // In this case we want to swap buffers 0 and 1 into VRAM. Therefore the memory limit should
        // be: 3,355 MB + 107 MB - 53 MB = 3405 MB (3,405,611,008)
        vec![
            // wq is the first weight in the layer while w2/ffn_down is the last weight in the
            // layer. This maps tok_embd.
            SplitPoint {
                at: Pos::Before(self.layers[0].wq),
                pin: true,
            },
            // Map everything after tok_embd before output_norm (i.e. the layers: attention heads
            // and the ffn network). Split this in two, though, due to Metal max buffer size on iOS
            // being 1.5-2GB.
            // SplitPoint { at: Pos::Before(self.layers[2].wq), pin: true, },
            // SplitPoint { at: Pos::Before(self.layers[4].wq), pin: true, },
            // SplitPoint { at: Pos::Before(self.layers[6].wq), pin: true, },
            SplitPoint {
                at: Pos::Before(self.layers[8].wq),
                pin: true,
            },
            // SplitPoint { at: Pos::Before(self.layers[10].wq), pin: true, },
            // SplitPoint { at: Pos::Before(self.layers[12].wq), pin: true, },
            // SplitPoint { at: Pos::Before(self.layers[14].wq), pin: true, },
            SplitPoint {
                at: Pos::Before(self.layers[16].wq),
                pin: true,
            },
            // SplitPoint { at: Pos::Before(self.layers[18].wq), pin: true, },
            // SplitPoint { at: Pos::Before(self.layers[20].wq), pin: true, },
            // SplitPoint { at: Pos::Before(self.layers[22].wq), pin: true, },
            SplitPoint {
                at: Pos::Before(self.layers[24].wq),
                pin: true,
            },
            // SplitPoint { at: Pos::Before(self.layers[26].wq), pin: true, },
            // SplitPoint { at: Pos::Before(self.layers[28].wq), pin: true, },
            // SplitPoint { at: Pos::Before(self.layers[30].wq), pin: true, },
            SplitPoint {
                at: Pos::Before(self.output_norm),
                pin: true,
            },
            // Map everything until after the output weight (i.e. till the end).
            SplitPoint {
                at: Pos::After(self.output),
                pin: true,
            },
        ]
    }

    fn load(
        tl: &mut TensorLoader,
        LoadArgs {
            params,
            hparams,
            ftype,
            name,
        }: LoadArgs<'_>,
    ) -> Result<Self, LoadError> {
        use TensorCategory::*;

        // TODO#perf: why is this always on the CPU? can we just move everything to GPU? experiment
        // with various configs
        let tok_embd = tl.load_global(
            TokenEmbd,
            MemoryLocation::Ram,
            [hparams.embd_dim, hparams.n_vocab],
        )?;

        let n_layer = hparams.n_layers;
        let Hyperparams {
            embd_dim,
            ff_hidden_dim,
            n_vocab,
            ..
        } = hparams;
        let kv_embd_dim = hparams.gqa_embd_dim();

        let (norm_loc, output_loc) = if params.n_gpu_layers.unwrap_or(usize::MAX) > n_layer {
            (MemoryLocation::Vram, MemoryLocation::Vram)
        } else {
            (MemoryLocation::Ram, MemoryLocation::Ram)
        };
        let output_norm = tl.load_global(OutputNorm, norm_loc, [embd_dim])?;
        let output = tl.load_global(Output, output_loc, [embd_dim, n_vocab])?;

        let mut layers = Vec::with_capacity(n_layer);
        let gpu_start_idx = n_layer - params.n_gpu_layers.unwrap_or(n_layer);

        for i in 0..n_layer {
            let loc = if i < gpu_start_idx {
                MemoryLocation::Ram
            } else {
                MemoryLocation::Vram
            };
            let loc_split = if i < gpu_start_idx {
                MemoryLocation::Ram
            } else {
                // TODO(perf): why? experiment with only using gpu here
                // MemoryLocation::Split
                MemoryLocation::Vram
            };

            let layer = Some(i);
            layers.push(Layer {
                // NB: All dimensions here are in the reverse ggml order.
                // TODO: change this.
                attn_norm: tl.load(AttnNorm, layer, loc, [embd_dim])?,
                wq: tl.load(AttnQ, layer, loc_split, [embd_dim, embd_dim])?,
                wk: tl.load(AttnK, layer, loc_split, [embd_dim, kv_embd_dim])?,
                wv: tl.load(AttnV, layer, loc_split, [embd_dim, kv_embd_dim])?,
                wo: tl.load(AttnOut, layer, loc_split, [embd_dim, embd_dim])?,
                ffn_norm: tl.load(FfnNorm, layer, loc, [embd_dim])?,
                w1: tl.load(FfnGate, layer, loc_split, [embd_dim, ff_hidden_dim])?,
                w2: tl.load(FfnDown, layer, loc_split, [ff_hidden_dim, embd_dim])?,
                w3: tl.load(FfnUp, layer, loc_split, [embd_dim, ff_hidden_dim])?,
            })
        }

        // Can this be read in from the gguf file?
        let param_size = match hparams.n_layers {
            26 => Some(ParamSize::_3B),
            32 => Some(ParamSize::_7B),
            40 => Some(ParamSize::_13B),
            60 => Some(ParamSize::_30B),
            80 => Some(if hparams.n_heads == hparams.n_kv_heads {
                ParamSize::_65B
            } else {
                ParamSize::_70B
            }),
            _ => None,
        };

        Ok(Self {
            common: Common {
                // TODO: maybe move param_size into model as some models will only have one type
                param_size,
                ftype,
                name,
                hparams,
            },

            tok_embd,
            output_norm,
            output,
            layers,

            n_gpu_layers: params.n_gpu_layers,
            // tensors_by_name,
        })
    }

    fn build_graph<B: DerefMut<Target = [u8]>, S: TensorSrc>(
        &self,
        GraphArgs {
            kv_cache,
            compute_tb: tb,
            past_len,
        }: GraphArgs<'_, '_, B, S>,
        input: GraphInput<'_>,
    ) -> Result<(), GraphError> {
        use Type::*;

        let Hyperparams {
            embd_dim, n_layers, ..
        } = self.common.hparams;
        let q_embd_dim = self.common.hparams.head_embd_dim();

        let mut graph = None;
        let seq_len;
        let is_measure;
        let mut input_embd;
        match input {
            Input::Tokens(tokens) => {
                // TODO: check for !tokens.is_empty() ?
                seq_len = tokens.len();
                let inp_tokens = tb.new_tensor(
                    D1::from(seq_len)
                        .with_type(I32)
                        .with_name("inp_tokens")
                        .prealloc(),
                )?;
                match tokens {
                    GraphOp::PreEval(tokens) => {
                        tb.copy_data_from(inp_tokens, tokens);
                        is_measure = false;
                    }
                    GraphOp::Measure { graph: gf, .. } => {
                        graph = Some(gf);
                        is_measure = true;
                    }
                }
                input_embd = tb.get_rows(self.tok_embd, inp_tokens)?;
            }
            Input::Embd(embd) => {
                // TODO: check for !embd.is_empty() ?
                seq_len = embd.len();
                input_embd = tb.new_tensor(
                    D2 {
                        rows: seq_len,
                        cols: embd_dim,
                    }
                    .with_type(F32)
                    .with_name("inp_l")
                    .prealloc(),
                )?;
                match embd {
                    GraphOp::PreEval(embd) => {
                        tb.copy_data_from(input_embd, &embd[..(embd_dim * seq_len)]);
                        is_measure = false;
                    }
                    GraphOp::Measure { graph: gf, .. } => {
                        graph = Some(gf);
                        is_measure = true;
                    }
                }
            }
        };

        let scores_scale = tb.new_tensor(
            D1::from(1)
                .with_type(F32)
                .with_name("1/sqrt(n_embd_head)")
                .prealloc(),
        )?;
        if !is_measure {
            tb.fill::<f32>(scores_scale, 1. / (q_embd_dim as f32).sqrt());
        }

        let gpu_start_idx = n_layers - self.n_gpu_layers.unwrap_or(n_layers);
        let offload_norm = self.n_gpu_layers.unwrap_or_default() > n_layers;
        let offload_v = self.n_gpu_layers.unwrap_or_default() > n_layers + 1;
        let offload_kq = self.n_gpu_layers.unwrap_or_default() > n_layers + 2;

        for (i, layer) in self.layers.iter().enumerate() {
            let offload = i >= gpu_start_idx;
            // NB: inp_l gets updated to a new tensor after each round, so we're not just
            // overwriting the same tensor's name here.
            tb[input_embd].set_name(format!("layer_inp_{i}"));

            let orig_input_embd = input_embd;

            // self-attention
            let x = self.norm(
                tb,
                input_embd,
                layer.attn_norm,
                offload,
                "rms_norm_0",
                "attention_norm_0",
            )?;
            let x = self.attention(
                GraphArgs {
                    kv_cache,
                    compute_tb: tb,
                    past_len,
                },
                &mut graph,
                (i, layer),
                scores_scale,
                x,
                seq_len,
                Offload {
                    default: offload,
                    v: offload_v,
                    kq: offload_kq,
                },
            )?;

            let ffn_input = tb.add(x, orig_input_embd)?;
            tb[ffn_input].set_name("inpFF").offload(offload);

            // feed-forward network
            let x = self.norm(
                tb,
                ffn_input,
                layer.ffn_norm,
                offload,
                "rms_norm_1",
                "ffn_norm",
            )?;
            let x = self.ffn(tb, layer, x, offload)?;

            let x = tb.add(x, ffn_input)?;
            tb[x].set_name("inpFF_+_result_w2").offload(offload);

            // input for next layer
            input_embd = x;
        }

        // norm
        let x = self.norm(
            tb,
            input_embd,
            self.output_norm,
            offload_norm,
            "rms_norm_2",
            "result_norm",
        )?;

        let output = tb.matmul(self.output, x)?;
        tb[output].set_name("result_output").offload(true);

        if let Some(graph) = graph {
            graph.expand_forward(output, tb)?;
        }

        Ok(())
    }
}

impl Llama2 {
    #[allow(clippy::too_many_arguments)]
    fn attention<B: DerefMut<Target = [u8]>, S: TensorSrc>(
        &self,
        GraphArgs {
            kv_cache,
            compute_tb: tb,
            past_len,
        }: GraphArgs<'_, '_, B, S>,
        graph: &mut Option<&mut ComputeGraph>,
        (layer_idx, layer): (usize, &Layer),
        scores_scale: TensorId,
        x: TensorId,
        seq_len: usize,
        offload: Offload,
    ) -> Result<TensorId, GraphError> {
        use Type::*;

        let Hyperparams {
            embd_dim,
            max_ctx_len,
            n_heads: n_q_heads,
            n_kv_heads,
            ..
        } = self.common.hparams;
        let q_embd_dim = self.common.hparams.head_embd_dim();
        let kv_embd_dim = self.common.hparams.gqa_embd_dim();
        let kv_elem_size = kv_cache.typ.size();

        // Apply rotary embeddings to K and Q.

        // Dims: [kv_embd_dim, embd_dim] x T[N, embd_dim] = T[kv_embd_dim, N] = [N, kv_embd_dim]
        let xk = tb.matmul(layer.wk, x)?;
        tb[xk].set_name("tmpk").offload(offload.kq);
        // Dims: [N, n_kv_heads, q_embd_dim]
        let xk = self.apply_rotary_embd(tb, n_kv_heads, xk, seq_len, past_len);
        tb[xk].set_name("Kcur").offload(offload.kq);

        // Dims: [embd_dim, embd_dim] x T[N, embd_dim] = T[embd_dim, N] = [N, embd_dim]
        let xq = tb.matmul(layer.wq, x)?;
        tb[xq].set_name("tmpq").offload(offload.kq);
        // Dims: [N, n_heads, q_embd_dim]
        let xq = self.apply_rotary_embd(tb, n_q_heads, xq, seq_len, past_len);
        tb[xq].set_name("Qcur").offload(offload.kq);

        self.update_kv_cache(
            GraphArgs {
                kv_cache,
                compute_tb: tb,
                past_len,
            },
            graph,
            (layer_idx, layer),
            x,
            xk,
            seq_len,
            offload,
        )?;

        // Transpose along the second and third dimensions.
        //
        // Dims: [1, n_kv_heads, N, q_embd_dim]
        let xq = tb.permute(xq, [0, 2, 1, 3]);
        tb[xq].set_name("Q").offload(offload.kq);

        // Split cached K into n_kv_head heads.
        //
        // Dims: [n_kv_heads, past_len + N, q_embd_dim]
        //
        // NB: The reference implementation first repeats a kv head n_kv_heads times and then does
        // two transposes to arrive at the same dimensions. This is done in a single operation here.
        //
        // TODO(doc): NB: The ordering of rows and cols is switched up compared to the reference
        // implementation, which is likely due to the weird matmul requirements in ggml.
        // TODO(doc): why is past_len + seq_len the row not col?
        let k = tb.view_3d(
            kv_cache.k,
            D3 {
                depth: n_kv_heads, // TODO: ref impl uses n_heads here, why the change?
                rows: past_len + seq_len,
                cols: q_embd_dim,
            },
            D3 {
                depth: kv_elem_size * q_embd_dim,
                rows: kv_elem_size * kv_embd_dim,
                ..Default::default()
            },
            kv_elem_size * kv_embd_dim * max_ctx_len * layer_idx,
        );
        tb[k].set_name("K").offload(offload.kq);

        // Calculate the attention scores: K * Q
        //
        // Dims: [n_kv_heads, past_len + N, q_embd_dim]
        //     x [n_heads, N, q_embd_dim]
        //     = [n_heads, N, past_len + N]
        //
        // Example dims: k=[8x9x128] x xq=[32x7x128] = scores=[32x7x9]
        // (n=7, past_len=2, n_kv_heads=8, n_heads=32, q_embd_dim=128, kv_embd_dim=1024)
        let scores = tb.matmul(k, xq)?;
        tb[scores].set_name("KQ").offload(offload.kq);

        // KQ_scaled = KQ / sqrt(n_embd_head)
        // Dims: [past_len + N, N, n_head, 1]
        let scores = tb.scale_inplace(scores, scores_scale);
        tb[scores].set_name("KQ_scaled").offload(offload.kq);

        // KQ_masked = mask_past(KQ_scaled)
        // TODO(doc): Mask future tokens
        let scores = tb.diag_mask_inf_inplace(scores, past_len);
        tb[scores].set_name("KQ_masked").offload(offload.kq);

        // KQ = soft_max(KQ_masked)
        let scores = tb.softmax_inplace(scores);
        // TODO: unclear why this is offload.v and not offload.kq but this is how
        // it is in the llama.cpp code. check back after updating
        // upstream if that is still the case.
        tb[scores].set_name("KQ_soft_max").offload(offload.v);

        // Split cached V into n_kv_head heads.
        //
        // Dims: [n_kv_heads, q_embd_dim, past_len + N]
        let v = tb.view_3d(
            kv_cache.v,
            D3 {
                depth: n_kv_heads,
                rows: q_embd_dim,
                cols: past_len + seq_len,
            },
            D3 {
                depth: kv_elem_size * max_ctx_len * q_embd_dim,
                rows: kv_elem_size * max_ctx_len,
                ..Default::default()
            },
            kv_elem_size * max_ctx_len * kv_embd_dim * layer_idx,
        );
        tb[v].set_name("V").offload(offload.v);

        // Calculate the final output.
        //
        // Dims: [n_kv_heads, q_embd_dim, past_len + N]
        //     x [n_heads, N, past_len + N]
        //     = [n_heads, N, q_embd_dim]
        //
        // Example dims: v=[8x128x9] x scores=[32x7x9] = output=[32x7x128]
        // (n=7, past_len=2, n_kv_heads=8, n_heads=32, q_embd_dim=128, kv_embd_dim=1024)
        //
        // TODO(perf): make V contiguous in memory to speed up the matmul, however we waste
        // time on the copy on M1 this is faster for the perplexity computation,
        // but ~5% slower for the single-token generation is there a better way?
        // is there a way to make V contiguous memory from the start?
        let output = tb.matmul(v, scores)?;
        tb[output].set_name("KQV").offload(offload.v);

        // Transpose along the second and third dimensions.
        //
        // Dims: [1, N, n_heads, q_embd_dim]
        let output = tb.permute(output, [0, 2, 1, 3]);
        tb[output].set_name("KQV_merged").offload(offload.v);

        // Make kqv_merged contiguous in memory by copying to a new tensor.
        let output_copy = tb.new_tensor(Layout::new(
            D2 {
                rows: seq_len,
                cols: embd_dim,
            },
            F32,
        ))?;
        let output = tb.copy(output, output_copy);
        tb[output]
            .set_name("KQV_merged_contiguous")
            .offload(offload.v);

        // Project into output dimension (no bias)
        let output = tb.matmul(layer.wo, output)?;
        tb[output].set_name("result_wo").offload(offload.default);

        Ok(output)
    }

    fn apply_rotary_embd<B: DerefMut<Target = [u8]>, S: TensorSrc>(
        &self,
        tb: &mut GraphBuilder<B, S>,
        n_heads: usize,
        x: TensorId,
        seq_len: usize,
        past_len: usize,
    ) -> TensorId {
        let Hyperparams {
            rope_freq_base,
            rope_freq_scale,
            ..
        } = self.common.hparams;
        let q_embd_dim = self.common.hparams.head_embd_dim();

        let x = tb.reshape(
            x,
            D3 {
                depth: seq_len,
                rows: n_heads,
                cols: q_embd_dim,
            },
        );

        tb.rope_inplace(
            x,
            Rope {
                n_past: past_len,
                n_dims: q_embd_dim,
                mode: 0,
                n_ctx: 0,
                custom: Some(RopeCustom {
                    freq_base: rope_freq_base,
                    freq_scale: rope_freq_scale,
                }),
            },
        )
    }

    /// Stores k and v in cache.
    ///
    /// This visualization helps understanding the KV cache:
    /// https://miro.medium.com/v2/resize:fit:1400/format:webp/0*WAJVBEY5vs6QHh2W.gif
    #[allow(clippy::too_many_arguments)]
    fn update_kv_cache<B: DerefMut<Target = [u8]>, S: TensorSrc>(
        &self,
        GraphArgs {
            kv_cache,
            compute_tb: tb,
            past_len,
        }: GraphArgs<'_, '_, B, S>,
        graph: &mut Option<&mut ComputeGraph>,
        (layer_idx, layer): (usize, &Layer),
        x: TensorId,
        xk: TensorId,
        seq_len: usize,
        offload: Offload,
    ) -> Result<(), GraphError> {
        let Hyperparams { max_ctx_len, .. } = self.common.hparams;
        let kv_embd_dim = self.common.hparams.gqa_embd_dim();
        let kv_elem_size = kv_cache.typ.size();

        // Compute the transposed [N, n_embd] V matrix.
        //
        // Dims: [kv_embd_dim, embd_dim] x T[N, embd_dim] = T[kv_embd_dim, N] = [N, kv_embd_dim]
        let xv = tb.matmul(layer.wv, x)?;
        tb[xv].set_name("tmpv").offload(offload.v);

        let xv_reshaped = tb.reshape(
            xv,
            D2 {
                rows: seq_len,
                cols: kv_embd_dim,
            },
        );
        // Dims: [kv_embd_dim, N]
        let xv = tb.transpose(xv_reshaped);
        tb[xv].set_name("Vcur").offload(offload.v);

        // Find the correct position to write in the K cache.
        let k = tb.view_1d(
            kv_cache.k,
            // We take the current sequence length number of embeddings from the KV cache.
            // NB: The embedding dimension is the dimension of the grouped KV head, not the
            // model embedding!
            D1::from(seq_len * kv_embd_dim),
            // Index into the K cache: index is on right hand side, element size in bytes is on the
            // left hand side.
            (kv_elem_size * kv_embd_dim) * (layer_idx * max_ctx_len + past_len),
        );
        tb[k].set_name("k").offload(offload.kq);

        // Find the correct position to write in the V cache.
        let v = tb.view_2d(
            kv_cache.v,
            D2 {
                rows: kv_embd_dim,
                cols: seq_len,
            },
            D2 {
                rows: max_ctx_len * kv_elem_size,
                ..Default::default()
            },
            ((layer_idx * max_ctx_len) * kv_elem_size * kv_embd_dim) + (past_len * kv_elem_size),
        );
        tb[v].set_name("v").offload(offload.v);

        // Copy V and RoPE-d K into KV cache.
        //
        // xk dims: [N, n_kv_heads, q_embd_dim] and k dims: [N * kv_embd_dim], but (TODO) this
        // doesn't check out: former will always be 2x the total elems!
        let k_copy = tb.copy(xk, k);
        // xv dims: [kv_embd_dim, N] and v dims: [kv_embd_dim, N]
        let v_copy = tb.copy(xv, v);
        // NB: We create the copy tensors even if not expanding the graph because the
        // compute graph must always consist of the same number and type of tensors.
        if let Some(graph) = graph {
            // NB: We do a graph expand here because the copy is a "dead-end" in the graph, so it
            // needs to be explicitly added. The copy result is only added to graph once, at
            // startup, when the compute graph is first expanded. Subsequent runs reuse the same
            // graph.
            graph.expand_forward(k_copy, tb)?;
            graph.expand_forward(v_copy, tb)?;
        }

        Ok(())
    }

    // self.w2(F.silu(self.w1(x)) * self.w3(x))
    fn ffn<B: DerefMut<Target = [u8]>, S: TensorSrc>(
        &self,
        tb: &mut GraphBuilder<B, S>,
        layer: &Layer,
        x: TensorId,
        offl: bool,
    ) -> Result<TensorId, Oom> {
        let w3x = tb.matmul(layer.w3, x)?;
        tb[w3x].set_name("result_w3").offload(offl);

        let w1x = tb.matmul(layer.w1, x)?;
        tb[w1x].set_name("result_w1").offload(offl);

        // SILU activation
        let silu = tb.silu(w1x)?;
        tb[silu].set_name("silu").offload(offl);

        // TODO: in the ref impl it's w1x that get's silu'd - what gives?
        let silu_x_w3x = tb.mul(silu, w3x)?;
        tb[silu_x_w3x].set_name("silu_x_result_w3").offload(offl);

        let result_w2 = tb.matmul(layer.w2, silu_x_w3x)?;
        tb[result_w2].set_name("result_w2").offload(offl);

        Ok(result_w2)
    }

    fn norm<B: DerefMut<Target = [u8]>, S: TensorSrc>(
        &self,
        tb: &mut GraphBuilder<B, S>,
        x: TensorId,
        norm: TensorId,
        offl: bool,
        rms_norm_name: &'static str,
        norm_name: &'static str,
    ) -> Result<TensorId, Oom> {
        let x = tb.rms_norm(x, self.common.hparams.f_norm_eps)?;
        tb[x].set_name(rms_norm_name).offload(offl);

        let x = tb.mul(x, norm)?;
        tb[x].set_name(norm_name).offload(offl);
        Ok(x)
    }
}

#[derive(Clone, Copy)]
struct Offload {
    default: bool,
    v: bool,
    kq: bool,
}

// TODO: move this into SPM? or is this Llama specific?
pub fn unescape_whitespace(input: &[u8]) -> Vec<u8> {
    let input_str = std::str::from_utf8(input).expect("invalid token text");

    let search = std::str::from_utf8(SPACE_RPLCMNT).unwrap();
    let replace = " ";

    input_str.replace(search, replace).into_bytes()
}

#[test]
fn unescapes_whitespace() {
    assert_eq!(
        unescape_whitespace(b"some\xe2\x96\x81text\xe2\x96\x81with\xe2\x96\x81\nspaces"),
        b"some text with \nspaces"
    );
}

#[cfg(test)]
mod tests {
    use std::{ffi::CString, os::unix::prelude::OsStrExt};

    use crate::{
        test_utils::{get_llama_runner, init_tracing},
        ModelRunner,
    };

    // #[ignore = "Using newer ggml kernel versions which causes slight difference in output."]
    #[test]
    fn matches_llamacpp() {
        init_tracing();

        // TODO: move out into oncecell or similar
        // let tracing_subscriber = tracing_subscriber::FmtSubscriber::builder()
        //     .with_target(false)
        //     .with_max_level(tracing::Level::TRACE)
        //     .with_span_events(tracing_subscriber::fmt::format::FmtSpan::ENTER)
        //     .with_writer(std::io::stderr) // TODO
        //     .finish();
        // tracing::subscriber::set_global_default(tracing_subscriber)
        //     .expect("setting default logger failed");

        let (mut backend, mut runner, path) = get_llama_runner();

        // Tokenize input and evaluat4e tokens.
        //         const PROMPT: &str = "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.

        // why is the sky blue?
        // ";
        const PROMPT: &str = "why is the sky blue?";

        // Tokenize manually as well (done later in eval) so we can compare with llama.cpp tokenizer.
        let mut infernal_tokens = Vec::new();
        runner.tokenize(PROMPT, false, &mut infernal_tokens);

        runner
            .eval(infernal_tokens.as_slice(), 0, &mut backend)
            .unwrap();
        let infernal_logits = runner.logits();

        // Test the detokenizer.
        let tok11_string = runner.token_to_string(11).unwrap();

        unsafe {
            let path_cstr = CString::new(path.as_os_str().as_bytes()).unwrap();
            let params = llama_cpp_sys::llama_context_default_params();
            let ggml_model = llama_cpp_sys::llama_load_model_from_file(path_cstr.as_ptr(), params);
            let ggml_ctx = llama_cpp_sys::llama_new_context_with_model(ggml_model, params);

            let mut ggml_tok11_string = vec![0; 100];
            let len = llama_cpp_sys::llama_token_to_piece(
                ggml_ctx,
                11,
                ggml_tok11_string.as_mut_ptr(),
                ggml_tok11_string.len() as i32,
            );
            let ggml_tok11_string = &ggml_tok11_string[..len as usize];

            let mut ggml_tokens = vec![0; 100];
            let prompt_cstr = CString::new(PROMPT).unwrap();
            let n_tokens = llama_cpp_sys::llama_tokenize_with_model(
                ggml_model,
                prompt_cstr.as_ptr(),
                ggml_tokens.as_mut_ptr(),
                ggml_tokens.len() as i32,
                true,
            );
            let ggml_tokens = &ggml_tokens[..n_tokens as usize];

            println!(
                "infernal tok11 string (n={}): {:?}",
                tok11_string.as_bytes().len(),
                tok11_string.as_bytes()
            );
            println!("llama.cpp tok11 string: {ggml_tok11_string:?}");
            assert_eq!(
                tok11_string.as_bytes(),
                std::mem::transmute::<_, &[u8]>(ggml_tok11_string)
            );

            println!(
                "infernal tokens (n={}):  {infernal_tokens:?}",
                infernal_tokens.len()
            );
            println!("llama.cpp tokens (n={n_tokens}): {ggml_tokens:?}",);
            assert_eq!(infernal_tokens.len(), n_tokens as usize);
            assert_eq!(infernal_tokens, ggml_tokens);

            llama_cpp_sys::llama_eval(ggml_ctx, ggml_tokens.as_ptr(), n_tokens, 0, 8);
            println!("llama.cpp eval done");
            println!(
                "infernal logits: len={} first(32)={:?}",
                infernal_logits.len(),
                &infernal_logits[..32]
            );

            // For single token evaluation, the number of logits matches the vocab size.
            let n_logits = llama_cpp_sys::llama_n_vocab(ggml_ctx) as usize;
            assert_eq!(n_logits, infernal_logits.len());

            let ggml_logits =
                std::slice::from_raw_parts(llama_cpp_sys::llama_get_logits(ggml_ctx), n_logits);
            for i in 0..n_logits {
                let infernal_logit = infernal_logits[i];
                let ggml_logit = ggml_logits[i];
                if infernal_logit != ggml_logit {
                    // Print next N logits for debugging.
                    const N: usize = 2 * 16;
                    let n = (i + N).min(n_logits);
                    for j in i..n {
                        println!(
                            "[{j}]: ours={}, llama.cpp={}",
                            infernal_logits[j], ggml_logits[j]
                        );
                    }
                }
                assert_eq!(
                    infernal_logit, ggml_logit,
                    "logit {i} mismatch: ours={infernal_logit}, llama.cpp={ggml_logit}"
                );
            }
        }

        println!("SUCCESS: All logits match!");
    }
}

# Infernal

[![License](https://img.shields.io/badge/license-Public%20Domain-blue.svg)](https://unlicense.org)
[![Compatibility](https://img.shields.io/badge/rust-1.70%2B-blue)](https://crates.io/crates/infernal)
[![Package](https://img.shields.io/crates/v/infernal)](https://crates.io/crates/infernal)

🚧 _We are building in public. This is presently under heavy construction._

Infernal is a fast, minimal, and production ready ML inferencing library written fully in Rust.

It's the spiritual successor of the famous [ggml](https://github.com/ggerganov/ggml) library for Rust and borrows heavily from other libraries, making it the best of all worlds.

See [below](#why-infernal) for rationale.

## 🚀 Features

**Accelerators**:
- [x] Metal
- [ ] [wgpu.rs](https://wgpu.rs/) + VULKAN for cross platform support
- [ ] basic CPU backend as fallback
- [ ] x86 SIMD, borrowing from Justine Tunney's incredible work
- [ ] [Higher Order Computing](https://github.com/HigherOrderCO/HVM) backend (experimental)

**Weight formats**:
- [x] gguf
- [ ] HF Safetensors

**Models**:
- [x] Llama 2
- [ ] Llama 3
- [x] Mistral
- [ ] Mixtral
- [ ] TurboSparse versions of Llama and Mistral

**Other features**:
TODO check which we have
- [ ] ggml K quants: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K
- [ ] ggml Q quants: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1
- [ ] LoRA weight merging
- [ ] runtime LoRA adapter swapping
- [ ] higher level NN API

**Optimizations**:
- [x] LLM KV cache
- [ ] quantized LLM KV cache
- [x] lazy operation chaining and batched execution
- [x] parallel compute graph execution
- [x] paging system to allow larger than RAM models or multiple models
- [ ] LLM speculative decoding
- [ ] kernel fusing
- [ ] kernel compilation
- [ ] PowerInfer sparse inferencing

## 👉 Examples

Usage should feel very familiar to those who have used ggml.

### LLM inference

```rust
use infernal_llm::{
    backend, Config, InferStep, Llama2, Llm, Template,
    Utf8Constructor, DEFAULT_SYS_PROMPT,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = backend::Metal::new()?;

    let mut llm: Llm<Llama2> = Llm::new(
        Config::new("~/.config/models/mistral-7b"),
        Template::new_llama2_chat(),
        DEFAULT_SYS_PROMPT,
        &mut backend,
    )?;

    llm.feed_prompt("Why is 42 the answer?")?;

    // Some returned tokens may be partial UTF-8 glyphs and so this is needed.
    let mut utf8_ctor = Utf8Constructor::default();
    while let Some(tok) = llm.infer_next(&mut backend)? {
        if let Some(s) = utf8_ctor.get(tok) {
            println!("{s}");
        }
    }

    // ---------------
    
    // Alternate, simpler way:
    // First, clear context window, embeddings, etc. Won't recompute logits.
    llm.clear();
    llm.feed_prompt("Why is 42 the answer?")?;
    llm.infer_loop(&mut backend, |s| println!("{s}"))?;

    Ok(())
}
```

Run this example via:
```bash
cargo run --release --example basic
```


### Low-level matmul

This example is using the low-level API of Infernal. This is quite verbose and is only recommended if implementing your own inference pipelines. The philosophy behind infernal is to expose API layers with different levels of abstraction: all the way from high-level for maximum comfort, to the lowest level for maximum control.

```rust
use infernal::{
    builder::TensorOptionsExt as _, Backend as _,
    ComputeGraph, Layout, TensorBuilder, Type, D2,
};

/// Performs a simple matrix multiplication chain, showcasing Infernal's basic primitives.
fn main() -> anyhow::Result<()> {
    let mut backend = infernal::backend::Metal::new()?;

    // A tensor builder is used to create tensors and tensor operations. It's a temporary
    // structure that, when building is finished, converts tensors into a finalized read-only
    // graph.
    let buf_size = 1024 * 1024;
    let mut tb = TensorBuilder::prealloc(buf_size)?;

    let layout = Layout::new(D2 { cols: 12, rows: 12 }, Type::F32);
    let x = tb.new_tensor(layout)?;
    let y = tb.new_tensor(layout.with_name("y"))?;
    let z = tb.new_tensor(layout.with_name("z"))?;
    tb.fill(x, 1.3_f32);
    tb.fill(y, 9.5_f32);
    tb.fill(z, 8.3_f32);

    // An operation creates a new tensor, but no computation takes place until later. You can
    // chain as many of these lazy ops as you want, they will be computed later.
    let xy = tb.matmul(x, y)?;
    let output = tb.matmul(xy, z)?;

    // Finishing the build means the resulting tensor context becomes read-only from here
    // on, and the builder loads the tensor data buffer onto the accelerator backend.
    let (ctx, _) = tb.finish(&mut backend)?;

    // Proof that tensor is lazy: data is still all zero.
    assert!(ctx[output]
        .elements::<f32>(&backend)
        .iter()
        .all(|&x| x == 0.0)
    );

    // A computation graph is used to plan how to compute the final output tensor and to find
    // optimizations using what amounts to an intermediate representation.
    let mut graph = ComputeGraph::default();
    // The graph needs to be expanded from the last output node.
    graph.expand_forward(output, &ctx)?;
    // Further, an execution plan is needed so that the backend knows how to calculate the graph.
    // These two steps have to be done once only, assuming the graph doesn't change later on.
    graph.find_exec_plan(backend.mem_limit(), &ctx)?;

    // Important! This is where all tensors are computed and materialized.
    backend.compute_graph(&graph, &ctx)?;

    // All tensors are now materialized and can be read from.
    let n_rows = layout.rows() as f32;
    assert_eq!(ctx[xy].elements::<f32>(&backend)[0], 1.3 * 9.5 * n_rows);
    assert_eq!(
        ctx[output].elements::<f32>(&backend)[0],
        1.3 * 9.5 * n_rows * 8.3 * n_rows
    );

    Ok(())
}
```

Run this example via:
```bash
cargo run --release --example matmul
```

## 🛠️ Prerequisites

- [Rust](https://rust-lang.org) 1.70+

## ⬇️ Installation

### Installation via Cargo

```console
$ cargo add infernal
```

## 👨‍💻 Development

```console
$ git clone https://github.com/HaltiaAI/infernal.rs.git
```

## Why Infernal?

Infernal was originally written for our very narrowly scoped use. This worked well and after battle-testing it we have decided to open source it and start adding more features.

We originally started with llama.cpp but quickly found that its impressive development velocity meant it was not suitable for production use. We experienced frequent crashes (perhaps excusable as inherent in C/C++), poor error handling and overall API ergonomics, multiple backwards compatibility breakage and so on. The other reason is that our stack is all Rust, making integration with and contributions in C/C++ tedious and flow-breaking.

These two things meant we had to own the stack and so have rewritten ggml and llama.cpp in Rust. Speed is a feature and so performance is at least as good as ggml and we are working on surpassing it.

## Why not MLX/Candle/Mistral.rs/etc?

Mainly that at the time MLX and Mistral.rs weren't around, Candle wasn't mature enough, and save for some half-finished attempts, there was no other Rust library to speak of.

There is another, more important reason: a clash of philosophies.

Almost all libraries we reviewed were built with a very different ethos compared to Infernal: They were large, heavy, tried to do many things. We appreciated the simplicity with which the original llama.cpp and ggml were built, and set out to follow the same ethos. We use only the necessary amount of abstraction and implement only what's most important, similar to [Tinygrad](https://github.com/tinygrad/tinygrad).

*We firmly believe that Rust can be simple and beautiful.*

To be clear, we applaud all of these libraries and so shamelessly borrow from all of them, implementing their achievements in a more modern and comfortable programming language.


- - -

[![Share on Twitter](https://img.shields.io/badge/share%20on-twitter-03A9F4?logo=twitter)](https://twitter.com/share?url=https://github.com/HaltiaAI/infernal.rs&text=Infernal.rs)
[![Share on Reddit](https://img.shields.io/badge/share%20on-reddit-red?logo=reddit)](https://reddit.com/submit?url=https://github.com/HaltiaAI/infernal.rs&title=Infernal.rs)
[![Share on Hacker News](https://img.shields.io/badge/share%20on-hacker%20news-orange?logo=ycombinator)](https://news.ycombinator.com/submitlink?u=https://github.com/HaltiaAI/infernal.rs&t=Infernal.rs)
[![Share on Facebook](https://img.shields.io/badge/share%20on-facebook-1976D2?logo=facebook)](https://www.facebook.com/sharer/sharer.php?u=https://github.com/HaltiaAI/infernal.rs)

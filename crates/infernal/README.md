# InfernalML

Haltia.AI's very own ML framework written in Rust and optimized for inference only workloads.

## Goals

- Only inferencing, no automatic differentiation
- Utter simplicity: no hidden magic, no Broadcasting, no implicit actions
- Very little dependencies
- Only support operations needed at Haltia.AI

## Development

### Conventions

- Rust idioms are used wherever possible and only specific overrides are mentioned in this list.
- Naming variables: be concise, use well known abbreviations (`len`, `addr`, `buf`, etc), but
  when something would cause confusion, prefer to be explicit.
- Naming counters:
  - tl;dr: more specific > more generic
  - Dimensions are always named suffixed with `_dim` if talking about a single dimension, or `_dims` if talking
    about a list of dimensions (but there exists an explicit type for that, `Dims`, which should be
    preferred).
  - Byte sizes are always named suffixed with `_size`. Do not use this convention anywhere else so
    that it can be relied upon. E.g. `page_size` would denote the number of bytes in a page.
    - In some cases it's acceptable to deviate from this, e.g. `vram_use`, `mem_use`, etc,
      which you'll see throughout the backend code.
  - Number of elements in a collection follow Rust's `len` convention, and this goes for anything
    that resembles a collection, such as the context window length: `ctx_window_len` (or `seq_len`).
  - All other counters are named `n_things`. Note the `n_` prefix and the plural form.
  - For things that could be both, use good taste: e.g. `n_queued` or `queue_len` are both valid,
    but `queue_size` is *not*, since in it would imply byte size, which is not what we want in this
    context. Given the above heuristic to prefer the more specific option, `queue_len` is chosen in
    this case.
    - To showcase why size != len is important: In the Metal backend code, we use both `queue_len`
      and `queue_size` to denote the number of elements in a queue and the total byte size of the
      tensor buffers in the queue. This only works due to this explicit convention.
- Generally, keep things concise when dealing with low-level code (while it is recommended to be
  more explicit in application level code, above Infernal).

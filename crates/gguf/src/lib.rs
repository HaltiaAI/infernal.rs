use std::str::Utf8Error;

use thiserror::Error;

pub use crate::{
    context::{Context, Header, TensorInfo, WEIGHT_ALIGN},
    dims::Dims,
    kv::{TypeCastError, TypeConvError, Value},
    typ::TensorType,
};

mod context;
mod dims;
pub mod g_quant;
pub mod k_quant;
pub mod kv;
mod typ;

#[derive(Debug, Error)]
pub enum LoadError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    InvalidType(#[from] TypeConvError),
    #[error("invalid GGML tensor type: {0}")]
    InvalidTensorType(u32),
    #[error("tensor name must be valid utf8")]
    InvalidTensorUtf8Name(#[from] Utf8Error),
    #[error("unsupported file version: {0}")]
    UnsupportedFileVersion(u32),
    #[error("tensor '{0}' missing from model")]
    MissingTensor(String),
    #[error(
        "tensor '{name}' number of elements ({n_elem}) is not a multiple of the block size ({block_size}"
    )]
    UnalignedTensorElements {
        name: String,
        n_elem: usize,
        block_size: usize,
    },
    #[error("tensor '{name}' has wrong shape: expected={expected:?}, actual={actual:?}")]
    TensorWrongShape {
        name: String,
        expected: Vec<u32>,
        actual: Vec<u32>,
    },
    #[error("array cannot contain nested arrays")]
    NestedArray,
}

pub(crate) fn pad(x: usize, alignment: usize) -> usize {
    let a = alignment;
    (x + a - 1) & !(a - 1)
}

#[cfg(test)]
mod tests {
    use std::{fs::File, path::PathBuf};

    use crate::{context::Context, kv::Value};

    // Example orca_mini_v3_7b.Q3_K_S.gguf kv:
    //
    // key=general.architecture, val=Some(String([108, 108, 97, 109, 97]))
    // key=general.name, val=Some(String([76, 76, 97, 77, 65, 32, 118, 50]))
    // key=llama.context_length, val=Some(Uint32(4096))
    // key=llama.embedding_length, val=Some(Uint32(4096))
    // key=llama.block_count, val=Some(Uint32(32))
    // key=llama.feed_forward_length, val=Some(Uint32(11008))
    // key=llama.rope.dimension_count, val=Some(Uint32(128))
    // key=llama.attention.head_count, val=Some(Uint32(32))
    // key=llama.attention.head_count_kv, val=Some(Uint32(32))
    // key=llama.attention.layer_norm_rms_epsilon, val=Some(Float32(1e-5))
    // key=general.file_type, val=Some(Uint32(11))
    // key=tokenizer.ggml.model, val=Some(String([108, 108, 97, 109, 97]))
    // key=tokenizer.ggml.tokens, val=array#32000[Some(String([60, 117, 110, 107, 62]))...]
    // key=tokenizer.ggml.scores, val=array#32000[Some(Float32(0.0))...]
    // key=tokenizer.ggml.token_type, val=array#32000[Some(Int32(2))...]
    // key=tokenizer.ggml.bos_token_id, val=Some(Uint32(1))
    // key=tokenizer.ggml.eos_token_id, val=Some(Uint32(2))
    // key=tokenizer.ggml.unknown_token_id, val=Some(Uint32(0))
    // key=general.quantization_version, val=Some(Uint32(2))
    #[test]
    fn kv_contains_llama_metadata() {
        let (ctx, _file, ..) = get_context();
        for key in [
            "general.architecture",
            "general.name",
            "llama.context_length",
            "llama.embedding_length",
            "llama.block_count",
            "llama.feed_forward_length",
            "llama.rope.dimension_count",
            "llama.attention.head_count",
            "llama.attention.head_count_kv",
            "llama.attention.layer_norm_rms_epsilon",
            "general.file_type",
            "tokenizer.ggml.model",
            "tokenizer.ggml.tokens",
            "tokenizer.ggml.scores",
            "tokenizer.ggml.token_type",
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.unknown_token_id",
            "general.quantization_version",
        ] {
            let val = ctx.get(key.as_bytes());
            if let Some(Value::Array(a)) = val {
                println!("key={key}, val=array#{}[{:?}...]", a.len(), a.first());
            } else {
                println!("key={key}, val={val:?}");
            }
            assert!(val.is_some());
        }
        assert!(ctx.get("not-present-for-sure".as_bytes()).is_none());
    }

    #[test]
    fn conv_value_to_u32() {
        let (ctx, _file, ..) = get_context();
        let res = ctx.get("llama.attention.head_count".as_bytes()).unwrap();
        assert!(res.as_u32().is_ok());
    }

    #[test]
    fn conv_value_to_array_len() {
        let (ctx, _file, ..) = get_context();
        let res = ctx.get("tokenizer.ggml.tokens".as_bytes()).unwrap();
        assert!(res.as_array_len().is_ok());
    }

    fn get_context() -> (Context, File, PathBuf) {
        let path = dirs::home_dir()
            .expect("$HOME must be set")
            .join(".config/haltia/orca_mini_v3_7b.Q3_K_S.gguf");
        assert!(
            path.exists(),
            "~/.config/haltia/orca_mini_v3_7b.Q3_K_S.gguf is required for this test to run"
        );
        let file = File::open(&path).unwrap();
        let (ctx, file) = Context::load(file).unwrap();
        (ctx, file, path)
    }
}

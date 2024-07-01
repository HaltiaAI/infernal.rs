use std::{
    borrow::Cow,
    collections::HashMap,
    fs::File,
    path::Path,
    str::{FromStr, Utf8Error},
    string::FromUtf8Error,
};

use gguf::{FileType, KvKey, TensorType};
use infernal::{alloc::Oom, builder::TensorOptionsExt, tensor::MemoryLocation, Layout, TensorId};
use thiserror::Error;
use tracing::{debug, info, trace, warn};

use crate::{
    model::{Arch, LoadArgs, Model},
    Hyperparams, Params, PartialHyperparams, SpecialTokens, TensorCategory, TokenId, Vocab,
    VocabType,
};
pub use infernal::builder::WeightsBuilder;

/// Loads the specified model from the given path.
///
/// Currently only [`gguf`] files are supported.
pub fn load<M: Model>(
    path: impl AsRef<Path>,
    params: &Params,
) -> Result<(M, M::Tokenizer, TensorLoader), LoadError> {
    let file = File::open(path)?;
    let (gguf, file) = gguf::Context::load(file)?;

    // Print metadata: n_kv, n_tensors.
    let mut total_elems = 0;
    let mut type_counts: HashMap<TensorType, usize> = HashMap::new();
    let mut type_max_count = None;
    for t in &gguf.tensors {
        total_elems += t.dims.total_elems();

        let c = type_counts.entry(t.typ).or_default();
        *c += 1;
        if *c > type_max_count.map(|(_, c)| c).unwrap_or_default() {
            type_max_count = Some((t.typ, *c));
        }

        info!(
            name = std::str::from_utf8(&t.name).unwrap_or("unknown"),
            tensor_type = ?t.typ,
            dims = %t.dims,
            "loaded gguf tensor metadata",
        );
    }

    let n_tensors = gguf.tensors.len();
    info!(
        n_kv = gguf.kv.len(),
        n_tensors, total_elems, "loaded metadata"
    );
    // Print type counts for each tensor quantization.
    for (typ, count) in &type_counts {
        if *count != 0 {
            info!(
                tensor_type = ?typ,
                count,
                "tensor type count"
            );
        }
    }
    for kv in &gguf.kv {
        let key = std::str::from_utf8(&kv.key).unwrap_or("unknown");
        match &kv.value {
            gguf::Value::Array(a) => {
                debug!(key, len = a.len(), "kv (array)");
            }
            gguf::Value::String(s) => {
                debug!(key, len = s.len(), "kv (string)");
            }
            _ => {
                debug!(key, val = ?kv.value, "kv");
            }
        }
    }

    let extractor = GgufExtractor { ctx: gguf };
    dbg!(&extractor
        .ctx
        .kv
        .iter()
        .map(|kv| std::str::from_utf8(&kv.key).unwrap())
        .collect::<Vec<_>>());
    let arch = extractor.get_arch()?;
    if arch != M::ARCH {
        // TODO: real error
        panic!("incorrect architecture");
    }

    let ftype = if let Some(ftype) = extractor.get_file_type() {
        let ftype = ftype?;
        debug!(file_type = ?ftype, "file type in metadata");
        Some(ftype)
    } else {
        let ftype = type_max_count
            .map(|(t, _)| t)
            .and_then(FileType::guess_from_tensor);
        if let Some(ftype) = ftype {
            debug!(
                file_type = ?ftype,
                "file type not in metadata, guessed based on max tensor quantization"
            );
        } else {
            warn!("file type unknown");
        }
        ftype
    };

    let hparams = extractor.get_hparams(arch)?;
    let name: Option<String> = extractor.get_parsed(KvKey::GeneralName)?;
    let hparams = Hyperparams::from_partial(hparams, params);
    let vocab = extractor.get_vocab(arch)?;
    if hparams.n_vocab != vocab.id_to_token.len() {
        return Err(LoadError::IncompleteVocab {
            got: hparams.n_vocab,
            exp: vocab.id_to_token.len(),
        });
    }

    let gguf = extractor.ctx;
    let mut tl = TensorLoader::new(gguf, file, arch)?;

    let model = M::load(
        &mut tl,
        LoadArgs {
            params,
            hparams,
            ftype,
            name,
        },
    )?;
    let tokenizer = M::new_tokenizer(vocab);

    Ok((model, tokenizer, tl))
}

#[derive(Debug, Error)]
pub enum LoadError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Gguf(#[from] gguf::LoadError),
    #[error(transparent)]
    Extract(#[from] ExtractError),
    #[error(
        "could not read in entire vocab or hparams is misconfigured: expected={exp}, got={got}"
    )]
    IncompleteVocab { exp: usize, got: usize },
    #[error("did not create all expected tensors: expected={exp}, got={got}")]
    IncompleteTensors { exp: usize, got: usize },
    #[error("there are more tensors than space in the mapped buffer: {0}")]
    TensorAlloc(#[from] Oom),
    #[error(transparent)]
    InvalidFtype(#[from] gguf::FileTypeConvError),
    #[error("tensor name must be valid utf8")]
    InvalidTensorUtf8Name(#[from] FromUtf8Error),
    #[error("tensor should not be {dim}-dimensional")]
    InvalidTensorDims { dim: usize },
    #[error("broken invariant: {0}")]
    InvariantBroken(String),
    #[error("tensor '{name}' GGUF tensor type '{typ}' not supported")]
    UnsupportedGgufTensorType { name: String, typ: gguf::TensorType },
    #[error("tensor '{0}' missing from model")]
    MissingTensor(String),
    #[error("tensor '{name}' has wrong shape: expected={exp}, got={got}")]
    TensorWrongShape {
        name: String,
        exp: gguf::Dims,
        got: gguf::Dims,
    },
}

#[derive(Debug, Error)]
pub enum ExtractError {
    #[error(transparent)]
    InvalidFtype(#[from] gguf::FileTypeConvError),
    #[error("missing required parameter {0:?}")]
    Missing(KvKey),
    // TODO: why is this different from Missing?
    #[error("missing required array parameter {0:?}")]
    MissingArray(KvKey),
    #[error("parameter {0:?} type error: {1}")]
    InvalidType(KvKey, gguf::TypeCastError),
    #[error(transparent)]
    // TODO: add key for identifcation
    UnknownTokenType(#[from] gguf::UnknownTokenType),
    #[error("parameter {0:?} is an invalid utf8 string")]
    InvalidUtf8(KvKey, Utf8Error),
    #[error("model architecture must be set")]
    MissingArch,
    #[error("parameter {0:?} breaks an invariant: {1}")]
    BrokenInvariant(KvKey, String),
}

struct GgufExtractor {
    ctx: gguf::Context,
}

impl GgufExtractor {
    fn get_arch(&self) -> Result<Arch, ExtractError> {
        self.get_parsed(KvKey::GeneralArchitecture)?
            .ok_or(ExtractError::MissingArch)
    }

    fn get_file_type(&self) -> Option<Result<FileType, ExtractError>> {
        if let Ok(ft) = self.get(KvKey::GeneralFileType, None, gguf::Value::as_u32)? {
            Some(ft.try_into().map_err(From::from))
        } else {
            None
        }
    }

    fn get_hparams(&self, arch: Arch) -> Result<PartialHyperparams, ExtractError> {
        // Mandatory params
        let n_vocab =
            self.get_required(KvKey::TokenizerList, arch.into(), gguf::Value::as_array_len)?;
        let n_ctx_train =
            self.get_required(KvKey::ContextLength, arch.into(), gguf::Value::as_u32)?;
        let n_embd = self.get_required(KvKey::EmbeddingLength, arch.into(), gguf::Value::as_u32)?;
        let n_ff = self.get_required(KvKey::FeedForwardLength, arch.into(), gguf::Value::as_u32)?;
        let n_head =
            self.get_required(KvKey::AttentionHeadCount, arch.into(), gguf::Value::as_u32)?;
        let n_layer = self.get_required(KvKey::BlockCount, arch.into(), gguf::Value::as_u32)?;

        let mut params = Params {
            ..Params::default()
        };

        // Optional params
        let n_head_kv = if let Some(n_head_kv) = self.get(
            KvKey::AttentionHeadCountKv,
            arch.into(),
            gguf::Value::as_u32,
        ) {
            n_head_kv?
        } else {
            n_head
        };
        if let Some(ropebase) = self.get(KvKey::RopeFreqBase, arch.into(), gguf::Value::as_f32) {
            params.rope_freq_base = ropebase?;
        }
        if let Some(ropescale) = self.get(KvKey::RopeScaleLinear, arch.into(), gguf::Value::as_f32)
        {
            params.rope_freq_scale = ropescale?;
        }
        // More of a sanity check:
        let default_n_rot = n_embd / n_head;
        let n_rot = if let Some(n_rot) =
            self.get(KvKey::RopeDimensionCount, arch.into(), gguf::Value::as_u32)
        {
            let n_rot = n_rot?;
            if (arch == Arch::Llama/* || arch == Arch::Falcon*/) && n_rot != default_n_rot {
                return Err(ExtractError::BrokenInvariant(
                    KvKey::RopeDimensionCount,
                    format!("invalid n_rot: {n_rot}, expected: {default_n_rot}"),
                ));
            }
            n_rot
        } else {
            default_n_rot
        };

        // TODO: generic
        let f_norm_eps = match arch {
            Arch::Llama => self.get_required(
                KvKey::AttentionLayernormRmsEps,
                arch.into(),
                gguf::Value::as_f32,
            )?,
            // Arch::Falcon => self.get_required(
            //     KvKey::AttentionLayernormEps,
            //     arch.into(),
            //     gguf::Value::as_f32,
            // )?,
        };

        Ok(PartialHyperparams {
            n_vocab,
            max_train_ctx_len: n_ctx_train as usize,
            embd_dim: n_embd as usize,
            n_heads: n_head as usize,
            n_kv_heads: n_head_kv as usize,
            n_layers: n_layer as usize,
            n_rot: n_rot as usize,
            ff_hidden_dim: n_ff as usize,
            f_norm_eps,
        })
    }

    fn get_vocab(&self, arch: Arch) -> Result<Vocab, ExtractError> {
        let tokenizer = self.get_required(
            KvKey::TokenizerModel,
            arch.into(),
            gguf::Value::as_bytestring,
        )?;
        let tokenizer = std::str::from_utf8(tokenizer)
            .map_err(|e| ExtractError::InvalidUtf8(KvKey::TokenizerModel, e))?;

        let bos_id = self
            .get(KvKey::TokenizerBosId, arch.into(), gguf::Value::as_u32)
            .transpose()?;
        let eos_id = self
            .get(KvKey::TokenizerEosId, arch.into(), gguf::Value::as_u32)
            .transpose()?;
        let unk_id = self
            .get(KvKey::TokenizerUnkId, arch.into(), gguf::Value::as_u32)
            .transpose()?;
        let sep_id = self
            .get(KvKey::TokenizerSepId, arch.into(), gguf::Value::as_u32)
            .transpose()?;
        let pad_id = self
            .get(KvKey::TokenizerPadId, arch.into(), gguf::Value::as_u32)
            .transpose()?;
        let (special, typ) = match tokenizer {
            "llama" => (
                SpecialTokens::llama_with_overrides(
                    bos_id.map(|i| i as i32),
                    eos_id.map(|i| i as i32),
                    unk_id.map(|i| i as i32),
                    sep_id.map(|i| i as i32),
                    pad_id.map(|i| i as i32),
                ),
                VocabType::Spm,
            ),
            _ => {
                unimplemented!("tokenizer {} not yet supported", tokenizer);
            }
        };

        // TODO: can we use hardcoded values? for now anyway
        // // determine the newline token: LLaMA "<0x0A>" == 10 == '\n', Falcon 193 ==
        // // '\n'
        // if vocab.typ == VocabType::Spm {
        //     vocab.special.linefeed_id = llama_byte_to_token(vocab, '\n');
        // } else {
        //     vocab.linefeed_id = llama_tokenize_internal(vocab, "\n", false)[0];
        // }

        let tokens = self.get_required(KvKey::TokenizerList, arch.into(), gguf::Value::as_array)?;
        let scores =
            self.get_required(KvKey::TokenizerScores, arch.into(), gguf::Value::as_array)?;
        let tok_types = self.get_required(
            KvKey::TokenizerTokenType,
            arch.into(),
            gguf::Value::as_array,
        )?;
        let mut vocab = Vocab::with_capacity(typ, special, tok_types.len());
        for (id, t) in tokens.iter().enumerate() {
            let tok = t
                .as_bytestring()
                .map_err(|e| ExtractError::InvalidType(KvKey::TokenizerTokenType, e))?;
            let score = scores[id]
                .as_f32()
                .map_err(|e| ExtractError::InvalidType(KvKey::TokenizerScores, e))?;
            let typ = tok_types[id]
                .as_i32()
                .map(|i| i as u32)
                .map_err(|e| ExtractError::InvalidType(KvKey::TokenizerTokenType, e))?
                .try_into()?;
            // TODO#perf: get rid of clone here
            vocab.insert(id as TokenId, tok.to_vec(), typ, score);
        }

        Ok(vocab)
    }

    fn get_parsed<T: FromStr>(&self, key: KvKey) -> Result<Option<T>, ExtractError> {
        if let Some(b) = self.get(key, None, gguf::Value::as_bytestring) {
            let s = std::str::from_utf8(b?).map_err(|e| ExtractError::InvalidUtf8(key, e))?;
            Ok(T::from_str(s).ok())
        } else {
            Ok(None)
        }
    }

    fn get_required<'s, T: 's>(
        &'s self,
        key: KvKey,
        arch: Option<Arch>,
        deser: impl FnOnce(&'s gguf::Value) -> Result<T, gguf::TypeCastError>,
    ) -> Result<T, ExtractError> {
        if let Some(r) = self.get(key, arch, deser) {
            Ok(r?)
        } else {
            Err(ExtractError::Missing(key))
        }
    }

    fn get<'s, T: 's>(
        &'s self,
        key: KvKey,
        arch: Option<Arch>,
        deser: impl FnOnce(&'s gguf::Value) -> Result<T, gguf::TypeCastError>,
    ) -> Option<Result<T, ExtractError>> {
        self.ctx
            .get(kv_key_to_string(key, arch).as_bytes())
            .map(|v| deser(v).map_err(|e| ExtractError::InvalidType(key, e)))
    }
}

// TODO: return bytevec instead since that's what the source data contains -- avoids utf8
// conversion error handling
fn kv_key_to_string(k: gguf::KvKey, arch: Option<Arch>) -> Cow<'static, str> {
    use gguf::KvKey::*;
    let arch = || arch.expect("must pass arch for this kv key");
    match k {
        GeneralArchitecture => "general.architecture".into(),
        GeneralQuantizationVersion => "general.quantization_version".into(),
        GeneralAlignment => "general.alignment".into(),
        GeneralName => "general.name".into(),
        GeneralAuthor => "general.author".into(),
        GeneralUrl => "general.url".into(),
        GeneralDescription => "general.description".into(),
        GeneralLicense => "general.license".into(),
        GeneralSourceUrl => "general.source_url".into(),
        GeneralSourceHfRepo => "general.source_hf_repo".into(),
        GeneralFileType => "general.file_type".into(),

        ContextLength => format!("{}.context_length", arch(),).into(),
        EmbeddingLength => format!("{}.embedding_length", arch()).into(),
        BlockCount => format!("{}.block_count", arch()).into(),
        FeedForwardLength => format!("{}.feed_forward_length", arch()).into(),
        UseParallelResidual => format!("{}.use_parallel_residual", arch()).into(),
        TensorDataLayout => format!("{}.tensor_data_layout", arch()).into(),

        AttentionHeadCount => format!("{}.attention.head_count", arch()).into(),
        AttentionHeadCountKv => format!("{}.attention.head_count_kv", arch()).into(),
        AttentionMaxAlibiBias => format!("{}.attention.max_alibi_bias", arch()).into(),
        AttentionClampKqv => format!("{}.attention.clamp_kqv", arch()).into(),
        AttentionLayernormEps => format!("{}.attention.layer_norm_epsilon", arch()).into(),
        AttentionLayernormRmsEps => format!("{}.attention.layer_norm_rms_epsilon", arch()).into(),

        RopeDimensionCount => format!("{}.rope.deimension_count", arch()).into(),
        RopeFreqBase => format!("{}.rope.freq_base", arch()).into(),
        RopeScaleLinear => format!("{}.rope.scale_linear", arch()).into(),

        TokenizerModel => "tokenizer.ggml.model".into(),
        TokenizerList => "tokenizer.ggml.tokens".into(),
        TokenizerTokenType => "tokenizer.ggml.token_type".into(),
        TokenizerScores => "tokenizer.ggml.scores".into(),
        TokenizerMerges => "tokenizer.ggml.merges".into(),
        TokenizerBosId => "tokenizer.ggml.bos_token_id".into(),
        TokenizerEosId => "tokenizer.ggml.eos_token_id".into(),
        TokenizerUnkId => "tokenizer.ggml.unknown_token_id".into(),
        TokenizerSepId => "tokenizer.ggml.seperator_token_id".into(),
        TokenizerPadId => "tokenizer.ggml.padding_token_id".into(),
        TokenizerHfJson => "tokenizer.huggingface.json".into(),
        TokenizerRwkv => "tokenizer.rwkv.world".into(),
    }
}

#[derive(Debug, Error)]
pub enum CreateTensorError {
    #[error("tensor '{0}' not found in ggml context")]
    NotFound(String),
    #[error("tensor '{name}' has shape {got} but expected {expected}")]
    IncorrectShape {
        name: String,
        got: Layout,
        expected: Layout,
    },
}

/// A MMAP based tensor loader.
pub struct TensorLoader {
    gguf: gguf::Context,
    weights_tb: WeightsBuilder,
    arch: Arch,
    n_loaded: usize,
    ram_weights: usize,
    vram_weights: usize,
}

impl TensorLoader {
    pub fn new(gguf: gguf::Context, file: File, arch: Arch) -> Result<Self, LoadError> {
        // Print out some stats (not used elsewhere).
        // FIXME: this calculation is broken for some reason
        let mut ctx_size = 0;
        let mut mapped_size = 0;

        let mut tensors = gguf.tensors.clone();
        tensors.sort_by_key(|t| t.offset);
        let mut prev_end = 0;
        for t in &tensors {
            info!(
                name = std::str::from_utf8(&t.name).unwrap_or("<unknown>"),
                typ = ?t.typ,
                offset = t.offset,
                size = t.size(),
                padding = t.offset - prev_end,
                "gguf weight tensor",
            );
            ctx_size += std::mem::size_of::<infernal::Tensor>();
            mapped_size += t.size();
            prev_end = t.offset + t.size();
        }
        info!(
            mapped_size,
            mapped_size_gb = mapped_size as f64 / 1024. / 1024. / 1024.,
            ctx_size,
            ctx_size_kb = ctx_size as f64 / 1024.,
            data_offset = gguf.data_offset,
            "mapping model weights"
        );

        let weights_tb = WeightsBuilder::mmap(file, gguf.data_offset)?.with_name("weights");

        Ok(Self {
            gguf,
            weights_tb,
            arch,
            n_loaded: 0,
            ram_weights: 0,
            vram_weights: 0,
        })
    }

    /// Loads a global tensor, that is, one that is not in the repeated attention blocks or layers.
    pub fn load_global(
        &mut self,
        category: TensorCategory,
        backend: MemoryLocation,
        expected_dims: impl Into<gguf::Dims>,
    ) -> Result<TensorId, LoadError> {
        self.load(category, None, backend, expected_dims)
    }

    pub fn load(
        &mut self,
        category: TensorCategory,
        layer: Option<usize>,
        // TODO: pass this to infernal so it knows which backend to load the tensor onto, once more
        // backends are supported
        loc: MemoryLocation,
        expected_dims: impl Into<gguf::Dims>,
    ) -> Result<TensorId, LoadError> {
        let expected_dims = expected_dims.into();
        // TODO: hardcode weight? are other suffixes used?
        let name = category.to_string(self.arch, Some("weight"), layer);
        let info = self
            .gguf
            .tensors
            .iter()
            .find(|t| t.name == name.as_bytes())
            .ok_or_else(|| LoadError::MissingTensor(name.to_string()))?;

        let layout = Layout::new(
            info.dims,
            info.typ
                .try_into()
                .map_err(|_| LoadError::UnsupportedGgufTensorType {
                    name: name.to_string(),
                    typ: info.typ,
                })?,
        );
        trace!(
            %name,
            %expected_dims,
            offset = self.gguf.data_offset + info.offset,
            size = layout.size(),
            %layout,
            "creating mapped tensor"
        );
        if expected_dims != layout.raw_dims() {
            return Err(LoadError::TensorWrongShape {
                name: name.to_string(),
                exp: expected_dims,
                got: layout.raw_dims().as_ref().into(),
            });
        }
        let tensor_id = self
            .weights_tb
            .new_mapped_tensor(layout.with_name(name).with_loc(loc), info.offset)?;
        let tensor = &self.weights_tb[tensor_id];
        // info!(
        //     name = %tensor.name,
        //     size = layout.size(),
        //     %expected_dims,
        //     dims = %info.dims,
        //     offset = info.offset,
        //     "tensor mapped"
        // );

        if loc == MemoryLocation::Vram {
            self.vram_weights += tensor.size();
        } else {
            self.ram_weights += tensor.size();
        }

        self.n_loaded += 1;

        Ok(tensor_id)
    }

    pub fn finish(self) -> Result<WeightsBuilder, LoadError> {
        let n_exp_tensors = self.gguf.tensors.len();
        if self.n_loaded == n_exp_tensors {
            info!(
                n_loaded = self.n_loaded,
                ram = self.ram_weights,
                vram = self.vram_weights,
                "finished loading tensors from mmap"
            );
            Ok(self.weights_tb)
        } else {
            Err(LoadError::IncompleteTensors {
                got: self.n_loaded,
                exp: n_exp_tensors,
            })
        }
    }
}

use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct Config {
    // TODO: remove unused configs
    pub runtime: RuntimeConfig,
    pub runner: RunnerConfig,
    pub model: ModelConfig,
    pub lora: Option<LoraConfig>,
}

impl Config {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Config {
            runtime: Default::default(),
            runner: Default::default(),
            model: ModelConfig::new(model_path),
            lora: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub threads: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        RuntimeConfig {
            threads: num_cpus::get(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RunnerConfig {
    /// RNG seed. If None, a random seed will be used.
    pub seed: Option<u32>,
    /// The length of the context window for inference.
    pub window_len: usize,
    /// The batched size for inference prompt processing.
    ///
    /// Must be >=32 to use BLAS.
    pub batch_len: usize,
    /// The number of GPU layers to store in VRAM. If set to None, all model layers are offloaded to
    /// the GPU.
    pub gpu_layers: Option<usize>,
    // /// The GPU that is used for scratch and small tensors.
    // pub main_gpu: usize,
    // /// How split tensors should be distributed across GPUs.
    // pub tensor_split: f32,
    /// RoPE base frequency.
    pub rope_freq_base: f32,
    /// RoPE frequency scale.
    pub rope_freq_scale: f32,
    pub f16_kv: bool,
    /// Ignore EOS tokens in the output and continue generating. This is useful for testing,
    /// benchmarking, or any time an exact number of tokens needs to be generated.
    pub ignore_eos: bool,
    // TODO: callback? or maybe defined elsewhere
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            seed: None,
            window_len: 1024,
            batch_len: 64,
            gpu_layers: None,
            rope_freq_base: 10_000.,
            rope_freq_scale: 1.,
            // TODO: what should be the default value here?
            f16_kv: true,
            ignore_eos: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub path: PathBuf,
    pub alias: Option<String>,
    pub prompt: Option<String>,
    pub input_prefix: Option<String>,
    pub input_suffix: Option<String>,
    pub grammar: Option<String>,
    pub antiprompt: Vec<String>,
}

impl ModelConfig {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        ModelConfig {
            path: path.into(),
            alias: None,
            prompt: None,
            input_prefix: None,
            input_suffix: None,
            grammar: None,
            antiprompt: vec![],
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LoraConfig {
    /// LoRA adapter path.
    pub adapter: PathBuf,
    /// Base model path for the LoRA adapter.
    pub base: PathBuf,
}

pub struct Config {
    pub llm: infernal_llama::Config,
    pub print_llamacpp_timings: bool,
    pub max_infer_steps: Option<usize>,
    pub warmup_rounds: Option<usize>,
    pub reps: Option<usize>,
    /// Only consider samples that have at least this many tokens evaluated.
    pub min_tokens: Option<usize>,
    // TODO: params for iphone specific slowdown evasion
}

impl Config {
    pub fn max_infer_steps(&self) -> usize {
        self.max_infer_steps.unwrap_or(defaults::MAX_INFER_STEPS)
    }

    pub fn warmup_rounds(&self) -> usize {
        self.warmup_rounds.unwrap_or(defaults::WARMUP_ROUNDS)
    }

    pub fn reps(&self) -> usize {
        self.reps.unwrap_or(defaults::REPS)
    }
}

pub mod defaults {
    pub const MAX_INFER_STEPS: usize = 1024;
    pub const WARMUP_ROUNDS: usize = 1;
    pub const REPS: usize = 10;
}

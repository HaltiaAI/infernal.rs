use crate::{
    sample::{
        f::{self, Apply},
        Candidates, DistSampler, Sample,
    },
    TokenId,
};

/// Mirostat 1.0 algorithm described on the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
pub struct MirostatV1 {
    temp: f::Temp,
    core: Core,
    n_vocab: usize,
    /// The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value
    /// that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the
    /// paper, they use `m = 100`, but you can experiment with different values to see how it
    /// affects the performance of the algorithm.
    m: usize,
}

impl MirostatV1 {
    pub fn new(temp: f::Temp, n_vocab: usize, tau: f32, eta: f32) -> Self {
        Self {
            temp,
            core: Core::new(tau, eta),
            n_vocab,
            m: 100,
        }
    }
}

impl Sample for MirostatV1 {
    fn sample(&mut self, cand: &mut Candidates) -> Option<TokenId> {
        self.temp.apply(cand);

        f::Softmax.apply(cand);

        // Estimate s_hat using the most probable m tokens.
        let mut sum_t_b = 0.;
        let mut sum_t_sq = 0.;
        let mut i = 0;
        while i + 1 < self.m && i + 1 < cand.list.len() {
            let t = ((i + 2) as f32 / (i + 1) as f32).ln();
            let b = (cand.list[i].p / cand.list[i + 1].p).ln();
            sum_t_b += t * b;
            sum_t_sq += t * t;
            i += 1;
        }
        let s_hat = sum_t_b / sum_t_sq;

        // Compute k from the estimated s_hat and target surprise value.
        let eps_hat = s_hat - 1.;
        let k = {
            let b =
                (eps_hat * 2f32.powf(self.core.mu)) / (1. - (self.n_vocab as f32).powf(-eps_hat));
            let e = 1. / s_hat;
            b.powf(e)
        };

        f::TopK {
            // TODO: is this correct? it's what the llama.cpp code does but not sure
            // if rust and cpp int conversions work the same
            k: k as usize,
            min_keep: 1,
        }
        .apply(cand);

        self.core.sample(cand)
    }
}

/// Mirostat 2.0 algorithm described on the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
pub struct MirostatV2 {
    temp: f::Temp,
    core: Core,
}

impl MirostatV2 {
    pub fn new(temp: f::Temp, tau: f32, eta: f32) -> Self {
        Self {
            temp,
            core: Core::new(tau, eta),
        }
    }
}

impl Sample for MirostatV2 {
    fn sample(&mut self, cand: &mut Candidates) -> Option<TokenId> {
        self.temp.apply(cand);

        f::Softmax.apply(cand);

        // Truncate the words with surprise values greater than mu.
        let new_len = cand.list.iter().position(|t| -t.p.log2() > self.core.mu)?;
        cand.list.truncate(if new_len == 0 { 1 } else { new_len });

        f::Softmax.apply(cand);

        self.core.sample(cand)
    }
}

struct Core {
    /// The target cross-entropy (or surprise) value you want to achieve for the generated text.
    /// A higher value corresponds to more surprising or less predictable text, while a lower value
    /// corresponds to less surprising or more predictable text.
    tau: f32,
    /// The learning rate used to update `mu` based on the error between the target and observed
    /// surprise of the sampled word. A larger learning rate will cause `mu` to be updated more
    /// quickly, while a smaller learning rate will result in slower updates.
    eta: f32,
    /// Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2
    /// * tau`) and is updated in the algorithm based on the error between the target and observed
    /// surprise.
    mu: f32,
}

impl Core {
    pub fn new(tau: f32, eta: f32) -> Self {
        Self {
            tau,
            eta,
            mu: 2. * tau,
        }
    }

    fn sample(&mut self, cand: &mut Candidates) -> Option<TokenId> {
        let x = DistSampler.sample(cand)?;

        // Compute error as the difference between observed surprise and target
        // surprise value
        let x_pos = cand.list.iter().position(|t| t.id == x)?;
        let observed_surprise = -cand.list[x_pos].p.log2();
        let e = observed_surprise - self.tau;

        // Update mu using the learning rate and error.
        self.mu -= self.eta * e;

        Some(x)
    }
}

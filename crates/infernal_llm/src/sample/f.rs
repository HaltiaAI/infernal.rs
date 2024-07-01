//! List of sampling function objects used by samplers to adjust the candidate list.

use crate::sample::Candidates;

// TODO: are the reserve calls correct? i think additional is based on len() so clearing first
// should work, but not sure

/// Applies a sampling function to the candidates list.
pub trait Apply {
    fn apply(&mut self, cand: &mut Candidates);
}

/// Sorts candidate tokens by their logits in descending order and calculates probabilities based on logits.
pub struct Softmax;

impl Apply for Softmax {
    #[inline(always)]
    fn apply(&mut self, cand: &mut Candidates) {
        if cand.list.is_empty() {
            // TODO: or should this be an error?
            return;
        }

        if !cand.sorted {
            // Sort scores in descending order
            cand.list
                .sort_by(|a, b| b.logit.partial_cmp(&a.logit).expect("invalid logit"));
            cand.sorted = true;
        }

        let max_logit = cand.list[0].logit;
        let mut sum = 0.;
        for t in &mut cand.list {
            t.logit = (t.logit - max_logit).exp();
            sum += t.logit;
        }

        for t in &mut cand.list {
            t.logit /= sum;
        }
    }
}

/// Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration":
/// https://arxiv.org/abs/1904.09751
pub struct TopK {
    pub k: usize,
    pub min_keep: usize,
}

impl Apply for TopK {
    #[inline(always)]
    fn apply(&mut self, cand: &mut Candidates) {
        self.k = self.k.max(self.min_keep);
        self.k = self.k.min(cand.list.len());

        if !cand.sorted {
            // Sort scores in descending order
            if self.k == cand.list.len() {
                cand.list
                    .sort_by(|a, b| b.logit.partial_cmp(&a.logit).expect("invalid logit"));
            } else {
                cand.list[..self.k].sort_by(|a, b| {
                    b.logit.partial_cmp(&a.logit).expect("invalid logit")
                    // .then_with(|| a.id.cmp(&b.id))
                });
                cand.list.truncate(self.k);
            }
            cand.sorted = true;
        } else if self.k != cand.list.len() {
            cand.list.truncate(self.k);
        }
    }
}

/// Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
pub struct TailFree {
    pub z: f32,
    pub min_keep: usize,
    deriv1: Vec<f32>,
    deriv2: Vec<f32>,
}

impl TailFree {
    pub fn new(z: f32, min_keep: usize) -> Self {
        Self {
            z,
            min_keep,
            deriv1: Vec::new(),
            deriv2: Vec::new(),
        }
    }
}

impl Apply for TailFree {
    #[inline(always)]
    fn apply(&mut self, cand: &mut Candidates) {
        if self.z >= 1. || cand.list.len() <= 2 {
            return;
        }

        Softmax.apply(cand);

        self.deriv1.clear();
        self.deriv2.clear();
        self.deriv1.reserve(cand.list.len() - 1);
        self.deriv2.reserve(cand.list.len() - 2);

        for i in 0..cand.list.len() - 1 {
            let p1 = cand.list[i].logit;
            let p2 = cand.list[i + 1].logit;
            self.deriv1.push(p1 - p2);
        }
        for i in 0..cand.list.len() - 2 {
            let p1 = self.deriv1[i];
            let p2 = self.deriv1[i + 1];
            self.deriv2.push(p1 - p2);
        }

        // Calculate absolute value of second derivatives.
        for d in &mut self.deriv2 {
            *d = d.abs();
        }

        // Normalize the second derivatives
        let mut sum = 0.;
        for d in &self.deriv2 {
            sum += d;
        }

        if sum > 1e-6 {
            for d in &mut self.deriv2 {
                *d /= sum;
            }
        } else {
            // Otherwise we just use the uniform distribution.
            let n = self.deriv2.len();
            for d in &mut self.deriv2 {
                *d = 1. / n as f32;
            }
        }

        // Calculate the threshold
        let mut sum = 0.;
        for (i, d) in self.deriv2.iter().enumerate() {
            sum += d;
            if sum >= self.z && i >= self.min_keep {
                cand.list.truncate(i);
                break;
            }
        }
    }
}

/// Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
pub struct TypicalP {
    pub p: f32,
    pub min_keep: usize,
    shifted_scores: Vec<f32>,
    indices: Vec<usize>,
}

impl TypicalP {
    pub fn new(p: f32, min_keep: usize) -> Self {
        Self {
            p,
            min_keep,
            shifted_scores: Vec::new(),
            indices: Vec::new(),
        }
    }
}

impl Apply for TypicalP {
    /// Reference implementation:
    /// https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
    #[inline(always)]
    fn apply(&mut self, cand: &mut Candidates) {
        if self.p >= 1. {
            return;
        }

        Softmax.apply(cand);

        let mut entropy = 0.;
        for t in &cand.list {
            entropy += -t.p * t.p.ln();
        }

        // Compute the absolute difference between negative log probability and
        // entropy for each candidate
        self.shifted_scores.clear();
        self.shifted_scores.reserve(cand.list.len());
        for t in &cand.list {
            let score = (-t.p.ln() - entropy).abs();
            self.shifted_scores.push(score);
        }

        // Sort tokens based on the shifted_scores and their corresponding indices
        // TODO#perf: I think we could get rid of the indices vec and just sort the shifted_scores
        // vec, find last idx below
        self.indices.clear();
        self.indices.extend(0..cand.list.len());
        self.indices.sort_by(|&a, &b| {
            self.shifted_scores[a]
                .partial_cmp(&self.shifted_scores[b])
                .expect("invalid score")
        });

        // Compute the cumulative probabilities
        let mut sum = 0.;
        let mut last_idx = self.indices.len();
        for (i, idx) in self.indices.iter().copied().enumerate() {
            sum += cand.list[idx].p;
            // Check if the running sum is greater than typical and if we have kept at
            // least min_keep tokens.
            if sum > self.p && i + 1 >= self.min_keep {
                last_idx = i + 1;
                break;
            }
        }

        // Truncate the output vector to keep only the locally typical tokens.
        // TODO#perf: can we avoid this new alloc?
        // TODO#perf: if not, we can just always keep one vec in self and memswap it with the input
        // one. that way only needto do allocs upfront and not on every call.
        let mut new_cands = Vec::with_capacity(last_idx);
        for idx in self.indices.iter().copied().take(last_idx) {
            new_cands.push(cand.list[idx]);
        }

        cand.list = new_cands;
    }
}

/// Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration":
/// https://arxiv.org/abs/1904.09751
pub struct TopP {
    pub p: f32,
    pub min_keep: usize,
}

impl Apply for TopP {
    #[inline(always)]
    fn apply(&mut self, cand: &mut Candidates) {
        if self.p >= 1. {
            return;
        }

        Softmax.apply(cand);

        let mut sum = 0.;
        for (i, t) in cand.list.iter().enumerate() {
            sum += t.p;
            if sum >= self.p && i + 1 >= self.min_keep {
                cand.list.truncate(i + 1);
                break;
            }
        }
    }
}

pub struct Temp {
    pub t: f32,
}

impl Apply for Temp {
    #[inline(always)]
    fn apply(&mut self, cand: &mut Candidates) {
        if self.t == 1. {
            return;
        }

        for t in &mut cand.list {
            t.logit /= self.t;
        }
    }
}

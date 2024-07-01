//! Sampling strategies for generating tokens.
//!
//! An LLM always generates a deterministic set of logits, output probabilities for each token in
//! its vocabulary. End user applications need to sample these logits according to certain
//! strategies. This module provides the most commonly used ones. For most use cases, the

use std::collections::HashMap;

use rand::distributions::{Distribution, WeightedIndex};

use crate::{sample::f::Apply, TokenId};

pub use self::{
    greedy::Greedy,
    mirostat::{MirostatV1, MirostatV2},
    temp::Temp,
};

pub(crate) mod f;
mod greedy;
mod mirostat;
mod temp;

#[derive(Debug, Clone, Copy)]
pub(crate) struct TokenData {
    pub(crate) id: TokenId,
    pub(crate) logit: f32,
    pub(crate) p: f32,
}

#[derive(Debug, Default)]
pub struct Candidates {
    pub(crate) list: Vec<TokenData>,
    pub(crate) sorted: bool,
}

pub trait Sample {
    /// Sample a token from the given candidates.
    fn sample(&mut self, cand: &mut Candidates) -> Option<TokenId>;
}

#[derive(Debug)]
pub struct Sampler<T: Sample> {
    sampler: T,
    cand: Candidates,
    n_vocab: usize,
    freq_map: HashMap<TokenId, usize>,
}

impl<T: Sample> Sampler<T> {
    pub fn new(sampler: T, n_vocab: usize) -> Self {
        Self {
            sampler,
            cand: Default::default(),
            n_vocab,
            freq_map: Default::default(),
        }
    }

    /// Samples the next token while applying repetition and frequency penalties.
    pub fn sample_next(&mut self, logits: &[f32], context_window: &[TokenId]) -> Option<TokenId> {
        debug_assert_eq!(logits.len(), self.n_vocab);

        self.cand.list.clear();
        for (id, logit) in logits.iter().enumerate() {
            // let p = logit.exp();
            self.cand.list.push(TokenData {
                id: id as TokenId,
                logit: *logit,
                p: 0.,
            });
        }

        // Apply penalties if tokens have already been generated before.
        if !context_window.is_empty() {
            self.penalize_repetition(context_window);
            // TODO: we didn't use this in the old code but llama.cpp does use it, so should prob
            // re-enable
            // self.penalize_frequency(context_window);
        }

        self.sampler.sample(&mut self.cand)
    }

    fn penalize_repetition(&mut self, context_window: &[TokenId]) {
        let penalty = 1.1; // TODO: add to config
        for t in &mut self.cand.list {
            let has_occurred = context_window.iter().any(|&id| id == t.id);
            if !has_occurred {
                continue;
            };

            // The academic publication that described this technique actually just only
            // divided, but that would cause tokens with negative logits to become more
            // likely, which is obviously wrong. This is common fix for this problem,
            // which is to multiply by the penalty instead of dividing.
            if t.logit <= 0. {
                t.logit *= penalty;
            } else {
                t.logit /= penalty;
            }
        }

        self.cand.sorted = false;
    }

    fn penalize_frequency(&mut self, context_window: &[TokenId]) {
        let alpha_freq = 0.; // TODO: add to config
        let alpha_presence = 0.; // TODO: add to config
        if alpha_freq == 0. && alpha_presence == 0. {
            return;
        }

        // Create a frequency map to count occurrences of each token in last_tokens
        self.freq_map.clear();
        for &id in context_window {
            *self.freq_map.entry(id).or_insert(0) += 1;
        }

        // Apply frequency and presence penalties to the candidates
        for t in &mut self.cand.list {
            let freq = self.freq_map.get(&t.id).copied().unwrap_or(0);
            if freq == 0 {
                continue;
            }

            t.logit -= freq as f32 * alpha_freq + if freq > 0 { alpha_presence } else { 0. };
        }
    }
}

pub struct DistSampler;

impl Sample for DistSampler {
    fn sample(&mut self, cand: &mut Candidates) -> Option<TokenId> {
        f::Softmax.apply(cand);

        let probs = cand.list.iter().map(|x| x.p as f64);
        let dist = WeightedIndex::new(probs).unwrap();
        // TODO: make rng configurable, although we're likely not going to use any sampler that uses
        // it (only greedy) for now, so not urgent
        let idx = dist.sample(&mut rand::thread_rng());
        Some(cand.list[idx].id)
    }
}

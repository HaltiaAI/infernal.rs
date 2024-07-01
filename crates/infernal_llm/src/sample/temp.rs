use crate::{
    sample::{
        f::{self, Apply},
        Candidates, DistSampler, Sample,
    },
    TokenId,
};

// TODO: add ctor, make fields priv
pub struct Temp {
    pub top_k: f::TopK,
    pub tail_free: f::TailFree,
    pub typical: f::TypicalP,
    pub top_p: f::TopP,
    pub temp: f::Temp,
}

impl Sample for Temp {
    fn sample(&mut self, cand: &mut Candidates) -> Option<TokenId> {
        self.top_k.apply(cand);
        self.tail_free.apply(cand);
        self.typical.apply(cand);
        self.top_p.apply(cand);
        self.temp.apply(cand);

        DistSampler.sample(cand)
    }
}

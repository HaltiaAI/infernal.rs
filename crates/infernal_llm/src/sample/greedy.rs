use crate::{
    sample::{Candidates, Sample},
    TokenId,
};

#[derive(Debug)]
pub struct Greedy;

impl Sample for Greedy {
    #[inline(always)]
    fn sample(&mut self, cand: &mut Candidates) -> Option<TokenId> {
        let mut max_token = None;
        for t in cand.list.iter() {
            if let Some(id) = max_token {
                if t.logit > cand.list[id as usize].logit {
                    max_token = Some(t.id);
                }
            } else {
                max_token = Some(t.id);
            }
        }
        max_token
    }
}

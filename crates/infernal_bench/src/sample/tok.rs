use std::{
    fmt,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
    time::Duration,
};

use crate::sample::{MeanDist, Sample, Samples, Sqrt};

pub type TokSamples = Samples<TokSample>;

#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct TokSample {
    pub tokens: usize,
    pub time: Duration,
}

impl Sample for TokSample {
    type Value = f32;

    type Aggr = TokSampleAggr;

    fn main_value(&self) -> Self::Value {
        self.tok_per_s()
    }
}

impl TokSample {
    pub fn new(tokens: usize, time: Duration) -> Self {
        Self { tokens, time }
    }

    pub fn tok_per_s(&self) -> f32 {
        self.tokens as f32 / self.time.as_secs_f32()
    }
}

impl fmt::Debug for TokSample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sample")
            .field("tok_per_s", &self.tok_per_s())
            .field("tokens", &self.tokens)
            .field("time", &self.time)
            .finish()
    }
}

impl fmt::Display for TokSample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{tok_per_s:.2}tok/s total_tokens={tok} time={time:?}]",
            tok_per_s = self.tok_per_s(),
            tok = self.tokens,
            time = self.time
        )
    }
}

/// An aggregation of samples (e.g. an average or mean, which may be the average of the midpoints
/// for even populations).
#[derive(Clone, Copy, Debug, Default)]
pub struct TokSampleAggr {
    pub tok_per_s: f32,
    pub tokens: f32,
    pub time_secs: f32,
}

impl Sqrt for TokSampleAggr {
    fn sqrt(&mut self) {
        self.tok_per_s = self.tok_per_s.sqrt();
        self.tokens = self.tokens.sqrt();
        self.time_secs = self.time_secs.sqrt();
    }
}

impl MeanDist for TokSampleAggr {
    fn dist_from_mean(&self, mean: &Self) -> Self {
        let mut dist = *self;
        dist -= *mean;
        dist *= dist;
        dist
    }
}

impl From<TokSample> for TokSampleAggr {
    fn from(sample: TokSample) -> Self {
        Self {
            tok_per_s: sample.tok_per_s(),
            tokens: sample.tokens as f32,
            time_secs: sample.time.as_secs_f32(),
        }
    }
}

impl MulAssign for TokSampleAggr {
    fn mul_assign(&mut self, rhs: TokSampleAggr) {
        self.tok_per_s *= rhs.tok_per_s;
        self.tokens *= rhs.tokens;
        self.time_secs *= rhs.time_secs;
    }
}

impl AddAssign for TokSampleAggr {
    fn add_assign(&mut self, rhs: TokSampleAggr) {
        self.tok_per_s += rhs.tok_per_s;
        self.tokens += rhs.tokens;
        self.time_secs += rhs.time_secs;
    }
}

impl SubAssign for TokSampleAggr {
    fn sub_assign(&mut self, rhs: TokSampleAggr) {
        self.tok_per_s -= rhs.tok_per_s;
        self.tokens -= rhs.tokens;
        self.time_secs -= rhs.time_secs;
    }
}

impl AddAssign<TokSample> for TokSampleAggr {
    fn add_assign(&mut self, s: TokSample) {
        self.tok_per_s += s.tok_per_s();
        self.tokens += s.tokens as f32;
        self.time_secs += s.time.as_secs_f32();
    }
}

impl DivAssign<usize> for TokSampleAggr {
    fn div_assign(&mut self, rhs: usize) {
        self.tok_per_s /= rhs as f32;
        self.tokens /= rhs as f32;
        self.time_secs /= u32::try_from(rhs).unwrap() as f32; // FIXME: possible loss of precision
    }
}

impl fmt::Display for TokSampleAggr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{tok_per_s:.2}tok/s total={tok} time={time}s]",
            tok_per_s = self.tok_per_s,
            tok = self.tokens,
            time = self.time_secs
        )
    }
}

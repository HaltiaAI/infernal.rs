use std::{
    fmt,
    ops::{AddAssign, DivAssign, MulAssign},
    time::Duration,
};

use crate::sample::{MeanDist, Sample, Samples, Sqrt};

pub type TimeSamples = Samples<TimeSample>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct TimeSample(Duration);

impl From<Duration> for TimeSample {
    fn from(d: Duration) -> Self {
        Self(d)
    }
}

impl Sample for TimeSample {
    type Value = Duration;
    type Aggr = Self;

    fn main_value(&self) -> Self::Value {
        self.0
    }
}

impl TimeSample {
    pub fn new(time: Duration) -> Self {
        Self(time)
    }
}

impl fmt::Display for TimeSample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0,)
    }
}

impl Sqrt for TimeSample {
    fn sqrt(&mut self) {
        // TODO: Is there a better way to do this, without precision loss?
        self.0 = Duration::from_secs_f64(self.0.as_secs_f64().sqrt());
    }
}

impl MeanDist for TimeSample {
    fn dist_from_mean(&self, mean: &Self) -> Self {
        let f = self.0.as_nanos();
        let m = mean.0.as_nanos();
        let d = f - m;
        let d = d * d;
        Self(Duration::from_nanos(d.try_into().unwrap()))
    }
}

impl MulAssign for TimeSample {
    fn mul_assign(&mut self, rhs: Self) {
        // TODO: Is there a better way to do this, without precision loss?
        self.0 = Duration::from_secs_f64(self.0.as_secs_f64() * rhs.0.as_secs_f64());
    }
}

impl AddAssign for TimeSample {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl DivAssign<usize> for TimeSample {
    fn div_assign(&mut self, rhs: usize) {
        self.0 /= rhs as u32;
    }
}

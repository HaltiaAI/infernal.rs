use std::{
    fmt,
    ops::{AddAssign, Div, MulAssign, SubAssign},
    time::Duration,
};

/// Stores measurement samples and provides methods for calculating statistics for tokens per
/// second.
#[derive(Default, Debug)]
pub struct Stats {
    pub init_time: Duration,
    sample: Vec<Sample>,
    feed: Vec<Sample>,
    infer: Vec<Sample>,
}

#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Sample {
    pub tokens: usize,
    pub time: Duration,
}

impl Sample {
    pub fn new(tokens: usize, time: Duration) -> Self {
        Self { tokens, time }
    }

    pub fn tok_per_s(&self) -> f32 {
        self.tokens as f32 / self.time.as_secs_f32()
    }
}

impl fmt::Debug for Sample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sample")
            .field("tok_per_s", &self.tok_per_s())
            .field("tokens", &self.tokens)
            .field("time", &self.time)
            .finish()
    }
}

impl fmt::Display for Sample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{tok_per_s:.2} tok/s (tok={tok} time={time:?})",
            tok_per_s = self.tok_per_s(),
            tok = self.tokens,
            time = self.time
        )
    }
}

/// An aggregation of samples (e.g. an average or mean, which may be the average of the midpoint).
#[derive(Clone, Copy, Debug, Default)]
pub struct AggrSample {
    pub tok_per_s: f32,
    pub tokens: f32,
    pub time_secs: f32,
}

impl AggrSample {
    fn sqrt(&mut self) {
        self.tok_per_s = self.tok_per_s.sqrt();
        self.tokens = self.tokens.sqrt();
        self.time_secs = self.time_secs.sqrt();
    }
}

impl From<Sample> for AggrSample {
    fn from(sample: Sample) -> Self {
        Self {
            tok_per_s: sample.tok_per_s(),
            tokens: sample.tokens as f32,
            time_secs: sample.time.as_secs_f32(),
        }
    }
}

impl MulAssign<AggrSample> for AggrSample {
    fn mul_assign(&mut self, rhs: AggrSample) {
        self.tok_per_s *= rhs.tok_per_s;
        self.tokens *= rhs.tokens;
        self.time_secs *= rhs.time_secs;
    }
}

impl AddAssign<AggrSample> for AggrSample {
    fn add_assign(&mut self, rhs: AggrSample) {
        self.tok_per_s += rhs.tok_per_s;
        self.tokens += rhs.tokens;
        self.time_secs += rhs.time_secs;
    }
}

impl SubAssign<AggrSample> for AggrSample {
    fn sub_assign(&mut self, rhs: AggrSample) {
        self.tok_per_s -= rhs.tok_per_s;
        self.tokens -= rhs.tokens;
        self.time_secs -= rhs.time_secs;
    }
}

impl Div<usize> for AggrSample {
    type Output = Self;

    fn div(self, rhs: usize) -> Self::Output {
        Self {
            tok_per_s: self.tok_per_s / rhs as f32,
            tokens: self.tokens / rhs as f32,
            time_secs: self.time_secs / u32::try_from(rhs).unwrap() as f32, // FIXME: possible loss of precision
        }
    }
}

impl fmt::Display for AggrSample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{tok_per_s:.2} tok/s (tok={tok} time={time}s)",
            tok_per_s = self.tok_per_s,
            tok = self.tokens,
            time = self.time_secs
        )
    }
}

impl Stats {
    pub fn insert_feed(&mut self, sample: Sample) {
        Self::insert(&mut self.feed, sample);
    }

    pub fn insert_infer(&mut self, sample: Sample) {
        Self::insert(&mut self.infer, sample);
    }

    /// Inserts sample into collection, keeping it sorted so that mean calculation works as expected.
    fn insert(coll: &mut Vec<Sample>, sample: Sample) {
        coll.push(sample);
        // let pos = match coll.binary_search_by(|e| e.cmp(&sample)) {
        //     Ok(pos) => pos + 1,
        //     Err(pos) => pos,
        // };
        // coll.insert(pos, sample);
    }

    pub fn feed_min_and_max(&self) -> Option<(Sample, Sample)> {
        Self::min_and_max(&self.feed)
    }

    pub fn infer_min_and_max(&self) -> Option<(Sample, Sample)> {
        Self::min_and_max(&self.infer)
    }

    fn min_and_max(samples: &[Sample]) -> Option<(Sample, Sample)> {
        let mut min: Option<Sample> = None;
        let mut max: Option<Sample> = None;

        for s in samples {
            let tok_per_sec = s.tok_per_s();
            if let Some(m) = min {
                if tok_per_sec < m.tok_per_s() {
                    min = Some(*s);
                }
            } else {
                min = Some(*s);
            }

            if let Some(m) = max {
                if tok_per_sec > m.tok_per_s() {
                    max = Some(*s);
                }
            } else {
                max = Some(*s);
            }
        }

        if let (Some(min), Some(max)) = (min, max) {
            Some((min, max))
        } else {
            None
        }
    }

    pub fn feed_mean(&self) -> AggrSample {
        Self::mean(&self.feed)
    }

    pub fn infer_mean(&self) -> AggrSample {
        Self::mean(&self.infer)
    }

    fn mean(samples: &[Sample]) -> AggrSample {
        if samples.len() % 2 == 1 {
            let mid = samples.len() / 2;
            samples[mid].into()
        } else if samples.len() > 1 {
            let mid = samples.len() / 2;
            Self::avg(&samples[mid - 1..=mid])
        } else {
            samples.first().cloned().unwrap_or_default().into()
        }
    }

    pub fn feed_avg(&self) -> AggrSample {
        Self::avg(&self.feed)
    }

    pub fn infer_avg(&self) -> AggrSample {
        Self::avg(&self.infer)
    }

    fn avg(samples: &[Sample]) -> AggrSample {
        let total = samples
            .iter()
            .fold(AggrSample::default(), |acc, &s| AggrSample {
                tok_per_s: acc.tok_per_s + s.tok_per_s(),
                tokens: acc.tokens + s.tokens as f32,
                time_secs: acc.time_secs + s.time.as_secs_f32(),
            });
        total / samples.len()
    }

    pub fn feed_stdev(&self) -> AggrSample {
        Self::stdev(&self.feed)
    }

    pub fn infer_stdev(&self) -> AggrSample {
        Self::stdev(&self.infer)
    }

    fn stdev(samples: &[Sample]) -> AggrSample {
        let mut v = Self::variance(samples);
        v.sqrt();
        v
    }

    pub fn feed_variance(&self) -> AggrSample {
        Self::variance(&self.feed)
    }

    pub fn infer_variance(&self) -> AggrSample {
        Self::variance(&self.infer)
    }

    fn variance(samples: &[Sample]) -> AggrSample {
        let mean = Self::avg(samples);
        let mut total = AggrSample::default();
        for &sample in samples {
            let mut diff = AggrSample::from(sample);
            diff -= mean;
            diff *= diff;
            total += diff;
        }
        total / samples.len()
    }
}

// TODO: add Display impl calculating important statistics

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_sample_eq {
        ($val:expr, $exp:expr) => {
            assert_sample_eq!($val, $exp, 0.0001)
        };
        ($val:expr, $exp:expr, $eps:expr) => {{
            eprintln!("value: {:?}", $val);
            eprintln!("expected: {:?}", $exp);
            assert!(($val.tok_per_s - $exp.tok_per_s).abs() < $eps as f32);
            assert!(($val.tokens - $exp.tokens).abs() < $eps);
            assert!(($val.time_secs - $exp.time_secs).abs() < $eps as f32);
        }};
    }

    macro_rules! s {
        ($s:expr) => {
            Duration::from_secs_f32($s as f32)
        };
    }

    #[test]
    fn avg() {
        let stats = Stats {
            feed: vec![
                Sample::new(1, s!(1)), // 1 tok/s
                Sample::new(9, s!(3)), // 3 tok/s
            ],
            ..Stats::default()
        };
        println!("{stats:#?}");
        assert_sample_eq!(
            stats.feed_avg(),
            AggrSample {
                tok_per_s: 2.,
                tokens: 5.,
                time_secs: 2.,
            }
        );
    }

    #[test]
    fn mean() {
        let stats = Stats {
            feed: vec![Sample::new(1, s!(1)), Sample::new(9, s!(3))],
            ..Stats::default()
        };
        assert_sample_eq!(
            stats.feed_mean(),
            AggrSample {
                tok_per_s: 2.,
                tokens: 5.,
                time_secs: 2.,
            }
        );

        let stats = Stats {
            feed: vec![
                Sample::new(1, s!(1)), // 1 tok/s
                Sample::new(6, s!(3)), // 2 tok/s
                Sample::new(8, s!(2)), // 4 tok/s
            ],
            ..Stats::default()
        };
        assert_sample_eq!(
            stats.feed_mean(),
            AggrSample {
                tok_per_s: 2.,
                tokens: 6.,
                time_secs: 3.,
            }
        );
    }

    #[test]
    fn min_and_max() {
        let stats = Stats {
            feed: vec![
                Sample::new(6, s!(3)), // 2 tok/s
                Sample::new(8, s!(2)), // 4 tok/s
                Sample::new(1, s!(1)), // 1 tok/s
            ],
            ..Stats::default()
        };
        assert_eq!(
            stats.feed_min_and_max(),
            Some((stats.feed[2], stats.feed[1])),
        );
    }

    #[test]
    fn variance() {
        let stats = Stats {
            feed: vec![
                Sample::new(12, s!(2)),   // 6 tok/s
                Sample::new(10, s!(1)),   // 10 tok/s
                Sample::new(20, s!(2.5)), // 8 tok/s
                Sample::new(21, s!(3)),   // 7 tok/s
                Sample::new(20, s!(2)),   // 10 tok/s
                Sample::new(2, s!(1)),    // 2 tok/s
            ],
            ..Stats::default()
        };
        assert_sample_eq!(
            stats.feed_variance(),
            AggrSample {
                tok_per_s: 7.4722222,
                tokens: 47.472_22,
                time_secs: 0.534_722_2,
            }
        );
    }

    #[test]
    fn stdev() {
        let stats = Stats {
            feed: vec![
                Sample::new(12, s!(2)),   // 6 tok/s
                Sample::new(10, s!(1)),   // 10 tok/s
                Sample::new(20, s!(2.5)), // 8 tok/s
                Sample::new(21, s!(3)),   // 7 tok/s
                Sample::new(20, s!(2)),   // 10 tok/s
                Sample::new(2, s!(1)),    // 2 tok/s
            ],
            ..Stats::default()
        };
        assert_sample_eq!(
            stats.feed_stdev(),
            AggrSample {
                tok_per_s: 2.7335366,
                tokens: 6.890_009,
                time_secs: 0.731_247,
            }
        );
    }
}

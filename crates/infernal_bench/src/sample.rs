pub mod time;
pub mod tok;

use std::ops::{AddAssign, DivAssign, MulAssign};

pub use time::*;
pub use tok::*;

#[derive(Default, Debug)]
pub struct Samples<T: Sample>(pub Vec<T>);

impl<T: Sample> Samples<T> {
    /// Inserts sample into collection, keeping it sorted so that mean calculation works as expected.
    pub fn insert(&mut self, sample: T) {
        let pos = match self.0.binary_search_by(|e| e.cmp(&sample)) {
            Ok(pos) => pos + 1,
            Err(pos) => pos,
        };
        self.0.insert(pos, sample);
    }

    pub fn min_and_max(&self) -> Option<(T, T)> {
        min_and_max(&self.0)
    }

    pub fn mean(&self) -> T::Aggr {
        mean(&self.0)
    }

    pub fn avg(&self) -> T::Aggr {
        avg(&self.0)
    }

    pub fn stdev(&self) -> T::Aggr {
        stdev(&self.0)
    }

    pub fn var(&self) -> T::Aggr {
        var(&self.0)
    }
}

pub trait Sample: Sized + Copy + Clone + Default + Ord {
    type Value: PartialEq + PartialOrd + Clone + Default;
    type Aggr: Copy
        + Clone
        + Default
        + Sqrt
        + MeanDist
        + MulAssign
        + AddAssign
        + AddAssign<Self>
        + DivAssign<usize>
        + From<Self>;

    fn main_value(&self) -> Self::Value;
}

#[doc(hidden)]
pub trait Sqrt {
    fn sqrt(&mut self);
}

#[doc(hidden)]
pub trait MeanDist {
    fn dist_from_mean(&self, mean: &Self) -> Self;
}

pub(crate) fn min_and_max<T: Sample>(samples: &[T]) -> Option<(T, T)> {
    let mut min: Option<T> = None;
    let mut max: Option<T> = None;

    for s in samples {
        let tok_per_sec = s.main_value();
        if let Some(m) = min {
            if tok_per_sec < m.main_value() {
                min = Some(*s);
            }
        } else {
            min = Some(*s);
        }

        if let Some(m) = max {
            if tok_per_sec > m.main_value() {
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

pub(crate) fn mean<T: Sample>(samples: &[T]) -> T::Aggr {
    if samples.len() % 2 == 1 {
        let mid = samples.len() / 2;
        samples[mid].into()
    } else if samples.len() > 1 {
        let mid = samples.len() / 2;
        avg(&samples[mid - 1..=mid])
    } else {
        samples.first().cloned().unwrap_or_default().into()
    }
}

pub(crate) fn avg<T: Sample>(samples: &[T]) -> T::Aggr {
    let mut total = samples.iter().fold(T::Aggr::default(), |mut acc, &s| {
        acc += s;
        acc
    });
    total /= samples.len();
    total
}

pub(crate) fn stdev<T: Sample>(samples: &[T]) -> T::Aggr {
    let mut v = var(samples);
    v.sqrt();
    v
}

pub(crate) fn var<T: Sample>(samples: &[T]) -> T::Aggr {
    let mean = avg(samples);
    let mut total = T::Aggr::default();
    for &s in samples {
        total += T::Aggr::from(s).dist_from_mean(&mean);
    }
    total /= samples.len();
    total
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

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
            Duration::from_secs_f64($s as f64)
        };
    }

    #[test]
    fn avg() {
        let samples = Samples(vec![
            TokSample::new(1, s!(1)), // 1 tok/s
            TokSample::new(9, s!(3)), // 3 tok/s
        ]);
        assert_sample_eq!(
            samples.avg(),
            TokSampleAggr {
                tok_per_s: 2.,
                tokens: 5.,
                time_secs: 2.,
            }
        );
    }

    #[test]
    fn mean() {
        let samples = Samples(vec![TokSample::new(1, s!(1)), TokSample::new(9, s!(3))]);
        assert_sample_eq!(
            samples.mean(),
            TokSampleAggr {
                tok_per_s: 2.,
                tokens: 5.,
                time_secs: 2.,
            }
        );

        let samples = Samples(vec![
            TokSample::new(1, s!(1)), // 1 tok/s
            TokSample::new(6, s!(3)), // 2 tok/s
            TokSample::new(8, s!(2)), // 4 tok/s
        ]);
        assert_sample_eq!(
            samples.mean(),
            TokSampleAggr {
                tok_per_s: 2.,
                tokens: 6.,
                time_secs: 3.,
            }
        );
    }

    #[test]
    fn min_and_max() {
        let samples = Samples(vec![
            TokSample::new(6, s!(3)), // 2 tok/s
            TokSample::new(8, s!(2)), // 4 tok/s
            TokSample::new(1, s!(1)), // 1 tok/s
        ]);
        assert_eq!(samples.min_and_max(), Some((samples.0[2], samples.0[1])),);
    }

    #[test]
    fn variance() {
        let samples = Samples(vec![
            TokSample::new(12, s!(2)),   // 6 tok/s
            TokSample::new(10, s!(1)),   // 10 tok/s
            TokSample::new(20, s!(2.5)), // 8 tok/s
            TokSample::new(21, s!(3)),   // 7 tok/s
            TokSample::new(20, s!(2)),   // 10 tok/s
            TokSample::new(2, s!(1)),    // 2 tok/s
        ]);
        assert_sample_eq!(
            samples.var(),
            TokSampleAggr {
                tok_per_s: 7.472222,
                tokens: 47.472_218,
                time_secs: 0.534_722,
            }
        );
    }

    #[test]
    fn stdev() {
        let samples = Samples(vec![
            TokSample::new(12, s!(2)),   // 6 tok/s
            TokSample::new(10, s!(1)),   // 10 tok/s
            TokSample::new(20, s!(2.5)), // 8 tok/s
            TokSample::new(21, s!(3)),   // 7 tok/s
            TokSample::new(20, s!(2)),   // 10 tok/s
            TokSample::new(2, s!(1)),    // 2 tok/s
        ]);
        assert_sample_eq!(
            samples.stdev(),
            TokSampleAggr {
                tok_per_s: 2.7335366,
                tokens: 6.890_009,
                time_secs: 0.731_247,
            }
        );
    }
}

// 2,1,2.5,3,2,1

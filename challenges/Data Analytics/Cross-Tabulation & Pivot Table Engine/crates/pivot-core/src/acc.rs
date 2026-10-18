//! Accumulators: one per cell and measure. Each can be updated one row at a time, merged with another
//! accumulator of the same kind (this is how subtotals and grand totals are built without touching
//! the rows again), and turned into a number at the end.

use std::collections::HashSet;

use crate::spec::Agg;

#[derive(Debug, Clone)]
pub enum Acc {
    /// Rows, or non-missing values of a field.
    Count(u64),
    Sum {
        n: u64,
        sum: f64,
    },
    /// Count, mean and the sum of squared deviations (Welford), plus extremes and the plain sum.
    Moments {
        n: u64,
        mean: f64,
        m2: f64,
        min: f64,
        max: f64,
    },
    Distinct(HashSet<u64>),
    /// Every value, for the aggregations that need them all: median and mode.
    Values(Vec<f64>),
    /// The value of the earliest (`first`) or latest row, by row number.
    Pick {
        row: usize,
        value: f64,
    },
    Weighted {
        sw: f64,
        swx: f64,
    },
}

impl Acc {
    pub fn new(agg: Agg) -> Acc {
        match agg {
            Agg::Count => Acc::Count(0),
            Agg::Sum => Acc::Sum { n: 0, sum: 0.0 },
            Agg::Mean | Agg::Min | Agg::Max | Agg::Std | Agg::Var => Acc::Moments {
                n: 0,
                mean: 0.0,
                m2: 0.0,
                min: f64::INFINITY,
                max: f64::NEG_INFINITY,
            },
            Agg::Distinct => Acc::Distinct(HashSet::new()),
            Agg::Median | Agg::Mode => Acc::Values(Vec::new()),
            Agg::First | Agg::Last => Acc::Pick {
                row: usize::MAX,
                value: f64::NAN,
            },
            Agg::WeightedMean => Acc::Weighted { sw: 0.0, swx: 0.0 },
        }
    }

    /// Add one row. `x` is the numeric value (if the field has one), `id` its identity, `w` the weight.
    pub fn update(
        &mut self,
        agg: Agg,
        row: usize,
        x: Option<f64>,
        id: Option<u64>,
        w: Option<f64>,
    ) {
        match self {
            Acc::Count(n) => {
                // row counting passes x = Some(1.0); counting a field passes its identity if present
                if x.is_some() || id.is_some() {
                    *n += 1;
                }
            }
            Acc::Sum { n, sum } => {
                if let Some(x) = x {
                    *n += 1;
                    *sum += x;
                }
            }
            Acc::Moments {
                n,
                mean,
                m2,
                min,
                max,
            } => {
                if let Some(x) = x {
                    *n += 1;
                    let d = x - *mean;
                    *mean += d / *n as f64;
                    *m2 += d * (x - *mean);
                    *min = min.min(x);
                    *max = max.max(x);
                }
            }
            Acc::Distinct(set) => {
                if let Some(id) = id {
                    set.insert(id);
                }
            }
            Acc::Values(v) => {
                if let Some(x) = x {
                    v.push(x);
                }
            }
            Acc::Pick { row: r, value } => {
                if let Some(x) = x {
                    let better = if agg == Agg::First {
                        row < *r || *r == usize::MAX
                    } else {
                        *r == usize::MAX || row > *r
                    };
                    if better {
                        *r = row;
                        *value = x;
                    }
                }
            }
            Acc::Weighted { sw, swx } => {
                if let (Some(x), Some(w)) = (x, w) {
                    *sw += w;
                    *swx += w * x;
                }
            }
        }
    }

    /// Fold `other` (same variant) into `self`.
    pub fn merge(&mut self, agg: Agg, other: &Acc) {
        match (self, other) {
            (Acc::Count(a), Acc::Count(b)) => *a += b,
            (Acc::Sum { n, sum }, Acc::Sum { n: n2, sum: s2 }) => {
                *n += n2;
                *sum += s2;
            }
            (
                Acc::Moments {
                    n,
                    mean,
                    m2,
                    min,
                    max,
                },
                Acc::Moments {
                    n: nb,
                    mean: mb,
                    m2: m2b,
                    min: minb,
                    max: maxb,
                },
            ) => {
                if *nb == 0 {
                    return;
                }
                if *n == 0 {
                    *n = *nb;
                    *mean = *mb;
                    *m2 = *m2b;
                } else {
                    // Chan et al.: combine two partial variances exactly
                    let (na, nbf) = (*n as f64, *nb as f64);
                    let total = na + nbf;
                    let delta = mb - *mean;
                    *m2 += m2b + delta * delta * na * nbf / total;
                    *mean += delta * nbf / total;
                    *n += nb;
                }
                *min = min.min(*minb);
                *max = max.max(*maxb);
            }
            (Acc::Distinct(a), Acc::Distinct(b)) => a.extend(b.iter().copied()),
            (Acc::Values(a), Acc::Values(b)) => a.extend_from_slice(b),
            (Acc::Pick { row, value }, Acc::Pick { row: rb, value: vb }) => {
                if *rb != usize::MAX {
                    let better = *row == usize::MAX
                        || if agg == Agg::First {
                            rb < row
                        } else {
                            rb > row
                        };
                    if better {
                        *row = *rb;
                        *value = *vb;
                    }
                }
            }
            (Acc::Weighted { sw, swx }, Acc::Weighted { sw: sw2, swx: swx2 }) => {
                *sw += sw2;
                *swx += swx2;
            }
            _ => unreachable!("accumulators of one measure always have the same kind"),
        }
    }

    /// The finished number, or `None` when there is nothing to report (no values, one value for a variance).
    pub fn finish(&self, agg: Agg) -> Option<f64> {
        match (self, agg) {
            (Acc::Count(n), _) => Some(*n as f64),
            (Acc::Sum { n, sum }, _) => (*n > 0).then_some(*sum),
            (Acc::Moments { n, mean, .. }, Agg::Mean) => (*n > 0).then_some(*mean),
            (Acc::Moments { n, min, .. }, Agg::Min) => (*n > 0).then_some(*min),
            (Acc::Moments { n, max, .. }, Agg::Max) => (*n > 0).then_some(*max),
            (Acc::Moments { n, m2, .. }, Agg::Var) => (*n > 1).then(|| m2 / (*n - 1) as f64),
            (Acc::Moments { n, m2, .. }, Agg::Std) => {
                (*n > 1).then(|| (m2 / (*n - 1) as f64).sqrt())
            }
            (Acc::Distinct(s), _) => Some(s.len() as f64),
            (Acc::Values(v), Agg::Median) => {
                if v.is_empty() {
                    return None;
                }
                let mut s = v.clone();
                let mid = s.len() / 2;
                s.select_nth_unstable_by(mid, f64::total_cmp);
                if s.len() % 2 == 1 {
                    Some(s[mid])
                } else {
                    let lower = s[..mid].iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    Some((lower + s[mid]) / 2.0)
                }
            }
            (Acc::Values(v), Agg::Mode) => {
                if v.is_empty() {
                    return None;
                }
                let mut s = v.clone();
                s.sort_by(f64::total_cmp);
                let (mut best, mut best_n, mut run_start) = (s[0], 0usize, 0usize);
                for i in 1..=s.len() {
                    if i == s.len() || s[i] != s[run_start] {
                        if i - run_start > best_n {
                            best = s[run_start];
                            best_n = i - run_start;
                        }
                        run_start = i;
                    }
                }
                Some(best)
            }
            (Acc::Pick { row, value }, _) => (*row != usize::MAX).then_some(*value),
            (Acc::Weighted { sw, swx }, _) => (*sw != 0.0).then(|| swx / sw),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(agg: Agg, xs: &[f64]) -> Option<f64> {
        let mut a = Acc::new(agg);
        for (i, &x) in xs.iter().enumerate() {
            a.update(agg, i, Some(x), Some(x.to_bits()), Some(1.0));
        }
        a.finish(agg)
    }

    #[test]
    fn simple_aggregations() {
        let xs = [4.0, 1.0, 7.0, 4.0];
        assert_eq!(run(Agg::Sum, &xs), Some(16.0));
        assert_eq!(run(Agg::Count, &xs), Some(4.0));
        assert_eq!(run(Agg::Mean, &xs), Some(4.0));
        assert_eq!(run(Agg::Min, &xs), Some(1.0));
        assert_eq!(run(Agg::Max, &xs), Some(7.0));
        assert_eq!(run(Agg::Distinct, &xs), Some(3.0));
        assert_eq!(run(Agg::Median, &xs), Some(4.0));
        assert_eq!(run(Agg::Mode, &xs), Some(4.0));
        assert_eq!(run(Agg::First, &xs), Some(4.0));
        assert_eq!(run(Agg::Last, &xs), Some(4.0));
    }

    #[test]
    fn variance_is_the_sample_variance() {
        let xs = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((run(Agg::Var, &xs).unwrap() - 32.0 / 7.0).abs() < 1e-12);
        assert!((run(Agg::Std, &xs).unwrap() - (32.0f64 / 7.0).sqrt()).abs() < 1e-12);
        assert_eq!(run(Agg::Std, &[3.0]), None);
        assert_eq!(run(Agg::Var, &[]), None);
    }

    #[test]
    fn median_of_even_and_odd_counts_and_mode_ties() {
        assert_eq!(run(Agg::Median, &[1.0, 2.0, 3.0, 10.0]), Some(2.5));
        assert_eq!(run(Agg::Median, &[9.0, 1.0, 5.0]), Some(5.0));
        assert_eq!(run(Agg::Median, &[]), None);
        assert_eq!(run(Agg::Mode, &[3.0, 1.0, 3.0, 1.0, 2.0]), Some(1.0)); // a tie: the smaller value
    }

    #[test]
    fn first_and_last_go_by_row_number_not_arrival_order() {
        let mut a = Acc::new(Agg::First);
        a.update(Agg::First, 5, Some(50.0), None, None);
        a.update(Agg::First, 2, Some(20.0), None, None);
        a.update(Agg::First, 9, Some(90.0), None, None);
        assert_eq!(a.finish(Agg::First), Some(20.0));
        let mut b = Acc::new(Agg::Last);
        b.update(Agg::Last, 5, Some(50.0), None, None);
        b.update(Agg::Last, 9, Some(90.0), None, None);
        b.update(Agg::Last, 2, Some(20.0), None, None);
        assert_eq!(b.finish(Agg::Last), Some(90.0));
    }

    #[test]
    fn missing_values_are_skipped_and_an_empty_cell_is_none() {
        let mut a = Acc::new(Agg::Sum);
        a.update(Agg::Sum, 0, None, None, None);
        assert_eq!(a.finish(Agg::Sum), None);
        let mut c = Acc::new(Agg::Count);
        c.update(Agg::Count, 0, None, None, None);
        c.update(Agg::Count, 1, Some(1.0), None, None);
        assert_eq!(c.finish(Agg::Count), Some(1.0));
    }

    #[test]
    fn weighted_mean() {
        let mut a = Acc::new(Agg::WeightedMean);
        a.update(Agg::WeightedMean, 0, Some(10.0), None, Some(1.0));
        a.update(Agg::WeightedMean, 1, Some(20.0), None, Some(3.0));
        a.update(Agg::WeightedMean, 2, Some(99.0), None, None); // no weight: ignored
        assert_eq!(a.finish(Agg::WeightedMean), Some(17.5));
        assert_eq!(Acc::new(Agg::WeightedMean).finish(Agg::WeightedMean), None);
    }

    #[test]
    fn merging_two_halves_equals_one_pass() {
        let xs: Vec<f64> = (0..50)
            .map(|i| ((i * 37 % 23) as f64) * 1.7 - 5.0)
            .collect();
        for agg in [
            Agg::Sum,
            Agg::Count,
            Agg::Mean,
            Agg::Min,
            Agg::Max,
            Agg::Std,
            Agg::Var,
            Agg::Distinct,
            Agg::Median,
            Agg::Mode,
            Agg::First,
            Agg::Last,
        ] {
            let whole = run(agg, &xs);
            let mut left = Acc::new(agg);
            let mut right = Acc::new(agg);
            for (i, &x) in xs.iter().enumerate() {
                let target = if i % 3 == 0 { &mut left } else { &mut right };
                target.update(agg, i, Some(x), Some(x.to_bits()), Some(1.0));
            }
            left.merge(agg, &right);
            let merged = left.finish(agg);
            match (whole, merged) {
                (Some(a), Some(b)) => assert!(
                    (a - b).abs() < 1e-9 * a.abs().max(1.0),
                    "{agg:?}: {a} vs {b}"
                ),
                (a, b) => assert_eq!(a, b, "{agg:?}"),
            }
        }
    }

    #[test]
    fn merging_with_an_empty_accumulator_changes_nothing() {
        let mut a = Acc::new(Agg::Std);
        for (i, x) in [1.0, 2.0, 4.0].into_iter().enumerate() {
            a.update(Agg::Std, i, Some(x), None, None);
        }
        let before = a.finish(Agg::Std);
        a.merge(Agg::Std, &Acc::new(Agg::Std));
        assert_eq!(a.finish(Agg::Std), before);
        let mut empty = Acc::new(Agg::Std);
        empty.merge(Agg::Std, &a);
        assert_eq!(empty.finish(Agg::Std), before);
    }
}

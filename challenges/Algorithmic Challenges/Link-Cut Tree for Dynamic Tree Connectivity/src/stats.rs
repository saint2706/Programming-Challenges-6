//! The aggregates every structure maintains.
//!
//! [`Stats`] (count, sum, min, max) answers path and range queries. `count` is what lets a vertex
//! be *neutral* (hold no value), which the connectivity application uses for its vertex nodes.
//! [`Sum`] is the invertible half (count, sum): the only aggregate a link-cut tree can keep over a
//! subtree, because its virtual-subtree bookkeeping needs subtraction.

/// Aggregate of a set of values. `Stats::NEUTRAL` is the aggregate of the empty set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stats {
    pub count: i64,
    pub sum: i64,
    pub min: i64,
    pub max: i64,
}

impl Stats {
    pub const NEUTRAL: Stats = Stats {
        count: 0,
        sum: 0,
        min: i64::MAX,
        max: i64::MIN,
    };

    #[inline]
    pub fn single(value: i64) -> Stats {
        Stats {
            count: 1,
            sum: value,
            min: value,
            max: value,
        }
    }

    #[inline]
    pub fn from_option(value: Option<i64>) -> Stats {
        value.map_or(Stats::NEUTRAL, Stats::single)
    }

    /// Aggregate of two disjoint sets. Commutative and associative, which is what lets a
    /// link-cut tree reverse a path without touching any aggregate.
    #[inline]
    pub fn combine(self, other: Stats) -> Stats {
        Stats {
            count: self.count + other.count,
            sum: self.sum + other.sum,
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Add `delta` to every value in the set. A neutral aggregate is returned unchanged, so the
    /// `i64::MAX` / `i64::MIN` sentinels never overflow.
    #[inline]
    pub fn shift(self, delta: i64) -> Stats {
        if self.count == 0 {
            self
        } else {
            Stats {
                count: self.count,
                sum: self.sum + delta * self.count,
                min: self.min + delta,
                max: self.max + delta,
            }
        }
    }

    /// The invertible part of the aggregate.
    #[inline]
    pub fn to_sum(self) -> Sum {
        Sum {
            count: self.count,
            sum: self.sum,
        }
    }
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.count == 0 {
            write!(f, "empty")
        } else {
            write!(
                f,
                "sum={} min={} max={} count={}",
                self.sum, self.min, self.max, self.count
            )
        }
    }
}

/// Count and sum of a set of values: an aggregate with an inverse.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sum {
    pub count: i64,
    pub sum: i64,
}

impl std::fmt::Display for Sum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sum={} count={}", self.sum, self.count)
    }
}

impl std::ops::Add for Sum {
    type Output = Sum;

    #[inline]
    fn add(self, o: Sum) -> Sum {
        Sum {
            count: self.count + o.count,
            sum: self.sum + o.sum,
        }
    }
}

impl std::ops::Sub for Sum {
    type Output = Sum;

    #[inline]
    fn sub(self, o: Sum) -> Sum {
        Sum {
            count: self.count - o.count,
            sum: self.sum - o.sum,
        }
    }
}

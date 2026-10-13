//! The aggregate every structure maintains: sum, min, max and a count.
//!
//! `count` is what lets a vertex be *neutral* (it holds no value). An edge-weighted tree becomes a
//! vertex-weighted one by subdividing every edge: edge vertices carry the weight, original vertices
//! are neutral, and a range-add leaves neutral vertices untouched because it scales by `count`.

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

    /// Aggregate of two disjoint sets. Commutative and associative.
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

    /// The single value if this aggregate holds exactly one.
    pub fn value(self) -> Option<i64> {
        (self.count == 1).then_some(self.sum)
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

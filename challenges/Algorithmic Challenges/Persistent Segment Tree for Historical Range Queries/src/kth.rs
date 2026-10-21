//! Historical range queries over an array: the k-th smallest value in a subarray, how many values
//! in it are below `x`, and how many distinct values it holds. All three answer in O(log n) after
//! an O(n log n)-node build, by keeping one persistent tree per prefix.
//!
//! **k-th smallest.** Compress the values to ranks `0..sigma` and let version `i` be a count tree
//! over ranks holding `a[0..i]`. The multiset of `a[l..r]` is version `r` minus version `l`, so one
//! simultaneous descent of two roots finds its k-th smallest (`Persistent::kth_between`).
//!
//! **Distinct count.** Let version `i` mark, over *positions*, the last occurrence of each value
//! in `a[0..i]`. The distinct values of `a[l..r]` are the marks at positions `>= l` in version `r`.
//! Adding `a[i]` unmarks its previous occurrence and marks `i`: two updates, so the first one's
//! version is retired at once and its nodes recycled (counting is on for this index).

use crate::arena::Version;
use crate::persistent::{Persistent, SumU32};

/// Range k-th smallest and range rank queries on a fixed array.
pub struct KthIndex {
    tree: Persistent<SumU32>,
    /// Distinct values, ascending; a value's rank is its index here.
    sorted: Vec<i64>,
    /// `prefix[i]` is the version holding `a[0..i]`.
    prefix: Vec<Version>,
    len: usize,
}

impl KthIndex {
    pub fn new(a: &[i64]) -> Self {
        let mut sorted: Vec<i64> = a.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        let mut tree = Persistent::<SumU32>::new(sorted.len().max(1), false);
        tree.reserve_updates(a.len());
        let mut prefix = Vec::with_capacity(a.len() + 1);
        prefix.push(tree.first());
        for &x in a {
            let rank = sorted
                .binary_search(&x)
                .expect("value is in its own sorted copy");
            let from = *prefix.last().expect("seeded with version 0");
            prefix.push(tree.modify(from, rank, |c| c + 1));
        }
        Self {
            tree,
            sorted,
            prefix,
            len: a.len(),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The `k`-th smallest (0-based) of `a[l..r]`, or `None` if `k >= r - l`.
    pub fn kth_smallest(&self, l: usize, r: usize, k: usize) -> Option<i64> {
        assert!(
            l <= r && r <= self.len,
            "range {l}..{r} out of bounds for {} elements",
            self.len
        );
        let rank = self
            .tree
            .kth_between(self.prefix[r], self.prefix[l], u32::try_from(k).ok()?)?;
        Some(self.sorted[rank])
    }

    /// How many elements of `a[l..r]` are strictly below `x`.
    pub fn count_below(&self, l: usize, r: usize, x: i64) -> usize {
        assert!(
            l <= r && r <= self.len,
            "range {l}..{r} out of bounds for {} elements",
            self.len
        );
        let rank = self.sorted.partition_point(|&v| v < x);
        self.tree
            .count_below_between(self.prefix[r], self.prefix[l], rank) as usize
    }

    /// Nodes the index holds, the figure its memory is quoted from.
    pub fn nodes(&self) -> usize {
        self.tree.live_nodes()
    }

    pub fn bytes(&self) -> usize {
        self.tree.stats().bytes
            + self.sorted.capacity() * size_of::<i64>()
            + self.prefix.capacity() * size_of::<Version>()
    }

    /// Distinct values in the array, the `sigma` of the count tree.
    pub fn distinct_values(&self) -> usize {
        self.sorted.len()
    }
}

/// Number of distinct values in a subarray, online.
pub struct DistinctIndex {
    tree: Persistent<SumU32>,
    /// `prefix[i]` is the version marking last occurrences within `a[0..i]`.
    prefix: Vec<Version>,
    len: usize,
}

impl DistinctIndex {
    pub fn new(a: &[i64]) -> Self {
        let mut sorted: Vec<i64> = a.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        let mut tree = Persistent::<SumU32>::new(a.len().max(1), true);
        // Two updates per element (unmark, mark); the first one's nodes are mostly recycled.
        tree.reserve_updates(a.len() + a.len() / 2);
        let mut last: Vec<Option<usize>> = vec![None; sorted.len()];
        let mut prefix = Vec::with_capacity(a.len() + 1);
        prefix.push(tree.first());
        for (i, &x) in a.iter().enumerate() {
            let rank = sorted
                .binary_search(&x)
                .expect("value is in its own sorted copy");
            let mut v = *prefix.last().expect("seeded with version 0");
            if let Some(p) = last[rank] {
                let unmarked = tree.modify(v, p, |c| c - 1);
                v = tree.modify(unmarked, i, |c| c + 1);
                tree.retire(unmarked);
            } else {
                v = tree.modify(v, i, |c| c + 1);
            }
            last[rank] = Some(i);
            prefix.push(v);
        }
        Self {
            tree,
            prefix,
            len: a.len(),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Distinct values in `a[l..r]`.
    pub fn distinct(&self, l: usize, r: usize) -> usize {
        assert!(
            l <= r && r <= self.len,
            "range {l}..{r} out of bounds for {} elements",
            self.len
        );
        if self.len == 0 {
            return 0;
        }
        self.tree.query(self.prefix[r], l, r) as usize
    }

    pub fn nodes(&self) -> usize {
        self.tree.live_nodes()
    }
}

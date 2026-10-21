//! The fat-node method (Driscoll, Sarnak, Sleator and Tarjan 1989) applied to a segment tree.
//!
//! Instead of copying a path, each node keeps a history: a list of `(version, aggregate)` pairs,
//! appended to when an update changes the node and never edited. A query for version `v` reads, at
//! each node it visits, the last entry whose version is `<= v`, found by binary search. The tree's
//! shape never changes, so there are no pointers to redirect.
//!
//! It is only *partially* persistent: updates apply to the latest version, so versions form one
//! timeline and cannot branch (path copying branches for free). And it saves nothing here. In a
//! structure whose nodes hold aggregates, an update changes the aggregate of every ancestor of the
//! leaf, so `log n + 1` nodes gain an entry, the same count path copying allocates, while a query
//! pays an extra binary search per node. Fat nodes win when an update touches O(1) nodes; this is
//! the case that shows what they cost when it does not.

/// A segment tree over `i64` sums with a version history per node. Positions are padded up to a
/// power of two; padding is zero.
pub struct FatTree {
    n: usize,
    size: usize,
    /// Heap layout: node `i` has children `2i` and `2i + 1`; leaves are `size..2 * size`.
    history: Vec<Vec<(u32, i64)>>,
    version: u32,
}

impl FatTree {
    pub fn new(values: &[i64]) -> Self {
        assert!(!values.is_empty());
        let size = values.len().next_power_of_two();
        let mut agg = vec![0i64; 2 * size];
        agg[size..size + values.len()].copy_from_slice(values);
        for i in (1..size).rev() {
            agg[i] = agg[2 * i] + agg[2 * i + 1];
        }
        let history = agg.into_iter().map(|a| vec![(0u32, a)]).collect();
        Self {
            n: values.len(),
            size,
            history,
            version: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// The latest version number (versions are `0..=latest`).
    pub fn latest(&self) -> u32 {
        self.version
    }

    fn current(&self, node: usize) -> i64 {
        self.history[node]
            .last()
            .expect("every node has an initial entry")
            .1
    }

    /// Sets `pos` to `value` in a new latest version and returns its number.
    pub fn update(&mut self, pos: usize, value: i64) -> u32 {
        assert!(pos < self.n);
        self.version += 1;
        let mut node = self.size + pos;
        self.history[node].push((self.version, value));
        node /= 2;
        while node >= 1 {
            let sum = self.current(2 * node) + self.current(2 * node + 1);
            self.history[node].push((self.version, sum));
            node /= 2;
        }
        self.version
    }

    fn at(&self, node: usize, version: u32) -> i64 {
        let h = &self.history[node];
        let idx = h.partition_point(|&(v, _)| v <= version);
        h[idx - 1].1
    }

    /// Sum over `[l, r)` as of `version`.
    pub fn query(&self, version: u32, l: usize, r: usize) -> i64 {
        assert!(l <= r && r <= self.n && version <= self.version);
        if l == r {
            return 0;
        }
        self.query_rec(1, 0, self.size, l, r, version)
    }

    fn query_rec(
        &self,
        node: usize,
        lo: usize,
        hi: usize,
        l: usize,
        r: usize,
        version: u32,
    ) -> i64 {
        if l <= lo && hi <= r {
            return self.at(node, version);
        }
        let mid = lo + (hi - lo) / 2;
        let left = if l < mid {
            self.query_rec(2 * node, lo, mid, l, r, version)
        } else {
            0
        };
        let right = if r > mid {
            self.query_rec(2 * node + 1, mid, hi, l, r, version)
        } else {
            0
        };
        left + right
    }

    /// Entries stored across all nodes, one per (node, change).
    pub fn entries(&self) -> usize {
        self.history.iter().map(Vec::len).sum()
    }

    /// Heap bytes: the entry vectors (capacity) plus the vector of vectors.
    pub fn bytes(&self) -> usize {
        self.history.capacity() * size_of::<Vec<(u32, i64)>>()
            + self
                .history
                .iter()
                .map(|h| h.capacity() * size_of::<(u32, i64)>())
                .sum::<usize>()
    }
}

//! Textbook heavy-light decomposition with one lazy segment tree over the heavy-first preorder
//! (cp-algorithms; Al.Cash, "Efficient and easy segment trees" HLD template). A path crosses
//! O(log n) chains and each costs an O(log n) segment-tree range, so path operations are
//! O(log² n). A subtree is one contiguous range, so subtree queries are O(log n).

use crate::tree::Layout;
use crate::{PathOps, Stats, SubtreeOps};

/// Segment tree over `0..n` with range add and range aggregate.
struct LazySegmentTree {
    n: usize,
    agg: Vec<Stats>,
    tag: Vec<i64>,
}

impl LazySegmentTree {
    fn new(values: &[Stats]) -> Self {
        let n = values.len();
        let mut tree = LazySegmentTree {
            n,
            agg: vec![Stats::NEUTRAL; 4 * n],
            tag: vec![0; 4 * n],
        };
        tree.build(1, 0, n - 1, values);
        tree
    }

    fn build(&mut self, node: usize, lo: usize, hi: usize, values: &[Stats]) {
        if lo == hi {
            self.agg[node] = values[lo];
            return;
        }
        let mid = (lo + hi) / 2;
        self.build(2 * node, lo, mid, values);
        self.build(2 * node + 1, mid + 1, hi, values);
        self.agg[node] = self.agg[2 * node].combine(self.agg[2 * node + 1]);
    }

    fn apply(&mut self, node: usize, delta: i64) {
        self.agg[node] = self.agg[node].shift(delta);
        self.tag[node] += delta;
    }

    fn push(&mut self, node: usize) {
        let delta = std::mem::take(&mut self.tag[node]);
        if delta != 0 {
            self.apply(2 * node, delta);
            self.apply(2 * node + 1, delta);
        }
    }

    fn add(&mut self, node: usize, lo: usize, hi: usize, a: usize, b: usize, delta: i64) {
        if b < lo || hi < a {
            return;
        }
        if a <= lo && hi <= b {
            self.apply(node, delta);
            return;
        }
        self.push(node);
        let mid = (lo + hi) / 2;
        self.add(2 * node, lo, mid, a, b, delta);
        self.add(2 * node + 1, mid + 1, hi, a, b, delta);
        self.agg[node] = self.agg[2 * node].combine(self.agg[2 * node + 1]);
    }

    fn query(&mut self, node: usize, lo: usize, hi: usize, a: usize, b: usize) -> Stats {
        if b < lo || hi < a {
            return Stats::NEUTRAL;
        }
        if a <= lo && hi <= b {
            return self.agg[node];
        }
        self.push(node);
        let mid = (lo + hi) / 2;
        self.query(2 * node, lo, mid, a, b)
            .combine(self.query(2 * node + 1, mid + 1, hi, a, b))
    }

    fn assign(&mut self, node: usize, lo: usize, hi: usize, i: usize, value: Stats) {
        if lo == hi {
            self.agg[node] = value;
            return;
        }
        self.push(node);
        let mid = (lo + hi) / 2;
        if i <= mid {
            self.assign(2 * node, lo, mid, i, value);
        } else {
            self.assign(2 * node + 1, mid + 1, hi, i, value);
        }
        self.agg[node] = self.agg[2 * node].combine(self.agg[2 * node + 1]);
    }
}

pub struct ClassicHld {
    layout: Layout,
    seg: LazySegmentTree,
}

impl ClassicHld {
    fn range_add(&mut self, a: usize, b: usize, delta: i64) {
        self.seg.add(1, 0, self.seg.n - 1, a, b, delta);
    }

    fn range_query(&mut self, a: usize, b: usize) -> Stats {
        self.seg.query(1, 0, self.seg.n - 1, a, b)
    }

    /// The path `u..=v` as inclusive position ranges, one per chain segment.
    fn path_ranges(&self, mut u: usize, mut v: usize) -> Vec<(usize, usize)> {
        let l = &self.layout;
        let mut ranges = Vec::new();
        while l.head[u] != l.head[v] {
            let (hu, hv) = (l.head[u] as usize, l.head[v] as usize);
            if l.depth[hu] > l.depth[hv] {
                ranges.push((l.pos[hu] as usize, l.pos[u] as usize));
                u = l.parent[hu] as usize;
            } else {
                ranges.push((l.pos[hv] as usize, l.pos[v] as usize));
                v = l.parent[hv] as usize;
            }
        }
        let (a, b) = (l.pos[u] as usize, l.pos[v] as usize);
        ranges.push((a.min(b), a.max(b)));
        ranges
    }
}

impl PathOps for ClassicHld {
    fn build(layout: Layout, values: &[Option<i64>]) -> Self {
        assert_eq!(values.len(), layout.len());
        let by_pos: Vec<Stats> = layout
            .order
            .iter()
            .map(|&v| Stats::from_option(values[v as usize]))
            .collect();
        ClassicHld {
            seg: LazySegmentTree::new(&by_pos),
            layout,
        }
    }

    fn len(&self) -> usize {
        self.layout.len()
    }

    fn path_query(&mut self, u: usize, v: usize) -> Stats {
        let mut result = Stats::NEUTRAL;
        for (a, b) in self.path_ranges(u, v) {
            result = result.combine(self.range_query(a, b));
        }
        result
    }

    fn path_add(&mut self, u: usize, v: usize, delta: i64) {
        for (a, b) in self.path_ranges(u, v) {
            self.range_add(a, b, delta);
        }
    }

    fn set(&mut self, v: usize, value: i64) {
        let p = self.layout.pos[v] as usize;
        self.seg
            .assign(1, 0, self.seg.n - 1, p, Stats::single(value));
    }

    fn get(&mut self, v: usize) -> Option<i64> {
        let p = self.layout.pos[v] as usize;
        self.range_query(p, p).value()
    }

    fn lca(&self, u: usize, v: usize) -> usize {
        self.layout.lca(u, v)
    }

    fn kth_ancestor(&self, v: usize, k: usize) -> Option<usize> {
        self.layout.kth_ancestor(v, k)
    }
}

impl SubtreeOps for ClassicHld {
    fn subtree_query(&mut self, v: usize) -> Stats {
        let start = self.layout.pos[v] as usize;
        self.range_query(start, start + self.layout.size[v] as usize - 1)
    }
}

impl crate::MaybeSubtree for ClassicHld {
    fn maybe_subtree(&mut self, v: usize) -> Option<Stats> {
        Some(self.subtree_query(v))
    }
}

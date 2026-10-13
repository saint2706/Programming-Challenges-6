//! The oracle: walk parent pointers. O(depth) per path operation, O(n) per subtree query.
//! Everything else is checked against this.

use crate::tree::{Layout, NONE};
use crate::{PathOps, Stats, SubtreeOps};

pub struct NaiveTree {
    layout: Layout,
    val: Vec<Stats>,
}

impl NaiveTree {
    /// Visit every vertex of the path `u..=v` by climbing the deeper endpoint (ties: `u`).
    fn walk(&self, mut u: usize, mut v: usize, mut visit: impl FnMut(usize)) {
        while u != v {
            if self.layout.depth[u] >= self.layout.depth[v] {
                visit(u);
                u = self.layout.parent[u] as usize;
            } else {
                visit(v);
                v = self.layout.parent[v] as usize;
            }
        }
        visit(u);
    }
}

impl PathOps for NaiveTree {
    fn build(layout: Layout, values: &[Option<i64>]) -> Self {
        assert_eq!(values.len(), layout.len());
        NaiveTree {
            val: values.iter().map(|&v| Stats::from_option(v)).collect(),
            layout,
        }
    }

    fn len(&self) -> usize {
        self.val.len()
    }

    fn path_query(&mut self, u: usize, v: usize) -> Stats {
        let mut result = Stats::NEUTRAL;
        let val = &self.val;
        self.walk(u, v, |x| result = result.combine(val[x]));
        result
    }

    fn path_add(&mut self, u: usize, v: usize, delta: i64) {
        let mut touched = Vec::new();
        self.walk(u, v, |x| touched.push(x));
        for x in touched {
            self.val[x] = self.val[x].shift(delta);
        }
    }

    fn set(&mut self, v: usize, value: i64) {
        self.val[v] = Stats::single(value);
    }

    fn get(&mut self, v: usize) -> Option<i64> {
        self.val[v].value()
    }

    /// Independent of the chain hopping in `Layout::lca`: lift to equal depth, then climb together.
    fn lca(&self, mut u: usize, mut v: usize) -> usize {
        let depth = &self.layout.depth;
        let parent = &self.layout.parent;
        while depth[u] > depth[v] {
            u = parent[u] as usize;
        }
        while depth[v] > depth[u] {
            v = parent[v] as usize;
        }
        while u != v {
            u = parent[u] as usize;
            v = parent[v] as usize;
        }
        u
    }

    fn kth_ancestor(&self, mut v: usize, k: usize) -> Option<usize> {
        for _ in 0..k {
            let p = self.layout.parent[v];
            if p == NONE {
                return None;
            }
            v = p as usize;
        }
        Some(v)
    }
}

impl SubtreeOps for NaiveTree {
    fn subtree_query(&mut self, v: usize) -> Stats {
        let (start, len) = (self.layout.pos[v] as usize, self.layout.size[v] as usize);
        self.layout.order[start..start + len]
            .iter()
            .fold(Stats::NEUTRAL, |acc, &w| acc.combine(self.val[w as usize]))
    }
}

impl crate::MaybeSubtree for NaiveTree {
    fn maybe_subtree(&mut self, v: usize) -> Option<Stats> {
        Some(self.subtree_query(v))
    }
}

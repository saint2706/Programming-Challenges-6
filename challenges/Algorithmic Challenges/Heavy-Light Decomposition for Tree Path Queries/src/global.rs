//! Globally balanced heavy-light decomposition and the static top tree built on it.
//!
//! Design sources (ported, not copied):
//!
//! * *Global biased tree* (myee, <https://www.cnblogs.com/myee/p/global-biased-tree.html>): each
//!   heavy chain becomes a static binary search tree whose root is the weighted centroid, with the
//!   weight of a vertex `v` being `size[v] - size[heavy[v]]` (one plus its light subtrees). A light
//!   edge is a "path-parent" pointer, as in a link-cut tree, so a walk from any vertex to the root
//!   crosses O(log n) nodes in total: a step down a light edge halves the subtree size, a step down
//!   a chain tree halves the weight.
//! * *Static top tree* (AtCoder ABC351-G editorial, <https://atcoder.jp/contests/abc351/editorial/9899>):
//!   the light children of a vertex are combined by a *rake* tree balanced by subtree size, so that
//!   changing one light child costs O(log(size(v) / size(child))) and the sum along a root path
//!   telescopes to O(log n). Without it a light-child heap or multiset costs O(log degree) per light
//!   edge, which is O(log² n) in the worst case.
//!
//! Chain-tree node ids are vertex ids (each vertex is exactly one node). A chain is a rake item of
//! its parent vertex, so rake nodes are indexed by chain id. The const parameter `SUB` switches the
//! rake trees and light aggregates on: [`BalancedHld`] is `SUB = false` (path operations only),
//! [`TopTree`] is `SUB = true` (adds subtree aggregates).
//!
//! Every aggregate here is commutative, so chain order never matters for a query, only which
//! vertices are covered.

use crate::tree::{Layout, NONE};
use crate::{PathOps, Stats, SubtreeOps};

/// Path operations in O(log n) by balancing every heavy chain globally.
pub type BalancedHld = GlobalTree<false>;
/// [`BalancedHld`] plus subtree aggregates through balanced rake trees of light children.
pub type TopTree = GlobalTree<true>;

pub struct GlobalTree<const SUB: bool> {
    layout: Layout,
    // Chain tree, indexed by vertex.
    left: Vec<u32>,
    right: Vec<u32>,
    up: Vec<u32>,
    /// Chain positions covered by the node's subtree, inclusive.
    cover_lo: Vec<u32>,
    cover_hi: Vec<u32>,
    val: Vec<Stats>,
    /// Aggregate of `val` over the node's chain-tree subtree.
    own: Vec<Stats>,
    /// Pending add for the node's children.
    tag: Vec<i64>,
    chain_root: Vec<u32>,
    // Light subtrees (SUB only). `light_agg[v]` is the aggregate of everything hanging off `v`
    // by a light edge; `light_sub[x]` sums that over x's chain-tree subtree.
    light_agg: Vec<Stats>,
    light_sub: Vec<Stats>,
    // Rake tree, indexed by chain id (SUB only).
    rake_left: Vec<u32>,
    rake_right: Vec<u32>,
    rake_up: Vec<u32>,
    rake_agg: Vec<Stats>,
    /// `touched[c] == epoch` marks a chain whose refresh is pending in the current `path_add`.
    touched: Vec<u64>,
    epoch: u64,
    scratch: Vec<u32>,
}

/// Index of the weighted centroid of `lo..hi`: the element whose removal leaves the more balanced
/// pair of sides (left and right weight both at most half, when one exists).
fn weighted_centroid(prefix: &[u64], lo: usize, hi: usize) -> usize {
    let total = prefix[hi] - prefix[lo];
    let target = prefix[lo] + total.div_ceil(2);
    let first = lo + prefix[lo + 1..=hi].partition_point(|&p| p < target);
    let worst = |x: usize| (prefix[x] - prefix[lo]).max(prefix[hi] - prefix[x + 1]);
    if first > lo && worst(first - 1) < worst(first) {
        first - 1
    } else {
        first
    }
}

impl<const SUB: bool> GlobalTree<SUB> {
    fn node_own(&self, x: u32) -> Stats {
        if x == NONE {
            Stats::NEUTRAL
        } else {
            self.own[x as usize]
        }
    }

    fn node_light_sub(&self, x: u32) -> Stats {
        if x == NONE {
            Stats::NEUTRAL
        } else {
            self.light_sub[x as usize]
        }
    }

    /// Aggregate of a whole chain: its vertices plus everything hanging off them.
    fn chain_total(&self, chain: usize) -> Stats {
        let root = self.chain_root[chain] as usize;
        self.own[root].combine(self.light_sub[root])
    }

    fn pull(&mut self, v: usize) {
        let (l, r) = (self.left[v], self.right[v]);
        self.own[v] = self.val[v]
            .combine(self.node_own(l))
            .combine(self.node_own(r));
        if SUB {
            self.light_sub[v] = self.light_agg[v]
                .combine(self.node_light_sub(l))
                .combine(self.node_light_sub(r));
        }
    }

    fn apply(&mut self, v: usize, delta: i64) {
        self.val[v] = self.val[v].shift(delta);
        self.own[v] = self.own[v].shift(delta);
        self.tag[v] += delta;
    }

    fn push(&mut self, v: usize) {
        let delta = std::mem::take(&mut self.tag[v]);
        if delta != 0 {
            for child in [self.left[v], self.right[v]] {
                if child != NONE {
                    self.apply(child as usize, delta);
                }
            }
        }
    }

    /// Push pending adds from the chain-tree root down to `v` (inclusive).
    fn push_path(&mut self, v: usize) {
        self.scratch.clear();
        let mut x = v as u32;
        while x != NONE {
            self.scratch.push(x);
            x = self.up[x as usize];
        }
        for i in (0..self.scratch.len()).rev() {
            let x = self.scratch[i] as usize;
            self.push(x);
        }
    }

    fn recompute_up(&mut self, v: usize) {
        let mut x = v as u32;
        while x != NONE {
            self.pull(x as usize);
            x = self.up[x as usize];
        }
    }

    /// A chain's content changed: refresh its rake node, then the parent vertex, and repeat until
    /// the root chain, or until a chain marked in the current epoch, which refreshes itself later.
    /// Requires `SUB`.
    fn propagate_light(&mut self, mut chain: usize) {
        loop {
            let head = self.layout.chain_head[chain] as usize;
            let parent = self.layout.parent[head];
            if parent == NONE {
                return;
            }
            let mut k = chain as u32;
            let top = loop {
                let i = k as usize;
                self.rake_agg[i] = self
                    .chain_total(i)
                    .combine(self.rake_agg_of(self.rake_left[i]))
                    .combine(self.rake_agg_of(self.rake_right[i]));
                if self.rake_up[i] == NONE {
                    break i;
                }
                k = self.rake_up[i];
            };
            let p = parent as usize;
            self.light_agg[p] = self.rake_agg[top];
            self.push_path(p);
            self.recompute_up(p);
            chain = self.layout.chain[p] as usize;
            if self.touched[chain] == self.epoch {
                return;
            }
        }
    }

    fn rake_agg_of(&self, k: u32) -> Stats {
        if k == NONE {
            Stats::NEUTRAL
        } else {
            self.rake_agg[k as usize]
        }
    }

    /// Chain-tree range add over chain positions `a..=b`.
    fn add_range(&mut self, x: u32, a: u32, b: u32, delta: i64) {
        if x == NONE {
            return;
        }
        let xi = x as usize;
        if b < self.cover_lo[xi] || self.cover_hi[xi] < a {
            return;
        }
        if a <= self.cover_lo[xi] && self.cover_hi[xi] <= b {
            self.apply(xi, delta);
            return;
        }
        self.push(xi);
        self.add_range(self.left[xi], a, b, delta);
        let p = self.layout.chain_pos(xi);
        if a <= p && p <= b {
            self.val[xi] = self.val[xi].shift(delta);
        }
        self.add_range(self.right[xi], a, b, delta);
        self.pull(xi);
    }

    /// Aggregate of the vertices (only) at chain positions `a..=b`.
    fn query_own(&mut self, x: u32, a: u32, b: u32) -> Stats {
        if x == NONE {
            return Stats::NEUTRAL;
        }
        let xi = x as usize;
        if b < self.cover_lo[xi] || self.cover_hi[xi] < a {
            return Stats::NEUTRAL;
        }
        if a <= self.cover_lo[xi] && self.cover_hi[xi] <= b {
            return self.own[xi];
        }
        self.push(xi);
        let mut result = self.query_own(self.left[xi], a, b);
        let p = self.layout.chain_pos(xi);
        if a <= p && p <= b {
            result = result.combine(self.val[xi]);
        }
        result.combine(self.query_own(self.right[xi], a, b))
    }

    /// Aggregate of the vertices at chain positions `a..=b` *and* everything hanging off them.
    fn query_full(&mut self, x: u32, a: u32, b: u32) -> Stats {
        if x == NONE {
            return Stats::NEUTRAL;
        }
        let xi = x as usize;
        if b < self.cover_lo[xi] || self.cover_hi[xi] < a {
            return Stats::NEUTRAL;
        }
        if a <= self.cover_lo[xi] && self.cover_hi[xi] <= b {
            return self.own[xi].combine(self.light_sub[xi]);
        }
        self.push(xi);
        let mut result = self.query_full(self.left[xi], a, b);
        let p = self.layout.chain_pos(xi);
        if a <= p && p <= b {
            result = result.combine(self.val[xi]).combine(self.light_agg[xi]);
        }
        result.combine(self.query_full(self.right[xi], a, b))
    }

    /// Weighted-centroid binary tree over `verts[lo..hi]`; returns its root.
    fn build_chain(&mut self, verts: &[u32], prefix: &[u64], lo: usize, hi: usize, up: u32) -> u32 {
        if lo >= hi {
            return NONE;
        }
        let x = weighted_centroid(prefix, lo, hi);
        let v = verts[x];
        let vi = v as usize;
        self.up[vi] = up;
        self.cover_lo[vi] = lo as u32;
        self.cover_hi[vi] = (hi - 1) as u32;
        self.left[vi] = self.build_chain(verts, prefix, lo, x, v);
        self.right[vi] = self.build_chain(verts, prefix, x + 1, hi, v);
        self.pull(vi);
        v
    }

    /// Weighted-centroid rake tree over the light chains `items[lo..hi]`; returns its root chain.
    fn build_rake(&mut self, items: &[u32], prefix: &[u64], lo: usize, hi: usize, up: u32) -> u32 {
        if lo >= hi {
            return NONE;
        }
        let x = weighted_centroid(prefix, lo, hi);
        let c = items[x];
        let ci = c as usize;
        self.rake_up[ci] = up;
        self.rake_left[ci] = self.build_rake(items, prefix, lo, x, c);
        self.rake_right[ci] = self.build_rake(items, prefix, x + 1, hi, c);
        self.rake_agg[ci] = self
            .chain_total(ci)
            .combine(self.rake_agg_of(self.rake_left[ci]))
            .combine(self.rake_agg_of(self.rake_right[ci]));
        c
    }

    /// Depth of the deepest vertex in the combined structure: for each vertex, the nodes touched
    /// walking from it to the root (chain-tree edges, then rake-tree edges per light edge).
    pub fn max_depth(&self) -> usize {
        let nc = self.layout.chain_head.len();
        let n = self.layout.len();
        let mut chain_base = vec![0usize; nc];
        for c in 0..nc {
            let head = self.layout.chain_head[c] as usize;
            let parent = self.layout.parent[head];
            if parent != NONE {
                let mut rake_depth = 0;
                let mut k = c as u32;
                if SUB {
                    while self.rake_up[k as usize] != NONE {
                        rake_depth += 1;
                        k = self.rake_up[k as usize];
                    }
                }
                let p = parent as usize;
                chain_base[c] =
                    chain_base[self.layout.chain[p] as usize] + self.tree_depth(p) + 1 + rake_depth;
            }
        }
        (0..n)
            .map(|v| chain_base[self.layout.chain[v] as usize] + self.tree_depth(v))
            .max()
            .unwrap_or(0)
    }

    fn tree_depth(&self, v: usize) -> usize {
        let mut depth = 0;
        let mut x = self.up[v];
        while x != NONE {
            depth += 1;
            x = self.up[x as usize];
        }
        depth
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }
}

impl<const SUB: bool> PathOps for GlobalTree<SUB> {
    fn build(layout: Layout, values: &[Option<i64>]) -> Self {
        let n = layout.len();
        assert_eq!(values.len(), n);
        let nc = layout.chain_head.len();
        let rake_len = if SUB { nc } else { 0 };
        let sub_len = if SUB { n } else { 0 };
        let val: Vec<Stats> = values.iter().map(|&v| Stats::from_option(v)).collect();
        let mut g = GlobalTree {
            left: vec![NONE; n],
            right: vec![NONE; n],
            up: vec![NONE; n],
            cover_lo: vec![0; n],
            cover_hi: vec![0; n],
            own: vec![Stats::NEUTRAL; n],
            val,
            tag: vec![0; n],
            chain_root: vec![NONE; nc],
            light_agg: vec![Stats::NEUTRAL; sub_len],
            light_sub: vec![Stats::NEUTRAL; sub_len],
            rake_left: vec![NONE; rake_len],
            rake_right: vec![NONE; rake_len],
            rake_up: vec![NONE; rake_len],
            rake_agg: vec![Stats::NEUTRAL; rake_len],
            touched: vec![0; rake_len],
            epoch: 0,
            scratch: Vec::new(),
            layout,
        };
        // Chain ids grow in preorder, so every light child's chain has a larger id than its
        // parent's: building in reverse finishes a chain's light subtrees before the chain itself.
        for c in (0..nc).rev() {
            let head = g.layout.chain_head[c] as usize;
            let len = g.layout.chain_len[c] as usize;
            let start = g.layout.pos[head] as usize;
            let verts: Vec<u32> = g.layout.order[start..start + len].to_vec();
            if SUB {
                for &v in &verts {
                    let vi = v as usize;
                    let items: Vec<u32> = g
                        .layout
                        .light_children(vi)
                        .iter()
                        .map(|&w| g.layout.chain[w as usize])
                        .collect();
                    if items.is_empty() {
                        continue;
                    }
                    let mut prefix = vec![0u64; items.len() + 1];
                    for (i, &chain) in items.iter().enumerate() {
                        let top = g.layout.chain_head[chain as usize] as usize;
                        prefix[i + 1] = prefix[i] + u64::from(g.layout.size[top]);
                    }
                    let root = g.build_rake(&items, &prefix, 0, items.len(), NONE);
                    g.light_agg[vi] = g.rake_agg[root as usize];
                }
            }
            let mut prefix = vec![0u64; len + 1];
            for (i, &v) in verts.iter().enumerate() {
                let vi = v as usize;
                let heavy = g.layout.heavy[vi];
                let below = if heavy == NONE {
                    0
                } else {
                    g.layout.size[heavy as usize]
                };
                prefix[i + 1] = prefix[i] + u64::from(g.layout.size[vi] - below);
            }
            g.chain_root[c] = g.build_chain(&verts, &prefix, 0, len, NONE);
        }
        g
    }

    fn len(&self) -> usize {
        self.layout.len()
    }

    fn path_query(&mut self, mut u: usize, mut v: usize) -> Stats {
        let mut result = Stats::NEUTRAL;
        while self.layout.chain[u] != self.layout.chain[v] {
            let (cu, cv) = (self.layout.chain[u] as usize, self.layout.chain[v] as usize);
            let (hu, hv) = (
                self.layout.chain_head[cu] as usize,
                self.layout.chain_head[cv] as usize,
            );
            if self.layout.depth[hu] > self.layout.depth[hv] {
                let root = self.chain_root[cu];
                result = result.combine(self.query_own(root, 0, self.layout.chain_pos(u)));
                u = self.layout.parent[hu] as usize;
            } else {
                let root = self.chain_root[cv];
                result = result.combine(self.query_own(root, 0, self.layout.chain_pos(v)));
                v = self.layout.parent[hv] as usize;
            }
        }
        let (a, b) = (self.layout.chain_pos(u), self.layout.chain_pos(v));
        let root = self.chain_root[self.layout.chain[u] as usize];
        result.combine(self.query_own(root, a.min(b), a.max(b)))
    }

    fn path_add(&mut self, mut u: usize, mut v: usize, delta: i64) {
        // Each touched chain is refreshed once, after all its ranges are applied.
        let mut touched: Vec<usize> = Vec::new();
        while self.layout.chain[u] != self.layout.chain[v] {
            let (cu, cv) = (self.layout.chain[u] as usize, self.layout.chain[v] as usize);
            let (hu, hv) = (
                self.layout.chain_head[cu] as usize,
                self.layout.chain_head[cv] as usize,
            );
            if self.layout.depth[hu] > self.layout.depth[hv] {
                let root = self.chain_root[cu];
                self.add_range(root, 0, self.layout.chain_pos(u), delta);
                touched.push(cu);
                u = self.layout.parent[hu] as usize;
            } else {
                let root = self.chain_root[cv];
                self.add_range(root, 0, self.layout.chain_pos(v), delta);
                touched.push(cv);
                v = self.layout.parent[hv] as usize;
            }
        }
        let (a, b) = (self.layout.chain_pos(u), self.layout.chain_pos(v));
        let c = self.layout.chain[u] as usize;
        let root = self.chain_root[c];
        self.add_range(root, a.min(b), a.max(b), delta);
        touched.push(c);
        if SUB {
            // Deepest chains first (ids grow with depth along a root path), each stopping where the
            // next touched chain takes over, so the whole refresh is one O(log n) climb.
            self.epoch += 1;
            for &c in &touched {
                self.touched[c] = self.epoch;
            }
            touched.sort_unstable_by_key(|&c| std::cmp::Reverse(c));
            for c in touched {
                self.propagate_light(c);
            }
        }
    }

    fn set(&mut self, v: usize, value: i64) {
        self.push_path(v);
        self.val[v] = Stats::single(value);
        self.recompute_up(v);
        if SUB {
            self.epoch += 1;
            self.propagate_light(self.layout.chain[v] as usize);
        }
    }

    fn get(&mut self, v: usize) -> Option<i64> {
        self.push_path(v);
        self.val[v].value()
    }

    fn lca(&self, u: usize, v: usize) -> usize {
        self.layout.lca(u, v)
    }

    fn kth_ancestor(&self, v: usize, k: usize) -> Option<usize> {
        self.layout.kth_ancestor(v, k)
    }
}

impl crate::MaybeSubtree for GlobalTree<false> {
    fn maybe_subtree(&mut self, _v: usize) -> Option<Stats> {
        None
    }
}

impl crate::MaybeSubtree for GlobalTree<true> {
    fn maybe_subtree(&mut self, v: usize) -> Option<Stats> {
        Some(self.subtree_query(v))
    }
}

impl SubtreeOps for GlobalTree<true> {
    fn subtree_query(&mut self, v: usize) -> Stats {
        let chain = self.layout.chain[v] as usize;
        let last = self.layout.chain_len[chain] - 1;
        let root = self.chain_root[chain];
        self.query_full(root, self.layout.chain_pos(v), last)
    }
}

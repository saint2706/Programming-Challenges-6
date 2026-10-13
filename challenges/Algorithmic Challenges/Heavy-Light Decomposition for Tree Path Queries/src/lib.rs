//! Tree path queries four ways, in safe Rust.
//!
//! A static tree, vertex values, and four operations: aggregate (sum, min, max) over a path or a
//! subtree, add to every vertex of a path, and assign one vertex. All four structures answer the
//! same queries, so tests and benchmarks treat them alike:
//!
//! | Structure | path query / add | point set | subtree query |
//! |---|---|---|---|
//! | [`NaiveTree`] | O(depth) | O(1) | O(n) |
//! | [`ClassicHld`] | O(log² n) | O(log n) | O(log n) |
//! | [`BalancedHld`] | O(log n) | O(log n) | not supported |
//! | [`TopTree`] | O(log n) | O(log n) | O(log n) |
//!
//! [`ClassicHld`] is the textbook decomposition with one lazy segment tree over the heavy-first
//! preorder. [`BalancedHld`] replaces each chain's segment tree with a static binary tree built by
//! weighted centroid, so a path crosses O(log n) tree nodes in total, not O(log n) per chain (the
//! "global biased tree"). [`TopTree`] adds the light subtrees to those chain trees through balanced
//! rake trees, as in the static top tree, which gives subtree aggregates at the same bound.
//!
//! An edge-weighted tree is handled by [`EdgeTree`], which subdivides every edge.

#![forbid(unsafe_code)]

pub mod classic;
pub mod edge;
pub mod global;
pub mod naive;
pub mod script;
pub mod shapes;
pub mod stats;
pub mod tree;

pub use classic::ClassicHld;
pub use edge::EdgeTree;
pub use global::{BalancedHld, TopTree};
pub use naive::NaiveTree;
pub use stats::Stats;
pub use tree::{Layout, Tree, TreeError};

/// Operations every structure supports. Vertices are `0..len`; an out-of-range vertex panics.
///
/// Values are `Option<i64>`: `None` is a neutral vertex that contributes nothing to any aggregate
/// and ignores `path_add` (used by [`EdgeTree`]). Sums must stay within `i64`.
pub trait PathOps: Sized {
    fn build(layout: Layout, values: &[Option<i64>]) -> Self;

    /// Build with a value on every vertex.
    fn from_weights(layout: Layout, weights: &[i64]) -> Self {
        let values: Vec<Option<i64>> = weights.iter().map(|&w| Some(w)).collect();
        Self::build(layout, &values)
    }

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        false
    }

    /// Aggregate over every vertex on the path `u..=v`. `Stats::NEUTRAL` if all are neutral.
    fn path_query(&mut self, u: usize, v: usize) -> Stats;

    /// Add `delta` to every value on the path `u..=v`.
    fn path_add(&mut self, u: usize, v: usize, delta: i64);

    /// Assign `value` to vertex `v`.
    fn set(&mut self, v: usize, value: i64);

    fn get(&mut self, v: usize) -> Option<i64>;

    fn lca(&self, u: usize, v: usize) -> usize;

    /// The `k`-th ancestor of `v` (`k = 0` is `v`), or `None` above the root.
    fn kth_ancestor(&self, v: usize, k: usize) -> Option<usize>;
}

/// Structures that can also aggregate a whole subtree (rooted at the build root).
pub trait SubtreeOps: PathOps {
    fn subtree_query(&mut self, v: usize) -> Stats;
}

/// Lets generic code ask any structure for a subtree aggregate; `None` where it cannot answer
/// ([`BalancedHld`]). Implemented for every structure in this crate.
pub trait MaybeSubtree: PathOps {
    fn maybe_subtree(&mut self, v: usize) -> Option<Stats>;
}

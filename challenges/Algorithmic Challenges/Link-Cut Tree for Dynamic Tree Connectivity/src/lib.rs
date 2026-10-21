//! Dynamic forests: link, cut and connectivity under edge insertions and deletions.
//!
//! A forest of `n` vertices starts with no edges. [`Forest`] is the common core every structure
//! answers (`link`, `cut`, `connected`); [`Weighted`], [`PathOps`] and [`SubtreeOps`] add vertex
//! values, path aggregates and subtree aggregates, which only some structures can offer:
//!
//! | Structure | link / cut / connected | path aggregate + add | subtree aggregate |
//! |---|---|---|---|
//! | [`NaiveForest`] | O(n) | O(n) | O(n) |
//! | [`LinkCutTree`] (`PathLct`) | O(log n) amortized | O(log n) amortized | no |
//! | [`LinkCutTree`] (`SubtreeLct`) | O(log n) amortized | O(log n) amortized | sum and count |
//! | [`EulerTourTree`] | O(log n) expected | no | sum, count, min, max |
//! | [`UfoTree`] | O(min(log n, diameter)) | no | no |
//!
//! [`LinkCutTree`] is the splay-tree structure of Sleator and Tarjan with the usual extensions
//! (evert, lazy path add, virtual-subtree sums). [`EulerTourTree`] stores each tree as a cyclic
//! tour in a treap. [`UfoTree`] is the 2026 tree-contraction design with unbounded fan-out.
//! [`connectivity`] uses a link-cut tree to answer offline fully dynamic graph connectivity.

#![forbid(unsafe_code)]

pub mod connectivity;
pub mod ett;
pub mod lct;
pub mod naive;
pub mod script;
pub mod shapes;
pub mod stats;
pub mod ufo;

pub use ett::EulerTourTree;
pub use lct::{LinkCutTree, PathLct, SubtreeLct};
pub use naive::NaiveForest;
pub use stats::{Stats, Sum};
pub use ufo::UfoTree;

/// A forest on vertices `0..len` under edge insertion and deletion. An out-of-range vertex panics.
pub trait Forest: Sized {
    /// `n` isolated vertices.
    fn new(n: usize) -> Self;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Add edge `u - v`. Returns `false` (and changes nothing) if `u == v` or they are already
    /// connected, since the result would not be a forest.
    fn link(&mut self, u: usize, v: usize) -> bool;

    /// Remove edge `u - v`. Returns `false` (and changes nothing) if there is no such edge.
    fn cut(&mut self, u: usize, v: usize) -> bool;

    /// Whether `u` and `v` are in the same tree. Takes `&mut self` because a link-cut tree
    /// restructures itself on every query.
    fn connected(&mut self, u: usize, v: usize) -> bool;
}

/// Forests whose vertices carry an optional `i64` value. `None` is a *neutral* vertex: it counts
/// for nothing in any aggregate and ignores `path_add`.
pub trait Weighted: Forest {
    fn from_values(values: &[Option<i64>]) -> Self;

    /// Every vertex starts with the value `0`.
    fn zeroed(n: usize) -> Self {
        Self::from_values(&vec![Some(0); n])
    }

    /// Give `v` the value `value` (making it non-neutral).
    fn set(&mut self, v: usize, value: i64);

    fn get(&mut self, v: usize) -> Option<i64>;
}

/// Aggregates and lazy updates over the path between two vertices.
pub trait PathOps: Weighted {
    /// Aggregate over every vertex on the path `u..=v`; `None` if they are not connected.
    fn path_query(&mut self, u: usize, v: usize) -> Option<Stats>;

    /// Add `delta` to every value on the path `u..=v`; `false` if they are not connected.
    fn path_add(&mut self, u: usize, v: usize, delta: i64) -> bool;
}

/// Aggregates over the part of a tree on `v`'s side of an edge.
pub trait SubtreeOps: Weighted {
    /// Count and sum over the whole tree containing `v`.
    fn component(&mut self, v: usize) -> Sum;

    /// Count and sum over the subtree of `v` when its tree is rooted at `p`, where `p - v` must be
    /// an edge; `None` if it is not.
    fn subtree_query(&mut self, p: usize, v: usize) -> Option<Sum>;
}

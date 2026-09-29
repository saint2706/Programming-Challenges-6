//! Order-statistic trees in safe Rust.
//!
//! An *order-statistic* structure is an ordered multiset that answers, besides
//! insert/remove:
//!
//! * `rank(x)`  -- how many elements are strictly smaller than `x`;
//! * `select(i)` -- the `i`-th smallest element (0-based).
//!
//! The headline implementation is [`OrderStatBTree`]: an arena-allocated
//! B-tree whose every child pointer carries its subtree's element count.
//! [`AvlTree`], [`Treap`] and [`RedBlackTree`] are size-augmented binary
//! trees (CLRS ch. 14); [`SortedVec`], [`Fenwick`] and [`NaiveBTreeMap`] are
//! reference points. All implement [`OrderStatistics`].

#![forbid(unsafe_code)]

pub mod avl;
pub mod baselines;
pub mod btree;
pub mod redblack;
pub mod rng;
pub mod treap;

pub use avl::AvlTree;
pub use baselines::{Fenwick, NaiveBTreeMap, SortedVec};
pub use btree::OrderStatBTree;
pub use redblack::RedBlackTree;
pub use treap::Treap;

/// Common interface so tests and benchmarks treat every structure alike.
///
/// Duplicates are allowed (multiset). `rank` counts elements strictly less
/// than the key, so `select(rank(x)) == Some(x)` whenever `x` is present.
pub trait OrderStatistics<K> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn insert(&mut self, key: K);
    /// Remove one occurrence; `false` if absent.
    fn remove(&mut self, key: &K) -> bool;
    /// Number of stored elements strictly less than `key`.
    fn rank(&self, key: &K) -> usize;
    /// The `idx`-th smallest element (0-based), if `idx < len()`.
    fn select(&self, idx: usize) -> Option<K>;
    /// Check internal invariants (used by the test-suite).
    fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

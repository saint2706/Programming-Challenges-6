//! The path-copying persistent segment tree (Sarnak and Tarjan 1986): the structure the brief
//! asks for, and the baseline every other structure here is measured against.
//!
//! A point update copies the root-to-leaf path and shares everything else, so it allocates
//! `depth + 1` nodes (at most `ceil(log2 n) + 1`) and leaves every earlier version untouched.
//! Any version can be queried in O(log n), and any *live* version can be the parent of a new one:
//! path copying is fully persistent (versions branch), not just partially (a single timeline).
//!
//! Trees start sparse. A tree over `n` positions with nothing set is the single node [`NIL`]; each
//! update materialises one path. [`Persistent::from_values`] builds a full tree instead, in
//! `2n - 1` nodes.

use std::fmt::Debug;

use crate::arena::{NIL, Node, NodeId, Stats, Store, Version};

/// An associative combine with an identity: what a node stores about its segment.
pub trait Monoid {
    type T: Copy + PartialEq + Debug;
    fn identity() -> Self::T;
    fn combine(a: Self::T, b: Self::T) -> Self::T;
}

/// Sum of `i64`.
pub struct SumI64;
/// Sum of `u32`: counts.
pub struct SumU32;
/// Minimum of `i64`, identity `i64::MAX`.
pub struct MinI64;
/// Maximum of `i64`, identity `i64::MIN`.
pub struct MaxI64;

impl Monoid for SumI64 {
    type T = i64;
    fn identity() -> i64 {
        0
    }
    fn combine(a: i64, b: i64) -> i64 {
        a + b
    }
}

impl Monoid for SumU32 {
    type T = u32;
    fn identity() -> u32 {
        0
    }
    fn combine(a: u32, b: u32) -> u32 {
        a + b
    }
}

impl Monoid for MinI64 {
    type T = i64;
    fn identity() -> i64 {
        i64::MAX
    }
    fn combine(a: i64, b: i64) -> i64 {
        a.min(b)
    }
}

impl Monoid for MaxI64 {
    type T = i64;
    fn identity() -> i64 {
        i64::MIN
    }
    fn combine(a: i64, b: i64) -> i64 {
        a.max(b)
    }
}

pub struct Persistent<M: Monoid> {
    n: usize,
    store: Store<M::T>,
}

impl<M: Monoid> Persistent<M> {
    /// `n` positions, all the identity, as version 0. `counting` turns on reference counts so that
    /// [`Persistent::retire`] frees memory at once (see `arena.rs`).
    pub fn new(n: usize, counting: bool) -> Self {
        assert!(n >= 1, "a tree needs at least one position");
        let mut store = Store::new(M::identity(), counting);
        store.push_root(NIL);
        Self { n, store }
    }

    /// A tree holding `values`, as version 0, built bottom-up in `2n - 1` nodes.
    pub fn from_values(values: &[M::T], counting: bool) -> Self {
        assert!(!values.is_empty(), "a tree needs at least one position");
        let mut store = Store::new(M::identity(), counting);
        let root = Self::build(&mut store, values, 0, values.len());
        store.push_root(root);
        Self {
            n: values.len(),
            store,
        }
    }

    fn build(store: &mut Store<M::T>, values: &[M::T], lo: usize, hi: usize) -> NodeId {
        if hi - lo == 1 {
            return store.arena.alloc(NIL, NIL, values[lo]);
        }
        let mid = lo + (hi - lo) / 2;
        let l = Self::build(store, values, lo, mid);
        let r = Self::build(store, values, mid, hi);
        let agg = M::combine(store.arena.agg(l), store.arena.agg(r));
        store.arena.alloc(l, r, agg)
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// Room for `updates` more point updates without growing the arena: each allocates at most
    /// `ceil(log2 n) + 1` nodes. Purely a memory-footprint knob; behaviour is unchanged.
    pub fn reserve_updates(&mut self, updates: usize) {
        let depth = self.n.next_power_of_two().trailing_zeros() as usize + 1;
        self.store.reserve(updates * depth);
    }

    /// The initial version.
    pub fn first(&self) -> Version {
        Version(0)
    }

    /// The most recently created version.
    pub fn latest(&self) -> Version {
        Version((self.store.versions() - 1) as u32)
    }

    pub fn versions(&self) -> usize {
        self.store.versions()
    }

    pub fn is_live(&self, v: Version) -> bool {
        self.store.is_live(v)
    }

    /// New version: `from` with position `pos` replaced by `f(old value)`.
    pub fn modify(&mut self, from: Version, pos: usize, f: impl Fn(M::T) -> M::T) -> Version {
        assert!(
            pos < self.n,
            "position {pos} out of range for {} positions",
            self.n
        );
        let root = self.store.root(from);
        let new_root = Self::modify_rec(&mut self.store, root, 0, self.n, pos, &f);
        self.store.push_root(new_root)
    }

    fn modify_rec(
        store: &mut Store<M::T>,
        node: NodeId,
        lo: usize,
        hi: usize,
        pos: usize,
        f: &impl Fn(M::T) -> M::T,
    ) -> NodeId {
        let Node { l, r, agg } = *store.arena.get(node);
        if hi - lo == 1 {
            return store.arena.alloc(NIL, NIL, f(agg));
        }
        let mid = lo + (hi - lo) / 2;
        let (nl, nr) = if pos < mid {
            (Self::modify_rec(store, l, lo, mid, pos, f), r)
        } else {
            (l, Self::modify_rec(store, r, mid, hi, pos, f))
        };
        let agg = M::combine(store.arena.agg(nl), store.arena.agg(nr));
        store.arena.alloc(nl, nr, agg)
    }

    /// New version: `from` with position `pos` set to `value`.
    pub fn update(&mut self, from: Version, pos: usize, value: M::T) -> Version {
        self.modify(from, pos, |_| value)
    }

    /// The value at `pos` in version `v`.
    pub fn get(&self, v: Version, pos: usize) -> M::T {
        assert!(pos < self.n);
        let (mut node, mut lo, mut hi) = (self.store.root(v), 0, self.n);
        while hi - lo > 1 && node != NIL {
            let mid = lo + (hi - lo) / 2;
            let n = self.store.arena.get(node);
            if pos < mid {
                node = n.l;
                hi = mid;
            } else {
                node = n.r;
                lo = mid;
            }
        }
        self.store.arena.agg(node)
    }

    /// The combined value over positions `[l, r)` in version `v`.
    pub fn query(&self, v: Version, l: usize, r: usize) -> M::T {
        assert!(
            l <= r && r <= self.n,
            "range {l}..{r} out of bounds for {} positions",
            self.n
        );
        if l == r {
            return M::identity();
        }
        self.query_rec(self.store.root(v), 0, self.n, l, r)
    }

    fn query_rec(&self, node: NodeId, lo: usize, hi: usize, l: usize, r: usize) -> M::T {
        if node == NIL {
            return M::identity();
        }
        if l <= lo && hi <= r {
            return self.store.arena.agg(node);
        }
        let mid = lo + (hi - lo) / 2;
        let n = self.store.arena.get(node);
        let left = if l < mid {
            self.query_rec(n.l, lo, mid, l, r)
        } else {
            M::identity()
        };
        let right = if r > mid {
            self.query_rec(n.r, mid, hi, l, r)
        } else {
            M::identity()
        };
        M::combine(left, right)
    }

    /// Forgets version `v` (see [`crate::arena::Store::retire`]).
    pub fn retire(&mut self, v: Version) -> bool {
        self.store.retire(v)
    }

    /// Mark-and-compact down to what the live versions reach.
    pub fn compact(&mut self) {
        self.store.compact();
    }

    pub fn stats(&self) -> Stats {
        self.store.stats()
    }

    /// Nodes in use: the figure the README's bytes-per-update numbers come from.
    pub fn live_nodes(&self) -> usize {
        self.store.arena.live()
    }
}

impl Persistent<SumU32> {
    /// 0-based position of the `k`-th smallest element of the multiset `hi - lo`, where `hi` and
    /// `lo` are versions of a count tree and `lo` was built from a prefix of what `hi` holds.
    /// This is the classic "k-th smallest in a subarray" query: two roots, one descent, O(log n).
    pub fn kth_between(&self, hi: Version, lo: Version, mut k: u32) -> Option<usize> {
        let arena = &self.store.arena;
        let (mut a, mut b) = (self.store.root(hi), self.store.root(lo));
        if k >= arena.agg(a) - arena.agg(b) {
            return None;
        }
        let (mut low, mut high) = (0, self.n);
        while high - low > 1 {
            let mid = low + (high - low) / 2;
            let (na, nb) = (arena.get(a), arena.get(b));
            let left = arena.agg(na.l) - arena.agg(nb.l);
            if k < left {
                (a, b, high) = (na.l, nb.l, mid);
            } else {
                k -= left;
                (a, b, low) = (na.r, nb.r, mid);
            }
        }
        Some(low)
    }

    /// How many elements of `hi - lo` sit at positions below `x`.
    pub fn count_below_between(&self, hi: Version, lo: Version, x: usize) -> u32 {
        let arena = &self.store.arena;
        let (mut a, mut b) = (self.store.root(hi), self.store.root(lo));
        if x >= self.n {
            return arena.agg(a) - arena.agg(b);
        }
        let (mut low, mut high) = (0, self.n);
        let mut total = 0;
        while high - low > 1 {
            let mid = low + (high - low) / 2;
            let (na, nb) = (arena.get(a), arena.get(b));
            if x < mid {
                (a, b, high) = (na.l, nb.l, mid);
            } else {
                total += arena.agg(na.l) - arena.agg(nb.l);
                (a, b, low) = (na.r, nb.r, mid);
            }
        }
        total
    }
}

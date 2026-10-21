//! Range updates on a persistent tree, two ways.
//!
//! Lazy propagation is the standard way to make range updates O(log n): a node covered by the
//! update records a pending tag instead of touching its children. In an ordinary tree a later
//! visit *pushes* the tag down. In a persistent tree pushing is expensive, because a node cannot be
//! modified: pushing a tag from a node into its two children means copying both children.
//!
//! * [`LazyAdd`] never pushes. An add-tag stays on the node where the update put it ("mark
//!   permanence"): the node's aggregate already includes it, and a query adds up the tags of the
//!   nodes it passes through. An update copies exactly the nodes it visits. It works because adds
//!   commute, so the order tags meet in does not matter; it works for sum and for min or max
//!   (adding `d` to every element adds `d` to the minimum).
//! * [`PushDown`] pushes, folding the push into the copy of each visited node. It is the general
//!   mechanism: tags are affine maps `v -> a*v + b`, which cover both add and assign, and compose
//!   in a fixed order. Range *assign* needs it, because assign and add do not commute, so a tag
//!   left in an ancestor could override a newer update below it. An update copies the visited
//!   nodes plus, wherever a tag is pending, the sibling it has to pass the tag to.
//!
//! Queries never copy in either variant: a query carries the tags it has passed instead of
//! pushing them.

use std::marker::PhantomData;

use crate::arena::{NIL, Node, NodeId, Stats, Store, Version};

/// What a node stores about its segment, and how a uniform add changes it.
pub trait Aggregate {
    fn identity() -> i64;
    fn combine(a: i64, b: i64) -> i64;
    /// The aggregate of a non-empty segment of `len` elements after adding `d` to each of them.
    fn add(agg: i64, d: i64, len: usize) -> i64;
}

pub struct Sum;
pub struct Min;
pub struct Max;

impl Aggregate for Sum {
    fn identity() -> i64 {
        0
    }
    fn combine(a: i64, b: i64) -> i64 {
        a + b
    }
    fn add(agg: i64, d: i64, len: usize) -> i64 {
        agg + d * len as i64
    }
}

impl Aggregate for Min {
    fn identity() -> i64 {
        i64::MAX
    }
    fn combine(a: i64, b: i64) -> i64 {
        a.min(b)
    }
    fn add(agg: i64, d: i64, _len: usize) -> i64 {
        agg + d
    }
}

impl Aggregate for Max {
    fn identity() -> i64 {
        i64::MIN
    }
    fn combine(a: i64, b: i64) -> i64 {
        a.max(b)
    }
    fn add(agg: i64, d: i64, _len: usize) -> i64 {
        agg + d
    }
}

/// Node payload of [`LazyAdd`]: the aggregate of the segment including this node's own `add`, and
/// the add itself, which applies to every element below and is never pushed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lz {
    agg: i64,
    add: i64,
}

pub struct LazyAdd<G: Aggregate> {
    n: usize,
    store: Store<Lz>,
    _aggregate: PhantomData<G>,
}

impl<G: Aggregate> LazyAdd<G> {
    /// A tree over `values` as version 0.
    pub fn new(values: &[i64], counting: bool) -> Self {
        assert!(!values.is_empty());
        let mut store = Store::new(
            Lz {
                agg: G::identity(),
                add: 0,
            },
            counting,
        );
        let root = Self::build(&mut store, values, 0, values.len());
        store.push_root(root);
        Self {
            n: values.len(),
            store,
            _aggregate: PhantomData,
        }
    }

    fn build(store: &mut Store<Lz>, values: &[i64], lo: usize, hi: usize) -> NodeId {
        if hi - lo == 1 {
            return store.arena.alloc(
                NIL,
                NIL,
                Lz {
                    agg: values[lo],
                    add: 0,
                },
            );
        }
        let mid = lo + (hi - lo) / 2;
        let l = Self::build(store, values, lo, mid);
        let r = Self::build(store, values, mid, hi);
        let agg = G::combine(store.arena.agg(l).agg, store.arena.agg(r).agg);
        store.arena.alloc(l, r, Lz { agg, add: 0 })
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn first(&self) -> Version {
        Version(0)
    }

    pub fn latest(&self) -> Version {
        Version((self.store.versions() - 1) as u32)
    }

    /// New version: `from` with `d` added to every element of `[l, r)`.
    pub fn range_add(&mut self, from: Version, l: usize, r: usize, d: i64) -> Version {
        assert!(l <= r && r <= self.n);
        let root = self.store.root(from);
        let new_root = if l == r {
            root
        } else {
            Self::add_rec(&mut self.store, root, 0, self.n, l, r, d)
        };
        self.store.push_root(new_root)
    }

    fn add_rec(
        store: &mut Store<Lz>,
        node: NodeId,
        lo: usize,
        hi: usize,
        l: usize,
        r: usize,
        d: i64,
    ) -> NodeId {
        let Node { l: nl, r: nr, agg } = *store.arena.get(node);
        if l <= lo && hi <= r {
            let moved = Lz {
                agg: G::add(agg.agg, d, hi - lo),
                add: agg.add + d,
            };
            return store.arena.alloc(nl, nr, moved);
        }
        let mid = lo + (hi - lo) / 2;
        let new_l = if l < mid {
            Self::add_rec(store, nl, lo, mid, l, r, d)
        } else {
            nl
        };
        let new_r = if r > mid {
            Self::add_rec(store, nr, mid, hi, l, r, d)
        } else {
            nr
        };
        let below = G::combine(store.arena.agg(new_l).agg, store.arena.agg(new_r).agg);
        let combined = Lz {
            agg: G::add(below, agg.add, hi - lo),
            add: agg.add,
        };
        store.arena.alloc(new_l, new_r, combined)
    }

    /// The aggregate over `[l, r)` in version `v`.
    pub fn query(&self, v: Version, l: usize, r: usize) -> i64 {
        assert!(l <= r && r <= self.n);
        if l == r {
            return G::identity();
        }
        self.query_rec(self.store.root(v), 0, self.n, l, r, 0)
    }

    /// `carried` is the sum of the add-tags of the strict ancestors: it applies to every element
    /// of this node's segment.
    fn query_rec(
        &self,
        node: NodeId,
        lo: usize,
        hi: usize,
        l: usize,
        r: usize,
        carried: i64,
    ) -> i64 {
        let n = self.store.arena.get(node);
        if l <= lo && hi <= r {
            return G::add(n.agg.agg, carried, hi - lo);
        }
        let mid = lo + (hi - lo) / 2;
        let below = carried + n.agg.add;
        let left = if l < mid {
            self.query_rec(n.l, lo, mid, l, r, below)
        } else {
            G::identity()
        };
        let right = if r > mid {
            self.query_rec(n.r, mid, hi, l, r, below)
        } else {
            G::identity()
        };
        G::combine(left, right)
    }

    pub fn retire(&mut self, v: Version) -> bool {
        self.store.retire(v)
    }

    pub fn compact(&mut self) {
        self.store.compact();
    }

    pub fn stats(&self) -> Stats {
        self.store.stats()
    }

    pub fn live_nodes(&self) -> usize {
        self.store.arena.live()
    }
}

/// An affine map `v -> a * v + b`. Add is `(1, d)`, assign is `(0, x)`, and composition stays
/// affine, so one tag type covers both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tag {
    a: i64,
    b: i64,
}

impl Tag {
    const IDENTITY: Tag = Tag { a: 1, b: 0 };

    pub fn add(d: i64) -> Tag {
        Tag { a: 1, b: d }
    }

    pub fn assign(x: i64) -> Tag {
        Tag { a: 0, b: x }
    }

    /// `self` first, then `next`.
    fn then(self, next: Tag) -> Tag {
        Tag {
            a: next.a * self.a,
            b: next.a * self.b + next.b,
        }
    }

    fn apply(self, sum: i64, len: usize) -> i64 {
        self.a * sum + self.b * len as i64
    }
}

/// Node payload of [`PushDown`]: the segment sum (already including `tag`) and the tag still
/// pending for the children.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pd {
    sum: i64,
    tag: Tag,
}

/// Range add and range assign with sum queries, pushing tags down as it copies.
pub struct PushDown {
    n: usize,
    store: Store<Pd>,
}

impl PushDown {
    pub fn new(values: &[i64], counting: bool) -> Self {
        assert!(!values.is_empty());
        let mut store = Store::new(
            Pd {
                sum: 0,
                tag: Tag::IDENTITY,
            },
            counting,
        );
        let root = Self::build(&mut store, values, 0, values.len());
        store.push_root(root);
        Self {
            n: values.len(),
            store,
        }
    }

    fn build(store: &mut Store<Pd>, values: &[i64], lo: usize, hi: usize) -> NodeId {
        if hi - lo == 1 {
            return store.arena.alloc(
                NIL,
                NIL,
                Pd {
                    sum: values[lo],
                    tag: Tag::IDENTITY,
                },
            );
        }
        let mid = lo + (hi - lo) / 2;
        let l = Self::build(store, values, lo, mid);
        let r = Self::build(store, values, mid, hi);
        let sum = store.arena.agg(l).sum + store.arena.agg(r).sum;
        store.arena.alloc(
            l,
            r,
            Pd {
                sum,
                tag: Tag::IDENTITY,
            },
        )
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn first(&self) -> Version {
        Version(0)
    }

    pub fn latest(&self) -> Version {
        Version((self.store.versions() - 1) as u32)
    }

    pub fn range_add(&mut self, from: Version, l: usize, r: usize, d: i64) -> Version {
        self.apply(from, l, r, Tag::add(d))
    }

    pub fn range_assign(&mut self, from: Version, l: usize, r: usize, x: i64) -> Version {
        self.apply(from, l, r, Tag::assign(x))
    }

    fn apply(&mut self, from: Version, l: usize, r: usize, op: Tag) -> Version {
        assert!(l <= r && r <= self.n);
        let root = self.store.root(from);
        let new_root = if l == r {
            root
        } else {
            Self::update_rec(&mut self.store, root, 0, self.n, l, r, op, Tag::IDENTITY)
        };
        self.store.push_root(new_root)
    }

    /// Copy of `node` with `inherited` applied; shares the children.
    fn carry(store: &mut Store<Pd>, node: NodeId, len: usize, inherited: Tag) -> NodeId {
        let Node { l, r, agg } = *store.arena.get(node);
        store.arena.alloc(
            l,
            r,
            Pd {
                sum: inherited.apply(agg.sum, len),
                tag: agg.tag.then(inherited),
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn update_rec(
        store: &mut Store<Pd>,
        node: NodeId,
        lo: usize,
        hi: usize,
        l: usize,
        r: usize,
        op: Tag,
        inherited: Tag,
    ) -> NodeId {
        let Node { l: cl, r: cr, agg } = *store.arena.get(node);
        let len = hi - lo;
        // This node's pending tag, then what its ancestors are passing down.
        let sum = inherited.apply(agg.sum, len);
        let tag = agg.tag.then(inherited);
        if l <= lo && hi <= r {
            return store.arena.alloc(
                cl,
                cr,
                Pd {
                    sum: op.apply(sum, len),
                    tag: tag.then(op),
                },
            );
        }
        let mid = lo + (hi - lo) / 2;
        let left = if l < mid {
            Self::update_rec(store, cl, lo, mid, l, r, op, tag)
        } else if tag == Tag::IDENTITY {
            cl
        } else {
            Self::carry(store, cl, mid - lo, tag)
        };
        let right = if r > mid {
            Self::update_rec(store, cr, mid, hi, l, r, op, tag)
        } else if tag == Tag::IDENTITY {
            cr
        } else {
            Self::carry(store, cr, hi - mid, tag)
        };
        let sum = store.arena.agg(left).sum + store.arena.agg(right).sum;
        store.arena.alloc(
            left,
            right,
            Pd {
                sum,
                tag: Tag::IDENTITY,
            },
        )
    }

    /// Sum over `[l, r)` in version `v`.
    pub fn query(&self, v: Version, l: usize, r: usize) -> i64 {
        assert!(l <= r && r <= self.n);
        if l == r {
            return 0;
        }
        self.query_rec(self.store.root(v), 0, self.n, l, r, Tag::IDENTITY)
    }

    fn query_rec(
        &self,
        node: NodeId,
        lo: usize,
        hi: usize,
        l: usize,
        r: usize,
        inherited: Tag,
    ) -> i64 {
        let n = self.store.arena.get(node);
        if l <= lo && hi <= r {
            return inherited.apply(n.agg.sum, hi - lo);
        }
        let mid = lo + (hi - lo) / 2;
        let down = n.agg.tag.then(inherited);
        let left = if l < mid {
            self.query_rec(n.l, lo, mid, l, r, down)
        } else {
            0
        };
        let right = if r > mid {
            self.query_rec(n.r, mid, hi, l, r, down)
        } else {
            0
        };
        left + right
    }

    pub fn retire(&mut self, v: Version) -> bool {
        self.store.retire(v)
    }

    pub fn compact(&mut self) {
        self.store.compact();
    }

    pub fn stats(&self) -> Stats {
        self.store.stats()
    }

    pub fn live_nodes(&self) -> usize {
        self.store.arena.live()
    }
}

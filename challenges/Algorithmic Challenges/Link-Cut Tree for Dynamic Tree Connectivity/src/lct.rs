//! The link-cut tree of Sleator and Tarjan, as a safe, arena-backed splay forest.
//!
//! Every tree is cut into vertex-disjoint *preferred paths*. Each path lives in its own splay
//! tree (an *auxiliary tree*) ordered by depth, and the root of an auxiliary tree points at the
//! parent of the path's top vertex (a *path-parent* pointer, stored in the same `p` field as an
//! ordinary splay parent). Everything is built on one operation, [`LinkCutTree::access`], which
//! makes the path from the tree root to a vertex preferred; splaying pays for it in O(log n)
//! amortized.
//!
//! * **Evert** (`make_root`) reverses a path with a lazy flag. Aggregates are commutative, so the
//!   flag only swaps children and never touches a [`Stats`].
//! * **Path aggregates** are the splay-subtree aggregate of the auxiliary tree that
//!   [`LinkCutTree::expose`] leaves holding exactly the path between two vertices; **path add**
//!   is a lazy tag on that root.
//! * **Subtree sums** (`SUB = true`) additionally keep, per vertex, the total of its *virtual*
//!   children: the subtrees hanging off it that are not on its preferred path. `access` moves one
//!   child in and one out of that total, which needs subtraction, so only count and sum (the
//!   [`Sum`] aggregate) are supported. Subtree min and max would need a multiset per vertex.
//!
//! Node 0 is a null sentinel (never written), so a missing child needs no branch; vertex `v` is
//! node `v + 1`.

use crate::stats::{Stats, Sum};
use crate::{Forest, PathOps, SubtreeOps, Weighted};

const NIL: u32 = 0;

#[derive(Clone, Copy)]
struct Node {
    /// Splay children: shallower and deeper on the preferred path.
    ch: [u32; 2],
    /// Splay parent, or the path-parent when this node roots its auxiliary tree.
    p: u32,
    /// Children of this node still need swapping (its own `ch` is already swapped).
    rev: bool,
    live: bool,
    val: i64,
    /// Pending add for both children (this node's own `val` and `agg` already include it).
    add: i64,
    /// Aggregate over this node's splay subtree (path vertices only).
    agg: Stats,
}

/// Per-node bookkeeping for subtree sums: virtual children, and the whole represented subtree.
#[derive(Clone, Copy, Default)]
struct Sub {
    virt: Sum,
    tot: Sum,
}

/// A dynamic forest with O(log n) amortized link, cut, connectivity and path operations.
///
/// `SUB = true` also answers subtree sums, at the price of a second array and extra work in
/// `access`. See [`PathLct`] and [`SubtreeLct`].
pub struct LinkCutTree<const SUB: bool> {
    nodes: Vec<Node>,
    sub: Vec<Sub>,
    stack: Vec<u32>,
}

/// Link-cut tree with path aggregates only.
pub type PathLct = LinkCutTree<false>;
/// Link-cut tree that also keeps virtual-subtree sums.
pub type SubtreeLct = LinkCutTree<true>;

impl<const SUB: bool> LinkCutTree<SUB> {
    /// Bytes per vertex in the main array (the subtree variant adds [`Self::sub_bytes`]).
    pub const fn node_bytes() -> usize {
        std::mem::size_of::<Node>()
    }

    pub const fn sub_bytes() -> usize {
        if SUB { std::mem::size_of::<Sub>() } else { 0 }
    }

    fn build(values: &[Option<i64>]) -> Self {
        let nil = Node {
            ch: [NIL; 2],
            p: NIL,
            rev: false,
            live: false,
            val: 0,
            add: 0,
            agg: Stats::NEUTRAL,
        };
        let mut nodes = Vec::with_capacity(values.len() + 1);
        nodes.push(nil);
        nodes.extend(values.iter().map(|&v| Node {
            live: v.is_some(),
            val: v.unwrap_or(0),
            agg: Stats::from_option(v),
            ..nil
        }));
        let mut sub = Vec::new();
        if SUB {
            sub.push(Sub::default());
            sub.extend(values.iter().map(|&v| {
                let s = Stats::from_option(v).to_sum();
                Sub {
                    virt: Sum::default(),
                    tot: s,
                }
            }));
        }
        LinkCutTree {
            nodes,
            sub,
            stack: Vec::new(),
        }
    }

    #[inline]
    fn is_root(&self, x: u32) -> bool {
        let p = &self.nodes[self.nodes[x as usize].p as usize];
        p.ch[0] != x && p.ch[1] != x
    }

    #[inline]
    fn own(&self, x: u32) -> Stats {
        let n = &self.nodes[x as usize];
        if n.live {
            Stats::single(n.val)
        } else {
            Stats::NEUTRAL
        }
    }

    #[inline]
    fn pull(&mut self, x: u32) {
        let [l, r] = self.nodes[x as usize].ch;
        let own = self.own(x);
        self.nodes[x as usize].agg = self.nodes[l as usize]
            .agg
            .combine(own)
            .combine(self.nodes[r as usize].agg);
        if SUB {
            let tot = self.sub[l as usize].tot
                + self.sub[r as usize].tot
                + own.to_sum()
                + self.sub[x as usize].virt;
            self.sub[x as usize].tot = tot;
        }
    }

    #[inline]
    fn apply_rev(&mut self, x: u32) {
        if x != NIL {
            let n = &mut self.nodes[x as usize];
            n.ch.swap(0, 1);
            n.rev = !n.rev;
        }
    }

    #[inline]
    fn apply_add(&mut self, x: u32, d: i64) {
        if x != NIL && d != 0 {
            let n = &mut self.nodes[x as usize];
            n.add += d;
            if n.live {
                n.val += d;
            }
            n.agg = n.agg.shift(d);
            if SUB {
                // Only path vertices move; virtual subtrees hang off the path and keep their values.
                let count = self.nodes[x as usize].agg.count;
                self.sub[x as usize].tot.sum += d * count;
            }
        }
    }

    #[inline]
    fn push(&mut self, x: u32) {
        let n = self.nodes[x as usize];
        if n.rev {
            self.apply_rev(n.ch[0]);
            self.apply_rev(n.ch[1]);
            self.nodes[x as usize].rev = false;
        }
        if n.add != 0 {
            self.apply_add(n.ch[0], n.add);
            self.apply_add(n.ch[1], n.add);
            self.nodes[x as usize].add = 0;
        }
    }

    /// Rotate `x` above its splay parent. Does not pull `x`; `splay` does that once at the end.
    fn rotate(&mut self, x: u32) {
        let p = self.nodes[x as usize].p;
        let g = self.nodes[p as usize].p;
        let dir = usize::from(self.nodes[p as usize].ch[1] == x);
        if !self.is_root(p) {
            let gdir = usize::from(self.nodes[g as usize].ch[1] == p);
            self.nodes[g as usize].ch[gdir] = x;
        }
        self.nodes[x as usize].p = g;
        let b = self.nodes[x as usize].ch[dir ^ 1];
        self.nodes[p as usize].ch[dir] = b;
        if b != NIL {
            self.nodes[b as usize].p = p;
        }
        self.nodes[x as usize].ch[dir ^ 1] = p;
        self.nodes[p as usize].p = x;
        self.pull(p);
    }

    /// Splay `x` to the root of its auxiliary tree, pushing every pending tag on the way down.
    fn splay(&mut self, x: u32) {
        self.stack.clear();
        let mut y = x;
        self.stack.push(y);
        while !self.is_root(y) {
            y = self.nodes[y as usize].p;
            self.stack.push(y);
        }
        while let Some(y) = self.stack.pop() {
            self.push(y);
        }
        while !self.is_root(x) {
            let p = self.nodes[x as usize].p;
            if !self.is_root(p) {
                let g = self.nodes[p as usize].p;
                let zigzig =
                    (self.nodes[g as usize].ch[0] == p) == (self.nodes[p as usize].ch[0] == x);
                self.rotate(if zigzig { p } else { x });
            }
            self.rotate(x);
        }
        self.pull(x);
    }

    /// Make the path from the root of `x`'s tree to `x` preferred and leave `x` at the root of the
    /// auxiliary tree that holds it. Returns the last auxiliary tree joined, which is the lowest
    /// common ancestor when called for a second vertex (see [`Self::lca`]).
    fn access(&mut self, x: u32) -> u32 {
        let mut last = NIL;
        let mut y = x;
        while y != NIL {
            self.splay(y);
            if SUB {
                let right = self.nodes[y as usize].ch[1];
                let moved = self.sub[right as usize].tot - self.sub[last as usize].tot;
                let s = &mut self.sub[y as usize];
                s.virt = s.virt + moved;
            }
            self.nodes[y as usize].ch[1] = last;
            self.pull(y);
            last = y;
            y = self.nodes[y as usize].p;
        }
        self.splay(x);
        last
    }

    /// Make `x` the root of its tree.
    pub fn make_root(&mut self, x: usize) {
        let x = x as u32 + 1;
        self.access(x);
        self.apply_rev(x);
    }

    fn make_root_node(&mut self, x: u32) {
        self.access(x);
        self.apply_rev(x);
    }

    /// Root `u`'s tree at `u`, then leave the path `u..=v` alone in one auxiliary tree rooted at
    /// `u`. Returns whether `v` is in `u`'s tree; if not, nothing useful was exposed.
    ///
    /// Connectivity falls out of the splay: after `access(v)` the auxiliary tree of `v` has no
    /// path-parent, and `splay(u)` gives `v` a parent exactly when `u` was in that tree.
    fn expose(&mut self, u: u32, v: u32) -> bool {
        self.make_root_node(u);
        self.access(v);
        self.splay(u);
        u == v || self.nodes[v as usize].p != NIL
    }

    /// The root of the tree containing `v` (the vertex last passed to `make_root`, if any).
    pub fn find_root(&mut self, v: usize) -> usize {
        let v = v as u32 + 1;
        self.access(v);
        let mut r = v;
        loop {
            self.push(r);
            let l = self.nodes[r as usize].ch[0];
            if l == NIL {
                break;
            }
            r = l;
        }
        self.splay(r);
        r as usize - 1
    }

    /// Lowest common ancestor of `u` and `v` in the tree rooted at `root`, if all three are in
    /// the same tree.
    pub fn lca(&mut self, root: usize, u: usize, v: usize) -> Option<usize> {
        if !self.connected(root, u) || !self.connected(root, v) {
            return None;
        }
        self.make_root(root);
        self.access(u as u32 + 1);
        Some(self.access(v as u32 + 1) as usize - 1)
    }

    /// If `u - v` is an edge, leave `u` as the sole left child of `v` (the auxiliary tree is just
    /// those two vertices, `v` at the root) and return `true`.
    fn expose_edge(&mut self, u: u32, v: u32) -> bool {
        if u == v {
            return false;
        }
        self.make_root_node(u);
        self.access(v);
        let nv = &self.nodes[v as usize];
        let nu = &self.nodes[u as usize];
        nv.ch[0] == u && nu.ch[0] == NIL && nu.ch[1] == NIL
    }
}

impl<const SUB: bool> Forest for LinkCutTree<SUB> {
    fn new(n: usize) -> Self {
        Self::build(&vec![Some(0); n])
    }

    fn len(&self) -> usize {
        self.nodes.len() - 1
    }

    fn link(&mut self, u: usize, v: usize) -> bool {
        let (u, v) = (u as u32 + 1, v as u32 + 1);
        if u == v || self.expose(u, v) {
            return false;
        }
        // `u` roots its own tree and `v` roots the auxiliary tree of the root path of its tree.
        self.nodes[u as usize].p = v;
        if SUB {
            let tu = self.sub[u as usize].tot;
            self.sub[v as usize].virt = self.sub[v as usize].virt + tu;
        }
        self.pull(v);
        true
    }

    fn cut(&mut self, u: usize, v: usize) -> bool {
        let (u, v) = (u as u32 + 1, v as u32 + 1);
        if !self.expose_edge(u, v) {
            return false;
        }
        self.nodes[v as usize].ch[0] = NIL;
        self.nodes[u as usize].p = NIL;
        self.pull(u);
        self.pull(v);
        true
    }

    fn connected(&mut self, u: usize, v: usize) -> bool {
        let (u, v) = (u as u32 + 1, v as u32 + 1);
        u == v || self.expose(u, v)
    }
}

impl<const SUB: bool> Weighted for LinkCutTree<SUB> {
    fn from_values(values: &[Option<i64>]) -> Self {
        Self::build(values)
    }

    fn set(&mut self, v: usize, value: i64) {
        let x = v as u32 + 1;
        self.access(x);
        let n = &mut self.nodes[x as usize];
        n.live = true;
        n.val = value;
        self.pull(x);
    }

    fn get(&mut self, v: usize) -> Option<i64> {
        let x = v as u32 + 1;
        self.access(x);
        let n = &self.nodes[x as usize];
        n.live.then_some(n.val)
    }
}

impl<const SUB: bool> PathOps for LinkCutTree<SUB> {
    fn path_query(&mut self, u: usize, v: usize) -> Option<Stats> {
        let (u, v) = (u as u32 + 1, v as u32 + 1);
        self.expose(u, v).then(|| self.nodes[u as usize].agg)
    }

    fn path_add(&mut self, u: usize, v: usize, delta: i64) -> bool {
        let (u, v) = (u as u32 + 1, v as u32 + 1);
        if !self.expose(u, v) {
            return false;
        }
        self.apply_add(u, delta);
        true
    }
}

impl SubtreeOps for LinkCutTree<true> {
    fn component(&mut self, v: usize) -> Sum {
        let x = v as u32 + 1;
        self.access(x);
        self.sub[x as usize].tot
    }

    fn subtree_query(&mut self, p: usize, v: usize) -> Option<Sum> {
        let (p, v) = (p as u32 + 1, v as u32 + 1);
        if !self.expose_edge(p, v) {
            return None;
        }
        // `v` is the deepest vertex of the rooted path, so what hangs below it is all virtual.
        Some(self.own(v).to_sum() + self.sub[v as usize].virt)
    }
}

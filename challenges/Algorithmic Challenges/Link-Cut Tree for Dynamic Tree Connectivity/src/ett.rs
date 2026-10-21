//! Euler tour trees: each tree is a cyclic tour stored in a treap.
//!
//! A tree is the closed walk that crosses every edge once in each direction. Its sequence holds
//! one node per vertex (carrying the value) and two neutral *arc* nodes per edge, one for each
//! direction of travel:
//!
//! ```text
//!   a - b - c      tour:  a  (a>b)  b  (b>c)  c  (c>b)  b  (b>a)
//!       |
//!       d          tour:  ... b (b>d) d (d>b) b ...        (cyclic)
//! ```
//!
//! * `reroot(v)` rotates the cycle so it starts at `v`: one split and one merge.
//! * `link(u, v)` is `reroot(u) + arc + reroot(v) + arc`: three merges.
//! * `cut(u, v)` removes the two arcs of the edge; what lies between them is `v`'s side.
//! * **Subtree aggregates fall out of the order**: the vertices on `v`'s side of the edge `p - v`
//!   are exactly the cyclic stretch between the arc `p > v` and the arc `v > p`. Sums are two
//!   prefix reads; min and max take a range split. This is the one thing the link-cut tree cannot
//!   do, and a path aggregate is the one thing this structure cannot do.
//!
//! The sequence is a treap with parent pointers, so any node finds its tree's root (connectivity)
//! and its position (ordering the two arcs of an edge) in expected O(log n).

use foldhash::HashMap;

use crate::shapes::Rng;
use crate::stats::{Stats, Sum};
use crate::{Forest, SubtreeOps, Weighted};

const NIL: u32 = 0;

#[derive(Clone, Copy)]
struct Node {
    ch: [u32; 2],
    p: u32,
    prio: u32,
    size: u32,
    live: bool,
    val: i64,
    agg: Stats,
}

pub struct EulerTourTree {
    nodes: Vec<Node>,
    n: usize,
    /// Edge `(lo, hi)` packed as `lo << 32 | hi` to the first of its two arc nodes: `lo > hi`
    /// at that id, `hi > lo` at the next one.
    edges: HashMap<u64, u32>,
    free_arcs: Vec<u32>,
    rng: Rng,
}

fn key(u: usize, v: usize) -> u64 {
    let (lo, hi) = if u < v { (u, v) } else { (v, u) };
    (lo as u64) << 32 | hi as u64
}

impl EulerTourTree {
    pub const fn node_bytes() -> usize {
        std::mem::size_of::<Node>()
    }

    fn build(values: &[Option<i64>]) -> Self {
        let mut rng = Rng::new(0x5EED_0E77);
        let nil = Node {
            ch: [NIL; 2],
            p: NIL,
            prio: 0,
            size: 0,
            live: false,
            val: 0,
            agg: Stats::NEUTRAL,
        };
        let mut nodes = Vec::with_capacity(values.len() * 3 + 1);
        nodes.push(nil);
        nodes.extend(values.iter().map(|&v| Node {
            prio: rng.next_u64() as u32,
            size: 1,
            live: v.is_some(),
            val: v.unwrap_or(0),
            agg: Stats::from_option(v),
            ..nil
        }));
        EulerTourTree {
            nodes,
            n: values.len(),
            edges: HashMap::default(),
            free_arcs: Vec::new(),
            rng,
        }
    }

    #[inline]
    fn node(&self, x: u32) -> &Node {
        &self.nodes[x as usize]
    }

    #[inline]
    fn own(&self, x: u32) -> Stats {
        let n = self.node(x);
        if n.live {
            Stats::single(n.val)
        } else {
            Stats::NEUTRAL
        }
    }

    /// Recompute `x`'s size and aggregate from its children, and point the children at `x`.
    #[inline]
    fn update(&mut self, x: u32) {
        let [l, r] = self.node(x).ch;
        let own = self.own(x);
        let size = self.node(l).size + 1 + self.node(r).size;
        let agg = self.node(l).agg.combine(own).combine(self.node(r).agg);
        let n = &mut self.nodes[x as usize];
        n.size = size;
        n.agg = agg;
        if l != NIL {
            self.nodes[l as usize].p = x;
        }
        if r != NIL {
            self.nodes[r as usize].p = x;
        }
    }

    fn merge_rec(&mut self, a: u32, b: u32) -> u32 {
        if a == NIL {
            return b;
        }
        if b == NIL {
            return a;
        }
        if self.node(a).prio > self.node(b).prio {
            let m = self.merge_rec(self.node(a).ch[1], b);
            self.nodes[a as usize].ch[1] = m;
            self.update(a);
            a
        } else {
            let m = self.merge_rec(a, self.node(b).ch[0]);
            self.nodes[b as usize].ch[0] = m;
            self.update(b);
            b
        }
    }

    /// Concatenate two sequences (given by their roots); returns the new root.
    fn merge(&mut self, a: u32, b: u32) -> u32 {
        let r = self.merge_rec(a, b);
        self.nodes[r as usize].p = NIL;
        r
    }

    fn split_rec(&mut self, t: u32, k: u32) -> (u32, u32) {
        if t == NIL {
            return (NIL, NIL);
        }
        let l = self.node(t).ch[0];
        let ls = self.node(l).size;
        if k <= ls {
            let (a, b) = self.split_rec(l, k);
            self.nodes[t as usize].ch[0] = b;
            self.update(t);
            (a, t)
        } else {
            let (a, b) = self.split_rec(self.node(t).ch[1], k - ls - 1);
            self.nodes[t as usize].ch[1] = a;
            self.update(t);
            (t, b)
        }
    }

    /// Split a sequence into its first `k` nodes and the rest.
    fn split(&mut self, t: u32, k: u32) -> (u32, u32) {
        let (a, b) = self.split_rec(t, k);
        for x in [a, b] {
            if x != NIL {
                self.nodes[x as usize].p = NIL;
            }
        }
        (a, b)
    }

    fn root_of(&self, mut x: u32) -> u32 {
        while self.node(x).p != NIL {
            x = self.node(x).p;
        }
        x
    }

    /// 0-based position of `x` in its sequence.
    fn pos(&self, x: u32) -> u32 {
        let mut pos = self.node(self.node(x).ch[0]).size;
        let mut y = x;
        while self.node(y).p != NIL {
            let p = self.node(y).p;
            if self.node(p).ch[1] == y {
                pos += self.node(self.node(p).ch[0]).size + 1;
            }
            y = p;
        }
        pos
    }

    /// Aggregate of everything up to and including `x`.
    fn prefix(&self, x: u32) -> Stats {
        let mut acc = self.node(self.node(x).ch[0]).agg.combine(self.own(x));
        let mut y = x;
        while self.node(y).p != NIL {
            let p = self.node(y).p;
            if self.node(p).ch[1] == y {
                acc = acc
                    .combine(self.node(self.node(p).ch[0]).agg)
                    .combine(self.own(p));
            }
            y = p;
        }
        acc
    }

    /// Rotate the tour of `x`'s tree to start at `x`; returns the root.
    fn reroot(&mut self, x: u32) -> u32 {
        let root = self.root_of(x);
        let k = self.pos(x);
        if k == 0 {
            return root;
        }
        let (a, b) = self.split(root, k);
        self.merge(b, a)
    }

    fn new_arcs(&mut self) -> u32 {
        if let Some(a) = self.free_arcs.pop() {
            return a;
        }
        let a = self.nodes.len() as u32;
        for _ in 0..2 {
            let prio = self.rng.next_u64() as u32;
            self.nodes.push(Node {
                ch: [NIL; 2],
                p: NIL,
                prio,
                size: 1,
                live: false,
                val: 0,
                agg: Stats::NEUTRAL,
            });
        }
        a
    }

    fn reset_arc(&mut self, a: u32) {
        let n = &mut self.nodes[a as usize];
        n.ch = [NIL; 2];
        n.p = NIL;
        n.size = 1;
        n.agg = Stats::NEUTRAL;
    }

    /// The arcs `(p > v, v > p)` of the edge `p - v`, if it exists.
    fn arcs(&self, p: usize, v: usize) -> Option<(u32, u32)> {
        let &e = self.edges.get(&key(p, v))?;
        Some(if p < v { (e, e + 1) } else { (e + 1, e) })
    }

    /// Aggregate over the vertices on `v`'s side of the edge `p - v`, including min and max. Unlike
    /// the sum in [`SubtreeOps::subtree_query`] this splits and re-joins the tour.
    pub fn subtree_stats(&mut self, p: usize, v: usize) -> Option<Stats> {
        let (a, b) = self.arcs(p, v)?;
        let root = self.root_of(a);
        let (pa, pb) = (self.pos(a), self.pos(b));
        if pa < pb {
            let (head, rest) = self.split(root, pa + 1);
            let (mid, tail) = self.split(rest, pb - pa - 1);
            let stats = self.node(mid).agg;
            let joined = self.merge(mid, tail);
            self.merge(head, joined);
            Some(stats)
        } else {
            let (head, rest) = self.split(root, pb);
            let (mid, tail) = self.split(rest, pa - pb + 1);
            let stats = self.node(head).agg.combine(self.node(tail).agg);
            let joined = self.merge(head, mid);
            self.merge(joined, tail);
            Some(stats)
        }
    }

    /// Check treap order, sizes, aggregates and parent pointers of every tour; used by tests.
    pub fn check_invariants(&self) {
        fn walk(t: &EulerTourTree, x: u32, parent: u32) -> usize {
            let n = t.node(x);
            assert_eq!(n.p, parent, "parent pointer of {x}");
            let mut size = 1;
            let mut agg = t.own(x);
            for &c in n.ch.iter().filter(|&&c| c != NIL) {
                assert!(n.prio >= t.node(c).prio, "heap order at {x}");
                size += walk(t, c, x);
                agg = agg.combine(t.node(c).agg);
            }
            assert_eq!(n.size as usize, size, "size of {x}");
            assert_eq!(n.agg, agg, "aggregate of {x}");
            size
        }
        let mut free = vec![false; self.nodes.len()];
        for &e in &self.free_arcs {
            free[e as usize] = true;
            free[e as usize + 1] = true;
        }
        let seen: usize = (1..self.nodes.len() as u32)
            .filter(|&x| self.node(x).p == NIL && !free[x as usize])
            .map(|x| walk(self, x, NIL))
            .sum();
        assert_eq!(
            seen,
            self.n + 2 * self.edges.len(),
            "every live node is in exactly one tour"
        );
    }
}

impl Forest for EulerTourTree {
    fn new(n: usize) -> Self {
        Self::build(&vec![Some(0); n])
    }

    fn len(&self) -> usize {
        self.n
    }

    fn link(&mut self, u: usize, v: usize) -> bool {
        let (x, y) = (u as u32 + 1, v as u32 + 1);
        assert!(u < self.n && v < self.n, "vertex out of range");
        if u == v || self.root_of(x) == self.root_of(y) {
            return false;
        }
        let tu = self.reroot(x);
        let tv = self.reroot(y);
        let e = self.new_arcs();
        self.edges.insert(key(u, v), e);
        let (a, b) = if u < v { (e, e + 1) } else { (e + 1, e) };
        let t = self.merge(tu, a);
        let t = self.merge(t, tv);
        self.merge(t, b);
        true
    }

    fn cut(&mut self, u: usize, v: usize) -> bool {
        assert!(u < self.n && v < self.n, "vertex out of range");
        let Some((a, b)) = self.arcs(u, v) else {
            return false;
        };
        self.edges.remove(&key(u, v));
        let root = self.root_of(a);
        let (pa, pb) = (self.pos(a), self.pos(b));
        let (lo, hi) = (pa.min(pb), pa.max(pb));
        // tour = head . first . between . second . tail
        let (head, rest) = self.split(root, lo);
        let (_first, rest) = self.split(rest, 1);
        let (_between, rest) = self.split(rest, hi - lo - 1);
        let (_second, tail) = self.split(rest, 1);
        if head != NIL || tail != NIL {
            self.merge(head, tail);
        }
        let e = a.min(b);
        self.reset_arc(e);
        self.reset_arc(e + 1);
        self.free_arcs.push(e);
        true
    }

    fn connected(&mut self, u: usize, v: usize) -> bool {
        assert!(u < self.n && v < self.n, "vertex out of range");
        u == v || self.root_of(u as u32 + 1) == self.root_of(v as u32 + 1)
    }
}

impl Weighted for EulerTourTree {
    fn from_values(values: &[Option<i64>]) -> Self {
        Self::build(values)
    }

    fn set(&mut self, v: usize, value: i64) {
        let x = v as u32 + 1;
        let n = &mut self.nodes[x as usize];
        n.live = true;
        n.val = value;
        let mut y = x;
        while y != NIL {
            self.update(y);
            y = self.node(y).p;
        }
    }

    fn get(&mut self, v: usize) -> Option<i64> {
        let n = self.node(v as u32 + 1);
        n.live.then_some(n.val)
    }
}

impl SubtreeOps for EulerTourTree {
    fn component(&mut self, v: usize) -> Sum {
        self.node(self.root_of(v as u32 + 1)).agg.to_sum()
    }

    fn subtree_query(&mut self, p: usize, v: usize) -> Option<Sum> {
        let (a, b) = self.arcs(p, v)?;
        let (pa, pb) = (self.pos(a), self.pos(b));
        let to_b = self.prefix(b);
        let to_a = self.prefix(a);
        // The arcs are neutral, so inclusive prefixes lose nothing at the boundaries.
        let stats = if pa < pb {
            to_b.to_sum() - to_a.to_sum()
        } else {
            let total = self.node(self.root_of(a)).agg;
            total.to_sum() - to_a.to_sum() + to_b.to_sum()
        };
        Some(stats)
    }
}

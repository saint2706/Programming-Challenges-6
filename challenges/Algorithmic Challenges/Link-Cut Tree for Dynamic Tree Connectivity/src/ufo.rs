//! UFO trees: dynamic trees by incremental tree contraction, with unbounded fan-out.
//!
//! A forest is contracted level by level into a hierarchy of *clusters*; level 0 is the vertices.
//! One round turns the level-`i` forest into the level-`i+1` forest by these rules, where the
//! *degree* of a cluster is its number of neighbours in its own level:
//!
//! 1. **Stars.** A cluster of degree at least 3 (a *centre*) merges with **all** of its degree-1
//!    neighbours, however many, into one parent. This is the unbounded fan-out that topology trees
//!    avoid by ternarizing the input, and it is why a star of `n` leaves has height 1.
//! 2. **Pairs.** The remaining clusters of degree 1 or 2 are matched along a *maximal* matching:
//!    a cluster is merged with an adjacent unmatched cluster whenever one exists, and otherwise
//!    gets a parent of its own.
//! 3. A cluster of degree 0 is the top of its tree and has no parent.
//!
//! Maximality makes every round shrink the forest geometrically, so the height is O(log n); a
//! path of `n` vertices contracts like a balanced tree and a star collapses in one round.
//!
//! **Updates.** `link` and `cut` change one edge at level 0 and project the change up through the
//! parents (an edge between two clusters is witnessed by exactly one edge below, because the
//! forest has no cycles). Every cluster whose neighbourhood changed is queued, and the levels are
//! then repaired bottom-up: a queued cluster re-checks the three rules for itself and, when it
//! breaks a pair or a star, queues the clusters it displaced. Only O(1) clusters per level are
//! displaced, except that attaching or detaching a leaf of a big star touches nothing but that
//! leaf: the star's parent is kept. That is the paper's key point and the reason a star of a
//! million leaves still updates in constant time.
//!
//! This is a from-scratch formulation of the rules in "UFO Trees: Practical and
//! Provably-Efficient Parallel Batch-Dynamic Trees" (De Man et al., 2026), repaired locally
//! instead of with the paper's delete-ancestors pass, and it supports connectivity only (the
//! paper's path and subtree queries are not ported). [`UfoTree::check_invariants`] verifies every
//! rule above, the edge projection and the connectedness of every cluster; the tests call it
//! after each operation.

use foldhash::HashSet;

use crate::Forest;

const NIL: u32 = u32::MAX;
const INLINE: usize = 4;

/// A set of `u32` that stays inline up to four members and spills to a hash set beyond that,
/// because most clusters have at most a few neighbours and children but a star hub has millions.
enum SmallSet {
    Inline(u8, [u32; INLINE]),
    Heap(Box<HashSet<u32>>),
}

enum SetIter<'a> {
    Inline(std::slice::Iter<'a, u32>),
    Heap(std::collections::hash_set::Iter<'a, u32>),
}

impl Iterator for SetIter<'_> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<u32> {
        match self {
            SetIter::Inline(it) => it.next().copied(),
            SetIter::Heap(it) => it.next().copied(),
        }
    }
}

impl SmallSet {
    fn new() -> Self {
        SmallSet::Inline(0, [0; INLINE])
    }

    #[inline]
    fn len(&self) -> usize {
        match self {
            SmallSet::Inline(n, _) => *n as usize,
            SmallSet::Heap(h) => h.len(),
        }
    }

    #[inline]
    fn contains(&self, x: u32) -> bool {
        match self {
            SmallSet::Inline(n, a) => a[..*n as usize].contains(&x),
            SmallSet::Heap(h) => h.contains(&x),
        }
    }

    fn insert(&mut self, x: u32) -> bool {
        match self {
            SmallSet::Inline(n, a) => {
                if a[..*n as usize].contains(&x) {
                    return false;
                }
                if (*n as usize) < INLINE {
                    a[*n as usize] = x;
                    *n += 1;
                } else {
                    let mut h: HashSet<u32> = HashSet::default();
                    h.extend(a.iter().copied());
                    h.insert(x);
                    *self = SmallSet::Heap(Box::new(h));
                }
                true
            }
            SmallSet::Heap(h) => h.insert(x),
        }
    }

    fn remove(&mut self, x: u32) -> bool {
        match self {
            SmallSet::Inline(n, a) => {
                let len = *n as usize;
                if let Some(i) = a[..len].iter().position(|&y| y == x) {
                    a[i] = a[len - 1];
                    *n -= 1;
                    true
                } else {
                    false
                }
            }
            SmallSet::Heap(h) => {
                let had = h.remove(&x);
                if had && h.len() < INLINE {
                    let mut a = [0; INLINE];
                    for (slot, v) in a.iter_mut().zip(h.iter()) {
                        *slot = *v;
                    }
                    *self = SmallSet::Inline(h.len() as u8, a);
                }
                had
            }
        }
    }

    fn iter(&self) -> SetIter<'_> {
        match self {
            SmallSet::Inline(n, a) => SetIter::Inline(a[..*n as usize].iter()),
            SmallSet::Heap(h) => SetIter::Heap(h.iter()),
        }
    }

    fn first(&self) -> Option<u32> {
        self.iter().next()
    }
}

struct Cluster {
    parent: u32,
    level: u32,
    /// For a star, the child of degree >= 3 that the other children are leaves of.
    center: u32,
    alive: bool,
    queued: bool,
    /// Neighbours in the same level.
    nbrs: SmallSet,
    /// Children one level down (empty at level 0).
    kids: SmallSet,
}

/// A dynamic forest by tree contraction. See the module docs.
pub struct UfoTree {
    cl: Vec<Cluster>,
    n: usize,
    free: Vec<u32>,
    /// Clusters freed during the current update; reused only afterwards, so a stale queue entry
    /// can never name a recycled id.
    pending_free: Vec<u32>,
    /// Per level, the clusters waiting to be re-checked.
    work: Vec<Vec<u32>>,
    pool: Vec<Vec<u32>>,
}

impl UfoTree {
    /// Bytes per cluster record (the neighbour and child sets may add a heap block).
    pub const fn cluster_bytes() -> usize {
        std::mem::size_of::<Cluster>()
    }

    /// Number of levels in the tallest tree (a lone vertex has height 1).
    pub fn height(&self) -> usize {
        self.cl
            .iter()
            .filter(|c| c.alive)
            .map(|c| c.level as usize + 1)
            .max()
            .unwrap_or(0)
    }

    /// Number of live clusters over all levels, vertices included.
    pub fn cluster_count(&self) -> usize {
        self.cl.iter().filter(|c| c.alive).count()
    }

    // ----- small accessors -------------------------------------------------------------------

    #[inline]
    fn c(&self, x: u32) -> &Cluster {
        &self.cl[x as usize]
    }

    #[inline]
    fn cm(&mut self, x: u32) -> &mut Cluster {
        &mut self.cl[x as usize]
    }

    #[inline]
    fn deg(&self, x: u32) -> usize {
        self.c(x).nbrs.len()
    }

    #[inline]
    fn parent(&self, x: u32) -> Option<u32> {
        let p = self.c(x).parent;
        (p != NIL).then_some(p)
    }

    fn top(&self, mut x: u32) -> u32 {
        while let Some(p) = self.parent(x) {
            x = p;
        }
        x
    }

    fn take_buf(&mut self) -> Vec<u32> {
        let mut v = self.pool.pop().unwrap_or_default();
        v.clear();
        v
    }

    fn put_buf(&mut self, v: Vec<u32>) {
        self.pool.push(v);
    }

    fn nbrs_of(&mut self, x: u32) -> Vec<u32> {
        let mut v = self.take_buf();
        v.extend(self.c(x).nbrs.iter());
        v
    }

    fn kids_of(&mut self, x: u32) -> Vec<u32> {
        let mut v = self.take_buf();
        v.extend(self.c(x).kids.iter());
        v
    }

    fn touch(&mut self, x: u32) {
        let c = self.cm(x);
        if c.queued {
            return;
        }
        c.queued = true;
        let level = c.level as usize;
        if self.work.len() <= level {
            self.work.resize_with(level + 1, Vec::new);
        }
        self.work[level].push(x);
    }

    fn new_cluster(&mut self, level: u32) -> u32 {
        let fresh = Cluster {
            parent: NIL,
            level,
            center: NIL,
            alive: true,
            queued: false,
            nbrs: SmallSet::new(),
            kids: SmallSet::new(),
        };
        if let Some(id) = self.free.pop() {
            self.cl[id as usize] = fresh;
            id
        } else {
            self.cl.push(fresh);
            (self.cl.len() - 1) as u32
        }
    }

    // ----- mutators that keep the parent/neighbour projection consistent --------------------

    /// Add the edge `a - b` at their level and project it through every pair of ancestors.
    fn add_edge(&mut self, mut a: u32, mut b: u32) {
        loop {
            debug_assert!(a != b);
            let fresh_a = self.cm(a).nbrs.insert(b);
            let fresh_b = self.cm(b).nbrs.insert(a);
            debug_assert!(fresh_a && fresh_b, "edge already present");
            self.touch(a);
            self.touch(b);
            match (self.parent(a), self.parent(b)) {
                (Some(pa), Some(pb)) => {
                    debug_assert!(pa != pb, "edge inside one cluster would close a cycle");
                    a = pa;
                    b = pb;
                }
                _ => return,
            }
        }
    }

    /// Remove the edge `a - b`, first splitting any cluster that holds both ends.
    fn del_edge(&mut self, mut a: u32, mut b: u32) {
        loop {
            if let (Some(pa), Some(pb)) = (self.parent(a), self.parent(b))
                && pa == pb
            {
                self.split_internal(pa, a, b);
            }
            self.cm(a).nbrs.remove(b);
            self.cm(b).nbrs.remove(a);
            self.touch(a);
            self.touch(b);
            match (self.parent(a), self.parent(b)) {
                (Some(pa), Some(pb)) => {
                    a = pa;
                    b = pb;
                }
                _ => return,
            }
        }
    }

    /// The edge `a - b` is about to disappear from inside the cluster `p`: detach the leaf of a
    /// star, or dissolve a pair.
    fn split_internal(&mut self, p: u32, a: u32, b: u32) {
        let center = self.c(p).center;
        if center == a {
            self.clear_parent(b);
        } else if center == b {
            self.clear_parent(a);
        } else {
            self.dissolve(p);
        }
    }

    /// Take `c` out of its parent `p` and remove the edges that `c` alone gave `p` upstairs.
    /// Never leaves `p` disconnected: callers remove a leaf of a star or every child.
    fn clear_parent(&mut self, c: u32) {
        let p = self.c(c).parent;
        debug_assert!(p != NIL);
        if self.c(p).alive && self.c(p).center == c && self.c(p).kids.len() > 1 {
            // The leaves of a star are only connected through its centre, so the star cannot
            // outlive it. (A cluster that is already being dissolved is torn down child by child.)
            self.dissolve(p);
            return;
        }
        let nbrs = self.nbrs_of(c);
        for &z in &nbrs {
            if let Some(q) = self.parent(z)
                && q != p
            {
                self.del_edge(p, q);
            }
        }
        self.put_buf(nbrs);
        self.cm(p).kids.remove(c);
        if self.c(p).center == c {
            self.cm(p).center = NIL;
        }
        self.cm(c).parent = NIL;
        self.touch(c);
        if self.c(p).kids.len() == 0 {
            // An empty cluster is garbage (unless it is already being dissolved).
            if self.c(p).alive {
                self.dissolve(p);
            }
            return;
        }
        // The cluster lost a child, so its remaining children must re-check their roles.
        if self.c(p).kids.len() <= INLINE {
            let kids = self.kids_of(p);
            for &k in &kids {
                self.touch(k);
            }
            self.put_buf(kids);
        } else if self.c(p).center != NIL {
            let center = self.c(p).center;
            self.touch(center);
        }
    }

    /// Put the parentless `c` into `p` and add the edges that this gives `p` upstairs.
    fn set_parent(&mut self, c: u32, p: u32) {
        debug_assert_eq!(self.c(c).parent, NIL);
        self.cm(c).parent = p;
        self.cm(p).kids.insert(c);
        let nbrs = self.nbrs_of(c);
        for &z in &nbrs {
            if let Some(q) = self.parent(z)
                && q != p
            {
                self.add_edge(p, q);
            }
        }
        self.put_buf(nbrs);
        self.touch(p);
    }

    /// Destroy the cluster `p`: its children become parentless and are queued.
    fn dissolve(&mut self, p: u32) {
        self.cm(p).alive = false;
        if self.parent(p).is_some() {
            self.clear_parent(p);
        }
        let kids = self.kids_of(p);
        for &k in &kids {
            if self.c(k).parent == p {
                self.clear_parent(k);
            }
        }
        self.put_buf(kids);
        debug_assert_eq!(self.deg(p), 0, "a dissolved cluster must be isolated");
        self.pending_free.push(p);
    }
}

// ----- the three contraction rules -----------------------------------------------------------

impl UfoTree {
    /// `y` has one neighbour, and that neighbour is a centre: `y` must join the centre's star.
    fn forced_leaf_of(&self, y: u32) -> Option<u32> {
        if self.deg(y) != 1 {
            return None;
        }
        let h = self.c(y).nbrs.first()?;
        (self.deg(h) >= 3).then_some(h)
    }

    /// Not matched with anything: no parent, or a parent of its own that holds nothing else.
    fn unmatched(&self, y: u32) -> bool {
        match self.parent(y) {
            None => true,
            Some(p) => self.c(p).kids.len() == 1 && self.c(p).center == NIL,
        }
    }

    /// A neighbour of `x` that `x` could pair with right now.
    fn find_partner(&self, x: u32) -> Option<u32> {
        self.c(x)
            .nbrs
            .iter()
            .find(|&z| self.deg(z) <= 2 && self.unmatched(z))
    }

    /// Re-establish the contraction rules around `x`, queueing every cluster it displaces.
    fn settle(&mut self, x: u32) {
        let d = self.deg(x);
        if d == 0 {
            // A tree of one cluster is finished: no parent. Its old parent held only `x`.
            if let Some(p) = self.parent(x) {
                self.dissolve(p);
            }
        } else if d >= 3 {
            self.settle_centre(x);
        } else if let Some(h) = self.forced_leaf_of(x) {
            self.settle_leaf(x, h);
        } else {
            self.settle_free(x);
        }
    }

    /// `h` has degree >= 3: give it a star parent holding exactly its degree-1 neighbours.
    fn settle_centre(&mut self, h: u32) {
        if let Some(p) = self.parent(h) {
            if self.c(p).center == h {
                // Already a star around `h`; the leaves that come and go settle themselves.
                return;
            }
            self.dissolve(p); // `h` was matched in a pair or alone
        }
        let level = self.c(h).level;
        let p = self.new_cluster(level + 1);
        self.cm(p).center = h;
        self.set_parent(h, p);
        let nbrs = self.nbrs_of(h);
        for &y in &nbrs {
            if self.deg(y) == 1 {
                self.absorb_leaf(y, p);
            }
        }
        self.put_buf(nbrs);
    }

    /// Move the degree-1 cluster `y` into the star `p`.
    fn absorb_leaf(&mut self, y: u32, p: u32) {
        match self.parent(y) {
            Some(old) if old == p => return,
            Some(old) => self.dissolve(old),
            None => {}
        }
        self.set_parent(y, p);
    }

    fn settle_leaf(&mut self, y: u32, h: u32) {
        let star = |t: &Self| t.parent(h).filter(|&p| t.c(p).center == h);
        if star(self).is_none() {
            self.settle_centre(h);
        }
        let p = star(self).expect("settling a centre gives it a star");
        self.absorb_leaf(y, p);
    }

    /// `x` has degree 1 or 2 and is not forced into a star: keep it in a valid pair, or alone with
    /// no unmatched neighbour to pair with.
    fn settle_free(&mut self, x: u32) {
        if let Some(p) = self.parent(x) {
            let center = self.c(p).center;
            if center == x {
                self.dissolve(p); // no longer a centre
            } else if center != NIL {
                self.clear_parent(x); // no longer a forced leaf
            }
        }
        if let Some(p) = self.parent(x) {
            match self.c(p).kids.len() {
                1 => {
                    if let Some(y) = self.find_partner(x) {
                        self.pair_into(p, y);
                    }
                    return;
                }
                2 => {
                    let y = self
                        .c(p)
                        .kids
                        .iter()
                        .find(|&k| k != x)
                        .expect("a pair has two children");
                    if self.c(x).nbrs.contains(y) && self.deg(y) <= 2 {
                        return;
                    }
                    self.dissolve(p);
                }
                _ => self.dissolve(p),
            }
        }
        // `x` has no parent now.
        match self.find_partner(x) {
            Some(y) => match self.parent(y) {
                Some(py) => self.set_parent(x, py),
                None => {
                    let p = self.new_cluster(self.c(x).level + 1);
                    self.set_parent(x, p);
                    self.set_parent(y, p);
                }
            },
            None => {
                let p = self.new_cluster(self.c(x).level + 1);
                self.set_parent(x, p);
            }
        }
    }

    /// Move the unmatched `y` into the cluster `p`, which holds only the neighbour it pairs with.
    fn pair_into(&mut self, p: u32, y: u32) {
        if let Some(old) = self.parent(y) {
            self.dissolve(old);
        }
        self.set_parent(y, p);
    }

    /// Re-check every queued cluster, level by level, until the rules hold everywhere.
    fn repair(&mut self) {
        let mut level = 0;
        let mut steps = 0usize;
        while level < self.work.len() {
            while let Some(x) = self.work[level].pop() {
                self.cm(x).queued = false;
                if !self.c(x).alive {
                    continue;
                }
                self.settle(x);
                steps += 1;
                debug_assert!(steps < 50_000_000, "repair is not converging");
            }
            level += 1;
        }
        self.work.clear();
        let freed = std::mem::take(&mut self.pending_free);
        for &id in &freed {
            let c = self.cm(id);
            c.nbrs = SmallSet::new();
            c.kids = SmallSet::new();
        }
        self.free.extend(freed);
    }
}

impl Forest for UfoTree {
    fn new(n: usize) -> Self {
        let cl = (0..n)
            .map(|_| Cluster {
                parent: NIL,
                level: 0,
                center: NIL,
                alive: true,
                queued: false,
                nbrs: SmallSet::new(),
                kids: SmallSet::new(),
            })
            .collect();
        UfoTree {
            cl,
            n,
            free: Vec::new(),
            pending_free: Vec::new(),
            work: Vec::new(),
            pool: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.n
    }

    fn link(&mut self, u: usize, v: usize) -> bool {
        assert!(u < self.n && v < self.n, "vertex out of range");
        let (u, v) = (u as u32, v as u32);
        if u == v || self.top(u) == self.top(v) {
            return false;
        }
        self.add_edge(u, v);
        self.repair();
        true
    }

    fn cut(&mut self, u: usize, v: usize) -> bool {
        assert!(u < self.n && v < self.n, "vertex out of range");
        let (u, v) = (u as u32, v as u32);
        if !self.c(u).nbrs.contains(v) {
            return false;
        }
        self.del_edge(u, v);
        self.repair();
        true
    }

    fn connected(&mut self, u: usize, v: usize) -> bool {
        assert!(u < self.n && v < self.n, "vertex out of range");
        u == v || self.top(u as u32) == self.top(v as u32)
    }
}

// ----- verification --------------------------------------------------------------------------

impl UfoTree {
    /// Whether `u - v` is currently an edge of the forest.
    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        self.c(u as u32).nbrs.contains(v as u32)
    }

    /// Verify every structural rule from scratch; panics with a description on the first failure.
    ///
    /// * neighbour sets are symmetric and stay within one level;
    /// * parent and child links agree, every parent is one level up, a cluster has a parent
    ///   exactly when its degree is positive;
    /// * every cluster's neighbours are exactly the parents of its children's neighbours, each
    ///   reached by exactly one edge, and its children induce a connected subtree;
    /// * stars: a child of degree >= 3 is the centre, and the children are exactly the centre
    ///   plus all of its degree-1 neighbours; otherwise a cluster has one child or two adjacent
    ///   children of degree <= 2;
    /// * maximality: no two adjacent clusters of degree <= 2 are both alone in their parent.
    pub fn check_invariants(&self) {
        for (i, c) in self.cl.iter().enumerate() {
            if !c.alive {
                continue;
            }
            let x = i as u32;
            for z in c.nbrs.iter() {
                assert!(self.c(z).alive, "{x}: dead neighbour {z}");
                assert!(z != x, "{x}: self loop");
                assert_eq!(
                    self.c(z).level,
                    c.level,
                    "{x}-{z}: neighbours on two levels"
                );
                assert!(self.c(z).nbrs.contains(x), "{x}-{z}: asymmetric edge");
            }
            let d = c.nbrs.len();
            match self.parent(x) {
                None => assert_eq!(d, 0, "{x}: degree {d} but no parent"),
                Some(p) => {
                    assert!(d > 0, "{x}: isolated cluster has parent {p}");
                    assert!(self.c(p).alive, "{x}: dead parent");
                    assert_eq!(self.c(p).level, c.level + 1, "{x}: parent level");
                    assert!(self.c(p).kids.contains(x), "{x}: parent does not list it");
                }
            }
            if c.level == 0 {
                assert_eq!(c.kids.len(), 0, "{x}: a vertex has children");
                continue;
            }
            self.check_cluster(x);
        }
        self.check_maximal();
    }

    fn check_cluster(&self, x: u32) {
        let c = self.c(x);
        let kids: Vec<u32> = c.kids.iter().collect();
        assert!(!kids.is_empty(), "{x}: no children");
        for &k in &kids {
            assert!(
                self.c(k).alive && self.c(k).parent == x,
                "{x}: child {k} disagrees"
            );
        }
        // The neighbours are exactly the projected neighbours, once each.
        let mut expect = Vec::new();
        for &k in &kids {
            for z in self.c(k).nbrs.iter() {
                match self.parent(z) {
                    Some(q) if q != x => expect.push(q),
                    Some(_) => {}
                    None => panic!("{x}: neighbour {z} of child {k} has no parent"),
                }
            }
        }
        expect.sort_unstable();
        let before = expect.len();
        expect.dedup();
        assert_eq!(
            before,
            expect.len(),
            "{x}: two edges reach one neighbour (a cycle)"
        );
        let mut have: Vec<u32> = c.nbrs.iter().collect();
        have.sort_unstable();
        assert_eq!(
            have, expect,
            "{x}: neighbours are not the projection of the children"
        );
        // The children form a connected subtree.
        let kid_set: std::collections::HashSet<u32> = kids.iter().copied().collect();
        let mut seen = std::collections::HashSet::from([kids[0]]);
        let mut stack = vec![kids[0]];
        while let Some(a) = stack.pop() {
            for z in self.c(a).nbrs.iter() {
                if kid_set.contains(&z) && seen.insert(z) {
                    stack.push(z);
                }
            }
        }
        assert_eq!(seen.len(), kids.len(), "{x}: children are not connected");
        // Shape.
        let centres: Vec<u32> = kids.iter().copied().filter(|&k| self.deg(k) >= 3).collect();
        if c.center != NIL {
            assert!(
                kids.contains(&c.center) && self.deg(c.center) >= 3,
                "{x}: bad centre"
            );
            assert_eq!(centres, [c.center], "{x}: more than one centre");
            let leaves = self.c(c.center).nbrs.iter().filter(|&z| self.deg(z) == 1);
            let mut want: Vec<u32> = leaves.chain([c.center]).collect();
            want.sort_unstable();
            let mut got = kids.clone();
            got.sort_unstable();
            assert_eq!(got, want, "{x}: star is not the centre plus all its leaves");
        } else {
            assert!(centres.is_empty(), "{x}: child of degree >= 3 but no star");
            match kids.as_slice() {
                [_] => {}
                [a, b] => {
                    assert!(self.c(*a).nbrs.contains(*b), "{x}: pair is not adjacent");
                    assert!(
                        self.deg(*a) <= 2 && self.deg(*b) <= 2,
                        "{x}: pair with a hub"
                    );
                }
                _ => panic!("{x}: {} children without a centre", kids.len()),
            }
        }
    }

    fn check_maximal(&self) {
        for (i, c) in self.cl.iter().enumerate() {
            if !c.alive || c.nbrs.len() == 0 {
                continue;
            }
            let x = i as u32;
            if c.nbrs.len() >= 3 {
                let p = self.parent(x).expect("a hub has a parent");
                assert_eq!(self.c(p).center, x, "{x}: hub is not its star's centre");
                continue;
            }
            for z in c.nbrs.iter() {
                if self.deg(z) <= 2 {
                    assert!(
                        !(self.unmatched(x) && self.unmatched(z)),
                        "{x} and {z} are adjacent, unmatched and could merge"
                    );
                }
            }
        }
    }
}

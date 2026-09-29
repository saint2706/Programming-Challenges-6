//! Arena-allocated B-tree with per-child subtree counts.
//!
//! Every node stores its keys in one contiguous `Vec`, and every child
//! pointer is paired with the number of keys in that child's subtree. A
//! descent therefore touches ~`log_{2T}(n)` nodes, each a couple of cache
//! lines, instead of ~`1.4*log2(n)` scattered binary-tree nodes -- the
//! layout argument of Khuong & Morin, *Cache-Friendly Search Trees*.
//!
//! * `T` is the minimum degree: non-root nodes hold `T-1 ..= 2T-1` keys.
//! * Nodes live in one `Vec` and refer to each other by index (no `unsafe`,
//!   no `Box` chasing, freed slots recycled through a free list).
//! * Keys sit in internal nodes too (a classic B-tree, not a B+-tree), so a
//!   single in-order sequence exists and duplicates need no special casing:
//!   inserts land after equal keys, `rank` is a lower bound.

use std::mem;
use std::ops::{Bound, RangeBounds};

use crate::OrderStatistics;

#[derive(Clone, Debug)]
struct Node<K> {
    keys: Vec<K>,
    /// Empty for leaves, otherwise `keys.len() + 1` entries.
    children: Vec<usize>,
    /// `counts[i]` = number of keys in the subtree rooted at `children[i]`.
    counts: Vec<usize>,
}

impl<K> Default for Node<K> {
    fn default() -> Self {
        Self {
            keys: Vec::new(),
            children: Vec::new(),
            counts: Vec::new(),
        }
    }
}

impl<K> Node<K> {
    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    fn total(&self) -> usize {
        self.keys.len() + self.counts.iter().sum::<usize>()
    }
}

/// Ordered multiset with O(log n) insert, remove, rank and select.
#[derive(Clone, Debug)]
pub struct OrderStatBTree<K, const T: usize = 16> {
    nodes: Vec<Node<K>>,
    free: Vec<usize>,
    root: usize,
    len: usize,
}

impl<K, const T: usize> Default for OrderStatBTree<K, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, const T: usize> OrderStatBTree<K, T> {
    const MAX_KEYS: usize = 2 * T - 1;
    const MIN_KEYS: usize = T - 1;

    pub fn new() -> Self {
        const { assert!(T >= 2, "minimum degree T must be at least 2") };
        Self {
            nodes: vec![Node {
                keys: Vec::new(),
                children: Vec::new(),
                counts: Vec::new(),
            }],
            free: Vec::new(),
            root: 0,
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of levels (1 for a lone leaf).
    pub fn height(&self) -> usize {
        let mut h = 1;
        let mut id = self.root;
        while !self.nodes[id].is_leaf() {
            id = self.nodes[id].children[0];
            h += 1;
        }
        h
    }

    /// Number of live nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len() - self.free.len()
    }

    fn alloc(&mut self, node: Node<K>) -> usize {
        if let Some(id) = self.free.pop() {
            self.nodes[id] = node;
            id
        } else {
            self.nodes.push(node);
            self.nodes.len() - 1
        }
    }

    fn release(&mut self, id: usize) {
        self.nodes[id] = Node {
            keys: Vec::new(),
            children: Vec::new(),
            counts: Vec::new(),
        };
        self.free.push(id);
    }

    // ---------------------------------------------------------------- queries

    /// The `idx`-th smallest key (0-based).
    pub fn select(&self, idx: usize) -> Option<&K> {
        if idx >= self.len {
            return None;
        }
        let mut id = self.root;
        let mut idx = idx;
        loop {
            let n = &self.nodes[id];
            if n.is_leaf() {
                return Some(&n.keys[idx]);
            }
            let mut next = None;
            for i in 0..n.children.len() {
                let c = n.counts[i];
                if idx < c {
                    next = Some(n.children[i]);
                    break;
                }
                idx -= c;
                if i < n.keys.len() {
                    if idx == 0 {
                        return Some(&n.keys[i]);
                    }
                    idx -= 1;
                }
            }
            id = next.expect("counts inconsistent with len");
        }
    }

    /// Alias of [`select`](Self::select).
    pub fn nth(&self, idx: usize) -> Option<&K> {
        self.select(idx)
    }

    pub fn min(&self) -> Option<&K> {
        self.select(0)
    }

    pub fn max(&self) -> Option<&K> {
        self.len.checked_sub(1).and_then(|i| self.select(i))
    }

    fn rank_by(&self, before: impl Fn(&K) -> bool) -> usize {
        let mut id = self.root;
        let mut r = 0;
        loop {
            let n = &self.nodes[id];
            let i = n.keys.partition_point(&before);
            if n.is_leaf() {
                return r + i;
            }
            r += i + n.counts[..i].iter().sum::<usize>();
            id = n.children[i];
        }
    }

    /// Elements strictly less than `key`.
    pub fn rank(&self, key: &K) -> usize
    where
        K: Ord,
    {
        self.rank_by(|k| k < key)
    }

    /// Elements less than or equal to `key`.
    pub fn rank_upper(&self, key: &K) -> usize
    where
        K: Ord,
    {
        self.rank_by(|k| k <= key)
    }

    /// Multiplicity of `key`.
    pub fn count(&self, key: &K) -> usize
    where
        K: Ord,
    {
        self.rank_upper(key) - self.rank(key)
    }

    pub fn contains(&self, key: &K) -> bool
    where
        K: Ord,
    {
        self.count(key) > 0
    }

    /// Largest element strictly below `key`.
    pub fn predecessor(&self, key: &K) -> Option<&K>
    where
        K: Ord,
    {
        self.rank(key).checked_sub(1).and_then(|i| self.select(i))
    }

    /// Smallest element strictly above `key`.
    pub fn successor(&self, key: &K) -> Option<&K>
    where
        K: Ord,
    {
        self.select(self.rank_upper(key))
    }

    fn rank_range<R: RangeBounds<K>>(&self, range: &R) -> (usize, usize)
    where
        K: Ord,
    {
        let start = match range.start_bound() {
            Bound::Included(a) => self.rank(a),
            Bound::Excluded(a) => self.rank_upper(a),
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(b) => self.rank_upper(b),
            Bound::Excluded(b) => self.rank(b),
            Bound::Unbounded => self.len,
        };
        (start, end.max(start))
    }

    /// Number of elements inside `range`, in O(log n).
    pub fn count_range<R: RangeBounds<K>>(&self, range: R) -> usize
    where
        K: Ord,
    {
        let (s, e) = self.rank_range(&range);
        e - s
    }

    /// In-order iterator over all keys.
    pub fn iter(&self) -> Iter<'_, K, T> {
        self.iter_from_rank(0, self.len)
    }

    /// In-order iterator over the keys inside `range`.
    pub fn range<R: RangeBounds<K>>(&self, range: R) -> Iter<'_, K, T>
    where
        K: Ord,
    {
        let (s, e) = self.rank_range(&range);
        self.iter_from_rank(s, e - s)
    }

    fn iter_from_rank(&self, rank: usize, take: usize) -> Iter<'_, K, T> {
        let mut stack = Vec::new();
        let mut idx = rank.min(self.len);
        let mut id = self.root;
        loop {
            let n = &self.nodes[id];
            if n.is_leaf() {
                stack.push((id, idx));
                break;
            }
            let mut descended = false;
            for i in 0..n.children.len() {
                let c = n.counts[i];
                // The last child also absorbs `idx == len` (an empty tail).
                if idx < c || i + 1 == n.children.len() {
                    stack.push((id, i));
                    id = n.children[i];
                    descended = true;
                    break;
                }
                idx -= c;
                if i < n.keys.len() {
                    if idx == 0 {
                        stack.push((id, i));
                        return Iter {
                            tree: self,
                            stack,
                            remaining: take,
                        };
                    }
                    idx -= 1;
                }
            }
            assert!(descended, "counts inconsistent with len");
        }
        Iter {
            tree: self,
            stack,
            remaining: take,
        }
    }

    // ---------------------------------------------------------------- insert

    /// Insert a key (after any equal keys).
    pub fn insert(&mut self, key: K)
    where
        K: Ord,
    {
        if let Some((mid, right)) = self.insert_rec(self.root, key) {
            let left = self.root;
            let lc = self.nodes[left].total();
            let rc = self.nodes[right].total();
            self.root = self.alloc(Node {
                keys: vec![mid],
                children: vec![left, right],
                counts: vec![lc, rc],
            });
        }
        self.len += 1;
    }

    fn insert_rec(&mut self, id: usize, key: K) -> Option<(K, usize)>
    where
        K: Ord,
    {
        let pos = self.nodes[id].keys.partition_point(|k| *k <= key);
        if self.nodes[id].is_leaf() {
            self.nodes[id].keys.insert(pos, key);
        } else {
            let child = self.nodes[id].children[pos];
            self.nodes[id].counts[pos] += 1;
            if let Some((mid, right)) = self.insert_rec(child, key) {
                let lc = self.nodes[child].total();
                let rc = self.nodes[right].total();
                let n = &mut self.nodes[id];
                n.keys.insert(pos, mid);
                n.children.insert(pos + 1, right);
                n.counts[pos] = lc;
                n.counts.insert(pos + 1, rc);
            }
        }
        (self.nodes[id].keys.len() > Self::MAX_KEYS).then(|| self.split(id))
    }

    /// Split an overfull node (`2T` keys) into `T` keys | median | `T-1` keys.
    fn split(&mut self, id: usize) -> (K, usize) {
        let node = &mut self.nodes[id];
        let mut right_keys = node.keys.split_off(T);
        let mid = right_keys.remove(0);
        let (children, counts) = if node.is_leaf() {
            (Vec::new(), Vec::new())
        } else {
            (node.children.split_off(T + 1), node.counts.split_off(T + 1))
        };
        let right = self.alloc(Node {
            keys: right_keys,
            children,
            counts,
        });
        (mid, right)
    }

    // ---------------------------------------------------------------- remove

    /// Remove one occurrence of `key`, returning it.
    pub fn remove(&mut self, key: &K) -> Option<K>
    where
        K: Ord,
    {
        let r = self.rank(key);
        if self.select(r).is_some_and(|k| k == key) {
            self.remove_at(r)
        } else {
            None
        }
    }

    /// Remove and return the `idx`-th smallest key.
    pub fn remove_at(&mut self, idx: usize) -> Option<K> {
        if idx >= self.len {
            return None;
        }
        let k = self.remove_at_rec(self.root, idx);
        self.len -= 1;
        let root = &self.nodes[self.root];
        if !root.is_leaf() && root.keys.is_empty() {
            let old = self.root;
            self.root = self.nodes[old].children[0];
            self.release(old);
        }
        Some(k)
    }

    fn remove_at_rec(&mut self, id: usize, mut idx: usize) -> K {
        if self.nodes[id].is_leaf() {
            return self.nodes[id].keys.remove(idx);
        }
        let nchild = self.nodes[id].children.len();
        for i in 0..nchild {
            let c = self.nodes[id].counts[i];
            let child = self.nodes[id].children[i];
            if idx < c {
                let k = self.remove_at_rec(child, idx);
                self.nodes[id].counts[i] -= 1;
                self.fix_child(id, i);
                return k;
            }
            idx -= c;
            if i < nchild - 1 {
                if idx == 0 {
                    // Target is the separator: swap in its in-order predecessor.
                    let pred = self.remove_at_rec(child, c - 1);
                    self.nodes[id].counts[i] -= 1;
                    let k = mem::replace(&mut self.nodes[id].keys[i], pred);
                    self.fix_child(id, i);
                    return k;
                }
                idx -= 1;
            }
        }
        unreachable!("counts inconsistent with len")
    }

    /// Restore the minimum-fill invariant of `children[i]` after a removal.
    fn fix_child(&mut self, id: usize, i: usize) {
        let child = self.nodes[id].children[i];
        if self.nodes[child].keys.len() >= Self::MIN_KEYS {
            return;
        }
        let nch = self.nodes[id].children.len();
        if i > 0 {
            let left = self.nodes[id].children[i - 1];
            if self.nodes[left].keys.len() > Self::MIN_KEYS {
                return self.borrow_from_left(id, i);
            }
        }
        if i + 1 < nch {
            let right = self.nodes[id].children[i + 1];
            if self.nodes[right].keys.len() > Self::MIN_KEYS {
                return self.borrow_from_right(id, i);
            }
        }
        self.merge(id, if i > 0 { i - 1 } else { i });
    }

    fn borrow_from_left(&mut self, id: usize, i: usize) {
        let (lid, cid) = (self.nodes[id].children[i - 1], self.nodes[id].children[i]);
        let mut left = mem::take(&mut self.nodes[lid]);
        let mut child = mem::take(&mut self.nodes[cid]);
        let last = left.keys.pop().expect("left sibling has spare key");
        let sep = mem::replace(&mut self.nodes[id].keys[i - 1], last);
        child.keys.insert(0, sep);
        let mut moved = 1;
        if !left.is_leaf() {
            let c = left.children.pop().expect("internal node has children");
            let n = left.counts.pop().expect("internal node has counts");
            child.children.insert(0, c);
            child.counts.insert(0, n);
            moved += n;
        }
        let parent = &mut self.nodes[id];
        parent.counts[i - 1] -= moved;
        parent.counts[i] += moved;
        self.nodes[lid] = left;
        self.nodes[cid] = child;
    }

    fn borrow_from_right(&mut self, id: usize, i: usize) {
        let (cid, rid) = (self.nodes[id].children[i], self.nodes[id].children[i + 1]);
        let mut child = mem::take(&mut self.nodes[cid]);
        let mut right = mem::take(&mut self.nodes[rid]);
        let first = right.keys.remove(0);
        let sep = mem::replace(&mut self.nodes[id].keys[i], first);
        child.keys.push(sep);
        let mut moved = 1;
        if !right.is_leaf() {
            let c = right.children.remove(0);
            let n = right.counts.remove(0);
            child.children.push(c);
            child.counts.push(n);
            moved += n;
        }
        let parent = &mut self.nodes[id];
        parent.counts[i] += moved;
        parent.counts[i + 1] -= moved;
        self.nodes[cid] = child;
        self.nodes[rid] = right;
    }

    /// Merge `children[i]`, separator `keys[i]` and `children[i+1]`.
    fn merge(&mut self, id: usize, i: usize) {
        let (lid, rid) = (self.nodes[id].children[i], self.nodes[id].children[i + 1]);
        let mut right = mem::take(&mut self.nodes[rid]);
        let parent = &mut self.nodes[id];
        let sep = parent.keys.remove(i);
        parent.children.remove(i + 1);
        let rcount = parent.counts.remove(i + 1);
        parent.counts[i] += rcount + 1;
        let left = &mut self.nodes[lid];
        left.keys.push(sep);
        left.keys.append(&mut right.keys);
        left.children.append(&mut right.children);
        left.counts.append(&mut right.counts);
        self.free.push(rid);
    }

    // ------------------------------------------------------------- bulk load

    /// Build a perfectly packed tree from already-sorted keys in O(n).
    ///
    /// # Panics
    /// If `keys` is not sorted (non-decreasing).
    pub fn from_sorted(keys: Vec<K>) -> Self
    where
        K: Ord,
    {
        assert!(
            keys.windows(2).all(|w| w[0] <= w[1]),
            "from_sorted requires sorted input"
        );
        let mut t = Self::new();
        let n = keys.len();
        if n == 0 {
            return t;
        }
        // Smallest height whose maximum capacity ((2T)^(h+1) - 1) holds n keys.
        let mut h = 0;
        while Self::max_size(h) < n as u128 {
            h += 1;
        }
        t.nodes.clear();
        let mut it = keys.into_iter();
        t.root = t.build(&mut it, n, h, true);
        t.len = n;
        t
    }

    /// Max keys in a subtree of the given height: `(2T)^(h+1) - 1`.
    fn max_size(h: u32) -> u128 {
        (2 * T as u128).saturating_pow(h + 1) - 1
    }

    fn build(
        &mut self,
        it: &mut std::vec::IntoIter<K>,
        size: usize,
        height: u32,
        is_root: bool,
    ) -> usize {
        if height == 0 {
            let keys: Vec<K> = it.by_ref().take(size).collect();
            return self.alloc(Node {
                keys,
                children: Vec::new(),
                counts: Vec::new(),
            });
        }
        let cap = Self::max_size(height - 1) + 1; // (2T)^height
        let lower = if is_root { 2 } else { T };
        let c = (size as u128 + 1).div_ceil(cap).max(lower as u128) as usize;
        let rest = size - (c - 1);
        let (base, extra) = (rest / c, rest % c);
        let mut keys = Vec::with_capacity(c - 1);
        let mut children = Vec::with_capacity(c);
        let mut counts = Vec::with_capacity(c);
        for j in 0..c {
            let s = base + usize::from(j < extra);
            children.push(self.build(it, s, height - 1, false));
            counts.push(s);
            if j + 1 < c {
                keys.push(it.next().expect("size accounting"));
            }
        }
        self.alloc(Node {
            keys,
            children,
            counts,
        })
    }

    // ------------------------------------------------------------ invariants

    /// Verify every structural invariant; `Err` names the first violation.
    pub fn check_invariants(&self) -> Result<(), String>
    where
        K: Ord,
    {
        let mut seen = 0;
        let (count, depth) = self.check_node(self.root, true, &mut seen)?;
        if count != self.len {
            return Err(format!("len {} but root subtree holds {count}", self.len));
        }
        if seen + self.free.len() != self.nodes.len() {
            return Err(format!(
                "arena leak: {seen} reachable + {} free != {} slots",
                self.free.len(),
                self.nodes.len()
            ));
        }
        let _ = depth;
        let mut prev: Option<&K> = None;
        for k in self.iter() {
            if prev.is_some_and(|p| p > k) {
                return Err("in-order traversal not sorted".into());
            }
            prev = Some(k);
        }
        if self.iter().count() != self.len {
            return Err("iterator length mismatch".into());
        }
        Ok(())
    }

    fn check_node(
        &self,
        id: usize,
        is_root: bool,
        seen: &mut usize,
    ) -> Result<(usize, usize), String> {
        *seen += 1;
        let n = &self.nodes[id];
        if n.keys.len() > Self::MAX_KEYS {
            return Err(format!("node {id} overfull: {}", n.keys.len()));
        }
        if !is_root && n.keys.len() < Self::MIN_KEYS {
            return Err(format!("node {id} underfull: {}", n.keys.len()));
        }
        if n.is_leaf() {
            return Ok((n.keys.len(), 1));
        }
        if is_root && n.keys.is_empty() {
            return Err("internal root without keys".into());
        }
        if n.children.len() != n.keys.len() + 1 || n.counts.len() != n.children.len() {
            return Err(format!("node {id} has inconsistent arities"));
        }
        let mut total = n.keys.len();
        let mut depth = None;
        for (i, &c) in n.children.iter().enumerate() {
            let (cnt, d) = self.check_node(c, false, seen)?;
            if cnt != n.counts[i] {
                return Err(format!(
                    "node {id} child {i}: stored count {} != actual {cnt}",
                    n.counts[i]
                ));
            }
            if *depth.get_or_insert(d) != d {
                return Err(format!("node {id}: leaves at different depths"));
            }
            total += cnt;
        }
        Ok((total, depth.unwrap_or(0) + 1))
    }
}

/// In-order iterator; see [`OrderStatBTree::iter`].
pub struct Iter<'a, K, const T: usize> {
    tree: &'a OrderStatBTree<K, T>,
    /// `(node, i)`: next key to yield in `node` is `keys[i]`.
    stack: Vec<(usize, usize)>,
    remaining: usize,
}

impl<'a, K, const T: usize> Iterator for Iter<'a, K, T> {
    type Item = &'a K;

    fn next(&mut self) -> Option<&'a K> {
        if self.remaining == 0 {
            return None;
        }
        loop {
            let (id, i) = *self.stack.last()?;
            let n = &self.tree.nodes[id];
            if i >= n.keys.len() {
                self.stack.pop();
                continue;
            }
            self.stack.last_mut()?.1 += 1;
            if !n.is_leaf() {
                let mut c = n.children[i + 1];
                loop {
                    self.stack.push((c, 0));
                    let cn = &self.tree.nodes[c];
                    if cn.is_leaf() {
                        break;
                    }
                    c = cn.children[0];
                }
            }
            self.remaining -= 1;
            return Some(&n.keys[i]);
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K, const T: usize> ExactSizeIterator for Iter<'_, K, T> {}

impl<K: Ord, const T: usize> FromIterator<K> for OrderStatBTree<K, T> {
    fn from_iter<I: IntoIterator<Item = K>>(iter: I) -> Self {
        let mut t = Self::new();
        t.extend(iter);
        t
    }
}

impl<K: Ord, const T: usize> Extend<K> for OrderStatBTree<K, T> {
    fn extend<I: IntoIterator<Item = K>>(&mut self, iter: I) {
        for k in iter {
            self.insert(k);
        }
    }
}

impl<K: Ord + Clone, const T: usize> OrderStatistics<K> for OrderStatBTree<K, T> {
    fn len(&self) -> usize {
        self.len
    }
    fn insert(&mut self, key: K) {
        OrderStatBTree::insert(self, key);
    }
    fn remove(&mut self, key: &K) -> bool {
        OrderStatBTree::remove(self, key).is_some()
    }
    fn rank(&self, key: &K) -> usize {
        OrderStatBTree::rank(self, key)
    }
    fn select(&self, idx: usize) -> Option<K> {
        OrderStatBTree::select(self, idx).cloned()
    }
    fn validate(&self) -> Result<(), String> {
        self.check_invariants()
    }
}

//! Arena AVL tree with a subtree-size field (CLRS ch. 14 augmentation).

use std::cmp::Ordering;

use crate::OrderStatistics;

const NIL: usize = usize::MAX;

#[derive(Clone, Debug)]
struct Node<K> {
    key: K,
    left: usize,
    right: usize,
    height: i32,
    size: usize,
}

#[derive(Clone, Debug)]
pub struct AvlTree<K> {
    nodes: Vec<Node<K>>,
    free: Vec<usize>,
    root: usize,
}

impl<K> Default for AvlTree<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K> AvlTree<K> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            free: Vec::new(),
            root: NIL,
        }
    }

    fn height(&self, i: usize) -> i32 {
        if i == NIL { 0 } else { self.nodes[i].height }
    }

    fn size(&self, i: usize) -> usize {
        if i == NIL { 0 } else { self.nodes[i].size }
    }

    fn update(&mut self, i: usize) {
        let (l, r) = (self.nodes[i].left, self.nodes[i].right);
        self.nodes[i].height = 1 + self.height(l).max(self.height(r));
        self.nodes[i].size = 1 + self.size(l) + self.size(r);
    }

    fn rotate_right(&mut self, y: usize) -> usize {
        let x = self.nodes[y].left;
        self.nodes[y].left = self.nodes[x].right;
        self.nodes[x].right = y;
        self.update(y);
        self.update(x);
        x
    }

    fn rotate_left(&mut self, x: usize) -> usize {
        let y = self.nodes[x].right;
        self.nodes[x].right = self.nodes[y].left;
        self.nodes[y].left = x;
        self.update(x);
        self.update(y);
        y
    }

    fn balance(&mut self, i: usize) -> usize {
        self.update(i);
        let bf = self.height(self.nodes[i].left) - self.height(self.nodes[i].right);
        if bf > 1 {
            let l = self.nodes[i].left;
            if self.height(self.nodes[l].left) < self.height(self.nodes[l].right) {
                self.nodes[i].left = self.rotate_left(l);
            }
            self.rotate_right(i)
        } else if bf < -1 {
            let r = self.nodes[i].right;
            if self.height(self.nodes[r].right) < self.height(self.nodes[r].left) {
                self.nodes[i].right = self.rotate_right(r);
            }
            self.rotate_left(i)
        } else {
            i
        }
    }

    fn alloc(&mut self, key: K) -> usize {
        let node = Node {
            key,
            left: NIL,
            right: NIL,
            height: 1,
            size: 1,
        };
        if let Some(i) = self.free.pop() {
            self.nodes[i] = node;
            i
        } else {
            self.nodes.push(node);
            self.nodes.len() - 1
        }
    }

    pub fn len(&self) -> usize {
        self.size(self.root)
    }

    pub fn is_empty(&self) -> bool {
        self.root == NIL
    }

    pub fn insert(&mut self, key: K)
    where
        K: Ord,
    {
        self.root = self.insert_rec(self.root, key);
    }

    fn insert_rec(&mut self, i: usize, key: K) -> usize
    where
        K: Ord,
    {
        if i == NIL {
            return self.alloc(key);
        }
        if key < self.nodes[i].key {
            let l = self.nodes[i].left;
            self.nodes[i].left = self.insert_rec(l, key);
        } else {
            let r = self.nodes[i].right;
            self.nodes[i].right = self.insert_rec(r, key);
        }
        self.balance(i)
    }

    /// Remove one occurrence of `key`; `false` if absent.
    pub fn remove(&mut self, key: &K) -> bool
    where
        K: Ord,
    {
        let mut removed = false;
        self.root = self.remove_rec(self.root, key, &mut removed);
        removed
    }

    fn remove_rec(&mut self, i: usize, key: &K, removed: &mut bool) -> usize
    where
        K: Ord,
    {
        if i == NIL {
            return NIL;
        }
        match key.cmp(&self.nodes[i].key) {
            Ordering::Less => {
                let l = self.nodes[i].left;
                self.nodes[i].left = self.remove_rec(l, key, removed);
            }
            Ordering::Greater => {
                let r = self.nodes[i].right;
                self.nodes[i].right = self.remove_rec(r, key, removed);
            }
            Ordering::Equal => {
                *removed = true;
                let (l, r) = (self.nodes[i].left, self.nodes[i].right);
                self.free.push(i);
                if l == NIL {
                    return r;
                }
                if r == NIL {
                    return l;
                }
                // Splice the in-order successor into this position.
                let (new_r, m) = self.detach_min(r);
                self.nodes[m].left = l;
                self.nodes[m].right = new_r;
                return self.balance(m);
            }
        }
        self.balance(i)
    }

    fn detach_min(&mut self, i: usize) -> (usize, usize) {
        let l = self.nodes[i].left;
        if l == NIL {
            return (self.nodes[i].right, i);
        }
        let (nl, m) = self.detach_min(l);
        self.nodes[i].left = nl;
        (self.balance(i), m)
    }

    pub fn rank(&self, key: &K) -> usize
    where
        K: Ord,
    {
        let (mut i, mut r) = (self.root, 0);
        while i != NIL {
            if self.nodes[i].key < *key {
                r += self.size(self.nodes[i].left) + 1;
                i = self.nodes[i].right;
            } else {
                i = self.nodes[i].left;
            }
        }
        r
    }

    pub fn select(&self, idx: usize) -> Option<&K> {
        if idx >= self.len() {
            return None;
        }
        let (mut i, mut idx) = (self.root, idx);
        loop {
            let ls = self.size(self.nodes[i].left);
            match idx.cmp(&ls) {
                Ordering::Less => i = self.nodes[i].left,
                Ordering::Equal => return Some(&self.nodes[i].key),
                Ordering::Greater => {
                    idx -= ls + 1;
                    i = self.nodes[i].right;
                }
            }
        }
    }

    pub fn check_invariants(&self) -> Result<(), String>
    where
        K: Ord,
    {
        fn go<K: Ord>(t: &AvlTree<K>, i: usize, live: &mut usize) -> Result<(i32, usize), String> {
            if i == NIL {
                return Ok((0, 0));
            }
            *live += 1;
            let n = &t.nodes[i];
            let (hl, sl) = go(t, n.left, live)?;
            let (hr, sr) = go(t, n.right, live)?;
            if (hl - hr).abs() > 1 {
                return Err(format!("node {i} unbalanced ({hl} vs {hr})"));
            }
            if n.height != 1 + hl.max(hr) || n.size != 1 + sl + sr {
                return Err(format!("node {i} stale height/size"));
            }
            if n.left != NIL && t.nodes[n.left].key > n.key {
                return Err(format!("node {i} left child larger"));
            }
            if n.right != NIL && t.nodes[n.right].key < n.key {
                return Err(format!("node {i} right child smaller"));
            }
            Ok((n.height, n.size))
        }
        let mut live = 0;
        go(self, self.root, &mut live)?;
        if live + self.free.len() != self.nodes.len() {
            return Err("arena leak".into());
        }
        Ok(())
    }
}

impl<K: Ord + Clone> OrderStatistics<K> for AvlTree<K> {
    fn len(&self) -> usize {
        AvlTree::len(self)
    }
    fn insert(&mut self, key: K) {
        AvlTree::insert(self, key);
    }
    fn remove(&mut self, key: &K) -> bool {
        AvlTree::remove(self, key)
    }
    fn rank(&self, key: &K) -> usize {
        AvlTree::rank(self, key)
    }
    fn select(&self, idx: usize) -> Option<K> {
        AvlTree::select(self, idx).cloned()
    }
    fn validate(&self) -> Result<(), String> {
        self.check_invariants()
    }
}

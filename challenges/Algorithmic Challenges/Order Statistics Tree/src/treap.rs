//! Arena treap (randomised BST) with subtree sizes, via split/merge.

use crate::OrderStatistics;
use crate::rng::SplitMix64;

const NIL: usize = usize::MAX;

#[derive(Clone, Debug)]
struct Node<K> {
    key: K,
    prio: u64,
    left: usize,
    right: usize,
    size: usize,
}

#[derive(Clone, Debug)]
pub struct Treap<K> {
    nodes: Vec<Node<K>>,
    free: Vec<usize>,
    root: usize,
    rng: SplitMix64,
}

impl<K> Default for Treap<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K> Treap<K> {
    pub fn new() -> Self {
        Self::with_seed(0x5EED)
    }

    pub fn with_seed(seed: u64) -> Self {
        Self {
            nodes: Vec::new(),
            free: Vec::new(),
            root: NIL,
            rng: SplitMix64::new(seed),
        }
    }

    fn size(&self, i: usize) -> usize {
        if i == NIL { 0 } else { self.nodes[i].size }
    }

    fn update(&mut self, i: usize) {
        self.nodes[i].size = 1 + self.size(self.nodes[i].left) + self.size(self.nodes[i].right);
    }

    fn alloc(&mut self, key: K) -> usize {
        let node = Node {
            key,
            prio: self.rng.next_u64(),
            left: NIL,
            right: NIL,
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

    /// Split into (keys `<` / `<=` `key`, rest).
    fn split(&mut self, t: usize, key: &K, inclusive: bool) -> (usize, usize)
    where
        K: Ord,
    {
        if t == NIL {
            return (NIL, NIL);
        }
        let goes_left = if inclusive {
            self.nodes[t].key <= *key
        } else {
            self.nodes[t].key < *key
        };
        if goes_left {
            let (a, b) = self.split(self.nodes[t].right, key, inclusive);
            self.nodes[t].right = a;
            self.update(t);
            (t, b)
        } else {
            let (a, b) = self.split(self.nodes[t].left, key, inclusive);
            self.nodes[t].left = b;
            self.update(t);
            (a, t)
        }
    }

    fn merge(&mut self, a: usize, b: usize) -> usize {
        if a == NIL {
            return b;
        }
        if b == NIL {
            return a;
        }
        if self.nodes[a].prio > self.nodes[b].prio {
            let r = self.merge(self.nodes[a].right, b);
            self.nodes[a].right = r;
            self.update(a);
            a
        } else {
            let l = self.merge(a, self.nodes[b].left);
            self.nodes[b].left = l;
            self.update(b);
            b
        }
    }

    pub fn insert(&mut self, key: K)
    where
        K: Ord,
    {
        let (a, b) = self.split(self.root, &key, true);
        let n = self.alloc(key);
        let left = self.merge(a, n);
        self.root = self.merge(left, b);
    }

    pub fn remove(&mut self, key: &K) -> bool
    where
        K: Ord,
    {
        let (a, rest) = self.split(self.root, key, false);
        let (m, b) = self.split(rest, key, true);
        if m == NIL {
            self.root = self.merge(a, b);
            return false;
        }
        let (l, r) = (self.nodes[m].left, self.nodes[m].right);
        self.free.push(m);
        let m2 = self.merge(l, r);
        let right = self.merge(m2, b);
        self.root = self.merge(a, right);
        true
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
            if idx < ls {
                i = self.nodes[i].left;
            } else if idx == ls {
                return Some(&self.nodes[i].key);
            } else {
                idx -= ls + 1;
                i = self.nodes[i].right;
            }
        }
    }

    pub fn check_invariants(&self) -> Result<(), String>
    where
        K: Ord,
    {
        fn go<K: Ord>(t: &Treap<K>, i: usize, live: &mut usize) -> Result<usize, String> {
            if i == NIL {
                return Ok(0);
            }
            *live += 1;
            let n = &t.nodes[i];
            let size = 1 + go(t, n.left, live)? + go(t, n.right, live)?;
            if n.size != size {
                return Err(format!("node {i} stale size"));
            }
            for c in [n.left, n.right] {
                if c != NIL && t.nodes[c].prio > n.prio {
                    return Err(format!("node {i} violates heap order"));
                }
            }
            if n.left != NIL && t.nodes[n.left].key > n.key {
                return Err(format!("node {i} left child larger"));
            }
            if n.right != NIL && t.nodes[n.right].key < n.key {
                return Err(format!("node {i} right child smaller"));
            }
            Ok(size)
        }
        let mut live = 0;
        go(self, self.root, &mut live)?;
        if live + self.free.len() != self.nodes.len() {
            return Err("arena leak".into());
        }
        Ok(())
    }
}

impl<K: Ord + Clone> OrderStatistics<K> for Treap<K> {
    fn len(&self) -> usize {
        Treap::len(self)
    }
    fn insert(&mut self, key: K) {
        Treap::insert(self, key);
    }
    fn remove(&mut self, key: &K) -> bool {
        Treap::remove(self, key)
    }
    fn rank(&self, key: &K) -> usize {
        Treap::rank(self, key)
    }
    fn select(&self, idx: usize) -> Option<K> {
        Treap::select(self, idx).cloned()
    }
    fn validate(&self) -> Result<(), String> {
        self.check_invariants()
    }
}

//! Classic CLRS red-black tree (parent pointers, shared NIL sentinel at
//! index 0) augmented with subtree sizes exactly as in CLRS section 14.1.

use std::cmp::Ordering;

use crate::OrderStatistics;

const NIL: usize = 0;

#[derive(Clone, Debug)]
struct Node<K> {
    key: K,
    left: usize,
    right: usize,
    parent: usize,
    red: bool,
    size: usize,
}

#[derive(Clone, Debug)]
pub struct RedBlackTree<K> {
    nodes: Vec<Node<K>>,
    free: Vec<usize>,
    root: usize,
}

impl<K: Ord + Default> Default for RedBlackTree<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord + Default> RedBlackTree<K> {
    pub fn new() -> Self {
        Self {
            nodes: vec![Node {
                key: K::default(),
                left: NIL,
                right: NIL,
                parent: NIL,
                red: false,
                size: 0,
            }],
            free: Vec::new(),
            root: NIL,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes[self.root].size
    }

    pub fn is_empty(&self) -> bool {
        self.root == NIL
    }

    fn resize(&mut self, x: usize) {
        let (l, r) = (self.nodes[x].left, self.nodes[x].right);
        self.nodes[x].size = self.nodes[l].size + self.nodes[r].size + 1;
    }

    fn rotate_left(&mut self, x: usize) {
        let y = self.nodes[x].right;
        let yl = self.nodes[y].left;
        self.nodes[x].right = yl;
        if yl != NIL {
            self.nodes[yl].parent = x;
        }
        let p = self.nodes[x].parent;
        self.nodes[y].parent = p;
        if p == NIL {
            self.root = y;
        } else if x == self.nodes[p].left {
            self.nodes[p].left = y;
        } else {
            self.nodes[p].right = y;
        }
        self.nodes[y].left = x;
        self.nodes[x].parent = y;
        self.nodes[y].size = self.nodes[x].size;
        self.resize(x);
    }

    fn rotate_right(&mut self, x: usize) {
        let y = self.nodes[x].left;
        let yr = self.nodes[y].right;
        self.nodes[x].left = yr;
        if yr != NIL {
            self.nodes[yr].parent = x;
        }
        let p = self.nodes[x].parent;
        self.nodes[y].parent = p;
        if p == NIL {
            self.root = y;
        } else if x == self.nodes[p].right {
            self.nodes[p].right = y;
        } else {
            self.nodes[p].left = y;
        }
        self.nodes[y].right = x;
        self.nodes[x].parent = y;
        self.nodes[y].size = self.nodes[x].size;
        self.resize(x);
    }

    pub fn insert(&mut self, key: K) {
        let (mut y, mut x) = (NIL, self.root);
        while x != NIL {
            y = x;
            self.nodes[x].size += 1;
            x = if key < self.nodes[x].key {
                self.nodes[x].left
            } else {
                self.nodes[x].right
            };
        }
        let less = y != NIL && key < self.nodes[y].key;
        let node = Node {
            key,
            left: NIL,
            right: NIL,
            parent: y,
            red: true,
            size: 1,
        };
        let z = if let Some(i) = self.free.pop() {
            self.nodes[i] = node;
            i
        } else {
            self.nodes.push(node);
            self.nodes.len() - 1
        };
        if y == NIL {
            self.root = z;
        } else if less {
            self.nodes[y].left = z;
        } else {
            self.nodes[y].right = z;
        }
        self.insert_fixup(z);
    }

    fn insert_fixup(&mut self, mut z: usize) {
        while self.nodes[self.nodes[z].parent].red {
            let p = self.nodes[z].parent;
            let g = self.nodes[p].parent;
            let parent_is_left = p == self.nodes[g].left;
            let u = if parent_is_left {
                self.nodes[g].right
            } else {
                self.nodes[g].left
            };
            if self.nodes[u].red {
                self.nodes[p].red = false;
                self.nodes[u].red = false;
                self.nodes[g].red = true;
                z = g;
                continue;
            }
            if parent_is_left {
                if z == self.nodes[p].right {
                    z = p;
                    self.rotate_left(z);
                }
                let p = self.nodes[z].parent;
                let g = self.nodes[p].parent;
                self.nodes[p].red = false;
                self.nodes[g].red = true;
                self.rotate_right(g);
            } else {
                if z == self.nodes[p].left {
                    z = p;
                    self.rotate_right(z);
                }
                let p = self.nodes[z].parent;
                let g = self.nodes[p].parent;
                self.nodes[p].red = false;
                self.nodes[g].red = true;
                self.rotate_left(g);
            }
        }
        let r = self.root;
        self.nodes[r].red = false;
    }

    fn transplant(&mut self, u: usize, v: usize) {
        let p = self.nodes[u].parent;
        if p == NIL {
            self.root = v;
        } else if u == self.nodes[p].left {
            self.nodes[p].left = v;
        } else {
            self.nodes[p].right = v;
        }
        self.nodes[v].parent = p;
    }

    /// Decrement the size of `i` and all its ancestors.
    fn dec_path(&mut self, mut i: usize) {
        while i != NIL {
            self.nodes[i].size -= 1;
            i = self.nodes[i].parent;
        }
    }

    fn find(&self, key: &K) -> usize {
        let mut x = self.root;
        while x != NIL {
            match key.cmp(&self.nodes[x].key) {
                Ordering::Less => x = self.nodes[x].left,
                Ordering::Greater => x = self.nodes[x].right,
                Ordering::Equal => return x,
            }
        }
        NIL
    }

    /// Remove one occurrence of `key`; `false` if absent.
    pub fn remove(&mut self, key: &K) -> bool {
        let z = self.find(key);
        if z == NIL {
            return false;
        }
        let mut removed_black = !self.nodes[z].red;
        let x;
        if self.nodes[z].left == NIL {
            x = self.nodes[z].right;
            self.dec_path(self.nodes[z].parent);
            self.transplant(z, x);
        } else if self.nodes[z].right == NIL {
            x = self.nodes[z].left;
            self.dec_path(self.nodes[z].parent);
            self.transplant(z, x);
        } else {
            let mut y = self.nodes[z].right;
            while self.nodes[y].left != NIL {
                y = self.nodes[y].left;
            }
            removed_black = !self.nodes[y].red;
            x = self.nodes[y].right;
            // The path from y's parent to the root includes z.
            self.dec_path(self.nodes[y].parent);
            if self.nodes[y].parent == z {
                self.nodes[x].parent = y;
            } else {
                self.transplant(y, x);
                let zr = self.nodes[z].right;
                self.nodes[y].right = zr;
                self.nodes[zr].parent = y;
            }
            self.transplant(z, y);
            let zl = self.nodes[z].left;
            self.nodes[y].left = zl;
            self.nodes[zl].parent = y;
            self.nodes[y].red = self.nodes[z].red;
            self.nodes[y].size = self.nodes[z].size;
        }
        if removed_black {
            self.delete_fixup(x);
        }
        self.free.push(z);
        self.nodes[NIL].parent = NIL; // fixup may have parked a parent here
        true
    }

    fn delete_fixup(&mut self, mut x: usize) {
        while x != self.root && !self.nodes[x].red {
            let p = self.nodes[x].parent;
            let x_is_left = x == self.nodes[p].left;
            // Mirror-symmetric: "near" is x's side, "far" the opposite.
            let sibling = |t: &Self, p: usize| {
                if x_is_left {
                    t.nodes[p].right
                } else {
                    t.nodes[p].left
                }
            };
            let mut w = sibling(self, p);
            if self.nodes[w].red {
                self.nodes[w].red = false;
                self.nodes[p].red = true;
                if x_is_left {
                    self.rotate_left(p);
                } else {
                    self.rotate_right(p);
                }
                w = sibling(self, self.nodes[x].parent);
            }
            let (near, far) = if x_is_left {
                (self.nodes[w].left, self.nodes[w].right)
            } else {
                (self.nodes[w].right, self.nodes[w].left)
            };
            if !self.nodes[near].red && !self.nodes[far].red {
                self.nodes[w].red = true;
                x = self.nodes[x].parent;
                continue;
            }
            if !self.nodes[far].red {
                self.nodes[near].red = false;
                self.nodes[w].red = true;
                if x_is_left {
                    self.rotate_right(w);
                } else {
                    self.rotate_left(w);
                }
                w = sibling(self, self.nodes[x].parent);
            }
            let p = self.nodes[x].parent;
            self.nodes[w].red = self.nodes[p].red;
            self.nodes[p].red = false;
            let far = if x_is_left {
                self.nodes[w].right
            } else {
                self.nodes[w].left
            };
            self.nodes[far].red = false;
            if x_is_left {
                self.rotate_left(p);
            } else {
                self.rotate_right(p);
            }
            x = self.root;
        }
        self.nodes[x].red = false;
    }

    pub fn rank(&self, key: &K) -> usize {
        let (mut x, mut r) = (self.root, 0);
        while x != NIL {
            if self.nodes[x].key < *key {
                r += self.nodes[self.nodes[x].left].size + 1;
                x = self.nodes[x].right;
            } else {
                x = self.nodes[x].left;
            }
        }
        r
    }

    pub fn select(&self, idx: usize) -> Option<&K> {
        if idx >= self.len() {
            return None;
        }
        let (mut x, mut idx) = (self.root, idx);
        loop {
            let ls = self.nodes[self.nodes[x].left].size;
            if idx < ls {
                x = self.nodes[x].left;
            } else if idx == ls {
                return Some(&self.nodes[x].key);
            } else {
                idx -= ls + 1;
                x = self.nodes[x].right;
            }
        }
    }

    pub fn check_invariants(&self) -> Result<(), String> {
        fn go<K: Ord + Default>(
            t: &RedBlackTree<K>,
            x: usize,
            live: &mut usize,
        ) -> Result<(usize, usize), String> {
            if x == NIL {
                return Ok((1, 0)); // (black height, size)
            }
            *live += 1;
            let n = &t.nodes[x];
            if n.red && (t.nodes[n.left].red || t.nodes[n.right].red) {
                return Err(format!("red node {x} has red child"));
            }
            for c in [n.left, n.right] {
                if c != NIL && t.nodes[c].parent != x {
                    return Err(format!("node {c} has wrong parent"));
                }
            }
            if n.left != NIL && t.nodes[n.left].key > n.key {
                return Err(format!("node {x} left child larger"));
            }
            if n.right != NIL && t.nodes[n.right].key < n.key {
                return Err(format!("node {x} right child smaller"));
            }
            let (bl, sl) = go(t, n.left, live)?;
            let (br, sr) = go(t, n.right, live)?;
            if bl != br {
                return Err(format!("node {x} black-height mismatch"));
            }
            if n.size != 1 + sl + sr {
                return Err(format!("node {x} stale size"));
            }
            Ok((bl + usize::from(!n.red), n.size))
        }
        if self.nodes[self.root].red {
            return Err("red root".into());
        }
        if self.nodes[NIL].size != 0 || self.nodes[NIL].red {
            return Err("sentinel corrupted".into());
        }
        let mut live = 0;
        go(self, self.root, &mut live)?;
        if live + self.free.len() + 1 != self.nodes.len() {
            return Err("arena leak".into());
        }
        Ok(())
    }
}

impl<K: Ord + Default + Clone> OrderStatistics<K> for RedBlackTree<K> {
    fn len(&self) -> usize {
        RedBlackTree::len(self)
    }
    fn insert(&mut self, key: K) {
        RedBlackTree::insert(self, key);
    }
    fn remove(&mut self, key: &K) -> bool {
        RedBlackTree::remove(self, key)
    }
    fn rank(&self, key: &K) -> usize {
        RedBlackTree::rank(self, key)
    }
    fn select(&self, idx: usize) -> Option<K> {
        RedBlackTree::select(self, idx).cloned()
    }
    fn validate(&self) -> Result<(), String> {
        self.check_invariants()
    }
}

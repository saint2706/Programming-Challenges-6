//! Edge-weighted trees by subdividing every edge.
//!
//! Edge `e = (a, b)` becomes a new vertex `n + e` joined to `a` and `b`. Edge vertices carry the
//! weight; the original vertices are neutral, so they add nothing to an aggregate and ignore
//! `path_add`. A path between original vertices then covers exactly the edges on it, and the
//! subtree of `v` covers the edges strictly below `v` (the edge to its parent is an ancestor).
//! Any structure in this crate works unchanged, and nothing needs an inverse (min/max have none).

use crate::tree::{Layout, Tree, TreeError};
use crate::{MaybeSubtree, PathOps, Stats, SubtreeOps};

pub struct EdgeTree<S> {
    inner: S,
    vertices: usize,
}

impl<S: PathOps> EdgeTree<S> {
    /// `edges[e] = (a, b, weight)`. Edge indices are the positions in `edges`.
    pub fn new(n: usize, edges: &[(usize, usize, i64)], root: usize) -> Result<Self, TreeError> {
        let pairs: Vec<(usize, usize)> = edges.iter().map(|&(a, b, _)| (a, b)).collect();
        Tree::new(n, &pairs)?; // reports errors in the caller's vertex numbering
        if root >= n {
            return Err(TreeError::RootOutOfRange(root)); // `n..` are edge vertices, not roots
        }
        let mut subdivided = Vec::with_capacity(2 * pairs.len());
        for (e, &(a, b)) in pairs.iter().enumerate() {
            subdivided.push((a, n + e));
            subdivided.push((n + e, b));
        }
        let tree = Tree::new(n + pairs.len(), &subdivided)?;
        let mut values: Vec<Option<i64>> = vec![None; n];
        values.extend(edges.iter().map(|&(_, _, w)| Some(w)));
        Ok(EdgeTree {
            inner: S::build(Layout::new(&tree, root)?, &values),
            vertices: n,
        })
    }

    pub fn len(&self) -> usize {
        self.vertices
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// Aggregate over the edges on the path `u..v` (`Stats::NEUTRAL` when `u == v`).
    pub fn path_query(&mut self, u: usize, v: usize) -> Stats {
        self.inner.path_query(u, v)
    }

    pub fn path_add(&mut self, u: usize, v: usize, delta: i64) {
        self.inner.path_add(u, v, delta);
    }

    pub fn set_edge(&mut self, e: usize, weight: i64) {
        self.inner.set(self.vertices + e, weight);
    }

    pub fn edge_weight(&mut self, e: usize) -> Option<i64> {
        self.inner.get(self.vertices + e)
    }

    pub fn lca(&self, u: usize, v: usize) -> usize {
        self.inner.lca(u, v)
    }

    /// Every original vertex is two steps from its parent in the subdivided tree.
    pub fn kth_ancestor(&self, v: usize, k: usize) -> Option<usize> {
        self.inner.kth_ancestor(v, 2 * k)
    }
}

impl<S: SubtreeOps> EdgeTree<S> {
    /// Aggregate over the edges strictly below `v`.
    pub fn subtree_query(&mut self, v: usize) -> Stats {
        self.inner.subtree_query(v)
    }
}

impl<S: MaybeSubtree> EdgeTree<S> {
    pub fn maybe_subtree(&mut self, v: usize) -> Option<Stats> {
        self.inner.maybe_subtree(v)
    }
}

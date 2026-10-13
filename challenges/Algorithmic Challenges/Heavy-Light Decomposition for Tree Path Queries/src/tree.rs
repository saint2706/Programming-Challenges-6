//! Input trees and the rooted heavy-light layout every structure shares.
//!
//! The layout follows the usual recipe (cp-algorithms, Al.Cash's blog): one iterative pass for
//! parent/depth, subtree sizes bottom-up, the largest child as the heavy child, then a preorder that
//! visits the heavy child first. In that order every subtree is a contiguous range and every heavy
//! chain is contiguous too. All passes are iterative, so a path of a million vertices cannot
//! overflow the stack.

use std::fmt;

/// "No vertex" marker in `u32` index arrays.
pub const NONE: u32 = u32::MAX;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TreeError {
    Empty,
    TooLarge(usize),
    WrongEdgeCount { vertices: usize, edges: usize },
    VertexOutOfRange(usize),
    SelfLoop(usize),
    NotConnected,
    RootOutOfRange(usize),
}

impl fmt::Display for TreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TreeError::Empty => write!(f, "a tree needs at least one vertex"),
            TreeError::TooLarge(n) => write!(f, "{n} vertices is more than this crate indexes"),
            TreeError::WrongEdgeCount { vertices, edges } => {
                write!(
                    f,
                    "a tree on {vertices} vertices has {} edges, got {edges}",
                    vertices - 1
                )
            }
            TreeError::VertexOutOfRange(v) => write!(f, "vertex {v} is out of range"),
            TreeError::SelfLoop(v) => write!(f, "edge from vertex {v} to itself"),
            TreeError::NotConnected => {
                write!(f, "the edges contain a cycle, so the graph is not a tree")
            }
            TreeError::RootOutOfRange(r) => write!(f, "root {r} is out of range"),
        }
    }
}

impl std::error::Error for TreeError {}

/// An undirected tree in compressed-adjacency form.
#[derive(Clone, Debug)]
pub struct Tree {
    offsets: Vec<u32>,
    neighbors: Vec<u32>,
}

impl Tree {
    /// Validates that `edges` form a tree on `n` vertices (right count, in range, no self loops,
    /// connected) and builds the adjacency.
    pub fn new(n: usize, edges: &[(usize, usize)]) -> Result<Tree, TreeError> {
        if n == 0 {
            return Err(TreeError::Empty);
        }
        if n >= NONE as usize {
            return Err(TreeError::TooLarge(n));
        }
        if edges.len() != n - 1 {
            return Err(TreeError::WrongEdgeCount {
                vertices: n,
                edges: edges.len(),
            });
        }
        let mut degree = vec![0u32; n + 1];
        for &(a, b) in edges {
            for v in [a, b] {
                if v >= n {
                    return Err(TreeError::VertexOutOfRange(v));
                }
            }
            if a == b {
                return Err(TreeError::SelfLoop(a));
            }
            degree[a] += 1;
            degree[b] += 1;
        }
        let mut offsets = vec![0u32; n + 1];
        for v in 0..n {
            offsets[v + 1] = offsets[v] + degree[v];
        }
        let mut fill = offsets.clone();
        let mut neighbors = vec![0u32; 2 * edges.len()];
        for &(a, b) in edges {
            neighbors[fill[a] as usize] = b as u32;
            fill[a] += 1;
            neighbors[fill[b] as usize] = a as u32;
            fill[b] += 1;
        }
        let tree = Tree { offsets, neighbors };
        // n - 1 edges plus connected means acyclic.
        let mut seen = vec![false; n];
        let mut stack = vec![0u32];
        seen[0] = true;
        let mut reached = 0;
        while let Some(v) = stack.pop() {
            reached += 1;
            for &w in tree.neighbors(v as usize) {
                if !seen[w as usize] {
                    seen[w as usize] = true;
                    stack.push(w);
                }
            }
        }
        if reached != n {
            return Err(TreeError::NotConnected);
        }
        Ok(tree)
    }

    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn neighbors(&self, v: usize) -> &[u32] {
        &self.neighbors[self.offsets[v] as usize..self.offsets[v + 1] as usize]
    }
}

/// A rooted tree with its heavy-light decomposition.
#[derive(Clone, Debug)]
pub struct Layout {
    pub root: usize,
    pub parent: Vec<u32>,
    pub depth: Vec<u32>,
    pub size: Vec<u32>,
    /// Heavy child (largest subtree, smallest id on ties), or `NONE` for a leaf.
    pub heavy: Vec<u32>,
    /// Top vertex of the vertex's heavy chain.
    pub head: Vec<u32>,
    /// Chain id of the vertex. Ids grow in preorder, so a parent's chain id is below its child's.
    pub chain: Vec<u32>,
    /// Preorder position (heavy child first). The subtree of `v` is `pos[v] .. pos[v] + size[v]`.
    pub pos: Vec<u32>,
    /// Inverse of `pos`.
    pub order: Vec<u32>,
    pub chain_head: Vec<u32>,
    pub chain_len: Vec<u32>,
    light_offsets: Vec<u32>,
    light: Vec<u32>,
}

impl Layout {
    pub fn new(tree: &Tree, root: usize) -> Result<Layout, TreeError> {
        let n = tree.len();
        if root >= n {
            return Err(TreeError::RootOutOfRange(root));
        }
        let mut parent = vec![NONE; n];
        let mut depth = vec![0u32; n];
        let mut bfs = Vec::with_capacity(n);
        let mut stack = vec![root as u32];
        while let Some(v) = stack.pop() {
            bfs.push(v);
            for &w in tree.neighbors(v as usize) {
                if w != parent[v as usize] {
                    parent[w as usize] = v;
                    depth[w as usize] = depth[v as usize] + 1;
                    stack.push(w);
                }
            }
        }
        let mut size = vec![1u32; n];
        for &v in bfs.iter().rev() {
            let p = parent[v as usize];
            if p != NONE {
                size[p as usize] += size[v as usize];
            }
        }
        let mut heavy = vec![NONE; n];
        for &v in &bfs {
            let p = parent[v as usize];
            if p == NONE {
                continue;
            }
            let h = heavy[p as usize];
            if h == NONE
                || size[v as usize] > size[h as usize]
                || (size[v as usize] == size[h as usize] && v < h)
            {
                heavy[p as usize] = v;
            }
        }

        let mut head = vec![0u32; n];
        let mut chain = vec![0u32; n];
        let mut pos = vec![0u32; n];
        let mut order = Vec::with_capacity(n);
        let mut chain_head = Vec::new();
        let mut chain_len: Vec<u32> = Vec::new();
        let mut light_offsets = vec![0u32; n + 1];
        let mut stack = vec![root as u32];
        while let Some(v) = stack.pop() {
            let vi = v as usize;
            pos[vi] = order.len() as u32;
            order.push(v);
            let p = parent[vi];
            if p == NONE || heavy[p as usize] != v {
                chain[vi] = chain_head.len() as u32;
                head[vi] = v;
                chain_head.push(v);
                chain_len.push(0);
            } else {
                chain[vi] = chain[p as usize];
                head[vi] = head[p as usize];
            }
            chain_len[chain[vi] as usize] += 1;
            for &w in tree.neighbors(vi) {
                if w != p && w != heavy[vi] {
                    stack.push(w);
                }
            }
            if heavy[vi] != NONE {
                stack.push(heavy[vi]);
            }
        }
        // Light children of each vertex in CSR form, in vertex-id order.
        let mut counts = vec![0u32; n];
        for (v, &p) in parent.iter().enumerate() {
            if p != NONE && heavy[p as usize] != v as u32 {
                counts[p as usize] += 1;
            }
        }
        for (v, &count) in counts.iter().enumerate() {
            light_offsets[v + 1] = light_offsets[v] + count;
        }
        let mut light = vec![0u32; light_offsets[n] as usize];
        let mut fill = light_offsets.clone();
        for (v, &p) in parent.iter().enumerate() {
            if p != NONE && heavy[p as usize] != v as u32 {
                light[fill[p as usize] as usize] = v as u32;
                fill[p as usize] += 1;
            }
        }
        Ok(Layout {
            root,
            parent,
            depth,
            size,
            heavy,
            head,
            chain,
            pos,
            order,
            chain_head,
            chain_len,
            light_offsets,
            light,
        })
    }

    pub fn len(&self) -> usize {
        self.parent.len()
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// Position of `v` inside its chain (0 = chain head).
    #[inline]
    pub fn chain_pos(&self, v: usize) -> u32 {
        self.depth[v] - self.depth[self.head[v] as usize]
    }

    /// Light children of `v` (each is the head of its own chain).
    pub fn light_children(&self, v: usize) -> &[u32] {
        &self.light[self.light_offsets[v] as usize..self.light_offsets[v + 1] as usize]
    }

    /// Lowest common ancestor by hopping chains, deeper chain head first.
    pub fn lca(&self, mut u: usize, mut v: usize) -> usize {
        while self.head[u] != self.head[v] {
            let (hu, hv) = (self.head[u] as usize, self.head[v] as usize);
            if self.depth[hu] > self.depth[hv] {
                u = self.parent[hu] as usize;
            } else {
                v = self.parent[hv] as usize;
            }
        }
        if self.depth[u] < self.depth[v] { u } else { v }
    }

    /// The `k`-th ancestor of `v` (`k = 0` is `v`), or `None` above the root.
    pub fn kth_ancestor(&self, mut v: usize, mut k: usize) -> Option<usize> {
        loop {
            let up = (self.depth[v] - self.depth[self.head[v] as usize]) as usize;
            if k <= up {
                return Some(self.order[self.pos[v] as usize - k] as usize);
            }
            k -= up + 1;
            let p = self.parent[self.head[v] as usize];
            if p == NONE {
                return None;
            }
            v = p as usize;
        }
    }

    /// Number of light edges on the path from `v` to the root.
    pub fn light_edges_to_root(&self, mut v: usize) -> usize {
        let mut count = 0;
        loop {
            let p = self.parent[self.head[v] as usize];
            if p == NONE {
                return count;
            }
            count += 1;
            v = p as usize;
        }
    }
}

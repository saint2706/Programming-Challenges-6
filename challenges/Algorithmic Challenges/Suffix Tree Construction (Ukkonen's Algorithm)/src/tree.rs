//! The suffix tree that every builder produces, stored flat in preorder.
//!
//! Node 0 is the root and children are visited in symbol order, so reading the leaves left to
//! right gives the suffix array. A node is three `u32`s:
//!
//! * `depth`: the string depth, the length of the path label from the root to the *end* of the edge
//!   into this node;
//! * `start`: where the label of the edge into this node begins in the text;
//! * `end`: the preorder index one past its last descendant. A leaf has `end == self + 1`.
//!
//! The first child of an internal node `v` is `v + 1`; the next sibling of a child `c` is
//! `end[c]`. There are no child pointers, and a descent is a walk over a contiguous array. The leaf
//! for suffix `i` has depth `len - i`, so no leaf stores its suffix.
//!
//! Every builder ([`crate::ukkonen`], [`crate::pwotd`], [`SuffixTree::from_sa_lcp`]) yields this
//! same layout, and for the same text they yield *identical* trees, which is what the tests check.

use crate::text::Symbols;

/// One node of a [`SuffixTree`]; see the module docs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Node {
    pub start: u32,
    pub depth: u32,
    pub end: u32,
}

pub struct SuffixTree<S: Symbols> {
    text: S,
    nodes: Vec<Node>,
    /// `leaves_before[v]`: leaves with preorder index below `v`; one extra entry at the end.
    leaves_before: Vec<u32>,
}

/// Marks a node without a suffix link.
pub const NO_LINK: u32 = u32::MAX;

/// Where a pattern ends: the node whose incoming edge holds the last matched symbol.
pub type Locus = u32;

impl<S: Symbols> SuffixTree<S> {
    /// Wrap preorder nodes produced by a builder. The caller guarantees the layout.
    pub fn from_nodes(text: S, nodes: Vec<Node>) -> Self {
        let mut leaves_before = Vec::with_capacity(nodes.len() + 1);
        let mut seen = 0u32;
        for (v, node) in nodes.iter().enumerate() {
            leaves_before.push(seen);
            seen += u32::from(node.end as usize == v + 1);
        }
        leaves_before.push(seen);
        SuffixTree {
            text,
            nodes,
            leaves_before,
        }
    }

    /// Build from the suffix array of the whole terminated text (`sa.len() == text.len()`) and its
    /// LCP array, in O(n).
    ///
    /// The scan runs over the suffix array *backwards*. A bottom-up traversal of the LCP-interval
    /// tree emits nodes in post-order; doing it on the mirror image emits post-order with children
    /// right to left, and reversing that list is exactly the left-to-right preorder.
    pub fn from_sa_lcp(text: S, sa: &[u32], lcp: &[u32]) -> Self {
        let n = sa.len();
        assert_eq!(n, text.len(), "the suffix array must cover the terminator");
        assert_eq!(n, lcp.len());
        struct Open {
            depth: u32,
            opened_at: u32,
            rep: u32,
        }
        // (depth, representative suffix, subtree size) in mirrored post-order.
        let mut raw: Vec<(u32, u32, u32)> = Vec::with_capacity(2 * n);
        let mut stack = vec![Open {
            depth: 0,
            opened_at: 0,
            rep: 0,
        }];
        for i in 0..n {
            let j = n - 1 - i;
            if i > 0 {
                // Leaf `i - 1` (suffix-array index j + 1) is emitted; close what it left open.
                let h = lcp[j + 1];
                let mut first = (raw.len() as u32 - 1, sa[j + 1]);
                while stack.last().is_some_and(|t| t.depth > h) {
                    let top = stack.pop().expect("checked");
                    let size = raw.len() as u32 - top.opened_at + 1;
                    raw.push((top.depth, top.rep, size));
                    first = (top.opened_at, top.rep);
                }
                if stack.last().is_some_and(|t| t.depth < h) {
                    stack.push(Open {
                        depth: h,
                        opened_at: first.0,
                        rep: first.1,
                    });
                }
            }
            raw.push((n as u32 - sa[j], sa[j], 1));
        }
        while let Some(top) = stack.pop() {
            let size = raw.len() as u32 - top.opened_at + 1;
            raw.push((top.depth, top.rep, size));
        }
        // Reverse into preorder; `start` temporarily holds the representative suffix.
        let mut nodes: Vec<Node> = raw
            .iter()
            .rev()
            .enumerate()
            .map(|(pre, &(depth, rep, size))| Node {
                start: rep,
                depth,
                end: pre as u32 + size,
            })
            .collect();
        drop(raw);
        // Now that parents precede children, turn the representative into an edge start.
        let mut path: Vec<(u32, u32)> = Vec::new(); // (end, depth) of open ancestors
        for (v, node) in nodes.iter_mut().enumerate() {
            while path.last().is_some_and(|&(end, _)| end as usize <= v) {
                path.pop();
            }
            let parent_depth = path.last().map_or(0, |&(_, d)| d);
            node.start += parent_depth;
            if node.end as usize > v + 1 {
                path.push((node.end, node.depth));
            }
        }
        nodes[0].start = 0;
        SuffixTree::from_nodes(text, nodes)
    }

    // ----- shape ---------------------------------------------------------------------------

    pub fn text(&self) -> &S {
        &self.text
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Number of nodes, root and leaves included.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// One leaf per suffix, terminator included.
    pub fn leaf_count(&self) -> usize {
        *self.leaves_before.last().expect("never empty") as usize
    }

    pub fn is_leaf(&self, v: u32) -> bool {
        self.nodes[v as usize].end == v + 1
    }

    pub fn depth(&self, v: u32) -> u32 {
        self.nodes[v as usize].depth
    }

    /// Bytes held by the node and rank arrays (not the text).
    pub fn heap_bytes(&self) -> usize {
        self.nodes.len() * std::mem::size_of::<Node>() + self.leaves_before.len() * 4
    }

    /// The children of `v` in symbol order.
    pub fn children(&self, v: u32) -> Children<'_> {
        let node = self.nodes[v as usize];
        Children {
            nodes: &self.nodes,
            next: v + 1,
            end: node.end,
        }
    }

    fn child_with(&self, v: u32, sym: u32) -> Option<u32> {
        self.children(v)
            .find(|&c| self.text.sym(self.nodes[c as usize].start as usize) == sym)
    }

    // ----- queries -------------------------------------------------------------------------

    /// Follow `pattern` from the root. `Some(v)` means the pattern occurs and every occurrence is
    /// a leaf in the subtree of `v`.
    pub fn locate_node(&self, pattern: &[u8]) -> Option<Locus> {
        let m = pattern.len();
        let (mut v, mut matched) = (0u32, 0usize);
        while matched < m {
            let c = self.child_with(v, self.text.byte_sym(pattern[matched]))?;
            let node = self.nodes[c as usize];
            let edge = (node.depth - self.nodes[v as usize].depth) as usize;
            let take = edge.min(m - matched);
            for t in 1..take {
                let want = self.text.byte_sym(pattern[matched + t]);
                if self.text.sym(node.start as usize + t) != want {
                    return None;
                }
            }
            matched += take;
            v = c;
        }
        Some(v)
    }

    pub fn contains(&self, pattern: &[u8]) -> bool {
        self.locate_node(pattern).is_some()
    }

    /// Number of occurrences, in O(|pattern|) plus the child scans.
    pub fn count(&self, pattern: &[u8]) -> usize {
        self.locate_node(pattern)
            .map_or(0, |v| self.leaves_under(v))
    }

    /// Leaves in the subtree of `v`.
    pub fn leaves_under(&self, v: u32) -> usize {
        let end = self.nodes[v as usize].end as usize;
        (self.leaves_before[end] - self.leaves_before[v as usize]) as usize
    }

    /// Start positions of every occurrence, ascending.
    pub fn locate(&self, pattern: &[u8]) -> Vec<usize> {
        let Some(v) = self.locate_node(pattern) else {
            return Vec::new();
        };
        let mut out = self.suffixes_under(v);
        out.sort_unstable();
        out
    }

    /// Start positions of the suffixes below `v`, in suffix-array order.
    pub fn suffixes_under(&self, v: u32) -> Vec<usize> {
        let n = self.text.len();
        let range = v as usize..self.nodes[v as usize].end as usize;
        range
            .filter(|&w| self.nodes[w].end as usize == w + 1)
            .map(|w| n - self.nodes[w].depth as usize)
            .collect()
    }

    /// The suffix array, read off the leaves.
    pub fn suffix_array(&self) -> Vec<u32> {
        self.suffixes_under(0)
            .into_iter()
            .map(|i| i as u32)
            .collect()
    }

    /// The LCP array: between consecutive leaves it is the depth of the parent of the node that
    /// follows the earlier leaf in preorder.
    pub fn lcp_array(&self) -> Vec<u32> {
        let mut lcp = Vec::with_capacity(self.leaf_count());
        let mut path: Vec<(u32, u32)> = Vec::new();
        let mut prev_leaf = false;
        for (v, node) in self.nodes.iter().enumerate() {
            while path.last().is_some_and(|&(end, _)| end as usize <= v) {
                path.pop();
            }
            if prev_leaf {
                lcp.push(path.last().map_or(0, |&(_, d)| d));
            }
            if node.end as usize == v + 1 {
                if lcp.is_empty() && !prev_leaf {
                    lcp.push(0);
                }
                prev_leaf = true;
            } else {
                path.push((node.end, node.depth));
                prev_leaf = false;
            }
        }
        lcp
    }

    /// The deepest internal node's string: the longest substring that occurs at least twice, as
    /// `(start, length)`. `(0, 0)` if every symbol is distinct.
    pub fn longest_repeated_substring(&self) -> (usize, usize) {
        let mut best = (0u32, 0u32); // (node, depth)
        for (v, node) in self.nodes.iter().enumerate().skip(1) {
            if node.end as usize > v + 1 && node.depth > best.1 {
                best = (v as u32, node.depth);
            }
        }
        if best.1 == 0 {
            return (0, 0);
        }
        let mut leaf = best.0 as usize + 1;
        while self.nodes[leaf].end as usize != leaf + 1 {
            leaf += 1;
        }
        (
            self.text.len() - self.nodes[leaf].depth as usize,
            best.1 as usize,
        )
    }

    /// Suffix links of the internal nodes (`NO_LINK` for leaves and the root): the link of the node
    /// spelling `xα` is the node spelling `α`. Built in O(n) by following the parent's link and
    /// skipping down whole edges, never comparing symbols (the target is known to exist).
    pub fn suffix_links(&self) -> Vec<u32> {
        let nodes = &self.nodes;
        let mut links = vec![NO_LINK; nodes.len()];
        let mut path: Vec<u32> = Vec::new(); // open ancestors
        for v in 0..nodes.len() as u32 {
            while path.last().is_some_and(|&a| nodes[a as usize].end <= v) {
                path.pop();
            }
            let node = nodes[v as usize];
            if node.end > v + 1 {
                if v != 0 {
                    let parent = *path.last().expect("every non-root node has a parent");
                    let pd = nodes[parent as usize].depth;
                    // Spell the node's label without its first symbol, starting at the parent's
                    // link (or at the root, minus the edge's first symbol, for a root child).
                    let (mut at, mut skip, mut pos) = if parent == 0 {
                        (0u32, node.depth - pd - 1, node.start + 1)
                    } else {
                        (links[parent as usize], node.depth - pd, node.start)
                    };
                    while skip > 0 {
                        let c = self
                            .child_with(at, self.text.sym(pos as usize))
                            .expect("a suffix link target exists");
                        let len = nodes[c as usize].depth - nodes[at as usize].depth;
                        debug_assert!(len <= skip, "link targets are explicit nodes");
                        pos += len;
                        skip -= len;
                        at = c;
                    }
                    links[v as usize] = at;
                }
                path.push(v);
            }
        }
        links
    }

    // ----- verification ----------------------------------------------------------------------

    /// Check every structural rule of a suffix tree; panics with a description on a failure.
    ///
    /// Preorder and `end` consistency, strictly increasing depth along edges, every internal node
    /// except possibly the root branching, siblings ordered by first symbol with no repeats, edge
    /// labels spelling the same string as the leaf below them, and one leaf per suffix.
    pub fn check_invariants(&self) {
        let n = self.text.len();
        let nodes = &self.nodes;
        assert_eq!(nodes[0].depth, 0, "root depth");
        assert_eq!(nodes[0].end as usize, nodes.len(), "root spans everything");
        assert_eq!(self.leaf_count(), n, "one leaf per suffix");
        let mut leaves_seen = vec![false; n];
        for v in 0..nodes.len() as u32 {
            let node = nodes[v as usize];
            assert!(
                node.end > v && node.end as usize <= nodes.len(),
                "end of {v}"
            );
            if self.is_leaf(v) {
                let i = n - node.depth as usize;
                assert!(
                    !std::mem::replace(&mut leaves_seen[i], true),
                    "suffix {i} twice"
                );
                continue;
            }
            let mut prev_sym = None;
            let mut kids = 0;
            let mut c = v + 1;
            while c < node.end {
                let child = nodes[c as usize];
                assert!(child.depth > node.depth, "depth must grow along {v} -> {c}");
                let first = self.text.sym(child.start as usize);
                assert!(
                    prev_sym.is_none_or(|p| p < first),
                    "siblings of {v} out of order"
                );
                prev_sym = Some(first);
                // The edge label must spell the same symbols as the leftmost suffix below it.
                let mut leaf = c as usize;
                while nodes[leaf].end as usize != leaf + 1 {
                    leaf += 1;
                }
                let suffix = n - nodes[leaf].depth as usize;
                for t in 0..(child.depth - node.depth) as usize {
                    assert_eq!(
                        self.text.sym(child.start as usize + t),
                        self.text.sym(suffix + node.depth as usize + t),
                        "edge {v} -> {c} does not spell its path"
                    );
                }
                kids += 1;
                c = child.end;
            }
            assert_eq!(c, node.end, "children of {v} tile its subtree");
            assert!(kids >= 2 || v == 0, "internal node {v} must branch");
        }
        assert!(leaves_seen.iter().all(|&s| s), "every suffix has a leaf");
    }
}

/// Iterator over the children of a node; see [`SuffixTree::children`].
pub struct Children<'a> {
    nodes: &'a [Node],
    next: u32,
    end: u32,
}

impl Iterator for Children<'_> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<u32> {
        if self.next >= self.end {
            return None;
        }
        let c = self.next;
        self.next = self.nodes[c as usize].end;
        Some(c)
    }
}

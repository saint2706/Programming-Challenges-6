//! Ukkonen's online suffix tree construction (1995), in O(n) time for a constant alphabet.
//!
//! The tree for the first `i` symbols is extended to the first `i + 1` by adding the new symbol
//! to every suffix. Three ideas make that O(1) amortized per symbol:
//!
//! * **Open leaves.** A leaf's edge label ends "at the current end", so extending every leaf by
//!   the new symbol costs nothing.
//! * **The active point.** Suffixes are processed from longest to shortest, and everything up to
//!   the first one that already continues with the new symbol (rule 3) is left alone; that point
//!   is remembered as (active node, active edge, active length) and the `remainder` counts the
//!   suffixes still owed.
//! * **Suffix links.** After inserting suffix `xα`, the next one, `α`, is reached by following the
//!   suffix link of the internal node above, with no search from the root.
//!
//! Between steps the tree is *implicit*: suffixes that are prefixes of other suffixes have no leaf
//! yet. A unique terminator (see [`crate::text`]) makes every suffix explicit at the end.
//!
//! The cost that dominates in practice is not the algorithm but the **branch lookup**: given a node
//! and a symbol, find the child (Larsson, Fuglsang and Karlsson, 2014). [`ChildMap`] makes it a
//! parameter, with two backends:
//!
//! * [`ListChildren`]: first-child / next-sibling lists. Compact and cache-friendly for small
//!   alphabets; a lookup scans the siblings.
//! * [`HashChildren`]: one hash map keyed by `(parent, symbol)`. O(1) per lookup, more memory and
//!   a hash on every branch.
//!
//! [`Ukkonen::freeze`] converts the finished tree to the flat preorder [`SuffixTree`].
//! [`OnlineSuffixTree`] keeps the live structure and answers queries on the implicit tree after
//! every appended byte.

use foldhash::HashMap;

use crate::text::{Bytes, Prefix, Symbols};
use crate::tree::{Node, SuffixTree};

const NIL: u32 = u32::MAX;
/// The `end` of a leaf edge: "the current end of the text".
const OPEN: u32 = u32::MAX;

/// The branch-lookup structure of a growing tree. Node ids are dense and added in order.
pub trait ChildMap {
    fn new() -> Self;

    /// Called once per new node, with the id it will have.
    fn add_node(&mut self);

    /// The child of `parent` whose edge starts with `sym`, or `NIL`.
    fn get(&self, parent: u32, sym: u32) -> u32;

    /// Make `child` the child of `parent` for `sym`, replacing any existing one.
    fn set(&mut self, parent: u32, sym: u32, child: u32);

    /// All edges as CSR: `offsets[v]..offsets[v + 1]` indexes `v`'s `(symbol, child)` pairs, sorted
    /// by symbol.
    fn csr(&self, node_count: usize) -> (Vec<u32>, Vec<(u32, u32)>);

    fn heap_bytes(&self) -> usize;
}

/// First-child / next-sibling lists, with the first symbol of each child's edge cached.
pub struct ListChildren {
    first: Vec<u32>,
    next: Vec<u32>,
    key: Vec<u32>,
}

impl ChildMap for ListChildren {
    fn new() -> Self {
        ListChildren {
            first: Vec::new(),
            next: Vec::new(),
            key: Vec::new(),
        }
    }

    fn add_node(&mut self) {
        self.first.push(NIL);
        self.next.push(NIL);
        self.key.push(0);
    }

    #[inline]
    fn get(&self, parent: u32, sym: u32) -> u32 {
        let mut c = self.first[parent as usize];
        while c != NIL {
            if self.key[c as usize] == sym {
                return c;
            }
            c = self.next[c as usize];
        }
        NIL
    }

    fn set(&mut self, parent: u32, sym: u32, child: u32) {
        self.key[child as usize] = sym;
        // Replace an existing child with this symbol, keeping its place in the list.
        let mut prev = NIL;
        let mut c = self.first[parent as usize];
        while c != NIL {
            if self.key[c as usize] == sym && c != child {
                self.next[child as usize] = self.next[c as usize];
                match prev {
                    NIL => self.first[parent as usize] = child,
                    p => self.next[p as usize] = child,
                }
                return;
            }
            prev = c;
            c = self.next[c as usize];
        }
        self.next[child as usize] = self.first[parent as usize];
        self.first[parent as usize] = child;
    }

    fn csr(&self, node_count: usize) -> (Vec<u32>, Vec<(u32, u32)>) {
        let mut offsets = Vec::with_capacity(node_count + 1);
        let mut pairs = Vec::new();
        for v in 0..node_count {
            offsets.push(pairs.len() as u32);
            let from = pairs.len();
            let mut c = self.first[v];
            while c != NIL {
                pairs.push((self.key[c as usize], c));
                c = self.next[c as usize];
            }
            pairs[from..].sort_unstable();
        }
        offsets.push(pairs.len() as u32);
        (offsets, pairs)
    }

    fn heap_bytes(&self) -> usize {
        (self.first.capacity() + self.next.capacity() + self.key.capacity()) * 4
    }
}

/// One hash map from `(parent, symbol)` to child.
pub struct HashChildren {
    map: HashMap<u64, u32>,
}

#[inline]
fn pack(parent: u32, sym: u32) -> u64 {
    u64::from(parent) << 32 | u64::from(sym)
}

impl ChildMap for HashChildren {
    fn new() -> Self {
        HashChildren {
            map: HashMap::default(),
        }
    }

    fn add_node(&mut self) {}

    #[inline]
    fn get(&self, parent: u32, sym: u32) -> u32 {
        self.map.get(&pack(parent, sym)).copied().unwrap_or(NIL)
    }

    fn set(&mut self, parent: u32, sym: u32, child: u32) {
        self.map.insert(pack(parent, sym), child);
    }

    fn csr(&self, node_count: usize) -> (Vec<u32>, Vec<(u32, u32)>) {
        let mut offsets = vec![0u32; node_count + 1];
        for &key in self.map.keys() {
            offsets[(key >> 32) as usize + 1] += 1;
        }
        for v in 0..node_count {
            offsets[v + 1] += offsets[v];
        }
        let mut fill = offsets.clone();
        let mut pairs = vec![(0u32, 0u32); self.map.len()];
        for (&key, &child) in &self.map {
            let parent = (key >> 32) as usize;
            pairs[fill[parent] as usize] = (key as u32, child);
            fill[parent] += 1;
        }
        for v in 0..node_count {
            pairs[offsets[v] as usize..offsets[v + 1] as usize].sort_unstable();
        }
        (offsets, pairs)
    }

    fn heap_bytes(&self) -> usize {
        // hashbrown keeps one control byte per slot next to 12-byte entries (8-byte key, 4-byte
        // value, padded to 16), at a load factor of at most 7/8.
        self.map.capacity() * (16 + 1)
    }
}

#[derive(Clone, Copy)]
struct UNode {
    start: u32,
    end: u32,
    link: u32,
}

/// The tree under construction. Use [`Ukkonen::build`] for a whole text.
pub struct Ukkonen<M: ChildMap> {
    nodes: Vec<UNode>,
    map: M,
    active_node: u32,
    active_edge: u32,
    active_len: u32,
    remainder: u32,
    leaves: usize,
}

impl<M: ChildMap> Default for Ukkonen<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: ChildMap> Ukkonen<M> {
    pub fn new() -> Self {
        let mut t = Ukkonen {
            nodes: Vec::new(),
            map: M::new(),
            active_node: 0,
            active_edge: 0,
            active_len: 0,
            remainder: 0,
            leaves: 0,
        };
        t.new_node(0, 0); // the root
        t
    }

    /// Run the algorithm over a whole terminated text.
    pub fn build<T: Symbols>(text: &T) -> Self {
        let mut t = Ukkonen::new();
        t.nodes.reserve(2 * text.len());
        for pos in 0..text.len() {
            t.extend(text, pos);
        }
        t
    }

    fn new_node(&mut self, start: u32, end: u32) -> u32 {
        self.nodes.push(UNode {
            start,
            end,
            link: 0,
        });
        self.map.add_node();
        (self.nodes.len() - 1) as u32
    }

    #[inline]
    fn edge_len(&self, v: u32, current_end: usize) -> u32 {
        let n = self.nodes[v as usize];
        n.end.min(current_end as u32) - n.start
    }

    /// Number of nodes so far, root and leaves included.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Suffixes that already have a leaf. While the tree is implicit, the suffixes from this
    /// index on are not explicit yet.
    pub fn leaf_count(&self) -> usize {
        self.leaves
    }

    /// Bytes held by the node array and the branch lookup structure.
    pub fn heap_bytes(&self) -> usize {
        self.nodes.capacity() * std::mem::size_of::<UNode>() + self.map.heap_bytes()
    }

    /// Process the symbol at `pos`; all earlier positions must have been processed.
    pub fn extend<T: Symbols>(&mut self, text: &T, pos: usize) {
        let c = text.sym(pos);
        self.remainder += 1;
        let mut last_new = NIL;
        while self.remainder > 0 {
            if self.active_len == 0 {
                self.active_edge = pos as u32;
            }
            let edge_sym = text.sym(self.active_edge as usize);
            let next = self.map.get(self.active_node, edge_sym);
            if next == NIL {
                // No edge for this symbol: hang a new leaf off the active node.
                let leaf = self.new_node(pos as u32, OPEN);
                self.leaves += 1;
                self.map.set(self.active_node, edge_sym, leaf);
                if last_new != NIL {
                    self.nodes[last_new as usize].link = self.active_node;
                    last_new = NIL;
                }
            } else {
                let len = self.edge_len(next, pos + 1);
                if self.active_len >= len {
                    // The active point is past this whole edge: walk down.
                    self.active_edge += len;
                    self.active_len -= len;
                    self.active_node = next;
                    continue;
                }
                let at = self.nodes[next as usize].start + self.active_len;
                if text.sym(at as usize) == c {
                    // Rule 3: the suffix already continues with `c`; so do all shorter ones.
                    if last_new != NIL {
                        self.nodes[last_new as usize].link = self.active_node;
                    }
                    self.active_len += 1;
                    break;
                }
                // Split the edge at the active point and hang the new leaf off the new node.
                let start = self.nodes[next as usize].start;
                let split = self.new_node(start, at);
                self.map.set(self.active_node, edge_sym, split);
                let leaf = self.new_node(pos as u32, OPEN);
                self.leaves += 1;
                self.map.set(split, c, leaf);
                self.nodes[next as usize].start = at;
                self.map.set(split, text.sym(at as usize), next);
                if last_new != NIL {
                    self.nodes[last_new as usize].link = split;
                }
                last_new = split;
            }
            self.remainder -= 1;
            if self.active_node == 0 && self.active_len > 0 {
                self.active_len -= 1;
                self.active_edge = pos as u32 + 1 - self.remainder;
            } else if self.active_node != 0 {
                self.active_node = self.nodes[self.active_node as usize].link;
            }
        }
    }

    /// Convert the finished tree (the text must have been terminated) into the flat preorder
    /// [`SuffixTree`]. Iterative, so a path-shaped tree of any depth is fine.
    pub fn freeze<S: Symbols>(self, text: S) -> SuffixTree<S> {
        assert_eq!(self.remainder, 0, "freeze needs a terminated text");
        let n = text.len() as u32;
        let (offsets, pairs) = self.map.csr(self.nodes.len());
        let mut out: Vec<Node> = Vec::with_capacity(self.nodes.len());
        out.push(Node {
            start: 0,
            depth: 0,
            end: 0,
        });
        // (old node id, index in `out`, next child cursor, string depth)
        let mut frames = vec![(0u32, 0u32, offsets[0], 0u32)];
        while let Some(frame) = frames.last_mut() {
            let (old, idx, cursor, depth) = *frame;
            if cursor < offsets[old as usize + 1] {
                frame.2 += 1;
                let child = pairs[cursor as usize].1;
                let c = self.nodes[child as usize];
                let end = if c.end == OPEN { n } else { c.end };
                let child_depth = depth + (end - c.start);
                let at = out.len() as u32;
                out.push(Node {
                    start: c.start,
                    depth: child_depth,
                    end: at + 1,
                });
                if offsets[child as usize + 1] > offsets[child as usize] {
                    frames.push((child, at, offsets[child as usize], child_depth));
                }
            } else {
                out[idx as usize].end = out.len() as u32;
                frames.pop();
            }
        }
        SuffixTree::from_nodes(text, out)
    }
}

/// The suffix tree of a byte string by Ukkonen's algorithm with the chosen branch backend.
pub fn tree_of_bytes<M: ChildMap>(bytes: &[u8]) -> SuffixTree<Bytes> {
    let text = Bytes::new(bytes);
    Ukkonen::<M>::build(&text).freeze(text)
}

/// A suffix tree that grows one byte at a time and answers queries in between.
///
/// Queries run on the *implicit* tree of the bytes so far. Every suffix that already has a leaf
/// is found below the match; the few suffixes without one (at most `remainder` of them, those that
/// are prefixes of earlier suffixes) are checked directly against the text.
pub struct OnlineSuffixTree {
    bytes: Vec<u8>,
    engine: Ukkonen<ListChildren>,
}

impl Default for OnlineSuffixTree {
    fn default() -> Self {
        Self::new()
    }
}

impl OnlineSuffixTree {
    pub fn new() -> Self {
        OnlineSuffixTree {
            bytes: Vec::new(),
            engine: Ukkonen::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Append one byte.
    pub fn push(&mut self, b: u8) {
        self.bytes.push(b);
        let pos = self.bytes.len() - 1;
        self.engine.extend(&Prefix(&self.bytes), pos);
    }

    pub fn extend_from_slice(&mut self, bytes: &[u8]) {
        bytes.iter().for_each(|&b| self.push(b));
    }

    pub fn node_count(&self) -> usize {
        self.engine.node_count()
    }

    /// Follow `pattern`; returns the node whose edge holds the end of the match and the string
    /// depth at the end of that edge.
    fn descend(&self, pattern: &[u8]) -> Option<(u32, u32)> {
        let end = self.bytes.len();
        let (mut v, mut depth, mut matched) = (0u32, 0u32, 0usize);
        while matched < pattern.len() {
            let c = self.engine.map.get(v, u32::from(pattern[matched]) + 1);
            if c == NIL {
                return None;
            }
            let len = self.engine.edge_len(c, end) as usize;
            let start = self.engine.nodes[c as usize].start as usize;
            let take = len.min(pattern.len() - matched);
            if self.bytes[start..start + take] != pattern[matched..matched + take] {
                return None;
            }
            matched += take;
            depth += len as u32;
            v = c;
        }
        Some((v, depth))
    }

    pub fn contains(&self, pattern: &[u8]) -> bool {
        self.descend(pattern).is_some()
    }

    /// Start positions of every occurrence in the bytes so far, ascending.
    pub fn locate(&self, pattern: &[u8]) -> Vec<usize> {
        let end = self.bytes.len();
        if pattern.is_empty() {
            return (0..=end).collect();
        }
        let Some((locus, locus_depth)) = self.descend(pattern) else {
            return Vec::new();
        };
        // Leaves below the locus, by iterative DFS accumulating string depth: a leaf at depth d
        // is the suffix that starts d symbols before the end.
        let mut out = Vec::new();
        let mut stack = vec![(locus, locus_depth)];
        while let Some((v, depth)) = stack.pop() {
            let mut c = self.engine.map.first[v as usize];
            if c == NIL {
                out.push(end - depth as usize);
                continue;
            }
            while c != NIL {
                stack.push((c, depth + self.engine.edge_len(c, end)));
                c = self.engine.map.next[c as usize];
            }
        }
        // Suffixes without a leaf yet start at `leaf_count` and are checked in place.
        let m = pattern.len();
        for j in self.engine.leaf_count()..end {
            if j + m <= end && self.bytes[j..j + m] == *pattern {
                out.push(j);
            }
        }
        out.sort_unstable();
        out
    }

    /// Terminate the text and freeze the tree, consuming the live structure.
    pub fn finish(mut self) -> SuffixTree<Bytes> {
        let text = Bytes::new(std::mem::take(&mut self.bytes));
        self.engine.extend(&text, text.len() - 1);
        self.engine.freeze(text)
    }
}

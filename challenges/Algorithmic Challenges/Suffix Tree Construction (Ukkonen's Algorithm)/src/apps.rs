//! Classic suffix-tree applications, all linear in the text (plus the node scans).
//!
//! * [`longest_common_substring`]: the deepest node whose subtree holds a suffix of every string
//!   of a generalized tree.
//! * [`shortest_unique_substring`]: the shortest substring that occurs exactly once.
//! * [`matching_statistics`]: for each position of a query, how much of it the text contains
//!   (Chang and Lawler), in O(|query|) by walking suffix links.

use crate::text::{Symbols, Words};
use crate::tree::{NO_LINK, SuffixTree};

/// Longest substring common to every string of `tree`, as `(string index, start, length)`.
///
/// Each node gets a bitset of the strings below it, computed bottom-up in reverse preorder; the
/// answer is the deepest node whose set is full. `None` if the strings share nothing.
pub fn longest_common_substring(tree: &SuffixTree<Words>) -> Option<(usize, usize, usize)> {
    let words = tree.text();
    let k = words.count();
    let nodes = tree.nodes();
    let width = k.div_ceil(64);
    let mut sets = vec![0u64; nodes.len() * width];
    let n = words.len();
    let mut best: Option<(u32, u32)> = None; // (node, depth)
    let mut acc = vec![0u64; width];
    for v in (0..nodes.len()).rev() {
        if tree.is_leaf(v as u32) {
            let owner = words.string_of(n - nodes[v].depth as usize);
            sets[v * width + owner / 64] |= 1 << (owner % 64);
            continue;
        }
        acc.fill(0);
        for c in tree.children(v as u32) {
            for (a, s) in acc.iter_mut().zip(&sets[c as usize * width..][..width]) {
                *a |= s;
            }
        }
        sets[v * width..][..width].copy_from_slice(&acc);
        let full = (0..k).all(|j| (acc[j / 64] >> (j % 64)) & 1 == 1);
        if v != 0 && full && best.is_none_or(|(_, d)| nodes[v].depth > d) {
            best = Some((v as u32, nodes[v].depth));
        }
    }
    let (node, depth) = best?;
    let leaf = tree.suffixes_under(node)[0];
    let j = words.string_of(leaf);
    Some((j, leaf - words.start_of(j), depth as usize))
}

/// The shortest substring that occurs once, as `(start, length)`; `None` only for an empty text.
///
/// A leaf's edge begins right after the deepest node shared with another suffix, so the shortest
/// unique prefix of that suffix is one symbol past its parent's depth, unless that symbol is the
/// terminator (then the suffix is a prefix of another and has no unique prefix).
pub fn shortest_unique_substring<S: Symbols>(tree: &SuffixTree<S>) -> Option<(usize, usize)> {
    let nodes = tree.nodes();
    let text = tree.text();
    let n = text.len();
    let mut path: Vec<(u32, u32)> = Vec::new(); // (end, depth) of open ancestors
    let mut best: Option<(usize, usize)> = None;
    for (v, node) in nodes.iter().enumerate() {
        while path.last().is_some_and(|&(end, _)| end as usize <= v) {
            path.pop();
        }
        if node.end as usize > v + 1 {
            path.push((node.end, node.depth));
            continue;
        }
        if text.is_terminator(text.sym(node.start as usize)) {
            continue;
        }
        let parent_depth = path.last().map_or(0, |&(_, d)| d);
        let (start, len) = (n - node.depth as usize, parent_depth as usize + 1);
        if best.is_none_or(|(_, l)| len < l) {
            best = Some((start, len));
        }
    }
    best
}

/// `ms[i]`: the length of the longest prefix of `query[i..]` that occurs in the text.
///
/// Keeps the current match as (explicit node `v`, length `l >= depth(v)`). After a position is
/// done, the match for the next one is the same string minus its first symbol, which sits at the
/// suffix link of `v`; the symbols between the link and `l` are known to match, so they are
/// skipped edge by edge instead of compared.
pub fn matching_statistics<S: Symbols>(tree: &SuffixTree<S>, query: &[u8]) -> Vec<usize> {
    let links = tree.suffix_links();
    let text = tree.text();
    let nodes = tree.nodes();
    let m = query.len();
    let q = |i: usize| text.byte_sym(query[i]);
    let mut out = Vec::with_capacity(m);
    let (mut v, mut l) = (0u32, 0usize);
    for i in 0..m {
        loop {
            let dv = nodes[v as usize].depth as usize;
            // The query symbol that picks the edge below `v`: the one right after the node's
            // string. Nothing to pick if the query ends there.
            if i + dv >= m {
                break;
            }
            let want = q(i + dv);
            let Some(c) = tree
                .children(v)
                .find(|&c| text.sym(nodes[c as usize].start as usize) == want)
            else {
                break;
            };
            let len = nodes[c as usize].depth as usize - dv;
            if l > dv && l - dv >= len {
                v = c; // already known to match: skip the whole edge
                continue;
            }
            let start = nodes[c as usize].start as usize;
            let mut matched = l - dv;
            while matched < len && i + l < m && text.sym(start + matched) == q(i + l) {
                matched += 1;
                l += 1;
            }
            if matched == len {
                v = c;
            } else {
                break;
            }
        }
        out.push(l);
        if l == 0 {
            continue;
        }
        l -= 1;
        if v != 0 {
            v = links[v as usize];
            debug_assert_ne!(v, NO_LINK);
        }
    }
    out
}

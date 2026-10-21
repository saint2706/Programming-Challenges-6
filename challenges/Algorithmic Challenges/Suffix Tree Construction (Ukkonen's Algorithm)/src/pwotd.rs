//! Top-down suffix tree construction with a prefix partition: PWOTD, the algorithm Tata, Hankins
//! and Patel ("Practical Suffix Tree Construction", VLDB 2004) found faster than Ukkonen's on real
//! hardware despite its O(n²) worst case.
//!
//! Ukkonen jumps between nodes along suffix links, so its memory accesses are scattered; PWOTD
//! never follows a link. It keeps one array of suffix start positions and builds the tree
//! depth-first:
//!
//! 1. **Partition.** One sequential scan of the text counts, then buckets, every suffix by its
//!    first `p` symbols. The array is now sorted on `p` symbols, having touched the text in order.
//! 2. **Expand a group** of suffixes that share `d` symbols: find the longest extension `L` they
//!    all share (that is the node's string depth), split the group by the symbol at depth `L`
//!    with a counting sort, emit the node, and recurse into the sub-groups in symbol order.
//!    A group of one suffix is a leaf.
//!
//! Finding the shared extension scans every suffix of the group, which is where the quadratic
//! worst case lives (a run of one symbol rescans the whole shrinking group at every level). On
//! texts with short repeats it is a handful of comparisons.
//!
//! **What the partition buys, exactly.** A range sorted on its first `s` symbols has its shared
//! prefix (within those `s`) equal to the LCP of its first and last suffix, and its sub-groups
//! sit at positions found by binary search, so the first `p` levels cost O(children · log n)
//! text reads instead of O(group size). Past depth `p` a group is unsorted and falls back to the
//! scan and the counting sort. `p = 0` is plain write-only top-down (WOTD).
//!
//! The output is the same flat preorder [`SuffixTree`] as the other builders, emitted directly in
//! order, so there is no conversion pass.

use crate::text::Symbols;
use crate::tree::{Node, SuffixTree};

/// Most partitions the automatic prefix length will create.
const MAX_PARTITIONS: usize = 1 << 16;

enum Task {
    /// A group `sa[lo..hi]` whose suffixes share `depth` symbols and are sorted on the first
    /// `sorted` symbols. `force` expands even a single suffix (only the root needs it).
    Group {
        lo: u32,
        hi: u32,
        depth: u32,
        sorted: u32,
        force: bool,
    },
    /// All of the node at this output index has been emitted: record where its subtree ends.
    Close(u32),
}

/// The longest prefix length `p` with `distinct^p <= MAX_PARTITIONS` and no more partitions than
/// there are suffixes.
pub fn auto_prefix_len(distinct: usize, n: usize) -> usize {
    let mut p = 0;
    let mut buckets = 1usize;
    while distinct > 1 && buckets * distinct <= MAX_PARTITIONS && buckets * distinct <= n {
        buckets *= distinct;
        p += 1;
    }
    p
}

/// Build with the automatic prefix length.
pub fn build<S: Symbols>(text: S) -> SuffixTree<S> {
    build_with(text, None)
}

/// Build with prefix length `p` (`Some(0)` is plain top-down; `None` picks one).
pub fn build_with<S: Symbols>(text: S, prefix_len: Option<usize>) -> SuffixTree<S> {
    let n = text.len();
    assert!(n < u32::MAX as usize, "text too long for u32 positions");

    // Dense symbol ranks, so a counting sort only ever looks at symbols that occur.
    let mut seen = vec![false; text.alphabet() as usize];
    for i in 0..n {
        seen[text.sym(i) as usize] = true;
    }
    let mut rank = vec![0u32; seen.len()];
    let mut distinct = 0u32;
    for (s, &used) in seen.iter().enumerate() {
        if used {
            rank[s] = distinct;
            distinct += 1;
        }
    }
    let sigma = distinct as usize;
    let p = prefix_len.unwrap_or_else(|| auto_prefix_len(sigma, n));

    let mut sa: Vec<u32> = vec![0; n];
    partition(&text, &rank, sigma, p, &mut sa);

    let mut tmp = vec![0u32; n];
    let mut counts = vec![0u32; sigma];
    let mut touched: Vec<u32> = Vec::new();
    let mut out: Vec<Node> = Vec::with_capacity(2 * n);
    let mut tasks = vec![Task::Group {
        lo: 0,
        hi: n as u32,
        depth: 0,
        sorted: p as u32,
        force: true,
    }];
    let mut kids: Vec<(u32, u32)> = Vec::new();

    while let Some(task) = tasks.pop() {
        let (lo, hi, depth, sorted, force) = match task {
            Task::Close(idx) => {
                out[idx as usize].end = out.len() as u32;
                continue;
            }
            Task::Group {
                lo,
                hi,
                depth,
                sorted,
                force,
            } => (
                lo as usize,
                hi as usize,
                depth as usize,
                sorted as usize,
                force,
            ),
        };
        if hi - lo == 1 && !force {
            let i = sa[lo] as usize;
            let at = out.len() as u32;
            out.push(Node {
                start: (i + depth) as u32,
                depth: (n - i) as u32,
                end: at + 1,
            });
            continue;
        }

        // The node's string depth: how far all suffixes of the group agree.
        let level = common_depth(&text, &sa[lo..hi], depth, sorted);
        let idx = out.len() as u32;
        out.push(Node {
            start: sa[lo] + depth as u32,
            depth: level as u32,
            end: 0,
        });
        tasks.push(Task::Close(idx));

        // Split by the symbol at `level`.
        kids.clear();
        if level < sorted {
            presorted_groups(&text, &sa[lo..hi], level, lo as u32, &mut kids);
        } else {
            counting_groups(
                &text,
                &rank,
                &mut sa[lo..hi],
                &mut tmp[lo..hi],
                level,
                lo as u32,
                &mut counts,
                &mut touched,
                &mut kids,
            );
        }
        let child_sorted = if level < sorted { sorted } else { 0 };
        for &(clo, chi) in kids.iter().rev() {
            tasks.push(Task::Group {
                lo: clo,
                hi: chi,
                depth: level as u32,
                sorted: child_sorted as u32,
                force: false,
            });
        }
    }
    SuffixTree::from_nodes(text, out)
}

/// Count and bucket every suffix by its first `p` symbols with one scan of the text.
fn partition<S: Symbols>(text: &S, rank: &[u32], sigma: usize, p: usize, sa: &mut [u32]) {
    let n = text.len();
    if p == 0 {
        for (i, slot) in sa.iter_mut().enumerate() {
            *slot = i as u32;
        }
        return;
    }
    let buckets = sigma.pow(p as u32);
    let top = buckets / sigma; // weight of the leading symbol
    let sym = |i: usize| {
        if i < n {
            rank[text.sym(i) as usize] as usize
        } else {
            0
        }
    };
    // Rolling code of the p-symbol window starting at i; positions past the end read as rank 0
    // (which is harmless: the terminator is unique, so no two windows differ only in padding).
    let first_code = |start: usize| (0..p).fold(0usize, |c, t| c * sigma + sym(start + t));
    let mut counts = vec![0u32; buckets + 1];
    let mut code = first_code(0);
    for i in 0..n {
        counts[code + 1] += 1;
        code = (code % top) * sigma + sym(i + p);
    }
    for b in 0..buckets {
        counts[b + 1] += counts[b];
    }
    let mut code = first_code(0);
    for i in 0..n {
        sa[counts[code] as usize] = i as u32;
        counts[code] += 1;
        code = (code % top) * sigma + sym(i + p);
    }
}

/// Length of the prefix shared by every suffix in `group` (all of which already share `depth`).
fn common_depth<S: Symbols>(text: &S, group: &[u32], depth: usize, sorted: usize) -> usize {
    if group.len() == 1 {
        return depth; // only the forced root of an empty text: nothing to agree with
    }
    let mut at = depth;
    if sorted > depth && group.len() > 1 {
        // Sorted on `sorted` symbols: the first and last suffix bound everything in between.
        let (a, z) = (group[0] as usize, group[group.len() - 1] as usize);
        while at < sorted && text.sym(a + at) == text.sym(z + at) {
            at += 1;
        }
        if at < sorted {
            return at;
        }
    }
    // Unsorted from here on: every suffix has to agree with the first one, column by column.
    let first = group[0] as usize;
    loop {
        let c = text.sym(first + at);
        if group[1..].iter().any(|&s| text.sym(s as usize + at) != c) {
            return at;
        }
        at += 1;
    }
}

/// Sub-groups of a range sorted on the symbol at `level`: runs of equal symbols, found by binary
/// search instead of by reading every suffix.
fn presorted_groups<S: Symbols>(
    text: &S,
    group: &[u32],
    level: usize,
    base: u32,
    kids: &mut Vec<(u32, u32)>,
) {
    let key = |k: usize| text.sym(group[k] as usize + level);
    let mut lo = 0;
    while lo < group.len() {
        let c = key(lo);
        // First index in (lo, len] whose symbol is greater than `c`.
        let (mut a, mut b) = (lo + 1, group.len());
        while a < b {
            let mid = a + (b - a) / 2;
            if key(mid) == c {
                a = mid + 1;
            } else {
                b = mid;
            }
        }
        kids.push((base + lo as u32, base + a as u32));
        lo = a;
    }
}

/// Sub-groups of an unsorted range: a counting sort on the symbol at `level`, using only the
/// symbols that occur (`touched`) so a node costs O(group + distinct symbols present).
#[allow(clippy::too_many_arguments)]
fn counting_groups<S: Symbols>(
    text: &S,
    rank: &[u32],
    group: &mut [u32],
    tmp: &mut [u32],
    level: usize,
    base: u32,
    counts: &mut [u32],
    touched: &mut Vec<u32>,
    kids: &mut Vec<(u32, u32)>,
) {
    touched.clear();
    for &s in group.iter() {
        let r = rank[text.sym(s as usize + level) as usize] as usize;
        if counts[r] == 0 {
            touched.push(r as u32);
        }
        counts[r] += 1;
    }
    touched.sort_unstable();
    let mut start = 0u32;
    for &r in touched.iter() {
        let c = counts[r as usize];
        kids.push((base + start, base + start + c));
        counts[r as usize] = start; // becomes the write cursor
        start += c;
    }
    for &s in group.iter() {
        let r = rank[text.sym(s as usize + level) as usize] as usize;
        tmp[counts[r] as usize] = s;
        counts[r] += 1;
    }
    group.copy_from_slice(tmp);
    for &r in touched.iter() {
        counts[r as usize] = 0;
    }
}

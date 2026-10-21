//! Longest-common-prefix arrays: `lcp[i]` is the length of the longest common prefix of the
//! suffixes `sa[i - 1]` and `sa[i]`, with `lcp[0] = 0`.
//!
//! Both algorithms rest on one fact: if suffix `i` shares `h` symbols with its predecessor in
//! suffix order, suffix `i + 1` shares at least `h - 1` with *its* predecessor, so a single scan in
//! text order does O(n) comparisons in total.
//!
//! * [`lcp_kasai`] (Kasai et al. 2001) needs the inverse suffix array, and reads it in text order
//!   while writing `lcp` in suffix order: a random access per position.
//! * [`lcp_phi`] (Kärkkäinen, Manzini and Puglisi 2009) goes through the *predecessor* array
//!   `phi[sa[i]] = sa[i - 1]` and a permuted LCP, so the heavy scan reads and writes sequentially;
//!   only building `phi` and un-permuting at the end are random.

use crate::sais::Symbol;

const NONE: u32 = u32::MAX;

/// Kasai's algorithm.
pub fn lcp_kasai<T: Symbol>(text: &[T], sa: &[u32]) -> Vec<u32> {
    let n = text.len();
    let mut rank = vec![0u32; n];
    for (i, &p) in sa.iter().enumerate() {
        rank[p as usize] = i as u32;
    }
    let mut lcp = vec![0u32; n];
    let mut h = 0usize;
    for i in 0..n {
        let r = rank[i] as usize;
        if r == 0 {
            h = 0;
            continue;
        }
        let j = sa[r - 1] as usize;
        while i + h < n && j + h < n && text[i + h].index() == text[j + h].index() {
            h += 1;
        }
        lcp[r] = h as u32;
        h = h.saturating_sub(1);
    }
    lcp
}

/// The Φ algorithm.
pub fn lcp_phi<T: Symbol>(text: &[T], sa: &[u32]) -> Vec<u32> {
    let n = text.len();
    let mut phi = vec![NONE; n];
    for w in sa.windows(2) {
        phi[w[1] as usize] = w[0];
    }
    // Reuse `phi` as the permuted LCP: plcp[i] = lcp of suffix i with its predecessor.
    let mut h = 0usize;
    for i in 0..n {
        let j = phi[i];
        if j == NONE {
            h = 0;
            phi[i] = 0;
            continue;
        }
        let j = j as usize;
        while i + h < n && j + h < n && text[i + h].index() == text[j + h].index() {
            h += 1;
        }
        phi[i] = h as u32;
        h = h.saturating_sub(1);
    }
    sa.iter().map(|&p| phi[p as usize]).collect()
}

/// The reference: compare each adjacent pair from scratch.
pub fn lcp_naive<T: Symbol>(text: &[T], sa: &[u32]) -> Vec<u32> {
    let mut lcp = vec![0u32; sa.len()];
    for i in 1..sa.len() {
        let (a, b) = (&text[sa[i - 1] as usize..], &text[sa[i] as usize..]);
        lcp[i] = a
            .iter()
            .zip(b)
            .take_while(|(x, y)| x.index() == y.index())
            .count() as u32;
    }
    lcp
}

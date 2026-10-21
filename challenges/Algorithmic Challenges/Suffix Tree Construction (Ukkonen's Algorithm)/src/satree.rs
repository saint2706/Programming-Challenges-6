//! Suffix trees from suffix arrays: SA-IS, then the Φ LCP algorithm, then the linear
//! LCP-interval construction in [`SuffixTree::from_sa_lcp`].
//!
//! This is the route modern tools take (Abouelhoda, Kurtz and Ohlebusch's "enhanced suffix
//! array"): three flat passes over arrays, no pointers chased while building.

use crate::lcp::lcp_phi;
use crate::sais::{suffix_array, suffix_array_u8};
use crate::text::{Bytes, Symbols, Words, to_vec};
use crate::tree::SuffixTree;

/// Suffix array and LCP array of a byte string *with* its terminator: `n + 1` entries, the
/// terminator's suffix first.
pub fn sa_lcp_bytes(bytes: &[u8]) -> (Vec<u32>, Vec<u32>) {
    let n = bytes.len();
    let core = suffix_array_u8(bytes);
    let core_lcp = lcp_phi(bytes, &core);
    let mut sa = Vec::with_capacity(n + 1);
    sa.push(n as u32);
    sa.extend_from_slice(&core);
    let mut lcp = Vec::with_capacity(n + 1);
    lcp.push(0);
    lcp.push(0);
    lcp.extend(core_lcp.iter().skip(1));
    lcp.truncate(n + 1);
    (sa, lcp)
}

/// Suffix array and LCP array of any terminated text (separators already inside).
pub fn sa_lcp_symbols<S: Symbols>(text: &S) -> (Vec<u32>, Vec<u32>) {
    let syms = to_vec(text);
    let sa = suffix_array(&syms, text.alphabet() as usize);
    let lcp = lcp_phi(&syms, &sa);
    (sa, lcp)
}

/// The suffix tree of a byte string, via SA-IS.
pub fn tree_of_bytes(bytes: &[u8]) -> SuffixTree<Bytes> {
    let (sa, lcp) = sa_lcp_bytes(bytes);
    SuffixTree::from_sa_lcp(Bytes::new(bytes), &sa, &lcp)
}

/// The generalized suffix tree of several byte strings, via SA-IS.
pub fn tree_of_words<S: AsRef<[u8]>>(strings: &[S]) -> SuffixTree<Words> {
    let words = Words::new(strings);
    let (sa, lcp) = sa_lcp_symbols(&words);
    SuffixTree::from_sa_lcp(words, &sa, &lcp)
}

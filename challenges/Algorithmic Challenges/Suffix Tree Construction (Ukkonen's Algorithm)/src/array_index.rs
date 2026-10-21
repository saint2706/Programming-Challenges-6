//! The suffix array as an index: what a suffix tree is usually compared against.
//!
//! A suffix array plus its LCP array answers the same questions as a suffix tree in a fraction of
//! the memory (8 bytes per symbol here, against 24 and up). Pattern search is a binary search over
//! the sorted suffixes, O(m log n) instead of O(m), and the LCP array alone finds the longest
//! repeated substring and, with a sliding window over the array, the longest common substring of
//! several strings.

use crate::lcp::lcp_phi;
use crate::sais::{suffix_array, suffix_array_u8};
use crate::text::{Symbols, Words};

/// A byte string's suffix array and LCP array, with the usual searches.
pub struct SuffixArrayIndex {
    text: Vec<u8>,
    /// Suffix starts in sorted order, `text.len()` entries (no empty suffix).
    sa: Vec<u32>,
    /// `lcp[i]` = common prefix of `sa[i - 1]` and `sa[i]`; `lcp[0] = 0`.
    lcp: Vec<u32>,
}

impl SuffixArrayIndex {
    pub fn new(text: impl Into<Vec<u8>>) -> Self {
        let text = text.into();
        let sa = suffix_array_u8(&text);
        let lcp = lcp_phi(&text, &sa);
        SuffixArrayIndex { text, sa, lcp }
    }

    pub fn suffix_array(&self) -> &[u32] {
        &self.sa
    }

    pub fn lcp_array(&self) -> &[u32] {
        &self.lcp
    }

    /// Bytes held by the two arrays (not the text).
    pub fn heap_bytes(&self) -> usize {
        (self.sa.len() + self.lcp.len()) * 4
    }

    /// The range of suffix-array rows whose suffix starts with `pattern`: two binary searches on
    /// the suffixes truncated to the pattern's length.
    fn range(&self, pattern: &[u8]) -> std::ops::Range<usize> {
        let m = pattern.len();
        let prefix = |p: u32| {
            let s = &self.text[p as usize..];
            &s[..m.min(s.len())]
        };
        let lo = self.sa.partition_point(|&p| prefix(p) < pattern);
        let hi = self.sa.partition_point(|&p| prefix(p) <= pattern);
        lo..hi
    }

    pub fn contains(&self, pattern: &[u8]) -> bool {
        !self.range(pattern).is_empty()
    }

    pub fn count(&self, pattern: &[u8]) -> usize {
        self.range(pattern).len()
    }

    /// Start positions of every occurrence, ascending.
    pub fn locate(&self, pattern: &[u8]) -> Vec<usize> {
        let mut out: Vec<usize> = self.sa[self.range(pattern)]
            .iter()
            .map(|&p| p as usize)
            .collect();
        out.sort_unstable();
        out
    }

    /// The longest substring that occurs twice, as `(start, length)`: the largest LCP value.
    pub fn longest_repeated_substring(&self) -> (usize, usize) {
        let (row, &len) = self
            .lcp
            .iter()
            .enumerate()
            .max_by_key(|&(_, &l)| l)
            .unwrap_or((0, &0));
        if len == 0 {
            (0, 0)
        } else {
            (self.sa[row] as usize, len as usize)
        }
    }
}

/// Longest common substring of all `strings`, as `(string index, start, length)`, by a sliding
/// window over the generalized suffix array: the shortest window that holds a suffix of every
/// string; the common prefix of a window is the minimum LCP inside it.
///
/// Returns `None` if there are no strings, one is empty, or they share nothing.
pub fn longest_common_substring_array<S: AsRef<[u8]>>(
    strings: &[S],
) -> Option<(usize, usize, usize)> {
    let k = strings.len();
    if k == 0 || strings.iter().any(|s| s.as_ref().is_empty()) {
        return None;
    }
    if k == 1 {
        return Some((0, 0, strings[0].as_ref().len()));
    }
    let words = Words::new(strings);
    let syms = words.as_symbols();
    let sa = suffix_array(syms, words.alphabet() as usize);
    let lcp = lcp_phi(syms, &sa);
    // The first k rows are the separators' own suffixes; every later row belongs to one string.
    let first = k;
    let owner = |row: usize| words.string_of(sa[row] as usize);

    let mut have = vec![0usize; k];
    let mut covered = 0;
    let mut best = (0usize, 0usize); // (length, text position)
    let mut mins: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    let mut left = first;
    for right in first..sa.len() {
        if have[owner(right)] == 0 {
            covered += 1;
        }
        have[owner(right)] += 1;
        if right > first {
            // Keep the indices of lcp[(left, right]] whose values are increasing: the front is
            // the minimum.
            while mins.back().is_some_and(|&b| lcp[b] >= lcp[right]) {
                mins.pop_back();
            }
            mins.push_back(right);
        }
        while covered == k {
            while mins.front().is_some_and(|&f| f <= left) {
                mins.pop_front();
            }
            let common = mins.front().map_or(0, |&f| lcp[f]) as usize;
            if common > best.0 {
                best = (common, sa[left] as usize);
            }
            let o = owner(left);
            have[o] -= 1;
            if have[o] == 0 {
                covered -= 1;
            }
            left += 1;
        }
    }
    if best.0 == 0 {
        return None;
    }
    let j = words.string_of(best.1);
    Some((j, best.1 - words.start_of(j), best.0))
}

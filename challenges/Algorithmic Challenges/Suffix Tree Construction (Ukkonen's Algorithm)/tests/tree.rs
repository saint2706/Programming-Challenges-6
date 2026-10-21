//! The suffix tree built from SA + LCP: structure, queries and applications against naive code.

use proptest::prelude::*;
use sufftree::lcp::lcp_naive;
use sufftree::sais::suffix_array_naive;
use sufftree::satree::{sa_lcp_bytes, tree_of_bytes};

/// Occurrences of `pattern` in `text` by brute force.
fn naive_find(text: &[u8], pattern: &[u8]) -> Vec<usize> {
    if pattern.is_empty() {
        return (0..=text.len()).collect();
    }
    (0..text.len().saturating_sub(pattern.len() - 1))
        .filter(|&i| text[i..].starts_with(pattern))
        .collect()
}

/// Longest repeated substring length by brute force.
fn naive_lrs(text: &[u8]) -> usize {
    let n = text.len();
    (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .map(|(i, j)| {
            text[i..]
                .iter()
                .zip(&text[j..])
                .take_while(|(a, b)| a == b)
                .count()
        })
        .max()
        .unwrap_or(0)
}

fn sa_with_terminator(text: &[u8]) -> Vec<u32> {
    let mut sa = vec![text.len() as u32];
    sa.extend(suffix_array_naive(text));
    sa
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1500))]

    #[test]
    fn tree_from_sa_is_a_valid_suffix_tree(text in prop::collection::vec(0u8..4, 0..120)) {
        let tree = tree_of_bytes(&text);
        tree.check_invariants();
        prop_assert_eq!(tree.suffix_array(), sa_with_terminator(&text));
        let (sa, lcp) = sa_lcp_bytes(&text);
        prop_assert_eq!(tree.lcp_array(), lcp.clone());
        // The LCP with the terminator is computed from scratch on the augmented text.
        let mut full: Vec<u32> = text.iter().map(|&b| u32::from(b) + 1).collect();
        full.push(0);
        prop_assert_eq!(lcp, lcp_naive(&full, &sa));
        prop_assert!(tree.node_count() - tree.leaf_count() <= tree.leaf_count());
    }

    #[test]
    fn queries_match_brute_force(
        text in prop::collection::vec(0u8..3, 0..80),
        pattern in prop::collection::vec(0u8..3, 0..6),
    ) {
        let tree = tree_of_bytes(&text);
        let want = naive_find(&text, &pattern);
        prop_assert_eq!(tree.contains(&pattern), !want.is_empty());
        prop_assert_eq!(tree.count(&pattern), want.len());
        prop_assert_eq!(tree.locate(&pattern), want);
    }

    #[test]
    fn longest_repeated_substring_is_longest_and_repeated(
        text in prop::collection::vec(0u8..3, 0..80),
    ) {
        let tree = tree_of_bytes(&text);
        let (start, len) = tree.longest_repeated_substring();
        prop_assert_eq!(len, naive_lrs(&text));
        if len > 0 {
            prop_assert!(naive_find(&text, &text[start..start + len]).len() >= 2);
        }
    }

    #[test]
    fn every_byte_value_is_usable(text in prop::collection::vec(any::<u8>(), 0..100)) {
        let tree = tree_of_bytes(&text);
        tree.check_invariants();
        for window in [1usize, 2, 5] {
            for start in (0..text.len().saturating_sub(window)).step_by(7) {
                let pat = &text[start..start + window];
                prop_assert_eq!(tree.locate(pat), naive_find(&text, pat));
            }
        }
    }
}

#[test]
fn the_classic_examples() {
    let banana = tree_of_bytes(b"banana");
    banana.check_invariants();
    assert_eq!(banana.locate(b"ana"), [1, 3]);
    assert_eq!(banana.count(b"a"), 3);
    assert!(!banana.contains(b"nab"));
    let (start, len) = banana.longest_repeated_substring();
    assert_eq!(&b"banana"[start..start + len], b"ana");
    // banana$ has 7 leaves and, besides the root, the internal nodes a, ana, na.
    assert_eq!(banana.leaf_count(), 7);
    assert_eq!(banana.node_count(), 7 + 3 + 1);

    let empty = tree_of_bytes(b"");
    empty.check_invariants();
    assert_eq!(empty.leaf_count(), 1);
    assert!(empty.contains(b""));
    assert!(!empty.contains(b"a"));
    assert_eq!(empty.longest_repeated_substring(), (0, 0));
}

/// A text of one repeated byte is the deepest tree possible; nothing may recurse.
#[test]
fn a_hundred_thousand_equal_bytes_do_not_overflow_the_stack() {
    let text = vec![b'a'; 100_000];
    let tree = tree_of_bytes(&text);
    assert_eq!(tree.node_count(), 2 * 100_001 - 1);
    assert_eq!(tree.count(b"aaaa"), 100_000 - 3);
    let (start, len) = tree.longest_repeated_substring();
    assert_eq!((len, start <= 1), (99_999, true));
}

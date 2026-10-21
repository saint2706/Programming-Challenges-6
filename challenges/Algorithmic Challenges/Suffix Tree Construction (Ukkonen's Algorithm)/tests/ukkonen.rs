//! Ukkonen's construction against the suffix-array route, and the live online tree after every
//! single appended byte.

use proptest::prelude::*;
use sufftree::satree::tree_of_bytes as tree_via_sais;
use sufftree::ukkonen::{HashChildren, ListChildren, OnlineSuffixTree, tree_of_bytes};

fn shape<S: sufftree::Symbols>(t: &sufftree::SuffixTree<S>) -> Vec<(u32, u32)> {
    t.nodes().iter().map(|n| (n.depth, n.end)).collect()
}

fn naive_find(text: &[u8], pattern: &[u8]) -> Vec<usize> {
    if pattern.is_empty() {
        return (0..=text.len()).collect();
    }
    (0..text.len().saturating_sub(pattern.len() - 1))
        .filter(|&i| text[i..].starts_with(pattern))
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn both_backends_build_the_same_tree_as_sais(text in prop::collection::vec(0u8..4, 0..150)) {
        let want = tree_via_sais(&text);
        let list = tree_of_bytes::<ListChildren>(&text);
        let hash = tree_of_bytes::<HashChildren>(&text);
        list.check_invariants();
        hash.check_invariants();
        prop_assert_eq!(shape(&list), shape(&want));
        prop_assert_eq!(shape(&hash), shape(&want));
        prop_assert_eq!(list.suffix_array(), want.suffix_array());
    }

    #[test]
    fn full_byte_alphabet(text in prop::collection::vec(any::<u8>(), 0..150)) {
        let want = tree_via_sais(&text);
        let got = tree_of_bytes::<ListChildren>(&text);
        got.check_invariants();
        prop_assert_eq!(shape(&got), shape(&want));
    }

    #[test]
    fn online_tree_answers_queries_after_every_byte(
        text in prop::collection::vec(0u8..3, 0..40),
        patterns in prop::collection::vec(prop::collection::vec(0u8..3, 0..6), 1..8),
    ) {
        let mut online = OnlineSuffixTree::new();
        for (i, &b) in text.iter().enumerate() {
            online.push(b);
            let prefix = &text[..=i];
            prop_assert_eq!(online.len(), prefix.len());
            for p in &patterns {
                let want = naive_find(prefix, p);
                prop_assert_eq!(online.contains(p), !want.is_empty(), "contains {:?} in {:?}", p, prefix);
                prop_assert_eq!(online.locate(p), want, "locate {:?} in {:?}", p, prefix);
            }
        }
        let frozen = online.finish();
        frozen.check_invariants();
        prop_assert_eq!(shape(&frozen), shape(&tree_via_sais(&text)));
    }
}

#[test]
fn banana_online() {
    let mut t = OnlineSuffixTree::new();
    t.extend_from_slice(b"banan");
    // "an" and "n" are suffixes of "banan" that have no leaf yet, but they must be found.
    assert_eq!(t.locate(b"an"), [1, 3]);
    assert_eq!(t.locate(b"n"), [2, 4]);
    t.push(b'a');
    assert_eq!(t.locate(b"ana"), [1, 3]);
    assert!(!t.contains(b"nab"));
    assert_eq!(t.finish().count(b"a"), 3);
}

#[test]
fn degenerate_inputs_do_not_recurse() {
    // A text of period p repeats everything but its first p symbols, so its longest repeat is n - p.
    let runs: [(Vec<u8>, usize); 2] = [
        (vec![b'a'; 100_000], 1),
        (
            (0..100_000u32).map(|i| b"ab"[(i % 2) as usize]).collect(),
            2,
        ),
    ];
    for (text, period) in runs {
        for tree in [
            tree_of_bytes::<ListChildren>(&text),
            tree_of_bytes::<HashChildren>(&text),
        ] {
            assert_eq!(tree.leaf_count(), text.len() + 1);
            assert_eq!(tree.longest_repeated_substring().1, text.len() - period);
        }
    }
}

#[test]
fn fibonacci_text_has_few_distinct_branch_depths() {
    let mut fib = (b"a".to_vec(), b"ab".to_vec());
    while fib.1.len() < 50_000 {
        let next = [fib.1.clone(), fib.0.clone()].concat();
        fib = (fib.1, next);
    }
    let a = tree_of_bytes::<ListChildren>(&fib.1);
    let b = tree_via_sais(&fib.1);
    assert_eq!(shape(&a), shape(&b));
}

mod pwotd {
    use super::*;
    use sufftree::Bytes;
    use sufftree::pwotd::{auto_prefix_len, build_with};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        #[test]
        fn builds_the_same_tree_for_every_prefix_length(
            text in prop::collection::vec(0u8..4, 0..150),
            p in 0usize..5,
        ) {
            let want = tree_via_sais(&text);
            let got = build_with(Bytes::new(text.clone()), Some(p));
            got.check_invariants();
            prop_assert_eq!(shape(&got), shape(&want));
        }

        #[test]
        fn full_byte_alphabet(text in prop::collection::vec(any::<u8>(), 0..150)) {
            let want = tree_via_sais(&text);
            let got = sufftree::pwotd::build(Bytes::new(text));
            got.check_invariants();
            prop_assert_eq!(shape(&got), shape(&want));
        }
    }

    #[test]
    fn degenerate_texts_and_the_automatic_prefix() {
        assert_eq!(auto_prefix_len(4, 1_000_000), 8); // 4^8 = 65,536
        assert_eq!(auto_prefix_len(256, 1_000_000), 2);
        assert_eq!(auto_prefix_len(1, 100), 0);
        assert_eq!(auto_prefix_len(4, 10), 1);
        // Quadratic worst cases stay correct (and small enough to be quick).
        for text in [
            vec![b'a'; 3000],
            (0..3000u32).map(|i| b"ab"[(i % 2) as usize]).collect(),
        ] {
            let got = sufftree::pwotd::build(Bytes::new(text.clone()));
            assert_eq!(shape(&got), shape(&tree_via_sais(&text)));
        }
    }
}

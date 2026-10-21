//! Searches, the array index and the suffix-tree applications against brute force.

use proptest::prelude::*;
use sufftree::apps::{longest_common_substring, matching_statistics, shortest_unique_substring};
use sufftree::array_index::{SuffixArrayIndex, longest_common_substring_array};
use sufftree::satree::{tree_of_bytes, tree_of_words};
use sufftree::tree::NO_LINK;
use sufftree::ukkonen::{ListChildren, Ukkonen};
use sufftree::{Symbols, Words};

fn find_all(text: &[u8], pattern: &[u8]) -> Vec<usize> {
    if pattern.is_empty() {
        return (0..=text.len()).collect();
    }
    (0..text.len().saturating_sub(pattern.len() - 1))
        .filter(|&i| text[i..].starts_with(pattern))
        .collect()
}

fn contains(text: &[u8], pattern: &[u8]) -> bool {
    pattern.is_empty() || text.windows(pattern.len()).any(|w| w == pattern)
}

/// Brute-force longest common substring length.
fn lcs_len(strings: &[Vec<u8>]) -> usize {
    let first = &strings[0];
    let mut best = 0;
    for i in 0..first.len() {
        for j in i + 1..=first.len() {
            if j - i > best && strings[1..].iter().all(|s| contains(s, &first[i..j])) {
                best = j - i;
            }
        }
    }
    best
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1500))]

    #[test]
    fn array_index_agrees_with_brute_force(
        text in prop::collection::vec(0u8..3, 0..80),
        pattern in prop::collection::vec(0u8..3, 0..6),
    ) {
        let index = SuffixArrayIndex::new(text.clone());
        let want = find_all(&text, &pattern);
        // The array has no entry for the empty suffix, so the empty pattern matches n positions.
        let want_len = if pattern.is_empty() { text.len() } else { want.len() };
        prop_assert_eq!(index.count(&pattern), want_len);
        prop_assert_eq!(index.contains(&pattern), !pattern.is_empty() && !want.is_empty() || pattern.is_empty() && !text.is_empty());
        if !pattern.is_empty() {
            prop_assert_eq!(index.locate(&pattern), want);
        }
        let (start, len) = index.longest_repeated_substring();
        prop_assert_eq!(len, tree_of_bytes(&text).longest_repeated_substring().1);
        if len > 0 {
            prop_assert!(find_all(&text, &text[start..start + len]).len() >= 2);
        }
    }

    #[test]
    fn longest_common_substring_matches_brute_force(
        strings in prop::collection::vec(prop::collection::vec(0u8..3, 1..25), 2..5),
    ) {
        let want = lcs_len(&strings);
        let tree = tree_of_words(&strings);
        tree.check_invariants();
        let from_tree = longest_common_substring(&tree);
        let from_array = longest_common_substring_array(&strings);
        prop_assert_eq!(from_tree.map_or(0, |t| t.2), want);
        prop_assert_eq!(from_array.map_or(0, |t| t.2), want);
        for (j, start, len) in [from_tree, from_array].into_iter().flatten() {
            let piece = &strings[j][start..start + len];
            prop_assert!(strings.iter().all(|s| contains(s, piece)));
        }
    }

    #[test]
    fn every_builder_agrees_on_generalized_text(
        strings in prop::collection::vec(prop::collection::vec(0u8..3, 0..20), 1..4),
    ) {
        let want = tree_of_words(&strings);
        let shape = |t: &sufftree::SuffixTree<Words>| {
            t.nodes().iter().map(|n| (n.depth, n.end)).collect::<Vec<_>>()
        };
        let words = Words::new(&strings);
        let ukk = Ukkonen::<ListChildren>::build(&words).freeze(words.clone());
        ukk.check_invariants();
        prop_assert_eq!(shape(&ukk), shape(&want));
        let top_down = sufftree::pwotd::build(words);
        top_down.check_invariants();
        prop_assert_eq!(shape(&top_down), shape(&want));
    }

    #[test]
    fn shortest_unique_substring_matches_brute_force(
        text in prop::collection::vec(0u8..3, 1..40),
    ) {
        let tree = tree_of_bytes(&text);
        let got = shortest_unique_substring(&tree);
        let want = (1..=text.len())
            .find(|&len| (0..=text.len() - len).any(|i| find_all(&text, &text[i..i + len]).len() == 1));
        prop_assert_eq!(got.map(|g| g.1), want);
        if let Some((start, len)) = got {
            prop_assert_eq!(find_all(&text, &text[start..start + len]).len(), 1);
        }
    }

    #[test]
    fn matching_statistics_match_brute_force(
        text in prop::collection::vec(0u8..3, 0..60),
        query in prop::collection::vec(0u8..4, 0..40),
    ) {
        let tree = tree_of_bytes(&text);
        let got = matching_statistics(&tree, &query);
        let want: Vec<usize> = (0..query.len())
            .map(|i| (0..=query.len() - i).rev().find(|&l| contains(&text, &query[i..i + l])).unwrap())
            .collect();
        prop_assert_eq!(got, want);
    }

    #[test]
    fn suffix_links_spell_the_path_minus_one_symbol(text in prop::collection::vec(0u8..3, 0..80)) {
        let tree = tree_of_bytes(&text);
        let links = tree.suffix_links();
        let nodes = tree.nodes();
        let n = tree.text().len();
        for v in 1..nodes.len() {
            if nodes[v].end as usize == v + 1 {
                prop_assert_eq!(links[v], NO_LINK);
                continue;
            }
            let u = links[v] as usize;
            prop_assert_eq!(nodes[u].depth, nodes[v].depth - 1);
            // The two nodes' paths: node v spells text[s..s + d]; its link spells text[s + 1..].
            let leaf = tree.suffixes_under(v as u32)[0];
            let leaf_u = tree.suffixes_under(u as u32);
            prop_assert!(leaf_u.contains(&(leaf + 1)) || leaf + 1 >= n);
        }
    }
}

#[test]
fn well_known_answers() {
    let strings: [&[u8]; 3] = [b"xabcdey", b"zabcdq", b"abcdw"];
    let tree = tree_of_words(&strings);
    let (j, start, len) = longest_common_substring(&tree).unwrap();
    assert_eq!(&strings[j][start..start + len], b"abcd");
    assert_eq!(longest_common_substring_array(&strings).unwrap().2, 4);
    assert!(longest_common_substring(&tree_of_words(&[b"abc".as_slice(), b"xyz"])).is_none());
    assert!(longest_common_substring_array(&[b"abc".as_slice(), b"xyz"]).is_none());

    let banana = tree_of_bytes(b"banana");
    // b appears once; so does the pair "ba" and every longer prefix.
    assert_eq!(shortest_unique_substring(&banana), Some((0, 1)));
    assert_eq!(matching_statistics(&banana, b"anab"), [3, 2, 1, 1]);
}

#[test]
fn array_index_edge_cases() {
    let empty = SuffixArrayIndex::new(Vec::new());
    assert!(!empty.contains(b"a"));
    assert_eq!(empty.longest_repeated_substring(), (0, 0));
    let aaa = SuffixArrayIndex::new(b"aaaa".to_vec());
    assert_eq!(aaa.locate(b"aa"), [0, 1, 2]);
    assert_eq!(aaa.count(b"aaaaa"), 0);
    assert_eq!(aaa.heap_bytes(), 8 * 4);
}

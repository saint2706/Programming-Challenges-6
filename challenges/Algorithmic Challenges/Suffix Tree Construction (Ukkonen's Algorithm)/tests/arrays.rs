//! SA-IS and the LCP algorithms against their naive references, and SA-IS against `libsais`.

use libsais::SuffixArrayConstruction;
use proptest::prelude::*;
use sufftree::lcp::{lcp_kasai, lcp_naive, lcp_phi};
use sufftree::sais::{suffix_array, suffix_array_naive, suffix_array_u8};

fn libsais_sa(text: &[u8]) -> Vec<u32> {
    let sa: Vec<i32> = SuffixArrayConstruction::for_text(text)
        .in_owned_buffer()
        .single_threaded()
        .run()
        .expect("libsais")
        .into_vec();
    sa.into_iter().map(|x| x as u32).collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn sais_matches_naive_on_tiny_alphabets(text in prop::collection::vec(0u8..3, 0..200)) {
        let sa = suffix_array_u8(&text);
        prop_assert_eq!(&sa, &suffix_array_naive(&text));
        let lcp = lcp_naive(&text, &sa);
        prop_assert_eq!(lcp_phi(&text, &sa), lcp.clone());
        prop_assert_eq!(lcp_kasai(&text, &sa), lcp);
    }

    #[test]
    fn sais_matches_naive_on_bytes(text in prop::collection::vec(any::<u8>(), 0..300)) {
        prop_assert_eq!(suffix_array_u8(&text), suffix_array_naive(&text));
    }

    #[test]
    fn sais_matches_naive_on_integer_alphabets(
        text in prop::collection::vec(0u32..50, 0..200),
    ) {
        let sa = suffix_array(&text, 50);
        prop_assert_eq!(&sa, &suffix_array_naive(&text));
        prop_assert_eq!(lcp_phi(&text, &sa), lcp_naive(&text, &sa));
    }

    #[test]
    fn sais_matches_libsais(text in prop::collection::vec(0u8..5, 1..2000)) {
        prop_assert_eq!(suffix_array_u8(&text), libsais_sa(&text));
    }
}

#[test]
fn degenerate_texts() {
    for n in [0usize, 1, 2, 3, 10, 257, 5000] {
        let ones = vec![7u8; n];
        assert_eq!(
            suffix_array_u8(&ones),
            suffix_array_naive(&ones),
            "all equal, n={n}"
        );
        let ramp_up: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
        assert_eq!(
            suffix_array_u8(&ramp_up),
            suffix_array_naive(&ramp_up),
            "ramp, n={n}"
        );
        let ramp_down: Vec<u8> = (0..n).map(|i| 255 - (i % 256) as u8).collect();
        assert_eq!(suffix_array_u8(&ramp_down), suffix_array_naive(&ramp_down));
        let period: Vec<u8> = (0..n).map(|i| b"ab"[i % 2]).collect();
        assert_eq!(suffix_array_u8(&period), suffix_array_naive(&period));
    }
}

/// Fibonacci strings are the classic stress for repeated structure: every level of the SA-IS
/// recursion has repeated names.
#[test]
fn fibonacci_and_thue_morse() {
    let mut fib = (b"a".to_vec(), b"ab".to_vec());
    while fib.1.len() < 20_000 {
        let next = [fib.1.clone(), fib.0.clone()].concat();
        fib = (fib.1, next);
    }
    assert_eq!(suffix_array_u8(&fib.1), libsais_sa(&fib.1));
    let tm: Vec<u8> = (0..20_000u32)
        .map(|i| b'a' + (i.count_ones() % 2) as u8)
        .collect();
    assert_eq!(suffix_array_u8(&tm), libsais_sa(&tm));
}

//! Range k-th smallest, rank and distinct-count queries: the persistent indexes and the wavelet
//! matrix against sorting and hashing.

use persegtree::oracle::{distinct_by_hashing, kth_by_sorting};
use persegtree::{DistinctIndex, History, KthIndex, SumI64, WaveletMatrix};
use proptest::prelude::*;

fn ranks(a: &[i64]) -> Vec<u32> {
    let mut s = a.to_vec();
    s.sort_unstable();
    s.dedup();
    a.iter()
        .map(|x| s.binary_search(x).unwrap() as u32)
        .collect()
}

#[test]
fn a_worked_example() {
    //            0  1  2  3  4  5  6
    let a = [5, 1, 4, 1, 5, 9, 2];
    let idx = KthIndex::new(&a);
    // a[1..5] = 1 4 1 5 -> sorted 1 1 4 5
    assert_eq!(idx.kth_smallest(1, 5, 0), Some(1));
    assert_eq!(idx.kth_smallest(1, 5, 1), Some(1));
    assert_eq!(idx.kth_smallest(1, 5, 2), Some(4));
    assert_eq!(idx.kth_smallest(1, 5, 3), Some(5));
    assert_eq!(idx.kth_smallest(1, 5, 4), None);
    assert_eq!(idx.count_below(1, 5, 4), 2);
    assert_eq!(idx.count_below(1, 5, 5), 3);
    assert_eq!(idx.count_below(0, 7, 100), 7);
    let d = DistinctIndex::new(&a);
    assert_eq!(d.distinct(0, 7), 5); // 5 1 4 9 2
    assert_eq!(d.distinct(1, 5), 3); // 1 4 5
    assert_eq!(d.distinct(2, 2), 0);
}

#[test]
fn degenerate_inputs() {
    let one = KthIndex::new(&[42]);
    assert_eq!(one.kth_smallest(0, 1, 0), Some(42));
    assert_eq!(one.kth_smallest(0, 0, 0), None);
    let empty = KthIndex::new(&[]);
    assert_eq!(empty.kth_smallest(0, 0, 0), None);
    assert_eq!(empty.count_below(0, 0, 5), 0);
    let same = KthIndex::new(&[7; 20]);
    assert_eq!(same.kth_smallest(3, 17, 13), Some(7));
    assert_eq!(same.kth_smallest(3, 17, 14), None);
    assert_eq!(DistinctIndex::new(&[7; 20]).distinct(0, 20), 1);
    assert_eq!(DistinctIndex::new(&[]).distinct(0, 0), 0);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(250))]

    #[test]
    fn kth_rank_and_distinct_match_the_oracles(
        a in prop::collection::vec(-30i64..30, 1..80),
        queries in prop::collection::vec((any::<usize>(), any::<usize>(), any::<usize>(), -40i64..40), 1..30),
    ) {
        let kth = KthIndex::new(&a);
        let distinct = DistinctIndex::new(&a);
        for (x, y, k, t) in queries {
            let (l, r) = (x % (a.len() + 1), y % (a.len() + 1));
            let (l, r) = (l.min(r), l.max(r));
            let k = k % (r - l + 2); // sometimes out of range on purpose
            prop_assert_eq!(kth.kth_smallest(l, r, k), kth_by_sorting(&a, l, r, k));
            let below = a[l..r].iter().filter(|&&v| v < t).count();
            prop_assert_eq!(kth.count_below(l, r, t), below);
            prop_assert_eq!(distinct.distinct(l, r), distinct_by_hashing(&a, l, r));
        }
    }

    #[test]
    fn the_wavelet_matrix_matches_the_persistent_index_and_the_oracle(
        a in prop::collection::vec(0u32..1000, 1..120),
        queries in prop::collection::vec((any::<usize>(), any::<usize>(), any::<usize>(), 0u32..1100, 0u32..1100), 1..30),
    ) {
        let wm = WaveletMatrix::new(&a);
        let signed: Vec<i64> = a.iter().map(|&v| i64::from(v)).collect();
        let idx = KthIndex::new(&signed);
        for (x, y, k, lo, hi) in queries {
            let (l, r) = (x % (a.len() + 1), y % (a.len() + 1));
            let (l, r) = (l.min(r), l.max(r));
            let k = k % (r - l + 2);
            let want = kth_by_sorting(&signed, l, r, k).map(|v| v as u32);
            prop_assert_eq!(wm.kth_smallest(l, r, k), want);
            prop_assert_eq!(idx.kth_smallest(l, r, k).map(|v| v as u32), want);
            let below = a[l..r].iter().filter(|&&v| v < lo).count();
            prop_assert_eq!(wm.count_below(l, r, lo), below);
            let in_range = a[l..r].iter().filter(|&&v| lo <= v && v < hi).count();
            prop_assert_eq!(wm.range_freq(l, r, lo, hi), in_range);
        }
    }
}

#[test]
fn wavelet_matrix_handles_full_width_values_and_degenerate_arrays() {
    let a = [u32::MAX, 0, 1 << 31, 12345, u32::MAX, 7];
    let wm = WaveletMatrix::new(&a);
    assert_eq!(wm.bits(), 32);
    let mut sorted = a.to_vec();
    sorted.sort_unstable();
    for (k, &want) in sorted.iter().enumerate() {
        assert_eq!(wm.kth_smallest(0, a.len(), k), Some(want));
    }
    assert_eq!(wm.kth_smallest(0, a.len(), a.len()), None);
    assert_eq!(wm.count_below(0, a.len(), u32::MAX), 4);
    assert_eq!(wm.count_below(0, a.len(), 0), 0);

    let zeros = WaveletMatrix::new(&[0; 5]);
    assert_eq!(zeros.bits(), 1);
    assert_eq!(zeros.kth_smallest(1, 4, 2), Some(0));
    assert_eq!(zeros.count_below(0, 5, 1), 5);
    assert_eq!(
        zeros.count_below(0, 5, 1000),
        5,
        "a threshold wider than the values counts everything"
    );

    let empty = WaveletMatrix::new(&[]);
    assert!(empty.is_empty());
    assert_eq!(empty.kth_smallest(0, 0, 0), None);
    assert_eq!(empty.count_below(0, 0, 5), 0);

    // Lengths around a 64-bit word boundary exercise the rank directory.
    for n in [63, 64, 65, 127, 128, 129] {
        let a: Vec<u32> = (0..n as u32)
            .map(|i| i.wrapping_mul(2654435761) % 97)
            .collect();
        let wm = WaveletMatrix::new(&a);
        let mut s = a.clone();
        s.sort_unstable();
        for k in (0..n).step_by(7) {
            assert_eq!(wm.kth_smallest(0, n, k), Some(s[k]), "n = {n}");
        }
        assert_eq!(
            wm.count_below(0, n, 50),
            a.iter().filter(|&&v| v < 50).count()
        );
    }
}

#[test]
fn the_wavelet_matrix_is_far_smaller_than_the_persistent_index() {
    let a: Vec<i64> = (0..20_000).map(|i| (i * 7919) % 5003).collect();
    let r = ranks(&a);
    let wm = WaveletMatrix::new(&r);
    let idx = KthIndex::new(&a);
    assert!(
        wm.bytes() * 20 < idx.bytes(),
        "wavelet {} bytes, persistent {} bytes",
        wm.bytes(),
        idx.bytes()
    );
}

#[test]
fn history_answers_as_of_any_time() {
    let mut h = History::<SumI64>::new(8);
    h.set(10, 0, 5);
    h.set(10, 1, 7); // same timestamp: a query at 10 sees both
    h.set(20, 0, 100);
    h.set(35, 3, 1);
    assert_eq!(h.get_at(0, 0), Some(0));
    assert_eq!(h.get_at(9, 0), Some(0));
    assert_eq!(h.get_at(10, 0), Some(5));
    assert_eq!(h.query_at(10, 0, 8), Some(12));
    assert_eq!(h.query_at(19, 0, 8), Some(12));
    assert_eq!(h.query_at(20, 0, 8), Some(107));
    assert_eq!(h.query_at(1_000, 0, 8), Some(108));
}

#[test]
fn history_forgets_old_versions_and_frees_their_nodes() {
    let n = 1 << 10;
    let mut h = History::<SumI64>::new(n);
    for t in 1..=500u64 {
        h.set(t * 10, (t as usize * 31) % n, t as i64);
    }
    let nodes_before = h.tree().live_nodes();
    let retained_before = h.retained();
    let retired = h.forget_before(4000);
    assert!(retired > 300);
    assert_eq!(h.retained(), retained_before - retired);
    assert!(h.tree().live_nodes() < nodes_before);
    // Times before the retained horizon are no longer answerable...
    assert_eq!(h.version_at(3999), None);
    assert_eq!(h.get_at(3000, 0), None);
    // ...and everything from the horizon on still is, with the right answers.
    let want: i64 = (1..=400u64).map(|t| t as i64).sum();
    assert_eq!(h.query_at(4000, 0, n), Some(want));
    assert_eq!(h.query_at(1_000_000, 0, n), Some((1..=500).sum()));
    h.compact();
    assert_eq!(h.query_at(4000, 0, n), Some(want));
    assert_eq!(h.tree().stats().slots, h.tree().stats().live_nodes);
}

#[test]
#[should_panic(expected = "time went backwards")]
fn history_rejects_out_of_order_timestamps() {
    let mut h = History::<SumI64>::new(4);
    h.set(10, 0, 1);
    h.set(9, 0, 2);
}

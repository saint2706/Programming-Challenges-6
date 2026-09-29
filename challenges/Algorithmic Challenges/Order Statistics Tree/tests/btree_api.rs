//! `OrderStatBTree`-specific API: ranges, neighbours, iteration, bulk-load,
//! `remove_at`, plus hand-picked edge cases.

use ost::OrderStatBTree;
use ost::rng::SplitMix64;

type Small = OrderStatBTree<u32, 2>;

fn random_tree(n: usize, domain: usize, seed: u64) -> (Small, Vec<u32>) {
    let mut rng = SplitMix64::new(seed);
    let mut t = Small::new();
    let mut v = Vec::new();
    for _ in 0..n {
        let k = rng.below(domain) as u32;
        t.insert(k);
        v.push(k);
    }
    v.sort_unstable();
    (t, v)
}

#[test]
fn empty_tree() {
    let t = Small::new();
    assert_eq!(t.len(), 0);
    assert!(t.is_empty());
    assert_eq!(t.select(0), None);
    assert_eq!(t.min(), None);
    assert_eq!(t.max(), None);
    assert_eq!(t.rank(&5), 0);
    assert_eq!(t.predecessor(&5), None);
    assert_eq!(t.successor(&5), None);
    assert_eq!(t.iter().count(), 0);
    assert_eq!(t.count_range(..), 0);
    assert_eq!(t.height(), 1);
    t.check_invariants().unwrap();
}

#[test]
fn single_element_and_remove_to_empty() {
    let mut t = Small::new();
    t.insert(7);
    assert_eq!(
        (t.min(), t.max(), t.select(0)),
        (Some(&7), Some(&7), Some(&7))
    );
    assert_eq!((t.rank(&7), t.rank_upper(&7), t.count(&7)), (0, 1, 1));
    assert_eq!(t.remove(&8), None);
    assert_eq!(t.remove(&7), Some(7));
    assert!(t.is_empty());
    assert_eq!(t.remove_at(0), None);
    t.check_invariants().unwrap();
}

#[test]
fn duplicates_are_a_multiset() {
    let mut t = Small::new();
    for _ in 0..50 {
        t.insert(3);
    }
    t.insert(1);
    t.insert(9);
    assert_eq!(t.len(), 52);
    assert_eq!((t.rank(&3), t.rank_upper(&3), t.count(&3)), (1, 51, 50));
    assert_eq!(t.select(25), Some(&3));
    for i in (0..50).rev() {
        assert_eq!(t.remove(&3), Some(3));
        assert_eq!(t.count(&3), i);
        t.check_invariants().unwrap();
    }
    assert!(!t.contains(&3));
    assert_eq!(t.iter().copied().collect::<Vec<_>>(), vec![1, 9]);
}

#[test]
fn predecessor_successor_are_strict() {
    let t: Small = [10, 20, 20, 30].into_iter().collect();
    assert_eq!(t.predecessor(&20), Some(&10));
    assert_eq!(t.successor(&20), Some(&30));
    assert_eq!(t.predecessor(&10), None);
    assert_eq!(t.successor(&30), None);
    assert_eq!(t.predecessor(&25), Some(&20));
    assert_eq!(t.successor(&25), Some(&30));
    assert_eq!(t.predecessor(&99), Some(&30));
    assert_eq!(t.successor(&0), Some(&10));
}

#[test]
fn ranges_and_iteration_match_a_sorted_vec() {
    let (t, v) = random_tree(600, 90, 1);
    t.check_invariants().unwrap();
    assert_eq!(t.iter().copied().collect::<Vec<_>>(), v);
    assert_eq!(t.iter().len(), v.len());
    let mut rng = SplitMix64::new(2);
    for _ in 0..300 {
        let (a, b) = (rng.below(100) as u32, rng.below(100) as u32);
        let expect = |lo_inc: bool, hi_inc: bool| -> Vec<u32> {
            v.iter()
                .copied()
                .filter(|&x| {
                    (if lo_inc { x >= a } else { x > a }) && (if hi_inc { x <= b } else { x < b })
                })
                .collect()
        };
        assert_eq!(
            t.range(a..b).copied().collect::<Vec<_>>(),
            expect(true, false)
        );
        assert_eq!(
            t.range(a..=b).copied().collect::<Vec<_>>(),
            expect(true, true)
        );
        assert_eq!(t.count_range(a..b), expect(true, false).len());
        assert_eq!(t.count_range(a..=b), expect(true, true).len());
        assert_eq!(t.range(a..).count(), v.iter().filter(|&&x| x >= a).count());
        assert_eq!(t.range(..b).count(), v.iter().filter(|&&x| x < b).count());
    }
}

#[test]
fn remove_at_matches_vec_remove() {
    let (mut t, mut v) = random_tree(400, 60, 3);
    let mut rng = SplitMix64::new(4);
    while !v.is_empty() {
        let i = rng.below(v.len());
        assert_eq!(t.remove_at(i), Some(v.remove(i)));
        t.check_invariants().unwrap();
    }
    assert!(t.is_empty());
}

#[test]
fn bulk_load_every_small_size() {
    // Exercises every (n, T) shape the packer can produce, incl. root splits.
    fn all_sizes<const T: usize>() {
        for n in 0..1500u32 {
            let t = OrderStatBTree::<u32, T>::from_sorted((0..n).collect());
            t.check_invariants()
                .unwrap_or_else(|e| panic!("T={T} n={n}: {e}"));
            assert_eq!(t.len(), n as usize);
            assert_eq!(
                t.iter().copied().collect::<Vec<_>>(),
                (0..n).collect::<Vec<_>>()
            );
        }
    }
    all_sizes::<2>();
    all_sizes::<3>();
    all_sizes::<4>();
}

#[test]
fn bulk_loaded_tree_supports_further_updates() {
    let mut t = OrderStatBTree::<u32, 3>::from_sorted((0..1000).map(|x| x * 2).collect());
    let mut rng = SplitMix64::new(5);
    let mut v: Vec<u32> = (0..1000).map(|x| x * 2).collect();
    for _ in 0..2000 {
        if rng.below(2) == 0 {
            let k = rng.below(2100) as u32;
            t.insert(k);
            let p = v.partition_point(|&x| x <= k);
            v.insert(p, k);
        } else if !v.is_empty() {
            let i = rng.below(v.len());
            assert_eq!(t.remove_at(i), Some(v.remove(i)));
        }
    }
    t.check_invariants().unwrap();
    assert_eq!(t.iter().copied().collect::<Vec<_>>(), v);
}

#[test]
#[should_panic(expected = "sorted")]
fn bulk_load_rejects_unsorted_input() {
    let _ = Small::from_sorted(vec![3, 1, 2]);
}

#[test]
fn works_with_non_copy_keys() {
    let mut t: OrderStatBTree<String, 2> = OrderStatBTree::new();
    for w in ["pear", "apple", "fig", "apple", "kiwi", "banana"] {
        t.insert(w.to_string());
    }
    assert_eq!(t.select(0).map(String::as_str), Some("apple"));
    assert_eq!(t.rank(&"fig".to_string()), 3);
    assert_eq!(t.remove(&"apple".to_string()).as_deref(), Some("apple"));
    assert_eq!(t.count(&"apple".to_string()), 1);
    t.check_invariants().unwrap();
}

#[test]
fn tree_stays_shallow_and_arena_recycles() {
    let mut t = OrderStatBTree::<u32, 16>::new();
    for k in 0..100_000 {
        t.insert(k);
    }
    // 100k keys, >= 15 keys per node => at most 5 levels.
    assert!(t.height() <= 5, "height {}", t.height());
    let peak = t.node_count();
    for k in 0..100_000 {
        assert!(t.remove(&k).is_some());
    }
    assert!(t.is_empty());
    for k in 0..100_000 {
        t.insert(k);
    }
    // Freed slots are reused instead of growing the arena forever.
    assert!(
        t.node_count() <= peak * 11 / 10,
        "{} vs {peak}",
        t.node_count()
    );
}

/// Small deterministic workload, cheap enough for `cargo +nightly miri test`.
#[test]
fn miri_smoke() {
    let mut t = OrderStatBTree::<u32, 2>::new();
    let mut v = Vec::new();
    let mut rng = SplitMix64::new(9);
    for _ in 0..120 {
        let k = rng.below(20) as u32;
        t.insert(k);
        v.push(k);
    }
    v.sort_unstable();
    for _ in 0..80 {
        let i = rng.below(v.len());
        assert_eq!(t.remove_at(i), Some(v.remove(i)));
        assert_eq!(
            t.select(i.min(v.len().saturating_sub(1))),
            v.get(i.min(v.len().saturating_sub(1)))
        );
    }
    t.check_invariants().unwrap();
    let b = OrderStatBTree::<u32, 2>::from_sorted((0..40).collect());
    b.check_invariants().unwrap();
    assert_eq!(b.range(5..10).count(), 5);
}

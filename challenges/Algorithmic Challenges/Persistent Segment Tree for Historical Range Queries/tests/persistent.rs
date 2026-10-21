//! The path-copying tree against the brute-force oracle, and the invariants its memory management
//! has to keep: how many nodes an update allocates, that retiring frees exactly what nothing else
//! reaches, and that compaction changes no answer.

use persegtree::oracle::Naive;
use persegtree::{MaxI64, MinI64, Monoid, Persistent, SumI64, SumU32, Version};
use proptest::prelude::*;

fn ceil_log2(n: usize) -> usize {
    n.next_power_of_two().trailing_zeros() as usize
}

/// A scripted operation, decoded from random bytes against the set of live versions.
#[derive(Clone, Debug)]
enum Op {
    Update { from: usize, pos: usize, value: i64 },
    Retire { which: usize },
    Compact,
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        6 => (any::<usize>(), any::<usize>(), -50i64..50).prop_map(|(from, pos, value)| Op::Update { from, pos, value }),
        2 => any::<usize>().prop_map(|which| Op::Retire { which }),
        1 => Just(Op::Compact),
    ]
}

/// Runs `ops` on a persistent tree and on the oracle, comparing every live version's folds after
/// every step.
fn run_model<M: Monoid<T = i64>>(
    init: Vec<i64>,
    counting: bool,
    sparse_start: bool,
    ops: &[Op],
    fold: impl Fn(i64, i64) -> i64 + Copy,
) {
    let n = init.len();
    let mut tree = if sparse_start {
        Persistent::<M>::new(n, counting)
    } else {
        Persistent::<M>::from_values(&init, counting)
    };
    let mut oracle = Naive::new(init);
    // `ids[i]` is the oracle index and the tree version of the same logical version.
    let mut live: Vec<usize> = vec![0];
    for op in ops {
        match *op {
            Op::Update { from, pos, value } => {
                let parent = live[from % live.len()];
                let v = tree.update(Version(parent as u32), pos % n, value);
                let o = oracle.update(parent, pos % n, value);
                assert_eq!(v.0 as usize, o);
                live.push(o);
            }
            Op::Retire { which } => {
                if live.len() > 1 {
                    let victim = live.remove(which % live.len());
                    assert!(tree.retire(Version(victim as u32)));
                    oracle.retire(victim);
                    assert!(!tree.is_live(Version(victim as u32)));
                }
            }
            Op::Compact => tree.compact(),
        }
        for &v in &live {
            let ver = Version(v as u32);
            for (l, r) in [(0, n), (0, n / 2), (n / 3, n), (n / 2, n / 2 + 1)] {
                let want = oracle.fold(v, l, r, M::identity(), fold);
                assert_eq!(tree.query(ver, l, r), want, "version {v} range {l}..{r}");
            }
            let probe = (v * 7) % n;
            assert_eq!(tree.get(ver, probe), oracle.array(v)[probe]);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(150))]

    #[test]
    fn sum_tree_matches_the_oracle(
        init in prop::collection::vec(-20i64..20, 1..40),
        ops in prop::collection::vec(op_strategy(), 0..50),
        counting in any::<bool>(),
    ) {
        run_model::<SumI64>(init, counting, false, &ops, |a, b| a + b);
    }

    #[test]
    fn min_and_max_trees_match_the_oracle(
        init in prop::collection::vec(-20i64..20, 1..40),
        ops in prop::collection::vec(op_strategy(), 0..50),
        counting in any::<bool>(),
    ) {
        run_model::<MinI64>(init.clone(), counting, false, &ops, i64::min);
        run_model::<MaxI64>(init, counting, false, &ops, i64::max);
    }

    #[test]
    fn sparse_trees_start_as_all_identity(
        n in 1usize..40,
        ops in prop::collection::vec(op_strategy(), 0..50),
        counting in any::<bool>(),
    ) {
        run_model::<SumI64>(vec![0; n], counting, true, &ops, |a, b| a + b);
        run_model::<MaxI64>(vec![i64::MIN; n], counting, true, &ops, i64::max);
    }
}

#[test]
fn an_update_allocates_exactly_depth_plus_one_nodes() {
    for k in 0..=10 {
        let n = 1usize << k;
        let mut t = Persistent::<SumI64>::from_values(&vec![1; n], false);
        assert_eq!(t.live_nodes(), 2 * n - 1);
        let mut v = t.first();
        for pos in [0, n - 1, n / 2] {
            let before = t.live_nodes();
            v = t.update(v, pos, 9);
            assert_eq!(t.live_nodes() - before, k + 1, "n = 2^{k}");
        }
    }
}

#[test]
fn an_update_allocates_at_most_ceil_log2_plus_one_nodes_for_any_n() {
    for n in [1, 2, 3, 5, 7, 100, 1000, 12345] {
        let mut t = Persistent::<SumI64>::new(n, false);
        let mut v = t.first();
        for pos in (0..n).step_by((n / 7).max(1)) {
            let before = t.live_nodes();
            v = t.update(v, pos, 1);
            assert!(
                t.live_nodes() - before <= ceil_log2(n) + 1,
                "n = {n}, pos = {pos}"
            );
        }
    }
}

#[test]
fn old_versions_are_untouched_by_new_ones_and_versions_branch() {
    let mut t = Persistent::<SumI64>::from_values(&[1, 2, 3, 4, 5], false);
    let v0 = t.first();
    let v1 = t.update(v0, 0, 100);
    let v2 = t.update(v0, 4, 500); // branches from v0, not from v1
    let v3 = t.update(v1, 4, 7);
    assert_eq!(t.query(v0, 0, 5), 15);
    assert_eq!(t.query(v1, 0, 5), 114);
    assert_eq!(t.query(v2, 0, 5), 510);
    assert_eq!(t.query(v3, 0, 5), 116);
    assert_eq!((t.get(v2, 0), t.get(v2, 4)), (1, 500));
}

#[test]
fn retiring_every_version_frees_every_node() {
    let mut t = Persistent::<SumI64>::from_values(&(0..100).collect::<Vec<_>>(), true);
    let mut vs = vec![t.first()];
    for i in 0..200 {
        let from = vs[(i * 7) % vs.len()];
        vs.push(t.update(from, (i * 13) % 100, i as i64));
    }
    assert!(t.live_nodes() > 199 * 7);
    // Retire in an order unrelated to creation, so sharing is released from every side.
    for i in (0..vs.len()).map(|i| (i * 37) % vs.len()) {
        if t.is_live(vs[i]) {
            t.retire(vs[i]);
        }
    }
    assert_eq!(t.live_nodes(), 0);
    assert_eq!(t.stats().live_versions, 0);
}

#[test]
fn freed_slots_are_reused_so_a_sliding_window_has_a_flat_footprint() {
    // A full tree, so the live set does not grow: once the window of versions is full, an update
    // allocates depth + 1 nodes and retiring the oldest version frees about as many, and the
    // allocator takes them from the free list. Live nodes and arena slots must both stay flat.
    let n = 1 << 12;
    let depth = ceil_log2(n) + 1;
    let window = 8;
    let mut t = Persistent::<SumI64>::from_values(&vec![0; n], true);
    let mut vs: std::collections::VecDeque<Version> = [t.first()].into();
    let mut at_200 = t.stats();
    for i in 0..4000usize {
        let v = t.update(*vs.back().unwrap(), (i * 2654435761) % n, i as i64);
        vs.push_back(v);
        if vs.len() > window {
            t.retire(vs.pop_front().unwrap());
        }
        if i == 200 {
            at_200 = t.stats();
        }
    }
    let end = t.stats();
    // The window keeps the current tree (2n - 1 nodes) plus at most depth private nodes for each
    // other version it holds.
    assert!(
        end.live_nodes <= 2 * n - 1 + window * depth,
        "live {}",
        end.live_nodes
    );
    assert!(end.live_nodes >= 2 * n - 1);
    assert!(
        end.slots <= at_200.slots + depth,
        "slots grew from {} to {}",
        at_200.slots,
        end.slots
    );
    assert_eq!(end.live_versions, window);
}

#[test]
fn without_counting_retire_leaks_until_compaction() {
    let mut t = Persistent::<SumI64>::from_values(&[1; 64], false);
    let mut v = t.first();
    let mut all = vec![v];
    for i in 0..50 {
        v = t.update(v, i, 2);
        all.push(v);
    }
    let before = t.live_nodes();
    for &old in &all[..50] {
        t.retire(old);
    }
    assert_eq!(t.live_nodes(), before, "nothing is freed without counting");
    let bytes_before = t.stats().bytes;
    t.compact();
    assert!(t.live_nodes() < before);
    // Memory comes back too: the arena was once sized from the pre-collection node count, which
    // without reference counts is stale, so compaction kept all of its old capacity.
    assert!(
        t.stats().bytes * 2 < bytes_before,
        "{} bytes after compaction, {} before",
        t.stats().bytes,
        bytes_before
    );
    assert_eq!(
        t.stats().slots,
        t.live_nodes(),
        "compaction leaves no free slots"
    );
    assert_eq!(t.query(v, 0, 64), 50 * 2 + 14);
}

#[test]
fn compaction_keeps_exactly_the_reachable_nodes_and_shrinks_the_arena() {
    let mut t = Persistent::<SumI64>::from_values(&[3; 256], true);
    let mut v = t.first();
    let mut keep = Vec::new();
    for i in 0..300 {
        v = t.update(v, (i * 11) % 256, i as i64);
        if i % 50 == 0 {
            keep.push(v);
        }
    }
    let answers: Vec<i64> = keep.iter().map(|&k| t.query(k, 0, 256)).collect();
    let all: Vec<Version> = (0..t.versions() as u32).map(Version).collect();
    for x in all {
        if !keep.contains(&x) && x != t.first() {
            t.retire(x);
        }
    }
    let slots_before = t.stats().slots;
    t.compact();
    let stats = t.stats();
    assert!(stats.slots < slots_before);
    assert_eq!(stats.slots, stats.live_nodes);
    for (k, want) in keep.iter().zip(answers) {
        assert_eq!(t.query(*k, 0, 256), want);
    }
    // Counts were rebuilt correctly: retiring everything still frees everything.
    for &k in &keep {
        t.retire(k);
    }
    t.retire(t.first());
    assert_eq!(t.live_nodes(), 0);
}

#[test]
#[should_panic(expected = "retired")]
fn querying_a_retired_version_panics_with_a_clear_message() {
    let mut t = Persistent::<SumI64>::from_values(&[1, 2, 3], true);
    let v = t.update(t.first(), 0, 9);
    t.retire(v);
    let _ = t.query(v, 0, 3);
}

#[test]
fn retiring_twice_reports_it() {
    let mut t = Persistent::<SumU32>::new(4, true);
    let v = t.update(t.first(), 1, 1);
    assert!(t.retire(v));
    assert!(!t.retire(v));
}

#[test]
fn ranges_and_positions_are_bounds_checked() {
    let t = Persistent::<SumI64>::from_values(&[1, 2, 3], false);
    assert_eq!(t.query(t.first(), 2, 2), 0);
    assert!(
        std::panic::catch_unwind(|| {
            let t = Persistent::<SumI64>::from_values(&[1, 2, 3], false);
            t.query(t.first(), 1, 4)
        })
        .is_err()
    );
}

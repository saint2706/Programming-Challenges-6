//! Range updates (permanent marks and push-down) and the fat-node tree, against the oracle.

use persegtree::lazy::{Aggregate, Max, Min, Sum};
use persegtree::oracle::Naive;
use persegtree::{FatTree, LazyAdd, PushDown, Version};
use proptest::prelude::*;

fn ceil_log2(n: usize) -> usize {
    n.next_power_of_two().trailing_zeros() as usize
}

#[derive(Clone, Debug)]
enum Op {
    Add {
        from: usize,
        l: usize,
        r: usize,
        d: i64,
    },
    Assign {
        from: usize,
        l: usize,
        r: usize,
        x: i64,
    },
    Retire {
        which: usize,
    },
    Compact,
}

fn op_strategy(with_assign: bool) -> impl Strategy<Value = Op> {
    let add = (any::<usize>(), any::<usize>(), any::<usize>(), -9i64..9)
        .prop_map(|(from, l, r, d)| Op::Add { from, l, r, d });
    let assign = (any::<usize>(), any::<usize>(), any::<usize>(), -9i64..9)
        .prop_map(|(from, l, r, x)| Op::Assign { from, l, r, x });
    let rest = prop_oneof![
        2 => any::<usize>().prop_map(|which| Op::Retire { which }),
        1 => Just(Op::Compact),
    ];
    if with_assign {
        prop_oneof![5 => add, 4 => assign, 3 => rest].boxed()
    } else {
        prop_oneof![8 => add, 3 => rest].boxed()
    }
}

fn span(n: usize, a: usize, b: usize) -> (usize, usize) {
    let (a, b) = (a % (n + 1), b % (n + 1));
    (a.min(b), a.max(b))
}

/// Drives `LazyAdd<G>` against the oracle.
fn run_lazy<G: Aggregate>(
    init: Vec<i64>,
    counting: bool,
    ops: &[Op],
    fold: impl Fn(i64, i64) -> i64 + Copy,
) {
    let n = init.len();
    let mut tree = LazyAdd::<G>::new(&init, counting);
    let mut oracle = Naive::new(init);
    let mut live = vec![0usize];
    for op in ops {
        match *op {
            Op::Add { from, l, r, d } => {
                let (l, r) = span(n, l, r);
                let parent = live[from % live.len()];
                let v = tree.range_add(Version(parent as u32), l, r, d);
                let o = oracle.map_range(parent, l, r, |x| x + d);
                assert_eq!(v.0 as usize, o);
                live.push(o);
            }
            Op::Assign { .. } => {}
            Op::Retire { which } => {
                if live.len() > 1 {
                    let victim = live.remove(which % live.len());
                    tree.retire(Version(victim as u32));
                    oracle.retire(victim);
                }
            }
            Op::Compact => tree.compact(),
        }
        for &v in &live {
            for (l, r) in [
                (0, n),
                (0, n / 2),
                (n / 3, n),
                (n / 2, n / 2 + 1),
                (n / 4, 3 * n / 4),
            ] {
                if l < r {
                    let want = oracle.fold(v, l, r, G::identity(), fold);
                    assert_eq!(
                        tree.query(Version(v as u32), l, r),
                        want,
                        "version {v} range {l}..{r}"
                    );
                }
            }
        }
    }
}

/// Drives `PushDown` against the oracle, with assign as well as add.
fn run_push_down(init: Vec<i64>, counting: bool, ops: &[Op]) {
    let n = init.len();
    let mut tree = PushDown::new(&init, counting);
    let mut oracle = Naive::new(init);
    let mut live = vec![0usize];
    for op in ops {
        match *op {
            Op::Add { from, l, r, d } => {
                let (l, r) = span(n, l, r);
                let parent = live[from % live.len()];
                let v = tree.range_add(Version(parent as u32), l, r, d);
                let o = oracle.map_range(parent, l, r, |x| x + d);
                assert_eq!(v.0 as usize, o);
                live.push(o);
            }
            Op::Assign { from, l, r, x } => {
                let (l, r) = span(n, l, r);
                let parent = live[from % live.len()];
                let v = tree.range_assign(Version(parent as u32), l, r, x);
                let o = oracle.map_range(parent, l, r, |_| x);
                assert_eq!(v.0 as usize, o);
                live.push(o);
            }
            Op::Retire { which } => {
                if live.len() > 1 {
                    let victim = live.remove(which % live.len());
                    tree.retire(Version(victim as u32));
                    oracle.retire(victim);
                }
            }
            Op::Compact => tree.compact(),
        }
        for &v in &live {
            for (l, r) in [
                (0, n),
                (0, n / 2),
                (n / 3, n),
                (n / 2, n / 2 + 1),
                (n / 4, 3 * n / 4),
            ] {
                if l < r {
                    let want = oracle.fold(v, l, r, 0, |a, b| a + b);
                    assert_eq!(
                        tree.query(Version(v as u32), l, r),
                        want,
                        "version {v} range {l}..{r}"
                    );
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(150))]

    #[test]
    fn permanent_marks_match_the_oracle_for_sum_min_and_max(
        init in prop::collection::vec(-20i64..20, 1..40),
        ops in prop::collection::vec(op_strategy(false), 0..40),
        counting in any::<bool>(),
    ) {
        run_lazy::<Sum>(init.clone(), counting, &ops, |a, b| a + b);
        run_lazy::<Min>(init.clone(), counting, &ops, i64::min);
        run_lazy::<Max>(init, counting, &ops, i64::max);
    }

    #[test]
    fn push_down_matches_the_oracle_with_add_and_assign(
        init in prop::collection::vec(-20i64..20, 1..40),
        ops in prop::collection::vec(op_strategy(true), 0..40),
        counting in any::<bool>(),
    ) {
        run_push_down(init, counting, &ops);
    }
}

#[test]
fn assign_and_add_do_not_commute_and_push_down_gets_the_order_right() {
    let mut t = PushDown::new(&[1, 1, 1, 1, 1, 1, 1, 1], false);
    let a = t.range_assign(t.first(), 0, 8, 10); // 10 10 10 10 10 10 10 10
    let b = t.range_add(a, 2, 6, 5); //            10 10 15 15 15 15 10 10
    let c = t.range_assign(b, 3, 5, 0); //         10 10 15  0  0 15 10 10
    let d = t.range_add(c, 0, 8, 1); //            11 11 16  1  1 16 11 11
    assert_eq!(t.query(a, 0, 8), 80);
    assert_eq!(t.query(b, 0, 8), 100);
    assert_eq!(t.query(c, 0, 8), 70);
    assert_eq!(t.query(d, 0, 8), 78);
    assert_eq!(t.query(d, 2, 6), 34);
    // The other order on the same operations: add then assign overwrites the add.
    let e = t.range_add(t.first(), 2, 6, 5);
    let f = t.range_assign(e, 0, 8, 10);
    assert_eq!(t.query(e, 0, 8), 28);
    assert_eq!(t.query(f, 0, 8), 80);
}

#[test]
fn marks_stay_put_so_an_update_copies_only_the_nodes_it_visits() {
    for n in [1usize << 10, 1000, 4097] {
        let init = vec![0i64; n];
        let mut t = LazyAdd::<Sum>::new(&init, false);
        let mut v = t.first();
        let bound = 4 * (ceil_log2(n) + 1);
        for i in 0..200usize {
            let (l, r) = span(n, i * 7919, i * 104729 + 13);
            let tree = &mut t;
            let before = tree.live_nodes();
            v = tree.range_add(v, l, r, 1);
            let made = tree.live_nodes() - before;
            assert!(
                made <= bound,
                "n = {n}: range {l}..{r} made {made} nodes, bound {bound}"
            );
        }
    }
}

#[test]
fn push_down_never_allocates_fewer_nodes_than_permanent_marks_for_the_same_updates() {
    let n = 1 << 12;
    let init = vec![0i64; n];
    let mut permanent = LazyAdd::<Sum>::new(&init, false);
    let mut pushing = PushDown::new(&init, false);
    let (mut vp, mut vq) = (permanent.first(), pushing.first());
    let (mut total_permanent, mut total_pushing) = (0, 0);
    for i in 0..400usize {
        let (l, r) = span(n, i * 7919 + 1, i * 104729 + 977);
        let before = permanent.live_nodes();
        vp = permanent.range_add(vp, l, r, 3);
        let a = permanent.live_nodes() - before;
        let before = pushing.live_nodes();
        vq = pushing.range_add(vq, l, r, 3);
        let b = pushing.live_nodes() - before;
        assert!(
            b >= a,
            "update {i}: push-down made {b}, permanent marks {a}"
        );
        total_permanent += a;
        total_pushing += b;
    }
    assert!(
        total_pushing > total_permanent,
        "pending tags should force sibling copies"
    );
    // And they agree on every answer.
    for (l, r) in [(0, n), (100, 3000), (5, 6)] {
        assert_eq!(permanent.query(vp, l, r), pushing.query(vq, l, r));
    }
}

#[test]
fn fat_tree_matches_the_oracle_on_a_single_timeline() {
    for n in [1usize, 2, 3, 7, 8, 100] {
        let init: Vec<i64> = (0..n as i64).map(|i| i * 3 - 5).collect();
        let mut fat = FatTree::new(&init);
        let mut oracle = Naive::new(init);
        for step in 0..60usize {
            let pos = (step * 17 + 3) % n;
            let value = (step as i64) * 11 - 100;
            let v = fat.update(pos, value);
            let o = oracle.update(step, pos, value);
            assert_eq!(v as usize, o);
            for version in [0, step / 2, step + 1] {
                for (l, r) in [(0, n), (0, n / 2), (n / 3, n), (n / 2, n / 2 + 1)] {
                    if l < r {
                        let want = oracle.fold(version, l, r, 0, |a, b| a + b);
                        assert_eq!(
                            fat.query(version as u32, l, r),
                            want,
                            "n = {n} version {version}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn a_fat_node_update_adds_one_entry_per_ancestor_the_same_count_path_copying_allocates() {
    let n = 1usize << 10;
    let mut fat = FatTree::new(&vec![0; n]);
    let before = fat.entries();
    for i in 0..100 {
        fat.update((i * 31) % n, i as i64 + 1);
    }
    assert_eq!(fat.entries() - before, 100 * (ceil_log2(n) + 1));
}

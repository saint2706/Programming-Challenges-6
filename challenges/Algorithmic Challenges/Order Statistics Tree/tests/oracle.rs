//! Property tests: every structure must agree with a sorted-`Vec` oracle on
//! random operation sequences, and keep its own invariants after every step.

use ost::{
    AvlTree, Fenwick, NaiveBTreeMap, OrderStatBTree, OrderStatistics, RedBlackTree, SortedVec,
    Treap,
};
use proptest::prelude::*;

#[derive(Clone, Debug)]
enum Op {
    Insert(usize),
    Remove(usize),
    Rank(usize),
    Select(usize),
}

fn ops(domain: usize, max_len: usize) -> impl Strategy<Value = Vec<Op>> {
    let op = prop_oneof![
        4 => (0..domain).prop_map(Op::Insert),
        3 => (0..domain).prop_map(Op::Remove),
        2 => (0..domain + 2).prop_map(Op::Rank),
        2 => (0..4 * domain).prop_map(Op::Select),
    ];
    prop::collection::vec(op, 0..max_len)
}

fn run<S: OrderStatistics<usize>>(mut s: S, ops: &[Op], check_every_step: bool) {
    let mut oracle: Vec<usize> = Vec::new();
    for op in ops {
        match *op {
            Op::Insert(k) => {
                s.insert(k);
                let p = oracle.partition_point(|x| *x <= k);
                oracle.insert(p, k);
            }
            Op::Remove(k) => {
                let p = oracle.partition_point(|x| *x < k);
                let expected = oracle.get(p) == Some(&k);
                if expected {
                    oracle.remove(p);
                }
                assert_eq!(s.remove(&k), expected, "remove({k})");
            }
            Op::Rank(k) => {
                assert_eq!(s.rank(&k), oracle.partition_point(|x| *x < k), "rank({k})");
            }
            Op::Select(i) => assert_eq!(s.select(i), oracle.get(i).copied(), "select({i})"),
        }
        assert_eq!(s.len(), oracle.len());
        if check_every_step {
            s.validate().unwrap();
        }
    }
    s.validate().unwrap();
    for (i, k) in oracle.iter().enumerate() {
        assert_eq!(s.select(i), Some(*k));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    // Tiny key domain => heavy duplication and lots of rebalancing.
    #[test]
    fn btree_t2_dense(o in ops(24, 400)) { run(OrderStatBTree::<usize, 2>::new(), &o, true); }
    #[test]
    fn btree_t3_dense(o in ops(40, 400)) { run(OrderStatBTree::<usize, 3>::new(), &o, true); }
    #[test]
    fn btree_t4_wide(o in ops(500, 600)) { run(OrderStatBTree::<usize, 4>::new(), &o, true); }
    #[test]
    fn btree_t16_wide(o in ops(5000, 1500)) { run(OrderStatBTree::<usize, 16>::new(), &o, false); }
    #[test]
    fn avl_dense(o in ops(40, 400)) { run(AvlTree::new(), &o, true); }
    #[test]
    fn avl_wide(o in ops(2000, 800)) { run(AvlTree::new(), &o, true); }
    #[test]
    fn treap_dense(o in ops(40, 400)) { run(Treap::new(), &o, true); }
    #[test]
    fn treap_wide(o in ops(2000, 800)) { run(Treap::new(), &o, true); }
    #[test]
    fn redblack_dense(o in ops(40, 400)) { run(RedBlackTree::new(), &o, true); }
    #[test]
    fn redblack_wide(o in ops(2000, 800)) { run(RedBlackTree::new(), &o, true); }
    #[test]
    fn sorted_vec(o in ops(100, 400)) { run(SortedVec::new(), &o, true); }
    #[test]
    fn naive_btreemap(o in ops(100, 400)) { run(NaiveBTreeMap::new(), &o, true); }
    #[test]
    fn fenwick(o in ops(100, 400)) { run(Fenwick::new(128), &o, true); }
}

#[test]
fn ascending_and_descending_insert_then_drain() {
    // Sequential patterns are the classic way to break rebalancing code.
    fn drive<S: OrderStatistics<usize>>(mut s: S) {
        for k in 0..500 {
            s.insert(k);
            s.validate().unwrap();
        }
        for k in (0..500).rev() {
            assert!(s.remove(&k));
            s.validate().unwrap();
        }
        assert!(s.is_empty());
        for k in (0..500).rev() {
            s.insert(k);
        }
        for k in 0..500 {
            assert_eq!(s.select(0), Some(k));
            assert!(s.remove(&k));
            s.validate().unwrap();
        }
        assert!(s.is_empty());
    }
    drive(OrderStatBTree::<usize, 2>::new());
    drive(OrderStatBTree::<usize, 5>::new());
    drive(OrderStatBTree::<usize, 16>::new());
    drive(AvlTree::new());
    drive(Treap::new());
    drive(RedBlackTree::new());
}

//! Every structure must agree with the walk-the-graph oracle on every operation it supports,
//! on arbitrary link/cut sequences.

use lct::shapes::{Rng, Shape, generate};
use lct::{
    EulerTourTree, Forest, NaiveForest, PathLct, PathOps, Stats, SubtreeLct, SubtreeOps, Sum,
    Weighted,
};
use proptest::prelude::*;

/// What a structure can do, with `None` meaning "not supported" so one harness covers them all.
trait Caps: Weighted {
    fn path_query(&mut self, _: usize, _: usize) -> Option<Option<Stats>> {
        None
    }
    fn path_add(&mut self, _: usize, _: usize, _: i64) -> Option<bool> {
        None
    }
    fn subtree(&mut self, _: usize, _: usize) -> Option<Option<Sum>> {
        None
    }
    fn component(&mut self, _: usize) -> Option<Sum> {
        None
    }
    fn check(&self) {}
}

impl Caps for NaiveForest {
    fn path_query(&mut self, u: usize, v: usize) -> Option<Option<Stats>> {
        Some(PathOps::path_query(self, u, v))
    }
    fn path_add(&mut self, u: usize, v: usize, d: i64) -> Option<bool> {
        Some(PathOps::path_add(self, u, v, d))
    }
    fn subtree(&mut self, p: usize, v: usize) -> Option<Option<Sum>> {
        Some(SubtreeOps::subtree_query(self, p, v))
    }
    fn component(&mut self, v: usize) -> Option<Sum> {
        Some(SubtreeOps::component(self, v))
    }
}

impl Caps for PathLct {
    fn path_query(&mut self, u: usize, v: usize) -> Option<Option<Stats>> {
        Some(PathOps::path_query(self, u, v))
    }
    fn path_add(&mut self, u: usize, v: usize, d: i64) -> Option<bool> {
        Some(PathOps::path_add(self, u, v, d))
    }
}

impl Caps for SubtreeLct {
    fn path_query(&mut self, u: usize, v: usize) -> Option<Option<Stats>> {
        Some(PathOps::path_query(self, u, v))
    }
    fn path_add(&mut self, u: usize, v: usize, d: i64) -> Option<bool> {
        Some(PathOps::path_add(self, u, v, d))
    }
    fn subtree(&mut self, p: usize, v: usize) -> Option<Option<Sum>> {
        Some(SubtreeOps::subtree_query(self, p, v))
    }
    fn component(&mut self, v: usize) -> Option<Sum> {
        Some(SubtreeOps::component(self, v))
    }
}

impl Caps for EulerTourTree {
    fn subtree(&mut self, p: usize, v: usize) -> Option<Option<Sum>> {
        Some(SubtreeOps::subtree_query(self, p, v))
    }
    fn component(&mut self, v: usize) -> Option<Sum> {
        Some(SubtreeOps::component(self, v))
    }
    fn check(&self) {
        self.check_invariants();
    }
}

#[derive(Clone, Debug)]
enum Op {
    Link(usize, usize),
    Cut(usize, usize),
    /// Cut the k-th edge currently present, so most cuts are valid.
    CutNth(usize),
    Connected(usize, usize),
    Set(usize, i64),
    Get(usize),
    PathQuery(usize, usize),
    PathAdd(usize, usize, i64),
    Subtree(usize, usize),
    /// Subtree across the k-th present edge, in a random orientation.
    SubtreeNth(usize, bool),
    Component(usize),
}

fn op_strategy() -> impl Strategy<Value = Op> {
    let v = any::<usize>();
    prop_oneof![
        6 => (v, v).prop_map(|(a, b)| Op::Link(a, b)),
        1 => (v, v).prop_map(|(a, b)| Op::Cut(a, b)),
        3 => v.prop_map(Op::CutNth),
        3 => (v, v).prop_map(|(a, b)| Op::Connected(a, b)),
        2 => (v, -1000i64..1000).prop_map(|(a, x)| Op::Set(a, x)),
        1 => v.prop_map(Op::Get),
        3 => (v, v).prop_map(|(a, b)| Op::PathQuery(a, b)),
        3 => (v, v, -50i64..50).prop_map(|(a, b, d)| Op::PathAdd(a, b, d)),
        1 => (v, v).prop_map(|(a, b)| Op::Subtree(a, b)),
        3 => (v, any::<bool>()).prop_map(|(k, flip)| Op::SubtreeNth(k, flip)),
        1 => v.prop_map(Op::Component),
    ]
}

fn assert_same<S: Caps + Forest>(
    n: usize,
    values: &[Option<i64>],
    ops: &[Op],
    with_path_add: bool,
) {
    let mut oracle = NaiveForest::from_values(values);
    let mut subject = S::from_values(values);
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for (step, op) in ops.iter().enumerate() {
        let ctx = format!("step {step}: {op:?}");
        match *op {
            Op::Link(a, b) => {
                let (a, b) = (a % n, b % n);
                let want = oracle.link(a, b);
                assert_eq!(subject.link(a, b), want, "{ctx}");
                if want {
                    edges.push((a, b));
                }
            }
            Op::Cut(a, b) => {
                let (a, b) = (a % n, b % n);
                let want = oracle.cut(a, b);
                assert_eq!(subject.cut(a, b), want, "{ctx}");
                if want {
                    edges.retain(|&(x, y)| !((x, y) == (a, b) || (x, y) == (b, a)));
                }
            }
            Op::CutNth(k) => {
                if edges.is_empty() {
                    continue;
                }
                let (a, b) = edges.swap_remove(k % edges.len());
                assert!(oracle.cut(a, b));
                assert!(subject.cut(a, b), "{ctx}: ({a},{b})");
            }
            Op::Connected(a, b) => {
                let (a, b) = (a % n, b % n);
                assert_eq!(subject.connected(a, b), oracle.connected(a, b), "{ctx}");
            }
            Op::Set(a, x) => {
                oracle.set(a % n, x);
                subject.set(a % n, x);
            }
            Op::Get(a) => assert_eq!(subject.get(a % n), oracle.get(a % n), "{ctx}"),
            Op::PathQuery(a, b) => {
                let (a, b) = (a % n, b % n);
                if let Some(got) = subject.path_query(a, b) {
                    assert_eq!(got, PathOps::path_query(&mut oracle, a, b), "{ctx}");
                }
            }
            Op::PathAdd(a, b, d) => {
                let (a, b) = (a % n, b % n);
                if !with_path_add {
                    continue;
                }
                if let Some(got) = subject.path_add(a, b, d) {
                    assert_eq!(got, PathOps::path_add(&mut oracle, a, b, d), "{ctx}");
                }
            }
            Op::Subtree(a, b) => {
                let (a, b) = (a % n, b % n);
                if let Some(got) = subject.subtree(a, b) {
                    assert_eq!(got, SubtreeOps::subtree_query(&mut oracle, a, b), "{ctx}");
                }
            }
            Op::SubtreeNth(k, flip) => {
                if edges.is_empty() {
                    continue;
                }
                let (a, b) = edges[k % edges.len()];
                let (p, v) = if flip { (a, b) } else { (b, a) };
                if let Some(got) = subject.subtree(p, v) {
                    let want = SubtreeOps::subtree_query(&mut oracle, p, v);
                    assert!(want.is_some());
                    assert_eq!(got, want, "{ctx}: ({p},{v})");
                }
            }
            Op::Component(a) => {
                if let Some(got) = subject.component(a % n) {
                    assert_eq!(got, SubtreeOps::component(&mut oracle, a % n), "{ctx}");
                }
            }
        }
        subject.check();
    }
}

fn case() -> impl Strategy<Value = (usize, Vec<Option<i64>>, Vec<Op>)> {
    (2usize..=24).prop_flat_map(|n| {
        (
            Just(n),
            prop::collection::vec(prop::option::weighted(0.85, -1000i64..1000), n),
            prop::collection::vec(op_strategy(), 1..250),
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    #[test]
    fn path_lct_matches_oracle((n, values, ops) in case()) {
        assert_same::<PathLct>(n, &values, &ops, true);
    }

    #[test]
    fn subtree_lct_matches_oracle((n, values, ops) in case()) {
        assert_same::<SubtreeLct>(n, &values, &ops, true);
    }

    #[test]
    fn euler_tour_tree_matches_oracle((n, values, ops) in case()) {
        // The tour has no path add, but set and subtree aggregates must still agree.
        assert_same::<EulerTourTree>(n, &values, &ops, false);
    }
}

#[test]
fn euler_tour_tree_subtree_min_and_max_match_oracle() {
    let n = 60;
    let mut rng = Rng::new(7);
    let values: Vec<Option<i64>> = (0..n).map(|_| Some(rng.range(-500, 500))).collect();
    let mut oracle = NaiveForest::from_values(&values);
    let mut ett = EulerTourTree::from_values(&values);
    let mut edges = Vec::new();
    for step in 0..4000 {
        if rng.below(3) == 0 && !edges.is_empty() {
            let (a, b) = edges.swap_remove(rng.below(edges.len()));
            assert!(oracle.cut(a, b) && ett.cut(a, b));
        } else {
            let (a, b) = (rng.below(n), rng.below(n));
            let want = oracle.link(a, b);
            assert_eq!(ett.link(a, b), want);
            if want {
                edges.push((a, b));
            }
        }
        if !edges.is_empty() {
            let (a, b) = edges[rng.below(edges.len())];
            for (p, v) in [(a, b), (b, a)] {
                assert_eq!(
                    ett.subtree_stats(p, v),
                    oracle.subtree_stats(p, v),
                    "step {step}"
                );
                ett.check_invariants();
            }
        }
    }
}

/// Deterministic soak on every shape: build the tree, then churn it hard.
#[test]
fn soak_every_shape_against_the_oracle() {
    for (i, shape) in Shape::ALL.into_iter().enumerate() {
        let n = 150;
        let edges = generate(shape, n, 11 + i as u64);
        let mut rng = Rng::new(99 + i as u64);
        let values: Vec<Option<i64>> = (0..n).map(|_| Some(rng.range(-100, 100))).collect();
        let mut oracle = NaiveForest::from_values(&values);
        let mut path = PathLct::from_values(&values);
        let mut sub = SubtreeLct::from_values(&values);
        let mut ett = EulerTourTree::from_values(&values);
        for &(a, b) in &edges {
            assert!(oracle.link(a, b) && path.link(a, b) && sub.link(a, b) && ett.link(a, b));
        }
        let mut current = edges;
        for step in 0..3000 {
            let k = rng.below(current.len());
            let (u, v) = current[k];
            assert!(oracle.cut(u, v) && path.cut(u, v) && sub.cut(u, v) && ett.cut(u, v));
            let w = loop {
                let w = rng.below(n);
                if !oracle.connected(u, w) {
                    break w;
                }
            };
            assert!(oracle.link(u, w) && path.link(u, w) && sub.link(u, w) && ett.link(u, w));
            current[k] = (u, w);
            let (a, b) = (rng.below(n), rng.below(n));
            let d = rng.range(-5, 5);
            assert_eq!(
                PathOps::path_add(&mut path, a, b, d),
                PathOps::path_add(&mut oracle, a, b, d)
            );
            assert!(PathOps::path_add(&mut sub, a, b, d));
            let want = PathOps::path_query(&mut oracle, a, b);
            assert_eq!(
                PathOps::path_query(&mut path, a, b),
                want,
                "{shape:?} step {step}"
            );
            assert_eq!(
                PathOps::path_query(&mut sub, a, b),
                want,
                "{shape:?} step {step}"
            );
            let (p, q) = current[rng.below(current.len())];
            let want = SubtreeOps::subtree_query(&mut oracle, p, q);
            assert_eq!(SubtreeOps::subtree_query(&mut sub, p, q), want);
        }
        ett.check_invariants();
    }
}

//! The UFO tree against the oracle, with every structural rule re-verified after every update.

use lct::shapes::{Rng, Shape, generate};
use lct::{Forest, NaiveForest, UfoTree};
use proptest::prelude::*;

#[derive(Clone, Debug)]
enum Op {
    Link(usize, usize),
    Cut(usize, usize),
    CutNth(usize),
    Connected(usize, usize),
}

fn op_strategy() -> impl Strategy<Value = Op> {
    let v = any::<usize>();
    prop_oneof![
        8 => (v, v).prop_map(|(a, b)| Op::Link(a, b)),
        1 => (v, v).prop_map(|(a, b)| Op::Cut(a, b)),
        5 => v.prop_map(Op::CutNth),
        3 => (v, v).prop_map(|(a, b)| Op::Connected(a, b)),
    ]
}

fn run(n: usize, ops: &[Op]) {
    let mut oracle = NaiveForest::new(n);
    let mut ufo = UfoTree::new(n);
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for (step, op) in ops.iter().enumerate() {
        let ctx = format!("step {step}: {op:?}");
        match *op {
            Op::Link(a, b) => {
                let (a, b) = (a % n, b % n);
                let want = oracle.link(a, b);
                assert_eq!(ufo.link(a, b), want, "{ctx}");
                if want {
                    edges.push((a, b));
                }
            }
            Op::Cut(a, b) => {
                let (a, b) = (a % n, b % n);
                let want = oracle.cut(a, b);
                assert_eq!(ufo.cut(a, b), want, "{ctx}");
                edges.retain(|&(x, y)| !((x, y) == (a, b) || (x, y) == (b, a)));
            }
            Op::CutNth(k) => {
                if edges.is_empty() {
                    continue;
                }
                let (a, b) = edges.swap_remove(k % edges.len());
                assert!(oracle.cut(a, b) && ufo.cut(a, b), "{ctx}");
            }
            Op::Connected(a, b) => {
                let (a, b) = (a % n, b % n);
                assert_eq!(ufo.connected(a, b), oracle.connected(a, b), "{ctx}");
            }
        }
        ufo.check_invariants();
    }
    // The vertices' own adjacency is exactly the forest.
    for a in 0..n {
        for b in 0..n {
            assert_eq!(
                ufo.has_edge(a, b),
                edges.iter().any(|&e| e == (a, b) || e == (b, a))
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(600))]

    #[test]
    fn matches_oracle_and_keeps_every_rule(
        n in 2usize..20,
        ops in prop::collection::vec(op_strategy(), 1..250),
    ) {
        run(n, &ops);
    }
}

/// Hubs are where the star rule is exercised: grow and shrink big stars and double stars.
#[test]
fn hubs_grow_and_shrink() {
    let n = 40;
    let mut oracle = NaiveForest::new(n);
    let mut ufo = UfoTree::new(n);
    let mut rng = Rng::new(3);
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for step in 0..6000 {
        // Prefer edges at the two hubs 0 and 1 so their degree swings across the threshold of 3.
        let hub = rng.below(2);
        let other = 2 + rng.below(n - 2);
        if rng.below(100) < 55 {
            let (a, b) = if rng.below(8) == 0 {
                (0, 1)
            } else {
                (hub, other)
            };
            if oracle.link(a, b) {
                assert!(ufo.link(a, b));
                edges.push((a, b));
            }
        } else if !edges.is_empty() {
            let (a, b) = edges.swap_remove(rng.below(edges.len()));
            assert!(oracle.cut(a, b) && ufo.cut(a, b));
        }
        ufo.check_invariants();
        let (x, y) = (rng.below(n), rng.below(n));
        assert_eq!(ufo.connected(x, y), oracle.connected(x, y), "step {step}");
    }
}

#[test]
fn soak_every_shape() {
    for (i, shape) in Shape::ALL.into_iter().enumerate() {
        let n = 120;
        let edges = generate(shape, n, 40 + i as u64);
        let mut oracle = NaiveForest::new(n);
        let mut ufo = UfoTree::new(n);
        let mut rng = Rng::new(i as u64);
        for &(a, b) in &edges {
            assert!(oracle.link(a, b) && ufo.link(a, b));
            ufo.check_invariants();
        }
        let mut current = edges;
        for step in 0..1500 {
            let k = rng.below(current.len());
            let (u, v) = current[k];
            assert!(oracle.cut(u, v) && ufo.cut(u, v));
            ufo.check_invariants();
            let w = loop {
                let w = rng.below(n);
                if !oracle.connected(u, w) {
                    break w;
                }
            };
            assert!(oracle.link(u, w) && ufo.link(u, w));
            ufo.check_invariants();
            current[k] = (u, w);
            let (x, y) = (rng.below(n), rng.below(n));
            assert_eq!(
                ufo.connected(x, y),
                oracle.connected(x, y),
                "{shape:?} step {step}"
            );
        }
    }
}

#[test]
fn a_star_collapses_to_height_two_and_a_path_stays_logarithmic() {
    let n = 5000;
    let mut star = UfoTree::new(n);
    for leaf in 1..n {
        assert!(star.link(0, leaf));
    }
    assert!(star.height() <= 2, "star height {}", star.height());

    let mut path = UfoTree::new(n);
    for v in 1..n {
        assert!(path.link(v - 1, v));
    }
    path.check_invariants();
    let bound = 3 * (n as f64).log2().ceil() as usize;
    assert!(
        path.height() <= bound,
        "path height {} > {bound}",
        path.height()
    );

    // Tearing the star down leaves only isolated vertices and frees every cluster.
    for leaf in 1..n {
        assert!(star.cut(0, leaf));
    }
    assert_eq!(star.height(), 1);
    assert_eq!(star.cluster_count(), n);
}

/// Bigger forests, checked every few steps, with the height watched on the way.
#[test]
fn larger_forests_stay_valid_and_shallow() {
    for (i, shape) in Shape::ALL.into_iter().enumerate() {
        let n = 1000;
        let edges = generate(shape, n, 70 + i as u64);
        let mut oracle = NaiveForest::new(n);
        let mut ufo = UfoTree::new(n);
        let mut rng = Rng::new(500 + i as u64);
        for &(a, b) in &edges {
            assert!(oracle.link(a, b) && ufo.link(a, b));
        }
        ufo.check_invariants();
        let bound = 4 * (n as f64).log2().ceil() as usize;
        assert!(
            ufo.height() <= bound,
            "{shape:?}: height {} > {bound}",
            ufo.height()
        );
        let mut current = edges;
        let mut tallest = ufo.height();
        for step in 0..1200 {
            let k = rng.below(current.len());
            let (u, v) = current[k];
            assert!(oracle.cut(u, v) && ufo.cut(u, v));
            let w = loop {
                let w = rng.below(n);
                if !oracle.connected(u, w) {
                    break w;
                }
            };
            assert!(oracle.link(u, w) && ufo.link(u, w));
            current[k] = (u, w);
            tallest = tallest.max(ufo.height());
            if step % 100 == 0 {
                ufo.check_invariants();
            }
        }
        ufo.check_invariants();
        assert!(
            tallest <= bound,
            "{shape:?}: height reached {tallest} > {bound}"
        );
        for _ in 0..200 {
            let (x, y) = (rng.below(n), rng.below(n));
            assert_eq!(ufo.connected(x, y), oracle.connected(x, y), "{shape:?}");
        }
    }
}

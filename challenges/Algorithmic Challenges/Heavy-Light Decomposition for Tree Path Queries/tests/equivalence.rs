//! Every structure must agree with the parent-walking oracle on every operation, on arbitrary
//! trees, roots and neutral vertices.

use hld::shapes::{Rng, Shape, generate};
use hld::{
    BalancedHld, ClassicHld, Layout, MaybeSubtree, NaiveTree, PathOps, Stats, TopTree, Tree,
};
use proptest::prelude::*;

#[derive(Clone, Debug)]
enum Op {
    Path(usize, usize),
    Add(usize, usize, i64),
    Set(usize, i64),
    Get(usize),
    Sub(usize),
    Lca(usize, usize),
    Kth(usize, usize),
}

fn op_strategy() -> impl Strategy<Value = Op> {
    let any_vertex = any::<usize>();
    prop_oneof![
        3 => (any_vertex, any_vertex).prop_map(|(a, b)| Op::Path(a, b)),
        3 => (any_vertex, any_vertex, -50i64..50).prop_map(|(a, b, d)| Op::Add(a, b, d)),
        2 => (any_vertex, -1000i64..1000).prop_map(|(a, w)| Op::Set(a, w)),
        1 => any_vertex.prop_map(Op::Get),
        2 => any_vertex.prop_map(Op::Sub),
        1 => (any_vertex, any_vertex).prop_map(|(a, b)| Op::Lca(a, b)),
        1 => (any_vertex, 0usize..40).prop_map(|(a, k)| Op::Kth(a, k)),
    ]
}

fn assert_same_as_oracle<S: MaybeSubtree>(layout: &Layout, values: &[Option<i64>], ops: &[Op]) {
    let n = layout.len();
    let mut oracle = NaiveTree::build(layout.clone(), values);
    let mut subject = S::build(layout.clone(), values);
    for (step, op) in ops.iter().enumerate() {
        match *op {
            Op::Path(a, b) => {
                let (a, b) = (a % n, b % n);
                assert_eq!(
                    subject.path_query(a, b),
                    oracle.path_query(a, b),
                    "step {step}: {op:?}"
                );
            }
            Op::Add(a, b, d) => {
                let (a, b) = (a % n, b % n);
                subject.path_add(a, b, d);
                oracle.path_add(a, b, d);
            }
            Op::Set(a, w) => {
                subject.set(a % n, w);
                oracle.set(a % n, w);
            }
            Op::Get(a) => assert_eq!(subject.get(a % n), oracle.get(a % n), "step {step}: {op:?}"),
            Op::Sub(a) => {
                if let Some(got) = subject.maybe_subtree(a % n) {
                    assert_eq!(
                        got,
                        oracle.maybe_subtree(a % n).unwrap(),
                        "step {step}: {op:?}"
                    );
                }
            }
            Op::Lca(a, b) => {
                assert_eq!(
                    subject.lca(a % n, b % n),
                    oracle.lca(a % n, b % n),
                    "step {step}: {op:?}"
                );
            }
            Op::Kth(a, k) => assert_eq!(
                subject.kth_ancestor(a % n, k),
                oracle.kth_ancestor(a % n, k),
                "step {step}: {op:?}"
            ),
        }
    }
    for v in 0..n {
        assert_eq!(subject.get(v), oracle.get(v), "final value of {v}");
        if let Some(got) = subject.maybe_subtree(v) {
            assert_eq!(
                got,
                oracle.maybe_subtree(v).unwrap(),
                "final subtree of {v}"
            );
        }
    }
}

fn check_all(layout: &Layout, values: &[Option<i64>], ops: &[Op]) {
    assert_same_as_oracle::<ClassicHld>(layout, values, ops);
    assert_same_as_oracle::<BalancedHld>(layout, values, ops);
    assert_same_as_oracle::<TopTree>(layout, values, ops);
}

fn layout_from_parents(parents: &[u32], root_pick: usize) -> Layout {
    let n = parents.len() + 1;
    let edges: Vec<(usize, usize)> = parents
        .iter()
        .enumerate()
        .map(|(i, &p)| (p as usize % (i + 1), i + 1))
        .collect();
    Layout::new(&Tree::new(n, &edges).unwrap(), root_pick % n).unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn arbitrary_trees_roots_and_neutral_vertices(
        parents in prop::collection::vec(any::<u32>(), 0..70),
        root in any::<usize>(),
        values in prop::collection::vec(prop::option::weighted(0.8, -1000i64..1000), 71),
        ops in prop::collection::vec(op_strategy(), 0..120),
    ) {
        let layout = layout_from_parents(&parents, root);
        let values = &values[..layout.len()];
        check_all(&layout, values, &ops);
    }

    #[test]
    fn deep_narrow_trees(
        parents in prop::collection::vec(0u32..3, 1..90),
        ops in prop::collection::vec(op_strategy(), 0..120),
    ) {
        // parent = one of the last three vertices, so chains are long and light edges are few.
        let edges: Vec<(usize, usize)> = parents
            .iter()
            .enumerate()
            .map(|(i, &back)| ((i + 1).saturating_sub(1 + back as usize), i + 1))
            .collect();
        let n = parents.len() + 1;
        let layout = Layout::new(&Tree::new(n, &edges).unwrap(), 0).unwrap();
        let values: Vec<Option<i64>> = (0..n as i64).map(|i| Some(i * 7 - 40)).collect();
        check_all(&layout, &values, &ops);
    }
}

fn random_ops(n: usize, count: usize, seed: u64) -> Vec<Op> {
    let mut rng = Rng::new(seed);
    (0..count)
        .map(|_| {
            let (a, b) = (rng.below(n), rng.below(n));
            match rng.below(9) {
                0..=2 => Op::Path(a, b),
                3..=4 => Op::Add(a, b, rng.below(41) as i64 - 20),
                5 => Op::Set(a, rng.below(2001) as i64 - 1000),
                6 => Op::Sub(a),
                7 => Op::Lca(a, b),
                _ => Op::Kth(a, rng.below(30)),
            }
        })
        .collect()
}

#[test]
fn every_shape_at_a_few_thousand_vertices() {
    for shape in Shape::ALL {
        for (n, seed) in [(1, 1), (2, 2), (3, 3), (7, 4), (300, 5), (2500, 6)] {
            let g = generate(shape, n, seed);
            let layout = Layout::new(&Tree::new(g.n, &g.edges).unwrap(), g.root).unwrap();
            let mut rng = Rng::new(seed);
            let values: Vec<Option<i64>> = (0..n)
                .map(|_| Some(rng.below(2001) as i64 - 1000))
                .collect();
            check_all(&layout, &values, &random_ops(n, 1500, seed ^ shape as u64));
        }
    }
}

fn neutral_vertex_case<S: PathOps>() {
    let layout = Layout::new(&Tree::new(3, &[(0, 1), (1, 2)]).unwrap(), 0).unwrap();
    let mut s = S::build(layout, &[Some(5), None, Some(7)]);
    s.path_add(0, 2, 10);
    assert_eq!(s.get(1), None);
    assert_eq!(
        s.path_query(0, 2),
        Stats {
            count: 2,
            sum: 32,
            min: 15,
            max: 17
        }
    );
    s.set(1, 1);
    assert_eq!(
        s.path_query(0, 2),
        Stats {
            count: 3,
            sum: 33,
            min: 1,
            max: 17
        }
    );
}

#[test]
fn a_neutral_vertex_ignores_adds_but_becomes_real_when_set() {
    neutral_vertex_case::<NaiveTree>();
    neutral_vertex_case::<ClassicHld>();
    neutral_vertex_case::<BalancedHld>();
    neutral_vertex_case::<TopTree>();
}

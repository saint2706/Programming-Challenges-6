//! Layout invariants, input validation, depth bounds, stack safety and the edge-weighted wrapper.

use hld::shapes::{Rng, Shape, generate};
use hld::{
    BalancedHld, ClassicHld, EdgeTree, Layout, MaybeSubtree, PathOps, Stats, SubtreeOps, TopTree,
    Tree, TreeError,
};

fn layout_of(shape: Shape, n: usize, seed: u64) -> Layout {
    let g = generate(shape, n, seed);
    Layout::new(&Tree::new(g.n, &g.edges).unwrap(), g.root).unwrap()
}

#[test]
fn tree_validation_rejects_every_malformed_input() {
    assert_eq!(Tree::new(0, &[]).unwrap_err(), TreeError::Empty);
    assert_eq!(
        Tree::new(3, &[(0, 1)]).unwrap_err(),
        TreeError::WrongEdgeCount {
            vertices: 3,
            edges: 1
        }
    );
    assert_eq!(
        Tree::new(2, &[(0, 2)]).unwrap_err(),
        TreeError::VertexOutOfRange(2)
    );
    assert_eq!(Tree::new(2, &[(1, 1)]).unwrap_err(), TreeError::SelfLoop(1));
    // 4 vertices, 3 edges, but a triangle plus an isolated vertex.
    assert_eq!(
        Tree::new(4, &[(0, 1), (1, 2), (2, 0)]).unwrap_err(),
        TreeError::NotConnected
    );
    assert!(Tree::new(1, &[]).is_ok());
    let tree = Tree::new(2, &[(0, 1)]).unwrap();
    assert_eq!(
        Layout::new(&tree, 2).unwrap_err(),
        TreeError::RootOutOfRange(2)
    );
    assert!(TreeError::NotConnected.to_string().contains("cycle"));
}

/// Recompute everything from the parent array with plain loops and compare to the layout.
// The vertex index addresses a dozen parallel arrays here; an iterator would only obscure that.
#[allow(clippy::needless_range_loop)]
#[test]
fn layout_invariants_match_brute_force() {
    for shape in Shape::ALL {
        for (n, seed) in [(1, 1), (2, 2), (9, 3), (200, 4), (1500, 5)] {
            let l = layout_of(shape, n, seed);
            let children: Vec<Vec<usize>> = {
                let mut c = vec![vec![]; n];
                for v in 0..n {
                    if l.parent[v] != u32::MAX {
                        c[l.parent[v] as usize].push(v);
                    }
                }
                c
            };
            let depth_of = |mut v: usize| {
                let mut d = 0;
                while l.parent[v] != u32::MAX {
                    v = l.parent[v] as usize;
                    d += 1;
                }
                d
            };
            let is_descendant = |mut v: usize, anc: usize| loop {
                if v == anc {
                    break true;
                }
                if l.parent[v] == u32::MAX {
                    break false;
                }
                v = l.parent[v] as usize;
            };
            for v in 0..n {
                assert_eq!(l.depth[v] as usize, depth_of(v), "depth of {v}");
                let descendants: Vec<usize> = (0..n).filter(|&w| is_descendant(w, v)).collect();
                assert_eq!(l.size[v] as usize, descendants.len(), "size of {v}");
                // The subtree is exactly the contiguous preorder range.
                let range = &l.order[l.pos[v] as usize..(l.pos[v] + l.size[v]) as usize];
                let mut got: Vec<usize> = range.iter().map(|&w| w as usize).collect();
                got.sort_unstable();
                assert_eq!(got, descendants, "subtree range of {v}");
                // The heavy child has the (joint) largest subtree.
                match children[v].iter().map(|&c| l.size[c]).max() {
                    None => assert_eq!(l.heavy[v], u32::MAX),
                    Some(m) => assert_eq!(l.size[l.heavy[v] as usize], m),
                }
                // Chain bookkeeping: the head is an ancestor on the same chain, at chain_pos below it.
                let h = l.head[v] as usize;
                assert_eq!(l.chain[v], l.chain[h]);
                assert_eq!(l.chain_pos(v) as usize, (l.depth[v] - l.depth[h]) as usize);
                assert_eq!(
                    l.order[l.pos[h] as usize + l.chain_pos(v) as usize] as usize,
                    v
                );
                assert!(l.light_edges_to_root(v) as f64 <= (n as f64).log2() + 1e-9);
            }
            let covered: u32 = l.chain_len.iter().sum();
            assert_eq!(covered as usize, n, "chains partition the vertices");
            for (v, kids) in children.iter().enumerate() {
                assert_eq!(l.light_children(v).len(), kids.len().saturating_sub(1));
            }
        }
    }
}

#[test]
fn lca_and_kth_ancestor_match_climbing() {
    for shape in Shape::ALL {
        let l = layout_of(shape, 400, 11);
        let mut rng = Rng::new(3);
        for _ in 0..800 {
            let (u, v) = (rng.below(400), rng.below(400));
            let (mut a, mut b) = (u, v);
            while l.depth[a] > l.depth[b] {
                a = l.parent[a] as usize;
            }
            while l.depth[b] > l.depth[a] {
                b = l.parent[b] as usize;
            }
            while a != b {
                a = l.parent[a] as usize;
                b = l.parent[b] as usize;
            }
            assert_eq!(l.lca(u, v), a);
            let k = rng.below(60);
            let mut x = Some(u);
            for _ in 0..k {
                x = x.and_then(|y| (l.parent[y] != u32::MAX).then(|| l.parent[y] as usize));
            }
            assert_eq!(l.kth_ancestor(u, k), x);
        }
    }
}

/// The global trees promise O(log n): measured, the deepest vertex is at most about log2(n) + 2.
#[test]
fn global_trees_have_logarithmic_depth_on_every_shape() {
    for shape in Shape::ALL {
        for n in [2usize, 3, 10, 1_000, 50_000, 200_000] {
            let layout = layout_of(shape, n, 7);
            let zeros = vec![0i64; n];
            let bound = (n as f64).log2().ceil() as usize + 4;
            let balanced = BalancedHld::from_weights(layout.clone(), &zeros).max_depth();
            let top = TopTree::from_weights(layout, &zeros).max_depth();
            assert!(
                balanced <= bound,
                "{} n={n}: balanced depth {balanced} > {bound}",
                shape.name()
            );
            assert!(
                top <= bound,
                "{} n={n}: top tree depth {top} > {bound}",
                shape.name()
            );
        }
    }
}

#[test]
fn million_vertex_path_does_not_overflow_the_stack() {
    let n = 1_000_000;
    let layout = layout_of(Shape::Path, n, 1);
    let values: Vec<i64> = (0..n as i64).collect();
    let mut tree = TopTree::from_weights(layout.clone(), &values);
    let g = generate(Shape::Path, n, 1);
    let total = tree.path_query(g.far.0, g.far.1);
    assert_eq!(total.count, n as i64);
    assert_eq!(total.sum, (n as i64 - 1) * n as i64 / 2);
    tree.path_add(g.far.0, g.far.1, 1);
    assert_eq!(tree.subtree_query(g.root).sum, total.sum + n as i64);
    let mut classic = ClassicHld::from_weights(layout, &values);
    assert_eq!(classic.path_query(g.far.0, g.far.1), total);
}

#[test]
fn single_vertex_tree_and_repeated_endpoints() {
    let layout = layout_of(Shape::Random, 1, 1);
    for mut s in [
        Box::new(ClassicHld::from_weights(layout.clone(), &[5])) as Box<dyn Probe>,
        Box::new(BalancedHld::from_weights(layout.clone(), &[5])),
        Box::new(TopTree::from_weights(layout.clone(), &[5])),
    ] {
        assert_eq!(s.query(0, 0).sum, 5);
        s.add(0, 0, 3);
        assert_eq!(s.query(0, 0), Stats::single(8));
        assert_eq!(s.ancestor(0, 0), Some(0));
        assert_eq!(s.ancestor(0, 1), None);
    }
}

trait Probe {
    fn query(&mut self, u: usize, v: usize) -> Stats;
    fn add(&mut self, u: usize, v: usize, d: i64);
    fn ancestor(&self, v: usize, k: usize) -> Option<usize>;
}

impl<S: PathOps> Probe for S {
    fn query(&mut self, u: usize, v: usize) -> Stats {
        self.path_query(u, v)
    }
    fn add(&mut self, u: usize, v: usize, d: i64) {
        self.path_add(u, v, d);
    }
    fn ancestor(&self, v: usize, k: usize) -> Option<usize> {
        self.kth_ancestor(v, k)
    }
}

// ---- edge-weighted trees against an independent brute force ---------------------------------

struct EdgeBrute {
    parent: Vec<Option<(usize, usize)>>, // (parent vertex, edge index)
    depth: Vec<usize>,
    children: Vec<Vec<(usize, usize)>>,
    weight: Vec<i64>,
}

impl EdgeBrute {
    fn new(n: usize, edges: &[(usize, usize, i64)], root: usize) -> Self {
        let mut adj = vec![vec![]; n];
        for (e, &(a, b, _)) in edges.iter().enumerate() {
            adj[a].push((b, e));
            adj[b].push((a, e));
        }
        let mut parent = vec![None; n];
        let mut depth = vec![0; n];
        let mut children = vec![vec![]; n];
        let mut seen = vec![false; n];
        seen[root] = true;
        let mut stack = vec![root];
        while let Some(v) = stack.pop() {
            for &(w, e) in &adj[v] {
                if !seen[w] {
                    seen[w] = true;
                    parent[w] = Some((v, e));
                    depth[w] = depth[v] + 1;
                    children[v].push((w, e));
                    stack.push(w);
                }
            }
        }
        EdgeBrute {
            parent,
            depth,
            children,
            weight: edges.iter().map(|e| e.2).collect(),
        }
    }

    fn path(&self, mut u: usize, mut v: usize) -> Vec<usize> {
        let mut edges = vec![];
        while u != v {
            let climb = if self.depth[u] >= self.depth[v] {
                &mut u
            } else {
                &mut v
            };
            let (p, e) = self.parent[*climb].unwrap();
            edges.push(e);
            *climb = p;
        }
        edges
    }

    fn below(&self, v: usize) -> Vec<usize> {
        let mut out = vec![];
        let mut stack = vec![v];
        while let Some(x) = stack.pop() {
            for &(w, e) in &self.children[x] {
                out.push(e);
                stack.push(w);
            }
        }
        out
    }

    fn stats(&self, edges: &[usize]) -> Stats {
        edges.iter().fold(Stats::NEUTRAL, |acc, &e| {
            acc.combine(Stats::single(self.weight[e]))
        })
    }
}

fn edge_tree_matches_brute_force<S: MaybeSubtree>() {
    for shape in Shape::ALL {
        for (n, seed) in [(1, 1), (2, 2), (6, 3), (120, 4), (900, 5)] {
            let g = generate(shape, n, seed);
            let mut rng = Rng::new(seed + 99);
            let edges: Vec<(usize, usize, i64)> = g
                .edges
                .iter()
                .map(|&(a, b)| (a, b, rng.below(201) as i64 - 100))
                .collect();
            let mut oracle = EdgeBrute::new(n, &edges, g.root);
            let mut tree = EdgeTree::<S>::new(n, &edges, g.root).unwrap();
            for step in 0..600 {
                let (u, v) = (rng.below(n), rng.below(n));
                match rng.below(5) {
                    0 | 1 => assert_eq!(
                        tree.path_query(u, v),
                        oracle.stats(&oracle.path(u, v)),
                        "{} n={n} step {step}",
                        shape.name()
                    ),
                    2 => {
                        let d = rng.below(21) as i64 - 10;
                        tree.path_add(u, v, d);
                        for e in oracle.path(u, v) {
                            oracle.weight[e] += d;
                        }
                    }
                    3 if n > 1 => {
                        let e = rng.below(n - 1);
                        let w = rng.below(401) as i64 - 200;
                        tree.set_edge(e, w);
                        oracle.weight[e] = w;
                        assert_eq!(tree.edge_weight(e), Some(w));
                    }
                    _ => {
                        if let Some(got) = tree.maybe_subtree(u) {
                            assert_eq!(got, oracle.stats(&oracle.below(u)), "subtree of {u}");
                        }
                    }
                }
            }
            if n > 1 {
                assert_eq!(tree.path_query(g.root, g.root), Stats::NEUTRAL);
                assert_eq!(tree.len(), n);
            }
            for _ in 0..50 {
                let (v, k) = (rng.below(n), rng.below(8));
                let mut x = Some(v);
                for _ in 0..k {
                    x = x.and_then(|y| oracle.parent[y].map(|p| p.0));
                }
                assert_eq!(tree.kth_ancestor(v, k), x);
            }
        }
    }
}

#[test]
fn edge_weighted_variant_agrees_with_brute_force_on_every_structure() {
    edge_tree_matches_brute_force::<hld::NaiveTree>();
    edge_tree_matches_brute_force::<ClassicHld>();
    edge_tree_matches_brute_force::<BalancedHld>();
    edge_tree_matches_brute_force::<TopTree>();
}

#[test]
fn edge_tree_reports_malformed_input_in_the_callers_numbering() {
    assert_eq!(
        EdgeTree::<ClassicHld>::new(3, &[(0, 1, 5)], 0).err(),
        Some(TreeError::WrongEdgeCount {
            vertices: 3,
            edges: 1
        })
    );
    assert_eq!(
        EdgeTree::<ClassicHld>::new(2, &[(0, 9, 5)], 0).err(),
        Some(TreeError::VertexOutOfRange(9))
    );
    assert_eq!(
        EdgeTree::<ClassicHld>::new(2, &[(0, 1, 5)], 2).err(),
        Some(TreeError::RootOutOfRange(2))
    );
}

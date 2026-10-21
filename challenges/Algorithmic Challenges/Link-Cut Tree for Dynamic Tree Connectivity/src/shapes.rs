//! Deterministic tree shapes and operation streams for tests, benchmarks and the CLI generator.

use crate::{Forest, Weighted};

/// SplitMix64: tiny, fast, and good enough to shuffle labels and draw operations.
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (modulo bias is irrelevant at these sizes).
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform in `lo..hi`.
    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below((hi - lo) as usize) as i64
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// Random recursive tree: depth O(log n).
    Random,
    /// One long path: diameter n - 1, the worst case for anything that walks the tree.
    Path,
    /// One hub of degree n - 1: the worst case for anything that stores neighbours in order.
    Star,
    /// Complete binary tree: diameter 2 log n.
    Binary,
    Caterpillar,
    /// A long handle ending in a star.
    Broom,
    /// Random parent among the last few vertices: deep, with few branches.
    Deep,
}

impl Shape {
    pub const ALL: [Shape; 7] = [
        Shape::Random,
        Shape::Path,
        Shape::Star,
        Shape::Binary,
        Shape::Caterpillar,
        Shape::Broom,
        Shape::Deep,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Shape::Random => "random",
            Shape::Path => "path",
            Shape::Star => "star",
            Shape::Binary => "binary",
            Shape::Caterpillar => "caterpillar",
            Shape::Broom => "broom",
            Shape::Deep => "deep",
        }
    }

    pub fn from_name(name: &str) -> Option<Shape> {
        Shape::ALL.into_iter().find(|s| s.name() == name)
    }
}

/// The edges of a tree on exactly `n` vertices with shuffled labels.
pub fn generate(shape: Shape, n: usize, seed: u64) -> Vec<(usize, usize)> {
    assert!(n >= 1);
    let mut rng = Rng::new(seed);
    let mut parent = vec![0usize; n];
    match shape {
        Shape::Random => (1..n).for_each(|i| parent[i] = rng.below(i)),
        Shape::Path => (1..n).for_each(|i| parent[i] = i - 1),
        Shape::Star => {}
        Shape::Binary => (1..n).for_each(|i| parent[i] = (i - 1) / 2),
        Shape::Caterpillar => {
            let spine = n.div_ceil(2);
            for (i, p) in parent.iter_mut().enumerate().skip(1) {
                *p = if i < spine { i - 1 } else { i - spine };
            }
        }
        Shape::Broom => {
            let handle = n.div_ceil(2);
            for (i, p) in parent.iter_mut().enumerate().skip(1) {
                *p = if i < handle { i - 1 } else { handle - 1 };
            }
        }
        Shape::Deep => (1..n).for_each(|i| parent[i] = i.saturating_sub(1 + rng.below(4))),
    }
    let mut label: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        label.swap(i, rng.below(i + 1));
    }
    (1..n).map(|i| (label[parent[i]], label[i])).collect()
}

/// One operation of a replayable stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Link(usize, usize),
    Cut(usize, usize),
    Connected(usize, usize),
}

/// A stream of valid link/cut/connected operations on a forest that starts as `edges`.
///
/// Every cut removes a current edge `(u, v)` and the following link joins `u` to a random vertex
/// that is no longer in `u`'s tree, so the forest keeps changing shape instead of returning to
/// where it started. `query_percent` of the operations are connectivity queries, drawn from
/// `hot` vertices only when `hot > 0` (a skewed workload that a self-adjusting tree can exploit).
/// The stream is generated against `F`, so it is valid for every correct structure.
pub fn churn_stream<F: Forest>(
    n: usize,
    edges: &[(usize, usize)],
    ops: usize,
    query_percent: usize,
    hot: usize,
    seed: u64,
) -> Vec<Op> {
    let mut rng = Rng::new(seed);
    let mut forest = F::new(n);
    for &(u, v) in edges {
        assert!(forest.link(u, v));
    }
    let mut current: Vec<(usize, usize)> = edges.to_vec();
    let pick = |rng: &mut Rng| {
        if hot > 0 {
            rng.below(hot.min(n))
        } else {
            rng.below(n)
        }
    };
    let mut out = Vec::with_capacity(ops);
    while out.len() < ops {
        if current.is_empty() || rng.below(100) < query_percent {
            let (u, v) = (pick(&mut rng), pick(&mut rng));
            out.push(Op::Connected(u, v));
            continue;
        }
        let i = rng.below(current.len());
        let (u, v) = current[i];
        out.push(Op::Cut(u, v));
        assert!(forest.cut(u, v));
        // Reattach `u` to a vertex of the other side; fall back to re-linking the same edge.
        let mut target = v;
        for _ in 0..8 {
            let x = rng.below(n);
            if !forest.connected(u, x) {
                target = x;
                break;
            }
        }
        out.push(Op::Link(u, target));
        assert!(forest.link(u, target));
        current[i] = (u, target);
    }
    out
}

/// [`churn_stream`] followed by its exact inverse, so replaying it leaves the forest where it
/// started and a benchmark can loop it forever. `ops` counts the forward half.
pub fn cyclic_stream<F: Forest>(
    n: usize,
    edges: &[(usize, usize)],
    ops: usize,
    query_percent: usize,
    hot: usize,
    seed: u64,
) -> Vec<Op> {
    let mut stream = churn_stream::<F>(n, edges, ops, query_percent, hot, seed);
    let undo: Vec<Op> = stream
        .iter()
        .rev()
        .map(|&op| match op {
            Op::Link(u, v) => Op::Cut(u, v),
            Op::Cut(u, v) => Op::Link(u, v),
            q @ Op::Connected(..) => q,
        })
        .collect();
    stream.extend(undo);
    stream
}

/// Replay a stream on `forest`, returning how many connectivity queries answered `true`.
pub fn replay<F: Forest>(forest: &mut F, stream: &[Op]) -> usize {
    let mut yes = 0;
    for &op in stream {
        match op {
            Op::Link(u, v) => {
                forest.link(u, v);
            }
            Op::Cut(u, v) => {
                forest.cut(u, v);
            }
            Op::Connected(u, v) => yes += usize::from(forest.connected(u, v)),
        }
    }
    yes
}

/// Random vertex values in `-1000..1000` for a weighted structure.
pub fn random_values(n: usize, seed: u64) -> Vec<Option<i64>> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| Some(rng.range(-1000, 1000))).collect()
}

/// Build a weighted forest holding `edges`.
pub fn build<F: Weighted>(n: usize, edges: &[(usize, usize)], seed: u64) -> F {
    let mut f = F::from_values(&random_values(n, seed));
    for &(u, v) in edges {
        assert!(f.link(u, v));
    }
    f
}

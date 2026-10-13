//! Deterministic tree shapes for tests, benchmarks and the CLI generator.

/// SplitMix64: tiny, fast, and good enough to shuffle labels and draw queries.
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// Random recursive tree: depth O(log n), many light edges.
    Random,
    Path,
    Star,
    /// Complete binary tree: every root path has ~log n light edges, all chains short.
    Binary,
    Caterpillar,
    Broom,
    /// Random parent among the last few vertices: deep, with few light edges.
    Deep,
    /// The classic worst case for chain-by-chain HLD: a root path crosses ~log n light edges and
    /// each chain it enters is long, so a segment tree per chain costs ~log² n.
    Adversarial,
}

impl Shape {
    pub const ALL: [Shape; 8] = [
        Shape::Random,
        Shape::Path,
        Shape::Star,
        Shape::Binary,
        Shape::Caterpillar,
        Shape::Broom,
        Shape::Deep,
        Shape::Adversarial,
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
            Shape::Adversarial => "adversarial",
        }
    }

    pub fn from_name(name: &str) -> Option<Shape> {
        Shape::ALL.into_iter().find(|s| s.name() == name)
    }
}

pub struct Generated {
    pub n: usize,
    pub edges: Vec<(usize, usize)>,
    /// Where vertex 0 of the unshuffled shape ended up; a natural root.
    pub root: usize,
    /// A far-apart pair: the deepest vertex of the construction and the root.
    pub far: (usize, usize),
}

/// A tree of exactly `n` vertices with shuffled labels.
pub fn generate(shape: Shape, n: usize, seed: u64) -> Generated {
    assert!(n >= 1);
    let mut rng = Rng::new(seed);
    let mut parent = vec![0usize; n];
    let mut deepest = n - 1;
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
            deepest = spine - 1;
        }
        Shape::Broom => {
            let handle = n.div_ceil(2);
            for (i, p) in parent.iter_mut().enumerate().skip(1) {
                *p = if i < handle { i - 1 } else { handle - 1 };
            }
        }
        Shape::Deep => (1..n).for_each(|i| parent[i] = i.saturating_sub(1 + rng.below(4))),
        Shape::Adversarial => {
            // T(s) = root + a spine of `a` vertices + a light copy T(b), a >= b so the spine is heavy.
            let (mut next, mut root, mut s) = (1, 0, n);
            while s >= 3 {
                let b = (s - 1) / 2;
                let a = s - 1 - b;
                let mut prev = root;
                for _ in 0..a {
                    parent[next] = prev;
                    prev = next;
                    next += 1;
                }
                parent[next] = root;
                root = next;
                next += 1;
                s = b;
            }
            if s == 2 {
                parent[next] = root;
                next += 1;
            }
            debug_assert_eq!(next, n);
            deepest = root;
        }
    }
    let mut label: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        label.swap(i, rng.below(i + 1));
    }
    let edges = (1..n).map(|i| (label[parent[i]], label[i])).collect();
    Generated {
        n,
        edges,
        root: label[0],
        far: (label[deepest], label[0]),
    }
}

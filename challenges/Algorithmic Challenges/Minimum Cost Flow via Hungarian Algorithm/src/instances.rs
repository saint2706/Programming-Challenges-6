//! Instance families for tests and benchmarks.
//!
//! The generator is SplitMix64 written out here rather than the `rand` crate: a seed must name the
//! same instance in every build, on every machine, in every future crate version, because the
//! README's numbers are quoted against those seeds. Every generator is deterministic in its
//! arguments.

use crate::error::Result;
use crate::flow::{MAX_CAP, Network};
use crate::matrix::{Cost, Dense, FORBIDDEN, MAX_ABS_COST, Sparse};

/// SplitMix64 (Steele, Lea, Flood 2014): 64 bits of state, passes BigCrush, one multiply-shift
/// finaliser per output.
#[derive(Clone, Debug)]
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

    /// Uniform in `0..n` (Lemire's multiply-shift; the bias is below 2^-64 * n, irrelevant here).
    pub fn below(&mut self, n: u64) -> u64 {
        ((self.next_u64() as u128 * n as u128) >> 64) as u64
    }

    /// Uniform in `lo..=hi`.
    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below((hi - lo) as u64 + 1) as i64
    }

    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i as u64 + 1) as usize);
        }
    }

    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        self.shuffle(&mut p);
        p
    }
}

/// Independent uniform costs in `0..=max_cost`. A small `max_cost` makes many ties.
pub fn uniform(rows: usize, cols: usize, max_cost: Cost, seed: u64) -> Dense {
    let mut rng = Rng::new(seed);
    Dense::from_fn(rows, cols, |_, _| rng.range(0, max_cost)).expect("costs in range")
}

/// Machol-Wien's hard instance for the Hungarian method: `c_ij = i * j` (0-based). By the
/// rearrangement inequality the optimum pairs `i` with `n - 1 - i`, so the answer is known in
/// closed form, `sum(i * (n - 1 - i))`, at any size (see [`machol_wien_optimum`]).
pub fn machol_wien(n: usize) -> Dense {
    assert!(
        (n as u64) * (n as u64) <= MAX_ABS_COST as u64,
        "n too large for 2^32 costs"
    );
    Dense::from_fn(n, n, |i, j| (i * j) as Cost).expect("costs in range")
}

pub fn machol_wien_optimum(n: usize) -> Cost {
    (0..n).map(|i| (i * (n - 1 - i)) as Cost).sum()
}

/// `n` workers and `n` tasks at random points of a `grid x grid` square; the cost is the rounded
/// Euclidean distance times 16. Costs are metric, so many near-ties exist.
pub fn geometric(n: usize, grid: i64, seed: u64) -> Dense {
    let mut rng = Rng::new(seed);
    let mut points = |k: usize| -> Vec<(i64, i64)> {
        (0..k)
            .map(|_| (rng.range(0, grid), rng.range(0, grid)))
            .collect()
    };
    let (w, t) = (points(n), points(n));
    Dense::from_fn(n, n, |i, j| {
        let (dx, dy) = ((w[i].0 - t[j].0) as f64, (w[i].1 - t[j].1) as f64);
        (dx.hypot(dy) * 16.0).round() as Cost
    })
    .expect("costs in range")
}

/// `c_ij = a_i + b_j + noise`: every assignment costs `sum(a) + sum(b)` plus noise, so the optimum
/// hides in a sea of near-ties, which is a stress test for the tie-breaking in every solver.
pub fn separable(n: usize, spread: Cost, noise: Cost, seed: u64) -> Dense {
    let mut rng = Rng::new(seed);
    let a: Vec<Cost> = (0..n).map(|_| rng.range(0, spread)).collect();
    let b: Vec<Cost> = (0..n).map(|_| rng.range(0, spread)).collect();
    Dense::from_fn(n, n, |i, j| a[i] + b[j] + rng.range(0, noise)).expect("costs in range")
}

/// Forbids each entry independently with probability `percent / 100`, except along a planted
/// random perfect matching so the result stays feasible (requires `rows <= cols`).
pub fn forbid(d: &Dense, percent: u64, seed: u64) -> Dense {
    assert!(d.rows() <= d.cols());
    let mut rng = Rng::new(seed);
    let planted: Vec<usize> = rng
        .permutation(d.cols())
        .into_iter()
        .take(d.rows())
        .collect();
    Dense::from_fn(d.rows(), d.cols(), |i, j| {
        if planted[i] == j || rng.below(100) >= percent {
            d.get(i, j)
        } else {
            FORBIDDEN
        }
    })
    .expect("costs in range")
}

/// Sparse feasible instance: a planted random perfect matching plus about `degree - 1` random
/// extra edges per row, costs uniform in `0..=max_cost`. With `degree = 2 * log2(n)` and
/// `max_cost = 100` this is the shape of the DIMACS "low-cost" assignment family.
pub fn planted_sparse(n: usize, degree: usize, max_cost: Cost, seed: u64) -> Sparse {
    let mut rng = Rng::new(seed);
    let planted = rng.permutation(n);
    let mut triplets = Vec::with_capacity(n * degree);
    for (i, &p) in planted.iter().enumerate() {
        triplets.push((i, p, rng.range(0, max_cost)));
        for _ in 1..degree {
            triplets.push((i, rng.below(n as u64) as usize, rng.range(0, max_cost)));
        }
    }
    Sparse::from_triplets(n, n, triplets).expect("valid triplets")
}

/// The DIMACS-style sparse family for size `n`: degree `2 * ceil(log2 n)`, costs `0..=100`.
pub fn dimacs_sparse(n: usize, seed: u64) -> Sparse {
    let degree = 2 * (usize::BITS - n.max(2).saturating_sub(1).leading_zeros()) as usize;
    planted_sparse(n, degree, 100, seed)
}

/// A random feasible min-cost-flow network: a ring `i -> i + 1` with capacity equal to the total
/// supply (so any supply can be routed), `arcs - nodes` random arcs with capacities in
/// `1..=max_cap`, costs in `-max_cost/4..=max_cost` when `negative` (else `0..=max_cost`), and
/// supplies split among about `sqrt(nodes)` sources and as many sinks.
pub fn random_network(
    nodes: usize,
    arcs: usize,
    max_cap: i64,
    max_cost: Cost,
    negative: bool,
    seed: u64,
) -> Result<Network> {
    assert!(nodes >= 3 && arcs >= nodes);
    let mut rng = Rng::new(seed);
    let mut net = Network::new(nodes);
    let k = ((nodes as f64).sqrt() as usize).clamp(1, nodes / 2);
    let order = rng.permutation(nodes);
    let (sources, sinks) = (&order[..k], &order[k..2 * k]);
    let mut total = 0i64;
    for (&s, &t) in sources.iter().zip(sinks) {
        let amount = rng.range(1, max_cap.max(1));
        net.set_supply(s, amount)?;
        net.set_supply(t, -amount)?;
        total += amount;
    }
    let cost_lo = if negative { -max_cost / 4 } else { 0 };
    let ring_cap = total.min(MAX_CAP);
    for v in 0..nodes {
        net.add_arc(v, (v + 1) % nodes, ring_cap, rng.range(cost_lo, max_cost))?;
    }
    for _ in nodes..arcs {
        let from = rng.below(nodes as u64) as usize;
        let mut to = rng.below(nodes as u64 - 1) as usize;
        if to >= from {
            to += 1;
        }
        net.add_arc(
            from,
            to,
            rng.range(1, max_cap),
            rng.range(cost_lo, max_cost),
        )?;
    }
    Ok(net)
}

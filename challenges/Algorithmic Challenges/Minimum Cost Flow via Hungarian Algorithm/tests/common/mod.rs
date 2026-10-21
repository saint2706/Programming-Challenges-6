//! Shared test helpers: a brute-force oracle and small random instances.

#![allow(dead_code)]

use mcflow::instances::Rng;
use mcflow::{Cost, Dense, FORBIDDEN};

/// The cheapest assignment of size `min(rows, cols)` by exhaustive search over column subsets
/// (a bitmask DP, exact), or `None` when forbidden edges make that size impossible.
pub fn brute_force(d: &Dense) -> Option<Cost> {
    if d.rows() > d.cols() {
        return brute_force(&d.transposed());
    }
    let (n, m) = (d.rows(), d.cols());
    assert!(m <= 16, "oracle is exponential in the columns");
    let mut best = vec![None::<Cost>; 1 << m];
    best[0] = Some(0);
    for i in 0..n {
        let mut next = vec![None::<Cost>; 1 << m];
        for (mask, &cost) in best.iter().enumerate() {
            let Some(cost) = cost else { continue };
            if mask.count_ones() as usize != i {
                continue;
            }
            for j in 0..m {
                let c = d.get(i, j);
                if mask >> j & 1 == 1 || c == FORBIDDEN {
                    continue;
                }
                let slot = &mut next[mask | 1 << j];
                let total = cost + c;
                if slot.is_none_or(|old| total < old) {
                    *slot = Some(total);
                }
            }
        }
        best = next;
    }
    best.into_iter().flatten().min()
}

/// A random small instance: costs in `lo..=hi`, each entry forbidden with probability
/// `forbid_percent`%.
pub fn small(
    rows: usize,
    cols: usize,
    lo: Cost,
    hi: Cost,
    forbid_percent: u64,
    seed: u64,
) -> Dense {
    let mut rng = Rng::new(seed);
    Dense::from_fn(rows, cols, |_, _| {
        if rng.below(100) < forbid_percent {
            FORBIDDEN
        } else {
            rng.range(lo, hi)
        }
    })
    .unwrap()
}

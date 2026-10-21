//! Bertsekas' auction algorithm with epsilon scaling (1979, 1988), exact for integer costs.
//!
//! Persons (rows) bid for objects (columns). A person takes its best object at a price that makes
//! it only `epsilon` better off than its second choice, evicting whoever held it. For a fixed
//! `epsilon` the result is within `n * epsilon` of optimal, so costs are first multiplied by
//! `n + 1`: at `epsilon = 1` the total slack is below one unit of the original cost, and with
//! integer costs that means optimal. Scaling starts coarse and divides `epsilon` by a constant each
//! phase, keeping the prices from the previous phase, which is what makes it fast.
//!
//! The auction cannot tell an infeasible instance from a slow one (prices just keep rising), so a
//! Hopcroft-Karp pass decides feasibility first. It handles square matrices; the caller pads a
//! rectangular one with zero-cost dummy rows. It does not produce exact dual variables; the
//! verifier derives a certificate from the assignment instead.

use crate::error::{Error, Result};
use crate::hungarian::Core;
use crate::matching::max_matching;
use crate::matrix::{Cost, Sparse};

const NONE: usize = usize::MAX;
/// Epsilon is divided by this each phase. 4 to 10 is the usual range; see the README for the sweep.
pub const SCALING_FACTOR: Cost = 7;

pub(crate) fn solve(g: &Sparse) -> Result<Core> {
    solve_with(g, SCALING_FACTOR).map(|(core, _)| core)
}

/// Number of bids the auction makes on `g` with the given scaling factor (the README's sweep).
pub fn count_bids(g: &Sparse, factor: Cost) -> Result<u64> {
    solve_with(g, factor).map(|(_, bids)| bids)
}

/// Like [`solve`] with an explicit scaling factor; also returns the number of bids made.
fn solve_with(g: &Sparse, factor: Cost) -> Result<(Core, u64)> {
    let n = g.rows();
    if n != g.cols() {
        return Err(Error::Unsupported("the auction needs a square matrix"));
    }
    if n == 0 {
        return Ok((
            Core {
                row_to_col: Vec::new(),
                duals: None,
            },
            0,
        ));
    }
    if max_matching(g) < n {
        return Err(Error::Infeasible);
    }
    let (lo, hi) = g.allowed_range().expect("a perfect matching has edges");
    let scale = n as Cost + 1;
    let range = hi - lo;
    // Prices can climb to about n * scaled_range; keep that far below 2^63.
    if (range as i128 + 1) * scale as i128 * (n as i128 + 2) > 1i128 << 61 {
        return Err(Error::OutOfRange(
            "costs too large for the auction's scaled prices".into(),
        ));
    }

    // Benefit of person i for object j is -(c - lo) * scale; value = benefit - price.
    let benefit = |c: Cost| -(c - lo) * scale;
    let mut price = vec![0 as Cost; n];
    let mut owner = vec![NONE; n];
    let mut assigned = vec![NONE; n];
    let mut stack: Vec<usize> = Vec::with_capacity(n);
    let mut bids = 0u64;
    let mut eps = (range * scale / 2).max(1);
    loop {
        owner.fill(NONE);
        assigned.fill(NONE);
        stack.clear();
        stack.extend((0..n).rev());
        while let Some(i) = stack.pop() {
            let (mut best_j, mut best, mut second) = (NONE, Cost::MIN, Cost::MIN);
            for (&j, &c) in g.row_cols(i).iter().zip(g.row_costs(i)) {
                let val = benefit(c) - price[j as usize];
                if val > best {
                    second = best;
                    best = val;
                    best_j = j as usize;
                } else if val > second {
                    second = val;
                }
            }
            // A row with a single edge has no runner-up: it just outbids by epsilon.
            let gap = if second == Cost::MIN {
                0
            } else {
                best - second
            };
            price[best_j] += gap + eps;
            bids += 1;
            let previous = owner[best_j];
            owner[best_j] = i;
            assigned[i] = best_j;
            if previous != NONE {
                assigned[previous] = NONE;
                stack.push(previous);
            }
        }
        if eps == 1 {
            break;
        }
        eps = (eps / factor).max(1);
    }
    Ok((
        Core {
            row_to_col: assigned.iter().map(|&j| j as u32).collect(),
            duals: None,
        },
        bids,
    ))
}

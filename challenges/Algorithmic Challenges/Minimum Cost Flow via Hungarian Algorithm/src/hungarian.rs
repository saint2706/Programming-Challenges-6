//! The Hungarian algorithm (Kuhn 1955, Munkres 1957) in its O(n^2 m) potentials form.
//!
//! This is the version Edmonds and Karp (1972) and Tomizawa (1971) reduced it to: rows are added
//! one at a time, each by a Dijkstra-like search for the cheapest augmenting path in the graph of
//! tight edges, and the dual variables `u` (rows) and `v` (columns) absorb the path length. It is
//! the baseline every other dense solver here is measured against. The step-by-step
//! "cover the zeros with the fewest lines" presentation from textbooks is the same algorithm with
//! worse bookkeeping (O(n^4) naively), so it is not implemented separately.
//!
//! One detail decides whether it behaves like O(n^2) or O(n^3) on matrices with many equal costs:
//! when several columns tie for the smallest slack, which one is taken next. Taking the first (the
//! usual code) can walk through every tied *matched* column before it meets a free one; taking a
//! free one ends the search at once. [`TieBreak`] exposes both; the solvers use `PreferFree`.

use crate::error::{Error, Result};
use crate::matrix::{Cost, Dense, FORBIDDEN};

/// What a rectangular core solver returns: every row of the (smaller) row side is matched.
pub(crate) struct Core {
    /// Column of every row.
    pub row_to_col: Vec<u32>,
    /// Dual variables `(u, v)` with `c_ij >= u_i + v_j`, tight on matched edges, if the solver
    /// produces them.
    pub duals: Option<(Vec<Cost>, Vec<Cost>)>,
}

const INF: Cost = Cost::MAX;

/// How the search picks among columns tied for the smallest slack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TieBreak {
    /// The lowest-numbered one, as in the usual presentation.
    First,
    /// A column that is still free if there is one: the augmenting path is complete.
    PreferFree,
}

/// Optimal cost under a chosen tie-break (the two always agree; only the running time differs).
pub fn optimal_cost(c: &Dense, tie: TieBreak) -> Result<Cost> {
    let owned;
    let work = if c.rows() > c.cols() {
        owned = c.transposed();
        &owned
    } else {
        c
    };
    let core = solve_with(work, tie)?;
    Ok(core
        .row_to_col
        .iter()
        .enumerate()
        .map(|(i, &j)| work.get(i, j as usize))
        .sum())
}

/// Solves the assignment problem on `c` (`rows <= cols`), forbidden edges allowed.
pub(crate) fn solve(c: &Dense) -> Result<Core> {
    solve_with(c, TieBreak::PreferFree)
}

pub(crate) fn solve_with(c: &Dense, tie: TieBreak) -> Result<Core> {
    let (n, m) = (c.rows(), c.cols());
    debug_assert!(n <= m);
    // 1-based, with index 0 as the virtual column the current row hangs from.
    let mut u = vec![0 as Cost; n + 1];
    let mut v = vec![0 as Cost; m + 1];
    let mut matched_row = vec![0usize; m + 1]; // row (1-based) holding column j, 0 = free
    let mut way = vec![0usize; m + 1]; // previous column on the shortest path to column j
    let mut minv = vec![INF; m + 1];
    let mut used = vec![false; m + 1];

    for i in 1..=n {
        matched_row[0] = i;
        let mut j0 = 0usize;
        minv.fill(INF);
        used.fill(false);
        loop {
            used[j0] = true;
            let i0 = matched_row[j0];
            let row = c.row(i0 - 1);
            let mut delta = INF;
            let mut j1 = 0usize;
            for j in 1..=m {
                if used[j] {
                    continue;
                }
                let w = row[j - 1];
                if w != FORBIDDEN {
                    let cur = w - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                }
                if minv[j] < delta
                    || (tie == TieBreak::PreferFree && minv[j] == delta && matched_row[j] == 0)
                {
                    delta = minv[j];
                    j1 = j;
                }
            }
            if delta == INF {
                return Err(Error::Infeasible);
            }
            for j in 0..=m {
                if used[j] {
                    u[matched_row[j]] += delta;
                    v[j] -= delta;
                } else if minv[j] != INF {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if matched_row[j0] == 0 {
                break;
            }
        }
        // Flip the augmenting path.
        loop {
            let j1 = way[j0];
            matched_row[j0] = matched_row[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut row_to_col = vec![0u32; n];
    for j in 1..=m {
        if matched_row[j] != 0 {
            row_to_col[matched_row[j] - 1] = (j - 1) as u32;
        }
    }
    Ok(Core {
        row_to_col,
        duals: Some((u[1..].to_vec(), v[1..].to_vec())),
    })
}

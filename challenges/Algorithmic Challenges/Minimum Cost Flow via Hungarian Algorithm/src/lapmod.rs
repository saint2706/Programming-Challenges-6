//! Sparse assignment: successive shortest augmenting paths over a compressed-sparse-row matrix, so
//! a missing edge costs nothing and a row with `d` edges costs `d`, not `m`. It follows the idea of
//! LAPMOD, the sparse form of Jonker-Volgenant (Jonker and Volgenant 1987 treat the sparse case;
//! Volgenant 1996 adds a core-oriented version), but it is not a port of it.
//!
//! What this version does: starts every row on its cheapest edge (taking a tight edge to a free
//! column when there is one), then runs one Dijkstra per still-free row over the sparse edges with
//! a binary heap, stopping at the first free column. What it does not do: LAPMOD's core-oriented
//! refinement or its column-reduction initialisation. Dual updates are exactly Crouse's (see
//! `jv.rs`), so the same certificate applies.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::error::{Error, Result};
use crate::hungarian::Core;
use crate::matrix::{Cost, Sparse};

const INF: Cost = Cost::MAX;
const NONE: usize = usize::MAX;

/// Solves the assignment problem on `g` (`rows <= cols`).
pub(crate) fn solve(g: &Sparse) -> Result<Core> {
    let (n, m) = (g.rows(), g.cols());
    debug_assert!(n <= m);
    let mut u = vec![0 as Cost; n];
    let mut v = vec![0 as Cost; m];
    let mut row_to_col = vec![NONE; n];
    let mut col_to_row = vec![NONE; m];

    // Initialisation: u_i = cheapest edge of row i, then take a tight edge to a free column.
    for i in 0..n {
        let costs = g.row_costs(i);
        let Some(&min) = costs.iter().min() else {
            return Err(Error::Infeasible);
        };
        u[i] = min;
        for (&j, &c) in g.row_cols(i).iter().zip(costs) {
            if c == min && col_to_row[j as usize] == NONE {
                row_to_col[i] = j as usize;
                col_to_row[j as usize] = i;
                break;
            }
        }
    }

    let mut dist = vec![INF; m];
    let mut pred = vec![NONE; m];
    let mut done = vec![false; m];
    let mut touched: Vec<usize> = Vec::new();
    let mut visited_rows: Vec<usize> = Vec::new();
    let mut finished_cols: Vec<usize> = Vec::new();
    let mut heap: BinaryHeap<Reverse<(Cost, u32)>> = BinaryHeap::new();

    for r0 in 0..n {
        if row_to_col[r0] != NONE {
            continue;
        }
        heap.clear();
        visited_rows.clear();
        finished_cols.clear();

        let relax = |i: usize,
                     base: Cost,
                     dist: &mut [Cost],
                     pred: &mut [usize],
                     done: &[bool],
                     touched: &mut Vec<usize>,
                     heap: &mut BinaryHeap<Reverse<(Cost, u32)>>,
                     u: &[Cost],
                     v: &[Cost]| {
            for (&j, &c) in g.row_cols(i).iter().zip(g.row_costs(i)) {
                let j = j as usize;
                if done[j] {
                    continue;
                }
                let nd = base + c - u[i] - v[j];
                if nd < dist[j] {
                    if dist[j] == INF {
                        touched.push(j);
                    }
                    dist[j] = nd;
                    pred[j] = i;
                    heap.push(Reverse((nd, j as u32)));
                }
            }
        };

        visited_rows.push(r0);
        relax(
            r0,
            0,
            &mut dist,
            &mut pred,
            &done,
            &mut touched,
            &mut heap,
            &u,
            &v,
        );
        let (sink, min_val) = loop {
            let Some(Reverse((d, j))) = heap.pop() else {
                reset(&touched, &mut dist, &mut done);
                touched.clear();
                return Err(Error::Infeasible);
            };
            let j = j as usize;
            if done[j] || d > dist[j] {
                continue;
            }
            done[j] = true;
            finished_cols.push(j);
            if col_to_row[j] == NONE {
                break (j, d);
            }
            let i = col_to_row[j];
            visited_rows.push(i);
            relax(
                i,
                d,
                &mut dist,
                &mut pred,
                &done,
                &mut touched,
                &mut heap,
                &u,
                &v,
            );
        };

        u[r0] += min_val;
        for &i in &visited_rows[1..] {
            u[i] += min_val - dist[row_to_col[i]];
        }
        for &j in &finished_cols {
            v[j] -= min_val - dist[j];
        }

        let mut j = sink;
        loop {
            let i = pred[j];
            col_to_row[j] = i;
            let previous = std::mem::replace(&mut row_to_col[i], j);
            if i == r0 {
                break;
            }
            j = previous;
        }
        reset(&touched, &mut dist, &mut done);
        touched.clear();
    }

    Ok(Core {
        row_to_col: row_to_col.iter().map(|&j| j as u32).collect(),
        duals: Some((u, v)),
    })
}

fn reset(touched: &[usize], dist: &mut [Cost], done: &mut [bool]) {
    for &j in touched {
        dist[j] = INF;
        done[j] = false;
    }
}

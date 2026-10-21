//! Jonker-Volgenant (1987) shortest augmenting paths, in two forms.
//!
//! * [`crouse`]: the rectangular version from Crouse (IEEE TAES 52(4), 2016) that SciPy's
//!   `linear_sum_assignment` runs. It is the Hungarian algorithm's augmenting-path search with the
//!   next column chosen by a scan over the remaining columns, and no initialisation heuristics.
//!   Handles `rows <= cols` directly and forbidden edges as unreachable.
//! * [`lapjv`]: the full LAPJV with its three initialisation phases (column reduction, reduction
//!   transfer, two rounds of augmenting row reduction) before the shortest-path phase. Square
//!   only, no forbidden edges (that is what LAPMOD in `lapmod.rs` is for); a rectangular input is
//!   padded with zero-cost rows, which leaves the optimum unchanged.

use crate::error::{Error, Result};
use crate::hungarian::Core;
use crate::matrix::{Cost, Dense, FORBIDDEN};

const INF: Cost = Cost::MAX;
const NONE: usize = usize::MAX;

pub(crate) fn crouse(c: &Dense) -> Result<Core> {
    let (nr, nc) = (c.rows(), c.cols());
    debug_assert!(nr <= nc);
    let mut u = vec![0 as Cost; nr];
    let mut v = vec![0 as Cost; nc];
    let mut shortest = vec![INF; nc];
    let mut path = vec![NONE; nc];
    let mut col4row = vec![NONE; nr];
    let mut row4col = vec![NONE; nc];
    let mut seen_row = vec![false; nr];
    let mut seen_col = vec![false; nc];
    let mut remaining: Vec<usize> = Vec::with_capacity(nc);

    for cur_row in 0..nr {
        remaining.clear();
        remaining.extend((0..nc).rev());
        seen_row.fill(false);
        seen_col.fill(false);
        shortest.fill(INF);

        let mut min_val: Cost = 0;
        let mut i = cur_row;
        let sink = loop {
            seen_row[i] = true;
            let row = c.row(i);
            let mut index = NONE;
            let mut lowest = INF;
            for (it, &j) in remaining.iter().enumerate() {
                let w = row[j];
                if w != FORBIDDEN {
                    let r = min_val + w - u[i] - v[j];
                    if r < shortest[j] {
                        path[j] = i;
                        shortest[j] = r;
                    }
                }
                // On a tie prefer a free column: it ends the search at once.
                if shortest[j] < lowest || (shortest[j] == lowest && row4col[j] == NONE) {
                    lowest = shortest[j];
                    index = it;
                }
            }
            if index == NONE || lowest == INF {
                return Err(Error::Infeasible);
            }
            min_val = lowest;
            let j = remaining.swap_remove(index);
            seen_col[j] = true;
            if row4col[j] == NONE {
                break j;
            }
            i = row4col[j];
        };

        u[cur_row] += min_val;
        for r in 0..nr {
            if seen_row[r] && r != cur_row {
                u[r] += min_val - shortest[col4row[r]];
            }
        }
        for (j, vj) in v.iter_mut().enumerate() {
            if seen_col[j] {
                *vj -= min_val - shortest[j];
            }
        }

        let mut j = sink;
        loop {
            let r = path[j];
            row4col[j] = r;
            std::mem::swap(&mut col4row[r], &mut j);
            if r == cur_row {
                break;
            }
        }
    }

    Ok(Core {
        row_to_col: col4row.iter().map(|&j| j as u32).collect(),
        duals: Some((u, v)),
    })
}

/// LAPJV on a square dense matrix without forbidden edges. Returns the row assignment and the
/// column prices `v` (reduced cost `c_ij - v_j`).
fn lapjv_square(c: &[Cost], dim: usize) -> (Vec<usize>, Vec<Cost>) {
    let at = |i: usize, j: usize| c[i * dim + j];
    let mut v = vec![0 as Cost; dim];
    let mut rowsol = vec![NONE; dim];
    let mut colsol = vec![NONE; dim];
    let mut matches = vec![0usize; dim];
    let mut free: Vec<usize> = Vec::with_capacity(dim);

    // Column reduction, scanning columns last to first (gives better starts than first to last).
    for j in (0..dim).rev() {
        let (mut imin, mut min) = (0usize, at(0, j));
        for i in 1..dim {
            if at(i, j) < min {
                min = at(i, j);
                imin = i;
            }
        }
        v[j] = min;
        matches[imin] += 1;
        if matches[imin] == 1 {
            rowsol[imin] = j;
            colsol[j] = imin;
        } else if v[j] < v[rowsol[imin]] {
            let j1 = rowsol[imin];
            rowsol[imin] = j;
            colsol[j] = imin;
            colsol[j1] = NONE;
        } else {
            colsol[j] = NONE;
        }
    }

    // Reduction transfer from rows assigned exactly once.
    for i in 0..dim {
        if matches[i] == 0 {
            free.push(i);
        } else if matches[i] == 1 {
            let j1 = rowsol[i];
            let min = (0..dim).filter(|&j| j != j1).map(|j| at(i, j) - v[j]).min();
            if let Some(min) = min {
                v[j1] -= min;
            }
        }
    }

    // Augmenting row reduction, twice.
    for _ in 0..2 {
        if free.is_empty() {
            break;
        }
        let prev = std::mem::take(&mut free);
        let mut queue = prev;
        let mut k = 0;
        while k < queue.len() {
            let i = queue[k];
            k += 1;
            let mut umin = at(i, 0) - v[0];
            let mut j1 = 0usize;
            let mut usubmin = INF;
            let mut j2 = NONE;
            for (j, &vj) in v.iter().enumerate().skip(1) {
                let h = at(i, j) - vj;
                if h < usubmin {
                    if h >= umin {
                        usubmin = h;
                        j2 = j;
                    } else {
                        usubmin = umin;
                        umin = h;
                        j2 = j1;
                        j1 = j;
                    }
                }
            }
            let mut i0 = colsol[j1];
            if umin < usubmin {
                // Raise the minimum column's price until row i is indifferent between two columns.
                v[j1] -= usubmin - umin;
            } else if i0 != NONE && j2 != NONE {
                // A tie: take the second column instead, it may be free.
                j1 = j2;
                i0 = colsol[j2];
            }
            rowsol[i] = j1;
            colsol[j1] = i;
            if i0 != NONE {
                if umin < usubmin {
                    // Continue the same augmenting path with the displaced row.
                    k -= 1;
                    queue[k] = i0;
                } else {
                    free.push(i0);
                }
            }
        }
    }

    // Shortest augmenting paths for every row still free.
    let mut d = vec![0 as Cost; dim];
    let mut pred = vec![0usize; dim];
    let mut collist = vec![0usize; dim];
    for &free_row in &free {
        for j in 0..dim {
            d[j] = at(free_row, j) - v[j];
            pred[j] = free_row;
            collist[j] = j;
        }
        let (mut low, mut up) = (0usize, 0usize);
        let mut min = 0 as Cost;
        let mut last = 0usize;
        let sink_col: usize;
        'search: loop {
            if up == low {
                last = low;
                min = d[collist[up]];
                up += 1;
                // The scan starts where `up` stood; `up` then advances as columns tie the minimum.
                let scan_from = up;
                for k in scan_from..dim {
                    let j = collist[k];
                    let h = d[j];
                    if h <= min {
                        if h < min {
                            up = low;
                            min = h;
                        }
                        collist[k] = collist[up];
                        collist[up] = j;
                        up += 1;
                    }
                }
                for k in low..up {
                    if colsol[collist[k]] == NONE {
                        sink_col = collist[k];
                        break 'search;
                    }
                }
            }
            let j1 = collist[low];
            low += 1;
            let i = colsol[j1];
            let h = at(i, j1) - v[j1] - min;
            let scan_from = up;
            for k in scan_from..dim {
                let j = collist[k];
                let v2 = at(i, j) - v[j] - h;
                if v2 < d[j] {
                    pred[j] = i;
                    if v2 == min {
                        if colsol[j] == NONE {
                            sink_col = j;
                            break 'search;
                        }
                        collist[k] = collist[up];
                        collist[up] = j;
                        up += 1;
                    }
                    d[j] = v2;
                }
            }
        }
        for &j1 in &collist[..last] {
            v[j1] += d[j1] - min;
        }
        let mut end_of_path = sink_col;
        loop {
            let i = pred[end_of_path];
            colsol[end_of_path] = i;
            std::mem::swap(&mut end_of_path, &mut rowsol[i]);
            if i == free_row {
                break;
            }
        }
    }
    (rowsol, v)
}

pub(crate) fn lapjv(c: &Dense) -> Result<Core> {
    if c.has_forbidden() {
        return Err(Error::Unsupported(
            "lapjv has no forbidden edges; use lapmod, hungarian or crouse",
        ));
    }
    let (n, m) = (c.rows(), c.cols());
    debug_assert!(n <= m);
    if n == 0 {
        return Ok(Core {
            row_to_col: Vec::new(),
            duals: Some((Vec::new(), vec![0; m])),
        });
    }
    // Pad to a square with zero-cost rows when there are more columns than rows.
    let padded;
    let square: &[Cost] = if n == m {
        c.as_slice()
    } else {
        let mut p = Vec::with_capacity(m * m);
        p.extend_from_slice(c.as_slice());
        p.resize(m * m, 0);
        padded = p;
        &padded
    };
    let (rowsol, mut v) = lapjv_square(square, m);

    let row_to_col: Vec<u32> = rowsol[..n].iter().map(|&j| j as u32).collect();
    let mut u: Vec<Cost> = (0..n).map(|i| c.get(i, rowsol[i]) - v[rowsol[i]]).collect();
    if n < m {
        // The dummy rows are identical, so every column they hold has the same (largest) price.
        // Shifting it into the rows leaves u_i + v_j unchanged and gives the rectangular
        // certificate: v <= 0, and exactly 0 on unmatched columns.
        let top = *v.iter().max().expect("m > 0");
        for x in &mut u {
            *x += top;
        }
        for x in &mut v {
            *x -= top;
        }
    }
    Ok(Core {
        row_to_col,
        duals: Some((u, v)),
    })
}

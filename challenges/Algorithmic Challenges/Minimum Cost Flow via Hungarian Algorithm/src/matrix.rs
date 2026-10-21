//! Cost matrices, dense and sparse, and the limits that keep every solver's arithmetic exact.
//!
//! Costs are `i64`. Floating-point duals are the classic way to get an assignment solver stuck
//! (a reduced cost that should be 0 comes out as -1e-16 and a Dijkstra step goes the wrong way), so
//! nothing here is floating point. Instead the inputs are bounded so that every sum a solver forms
//! (potentials up to `n * max_cost`, auction prices scaled by `n + 1`) fits in an `i64`:
//!
//! * `|cost| <= MAX_ABS_COST` (2^32), or the sentinel [`FORBIDDEN`] meaning "no such edge";
//! * each dimension at most `MAX_DIM` (2^24).
//!
//! Solvers that scale further (the auction, cost scaling) check their own tighter bound and return
//! [`Error::OutOfRange`] rather than wrap around.

use crate::error::{Error, Result};

pub type Cost = i64;

/// Marks an edge that must not be used. Only [`Dense`] stores it; [`Sparse`] omits such edges.
pub const FORBIDDEN: Cost = Cost::MAX;
pub const MAX_ABS_COST: Cost = 1 << 32;
pub const MAX_DIM: usize = 1 << 24;

fn check_cost(c: Cost) -> Result<()> {
    if c == FORBIDDEN || (-MAX_ABS_COST..=MAX_ABS_COST).contains(&c) {
        Ok(())
    } else {
        Err(Error::OutOfRange(format!(
            "cost {c} exceeds the supported magnitude 2^32"
        )))
    }
}

fn check_dims(rows: usize, cols: usize) -> Result<()> {
    if rows > MAX_DIM || cols > MAX_DIM {
        Err(Error::OutOfRange(format!(
            "dimension {rows}x{cols} exceeds 2^24"
        )))
    } else {
        Ok(())
    }
}

/// A row-major `rows x cols` cost matrix. Row `i` is a worker, column `j` a task.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dense {
    rows: usize,
    cols: usize,
    data: Vec<Cost>,
}

impl Dense {
    pub fn new(rows: usize, cols: usize, data: Vec<Cost>) -> Result<Self> {
        check_dims(rows, cols)?;
        if data.len() != rows * cols {
            return Err(Error::Invalid(format!(
                "{rows}x{cols} matrix needs {} entries, got {}",
                rows * cols,
                data.len()
            )));
        }
        for &c in &data {
            check_cost(c)?;
        }
        Ok(Self { rows, cols, data })
    }

    pub fn from_rows(rows: &[Vec<Cost>]) -> Result<Self> {
        let cols = rows.first().map_or(0, Vec::len);
        if rows.iter().any(|r| r.len() != cols) {
            return Err(Error::Invalid("ragged rows".into()));
        }
        Self::new(rows.len(), cols, rows.concat())
    }

    pub fn from_fn(
        rows: usize,
        cols: usize,
        mut f: impl FnMut(usize, usize) -> Cost,
    ) -> Result<Self> {
        check_dims(rows, cols)?;
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(f(i, j));
            }
        }
        Self::new(rows, cols, data)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> Cost {
        self.data[i * self.cols + j]
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[Cost] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// The entries in row-major order.
    pub fn as_slice(&self) -> &[Cost] {
        &self.data
    }

    pub fn has_forbidden(&self) -> bool {
        self.data.contains(&FORBIDDEN)
    }

    pub fn transposed(&self) -> Dense {
        let mut data = Vec::with_capacity(self.data.len());
        for j in 0..self.cols {
            for i in 0..self.rows {
                data.push(self.get(i, j));
            }
        }
        Dense {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }

    /// Smallest and largest allowed cost, `None` if every entry is forbidden.
    pub fn allowed_range(&self) -> Option<(Cost, Cost)> {
        let mut it = self.data.iter().copied().filter(|&c| c != FORBIDDEN);
        let first = it.next()?;
        Some(it.fold((first, first), |(lo, hi), c| (lo.min(c), hi.max(c))))
    }

    pub fn to_sparse(&self) -> Sparse {
        let mut row_ptr = Vec::with_capacity(self.rows + 1);
        let mut col = Vec::new();
        let mut cost = Vec::new();
        row_ptr.push(0);
        for i in 0..self.rows {
            for (j, &c) in self.row(i).iter().enumerate() {
                if c != FORBIDDEN {
                    col.push(j as u32);
                    cost.push(c);
                }
            }
            row_ptr.push(col.len());
        }
        Sparse {
            rows: self.rows,
            cols: self.cols,
            row_ptr,
            col,
            cost,
        }
    }
}

/// A compressed-sparse-row cost matrix; an absent entry is a forbidden edge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sparse {
    rows: usize,
    cols: usize,
    row_ptr: Vec<usize>,
    col: Vec<u32>,
    cost: Vec<Cost>,
}

impl Sparse {
    /// Builds a matrix from `(row, col, cost)` triplets. Parallel edges keep the cheapest cost and
    /// `FORBIDDEN` entries are dropped.
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        triplets: impl IntoIterator<Item = (usize, usize, Cost)>,
    ) -> Result<Self> {
        check_dims(rows, cols)?;
        let mut t: Vec<(u32, u32, Cost)> = Vec::new();
        for (i, j, c) in triplets {
            if i >= rows || j >= cols {
                return Err(Error::Invalid(format!(
                    "edge ({i}, {j}) outside a {rows}x{cols} matrix"
                )));
            }
            check_cost(c)?;
            if c != FORBIDDEN {
                t.push((i as u32, j as u32, c));
            }
        }
        t.sort_unstable();
        let mut row_ptr = vec![0usize; rows + 1];
        let mut col = Vec::with_capacity(t.len());
        let mut cost = Vec::with_capacity(t.len());
        let mut last = (u32::MAX, u32::MAX);
        for &(i, j, c) in &t {
            if (i, j) == last {
                continue; // sorted by cost within a duplicate run, so the first one is the cheapest
            }
            last = (i, j);
            row_ptr[i as usize + 1] += 1;
            col.push(j);
            cost.push(c);
        }
        for i in 0..rows {
            row_ptr[i + 1] += row_ptr[i];
        }
        Ok(Self {
            rows,
            cols,
            row_ptr,
            col,
            cost,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn nnz(&self) -> usize {
        self.col.len()
    }

    #[inline]
    pub fn row_cols(&self, i: usize) -> &[u32] {
        &self.col[self.row_ptr[i]..self.row_ptr[i + 1]]
    }

    #[inline]
    pub fn row_costs(&self, i: usize) -> &[Cost] {
        &self.cost[self.row_ptr[i]..self.row_ptr[i + 1]]
    }

    /// `(column, cost)` pairs of row `i`, columns ascending.
    pub fn row(&self, i: usize) -> impl Iterator<Item = (usize, Cost)> + '_ {
        self.row_cols(i)
            .iter()
            .zip(self.row_costs(i))
            .map(|(&j, &c)| (j as usize, c))
    }

    pub fn get(&self, i: usize, j: usize) -> Option<Cost> {
        let cols = self.row_cols(i);
        cols.binary_search(&(j as u32))
            .ok()
            .map(|k| self.row_costs(i)[k])
    }

    pub fn allowed_range(&self) -> Option<(Cost, Cost)> {
        let first = *self.cost.first()?;
        Some(
            self.cost
                .iter()
                .fold((first, first), |(lo, hi), &c| (lo.min(c), hi.max(c))),
        )
    }

    pub fn transposed(&self) -> Sparse {
        Sparse::from_triplets(
            self.cols,
            self.rows,
            (0..self.rows).flat_map(|i| self.row(i).map(move |(j, c)| (j, i, c))),
        )
        .expect("a valid matrix transposes to a valid matrix")
    }

    pub fn to_dense(&self) -> Result<Dense> {
        check_dims(self.rows, self.cols)?;
        let mut data = vec![FORBIDDEN; self.rows * self.cols];
        for i in 0..self.rows {
            for (j, c) in self.row(i) {
                data[i * self.cols + j] = c;
            }
        }
        Ok(Dense {
            rows: self.rows,
            cols: self.cols,
            data,
        })
    }
}

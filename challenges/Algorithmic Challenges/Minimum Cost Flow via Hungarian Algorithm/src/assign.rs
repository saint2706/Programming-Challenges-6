//! The assignment problem: pick, for every row of the smaller side, a distinct column of the other
//! side, minimising total cost.
//!
//! For `rows <= cols` every row is matched and some columns stay free; for `rows > cols` every
//! column is matched and some rows stay free (the same convention as SciPy's
//! `linear_sum_assignment`). Forbidden edges ([`FORBIDDEN`] in a [`Dense`], absent in a
//! [`Sparse`]) may not be used; if no assignment of full size avoids them the result is
//! [`Error::Infeasible`](crate::Error::Infeasible).
//!
//! [`solve_dense`] and [`solve_sparse`] run any [`Algorithm`] on either layout, converting and
//! transposing as needed, so all of them can be compared on the same input.

use std::fmt;
use std::str::FromStr;

use crate::auction;
use crate::error::{Error, Result};
use crate::flow;
use crate::hungarian::{self, Core};
use crate::jv;
use crate::lapmod;
use crate::matrix::{Cost, Dense, Sparse};

/// Dual variables of the assignment LP: `c_ij >= u_i + v_j` for every edge, with equality on
/// matched ones. On the larger side the duals are `<= 0`, and `0` on unmatched vertices; then
/// `sum(u) + sum(v)` equals the optimal cost, which is how [`crate::verify`] proves optimality.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Duals {
    pub u: Vec<Cost>,
    pub v: Vec<Cost>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assignment {
    pub cost: Cost,
    /// Column assigned to each row, `None` for a row left free (only when `rows > cols`).
    pub row_to_col: Vec<Option<u32>>,
    /// Present when the solver produces exact duals (not the auction or the flow solvers).
    pub duals: Option<Duals>,
}

impl Assignment {
    pub fn matched(&self) -> usize {
        self.row_to_col.iter().flatten().count()
    }

    pub fn pairs(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.row_to_col
            .iter()
            .enumerate()
            .filter_map(|(i, j)| j.map(|j| (i, j as usize)))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Algorithm {
    /// Hungarian algorithm, potentials form, O(n^2 m). Dense, any shape, forbidden edges.
    Hungarian,
    /// Crouse's rectangular Jonker-Volgenant shortest augmenting path (SciPy's algorithm).
    Crouse,
    /// Full LAPJV with its initialisation phases. Dense, no forbidden edges.
    Jv,
    /// Sparse shortest augmenting paths in the spirit of LAPMOD.
    Lapmod,
    /// Auction algorithm with epsilon scaling.
    Auction,
    /// Min-cost flow by successive shortest paths on the assignment network.
    Ssp,
    /// Min-cost flow by Goldberg-Tarjan cost scaling on the assignment network.
    CostScaling,
    /// Min-cost flow by network simplex on the assignment network.
    NetworkSimplex,
}

impl Algorithm {
    pub const ALL: [Algorithm; 8] = [
        Algorithm::Hungarian,
        Algorithm::Crouse,
        Algorithm::Jv,
        Algorithm::Lapmod,
        Algorithm::Auction,
        Algorithm::Ssp,
        Algorithm::CostScaling,
        Algorithm::NetworkSimplex,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Algorithm::Hungarian => "hungarian",
            Algorithm::Crouse => "crouse",
            Algorithm::Jv => "jv",
            Algorithm::Lapmod => "lapmod",
            Algorithm::Auction => "auction",
            Algorithm::Ssp => "ssp",
            Algorithm::CostScaling => "cost-scaling",
            Algorithm::NetworkSimplex => "network-simplex",
        }
    }

    /// The flow solver behind the three network-flow variants.
    pub fn flow_algorithm(self) -> Option<flow::FlowAlgorithm> {
        match self {
            Algorithm::Ssp => Some(flow::FlowAlgorithm::Ssp),
            Algorithm::CostScaling => Some(flow::FlowAlgorithm::CostScaling),
            Algorithm::NetworkSimplex => Some(flow::FlowAlgorithm::NetworkSimplex),
            _ => None,
        }
    }
}

impl fmt::Display for Algorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Algorithm {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Algorithm::ALL
            .into_iter()
            .find(|a| a.name() == s)
            .ok_or_else(|| {
                let names: Vec<_> = Algorithm::ALL.iter().map(|a| a.name()).collect();
                Error::Invalid(format!(
                    "unknown algorithm {s:?}; one of {}",
                    names.join(", ")
                ))
            })
    }
}

fn finish(
    rows: usize,
    cols: usize,
    transposed: bool,
    core: Core,
    cost_of: impl Fn(usize, usize) -> Cost,
) -> Assignment {
    let mut row_to_col = vec![None; rows];
    if transposed {
        // The core solved the transposed problem: its "rows" are our columns.
        for (c, &r) in core.row_to_col.iter().enumerate() {
            row_to_col[r as usize] = Some(c as u32);
        }
    } else {
        for (r, &c) in core.row_to_col.iter().enumerate() {
            row_to_col[r] = Some(c);
        }
    }
    let cost = row_to_col
        .iter()
        .enumerate()
        .filter_map(|(i, j)| j.map(|j| cost_of(i, j as usize)))
        .sum();
    debug_assert!(core.row_to_col.len() == rows.min(cols));
    let duals = core.duals.map(|(u, v)| {
        if transposed {
            Duals { u: v, v: u }
        } else {
            Duals { u, v }
        }
    });
    Assignment {
        cost,
        row_to_col,
        duals,
    }
}

/// Solves a dense instance with `algo`.
pub fn solve_dense(d: &Dense, algo: Algorithm) -> Result<Assignment> {
    match algo {
        Algorithm::Hungarian | Algorithm::Crouse | Algorithm::Jv => {
            let transposed = d.rows() > d.cols();
            let owned;
            let work = if transposed {
                owned = d.transposed();
                &owned
            } else {
                d
            };
            let core = match algo {
                Algorithm::Hungarian => hungarian::solve(work)?,
                Algorithm::Crouse => jv::crouse(work)?,
                _ => jv::lapjv(work)?,
            };
            Ok(finish(d.rows(), d.cols(), transposed, core, |i, j| {
                d.get(i, j)
            }))
        }
        _ => solve_sparse(&d.to_sparse(), algo),
    }
}

/// Adds zero-cost dummy rows, each connected to every column, until the matrix is square.
fn pad_square(g: &Sparse) -> Result<Sparse> {
    debug_assert!(g.rows() <= g.cols());
    let (n, m) = (g.rows(), g.cols());
    let real = (0..n).flat_map(|i| g.row(i).map(move |(j, c)| (i, j, c)));
    let dummy = (n..m).flat_map(|i| (0..m).map(move |j| (i, j, 0)));
    Sparse::from_triplets(m, m, real.chain(dummy))
}

/// Solves a sparse instance with `algo`.
pub fn solve_sparse(g: &Sparse, algo: Algorithm) -> Result<Assignment> {
    if let Some(flow_algo) = algo.flow_algorithm() {
        let row_to_col = flow::solve_assignment(g, flow_algo)?;
        let cost = row_to_col
            .iter()
            .enumerate()
            .filter_map(|(i, j)| j.map(|j| g.get(i, j as usize).expect("flow uses real edges")))
            .sum();
        return Ok(Assignment {
            cost,
            row_to_col,
            duals: None,
        });
    }
    if matches!(
        algo,
        Algorithm::Hungarian | Algorithm::Crouse | Algorithm::Jv
    ) {
        return solve_dense(&g.to_dense()?, algo);
    }
    let transposed = g.rows() > g.cols();
    let owned;
    let work = if transposed {
        owned = g.transposed();
        &owned
    } else {
        g
    };
    let core = match algo {
        Algorithm::Lapmod => lapmod::solve(work)?,
        _ => {
            let mut core = if work.rows() == work.cols() {
                auction::solve(work)?
            } else {
                auction::solve(&pad_square(work)?)?
            };
            core.row_to_col.truncate(work.rows());
            core
        }
    };
    Ok(finish(g.rows(), g.cols(), transposed, core, |i, j| {
        g.get(i, j).expect("solvers use real edges")
    }))
}

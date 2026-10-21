//! Minimum-cost flow networks, and the reduction that turns an assignment into one.
//!
//! A [`Network`] has node supplies `b` (positive = source, negative = sink, summing to zero) and
//! arcs with a capacity and a cost per unit. A feasible flow respects `0 <= x <= cap` and, at every
//! node, `out - in = b`; the objective is `sum(cost * x)`. The three solvers behind
//! [`FlowAlgorithm`] all return the same optimum; they differ in how they get there.
//!
//! Limits, checked on construction so the solvers' i64 arithmetic cannot overflow: at most 2^24
//! nodes and 2^26 arcs, capacities and supplies up to 2^30, costs up to 2^32 in magnitude. The
//! objective is reported as `i128`. Self-loops are rejected (a loop with negative cost would just
//! be saturated; it carries no information).
//!
//! Hungarian algorithm and flow are the same thing: [`assignment_network`] builds the transportation
//! network (rows supply one unit, columns demand one), and successive shortest paths on it *is* the
//! Hungarian algorithm with its dual variables playing the role of node potentials.

use std::collections::BinaryHeap;

use crate::error::{Error, Result};
use crate::matrix::{Cost, MAX_ABS_COST, MAX_DIM, Sparse};
use crate::{cost_scaling, simplex, ssp};

pub const MAX_CAP: i64 = 1 << 30;
pub const MAX_ARCS: usize = 1 << 26;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arc {
    pub from: u32,
    pub to: u32,
    pub cap: i64,
    pub cost: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Network {
    supply: Vec<i64>,
    arcs: Vec<Arc>,
}

impl Network {
    pub fn new(nodes: usize) -> Self {
        assert!(nodes <= MAX_DIM, "at most 2^24 nodes");
        Self {
            supply: vec![0; nodes],
            arcs: Vec::new(),
        }
    }

    pub fn nodes(&self) -> usize {
        self.supply.len()
    }

    pub fn supplies(&self) -> &[i64] {
        &self.supply
    }

    pub fn arcs(&self) -> &[Arc] {
        &self.arcs
    }

    pub fn set_supply(&mut self, node: usize, b: i64) -> Result<()> {
        if node >= self.nodes() {
            return Err(Error::Invalid(format!("node {node} does not exist")));
        }
        if b.abs() > MAX_CAP {
            return Err(Error::OutOfRange(format!("supply {b} exceeds 2^30")));
        }
        self.supply[node] = b;
        Ok(())
    }

    /// Adds an arc and returns its index.
    pub fn add_arc(&mut self, from: usize, to: usize, cap: i64, cost: Cost) -> Result<usize> {
        let n = self.nodes();
        if from >= n || to >= n {
            return Err(Error::Invalid(format!(
                "arc ({from}, {to}) names a missing node"
            )));
        }
        if from == to {
            return Err(Error::Invalid(format!("self-loop at node {from}")));
        }
        if !(0..=MAX_CAP).contains(&cap) {
            return Err(Error::OutOfRange(format!(
                "capacity {cap} outside [0, 2^30]"
            )));
        }
        if cost.abs() > MAX_ABS_COST {
            return Err(Error::OutOfRange(format!("cost {cost} exceeds 2^32")));
        }
        if self.arcs.len() >= MAX_ARCS {
            return Err(Error::OutOfRange("more than 2^26 arcs".into()));
        }
        self.arcs.push(Arc {
            from: from as u32,
            to: to as u32,
            cap,
            cost,
        });
        Ok(self.arcs.len() - 1)
    }

    pub fn check_balanced(&self) -> Result<()> {
        let sum: i128 = self.supply.iter().map(|&b| b as i128).sum();
        if sum != 0 {
            return Err(Error::Unbalanced { sum: sum as i64 });
        }
        Ok(())
    }

    /// Largest `|cost|` over the arcs.
    pub(crate) fn max_abs_cost(&self) -> i64 {
        self.arcs.iter().map(|a| a.cost.abs()).max().unwrap_or(0)
    }
}

/// An optimal flow.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Flow {
    pub cost: i128,
    /// Flow on each arc, in the order the arcs were added.
    pub flow: Vec<i64>,
    /// Node potentials `pi` with reduced cost `cost + pi[from] - pi[to]` that is `>= 0` on arcs
    /// below capacity and `<= 0` on arcs carrying flow. Reported by the solvers that maintain
    /// exact potentials (successive shortest paths, network simplex).
    pub potential: Option<Vec<i64>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FlowAlgorithm {
    /// Successive shortest paths with Johnson potentials and Dijkstra.
    Ssp,
    /// Goldberg-Tarjan cost scaling with push-relabel refinement.
    CostScaling,
    /// Primal network simplex on a strongly feasible spanning tree, block-search pivoting.
    NetworkSimplex,
}

impl FlowAlgorithm {
    pub const ALL: [FlowAlgorithm; 3] = [
        FlowAlgorithm::Ssp,
        FlowAlgorithm::CostScaling,
        FlowAlgorithm::NetworkSimplex,
    ];

    pub fn name(self) -> &'static str {
        match self {
            FlowAlgorithm::Ssp => "ssp",
            FlowAlgorithm::CostScaling => "cost-scaling",
            FlowAlgorithm::NetworkSimplex => "network-simplex",
        }
    }
}

/// Solves the min-cost flow problem; [`Error::Infeasible`] if the supplies cannot be routed.
pub fn solve(net: &Network, algo: FlowAlgorithm) -> Result<Flow> {
    net.check_balanced()?;
    let (flow, potential) = match algo {
        FlowAlgorithm::Ssp => {
            let (flow, pi) = ssp::solve(net)?;
            (flow, Some(pi))
        }
        FlowAlgorithm::CostScaling => (cost_scaling::solve(net)?, None),
        FlowAlgorithm::NetworkSimplex => {
            let (flow, pi) = simplex::solve(net)?;
            (flow, Some(pi))
        }
    };
    let cost = net
        .arcs()
        .iter()
        .zip(&flow)
        .map(|(a, &x)| x as i128 * a.cost as i128)
        .sum();
    Ok(Flow {
        cost,
        flow,
        potential,
    })
}

/// The transportation network of an assignment: every row supplies one unit, every column demands
/// one unit, and each allowed edge is an arc of capacity 1 and its cost. When the sides differ, a
/// dummy node absorbs the surplus of the larger one through zero-cost arcs (it supplies `m - n`
/// units to the columns, or demands `n - m` from the rows), so the supplies still sum to zero.
/// Nodes: rows `0..n`, columns `n..n+m`, and the dummy `n+m` when `n != m`.
///
/// Searching from a single row, as successive shortest paths does here, is exactly one step of the
/// Hungarian algorithm; the potentials become its dual variables.
pub fn assignment_network(g: &Sparse) -> Result<Network> {
    let (n, m) = (g.rows(), g.cols());
    let dummy = n + m;
    let mut net = Network::new(if n == m { n + m } else { n + m + 1 });
    for i in 0..n {
        net.set_supply(i, 1)?;
    }
    for j in 0..m {
        net.set_supply(n + j, -1)?;
    }
    for i in 0..n {
        for (j, c) in g.row(i) {
            net.add_arc(i, n + j, 1, c)?;
        }
    }
    if n < m {
        net.set_supply(dummy, (m - n) as i64)?;
        for j in 0..m {
            net.add_arc(dummy, n + j, 1, 0)?;
        }
    } else if n > m {
        net.set_supply(dummy, -((n - m) as i64))?;
        for i in 0..n {
            net.add_arc(i, dummy, 1, 0)?;
        }
    }
    Ok(net)
}

/// Solves an assignment through the flow network; returns the column of each row.
pub(crate) fn solve_assignment(g: &Sparse, algo: FlowAlgorithm) -> Result<Vec<Option<u32>>> {
    let net = assignment_network(g)?;
    let flow = solve(&net, algo)?;
    let n = g.rows();
    let mut row_to_col = vec![None; n];
    for (arc, &x) in net.arcs().iter().zip(&flow.flow) {
        let (from, to) = (arc.from as usize, arc.to as usize);
        if x == 1 && from < n && (n..n + g.cols()).contains(&to) {
            row_to_col[from] = Some((to - n) as u32);
        }
    }
    Ok(row_to_col)
}

/// The residual graph used by successive shortest paths, cost scaling and max-flow: arc `k` of
/// the input becomes residual arcs `2k` (forward) and `2k + 1` (backward, zero capacity, negated
/// cost), stored in compressed per-node adjacency lists.
pub(crate) struct Residual {
    pub to: Vec<u32>,
    pub rescap: Vec<i64>,
    pub cost: Vec<i64>,
    pub adj_start: Vec<usize>,
    pub adj: Vec<u32>,
}

impl Residual {
    pub fn new(nodes: usize, arcs: &[Arc]) -> Self {
        let mut to = Vec::with_capacity(arcs.len() * 2);
        let mut rescap = Vec::with_capacity(arcs.len() * 2);
        let mut cost = Vec::with_capacity(arcs.len() * 2);
        let mut degree = vec![0usize; nodes + 1];
        for a in arcs {
            to.push(a.to);
            to.push(a.from);
            rescap.push(a.cap);
            rescap.push(0);
            cost.push(a.cost);
            cost.push(-a.cost);
            degree[a.from as usize] += 1;
            degree[a.to as usize] += 1;
        }
        let mut adj_start = vec![0usize; nodes + 1];
        for v in 0..nodes {
            adj_start[v + 1] = adj_start[v] + degree[v];
        }
        let mut fill = adj_start.clone();
        let mut adj = vec![0u32; arcs.len() * 2];
        for (k, a) in arcs.iter().enumerate() {
            adj[fill[a.from as usize]] = (2 * k) as u32;
            fill[a.from as usize] += 1;
            adj[fill[a.to as usize]] = (2 * k + 1) as u32;
            fill[a.to as usize] += 1;
        }
        Self {
            to,
            rescap,
            cost,
            adj_start,
            adj,
        }
    }

    #[inline]
    pub fn out(&self, v: usize) -> &[u32] {
        &self.adj[self.adj_start[v]..self.adj_start[v + 1]]
    }

    #[inline]
    pub fn tail(&self, e: usize) -> usize {
        self.to[e ^ 1] as usize
    }

    #[inline]
    pub fn push(&mut self, e: usize, amount: i64) {
        self.rescap[e] -= amount;
        self.rescap[e ^ 1] += amount;
    }
}

/// A min-heap of `(key, node)`.
pub(crate) type MinHeap = BinaryHeap<std::cmp::Reverse<(i64, u32)>>;

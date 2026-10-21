//! Checking answers without trusting the solver that produced them.
//!
//! Two independent proofs of optimality are offered for an assignment:
//!
//! * [`check_duals`]: the solver's own dual variables satisfy LP complementary slackness, so
//!   `sum(u) + sum(v)` is a lower bound that the assignment meets (O(nnz), fine for big inputs);
//! * [`certify`]: no negative cycle exists in the residual graph, found by Bellman-Ford with no
//!   reference to any dual (O(V * E), for small and medium inputs). This works for every solver,
//!   including the auction and the flow solvers, which hand back no duals.
//!
//! Flows get the same two treatments in [`check_flow`] and [`certify_flow`].

use std::collections::VecDeque;

use crate::assign::Assignment;
use crate::flow::{Flow, Network};
use crate::matrix::{Cost, Sparse};

/// The assignment is legal and its stated cost is right: distinct columns, real edges, full size.
pub fn check_assignment(g: &Sparse, a: &Assignment) -> Result<(), String> {
    if a.row_to_col.len() != g.rows() {
        return Err(format!(
            "row_to_col has {} entries for {} rows",
            a.row_to_col.len(),
            g.rows()
        ));
    }
    let mut used = vec![false; g.cols()];
    let mut cost: i128 = 0;
    for (i, j) in a.pairs() {
        if j >= g.cols() {
            return Err(format!("row {i} assigned to column {j}, out of range"));
        }
        if std::mem::replace(&mut used[j], true) {
            return Err(format!("column {j} assigned twice"));
        }
        let Some(c) = g.get(i, j) else {
            return Err(format!("edge ({i}, {j}) is forbidden"));
        };
        cost += c as i128;
    }
    let want = g.rows().min(g.cols());
    if a.matched() != want {
        return Err(format!("matched {} of {want} required", a.matched()));
    }
    if cost != a.cost as i128 {
        return Err(format!(
            "stated cost {} but the edges sum to {cost}",
            a.cost
        ));
    }
    Ok(())
}

/// Complementary slackness of the solver's dual variables; proves the assignment optimal.
pub fn check_duals(g: &Sparse, a: &Assignment) -> Result<(), String> {
    check_assignment(g, a)?;
    let Some(d) = &a.duals else {
        return Err("this solver reports no duals".into());
    };
    if d.u.len() != g.rows() || d.v.len() != g.cols() {
        return Err("dual vectors have the wrong length".into());
    }
    for i in 0..g.rows() {
        for (j, c) in g.row(i) {
            let slack = c as i128 - d.u[i] as i128 - d.v[j] as i128;
            if slack < 0 {
                return Err(format!("dual infeasible on ({i}, {j}): slack {slack}"));
            }
            if a.row_to_col[i] == Some(j as u32) && slack != 0 {
                return Err(format!("matched edge ({i}, {j}) not tight: slack {slack}"));
            }
        }
    }
    if g.rows() < g.cols() {
        let mut matched = vec![false; g.cols()];
        for (_, j) in a.pairs() {
            matched[j] = true;
        }
        for (j, &vj) in d.v.iter().enumerate() {
            if vj > 0 || (!matched[j] && vj != 0) {
                return Err(format!(
                    "column {j}: dual {vj} breaks v <= 0 / zero when free"
                ));
            }
        }
    } else if g.rows() > g.cols() {
        for (i, &ui) in d.u.iter().enumerate() {
            if ui > 0 || (a.row_to_col[i].is_none() && ui != 0) {
                return Err(format!("row {i}: dual {ui} breaks u <= 0 / zero when free"));
            }
        }
    }
    let total: i128 = d.u.iter().chain(&d.v).map(|&x| x as i128).sum();
    if total != a.cost as i128 {
        return Err(format!(
            "dual objective {total} differs from cost {}",
            a.cost
        ));
    }
    Ok(())
}

/// Shortest-path labels from a virtual source to every vertex of a graph given as an edge list,
/// or `Err` if a negative cycle exists. Queue-based Bellman-Ford (SPFA) with a relaxation count.
fn negative_cycle_free(vertices: usize, edges: &[(u32, u32, i64)]) -> bool {
    let mut first = vec![usize::MAX; vertices];
    let mut next = vec![usize::MAX; edges.len()];
    for (k, &(a, _, _)) in edges.iter().enumerate() {
        next[k] = first[a as usize];
        first[a as usize] = k;
    }
    let mut dist = vec![0i128; vertices];
    let mut count = vec![0usize; vertices];
    let mut queued = vec![true; vertices];
    let mut queue: VecDeque<usize> = (0..vertices).collect();
    while let Some(a) = queue.pop_front() {
        queued[a] = false;
        let mut k = first[a];
        while k != usize::MAX {
            let (_, b, w) = edges[k];
            let b = b as usize;
            let nd = dist[a] + w as i128;
            if nd < dist[b] {
                dist[b] = nd;
                if !queued[b] {
                    count[b] += 1;
                    if count[b] > vertices {
                        return false;
                    }
                    queued[b] = true;
                    queue.push_back(b);
                }
            }
            k = next[k];
        }
    }
    true
}

/// Proves the assignment optimal with no reference to duals: after [`check_assignment`], the
/// residual graph (unused edges forward at cost `c`, used edges backward at `-c`) must have no
/// negative cycle. When one side is larger, a zero-cost hub joins its free and matched vertices so
/// that swapping a matched vertex for a free one also shows up as a cycle.
pub fn certify(g: &Sparse, a: &Assignment) -> Result<(), String> {
    check_assignment(g, a)?;
    let (n, m) = (g.rows(), g.cols());
    let hub = (n + m) as u32;
    let mut edges: Vec<(u32, u32, i64)> = Vec::with_capacity(g.nnz() + n.max(m));
    for i in 0..n {
        for (j, c) in g.row(i) {
            if a.row_to_col[i] == Some(j as u32) {
                edges.push(((n + j) as u32, i as u32, -c));
            } else {
                edges.push((i as u32, (n + j) as u32, c));
            }
        }
    }
    let mut col_matched = vec![false; m];
    for (_, j) in a.pairs() {
        col_matched[j] = true;
    }
    if n < m {
        for (j, &matched) in col_matched.iter().enumerate() {
            let v = (n + j) as u32;
            edges.push(if matched { (hub, v, 0) } else { (v, hub, 0) });
        }
    } else if n > m {
        for (i, j) in a.row_to_col.iter().enumerate() {
            edges.push(if j.is_some() {
                (i as u32, hub, 0)
            } else {
                (hub, i as u32, 0)
            });
        }
    }
    if negative_cycle_free(n + m + 1, &edges) {
        Ok(())
    } else {
        Err("the residual graph has a negative cycle: the assignment is not optimal".into())
    }
}

/// Flow feasibility: capacity bounds, conservation against the supplies, stated cost.
pub fn check_flow(net: &Network, f: &Flow) -> Result<(), String> {
    if f.flow.len() != net.arcs().len() {
        return Err("flow vector has the wrong length".into());
    }
    let mut balance: Vec<i128> = net.supplies().iter().map(|&b| b as i128).collect();
    let mut cost: i128 = 0;
    for (k, (arc, &x)) in net.arcs().iter().zip(&f.flow).enumerate() {
        if x < 0 || x > arc.cap {
            return Err(format!("arc {k}: flow {x} outside [0, {}]", arc.cap));
        }
        balance[arc.from as usize] -= x as i128;
        balance[arc.to as usize] += x as i128;
        cost += x as i128 * arc.cost as i128;
    }
    if let Some(v) = balance.iter().position(|&b| b != 0) {
        return Err(format!("node {v} is out of balance by {}", balance[v]));
    }
    if cost != f.cost {
        return Err(format!("stated cost {} but the arcs sum to {cost}", f.cost));
    }
    Ok(())
}

/// Optimality without reference to potentials: no negative cycle in the residual network.
pub fn certify_flow(net: &Network, f: &Flow) -> Result<(), String> {
    check_flow(net, f)?;
    let mut edges: Vec<(u32, u32, i64)> = Vec::with_capacity(net.arcs().len() * 2);
    for (arc, &x) in net.arcs().iter().zip(&f.flow) {
        if x < arc.cap {
            edges.push((arc.from, arc.to, arc.cost));
        }
        if x > 0 {
            edges.push((arc.to, arc.from, -arc.cost));
        }
    }
    if negative_cycle_free(net.nodes(), &edges) {
        Ok(())
    } else {
        Err("the residual network has a negative cycle: the flow is not optimal".into())
    }
}

/// Reduced-cost optimality of a flow against node potentials `pi` (reduced cost
/// `c + pi[from] - pi[to]`): `>= 0` on arcs below capacity, `<= 0` on arcs above zero.
pub fn check_potentials(net: &Network, f: &Flow, pi: &[Cost]) -> Result<(), String> {
    for (k, (arc, &x)) in net.arcs().iter().zip(&f.flow).enumerate() {
        let r = arc.cost as i128 + pi[arc.from as usize] as i128 - pi[arc.to as usize] as i128;
        if x < arc.cap && r < 0 {
            return Err(format!("arc {k} below capacity with reduced cost {r}"));
        }
        if x > 0 && r > 0 {
            return Err(format!("arc {k} carries flow with reduced cost {r}"));
        }
    }
    Ok(())
}

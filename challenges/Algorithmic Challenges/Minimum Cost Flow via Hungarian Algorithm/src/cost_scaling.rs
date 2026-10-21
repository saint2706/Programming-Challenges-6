//! Cost scaling (Goldberg and Tarjan 1990; Goldberg 1997): the push-relabel analogue for
//! min-cost flow, and the method behind Goldberg's CS2 solver, LEMON's `CostScaling` and OR-Tools'
//! `SimpleMinCostFlow`.
//!
//! A pseudoflow with node prices `pi` is *epsilon-optimal* when every residual arc has reduced
//! cost `c + pi[u] - pi[v] >= -epsilon`. Costs are multiplied by `n + 1`, so an `epsilon = 1`
//! flow is optimal: its total slack is under `n` and the true costs are integers. Starting from a
//! large epsilon, each *refine* phase divides epsilon by a constant, saturates every residual arc
//! with a negative reduced cost (the flow is now 0-optimal but out of balance), and then pushes
//! excess along admissible arcs (reduced cost < 0), lowering a node's price by
//! `epsilon + min reduced cost` when it has none (a relabel), until no node has excess.
//!
//! Plain relabeling is slow on assignment-like networks: a unit of excess can wander for many
//! relabels before it finds a path to a deficit. CS2's remedy, used here, is the *global price
//! update* (Goldberg 1997, "set relabeling"): occasionally run a shortest-path computation backwards
//! from the deficit nodes, with each arc's length its reduced cost in units of epsilon (rounded
//! up, 0 for admissible arcs), and lower every price by epsilon times its label. That keeps
//! epsilon-optimality (see `price_update`) and leaves an admissible path from every node that can
//! reach a deficit. [`Options::price_updates`] switches it off, so the README can show the gap.
//!
//! Infeasible networks make push-relabel run forever, so a max-flow check goes first.

use std::cmp::Reverse;
use std::collections::VecDeque;

use crate::error::{Error, Result};
use crate::flow::{Arc, MinHeap, Network, Residual};
use crate::maxflow;

/// Epsilon is divided by this each phase (Goldberg uses 16 for assignment-like instances).
pub const ALPHA: i64 = 16;
/// An update costs about one pass over the residual arcs, so it is run once relabels have scanned
/// this many times that many arcs (1 and 2 were within about 25% of each other and far ahead of
/// 0.05 on three dense families; see the README).
pub const UPDATE_WORK: f64 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    /// Epsilon shrinks by this factor each phase.
    pub alpha: i64,
    /// Run the global price update at the start of a phase and whenever the arcs scanned by
    /// relabels since the last one exceed `update_work` times the number of residual arcs.
    pub price_updates: bool,
    pub update_work: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            alpha: ALPHA,
            price_updates: true,
            update_work: UPDATE_WORK,
        }
    }
}

/// What one refine phase did. `worst_slack` is the most negative reduced cost left on any residual
/// arc when the phase ended; epsilon-optimality means it is at most `eps`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhaseStats {
    pub eps: i64,
    pub pushes: u64,
    pub relabels: u64,
    pub updates: u64,
    pub worst_slack: i64,
}

pub(crate) fn solve(net: &Network) -> Result<Vec<i64>> {
    solve_with_stats(net, Options::default()).map(|(flow, _)| flow)
}

/// Lowers every price by `epsilon` times its shortest-path label to a deficit node.
///
/// A node's label `d` is the length of its cheapest residual path to any node with negative
/// excess, where an arc `(v, w)` is `c_p / eps + 1` long if its reduced cost `c_p` is non-negative
/// and 0 otherwise. For every residual arc `d_v <= d_w + length`, so after the shift the reduced
/// cost is `c_p - eps * (d_v - d_w)`: at least `-eps` if `c_p >= 0` (the `+ 1` pays for the
/// rounding) and at least `c_p` if `c_p < 0` (then `d_v <= d_w`). Nodes that reach no deficit take
/// the largest label, which only raises reduced costs of arcs into them.
fn price_update(
    res: &Residual,
    pi: &mut [i64],
    excess: &[i64],
    eps: i64,
    dist: &mut [i64],
    heap: &mut MinHeap,
) {
    dist.fill(i64::MAX);
    heap.clear();
    for (v, &e) in excess.iter().enumerate() {
        if e < 0 {
            dist[v] = 0;
            heap.push(Reverse((0, v as u32)));
        }
    }
    if heap.is_empty() {
        return;
    }
    let mut max_label = 0;
    while let Some(Reverse((d, w))) = heap.pop() {
        let w = w as usize;
        if d > dist[w] {
            continue;
        }
        max_label = max_label.max(d);
        for &out in res.out(w) {
            // `out` leaves w, so its partner enters w, from the node `out` points at.
            let arc = (out ^ 1) as usize;
            if res.rescap[arc] == 0 {
                continue;
            }
            let v = res.to[out as usize] as usize;
            let reduced = res.cost[arc] + pi[v] - pi[w];
            let length = if reduced >= 0 { reduced / eps + 1 } else { 0 };
            if d + length < dist[v] {
                dist[v] = d + length;
                heap.push(Reverse((d + length, v as u32)));
            }
        }
    }
    for (v, p) in pi.iter_mut().enumerate() {
        let label = if dist[v] == i64::MAX {
            max_label
        } else {
            dist[v]
        };
        *p -= label * eps;
    }
}

/// Like the plain solver, also returning the statistics of every phase.
pub fn solve_with_stats(net: &Network, opts: Options) -> Result<(Vec<i64>, Vec<PhaseStats>)> {
    let n = net.nodes();
    let m = net.arcs().len();
    if !maxflow::routable(net) {
        return Err(Error::Infeasible);
    }
    let scale = n as i64 + 1;
    let max_cost = net.max_abs_cost();
    // Prices move by at most about 3 * n * eps per phase; keep the sum far below 2^63.
    if 3 * (n as i128 + 1) * scale as i128 * (max_cost as i128 + 1) > 1i128 << 61 {
        return Err(Error::OutOfRange(
            "costs times nodes too large for cost scaling".into(),
        ));
    }
    let arcs: Vec<Arc> = net
        .arcs()
        .iter()
        .map(|a| Arc {
            cost: a.cost * scale,
            ..*a
        })
        .collect();
    let mut res = Residual::new(n, &arcs);
    let mut excess: Vec<i64> = net.supplies().to_vec();
    let mut pi = vec![0i64; n];
    let mut eps = (max_cost * scale).max(1);
    let mut cur = vec![0usize; n];
    let mut queued = vec![false; n];
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut phases: Vec<PhaseStats> = Vec::new();
    let mut dist = vec![0i64; n];
    let mut heap = MinHeap::new();

    loop {
        let (mut pushes, mut relabels, mut updates) = (0u64, 0u64, 0u64);
        // Refine: make the pseudoflow 0-optimal by saturating negative reduced-cost arcs.
        for e in 0..2 * m {
            if res.rescap[e] > 0 {
                let u = res.tail(e);
                let v = res.to[e] as usize;
                if res.cost[e] + pi[u] - pi[v] < 0 {
                    let amount = res.rescap[e];
                    excess[u] -= amount;
                    excess[v] += amount;
                    res.push(e, amount);
                }
            }
        }
        if opts.price_updates {
            price_update(&res, &mut pi, &excess, eps, &mut dist, &mut heap);
            updates += 1;
        }
        cur.copy_from_slice(&res.adj_start[..n]);
        for v in 0..n {
            if excess[v] > 0 {
                queued[v] = true;
                queue.push_back(v);
            }
        }
        let mut work_since_update = 0usize;
        let update_after = (opts.update_work * (2 * m) as f64) as usize;
        while let Some(u) = queue.pop_front() {
            queued[u] = false;
            while excess[u] > 0 {
                if cur[u] == res.adj_start[u + 1] {
                    if opts.price_updates && work_since_update >= update_after {
                        price_update(&res, &mut pi, &excess, eps, &mut dist, &mut heap);
                        updates += 1;
                        work_since_update = 0;
                        cur.copy_from_slice(&res.adj_start[..n]);
                        continue;
                    }
                    // Relabel: no admissible arc left.
                    let mut min = i64::MAX;
                    for &e in res.out(u) {
                        let e = e as usize;
                        if res.rescap[e] > 0 {
                            min = min.min(res.cost[e] + pi[u] - pi[res.to[e] as usize]);
                        }
                    }
                    if min == i64::MAX {
                        return Err(Error::Infeasible);
                    }
                    pi[u] -= min + eps;
                    cur[u] = res.adj_start[u];
                    relabels += 1;
                    work_since_update += res.out(u).len();
                    continue;
                }
                let e = res.adj[cur[u]] as usize;
                let v = res.to[e] as usize;
                if res.rescap[e] > 0 && res.cost[e] + pi[u] - pi[v] < 0 {
                    let amount = excess[u].min(res.rescap[e]);
                    res.push(e, amount);
                    excess[u] -= amount;
                    excess[v] += amount;
                    pushes += 1;
                    if !queued[v] && excess[v] > 0 {
                        queued[v] = true;
                        queue.push_back(v);
                    }
                } else {
                    cur[u] += 1;
                }
            }
        }
        let worst_slack = (0..2 * m)
            .filter(|&e| res.rescap[e] > 0)
            .map(|e| -(res.cost[e] + pi[res.tail(e)] - pi[res.to[e] as usize]))
            .max()
            .unwrap_or(0)
            .max(0);
        phases.push(PhaseStats {
            eps,
            pushes,
            relabels,
            updates,
            worst_slack,
        });
        if eps == 1 {
            break;
        }
        eps = (eps / opts.alpha).max(1);
    }

    Ok(((0..m).map(|k| res.rescap[2 * k + 1]).collect(), phases))
}

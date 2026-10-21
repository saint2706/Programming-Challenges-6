//! Successive shortest paths with node potentials (Edmonds-Karp 1972, Tomizawa 1971).
//!
//! Pick one node with excess, run Dijkstra on *reduced* costs `c + pi[u] - pi[v]` (which the
//! potentials keep non-negative) until it settles a node with a deficit, and push as much as fits
//! along that cheapest path. On an assignment network this is the Hungarian algorithm: the search
//! from one row is its "grow the tight-edge forest" step, and the potentials are `u` and `v`.
//!
//! Details that matter in practice:
//!
//! * Negative-cost arcs would break Dijkstra (and could form negative cycles), so they are
//!   saturated up front; the arc then has residual capacity only backwards, at the positive cost
//!   `-c`, and the imbalance this creates joins the supplies. Zero potentials are then valid.
//! * The search stops at the first deficit node, at distance `D`. Settled nodes move by
//!   `dist - D` and every other node by 0; that is the usual `min(dist, D)` update with the
//!   constant `D` subtracted everywhere, which changes no reduced cost. So an iteration costs time
//!   proportional to the part of the graph it explored, not to the whole network.
//! * Searching from one excess node at a time, not from a super-source, is what keeps each
//!   iteration local; a super-source starts Dijkstra from every free row at once.

use std::cmp::Reverse;

use crate::error::{Error, Result};
use crate::flow::{MinHeap, Network, Residual};

const INF: i64 = i64::MAX;
const NONE: usize = usize::MAX;

/// Returns the flow on every arc and the node potentials.
pub(crate) fn solve(net: &Network) -> Result<(Vec<i64>, Vec<i64>)> {
    let n = net.nodes();
    let m = net.arcs().len();

    let mut excess: Vec<i64> = net.supplies().to_vec();
    let mut res = Residual::new(n, net.arcs());
    for (k, a) in net.arcs().iter().enumerate() {
        if a.cost < 0 {
            res.push(2 * k, a.cap);
            excess[a.from as usize] -= a.cap;
            excess[a.to as usize] += a.cap;
        }
    }

    let mut pot = vec![0i64; n];
    let mut dist = vec![INF; n];
    let mut prev = vec![NONE; n];
    let mut done = vec![false; n];
    let mut touched: Vec<usize> = Vec::new();
    let mut settled: Vec<usize> = Vec::new();
    let mut heap = MinHeap::new();
    let mut sources: Vec<usize> = (0..n).filter(|&v| excess[v] > 0).collect();

    while let Some(&s) = sources.last() {
        if excess[s] <= 0 {
            sources.pop();
            continue;
        }
        heap.clear();
        settled.clear();
        dist[s] = 0;
        touched.push(s);
        heap.push(Reverse((0, s as u32)));
        let mut sink = NONE;
        let mut sink_dist = 0;
        while let Some(Reverse((d, u))) = heap.pop() {
            let u = u as usize;
            if done[u] {
                continue;
            }
            done[u] = true;
            settled.push(u);
            if excess[u] < 0 {
                sink = u;
                sink_dist = d;
                break;
            }
            for &e in res.out(u) {
                let e = e as usize;
                if res.rescap[e] == 0 {
                    continue;
                }
                let v = res.to[e] as usize;
                if done[v] {
                    continue;
                }
                let nd = d + res.cost[e] + pot[u] - pot[v];
                if nd < dist[v] {
                    if dist[v] == INF {
                        touched.push(v);
                    }
                    dist[v] = nd;
                    prev[v] = e;
                    heap.push(Reverse((nd, v as u32)));
                }
            }
        }
        if sink == NONE {
            return Err(Error::Infeasible);
        }
        for &v in &settled {
            pot[v] += dist[v] - sink_dist;
        }
        let mut delta = excess[s].min(-excess[sink]);
        let mut v = sink;
        while v != s {
            let e = prev[v];
            delta = delta.min(res.rescap[e]);
            v = res.tail(e);
        }
        let mut v = sink;
        while v != s {
            let e = prev[v];
            res.push(e, delta);
            v = res.tail(e);
        }
        excess[s] -= delta;
        excess[sink] += delta;
        for &v in &touched {
            dist[v] = INF;
            done[v] = false;
        }
        touched.clear();
    }

    Ok(((0..m).map(|k| res.rescap[2 * k + 1]).collect(), pot))
}

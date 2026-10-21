//! Dinic's maximum flow. Used to decide whether a network's supplies can be routed before cost
//! scaling starts (push-relabel on an infeasible instance never terminates), and as an independent
//! check in the tests.

use crate::flow::{Arc, Network, Residual};

const NONE: u32 = u32::MAX;

/// Maximum `source -> sink` flow over `arcs` (costs ignored).
pub fn max_flow(nodes: usize, arcs: &[Arc], source: usize, sink: usize) -> i64 {
    if source == sink {
        return 0;
    }
    let mut res = Residual::new(nodes, arcs);
    let mut level = vec![NONE; nodes];
    let mut cur = vec![0usize; nodes];
    let mut queue: Vec<usize> = Vec::with_capacity(nodes);
    let mut path: Vec<usize> = Vec::new();
    let mut total = 0i64;
    loop {
        level.fill(NONE);
        level[source] = 0;
        queue.clear();
        queue.push(source);
        let mut head = 0;
        while head < queue.len() {
            let v = queue[head];
            head += 1;
            for &e in res.out(v) {
                let w = res.to[e as usize] as usize;
                if res.rescap[e as usize] > 0 && level[w] == NONE {
                    level[w] = level[v] + 1;
                    queue.push(w);
                }
            }
        }
        if level[sink] == NONE {
            return total;
        }
        cur.copy_from_slice(&res.adj_start[..nodes]);
        'blocking: loop {
            path.clear();
            let mut v = source;
            while v != sink {
                let end = res.adj_start[v + 1];
                let mut advanced = false;
                while cur[v] < end {
                    let e = res.adj[cur[v]] as usize;
                    let w = res.to[e] as usize;
                    if res.rescap[e] > 0 && level[w] == level[v] + 1 {
                        path.push(e);
                        v = w;
                        advanced = true;
                        break;
                    }
                    cur[v] += 1;
                }
                if !advanced {
                    if v == source {
                        break 'blocking;
                    }
                    level[v] = NONE; // dead end: never revisit in this phase
                    let e = path.pop().expect("a non-source node was entered by an arc");
                    v = res.tail(e);
                    cur[v] += 1;
                }
            }
            let bottleneck = path
                .iter()
                .map(|&e| res.rescap[e])
                .min()
                .expect("non-empty path");
            for &e in &path {
                res.push(e, bottleneck);
            }
            total += bottleneck;
        }
    }
}

/// Can every unit of supply reach a unit of demand through the arc capacities?
pub(crate) fn routable(net: &Network) -> bool {
    let n = net.nodes();
    let (s, t) = (n, n + 1);
    let mut arcs: Vec<Arc> = net.arcs().to_vec();
    let mut need = 0i64;
    for (v, &b) in net.supplies().iter().enumerate() {
        if b > 0 {
            arcs.push(Arc {
                from: s as u32,
                to: v as u32,
                cap: b,
                cost: 0,
            });
            need += b;
        } else if b < 0 {
            arcs.push(Arc {
                from: v as u32,
                to: t as u32,
                cap: -b,
                cost: 0,
            });
        }
    }
    max_flow(n + 2, &arcs, s, t) == need
}

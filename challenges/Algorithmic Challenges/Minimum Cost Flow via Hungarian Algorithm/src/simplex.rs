//! Primal network simplex (Dantzig 1951; Cunningham 1976 for the anti-cycling rule), the method
//! behind LEMON's `NetworkSimplex` and the exact earth mover's distance in POT (`ot.emd`).
//!
//! The basis is a spanning tree: tree arcs have reduced cost 0, every other arc sits at its lower
//! (0) or upper (capacity) bound. A pivot picks a non-tree arc whose reduced cost points the wrong
//! way, pushes flow around the cycle it closes with the tree until some arc blocks, and swaps the
//! blocking arc out of the tree. Entering arcs come from a block search (scan about `sqrt(m)` arcs
//! from where the last search stopped, take the best in the block). Cycling is impossible because
//! the tree is kept *strongly feasible* (Cunningham): among blocking arcs the one last met on the
//! cycle leaves.
//!
//! Start: every node hangs off an artificial root. A supply node `v` gets the free arc `v -> root`
//! carrying `b_v`; a demand node gets the arc `root -> v` at cost `M > n * max_cost` carrying
//! `-b_v`. The instance is feasible exactly when no artificial arc carries flow at the optimum.
//! The tree is stored as parent pointers plus linked child lists, so a pivot re-hangs only the
//! subtree that moves (re-rooting it along the path from the entering arc's endpoint).

use crate::error::{Error, Result};
use crate::flow::Network;

const NONE: usize = usize::MAX;
const INF: i64 = i64::MAX / 2;
const LOWER: i8 = 1;
const UPPER: i8 = -1;
const TREE: i8 = 0;
/// `pred_dir[v]`: the tree arc of `v` points from `v` up to its parent.
const UP: i8 = 1;
const DOWN: i8 = -1;

struct Simplex {
    root: usize,
    real_arcs: usize,
    src: Vec<usize>,
    dst: Vec<usize>,
    cap: Vec<i64>,
    cost: Vec<i64>,
    flow: Vec<i64>,
    state: Vec<i8>,
    parent: Vec<usize>,
    pred: Vec<usize>,
    pred_dir: Vec<i8>,
    depth: Vec<usize>,
    pi: Vec<i64>,
    first_child: Vec<usize>,
    next_sib: Vec<usize>,
    prev_sib: Vec<usize>,
    next_arc: usize,
    block: usize,
}

impl Simplex {
    fn new(net: &Network) -> Self {
        let n = net.nodes();
        let m = net.arcs().len();
        let root = n;
        let art_cost = n as i64 * net.max_abs_cost() + 1;
        let total = m + n;
        let mut s = Simplex {
            root,
            real_arcs: m,
            src: Vec::with_capacity(total),
            dst: Vec::with_capacity(total),
            cap: Vec::with_capacity(total),
            cost: Vec::with_capacity(total),
            flow: vec![0; total],
            state: vec![LOWER; total],
            parent: vec![NONE; n + 1],
            pred: vec![NONE; n + 1],
            pred_dir: vec![0; n + 1],
            depth: vec![0; n + 1],
            pi: vec![0; n + 1],
            first_child: vec![NONE; n + 1],
            next_sib: vec![NONE; n + 1],
            prev_sib: vec![NONE; n + 1],
            next_arc: 0,
            block: ((m as f64).sqrt() as usize).max(10),
        };
        for a in net.arcs() {
            s.src.push(a.from as usize);
            s.dst.push(a.to as usize);
            s.cap.push(a.cap);
            s.cost.push(a.cost);
        }
        for (v, &b) in net.supplies().iter().enumerate() {
            let e = m + v;
            if b >= 0 {
                s.src.push(v);
                s.dst.push(root);
                s.cost.push(0);
                s.pred_dir[v] = UP;
                s.pi[v] = 0;
                s.flow[e] = b;
            } else {
                s.src.push(root);
                s.dst.push(v);
                s.cost.push(art_cost);
                s.pred_dir[v] = DOWN;
                s.pi[v] = art_cost;
                s.flow[e] = -b;
            }
            s.cap.push(INF);
            s.state[e] = TREE;
            s.parent[v] = root;
            s.pred[v] = e;
            s.depth[v] = 1;
            s.link(root, v);
        }
        s
    }

    fn link(&mut self, p: usize, x: usize) {
        let f = self.first_child[p];
        self.next_sib[x] = f;
        self.prev_sib[x] = NONE;
        if f != NONE {
            self.prev_sib[f] = x;
        }
        self.first_child[p] = x;
    }

    fn unlink(&mut self, p: usize, x: usize) {
        let (pr, nx) = (self.prev_sib[x], self.next_sib[x]);
        if pr != NONE {
            self.next_sib[pr] = nx;
        } else {
            self.first_child[p] = nx;
        }
        if nx != NONE {
            self.prev_sib[nx] = pr;
        }
    }

    #[inline]
    fn reduced(&self, e: usize) -> i64 {
        self.cost[e] + self.pi[self.src[e]] - self.pi[self.dst[e]]
    }

    /// Block search over the real arcs; `None` when no arc can improve the objective.
    fn find_entering(&mut self) -> Option<usize> {
        let m = self.real_arcs;
        if m == 0 {
            return None;
        }
        let mut best = NONE;
        let mut best_val = 0i64;
        let mut count = 0usize;
        let mut e = self.next_arc;
        for _ in 0..m {
            let st = self.state[e];
            if st != TREE && self.cap[e] > 0 {
                let c = st as i64 * self.reduced(e);
                if c < best_val {
                    best_val = c;
                    best = e;
                }
            }
            count += 1;
            e += 1;
            if e == m {
                e = 0;
            }
            if count == self.block {
                if best != NONE {
                    self.next_arc = e;
                    return Some(best);
                }
                count = 0;
            }
        }
        self.next_arc = e;
        (best != NONE).then_some(best)
    }

    fn find_join(&self, mut a: usize, mut b: usize) -> usize {
        while a != b {
            if self.depth[a] > self.depth[b] {
                a = self.parent[a];
            } else if self.depth[b] > self.depth[a] {
                b = self.parent[b];
            } else {
                a = self.parent[a];
                b = self.parent[b];
            }
        }
        a
    }

    /// Residual capacity of tree arc `pred[u]` in the direction the cycle traverses it.
    #[inline]
    fn residual(&self, u: usize, forward_when: i8) -> i64 {
        let a = self.pred[u];
        if self.pred_dir[u] == forward_when {
            self.cap[a] - self.flow[a]
        } else {
            self.flow[a]
        }
    }

    /// One pivot; returns the amount of flow it moved.
    fn pivot(&mut self, e: usize) -> Result<i64> {
        let (first, second) = if self.state[e] == LOWER {
            (self.src[e], self.dst[e])
        } else {
            (self.dst[e], self.src[e])
        };
        let join = self.find_join(first, second);

        // Leaving arc, with Cunningham's rule: strict `<` along the first path (met last on the
        // cycle), `<=` along the second.
        let mut delta = self.cap[e];
        let mut out: Option<(usize, u8)> = None;
        let mut u = first;
        while u != join {
            let d = self.residual(u, DOWN);
            if d < delta {
                delta = d;
                out = Some((u, 1));
            }
            u = self.parent[u];
        }
        let mut u = second;
        while u != join {
            let d = self.residual(u, UP);
            if d <= delta {
                delta = d;
                out = Some((u, 2));
            }
            u = self.parent[u];
        }
        if delta >= INF / 2 {
            return Err(Error::OutOfRange("unbounded negative cycle".into()));
        }

        if delta > 0 {
            let val = self.state[e] as i64 * delta;
            self.flow[e] += val;
            let mut u = self.src[e];
            while u != join {
                self.flow[self.pred[u]] -= self.pred_dir[u] as i64 * val;
                u = self.parent[u];
            }
            let mut u = self.dst[e];
            while u != join {
                self.flow[self.pred[u]] += self.pred_dir[u] as i64 * val;
                u = self.parent[u];
            }
        }

        let Some((u_out, side)) = out else {
            // The entering arc blocks itself: it just jumps to its other bound.
            self.state[e] = -self.state[e];
            return Ok(delta);
        };
        let leaving = self.pred[u_out];
        self.state[leaving] = if self.flow[leaving] == 0 {
            LOWER
        } else {
            UPPER
        };
        self.state[e] = TREE;
        let (u_in, v_in) = if side == 1 {
            (first, second)
        } else {
            (second, first)
        };
        self.rehang(e, u_in, v_in, u_out);
        Ok(delta)
    }

    /// Re-roots the subtree below `u_out` at `u_in` and hangs it from `v_in` through arc `e`.
    fn rehang(&mut self, e: usize, u_in: usize, v_in: usize, u_out: usize) {
        let mut node = u_in;
        let mut new_parent = v_in;
        let mut new_arc = e;
        let mut new_dir = if self.src[e] == u_in { UP } else { DOWN };
        loop {
            let old_parent = self.parent[node];
            let old_arc = self.pred[node];
            let old_dir = self.pred_dir[node];
            self.unlink(old_parent, node);
            self.parent[node] = new_parent;
            self.pred[node] = new_arc;
            self.pred_dir[node] = new_dir;
            self.link(new_parent, node);
            if node == u_out {
                break;
            }
            new_parent = node;
            new_arc = old_arc;
            new_dir = -old_dir;
            node = old_parent;
        }
        // Tree arcs have reduced cost 0, so the whole moved subtree shifts by one constant.
        let sigma = self.pi[v_in] - self.pi[u_in] - self.pred_dir[u_in] as i64 * self.cost[e];
        let mut stack = vec![u_in];
        while let Some(x) = stack.pop() {
            self.depth[x] = self.depth[self.parent[x]] + 1;
            self.pi[x] += sigma;
            let mut c = self.first_child[x];
            while c != NONE {
                stack.push(c);
                c = self.next_sib[c];
            }
        }
    }
}

/// Pivot counts of a run; a degenerate pivot moves no flow (delta = 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PivotStats {
    pub pivots: u64,
    pub degenerate: u64,
}

/// Returns the flow on every arc and the node potentials.
pub(crate) fn solve(net: &Network) -> Result<(Vec<i64>, Vec<i64>)> {
    solve_with_stats(net).map(|(flow, pi, _)| (flow, pi))
}

/// Like the plain solver, also returning how many pivots it took.
pub fn solve_with_stats(net: &Network) -> Result<(Vec<i64>, Vec<i64>, PivotStats)> {
    let n = net.nodes();
    let m = net.arcs().len();
    let mut s = Simplex::new(net);
    let mut stats = PivotStats {
        pivots: 0,
        degenerate: 0,
    };
    while let Some(e) = s.find_entering() {
        stats.pivots += 1;
        if s.pivot(e)? == 0 {
            stats.degenerate += 1;
        }
    }
    if s.flow[m..m + n].iter().any(|&x| x != 0) {
        return Err(Error::Infeasible);
    }
    let potential = s.pi[..n].to_vec();
    s.flow.truncate(m);
    debug_assert_eq!(s.root, n);
    Ok((s.flow, potential, stats))
}

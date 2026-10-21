//! The oracle: adjacency lists and a graph walk per operation, O(n) each.
//!
//! It shares no code with the real structures, so agreeing with it means something.

use crate::stats::{Stats, Sum};
use crate::{Forest, PathOps, SubtreeOps, Weighted};

pub struct NaiveForest {
    adj: Vec<Vec<usize>>,
    values: Vec<Option<i64>>,
}

impl NaiveForest {
    /// The vertices on the path `u..=v` in order, or `None` if they are not connected.
    pub fn path(&self, u: usize, v: usize) -> Option<Vec<usize>> {
        let mut parent = vec![usize::MAX; self.adj.len()];
        parent[u] = u;
        let mut stack = vec![u];
        while let Some(x) = stack.pop() {
            if x == v {
                break;
            }
            for &y in &self.adj[x] {
                if parent[y] == usize::MAX {
                    parent[y] = x;
                    stack.push(y);
                }
            }
        }
        if parent[v] == usize::MAX {
            return None;
        }
        let mut path = vec![v];
        while *path.last().unwrap() != u {
            path.push(parent[*path.last().unwrap()]);
        }
        path.reverse();
        Some(path)
    }

    /// Every vertex reachable from `v` without stepping onto `blocked`.
    fn reach(&self, v: usize, blocked: Option<usize>) -> Vec<usize> {
        let mut seen = vec![false; self.adj.len()];
        seen[v] = true;
        if let Some(b) = blocked {
            seen[b] = true;
        }
        let mut stack = vec![v];
        let mut out = Vec::new();
        while let Some(x) = stack.pop() {
            out.push(x);
            for &y in &self.adj[x] {
                if !seen[y] {
                    seen[y] = true;
                    stack.push(y);
                }
            }
        }
        out
    }

    fn sum_of(&self, vertices: &[usize]) -> Sum {
        vertices
            .iter()
            .fold(Stats::NEUTRAL, |acc, &x| {
                acc.combine(Stats::from_option(self.values[x]))
            })
            .to_sum()
    }

    /// Full aggregate over the part of the tree on `v`'s side of the edge `p - v`.
    pub fn subtree_stats(&self, p: usize, v: usize) -> Option<Stats> {
        if !self.adj[p].contains(&v) {
            return None;
        }
        Some(
            self.reach(v, Some(p))
                .iter()
                .fold(Stats::NEUTRAL, |acc, &x| {
                    acc.combine(Stats::from_option(self.values[x]))
                }),
        )
    }
}

impl Forest for NaiveForest {
    fn new(n: usize) -> Self {
        NaiveForest {
            adj: vec![Vec::new(); n],
            values: vec![Some(0); n],
        }
    }

    fn len(&self) -> usize {
        self.adj.len()
    }

    fn link(&mut self, u: usize, v: usize) -> bool {
        if u == v || self.path(u, v).is_some() {
            return false;
        }
        self.adj[u].push(v);
        self.adj[v].push(u);
        true
    }

    fn cut(&mut self, u: usize, v: usize) -> bool {
        if !self.adj[u].contains(&v) {
            return false;
        }
        self.adj[u].retain(|&x| x != v);
        self.adj[v].retain(|&x| x != u);
        true
    }

    fn connected(&mut self, u: usize, v: usize) -> bool {
        self.path(u, v).is_some()
    }
}

impl Weighted for NaiveForest {
    fn from_values(values: &[Option<i64>]) -> Self {
        NaiveForest {
            adj: vec![Vec::new(); values.len()],
            values: values.to_vec(),
        }
    }

    fn set(&mut self, v: usize, value: i64) {
        self.values[v] = Some(value);
    }

    fn get(&mut self, v: usize) -> Option<i64> {
        self.values[v]
    }
}

impl PathOps for NaiveForest {
    fn path_query(&mut self, u: usize, v: usize) -> Option<Stats> {
        let path = self.path(u, v)?;
        Some(path.iter().fold(Stats::NEUTRAL, |acc, &x| {
            acc.combine(Stats::from_option(self.values[x]))
        }))
    }

    fn path_add(&mut self, u: usize, v: usize, delta: i64) -> bool {
        let Some(path) = self.path(u, v) else {
            return false;
        };
        for x in path {
            if let Some(val) = &mut self.values[x] {
                *val += delta;
            }
        }
        true
    }
}

impl SubtreeOps for NaiveForest {
    fn component(&mut self, v: usize) -> Sum {
        self.sum_of(&self.reach(v, None))
    }

    fn subtree_query(&mut self, p: usize, v: usize) -> Option<Sum> {
        self.subtree_stats(p, v).map(Stats::to_sum)
    }
}

//! Offline fully dynamic connectivity on a general graph, by a link-cut tree.
//!
//! The graph changes by edge insertions and deletions and is queried for "are `u` and `v`
//! connected?". A link-cut tree only holds a forest, so the graph's cycles need a policy. The
//! offline trick: because every deletion time is known in advance, keep a **maximum spanning
//! forest keyed by deletion time**.
//!
//! * Each edge `i` becomes its own tree node `n + i` carrying the key `deletion_time << 32 | i`;
//!   graph vertices are neutral. The minimum over a path then names the edge that dies soonest.
//! * Inserting `(u, v)`: if `u` and `v` are in different trees, link through the edge node.
//!   Otherwise look at the soonest-dying edge on the tree path. If it dies *before* the new edge,
//!   swap them. If not, the new edge is a non-tree edge that will be dead before every tree edge
//!   it could ever replace, so it is dropped for good.
//! * Deleting a tree edge never needs a replacement search. The forest is always a maximum
//!   spanning forest of the live edges by deletion time, and the edge being deleted now has the
//!   earliest deletion time of all of them, so no live non-tree edge can cover it: such an edge
//!   was dropped because it dies no later than the soonest edge on its path. That is why this is
//!   O(log n) per event with no search, where an online structure (Holm, de Lichtenberg and
//!   Thorup) needs a hierarchy of Euler tour forests and O(log² n) amortized.
//!
//! The price is that the whole event list must be known up front.

use std::collections::{BTreeSet, HashMap};

use crate::shapes::Rng;
use crate::{Forest, PathLct, PathOps, Weighted};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    Add(usize, usize),
    Remove(usize, usize),
    Query(usize, usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventError {
    /// A vertex is `>= n`.
    OutOfRange {
        at: usize,
    },
    SelfLoop {
        at: usize,
    },
    /// The edge is already present.
    Duplicate {
        at: usize,
    },
    /// The edge is not present.
    Missing {
        at: usize,
    },
}

impl std::fmt::Display for EventError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            EventError::OutOfRange { at } => write!(f, "event {at}: vertex out of range"),
            EventError::SelfLoop { at } => write!(f, "event {at}: self-loops are not allowed"),
            EventError::Duplicate { at } => write!(f, "event {at}: edge is already present"),
            EventError::Missing { at } => write!(f, "event {at}: edge is not present"),
        }
    }
}

impl std::error::Error for EventError {}

/// Answers plus counters that show what the spanning forest did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Solution {
    /// One answer per `Query` event, in order.
    pub answers: Vec<bool>,
    /// Insertions that joined two trees.
    pub tree_inserts: usize,
    /// Insertions that evicted a sooner-dying tree edge.
    pub swaps: usize,
    /// Insertions dropped as non-tree edges.
    pub dropped: usize,
    /// Deletions of an edge that was in the forest.
    pub tree_deletes: usize,
}

fn norm(u: usize, v: usize) -> (usize, usize) {
    (u.min(v), u.max(v))
}

/// Check every event and find each edge's deletion time (its `Remove` event, or `events.len()`).
fn plan(n: usize, events: &[Event]) -> Result<Vec<usize>, EventError> {
    let mut live: HashMap<(usize, usize), usize> = HashMap::new();
    let mut deletion = Vec::new();
    for (at, &ev) in events.iter().enumerate() {
        let (Event::Add(u, v) | Event::Remove(u, v) | Event::Query(u, v)) = ev;
        if u >= n || v >= n {
            return Err(EventError::OutOfRange { at });
        }
        match ev {
            Event::Add(..) => {
                if u == v {
                    return Err(EventError::SelfLoop { at });
                }
                if live.insert(norm(u, v), deletion.len()).is_some() {
                    return Err(EventError::Duplicate { at });
                }
                deletion.push(events.len());
            }
            Event::Remove(..) => {
                let i = live.remove(&norm(u, v)).ok_or(EventError::Missing { at })?;
                deletion[i] = at;
            }
            Event::Query(..) => {}
        }
    }
    Ok(deletion)
}

/// Answer every `Query` with a link-cut tree over `n` vertices plus one node per edge.
pub fn solve_offline(n: usize, events: &[Event]) -> Result<Solution, EventError> {
    let deletion = plan(n, events)?;
    let m = deletion.len();
    let mut values = vec![None; n];
    values.extend(
        deletion
            .iter()
            .enumerate()
            .map(|(i, &d)| Some(((d as i64) << 32) | i as i64)),
    );
    let mut tree = PathLct::from_values(&values);
    let mut ends = vec![(0, 0); m];
    let mut in_forest = vec![false; m];
    let mut current: HashMap<(usize, usize), usize> = HashMap::new();
    let mut next = 0;
    let mut out = Solution::default();

    let attach = |tree: &mut PathLct, i: usize, (u, v): (usize, usize)| {
        assert!(tree.link(u, n + i) && tree.link(n + i, v));
    };
    let detach = |tree: &mut PathLct, i: usize, (u, v): (usize, usize)| {
        assert!(tree.cut(u, n + i) && tree.cut(n + i, v));
    };

    for &ev in events {
        match ev {
            Event::Add(u, v) => {
                let i = next;
                next += 1;
                ends[i] = (u, v);
                current.insert(norm(u, v), i);
                if !tree.connected(u, v) {
                    attach(&mut tree, i, (u, v));
                    in_forest[i] = true;
                    out.tree_inserts += 1;
                } else {
                    let soonest = tree.path_query(u, v).expect("connected").min;
                    let j = (soonest & 0xFFFF_FFFF) as usize;
                    if deletion[j] < deletion[i] {
                        detach(&mut tree, j, ends[j]);
                        in_forest[j] = false;
                        attach(&mut tree, i, (u, v));
                        in_forest[i] = true;
                        out.swaps += 1;
                    } else {
                        out.dropped += 1;
                    }
                }
            }
            Event::Remove(u, v) => {
                let i = current.remove(&norm(u, v)).expect("validated by plan");
                if in_forest[i] {
                    detach(&mut tree, i, ends[i]);
                    in_forest[i] = false;
                    out.tree_deletes += 1;
                }
            }
            Event::Query(u, v) => out.answers.push(tree.connected(u, v)),
        }
    }
    Ok(out)
}

/// The slow reference: rebuild a union-find over the live edges for every query.
pub fn solve_rebuild(n: usize, events: &[Event]) -> Result<Vec<bool>, EventError> {
    plan(n, events)?;
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut live: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut answers = Vec::new();
    for &ev in events {
        match ev {
            Event::Add(u, v) => {
                live.insert(norm(u, v));
            }
            Event::Remove(u, v) => {
                live.remove(&norm(u, v));
            }
            Event::Query(u, v) => {
                let mut parent: Vec<usize> = (0..n).collect();
                for &(a, b) in &live {
                    let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                    parent[ra] = rb;
                }
                answers.push(find(&mut parent, u) == find(&mut parent, v));
            }
        }
    }
    Ok(answers)
}

/// Whether to add an edge: likely below the target edge count, unlikely above it.
fn add_bias(live: usize, target: usize, roll: usize) -> bool {
    if live < target {
        roll % 10 < 7
    } else {
        roll % 10 < 3
    }
}

/// A valid random event list: the graph hovers around `target_edges` live edges, and
/// `query_percent` of the events are queries.
pub fn random_events(
    n: usize,
    target_edges: usize,
    count: usize,
    query_percent: usize,
    seed: u64,
) -> Vec<Event> {
    assert!(n >= 2);
    let mut rng = Rng::new(seed);
    let mut live: HashMap<(usize, usize), usize> = HashMap::new();
    let mut list: Vec<(usize, usize)> = Vec::new();
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let (u, v) = (rng.below(n), rng.below(n));
        let roll = rng.below(100);
        if roll < query_percent {
            out.push(Event::Query(u, v));
        } else if list.is_empty() || add_bias(list.len(), target_edges, roll) {
            if u != v && !live.contains_key(&norm(u, v)) {
                live.insert(norm(u, v), list.len());
                list.push(norm(u, v));
                out.push(Event::Add(u, v));
            }
        } else {
            let k = rng.below(list.len());
            let e = list.swap_remove(k);
            live.remove(&e);
            if k < list.len() {
                live.insert(list[k], k);
            }
            out.push(Event::Remove(e.0, e.1));
        }
    }
    out
}

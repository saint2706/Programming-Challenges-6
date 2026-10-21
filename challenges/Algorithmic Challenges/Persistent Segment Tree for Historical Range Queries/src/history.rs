//! A time-travel table: values over positions `0..n` that change at timestamps, queryable as of
//! any time still retained.
//!
//! Wraps a [`Persistent`] tree with the one thing it lacks, a clock. Timestamps are `u64`, appended
//! in non-decreasing order; [`History::version_at`] finds the version in effect at a time with a
//! binary search, and [`History::forget_before`] retires everything older, which with reference
//! counting on frees that history's nodes immediately. This is the shape of an MVCC reader view or
//! a "what did this range look like last Tuesday" query.

use crate::arena::Version;
use crate::persistent::{Monoid, Persistent};

pub struct History<M: Monoid> {
    tree: Persistent<M>,
    /// `(time, version)` in time order; the first entry is the empty initial state at time 0.
    log: Vec<(u64, Version)>,
    /// Entries before this index have been forgotten.
    start: usize,
}

impl<M: Monoid> History<M> {
    /// `n` positions, all the identity, in effect from time 0.
    pub fn new(n: usize) -> Self {
        let tree = Persistent::new(n, true);
        let first = tree.first();
        Self {
            tree,
            log: vec![(0, first)],
            start: 0,
        }
    }

    /// Sets `pos` to `value` at `time`, which must not be earlier than the last change. Several
    /// changes may share a timestamp; a query at that time sees all of them.
    pub fn set(&mut self, time: u64, pos: usize, value: M::T) -> Version {
        let (last_time, last) = *self.log.last().expect("the log is never empty");
        assert!(
            time >= last_time,
            "time went backwards: {time} after {last_time}"
        );
        let v = self.tree.update(last, pos, value);
        self.log.push((time, v));
        v
    }

    /// The version in effect at `time`, or `None` if `time` predates what is still retained.
    pub fn version_at(&self, time: u64) -> Option<Version> {
        let live = &self.log[self.start..];
        let idx = live.partition_point(|&(t, _)| t <= time);
        idx.checked_sub(1).map(|i| live[i].1)
    }

    /// The value at `pos` as of `time`.
    pub fn get_at(&self, time: u64, pos: usize) -> Option<M::T> {
        self.version_at(time).map(|v| self.tree.get(v, pos))
    }

    /// The combined value over `[l, r)` as of `time`.
    pub fn query_at(&self, time: u64, l: usize, r: usize) -> Option<M::T> {
        self.version_at(time).map(|v| self.tree.query(v, l, r))
    }

    /// Retires every version that no query at `time` or later can reach: the version in effect at
    /// `time` and everything after it stay. Returns how many versions were retired.
    pub fn forget_before(&mut self, time: u64) -> usize {
        let Some(keep) = self.version_at(time) else {
            return 0;
        };
        let keep_at = self.log[self.start..]
            .iter()
            .position(|&(_, v)| v == keep)
            .expect("version_at returns a retained version")
            + self.start;
        for &(_, v) in &self.log[self.start..keep_at] {
            self.tree.retire(v);
        }
        let retired = keep_at - self.start;
        self.start = keep_at;
        retired
    }

    /// Time of the most recent change (0 before any).
    pub fn latest_time(&self) -> u64 {
        self.log.last().map_or(0, |&(t, _)| t)
    }

    /// Versions still retained.
    pub fn retained(&self) -> usize {
        self.log.len() - self.start
    }

    pub fn compact(&mut self) {
        self.tree.compact();
    }

    pub fn tree(&self) -> &Persistent<M> {
        &self.tree
    }
}

//! Reference points: what you get *without* an order-statistic tree.

use std::collections::BTreeMap;

use crate::OrderStatistics;

/// Sorted `Vec`: O(log n) rank/select, O(n) insert/remove (memmove).
#[derive(Clone, Debug, Default)]
pub struct SortedVec<K>(Vec<K>);

impl<K> SortedVec<K> {
    pub fn new() -> Self {
        Self(Vec::new())
    }
}

impl<K: Ord + Clone> OrderStatistics<K> for SortedVec<K> {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn insert(&mut self, key: K) {
        let pos = self.0.partition_point(|k| *k <= key);
        self.0.insert(pos, key);
    }
    fn remove(&mut self, key: &K) -> bool {
        let pos = self.0.partition_point(|k| k < key);
        if self.0.get(pos) == Some(key) {
            self.0.remove(pos);
            true
        } else {
            false
        }
    }
    fn rank(&self, key: &K) -> usize {
        self.0.partition_point(|k| k < key)
    }
    fn select(&self, idx: usize) -> Option<K> {
        self.0.get(idx).cloned()
    }
    fn validate(&self) -> Result<(), String> {
        if self.0.windows(2).all(|w| w[0] <= w[1]) {
            Ok(())
        } else {
            Err("not sorted".into())
        }
    }
}

/// Fenwick (binary indexed) tree over the fixed universe `0..universe`.
///
/// The classic competitive-programming answer: O(log U) everything with tiny
/// constants, but keys must be small integers known up front.
#[derive(Clone, Debug)]
pub struct Fenwick {
    tree: Vec<u32>,
    len: usize,
}

impl Fenwick {
    pub fn new(universe: usize) -> Self {
        Self {
            tree: vec![0; universe + 1],
            len: 0,
        }
    }

    fn universe(&self) -> usize {
        self.tree.len() - 1
    }

    fn add(&mut self, key: usize, delta: i32) {
        let mut i = key + 1;
        while i < self.tree.len() {
            self.tree[i] = self.tree[i].wrapping_add_signed(delta);
            i += i & i.wrapping_neg();
        }
    }

    /// Sum of counts over keys `< key`.
    fn prefix(&self, key: usize) -> usize {
        let (mut i, mut s) = (key.min(self.universe()), 0);
        while i > 0 {
            s += self.tree[i] as usize;
            i &= i - 1;
        }
        s
    }
}

impl OrderStatistics<usize> for Fenwick {
    fn len(&self) -> usize {
        self.len
    }
    fn insert(&mut self, key: usize) {
        assert!(key < self.universe(), "key outside Fenwick universe");
        self.add(key, 1);
        self.len += 1;
    }
    fn remove(&mut self, key: &usize) -> bool {
        if *key >= self.universe() || self.prefix(key + 1) == self.prefix(*key) {
            return false;
        }
        self.add(*key, -1);
        self.len -= 1;
        true
    }
    fn rank(&self, key: &usize) -> usize {
        self.prefix(*key)
    }
    fn select(&self, idx: usize) -> Option<usize> {
        if idx >= self.len {
            return None;
        }
        // Binary lifting: largest position whose prefix sum is <= idx.
        let (mut pos, mut rem) = (0, idx);
        let mut step = self.universe().next_power_of_two();
        while step > 0 {
            let next = pos + step;
            if next <= self.universe() && self.tree[next] as usize <= rem {
                pos = next;
                rem -= self.tree[next] as usize;
            }
            step >>= 1;
        }
        Some(pos)
    }
}

/// `std::collections::BTreeMap<key, multiplicity>` with rank/select done the
/// only way std allows: a linear walk. Exists to show *why* std's B-tree
/// cannot substitute for an order-statistic tree.
#[derive(Clone, Debug, Default)]
pub struct NaiveBTreeMap<K> {
    map: BTreeMap<K, usize>,
    len: usize,
}

impl<K> NaiveBTreeMap<K> {
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
            len: 0,
        }
    }
}

impl<K: Ord + Clone> OrderStatistics<K> for NaiveBTreeMap<K> {
    fn len(&self) -> usize {
        self.len
    }
    fn insert(&mut self, key: K) {
        *self.map.entry(key).or_insert(0) += 1;
        self.len += 1;
    }
    fn remove(&mut self, key: &K) -> bool {
        match self.map.get_mut(key) {
            Some(c) if *c > 1 => *c -= 1,
            Some(_) => {
                self.map.remove(key);
            }
            None => return false,
        }
        self.len -= 1;
        true
    }
    fn rank(&self, key: &K) -> usize {
        self.map.range(..key).map(|(_, c)| c).sum()
    }
    fn select(&self, idx: usize) -> Option<K> {
        let mut rem = idx;
        for (k, &c) in &self.map {
            if rem < c {
                return Some(k.clone());
            }
            rem -= c;
        }
        None
    }
}

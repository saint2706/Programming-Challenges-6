//! The brute-force reference every structure is tested against: a full copy of the array for every
//! version, O(n) per update and per query, no cleverness to get wrong.

/// One `Vec` per version.
#[derive(Clone, Debug)]
pub struct Naive<T: Copy> {
    versions: Vec<Option<Vec<T>>>,
}

impl<T: Copy> Naive<T> {
    pub fn new(values: Vec<T>) -> Self {
        Self {
            versions: vec![Some(values)],
        }
    }

    pub fn versions(&self) -> usize {
        self.versions.len()
    }

    pub fn is_live(&self, v: usize) -> bool {
        self.versions[v].is_some()
    }

    pub fn array(&self, v: usize) -> &[T] {
        self.versions[v].as_deref().expect("retired version")
    }

    /// New version: `from` with `pos` replaced.
    pub fn update(&mut self, from: usize, pos: usize, value: T) -> usize {
        let mut a = self.array(from).to_vec();
        a[pos] = value;
        self.versions.push(Some(a));
        self.versions.len() - 1
    }

    /// New version: `from` with `f` applied to every element of `l..r`.
    pub fn map_range(&mut self, from: usize, l: usize, r: usize, f: impl Fn(T) -> T) -> usize {
        let mut a = self.array(from).to_vec();
        for x in &mut a[l..r] {
            *x = f(*x);
        }
        self.versions.push(Some(a));
        self.versions.len() - 1
    }

    pub fn retire(&mut self, v: usize) {
        self.versions[v] = None;
    }

    /// Fold of `l..r` in version `v`.
    pub fn fold(&self, v: usize, l: usize, r: usize, init: T, f: impl Fn(T, T) -> T) -> T {
        self.array(v)[l..r].iter().fold(init, |acc, &x| f(acc, x))
    }
}

/// The `k`-th smallest (0-based) of `a[l..r]` by sorting a copy.
pub fn kth_by_sorting(a: &[i64], l: usize, r: usize, k: usize) -> Option<i64> {
    let mut s = a[l..r].to_vec();
    s.sort_unstable();
    s.get(k).copied()
}

/// Distinct values of `a[l..r]` by hashing.
pub fn distinct_by_hashing(a: &[i64], l: usize, r: usize) -> usize {
    a[l..r]
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len()
}

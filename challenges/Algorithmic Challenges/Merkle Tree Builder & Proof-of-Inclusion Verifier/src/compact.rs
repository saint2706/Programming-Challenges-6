//! Append-only tree that stores only its *frontier*.
//!
//! The RFC tree over `n` leaves is a right-fold of perfect subtrees, one per set bit of
//! `n` (largest first). Keeping just those O(log n) roots is enough to append a leaf and to
//! compute the current root -- the same idea as a compact range (Trillian) or a Merkle
//! mountain range. Appending is a binary-counter increment: merge equal-sized peaks.

use crate::hasher::{Hash, Hasher};
use crate::tree::ProofError;
use std::marker::PhantomData;

pub struct CompactTree<H: Hasher> {
    size: usize,
    /// Perfect-subtree roots, one per set bit of `size`, largest subtree first.
    frontier: Vec<Hash>,
    _hasher: PhantomData<fn() -> H>,
}

impl<H: Hasher> Default for CompactTree<H> {
    fn default() -> Self {
        Self::new()
    }
}

impl<H: Hasher> CompactTree<H> {
    pub fn new() -> Self {
        Self {
            size: 0,
            frontier: Vec::new(),
            _hasher: PhantomData,
        }
    }

    /// Resume from a stored frontier (e.g. from a log checkpoint). Rejects a frontier whose
    /// length is not `size.count_ones()`.
    pub fn from_frontier(size: usize, frontier: Vec<Hash>) -> Result<Self, ProofError> {
        if frontier.len() != size.count_ones() as usize {
            return Err(ProofError::Mismatch);
        }
        Ok(Self {
            size,
            frontier,
            _hasher: PhantomData,
        })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn frontier(&self) -> &[Hash] {
        &self.frontier
    }

    pub fn append(&mut self, data: &[u8]) {
        self.append_leaf_hash(H::leaf(data));
    }

    pub fn append_leaf_hash(&mut self, leaf: Hash) {
        let mut h = leaf;
        let mut n = self.size;
        while n & 1 == 1 {
            let left = self.frontier.pop().expect("one peak per set bit");
            h = H::node(&left, &h);
            n >>= 1;
        }
        self.frontier.push(h);
        self.size += 1;
    }

    /// Current root: fold the peaks from the smallest up.
    pub fn root(&self) -> Hash {
        let mut it = self.frontier.iter().rev();
        let Some(&last) = it.next() else {
            return H::empty();
        };
        it.fold(last, |acc, left| H::node(left, &acc))
    }
}

//! Multi-leaf inclusion proofs.
//!
//! Proving `k` leaves with `k` separate audit paths repeats every sibling the paths share
//! near the root. A multi-proof walks the tree level by level from the requested leaves and
//! emits a sibling hash only when it is *not* derivable from the leaves being proven -- so
//! adjacent leaves cost nothing for each other and the top of the tree is sent once.
//! Hashes are listed level by level (bottom first), left to right within a level; the
//! verifier replays the same walk from `size` alone, so no positions are transmitted.

use crate::hasher::{Hash, Hasher};
use crate::tree::{MerkleTree, ProofError};

fn check_indices(
    indices: impl Iterator<Item = usize> + Clone,
    size: usize,
) -> Result<(), ProofError> {
    let mut prev: Option<usize> = None;
    let mut any = false;
    for i in indices {
        any = true;
        if i >= size {
            return Err(ProofError::IndexOutOfRange { index: i, size });
        }
        if prev.is_some_and(|p| p >= i) {
            return Err(ProofError::InvalidIndices);
        }
        prev = Some(i);
    }
    if any {
        Ok(())
    } else {
        Err(ProofError::InvalidIndices)
    }
}

impl<H: Hasher> MerkleTree<H> {
    /// Deduplicated inclusion proof for the leaves at `indices` (strictly increasing) in the
    /// current tree.
    pub fn multi_proof(&self, indices: &[usize]) -> Result<Vec<Hash>, ProofError> {
        check_indices(indices.iter().copied(), self.len())?;
        let mut cur: Vec<usize> = indices.to_vec();
        let mut proof = Vec::new();
        for level in &self.levels[..self.levels.len() - 1] {
            let mut next: Vec<usize> = Vec::with_capacity(cur.len());
            let mut i = 0;
            while i < cur.len() {
                let idx = cur[i];
                if idx & 1 == 0 && cur.get(i + 1) == Some(&(idx + 1)) {
                    i += 1; // sibling is also being proven
                } else if let Some(sib) = level.get(idx ^ 1) {
                    proof.push(*sib);
                } // else: unpaired last node, promoted with no sibling
                if next.last() != Some(&(idx >> 1)) {
                    next.push(idx >> 1);
                }
                i += 1;
            }
            cur = next;
        }
        Ok(proof)
    }
}

/// Verify `leaves` (`(index, leaf_hash)`, indices strictly increasing) against `root` for a
/// tree of `size` leaves.
pub fn verify_multi<H: Hasher>(
    leaves: &[(usize, Hash)],
    size: usize,
    proof: &[Hash],
    root: &Hash,
) -> Result<(), ProofError> {
    check_indices(leaves.iter().map(|(i, _)| *i), size)?;
    let mut cur: Vec<(usize, Hash)> = leaves.to_vec();
    let mut supplied = proof.iter();
    let mut width = size;
    while width > 1 {
        let mut next: Vec<(usize, Hash)> = Vec::with_capacity(cur.len());
        let mut i = 0;
        while i < cur.len() {
            let (idx, h) = cur[i];
            let parent = if idx & 1 == 0 && cur.get(i + 1).is_some_and(|c| c.0 == idx + 1) {
                i += 1;
                H::node(&h, &cur[i].1)
            } else if idx ^ 1 >= width {
                h // promoted
            } else {
                let sib = supplied.next().ok_or(ProofError::Mismatch)?;
                if idx & 1 == 0 {
                    H::node(&h, sib)
                } else {
                    H::node(sib, &h)
                }
            };
            next.push((idx >> 1, parent));
            i += 1;
        }
        cur = next;
        width = width.div_ceil(2);
    }
    if supplied.next().is_none() && cur.len() == 1 && cur[0].1 == *root {
        Ok(())
    } else {
        Err(ProofError::Mismatch)
    }
}

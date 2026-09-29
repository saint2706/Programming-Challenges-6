//! Stateless proof verifiers, transcribed from RFC 9162 sections 2.1.3.2 and 2.1.4.2.
//!
//! They use the RFC's iterative `fn`/`sn` bit-shifting formulation: `fn` is the index of
//! the node being folded at the current level, `sn` the index of the last node at that
//! level. A node is a *right* child when `fn` is odd, or when `fn == sn` (the promoted
//! unpaired node -- its "sibling" in the proof then is a left subtree hanging off it).

use crate::hasher::{Hash, Hasher};
use crate::tree::ProofError;

/// Verify that `leaf_hash` is leaf `index` of a tree of `size` leaves with root `root`.
pub fn verify_inclusion<H: Hasher>(
    leaf_hash: &Hash,
    index: usize,
    size: usize,
    proof: &[Hash],
    root: &Hash,
) -> Result<(), ProofError> {
    if index >= size {
        return Err(ProofError::IndexOutOfRange { index, size });
    }
    let (mut f, mut s) = (index, size - 1);
    let mut r = *leaf_hash;
    for p in proof {
        if s == 0 {
            return Err(ProofError::Mismatch);
        }
        if f & 1 == 1 || f == s {
            r = H::node(p, &r);
            if f & 1 == 0 {
                while f & 1 == 0 && f != 0 {
                    f >>= 1;
                    s >>= 1;
                }
            }
        } else {
            r = H::node(&r, p);
        }
        f >>= 1;
        s >>= 1;
    }
    if s == 0 && r == *root {
        Ok(())
    } else {
        Err(ProofError::Mismatch)
    }
}

/// [`verify_inclusion`] for raw leaf data.
pub fn verify_inclusion_data<H: Hasher>(
    data: &[u8],
    index: usize,
    size: usize,
    proof: &[Hash],
    root: &Hash,
) -> Result<(), ProofError> {
    verify_inclusion::<H>(&H::leaf(data), index, size, proof, root)
}

/// Verify that the tree of `second` leaves (root `second_root`) is an append-only extension
/// of the tree of `first` leaves (root `first_root`).
pub fn verify_consistency<H: Hasher>(
    first: usize,
    second: usize,
    first_root: &Hash,
    second_root: &Hash,
    proof: &[Hash],
) -> Result<(), ProofError> {
    if first == 0 || first > second {
        return Err(ProofError::InvalidSizes { first, second });
    }
    if first == second {
        return if proof.is_empty() && first_root == second_root {
            Ok(())
        } else {
            Err(ProofError::Mismatch)
        };
    }
    // When `first` is a power of two the old root is itself a node of the new tree and is
    // not repeated in the proof, so put it back at the front.
    let mut full: Vec<Hash> = Vec::with_capacity(proof.len() + 1);
    if first.is_power_of_two() {
        full.push(*first_root);
    }
    full.extend_from_slice(proof);
    let Some((seed, rest)) = full.split_first() else {
        return Err(ProofError::Mismatch);
    };

    let (mut f, mut s) = (first - 1, second - 1);
    while f & 1 == 1 {
        f >>= 1;
        s >>= 1;
    }
    let (mut fr, mut sr) = (*seed, *seed);
    for c in rest {
        if s == 0 {
            return Err(ProofError::Mismatch);
        }
        if f & 1 == 1 || f == s {
            fr = H::node(c, &fr);
            sr = H::node(c, &sr);
            if f & 1 == 0 {
                while f & 1 == 0 && f != 0 {
                    f >>= 1;
                    s >>= 1;
                }
            }
        } else {
            sr = H::node(&sr, c);
        }
        f >>= 1;
        s >>= 1;
    }
    if s == 0 && fr == *first_root && sr == *second_root {
        Ok(())
    } else {
        Err(ProofError::Mismatch)
    }
}

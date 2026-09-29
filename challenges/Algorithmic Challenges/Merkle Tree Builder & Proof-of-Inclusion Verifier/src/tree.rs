//! Batch-built Merkle tree with proof generation.
//!
//! The tree is stored bottom-up as levels: `levels[0]` are the leaf hashes and
//! `levels[l+1][i] = node(levels[l][2i], levels[l][2i+1])`; an unpaired last node is
//! *promoted* unchanged (never duplicated). That shape is exactly the RFC's recursive
//! "split at the largest power of two below n" tree, which is what makes every RFC subtree
//! hash a stored node or a short combination of stored nodes.

use crate::hasher::{Hash, Hasher};
use crate::split_point;
use rayon::prelude::*;
use std::fmt;
use std::marker::PhantomData;

/// Build strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Build {
    Sequential,
    /// rayon for both leaf hashing and every level wide enough to be worth splitting.
    Parallel,
}

/// Levels narrower than this are combined sequentially even in [`Build::Parallel`]: rayon's
/// dispatch costs more than a few dozen hashes.
const PAR_LEVEL_MIN: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofError {
    IndexOutOfRange {
        index: usize,
        size: usize,
    },
    SizeOutOfRange {
        size: usize,
        len: usize,
    },
    /// Consistency proofs need `0 < first <= second`.
    InvalidSizes {
        first: usize,
        second: usize,
    },
    /// Multi-proofs need at least one strictly-increasing index.
    InvalidIndices,
    /// The proof had the wrong length or did not hash to the expected root(s).
    Mismatch,
}

impl fmt::Display for ProofError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IndexOutOfRange { index, size } => {
                write!(f, "leaf index {index} out of range for tree size {size}")
            }
            Self::SizeOutOfRange { size, len } => {
                write!(f, "tree size {size} exceeds the {len} leaves present")
            }
            Self::InvalidSizes { first, second } => {
                write!(f, "invalid consistency sizes {first} -> {second}")
            }
            Self::InvalidIndices => write!(f, "indices must be non-empty and strictly increasing"),
            Self::Mismatch => write!(f, "proof does not match the expected root"),
        }
    }
}

impl std::error::Error for ProofError {}

pub struct MerkleTree<H: Hasher> {
    pub(crate) levels: Vec<Vec<Hash>>,
    _hasher: PhantomData<fn() -> H>,
}

impl<H: Hasher> MerkleTree<H> {
    /// Hash `leaves` (raw entries) and build the tree.
    pub fn build<T: AsRef<[u8]> + Sync>(leaves: &[T], mode: Build) -> Self {
        let level0: Vec<Hash> = match mode {
            Build::Sequential => leaves.iter().map(|l| H::leaf(l.as_ref())).collect(),
            Build::Parallel => leaves.par_iter().map(|l| H::leaf(l.as_ref())).collect(),
        };
        Self::from_leaf_hashes(level0, mode)
    }

    /// Build from already-hashed leaves.
    pub fn from_leaf_hashes(level0: Vec<Hash>, mode: Build) -> Self {
        let mut levels = vec![level0];
        while levels.last().is_some_and(|l| l.len() > 1) {
            let prev = levels.last().expect("non-empty");
            let next: Vec<Hash> = if mode == Build::Parallel && prev.len() >= PAR_LEVEL_MIN {
                prev.par_chunks(2).map(Self::combine).collect()
            } else {
                prev.chunks(2).map(Self::combine).collect()
            };
            levels.push(next);
        }
        Self {
            levels,
            _hasher: PhantomData,
        }
    }

    #[inline]
    fn combine(pair: &[Hash]) -> Hash {
        match pair {
            [l, r] => H::node(l, r),
            [only] => *only,
            _ => unreachable!("chunks(2) yields 1 or 2 items"),
        }
    }

    /// Number of leaves.
    pub fn len(&self) -> usize {
        self.levels[0].len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Hash of leaf `index`.
    pub fn leaf_hash(&self, index: usize) -> Option<Hash> {
        self.levels[0].get(index).copied()
    }

    /// Root over the first `size` leaves (`size <= len()`); `HASH("")` when `size == 0`.
    pub fn root_at(&self, size: usize) -> Result<Hash, ProofError> {
        if size > self.len() {
            return Err(ProofError::SizeOutOfRange {
                size,
                len: self.len(),
            });
        }
        Ok(if size == 0 {
            H::empty()
        } else {
            self.range_hash(0, size)
        })
    }

    /// Root over all leaves.
    pub fn root(&self) -> Hash {
        self.root_at(self.len()).expect("size == len")
    }

    /// RFC `MTH(D[s:e])`, `s < e <= len`. O(1) when the range is a stored node, else
    /// O(log n) by the RFC's own split.
    pub(crate) fn range_hash(&self, s: usize, e: usize) -> Hash {
        let size = e - s;
        let h = size.next_power_of_two().trailing_zeros() as usize;
        let block = 1usize << h;
        if s.is_multiple_of(block) && (e == s + block || e == self.len()) {
            return self.levels[h][s >> h];
        }
        let k = split_point(size);
        H::node(&self.range_hash(s, s + k), &self.range_hash(s + k, e))
    }

    /// RFC 9162 section 2.1.3.1 audit path for leaf `index` in the tree of the first `size`
    /// leaves, deepest sibling first.
    pub fn inclusion_proof(&self, index: usize, size: usize) -> Result<Vec<Hash>, ProofError> {
        if size > self.len() {
            return Err(ProofError::SizeOutOfRange {
                size,
                len: self.len(),
            });
        }
        if index >= size {
            return Err(ProofError::IndexOutOfRange { index, size });
        }
        let (mut s, mut e) = (0, size);
        let mut path = Vec::new();
        while e - s > 1 {
            let k = split_point(e - s);
            if index - s < k {
                path.push(self.range_hash(s + k, e));
                e = s + k;
            } else {
                path.push(self.range_hash(s, s + k));
                s += k;
            }
        }
        path.reverse();
        Ok(path)
    }

    /// RFC 9162 section 2.1.4.1 consistency proof between the trees of the first `first` and
    /// `second` leaves.
    pub fn consistency_proof(&self, first: usize, second: usize) -> Result<Vec<Hash>, ProofError> {
        if second > self.len() {
            return Err(ProofError::SizeOutOfRange {
                size: second,
                len: self.len(),
            });
        }
        if first == 0 || first > second {
            return Err(ProofError::InvalidSizes { first, second });
        }
        let mut out = Vec::new();
        self.subproof(first, 0, second, true, &mut out);
        Ok(out)
    }

    /// RFC `SUBPROOF(m, D[s:e], b)`.
    fn subproof(&self, m: usize, s: usize, e: usize, complete: bool, out: &mut Vec<Hash>) {
        let n = e - s;
        if m == n {
            if !complete {
                out.push(self.range_hash(s, e));
            }
            return;
        }
        let k = split_point(n);
        if m <= k {
            self.subproof(m, s, s + k, complete, out);
            out.push(self.range_hash(s + k, e));
        } else {
            self.subproof(m - k, s + k, e, false, out);
            out.push(self.range_hash(s, s + k));
        }
    }
}

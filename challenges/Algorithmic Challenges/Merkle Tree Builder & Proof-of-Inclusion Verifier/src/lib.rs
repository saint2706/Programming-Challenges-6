//! RFC 9162 (Certificate Transparency v2) Merkle trees.
//!
//! - [`hasher`]: the [`Hasher`] trait and the SHA-256 / BLAKE3 instantiations.
//! - [`insecure`]: a deliberately broken no-domain-separation hasher, used only to
//!   demonstrate the second-preimage attack.
//! - [`tree`]: [`MerkleTree`] -- batch build (sequential and rayon-parallel), inclusion
//!   and consistency proof generation for any tree size up to the current one.
//! - [`verify`]: stateless RFC 9162 section 2.1.3.2 / 2.1.4.2 verifiers.
//! - [`multi`]: deduplicated multi-leaf inclusion proofs.
//! - [`compact`]: an append-only tree keeping only an O(log n) frontier.
//! - [`reference`]: a naive recursive oracle transcribed from the RFC text, for tests.

pub mod compact;
pub mod hasher;
pub mod insecure;
pub mod multi;
pub mod reference;
pub mod tree;
pub mod verify;

pub use compact::CompactTree;
pub use hasher::{Blake3Hasher, Hash, Hasher, Sha256Hasher};
pub use tree::{Build, MerkleTree, ProofError};

/// Largest power of two strictly less than `n` (`n >= 2`), the RFC's split point `k`.
#[inline]
pub(crate) fn split_point(n: usize) -> usize {
    debug_assert!(n >= 2);
    1usize << (usize::BITS - 1 - (n - 1).leading_zeros())
}

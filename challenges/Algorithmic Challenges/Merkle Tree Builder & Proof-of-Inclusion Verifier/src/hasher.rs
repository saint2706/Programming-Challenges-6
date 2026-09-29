//! Hash-function abstraction with RFC 9162 domain separation.

use sha2::{Digest, Sha256};

/// A 256-bit tree hash.
pub type Hash = [u8; 32];

/// Prefix byte for leaf hashes (RFC 9162 section 2.1.1).
pub const LEAF_PREFIX: u8 = 0x00;
/// Prefix byte for interior-node hashes.
pub const NODE_PREFIX: u8 = 0x01;

/// The three primitives an RFC 9162 tree needs. All methods are associated functions so a
/// hasher is a zero-sized type parameter.
pub trait Hasher: 'static {
    /// Root of the empty tree: `HASH("")`.
    fn empty() -> Hash;
    /// `HASH(0x00 || data)`.
    fn leaf(data: &[u8]) -> Hash;
    /// `HASH(0x01 || left || right)`.
    fn node(left: &Hash, right: &Hash) -> Hash;
}

/// SHA-256, byte-for-byte compatible with RFC 6962 / RFC 9162 logs.
pub struct Sha256Hasher;

impl Hasher for Sha256Hasher {
    fn empty() -> Hash {
        Sha256::digest([]).into()
    }
    fn leaf(data: &[u8]) -> Hash {
        let mut h = Sha256::new();
        h.update([LEAF_PREFIX]);
        h.update(data);
        h.finalize().into()
    }
    fn node(left: &Hash, right: &Hash) -> Hash {
        let mut h = Sha256::new();
        h.update([NODE_PREFIX]);
        h.update(left);
        h.update(right);
        h.finalize().into()
    }
}

/// BLAKE3 with the same 0x00 / 0x01 domain separation. Not interoperable with CT logs, but
/// several times faster on SIMD-capable CPUs.
pub struct Blake3Hasher;

impl Hasher for Blake3Hasher {
    fn empty() -> Hash {
        *blake3::hash(&[]).as_bytes()
    }
    fn leaf(data: &[u8]) -> Hash {
        let mut h = blake3::Hasher::new();
        h.update(&[LEAF_PREFIX]);
        h.update(data);
        *h.finalize().as_bytes()
    }
    fn node(left: &Hash, right: &Hash) -> Hash {
        let mut h = blake3::Hasher::new();
        h.update(&[NODE_PREFIX]);
        h.update(left);
        h.update(right);
        *h.finalize().as_bytes()
    }
}

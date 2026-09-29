//! A deliberately broken hasher: **do not use**. It exists only so the tests can show the
//! second-preimage attack that RFC 9162's 0x00/0x01 prefixes prevent.

use crate::hasher::{Hash, Hasher};
use sha2::{Digest, Sha256};

/// SHA-256 with *no* leaf/node domain separation: `leaf(d) = H(d)`, `node(l, r) = H(l || r)`.
///
/// A 64-byte "leaf" `H(a) || H(b)` then hashes to exactly the interior node over `a` and
/// `b`, so an attacker can present a shorter tree whose leaves are interior nodes of the
/// real one and get the same root.
pub struct NoDomainSha256;

impl Hasher for NoDomainSha256 {
    fn empty() -> Hash {
        Sha256::digest([]).into()
    }
    fn leaf(data: &[u8]) -> Hash {
        Sha256::digest(data).into()
    }
    fn node(left: &Hash, right: &Hash) -> Hash {
        let mut h = Sha256::new();
        h.update(left);
        h.update(right);
        h.finalize().into()
    }
}

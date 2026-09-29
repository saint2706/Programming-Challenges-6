//! Naive recursive oracle, transcribed directly from RFC 9162 section 2.1 (the `MTH`,
//! `PATH` and `SUBPROOF` definitions). O(n log n) per proof and allocation-happy -- it
//! exists so the optimised tree can be checked against the spec's own text.

use crate::hasher::{Hash, Hasher};
use crate::split_point;

/// `MTH(D[n])`.
pub fn mth<H: Hasher, T: AsRef<[u8]>>(d: &[T]) -> Hash {
    match d.len() {
        0 => H::empty(),
        1 => H::leaf(d[0].as_ref()),
        n => {
            let k = split_point(n);
            H::node(&mth::<H, T>(&d[..k]), &mth::<H, T>(&d[k..]))
        }
    }
}

/// `PATH(m, D[n])`, deepest sibling first.
pub fn path<H: Hasher, T: AsRef<[u8]>>(m: usize, d: &[T]) -> Vec<Hash> {
    let n = d.len();
    if n <= 1 {
        return Vec::new();
    }
    let k = split_point(n);
    if m < k {
        let mut p = path::<H, T>(m, &d[..k]);
        p.push(mth::<H, T>(&d[k..]));
        p
    } else {
        let mut p = path::<H, T>(m - k, &d[k..]);
        p.push(mth::<H, T>(&d[..k]));
        p
    }
}

/// `PROOF(m, D[n]) = SUBPROOF(m, D[n], true)`.
pub fn consistency<H: Hasher, T: AsRef<[u8]>>(m: usize, d: &[T]) -> Vec<Hash> {
    subproof::<H, T>(m, d, true)
}

fn subproof<H: Hasher, T: AsRef<[u8]>>(m: usize, d: &[T], b: bool) -> Vec<Hash> {
    let n = d.len();
    if m == n {
        return if b { Vec::new() } else { vec![mth::<H, T>(d)] };
    }
    let k = split_point(n);
    if m <= k {
        let mut p = subproof::<H, T>(m, &d[..k], b);
        p.push(mth::<H, T>(&d[k..]));
        p
    } else {
        let mut p = subproof::<H, T>(m - k, &d[k..], false);
        p.push(mth::<H, T>(&d[..k]));
        p
    }
}

//! A multiply-and-rotate hasher (the "Fx" family). The keys here are already well-spread small
//! integers or short strings, and the default SipHash costs more than the aggregation it protects.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default, Clone, Copy)]
pub struct Fx(u64);

const SEED: u64 = 0x517c_c1b7_2722_0a95;

impl Fx {
    #[inline]
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for Fx {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let (chunks, rest) = bytes.as_chunks::<8>();
        for c in chunks {
            self.add(u64::from_le_bytes(*c));
        }
        if !rest.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(buf) ^ ((rest.len() as u64) << 56));
        }
    }

    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.add(n);
    }

    #[inline]
    fn write_u32(&mut self, n: u32) {
        self.add(n as u64);
    }

    #[inline]
    fn write_u8(&mut self, n: u8) {
        self.add(n as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

pub type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn behaves_like_a_hash_map() {
        let mut m: FxMap<&str, u32> = FxMap::default();
        for (i, w) in ["a", "ab", "abcdefgh", "abcdefghi", "", "abcdefgh\0"]
            .iter()
            .enumerate()
        {
            m.insert(w, i as u32);
        }
        assert_eq!(m.len(), 6);
        assert_eq!(m["abcdefghi"], 3);
        assert_eq!(m[""], 4);
        let mut t: FxMap<(u64, u64), u8> = FxMap::default();
        t.insert((1, 2), 1);
        t.insert((2, 1), 2);
        assert_eq!((t[&(1, 2)], t[&(2, 1)]), (1, 2));
    }

    #[test]
    fn different_lengths_of_zero_bytes_hash_differently() {
        let h = |b: &[u8]| {
            let mut f = Fx::default();
            f.write(b);
            f.finish()
        };
        assert_ne!(h(&[0]), h(&[0, 0]));
        assert_ne!(h(&[0; 8]), h(&[0; 9]));
    }
}

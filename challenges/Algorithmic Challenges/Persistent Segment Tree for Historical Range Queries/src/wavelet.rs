//! The wavelet matrix (Claude, Navarro and Ordonez 2015): the static answer to the same questions.
//!
//! If the array never changes, a persistent tree is the wrong tool for "k-th smallest in a
//! subarray". The wavelet matrix stores the array in `ceil(log2(max + 1))` levels of one bit per
//! element, so it takes about `n * bits` bits (plus a rank directory) where the persistent count
//! tree takes `n * (log2(sigma) + 1)` nodes of 12 bytes or more, and it answers in `bits` rank
//! operations over contiguous memory instead of `log2(sigma)` pointer hops.
//!
//! Level by level, from the most significant bit: write each element's bit into a bit vector,
//! then stably reorder the elements, all the zero-bit ones first and then all the one-bit ones. A
//! range `[l, r)` at one level maps to a range at the next by rank: zeros go to
//! `[rank0(l), rank0(r))`, ones to `[zeros + rank1(l), zeros + rank1(r))`. Walking `bits` levels
//! with that rule finds k-th smallest, counts values below a threshold, and counts values in a
//! range, all in O(bits).

/// One level: the bit vector and a rank directory (ones before each 64-bit word).
struct Level {
    words: Vec<u64>,
    ones_before: Vec<u32>,
    zeros: usize,
}

impl Level {
    fn new(bits: &[bool]) -> Self {
        let n = bits.len();
        // One extra word so that rank at position n is always in range.
        let mut words = vec![0u64; n / 64 + 1];
        for (i, &b) in bits.iter().enumerate() {
            if b {
                words[i / 64] |= 1 << (i % 64);
            }
        }
        let mut ones_before = Vec::with_capacity(words.len());
        let mut total = 0u32;
        for w in &words {
            ones_before.push(total);
            total += w.count_ones();
        }
        Self {
            words,
            ones_before,
            zeros: n - total as usize,
        }
    }

    /// Ones among the first `i` bits.
    #[inline]
    fn rank1(&self, i: usize) -> usize {
        let w = i / 64;
        let below = self.words[w] & ((1u64 << (i % 64)) - 1);
        self.ones_before[w] as usize + below.count_ones() as usize
    }

    #[inline]
    fn rank0(&self, i: usize) -> usize {
        i - self.rank1(i)
    }
}

pub struct WaveletMatrix {
    n: usize,
    bits: u32,
    levels: Vec<Level>,
}

impl WaveletMatrix {
    pub fn new(values: &[u32]) -> Self {
        let max = values.iter().copied().max().unwrap_or(0);
        let bits = (u32::BITS - max.leading_zeros()).max(1);
        let mut current: Vec<u32> = values.to_vec();
        let mut next: Vec<u32> = Vec::with_capacity(values.len());
        let mut levels = Vec::with_capacity(bits as usize);
        for level in 0..bits {
            let shift = bits - 1 - level;
            let bit_of = |v: u32| (v >> shift) & 1 == 1;
            let column: Vec<bool> = current.iter().map(|&v| bit_of(v)).collect();
            levels.push(Level::new(&column));
            next.clear();
            next.extend(current.iter().copied().filter(|&v| !bit_of(v)));
            next.extend(current.iter().copied().filter(|&v| bit_of(v)));
            std::mem::swap(&mut current, &mut next);
        }
        Self {
            n: values.len(),
            bits,
            levels,
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Bits per value, the number of levels.
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// The `k`-th smallest (0-based) of `a[l..r]`, or `None` if `k >= r - l`.
    pub fn kth_smallest(&self, mut l: usize, mut r: usize, mut k: usize) -> Option<u32> {
        assert!(l <= r && r <= self.n);
        if k >= r - l {
            return None;
        }
        let mut value = 0u32;
        for level in &self.levels {
            let (l0, r0) = (level.rank0(l), level.rank0(r));
            let zeros_here = r0 - l0;
            value <<= 1;
            if k < zeros_here {
                (l, r) = (l0, r0);
            } else {
                k -= zeros_here;
                value |= 1;
                (l, r) = (level.zeros + (l - l0), level.zeros + (r - r0));
            }
        }
        Some(value)
    }

    /// How many elements of `a[l..r]` are strictly below `x`.
    pub fn count_below(&self, mut l: usize, mut r: usize, x: u32) -> usize {
        assert!(l <= r && r <= self.n);
        if self.bits < u32::BITS && x >> self.bits != 0 {
            return r - l;
        }
        let mut total = 0;
        for (i, level) in self.levels.iter().enumerate() {
            let shift = self.bits - 1 - i as u32;
            let (l0, r0) = (level.rank0(l), level.rank0(r));
            if (x >> shift) & 1 == 1 {
                total += r0 - l0;
                (l, r) = (level.zeros + (l - l0), level.zeros + (r - r0));
            } else {
                (l, r) = (l0, r0);
            }
        }
        total
    }

    /// How many elements of `a[l..r]` lie in `[lo, hi)`.
    pub fn range_freq(&self, l: usize, r: usize, lo: u32, hi: u32) -> usize {
        if lo >= hi {
            return 0;
        }
        self.count_below(l, r, hi) - self.count_below(l, r, lo)
    }

    /// Heap bytes: bit vectors plus rank directories.
    pub fn bytes(&self) -> usize {
        self.levels
            .iter()
            .map(|l| l.words.capacity() * 8 + l.ones_before.capacity() * 4)
            .sum()
    }
}

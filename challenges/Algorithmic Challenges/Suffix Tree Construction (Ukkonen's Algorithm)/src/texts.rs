//! Synthetic texts for tests and benchmarks: the easy case, and the structured ones that stress
//! construction algorithms differently.

/// SplitMix64: tiny, fast, and good enough to draw symbols.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// `n` independent uniform symbols from the first `sigma` letters starting at `a`.
pub fn uniform(n: usize, sigma: usize, seed: u64) -> Vec<u8> {
    assert!((1..=26).contains(&sigma));
    let mut rng = Rng::new(seed);
    (0..n).map(|_| b'a' + rng.below(sigma) as u8).collect()
}

/// The first `n` symbols of the infinite Fibonacci word (`a -> ab`, `b -> a`): highly repetitive,
/// with a suffix tree of the maximum possible shape for its length.
pub fn fibonacci(n: usize) -> Vec<u8> {
    let (mut prev, mut cur) = (b"a".to_vec(), b"ab".to_vec());
    while cur.len() < n {
        let next = [cur.as_slice(), prev.as_slice()].concat();
        prev = cur;
        cur = next;
    }
    cur.truncate(n);
    cur
}

/// The Thue-Morse word: overlap-free, so no substring repeats too quickly.
pub fn thue_morse(n: usize) -> Vec<u8> {
    (0..n as u64)
        .map(|i| b'a' + (i.count_ones() % 2) as u8)
        .collect()
}

/// `n` copies of one symbol: the deepest tree, and the quadratic worst case of top-down builders.
pub fn run(n: usize) -> Vec<u8> {
    vec![b'a'; n]
}

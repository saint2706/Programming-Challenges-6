//! Suffix array construction by induced sorting (SA-IS, Nong, Zhang and Chan 2009).
//!
//! The suffix array is built in linear time without comparing suffixes: classify every position as
//! S-type (the suffix is smaller than the next one) or L-type; the *LMS* positions (an S after an
//! L) cut the text into substrings; sort those substrings by induction from a bucket placement of
//! the LMS positions; name them; recurse on the string of names if any name repeats; and finally
//! induce the order of every suffix from the sorted LMS suffixes.
//!
//! This is a straightforward, allocation-honest version: the reduced string and its suffix array
//! get their own vectors instead of being squeezed into the output array. [`libsais`] is much
//! faster; the benchmark shows by how much. Texts are slices of [`Symbol`]s with an *implicit*
//! sentinel that sorts below everything, and the result has one entry per text position (no entry
//! for the empty suffix).
//!
//! [`libsais`]: https://github.com/IlyaGrebnov/libsais

/// An alphabet symbol usable as a bucket index.
pub trait Symbol: Copy {
    fn index(self) -> usize;
}

impl Symbol for u8 {
    #[inline]
    fn index(self) -> usize {
        self as usize
    }
}

impl Symbol for u32 {
    #[inline]
    fn index(self) -> usize {
        self as usize
    }
}

const EMPTY: u32 = u32::MAX;

/// Suffix array of `text`, whose symbols are all below `k`. Panics if `text` has `u32::MAX` or
/// more symbols.
pub fn suffix_array<T: Symbol>(text: &[T], k: usize) -> Vec<u32> {
    assert!(text.len() < EMPTY as usize, "text too long for u32 indices");
    let mut sa = vec![EMPTY; text.len()];
    match text.len() {
        0 => {}
        1 => sa[0] = 0,
        _ => sais(text, &mut sa, k),
    }
    sa
}

/// Suffix array of a byte string (alphabet 256).
pub fn suffix_array_u8(text: &[u8]) -> Vec<u32> {
    suffix_array(text, 256)
}

fn bucket_sizes<T: Symbol>(t: &[T], k: usize) -> Vec<u32> {
    let mut sizes = vec![0u32; k];
    for &c in t {
        sizes[c.index()] += 1;
    }
    sizes
}

fn bucket_heads(sizes: &[u32], out: &mut [u32]) {
    let mut sum = 0;
    for (o, &s) in out.iter_mut().zip(sizes) {
        *o = sum;
        sum += s;
    }
}

fn bucket_tails(sizes: &[u32], out: &mut [u32]) {
    let mut sum = 0;
    for (o, &s) in out.iter_mut().zip(sizes) {
        sum += s;
        *o = sum;
    }
}

/// Induce the L-type suffixes from whatever is in `sa`, scanning left to right.
fn induce_l<T: Symbol>(t: &[T], sa: &mut [u32], stype: &[bool], sizes: &[u32], buf: &mut [u32]) {
    let n = t.len();
    bucket_heads(sizes, buf);
    // The implicit sentinel is the smallest suffix, and the suffix before it is L-type.
    let c = t[n - 1].index();
    sa[buf[c] as usize] = (n - 1) as u32;
    buf[c] += 1;
    for i in 0..n {
        let j = sa[i];
        if j != EMPTY && j > 0 && !stype[j as usize - 1] {
            let c = t[j as usize - 1].index();
            sa[buf[c] as usize] = j - 1;
            buf[c] += 1;
        }
    }
}

/// Induce the S-type suffixes, scanning right to left.
fn induce_s<T: Symbol>(t: &[T], sa: &mut [u32], stype: &[bool], sizes: &[u32], buf: &mut [u32]) {
    let n = t.len();
    bucket_tails(sizes, buf);
    for i in (0..n).rev() {
        let j = sa[i];
        if j != EMPTY && j > 0 && stype[j as usize - 1] {
            let c = t[j as usize - 1].index();
            buf[c] -= 1;
            sa[buf[c] as usize] = j - 1;
        }
    }
}

fn sais<T: Symbol>(t: &[T], sa: &mut [u32], k: usize) {
    let n = t.len();
    // stype[i]: the suffix at i is smaller than the one at i + 1. The last position is L-type
    // because the implicit sentinel after it is smaller.
    let mut stype = vec![false; n];
    for i in (0..n - 1).rev() {
        let (a, b) = (t[i].index(), t[i + 1].index());
        stype[i] = a < b || (a == b && stype[i + 1]);
    }
    let lms = |i: usize| i > 0 && stype[i] && !stype[i - 1];
    let sizes = bucket_sizes(t, k);
    let mut buf = vec![0u32; k];

    // Step 1: sort the LMS substrings by inducing from an arbitrary placement of the LMS positions.
    sa.fill(EMPTY);
    bucket_tails(&sizes, &mut buf);
    for (i, &sym) in t.iter().enumerate().skip(1) {
        if lms(i) {
            let c = sym.index();
            buf[c] -= 1;
            sa[buf[c] as usize] = i as u32;
        }
    }
    induce_l(t, sa, &stype, &sizes, &mut buf);
    induce_s(t, sa, &stype, &sizes, &mut buf);

    // Step 2: gather the sorted LMS substrings and name them.
    let mut m = 0;
    for i in 0..n {
        let p = sa[i];
        if p != EMPTY && lms(p as usize) {
            sa[m] = p;
            m += 1;
        }
    }
    sa[m..].fill(EMPTY);
    let mut names = 0u32;
    let mut prev = EMPTY;
    for i in 0..m {
        let pos = sa[i];
        if prev == EMPTY || !lms_equal(t, &stype, prev as usize, pos as usize) {
            names += 1;
            prev = pos;
        }
        // LMS positions are at least 2 apart, so pos / 2 is unique.
        sa[m + pos as usize / 2] = names - 1;
    }
    let mut reduced = Vec::with_capacity(m);
    reduced.extend(sa[m..].iter().copied().filter(|&x| x != EMPTY));
    debug_assert_eq!(reduced.len(), m);

    // Step 3: sort the LMS suffixes, recursively if names repeat.
    let sa1: Vec<u32> = if (names as usize) < m {
        let mut sa1 = vec![EMPTY; m];
        match m {
            1 => sa1[0] = 0,
            _ => sais(&reduced, &mut sa1, names as usize),
        }
        sa1
    } else {
        let mut sa1 = vec![0u32; m];
        for (i, &name) in reduced.iter().enumerate() {
            sa1[name as usize] = i as u32;
        }
        sa1
    };

    // Step 4: place the sorted LMS suffixes in their buckets and induce everything else.
    let positions: Vec<u32> = (1..n).filter(|&i| lms(i)).map(|i| i as u32).collect();
    debug_assert_eq!(positions.len(), m);
    sa.fill(EMPTY);
    bucket_tails(&sizes, &mut buf);
    for &r in sa1.iter().rev() {
        let p = positions[r as usize];
        let c = t[p as usize].index();
        buf[c] -= 1;
        sa[buf[c] as usize] = p;
    }
    induce_l(t, sa, &stype, &sizes, &mut buf);
    induce_s(t, sa, &stype, &sizes, &mut buf);
}

/// Whether the LMS substrings starting at `a` and `b` are identical (same symbols and types, up
/// to and including the next LMS position). A substring that runs into the sentinel is unique.
fn lms_equal<T: Symbol>(t: &[T], stype: &[bool], a: usize, b: usize) -> bool {
    let n = t.len();
    let lms = |i: usize| i > 0 && stype[i] && !stype[i - 1];
    for d in 0.. {
        let (x, y) = (a + d, b + d);
        if x == n || y == n {
            return false;
        }
        if t[x].index() != t[y].index() || stype[x] != stype[y] {
            return false;
        }
        if d > 0 && (lms(x) || lms(y)) {
            return lms(x) && lms(y);
        }
    }
    unreachable!()
}

/// The reference: sort the suffixes by comparison. O(n² log n) worst case; for tests.
pub fn suffix_array_naive<T: Symbol>(text: &[T]) -> Vec<u32> {
    let mut sa: Vec<u32> = (0..text.len() as u32).collect();
    sa.sort_by(|&a, &b| {
        let (x, y) = (&text[a as usize..], &text[b as usize..]);
        x.iter().map(|c| c.index()).cmp(y.iter().map(|c| c.index()))
    });
    sa
}

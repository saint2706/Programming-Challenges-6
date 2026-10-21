//! The text model shared by every builder.
//!
//! A suffix tree needs every suffix to end at a leaf, which is guaranteed by a terminator symbol
//! that occurs nowhere else. Rather than reserving a byte value (which would make a text
//! containing that byte unrepresentable), the terminator lives *outside* the byte range: symbols
//! are `u32`, and the terminator is a symbol smaller than every byte.
//!
//! * [`Bytes`] is one byte string. Symbol `0` is the terminator at position `len`; byte `b` is
//!   symbol `b + 1`. All 256 byte values are usable, including `0`.
//! * [`Words`] is several strings joined for a generalized suffix tree. String `j` is followed by
//!   its own separator, symbol `j`, so every separator is unique and smaller than every byte; the
//!   last separator is the terminator. Byte `b` is symbol `k + b`.
//!
//! Because the terminator is the smallest symbol, suffix order is the usual lexicographic order
//! with "a prefix sorts before its extensions", which is also what `libsais` and the standard
//! SA-IS formulation produce.

/// Read access to a terminated text. `sym(i)` is the symbol at `i`; the last symbol of a text is
/// its (unique) terminator.
pub trait Symbols {
    /// Number of symbols, terminator included.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn sym(&self, i: usize) -> u32;

    /// Every symbol is below this.
    fn alphabet(&self) -> u32;

    /// The symbol that the byte `b` of a pattern stands for.
    fn byte_sym(&self, b: u8) -> u32;

    /// Whether `sym` is a terminator or separator (it occurs once and is not a byte).
    fn is_terminator(&self, sym: u32) -> bool;
}

/// One byte string plus an implicit terminator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bytes {
    bytes: Vec<u8>,
}

impl Bytes {
    pub fn new(bytes: impl Into<Vec<u8>>) -> Self {
        Bytes {
            bytes: bytes.into(),
        }
    }

    /// The text without its terminator.
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }
}

impl Symbols for Bytes {
    #[inline]
    fn len(&self) -> usize {
        self.bytes.len() + 1
    }

    #[inline]
    fn sym(&self, i: usize) -> u32 {
        match self.bytes.get(i) {
            Some(&b) => u32::from(b) + 1,
            None => 0,
        }
    }

    fn alphabet(&self) -> u32 {
        257
    }

    #[inline]
    fn byte_sym(&self, b: u8) -> u32 {
        u32::from(b) + 1
    }

    #[inline]
    fn is_terminator(&self, sym: u32) -> bool {
        sym == 0
    }
}

/// Several byte strings, each followed by its own separator. See the module docs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Words {
    syms: Vec<u32>,
    /// Start of each string in `syms`, plus the total length at the end.
    starts: Vec<usize>,
}

impl Words {
    /// At least one string; the strings may be empty.
    pub fn new<S: AsRef<[u8]>>(strings: &[S]) -> Self {
        assert!(!strings.is_empty(), "need at least one string");
        let k = strings.len() as u32;
        let mut syms = Vec::new();
        let mut starts = Vec::with_capacity(strings.len() + 1);
        for (j, s) in strings.iter().enumerate() {
            starts.push(syms.len());
            syms.extend(s.as_ref().iter().map(|&b| k + u32::from(b)));
            syms.push(j as u32);
        }
        starts.push(syms.len());
        Words { syms, starts }
    }

    /// Number of strings.
    pub fn count(&self) -> usize {
        self.starts.len() - 1
    }

    /// Which string position `i` belongs to (a separator belongs to the string it ends).
    pub fn string_of(&self, i: usize) -> usize {
        self.starts.partition_point(|&s| s <= i) - 1
    }

    /// First position of string `j`.
    pub fn start_of(&self, j: usize) -> usize {
        self.starts[j]
    }

    /// The symbols as a slice, for algorithms that want random access.
    pub fn as_symbols(&self) -> &[u32] {
        &self.syms
    }

    /// Render symbols `range` back to bytes; separators become `b'#'` (they are not bytes).
    pub fn render(&self, range: std::ops::Range<usize>) -> Vec<u8> {
        let k = self.count() as u32;
        self.syms[range]
            .iter()
            .map(|&s| if s < k { b'#' } else { (s - k) as u8 })
            .collect()
    }
}

impl Symbols for Words {
    #[inline]
    fn len(&self) -> usize {
        self.syms.len()
    }

    #[inline]
    fn sym(&self, i: usize) -> u32 {
        self.syms[i]
    }

    fn alphabet(&self) -> u32 {
        self.count() as u32 + 256
    }

    #[inline]
    fn byte_sym(&self, b: u8) -> u32 {
        self.count() as u32 + u32::from(b)
    }

    #[inline]
    fn is_terminator(&self, sym: u32) -> bool {
        sym < self.count() as u32
    }
}

/// A prefix of a growing byte string without a terminator, for online construction.
pub(crate) struct Prefix<'a>(pub &'a [u8]);

impl Symbols for Prefix<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn sym(&self, i: usize) -> u32 {
        u32::from(self.0[i]) + 1
    }

    fn alphabet(&self) -> u32 {
        257
    }

    #[inline]
    fn byte_sym(&self, b: u8) -> u32 {
        u32::from(b) + 1
    }

    #[inline]
    fn is_terminator(&self, _sym: u32) -> bool {
        false
    }
}

/// The text of `symbols` as a plain `u32` vector.
pub fn to_vec<S: Symbols>(text: &S) -> Vec<u32> {
    (0..text.len()).map(|i| text.sym(i)).collect()
}

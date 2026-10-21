//! Suffix trees, built three ways and checked against suffix arrays.

#![forbid(unsafe_code)]

pub mod apps;
pub mod array_index;
pub mod lcp;
pub mod pwotd;
pub mod sais;
pub mod satree;
pub mod text;
pub mod texts;
pub mod tree;
pub mod ukkonen;

pub use text::{Bytes, Symbols, Words};
pub use tree::SuffixTree;

//! A pivot table and cross-tabulation engine written from scratch.

pub mod acc;
pub mod crosstab;
pub mod csv;
pub mod date;
pub mod drill;
pub mod encode;
pub mod error;
pub mod expr;
pub mod filter;
pub mod fxhash;
pub mod pivot;
pub mod spec;
pub mod special;
pub mod table;

pub use error::{Error, Result};

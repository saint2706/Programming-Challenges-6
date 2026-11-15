//! A small columnar storage engine and query executor.
//!
//! * [`format`]: the `.col` file format (row groups, per-chunk encodings, zone maps, footer).
//! * [`sql`]: tokenizer and recursive-descent parser for a `SELECT` subset.
//! * [`plan`]: logical plan, projection and predicate pushdown, zone-map pruning, `EXPLAIN`.
//! * [`exec`]: vectorized executor working on one row group (a [`Batch`]) at a time.

pub mod csv;
pub mod encoding;
pub mod exec;
pub mod format;
pub mod plan;
pub mod sql;
pub mod types;

pub use exec::{Catalog, QueryResult, query};
pub use types::{Batch, Column, DataType, Value};

/// Every fallible operation returns a plain message; the CLI prints it.
pub type Result<T> = std::result::Result<T, String>;

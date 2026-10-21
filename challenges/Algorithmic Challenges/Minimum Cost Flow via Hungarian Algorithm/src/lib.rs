//! Assignment and minimum-cost flow, solved eight ways and checked against each other.
//!
//! * Assignment ([`assign`]): the Hungarian algorithm, Jonker-Volgenant (Crouse's rectangular form
//!   and full LAPJV), a LAPMOD-style sparse solver, and an epsilon-scaling auction.
//! * Min-cost flow ([`flow`]): successive shortest paths, Goldberg-Tarjan cost scaling and network
//!   simplex; each also solves an assignment through [`flow::assignment_network`].
//! * [`verify`] proves results optimal without trusting the solver; [`instances`] makes the benchmark
//!   instance families; [`dimacs`] reads and writes the standard file formats.

#![forbid(unsafe_code)]

pub mod assign;
pub mod auction;
pub mod cost_scaling;
pub mod dimacs;
pub mod error;
pub mod flow;
pub mod hungarian;
pub mod instances;
pub mod jv;
pub mod lapmod;
pub mod matching;
pub mod matrix;
pub mod maxflow;
pub mod simplex;
pub mod ssp;
pub mod verify;

pub use assign::{Algorithm, Assignment, Duals, solve_dense, solve_sparse};
pub use error::{Error, Result};
pub use flow::{Flow, FlowAlgorithm, Network};
pub use matrix::{Cost, Dense, FORBIDDEN, Sparse};

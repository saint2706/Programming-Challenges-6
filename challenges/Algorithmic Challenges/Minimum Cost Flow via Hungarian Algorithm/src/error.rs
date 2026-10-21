//! One error type for every solver.

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// No assignment (no perfect matching of the smaller side) or no flow satisfies the input.
    Infeasible,
    /// Node supplies do not sum to zero.
    Unbalanced { sum: i64 },
    /// A value is outside the range for which the solvers' i64 arithmetic is proven exact.
    OutOfRange(String),
    /// Malformed input (bad index, wrong length, unparsable file).
    Invalid(String),
    /// The chosen algorithm cannot take this input (stated in the message).
    Unsupported(&'static str),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Infeasible => f.write_str("infeasible: no solution satisfies the constraints"),
            Error::Unbalanced { sum } => write!(f, "supplies sum to {sum}, not zero"),
            Error::OutOfRange(m) => write!(f, "out of range: {m}"),
            Error::Invalid(m) => write!(f, "invalid input: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
        }
    }
}

impl std::error::Error for Error {}

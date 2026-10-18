use std::fmt;

/// Every failure the engine reports: a message a person can act on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(String);

impl Error {
    pub fn new(message: impl Into<String>) -> Self {
        Error(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error(format!("bad request: {e}"))
    }
}

pub type Result<T> = std::result::Result<T, Error>;

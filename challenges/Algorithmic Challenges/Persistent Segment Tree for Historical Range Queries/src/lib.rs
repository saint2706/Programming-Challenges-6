//! Persistent segment trees for historical range queries, against the structures that compete
//! with them.
//!
//! * [`persistent`]: the path-copying tree, any version queryable in O(log n), versions that
//!   branch, and the k-th smallest descent over two roots.
//! * [`kth`]: range k-th smallest, range rank and range distinct-count indexes built on it.
//! * [`lazy`]: range add with permanent marks, and range add / assign with push-down.
//! * [`fat`]: the fat-node method, which is only partially persistent and saves no memory here.
//! * [`wavelet`]: a static wavelet matrix, the better answer when the array never changes.
//! * [`history`]: a timestamped table over the persistent tree, with retention.
//! * [`arena`]: node storage, reference counting and mark-and-compact collection for all of them.
//! * [`oracle`]: the brute-force reference the tests use.

#![forbid(unsafe_code)]

pub mod arena;
pub mod fat;
pub mod history;
pub mod kth;
pub mod lazy;
pub mod oracle;
pub mod persistent;
pub mod wavelet;

pub use arena::{NIL, Stats, Version};
pub use fat::FatTree;
pub use history::History;
pub use kth::{DistinctIndex, KthIndex};
pub use lazy::{LazyAdd, PushDown};
pub use persistent::{MaxI64, MinI64, Monoid, Persistent, SumI64, SumU32};
pub use wavelet::WaveletMatrix;

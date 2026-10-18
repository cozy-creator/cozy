//! varena: memory and transfer mechanisms used in the calling process.
//! In Cozy, TensorD coordinates machine-wide budgets and retention and uses TensorFS to
//! prepare shared CPU weight buffers. Runtime constructs models and drives GPU allocation,
//! copies, mapping and unmapping through the plane/device APIs within those grants.
//! TensorD can retain exported GPU allocation handles; retaining bytes does not construct
//! a model or perform its GPU operations. TensorFS is a library on both sides, not a daemon.
//!
//! A plane applies caller-supplied budgets, regions and priorities for fills, prefetch,
//! eviction and streaming. It does not measure free memory or reorder the caller's schedule.
//! `tensorfs-core` supplies verified serialized bytes and metadata and stays CUDA-free.
//!
//! The plane is a Unix mechanism: weight sets cross processes as descriptors. Its host tier
//! (sealed memfds claimed by OFD locks) is Linux-only and answers [`Error::Unsupported`]
//! elsewhere; a Mac's host tier is the page cache of the store's objects. On Windows the crate
//! is empty until its tier exists (xs-043 unit WM).
#![cfg(unix)]

pub mod cuda;
pub mod cursor;
pub mod device;
pub mod faults;
pub mod host;
pub mod io;
pub mod layout;
pub mod plane;
pub mod window;

pub use cursor::{Cursor, RingView};
pub use plane::{Exported, Lease, Plane, PlaneConfig, Stats, Ticket, Tier, Trim, View, WsId};

use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum Error {
    #[error("cuda driver: {op} failed with {name} ({code})")]
    Cuda {
        op: &'static str,
        code: i32,
        name: String,
    },
    #[error("cuda driver unavailable: {0}")]
    CudaUnavailable(String),
    #[error("budget exceeded in {pool}: requested {requested} bytes, {available} available")]
    BudgetExceeded {
        pool: &'static str,
        requested: u64,
        available: u64,
    },
    #[error(
        "shortfall in {pool}: need {need} bytes, {available} free; evicting lower priorities is \
         not enough, blocked by {blockers:?}"
    )]
    Shortfall {
        pool: &'static str,
        need: u64,
        available: u64,
        blockers: Vec<String>,
    },
    #[error("below floor in {pool}: need {need} bytes, the whole budget is {budget}")]
    BelowFloor { pool: &'static str, need: u64, budget: u64 },
    #[error("{op} refused: {live} live and {pending} pending lease(s) cover {what}")]
    LeaseViolation {
        op: &'static str,
        live: u64,
        pending: u64,
        what: String,
    },
    #[error("poisoned: {0}")]
    Poisoned(String),
    #[error("invalid: {0}")]
    Invalid(String),
    #[error("io: {0}")]
    Io(String),
    #[error("source: {0}")]
    Source(#[from] tensorfs_core::err::Refusal),
    /// What closed: a weight set, a cursor, a copy stream.
    #[error("closed: {0}")]
    Closed(String),
    /// This OS has no such mechanism (the host tier off Linux).
    #[error("unsupported: {0}")]
    Unsupported(String),
}

/// An OS seam's failure, named by what was attempted.
pub(crate) fn os(what: &'static str) -> impl FnOnce(cozy_platform::Error) -> Error {
    move |error| {
        if error.is_unsupported() {
            Error::Unsupported(format!("{what}: {error}"))
        } else {
            Error::Io(format!("{what}: {}", std::io::Error::from(error)))
        }
    }
}

impl Error {
    /// The stable machine name of the error class.
    pub fn code(&self) -> &'static str {
        match self {
            Error::Cuda { .. } => "CUDA",
            Error::CudaUnavailable(_) => "CUDA_UNAVAILABLE",
            Error::BudgetExceeded { .. } => "BUDGET_EXCEEDED",
            Error::Shortfall { .. } => "SHORTFALL",
            Error::BelowFloor { .. } => "BELOW_FLOOR",
            Error::LeaseViolation { .. } => "LEASE_VIOLATION",
            Error::Poisoned(_) => "POISONED",
            Error::Invalid(_) => "INVALID",
            Error::Io(_) => "IO",
            Error::Source(_) => "SOURCE",
            Error::Closed(_) => "CLOSED",
            Error::Unsupported(_) => "UNSUPPORTED",
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

//! tensorfs-plane: the weight plane. It owns pinned host memory and GPU memory and executes
//! fills, prefetch, eviction and streaming exactly as its one client (cozy-runtime's policy)
//! directs. It never measures free memory, never chooses among equals, never reorders a
//! schedule. `tensorfs-core` supplies verified bytes and stays CUDA-free.

pub mod cuda;
pub mod cursor;
pub mod device;
pub mod faults;
pub mod host;
pub mod io;
pub mod layout;
pub mod plane;

pub use cursor::{Cursor, RingView};
pub use plane::{Lease, Plane, PlaneConfig, Stats, Ticket, Tier, Trim, View, WsId};

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
    #[error("closed")]
    Closed,
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
            Error::Closed => "CLOSED",
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

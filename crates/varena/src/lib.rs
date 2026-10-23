//! varena: the memory of the processes computing with model weights (design `varena-leases.md`).
//!
//! [`lease`]'s manager, hosted by TensorD, allocates every GPU chunk and leases it to each model
//! process's agent; [`Memory`] is a process's memory on one device over a [`Backend`]
//! ([`CudaBackend`], or host pages): its weight pages, [`arena`]'s torch blocks and [`region`]'s
//! regions, all leased, or standalone within a budget when no manager serves the GPU. [`table`]
//! holds a process's facts and [`decide`] orders its own pages. Host RAM is the OS page cache:
//! TensorFS's `io` moves every byte.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
#![deny(unsafe_code)]

pub mod arena;
#[allow(unsafe_code)]
pub mod backend;
#[allow(unsafe_code)]
pub mod cuda;
pub mod decide;
#[cfg(unix)]
#[allow(unsafe_code)]
pub mod derived;
pub mod layout;
pub mod lease;
#[allow(unsafe_code)]
pub mod memory;
pub mod os;
pub mod region;
pub mod table;

pub use backend::{Access, Backend, Budget, Event, HostBackend, Physical, Va};
pub use cuda::CudaBackend;
pub use memory::{Allocating, Fills, MakeRoom, Memory, Moved, Origin, Pass, Pin, PoolId, Ref, Request, Retained, SetId, Stats, StepStats, TorchStats, Yielded};
pub use region::{GraphRegion, Kind as RegionKind, Reclaimable, Reclaimed, Record as RegionRecord, Region, RegionPin, Room, Word as GraphWord};
pub use table::{headroom, Fill, Fingerprint, Level, PartId, Plan, SetSpec, Step};

use thiserror::Error;

/// A device as the shared table names it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GpuId {
    /// The driver's UUID (`GPU-…`); a host-page device uses any stable name.
    pub uuid: String,
    /// The model's name (NVML): peaks are kept per GPU model.
    pub model: String,
    /// The card's total bytes (NVML), or a host-page device's limit.
    pub memory: u64,
}

/// `Exhausted` is the one memory error; it carries the request's record.
#[derive(Debug, Clone)]
pub struct Exhausted {
    pub need: u64,
    pub site: Option<String>,
    pub stats: Stats,
    /// What holds the device besides cozy, when known: the largest foreign process and its bytes
    /// (Linux, TensorD's sample), or this process's budget and usage (Windows).
    pub foreign: Option<String>,
}

#[derive(Debug, Clone, Error)]
pub enum Error {
    #[error("exhausted: {} bytes short{}{}", .0.need, .0.site.as_deref().map(|s| format!(" at {s}")).unwrap_or_default(), .0.foreign.as_deref().map(|f| format!(" (GPU memory in use by {f})")).unwrap_or_default())]
    Exhausted(Box<Exhausted>),
    #[error("device ownership is unproven: {0}")]
    Ownership(String),
    #[error("stale graph binding: {0}")]
    StaleBinding(String),
    #[error("invalid: {0}")]
    Invalid(String),
    /// A driver call failed; `code` is the driver's (CUresult), `name` its name.
    #[error("device: {op} failed with {name} ({code})")]
    Device { op: &'static str, code: i32, name: String },
    #[error("io: {0}")]
    Io(String),
    /// What closed: a set, the table, the custody channel.
    #[error("closed: {0}")]
    Closed(String),
    /// A panic or broken invariant inside varena: the request fails, the process keeps serving.
    #[error("internal: {0}")]
    Internal(String),
    /// A bind that needs a page only the calling thread pins (re-binding a page it is
    /// transforming, or an identity switch under its own hold): it would wait on itself.
    #[error("held: {0}")]
    Held(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// The stable machine name of the error class.
    pub fn code(&self) -> &'static str {
        match self {
            Error::Exhausted(_) => "EXHAUSTED",
            Error::Invalid(_) => "INVALID",
            Error::StaleBinding(_) => "STALE_BINDING",
            Error::Ownership(_) => "OWNERSHIP",
            Error::Device { .. } => "DEVICE",
            Error::Io(_) => "IO",
            Error::Closed(_) => "CLOSED",
            Error::Internal(_) => "INTERNAL",
            Error::Held(_) => "HELD",
        }
    }

    /// CUDA documents these as destroying the context: the process must end, and the GPU's kept
    /// pages go (cuda 13.0 driver API, CUresult).
    pub fn is_sticky(&self) -> bool {
        matches!(self, Error::Ownership(_) | Error::Device { code: 700 | 702 | 709 | 710 | 714..=719 | 721 | 911, .. })
    }

    pub(crate) fn io(what: &str, e: impl std::fmt::Display) -> Error {
        Error::Io(format!("{what}: {e}"))
    }

}

impl From<platform::Error> for Error {
    fn from(e: platform::Error) -> Error {
        Error::Io(std::io::Error::from(e).to_string())
    }
}

impl From<tensorfs::io::cuda::Error> for Error {
    fn from(e: tensorfs::io::cuda::Error) -> Error {
        match e {
            tensorfs::io::cuda::Error::Cuda { op, code, name } => Error::Device { op, code, name },
            tensorfs::io::cuda::Error::CudaUnavailable(d) => Error::Device { op: "load the driver", code: -1, name: d },
        }
    }
}

impl From<tensorfs::io::Error> for Error {
    fn from(e: tensorfs::io::Error) -> Error {
        match e.cuda().cloned() {
            Some(cuda) => cuda.into(),
            None => Error::Io(e.to_string()),
        }
    }
}

#[cfg(unix)]
impl From<tensorfs::io::source::Error> for Error {
    fn from(e: tensorfs::io::source::Error) -> Error {
        match e {
            tensorfs::io::source::Error::Read(e) => e.into(),
            tensorfs::io::source::Error::Source(e) => Error::Io(e.to_string()),
        }
    }
}

/// `f`, a panic in it returned as [`Error::Internal`]: varena's public boundary.
pub(crate) fn caught<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|p| {
        let msg = p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned());
        Err(Error::Internal(format!("panic: {}", msg.unwrap_or_default())))
    })
}

//! varena: the memory of the processes computing with model weights (design `varena-slim.md`).
//!
//! [`table`] holds one process's facts and [`decide`] is the one pure decision procedure over
//! them; [`Memory`] applies its actions in an executor over a [`Backend`] (CUDA VMM, or host
//! pages). [`Ledger`] is the GPU's shared table, run by TensorD or opened by name by a
//! standalone engine; [`custody`] keeps filled pages across a process's death; [`Tier`] is the
//! machine's RAM tier. TensorFS's `io` moves every byte.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

pub mod backend;
pub mod custody;
pub mod decide;
#[cfg(feature = "faults")]
pub mod faults;
pub mod layout;
pub mod ledger;
pub mod memory;
mod os;
pub mod table;
pub mod tier;

pub use backend::{Access, Backend, Event, HostBackend, Physical, Va};
pub use ledger::{Grant, Ledger, Peak, RowStatus, Status};
pub use memory::{CacheId, Fills, Measured, Memory, Moved, Next, Origin, Pass, Pin, Ref, Request, SetId, Site, SitePeak, Stats, StepStats, Torch, Yielded};
pub use table::{Fill, Fingerprint, Level, PartId, Plan, SetSpec, Step};
pub use tier::{Pressure, Tier};

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
}

#[derive(Debug, Clone, Error)]
pub enum Error {
    #[error("exhausted: {} bytes short{}", .0.need, .0.site.as_deref().map(|s| format!(" at {s}")).unwrap_or_default())]
    Exhausted(Box<Exhausted>),
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
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// The stable machine name of the error class.
    pub fn code(&self) -> &'static str {
        match self {
            Error::Exhausted(_) => "EXHAUSTED",
            Error::Invalid(_) => "INVALID",
            Error::Device { .. } => "DEVICE",
            Error::Io(_) => "IO",
            Error::Closed(_) => "CLOSED",
            Error::Internal(_) => "INTERNAL",
        }
    }

    /// CUDA documents these as destroying the context: the process must end, and the GPU's kept
    /// pages go (cuda 13.0 driver API, CUresult).
    pub fn is_sticky(&self) -> bool {
        matches!(self, Error::Device { code: 700 | 702 | 709 | 710 | 714..=719 | 721, .. })
    }

    pub(crate) fn io(what: &str, e: impl std::fmt::Display) -> Error {
        Error::Io(format!("{what}: {e}"))
    }

    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub(crate) fn os(what: &str) -> Error {
        Error::io(what, std::io::Error::last_os_error())
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

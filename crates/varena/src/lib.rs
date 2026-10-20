//! varena: the memory of the process computing with model weights.
//!
//! `memory` places a weight set's regions on a GPU (home pages, scratch slots, pins as tensor
//! lifetimes, eviction of what nothing uses); `Host` serves them from host memory where there
//! is no GPU. The host tier (`host`, `window`) is the machine's: sealed memfds a TensorD fills
//! and executors adopt read-only. `tensorfs` owns parsing, source custody and every host/device byte transfer.
//!
//! The host tier is Linux-only and answers [`Error::Unsupported`] elsewhere. On Windows the
//! crate is empty until its tier exists (xs-043 unit WM).
#![cfg(unix)]

pub use tensorfs::io::{cuda, metrics as ceiling};
pub mod faults;
pub mod gpu;
pub mod host;
pub mod io;
pub mod layout;
pub mod ledger;
pub mod plan;
pub mod memory;
pub mod window;

pub use memory::{Gpu, GpuStats, Host, HostPinned, HostSet, Pinned, WeightSet};

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
    /// The driver refused to map: `deficit` of the `need` bytes found no device memory.
    #[error("device full: mapping {need} bytes, {deficit} short")]
    DeviceFull { need: u64, deficit: u64 },
    #[error("invalid: {0}")]
    Invalid(String),
    #[error("internal varena failure: {0}")]
    Internal(&'static str),
    #[error("io: {0}")]
    Io(String),
    #[error("source: {0}")]
    Source(#[from] tensorfs::failure::Failure),
    /// What closed: a weight set, a source.
    #[error("closed: {0}")]
    Closed(String),
    /// This OS has no such mechanism (the host tier off Linux).
    #[error("unsupported: {0}")]
    Unsupported(String),
}

/// An OS seam's failure, named by what was attempted.
pub(crate) fn os(what: &'static str) -> impl FnOnce(platform::Error) -> Error {
    move |error| {
        if error.is_unsupported() {
            Error::Unsupported(format!("{what}: {error}"))
        } else {
            Error::Io(format!("{what}: {}", std::io::Error::from(error)))
        }
    }
}

impl Error {
    /// CUDA documents these failures as destroying the context or leaving the process
    /// inconsistent. Allocation pressure and invalid inputs are request failures instead.
    /// https://docs.nvidia.com/cuda/archive/13.0.3/cuda-driver-api/group__CUDA__TYPES.html
    pub fn is_sticky(&self) -> bool {
        matches!(self, Self::Cuda { code: 700 | 702 | 709 | 710 | 714..=719 | 721, .. })
    }

    pub fn is_oom(&self) -> bool {
        matches!(self, Self::DeviceFull { .. } | Self::Cuda { code: 2, .. })
    }

    /// The stable machine name of the error class.
    pub fn code(&self) -> &'static str {
        match self {
            Error::Cuda { .. } => "CUDA",
            Error::CudaUnavailable(_) => "CUDA_UNAVAILABLE",
            Error::DeviceFull { .. } => "DEVICE_FULL",
            Error::Invalid(_) => "INVALID",
            Error::Internal(_) => "INTERNAL",
            Error::Io(_) => "IO",
            Error::Source(_) => "SOURCE",
            Error::Closed(_) => "CLOSED",
            Error::Unsupported(_) => "UNSUPPORTED",
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<tensorfs::err::Refusal> for Error {
    fn from(error: tensorfs::err::Refusal) -> Error { Error::Source(error.into()) }
}

impl From<tensorfs::io::cuda::Error> for Error {
    fn from(error: tensorfs::io::cuda::Error) -> Error {
        match error {
            tensorfs::io::cuda::Error::Cuda { op, code, name } => Error::Cuda { op, code, name },
            tensorfs::io::cuda::Error::CudaUnavailable(detail) => Error::CudaUnavailable(detail),
        }
    }
}
impl From<tensorfs::io::Error> for Error {
    fn from(error: tensorfs::io::Error) -> Error {
        match error.cuda().cloned() {
            Some(cuda) => cuda.into(),
            None => Error::Io(error.to_string()),
        }
    }
}
impl From<tensorfs::io::source::Error> for Error {
    fn from(error: tensorfs::io::source::Error) -> Error {
        match error {
            tensorfs::io::source::Error::Read(error) => error.into(),
            tensorfs::io::source::Error::Source(error) => error.into(),
        }
    }
}

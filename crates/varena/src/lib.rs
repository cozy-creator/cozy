//! varena: the memory of the process computing with model weights.
//!
//! `memory` places a weight set's regions on a GPU (home pages, scratch slots, pins as tensor
//! lifetimes, eviction of what nothing uses); `Host` serves them from host memory where there
//! is no GPU. The host tier (`host`, `window`) is the machine's: sealed memfds a TensorD fills
//! and executors adopt read-only. `tensorfs-core` supplies verified bytes and stays CUDA-free.
//!
//! The host tier is Linux-only and answers [`Error::Unsupported`] elsewhere. On Windows the
//! crate is empty until its tier exists (xs-043 unit WM).
#![cfg(unix)]

#[cfg(feature = "store")]
pub mod checkpoint;
pub mod cuda;
pub mod faults;
pub mod host;
pub mod io;
pub mod layout;
pub mod ledger;
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
    #[error("io: {0}")]
    Io(String),
    #[error("source: {0}")]
    Source(#[from] tensorfs_core::err::Refusal),
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
    /// The stable machine name of the error class.
    pub fn code(&self) -> &'static str {
        match self {
            Error::Cuda { .. } => "CUDA",
            Error::CudaUnavailable(_) => "CUDA_UNAVAILABLE",
            Error::DeviceFull { .. } => "DEVICE_FULL",
            Error::Invalid(_) => "INVALID",
            Error::Io(_) => "IO",
            Error::Source(_) => "SOURCE",
            Error::Closed(_) => "CLOSED",
            Error::Unsupported(_) => "UNSUPPORTED",
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

//! tensorfs-gpu: the TensorFS weight plane, a public mechanism with no policy.
//!
//! A weight set is a checkpoint's chunks grouped into regions (`tensorfs_core::client::regions`)
//! with one byte layout: the same offsets in a sealed host layout and in the device VA. The plane
//! maps a region when asked (`want`, a lease's miss, a cursor's ring), fills it from the chunk
//! files or a sealed host layout, and unmaps it only on `drop`. There is no budget and no
//! eviction: it maps until the driver refuses or the floor would be crossed (`DeviceFull`), and
//! only a live lease pins. The caller chooses what to drop from `stats()`.
//!
//! The plane is a Unix mechanism: a sealed host layout crosses processes as a memfd (Linux).
#![cfg(unix)]

pub mod cuda;
pub mod cursor;
pub mod device;
pub mod faults;
pub mod host;
pub mod io;
pub mod layout;
pub mod plane;

pub use cursor::Cursor;
pub use layout::Layout;
pub use plane::{Exported, Host, Lease, Plane, PlaneConfig, Stats, Ticket, WsId};

use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum Error {
    /// The driver refused memory, or mapping would cross the floor: free `deficit` bytes.
    #[error("device full: free {deficit} more bytes")]
    DeviceFull { deficit: u64 },
    /// `need` exceeds what the device can ever give (its memory less the floor).
    #[error("floor: {need} bytes exceed the device's {capacity} usable bytes")]
    Floor { need: u64, capacity: u64 },
    /// A driver call failed, or there is no usable driver (`code` -1).
    #[error("cuda driver: {op} failed with {name} ({code})")]
    Cuda { op: &'static str, code: i32, name: String },
    /// What closed: a weight set, a cursor, a copy stream, the plane.
    #[error("closed: {0}")]
    Closed(String),
    #[error("invalid: {0}")]
    Invalid(String),
    /// A device error left state the plane cannot vouch for; every later call refuses.
    #[error("poisoned: {0}")]
    Poisoned(String),
    #[error("io: {0}")]
    Io(String),
}

impl Error {
    /// The stable machine name of the error class.
    pub fn code(&self) -> &'static str {
        match self {
            Error::DeviceFull { .. } => "DEVICE_FULL",
            Error::Floor { .. } => "FLOOR",
            Error::Cuda { .. } => "CUDA",
            Error::Closed(_) => "CLOSED",
            Error::Invalid(_) => "INVALID",
            Error::Poisoned(_) => "POISONED",
            Error::Io(_) => "IO",
        }
    }
}

/// An OS seam's failure, named by what was attempted.
pub(crate) fn os(what: &'static str) -> impl FnOnce(cozy_platform::Error) -> Error {
    move |error| Error::Io(format!("{what}: {}", std::io::Error::from(error)))
}

pub type Result<T> = std::result::Result<T, Error>;

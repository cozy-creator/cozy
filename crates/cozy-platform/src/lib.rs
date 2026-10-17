//! Every OS-specific seam of the `cozy` binary (design §6.2). Callers never write `cfg(target_os)`:
//! an OS without an implementation answers [`Error::Unsupported`], never a panic.
//!
//! Moved code keeps `io::Result` inside; the two conversions below let a moved body's `?` and a
//! caller's `?` work across the boundary, an `io::Error` keeping its errno both ways.
use std::{fmt, io, path::PathBuf};

pub mod contain;
pub mod durable;
pub mod gpu;
#[cfg(unix)]
pub mod handles;
pub mod ipc;
pub mod lock;
pub mod paths;
pub mod process;
pub mod service;
#[cfg(unix)]
pub mod shm;
pub mod tail;
pub mod tools;
pub mod watch;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// This seam has no implementation on this OS yet.
    #[error("{seam} is not supported on {os}")]
    Unsupported { seam: Seam, os: Os },
    #[error("{op} {}: {source}", path.display())]
    Io {
        op: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("{op}: {source}")]
    Os {
        op: &'static str,
        #[source]
        source: io::Error,
    },
    #[error("{manager}: {detail}")]
    Service {
        manager: &'static str,
        detail: String,
    },
    #[error("NVML {op}: {detail}")]
    Gpu { op: &'static str, detail: String },
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn unsupported(seam: Seam) -> Self {
        Self::Unsupported {
            seam,
            os: Os::CURRENT,
        }
    }

    pub fn is_unsupported(&self) -> bool {
        matches!(self, Self::Unsupported { .. })
    }

    pub(crate) fn io(op: &'static str, path: impl Into<PathBuf>) -> impl FnOnce(io::Error) -> Self {
        let path = path.into();
        move |source| Self::Io { op, path, source }
    }

    pub(crate) fn os(op: &'static str) -> impl FnOnce(io::Error) -> Self {
        move |source| Self::Os { op, source }
    }
}

impl From<io::Error> for Error {
    fn from(source: io::Error) -> Self {
        Self::Os {
            op: "system call",
            source,
        }
    }
}

impl From<Error> for io::Error {
    /// An `Io`/`Os` error gives back the `io::Error` it carried, errno intact; `Unsupported` is
    /// `ErrorKind::Unsupported` carrying the typed error.
    fn from(error: Error) -> Self {
        match error {
            Error::Io { source, .. } | Error::Os { source, .. } => source,
            Error::Unsupported { .. } => io::Error::new(io::ErrorKind::Unsupported, error),
            other => io::Error::other(other),
        }
    }
}

/// The rows of design §6.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seam {
    Process,
    Containment,
    Pressure,
    SharedMemory,
    HandlePassing,
    PeerIdentity,
    LocalIpc,
    Durability,
    Gpu,
    Service,
    Paths,
    FileWatch,
}

impl fmt::Display for Seam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Process => "process control",
            Self::Containment => "process containment",
            Self::Pressure => "memory pressure",
            Self::SharedMemory => "shared memory",
            Self::HandlePassing => "handle passing",
            Self::PeerIdentity => "peer identity",
            Self::LocalIpc => "local IPC",
            Self::Durability => "file durability",
            Self::Gpu => "GPU probe",
            Self::Service => "user services",
            Self::Paths => "paths",
            Self::FileWatch => "file watching",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Os {
    Linux,
    Macos,
    Windows,
}

impl Os {
    #[cfg(target_os = "linux")]
    pub const CURRENT: Os = Os::Linux;
    #[cfg(target_os = "macos")]
    pub const CURRENT: Os = Os::Macos;
    #[cfg(windows)]
    pub const CURRENT: Os = Os::Windows;
}

impl fmt::Display for Os {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Linux => "Linux",
            Self::Macos => "macOS",
            Self::Windows => "Windows",
        })
    }
}

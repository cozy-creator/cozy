//! Every OS-specific seam of the `cozy` binary (design §6.2). Callers never write `cfg(target_os)`:
//! an OS without an implementation answers [`Error::Unsupported`], never a panic.
//!
//! Containment, shared memory, handle passing and local IPC arrive with TensorD's and TensorFS's
//! move into the workspace (xs-043 unit W), together with their call sites.
use std::{fmt, io, path::PathBuf};

pub mod durable;
pub mod gpu;
pub mod lock;
pub mod paths;
pub mod process;
pub mod service;

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

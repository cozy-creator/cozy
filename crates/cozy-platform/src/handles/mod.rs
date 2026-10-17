//! Descriptor passing and peer identity on a local socket (TensorD `os.rs` and `protocol.rs`):
//! `SCM_RIGHTS`, `SO_PEERCRED` and `SO_PEERPIDFD` on Linux. macOS answers `Unsupported` until the
//! Apple tier (AP) ports them to `getpeereid` and `LOCAL_PEERPID`.
#[cfg_attr(target_os = "linux", path = "linux.rs")]
#[cfg_attr(not(target_os = "linux"), path = "unsupported.rs")]
mod imp;

pub use imp::*;

/// The process on the other end of a local socket, always this user (another is refused).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Peer {
    pub pid: u32,
    pub uid: u32,
}

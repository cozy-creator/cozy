//! Descriptor passing and peer identity on a local stream (TensorD `os.rs` and `protocol.rs`).
//! Unix: `SCM_RIGHTS`, the peer from `SO_PEERCRED`/`SO_PEERPIDFD` (Linux) or `getpeereid` and
//! `LOCAL_PEERPID` (macOS). Windows: the handle is duplicated into the peer's process and its
//! value sent inline; the peer is the pipe's other end.
#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;

#[cfg(unix)]
pub use unix::*;
#[cfg(windows)]
pub use windows::*;

/// An owned descriptor or handle, as `recv_fd` returns it.
#[cfg(unix)]
pub type Owned = std::os::fd::OwnedFd;
/// An owned descriptor or handle, as `recv_fd` returns it.
#[cfg(windows)]
pub type Owned = std::os::windows::io::OwnedHandle;

/// The process on the other end of a local stream and the ids it runs as (0 on Windows, which
/// names no uid: its pipes admit only this user).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Peer {
    pub pid: u32,
    pub uid: u32,
    pub gid: u32,
}

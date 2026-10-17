//! Descriptor passing and peer identity on a local socket (TensorD `os.rs` and `protocol.rs`):
//! `SCM_RIGHTS` on both Unixes; the peer from `SO_PEERCRED` and `SO_PEERPIDFD` on Linux,
//! `getpeereid` and `LOCAL_PEERPID` on macOS.
mod unix;

pub use unix::*;

/// The process on the other end of a local socket and the ids it runs as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Peer {
    pub pid: u32,
    pub uid: u32,
    pub gid: u32,
}

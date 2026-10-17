use super::Peer;
use crate::{Error, Result, Seam};
use std::{
    fs::File,
    os::{
        fd::{AsRawFd, OwnedFd},
        unix::net::UnixStream,
    },
};

pub fn peer_credentials(_: &UnixStream) -> Result<Peer> {
    Err(Error::unsupported(Seam::PeerIdentity))
}

pub fn peer_descends_from_machine(_: &UnixStream) -> Result<bool> {
    Err(Error::unsupported(Seam::PeerIdentity))
}

pub fn peer_pidfd(_: &UnixStream) -> Result<File> {
    Err(Error::unsupported(Seam::PeerIdentity))
}

pub fn send_fd(_: &UnixStream, _: &impl AsRawFd) -> Result<()> {
    Err(Error::unsupported(Seam::HandlePassing))
}

pub fn recv_fd(_: &UnixStream) -> Result<OwnedFd> {
    Err(Error::unsupported(Seam::HandlePassing))
}

/// The peer end of a lease is closed: the reader released, or its process is gone.
pub fn hung_up(lease: &UnixStream) -> bool {
    use std::os::fd::AsRawFd;
    let mut poll = libc::pollfd {
        fd: lease.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: one live descriptor, instantaneous observation.
    let ready = unsafe { libc::poll(&mut poll, 1, 0) } > 0;
    ready && poll.revents & (libc::POLLHUP | libc::POLLERR) != 0
}

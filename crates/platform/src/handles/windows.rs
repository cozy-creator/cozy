//! Windows: a handle crosses by being duplicated into the peer's process, through the process
//! handle the stream keeps; the stream carries a zero marker byte and its value there.
#![allow(unsafe_code)]
use super::Peer;
use crate::{Error, Result, ipc::LocalStream};
use std::{
    io::{self, Read, Write},
    os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle, RawHandle},
};

/// The peer as the pipe names it (`GetNamedPipe{Client,Server}ProcessId`), checked by its
/// process token: it runs as this user (uid 0, which is [`crate::memory::euid`] here), or it
/// is refused. The pipes' access list admits only this user; the token check also refuses a
/// squatter's pipe to a dialer, and holds once a pipe admits more.
pub fn peer(stream: &LocalStream) -> io::Result<Peer> {
    let pid = stream.peer_pid()?;
    let (user, me) = (stream.peer_user()?, crate::ipc::this_user()?);
    if user != me {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!("process {pid} runs as {user}, not as this user ({me})"),
        ));
    }
    Ok(Peer { pid, uid: 0, gid: 0 })
}

/// The peer's birth, read from the process handle the stream keeps (the one handles are
/// duplicated into), so a pid reused after the pipe was handed on is never taken for it.
pub fn peer_birth(stream: &LocalStream) -> Result<crate::process::Birth> {
    Ok(crate::process::birth_of_handle(stream.peer_pid()?, stream.peer_process()?)?)
}

/// Whether [`send_fd`] can reach the peer; the error says why not (a peer of a higher
/// integrity level), before anything is written to the stream.
pub fn accepts_handles(stream: &LocalStream) -> io::Result<()> {
    stream.peer_accepts_handles()
}

/// The peer, always this user (see [`peer`]).
pub fn peer_credentials(stream: &LocalStream) -> Result<Peer> {
    Ok(peer(stream)?)
}

/// Whether the connecting process descends from this machine.
pub fn peer_descends_from_machine(stream: &LocalStream) -> Result<bool> {
    let mut pid = peer(stream)?.pid;
    let machine = std::process::id();
    while pid > 0 {
        if pid == machine {
            return Ok(true);
        }
        let parent = crate::process::parent_of(pid)?;
        // A parent's ID can be reused by a later process: a parent that is its own child's
        // successor ends the walk rather than looping.
        if parent == pid {
            break;
        }
        pid = parent;
    }
    Ok(false)
}

/// One handle, duplicated into the peer and named there.
pub fn send_fd(stream: &LocalStream, handle: &impl AsRawHandle) -> Result<()> {
    let value = crate::ipc::duplicate_into_peer(stream, handle.as_raw_handle())
        .map_err(Error::os("send a handle"))?;
    let mut message = [0u8; 9];
    message[1..].copy_from_slice(&value.to_le_bytes());
    let mut writer = stream;
    writer
        .write_all(&message)
        .map_err(Error::os("send a handle"))?;
    Ok(())
}

/// One handle the peer duplicated into this process.
pub fn recv_fd(stream: &LocalStream) -> Result<OwnedHandle> {
    let mut message = [0u8; 9];
    let mut reader = stream;
    reader
        .read_exact(&mut message)
        .map_err(Error::os("receive a handle"))?;
    if message[0] != 0 {
        return Err(Error::os("receive a handle")(io::Error::new(
            io::ErrorKind::InvalidData,
            "expected a handle marker",
        )));
    }
    let value = u64::from_le_bytes(message[1..].try_into().expect("eight bytes"));
    // SAFETY: the peer duplicated this handle into this process for us to own.
    Ok(unsafe { OwnedHandle::from_raw_handle(value as RawHandle) })
}

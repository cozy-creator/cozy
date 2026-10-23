//! The OS mechanisms under the lease socket, as cozy-platform's `memory` seams give them on every
//! OS (local services whose bound name is their lock, mapped and sealed pages); which peers are
//! served; and the typed messages varena sends over a service stream.

use std::io::{Read, Write};

pub(crate) use platform::handles::Owned;
pub(crate) use platform::memory::{euid, size, Listener, Map, Stream};

use crate::{Error, Result};

/// A message on a service stream: JSON bytes and, maybe, one handle.
pub(crate) type Message = (Vec<u8>, Option<Owned>);

static NAMESPACE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Tests only: a process-wide prefix for every service name (the lease socket), so a test's
/// never meet a machine's. Set once, before any open; false when another prefix was set first.
pub fn namespace(prefix: &str) -> bool {
    NAMESPACE.get_or_init(|| prefix.to_string()) == prefix
}

/// `name` within this process's namespace.
fn named(name: &str) -> String {
    format!("{}{name}", NAMESPACE.get().map_or("", String::as_str))
}

/// Tests only: removes the table of `gpu` (a test's own, in its namespace).
pub fn remove_table(gpu: &str) {
    platform::memory::unlink(&named(gpu));
}

/// Peers of this user are served and served by.
fn accepted(uid: u32) -> bool {
    uid == euid()
}

/// `len` bytes of `f` from its start.
pub(crate) fn map(f: &Owned, len: usize, write: bool) -> Result<Map> {
    Ok(platform::memory::map(f, 0, len, write)?)
}

/// A shared page (a lease table) no holder can resize under another's mapping: a memfd sealed
/// against resizing (an agent's ftruncate would SIGBUS the manager), or on Windows a pagefile
/// section, which no holder can shrink.
pub(crate) fn sealed_page(len: u64) -> Result<Owned> {
    let page = platform::memory::page(len)?;
    #[cfg(unix)]
    platform::shm::seal_with(std::os::fd::AsFd::as_fd(&page), platform::shm::RESIZE_SEALS)?;
    Ok(page)
}

/// Binds service `name` in user `ns`'s namespace: `None` while another process holds it.
pub(crate) fn listen(ns: u32, name: &str) -> Result<Option<Listener>> {
    Ok(platform::memory::listen(ns, &named(name))?)
}

/// Connects to service `name` in user `ns`'s namespace; `None` when nobody serves it. A
/// service of a user not accepted is refused.
pub(crate) fn connect(ns: u32, name: &str) -> Result<Option<Stream>> {
    let Some(s) = platform::memory::connect(ns, &named(name))? else { return Ok(None) };
    peer(&s)?;
    Ok(Some(s))
}

/// The peer's pid; a peer of a user not accepted is refused (an abstract name has no
/// permissions of its own).
pub(crate) fn peer(stream: &Stream) -> Result<u32> {
    let (pid, uid) = platform::memory::peer(stream)?;
    if !accepted(uid) {
        return Err(Error::Invalid(format!("a peer of uid {uid} is not accepted")));
    }
    Ok(pid)
}

/// A page (a lease table, a host chunk) for another process to map read and write, never resize.
pub(crate) fn lend(page: &Owned) -> Result<Owned> {
    Ok(platform::memory::lend(page)?)
}

pub(crate) fn send(stream: &Stream, (bytes, handle): &Message) -> Result<()> {
    let mut s = stream;
    let mut head = (bytes.len() as u32).to_le_bytes().to_vec();
    head.push(handle.is_some() as u8);
    head.extend_from_slice(bytes);
    s.write_all(&head).map_err(|e| Error::io("send", e))?;
    if let Some(h) = handle {
        platform::handles::send_fd(stream, h)?;
    }
    Ok(())
}

/// The next message; `None` at the end of the stream.
pub(crate) fn recv(stream: &Stream) -> Result<Option<Message>> {
    let mut s = stream;
    let mut head = [0u8; 5];
    match s.read_exact(&mut head) {
        Ok(()) => {}
        Err(e) if matches!(e.kind(), std::io::ErrorKind::UnexpectedEof | std::io::ErrorKind::ConnectionReset | std::io::ErrorKind::BrokenPipe) => return Ok(None),
        Err(e) => return Err(Error::io("recv", e)),
    }
    let [a, b, c, d, fd] = head;
    let len = u64::from(u32::from_le_bytes([a, b, c, d]));
    // Grows with what arrives, never with what the peer claims.
    let mut bytes = vec![];
    s.take(len).read_to_end(&mut bytes).map_err(|e| Error::io("recv", e))?;
    if bytes.len() as u64 != len {
        return Ok(None);
    }
    let handle = if fd == 1 { Some(platform::handles::recv_fd(stream)?) } else { None };
    Ok(Some((bytes, handle)))
}

/// A typed message's bytes.
pub(crate) fn encode<T: serde::Serialize>(m: &T) -> Result<Vec<u8>> {
    serde_json::to_vec(m).map_err(|e| Error::Internal(format!("encode: {e}")))
}

pub(crate) fn decode<T: serde::de::DeserializeOwned>(b: &[u8]) -> Result<T> {
    serde_json::from_slice(b).map_err(|e| Error::Invalid(format!("decode: {e}")))
}


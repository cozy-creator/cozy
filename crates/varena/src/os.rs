//! The OS mechanisms under the ledger and the services, as cozy-platform's `memory` seams give
//! them on every OS (a named shared table, range locks that die with their holder, a local
//! event, the GPU's named wake, local services whose bound name is their lock, host pages);
//! which peers are served; and the typed messages varena sends over a service stream.

use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::sync::atomic::AtomicBool;

use parking_lot::Mutex;

pub(crate) use platform::handles::Owned;
pub(crate) use platform::memory::{allocated, await_event, euid, held, lock, own, punch, resize, share, size, stalled, unlock, Endpoint, Listener, Map, Notifier, Stream};

use crate::{Error, Result};

/// A message on a service stream: JSON bytes and, maybe, one handle.
pub(crate) type Message = (Vec<u8>, Option<Owned>);

static NAMESPACE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Tests only: a process-wide prefix for every table and service name (custody, the tier, the
/// wake, the asks), so a test's never meet a machine's. Set once, before any open; false when
/// another prefix was set first.
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

/// The table named `name` for this user, created when absent.
pub(crate) fn shared(name: &str, _create: bool) -> Result<Owned> {
    Ok(platform::memory::table(&named(name))?)
}

pub(crate) fn unlink(name: &str) {
    platform::memory::unlink(&named(name));
}

pub(crate) fn tid() -> u32 {
    platform::memory::thread_id()
}

/// Peers of these uids are served and served by, beside this process's own: the owners of the
/// tables it adopted, and the executors its ledger granted.
static ACCEPTED: Mutex<BTreeSet<u32>> = Mutex::new(BTreeSet::new());

pub(crate) fn accept_uid(uid: u32) {
    ACCEPTED.lock().insert(uid);
}

fn accepted(uid: u32) -> bool {
    uid == euid() || ACCEPTED.lock().contains(&uid)
}

/// The effective uid of `pid`.
pub(crate) fn uid_of(pid: u32) -> Result<u32> {
    Ok(platform::memory::uid_of(pid)?)
}

/// `len` bytes of `f` from its start.
pub(crate) fn map(f: &Owned, len: usize, write: bool) -> Result<Map> {
    Ok(platform::memory::map(f, 0, len, write)?)
}

/// `len` bytes of `f` from `offset`, a multiple of the mapping granularity.
pub(crate) fn map_at(f: &Owned, offset: u64, len: usize, write: bool) -> Result<Map> {
    Ok(platform::memory::map(f, offset, len, write)?)
}

/// A new description of the same file, read-write: its locks are its own.
pub(crate) fn reopen(f: &Owned, _write: bool) -> Result<Owned> {
    Ok(platform::memory::reopen(f)?)
}

pub(crate) fn linked(f: &Owned) -> Result<bool> {
    Ok(platform::memory::linked(f)?)
}

/// A local event: (the waiting end, the signalling end).
pub(crate) fn event() -> Result<(Owned, Owned)> {
    Ok(platform::memory::event()?)
}

pub(crate) fn signal(writer: &Owned) {
    platform::memory::signal(writer);
}

/// A RAM-tier layout's file (Linux's; elsewhere the tier is absent).
pub(crate) fn tier_file(name: &str) -> Result<Owned> {
    Ok(platform::memory::tier_file(name)?)
}

/// Binds the GPU's wake in user `ns`'s namespace; `None` while another process holds it.
pub(crate) fn wake_bind(ns: u32, name: &str) -> Result<Option<Endpoint>> {
    Ok(Endpoint::bind(ns, &named(name))?)
}

pub(crate) fn wake_sender() -> Result<Notifier> {
    Ok(Notifier::new()?)
}

/// Signals the GPU's wake in user `ns`'s namespace.
pub(crate) fn wake(sender: &Notifier, ns: u32, name: &str) {
    sender.notify(ns, &named(name));
}

/// Clears every wake so far; true when one came from an accepted user.
pub(crate) fn wake_clear(endpoint: &Endpoint) -> bool {
    endpoint.clear(accepted)
}

/// Blocks until an accepted user wakes the endpoint, clearing every wake so far.
pub(crate) fn wake_await(endpoint: &Endpoint) -> Result<()> {
    Ok(endpoint.wait(accepted)?)
}

/// Binds service `name` in user `ns`'s namespace: `None` while another process holds it.
pub(crate) fn listen(ns: u32, name: &str) -> Result<Option<Listener>> {
    Ok(platform::memory::listen(ns, &named(name))?)
}

/// Every connection the service accepts, for its life.
pub(crate) fn incoming(listener: Listener) -> impl Iterator<Item = Stream> {
    std::iter::repeat_with(move || platform::memory::accept(&listener)).filter_map(|c| c.ok())
}

/// Serves every connection at once until `stop`: each one's message answered with `answer`'s
/// bytes. Peers not accepted are dropped.
pub(crate) fn serve_all(listener: Listener, stop: &AtomicBool, answer: impl Fn(&[u8]) -> Vec<u8> + Send + Sync + 'static) {
    platform::memory::serve_all(listener, stop, accepted, answer);
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

/// Ends both directions of a stream (its peer reads its end).
pub(crate) fn close(stream: &Stream) {
    let _ = stream.shutdown(std::net::Shutdown::Both);
}

pub(crate) fn dup(h: &Owned) -> Result<Owned> {
    h.try_clone().map_err(|e| Error::io("dup", e))
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

/// Sends `m` with `handle` and reads the typed reply.
pub(crate) fn call<T: serde::Serialize, R: serde::de::DeserializeOwned>(stream: &Stream, m: &T, handle: Option<Owned>) -> Result<(R, Option<Owned>)> {
    send(stream, &(encode(m)?, handle))?;
    let (bytes, handle) = recv(stream)?.ok_or_else(|| Error::Closed("the service".into()))?;
    Ok((decode(&bytes)?, handle))
}

/// This process's resident high-water (`None`: unmeasured here).
pub(crate) fn high_water() -> Option<u64> {
    platform::memory::high_water()
}

/// Starts this process's high-water over from its use now.
pub(crate) fn reset_high_water() {
    platform::memory::reset_high_water()
}

/// `pid`'s anonymous bytes now.
pub(crate) fn anon(pid: u32) -> Option<u64> {
    platform::memory::anon(pid)
}

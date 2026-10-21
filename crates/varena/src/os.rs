//! The OS mechanisms under the ledger and the services: a named shared file, row locks that die
//! with their holder (OFD locks), a wake event (eventfd) and local sockets, a service's bound
//! name being its lock. Linux has them; elsewhere every call refuses
//! until its seam exists, and the decision code above is the same.

use platform::handles::Owned;

use crate::{Error, Result};

/// A shared mapping; unmapped on drop.
pub(crate) struct Map {
    pub base: *mut u8,
    pub len: usize,
}

// SAFETY: the mapping is reached only through atomics and bytes its users synchronise.
unsafe impl Send for Map {}
// SAFETY: as above.
unsafe impl Sync for Map {}

/// A message on a service stream: JSON bytes and, maybe, one handle.
pub(crate) type Message = (Vec<u8>, Option<Owned>);

#[cfg(target_os = "linux")]
mod imp {
    use std::io::{Read, Write};
    use std::os::fd::{AsFd, AsRawFd, FromRawFd};
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering::Acquire;
    use std::os::linux::net::SocketAddrExt;
    use std::os::unix::net::{SocketAddr, UnixListener, UnixStream};

    use super::{Map, Message};
    use crate::{Error, Result};
    use platform::handles::Owned;
    use platform::shm::Region;

    pub(crate) type Listener = UnixListener;
    pub(crate) type Stream = UnixStream;

    /// A file in shared memory named per user, `/dev/shm/cozy-varena-<uid>-<name>`, created
    /// when absent.
    pub(crate) fn shared(name: &str, create: bool) -> Result<Owned> {
        // SAFETY: geteuid reads a process attribute.
        let uid = unsafe { libc::geteuid() };
        let clean: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect();
        let path = format!("/dev/shm/cozy-varena-{uid}-{clean}");
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true).create(create);
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        let file = options.open(&path).map_err(|e| Error::io(&format!("open {path}"), e))?;
        Ok(file.into())
    }

    /// Refuses a table this user does not own, or that others may write.
    pub(crate) fn own(fd: &Owned) -> Result<()> {
        use std::os::unix::fs::MetadataExt;
        let file = std::fs::File::from(fd.try_clone().map_err(|e| Error::io("dup", e))?);
        let meta = file.metadata().map_err(|e| Error::io("stat", e))?;
        // SAFETY: geteuid reads a process attribute.
        if meta.uid() != unsafe { libc::geteuid() } || meta.mode() & 0o022 != 0 {
            return Err(Error::Invalid("the shared table belongs to someone else".into()));
        }
        Ok(())
    }

    pub(crate) fn size(fd: &Owned) -> Result<u64> {
        let file = std::fs::File::from(fd.try_clone().map_err(|e| Error::io("dup", e))?);
        Ok(file.metadata().map_err(|e| Error::io("stat", e))?.len())
    }

    pub(crate) fn resize(fd: &Owned, len: u64) -> Result<()> {
        // SAFETY: sizing a file we hold.
        if unsafe { libc::ftruncate(fd.as_raw_fd(), len as libc::off_t) } != 0 {
            return Err(Error::os("resize"));
        }
        Ok(())
    }

    pub(crate) fn map(fd: &Owned, len: usize, write: bool) -> Result<Map> {
        let prot = if write { libc::PROT_READ | libc::PROT_WRITE } else { libc::PROT_READ };
        // SAFETY: a new shared mapping of a file we hold.
        let base = unsafe { libc::mmap(std::ptr::null_mut(), len, prot, libc::MAP_SHARED, fd.as_raw_fd(), 0) };
        if base == libc::MAP_FAILED {
            return Err(Error::os("map"));
        }
        Ok(Map { base: base.cast(), len })
    }

    /// `len` bytes of `fd` from `offset`, a multiple of the page size.
    pub(crate) fn map_at(fd: &Owned, offset: u64, len: usize, write: bool) -> Result<Map> {
        let prot = if write { libc::PROT_READ | libc::PROT_WRITE } else { libc::PROT_READ };
        // SAFETY: a new shared mapping of a file we hold.
        let base = unsafe { libc::mmap(std::ptr::null_mut(), len, prot, libc::MAP_SHARED, fd.as_raw_fd(), offset as libc::off_t) };
        if base == libc::MAP_FAILED {
            return Err(Error::os("map"));
        }
        Ok(Map { base: base.cast(), len })
    }

    impl Drop for Map {
        fn drop(&mut self) {
            // SAFETY: our own mapping; nothing borrows it past its owner.
            unsafe { libc::munmap(self.base.cast(), self.len) };
        }
    }

    pub(crate) fn reopen(fd: &Owned, write: bool) -> Result<Owned> {
        Ok(platform::shm::reopen(fd.as_fd(), write)?)
    }

    /// Takes (or, `wait`, waits for) a write lock on a byte range of this description.
    pub(crate) fn lock(fd: &Owned, start: u64, len: u64, wait: bool) -> Result<bool> {
        Ok(platform::shm::region_lock(fd.as_fd(), start as usize, len as usize, Region::Write, wait)?)
    }

    /// Takes (or waits for) a read lock: many readers, no writer.
    pub(crate) fn share(fd: &Owned, start: u64, len: u64, wait: bool) -> Result<bool> {
        Ok(platform::shm::region_lock(fd.as_fd(), start as usize, len as usize, Region::Read, wait)?)
    }

    pub(crate) fn unlock(fd: &Owned, start: u64, len: u64) -> Result<()> {
        platform::shm::region_lock(fd.as_fd(), start as usize, len as usize, Region::Unlock, false)?;
        Ok(())
    }

    /// Whether another description holds a lock over the range.
    pub(crate) fn held(fd: &Owned, start: u64, len: u64) -> Result<bool> {
        Ok(platform::shm::region_holder(fd.as_fd(), start as usize, len as usize)?.is_some())
    }

    pub(crate) fn punch(fd: &Owned, offset: u64, len: u64) -> Result<()> {
        Ok(platform::shm::punch(fd.as_fd(), offset, len)?)
    }

    /// The bytes a file really holds (its allocated blocks).
    pub(crate) fn allocated(fd: &Owned) -> u64 {
        use std::os::unix::fs::MetadataExt;
        fd.try_clone().ok().and_then(|f| std::fs::File::from(f).metadata().ok()).map_or(0, |m| m.blocks() * 512)
    }

    pub(crate) fn memfd(name: &str) -> Result<Owned> {
        Ok(platform::shm::memfd_named(name)?)
    }




    pub(crate) fn event() -> Result<Owned> {
        // SAFETY: eventfd(2) with no shared state.
        let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
        if fd < 0 {
            return Err(Error::os("eventfd"));
        }
        // SAFETY: a descriptor we just created.
        Ok(unsafe { Owned::from_raw_fd(fd) })
    }

    pub(crate) fn signal(event: &Owned) {
        let one = 1u64.to_ne_bytes();
        // SAFETY: an 8-byte write to an eventfd; a full counter is a signal already.
        unsafe { libc::write(event.as_raw_fd(), one.as_ptr().cast(), 8) };
    }

    /// Blocks until `event` is readable, then clears it.
    pub(crate) fn await_event(event: &Owned) -> Result<()> {
        let mut poll = libc::pollfd { fd: event.as_raw_fd(), events: libc::POLLIN, revents: 0 };
        loop {
            // SAFETY: one live descriptor.
            if unsafe { libc::poll(&mut poll, 1, -1) } > 0 {
                let mut buf = [0u8; 8];
                // SAFETY: an 8-byte read into our buffer; EAGAIN when another reader took it.
                unsafe { libc::read(event.as_raw_fd(), buf.as_mut_ptr().cast(), 8) };
                return Ok(());
            }
            let e = std::io::Error::last_os_error();
            if e.kind() != std::io::ErrorKind::Interrupted {
                return Err(Error::io("poll", e));
            }
        }
    }

    fn address(name: &str) -> Result<SocketAddr> {
        // SAFETY: geteuid reads a process attribute.
        let uid = unsafe { libc::geteuid() };
        SocketAddr::from_abstract_name(format!("cozy-varena-{uid}-{name}")).map_err(|e| Error::io("socket name", e))
    }

    /// Binds the service name: `None` while another process holds it. The bound name is the
    /// service lock, released by the kernel when its holder dies.
    pub(crate) fn listen(name: &str) -> Result<Option<Listener>> {
        match UnixListener::bind_addr(&address(name)?) {
            Ok(l) => Ok(Some(l)),
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => Ok(None),
            Err(e) => Err(Error::io("bind service", e)),
        }
    }

    /// Every connection the service accepts, for its life.
    pub(crate) fn incoming(listener: Listener) -> impl Iterator<Item = Stream> {
        std::iter::repeat_with(move || listener.accept()).filter_map(|c| c.ok().map(|(s, _)| s))
    }

    /// Serves every connection to `listener` at once (one poll over all) until `stop` is seen:
    /// each connection's one message, which carries no handle, is answered with `answer`'s bytes
    /// and the connection closed. A peer of another user is refused at accept; a silent or
    /// stopped one holds nobody up.
    pub(crate) fn serve_all(listener: Listener, stop: &AtomicBool, mut answer: impl FnMut(&[u8]) -> Vec<u8>) {
        if listener.set_nonblocking(true).is_err() {
            return;
        }
        let mut conns: Vec<(Stream, Vec<u8>)> = vec![];
        loop {
            let mut polls: Vec<libc::pollfd> = std::iter::once(listener.as_raw_fd()).chain(conns.iter().map(|(s, _)| s.as_raw_fd())).map(|fd| libc::pollfd { fd, events: libc::POLLIN, revents: 0 }).collect();
            // SAFETY: live descriptors in a correctly sized array.
            if unsafe { libc::poll(polls.as_mut_ptr(), polls.len() as libc::nfds_t, -1) } < 0 && std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted {
                return;
            }
            if stop.load(Acquire) {
                return;
            }
            let ready: Vec<bool> = polls.iter().map(|p| p.revents != 0).collect();
            let mut next = vec![];
            for (i, (s, mut buf)) in conns.drain(..).enumerate() {
                if !ready.get(i + 1).copied().unwrap_or(false) {
                    next.push((s, buf));
                    continue;
                }
                let open = drain(&s, &mut buf);
                match frame(&buf) {
                    Frame::Whole(body) => {
                        let reply = answer(body);
                        let _ = (&s).write_all(&reply);
                    }
                    Frame::Partial if open => next.push((s, buf)),
                    Frame::Partial | Frame::Bad => {}
                }
            }
            if ready.first().copied().unwrap_or(false) {
                while let Ok((s, _)) = listener.accept() {
                    if peer(&s).is_ok() && s.set_nonblocking(true).is_ok() {
                        next.push((s, vec![]));
                    }
                }
            }
            conns = next;
        }
    }

    /// Reads what has arrived; false at the end of the stream or on an error.
    fn drain(s: &Stream, buf: &mut Vec<u8>) -> bool {
        let mut chunk = [0u8; 512];
        loop {
            match (&*s).read(&mut chunk) {
                Ok(0) => return false,
                Ok(n) => buf.extend_from_slice(chunk.get(..n).unwrap_or_default()),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return true,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(_) => return false,
            }
        }
    }

    enum Frame<'a> {
        Whole(&'a [u8]),
        Partial,
        /// A message with a handle, which an ask never carries.
        Bad,
    }

    fn frame(buf: &[u8]) -> Frame<'_> {
        let Some(&[a, b, c, d, fd]) = buf.get(..5) else { return Frame::Partial };
        if fd != 0 {
            return Frame::Bad;
        }
        let len = u32::from_le_bytes([a, b, c, d]) as usize;
        buf.get(5..5 + len).map_or(Frame::Partial, Frame::Whole)
    }

    /// Connects to the service; `None` when nobody serves it. A service of another user is
    /// refused.
    pub(crate) fn connect(name: &str) -> Result<Option<Stream>> {
        match UnixStream::connect_addr(&address(name)?) {
            Ok(s) => {
                peer(&s)?;
                Ok(Some(s))
            }
            Err(e) if e.kind() == std::io::ErrorKind::ConnectionRefused => Ok(None),
            Err(e) => Err(Error::io("connect service", e)),
        }
    }

    /// The peer's pid; a peer of another user is refused (abstract names have no permissions).
    pub(crate) fn peer(stream: &Stream) -> Result<u32> {
        Ok(platform::handles::peer_credentials(stream)?.pid)
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
            Err(e) if matches!(e.kind(), std::io::ErrorKind::UnexpectedEof | std::io::ErrorKind::ConnectionReset) => return Ok(None),
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


    pub(crate) fn dup(h: &Owned) -> Result<Owned> {
        h.try_clone().map_err(|e| Error::io("dup", e))
    }

    /// The calling thread's id.
    pub(crate) fn tid() -> u32 {
        // SAFETY: gettid has no arguments and cannot fail.
        unsafe { libc::syscall(libc::SYS_gettid) as u32 }
    }

    /// Thread `tid` of `pid` cannot run: it is stopped, or its process is a zombie. A thread
    /// or process that is gone is not stalled (its row reads dead).
    pub(crate) fn stalled(pid: u32, tid: u32) -> bool {
        // The state follows the parenthesised name, which may itself hold spaces and ')'.
        let state = |path: String| std::fs::read_to_string(path).ok().and_then(|s| s.rsplit_once(')').and_then(|(_, rest)| rest.split_whitespace().next().map(str::to_string)));
        let process = state(format!("/proc/{pid}/stat"));
        let thread = state(format!("/proc/{pid}/task/{tid}/stat"));
        process.as_deref() == Some("Z") || matches!(thread.as_deref(), Some("T" | "t"))
    }

    /// Whether the file still has a name (a table whose name was unlinked is nobody's).
    pub(crate) fn linked(fd: &Owned) -> Result<bool> {
        use std::os::unix::fs::MetadataExt;
        Ok(std::fs::File::from(dup(fd)?).metadata().map_err(|e| Error::io("stat the table", e))?.nlink() > 0)
    }

    pub(crate) fn unlink(name: &str) {
        // SAFETY: geteuid reads a process attribute.
        let uid = unsafe { libc::geteuid() };
        let clean: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect();
        let _ = std::fs::remove_file(format!("/dev/shm/cozy-varena-{uid}-{clean}"));
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {

    use super::{Map, Message};
    use crate::{Error, Result};
    use platform::handles::Owned;

    /// A service endpoint this OS does not have yet.
    pub(crate) enum Listener {}
    pub(crate) enum Stream {}

    fn later() -> Error {
        Error::Invalid(format!("the shared table is not yet on {}", std::env::consts::OS))
    }

    pub(crate) fn shared(_: &str, _: bool) -> Result<Owned> {
        Err(later())
    }
    pub(crate) fn own(_: &Owned) -> Result<()> {
        Err(later())
    }
    pub(crate) fn size(_: &Owned) -> Result<u64> {
        Err(later())
    }
    pub(crate) fn resize(_: &Owned, _: u64) -> Result<()> {
        Err(later())
    }
    pub(crate) fn map(_: &Owned, _: usize, _: bool) -> Result<Map> {
        Err(later())
    }
    pub(crate) fn map_at(_: &Owned, _: u64, _: usize, _: bool) -> Result<Map> {
        Err(later())
    }
    impl Drop for Map {
        fn drop(&mut self) {}
    }
    pub(crate) fn reopen(_: &Owned, _: bool) -> Result<Owned> {
        Err(later())
    }
    pub(crate) fn lock(_: &Owned, _: u64, _: u64, _: bool) -> Result<bool> {
        Err(later())
    }
    pub(crate) fn share(_: &Owned, _: u64, _: u64, _: bool) -> Result<bool> {
        Err(later())
    }
    pub(crate) fn unlock(_: &Owned, _: u64, _: u64) -> Result<()> {
        Err(later())
    }
    pub(crate) fn held(_: &Owned, _: u64, _: u64) -> Result<bool> {
        Err(later())
    }
    pub(crate) fn punch(_: &Owned, _: u64, _: u64) -> Result<()> {
        Err(later())
    }
    pub(crate) fn allocated(_: &Owned) -> u64 {
        0
    }
    pub(crate) fn memfd(_: &str) -> Result<Owned> {
        Err(later())
    }
    pub(crate) fn event() -> Result<Owned> {
        Err(later())
    }
    pub(crate) fn signal(_: &Owned) {}
    pub(crate) fn await_event(_: &Owned) -> Result<()> {
        Err(later())
    }
    pub(crate) fn listen(_: &str) -> Result<Option<Listener>> {
        Err(later())
    }
    pub(crate) fn incoming(listener: Listener) -> std::iter::Empty<Stream> {
        match listener {}
    }
    pub(crate) fn serve_all(listener: Listener, _: &std::sync::atomic::AtomicBool, _: impl FnMut(&[u8]) -> Vec<u8>) {
        match listener {}
    }
    pub(crate) fn connect(_: &str) -> Result<Option<Stream>> {
        Err(later())
    }
    pub(crate) fn peer(s: &Stream) -> Result<u32> {
        match *s {}
    }
    pub(crate) fn send(s: &Stream, _: &Message) -> Result<()> {
        match *s {}
    }
    pub(crate) fn recv(s: &Stream) -> Result<Option<Message>> {
        match *s {}
    }
    pub(crate) fn dup(_: &Owned) -> Result<Owned> {
        Err(later())
    }
    pub(crate) fn tid() -> u32 {
        0
    }
    pub(crate) fn stalled(_: u32, _: u32) -> bool {
        false
    }
    pub(crate) fn linked(_: &Owned) -> Result<bool> {
        Err(later())
    }
    pub(crate) fn unlink(_: &str) {}
}

pub(crate) use imp::*;


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

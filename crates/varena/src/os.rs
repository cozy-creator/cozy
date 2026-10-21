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
    unlink(gpu);
}

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
        let clean: String = super::named(name).chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect();
        let path = format!("/dev/shm/cozy-varena-{uid}-{clean}");
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true).create(create);
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        let file = options.open(&path).map_err(|e| Error::io(&format!("open {path}"), e))?;
        Ok(file.into())
    }

    /// The table's owner: this user, or root (a TensorD running as root grants its tables to
    /// executors of another uid). Any other owner, or a table others may write, is refused.
    pub(crate) fn own(fd: &Owned) -> Result<u32> {
        use std::os::unix::fs::MetadataExt;
        let file = std::fs::File::from(fd.try_clone().map_err(|e| Error::io("dup", e))?);
        let meta = file.metadata().map_err(|e| Error::io("stat", e))?;
        if (meta.uid() != euid() && meta.uid() != 0) || meta.mode() & 0o022 != 0 {
            return Err(Error::Invalid("the shared table belongs to someone else".into()));
        }
        Ok(meta.uid())
    }

    pub(crate) fn euid() -> u32 {
        // SAFETY: geteuid reads a process attribute.
        unsafe { libc::geteuid() }
    }

    /// Peers of these uids are served and served by, beside this process's own: the owners of
    /// the tables it adopted, and the executors its ledger granted.
    static ACCEPTED: parking_lot::Mutex<std::collections::BTreeSet<u32>> = parking_lot::Mutex::new(std::collections::BTreeSet::new());

    pub(crate) fn accept_uid(uid: u32) {
        ACCEPTED.lock().insert(uid);
    }

    /// The effective uid of `pid`, from `/proc`.
    pub(crate) fn uid_of(pid: u32) -> Result<u32> {
        let status = std::fs::read_to_string(format!("/proc/{pid}/status")).map_err(|e| Error::io("read a process's status", e))?;
        let uids = status.lines().find_map(|l| l.strip_prefix("Uid:")).ok_or_else(|| Error::Invalid(format!("no uid for {pid}")))?;
        uids.split_whitespace().nth(1).and_then(|u| u.parse().ok()).ok_or_else(|| Error::Invalid(format!("no uid for {pid}")))
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

    /// The GPU's wake endpoint, bound by its TensorD: a datagram socket named for the GPU that
    /// every participant signals, so a standalone process's frees reach it too. `None` while
    /// another process holds it.
    pub(crate) fn wake_bind(ns: u32, name: &str) -> Result<Option<Owned>> {
        match std::os::unix::net::UnixDatagram::bind_addr(&address(ns, &format!("wake-{name}"))?) {
            Ok(s) => {
                s.set_nonblocking(true).map_err(|e| Error::io("wake", e))?;
                // Each datagram carries its sender's credentials: only accepted users wake it.
                let one: libc::c_int = 1;
                // SAFETY: an int option on a socket we hold.
                if unsafe { libc::setsockopt(s.as_raw_fd(), libc::SOL_SOCKET, libc::SO_PASSCRED, (&one as *const libc::c_int).cast(), std::mem::size_of::<libc::c_int>() as libc::socklen_t) } != 0 {
                    return Err(Error::io("wake credentials", std::io::Error::last_os_error()));
                }
                Ok(Some(s.into()))
            }
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => Ok(None),
            Err(e) => Err(Error::io("bind the wake", e)),
        }
    }

    /// A participant's sending end (unbound).
    pub(crate) fn wake_sender() -> Result<Owned> {
        let s = std::os::unix::net::UnixDatagram::unbound().map_err(|e| Error::io("wake", e))?;
        s.set_nonblocking(true).map_err(|e| Error::io("wake", e))?;
        Ok(s.into())
    }

    /// Signals the GPU's wake; nobody bound (no TensorD), a full queue (a wake is pending
    /// already) and a bad name are all no signal needed.
    pub(crate) fn wake(sender: &Owned, ns: u32, name: &str) {
        let Ok(addr) = address(ns, &format!("wake-{name}")) else { return };
        let Ok(s) = sender.try_clone().map(std::os::unix::net::UnixDatagram::from) else { return };
        let _ = s.send_to_addr(&[1], &addr);
    }

    /// Clears every wake received so far; true when one came from an accepted user.
    pub(crate) fn wake_clear(endpoint: &Owned) -> bool {
        let mut any = false;
        loop {
            let mut buf = [0u8; 8];
            let mut iov = libc::iovec { iov_base: buf.as_mut_ptr().cast(), iov_len: buf.len() };
            // SAFETY: CMSG_SPACE is a size computation.
            let space = unsafe { libc::CMSG_SPACE(std::mem::size_of::<libc::ucred>() as u32) } as usize;
            let mut control = vec![0u8; space];
            // SAFETY: a zeroed header we fill below.
            let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
            msg.msg_iov = &mut iov;
            msg.msg_iovlen = 1;
            msg.msg_control = control.as_mut_ptr().cast();
            msg.msg_controllen = space;
            // SAFETY: a nonblocking receive into buffers that outlive the call.
            if unsafe { libc::recvmsg(endpoint.as_raw_fd(), &mut msg, libc::MSG_DONTWAIT) } < 0 {
                return any;
            }
            // SAFETY: the kernel filled the control buffer; the header walks only within it.
            let cmsg = unsafe { libc::CMSG_FIRSTHDR(&msg) };
            // SAFETY: a credentials message of the size the kernel wrote.
            let uid = (!cmsg.is_null() && unsafe { (*cmsg).cmsg_type } == libc::SCM_CREDENTIALS).then(|| unsafe { std::ptr::read_unaligned(libc::CMSG_DATA(cmsg) as *const libc::ucred).uid });
            any |= uid.is_some_and(|u| u == euid() || ACCEPTED.lock().contains(&u));
        }
    }

    /// Blocks until an accepted user wakes the endpoint, clearing every wake so far.
    pub(crate) fn wake_await(endpoint: &Owned) -> Result<()> {
        loop {
            let mut poll = libc::pollfd { fd: endpoint.as_raw_fd(), events: libc::POLLIN, revents: 0 };
            // SAFETY: one live descriptor.
            if unsafe { libc::poll(&mut poll, 1, -1) } < 0 {
                let e = std::io::Error::last_os_error();
                if e.kind() != std::io::ErrorKind::Interrupted {
                    return Err(Error::io("poll", e));
                }
            }
            if wake_clear(endpoint) {
                return Ok(());
            }
        }
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

    /// A service's name in the namespace of user `ns` (the table owner's: a granted executor
    /// finds its TensorD's services there).
    fn address(ns: u32, name: &str) -> Result<SocketAddr> {
        SocketAddr::from_abstract_name(format!("cozy-varena-{ns}-{}", super::named(name))).map_err(|e| Error::io("socket name", e))
    }

    /// Binds the service name: `None` while another process holds it. The bound name is the
    /// service lock, released by the kernel when its holder dies.
    pub(crate) fn listen(ns: u32, name: &str) -> Result<Option<Listener>> {
        match UnixListener::bind_addr(&address(ns, name)?) {
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

    /// Connects to the service; `None` when nobody serves it. A service of a user not accepted
    /// is refused.
    pub(crate) fn connect(ns: u32, name: &str) -> Result<Option<Stream>> {
        match UnixStream::connect_addr(&address(ns, name)?) {
            Ok(s) => {
                peer(&s)?;
                Ok(Some(s))
            }
            Err(e) if e.kind() == std::io::ErrorKind::ConnectionRefused => Ok(None),
            Err(e) => Err(Error::io("connect service", e)),
        }
    }

    /// The peer's pid; a peer of a user not accepted is refused (abstract names have no
    /// permissions).
    pub(crate) fn peer(stream: &Stream) -> Result<u32> {
        let p = platform::handles::peer(stream).map_err(|e| Error::io("peer", e))?;
        if p.uid != euid() && !ACCEPTED.lock().contains(&p.uid) {
            return Err(Error::Invalid(format!("a peer of uid {} is not accepted", p.uid)));
        }
        Ok(p.pid)
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


    /// Ends both directions of a stream (its peer reads its end).
    pub(crate) fn close(stream: &Stream) {
        let _ = stream.shutdown(std::net::Shutdown::Both);
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
        // `t` (traced: py-spy, gdb) is never a stall.
        process.as_deref() == Some("Z") || thread.as_deref() == Some("T")
    }

    /// Whether the file still has a name (a table whose name was unlinked is nobody's).
    pub(crate) fn linked(fd: &Owned) -> Result<bool> {
        use std::os::unix::fs::MetadataExt;
        Ok(std::fs::File::from(dup(fd)?).metadata().map_err(|e| Error::io("stat the table", e))?.nlink() > 0)
    }

    pub(crate) fn unlink(name: &str) {
        // SAFETY: geteuid reads a process attribute.
        let uid = unsafe { libc::geteuid() };
        let clean: String = super::named(name).chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect();
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
    pub(crate) fn euid() -> u32 {
        0
    }
    pub(crate) fn accept_uid(_: u32) {}
    pub(crate) fn uid_of(_: u32) -> Result<u32> {
        Err(later())
    }
    pub(crate) fn own(_: &Owned) -> Result<u32> {
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
    pub(crate) fn wake_bind(_: u32, _: &str) -> Result<Option<Owned>> {
        Err(later())
    }
    pub(crate) fn wake_sender() -> Result<Owned> {
        Err(later())
    }
    pub(crate) fn wake(_: &Owned, _: u32, _: &str) {}
    pub(crate) fn wake_clear(_: &Owned) -> bool {
        false
    }
    pub(crate) fn wake_await(_: &Owned) -> Result<()> {
        Err(later())
    }
    pub(crate) fn await_event(_: &Owned) -> Result<()> {
        Err(later())
    }
    pub(crate) fn listen(_: u32, _: &str) -> Result<Option<Listener>> {
        Err(later())
    }
    pub(crate) fn incoming(listener: Listener) -> std::iter::Empty<Stream> {
        match listener {}
    }
    pub(crate) fn serve_all(listener: Listener, _: &std::sync::atomic::AtomicBool, _: impl FnMut(&[u8]) -> Vec<u8>) {
        match listener {}
    }
    pub(crate) fn connect(_: u32, _: &str) -> Result<Option<Stream>> {
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
    pub(crate) fn close(s: &Stream) {
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

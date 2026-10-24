//! Linux: the table in `/dev/shm`, OFD locks, eventfd, abstract sockets, memfd pages.
#![allow(unsafe_code)]
use super::clean;
use super::unix::{Owned, Stream};
use crate::shm::Region;
use crate::{Error, Result};
use std::os::fd::{AsFd, AsRawFd};
use std::os::linux::net::SocketAddrExt;
use std::os::unix::net::{SocketAddr, UnixListener, UnixStream};

fn path(name: &str) -> String {
    format!("/dev/shm/cozy-varena-{}-{}", super::unix::euid(), clean(name))
}

/// The table named `name` for this user, created when absent.
pub fn table(name: &str) -> Result<Owned> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true).write(true).create(true);
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let path = path(name);
    Ok(options.open(&path).map_err(Error::io("open the table", &path))?.into())
}

pub fn unlink(name: &str) {
    let _ = std::fs::remove_file(path(name));
}

/// A new description of the same file: its locks are its own.
pub fn reopen(f: &Owned, write: bool) -> Result<Owned> {
    crate::shm::reopen(f.as_fd(), write)
}

/// Takes (or, `wait`, waits for) a write lock on a byte range of this description.
pub fn lock(f: &Owned, start: u64, len: u64, wait: bool) -> Result<bool> {
    crate::shm::region_lock(f.as_fd(), start as usize, len as usize, Region::Write, wait)
}

/// Takes (or waits for) a read lock: many readers, no writer.
pub fn share(f: &Owned, start: u64, len: u64, wait: bool) -> Result<bool> {
    crate::shm::region_lock(f.as_fd(), start as usize, len as usize, Region::Read, wait)
}

pub fn unlock(f: &Owned, start: u64, len: u64) -> Result<()> {
    crate::shm::region_lock(f.as_fd(), start as usize, len as usize, Region::Unlock, false).map(|_| ())
}

/// Whether another description holds a lock over the range.
pub fn held(f: &Owned, start: u64, len: u64) -> Result<bool> {
    Ok(crate::shm::region_holder(f.as_fd(), start as usize, len as usize)?.is_some())
}

/// The GPU's wake endpoint: an abstract datagram socket every participant signals by name.
pub struct Endpoint(Owned);

impl Endpoint {
    /// Binds endpoint `name` in user `ns`'s namespace; `None` while another process holds it.
    pub fn bind(ns: u32, name: &str) -> Result<Option<Endpoint>> {
        match std::os::unix::net::UnixDatagram::bind_addr(&address(ns, &format!("wake-{name}"))?) {
            Ok(s) => {
                s.set_nonblocking(true)?;
                // Each datagram carries its sender's credentials.
                let one: libc::c_int = 1;
                // SAFETY: an int option on a socket we hold.
                if unsafe { libc::setsockopt(s.as_raw_fd(), libc::SOL_SOCKET, libc::SO_PASSCRED, (&one as *const libc::c_int).cast(), std::mem::size_of::<libc::c_int>() as libc::socklen_t) } != 0 {
                    return Err(Error::os("wake credentials")(std::io::Error::last_os_error()));
                }
                Ok(Some(Endpoint(s.into())))
            }
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => Ok(None),
            Err(e) => Err(Error::os("bind a wake")(e)),
        }
    }

    /// Readable when signalled ([`await_event`] waits on it).
    pub fn handle(&self) -> &Owned {
        &self.0
    }

    /// Blocks until a uid `accept` takes signals it, clearing every signal so far.
    pub fn wait(&self, accept: impl Fn(u32) -> bool) -> Result<()> {
        loop {
            let mut poll = libc::pollfd { fd: self.handle().as_raw_fd(), events: libc::POLLIN, revents: 0 };
            // SAFETY: one live descriptor.
            if unsafe { libc::poll(&mut poll, 1, -1) } < 0 {
                let e = std::io::Error::last_os_error();
                if e.kind() != std::io::ErrorKind::Interrupted {
                    return Err(Error::os("wait for a wake")(e));
                }
            }
            if self.clear(&accept) {
                return Ok(());
            }
        }
    }

    /// Clears every signal so far; true when one came from a uid `accept` takes.
    pub fn clear(&self, accept: impl Fn(u32) -> bool) -> bool {
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
            if unsafe { libc::recvmsg(self.0.as_raw_fd(), &mut msg, libc::MSG_DONTWAIT) } < 0 {
                return any;
            }
            // SAFETY: the kernel filled the control buffer; the header walks only within it.
            let cmsg = unsafe { libc::CMSG_FIRSTHDR(&msg) };
            // SAFETY: a credentials message of the size the kernel wrote.
            let uid = (!cmsg.is_null() && unsafe { (*cmsg).cmsg_type } == libc::SCM_CREDENTIALS).then(|| unsafe { std::ptr::read_unaligned(libc::CMSG_DATA(cmsg) as *const libc::ucred).uid });
            any |= uid.is_some_and(&accept);
        }
    }
}

/// A participant's sending end.
pub struct Notifier(std::os::unix::net::UnixDatagram);

impl Notifier {
    pub fn new() -> Result<Notifier> {
        let s = std::os::unix::net::UnixDatagram::unbound()?;
        s.set_nonblocking(true)?;
        Ok(Notifier(s))
    }

    /// Signals endpoint `name`; nobody bound and a full queue (a signal pending already) are no
    /// signal needed.
    pub fn notify(&self, ns: u32, name: &str) {
        if let Ok(addr) = address(ns, &format!("wake-{name}")) {
            let _ = self.0.send_to_addr(&[1], &addr);
        }
    }
}

/// A name in user `ns`'s namespace (the table owner's: a granted executor finds its TensorD's
/// services there).
fn address(ns: u32, name: &str) -> Result<SocketAddr> {
    Ok(SocketAddr::from_abstract_name(format!("cozy-varena-{ns}-{name}"))?)
}

pub type Listener = UnixListener;

/// Binds the service name: `None` while another process holds it. The bound name is the
/// service lock, released by the kernel when its holder dies.
pub fn listen(ns: u32, name: &str) -> Result<Option<Listener>> {
    match UnixListener::bind_addr(&address(ns, name)?) {
        Ok(l) => Ok(Some(l)),
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => Ok(None),
        Err(e) => Err(Error::os("bind a service")(e)),
    }
}

pub fn accept(listener: &Listener) -> Result<Stream> {
    Ok(listener.accept()?.0)
}

pub fn serve_all(listener: Listener, stop: &std::sync::atomic::AtomicBool, accept: impl Fn(u32) -> bool, answer: impl Fn(&[u8]) -> Vec<u8> + Send + Sync + 'static) {
    super::unix::serve_listener(&listener, stop, accept, answer)
}

/// Connects to the service; `None` when nobody serves it.
pub fn connect(ns: u32, name: &str) -> Result<Option<Stream>> {
    match UnixStream::connect_addr(&address(ns, name)?) {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.kind() == std::io::ErrorKind::ConnectionRefused => Ok(None),
        Err(e) => Err(Error::os("connect a service")(e)),
    }
}

/// A shareable host page of `len` bytes.
pub fn page(len: u64) -> Result<Owned> {
    let fd = crate::shm::memfd_named("varena-page")?;
    super::unix::resize(&fd, len)?;
    Ok(fd)
}

/// A host page of `len` bytes the kernel can write back and drop under memory pressure: an
/// unlinked regular file in `dir` (a memfd, like any shared anonymous memory, pages only to
/// swap, so a capped cgroup is OOM-killed instead). Its disk is reserved now: a full disk is
/// `ENOSPC` here, never a SIGBUS later.
pub fn spill_page(dir: &std::path::Path, len: u64) -> Result<Owned> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut options = std::fs::OpenOptions::new();
    options.read(true).write(true).mode(0o600);
    let file = match options.clone().custom_flags(libc::O_TMPFILE).open(dir) {
        Ok(file) => file,
        // A filesystem with no O_TMPFILE: a fresh name, unlinked at once.
        Err(e) if matches!(e.raw_os_error(), Some(libc::EOPNOTSUPP | libc::EISDIR | libc::EINVAL)) => {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let path = dir.join(format!("varena-{}-{}", std::process::id(), NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
            let file = options.create_new(true).open(&path).map_err(Error::os("create a spill page"))?;
            std::fs::remove_file(&path).map_err(Error::os("unlink a spill page"))?;
            file
        }
        Err(e) => return Err(Error::os("create a spill page")(e)),
    };
    // SAFETY: reserving blocks of a file this function opened.
    let errno = unsafe { libc::posix_fallocate(file.as_raw_fd(), 0, len as libc::off_t) };
    if errno != 0 {
        return Err(Error::os("reserve a spill page")(std::io::Error::from_raw_os_error(errno)));
    }
    Ok(file.into())
}

/// Starts writing a spill page's dirty bytes back, so the kernel can drop them at once under
/// pressure: a fill dirties faster than a capped cgroup's writeback cleans.
pub fn write_back(page: &Owned, len: u64) -> Result<()> {
    // SAFETY: an asynchronous writeback request on a descriptor the caller holds.
    if unsafe { libc::sync_file_range(page.as_raw_fd(), 0, len as libc::off64_t, libc::SYNC_FILE_RANGE_WRITE) } != 0 {
        return Err(Error::os("write a spill page back")(std::io::Error::last_os_error()));
    }
    Ok(())
}

/// A read-only description of a page: what another process imports.
pub fn export(page: &Owned) -> Result<Owned> {
    crate::shm::reopen(page.as_fd(), false)
}

/// The effective uid of `pid`, from `/proc`.
pub fn uid_of(pid: u32) -> Result<u32> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status"))?;
    let uids = status.lines().find_map(|l| l.strip_prefix("Uid:"));
    uids.and_then(|u| u.split_whitespace().nth(1)).and_then(|u| u.parse().ok()).ok_or_else(|| Error::os("read a uid")(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("no uid for {pid}"))))
}

/// The calling thread's id.
pub fn thread_id() -> u32 {
    // SAFETY: gettid has no arguments and cannot fail.
    unsafe { libc::syscall(libc::SYS_gettid) as u32 }
}

/// Thread `tid` of `pid` cannot run: it is stopped (`T`; a traced `t` never counts), or its
/// process is a zombie. A thread or process that is gone is not stalled (its row reads dead).
pub fn stalled(pid: u32, tid: u32) -> bool {
    // The state follows the parenthesised name, which may itself hold spaces and ')'.
    let state = |path: String| std::fs::read_to_string(path).ok().and_then(|s| s.rsplit_once(')').and_then(|(_, rest)| rest.split_whitespace().next().map(str::to_string)));
    let process = state(format!("/proc/{pid}/stat"));
    let thread = state(format!("/proc/{pid}/task/{tid}/stat"));
    process.as_deref() == Some("Z") || thread.as_deref() == Some("T")
}

/// `f` is closing: its OFD locks go with its description's last descriptor, a duplicate's
/// included, so nothing is released here.
pub fn closing(_: &Owned) {}

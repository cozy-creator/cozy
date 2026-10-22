//! macOS: the table, its range locks and the service sockets as files in a private directory
//! under `/tmp` (`flock` on a file per range: macOS has no OFD byte-range locks); a datagram
//! socket pair as the wake event; anonymous `shm_open` objects as host pages.
use super::clean;
use super::unix::{Owned, Stream};
use crate::{Error, Result, Seam};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::ffi::CStr;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// `/tmp/cozy-varena-<ns>`, user `ns`'s alone (the table owner's: a granted executor finds its
/// TensorD's services there). `/tmp` rather than `$TMPDIR`: a socket path must fit `sun_path`
/// (104 bytes).
fn dir_of(ns: u32) -> Result<PathBuf> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt};
    let dir = PathBuf::from(format!("/tmp/cozy-varena-{ns}"));
    match std::fs::DirBuilder::new().mode(0o700).create(&dir) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(Error::io("create", &dir)(e)),
    }
    let meta = std::fs::symlink_metadata(&dir).map_err(Error::io("stat", &dir))?;
    if !meta.is_dir() || meta.uid() != ns || meta.mode() & 0o077 != 0 {
        return Err(Error::io("own", &dir)(std::io::Error::new(std::io::ErrorKind::PermissionDenied, "not this user's private directory")));
    }
    Ok(dir)
}

fn dir() -> Result<PathBuf> {
    dir_of(super::unix::euid())
}

fn open(path: &std::path::Path) -> Result<Owned> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true).write(true).create(true);
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    Ok(options.open(path).map_err(Error::io("open", path))?.into())
}

/// The table named `name` for this user, created when absent.
pub fn table(name: &str) -> Result<Owned> {
    open(&dir()?.join(format!("{}.table", clean(name))))
}

pub fn unlink(name: &str) {
    if let Ok(dir) = dir() {
        let _ = std::fs::remove_file(dir.join(format!("{}.table", clean(name))));
    }
}

/// The file behind a descriptor.
fn path_of(f: &Owned) -> Result<PathBuf> {
    let mut buf = [0 as libc::c_char; libc::MAXPATHLEN as usize];
    // SAFETY: F_GETPATH writes at most MAXPATHLEN bytes, NUL-terminated.
    if unsafe { libc::fcntl(f.as_raw_fd(), libc::F_GETPATH, buf.as_mut_ptr()) } != 0 {
        return Err(Error::os("name a descriptor")(std::io::Error::last_os_error()));
    }
    // SAFETY: NUL-terminated by the call above.
    let path = unsafe { CStr::from_ptr(buf.as_ptr()) };
    Ok(PathBuf::from(path.to_string_lossy().into_owned()))
}

/// A new description of the same file: its locks are its own.
pub fn reopen(f: &Owned) -> Result<Owned> {
    open(&path_of(f)?)
}

/// The lock files this process holds, by (description, range start). A lock goes when its
/// file closes: on `unlock`, or with the process.
fn held_locks() -> &'static Mutex<HashMap<(i32, u64), OwnedFd>> {
    static LOCKS: OnceLock<Mutex<HashMap<(i32, u64), OwnedFd>>> = OnceLock::new();
    LOCKS.get_or_init(Default::default)
}

fn lock_file(f: &Owned, start: u64) -> Result<Owned> {
    let mut path = path_of(f)?.into_os_string();
    path.push(format!(".{start}.lock"));
    open(std::path::Path::new(&path))
}

/// `flock` on `file`: Ok(false) when another description holds it and `wait` is false.
fn flock(file: &Owned, kind: libc::c_int, wait: bool) -> Result<bool> {
    let op = if wait { kind } else { kind | libc::LOCK_NB };
    loop {
        // SAFETY: a lock on a descriptor we hold.
        if unsafe { libc::flock(file.as_raw_fd(), op) } == 0 {
            return Ok(true);
        }
        let e = std::io::Error::last_os_error();
        match e.raw_os_error() {
            Some(libc::EINTR) => continue,
            Some(libc::EWOULDBLOCK) if !wait => return Ok(false),
            _ => return Err(Error::os("lock")(e)),
        }
    }
}

fn take(f: &Owned, start: u64, kind: libc::c_int, wait: bool) -> Result<bool> {
    let file = lock_file(f, start)?;
    if !flock(&file, kind, wait)? {
        return Ok(false);
    }
    held_locks().lock().unwrap_or_else(|e| e.into_inner()).insert((f.as_raw_fd(), start), file);
    Ok(true)
}

/// Takes (or, `wait`, waits for) the write lock on the range starting at `start`.
pub fn lock(f: &Owned, start: u64, _len: u64, wait: bool) -> Result<bool> {
    take(f, start, libc::LOCK_EX, wait)
}

/// Takes (or waits for) a read lock: many readers, no writer.
pub fn share(f: &Owned, start: u64, _len: u64, wait: bool) -> Result<bool> {
    take(f, start, libc::LOCK_SH, wait)
}

pub fn unlock(f: &Owned, start: u64, _len: u64) -> Result<()> {
    held_locks().lock().unwrap_or_else(|e| e.into_inner()).remove(&(f.as_raw_fd(), start));
    Ok(())
}

/// Whether another description holds a lock on the range: a lock this one can take is free.
pub fn held(f: &Owned, start: u64, _len: u64) -> Result<bool> {
    let probe = lock_file(f, start)?;
    Ok(!flock(&probe, libc::LOCK_EX, false)?)
}

/// A local event: (the reader, the writer) of a datagram socket pair.
pub fn event() -> Result<(Owned, Owned)> {
    let mut fds = [0; 2];
    // SAFETY: a socket pair into our array.
    if unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_DGRAM, 0, fds.as_mut_ptr()) } != 0 {
        return Err(Error::os("socketpair")(std::io::Error::last_os_error()));
    }
    // SAFETY: two descriptors we just created.
    let (reader, writer) = unsafe { (Owned::from_raw_fd(fds[0]), Owned::from_raw_fd(fds[1])) };
    for fd in [&reader, &writer] {
        // SAFETY: flags on descriptors we hold.
        unsafe {
            libc::fcntl(fd.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC);
            libc::fcntl(fd.as_raw_fd(), libc::F_SETFL, libc::O_NONBLOCK);
        }
    }
    Ok((reader, writer))
}

pub fn signal(writer: &Owned) {
    let one = [1u8];
    // SAFETY: a one-byte datagram; a full buffer is a signal already.
    unsafe { libc::send(writer.as_raw_fd(), one.as_ptr().cast(), 1, libc::MSG_DONTWAIT) };
}

/// Blocks until the event is signalled, then clears it.
pub fn await_event(reader: &Owned) -> Result<()> {
    let mut poll = libc::pollfd { fd: reader.as_raw_fd(), events: libc::POLLIN, revents: 0 };
    loop {
        // SAFETY: one live descriptor.
        if unsafe { libc::poll(&mut poll, 1, -1) } > 0 {
            let mut buf = [0u8; 64];
            // SAFETY: datagram reads into our buffer until none is left.
            while unsafe { libc::recv(reader.as_raw_fd(), buf.as_mut_ptr().cast(), buf.len(), libc::MSG_DONTWAIT) } > 0 {}
            return Ok(());
        }
        let e = std::io::Error::last_os_error();
        if e.kind() != std::io::ErrorKind::Interrupted {
            return Err(Error::os("wait for a wake")(e));
        }
    }
}

/// A service: its socket and the lock that makes it the service's.
pub struct Listener {
    socket: UnixListener,
    _lock: Owned,
}

/// The socket and lock files of a service; hashed, so the socket path fits `sun_path`.
fn service(ns: u32, name: &str) -> Result<(PathBuf, PathBuf)> {
    let short = &hex::encode(Sha256::digest(name.as_bytes()))[..24];
    let dir = dir_of(ns)?;
    Ok((dir.join(format!("{short}.sock")), dir.join(format!("{short}.lock"))))
}

/// Takes a service's lock file: `None` while another process holds it.
fn hold(lock: &std::path::Path) -> Result<Option<Owned>> {
    let lock = open(lock)?;
    Ok(flock(&lock, libc::LOCK_EX, false)?.then_some(lock))
}

/// Removes a socket file its dead holder left.
fn clear(socket: &std::path::Path) -> Result<()> {
    match std::fs::remove_file(socket) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(Error::io("remove a stale socket", socket)(e)),
    }
}

/// The GPU's wake endpoint: a datagram socket file every participant signals by name, its
/// lock file held while bound.
pub struct Endpoint {
    socket: Owned,
    _lock: Owned,
}

impl Endpoint {
    /// Binds endpoint `name` in user `ns`'s directory; `None` while another process holds it.
    pub fn bind(ns: u32, name: &str) -> Result<Option<Endpoint>> {
        let (socket, lock) = service(ns, &format!("wake-{name}"))?;
        let Some(lock) = hold(&lock)? else { return Ok(None) };
        clear(&socket)?;
        let s = std::os::unix::net::UnixDatagram::bind(&socket).map_err(Error::io("bind", &socket))?;
        s.set_nonblocking(true)?;
        Ok(Some(Endpoint { socket: s.into(), _lock: lock }))
    }

    pub fn handle(&self) -> &Owned {
        &self.socket
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

    /// Clears every signal so far; true when one came. Only its user can reach the socket file
    /// (a 0700 directory), so every signal is accepted.
    pub fn clear(&self, _accept: impl Fn(u32) -> bool) -> bool {
        let mut buf = [0u8; 64];
        let mut any = false;
        // SAFETY: nonblocking reads into our buffer until the queue is empty.
        while unsafe { libc::recv(self.socket.as_raw_fd(), buf.as_mut_ptr().cast(), buf.len(), libc::MSG_DONTWAIT) } > 0 {
            any = true;
        }
        any
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

    /// Signals endpoint `name`; nobody bound and a full queue are no signal needed.
    pub fn notify(&self, ns: u32, name: &str) {
        if let Ok((socket, _)) = service(ns, &format!("wake-{name}")) {
            let _ = self.0.send_to(&[1], socket);
        }
    }
}

/// Takes the service's lock and binds its socket: `None` while another process holds it. The
/// lock dies with its holder; a socket file it left behind is replaced.
pub fn listen(ns: u32, name: &str) -> Result<Option<Listener>> {
    let (socket, lock) = service(ns, name)?;
    let Some(lock) = hold(&lock)? else { return Ok(None) };
    clear(&socket)?;
    let socket = UnixListener::bind(&socket).map_err(Error::io("bind", &socket))?;
    Ok(Some(Listener { socket, _lock: lock }))
}

pub fn accept(listener: &Listener) -> Result<Stream> {
    Ok(listener.socket.accept()?.0)
}

pub fn serve_all(listener: Listener, stop: &std::sync::atomic::AtomicBool, accept: impl Fn(u32) -> bool, answer: impl Fn(&[u8]) -> Vec<u8> + Send + Sync + 'static) {
    super::unix::serve_listener(&listener.socket, stop, accept, answer)
}

/// Connects to the service; `None` when nobody serves it.
pub fn connect(ns: u32, name: &str) -> Result<Option<Stream>> {
    let (socket, _) = service(ns, name)?;
    match UnixStream::connect(&socket) {
        Ok(s) => Ok(Some(s)),
        Err(e) if matches!(e.kind(), std::io::ErrorKind::ConnectionRefused | std::io::ErrorKind::NotFound) => Ok(None),
        Err(e) => Err(Error::io("connect", &socket)(e)),
    }
}

/// A shareable host page of `len` bytes: a `shm_open` object, unlinked at once.
pub fn page(len: u64) -> Result<Owned> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    loop {
        // shm names are at most 31 bytes (PSHMNAMLEN).
        let name = format!("/cv{}.{}\0", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed));
        // SAFETY: a NUL-terminated name; the mode is passed as the variadic argument.
        let fd = unsafe { libc::shm_open(name.as_ptr().cast(), libc::O_RDWR | libc::O_CREAT | libc::O_EXCL, 0o600 as libc::c_uint) };
        if fd < 0 {
            let e = std::io::Error::last_os_error();
            if e.raw_os_error() == Some(libc::EEXIST) {
                continue;
            }
            return Err(Error::os("shm_open")(e));
        }
        // SAFETY: a descriptor we just created; its name goes now, the object with its last user.
        let fd = unsafe {
            libc::shm_unlink(name.as_ptr().cast());
            Owned::from_raw_fd(fd)
        };
        super::unix::resize(&fd, len)?;
        return Ok(fd);
    }
}

/// What another process imports: macOS cannot reopen a shm object read-only, so a duplicate;
/// the importer maps it read-only.
pub fn export(page: &Owned) -> Result<Owned> {
    Ok(page.try_clone()?)
}

pub fn tier_file(_: &str) -> Result<Owned> {
    Err(Error::unsupported(Seam::SharedMemory))
}

pub fn punch(_: &Owned, _: u64, _: u64) -> Result<()> {
    Err(Error::unsupported(Seam::SharedMemory))
}

pub fn allocated(_: &Owned) -> u64 {
    0
}

/// The uid of `pid` (its real uid, as the kernel reports it).
pub fn uid_of(pid: u32) -> Result<u32> {
    // SAFETY: plain-old-data the call fills.
    let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_bsdinfo>() as libc::c_int;
    // SAFETY: a buffer of the size we pass.
    let got = unsafe { libc::proc_pidinfo(pid as libc::c_int, libc::PROC_PIDTBSDINFO, 0, (&mut info as *mut libc::proc_bsdinfo).cast(), size) };
    if got != size {
        return Err(Error::os("read a uid")(std::io::Error::last_os_error()));
    }
    Ok(info.pbi_uid)
}

/// A process's anonymous high-water is measured on Linux only (`/proc`): here a call is
/// unmeasured, and the RAM tier (Linux's) is absent anyway.
pub fn high_water() -> Option<u64> {
    None
}

pub fn reset_high_water() {}

pub fn anon(_: u32) -> Option<u64> {
    None
}

/// The calling thread's id (its low 32 bits).
pub fn thread_id() -> u32 {
    let mut id = 0u64;
    // SAFETY: the calling thread's id into our variable.
    unsafe { libc::pthread_threadid_np(0, &mut id) };
    id as u32
}

/// `pid` cannot run: it is stopped (a stop stops every thread) or a zombie.
pub fn stalled(pid: u32, _tid: u32) -> bool {
    // SAFETY: plain-old-data the call fills.
    let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_bsdinfo>() as libc::c_int;
    // SAFETY: a buffer of the size we pass.
    let got = unsafe { libc::proc_pidinfo(pid as libc::c_int, libc::PROC_PIDTBSDINFO, 0, (&mut info as *mut libc::proc_bsdinfo).cast(), size) };
    got == size && matches!(info.pbi_status, libc::SSTOP | libc::SZOMB)
}

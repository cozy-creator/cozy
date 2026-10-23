//! What Linux and macOS share: mappings, sizes and host pages in reserved address space.
#![allow(unsafe_code)]
use super::Map;
use crate::{Error, Result};
use std::os::fd::{AsRawFd, OwnedFd};

pub type Owned = OwnedFd;

impl Drop for Map {
    fn drop(&mut self) {
        // SAFETY: our own mapping; nothing borrows it past its owner.
        unsafe { libc::munmap(self.base.cast(), self.len) };
    }
}

fn prot(write: bool) -> libc::c_int {
    if write { libc::PROT_READ | libc::PROT_WRITE } else { libc::PROT_READ }
}

pub fn size(f: &Owned) -> Result<u64> {
    let file = std::fs::File::from(f.try_clone()?);
    Ok(file.metadata()?.len())
}

pub fn resize(f: &Owned, len: u64) -> Result<()> {
    // SAFETY: sizing a file we hold.
    if unsafe { libc::ftruncate(f.as_raw_fd(), len as libc::off_t) } != 0 {
        return Err(Error::os("resize")(std::io::Error::last_os_error()));
    }
    Ok(())
}

/// `len` bytes of `f` from `offset` (a multiple of [`granularity`]).
pub fn map(f: &Owned, offset: u64, len: usize, write: bool) -> Result<Map> {
    // SAFETY: a new shared mapping of a file we hold.
    let base = unsafe { libc::mmap(std::ptr::null_mut(), len, prot(write), libc::MAP_SHARED, f.as_raw_fd(), offset as libc::off_t) };
    if base == libc::MAP_FAILED {
        return Err(Error::os("map")(std::io::Error::last_os_error()));
    }
    Ok(Map { base: base.cast(), len })
}

/// Mapping offsets and host pages are multiples of this.
pub fn granularity() -> u64 {
    crate::shm::page_size() as u64
}

/// Address space for host pages: nothing mapped, nothing committed.
pub fn reserve(len: u64) -> Result<u64> {
    let flags = libc::MAP_PRIVATE | libc::MAP_ANON;
    // SAFETY: a new anonymous inaccessible mapping; nothing aliases it.
    let base = unsafe { libc::mmap(std::ptr::null_mut(), len as usize, libc::PROT_NONE, flags, -1, 0) };
    if base == libc::MAP_FAILED {
        return Err(Error::os("reserve")(std::io::Error::last_os_error()));
    }
    Ok(base as u64)
}

pub fn release(base: u64, len: u64) {
    // SAFETY: a reservation [`reserve`] made.
    unsafe { libc::munmap(base as *mut libc::c_void, len as usize) };
}

/// Maps a host page at `at`, inside a reservation.
pub fn map_page(at: u64, page: &Owned, len: u64, write: bool) -> Result<()> {
    let flags = libc::MAP_SHARED | libc::MAP_FIXED;
    // SAFETY: a fixed shared mapping inside the caller's reservation.
    if unsafe { libc::mmap(at as *mut libc::c_void, len as usize, prot(write), flags, page.as_raw_fd(), 0) } == libc::MAP_FAILED {
        return Err(Error::os("map a page")(std::io::Error::last_os_error()));
    }
    Ok(())
}

pub fn protect(at: u64, len: u64, write: bool) -> Result<()> {
    // SAFETY: a range of the caller's reservation.
    if unsafe { libc::mprotect(at as *mut libc::c_void, len as usize, prot(write)) } != 0 {
        return Err(Error::os("protect a page")(std::io::Error::last_os_error()));
    }
    Ok(())
}

/// The range goes back to reserved and inaccessible.
pub fn unmap_page(at: u64, len: u64) -> Result<()> {
    let flags = libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED;
    // SAFETY: replaces a range of the caller's reservation with inaccessible memory.
    if unsafe { libc::mmap(at as *mut libc::c_void, len as usize, libc::PROT_NONE, flags, -1, 0) } == libc::MAP_FAILED {
        return Err(Error::os("unmap a page")(std::io::Error::last_os_error()));
    }
    Ok(())
}

/// The table's owner: this user, or root (a TensorD running as root grants its tables to
/// executors of another uid). Any other owner, or a file others may write, is refused.
pub fn own(f: &Owned) -> Result<u32> {
    use std::os::unix::fs::MetadataExt;
    let meta = std::fs::File::from(f.try_clone()?).metadata()?;
    if (meta.uid() != euid() && meta.uid() != 0) || meta.mode() & 0o022 != 0 {
        return Err(Error::os("own the table")(std::io::Error::new(std::io::ErrorKind::PermissionDenied, "the shared table belongs to someone else")));
    }
    Ok(meta.uid())
}

pub fn euid() -> u32 {
    // SAFETY: geteuid reads a process attribute.
    unsafe { libc::geteuid() }
}

/// The peer: its pid and uid, whoever it runs as (the caller decides whom it accepts).
pub fn peer(stream: &Stream) -> Result<(u32, u32)> {
    let p = crate::handles::peer(stream)?;
    Ok((p.pid, p.uid))
}

pub type Stream = std::os::unix::net::UnixStream;

/// What another process maps read and write: another descriptor of the page (sealing, where
/// the OS has it, keeps its size).
pub fn lend(page: &Owned) -> Result<Owned> {
    Ok(page.try_clone()?)
}

/// Whether the file still has a name.
pub fn linked(f: &Owned) -> Result<bool> {
    use std::os::unix::fs::MetadataExt;
    Ok(std::fs::File::from(f.try_clone()?).metadata()?.nlink() > 0)
}

/// Serves every connection to `listener` at once (one poll over all) until `stop` is seen:
/// each connection's one message is answered with `answer`'s bytes and the connection closed.
/// A peer whose uid `accept` refuses is dropped at accept; a silent or stopped one holds nobody
/// up.
pub(super) fn serve_listener(listener: &std::os::unix::net::UnixListener, stop: &std::sync::atomic::AtomicBool, accept: impl Fn(u32) -> bool, answer: impl Fn(&[u8]) -> Vec<u8>) {
    use std::io::{Read, Write};
    use std::os::fd::AsRawFd;
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
        if stop.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        let ready: Vec<bool> = polls.iter().map(|p| p.revents != 0).collect();
        let mut next = vec![];
        for (i, (s, mut buf)) in conns.drain(..).enumerate() {
            if !ready.get(i + 1).copied().unwrap_or(false) {
                next.push((s, buf));
                continue;
            }
            let mut chunk = [0u8; 512];
            let open = loop {
                match (&s).read(&mut chunk) {
                    Ok(0) => break false,
                    Ok(n) => buf.extend_from_slice(chunk.get(..n).unwrap_or_default()),
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break true,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(_) => break false,
                }
            };
            match super::frame(&buf) {
                super::Frame::Whole(body) => {
                    let _ = (&s).write_all(&answer(body));
                }
                super::Frame::Partial if open => next.push((s, buf)),
                super::Frame::Partial | super::Frame::Bad => {}
            }
        }
        if ready.first().copied().unwrap_or(false) {
            while let Ok((s, _)) = listener.accept() {
                if peer(&s).is_ok_and(|(_, uid)| accept(uid)) && s.set_nonblocking(true).is_ok() {
                    next.push((s, vec![]));
                }
            }
        }
        conns = next;
    }
}

/// What [`wait_any`] waits on: a descriptor that turns readable.
pub type Waitable<'a> = std::os::fd::BorrowedFd<'a>;

/// A local wake one process sets and its own threads wait on: a socket pair whose read end is
/// readable from a set until it is cleared.
pub struct Signal {
    read: std::os::unix::net::UnixStream,
    write: std::os::unix::net::UnixStream,
}

impl Signal {
    pub fn new() -> Result<Signal> {
        let (read, write) = std::os::unix::net::UnixStream::pair()?;
        read.set_nonblocking(true)?;
        write.set_nonblocking(true)?;
        Ok(Signal { read, write })
    }

    /// A full pair is set already: the byte that did not fit is not needed.
    pub fn set(&self) {
        let _ = std::io::Write::write(&mut &self.write, &[1]);
    }

    /// Clears it; true when it was set.
    pub fn clear(&self) -> bool {
        let mut buf = [0u8; 64];
        let mut set = false;
        while let Ok(1..) = std::io::Read::read(&mut &self.read, &mut buf) {
            set = true;
        }
        set
    }

    pub fn waitable(&self) -> Waitable<'_> {
        std::os::fd::AsFd::as_fd(&self.read)
    }
}

/// Blocks until any of `items` is readable, hung up or failed, then answers every one that is, in
/// order, so none starves behind a lower one. No timeout.
pub fn wait_any(items: &[Waitable<'_>]) -> Result<Vec<super::Ready>> {
    let mut fds: Vec<libc::pollfd> = items.iter().map(|fd| libc::pollfd { fd: fd.as_raw_fd(), events: libc::POLLIN, revents: 0 }).collect();
    loop {
        // SAFETY: descriptors the caller borrows for the call; no timeout.
        if unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, -1) } >= 0 {
            let ready: Vec<super::Ready> = fds
                .iter()
                .enumerate()
                .filter(|(_, fd)| fd.revents != 0)
                .map(|(index, fd)| super::Ready { index, broken: fd.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 })
                .collect();
            if !ready.is_empty() {
                return Ok(ready);
            }
            continue;
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err(Error::os("wait")(error));
        }
    }
}

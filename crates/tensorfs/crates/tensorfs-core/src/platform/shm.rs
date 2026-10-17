//! Shared memory for the plane's host tier: a sealed memfd handed between processes by
//! descriptor and claimed region by region with open-file-description locks. Linux only: off
//! Linux every call answers `Unsupported` and nothing pretends a memfd exists (a Mac's host
//! tier is its page cache; Windows' is unit WM's).
use super::Result;
use std::os::fd::{BorrowedFd, OwnedFd};

/// What makes a memfd immutable: no write, resize, punch or unsealing, by anyone. Future-write
/// (Linux 5.1) refuses new writable mappings and writes; the filler's own mapping may finish.
pub const FULL_SEALS: i32 = SEAL_SEAL | SEAL_SHRINK | SEAL_GROW | SEAL_FUTURE_WRITE;
/// What an adopter of a sealed memfd requires besides one of [`WRITE_SEALS`].
pub const RESIZE_SEALS: i32 = SEAL_SEAL | SEAL_SHRINK | SEAL_GROW;
pub const WRITE_SEALS: i32 = SEAL_WRITE | SEAL_FUTURE_WRITE;
const SEAL_SEAL: i32 = 0x1;
const SEAL_SHRINK: i32 = 0x2;
const SEAL_GROW: i32 = 0x4;
const SEAL_WRITE: i32 = 0x8;
const SEAL_FUTURE_WRITE: i32 = 0x10;

/// A byte-range lock request on an open file description.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Region {
    Read,
    Write,
    Unlock,
}

/// A new sealable memfd, close-on-exec.
pub fn memfd_named(name: &str) -> Result<OwnedFd> {
    sys::memfd_named(name)
}

/// Adds [`FULL_SEALS`].
pub fn seal(fd: BorrowedFd<'_>) -> Result<()> {
    sys::seal(fd)
}

pub fn seals(fd: BorrowedFd<'_>) -> Result<i32> {
    sys::seals(fd)
}

/// Ok(false) when a non-waiting request conflicts with another description's lock.
pub fn region_lock(fd: BorrowedFd<'_>, start: usize, len: usize, region: Region, wait: bool) -> Result<bool> {
    sys::region_lock(fd, start, len, region, wait)
}

/// The span `(start, len)` of a lock another description holds over any byte of the range
/// (`len` 0: to the end), or `None`. One query answers one lock, not necessarily the lowest.
pub fn region_holder(fd: BorrowedFd<'_>, start: usize, len: usize) -> Result<Option<(usize, usize)>> {
    sys::region_holder(fd, start, len)
}

/// Frees `[off, off+span)`'s pages; the file keeps its size.
pub fn punch(fd: BorrowedFd<'_>, off: u64, span: u64) -> Result<()> {
    sys::punch(fd, off, span)
}

/// A new open file description of the file behind `fd`: a dup would share `fd`'s, and with it
/// its locks.
pub fn reopen(fd: BorrowedFd<'_>, write: bool) -> Result<OwnedFd> {
    sys::reopen(fd, write)
}

/// Transparent huge pages for a shared mapping where shmem THP is enabled; advice only.
pub fn advise_huge(base: *mut u8, len: usize) {
    sys::advise_huge(base, len)
}

pub fn page_size() -> usize {
    // SAFETY: sysconf reads a constant.
    (unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize).max(4096)
}

/// Fault in every page of `[base, base+len)` for writing, in `threads` slices: pages a memfd
/// lacks are allocated, pages it has are only mapped. Without MADV_POPULATE_WRITE (Linux
/// before 5.14, other OSes) each page is touched instead, writing the zero it reads as.
pub fn populate(base: *mut u8, len: usize, threads: usize) {
    let page = page_size();
    let threads = threads.max(1);
    let step = len.div_ceil(threads).next_multiple_of(page);
    let base = base as usize;
    std::thread::scope(|s| {
        for k in 0..threads {
            let (lo, hi) = ((k * step).min(len), ((k + 1) * step).min(len));
            if lo == hi {
                continue;
            }
            s.spawn(move || {
                if !sys::populate_write((base + lo) as *mut u8, hi - lo) {
                    for p in (lo..hi).step_by(page) {
                        // SAFETY: a page of the caller's live, writable shared mapping.
                        unsafe { std::ptr::write_volatile((base + p) as *mut u8, 0) };
                    }
                }
            });
        }
    });
}

#[cfg(target_os = "linux")]
mod sys {
    use super::{Region, Result, FULL_SEALS};
    use std::ffi::CString;
    use std::io;
    use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd};

    pub(super) fn memfd_named(name: &str) -> Result<OwnedFd> {
        let name = CString::new(name).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        // SAFETY: plain syscall; the name is NUL-terminated.
        let fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING) };
        if fd < 0 {
            return Err(io::Error::last_os_error().into());
        }
        // SAFETY: we own the fresh descriptor.
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    pub(super) fn seal(fd: BorrowedFd<'_>) -> Result<()> {
        // SAFETY: a scalar fcntl on a descriptor the caller holds.
        if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, FULL_SEALS) } != 0 {
            return Err(io::Error::last_os_error().into());
        }
        Ok(())
    }

    pub(super) fn seals(fd: BorrowedFd<'_>) -> Result<i32> {
        // SAFETY: a scalar query of a descriptor the caller holds.
        let seals = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GET_SEALS) };
        if seals < 0 {
            return Err(io::Error::last_os_error().into());
        }
        Ok(seals)
    }

    pub(super) fn region_lock(fd: BorrowedFd<'_>, start: usize, len: usize, region: Region, wait: bool) -> Result<bool> {
        let mut fl = flock(start, len, region);
        let cmd = if wait { libc::F_OFD_SETLKW } else { libc::F_OFD_SETLK };
        // SAFETY: OFD byte-range lock on a descriptor the caller holds.
        if unsafe { libc::fcntl(fd.as_raw_fd(), cmd, &mut fl) } == 0 {
            return Ok(true);
        }
        let error = io::Error::last_os_error();
        match error.raw_os_error() {
            Some(libc::EAGAIN) | Some(libc::EACCES) if !wait => Ok(false),
            _ => Err(error.into()),
        }
    }

    pub(super) fn region_holder(fd: BorrowedFd<'_>, start: usize, len: usize) -> Result<Option<(usize, usize)>> {
        let mut fl = flock(start, len, Region::Write);
        // SAFETY: a lock query on a descriptor the caller holds.
        if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_OFD_GETLK, &mut fl) } != 0 {
            return Err(io::Error::last_os_error().into());
        }
        Ok((fl.l_type != libc::F_UNLCK as libc::c_short).then_some((fl.l_start as usize, fl.l_len as usize)))
    }

    fn flock(start: usize, len: usize, region: Region) -> libc::flock {
        libc::flock {
            l_type: match region {
                Region::Read => libc::F_RDLCK,
                Region::Write => libc::F_WRLCK,
                Region::Unlock => libc::F_UNLCK,
            } as libc::c_short,
            l_whence: libc::SEEK_SET as libc::c_short,
            l_start: start as libc::off_t,
            l_len: len as libc::off_t,
            l_pid: 0,
        }
    }

    pub(super) fn punch(fd: BorrowedFd<'_>, off: u64, span: u64) -> Result<()> {
        // SAFETY: punching a hole in a memfd the caller holds.
        let rc = unsafe {
            libc::fallocate(
                fd.as_raw_fd(),
                libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                off as libc::off_t,
                span as libc::off_t,
            )
        };
        if rc != 0 {
            return Err(io::Error::last_os_error().into());
        }
        Ok(())
    }

    pub(super) fn reopen(fd: BorrowedFd<'_>, write: bool) -> Result<OwnedFd> {
        let path = CString::new(format!("/proc/self/fd/{}", fd.as_raw_fd())).expect("no NUL");
        let access = if write { libc::O_RDWR } else { libc::O_RDONLY };
        // SAFETY: plain open of a magic link to a descriptor the caller holds.
        let own = unsafe { libc::open(path.as_ptr(), access | libc::O_CLOEXEC) };
        if own < 0 {
            return Err(io::Error::last_os_error().into());
        }
        // SAFETY: fresh descriptor from the kernel.
        Ok(unsafe { OwnedFd::from_raw_fd(own) })
    }

    pub(super) fn advise_huge(base: *mut u8, len: usize) {
        // SAFETY: advice on the caller's mapping.
        unsafe { libc::madvise(base.cast(), len, libc::MADV_HUGEPAGE) };
    }

    pub(super) fn populate_write(base: *mut u8, len: usize) -> bool {
        const MADV_POPULATE_WRITE: libc::c_int = 23;
        // SAFETY: a page-aligned slice of the caller's live, writable shared mapping.
        unsafe { libc::madvise(base.cast(), len, MADV_POPULATE_WRITE) == 0 }
    }
}

#[cfg(not(target_os = "linux"))]
mod sys {
    use super::super::{Error, Result, Seam};
    use super::Region;
    use std::os::fd::{BorrowedFd, OwnedFd};

    fn unsupported<T>() -> Result<T> {
        Err(Error::unsupported(Seam::SharedMemory))
    }

    pub(super) fn memfd_named(_: &str) -> Result<OwnedFd> {
        unsupported()
    }

    pub(super) fn seal(_: BorrowedFd<'_>) -> Result<()> {
        unsupported()
    }

    pub(super) fn seals(_: BorrowedFd<'_>) -> Result<i32> {
        unsupported()
    }

    pub(super) fn region_lock(_: BorrowedFd<'_>, _: usize, _: usize, _: Region, _: bool) -> Result<bool> {
        unsupported()
    }

    pub(super) fn region_holder(_: BorrowedFd<'_>, _: usize, _: usize) -> Result<Option<(usize, usize)>> {
        unsupported()
    }

    pub(super) fn punch(_: BorrowedFd<'_>, _: u64, _: u64) -> Result<()> {
        unsupported()
    }

    pub(super) fn reopen(_: BorrowedFd<'_>, _: bool) -> Result<OwnedFd> {
        unsupported()
    }

    pub(super) fn advise_huge(_: *mut u8, _: usize) {}

    pub(super) fn populate_write(_: *mut u8, _: usize) -> bool {
        false
    }
}

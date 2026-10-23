#![allow(unsafe_code)]
use super::{FULL_SEALS, Region};
use crate::Result;
use std::{
    ffi::CString,
    io,
    os::{
        fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd},
        unix::fs::MetadataExt,
    },
};

/// A new sealable memfd, close-on-exec. `name` shows in `/proc/<pid>/fd`.
pub fn memfd_named(name: &str) -> Result<OwnedFd> {
    let name = CString::new(name).map_err(io::Error::other)?;
    // SAFETY: valid NUL-terminated name, no borrowed output pointers.
    let fd =
        unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING) };
    if fd < 0 {
        return Err(io::Error::last_os_error().into());
    }
    // SAFETY: memfd_create returned a new descriptor that is now ours.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

pub fn seals(fd: BorrowedFd<'_>) -> Result<i32> {
    // SAFETY: live descriptor; fcntl has no pointer argument for this operation.
    let result = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GET_SEALS) };
    if result < 0 {
        return Err(io::Error::last_os_error().into());
    }
    Ok(result)
}

/// Adds [`FULL_SEALS`].
pub fn seal(fd: BorrowedFd<'_>) -> Result<()> {
    seal_with(fd, FULL_SEALS)
}

pub fn seal_with(fd: BorrowedFd<'_>, seals: i32) -> Result<()> {
    // SAFETY: live descriptor, scalar seal bitmask.
    if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, seals) } < 0 {
        return Err(io::Error::last_os_error().into());
    }
    Ok(())
}

/// Ok(false) when a non-waiting request conflicts with another description's lock.
pub fn region_lock(
    fd: BorrowedFd<'_>,
    start: usize,
    len: usize,
    region: Region,
    wait: bool,
) -> Result<bool> {
    let mut fl = flock(start, len, region);
    let cmd = if wait {
        libc::F_OFD_SETLKW
    } else {
        libc::F_OFD_SETLK
    };
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

/// The span `(start, len)` of a lock another description holds over any byte of the range
/// (`len` 0: to the end), or `None`. One query answers one lock, not necessarily the lowest.
pub fn region_holder(
    fd: BorrowedFd<'_>,
    start: usize,
    len: usize,
) -> Result<Option<(usize, usize)>> {
    let mut fl = flock(start, len, Region::Write);
    // SAFETY: a lock query on a descriptor the caller holds.
    if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_OFD_GETLK, &mut fl) } != 0 {
        return Err(io::Error::last_os_error().into());
    }
    Ok((fl.l_type != libc::F_UNLCK as libc::c_short)
        .then_some((fl.l_start as usize, fl.l_len as usize)))
}

/// Whether another description holds a write lock over any byte of the range.
pub fn written(fd: BorrowedFd<'_>, start: usize, len: usize) -> Result<bool> {
    let mut fl = flock(start, len, Region::Read);
    // SAFETY: a lock query on a descriptor the caller holds.
    if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_OFD_GETLK, &mut fl) } != 0 {
        return Err(io::Error::last_os_error().into());
    }
    Ok(fl.l_type != libc::F_UNLCK as libc::c_short)
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

/// A new open file description of the file behind `fd`: a dup would share `fd`'s, and with it
/// its locks.
pub fn reopen(fd: BorrowedFd<'_>, write: bool) -> Result<OwnedFd> {
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

/// Transparent huge pages for a shared mapping where shmem THP is enabled; advice only.
pub fn advise_huge(base: *mut u8, len: usize) {
    // SAFETY: advice on the caller's mapping.
    unsafe { libc::madvise(base.cast(), len, libc::MADV_HUGEPAGE) };
}

pub(super) fn populate_write(base: *mut u8, len: usize) -> bool {
    const MADV_POPULATE_WRITE: libc::c_int = 23;
    // SAFETY: a page-aligned slice of the caller's live, writable shared mapping.
    unsafe { libc::madvise(base.cast(), len, MADV_POPULATE_WRITE) == 0 }
}

/// `bytes` as a file no process can write or resize: a memfd sealed against writes, handed on
/// as a new read-only description.
pub fn frozen(name: &str, bytes: &[u8]) -> Result<std::fs::File> {
    use std::{io::Write, os::fd::AsFd};
    let mut file = std::fs::File::from(memfd_named(name)?);
    file.write_all(bytes)?;
    seal_with(file.as_fd(), super::NO_WRITE_SEALS)?;
    Ok(std::fs::File::from(reopen(file.as_fd(), false)?))
}

/// Whether a process still holds the file `(dev, ino)`: an open descriptor of it or a mapping.
/// Only processes this one may inspect are asked; another user's process cannot have been
/// handed this process's files.
pub fn held(dev: u64, ino: u64) -> Result<bool> {
    let device = format!("{:02x}:{:02x}", libc::major(dev), libc::minor(dev));
    let ino_text = ino.to_string();
    let processes = std::fs::read_dir("/proc").into_iter().flatten().flatten();
    for process in processes.filter(|p| {
        p.file_name()
            .to_string_lossy()
            .bytes()
            .all(|b| b.is_ascii_digit())
    }) {
        let descriptors = std::fs::read_dir(process.path().join("fd"))
            .into_iter()
            .flatten()
            .flatten();
        if descriptors
            .filter_map(|fd| std::fs::metadata(fd.path()).ok())
            .any(|m| m.dev() == dev && m.ino() == ino)
        {
            return Ok(true);
        }
        let maps = std::fs::read_to_string(process.path().join("maps")).unwrap_or_default();
        if maps.lines().any(|line| {
            let mut fields = line.split_whitespace().skip(3);
            fields.next() == Some(device.as_str()) && fields.next() == Some(ino_text.as_str())
        }) {
            return Ok(true);
        }
    }
    Ok(false)
}

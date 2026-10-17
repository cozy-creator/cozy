//! Directory claims, and who holds a POSIX record lock (SQLite's): the kernel answers a lock
//! test (`F_GETLK`), never `/proc/locks`. The test needs a descriptor on the file, and closing
//! any descriptor on an inode drops every POSIX lock this process holds on it (SQLite's own
//! included), so the probe descriptor for an inode, once opened, is never closed. Windows'
//! locks name no holder.
use super::Result;
use std::marker::PhantomData;
use std::path::Path;

/// An exclusive claim on a directory that leaves nothing in it, held until dropped on the
/// thread that took it: `flock` on the directory (Unix); a session-wide named mutex for its
/// canonical path on Windows, whose directories take no byte-range locks. Waits for a holder.
pub struct DirLock(#[allow(dead_code)] sys::DirLock, PhantomData<*const ()>);

pub fn dir(path: &Path) -> Result<DirLock> {
    Ok(DirLock(sys::dir(path)?, PhantomData))
}

/// Claims a file that is also data, exclusively and without waiting; false while another
/// handle holds it. Unix `flock` bars no read or write. A Windows lock bars every other
/// handle's I/O on what it covers, so the claim is one byte far past any data.
pub fn try_claim(file: &std::fs::File) -> Result<bool> {
    sys::try_claim(file)
}

/// Other processes holding a write lock on any byte of `path`; empty when `path` is absent.
pub fn posix_writers(path: &Path) -> Result<Vec<u32>> {
    sys::posix_writers(path)
}

#[cfg(unix)]
mod sys {
    use super::super::{durable::FileId, Error, Result};
    use std::collections::{hash_map::Entry, HashMap};
    use std::fs::File;
    use std::os::fd::AsRawFd;
    use std::path::Path;
    use std::sync::{Mutex, OnceLock};
    use std::{io, mem};

    static PROBES: OnceLock<Mutex<HashMap<FileId, File>>> = OnceLock::new();

    pub(super) fn try_claim(file: &File) -> Result<bool> {
        match file.try_lock() {
            Ok(()) => Ok(true),
            Err(std::fs::TryLockError::WouldBlock) => Ok(false),
            Err(std::fs::TryLockError::Error(error)) => Err(error.into()),
        }
    }

    /// Released when the descriptor closes.
    pub(super) struct DirLock {
        _directory: File,
    }

    pub(super) fn dir(path: &Path) -> Result<DirLock> {
        let directory = File::open(path)?;
        directory.lock()?;
        Ok(DirLock { _directory: directory })
    }

    pub(super) fn posix_writers(path: &Path) -> Result<Vec<u32>> {
        let id = match FileId::at(path) {
            Ok(id) => id,
            Err(Error::Os { source, .. }) if source.kind() == io::ErrorKind::NotFound => return Ok(vec![]),
            Err(error) => return Err(error),
        };
        let mut probes = PROBES.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner());
        let mut key = id;
        if !probes.contains_key(&id) {
            let file = File::open(path)?;
            key = FileId::of(&file)?;
            match probes.entry(key) {
                // Replaced between the stat and the open by an inode already probed: this
                // descriptor may not be closed either.
                Entry::Occupied(_) => mem::forget(file),
                Entry::Vacant(vacant) => {
                    vacant.insert(file);
                }
            }
        }
        let probe = &probes[&key];
        let mut writers = vec![];
        // Ranges still to test; a test answers one conflicting lock, maybe not the lowest.
        let mut ranges: Vec<(i64, i64)> = vec![(0, 0)];
        while let Some((start, len)) = ranges.pop() {
            // SAFETY: plain-old-data the call fills, on a descriptor the probe keeps open.
            let mut lock: libc::flock = unsafe { mem::zeroed() };
            lock.l_type = libc::F_RDLCK as _;
            lock.l_whence = libc::SEEK_SET as _;
            lock.l_start = start as _;
            lock.l_len = len as _;
            if unsafe { libc::fcntl(probe.as_raw_fd(), libc::F_GETLK, &mut lock) } != 0 {
                return Err(io::Error::last_os_error().into());
            }
            if lock.l_type == libc::F_UNLCK as libc::c_short {
                continue;
            }
            // An open-file-description lock names no process (-1).
            if lock.l_pid > 0 && !writers.contains(&(lock.l_pid as u32)) {
                writers.push(lock.l_pid as u32);
            }
            let (held, span): (i64, i64) = (lock.l_start as _, lock.l_len as _);
            if held > start {
                ranges.push((start, held - start));
            }
            if span != 0 && (len == 0 || held + span < start + len) {
                ranges.push((held + span, if len == 0 { 0 } else { start + len - held - span }));
            }
        }
        Ok(writers)
    }
}

#[cfg(windows)]
mod sys {
    use super::super::{Error, Result, Seam};
    use sha2::{Digest, Sha256};
    use std::io;
    use std::path::Path;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, WAIT_ABANDONED, WAIT_OBJECT_0};
    use windows_sys::Win32::System::Threading::{CreateMutexW, ReleaseMutex, WaitForSingleObject, INFINITE};

    pub(super) fn try_claim(file: &std::fs::File) -> Result<bool> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::ERROR_LOCK_VIOLATION;
        use windows_sys::Win32::Storage::FileSystem::{LockFileEx, LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY};
        use windows_sys::Win32::System::IO::OVERLAPPED;
        // SAFETY: plain-old-data; the byte at 2^63 - 2, which no file reaches.
        let mut at: OVERLAPPED = unsafe { std::mem::zeroed() };
        at.Anonymous.Anonymous.Offset = u32::MAX - 1;
        at.Anonymous.Anonymous.OffsetHigh = i32::MAX as u32;
        // SAFETY: a handle this File owns and an OVERLAPPED that outlives the call.
        let flags = LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY;
        if unsafe { LockFileEx(file.as_raw_handle(), flags, 0, 1, 0, &mut at) } != 0 {
            return Ok(true);
        }
        let error = io::Error::last_os_error();
        match error.raw_os_error() {
            Some(code) if code == ERROR_LOCK_VIOLATION as i32 => Ok(false),
            _ => Err(error.into()),
        }
    }

    pub(super) struct DirLock(HANDLE);

    pub(super) fn dir(path: &Path) -> Result<DirLock> {
        let canonical = std::fs::canonicalize(path)?.to_string_lossy().to_lowercase();
        let digest = Sha256::digest(canonical.as_bytes());
        let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        let name: Vec<u16> = format!("Local\\tensorfs-dir-{hex}").encode_utf16().chain([0]).collect();
        // SAFETY: a NUL-terminated name; the handle is closed on drop.
        let mutex = unsafe { CreateMutexW(std::ptr::null(), 0, name.as_ptr()) };
        if mutex.is_null() {
            return Err(io::Error::last_os_error().into());
        }
        let lock = DirLock(mutex);
        // An abandoned mutex (its holder died) is granted to the waiter.
        match unsafe { WaitForSingleObject(mutex, INFINITE) } {
            WAIT_OBJECT_0 | WAIT_ABANDONED => Ok(lock),
            _ => Err(io::Error::last_os_error().into()),
        }
    }

    impl Drop for DirLock {
        fn drop(&mut self) {
            // SAFETY: the mutex this thread took, then its handle.
            unsafe {
                ReleaseMutex(self.0);
                CloseHandle(self.0);
            }
        }
    }

    pub(super) fn posix_writers(_: &Path) -> Result<Vec<u32>> {
        Err(Error::unsupported(Seam::Process))
    }
}

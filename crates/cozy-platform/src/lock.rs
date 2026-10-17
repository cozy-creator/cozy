//! Claims the kernel holds on a file and drops when its holder dies, SIGKILL and TerminateProcess
//! included. Whether something holds a lock is answered by trying to take it, never by a pidfile
//! or `/proc/locks`. std's `File::lock` is `flock(2)` on Unix and `LockFileEx` on Windows.
//!
//! Who holds a POSIX record lock (SQLite's) is the kernel's answer to a lock test (`F_GETLK`).
//! The test needs a descriptor on the file, and closing any descriptor on an inode drops every
//! POSIX lock this process holds on it (SQLite's own included), so the probe descriptor for an
//! inode, once opened, is never closed. Windows' locks name no holder.
use crate::{Error, Result, durable};
use std::{
    fs::{self, File, OpenOptions, TryLockError},
    io,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Exclusive,
    Shared,
}

#[derive(Clone, Debug)]
pub struct LockFile {
    path: PathBuf,
}

/// A claim; dropping it erases its published record, then releases it.
#[derive(Debug)]
pub struct Held {
    holder: PathBuf,
    published: AtomicBool,
    _file: File,
}

impl LockFile {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// `None`: someone holds it.
    pub fn try_lock(&self, kind: Kind) -> Result<Option<Held>> {
        let file = self.open()?;
        let taken = match kind {
            Kind::Exclusive => file.try_lock(),
            Kind::Shared => file.try_lock_shared(),
        };
        match taken {
            Ok(()) => self.held(file, kind).map(Some),
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Error(error)) => Err(Error::io("lock", &self.path)(error)),
        }
    }

    /// Waits until the holder lets go: the kernel wakes the caller, nothing samples.
    pub fn lock(&self, kind: Kind) -> Result<Held> {
        let file = self.open()?;
        match kind {
            Kind::Exclusive => file.lock(),
            Kind::Shared => file.lock_shared(),
        }
        .map_err(Error::io("lock", &self.path))?;
        self.held(file, kind)
    }

    /// The record the holder published, read after a failed `try_lock`. A holder killed before it
    /// released leaves its record behind until the next exclusive taker erases it, so the record of
    /// a free lock names nobody.
    pub fn holder(&self) -> Result<Option<Vec<u8>>> {
        let path = self.holder_path();
        match fs::read(&path) {
            Ok(record) => Ok(Some(record)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(Error::io("read lock holder", path)(error)),
        }
    }

    fn holder_path(&self) -> PathBuf {
        let mut name = self.path.clone().into_os_string();
        name.push(".holder");
        name.into()
    }

    fn open(&self) -> Result<File> {
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        options
            .open(&self.path)
            .map_err(Error::io("open lock", &self.path))
    }

    /// An exclusive taker erases a killed predecessor's record: it names nobody now.
    fn held(&self, file: File, kind: Kind) -> Result<Held> {
        let holder = self.holder_path();
        if kind == Kind::Exclusive
            && let Err(error) = fs::remove_file(&holder)
            && error.kind() != io::ErrorKind::NotFound
        {
            return Err(Error::io("remove a stale lock holder", holder)(error));
        }
        Ok(Held {
            holder,
            published: AtomicBool::new(false),
            _file: file,
        })
    }
}

impl Held {
    /// Publishes what a caller whose `try_lock` failed may read: an address, a token, a birth. It
    /// lives beside the lock, not in it, because a Windows lock blocks other handles' reads. It
    /// describes a live holder, which a crash makes stale anyway, so it is never flushed: a
    /// holder's readiness never waits on the disk.
    pub fn publish(&self, record: &[u8]) -> Result<()> {
        durable::swap(&self.holder, record, 0o600)?;
        self.published.store(true, Ordering::Relaxed);
        Ok(())
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        if self.published.load(Ordering::Relaxed) {
            let _ = fs::remove_file(&self.holder);
        }
    }
}

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
    use crate::{Error, Result, durable::FileId};
    use std::collections::{HashMap, hash_map::Entry};
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
        Ok(DirLock {
            _directory: directory,
        })
    }

    pub(super) fn posix_writers(path: &Path) -> Result<Vec<u32>> {
        let id = match FileId::at(path) {
            Ok(id) => id,
            Err(Error::Os { source, .. }) if source.kind() == io::ErrorKind::NotFound => {
                return Ok(vec![]);
            }
            Err(error) => return Err(error),
        };
        let mut probes = PROBES
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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
                ranges.push((
                    held + span,
                    if len == 0 {
                        0
                    } else {
                        start + len - held - span
                    },
                ));
            }
        }
        Ok(writers)
    }
}

#[cfg(windows)]
mod sys {
    use crate::{Error, Result, Seam};
    use sha2::{Digest, Sha256};
    use std::io;
    use std::path::Path;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, WAIT_ABANDONED, WAIT_OBJECT_0};
    use windows_sys::Win32::System::Threading::{
        CreateMutexW, INFINITE, ReleaseMutex, WaitForSingleObject,
    };

    pub(super) fn try_claim(file: &std::fs::File) -> Result<bool> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::ERROR_LOCK_VIOLATION;
        use windows_sys::Win32::Storage::FileSystem::{
            LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY, LockFileEx,
        };
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
        let canonical = std::fs::canonicalize(path)?
            .to_string_lossy()
            .to_lowercase();
        let digest = hex::encode(Sha256::digest(canonical.as_bytes()));
        let name: Vec<u16> = format!("Local\\cozy-dir-{digest}")
            .encode_utf16()
            .chain([0])
            .collect();
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

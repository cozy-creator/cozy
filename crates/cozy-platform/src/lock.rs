//! Claims the kernel holds on a file and drops when its holder dies, SIGKILL and TerminateProcess
//! included. Whether something holds a lock is answered by trying to take it, never by a pidfile
//! or `/proc/locks`. std's `File::lock` is `flock(2)` on Unix and `LockFileEx` on Windows.
use crate::{Error, Result, durable};
use std::{
    fs::{self, File, OpenOptions, TryLockError},
    io,
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
    /// lives beside the lock, not in it, because a Windows lock blocks other handles' reads.
    pub fn publish(&self, record: &[u8]) -> Result<()> {
        durable::replace(&self.holder, record, 0o600)?;
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

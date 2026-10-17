use super::Region;
use crate::{Error, Result, Seam};
use std::os::fd::{BorrowedFd, OwnedFd};

pub fn memfd_named(_: &str) -> Result<OwnedFd> {
    Err(Error::unsupported(Seam::SharedMemory))
}

pub fn seals(_: BorrowedFd<'_>) -> Result<i32> {
    Err(Error::unsupported(Seam::SharedMemory))
}

pub fn seal(_: BorrowedFd<'_>) -> Result<()> {
    Err(Error::unsupported(Seam::SharedMemory))
}

pub fn seal_with(_: BorrowedFd<'_>, _: i32) -> Result<()> {
    Err(Error::unsupported(Seam::SharedMemory))
}

pub fn held(_: u64, _: u64) -> Result<bool> {
    Err(Error::unsupported(Seam::SharedMemory))
}

pub fn region_lock(_: BorrowedFd<'_>, _: usize, _: usize, _: Region, _: bool) -> Result<bool> {
    Err(Error::unsupported(Seam::SharedMemory))
}

pub fn region_holder(_: BorrowedFd<'_>, _: usize, _: usize) -> Result<Option<(usize, usize)>> {
    Err(Error::unsupported(Seam::SharedMemory))
}

pub fn punch(_: BorrowedFd<'_>, _: u64, _: u64) -> Result<()> {
    Err(Error::unsupported(Seam::SharedMemory))
}

pub fn reopen(_: BorrowedFd<'_>, _: bool) -> Result<OwnedFd> {
    Err(Error::unsupported(Seam::SharedMemory))
}

pub fn advise_huge(_: *mut u8, _: usize) {}

pub(super) fn populate_write(_: *mut u8, _: usize) -> bool {
    false
}

/// `bytes` as a file no process can write: no seals here, so an unlinked file in this user's
/// temporary directory, read-only by its mode and by the only description handed on.
pub fn frozen(name: &str, bytes: &[u8]) -> Result<std::fs::File> {
    use std::{
        fs::{self, OpenOptions},
        io::Write,
        os::unix::fs::{OpenOptionsExt, PermissionsExt},
        sync::atomic::{AtomicU64, Ordering},
    };
    static NAMES: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir();
    let (path, mut file) = loop {
        let n = NAMES.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!(".{name}.{}.{n}", std::process::id()));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)
        {
            Ok(file) => break (path, file),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(crate::Error::io("create", path)(error)),
        }
    };
    let frozen = (|| {
        file.write_all(bytes)?;
        file.set_permissions(fs::Permissions::from_mode(0o400))?;
        fs::File::open(&path)
    })();
    let _ = fs::remove_file(&path);
    frozen.map_err(crate::Error::io("freeze", path))
}

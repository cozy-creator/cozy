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

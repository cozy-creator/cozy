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

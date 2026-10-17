use crate::{Error, Result, Seam};
use std::fs::File;

pub const FULL_SEALS: i32 = 0;

pub fn memfd_named(_: &str) -> Result<File> {
    Err(Error::unsupported(Seam::SharedMemory))
}

pub fn seals(_: &File) -> Result<i32> {
    Err(Error::unsupported(Seam::SharedMemory))
}

pub fn seal(_: &File) -> Result<()> {
    Err(Error::unsupported(Seam::SharedMemory))
}

pub fn held(_: u64, _: u64) -> Result<bool> {
    Err(Error::unsupported(Seam::SharedMemory))
}

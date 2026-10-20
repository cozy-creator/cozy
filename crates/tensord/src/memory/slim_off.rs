//! The placement ledger requires Linux OFD locks and eventfd. Other hosts still build and
//! serve CPU work; a GPU grant is explicitly unavailable until their native tier exists.
use platform::handles::Owned;
use std::io;

pub struct Grant {
    pub table: Owned,
    pub events: Owned,
    pub channel: Owned,
    pub pid: u32,
}
pub struct Memory;
fn unsupported<T>() -> io::Result<T> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "the placement ledger is unavailable on this platform",
    ))
}
impl Memory {
    pub fn create(_: &str, _: u64, _: Option<u64>) -> io::Result<Self> {
        unsupported()
    }
    pub fn grant_unbound(&self) -> io::Result<Grant> {
        unsupported()
    }
    pub fn grant(&self, _: u32) -> io::Result<Grant> {
        unsupported()
    }
    pub fn reap(&self, _: u32) -> io::Result<()> {
        unsupported()
    }
    pub fn fits(&self, _: &str, _: &str) -> io::Result<bool> {
        unsupported()
    }
    pub fn release_unused_ram(&self) -> bool {
        false
    }
    pub fn capacity(&self) -> u64 {
        0
    }
}

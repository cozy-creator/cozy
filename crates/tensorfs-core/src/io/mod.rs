//! The read engine (design §13): byte ranges of files into destinations. TensorFS reads every
//! byte; a caller (varena, the Python surface) only says where they go. Raw destinations make
//! this the crate's one unsafe seam outside `platform`.
#![allow(unsafe_code)]

mod host;
mod plan;
mod probe;

pub use plan::Target;
pub use probe::{ceiling, disk, Ceiling};

use std::fs::File;
use std::time::Instant;

/// `dst.len()` bytes of `src` into `dst`.
#[derive(Clone, Copy, Debug)]
pub struct Copy<'a> {
    pub src: Src<'a>,
    pub dst: Dst,
}

#[derive(Clone, Copy, Debug)]
pub enum Src<'a> {
    /// A file from an offset.
    File(&'a File, u64),
    /// Host memory (a sealed layout's mapping): RAM to the destination, no disk.
    Host(*const u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dst {
    Host { ptr: *mut u8, len: usize },
    /// Device memory (`ptr` a `CUdeviceptr`), ordered on `stream` (a `CUstream`; 0: the
    /// engine's own).
    Device { device: i32, ptr: u64, len: usize, stream: usize },
}

// SAFETY: sources and destinations are addresses; `read`'s contract lends them to the engine.
unsafe impl Send for Dst {}
unsafe impl Sync for Dst {}
unsafe impl Send for Src<'_> {}
unsafe impl Sync for Src<'_> {}

impl Dst {
    pub fn host(buf: &mut [u8]) -> Dst {
        Dst::Host { ptr: buf.as_mut_ptr(), len: buf.len() }
    }

    pub fn len(&self) -> usize {
        match *self {
            Dst::Host { len, .. } | Dst::Device { len, .. } => len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `len` bytes of it from `at`.
    pub(crate) fn slice(self, at: usize, len: usize) -> Dst {
        assert!(at + len <= self.len(), "{at}+{len} outside a {} byte destination", self.len());
        match self {
            Dst::Host { ptr, .. } => Dst::Host { ptr: ptr.wrapping_add(at), len },
            Dst::Device { device, ptr, stream, .. } => Dst::Device { device, ptr: ptr + at as u64, len, stream },
        }
    }
}

#[derive(Clone, Debug)]
pub struct ReadOpts {
    /// Reads in flight at once.
    pub threads: usize,
    /// Bytes per read: a copy is cut on this grid of file offsets.
    pub request: usize,
    /// Copy streams per device (device destinations).
    pub streams: usize,
    /// Read what the page cache lacks with O_DIRECT, where the filesystem allows.
    pub direct: bool,
}

impl Default for ReadOpts {
    fn default() -> ReadOpts {
        ReadOpts { threads: 16, request: 1 << 20, streams: 2, direct: true }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReadStats {
    pub bytes: u64,
    pub ns: u64,
    /// Bytes read with O_DIRECT, past the page cache.
    pub direct: u64,
    /// Bytes read through the page cache.
    pub cached: u64,
    /// Of `direct`, bytes that went through an aligned bounce buffer (a destination or end
    /// O_DIRECT cannot reach).
    pub bounced: u64,
}

impl ReadStats {
    pub fn gbps(&self) -> f64 {
        self.bytes as f64 / self.ns.max(1) as f64
    }
}

/// Copy `copy` failed.
#[derive(Debug, thiserror::Error)]
#[error("copy {copy}: {source}")]
pub struct Error {
    pub copy: usize,
    #[source]
    pub source: std::io::Error,
}

/// Every copy, at once: reads on `opts.threads` threads, each cut on the `opts.request` grid,
/// O_DIRECT for what the page cache lacks (straight into the destination where it is aligned
/// with the file, else through a bounce buffer), buffered for what it holds.
///
/// # Safety
/// Each destination is valid for writes of its length, the destinations do not overlap, and
/// nothing else reads or writes them until this returns; each host source is valid for reads
/// of its destination's length and not written until this returns.
pub unsafe fn read(copies: &[Copy], opts: &ReadOpts) -> Result<ReadStats, Error> {
    let start = Instant::now();
    if let Some(copy) = copies.iter().position(|c| matches!(c.dst, Dst::Device { .. })) {
        let source = std::io::Error::new(std::io::ErrorKind::Unsupported, "device destinations need the CUDA engine");
        return Err(Error { copy, source });
    }
    // SAFETY: per the contract.
    let mut stats = unsafe { host::fill(copies, opts) }?;
    stats.bytes = copies.iter().map(|c| c.dst.len() as u64).sum();
    stats.ns = start.elapsed().as_nanos() as u64;
    Ok(stats)
}

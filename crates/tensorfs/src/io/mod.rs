//! The read engine (design §13): byte ranges of files into destinations. TensorFS reads every
//! byte; a caller (varena, the Python surface) only says where they go. Raw destinations make
//! this the crate's one unsafe seam outside `platform`.
#![allow(unsafe_code)]

pub mod cuda;
#[cfg(feature = "store")]
pub mod decode;
mod device;
mod host;
#[cfg(unix)]
pub mod metrics;
pub mod pins;
mod plan;
mod probe;
#[cfg(unix)]
pub mod source;
#[cfg(feature = "store")]
pub(crate) mod view;

pub use device::DeviceBuffer;
pub use plan::Target;
pub use probe::{ceiling, disk, Ceiling};
#[cfg(feature = "store")]
pub use view::{Span, StoredPart};

/// `len` bytes of object `obj` from `off`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectRange {
    pub obj: crate::format::ObjectRef,
    pub off: u64,
    pub len: u64,
}

/// Where placed bytes come from.
#[derive(Debug, Clone)]
pub enum ItemSource {
    Object(ObjectRange),
    Inline(std::sync::Arc<[u8]>),
}

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
    /// Fill the destination with zeroes, without a file or a host staging slab.
    Zero,
    /// A file from an offset.
    File(&'a File, u64),
    /// Host memory (a sealed layout's mapping): RAM to the destination, no disk.
    Host(*const u8),
    /// Device bytes ordered on the producer's stream (0: the CUDA default stream).
    Device {
        device: i32,
        ptr: u64,
        stream: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dst {
    Host {
        ptr: *mut u8,
        len: usize,
    },
    /// Device memory (`ptr` a `CUdeviceptr`), ordered on `stream` (a `CUstream`; 0: the
    /// engine's own).
    Device {
        device: i32,
        ptr: u64,
        len: usize,
        stream: usize,
    },
}

// SAFETY: sources and destinations are addresses; `read`'s contract lends them to the engine.
unsafe impl Send for Dst {}
unsafe impl Sync for Dst {}
unsafe impl Send for Src<'_> {}
unsafe impl Sync for Src<'_> {}

impl Dst {
    pub fn host(buf: &mut [u8]) -> Dst {
        Dst::Host {
            ptr: buf.as_mut_ptr(),
            len: buf.len(),
        }
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
        assert!(
            at + len <= self.len(),
            "{at}+{len} outside a {} byte destination",
            self.len()
        );
        match self {
            Dst::Host { ptr, .. } => Dst::Host {
                ptr: ptr.wrapping_add(at),
                len,
            },
            Dst::Device {
                device,
                ptr,
                stream,
                ..
            } => Dst::Device {
                device,
                ptr: ptr + at as u64,
                len,
                stream,
            },
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
        ReadOpts {
            threads: 16,
            request: 1 << 20,
            streams: 2,
            direct: true,
        }
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
    pub pin_ns: u64,
    pub staged: u64,
    pub busy_ns: u64,
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

impl Error {
    pub fn cuda(&self) -> Option<&cuda::Error> {
        self.source.get_ref()?.downcast_ref()
    }
}

/// Check completion of this device's engine streams and retained decode work. No driver,
/// context or staging pool is initialized if this process has never made an engine for it.
/// A producer's own stream must be checked separately before its owner is released.
pub fn drain(device: i32) -> cuda::Result<()> {
    device::drain(device)
}

/// Every copy, at once: reads on `opts.threads` threads, each cut on the `opts.request` grid,
/// O_DIRECT for what the page cache lacks (straight into the destination where it is aligned
/// with the file, else through a bounce buffer), buffered for what it holds.
///
/// # Safety
/// Each destination is valid for writes of its length, the destinations do not overlap, and
/// nothing else reads or writes them while the operation may access them; each host source
/// is valid for reads of its destination's length. Success proves completion. After a CUDA
/// error, retain all device destinations and direct host/device source owners until `drain`
/// succeeds for every participating device and producer streams are also checked complete.
/// An error or best-effort destructor is not a proof that borrowed memory is reusable.
/// Host destinations use joined CPU copies from owned staging, never asynchronous GPU writes.
pub unsafe fn read(copies: &[Copy], opts: &ReadOpts) -> Result<ReadStats, Error> {
    unsafe { read_impl(copies, opts, false) }
}

/// Temporary host sources are copied only into the same bounded engine-owned B7 slots.
/// GPU work never borrows those host source addresses beyond this call, including on error.
/// Device destination ownership still follows `read`'s checked error-completion contract.
#[cfg(any(feature = "store", unix))]
pub(crate) unsafe fn read_staged(copies: &[Copy], opts: &ReadOpts) -> Result<ReadStats, Error> {
    unsafe { read_impl(copies, opts, true) }
}

unsafe fn read_impl(
    copies: &[Copy],
    opts: &ReadOpts,
    staged_host: bool,
) -> Result<ReadStats, Error> {
    let start = Instant::now();
    for (copy, c) in copies.iter().enumerate() {
        let n = c.dst.len() as u64;
        let invalid = match c.src {
            Src::File(_, offset) if offset.checked_add(n).is_none() => {
                Some("source file range overflows")
            }
            Src::Device { ptr, .. } if (ptr == 0 && n != 0) || ptr.checked_add(n).is_none() => {
                Some("invalid CUDA source range")
            }
            _ => match c.dst {
                Dst::Device { ptr, .. } if (ptr == 0 && n != 0) || ptr.checked_add(n).is_none() => {
                    Some("invalid CUDA destination range")
                }
                _ => None,
            },
        };
        if let Some(detail) = invalid {
            return Err(Error {
                copy,
                source: std::io::Error::new(std::io::ErrorKind::InvalidInput, detail),
            });
        }
    }
    let (host, devices): (Vec<_>, Vec<_>) =
        copies.iter().copied().enumerate().partition(|(_, c)| {
            !matches!(c.src, Src::Device { .. }) && !matches!(c.dst, Dst::Device { .. })
        });
    let host_copies: Vec<_> = host.iter().map(|(_, c)| *c).collect();
    let mut stats = if host_copies.is_empty() {
        ReadStats::default()
    } else {
        let host_start = Instant::now();
        let stats = unsafe { host::fill(&host_copies, opts) }.map_err(|e| Error {
            copy: host[e.copy].0,
            source: e.source,
        })?;
        #[cfg(unix)]
        disk_run(&host_copies, &stats, host_start.elapsed().as_nanos() as u64);
        stats
    };
    let device = unsafe { device::fill(&devices, opts, staged_host) }?;
    stats.direct += device.direct;
    stats.cached += device.cached;
    stats.bounced += device.bounced;
    stats.pin_ns += device.pin_ns;
    stats.staged += device.staged;
    stats.busy_ns += device.busy_ns;
    stats.bytes = copies.iter().map(|c| c.dst.len() as u64).sum();
    stats.ns = start.elapsed().as_nanos() as u64;
    Ok(stats)
}

/// Logs the file reads among `copies` as one disk run (`metrics`).
#[cfg(unix)]
fn disk_run(copies: &[Copy], stats: &ReadStats, ns: u64) {
    let mut files: Vec<&File> = Vec::new();
    let mut bytes = 0;
    for c in copies {
        if let Src::File(f, _) = c.src {
            bytes += c.dst.len() as u64;
            if !files.iter().any(|known| std::ptr::eq(*known, f)) {
                files.push(f);
            }
        }
    }
    if bytes > 0 {
        metrics::disk_run(&files, bytes, ns, stats.direct, stats.cached);
    }
}

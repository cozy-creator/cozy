//! Sealed shared memory for weight layouts and host pages: a Linux memfd with file
//! seals, handed between processes by descriptor and claimed region by region with
//! open-file-description locks. macOS has neither memfd nor seals; the Apple tier (AP) designs its
//! own, and until then every call answers `Unsupported`.
//! Windows has only `frozen`.
#![allow(unsafe_code)]
#[cfg_attr(target_os = "linux", path = "linux.rs")]
#[cfg_attr(all(unix, not(target_os = "linux")), path = "unsupported.rs")]
#[cfg_attr(windows, path = "windows.rs")]
mod imp;

pub use imp::*;

const SEAL_SEAL: i32 = 0x1;
const SEAL_SHRINK: i32 = 0x2;
const SEAL_GROW: i32 = 0x4;
const SEAL_WRITE: i32 = 0x8;
const SEAL_FUTURE_WRITE: i32 = 0x10;

/// What makes a memfd immutable to everyone else: no write, resize, punch or unsealing.
/// Future-write (Linux 5.1) refuses new writable mappings and writes; the filler's own mapping
/// may finish.
pub const FULL_SEALS: i32 = SEAL_SEAL | SEAL_SHRINK | SEAL_GROW | SEAL_FUTURE_WRITE;
/// What an adopter of a sealed memfd requires besides one of [`WRITE_SEALS`].
pub const RESIZE_SEALS: i32 = SEAL_SEAL | SEAL_SHRINK | SEAL_GROW;
pub const WRITE_SEALS: i32 = SEAL_WRITE | SEAL_FUTURE_WRITE;
/// No write at all, and no writable mapping may exist when it is sealed: the set TensorD seals
/// with and requires of a plan it receives.
pub const NO_WRITE_SEALS: i32 = RESIZE_SEALS | SEAL_WRITE;

/// A byte-range lock request on an open file description.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(unix)]
pub enum Region {
    Read,
    Write,
    Unlock,
}

#[cfg(unix)]
pub fn page_size() -> usize {
    // SAFETY: sysconf reads a constant.
    (unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize).max(4096)
}

/// Faults in every page of `[base, base+len)` for writing, in `threads` slices: pages a memfd
/// lacks are allocated, pages it has are only mapped. Without MADV_POPULATE_WRITE (Linux
/// before 5.14, other OSes) each page is touched instead, writing the zero it reads as.
///
/// # Safety
/// `[base, base+len)` is a live, writable shared mapping of pages nothing has written yet (the
/// fallback writes a zero into each), and nothing else accesses it during the call.
#[cfg(unix)]
pub unsafe fn populate(base: *mut u8, len: usize, threads: usize) {
    let page = page_size();
    let threads = threads.max(1);
    let step = len.div_ceil(threads).next_multiple_of(page);
    let base = base as usize;
    std::thread::scope(|s| {
        for k in 0..threads {
            let (lo, hi) = ((k * step).min(len), ((k + 1) * step).min(len));
            if lo == hi {
                continue;
            }
            s.spawn(move || {
                if !imp::populate_write((base + lo) as *mut u8, hi - lo) {
                    for p in (lo..hi).step_by(page) {
                        // SAFETY: a page of the caller's live, writable, unwritten mapping (contract above).
                        unsafe { std::ptr::write_volatile((base + p) as *mut u8, 0) };
                    }
                }
            });
        }
    });
}

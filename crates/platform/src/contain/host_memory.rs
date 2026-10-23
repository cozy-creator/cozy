//! Live host memory: what this process can still allocate before its tightest limit. Read on
//! every decision, never cached: a budget fixed at boot describes a machine nobody else is on.
//! Linux ports Runtime `proctree.host_memory` (cgroup v1 and v2, every parent, and
//! `MemAvailable`); macOS reads the VM statistics, Windows the global memory status.
use serde::Serialize;
use std::io;

#[cfg_attr(target_os = "linux", path = "host_memory_linux.rs")]
#[cfg_attr(target_os = "macos", path = "host_memory_macos.rs")]
#[cfg_attr(windows, path = "host_memory_windows.rs")]
mod sys;

#[cfg(target_os = "linux")]
pub(crate) use sys::{cgroups, number};

#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq, Eq)]
pub struct HostMemory {
    /// Bytes left before the tightest limit on this process's cgroup path, or the host's
    /// available memory when that is tighter. -1: unreadable (never 0).
    pub available: i64,
    /// Shared memory (memfd/tmpfs) charged where that limit is: host pages and sealed layouts.
    /// 0 where weights are never held in shared memory (macOS, Windows).
    pub shmem: i64,
    /// The host's available memory (`MemAvailable` on Linux), for the record; -1 when unreadable.
    pub mem_available: i64,
    /// The limit `available` is measured against: the tightest cgroup's, else physical memory.
    pub bound: i64,
}

/// Physical memory; -1 when unreadable.
pub fn total() -> i64 {
    sys::total()
}

/// The live budget: the tightest of the host's available memory and, on Linux, every cgroup
/// on this process's path.
pub fn read() -> HostMemory {
    sys::read()
}

/// One process's resident and proportional set sizes: Linux's PSS counts a shared sealed layout
/// once across every process that maps it; macOS gives its physical footprint and Windows its
/// private commit there, with no shared part.
#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq, Eq)]
pub struct ProcessMemory {
    pub rss: u64,
    pub pss: u64,
    pub pss_shmem: u64,
}

pub fn process(pid: u32) -> io::Result<ProcessMemory> {
    sys::process(pid)
}

/// One process's resident bytes now, anonymous, file and shmem (a sealed layout it maps), and
/// its resident high-water since it started or [`reset_peak`]. Linux keeps no anonymous
/// high-water; varena's request window bounds one from these. Linux only; elsewhere
/// `Unsupported`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Resident {
    pub anon: u64,
    pub file: u64,
    pub shmem: u64,
    pub peak: u64,
}

pub fn resident(pid: u32) -> io::Result<Resident> {
    sys::resident(pid)
}

/// Starts `pid`'s high-water over from its use now.
pub fn reset_peak(pid: u32) -> io::Result<()> {
    sys::reset_peak(pid)
}

/// Private bytes (PSS less shared pages) of `pid` and every process descended from it: what a
/// process tree holds that no one else counts.
pub fn tree_private(pid: u32) -> u64 {
    std::iter::once(pid)
        .chain(crate::process::descendants(pid))
        .filter_map(|pid| process(pid).ok())
        .map(|m| m.pss.saturating_sub(m.pss_shmem))
        .sum()
}

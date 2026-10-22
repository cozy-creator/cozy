//! The host asks for memory back (MEM/HOST-PRESSURE.md). One thread blocks in poll() with no
//! timeout on what the kernel lets this process watch: a PSI trigger (`shared`: the system's
//! `some` stall, anyone's; `dedicated`: this cgroup's `full`, else the system's) and inotify on
//! `memory.events` of every limited cgroup v2 on its path, whose counters change as a limit is
//! reached (containers deny PSI triggers, not inotify). Cgroup v1 has neither: system `some` wakes
//! the watcher and only a rise of a limited cgroup's `memory.failcnt` counts. A wakeup only says
//! "look": rungs go back while the live budget is below what must stay free (each live call's
//! measured growth still to come, and [`Growth`]), so benign reclaim at a limit, which a pod
//! counts 100k times, gives nothing back.
use super::host_memory::HostMemory;
use serde::Deserialize;
use std::time::Duration;

/// How the machine shares its host (`host.mode` in the GPU config): a guest on someone's
/// computer (the default), or the whole of a rented pod (its image or grant says so).
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum HostMode {
    #[default]
    Shared,
    Dedicated,
}

/// The trigger's window: the shortest an unprivileged process may arm (a multiple of 2 s on
/// Linux 6.x and 7.0). The threshold is the smallest measurable stall, so an event means
/// "something stalled on memory in the last window".
pub const WINDOW: Duration = Duration::from_secs(2);

/// The most non-reclaimable bytes, shared memory aside, grew between two consecutive looks: what
/// can arrive before the watcher answers, so it stays free. Shared memory is the host tier itself
/// (sealed layouts), whose fill is admitted, never answered. A high-water; a look against
/// another bound (the tightest limit changed) starts over from it.
#[derive(Debug, Default)]
pub struct Growth {
    last: Option<HostMemory>,
    most: u64,
}

impl Growth {
    pub fn observe(&mut self, host: HostMemory) -> u64 {
        if host.available < 0 || host.bound <= 0 {
            return self.most;
        }
        if let Some(last) = self.last.filter(|last| last.bound == host.bound) {
            let grew = (last.available - host.available) - (host.shmem - last.shmem);
            self.most = self.most.max(grew.max(0) as u64);
        }
        self.last = Some(host);
        self.most
    }

    pub fn most(&self) -> u64 {
        self.most
    }
}

/// What shared mode leaves the host's other programs beyond their use now: the most they used
/// recently. Raised to their use at every reading and every stall; each bring-back pass that
/// found no stall since the previous one halves the excess, so a spike that caused no pressure
/// (a compile, a browser burst) holds warming back for a few calls, not until a restart.
/// Measured, never a constant. Dedicated mode leaves nothing.
#[derive(Debug, Default)]
pub struct Reserve {
    peak: u64,
}

impl Reserve {
    /// The other programs use `others` bytes now: the bytes discretionary holdings must leave.
    pub fn observe(&mut self, others: u64) -> u64 {
        self.peak = self.peak.max(others);
        self.peak - others
    }

    /// A bring-back pass with no stall since the last one: the excess halves.
    pub fn quiet(&mut self, others: u64) {
        self.peak = others.max(self.peak - self.peak.saturating_sub(others) / 2);
    }
}

/// Bytes the host's other programs use: everything in use but what the machine holds.
pub fn others(host: &HostMemory, total: u64, ours: u64) -> Option<u64> {
    let available = u64::try_from(host.mem_available).ok()?;
    Some(total.saturating_sub(available).saturating_sub(ours))
}

#[cfg_attr(target_os = "linux", path = "host_pressure_linux.rs")]
#[cfg_attr(not(target_os = "linux"), path = "host_pressure_unsupported.rs")]
mod trigger;

pub use trigger::Pressure;

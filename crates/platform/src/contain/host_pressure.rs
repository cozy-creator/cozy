//! The host asks for memory back (`tensord::host_tier`). One thread blocks in poll() with no
//! timeout on whatever this process may watch: a PSI trigger (`shared`: the system's `some`
//! stall, anyone's; `dedicated`: this cgroup's `full`, else the system's `some`), inotify on the
//! `memory.events` of each limited cgroup v2 on its path that it can watch (containers deny PSI
//! triggers, not inotify), and a [`Nudge`]: TensorD's own events (a call begins, an executor
//! frame arrives), which are the only wakeups where the kernel allows none (cgroup v1 with PSI
//! denied). A wakeup only says "look": rungs go back while the live budget is below what must
//! stay free, so benign reclaim at a limit, which a pod counts 100k times, gives nothing back.
use super::host_memory::HostMemory;
use serde::Deserialize;
use std::{collections::BTreeMap, time::Duration};

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

/// The most the live calls grew their anonymous bytes beyond what was measured to come, between
/// two looks: what can arrive before the next look answers, so it stays free. An unmeasured
/// call's every byte is unforeseen, a measured one's only past its peak. Sealed layouts and page
/// cache are no part of it (a layout's fill is admitted, never answered; clean cache is room).
/// Each call that ends measured settles it: what was seen since the last one stays, the rest
/// halves.
#[derive(Debug, Default)]
pub struct Growth {
    last: BTreeMap<String, u64>,
    most: u64,
    seen: u64,
}

impl Growth {
    /// Each live call's unforeseen anonymous bytes now; a call first seen has grown nothing.
    pub fn observe(&mut self, unforeseen: BTreeMap<String, u64>) -> u64 {
        let grew = unforeseen.iter().filter_map(|(call, now)| Some(now.saturating_sub(*self.last.get(call)?))).sum();
        self.seen = self.seen.max(grew);
        self.most = self.most.max(grew);
        self.last = unforeseen;
        self.most
    }

    /// A call ended measured.
    pub fn settle(&mut self) {
        self.most = self.seen + (self.most - self.seen) / 2;
        self.seen = 0;
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

pub use trigger::{Nudge, Pressure};

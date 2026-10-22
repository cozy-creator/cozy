//! The host asks for memory back (MEM/HOST-PRESSURE.md). A PSI trigger wakes one thread blocked
//! in poll() with no timeout: in `shared` mode on the whole system's memory pressure (anyone
//! stalled, the owner's apps included), in `dedicated` mode on this cgroup's (every task of it
//! stalled). Cgroup v1 has no per-cgroup PSI: system `some` wakes the watcher, but only
//! this cgroup's limit-hit counters may request reclaim, including limits on its parents.
//! Each event gives one rung back while each rung lowers the pressure; one that did not
//! stops it, since that stall is not ours to fix (page-cache refaults on a slow disk), until the
//! stall rises above where it stopped (`Feedback`).
use serde::Deserialize;
use std::time::{Duration, Instant};

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

/// Different kernel counters, never synthetic stall microseconds. V1's memory.failcnt
/// counts charges that reached a limit and triggered reclaim, not just failed allocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sample {
    StalledUs(u64),
    LimitHits(u64),
}

impl Sample {
    pub fn total(self) -> u64 {
        match self {
            Self::StalledUs(total) | Self::LimitHits(total) => total,
        }
    }
    pub fn stalled_us(self) -> Option<u64> {
        match self {
            Self::StalledUs(total) => Some(total),
            _ => None,
        }
    }
    pub fn limit_hits(self) -> Option<u64> {
        match self {
            Self::LimitHits(total) => Some(total),
            _ => None,
        }
    }
}

/// Whether the next event gives a rung back. Each event brings the pressure counter's rate:
/// stalled time for PSI, limit hits for v1. After a rung, a rate lower than before by more than the noise
/// says the rung helped, and another may go; one that is not stops giving, and the noise is then
/// measured: the largest change between consecutive shares while nothing is given. Giving
/// resumes only once the share rises above where it stopped by more than that noise. So a stall
/// giving back cannot touch (page-cache refaults on a slow disk) sheds a rung or two, however
/// long it lasts, while a hog's stall falls rung by rung and a rising one is answered. Nothing
/// here is a constant: the window is the trigger's, the noise is measured. An event more than
/// two windows after the last means a whole window passed without a stall: a new episode, in
/// which a stall no higher than where giving stopped still gives nothing (background refault
/// bursts would otherwise shed a rung each).
#[derive(Debug)]
pub struct Feedback {
    total: u64,
    at: Instant,
    last: Option<f64>,
    rung: Option<f64>,
    stopped: Option<f64>,
    noise: Option<f64>,
}

impl Feedback {
    pub fn new(total: u64, at: Instant) -> Self {
        Self {
            total,
            at,
            last: None,
            rung: None,
            stopped: None,
            noise: None,
        }
    }

    pub fn give(&mut self, total: u64, at: Instant) -> bool {
        let elapsed = at.saturating_duration_since(self.at);
        let share = total.saturating_sub(self.total) as f64 / elapsed.as_micros().max(1) as f64;
        (self.total, self.at) = (total, at);
        if elapsed > 2 * WINDOW {
            // a new episode; where giving stopped, and the noise, still hold
            (self.last, self.rung) = (None, None);
        }
        let previous = self.last.replace(share);
        if let Some(level) = self.stopped {
            if self.noise.is_some_and(|noise| share > level + noise) {
                (self.stopped, self.rung) = (None, Some(share));
                return true;
            }
            if let Some(previous) = previous {
                let change = (share - previous).abs();
                self.noise = Some(self.noise.map_or(change, |noise| noise.max(change)));
            }
            return false;
        }
        let noise = self.noise.unwrap_or(0.0);
        if self.rung.is_some_and(|before| share >= before - noise) {
            self.stopped = Some(share);
            return false;
        }
        self.rung = Some(share);
        true
    }

    /// None is an unrelated host wakeup (or a reset v1 counter), not local pressure.
    pub fn observe(&mut self, sample: Sample, at: Instant) -> Option<bool> {
        if let Sample::LimitHits(total) = sample {
            // A host PSI wakeup without another local limit hit is not this pod's
            // pressure. Administrators may reset failcnt: start a fresh baseline.
            if total < self.total {
                *self = Self::new(total, at);
                return None;
            }
            if total == self.total {
                return None;
            }
        }
        Some(self.give(sample.total(), at))
    }

    /// Nothing was left to give: wait for more pressure than now.
    pub fn exhausted(&mut self) {
        self.stopped = self.rung.take();
    }

    /// The share giving stopped at, for the record; None while it gives.
    pub fn stopped_at(&self) -> Option<f64> {
        self.stopped
    }
}

#[cfg_attr(target_os = "linux", path = "host_pressure_linux.rs")]
#[cfg_attr(not(target_os = "linux"), path = "host_pressure_unsupported.rs")]
mod trigger;

pub use trigger::Pressure;

//! The machine's RAM tier: one host layout per layout digest with a ready word per region,
//! mapped read-only by executors, held by whichever participant holds the machine's service
//! lock (TensorD when present). Under pressure it gives back staging, then orphaned layouts
//! (least recently used), then unused layouts, then the furthest declared, never one in flight.

use std::os::fd::OwnedFd;

use crate::{Error, Result};

/// Where pressure comes from.
pub enum Pressure {
    /// The host's PSI trigger and cgroup events.
    Host(platform::contain::host_pressure::Pressure),
    /// A descriptor that turns readable on pressure (an eventfd).
    Events(OwnedFd),
}

/// The machine's RAM-tier service, served from this process while it lives.
pub struct Tier {
    _pressure: Pressure,
}

impl Tier {
    /// Takes the machine's service lock and serves the tier; `None` when another participant
    /// serves it.
    pub fn serve(pressure: Pressure) -> Result<Option<Tier>> {
        let _ = pressure;
        Err(Error::Internal("not built yet".into()))
    }
}

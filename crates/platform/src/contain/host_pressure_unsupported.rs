//! Memory-pressure triggers exist only on Linux so far (PSI and cgroup counters).
use super::HostMode;
use crate::{Error, Result, Seam};

pub struct Pressure(());

/// Nothing to wake where nothing watches.
#[derive(Clone)]
pub struct Nudge(());

impl Nudge {
    pub fn send(&self) {}
}

impl Pressure {
    pub fn arm(_: HostMode) -> Result<Self> {
        Err(Error::unsupported(Seam::Pressure))
    }

    pub fn how(&self) -> &str {
        ""
    }

    pub fn nudge(&self) -> Nudge {
        Nudge(())
    }

    pub fn wait(&mut self) -> Result<()> {
        Err(Error::unsupported(Seam::Pressure))
    }
}

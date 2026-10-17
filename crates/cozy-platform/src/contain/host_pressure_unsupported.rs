//! Memory-pressure triggers exist only on Linux so far (PSI and cgroup counters).
use super::{HostMode, Sample};
use crate::{Error, Result, Seam};

pub struct Pressure(());

impl Pressure {
    pub fn arm(_: HostMode) -> Result<Self> {
        Err(Error::unsupported(Seam::Pressure))
    }

    pub fn wait(&self) -> Result<()> {
        Err(Error::unsupported(Seam::Pressure))
    }

    pub fn stalled_us(&self) -> Result<u64> {
        Err(Error::unsupported(Seam::Pressure))
    }

    pub fn sample(&self) -> Result<Sample> {
        Err(Error::unsupported(Seam::Pressure))
    }
}

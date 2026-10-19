//! Policy for an explicitly enabled library cache. Raw object stores keep their own policy.

use std::path::Path;
use std::time::Duration;

use super::{platform, Result};

pub(super) const TTL: Duration = Duration::from_secs(7 * 24 * 60 * 60);
pub(super) const SWEEP: Duration = Duration::from_secs(10 * 60);
const RESERVE_MIN: u64 = 1 << 30;
const RESERVE_MAX: u64 = 10 << 30;
const RESERVE_INODES: u64 = 32;

/// The existing filesystem sample seam; pressure tests sample space and delete real files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Disk {
    pub capacity: u64,
    pub available: u64,
    pub inodes: u64,
    pub available_inodes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Need {
    pub bytes: u64,
    pub inodes: u64,
}

impl Disk {
    pub fn measure(root: &Path) -> Result<Self> {
        let sample = platform::durable::space(root).map_err(platform("measure space", root))?;
        Ok(Self {
            capacity: sample.total,
            available: sample.free,
            inodes: sample.inodes,
            available_inodes: sample.free_inodes,
        })
    }

    pub fn reserve(self) -> u64 {
        (self.capacity / 50).clamp(RESERVE_MIN, RESERVE_MAX)
    }

    pub fn need(self, incoming: u64) -> Need {
        Need {
            bytes: self
                .reserve()
                .saturating_add(incoming.max(1))
                .saturating_sub(self.available),
            inodes: if self.inodes == 0 {
                0
            } else {
                (RESERVE_INODES + 1).saturating_sub(self.available_inodes)
            },
        }
    }
}

impl Need {
    pub fn covered(self, bytes: u64, inodes: u64) -> bool {
        bytes >= self.bytes && inodes >= self.inodes
    }

    pub fn empty(self) -> bool {
        self == Self::default()
    }
}

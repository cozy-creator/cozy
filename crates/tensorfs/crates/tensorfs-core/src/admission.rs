//! The disk a download is admitted against, and the GC pass that relieves a low one. Disk
//! only: nothing here looks at RAM (owner rule: no RAM floor; the Runtime pages to disk).
//!
//! A reserve of 1/50 of the filesystem, clamped to 1–10 GiB, and 32 inodes is never admitted;
//! an object needs its bytes and 3 inodes, a download 16 more. A disk is low when no more than
//! the reserve is free; GC then takes unreferenced bytes and the fewest least-recently-used
//! cache roots that cover what is missing (`gc::collect_cached_for`).

use std::ops::Add;
use std::path::Path;

use crate::err::{Code, Refusal, Result};
use crate::gc;
use crate::store::Store;

/// 1/50 of the filesystem, at least 1 GiB and at most 10 GiB: 2% of a large disk can be
/// everything still free on it.
const RESERVE_BYTES: (u64, u64) = (1 << 30, 10 << 30);
const RESERVE_DIVISOR: u64 = 50;
const RESERVE_INODES: u64 = 32;
const METADATA_INODES: u64 = 16;
const OBJECT_INODES: u64 = 3;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Space {
    pub bytes: u64,
    pub inodes: u64,
}

impl Add for Space {
    type Output = Space;
    fn add(self, other: Space) -> Space {
        Space {
            bytes: self.bytes.saturating_add(other.bytes),
            inodes: self.inodes.saturating_add(other.inodes),
        }
    }
}

impl Space {
    /// What `objects` new objects of `bytes` in total cost the disk.
    pub fn of(bytes: u64, objects: u64) -> Space {
        Space {
            bytes,
            inodes: if objects == 0 {
                0
            } else {
                METADATA_INODES + OBJECT_INODES * objects
            },
        }
    }

    #[cfg(test)]
    fn within(self, free: Space) -> bool {
        self.bytes <= free.bytes && self.inodes <= free.inodes
    }
}

/// One filesystem: its size and what is free on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Disk {
    pub capacity: u64,
    pub available: u64,
    /// Zero when the filesystem has no inode limit.
    pub inodes: u64,
    pub available_inodes: u64,
}

impl Disk {
    pub fn measure(root: &Path) -> Result<Disk> {
        let space = crate::platform::durable::space(root).map_err(|e| Refusal {
            code: Code::CAPACITY_EXHAUSTED,
            detail: format!(
                "cannot measure the filesystem under {}: {e}; admission fails closed",
                root.display()
            ),
        })?;
        Ok(Disk {
            capacity: space.total,
            available: space.free,
            inodes: space.inodes,
            available_inodes: space.free_inodes,
        })
    }

    /// Never admitted to a download, and never written into by an optional cache.
    pub fn reserve(&self) -> u64 {
        (self.capacity / RESERVE_DIVISOR).clamp(RESERVE_BYTES.0, RESERVE_BYTES.1)
    }

    /// A low disk has no more than its reserve free. Answers the bytes to free before it
    /// has more, or `None` while it is not low.
    pub fn short(&self) -> Option<u64> {
        let low = self.available <= self.reserve()
            || (self.inodes > 0 && self.available_inodes <= RESERVE_INODES);
        low.then(|| (self.reserve() + 1).saturating_sub(self.available).max(1))
    }
}

/// What `disk` admits once its reserve and `others` are set aside.
pub fn free(disk: &Disk, others: Space) -> Space {
    Space {
        bytes: disk
            .available
            .saturating_sub(disk.reserve())
            .saturating_sub(others.bytes),
        inodes: if disk.inodes == 0 {
            u64::MAX
        } else {
            disk.available_inodes
                .saturating_sub(RESERVE_INODES)
                .saturating_sub(others.inodes)
        },
    }
}

/// Free `short` bytes under the GC policy: the bytes freed, and why GC could not run.
fn collect(store: &Store, protected: &[String], short: u64) -> (u64, Option<Refusal>) {
    match gc::collect_cached_for(store.root(), protected, short) {
        Ok(report) => (report.reclaimed_bytes, None),
        Err(refusal) => (0, Some(refusal)),
    }
}

/// What one pass over a low disk found and did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relief {
    /// The disk was low when the pass began.
    pub pressure: bool,
    pub collected_bytes: u64,
    pub capacity_bytes: u64,
    pub available_bytes: u64,
    /// Why GC could not run, when it could not.
    pub unable: Option<String>,
}

/// One pass over a low disk: GC frees what brings it back above its reserve, or only the
/// unreferenced bytes when no cache roots cover that. `keep` is never evicted, nor what a
/// fetch holds. A disk that is not low is left alone.
pub fn relieve(store: &Store, keep: &[String]) -> Result<Relief> {
    let mut disk = Disk::measure(store.root())?;
    let short = disk.short();
    let (freed, unable) = short.map_or((0, None), |short| collect(store, keep, short));
    if short.is_some() {
        disk = Disk::measure(store.root())?;
    }
    Ok(Relief {
        pressure: short.is_some(),
        collected_bytes: freed,
        capacity_bytes: disk.capacity,
        available_bytes: disk.available,
        unable: unable.map(|r| r.to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn disk(capacity: u64, available: u64) -> Disk {
        Disk {
            capacity,
            available,
            inodes: 1000,
            available_inodes: 500,
        }
    }

    #[test]
    fn the_reserve_is_a_fiftieth_within_a_gib_and_ten_and_others_are_spoken_for() {
        // 10 GiB disk: the reserve is the 1 GiB floor.
        assert_eq!(disk(10 * GIB, 5 * GIB).reserve(), GIB);
        assert_eq!(
            free(&disk(10 * GIB, 5 * GIB), Space::default()).bytes,
            4 * GIB
        );
        // 250 GiB disk: a fiftieth, 5 GiB.
        assert_eq!(disk(250 * GIB, 100 * GIB).reserve(), 5 * GIB);
        // 1000 GiB disk: a fiftieth is 20 GiB, capped at 10 GiB.
        assert_eq!(
            free(&disk(1000 * GIB, 100 * GIB), Space::default()).bytes,
            90 * GIB
        );
        // The owner's laptop: 3.6 TiB with 73 GiB free is not low, and admits 63 GiB.
        let laptop = disk(3686 * GIB, 73 * GIB);
        assert_eq!((laptop.reserve(), laptop.short()), (10 * GIB, None));
        assert_eq!(free(&laptop, Space::default()).bytes, 63 * GIB);
        let others = Space::of(GIB, 10);
        let left = free(&disk(10 * GIB, 5 * GIB), others);
        assert_eq!(
            left,
            Space {
                bytes: 3 * GIB,
                inodes: 500 - 32 - 46
            }
        );
        // Nothing left is zero, never an underflow.
        assert_eq!(free(&disk(10 * GIB, GIB / 2), others).bytes, 0);
    }

    #[test]
    fn a_disk_is_low_at_its_reserve_and_short_by_what_lifts_it_above() {
        // 1000 GiB: the reserve is 10 GiB. A hundredth free is not low.
        assert_eq!(disk(1000 * GIB, 100 * GIB).short(), None);
        assert_eq!(disk(1000 * GIB, 10 * GIB + 1).short(), None);
        assert_eq!(disk(1000 * GIB, 10 * GIB).short(), Some(1));
        assert_eq!(disk(1000 * GIB, 5 * GIB).short(), Some(5 * GIB + 1));
        // A small disk keeps the 1 GiB floor.
        assert_eq!(disk(10 * GIB, GIB / 2).short(), Some(GIB / 2 + 1));
        // Out of inodes alone is low.
        let inodes = Disk {
            available_inodes: 32,
            ..disk(1000 * GIB, 500 * GIB)
        };
        assert_eq!(inodes.short(), Some(1));
    }

    #[test]
    fn a_warm_flight_needs_nothing_and_inodes_can_refuse_alone() {
        assert!(Space::of(0, 0).within(Space::default()));
        let tight = Disk {
            inodes: 1000,
            available_inodes: 40,
            ..disk(10 * GIB, 5 * GIB)
        };
        assert!(!Space::of(1, 1).within(free(&tight, Space::default())));
        let unlimited = Disk {
            inodes: 0,
            available_inodes: 0,
            ..tight
        };
        assert!(Space::of(1, 1000).within(free(&unlimited, Space::default())));
    }
}

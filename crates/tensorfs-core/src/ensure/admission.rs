//! Disk admission for one flight, and the GC policy that makes room. Disk only: nothing here
//! looks at RAM (owner rule: no RAM floor; the Runtime pages to disk).
//!
//! The numbers are the Host's storage-pressure policy (pod-supervisor `pressure.go`,
//! `fetch.go`): a reserve of max(1 GiB, 1/50 of the filesystem) and 32 inodes is never
//! admitted; a download needs its wanted bytes and 16 + 3 inodes per wanted object; live
//! flights' unwritten remainders are already spoken for. When a flight does not fit, the GC
//! policy runs in order: unreferenced bytes, delivered products (oldest delivery first),
//! then cached model lanes (the dedup-aware reclaim planner); the answer after that is the
//! answer.

use std::ops::Add;
use std::path::Path;

use crate::err::{refuse, Code, Refusal, Result};
use crate::gc;
use crate::source_artifact;
use crate::store::Store;

use super::flight::{self, Flight, Record};

pub(super) const RESERVE_BYTES: u64 = 1 << 30;
pub(super) const RESERVE_DIVISOR: u64 = 50;
pub(super) const RESERVE_INODES: u64 = 32;
pub(super) const METADATA_INODES: u64 = 16;
pub(super) const OBJECT_INODES: u64 = 3;
/// Pressure begins at 1/10 of the filesystem free and ends above 1/5.
pub(super) const ENTER_DIVISOR: u64 = 10;
pub(super) const LEAVE_DIVISOR: u64 = 5;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Space {
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

    fn within(self, free: Space) -> bool {
        self.bytes <= free.bytes && self.inodes <= free.inodes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Disk {
    pub capacity: u64,
    pub available: u64,
    /// Zero when the filesystem has no inode limit.
    pub inodes: u64,
    pub available_inodes: u64,
}

pub(super) fn measure(root: &Path) -> Result<Disk> {
    let stat = rustix::fs::statvfs(root).map_err(|e| Refusal {
        code: Code::CAPACITY_EXHAUSTED,
        detail: format!(
            "cannot measure the filesystem under {}: {e}; admission fails closed",
            root.display()
        ),
    })?;
    let unit = stat.f_frsize.max(1);
    Ok(Disk {
        capacity: stat.f_blocks.saturating_mul(unit),
        available: stat.f_bavail.saturating_mul(unit),
        inodes: stat.f_files,
        available_inodes: stat.f_favail,
    })
}

/// What `disk` admits once its reserve and `others` are set aside.
pub(super) fn free(disk: &Disk, others: Space) -> Space {
    let reserve = RESERVE_BYTES.max(disk.capacity / RESERVE_DIVISOR);
    Space {
        bytes: disk
            .available
            .saturating_sub(reserve)
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

/// The GC policy's outcome: what it freed, why a tier could not run, and the disk after.
struct Collected {
    bytes: u64,
    unable: Option<Refusal>,
    disk: Disk,
}

/// Run the GC policy's tiers until `short(disk)` says nothing is short. `short` answers
/// the bytes still missing, or `None` once the disk is enough.
fn collect_until(
    store: &Store,
    protected: &[String],
    short: &dyn Fn(&Disk) -> Option<u64>,
    collecting: &dyn Fn(),
) -> Result<Collected> {
    let root = store.root();
    let mut collected = Collected {
        bytes: 0,
        unable: None,
        disk: measure(root)?,
    };
    for tier in 0..3 {
        let Some(short) = short(&collected.disk) else {
            break;
        };
        if tier == 0 {
            collecting();
        }
        let freed = match tier {
            0 => gc::collect(root, false).map(|report| report.reclaimed_bytes),
            1 => release_delivered(store, short.max(1)),
            _ => gc::collect_cached_for(root, protected, short.max(1))
                .map(|report| report.reclaimed_bytes),
        };
        match freed {
            Ok(bytes) => collected.bytes += bytes,
            Err(refusal) => collected.unable = Some(refusal),
        }
        collected.disk = measure(root)?;
    }
    Ok(collected)
}

/// Live flights under the admission gate: what they still owe the disk, and their
/// manifests, which no tier may evict.
fn flights(store: &Store, mine: &str, keep: &[String]) -> Result<(Space, Vec<String>)> {
    let live = flight::live(store, mine)?;
    let owed = live
        .iter()
        .filter_map(|(_, record)| record.as_ref().map(Record::remaining))
        .fold(Space::default(), Add::add);
    let mut protected = keep.to_vec();
    protected.extend(live.into_iter().map(|(manifest, _)| manifest));
    protected.extend((!mine.is_empty()).then(|| mine.to_string()));
    Ok((owed, protected))
}

/// Admit `record`'s flight, collecting under the GC policy when it does not fit. Returns
/// the bytes collected. `keep` names manifests no tier may evict.
pub(super) fn admit(
    store: &Store,
    flight: &Flight,
    record: &Record,
    keep: &[String],
    collecting: &dyn Fn(),
) -> Result<u64> {
    let need = Space::of(record.wanted_bytes, record.wanted_objects);
    let _gate = flight::gate(store)?;
    let (others, protected) = flights(store, &flight.manifest, keep)?;
    let short = |disk: &Disk| {
        let free = free(disk, others);
        (!need.within(free)).then(|| need.bytes.saturating_sub(free.bytes))
    };
    let collected = collect_until(store, &protected, &short, collecting)?;
    if short(&collected.disk).is_none() {
        flight.record(record)?;
        return Ok(collected.bytes);
    }
    let free = free(&collected.disk, others);
    let inodes = match collected.disk.inodes {
        0 => String::new(),
        _ => format!(" ({} inodes, {} free)", need.inodes, free.inodes),
    };
    let why = collected
        .unable
        .map_or(String::new(), |r| format!("; GC could not run: {r}"));
    refuse(
        Code::CAPACITY_EXHAUSTED,
        format!(
            "this download needs {} B{inodes}; the disk under {} admits {} B after its \
             reserve and {} B still owed to other downloads; GC freed {} B{why}",
            need.bytes,
            store.root().display(),
            free.bytes,
            others.bytes,
            collected.bytes,
        ),
    )
}

/// What one pressure pass found and did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relief {
    /// The disk was under pressure when the pass began.
    pub pressure: bool,
    pub collected_bytes: u64,
    pub capacity_bytes: u64,
    pub available_bytes: u64,
    /// Why a tier could not run, when one could not.
    pub unable: Option<String>,
}

/// The Host's pressure hysteresis in one pass: a disk with at most 1/10 free (or under its
/// reserve) is collected under the GC policy until more than 1/5 is free. Live flights'
/// manifests and `keep` are never evicted. A disk not under pressure is left alone.
pub fn relieve(store: &Store, keep: &[String]) -> Result<Relief> {
    let _gate = flight::gate(store)?;
    let (_, protected) = flights(store, "", keep)?;
    let disk = measure(store.root())?;
    let reserve = RESERVE_BYTES.max(disk.capacity / RESERVE_DIVISOR);
    let low = |disk: &Disk, divisor: u64| {
        disk.available <= reserve.max(disk.capacity / divisor)
            || (disk.inodes > 0
                && disk.available_inodes <= RESERVE_INODES.max(disk.inodes / divisor))
    };
    let pressure = low(&disk, ENTER_DIVISOR);
    let short = |disk: &Disk| {
        low(disk, LEAVE_DIVISOR).then(|| {
            (reserve.max(disk.capacity / LEAVE_DIVISOR) + 1).saturating_sub(disk.available)
        })
    };
    let collected = if pressure {
        collect_until(store, &protected, &short, &|| {})?
    } else {
        Collected {
            bytes: 0,
            unable: None,
            disk,
        }
    };
    Ok(Relief {
        pressure,
        collected_bytes: collected.bytes,
        capacity_bytes: collected.disk.capacity,
        available_bytes: collected.disk.available,
        unable: collected.unable.map(|r| r.to_string()),
    })
}

/// Release delivered products oldest delivery first, in doubling batches, collecting after
/// each batch, until `short` bytes are freed or none are left.
fn release_delivered(store: &Store, short: u64) -> Result<u64> {
    let owners = source_artifact::delivered(store)?;
    let (mut freed, mut at, mut batch) = (0u64, 0usize, 1usize);
    while at < owners.len() && freed < short {
        for owner in &owners[at..(at + batch).min(owners.len())] {
            match source_artifact::release(store, owner) {
                Err(refusal) if refusal.code == Code::STORE_BUSY => {}
                other => other?,
            }
        }
        at += batch;
        batch *= 2;
        freed += gc::collect(store.root(), false)?.reclaimed_bytes;
    }
    Ok(freed)
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
    fn the_reserve_is_a_gib_or_a_fiftieth_and_others_are_spoken_for() {
        // 10 GiB disk: the reserve is the 1 GiB floor.
        assert_eq!(
            free(&disk(10 * GIB, 5 * GIB), Space::default()).bytes,
            4 * GIB
        );
        // 1000 GiB disk: the reserve is 20 GiB.
        assert_eq!(
            free(&disk(1000 * GIB, 100 * GIB), Space::default()).bytes,
            80 * GIB
        );
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

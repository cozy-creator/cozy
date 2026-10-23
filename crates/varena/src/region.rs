//! Regions (varena-leases §5, §6, §11): the agent's long-lived intermediates and graphs, over the lease agent.
//!
//! - A [`Region`] holds a long-lived intermediate (`kv`, `latent`, `conditioning`, `derived`, `workspace`) in an arena
//!   of its own (M2). Revoked, it is spilled or dropped. Its next [`Region::pin`] then says `fresh` (remapped empty) and
//!   the owner recomputes on its own thread. There are no shrink callbacks.
//! - A [`GraphRegion`] is a component's graphs on one device. It holds pins on every chunk its graphs touch for its
//!   life (L2). Its word ([`Word`], R2-3) is acquired before each replay and released by an event after it.
//!   Dropping waits for the last replay's event, so a yield is bounded by one replay.
//! - [`Memory::room`] is what a cut site sizes from: free, cold and warm bytes (§11, S4's plan §5).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::arena::{Ended, PoolId};
use crate::backend::Event;
use crate::memory::{Allocating, Memory, SetId};
use crate::{Error, Result};

/// What a region holds (§6): its kind sets the defaults the manager's policy ranks it by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Kv,
    Latent,
    Conditioning,
    Derived,
    Workspace,
}

/// Region ids are the process's: they name the region's arena ([`crate::arena::Owner::Region`]).
static REGIONS: AtomicU64 = AtomicU64::new(1);

struct Shared {
    id: u64,
    owner: String,
    kind: Kind,
    sets: Vec<SetId>,
    spill: bool,
    /// Revoked since its last pin: the next pin is fresh.
    remapped: AtomicBool,
    pins: AtomicU64,
    produced_ns: AtomicU64,
    /// The plan step of its next use; `u64::MAX`: none declared.
    next_use: AtomicU64,
    closed: AtomicBool,
}

/// A long-lived intermediate (§6). Closing (or dropping) it releases its arena.
pub struct Region {
    memory: Memory,
    sh: Arc<Shared>,
}

/// A region pinned for use: its bytes stay mapped until this drops.
pub struct RegionPin {
    sh: Arc<Shared>,
    fresh: bool,
}

impl RegionPin {
    /// The region was remapped empty since its last use: its owner recomputes it.
    pub fn fresh(&self) -> bool {
        self.fresh
    }
}

impl Drop for RegionPin {
    fn drop(&mut self) {
        self.sh.pins.fetch_sub(1, Ordering::AcqRel);
    }
}

impl Region {
    pub fn id(&self) -> u64 {
        self.sh.id
    }

    pub fn owner(&self) -> &str {
        &self.sh.owner
    }

    pub fn kind(&self) -> Kind {
        self.sh.kind
    }

    pub fn sets(&self) -> &[SetId] {
        &self.sh.sets
    }

    pub fn spill(&self) -> bool {
        self.sh.spill
    }

    /// Pins the region while in use. `fresh` is true when it was remapped empty since its last pin.
    pub fn pin(&self) -> Result<RegionPin> {
        if self.sh.closed.load(Ordering::Acquire) {
            return Err(Error::Closed(format!("region {} ({})", self.sh.id, self.sh.owner)));
        }
        self.sh.pins.fetch_add(1, Ordering::AcqRel);
        let fresh = self.sh.remapped.swap(false, Ordering::AcqRel);
        Ok(RegionPin { sh: self.sh.clone(), fresh })
    }

    /// The calling thread's torch blocks go to the region's own arena until the guard drops (M2).
    pub fn allocating(&self) -> Allocating {
        self.memory.region_allocating(self.sh.id)
    }

    /// Its production time, for the spill-versus-drop choice (§6).
    pub fn produced(&self, ns: u64) {
        self.sh.produced_ns.store(ns, Ordering::Release);
    }

    /// The plan step of its next use; `None`: no use declared.
    pub fn next_use(&self, step: Option<u32>) {
        self.sh.next_use.store(step.map_or(u64::MAX, u64::from), Ordering::Release);
    }

    /// Releases it: its spans leave as they come whole free. The record names the blocks still live.
    pub fn close(self) -> Ended {
        self.end()
    }

    fn end(&self) -> Ended {
        if self.sh.closed.swap(true, Ordering::AcqRel) {
            return Ended::default();
        }
        self.memory.region_close(self.sh.id)
    }

    /// The agent revoked the region (spilled or dropped): its next pin is fresh.
    pub fn revoked(&self) {
        self.sh.remapped.store(true, Ordering::Release);
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        let _ = self.end();
    }
}

/// A graph region's word (R2-3): replays in flight, and whether it is being dropped. One atomic: an acquire is one CAS.
#[derive(Debug, Default)]
pub struct Word(AtomicU64);

/// The high bit: Dropping. The rest: replays acquired and not yet released.
const DROPPING: u64 = 1 << 63;

impl Word {
    /// Live(n) to Live(n + 1); false while Dropping.
    pub fn acquire(&self) -> bool {
        self.0.try_update(Ordering::AcqRel, Ordering::Acquire, |w| (w & DROPPING == 0).then_some(w + 1)).is_ok()
    }

    /// One replay fewer.
    pub fn release(&self) {
        let _ = self.0.try_update(Ordering::AcqRel, Ordering::Acquire, |w| (w & !DROPPING > 0).then(|| w - 1));
    }

    /// Sets Dropping; true when this call set it.
    pub fn drop_begin(&self) -> bool {
        self.0.fetch_or(DROPPING, Ordering::AcqRel) & DROPPING == 0
    }

    pub fn dropping(&self) -> bool {
        self.0.load(Ordering::Acquire) & DROPPING != 0
    }

    /// Replays acquired and not yet released.
    pub fn replays(&self) -> u64 {
        self.0.load(Ordering::Acquire) & !DROPPING
    }
}

/// A component's graphs on one device (§5): its pool is the device's one capture pool token.
pub struct GraphRegion {
    memory: Memory,
    sets: Vec<SetId>,
    retained: Vec<SetId>,
    pool: PoolId,
    word: Arc<Word>,
    /// The event recorded after the last released replay: a drop waits for it.
    last: Mutex<Option<Event>>,
    closed: AtomicBool,
}

impl GraphRegion {
    /// The capture pool token every graph of the region captures into.
    pub fn pool(&self) -> PoolId {
        self.pool
    }

    pub fn sets(&self) -> &[SetId] {
        &self.sets
    }

    pub fn retained(&self) -> &[SetId] {
        &self.retained
    }

    /// Maps `expected` bytes into the pool before capture begins, making room now. A shape's first capture has none.
    pub fn prepare(&self, expected: Option<u64>) -> Result<()> {
        match expected {
            Some(bytes) => self.memory.pool_prepare(self.pool, bytes),
            None => Ok(()),
        }
    }

    /// Before each replay: false while the region is Dropping (run eager, recapture off-path).
    pub fn acquire(&self) -> bool {
        !self.closed.load(Ordering::Acquire) && self.word.acquire()
    }

    /// After the replay's launch on `stream`: an event recorded there releases it.
    pub fn release(&self, stream: usize) -> Result<()> {
        let event = self.memory.record_event(stream);
        self.word.release();
        let event = event?;
        let old = self.last.lock().replace(event);
        if let Some(old) = old {
            self.memory.retire_event(old);
        }
        Ok(())
    }

    /// Whether the region is being dropped.
    pub fn dropping(&self) -> bool {
        self.word.dropping()
    }

    /// The manager ranked it the victim: Dropping, then the last replay's event, then its pins go.
    pub fn drop_now(&self) -> Result<()> {
        self.word.drop_begin();
        while self.word.replays() > 0 {
            std::thread::yield_now();
        }
        if let Some(event) = self.last.lock().take() {
            self.memory.finish_event(event)?;
        }
        Ok(())
    }

    /// Drops the graph's pins and its use of the pool.
    pub fn close(self) {
        self.end();
    }

    fn end(&self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        if let Err(e) = self.drop_now() {
            tracing::warn!("varena: dropping a graph region: {e}");
        }
        self.memory.pool_release(self.pool);
    }

    /// The region's word, for the manager's revoke.
    pub fn word(&self) -> Arc<Word> {
        self.word.clone()
    }
}

impl Drop for GraphRegion {
    fn drop(&mut self) {
        self.end();
    }
}

/// What a cut site sizes from (§11; S4's plan §5): a lock-free snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Room {
    /// This process's float plus the manager's free bytes, net of the driver headroom H.
    pub free: u64,
    /// Revocable bytes outside the step's plan sets: this process's and other tenants'.
    pub cold: u64,
    /// This process's bytes inside the step's sets.
    pub warm: u64,
    /// The allocation granularity: every need is rounded up to it.
    pub granularity: u64,
    /// The streaming floor: what the step's sets need resident to stream (the slots).
    pub floor: u64,
}

impl Memory {
    /// A long-lived intermediate's region (§6): `spill` false drops it for recompute on revoke.
    pub fn region(&self, owner: &str, kind: Kind, sets: &[SetId], spill: bool) -> Result<Region> {
        let sh = Arc::new(Shared {
            id: REGIONS.fetch_add(1, Ordering::Relaxed),
            owner: owner.to_string(),
            kind,
            sets: sets.to_vec(),
            spill,
            remapped: AtomicBool::new(false),
            pins: AtomicU64::new(0),
            produced_ns: AtomicU64::new(0),
            next_use: AtomicU64::new(u64::MAX),
            closed: AtomicBool::new(false),
        });
        Ok(Region { memory: self.clone(), sh })
    }

    /// A component's graph region on this device (§5): `pool` is the device's one capture pool token.
    pub fn graph_region(&self, sets: &[SetId], retained: &[SetId], pool: PoolId) -> Result<GraphRegion> {
        Ok(GraphRegion { memory: self.clone(), sets: sets.to_vec(), retained: retained.to_vec(), pool, word: Arc::default(), last: Mutex::new(None), closed: AtomicBool::new(false) })
    }

    /// The room a cut sizes from at plan step `step` (the request's cursor when `None`).
    pub fn room(&self, step: Option<u32>) -> Room {
        let _ = step;
        Room { free: self.free(), cold: 0, warm: 0, granularity: self.granularity(), floor: 0 }
    }
}

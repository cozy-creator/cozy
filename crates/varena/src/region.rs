//! Regions (varena-leases §5, §6, §11, §12): the agent's long-lived intermediates and graphs.
//!
//! - A [`Region`] holds a long-lived intermediate (`kv`, `latent`, `conditioning`, `derived`, `workspace`) in an arena
//!   of its own (M2). Reclaimed only while unpinned, it is swapped to a RAM file or discarded, keeping its addresses.
//!   Its next [`Region::pin`] maps it again there: `fresh` says it was discarded, and its owner recomputes it on its
//!   own thread. There are no shrink callbacks.
//! - A [`GraphRegion`] is a component's graphs on one device. It pins every weight chunk its graphs read and holds a
//!   use of its pool for its life (L2). Its [`Word`] (R2-3) is acquired before each replay and released by an event
//!   recorded after it, one per stream. A reclaim sets Dropping and waits for those events, so it is bounded by one
//!   replay.
//! - [`Memory::room`] is what a cut site sizes from: free, cold and warm bytes, and the streaming floor.
//! - [`Memory::region_reclaim`] is the manager's Reclaim of one region, whole; [`Memory::region_records`] its view.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, OnceLock, Weak};

use parking_lot::Mutex;

use crate::arena::{Ended, Evicted, Owner, PoolId};
use crate::backend::Event;
use crate::memory::{Allocating, Memory, Pin, Retained, SetId};
use crate::table::SLOTS;
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

/// How the manager may reclaim a region (the Region frame's kind).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Reclaimable {
    Graph,
    Swappable,
    Discardable,
}

/// One region as the manager's policy sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub id: u64,
    pub kind: Reclaimable,
    /// Bytes it maps now (0 once reclaimed).
    pub bytes: u64,
    /// The lease slots of the chunks it holds (none standalone).
    pub slots: Vec<u32>,
    /// Its measured production (or capture) time.
    pub cost_ns: u64,
    /// Its last use, CLOCK_MONOTONIC ns (machine-wide).
    pub last_use: u64,
    /// Its advice: the plan step of its next use.
    pub next_use: Option<u32>,
}

/// What a reclaim did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reclaimed {
    /// Pinned, or a capture is open: it is reclaimed at its last unpin.
    Busy,
    /// Closed, never existed, or reclaimed already.
    Gone,
    Swapped(u64),
    Discarded(u64),
    /// A graph region: Dropping, its replays completed, its pins gone.
    Dropped,
}

/// Region ids are the process's: they name the region's arena ([`Owner::Region`]).
static IDS: AtomicU64 = AtomicU64::new(1);

enum Live {
    Cache(Weak<Shared>),
    Graph(Weak<Graph>),
}

enum Found {
    Cache(Arc<Shared>),
    Graph(Arc<Graph>),
}

impl Live {
    fn upgrade(&self) -> Option<Found> {
        match self {
            Live::Cache(w) => w.upgrade().map(Found::Cache),
            Live::Graph(w) => w.upgrade().map(Found::Graph),
        }
    }
}

/// Every live region of the process: the manager's Reclaim finds its victim here.
static LIVE: Mutex<BTreeMap<u64, Live>> = Mutex::new(BTreeMap::new());

/// The machine-wide clock the manager ranks regions by (CLOCK_MONOTONIC ns); elsewhere, this
/// process's own.
fn now_ns() -> u64 {
    #[cfg(unix)]
    return crate::memory::monotonic_ns();
    #[cfg(not(unix))]
    {
        static START: OnceLock<std::time::Instant> = OnceLock::new();
        START.get_or_init(std::time::Instant::now).elapsed().as_nanos() as u64
    }
}

/// A reclaim deferred by a pin runs here at the last unpin: never on the unpinning thread, which
/// may hold the GIL while the eviction waits for the region's streams.
fn deferred() -> Option<&'static Sender<u64>> {
    static WORKER: OnceLock<Option<Sender<u64>>> = OnceLock::new();
    WORKER
        .get_or_init(|| {
            let (tx, rx) = mpsc::channel::<u64>();
            let spawned = std::thread::Builder::new().name("varena-reclaim".into()).spawn(move || {
                while let Ok(id) = rx.recv() {
                    let shared = match LIVE.lock().get(&id) {
                        Some(Live::Cache(w)) => w.upgrade(),
                        _ => None,
                    };
                    if let Some(sh) = shared {
                        if let Err(e) = sh.reclaim(None) {
                            tracing::warn!("varena: reclaiming region {id} ({}): {e}", sh.owner);
                        }
                    }
                }
            });
            match spawned {
                Ok(_) => Some(tx),
                Err(e) => {
                    tracing::error!("varena: no reclaim thread ({e}): a pinned region's reclaim waits for the manager's next ask");
                    None
                }
            }
        })
        .as_ref()
}

#[derive(Default)]
struct Res {
    pins: u32,
    closed: bool,
    evicted: Option<Evicted>,
}

/// A reclaim asked and not yet done: none, discard, swap.
const ASKED_NONE: u8 = 0;
const ASKED_DISCARD: u8 = 1;
const ASKED_SWAP: u8 = 2;

struct Shared {
    id: u64,
    owner: String,
    kind: Kind,
    sets: Vec<SetId>,
    spill: bool,
    memory: Memory,
    /// Taken by a pin (making it resident may wait for the manager), an unpin, a reclaim and the
    /// close. A reclaim only tries it: it never waits behind a pin that waits for the manager.
    res: Mutex<Res>,
    /// Asked while pinned or locked: the last unpin hands it to the reclaim thread.
    asked: AtomicU8,
    produced_ns: AtomicU64,
    /// The plan step of its next use; `u64::MAX`: none declared.
    next_use: AtomicU64,
    last_use: AtomicU64,
}

impl Shared {
    fn pin(self: &Arc<Self>, route: bool) -> Result<RegionPin> {
        let mut r = self.res.lock();
        if r.closed {
            return Err(Error::Closed(format!("region {} ({})", self.id, self.owner)));
        }
        let remapped = r.evicted.is_some();
        let fresh = match r.evicted.as_mut() {
            Some(ev) => self.memory.torch()?.make_resident(Owner::Region(self.id), ev)?,
            None => false,
        };
        r.evicted = None;
        r.pins += 1;
        drop(r);
        if remapped {
            self.publish();
        }
        let route = route.then(|| self.memory.region_allocating(self.id));
        Ok(RegionPin { sh: self.clone(), fresh, _route: route })
    }

    fn record(&self) -> Record {
        let torch = self.memory.torch_if();
        let owner = Owner::Region(self.id);
        Record {
            id: self.id,
            kind: if self.spill { Reclaimable::Swappable } else { Reclaimable::Discardable },
            bytes: torch.as_ref().map_or(0, |t| t.mapped_by(owner)),
            slots: torch.as_ref().map_or_else(Vec::new, |t| t.slots(owner)),
            cost_ns: self.produced_ns.load(Ordering::Acquire),
            last_use: self.last_use.load(Ordering::Acquire),
            next_use: u32::try_from(self.next_use.load(Ordering::Acquire)).ok(),
        }
    }

    /// Its facts to the lease manager, at its create and its changes.
    fn publish(&self) {
        self.memory.region_facts(self.id, Some(&self.record()));
    }

    fn unpin(&self) {
        self.last_use.store(now_ns(), Ordering::Release);
        let last = {
            let mut r = self.res.lock();
            r.pins = r.pins.saturating_sub(1);
            r.pins == 0 && !r.closed
        };
        // Read after the lock is let go: an ask that found it locked was stored before its try.
        if last && self.asked.load(Ordering::Acquire) != ASKED_NONE {
            if let Some(tx) = deferred() {
                let _ = tx.send(self.id);
            }
        }
    }

    /// `swap`: the manager's choice (`None`: the one asked before); a region that does not spill is
    /// discarded. Busy while pinned, while another holds its lock, or while a capture is open.
    fn reclaim(&self, swap: Option<bool>) -> Result<Reclaimed> {
        if let Some(swap) = swap {
            self.asked.store(if swap { ASKED_SWAP } else { ASKED_DISCARD }, Ordering::Release);
        }
        let Some(mut r) = self.res.try_lock() else { return Ok(Reclaimed::Busy) };
        let swap = match self.asked.load(Ordering::Acquire) {
            ASKED_NONE => return Ok(Reclaimed::Gone),
            asked => asked == ASKED_SWAP,
        };
        if r.closed || r.evicted.is_some() {
            self.asked.store(ASKED_NONE, Ordering::Release);
            return Ok(Reclaimed::Gone);
        }
        if r.pins > 0 {
            return Ok(Reclaimed::Busy);
        }
        let Some(torch) = self.memory.torch_if() else { return Ok(Reclaimed::Gone) };
        let Some(ev) = torch.evict(Owner::Region(self.id), swap && self.spill)? else { return Ok(Reclaimed::Busy) };
        self.asked.store(ASKED_NONE, Ordering::Release);
        let (bytes, swapped) = (ev.bytes, ev.swapped());
        r.evicted = Some(ev);
        drop(r);
        self.publish();
        Ok(if swapped { Reclaimed::Swapped(bytes) } else { Reclaimed::Discarded(bytes) })
    }

    fn end(&self) -> Ended {
        let mut r = self.res.lock();
        if std::mem::replace(&mut r.closed, true) {
            return Ended::default();
        }
        r.evicted = None;
        self.asked.store(ASKED_NONE, Ordering::Release);
        drop(r);
        LIVE.lock().remove(&self.id);
        self.memory.region_facts(self.id, None);
        if let Some(t) = self.memory.torch_if() {
            t.forget_evicted(Owner::Region(self.id));
        }
        self.memory.region_close(self.id)
    }
}

/// A long-lived intermediate (§6). Closing (or dropping) it releases its arena.
pub struct Region {
    sh: Arc<Shared>,
}

/// A region pinned for use: its bytes stay mapped until this drops.
pub struct RegionPin {
    sh: Arc<Shared>,
    fresh: bool,
    _route: Option<Allocating>,
}

impl RegionPin {
    /// The region was discarded since its last use: its owner recomputes it.
    pub fn fresh(&self) -> bool {
        self.fresh
    }
}

impl Drop for RegionPin {
    fn drop(&mut self) {
        self.sh.unpin();
        // Filled: its arena's chunks changed.
        if self._route.is_some() {
            self.sh.publish();
        }
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

    /// Pins the region while in use, mapping it again first if it was reclaimed (waiting for a
    /// reclaim in progress). `fresh`: it was discarded, so its owner recomputes it.
    pub fn pin(&self) -> Result<RegionPin> {
        self.sh.pin(false)
    }

    /// [`Region::pin`], and the calling thread's torch blocks go to the region's own arena (M2)
    /// until the pin drops.
    pub fn allocating(&self) -> Result<RegionPin> {
        self.sh.pin(true)
    }

    /// Its production time, for the swap-versus-discard choice (§6).
    pub fn produced(&self, ns: u64) {
        self.sh.produced_ns.store(ns, Ordering::Release);
        self.sh.publish();
    }

    /// The plan step of its next use; `None`: no use declared.
    pub fn next_use(&self, step: Option<u32>) {
        self.sh.next_use.store(step.map_or(u64::MAX, u64::from), Ordering::Release);
        self.sh.publish();
    }

    /// Releases it: its spans leave as they come whole free. The record names the blocks still live.
    pub fn close(self) -> Ended {
        self.sh.end()
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        let _ = self.sh.end();
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

#[derive(Default)]
struct Held {
    /// Per stream, the event recorded after its last released replay.
    events: HashMap<usize, Event>,
    /// A replay whose completion is unknown (no event, no synchronize): its pins are never released.
    lost: bool,
    pins: Vec<Pin>,
    bindings: Vec<Retained>,
}

struct Graph {
    id: u64,
    memory: Memory,
    sets: Vec<SetId>,
    retained: Vec<SetId>,
    pool: PoolId,
    word: Arc<Word>,
    held: Mutex<Held>,
    /// Released once: its pins and its pool use.
    dropped: AtomicBool,
    closed: AtomicBool,
    capture_ns: AtomicU64,
    last_use: AtomicU64,
}

impl Graph {
    fn release(&self, stream: usize) -> Result<()> {
        let recorded = self.memory.record_event(stream).or_else(|e| {
            tracing::warn!("varena: recording a graph replay's event failed ({e}); synchronizing its stream");
            self.memory.synchronize(stream).map(|()| Event::DONE)
        });
        // The event is stored before the word lets a drop see this replay gone.
        let out = {
            let mut h = self.held.lock();
            match recorded {
                Ok(event) => {
                    if let Some(old) = h.events.insert(stream, event) {
                        self.memory.retire_event(old);
                    }
                    Ok(())
                }
                Err(e) => {
                    h.lost = true;
                    Err(e)
                }
            }
        };
        self.last_use.store(now_ns(), Ordering::Release);
        self.word.release();
        out
    }

    /// Dropping, then every stream's last replay completed, then its weight pins go. Once.
    fn drop_now(&self) -> Result<Reclaimed> {
        self.word.drop_begin();
        while self.word.replays() > 0 {
            std::thread::yield_now();
        }
        let mut h = self.held.lock();
        if self.dropped.load(Ordering::Acquire) {
            return Ok(Reclaimed::Gone);
        }
        if h.lost {
            return Err(Error::Internal(format!("graph region {}: a replay's completion is unknown, so its pins stay", self.id)));
        }
        while let Some((&stream, &event)) = h.events.iter().next() {
            self.memory.finish_event(event)?;
            h.events.remove(&stream);
            self.memory.retire_event(event);
        }
        h.pins.clear();
        h.bindings.clear();
        self.dropped.store(true, Ordering::Release);
        drop(h);
        if !self.closed.load(Ordering::Acquire) {
            self.publish();
        }
        Ok(Reclaimed::Dropped)
    }

    fn record(&self) -> Record {
        let dropped = self.dropped.load(Ordering::Acquire);
        let mut slots = Vec::new();
        if !dropped {
            let sets: Vec<SetId> = self.sets.iter().chain(&self.retained).copied().collect();
            slots = self.memory.lease_slots(&sets);
            slots.extend(self.memory.torch_if().map_or_else(Vec::new, |t| t.slots(Owner::Pool(self.pool))));
        }
        Record {
            id: self.id,
            kind: Reclaimable::Graph,
            bytes: if dropped { 0 } else { self.held.lock().pins.iter().map(|p| p.len).sum() },
            slots,
            cost_ns: self.capture_ns.load(Ordering::Acquire),
            last_use: self.last_use.load(Ordering::Acquire),
            next_use: None,
        }
    }

    fn publish(&self) {
        self.memory.region_facts(self.id, Some(&self.record()));
    }

    fn end(&self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        LIVE.lock().remove(&self.id);
        if let Err(e) = self.drop_now() {
            tracing::warn!("varena: dropping graph region {}: {e}", self.id);
        }
        self.memory.region_facts(self.id, None);
        self.memory.pool_release(self.pool);
    }
}

/// A component's graphs on one device (§5): its pool is the device's one capture pool token.
pub struct GraphRegion {
    g: Arc<Graph>,
}

impl GraphRegion {
    pub fn id(&self) -> u64 {
        self.g.id
    }

    /// The capture pool token every graph of the region captures into.
    pub fn pool(&self) -> PoolId {
        self.g.pool
    }

    pub fn sets(&self) -> &[SetId] {
        &self.g.sets
    }

    pub fn retained(&self) -> &[SetId] {
        &self.g.retained
    }

    /// Maps `expected` bytes into the pool before capture begins, making room now. A shape's first capture has none.
    pub fn prepare(&self, expected: Option<u64>) -> Result<()> {
        match expected {
            Some(bytes) => self.g.memory.pool_prepare(self.g.pool, bytes),
            None => Ok(()),
        }
    }

    /// A capture's measured time, for the manager's ranking.
    pub fn captured(&self, ns: u64) {
        self.g.capture_ns.store(ns, Ordering::Release);
        self.g.publish();
    }

    /// Before each replay: false while the region is Dropping (run eager, recapture off-path).
    pub fn acquire(&self) -> bool {
        !self.g.closed.load(Ordering::Acquire) && self.g.word.acquire()
    }

    /// After the replay's launch on `stream`: an event recorded there releases it. Pair every true
    /// [`GraphRegion::acquire`] with one release.
    pub fn release(&self, stream: usize) -> Result<()> {
        self.g.release(stream)
    }

    /// Whether the region is being dropped.
    pub fn dropping(&self) -> bool {
        self.g.word.dropping()
    }

    /// Drops the graph's pins and its use of the pool.
    pub fn close(self) {
        self.g.end();
    }
}

impl Drop for GraphRegion {
    fn drop(&mut self) {
        self.g.end();
    }
}

/// What a cut site sizes from (§11; S4's plan §5).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Room {
    /// Allocatable now: the manager's free bytes net of the reserved pool (standalone: the budget
    /// left), plus this process's free list (its idle spans).
    pub free: u64,
    /// Reclaimable without touching the step's sets: the manager's figure for other tenants and its
    /// page cache, plus this process's unpinned regions and its idle pages outside the step's sets.
    pub cold: u64,
    /// This process's bytes inside the step's sets.
    pub warm: u64,
    /// The allocation granularity: every need is rounded up to it.
    pub granularity: u64,
    /// The streaming floor: SLOTS × the largest block of the step's sets.
    pub floor: u64,
}

impl Memory {
    /// A long-lived intermediate's region (§6): `spill` false discards it for recompute on reclaim.
    pub fn region(&self, owner: &str, kind: Kind, sets: &[SetId], spill: bool) -> Result<Region> {
        let sh = Arc::new(Shared {
            id: IDS.fetch_add(1, Ordering::Relaxed),
            owner: owner.to_string(),
            kind,
            sets: sets.to_vec(),
            spill,
            memory: self.clone(),
            res: Mutex::new(Res::default()),
            asked: AtomicU8::new(ASKED_NONE),
            produced_ns: AtomicU64::new(0),
            next_use: AtomicU64::new(u64::MAX),
            last_use: AtomicU64::new(now_ns()),
        });
        LIVE.lock().insert(sh.id, Live::Cache(Arc::downgrade(&sh)));
        sh.publish();
        Ok(Region { sh })
    }

    /// A component's graph region on this device (§5): `pool` is the device's one capture pool
    /// token. Every block of `sets` and `retained` is pinned (made resident now) for its life.
    pub fn graph_region(&self, sets: &[SetId], retained: &[SetId], pool: PoolId) -> Result<GraphRegion> {
        let mut held = Held::default();
        for &set in sets.iter().chain(retained) {
            let binding = self.retain(set, &self.blocks(set)?)?;
            held.pins.extend(binding.prepare(0)?);
            held.bindings.push(binding);
        }
        self.torch()?.pool_retain(pool);
        let g = Arc::new(Graph {
            id: IDS.fetch_add(1, Ordering::Relaxed),
            memory: self.clone(),
            sets: sets.to_vec(),
            retained: retained.to_vec(),
            pool,
            word: Arc::default(),
            held: Mutex::new(held),
            dropped: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            capture_ns: AtomicU64::new(0),
            last_use: AtomicU64::new(now_ns()),
        });
        LIVE.lock().insert(g.id, Live::Graph(Arc::downgrade(&g)));
        g.publish();
        Ok(GraphRegion { g })
    }

    /// The manager's Reclaim of region `id`, whole (from the agent's mechanism thread, never its
    /// reader): a cache region is swapped (`swap`, when it spills) or discarded once unpinned; a graph
    /// region is dropped.
    pub fn region_reclaim(&self, id: u64, swap: bool) -> Result<Reclaimed> {
        let found = LIVE.lock().get(&id).and_then(Live::upgrade);
        match found {
            Some(Found::Cache(sh)) if sh.memory.same(self) => {}
            Some(Found::Graph(g)) if g.memory.same(self) => {}
            _ => return Ok(Reclaimed::Gone),
        }
        reclaim(id, swap)
    }

    /// This device's live regions, as the manager's policy sees them.
    pub fn region_records(&self) -> Vec<Record> {
        let live: Vec<Found> = LIVE.lock().values().filter_map(Live::upgrade).collect();
        live.into_iter()
            .filter_map(|l| match l {
                Found::Cache(sh) if sh.memory.same(self) => Some(sh.record()),
                Found::Graph(g) if g.memory.same(self) => Some(g.record()),
                _ => None,
            })
            .collect()
    }

    /// The room a cut sizes from at plan step `step` (each open request's current step when `None`).
    pub fn room(&self, step: Option<u32>) -> Room {
        let (warm, idle_pages, largest) = self.step_bytes(step);
        let torch = self.torch_if();
        let float = torch.as_ref().map_or(0, |t| t.idle().0);
        let unpinned: Vec<u64> = LIVE
            .lock()
            .iter()
            .filter_map(|(id, l)| match l {
                // Locked (a pin making it resident, a reclaim): not cold either way.
                Live::Cache(w) => w.upgrade().filter(|s| s.memory.same(self) && s.res.try_lock().is_some_and(|r| r.pins == 0)).map(|_| *id),
                Live::Graph(_) => None,
            })
            .collect();
        let regions: u64 = torch.as_ref().map_or(0, |t| unpinned.iter().map(|id| t.mapped_by(Owner::Region(*id))).sum());
        let (free, others) = (self.spare(), self.figures().map_or(0, |f| f.revocable));
        Room { free: free + float, cold: others + regions + idle_pages, warm, granularity: self.granularity(), floor: SLOTS * largest }
    }
}

/// The manager's Reclaim of region `id`, whole: a cache region is swapped (`swap`, when it spills)
/// or discarded once unpinned, a graph region dropped. Region ids are the process's, so the agent's
/// mechanism thread finds it by id alone.
pub(crate) fn reclaim(id: u64, swap: bool) -> Result<Reclaimed> {
    // Bound first: the registry's lock is never held across a reclaim's wait.
    let found = LIVE.lock().get(&id).and_then(Live::upgrade);
    match found {
        Some(Found::Cache(sh)) => sh.reclaim(Some(swap)),
        Some(Found::Graph(g)) => g.drop_now(),
        None => Ok(Reclaimed::Gone),
    }
}

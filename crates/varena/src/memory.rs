//! One executor's memory on one device (varena-leases). Leased, every page is a chunk the
//! manager allocates, and room between processes is its reclaim. Standalone (no compatible
//! manager), pages come from the device within a budget fixed at open (its free bytes less the
//! reserve):
//! `make_room` takes its own pages in [`crate::decide::victims`]' order, and a shortfall is
//! `Exhausted`. One `parking_lot` mutex guards the table and the mechanism's state; it is never
//! held across a fill, a wait on an event or a caller's callback.
//!
//! Lock order: none. The state lock is only ever taken alone; every other lock (the release
//! receiver, the fill queue) is taken alone or inside the state lock and never takes another,
//! and the state lock is never taken while one of them is held. Fills run on fill threads
//! (demand misses before prefetches); pin releases arrive through a channel. A panic is caught at
//! every public call and on every thread: the call fails `Internal`, the table is derived again
//! from the mappings, and the process keeps serving.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

#[cfg(unix)]
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use crossbeam_channel::{select_biased, unbounded, Receiver, Sender};
use std::time::Instant;

use parking_lot::{Condvar, Mutex, MutexGuard};
use platform::contain::host_memory;

#[cfg(unix)]
use tensorfs::io::source::Placement;
use tensorfs::io::Dst;

use crate::backend::{Access, Allowance, Backend, Event, Physical, Va};
use crate::decide::{self, Victim};
use crate::table::{Class, Fill, Key, Level, Page, Plan, Req, Set, SetSpec, Table};
use crate::{Error, Exhausted, GpuId, Result};

mod stream;
mod torch;
pub use torch::{Allocating, PoolId, TorchStats};
use stream::{StreamIn, Streaming};

/// A loaded set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SetId(pub u32);

/// A graph's immutable address/layout/fill binding. Retention keeps the virtual reservation,
/// never its physical pages. `prepare` leases the complete declared range before launch.
pub struct Retained {
    inner: Owner,
    set: u32,
    blocks: Vec<u32>,
    identity: String,
}

/// Preparation's claims prevent eviction of any dependency while all its pages are admitted.
/// On failure they unwind without launching a consumer; completed fills may remain cached.
struct Preparing {
    inner: Owner,
    pins: Vec<(u64, Key)>,
}

impl Drop for Preparing {
    fn drop(&mut self) {
        let mut st = self.inner.lock();
        for (id, key) in &self.pins {
            st.pins.remove(id);
            st.sync(*key);
        }
        self.inner.close_sets(&mut st);
        self.inner.landed.notify_all();
    }
}

impl Retained {
    pub fn identity(&self) -> &str { &self.identity }

    pub fn valid(&self) -> bool {
        let st = self.inner.state.lock();
        st.mech(self.set).is_ok_and(|m| retained_identity(self.set, m, &self.blocks) == self.identity)
    }

    /// Every block is mapped and filled now: a read (no pin, no room made), for a capture that
    /// may only run into what is resident.
    pub fn resident(&self) -> bool {
        let st = self.inner.state.lock();
        self.blocks.iter().all(|b| st.table.page((self.set, *b)).is_some_and(|p| p.mapped && p.filled && !p.filling))
    }

    /// Acquires all physical dependencies atomically against eviction. The returned pins hold
    /// them until completion on `stream`, including a cancelled caller's queued work. Callers
    /// join every additional consumer stream back into `stream` before releasing these pins.
    pub fn prepare(&self, stream: usize) -> Result<Vec<Pin>> { self.prepare_for(stream, None) }

    pub fn prepare_for(&self, stream: usize, request: Option<&Request>) -> Result<Vec<Pin>> {
        if request.is_some_and(|r| !Arc::ptr_eq(&self.inner, &r.inner)) {
            return Err(Error::Invalid("range and request use different devices".into()));
        }
        self.inner.guard(|| {
            let (preparing, req) = {
                let mut st = self.inner.lock();
                let m = st.mech(self.set)?;
                if retained_identity(self.set, m, &self.blocks) != self.identity {
                    return Err(Error::StaleBinding("retained range identity changed; recapture required".into()));
                }
                let req = request.map(|r| r.id).or_else(|| st.binding(self.set));
                let mut pins = Vec::with_capacity(self.blocks.len());
                for &block in &self.blocks {
                    let key = (self.set, block);
                    let id = st.tick();
                    st.pins.insert(id, (key, req, std::thread::current().id()));
                    st.sync(key);
                    pins.push((id, key));
                }
                (Preparing { inner: self.inner.clone().into(), pins }, req)
            };
            // Every missing page's room together, before any consumer launches.
            let (missing, keep): (u64, Vec<Key>) = {
                let st = self.inner.lock();
                let keys: Vec<Key> = self.blocks.iter().map(|b| (self.set, *b)).collect();
                (keys.iter().filter_map(|k| st.table.page(*k)).filter(|p| !p.mapped).map(|p| p.bytes).sum(), keys)
            };
            if missing > 0 {
                self.inner.make_room(missing, &keep, "prepare", req, false)?;
            }
            let claims: Vec<u64> = preparing.pins.iter().map(|(id, _)| *id).collect();
            let pins = self.blocks.iter().map(|b| self.inner.bind_for(self.set, *b, stream, req, &claims, false)).collect::<Result<Vec<_>>>()?;
            let st = self.inner.lock();
            if retained_identity(self.set, st.mech(self.set)?, &self.blocks) != self.identity {
                return Err(Error::StaleBinding("retained range identity changed during preparation".into()));
            }
            drop(st);
            drop(preparing);
            Ok(pins)
        })
    }
}

impl Drop for Retained {
    fn drop(&mut self) {
        let mut st = self.inner.lock();
        if let Some(Some(m)) = st.sets.get_mut(self.set as usize) {
            m.retained = m.retained.saturating_sub(1);
        }
        self.inner.close_sets(&mut st);
    }
}

/// Where a bound block's bytes came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Origin {
    /// Already mapped and filled.
    Hit,
    /// Adopted from the lease manager's retention: no copy.
    Kept,
    /// The store, through the page cache.
    Store,
}

/// One bind: the page-reference trace the gate computes Belady MIN from.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Ref {
    pub step: u32,
    pub set: String,
    pub block: u32,
    pub bytes: u64,
    pub origin: Origin,
    /// It was prefetched before this bind.
    pub speculative: bool,
}

/// Bytes filled from one source and the time the fills took, summed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct Fills {
    pub bytes: u64,
    pub ns: u64,
}

/// Bytes moved, by source: fill time over bytes against the measured ceiling gives the %.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct Moved {
    /// Leased pages adopted from the manager's retention (no copy; `ns` is the ask, import and map).
    pub kept: Fills,
    /// Pages leased from the manager and filled here (`ns`: the ask, import and map).
    pub leased: Fills,
    pub store: Fills,
    /// Of the above, bytes filled speculatively.
    pub speculative: u64,
}

/// One step of a request.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct StepStats {
    /// The free bytes at its start: the manager's figure, or the standalone budget's.
    pub capacity: u64,
    /// Bytes `make_room` took during the step, by class (F, O, S, N, D, W, P).
    pub evicted: [u64; 7],
}

/// One request's record (the design's §5 bar).
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct Stats {
    pub request: String,
    /// Every fill since the previous request ended (loads included) and during this one.
    pub moved: Moved,
    /// One entry per bind, after one per fill made before it opened (its loads: step 0, with
    /// the fill's origin).
    pub trace: Vec<Ref>,
    pub steps: Vec<StepStats>,
    /// Time binds waited on fills in flight (a demand miss, or a prefetch not yet landed).
    pub stall_ns: u64,
    /// Bytes `make_room` took for this request, by class (F, O, S, N, D, W, P).
    pub evicted: [u64; 7],
    /// The most it held at a pass end: pages and torch's bytes (the peak its passes reported,
    /// when more).
    pub peak: u64,
    /// The most its passes allocated beside its weights, as they reported it: its compute bytes.
    pub compute: u64,
    /// The process's anonymous host high-water over it (file mappings aside); `None` where it
    /// cannot be read.
    pub host: Option<u64>,
    /// Bytes the request still held when it ended (pins): 0.
    pub leftover: u64,
    /// Each `make_room` that had to take bytes for it, in order.
    pub make_room: Vec<MakeRoom>,
    pub outlived: Vec<crate::arena::Outlived>, // its torch blocks live at its end (the audit)
}

/// One `make_room` that had to take bytes: what asked, how many bytes, what it took.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct MakeRoom {
    /// The request's plan step then.
    pub step: u32,
    /// A bind or a range's preparation.
    pub trigger: &'static str,
    /// The free bytes asked for.
    pub need: u64,
    /// The bytes its rounds took (pages unmapped).
    pub freed: u64,
    /// Release events it waited on before taking their pages.
    pub waited: u32,
}

/// Public handles count explicitly; internal worker references are not public ownership.
/// The final public handle transfers its actual Arc, never a racing temporary clone.
struct Owner(std::mem::ManuallyDrop<Arc<Inner>>);

impl From<Arc<Inner>> for Owner {
    fn from(inner: Arc<Inner>) -> Self {
        inner.public_owners.fetch_add(1, Ordering::Relaxed);
        Self(std::mem::ManuallyDrop::new(inner))
    }
}
impl std::ops::Deref for Owner {
    type Target = Arc<Inner>;
    fn deref(&self) -> &Self::Target { &self.0 }
}
impl Drop for Owner {
    fn drop(&mut self) {
        // SAFETY: Owner always contains one initialized Arc; only this Drop moves it out,
        // and ManuallyDrop prevents a second release after the transfer.
        let inner = unsafe { std::mem::ManuallyDrop::take(&mut self.0) };
        if inner.public_owners.fetch_sub(1, Ordering::AcqRel) != 1 { return; }
        inner._device.retiring(inner.ownership_lost.load(Ordering::Acquire));
        // Its retirement waits on the work it needs done, on a thread of its own: no caller
        // (Python, a DLPack destructor, the allocator) waits for it.
        let (tx, rx) = std::sync::mpsc::sync_channel::<Arc<Inner>>(1);
        let spawned = std::thread::Builder::new().name("varena-retire".into()).spawn(move || {
            if let Ok(inner) = rx.recv() {
                inner.retire();
            }
        });
        let unsent = match spawned {
            Ok(_) => tx.send(inner).err().map(|e| e.0),
            Err(_) => Some(inner),
        };
        if let Some(inner) = unsent {
            inner._device.retiring(true);
            std::mem::forget(inner);
        }
    }
}

/// One process's memory on one device. A clone is another public handle on it.
pub struct Memory {
    inner: Owner,
    /// Memory handles (clones): the last one's drop closes it.
    handles: Arc<AtomicUsize>,
}

impl Clone for Memory {
    fn clone(&self) -> Memory {
        self.handles.fetch_add(1, Ordering::AcqRel);
        Memory { inner: Arc::clone(&*self.inner).into(), handles: self.handles.clone() }
    }
}

/// region.rs's hooks: the device's events, its free bytes, the manager's figures, and the plan.
impl Memory {
    pub(crate) fn record_event(&self, stream: usize) -> Result<Event> {
        self.inner.backend.record(stream)
    }

    pub(crate) fn finish_event(&self, event: Event) -> Result<()> {
        self.inner.backend.finish(event)
    }

    pub(crate) fn retire_event(&self, event: Event) {
        self.inner.backend.retire(event)
    }

    pub(crate) fn synchronize(&self, stream: usize) -> Result<()> {
        self.inner.backend.synchronize(stream)
    }

    pub(crate) fn granularity(&self) -> u64 {
        self.inner.backend.granularity()
    }

    /// Free bytes nothing else needs reclaimed for ([`Inner::spare`]).
    pub(crate) fn spare(&self) -> u64 {
        let st = self.inner.state.lock();
        self.inner.spare(&st)
    }

    /// The device's torch allocator, if anything allocated through it yet.
    pub(crate) fn torch_if(&self) -> Option<Arc<crate::arena::Torch>> {
        self.inner.torch.get().cloned()
    }

    /// The same device memory: two handles of one open.
    pub(crate) fn same(&self, other: &Memory) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// The manager's figures for this executor, while its pages are leased.
    pub(crate) fn figures(&self) -> Option<crate::lease::Figures> {
        #[cfg(unix)]
        return self.inner.lease.get().map(|a| a.table().figures());
        #[cfg(not(unix))]
        None
    }

    /// Plan step `step`'s sets (each open request's current step when `None`): this process's
    /// mapped bytes in them, its idle mapped bytes outside them, and their largest block.
    pub(crate) fn step_bytes(&self, step: Option<u32>) -> (u64, u64, u64) {
        let st = self.inner.state.lock();
        let sets: BTreeSet<u32> = st.table.requests.iter().filter_map(|r| r.steps.get(step.unwrap_or(r.cursor.step) as usize)).flat_map(|(s, _)| s.iter().copied()).collect();
        let (mut warm, mut cold, mut largest) = (0, 0, 0);
        for ((set, _), p) in st.table.pages() {
            if sets.contains(&set) {
                largest = largest.max(p.bytes);
                warm += if p.mapped { p.bytes } else { 0 };
            } else if p.mapped && p.pins.is_empty() && p.releasing.is_empty() && !p.filling {
                cold += p.bytes;
            }
        }
        (warm, cold, largest)
    }

    /// A region's facts to the lease manager (`None`: it is gone), while leased.
    pub(crate) fn region_facts(&self, id: u64, record: Option<&crate::region::Record>) {
        #[cfg(unix)]
        if let Some(agent) = self.inner.lease.get() {
            let sent = match record {
                Some(r) => agent.region(crate::lease::wire::RegionFacts { id, kind: r.kind, slots: r.slots.clone(), cost_ns: r.cost_ns, last_use: r.last_use, next_use: r.next_use.map_or(u64::MAX, u64::from) }),
                None => agent.region_gone(id),
            };
            if let Err(e) = sent {
                tracing::debug!("varena: region {id}'s facts did not reach the lease manager: {e}");
            }
        }
        #[cfg(not(unix))]
        let _ = (id, record);
    }

    /// The lease slots of `sets`' leased pages.
    pub(crate) fn lease_slots(&self, sets: &[SetId]) -> Vec<u32> {
        #[cfg(unix)]
        return self.inner.leased.lock().values().filter(|(_, key, _)| sets.iter().any(|s| s.0 == key.0)).map(|(t, ..)| t.slot).collect();
        #[cfg(not(unix))]
        {
            let _ = sets;
            Vec::new()
        }
    }

    /// Every block of `set`.
    pub(crate) fn blocks(&self, set: SetId) -> Result<Vec<u32>> {
        let st = self.inner.state.lock();
        let n = st.table.sets.get(set.0 as usize).and_then(|s| s.as_ref()).map(|s| s.pages.len()).ok_or_else(|| Error::Closed(format!("set {}", set.0)))?;
        Ok((0..n as u32).collect())
    }
}

/// A request in flight; dropping it ends it: its pins go, and the leftover is measured.
pub struct Request {
    inner: Owner,
    id: u64,
    ended: bool,
}

/// A block bound for use: mapped, and filled once `ready` completes (the binding stream already
/// waits on it). Dropping it releases the block after the binding stream's work so far.
pub struct Pin {
    pub ptr: u64,
    pub len: u64,
    pub ready: Event,
    /// Run the transform now: the page's bytes are not the weights yet (a declared or
    /// undeclared fill's store bytes); [`Pin::filled`] after it. Never for bytes that are the
    /// weights (stored, kept, or transformed).
    pub fresh: bool,
    inner: Owner,
    key: Key,
    id: u64,
    stream: usize,
    /// Fresh bytes of a set with a transform: refilled on release unless [`Pin::filled`].
    transform: bool,
    transformed: AtomicBool,
}

/// A pass of a set: the plan's cursor moves to it. Dropping it without [`Pass::end`] records
/// no peak.
pub struct Pass {
    inner: Owner,
    req: u64,
    set: u32,
    id: u64,
}

struct SetMech {
    spec: SetSpec,
    /// Its fingerprint's key at its current fill (derived once: a hash).
    key: String,
    va: Va,
    closing: bool,
    /// Graph/address references, independent of physical residency.
    retained: u64,
}

fn retained_identity(set: u32, m: &SetMech, blocks: &[u32]) -> String {
    format!("{set}:{}:{}:{}:{blocks:?}", m.va.base, m.va.len, m.key)
}

struct Home {
    /// `None` while the mechanism thread creates and maps it.
    phys: Option<Physical>,
    ready: Event,
    /// The stream `ready` is on ([`COPY`] for a fill's).
    ready_on: usize,
    /// Tells a fill whether its page is still the one it was queued for.
    gen: u64,
    filled: bool,
    filling: bool,
    /// Read ready at issue (`issued`), its batch not ended yet: the fill thread still accounts
    /// for it and its copies may still land.
    landing: bool,
    /// A binder holds it fresh ([`Home::fresh`]) and runs the set's transform.
    transforming: bool,
    origin: Origin,
    speculative: bool,
    /// The fill identity its bytes were made under ([`crate::Fingerprint::key`]); empty: its
    /// bytes are not the weights (a transform released undone), so it goes once unpinned.
    identity: String,
}

impl Home {
    /// A page the mechanism maps now, filling for `identity`.
    fn new(gen: u64, speculative: bool, identity: String) -> Home {
        Home { phys: None, ready: Event::DONE, ready_on: COPY, gen, filled: false, filling: true, landing: false, transforming: false, origin: Origin::Store, speculative, identity }
    }
}

impl Home {
    /// Its bytes landed and are not the weights yet: its binder runs the transform.
    fn fresh(&self) -> bool {
        !self.filled && !self.filling && !self.transforming
    }
}

/// Wakes for state waiters: an epoch under a leaf mutex. A producer bumps it after its change
/// (with the state lock or without: the allocator callback, a pin's drop); a waiter sleeps only
/// while the epoch is the one it read before its last check, so no wake is lost between them.
#[derive(Default)]
struct Wakes {
    epoch: Mutex<u64>,
    cv: Condvar,
}

impl Wakes {
    fn notify_all(&self) {
        *self.epoch.lock() += 1;
        self.cv.notify_all();
    }

    fn seen(&self) -> u64 {
        *self.epoch.lock()
    }

    fn wait_past(&self, seen: u64) {
        let mut epoch = self.epoch.lock();
        while *epoch == seen {
            self.cv.wait(&mut epoch);
        }
    }
}

/// A call to the device's virtual-memory mechanism (`cuMemCreate`, `cuMemMap`, `cuMemUnmap`):
/// each may wait on the host for queued device work, so none runs under the state lock. One
/// thread runs them in the order decided (an eviction's unmap before the create its bytes go
/// to); each result re-enters the state.
enum Mech {
    /// Create (or lease) a page, map it read-write, then queue its fill; `decided`: the bind's
    /// (none: a speculative page, which goes quietly when refused).
    Map { key: Key, gen: u64, va: Va, off: u64, span: u64, job: Job, decided: bool },
    /// Unmap a page (the driver lets its consumers finish) and give it back.
    Unmap { va: Va, off: u64, span: u64, phys: Option<Physical>, ready: Event },
    /// A leased page whose fill identity changed (a grant's identity never changes): unmapped
    /// after its consumers and given back, and a new page filled in its place.
    Renew { key: Key, gen: u64, va: Va, off: u64, span: u64, old: Option<Physical>, after: Vec<Event>, job: Job },
    /// A streamed set's slot moves from the block at `from` to `key`'s block, after the old block's consumers
    /// (`after`); then `key`'s fill is queued.
    Move { key: Key, gen: u64, va: Va, from: u64, off: u64, span: u64, phys: Physical, after: Vec<Event>, job: Job },
    /// A closed set's reservation, after its pages' unmaps.
    Release(Va),
    /// The lease manager asked pages back: the quiet ones go, the busy ones once idle.
    #[cfg(unix)]
    Revoke,
    /// The lease manager reclaims a region whole (crate::region).
    #[cfg(unix)]
    Reclaim { region: u64, swap: bool },
    /// The Memory went: its thread ends.
    Close,
}

struct Active {
    id: u64,
    thread: std::thread::ThreadId,
}

struct Job {
    key: Key,
    gen: u64,
    req: Option<u64>,
    /// The page's bytes: a batch is cut by them.
    span: u64,
    /// The page's earlier consumers (release events) a refill in place writes after: the copy
    /// stream waits on them in-stream, and they are retired once the fill's event follows them.
    after: Vec<Event>,
}

/// Fill jobs, in groups: the pages the mechanism mapped together (a bind's block) reach a fill
/// thread as one, so they read as one run. A demand miss goes before any prefetch.
struct Jobs {
    demand: Sender<Vec<Job>>,
    ahead: Sender<Vec<Job>>,
}

/// The backend's copy stream, where fills' ready events are ([`Backend::fence`]); no caller's
/// stream is this.
const COPY: usize = usize::MAX;

/// One page of a fill batch: where it goes, and how its copy ended.
struct Copying {
    key: Key,
    gen: u64,
    dst: Dst,
    out: Option<Result<Origin>>,
}

/// The fill threads' work: jobs queued, batches running, the bytes filled since they were last
/// all idle, and whether the Memory is gone. Its lock is a leaf; its condvar wakes the
/// copies when the fills are idle.
#[derive(Default)]
struct Burst {
    queued: u64,
    batches: u32,
    bytes: u64,
    closed: bool,
}

impl Burst {
    fn idle(&self) -> bool {
        self.queued == 0 && self.batches == 0
    }
}

struct Release {
    key: Key,
    pin: u64,
    event: Event,
    /// The stream it was recorded on.
    stream: usize,
    /// Released fresh without its transform: its bytes are not the weights.
    untransformed: bool,
}

#[derive(Default)]
struct State {
    /// The wake epoch read when this lock was taken, before its drain: a wait sleeps only while
    /// no change came after it.
    seen: u64,
    /// Each page's last bind on the machine-wide clock (CLOCK_MONOTONIC ns): the manager
    /// compares idle pages across processes by it.
    #[cfg(unix)]
    used_ns: HashMap<Key, u64>,
    table: Table,
    sets: Vec<Option<SetMech>>,
    homes: HashMap<Key, Home>,
    /// Pin → (page, request, the thread that holds it).
    pins: HashMap<u64, (Key, Option<u64>, std::thread::ThreadId)>,
    /// Each stream's pending release events, in the order recorded: one stream's events complete
    /// in order, so a drain queries only each stream's oldest.
    events: BTreeMap<usize, VecDeque<(Event, Key)>>,
    /// Sets streaming their upper blocks through slots (step 2's watermark), by set.
    streams: HashMap<u32, Streaming>,
    /// Ids and the logical clock.
    clock: u64,
    /// Pages whose create the driver refused, with the driver's free bytes measured at the
    /// refusal: the truth over the table's figure for the bind's next round.
    short: HashMap<Key, u64>,
    stats: HashMap<u64, Stats>,
    /// Fills no open request made (loads): the next request's record opens with them.
    unclaimed: Stats,
    last: Option<Stats>,
    /// Passes running, per (request, set): a joint step's sets each have their own.
    active: HashMap<(u64, u32), Active>,
    failed: HashMap<Key, Error>,
    /// The latest unmap's mechanism ticket: whoever counts on its bytes settles up to it.
    last_unmap: Option<u64>,
    /// Requests whose last step is one their plan did not declare.
    synthetic: HashSet<u64>,
    /// This process's anonymous host peak while requests run.
    window: Option<HostWindow>,
}

/// This process's anonymous host high-water since a start (file mappings aside): a request's
/// host peak, for the manager's admission.
struct HostWindow {
    peak: u64,
    file: u64,
}

impl HostWindow {
    /// Measured from now; `None` where it cannot be read.
    fn start() -> Option<HostWindow> {
        let now = host_memory::resident(std::process::id()).ok()?;
        HostWindow::restart();
        Some(HostWindow { peak: now.anon, file: now.file })
    }

    /// The anonymous peak so far; `None` where it cannot be read.
    fn look(&mut self) -> Option<u64> {
        let now = host_memory::resident(std::process::id()).ok()?;
        let bound = now.peak.saturating_sub(self.file.min(now.file) + now.shmem);
        self.peak = self.peak.max(bound).max(now.anon);
        self.file = now.file;
        HostWindow::restart();
        Some(self.peak)
    }

    /// The process's resident high-water starts over (where it may not, the next look bounds
    /// from further back).
    fn restart() {
        if let Err(e) = host_memory::reset_peak(std::process::id()) {
            tracing::debug!("varena: host high-water not reset: {e}");
        }
    }
}

impl State {
    /// No work can touch the page: unpinned, no release event pending, not filling or being
    /// transformed.
    fn idle(&self, key: Key) -> bool {
        self.table.page(key).is_some_and(|p| p.pins.is_empty() && p.releasing.is_empty())
            && self.homes.get(&key).is_some_and(|h| !h.filling && !h.transforming)
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn mech(&self, set: u32) -> Result<&SetMech> {
        self.sets.get(set as usize).and_then(Option::as_ref).ok_or_else(|| Error::Invalid(format!("no set {set}")))
    }

    /// Where a page lives: its set's VA, its offset and span.
    fn place(&self, (set, block): Key) -> Result<(Va, u64, u64)> {
        let m = self.mech(set)?;
        let r = m.spec.layout.region(block)?;
        Ok((m.va, r.offset, r.span))
    }

    /// The table's page facts from the mechanism's.
    fn sync(&mut self, key: Key) {
        let home = self.homes.get(&key).map(|h| (h.filled, h.filling, h.speculative));
        let pins: Vec<u64> = self.pins.iter().filter(|(_, (k, ..))| *k == key).map(|(id, _)| *id).collect();
        if let Some(p) = self.table.page_mut(key) {
            p.mapped = home.is_some();
            p.filled = home.is_some_and(|h| h.0);
            p.filling = home.is_some_and(|h| h.1);
            p.pins = pins;
            if home.is_none_or(|h| !h.2) {
                p.speculative = None;
            }
        }
    }

    fn stats(&mut self, req: Option<u64>) -> Option<&mut Stats> {
        self.stats.get_mut(&req?)
    }

    /// Where a fill is recorded: its request's record, else (a load's, or one with no request
    /// open) the next request's, which opens with them.
    fn fills(&mut self, req: Option<u64>, key: Key, bytes: u64, origin: Origin) -> &mut Stats {
        match req.filter(|r| self.stats.contains_key(r)) {
            Some(r) => self.stats.entry(r).or_default(),
            None => {
                let set = self.sets.get(key.0 as usize).and_then(Option::as_ref).map(|m| m.spec.name.clone()).unwrap_or_default();
                self.unclaimed.trace.push(Ref { step: 0, set, block: key.1, bytes, origin, speculative: false });
                &mut self.unclaimed
            }
        }
    }

    fn step_stats(&mut self, req: u64) -> Option<&mut StepStats> {
        let step = self.table.request(req)?.cursor.step as usize;
        let s = self.stats.get_mut(&req)?;
        if s.steps.len() <= step {
            s.steps.resize(step + 1, StepStats::default());
        }
        s.steps.get_mut(step)
    }

    /// The request whose current pass runs `set`.
    fn binding(&self, set: u32) -> Option<u64> {
        self.active.iter().filter(|((_, s), a)| *s == set && a.thread == std::thread::current().id()).map(|((r, _), _)| *r).max()
    }
}

/// Signalled each time a worker thread lets go of its reference to the Memory: a retirement
/// waits on it for the last to go.
#[derive(Default)]
struct Quiet {
    lock: Mutex<()>,
    gone: Condvar,
}

/// A worker thread's reference to the Memory for one item; letting go of it wakes a retirement.
struct Work {
    inner: std::mem::ManuallyDrop<Arc<Inner>>,
    quiet: Arc<Quiet>,
}

impl Work {
    fn of(inner: &Weak<Inner>) -> Option<Work> {
        let inner = inner.upgrade()?;
        let quiet = inner.quiet.clone();
        Some(Work { inner: std::mem::ManuallyDrop::new(inner), quiet })
    }
}

impl std::ops::Deref for Work {
    type Target = Arc<Inner>;
    fn deref(&self) -> &Arc<Inner> {
        &self.inner
    }
}

impl Drop for Work {
    fn drop(&mut self) {
        // SAFETY: taken once, here; ManuallyDrop never drops it again.
        drop(unsafe { std::mem::ManuallyDrop::take(&mut self.inner) });
        let _quiet = self.quiet.lock.lock();
        self.quiet.gone.notify_all();
    }
}

struct Inner {
    backend: Arc<dyn Backend>,
    /// Standalone, every page's and torch chunk's: the device's free bytes at open less its
    /// reserve.
    allowance: Arc<Allowance>,
    /// The lease agent once attached (varena-leases): this GPU's pages are the manager's chunks.
    #[cfg(unix)]
    lease: std::sync::OnceLock<Arc<crate::lease::Agent>>,
    /// Leased pages by physical id: their grant, their key, and whether their bytes are still
    /// the region they were granted for (a moved slot's never are again).
    #[cfg(unix)]
    leased: Mutex<HashMap<u64, (crate::lease::Ticket, Key, bool)>>,
    /// Asked back while in use: pinned in the table until idle, then given.
    #[cfg(unix)]
    declined: Mutex<HashSet<crate::lease::Ticket>>,
    /// Leased pages filled whose bytes the manager has not heard landed yet: acked once their
    /// ready event completes (only landed bytes are retained).
    #[cfg(unix)]
    to_land: Mutex<Vec<Key>>,
    state: Mutex<State>,
    /// Every change a state waiter may test, from any thread (a pin's drop holds no state lock):
    /// an epoch the waiter read before its last check.
    landed: Wakes,
    /// The device's mechanism calls, for its one thread.
    mech: Arc<Mechs>,
    release_tx: Sender<Release>,
    releases: Receiver<Release>,
    /// Fill jobs; the fill threads hold the receiving ends and a weak reference to the rest,
    /// so dropping the Memory ends them.
    jobs: Jobs,
    /// Pages the mechanism thread mapped and has not handed to the fill threads yet (demand,
    /// ahead): it hands them over together once its queue is empty or they make a batch.
    grouped: Mutex<(Vec<Job>, Vec<Job>)>,
    burst: Arc<(Mutex<Burst>, Condvar)>,
    quiet: Arc<Quiet>,
    ownership_lost: AtomicBool,
    public_owners: AtomicU64,
    #[cfg(unix)]
    tally: tensorfs::io::source::IoTally,
    torch: std::sync::OnceLock<Arc<crate::arena::Torch>>,
    _device: Device,
}

/// The mechanism's queue: its calls in the order decided, each with a ticket, so a caller waits
/// for the calls up to one ([`Mechs::settle_to`]).
struct Mechs {
    tx: Sender<(u64, Mech)>,
    rx: Receiver<(u64, Mech)>,
    /// Tickets are taken and sent under it: the channel's order is the tickets'.
    sending: Mutex<()>,
    posted: AtomicU64,
    done: AtomicU64,
    wakes: Wakes,
}

impl Mechs {
    fn new() -> Mechs {
        let (tx, rx) = unbounded();
        Mechs { tx, rx, sending: Mutex::new(()), posted: AtomicU64::new(0), done: AtomicU64::new(0), wakes: Wakes::default() }
    }

    /// Queues `op`; its ticket.
    fn post(&self, op: Mech) -> u64 {
        let _one = self.sending.lock();
        let ticket = self.posted.fetch_add(1, Ordering::AcqRel) + 1;
        if self.tx.send((ticket, op)).is_err() {
            self.finished(ticket);
        }
        ticket
    }

    fn finished(&self, ticket: u64) {
        self.done.fetch_max(ticket, Ordering::AcqRel);
        self.wakes.notify_all();
    }

    /// Blocks until the calls up to `ticket` ran.
    fn settle_to(&self, ticket: u64) {
        loop {
            let seen = self.wakes.seen();
            if self.done.load(Ordering::Acquire) >= ticket {
                return;
            }
            self.wakes.wait_past(seen);
        }
    }

    /// Blocks until every call posted so far ran.
    fn settle(&self) {
        self.settle_to(self.posted.load(Ordering::Acquire));
    }

    fn idle(&self) -> bool {
        self.done.load(Ordering::Acquire) >= self.posted.load(Ordering::Acquire)
    }
}

fn exhausted(need: u64, site: Option<&str>, stats: Stats) -> Box<Exhausted> {
    Box::new(Exhausted { need, site: site.map(str::to_string), stats, foreign: None })
}

impl Memory {
    /// Opens `gpu` for this process: one Memory per device per process. Its pages are leased
    /// once [`Memory::lease`] attaches it to a manager; until then (or with none) they come from
    /// the device within the standalone budget: its free bytes now less the reserve (rev 6).
    pub fn open(gpu: GpuId, backend: Box<dyn Backend>) -> Result<Memory> {
        crate::caught(|| {
            if !crate::layout::REGION_ALIGN.is_multiple_of(backend.granularity().max(1)) {
                return Err(Error::Invalid(format!("granularity {} does not divide a region", backend.granularity())));
            }
            let device = Device::open(&gpu.uuid)?;
            let (release_tx, releases) = unbounded();
            let ((demand, demanded), (ahead, wanted_ahead)) = (unbounded(), unbounded());
            let allowance = Arc::new(Allowance::of(&*backend)?);
            let inner = Arc::new(Inner {
                allowance,
                backend: Arc::from(backend),
                #[cfg(unix)]
                lease: std::sync::OnceLock::new(),
                #[cfg(unix)]
                leased: Mutex::default(),
                #[cfg(unix)]
                declined: Mutex::default(),
                #[cfg(unix)]
                to_land: Mutex::default(),
                _device: device,
                state: Mutex::default(),
                landed: Wakes::default(),
                mech: Arc::new(Mechs::new()),
                release_tx,
                releases,
                jobs: Jobs { demand, ahead },
                grouped: Mutex::default(),
                burst: Arc::new((Mutex::new(Burst::default()), Condvar::new())),
                quiet: Arc::default(),
                ownership_lost: AtomicBool::new(false),
                public_owners: AtomicU64::new(0),
                #[cfg(unix)]
                tally: Default::default(),
                torch: Default::default(),
            });
            let (weak, queue) = (Arc::downgrade(&inner), inner.mech.clone());
            std::thread::Builder::new().name("varena-mechanism".into()).spawn(move || mech_loop(&queue, &weak)).map_err(|e| Error::io("spawn mechanism", e))?;
            // One fill thread per copy stream TensorFS runs per device: a fill waits on its
            // streams, so this many keep them busy.
            for _ in 0..tensorfs::io::ReadOpts::default().streams.max(1) {
                let (weak, demand, ahead) = (Arc::downgrade(&inner), demanded.clone(), wanted_ahead.clone());
                std::thread::Builder::new().name("varena-fill".into()).spawn(move || fill_loop(&demand, &ahead, &weak)).map_err(|e| Error::io("spawn fill", e))?;
            }
            Ok(Memory { inner: inner.into(), handles: Arc::new(AtomicUsize::new(1)) })
        })
    }

    /// Leases this GPU's pages from the manager serving it (TensorD's, in this user's
    /// namespace), right after open, before any load: from then on its pages are the manager's
    /// chunks, retained by it across this process's exit, and asked back by it. A refusal comes
    /// back typed, `Some((code, why))`: Incompatible or Absent runs standalone; the caller fails
    /// on any other (TensorD parks a peer it has not registered yet, so the dial itself waits).
    #[cfg(unix)]
    pub fn lease(&self) -> Result<Option<(crate::lease::Refusal, String)>> {
        let uuid = self.inner._device.0.clone();
        match crate::lease::connect(crate::os::euid(), &uuid)? {
            Some(stream) => self.lease_over(stream),
            None => Ok(Some((crate::lease::Refusal::Absent, format!("no lease manager serves GPU {uuid}")))),
        }
    }

    /// Whether its pages are leased ([`Memory::lease`]).
    pub fn leased(&self) -> bool {
        self.inner.leasing()
    }

    /// [`Memory::lease`] over a stream its host connected.
    #[cfg(unix)]
    pub fn lease_over(&self, stream: crate::os::Stream) -> Result<Option<(crate::lease::Refusal, String)>> {
        let mech = self.inner.mech.clone();
        let revoked = move || {
            mech.post(Mech::Revoke);
        };
        match crate::lease::Agent::attach(stream, self.inner.backend.clone(), revoked)? {
            crate::lease::Attach::Leased(agent) => {
                let mech = self.inner.mech.clone();
                agent.on_reclaim(Box::new(move |region, swap| {
                    mech.post(Mech::Reclaim { region, swap });
                }))?;
                if self.inner.lease.set(agent).is_err() {
                    return Err(Error::Invalid("this Memory is leased already".into()));
                }
                // torch's blocks come from leased chunks too: the agent is its only allocator.
                self.serve_leased_torch()?;
                Ok(None)
            }
            crate::lease::Attach::Refused { code, why } => Ok(Some((code, why))),
        }
    }

    /// Registers a set and brings it to `level`; a set loaded again is re-levelled in place.
    /// Leased, a page the manager retains comes back with no copy; the rest fill from the store,
    /// through the page cache.
    pub fn load(&self, set: SetSpec, level: Level) -> Result<SetId> {
        let (name, bytes) = (set.name.clone(), set.layout.nbytes);
        let id = self.inner.guard(|| self.inner.load(set, level))?;
        if level == Level::Gpu {
            self.inner.measured(crate::lease::Measured::Set { name, bytes });
        }
        Ok(id)
    }

    /// Unmaps the set's pages (each as its pin releases) and forgets it.
    pub fn unload(&self, set: SetId) {
        let _ = self.inner.guard(|| {
            let mut st = self.inner.lock();
            if let Some(Some(m)) = st.sets.get_mut(set.0 as usize) {
                m.closing = true;
            }
            self.inner.close_sets(&mut st);
            self.inner.unlock(st);
            Ok(())
        });
    }

    /// Sets the set's current fill identity (a no-op when unchanged). Pages filled under another
    /// one are unmapped (a pinned one when it is released), so each next bind fills again, from
    /// kept pages of the new identity or the store, and comes back `fresh`.
    pub fn fill(&self, set: SetId, fill: Fill) {
        let _ = self.inner.guard(|| {
            let mut st = self.inner.lock();
            let r = self.inner.refill(&mut st, set.0, fill);
            self.inner.unlock(st);
            r
        });
    }

    /// Opens a request with its plan; Belady runs over the open requests' plans.
    pub fn request(&self, id: &str, plan: Plan) -> Result<Request> {
        let rid = self.inner.guard(|| self.inner.open_request(id, &plan, false))?;
        Ok(Request { inner: self.inner.clone().into(), id: rid, ended: false })
    }

    /// Opens a request for idle work (a graph capture off the request path): its binds, its
    /// ranges' preparation and its allocator hook's room take free bytes only, `Exhausted` at
    /// once when they lack them; it asks no peer and withdraws no kept page, so it never holds
    /// up another request.
    pub fn request_idle(&self, id: &str, plan: Plan) -> Result<Request> {
        let rid = self.inner.guard(|| self.inner.open_request(id, &plan, true))?;
        Ok(Request { inner: self.inner.clone().into(), id: rid, ended: false })
    }

    /// Maps and fills block `block` of `set`; `stream` (a `CUstream`, 0 the default) waits on
    /// the fill in-stream, and the release is ordered after its work.
    pub fn bind(&self, set: SetId, block: u32, stream: usize) -> Result<Pin> {
        self.inner.guard(|| self.inner.bind_for(set.0, block, stream, None, &[], false))
    }

    /// [`Memory::bind`] into free bytes only: `Exhausted` at once when they lack the block's,
    /// nothing taken.
    pub fn bind_free(&self, set: SetId, block: u32, stream: usize) -> Result<Pin> {
        self.inner.guard(|| self.inner.bind_for(set.0, block, stream, None, &[], true))
    }

    /// Retains stable virtual addresses for exactly these blocks, without making them resident.
    pub fn retain(&self, set: SetId, blocks: &[u32]) -> Result<Retained> {
        self.inner.guard(|| {
            let mut st = self.inner.lock();
            let blocks: Vec<u32> = blocks.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();
            let m = st.mech(set.0)?;
            if m.closing {
                return Err(Error::Closed(format!("set {}", set.0)));
            }
            for &b in &blocks { m.spec.layout.region(b)?; }
            let identity = retained_identity(set.0, m, &blocks);
            if let Some(Some(m)) = st.sets.get_mut(set.0 as usize) { m.retained += 1; }
            Ok(Retained { inner: self.inner.clone().into(), set: set.0, blocks, identity })
        })
    }

    /// Whether driver-owned context growth has a measurable per-process owner. A graph
    /// capacity proof must report this; the opening-context fallback alone cannot prove it.
    pub fn context_measurable(&self) -> bool { self.inner.backend.process_used().is_ok() }

    /// Blocks until every mechanism call decided so far was made (maps, unmaps, give-backs).
    pub fn settle(&self) {
        self.inner.mech.settle();
    }

    /// The record of the request that ended last.
    pub fn stats(&self) -> Option<Stats> {
        let st = self.inner.state.lock();
        st.last.clone()
    }

    /// This process's mapped page bytes by class ([`Class::index`]).
    pub fn classes(&self) -> [u64; 7] {
        let st = self.inner.lock();
        let mut out = [0u64; 7];
        for (key, page) in st.table.pages() {
            if let Some(slot) = st.table.class(key).and_then(|c| out.get_mut(c.index())) {
                *slot += page.bytes;
            }
        }
        out
    }
}

impl Inner {
    /// `f`, a panic in it caught: the call fails `Internal` and the table is derived again.
    fn guard<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        self.check_ownership()?;
        let out = crate::caught(f);
        if out.as_ref().err().is_some_and(Error::is_sticky) { self.ownership_lost.store(true, Ordering::Release); }
        if matches!(&out, Err(Error::Internal(m)) if m.starts_with("panic")) {
            self.recover();
        }
        out
    }

    fn check_ownership(&self) -> Result<()> {
        if self.ownership_lost.load(Ordering::Acquire) {
            Err(Error::Ownership("consumer completion failed; retire this executor".into()))
        } else { Ok(()) }
    }

    /// After a panic: the table's pages derived again from the homes (each recorded in the same
    /// statement as its mapping, so they are the mappings).
    fn recover(&self) {
        let mut st = self.state.lock();
        let keys: Vec<Key> = st.table.pages().map(|(k, _)| k).collect();
        for key in keys {
            st.sync(key);
        }
        self.unlock(st);
        self.landed.notify_all();
    }

    /// The state, with pin releases applied and completed events acted on.
    fn lock(&self) -> MutexGuard<'_, State> {
        let mut st = self.state.lock();
        st.seen = self.landed.seen();
        self.drain(&mut st);
        st
    }

    fn unlock(&self, st: MutexGuard<'_, State>) {
        drop(st);
    }

    /// Waits, the state unlocked, for a change after the caller's last check (a fill, a release,
    /// a reclaim), then reconciles and drains.
    fn wait(&self, st: &mut MutexGuard<'_, State>) -> Result<()> {
        self.check_ownership()?;
        let seen = st.seen;
        MutexGuard::unlocked(st, || self.landed.wait_past(seen));
        st.seen = self.landed.seen();
        self.check_ownership()?;
        self.drain(st);
        Ok(())
    }

    fn poll_releases(&self, st: &mut State) -> bool {
        let released: Vec<Release> = self.releases.try_iter().collect();
        let mut wake = false;
        for r in released {
            st.pins.remove(&r.pin);
            if r.event != Event::DONE {
                if let Some(p) = st.table.page_mut(r.key) {
                    p.releasing.push(r.event.0);
                }
                st.events.entry(r.stream).or_default().push_back((r.event, r.key));
            }
            if r.untransformed {
                // Its bytes are not the weights: it goes once unpinned and its release event
                // completed, and the next bind fills again.
                if let Some(h) = st.homes.get_mut(&r.key) {
                    h.transforming = false;
                    h.identity.clear();
                }
                wake = true;
            }
            st.sync(r.key);
        }
        // Completed events, oldest first on each stream: the first not yet complete ends that
        // stream's turn, so a drain costs a query per stream, not per pending event.
        let streams: Vec<usize> = st.events.keys().copied().collect();
        for stream in streams {
            while let Some((event, key)) = st.events.get(&stream).and_then(|q| q.front()).copied() {
                if !self.backend.query(event).unwrap_or(false) {
                    break;
                }
                if let Some(q) = st.events.get_mut(&stream) {
                    q.pop_front();
                }
                self.completed(st, event, key);
            }
        }
        st.events.retain(|_, q| !q.is_empty());
        wake
    }

    /// `key`'s pending `event` completed: it leaves the page's releases (retired unless it is
    /// the page's ready event).
    fn completed(&self, st: &mut State, event: Event, key: Key) {
        let ready = st.homes.get(&key).map(|h| h.ready);
        if let Some(i) = st.table.page(key).and_then(|p| p.releasing.iter().position(|e| *e == event.0)) {
            if let Some(p) = st.table.page_mut(key) {
                p.releasing.swap_remove(i);
            }
            if Some(event) != ready {
                self.backend.retire(event);
            }
        }
    }

    /// Waits for every pending event, ready events included (a retirement; host waits), and
    /// completes them as a drain would. Their pages are pinned meanwhile, so nothing retires the
    /// events or evicts the pages.
    fn finish_pending(&self) -> Result<()> {
        let (pins, taken, ready) = {
            let mut st = self.lock();
            let taken: Vec<(usize, Event, Key)> = std::mem::take(&mut st.events).into_iter().flat_map(|(stream, q)| q.into_iter().map(move |(e, k)| (stream, e, k))).collect();
            let ready: Vec<(Key, Event)> = st.homes.iter().filter(|(_, h)| !self.backend.query(h.ready).unwrap_or(false)).map(|(k, h)| (*k, h.ready)).collect();
            let me = std::thread::current().id();
            let keys: BTreeSet<Key> = taken.iter().map(|(.., k)| *k).chain(ready.iter().map(|(k, _)| *k)).collect();
            let pins: Vec<u64> = keys
                .into_iter()
                .map(|key| {
                    let pin = st.tick();
                    st.pins.insert(pin, (key, None, me));
                    st.sync(key);
                    pin
                })
                .collect();
            self.unlock(st);
            (pins, taken, ready)
        };
        // A page's own ready event stays its home's: waited on, never retired here.
        let ready: Result<()> = ready.iter().try_for_each(|(_, event)| self.backend.finish(*event));
        let done: Vec<bool> = taken.iter().map(|(_, event, _)| self.backend.finish(*event).is_ok()).collect();
        let mut st = self.lock();
        for pin in pins {
            if let Some((key, ..)) = st.pins.remove(&pin) {
                st.sync(key);
            }
        }
        let unproven = done.iter().filter(|d| !**d).count();
        for ((stream, event, key), done) in taken.into_iter().zip(done) {
            match done {
                true => self.completed(&mut st, event, key),
                // Unproven: still pending, for the drain.
                false => st.events.entry(stream).or_default().push_back((event, key)),
            }
        }
        self.drain(&mut st);
        self.landed.notify_all();
        self.unlock(st);
        match unproven {
            0 => ready,
            n => Err(Error::Ownership(format!("{n} pending events could not be waited on"))),
        }
    }

    fn drain(&self, st: &mut State) {
        let wake = self.poll_releases(st);
        #[cfg(unix)]
        if let Some(agent) = self.lease.get() {
            if wake {
                agent.progressed();
            }
            // Events still pending: its GPU work is running, never a stall.
            agent.busy(st.events.values().any(|q| !q.is_empty()));
            self.redecline(st);
            self.land(st, agent);
            self.advise(st, agent);
        }
        self.drop_stale(st);
        self.close_sets(st);
        if wake {
            self.landed.notify_all();
        }
    }

    /// Unmaps unpinned pages whose bytes are not their set's weights now (another fill identity,
    /// or a transform released undone), once no release event is pending.
    fn drop_stale(&self, st: &mut State) {
        let stale: Vec<Key> = st
            .homes
            .iter()
            .filter(|(k, h)| !h.filling && st.mech(k.0).is_ok_and(|m| m.key != h.identity))
            .filter(|(k, _)| st.table.page(**k).is_some_and(|p| p.pins.is_empty() && p.releasing.is_empty()))
            .map(|(k, _)| *k)
            .collect();
        for key in stale {
            let _ = self.unmap(st, key, Class::N, None);
        }
    }

    /// The set's fill identity becomes `fill`; pages of another identity go as they unpin.
    fn refill(&self, st: &mut State, set: u32, fill: Fill) -> Result<()> {
        let m = st.sets.get_mut(set as usize).and_then(Option::as_mut).ok_or_else(|| Error::Invalid(format!("no set {set}")))?;
        if m.spec.fingerprint.fill == fill {
            return Ok(());
        }
        m.spec.fingerprint.fill = fill;
        m.key = m.spec.fingerprint.key();
        let declared = m.spec.fingerprint.declared();
        if let Some(Some(s)) = st.table.sets.get_mut(set as usize) {
            s.declared = declared;
        }
        self.drop_stale(st);
        Ok(())
    }

    /// Unmaps the unpinned pages of closing sets; a set goes once it has none. Its slot stays
    /// empty: open plans name it by index.
    fn close_sets(&self, st: &mut State) {
        let closing: Vec<u32> = st.sets.iter().enumerate().filter(|(_, m)| m.as_ref().is_some_and(|m| m.closing)).map(|(i, _)| i as u32).collect();
        for set in closing {
            let keys: Vec<Key> = st.table.pages().filter(|(k, p)| k.0 == set && p.mapped && p.pins.is_empty() && p.releasing.is_empty() && !p.filling).map(|(k, _)| k).collect();
            for key in keys {
                let _ = self.unmap(st, key, Class::N, None);
            }
            if st.table.pages().any(|(k, p)| k.0 == set && p.mapped) {
                continue;
            }
            if st.mech(set).is_ok_and(|m| m.retained != 0) {
                continue;
            }
            if let Some(m) = st.sets.get_mut(set as usize).and_then(Option::take) {
                self.mech.post(Mech::Release(m.va));
            }
            if let Some(slot) = st.table.sets.get_mut(set as usize) {
                *slot = None;
            }
        }
    }

    // ---- the mechanism

    /// Unmaps a page and gives it back (an eviction, a revoke or an unload).
    fn unmap(&self, st: &mut State, key: Key, class: Class, req: Option<u64>) -> Result<()> {
        let Some(home) = st.homes.remove(&key) else { return Ok(()) };
        let (va, off, span) = st.place(key)?;
        if let Some(p) = &home.phys {
            self.commit_page(p);
        }
        st.last_unmap = Some(self.mech.post(Mech::Unmap { va, off, span, phys: home.phys, ready: home.ready }));
        st.sync(key);
        if let Some(s) = st.stats(req) {
            if let Some(e) = s.evicted.get_mut(class.index()) {
                *e += span;
            }
        }
        if let Some(e) = req.and_then(|r| st.step_stats(r)).and_then(|s| s.evicted.get_mut(class.index())) {
            *e += span;
        }
        Ok(())
    }

    /// Maps a page read-write and queues its fill. A speculative map that finds no device
    /// memory stops softly (false).
    fn map(&self, st: &mut State, key: Key, req: Option<u64>, speculative: bool) -> Result<bool> {
        if st.homes.contains_key(&key) {
            return Ok(true);
        }
        let (va, off, span) = st.place(key)?;
        let gen = st.tick();
        let identity = st.mech(key.0)?.key.clone();
        // Counted mapped (and pinned, filling) now; the mechanism creates and maps it after
        // the unmaps decided before it, then queues its fill.
        st.homes.insert(key, Home::new(gen, speculative, identity));
        st.sync(key);
        self.mech.post(Mech::Map { key, gen, va, off, span, job: Job { key, gen, req, span, after: vec![] }, decided: !speculative });
        Ok(true)
    }

    /// A stale page (another fill identity, or a transform released undone) nobody pins is
    /// filled again in place: its mapping stays and the copy waits in-stream on its pending
    /// release events. Preparation may wait outside capture/replay for the source/staging
    /// owners' copy completion; resident replay never performs this preparation.
    fn refill_in_place(&self, st: &mut State, key: Key, req: Option<u64>) -> Result<()> {
        let (va, off, span) = st.place(key)?;
        let identity = st.mech(key.0)?.key.clone();
        let gen = st.tick();
        let after = st.table.page_mut(key).map(|p| std::mem::take(&mut p.releasing)).unwrap_or_default();
        // Its events are the refill's now (it waits on them, then retires them).
        st.events.values_mut().for_each(|q| q.retain(|(_, k)| *k != key));
        let Some(h) = st.homes.get_mut(&key) else { return Ok(()) };
        let old = std::mem::replace(&mut h.ready, Event::DONE);
        // Leased: a grant's identity never changes, so new content takes a new grant.
        let phys = if self.leased_page_of(h.phys.as_ref()) { h.phys.take() } else { None };
        let renew = phys.is_some();
        *h = Home { phys: h.phys.take(), ..Home::new(gen, false, identity) };
        st.sync(key);
        let after: Vec<Event> = after.into_iter().map(Event).collect();
        let job = Job { key, gen, req, span, after: vec![] };
        if renew {
            // A new grant takes its place once its consumers are done.
            if !after.contains(&old) {
                self.backend.retire(old);
            }
            self.mech.post(Mech::Renew { key, gen, va, off, span, old: phys, after, job });
            return Ok(());
        }
        // Read-write and nobody else's: filled again in place, after its consumers in-stream.
        if !after.contains(&old) {
            self.backend.retire(old);
        }
        self.queue_fills(vec![Job { after, ..job }], false)
    }

    /// A fill thread's batch: each page from the store through the page cache, in as few engine
    /// reads as its sets allow; then each page's ready event.
    fn fill(self: &Arc<Self>, jobs: &[Job]) {
        self.fill_batch(jobs);
        // Their fills' events follow them in-stream (or were dropped): nobody needs them now.
        jobs.iter().flat_map(|j| &j.after).for_each(|e| self.backend.retire(*e));
    }

    fn fill_batch(self: &Arc<Self>, jobs: &[Job]) {
        // Under one lock: each live job's place, and its set's spec.
        let (sets, pages) = {
            let st = self.state.lock();
            let mut sets: BTreeMap<u32, SetSpec> = BTreeMap::new();
            let mut pages = vec![];
            for job in jobs {
                let (Some(h), Ok(m), Ok(place)) = (st.homes.get(&job.key), st.mech(job.key.0), st.place(job.key)) else { continue };
                if h.gen == job.gen {
                    sets.entry(job.key.0).or_insert_with(|| m.spec.clone());
                    pages.push((job, place));
                }
            }
            (sets, pages)
        };
        if pages.is_empty() {
            return;
        }
        let start = Instant::now();
        let copied = crate::caught(|| Ok(self.copy(&sets, &pages))).unwrap_or_else(|e| vec![Err(e); pages.len()]);
        // An event per page, after every copy so far.
        let landed: Vec<Result<(Origin, Event)>> = copied.into_iter().map(|c| c.and_then(|origin| Ok((origin, self.backend.fence()?)))).collect();
        let (ns, bytes) = (start.elapsed().as_nanos() as u64, pages.iter().map(|(_, (.., span))| span).sum::<u64>().max(1));
        // After a copy error the engine's copies may still run: a failed page is unmapped only
        // once the copy stream is proven drained.
        let ownership_lost = landed.iter().any(|out| matches!(out, Err(Error::Ownership(_))));
        let drained = !ownership_lost && (landed.iter().all(Result::is_ok) || self.backend.drain().is_ok());
        let mut sticky = false;
        let mut st = self.state.lock();
        if !drained {
            self.ownership_lost.store(true, Ordering::Release);
            std::mem::forget(self.clone());
        }
        for ((job, (.., span)), landed) in pages.iter().zip(landed) {
            let span = *span;
            if st.homes.get(&job.key).is_none_or(|h| h.gen != job.gen) {
                // The page changed under the fill: its event is nobody's.
                if let Ok((_, ready)) = landed {
                    self.backend.retire(ready);
                }
                continue;
            }
            // Read ready at issue (`issued`): its own copies all went out, whatever the batch's end.
            let early = st.homes.get_mut(&job.key).is_some_and(|h| std::mem::take(&mut h.landing));
            let landed = match (early, landed) {
                (true, Ok((_, ready))) => {
                    self.backend.retire(ready);
                    Ok((Origin::Store, None))
                }
                (true, Err(e)) => {
                    sticky |= e.is_sticky();
                    Ok((Origin::Store, None))
                }
                (false, landed) => landed.map(|(origin, ready)| (origin, Some(ready))),
            };
            match landed {
                Ok((origin, ready)) => {
                    // Stored bytes are the weights; a declared set's are once transformed.
                    let filled = sets.get(&job.key.0).is_some_and(|spec| matches!(spec.fingerprint.fill, Fill::Stored));
                    let speculative = st.homes.get(&job.key).is_some_and(|h| h.speculative);
                    // A page read ready at issue is its binders' since: a transform may have
                    // filled it already, so only its accounting is left.
                    if let (Some(ready), Some(h)) = (ready, st.homes.get_mut(&job.key)) {
                        (h.ready, h.ready_on, h.filling, h.filled, h.origin) = (ready, COPY, false, filled, origin);
                        st.sync(job.key);
                    }
                    #[cfg(unix)]
                    if filled && self.leasing() {
                        self.to_land.lock().push(job.key);
                    }
                    let s = st.fills(job.req, job.key, span, origin);
                    let f = &mut s.moved.store;
                    f.bytes += span;
                    // The batch's time, shared by its bytes.
                    f.ns += (u128::from(ns) * u128::from(span) / u128::from(bytes)) as u64;
                    if speculative {
                        s.moved.speculative += span;
                    }
                }
                Err(e) => {
                    tracing::warn!("varena: filling {:?} failed: {e}", job.key);
                    sticky |= e.is_sticky();
                    let speculative = st.homes.get(&job.key).is_some_and(|h| h.speculative);
                    // Undrained, it stays filling (P) for good: never unmapped under a copy.
                    if drained {
                        if let Some(h) = st.homes.get_mut(&job.key) {
                            h.filling = false;
                        }
                        let _ = self.unmap(&mut st, job.key, Class::P, None);
                    }
                    if !speculative || !drained {
                        let error = if drained { e } else { Error::Ownership(format!("fill failed ({e}); its copies did not drain")) };
                        st.failed.insert(job.key, error);
                    }
                }
            }
        }
        if sticky { self.ownership_lost.store(true, Ordering::Release); }
        self.unlock(st);
        self.landed.notify_all();
    }

    /// The batch's copies from the store, through the page cache: one engine read per set.
    /// Each page's origin, or its error.
    fn copy(&self, sets: &BTreeMap<u32, SetSpec>, pages: &[(&Job, (Va, u64, u64))]) -> Vec<Result<Origin>> {
        let mut copies: Vec<Copying> = pages
            .iter()
            .map(|(job, (va, off, span))| {
                let dst = self.backend.target(*va, *off, *span);
                // A refill in place writes after the page's last consumers, in-stream (a host
                // page has no stream: 0).
                let stream = match dst {
                    Dst::Device { stream, .. } => stream,
                    Dst::Host { .. } => 0,
                };
                let out = job.after.iter().try_for_each(|e| self.backend.wait(*e, stream)).err().map(Err);
                Copying { key: job.key, gen: job.gen, dst, out }
            })
            .collect();
        for (set, spec) in sets {
            let todo = |c: &Copying| c.key.0 == *set && c.out.is_none();
            let direct: Vec<(u32, Dst)> = copies.iter().filter(|c| todo(c)).map(|c| (c.key.1, c.dst)).collect();
            if !direct.is_empty() {
                let owners: Vec<(Key, u64, usize)> = copies.iter().filter(|c| todo(c)).map(|c| {
                    (c.key, c.gen, match c.dst { Dst::Device { stream, .. } => stream, Dst::Host { .. } => 0 })
                }).collect();
                let filled = matches!(spec.fingerprint.fill, Fill::Stored);
                let issued = |i: usize, streams: &[usize]| owners.get(i).into_iter().for_each(|o| self.issued(*o, streams, filled));
                let stored = self.store(spec, &direct, Some(&issued)).map(|()| Origin::Store);
                copies.iter_mut().filter(|c| todo(c)).for_each(|c| c.out = Some(stored.clone()));
            }
        }
        copies.into_iter().map(|c| c.out.unwrap_or_else(|| Err(Error::Internal("a page left unfilled".into())))).collect()
    }

    /// A fill thread's batch failed outside its copies (a panic): its pages go, their binders fail.
    fn fail(self: &Arc<Self>, jobs: &[Job], e: Error) {
        // Its copies may have been issued: unmapped only once the copy stream is drained.
        let drained = self.backend.drain().is_ok();
        let mut st = self.state.lock();
        let e = if drained { e } else {
            self.ownership_lost.store(true, Ordering::Release);
            std::mem::forget(self.clone());
            Error::Ownership(format!("fill panicked ({e}); its copies did not drain"))
        };
        for job in jobs {
            // A page read ready at issue had all its copies go out: its binders keep it.
            if let Some(h) = st.homes.get_mut(&job.key).filter(|h| h.gen == job.gen) {
                h.landing = false;
            }
            if st.homes.get(&job.key).is_some_and(|h| h.gen == job.gen && h.filling) {
                if drained {
                    if let Some(h) = st.homes.get_mut(&job.key) {
                        h.filling = false;
                    }
                    let _ = self.unmap(&mut st, job.key, Class::P, None);
                }
                st.failed.insert(job.key, e.clone());
            }
        }
        self.unlock(st);
        self.landed.notify_all();
    }

    /// `first` and the jobs queued behind it (demand misses first), up to the burst's bytes so
    /// far: batches double, so the first pages are ready soonest and a few reads carry the
    /// rest. Never less than the engine keeps in flight (its threads' requests): a smaller read
    /// leaves its threads idle. Groups stay whole.
    fn batch(&self, first: Vec<Job>, demand: &Receiver<Vec<Job>>, ahead: &Receiver<Vec<Job>>) -> Vec<Job> {
        let budget = self.fill_budget();
        self.burst.0.lock().batches += 1;
        let mut jobs = first;
        let mut bytes: u64 = jobs.iter().map(|j| j.span).sum();
        while bytes < budget {
            let Ok(group) = demand.try_recv().or_else(|_| ahead.try_recv()) else { break };
            bytes += group.iter().map(|j| j.span).sum::<u64>();
            jobs.extend(group);
        }
        let mut b = self.burst.0.lock();
        b.queued = b.queued.saturating_sub(jobs.len() as u64);
        jobs
    }

    /// A batch's bytes: what this burst filled so far, at least one engine round of requests.
    fn fill_budget(&self) -> u64 {
        let engine = tensorfs::io::ReadOpts::default();
        self.burst.0.lock().bytes.max((engine.threads * engine.request) as u64)
    }

    /// Queues pages' fills as one group: demand misses, or `ahead` of their use.
    fn queue_fills(&self, jobs: Vec<Job>, ahead: bool) -> Result<()> {
        if jobs.is_empty() {
            return Ok(());
        }
        let n = jobs.len() as u64;
        self.burst.0.lock().queued += n;
        let queue = if ahead { &self.jobs.ahead } else { &self.jobs.demand };
        queue.send(jobs).map_err(|_| {
            self.burst.0.lock().queued -= n;
            Error::Closed("the fill threads".into())
        })
    }

    /// The mechanism thread's mapped pages go to the fill threads, demand first.
    fn flush_fills(&self) {
        let (demand, ahead) = std::mem::take(&mut *self.grouped.lock());
        if self.queue_fills(demand, false).and_then(|()| self.queue_fills(ahead, true)).is_err() {
            tracing::warn!("varena: the fill threads are gone");
        }
    }

    /// Bytes the mechanism thread holds for the fill threads.
    fn grouped_bytes(&self) -> u64 {
        let g = self.grouped.lock();
        g.0.iter().chain(&g.1).map(|j| j.span).sum()
    }

    /// A batch is done: its bytes count for its burst, which ends when nothing is queued or
    /// filling.
    fn batched(&self, jobs: &[Job]) {
        let mut b = self.burst.0.lock();
        b.batches = b.batches.saturating_sub(1);
        b.bytes += jobs.iter().map(|j| j.span).sum::<u64>();
        if b.idle() {
            b.bytes = 0;
            self.burst.1.notify_all();
        }
    }

    /// The set's regions from its store into their destinations, in one engine read, in
    /// region order: neighbours in the file and the reservation read as one run. `issued`
    /// hears of each device region (its index in `regions`) once its copies are all issued.
    #[cfg(unix)]
    fn store(&self, spec: &SetSpec, regions: &[(u32, Dst)], issued: Option<tensorfs::io::Issued>) -> Result<()> {
        let mut order: Vec<(usize, &(u32, Dst))> = regions.iter().enumerate().collect();
        order.sort_unstable_by_key(|(_, (r, _))| *r);
        let placements = order.iter().map(|(_, (r, dst))| spec.layout.region(*r).map(|region| Placement { items: &region.items, origin: region.offset, dst: *dst })).collect::<Result<Vec<_>>>()?;
        let threads = tensorfs::io::ReadOpts::default().threads;
        let placed = |p: usize, streams: &[usize]| {
            if let (Some(issued), Some((i, _))) = (issued, order.get(p)) {
                issued(*i, streams);
            }
        };
        // SAFETY: each `dst` is a mapped page only this fill writes.
        unsafe { spec.source.read_placements(&placements, threads, tensorfs::io::source::Mode::Buffered, &self.tally, issued.map(|_| &placed as tensorfs::io::Issued)) }?;
        Ok(())
    }

    #[cfg(windows)]
    fn store(&self, spec: &SetSpec, _: &[(u32, Dst)], _: Option<tensorfs::io::Issued>) -> Result<()> {
        match *spec.source {}
    }

    /// A store fill's copies of page `key` are all issued: it reads ready now, its event after
    /// them on the copy stream, so its binders wait in-stream instead of for the whole batch.
    fn issued(&self, (key, gen, copy): (Key, u64, usize), streams: &[usize], filled: bool) {
        let ready = streams.iter().try_for_each(|&s| {
            let e = self.backend.record(s)?;
            let waited = self.backend.wait(e, copy);
            self.backend.retire(e);
            waited
        });
        // A failed record leaves the page to the batch's end.
        let Ok(ready) = ready.and_then(|()| self.backend.fence()) else { return };
        let mut st = self.state.lock();
        match st.homes.get_mut(&key) {
            Some(h) if h.gen == gen && h.filling => {
                (h.ready, h.ready_on, h.filling, h.landing, h.filled, h.origin) = (ready, COPY, false, true, filled, Origin::Store);
            }
            _ => {
                self.unlock(st);
                return self.backend.retire(ready);
            }
        }
        st.sync(key);
        self.unlock(st);
        self.landed.notify_all();
    }

    // ---- demands

    /// Free bytes for a page now, from varena's own figures (never the device's: a pass reads
    /// none). Leased: what the manager can allocate and reclaim from others. Standalone: what
    /// the budget its pages and torch's share has left (a refused create says the device has
    /// less).
    fn free(&self, _st: &State) -> u64 {
        #[cfg(unix)]
        if let Some(agent) = self.lease.get() {
            let f = agent.table().figures();
            return f.free.saturating_add(f.revocable);
        }
        self.allowance.left()
    }

    /// Free bytes speculative work (prefetch, a streamed set's regrowth) may take: never another
    /// tenant's (leased, the manager's free bytes alone).
    fn spare(&self, st: &State) -> u64 {
        #[cfg(unix)]
        if let Some(agent) = self.lease.get() {
            return agent.table().figures().free;
        }
        self.free(st)
    }

    /// Makes `bytes` free for a bind or a range's preparation, taking its own pages in
    /// [`decide::victims`]' order, never `keep` (the pages being bound): idle pages unmapped, a
    /// running set streamed through slots, pages whose last consumers still run taken once their
    /// release events complete (waited on here). `Exhausted` at once beyond what it could ever
    /// have, else only when a round takes nothing and nothing of its own is about to come back
    /// (a fill: waited for). `why` and `req` name it in the request's record.
    fn make_room(&self, bytes: u64, keep: &[Key], why: &'static str, req: Option<u64>, free_only: bool) -> Result<u64> {
        let mut record = MakeRoom { step: 0, trigger: why, need: bytes, freed: 0, waited: 0 };
        let out = loop {
            let mut st = self.lock();
            // A refused create, or a manager's refusal, says there is less than the figures: the
            // round takes the page's bytes itself. Leased, a page is asked of the manager first
            // (its reclaim makes room across processes, this one's idle pages too). Standalone, a
            // miss bounds the budget by the driver's free bytes (another tenant's churn); a pass
            // never reads them.
            let refused = keep.iter().any(|k| st.short.remove(k).is_some());
            let free = match (refused, self.leasing()) {
                (true, _) => 0,
                (false, true) => u64::MAX,
                (false, false) => self.free(&st).min(self.backend.free().unwrap_or(u64::MAX)),
            };
            let req = req.or_else(|| st.active.iter().find(|(_, a)| a.thread == std::thread::current().id()).map(|((r, _), _)| *r));
            let short = |st: &State| -> Result<u64> { Err(Error::Exhausted(exhausted(bytes.saturating_sub(free), None, req.and_then(|r| st.stats.get(&r)).cloned().unwrap_or_default()))) };
            if free >= bytes {
                self.unlock(st);
                break Ok(free);
            }
            // Idle work (a request opened idle, a free-only bind) takes free bytes only; nothing
            // takes more than it could ever have (its pinned pages stay).
            let pinned: u64 = st.table.pages().filter(|(k, p)| p.mapped && (!p.pins.is_empty() || p.filling) && !keep.contains(k)).map(|(_, p)| p.bytes).sum();
            let idle = free_only || req.and_then(|r| st.table.request(r)).is_some_and(|r| r.idle);
            if idle || bytes > (free + st.table.mapped()).saturating_sub(pinned) {
                let out = short(&st);
                self.unlock(st);
                break out;
            }
            record.step = req.and_then(|r| st.table.request(r)).map_or(0, |r| r.cursor.step);
            let mut want = bytes - free;
            let (mut unmapped, mut events, mut streamed) = (0, vec![], HashSet::new());
            for Victim { key, bytes: b, releasing } in decide::victims(&st.table, keep) {
                if want == 0 {
                    break;
                }
                // Step 2: a running set's own pages are not evicted one by one; the set streams its upper blocks
                // through slots instead, once per round, and its other pages stay.
                if st.table.distance(key).is_some_and(|d| d.0 == 0) {
                    if streamed.insert(key.0) {
                        let taken = self.stream(&mut st, key.0, want, req)?;
                        (unmapped, want) = (unmapped + taken.unmapped, want.saturating_sub(taken.unmapped + taken.pending));
                        events.extend(taken.events);
                    }
                    continue;
                }
                if releasing {
                    events.extend(st.table.page(key).map(|p| p.releasing.clone()).unwrap_or_default());
                } else {
                    let class = st.table.class(key).unwrap_or(Class::N);
                    self.unmap(&mut st, key, class, req)?;
                    unmapped += b;
                }
                want = want.saturating_sub(b);
            }
            let coming = st.homes.iter().any(|(k, h)| (h.filling || h.landing) && !keep.contains(k));
            let last = st.last_unmap;
            self.unlock(st);
            // The unmaps done before the bytes count.
            if let Some(t) = last.filter(|_| unmapped > 0) {
                self.mech.settle_to(t);
            }
            for e in &events {
                self.backend.finish(Event(*e))?;
            }
            record.waited += events.len() as u32;
            record.freed += unmapped;
            if unmapped == 0 && events.is_empty() {
                let mut st = self.lock();
                if coming {
                    self.wait(&mut st)?;
                    self.unlock(st);
                    continue;
                }
                let out = short(&st);
                self.unlock(st);
                break out;
            }
        };
        if record.freed > 0 || record.waited > 0 {
            let mut st = self.state.lock();
            let req = req.or_else(|| st.active.iter().find(|(_, a)| a.thread == std::thread::current().id()).map(|((r, _), _)| *r));
            if let Some(s) = st.stats(req) {
                s.make_room.push(record);
            }
        }
        out
    }

    /// Recovery, at a pass boundary only (a shrink is at once): a streaming set takes back the
    /// blocks at its watermark, one by one, then refills its slots, into spare bytes only, mapped
    /// and filled on the copy stream, never evicting.
    fn regrow(&self, req: u64, set: u32) {
        let mut st = self.lock();
        let (Some(watermark), Ok(blocks)) = (st.streams.get(&set).map(|s| s.watermark), st.mech(set).map(|m| m.spec.layout.regions.len() as u32)) else {
            return self.unlock(st);
        };
        let mut room = self.spare(&st);
        let mut w = watermark;
        while let Some(bytes) = (w < blocks).then(|| st.table.page((set, w)).map(|p| p.bytes)).flatten().filter(|b| *b <= room) {
            // A slot already there is resident now at no cost; a block with no page is mapped.
            if !st.homes.contains_key(&(set, w)) {
                if !self.map(&mut st, (set, w), Some(req), true).unwrap_or(false) {
                    break;
                }
                room -= bytes;
            }
            w += 1;
        }
        if w >= blocks {
            st.streams.remove(&set);
        } else if let Some(s) = st.streams.get_mut(&set) {
            s.watermark = w;
        }
        // Its slots, refilled here only (a bind mid-pass moves one, never makes one): the next
        // upper blocks with no page.
        let mut next = w;
        for _ in st.slots(set, w)..stream::SLOTS {
            while next < blocks && st.homes.contains_key(&(set, next)) {
                next += 1;
            }
            let Some(bytes) = (next < blocks).then(|| st.table.page((set, next)).map(|p| p.bytes)).flatten().filter(|b| *b <= room) else { break };
            if !self.map(&mut st, (set, next), Some(req), true).unwrap_or(false) {
                break;
            }
            room -= bytes;
            next += 1;
        }
        self.unlock(st);
    }

    /// Maps the next blocks of `req`'s current step into spare bytes only.
    fn prefetch(&self, req: u64) {
        let mut st = self.lock();
        let spare = self.spare(&st);
        for key in decide::prefetch(&st.table, req, spare) {
            if self.map(&mut st, key, Some(req), true).unwrap_or(false) {
                let step = st.table.request(req).map_or(0, |r| r.cursor.step);
                if let Some(p) = st.table.page_mut(key) {
                    p.speculative = Some((req, step));
                }
            }
        }
        self.unlock(st);
    }

    // ---- the calls

    fn load(self: &Arc<Self>, spec: SetSpec, level: Level) -> Result<SetId> {
        let (id, again) = {
            let mut st = self.lock();
            let existing = st.sets.iter().position(|m| m.as_ref().is_some_and(|m| m.spec.name == spec.name && !m.closing));
            let out = match existing {
                Some(i) if st.mech(i as u32)?.spec.fingerprint.digest == spec.fingerprint.digest => {
                    self.refill(&mut st, i as u32, spec.fingerprint.fill)?;
                    (i as u32, true)
                }
                Some(_) => return Err(Error::Invalid(format!("set {} is loaded with another layout or release", spec.name))),
                None => {
                    let layout = &spec.layout;
                    let va = self.backend.reserve(layout.nbytes)?;
                    let pages = layout.regions.iter().map(|r| Page { bytes: r.span, ..Page::default() }).collect();
                    let set = Set { name: spec.name.clone(), pages, declared: spec.fingerprint.declared() };
                    let i = st.sets.len() as u32;
                    st.sets.push(Some(SetMech { key: spec.fingerprint.key(), spec, va, closing: false, retained: 0 }));
                    st.table.sets.push(Some(set));
                    (i, false)
                }
            };
            self.unlock(st);
            out
        };
        match level {
            Level::Ram => {
                if again {
                    // Down to RAM: its unpinned device pages go.
                    let mut st = self.lock();
                    let keys: Vec<Key> = st.table.pages().filter(|(k, p)| k.0 == id && p.mapped && p.pins.is_empty() && p.releasing.is_empty() && !p.filling).map(|(k, _)| k).collect();
                    for key in keys {
                        self.unmap(&mut st, key, Class::N, None)?;
                    }
                    self.unlock(st);
                }
                self.readahead(id)?
            }
            Level::Gpu => {
                // In block order, evicting nothing of its own: standalone into spare bytes;
                // leased, every page is asked of the manager (retained ones come back with no
                // copy), and one it refuses stays unmapped quietly.
                {
                    let mut st = self.lock();
                    let leased = self.leasing();
                    let mut free = if leased { u64::MAX } else { self.spare(&st) };
                    let keys: Vec<(Key, u64)> = st.table.pages().filter(|(k, p)| k.0 == id && !p.mapped).map(|(k, p)| (k, p.bytes)).collect();
                    for (key, bytes) in keys {
                        if bytes > free {
                            break;
                        }
                        self.map(&mut st, key, None, leased)?;
                        free -= bytes;
                    }
                    self.unlock(st);
                }
                let mut st = self.lock();
                // Until its fills' batches end (landed and accounted for), not only issued.
                while st.table.pages().any(|(k, p)| k.0 == id && p.filling) || st.homes.iter().any(|(k, h)| k.0 == id && h.landing) {
                    self.wait(&mut st)?;
                }
                self.unlock(st);
            }
        }
        Ok(SetId(id))
    }

    /// Brings the set's stored bytes into host RAM, the page cache, ahead of its fills.
    #[cfg(unix)]
    fn readahead(&self, set: u32) -> Result<()> {
        let spec = {
            let st = self.lock();
            let spec = st.mech(set)?.spec.clone();
            self.unlock(st);
            spec
        };
        let items: Vec<tensorfs::io::source::Item> = spec.layout.regions.iter().flat_map(|r| r.items.iter().cloned()).collect();
        Ok(spec.source.readahead(&items)?)
    }

    #[cfg(windows)]
    fn readahead(&self, set: u32) -> Result<()> {
        let st = self.lock();
        match *st.mech(set)?.spec.source {}
    }

    fn resolve(st: &State, plan: &Plan) -> Vec<(Vec<u32>, u32)> {
        let index = |name: &String| st.sets.iter().position(|m| m.as_ref().is_some_and(|m| &m.spec.name == name && !m.closing)).map(|i| i as u32);
        plan.steps.iter().map(|s| (s.sets.iter().filter_map(index).collect(), s.passes.max(1))).collect()
    }

    fn open_request(&self, id: &str, plan: &Plan, idle: bool) -> Result<u64> {
        let mut st = self.lock();
        let rid = st.tick();
        let steps = Self::resolve(&st, plan);
        // Its anonymous host peak is measured from its start (the process's, when it runs alone).
        match st.window.as_mut() {
            Some(window) => drop(window.look()),
            None => st.window = HostWindow::start(),
        }
        st.table.requests.push(Req { id: rid, name: id.to_string(), steps, idle, ..Req::default() });
        // A cold request's record carries the fills since the last one ended (its loads).
        let earlier = std::mem::take(&mut st.unclaimed);
        st.stats.insert(rid, Stats { request: id.to_string(), moved: earlier.moved, trace: earlier.trace, ..Stats::default() });
        self.unlock(st);
        Ok(rid)
    }

    /// Binds `block` of `set` for `stream`; `claims` are the caller's own preparation pins,
    /// which hold the page for this bind and never against it.
    fn bind_for(self: &Arc<Self>, set: u32, block: u32, stream: usize, request: Option<u64>, claims: &[u64], free_only: bool) -> Result<Pin> {
        let key = (set, block);
        let start = Instant::now();
        let mut st = self.lock();
        let req = request.or_else(|| st.binding(set));
        if st.table.page(key).is_none() {
            return Err(Error::Invalid(format!("no block {block} in set {set}")));
        }
        let (mut hit, mut waited, mut speculative) = (true, false, false);
        loop {
            if let Some(e) = st.failed.get(&key).cloned() {
                // A page whose failed copies could not be proven drained keeps failing.
                if !st.homes.get(&key).is_some_and(|h| h.filling) {
                    st.failed.remove(&key);
                }
                return Err(e);
            }
            let identity = st.mech(set)?.key.clone();
            let pinned = st.table.page(key).is_some_and(|p| p.pins.iter().any(|id| !claims.contains(id)));
            // Waiting on pins only this thread holds would wait on itself.
            let me = std::thread::current().id();
            let mine = pinned && st.pins.iter().filter(|(id, (k, ..))| *k == key && !claims.contains(id)).all(|(_, (.., t))| *t == me);
            // A streamed set's upper block takes a slot rather than a new page (step 2's watermark).
            match self.stream_in(&mut st, key, req)? {
                StreamIn::Moved => {
                    hit = false;
                    continue;
                }
                StreamIn::Wait => {
                    (hit, waited) = (false, true);
                    self.wait(&mut st)?;
                    continue;
                }
                StreamIn::No => {}
            }
            match st.homes.get(&key) {
                // Not the set's weights now, and unpinned: filled again in place.
                Some(h) if h.identity != identity && !h.filling && !pinned => {
                    hit = false;
                    self.refill_in_place(&mut st, key, req)?;
                }
                Some(h) if (h.identity != identity || h.transforming) && !h.filling && mine => {
                    return Err(Error::Held(format!("block {block} of set {set} is pinned by this thread alone")));
                }
                // Still pinned or filling: its pin's drop or its fill wakes the binder.
                Some(h) if h.identity != identity => {
                    hit = false;
                    waited = true;
                    self.wait(&mut st)?;
                }
                Some(h) if h.filled || !(h.filling || h.transforming) => {
                    speculative |= h.speculative;
                    break;
                }
                Some(h) => {
                    speculative |= h.speculative;
                    hit = false;
                    waited = true;
                    self.wait(&mut st)?;
                }
                None => {
                    hit = false;
                    let bytes = st.table.page(key).map_or(0, |p| p.bytes);
                    self.unlock(st);
                    let refused = match self.make_room(bytes, &[key], if free_only { "bind_free" } else { "bind" }, req, free_only) {
                        Err(e @ Error::Exhausted(_)) if !free_only => Some(e),
                        made => made.map(|_| None)?,
                    };
                    st = self.lock();
                    if let Some(e) = refused {
                        // A streamed set with only its slots left: one of them moves to the block (the floor: one block).
                        match self.stream_in(&mut st, key, req)? {
                            StreamIn::Moved => continue,
                            StreamIn::Wait => {
                                waited = true;
                                self.wait(&mut st)?;
                                continue;
                            }
                            StreamIn::No => return Err(e),
                        }
                    }
                    if !st.homes.contains_key(&key) {
                        self.map(&mut st, key, req, false)?;
                    }
                }
            }
        }
        let (va, off, span) = st.place(key)?;
        let id = st.tick();
        let now = st.clock;
        let Some(h) = st.homes.get_mut(&key) else { return Err(Error::Internal("a bound page lost its home".into())) };
        let fresh = h.fresh();
        let transform = fresh;
        if transform {
            h.transforming = true;
        }
        let (ready, origin) = (h.ready, if hit { Origin::Hit } else { h.origin });
        st.pins.insert(id, (key, req, std::thread::current().id()));
        if let Some(p) = st.table.page_mut(key) {
            p.last_use = now;
            #[cfg(unix)]
            st.used_ns.insert(key, monotonic_ns());
        }
        st.sync(key);
        if let Some(r) = req.and_then(|r| st.table.request_mut(r)) {
            r.cursor.next.insert(set, block + 1);
        }
        let name = st.mech(set)?.spec.name.clone();
        let step = req.and_then(|r| st.table.request(r)).map_or(0, |r| r.cursor.step);
        if let Some(s) = st.stats(req) {
            s.trace.push(Ref { step, set: name, block, bytes: span, origin, speculative });
            if waited || !hit {
                s.stall_ns += start.elapsed().as_nanos() as u64;
            }
        }
        self.unlock(st);
        if let Some(r) = req {
            self.prefetch(r);
        }
        let pin = Pin { ptr: va.base + off, len: span, ready, fresh, inner: self.clone().into(), key, id, stream, transform, transformed: AtomicBool::new(false) };
        self.backend.wait(ready, stream)?;
        Ok(pin)
    }
    /// Moves the request's cursor to the pass of `set`; true when a step began. A pass the plan
    /// does not declare takes one synthetic step at the end, reused by the next such pass.
    fn advance(st: &mut State, req: u64, set: u32) -> Result<bool> {
        let synthetic = st.synthetic.contains(&req);
        let r = st.table.request_mut(req).ok_or_else(|| Error::Closed("the request".into()))?;
        let i = r.cursor.step;
        let current = r.steps.get(i as usize).is_some_and(|(s, _)| s.contains(&set));
        let begun = r.cursor.passes.get(&set).copied().unwrap_or(0);
        let later = (i + 1..r.steps.len() as u32).find(|j| r.steps.get(*j as usize).is_some_and(|(s, _)| s.contains(&set)));
        let first = begun == 0 && r.cursor.passes.is_empty();
        // A set of the current step stays in it, however many passes it runs; only a set the
        // step lacks moves the cursor, to the first later step that has it.
        if current {
            r.cursor.passes.insert(set, begun + 1);
            r.cursor.next.insert(set, 0);
            return Ok(first);
        }
        let j = match later {
            Some(j) => j,
            None if synthetic => {
                let last = r.steps.len().saturating_sub(1);
                if let Some(step) = r.steps.get_mut(last) {
                    *step = (vec![set], 1);
                }
                last as u32
            }
            None => {
                r.steps.push((vec![set], 1));
                st.synthetic.insert(req);
                return Self::advance(st, req, set);
            }
        };
        r.cursor = crate::table::Cursor { step: j, passes: BTreeMap::from([(set, 1)]), next: BTreeMap::from([(set, 0)]) };
        Ok(true)
    }

    /// A pass of `set` for `req` (its dispatch key's `key`): the plan's cursor moves to it, the
    /// kept pages of its step are adopted, peers are asked when it is short (co-residency), and
    /// its first blocks are prefetched into free bytes.
    fn pass(&self, req: u64, set: u32) -> Result<u64> {
        let id = {
            let mut st = self.lock();
            st.mech(set)?;
            if Self::advance(&mut st, req, set)? {
                let free = self.free(&st);
                if let Some(s) = st.step_stats(req) {
                    s.capacity = free;
                }
            }
            let id = st.tick();
            st.active.insert((req, set), Active { id, thread: std::thread::current().id() });
            self.unlock(st);
            id
        };
        self.regrow(req, set);
        self.prefetch(req);
        Ok(id)
    }

    /// A pass ends: `peak` (what torch allocated at most) and the pages held now go into the
    /// request's record.
    fn end_pass(&self, req: u64, pass: u64, peak: u64) -> Result<()> {
        let mut st = self.lock();
        st.active.retain(|(r, _), a| !(*r == req && a.id == pass));
        let held = st.table.mapped() + peak;
        if let Some(s) = st.stats(Some(req)) {
            s.peak = s.peak.max(held);
            s.compute = s.compute.max(peak);
        }
        self.unlock(st);
        Ok(())
    }

    fn end_request(&self, req: u64) -> Stats {
        let outlived = self.torch.get().map(|t| t.end(crate::arena::Owner::Request(req)).outlived).unwrap_or_default();
        let mut st = self.lock();
        self.unstream(&mut st, req);
        st.active.retain(|(r, _), _| *r != req);
        st.synthetic.remove(&req);
        st.table.requests.retain(|r| r.id != req);
        let leftover: u64 = st.pins.values().filter(|(_, r, _)| *r == Some(req)).filter_map(|(k, ..)| st.table.page(*k)).map(|p| p.bytes).sum();
        let host = st.window.as_mut().and_then(HostWindow::look);
        if st.table.requests.is_empty() {
            st.window = None;
        }
        let stats = Stats { outlived, leftover, host, ..st.stats.remove(&req).unwrap_or_default() };
        st.last = Some(stats.clone());
        self.unlock(st);
        stats
    }

}

/// This process's open devices: one Memory each.
#[derive(Clone, Copy)]
enum DeviceState { Active, Retiring, Quarantined }
static OPEN: Mutex<BTreeMap<String, DeviceState>> = Mutex::new(BTreeMap::new());
static CLOSED: Condvar = Condvar::new();

/// A device this process has open, given back only after its prior owner has retired.
struct Device(String);

impl Device {
    fn open(uuid: &str) -> Result<Device> {
        let mut open = OPEN.lock();
        loop {
            match open.get(uuid) {
                Some(DeviceState::Active) => return Err(Error::Invalid(format!("{uuid} is already open in this process"))),
                Some(DeviceState::Quarantined) => return Err(Error::Ownership(format!("{uuid} has quarantined device ownership; retire this executor"))),
                // Opening is preparation, outside Python/allocator locks. Existing public
                // owners still refuse immediately; only their completed handoff is awaited.
                Some(DeviceState::Retiring) => CLOSED.wait(&mut open),
                None => break,
            }
        }
        open.insert(uuid.to_string(), DeviceState::Active);
        Ok(Device(uuid.to_string()))
    }

    fn retiring(&self, quarantined: bool) {
        let mut open = OPEN.lock();
        if let Some(state) = open.get_mut(&self.0) {
            if quarantined { *state = DeviceState::Quarantined; }
            else if !matches!(state, DeviceState::Quarantined) { *state = DeviceState::Retiring; }
        }
        drop(open);
        CLOSED.notify_all();
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        OPEN.lock().remove(&self.0);
        CLOSED.notify_all();
    }
}

impl Memory {
    /// Drops this handle and, once every public handle of it is gone (requests, pins,
    /// passes), waits until its memory has retired: its pages given back. A process ends after
    /// it; ending without it leaves its leases to the manager's reap.
    pub fn retire(self) -> Result<()> {
        let uuid = self.inner._device.0.clone();
        drop(self);
        let mut open = OPEN.lock();
        loop {
            match open.get(&uuid) {
                None => return Ok(()),
                Some(DeviceState::Quarantined) => return Err(Error::Ownership(format!("{uuid} retired with quarantined device ownership"))),
                Some(_) => CLOSED.wait(&mut open),
            }
        }
    }
}

impl Drop for Memory {
    /// Final physical retirement is transferred by Owner, so a Python/DLPack destructor never
    /// waits for a CUDA consumer or a host callback.
    fn drop(&mut self) {
        if self.handles.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        self.inner.burst.0.lock().closed = true;
        self.inner.burst.1.notify_all();
        self.inner.landed.notify_all();
    }
}

impl Inner {
    /// Only the last public owner is queued here. Internal references may still finish work;
    /// their count is a quiescence check, never the ownership-transfer decision.
    fn retire_public(self: &Arc<Self>) -> Result<bool> {
        self.check_ownership()?;
        if Arc::strong_count(self) != 1 { return Ok(false); }
        let mut st = self.state.lock();
        self.drain(&mut st);
        for (event, _) in st.events.values().flatten() {
            if !self.backend.query(*event)? { return Ok(false); }
        }
        for home in st.homes.values() {
            if home.filling || home.transforming || !self.backend.query(home.ready)? { return Ok(false); }
        }
        // Leased pages go back to the manager, which retains the landed ones.
        let keys: Vec<Key> = st.homes.keys().copied().collect();
        for key in keys { self.unmap(&mut st, key, Class::N, None)?; }
        let empty = st.homes.is_empty();
        drop(st);
        Ok(empty && self.mech.idle())
    }
}

impl Inner {
    /// The last public owner went: retires once nothing is left to finish. Each round waits on
    /// the work it found unfinished, never on a clock; a failure quarantines it.
    fn retire(self: Arc<Self>) {
        loop {
            let round = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.retire_public().and_then(|done| if done { Ok(true) } else { self.quiesce().map(|()| false) })));
            match round {
                Ok(Ok(true)) => return,
                Ok(Ok(false)) => {}
                Ok(Err(_)) | Err(_) => {
                    self.ownership_lost.store(true, Ordering::Release);
                    self._device.retiring(true);
                    self.landed.notify_all();
                    std::mem::forget(self);
                    return;
                }
            }
        }
    }

    /// Blocks until the work a retirement waits on is done: fills, every pending event and
    /// ready event (a host wait, on the retirer's own thread), mechanism calls, and every
    /// worker's reference.
    fn quiesce(self: &Arc<Self>) -> Result<()> {
        {
            let mut st = self.lock();
            while st.homes.values().any(|h| h.filling) {
                self.wait(&mut st)?;
            }
            self.unlock(st);
        }
        self.finish_pending()?;
        self.mech.settle();
        let mut alone = self.quiet.lock.lock();
        while Arc::strong_count(self) > 1 {
            self.quiet.gone.wait(&mut alone);
        }
        Ok(())
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.burst.0.lock().closed = true;
        self.burst.1.notify_all();
        // The mechanism's thread ends with the Memory: what it was still to do happens here,
        // then its Close (taken by it alone) ends it.
        let left: Vec<(u64, Mech)> = self.mech.rx.try_iter().collect();
        self.mech.post(Mech::Close);
        for (_, op) in left {
            match op {
                Mech::Unmap { va, off, span, phys, ready } => {
                    let _ = self.backend.finish(ready);
                    let _ = self.backend.unmap(va, off, span);
                    phys.into_iter().for_each(|p| { let _ = self.give_page(p); });
                    self.backend.retire(ready);
                }
                Mech::Renew { va, off, span, old, .. } => {
                    let _ = self.backend.unmap(va, off, span);
                    old.into_iter().for_each(|p| { let _ = self.give_page(p); });
                }
                Mech::Move { va, from, span, phys, .. } => {
                    let _ = self.backend.unmap(va, from, span);
                    let _ = self.give_page(phys);
                }
                Mech::Release(va) => self.backend.release(va),
                Mech::Map { .. } | Mech::Close => {}
                #[cfg(unix)]
                Mech::Revoke | Mech::Reclaim { .. } => {}
            }
        }
        let mut st = self.state.lock();
        let homes: Vec<(Key, Home)> = st.homes.drain().collect();
        for (key, home) in homes {
            let Some(phys) = home.phys else { continue };
            if let Ok((va, off, span)) = st.place(key) {
                let _ = self.backend.unmap(va, off, span);
            }
            let _ = self.give_page(phys);
        }
        let sets: Vec<SetMech> = st.sets.drain(..).flatten().collect();
        drop(st);
        for m in sets {
            self.backend.release(m.va);
        }
        // Its retired events that completed go with it (one still pending, with the backend).
        self.backend.sweep();
    }
}

fn fill_loop(demand: &Receiver<Vec<Job>>, ahead: &Receiver<Vec<Job>>, inner: &Weak<Inner>) {
    loop {
        // A demand miss first; both queues close with the Memory.
        let group = select_biased! {
            recv(demand) -> j => j,
            recv(ahead) -> j => j,
        };
        let Ok(group) = group else { return };
        let Some(me) = Work::of(inner) else { return };
        let batch = me.batch(group, demand, ahead);
        // A panic outside the copies fails the batch's binders; the thread keeps serving.
        if let Err(p) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| me.fill(&batch))) {
            let msg = p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned());
            me.fail(&batch, Error::Internal(format!("panic: {}", msg.unwrap_or_default())));
        }
        me.batched(&batch);
    }
}

/// The mechanism thread: the device's VMM calls, in the order decided, off the state lock.
fn mech_loop(queue: &Mechs, inner: &Weak<Inner>) {
    while let Ok((ticket, op)) = queue.rx.recv() {
        if matches!(op, Mech::Close) {
            queue.finished(ticket);
            return;
        }
        let Some(me) = Work::of(inner) else { return };
        // Consecutive maps (a bind's block) reach the fill threads as one group: handed over
        // before any other call, once nothing waits for this thread, or once they make a batch.
        if !matches!(op, Mech::Map { .. }) {
            me.flush_fills();
        }
        me.mechanism(op);
        if queue.rx.is_empty() || me.grouped_bytes() >= me.fill_budget() {
            me.flush_fills();
        }
        me.landed.notify_all();
        // Done with the Memory before it reads as done: a settled Memory is this thread's no
        // more.
        drop(me);
        queue.finished(ticket);
        // Retired events destroyed here, never on a caller's thread: a destroy can wait on a
        // driver lock another thread's device sync holds.
        if let Some(me) = Work::of(inner) {
            me.backend.sweep();
        }
    }
}

impl Request {
    pub fn check(&self) -> Result<()> { self.inner.check_ownership() }

    /// Replaces the plan (packages declare it after the request opened); the cursor starts over.
    pub fn plan(&self, plan: Plan) {
        let _ = self.inner.guard(|| {
            let mut st = self.inner.lock();
            let steps = Inner::resolve(&st, &plan);
            if let Some(r) = st.table.request_mut(self.id) {
                r.steps = steps;
                r.cursor = Default::default();
            }
            st.synthetic.remove(&self.id);
            self.inner.unlock(st);
            Ok(())
        });
    }

    /// Starts a pass of `set` (`key`: the execution layer's name for its entry and shape): the
    /// plan's cursor moves to it, peers are asked when the request is short, and its first
    /// blocks are prefetched into free bytes.
    pub fn pass(&self, set: SetId, key: &str) -> Result<Pass> {
        let _ = key;
        let id = self.inner.guard(|| self.inner.pass(self.id, set.0))?;
        self.inner.torch_pass(self.id);
        Ok(Pass { inner: self.inner.clone().into(), req: self.id, set: set.0, id })
    }

    /// The request waits on a child call: until the guard drops, its step's pages are a later
    /// use (taken before the child's own).
    pub fn yielding(&self) -> Yielded {
        let _ = self.inner.guard(|| {
            let mut st = self.inner.lock();
            if let Some(r) = st.table.request_mut(self.id) {
                r.yielded = true;
            }
            self.inner.unlock(st);
            Ok(())
        });
        Yielded { wait: Some((self.inner.clone().into(), self.id)) }
    }

    pub fn stats(&self) -> Stats {
        self.inner.state.lock().stats.get(&self.id).cloned().unwrap_or_default()
    }

    /// Ends the request, returning its record.
    pub fn end(mut self) -> Stats {
        self.ended = true;
        let id = self.id;
        let stats = self.inner.guard(|| Ok(self.inner.end_request(id))).unwrap_or_default();
        self.inner.measured(crate::lease::Measured::Request { request: stats.request.clone(), working: stats.peak, compute: stats.compute, host: stats.host.unwrap_or(0) });
        stats
    }
}

impl Drop for Request {
    fn drop(&mut self) {
        if !self.ended {
            let id = self.id;
            let _ = self.inner.guard(|| Ok(self.inner.end_request(id)));
        }
    }
}

impl Pin {
    /// The caller enqueued its fill transform on the bind stream: the block's bytes are the
    /// weights once the stream's work so far completes. Other binders wait on that in-stream.
    pub fn filled(&self) {
        self.transformed.store(true, Ordering::Release);
        let _ = self.inner.guard(|| {
            let event = self.inner.backend.record(self.stream)?;
            let mut st = self.inner.lock();
            if let Some(h) = st.homes.get_mut(&self.key) {
                let old = std::mem::replace(&mut h.ready, event);
                h.ready_on = self.stream;
                h.filled = true;
                h.transforming = false;
                if old != event {
                    self.inner.backend.retire(old);
                }
            }
            st.sync(self.key);
            #[cfg(unix)]
            if self.inner.leasing() {
                self.inner.to_land.lock().push(self.key);
            }
            self.inner.unlock(st);
            self.inner.landed.notify_all();
            Ok(())
        });
    }
}

impl Drop for Pin {
    fn drop(&mut self) {
        // A DLPack deleter may hold the GIL or an allocator lock. A failed event record
        // cannot be rescued by synchronizing here: a CUDA host callback may need that lock.
        let event = match self.inner.backend.record(self.stream) {
            Ok(event) => event,
            Err(error) => {
                tracing::error!("varena: release completion is unknown; quarantining pin {}: {error}", self.id);
                self.inner.lost_backing(error);
                return;
            }
        };
        let untransformed = self.transform && !self.transformed.load(Ordering::Acquire);
        let _ = self.inner.release_tx.send(Release { key: self.key, pin: self.id, event, stream: self.stream, untransformed });
        self.inner.landed.notify_all();
    }
}

impl Pass {
    /// Ends the pass: `peak`, the most the execution layer saw it allocate, goes into the
    /// request's record (its dispatch key's peak at the request's end).
    pub fn end(self, peak: u64) {
        // The pass's torch peak is varena's own when torch allocates from it.
        let peak = peak.max(self.inner.torch_peak(self.req));
        let _ = self.inner.guard(|| self.inner.end_pass(self.req, self.id, peak));
    }
}

impl Inner {
    /// Memory whose release could not be proved: this executor is quarantined, and it leaks
    /// rather than reuse what another may still hold.
    fn lost_backing(self: &Arc<Self>, why: impl std::fmt::Display) -> Error {
        self.ownership_lost.store(true, Ordering::Release);
        self._device.retiring(true);
        self.landed.notify_all();
        std::mem::forget(self.clone());
        Error::Ownership(format!("physical release could not be proved: {why}"))
    }

    /// Runs one mechanism call (the state lock free), then applies its result under it.
    fn mechanism(self: &Arc<Self>, op: Mech) {
        if self.check_ownership().is_err() {
            std::mem::forget(op);
            return;
        }
        match op {
            Mech::Map { key, gen, va, off, span, job, decided } => self.place_page(key, gen, va, off, span, job, decided),
            #[cfg(unix)]
            Mech::Revoke => self.revoke(),
            #[cfg(unix)]
            Mech::Reclaim { region, swap } => {
                if let Err(e) = crate::region::reclaim(region, swap) {
                    tracing::warn!("varena: reclaiming region {region}: {e}");
                }
            }
            Mech::Renew { key, gen, va, off, span, old, after, job } => {
                // Its consumers finish before the unmap: this thread waits for their events.
                let released = after.iter().try_for_each(|e| self.backend.finish(*e))
                    .and_then(|()| self.backend.unmap(va, off, span));
                if let Err(error) = released {
                    tracing::warn!("varena: renewed page ownership could not be released: {error}");
                    let st = self.state.lock();
                    self.ownership_lost.store(true, Ordering::Release);
                    drop(st);
                    self.landed.notify_all();
                    std::mem::forget((self.clone(), old, after));
                    return;
                }
                if let Some(old) = old {
                    if let Err(error) = self.give_page(old) {
                        self.lost_backing(error);
                        return;
                    }
                }
                after.into_iter().for_each(|e| self.backend.retire(e));
                self.place_page(key, gen, va, off, span, job, true);
            }
            Mech::Unmap { va, off, span, phys, ready } => {
                // A page read ready at issue may still be landing: its copies end first.
                if let Err(error) = self.backend.finish(ready).and_then(|()| self.backend.unmap(va, off, span)) {
                    tracing::warn!("varena: page ownership could not be released: {error}");
                    let st = self.state.lock();
                    self.ownership_lost.store(true, Ordering::Release);
                    drop(st);
                    self.landed.notify_all();
                    std::mem::forget((self.clone(), phys, ready));
                    return;
                }
                if let Some(p) = phys {
                    if let Err(error) = self.give_page(p) {
                        self.lost_backing(error);
                        return;
                    }
                }
                self.backend.retire(ready);
            }
            Mech::Move { key, gen, va, from, off, span, phys, after, job } => self.move_slot(key, gen, va, from, off, span, phys, after, job),
            Mech::Release(va) => self.backend.release(va),
            Mech::Close => {}
        }
    }

    /// Creates and maps a page for `key` and queues its fill; a create the device cannot serve
    /// leaves the page unmapped (`decided` false: a speculative page, which goes quietly).
    /// Leased, the page is asked of the manager and placed on the agent's thread when granted:
    /// this thread never waits on it.
    #[allow(clippy::too_many_arguments)]
    fn place_page(self: &Arc<Self>, key: Key, gen: u64, va: Va, off: u64, span: u64, job: Job, decided: bool) {
        #[cfg(unix)]
        if let Some(agent) = self.lease.get().cloned() {
            let (start, region) = (Instant::now(), self.lease_region(key));
            let me = Arc::downgrade(self);
            let asked = agent.ask(
                span,
                crate::lease::Kind::Weights,
                region,
                Box::new(move |got| {
                    if let Some(inner) = me.upgrade() {
                        inner.leased_page(key, gen, va, off, span, job, decided, got, start, region.is_some());
                    }
                }),
            );
            if let Err(error) = asked {
                tracing::warn!("varena: asking the lease manager for {key:?}: {error}");
            }
            return;
        }
        let made = self.allowance.create(&*self.backend, span).and_then(|p| self.map_rw(va, off, p));
        self.placed(key, gen, va, off, span, job, decided, made.map(|p| (p, false, None)));
    }

    /// `p` mapped read-write at `va + off`; unmappable, it goes back.
    fn map_rw(self: &Arc<Self>, va: Va, off: u64, p: Physical) -> Result<Physical> {
        match self.backend.map(va, off, &p, Access::ReadWrite) {
            Ok(()) => Ok(p),
            Err(error @ Error::Ownership(_)) => Err(self.lost_backing(error)),
            Err(error) => match self.give_page(p) {
                Ok(()) => Err(error),
                Err(release) => Err(self.lost_backing(release)),
            },
        }
    }

    /// A page made (adopted: its bytes are the weights already; leased: the ask's time) or
    /// refused, for `key` if it is still that page.
    #[allow(clippy::too_many_arguments)]
    fn placed(self: &Arc<Self>, key: Key, gen: u64, va: Va, off: u64, span: u64, job: Job, decided: bool, made: Result<(Physical, bool, Option<u64>)>) {
        let speculative = !decided;
        let mut st = self.state.lock();
        if !st.homes.get(&key).is_some_and(|h| h.gen == gen) {
            drop(st);
            if let Ok((p, ..)) = made {
                self.commit_page(&p);
                let released = self.backend.unmap(va, off, span).and_then(|()| self.give_page(p));
                if let Err(error) = released {
                    self.lost_backing(error);
                }
            }
            return;
        }
        match made {
            Ok((p, true, ns)) => {
                // Adopted from the manager's retention: no copy, nothing to fill.
                if let Some(h) = st.homes.get_mut(&key) {
                    (h.phys, h.ready, h.ready_on, h.filling, h.filled, h.origin) = (Some(p), Event::DONE, COPY, false, true, Origin::Kept);
                }
                st.sync(key);
                let s = st.fills(job.req, key, span, Origin::Kept);
                s.moved.kept.bytes += span;
                s.moved.kept.ns += ns.unwrap_or(0);
                self.unlock(st);
                self.landed.notify_all();
            }
            Ok((p, false, ns)) => {
                if let Some(h) = st.homes.get_mut(&key) {
                    h.phys = Some(p);
                }
                if let Some(ns) = ns {
                    let s = st.fills(job.req, key, span, Origin::Store);
                    s.moved.leased.bytes += span;
                    s.moved.leased.ns += ns;
                }
                drop(st);
                // Handed over with the pages mapped beside it (`flush_fills`).
                let mut g = self.grouped.lock();
                if speculative { g.1.push(job) } else { g.0.push(job) }
            }
            Err(e) => {
                st.homes.remove(&key);
                st.sync(key);
                if !speculative {
                    match e {
                        // The manager could make no room: the bind takes this process's own
                        // pages (a running set streams) and asks again.
                        Error::Exhausted(_) => drop(st.short.insert(key, 0)),
                        // The device had less than the figures showed: the bind decides again,
                        // taking the page's bytes itself.
                        error @ Error::Device { code: crate::backend::OUT_OF_MEMORY, .. } => {
                            let driver_free = self.backend.free().unwrap_or(0);
                            tracing::debug!(target: "varena::memory", ?key, ?error, span, driver_free, mapped = st.table.mapped(), "physical allocation refused; the bind decides again");
                            st.short.insert(key, driver_free);
                        }
                        e => drop(st.failed.insert(key, e)),
                    }
                }
                drop(st);
                self.landed.notify_all();
            }
        }
    }

    /// A figure for the lease manager's admission and placement (a request's peaks at its end,
    /// a set's bytes at its load); nothing when not leased.
    fn measured(&self, _m: crate::lease::Measured) {
        #[cfg(unix)]
        if let Some(agent) = self.lease.get() {
            if let Err(error) = agent.measured(_m) {
                tracing::debug!("varena: a measured figure for the lease manager: {error}");
            }
        }
    }

    /// Pages come from the lease manager.
    fn leasing(&self) -> bool {
        #[cfg(unix)]
        return self.lease.get().is_some();
        #[cfg(not(unix))]
        false
    }

    /// A page's physical memory goes: a leased page back to the manager (committed now if its
    /// unmap's decision did not), any other destroyed.
    fn give_page(&self, p: Physical) -> Result<()> {
        #[cfg(unix)]
        if let Some(agent) = self.lease.get() {
            if let Some((ticket, ..)) = self.leased.lock().remove(&p.id) {
                self.declined.lock().remove(&ticket);
                return agent.give(ticket, p);
            }
        }
        self.allowance.destroy(&*self.backend, p)
    }

    /// The decision to give a leased page back, before its fence: the manager sees it
    /// progressing (Revoking), never stalled.
    fn commit_page(&self, _p: &Physical) {
        #[cfg(unix)]
        if let Some(agent) = self.lease.get() {
            if let Some((ticket, ..)) = self.leased.lock().get(&_p.id).copied() {
                self.declined.lock().remove(&ticket);
                if let Err(error) = agent.decide(ticket) {
                    tracing::warn!("varena: committing a lease's return: {error}");
                }
            }
        }
    }

    /// A leased page's bytes become `key`'s block (a streamed slot moving): no longer the
    /// weights of the region it was granted for, it is never retained again (a grant's identity
    /// never changes).
    fn moved_page(&self, _p: &Physical, _key: Key) {
        #[cfg(unix)]
        if let Some(agent) = self.lease.get() {
            let mut leased = self.leased.lock();
            if let Some((ticket, at, retainable)) = leased.get_mut(&_p.id) {
                (*at, *retainable) = (_key, false);
                if let Err(error) = agent.unland(*ticket) {
                    tracing::warn!("varena: a moved slot's lease: {error}");
                }
            }
        }
    }

    /// Whether `p` is a leased page.
    fn leased_page_of(&self, _p: Option<&Physical>) -> bool {
        #[cfg(unix)]
        return _p.is_some_and(|p| self.leased.lock().contains_key(&p.id));
        #[cfg(not(unix))]
        false
    }

    /// What a leased page holds, for the manager's retention: a declared set's fill identity and
    /// the page's index.
    #[cfg(unix)]
    fn lease_region(&self, key: Key) -> Option<crate::lease::Region> {
        let st = self.state.lock();
        let m = st.mech(key.0).ok().filter(|m| m.spec.fingerprint.declared())?;
        let digest = Sha256::digest(m.key.as_bytes());
        let fingerprint = digest.get(..8).and_then(|b| <[u8; 8]>::try_from(b).ok()).map(u64::from_le_bytes)?;
        Some(crate::lease::Region { fingerprint, region: key.1 })
    }

    /// The manager's answer to a page's ask, on the agent's thread: one chunk mapped and placed,
    /// or the refusal.
    #[cfg(unix)]
    #[allow(clippy::too_many_arguments)]
    fn leased_page(self: &Arc<Self>, key: Key, gen: u64, va: Va, off: u64, span: u64, job: Job, decided: bool, got: Result<Vec<crate::lease::Lease>>, start: Instant, region: bool) {
        let Some(agent) = self.lease.get().cloned() else { return };
        let retainable = region;
        let made = got.and_then(|leases| {
            let mut leases = leases.into_iter();
            let (Some(lease), None) = (leases.next(), leases.next()) else {
                return Err(Error::Invalid(format!("a page of {span} bytes leased as other than one chunk")));
            };
            let crate::lease::Lease { ticket, phys, adopted } = lease;
            self.leased.lock().insert(phys.id, (ticket, key, retainable));
            let p = self.map_rw(va, off, phys)?;
            agent.mapped(&[ticket])?;
            Ok((p, adopted, Some(start.elapsed().as_nanos() as u64)))
        });
        self.placed(key, gen, va, off, span, job, decided, made);
        self.flush_fills();
    }

    /// The manager asked pages back: the quiet ones are unmapped (their return committed now,
    /// given after their consumers); the busy ones are pinned in the table, so the manager looks
    /// elsewhere, and go once idle.
    #[cfg(unix)]
    fn revoke(self: &Arc<Self>) {
        let Some(agent) = self.lease.get().cloned() else { return };
        let asked: HashSet<crate::lease::Ticket> = agent.asked_back().into_iter().collect();
        if asked.is_empty() {
            return;
        }
        // torch's chunks the manager asked back, given by the arena when idle and quiet (no
        // state lock held: R10).
        self.revoke_torch(&asked);
        let keys: Vec<(crate::lease::Ticket, Key)> = self.leased.lock().values().filter(|(t, ..)| asked.contains(t)).map(|(t, k, _)| (*t, *k)).collect();
        let mut st = self.lock();
        for (ticket, key) in keys {
            let quiet = st.idle(key) && st.homes.get(&key).is_some_and(|h| !h.landing && h.phys.is_some());
            if quiet {
                let class = st.table.class(key).unwrap_or(Class::N);
                if let Err(error) = self.unmap(&mut st, key, class, None) {
                    tracing::warn!("varena: giving back a lease of {key:?}: {error}");
                }
            } else if self.declined.lock().insert(ticket) {
                let _ = agent.pin(ticket);
            }
        }
        self.unlock(st);
    }

    /// What each leased page is to its plan, for the manager's victim order: the current
    /// step's (memory.min, its distance: the nearest are its streaming slots), a later step's
    /// (its distance), or idle (its last bind on the machine-wide clock). Atomics only, under
    /// the state lock.
    #[cfg(unix)]
    fn advise(&self, st: &State, agent: &crate::lease::Agent) {
        use crate::lease::Advice;
        for (ticket, key, _) in self.leased.lock().values() {
            let at = st.table.distance(*key).map(|(step, pass, block)| u64::from(step) << 40 | u64::from(pass) << 20 | u64::from(block));
            let advice = match (st.table.class(*key), at) {
                (Some(Class::P), _) => Advice::Step { at: 0 },
                (Some(Class::W), at) => Advice::Step { at: at.unwrap_or(0) },
                (_, Some(at)) => Advice::Later { at },
                _ => Advice::Idle { last_use: st.used_ns.get(key).copied().unwrap_or(0) },
            };
            let _ = agent.advise(*ticket, advice);
        }
    }

    /// Leased pages whose fill's ready event completed: the manager hears they landed. Under
    /// the state lock.
    #[cfg(unix)]
    fn land(&self, st: &State, agent: &crate::lease::Agent) {
        let mut to_land = self.to_land.lock();
        if to_land.is_empty() {
            return;
        }
        let leased = self.leased.lock();
        to_land.retain(|key| {
            let Some(h) = st.homes.get(key).filter(|h| h.filled) else { return false };
            let Some(ticket) = h.phys.as_ref().and_then(|p| leased.get(&p.id)).filter(|(.., retainable)| *retainable).map(|(t, ..)| *t) else { return false };
            if !self.backend.query(h.ready).unwrap_or(false) {
                return true;
            }
            if let Err(error) = agent.landed(ticket) {
                tracing::warn!("varena: a lease's landed ack: {error}");
            }
            false
        });
    }

    /// Pages asked back while busy that are idle now: unpinned in the table, and the revoke
    /// runs again. Under the state lock.
    #[cfg(unix)]
    fn redecline(&self, st: &State) {
        let Some(agent) = self.lease.get() else { return };
        let idle: Vec<crate::lease::Ticket> = {
            let leased = self.leased.lock();
            let mut declined = self.declined.lock();
            let idle: Vec<_> = declined.iter().copied().filter(|t| leased.values().find(|(x, ..)| x == t).is_none_or(|(_, k, _)| st.idle(*k))).collect();
            idle.iter().for_each(|t| {
                declined.remove(t);
            });
            idle
        };
        if idle.is_empty() {
            return;
        }
        for t in idle {
            let _ = agent.unpin(t);
        }
        self.mech.post(Mech::Revoke);
    }

    /// The wait on a child ended: its step is in use again.
    fn unyield(&self, req: u64) -> Result<()> {
        let mut st = self.lock();
        if let Some(r) = st.table.request_mut(req) {
            r.yielded = false;
        }
        self.unlock(st);
        Ok(())
    }
}

/// A request waiting on a child call ([`Request::yielding`]); dropped, the wait is over.
pub struct Yielded {
    wait: Option<(Owner, u64)>,
}

impl Yielded {
    /// The wait is over (dropping it does the same, quietly).
    pub fn end(mut self) -> Result<()> {
        match self.wait.take() {
            Some((inner, req)) => inner.guard(|| inner.unyield(req)),
            None => Ok(()),
        }
    }
}

impl Drop for Yielded {
    fn drop(&mut self) {
        if let Some((inner, req)) = self.wait.take() {
            let _ = inner.guard(|| inner.unyield(req));
        }
    }
}

impl Drop for Pass {
    /// Ends only its own pass: a newer one of the same request keeps its state.
    fn drop(&mut self) {
        let mut st = self.inner.state.lock();
        if st.active.get(&(self.req, self.set)).is_some_and(|a| a.id == self.id) {
            st.active.remove(&(self.req, self.set));
        }
    }
}

/// CLOCK_MONOTONIC in nanoseconds: one clock for every process on the machine.
#[cfg(unix)]
pub(crate) fn monotonic_ns() -> u64 {
    let mut t = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: a valid clock id and a writable timespec.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut t) };
    u64::try_from(t.tv_sec).unwrap_or(0).saturating_mul(1_000_000_000).saturating_add(u64::try_from(t.tv_nsec).unwrap_or(0))
}

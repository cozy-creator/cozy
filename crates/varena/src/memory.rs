//! One executor's memory on one device: the applier of [`crate::decide`]'s actions. One
//! `parking_lot` mutex guards the table and the mechanism's state; it is held for a decision and
//! its application, never across a fill, a remote call or a caller's callback: custody and tier
//! calls made while it is held go to an outbox, sent in order once it is released. Fills run
//! on fill threads (demand misses before prefetches); pin releases arrive through a channel;
//! asks are answered from pages by a thread of its own that runs no caller code; an idle
//! process's trim and cache shrinks run on its trim thread. A panic is caught at every public
//! call and on every thread: the call fails `Internal`, the table is derived again from the
//! mappings, and the process keeps serving.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Weak};
use std::time::Instant;

use parking_lot::{Condvar, Mutex, MutexGuard};
use platform::handles::Owned;

use crate::backend::{Access, Backend, Event, Physical, Va};
use crate::custody;
use crate::decide::{decide, Action, Trigger, Tried};
use crate::ledger::{self, Facts, Grant, Room, Row, Shared};
use crate::table::{Class, CutSite, Fill, Key, Level, Page, Plan, Req, Set, SetSpec, Source, Table};
use crate::tier;
use crate::{Error, Exhausted, GpuId, Result};

/// A loaded set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SetId(pub u32);

/// A registered cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CacheId(pub u32);

/// The torch allocator's bytes, read by the adapter (`memory_allocated`, `memory_reserved`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Torch {
    pub allocated: u64,
    pub reserved: u64,
}

/// What a pass measured, by the execution layer around it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Measured {
    /// The allocator's allocated peak during the pass (absolute, including what was allocated
    /// at its start).
    pub allocated_peak: u64,
    /// The allocator's reserved less allocated, at that peak.
    pub slack: u64,
    /// Each cut site's own allocated peak, measured around its op.
    pub sites: Vec<(String, u64)>,
}

/// The answer to an out-of-memory at an op.
#[derive(Clone, Debug)]
pub enum Next {
    /// Pages were unmapped (and claimed for this request) or the watermark lowered: run the op
    /// again under the new [`Memory::torch_limit`].
    Retry,
    /// Run the op one level finer: at this level.
    Cut(u32),
    /// Move idle activations to the host, at most these bytes (the measured host headroom),
    /// report them through [`Request::parked`], then run the op again.
    Park(u64),
    Exhausted(Box<Exhausted>),
}

/// Where a bound block's bytes came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Origin {
    /// Already mapped and filled.
    Hit,
    /// A kept page imported by fingerprint: no copy.
    Kept,
    Tier,
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
    /// Kept pages imported by fingerprint (no copy; `ns` is the import).
    pub kept: Fills,
    pub tier: Fills,
    pub store: Fills,
    /// Of the above, bytes filled speculatively.
    pub speculative: u64,
}

/// One step of a request.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct StepStats {
    /// The capacity the step ran against (the device's, less foreign bytes, other rows and
    /// contexts), measured at its start.
    pub capacity: u64,
    /// Its working set: weights plus the largest pass room.
    pub w: u64,
    /// Bytes evicted during the step, by class (F, O, S, N, D, W, P).
    pub evicted: [u64; 7],
}

/// One request's record (the design's §5 bar).
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct Stats {
    pub request: String,
    pub moved: Moved,
    /// One entry per bind.
    pub trace: Vec<Ref>,
    pub steps: Vec<StepStats>,
    /// Time binds waited on fills in flight (a demand miss, or a prefetch not yet landed).
    pub stall_ns: u64,
    /// Bytes evicted for this request, by class (F, O, S, N, D, W, P).
    pub evicted: [u64; 7],
    /// Of the W evictions, those made after lowering (the only W evictions allowed).
    pub evicted_lowered: u64,
    /// Declared-use (D) bytes speculation evicted: always 0.
    pub speculative_evicted_d: u64,
    pub retries: u32,
    pub cuts: u32,
    pub parks: u32,
    /// Bytes parked, as reported.
    pub parked: u64,
    pub lowered: bool,
    /// The largest working set a step had: weights plus its passes' rooms.
    pub peak: u64,
    /// Bytes the request still held when it ended (pins, rooms, caches): 0.
    pub leftover: u64,
}

/// An op site that ran short, as `oom` names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Site<'a> {
    pub name: &'a str,
    /// A collective's site: ranks agree on cut levels at pass start ([`Pass::levels`],
    /// [`Pass::agree`]), so a finer level is proposed for the next pass and this one recovers
    /// locally.
    pub collective: bool,
}

/// One process's memory on one device.
pub struct Memory {
    inner: Arc<Inner>,
}

/// A request in flight; dropping it ends it: its room, its claims and its caches go, and the
/// leftover is measured.
pub struct Request {
    inner: Arc<Inner>,
    id: u64,
    ended: bool,
}

/// A block bound for use: mapped, and filled once `ready` completes (the binding stream already
/// waits on it). Dropping it releases the block after the binding stream's work so far.
pub struct Pin {
    pub ptr: u64,
    pub len: u64,
    pub ready: Event,
    /// The bytes were just copied: a set with a declared fill runs its transform, then
    /// [`Pin::filled`].
    pub fresh: bool,
    inner: Arc<Inner>,
    key: Key,
    id: u64,
    stream: usize,
    /// Fresh bytes of a set with a transform: refilled on release unless [`Pin::filled`].
    transform: bool,
    transformed: AtomicBool,
}

/// A pass of a set at a shape, its measured room made. Dropping it without [`Pass::end`]
/// records nothing.
pub struct Pass {
    inner: Arc<Inner>,
    req: u64,
    id: u64,
}

type Shrink = Arc<Mutex<Box<dyn FnMut(u64) -> u64 + Send>>>;
type Trim = Arc<Mutex<Box<dyn FnMut() -> u64 + Send>>>;

struct SetMech {
    spec: SetSpec,
    va: Va,
    held: Option<Arc<tier::Held>>,
    closing: bool,
}

struct Home {
    phys: Physical,
    ready: Event,
    /// Tells a fill whether its page is still the one it was queued for.
    gen: u64,
    filled: bool,
    filling: bool,
    /// Landed and not yet bound: the next binder gets `fresh`.
    fresh: bool,
    /// A binder holds it fresh and runs the set's transform.
    transforming: bool,
    exported: bool,
    origin: Origin,
    speculative: bool,
    /// The fill identity its bytes were made under ([`crate::Fingerprint::key`]); empty: its
    /// bytes are not the weights (a transform released undone), so it goes once unpinned.
    identity: String,
    /// Keep (read-only, exported) once `ready` completes: a write may still be running.
    keep_after: bool,
}

struct Active {
    id: u64,
    set: u32,
    shape: String,
    allocated: u64,
    /// The context at the pass's start: its growth is the pass's.
    context: u64,
}

struct Job {
    key: Key,
    gen: u64,
    req: Option<u64>,
}

/// Fill jobs: a demand miss goes before any prefetch.
#[derive(Default)]
struct Jobs {
    demand: VecDeque<Job>,
    ahead: VecDeque<Job>,
    closed: bool,
}

struct Release {
    key: Key,
    pin: u64,
    event: Event,
    /// Released fresh without its transform: its bytes are not the weights.
    untransformed: bool,
}

/// A custody or tier call made under the lock, sent in order once it is released.
enum Call {
    Export { fp: String, page: u32, len: u64, handle: Owned },
    Withdraw { fp: String, page: u32 },
    Disown { fp: String, page: u32 },
    Use { digest: String, distance: Option<u32> },
    Close { digest: String },
}

/// What an idle process's trim thread runs: the trim hook and cache shrinks.
#[derive(Default)]
struct Wanted {
    trim: bool,
    shrinks: Vec<(u32, u64)>,
    closed: bool,
}

#[derive(Default)]
struct State {
    table: Table,
    sets: Vec<Option<SetMech>>,
    homes: HashMap<Key, Home>,
    /// Pin → (page, request).
    pins: HashMap<u64, (Key, Option<u64>)>,
    releasing: BTreeSet<Key>,
    /// Ids and the logical clock.
    clock: u64,
    shrinks: HashMap<u32, Shrink>,
    trim: Option<Trim>,
    /// Rooms read from the ledger, by key.
    rooms: HashMap<String, Room>,
    /// Cut levels in effect per (set, shape); a collective site's are the agreed ones.
    cuts: HashMap<(u32, String), BTreeMap<String, u32>>,
    /// Finer levels proposed for collective sites, for the next pass's agreement.
    proposed: HashMap<(u32, String), BTreeMap<String, u32>>,
    collective: HashSet<(u32, String, String)>,
    stuck: BTreeSet<(u32, String, String)>,
    /// The room each cut configuration failed at.
    failed_at: HashMap<String, u64>,
    stats: HashMap<u64, Stats>,
    last: Option<Stats>,
    active: HashMap<u64, Active>,
    failed: HashMap<Key, Error>,
    /// Request → the `<package>|<shape>` its peak is recorded under.
    peaks: HashMap<u64, String>,
    /// Requests whose last step is one their plan did not declare.
    synthetic: HashSet<u64>,
    /// Bytes asked of other rows and not yet decided on: published as held, so nobody else
    /// takes them first.
    claim: u64,
    outbox: Vec<Call>,
}

impl State {
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
        let pins: Vec<u64> = self.pins.iter().filter(|(_, (k, _))| *k == key).map(|(id, _)| *id).collect();
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
        self.active.iter().filter(|(_, a)| a.set == set).map(|(r, _)| *r).max()
    }

    fn levels(&self, set: u32, shape: &str) -> BTreeMap<String, u32> {
        self.cuts.get(&(set, shape.to_string())).cloned().unwrap_or_default()
    }

    /// Bumps `site`'s level in effect; returns it.
    fn cut(&mut self, set: u32, shape: &str, site: &str) -> u32 {
        let level = self.cuts.entry((set, shape.to_string())).or_default().entry(site.to_string()).or_insert(0);
        *level += 1;
        *level
    }
}

/// `<set>|<shape>|<site=level,...>`.
fn room_key(set: &str, shape: &str, levels: &BTreeMap<String, u32>) -> String {
    let cuts: Vec<String> = levels.iter().filter(|(_, l)| **l > 0).map(|(s, l)| format!("{s}={l}")).collect();
    format!("{set}|{shape}|{}", cuts.join(","))
}

fn dims(shape: &str) -> Option<Vec<u64>> {
    shape.split('x').map(|d| d.parse().ok()).collect()
}

fn with(levels: &BTreeMap<String, u32>, site: &str, level: u32) -> BTreeMap<String, u32> {
    let mut l = levels.clone();
    l.insert(site.to_string(), level);
    l
}

struct Inner {
    backend: Box<dyn Backend>,
    shared: Arc<Shared>,
    row: Arc<Row>,
    custody: Option<custody::Client>,
    tier: Option<tier::Client>,
    state: Mutex<State>,
    landed: Condvar,
    /// Held from before the state lock is released until its outbox is sent: calls go out in
    /// the order they were made.
    calls: Mutex<()>,
    release_tx: Sender<Release>,
    releases: Mutex<Receiver<Release>>,
    /// Fill jobs and the trim thread's work: shared with the threads, which hold only a weak
    /// reference to the rest while they wait, so dropping the Memory ends them.
    jobs: Arc<(Mutex<Jobs>, Condvar)>,
    wanted: Arc<(Mutex<Wanted>, Condvar)>,
    stop: Arc<AtomicBool>,
    #[cfg(unix)]
    tally: tensorfs::io::source::IoTally,
}

fn exhausted(need: u64, site: Option<&str>, stats: Stats) -> Box<Exhausted> {
    Box::new(Exhausted { need, site: site.map(str::to_string), stats })
}

impl Memory {
    /// Opens `gpu`'s table through `grant` (TensorD's), or by name without one (standalone),
    /// and claims this process's row on it.
    pub fn open(gpu: GpuId, grant: Option<Grant>, backend: Box<dyn Backend>) -> Result<Memory> {
        crate::caught(|| {
            if !crate::layout::REGION_ALIGN.is_multiple_of(backend.granularity().max(1)) {
                return Err(Error::Invalid(format!("granularity {} does not divide a region", backend.granularity())));
            }
            let grant = match grant {
                Some(g) => g,
                None => Grant::open(&gpu, None)?,
            };
            let capacity = gpu.memory.min(backend.capacity());
            let shared = Arc::new(Shared::adopt(grant.table, &gpu, capacity, Some(grant.events))?);
            shared.reap(None)?;
            let row = Arc::new(shared.claim(std::process::id())?);
            let listener = row.listen()?;
            let custody = custody::Client::open(&shared).map_err(|e| tracing::warn!("varena: no custody: {e}")).ok();
            let tier = tier::Client::open().map_err(|e| tracing::warn!("varena: no RAM tier: {e}")).ok();
            let (release_tx, releases) = channel();
            let table = Table { capacity: shared.capacity()?, context: backend.process_used()?, ..Table::default() };
            let inner = Arc::new(Inner {
                backend,
                shared,
                row: row.clone(),
                custody,
                tier,
                state: Mutex::new(State { table, ..State::default() }),
                landed: Condvar::new(),
                calls: Mutex::new(()),
                release_tx,
                releases: Mutex::new(releases),
                jobs: Arc::new((Mutex::new(Jobs::default()), Condvar::new())),
                wanted: Arc::new((Mutex::new(Wanted::default()), Condvar::new())),
                stop: Arc::new(AtomicBool::new(false)),
                #[cfg(unix)]
                tally: Default::default(),
            });
            // One fill thread per copy stream TensorFS runs per device: a fill waits on its
            // streams, so this many keep them busy.
            for _ in 0..tensorfs::io::ReadOpts::default().streams.max(1) {
                let (weak, jobs) = (Arc::downgrade(&inner), inner.jobs.clone());
                std::thread::Builder::new().name("varena-fill".into()).spawn(move || fill_loop(&jobs, &weak)).map_err(|e| Error::io("spawn fill", e))?;
            }
            // The row is the Memory's alone: it goes (publishing zeros) the moment the Memory does.
            let (weak, stop, row) = (Arc::downgrade(&inner), inner.stop.clone(), Arc::downgrade(&row));
            std::thread::Builder::new().name("varena-ask".into()).spawn(move || ask_loop(listener, &row, &weak, &stop)).map_err(|e| Error::io("spawn ask", e))?;
            let (weak, wanted) = (Arc::downgrade(&inner), inner.wanted.clone());
            std::thread::Builder::new().name("varena-trim".into()).spawn(move || trim_loop(&wanted, &weak)).map_err(|e| Error::io("spawn trim", e))?;
            let st = inner.lock();
            inner.publish(&st);
            inner.unlock(st);
            Ok(Memory { inner })
        })
    }

    /// Registers a set and brings it to `level`; a set loaded again is re-levelled in place. On
    /// every start, first or restart, its pages kept under its fingerprint are imported, and
    /// the rest fill from the RAM tier, else the store.
    pub fn load(&self, set: SetSpec, level: Level) -> Result<SetId> {
        self.inner.guard(|| self.inner.load(set, level))
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

    /// Registers bytes the caller holds (torch tensors) that `shrink(want) -> freed` gives back.
    /// A cache whose `owner` is a request id ends with that request.
    pub fn cache(&self, owner: &str, shrink: Box<dyn FnMut(u64) -> u64 + Send>) -> CacheId {
        let shrink = Arc::new(Mutex::new(shrink));
        self.inner
            .guard(|| {
                let mut st = self.inner.lock();
                let id = st.tick() as u32;
                let last_use = st.clock;
                st.table.caches.push(crate::table::Cache { id, owner: owner.to_string(), bytes: 0, last_use });
                st.shrinks.insert(id, shrink);
                self.inner.unlock(st);
                Ok(CacheId(id))
            })
            .unwrap_or(CacheId(0))
    }

    /// The cache now holds `bytes`.
    pub fn cached(&self, cache: CacheId, bytes: u64) {
        let _ = self.inner.guard(|| {
            let mut st = self.inner.lock();
            let now = st.tick();
            if let Some(c) = st.table.caches.iter_mut().find(|c| c.id == cache.0) {
                c.bytes = bytes;
                c.last_use = now;
            }
            self.inner.publish(&st);
            self.inner.unlock(st);
            Ok(())
        });
    }

    pub fn uncache(&self, cache: CacheId) {
        let _ = self.inner.guard(|| {
            let mut st = self.inner.lock();
            st.table.caches.retain(|c| c.id != cache.0);
            st.shrinks.remove(&cache.0);
            self.inner.publish(&st);
            self.inner.unlock(st);
            Ok(())
        });
    }

    /// Registers the allocator's trim (torch's `empty_cache`), returning the bytes it freed: the
    /// allocator's slack is free (F), and a demand trims before it evicts anything.
    pub fn trim_with(&self, trim: Box<dyn FnMut() -> u64 + Send>) {
        let trim = Arc::new(Mutex::new(trim));
        let _ = self.inner.guard(|| {
            let mut st = self.inner.lock();
            st.trim = Some(trim);
            self.inner.unlock(st);
            Ok(())
        });
    }

    /// Sets the set's current fill identity (a no-op when unchanged). Pages filled under another
    /// one are unmapped (a pinned one when it is released), so each next bind fills again, from
    /// kept pages of the new identity, the tier or the store, and comes back `fresh`.
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
        let rid = self.inner.guard(|| self.inner.open_request(id, &plan))?;
        Ok(Request { inner: self.inner.clone(), id: rid, ended: false })
    }

    /// Maps and fills block `block` of `set`; `stream` (a `CUstream`, 0 the default) waits on
    /// the fill in-stream, and the release is ordered after its work.
    pub fn bind(&self, set: SetId, block: u32, stream: usize) -> Result<Pin> {
        self.inner.guard(|| self.inner.bind(set.0, block, stream))
    }

    /// Bytes torch's allocator may hold now: capacity − context − mapped − other rows − foreign
    /// bytes. The adapter applies it at every pass start and after every `oom` answer.
    pub fn torch_limit(&self) -> u64 {
        self.inner
            .guard(|| {
                let mut st = self.inner.lock();
                self.inner.refresh(&mut st)?;
                let limit = st.table.torch_limit();
                self.inner.unlock(st);
                Ok(limit)
            })
            .unwrap_or(0)
    }

    /// The record of the request that ended last.
    pub fn stats(&self) -> Option<Stats> {
        let st = self.inner.state.lock();
        st.last.clone()
    }
}

impl Inner {
    /// `f`, a panic in it caught: the call fails `Internal` and the table is derived again.
    fn guard<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        let out = crate::caught(f);
        if matches!(&out, Err(Error::Internal(m)) if m.starts_with("panic")) {
            self.recover();
        }
        out
    }

    /// After a panic: the table's pages derived again from the homes (each recorded in the same
    /// statement as its mapping, so they are the mappings), each open request's episode afresh.
    fn recover(&self) {
        let mut st = self.state.lock();
        let keys: Vec<Key> = st.table.pages().map(|(k, _)| k).collect();
        for key in keys {
            st.sync(key);
        }
        for r in &mut st.table.requests {
            r.episode = Default::default();
            r.speculated.clear();
        }
        st.claim = 0;
        self.publish(&st);
        self.unlock(st);
        self.landed.notify_all();
    }

    /// The state, with pin releases applied and completed events acted on.
    fn lock(&self) -> MutexGuard<'_, State> {
        let mut st = self.state.lock();
        self.drain(&mut st);
        st
    }

    /// Releases the state, then sends the calls it made, in order.
    fn unlock(&self, mut st: MutexGuard<'_, State>) {
        let calls = std::mem::take(&mut st.outbox);
        if calls.is_empty() {
            return;
        }
        let order = self.calls.lock();
        drop(st);
        self.send(calls);
        drop(order);
    }

    /// Sends the outbox while the state stays locked across (a waiter about to sleep).
    fn flush(&self, st: &mut MutexGuard<'_, State>) {
        let calls = std::mem::take(&mut st.outbox);
        if calls.is_empty() {
            return;
        }
        let order = self.calls.lock();
        MutexGuard::unlocked(st, || self.send(calls));
        drop(order);
    }

    fn send(&self, calls: Vec<Call>) {
        for call in calls {
            let _ = match call {
                Call::Export { fp, page, len, handle } => self.custody.as_ref().map(|c| c.export(&fp, page, len, handle)),
                Call::Withdraw { fp, page } => self.custody.as_ref().map(|c| c.withdraw(&fp, page)),
                Call::Disown { fp, page } => self.custody.as_ref().map(|c| c.disown(&fp, page)),
                Call::Use { digest, distance } => self.tier.as_ref().map(|t| t.use_layout(&digest, distance)),
                Call::Close { digest } => self.tier.as_ref().map(|t| t.close(&digest)),
            };
        }
    }

    /// Waits for a fill or release, the outbox sent first.
    fn wait(&self, st: &mut MutexGuard<'_, State>) {
        if !st.outbox.is_empty() {
            // Sent, and the caller looks again before it sleeps.
            self.flush(st);
            return;
        }
        self.landed.wait(st);
        self.drain(st);
    }

    fn drain(&self, st: &mut State) {
        let released: Vec<Release> = self.releases.lock().try_iter().collect();
        let mut wake = false;
        for r in released {
            st.pins.remove(&r.pin);
            if r.event != Event::DONE {
                if let Some(p) = st.table.page_mut(r.key) {
                    p.releasing.push(r.event.0);
                }
                st.releasing.insert(r.key);
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
        let pending: Vec<Key> = st.releasing.iter().copied().collect();
        for key in pending {
            let ready = st.homes.get(&key).map(|h| h.ready.0);
            let Some(p) = st.table.page_mut(key) else { continue };
            let events = std::mem::take(&mut p.releasing);
            let (left, done): (Vec<u64>, Vec<u64>) = events.into_iter().partition(|e| !self.backend.query(Event(*e)).unwrap_or(true));
            done.into_iter().filter(|e| Some(*e) != ready).for_each(|e| self.backend.retire(Event(e)));
            let empty = left.is_empty();
            if let Some(p) = st.table.page_mut(key) {
                p.releasing = left;
            }
            if empty {
                st.releasing.remove(&key);
            }
            // A transform or fill whose event completed: now read-only and kept.
            if st.homes.get(&key).is_some_and(|h| h.keep_after && self.backend.query(h.ready).unwrap_or(false)) {
                if let Some(h) = st.homes.get_mut(&key) {
                    h.keep_after = false;
                }
                self.keep(st, key);
            }
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
            .filter(|(k, h)| !h.filling && st.mech(k.0).is_ok_and(|m| m.spec.fingerprint.key() != h.identity))
            .filter(|(k, _)| st.table.page(**k).is_some_and(|p| p.pins.is_empty() && p.releasing.is_empty()))
            .map(|(k, _)| *k)
            .collect();
        for key in stale {
            let _ = self.unmap(st, key, Class::N, None, true);
        }
    }

    /// The set's fill identity becomes `fill`; pages of another identity go as they unpin.
    fn refill(&self, st: &mut State, set: u32, fill: Fill) -> Result<()> {
        let m = st.sets.get_mut(set as usize).and_then(Option::as_mut).ok_or_else(|| Error::Invalid(format!("no set {set}")))?;
        if m.spec.fingerprint.fill == fill {
            return Ok(());
        }
        m.spec.fingerprint.fill = fill;
        let declared = m.spec.fingerprint.declared();
        if let Some(Some(s)) = st.table.sets.get_mut(set as usize) {
            s.declared = declared;
        }
        self.drop_stale(st);
        self.publish(st);
        Ok(())
    }

    /// Unmaps the unpinned pages of closing sets; a set goes once it has none. Its slot stays
    /// empty: open plans name it by index.
    fn close_sets(&self, st: &mut State) {
        let closing: Vec<u32> = st.sets.iter().enumerate().filter(|(_, m)| m.as_ref().is_some_and(|m| m.closing)).map(|(i, _)| i as u32).collect();
        for set in closing {
            let keys: Vec<Key> = st.table.pages().filter(|(k, p)| k.0 == set && p.mapped && p.pins.is_empty() && p.releasing.is_empty() && !p.filling).map(|(k, _)| k).collect();
            for key in keys {
                let _ = self.unmap(st, key, Class::N, None, true);
            }
            if st.table.pages().any(|(k, p)| k.0 == set && p.mapped) {
                continue;
            }
            if let Some(m) = st.sets.get_mut(set as usize).and_then(Option::take) {
                self.backend.release(m.va);
                if let Some(held) = &m.held {
                    held.unpin(|p| self.backend.unregister_host(p));
                }
                st.outbox.push(Call::Close { digest: m.spec.layout.digest.clone() });
            }
            if let Some(slot) = st.table.sets.get_mut(set as usize) {
                *slot = None;
            }
        }
    }

    /// The other rows, orphans, capacity, mode and foreign bytes, read now.
    fn refresh(&self, st: &mut State) -> Result<()> {
        let mut rows = self.shared.rows()?;
        if rows.iter().any(|r| !r.live) && self.shared.reap(None)? {
            rows = self.shared.rows()?;
        }
        let (others, orphans) = ledger::remotes(&rows, self.row.at());
        st.table.others = others;
        st.table.orphans = orphans;
        st.table.capacity = self.shared.capacity()?;
        st.table.shared = self.shared.shared_mode()?;
        let others: u64 = st.table.others.iter().map(crate::table::Remote::held).sum();
        let used = self.backend.capacity().saturating_sub(self.backend.free()?);
        st.table.foreign = used.saturating_sub(others + orphans + st.table.held());
        self.shared.set_foreign(st.table.foreign)?;
        Ok(())
    }

    fn publish(&self, st: &State) {
        let t = &st.table;
        let remaining: Vec<Option<u64>> = t.requests.iter().map(|r| t.remaining(r)).collect();
        let idle = t.requests.is_empty();
        self.row.publish(&Facts {
            context: t.context,
            torch: t.torch() + st.claim,
            caches: if idle { t.slack() + t.caches.iter().map(|c| c.bytes).sum::<u64>() } else { 0 },
            classes: t.classes(),
            remaining: remaining.iter().flatten().sum(),
            unmeasured: remaining.iter().filter(|r| r.is_none()).count() as u64,
            running: t.requests.len() as u64,
            outside: t.outside(),
        });
    }

    // ---- the mechanism

    /// Unmaps a page; `withdraw` drops its kept copy too (an eviction or unload; a process that
    /// ends leaves its kept pages as orphans).
    fn unmap(&self, st: &mut State, key: Key, class: Class, req: Option<u64>, withdraw: bool) -> Result<()> {
        let Some(home) = st.homes.remove(&key) else { return Ok(()) };
        let (va, off, span) = st.place(key)?;
        if home.exported && withdraw {
            st.outbox.push(Call::Withdraw { fp: home.identity.clone(), page: key.1 });
        }
        let unmapped = self.backend.unmap(va, off, span);
        self.backend.destroy(home.phys);
        self.backend.retire(home.ready);
        st.sync(key);
        if let Some(s) = st.stats(req) {
            if let Some(e) = s.evicted.get_mut(class.index()) {
                *e += span;
            }
            if class == Class::W {
                s.evicted_lowered += span;
            }
        }
        if let Some(e) = req.and_then(|r| st.step_stats(r)).and_then(|s| s.evicted.get_mut(class.index())) {
            *e += span;
        }
        unmapped
    }

    /// Imports the kept pages of `keys` custody holds under their sets' fingerprints: no copy,
    /// and orphans are adopted before any demand could release them. The calls run unlocked;
    /// an import nobody needs any more goes back.
    fn adopt(&self, keys: Vec<Key>, req: Option<u64>) -> Result<u64> {
        let Some(c) = &self.custody else { return Ok(0) };
        let wanted: Vec<(Key, String)> = {
            let st = self.lock();
            let w = keys
                .into_iter()
                .filter(|k| st.table.page(*k).is_some_and(|p| !p.mapped))
                .filter_map(|k| st.mech(k.0).ok().filter(|m| m.spec.fingerprint.declared()).map(|m| (k, m.spec.fingerprint.key())))
                .collect();
            self.unlock(st);
            w
        };
        let mut adopted = 0;
        for (key, identity) in wanted {
            let start = Instant::now();
            let Some(handle) = c.import(&identity, key.1) else { continue };
            let mut st = self.lock();
            let current = st.mech(key.0).ok().map(|m| m.spec.fingerprint.key());
            if st.homes.contains_key(&key) || current.as_ref() != Some(&identity) {
                st.outbox.push(Call::Disown { fp: identity, page: key.1 });
                self.unlock(st);
                continue;
            }
            let (va, off, span) = st.place(key)?;
            let phys = self.backend.import(handle, span)?;
            if let Err(e) = self.backend.map(va, off, &phys, Access::Read) {
                self.backend.destroy(phys);
                st.outbox.push(Call::Disown { fp: identity, page: key.1 });
                self.unlock(st);
                return Err(e);
            }
            let gen = st.tick();
            st.homes.insert(key, Home { phys, ready: Event::DONE, gen, filled: true, filling: false, fresh: false, transforming: false, exported: true, origin: Origin::Kept, speculative: false, identity, keep_after: false });
            st.sync(key);
            if let Some(s) = st.stats(req) {
                s.moved.kept.bytes += span;
                s.moved.kept.ns += start.elapsed().as_nanos() as u64;
            }
            adopted += span;
            self.publish(&st);
            self.unlock(st);
        }
        Ok(adopted)
    }

    /// Maps a page read-write and queues its fill. A speculative map that finds no device
    /// memory stops softly (false).
    fn map(&self, st: &mut State, key: Key, req: Option<u64>, speculative: bool) -> Result<bool> {
        if st.homes.contains_key(&key) {
            return Ok(true);
        }
        let (va, off, span) = st.place(key)?;
        let phys = match self.backend.create(span) {
            Ok(p) => p,
            Err(_) if speculative => return Ok(false),
            Err(e) => return Err(e),
        };
        if let Err(e) = self.backend.map(va, off, &phys, Access::ReadWrite) {
            self.backend.destroy(phys);
            return if speculative { Ok(false) } else { Err(e) };
        }
        let gen = st.tick();
        let identity = st.mech(key.0)?.spec.fingerprint.key();
        st.homes.insert(key, Home { phys, ready: Event::DONE, gen, filled: false, filling: true, fresh: false, transforming: false, exported: false, origin: Origin::Store, speculative, identity, keep_after: false });
        st.sync(key);
        let (lock, queued) = &*self.jobs;
        let mut jobs = lock.lock();
        if jobs.closed {
            return Err(Error::Closed("the fill threads".into()));
        }
        let job = Job { key, gen, req };
        if speculative { jobs.ahead.push_back(job) } else { jobs.demand.push_back(job) }
        drop(jobs);
        queued.notify_one();
        Ok(true)
    }

    /// Exports a filled page of a declared set into custody, read-only, only once its ready
    /// event completed: a write still running must not meet a read-only page, and another
    /// process must not import a half-made one.
    fn keep(&self, st: &mut State, key: Key) {
        if self.custody.is_none() {
            return;
        }
        let Ok((va, off, span)) = st.place(key) else { return };
        let Ok(m) = st.mech(key.0) else { return };
        let fp = m.spec.fingerprint.clone();
        let Some(home) = st.homes.get_mut(&key) else { return };
        if !fp.declared() || !home.filled || home.exported || home.keep_after || home.identity != fp.key() {
            return;
        }
        if !self.backend.query(home.ready).unwrap_or(false) {
            // Pinned (P) until the event completes; the next drain keeps it.
            home.keep_after = true;
            let ready = home.ready.0;
            if let Some(p) = st.table.page_mut(key) {
                p.releasing.push(ready);
            }
            st.releasing.insert(key);
            return;
        }
        match self.backend.protect(va, off, span, Access::Read).and_then(|_| self.backend.export(&home.phys)) {
            Ok(handle) => {
                home.exported = true;
                let fp = home.identity.clone();
                st.outbox.push(Call::Export { fp, page: key.1, len: span, handle });
            }
            Err(e) => tracing::warn!("varena: keeping {key:?} failed: {e}"),
        }
    }

    /// A fill thread's job: tier, else store (through the tier when it has room), then the
    /// page's ready event.
    fn fill(&self, job: &Job) {
        let prepared = {
            let st = self.state.lock();
            match (st.homes.get(&job.key), st.mech(job.key.0), st.place(job.key)) {
                (Some(h), Ok(m), Ok(place)) if h.gen == job.gen => Some((m.spec.clone(), m.held.clone(), place)),
                _ => None,
            }
        };
        let Some((spec, held, (va, off, span))) = prepared else { return };
        let start = Instant::now();
        let result = crate::caught(|| {
            #[cfg(feature = "faults")]
            crate::faults::hit(crate::faults::Point::Fill);
            let dst = self.backend.target(va, off, span);
            let region = spec.layout.region(job.key.1)?;
            let pin = |held: &tier::Held| held.pin(job.key.1, |p, l| self.backend.register_host(p, l), |p| self.backend.unregister_host(p));
            if let Some(held) = &held {
                pin(held)?;
                if held.read(job.key.1, dst)? {
                    return Ok((Origin::Tier, self.backend.fence()?));
                }
                // Staged through the tier when the host has the room (the tier's decision).
                if self.tier.as_ref().is_some_and(|t| t.room(span).unwrap_or(false)) {
                    held.fill(job.key.1, |host| self.store(&spec.source, &region.items, region.offset, host))?;
                    pin(held)?;
                    if held.read(job.key.1, dst)? {
                        return Ok((Origin::Store, self.backend.fence()?));
                    }
                }
            }
            self.store(&spec.source, &region.items, region.offset, dst)?;
            Ok((Origin::Store, self.backend.fence()?))
        });
        let ns = start.elapsed().as_nanos() as u64;
        let mut st = self.state.lock();
        if st.homes.get(&job.key).is_none_or(|h| h.gen != job.gen) {
            return;
        }
        match result {
            Ok((origin, ready)) => {
                let stored = spec.fingerprint.fill == Fill::Stored;
                let speculative = st.homes.get(&job.key).is_some_and(|h| h.speculative);
                if let Some(h) = st.homes.get_mut(&job.key) {
                    h.ready = ready;
                    h.filling = false;
                    h.filled = stored;
                    h.fresh = true;
                    h.origin = origin;
                }
                st.sync(job.key);
                if let Some(s) = st.stats(job.req) {
                    let f = if origin == Origin::Tier { &mut s.moved.tier } else { &mut s.moved.store };
                    f.bytes += span;
                    f.ns += ns;
                    if speculative {
                        s.moved.speculative += span;
                    }
                }
                if stored {
                    self.keep(&mut st, job.key);
                }
            }
            Err(e) => {
                tracing::warn!("varena: filling {:?} failed: {e}", job.key);
                // A sticky error ends the process, and the GPU's kept pages go with it.
                if e.is_sticky() {
                    if let Some(c) = &self.custody {
                        let _ = c.fault();
                    }
                }
                if let Some(h) = st.homes.get_mut(&job.key) {
                    h.filling = false;
                }
                let speculative = st.homes.get(&job.key).is_some_and(|h| h.speculative);
                let _ = self.unmap(&mut st, job.key, Class::P, None, true);
                if !speculative {
                    st.failed.insert(job.key, e);
                }
            }
        }
        self.publish(&st);
        self.unlock(st);
        self.landed.notify_all();
    }

    /// A fill thread's job failed outside its copy (a panic): its page goes, its binder fails.
    fn fail(&self, job: &Job, e: Error) {
        let mut st = self.state.lock();
        if st.homes.get(&job.key).is_some_and(|h| h.gen == job.gen) {
            if let Some(h) = st.homes.get_mut(&job.key) {
                h.filling = false;
            }
            let _ = self.unmap(&mut st, job.key, Class::P, None, true);
            st.failed.insert(job.key, e);
        }
        self.unlock(st);
        self.landed.notify_all();
    }

    #[cfg(unix)]
    fn store(&self, source: &Source, items: &[crate::layout::Item], origin: u64, dst: tensorfs::io::Dst) -> Result<()> {
        let threads = tensorfs::io::ReadOpts::default().threads;
        // SAFETY: `dst` is a mapped page (or a locked tier region) only this fill writes.
        unsafe { source.read_items(items, origin, dst, threads, tensorfs::io::source::Mode::Cache, &self.tally) }?;
        Ok(())
    }

    #[cfg(windows)]
    fn store(&self, source: &Source, _: &[crate::layout::Item], _: u64, _: tensorfs::io::Dst) -> Result<()> {
        match *source {}
    }

    // ---- demands

    /// Decides and applies until the trigger is met or ends. Remote rungs (the trim hook,
    /// orphans, asks) run with the lock released; bytes asked of others are claimed in this
    /// row meanwhile, so nobody takes them first; then the procedure decides again with them
    /// tried and what every round freed counted.
    fn demand(&self, mut trigger: Trigger, req: Option<u64>) -> Result<Vec<Action>> {
        loop {
            let mut st = self.lock();
            self.refresh(&mut st)?;
            if matches!(trigger, Trigger::Oom { .. }) {
                st.table.headroom = u64::try_from(platform::contain::host_memory::read().available).unwrap_or(0);
            }
            let actions = decide(&st.table, &trigger);
            let (mut remote, mut shrinks, mut freed) = (vec![], vec![], 0);
            for a in &actions {
                freed += self.apply(&mut st, a, req, &mut remote, &mut shrinks)?;
            }
            st.claim = remote.iter().map(|a| if let Action::Ask { bytes, .. } = a { *bytes } else { 0 }).sum();
            // A pass whose cut changed its room decides again at the new levels.
            if let (Trigger::Pass { room, unmeasured, .. }, Some(r)) = (&mut trigger, req) {
                if actions.iter().any(|a| matches!(a, Action::Cut { .. })) && !remote.is_empty() {
                    (*room, *unmeasured) = self.pass_room(&mut st, r)?;
                }
            }
            self.publish(&st);
            self.unlock(st);
            self.shrink(shrinks);
            if remote.is_empty() {
                return Ok(actions);
            }
            let tried = match &mut trigger {
                Trigger::Bind { tried, .. } | Trigger::Pass { tried, .. } | Trigger::Oom { tried, .. } | Trigger::Speculate { tried, .. } => tried,
                Trigger::Ask { .. } | Trigger::Load { .. } => return Ok(actions),
            };
            tried.freed += freed;
            for a in &actions {
                if let Action::Cut { site, .. } | Action::Propose { site, .. } = a {
                    tried.cut.insert(site.clone());
                }
            }
            for a in remote {
                match a {
                    Action::Trim(_) => {
                        tried.trimmed = true;
                        tried.freed += self.trim();
                    }
                    Action::ReleaseOrphans(bytes) => {
                        tried.orphans = true;
                        tried.freed += self.custody.as_ref().map_or(0, |c| c.release(bytes));
                    }
                    Action::Ask { row, ceiling, bytes } => {
                        tried.asked.insert((row, ceiling));
                        tried.freed += self.shared.ask(row, ceiling, bytes)?;
                    }
                    _ => {}
                }
            }
        }
    }

    /// Applies one action; returns the bytes it unmapped.
    fn apply(&self, st: &mut State, a: &Action, req: Option<u64>, remote: &mut Vec<Action>, shrinks: &mut Vec<(u32, u64, Shrink)>) -> Result<u64> {
        match a {
            Action::Unmap { page, class } => {
                let bytes = st.table.page(*page).map_or(0, |p| p.bytes);
                self.unmap(st, *page, *class, req, true)?;
                return Ok(bytes);
            }
            Action::Trim(_) | Action::ReleaseOrphans(_) | Action::Ask { .. } => remote.push(a.clone()),
            Action::Shrink { cache, bytes } => {
                if let Some(s) = st.shrinks.get(cache) {
                    shrinks.push((*cache, *bytes, s.clone()));
                }
            }
            Action::Starve(r) => {
                if let Some(r) = st.table.request_mut(*r) {
                    r.starved = true;
                }
            }
            Action::Map(k) => {
                self.map(st, *k, req, false)?;
            }
            Action::Prefetch(k) => {
                let step = req.and_then(|r| st.table.request(r)).and_then(|r| r.distance(*k).map(|d| r.cursor.step + d.0));
                if let Some(r) = req.and_then(|r| st.table.request_mut(r)) {
                    r.speculated.insert(*k);
                }
                if self.map(st, *k, req, true)? {
                    if let (Some(r), Some(step), Some(p)) = (req, step, st.table.page_mut(*k)) {
                        p.speculative = Some((r, step));
                    }
                }
            }
            Action::Claim { req, bytes } => {
                if let Some(r) = st.table.request_mut(*req) {
                    r.claimed += bytes;
                    r.episode.claimed_free = true;
                }
            }
            Action::Reserve { req, room, measured } => {
                if let Some(r) = st.table.request_mut(*req) {
                    r.room = *room;
                    r.measured = *measured;
                }
            }
            Action::Cut { req, site, failed_at } => {
                if let Some((set, shape)) = st.active.get(req).map(|a| (a.set, a.shape.clone())) {
                    let name = st.mech(set)?.spec.name.clone();
                    let key = room_key(&name, &shape, &st.levels(set, &shape));
                    st.failed_at.insert(key, *failed_at);
                    st.cut(set, &shape, site);
                }
                if let Some(s) = st.stats(Some(*req)) {
                    s.cuts += 1;
                }
            }
            Action::Relax { req, site } => {
                if let Some((set, shape)) = st.active.get(req).map(|a| (a.set, a.shape.clone())) {
                    if let Some(l) = st.cuts.get_mut(&(set, shape)).and_then(|c| c.get_mut(site)) {
                        *l = l.saturating_sub(1);
                    }
                }
            }
            Action::Propose { req, site } => {
                if let Some((set, shape)) = st.active.get(req).map(|a| (a.set, a.shape.clone())) {
                    let now = st.levels(set, &shape).get(site).copied().unwrap_or(0);
                    let p = st.proposed.entry((set, shape)).or_default().entry(site.clone()).or_insert(0);
                    *p = (*p).max(now + 1);
                }
            }
            Action::Lower(r) => {
                if let Some(rq) = st.table.request_mut(*r) {
                    rq.lowered = true;
                }
                if let Some(s) = st.stats(Some(*r)) {
                    s.lowered = true;
                }
            }
            Action::Park(_) | Action::Exhausted { .. } => {}
        }
        Ok(0)
    }

    /// Runs caches' shrink callbacks, the lock released (they are the caller's code).
    fn shrink(&self, shrinks: Vec<(u32, u64, Shrink)>) {
        for (id, bytes, f) in shrinks {
            let freed = crate::caught(|| Ok((f.lock())(bytes))).unwrap_or(0);
            let mut st = self.state.lock();
            if let Some(c) = st.table.caches.iter_mut().find(|c| c.id == id) {
                c.bytes = c.bytes.saturating_sub(freed);
            }
        }
    }

    /// Runs the trim hook, the lock released (it is the caller's code); the slack it freed
    /// leaves the reserved figure.
    fn trim(&self) -> u64 {
        let Some(hook) = self.state.lock().trim.clone() else { return 0 };
        let freed = crate::caught(|| Ok((hook.lock())())).unwrap_or(0);
        let mut st = self.state.lock();
        st.table.reserved = st.table.reserved.saturating_sub(freed);
        self.publish(&st);
        freed
    }

    /// Fetches ahead for `req` into the surplus.
    fn speculate(&self, req: Option<u64>) {
        if let Some(r) = req {
            let _ = self.demand(Trigger::Speculate { req: r, tried: Tried::default() }, Some(r));
        }
    }

    // ---- the calls

    fn load(&self, spec: SetSpec, level: Level) -> Result<SetId> {
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
                    st.sets.push(Some(SetMech { spec, va, held: None, closing: false }));
                    st.table.sets.push(Some(set));
                    (i, false)
                }
            };
            self.unlock(st);
            out
        };
        if !again {
            self.open_tier(id)?;
        }
        // One load path: whatever custody keeps for this fingerprint comes first, at any level.
        let keys: Vec<Key> = {
            let st = self.lock();
            let k = st.table.pages().filter(|(k, p)| k.0 == id && !p.mapped).map(|(k, _)| k).collect();
            self.unlock(st);
            k
        };
        self.adopt(keys, None)?;
        match level {
            Level::Ram => {
                if again {
                    // Down to RAM: its unpinned device pages go.
                    let mut st = self.lock();
                    let keys: Vec<Key> = st.table.pages().filter(|(k, p)| k.0 == id && p.mapped && p.pins.is_empty() && p.releasing.is_empty() && !p.filling).map(|(k, _)| k).collect();
                    for key in keys {
                        self.unmap(&mut st, key, Class::N, None, true)?;
                    }
                    self.publish(&st);
                    self.unlock(st);
                }
                self.load_ram(id)?
            }
            Level::Gpu => {
                self.demand(Trigger::Load { set: id }, None)?;
                let mut st = self.lock();
                while st.table.pages().any(|(k, p)| k.0 == id && p.filling) {
                    self.wait(&mut st);
                }
                self.unlock(st);
            }
        }
        Ok(SetId(id))
    }

    /// The set's layout in the RAM tier, opened unlocked (a call to the tier).
    fn open_tier(&self, set: u32) -> Result<()> {
        let Some(tier) = &self.tier else { return Ok(()) };
        let layout = {
            let st = self.lock();
            let l = st.mech(set)?.spec.layout.clone();
            self.unlock(st);
            l
        };
        let regions: Vec<(u64, u64)> = layout.regions.iter().map(|r| (r.offset, r.span)).collect();
        let held = tier.open_layout(&layout.digest, regions).map_err(|e| tracing::warn!("varena: tier: {e}")).ok().map(Arc::new);
        let mut st = self.lock();
        if let Some(Some(m)) = st.sets.get_mut(set as usize) {
            m.held = held;
        }
        self.unlock(st);
        Ok(())
    }

    /// Fills the set's layout in the RAM tier, region by region, as host headroom allows.
    fn load_ram(&self, set: u32) -> Result<()> {
        let (spec, held) = {
            let st = self.lock();
            let m = st.mech(set)?;
            let out = (m.spec.clone(), m.held.clone());
            self.unlock(st);
            out
        };
        let (Some(held), Some(tier)) = (held, &self.tier) else { return Ok(()) };
        for (r, region) in spec.layout.regions.iter().enumerate() {
            if !tier.room(region.span)? {
                break;
            }
            held.fill(r as u32, |host| self.store(&spec.source, &region.items, region.offset, host))?;
        }
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn resolve(&self, st: &State, plan: &Plan) -> (Vec<(Vec<u32>, u32)>, Vec<Option<u64>>) {
        let index = |name: &String| st.sets.iter().position(|m| m.as_ref().is_some_and(|m| &m.spec.name == name && !m.closing)).map(|i| i as u32);
        let steps: Vec<(Vec<u32>, u32)> = plan.steps.iter().map(|s| (s.sets.iter().filter_map(index).collect(), s.passes.max(1))).collect();
        let rooms = steps.iter().map(|(sets, _)| self.largest_room(st, sets)).collect();
        (steps, rooms)
    }

    /// The largest room measured for any of these sets, at any shape; `None` if one has none.
    fn largest_room(&self, st: &State, sets: &[u32]) -> Option<u64> {
        sets.iter()
            .map(|s| {
                let name = st.mech(*s).ok()?.spec.name.clone();
                let rooms = self.shared.facts(&format!("room/{name}|")).ok()?;
                rooms.iter().filter_map(|(_, v)| serde_json::from_slice::<Room>(v).ok()).map(|r| r.bytes()).max()
            })
            .try_fold(0, |m, r| r.map(|r| m.max(r)))
    }

    fn open_request(&self, id: &str, plan: &Plan) -> Result<u64> {
        let mut st = self.lock();
        let rid = st.tick();
        let (steps, rooms) = self.resolve(&st, plan);
        let key = format!("dispatch/{id}");
        if let Some(peak) = self.shared.fact(&key)? {
            st.peaks.insert(rid, String::from_utf8_lossy(&peak).into_owned());
            self.shared.put(&key, None)?;
        }
        st.table.requests.push(Req { id: rid, name: id.to_string(), steps, rooms, ..Req::default() });
        st.stats.insert(rid, Stats { request: id.to_string(), ..Stats::default() });
        self.publish(&st);
        self.unlock(st);
        Ok(rid)
    }

    fn bind(self: &Arc<Self>, set: u32, block: u32, stream: usize) -> Result<Pin> {
        let key = (set, block);
        let start = Instant::now();
        let mut st = self.lock();
        let req = st.binding(set);
        if st.table.page(key).is_none() {
            return Err(Error::Invalid(format!("no block {block} in set {set}")));
        }
        let (mut hit, mut waited, mut speculative) = (true, false, false);
        loop {
            if let Some(e) = st.failed.remove(&key) {
                return Err(e);
            }
            let identity = st.mech(set)?.spec.fingerprint.key();
            match st.homes.get(&key) {
                // Not the set's weights now and still pinned: it goes when released.
                Some(h) if h.identity != identity => {
                    hit = false;
                    waited = true;
                    self.wait(&mut st);
                }
                Some(h) if h.filled || !(h.filling || h.transforming) => {
                    speculative |= h.speculative;
                    break;
                }
                Some(h) => {
                    speculative |= h.speculative;
                    hit = false;
                    waited = true;
                    self.wait(&mut st);
                }
                None => {
                    hit = false;
                    let orphans = st.table.orphans > 0;
                    self.unlock(st);
                    if orphans && self.adopt(vec![key], req)? > 0 {
                        st = self.lock();
                        continue;
                    }
                    let actions = self.demand(Trigger::Bind { page: key, req, tried: Tried::default() }, req)?;
                    st = self.lock();
                    if let Some(Action::Exhausted { need }) = actions.last() {
                        let stats = st.stats(req).cloned().unwrap_or_default();
                        self.unlock(st);
                        return Err(Error::Exhausted(exhausted(*need, None, stats)));
                    }
                }
            }
        }
        let (va, off, span) = st.place(key)?;
        let id = st.tick();
        let now = st.clock;
        let Some(h) = st.homes.get_mut(&key) else { return Err(Error::Internal("a bound page lost its home".into())) };
        let fresh = std::mem::take(&mut h.fresh);
        let transform = fresh && !h.filled;
        if transform {
            h.transforming = true;
        }
        let (ready, origin) = (h.ready, if hit { Origin::Hit } else { h.origin });
        st.pins.insert(id, (key, req));
        if let Some(p) = st.table.page_mut(key) {
            p.last_use = now;
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
        self.publish(&st);
        self.unlock(st);
        self.speculate(req);
        self.backend.wait(ready, stream)?;
        Ok(Pin { ptr: va.base + off, len: span, ready, fresh, inner: self.clone(), key, id, stream, transform, transformed: AtomicBool::new(false) })
    }

    /// A configuration's measured room: exact, else bounded by the nearest larger measured
    /// shape, else scaled from the nearest smaller by its largest dimension ratio.
    fn measured(&self, st: &mut State, name: &str, shape: &str, levels: &BTreeMap<String, u32>) -> Result<Option<Room>> {
        if let Some(room) = self.memo(st, &room_key(name, shape, levels))? {
            return Ok(Some(room));
        }
        let cuts = room_key("", "", levels);
        let cuts = cuts.trim_start_matches("||");
        let rooms = self.shared.facts(&format!("room/{name}|"))?;
        let Some(ours) = dims(shape) else { return Ok(None) };
        let mut larger: Option<(u64, Room)> = None;
        let mut smaller: Option<(u64, f64, Room)> = None;
        for (key, v) in &rooms {
            let mut parts = key.trim_start_matches("room/").rsplitn(3, '|');
            let (Some(c), Some(s)) = (parts.next(), parts.next()) else { continue };
            let (Some(theirs), Ok(room)) = (dims(s), serde_json::from_slice::<Room>(v)) else { continue };
            if c != cuts || theirs.len() != ours.len() {
                continue;
            }
            let product: u64 = theirs.iter().product();
            if theirs.iter().zip(&ours).all(|(t, o)| t >= o) && larger.as_ref().is_none_or(|(p, _)| product < *p) {
                larger = Some((product, room));
            } else if theirs.iter().zip(&ours).all(|(t, o)| t <= o) && smaller.as_ref().is_none_or(|(p, ..)| product > *p) {
                let ratio = theirs.iter().zip(&ours).map(|(t, o)| *o as f64 / (*t).max(1) as f64).fold(1.0, f64::max);
                smaller = Some((product, ratio, room));
            }
        }
        if let Some((_, room)) = larger {
            return Ok(Some(room));
        }
        Ok(smaller.map(|(_, ratio, room)| Room { delta: (room.bytes() as f64 * ratio) as u64, slack: 0, context: 0, sites: room.sites }))
    }

    /// The pass's room at its cut levels (`None`: unmeasured) and the unmeasured reserve (the
    /// largest context growth measured on the machine), with its sites' finer and coarser
    /// measured rooms for the decision.
    fn room(&self, st: &mut State, set: u32, shape: &str, allocated: u64) -> Result<(Option<u64>, u64, Vec<CutSite>)> {
        let name = st.mech(set)?.spec.name.clone();
        let levels = st.levels(set, shape);
        let now = self.measured(st, &name, shape, &levels)?;
        let mut names: BTreeSet<String> = now.iter().flat_map(|r| r.sites.iter().map(|(n, _)| n.clone())).collect();
        names.extend(levels.keys().cloned());
        names.extend(st.collective.iter().filter(|(s, sh, _)| *s == set && sh == shape).map(|(.., n)| n.clone()));
        let mut sites = vec![];
        for n in names {
            let level = levels.get(&n).copied().unwrap_or(0);
            let peak = now.as_ref().and_then(|r| r.sites.iter().find(|(s, _)| *s == n)).map_or(0, |(_, p)| *p);
            let finer = self.measured(st, &name, shape, &with(&levels, &n, level + 1))?.map(|r| allocated + r.bytes());
            let coarser = match level {
                0 => None,
                l => {
                    let at = with(&levels, &n, l - 1);
                    let failed = st.failed_at.get(&room_key(&name, shape, &at)).copied();
                    self.measured(st, &name, shape, &at)?.map(|r| (allocated + r.bytes(), failed))
                }
            };
            let key = (set, shape.to_string(), n.clone());
            sites.push(CutSite { stuck: st.stuck.contains(&key), collective: st.collective.contains(&key), name: n, level, peak, finer, coarser });
        }
        let growth = self.shared.facts("room/")?.iter().filter_map(|(_, v)| serde_json::from_slice::<Room>(v).ok()).map(|r| r.context).max().unwrap_or(0);
        Ok((now.map(|r| allocated + r.bytes()), allocated + growth, sites))
    }

    /// The active pass's room now (its levels may have just changed), its sites set on its
    /// request.
    fn pass_room(&self, st: &mut State, req: u64) -> Result<(Option<u64>, u64)> {
        let Some((set, shape, allocated)) = st.active.get(&req).map(|a| (a.set, a.shape.clone(), a.allocated)) else { return Ok((None, 0)) };
        let (room, unmeasured, sites) = self.room(st, set, &shape, allocated)?;
        if let Some(r) = st.table.request_mut(req) {
            r.sites = sites;
        }
        Ok((room, unmeasured))
    }

    fn memo(&self, st: &mut State, key: &str) -> Result<Option<Room>> {
        if let Some(r) = st.rooms.get(key) {
            return Ok(Some(r.clone()));
        }
        let room = self.shared.fact(&format!("room/{key}"))?.and_then(|v| serde_json::from_slice::<Room>(&v).ok());
        if let Some(r) = &room {
            st.rooms.insert(key.to_string(), r.clone());
        }
        Ok(room)
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
        // Each page is fetched ahead at most once per pass.
        r.speculated.clear();
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
                r.rooms.push(None);
                st.synthetic.insert(req);
                return Self::advance(st, req, set);
            }
        };
        r.cursor = crate::table::Cursor { step: j, passes: BTreeMap::from([(set, 1)]), next: BTreeMap::from([(set, 0)]) };
        r.starved = false;
        Ok(true)
    }

    fn pass(&self, req: u64, set: u32, shape: &str, torch: Torch) -> Result<u64> {
        let (id, room, unmeasured, step_pages, orphans) = {
            let mut st = self.lock();
            st.mech(set)?;
            self.refresh(&mut st)?;
            if Self::advance(&mut st, req, set)? {
                let t = &st.table;
                let capacity = t.capacity.saturating_sub(t.foreign + t.context + t.others.iter().map(crate::table::Remote::held).sum::<u64>());
                if let Some(s) = st.step_stats(req) {
                    s.capacity = capacity;
                }
                self.tier_uses(&mut st, req);
            }
            self.measure(&mut st, torch);
            let (id, context) = (st.tick(), st.table.context);
            st.active.insert(req, Active { id, set, shape: shape.to_string(), allocated: torch.allocated, context });
            if let Some(r) = st.table.request_mut(req) {
                r.episode = Default::default();
                r.claimed = 0;
            }
            let (room, unmeasured) = self.pass_room(&mut st, req)?;
            let step: Vec<Key> = st.table.request(req).and_then(|r| r.steps.get(r.cursor.step as usize)).map(|(sets, _)| st.table.pages().filter(|(k, p)| sets.contains(&k.0) && !p.mapped).map(|(k, _)| k).collect()).unwrap_or_default();
            let orphans = st.table.orphans > 0;
            self.unlock(st);
            (id, room, unmeasured, step, orphans)
        };
        // Kept pages of this step are adopted before the demand could release them as orphans.
        if orphans {
            self.adopt(step_pages, Some(req))?;
        }
        let actions = self.demand(Trigger::Pass { req, room, unmeasured, tried: Tried::default() }, Some(req))?;
        if let Some(Action::Exhausted { need }) = actions.last() {
            let st = self.lock();
            let stats = st.stats.get(&req).cloned().unwrap_or_default();
            self.unlock(st);
            return Err(Error::Exhausted(exhausted(*need, None, stats)));
        }
        let mut st = self.lock();
        let w = st.table.request(req).map_or(0, |r| st.table.w(r));
        if let Some(s) = st.step_stats(req) {
            s.w = s.w.max(w);
        }
        if let Some(s) = st.stats(Some(req)) {
            s.peak = s.peak.max(w);
        }
        self.publish(&st);
        self.unlock(st);
        self.speculate(Some(req));
        Ok(id)
    }

    /// Tells the tier each set's next use in this request, for its release order.
    fn tier_uses(&self, st: &mut State, req: u64) {
        if self.tier.is_none() {
            return;
        }
        let Some(r) = st.table.request(req) else { return };
        let uses: Vec<Call> = st.sets.iter().enumerate().filter_map(|(i, m)| m.as_ref().map(|m| Call::Use { digest: m.spec.layout.digest.clone(), distance: r.distance((i as u32, 0)).map(|d| d.0) })).collect();
        st.outbox.extend(uses);
    }

    /// Torch's figures now, and the context derived from them: this process's device bytes
    /// less torch's reserved and the pages it created (an imported page is not charged to the
    /// importer's process).
    fn measure(&self, st: &mut State, torch: Torch) {
        st.table.allocated = torch.allocated;
        st.table.reserved = torch.reserved;
        if let Ok(used) = self.backend.process_used() {
            let created: u64 = st.homes.iter().filter(|(_, h)| h.origin != Origin::Kept).filter_map(|(k, _)| st.table.page(*k)).map(|p| p.bytes).sum();
            st.table.context = used.saturating_sub(torch.reserved + created);
        }
    }

    fn end_pass(&self, req: u64, pass: u64, measured: Measured, torch: Torch) -> Result<()> {
        let mut st = self.lock();
        self.measure(&mut st, torch);
        if st.active.get(&req).is_none_or(|a| a.id != pass) {
            self.unlock(st);
            return Ok(());
        }
        let Some(active) = st.active.remove(&req) else { return Ok(()) };
        let name = st.mech(active.set)?.spec.name.clone();
        let levels = st.levels(active.set, &active.shape);
        let key = room_key(&name, &active.shape, &levels);
        let growth = st.table.context.saturating_sub(active.context);
        let room = Room { delta: measured.allocated_peak.saturating_sub(active.allocated), slack: measured.slack, context: growth, sites: measured.sites.clone() };
        // A site whose finer level did not lower the allocated peak is not cut further here.
        for (site, level) in levels.iter().filter(|(_, l)| **l > 0) {
            if self.memo(&mut st, &room_key(&name, &active.shape, &with(&levels, site, level - 1)))?.is_some_and(|c| room.delta >= c.delta) {
                st.stuck.insert((active.set, active.shape.clone(), site.clone()));
            }
        }
        let known = self.memo(&mut st, &key)?;
        let merged = match &known {
            Some(k) => Room { delta: k.delta.max(room.delta), slack: k.slack.max(room.slack), context: k.context.max(room.context), sites: if room.sites.is_empty() { k.sites.clone() } else { room.sites.clone() } },
            None => room.clone(),
        };
        if known.as_ref() != Some(&merged) {
            st.rooms.insert(key.clone(), merged.clone());
            self.shared.put(&format!("room/{key}"), Some(&crate::os::encode(&merged)?))?;
            self.shared.ring();
        }
        self.publish(&st);
        self.unlock(st);
        Ok(())
    }

    fn oom(&self, req: u64, need: Option<u64>, site: Option<Site<'_>>, torch: Torch) -> Result<Next> {
        {
            let mut st = self.lock();
            self.measure(&mut st, torch);
            if let (Some(s), Some(a)) = (site.filter(|s| s.collective), st.active.get(&req)) {
                let key = (a.set, a.shape.clone(), s.name.to_string());
                st.collective.insert(key);
            }
            self.unlock(st);
        }
        let trigger = Trigger::Oom { req, need, site: site.map(|s| s.name.to_string()), collective: site.is_some_and(|s| s.collective), tried: Tried::default() };
        let actions = self.demand(trigger, Some(req))?;
        let mut st = self.lock();
        let stats = |st: &State| st.stats.get(&req).cloned().unwrap_or_default();
        let next = match actions.iter().rev().find(|a| matches!(a, Action::Cut { .. } | Action::Park(_) | Action::Exhausted { .. })) {
            Some(Action::Cut { site, .. }) => {
                let level = st.active.get(&req).map(|a| (a.set, a.shape.clone())).and_then(|(set, shape)| st.cuts.get(&(set, shape)).and_then(|c| c.get(site).copied()));
                Next::Cut(level.unwrap_or(1))
            }
            Some(Action::Park(_)) => {
                if let Some(r) = st.table.request_mut(req) {
                    r.episode.parked = true;
                }
                Next::Park(st.table.headroom)
            }
            Some(Action::Exhausted { need: short }) => Next::Exhausted(exhausted(*short, site.map(|s| s.name), stats(&st))),
            _ => {
                if let Some(s) = st.stats(Some(req)) {
                    s.retries += 1;
                }
                Next::Retry
            }
        };
        self.publish(&st);
        self.unlock(st);
        Ok(next)
    }

    fn end_request(&self, req: u64) -> Stats {
        let mut st = self.lock();
        let name = st.table.request(req).map(|r| r.name.clone()).unwrap_or_default();
        st.active.remove(&req);
        st.synthetic.remove(&req);
        st.table.requests.retain(|r| r.id != req);
        let owned: Vec<u32> = st.table.caches.iter().filter(|c| c.owner == name).map(|c| c.id).collect();
        let holding: Vec<u32> = st.table.caches.iter().filter(|c| c.owner == name && c.bytes > 0).map(|c| c.id).collect();
        let shrinks: Vec<(u32, u64, Shrink)> = holding.iter().filter_map(|id| st.shrinks.get(id).map(|s| (*id, u64::MAX, s.clone()))).collect();
        let leftover: u64 = st.pins.values().filter(|(_, r)| *r == Some(req)).filter_map(|(k, _)| st.table.page(*k)).map(|p| p.bytes).sum();
        let mut stats = st.stats.remove(&req).unwrap_or_default();
        if let Some(key) = st.peaks.remove(&req) {
            let key = format!("peak/{key}");
            let known = self.shared.fact(&key).ok().flatten().and_then(|v| v.try_into().ok()).map(u64::from_le_bytes);
            if known.is_none_or(|k| k < stats.peak) {
                let _ = self.shared.put(&key, Some(&stats.peak.to_le_bytes()));
            }
        }
        self.unlock(st);
        self.shrink(shrinks);
        let mut st = self.lock();
        st.table.caches.retain(|c| !owned.contains(&c.id));
        owned.iter().for_each(|id| {
            st.shrinks.remove(id);
        });
        stats.leftover = leftover;
        st.last = Some(stats.clone());
        self.publish(&st);
        let uses: Vec<Call> = st.sets.iter().flatten().map(|m| Call::Use { digest: m.spec.layout.digest.clone(), distance: None }).collect();
        if self.tier.is_some() {
            st.outbox.extend(uses);
        }
        self.unlock(st);
        self.shared.ring();
        stats
    }

    /// The ask thread's answer, from pages only: it runs no caller code. An idle process's trim
    /// and cache shrinks go to its trim thread, after the answer.
    fn answer(&self, ceiling: Class, bytes: u64) -> u64 {
        let out = crate::caught(|| {
            let mut st = self.lock();
            let actions = decide(&st.table, &Trigger::Ask { ceiling, bytes });
            let mut freed = 0;
            let mut wanted = Wanted::default();
            for a in &actions {
                match a {
                    Action::Unmap { page, class } => {
                        freed += st.table.page(*page).map_or(0, |p| p.bytes);
                        self.unmap(&mut st, *page, *class, None, true)?;
                    }
                    Action::Trim(_) => wanted.trim = true,
                    Action::Shrink { cache, bytes } => wanted.shrinks.push((*cache, *bytes)),
                    _ => {}
                }
            }
            self.publish(&st);
            self.unlock(st);
            if wanted.trim || !wanted.shrinks.is_empty() {
                let (lock, want) = &*self.wanted;
                let mut w = lock.lock();
                w.trim |= wanted.trim;
                w.shrinks.append(&mut wanted.shrinks);
                want.notify_one();
            }
            Ok(freed)
        });
        if out.is_err() {
            self.recover();
        }
        out.unwrap_or(0)
    }

    /// The trim thread's work: the trim hook and shrinks an ask wanted, then a ledger event.
    fn idle_trim(&self, wanted: Wanted) {
        if wanted.trim {
            self.trim();
        }
        let shrinks: Vec<(u32, u64, Shrink)> = {
            let st = self.state.lock();
            wanted.shrinks.iter().filter_map(|(id, b)| st.shrinks.get(id).map(|s| (*id, *b, s.clone()))).collect()
        };
        self.shrink(shrinks);
        let st = self.state.lock();
        self.publish(&st);
        drop(st);
        self.shared.ring();
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.row.knock();
        self.jobs.0.lock().closed = true;
        self.jobs.1.notify_all();
        self.wanted.0.lock().closed = true;
        self.wanted.1.notify_all();
        let mut st = self.state.lock();
        let homes: Vec<(Key, Home)> = st.homes.drain().collect();
        for (key, home) in homes {
            if let Ok((va, off, span)) = st.place(key) {
                let _ = self.backend.unmap(va, off, span);
            }
            self.backend.destroy(home.phys);
        }
        for m in st.sets.drain(..).flatten() {
            self.backend.release(m.va);
            if let Some(held) = &m.held {
                held.unpin(|p| self.backend.unregister_host(p));
            }
        }
        let calls = std::mem::take(&mut st.outbox);
        drop(st);
        self.send(calls);
    }
}

fn fill_loop(jobs: &(Mutex<Jobs>, Condvar), inner: &Weak<Inner>) {
    loop {
        let job = {
            let (lock, queued) = jobs;
            let mut jobs = lock.lock();
            loop {
                if jobs.closed {
                    return;
                }
                if let Some(job) = jobs.demand.pop_front().or_else(|| jobs.ahead.pop_front()) {
                    break job;
                }
                queued.wait(&mut jobs);
            }
        };
        let Some(me) = inner.upgrade() else { return };
        // A panic outside the copy fails the job's binder; the thread keeps serving.
        if let Err(p) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| me.fill(&job))) {
            let msg = p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned());
            me.fail(&job, Error::Internal(format!("panic: {}", msg.unwrap_or_default())));
        }
    }
}

fn ask_loop(listener: crate::os::Listener, row: &Weak<Row>, inner: &Weak<Inner>, stop: &AtomicBool) {
    for stream in crate::os::incoming(listener) {
        if stop.load(Ordering::Acquire) {
            return;
        }
        let (Some(me), Some(row)) = (inner.upgrade(), row.upgrade()) else { return };
        // Never dies: a failed or panicking answer frees nothing, and the next ask is served.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| row.serve(&stream, |ceiling, bytes| me.answer(ceiling, bytes))));
    }
}

fn trim_loop(wanted: &(Mutex<Wanted>, Condvar), inner: &Weak<Inner>) {
    loop {
        let wanted = {
            let (lock, want) = wanted;
            let mut w = lock.lock();
            while !w.trim && w.shrinks.is_empty() {
                if w.closed {
                    return;
                }
                want.wait(&mut w);
            }
            std::mem::take(&mut *w)
        };
        let Some(me) = inner.upgrade() else { return };
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| me.idle_trim(wanted)));
    }
}

impl Request {
    /// Replaces the plan (packages declare it after the request opened); the cursor starts over.
    pub fn plan(&self, plan: Plan) {
        let _ = self.inner.guard(|| {
            let mut st = self.inner.lock();
            let (steps, rooms) = self.inner.resolve(&st, &plan);
            if let Some(r) = st.table.request_mut(self.id) {
                r.steps = steps;
                r.rooms = rooms;
                r.cursor = Default::default();
            }
            st.synthetic.remove(&self.id);
            self.inner.publish(&st);
            self.inner.unlock(st);
            Ok(())
        });
    }

    /// Starts a pass of `set` at `shape` (dimensions joined by `x`, e.g. `1x81x480x832`), with
    /// torch's bytes now; returns once the pass's measured room is free.
    pub fn pass(&self, set: SetId, shape: &str, torch: Torch) -> Result<Pass> {
        let id = self.inner.guard(|| self.inner.pass(self.id, set.0, shape, torch))?;
        Ok(Pass { inner: self.inner.clone(), req: self.id, id })
    }

    /// An op ran out of memory, needing `need` more bytes when the error said; `site` is its
    /// cut site, `None` when the op cannot be cut; `torch` the allocator's figures now.
    pub fn oom(&self, need: Option<u64>, site: Option<Site<'_>>, torch: Torch) -> Next {
        match self.inner.guard(|| self.inner.oom(self.id, need, site, torch)) {
            Ok(next) => next,
            Err(Error::Exhausted(e)) => Next::Exhausted(e),
            Err(_) => Next::Exhausted(exhausted(need.unwrap_or(0), site.map(|s| s.name), self.stats())),
        }
    }

    /// What the caller parked after a `Park` answer; 0 makes the next `oom` answer Exhausted.
    pub fn parked(&self, bytes: u64) {
        let _ = self.inner.guard(|| {
            let mut st = self.inner.lock();
            if let Some(s) = st.stats(Some(self.id)) {
                s.parks += 1;
                s.parked += bytes;
            }
            if let Some(r) = st.table.request_mut(self.id) {
                r.episode.parked = bytes == 0;
            }
            self.inner.unlock(st);
            Ok(())
        });
    }

    /// The current pass's cut level for op site `site` (0: whole); a collective site's is the
    /// level the ranks agreed.
    pub fn cut(&self, site: &str) -> u32 {
        self.inner
            .guard(|| {
                let st = self.inner.lock();
                let level = st.active.get(&self.id).and_then(|a| st.cuts.get(&(a.set, a.shape.clone()))).and_then(|c| c.get(site)).copied().unwrap_or(0);
                self.inner.unlock(st);
                Ok(level)
            })
            .unwrap_or(0)
    }

    /// The current pass's collective sites' proposed levels ([`Pass::levels`]).
    pub fn levels(&self) -> Vec<(String, u32)> {
        let pass = self.inner.state.lock().active.get(&self.id).map(|a| a.id);
        pass.map(|id| self.inner.levels(self.id, id)).unwrap_or_default()
    }

    /// The levels the ranks agreed, for the current pass ([`Pass::agree`]).
    pub fn agree(&self, levels: Vec<(String, u32)>) {
        let pass = self.inner.state.lock().active.get(&self.id).map(|a| a.id);
        if let Some(id) = pass {
            self.inner.agree(self.id, id, levels);
        }
    }

    pub fn stats(&self) -> Stats {
        let st = self.inner.state.lock();
        st.stats.get(&self.id).cloned().unwrap_or_default()
    }

    /// Ends the request, returning its record.
    pub fn end(mut self) -> Stats {
        self.ended = true;
        let id = self.id;
        self.inner.guard(|| Ok(self.inner.end_request(id))).unwrap_or_default()
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
    /// weights once the stream's work so far completes. Other binders wait on that in-stream;
    /// the block stays pinned and is kept (read-only, exported) only after it.
    pub fn filled(&self) {
        self.transformed.store(true, Ordering::Release);
        let _ = self.inner.guard(|| {
            let event = self.inner.backend.record(self.stream)?;
            let mut st = self.inner.lock();
            if let Some(h) = st.homes.get_mut(&self.key) {
                let old = std::mem::replace(&mut h.ready, event);
                h.filled = true;
                h.transforming = false;
                if old != event {
                    self.inner.backend.retire(old);
                }
            }
            st.sync(self.key);
            self.inner.keep(&mut st, self.key);
            self.inner.publish(&st);
            self.inner.unlock(st);
            self.inner.landed.notify_all();
            Ok(())
        });
    }
}

impl Drop for Pin {
    fn drop(&mut self) {
        let event = self.inner.backend.record(self.stream).unwrap_or(Event::DONE);
        let untransformed = self.transform && !self.transformed.load(Ordering::Acquire);
        let _ = self.inner.release_tx.send(Release { key: self.key, pin: self.id, event, untransformed });
        self.inner.landed.notify_all();
    }
}

impl Pass {
    /// Records what the pass measured, with the allocator's figures at its end.
    pub fn end(self, measured: Measured, torch: Torch) {
        let _ = self.inner.guard(|| self.inner.end_pass(self.req, self.id, measured, torch));
    }

    /// The collective sites' proposed levels for this pass: the level in effect, or a finer
    /// one an OOM proposed. cozy-runtime all-reduces their maximum across the ranks and hands
    /// it to [`Pass::agree`] before the pass runs its collectives.
    pub fn levels(&self) -> Vec<(String, u32)> {
        self.inner.levels(self.req, self.id)
    }

    /// The levels the ranks agreed (the maximum of their proposals): in effect for this pass.
    pub fn agree(&self, levels: Vec<(String, u32)>) {
        self.inner.agree(self.req, self.id, levels);
    }
}

impl Inner {
    fn levels(&self, req: u64, pass: u64) -> Vec<(String, u32)> {
        self.guard(|| {
            let st = self.lock();
            let out = match st.active.get(&req).filter(|a| a.id == pass) {
                Some(a) => {
                    let now = st.levels(a.set, &a.shape);
                    let proposed = st.proposed.get(&(a.set, a.shape.clone())).cloned().unwrap_or_default();
                    let mut sites: Vec<String> = st.collective.iter().filter(|(s, sh, _)| *s == a.set && *sh == a.shape).map(|(.., n)| n.clone()).collect();
                    sites.sort();
                    sites.into_iter().map(|n| (n.clone(), now.get(&n).copied().unwrap_or(0).max(proposed.get(&n).copied().unwrap_or(0)))).collect()
                }
                None => vec![],
            };
            self.unlock(st);
            Ok(out)
        })
        .unwrap_or_default()
    }

    fn agree(&self, req: u64, pass: u64, levels: Vec<(String, u32)>) {
        let _ = self.guard(|| {
            let mut st = self.lock();
            if let Some((set, shape)) = st.active.get(&req).filter(|a| a.id == pass).map(|a| (a.set, a.shape.clone())) {
                for (site, level) in levels {
                    st.collective.insert((set, shape.clone(), site.clone()));
                    let now = st.cuts.entry((set, shape.clone())).or_default().entry(site.clone()).or_insert(0);
                    *now = (*now).max(level);
                    if let Some(p) = st.proposed.get_mut(&(set, shape.clone())) {
                        if p.get(&site).is_some_and(|l| *l <= level) {
                            p.remove(&site);
                        }
                    }
                }
            }
            self.unlock(st);
            Ok(())
        });
    }
}

impl Drop for Pass {
    /// Ends only its own pass: a newer one of the same request keeps its state.
    fn drop(&mut self) {
        let mut st = self.inner.state.lock();
        if st.active.get(&self.req).is_some_and(|a| a.id == self.id) {
            st.active.remove(&self.req);
        }
    }
}

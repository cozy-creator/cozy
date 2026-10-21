//! One executor's memory on one device: the applier of [`crate::decide`]'s actions. One
//! `parking_lot` mutex guards the table and the mechanism's state; it is held for a decision and
//! its application, never across a fill, a remote answer or a caller's callback. Fills run on
//! fill threads; pin releases arrive through a channel; asks are answered by a thread of its
//! own. A panic is caught at every public call: the call fails `Internal`, the table is derived
//! again from the mappings, and the process keeps serving.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Weak};
use std::time::Instant;

use parking_lot::{Condvar, Mutex, MutexGuard};

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
    /// What the driver's context grew by during the pass (workspaces, modules, communicators).
    pub context: u64,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// Already mapped and filled.
    Hit,
    /// A kept page imported by fingerprint: no copy.
    Kept,
    Tier,
    Store,
}

/// One bind: the page-reference trace the gate computes Belady MIN from.
#[derive(Clone, Debug, PartialEq, Eq)]
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fills {
    pub bytes: u64,
    pub ns: u64,
}

/// Bytes moved, by source: fill time over bytes against the measured ceiling gives the %.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Moved {
    /// Kept pages imported by fingerprint (no copy; `ns` is the import).
    pub kept: Fills,
    pub tier: Fills,
    pub store: Fills,
    /// Of the above, bytes filled speculatively.
    pub speculative: u64,
}

/// One step of a request.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
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
#[derive(Clone, Debug, Default, PartialEq, Eq)]
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
    /// A collective's site: ranks agree on cut levels at pass start, so a finer level is
    /// recorded for the next pass and this one recovers locally.
    pub collective: bool,
}

/// One process's memory on one device.
pub struct Memory {
    inner: Arc<Inner>,
}

/// A request in flight; dropping it ends it: its pins' room, its caches and its claims go, and
/// the leftover is measured.
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
    /// Fresh bytes of a set with a transform: unmapped on release unless [`Pin::filled`].
    transform: bool,
    transformed: AtomicBool,
}

/// A pass of a set at a shape, its measured room made. Dropping it without [`Pass::end`]
/// records nothing.
pub struct Pass {
    inner: Arc<Inner>,
    req: u64,
}

type Shrink = Arc<Mutex<Box<dyn FnMut(u64) -> u64 + Send>>>;

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
}

struct Active {
    set: u32,
    shape: String,
    allocated: u64,
}

struct Job {
    key: Key,
    gen: u64,
    req: Option<u64>,
}

struct Release {
    key: Key,
    pin: u64,
    event: Event,
    /// Released fresh without its transform: its bytes are not the weights.
    untransformed: bool,
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
    /// Rooms read from the ledger, by key.
    rooms: HashMap<String, Room>,
    /// Cut levels per (set, shape).
    cuts: HashMap<(u32, String), BTreeMap<String, u32>>,
    stuck: BTreeSet<(u32, String, String)>,
    /// The room each cut configuration failed at.
    failed_at: HashMap<String, u64>,
    stats: HashMap<u64, Stats>,
    last: Option<Stats>,
    active: HashMap<u64, Active>,
    failed: HashMap<Key, Error>,
    /// Request → the `<package>|<shape>` its peak is recorded under.
    peaks: HashMap<u64, String>,
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
}

/// `<set>|<shape>|<site=level,...>`.
fn room_key(set: &str, shape: &str, levels: &BTreeMap<String, u32>) -> String {
    let cuts: Vec<String> = levels.iter().filter(|(_, l)| **l > 0).map(|(s, l)| format!("{s}={l}")).collect();
    format!("{set}|{shape}|{}", cuts.join(","))
}

fn dims(shape: &str) -> Option<Vec<u64>> {
    shape.split('x').map(|d| d.parse().ok()).collect()
}

struct Inner {
    backend: Box<dyn Backend>,
    shared: Arc<Shared>,
    row: Arc<Row>,
    custody: Option<custody::Client>,
    tier: Option<tier::Client>,
    state: Mutex<State>,
    landed: Condvar,
    release_tx: Sender<Release>,
    releases: Mutex<Receiver<Release>>,
    jobs: Mutex<Option<Sender<Job>>>,
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
            let capacity = gpu.memory.min(backend.capacity());
            let shared = match grant {
                Some(g) => Shared::adopt(g.table, &gpu, capacity, Some(g.events))?,
                None => Shared::open(&gpu, capacity, None)?,
            };
            let shared = Arc::new(shared);
            shared.reap(None)?;
            let row = Arc::new(shared.claim(std::process::id())?);
            let custody = custody::Client::open(&gpu.uuid, &row).map_err(|e| tracing::warn!("varena: no custody: {e}")).ok();
            let tier = tier::Client::open().map_err(|e| tracing::warn!("varena: no RAM tier: {e}")).ok();
            let (release_tx, releases) = channel();
            let (jobs, queue) = channel::<Job>();
            let table = Table { capacity: shared.capacity()?, context: backend.context(), ..Table::default() };
            let inner = Arc::new(Inner {
                backend,
                shared,
                row: row.clone(),
                custody,
                tier,
                state: Mutex::new(State { table, ..State::default() }),
                landed: Condvar::new(),
                release_tx,
                releases: Mutex::new(releases),
                jobs: Mutex::new(Some(jobs)),
                stop: Arc::new(AtomicBool::new(false)),
                #[cfg(unix)]
                tally: Default::default(),
            });
            // One fill thread per copy stream TensorFS runs per device: a fill waits on its
            // streams, so this many keep them busy.
            let queue = Arc::new(Mutex::new(queue));
            for _ in 0..tensorfs::io::ReadOpts::default().streams.max(1) {
                let (queue, weak) = (queue.clone(), Arc::downgrade(&inner));
                std::thread::Builder::new().name("varena-fill".into()).spawn(move || fill_loop(&queue, &weak)).map_err(|e| Error::io("spawn fill", e))?;
            }
            let (weak, stop) = (Arc::downgrade(&inner), inner.stop.clone());
            std::thread::Builder::new().name("varena-ask".into()).spawn(move || ask_loop(&row, &weak, &stop)).map_err(|e| Error::io("spawn ask", e))?;
            inner.publish(&inner.state.lock());
            Ok(Memory { inner })
        })
    }

    /// Registers a set and brings it to `level`. On every start, first or restart, each page
    /// imports a kept page by fingerprint, else fills from the RAM tier, else from the store.
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
            Ok(())
        });
    }

    /// Registers bytes the caller holds (torch tensors) that `shrink(want) -> freed` gives back.
    /// A cache whose `owner` is a request id ends with that request.
    pub fn cache(&self, owner: &str, shrink: Box<dyn FnMut(u64) -> u64 + Send>) -> CacheId {
        let mut st = self.inner.lock();
        let id = st.tick() as u32;
        let last_use = st.clock;
        st.table.caches.push(crate::table::Cache { id, owner: owner.to_string(), bytes: 0, last_use });
        st.shrinks.insert(id, Arc::new(Mutex::new(shrink)));
        CacheId(id)
    }

    /// The cache now holds `bytes`.
    pub fn cached(&self, cache: CacheId, bytes: u64) {
        let mut st = self.inner.lock();
        let now = st.tick();
        if let Some(c) = st.table.caches.iter_mut().find(|c| c.id == cache.0) {
            c.bytes = bytes;
            c.last_use = now;
        }
        self.inner.publish(&st);
    }

    pub fn uncache(&self, cache: CacheId) {
        let mut st = self.inner.lock();
        st.table.caches.retain(|c| c.id != cache.0);
        st.shrinks.remove(&cache.0);
        self.inner.publish(&st);
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
        let mut st = self.inner.lock();
        let _ = self.inner.refresh(&mut st);
        st.table.torch_limit()
    }

    /// The record of the request that ended last.
    pub fn stats(&self) -> Option<Stats> {
        self.inner.lock().last.clone()
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

    /// After a panic: the table's pages from the mappings, each open request's episode afresh.
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
        self.publish(&st);
        self.landed.notify_all();
    }

    /// The state, with pin releases applied and completed release events retired.
    fn lock(&self) -> MutexGuard<'_, State> {
        let mut st = self.state.lock();
        self.drain(&mut st);
        st
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
                if let Some(h) = st.homes.get_mut(&r.key) {
                    h.transforming = false;
                }
                // Its bytes are not the weights: they go, and the next bind fills again.
                let _ = self.unmap(st, r.key, Class::P, None, true);
                wake = true;
            }
            st.sync(r.key);
        }
        let pending: Vec<Key> = st.releasing.iter().copied().collect();
        for key in pending {
            let Some(p) = st.table.page_mut(key) else { continue };
            let events = std::mem::take(&mut p.releasing);
            let left: Vec<u64> = events.into_iter().filter(|e| !self.backend.query(Event(*e)).unwrap_or(true)).collect();
            let done = left.is_empty();
            if let Some(p) = st.table.page_mut(key) {
                p.releasing = left;
            }
            if done {
                st.releasing.remove(&key);
            }
        }
        self.close_sets(st);
        if wake {
            self.landed.notify_all();
        }
    }

    /// Unmaps the unpinned pages of closing sets; a set goes once it has none.
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
                if let Some(t) = &self.tier {
                    let _ = t.close(&m.spec.layout.digest);
                }
            }
            if let Some(slot) = st.table.sets.get_mut(set as usize) {
                *slot = None;
            }
        }
    }

    /// The other rows, orphans, capacity and foreign bytes, read now.
    fn refresh(&self, st: &mut State) -> Result<()> {
        let mut rows = self.shared.rows()?;
        if rows.iter().any(|r| !r.live) && self.shared.reap(None)? {
            rows = self.shared.rows()?;
        }
        let (others, orphans) = ledger::remotes(&rows, self.row.at());
        st.table.others = others;
        st.table.orphans = orphans;
        st.table.capacity = self.shared.capacity()?;
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
            torch: t.torch(),
            caches: if idle { t.torch() } else { 0 },
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
            if let (Some(c), Ok(m)) = (&self.custody, st.mech(key.0)) {
                let _ = c.withdraw(&m.spec.fingerprint.digest, key.1);
            }
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

    /// Imports the page kept under its fingerprint, if custody holds one: no copy.
    fn import(&self, st: &mut State, key: Key, req: Option<u64>) -> Result<bool> {
        let Some(c) = &self.custody else { return Ok(false) };
        let m = st.mech(key.0)?;
        if !m.spec.fingerprint.declared() {
            return Ok(false);
        }
        let start = Instant::now();
        let Some(handle) = c.import(&m.spec.fingerprint.digest, key.1)? else { return Ok(false) };
        let (va, off, span) = st.place(key)?;
        let phys = self.backend.import(handle, span)?;
        if let Err(e) = self.backend.map(va, off, &phys, Access::Read) {
            self.backend.destroy(phys);
            return Err(e);
        }
        let gen = st.tick();
        st.homes.insert(key, Home { phys, ready: Event::DONE, gen, filled: true, filling: false, fresh: false, transforming: false, exported: true, origin: Origin::Kept, speculative: false });
        st.sync(key);
        if let Some(s) = st.stats(req) {
            s.moved.kept.bytes += span;
            s.moved.kept.ns += start.elapsed().as_nanos() as u64;
        }
        Ok(true)
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
        st.homes.insert(key, Home { phys, ready: Event::DONE, gen, filled: false, filling: true, fresh: false, transforming: false, exported: false, origin: Origin::Store, speculative });
        st.sync(key);
        let queued = self.jobs.lock().as_ref().is_some_and(|j| j.send(Job { key, gen, req }).is_ok());
        if !queued {
            return Err(Error::Closed("the fill threads".into()));
        }
        Ok(true)
    }

    /// Exports a filled page of a declared set into custody, read-only.
    fn keep(&self, st: &mut State, key: Key) {
        let Some(c) = &self.custody else { return };
        let Ok((va, off, span)) = st.place(key) else { return };
        let Ok(m) = st.mech(key.0) else { return };
        let fp = m.spec.fingerprint.clone();
        let Some(home) = st.homes.get_mut(&key) else { return };
        if !fp.declared() || !home.filled || home.exported {
            return;
        }
        let kept = self.backend.protect(va, off, span, Access::Read).and_then(|_| self.backend.export(&home.phys)).and_then(|h| c.export(&fp.digest, key.1, span, h));
        match kept {
            Ok(()) => home.exported = true,
            Err(e) => tracing::warn!("varena: keeping {key:?} failed: {e}"),
        }
    }

    /// A fill thread's job: tier, else store (through the tier when it has room), then the
    /// page's ready event.
    fn fill(&self, job: Job) {
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
            let origin = match &held {
                Some(held) if held.read(job.key.1, dst)? => Origin::Tier,
                Some(held) if self.tier.as_ref().is_some_and(|t| t.room(span).unwrap_or(false)) => {
                    held.fill(job.key.1, |host| self.store(&spec.source, &region.items, region.offset, host))?;
                    match held.read(job.key.1, dst)? {
                        true => Origin::Store,
                        false => {
                            self.store(&spec.source, &region.items, region.offset, dst)?;
                            Origin::Store
                        }
                    }
                }
                _ => {
                    self.store(&spec.source, &region.items, region.offset, dst)?;
                    Origin::Store
                }
            };
            Ok((origin, self.backend.fence()?))
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
        drop(st);
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

    /// Decides and applies until the trigger is met or ends: remote rungs (orphans, asks) run
    /// with the lock released, then the procedure decides again with them tried.
    fn demand(&self, mut trigger: Trigger, req: Option<u64>) -> Result<Vec<Action>> {
        loop {
            let mut st = self.lock();
            self.refresh(&mut st)?;
            if matches!(trigger, Trigger::Oom { .. }) {
                st.table.headroom = u64::try_from(platform::contain::host_memory::read().available).unwrap_or(0);
            }
            let actions = decide(&st.table, &trigger);
            let mut remote = vec![];
            let mut shrinks = vec![];
            for a in &actions {
                self.apply(&mut st, a, req, &mut remote, &mut shrinks)?;
            }
            self.publish(&st);
            drop(st);
            self.shrink(shrinks);
            if remote.is_empty() {
                return Ok(actions);
            }
            let tried = match &mut trigger {
                Trigger::Bind { tried, .. } | Trigger::Pass { tried, .. } | Trigger::Oom { tried, .. } | Trigger::Speculate { tried, .. } => tried,
                Trigger::Ask { .. } => return Ok(actions),
            };
            for a in remote {
                match a {
                    Action::ReleaseOrphans(bytes) => {
                        tried.orphans = true;
                        if let Some(c) = &self.custody {
                            tried.freed += c.release(bytes)?;
                        }
                    }
                    Action::Ask { row, ceiling, bytes } => {
                        tried.asked.insert((row, ceiling));
                        let at = ledger::row_offset(row);
                        let until = self.shared.ask(at, ceiling, bytes)?;
                        tried.freed += self.shared.answered(at, ceiling, until)?;
                    }
                    _ => {}
                }
            }
        }
    }

    fn apply(&self, st: &mut State, a: &Action, req: Option<u64>, remote: &mut Vec<Action>, shrinks: &mut Vec<(u32, u64, Shrink)>) -> Result<()> {
        match a {
            Action::Unmap { page, class } => self.unmap(st, *page, *class, req, true)?,
            Action::ReleaseOrphans(_) | Action::Ask { .. } => remote.push(a.clone()),
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
                if let (Some(r), Some(step)) = (req, step) {
                    if let Some(rq) = st.table.request_mut(r) {
                        rq.speculated.insert(*k);
                    }
                    if self.map(st, *k, req, true)? {
                        if let Some(p) = st.table.page_mut(*k) {
                            p.speculative = Some((r, step));
                        }
                    }
                }
            }
            Action::Claim { req, bytes } => {
                if let Some(r) = st.table.request_mut(*req) {
                    r.claimed += bytes;
                    r.episode.claimed_free = true;
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
            Action::Cut { .. } | Action::Park(_) | Action::Exhausted { .. } => {}
        }
        Ok(())
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

    /// Fetches ahead for `req` into the surplus.
    fn speculate(&self, req: Option<u64>) {
        if let Some(r) = req {
            let _ = self.demand(Trigger::Speculate { req: r, tried: Tried::default() }, Some(r));
        }
    }

    // ---- the calls

    fn load(&self, spec: SetSpec, level: Level) -> Result<SetId> {
        let id = {
            let mut st = self.lock();
            let existing = st.sets.iter().position(|m| m.as_ref().is_some_and(|m| m.spec.name == spec.name && !m.closing));
            match existing {
                Some(i) if st.mech(i as u32)?.spec.fingerprint == spec.fingerprint => i as u32,
                Some(_) => return Err(Error::Invalid(format!("set {} is loaded with another fingerprint", spec.name))),
                None => {
                    let layout = &spec.layout;
                    let va = self.backend.reserve(layout.nbytes)?;
                    let regions: Vec<(u64, u64)> = layout.regions.iter().map(|r| (r.offset, r.span)).collect();
                    let held = self.tier.as_ref().and_then(|t| t.open_layout(&layout.digest, regions).map_err(|e| tracing::warn!("varena: tier: {e}")).ok()).map(Arc::new);
                    let pages = layout.regions.iter().map(|r| Page { bytes: r.span, ..Page::default() }).collect();
                    let set = Set { name: spec.name.clone(), pages, declared: spec.fingerprint.declared() };
                    let i = st.sets.iter().position(Option::is_none).unwrap_or(st.sets.len());
                    let mech = SetMech { spec, va, held, closing: false };
                    if i == st.sets.len() {
                        st.sets.push(Some(mech));
                        st.table.sets.push(Some(set));
                    } else {
                        if let Some(slot) = st.sets.get_mut(i) {
                            *slot = Some(mech);
                        }
                        if let Some(slot) = st.table.sets.get_mut(i) {
                            *slot = Some(set);
                        }
                    }
                    i as u32
                }
            }
        };
        match level {
            Level::Ram => self.load_ram(id)?,
            Level::Gpu => self.load_gpu(id)?,
        }
        Ok(SetId(id))
    }

    /// Fills the set's layout in the RAM tier, region by region, as host headroom allows.
    fn load_ram(&self, set: u32) -> Result<()> {
        let (spec, held) = {
            let st = self.lock();
            let m = st.mech(set)?;
            (m.spec.clone(), m.held.clone())
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

    /// Imports every kept page, then fills the rest into free bytes only; waits for the fills.
    fn load_gpu(&self, set: u32) -> Result<()> {
        let mut st = self.lock();
        self.refresh(&mut st)?;
        let keys: Vec<Key> = st.table.pages().filter(|(k, p)| k.0 == set && !p.mapped).map(|(k, _)| k).collect();
        for key in keys {
            if self.import(&mut st, key, None)? {
                continue;
            }
            let bytes = st.table.page(key).map_or(0, |p| p.bytes);
            if st.table.free() < bytes || !self.map(&mut st, key, None, true)? {
                break;
            }
        }
        self.publish(&st);
        while st.table.pages().any(|(k, p)| k.0 == set && p.filling) {
            self.landed.wait(&mut st);
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
            match st.homes.get(&key) {
                Some(h) if h.filled || !(h.filling || h.transforming) => {
                    speculative |= h.speculative;
                    break;
                }
                Some(h) => {
                    speculative |= h.speculative;
                    hit = false;
                    waited = true;
                    self.landed.wait(&mut st);
                    self.drain(&mut st);
                }
                None => {
                    hit = false;
                    if self.import(&mut st, key, req)? {
                        continue;
                    }
                    drop(st);
                    let actions = self.demand(Trigger::Bind { page: key, req, tried: Tried::default() }, req)?;
                    st = self.lock();
                    if let Some(Action::Exhausted { need }) = actions.last() {
                        let stats = st.stats(req).cloned().unwrap_or_default();
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
        drop(st);
        self.speculate(req);
        self.backend.wait(ready, stream)?;
        Ok(Pin { ptr: va.base + off, len: span, ready, fresh, inner: self.clone(), key, id, stream, transform, transformed: AtomicBool::new(false) })
    }

    /// The pass's room: measured at its cuts, else bounded by a measured shape, else the
    /// unmeasured reserve (the largest context growth measured on the machine).
    fn room(&self, st: &mut State, set: u32, shape: &str, allocated: u64) -> Result<(u64, bool, Vec<CutSite>)> {
        let name = st.mech(set)?.spec.name.clone();
        let levels = st.levels(set, shape);
        let sites = |st: &State, room: &Room| {
            room.sites.iter().map(|(n, peak)| CutSite { name: n.clone(), level: levels.get(n).copied().unwrap_or(0), peak: *peak, stuck: st.stuck.contains(&(set, shape.to_string(), n.clone())) }).collect()
        };
        if let Some(room) = self.memo(st, &room_key(&name, shape, &levels))? {
            return Ok((allocated + room.bytes(), true, sites(st, &room)));
        }
        let cuts = room_key("", "", &levels);
        let cuts = cuts.trim_start_matches("||");
        let rooms = self.shared.facts(&format!("room/{name}|"))?;
        let ours = dims(shape);
        let mut larger: Option<(u64, Room)> = None;
        let mut smaller: Option<(u64, f64, Room)> = None;
        for (key, v) in &rooms {
            let mut parts = key.trim_start_matches("room/").rsplitn(3, '|');
            let (Some(c), Some(s)) = (parts.next(), parts.next()) else { continue };
            let (Some(theirs), Some(ours), Ok(room)) = (dims(s), ours.as_ref(), serde_json::from_slice::<Room>(v)) else { continue };
            if c != cuts || theirs.len() != ours.len() {
                continue;
            }
            let product: u64 = theirs.iter().product();
            if theirs.iter().zip(ours).all(|(t, o)| t >= o) && larger.as_ref().is_none_or(|(p, _)| product < *p) {
                larger = Some((product, room));
            } else if theirs.iter().zip(ours).all(|(t, o)| t <= o) && smaller.as_ref().is_none_or(|(p, ..)| product > *p) {
                let ratio = theirs.iter().zip(ours).map(|(t, o)| *o as f64 / (*t).max(1) as f64).fold(1.0, f64::max);
                smaller = Some((product, ratio, room));
            }
        }
        if let Some((_, room)) = larger {
            return Ok((allocated + room.bytes(), true, sites(st, &room)));
        }
        if let Some((_, ratio, room)) = smaller {
            return Ok((allocated + (room.bytes() as f64 * ratio) as u64, true, sites(st, &room)));
        }
        let growth = self.shared.facts("room/")?.iter().filter_map(|(_, v)| serde_json::from_slice::<Room>(v).ok()).map(|r| r.context).max().unwrap_or(0);
        Ok((allocated + growth, false, vec![]))
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

    /// Relaxes one cut a level when that level is measured and its room fits the surplus (F,
    /// O and own N) and beats the room it last failed at.
    fn relax(&self, st: &mut State, set: u32, shape: &str) -> Result<()> {
        let name = st.mech(set)?.spec.name.clone();
        let levels = st.levels(set, shape);
        let Some(now) = self.memo(st, &room_key(&name, shape, &levels))? else { return Ok(()) };
        let surplus = st.table.free() + st.table.orphans + st.table.classes().get(Class::N.index()).copied().unwrap_or(0);
        for (site, level) in levels.iter().filter(|(_, l)| **l > 0) {
            let mut coarser = levels.clone();
            coarser.insert(site.clone(), level - 1);
            let key = room_key(&name, shape, &coarser);
            let Some(room) = self.memo(st, &key)? else { continue };
            let fits = room.bytes().saturating_sub(now.bytes()) <= surplus && st.failed_at.get(&key).is_none_or(|f| surplus > *f);
            if fits {
                st.cuts.insert((set, shape.to_string()), coarser);
                return Ok(());
            }
        }
        Ok(())
    }

    /// Moves the request's cursor to the pass of `set`; true when a step began.
    fn advance(st: &mut State, req: u64, set: u32) -> Result<bool> {
        let r = st.table.request_mut(req).ok_or_else(|| Error::Closed("the request".into()))?;
        let i = r.cursor.step;
        let current = r.steps.get(i as usize).filter(|(s, _)| s.contains(&set)).map(|(_, p)| *p);
        let begun = r.cursor.passes.get(&set).copied().unwrap_or(0);
        let later = (i + 1..r.steps.len() as u32).find(|j| r.steps.get(*j as usize).is_some_and(|(s, _)| s.contains(&set)));
        let first = begun == 0 && r.cursor.passes.is_empty();
        if let Some(passes) = current.filter(|p| begun < *p || later.is_none()) {
            let _ = passes;
            r.cursor.passes.insert(set, begun + 1);
            r.cursor.next.insert(set, 0);
            return Ok(first);
        }
        let j = match later {
            Some(j) => j,
            None => {
                // A pass the plan does not declare: a step of its own at the end.
                r.steps.push((vec![set], 1));
                r.rooms.push(None);
                r.steps.len() as u32 - 1
            }
        };
        r.cursor = crate::table::Cursor { step: j, passes: BTreeMap::from([(set, 1)]), next: BTreeMap::from([(set, 0)]) };
        r.starved = false;
        r.speculated.clear();
        Ok(true)
    }

    fn pass(&self, req: u64, set: u32, shape: &str, torch: Torch) -> Result<()> {
        let (mut room, mut measured) = {
            let mut st = self.lock();
            st.mech(set)?;
            self.refresh(&mut st)?;
            if Self::advance(&mut st, req, set)? {
                let t = &st.table;
                let capacity = t.capacity.saturating_sub(t.foreign + t.context + t.others.iter().map(crate::table::Remote::held).sum::<u64>());
                if let Some(s) = st.step_stats(req) {
                    s.capacity = capacity;
                }
                self.tier_uses(&st, req);
            }
            st.active.insert(req, Active { set, shape: shape.to_string(), allocated: torch.allocated });
            if let Some(r) = st.table.request_mut(req) {
                r.episode = Default::default();
                r.claimed = 0;
            }
            self.relax(&mut st, set, shape)?;
            let (room, measured, sites) = self.room(&mut st, set, shape, torch.allocated)?;
            if let Some(r) = st.table.request_mut(req) {
                r.sites = sites;
            }
            (room, measured)
        };
        let mut tried = Tried::default();
        loop {
            let actions = self.demand(Trigger::Pass { req, room, tried: tried.clone() }, Some(req))?;
            match actions.last() {
                Some(Action::Exhausted { need }) => {
                    let st = self.lock();
                    let stats = st.stats.get(&req).cloned().unwrap_or_default();
                    return Err(Error::Exhausted(exhausted(*need, None, stats)));
                }
                Some(Action::Cut { site, .. }) => {
                    tried.cut.insert(site.clone());
                    let mut st = self.lock();
                    let level = st.cuts.entry((set, shape.to_string())).or_default().entry(site.clone()).or_insert(0);
                    *level += 1;
                    let peak = st.table.request(req).and_then(|r| r.sites.iter().find(|s| &s.name == site)).map_or(0, |s| s.peak);
                    if let Some(s) = st.stats(Some(req)) {
                        s.cuts += 1;
                    }
                    let (r, m, sites) = self.room(&mut st, set, shape, torch.allocated)?;
                    // Unmeasured at the finer level: the site's peak halves with it, as an
                    // estimate the catch backs.
                    (room, measured) = if m && r < room { (r, m) } else { (room.saturating_sub(peak / 2), measured) };
                    if let Some(rq) = st.table.request_mut(req) {
                        rq.sites = sites;
                    }
                }
                _ => break,
            }
        }
        let mut st = self.lock();
        if let Some(r) = st.table.request_mut(req) {
            r.room = room;
            r.measured = measured;
        }
        let w = st.table.request(req).map_or(0, |r| st.table.w(r));
        if let Some(s) = st.step_stats(req) {
            s.w = s.w.max(w);
        }
        if let Some(s) = st.stats(Some(req)) {
            s.peak = s.peak.max(w);
        }
        self.publish(&st);
        drop(st);
        self.speculate(Some(req));
        Ok(())
    }

    /// Tells the tier each set's next use in this request, for its release order.
    fn tier_uses(&self, st: &State, req: u64) {
        let (Some(tier), Some(r)) = (&self.tier, st.table.request(req)) else { return };
        for (i, m) in st.sets.iter().enumerate() {
            let Some(m) = m else { continue };
            let distance = r.distance((i as u32, 0)).map(|d| d.0);
            let _ = tier.use_layout(&m.spec.layout.digest, distance);
        }
    }

    fn end_pass(&self, req: u64, measured: Measured) -> Result<()> {
        let mut st = self.lock();
        let Some(active) = st.active.remove(&req) else { return Ok(()) };
        let name = st.mech(active.set)?.spec.name.clone();
        let levels = st.levels(active.set, &active.shape);
        let key = room_key(&name, &active.shape, &levels);
        let room = Room { delta: measured.allocated_peak.saturating_sub(active.allocated), slack: measured.slack, context: measured.context, sites: measured.sites.clone() };
        // A site whose finer level did not lower the allocated peak is not cut further here.
        for (site, level) in levels.iter().filter(|(_, l)| **l > 0) {
            let mut coarser = levels.clone();
            coarser.insert(site.clone(), level - 1);
            if self.memo(&mut st, &room_key(&name, &active.shape, &coarser))?.is_some_and(|c| room.delta >= c.delta) {
                st.stuck.insert((active.set, active.shape.clone(), site.clone()));
            }
        }
        let known = self.memo(&mut st, &key)?;
        let merged = match known {
            Some(k) => Room { delta: k.delta.max(room.delta), slack: k.slack.max(room.slack), context: k.context.max(room.context), sites: if room.sites.is_empty() { k.sites.clone() } else { room.sites.clone() } },
            None => room.clone(),
        };
        st.table.context += measured.context;
        if self.memo(&mut st, &key)?.as_ref() != Some(&merged) {
            st.rooms.insert(key.clone(), merged.clone());
            self.shared.put(&format!("room/{key}"), Some(&crate::os::encode(&merged)?))?;
            self.shared.ring();
        }
        self.publish(&st);
        Ok(())
    }

    fn oom(&self, req: u64, need: Option<u64>, site: Option<Site<'_>>) -> Result<Next> {
        let collective = site.is_some_and(|s| s.collective);
        if let (true, Some(site)) = (collective, site) {
            // Ranks agree on levels at pass start: the finer level waits for the next pass, and
            // stays recorded even if this pass is dropped.
            let mut st = self.lock();
            if let Some(a) = st.active.get(&req) {
                let k = (a.set, a.shape.clone());
                *st.cuts.entry(k).or_default().entry(site.name.to_string()).or_insert(0) += 1;
            }
        }
        let local = site.filter(|s| !s.collective).map(|s| s.name.to_string());
        let actions = self.demand(Trigger::Oom { req, need, site: local.clone(), tried: Tried::default() }, Some(req))?;
        let mut st = self.lock();
        let stats = |st: &State| st.stats.get(&req).cloned().unwrap_or_default();
        let next = match actions.iter().rev().find(|a| matches!(a, Action::Cut { .. } | Action::Park(_) | Action::Exhausted { .. })) {
            Some(Action::Cut { site, .. }) => {
                let limit = st.table.torch_limit();
                let a = st.active.get(&req).map(|a| (a.set, a.shape.clone()));
                let Some((set, shape)) = a else { return Ok(Next::Exhausted(exhausted(need.unwrap_or(0), Some(site), stats(&st)))) };
                let name = st.mech(set)?.spec.name.clone();
                let levels = st.levels(set, &shape);
                st.failed_at.insert(room_key(&name, &shape, &levels), limit);
                let level = st.cuts.entry((set, shape)).or_default().entry(site.clone()).or_insert(0);
                *level += 1;
                let level = *level;
                if let Some(s) = st.stats(Some(req)) {
                    s.cuts += 1;
                }
                Next::Cut(level)
            }
            Some(Action::Park(_)) => {
                if let Some(r) = st.table.request_mut(req) {
                    r.episode.parked = true;
                }
                Next::Park(st.table.headroom)
            }
            Some(Action::Exhausted { need: short }) => Next::Exhausted(exhausted(*short, local.as_deref(), stats(&st))),
            _ => {
                if let Some(s) = st.stats(Some(req)) {
                    s.retries += 1;
                }
                Next::Retry
            }
        };
        self.publish(&st);
        Ok(next)
    }

    fn end_request(&self, req: u64) -> Stats {
        let mut st = self.lock();
        let name = st.table.request(req).map(|r| r.name.clone()).unwrap_or_default();
        st.active.remove(&req);
        st.table.requests.retain(|r| r.id != req);
        let owned: Vec<u32> = st.table.caches.iter().filter(|c| c.owner == name).map(|c| c.id).collect();
        let shrinks: Vec<(u32, u64, Shrink)> = owned.iter().filter_map(|id| st.shrinks.get(id).map(|s| (*id, u64::MAX, s.clone()))).collect();
        let leftover: u64 = st.pins.values().filter(|(_, r)| *r == Some(req)).filter_map(|(k, _)| st.table.page(*k)).map(|p| p.bytes).sum();
        let mut stats = st.stats.remove(&req).unwrap_or_default();
        if let Some(key) = st.peaks.remove(&req) {
            let key = format!("peak/{key}");
            let known = self.shared.fact(&key).ok().flatten().and_then(|v| v.try_into().ok()).map(u64::from_le_bytes);
            if known.is_none_or(|k| k < stats.peak) {
                let _ = self.shared.put(&key, Some(&stats.peak.to_le_bytes()));
            }
        }
        drop(st);
        self.shrink(shrinks);
        let mut st = self.lock();
        st.table.caches.retain(|c| !owned.contains(&c.id));
        owned.iter().for_each(|id| {
            st.shrinks.remove(id);
        });
        stats.leftover = leftover;
        st.last = Some(stats.clone());
        self.publish(&st);
        if let Some(tier) = &self.tier {
            for m in st.sets.iter().flatten() {
                let _ = tier.use_layout(&m.spec.layout.digest, None);
            }
        }
        drop(st);
        self.shared.ring();
        stats
    }

    /// The own ask thread's answer: pages only while a request runs here; an idle process also
    /// shrinks its caches (after answering, so the asker never waits on them).
    fn answer(&self, ceiling: Class, bytes: u64) -> (u64, Vec<(u32, u64, Shrink)>) {
        let out = crate::caught(|| {
            let mut st = self.lock();
            let actions = decide(&st.table, &Trigger::Ask { ceiling, bytes });
            let (mut freed, mut shrinks, mut remote) = (0, vec![], vec![]);
            for a in &actions {
                match a {
                    Action::Unmap { page, .. } => {
                        freed += st.table.page(*page).map_or(0, |p| p.bytes);
                        self.unmap(&mut st, *page, Class::N, None, true)?;
                    }
                    Action::Shrink { cache, bytes } => {
                        freed += bytes;
                        self.apply(&mut st, a, None, &mut remote, &mut shrinks)?;
                        let _ = cache;
                    }
                    _ => self.apply(&mut st, a, None, &mut remote, &mut shrinks)?,
                }
            }
            self.publish(&st);
            Ok((freed, shrinks))
        });
        if out.is_err() {
            self.recover();
        }
        out.unwrap_or_default()
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.row.wake();
        self.jobs.lock().take();
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
        }
    }
}

fn fill_loop(queue: &Mutex<Receiver<Job>>, inner: &Weak<Inner>) {
    loop {
        let job = queue.lock().recv();
        let (Ok(job), Some(inner)) = (job, inner.upgrade()) else { return };
        inner.fill(job);
    }
}

fn ask_loop(row: &Row, inner: &Weak<Inner>, stop: &AtomicBool) {
    loop {
        let wants = row.wait_asked(stop);
        if stop.load(Ordering::Acquire) {
            return;
        }
        let Some(inner) = inner.upgrade() else { return };
        for (ceiling, bytes, asked) in wants {
            let (freed, shrinks) = inner.answer(ceiling, bytes);
            row.answer(ceiling, asked, freed);
            inner.shrink(shrinks);
        }
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
            self.inner.publish(&st);
            Ok(())
        });
    }

    /// Starts a pass of `set` at `shape` (dimensions joined by `x`, e.g. `1x81x480x832`), with
    /// torch's bytes now; returns once the pass's measured room is free.
    pub fn pass(&self, set: SetId, shape: &str, torch: Torch) -> Result<Pass> {
        self.inner.guard(|| self.inner.pass(self.id, set.0, shape, torch))?;
        Ok(Pass { inner: self.inner.clone(), req: self.id })
    }

    /// An op ran out of memory, needing `need` more bytes when the error said; `site` is its
    /// cut site, `None` when the op cannot be cut.
    pub fn oom(&self, need: Option<u64>, site: Option<Site<'_>>) -> Next {
        match self.inner.guard(|| self.inner.oom(self.id, need, site)) {
            Ok(next) => next,
            Err(Error::Exhausted(e)) => Next::Exhausted(e),
            Err(_) => Next::Exhausted(exhausted(need.unwrap_or(0), site.map(|s| s.name), self.stats())),
        }
    }

    /// What the caller parked after a `Park` answer; 0 makes the next `oom` answer Exhausted.
    pub fn parked(&self, bytes: u64) {
        let mut st = self.inner.lock();
        if let Some(s) = st.stats(Some(self.id)) {
            s.parks += 1;
            s.parked += bytes;
        }
        if let Some(r) = st.table.request_mut(self.id) {
            r.episode.parked = bytes == 0;
        }
    }

    /// The current pass's cut level for op site `site` (0: whole).
    pub fn cut(&self, site: &str) -> u32 {
        let st = self.inner.lock();
        st.active.get(&self.id).and_then(|a| st.cuts.get(&(a.set, a.shape.clone()))).and_then(|c| c.get(site)).copied().unwrap_or(0)
    }

    pub fn stats(&self) -> Stats {
        self.inner.lock().stats.get(&self.id).cloned().unwrap_or_default()
    }

    /// Ends the request, returning its record.
    pub fn end(mut self) -> Stats {
        self.ended = true;
        self.inner.end_request(self.id)
    }
}

impl Drop for Request {
    fn drop(&mut self) {
        if !self.ended {
            self.inner.end_request(self.id);
        }
    }
}

impl Pin {
    /// The caller's fill transform completed: the block may now be kept.
    pub fn filled(&self) {
        self.transformed.store(true, Ordering::Release);
        let mut st = self.inner.lock();
        if let Some(h) = st.homes.get_mut(&self.key) {
            h.filled = true;
            h.transforming = false;
        }
        st.sync(self.key);
        self.inner.keep(&mut st, self.key);
        drop(st);
        self.inner.landed.notify_all();
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
    /// Records what the pass measured.
    pub fn end(self, measured: Measured) {
        let _ = self.inner.guard(|| self.inner.end_pass(self.req, measured));
    }
}

impl Drop for Pass {
    fn drop(&mut self) {
        self.inner.state.lock().active.remove(&self.req);
    }
}

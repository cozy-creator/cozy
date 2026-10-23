//! One executor's memory on one device. One budget: the ledger's capacity less the other rows;
//! one loop, `make_room`, takes bytes in [`crate::decide::victims`]' order for the allocator
//! hook, a bind and a range's preparation. One `parking_lot` mutex guards the table and the
//! mechanism's state; it is never held across a fill, a remote call, a wait on an event or a
//! caller's callback: custody calls made while it is held are posted, in order, to one sender
//! thread.
//!
//! Lock order: none. The state lock is only ever taken alone; every other lock (the release
//! receiver, the fill queue, the trim wants, the custody client's stream) is taken alone or inside the state lock and never takes another, and the state lock is
//! never taken while one of them is held. A binder waits on the state lock's condvar only. Fills run
//! on fill threads (demand misses before prefetches); pin releases arrive through a channel;
//! asks are answered from pages by a thread of its own that runs no caller code; an idle
//! process's trim and cache shrinks run on its trim thread. A panic is caught at every public
//! call and on every thread: the call fails `Internal`, the table is derived again from the
//! mappings, and the process keeps serving.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use crossbeam_channel::{select_biased, unbounded, Receiver, Sender};

use crate::outbox::Outbox;
use std::time::Instant;

use parking_lot::{Condvar, Mutex, MutexGuard};
use platform::handles::Owned;

#[cfg(unix)]
use tensorfs::io::source::Placement;
use tensorfs::io::Dst;

use crate::backend::{Access, Backend, Event, Physical, Va};
use crate::custody;
use crate::decide::{self, Victim};
use crate::ledger::{self, Facts, Grant, PhysicalChange, Row, Shared};
use crate::table::{Class, Fill, Key, Level, Page, Plan, Req, Set, SetSpec, Table};
use crate::{Error, Exhausted, GpuId, Result};

mod stream;
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
        self.inner.keeps(&mut st);
        self.inner.close_sets(&mut st);
        self.inner.publish(&mut st);
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
                self.inner.publish(&mut st);
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
        self.inner.publish(&mut st);
    }
}

/// One committed external allocation. The allocator retains this token until checked
/// physical release; quarantined allocations retain it until executor exit.
pub struct Resident {
    inner: Owner,
    bytes: u64,
}
impl Resident {
    /// Marks its free before the driver call ([`Memory::physical_change`]); finish after this
    /// token drops, and drop it unfinished if the free failed.
    pub fn releasing(&self) -> PhysicalChange {
        self.inner.row.physical_change()
    }
}

impl Drop for Resident {
    fn drop(&mut self) { self.inner.native_resident.fetch_sub(self.bytes, Ordering::AcqRel); }
}

/// Room [`Memory::make_room`] made for the allocator hook's segment, held for it alone: no
/// other mapper takes these bytes. [`Claim::resident`] turns it into the segment's count once
/// the driver allocated it; dropped unused, the bytes are free again.
pub struct Claim {
    inner: Owner,
    bytes: u64,
}

impl Claim {
    /// The segment is allocated: its bytes are counted resident from now on, never twice.
    pub fn resident(mut self) -> Resident {
        let bytes = std::mem::take(&mut self.bytes);
        self.inner.native_resident.fetch_add(bytes, Ordering::AcqRel);
        self.inner.claimed.fetch_sub(bytes, Ordering::AcqRel);
        Resident { inner: self.inner.clone().into(), bytes }
    }
}

impl std::fmt::Debug for Claim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Claim({})", self.bytes)
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        self.inner.claimed.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

/// A registered cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CacheId(pub u32);

/// Where a bound block's bytes came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Origin {
    /// Already mapped and filled.
    Hit,
    /// A kept page imported by fingerprint: no copy.
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
    /// Kept pages imported by fingerprint (no copy; `ns` is the import).
    pub kept: Fills,
    pub store: Fills,
    /// Of the above, bytes filled speculatively.
    pub speculative: u64,
}

/// One step of a request.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct StepStats {
    /// The budget the step ran against (the ledger's capacity less foreign bytes and the other
    /// rows), read at its start.
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
    /// The most it held at a pass end: pages and the hook's segments (the peak its passes
    /// reported, when more).
    pub peak: u64,
    /// The most its passes allocated beside its weights, as they reported it: its compute bytes.
    pub compute: u64,
    /// The process's anonymous host high-water over it (file mappings aside); `None` where it
    /// cannot be read.
    pub host: Option<u64>,
    /// Bytes the request still held when it ended (pins, caches): 0.
    pub leftover: u64,
    /// Bytes due to be kept (read-only, exported to custody) and not kept yet: pages not idle
    /// (unpinned, no release pending) since their fill. Each is kept at its first idle moment.
    pub keep_pending: u64,
    /// Each `make_room` that had to take bytes for it, in order.
    pub make_room: Vec<MakeRoom>,
}

/// One `make_room` that had to take bytes: what asked, how many bytes, what it took.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct MakeRoom {
    /// The request's plan step then.
    pub step: u32,
    /// The allocator hook, a bind or a range's preparation.
    pub trigger: &'static str,
    /// The free bytes asked for.
    pub need: u64,
    /// The bytes its rounds took (pages unmapped, caches shrunk).
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

/// One process's memory on one device.
pub struct Memory {
    inner: Owner,
}

/// A request in flight; dropping it ends it: its pins and its caches go, and the leftover is
/// measured.
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

/// A cache's shrink: called with no lock held (it may call back in) with the bytes wanted;
/// returns the bytes it freed.
type Shrink = Arc<dyn Fn(u64) -> u64 + Send + Sync>;

/// A hold on a cache ([`Memory::hold_cache`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HoldId(pub u64);

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

/// The table's pin for a page whose Keep's protect runs (no caller holds it).
const PROTECTING: u64 = u64::MAX;

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
    exported: bool,
    origin: Origin,
    speculative: bool,
    /// The fill identity its bytes were made under ([`crate::Fingerprint::key`]); empty: its
    /// bytes are not the weights (a transform released undone), so it goes once unpinned.
    identity: String,
    /// Keep (read-only, exported) once `ready` completes: a write may still be running.
    keep_after: bool,
    /// Its Keep (read-only, export) is due: queued for its set, or waiting for the page to
    /// be idle.
    keeping: bool,
    /// Its Keep's protect and export run now. No work may touch a page while its access
    /// changes (`cuMemSetAccess` under a reading kernel faults it): binds wait.
    protecting: bool,
    /// The allocator hook unmaps it in place now, on its own thread (never queued behind the
    /// mechanism's other work): nobody else touches it meanwhile.
    reclaiming: bool,
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

/// A call to the device's virtual-memory mechanism (`cuMemCreate`, `cuMemMap`,
/// `cuMemSetAccess`, `cuMemUnmap`): each may wait on the host for queued device work, so none
/// runs under the state lock. One thread runs them in the order decided (an eviction's unmap
/// before the create its bytes go to); each result re-enters the state.
enum Mech {
    /// Create a page, map it read-write, then queue its fill; `decided` is the free bytes the
    /// decision saw.
    Map { key: Key, gen: u64, va: Va, off: u64, span: u64, job: Job, speculative: bool, decided: u64 },
    /// Map an imported kept page read-only.
    Adopt { key: Key, gen: u64, va: Va, off: u64, phys: u64, len: u64, identity: String },
    /// Unmap a page (the driver lets its consumers finish) and destroy it.
    Unmap { va: Va, off: u64, span: u64, phys: Option<Physical>, ready: Event },
    /// Make the set's due pages that are idle read-only (each contiguous run in one call) and
    /// hand their exports to custody.
    Keep { set: u32 },
    /// A read-only kept page whose fill identity changed: never protected again, it is
    /// unmapped (after its consumers) and a fresh page filled in its place.
    Renew { key: Key, gen: u64, va: Va, off: u64, span: u64, old: Option<Physical>, after: Vec<Event>, job: Job },
    /// A streamed set's slot moves from the block at `from` to `key`'s block, after the old block's consumers
    /// (`after`); then `key`'s fill is queued.
    Move { key: Key, gen: u64, va: Va, from: u64, off: u64, span: u64, phys: Physical, after: Vec<Event>, job: Job },
    /// A closed set's reservation, after its pages' unmaps.
    Release(Va),
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

/// A custody call made under the lock, sent in order once it is released.
enum Call {
    Export { fp: String, page: u32, len: u64, handle: Owned },
    Withdraw { fp: String, page: u32 },
    Disown { fp: String, page: u32 },
}

/// Which pending events [`Inner::finish_pending`] waits for.
#[derive(Clone, Copy)]
enum Pending {
    /// Every page's, ready events included (a retirement).
    All,
    /// The pages' whose Keep is due ([`Memory::settle`]).
    Due,
    /// The ready events of copies a Keep waits for (the trim thread's).
    Landing,
}

/// What the trim thread runs: after an answer, custody settled for the kept pages it withdrew,
/// then the bytes published free; a Keep waiting on a copy still landing (no request open),
/// made once it lands with no call to drive it.
#[derive(Default)]
struct Wanted {
    settle: bool,
    keep: bool,
    closed: bool,
}

struct Retirement {
    bytes: u64,
    created: bool,
    mechanism: Option<u64>,
    custody: Option<u64>,
    kept: bool,
}

#[derive(Default)]
struct State {
    /// The wake epoch read when this lock was taken, before its drain: a wait sleeps only while
    /// no change came after it.
    seen: u64,
    table: Table,
    physical_epoch: u64,
    retired: HashMap<(Key, u64), Retirement>,
    sets: Vec<Option<SetMech>>,
    homes: HashMap<Key, Home>,
    /// Pin → (page, request).
    /// Pin → (page, request, the thread that holds it).
    pins: HashMap<u64, (Key, Option<u64>, std::thread::ThreadId)>,
    /// Each stream's pending events (a page's release, or the ready event a Keep waits for),
    /// in the order recorded: one stream's events complete in order, so a drain queries only
    /// each stream's oldest.
    events: BTreeMap<usize, VecDeque<(Event, Key)>>,
    /// Sets streaming their upper blocks through slots (step 2's watermark), by set.
    streams: HashMap<u32, Streaming>,
    /// Ids and the logical clock.
    clock: u64,
    shrinks: HashMap<u32, Shrink>,
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
    /// Request → its recorded anonymous host peak (`None`: unmeasured).
    host: HashMap<u64, Option<u64>>,
    /// This process's anonymous peak while requests run.
    window: Option<ledger::Window>,
    /// Request → the `<package>|<shape>` its peak is recorded under.
    peaks: HashMap<u64, String>,
    /// Requests whose last step is one their plan did not declare.
    synthetic: HashSet<u64>,
    /// The custody connection the kept pages were exported over.
    holder: u64,
    /// Pages whose Keep is due ([`Home::keeping`]), and sets with a Keep queued.
    keep_due: BTreeSet<Key>,
    keep_queued: BTreeSet<u32>,
}

/// A page the allocator hook takes in place, unmapped and destroyed on its own thread: (page, its generation, class,
/// place, backing, ready event).
type Taking = (Key, u64, Class, (Va, u64, u64), Physical, Event);

impl State {
    /// Marks `key` the hook's to take in place (nobody else touches it meanwhile) and takes its backing out of its
    /// home; `None` when it is not the hook's to take (kept, being kept, protected, already taken, or unbacked).
    fn take_in_place(&mut self, key: Key) -> Result<Option<Taking>> {
        let (place, class) = (self.place(key)?, self.table.class(key).unwrap_or(Class::N));
        // Never a page whose copies may still be landing (read ready at issue).
        let Some(h) = self.homes.get_mut(&key).filter(|h| !h.keeping && !h.keep_after && !h.protecting && !h.reclaiming && !h.exported && !h.landing) else { return Ok(None) };
        let Some(phys) = h.phys.take() else { return Ok(None) };
        h.reclaiming = true;
        let taking = (key, h.gen, class, place, phys, h.ready);
        self.sync(key);
        Ok(Some(taking))
    }

    /// No work can touch the page: unpinned, no release event pending, not filling or being
    /// transformed.
    fn idle(&self, key: Key) -> bool {
        self.table.page(key).is_some_and(|p| p.pins.is_empty() && p.releasing.is_empty())
            && self.homes.get(&key).is_some_and(|h| !h.filling && !h.transforming)
    }

    /// Bytes due to be kept and not kept yet.
    fn keep_pending(&self) -> u64 {
        let protecting = self.homes.iter().filter(|(_, h)| h.protecting).map(|(k, _)| *k);
        self.keep_due.iter().copied().chain(protecting).filter_map(|k| self.table.page(k)).map(|p| p.bytes).sum()
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
        if !self.homes.get(&key).is_some_and(|h| h.keeping) {
            self.keep_due.remove(&key);
        }
        let home = self.homes.get(&key).map(|h| (h.filled, h.filling, h.speculative, h.exported));
        let mut pins: Vec<u64> = self.pins.iter().filter(|(_, (k, ..))| *k == key).map(|(id, _)| *id).collect();
        // A protect in flight pins the page (P): nothing evicts or unmaps it meanwhile.
        if self.homes.get(&key).is_some_and(|h| h.protecting || h.reclaiming) {
            pins.push(PROTECTING);
        }
        if let Some(p) = self.table.page_mut(key) {
            p.mapped = home.is_some();
            p.filled = home.is_some_and(|h| h.0);
            p.filling = home.is_some_and(|h| h.1);
            p.kept = home.is_some_and(|h| h.3);
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
    backend: Box<dyn Backend>,
    shared: Arc<Shared>,
    row: Arc<Row>,
    custody: Option<Arc<custody::Client>>,
    state: Mutex<State>,
    /// Every change a state waiter may test, from any thread (the allocator callback and a pin's
    /// drop hold no state lock): an epoch the waiter read before its last check.
    landed: Wakes,
    /// Custody's calls, in the order they were made, for one sender thread.
    to_custody: Arc<Outbox<Call>>,
    /// The device's mechanism calls, for its one thread; their bytes are the unmaps in flight.
    mech: Arc<Outbox<Mech>>,
    release_tx: Sender<Release>,
    releases: Receiver<Release>,
    /// Fill jobs; the fill threads hold the receiving ends and a weak reference to the rest,
    /// so dropping the Memory ends them. The trim thread's work likewise.
    jobs: Jobs,
    /// Pages the mechanism thread mapped and has not handed to the fill threads yet (demand,
    /// ahead): it hands them over together once its queue is empty or they make a batch.
    grouped: Mutex<(Vec<Job>, Vec<Job>)>,
    burst: Arc<(Mutex<Burst>, Condvar)>,
    wanted: Arc<(Mutex<Wanted>, Condvar)>,
    stop: Arc<AtomicBool>,
    quiet: Arc<Quiet>,
    ownership_lost: AtomicBool,
    public_owners: AtomicU64,
    /// The allocator hook's segments ([`Memory::resident`]), counted exactly, and the room its
    /// claims hold for segments not allocated yet.
    native_resident: AtomicU64,
    claimed: AtomicU64,
    /// The budget less this row's own bytes but the hook's segments, as the last publication
    /// computed it: [`Memory::free`] reads it without the state lock.
    room: AtomicU64,
    #[cfg(unix)]
    tally: tensorfs::io::source::IoTally,
    _device: Device,
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
            // A row is a process: one Memory per device per process.
            let device = Device::open(&gpu.uuid)?;
            let grant = match grant {
                Some(g) => g,
                None => Grant::open(&gpu, None)?,
            };
            let capacity = gpu.memory.min(backend.capacity());
            let shared = Arc::new(Shared::adopt(grant.table, &gpu, capacity)?);
            shared.reap(None)?;
            let row = Arc::new(shared.claim_with(std::process::id(), grant.row)?);
            let listener = row.listen()?;
            let custody = custody::Client::open(&shared).map_err(|e| tracing::warn!("varena: no custody: {e}")).ok().map(Arc::new);
            let (to_custody, mech) = (Arc::new(Outbox::default()), Arc::new(Outbox::default()));
            let (release_tx, releases) = unbounded();
            let ((demand, demanded), (ahead, wanted_ahead)) = (unbounded(), unbounded());
            let holder = custody.as_ref().map_or(0, |c| c.holder());
            let context = backend.process_used().unwrap_or_else(|_| backend.opened_context());
            let table = Table { capacity: shared.capacity()?, limit: grant.limit, context, ..Table::default() };
            let inner = Arc::new(Inner {
                backend,
                shared,
                row: row.clone(),
                _device: device,
                custody,
                state: Mutex::new(State { table, holder, ..State::default() }),
                landed: Wakes::default(),
                to_custody,
                mech,
                release_tx,
                releases,
                jobs: Jobs { demand, ahead },
                grouped: Mutex::default(),
                burst: Arc::new((Mutex::new(Burst::default()), Condvar::new())),
                wanted: Arc::new((Mutex::new(Wanted::default()), Condvar::new())),
                stop: Arc::new(AtomicBool::new(false)),
                quiet: Arc::default(),
                ownership_lost: AtomicBool::new(false),
                public_owners: AtomicU64::new(0),
                native_resident: AtomicU64::new(0),
                claimed: AtomicU64::new(0),
                room: AtomicU64::new(0),
                #[cfg(unix)]
                tally: Default::default(),
            });
            let (weak, queue) = (Arc::downgrade(&inner), inner.mech.clone());
            std::thread::Builder::new().name("varena-mechanism".into()).spawn(move || mech_loop(&queue, &weak)).map_err(|e| Error::io("spawn mechanism", e))?;
            let (weak, outbox, c) = (Arc::downgrade(&inner), inner.to_custody.clone(), inner.custody.clone());
            std::thread::Builder::new().name("varena-custody-calls".into()).spawn(move || send_loop(&outbox, c.as_deref(), &weak)).map_err(|e| Error::io("spawn calls", e))?;
            // One fill thread per copy stream TensorFS runs per device: a fill waits on its
            // streams, so this many keep them busy.
            for _ in 0..tensorfs::io::ReadOpts::default().streams.max(1) {
                let (weak, demand, ahead) = (Arc::downgrade(&inner), demanded.clone(), wanted_ahead.clone());
                std::thread::Builder::new().name("varena-fill".into()).spawn(move || fill_loop(&demand, &ahead, &weak)).map_err(|e| Error::io("spawn fill", e))?;
            }
            // The row is the Memory's alone: it goes (publishing zeros) the moment the Memory does.
            let (weak, stop, row) = (Arc::downgrade(&inner), inner.stop.clone(), Arc::downgrade(&row));
            std::thread::Builder::new().name("varena-ask".into()).spawn(move || ask_loop(listener, &row, &weak, &stop)).map_err(|e| Error::io("spawn ask", e))?;
            let (weak, wanted) = (Arc::downgrade(&inner), inner.wanted.clone());
            std::thread::Builder::new().name("varena-trim".into()).spawn(move || trim_loop(&wanted, &weak)).map_err(|e| Error::io("spawn trim", e))?;
            let mut st = inner.lock();
            inner.publish(&mut st);
            inner.unlock(st);
            Ok(Memory { inner: inner.into() })
        })
    }

    /// Registers a set and brings it to `level`; a set loaded again is re-levelled in place. On
    /// every start, first or restart, its pages kept under its fingerprint are imported, and
    /// the rest fill from the store, through the page cache. A GPU load with no request open
    /// returns with its pages custody's ([`Memory::settle`]).
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

    /// Registers bytes the caller holds (torch tensors) that `shrink(want) -> freed` gives back:
    /// `make_room` takes a cache nobody holds after other sets' idle pages, ranked with the
    /// current step's pages by next use, its components' (`sets`: a graph pool names its
    /// component; none, no declared use), a page before it on a tie (never inside the
    /// allocator: its shrink frees through it). A cache whose `owner` is a request id ends with
    /// that request.
    pub fn cache(&self, owner: &str, sets: &[SetId], shrink: impl Fn(u64) -> u64 + Send + Sync + 'static) -> CacheId {
        let shrink: Shrink = Arc::new(shrink);
        self.inner
            .guard(|| {
                let mut st = self.inner.lock();
                let id = st.tick() as u32;
                let last_use = st.clock;
                let sets = sets.iter().map(|s| s.0).collect();
                st.table.caches.push(crate::table::Cache { id, owner: owner.to_string(), sets, last_use, ..Default::default() });
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
            if let Some(c) = st.table.cache_mut(cache.0) {
                c.last_use = now;
                match c.shrinking {
                    0 => c.bytes = bytes,
                    _ => c.reported = Some(bytes),
                }
            }
            self.inner.publish(&mut st);
            self.inner.unlock(st);
            self.inner.landed.notify_all();
            Ok(())
        });
    }

    /// Holds a cache (a graph pool from a replay's preparation): never shrunk until released.
    /// Holds nest.
    pub fn hold_cache(&self, cache: CacheId) -> HoldId {
        self.inner
            .guard(|| {
                let mut st = self.inner.lock();
                let id = st.tick();
                if let Some(c) = st.table.cache_mut(cache.0) {
                    c.holds += 1;
                }
                self.inner.publish(&mut st);
                self.inner.unlock(st);
                Ok(HoldId(id))
            })
            .unwrap_or(HoldId(0))
    }

    /// Ends a hold once the work queued on `stream` so far completes (a replay and every
    /// reader of its lent outputs, joined into it): until then the cache stays held.
    pub fn release_cache(&self, cache: CacheId, _hold: HoldId, stream: usize) {
        let _ = self.inner.guard(|| {
            let event = self.inner.backend.record(stream)?;
            let mut st = self.inner.lock();
            if let Some(c) = st.table.cache_mut(cache.0) {
                c.holds = c.holds.saturating_sub(1);
                if event != Event::DONE {
                    c.releasing.push(event.0);
                }
            }
            self.inner.publish(&mut st);
            self.inner.unlock(st);
            Ok(())
        });
    }

    pub fn uncache(&self, cache: CacheId) {
        let _ = self.inner.guard(|| {
            let mut st = self.inner.lock();
            st.table.caches.retain(|c| c.id != cache.0);
            st.shrinks.remove(&cache.0);
            self.inner.publish(&mut st);
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

    /// Marks a physical operation (a native segment's allocation) before its driver call:
    /// until it finishes, this row's physical counts are no evidence for foreign bytes. Only
    /// atomics: safe inside the allocator. Finish once the outcome is certain and counted
    /// (after [`Memory::resident`], or after a failed call that allocated nothing); an uncertain
    /// one is dropped unfinished and keeps the row's counts unproven.
    pub fn physical_change(&self) -> PhysicalChange {
        self.inner.row.physical_change()
    }

    /// Records a successful native segment allocation. Call only after the driver succeeds;
    /// drop the returned token only after its physical free succeeds, outside allocator locks.
    pub fn resident(&self, bytes: u64) -> Resident {
        let inner: Owner = self.inner.clone().into();
        inner.native_resident.fetch_add(bytes, Ordering::AcqRel);
        Resident { inner, bytes }
    }

    /// Bytes this row may take now without taking any: the budget (the ledger's capacity less
    /// foreign bytes, orphans and the other rows, within its own cap) less its context, pages
    /// and the hook's segments. Never waits for the state: the last publication's figure less
    /// the segments counted since. Only atomics: safe inside the allocator.
    pub fn free(&self) -> u64 {
        if self.inner.check_ownership().is_err() {
            return 0;
        }
        let held = self.inner.native_resident.load(Ordering::Acquire) + self.inner.claimed.load(Ordering::Acquire);
        self.inner.room.load(Ordering::Acquire).saturating_sub(held)
    }

    /// The allocator hook's room for a `bytes` segment, held for it ([`Claim`]): taken in
    /// [`decide::victims`]' order from its own idle pages that are not kept (never a pinned page;
    /// one whose release is pending once it completes, waited on here). Local work only: no
    /// cache shrink (Python, under torch's lock), no custody or peer call; those run at binds
    /// and pass starts. `Exhausted` once a round takes nothing. Synchronous, from any thread,
    /// without the GIL.
    pub fn make_room(&self, bytes: u64) -> Result<Claim> {
        self.inner.guard(|| self.inner.make_room(bytes, &[], "hook", None, false))?;
        Ok(Claim { inner: self.inner.clone().into(), bytes })
    }

    /// Blocks until every Keep due so far is made (waiting for its page's pending events), and
    /// every mechanism call and every custody call decided so far has been made: then the kept
    /// pages are custody's (a planned handover), and a dropped Memory's row is gone when the drop
    /// returns.
    pub fn settle(&self) {
        self.inner.settle();
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
        self.publish(&mut st);
        self.unlock(st);
        self.landed.notify_all();
    }

    /// The state, with pin releases applied and completed events acted on.
    fn lock(&self) -> MutexGuard<'_, State> {
        let mut st = self.state.lock();
        st.seen = self.landed.seen();
        self.physical(&mut st);
        self.drain(&mut st);
        st
    }

    fn unlock(&self, st: MutexGuard<'_, State>) {
        drop(st);
    }

    /// A custody call, sent after everything posted before it.
    fn post(&self, call: Call) {
        self.to_custody.post(call, 0);
    }

    /// Withdraws kept page `page` of `fp`: an Export not yet sent is cancelled (nothing was
    /// kept, and its handle goes); else a Withdraw is queued, `bytes` still held until it is
    /// sent (custody's handle keeps the memory). Its ticket, `None` when cancelled.
    fn withdraw(&self, fp: String, page: u32, bytes: u64) -> Option<u64> {
        let queued = |c: &Call| matches!(c, Call::Export { fp: f, page: p, .. } if *f == fp && *p == page);
        if self.to_custody.cancel(queued) {
            return None;
        }
        Some(self.to_custody.post(Call::Withdraw { fp, page }, bytes))
    }

    /// Waits, the state unlocked, for a change after the caller's last check (a fill, a release,
    /// a reclaim), then reconciles and drains.
    fn wait(&self, st: &mut MutexGuard<'_, State>) -> Result<()> {
        self.check_ownership()?;
        let seen = st.seen;
        MutexGuard::unlocked(st, || self.landed.wait_past(seen));
        st.seen = self.landed.seen();
        self.check_ownership()?;
        self.physical(st);
        self.drain(st);
        Ok(())
    }

    fn poll_releases(&self, st: &mut State, keep: bool) -> bool {
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
                self.completed(st, event, key, keep);
            }
        }
        st.events.retain(|_, q| !q.is_empty());
        wake
    }

    /// `key`'s pending `event` completed: it leaves the page's releases (retired unless it is
    /// the page's ready event), and a Keep waiting on it falls due.
    fn completed(&self, st: &mut State, event: Event, key: Key, keep: bool) {
        let ready = st.homes.get(&key).map(|h| h.ready);
        if let Some(i) = st.table.page(key).and_then(|p| p.releasing.iter().position(|e| *e == event.0)) {
            if let Some(p) = st.table.page_mut(key) {
                p.releasing.swap_remove(i);
            }
            if Some(event) != ready {
                self.backend.retire(event);
            }
        }
        // A transform or fill whose event completed: now read-only and kept.
        if keep && st.homes.get(&key).is_some_and(|h| h.keep_after && h.ready == event) {
            if let Some(h) = st.homes.get_mut(&key) {
                h.keep_after = false;
            }
            self.keep(st, key);
        }
    }

    /// [`Memory::settle`].
    fn settle(&self) {
        self.hand_over();
        self.mech.settle();
        self.to_custody.settle();
    }

    /// A planned handover ([`Memory::settle`]): every Keep due so far is queued. A due page
    /// waits on its pending events (its consumers' releases, or its fill's): this call takes
    /// them from the drain under a pin of its own, so nothing else retires or evicts them,
    /// waits for them unlocked, and completes them as a drain would.
    fn hand_over(&self) {
        if let Err(e) = self.finish_pending(Pending::Due) {
            tracing::warn!("varena: a handover's events are unproven: {e}");
        }
    }

    /// Waits for pending events (host waits) and completes them as a drain would: those `which`
    /// names. Their pages are pinned meanwhile, so nothing retires the events or evicts the pages.
    fn finish_pending(&self, which: Pending) -> Result<()> {
        let (pins, taken, ready) = {
            let mut st = self.lock();
            let due: BTreeSet<Key> = st.keep_due.iter().copied().chain(st.homes.iter().filter(|(_, h)| h.keep_after).map(|(k, _)| *k)).collect();
            let landing: Vec<(Key, Event)> = st.homes.iter().filter(|(_, h)| h.keep_after).map(|(k, h)| (*k, h.ready)).collect();
            let mut taken = vec![];
            for (stream, q) in &mut st.events {
                q.retain(|&(event, key)| {
                    let mine = match which {
                        Pending::All => true,
                        Pending::Due => due.contains(&key),
                        Pending::Landing => landing.contains(&(key, event)),
                    };
                    if mine {
                        taken.push((*stream, event, key));
                    }
                    !mine
                });
            }
            let ready: Vec<(Key, Event)> = match which {
                Pending::All => st.homes.iter().filter(|(_, h)| !self.backend.query(h.ready).unwrap_or(false)).map(|(k, h)| (*k, h.ready)).collect(),
                Pending::Due | Pending::Landing => vec![],
            };
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
                true => self.completed(&mut st, event, key, true),
                // Unproven: still pending, for the drain.
                false => st.events.entry(stream).or_default().push_back((event, key)),
            }
        }
        self.drain(&mut st);
        self.publish(&mut st);
        self.landed.notify_all();
        // Due copies another waiter took (the trim thread): their Keeps are queued once it
        // completes them.
        let queued = |st: &State, key: Key, ready: Event| st.events.values().any(|q| q.contains(&(ready, key)));
        while matches!(which, Pending::Due) && st.homes.iter().any(|(k, h)| h.keep_after && !queued(&st, *k, h.ready)) {
            if self.wait(&mut st).is_err() {
                break;
            }
        }
        self.unlock(st);
        match unproven {
            0 => ready,
            n => Err(Error::Ownership(format!("{n} pending events could not be waited on"))),
        }
    }

    fn drain(&self, st: &mut State) {
        let mut wake = self.poll_releases(st, true);
        // A held cache's release completes on its stream: it is its class again.
        for c in &mut st.table.caches {
            let before = c.releasing.len();
            c.releasing.retain(|e| {
                let done = self.backend.query(Event(*e)).unwrap_or(false);
                if done {
                    self.backend.retire(Event(*e));
                }
                !done
            });
            wake |= c.releasing.len() != before;
        }
        self.keeps(st);
        if let Some(h) = self.custody.as_ref().map(|c| c.holder()).filter(|h| *h != st.holder) {
            // A new custody holder knows none of the kept pages: all go to it again.
            st.holder = h;
            let kept: Vec<Key> = st.homes.iter_mut().filter(|(_, h)| h.exported).map(|(k, h)| {
                h.exported = false;
                *k
            }).collect();
            kept.into_iter().for_each(|k| self.keep(st, k));
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
        m.key = m.spec.fingerprint.key();
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
            if st.mech(set).is_ok_and(|m| m.retained != 0) {
                continue;
            }
            if let Some(m) = st.sets.get_mut(set as usize).and_then(Option::take) {
                self.mech.post(Mech::Release(m.va), 0);
            }
            if let Some(slot) = st.table.sets.get_mut(set as usize) {
                *slot = None;
            }
        }
    }

    /// The device's figures (this process's context, the driver's free bytes) sampled, and the
    /// rows read: only where they change a decision (a refusal, a peer ask, a request's open),
    /// never per pass.
    fn refresh(&self, st: &mut State) -> Result<ledger::Admission<'_>> {
        self.measure(st);
        // Device queries are outside the shared metadata transaction. Revalidate their
        // physical attribution against the authoritative rows after taking admission.
        let sample = self.shared.physical_sample()?;
        let card = self.backend.capacity();
        let used = card.saturating_sub(self.backend.free()?);
        let (admission, rows) = self.read_rows(st)?;
        // This row's own bytes are known under the state lock: a retiring page stays counted
        // (and held) until its unmap returns, so a driver free before that hides nothing; a
        // create not yet recorded is not counted, so it reads as foreign meanwhile.
        let own = st.table.physical;
        let (known, proven) = (sample.known(&rows, self.row.id()) + own, sample.complete(&rows, self.row.id()));
        // Foreign bytes take from the card, not from the table's cap: they count only beyond the
        // card's room outside the cap (capacity = min(cap, card - foreign)).
        st.table.foreign = st.table.foreign_from_known(card, used, known);
        // A row's change in flight makes this figure conservative for this decision only: the
        // shared figure (TensorD's admission) changes on proven samples alone.
        if proven {
            self.shared.set_foreign(st.table.foreign)?;
        }
        Ok(admission)
    }

    /// The other rows, orphans, capacity and mode, read now (shared memory only); foreign bytes
    /// and the context as last sampled.
    fn rows(&self, st: &mut State) -> Result<ledger::Admission<'_>> {
        self.read_rows(st).map(|(admission, _)| admission)
    }

    fn read_rows(&self, st: &mut State) -> Result<(ledger::Admission<'_>, Vec<ledger::RowView>)> {
        let admission = self.shared.admission()?;
        let mut rows = admission.rows()?;
        if rows.iter().any(|r| !r.live) && self.shared.reap(None)? {
            rows = admission.rows()?;
        }
        let (others, orphans) = ledger::remotes(&rows, self.row.at());
        st.table.others = others;
        st.table.orphans = orphans;
        st.table.capacity = self.shared.capacity()?;
        st.table.shared = self.shared.shared_mode()?;
        st.table.pending = self.to_custody.pending();
        self.physical(st);
        // Unmaps in flight are this process's still (and already spent on the creates queued
        // after them).
        st.table.inflight = self.mech.pending();
        Ok((admission, rows))
    }

    fn facts(st: &State) -> Facts {
        let t = &st.table;
        let remaining: Vec<Option<u64>> = t.requests.iter().map(|r| t.remaining(r)).collect();
        let idle = t.requests.is_empty();
        Facts {
            physical: t.physical,
            physical_epoch: st.physical_epoch,
            context: t.context,
            torch: t.segments,
            caches: if idle { t.caches.iter().map(|c| c.bytes).sum::<u64>() } else { 0 },
            classes: t.classes(),
            remaining: remaining.iter().flatten().sum(),
            unmeasured: remaining.iter().filter(|r| r.is_none()).count() as u64,
            running: t.requests.len() as u64,
            outside: t.outside(),
            yielded: t.yielded_by_family(),
            host_peak: st.host.values().flatten().copied().max().unwrap_or(0),
            host_unmeasured: st.host.values().filter(|h| h.is_none()).count() as u64,
            returning: t.returning,
        }
    }

    /// The row's facts, and the room [`Memory::free`] reads without the lock.
    fn publish(&self, st: &mut State) {
        self.physical(st);
        let hook = st.table.segments + st.table.claimed;
        self.room.store(st.table.free().saturating_add(hook), Ordering::Release);
        self.row.publish(&Self::facts(st));
    }

    /// Physical backing and promises have separate owners. A page retiring through both
    /// VMM and custody is one allocation; it stays charged until both receipts complete.
    fn physical(&self, st: &mut State) {
        let epoch = self.row.physical_epoch(); // sampled BEFORE these physical counts
        let quarantined = self.ownership_lost.load(Ordering::Acquire);
        if !quarantined {
            st.retired.retain(|_, r| r.kept || r.mechanism.is_some_and(|t| !self.mech.completed(t)) || r.custody.is_some_and(|t| !self.to_custody.completed(t)));
        }
        st.table.vmm = st.homes.values().filter_map(|h| h.phys.as_ref()).map(|p| p.len).sum::<u64>()
            + st.retired.values().map(|r| r.bytes).sum::<u64>();
        st.table.mapping = st.homes.iter().filter(|(_, h)| h.phys.is_none()).filter_map(|(k, _)| st.table.page(*k)).map(|p| p.bytes).sum();
        st.table.returning = if quarantined { 0 } else { st.retired.values().filter(|r| !r.kept).map(|r| r.bytes).sum() };
        st.table.segments = self.native_resident.load(Ordering::Acquire);
        st.table.claimed = self.claimed.load(Ordering::Acquire);
        st.table.physical = st.table.context.saturating_add(st.table.segments).saturating_add(st.table.vmm);
        st.physical_epoch = epoch;
    }

    /// The context: this process's device bytes less the hook's segments and the pages it
    /// created (an imported page is not charged to the importer's process); torch's own pool
    /// outside the hook counts here.
    fn measure(&self, st: &mut State) {
        st.table.context = match self.backend.process_used() {
            Ok(used) => {
                let created: u64 = st.homes.values().filter(|h| h.origin != Origin::Kept).filter_map(|h| h.phys.as_ref()).map(|p| p.len).sum::<u64>()
                    + st.retired.values().filter(|r| r.created).map(|r| r.bytes).sum::<u64>();
                used.saturating_sub(self.native_resident.load(Ordering::Acquire) + created)
            }
            // Not measurable here: the context as open measured it.
            Err(_) => self.backend.opened_context(),
        };
    }

    // ---- the mechanism

    /// Unmaps a page; `withdraw` drops its kept copy too (an eviction or unload; a process that
    /// ends leaves its kept pages as orphans).
    fn unmap(&self, st: &mut State, key: Key, class: Class, req: Option<u64>, withdraw: bool) -> Result<()> {
        let Some(home) = st.homes.remove(&key) else { return Ok(()) };
        let (va, off, span) = st.place(key)?;
        // Custody's handle holds the memory until the Withdraw is sent: held (P) meanwhile.
        let custody = if home.exported && withdraw { self.withdraw(home.identity.clone(), key.1, span) } else { None };
        if custody.is_some() { st.table.pending += span; }
        let physical = home.phys.as_ref().map(|p| p.len);
        let ticket = self.mech.post(Mech::Unmap { va, off, span, phys: home.phys, ready: home.ready }, span);
        if let Some(bytes) = physical {
            st.retired.insert((key, home.gen), Retirement { bytes, created: home.origin != Origin::Kept, mechanism: Some(ticket), custody, kept: home.exported && !withdraw });
        }
        st.last_unmap = Some(ticket);
        st.table.inflight = self.mech.pending();
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
                .filter_map(|k| st.mech(k.0).ok().filter(|m| m.spec.fingerprint.declared()).map(|m| (k, m.key.clone())))
                .collect();
            self.unlock(st);
            w
        };
        let mut adopted = 0;
        for (key, identity) in wanted {
            let start = Instant::now();
            let Some(handle) = c.import(&identity, key.1) else { continue };
            let mut st = self.lock();
            let current = st.mech(key.0).ok().map(|m| m.key.clone());
            if st.homes.contains_key(&key) || current.as_ref() != Some(&identity) {
                self.post(Call::Disown { fp: identity, page: key.1 });
                self.unlock(st);
                continue;
            }
            let (va, off, span) = st.place(key)?;
            let change = self.row.physical_change();
            let phys = match self.backend.import(handle, span) {
                Ok(phys) => phys,
                Err(e) => {
                    // Nothing was imported: certain, so the mark ends.
                    change.finish();
                    return Err(e);
                }
            };
            let gen = st.tick();
            // Pinned (filling) until the mechanism maps it.
            self.mech.post(Mech::Adopt { key, gen, va, off, phys: phys.id, len: phys.len, identity: identity.clone() }, 0);
            st.homes.insert(key, Home { phys: Some(phys), ready: Event::DONE, ready_on: COPY, gen, filled: false, filling: true, transforming: false, exported: true, origin: Origin::Kept, speculative: false, identity, keep_after: false, keeping: false, protecting: false, reclaiming: false, landing: false });
            st.sync(key);
            let s = st.fills(req, key, span, Origin::Kept);
            s.moved.kept.bytes += span;
            s.moved.kept.ns += start.elapsed().as_nanos() as u64;
            adopted += span;
            self.publish(&mut st);
            change.finish();
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
        let gen = st.tick();
        let identity = st.mech(key.0)?.key.clone();
        let decided = st.table.free();
        // Counted mapped (and pinned, filling) now; the mechanism creates and maps it after
        // the unmaps decided before it, then queues its fill.
        st.homes.insert(key, Home { phys: None, ready: Event::DONE, ready_on: COPY, gen, filled: false, filling: true, transforming: false, exported: false, origin: Origin::Store, speculative, identity, keep_after: false, keeping: false, protecting: false, reclaiming: false, landing: false });
        st.sync(key);
        self.mech.post(Mech::Map { key, gen, va, off, span, job: Job { key, gen, req, span, after: vec![] }, speculative, decided }, 0);
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
        let (old, exported, old_identity, old_gen, created) = (std::mem::replace(&mut h.ready, Event::DONE), h.exported, h.identity.clone(), h.gen, h.origin != Origin::Kept);
        // Kept, or with a Keep queued: never written again; a fresh page takes its place.
        let phys = if exported || h.keeping { h.phys.take() } else { None };
        let renew = phys.is_some();
        *h = Home { phys: h.phys.take(), ready: Event::DONE, ready_on: COPY, gen, filled: false, filling: true, transforming: false, exported: false, origin: Origin::Store, speculative: false, identity, keep_after: false, keeping: false, protecting: false, reclaiming: false, landing: false };
        st.sync(key);
        let after: Vec<Event> = after.into_iter().map(Event).collect();
        let job = Job { key, gen, req, span, after: vec![] };
        if renew {
            // Read-only and kept under its old identity (or about to be): never written or
            // protected again. Custody gives the old key up; a fresh page takes its place once
            // its consumers are done.
            let custody = if exported { self.withdraw(old_identity, key.1, span) } else { None };
            if custody.is_some() { st.table.pending += span; }
            // A ready event still pending is among `after` (a Keep waited on it): retired with them.
            if !after.contains(&old) {
                self.backend.retire(old);
            }
            let ticket = self.mech.post(Mech::Renew { key, gen, va, off, span, old: phys, after, job }, 0);
            st.retired.insert((key, old_gen), Retirement { bytes: span, created, mechanism: Some(ticket), custody, kept: false });
            return Ok(());
        }
        // Read-write and nobody else's: filled again in place, after its consumers in-stream.
        if !after.contains(&old) {
            self.backend.retire(old);
        }
        self.queue_fills(vec![Job { after, ..job }], false)
    }

    /// Exports a filled page of a declared set into custody, read-only, only once its ready
    /// event completed: a write still running must not meet a read-only page, and another
    /// process must not import a half-made one.
    fn keep(&self, st: &mut State, key: Key) {
        // While a request runs its pages stay the allocator hook's to take (it makes no custody
        // call); they are kept when it ends.
        if self.custody.is_none() || !st.table.requests.is_empty() {
            return;
        }
        let Ok(m) = st.mech(key.0) else { return };
        let (declared, current) = (m.spec.fingerprint.declared(), m.key.clone());
        let Some(home) = st.homes.get_mut(&key) else { return };
        if !declared || !home.filled || home.exported || home.keep_after || home.identity != current {
            return;
        }
        if !self.backend.query(home.ready).unwrap_or(false) {
            // Pinned (P) until the event completes; the drain that sees it complete keeps it.
            home.keep_after = true;
            let (ready, stream) = (home.ready, home.ready_on);
            if let Some(p) = st.table.page_mut(key) {
                p.releasing.push(ready.0);
            }
            st.events.entry(stream).or_default().push_back((ready, key));
            self.want_keep();
            return;
        }
        if home.phys.is_none() || home.keeping {
            return;
        }
        home.keeping = true;
        st.keep_due.insert(key);
        self.keeps(st);
    }

    /// Queues a Keep for each set with a due page that is idle now. A page in use stays due:
    /// the drain that finds it idle (its last release complete) queues it, so a page pinned
    /// across a whole request is kept at its first idle moment.
    fn keeps(&self, st: &mut State) {
        if st.keep_due.is_empty() {
            return;
        }
        let sets: BTreeSet<u32> = st.keep_due.iter().filter(|k| st.idle(**k)).map(|k| k.0).collect();
        for set in sets {
            if st.keep_queued.insert(set) {
                self.mech.post(Mech::Keep { set }, 0);
            }
        }
    }

    /// The trim thread finishes the Keeps waiting on events ([`Wanted`]).
    fn want_keep(&self) {
        let (lock, want) = &*self.wanted;
        lock.lock().keep = true;
        want.notify_one();
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
                    let s = st.fills(job.req, job.key, span, origin);
                    let f = &mut s.moved.store;
                    f.bytes += span;
                    // The batch's time, shared by its bytes.
                    f.ns += (u128::from(ns) * u128::from(span) / u128::from(bytes)) as u64;
                    if speculative {
                        s.moved.speculative += span;
                    }
                    if filled {
                        self.keep(&mut st, job.key);
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
                        let _ = self.unmap(&mut st, job.key, Class::P, None, true);
                    }
                    if !speculative || !drained {
                        let error = if drained { e } else { Error::Ownership(format!("fill failed ({e}); its copies did not drain")) };
                        st.failed.insert(job.key, error);
                    }
                }
            }
        }
        if sticky { self.ownership_lost.store(true, Ordering::Release); }
        self.publish(&mut st);
        self.unlock(st);
        self.landed.notify_all();
        // A sticky error ends the process, and the GPU's kept pages go with it (a custody call:
        // off the state lock).
        if sticky {
            if let Some(c) = &self.custody {
                let _ = c.fault();
            }
        }
    }

    /// The batch's copies from the store, through the page cache: one engine read per set.
    /// Each page's origin, or its error.
    fn copy(&self, sets: &BTreeMap<u32, SetSpec>, pages: &[(&Job, (Va, u64, u64))]) -> Vec<Result<Origin>> {
        #[cfg(feature = "faults")]
        crate::faults::hit(crate::faults::Point::Fill);
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
                    let _ = self.unmap(&mut st, job.key, Class::P, None, true);
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

    /// Makes `bytes` free, taking in [`decide::victims`]' order, never `keep` (the pages being
    /// bound), after orphans (kept pages no live process maps). Each round takes what covers
    /// the shortfall: idle pages unmapped (kept ones withdrawn, custody settled after), caches
    /// shrunk (outside the lock), pages whose last consumers still run taken once their release
    /// events complete (waited on here). It fails `Exhausted` only when a round takes nothing
    /// and nothing of its own is about to come back (a fill or a Keep's protect:
    /// waited for). `why` and `req` name it in the request's record. For the allocator `hook`,
    /// local work only (no orphans, caches or kept pages: custody calls and Python), and the
    /// room it made is claimed for the hook's segment before the lock goes.
    fn make_room(&self, bytes: u64, keep: &[Key], why: &'static str, req: Option<u64>, free_only: bool) -> Result<u64> {
        let hook = why == "hook";
        let mut record = MakeRoom { step: 0, trigger: why, need: bytes, freed: 0, waited: 0 };
        let (mut orphans_released, mut settled) = (false, false);
        let out = loop {
            let mut st = self.lock();
            let admission = self.refresh(&mut st)?;
            // The driver's free bytes bound the table's (its own slack, another tenant's churn).
            // A refused create says the driver had less than both: its free bytes count for
            // nothing, and the round takes the page's bytes itself.
            let driver = self.backend.free().unwrap_or(u64::MAX);
            let refused = keep.iter().any(|k| st.short.remove(k).is_some());
            let free = if refused { 0 } else { st.table.free().min(driver) };
            let req = req.or_else(|| st.active.iter().find(|(_, a)| a.thread == std::thread::current().id()).map(|((r, _), _)| *r));
            if free >= bytes {
                if hook {
                    self.claimed.fetch_add(bytes, Ordering::AcqRel);
                }
                self.publish(&mut st);
                drop(admission);
                self.unlock(st);
                break Ok(free);
            }
            // Idle work (a request opened idle, a free-only bind) takes free bytes only.
            if free_only || req.and_then(|r| st.table.request(r)).is_some_and(|r| r.idle) {
                let stats = st.stats(req).cloned().unwrap_or_default();
                drop(admission);
                self.unlock(st);
                break Err(Error::Exhausted(exhausted(bytes.saturating_sub(free), None, stats)));
            }
            // Beyond what it could ever have (the budget with orphans released, less its context,
            // the hook's segments but its caches', its pinned pages): Exhausted at once.
            // (A page pinned by varena's own short work, a Keep's protect, comes back.)
            let pinned: u64 = st.table.pages().filter(|(k, p)| p.mapped && (!p.pins.is_empty() || p.filling) && !keep.contains(k)).map(|(_, p)| p.bytes).sum();
            let caches: u64 = st.table.caches.iter().filter(|c| !c.held()).map(|c| c.bytes).sum();
            let fixed = st.table.context + st.table.segments.saturating_sub(caches) + pinned.saturating_sub(st.table.transient);
            if bytes > (st.table.budget() + st.table.orphans).saturating_sub(fixed) {
                let stats = st.stats(req).cloned().unwrap_or_default();
                drop(admission);
                self.unlock(st);
                break Err(Error::Exhausted(exhausted(bytes.saturating_sub(free), None, stats)));
            }
            record.step = req.and_then(|r| st.table.request(r)).map_or(0, |r| r.cursor.step);
            // Kept pages no live process maps go first: nobody's.
            if !hook && !orphans_released && st.table.orphans > 0 {
                orphans_released = true;
                let want = (bytes - free).min(st.table.orphans);
                drop(admission);
                self.unlock(st);
                record.freed += self.custody.as_ref().map_or(0, |c| c.release(want));
                continue;
            }
            let mut want = bytes - free;
            let (mut unmapped, mut withdrawn, mut events, mut shrinks) = (0, false, vec![], vec![]);
            let mut taking: Vec<Taking> = vec![];
            let mut streamed = HashSet::new();
            for v in decide::victims(&st.table, keep) {
                if want == 0 {
                    break;
                }
                // Step 2: a running set's own pages are not evicted one by one; the set streams its upper blocks
                // through slots instead, once per round, and its other pages stay.
                if let Victim::Page { key, .. } = &v {
                    if st.table.distance(*key).is_some_and(|d| d.0 == 0) {
                        if streamed.insert(key.0) {
                            let taken = self.stream(&mut st, key.0, want, req, hook)?;
                            for k in taken.takes {
                                taking.extend(st.take_in_place(k)?);
                            }
                            (unmapped, want) = (unmapped + taken.unmapped, want.saturating_sub(taken.unmapped + taken.taking + taken.pending));
                            events.extend(taken.events);
                        }
                        continue;
                    }
                }
                let kept = |key: Key| st.table.page(key).is_some_and(|p| p.kept);
                match v {
                    Victim::Page { key, .. } if hook && kept(key) => {}
                    Victim::Cache { .. } if hook => {}
                    Victim::Page { key, bytes: b, releasing: false } if hook => {
                        let Some(t) = st.take_in_place(key)? else { continue };
                        taking.push(t);
                        want = want.saturating_sub(b);
                    }
                    Victim::Page { key, bytes: b, releasing: false } => {
                        let class = st.table.class(key).unwrap_or(Class::N);
                        withdrawn |= st.table.page(key).is_some_and(|p| p.kept);
                        self.unmap(&mut st, key, class, req, true)?;
                        unmapped += b;
                        want = want.saturating_sub(b);
                    }
                    Victim::Page { key, bytes: b, releasing: true } => {
                        events.extend(st.table.page(key).map(|p| p.releasing.clone()).unwrap_or_default());
                        want = want.saturating_sub(b);
                    }
                    Victim::Cache { id, bytes: b } => {
                        if let Some(f) = st.shrinks.get(&id) {
                            shrinks.push((id, b.min(want), f.clone()));
                        }
                        want = want.saturating_sub(b);
                    }
                }
            }
            // Its own bytes on their way back (unmaps and withdrawals in flight), and pages about
            // to be evictable again: varena's own short work, and (not for the hook, which never
            // takes a page being filled) a fill.
            let returning = st.table.returning > 0 || (!hook && st.table.pending > 0);
            let coming = st.table.transient > 0 || st.homes.iter().any(|(k, h)| (h.protecting || (!hook && (h.filling || h.landing))) && !keep.contains(k));
            let last = st.last_unmap;
            self.publish(&mut st);
            drop(admission);
            self.unlock(st);
            // The unmaps done, and custody's handles on the kept pages let go, before the
            // bytes count.
            if let Some(t) = last.filter(|_| unmapped > 0) {
                self.mech.settle_to(t + 1);
            }
            if withdrawn {
                self.to_custody.settle();
            }
            // The hook's pages: unmapped and destroyed here, on its own thread.
            for (key, gen, class, (va, off, span), phys, ready) in taking {
                let change = self.row.physical_change();
                let given = self.backend.unmap(va, off, span).and_then(|()| self.backend.destroy(phys));
                let mut st = self.lock();
                if st.homes.get(&key).is_some_and(|h| h.gen == gen) {
                    st.homes.remove(&key);
                    self.backend.retire(ready);
                }
                st.sync(key);
                if let Some(e) = st.stats(req).and_then(|s| s.evicted.get_mut(class.index())) {
                    *e += span;
                }
                if let Err(e) = given {
                    // Its memory could not be proved given back: it stays charged, and this
                    // executor retires (as lost backing does).
                    let id = st.tick();
                    st.retired.insert(((u32::MAX, u32::MAX), id), Retirement { bytes: span, created: true, mechanism: None, custody: None, kept: true });
                    self.ownership_lost.store(true, Ordering::Release);
                    self._device.retiring(true);
                    self.publish(&mut st);
                    self.unlock(st);
                    self.landed.notify_all();
                    return Err(Error::Ownership(format!("a page the hook took could not be proved given back: {e}")));
                }
                unmapped += span;
                self.publish(&mut st);
                self.unlock(st);
                change.finish();
                self.landed.notify_all();
            }
            let shrunk = self.shrink(shrinks);
            for e in &events {
                self.backend.finish(Event(*e))?;
            }
            record.waited += events.len() as u32;
            record.freed += unmapped + shrunk;
            if unmapped + shrunk == 0 && events.is_empty() {
                // Waited for once: what is still returning after that is lost backing.
                if returning && !settled {
                    settled = true;
                    self.mech.settle();
                    if !hook {
                        self.to_custody.settle();
                    }
                    let mut st = self.lock();
                    st.table.pending = self.to_custody.pending();
                    self.unlock(st);
                    continue;
                }
                if coming {
                    let mut st = self.lock();
                    self.wait(&mut st)?;
                    self.unlock(st);
                    continue;
                }
                let mut st = self.lock();
                let free = st.table.free();
                let stats = st.stats(req).cloned().unwrap_or_default();
                self.unlock(st);
                break Err(Error::Exhausted(exhausted(bytes.saturating_sub(free), None, stats)));
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

    /// Runs caches' shrinks, the lock released (the caller's code); returns the bytes freed. A
    /// cache's report made during its shrink is applied after it, as the latest word.
    fn shrink(&self, shrinks: Vec<(u32, u64, Shrink)>) -> u64 {
        let mut total = 0;
        for (id, bytes, f) in shrinks {
            if let Some(c) = self.state.lock().table.cache_mut(id) {
                c.shrinking += 1;
            }
            let freed = crate::caught(|| Ok(f(bytes))).unwrap_or(0);
            let mut st = self.state.lock();
            if let Some(c) = st.table.cache_mut(id) {
                c.bytes = c.bytes.saturating_sub(freed);
                c.shrinking = c.shrinking.saturating_sub(1);
                if let Some(b) = c.reported.take_if(|_| c.shrinking == 0) {
                    c.bytes = b;
                }
            }
            total += freed;
            self.publish(&mut st);
            drop(st);
            self.landed.notify_all();
        }
        total
    }

    /// Maps the next blocks of `req`'s current step into free bytes only (never in shared mode,
    /// where the footprint never grows).
    fn prefetch(&self, req: u64) {
        let mut st = self.lock();
        if !st.table.shared {
            for key in decide::prefetch(&st.table, req) {
                if self.map(&mut st, key, Some(req), true).unwrap_or(false) {
                    let step = st.table.request(req).map_or(0, |r| r.cursor.step);
                    if let Some(p) = st.table.page_mut(key) {
                        p.speculative = Some((req, step));
                    }
                }
            }
            self.publish(&mut st);
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
                    // Its resident bytes, for placement: exact, from its layout.
                    ledger::record_set(&self.shared, &spec.name, layout.nbytes)?;
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
                    self.publish(&mut st);
                    self.unlock(st);
                }
                self.readahead(id)?
            }
            Level::Gpu => {
                // Into free bytes only, in block order: a GPU-level load evicts nothing.
                {
                    let mut st = self.lock();
                    let admission = self.refresh(&mut st)?;
                    let mut free = st.table.free();
                    let keys: Vec<(Key, u64)> = st.table.pages().filter(|(k, p)| k.0 == id && !p.mapped).map(|(k, p)| (k, p.bytes)).collect();
                    for (key, bytes) in keys {
                        if bytes > free {
                            break;
                        }
                        self.map(&mut st, key, None, false)?;
                        free -= bytes;
                    }
                    self.publish(&mut st);
                    drop(admission);
                    self.unlock(st);
                }
                let mut st = self.lock();
                // Until its fills' batches end (landed, accounted for and kept), not only issued.
                while st.table.pages().any(|(k, p)| k.0 == id && p.filling) || st.homes.iter().any(|(k, h)| k.0 == id && h.landing) {
                    self.wait(&mut st)?;
                }
                let idle = st.table.requests.is_empty();
                self.unlock(st);
                // Warm-phase work, off any request's path: an idle load returns once its copies
                // landed and its pages are custody's, safe from a crash at once. Inside a request
                // its pages are kept when the last request ends.
                if idle {
                    self.settle();
                }
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
        let key = format!("dispatch/{id}");
        if let Some(peak) = self.shared.fact(&key)? {
            st.peaks.insert(rid, String::from_utf8_lossy(&peak).into_owned());
            self.shared.put(&key, None)?;
        }
        // Its anonymous host peak is measured from its start (the process's, when it runs alone).
        match st.window.as_mut() {
            Some(window) => {
                window.look();
            }
            None => st.window = ledger::Window::start(),
        }
        let host = match st.peaks.get(&rid) {
            Some(key) => self.shared.fact(&format!("host/{key}"))?.and_then(|v| v.try_into().ok()).map(u64::from_le_bytes),
            None => None,
        };
        st.host.insert(rid, host);
        // Its dispatch key's recorded peak: what it asks peers for (none: every idle byte).
        let peak = match st.peaks.get(&rid) {
            Some(key) => self.shared.fact(&format!("peak/{key}"))?.and_then(|v| v.try_into().ok()).map(u64::from_le_bytes),
            None => None,
        };
        st.table.requests.push(Req { id: rid, name: id.to_string(), steps, peak, idle, ..Req::default() });
        // A cold request's record carries the fills since the last one ended (its loads).
        let earlier = std::mem::take(&mut st.unclaimed);
        st.stats.insert(rid, Stats { request: id.to_string(), moved: earlier.moved, trace: earlier.trace, ..Stats::default() });
        self.publish(&mut st);
        self.unlock(st);
        if !idle {
            self.coreside(rid, true)?;
            self.unkeep(rid);
        }
        Ok(rid)
    }

    /// At a request's open and its pass starts (never in the allocator hook, which makes no
    /// custody call): kept pages
    /// the hook may need are withdrawn from custody and stay mapped, so its local work can take
    /// them, in [`decide::victims`]' order: all of them for an unmeasured request, else as many
    /// as its recorded peak lacks beside the free bytes and what the hook could take already.
    /// They are kept again when no request is open.
    fn unkeep(&self, req: u64) {
        let mut st = self.lock();
        // A Keep already protecting when the request opened ends kept: withdrawn below.
        while st.homes.values().any(|h| h.protecting) {
            if self.wait(&mut st).is_err() {
                return self.unlock(st);
            }
        }
        if !st.homes.values().any(|h| h.exported) {
            return self.unlock(st);
        }
        let Some(peak) = st.table.request(req).map(|r| r.peak) else { return self.unlock(st) };
        let victims = decide::victims(&st.table, &[]);
        let kept = |st: &State, key: Key| st.table.page(key).is_some_and(|p| p.kept);
        let takeable: u64 = victims.iter().filter_map(|v| if let Victim::Page { key, bytes, .. } = v { (!kept(&st, *key)).then_some(*bytes) } else { None }).sum();
        let mut need = peak.map_or(u64::MAX, |p| p.saturating_sub(st.table.free() + takeable));
        let mut withdrawn = vec![];
        for v in victims {
            let Victim::Page { key, bytes, .. } = v else { continue };
            if need == 0 {
                break;
            }
            if !kept(&st, key) {
                continue;
            }
            let Some(h) = st.homes.get(&key).filter(|h| h.exported) else { continue };
            let (fp, gen) = (h.identity.clone(), h.gen);
            // None: its Export was still queued and is cancelled; custody never had it.
            withdrawn.push((key, gen, self.withdraw(fp, key.1, 0)));
            need = need.saturating_sub(bytes);
        }
        self.unlock(st);
        // Still kept (custody holds the memory, the hook never takes it) until custody let go.
        if let Some(last) = withdrawn.iter().filter_map(|(.., t)| *t).max() {
            self.to_custody.settle_to(last + 1);
        }
        let mut st = self.lock();
        for (key, gen, _) in withdrawn {
            if let Some(h) = st.homes.get_mut(&key).filter(|h| h.gen == gen) {
                h.exported = false;
                st.sync(key);
            }
        }
        self.publish(&mut st);
        self.unlock(st);
    }

    /// Co-residency, at a request's open and its pass starts (never in the hook): short of its
    /// recorded peak beside its free bytes and its own evictable ones, it asks the other rows
    /// for their idle pages (pages only, never caches); unmeasured, for every idle page they
    /// have.
    fn coreside(&self, req: u64, open: bool) -> Result<()> {
        let asks: Vec<(u64, u64)> = {
            let mut st = self.lock();
            // A request's open samples the device; a pass start reads the rows and the device as
            // last sampled, and samples it again only before an ask.
            let mut admission = if open { self.refresh(&mut st)? } else { self.rows(&mut st)? };
            if !open && !Self::asks(&st, req).is_empty() {
                drop(admission);
                admission = self.refresh(&mut st)?;
            }
            let asks = Self::asks(&st, req);
            drop(admission);
            self.unlock(st);
            asks
        };
        for (row, bytes) in asks {
            self.row.ask(row, Class::N, bytes)?;
        }
        Ok(())
    }


    /// The rows `req` asks for idle pages, and how many bytes each: short of its recorded peak
    /// beside its free bytes and its own evictable ones; unmeasured, every idle byte.
    fn asks(st: &State, req: u64) -> Vec<(u64, u64)> {
        let Some(r) = st.table.request(req) else { return vec![] };
        let mut short = match r.peak {
            Some(p) => {
                let own: u64 = decide::victims(&st.table, &[]).iter().map(|v| match v {
                    Victim::Page { bytes, .. } | Victim::Cache { bytes, .. } => *bytes,
                }).sum();
                p.saturating_sub(st.table.free() + own)
            }
            None => u64::MAX,
        };
        let mut asks = vec![];
        for o in &st.table.others {
            if short == 0 {
                break;
            }
            let bytes = short.min(o.idle());
            if bytes > 0 {
                asks.push((o.row, bytes));
                short = short.saturating_sub(bytes);
            }
        }
        asks
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
                // Its Keep's protect runs now: no pin (and no refill) while its access changes;
                // the Keep's end wakes the binder.
                Some(h) if h.protecting || h.reclaiming => {
                    waited = true;
                    self.wait(&mut st)?;
                }
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
                    let orphans = st.table.orphans > 0;
                    let bytes = st.table.page(key).map_or(0, |p| p.bytes);
                    self.unlock(st);
                    if orphans && self.adopt(vec![key], req)? > 0 {
                        st = self.lock();
                        continue;
                    }
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
        self.publish(&mut st);
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
        let (id, step_pages, orphans) = {
            let mut st = self.lock();
            st.mech(set)?;
            let _admission = self.rows(&mut st)?;
            if Self::advance(&mut st, req, set)? {
                let budget = st.table.budget();
                if let Some(s) = st.step_stats(req) {
                    s.capacity = budget;
                }
            }
            let id = st.tick();
            st.active.insert((req, set), Active { id, thread: std::thread::current().id() });
            let step: Vec<Key> = st.table.request(req).and_then(|r| r.steps.get(r.cursor.step as usize)).map(|(sets, _)| st.table.pages().filter(|(k, p)| sets.contains(&k.0) && !p.mapped).map(|(k, _)| k).collect()).unwrap_or_default();
            let orphans = st.table.orphans > 0;
            self.publish(&mut st);
            self.unlock(st);
            (id, step, orphans)
        };
        // Kept pages of this step are adopted before anything could release them as orphans.
        if orphans {
            self.adopt(step_pages, Some(req))?;
        }
        if !self.state.lock().table.request(req).is_some_and(|r| r.idle) {
            self.coreside(req, false)?;
            self.unkeep(req);
        }
        self.prefetch(req);
        Ok(id)
    }

    /// A pass ends: `peak` (what the execution layer saw it allocate at most) and what the row
    /// holds now go into the request's record.
    fn end_pass(&self, req: u64, pass: u64, peak: u64) -> Result<()> {
        let mut st = self.lock();
        self.physical(&mut st);
        st.active.retain(|(r, _), a| !(*r == req && a.id == pass));
        let compute = st.table.segments.max(peak);
        let held = st.table.vmm + st.table.mapping + compute;
        if let Some(s) = st.stats(Some(req)) {
            s.peak = s.peak.max(held);
            s.compute = s.compute.max(compute);
        }
        self.publish(&mut st);
        self.unlock(st);
        Ok(())
    }

    fn end_request(&self, req: u64) -> Stats {
        let mut st = self.lock();
        self.unstream(&mut st, req);
        let name = st.table.request(req).map(|r| r.name.clone()).unwrap_or_default();
        st.active.retain(|(r, _), _| *r != req);
        st.synthetic.remove(&req);
        st.table.requests.retain(|r| r.id != req);
        let owned: Vec<u32> = st.table.caches.iter().filter(|c| c.owner == name).map(|c| c.id).collect();
        let holding: Vec<u32> = st.table.caches.iter().filter(|c| c.owner == name && c.bytes > 0 && c.holds == 0 && c.releasing.is_empty()).map(|c| c.id).collect();
        let shrinks: Vec<(u32, u64, Shrink)> = holding.iter().filter_map(|id| st.shrinks.get(id).map(|s| (*id, u64::MAX, s.clone()))).collect();
        let leftover: u64 = st.pins.values().filter(|(_, r, _)| *r == Some(req)).filter_map(|(k, ..)| st.table.page(*k)).map(|p| p.bytes).sum();
        let mut stats = st.stats.remove(&req).unwrap_or_default();
        st.host.remove(&req);
        // Its anonymous host peak (the process's since the first open request started).
        stats.host = st.window.as_mut().and_then(ledger::Window::look);
        if st.table.requests.is_empty() {
            st.window = None;
        }
        if let Some(key) = st.peaks.remove(&req) {
            let _ = ledger::record_peak(&self.shared, &format!("peak/{key}"), stats.peak);
            let _ = ledger::record_compute(&self.shared, &key, stats.compute);
            if let Some(h) = stats.host {
                let _ = ledger::record_host(&self.shared, &key, h);
            }
        }
        self.unlock(st);
        self.shrink(shrinks);
        let mut st = self.lock();
        // No request open: every page kept that can be (the hook's to take until now).
        if st.table.requests.is_empty() {
            let keys: Vec<Key> = st.homes.keys().copied().collect();
            keys.into_iter().for_each(|k| self.keep(&mut st, k));
        }
        st.table.caches.retain(|c| !owned.contains(&c.id));
        owned.iter().for_each(|id| {
            st.shrinks.remove(id);
        });
        stats.leftover = leftover;
        stats.keep_pending = st.keep_pending();
        st.last = Some(stats.clone());
        self.publish(&mut st);
        self.unlock(st);
        self.shared.ring();
        stats
    }

    /// The ask thread's answer: idle pages no plan of this row's uses ([`decide::answer`]),
    /// never a cache, never waiting on anything but its own protects; it runs
    /// no caller code. Kept pages' bytes come back once custody lets go: the trim thread
    /// settles that after the reply and publishes them free.
    fn answer(&self, bytes: u64) -> u64 {
        let out = crate::caught(|| {
            let mut st = self.lock();
            // A protect's page comes back with it: the answer counts it.
            while st.table.transient > 0 {
                self.wait(&mut st)?;
            }
            let mut freed = 0;
            for key in decide::answer(&st.table, bytes) {
                freed += st.table.page(key).map_or(0, |p| p.bytes);
                self.unmap(&mut st, key, Class::N, None, true)?;
            }
            let settle = st.table.pending > 0;
            let unmapped = st.last_unmap.filter(|_| freed > 0);
            self.publish(&mut st);
            self.unlock(st);
            // Its own unmaps first (the mechanism's, up to its last): the asker then finds the
            // bytes free.
            if let Some(t) = unmapped {
                self.mech.settle_to(t + 1);
            }
            if settle {
                let (lock, want) = &*self.wanted;
                lock.lock().settle = true;
                want.notify_one();
            }
            Ok(freed)
        });
        if out.is_err() {
            self.recover();
        }
        out.unwrap_or(0)
    }

    /// The trim thread's work: the Keeps waiting on events, custody settled after an answer,
    /// then a ledger event.
    fn idle_trim(&self, wanted: Wanted) {
        self.backend.sweep();
        if wanted.keep {
            // Only copies still landing: a consumer's release is never waited on under a pin
            // (a renewing bind would wait for its forward).
            if let Err(e) = self.finish_pending(Pending::Landing) {
                tracing::warn!("varena: a landing copy's event is unproven: {e}");
            }
        }
        if wanted.settle {
            self.to_custody.settle();
            let mut st = self.state.lock();
            st.table.pending = self.to_custody.pending();
            drop(st);
        }
        let mut st = self.state.lock();
        self.publish(&mut st);
        drop(st);
        self.shared.ring();
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
    /// passes), waits until its memory has retired: its row gone, TensorD woken. A process
    /// ends after it; ending without it leaves the retirement to TensorD's reap.
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
            if home.filling || home.transforming || home.keeping || !self.backend.query(home.ready)? { return Ok(false); }
        }
        // A clean executor exit keeps exported pages in custody for the next owner.
        // Explicit unload withdraws them; final public retirement only drops local mappings.
        let keys: Vec<Key> = st.homes.keys().copied().collect();
        for key in keys { self.unmap(&mut st, key, Class::N, None, false)?; }
        self.publish(&mut st);
        let empty = st.homes.is_empty();
        drop(st);
        Ok(empty && self.mech.idle() && self.to_custody.idle())
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
    /// ready event (a host wait, on the retirer's own thread), Keeps, queued calls, and every
    /// worker's reference.
    fn quiesce(self: &Arc<Self>) -> Result<()> {
        {
            let mut st = self.lock();
            while st.homes.values().any(|h| h.filling) {
                self.wait(&mut st)?;
            }
            self.unlock(st);
        }
        self.finish_pending(Pending::All)?;
        self.mech.settle();
        self.to_custody.settle();
        let mut alone = self.quiet.lock.lock();
        while Arc::strong_count(self) > 1 {
            self.quiet.gone.wait(&mut alone);
        }
        Ok(())
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.to_custody.close();
        self.row.knock();
        self.wanted.0.lock().closed = true;
        self.wanted.1.notify_all();
        self.burst.0.lock().closed = true;
        self.burst.1.notify_all();
        // The mechanism's thread is gone with the Memory: what it was still to do happens here.
        self.mech.close();
        for op in self.mech.drain() {
            match op {
                Mech::Unmap { va, off, span, phys, ready } => {
                    let _ = self.backend.finish(ready);
                    let _ = self.backend.unmap(va, off, span);
                    phys.into_iter().for_each(|p| { let _ = self.backend.destroy(p); });
                    self.backend.retire(ready);
                }
                Mech::Renew { va, off, span, old, .. } => {
                    let _ = self.backend.unmap(va, off, span);
                    old.into_iter().for_each(|p| { let _ = self.backend.destroy(p); });
                }
                Mech::Move { va, from, span, phys, .. } => {
                    let _ = self.backend.unmap(va, from, span);
                    let _ = self.backend.destroy(phys);
                }
                Mech::Release(va) => self.backend.release(va),
                Mech::Map { .. } | Mech::Adopt { .. } | Mech::Keep { .. } => {}
            }
        }
        let mut st = self.state.lock();
        let homes: Vec<(Key, Home)> = st.homes.drain().collect();
        for (key, home) in homes {
            let Some(phys) = home.phys else { continue };
            if let Ok((va, off, span)) = st.place(key) {
                let _ = self.backend.unmap(va, off, span);
            }
            let _ = self.backend.destroy(phys);
        }
        let sets: Vec<SetMech> = st.sets.drain(..).flatten().collect();
        drop(st);
        for m in sets {
            self.backend.release(m.va);
        }
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

fn ask_loop(listener: crate::os::Listener, row: &Weak<Row>, inner: &Weak<Inner>, stop: &AtomicBool) {
    if let Some(row) = row.upgrade() {
        row.ask_thread();
    }
    let inner = inner.clone();
    Row::serve_asks(listener, stop, row.clone(), move |ask| {
        let Some(me) = Work::of(&inner) else { return 0 };
        // Never dies: a failed or panicking answer frees nothing, and the next ask is served.
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match ask {
            ledger::Ask::Room { bytes, .. } => me.answer(bytes),
        }))
        .unwrap_or(0)
    });
}

/// The mechanism thread: the device's VMM calls, in the order decided, off the state lock.
fn mech_loop(queue: &Outbox<Mech>, inner: &Weak<Inner>) {
    while let Some((ticket, bytes, op)) = queue.next() {
        let Some(me) = Work::of(inner) else { return };
        // An op that creates or destroys physical memory: the row's counts are no foreign-byte
        // evidence until its outcome is published (an uncertain one never finishes).
        let change = matches!(op, Mech::Map { .. } | Mech::Unmap { .. } | Mech::Renew { .. }).then(|| me.row.physical_change());
        // Consecutive maps (a bind's block) reach the fill threads as one group: handed over
        // before any other call, once nothing waits for this thread, or once they make a batch.
        if !matches!(op, Mech::Map { .. }) {
            me.flush_fills();
        }
        me.mechanism(op);
        if queue.waiting() == 0 || me.grouped_bytes() >= me.fill_budget() {
            me.flush_fills();
        }
        if me.check_ownership().is_err() {
            // The fault path retained the executor and uncertain resources. Tickets finish so
            // callers can retire it, but physical bytes are never advertised as returned.
            drop(change); // unfinished: uncertain ownership stays unproven
            queue.sent(ticket, 0);
            continue;
        }
        let mut st = me.lock();
        // Publish only after refresh will also read the physically completed byte count.
        // The ticket remains live until this worker releases its Memory owner below.
        queue.returned(ticket, bytes);
        st.table.inflight = queue.pending();
        me.publish(&mut st);
        if let Some(change) = change {
            change.finish();
        }
        me.unlock(st);
        me.landed.notify_all();
        // The row's facts changed (a page mapped, kept or gone).
        me.shared.ring();
        // Done with the Memory before it reads as done: a settled Memory is this thread's no
        // more.
        drop(me);
        queue.sent(ticket, 0);
        // Retired events destroyed here (and on the trim thread), never on a caller's thread: a
        // destroy can wait on a driver lock another thread's device sync holds.
        if let Some(me) = Work::of(inner) {
            me.backend.sweep();
        }
    }
}

/// The sender thread: custody calls, in the order they were posted, until the Memory goes. It
/// holds no lock of the Memory's while it calls; a call that reached a new custody holder takes
/// the state lock alone after, so every kept page is exported to it again.
fn send_loop(outbox: &Outbox<Call>, custody: Option<&custody::Client>, inner: &Weak<Inner>) {
    while let Some((ticket, bytes, call)) = outbox.next() {
        let holder = custody.map(custody::Client::holder);
        let result = match call {
            Call::Export { fp, page, len, handle } => custody.map(|c| c.export(&fp, page, len, handle)),
            Call::Withdraw { fp, page } => custody.map(|c| c.withdraw(&fp, page)),
            Call::Disown { fp, page } => custody.map(|c| c.disown(&fp, page)),
        };
        if bytes > 0 && !matches!(result, Some(Ok(()))) {
            if let Some(me) = Work::of(inner) {
                let mut st = me.state.lock();
                me.lost_backing(&mut st, bytes, true, "custody did not acknowledge withdrawal");
            }
            outbox.sent(ticket, 0); // processed, but its physical bytes remain charged
            continue;
        }
        outbox.sent(ticket, bytes);
        let moved = holder != custody.map(custody::Client::holder);
        if moved || bytes > 0 {
            if let Some(me) = Work::of(inner) {
                // A new custody holder gets every kept page again (the drain does it);
                // withdrawn bytes are free for the next decision.
                let mut st = me.lock();
                st.table.pending = me.to_custody.pending();
                me.publish(&mut st);
                me.unlock(st);
                me.landed.notify_all();
                me.shared.ring();
            }
        }
    }
}

fn trim_loop(wanted: &(Mutex<Wanted>, Condvar), inner: &Weak<Inner>) {
    loop {
        let wanted = {
            let (lock, want) = wanted;
            let mut w = lock.lock();
            while !w.settle && !w.keep {
                if w.closed {
                    return;
                }
                want.wait(&mut w);
            }
            std::mem::take(&mut *w)
        };
        let Some(me) = Work::of(inner) else { return };
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| me.idle_trim(wanted)));
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
            self.inner.publish(&mut st);
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
        Ok(Pass { inner: self.inner.clone().into(), req: self.id, set: set.0, id })
    }

    /// The request waits on a child call of its family `family`: until the guard drops, its
    /// peak is published apart (for every request's admission but that family's child) and its
    /// step's pages are a later use (taken before the child's own).
    pub fn yielding(&self, family: &str) -> Yielded {
        let _ = self.inner.guard(|| {
            let mut st = self.inner.lock();
            if let Some(r) = st.table.request_mut(self.id) {
                r.yielded = true;
                r.family = ledger::family_id(family);
            }
            self.inner.publish(&mut st);
            self.inner.unlock(st);
            self.inner.shared.ring();
            Ok(())
        });
        Yielded { wait: Some((self.inner.clone().into(), self.id)) }
    }

    pub fn stats(&self) -> Stats {
        let st = self.inner.state.lock();
        let mut stats = st.stats.get(&self.id).cloned().unwrap_or_default();
        stats.keep_pending = st.keep_pending();
        stats
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
                h.ready_on = self.stream;
                h.filled = true;
                h.transforming = false;
                if old != event {
                    self.inner.backend.retire(old);
                }
            }
            st.sync(self.key);
            self.inner.keep(&mut st, self.key);
            self.inner.publish(&mut st);
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
                let mut st = self.inner.state.lock();
                self.inner.lost_backing(&mut st, 0, true, error);
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
        let _ = self.inner.guard(|| self.inner.end_pass(self.req, self.id, peak));
    }
}

impl Inner {
    fn lost_backing(self: &Arc<Self>, st: &mut State, bytes: u64, tracked: bool, why: impl std::fmt::Display) -> Error {
        if !tracked {
            let id = st.tick();
            st.retired.insert(((u32::MAX, u32::MAX), id), Retirement { bytes, created: true, mechanism: None, custody: None, kept: true });
        }
        self.ownership_lost.store(true, Ordering::Release);
        self._device.retiring(true);
        self.publish(st);
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
            Mech::Map { key, gen, va, off, span, job, speculative, decided } => self.place_page(key, gen, va, off, span, job, (!speculative).then_some(decided)),
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
                    if let Err(error) = self.backend.destroy(old) {
                        let mut st = self.state.lock();
                        self.lost_backing(&mut st, span, true, error);
                        return;
                    }
                }
                after.into_iter().for_each(|e| self.backend.retire(e));
                self.place_page(key, gen, va, off, span, job, Some(u64::MAX));
            }
            Mech::Adopt { key, gen, va, off, phys, len, identity } => {
                let mapped = self.backend.map(va, off, &Physical { id: phys, len }, Access::Read);
                let mut st = self.state.lock();
                let current = st.homes.get(&key).is_some_and(|h| h.gen == gen);
                if let Err(error @ Error::Ownership(_)) = &mapped {
                    self.lost_backing(&mut st, len, current, error);
                    return;
                }
                match (mapped, current) {
                    (Ok(()), true) => {
                        if let Some(h) = st.homes.get_mut(&key) {
                            (h.filled, h.filling) = (true, false);
                        }
                        st.sync(key);
                    }
                    (mapped, current) => {
                        // Gone meanwhile, or unmappable: back to custody, and the page refills.
                        let home = current.then(|| st.homes.remove(&key)).flatten();
                        st.sync(key);
                        drop(st);
                        if mapped.is_ok() {
                            if let Err(error) = self.backend.unmap(va, off, len) {
                                let mut st = self.state.lock();
                                self.lost_backing(&mut st, len, false, error);
                                return;
                            }
                        }
                        if let Some(Some(p)) = home.map(|h| h.phys) {
                            if let Err(error) = self.backend.destroy(p) {
                                let mut st = self.state.lock();
                                self.lost_backing(&mut st, len, false, error);
                                return;
                            }
                        }
                        self.post(Call::Disown { fp: identity, page: key.1 });
                    }
                }
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
                    if let Err(error) = self.backend.destroy(p) {
                        let mut st = self.state.lock();
                        self.lost_backing(&mut st, span, true, error);
                        return;
                    }
                }
                self.backend.retire(ready);
            }
            Mech::Keep { set } => self.keep_set(set),
            Mech::Move { key, gen, va, from, off, span, phys, after, job } => self.move_slot(key, gen, va, from, off, span, phys, after, job),
            Mech::Release(va) => self.backend.release(va),
        }
    }

    /// The set's due pages that are idle now, made read-only (each contiguous run in one
    /// protect) and exported. A page pinned, with a release pending, filling or transformed
    /// stays due; one unmapped or renewed since is not kept. Binds and saves of a page wait
    /// while its protect runs.
    fn keep_set(self: &Arc<Self>, set: u32) {
        let (va, pages) = {
            let mut st = self.state.lock();
            st.keep_queued.remove(&set);
            let Ok(va) = st.mech(set).map(|m| m.va) else { return };
            let due: Vec<Key> = st.keep_due.range((set, 0)..=(set, u32::MAX)).copied().collect();
            // A request opened since this Keep was queued: the pages stay the hook's, and its
            // end keeps them.
            if !st.table.requests.is_empty() {
                for key in due {
                    st.keep_due.remove(&key);
                    if let Some(h) = st.homes.get_mut(&key) {
                        h.keeping = false;
                    }
                    st.sync(key);
                }
                return;
            }
            let mut pages = vec![];
            for key in due {
                let (place, idle) = (st.place(key), st.idle(key));
                let Some(home) = st.homes.get_mut(&key).filter(|h| h.keeping && !h.exported) else {
                    st.keep_due.remove(&key);
                    continue;
                };
                let (Ok((_, off, span)), Some(phys), true) = (place, home.phys.as_ref().map(|p| (p.id, p.len)), idle) else { continue };
                home.protecting = true;
                pages.push((key, home.gen, off, span, phys));
                st.table.transient += span;
            }
            for (key, ..) in &pages {
                st.keep_due.remove(key);
                st.sync(*key);
            }
            (va, pages)
        };
        let mut made = Vec::with_capacity(pages.len());
        // Each contiguous run (a page starting where the one before ends) in one protect.
        for run in pages.chunk_by(|a, b| a.2 + a.3 == b.2) {
            let (Some(first), Some(last)) = (run.first(), run.last()) else { continue };
            let protected = self.backend.protect(va, first.2, last.2 + last.3 - first.2, Access::Read);
            made.extend(run.iter().map(|(.., (id, len))| protected.clone().and_then(|()| self.backend.export(&Physical { id: *id, len: *len }))));
        }
        let mut st = self.state.lock();
        st.table.transient = st.table.transient.saturating_sub(pages.iter().map(|p| p.3).sum());
        for ((key, gen, _, span, _), made) in pages.iter().zip(made) {
            let Some(home) = st.homes.get_mut(key).filter(|h| h.gen == *gen) else { continue };
            (home.keeping, home.protecting) = (false, false);
            match made {
                Ok(handle) => {
                    home.exported = true;
                    let fp = home.identity.clone();
                    self.post(Call::Export { fp, page: key.1, len: *span, handle });
                }
                Err(e) => tracing::warn!("varena: keeping {key:?} failed: {e}"),
            }
            st.sync(*key);
        }
        drop(st);
        self.landed.notify_all();
    }

    /// Creates and maps a page for `key` and queues its fill; a create the device cannot
    /// serve leaves the page unmapped, with the free bytes it was decided on (`decided`;
    /// `None`: a speculative page, which goes quietly).
    #[allow(clippy::too_many_arguments)]
    fn place_page(self: &Arc<Self>, key: Key, gen: u64, va: Va, off: u64, span: u64, job: Job, decided: Option<u64>) {
        let speculative = decided.is_none();
        let made = self.backend.create(span).and_then(|p| match self.backend.map(va, off, &p, Access::ReadWrite) {
            Ok(()) => Ok(p),
            Err(error @ Error::Ownership(_)) => {
                let mut st = self.state.lock();
                Err(self.lost_backing(&mut st, p.len, false, error))
            }
            Err(error) => {
                let bytes = p.len;
                match self.backend.destroy(p) {
                    Ok(()) => Err(error),
                    Err(release) => {
                        let mut st = self.state.lock();
                        Err(self.lost_backing(&mut st, bytes, false, release))
                    }
                }
            }
        });
        let mut st = self.state.lock();
        if !st.homes.get(&key).is_some_and(|h| h.gen == gen) {
            drop(st);
            if let Ok(p) = made {
                let released = self.backend.unmap(va, off, span).and_then(|()| self.backend.destroy(p));
                if let Err(error) = released {
                    let mut st = self.state.lock();
                    self.lost_backing(&mut st, span, false, error);
                }
            }
            return;
        }
        match made {
            Ok(p) => {
                if let Some(h) = st.homes.get_mut(&key) {
                    h.phys = Some(p);
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
                        // The device had less than the table showed: the bind measures and
                        // decides again.
                        error @ Error::Device { code: crate::backend::OUT_OF_MEMORY, .. } => {
                            let driver_free = self.backend.free().unwrap_or(0);
                            tracing::debug!(target: "varena::memory", ?key, ?error, span, decided, driver_free,
                                estimated_free = st.table.free(), mapped = st.table.mapped(),
                                pending = st.table.pending, inflight = st.table.inflight,
                                "physical allocation refused; the bind decides again on the driver's free bytes");
                            st.short.insert(key, driver_free);
                        }
                        e => drop(st.failed.insert(key, e)),
                    }
                }
            }
        }
    }

    /// The wait on a child ended: its step is in use again, and peers are asked when it is
    /// short (co-residency, as at a pass start).
    fn unyield(&self, req: u64) -> Result<()> {
        {
            let mut st = self.lock();
            if let Some(r) = st.table.request_mut(req) {
                r.yielded = false;
            }
            self.publish(&mut st);
            self.unlock(st);
        }
        self.coreside(req, false)
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

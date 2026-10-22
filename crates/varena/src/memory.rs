//! One executor's memory on one device: the applier of [`crate::decide`]'s actions. One
//! `parking_lot` mutex guards the table and the mechanism's state; it is held for a decision and
//! its application, never across a fill, a remote call or a caller's callback: custody and tier
//! calls made while it is held are posted, in order, to one sender thread.
//!
//! Lock order: none. The state lock is only ever taken alone; every other lock (the release
//! receiver, the fill queue, the trim wants, a custody or tier client's stream, a held layout's
//! pins) is taken alone or inside the state lock and never takes another, and the state lock is
//! never taken while one of them is held. A binder waits on the state lock's condvar only. Fills run
//! on fill threads (demand misses before prefetches); pin releases arrive through a channel;
//! asks are answered from pages by a thread of its own that runs no caller code; an idle
//! process's trim and cache shrinks run on its trim thread. A panic is caught at every public
//! call and on every thread: the call fails `Internal`, the table is derived again from the
//! mappings, and the process keeps serving.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use crossbeam_channel::{select_biased, unbounded, Receiver, Sender};

use crate::outbox::Outbox;
use std::time::Instant;

use parking_lot::{Condvar, Mutex, MutexGuard};
use platform::handles::Owned;

use tensorfs::io::source::Placement;
use tensorfs::io::Dst;

use crate::backend::{Access, Backend, Event, Physical, Va};
use crate::custody;
use crate::decide::{decide, Action, Trigger, Tried};
use crate::ledger::{self, Facts, Grant, Room, Row, Shared};
use crate::table::{Class, CutSite, Fill, Key, Level, Page, Plan, Req, Set, SetSpec, Table};
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
    pub sites: Vec<SitePeak>,
}

/// A cut site as a pass measured it: its peak, and whether it is a collective's (the flag is
/// kept with the room, so a first pass on another rank knows it).
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SitePeak {
    pub name: String,
    pub peak: u64,
    pub collective: bool,
    /// The finest level its op has.
    pub finest: u32,
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
    /// Filled pages copied into the RAM tier afterwards: speculation, off every bind's path.
    pub saved: Fills,
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
    /// Every fill since the previous request ended (loads included) and during this one.
    pub moved: Moved,
    /// One entry per bind, after one per fill made before it opened (its loads: step 0, with
    /// the fill's origin).
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
    /// The finest level the op has: never proposed or answered beyond it.
    pub finest: u32,
    /// The op's shape key: outside any pass the request's memo is per (site, shape); a pass's
    /// own sites go by the pass's shape.
    pub shape: &'a str,
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
    /// Run the transform now: the page's bytes are not the weights yet (a declared or
    /// undeclared fill's store bytes); [`Pin::filled`] after it. Never for bytes that are the
    /// weights (stored, kept, or a transformed page from the RAM tier).
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
    set: u32,
    id: u64,
}

/// The caller's callbacks: called with no lock held, so they may call back in.
type Shrink = Arc<dyn Fn(u64) -> u64 + Send + Sync>;
type Trim = Arc<dyn Fn() -> u64 + Send + Sync>;

struct SetMech {
    spec: SetSpec,
    va: Va,
    /// The RAM tier's name for its pages' bytes ([`tier_name`]) and its layout there.
    tier: String,
    held: Option<Arc<tier::Held>>,
    closing: bool,
}

/// The RAM tier's name for the bytes a set's pages hold: the store's (stored, or transformed
/// after each fill), or a declared transform's under its identity.
fn tier_name(spec: &SetSpec) -> String {
    match &spec.fingerprint.fill {
        Fill::Declared(id) => tensorfs::sha256::hex_digest(format!("{}\0declared\0{id}", spec.layout.digest).as_bytes()),
        Fill::Stored | Fill::Undeclared => spec.layout.digest.clone(),
    }
}

struct Home {
    /// `None` while the mechanism thread creates and maps it.
    phys: Option<Physical>,
    ready: Event,
    /// Tells a fill whether its page is still the one it was queued for.
    gen: u64,
    filled: bool,
    filling: bool,
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
    /// Its Keep (read-only, export) is with the mechanism thread.
    keeping: bool,
}

impl Home {
    /// Its bytes landed and are not the weights yet: its binder runs the transform.
    fn fresh(&self) -> bool {
        !self.filled && !self.filling && !self.transforming
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
    /// Make a filled page read-only and hand its export to custody.
    Keep { key: Key, gen: u64, va: Va, off: u64, span: u64, phys: u64, len: u64 },
    /// A read-only kept page whose fill identity changed: never protected again, it is
    /// unmapped (after its consumers) and a fresh page filled in its place.
    Renew { key: Key, gen: u64, va: Va, off: u64, span: u64, old: Option<Physical>, after: Vec<Event>, job: Job },
    /// A closed set's reservation, after its pages' unmaps.
    Release(Va),
}

struct Active {
    id: u64,
    set: u32,
    /// The package entry the pass runs, and its shape (dimensions joined by `x`).
    entry: String,
    dims: String,
    /// Its configuration's key, (entry, shape) escaped: cut levels, proposals and flags are
    /// kept per (set, this).
    shape: String,
    allocated: u64,
    /// Its own room (its start's allocation included), measured or the unmeasured reserve.
    room: u64,
    /// The context at the pass's start: its growth is the pass's.
    context: u64,
    /// Its room is the one measured for its set, shape and cut levels.
    measured: bool,
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

/// Fill jobs: a demand miss goes before any prefetch.
struct Jobs {
    demand: Sender<Job>,
    ahead: Sender<Job>,
}

/// One page of a fill batch: where it goes, and how its copy ended.
struct Copying {
    key: Key,
    dst: Dst,
    out: Option<Result<Origin>>,
}

/// The fill threads' work: jobs queued, batches running, the bytes filled since they were last
/// all idle, and whether the Memory is gone. Its lock is a leaf; its condvar wakes the tier
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

/// A filled page to copy into the RAM tier layout `tier` (`gen`: as it was filled) for request
/// `req`.
struct Save {
    key: Key,
    gen: u64,
    tier: String,
    req: Option<u64>,
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
    /// A layout this process holds, handed to a tier that may not know it (a restarted one).
    Readopt { digest: String, regions: Vec<(u64, u64)>, ask: String, file: Owned },
    /// A closing set's tier regions are unlocked, off the state lock.
    Unpin(Arc<tier::Held>),
    Close { digest: String },
}

/// What an idle process's trim thread runs: the trim hook and cache shrinks.
#[derive(Default)]
struct Wanted {
    trim: bool,
    shrinks: Vec<(u32, u64)>,
    /// An answer withdrew kept pages: settle custody, then publish them free.
    settle: bool,
    closed: bool,
}

#[derive(Default)]
struct State {
    table: Table,
    sets: Vec<Option<SetMech>>,
    homes: HashMap<Key, Home>,
    /// Pin → (page, request).
    /// Pin → (page, request, the thread that holds it).
    pins: HashMap<u64, (Key, Option<u64>, std::thread::ThreadId)>,
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
    /// Sites known to be a collective's (an OOM said so, the ranks agreed on it, or its room
    /// was measured so), and sites an OOM on this rank said are not: a site in neither is
    /// treated as a collective's.
    collective: HashSet<(u32, String, String)>,
    single: HashSet<(u32, String, String)>,
    /// The finest level each site's op has, as its OOMs and measured rooms said.
    finest: HashMap<(u32, String, String), u32>,
    /// Pages whose create found less device memory than the table showed (a foreign
    /// allocation landed meanwhile), with their bytes and the free bytes measured then: a bind
    /// that measures the same again knows nothing more and ends exhausted.
    short: HashMap<Key, (u64, u64)>,
    stuck: BTreeSet<(u32, String, String)>,
    /// The room each cut configuration failed at.
    failed_at: HashMap<String, u64>,
    stats: HashMap<u64, Stats>,
    /// Fills no open request made (loads): the next request's record opens with them.
    unclaimed: Stats,
    last: Option<Stats>,
    /// Passes running, per (request, set): a joint step's sets each have their own.
    active: HashMap<(u64, u32), Active>,
    failed: HashMap<Key, Error>,
    /// The page a tier copy holds now (pinned, like a consumer).
    saving: Option<Key>,
    /// The latest unmap's mechanism ticket: whoever counts on its bytes settles up to it.
    last_unmap: Option<u64>,
    /// Request → the `<package>|<shape>` its peak is recorded under.
    peaks: HashMap<u64, String>,
    /// Requests whose last step is one their plan did not declare.
    synthetic: HashSet<u64>,
    /// Per demand in flight: bytes asked of other rows and not yet decided on, with what its
    /// own earlier rounds unmapped. Published as held, so nobody else takes them first.
    claims: HashMap<u64, u64>,
    /// The custody connection the kept pages were exported over.
    holder: u64,
    /// The tier connection this process's layouts were opened or handed over on.
    tier_holder: u64,
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
        let home = self.homes.get(&key).map(|h| (h.filled, h.filling, h.speculative, h.exported));
        let pins: Vec<u64> = self.pins.iter().filter(|(_, (k, ..))| *k == key).map(|(id, _)| *id).collect();
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
        self.active.keys().filter(|(_, s)| *s == set).map(|(r, _)| *r).max()
    }

    /// The request's most recently started pass (the one an op, a cut or an OOM is in).
    fn current(&self, req: u64) -> Option<&Active> {
        self.active.iter().filter(|((r, _), _)| *r == req).map(|(_, a)| a).max_by_key(|a| a.id)
    }

    fn pass_of(&self, req: u64, pass: u64) -> Option<&Active> {
        self.active.iter().find(|((r, _), a)| *r == req && a.id == pass).map(|(_, a)| a)
    }

    /// The request's room with `this` pass's own (`own`, allocated `allocated`) beside its other
    /// running passes: the largest allocation at a start, plus each pass's growth beyond it.
    fn joint(&self, req: u64, this: u64, allocated: u64, own: u64) -> u64 {
        let others = self.active.iter().filter(|((r, _), a)| *r == req && a.id != this).map(|(_, a)| a);
        let (high, growth) = others.fold((allocated, own.saturating_sub(allocated)), |(h, g), a| (h.max(a.allocated), g + a.room.saturating_sub(a.allocated)));
        high + growth
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

/// A name's separators escaped, so a joined key reads one way only.
fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('|', "\\|").replace(',', "\\,").replace('=', "\\=")
}

/// A pass's configuration within its set: `<entry>|<shape>`, escaped.
fn config(entry: &str, dims: &str) -> String {
    format!("{}|{}", esc(entry), esc(dims))
}

/// The cut levels in effect (level 0 is no cut).
fn cuts(levels: &BTreeMap<String, u32>) -> BTreeMap<String, u32> {
    levels.iter().filter(|(_, l)| **l > 0).map(|(s, l)| (s.clone(), *l)).collect()
}

/// `<set>|<entry>|<shape>|<site=level,...>`, escaped.
fn room_key(set: &str, config: &str, levels: &BTreeMap<String, u32>) -> String {
    let cuts: Vec<String> = cuts(levels).iter().map(|(s, l)| format!("{}={l}", esc(s))).collect();
    format!("{}|{config}|{}", esc(set), cuts.join(","))
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
    _device: Device,
    /// Signalled by every change of the table while this process waits for room.
    room_wake: Owned,
    custody: Option<Arc<custody::Client>>,
    tier: Option<Arc<tier::Client>>,
    state: Mutex<State>,
    landed: Condvar,
    /// Custody's calls and the tier's, each in the order they were made, for one sender thread
    /// per service.
    to_custody: Arc<Outbox<Call>>,
    to_tier: Arc<Outbox<Call>>,
    /// The device's mechanism calls, for its one thread; their bytes are the unmaps in flight.
    mech: Arc<Outbox<Mech>>,
    release_tx: Sender<Release>,
    releases: Receiver<Release>,
    /// Fill jobs; the fill threads hold the receiving ends and a weak reference to the rest,
    /// so dropping the Memory ends them. The trim thread's work likewise.
    jobs: Jobs,
    burst: Arc<(Mutex<Burst>, Condvar)>,
    /// Filled pages to copy into the RAM tier, for its one thread.
    saves: Arc<Outbox<Save>>,
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
            let room_wake = row.endpoint()?;
            let custody = custody::Client::open(&shared).map_err(|e| tracing::warn!("varena: no custody: {e}")).ok().map(Arc::new);
            let tier = tier::Client::open(shared.ns).map_err(|e| tracing::warn!("varena: no RAM tier: {e}")).ok().map(Arc::new);
            let (to_custody, to_tier, mech) = (Arc::new(Outbox::default()), Arc::new(Outbox::default()), Arc::new(Outbox::default()));
            let (release_tx, releases) = unbounded();
            let ((demand, demanded), (ahead, wanted_ahead)) = (unbounded(), unbounded());
            let holder = custody.as_ref().map_or(0, |c| c.holder());
            let tier_holder = tier.as_ref().map_or(0, |t| t.holder());
            let context = backend.process_used().unwrap_or_else(|_| backend.opened_context());
            let table = Table { capacity: shared.capacity()?, context, ..Table::default() };
            let inner = Arc::new(Inner {
                backend,
                shared,
                row: row.clone(),
                _device: device,
                room_wake,
                custody,
                tier,
                state: Mutex::new(State { table, holder, tier_holder, ..State::default() }),
                landed: Condvar::new(),
                to_custody,
                to_tier,
                mech,
                release_tx,
                releases,
                jobs: Jobs { demand, ahead },
                burst: Arc::new((Mutex::new(Burst::default()), Condvar::new())),
                saves: Arc::new(Outbox::default()),
                wanted: Arc::new((Mutex::new(Wanted::default()), Condvar::new())),
                stop: Arc::new(AtomicBool::new(false)),
                #[cfg(unix)]
                tally: Default::default(),
            });
            let (weak, queue) = (Arc::downgrade(&inner), inner.mech.clone());
            std::thread::Builder::new().name("varena-mechanism".into()).spawn(move || mech_loop(&queue, &weak)).map_err(|e| Error::io("spawn mechanism", e))?;
            for (name, outbox) in [("varena-custody-calls", inner.to_custody.clone()), ("varena-tier-calls", inner.to_tier.clone())] {
                let (weak, c, t) = (Arc::downgrade(&inner), inner.custody.clone(), inner.tier.clone());
                std::thread::Builder::new().name(name.into()).spawn(move || send_loop(&outbox, c.as_deref(), t.as_deref(), &weak)).map_err(|e| Error::io("spawn calls", e))?;
            }
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
            let (weak, saves, burst) = (Arc::downgrade(&inner), inner.saves.clone(), inner.burst.clone());
            std::thread::Builder::new().name("varena-tier-copies".into()).spawn(move || save_loop(&saves, &burst, &weak)).map_err(|e| Error::io("spawn tier copies", e))?;
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
    pub fn cache(&self, owner: &str, shrink: impl Fn(u64) -> u64 + Send + Sync + 'static) -> CacheId {
        let shrink: Shrink = Arc::new(shrink);
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
    pub fn trim_with(&self, trim: impl Fn() -> u64 + Send + Sync + 'static) {
        let trim: Trim = Arc::new(trim);
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
            r?;
            self.inner.open_tier(set.0)
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

    /// Blocks until every mechanism call and every custody and tier call decided so far has
    /// been made: then the kept pages are custody's (a planned handover), and a dropped Memory's
    /// row is gone when the drop returns.
    pub fn settle(&self) {
        self.inner.mech.settle();
        self.inner.to_custody.settle();
        self.inner.to_tier.settle();
        self.inner.saves.settle();
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
        st.claims.clear();
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

    fn unlock(&self, st: MutexGuard<'_, State>) {
        drop(st);
    }

    /// A custody or tier call, sent after everything posted to its service before it.
    fn post(&self, call: Call) {
        match call {
            Call::Export { .. } | Call::Withdraw { .. } | Call::Disown { .. } => self.to_custody.post(call, 0),
            Call::Use { .. } | Call::Readopt { .. } | Call::Unpin(_) | Call::Close { .. } => self.to_tier.post(call, 0),
        };
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

    /// Waits for a fill or release on the state lock's condvar.
    fn wait(&self, st: &mut MutexGuard<'_, State>) {
        self.landed.wait(st);
        self.drain(st);
    }

    fn drain(&self, st: &mut State) {
        let released: Vec<Release> = self.releases.try_iter().collect();
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
        if let Some(h) = self.custody.as_ref().map(|c| c.holder()).filter(|h| *h != st.holder) {
            // A new custody holder knows none of the kept pages: all go to it again.
            st.holder = h;
            let kept: Vec<Key> = st.homes.iter_mut().filter(|(_, h)| h.exported).map(|(k, h)| {
                h.exported = false;
                *k
            }).collect();
            kept.into_iter().for_each(|k| self.keep(st, k));
        }
        if let Some(h) = self.tier.as_ref().map(|t| t.holder()).filter(|h| *h != st.tier_holder) {
            // A new tier (its server restarted) knows none of the layouts this process holds:
            // they go to it with their files, ready regions and all.
            st.tier_holder = h;
            let ask = self.row.ask_name();
            let held: Vec<Call> = st
                .sets
                .iter()
                .flatten()
                .filter_map(|m| m.held.as_ref().and_then(|h| h.handover().ok()).map(|(file, regions)| Call::Readopt { digest: m.tier.clone(), regions, ask: ask.clone(), file }))
                .collect();
            held.into_iter().for_each(|c| self.post(c));
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
        // The tier holds bytes by identity: another identity's layout is another layout.
        let tier = tier_name(&m.spec);
        if tier != m.tier {
            let (old, held) = (std::mem::replace(&mut m.tier, tier), m.held.take());
            if let Some(held) = held {
                self.post(Call::Unpin(held));
            }
            self.post(Call::Close { digest: old });
        }
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
                self.mech.post(Mech::Release(m.va), 0);
                if let Some(held) = m.held {
                    self.post(Call::Unpin(held));
                }
                self.post(Call::Close { digest: m.tier.clone() });
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
        st.table.pending = self.to_custody.pending();
        let others: u64 = st.table.others.iter().map(crate::table::Remote::held).sum();
        let card = self.backend.capacity();
        let used = card.saturating_sub(self.backend.free()?);
        // Unmaps in flight are this process's still (and already spent on the creates queued
        // after them).
        st.table.inflight = self.mech.pending();
        // Foreign bytes take from the card, not from the table's cap: they count only beyond the
        // card's room outside the cap (capacity = min(cap, card - foreign)).
        let foreign = used.saturating_sub(others + orphans + st.table.held() + st.table.inflight);
        st.table.foreign = foreign.saturating_sub(card.saturating_sub(st.table.capacity));
        self.shared.set_foreign(st.table.foreign)?;
        Ok(())
    }

    fn publish(&self, st: &State) {
        let t = &st.table;
        let remaining: Vec<Option<u64>> = t.requests.iter().map(|r| t.remaining(r)).collect();
        // A request waiting on a child call is no stranger's: it runs, and its W counts for
        // everyone but its family's child.
        let idle = t.requests.is_empty();
        self.row.publish(&Facts {
            context: t.context,
            torch: t.torch() + st.claims.values().sum::<u64>(),
            caches: if idle { t.slack() + t.caches.iter().map(|c| c.bytes).sum::<u64>() } else { 0 },
            classes: t.classes(),
            remaining: remaining.iter().flatten().sum(),
            unmeasured: remaining.iter().filter(|r| r.is_none()).count() as u64,
            running: t.requests.len() as u64,
            outside: t.outside(),
            yielded: t.requests.iter().map(|r| t.yielded_w(r)).sum(),
            family: t.requests.iter().find(|r| r.yielded.is_some()).map_or(0, |r| r.family),
        });
    }

    // ---- the mechanism

    /// Unmaps a page; `withdraw` drops its kept copy too (an eviction or unload; a process that
    /// ends leaves its kept pages as orphans).
    fn unmap(&self, st: &mut State, key: Key, class: Class, req: Option<u64>, withdraw: bool) -> Result<()> {
        let Some(home) = st.homes.remove(&key) else { return Ok(()) };
        let (va, off, span) = st.place(key)?;
        // Custody's handle holds the memory until the Withdraw is sent: held (P) meanwhile.
        if home.exported && withdraw && self.withdraw(home.identity.clone(), key.1, span).is_some() {
            st.table.pending += span;
        }
        st.last_unmap = Some(self.mech.post(Mech::Unmap { va, off, span, phys: home.phys, ready: home.ready }, span));
        st.table.inflight = self.mech.pending();
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
                self.post(Call::Disown { fp: identity, page: key.1 });
                self.unlock(st);
                continue;
            }
            let (va, off, span) = st.place(key)?;
            let phys = self.backend.import(handle, span)?;
            let gen = st.tick();
            // Pinned (filling) until the mechanism maps it.
            self.mech.post(Mech::Adopt { key, gen, va, off, phys: phys.id, len: phys.len, identity: identity.clone() }, 0);
            st.homes.insert(key, Home { phys: Some(phys), ready: Event::DONE, gen, filled: false, filling: true, transforming: false, exported: true, origin: Origin::Kept, speculative: false, identity, keep_after: false, keeping: false });
            st.sync(key);
            let s = st.fills(req, key, span, Origin::Kept);
            s.moved.kept.bytes += span;
            s.moved.kept.ns += start.elapsed().as_nanos() as u64;
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
        let gen = st.tick();
        let identity = st.mech(key.0)?.spec.fingerprint.key();
        let decided = st.table.free();
        // Counted mapped (and pinned, filling) now; the mechanism creates and maps it after
        // the unmaps decided before it, then queues its fill.
        st.homes.insert(key, Home { phys: None, ready: Event::DONE, gen, filled: false, filling: true, transforming: false, exported: false, origin: Origin::Store, speculative, identity, keep_after: false, keeping: false });
        st.sync(key);
        self.mech.post(Mech::Map { key, gen, va, off, span, job: Job { key, gen, req, span, after: vec![] }, speculative, decided }, 0);
        Ok(true)
    }

    /// A stale page (another fill identity, or a transform released undone) nobody pins is
    /// filled again in place: its mapping and memory stay, the copy waits in-stream on its
    /// pending release events, and its binder waits in-stream on the fill. No host thread waits
    /// on a device event.
    fn refill_in_place(&self, st: &mut State, key: Key, req: Option<u64>) -> Result<()> {
        let (va, off, span) = st.place(key)?;
        let identity = st.mech(key.0)?.spec.fingerprint.key();
        let gen = st.tick();
        let after = st.table.page_mut(key).map(|p| std::mem::take(&mut p.releasing)).unwrap_or_default();
        st.releasing.remove(&key);
        let Some(h) = st.homes.get_mut(&key) else { return Ok(()) };
        let (old, exported, old_identity) = (std::mem::replace(&mut h.ready, Event::DONE), h.exported, h.identity.clone());
        // Kept, or with a Keep queued: never written again; a fresh page takes its place.
        let phys = if exported || h.keeping { h.phys.take() } else { None };
        let renew = phys.is_some();
        *h = Home { phys: h.phys.take(), ready: Event::DONE, gen, filled: false, filling: true, transforming: false, exported: false, origin: Origin::Store, speculative: false, identity, keep_after: false, keeping: false };
        st.sync(key);
        let after: Vec<Event> = after.into_iter().map(Event).collect();
        let job = Job { key, gen, req, span, after: vec![] };
        if renew {
            // Read-only and kept under its old identity (or about to be): never written or
            // protected again. Custody gives the old key up; a fresh page takes its place once
            // its consumers are done.
            if exported && self.withdraw(old_identity, key.1, span).is_some() {
                st.table.pending += span;
            }
            self.backend.retire(old);
            self.mech.post(Mech::Renew { key, gen, va, off, span, old: phys, after, job }, 0);
            return Ok(());
        }
        // Read-write and nobody else's: filled again in place, after its consumers in-stream.
        self.backend.retire(old);
        self.queue_fill(Job { after, ..job }, false)
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
        let Some(phys) = home.phys.as_ref().map(|p| (p.id, p.len)).filter(|_| !home.keeping) else { return };
        home.keeping = true;
        self.mech.post(Mech::Keep { key, gen: home.gen, va, off, span, phys: phys.0, len: phys.1 }, 0);
    }

    /// A fill thread's batch: each page from the tier, else the store, in as few engine reads as
    /// its sets allow; then each page's ready event.
    fn fill(&self, jobs: &[Job]) {
        self.fill_batch(jobs);
        // Their fills' events follow them in-stream (or were dropped): nobody needs them now.
        jobs.iter().flat_map(|j| &j.after).for_each(|e| self.backend.retire(*e));
    }

    fn fill_batch(&self, jobs: &[Job]) {
        // Under one lock: each live job's place, and its set's spec and tier layout.
        let (sets, pages) = {
            let st = self.state.lock();
            let mut sets: BTreeMap<u32, (SetSpec, Option<Arc<tier::Held>>)> = BTreeMap::new();
            let mut pages = vec![];
            for job in jobs {
                let (Some(h), Ok(m), Ok(place)) = (st.homes.get(&job.key), st.mech(job.key.0), st.place(job.key)) else { continue };
                if h.gen == job.gen {
                    sets.entry(job.key.0).or_insert_with(|| (m.spec.clone(), m.held.clone()));
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
        let mut sticky = false;
        let mut st = self.state.lock();
        for ((job, (.., span)), landed) in pages.iter().zip(landed) {
            let span = *span;
            if st.homes.get(&job.key).is_none_or(|h| h.gen != job.gen) {
                // The page changed under the fill: its event is nobody's.
                if let Ok((_, ready)) = landed {
                    self.backend.retire(ready);
                }
                continue;
            }
            match landed {
                Ok((origin, ready)) => {
                    // Stored bytes are the weights; a declared set's tier holds its transformed
                    // bytes.
                    let fill = sets.get(&job.key.0).map(|(spec, _)| &spec.fingerprint.fill);
                    let filled = matches!(fill, Some(Fill::Stored)) || (origin == Origin::Tier && matches!(fill, Some(Fill::Declared(_))));
                    let speculative = st.homes.get(&job.key).is_some_and(|h| h.speculative);
                    if let Some(h) = st.homes.get_mut(&job.key) {
                        h.ready = ready;
                        h.filling = false;
                        h.filled = filled;
                        h.origin = origin;
                    }
                    st.sync(job.key);
                    let s = st.fills(job.req, job.key, span, origin);
                    let f = if origin == Origin::Tier { &mut s.moved.tier } else { &mut s.moved.store };
                    f.bytes += span;
                    // The batch's time, shared by its bytes.
                    f.ns += (u128::from(ns) * u128::from(span) / u128::from(bytes)) as u64;
                    if speculative {
                        s.moved.speculative += span;
                    }
                    if filled {
                        self.keep(&mut st, job.key);
                    }
                    if filled && origin == Origin::Store {
                        self.save_later(&st, job.key, job.req);
                    }
                }
                Err(e) => {
                    tracing::warn!("varena: filling {:?} failed: {e}", job.key);
                    sticky |= e.is_sticky();
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
        }
        self.publish(&st);
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

    /// The batch's copies: from the tier where ready, else straight from the store (the tier
    /// gets its copy from the filled page afterwards); one engine read for each per set. Each
    /// page's origin, or its error.
    #[allow(clippy::type_complexity)]
    fn copy(&self, sets: &BTreeMap<u32, (SetSpec, Option<Arc<tier::Held>>)>, pages: &[(&Job, (Va, u64, u64))]) -> Vec<Result<Origin>> {
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
                Copying { key: job.key, dst, out }
            })
            .collect();
        let unregister = |p| self.backend.unregister_host(p);
        for (set, (spec, held)) in sets {
            let todo = |c: &Copying| c.key.0 == *set && c.out.is_none();
            if let Some(held) = held {
                // A copy from the tier, each region settled after (unlocked if the tier asked
                // for it).
                let read = |copies: &mut [Copying]| {
                    for c in copies.iter_mut().filter(|c| todo(c)) {
                        if let Err(e) = held.pin(c.key.1, |p, l| self.backend.register_host(p, l), unregister) {
                            c.out = Some(Err(e));
                        }
                    }
                    let mut from: Vec<&mut Copying> = copies.iter_mut().filter(|c| todo(c)).collect();
                    let copied = held.read(&from.iter().map(|c| (c.key.1, c.dst)).collect::<Vec<_>>());
                    from.iter().for_each(|c| held.settle(c.key.1, unregister));
                    match copied {
                        Ok(copied) => from.iter_mut().zip(copied).filter(|(_, ok)| *ok).for_each(|(c, _)| c.out = Some(Ok(Origin::Tier))),
                        Err(e) => from.iter_mut().for_each(|c| c.out = Some(Err(e.clone()))),
                    }
                };
                read(&mut copies);
            }
            let direct: Vec<(u32, Dst)> = copies.iter().filter(|c| todo(c)).map(|c| (c.key.1, c.dst)).collect();
            if !direct.is_empty() {
                let stored = self.store(spec, &direct).map(|()| Origin::Store);
                copies.iter_mut().filter(|c| todo(c)).for_each(|c| c.out = Some(stored.clone()));
            }
        }
        copies.into_iter().map(|c| c.out.unwrap_or_else(|| Err(Error::Internal("a page left unfilled".into())))).collect()
    }

    /// A fill thread's batch failed outside its copies (a panic): its pages go, their binders fail.
    fn fail(&self, jobs: &[Job], e: Error) {
        let mut st = self.state.lock();
        for job in jobs {
            if st.homes.get(&job.key).is_some_and(|h| h.gen == job.gen) {
                if let Some(h) = st.homes.get_mut(&job.key) {
                    h.filling = false;
                }
                let _ = self.unmap(&mut st, job.key, Class::P, None, true);
                st.failed.insert(job.key, e.clone());
            }
        }
        self.unlock(st);
        self.landed.notify_all();
    }

    /// `first` and the jobs queued behind it (demand misses first), up to the burst's bytes so
    /// far: batches double, so the first pages are ready soonest and a few reads carry the
    /// rest. Never less than the engine keeps in flight (its threads' requests): a smaller read
    /// leaves its threads idle.
    fn batch(&self, first: Job, demand: &Receiver<Job>, ahead: &Receiver<Job>) -> Vec<Job> {
        let engine = tensorfs::io::ReadOpts::default();
        let budget = {
            let mut b = self.burst.0.lock();
            b.batches += 1;
            b.bytes.max((engine.threads * engine.request) as u64)
        };
        let mut bytes = first.span;
        let mut jobs = vec![first];
        while bytes < budget {
            let Ok(job) = demand.try_recv().or_else(|_| ahead.try_recv()) else { break };
            bytes += job.span;
            jobs.push(job);
        }
        let mut b = self.burst.0.lock();
        b.queued = b.queued.saturating_sub(jobs.len() as u64);
        jobs
    }

    /// Queues a page's fill: a demand miss, or `ahead` of its use.
    fn queue_fill(&self, job: Job, ahead: bool) -> Result<()> {
        self.burst.0.lock().queued += 1;
        let queue = if ahead { &self.jobs.ahead } else { &self.jobs.demand };
        queue.send(job).map_err(|_| {
            self.burst.0.lock().queued -= 1;
            Error::Closed("the fill threads".into())
        })
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

    /// Waits until no page is decided to fill (one being mapped or queued counts); false once
    /// the Memory is going.
    fn after_fills(&self) -> bool {
        let mut st = self.lock();
        while !self.burst.0.lock().closed && st.homes.values().any(|h| h.filling) {
            self.wait(&mut st);
        }
        !self.burst.0.lock().closed
    }

    /// Queues a copy of a page just filled with its tier's bytes into the RAM tier, unless it
    /// has no tier layout or the tier holds it.
    fn save_later(&self, st: &State, key: Key, req: Option<u64>) {
        let Ok(m) = st.mech(key.0) else { return };
        // Bytes of the set's current identity only: its layout is that identity's.
        let current = |h: &&Home| h.identity == m.spec.fingerprint.key();
        let Some(gen) = st.homes.get(&key).filter(current).map(|h| h.gen).filter(|_| m.held.as_ref().is_some_and(|h| !h.holds(key.1))) else { return };
        self.saves.post(Save { key, gen, tier: m.tier.clone(), req }, 0);
    }

    /// Copies a filled page into the RAM tier when the host has the room: speculation, so it
    /// soft-fails. The page is pinned meanwhile (a consumer: never unmapped under the copy); one
    /// unmapped or changed since it was queued is not copied.
    fn save(&self, s: &Save) {
        let (pin, held, (va, off, span), ready) = {
            let mut st = self.lock();
            let home = st.homes.get(&s.key).filter(|h| h.gen == s.gen && h.filled && !h.filling && !h.transforming).map(|h| h.ready);
            // Only into the layout of the identity it was filled under.
            let held = st.mech(s.key.0).ok().filter(|m| m.tier == s.tier).and_then(|m| m.held.clone()).filter(|h| !h.holds(s.key.1));
            let (Some(ready), Some(held), Ok(place)) = (home, held, st.place(s.key)) else { return };
            let pin = st.tick();
            st.pins.insert(pin, (s.key, None, std::thread::current().id()));
            st.saving = Some(s.key);
            st.sync(s.key);
            self.unlock(st);
            (pin, held, place, ready)
        };
        let start = Instant::now();
        let saved = self.tier.as_ref().is_some_and(|t| t.room(span))
            && held
                .fill(&[s.key.1], |regions| {
                    regions.iter().try_for_each(|(_, dst)| match dst {
                        // SAFETY: the region is mapped writable for its `len` bytes and
                        // write-locked by this process.
                        Dst::Host { ptr, len } => self.backend.save(va, off, unsafe { std::slice::from_raw_parts_mut(*ptr, (*len).min(span as usize)) }, ready),
                        Dst::Device { .. } => Err(Error::Internal("a tier region is host memory".into())),
                    })
                })
                .map_err(|e| tracing::warn!("varena: copying {:?} into the RAM tier failed: {e}", s.key))
                .is_ok();
        let ns = start.elapsed().as_nanos() as u64;
        let mut st = self.lock();
        st.pins.remove(&pin);
        st.saving = None;
        st.sync(s.key);
        // Its class while pinned may have been published meanwhile: no longer.
        self.publish(&st);
        if saved {
            let stats = match s.req.filter(|r| st.stats.contains_key(r)) {
                Some(r) => st.stats.entry(r).or_default(),
                None => &mut st.unclaimed,
            };
            stats.moved.saved.bytes += span;
            stats.moved.saved.ns += ns;
        }
        self.unlock(st);
        self.landed.notify_all();
    }

    /// The set's regions from its store into their destinations, in one engine read.
    #[cfg(unix)]
    fn store(&self, spec: &SetSpec, regions: &[(u32, Dst)]) -> Result<()> {
        let placements = regions.iter().map(|(r, dst)| spec.layout.region(*r).map(|region| Placement { items: &region.items, origin: region.offset, dst: *dst })).collect::<Result<Vec<_>>>()?;
        let threads = tensorfs::io::ReadOpts::default().threads;
        // SAFETY: each `dst` is a mapped page (or a locked tier region) only this fill writes.
        unsafe { spec.source.read_placements(&placements, threads, tensorfs::io::source::Mode::Cache, &self.tally) }?;
        Ok(())
    }

    #[cfg(windows)]
    fn store(&self, spec: &SetSpec, _: &[(u32, Dst)]) -> Result<()> {
        match *spec.source {}
    }

    // ---- demands

    /// Decides and applies until the trigger is met or ends. Remote rungs (the trim hook,
    /// orphans, asks) run with the lock released; bytes asked of others are claimed in this
    /// row meanwhile, so nobody takes them first; then the procedure decides again with them
    /// tried and what every round freed counted.
    fn demand(&self, trigger: Trigger, req: Option<u64>) -> Result<Vec<Action>> {
        let id = {
            let mut st = self.lock();
            let id = st.tick();
            self.unlock(st);
            id
        };
        let out = self.rounds(id, trigger, req);
        let mut st = self.lock();
        if st.claims.remove(&id).is_some() {
            self.publish(&st);
        }
        self.unlock(st);
        out
    }

    fn rounds(&self, id: u64, mut trigger: Trigger, req: Option<u64>) -> Result<Vec<Action>> {
        // What this demand's own rounds unmapped while its asks were out: still its room.
        let mut own = 0;
        loop {
            let mut st = self.lock();
            self.refresh(&mut st)?;
            if matches!(trigger, Trigger::Oom { .. }) {
                st.table.headroom = u64::try_from(platform::contain::host_memory::read().available).unwrap_or(0);
            }
            // A page whose create fell short, measured the same again: nothing more is known.
            if let Trigger::Bind { page, .. } = &trigger {
                if let Some((need, at)) = st.short.remove(page) {
                    if at == st.table.free() {
                        self.unlock(st);
                        return Ok(vec![Action::Exhausted { need }]);
                    }
                }
            }
            let actions = decide(&st.table, &trigger);
            let (mut remote, mut shrinks, mut freed) = (vec![], vec![], 0);
            for a in &actions {
                freed += self.apply(&mut st, a, req, &mut remote, &mut shrinks)?;
            }
            let asked: u64 = remote.iter().map(|a| if let Action::Ask { bytes, .. } = a { *bytes } else { 0 }).sum();
            if remote.is_empty() {
                st.claims.remove(&id);
            } else {
                own += freed;
                st.claims.insert(id, own + asked);
            }
            // A pass whose cut changed its room decides again at the new levels.
            if let (Trigger::Pass { room, unmeasured, set, .. }, Some(r)) = (&mut trigger, req) {
                if actions.iter().any(|a| matches!(a, Action::Cut { .. })) && !remote.is_empty() {
                    (*room, *unmeasured) = self.pass_room(&mut st, r, *set)?;
                }
            }
            // A tier copy in flight holds its page like a consumer: a decision short of bytes
            // waits for it, then decides again.
            if remote.is_empty() && st.saving.is_some() && matches!(actions.last(), Some(Action::Exhausted { .. } | Action::Park(_))) {
                while st.saving.is_some() {
                    self.wait(&mut st);
                }
                self.unlock(st);
                continue;
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
                if let Action::Cut { site, .. } | Action::CutOutside { site, .. } | Action::Propose { site, .. } = a {
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
                        tried.freed += self.row.ask(row, ceiling, bytes)?;
                    }
                    // Custody lets go of the kept pages unmapped: then they are free.
                    Action::Settle => self.to_custody.settle(),
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
            Action::Trim(_) | Action::ReleaseOrphans(_) | Action::Ask { .. } | Action::Settle => remote.push(a.clone()),
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
            Action::CutOutside { req, site, shape } => {
                if let Some(r) = st.table.request_mut(*req) {
                    *r.outside.entry((site.clone(), shape.clone())).or_insert(0) += 1;
                }
                if let Some(s) = st.stats(Some(*req)) {
                    s.cuts += 1;
                }
            }
            Action::Cut { req, set, site, failed_at } => {
                if let Some(shape) = st.active.get(&(*req, *set)).map(|a| a.shape.clone()) {
                    let name = st.mech(*set)?.spec.name.clone();
                    let key = room_key(&name, &shape, &st.levels(*set, &shape));
                    st.failed_at.insert(key, *failed_at);
                    st.cut(*set, &shape, site);
                }
                if let Some(s) = st.stats(Some(*req)) {
                    s.cuts += 1;
                }
            }
            Action::Relax { req, set, site } => {
                if let Some(shape) = st.active.get(&(*req, *set)).map(|a| a.shape.clone()) {
                    if let Some(l) = st.cuts.get_mut(&(*set, shape)).and_then(|c| c.get_mut(site)) {
                        *l = l.saturating_sub(1);
                    }
                }
            }
            Action::Propose { req, set, site, level, failed_at } => {
                if let Some(shape) = st.active.get(&(*req, *set)).map(|a| a.shape.clone()) {
                    let levels = st.levels(*set, &shape);
                    let now = levels.get(site).copied().unwrap_or(0);
                    if let Some(f) = failed_at {
                        let name = st.mech(*set)?.spec.name.clone();
                        st.failed_at.insert(room_key(&name, &shape, &levels), *f);
                    }
                    let p = st.proposed.entry((*set, shape)).or_default();
                    // A finer proposal (an OOM's) stands over a coarser one.
                    let finer = p.get(site).is_some_and(|q| *q > now);
                    if *level > now {
                        let q = p.entry(site.clone()).or_insert(0);
                        *q = (*q).max(*level);
                    } else if !finer {
                        p.insert(site.clone(), *level);
                    }
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
            let freed = crate::caught(|| Ok(f(bytes))).unwrap_or(0);
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
        let freed = crate::caught(|| Ok(hook())).unwrap_or(0);
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
                    // Its resident bytes, for placement: exact, from its layout.
                    ledger::record_set(&self.shared, &spec.name, layout.nbytes)?;
                    let va = self.backend.reserve(layout.nbytes)?;
                    let pages = layout.regions.iter().map(|r| Page { bytes: r.span, ..Page::default() }).collect();
                    let set = Set { name: spec.name.clone(), pages, declared: spec.fingerprint.declared() };
                    let i = st.sets.len() as u32;
                    st.sets.push(Some(SetMech { tier: tier_name(&spec), spec, va, held: None, closing: false }));
                    st.table.sets.push(Some(set));
                    (i, false)
                }
            };
            self.unlock(st);
            out
        };
        self.open_tier(id)?;
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

    /// The set's layout in the RAM tier, opened unlocked (a call to the tier) unless it is
    /// open: at its load, and again at a load or pass start after the tier took it back.
    fn open_tier(&self, set: u32) -> Result<()> {
        let Some(tier) = &self.tier else { return Ok(()) };
        let layout = {
            let st = self.lock();
            let m = st.mech(set)?;
            let l = (m.held.is_none()).then(|| (m.spec.layout.clone(), m.tier.clone()));
            self.unlock(st);
            l
        };
        let Some((layout, name)) = layout else { return Ok(()) };
        let regions: Vec<(u64, u64)> = layout.regions.iter().map(|r| (r.offset, r.span)).collect();
        let held = tier.open_layout(&name, regions, &self.row.ask_name()).map_err(|e| tracing::warn!("varena: tier: {e}")).ok().map(Arc::new);
        let mut st = self.lock();
        if let Some(Some(m)) = st.sets.get_mut(set as usize) {
            if m.held.is_none() {
                m.held = held;
            }
        }
        self.unlock(st);
        Ok(())
    }

    /// Fills the set's layout in the RAM tier from the store as host headroom allows:
    /// everything left in one read when it has the room, else half as much, until a region does
    /// not fit. A declared set's tier holds its transformed bytes: only its filled pages go
    /// there.
    fn load_ram(&self, set: u32) -> Result<()> {
        let (spec, held) = {
            let st = self.lock();
            let m = st.mech(set)?;
            let out = (m.spec.clone(), m.held.clone());
            self.unlock(st);
            out
        };
        let (Some(held), Some(tier), false) = (held, &self.tier, matches!(spec.fingerprint.fill, Fill::Declared(_))) else { return Ok(()) };
        let regions = &spec.layout.regions;
        let (mut at, mut want) = (0, regions.len());
        while want > 0 {
            let batch: Vec<u32> = (at..at + want).map(|r| r as u32).collect();
            let bytes = regions.iter().skip(at).take(want).map(|r| r.span).sum();
            if !tier.room(bytes) {
                want /= 2;
                continue;
            }
            held.fill(&batch, |hosts| self.store(&spec, hosts))?;
            at += want;
            want = regions.len() - at;
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

    /// The largest absolute room measured for any of these sets, at any shape; `None` if one
    /// has none.
    fn largest_room(&self, st: &State, sets: &[u32]) -> Option<u64> {
        sets.iter()
            .map(|s| {
                let name = st.mech(*s).ok()?.spec.name.clone();
                let rooms = self.shared.facts(&format!("room/{}|", esc(&name))).ok()?;
                rooms.iter().filter_map(|(_, v)| serde_json::from_slice::<Room>(v).ok()).map(|r| r.absolute()).max()
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
        // A cold request's record carries the fills since the last one ended (its loads).
        let earlier = std::mem::take(&mut st.unclaimed);
        st.stats.insert(rid, Stats { request: id.to_string(), moved: earlier.moved, trace: earlier.trace, ..Stats::default() });
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
            let pinned = st.table.page(key).is_some_and(|p| !p.pins.is_empty());
            // Waiting on pins only this thread holds would wait on itself.
            let me = std::thread::current().id();
            let mine = pinned && st.pins.values().filter(|(k, ..)| *k == key).all(|(.., t)| *t == me);
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
        self.publish(&st);
        self.unlock(st);
        self.speculate(req);
        self.backend.wait(ready, stream)?;
        Ok(Pin { ptr: va.base + off, len: span, ready, fresh, inner: self.clone(), key, id, stream, transform, transformed: AtomicBool::new(false) })
    }

    /// A configuration's measured room: exact, else bounded by the nearest larger measured
    /// shape of the same entry, else scaled from the nearest smaller by its largest dimension
    /// ratio.
    fn measured(&self, st: &mut State, name: &str, entry: &str, shape: &str, levels: &BTreeMap<String, u32>) -> Result<Option<Room>> {
        if let Some(room) = self.memo(st, &room_key(name, &config(entry, shape), levels))? {
            return Ok(Some(room));
        }
        let want = cuts(levels);
        let rooms = self.shared.facts(&format!("room/{}|{}|", esc(name), esc(entry)))?;
        let Some(ours) = dims(shape) else { return Ok(None) };
        let mut larger: Option<(u64, Room)> = None;
        let mut smaller: Option<(u64, f64, Room)> = None;
        for (_, v) in &rooms {
            let Ok(room) = serde_json::from_slice::<Room>(v) else { continue };
            let Some(theirs) = dims(&room.shape) else { continue };
            if room.entry != entry || room.cuts != want || theirs.len() != ours.len() {
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
        Ok(smaller.map(|(_, ratio, room)| Room { start: (room.start as f64 * ratio) as u64, delta: (room.bytes() as f64 * ratio) as u64, slack: 0, context: 0, ..room }))
    }

    /// The pass's room at its cut levels (`None`: unmeasured) and the unmeasured reserve (the
    /// largest context growth measured on the machine), with its sites' finer and coarser
    /// measured rooms for the decision.
    fn room(&self, st: &mut State, set: u32, (entry, dims, shape): (&str, &str, &str), allocated: u64) -> Result<(Option<u64>, u64, Vec<CutSite>)> {
        let name = st.mech(set)?.spec.name.clone();
        let levels = st.levels(set, shape);
        let now = self.measured(st, &name, entry, dims, &levels)?;
        let measured: Vec<SitePeak> = now.iter().flat_map(|r| r.sites.iter().cloned()).collect();
        // A collective's site measured on any rank is offered for agreement from the first pass.
        for s in measured.iter().filter(|s| s.collective) {
            st.collective.insert((set, shape.to_string(), s.name.clone()));
        }
        let mut names: BTreeSet<String> = measured.iter().map(|s| s.name.clone()).collect();
        names.extend(levels.keys().cloned());
        names.extend(st.collective.iter().filter(|(s, sh, _)| *s == set && sh == shape).map(|(.., n)| n.clone()));
        let mut sites = vec![];
        for n in names {
            let level = levels.get(&n).copied().unwrap_or(0);
            let site = measured.iter().find(|s| s.name == n);
            let peak = site.map_or(0, |s| s.peak);
            let finer = self.measured(st, &name, entry, dims, &with(&levels, &n, level + 1))?.map(|r| allocated + r.bytes());
            let coarser = match level {
                0 => None,
                l => {
                    let at = with(&levels, &n, l - 1);
                    let failed = st.failed_at.get(&room_key(&name, shape, &at)).copied();
                    self.measured(st, &name, entry, dims, &at)?.map(|r| (allocated + r.bytes(), failed))
                }
            };
            let key = (set, shape.to_string(), n.clone());
            let single = st.single.contains(&key) || site.is_some_and(|s| !s.collective);
            // Unknown: 0, never cut at a pass start.
            let finest = st.finest.get(&key).copied().or(site.map(|s| s.finest)).unwrap_or(0);
            sites.push(CutSite { stuck: st.stuck.contains(&key), collective: st.collective.contains(&key) || !single, name: n, level, finest, peak, finer, coarser });
        }
        let growth = self.shared.facts("room/")?.iter().filter_map(|(_, v)| serde_json::from_slice::<Room>(v).ok()).map(|r| r.context).max().unwrap_or(0);
        Ok((now.map(|r| allocated + r.bytes()), allocated + growth, sites))
    }

    /// The request's room for its pass over `set` now (its levels may have just changed),
    /// beside its other running passes; the pass's sites set on the request with the same
    /// offset.
    fn pass_room(&self, st: &mut State, req: u64, set: u32) -> Result<(Option<u64>, u64)> {
        let Some((id, entry, dims, shape, allocated)) = st.active.get(&(req, set)).map(|a| (a.id, a.entry.clone(), a.dims.clone(), a.shape.clone(), a.allocated)) else { return Ok((None, 0)) };
        let (room, unmeasured, mut sites) = self.room(st, set, (&entry, &dims, &shape), allocated)?;
        let name = st.mech(set)?.spec.name.clone();
        let exact = self.memo(st, &room_key(&name, &shape, &st.levels(set, &shape)))?.is_some();
        if let Some(a) = st.active.get_mut(&(req, set)) {
            a.room = room.unwrap_or(unmeasured);
            a.measured = exact;
        }
        let joint = |own: u64| st.joint(req, id, allocated, own);
        for s in &mut sites {
            s.finer = s.finer.map(joint);
            s.coarser = s.coarser.map(|(c, f)| (joint(c), f));
        }
        let (room, unmeasured) = (room.map(joint), joint(unmeasured));
        if let Some(r) = st.table.request_mut(req) {
            r.sites.insert(set, sites);
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

    fn pass(&self, req: u64, set: u32, entry: &str, dims: &str, torch: Torch) -> Result<u64> {
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
            st.active.insert((req, set), Active { id, set, entry: entry.to_string(), dims: dims.to_string(), shape: config(entry, dims), allocated: torch.allocated, room: torch.allocated, context, measured: false });
            if let Some(r) = st.table.request_mut(req) {
                r.episode = Default::default();
                r.claimed = 0;
            }
            let (room, unmeasured) = self.pass_room(&mut st, req, set)?;
            let step: Vec<Key> = st.table.request(req).and_then(|r| r.steps.get(r.cursor.step as usize)).map(|(sets, _)| st.table.pages().filter(|(k, p)| sets.contains(&k.0) && !p.mapped).map(|(k, _)| k).collect()).unwrap_or_default();
            let orphans = st.table.orphans > 0;
            self.unlock(st);
            (id, room, unmeasured, step, orphans)
        };
        // Kept pages of this step are adopted before the demand could release them as orphans.
        if orphans {
            self.adopt(step_pages, Some(req))?;
        }
        let actions = self.demand(Trigger::Pass { req, set, room, unmeasured, tried: Tried::default() }, Some(req))?;
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
        let uses: Vec<Call> = st.sets.iter().enumerate().filter_map(|(i, m)| m.as_ref().map(|m| Call::Use { digest: m.tier.clone(), distance: r.distance((i as u32, 0)).map(|d| d.0) })).collect();
        uses.into_iter().for_each(|c| self.post(c));
    }

    /// Torch's figures now, and the context derived from them: this process's device bytes
    /// less torch's reserved and the pages it created (an imported page is not charged to the
    /// importer's process).
    fn measure(&self, st: &mut State, torch: Torch) {
        st.table.allocated = torch.allocated;
        st.table.reserved = torch.reserved;
        st.table.context = match self.backend.process_used() {
            Ok(used) => {
                let created: u64 = st.homes.iter().filter(|(_, h)| h.origin != Origin::Kept).filter_map(|(k, _)| st.table.page(*k)).map(|p| p.bytes).sum();
                used.saturating_sub(torch.reserved + created)
            }
            // Not measurable here: the context as open measured it.
            Err(_) => self.backend.opened_context(),
        };
    }

    fn end_pass(&self, req: u64, pass: u64, measured: Measured, torch: Torch) -> Result<()> {
        let mut st = self.lock();
        self.measure(&mut st, torch);
        let Some(key) = st.active.iter().find(|((r, _), a)| *r == req && a.id == pass).map(|(k, _)| *k) else {
            self.unlock(st);
            return Ok(());
        };
        let Some(active) = st.active.remove(&key) else { return Ok(()) };
        let name = st.mech(active.set)?.spec.name.clone();
        let levels = st.levels(active.set, &active.shape);
        let key = room_key(&name, &active.shape, &levels);
        let growth = st.table.context.saturating_sub(active.context);
        let room = Room { entry: active.entry.clone(), shape: active.dims.clone(), cuts: cuts(&levels), start: active.allocated, delta: measured.allocated_peak.saturating_sub(active.allocated), slack: measured.slack, context: growth, sites: measured.sites.clone() };
        // A site whose finer level did not lower the allocated peak is not cut further here.
        for (site, level) in levels.iter().filter(|(_, l)| **l > 0) {
            if self.memo(&mut st, &room_key(&name, &active.shape, &with(&levels, site, level - 1)))?.is_some_and(|c| room.delta >= c.delta) {
                st.stuck.insert((active.set, active.shape.clone(), site.clone()));
            }
        }
        // The latest completed measurement is the room (it comes down when a pass needs less).
        if self.memo(&mut st, &key)?.as_ref() != Some(&room) {
            ledger::record_room(&self.shared, &key, &room)?;
            st.rooms.insert(key.clone(), room.clone());
            self.shared.ring();
        }
        // The request's working set as this pass really used it: its step's pages and the room
        // it measured, beside its other running passes.
        let own = active.allocated + room.bytes();
        let joint = st.joint(req, pass, active.allocated, own);
        if let Some(r) = st.table.request_mut(req) {
            let step = r.cursor.step as usize;
            if let Some(known) = r.rooms.get_mut(step) {
                *known = Some(known.unwrap_or(0).max(joint));
            }
        }
        let w = st.table.request(req).map_or(0, |r| st.table.step_held(r) + joint.max(r.room + r.claimed));
        if let Some(s) = st.step_stats(req) {
            s.w = s.w.max(w);
        }
        if let Some(s) = st.stats(Some(req)) {
            s.peak = s.peak.max(w);
        }
        self.publish(&st);
        self.unlock(st);
        Ok(())
    }

    fn oom(&self, req: u64, set: Option<u32>, need: Option<u64>, site: Option<Site<'_>>, torch: Torch) -> Result<Next> {
        {
            let mut st = self.lock();
            self.measure(&mut st, torch);
            if let Some(set) = set {
                self.outgrown(&mut st, req, set, torch.allocated + need.unwrap_or(0))?;
                if let (Some(s), Some(shape)) = (site, st.active.get(&(req, set)).map(|a| a.shape.clone())) {
                    let key = (set, shape, s.name.to_string());
                    st.finest.insert(key.clone(), s.finest);
                    match s.collective {
                        true => st.collective.insert(key),
                        false => st.single.insert(key),
                    };
                }
                // The decision reads this set's sites.
                self.pass_room(&mut st, req, set)?;
            }
            self.unlock(st);
        }
        let site_of = site.map(|s| crate::decide::OomSite { name: s.name.to_string(), collective: s.collective, finest: s.finest, shape: s.shape.to_string() });
        let trigger = Trigger::Oom { req, set, need, site: site_of, tried: Tried::default() };
        let actions = self.demand(trigger, Some(req))?;
        let mut st = self.lock();
        let stats = |st: &State| st.stats.get(&req).cloned().unwrap_or_default();
        let next = match actions.iter().rev().find(|a| matches!(a, Action::Cut { .. } | Action::CutOutside { .. } | Action::Park(_) | Action::Exhausted { .. })) {
            Some(Action::Cut { set, site, .. }) => {
                let level = st.active.get(&(req, *set)).and_then(|a| st.cuts.get(&(*set, a.shape.clone()))).and_then(|c| c.get(site).copied());
                Next::Cut(level.unwrap_or(1))
            }
            Some(Action::CutOutside { site, shape, .. }) => Next::Cut(st.table.request(req).and_then(|r| r.outside.get(&(site.clone(), shape.clone())).copied()).unwrap_or(1)),
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
        let unmapped = st.last_unmap;
        self.publish(&st);
        self.unlock(st);
        // Its unmaps are done before the caller retries the allocation. The mechanism never
        // waits on a pin: it unmaps only unpinned pages (a renewal waits on device events).
        if let Some(t) = unmapped {
            self.mech.settle_to(t + 1);
        }
        Ok(next)
    }

    /// An OOM in a pass whose room was measured: the room it needed (`allocated`, its
    /// allocation with the failing op's) is recorded at once, so the next pass reserves it.
    fn outgrown(&self, st: &mut State, req: u64, set: u32, allocated: u64) -> Result<()> {
        let Some((shape, start)) = st.active.get(&(req, set)).filter(|a| a.measured).map(|a| (a.shape.clone(), a.allocated)) else { return Ok(()) };
        let name = st.mech(set)?.spec.name.clone();
        let key = room_key(&name, &shape, &st.levels(set, &shape));
        let Some(known) = self.memo(st, &key)? else { return Ok(()) };
        let delta = allocated.saturating_sub(start);
        if delta > known.delta {
            let room = Room { delta, ..known };
            ledger::record_room(&self.shared, &key, &room)?;
            st.rooms.insert(key, room);
        }
        Ok(())
    }

    fn end_request(&self, req: u64) -> Stats {
        let mut st = self.lock();
        let name = st.table.request(req).map(|r| r.name.clone()).unwrap_or_default();
        st.active.retain(|(r, _), _| *r != req);
        st.synthetic.remove(&req);
        st.table.requests.retain(|r| r.id != req);
        let owned: Vec<u32> = st.table.caches.iter().filter(|c| c.owner == name).map(|c| c.id).collect();
        let holding: Vec<u32> = st.table.caches.iter().filter(|c| c.owner == name && c.bytes > 0).map(|c| c.id).collect();
        let shrinks: Vec<(u32, u64, Shrink)> = holding.iter().filter_map(|id| st.shrinks.get(id).map(|s| (*id, u64::MAX, s.clone()))).collect();
        let leftover: u64 = st.pins.values().filter(|(_, r, _)| *r == Some(req)).filter_map(|(k, ..)| st.table.page(*k)).map(|p| p.bytes).sum();
        let mut stats = st.stats.remove(&req).unwrap_or_default();
        if let Some(key) = st.peaks.remove(&req) {
            let _ = ledger::record_peak(&self.shared, &key, stats.peak);
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
        let uses: Vec<Call> = st.sets.iter().flatten().map(|m| Call::Use { digest: m.tier.clone(), distance: None }).collect();
        if self.tier.is_some() {
            uses.into_iter().for_each(|c| self.post(c));
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
            wanted.settle = st.table.pending > 0;
            let unmapped = st.last_unmap.filter(|_| freed > 0);
            self.publish(&st);
            self.unlock(st);
            // Its own unmaps first (the mechanism's, up to its last): the asker then finds the
            // bytes free. Kept pages' bytes are marked pending (custody's handle holds them):
            // the reply does not wait on custody; the trim thread settles it and publishes them
            // free.
            if let Some(t) = unmapped {
                self.mech.settle_to(t + 1);
            }
            if wanted.trim || wanted.settle || !wanted.shrinks.is_empty() {
                let (lock, want) = &*self.wanted;
                let mut w = lock.lock();
                w.trim |= wanted.trim;
                w.settle |= wanted.settle;
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

    /// The RAM tier wants layout `digest` back: its regions are unlocked (one being read when
    /// its copy ends) and its mapping goes with the last fill holding it. Rust only, at once.
    fn release_tier(&self, digest: &str) -> u64 {
        let held: Vec<Arc<tier::Held>> = {
            let mut st = self.state.lock();
            st.sets.iter_mut().flatten().filter(|m| m.tier == digest).filter_map(|m| m.held.take()).collect()
        };
        held.iter().map(|h| h.release(|p| self.backend.unregister_host(p))).sum()
    }

    /// The trim thread's work: the trim hook and shrinks an ask wanted, then a ledger event.
    fn idle_trim(&self, wanted: Wanted) {
        if wanted.settle {
            self.to_custody.settle();
            let mut st = self.state.lock();
            st.table.pending = self.to_custody.pending();
            drop(st);
        }
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

/// This process's open devices: one Memory each.
static OPEN: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());

/// A device this process has open, given back when its Memory goes.
struct Device(String);

impl Device {
    fn open(uuid: &str) -> Result<Device> {
        if !OPEN.lock().insert(uuid.to_string()) {
            return Err(Error::Invalid(format!("{uuid} is already open in this process")));
        }
        Ok(Device(uuid.to_string()))
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        OPEN.lock().remove(&self.0);
    }
}

impl Drop for Memory {
    /// The mechanism's calls finish first and no tier copy is left to run: the row and its
    /// pages go when the drop returns.
    fn drop(&mut self) {
        self.inner.saves.close();
        self.inner.saves.drain();
        self.inner.burst.0.lock().closed = true;
        self.inner.burst.1.notify_all();
        // Under the state lock: a tier copy waiting on the fills sees it closed or is woken.
        drop(self.inner.state.lock());
        self.inner.landed.notify_all();
        self.inner.saves.settle();
        self.inner.mech.settle();
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.to_custody.close();
        self.to_tier.close();
        self.row.knock();
        self.wanted.0.lock().closed = true;
        self.wanted.1.notify_all();
        self.saves.close();
        self.burst.0.lock().closed = true;
        self.burst.1.notify_all();
        // The mechanism's thread is gone with the Memory: what it was still to do happens here.
        self.mech.close();
        for op in self.mech.drain() {
            match op {
                Mech::Unmap { va, off, span, phys, ready } => {
                    let _ = self.backend.unmap(va, off, span);
                    phys.into_iter().for_each(|p| self.backend.destroy(p));
                    self.backend.retire(ready);
                }
                Mech::Renew { va, off, span, old, .. } => {
                    let _ = self.backend.unmap(va, off, span);
                    old.into_iter().for_each(|p| self.backend.destroy(p));
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
            self.backend.destroy(phys);
        }
        let sets: Vec<SetMech> = st.sets.drain(..).flatten().collect();
        drop(st);
        for m in sets {
            self.backend.release(m.va);
            if let Some(held) = &m.held {
                held.unpin(|p| self.backend.unregister_host(p));
            }
        }
    }
}

fn fill_loop(demand: &Receiver<Job>, ahead: &Receiver<Job>, inner: &Weak<Inner>) {
    loop {
        // A demand miss first; both queues close with the Memory.
        let job = select_biased! {
            recv(demand) -> j => j,
            recv(ahead) -> j => j,
        };
        let Ok(job) = job else { return };
        let Some(me) = inner.upgrade() else { return };
        let batch = me.batch(job, demand, ahead);
        // A panic outside the copies fails the batch's binders; the thread keeps serving.
        if let Err(p) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| me.fill(&batch))) {
            let msg = p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned());
            me.fail(&batch, Error::Internal(format!("panic: {}", msg.unwrap_or_default())));
        }
        me.batched(&batch);
    }
}

/// The RAM tier's copies of filled pages, one at a time and only while the fill threads are
/// idle: never ahead of a fill, never on a bind's path.
fn save_loop(saves: &Outbox<Save>, burst: &(Mutex<Burst>, Condvar), inner: &Weak<Inner>) {
    while let Some((ticket, bytes, save)) = saves.next() {
        let mut b = burst.0.lock();
        while !b.closed && !b.idle() {
            burst.1.wait(&mut b);
        }
        let closed = b.closed;
        drop(b);
        if let Some(me) = inner.upgrade().filter(|me| !closed && me.after_fills()) {
            me.save(&save);
        }
        saves.sent(ticket, bytes);
    }
}

fn ask_loop(listener: crate::os::Listener, row: &Weak<Row>, inner: &Weak<Inner>, stop: &AtomicBool) {
    if let Some(row) = row.upgrade() {
        row.ask_thread();
    }
    let inner = inner.clone();
    Row::serve_asks(listener, stop, row.clone(), move |ask| {
        let Some(me) = inner.upgrade() else { return 0 };
        // Never dies: a failed or panicking answer frees nothing, and the next ask is served.
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match ask {
            ledger::Ask::Room { ceiling, bytes } => me.answer(if ceiling >= Class::D.index() { Class::D } else { Class::N }, bytes),
            ledger::Ask::Host { digest } => me.release_tier(&digest),
        }))
        .unwrap_or(0)
    });
}

/// The mechanism thread: the device's VMM calls, in the order decided, off the state lock.
fn mech_loop(queue: &Outbox<Mech>, inner: &Weak<Inner>) {
    while let Some((ticket, bytes, op)) = queue.next() {
        let Some(me) = inner.upgrade() else { return };
        me.mechanism(op);
        let mut st = me.lock();
        st.table.inflight = queue.pending().saturating_sub(bytes);
        me.publish(&st);
        me.unlock(st);
        me.landed.notify_all();
        // The row's facts changed (a page mapped, kept or gone).
        me.shared.ring();
        // Done with the Memory before it reads as done: a settled Memory is this thread's no
        // more.
        drop(me);
        queue.sent(ticket, bytes);
    }
}

/// The sender thread: custody and tier calls, in the order they were posted, until the Memory
/// goes. It holds no lock of the Memory's while it calls; a call that reached a new custody
/// holder takes the state lock alone after, so every kept page is exported to it again.
fn send_loop(outbox: &Outbox<Call>, custody: Option<&custody::Client>, tier: Option<&tier::Client>, inner: &Weak<Inner>) {
    while let Some((ticket, bytes, call)) = outbox.next() {
        let holder = (custody.map(custody::Client::holder), tier.map(tier::Client::holder));
        let _ = match call {
            Call::Export { fp, page, len, handle } => custody.map(|c| c.export(&fp, page, len, handle)),
            Call::Withdraw { fp, page } => custody.map(|c| c.withdraw(&fp, page)),
            Call::Disown { fp, page } => custody.map(|c| c.disown(&fp, page)),
            Call::Use { digest, distance } => tier.map(|t| t.use_layout(&digest, distance)),
            Call::Readopt { digest, regions, ask, file } => tier.map(|t| t.adopt_layout(&digest, regions, &ask, file)),
            Call::Unpin(held) => {
                if let Some(me) = inner.upgrade() {
                    held.unpin(|p| me.backend.unregister_host(p));
                }
                None
            }
            Call::Close { digest } => tier.map(|t| t.close(&digest)),
        };
        outbox.sent(ticket, bytes);
        let moved = holder != (custody.map(custody::Client::holder), tier.map(tier::Client::holder));
        if moved || bytes > 0 {
            if let Some(me) = inner.upgrade() {
                // A new custody holder gets every kept page again, a new tier every held
                // layout (the drain does it); withdrawn bytes are free for the next decision.
                let mut st = me.lock();
                st.table.pending = me.to_custody.pending();
                me.publish(&st);
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
            while !w.trim && !w.settle && w.shrinks.is_empty() {
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

    /// Starts a pass of `set` running package entry `entry` at `shape` (dimensions joined by
    /// `x`, e.g. `1x81x480x832`), with torch's bytes now; returns once the pass's measured room
    /// is free. Rooms are matched per (set, entry), interpolated over shapes within one entry.
    pub fn pass(&self, set: SetId, entry: &str, shape: &str, torch: Torch) -> Result<Pass> {
        self.inner.guard(|| self.inner.open_tier(set.0))?;
        let id = self.inner.guard(|| self.inner.pass(self.id, set.0, entry, shape, torch))?;
        Ok(Pass { inner: self.inner.clone(), req: self.id, set: set.0, id })
    }

    /// An op of the pass over `set` ran out of memory, needing `need` more bytes when the error
    /// said; `site` is its cut site, `None` when the op cannot be cut; `torch` the allocator's
    /// figures now. In a joint step only that set's pass cuts. `set` is `None` outside any pass
    /// (the package's own code between passes): the demand order, then a park, then
    /// exhaustion; nothing is cut or lowered.
    pub fn oom(&self, set: Option<SetId>, need: Option<u64>, site: Option<Site<'_>>, torch: Torch) -> Next {
        match self.inner.guard(|| self.inner.oom(self.id, set.map(|s| s.0), need, site, torch)) {
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

    /// The cut level for op site `site` (0: whole) in the pass over `set`; a collective site's
    /// is the level the ranks agreed. `None`: a site outside any pass, the request's own memo
    /// at the op's `shape` (a pass's own sites go by the pass's shape).
    pub fn cut(&self, set: Option<SetId>, site: &str, shape: &str) -> u32 {
        self.inner
            .guard(|| {
                let st = self.inner.lock();
                let level = match set {
                    Some(set) => st.active.get(&(self.id, set.0)).and_then(|a| st.cuts.get(&(set.0, a.shape.clone()))).and_then(|c| c.get(site)).copied(),
                    None => st.table.request(self.id).and_then(|r| r.outside.get(&(site.to_string(), shape.to_string())).copied()),
                }
                .unwrap_or(0);
                self.inner.unlock(st);
                Ok(level)
            })
            .unwrap_or(0)
    }

    /// The collective sites' proposed levels for the pass over `set` ([`Pass::levels`]).
    pub fn levels(&self, set: SetId) -> Vec<(String, u32)> {
        let pass = self.inner.state.lock().active.get(&(self.id, set.0)).map(|a| a.id);
        pass.map(|id| self.inner.levels(self.id, id)).unwrap_or_default()
    }

    /// The levels the ranks agreed, for the pass over `set` ([`Pass::agree`]).
    pub fn agree(&self, set: SetId, levels: Vec<(String, u32)>) {
        let pass = self.inner.state.lock().active.get(&(self.id, set.0)).map(|a| a.id);
        if let Some(id) = pass {
            self.inner.agree(self.id, id, levels);
        }
    }

    /// Whether the current pass's room is the one measured for its set, shape and cut levels
    /// (false when it is estimated, or no pass is open).
    pub fn measured(&self) -> bool {
        self.inner.state.lock().current(self.id).is_some_and(|a| a.measured)
    }

    /// The request waits on a child call of its family `family`: until the guard drops it gives
    /// back its reserved room and counts only what is allocated; its remaining working set is
    /// published apart, for every request's admission but that family's child; its step's
    /// pages may be asked for (a later use). Dropped, it takes its room back as a request first
    /// gets it (a decision, then waits on the table's changes).
    pub fn yielding(&self, family: &str) -> Yielded {
        let _ = self.inner.guard(|| {
            let mut st = self.inner.lock();
            if let Some(r) = st.table.request_mut(self.id).filter(|r| r.yielded.is_none()) {
                r.yielded = Some(r.room + r.claimed);
                (r.room, r.claimed) = (0, 0);
                r.family = ledger::family_id(family);
            }
            self.inner.publish(&st);
            self.inner.unlock(st);
            self.inner.shared.ring();
            Ok(())
        });
        Yielded { wait: Some((self.inner.clone(), self.id)) }
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
            if matches!(st.mech(self.key.0).map(|m| &m.spec.fingerprint.fill), Ok(Fill::Declared(_))) {
                let req = st.binding(self.key.0);
                self.inner.save_later(&st, self.key, req);
            }
            self.inner.publish(&st);
            self.inner.unlock(st);
            self.inner.landed.notify_all();
            Ok(())
        });
    }
}

impl Drop for Pin {
    fn drop(&mut self) {
        // Without an event the release waits on the host for this one pin's stream (rare).
        let event = self.inner.backend.record(self.stream).unwrap_or_else(|e| {
            tracing::warn!("varena: recording a release failed ({e}); synchronizing its stream");
            if let Err(e) = self.inner.backend.synchronize(self.stream) {
                tracing::warn!("varena: synchronizing a released pin's stream failed: {e}");
            }
            Event::DONE
        });
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

    /// Whether this pass's room is the one measured for its set, shape and cut levels (false
    /// when it is estimated or the pass has ended).
    pub fn measured(&self) -> bool {
        self.inner.state.lock().pass_of(self.req, self.id).is_some_and(|a| a.measured)
    }

    /// The collective sites' proposed levels for this pass: the level in effect, a finer one an
    /// OOM proposed, or a coarser one this rank's surplus allows. cozy-runtime all-reduces
    /// their maximum across the ranks and hands it to [`Pass::agree`] before the pass runs its
    /// collectives: a level relaxes only when every rank proposes it.
    pub fn levels(&self) -> Vec<(String, u32)> {
        self.inner.levels(self.req, self.id)
    }

    /// The levels the ranks agreed (the maximum of their proposals): in effect for this pass.
    pub fn agree(&self, levels: Vec<(String, u32)>) {
        self.inner.agree(self.req, self.id, levels);
    }
}

impl Inner {
    /// Runs one mechanism call (the state lock free), then applies its result under it.
    fn mechanism(&self, op: Mech) {
        match op {
            Mech::Map { key, gen, va, off, span, job, speculative, decided } => self.place_page(key, gen, va, off, span, job, (!speculative).then_some(decided)),
            Mech::Renew { key, gen, va, off, span, old, after, job } => {
                // Its consumers finish before the unmap: this thread waits for their events.
                for e in &after {
                    if let Err(e) = self.backend.finish(*e) {
                        tracing::warn!("varena: waiting for a renewed page's consumers failed: {e}");
                    }
                }
                let _ = self.backend.unmap(va, off, span);
                if let Some(old) = old {
                    self.backend.destroy(old);
                }
                after.into_iter().for_each(|e| self.backend.retire(e));
                self.place_page(key, gen, va, off, span, job, Some(u64::MAX));
            }
            Mech::Adopt { key, gen, va, off, phys, len, identity } => {
                let mapped = self.backend.map(va, off, &Physical { id: phys, len }, Access::Read);
                let mut st = self.state.lock();
                let current = st.homes.get(&key).is_some_and(|h| h.gen == gen);
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
                            let _ = self.backend.unmap(va, off, len);
                        }
                        if let Some(Some(p)) = home.map(|h| h.phys) {
                            self.backend.destroy(p);
                        }
                        self.post(Call::Disown { fp: identity, page: key.1 });
                    }
                }
            }
            Mech::Unmap { va, off, span, phys, ready } => {
                if let Err(e) = self.backend.unmap(va, off, span) {
                    tracing::warn!("varena: unmapping failed: {e}");
                }
                if let Some(p) = phys {
                    self.backend.destroy(p);
                }
                self.backend.retire(ready);
            }
            Mech::Keep { key, gen, va, off, span, phys, len } => {
                // A Keep for an earlier life of the page is dropped before it protects: that
                // page was renewed (a Keep queued forces a renewal), never refilled in place.
                if !self.state.lock().homes.get(&key).is_some_and(|h| h.gen == gen) {
                    return;
                }
                let made = self.backend.protect(va, off, span, Access::Read).and_then(|_| self.backend.export(&Physical { id: phys, len }));
                let mut st = self.state.lock();
                let Some(home) = st.homes.get_mut(&key).filter(|h| h.gen == gen) else { return };
                home.keeping = false;
                match made {
                    Ok(handle) => {
                        home.exported = true;
                        let fp = home.identity.clone();
                        self.post(Call::Export { fp, page: key.1, len: span, handle });
                        st.sync(key);
                    }
                    Err(e) => tracing::warn!("varena: keeping {key:?} failed: {e}"),
                }
            }
            Mech::Release(va) => self.backend.release(va),
        }
    }

    /// Creates and maps a page for `key` and queues its fill; a create the device cannot
    /// serve leaves the page unmapped, with the free bytes it was decided on (`decided`;
    /// `None`: a speculative page, which goes quietly).
    #[allow(clippy::too_many_arguments)]
    fn place_page(&self, key: Key, gen: u64, va: Va, off: u64, span: u64, job: Job, decided: Option<u64>) {
        let speculative = decided.is_none();
        let made = self.backend.create(span).and_then(|p| match self.backend.map(va, off, &p, Access::ReadWrite) {
            Ok(()) => Ok(p),
            Err(e) => {
                self.backend.destroy(p);
                Err(e)
            }
        });
        let mut st = self.state.lock();
        if !st.homes.get(&key).is_some_and(|h| h.gen == gen) {
            drop(st);
            if let Ok(p) = made {
                let _ = self.backend.unmap(va, off, span);
                self.backend.destroy(p);
            }
            return;
        }
        match made {
            Ok(p) => {
                if let Some(h) = st.homes.get_mut(&key) {
                    h.phys = Some(p);
                }
                drop(st);
                if self.queue_fill(job, speculative).is_err() {
                    tracing::warn!("varena: the fill threads are gone");
                }
            }
            Err(e) => {
                st.homes.remove(&key);
                st.sync(key);
                if !speculative {
                    match e {
                        // The device had less than the table showed: the bind measures and
                        // decides again.
                        Error::Device { code: crate::backend::OUT_OF_MEMORY, .. } => {
                            st.short.insert(key, (span, decided.unwrap_or(0)));
                        }
                        e => drop(st.failed.insert(key, e)),
                    }
                }
            }
        }
    }

    /// The wait on a child ended: the request's room comes back, made as a pass start makes
    /// it (a shortfall is the next op's OOM).
    fn unyield(&self, req: u64) -> Result<()> {
        let back = {
            let mut st = self.lock();
            let set = st.current(req).map(|a| a.set);
            let back = st.table.request_mut(req).and_then(|r| r.yielded.take().map(|room| (room, r.measured)));
            self.publish(&st);
            self.unlock(st);
            back.zip(set)
        };
        let Some(((room, measured), set)) = back else { return Ok(()) };
        // Its room back as a request first gets it: a decision, then waits on the table's
        // changes. Exhausted only if the card cannot hold it at all.
        self.row.set_waiting(true);
        let out = loop {
            let actions = match self.demand(Trigger::Pass { req, set, room: measured.then_some(room), unmeasured: room, tried: Tried::default() }, Some(req)) {
                Ok(a) => a,
                Err(e) => break Err(e),
            };
            let Some(Action::Exhausted { need }) = actions.last() else { break Ok(()) };
            let st = self.lock();
            let never = room > st.table.capacity;
            let stats = st.stats.get(&req).cloned().unwrap_or_default();
            self.unlock(st);
            if never {
                break Err(Error::Exhausted(exhausted(*need, None, stats)));
            }
            if let Err(e) = crate::os::wake_await(&self.room_wake) {
                break Err(e);
            }
        };
        self.row.set_waiting(false);
        out
    }

    fn levels(&self, req: u64, pass: u64) -> Vec<(String, u32)> {
        self.guard(|| {
            let st = self.lock();
            let out = match st.pass_of(req, pass) {
                Some(a) => {
                    let now = st.levels(a.set, &a.shape);
                    let proposed = st.proposed.get(&(a.set, a.shape.clone())).cloned().unwrap_or_default();
                    let mut sites: Vec<String> = st.collective.iter().filter(|(s, sh, _)| *s == a.set && *sh == a.shape).map(|(.., n)| n.clone()).collect();
                    sites.sort();
                    sites.into_iter().map(|n| (n.clone(), proposed.get(&n).or(now.get(&n)).copied().unwrap_or(0))).collect()
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
            if let Some((set, shape)) = st.pass_of(req, pass).map(|a| (a.set, a.shape.clone())) {
                for (site, level) in levels {
                    st.collective.insert((set, shape.clone(), site.clone()));
                    st.cuts.entry((set, shape.clone())).or_default().insert(site.clone(), level);
                    if let Some(p) = st.proposed.get_mut(&(set, shape.clone())) {
                        p.remove(&site);
                    }
                }
            }
            self.unlock(st);
            Ok(())
        });
    }
}

/// A request waiting on a child call ([`Request::yielding`]); dropped, the wait is over.
pub struct Yielded {
    wait: Option<(Arc<Inner>, u64)>,
}

impl Yielded {
    /// The wait is over: the room is taken back, waiting for it as a request first gets it;
    /// `Exhausted` only when the card cannot hold it. Dropping it does the same, quietly.
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

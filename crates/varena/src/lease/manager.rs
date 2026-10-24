//! The lease manager (varena-leases §1, §2, §2a, §11, §12): it creates every physical chunk,
//! allocates chunks to executors' agents, reclaims for room, and takes an exited executor's chunks
//! back once its pidfd shows the exit: landed weights go to the GPU page cache for the next
//! executor of that trust domain and region (no copy), the rest are destroyed. Only the manager
//! exports. What to allocate, reclaim and refuse is [`policy::decide`]'s: the manager builds its
//! snapshot from the page accounting and applies the answer.
//!
//! The room is the device's own figure (§12, 18:40Z): the driver's free bytes as its host reads
//! them with no context of ours (NVML; Windows: the DXGI budget less its usage), read outside the
//! lock before every decision, less what was created since that read, less the reserved pool. Every context, module, library buffer and foreign byte is simply used: the
//! page accounting never rests on an agent's word.
//!
//! One manager per GPU, TensorD's, with the GPU's listener; TensorD reacts to [`Manager::events`].
//! It also holds what admission and placement read (TensorD's dispatches and the measured peaks)
//! and the allocation-wait spans PSI reads.
//!
//! The table is untrusted input: identity, pid, a chunk's index and whether it landed are the
//! manager's own records (peer credentials, grant time, fill-ack time), never read back at exit.
//!
//! Locks: `books` is the only one, and nothing blocks under it (sends and records are made after
//! it drops).

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};
use platform::contain::host_memory;
use platform::handles::Peer;
use platform::memory::{Signal, Waitable};
use platform::process::Exact;

use super::policy::{self, Answer, Pick};
use super::table::{self, Content, Figures, State, Table};
use super::wire::{End, Kind, Measured, Refusal, Region, RegionFacts, ToAgent, ToManager};
use crate::backend::{Backend, Physical};
use crate::decide;
use crate::os::{self, Listener, Owned, Stream};
use crate::{Error, GpuId, Result};

/// The largest chunk: one handle each (§3).
pub const CHUNK: u64 = 32 << 20;
macro_rules! protocol {
    () => {
        2
    };
}
/// The lease protocol an agent speaks in its Hello (1 is the ledger's).
pub const PROTOCOL: u32 = protocol!();
/// [`PROTOCOL`] as TensorFS's extension carries it, for TensorD to read from a wheel's zip
/// without importing it.
pub const MARKER: &str = concat!("cozy-lease-protocol=", protocol!(), ";");
/// The first lease protocol: the manager never serves the ledger's.
const FIRST: u32 = 2;

/// Whether this manager serves an agent speaking `protocol`: its own and the one before (§2).
pub fn accepts(protocol: u32) -> bool {
    protocol == PROTOCOL || (protocol >= FIRST && protocol + 1 == PROTOCOL)
}
/// The stall look's cadence (TensorD's sampler's): a period, never a verdict.
const LOOK: Duration = Duration::from_millis(500);
/// Looks in a row with neither a release nor any of its GPU work completing, while an unpinned
/// ask of it waits, before an executor is stalled.
const LOOKS: u32 = 3;

/// Who a connecting peer is, as its host knows it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    /// Its executor group (a multi-GPU executor's followers share it).
    pub executor: String,
    /// The trust domain retained weights are adopted within.
    pub domain: String,
}

/// What the manager did, for its host's memory log.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Record {
    Allocate { pid: u32, ask: u64, chunks: usize, bytes: u64, cached: bool, create_us: u64, export_us: u64, total_us: u64 },
    /// An allocation served from the GPU page cache (no copy).
    Cached { pid: u32, region: Region, chunks: usize, bytes: u64 },
    Refuse { pid: u32, ask: u64, code: Refusal, why: String },
    /// Reclaim requests sent (cooperative: the agent's CAS decides).
    Reclaim { pid: u32, slots: Vec<u32> },
    Released { pid: u32, released: usize, us: u64 },
    /// An executor's exit: its chunks destroyed, or kept in the page cache.
    Exit { pid: u32, freed_bytes: u64, cached_bytes: u64, us: u64 },
    /// Judged stalled: `looks` looks with no release, no GPU progress and no GPU work running,
    /// its oldest unpinned ask `waited_ms` old.
    Stalled { pid: u32, looks: u32, waited_ms: u64 },
    /// An ask back held by a pin (`slot`, `pins` of them) for `looks` looks: reported, never
    /// ended.
    Pinned { pid: u32, slot: u32, pins: u32, looks: u32, waited_ms: u64 },
    /// An ask left waiting: what it needs, what counts as coming, the pids whose pins it waits
    /// on (each running a request), and its own chunks Revoking and asked back. Logged when any
    /// of these changes.
    Wait { pid: u32, ask: u64, bytes: u64, coming: u64, on: Vec<u32>, own_revoking: u64, own_asked: u64 },
    /// An ask short of room that idle executors (`idle`, no request running) pin: the host is
    /// asked to end them, and the ask waits for their exit.
    EndIdle { pid: u32, ask: u64, idle: Vec<u32> },
}

/// What TensorD answers for the manager.
pub trait Hosting: Send + Sync {
    /// The identity of a peer on the GPU's listener (any uid: the host judges it); `None`
    /// refuses it.
    fn identify(&self, peer: &Peer) -> Option<Identity>;

    /// A record of what the manager did, after its books dropped.
    fn record(&self, _record: Record) {}

    /// Whether the host ends idle executors on [`Hosting::end_idle`]: then an ask short of room
    /// that idle executors pin waits for their exit instead of being refused.
    fn ends_idle(&self) -> bool {
        false
    }

    /// End these executors (by pid): idle (no request running), their pins hold room a waiting
    /// ask needs. After the books dropped.
    fn end_idle(&self, _pids: &[u32]) {}

    /// The derived-weights disk cache (§6): a new unlinked entry of `domain` for `bytes`, its
    /// own, for the agent that derived it to write. None: none kept (low disk, unhosted).
    fn derive(&self, _domain: &str, _key: &str, _bytes: u64) -> Option<super::DerivedEntry> {
        None
    }

    /// Entry `entry` of `key` was written whole by its agent: from now on it is read. Another
    /// id (a stale or unknown one) marks nothing.
    fn derived(&self, _domain: &str, _key: &str, _entry: u64) {}

    /// A written entry a grant refills from instead of deriving again.
    fn read_derived(&self, _domain: &str, _key: &str) -> Option<std::fs::File> {
        None
    }

    /// The page cache dropped the chunk it filled (after the manager's books dropped): the entry
    /// is now its only copy.
    fn forget_derived(&self, _domain: &str, _key: &str) {}

    /// The device's free bytes now as the driver counts them, needing no CUDA context of
    /// TensorD's (NVML's memory info; Windows: the DXGI budget less its usage). `None`: unread
    /// here, and the manager falls back to its open backend's (cuMemGetInfo), if it has one.
    fn free(&self) -> Option<u64> {
        None
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct ExecutorHolding {
    pub pid: u32,
    pub executor: String,
    pub chunks: usize,
    pub bytes: u64,
    /// Of `bytes`, what its fills acked landed: what its exit would keep in the page cache.
    pub landed: u64,
}

/// PSI `some` (§12): the share of wall time in which at least one allocation waited.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub struct Pressure {
    pub some_10s: f32,
    pub some_60s: f32,
}

/// What a manager holds, for its host's status and the tests: free = capacity − created.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct Holdings {
    pub capacity: u64,
    /// TensorD's own context while its backend is open, else 0.
    pub context: u64,
    pub free: u64,
    pub foreign: u64,
    pub created: u64,
    pub created_chunks: usize,
    pub allocated: u64,
    pub page_cache: u64,
    pub page_cache_chunks: usize,
    /// Chunks a previous manager created that agents still map (0 until re-attach exists).
    pub books_only: u64,
    pub executors: Vec<ExecutorHolding>,
    /// Request id → `<package>|<shape>`.
    pub dispatched: Vec<(String, String)>,
    pub pressure: Pressure,
}

/// The machine's sharing mode, as TensorD sets it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub enum Mode {
    /// Someone else uses the machine too.
    #[default]
    Shared,
    Dedicated,
}

/// A measured peak, as TensorD persists it per GPU model: restored exactly as it was read.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Peak {
    /// A request's largest working set at `<package>|<shape>`.
    Request { key: String, bytes: u64 },
    /// A request's largest allocations beside its weights at `<package>|<shape>`: what placement
    /// keeps free on GPU 0.
    Compute { key: String, bytes: u64 },
    /// A set's resident bytes, by set name.
    Set { key: String, bytes: u64 },
    /// A request's anonymous host high-water at `<package>|<shape>`.
    Host { key: String, bytes: u64 },
}

/// The driver's free bytes, and the bytes created and destroyed before that read; the host's
/// memory beside it (what a swap to RAM may take).
#[derive(Clone, Copy, Debug)]
struct Sample {
    free: u64,
    created: u64,
    destroyed: u64,
    host: policy::Host,
}

/// Opens the GPU's backend (TensorD's CUDA context), at the first ask.
pub type Opener = Box<dyn Fn() -> Result<Arc<dyn Backend>> + Send + Sync>;

/// The backend while chunks or executors need it, with what it measured at open.
struct Device {
    backend: Arc<dyn Backend>,
    /// min(limit, card).
    card: u64,
    granularity: u64,
    /// The reserved pool H, never allocated in any mode (cr-215).
    headroom: u64,
    /// TensorD's own context, measured at open (the books check's: NVML charges it to us).
    context: u64,
}

/// A table's slots: one per smallest chunk the card could hold.
const SLOT_UNIT: u64 = 2 << 20;

pub struct Manager {
    /// None: a GPU with no VMM for us (exclusive compute mode): every agent runs standalone.
    open: Option<Opener>,
    /// One open at a time (never under `books`).
    opening: Mutex<()>,
    gpu_memory: u64,
    limit: Option<u64>,
    hosting: Arc<dyn Hosting>,
    retired: AtomicBool,
    books: Mutex<Books>,
    /// Wakes the stall look when asks go out.
    looking: Condvar,
    /// Set on every change TensorD reacts to.
    events: Signal,
    /// PSI's clock.
    born: Instant,
}

#[derive(Default)]
struct Books {
    /// Opened at the first ask, dropped once no chunk and no executor is left (§2a: TensorD's
    /// context keeps a laptop's GPU awake and costs VRAM).
    device: Option<Arc<Device>>,
    created: u64,
    created_chunks: usize,
    seq: u32,
    next: u64,
    executors: BTreeMap<u64, Exec>,
    /// Landed weights of exited executors, by trust domain and region: a region's chunks by
    /// index, oldest first in `order`.
    retained: BTreeMap<Kept, BTreeMap<u32, Physical>>,
    order: VecDeque<Kept>,
    /// A retained region's fill key: its derived entry, forgotten when the region goes.
    fills: BTreeMap<Kept, String>,
    /// Every byte ever created and destroyed: what changed since a sample is these less its own.
    created_total: u64,
    destroyed_total: u64,
    /// min(limit, card), and the reserved pool: the backend's once it opened.
    card: u64,
    reserve: u64,
    /// The driver's last reading; `None`: unread, and nothing grows.
    sample: Option<Sample>,
    /// Device bytes no process of ours holds, as TensorD last sampled them (shown only), and
    /// whether that reading saw a new foreign process (consumed by the next decision).
    foreign: u64,
    new_foreign: bool,
    mode: Mode,
    /// Measured figures by `<kind>/<key>`: peak, compute and host by `<package>|<shape>`, set by
    /// name.
    figures: BTreeMap<String, u64>,
    /// Dispatched request → `<package>|<shape>`.
    dispatched: BTreeMap<String, String>,
    /// PSI: spans (ns on `born`) in which an allocation waited, the last minute's, and the
    /// start of the one open now.
    waits: VecDeque<(u64, u64)>,
    waiting: Option<u64>,
}

type Kept = (String, Region);

struct Exec {
    /// From the peer's credentials, never the table.
    pid: u32,
    identity: Identity,
    table: Arc<Table>,
    end: Arc<End>,
    free: BTreeSet<u32>,
    leased: BTreeMap<u32, Leased>,
    waiting: VecDeque<Waiting>,
    /// Slots asked back and not yet released, and when each was asked.
    asked: BTreeMap<u32, Instant>,
    /// Releases the manager reclaimed itself, and with the agent's GPU progress, what a stall
    /// is judged by.
    progress: u64,
    /// The stall look's: (releases, GPU progress) at the last look that saw either move, and
    /// looks since.
    seen: (u64, u64),
    still: u32,
    stalled: bool,
    /// Asked slots already reported held by a pin.
    reported: BTreeSet<u32>,
    /// The executors whose pins its waiting ask waits on.
    waits_on: BTreeSet<u64>,
    /// Its regions and graphs, as its agent last reported them, and those asked back whole.
    regions: BTreeMap<u64, RegionFacts>,
    reclaiming: BTreeSet<u64>,
    /// Its waiting ask's last Wait record: (ask, coming, on), so an unchanged wait logs once.
    logged: Option<(u64, u64, Vec<u64>)>,
    /// It failed or was poisoned ([`Manager::fault`]): none of its chunks counts or is asked
    /// for, and its exit destroys them all (none is kept for another executor to adopt).
    faulted: bool,
    /// When the host was asked to end it (idle, its pins holding a waiting ask's room). An ask
    /// waits for its exit only the stall look's looks, then is refused.
    ending: Option<Instant>,
}

struct Leased {
    phys: Physical,
    seq: u32,
    region: Option<Region>,
    index: u32,
    /// The agent acked its fill (recorded at that ack, never read at exit).
    landed: bool,
    /// Its ask's fill key.
    fill: Option<String>,
}

struct Waiting {
    id: u64,
    bytes: u64,
    kind: Kind,
    region: Option<Region>,
    fill: Option<String>,
    /// The derived entry kept for `fill`, read before the books were taken.
    entry: Option<std::fs::File>,
}

/// What to send and record once `books` dropped, and a backend nothing needs any more.
#[derive(Default)]
struct Out {
    sends: Vec<(Arc<End>, ToAgent, Option<Owned>)>,
    /// Of the sends, the grants: (executor, slot), to roll back if their send fails.
    grants: Vec<(usize, u64, u32)>,
    records: Vec<Record>,
    /// Derived entries whose retained region went: (domain, fill key).
    forgets: Vec<(String, String)>,
    /// Idle executors the host is asked to end (pids).
    end_idle: Vec<u32>,
    idle: Option<Arc<Device>>,
}

impl Out {
    fn append(&mut self, mut other: Out) {
        let at = self.sends.len();
        self.grants.extend(other.grants.drain(..).map(|(i, e, s)| (i + at, e, s)));
        self.sends.append(&mut other.sends);
        self.records.append(&mut other.records);
        self.forgets.append(&mut other.forgets);
        self.end_idle.append(&mut other.end_idle);
        self.idle = self.idle.take().or(other.idle);
    }
}

/// A region's chunk sizes: `CHUNK`s, the tail rounded up to the device's `granularity`.
pub fn chunks(bytes: u64, granularity: u64) -> Vec<u64> {
    let unit = granularity.max(1);
    let mut out = vec![];
    let mut left = bytes.div_ceil(unit) * unit;
    while left > 0 {
        let take = left.min(CHUNK);
        out.push(take);
        left -= take;
    }
    out
}

/// An ask's chunks: a weights page is one chunk sized to it (§3, never split, declared or not);
/// torch and cache bytes split at [`CHUNK`].
fn sizes(ask: &Waiting, granularity: u64) -> Vec<u64> {
    match ask.kind {
        Kind::Weights => {
            let unit = granularity.max(1);
            vec![ask.bytes.div_ceil(unit) * unit]
        }
        Kind::Torch | Kind::Cache => chunks(ask.bytes, granularity),
    }
}

/// A cache key's chunks when they serve an ask whole: every chunk, by index, the size it cuts.
fn cached(b: &Books, domain: &str, region: Option<Region>, sizes: &[u64]) -> Option<Kept> {
    let key = (domain.to_string(), region?);
    let kept = b.retained.get(&key)?;
    let whole = kept.len() == sizes.len() && kept.iter().zip(sizes).enumerate().all(|(i, ((index, p), len))| *index as usize == i && p.len == *len);
    whole.then_some(key)
}

/// The device's free bytes as the books stand (the policy's `free`): the driver's last reading less
/// what was created since and plus what was destroyed since, within the card or limit; `None`
/// while unread.
fn free(b: &Books) -> Option<u64> {
    let sample = b.sample?;
    let driver = (sample.free + (b.destroyed_total - sample.destroyed)).saturating_sub(b.created_total - sample.created);
    Some(driver.min(b.card.saturating_sub(b.created) + b.reserve))
}

/// What may be created now: free less the reserved pool.
fn room(b: &Books) -> u64 {
    free(b).map_or(0, |f| f.saturating_sub(b.reserve))
}

/// A figure at `<kind>/<key>`; a key ending in `|` (no shape) means the widest measured.
fn figure(b: &Books, kind: &str, key: &str) -> Option<u64> {
    let full = format!("{kind}/{key}");
    match key.ends_with('|') {
        true => b.figures.range(full.clone()..).take_while(|(k, _)| k.starts_with(&full)).map(|(_, v)| *v).max(),
        false => b.figures.get(&full).copied(),
    }
}

fn larger(figures: &mut BTreeMap<String, u64>, key: String, bytes: u64) {
    let v = figures.entry(key).or_insert(bytes);
    *v = (*v).max(bytes);
}

fn micros(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_micros()).unwrap_or(u64::MAX)
}

impl Manager {
    /// The GPU's manager, `listener` served on the manager's own thread until
    /// [`Manager::retire`]. Its backend opens at the first ask (`open`; `None`: no VMM for us,
    /// every agent runs standalone); it then creates only into the driver's free bytes less the
    /// reserved pool, within `limit` (a `--gpu-memory-limit`, else the card).
    pub fn create(gpu: &GpuId, limit: Option<u64>, mode: Mode, open: Option<Opener>, listener: Listener, hosting: Arc<dyn Hosting>) -> Result<Arc<Manager>> {
        let events = Signal::new().map_err(|e| Error::Io(e.to_string()))?;
        let card = limit.unwrap_or(gpu.memory).min(gpu.memory);
        let books = Mutex::new(Books { mode, card, reserve: crate::backend::RESERVE, ..Books::default() });
        let manager = Arc::new(Manager { open, opening: Mutex::new(()), gpu_memory: gpu.memory, limit, hosting, retired: AtomicBool::new(false), books, looking: Condvar::new(), events, born: Instant::now() });
        let weak = Arc::downgrade(&manager);
        std::thread::Builder::new().name("lease-look".into()).spawn(move || look_loop(&weak)).map_err(|e| Error::io("spawn lease-look", e))?;
        let serving = manager.clone();
        std::thread::Builder::new()
            .name("lease-listener".into())
            .spawn(move || {
                loop {
                    let stream = platform::memory::accept(&listener);
                    if serving.retired.load(Relaxed) {
                        break;
                    }
                    match stream {
                        Ok(stream) => serving.serve(stream),
                        Err(error) => tracing::warn!("lease: accepting an agent: {error}"),
                    }
                }
            })
            .map_err(|e| Error::io("spawn lease-listener", e))?;
        Ok(manager)
    }

    /// Readable on every allocation, release, exit and figure; [`Manager::clear_events`] clears
    /// it.
    pub fn events(&self) -> Waitable<'_> {
        self.events.waitable()
    }

    pub fn clear_events(&self) -> bool {
        self.events.clear()
    }

    /// Executors the stall look judged stalled: an unpinned chunk asked back, and neither a
    /// release nor any GPU progress of it across [`LOOKS`] looks. A getter: any caller, any rate.
    pub fn stalled(&self) -> Vec<u32> {
        let mut pids: Vec<u32> = self.books.lock().executors.values().filter(|e| e.stalled).map(|e| e.pid).collect();
        pids.sort_unstable();
        pids.dedup();
        pids
    }

    /// One stall look (every [`LOOK`] while asks are out): whether any ask is out, the
    /// executors newly judged stalled, and whether a waiting ask must be decided again (its own
    /// chunks went overdue, or an idle holder it waits on outlived the looks).
    fn look(&self) -> (bool, Vec<Record>, bool) {
        let mut b = self.books.lock();
        let (mut out, mut records, mut again) = (false, vec![], false);
        for e in b.executors.values_mut() {
            out |= !e.asked.is_empty();
            again |= !e.waiting.is_empty() && e.asked.values().any(|at| at.elapsed() >= LOOK * LOOKS);
            again |= e.ending.is_some_and(|at| at.elapsed() >= LOOK * LOOKS);
            // An ask a pin holds is reported once it waited the looks, never judged.
            for (slot, at) in &e.asked {
                let held = e.table.word(*slot).filter(|w| w.state == State::Mapped && w.pins > 0);
                if let Some(w) = held.filter(|_| at.elapsed() >= LOOK * LOOKS && !e.reported.contains(slot)) {
                    e.reported.insert(*slot);
                    records.push(Record::Pinned { pid: e.pid, slot: *slot, pins: w.pins, looks: LOOKS, waited_ms: u64::try_from(at.elapsed().as_millis()).unwrap_or(u64::MAX) });
                }
            }
            e.reported.retain(|s| e.asked.contains_key(s));
            let now = (e.progress, e.table.progress());
            let oldest = e.asked.iter().filter(|(s, _)| e.table.word(**s).is_some_and(|w| w.state == State::Mapped && w.pins == 0)).map(|(_, at)| *at).min();
            // Busy (GPU work running) or moving: never a stall.
            let Some(oldest) = oldest.filter(|_| now == e.seen && !e.table.busy()) else {
                (e.seen, e.still, e.stalled) = (now, 0, false);
                continue;
            };
            e.still += 1;
            if e.still >= LOOKS && !e.stalled {
                e.stalled = true;
                records.push(Record::Stalled { pid: e.pid, looks: e.still, waited_ms: u64::try_from(oldest.elapsed().as_millis()).unwrap_or(u64::MAX) });
            }
        }
        let waiting = b.executors.values().any(|e| !e.waiting.is_empty());
        (out, records, again && waiting)
    }

    /// A decision with no event behind it (the look found a waiting ask's own chunks overdue,
    /// or an idle holder past the looks).
    fn decide_again(&self) {
        let sample = self.read();
        let out = {
            let mut b = self.books.lock();
            b.sample = sample;
            self.settle(&mut b)
        };
        self.emit(out);
    }

    pub fn holdings(&self) -> Holdings {
        let b = self.books.lock();
        let executors: Vec<ExecutorHolding> = b
            .executors
            .values()
            .map(|e| ExecutorHolding {
                pid: e.pid,
                executor: e.identity.executor.clone(),
                chunks: e.leased.len(),
                bytes: e.leased.values().map(|l| l.phys.len).sum(),
                landed: e.leased.values().filter(|l| l.landed && l.region.is_some()).map(|l| l.phys.len).sum(),
            })
            .collect();
        let (now, spans) = (self.now(), b.waits.iter().copied().collect::<Vec<_>>());
        let pressure = Pressure { some_10s: policy::pressure(&spans, b.waiting, now, 10_000_000_000), some_60s: policy::pressure(&spans, b.waiting, now, 60_000_000_000) };
        let free = room(&b);
        Holdings {
            capacity: b.created + free,
            context: b.device.as_ref().map_or(0, |d| d.context),
            free,
            foreign: b.foreign,
            created: b.created,
            created_chunks: b.created_chunks,
            allocated: executors.iter().map(|e| e.bytes).sum(),
            page_cache: b.retained.values().flat_map(|v| v.values()).map(|p| p.len).sum(),
            page_cache_chunks: b.retained.values().map(BTreeMap::len).sum(),
            books_only: 0,
            executors,
            dispatched: b.dispatched.iter().map(|(r, k)| (r.clone(), k.clone())).collect(),
            pressure,
        }
    }

    pub fn set_mode(&self, mode: Mode) -> Result<()> {
        self.books.lock().mode = mode;
        Ok(())
    }

    fn now(&self) -> u64 {
        u64::try_from(self.born.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    /// Stops accepting agents; connected ones keep their leases until they exit.
    pub fn retire(&self) {
        self.retired.store(true, Relaxed);
    }

    /// One executor's connection, on its own thread until the executor has exited and its
    /// leases are back.
    fn serve(self: &Arc<Self>, stream: Stream) {
        let manager = self.clone();
        let spawned = std::thread::Builder::new().name("lease-executor".into()).spawn(move || {
            if let Err(error) = manager.executor(stream) {
                tracing::warn!("lease: an executor's connection ended: {error}");
            }
        });
        if let Err(error) = spawned {
            tracing::warn!("lease: serving an agent: {error}");
        }
    }

    fn executor(self: &Arc<Self>, stream: Stream) -> Result<()> {
        let peer = platform::handles::peer(&stream).map_err(|e| Error::io("lease peer", e))?;
        let end = Arc::new(End::new(stream));
        // Every answer, a refusal too, answers a Hello: never one the agent cannot read.
        let Some((ToManager::Hello { protocol, birth }, again)) = end.recv::<ToManager>()? else {
            return Err(Error::Invalid("a lease connection opened without Hello".into()));
        };
        let Some(identity) = self.hosting.identify(&peer) else {
            return end.send(&ToAgent::Refuse { id: 0, code: Refusal::NotRegistered, why: format!("pid {} is no executor of this machine", peer.pid) }, None);
        };
        if self.open.is_none() {
            return end.send(&ToAgent::Refuse { id: 0, code: Refusal::Incompatible, why: "this GPU has no VMM for leases (exclusive compute mode)".into() }, None);
        }
        if !accepts(protocol) || again.is_some() {
            // A re-attach after a manager restart comes after slice 1: that agent runs standalone.
            let why = if again.is_some() { "re-attach is not served yet".into() } else { format!("lease protocol {protocol}; this manager speaks {PROTOCOL}") };
            return end.send(&ToAgent::Refuse { id: 0, code: Refusal::Incompatible, why }, None);
        }
        // The birth of the process handles will be duplicated into (Windows: the one the pipe
        // keeps open), so a pid reused after the stream was handed on never passes.
        let found = platform::handles::peer_birth(end.stream()).map_err(|e| Error::Io(e.to_string()))?;
        if found.start_ticks != birth {
            return Err(Error::Invalid(format!("lease Hello for birth {birth} from pid {} (birth {})", peer.pid, found.start_ticks)));
        }
        // A peer no handle can reach (Windows: of a higher integrity level) runs standalone.
        if let Err(error) = platform::handles::accepts_handles(end.stream()) {
            return end.send(&ToAgent::Refuse { id: 0, code: Refusal::Incompatible, why: error.to_string() }, None);
        }
        let process = Exact::open(&found).map_err(|e| Error::Io(e.to_string()))?.ok_or_else(|| Error::Closed(format!("process {}", peer.pid)))?;
        let slots = u32::try_from(self.limit.unwrap_or(self.gpu_memory) / SLOT_UNIT).unwrap_or(u32::MAX).max(1);
        // Sealed against resizing before the agent sees it.
        let file = os::sealed_page(table::bytes(slots) as u64)?;
        let table = Arc::new(Table::create(os::map(&file, table::bytes(slots), true)?, slots, peer.pid, birth)?);
        let epoch = table.attach();
        let id = {
            let mut b = self.books.lock();
            b.next += 1;
            let id = b.next;
            let exec = Exec { pid: peer.pid, identity, table, end: end.clone(), free: (0..slots).collect(), leased: BTreeMap::new(), waiting: VecDeque::new(), asked: BTreeMap::new(), progress: 0, seen: (0, 0), still: 0, stalled: false, reported: BTreeSet::new(), waits_on: BTreeSet::new(), regions: BTreeMap::new(), reclaiming: BTreeSet::new(), logged: None, faulted: false, ending: None };
            b.executors.insert(id, exec);
            id
        };
        end.send(&ToAgent::Welcome { epoch }, Some(os::lend(&file)?))?;
        drop(file);
        loop {
            match end.recv::<ToManager>() {
                Ok(Some((message, _))) => {
                    let out = match message {
                        ToManager::Ask { id: ask, bytes, kind, region, fill } => self.ask(id, Waiting { id: ask, bytes, kind, region, fill, entry: None }),
                        ToManager::Derive { key, bytes } => self.derive(id, key, bytes),
                        ToManager::Derived { key, entry } => {
                            if let Some(domain) = self.domain(id) {
                                self.hosting.derived(&domain, &key, entry);
                            }
                            Out::default()
                        }
                        ToManager::Answered => self.answered(id),
                        ToManager::Region(facts) => self.region(id, facts.id, Some(facts)),
                        ToManager::RegionGone { id: region } => self.region(id, region, None),
                        ToManager::Measured(m) => self.measured(id, m),
                        ToManager::Hello { .. } => Out::default(),
                    };
                    self.emit(out);
                }
                Ok(None) => break,
                Err(error) => {
                    tracing::warn!("lease: executor {}: {error}", peer.pid);
                    break;
                }
            }
        }
        // The socket closed: its leases come back once the process's exit shows (its mappings
        // are gone then), never before.
        process.wait().map_err(|e| Error::Io(e.to_string()))?;
        let out = self.exited(id);
        self.emit(out);
        Ok(())
    }

    fn emit(&self, out: Out) {
        let mut failed = vec![];
        for (i, (end, message, handle)) in out.sends.into_iter().enumerate() {
            if let Err(error) = end.send(&message, handle) {
                tracing::debug!("lease: a send to an executor failed (it is going): {error}");
                if let (Some((_, exec, slot)), ToAgent::Grant { id, .. }) = (out.grants.iter().find(|(at, ..)| *at == i), &message) {
                    failed.push((*exec, *slot, *id));
                }
            }
        }
        // A grant that never reached its agent goes back: its slot free, its chunk destroyed,
        // its ask refused (never left waiting on a grant it cannot see).
        if !failed.is_empty() {
            let mut refusals = Out::default();
            let mut b = self.books.lock();
            for (exec, slot, ask) in failed {
                let gone = b.executors.get_mut(&exec).and_then(|e| {
                    let _ = e.table.reclaim(slot, true);
                    e.free.insert(slot);
                    e.leased.remove(&slot)
                });
                if let Some(l) = gone {
                    self.destroy(&mut b, l.phys);
                }
                refuse(&mut b, &mut refusals, exec, ask, Refusal::Failed, "its grant could not be sent".into());
            }
            drop(b);
            for (end, message, handle) in refusals.sends {
                let _ = end.send(&message, handle);
            }
        }
        if !out.end_idle.is_empty() {
            self.hosting.end_idle(&out.end_idle);
        }
        for record in out.records {
            self.hosting.record(record);
        }
        for (domain, key) in out.forgets {
            self.hosting.forget_derived(&domain, &key);
        }
        drop(out.idle);
        self.events.set();
    }

    /// The backend, opened now if it is not (outside `books`); the open's failure refuses.
    fn device(&self) -> Result<Arc<Device>> {
        if let Some(d) = self.books.lock().device.clone() {
            return Ok(d);
        }
        let Some(open) = &self.open else { return Err(Error::Invalid("this GPU has no VMM for leases".into())) };
        let _one = self.opening.lock();
        if let Some(d) = self.books.lock().device.clone() {
            return Ok(d);
        }
        let backend = open()?;
        let card = self.limit.unwrap_or(self.gpu_memory).min(backend.capacity());
        let device = Arc::new(Device { card, granularity: backend.granularity().max(1), headroom: backend.headroom(), context: backend.opened_context(), backend });
        let mut b = self.books.lock();
        (b.card, b.reserve) = (device.card, device.headroom);
        b.device = Some(device.clone());
        drop(b);
        Ok(device)
    }

    /// The driver's free bytes now, read outside the lock between the created counter (read
    /// before) and the destroyed one (read after): a chunk created during the read counts twice,
    /// one destroyed during it not at all until the next read, so neither over-grants. `None`
    /// when nothing could be read: then nothing grows.
    fn read(&self) -> Option<Sample> {
        let (device, created) = {
            let b = self.books.lock();
            (b.device.clone(), b.created_total)
        };
        let free = self.hosting.free().or_else(|| device?.backend.free().ok())?;
        let destroyed = self.books.lock().destroyed_total;
        let available = u64::try_from(host_memory::read().available).unwrap_or(0);
        let host = policy::Host { available, total: host_memory::limit().unwrap_or(0), ..policy::Host::default() };
        Some(Sample { free, created, destroyed, host })
    }

    /// TensorD's reading of device bytes no process of ours holds (shown only: the driver's free
    /// bytes already count them), and whether a new foreign process just crossed its threshold.
    /// Every reading is also the background-reclaim tick: free under low reclaims cold pages up
    /// to high (§12), a new foreign process takes every cold page.
    pub fn set_foreign(&self, bytes: u64, new_pid: bool) {
        let sample = self.read();
        let out = {
            let mut b = self.books.lock();
            b.sample = sample;
            b.foreign = bytes;
            b.new_foreign = new_pid;
            self.settle(&mut b)
        };
        self.emit(out);
    }

    /// A GPU fault or sticky error: the page cache goes. `pids` failed or were poisoned: from
    /// now their chunks neither count nor are asked for, their waiting asks are refused, and
    /// their exit destroys every chunk (none is kept for adoption).
    pub fn fault(&self, pids: &[u32]) {
        let out = {
            let mut b = self.books.lock();
            let mut out = Out::default();
            let faulted: Vec<u64> = b.executors.iter().filter(|(_, e)| pids.contains(&e.pid)).map(|(id, _)| *id).collect();
            for id in faulted {
                let asks: Vec<u64> = b.executors.get_mut(&id).map(|e| {
                    e.faulted = true;
                    e.waiting.drain(..).map(|w| w.id).collect()
                }).unwrap_or_default();
                for ask in asks {
                    refuse(&mut b, &mut out, id, ask, Refusal::Failed, "its executor failed".into());
                }
            }
            for key in b.order.clone() {
                self.drop_retained(&mut b, &key, &mut out);
            }
            out.append(self.settle(&mut b));
            out.idle = Self::idle(&mut b);
            out
        };
        self.emit(out);
    }

    /// What an agent measured: a request's peaks under TensorD's dispatch key (the larger stays;
    /// the compute bytes are the newest), a set's bytes (the larger).
    fn measured(&self, exec: u64, m: Measured) -> Out {
        let mut b = self.books.lock();
        if !b.executors.contains_key(&exec) {
            return Out::default();
        }
        match m {
            Measured::Request { request, working, compute, host } => {
                let Some(key) = b.dispatched.get(&request).cloned() else { return Out::default() };
                larger(&mut b.figures, format!("peak/{key}"), working);
                b.figures.insert(format!("compute/{key}"), compute);
                larger(&mut b.figures, format!("host/{key}"), host);
            }
            Measured::Set { name, bytes } => larger(&mut b.figures, format!("set/{name}"), bytes),
        }
        Out::default()
    }

    /// A region's facts (`None`: gone). One reported with no chunks after its reclaim is no
    /// longer being reclaimed.
    fn region(&self, exec: u64, id: u64, facts: Option<RegionFacts>) -> Out {
        let mut b = self.books.lock();
        if let Some(e) = b.executors.get_mut(&exec) {
            match facts {
                Some(f) => {
                    if f.slots.is_empty() {
                        e.reclaiming.remove(&id);
                    }
                    e.regions.insert(id, f);
                }
                None => {
                    e.regions.remove(&id);
                    e.reclaiming.remove(&id);
                }
            }
        }
        self.settle(&mut b)
    }

    // ---- Admission and placement (TensorD's): its own dispatches and the measured peaks.

    /// The device bytes requests can use: free and every chunk of ours (all of it reclaimable),
    /// less the reserved pool; 0 until the device was read (a request then waits).
    fn usable(b: &Books) -> u64 {
        free(b).map_or(0, |f| (f + b.created).saturating_sub(b.reserve))
    }

    /// A fresh reading for admission and placement, which run before anything is asked.
    fn refresh(&self) {
        if let Some(sample) = self.read() {
            self.books.lock().sample = Some(sample);
        }
    }

    /// The measured peaks of every dispatched request (`None`: unmeasured).
    fn dispatched_peaks(b: &Books) -> Vec<Option<u64>> {
        b.dispatched.values().map(|key| figure(b, "peak", key)).collect()
    }

    /// Whether a request of `package` at `shape` (empty: its widest measured shape) can start now
    /// beside every request dispatched here ([`decide::fits`]): their measured peaks and its own
    /// within the usable bytes; anything unmeasured only on an idle GPU. A request that does not
    /// fit queues before it starts, so two never wait on each other's protected steps.
    pub fn fits(&self, package: &str, shape: &str) -> bool {
        self.refresh();
        let b = self.books.lock();
        decide::fits(Self::usable(&b), 0, &[], &Self::dispatched_peaks(&b), figure(&b, "peak", &format!("{package}|{shape}")), None)
    }

    /// A multi-GPU group: dispatches `request` on every GPU of `group` if it fits on all, else
    /// on none.
    pub fn admit(group: &[&Manager], request: &str, package: &str, shape: &str) -> Result<bool> {
        if !group.iter().all(|m| m.fits(package, shape)) {
            return Ok(false);
        }
        for m in group {
            m.dispatch(request, package, shape)?;
        }
        Ok(true)
    }

    /// Names the `<package>|<shape>` a request's peaks are recorded under, before it runs.
    pub fn dispatch(&self, request: &str, package: &str, shape: &str) -> Result<()> {
        self.books.lock().dispatched.insert(request.to_string(), format!("{package}|{shape}"));
        Ok(())
    }

    pub fn undispatch(&self, request: &str) {
        self.books.lock().dispatched.remove(request);
    }

    /// The GPU of `group` (an index) that keeps each set of `package`'s request at `shape`
    /// ([`decide::place`]). A set's bytes are its layout's when the caller has them, else as a
    /// load measured them on any GPU of the group. Each GPU's room is its usable bytes less its
    /// dispatched requests' measured peaks; GPU 0's less the package's measured compute bytes
    /// too.
    pub fn place(group: &[&Manager], package: &str, shape: &str, sets: &[(&str, Option<u64>)]) -> Result<Vec<u32>> {
        let key = format!("{package}|{shape}");
        let mut rooms = Vec::with_capacity(group.len());
        for (g, m) in group.iter().enumerate() {
            m.refresh();
            let b = m.books.lock();
            let running: u64 = Self::dispatched_peaks(&b).into_iter().flatten().sum();
            let working = if g == 0 { figure(&b, "compute", &key).unwrap_or(0) } else { 0 };
            rooms.push(Self::usable(&b).saturating_sub(running + working));
        }
        let bytes: Vec<Option<u64>> = sets.iter().map(|(name, known)| known.or_else(|| group.iter().filter_map(|m| figure(&m.books.lock(), "set", name)).max())).collect();
        Ok(decide::place(&rooms, &bytes))
    }

    /// A request's measured anonymous host peak at `<package>|<shape>`: TensorD's host rule.
    pub fn host_peak(&self, package: &str, shape: &str) -> Option<u64> {
        figure(&self.books.lock(), "host", &format!("{package}|{shape}"))
    }

    /// The measured peaks, to persist per GPU model.
    pub fn peaks(&self) -> Vec<Peak> {
        let b = self.books.lock();
        let peak = |(k, &bytes): (&String, &u64)| {
            let (kind, key) = k.split_once('/')?;
            let key = key.to_string();
            Some(match kind {
                "peak" => Peak::Request { key, bytes },
                "compute" => Peak::Compute { key, bytes },
                "set" => Peak::Set { key, bytes },
                "host" => Peak::Host { key, bytes },
                _ => return None,
            })
        };
        b.figures.iter().filter_map(peak).collect()
    }

    /// Peaks saved earlier on this machine for this GPU model: what this run measured wins (sets
    /// and host peaks: the larger).
    pub fn restore(&self, peaks: &[Peak]) -> Result<()> {
        let mut b = self.books.lock();
        for p in peaks {
            match p {
                Peak::Request { key, bytes } => drop(b.figures.entry(format!("peak/{key}")).or_insert(*bytes)),
                Peak::Compute { key, bytes } => drop(b.figures.entry(format!("compute/{key}")).or_insert(*bytes)),
                Peak::Set { key, bytes } => larger(&mut b.figures, format!("set/{key}"), *bytes),
                Peak::Host { key, bytes } => larger(&mut b.figures, format!("host/{key}"), *bytes),
            }
        }
        Ok(())
    }

    /// The backend goes once nothing needs it; its drop (a context's destroy) runs unlocked.
    fn idle(b: &mut Books) -> Option<Arc<Device>> {
        (b.created == 0 && b.executors.is_empty()).then(|| b.device.take()).flatten()
    }

    fn domain(&self, exec: u64) -> Option<String> {
        self.books.lock().executors.get(&exec).map(|e| e.identity.domain.clone())
    }

    /// A disk entry for the agent to write `key`'s derived bytes into, or none kept: the host's
    /// call, outside the books.
    fn derive(&self, exec: u64, key: String, bytes: u64) -> Out {
        let Some((domain, end)) = self.books.lock().executors.get(&exec).map(|e| (e.identity.domain.clone(), e.end.clone())) else { return Out::default() };
        let (entry, file) = self.hosting.derive(&domain, &key, bytes).map_or((0, None), |e| (e.id, Some(Owned::from(e.file))));
        let mut out = Out::default();
        out.sends.push((end, ToAgent::Entry { id: 0, key, entry }, file));
        out
    }

    fn ask(&self, exec: u64, mut ask: Waiting) -> Out {
        // A derived entry kept for its fill, read outside the books (a grant that adopts drops it).
        if let (Kind::Weights, Some(key), Some(domain)) = (ask.kind, &ask.fill, self.domain(exec)) {
            ask.entry = self.hosting.read_derived(&domain, key);
        }
        if let Err(error) = self.device() {
            let b = self.books.lock();
            let mut out = Out::default();
            if let Some(e) = b.executors.get(&exec) {
                out.sends.push((e.end.clone(), ToAgent::Refuse { id: ask.id, code: Refusal::Failed, why: error.to_string() }, None));
                out.records.push(Record::Refuse { pid: e.pid, ask: ask.id, code: Refusal::Failed, why: error.to_string() });
            }
            return out;
        }
        let sample = self.read();
        let mut b = self.books.lock();
        b.sample = sample;
        if b.executors.get(&exec).is_some_and(|e| e.faulted) {
            let mut out = Out::default();
            refuse(&mut b, &mut out, exec, ask.id, Refusal::Failed, "its executor failed".into());
            return out;
        }
        if let Some(e) = b.executors.get_mut(&exec) {
            e.waiting.push_back(ask);
        }
        self.settle(&mut b)
    }

    /// The agent moved slots: fills acked landed are recorded; Released and Refused slots are
    /// reclaimed, their chunks destroyed; then the waiting asks (their victims picked again).
    fn answered(&self, exec: u64) -> Out {
        let start = Instant::now();
        let sample = self.read();
        let mut b = self.books.lock();
        b.sample = sample;
        let mut gone = vec![];
        let mut out = Out::default();
        if let Some(e) = b.executors.get_mut(&exec) {
            let mut done = vec![];
            for (slot, l) in e.leased.iter_mut() {
                match e.table.word(*slot).map(|w| (w.state, w.seq)) {
                    Some((State::Mapped, seq)) if seq == l.seq => l.landed = e.table.landed(*slot),
                    Some((State::Released | State::Refused, seq)) if seq == l.seq => done.push(*slot),
                    _ => {}
                }
            }
            for slot in done {
                if let (Ok(Some(_)), Some(l)) = (e.table.reclaim(slot, false), e.leased.remove(&slot)) {
                    e.free.insert(slot);
                    e.asked.remove(&slot);
                    e.progress += 1;
                    gone.push(l.phys);
                }
            }
            if !gone.is_empty() {
                out.records.push(Record::Released { pid: e.pid, released: gone.len(), us: 0 });
            }
        }
        for phys in gone {
            self.destroy(&mut b, phys);
        }
        if let Some(Record::Released { us, .. }) = out.records.last_mut() {
            *us = micros(start);
        }
        out.append(self.settle(&mut b));
        out.idle = Self::idle(&mut b);
        out
    }

    /// An exited executor's every slot: landed weights (by the manager's records) retained in
    /// its domain, the rest destroyed.
    fn exited(&self, exec: u64) -> Out {
        let start = Instant::now();
        let sample = self.read();
        let mut b = self.books.lock();
        b.sample = sample;
        let Some(e) = b.executors.remove(&exec) else { return Out::default() };
        let (mut gone, mut retained) = (vec![], 0);
        for (slot, l) in e.leased {
            // Both say landed: the manager's record of this grant's ack, and the table now (an
            // unland whose doorbell was still queued wins).
            match l.region {
                Some(region) if !e.faulted && l.landed && e.table.landed(slot) && e.table.word(slot).is_some_and(|w| w.seq == l.seq) => {
                    retained += l.phys.len;
                    let key = (e.identity.domain.clone(), region);
                    if let Some(fill) = l.fill {
                        b.fills.insert(key.clone(), fill);
                    }
                    let kept = b.retained.entry(key.clone()).or_default();
                    let first = kept.is_empty();
                    gone.extend(kept.insert(l.index, l.phys));
                    if first {
                        b.order.push_back(key);
                    }
                }
                _ => gone.push(l.phys),
            }
        }
        let destroyed = gone.iter().map(|p| p.len).sum();
        for phys in gone {
            self.destroy(&mut b, phys);
        }
        let mut out = self.settle(&mut b);
        out.records.insert(0, Record::Exit { pid: e.pid, freed_bytes: destroyed, cached_bytes: retained, us: micros(start) });
        out.idle = Self::idle(&mut b);
        out
    }

    fn destroy(&self, b: &mut Books, phys: Physical) {
        b.created = b.created.saturating_sub(phys.len);
        b.destroyed_total += phys.len;
        b.created_chunks = b.created_chunks.saturating_sub(1);
        let Some(d) = &b.device else { return };
        if let Err(error) = d.backend.destroy(phys) {
            tracing::warn!("lease: destroying a chunk: {error}");
        }
    }

    fn drop_retained(&self, b: &mut Books, key: &Kept, out: &mut Out) {
        b.order.retain(|k| k != key);
        for phys in b.retained.remove(key).unwrap_or_default().into_values() {
            self.destroy(b, phys);
        }
        if let Some(fill) = b.fills.remove(key) {
            out.forgets.push((key.0.clone(), fill));
        }
    }

    /// Decides and applies until nothing more can move at once, then records the wait spans and
    /// publishes every table's figures.
    fn settle(&self, b: &mut Books) -> Out {
        let mut out = Out::default();
        while self.settle_once(b, &mut out) {}
        self.waited(b);
        self.publish(b);
        out
    }

    /// The decision's snapshot: the page accounting, every waiting ask in each executor's
    /// order, and the page cache's keys by the index the decision names them.
    fn snapshot(&self, b: &Books, free: u64) -> (policy::Snapshot, Vec<policy::Ask>, Vec<Kept>) {
        let (reserved, granularity) = (b.reserve, b.device.as_ref().map_or(1, |d| d.granularity));
        let cache: Vec<Kept> = b.order.iter().cloned().collect();
        let (mut asks, mut executors) = (vec![], vec![]);
        for (id, e) in b.executors.iter().filter(|(_, e)| !e.faulted) {
            for w in &e.waiting {
                let sizes = sizes(w, granularity);
                let hit = cached(b, &e.identity.domain, w.region, &sizes).and_then(|k| cache.iter().position(|c| *c == k));
                asks.push(policy::Ask { executor: *id, bytes: sizes.iter().sum(), speculative: false, cached: hit });
            }
            let chunk = |(slot, l): (&u32, &Leased)| {
                let w = e.table.word(*slot).filter(|w| w.seq == l.seq)?;
                let asked = e.asked.get(slot);
                let overdue = asked.is_some_and(|at| at.elapsed() >= LOOK * LOOKS);
                Some(policy::Chunk { slot: *slot, seq: l.seq, bytes: l.phys.len, state: w.state, pins: w.pins, asked: asked.is_some(), overdue, advice: e.table.advice(*slot) })
            };
            let waits_on = if e.waiting.is_empty() { vec![] } else { e.waits_on.iter().copied().collect() };
            let (mut regions, mut graphs) = (vec![], vec![]);
            for (rid, f) in &e.regions {
                let next_use = (f.next_use != u64::MAX).then_some(f.next_use);
                match f.kind {
                    crate::region::Reclaimable::Graph => graphs.push(policy::Graph { id: *rid, chunks: f.slots.clone(), last_use: f.last_use, next_use, capture_ns: f.cost_ns, dropping: e.reclaiming.contains(rid) }),
                    // One being reclaimed already: its chunks are asked; it is no candidate.
                    _ if e.reclaiming.contains(rid) => {}
                    kind => regions.push(policy::Region { id: *rid, chunks: f.slots.clone(), swappable: kind == crate::region::Reclaimable::Swappable, last_use: f.last_use, next_use, produce_ns: f.cost_ns }),
                }
            }
            executors.push(policy::Executor { id: *id, chunks: e.leased.iter().filter_map(chunk).collect(), regions, graphs, waits_on, running: e.table.running() });
        }
        let page_cache = cache.iter().map(|k| b.retained.get(k).map_or(0, |v| v.values().map(|p| p.len).sum())).collect();
        let host = b.sample.map(|s| s.host).unwrap_or_default();
        let snapshot = policy::Snapshot { free, reserved, new_foreign: b.new_foreign, page_cache, executors, host };
        (snapshot, asks, cache)
    }

    /// One decision applied: reclaim requests sent, page-cache entries dropped, asks allocated,
    /// left waiting or refused. True when dropped entries freed room at once: another round
    /// allocates it.
    fn settle_once(&self, b: &mut Books, out: &mut Out) -> bool {
        // Unread (no device yet, or the driver's read failed): nothing grows or is reclaimed; the
        // asks wait for the next read.
        let Some(free) = free(b) else { return false };
        let (snapshot, asks, cache) = self.snapshot(b, free);
        b.new_foreign = false;
        let decision = policy::decide(&snapshot, &asks);
        let (mut freed, mut asked) = (false, BTreeMap::<u64, Vec<u32>>::new());
        for pick in decision.picks {
            match pick {
                Pick::Chunk { executor, slot, seq, .. } => {
                    let Some(e) = b.executors.get_mut(&executor) else { continue };
                    if e.table.request(slot, seq).is_ok() {
                        e.asked.insert(slot, Instant::now());
                        asked.entry(executor).or_default().push(slot);
                    }
                }
                Pick::PageCache { index, .. } => {
                    if let Some(key) = cache.get(index) {
                        self.drop_retained(b, key, out);
                        freed = true;
                    }
                }
                // A region or graph whole: its agent swaps, discards or drops it (its chunks follow
                // as reclaim requests).
                Pick::Region { executor, region, swap } => self.reclaim_region(b, out, executor, region, swap),
                Pick::Graph { executor, region } => self.reclaim_region(b, out, executor, region, false),
            }
        }
        for (id, slots) in asked {
            if let Some(e) = b.executors.get(&id) {
                out.sends.push((e.end.clone(), ToAgent::Revoke, None));
                out.records.push(Record::Reclaim { pid: e.pid, slots });
            }
        }
        let mut stuck = BTreeSet::new();
        for (a, answer) in asks.iter().zip(decision.answers) {
            let exec = a.executor;
            if stuck.contains(&exec) {
                continue;
            }
            match answer {
                Answer::Allocate => {
                    let Some(ask) = b.executors.get_mut(&exec).and_then(|e| e.waiting.pop_front()) else { continue };
                    match self.answer(b, exec, &ask) {
                        Ok(Some(allocated)) => {
                            out.append(allocated);
                            if let Some(e) = b.executors.get_mut(&exec) {
                                e.waits_on.clear();
                            }
                        }
                        Ok(None) => {
                            if let Some(e) = b.executors.get_mut(&exec) {
                                e.waiting.push_front(ask);
                            }
                            stuck.insert(exec);
                        }
                        Err(error) => refuse(b, out, exec, ask.id, Refusal::Failed, error.to_string()),
                    }
                }
                Answer::Wait { on } => {
                    let pids: Vec<u32> = on.iter().filter_map(|x| b.executors.get(x).map(|e| e.pid)).collect();
                    let coming = policy::coming_for(&snapshot, exec);
                    let own = snapshot.executors.iter().find(|x| x.id == exec).map_or((0, 0), |x| {
                        let sum = |f: &dyn Fn(&policy::Chunk) -> bool| x.chunks.iter().filter(|c| f(c)).map(|c| c.bytes).sum::<u64>();
                        (sum(&|c| c.state == State::Revoking), sum(&|c| c.state == State::Mapped && c.asked))
                    });
                    if let Some(e) = b.executors.get_mut(&exec) {
                        if let Some(ask) = e.waiting.front() {
                            let now = (ask.id, coming, on.clone());
                            if e.logged.as_ref() != Some(&now) {
                                out.records.push(Record::Wait { pid: e.pid, ask: ask.id, bytes: ask.bytes, coming, on: pids, own_revoking: own.0, own_asked: own.1 });
                                e.logged = Some(now);
                            }
                        }
                        e.waits_on = on.into_iter().collect();
                    }
                    // Starts the look's cadence if it slept; never makes a look early.
                    self.looking.notify_all();
                }
                Answer::Refuse { room } => {
                    // Idle executors pinning room: the host ends them and the ask waits for
                    // their exit (logged), when it ends idle executors at all.
                    let idle: Vec<u64> = if self.hosting.ends_idle() {
                        b.executors.iter().filter(|(id, e)| **id != exec && !e.faulted && !e.table.running() && e.leased.keys().any(|s| e.table.word(*s).is_some_and(|w| w.pins > 0))).map(|(id, _)| *id).collect()
                    } else {
                        vec![]
                    };
                    // Waited for only while one has not outlived the looks since it was named.
                    let idle: Vec<u64> = idle.into_iter().filter(|id| b.executors.get(id).is_some_and(|e| e.ending.is_none_or(|at| at.elapsed() < LOOK * LOOKS))).collect();
                    if !idle.is_empty() {
                        let mut ended = vec![];
                        for id in &idle {
                            if let Some(e) = b.executors.get_mut(id).filter(|e| e.ending.is_none()) {
                                e.ending = Some(Instant::now());
                                ended.push(e.pid);
                            }
                        }
                        if !ended.is_empty() {
                            let (pid, ask) = b.executors.get(&exec).map_or((0, 0), |e| (e.pid, e.waiting.front().map_or(0, |w| w.id)));
                            out.records.push(Record::EndIdle { pid, ask, idle: ended.clone() });
                            out.end_idle.extend(ended);
                        }
                        continue;
                    }
                    if let Some(ask) = b.executors.get_mut(&exec).and_then(|e| e.waiting.pop_front()) {
                        refuse(b, out, exec, ask.id, Refusal::NothingRevocable, format!("nothing reclaimable for {} bytes ({room} could come)", ask.bytes));
                    }
                }
                Answer::Behind => {}
            }
        }
        freed
    }

    fn reclaim_region(&self, b: &mut Books, out: &mut Out, exec: u64, region: u64, swap: bool) {
        if let Some(e) = b.executors.get_mut(&exec) {
            e.reclaiming.insert(region);
            out.sends.push((e.end.clone(), ToAgent::Reclaim { region, swap }, None));
        }
    }

    /// PSI's spans: one opens when an ask starts waiting and closes when none does.
    fn waited(&self, b: &mut Books) {
        let now = self.now();
        match (b.executors.values().any(|e| !e.waiting.is_empty()), b.waiting) {
            (true, None) => b.waiting = Some(now),
            (false, Some(start)) => {
                b.waits.push_back((start, now));
                b.waiting = None;
            }
            _ => {}
        }
        while b.waits.front().is_some_and(|(_, end)| now.saturating_sub(*end) > 60_000_000_000) {
            b.waits.pop_front();
        }
    }

    /// Every table's figures: free bytes, and the cold room reclaim could give that executor.
    fn publish(&self, b: &Books) {
        let headroom = b.reserve;
        let (snapshot, _, _) = self.snapshot(b, free(b).unwrap_or(0));
        let free = room(b);
        for (id, e) in &b.executors {
            e.table.set_figures(Figures { free, revocable: policy::cold(&snapshot, *id), headroom, foreign: b.foreign });
        }
    }

    /// The grants for `ask` when it fits now; `None` when room must be made first. A failure
    /// partway rolls the whole ask back: no slot stays granted, no chunk leaks.
    fn answer(&self, b: &mut Books, exec: u64, ask: &Waiting) -> Result<Option<Out>> {
        let start = Instant::now();
        let Some(dev) = b.device.clone() else { return Err(Error::Closed("the GPU's backend".into())) };
        let sizes = sizes(ask, dev.granularity);
        let Some(pid) = b.executors.get(&exec).map(|e| e.pid) else { return Err(Error::Closed("the executor".into())) };
        let domain = b.executors.get(&exec).map(|e| e.identity.domain.clone()).unwrap_or_default();
        let adopt = cached(b, &domain, ask.region, &sizes);
        let kept_fill = adopt.as_ref().and_then(|key| b.fills.remove(key));
        let phys: Vec<Physical> = if let Some(key) = &adopt {
            b.order.retain(|k| k != key);
            b.retained.remove(key).unwrap_or_default().into_values().collect()
        } else {
            if sizes.iter().sum::<u64>() > room(b) {
                return Ok(None);
            }
            let mut made = vec![];
            for &len in &sizes {
                match dev.backend.create(len) {
                    Ok(p) => made.push(p),
                    Err(error) => {
                        made.into_iter().for_each(|p| drop(dev.backend.destroy(p)));
                        return Err(error);
                    }
                }
            }
            b.created += sizes.iter().sum::<u64>();
            b.created_total += sizes.iter().sum::<u64>();
            b.created_chunks += sizes.len();
            made
        };
        let create_us = micros(start);
        let exporting = Instant::now();
        let n = phys.len();
        let bytes: u64 = phys.iter().map(|p| p.len).sum();
        let mut out = Out::default();
        // Not adopted, with a derived entry kept for its fill: the entry first, to refill from.
        if let (None, Some(key), Some(entry), Some(e)) = (&adopt, &ask.fill, &ask.entry, b.executors.get(&exec)) {
            match entry.try_clone() {
                Ok(entry) => out.sends.push((e.end.clone(), ToAgent::Entry { id: ask.id, key: key.clone(), entry: 0 }, Some(Owned::from(entry)))),
                Err(error) => tracing::debug!("lease: a derived entry for {key}: {error}"),
            }
        }
        let fill = ask.fill.clone().or_else(|| kept_fill.clone());
        let mut granted: Vec<u32> = vec![];
        let mut failed = None;
        let mut spare = vec![];
        for (index, p) in phys.into_iter().enumerate() {
            if failed.is_some() {
                spare.push(p);
                continue;
            }
            b.seq = b.seq.wrapping_add(1).max(1);
            let seq = b.seq;
            let Some(e) = b.executors.get_mut(&exec) else {
                spare.push(p);
                failed = Some(Error::Closed("the executor".into()));
                continue;
            };
            let Some(slot) = e.free.pop_first() else {
                spare.push(p);
                failed = Some(Error::Invalid("the executor's lease table is full".into()));
                continue;
            };
            let content = ask.region.map(|r| Content { fingerprint: r.fingerprint, region: r.region, index: index as u32 });
            let exported = e.table.grant(slot, seq, p.len, content, adopt.is_some()).and_then(|ok| if ok { dev.backend.lend(&p).map(Some) } else { Ok(None) });
            match exported {
                Ok(Some(handle)) => {
                    out.grants.push((out.sends.len(), exec, slot));
                    out.sends.push((e.end.clone(), ToAgent::Grant { id: ask.id, slot, seq, bytes: p.len, adopted: adopt.is_some(), last: index + 1 == n }, Some(handle)));
                    e.leased.insert(slot, Leased { phys: p, seq, region: ask.region, index: index as u32, landed: adopt.is_some(), fill: fill.clone() });
                    granted.push(slot);
                }
                Ok(None) => {
                    e.free.insert(slot);
                    spare.push(p);
                    failed = Some(Error::Invalid(format!("lease slot {slot} was not free")));
                }
                Err(error) => {
                    let _ = e.table.reclaim(slot, true);
                    e.free.insert(slot);
                    spare.push(p);
                    failed = Some(error);
                }
            }
        }
        let Some(error) = failed else {
            let adopted = adopt.is_some();
            out.records.push(Record::Allocate { pid, ask: ask.id, chunks: n, bytes, cached: adopted, create_us, export_us: micros(exporting), total_us: micros(start) });
            if let (true, Some(region)) = (adopted, ask.region) {
                out.records.push(Record::Cached { pid, region, chunks: n, bytes });
            }
            return Ok(Some(out));
        };
        // Roll back: the slots granted so far are Free again, their chunks with the rest.
        if let Some(e) = b.executors.get_mut(&exec) {
            for slot in granted {
                let _ = e.table.reclaim(slot, true);
                e.free.insert(slot);
                if let Some(l) = e.leased.remove(&slot) {
                    spare.insert(l.index as usize, l.phys);
                }
            }
        }
        match adopt {
            // Adopted chunks go back to retention whole, by index: nothing was copied.
            Some(key) if spare.len() == n => {
                b.retained.insert(key.clone(), spare.into_iter().enumerate().map(|(i, p)| (i as u32, p)).collect());
                if let Some(fill) = kept_fill {
                    b.fills.insert(key.clone(), fill);
                }
                b.order.push_front(key);
            }
            _ => spare.into_iter().for_each(|p| self.destroy(b, p)),
        }
        Err(error)
    }

}

fn refuse(b: &mut Books, out: &mut Out, exec: u64, ask: u64, code: Refusal, why: String) {
    let Some(e) = b.executors.get_mut(&exec) else { return };
    e.waits_on.clear();
    out.sends.push((e.end.clone(), ToAgent::Refuse { id: ask, code, why: why.clone() }, None));
    out.records.push(Record::Refuse { pid: e.pid, ask, code, why });
}

/// The stall look: every [`LOOK`] while asks are out, else asleep until one goes out.
fn look_loop(manager: &Weak<Manager>) {
    // A look runs only on the cadence: a wake (asks going out) never makes one early.
    let mut last: Option<Instant> = None;
    loop {
        let Some(m) = manager.upgrade() else { return };
        if let Some(wait) = last.map(|at| LOOK.saturating_sub(at.elapsed())).filter(|w| !w.is_zero()) {
            let mut b = m.books.lock();
            m.looking.wait_for(&mut b, wait);
            continue;
        }
        last = Some(Instant::now());
        let (out, records, again) = m.look();
        for record in records {
            m.hosting.record(record);
        }
        if again {
            m.decide_again();
        }
        if !out {
            let mut b = m.books.lock();
            m.looking.wait_for(&mut b, LOOK * 20);
            // Asks went out (or the idle period ended): the first look is a full period later.
            last = Some(Instant::now());
        }
    }
}

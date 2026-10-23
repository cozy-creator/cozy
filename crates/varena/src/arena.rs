//! varena as torch's only GPU allocator (varena-leases §1, §11; step-6 design and review R1–R11).
//! Torch's blocks are carved out of leased chunks, mapped in spans of the device's own address range:
//! - a block of at most [`SMALL`] bytes takes a slot of its size class, in a page carved from a 2 MiB run;
//! - a larger one is best-fit in a run: a 2 MiB span up to [`MEDIUM`], a [`CHUNK`]-or-larger one above;
//! - a block of [`OWN`] bytes or more maps a span of its own.
//!
//! Arenas: the process's, one per request, one per cache region (M2, R11) and one per graph pool. A
//! span is one arena's and one stream's, so its free blocks serve that stream at once: stream order
//! covers them. A span that comes whole free leaves its arena for the device's idle list, which any
//! arena re-adopts with no driver call and the [`Source`] reclaims from ([`Torch::reclaim`]).
//!
//! - R1: a span leaves its stream (to another stream or the source) only once an epoch event recorded
//!   on its stream after its last free has completed. A poller thread records and queries the epochs,
//!   never under an arena lock (R11a). A block torch recorded on other streams frees after theirs.
//! - R2: while a capture is open, a non-pool block's free waits for the capture's end, and nothing is
//!   recorded: [`Torch::pool_begin`] records every pending epoch before the capture starts, so an idle
//!   span always completes by GPU progress alone.
//! - R4/R5: a pool keeps every span until its use count is 0 and its fences have completed.
//! - R6: an allocation routed to a pool, or made on a capturing thread, is served from mapped free
//!   bytes or refused. It never reaches the source.
//! - R10: an arena lock is taken alone, or before a leaf (idle, address space, directory, gate). The
//!   source is called with no lock held, so the arenas never nest with Memory's state lock.
//! - No learning loop: a shortfall waits for the source's grant with no lock held. It is refused only
//!   when the source refuses and the process has no idle span left to give back.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::atomic::Ordering::{AcqRel, Acquire, Release, SeqCst};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize};
use std::sync::{Arc, Weak};
use std::thread::ThreadId;
use std::time::Duration;

use parking_lot::{Condvar, Mutex, RwLock};

use crate::backend::{Access, Backend, Event, Physical, Va, OUT_OF_MEMORY};
use crate::{Error, Exhausted, Result};

/// A span's length, and every chunk's, is a multiple of this.
pub const UNIT: u64 = 2 << 20;
/// The largest chunk (the manager's lease size).
pub const CHUNK: u64 = 32 << 20;
/// A block this big maps a span of its own.
pub const OWN: u64 = 64 << 20;
/// Blocks up to this take a size-class slot.
const SMALL: u64 = 64 << 10;
/// Runs up to this live in 2 MiB spans, so a small survivor pins little.
const MEDIUM: u64 = 1 << 20;
/// torch's rounding.
const ALIGN: u64 = 512;
const CLASSES: [u64; 15] = [512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536];
/// A pool span no stream has used yet: the capture stream that first needs it takes it.
const ANY: usize = usize::MAX;
/// The poller's cadence while epochs are outstanding: it doubles to the longest while nothing lands.
const NAP: Duration = Duration::from_micros(50);
const NAP_MAX: Duration = Duration::from_millis(2);

/// torch's `MempoolId_t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PoolId(pub u64, pub u64);

/// Whose arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Owner {
    Process,
    Request(u64),
    /// A cache region (M2): its blocks never share a span with another arena's.
    Region(u64),
    Pool(PoolId),
}

/// What a chunk is leased for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Torch,
    Pool,
}

/// One chunk mapped in a span: the source's id for it, and its bytes.
#[derive(Debug, PartialEq, Eq)]
pub struct Chunk {
    pub id: u64,
    pub len: u64,
}

/// Where the arenas' chunks come from: the lease agent (its float, then the manager), or a
/// standalone budget ([`Direct`]).
pub trait Source: Send + Sync {
    /// Maps chunks covering `va` (its length a multiple of [`UNIT`]; each chunk at most [`CHUNK`],
    /// in address order) and returns them. It waits, with no lock of the caller's held, until they
    /// are granted, and the wait never needs Python: the caller may hold the GIL. It is never called
    /// on a capturing thread; its driver calls run in relaxed capture mode, since another thread may
    /// be capturing. It errs only at the floor: `Exhausted` (nothing revocable) or `Closed`.
    fn map(&self, va: Va, kind: Kind) -> Result<Vec<Chunk>>;
    /// Unmaps chunks the arenas are done with (whole free, their epochs passed) and gives them back.
    fn unmap(&self, va: Va, chunks: Vec<Chunk>);
    /// A hint: the chunks went whole free (`idle`) or back into use. Atomics only: it is called
    /// under an arena lock.
    fn idle(&self, _chunks: &[Chunk], _idle: bool) {}
}

/// A block of a request's arena still live at the request's end.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Outlived {
    pub ptr: u64,
    pub bytes: u64,
    pub stream: usize,
}

/// An arena's record at its end.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ended {
    pub allocated: u64,
    pub peak: u64,
    pub outlived: Vec<Outlived>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArenaStats {
    /// Bytes of live blocks as placed (a slot its class, a run its 512 B rounding).
    pub allocated: u64,
    pub peak: u64,
    /// Bytes of the spans it holds.
    pub mapped: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub allocated: u64,
    pub peak: u64,
    /// Every span mapped: the arenas' and the idle list's.
    pub mapped: u64,
    pub idle: u64,
    pub arenas: Vec<(Owner, ArenaStats)>,
}

/// A stream's epochs. An event numbered `n` is recorded only after `issued` reads `n`, so a free
/// that read `issued` before needs `issued + 1`: that event covers everything enqueued before it.
struct Clock {
    stream: usize,
    issued: AtomicU64,
    done: AtomicU64,
    wanted: AtomicU64,
    /// Recording and its fallback both failed: what waits on it never leaves.
    lost: AtomicBool,
    flight: Mutex<VecDeque<(u64, Event)>>,
}

impl Clock {
    fn passed(&self, need: u64) -> bool {
        self.done.load(Acquire) >= need
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Tier {
    Small,
    Large,
    Pool,
    Own,
}

struct Span {
    len: u64,
    chunks: Vec<Chunk>,
    /// Its stream's; `None`: a pool span no stream has used yet ([`ANY`]).
    clock: Option<Arc<Clock>>,
    tier: Tier,
    /// Blocks carved from it (a slot page counts once).
    live: u32,
}

impl Span {
    fn stream(&self) -> usize {
        self.clock.as_ref().map_or(ANY, |c| c.stream)
    }
}

#[derive(Clone, Copy)]
struct Block {
    len: u64,
    span: u64,
    free: bool,
}

struct Page {
    class: usize,
    len: u64,
    slots: u32,
    free: u128,
    stream: usize,
}

/// A freed block waiting for other streams' epochs (record_stream), or for a capture's end.
struct Deferred {
    ptr: u64,
    bytes: u64,
    uses: Vec<usize>,
    waits: Vec<(Arc<Clock>, u64)>,
    capture: bool,
}

#[derive(Clone)]
enum Fence {
    Event(usize, Event),
    Epoch(Arc<Clock>, u64),
}

enum Freed {
    Unknown,
    Twice,
    Kept,
    Whole(u64),
}

#[derive(Default)]
struct State {
    spans: HashMap<u64, Span>,
    blocks: BTreeMap<u64, Block>,
    /// Free runs: (stream, tier, length, address), so best fit is one range query.
    free: BTreeSet<(usize, Tier, u64, u64)>,
    pages: BTreeMap<u64, Page>,
    /// (stream, class) → pages with a free slot.
    open: HashMap<(usize, usize), Vec<u64>>,
    /// Block → the other streams torch recorded it on.
    uses: HashMap<u64, Vec<usize>>,
    deferred: Vec<Deferred>,
    /// Whole-free spans kept while a capture is open: they leave once it ends.
    held: Vec<u64>,
    allocated: u64,
    peak: u64,
    mapped: u64,
    /// A pool: captures begun less graphs released (torch's use count).
    users: u32,
    /// A pool released with no fence: its epoch fallback is stamped once no capture is open.
    unfenced: bool,
}

struct Arena {
    owner: Owner,
    ended: AtomicBool,
    queued: AtomicBool,
    /// A pool whose use count reached 0.
    released: AtomicBool,
    /// Every fence of a released pool completed.
    fenced: AtomicBool,
    fences: Mutex<Vec<Fence>>,
    st: Mutex<State>,
}

impl Arena {
    fn new(owner: Owner) -> Arc<Arena> {
        Arc::new(Arena {
            owner,
            ended: AtomicBool::new(false),
            queued: AtomicBool::new(false),
            released: AtomicBool::new(false),
            fenced: AtomicBool::new(false),
            fences: Mutex::new(Vec::new()),
            st: Mutex::new(State::default()),
        })
    }

    fn pool(&self) -> bool {
        matches!(self.owner, Owner::Pool(_))
    }
}

fn full(slots: u32) -> u128 {
    if slots >= 128 { u128::MAX } else { (1u128 << slots) - 1 }
}

fn page_len(class: u64) -> u64 {
    (class * 16).max(SMALL)
}

fn class_of(size: u64) -> Option<usize> {
    CLASSES.iter().position(|&c| c >= size)
}

fn tier(pool: bool, size: u64) -> Tier {
    match size {
        _ if pool => Tier::Pool,
        s if s <= MEDIUM => Tier::Small,
        s if s < OWN => Tier::Large,
        _ => Tier::Own,
    }
}

/// The least span that serves `size`, and the span a new one maps.
fn span_len(t: Tier, size: u64) -> (u64, u64) {
    let exact = size.div_ceil(UNIT) * UNIT;
    match t {
        Tier::Small => (UNIT, UNIT),
        Tier::Large => (exact, exact.max(CHUNK)),
        Tier::Pool | Tier::Own => (exact, exact),
    }
}

fn exhausted(need: u64, why: &str) -> Error {
    Error::Exhausted(Box::new(Exhausted { need, site: Some(format!("torch: {why}")), stats: crate::Stats::default(), foreign: None }))
}

impl State {
    /// Best fit among `stream`'s free runs of `tier`; the rest of the run stays free.
    fn fit(&mut self, stream: usize, tier: Tier, len: u64) -> Option<u64> {
        let key = *self.free.range((stream, tier, len, 0)..=(stream, tier, u64::MAX, u64::MAX)).next()?;
        self.free.remove(&key);
        let (_, _, have, at) = key;
        let span = self.blocks.get(&at)?.span;
        if have > len {
            self.blocks.insert(at + len, Block { len: have - len, span, free: true });
            self.free.insert((stream, tier, have - len, at + len));
        }
        self.blocks.insert(at, Block { len, span, free: false });
        self.spans.get_mut(&span)?.live += 1;
        Some(at)
    }

    /// Frees a run block, merging it with free neighbours of its span.
    fn unfit(&mut self, at: u64) -> Freed {
        let Some(b) = self.blocks.get(&at).copied() else { return Freed::Unknown };
        if b.free {
            return Freed::Twice;
        }
        let Some(span) = self.spans.get_mut(&b.span) else { return Freed::Unknown };
        span.live = span.live.saturating_sub(1);
        let (stream, tier, whole) = (span.stream(), span.tier, span.live == 0);
        let (mut base, mut len) = (at, b.len);
        if let Some(next) = self.blocks.get(&(at + b.len)).copied().filter(|n| n.free && n.span == b.span) {
            self.free.remove(&(stream, tier, next.len, at + b.len));
            self.blocks.remove(&(at + b.len));
            len += next.len;
        }
        if let Some((&pa, prev)) = self.blocks.range(..at).next_back() {
            if prev.free && prev.span == b.span && pa + prev.len == at {
                let plen = prev.len;
                self.free.remove(&(stream, tier, plen, pa));
                self.blocks.remove(&at);
                (base, len) = (pa, len + plen);
            }
        }
        self.blocks.insert(base, Block { len, span: b.span, free: true });
        self.free.insert((stream, tier, len, base));
        if whole { Freed::Whole(b.span) } else { Freed::Kept }
    }

    /// A slot of `class` on `stream`, a new page carved for it when none has one free.
    fn slot(&mut self, stream: usize, class: usize) -> Option<u64> {
        let size = *CLASSES.get(class)?;
        let at = match self.open.get(&(stream, class)).and_then(|v| v.last().copied()) {
            Some(at) => at,
            None => {
                let len = page_len(size);
                let at = self.fit(stream, Tier::Small, len)?;
                let slots = (len / size) as u32;
                self.pages.insert(at, Page { class, len, slots, free: full(slots), stream });
                self.open.entry((stream, class)).or_default().push(at);
                at
            }
        };
        let page = self.pages.get_mut(&at)?;
        let i = page.free.trailing_zeros();
        page.free &= !(1u128 << i);
        if page.free == 0 {
            if let Some(v) = self.open.get_mut(&(stream, class)) {
                v.pop();
            }
        }
        Some(at + u64::from(i) * size)
    }

    /// Frees a slot: `None` when `ptr` is no slot. An empty page goes back to its run.
    fn unslot(&mut self, ptr: u64) -> Option<(Freed, u64)> {
        let (&at, page) = self.pages.range_mut(..=ptr).next_back()?;
        let size = *CLASSES.get(page.class)?;
        let off = ptr - at;
        if ptr >= at + page.len || !off.is_multiple_of(size) || off / size >= u64::from(page.slots) {
            return None;
        }
        let bit = 1u128 << (off / size);
        if page.free & bit != 0 {
            return Some((Freed::Twice, 0));
        }
        let was_full = page.free == 0;
        page.free |= bit;
        let (key, empty) = ((page.stream, page.class), page.free == full(page.slots));
        if empty {
            self.pages.remove(&at);
            if let Some(v) = self.open.get_mut(&key) {
                v.retain(|p| *p != at);
            }
            return Some((self.unfit(at), size));
        }
        if was_full {
            self.open.entry(key).or_default().push(at);
        }
        Some((Freed::Kept, size))
    }

    /// The stream a live block's span serves.
    fn stream_of(&self, ptr: u64) -> Option<usize> {
        if let Some((&at, page)) = self.pages.range(..=ptr).next_back() {
            if ptr < at + page.len {
                return Some(page.stream);
            }
        }
        let b = self.blocks.get(&ptr).filter(|b| !b.free)?;
        self.spans.get(&b.span).map(Span::stream)
    }

    /// Every live block: runs (pages aside) and slots.
    fn live(&self) -> Vec<Outlived> {
        let mut out: Vec<Outlived> = self
            .blocks
            .iter()
            .filter(|(at, b)| !b.free && !self.pages.contains_key(*at))
            .map(|(&ptr, b)| Outlived { ptr, bytes: b.len, stream: self.spans.get(&b.span).map_or(ANY, Span::stream) })
            .collect();
        for (&at, p) in &self.pages {
            let size = CLASSES.get(p.class).copied().unwrap_or(0);
            out.extend((0..p.slots).filter(|i| p.free & (1u128 << i) == 0).map(|i| Outlived { ptr: at + u64::from(i) * size, bytes: size, stream: p.stream }));
        }
        out
    }

    fn free_bytes(&self) -> u64 {
        self.free.iter().map(|k| k.2).sum()
    }
}

/// A whole-free span on the idle list.
struct Spare {
    base: u64,
    chunks: Vec<Chunk>,
    /// Its last stream and the epoch it waits for; `None`: quiet now.
    clock: Option<Arc<Clock>>,
    need: u64,
    tier: Tier,
    seq: u64,
}

impl Spare {
    fn len(&self) -> u64 {
        self.chunks.iter().map(|c| c.len).sum()
    }

    fn stream(&self) -> usize {
        self.clock.as_ref().map_or(ANY, |c| c.stream)
    }

    fn quiet(&self) -> bool {
        self.clock.as_ref().is_none_or(|c| c.passed(self.need))
    }

    fn lost(&self) -> bool {
        self.clock.as_ref().is_some_and(|c| c.lost.load(Acquire))
    }
}

#[derive(Default)]
struct Idle {
    spares: BTreeMap<(u64, u64), Spare>,
    seq: u64,
}

impl Idle {
    fn put(&mut self, mut s: Spare) {
        self.seq += 1;
        s.seq = self.seq;
        self.spares.insert((s.len(), s.base), s);
    }

    /// A spare `stream` may take (its own stream's, or a quiet one): the least of `tier` holding
    /// `new` bytes (what a new span would map, so a warm request replays its spans), else the least
    /// of any tier holding `want`. Chunks beyond `want` stay idle as a spare of their own.
    fn take(&mut self, (want, new): (u64, u64), tier: Tier, stream: usize, quiet_only: bool) -> Option<Spare> {
        let ok = |s: &Spare| s.quiet() || (!quiet_only && stream != ANY && s.stream() == stream);
        let key = self.spares.range((new, 0)..).find(|(_, s)| s.tier == tier && ok(s)).map(|(k, _)| *k);
        let key = key.or_else(|| self.spares.range((want, 0)..).find(|(_, s)| ok(s)).map(|(k, _)| *k))?;
        let mut s = self.spares.remove(&key)?;
        let (mut keep, mut n) = (0, 0);
        for c in &s.chunks {
            if keep >= want {
                break;
            }
            keep += c.len;
            n += 1;
        }
        if n < s.chunks.len() {
            let rest = s.chunks.split_off(n);
            self.put(Spare { base: s.base + keep, chunks: rest, clock: s.clock.clone(), need: s.need, tier: s.tier, seq: 0 });
        }
        Some(s)
    }

    /// The oldest quiet spares, at least `want` bytes of them when there are.
    fn quiet(&mut self, want: u64) -> Vec<Spare> {
        let mut keys: Vec<(u64, (u64, u64))> = self.spares.iter().filter(|(_, s)| s.quiet()).map(|(k, s)| (s.seq, *k)).collect();
        keys.sort_unstable();
        let (mut out, mut got) = (Vec::new(), 0);
        for (_, k) in keys {
            if got >= want {
                break;
            }
            if let Some(s) = self.spares.remove(&k) {
                got += s.len();
                out.push(s);
            }
        }
        out
    }

    /// (quiet, pending) bytes; a spare on a lost stream is neither.
    fn bytes(&self) -> (u64, u64) {
        self.spares.values().fold((0, 0), |(q, p), s| match () {
            _ if s.quiet() => (q + s.len(), p),
            _ if s.lost() => (q, p),
            _ => (q, p + s.len()),
        })
    }
}

/// The torch address space: reservations and their free ranges.
#[derive(Default)]
struct Space {
    reserved: Vec<Va>,
    by_len: BTreeSet<(u64, u64)>,
    at: BTreeMap<u64, u64>,
}

impl Space {
    fn add(&mut self, va: Va) {
        self.reserved.push(va);
        self.insert(va.base, va.len);
    }

    fn insert(&mut self, base: u64, len: u64) {
        self.by_len.insert((len, base));
        self.at.insert(base, len);
    }

    fn remove(&mut self, base: u64, len: u64) {
        self.by_len.remove(&(len, base));
        self.at.remove(&base);
    }

    fn take(&mut self, len: u64) -> Option<u64> {
        let (have, base) = *self.by_len.range((len, 0)..).next()?;
        self.remove(base, have);
        if have > len {
            self.insert(base + len, have - len);
        }
        Some(base)
    }

    /// Which reservation `at` is in: ranges merge only within one.
    fn home(&self, at: u64) -> Option<usize> {
        self.reserved.iter().position(|v| at >= v.base && at < v.base + v.len)
    }

    fn give(&mut self, mut base: u64, mut len: u64) {
        let home = self.home(base);
        if let Some(&next) = self.at.get(&(base + len)) {
            if self.home(base + len) == home {
                self.remove(base + len, next);
                len += next;
            }
        }
        if let Some((&pb, &pl)) = self.at.range(..base).next_back() {
            if pb + pl == base && self.home(pb) == home {
                self.remove(pb, pl);
                (base, len) = (pb, len + pl);
            }
        }
        self.insert(base, len);
    }
}

/// A thread's route into an arena: dropped on any thread, it stops routing.
struct RouteTo {
    arena: Arc<Arena>,
    live: AtomicBool,
}

/// While it lives, its thread's torch blocks go to its arena ([`Torch::route`]).
pub struct Route(Arc<RouteTo>);

impl Drop for Route {
    fn drop(&mut self) {
        self.0.live.store(false, Release);
    }
}

thread_local! {
    static ROUTES: RefCell<Vec<(usize, Arc<RouteTo>)>> = const { RefCell::new(Vec::new()) };
}

static TORCHES: AtomicUsize = AtomicUsize::new(1);

struct Shared {
    id: usize,
    backend: Arc<dyn Backend>,
    source: Arc<dyn Source>,
    space: Mutex<Space>,
    /// Span base → (length, its arena): where a free goes.
    dir: RwLock<BTreeMap<u64, (u64, Arc<Arena>)>>,
    idle: Mutex<Idle>,
    arenas: Mutex<HashMap<Owner, Arc<Arena>>>,
    process: Arc<Arena>,
    clocks: RwLock<HashMap<usize, Arc<Clock>>>,
    /// The poller records under its read side, a capture begins and ends under its write side:
    /// nothing is recorded on a stream once its capture began. No arena lock is held with it.
    gate: RwLock<()>,
    /// Open captures. A stamp reads it before and after (R2): one a capture's begin could not
    /// record is held instead.
    captures: AtomicU32,
    capturers: Mutex<Vec<(ThreadId, PoolId)>>,
    /// Arenas with deferred frees, held spans or a released pool, for the poller.
    pending: Mutex<Vec<Arc<Arena>>>,
    wake: (Mutex<bool>, Condvar),
    /// Bumped at every epoch, fence or hand-off the poller sees land.
    progress: (Mutex<u64>, Condvar),
    allocated: AtomicU64,
    peak: AtomicU64,
    mapped: AtomicU64,
    stop: AtomicBool,
}

/// One device's torch allocator: every arena, the idle list and the epoch poller.
pub struct Torch {
    sh: Arc<Shared>,
}

impl Torch {
    pub fn new(backend: Arc<dyn Backend>, source: Arc<dyn Source>) -> Result<Arc<Torch>> {
        if !UNIT.is_multiple_of(backend.granularity().max(1)) {
            return Err(Error::Invalid(format!("granularity {} does not divide {UNIT}", backend.granularity())));
        }
        let first = backend.reserve((2 * backend.capacity()).max(4 * CHUNK).div_ceil(CHUNK) * CHUNK)?;
        let mut space = Space::default();
        space.add(first);
        let sh = Arc::new(Shared {
            id: TORCHES.fetch_add(1, AcqRel),
            backend,
            source,
            space: Mutex::new(space),
            dir: RwLock::new(BTreeMap::new()),
            idle: Mutex::new(Idle::default()),
            arenas: Mutex::new(HashMap::new()),
            process: Arena::new(Owner::Process),
            clocks: RwLock::new(HashMap::new()),
            gate: RwLock::new(()),
            captures: AtomicU32::new(0),
            capturers: Mutex::new(Vec::new()),
            pending: Mutex::new(Vec::new()),
            wake: (Mutex::new(false), Condvar::new()),
            progress: (Mutex::new(0), Condvar::new()),
            allocated: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            mapped: AtomicU64::new(0),
            stop: AtomicBool::new(false),
        });
        let weak = Arc::downgrade(&sh);
        std::thread::Builder::new().name("varena-epochs".into()).spawn(move || poll(&weak)).map_err(|e| Error::io("spawn the epoch poller", e))?;
        Ok(Arc::new(Torch { sh }))
    }

    /// A torch block on `stream`: in `pool`'s arena while a capture routes there, else the thread's
    /// routed arena, else the process's. Err is `Exhausted`, whose text is torch's OOM reason.
    pub fn alloc(&self, bytes: u64, stream: usize, pool: Option<PoolId>) -> Result<u64> {
        let size = bytes.max(1).checked_next_multiple_of(ALIGN).filter(|s| *s < 1 << 50).ok_or_else(|| exhausted(bytes, "no device holds that much"))?;
        let sh = &self.sh;
        match pool {
            Some(p) => sh.alloc_in(&sh.arena(Owner::Pool(p)), size, stream, true),
            None => sh.alloc_in(&sh.routed(), size, stream, sh.capturing()),
        }
    }

    /// Never fails and never waits on the device; a pointer no arena holds is logged.
    pub fn free(&self, ptr: u64, bytes: u64, stream: usize) {
        let Some(a) = self.sh.owner_of(ptr) else {
            tracing::error!("varena: torch freed {ptr:#x} ({bytes} bytes, stream {stream:#x}), which no arena holds");
            return;
        };
        let mut st = a.st.lock();
        self.sh.free_in(&a, &mut st, ptr, bytes);
    }

    /// torch used `ptr` on `stream` too: its free waits for that stream's epoch. A pointer that is
    /// not this device's is ignored.
    pub fn record_stream(&self, ptr: u64, stream: usize) {
        let Some(a) = self.sh.owner_of(ptr) else { return };
        let mut st = a.st.lock();
        if st.stream_of(ptr).is_some_and(|own| own != stream) {
            self.sh.clock(stream);
            let uses = st.uses.entry(ptr).or_default();
            if !uses.contains(&stream) {
                uses.push(stream);
            }
        }
    }

    /// torch's begin-allocate-to-pool hook, on the capturing thread before its capture starts:
    /// one more use of `pool`, and this thread is served from mapped free bytes until its end. Every
    /// epoch pending is recorded now, while nothing captures yet.
    pub fn pool_begin(&self, pool: PoolId) {
        let sh = &self.sh;
        let me = std::thread::current().id();
        let clocks: Vec<Arc<Clock>> = sh.clocks.read().values().cloned().collect();
        {
            let _gate = sh.gate.write();
            let mut caps = sh.capturers.lock();
            // A capture this thread left open died (torch never ended it): this one replaces it.
            caps.retain(|(t, p)| !(*t == me && *p == pool));
            caps.push((me, pool));
            sh.captures.store(caps.len() as u32, SeqCst);
            drop(caps);
            clocks.iter().for_each(|c| sh.record(c));
        }
        let a = sh.arena(Owner::Pool(pool));
        let mut st = a.st.lock();
        st.users += 1;
        sh.revive(&a, &mut st);
    }

    /// torch's end-allocate-to-pool hook, after the capture ended: frees deferred by it apply once
    /// no capture is open. Again, or for a capture never begun: nothing.
    pub fn pool_end(&self, pool: PoolId) {
        let sh = &self.sh;
        let me = std::thread::current().id();
        let gate = sh.gate.write();
        let mut caps = sh.capturers.lock();
        let at = caps.iter().position(|(t, p)| *t == me && *p == pool).or_else(|| caps.iter().position(|(_, p)| *p == pool));
        if let Some(i) = at {
            caps.remove(i);
        }
        sh.captures.store(caps.len() as u32, SeqCst);
        drop((caps, gate));
        sh.kick();
    }

    /// torch's release-pool hook (a graph's reset): one use fewer. At 0, the pool's spans leave once
    /// every fence completed (R4/R5); with none given, after an epoch on every stream seen.
    pub fn pool_release(&self, pool: PoolId) {
        let Some(a) = self.sh.arenas.lock().get(&Owner::Pool(pool)).cloned() else { return };
        let mut st = a.st.lock();
        st.users = st.users.saturating_sub(1);
        if st.users > 0 || a.released.load(Acquire) {
            return;
        }
        a.released.store(true, Release);
        st.unfenced = a.fences.lock().is_empty();
        self.sh.queue(&a, true);
    }

    /// An event on `stream` after the pool's use there (a replay, and its readers): a released
    /// pool's spans leave only once every fence completed. Not from a capturing thread.
    pub fn pool_fence(&self, pool: PoolId, stream: usize) -> Result<()> {
        let sh = &self.sh;
        let a = sh.arena(Owner::Pool(pool));
        let fence = match sh.backend.record(stream) {
            Ok(event) => Fence::Event(stream, event),
            Err(e) => {
                tracing::warn!("varena: recording a pool fence failed ({e}); synchronizing its stream");
                sh.backend.synchronize(stream)?;
                return Ok(());
            }
        };
        let mut fences = a.fences.lock();
        fences.retain(|f| match f {
            Fence::Event(s, e) if *s == stream => {
                sh.backend.retire(*e);
                false
            }
            _ => true,
        });
        fences.push(fence);
        a.fenced.store(false, Release);
        Ok(())
    }

    /// Before a capture into `pool` begins (never on a capturing thread): `expected` bytes of it
    /// mapped and free, made room for here.
    pub fn pool_prepare(&self, pool: PoolId, expected: u64) -> Result<()> {
        let sh = &self.sh;
        if sh.capturing() {
            return Err(Error::Invalid("a pool is prepared before its capture begins".into()));
        }
        let a = sh.arena(Owner::Pool(pool));
        loop {
            let short = {
                let mut st = a.st.lock();
                sh.revive(&a, &mut st);
                let short = expected.saturating_sub(st.free_bytes());
                if short == 0 {
                    return Ok(());
                }
                let want = short.div_ceil(UNIT) * UNIT;
                let spare = sh.idle.lock().take((want, want), Tier::Pool, ANY, true);
                if let Some(s) = spare {
                    sh.source.idle(&s.chunks, false);
                    sh.install(&a, &mut st, s.base, s.chunks, None, Tier::Pool);
                    continue;
                }
                want
            };
            let va = sh.reserve(short)?;
            match sh.source.map(va, Kind::Pool) {
                Ok(chunks) => {
                    sh.mapped.fetch_add(va.len, AcqRel);
                    let mut st = a.st.lock();
                    sh.install(&a, &mut st, va.base, chunks, None, Tier::Pool);
                }
                Err(e) => {
                    sh.space.lock().give(va.base, va.len);
                    if !(matches!(e, Error::Exhausted(_)) && sh.give_back()) {
                        return Err(e);
                    }
                }
            }
        }
    }

    /// Routes this thread's blocks to `owner`'s arena (made at first use) until the route drops.
    pub fn route(&self, owner: Owner) -> Route {
        let arena = match owner {
            Owner::Process => self.sh.process.clone(),
            o => self.sh.arena(o),
        };
        let to = Arc::new(RouteTo { arena, live: AtomicBool::new(true) });
        let id = self.sh.id;
        let _ = ROUTES.try_with(|r| r.borrow_mut().push((id, to.clone())));
        Route(to)
    }

    /// Ends a request's or a region's arena: nothing routes there again, and its spans leave as they
    /// come whole free. The record names every block still live (the request-end audit).
    pub fn end(&self, owner: Owner) -> Ended {
        if owner == Owner::Process {
            return Ended::default();
        }
        let Some(a) = self.sh.arenas.lock().remove(&owner) else { return Ended::default() };
        a.ended.store(true, Release);
        let st = a.st.lock();
        Ended { allocated: st.allocated, peak: st.peak, outlived: st.live() }
    }

    pub fn stats(&self) -> Stats {
        let sh = &self.sh;
        let mut arenas: Vec<Arc<Arena>> = sh.arenas.lock().values().cloned().collect();
        arenas.push(sh.process.clone());
        let arenas = arenas
            .into_iter()
            .map(|a| {
                let st = a.st.lock();
                (a.owner, ArenaStats { allocated: st.allocated, peak: st.peak, mapped: st.mapped })
            })
            .collect();
        let (quiet, pending) = sh.idle.lock().bytes();
        Stats { allocated: sh.allocated.load(Acquire), peak: sh.peak.load(Acquire), mapped: sh.mapped.load(Acquire), idle: quiet + pending, arenas }
    }

    /// An arena's peak since the last reset (a pass's compute peak); it restarts at what is live.
    pub fn reset_peak(&self, owner: Owner) -> u64 {
        let a = match owner {
            Owner::Process => Some(self.sh.process.clone()),
            o => self.sh.arenas.lock().get(&o).cloned(),
        };
        a.map_or(0, |a| {
            let mut st = a.st.lock();
            let now = st.allocated;
            std::mem::replace(&mut st.peak, now)
        })
    }

    /// Gives up to `want` bytes of quiet idle spans back through [`Source::unmap`], oldest first;
    /// the bytes given. The agent's revoke path: it takes no arena lock and needs no other.
    pub fn reclaim(&self, want: u64) -> u64 {
        self.sh.reclaim(want)
    }

    /// Idle bytes: (quiet, waiting for their stream's epoch).
    pub fn idle(&self) -> (u64, u64) {
        self.sh.idle.lock().bytes()
    }
}

impl Drop for Torch {
    fn drop(&mut self) {
        let sh = &self.sh;
        sh.stop.store(true, Release);
        sh.kick();
        sh.reclaim(u64::MAX);
        let mapped = sh.mapped.load(Acquire);
        if mapped == 0 {
            std::mem::take(&mut sh.space.lock().reserved).into_iter().for_each(|va| sh.backend.release(va));
        } else {
            tracing::warn!("varena: torch's arenas still map {mapped} bytes at close: their address space stays reserved");
        }
    }
}

impl Shared {
    fn kick(&self) {
        let mut w = self.wake.0.lock();
        if !*w {
            *w = true;
            self.wake.1.notify_one();
        }
    }

    fn progressed(&self) {
        *self.progress.0.lock() += 1;
        self.progress.1.notify_all();
    }

    fn capturing(&self) -> bool {
        if self.captures.load(Acquire) == 0 {
            return false;
        }
        let me = std::thread::current().id();
        self.capturers.lock().iter().any(|(t, _)| *t == me)
    }

    fn routed(&self) -> Arc<Arena> {
        ROUTES
            .try_with(|r| {
                let mut r = r.borrow_mut();
                while r.last().is_some_and(|(_, to)| !to.live.load(Acquire)) {
                    r.pop();
                }
                r.iter().rev().find(|(id, to)| *id == self.id && to.live.load(Acquire) && !to.arena.ended.load(Acquire)).map(|(_, to)| to.arena.clone())
            })
            .ok()
            .flatten()
            .unwrap_or_else(|| self.process.clone())
    }

    fn arena(&self, owner: Owner) -> Arc<Arena> {
        if owner == Owner::Process {
            return self.process.clone();
        }
        self.arenas.lock().entry(owner).or_insert_with(|| Arena::new(owner)).clone()
    }

    fn owner_of(&self, ptr: u64) -> Option<Arc<Arena>> {
        let dir = self.dir.read();
        let (&base, (len, a)) = dir.range(..=ptr).next_back()?;
        (ptr < base + len).then(|| a.clone())
    }

    fn clock(&self, stream: usize) -> Arc<Clock> {
        if let Some(c) = self.clocks.read().get(&stream) {
            return c.clone();
        }
        self.clocks
            .write()
            .entry(stream)
            .or_insert_with(|| {
                Arc::new(Clock { stream, issued: AtomicU64::new(0), done: AtomicU64::new(0), wanted: AtomicU64::new(0), lost: AtomicBool::new(false), flight: Mutex::new(VecDeque::new()) })
            })
            .clone()
    }

    /// For each of `streams`, the epoch covering everything enqueued on it so far. `None` while a
    /// capture is open: one that opened meanwhile may have recorded before these were wanted, so
    /// they would wait for its end (they stay wanted, and harmless).
    fn stamps(&self, streams: &[usize]) -> Option<Vec<(Arc<Clock>, u64)>> {
        if self.captures.load(SeqCst) > 0 {
            return None;
        }
        let stamped = streams
            .iter()
            .map(|s| {
                let c = self.clock(*s);
                let n = c.issued.load(SeqCst) + 1;
                c.wanted.fetch_max(n, SeqCst);
                (c, n)
            })
            .collect();
        self.kick();
        (self.captures.load(SeqCst) == 0).then_some(stamped)
    }

    /// Records `c`'s wanted epoch. The caller holds the gate: no capture is open, or one is about to
    /// begin and none has started.
    fn record(&self, c: &Clock) {
        let n = c.issued.load(SeqCst) + 1;
        if c.lost.load(Acquire) || c.wanted.load(SeqCst) < n {
            return;
        }
        // Announced before the event is recorded: a stamp that read the old value is covered.
        c.issued.store(n, SeqCst);
        match self.backend.record(c.stream) {
            Ok(event) => c.flight.lock().push_back((n, event)),
            Err(e) => self.fallback(c, n, &e),
        }
    }

    /// The poller's: events of `c` that completed.
    fn query(&self, c: &Clock) -> bool {
        let mut progress = false;
        loop {
            let Some((n, event)) = c.flight.lock().front().copied() else { break };
            let done = self.backend.query(event);
            if matches!(done, Ok(false)) {
                break;
            }
            c.flight.lock().pop_front();
            self.backend.retire(event);
            match done {
                Ok(_) => {
                    c.done.fetch_max(n, AcqRel);
                    progress = true;
                }
                Err(e) => {
                    self.fallback(c, n, &e);
                    progress = true;
                }
            }
        }
        progress
    }

    /// An epoch whose event failed: the stream is synchronized instead (on the poller or a capture's
    /// begin, never under an arena lock); failing that too, what waits on it never leaves.
    fn fallback(&self, c: &Clock, n: u64, why: &Error) {
        match self.backend.synchronize(c.stream) {
            Ok(()) => {
                c.done.fetch_max(n, AcqRel);
            }
            Err(e) => {
                c.lost.store(true, Release);
                tracing::error!("varena: stream {:#x}'s epoch failed ({why}) and so did its synchronize ({e}): its spans are quarantined", c.stream);
            }
        }
    }

    fn queue(&self, a: &Arc<Arena>, kick: bool) {
        if !a.queued.swap(true, AcqRel) {
            self.pending.lock().push(a.clone());
        }
        if kick {
            self.kick();
        }
    }

    fn count(&self, st: &mut State, n: u64) {
        st.allocated += n;
        st.peak = st.peak.max(st.allocated);
        let now = self.allocated.fetch_add(n, AcqRel) + n;
        self.peak.fetch_max(now, AcqRel);
    }

    fn uncount(&self, st: &mut State, n: u64) {
        st.allocated = st.allocated.saturating_sub(n);
        self.allocated.fetch_sub(n.min(self.allocated.load(Acquire)), AcqRel);
    }

    /// A block from the arena's free bytes, no driver call.
    fn carve(&self, a: &Arena, st: &mut State, size: u64, stream: usize) -> Option<u64> {
        let pool = a.pool();
        let (at, placed) = match tier(pool, size) {
            Tier::Small if size <= SMALL => {
                let class = class_of(size)?;
                (st.slot(stream, class)?, *CLASSES.get(class)?)
            }
            Tier::Pool => (st.fit(stream, Tier::Pool, size).or_else(|| self.claim(st, stream, size))?, size),
            t => (st.fit(stream, t, size)?, size),
        };
        self.count(st, placed);
        Some(at)
    }

    /// A prepared pool span no stream used yet becomes `stream`'s.
    fn claim(&self, st: &mut State, stream: usize, size: u64) -> Option<u64> {
        let key = *st.free.range((ANY, Tier::Pool, size, 0)..=(ANY, Tier::Pool, u64::MAX, u64::MAX)).next()?;
        let (_, _, len, base) = key;
        st.free.remove(&key);
        st.free.insert((stream, Tier::Pool, len, base));
        st.spans.get_mut(&base)?.clock = Some(self.clock(stream));
        st.fit(stream, Tier::Pool, size)
    }

    fn install(&self, a: &Arc<Arena>, st: &mut State, base: u64, chunks: Vec<Chunk>, clock: Option<Arc<Clock>>, tier: Tier) {
        let len = chunks.iter().map(|c| c.len).sum();
        let stream = clock.as_ref().map_or(ANY, |c| c.stream);
        st.spans.insert(base, Span { len, chunks, clock, tier, live: 0 });
        st.blocks.insert(base, Block { len, span: base, free: true });
        st.free.insert((stream, tier, len, base));
        st.mapped += len;
        self.dir.write().insert(base, (len, a.clone()));
    }

    fn alloc_in(&self, a: &Arc<Arena>, size: u64, stream: usize, capture: bool) -> Result<u64> {
        let t = tier(a.pool(), size);
        let (least, new) = span_len(t, size);
        loop {
            {
                let mut st = a.st.lock();
                if let Some(ptr) = self.carve(a, &mut st, size, stream) {
                    return Ok(ptr);
                }
                // A pool's span is written by replays on any stream: only a quiet spare will do.
                let spare = self.idle.lock().take((least, new), t, stream, a.pool());
                if let Some(s) = spare {
                    self.source.idle(&s.chunks, false);
                    let clock = (stream != ANY).then(|| self.clock(stream));
                    self.install(a, &mut st, s.base, s.chunks, clock, t);
                    continue;
                }
            }
            if capture {
                return Err(exhausted(size, "a capture is served from mapped free bytes only, and its arena has none that fit (R6)"));
            }
            let va = self.reserve(new)?;
            match self.source.map(va, if a.pool() { Kind::Pool } else { Kind::Torch }) {
                Ok(chunks) => {
                    self.mapped.fetch_add(va.len, AcqRel);
                    let mut st = a.st.lock();
                    self.install(a, &mut st, va.base, chunks, Some(self.clock(stream)), t);
                    if let Some(ptr) = self.carve(a, &mut st, size, stream) {
                        return Ok(ptr);
                    }
                }
                Err(e) => {
                    self.space.lock().give(va.base, va.len);
                    if !(matches!(e, Error::Exhausted(_)) && self.give_back()) {
                        return Err(e);
                    }
                }
            }
        }
    }

    /// The source is short: this process's idle spans go back to it. Pending ones are waited for
    /// (the poller, a Rust thread, sees them land); false when none is left to give.
    fn give_back(&self) -> bool {
        loop {
            let seen = *self.progress.0.lock();
            let (quiet, pending) = self.idle.lock().bytes();
            if quiet > 0 {
                // Given, or adopted meanwhile by another arena: either way the source is asked again.
                self.reclaim(u64::MAX);
                return true;
            }
            if pending == 0 {
                return false;
            }
            self.kick();
            let mut p = self.progress.0.lock();
            while *p == seen {
                self.progress.1.wait(&mut p);
            }
        }
    }

    fn reclaim(&self, want: u64) -> u64 {
        let spares = self.idle.lock().quiet(want);
        let mut given = 0;
        for s in spares {
            let len = s.len();
            self.source.unmap(Va { base: s.base, len }, s.chunks);
            self.space.lock().give(s.base, len);
            self.mapped.fetch_sub(len.min(self.mapped.load(Acquire)), AcqRel);
            given += len;
        }
        given
    }

    fn reserve(&self, len: u64) -> Result<Va> {
        if let Some(base) = self.space.lock().take(len) {
            return Ok(Va { base, len });
        }
        let va = self.backend.reserve(len.max(self.backend.capacity()).div_ceil(CHUNK) * CHUNK)?;
        let mut space = self.space.lock();
        space.add(va);
        space.take(len).map(|base| Va { base, len }).ok_or_else(|| Error::Internal("a new reservation did not fit its span".into()))
    }

    fn free_in(&self, a: &Arc<Arena>, st: &mut State, ptr: u64, bytes: u64) {
        let uses = if st.uses.is_empty() { None } else { st.uses.remove(&ptr) };
        let capture = !a.pool() && self.captures.load(Acquire) > 0;
        if !capture && uses.is_none() {
            return self.release(a, st, ptr, bytes);
        }
        let mut d = Deferred { ptr, bytes, uses: uses.unwrap_or_default(), waits: Vec::new(), capture: true };
        if !capture {
            if let Some(waits) = self.stamps(&d.uses) {
                (d.waits, d.capture) = (waits, false);
            }
        }
        st.deferred.push(d);
        self.queue(a, true);
    }

    /// The block is free now.
    fn release(&self, a: &Arc<Arena>, st: &mut State, ptr: u64, bytes: u64) {
        let (freed, placed) = match st.unslot(ptr) {
            Some(f) => f,
            None => {
                let len = st.blocks.get(&ptr).map_or(0, |b| b.len);
                (st.unfit(ptr), len)
            }
        };
        match freed {
            Freed::Unknown => tracing::error!("varena: torch freed {ptr:#x} ({bytes} bytes), which its arena does not hold"),
            Freed::Twice => tracing::error!("varena: torch freed {ptr:#x} ({bytes} bytes) twice"),
            Freed::Kept => self.uncount(st, placed),
            Freed::Whole(span) => {
                self.uncount(st, placed);
                self.whole(a, st, span);
            }
        }
    }

    /// A span came whole free. A pool's stays until it is released and fenced (R4/R5); any other
    /// leaves for the idle list, stamped with its stream's epoch (R1), or is held while a capture
    /// is open.
    fn whole(&self, a: &Arc<Arena>, st: &mut State, base: u64) {
        if a.pool() {
            if a.released.load(Acquire) && a.fenced.load(Acquire) {
                self.to_idle(st, base, None);
            }
            return;
        }
        let Some(stream) = st.spans.get(&base).and_then(|s| s.clock.as_ref()).map(|c| c.stream) else { return self.to_idle(st, base, None) };
        match self.stamps(&[stream]) {
            Some(mut need) => self.to_idle(st, base, need.pop()),
            None => {
                st.held.push(base);
                self.queue(a, true);
            }
        }
    }

    /// The freeing thread hands a whole-free span to the idle list (R10): no driver call.
    fn to_idle(&self, st: &mut State, base: u64, need: Option<(Arc<Clock>, u64)>) {
        let Some(span) = st.spans.remove(&base) else { return };
        st.free.remove(&(span.stream(), span.tier, span.len, base));
        st.blocks.remove(&base);
        st.mapped = st.mapped.saturating_sub(span.len);
        self.dir.write().remove(&base);
        self.source.idle(&span.chunks, true);
        let (clock, need) = need.map_or((None, 0), |(c, n)| (Some(c), n));
        self.idle.lock().put(Spare { base, chunks: span.chunks, clock, need, tier: span.tier, seq: 0 });
    }

    /// A pool used again: no longer released or fenced.
    fn revive(&self, a: &Arena, st: &mut State) {
        if a.released.swap(false, AcqRel) {
            a.fenced.store(false, Release);
            st.unfenced = false;
            a.fences.lock().drain(..).for_each(|f| {
                if let Fence::Event(_, e) = f {
                    self.backend.retire(e);
                }
            });
        }
    }

    /// Whether every fence of a released pool completed; queried with no arena lock held.
    fn fences_done(&self, a: &Arena) -> bool {
        let fences: Vec<Fence> = a.fences.lock().clone();
        !fences.is_empty()
            && fences.iter().all(|f| match f {
                Fence::Epoch(c, n) => c.passed(*n),
                Fence::Event(stream, e) => match self.backend.query(*e) {
                    Ok(done) => done,
                    Err(why) => {
                        tracing::warn!("varena: a pool fence's query failed ({why}); synchronizing its stream");
                        self.backend.synchronize(*stream).is_ok()
                    }
                },
            })
    }

    /// The poller's pass over an arena: deferred frees, held spans and a released pool's spans.
    fn settle(&self, a: &Arc<Arena>) -> bool {
        let mut progress = false;
        if a.pool() && a.released.load(Acquire) && !a.fenced.load(Acquire) && self.fences_done(a) {
            a.fenced.store(true, Release);
            progress = true;
        }
        let mut st = a.st.lock();
        if st.unfenced && a.released.load(Acquire) {
            let streams: Vec<usize> = self.clocks.read().keys().copied().collect();
            if let Some(stamps) = self.stamps(&streams) {
                st.unfenced = false;
                let mut fences = a.fences.lock();
                fences.extend(stamps.into_iter().map(|(c, n)| Fence::Epoch(c, n)));
                if fences.is_empty() {
                    // No stream ever ran: nothing to wait for.
                    fences.push(Fence::Event(0, Event::DONE));
                }
            }
        }
        let open = self.captures.load(Acquire) > 0;
        for mut d in std::mem::take(&mut st.deferred) {
            if d.capture && !open {
                if let Some(waits) = self.stamps(&d.uses) {
                    (d.waits, d.capture) = (waits, false);
                }
            }
            if !d.capture && d.waits.iter().all(|(c, n)| c.passed(*n)) {
                self.release(a, &mut st, d.ptr, d.bytes);
                progress = true;
            } else {
                st.deferred.push(d);
            }
        }
        if !open {
            for base in std::mem::take(&mut st.held) {
                if st.spans.get(&base).is_some_and(|s| s.live == 0) {
                    self.whole(a, &mut st, base);
                    progress = true;
                }
            }
        }
        if a.pool() && a.released.load(Acquire) && a.fenced.load(Acquire) {
            let wholes: Vec<u64> = st.spans.iter().filter(|(_, s)| s.live == 0).map(|(b, _)| *b).collect();
            progress |= !wholes.is_empty();
            wholes.into_iter().for_each(|b| self.to_idle(&mut st, b, None));
        }
        let waiting = !st.deferred.is_empty() || !st.held.is_empty() || (a.released.load(Acquire) && !a.fenced.load(Acquire));
        drop(st);
        if waiting {
            self.queue(a, false);
        }
        progress
    }

    /// One round: record wanted epochs (no capture open), query those in flight, settle arenas.
    /// Returns (progress, busy).
    fn tick(&self) -> (bool, bool) {
        let clocks: Vec<Arc<Clock>> = self.clocks.read().values().cloned().collect();
        {
            let _gate = self.gate.read();
            if self.captures.load(SeqCst) == 0 {
                clocks.iter().for_each(|c| self.record(c));
            }
        }
        let mut progress = false;
        for c in &clocks {
            progress |= self.query(c);
        }
        let pending = std::mem::take(&mut *self.pending.lock());
        for a in pending {
            a.queued.store(false, Release);
            progress |= self.settle(&a);
        }
        self.backend.sweep();
        let busy = !self.pending.lock().is_empty() || clocks.iter().any(|c| !c.lost.load(Acquire) && c.wanted.load(Acquire) > c.done.load(Acquire));
        (progress, busy)
    }
}

/// The epoch poller: off every allocation's path, in relaxed capture mode so its queries never
/// disturb another thread's capture.
fn poll(weak: &Weak<Shared>) {
    if let Some(sh) = weak.upgrade() {
        sh.backend.relax();
    }
    let mut nap = NAP;
    loop {
        let Some(sh) = weak.upgrade() else { return };
        if sh.stop.load(Acquire) {
            return;
        }
        let (progress, busy) = sh.tick();
        if progress {
            sh.progressed();
            nap = NAP;
        } else {
            nap = (nap * 2).min(NAP_MAX);
        }
        let mut woke = sh.wake.0.lock();
        if !*woke && !sh.stop.load(Acquire) {
            if busy {
                sh.wake.1.wait_for(&mut woke, nap);
            } else {
                sh.wake.1.wait(&mut woke);
            }
        }
        if *woke {
            nap = NAP;
        }
        *woke = false;
    }
}

/// A standalone budget: it creates its own pages, up to `limit` bytes. With no manager, Exhausted
/// is the device's own out-of-memory, or the limit.
pub struct Direct {
    backend: Arc<dyn Backend>,
    limit: u64,
    held: Mutex<(u64, HashMap<u64, Physical>)>,
}

impl Direct {
    pub fn new(backend: Arc<dyn Backend>, limit: u64) -> Direct {
        Direct { backend, limit, held: Mutex::new((0, HashMap::new())) }
    }

    fn one(&self, va: Va, off: u64, len: u64) -> Result<Chunk> {
        {
            let mut held = self.held.lock();
            if held.0 + len > self.limit {
                return Err(exhausted(len, &format!("the standalone budget of {} bytes is spent", self.limit)));
            }
            held.0 += len;
        }
        let made = self.backend.create(len).map_err(|e| match e {
            Error::Device { code: OUT_OF_MEMORY, .. } => exhausted(len, "the device is out of memory"),
            e => e,
        });
        let page = made.and_then(|p| match self.backend.map(va, off, &p, Access::ReadWrite) {
            Ok(()) => Ok(p),
            Err(e) => {
                let _ = self.backend.destroy(p);
                Err(e)
            }
        });
        let mut held = self.held.lock();
        match page {
            Ok(p) => {
                let id = p.id;
                held.1.insert(id, p);
                Ok(Chunk { id, len })
            }
            Err(e) => {
                held.0 -= len;
                Err(e)
            }
        }
    }
}

impl Source for Direct {
    fn map(&self, va: Va, _: Kind) -> Result<Vec<Chunk>> {
        let (mut chunks, mut off) = (Vec::new(), 0);
        while off < va.len {
            let len = (va.len - off).min(CHUNK);
            match self.one(va, off, len) {
                Ok(c) => chunks.push(c),
                Err(e) => {
                    self.unmap(Va { base: va.base, len: off }, chunks);
                    return Err(e);
                }
            }
            off += len;
        }
        Ok(chunks)
    }

    fn unmap(&self, va: Va, chunks: Vec<Chunk>) {
        let mut off = 0;
        for c in chunks {
            if let Err(e) = self.backend.unmap(va, off, c.len) {
                tracing::error!("varena: unmapping a torch chunk at {:#x} failed: {e}", va.base + off);
            }
            let page = {
                let mut held = self.held.lock();
                held.0 = held.0.saturating_sub(c.len);
                held.1.remove(&c.id)
            };
            if let Some(p) = page {
                let _ = self.backend.destroy(p);
            }
            off += c.len;
        }
    }
}

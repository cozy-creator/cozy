//! The lease manager's policy (varena-leases §1, §2, §6, §8, §10, §12; vram-yield §3): what to
//! allocate, reclaim and refuse, decided from one snapshot of a GPU's page accounting. Pure: no IO,
//! lock, clock or driver call. The manager applies it; a reclaim request is cooperative (each
//! agent's CAS decides).
//!
//! Free is the device's own figure (the driver's free bytes, less what was created since it was
//! read): every context, module, library buffer and foreign byte is simply used, so the page
//! accounting never rests on an agent's word.
//!
//! Watermarks (§12, Linux's min/low/high; no card fractions): min is the reserved pool H (a
//! fixed 512 MiB per GPU, the driver's: never allocated, cr-215); low = min + H, high = min + 2H,
//! so 1 GiB and 1.5 GiB on every GPU (a host test device with no reserve has none).
//! A demand allocation may take free down to min, a speculative one only down to high. Free under
//! low starts background reclaim of cold pages up to high; free under min (foreign growth)
//! reclaims everything unprotected up to high. A new foreign process takes every cold page.
//!
//! Reclaim takes, machine-wide, each stage only while the cheaper ones fall short:
//! 1. per-process free lists (never the asker's own; all of them while another process waits);
//! 2. wholly free arena chunks, then the GPU page cache, oldest first;
//! 3. idle sets' pages, least recently used first;
//! 4. regions: idle ones least recently used first, then by restore cost per byte freed; a graph
//!    goes as a unit (graphs sharing a pool together), a region swaps or is discarded by
//!    measured cost.
//!
//! Stages 1-4 are cold: background reclaim stops there. Then the protected classes (§12, cgroup
//! memory.low / memory.min; against foreign pressure, memory.min is only the streaming floor):
//! 5. later steps' pages (memory.low), furthest advised use first (plan-Belady): the asker's
//!    own, then other executors', the largest holder first;
//! 6. the current step's pages (memory.min): only the asker's own, never another executor's;
//!    furthest first, its next [`SLOTS`] blocks last. Foreign pressure (under min, no asker)
//!    takes every running set's current step down to its floor: its pinned blocks and slots;
//! 7. other executors' pinned, unprotected pages: asked now, they come when their holder's step
//!    ends and unpins them, so the ask waits; unless the holder itself waits on the asker (a wait
//!    cycle), when the later ask is refused instead. The asker's own pins are its floor.
//!
//! Advice is agent-written, so the order
//! is fair only among cooperating executors: an agent that advises every chunk as its current
//! step keeps its bytes longest. Pins and the reserved pool hold regardless.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};

use super::table::State;

/// The blocks past the bound one a running set keeps last: its streaming slots.
pub const SLOTS: usize = 2;

/// One GPU's page accounting at a decision.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    /// The device's free bytes: the driver's last reading less what was created since, within
    /// a `--gpu-memory-limit`.
    pub free: u64,
    /// The reserved pool H: the driver's, never allocated; the min watermark.
    pub reserved: u64,
    /// This reading only: a foreign process absent when sampling began crossed 256 MiB.
    pub new_foreign: bool,
    /// The GPU page cache's entries' bytes, oldest first.
    pub page_cache: Vec<u64>,
    pub executors: Vec<Executor>,
    pub host: Host,
}

/// What a swap to RAM would cost and whether it fits.
#[derive(Clone, Copy, Debug, Default)]
pub struct Host {
    /// MemAvailable and RAM, bytes.
    pub available: u64,
    pub total: u64,
    /// Measured copy rates, bytes/s (0: unmeasured).
    pub d2h: u64,
    pub h2d: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Executor {
    pub id: u64,
    pub chunks: Vec<Chunk>,
    pub regions: Vec<Region>,
    pub graphs: Vec<Graph>,
    /// Executors whose pins its waiting ask counts on (its last [`Answer::Wait`]).
    pub waits_on: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct Chunk {
    pub slot: u32,
    pub seq: u32,
    pub bytes: u64,
    pub state: State,
    /// Every pin, a graph's included.
    pub pins: u32,
    /// A reclaim request is out for it.
    pub asked: bool,
    pub advice: Advice,
}

/// A chunk's advice (madvise; WDDM residency priority), as its agent publishes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Advice {
    /// None (MADV_NORMAL): reclaimed with later steps, nearest last.
    Unset,
    /// On its process's free list: allocated, unassigned.
    FreeList { since: u64 },
    /// Holds nothing once unpinned: a wholly free arena chunk, or a graph pool's.
    Free { since: u64 },
    /// No open plan uses it; machine-wide monotonic ns.
    Idle { last_use: u64 },
    /// A later step uses it (memory.low): its next use, ordered within its executor only.
    Later { at: u64 },
    /// The current step uses it (memory.min): its next use, ordered within its executor only.
    Step { at: u64 },
}

/// A long-lived intermediate (KV, latents, conditioning, a preempted job's state).
#[derive(Clone, Debug)]
pub struct Region {
    pub id: u64,
    /// The chunks it holds (its arena's): it goes whole, never one of them alone.
    pub chunks: Vec<u32>,
    /// Swappable to RAM; otherwise discardable only.
    pub swappable: bool,
    pub last_use: u64,
    /// None: no advised use.
    pub next_use: Option<u64>,
    /// Measured production time, ns (0: unmeasured).
    pub produce_ns: u64,
}

/// A captured graph: one pin on each chunk it reads or writes, its pool's included.
#[derive(Clone, Debug)]
pub struct Graph {
    pub id: u64,
    pub chunks: Vec<u32>,
    pub last_use: u64,
    pub next_use: Option<u64>,
    /// Measured capture plus instantiate, ns (0: unmeasured).
    pub capture_ns: u64,
    /// Already being dropped: its pins go after its last replay.
    pub dropping: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ask {
    pub executor: u64,
    /// The chunk sizes it cuts, summed.
    pub bytes: u64,
    /// Prefetch or a free-list refill: free bytes above high only, never a reclaim.
    pub speculative: bool,
    /// Served whole from this page-cache entry: no new bytes.
    pub cached: Option<usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Decision {
    /// One per ask, in order.
    pub answers: Vec<Answer>,
    /// In the order taken.
    pub picks: Vec<Pick>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Answer {
    Allocate,
    /// Room is coming; `on`: executors whose pins it counts on.
    Wait { on: Vec<u64> },
    /// Queued behind its executor's earlier ask.
    Behind,
    /// No room can come; `room`: what could, for a cut's sizing.
    Refuse { room: u64 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pick {
    /// A reclaim request for a chunk.
    Chunk { executor: u64, slot: u32, seq: u32, bytes: u64, stage: Stage },
    PageCache { index: usize, bytes: u64 },
    /// A region given back whole, swapped or discarded; its chunks follow as [`Pick::Chunk`].
    Region { executor: u64, region: u64, swap: bool },
    /// A graph dropped; the chunks it alone held that nothing needs follow.
    Graph { executor: u64, region: u64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    FreeList,
    Free,
    Idle,
    Region,
    Later,
    Step,
    Pinned,
}

/// The GPU's (min, low, high) watermarks.
pub fn watermarks(s: &Snapshot) -> (u64, u64, u64) {
    (s.reserved, 2 * s.reserved, 3 * s.reserved)
}

/// Every pick for `asks`, plus background and foreign reclaim.
pub fn decide(s: &Snapshot, asks: &[Ask]) -> Decision {
    let facts = Facts::of(s);
    let mut run = Run::new(s, &facts);
    run.gone.extend(asks.iter().filter_map(|a| a.cached));
    let (min, low, high) = watermarks(s);
    let (min, low, high) = (i128::from(min), i128::from(low), i128::from(high));
    let new = |a: &Ask| i128::from(if a.cached.is_some() { 0 } else { a.bytes });
    let free = i128::from(s.free);
    if s.new_foreign {
        run.cover(u64::MAX, None, Scope::Cold);
    }
    if free < low {
        let (coming, _, _) = coming(s, None, &BTreeSet::new());
        let short = high - free - i128::from(coming) - i128::from(run.got);
        if short > 0 {
            run.cover(clamp(short), None, if free < min { Scope::Unprotected } else { Scope::Cold });
        }
    }
    // Every ask that fits now, then room for each executor's first that does not.
    let (mut answers, mut heads, mut blocked) = (vec![], vec![], BTreeSet::new());
    let mut allocated = 0i128;
    for (i, a) in asks.iter().enumerate() {
        let (now, bytes) = (free - allocated - min, new(a));
        answers.push(if blocked.contains(&a.executor) {
            Answer::Behind
        } else if a.speculative && bytes > now + min - high {
            Answer::Refuse { room: clamp(now + min - high) }
        } else if bytes <= now {
            allocated += bytes;
            Answer::Allocate
        } else {
            blocked.insert(a.executor);
            heads.push((i, a));
            Answer::Behind
        });
    }
    let mut waits: BTreeMap<u64, Vec<u64>> = s.executors.iter().map(|e| (e.id, e.waits_on.clone())).collect();
    let mut owed = 0i128;
    for (i, a) in heads {
        waits.remove(&a.executor);
        let cyclic = waiting_on(&waits, a.executor);
        let (coming, later, mut on) = coming(s, Some(a.executor), &cyclic);
        let projected = free - allocated - min - owed + i128::from(coming) + i128::from(run.got);
        let bytes = new(a);
        let from = run.picks.len();
        let asker = s.executors.iter().position(|e| e.id == a.executor);
        run.cyclic = cyclic;
        let covered = run.cover_all(bytes - projected, asker, later);
        let answer = match covered {
            Ok(()) => {
                on.extend(run.picks.iter().skip(from).filter_map(|p| match p {
                    Pick::Graph { executor, .. } | Pick::Chunk { executor, stage: Stage::Pinned, .. } if *executor != a.executor => Some(*executor),
                    _ => None,
                }));
                let on: Vec<u64> = on.into_iter().collect();
                waits.insert(a.executor, on.clone());
                owed += bytes;
                Answer::Wait { on }
            }
            Err(could) => Answer::Refuse { room: clamp(projected + i128::from(could)) },
        };
        if let Some(x) = answers.get_mut(i) {
            *x = answer;
        }
    }
    Decision { answers, picks: run.picks }
}

/// What reclaim could give `executor` from cold pages alone (stages 1-4): the `cold` room a cut
/// sizes itself against, beside the free bytes.
pub fn cold(s: &Snapshot, executor: u64) -> u64 {
    let facts = Facts::of(s);
    let mut run = Run::new(s, &facts);
    run.cover(u64::MAX, s.executors.iter().position(|e| e.id == executor), Scope::Cold);
    run.got
}

/// PSI `some` (§12): the share of the `window` before `now` in which at least one ask waited.
/// `waits` are the closed (start, end) spans of "some ask waiting", `open` the start of one
/// still open; one clock, ns.
pub fn pressure(waits: &[(u64, u64)], open: Option<u64>, now: u64, window: u64) -> f32 {
    let from = now.saturating_sub(window);
    let spans = waits.iter().copied().chain(open.map(|start| (start, now)));
    let waited: u64 = spans.map(|(a, b)| b.min(now).saturating_sub(a.max(from))).sum();
    (waited as f64 / window.max(1) as f64).min(1.0) as f32
}

fn clamp(x: i128) -> u64 {
    x.clamp(0, i128::from(u64::MAX)) as u64
}

/// Bytes already coming back (asked, being reclaimed, released) and whose pins they wait on. An
/// asked chunk still pinned counts unless its executor waits, directly or not, on the asker.
/// Also, for an ask, other executors' allocations not yet acked: never asked before their ack
/// (L3), reclaimable after it.
fn coming(s: &Snapshot, asker: Option<u64>, cyclic: &BTreeSet<u64>) -> (u64, u64, BTreeSet<u64>) {
    let (mut bytes, mut later, mut on) = (0, 0, BTreeSet::new());
    for e in &s.executors {
        for c in &e.chunks {
            match c.state {
                State::Granted if asker.is_some_and(|a| a != e.id) => later += c.bytes,
                State::Revoking | State::Released | State::Refused => bytes += c.bytes,
                State::Mapped if c.asked && c.pins == 0 => bytes += c.bytes,
                State::Mapped if c.asked && Some(e.id) != asker && !cyclic.contains(&e.id) => {
                    bytes += c.bytes;
                    on.insert(e.id);
                }
                _ => {}
            }
        }
    }
    (bytes, later, on)
}

/// Every executor whose waiting ask waits, directly or through others, on `of`'s pins.
fn waiting_on(waits: &BTreeMap<u64, Vec<u64>>, of: u64) -> BTreeSet<u64> {
    let (mut found, mut frontier) = (BTreeSet::new(), vec![of]);
    while let Some(x) = frontier.pop() {
        for (id, on) in waits {
            if on.contains(&x) && found.insert(*id) {
                frontier.push(*id);
            }
        }
    }
    found
}

/// Indices derived once per decision, per executor: each chunk's pinning graphs and whether a
/// region holds it, each graph's chunks, each region's chunks.
struct Facts {
    pinned_by: Vec<Vec<Vec<usize>>>,
    owned: Vec<Vec<bool>>,
    graph_chunks: Vec<Vec<Vec<usize>>>,
    region_chunks: Vec<Vec<Vec<usize>>>,
}

impl Facts {
    fn of(s: &Snapshot) -> Facts {
        let mut f = Facts { pinned_by: vec![], owned: vec![], graph_chunks: vec![], region_chunks: vec![] };
        for e in &s.executors {
            let slots: BTreeMap<u32, usize> = e.chunks.iter().enumerate().map(|(i, c)| (c.slot, i)).collect();
            let graphs: Vec<Vec<usize>> = e.graphs.iter().map(|g| g.chunks.iter().filter_map(|s| slots.get(s).copied()).collect()).collect();
            let pinned_by = (0..e.chunks.len()).map(|c| graphs.iter().enumerate().filter(|(_, cs)| cs.contains(&c)).map(|(g, _)| g).collect()).collect();
            let regions: Vec<Vec<usize>> = e.regions.iter().map(|r| r.chunks.iter().filter_map(|s| slots.get(s).copied()).collect()).collect();
            let mut owned = vec![false; e.chunks.len()];
            regions.iter().flatten().for_each(|&c| owned.get_mut(c).into_iter().for_each(|o| *o = true));
            f.pinned_by.push(pinned_by);
            f.owned.push(owned);
            f.graph_chunks.push(graphs);
            f.region_chunks.push(regions);
        }
        f
    }
}

fn at<T>(v: &[Vec<T>], e: usize, i: usize) -> Option<&T> {
    v.get(e).and_then(|x| x.get(i))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Scope {
    /// Stages 1-3 and idle regions: background reclaim, a new foreign process.
    Cold,
    /// Every stage, running sets down to their floors: reclaim with no asker (foreign growth
    /// under min).
    Unprotected,
    /// Every stage, the asker's own current step last.
    All,
}

/// A region's order: used, restore ns per MiB freed, last use, executor, region.
type Rank = (bool, u128, u64, u64, u64);

/// An executor's running chunks: later steps', its current step's above the floor, its slots.
type Running = (Vec<usize>, Vec<usize>, Vec<usize>);

/// A water-fill queue: bytes left, its order, executor, chunks.
type Queue = (u64, Reverse<usize>, usize, std::vec::IntoIter<usize>);

enum Unit {
    Region(usize, usize),
    /// One graph, or every graph holding a shared chunk.
    Graphs(usize, Vec<usize>),
}

/// One decision's picks so far; chunks, graphs and regions by (executor, index).
#[derive(Clone)]
struct Run<'a> {
    s: &'a Snapshot,
    f: &'a Facts,
    taken: BTreeSet<(usize, usize)>,
    dropped: BTreeSet<(usize, usize)>,
    given: BTreeSet<(usize, usize)>,
    /// Page-cache entries picked or served.
    gone: BTreeSet<usize>,
    /// Host bytes left after this decision's swaps.
    host: u64,
    got: u64,
    picks: Vec<Pick>,
    /// Executors waiting, directly or not, on the current asker: their pins are never waited on.
    cyclic: BTreeSet<u64>,
}

impl<'a> Run<'a> {
    fn new(s: &'a Snapshot, f: &'a Facts) -> Run<'a> {
        Run { s, f, taken: BTreeSet::new(), dropped: BTreeSet::new(), given: BTreeSet::new(), gone: BTreeSet::new(), host: s.host.available, got: 0, picks: vec![], cyclic: BTreeSet::new() }
    }

    fn ex(&self, e: usize) -> Option<&'a Executor> {
        self.s.executors.get(e)
    }

    fn chunk(&self, e: usize, c: usize) -> Option<&'a Chunk> {
        self.ex(e).and_then(|x| x.chunks.get(c))
    }

    fn bytes(&self, e: usize, c: usize) -> u64 {
        self.chunk(e, c).map_or(0, |ch| ch.bytes)
    }

    fn open(&self, e: usize, c: usize) -> bool {
        self.chunk(e, c).is_some_and(|ch| ch.state == State::Mapped && !ch.asked && !self.taken.contains(&(e, c)))
    }

    /// The graphs to drop before `c` can go; `None` when something else pins it.
    fn needs(&self, e: usize, c: usize) -> Option<Vec<usize>> {
        let (ch, by, graphs) = (self.chunk(e, c)?, at(&self.f.pinned_by, e, c)?, &self.ex(e)?.graphs);
        if ch.pins == 0 {
            return Some(vec![]);
        }
        if ch.pins as usize > by.len() {
            return None;
        }
        Some(by.iter().copied().filter(|&g| !graphs.get(g).is_some_and(|x| x.dropping) && !self.dropped.contains(&(e, g))).collect())
    }

    fn free_now(&self, e: usize, c: usize) -> bool {
        self.open(e, c) && self.needs(e, c).is_some_and(|n| n.is_empty())
    }

    /// A region's chunk: it goes with its region, never alone.
    fn owned(&self, e: usize, c: usize) -> bool {
        at(&self.f.owned, e, c).copied().unwrap_or(false)
    }

    fn take(&mut self, e: usize, c: usize, stage: Stage) {
        for g in self.needs(e, c).unwrap_or_default() {
            self.drop_graph(e, g);
        }
        let (Some(x), Some(ch)) = (self.ex(e), self.chunk(e, c)) else { return };
        self.taken.insert((e, c));
        self.got += ch.bytes;
        self.picks.push(Pick::Chunk { executor: x.id, slot: ch.slot, seq: ch.seq, bytes: ch.bytes, stage });
    }

    /// Chunks a dropped graph frees outright: its pool's and idle sets'.
    fn credited(&self, e: usize, c: usize) -> bool {
        self.chunk(e, c).is_some_and(|ch| matches!(ch.advice, Advice::Free { .. } | Advice::Idle { .. }))
    }

    fn graph_chunks(&self, e: usize, g: usize) -> &'a [usize] {
        at(&self.f.graph_chunks, e, g).map_or(&[], Vec::as_slice)
    }

    fn drop_graph(&mut self, e: usize, g: usize) {
        let Some(graph) = self.ex(e).and_then(|x| x.graphs.get(g).map(|graph| (x.id, graph.id))) else { return };
        if !self.dropped.insert((e, g)) {
            return;
        }
        self.picks.push(Pick::Graph { executor: graph.0, region: graph.1 });
        for &c in self.graph_chunks(e, g) {
            if self.credited(e, c) && self.free_now(e, c) {
                self.take(e, c, Stage::Region);
            }
        }
    }

    /// What dropping `graphs` frees outright: credited chunks only they hold.
    fn graphs_credit(&self, e: usize, graphs: &[usize]) -> u64 {
        let held = |c: usize| self.needs(e, c).is_some_and(|n| !n.is_empty() && n.iter().all(|g| graphs.contains(g)));
        let chunks: BTreeSet<usize> = graphs.iter().flat_map(|&g| self.graph_chunks(e, g).iter().copied()).collect();
        chunks.into_iter().filter(|&c| self.credited(e, c) && self.open(e, c) && held(c)).map(|c| self.bytes(e, c)).sum()
    }

    /// A region's chunks and bytes, when all of it can go now.
    fn region_bytes(&self, e: usize, k: usize) -> Option<(&'a [usize], u64)> {
        let chunks = at(&self.f.region_chunks, e, k)?;
        (!chunks.is_empty() && chunks.iter().all(|&c| self.free_now(e, c))).then(|| (chunks.as_slice(), chunks.iter().map(|&c| self.bytes(e, c)).sum()))
    }

    /// Swap (rather than discard) when it is swappable, RAM keeps 10% free after it, and its
    /// measured round trip beats its measured production (an unmeasured production swaps).
    fn swap(&self, region: &Region, bytes: u64) -> Option<u64> {
        let h = &self.s.host;
        let ns = |rate: u64| (rate > 0).then(|| (u128::from(bytes) * 1_000_000_000 / u128::from(rate)) as u64);
        let trip = ns(h.d2h)?.saturating_add(ns(h.h2d)?);
        let fits = self.host.saturating_sub(bytes) >= h.total / 10;
        (region.swappable && fits && (region.produce_ns == 0 || trip < region.produce_ns)).then_some(trip)
    }

    /// The next region to give: idle ones least recently used first, then the cheapest restore
    /// per byte freed (unmeasured last). Cold scope: idle ones only.
    fn best_region(&self, scope: Scope) -> Option<Unit> {
        let per_byte = |ns: u64, bytes: u64| u128::from(ns) * (1 << 20) / u128::from(bytes.max(1));
        let mut best: Option<(Rank, Unit)> = None;
        let mut offer = |key, unit| {
            if best.as_ref().is_none_or(|(b, _)| key < *b) {
                best = Some((key, unit));
            }
        };
        for (e, x) in self.s.executors.iter().enumerate() {
            for (k, region) in x.regions.iter().enumerate() {
                let used = region.next_use.is_some();
                let Some((_, bytes)) = self.region_bytes(e, k).filter(|_| !self.given.contains(&(e, k)) && (scope != Scope::Cold || !used)) else { continue };
                let cost = self.swap(region, bytes).unwrap_or(if region.produce_ns > 0 { region.produce_ns } else { u64::MAX });
                offer((used, if used { per_byte(cost, bytes) } else { 0 }, region.last_use, x.id, region.id), Unit::Region(e, k));
            }
            let live = |g: usize| x.graphs.get(g).is_some_and(|graph| !graph.dropping) && !self.dropped.contains(&(e, g));
            let mut units: BTreeSet<Vec<usize>> = (0..x.graphs.len()).filter(|&g| live(g)).map(|g| vec![g]).collect();
            units.extend((0..x.chunks.len()).filter(|&c| self.credited(e, c) && self.open(e, c)).filter_map(|c| self.needs(e, c)).filter(|n| n.len() > 1));
            for unit in units {
                let graphs: Vec<&Graph> = unit.iter().filter_map(|&g| x.graphs.get(g)).collect();
                let used = graphs.iter().any(|g| g.next_use.is_some());
                let credit = self.graphs_credit(e, &unit);
                if credit == 0 || (scope == Scope::Cold && used) {
                    continue;
                }
                let cost = graphs.iter().map(|g| if g.capture_ns > 0 { g.capture_ns } else { u64::MAX }).fold(0, u64::saturating_add);
                let last = graphs.iter().map(|g| g.last_use).max().unwrap_or(0);
                let id = graphs.first().map_or(0, |g| g.id);
                offer((used, if used { per_byte(cost, credit) } else { 0 }, last, x.id, id), Unit::Graphs(e, unit));
            }
        }
        best.map(|(_, unit)| unit)
    }

    fn give_region(&mut self, e: usize, k: usize) {
        let (Some(x), Some((chunks, bytes))) = (self.ex(e), self.region_bytes(e, k)) else { return };
        let Some(region) = x.regions.get(k) else { return };
        let swap = self.swap(region, bytes).is_some();
        if swap {
            self.host -= bytes;
        }
        self.given.insert((e, k));
        self.picks.push(Pick::Region { executor: x.id, region: region.id, swap });
        for &c in chunks {
            self.take(e, c, Stage::Region);
        }
    }

    /// Chunks that can go now whose advice `key` orders (None: not this stage), in key order.
    fn listed(&self, key: impl Fn(usize, &Advice) -> Option<u64>) -> Vec<(usize, usize)> {
        let mut out = vec![];
        for (e, x) in self.s.executors.iter().enumerate() {
            for (c, ch) in x.chunks.iter().enumerate() {
                if let Some(k) = key(e, &ch.advice).filter(|_| self.free_now(e, c) && !self.owned(e, c)) {
                    out.push(((k, x.id, ch.slot), (e, c)));
                }
            }
        }
        out.sort();
        out.into_iter().map(|(_, at)| at).collect()
    }

    /// An executor's later steps' chunks (memory.low; unadvised chunks too, nearest), its current step's above the floor (memory.min), and its streaming slots (the
    /// nearest [`SLOTS`] unpinned uses); each furthest first, unpinned before a graph's.
    fn running(&self, e: usize) -> Running {
        let Some(x) = self.ex(e) else { return (vec![], vec![], vec![]) };
        let (mut later, mut step) = (vec![], vec![]);
        for (c, ch) in x.chunks.iter().enumerate() {
            let (list, at) = match ch.advice {
                Advice::Later { at } => (&mut later, at),
                Advice::Step { at } => (&mut step, at),
                Advice::Unset => (&mut later, 0),
                _ => continue,
            };
            if self.owned(e, c) {
                continue;
            }
            if let Some(n) = self.needs(e, c).filter(|_| self.open(e, c)) {
                list.push((!n.is_empty(), Reverse(at), ch.slot, c));
            }
        }
        later.sort();
        step.sort();
        let near: BTreeSet<u64> = step.iter().filter(|i| !i.0).map(|i| i.1 .0).collect::<BTreeSet<_>>().into_iter().take(SLOTS).collect();
        let (slots, far): (Vec<_>, Vec<_>) = step.into_iter().partition(|i| !i.0 && near.contains(&i.1 .0));
        let ids = |v: Vec<(bool, Reverse<u64>, u32, usize)>| v.into_iter().map(|i| i.3).collect();
        (ids(later), ids(far), ids(slots))
    }

    /// Per-process free lists are never kept while another process waits (§12): every other
    /// executor's goes whole.
    fn drain_free_lists(&mut self, asker: Option<usize>) {
        for (e, x) in self.s.executors.iter().enumerate().filter(|(e, _)| Some(*e) != asker) {
            for (c, ch) in x.chunks.iter().enumerate() {
                if matches!(ch.advice, Advice::FreeList { .. }) && self.free_now(e, c) && !self.owned(e, c) {
                    self.take(e, c, Stage::FreeList);
                }
            }
        }
    }

    /// Takes from `queues` (executor, chunks in order) the largest remaining first, until
    /// `target`.
    fn water(&mut self, queues: Vec<(usize, Vec<usize>)>, stage: Stage, target: u64) {
        let mut queues: Vec<Queue> =
            queues.into_iter().enumerate().map(|(i, (e, q))| (q.iter().map(|&c| self.bytes(e, c)).sum(), Reverse(i), e, q.into_iter())).collect();
        while self.got < target {
            let Some((left, _, e, q)) = queues.iter_mut().filter(|q| q.0 > 0).max_by_key(|q| (q.0, q.1)) else { return };
            let Some(c) = q.next() else {
                *left = 0;
                continue;
            };
            let e = *e;
            *left -= self.bytes(e, c);
            if self.open(e, c) && self.needs(e, c).is_some() {
                self.take(e, c, stage);
            }
        }
    }

    /// A waiting ask's picks, all or nothing: every other executor's free list, then
    /// [`Run::cover`] over every stage for what is still short of `need` (`later` bytes, coming
    /// once acked, may make up the rest). What it could get when short.
    fn cover_all(&mut self, need: i128, asker: Option<usize>, later: u64) -> Result<(), u64> {
        let mut trial = self.clone();
        trial.drain_free_lists(asker);
        let got = |t: &Run| i128::from(t.got - self.got);
        let rest = need - got(&trial);
        if rest <= 0 || trial.cover(clamp(rest), asker, Scope::All) || got(&trial) + i128::from(later) >= need {
            *self = trial;
            Ok(())
        } else {
            Err(trial.got - self.got)
        }
    }

    /// Picks at least `need` more bytes in stage order within `scope`; false when it fell short.
    fn cover(&mut self, need: u64, asker: Option<usize>, scope: Scope) -> bool {
        let target = self.got.saturating_add(need);
        for stage in [Stage::FreeList, Stage::Free, Stage::Idle] {
            // Each stage's chunks and their order: since, since, last use.
            let key = |e: usize, a: &Advice| match (stage, *a) {
                (Stage::FreeList, Advice::FreeList { since }) if Some(e) != asker => Some(since),
                (Stage::Free, Advice::Free { since }) => Some(since),
                (Stage::Idle, Advice::Idle { last_use }) => Some(last_use),
                _ => None,
            };
            for (e, c) in self.listed(key) {
                if self.got >= target {
                    return true;
                }
                if self.free_now(e, c) {
                    self.take(e, c, stage);
                }
            }
            if stage == Stage::Free {
                for (index, &bytes) in self.s.page_cache.iter().enumerate() {
                    if self.got >= target {
                        return true;
                    }
                    if self.gone.insert(index) {
                        self.got += bytes;
                        self.picks.push(Pick::PageCache { index, bytes });
                    }
                }
            }
        }
        while self.got < target {
            match self.best_region(scope) {
                Some(Unit::Region(e, k)) => self.give_region(e, k),
                Some(Unit::Graphs(e, graphs)) => graphs.into_iter().for_each(|g| self.drop_graph(e, g)),
                None => break,
            }
        }
        if scope == Scope::Cold || self.got >= target {
            return self.got >= target;
        }
        let own = asker.map(|e| (e, self.running(e)));
        let others: Vec<(usize, Running)> = (0..self.s.executors.len()).filter(|&e| Some(e) != asker).map(|e| (e, self.running(e))).collect();
        if let Some((e, (own_later, _, _))) = &own {
            self.water(vec![(*e, own_later.clone())], Stage::Later, target);
        }
        self.water(others.iter().map(|(e, r)| (*e, r.0.clone())).collect(), Stage::Later, target);
        if scope == Scope::Unprotected {
            self.water(others.into_iter().map(|(e, r)| (e, r.1)).collect(), Stage::Step, target);
            return self.got >= target;
        }
        let Some((asker, (_, far, slots))) = own else { return self.got >= target };
        self.water(vec![(asker, far.into_iter().chain(slots).collect())], Stage::Step, target);
        // Others' pinned, unprotected chunks: asked now, given at their unpin.
        for (e, x) in self.s.executors.iter().enumerate() {
            if e == asker || self.cyclic.contains(&x.id) {
                continue;
            }
            for (c, ch) in x.chunks.iter().enumerate() {
                if self.got >= target {
                    return true;
                }
                let protected = matches!(ch.advice, Advice::Step { .. }) || self.owned(e, c);
                if ch.pins > 0 && !protected && self.open(e, c) && self.needs(e, c).is_none() {
                    self.taken.insert((e, c));
                    self.got += ch.bytes;
                    self.picks.push(Pick::Chunk { executor: x.id, slot: ch.slot, seq: ch.seq, bytes: ch.bytes, stage: Stage::Pinned });
                }
            }
        }
        self.got >= target
    }
}

impl Advice {
    /// The lease table slot's `(class, value)` words.
    pub fn encode(&self) -> (u32, u64) {
        match *self {
            Advice::Unset => (0, 0),
            Advice::FreeList { since } => (1, since),
            Advice::Free { since } => (2, since),
            Advice::Idle { last_use } => (3, last_use),
            Advice::Later { at } => (4, at),
            Advice::Step { at } => (6, at),
        }
    }

    /// The table's words back (untrusted: anything unknown is `Unset`).
    pub fn decode(class: u32, value: u64) -> Advice {
        match class {
            1 => Advice::FreeList { since: value },
            2 => Advice::Free { since: value },
            3 => Advice::Idle { last_use: value },
            4 => Advice::Later { at: value },
            6 => Advice::Step { at: value },
            _ => Advice::Unset,
        }
    }
}

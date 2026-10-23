//! The lease manager's policy (varena-leases §1, §2, §6, §8, §10, §12; vram-yield §3): what to
//! allocate, reclaim and refuse, decided from one snapshot of a GPU's page accounting. Pure: no IO,
//! lock, clock or driver call. The manager applies it; a reclaim request is cooperative (each
//! agent's CAS decides).
//!
//! Watermarks (§12, Linux's min/low/high): min is the reserved pool H, never allocated (cr-215);
//! low = min + card/64; high = min + card/32. A demand allocation may take free down to min, a
//! speculative one only down to high. Free under low starts background reclaim of cold pages up
//! to high; free under min (foreign growth) reclaims everything unprotected up to high. A new
//! foreign process takes every cold page.
//!
//! Reclaim takes, machine-wide, each stage only while the cheaper ones fall short:
//! 1. per-process free lists (never the asker's own);
//! 2. wholly free arena chunks, then the GPU page cache, oldest first;
//! 3. idle sets' pages, least recently used first;
//! 4. regions: idle ones least recently used first, then by restore cost per byte freed; a graph
//!    goes as a unit (graphs sharing a pool together), a region swaps or is discarded by
//!    measured cost.
//!
//! Stages 1-4 are cold: background reclaim stops there. Then the protected classes (§12, cgroup
//! memory.low / memory.min):
//! 5. later steps' pages (memory.low), furthest advised use first (plan-Belady): the asker's
//!    own, then other executors', the largest holder first;
//! 6. the current step's pages (memory.min): only the asker's own, never for anyone else;
//!    furthest first, its next [`SLOTS`] blocks last.
//!
//! A chunk pinned by anything but a graph is never taken. Advice is agent-written, so the order
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
    /// min(card, limit) less TensorD's own context.
    pub card: u64,
    /// The reserved pool H = max(512 MiB, card/16): the driver's, never allocated; the min
    /// watermark.
    pub reserved: u64,
    /// Every executor's measured fixed overhead (context, modules, graph exec): outside the
    /// page accounting.
    pub fixed: u64,
    /// Other processes' bytes at the last sample (NVML used − created − fixed).
    pub foreign: u64,
    /// This reading only: a foreign process absent when sampling began crossed 256 MiB.
    pub new_foreign: bool,
    /// Every chunk the manager created: allocated, free lists and the page cache.
    pub created: u64,
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
    /// None yet: protected as the current step's.
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
    /// A chunk of region `id`.
    Region { id: u32 },
}

/// A long-lived intermediate (KV, latents, conditioning, a preempted job's state).
#[derive(Clone, Debug)]
pub struct Region {
    pub id: u32,
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
    pub id: u32,
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
    Region { executor: u64, region: u32, swap: bool },
    /// A graph dropped; the chunks it alone held that nothing needs follow.
    Graph { executor: u64, region: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    FreeList,
    Free,
    Idle,
    Region,
    Later,
    Step,
}

/// The GPU's (min, low, high) watermarks.
pub fn watermarks(s: &Snapshot) -> (u64, u64, u64) {
    (s.reserved, s.reserved + s.card / 64, s.reserved + s.card / 32)
}

/// Every pick for `asks`, plus background and foreign reclaim.
pub fn decide(s: &Snapshot, asks: &[Ask]) -> Decision {
    let facts = Facts::of(s);
    let mut run = Run::new(s, &facts);
    run.gone.extend(asks.iter().filter_map(|a| a.cached));
    let (min, low, high) = watermarks(s);
    let (min, low, high) = (i128::from(min), i128::from(low), i128::from(high));
    let new = |a: &Ask| i128::from(if a.cached.is_some() { 0 } else { a.bytes });
    let free = i128::from(s.card) - i128::from(s.fixed) - i128::from(s.foreign) - i128::from(s.created);
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
        let covered = if bytes > projected { run.cover_all(clamp(bytes - projected), asker, clamp(i128::from(later))) } else { Ok(()) };
        let answer = match covered {
            Ok(()) => {
                on.extend(run.picks.iter().skip(from).filter_map(|p| match p {
                    Pick::Graph { executor, .. } if *executor != a.executor => Some(*executor),
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

/// Indices derived once per decision, per executor: each chunk's pinning graphs, each graph's
/// chunks, each region's chunks.
struct Facts {
    pinned_by: Vec<Vec<Vec<usize>>>,
    graph_chunks: Vec<Vec<Vec<usize>>>,
    region_chunks: Vec<Vec<Vec<usize>>>,
}

impl Facts {
    fn of(s: &Snapshot) -> Facts {
        let mut f = Facts { pinned_by: vec![], graph_chunks: vec![], region_chunks: vec![] };
        for e in &s.executors {
            let slots: BTreeMap<u32, usize> = e.chunks.iter().enumerate().map(|(i, c)| (c.slot, i)).collect();
            let graphs: Vec<Vec<usize>> = e.graphs.iter().map(|g| g.chunks.iter().filter_map(|s| slots.get(s).copied()).collect()).collect();
            let pinned_by = (0..e.chunks.len()).map(|c| graphs.iter().enumerate().filter(|(_, cs)| cs.contains(&c)).map(|(g, _)| g).collect()).collect();
            let regions = e.regions.iter().map(|r| e.chunks.iter().enumerate().filter(|(_, c)| c.advice == Advice::Region { id: r.id }).map(|(i, _)| i).collect()).collect();
            f.pinned_by.push(pinned_by);
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
    /// Every stage but the current step's: reclaim with no asker (foreign growth under min).
    Unprotected,
    /// Every stage, the asker's own current step last.
    All,
}

/// A region's order: used, restore ns per MiB freed, last use, executor, region.
type Rank = (bool, u128, u64, u64, u32);

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
}

impl<'a> Run<'a> {
    fn new(s: &'a Snapshot, f: &'a Facts) -> Run<'a> {
        Run { s, f, taken: BTreeSet::new(), dropped: BTreeSet::new(), given: BTreeSet::new(), gone: BTreeSet::new(), host: s.host.available, got: 0, picks: vec![] }
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
                if let Some(k) = key(e, &ch.advice).filter(|_| self.free_now(e, c)) {
                    out.push(((k, x.id, ch.slot), (e, c)));
                }
            }
        }
        out.sort();
        out.into_iter().map(|(_, at)| at).collect()
    }

    /// An executor's later steps' chunks (memory.low), and its current step's (memory.min,
    /// unadvised and orphaned region chunks too); each furthest first, unpinned before a graph's,
    /// the current step's next [`SLOTS`] unpinned uses last.
    fn running(&self, e: usize) -> (Vec<usize>, Vec<usize>) {
        let Some(x) = self.ex(e) else { return (vec![], vec![]) };
        let (mut later, mut step) = (vec![], vec![]);
        for (c, ch) in x.chunks.iter().enumerate() {
            let (list, at) = match ch.advice {
                Advice::Later { at } => (&mut later, at),
                Advice::Step { at } => (&mut step, at),
                Advice::Unset => (&mut step, 0),
                Advice::Region { id } if !x.regions.iter().any(|r| r.id == id) => (&mut step, 0),
                _ => continue,
            };
            if let Some(n) = self.needs(e, c).filter(|_| self.open(e, c)) {
                list.push((!n.is_empty(), Reverse(at), ch.slot, c));
            }
        }
        later.sort();
        step.sort();
        let near: BTreeSet<u64> = step.iter().filter(|i| !i.0).map(|i| i.1 .0).collect::<BTreeSet<_>>().into_iter().take(SLOTS).collect();
        let (slots, far): (Vec<_>, Vec<_>) = step.into_iter().partition(|i| !i.0 && near.contains(&i.1 .0));
        (later.into_iter().map(|i| i.3).collect(), far.into_iter().chain(slots).map(|i| i.3).collect())
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

    /// [`Run::cover`] over every stage, all or nothing (`later` bytes, coming once acked, may
    /// make up the rest): what it could get when short.
    fn cover_all(&mut self, need: u64, asker: Option<usize>, later: u64) -> Result<(), u64> {
        let mut trial = self.clone();
        if trial.cover(need, asker, Scope::All) || trial.got - self.got + later >= need {
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
        let later = (0..self.s.executors.len()).filter(|&e| Some(e) != asker).map(|e| (e, self.running(e).0)).collect();
        if let Some((e, (own_later, _))) = &own {
            self.water(vec![(*e, own_later.clone())], Stage::Later, target);
        }
        self.water(later, Stage::Later, target);
        if let Some((e, (_, own_step))) = own.filter(|_| scope == Scope::All) {
            self.water(vec![(e, own_step)], Stage::Step, target);
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
            Advice::Region { id } => (5, u64::from(id)),
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
            5 => u32::try_from(value).map_or(Advice::Unset, |id| Advice::Region { id }),
            6 => Advice::Step { at: value },
            _ => Advice::Unset,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]
    use super::*;

    const M: u64 = 1 << 20;

    fn c(slot: u32, mib: u64, advice: Advice) -> Chunk {
        Chunk { slot, seq: slot + 1000, bytes: mib * M, state: State::Mapped, pins: 0, asked: false, advice }
    }

    fn pinned(pins: u32, mut ch: Chunk) -> Chunk {
        ch.pins = pins;
        ch
    }

    fn asked(mut ch: Chunk) -> Chunk {
        ch.asked = true;
        ch
    }

    fn step(slot: u32, at: u64) -> Chunk {
        c(slot, 1, Advice::Step { at })
    }

    fn later(slot: u32, at: u64) -> Chunk {
        c(slot, 1, Advice::Later { at })
    }

    fn idle(slot: u32, last_use: u64) -> Chunk {
        c(slot, 1, Advice::Idle { last_use })
    }

    fn ex(id: u64, chunks: Vec<Chunk>) -> Executor {
        Executor { id, chunks, ..Executor::default() }
    }

    fn region(id: u32, next_use: Option<u64>, last_use: u64, produce_ns: u64) -> Region {
        Region { id, swappable: true, last_use, next_use, produce_ns }
    }

    fn graph(id: u32, chunks: &[u32], next_use: Option<u64>, last_use: u64, capture_ns: u64) -> Graph {
        Graph { id, chunks: chunks.to_vec(), last_use, next_use, capture_ns, dropping: false }
    }

    /// A card with `free` MiB uncreated and a reserved pool of `reserved` MiB. Its card is the
    /// created bytes plus `free`: so small that low and high sit under one chunk above min.
    fn snap(free: i64, reserved: u64, page_cache: Vec<u64>, executors: Vec<Executor>) -> Snapshot {
        let created = executors.iter().flat_map(|e| &e.chunks).map(|c| c.bytes).sum::<u64>() + page_cache.iter().map(|r| r * M).sum::<u64>();
        let card = (created as i64 + free * M as i64) as u64;
        Snapshot { card, reserved: reserved * M, created, page_cache: page_cache.into_iter().map(|r| r * M).collect(), executors, ..Snapshot::default() }
    }

    fn demand(executor: u64, mib: u64) -> Ask {
        Ask { executor, bytes: mib * M, speculative: false, cached: None }
    }

    /// `s` on a 64 GiB card with free exactly at low: no background reclaim.
    fn quiet(mut s: Snapshot) -> Snapshot {
        s.card = 64 << 30;
        let (_, low, _) = watermarks(&s);
        s.fixed = s.card - s.created - low;
        s
    }

    /// An ask `mib` past what `s` (quiet) allocates without reclaim.
    fn past(s: &Snapshot, executor: u64, mib: u64) -> Ask {
        let (min, low, _) = watermarks(s);
        Ask { bytes: low - min + mib * M, ..demand(executor, 0) }
    }

    fn show(d: &Decision) -> Vec<String> {
        d.picks
            .iter()
            .map(|p| match p {
                Pick::Chunk { executor, slot, .. } => format!("{executor}:{slot}"),
                Pick::PageCache { index, .. } => format!("P{index}"),
                Pick::Region { executor, region, swap } => format!("{}{executor}:{region}", if *swap { "S" } else { "D" }),
                Pick::Graph { executor, region } => format!("G{executor}:{region}"),
            })
            .collect()
    }

    fn bytes(d: &Decision) -> u64 {
        d.picks
            .iter()
            .map(|p| match p {
                Pick::Chunk { bytes, .. } | Pick::PageCache { bytes, .. } => *bytes,
                _ => 0,
            })
            .sum()
    }

    /// Executor 1 asks while running; 2 runs beside it; 3 is idle with a region and a graph.
    fn ladder() -> Snapshot {
        let one = ex(1, vec![c(1, 1, Advice::FreeList { since: 1 }), pinned(1, step(10, 1)), step(11, 2), step(12, 3), step(13, 7), later(14, 9), later(15, 8)]);
        let mut two = ex(2, vec![c(1, 1, Advice::FreeList { since: 5 }), pinned(1, step(20, 1)), step(21, 2), step(22, 3), later(23, 8), later(24, 6), c(25, 1, Advice::Region { id: 9 }), c(26, 1, Advice::Region { id: 9 })]);
        two.regions = vec![region(9, Some(4), 1, 1_000_000)];
        let mut three = ex(3, vec![c(1, 1, Advice::Free { since: 3 }), idle(2, 50), idle(3, 10), c(4, 1, Advice::Region { id: 7 }), c(5, 1, Advice::Region { id: 7 }), pinned(1, idle(6, 20)), pinned(1, c(7, 1, Advice::Free { since: 0 }))]);
        three.regions = vec![region(7, None, 30, 1_000_000)];
        three.graphs = vec![graph(8, &[6, 7], None, 15, 1_000_000)];
        snap(0, 0, vec![1], vec![one, two, three])
    }

    const LADDER: &[&str] = &[
        "2:1", // free lists, never the asker's own (1:1)
        "3:1", "P0", // free arena chunks, then the page cache
        "3:3", "3:2", // idle sets, LRU across executors
        "G3:8", "3:6", "3:7", // an idle graph as a unit: its idle set's chunk and pool
        "D3:7", "3:4", "3:5", // an idle region
        "D2:9", "2:25", "2:26", // a used region
        "1:14", "1:15", // memory.low: the asker's own later steps, furthest first
        "2:23", "2:24", // then another executor's
        "1:13", "1:12", "1:11", // memory.min: the asker's own current step, slots (12, 11) last
                                // never 2:21, 2:22 (another's current step), 1:10, 2:20 (pinned)
    ];

    #[test]
    fn each_stage_runs_only_once_the_cheaper_ones_are_short() {
        // (need MiB past the free room, picks taken: a prefix of LADDER)
        let s = quiet(ladder());
        let rows: &[(u64, usize)] = &[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 8), (7, 8), (8, 11), (9, 11), (10, 14), (11, 14), (12, 15), (13, 16), (14, 17), (15, 18), (16, 19), (17, 20), (18, 21)];
        for &(need, n) in rows {
            let d = decide(&s, &[past(&s, 1, need)]);
            assert_eq!(show(&d), LADDER[..n], "need {need} MiB");
            // From the graph on, it waits on executor 3 releasing the graph's pins.
            let on = if n >= 6 { vec![3] } else { vec![] };
            assert_eq!(d.answers, vec![Answer::Wait { on }], "need {need} MiB");
        }
        let d = decide(&s, &[past(&s, 1, 19)]);
        let (min, low, _) = watermarks(&s);
        assert_eq!(d.answers, vec![Answer::Refuse { room: low - min + 18 * M }]);
        assert!(d.picks.is_empty(), "a refused ask reclaims nothing");
        // Under low, background reclaim takes cold pages first, the asker's free list too.
        assert_eq!(show(&decide(&ladder(), &[])), ["1:1"]);
    }

    #[test]
    fn free_room_allocates_with_no_reclaim() {
        let mut s = ladder();
        s.card += 4 * M;
        let d = decide(&s, &[demand(1, 3), demand(2, 1), demand(1, 2)]);
        assert_eq!(d.answers, vec![Answer::Allocate, Answer::Allocate, Answer::Wait { on: vec![] }]);
        assert_eq!(show(&d), ["2:1", "3:1"]);
        let d = decide(&s, &[demand(1, 9), demand(1, 1)]);
        assert_eq!(d.answers[1], Answer::Behind);
    }

    /// Watermarks on a 64 MiB card with min = 4 MiB: low = 5, high = 6. (free MiB, MiB already
    /// coming back, MiB picked, the last stage reached).
    #[test]
    fn watermarks_reclaim_cold_pages_in_the_background_and_the_unprotected_below_min() {
        let rows: &[(i64, u64, u64, Option<Stage>)] = &[
            (8, 0, 0, None),
            (6, 0, 0, None),
            (5, 0, 0, None),                // at low: nothing
            (4, 0, 2, Some(Stage::Idle)),   // under low: cold only, up to high
            (4, 1, 1, Some(Stage::Idle)),   // what is coming counts
            (3, 0, 3, Some(Stage::Later)),  // under min: idle (2 MiB), then memory.low
            (-2, 0, 4, Some(Stage::Later)), // under min with too little: all but the current step
        ];
        for &(free, coming, picked, last) in rows {
            let mut chunks = vec![idle(1, 1), idle(2, 2), later(3, 9), later(4, 8), step(5, 2), step(6, 1)];
            chunks.extend((0..coming as u32).map(|i| asked(later(100 + i, 1))));
            let mut s = snap(0, 4, vec![], vec![ex(1, chunks)]);
            s.card = 64 * M;
            s.fixed = (i128::from(s.card) - i128::from(s.created) - i128::from(free) * i128::from(M)) as u64;
            let d = decide(&s, &[]);
            assert_eq!(bytes(&d), picked * M, "free {free} MiB, {coming} MiB coming");
            let stages: Vec<Stage> = d.picks.iter().filter_map(|p| if let Pick::Chunk { stage, .. } = p { Some(*stage) } else { None }).collect();
            assert_eq!(stages.last().copied(), last, "free {free} MiB");
            assert!(!stages.contains(&Stage::Step), "free {free} MiB: no reclaim without an asker takes a current step");
        }
        // Demand allocates down to min; speculation only down to high.
        let spec = |mib| Ask { speculative: true, ..demand(1, mib) };
        let mut s = snap(0, 4, vec![], vec![ex(1, vec![])]);
        s.card = 64 * M;
        s.fixed = s.card - 10 * M;
        assert_eq!(decide(&s, &[demand(1, 6)]).answers, vec![Answer::Allocate]);
        assert_eq!(decide(&s, &[demand(1, 7)]).answers, vec![Answer::Refuse { room: 6 * M }]);
        assert_eq!(decide(&s, &[spec(4)]).answers, vec![Answer::Allocate]);
        let d = decide(&s, &[spec(5)]);
        assert_eq!((d.answers, d.picks), (vec![Answer::Refuse { room: 4 * M }], vec![]));
        assert_eq!(decide(&s, &[demand(1, 5), spec(1)]).answers[1], Answer::Refuse { room: 0 });
    }

    #[test]
    fn a_new_foreign_process_takes_every_cold_page_and_nothing_protected() {
        let mut s = ladder();
        s.card += 100 * M;
        s.new_foreign = true;
        let d = decide(&s, &[]);
        assert_eq!(show(&d), ["1:1", "2:1", "3:1", "P0", "3:3", "3:2", "G3:8", "3:6", "3:7", "D3:7", "3:4", "3:5"]);
        // An ask served from the page cache keeps its entry.
        let d = decide(&s, &[Ask { cached: Some(0), ..demand(1, 1) }]);
        assert!(!show(&d).contains(&"P0".to_string()));
        assert_eq!(d.answers, vec![Answer::Allocate]);
    }

    /// 1 asks, running; 2 and 3 run beside it; 4 is idle.
    fn co_resident() -> Snapshot {
        let a = ex(1, vec![pinned(1, step(1, 1)), step(2, 2), step(3, 3), step(5, 5), later(6, 6)]);
        let b = ex(2, vec![pinned(1, step(1, 1)), step(2, 2), step(3, 3), later(7, 7), later(8, 8), later(9, 9)]);
        let c3 = ex(3, vec![pinned(1, step(1, 1)), step(2, 2), later(4, 4), later(5, 5)]);
        let d = ex(4, vec![idle(1, 7), idle(2, 3)]);
        snap(0, 0, vec![], vec![a, b, c3, d])
    }

    #[test]
    fn co_residents_lose_only_later_steps_and_only_once_the_asker_streams() {
        // (need MiB, picks)
        let rows: &[(u64, &[&str])] = &[
            (2, &["4:2", "4:1"]),
            (3, &["4:2", "4:1", "1:6"]),
            // Largest holder of later steps first: 2 has 3, 3 has 2; ties go to the first.
            (6, &["4:2", "4:1", "1:6", "2:9", "2:8", "3:5"]),
            (8, &["4:2", "4:1", "1:6", "2:9", "2:8", "3:5", "2:7", "3:4"]),
            // Then the asker's own current step, furthest first, its slots last.
            (11, &["4:2", "4:1", "1:6", "2:9", "2:8", "3:5", "2:7", "3:4", "1:5", "1:3", "1:2"]),
        ];
        for &(need, picks) in rows {
            let d = decide(&co_resident(), &[demand(1, need)]);
            assert_eq!(show(&d), picks, "need {need} MiB");
        }
        // No spike: while idle sets and the asker's own later steps cover it, no co-resident
        // loses a chunk; and another's current step never goes.
        for need in 1..=3 {
            let d = decide(&co_resident(), &[demand(1, need)]);
            assert!(d.picks.iter().all(|p| matches!(p, Pick::Chunk { executor: 1 | 4, .. })), "need {need} MiB: {:?}", show(&d));
        }
        let d = decide(&co_resident(), &[demand(1, 12)]);
        assert_eq!((show(&d), d.answers), (vec!["4:2".to_string()], vec![Answer::Refuse { room: 11 * M }]));
    }

    #[test]
    fn the_askers_own_step_goes_furthest_next_use_first() {
        // Unpinned uses 5,1,9,3,7,2; 0 is bound; 8 is held only by a running graph.
        let mut a = ex(1, vec![pinned(1, step(0, 0)), step(5, 5), step(1, 1), step(9, 9), step(3, 3), step(7, 7), step(2, 2), pinned(1, step(8, 8))]);
        a.graphs = vec![graph(4, &[8], Some(1), 1, 1_000_000)];
        let order = ["1:9", "1:7", "1:5", "1:3", "G1:4", "1:8", "1:2", "1:1"];
        for need in 1..=7 {
            let d = decide(&snap(0, 0, vec![], vec![a.clone()]), &[demand(1, need)]);
            let want = &order[..need as usize + usize::from(need >= 5)];
            assert_eq!(show(&d), want, "need {need} MiB");
        }
        let d = decide(&snap(0, 0, vec![], vec![a]), &[demand(1, 8)]);
        assert_eq!(d.answers, vec![Answer::Refuse { room: 7 * M }], "the bound chunk never goes");
    }

    #[test]
    fn regions_go_by_rank_and_a_graph_as_a_unit() {
        let host = Host { available: 64 << 30, total: 64 << 30, d2h: 1 << 30, h2d: 1 << 30 };
        let mut a = ex(1, vec![c(1, 2, Advice::Region { id: 1 }), c(2, 2, Advice::Region { id: 2 }), c(3, 4, Advice::Region { id: 3 }), pinned(2, c(4, 1, Advice::Free { since: 0 })), pinned(1, idle(5, 0)), pinned(1, later(6, 6))]);
        // Regions 1 and 2 used: 1 costs 4 ms to make (swap 3.9 ms), 2 costs 50 µs (discarded).
        // Region 3 has no advised use. Graphs 5 and 6 share the pool chunk 4; 5 holds later 6,
        // 6 holds idle 5.
        a.regions = vec![region(1, Some(2), 1, 4_000_000), region(2, Some(2), 1, 50_000), region(3, None, 9, 0)];
        a.graphs = vec![graph(5, &[4, 6], Some(1), 2, 1_000_000), graph(6, &[4, 5], None, 3, 1_000_000)];
        let mut s = snap(0, 0, vec![], vec![a, ex(2, vec![])]);
        s.host = host;
        // Idle first, LRU: graph 6 (its idle chunk 5; the pool stays held by 5), then region 3
        // (unmeasured: swapped); then used, cheapest per byte: region 2 (discarded), graph 5 (the
        // pool, now its alone; 1 ms/MiB), region 1 (1.95 ms/MiB). Later chunk 6 stays.
        let d = decide(&s, &[demand(2, 10)]);
        assert_eq!(show(&d), ["G1:6", "1:5", "S1:3", "1:3", "D1:2", "1:2", "G1:5", "1:4", "S1:1", "1:1"]);
        // Graph 6 dropping already: graph 5 alone holds the pool; then, once the regions are
        // short, the later chunk graph 5 held.
        s.executors[0].graphs[1].dropping = true;
        s.executors[0].chunks[4].asked = true;
        let d = decide(&s, &[demand(2, 11)]);
        assert_eq!(show(&d), ["S1:3", "1:3", "D1:2", "1:2", "G1:5", "1:4", "S1:1", "1:1", "1:6"]);
        assert_eq!(d.answers, vec![Answer::Wait { on: vec![1] }]);
    }

    /// Graphs 1 (100 µs to capture) and 2 (1 s) share pool chunk 1: neither frees it alone, so
    /// region 3 (10 ms/MiB) goes before the pair (1.0001 s/MiB), and the pair before later 3.
    #[test]
    fn graphs_sharing_a_pool_go_together() {
        let mut a = ex(1, vec![pinned(2, c(1, 1, Advice::Free { since: 0 })), c(2, 1, Advice::Region { id: 3 }), later(3, 1)]);
        a.regions = vec![region(3, Some(1), 0, 10_000_000)];
        a.graphs = vec![graph(1, &[1], Some(1), 0, 100_000), graph(2, &[1], Some(1), 0, 1_000_000_000)];
        let rows: &[(u64, &[&str])] = &[(1, &["D1:3", "1:2"]), (2, &["D1:3", "1:2", "G1:1", "G1:2", "1:1"]), (3, &["D1:3", "1:2", "G1:1", "G1:2", "1:1", "1:3"])];
        for &(need, picks) in rows {
            let d = decide(&snap(0, 0, vec![], vec![a.clone(), ex(2, vec![])]), &[demand(2, need)]);
            assert_eq!(show(&d), picks, "need {need} MiB");
        }
    }

    #[test]
    fn swap_or_discard_by_measured_cost() {
        let gb = 1 << 30;
        // (swappable, produce ns, d2h = h2d, host available, swapped); 4 MiB round trip at 1 GiB/s ≈ 7.8 ms.
        let rows: &[(bool, u64, u64, u64, bool)] = &[
            (true, 10_000_000, gb, 64 * gb, true),
            (true, 5_000_000, gb, 64 * gb, false),
            (false, 10_000_000, gb, 64 * gb, false),
            (true, 10_000_000, gb, 64 * gb / 10, false), // RAM would go under 10%
            (true, 0, gb, 64 * gb, true),                // unmeasured production keeps its state
            (true, 10_000_000, 0, 64 * gb, false),       // unmeasured copies
        ];
        for &(swappable, produce_ns, rate, available, swapped) in rows {
            let mut a = ex(1, vec![c(1, 4, Advice::Region { id: 1 })]);
            a.regions = vec![Region { swappable, ..region(1, Some(1), 0, produce_ns) }];
            let mut s = snap(0, 0, vec![], vec![a, ex(2, vec![])]);
            s.host = Host { available, total: 64 * gb, d2h: rate, h2d: rate };
            let d = decide(&s, &[demand(2, 4)]);
            assert_eq!(d.picks[0], Pick::Region { executor: 1, region: 1, swap: swapped }, "{swappable} {produce_ns} {rate} {available}");
        }
    }

    #[test]
    fn pins_wait_or_refuse() {
        // Only the asker's own bound chunk: refused, nothing taken.
        let d = decide(&snap(0, 0, vec![], vec![ex(1, vec![pinned(1, step(1, 1))])]), &[demand(1, 1)]);
        assert_eq!((d.answers, d.picks), (vec![Answer::Refuse { room: 0 }], vec![]));
        // Another's chunk asked while unpinned and pinned since: it comes at its unpin.
        let s = |waits_on: Vec<u64>| {
            let mut b = ex(2, vec![asked(pinned(1, later(1, 1)))]);
            b.waits_on = waits_on;
            snap(1, 0, vec![], vec![ex(1, vec![pinned(1, step(1, 1))]), b])
        };
        assert_eq!(decide(&s(vec![]), &[demand(1, 2)]).answers, vec![Answer::Wait { on: vec![2] }]);
        // Unless that executor waits on the asker's pins: a cycle.
        assert_eq!(decide(&s(vec![1]), &[demand(1, 2)]).answers, vec![Answer::Refuse { room: M }]);
        // Another's allocation not yet acked is never asked (L3) but makes up the rest: the ask
        // takes what it can now and waits. Background reclaim does not count it.
        let allocated = |mut ch: Chunk| {
            ch.state = State::Granted;
            ch
        };
        let s = snap(1, 0, vec![], vec![ex(1, vec![]), ex(2, vec![idle(1, 1), allocated(later(2, 1))])]);
        let d = decide(&s, &[demand(1, 3)]);
        assert_eq!((show(&d), d.answers), (vec!["2:1".to_string()], vec![Answer::Wait { on: vec![] }]));
        assert_eq!(decide(&s, &[demand(1, 4)]).answers, vec![Answer::Refuse { room: 2 * M }]);
    }

    /// Deterministic random page accounting: no ask takes a pinned chunk, the asker's own free
    /// list or another's current step; stages run in order; a protected class loses nothing
    /// while cheaper ones are left; and no pick but the last was needed.
    #[test]
    fn no_protected_pick_while_cold_pages_cover() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rand = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for case in 0..400 {
            let executors: Vec<Executor> = (1..=3)
                .map(|id| {
                    let chunks = (0..12)
                        .map(|slot| {
                            let v = rand(100);
                            let advice = match rand(6) {
                                0 => Advice::FreeList { since: v },
                                1 => Advice::Free { since: v },
                                2 => Advice::Idle { last_use: v },
                                3 => Advice::Later { at: v },
                                4 => Advice::Step { at: v },
                                _ => Advice::Unset,
                            };
                            let ch = c(slot, 1 + rand(4), advice);
                            if rand(5) == 0 { pinned(1, ch) } else { ch }
                        })
                        .collect();
                    ex(id, chunks)
                })
                .collect();
            let need = 1 + rand(60);
            let s = quiet(snap(0, 0, vec![], executors));
            let d = decide(&s, &[past(&s, 1, need)]);
            let chunk = |e: u64, slot: u32| s.executors.iter().find(|x| x.id == e).and_then(|x| x.chunks.iter().find(|c| c.slot == slot)).unwrap();
            let mut stages = vec![];
            for p in &d.picks {
                let Pick::Chunk { executor, slot, stage, .. } = p else { continue };
                let ch = chunk(*executor, *slot);
                assert_eq!(ch.pins, 0, "case {case}: a pinned chunk");
                assert!(!(*executor == 1 && matches!(ch.advice, Advice::FreeList { .. })), "case {case}: the asker's own free list");
                assert!(*executor == 1 || !matches!(ch.advice, Advice::Step { .. } | Advice::Unset), "case {case}: another's current step");
                stages.push((*stage, ch.bytes));
            }
            assert!(stages.windows(2).all(|w| w[0].0 <= w[1].0), "case {case}: stages out of order {stages:?}");
            let short = need * M;
            let picked: BTreeSet<String> = show(&d).into_iter().collect();
            for protected in [Stage::Later, Stage::Step] {
                if !stages.iter().any(|s| s.0 == protected) {
                    continue;
                }
                let cheaper: u64 = stages.iter().filter(|s| s.0 < protected).map(|s| s.1).sum();
                assert!(cheaper < short, "case {case}: a {protected:?} pick with {cheaper} cheaper bytes for {short}");
                for e in &s.executors {
                    for ch in e.chunks.iter().filter(|ch| ch.pins == 0) {
                        let cheap = match ch.advice {
                            Advice::Free { .. } | Advice::Idle { .. } => true,
                            Advice::FreeList { .. } => e.id != 1,
                            Advice::Later { .. } => protected == Stage::Step,
                            _ => false,
                        };
                        assert!(!cheap || picked.contains(&format!("{}:{}", e.id, ch.slot)), "case {case}: {}:{} left for a {protected:?} pick", e.id, ch.slot);
                    }
                }
            }
            if let (Answer::Wait { .. }, Some(last)) = (&d.answers[0], stages.last()) {
                assert!(bytes(&d) - last.1 < short, "case {case}: picked past the need");
            }
        }
    }

    #[test]
    fn cold_room_is_what_reclaim_gives_without_a_running_set() {
        let s = ladder();
        // Executor 1's: 2's free list, 3's free chunk, the page cache, 3's idle sets (2), its idle
        // graph's two chunks, its idle region (2); not 2's used region nor any running set.
        assert_eq!(cold(&s, 1), 9 * M);
        // Executor 2's own free list is not its room; 1's is.
        assert_eq!(cold(&s, 2), 9 * M);
        assert_eq!(cold(&s, 3), 10 * M);
    }

    /// PSI `some` over a window: (closed spans, open since, now, window, share).
    #[test]
    fn pressure_is_the_share_of_the_window_with_an_ask_waiting() {
        let s = 1_000_000_000u64;
        let rows: &[(&[(u64, u64)], Option<u64>, u64, u64, f32)] = &[
            (&[], None, 100 * s, 10 * s, 0.0),
            (&[(91 * s, 96 * s)], None, 100 * s, 10 * s, 0.5),
            (&[(85 * s, 95 * s)], None, 100 * s, 10 * s, 0.5), // only the part inside the window
            (&[(91 * s, 92 * s)], Some(98 * s), 100 * s, 10 * s, 0.3),
            (&[(25 * s, 49 * s), (91 * s, 92 * s)], Some(98 * s), 100 * s, 60 * s, 0.2),
            (&[], Some(0), 100 * s, 10 * s, 1.0),
        ];
        for &(waits, open, now, window, share) in rows {
            let got = pressure(waits, open, now, window);
            assert!((got - share).abs() < 1e-6, "{waits:?} {open:?}: {got} for {share}");
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Ev {
        Demand(u64, u64),
        Spec(u64, u64),
        Cached(u64),
        /// An executor's request ends: its chunks go idle.
        Idle(u64),
        /// An executor exits: its chunks go to the page cache.
        Exit(u64),
        Foreign(u64),
        Fixed(u64),
    }

    /// A manager applying each decision at once (every reclaim request released, every
    /// allocation mapped): device free = card − fixed − foreign − created.
    struct Sim {
        s: Snapshot,
        clock: u64,
    }

    impl Sim {
        fn new() -> Sim {
            let s = Snapshot { card: 64 * M, reserved: 4 * M, executors: (1..=3).map(|id| ex(id, vec![])).collect(), ..Snapshot::default() };
            Sim { s, clock: 0 }
        }

        fn free(&self) -> i128 {
            i128::from(self.s.card) - i128::from(self.s.fixed) - i128::from(self.s.foreign) - i128::from(self.s.created)
        }

        fn chunks(&mut self, id: u64) -> &mut Vec<Chunk> {
            &mut self.s.executors.iter_mut().find(|e| e.id == id).unwrap().chunks
        }

        /// Serves `asks` until none waits; every decision that allocates leaves min free.
        fn serve(&mut self, mut asks: Vec<Ask>, trace: &str) {
            for _ in 0..8 {
                let d = decide(&self.s, &asks);
                for p in &d.picks {
                    match *p {
                        Pick::Chunk { executor, slot, bytes, .. } => {
                            self.chunks(executor).retain(|c| c.slot != slot);
                            self.s.created -= bytes;
                        }
                        Pick::PageCache { index, bytes } => {
                            self.s.page_cache[index] = 0;
                            self.s.created -= bytes;
                        }
                        _ => {}
                    }
                }
                let mut allocated = false;
                let mut left = vec![];
                for (a, answer) in asks.iter().zip(&d.answers) {
                    match answer {
                        Answer::Allocate => {
                            allocated = true;
                            self.clock += 1;
                            let slot = self.clock as u32;
                            let bytes = match a.cached {
                                Some(i) => std::mem::take(&mut self.s.page_cache[i]),
                                None => {
                                    self.s.created += a.bytes;
                                    a.bytes
                                }
                            };
                            let advice = if a.speculative { Advice::FreeList { since: self.clock } } else { Advice::Later { at: self.clock } };
                            self.chunks(a.executor).push(Chunk { bytes, ..c(slot, 0, advice) });
                        }
                        Answer::Wait { .. } | Answer::Behind => left.push(*a),
                        Answer::Refuse { .. } => {}
                    }
                }
                if allocated {
                    assert!(self.free() >= i128::from(self.s.reserved), "{trace}: an allocation left {} MiB free, in the reserved pool", self.free() >> 20);
                }
                if left.is_empty() {
                    return;
                }
                asks = left;
            }
        }

        fn run(&mut self, events: &[Ev]) {
            for (i, ev) in events.iter().enumerate() {
                let trace = format!("event {i} {ev:?}");
                match *ev {
                    Ev::Demand(e, mib) => self.serve(vec![demand(e, mib)], &trace),
                    Ev::Spec(e, mib) => self.serve(vec![Ask { speculative: true, ..demand(e, mib) }], &trace),
                    Ev::Cached(e) => {
                        if let Some(i) = self.s.page_cache.iter().position(|&b| b > 0) {
                            self.serve(vec![Ask { cached: Some(i), ..demand(e, 0) }], &trace);
                        }
                    }
                    Ev::Idle(e) => {
                        let clock = self.clock;
                        for c in self.chunks(e).iter_mut() {
                            c.advice = Advice::Idle { last_use: clock };
                        }
                    }
                    Ev::Exit(e) => {
                        let gone: u64 = std::mem::take(self.chunks(e)).iter().map(|c| c.bytes).sum();
                        self.s.page_cache.push(gone);
                    }
                    Ev::Foreign(mib) => self.s.foreign = mib * M,
                    Ev::Fixed(mib) => self.s.fixed = mib * M,
                }
                self.serve(vec![], &trace);
                // Whatever moved, while reclaimable bytes remain the device is back to min.
                let reclaimable = self.s.page_cache.iter().sum::<u64>() + self.s.executors.iter().flat_map(|e| &e.chunks).filter(|c| c.pins == 0).map(|c| c.bytes).sum::<u64>();
                assert!(self.free() >= i128::from(self.s.reserved) || reclaimable == 0, "{trace}: {} MiB free with {reclaimable} reclaimable", self.free() >> 20);
            }
        }
    }

    /// The reserved pool (cr-215): no sequence of asks, exits, page-cache hits, foreign or
    /// context growth allocates into min. Card 64 MiB, min 4 MiB.
    #[test]
    fn no_decision_sequence_allocates_into_the_reserved_pool() {
        use Ev::*;
        let rows: &[&[Ev]] = &[
            &[Demand(1, 30), Demand(2, 30), Demand(3, 1), Spec(1, 1)],
            &[Demand(1, 58), Spec(2, 1), Demand(2, 8), Fixed(6), Demand(3, 2)],
            &[Demand(1, 40), Idle(1), Demand(2, 30), Foreign(10), Spec(3, 4), Demand(3, 4)],
            &[Demand(1, 50), Exit(1), Foreign(14), Cached(2), Demand(2, 10)],
            &[Demand(1, 20), Demand(2, 20), Demand(3, 20), Fixed(8), Foreign(5), Spec(1, 2), Demand(2, 3)],
        ];
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rand = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        let mut generated = vec![];
        for _ in 0..300 {
            let events: Vec<Ev> = (0..30)
                .map(|_| {
                    let e = 1 + rand(3);
                    match rand(10) {
                        0..=3 => Demand(e, 1 + rand(24)),
                        4 => Spec(e, 1 + rand(8)),
                        5 => Cached(e),
                        6 => Idle(e),
                        7 => Exit(e),
                        8 => Foreign(rand(20)),
                        _ => Fixed(rand(10)),
                    }
                })
                .collect();
            generated.push(events);
        }
        for events in rows.iter().copied().chain(generated.iter().map(Vec::as_slice)) {
            Sim::new().run(events);
        }
    }

    #[test]
    fn advice_round_trips_the_table_words() {
        for a in [Advice::Unset, Advice::FreeList { since: 7 }, Advice::Free { since: 8 }, Advice::Idle { last_use: 9 }, Advice::Later { at: 10 }, Advice::Region { id: 11 }, Advice::Step { at: 12 }] {
            let (class, value) = a.encode();
            assert_eq!(Advice::decode(class, value), a);
        }
        assert_eq!(Advice::decode(5, u64::MAX), Advice::Unset);
        assert_eq!(Advice::decode(99, 1), Advice::Unset);
    }
}

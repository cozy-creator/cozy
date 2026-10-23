//! The lease manager's policy (varena-leases §1, §2, §6, §8, §10; vram-yield §3; varena-slim
//! steps 1-2): what to grant, ask back and refuse, decided from one snapshot of a GPU's books.
//! Pure: no IO, lock, clock or driver call. The manager applies it; each agent's CAS decides.
//!
//! Room the manager has not created comes first and revokes nothing. Then, machine-wide, each
//! stage only while the cheaper ones could not cover the need:
//! 1. floats (never the asker's own);
//! 2. wholly free arena chunks, then retained regions, oldest first;
//! 3. idle chunks, least recently used first;
//! 4. cache and graph regions: idle ones least recently used first, then by restore cost per
//!    byte freed; a graph goes as a unit, a cache spills or drops by measured cost;
//! 5. running sets' chunks, furthest declared next use first (plan-Belady): the asker's own,
//!    then other executors', the largest holder first;
//! 6. the watermark: each running set's next [`SLOTS`] blocks, the asker's first.
//!
//! A chunk pinned by anything but a graph is never picked. Graphs sharing a chunk (a process's
//! capture pool) free it only together, so they go as one unit.
//!
//! Hints are agent-written, so the order is fair only among cooperating executors: an agent that
//! marks every chunk running keeps its bytes longest. Pins and H hold regardless.
//!
//! Driver headroom H (cr-215) is never leased, in either mode: no grant, float, prefetch, refill
//! or adoption leaves the device under H, and free under H asks back. VRAM yield (shared mode):
//! it asks back to 2H, and speculation grows only into free − 2H, so the H..2H band is the
//! hysteresis; a new foreign process takes every idle chunk and region and nothing in use.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};

use super::table::State;

/// The blocks past the bound one a running set keeps last: its streaming slots.
pub const SLOTS: usize = 2;

/// One GPU's books at a decision.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    /// min(card, limit) less TensorD's own context.
    pub card: u64,
    /// H = max(512 MiB, card/16): the driver's, never leased.
    pub headroom: u64,
    /// Shared mode (local): yield to other apps through the H..2H band.
    pub shared: bool,
    /// Every executor's measured fixed overhead (context, modules, graph exec): outside the books.
    pub fixed: u64,
    /// Other processes' bytes at the last sample (NVML used − created − fixed).
    pub foreign: u64,
    /// This reading only: a foreign process absent when sampling began crossed 256 MiB.
    pub new_foreign: bool,
    /// Every chunk the manager created: leased, floats and retained.
    pub created: u64,
    /// Retained regions' bytes, oldest first.
    pub retained: Vec<u64>,
    pub executors: Vec<Executor>,
    pub host: Host,
}

/// What a spill to RAM would cost and whether it fits.
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
    pub caches: Vec<Cache>,
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
    /// Asked back and not yet released.
    pub asked: bool,
    pub hint: Hint,
}

/// A chunk's class, as its agent publishes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hint {
    /// Not yet hinted: the running set's nearest.
    Unset,
    /// Granted, unassigned.
    Float { since: u64 },
    /// Holds nothing once unpinned: a wholly free arena chunk, or a graph pool's.
    Free { since: u64 },
    /// No open plan uses it; machine-wide monotonic ns.
    Idle { last_use: u64 },
    /// A declared next use, ordered within its executor only.
    Next { at: u64 },
    /// A chunk of cache region `region`.
    Cache { region: u32 },
}

/// A spillable region (KV, latents, conditioning, a preempted job's state).
#[derive(Clone, Debug)]
pub struct Cache {
    pub id: u32,
    pub spillable: bool,
    pub last_use: u64,
    /// None: no declared use.
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
    /// Prefetch or a float refill: free bytes above 2H only, never a revoke.
    pub speculative: bool,
    /// Adopts this retained region whole: no new bytes.
    pub adopts: Option<usize>,
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
    Grant,
    /// Room is coming; `on`: executors whose pins it counts on.
    Wait { on: Vec<u64> },
    /// Queued behind its executor's earlier ask.
    Behind,
    /// No room can come; `room`: what could, for a cut's sizing.
    Refuse { room: u64 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pick {
    Chunk { executor: u64, slot: u32, seq: u32, bytes: u64, stage: Stage },
    Retained { index: usize, bytes: u64 },
    /// A cache region given back whole; its chunks follow as [`Pick::Chunk`].
    Cache { executor: u64, region: u32, spill: bool },
    /// A graph dropped; the chunks it alone held that nothing needs follow.
    Graph { executor: u64, region: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    Float,
    Free,
    Idle,
    Region,
    Running,
    Watermark,
}

/// Every pick for `asks`, plus the yield's.
pub fn decide(s: &Snapshot, asks: &[Ask]) -> Decision {
    let facts = Facts::of(s);
    let mut run = Run::new(s, &facts);
    run.gone.extend(asks.iter().filter_map(|a| a.adopts));
    let h = i128::from(s.headroom);
    // Where a shrink stops and speculation's growth ends: 2H when shared, H when dedicated.
    let top = if s.shared { 2 * h } else { h };
    let new = |a: &Ask| i128::from(if a.adopts.is_some() { 0 } else { a.bytes });
    let free = i128::from(s.card) - i128::from(s.fixed) - i128::from(s.foreign) - i128::from(s.created);
    if s.new_foreign {
        run.cover(u64::MAX, None, Scope::Idle);
    }
    if free < h {
        let (coming, _, _) = coming(s, None, &BTreeSet::new());
        let short = top - free - i128::from(coming) - i128::from(run.got);
        if short > 0 {
            run.cover(clamp(short), None, Scope::All);
        }
    }
    // Every ask that fits now, then room for each executor's first that does not.
    let (mut answers, mut heads, mut blocked) = (vec![], vec![], BTreeSet::new());
    let mut granted = 0i128;
    for (i, a) in asks.iter().enumerate() {
        let (now, bytes) = (free - granted - h, new(a));
        answers.push(if blocked.contains(&a.executor) {
            Answer::Behind
        } else if a.speculative && bytes > now + h - top {
            Answer::Refuse { room: clamp(now + h - top) }
        } else if bytes <= now {
            granted += bytes;
            Answer::Grant
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
        let projected = free - granted - h - owed + i128::from(coming) + i128::from(run.got);
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

fn clamp(x: i128) -> u64 {
    x.clamp(0, i128::from(u64::MAX)) as u64
}

/// Bytes already coming back (asked, being revoked, released) and whose pins they wait on. An
/// asked chunk still pinned counts unless its executor waits, directly or not, on the asker.
/// Also, for an ask, other executors' grants not yet acked: never asked before their ack (L3),
/// revocable after it.
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
/// chunks, each cache's chunks.
struct Facts {
    pinned_by: Vec<Vec<Vec<usize>>>,
    graph_chunks: Vec<Vec<Vec<usize>>>,
    cache_chunks: Vec<Vec<Vec<usize>>>,
}

impl Facts {
    fn of(s: &Snapshot) -> Facts {
        let mut f = Facts { pinned_by: vec![], graph_chunks: vec![], cache_chunks: vec![] };
        for e in &s.executors {
            let slots: BTreeMap<u32, usize> = e.chunks.iter().enumerate().map(|(i, c)| (c.slot, i)).collect();
            let graphs: Vec<Vec<usize>> = e.graphs.iter().map(|g| g.chunks.iter().filter_map(|s| slots.get(s).copied()).collect()).collect();
            let pinned_by = (0..e.chunks.len()).map(|c| graphs.iter().enumerate().filter(|(_, cs)| cs.contains(&c)).map(|(g, _)| g).collect()).collect();
            let caches = e.caches.iter().map(|k| e.chunks.iter().enumerate().filter(|(_, c)| c.hint == Hint::Cache { region: k.id }).map(|(i, _)| i).collect()).collect();
            f.pinned_by.push(pinned_by);
            f.graph_chunks.push(graphs);
            f.cache_chunks.push(caches);
        }
        f
    }
}

fn at<T>(v: &[Vec<T>], e: usize, i: usize) -> Option<&T> {
    v.get(e).and_then(|x| x.get(i))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Scope {
    /// Stages 1-3 and idle regions: what a new foreign process takes.
    Idle,
    All,
}

/// A region's order: used, restore ns per MiB freed, last use, executor, region.
type Rank = (bool, u128, u64, u64, u32);

/// A water-fill queue: bytes left, its order, executor, chunks.
type Queue = (u64, Reverse<usize>, usize, std::vec::IntoIter<usize>);

enum Unit {
    Cache(usize, usize),
    /// One graph, or every graph holding a shared chunk.
    Graphs(usize, Vec<usize>),
}

/// One decision's picks so far; chunks, graphs and caches by (executor, index).
#[derive(Clone)]
struct Run<'a> {
    s: &'a Snapshot,
    f: &'a Facts,
    taken: BTreeSet<(usize, usize)>,
    dropped: BTreeSet<(usize, usize)>,
    given: BTreeSet<(usize, usize)>,
    /// Retained regions picked or adopted.
    gone: BTreeSet<usize>,
    /// Host bytes left after this decision's spills.
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

    /// Chunks a dropped graph frees outright: its pool's and idle weights'.
    fn credited(&self, e: usize, c: usize) -> bool {
        self.chunk(e, c).is_some_and(|ch| matches!(ch.hint, Hint::Free { .. } | Hint::Idle { .. }))
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

    /// A cache's chunks and bytes, when all of it can go now.
    fn cache_bytes(&self, e: usize, k: usize) -> Option<(&'a [usize], u64)> {
        let chunks = at(&self.f.cache_chunks, e, k)?;
        (!chunks.is_empty() && chunks.iter().all(|&c| self.free_now(e, c))).then(|| (chunks.as_slice(), chunks.iter().map(|&c| self.bytes(e, c)).sum()))
    }

    /// Spill (rather than drop) when it is spillable, RAM keeps 10% free after it, and its
    /// measured round trip beats its measured production (an unmeasured production spills).
    fn spill(&self, cache: &Cache, bytes: u64) -> Option<u64> {
        let h = &self.s.host;
        let ns = |rate: u64| (rate > 0).then(|| (u128::from(bytes) * 1_000_000_000 / u128::from(rate)) as u64);
        let trip = ns(h.d2h)?.saturating_add(ns(h.h2d)?);
        let fits = self.host.saturating_sub(bytes) >= h.total / 10;
        (cache.spillable && fits && (cache.produce_ns == 0 || trip < cache.produce_ns)).then_some(trip)
    }

    /// The next region to give: idle ones least recently used first, then the cheapest restore
    /// per byte freed (unmeasured last).
    fn best_region(&self, scope: Scope) -> Option<Unit> {
        let per_byte = |ns: u64, bytes: u64| u128::from(ns) * (1 << 20) / u128::from(bytes.max(1));
        let mut best: Option<(Rank, Unit)> = None;
        let mut offer = |key, unit| {
            if best.as_ref().is_none_or(|(b, _)| key < *b) {
                best = Some((key, unit));
            }
        };
        for (e, x) in self.s.executors.iter().enumerate() {
            for (k, cache) in x.caches.iter().enumerate() {
                let used = cache.next_use.is_some();
                let Some((_, bytes)) = self.cache_bytes(e, k).filter(|_| !self.given.contains(&(e, k)) && (scope == Scope::All || !used)) else { continue };
                let cost = self.spill(cache, bytes).unwrap_or(if cache.produce_ns > 0 { cache.produce_ns } else { u64::MAX });
                offer((used, if used { per_byte(cost, bytes) } else { 0 }, cache.last_use, x.id, cache.id), Unit::Cache(e, k));
            }
            let live = |g: usize| x.graphs.get(g).is_some_and(|graph| !graph.dropping) && !self.dropped.contains(&(e, g));
            let mut units: BTreeSet<Vec<usize>> = (0..x.graphs.len()).filter(|&g| live(g)).map(|g| vec![g]).collect();
            units.extend((0..x.chunks.len()).filter(|&c| self.credited(e, c) && self.open(e, c)).filter_map(|c| self.needs(e, c)).filter(|n| n.len() > 1));
            for unit in units {
                let graphs: Vec<&Graph> = unit.iter().filter_map(|&g| x.graphs.get(g)).collect();
                let used = graphs.iter().any(|g| g.next_use.is_some());
                let credit = self.graphs_credit(e, &unit);
                if credit == 0 || (scope == Scope::Idle && used) {
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

    fn give_cache(&mut self, e: usize, k: usize) {
        let (Some(x), Some((chunks, bytes))) = (self.ex(e), self.cache_bytes(e, k)) else { return };
        let Some(cache) = x.caches.get(k) else { return };
        let spill = self.spill(cache, bytes).is_some();
        if spill {
            self.host -= bytes;
        }
        self.given.insert((e, k));
        self.picks.push(Pick::Cache { executor: x.id, region: cache.id, spill });
        for &c in chunks {
            self.take(e, c, Stage::Region);
        }
    }

    /// Chunks that can go now whose hint `key` orders (None: not this stage), in key order.
    fn listed(&self, key: impl Fn(usize, &Hint) -> Option<u64>) -> Vec<(usize, usize)> {
        let mut out = vec![];
        for (e, x) in self.s.executors.iter().enumerate() {
            for (c, ch) in x.chunks.iter().enumerate() {
                if let Some(k) = key(e, &ch.hint).filter(|_| self.free_now(e, c)) {
                    out.push(((k, x.id, ch.slot), (e, c)));
                }
            }
        }
        out.sort();
        out.into_iter().map(|(_, at)| at).collect()
    }

    /// A running set's chunks: far ones (unpinned, then a graph's), furthest first, and its
    /// slots (the nearest [`SLOTS`] uses unpinned), furthest first.
    fn running(&self, e: usize) -> (Vec<usize>, Vec<usize>) {
        let Some(x) = self.ex(e) else { return (vec![], vec![]) };
        let mut items = vec![];
        for (c, ch) in x.chunks.iter().enumerate() {
            let at = match ch.hint {
                Hint::Next { at } => at,
                Hint::Unset => 0,
                Hint::Cache { region } if !x.caches.iter().any(|k| k.id == region) => 0,
                _ => continue,
            };
            if let Some(n) = self.needs(e, c).filter(|_| self.open(e, c)) {
                items.push((!n.is_empty(), Reverse(at), ch.slot, c));
            }
        }
        items.sort();
        let near: BTreeSet<u64> = items.iter().filter(|i| !i.0).map(|i| i.1 .0).collect::<BTreeSet<_>>().into_iter().take(SLOTS).collect();
        let (slots, far): (Vec<_>, Vec<_>) = items.into_iter().partition(|i| !i.0 && near.contains(&i.1 .0));
        (far.into_iter().map(|i| i.3).collect(), slots.into_iter().map(|i| i.3).collect())
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

    /// Picks at least `need` more bytes in stage order; false when everything in `scope` went
    /// short of it.
    fn cover(&mut self, need: u64, asker: Option<usize>, scope: Scope) -> bool {
        let target = self.got.saturating_add(need);
        for stage in [Stage::Float, Stage::Free, Stage::Idle] {
            // Each stage's chunks and their order: since, since, last use.
            let key = |e: usize, h: &Hint| match (stage, *h) {
                (Stage::Float, Hint::Float { since }) if Some(e) != asker => Some(since),
                (Stage::Free, Hint::Free { since }) => Some(since),
                (Stage::Idle, Hint::Idle { last_use }) => Some(last_use),
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
                for (index, &bytes) in self.s.retained.iter().enumerate() {
                    if self.got >= target {
                        return true;
                    }
                    if self.gone.insert(index) {
                        self.got += bytes;
                        self.picks.push(Pick::Retained { index, bytes });
                    }
                }
            }
        }
        while self.got < target {
            match self.best_region(scope) {
                Some(Unit::Cache(e, k)) => self.give_cache(e, k),
                Some(Unit::Graphs(e, graphs)) => graphs.into_iter().for_each(|g| self.drop_graph(e, g)),
                None => break,
            }
        }
        if scope == Scope::Idle || self.got >= target {
            return self.got >= target;
        }
        let own = asker.map(|e| (e, self.running(e)));
        let (far, slots): (Vec<_>, Vec<_>) = (0..self.s.executors.len()).filter(|&e| Some(e) != asker).map(|e| (e, self.running(e))).map(|(e, (f, s))| ((e, f), (e, s))).unzip();
        if let Some((e, (own_far, _))) = &own {
            self.water(vec![(*e, own_far.clone())], Stage::Running, target);
        }
        self.water(far, Stage::Running, target);
        if let Some((e, (_, own_slots))) = own {
            self.water(vec![(e, own_slots)], Stage::Watermark, target);
        }
        self.water(slots, Stage::Watermark, target);
        self.got >= target
    }
}

impl Hint {
    /// The lease table slot's `(class, next_use)` words.
    pub fn encode(&self) -> (u32, u64) {
        match *self {
            Hint::Unset => (0, 0),
            Hint::Float { since } => (1, since),
            Hint::Free { since } => (2, since),
            Hint::Idle { last_use } => (3, last_use),
            Hint::Next { at } => (4, at),
            Hint::Cache { region } => (5, u64::from(region)),
        }
    }

    /// The table's words back (untrusted: anything unknown is `Unset`).
    pub fn decode(class: u32, value: u64) -> Hint {
        match class {
            1 => Hint::Float { since: value },
            2 => Hint::Free { since: value },
            3 => Hint::Idle { last_use: value },
            4 => Hint::Next { at: value },
            5 => u32::try_from(value).map_or(Hint::Unset, |region| Hint::Cache { region }),
            _ => Hint::Unset,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]
    use super::*;

    const M: u64 = 1 << 20;

    fn c(slot: u32, mib: u64, hint: Hint) -> Chunk {
        Chunk { slot, seq: slot + 1000, bytes: mib * M, state: State::Mapped, pins: 0, asked: false, hint }
    }

    fn pinned(pins: u32, mut ch: Chunk) -> Chunk {
        ch.pins = pins;
        ch
    }

    fn asked(mut ch: Chunk) -> Chunk {
        ch.asked = true;
        ch
    }

    fn next(slot: u32, at: u64) -> Chunk {
        c(slot, 1, Hint::Next { at })
    }

    fn idle(slot: u32, last_use: u64) -> Chunk {
        c(slot, 1, Hint::Idle { last_use })
    }

    fn ex(id: u64, chunks: Vec<Chunk>) -> Executor {
        Executor { id, chunks, ..Executor::default() }
    }

    fn cache(id: u32, next_use: Option<u64>, last_use: u64, produce_ns: u64) -> Cache {
        Cache { id, spillable: true, last_use, next_use, produce_ns }
    }

    fn graph(id: u32, chunks: &[u32], next_use: Option<u64>, last_use: u64, capture_ns: u64) -> Graph {
        Graph { id, chunks: chunks.to_vec(), last_use, next_use, capture_ns, dropping: false }
    }

    /// A card with `free` MiB uncreated, H `h` MiB.
    fn snap(free: i64, h: u64, retained: Vec<u64>, executors: Vec<Executor>) -> Snapshot {
        let created = executors.iter().flat_map(|e| &e.chunks).map(|c| c.bytes).sum::<u64>() + retained.iter().map(|r| r * M).sum::<u64>();
        let card = (created as i64 + free * M as i64) as u64;
        Snapshot { card, headroom: h * M, shared: true, created, retained: retained.into_iter().map(|r| r * M).collect(), executors, ..Snapshot::default() }
    }

    fn demand(executor: u64, mib: u64) -> Ask {
        Ask { executor, bytes: mib * M, speculative: false, adopts: None }
    }

    fn show(d: &Decision) -> Vec<String> {
        d.picks
            .iter()
            .map(|p| match p {
                Pick::Chunk { executor, slot, .. } => format!("{executor}:{slot}"),
                Pick::Retained { index, .. } => format!("R{index}"),
                Pick::Cache { executor, region, spill } => format!("{}{executor}:{region}", if *spill { "S" } else { "D" }),
                Pick::Graph { executor, region } => format!("G{executor}:{region}"),
            })
            .collect()
    }

    fn bytes(d: &Decision) -> u64 {
        d.picks
            .iter()
            .map(|p| match p {
                Pick::Chunk { bytes, .. } | Pick::Retained { bytes, .. } => *bytes,
                _ => 0,
            })
            .sum()
    }

    /// Executor 1 asks while running; 2 runs beside it; 3 is idle with a cache and a graph.
    fn ladder() -> Snapshot {
        let one = ex(1, vec![c(1, 1, Hint::Float { since: 1 }), pinned(1, next(10, 1)), next(11, 2), next(12, 3), next(13, 7), next(14, 9)]);
        let mut two = ex(2, vec![c(1, 1, Hint::Float { since: 5 }), pinned(1, next(20, 1)), next(21, 2), next(22, 3), next(23, 8), next(24, 6), c(25, 1, Hint::Cache { region: 9 }), c(26, 1, Hint::Cache { region: 9 })]);
        two.caches = vec![cache(9, Some(4), 1, 1_000_000)];
        let mut three = ex(3, vec![c(1, 1, Hint::Free { since: 3 }), idle(2, 50), idle(3, 10), c(4, 1, Hint::Cache { region: 7 }), c(5, 1, Hint::Cache { region: 7 }), pinned(1, idle(6, 20)), pinned(1, c(7, 1, Hint::Free { since: 0 }))]);
        three.caches = vec![cache(7, None, 30, 1_000_000)];
        three.graphs = vec![graph(8, &[6, 7], None, 15, 1_000_000)];
        snap(0, 0, vec![1], vec![one, two, three])
    }

    const LADDER: &[&str] = &[
        "2:1", // floats, never the asker's own (1:1)
        "3:1", "R0", // free arena chunks, then retained
        "3:3", "3:2", // idle, LRU across executors
        "G3:8", "3:6", "3:7", // an idle graph as a unit: its idle weights and pool
        "D3:7", "3:4", "3:5", // an idle cache
        "D2:9", "2:25", "2:26", // a used cache
        "1:14", "1:13", // the asker's own far chunks, furthest first
        "2:23", "2:24", // then another running set's
        "1:12", "1:11", // the watermark: the asker's slots
        "2:22", "2:21", // then the other's; never 1:10, 2:20 (pinned)
    ];

    #[test]
    fn each_stage_runs_only_once_the_cheaper_ones_are_short() {
        // (need MiB, picks taken: a prefix of LADDER)
        let rows: &[(u64, usize)] = &[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 8), (7, 8), (8, 11), (9, 11), (10, 14), (11, 14), (12, 15), (13, 16), (14, 17), (15, 18), (16, 19), (17, 20), (18, 21), (19, 22)];
        for &(need, n) in rows {
            let d = decide(&ladder(), &[demand(1, need)]);
            assert_eq!(show(&d), LADDER[..n], "need {need} MiB");
            // From the graph on, it waits on executor 3 releasing the graph's pins.
            let on = if n >= 6 { vec![3] } else { vec![] };
            assert_eq!(d.answers, vec![Answer::Wait { on }], "need {need} MiB");
        }
        let d = decide(&ladder(), &[demand(1, 20)]);
        assert_eq!(d.answers, vec![Answer::Refuse { room: 19 * M }]);
        assert!(d.picks.is_empty(), "a refused ask evicts nothing");
    }

    #[test]
    fn uncreated_room_grants_with_no_revoke() {
        let mut s = ladder();
        s.card += 4 * M;
        let d = decide(&s, &[demand(1, 3), demand(2, 1), demand(1, 2)]);
        assert_eq!(d.answers, vec![Answer::Grant, Answer::Grant, Answer::Wait { on: vec![] }]);
        assert_eq!(show(&d), ["2:1", "3:1"]);
        let d = decide(&s, &[demand(1, 9), demand(1, 1)]);
        assert_eq!(d.answers[1], Answer::Behind);
    }

    #[test]
    fn yield_keeps_the_h_to_2h_band() {
        // H = 4 MiB; 16 idle MiB. (foreign MiB taken from 8 free, MiB asked back already, MiB picked)
        let rows: &[(u64, u64, u64)] = &[(0, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 5), (8, 0, 8), (10, 0, 10), (5, 2, 3), (8, 8, 0)];
        for &(foreign, coming, picked) in rows {
            let mut chunks: Vec<Chunk> = (0..16).map(|i| idle(i, u64::from(i))).collect();
            chunks.extend((0..coming as u32).map(|i| asked(next(100 + i, 1))));
            let mut s = snap(8, 4, vec![], vec![ex(1, chunks)]);
            s.foreign = foreign * M;
            let d = decide(&s, &[]);
            assert_eq!(bytes(&d), picked * M, "foreign {foreign} MiB, {coming} MiB coming");
            assert!(d.picks.iter().all(|p| matches!(p, Pick::Chunk { stage: Stage::Idle, .. })));
        }
        // Demand grows into free − H; speculation only into free − 2H.
        let spec = |mib| Ask { speculative: true, ..demand(1, mib) };
        let s = snap(10, 4, vec![], vec![ex(1, vec![idle(1, 1)])]);
        assert_eq!(decide(&s, &[demand(1, 6)]).answers, vec![Answer::Grant]);
        assert_eq!(decide(&s, &[spec(2)]).answers, vec![Answer::Grant]);
        let d = decide(&s, &[spec(3)]);
        assert_eq!((d.answers, d.picks), (vec![Answer::Refuse { room: 2 * M }], vec![]));
        assert_eq!(decide(&s, &[demand(1, 6), spec(1)]).answers[1], Answer::Refuse { room: 0 });
    }

    #[test]
    fn a_new_foreign_process_takes_every_idle_chunk_and_nothing_in_use() {
        let mut s = ladder();
        s.card += 100 * M;
        s.new_foreign = true;
        let d = decide(&s, &[]);
        assert_eq!(show(&d), ["1:1", "2:1", "3:1", "R0", "3:3", "3:2", "G3:8", "3:6", "3:7", "D3:7", "3:4", "3:5"]);
        // An ask adopting a retained region keeps it.
        let d = decide(&s, &[Ask { adopts: Some(0), ..demand(1, 1) }]);
        assert!(!show(&d).contains(&"R0".to_string()));
        assert_eq!(d.answers, vec![Answer::Grant]);
    }

    /// 1 asks, running; 2 and 3 run beside it; 4 is idle.
    fn co_resident() -> Snapshot {
        let a = ex(1, vec![pinned(1, next(1, 1)), next(2, 2), next(3, 3), next(5, 5), next(6, 6)]);
        let b = ex(2, vec![pinned(1, next(1, 1)), next(2, 2), next(3, 3), next(7, 7), next(8, 8), next(9, 9)]);
        let c3 = ex(3, vec![pinned(1, next(1, 1)), next(2, 2), next(3, 3), next(4, 4), next(5, 5)]);
        let d = ex(4, vec![idle(1, 7), idle(2, 3)]);
        snap(0, 0, vec![], vec![a, b, c3, d])
    }

    #[test]
    fn co_residents_page_only_once_the_asker_streams() {
        // (need MiB, picks)
        let rows: &[(u64, &[&str])] = &[
            (2, &["4:2", "4:1"]),
            (4, &["4:2", "4:1", "1:6", "1:5"]),
            // Largest running holder first: 2 has 3 far chunks, 3 has 2; ties go to the first.
            (7, &["4:2", "4:1", "1:6", "1:5", "2:9", "2:8", "3:5"]),
            (9, &["4:2", "4:1", "1:6", "1:5", "2:9", "2:8", "3:5", "2:7", "3:4"]),
            (11, &["4:2", "4:1", "1:6", "1:5", "2:9", "2:8", "3:5", "2:7", "3:4", "1:3", "1:2"]),
            (12, &["4:2", "4:1", "1:6", "1:5", "2:9", "2:8", "3:5", "2:7", "3:4", "1:3", "1:2", "2:3"]),
        ];
        for &(need, picks) in rows {
            let d = decide(&co_resident(), &[demand(1, need)]);
            assert_eq!(show(&d), picks, "need {need} MiB");
        }
        // The no-spike bar: while the idle set and the asker's own far chunks cover it, no
        // co-resident running set loses a chunk.
        for need in 1..=4 {
            let d = decide(&co_resident(), &[demand(1, need)]);
            assert!(d.picks.iter().all(|p| matches!(p, Pick::Chunk { executor: 1 | 4, .. })), "need {need} MiB: {:?}", show(&d));
        }
    }

    #[test]
    fn a_running_set_goes_furthest_next_use_first() {
        // Unpinned uses 5,1,9,3,7,2; 0 is bound; 8 is held only by a running graph.
        let mut a = ex(1, vec![pinned(1, next(0, 0)), next(5, 5), next(1, 1), next(9, 9), next(3, 3), next(7, 7), next(2, 2), pinned(1, next(8, 8))]);
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
        let mut a = ex(1, vec![c(1, 2, Hint::Cache { region: 1 }), c(2, 2, Hint::Cache { region: 2 }), c(3, 4, Hint::Cache { region: 3 }), pinned(2, c(4, 1, Hint::Free { since: 0 })), pinned(1, idle(5, 0)), pinned(1, next(6, 6))]);
        // Caches 1 and 2 used: 1 costs 4 ms to make (spill 3.9 ms), 2 costs 50 µs (dropped).
        // Cache 3 has no declared use. Graphs 5 and 6 share the pool chunk 4; 5 holds running 6, 6 holds idle 5.
        a.caches = vec![cache(1, Some(2), 1, 4_000_000), cache(2, Some(2), 1, 50_000), cache(3, None, 9, 0)];
        a.graphs = vec![graph(5, &[4, 6], Some(1), 2, 1_000_000), graph(6, &[4, 5], None, 3, 1_000_000)];
        let mut s = snap(0, 0, vec![], vec![a, ex(2, vec![])]);
        s.host = host;
        // Idle first, LRU: graph 6 (its idle chunk 5; the pool stays held by 5), then cache 3
        // (unmeasured: spilled); then used, cheapest per byte: cache 2 (dropped), graph 5 (the
        // pool, now its alone; 1 ms/MiB), cache 1 (1.95 ms/MiB). Running chunk 6 stays.
        let d = decide(&s, &[demand(2, 10)]);
        assert_eq!(show(&d), ["G1:6", "1:5", "S1:3", "1:3", "D1:2", "1:2", "G1:5", "1:4", "S1:1", "1:1"]);
        // Graph 5 frees the pool only once 6 is gone; with 6 dropping already, 5 credits it.
        s.executors[0].graphs[1].dropping = true;
        s.executors[0].chunks[4].asked = true;
        // Then, once the regions are short, the running chunk graph 5 held.
        let d = decide(&s, &[demand(2, 11)]);
        assert_eq!(show(&d), ["S1:3", "1:3", "D1:2", "1:2", "G1:5", "1:4", "S1:1", "1:1", "1:6"]);
        assert_eq!(d.answers, vec![Answer::Wait { on: vec![1] }]);
    }

    /// Graphs 1 (100 µs to capture) and 2 (1 s) share pool chunk 1: neither frees it alone, so
    /// cache 3 (10 ms/MiB) goes before the pair (1.0001 s/MiB), and the pair before running 3.
    #[test]
    fn graphs_sharing_a_pool_go_together() {
        let mut a = ex(1, vec![pinned(2, c(1, 1, Hint::Free { since: 0 })), c(2, 1, Hint::Cache { region: 3 }), next(3, 1)]);
        a.caches = vec![cache(3, Some(1), 0, 10_000_000)];
        a.graphs = vec![graph(1, &[1], Some(1), 0, 100_000), graph(2, &[1], Some(1), 0, 1_000_000_000)];
        let rows: &[(u64, &[&str])] = &[(1, &["D1:3", "1:2"]), (2, &["D1:3", "1:2", "G1:1", "G1:2", "1:1"]), (3, &["D1:3", "1:2", "G1:1", "G1:2", "1:1", "1:3"])];
        for &(need, picks) in rows {
            let d = decide(&snap(0, 0, vec![], vec![a.clone(), ex(2, vec![])]), &[demand(2, need)]);
            assert_eq!(show(&d), picks, "need {need} MiB");
        }
    }

    #[test]
    fn spill_or_drop_by_measured_cost() {
        let gb = 1 << 30;
        // (spillable, produce ns, d2h = h2d, host available, spilled); 4 MiB round trip at 1 GiB/s ≈ 7.8 ms.
        let rows: &[(bool, u64, u64, u64, bool)] = &[
            (true, 10_000_000, gb, 64 * gb, true),
            (true, 5_000_000, gb, 64 * gb, false),
            (false, 10_000_000, gb, 64 * gb, false),
            (true, 10_000_000, gb, 64 * gb / 10, false), // RAM would go under 10%
            (true, 0, gb, 64 * gb, true),                // unmeasured production keeps its state
            (true, 10_000_000, 0, 64 * gb, false),       // unmeasured copies
        ];
        for &(spillable, produce_ns, rate, available, spilled) in rows {
            let mut a = ex(1, vec![c(1, 4, Hint::Cache { region: 1 })]);
            a.caches = vec![Cache { spillable, ..cache(1, Some(1), 0, produce_ns) }];
            let mut s = snap(0, 0, vec![], vec![a, ex(2, vec![])]);
            s.host = Host { available, total: 64 * gb, d2h: rate, h2d: rate };
            let d = decide(&s, &[demand(2, 4)]);
            assert_eq!(d.picks[0], Pick::Cache { executor: 1, region: 1, spill: spilled }, "{spillable} {produce_ns} {rate} {available}");
        }
    }

    #[test]
    fn pins_wait_or_refuse() {
        // Only the asker's own bound chunk: refused, nothing picked.
        let d = decide(&snap(0, 0, vec![], vec![ex(1, vec![pinned(1, next(1, 1))])]), &[demand(1, 1)]);
        assert_eq!((d.answers, d.picks), (vec![Answer::Refuse { room: 0 }], vec![]));
        // Another's chunk asked while unpinned and pinned since: it comes at its unpin.
        let s = |waits_on: Vec<u64>| {
            let mut b = ex(2, vec![asked(pinned(1, next(1, 1)))]);
            b.waits_on = waits_on;
            snap(0, 0, vec![], vec![ex(1, vec![pinned(1, next(1, 1))]), b])
        };
        assert_eq!(decide(&s(vec![]), &[demand(1, 1)]).answers, vec![Answer::Wait { on: vec![2] }]);
        // Unless that executor waits on the asker's pins: a cycle.
        assert_eq!(decide(&s(vec![1]), &[demand(1, 1)]).answers, vec![Answer::Refuse { room: 0 }]);
        // Another's grant not yet acked is never asked (L3) but makes up the rest: the ask
        // takes what it can now and waits. The yield does not count it.
        let granted = |mut ch: Chunk| {
            ch.state = State::Granted;
            ch
        };
        let s = snap(0, 0, vec![], vec![ex(1, vec![]), ex(2, vec![idle(1, 1), granted(next(2, 1))])]);
        let d = decide(&s, &[demand(1, 2)]);
        assert_eq!((show(&d), d.answers), (vec!["2:1".to_string()], vec![Answer::Wait { on: vec![] }]));
        assert_eq!(decide(&s, &[demand(1, 3)]).answers, vec![Answer::Refuse { room: M }]);
    }

    /// Deterministic random books: no ask may pick a pinned chunk or the asker's float, stages
    /// run in order, a running set loses nothing while cheaper classes cover the ask, and no
    /// pick but the last was needed.
    #[test]
    fn no_running_pick_while_cheaper_ones_cover() {
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
                            let hint = match rand(5) {
                                0 => Hint::Float { since: v },
                                1 => Hint::Free { since: v },
                                2 => Hint::Idle { last_use: v },
                                3 => Hint::Next { at: v },
                                _ => Hint::Unset,
                            };
                            let ch = c(slot, 1 + rand(4), hint);
                            if rand(5) == 0 { pinned(1, ch) } else { ch }
                        })
                        .collect();
                    ex(id, chunks)
                })
                .collect();
            let free = rand(8) as i64;
            let need = 1 + rand(60);
            let s = snap(free, 0, vec![], executors);
            let d = decide(&s, &[demand(1, need)]);
            let chunk = |e: u64, slot: u32| s.executors.iter().find(|x| x.id == e).and_then(|x| x.chunks.iter().find(|c| c.slot == slot)).unwrap();
            let mut stages = vec![];
            for p in &d.picks {
                let Pick::Chunk { executor, slot, stage, .. } = p else { continue };
                let ch = chunk(*executor, *slot);
                assert_eq!(ch.pins, 0, "case {case}: a pinned chunk");
                assert!(!(*executor == 1 && matches!(ch.hint, Hint::Float { .. })), "case {case}: the asker's own float");
                stages.push((*stage, ch.bytes));
            }
            assert!(stages.windows(2).all(|w| w[0].0 <= w[1].0), "case {case}: stages out of order {stages:?}");
            let short = (need - (free as u64).min(need)) * M;
            let cheaper: u64 = stages.iter().filter(|s| s.0 < Stage::Running).map(|s| s.1).sum();
            if stages.iter().any(|s| s.0 >= Stage::Running) {
                assert!(cheaper < short, "case {case}: a running pick with {cheaper} cheaper bytes for {short}");
                let picked: BTreeSet<String> = show(&d).into_iter().collect();
                for e in &s.executors {
                    for ch in e.chunks.iter().filter(|ch| ch.pins == 0) {
                        let cheap = matches!(ch.hint, Hint::Free { .. } | Hint::Idle { .. }) || (e.id != 1 && matches!(ch.hint, Hint::Float { .. }));
                        assert!(!cheap || picked.contains(&format!("{}:{}", e.id, ch.slot)), "case {case}: {}:{} left for a running pick", e.id, ch.slot);
                    }
                }
            }
            if let (Answer::Wait { .. }, Some(last)) = (&d.answers[0], stages.last()) {
                assert!(bytes(&d) - last.1 < short, "case {case}: picked past the need");
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Ev {
        Demand(u64, u64),
        Spec(u64, u64),
        Adopt(u64),
        /// An executor's request ends: its chunks go idle.
        Idle(u64),
        /// An executor exits: its chunks go to retention.
        Exit(u64),
        Foreign(u64),
        Fixed(u64),
    }

    /// A manager applying each decision at once (every ask back released, every grant mapped):
    /// device free = card − fixed − foreign − created.
    struct Sim {
        s: Snapshot,
        clock: u64,
    }

    impl Sim {
        fn new(shared: bool) -> Sim {
            let s = Snapshot { card: 64 * M, headroom: 4 * M, shared, executors: (1..=3).map(|id| ex(id, vec![])).collect(), ..Snapshot::default() };
            Sim { s, clock: 0 }
        }

        fn free(&self) -> i128 {
            i128::from(self.s.card) - i128::from(self.s.fixed) - i128::from(self.s.foreign) - i128::from(self.s.created)
        }

        fn chunks(&mut self, id: u64) -> &mut Vec<Chunk> {
            &mut self.s.executors.iter_mut().find(|e| e.id == id).unwrap().chunks
        }

        /// Serves `asks` until none waits; checks every decision that grants leaves H free.
        fn serve(&mut self, mut asks: Vec<Ask>, trace: &str) {
            for _ in 0..8 {
                let d = decide(&self.s, &asks);
                for p in &d.picks {
                    match *p {
                        Pick::Chunk { executor, slot, bytes, .. } => {
                            self.chunks(executor).retain(|c| c.slot != slot);
                            self.s.created -= bytes;
                        }
                        Pick::Retained { index, bytes } => {
                            self.s.retained[index] = 0;
                            self.s.created -= bytes;
                        }
                        _ => {}
                    }
                }
                let mut granted = false;
                let mut left = vec![];
                for (a, answer) in asks.iter().zip(&d.answers) {
                    match answer {
                        Answer::Grant => {
                            granted = true;
                            self.clock += 1;
                            let slot = self.clock as u32;
                            let bytes = match a.adopts {
                                Some(i) => std::mem::take(&mut self.s.retained[i]),
                                None => {
                                    self.s.created += a.bytes;
                                    a.bytes
                                }
                            };
                            let hint = if a.speculative { Hint::Float { since: self.clock } } else { Hint::Next { at: self.clock } };
                            self.chunks(a.executor).push(Chunk { bytes, ..c(slot, 0, hint) });
                        }
                        Answer::Wait { .. } | Answer::Behind => left.push(*a),
                        Answer::Refuse { .. } => {}
                    }
                }
                if granted {
                    assert!(self.free() >= i128::from(self.s.headroom), "{trace}: a grant left {} MiB free, under H", self.free() >> 20);
                }
                if left.is_empty() {
                    return;
                }
                asks = left;
            }
        }

        fn run(&mut self, events: &[Ev]) {
            for (i, ev) in events.iter().enumerate() {
                let trace = format!("shared {} event {i} {ev:?}", self.s.shared);
                match *ev {
                    Ev::Demand(e, mib) => self.serve(vec![demand(e, mib)], &trace),
                    Ev::Spec(e, mib) => self.serve(vec![Ask { speculative: true, ..demand(e, mib) }], &trace),
                    Ev::Adopt(e) => {
                        if let Some(i) = self.s.retained.iter().position(|&b| b > 0) {
                            self.serve(vec![Ask { adopts: Some(i), ..demand(e, 0) }], &trace);
                        }
                    }
                    Ev::Idle(e) => {
                        let clock = self.clock;
                        for c in self.chunks(e).iter_mut() {
                            c.hint = Hint::Idle { last_use: clock };
                        }
                    }
                    Ev::Exit(e) => {
                        let gone: u64 = std::mem::take(self.chunks(e)).iter().map(|c| c.bytes).sum();
                        self.s.retained.push(gone);
                    }
                    Ev::Foreign(mib) => self.s.foreign = mib * M,
                    Ev::Fixed(mib) => self.s.fixed = mib * M,
                }
                self.serve(vec![], &trace);
                // Whatever moved, once revocable bytes remain the device is back to H.
                let revocable = self.s.retained.iter().sum::<u64>() + self.s.executors.iter().flat_map(|e| &e.chunks).filter(|c| c.pins == 0).map(|c| c.bytes).sum::<u64>();
                assert!(self.free() >= i128::from(self.s.headroom) || revocable == 0, "{trace}: {} MiB free with {revocable} revocable", self.free() >> 20);
            }
        }
    }

    /// Driver headroom (cr-215): no sequence of asks, exits, adoptions, foreign or context
    /// growth leases into H, in either mode. Card 64 MiB, H 4 MiB.
    #[test]
    fn no_decision_sequence_leases_into_h() {
        use Ev::*;
        let rows: &[&[Ev]] = &[
            &[Demand(1, 30), Demand(2, 30), Demand(3, 1), Spec(1, 1)],
            &[Demand(1, 60), Spec(2, 1), Demand(2, 8), Fixed(6), Demand(3, 2)],
            &[Demand(1, 40), Idle(1), Demand(2, 30), Foreign(10), Spec(3, 4), Demand(3, 4)],
            &[Demand(1, 50), Exit(1), Foreign(14), Adopt(2), Demand(2, 10)],
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
                        5 => Adopt(e),
                        6 => Idle(e),
                        7 => Exit(e),
                        8 => Foreign(rand(20)),
                        _ => Fixed(rand(10)),
                    }
                })
                .collect();
            generated.push(events);
        }
        for shared in [true, false] {
            for events in rows.iter().copied().chain(generated.iter().map(Vec::as_slice)) {
                Sim::new(shared).run(events);
            }
        }
    }

    #[test]
    fn hints_round_trip_the_table_words() {
        for h in [Hint::Unset, Hint::Float { since: 7 }, Hint::Free { since: 8 }, Hint::Idle { last_use: 9 }, Hint::Next { at: 10 }, Hint::Cache { region: 11 }] {
            let (class, value) = h.encode();
            assert_eq!(Hint::decode(class, value), h);
        }
        assert_eq!(Hint::decode(5, u64::MAX), Hint::Unset);
        assert_eq!(Hint::decode(99, 1), Hint::Unset);
    }
}

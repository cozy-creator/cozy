//! One process's facts: its sets and their pages, the plans of its requests, their cursors and
//! the measured rooms; plus the other rows' published bytes. Classes are derived here, never
//! stored: [`Table::class`] answers from the facts each time.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use sha2::{Digest, Sha256};
#[cfg(unix)]
pub use tensorfs::io::source::Source;

/// TensorFS's sources are Unix's so far: no set loads on Windows until they exist there.
#[cfg(windows)]
pub enum Source {}

use crate::layout::Layout;

/// How far [`crate::Memory::load`] brings a set: into the machine's RAM tier, or onto the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Ram,
    Gpu,
}

/// What turns a page's stored bytes into its weights.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Fill {
    /// The stored bytes are the weights: a page is filled when its copy lands.
    Stored,
    /// The caller transforms each fresh page (a LoRA bake) and calls [`crate::Pin::filled`].
    /// The string is the transform's identity: the LoRA content digests, scales, bake recipe
    /// and torch/GPU arch. Declared: its pages may be kept and fetched speculatively.
    Declared(String),
    /// A transform whose identity is not known yet: never kept, never fetched speculatively.
    Undeclared,
}

/// One stored part's identity, as the fingerprint covers it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartId {
    pub name: String,
    pub encoding: String,
    pub dtype: String,
}

/// What a filled page's bytes are. `digest` covers the layout digest, each part's stored name,
/// encoding and dtype, the package release (the bake code's identity), the rank shard and the
/// GPU UUID; [`Fingerprint::key`] adds the fill identity, which may change between passes
/// ([`crate::Memory::fill`]). Kept pages are found by the key, so two pages with one key hold the
/// same bytes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Fingerprint {
    pub digest: String,
    pub fill: Fill,
}

impl Fingerprint {
    /// `release` names the package release, `shard` the rank's slice ("0/1" whole), `gpu` the
    /// device's UUID.
    pub fn new(layout: &Layout, parts: &[PartId], fill: Fill, release: &str, shard: &str, gpu: &str) -> Fingerprint {
        let mut h = Sha256::new();
        let mut field = |b: &[u8]| {
            h.update((b.len() as u64).to_le_bytes());
            h.update(b);
        };
        field(layout.digest.as_bytes());
        for p in parts {
            field(p.name.as_bytes());
            field(p.encoding.as_bytes());
            field(p.dtype.as_bytes());
        }
        field(release.as_bytes());
        field(shard.as_bytes());
        field(gpu.as_bytes());
        Fingerprint { digest: hex(h.finalize().as_slice()), fill }
    }

    /// The kept pages' key: the digest and the current fill identity.
    pub fn key(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.digest.as_bytes());
        match &self.fill {
            Fill::Stored => h.update(b"\0stored"),
            Fill::Declared(id) => {
                h.update(b"\0declared\0");
                h.update(id.as_bytes());
            }
            Fill::Undeclared => h.update(b"\0undeclared"),
        }
        hex(h.finalize().as_slice())
    }

    /// Its pages may be kept, and fetched before a declared use.
    pub fn declared(&self) -> bool {
        self.fill != Fill::Undeclared
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A weight set: its layout (one page per region), where its bytes come from, and what they are.
#[derive(Clone)]
pub struct SetSpec {
    /// Unique per package release (`<package>@<release>/<component>`): measured peaks and cut
    /// levels are kept under it.
    pub name: String,
    pub layout: Arc<Layout>,
    pub source: Arc<Source>,
    pub fingerprint: Fingerprint,
}

/// A request's declared use: its steps in order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Plan {
    pub steps: Vec<Step>,
}

/// One step: these sets run together (no order among them), each `passes` times.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    pub sets: Vec<String>,
    pub passes: u32,
}

/// A page: block `.1` of set `.0`.
pub type Key = (u32, u32);

/// The classes the decision procedure orders bytes by, derived from the facts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Class {
    /// Free.
    F,
    /// Kept pages no live process maps.
    O,
    /// Speculative: mapped for a later step before it began.
    S,
    /// No declared use (least recently used first).
    N,
    /// A declared later use (furthest first).
    D,
    /// The current step's working set.
    W,
    /// Pinned: bound, filling, or with a release event pending.
    P,
}

impl Class {
    pub const ALL: [Class; 7] = [Class::F, Class::O, Class::S, Class::N, Class::D, Class::W, Class::P];

    pub fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Page {
    pub bytes: u64,
    pub mapped: bool,
    /// Its bytes are the weights (landed, and transformed when the set has a fill).
    pub filled: bool,
    /// A fill or import is in flight.
    pub filling: bool,
    /// Live pins.
    pub pins: Vec<u64>,
    /// Release events not yet complete.
    pub releasing: Vec<u64>,
    /// (request, step) it was mapped for speculatively.
    pub speculative: Option<(u64, u32)>,
    /// Kept (exported to custody): unmapping it frees its bytes only once custody lets go.
    pub kept: bool,
    /// A logical clock, not time.
    pub last_use: u64,
}

impl Page {
    fn pinned(&self) -> bool {
        self.filling || !self.pins.is_empty() || !self.releasing.is_empty()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Set {
    pub name: String,
    pub pages: Vec<Page>,
    /// The fill identity is declared: pages may be kept and fetched speculatively.
    pub declared: bool,
}

/// A cut site's facts for the current pass's (set, shape).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CutSite {
    pub name: String,
    pub level: u32,
    /// The finest level its op has: never cut beyond.
    pub finest: u32,
    /// Its measured peak at this level.
    pub peak: u64,
    /// A finer level did not lower the pass's allocated peak: not cut further at this shape.
    pub stuck: bool,
    /// A collective's site: its level changes only as the ranks agree at a pass start.
    pub collective: bool,
    /// The pass's measured room one level finer; `None`: unmeasured there.
    pub finer: Option<u64>,
    /// The pass's measured room one level coarser, and the room that level last failed at.
    pub coarser: Option<(u64, Option<u64>)>,
}

/// Pages a streamed (lowered) step holds at once: one bound while the next lands.
pub const SLOTS: u64 = 2;

/// Where a request is in its plan.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cursor {
    pub step: u32,
    /// Passes begun in this step, per set.
    pub passes: BTreeMap<u32, u32>,
    /// The next block per set in the current pass.
    pub next: BTreeMap<u32, u32>,
}

/// The state of one out-of-memory episode (from a pass start to its end).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Episode {
    /// Free bytes were claimed once (the cap was stale).
    pub claimed_free: bool,
    pub parked: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Req {
    pub id: u64,
    pub name: String,
    /// (set indices, passes) per step; a set not loaded has no index and is not in the plan.
    pub steps: Vec<(Vec<u32>, u32)>,
    pub cursor: Cursor,
    /// The current pass's room: measured or bounded (`measured`), else the unmeasured reserve.
    pub room: u64,
    pub measured: bool,
    /// Bytes claimed by `oom` answers during the current pass.
    pub claimed: u64,
    pub starved: bool,
    pub lowered: bool,
    /// Pages fetched speculatively during the current pass.
    pub speculated: BTreeSet<Key>,
    /// The largest known pass room per step (None: unmeasured), for the remaining W.
    pub rooms: Vec<Option<u64>>,
    /// The cut sites of each set's current pass (a joint step runs several).
    pub sites: BTreeMap<u32, Vec<CutSite>>,
    /// Cut levels of op sites outside any pass (the package's own code): the request's memo.
    pub outside: BTreeMap<String, u32>,
    /// Waiting on a child call: the room it gave back, retaken when the wait ends. Meanwhile
    /// its current step's pages are a later use (D) and it has no remaining W.
    pub yielded: Option<u64>,
    pub episode: Episode,
}

impl Req {
    fn step(&self, i: u32) -> Option<&(Vec<u32>, u32)> {
        self.steps.get(i as usize)
    }

    /// How far `key`'s next use is: (steps, passes, blocks) ahead; None when the plan has none.
    /// A yielded request uses nothing now: its current step is one step ahead.
    pub fn distance(&self, key: Key) -> Option<(u32, u32, u32)> {
        let d = self.plan_distance(key)?;
        Some(if self.yielded.is_some() && d.0 == 0 { (1, d.1, d.2) } else { d })
    }

    fn plan_distance(&self, key: Key) -> Option<(u32, u32, u32)> {
        let (set, block) = key;
        let (sets, passes) = self.step(self.cursor.step)?;
        if sets.contains(&set) {
            let begun = self.cursor.passes.get(&set).copied().unwrap_or(0);
            let next = self.cursor.next.get(&set).copied().unwrap_or(0);
            if begun == 0 || block >= next {
                return Some((0, 0, block.saturating_sub(next)));
            }
            if begun < *passes {
                return Some((0, 1, block));
            }
        }
        let later = (self.cursor.step + 1..self.steps.len() as u32).find(|i| self.step(*i).is_some_and(|(s, _)| s.contains(&set)));
        later.map(|i| (i - self.cursor.step, 0, block))
    }
}

/// A cache the caller holds: its bytes are torch's, given back through its shrink callback.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cache {
    pub id: u32,
    pub owner: String,
    pub bytes: u64,
    pub last_use: u64,
}

/// Another process's row, as it published it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Remote {
    /// Its row's offset and generation (a reused offset is another row).
    pub row: u64,
    /// It has a request open: an ask reaches only its pages.
    pub running: bool,
    pub context: u64,
    pub torch: u64,
    /// Of `torch`, what its trim hook and caches give back when it is idle.
    pub caches: u64,
    pub classes: [u64; 7],
}

impl Remote {
    pub fn held(&self) -> u64 {
        self.context + self.torch + self.classes.iter().enumerate().filter(|(c, _)| *c != Class::O.index()).map(|(_, b)| b).sum::<u64>()
    }

    /// What its ask thread can free under `ceiling`.
    pub fn evictable(&self, ceiling: Class) -> u64 {
        let pages: u64 = [Class::S, Class::N, Class::D].iter().filter(|c| **c <= ceiling).filter_map(|c| self.classes.get(c.index())).sum();
        pages + if self.running { 0 } else { self.caches }
    }
}

/// One process's facts on one device, plus what the others published.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Table {
    pub capacity: u64,
    /// Bytes on the device no row accounts for, measured, beyond what the card leaves outside
    /// the table's cap (`capacity`): `capacity - foreign` is min(cap, card - foreign bytes).
    pub foreign: u64,
    /// This process's driver context.
    pub context: u64,
    /// Kept pages no live process maps, on this device (custody's).
    pub orphans: u64,
    /// Host bytes available now, for parking.
    pub headroom: u64,
    /// Bytes of kept pages this process unmapped whose Withdraw custody has not had yet: its
    /// handle still holds them (pinned, P).
    pub pending: u64,
    /// Torch's allocator, as last measured (pass start and end, every OOM).
    pub allocated: u64,
    pub reserved: u64,
    /// Shared mode (a machine its owner also uses): speculation never grows this process's
    /// footprint, so it takes only what it frees of its own.
    pub shared: bool,
    /// Indexed by set id; `None` once unloaded.
    pub sets: Vec<Option<Set>>,
    pub requests: Vec<Req>,
    pub caches: Vec<Cache>,
    pub others: Vec<Remote>,
}

impl Table {
    pub fn page(&self, (set, block): Key) -> Option<&Page> {
        self.sets.get(set as usize)?.as_ref()?.pages.get(block as usize)
    }

    pub fn page_mut(&mut self, (set, block): Key) -> Option<&mut Page> {
        self.sets.get_mut(set as usize)?.as_mut()?.pages.get_mut(block as usize)
    }

    pub fn request(&self, id: u64) -> Option<&Req> {
        self.requests.iter().find(|r| r.id == id)
    }

    pub fn request_mut(&mut self, id: u64) -> Option<&mut Req> {
        self.requests.iter_mut().find(|r| r.id == id)
    }

    /// Every page, mapped or not.
    pub fn pages(&self) -> impl Iterator<Item = (Key, &Page)> {
        self.sets.iter().enumerate().filter_map(|(s, set)| set.as_ref().map(|set| (s as u32, set))).flat_map(|(s, set)| set.pages.iter().enumerate().map(move |(b, p)| ((s, b as u32), p)))
    }

    /// The nearest next use over the open requests.
    pub fn distance(&self, key: Key) -> Option<(u32, u32, u32)> {
        self.requests.iter().filter_map(|r| r.distance(key)).min()
    }

    /// The request whose current step uses `key` next, nearest first.
    pub fn user(&self, key: Key) -> Option<&Req> {
        self.requests.iter().filter(|r| r.distance(key).is_some_and(|d| d.0 == 0)).min_by_key(|r| r.distance(key))
    }

    /// The class of a mapped page; `None` when unmapped.
    pub fn class(&self, key: Key) -> Option<Class> {
        let page = self.page(key).filter(|p| p.mapped)?;
        if page.pinned() {
            return Some(Class::P);
        }
        let distance = self.distance(key);
        if distance.is_some_and(|d| d.0 == 0) {
            return Some(Class::W);
        }
        let ahead = |(r, step): (u64, u32)| self.request(r).is_some_and(|r| r.cursor.step < step);
        if page.speculative.is_some_and(ahead) {
            return Some(Class::S);
        }
        Some(if distance.is_some() { Class::D } else { Class::N })
    }

    /// This process's page bytes by class (withdrawn kept pages custody still holds are P).
    pub fn classes(&self) -> [u64; 7] {
        let mut out = [0u64; 7];
        if let Some(p) = out.get_mut(Class::P.index()) {
            *p = self.pending;
        }
        for (key, page) in self.pages() {
            if let Some(slot) = self.class(key).and_then(|c| out.get_mut(c.index())) {
                *slot += page.bytes;
            }
        }
        out
    }

    pub fn mapped(&self) -> u64 {
        self.pages().filter(|(_, p)| p.mapped).map(|(_, p)| p.bytes).sum()
    }

    /// Torch's bytes as this row counts them: the open requests' rooms and claims, or what the
    /// allocator holds when that is more.
    pub fn torch(&self) -> u64 {
        self.requests.iter().map(|r| r.room + r.claimed).sum::<u64>().max(self.reserved)
    }

    /// The allocator's slack beyond what the rooms need: free once the trim hook runs (F).
    pub fn slack(&self) -> u64 {
        let rooms: u64 = self.requests.iter().map(|r| r.room + r.claimed).sum();
        self.reserved.saturating_sub(rooms.max(self.allocated))
    }

    pub fn held(&self) -> u64 {
        self.context + self.torch() + self.mapped() + self.pending
    }

    fn others_held(&self) -> u64 {
        self.others.iter().map(Remote::held).sum()
    }

    /// F: what no row, orphan or foreign tenant holds.
    pub fn free(&self) -> u64 {
        self.capacity.saturating_sub(self.foreign + self.orphans + self.held() + self.others_held())
    }

    /// What torch's allocator may hold: capacity − context − mapped − other rows − foreign.
    pub fn torch_limit(&self) -> u64 {
        self.capacity.saturating_sub(self.foreign + self.orphans + self.context + self.mapped() + self.others_held())
    }

    /// The bytes of a step's pages.
    fn step_pages(&self, sets: &[u32]) -> u64 {
        sets.iter().filter_map(|s| self.sets.get(*s as usize)?.as_ref()).flat_map(|s| s.pages.iter()).map(|p| p.bytes).sum()
    }

    /// A step's pages as the request holds them: all, or, lowered, every one it holds mapped
    /// plus the streaming slots (never more than all).
    fn held_pages(&self, req: &Req, sets: &[u32]) -> u64 {
        let all = self.step_pages(sets);
        if !req.lowered {
            return all;
        }
        let pages = || sets.iter().filter_map(|s| self.sets.get(*s as usize)?.as_ref()).flat_map(|s| s.pages.iter());
        let largest = pages().map(|p| p.bytes).max().unwrap_or(0);
        let mapped: u64 = pages().filter(|p| p.mapped).map(|p| p.bytes).sum();
        all.min(mapped + largest * SLOTS)
    }

    /// The request's working set now: its current step's pages (as it holds them) plus its
    /// room; all absolute bytes.
    pub fn w(&self, req: &Req) -> u64 {
        self.step_held(req) + req.room + req.claimed
    }

    /// The current step's pages as the request holds them.
    pub fn step_held(&self, req: &Req) -> u64 {
        req.step(req.cursor.step).map_or(0, |(sets, _)| self.held_pages(req, sets))
    }

    /// The largest W left in the request's plan (absolute: each step's pages plus its room as
    /// started from its measured start allocation, the current step's at least its room now);
    /// `None` while a remaining step is unmeasured. A yielded request has none.
    pub fn remaining(&self, req: &Req) -> Option<u64> {
        if req.yielded.is_some() {
            return Some(0);
        }
        let now = req.measured.then_some(req.room + req.claimed);
        let steps = req.steps.iter().zip(&req.rooms).enumerate().skip(req.cursor.step as usize);
        steps
            .map(|(i, ((sets, _), room))| {
                let room = if i == req.cursor.step as usize { (*room).max(now) } else { *room };
                room.map(|room| self.held_pages(req, sets) + room)
            })
            .try_fold(0, |m, w| w.map(|w| m.max(w)))
    }

    /// Pinned bytes outside every open request's current step (counted apart by `fits`).
    pub fn outside(&self) -> u64 {
        let current = |(set, _): Key| self.requests.iter().any(|r| r.step(r.cursor.step).is_some_and(|(s, _)| s.contains(&set)));
        self.pages().filter(|(k, p)| p.mapped && p.pinned() && !current(*k)).map(|(_, p)| p.bytes).sum()
    }
}

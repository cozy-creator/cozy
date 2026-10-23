//! One process's facts: its sets and their pages, the plans of its requests and their cursors;
//! plus the other rows' published bytes. Classes are derived here, never stored.

use std::collections::BTreeMap;
use std::sync::Arc;

use sha2::{Digest, Sha256};
#[cfg(unix)]
pub use tensorfs::io::source::Source;

/// TensorFS's sources are Unix's so far: no set loads on Windows until they exist there.
#[cfg(windows)]
pub enum Source {}

use crate::layout::Layout;

/// How far [`crate::Memory::load`] brings a set: into host RAM (the page cache, read ahead), or
/// onto the GPU.
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
    let digit = |n: u8| char::from(if n < 10 { b'0' + n } else { b'a' + n - 10 });
    bytes.iter().flat_map(|b| [digit(b >> 4), digit(b & 0xf)]).collect()
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

/// H: the device bytes a shared machine leaves to the display and other programs. A shrink
/// leaves `2H` free, and growth stops at `2H`: the H..2H band is the hysteresis, with no timer.
pub fn headroom(card: u64) -> u64 {
    (512 << 20).max(card / 16)
}

/// Blocks ahead of a pass's cursor its prefetch maps (free bytes only): one bound while the next
/// lands.
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Req {
    pub id: u64,
    pub name: String,
    /// (set indices, passes) per step; a set not loaded has no index and is not in the plan.
    pub steps: Vec<(Vec<u32>, u32)>,
    pub cursor: Cursor,
    /// Its dispatch key's recorded peak (the most it held: pages and allocations): what TensorD
    /// admits it against and what it asks peers for at its open and pass starts. `None`:
    /// unmeasured (it asks for every idle byte).
    pub peak: Option<u64>,
    /// Waiting on a child call: its current step is a later use meanwhile, and its peak counts
    /// for everyone's admission but its family's child (`crate::ledger` publishes it).
    pub yielded: bool,
    pub family: u64,
    /// Idle work (a graph capture off the request path): free bytes only, never evicting.
    pub idle: bool,
}

impl Req {
    fn step(&self, i: u32) -> Option<&(Vec<u32>, u32)> {
        self.steps.get(i as usize)
    }

    /// How far `key`'s next use is: (steps, passes, blocks) ahead; None when the plan has none.
    /// A yielded request uses nothing now: its current step is one step ahead.
    pub fn distance(&self, key: Key) -> Option<(u32, u32, u32)> {
        let d = self.plan_distance(key)?;
        Some(if self.yielded && d.0 == 0 { (1, d.1, d.2) } else { d })
    }

    /// The current step's sets are in use until the cursor leaves the step, however many
    /// passes they run (a tiled decode runs one per tile): a block used this pass is next used
    /// by the next pass. Declared passes only order the steps after it.
    fn plan_distance(&self, key: Key) -> Option<(u32, u32, u32)> {
        let (set, block) = key;
        let (sets, _) = self.step(self.cursor.step)?;
        if sets.contains(&set) {
            let begun = self.cursor.passes.get(&set).copied().unwrap_or(0);
            let next = self.cursor.next.get(&set).copied().unwrap_or(0);
            return Some(if begun == 0 || block >= next { (0, 0, block.saturating_sub(next)) } else { (0, 1, block) });
        }
        self.later(set).map(|i| (i, 0, block))
    }

    /// Steps until the first later step that runs `set`.
    fn later(&self, set: u32) -> Option<u32> {
        (self.cursor.step + 1..self.steps.len() as u32).find(|i| self.step(*i).is_some_and(|(s, _)| s.contains(&set))).map(|i| i - self.cursor.step)
    }

    /// Whether `set` is in its current step.
    pub fn runs(&self, set: u32) -> bool {
        self.step(self.cursor.step).is_some_and(|(s, _)| s.contains(&set))
    }
}

/// A cache the caller holds: its bytes are torch's, given back through its shrink callback.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cache {
    pub id: u32,
    pub owner: String,
    /// The components (sets) it serves: its next use is theirs (none: no declared use).
    pub sets: Vec<u32>,
    pub bytes: u64,
    pub last_use: u64,
    /// Holds out (a graph replay's preparation until its GPU completion) and the release
    /// events still pending: held, nothing shrinks it.
    pub holds: u32,
    pub releasing: Vec<u64>,
    /// Its shrinks running, and the bytes it reported meanwhile: applied after them, so a
    /// shrink's return never overwrites a report made during it.
    pub shrinking: u32,
    pub reported: Option<u64>,
}

impl Cache {
    pub fn held(&self) -> bool {
        self.holds > 0 || !self.releasing.is_empty()
    }
}

/// Another process's row, as it published it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Remote {
    /// Its row's offset and generation (a reused offset is another row).
    pub row: u64,
    /// It has a request open.
    pub running: bool,
    pub context: u64,
    pub torch: u64,
    /// Actual unique device backing attributed to this row, including context/native bytes.
    /// Orphans remain on custody's separate counter.
    pub physical: u64,
    /// Of `torch`, its caches' bytes when it is idle (display).
    pub caches: u64,
    pub classes: [u64; 7],
    /// Bytes it is giving back: retiring pages (of P); still held, reachable once they land.
    pub returning: u64,
}

impl Remote {
    pub fn held(&self) -> u64 {
        self.context + self.torch + self.classes.iter().enumerate().filter(|(c, _)| *c != Class::O.index()).map(|(_, b)| b).sum::<u64>()
    }

    /// The pages its ask thread may give back: those no running step of its uses (S, N and D;
    /// of D, it gives what only a request waiting on a child call uses).
    pub fn idle(&self) -> u64 {
        [Class::S, Class::N, Class::D].iter().filter_map(|c| self.classes.get(c.index())).sum()
    }
}

/// One process's facts on one device, plus what the others published.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Table {
    pub capacity: u64,
    /// The driver's headroom (the backend's): never filled, in any mode (cr-215).
    pub driver: u64,
    /// This row's own cap (a standalone opener's limit); the table's capacity is TensorD's.
    pub limit: Option<u64>,
    /// Bytes on the device no row accounts for, measured, beyond what the card leaves outside
    /// the table's cap (`capacity`): `capacity - foreign` is min(cap, card - foreign bytes).
    pub foreign: u64,
    /// Actual unique device backing for this row: context, committed native allocations and
    /// resident VMM backing. No placeholders; retiring aliases count once.
    pub physical: u64,
    /// Unique actual VMM backing, including retired/quarantined backing until ownership ends.
    pub vmm: u64,
    /// Future VMM backing: exactly the mapped Home placeholders with no physical allocation.
    pub mapping: u64,
    /// Unique retiring subset of vmm; ownership-lost quarantines remain vmm but cannot return.
    pub returning: u64,
    /// This process's driver context (its device bytes beyond pages and the hook's segments).
    pub context: u64,
    /// The allocator hook's segments (torch's bytes in varena's pool), exactly, and the room
    /// held for its segments not allocated yet ([`crate::memory::Claim`]).
    pub segments: u64,
    pub claimed: u64,
    /// Kept pages no live process maps, on this device (custody's).
    pub orphans: u64,
    /// Bytes of kept pages this process unmapped whose Withdraw custody has not had yet: its
    /// handle still holds them (pinned, P).
    pub pending: u64,
    /// Bytes of pages being unmapped by the mechanism thread: still this process's on the
    /// device (P to others), already spent on the creates queued after them (not to itself).
    pub inflight: u64,
    /// Own pages pinned only by varena's own short work (a Keep's protect):
    /// evictable again once it ends.
    pub transient: u64,
    /// Shared mode (a machine its owner also uses): prefetch never grows this process's
    /// footprint.
    pub shared: bool,
    /// TensorD samples foreign bytes (`foreign` is its reading): in shared mode the budget then
    /// leaves H to the display and other programs.
    pub sampled: bool,
    /// The card's bytes (H's basis).
    pub card: u64,
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

    pub fn cache_mut(&mut self, id: u32) -> Option<&mut Cache> {
        self.caches.iter_mut().find(|c| c.id == id)
    }

    pub fn pages(&self) -> impl Iterator<Item = (Key, &Page)> {
        self.sets.iter().enumerate().filter_map(|(s, set)| set.as_ref().map(|set| (s as u32, set))).flat_map(|(s, set)| set.pages.iter().enumerate().map(move |(b, p)| ((s, b as u32), p)))
    }

    /// The nearest next use over the open requests.
    pub fn distance(&self, key: Key) -> Option<(u32, u32, u32)> {
        self.requests.iter().filter_map(|r| r.distance(key)).min()
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
            // Unmapped physical backing includes quarantines as well as live retirement.
            // Mapped placeholders are counted by their page's P class below.
            *p = self.vmm.saturating_sub(self.mapped().saturating_sub(self.mapping));
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

    pub fn held(&self) -> u64 {
        self.context + self.segments + self.claimed + self.vmm + self.mapping
    }

    /// Unique backing already retiring, own and others': still charged until its owners finish.
    pub fn returning(&self) -> u64 {
        self.returning.min(self.vmm) + self.transient + self.others.iter().map(|r| r.returning.min(r.classes.get(Class::P.index()).copied().unwrap_or(0))).sum::<u64>()
    }

    fn others_held(&self) -> u64 {
        self.others.iter().map(Remote::held).sum()
    }

    /// Driver usage not owned by a row or custody, charged only past the card's room outside
    /// this cap.
    pub fn foreign_used(&self, card_capacity: u64, driver_used: u64) -> u64 {
        let physical = self.others.iter().fold(self.physical.saturating_add(self.orphans), |held, row| held.saturating_add(row.physical));
        self.foreign_from_known(card_capacity, driver_used, physical)
    }

    /// Foreign attribution with only the physical bytes proved stable across this driver
    /// sample. Dirty/unpublished backing remains an admission charge, never subtraction.
    pub fn foreign_from_known(&self, card_capacity: u64, driver_used: u64, known: u64) -> u64 {
        driver_used.saturating_sub(known).saturating_sub(card_capacity.saturating_sub(self.capacity))
    }

    /// The one budget: the ledger's capacity less foreign bytes, orphans and the other rows,
    /// within this row's own cap.
    pub fn budget(&self) -> u64 {
        let room = self.capacity.saturating_sub(self.foreign + self.orphans + self.others_held() + self.headroom());
        self.limit.map_or(room, |l| room.min(l))
    }

    /// Bytes never filled: the driver's headroom always, and H in shared mode while TensorD
    /// samples foreign bytes (its rule asks below it).
    pub fn headroom(&self) -> u64 {
        let yielding = if self.shared && self.sampled { headroom(self.card) } else { 0 };
        self.driver.max(yielding)
    }

    /// F: the budget less what this row holds.
    pub fn free(&self) -> u64 {
        self.budget().saturating_sub(self.held())
    }

    /// The largest W left: the request's recorded peak; `None` unmeasured. A yielded request
    /// has none (its family's child runs in its place).
    pub fn remaining(&self, req: &Req) -> Option<u64> {
        if req.yielded {
            return Some(0);
        }
        req.peak
    }

    /// Yielded W per family waited for, in a row's slots: the largest families apart, any more
    /// than the slots together under family 0 (counted for every child).
    pub fn yielded_by_family(&self) -> [(u64, u64); crate::decide::YIELDED] {
        let mut by: BTreeMap<u64, u64> = BTreeMap::new();
        for r in self.requests.iter().filter(|r| r.yielded) {
            *by.entry(r.family).or_default() += r.peak.unwrap_or_else(|| self.step_pages(r));
        }
        let mut families: Vec<(u64, u64)> = by.into_iter().collect();
        families.sort_by_key(|(f, w)| (std::cmp::Reverse(*w), *f));
        let mut out = [(0, 0); crate::decide::YIELDED];
        let fits = families.len() <= crate::decide::YIELDED;
        for (i, (f, w)) in families.into_iter().enumerate() {
            match out.get_mut(i) {
                Some(slot) if fits || i + 1 < crate::decide::YIELDED => *slot = (f, w),
                _ => {
                    if let Some(last) = out.last_mut() {
                        *last = (0, last.1 + w);
                    }
                }
            }
        }
        out
    }

    /// The bytes of the request's current step's pages.
    fn step_pages(&self, r: &Req) -> u64 {
        r.steps.get(r.cursor.step as usize).map_or(0, |(sets, _)| sets.iter().filter_map(|s| self.sets.get(*s as usize)?.as_ref()).flat_map(|s| s.pages.iter()).map(|p| p.bytes).sum())
    }

    /// Pinned bytes outside every open request's current step (counted apart by `fits`).
    pub fn outside(&self) -> u64 {
        let current = |(set, _): Key| self.requests.iter().any(|r| r.runs(set));
        self.pages().filter(|(k, p)| p.mapped && p.pinned() && !current(*k)).map(|(_, p)| p.bytes).sum()
    }
}

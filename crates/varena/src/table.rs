//! One process's facts: its sets and their pages, the plans of its requests and their cursors.
//! Classes are derived here, never stored.

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
    /// Waiting on a child call: its current step is a later use meanwhile.
    pub yielded: bool,
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

/// One process's facts on one device.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Table {
    /// Indexed by set id; `None` once unloaded.
    pub sets: Vec<Option<Set>>,
    pub requests: Vec<Req>,
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

    pub fn mapped(&self) -> u64 {
        self.pages().filter(|(_, p)| p.mapped).map(|(_, p)| p.bytes).sum()
    }
}

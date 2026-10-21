//! One executor's memory on one device: the applier of [`crate::decide`]'s actions.

use std::sync::Arc;

use crate::backend::{Backend, Event};
use crate::ledger::Grant;
use crate::table::{Level, Plan, SetSpec};
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
    /// What the driver's context grew by during the pass (workspaces, modules, communicators).
    pub context: u64,
    /// Each cut site's own allocated peak, measured around its op.
    pub sites: Vec<(String, u64)>,
}

/// The answer to an out-of-memory at an op.
#[derive(Clone, Debug)]
pub enum Next {
    /// Pages were unmapped (and claimed for this request) or the watermark lowered: run the op
    /// again under the new [`Memory::torch_limit`].
    Retry,
    /// Run the op one level finer: at this level.
    Cut(u32),
    /// Move idle activations to the host (measured headroom allows it), then run the op again.
    Park,
    Exhausted(Box<Exhausted>),
}

/// Where a bound block's bytes came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// Already mapped and filled.
    Hit,
    /// A kept page imported by fingerprint: no copy.
    Kept,
    Tier,
    Store,
}

/// One bind: the page-reference trace the gate computes Belady MIN from.
#[derive(Clone, Debug, PartialEq, Eq)]
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fills {
    pub bytes: u64,
    pub ns: u64,
}

/// Bytes moved, by source: fill time over bytes against the measured ceiling gives the %.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Moved {
    /// Kept pages imported by fingerprint (no copy; `ns` is the import).
    pub kept: Fills,
    pub tier: Fills,
    pub store: Fills,
    /// Of the above, bytes filled speculatively.
    pub speculative: u64,
}

/// One step of a request.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
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
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub request: String,
    pub moved: Moved,
    /// One entry per bind.
    pub trace: Vec<Ref>,
    pub steps: Vec<StepStats>,
    /// Device-timed time binding streams waited on fills.
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
    pub lowered: bool,
    /// The largest working set a step had: weights plus its passes' rooms.
    pub peak: u64,
    /// Bytes the request still held when it ended (pins, rooms, caches): 0.
    pub leftover: u64,
}

/// One process's memory on one device.
pub struct Memory {
    _inner: Arc<()>,
}

/// A request in flight; dropping it ends it: its pins, rooms and caches go, and the leftover
/// is measured.
pub struct Request {
    _memory: Arc<()>,
}

/// A block bound for use: mapped, and filled once `ready` completes (the binding stream already
/// waits on it). Dropping it releases the block after the binding stream's work so far.
pub struct Pin {
    pub ptr: u64,
    pub len: u64,
    pub ready: Event,
    /// The bytes were just copied: a set with a declared fill runs its transform, then
    /// [`Pin::filled`].
    pub fresh: bool,
}

/// A pass of a set at a shape, its measured room made. Dropping it without [`Pass::end`]
/// records nothing.
pub struct Pass {
    _memory: Arc<()>,
}

impl Memory {
    /// Opens `gpu`'s table through `grant` (TensorD's), or by name without one (standalone),
    /// and claims this process's row on it.
    pub fn open(gpu: GpuId, grant: Option<Grant>, backend: Box<dyn Backend>) -> Result<Memory> {
        let _ = (gpu, grant, backend);
        Err(Error::Internal("not built yet".into()))
    }

    /// Registers a set and brings it to `level`. On every start, first or restart, each page
    /// imports a kept page by fingerprint, else fills from the RAM tier, else from the store.
    pub fn load(&self, set: SetSpec, level: Level) -> Result<SetId> {
        let _ = (set, level);
        Err(Error::Internal("not built yet".into()))
    }

    /// Unmaps the set's pages (each as its pin releases) and forgets it.
    pub fn unload(&self, set: SetId) {
        let _ = set;
    }

    /// Registers bytes the caller holds (torch tensors) that `shrink(want) -> freed` gives back.
    /// A cache whose `owner` is a request id ends with that request.
    pub fn cache(&self, owner: &str, shrink: Box<dyn FnMut(u64) -> u64 + Send>) -> CacheId {
        let _ = (owner, shrink);
        CacheId(0)
    }

    /// The cache now holds `bytes`.
    pub fn cached(&self, cache: CacheId, bytes: u64) {
        let _ = (cache, bytes);
    }

    pub fn uncache(&self, cache: CacheId) {
        let _ = cache;
    }

    /// Opens a request with its plan; Belady runs over the open requests' plans.
    pub fn request(&self, id: &str, plan: Plan) -> Result<Request> {
        let _ = (id, plan);
        Err(Error::Internal("not built yet".into()))
    }

    /// Maps and fills block `block` of `set`; `stream` (a `CUstream`, 0 the default) waits on
    /// the fill in-stream, and the release is ordered after its work.
    pub fn bind(&self, set: SetId, block: u32, stream: usize) -> Result<Pin> {
        let _ = (set, block, stream);
        Err(Error::Internal("not built yet".into()))
    }

    /// Bytes torch's allocator may hold now: capacity − context − mapped − other rows − foreign
    /// bytes. The adapter applies it at every pass start and after every `oom` answer.
    pub fn torch_limit(&self) -> u64 {
        0
    }

    /// The record of the request that ended last.
    pub fn stats(&self) -> Option<Stats> {
        None
    }
}

impl Request {
    /// Starts a pass of `set` at `shape` (dimensions joined by `x`, e.g. `1x81x480x832`), with
    /// torch's bytes now; returns once the pass's measured room is free.
    pub fn pass(&self, set: SetId, shape: &str, torch: Torch) -> Result<Pass> {
        let _ = (set, shape, torch);
        Err(Error::Internal("not built yet".into()))
    }

    /// An op ran out of memory, needing `need` more bytes when the error said; `site` is its
    /// cut site, `None` when the op cannot be cut.
    pub fn oom(&self, need: Option<u64>, site: Option<&str>) -> Next {
        let _ = (need, site);
        Next::Exhausted(Box::new(Exhausted { need: need.unwrap_or(0), site: site.map(str::to_string), stats: Stats::default() }))
    }

    /// The current pass's cut level for op site `site` (0: whole).
    pub fn cut(&self, site: &str) -> u32 {
        let _ = site;
        0
    }

    pub fn stats(&self) -> Stats {
        Stats::default()
    }

    /// Ends the request, returning its record.
    pub fn end(self) -> Stats {
        Stats::default()
    }
}

impl Pin {
    /// The caller's fill transform completed: the block may now be kept.
    pub fn filled(&self) {}
}

impl Pass {
    /// Records what the pass measured.
    pub fn end(self, measured: Measured) {
        let _ = measured;
    }
}

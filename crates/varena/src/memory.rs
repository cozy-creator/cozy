//! Weights on one GPU, for the process computing with them (design memory-varena §5.1): a VA
//! range per weight set, its home regions mapped in chunks, the rest streamed through scratch
//! slots. A `Pinned` keeps its pages mapped and records a release event when it drops; eviction
//! takes only unpinned pages whose fill and release events completed. Copies run on our own
//! stream; consumers wait on events, the host never syncs.

use std::collections::{BTreeMap, VecDeque};
use std::ffi::{c_int, c_void};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
#[cfg(feature = "store")]
use std::os::fd::RawFd;
use std::os::unix::net::UnixStream;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};
use std::thread::JoinHandle;
use std::time::Instant;

#[cfg(feature = "store")]
use tensorfs::Checkpoint;
use crate::cuda::{self, AccessDesc, AllocationProp, CUcontext, CUevent, CUstream, CtxGuard, Handle};
use crate::host::HostMem;
use crate::io::{IoTally, Mode, Source};
use crate::layout::{align_up, Layout, PART_ALIGN, REGION_ALIGN};
use tensorfs::io::{self as byte_io, Copy, Dst, ReadOpts, Src};
use crate::{lock, Error, Result};

#[cfg(test)]
mod tests;

/// Scratch slots kept while a region streams; more exist only while every one is pinned.
const SLOTS: usize = 3;
/// Idle chunks kept after an eviction for the next map; the rest go back to the driver.
const IDLE_KEEP: u64 = 64 << 20;

/// A checked driver call on handles this core made, in its context.
macro_rules! call {
    ($op:literal, $f:ident($($a:expr),* $(,)?)) => {
        // SAFETY: live handles of our context; out-params point at live locals.
        cuda::check($op, unsafe { (cuda::driver()?.$f)($($a),*) }).map_err(Error::from)
    };
}

struct Ev {
    raw: CUevent,
    timing: bool,
    pool: Arc<Events>,
}
unsafe impl Send for Ev {}
unsafe impl Sync for Ev {}
type EvRef = Arc<Ev>;

impl Ev {
    fn done(&self) -> Result<bool> {
        // SAFETY: a live event of our context.
        match unsafe { (cuda::driver()?.event_query)(self.raw) } {
            cuda::SUCCESS => Ok(true),
            cuda::ERROR_NOT_READY => Ok(false),
            rc => cuda::check("cuEventQuery", rc).map(|_| false).map_err(Error::from),
        }
    }

    fn sync(&self) -> Result<()> {
        call!("cuEventSynchronize", event_synchronize(self.raw))
    }

    /// Nanoseconds from `start` to this event, both completed timing events.
    fn since(&self, start: &Ev) -> Result<u64> {
        let mut ms = 0f32;
        call!("cuEventElapsedTime", event_elapsed_time(&mut ms, start.raw, self.raw))?;
        Ok((ms as f64 * 1e6) as u64)
    }
}

impl Drop for Ev {
    fn drop(&mut self) {
        lock(&self.pool.free)[self.timing as usize].push(self.raw);
    }
}

/// Reused events: a dropped one's waits are already enqueued, so reuse is safe.
struct Events {
    ctx: CUcontext,
    free: Mutex<[Vec<CUevent>; 2]>,
}
unsafe impl Send for Events {}
unsafe impl Sync for Events {}

impl Events {
    /// An event recorded on `stream` now.
    fn record(self: &Arc<Self>, stream: CUstream, timing: bool) -> Result<EvRef> {
        let raw = match lock(&self.free)[timing as usize].pop() {
            Some(raw) => raw,
            None => {
                let _c = CtxGuard::enter(self.ctx)?;
                let mut e: CUevent = std::ptr::null_mut();
                call!("cuEventCreate", event_create(&mut e, if timing { cuda::EVENT_DEFAULT } else { cuda::EVENT_DISABLE_TIMING }))?;
                e
            }
        };
        let e = Arc::new(Ev { raw, timing, pool: self.clone() });
        call!("cuEventRecord", event_record(e.raw, stream))?;
        Ok(e)
    }
}

impl Drop for Events {
    fn drop(&mut self) {
        if let Ok(d) = cuda::driver() {
            for e in self.free.get_mut().unwrap().iter_mut().flat_map(|v| v.drain(..)) {
                // SAFETY: our events.
                unsafe { (d.event_destroy)(e) };
            }
        }
    }
}

/// Wait on `ev` in `stream` (enqueued; the host does not wait).
fn wait_in(stream: CUstream, ev: &Ev) -> Result<()> {
    call!("cuStreamWaitEvent", stream_wait_event(stream, ev.raw, 0))
}

/// Keeps the events not yet passed; true when none is left.
fn passed(events: &mut Vec<EvRef>) -> Result<bool> {
    let mut err = None;
    events.retain(|e| match e.done() {
        Ok(done) => !done,
        Err(x) => {
            err.get_or_insert(x);
            true
        }
    });
    err.map_or(Ok(events.is_empty()), Err)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Counters {
    pub binds: u64,
    pub faults: u64,
    pub paged_in: u64,
    pub streamed_bytes: u64,
    pub evicted: u64,
    pub copy_ns: u64,
    pub stall_ns: u64,
    pub map_ns: u64,
}

#[derive(Debug, Clone)]
pub struct SetStats {
    pub name: String,
    pub nbytes: u64,
    pub regions: u32,
    pub home: u64,
    pub resident: u64,
    pub pinned: u64,
    pub pins: u32,
    pub slot_bytes: u64,
    pub step: Counters,
    pub last: Counters,
    pub total: Counters,
}

#[derive(Debug, Clone)]
pub struct GpuStats {
    pub device: i32,
    pub chunk: u64,
    pub steps: u64,
    pub mapped: u64,
    pub idle: u64,
    pub slots: u32,
    pub slot_bytes: u64,
    pub slot_span: u64,
    pub driver_free: u64,
    pub driver_total: u64,
    pub sets: Vec<SetStats>,
}

/// One stored part, `offset` bytes into its region; `role` is `""` for a plain tensor's data.
#[derive(Debug, Clone)]
pub struct Part {
    pub component: String,
    pub key: String,
    pub role: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub region: u32,
    pub offset: u64,
    pub nbytes: u64,
}

/// A registered checkpoint's regions, and what copies read: chunk files or a sealed host layout.
pub struct Weights {
    pub name: String,
    pub layout: Layout,
    pub parts: Vec<Part>,
    source: Source,
    host: Option<HostMem>,
}

#[cfg(feature = "store")]
impl Weights {
    /// Copies come from `host_fd`'s sealed layout if given, else the checkpoint's chunks.
    fn open(name: &str, ck: &Checkpoint, regions: &[Vec<String>], host_fd: Option<RawFd>) -> Result<Weights> {
        let (layout, source) = open_checkpoint(ck, regions, false)?;
        let host = host_fd.map(|fd| HostMem::adopt_sealed(fd, &layout)).transpose()?;
        if host.as_ref().is_some_and(|h| h.window().is_some()) {
            return Err(Error::Invalid("a streamed host layout: register takes a whole sealed layout".into()));
        }
        Ok(Weights { name: name.into(), parts: parts(ck, &layout), layout, source, host })
    }
}

/// Each part of `layout` with its stored dtype and shape.
#[cfg(feature = "store")]
fn parts(ck: &Checkpoint, layout: &Layout) -> Vec<Part> {
    layout.parts.iter().filter_map(|part| {
        let info = ck.stored_part(&part.what)?;
        Some(Part { component: info.component, key: info.key, role: info.role, dtype: info.dtype,
            shape: info.shape, region: part.region,
            offset: part.offset - layout.regions[part.region as usize].offset, nbytes: part.nbytes })
    }).collect()
}

#[derive(Clone)]
enum Filled {
    Ready(EvRef),
    Failed(Error),
    Unknown(Error),
}

/// A failed fill is reclaimable only after its producer stream completed.
struct Fill {
    state: Mutex<Option<Filled>>,
    cv: Condvar,
}

impl Fill {
    fn set(&self, r: Filled) {
        *lock(&self.state) = Some(r);
        self.cv.notify_all();
    }

    /// Until the copies are enqueued: the host waits only for reads that feed them.
    fn wait(&self) -> Result<EvRef> {
        match self.completion() {
            Filled::Ready(ev) => Ok(ev),
            Filled::Failed(e) | Filled::Unknown(e) => Err(e),
        }
    }

    fn completion(&self) -> Filled {
        self.cv.wait_while(lock(&self.state), |s| s.is_none()).unwrap_or_else(PoisonError::into_inner).clone().unwrap()
    }

    fn busy(&self) -> bool {
        matches!(&*lock(&self.state), None | Some(Filled::Unknown(_)))
    }

    /// Nothing of it is in flight on the device.
    fn landed(&self) -> Result<bool> {
        match &*lock(&self.state) {
            Some(Filled::Ready(ready)) => {
                crate::faults::hit("fill-query")?;
                ready.done()
            }
            Some(Filled::Failed(_)) => Ok(true),
            Some(Filled::Unknown(_)) => Ok(false),
            None => Ok(false),
        }
    }
}

struct Job {
    set: usize,
    w: Arc<Weights>,
    region: u32,
    dst: u64,
    waits: Vec<EvRef>,
    fill: Arc<Fill>,
}

/// The copy stream busy from its first job to its last copy landing.
#[derive(Default)]
struct Run {
    copies: Vec<(EvRef, EvRef)>,
}

#[derive(Default)]
struct Jobs {
    q: VecDeque<Job>,
    stop: bool,
}

#[derive(Clone, Copy)]
struct Chunk {
    handle: Handle,
    size: u64,
    /// One region's whole allocation, which another process may hold by fd: never reused.
    shared: bool,
}

/// Chunks mapped at `va`, the fill that put bytes there, and the pins on them.
#[derive(Default)]
struct Mapping {
    va: u64,
    span: u64,
    chunks: Vec<Chunk>,
    fill: Option<Arc<Fill>>,
    /// Filled and not yet bound.
    fresh: bool,
    pins: u32,
    released: Vec<EvRef>,
    /// Kept pages: a socket whose queue holds the allocation's fd, a dup of which TensorD holds,
    /// so the pages outlive this process. Unmapping drains it; only our death leaves it full.
    kept: Option<UnixStream>,
    /// `kept` was handed out (`WeightSet::kept`).
    deposited: bool,
    read_only: bool,
}

impl Mapping {
    fn settled(&self) -> bool {
        self.pins == 0 && self.fill.as_ref().is_none_or(|f| !f.busy())
    }

    /// Nothing pins it and nothing is in flight on it.
    fn quiet(&mut self) -> Result<bool> {
        Ok(self.settled() && self.fill.as_ref().map_or(Ok(true), |f| f.landed())? && passed(&mut self.released)?)
    }

    /// A fill that failed, which nothing pins: it holds no bytes.
    fn dead(&self) -> bool {
        self.pins == 0 && self.fill.as_ref().is_some_and(|f| matches!(*lock(&f.state), Some(Filled::Failed(_))))
    }

    /// A new fill of it, and the releases its copies must wait for.
    fn refill(&mut self) -> (Arc<Fill>, Vec<EvRef>) {
        let fill = Arc::new(Fill { state: Mutex::new(None), cv: Condvar::new() });
        (self.fill, self.fresh) = (Some(fill.clone()), true);
        (fill, std::mem::take(&mut self.released))
    }

    fn pin(&mut self) -> (u64, Arc<Fill>, bool) {
        self.pins += 1;
        (self.va, self.fill.clone().expect("a bound mapping has a fill"), std::mem::take(&mut self.fresh))
    }
}

struct Reg {
    home: bool,
    map: Option<Mapping>,
}

struct Set {
    w: Arc<Weights>,
    /// Every submitted source read until positively completed, independent of slot reassignment.
    pending: Vec<Arc<Fill>>,
    base: u64,
    regs: Vec<Reg>,
    step: Counters,
    last: Counters,
    total: Counters,
    /// Asks take idle sets before warm ones, least recently bound (`used`) first, and never a
    /// busy one (a forward in flight).
    used: u64,
    warm: bool,
    busy: bool,
    /// Home regions map as one shareable allocation each, whose pages can be kept.
    shared: bool,
}

impl Set {
    fn track_fill(&mut self, fill: Arc<Fill>) {
        self.pending.push(fill);
    }

    fn reap_fills(&mut self) -> Result<()> {
        let mut error = None;
        self.pending.retain(|fill| match fill.landed() {
            Ok(done) => !done,
            Err(e) => { error.get_or_insert(e); true }
        });
        error.map_or(Ok(()), Err)
    }

    fn count(&mut self, f: impl Fn(&mut Counters)) {
        f(&mut self.step);
        f(&mut self.total);
    }
}

struct Slot {
    m: Mapping,
    /// (set, region) whose bytes it holds or is filling.
    holds: Option<(usize, u32)>,
    /// Prefetched and not yet bound.
    pending: bool,
    used: u64,
}

/// A GPU-timed interval, read into `field` once both events completed.
struct Timing {
    set: usize,
    step: u64,
    field: fn(&mut Counters) -> &mut u64,
    a: EvRef,
    b: EvRef,
}

#[derive(Default)]
struct State {
    idle: BTreeMap<u64, Vec<Handle>>,
    mapped: u64,
    sets: Vec<Option<Set>>,
    slots: Vec<Option<Slot>>,
    /// The largest streamed region's span: what a new slot maps.
    slot_span: u64,
    clock: u64,
    steps: u64,
    timings: Vec<Timing>,
}

impl State {
    fn close_fences(&self, id: usize) -> Option<(Vec<Arc<Fill>>, Vec<EvRef>)> {
        let set = self.sets.get(id)?.as_ref()?;
        let maps = set.regs.iter().filter_map(|r| r.map.as_ref()).chain(
            self.slots.iter().flatten().filter(|s| s.holds.is_some_and(|(set, _)| set == id)).map(|s| &s.m));
        let mut fills = set.pending.clone();
        let mut releases = vec![];
        for m in maps {
            if let Some(fill) = m.fill.as_ref().filter(|fill| !fills.iter().any(|held| Arc::ptr_eq(held, fill))) {
                fills.push(fill.clone());
            }
            releases.extend(m.released.iter().cloned());
        }
        Some((fills, releases))
    }

    fn set(&mut self, id: usize) -> &mut Set {
        self.sets[id].as_mut().expect("a live weight set")
    }

    fn reg(&mut self, id: usize, r: u32) -> Result<&mut Reg> {
        let regs = &mut self.set(id).regs;
        let n = regs.len();
        regs.get_mut(r as usize).ok_or_else(|| Error::Invalid(format!("region {r} of {n}")))
    }

    fn slot(&mut self, i: usize) -> &mut Slot {
        self.slots[i].as_mut().expect("a live slot")
    }

    fn idle_bytes(&self) -> u64 {
        self.idle.iter().map(|(s, v)| s * v.len() as u64).sum()
    }

    /// An idle chunk of `size`. An emptied size leaves the map, so `keep_idle`'s largest entry
    /// always holds a chunk.
    fn take_idle(&mut self, size: u64) -> Option<Handle> {
        let std::collections::btree_map::Entry::Occupied(mut e) = self.idle.entry(size) else { return None };
        let h = e.get_mut().pop();
        if e.get().is_empty() {
            e.remove();
        }
        h
    }

    fn resolve(&mut self) {
        let (sets, steps) = (&mut self.sets, self.steps);
        self.timings.retain(|t| {
            match t.b.done() {
                Ok(true) => {}
                Ok(false) => return true,
                Err(_) => return false,
            }
            let ns = t.b.since(&t.a).unwrap_or(0);
            if let Some(Some(s)) = sets.get_mut(t.set) {
                *(t.field)(&mut s.total) += ns;
                if t.step == steps {
                    *(t.field)(&mut s.step) += ns;
                } else if t.step + 1 == steps {
                    *(t.field)(&mut s.last) += ns;
                }
            }
            false
        });
    }
}

// ---------------------------------------------------------------- the GPU

#[derive(Default)]
struct Custody {
    error: Option<Error>,
    gpu: Option<Arc<Gpu>>,
    pins: BTreeMap<usize, Arc<WeightSet>>,
    releases: Vec<EvRef>,
}

struct Inner {
    ordinal: i32,
    ctx: CUcontext,
    prop: AllocationProp,
    /// Allocations exportable as POSIX fds, where the device supports them.
    shared: Option<AllocationProp>,
    chunk: u64,
    events: Arc<Events>,
    copy: CUstream,
    tally: IoTally,
    jobs: (Mutex<Jobs>, Condvar),
    st: Mutex<State>,
    owner: Mutex<Weak<Gpu>>,
    custody: Mutex<Custody>,
}
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

/// One GPU of this process: its VA, chunk cache, copy stream and scratch slots.
pub struct Gpu {
    inner: Arc<Inner>,
    mover: Option<JoinHandle<()>>,
}

impl Gpu {
    /// CUDA device `ordinal`, mapping in physical chunks of `chunk` bytes (a multiple of 2 MiB).
    pub fn open(ordinal: i32, chunk: u64) -> Result<Arc<Gpu>> {
        if chunk == 0 || !chunk.is_multiple_of(REGION_ALIGN) {
            return Err(Error::Invalid(format!("chunks of {chunk} bytes: a multiple of {REGION_ALIGN}")));
        }
        let (dev, ctx) = cuda::primary_context(ordinal)?;
        let _c = CtxGuard::enter(ctx)?;
        if cuda::attribute(dev, cuda::ATTR_VMM_SUPPORTED)? != 1 {
            return Err(Error::Invalid(format!("device {ordinal} has no virtual memory management")));
        }
        let prop = AllocationProp::device(dev, cuda::MEM_HANDLE_TYPE_NONE);
        let fds = cuda::attribute(dev, cuda::ATTR_POSIX_FD_SUPPORTED)? == 1;
        let shared = fds.then(|| AllocationProp::device(dev, cuda::MEM_HANDLE_TYPE_POSIX_FD));
        let mut g = 0usize;
        call!("cuMemGetAllocationGranularity", mem_get_allocation_granularity(&mut g, &prop, cuda::MEM_GRANULARITY_MINIMUM))?;
        if g == 0 || !REGION_ALIGN.is_multiple_of(g as u64) {
            return Err(Error::Invalid(format!("device {ordinal}: granularity {g} does not divide {REGION_ALIGN}")));
        }
        let mut s: CUstream = std::ptr::null_mut();
        call!("cuStreamCreate", stream_create(&mut s, cuda::STREAM_NON_BLOCKING))?;
        let inner = Arc::new(Inner {
            ordinal,
            ctx,
            prop,
            shared,
            chunk,
            events: Arc::new(Events { ctx, free: Mutex::default() }),
            copy: s,
            tally: IoTally::default(),
            jobs: Default::default(),
            st: Mutex::default(),
            owner: Mutex::default(),
            custody: Mutex::default(),
        });
        let m = inner.clone();
        let mover = std::thread::Builder::new().name(format!("varena-copy-{ordinal}")).spawn(move || m.mover());
        let gpu = Arc::new(Gpu { inner: inner.clone(), mover: Some(mover.map_err(|e| Error::Io(format!("spawn copy thread: {e}")))?) });
        *lock(&inner.owner) = Arc::downgrade(&gpu);
        Ok(gpu)
    }

    pub fn device(&self) -> i32 {
        self.inner.ordinal
    }

    pub fn chunk(&self) -> u64 {
        self.inner.chunk
    }

    /// `regions` (`tensorfs::raw::regions`) of a checkpoint as one weight set.
    #[cfg(feature = "store")]
    pub fn register(self: &Arc<Self>, name: &str, ck: &Checkpoint, regions: &[Vec<String>], host_fd: Option<RawFd>) -> Result<Arc<WeightSet>> {
        let weights = Arc::new(Weights::open(name, ck, regions, host_fd)?);
        let g = &self.inner;
        g.check()?;
        let _c = CtxGuard::enter(g.ctx)?;
        let mut st = g.lock();
        g.check()?;
        let base = reserve_va(weights.layout.nbytes.max(REGION_ALIGN))?;
        let regs = weights.layout.regions.iter().map(|_| Reg { home: true, map: None }).collect();
        let (w, z) = (weights.clone(), Counters::default());
        st.sets.push(Some(Set { w, pending: vec![], base, regs, step: z, last: z, total: z, used: 0, warm: true, busy: false, shared: false }));
        Ok(Arc::new(WeightSet { gpu: self.clone(), id: st.sets.len() - 1, base, weights }))
    }

    /// A sealed host layout (TensorD's deposit) registered from the regions it carries: no
    /// module needed, so its copy to the GPU (`WeightSet::prefetch`) runs while the module is
    /// still being built. A module whose regions differ (`WeightSet::regions`) cannot use it.
    #[cfg(feature = "store")]
    pub fn adopt(self: &Arc<Self>, name: &str, ck: &Checkpoint, host_fd: RawFd) -> Result<Arc<WeightSet>> {
        self.register(name, ck, &crate::host::sealed_regions(host_fd)?, Some(host_fd))
    }

    pub fn trim(&self) -> Result<u64> {
        let g = &self.inner;
        g.with(|st| {
            g.shed(st, SLOTS)?;
            g.keep_idle(st, 0)
        })
    }

    /// Frees up to `want` for an ask (design §5.2): idle chunks and slots (under ""), then home
    /// pages of sets as `Set::used` orders them, then the slots no bound block uses, which the
    /// next streamed block maps again. Returns what each gave.
    pub fn give(&self, want: u64) -> Result<Vec<(String, u64)>> {
        let trimmed = self.trim()?;
        let g = &self.inner;
        g.with(|st| {
            let (mut gave, mut freed) = (vec![], trimmed);
            if trimmed > 0 {
                gave.push((String::new(), trimmed));
            }
            let live = st.sets.iter().enumerate().filter_map(|(id, s)| Some((id, s.as_ref().filter(|s| !s.busy)?)));
            let mut order: Vec<(bool, u64, usize)> = live.map(|(id, s)| (s.warm, s.used, id)).collect();
            order.sort_unstable();
            for (_, _, id) in order {
                if freed >= want {
                    break;
                }
                let n = g.evict(st, id, None, want - freed)?;
                if n > 0 {
                    gave.push((st.set(id).w.name.clone(), n));
                    freed += n;
                }
            }
            if freed < want {
                let n = g.shed(st, 0)?;
                if n > 0 {
                    gave.push((String::new(), n));
                }
            }
            g.keep_idle(st, IDLE_KEEP)?;
            Ok(gave)
        })
    }

    pub fn step(&self) {
        let mut st = self.inner.lock();
        st.resolve();
        st.steps += 1;
        for s in st.sets.iter_mut().flatten() {
            s.last = std::mem::take(&mut s.step);
        }
    }

    pub fn stats(&self) -> Result<GpuStats> {
        let g = &self.inner;
        g.with(|st| {
            let (mut free, mut total) = (0usize, 0usize);
            call!("cuMemGetInfo", mem_get_info(&mut free, &mut total))?;
            st.resolve();
            let held: Vec<(usize, u32, u64)> = st.slots.iter().flatten().filter_map(|s| s.holds.map(|(set, _)| (set, s.m.pins, s.m.span))).collect();
            let set = |(id, s): (usize, &Set)| {
                let regions = &s.w.layout.regions;
                let sum = |f: &dyn Fn(&Reg) -> bool| s.regs.iter().zip(regions).filter(|(r, _)| f(r)).map(|(_, x)| x.span).sum();
                let mine = || held.iter().filter(move |h| h.0 == id);
                SetStats {
                    name: s.w.name.clone(),
                    nbytes: s.w.layout.nbytes,
                    regions: regions.len() as u32,
                    home: sum(&|r| r.home),
                    resident: sum(&|r| r.map.is_some()),
                    pinned: sum(&|r| r.map.as_ref().is_some_and(|m| m.pins > 0)),
                    pins: s.regs.iter().filter_map(|r| r.map.as_ref()).map(|m| m.pins).sum::<u32>() + mine().map(|h| h.1).sum::<u32>(),
                    slot_bytes: mine().map(|h| h.2).sum(),
                    step: s.step,
                    last: s.last,
                    total: s.total,
                }
            };
            let sets = st.sets.iter().enumerate().filter_map(|(id, s)| Some((id, s.as_ref()?))).map(set).collect();
            Ok(GpuStats {
                device: g.ordinal,
                chunk: g.chunk,
                steps: st.steps,
                mapped: st.mapped,
                idle: st.idle_bytes(),
                slots: st.slots.iter().flatten().count() as u32,
                slot_bytes: st.slots.iter().flatten().map(|s| s.m.span).sum(),
                slot_span: st.slot_span,
                driver_free: free as u64,
                driver_total: total as u64,
                sets,
            })
        })
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        lock(&self.inner.jobs.0).stop = true;
        self.inner.jobs.1.notify_all();
        if let Some(t) = self.mover.take() {
            let _ = t.join();
        }
    }
}

impl Inner {
    fn reap_fills(&self, st: &mut State, id: usize) -> Result<()> {
        st.set(id).reap_fills().map_err(|e| self.poison(e))
    }

    fn check(&self) -> Result<()> {
        lock(&self.custody).error.clone().map_or(Ok(()), Err)
    }

    /// One retained owner graph per failed GPU, not one leak per later accessor.
    fn poison(&self, error: Error) -> Error {
        let (error, first) = {
            let mut c = lock(&self.custody);
            let first = c.error.is_none();
            if first {
                c.error = Some(error);
                c.gpu = lock(&self.owner).upgrade();
            }
            (c.error.as_ref().unwrap().clone(), first)
        };
        if first {
            tracing::warn!("varena: GPU {} completion is unknown; retaining its owners: {error}", self.ordinal);
        }
        error
    }

    fn retain_pin(&self, ws: &Arc<WeightSet>, error: Error) {
        self.poison(error);
        lock(&self.custody).pins.entry(ws.id).or_insert_with(|| ws.clone());
    }

    fn retain_waits(&self, waits: &[EvRef]) {
        let mut c = lock(&self.custody);
        for ev in waits {
            if !c.releases.iter().any(|held| Arc::ptr_eq(held, ev)) {
                c.releases.push(ev.clone());
            }
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        lock(&self.st)
    }

    /// `f` on the books, with our context current.
    fn with<T>(&self, f: impl FnOnce(&mut State) -> Result<T>) -> Result<T> {
        self.check()?;
        let _c = CtxGuard::enter(self.ctx)?;
        let mut st = self.lock();
        self.check()?;
        f(&mut st)
    }

    fn submit(&self, job: Job, urgent: bool) {
        let mut j = lock(&self.jobs.0);
        if urgent { j.q.push_front(job) } else { j.q.push_back(job) }
        self.jobs.1.notify_all();
    }

    /// A queued fill someone now waits for goes first.
    fn hurry(&self, fill: &Arc<Fill>) {
        let mut j = lock(&self.jobs.0);
        if let Some(i) = j.q.iter().position(|job| Arc::ptr_eq(&job.fill, fill)) {
            let job = j.q.remove(i).unwrap();
            j.q.push_front(job);
        }
    }

    /// One fill at a time, so fills of one destination land in the order they were asked.
    fn mover(self: Arc<Self>) {
        if let Ok(d) = cuda::driver() {
            // SAFETY: binding this thread to our retained primary context.
            unsafe { (d.ctx_set_current)(self.ctx) };
        }
        let mut run: Option<Run> = None;
        loop {
            let j = lock(&self.jobs.0);
            if j.q.is_empty() && run.is_some() {
                drop(j);
                run = self.land(run.take().unwrap());
                continue;
            }
            let job = match self.jobs.1.wait_while(j, |j| j.q.is_empty() && !j.stop).unwrap_or_else(PoisonError::into_inner).q.pop_front() {
                Some(job) => job,
                None => return,
            };
            let r = run.get_or_insert_with(Run::default);
            // A panic is this job's failure, never the thread's end.
            let filled = crate::caught(|| self.check().and_then(|_| self.run(&job, r)).map(|(a, b)| {
                r.copies.push((a.clone(), b.clone()));
                let mut st = self.lock();
                let step = st.steps;
                st.timings.push(Timing { set: job.set, step, field: |c| &mut c.copy_ns, a, b: b.clone() });
                b
            }));
            let filled = match filled {
                Ok(ready) => Filled::Ready(ready),
                Err(e) => self.drained(e),
            };
            if matches!(&filled, Filled::Unknown(_)) {
                self.retain_waits(&job.waits);
            }
            job.fill.set(filled);
        }
    }

    /// Retire completed placement events once the queue becomes idle.
    fn land(&self, r: Run) -> Option<Run> {
        if self.check().is_err() {
            return None;
        }
        if let Some((_, last)) = r.copies.last() {
            if let Err(e) = last.sync() {
                self.poison(e);
                return None;
            }
        }
        if !lock(&self.jobs.0).q.is_empty() {
            return Some(r);
        }
        None
    }

    /// A failure is reported once nothing it queued is in flight.
    fn drained(&self, e: Error) -> Filled {
        if let Err(error) = self.check() {
            return Filled::Unknown(error);
        }
        self.failed_fill(e, self.drain_copy())
    }

    fn drain_copy(&self) -> Result<()> {
        self.check()?;
        let _c = CtxGuard::enter(self.ctx)?;
        crate::faults::hit("copy-drain")?;
        call!("cuStreamSynchronize", stream_synchronize(self.copy))
    }

    fn failed_fill(&self, e: Error, drained: Result<()>) -> Filled {
        if let Err(error) = self.check() {
            return Filled::Unknown(error);
        }
        match drained {
            Ok(()) => Filled::Failed(e),
            Err(s) => Filled::Unknown(self.poison(Error::Io(format!("{e}; and the copy stream could not drain: {s}")))),
        }
    }

    fn run(self: &Arc<Self>, job: &Job, _run: &mut Run) -> Result<(EvRef, EvRef)> {
        let s = self.copy;
        for w in &job.waits {
            if let Err(e) = crate::faults::hit("release-wait").and_then(|_| wait_in(s, w)) {
                let completed = crate::faults::hit("release-wait-sync").and_then(|_| job.waits.iter().try_for_each(|ev| ev.sync()));
                return Err(self.failed_dependencies(e, &job.waits, completed));
            }
        }
        let start = self.events.record(s, true)?;
        let (region, layout) = (job.region, job.w.layout.region(job.region)?);
        let dst = Dst::Device { device: self.ordinal, ptr: job.dst, len: layout.nbytes as usize, stream: s as usize };
        // a region its sealer failed or died filling reads the chunks instead
        let ready = |host: &&HostMem| !host.hole(region) && host.wait_ready(region).inspect_err(|e| tracing::warn!("varena: {}: {e}", job.w.name)).is_ok();
        match job.w.host.as_ref().filter(ready) {
            Some(host) if layout.nbytes > 0 => {
                host.pin_region(region, layout.offset, align_up(layout.nbytes, PART_ALIGN));
                // SAFETY: source is complete inside the retained host layout; destination is the
                // placement's exclusive mapped region until this fill is published.
                let ptr = unsafe { host.base().add(layout.offset as usize) };
                unsafe { byte_io::read(&[Copy { src: Src::Host(ptr), dst }], &ReadOpts::default()) }?;
            }
            _ if layout.nbytes > 0 => {
                // SAFETY: layout items are disjoint ranges of this exclusive device mapping.
                unsafe { job.w.source.read_items(&layout.items, layout.offset, dst, 8, Mode::Buffered, &self.tally) }?;
            }
            _ => {}
        }
        crate::faults::hit("fill-record")?;
        Ok((start, self.events.record(s, true)?))
    }

    fn failed_dependencies(&self, error: Error, waits: &[EvRef], completed: Result<()>) -> Error {
        match completed {
            Ok(()) => error,
            Err(e) => {
                self.retain_waits(waits);
                self.poison(Error::Io(format!("consumer wait failed: {error}; prior consumer completion is unknown: {e}")))
            }
        }
    }

    /// A new chunk, releasing idle chunks first if the device is full; None if it still is.
    fn create(&self, st: &mut State, size: u64, prop: &AllocationProp) -> Result<Option<Handle>> {
        let mut h: Handle = 0;
        loop {
            // SAFETY: out-param; device memory of our context's device.
            match unsafe { (cuda::driver()?.mem_create)(&mut h, size as usize, prop, 0) } {
                cuda::SUCCESS => return Ok(Some(h)),
                cuda::ERROR_OUT_OF_MEMORY if st.idle_bytes() > 0 => {
                    self.keep_idle(st, 0)?;
                }
                cuda::ERROR_OUT_OF_MEMORY => return Ok(None),
                rc => return cuda::check("cuMemCreate", rc).map(|_| None).map_err(Error::from),
            }
        }
    }

    /// Maps `span` bytes at `va`, idle chunks first, or as one exportable allocation.
    /// Under pressure, a shareable allocation falls back to ordinary chunks, then 2 MiB pages.
    fn map(&self, st: &mut State, va: u64, span: u64, shared: bool) -> Result<Mapping> {
        let mut m = Mapping { va, ..Default::default() };
        let mut prop = self.shared.filter(|_| shared);
        let mut chunk = self.chunk;
        let mapped = (|| {
            while m.span < span {
                let size = if prop.is_some() { span } else { chunk.min(span - m.span) };
                let idle = if prop.is_none() { st.take_idle(size) } else { None };
                let handle = match idle {
                    Some(h) => h,
                    None => match self.create(st, size, prop.as_ref().unwrap_or(&self.prop))? {
                        Some(h) => h,
                        None if prop.is_some() => {
                            prop = None;
                            continue;
                        }
                        None if size > REGION_ALIGN => {
                            chunk = REGION_ALIGN;
                            continue;
                        }
                        None => return Err(Error::DeviceFull { need: span, deficit: span - m.span }),
                    },
                };
                m.chunks.push(Chunk { handle, size, shared: prop.is_some() });
                call!("cuMemMap", mem_map(va + m.span, size as usize, 0, handle, 0))?;
                m.span += size;
            }
            let desc = AccessDesc { location: self.prop.location, flags: cuda::MEM_ACCESS_PROT_READWRITE };
            call!("cuMemSetAccess", mem_set_access(va, span as usize, &desc, 1))
        })();
        if let Err(e) = mapped {
            if m.span > 0 {
                call!("cuMemUnmap", mem_unmap(va, m.span as usize))?;
            }
            self.give_back(st, m.chunks)?;
            return Err(e);
        }
        st.mapped += span;
        Ok(m)
    }

    /// Imports kept pages `fd` and maps them at `va`; the handle, or nothing mapped.
    fn import(&self, va: u64, span: u64, fd: &OwnedFd) -> Result<Handle> {
        let mut h: Handle = 0;
        call!("cuMemImportFromShareableHandle", mem_import(&mut h, fd.as_raw_fd() as usize as *mut c_void, cuda::MEM_HANDLE_TYPE_POSIX_FD))?;
        let desc = AccessDesc { location: self.prop.location, flags: cuda::MEM_ACCESS_PROT_READ };
        let mapped = call!("cuMemMap", mem_map(va, span as usize, 0, h, 0));
        let access = match &mapped {
            Ok(()) => call!("cuMemSetAccess", mem_set_access(va, span as usize, &desc, 1)),
            Err(e) => Err(e.clone()),
        };
        if let Err(e) = access {
            if mapped.is_ok() {
                call!("cuMemUnmap", mem_unmap(va, span as usize))?;
            }
            call!("cuMemRelease", mem_release(h))?;
            return Err(e);
        }
        Ok(h)
    }

    /// Kept pages mapped at `va` read back what the host layout at `host` holds, at their first,
    /// middle and last bytes: else (a stale or foreign page, a mapping that reads nothing) they
    /// unmap and the region fills from the host. Without a host copy they are taken on trust.
    fn holds(&self, h: Handle, va: u64, span: u64, nbytes: u64, host: Option<usize>) -> Result<Handle> {
        let Some(host) = host.filter(|_| nbytes > 0) else { return Ok(h) };
        let n = nbytes.min(4096);
        let mut got = vec![0u8; n as usize];
        let read = [0, nbytes / 2 / 4096 * 4096, nbytes - n].iter().try_for_each(|&at| {
            call!("cuMemcpyDtoH", memcpy_dtoh(got.as_mut_ptr() as *mut c_void, va + at, n as usize))?;
            // SAFETY: inside the host layout's live mapping; the region is Ready.
            let want = unsafe { std::slice::from_raw_parts((host + at as usize) as *const u8, n as usize) };
            match got == want {
                true => Ok(()),
                false => Err(Error::Invalid(format!("byte {at} differs from the host layout"))),
            }
        });
        if let Err(e) = read {
            call!("cuMemUnmap", mem_unmap(va, span as usize))?;
            call!("cuMemRelease", mem_release(h))?;
            return Err(e);
        }
        Ok(h)
    }

    /// Unmaps a mapping nothing in flight touches; its chunks go idle, its kept pages too.
    fn unmap(&self, st: &mut State, m: Mapping) -> Result<()> {
        call!("cuMemUnmap", mem_unmap(m.va, m.span as usize))?;
        st.mapped -= m.span;
        if let Some(kept) = &m.kept {
            drop(platform::handles::recv_fd(kept)); // empty: TensorD already let it go
        }
        self.give_back(st, m.chunks)
    }

    fn give_back(&self, st: &mut State, chunks: Vec<Chunk>) -> Result<()> {
        for c in chunks {
            match c.shared {
                true => call!("cuMemRelease", mem_release(c.handle))?,
                false => st.idle.entry(c.size).or_default().push(c.handle),
            }
        }
        Ok(())
    }

    /// Unmaps quiet home regions of set `id` (all when None) until `budget`; returns the bytes.
    fn evict(&self, st: &mut State, id: usize, regions: Option<&[u32]>, budget: u64) -> Result<u64> {
        let all: Vec<u32> = (0..st.set(id).regs.len() as u32).collect();
        let mut freed = 0;
        for &r in regions.unwrap_or(&all) {
            if freed >= budget {
                break;
            }
            let reg = st.reg(id, r)?;
            if !reg.map.as_mut().map_or(Ok(false), Mapping::quiet)? {
                continue;
            }
            let m = reg.map.take().unwrap();
            let span = m.span;
            self.unmap(st, m)?;
            st.set(id).count(|c| c.evicted += span);
            freed += span;
        }
        Ok(freed)
    }

    /// Releases idle chunks, largest first, until at most `keep` bytes are idle.
    fn keep_idle(&self, st: &mut State, keep: u64) -> Result<u64> {
        let mut freed = 0;
        while st.idle_bytes() > keep {
            let mut last = st.idle.last_entry().unwrap();
            let (size, h) = (*last.key(), last.get_mut().pop().unwrap());
            if last.get().is_empty() {
                last.remove();
            }
            call!("cuMemRelease", mem_release(h))?;
            freed += size;
        }
        Ok(freed)
    }

    /// Unmaps quiet slots: undersized ones, all once nothing streams, else beyond `keep` live.
    fn shed(&self, st: &mut State, keep: usize) -> Result<u64> {
        let span = st.slot_span;
        let (mut live, mut freed) = (st.slots.iter().flatten().count(), 0);
        for i in 0..st.slots.len() {
            let Some(s) = st.slots[i].as_mut() else { continue };
            if (span == 0 || s.m.span < span || live > keep) && s.m.quiet()? {
                freed += s.m.span;
                self.drop_slot(st, i)?;
                live -= 1;
            }
        }
        Ok(freed)
    }

    fn new_slot(&self, st: &mut State, span: u64) -> Result<usize> {
        for i in 0..st.slots.len() {
            if st.slots[i].as_ref().is_some_and(|s| s.m.span < span) && st.slot(i).m.quiet()? {
                self.drop_slot(st, i)?;
            }
        }
        let va = reserve_va(span)?;
        let m = self.map(st, va, span, false).inspect_err(|_| drop(free_va(va, span)))?;
        let slot = Some(Slot { m, holds: None, pending: false, used: 0 });
        match st.slots.iter().position(Option::is_none) {
            Some(i) => st.slots[i] = slot,
            None => st.slots.push(slot),
        }
        Ok(st.slots.iter().position(|s| s.as_ref().is_some_and(|s| s.m.va == va)).unwrap())
    }

    fn drop_slot(&self, st: &mut State, i: usize) -> Result<()> {
        let m = st.slots[i].take().expect("a live slot").m;
        let (va, span) = (m.va, m.span);
        self.unmap(st, m)?;
        free_va(va, span)
    }

    /// A slot for streamed region `r`: the one holding it, else a settled one (empty, consumed,
    /// then for a bind an unbound prefetch), else a new one; a prefetch adds none past `SLOTS`.
    fn slot_for(&self, st: &mut State, set: usize, r: u32, urgent: bool) -> Result<Option<usize>> {
        let (w, span) = (st.set(set).w.clone(), st.slot_span);
        let x = w.layout.region(r)?;
        let held = st.slots.iter().position(|s| s.as_ref().is_some_and(|s| s.holds == Some((set, r))));
        if let Some(i) = held {
            if !st.slot(i).m.dead() {
                return Ok(Some(i));
            }
        }
        self.reap_fills(st, set)?;
        let i = match held {
            Some(i) => i,
            None => {
                let live = st.slots.iter().enumerate().filter_map(|(i, s)| Some((i, s.as_ref()?)));
                let settled = live.filter(|(_, s)| s.m.settled() && s.m.span >= x.span && (urgent || !s.pending));
                match settled.min_by_key(|(_, s)| (s.holds.is_some(), s.pending, s.used)).map(|(i, _)| i) {
                    Some(i) => i,
                    None if urgent || st.slots.iter().flatten().count() < SLOTS => self.new_slot(st, x.span.max(span))?,
                    None => return Ok(None),
                }
            }
        };
        let clock = st.clock;
        let s = st.slot(i);
        (s.holds, s.pending, s.used) = (Some((set, r)), !urgent, clock);
        let ((fill, waits), dst) = (s.m.refill(), s.m.va);
        st.set(set).count(|c| {
            c.streamed_bytes += x.nbytes;
            c.faults += urgent as u64;
        });
        st.set(set).track_fill(fill.clone());
        self.submit(Job { set, w, region: r, dst, waits, fill }, urgent);
        Ok(Some(i))
    }

    /// Maps region `r` at home and queues its fill.
    fn map_home(&self, st: &mut State, set: usize, r: u32, urgent: bool) -> Result<()> {
        self.reap_fills(st, set)?;
        let (w, base) = (st.set(set).w.clone(), st.set(set).base);
        let x = w.layout.region(r)?;
        let t = Instant::now();
        let shared = st.set(set).shared;
        let mut m = self.map(st, base + x.offset, x.span, shared)?;
        let ns = t.elapsed().as_nanos() as u64;
        let ((fill, waits), dst) = (m.refill(), m.va);
        st.reg(set, r)?.map = Some(m);
        st.set(set).count(|c| {
            c.map_ns += ns;
            c.paged_in += x.nbytes;
            c.faults += urgent as u64;
        });
        st.set(set).track_fill(fill.clone());
        self.submit(Job { set, w, region: r, dst, waits, fill }, urgent);
        Ok(())
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        if self.check().is_err() {
            self.retain_resources();
            return;
        }
        let Ok(d) = crate::faults::hit("inner-drop-driver").and_then(|_| cuda::driver().map_err(Error::from)) else { self.retain_resources(); return };
        let Ok(_c) = CtxGuard::enter(self.ctx) else { self.retain_resources(); return };
        // SAFETY: our live stream in its current context. Drop cannot use the fallible macro.
        if self.check().is_err() || cuda::check("cuStreamSynchronize", unsafe { (d.stream_synchronize)(self.copy) }).is_err() {
            self.retain_resources();
            return;
        }
        let releases_failed = self.st.get_mut().unwrap().slots.iter().flatten().any(|s| s.m.released.iter().any(|e| e.sync().is_err()));
        if releases_failed {
            self.retain_resources();
            return;
        }
        // SAFETY: our stream, drained before anything it wrote to goes.
        let mut st = std::mem::take(self.st.get_mut().unwrap());
        for i in 0..st.slots.len() {
            if st.slots[i].is_some() {
                let _ = self.drop_slot(&mut st, i);
            }
        }
        let _ = self.keep_idle(&mut st, 0);
        // SAFETY: our drained stream.
        unsafe { (d.stream_destroy)(self.copy) };
    }
}

impl Inner {
    /// The final owner cannot manufacture completion from a destructor error. Keep the actual
    /// source/mapping/buffer/event objects until process exit; the primary context lives as long.
    fn retain_resources(&mut self) {
        let state = std::mem::take(self.st.get_mut().unwrap());
        let jobs = std::mem::take(self.jobs.0.get_mut().unwrap());
        std::mem::forget((state, jobs, self.events.clone()));
    }
}

/// One weight set on one GPU. Dropping it waits for its copies and releases, then unmaps it.
pub struct WeightSet {
    gpu: Arc<Gpu>,
    id: usize,
    base: u64,
    pub weights: Arc<Weights>,
}

impl WeightSet {
    pub fn device(&self) -> i32 {
        self.gpu.inner.ordinal
    }

    /// The regions it was registered with.
    pub fn regions(&self) -> &[Vec<String>] {
        &self.weights.layout.names
    }

    /// The watermark: `regions` live at home, the rest stream (a mapped one serves until evicted).
    pub fn home(&self, regions: &[u32]) -> Result<()> {
        self.gpu.inner.check()?;
        let n = self.weights.layout.regions.len() as u32;
        if let Some(r) = regions.iter().find(|&&r| r >= n) {
            return Err(Error::Invalid(format!("region {r} of {n}")));
        }
        let mut st = self.gpu.inner.lock();
        self.gpu.inner.check()?;
        for (r, reg) in st.set(self.id).regs.iter_mut().enumerate() {
            reg.home = regions.contains(&(r as u32));
        }
        let streamed = st.sets.iter().flatten().flat_map(|s| s.regs.iter().zip(&s.w.layout.regions).filter(|(r, _)| !r.home));
        st.slot_span = streamed.map(|(_, x)| x.span).max().unwrap_or(0);
        Ok(())
    }

    /// Region `r` for use on `stream`, at home or in a slot; `stream` waits for the bytes, the
    /// host only for reads that feed a copy.
    pub fn bind(self: &Arc<Self>, region: u32, stream: u64) -> Result<Pinned> {
        let g = &self.gpu.inner;
        g.check()?;
        let _c = CtxGuard::enter(g.ctx)?;
        let (at, ptr, fill, fresh, writable) = {
            let mut st = g.lock();
            g.check()?;
            st.clock += 1;
            if st.timings.len() > 256 {
                st.resolve();
            }
            let clock = st.clock;
            let set = st.set(self.id);
            set.used = clock;
            set.count(|c| c.binds += 1);
            let reg = st.reg(self.id, region)?;
            if reg.map.as_ref().is_some_and(Mapping::dead) {
                let m = reg.map.take().unwrap();
                g.unmap(&mut st, m)?;
            }
            let reg = st.reg(self.id, region)?;
            if reg.map.is_some() || reg.home {
                if reg.map.is_none() {
                    g.map_home(&mut st, self.id, region, true)?;
                }
                let m = st.reg(self.id, region)?.map.as_mut().unwrap();
                let (ptr, fill, fresh) = m.pin();
                (At::Home, ptr, fill, fresh, !m.read_only)
            } else {
                let i = g.slot_for(&mut st, self.id, region, true)?.expect("a bind always gets a slot");
                let s = st.slot(i);
                s.pending = false;
                let (ptr, fill, fresh) = s.m.pin();
                (At::Slot(i), ptr, fill, fresh, !s.m.read_only)
            }
        };
        g.hurry(&fill);
        let ready = fill.wait().inspect_err(|_| self.release(at, region, None))?;
        if let Err(e) = g.check() {
            self.release(at, region, None);
            return Err(e);
        }
        let nbytes = self.weights.layout.regions[region as usize].nbytes;
        let admitted = (|| {
            if !ready.done()? {
                let s = stream as CUstream;
                let a = g.events.record(s, true)?;
                wait_in(s, &ready)?;
                let b = g.events.record(s, true)?;
                let mut st = g.lock();
                let step = st.steps;
                st.timings.push(Timing { set: self.id, step, field: |c| &mut c.stall_ns, a, b });
            }
            g.check()
        })();
        if let Err(e) = admitted {
            // No user received this temporary pin, so no consumer work uses its mapping.
            self.release(at, region, None);
            return Err(e);
        }
        let stream = ConsumerStream::current(stream);
        Ok(Pinned { ws: self.clone(), at, region, ptr, nbytes, fresh, writable, stream, consumers: Mutex::new(vec![stream]), ready })
    }

    /// Start filling region `r` ahead of its bind: at home, or into a free slot. Never waits.
    pub fn prefetch(&self, region: u32) -> Result<()> {
        let g = &self.gpu.inner;
        g.with(|st| {
            let reg = st.reg(self.id, region)?;
            if reg.map.is_none() {
                if reg.home {
                    g.map_home(st, self.id, region, false)?;
                } else {
                    g.slot_for(st, self.id, region, false)?;
                }
            }
            Ok(())
        })
    }

    /// Unmaps quiet home regions (all when None); returns the bytes.
    pub fn evict(&self, regions: Option<&[u32]>) -> Result<u64> {
        let g = &self.gpu.inner;
        g.with(|st| {
            let freed = g.evict(st, self.id, regions, u64::MAX)?;
            g.keep_idle(st, IDLE_KEEP)?;
            Ok(freed)
        })
    }

    /// Region `r` is mapped at home with bytes there or on their way.
    pub fn resident(&self, region: u32) -> Result<bool> {
        Ok(self.gpu.inner.lock().reg(self.id, region)?.map.as_ref().is_some_and(|m| !m.dead()))
    }

    pub fn warm(&self, warm: bool) {
        self.gpu.inner.lock().set(self.id).warm = warm;
    }

    /// Its home regions map from now on as one shareable allocation each, so their pages can be
    /// kept (`kept`) past this process. False where the device cannot share them.
    pub fn shareable(&self) -> bool {
        let g = &self.gpu.inner;
        let ok = g.shared.is_some();
        g.lock().set(self.id).shared = ok;
        ok
    }

    /// Home regions whose pages landed and were not handed out yet, each kept for TensorD:
    /// `(region, span, container)`, a socket whose queue holds the allocation's fd. The pages
    /// outlive this process while TensorD holds the container; unmapping them here drains it.
    pub fn kept(&self) -> Result<Vec<(u32, u64, OwnedFd)>> {
        self.keep(false)
    }

    /// Exports after the caller has stopped admission and drained every use of these weights.
    /// Lifetime address pins (cached graphs) may remain; no work may read or write them now.
    /// Their addresses and pins stay intact, and incomplete copies still cannot be exported.
    pub fn kept_quiescent(&self) -> Result<Vec<(u32, u64, OwnedFd)>> {
        self.keep(true)
    }

    fn keep(&self, quiescent: bool) -> Result<Vec<(u32, u64, OwnedFd)>> {
        let g = &self.gpu.inner;
        g.with(|st| {
            let mut out = vec![];
            for (r, reg) in st.set(self.id).regs.iter_mut().enumerate() {
                let Some(m) = reg.map.as_mut().filter(|m| m.chunks.len() == 1 && m.chunks[0].shared && !m.deposited) else { continue };
                let ready = if quiescent {
                    m.fill.as_ref().map_or(Ok(false), |f| f.landed())? && passed(&mut m.released)?
                } else {
                    m.quiet()?
                };
                if !ready {
                    continue;
                }
                let filled = m.fill.as_ref().and_then(|f| lock(&f.state).clone());
                let landed = match filled {
                    Some(Filled::Ready(ev)) => ev.done()?,
                    _ => false,
                };
                if m.kept.is_none() && landed {
                    let desc = AccessDesc { location: g.prop.location, flags: cuda::MEM_ACCESS_PROT_READ };
                    call!("cuMemSetAccess", mem_set_access(m.va, m.span as usize, &desc, 1))?;
                    m.read_only = true;
                    let mut fd: c_int = -1;
                    call!("cuMemExportToShareableHandle", mem_export(&mut fd as *mut c_int as *mut c_void, m.chunks[0].handle, cuda::MEM_HANDLE_TYPE_POSIX_FD, 0))?;
                    // SAFETY: the driver just opened this descriptor for us.
                    m.kept = Some(contain(unsafe { OwnedFd::from_raw_fd(fd) })?);
                }
                if let Some(kept) = &m.kept {
                    let dup = kept.try_clone().map_err(|e| Error::Io(format!("dup a kept container: {e}")))?;
                    out.push((r as u32, m.span, dup.into()));
                    m.deposited = true;
                }
            }
            Ok(out)
        })
    }

    /// Home regions mapped from pages another process kept (`kept`'s fds, taken out of their
    /// containers): imported and mapped, no copy. A page that will not import or map is skipped
    /// and fills as usual. Returns the bytes adopted.
    pub fn adopt_pages(&self, pages: Vec<(u32, OwnedFd)>) -> Result<u64> {
        let g = &self.gpu.inner;
        g.with(|st| {
            let (base, w, mut adopted) = (st.set(self.id).base, st.set(self.id).w.clone(), 0);
            for (r, fd) in pages {
                let x = w.layout.region(r)?;
                if !st.set(self.id).shared || st.reg(self.id, r)?.map.is_some() || !st.reg(self.id, r)?.home {
                    continue;
                }
                let host = w.host.as_ref().filter(|m| m.ready(r)).map(|m| m.base() as usize + x.offset as usize);
                let va = base + x.offset;
                match g.import(va, x.span, &fd).and_then(|h| g.holds(h, va, x.span, x.nbytes, host)) {
                    Ok(handle) => {
                        let ready = match g.events.record(g.copy, false) {
                            Ok(ev) => Filled::Ready(ev),
                            Err(e) => g.drained(e),
                        };
                        let fill = Arc::new(Fill { state: Mutex::new(Some(ready)), cv: Condvar::new() });
                        let chunks = vec![Chunk { handle, size: x.span, shared: true }];
                        let m = Mapping { va: base + x.offset, span: x.span, chunks, fill: Some(fill), fresh: true, kept: Some(contain(fd)?), read_only: true, ..Default::default() };
                        st.reg(self.id, r)?.map = Some(m);
                        st.mapped += x.span;
                        adopted += x.nbytes;
                    }
                    Err(e) => tracing::warn!("varena: kept region {r} of {} did not map, it fills: {e}", w.name),
                }
            }
            Ok(adopted)
        })
    }

    pub fn busy(&self, busy: bool) {
        self.gpu.inner.lock().set(self.id).busy = busy;
    }

    fn release(&self, at: At, region: u32, ev: Option<EvRef>) {
        self.release_events(at, region, ev.into_iter().collect());
    }

    fn release_events(&self, at: At, region: u32, events: Vec<EvRef>) {
        let mut st = self.gpu.inner.lock();
        let m = match at {
            At::Home => match st.reg(self.id, region) {
                Ok(Reg { map: Some(m), .. }) => m,
                _ => return,
            },
            At::Slot(i) => &mut st.slot(i).m,
        };
        m.pins -= 1;
        if !events.is_empty() {
            let _ = passed(&mut m.released);
            m.released.extend(events);
        }
    }

    fn finish_pin(self: &Arc<Self>, at: At, region: u32, events: Vec<EvRef>, failed: Option<Error>, drain: impl FnOnce() -> Result<()>) {
        if let Some(record) = failed {
            if let Err(e) = drain() {
                // Successful partial records remain owned until every consumer is fenced.
                self.gpu.inner.retain_waits(&events);
                self.gpu.inner.retain_pin(self, Error::Io(format!("release event failed: {record}; consumer completion is unknown: {e}")));
                return;
            }
            self.release(at, region, None);
        } else {
            self.release_events(at, region, events);
        }
    }

    fn unregister_host(&self) -> bool {
        let released = crate::faults::hit("set-unregister").and_then(|_| self.weights.host.as_ref().map_or(Ok(()), HostMem::unpin_all));
        match released {
            Ok(()) => true,
            Err(e) => { self.gpu.inner.poison(e); false }
        }
    }
}

impl Drop for WeightSet {
    fn drop(&mut self) {
        let g = &self.gpu.inner;
        if g.check().is_err() {
            return;
        }
        let _c = match crate::faults::hit("set-drop-context").and_then(|_| CtxGuard::enter(g.ctx).map_err(Error::from)) {
            Ok(c) => c,
            Err(e) => { g.poison(e); return; }
        };
        // Leave the source and mappings in State until every copy and consumer fence is proven.
        let Some((fills, releases)) = g.lock().close_fences(self.id) else { return };
        for fill in fills {
            let completed = match fill.completion() {
                Filled::Ready(ev) => ev.sync(),
                Filled::Failed(_) => Ok(()),
                Filled::Unknown(e) => Err(e),
            };
            if let Err(e) = completed {
                g.poison(e);
                return;
            }
        }
        for release in releases {
            if let Err(e) = release.sync() {
                g.poison(e);
                return;
            }
        }
        if g.check().is_err() {
            return;
        }
        if !self.unregister_host() {
            return;
        }
        let set = {
            let mut st = g.lock();
            if g.check().is_err() {
                return;
            }
            for s in st.slots.iter_mut().flatten().filter(|s| s.holds.is_some_and(|(set, _)| set == self.id)) {
                s.holds = None;
            }
            st.sets[self.id].take()
        };
        let Some(set) = set else { return };
        let maps: Vec<Mapping> = set.regs.into_iter().filter_map(|r| r.map).collect();
        let mut st = g.lock();
        for m in maps {
            let _ = g.unmap(&mut st, m);
        }
        let _ = g.keep_idle(&mut st, IDLE_KEEP);
        let _ = free_va(self.base, self.weights.layout.nbytes.max(REGION_ALIGN));
    }
}

#[derive(Clone, Copy)]
enum At {
    Home,
    Slot(usize),
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ConsumerStream {
    raw: u64,
    /// CUDA's per-thread default handle means a different stream on another host thread.
    origin: Option<std::thread::ThreadId>,
}

impl ConsumerStream {
    fn current(raw: u64) -> Self {
        Self { raw, origin: (raw == 2).then(|| std::thread::current().id()) }
    }

    fn here(self) -> Result<()> {
        if self.origin.is_some_and(|origin| origin != std::thread::current().id()) {
            return Err(Error::Io("per-thread consumer stream completion is unknown on another host thread".into()));
        }
        Ok(())
    }
}

/// A bound region: its pages stay mapped while it lives; dropping it fences every admitted stream.
pub struct Pinned {
    ws: Arc<WeightSet>,
    at: At,
    pub region: u32,
    pub ptr: u64,
    pub nbytes: u64,
    /// This bind is the first since a copy put the bytes there.
    pub fresh: bool,
    pub writable: bool,
    stream: ConsumerStream,
    consumers: Mutex<Vec<ConsumerStream>>,
    ready: EvRef,
}

impl Pinned {
    pub fn home(&self) -> bool {
        matches!(self.at, At::Home)
    }

    pub fn device(&self) -> i32 {
        self.ws.device()
    }

    /// Make `stream` wait for the bytes too (a consumer other than the bind stream).
    pub fn wait_on(&self, stream: u64) -> Result<()> {
        self.admit_stream(stream, || {
            let _c = CtxGuard::enter(self.ws.gpu.inner.ctx)?;
            self.ready.done()
        }, || {
            let _c = CtxGuard::enter(self.ws.gpu.inner.ctx)?;
            wait_in(stream as CUstream, &self.ready)
        })
    }

    fn admit_stream(&self, stream: u64, ready: impl FnOnce() -> Result<bool>, wait: impl FnOnce() -> Result<()>) -> Result<()> {
        let g = &self.ws.gpu.inner;
        g.check()?;
        // Serialize admission through its wait. Completion of the fill does not complete a
        // consumer's later work, so even the ready fast path must retain its stream.
        let mut consumers = lock(&self.consumers);
        g.check()?;
        let stream = ConsumerStream::current(stream);
        if !consumers.contains(&stream) {
            consumers.push(stream);
        }
        if stream != self.stream && !ready()? {
            wait()?;
        }
        g.check()
    }

    fn finish_streams(&self, mut record: impl FnMut(u64) -> Result<EvRef>, mut drain: impl FnMut(u64) -> Result<()>) {
        let streams = lock(&self.consumers).clone();
        let mut events = vec![];
        let failed = streams.iter().try_for_each(|&stream| stream.here().and_then(|_| record(stream.raw)).map(|ev| events.push(ev))).err();
        self.ws.finish_pin(self.at, self.region, events, failed, || {
            let mut failed = None;
            for stream in streams {
                if let Err(e) = stream.here().and_then(|_| drain(stream.raw)) {
                    failed.get_or_insert(e);
                }
            }
            failed.map_or(Ok(()), Err)
        });
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        let g = &self.ws.gpu.inner;
        if let Err(e) = g.check() {
            g.retain_pin(&self.ws, e);
            return;
        }
        self.finish_streams(|stream| {
            let _c = crate::faults::hit("pin-release-context").and_then(|_| CtxGuard::enter(g.ctx).map_err(Error::from))?;
            crate::faults::hit("pin-release")?;
            g.events.record(stream as CUstream, false)
        }, |stream| {
            crate::faults::hit("pin-release-drain")?;
            let _c = CtxGuard::enter(g.ctx)?;
            call!("cuStreamSynchronize", stream_synchronize(stream as CUstream))
        });
    }
}

/// The host-only core, for CPU executors: regions as host bytes, never loading libcuda.
pub struct Host;

/// Regions in host memory: read-only views of a sealed layout, or private copies read once.
pub struct HostSet {
    pub weights: Weights,
    regs: Mutex<Vec<(Option<Arc<Anon>>, bool)>>,
}

impl Host {
    #[cfg(feature = "store")]
    pub fn register(&self, name: &str, ck: &Checkpoint, regions: &[Vec<String>], host_fd: Option<RawFd>) -> Result<Arc<HostSet>> {
        let weights = Weights::open(name, ck, regions, host_fd)?;
        Ok(Arc::new(HostSet { regs: Mutex::new(vec![(None, false); weights.layout.regions.len()]), weights }))
    }

    /// `Gpu::adopt` on the host: a sealed layout registered from the regions it carries.
    #[cfg(feature = "store")]
    pub fn adopt(&self, name: &str, ck: &Checkpoint, host_fd: RawFd) -> Result<Arc<HostSet>> {
        self.register(name, ck, &crate::host::sealed_regions(host_fd)?, Some(host_fd))
    }
}

impl HostSet {
    pub fn bind(self: &Arc<Self>, region: u32) -> Result<HostPinned> {
        let (w, mut regs) = (&self.weights, lock(&self.regs));
        let reg = w.layout.region(region)?;
        let (buf, bound) = &mut regs[region as usize];
        let sealed = w.host.as_ref().filter(|h| !h.hole(region) && h.wait_ready(region).is_ok());
        let (ptr, buf) = match sealed {
            Some(h) => {
                (h.base() as u64 + reg.offset, None)
            }
            None => {
                if buf.is_none() {
                    *buf = Some(Arc::new(Anon::read(w, region)?));
                }
                (buf.as_ref().unwrap().ptr as u64, buf.clone())
            }
        };
        let fresh = !std::mem::replace(bound, true);
        Ok(HostPinned { _set: self.clone(), _buf: buf, region, ptr, nbytes: reg.nbytes, fresh, writable: sealed.is_none() })
    }
}

/// A region read from the chunk files into private anonymous memory.
struct Anon {
    ptr: *mut u8,
    len: usize,
}
unsafe impl Send for Anon {}
unsafe impl Sync for Anon {}

impl Anon {
    fn read(w: &Weights, r: u32) -> Result<Anon> {
        let reg = w.layout.region(r)?;
        let len = align_up(reg.nbytes.max(1), PART_ALIGN) as usize;
        // SAFETY: a fresh private anonymous mapping.
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0) };
        if ptr == libc::MAP_FAILED {
            return Err(Error::Io(format!("map {len} bytes: {}", std::io::Error::last_os_error())));
        }
        let a = Anon { ptr: ptr as *mut u8, len };
        // SAFETY: all items lie in this fresh exclusive mapping.
        unsafe { w.source.read_items(&reg.items, reg.offset,
            Dst::Host { ptr: a.ptr, len }, 8, Mode::Buffered, &IoTally::default()) }?;
        Ok(a)
    }
}

impl Drop for Anon {
    fn drop(&mut self) {
        // SAFETY: our mapping, no view of it left.
        unsafe { libc::munmap(self.ptr as *mut c_void, self.len) };
    }
}

/// A region's host bytes; they live as long as this does.
pub struct HostPinned {
    _set: Arc<HostSet>,
    _buf: Option<Arc<Anon>>,
    pub region: u32,
    pub ptr: u64,
    pub nbytes: u64,
    pub fresh: bool,
    /// False for a view of a sealed host layout, mapped read-only.
    pub writable: bool,
}

/// The one layout of `regions` of `ck` and its chunks: what `register` maps and a sealed host
/// layout holds, the same by construction.
#[cfg(feature = "store")]
fn open_checkpoint(ck: &Checkpoint, regions: &[Vec<String>], direct_io: bool) -> Result<(Layout, Source)> {
    Ok((Layout::build(&ck.spans(regions)?, regions)?, ck.source(direct_io)))
}

/// `regions` of `ck` in a sealed host layout filled here: the memfd `register(host_fd=)` adopts.
#[cfg(feature = "store")]
pub fn seal(ck: &Checkpoint, regions: &[Vec<String>], name: &str, threads: usize) -> Result<OwnedFd> {
    let (layout, source) = open_checkpoint(ck, regions, true)?;
    let (fd, fill) = crate::host::open_sealed(name, &layout, None, None)?;
    let named = name.to_string();
    let run = move || {
        if let Err(e) = fill.run(&layout, &source, threads, &IoTally::default()) {
            tracing::warn!("varena: sealing {named} failed; its adopters read the chunks: {e}");
        }
    };
    std::thread::Builder::new().name(format!("varena-seal-{name}")).spawn(run).map_err(|e| Error::Io(format!("spawn a seal: {e}")))?;
    Ok(fd)
}

/// `fd` in a socket's queue, the sending end closed: whoever holds the other end can take it
/// once, and it lives while any holder does.
fn contain(fd: OwnedFd) -> Result<UnixStream> {
    let io = |e: std::io::Error| Error::Io(format!("contain kept pages: {e}"));
    let (tx, rx) = UnixStream::pair().map_err(io)?;
    platform::handles::send_fd(&tx, &fd).map_err(|e| io(e.into()))?;
    Ok(rx)
}

fn reserve_va(size: u64) -> Result<u64> {
    let mut base = 0u64;
    call!("cuMemAddressReserve", mem_address_reserve(&mut base, size as usize, REGION_ALIGN as usize, 0, 0))?;
    Ok(base)
}

fn free_va(base: u64, size: u64) -> Result<()> {
    call!("cuMemAddressFree", mem_address_free(base, size as usize))
}

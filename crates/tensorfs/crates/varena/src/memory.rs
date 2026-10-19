//! Weights on one GPU, for the process computing with them (design memory-varena §5.1).
//!
//! A weight set reserves one VA range per GPU laid out like its host layout; a region maps at
//! its home offset in physical chunks, or, outside the home set, streams through a few scratch
//! slots the size of the largest such region. A `Pinned` is a region's bytes in use: while one
//! lives its pages stay mapped, and its drop records a release event on its stream. Eviction
//! unmaps only unpinned pages whose fill and release events completed. Copies run on this GPU's
//! own stream, from a sealed host layout or from the chunk files through pinned staging; a
//! consumer waits on an event, the host never syncs.

use std::collections::{BTreeMap, VecDeque};
use std::ffi::c_void;
#[cfg(feature = "store")]
use std::fs::File;
#[cfg(feature = "store")]
use std::os::fd::{OwnedFd, RawFd};
use std::sync::{mpsc, Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Instant;

#[cfg(feature = "store")]
use tensorfs_core::{format::Header, ids::ObjectRef};

#[cfg(feature = "store")]
use crate::checkpoint::Ck;
use crate::cuda::{self, AccessDesc, AllocationProp, CUcontext, CUevent, CUstream, CtxGuard, Handle};
use crate::host::HostMem;
use crate::io::{IoTally, Mode, Readers, Source};
use crate::layout::{align_up, Item, ItemSource, Layout, PART_ALIGN, REGION_ALIGN};
use crate::{Error, Result};

/// Scratch slots kept while a region streams; more exist only while every one is pinned.
const SLOTS: usize = 3;
/// Idle chunks kept after an eviction for the next map; the rest go back to the driver.
const IDLE_KEEP: u64 = 64 << 20;
const STAGING: usize = 16;
const STAGING_BYTES: u64 = 2 << 20;
const READERS: usize = 8;

/// A checked driver call on handles this core made, in its context.
macro_rules! call {
    ($op:literal, $f:ident($($a:expr),* $(,)?)) => {
        // SAFETY: live handles of our context; out-params point at live locals.
        cuda::check($op, unsafe { (cuda::driver()?.$f)($($a),*) })
    };
}

// ---------------------------------------------------------------- events

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
            rc => cuda::check("cuEventQuery", rc).map(|_| false),
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
        self.pool.free.lock().unwrap()[self.timing as usize].push(self.raw);
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
        let raw = match self.free.lock().unwrap()[timing as usize].pop() {
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
    events.retain(|e| e.done().map(|done| !done).unwrap_or_else(|x| err.get_or_insert(x).is_ok()));
    err.map_or(Ok(events.is_empty()), Err)
}

// ---------------------------------------------------------------- stats

/// Counted per weight set: in the step under way, in the last finished step, and in total.
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

/// One stored part: where its bytes sit in its region (`offset` from its start), and what they
/// are; `role` is `""` for a plain tensor's data.
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

/// A registered checkpoint's regions: its layout and parts, and what copies read from (its chunk
/// files, and its sealed host layout if granted).
pub struct Weights {
    pub name: String,
    pub layout: Layout,
    pub parts: Vec<Part>,
    source: Source,
    host: Option<HostMem>,
}

#[cfg(feature = "store")]
impl Weights {
    /// With `host_fd`, a sealed host layout of the same regions (TensorD's grant) serves the
    /// copies; without, the chunk files do: `files` (a grant's descriptors), or the checkpoint's
    /// own paths.
    fn open(name: &str, ck: Ck<'_>, regions: &[Vec<String>], host_fd: Option<RawFd>, files: Option<Vec<(ObjectRef, File)>>) -> Result<Weights> {
        if host_fd.is_none() && files.is_none() && ck.paths.is_none() {
            return Err(Error::Invalid("a header parsed alone reads only a grant: chunk files or a sealed layout".into()));
        }
        let (layout, source) = open_checkpoint(ck, regions, false, files)?;
        let host = host_fd.map(|fd| HostMem::adopt_sealed(fd, &layout)).transpose()?;
        if host.as_ref().is_some_and(|h| h.window().is_some()) {
            return Err(Error::Invalid("a streamed host layout: register takes a whole sealed layout".into()));
        }
        Ok(Weights { name: name.into(), parts: parts(ck.header, &layout), layout, source, host })
    }
}

/// Each part of `layout` with its dtype and shape from `header`.
#[cfg(feature = "store")]
fn parts(header: &Header, layout: &Layout) -> Vec<Part> {
    let part = |p: &crate::layout::Part| {
        let (tensor, role) = p.what.rsplit_once('#')?;
        let (component, key) = tensor.split_once('/')?;
        let t = header.tensor(component, key)?;
        let role = if role == "value" { "" } else { role };
        let (dtype, shape) = match t.part(role) {
            Some(part) => (part.dtype.name().to_string(), part.shape.clone()),
            None => (t.dtype.name().to_string(), t.shape.clone()),
        };
        let offset = p.offset - layout.regions[p.region as usize].offset;
        Some(Part { component: component.into(), key: key.into(), role: role.into(), dtype, shape, region: p.region, offset, nbytes: p.nbytes })
    };
    layout.parts.iter().filter_map(part).collect()
}

// ---------------------------------------------------------------- fills

/// One region's copy: pending until it is enqueued on the copy stream (`Ok`, with the event that
/// marks it landed) or failed with nothing left in flight.
struct Fill {
    state: Mutex<Option<Result<EvRef>>>,
    cv: Condvar,
}

impl Fill {
    fn set(&self, r: Result<EvRef>) {
        *self.state.lock().unwrap() = Some(r);
        self.cv.notify_all();
    }

    /// Until the copies are enqueued: the host waits only for reads that feed them.
    fn wait(&self) -> Result<EvRef> {
        self.cv.wait_while(self.state.lock().unwrap(), |s| s.is_none()).unwrap().clone().unwrap()
    }

    fn busy(&self) -> bool {
        self.state.lock().unwrap().is_none()
    }

    /// Nothing of it is in flight on the device.
    fn landed(&self) -> Result<bool> {
        match &*self.state.lock().unwrap() {
            Some(Ok(ready)) => ready.done(),
            Some(Err(_)) => Ok(true),
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

#[derive(Default)]
struct Jobs {
    q: VecDeque<Job>,
    stop: bool,
}

struct Buf {
    ptr: *mut u8,
    last: Option<EvRef>,
}
unsafe impl Send for Buf {}

/// Pinned host buffers that page-cache bytes pass through; one is reused once its copy landed.
struct Staging {
    free: Mutex<Vec<Buf>>,
    cv: Condvar,
}

impl Staging {
    fn take(&self) -> Result<Buf> {
        let mut buf = self.cv.wait_while(self.free.lock().unwrap(), |f| f.is_empty()).unwrap().pop().unwrap();
        match buf.last.take().map_or(Ok(()), |last| last.sync()) {
            Ok(()) => Ok(buf),
            Err(e) => {
                self.put(buf);
                Err(e)
            }
        }
    }

    fn put(&self, buf: Buf) {
        self.free.lock().unwrap().push(buf);
        self.cv.notify_one();
    }
}

// ---------------------------------------------------------------- books

#[derive(Clone, Copy)]
struct Chunk {
    handle: Handle,
    size: u64,
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
        self.pins == 0 && self.fill.as_ref().is_some_and(|f| matches!(*f.state.lock().unwrap(), Some(Err(_))))
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
    base: u64,
    regs: Vec<Reg>,
    step: Counters,
    last: Counters,
    total: Counters,
    /// The clock at its last bind, and whether its model is warm (loaded at GPU level): asks
    /// take idle sets before warm ones, least recently bound first. A busy set (a forward in
    /// flight) is never asked: its unbound blocks are needed this pass.
    used: u64,
    warm: bool,
    busy: bool,
}

impl Set {
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

#[derive(Clone, Copy)]
struct Stream(CUstream);
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

struct Inner {
    ordinal: i32,
    ctx: CUcontext,
    prop: AllocationProp,
    chunk: u64,
    events: Arc<Events>,
    copy: Stream,
    readers: Readers,
    tally: IoTally,
    staging: Staging,
    jobs: (Mutex<Jobs>, Condvar),
    st: Mutex<State>,
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
        let prop = AllocationProp::device(dev);
        let mut g = 0usize;
        call!("cuMemGetAllocationGranularity", mem_get_allocation_granularity(&mut g, &prop, cuda::MEM_GRANULARITY_MINIMUM))?;
        if g == 0 || !REGION_ALIGN.is_multiple_of(g as u64) {
            return Err(Error::Invalid(format!("device {ordinal}: granularity {g} does not divide {REGION_ALIGN}")));
        }
        let mut s: CUstream = std::ptr::null_mut();
        call!("cuStreamCreate", stream_create(&mut s, cuda::STREAM_NON_BLOCKING))?;
        let mut bufs = Vec::with_capacity(STAGING);
        for _ in 0..STAGING {
            let mut p: *mut c_void = std::ptr::null_mut();
            call!("cuMemHostAlloc", mem_host_alloc(&mut p, STAGING_BYTES as usize, cuda::MEMHOSTALLOC_PORTABLE))?;
            bufs.push(Buf { ptr: p as *mut u8, last: None });
        }
        let inner = Arc::new(Inner {
            ordinal,
            ctx,
            prop,
            chunk,
            events: Arc::new(Events { ctx, free: Mutex::default() }),
            copy: Stream(s),
            readers: Readers::new(READERS),
            tally: IoTally::default(),
            staging: Staging { free: Mutex::new(bufs), cv: Condvar::new() },
            jobs: Default::default(),
            st: Mutex::default(),
        });
        let m = inner.clone();
        let mover = std::thread::Builder::new().name(format!("varena-copy-{ordinal}")).spawn(move || m.mover());
        Ok(Arc::new(Gpu { inner, mover: Some(mover.map_err(|e| Error::Io(format!("spawn copy thread: {e}")))?) }))
    }

    pub fn device(&self) -> i32 {
        self.inner.ordinal
    }

    pub fn chunk(&self) -> u64 {
        self.inner.chunk
    }

    /// `regions` (`tensorfs_core::client::regions`) of a v2 checkpoint as one weight set; its
    /// copies read as `Weights::open` says.
    #[cfg(feature = "store")]
    pub fn register<'c>(self: &Arc<Self>, name: &str, ck: impl Into<Ck<'c>>, regions: &[Vec<String>], host_fd: Option<RawFd>, files: Option<Vec<(ObjectRef, File)>>) -> Result<Arc<WeightSet>> {
        let weights = Arc::new(Weights::open(name, ck.into(), regions, host_fd, files)?);
        let g = &self.inner;
        let _c = CtxGuard::enter(g.ctx)?;
        let base = reserve_va(weights.layout.nbytes.max(REGION_ALIGN))?;
        let regs = weights.layout.regions.iter().map(|_| Reg { home: true, map: None }).collect();
        let (w, z) = (weights.clone(), Counters::default());
        let mut st = g.lock();
        st.sets.push(Some(Set { w, base, regs, step: z, last: z, total: z, used: 0, warm: true, busy: false }));
        Ok(Arc::new(WeightSet { gpu: self.clone(), id: st.sets.len() - 1, base, weights }))
    }

    /// Idle chunks and unneeded slots back to the driver; returns the bytes released.
    pub fn trim(&self) -> Result<u64> {
        let g = &self.inner;
        g.with(|st| {
            let span = st.slot_span;
            let mut live = st.slots.iter().flatten().count();
            for i in 0..st.slots.len() {
                let Some(s) = st.slots[i].as_mut() else { continue };
                if (span == 0 || s.m.span < span || live > SLOTS) && s.m.quiet()? {
                    g.drop_slot(st, i)?;
                    live -= 1;
                }
            }
            g.keep_idle(st, 0)
        })
    }

    /// Bytes for an ask (design §5.2), another process's (`serve`) or this one's: idle chunks and
    /// unneeded slots (under ""), then unpinned home pages, idle sets before warm ones, least
    /// recently bound first, until `want`. Busy sets and pinned pages stay. Returns what each gave.
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
            g.keep_idle(st, IDLE_KEEP)?;
            Ok(gave)
        })
    }

    /// Ends a step: the counters of the step under way become the last step's.
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
        self.inner.jobs.0.lock().unwrap().stop = true;
        self.inner.jobs.1.notify_all();
        if let Some(t) = self.mover.take() {
            let _ = t.join();
        }
    }
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.st.lock().unwrap()
    }

    /// `f` on the books, with our context current.
    fn with<T>(&self, f: impl FnOnce(&mut State) -> Result<T>) -> Result<T> {
        let _c = CtxGuard::enter(self.ctx)?;
        f(&mut *self.lock())
    }

    fn submit(&self, job: Job, urgent: bool) {
        let mut j = self.jobs.0.lock().unwrap();
        if urgent { j.q.push_front(job) } else { j.q.push_back(job) }
        self.jobs.1.notify_all();
    }

    /// A queued fill someone now waits for goes first.
    fn hurry(&self, fill: &Arc<Fill>) {
        let mut j = self.jobs.0.lock().unwrap();
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
        loop {
            let job = {
                let mut j = self.jobs.1.wait_while(self.jobs.0.lock().unwrap(), |j| j.q.is_empty() && !j.stop).unwrap();
                match j.q.pop_front() {
                    Some(job) => job,
                    None => return,
                }
            };
            let filled = self.run(&job).map(|(a, b)| {
                let mut st = self.lock();
                let step = st.steps;
                st.timings.push(Timing { set: job.set, step, field: |c| &mut c.copy_ns, a, b: b.clone() });
                b
            });
            job.fill.set(filled.map_err(|e| self.drained(e)));
        }
    }

    /// A failure is reported once nothing it queued is in flight.
    fn drained(&self, e: Error) -> Error {
        match call!("cuStreamSynchronize", stream_synchronize(self.copy.0)) {
            Ok(()) => e,
            Err(s) => Error::Io(format!("{e}; and the copy stream could not drain: {s}")),
        }
    }

    fn run(self: &Arc<Self>, job: &Job) -> Result<(EvRef, EvRef)> {
        let s = self.copy.0;
        for w in &job.waits {
            wait_in(s, w)?;
        }
        let start = self.events.record(s, true)?;
        let (r, reg) = (job.region, job.w.layout.region(job.region)?);
        let n = reg.nbytes;
        match job.w.host.as_ref().filter(|h| !h.hole(r)) {
            Some(h) if n > 0 => {
                h.wait_ready(r)?;
                // SAFETY: inside the live mapping the source keeps; the region is complete.
                let p = unsafe { h.base().add(reg.offset as usize) } as *const u8;
                if h.pin_region(r, reg.offset, align_up(n, PART_ALIGN)).0 {
                    call!("cuMemcpyHtoDAsync", memcpy_htod_async(job.dst, p as *const c_void, n as usize, s))?;
                } else {
                    self.staged(job, Some(p as usize))?;
                }
            }
            _ if n > 0 => self.staged(job, None)?,
            _ => {}
        }
        Ok((start, self.events.record(s, true)?))
    }

    /// The region through the staging buffers, a buffer per piece, read on the reader threads:
    /// from host memory at `host`, else from the chunk files through the page cache.
    fn staged(self: &Arc<Self>, job: &Job, host: Option<usize>) -> Result<()> {
        let n = job.w.layout.region(job.region)?.nbytes;
        let (tx, rx) = mpsc::channel();
        for at in (0..n).step_by(STAGING_BYTES as usize) {
            let (g, w, tx, r, dst) = (self.clone(), job.w.clone(), tx.clone(), job.region, job.dst);
            self.readers.submit(true, move || drop(tx.send(g.piece(&w, r, dst, at, STAGING_BYTES.min(n - at), host))));
        }
        drop(tx);
        let done: Vec<Result<()>> = rx.iter().collect();
        if done.len() < n.div_ceil(STAGING_BYTES) as usize {
            return Err(Error::Io("a staged read never ran".into()));
        }
        done.into_iter().collect()
    }

    fn piece(&self, w: &Weights, r: u32, dst: u64, at: u64, len: u64, host: Option<usize>) -> Result<()> {
        let mut buf = self.staging.take()?;
        let res = (|| {
            match host {
                // SAFETY: `len` bytes inside the region the source keeps mapped; the buffer is ours.
                Some(p) => unsafe { std::ptr::copy_nonoverlapping((p as *const u8).add(at as usize), buf.ptr, len as usize) },
                None => {
                    let reg = w.layout.region(r)?;
                    for it in slice_items(&reg.items, reg.offset + at, reg.offset + at + len) {
                        // SAFETY: a buffered read writes `it.len` bytes, inside this buffer.
                        unsafe { w.source.read_item(&it, buf.ptr.add((it.offset - reg.offset - at) as usize), Mode::Buffered, &self.tally)? };
                    }
                }
            }
            let _c = CtxGuard::enter(self.ctx)?;
            call!("cuMemcpyHtoDAsync", memcpy_htod_async(dst + at, buf.ptr as *const c_void, len as usize, self.copy.0))?;
            buf.last = Some(self.events.record(self.copy.0, false)?);
            Ok(())
        })();
        self.staging.put(buf);
        res
    }

    fn create(&self, size: u64) -> Result<Option<Handle>> {
        let mut h: Handle = 0;
        // SAFETY: out-param; device memory of our context's device.
        match unsafe { (cuda::driver()?.mem_create)(&mut h, size as usize, &self.prop, 0) } {
            cuda::SUCCESS => Ok(Some(h)),
            cuda::ERROR_OUT_OF_MEMORY => Ok(None),
            rc => cuda::check("cuMemCreate", rc).map(|_| None),
        }
    }

    /// Maps `span` bytes at `va` in chunks: idle ones first, then new ones. A driver refusal is
    /// `DeviceFull`, with nothing left mapped.
    fn map(&self, st: &mut State, va: u64, span: u64) -> Result<Mapping> {
        let mut sizes = vec![self.chunk; (span / self.chunk) as usize];
        if !span.is_multiple_of(self.chunk) {
            sizes.push(span % self.chunk);
        }
        let mut chunks: Vec<Chunk> = Vec::with_capacity(sizes.len());
        for (i, &size) in sizes.iter().enumerate() {
            if let Some(handle) = st.idle.get_mut(&size).and_then(Vec::pop) {
                chunks.push(Chunk { handle, size });
                continue;
            }
            let mut h = self.create(size)?;
            if h.is_none() && st.idle_bytes() > 0 {
                self.keep_idle(st, 0)?;
                h = self.create(size)?;
            }
            match h {
                Some(handle) => chunks.push(Chunk { handle, size }),
                None => {
                    self.give_back(st, chunks);
                    return Err(Error::DeviceFull { need: span, deficit: sizes[i..].iter().sum() });
                }
            }
        }
        let mut at = va;
        let mut failure = None;
        for c in &chunks {
            if let Err(e) = call!("cuMemMap", mem_map(at, c.size as usize, 0, c.handle, 0)) {
                failure = Some(e);
                break;
            }
            at += c.size;
        }
        if failure.is_none() {
            let desc = AccessDesc { location: self.prop.location, flags: cuda::MEM_ACCESS_PROT_READWRITE };
            failure = call!("cuMemSetAccess", mem_set_access(va, span as usize, &desc, 1)).err();
        }
        if let Some(e) = failure {
            if at > va {
                call!("cuMemUnmap", mem_unmap(va, (at - va) as usize))?;
            }
            self.give_back(st, chunks);
            return Err(e);
        }
        st.mapped += span;
        Ok(Mapping { va, span, chunks, ..Default::default() })
    }

    /// Unmaps a mapping nothing in flight touches; its chunks go idle.
    fn unmap(&self, st: &mut State, m: Mapping) -> Result<()> {
        call!("cuMemUnmap", mem_unmap(m.va, m.span as usize))?;
        st.mapped -= m.span;
        self.give_back(st, m.chunks);
        Ok(())
    }

    fn give_back(&self, st: &mut State, chunks: Vec<Chunk>) {
        for c in chunks {
            st.idle.entry(c.size).or_default().push(c.handle);
        }
    }

    /// Unmaps set `id`'s `regions` (all when None) at home that no pin holds and whose fill and
    /// release events completed, until `budget` bytes; returns the bytes unmapped.
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

    fn new_slot(&self, st: &mut State, span: u64) -> Result<usize> {
        for i in 0..st.slots.len() {
            if st.slots[i].as_ref().is_some_and(|s| s.m.span < span) && st.slot(i).m.quiet()? {
                self.drop_slot(st, i)?;
            }
        }
        let va = reserve_va(span)?;
        let m = self.map(st, va, span).inspect_err(|_| drop(free_va(va, span)))?;
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

    /// A slot for streamed region `r`: the one holding it, else a settled one (empty, then
    /// consumed, then, for a bind, a prefetch not yet bound), else a new one. A prefetch makes a
    /// new one only below `SLOTS`, and otherwise does nothing (None).
    fn slot_for(&self, st: &mut State, set: usize, r: u32, urgent: bool) -> Result<Option<usize>> {
        let (w, span) = (st.set(set).w.clone(), st.slot_span);
        let x = w.layout.region(r)?;
        let held = st.slots.iter().position(|s| s.as_ref().is_some_and(|s| s.holds == Some((set, r))));
        let i = match held {
            Some(i) if !st.slot(i).m.dead() => return Ok(Some(i)),
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
        self.submit(Job { set, w, region: r, dst, waits, fill }, urgent);
        Ok(Some(i))
    }

    /// Maps region `r` at home and queues its fill.
    fn map_home(&self, st: &mut State, set: usize, r: u32, urgent: bool) -> Result<()> {
        let (w, base) = (st.set(set).w.clone(), st.set(set).base);
        let x = w.layout.region(r)?;
        let t = Instant::now();
        let mut m = self.map(st, base + x.offset, x.span)?;
        let ns = t.elapsed().as_nanos() as u64;
        let ((fill, waits), dst) = (m.refill(), m.va);
        st.reg(set, r)?.map = Some(m);
        st.set(set).count(|c| {
            c.map_ns += ns;
            c.paged_in += x.nbytes;
            c.faults += urgent as u64;
        });
        self.submit(Job { set, w, region: r, dst, waits, fill }, urgent);
        Ok(())
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.readers.stop();
        let Ok(d) = cuda::driver() else { return };
        let Ok(_c) = CtxGuard::enter(self.ctx) else { return };
        // SAFETY: our stream, drained before anything it wrote to goes.
        unsafe { (d.stream_synchronize)(self.copy.0) };
        let mut st = std::mem::take(self.st.get_mut().unwrap());
        for i in 0..st.slots.len() {
            if let Some(s) = &st.slots[i] {
                s.m.released.iter().for_each(|e| drop(e.sync()));
                let _ = self.drop_slot(&mut st, i);
            }
        }
        let _ = self.keep_idle(&mut st, 0);
        for b in self.staging.free.get_mut().unwrap().drain(..) {
            // SAFETY: our pinned buffers; the stream that read them is drained.
            unsafe { (d.mem_free_host)(b.ptr as *mut c_void) };
        }
        // SAFETY: our drained stream.
        unsafe { (d.stream_destroy)(self.copy.0) };
    }
}

// ---------------------------------------------------------------- weight sets

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

    /// The watermark: `regions` live at their home VA, the rest stream through slots. A region
    /// left out stays mapped until evicted, and serves binds from home meanwhile.
    pub fn home(&self, regions: &[u32]) -> Result<()> {
        let n = self.weights.layout.regions.len() as u32;
        if let Some(r) = regions.iter().find(|&&r| r >= n) {
            return Err(Error::Invalid(format!("region {r} of {n}")));
        }
        let mut st = self.gpu.inner.lock();
        for (r, reg) in st.set(self.id).regs.iter_mut().enumerate() {
            reg.home = regions.contains(&(r as u32));
        }
        let streamed = st.sets.iter().flatten().flat_map(|s| s.regs.iter().zip(&s.w.layout.regions).filter(|(r, _)| !r.home));
        st.slot_span = streamed.map(|(_, x)| x.span).max().unwrap_or(0);
        Ok(())
    }

    /// Region `r` for use on `stream`: at home (mapped and filled if absent) or in the slot that
    /// holds it. `stream` waits for the bytes; the host waits only for reads that feed a copy.
    pub fn bind(self: &Arc<Self>, region: u32, stream: u64) -> Result<Pinned> {
        let g = &self.gpu.inner;
        let _c = CtxGuard::enter(g.ctx)?;
        let (at, ptr, fill, fresh) = {
            let mut st = g.lock();
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
                let (ptr, fill, fresh) = st.reg(self.id, region)?.map.as_mut().unwrap().pin();
                (At::Home, ptr, fill, fresh)
            } else {
                let i = g.slot_for(&mut st, self.id, region, true)?.expect("a bind always gets a slot");
                let s = st.slot(i);
                s.pending = false;
                let (ptr, fill, fresh) = s.m.pin();
                (At::Slot(i), ptr, fill, fresh)
            }
        };
        g.hurry(&fill);
        let ready = fill.wait().inspect_err(|_| self.release(at, region, None))?;
        let nbytes = self.weights.layout.regions[region as usize].nbytes;
        let pinned = Pinned { ws: self.clone(), at, region, ptr, nbytes, fresh, stream, ready };
        if !pinned.ready.done()? {
            let s = stream as CUstream;
            let a = g.events.record(s, true)?;
            wait_in(s, &pinned.ready)?;
            let b = g.events.record(s, true)?;
            let mut st = g.lock();
            let step = st.steps;
            st.timings.push(Timing { set: self.id, step, field: |c| &mut c.stall_ns, a, b });
        }
        Ok(pinned)
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

    /// Unmaps `regions` (all when None) at home, those unpinned whose fill and release events
    /// completed; returns the bytes unmapped. Their chunks go idle, beyond a small cache back to
    /// the driver.
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

    /// Whether its model is warm (loaded at GPU level): an ask takes idle sets first.
    pub fn warm(&self, warm: bool) {
        self.gpu.inner.lock().set(self.id).warm = warm;
    }

    /// Whether a forward of its model is in flight: an ask never takes a busy set's pages.
    pub fn busy(&self, busy: bool) {
        self.gpu.inner.lock().set(self.id).busy = busy;
    }

    fn release(&self, at: At, region: u32, ev: Option<EvRef>) {
        let mut st = self.gpu.inner.lock();
        let m = match at {
            At::Home => match st.reg(self.id, region) {
                Ok(Reg { map: Some(m), .. }) => m,
                _ => return,
            },
            At::Slot(i) => &mut st.slot(i).m,
        };
        m.pins -= 1;
        if let Some(e) = ev {
            let _ = passed(&mut m.released);
            m.released.push(e);
        }
    }
}

impl Drop for WeightSet {
    fn drop(&mut self) {
        let g = &self.gpu.inner;
        let Ok(_c) = CtxGuard::enter(g.ctx) else { return };
        let set = {
            let mut st = g.lock();
            for s in st.slots.iter_mut().flatten().filter(|s| s.holds.is_some_and(|(set, _)| set == self.id)) {
                s.holds = None;
            }
            st.sets[self.id].take()
        };
        let Some(set) = set else { return };
        let maps: Vec<Mapping> = set.regs.into_iter().filter_map(|r| r.map).collect();
        for m in &maps {
            if let Some(Ok(ready)) = m.fill.as_ref().map(|f| f.wait()) {
                let _ = ready.sync();
            }
            m.released.iter().for_each(|e| drop(e.sync()));
        }
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

/// A bound region's bytes on the device. While it lives its pages stay mapped; its drop
/// records a release event on its bind stream.
pub struct Pinned {
    ws: Arc<WeightSet>,
    at: At,
    pub region: u32,
    pub ptr: u64,
    pub nbytes: u64,
    /// This bind is the first since a copy put the bytes there.
    pub fresh: bool,
    stream: u64,
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
        if stream == self.stream || self.ready.done()? {
            return Ok(());
        }
        let _c = CtxGuard::enter(self.ws.gpu.inner.ctx)?;
        wait_in(stream as CUstream, &self.ready)
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        let g = &self.ws.gpu.inner;
        match CtxGuard::enter(g.ctx).and_then(|_c| g.events.record(self.stream as CUstream, false)) {
            Ok(e) => self.ws.release(self.at, self.region, Some(e)),
            // No event to order the release by: the pin stays, and the pages with it.
            Err(e) => tracing::warn!("varena: a release could not be recorded, region {} stays pinned: {e}", self.region),
        }
    }
}

// ---------------------------------------------------------------- host only

/// The host-only core, for CPU executors: regions as host bytes, never loading libcuda.
pub struct Host;

/// A weight set's regions in host memory: views of a sealed host layout (read-only, shared),
/// or private memory read once from the chunk files and kept until the set goes.
pub struct HostSet {
    pub weights: Weights,
    regs: Mutex<Vec<(Option<Arc<Anon>>, bool)>>,
}

impl Host {
    #[cfg(feature = "store")]
    pub fn register<'c>(&self, name: &str, ck: impl Into<Ck<'c>>, regions: &[Vec<String>], host_fd: Option<RawFd>, files: Option<Vec<(ObjectRef, File)>>) -> Result<Arc<HostSet>> {
        let weights = Weights::open(name, ck.into(), regions, host_fd, files)?;
        Ok(Arc::new(HostSet { regs: Mutex::new(vec![(None, false); weights.layout.regions.len()]), weights }))
    }
}

impl HostSet {
    pub fn bind(self: &Arc<Self>, region: u32) -> Result<HostPinned> {
        let (w, mut regs) = (&self.weights, self.regs.lock().unwrap());
        let reg = w.layout.region(region)?;
        let (buf, bound) = &mut regs[region as usize];
        let sealed = w.host.as_ref().filter(|h| !h.hole(region));
        let (ptr, buf) = match sealed {
            Some(h) => {
                h.wait_ready(region)?;
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
        for it in &reg.items {
            // SAFETY: a buffered read writes `it.len` bytes inside the mapping.
            unsafe { w.source.read_item(it, a.ptr.add((it.offset - reg.offset) as usize), Mode::Buffered, &IoTally::default())? };
        }
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

// ---------------------------------------------------------------- helpers

/// The one layout of `regions` of `ck`, and its chunk files (`files`, else the checkpoint's own
/// paths): what `register` maps and what a sealed host layout for it holds, the same by
/// construction; `HostMem::adopt_sealed` checks the digest.
#[cfg(feature = "store")]
fn open_checkpoint(ck: Ck<'_>, regions: &[Vec<String>], direct_io: bool, files: Option<Vec<(ObjectRef, File)>>) -> Result<(Layout, Source)> {
    let components: Vec<&str> = regions.iter().flatten().filter_map(|t| t.split_once('/').map(|(c, _)| c)).collect();
    let plan = crate::checkpoint::read_plan(ck.header, &components);
    let files = match (files, ck.paths) {
        (Some(f), _) => f,
        (None, Some(_)) => crate::checkpoint::chunk_files(ck, &plan)?,
        (None, None) => Vec::new(), // a parsed header's sealed layout: its holes have no source
    };
    Ok((Layout::build(&plan, regions)?, Source::files(files, direct_io, 0)))
}

/// `regions` of `ck` in a sealed host layout, filled in this process from the chunk files
/// (`files`, else the checkpoint's paths): the memfd `register(host_fd=)` adopts.
#[cfg(feature = "store")]
pub fn seal<'c>(ck: impl Into<Ck<'c>>, regions: &[Vec<String>], name: &str, threads: usize, files: Option<Vec<(ObjectRef, File)>>) -> Result<OwnedFd> {
    let (layout, source) = open_checkpoint(ck.into(), regions, true, files)?;
    crate::host::fill_sealed(name, &layout, &source, threads, &IoTally::default(), None)
}

fn reserve_va(size: u64) -> Result<u64> {
    let mut base = 0u64;
    call!("cuMemAddressReserve", mem_address_reserve(&mut base, size as usize, REGION_ALIGN as usize, 0, 0))?;
    Ok(base)
}

fn free_va(base: u64, size: u64) -> Result<()> {
    call!("cuMemAddressFree", mem_address_free(base, size as usize))
}

/// The parts of `items` inside layout range `[lo, hi)`, split at the edges.
fn slice_items(items: &[Item], lo: u64, hi: u64) -> Vec<Item> {
    let mut out = Vec::new();
    for it in items {
        let (s, e) = (it.offset.max(lo), (it.offset + it.len).min(hi));
        if s >= e {
            continue;
        }
        let skip = s - it.offset;
        let source = match &it.source {
            ItemSource::Object(o) => ItemSource::Object(tensorfs_core::read::ObjectRange { obj: o.obj.clone(), off: o.off + skip, len: e - s }),
            ItemSource::Inline(b) => ItemSource::Inline(b[skip as usize..(skip + e - s) as usize].into()),
        };
        out.push(Item { offset: s, len: e - s, source, part_end: it.part_end && e == it.offset + it.len });
    }
    out
}

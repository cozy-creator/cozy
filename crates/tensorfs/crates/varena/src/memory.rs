//! Weights on one GPU, for the process computing with them (design memory-varena §5.1).
//!
//! A weight set reserves one VA range per GPU laid out like its host layout; a region maps at
//! its home offset in physical chunks. Regions outside the home set stream through a few
//! scratch slots the size of the largest of them. A `Pinned` is a region's bytes in use: while
//! one lives its pages stay mapped, and its drop records a release event on its stream.
//! Eviction unmaps only unpinned pages whose fill and release events completed. Copies run on
//! this GPU's own stream, from a sealed host layout (one DMA once page-locked) or from the
//! chunk files through pinned staging; a consumer waits on an event, the host never syncs.

use std::collections::{BTreeMap, VecDeque};
use std::ffi::c_void;
#[cfg(feature = "store")]
use std::fs::File;
#[cfg(feature = "store")]
use std::os::fd::{OwnedFd, RawFd};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Instant;

#[cfg(feature = "store")]
use tensorfs_core::{format::Header, ids::ObjectRef, store2::Checkpoint};

use crate::cuda::{self, AccessDesc, AllocationProp, CUcontext, CUevent, CUstream, CtxGuard, Handle};
use crate::host::HostMem;
use crate::io::{IoTally, Mode, Readers, Source};
use crate::layout::{align_up, Item, ItemSource, Layout, PART_ALIGN, REGION_ALIGN};
use crate::{Error, Result};

/// Scratch slots kept while a region streams; more exist only while every one is pinned.
pub const SLOTS: usize = 3;
/// Idle chunks kept after an eviction for the next map; the rest go back to the driver.
const IDLE_KEEP: u64 = 64 << 20;
const STAGING: usize = 16;
const STAGING_BYTES: u64 = 2 << 20;
const READERS: usize = 8;

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
        let d = cuda::driver()?;
        // SAFETY: a live event of our context.
        match unsafe { (d.event_query)(self.raw) } {
            cuda::SUCCESS => Ok(true),
            cuda::ERROR_NOT_READY => Ok(false),
            rc => cuda::check("cuEventQuery", rc).map(|_| false),
        }
    }
    fn sync(&self) -> Result<()> {
        // SAFETY: a live event.
        cuda::check("cuEventSynchronize", unsafe { (cuda::driver()?.event_synchronize)(self.raw) })
    }
    fn record(&self, s: CUstream) -> Result<()> {
        // SAFETY: an event and a stream of the same context.
        cuda::check("cuEventRecord", unsafe { (cuda::driver()?.event_record)(self.raw, s) })
    }
    /// Nanoseconds from `start` to this event, both completed timing events.
    fn since(&self, start: &Ev) -> Result<u64> {
        let mut ms = 0f32;
        // SAFETY: two completed timing events.
        cuda::check("cuEventElapsedTime", unsafe { (cuda::driver()?.event_elapsed_time)(&mut ms, start.raw, self.raw) })?;
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
    fn get(self: &Arc<Self>, timing: bool) -> Result<EvRef> {
        let raw = match self.free.lock().unwrap()[timing as usize].pop() {
            Some(raw) => raw,
            None => {
                let _c = CtxGuard::enter(self.ctx)?;
                let mut e: CUevent = std::ptr::null_mut();
                let flags = if timing { cuda::EVENT_DEFAULT } else { cuda::EVENT_DISABLE_TIMING };
                // SAFETY: out-param, context bound.
                cuda::check("cuEventCreate", unsafe { (cuda::driver()?.event_create)(&mut e, flags) })?;
                e
            }
        };
        Ok(Arc::new(Ev { raw, timing, pool: self.clone() }))
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
    // SAFETY: an event and a stream of the same context.
    cuda::check("cuStreamWaitEvent", unsafe { (cuda::driver()?.stream_wait_event)(stream, ev.raw, 0) })
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

// ---------------------------------------------------------------- stats

/// Counted per weight set: in the step under way, in the last finished step, and in total.
#[derive(Debug, Clone, Copy, Default)]
pub struct Counters {
    pub binds: u64,
    /// Binds that found their region absent and waited for its fill.
    pub faults: u64,
    /// Bytes filled at home.
    pub paged_in: u64,
    /// Bytes copied into scratch slots.
    pub streamed_bytes: u64,
    pub evicted: u64,
    /// Copy stream time of the fills, GPU-timed.
    pub copy_ns: u64,
    /// Consumer streams' wait for fills not yet landed at bind, GPU-timed.
    pub stall_ns: u64,
    /// Host time creating and mapping chunks.
    pub map_ns: u64,
}

impl Counters {
    fn add(&mut self, kind: Kind, ns: u64) {
        match kind {
            Kind::Copy => self.copy_ns += ns,
            Kind::Stall => self.stall_ns += ns,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SetStats {
    pub name: String,
    pub nbytes: u64,
    pub regions: u32,
    /// Bytes of the home set.
    pub home: u64,
    /// Bytes mapped at home.
    pub resident: u64,
    /// Bytes mapped at home and pinned.
    pub pinned: u64,
    /// Live pins, at home and in slots.
    pub pins: u32,
    /// Bytes of the slots that hold this set's regions now.
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
    /// Bytes mapped: homes and slots.
    pub mapped: u64,
    /// Chunks held unmapped for reuse.
    pub idle: u64,
    pub slots: u32,
    /// Bytes the slots map now.
    pub slot_bytes: u64,
    pub slot_span: u64,
    pub driver_free: u64,
    pub driver_total: u64,
    pub sets: Vec<SetStats>,
}

/// One stored part: where its bytes sit in its region, and what they are.
#[derive(Debug, Clone)]
pub struct Part {
    pub component: String,
    pub key: String,
    /// `""` for a plain tensor's data, else the encoded tensor's part.
    pub role: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub region: u32,
    /// From the start of its region.
    pub offset: u64,
    pub nbytes: u64,
}

/// Each part of `layout` with its dtype and shape from `header`.
#[cfg(feature = "store")]
pub fn parts(header: &Header, layout: &Layout) -> Vec<Part> {
    layout
        .parts
        .iter()
        .filter_map(|p| {
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
        })
        .collect()
}

// ---------------------------------------------------------------- fills

enum FillState {
    Queued,
    Running,
    Done(EvRef),
    Failed(Error),
}

/// One region's copy: queued, then running on the mover, then enqueued on the copy stream
/// (`Done`, with the event that marks it landed) or failed with nothing left in flight.
struct Fill {
    state: Mutex<FillState>,
    cv: Condvar,
}

impl Fill {
    fn new() -> Arc<Fill> {
        Arc::new(Fill { state: Mutex::new(FillState::Queued), cv: Condvar::new() })
    }
    fn set(&self, s: FillState) {
        *self.state.lock().unwrap() = s;
        self.cv.notify_all();
    }
    /// Until the copies are enqueued: the host waits only for reads that feed them.
    fn wait(&self) -> Result<EvRef> {
        let mut s = self.state.lock().unwrap();
        while matches!(*s, FillState::Queued | FillState::Running) {
            s = self.cv.wait(s).unwrap();
        }
        match &*s {
            FillState::Done(ready) => Ok(ready.clone()),
            FillState::Failed(e) => Err(e.clone()),
            _ => unreachable!(),
        }
    }
    fn busy(&self) -> bool {
        matches!(*self.state.lock().unwrap(), FillState::Queued | FillState::Running)
    }
    fn failed(&self) -> bool {
        matches!(*self.state.lock().unwrap(), FillState::Failed(_))
    }
    /// Nothing of it is in flight on the device.
    fn landed(&self) -> Result<bool> {
        match &*self.state.lock().unwrap() {
            FillState::Done(ready) => ready.done(),
            FillState::Failed(_) => Ok(true),
            _ => Ok(false),
        }
    }
}

/// A weight set's bytes: its layout, its chunk files, and its sealed host layout if granted.
struct Src {
    layout: Arc<Layout>,
    source: Source,
    host: Option<HostMem>,
}

struct Job {
    set: usize,
    src: Arc<Src>,
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

/// A set of reader jobs with one result: the first error.
struct Latch {
    left: Mutex<(usize, Option<Error>)>,
    cv: Condvar,
}

impl Latch {
    fn done(&self, r: Result<()>) {
        let mut l = self.left.lock().unwrap();
        l.0 -= 1;
        if let Err(e) = r {
            l.1.get_or_insert(e);
        }
        self.cv.notify_all();
    }
    fn wait(&self) -> Result<()> {
        let mut l = self.left.lock().unwrap();
        while l.0 > 0 {
            l = self.cv.wait(l).unwrap();
        }
        l.1.take().map_or(Ok(()), Err)
    }
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
        let mut free = self.free.lock().unwrap();
        while free.is_empty() {
            free = self.cv.wait(free).unwrap();
        }
        let mut buf = free.pop().unwrap();
        drop(free);
        if let Some(last) = buf.last.take() {
            if let Err(e) = last.sync() {
                self.put(buf);
                return Err(e);
            }
        }
        Ok(buf)
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

struct Mapping {
    chunks: Vec<Chunk>,
    fill: Arc<Fill>,
    /// Filled and not yet bound.
    fresh: bool,
}

#[derive(Default)]
struct Reg {
    home: bool,
    map: Option<Mapping>,
    pins: u32,
    released: Vec<EvRef>,
}

struct Set {
    name: String,
    base: u64,
    src: Arc<Src>,
    regs: Vec<Reg>,
    step: Counters,
    last: Counters,
    total: Counters,
}

impl Set {
    fn count(&mut self, f: impl Fn(&mut Counters)) {
        f(&mut self.step);
        f(&mut self.total);
    }
}

struct Slot {
    va: u64,
    span: u64,
    chunks: Vec<Chunk>,
    /// (set, region) whose bytes it holds or is filling.
    holds: Option<(usize, u32)>,
    fill: Option<Arc<Fill>>,
    /// Filled and not yet bound.
    fresh: bool,
    pins: u32,
    released: Vec<EvRef>,
    /// Prefetched and not yet bound.
    pending: bool,
    used: u64,
}

impl Slot {
    fn settled(&self) -> bool {
        self.pins == 0 && self.fill.as_ref().is_none_or(|f| !f.busy())
    }
    fn quiet(&mut self) -> Result<bool> {
        Ok(self.settled() && self.fill.as_ref().map_or(Ok(true), |f| f.landed())? && passed(&mut self.released)?)
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Copy,
    Stall,
}

/// A GPU-timed interval, read once both events completed.
struct Timing {
    set: usize,
    step: u64,
    kind: Kind,
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
        let n = self.set(id).regs.len();
        self.set(id).regs.get_mut(r as usize).ok_or_else(|| Error::Invalid(format!("region {r} of {n}")))
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
                s.total.add(t.kind, ns);
                if t.step == steps {
                    s.step.add(t.kind, ns);
                } else if t.step + 1 == steps {
                    s.last.add(t.kind, ns);
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
    mover: Mutex<Option<JoinHandle<()>>>,
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
        let d = cuda::driver()?;
        let prop = AllocationProp::device(dev);
        let mut g = 0usize;
        // SAFETY: out-param.
        cuda::check("cuMemGetAllocationGranularity", unsafe { (d.mem_get_allocation_granularity)(&mut g, &prop, cuda::MEM_GRANULARITY_MINIMUM) })?;
        if g == 0 || !REGION_ALIGN.is_multiple_of(g as u64) {
            return Err(Error::Invalid(format!("device {ordinal}: granularity {g} does not divide {REGION_ALIGN}")));
        }
        let mut s: CUstream = std::ptr::null_mut();
        // SAFETY: out-param, context bound.
        cuda::check("cuStreamCreate", unsafe { (d.stream_create)(&mut s, cuda::STREAM_NON_BLOCKING) })?;
        let mut bufs = Vec::with_capacity(STAGING);
        for _ in 0..STAGING {
            let mut p: *mut c_void = std::ptr::null_mut();
            // SAFETY: out-param; pinned, portable host memory.
            cuda::check("cuMemHostAlloc", unsafe { (d.mem_host_alloc)(&mut p, STAGING_BYTES as usize, cuda::MEMHOSTALLOC_PORTABLE) })?;
            bufs.push(Buf { ptr: p as *mut u8, last: None });
        }
        let inner = Arc::new(Inner {
            ordinal,
            ctx,
            prop,
            chunk,
            events: Arc::new(Events { ctx, free: Mutex::new([Vec::new(), Vec::new()]) }),
            copy: Stream(s),
            readers: Readers::new(READERS),
            tally: IoTally::default(),
            staging: Staging { free: Mutex::new(bufs), cv: Condvar::new() },
            jobs: (Mutex::new(Jobs::default()), Condvar::new()),
            st: Mutex::new(State::default()),
        });
        let m = inner.clone();
        let mover = std::thread::Builder::new()
            .name(format!("varena-copy-{ordinal}"))
            .spawn(move || m.mover())
            .map_err(|e| Error::Io(format!("spawn copy thread: {e}")))?;
        Ok(Arc::new(Gpu { inner, mover: Mutex::new(Some(mover)) }))
    }

    pub fn device(&self) -> i32 {
        self.inner.ordinal
    }

    pub fn chunk(&self) -> u64 {
        self.inner.chunk
    }

    /// `regions` (`tensorfs_core::client::regions`) of a v2 checkpoint as one weight set. With
    /// `host_fd`, a sealed host layout of the same regions (TensorD's grant) serves the copies;
    /// without, the chunk files do, through the page cache: `files` (a grant's descriptors), or
    /// the checkpoint's own paths.
    #[cfg(feature = "store")]
    pub fn register(self: &Arc<Self>, name: &str, ck: &Checkpoint, regions: &[Vec<String>], host_fd: Option<RawFd>, files: Option<Vec<(ObjectRef, File)>>) -> Result<Arc<WeightSet>> {
        let (layout, source) = open_checkpoint(ck, regions, false, files)?;
        let host = host_fd.map(|fd| HostMem::adopt_sealed(fd, &layout)).transpose()?;
        if host.as_ref().is_some_and(|h| h.window().is_some()) {
            return Err(Error::Invalid("a streamed host layout: register takes a whole sealed layout".into()));
        }
        let parts = parts(ck.header(), &layout);
        self.insert(name, layout, source, host, parts)
    }

    fn insert(self: &Arc<Self>, name: &str, layout: Layout, source: Source, host: Option<HostMem>, parts: Vec<Part>) -> Result<Arc<WeightSet>> {
        let g = &self.inner;
        let _c = CtxGuard::enter(g.ctx)?;
        let base = reserve_va(layout.nbytes.max(REGION_ALIGN))?;
        let layout = Arc::new(layout);
        let regs = (0..layout.regions.len()).map(|_| Reg { home: true, ..Default::default() }).collect();
        let src = Arc::new(Src { layout: layout.clone(), source, host });
        let mut st = g.lock();
        st.sets.push(Some(Set { name: name.into(), base, src, regs, step: Counters::default(), last: Counters::default(), total: Counters::default() }));
        let id = st.sets.len() - 1;
        Ok(Arc::new(WeightSet { gpu: self.clone(), id, base, name: name.into(), layout, parts }))
    }

    /// Idle chunks and unneeded slots back to the driver; returns the bytes released.
    pub fn trim(&self) -> Result<u64> {
        let g = &self.inner;
        let _c = CtxGuard::enter(g.ctx)?;
        let mut st = g.lock();
        let span = st.slot_span;
        let mut live = st.slots.iter().flatten().count();
        for i in 0..st.slots.len() {
            let Some(s) = st.slots[i].as_mut() else { continue };
            if (span == 0 || s.span < span || live > SLOTS) && s.quiet()? {
                g.drop_slot(&mut st, i)?;
                live -= 1;
            }
        }
        g.keep_idle(&mut st, 0)
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
        let _c = CtxGuard::enter(g.ctx)?;
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: out-params, context bound.
        cuda::check("cuMemGetInfo", unsafe { (cuda::driver()?.mem_get_info)(&mut free, &mut total) })?;
        let mut st = g.lock();
        st.resolve();
        let held: Vec<(usize, u32, u64)> = st.slots.iter().flatten().filter_map(|s| s.holds.map(|(set, _)| (set, s.pins, s.span))).collect();
        let sets = st
            .sets
            .iter()
            .enumerate()
            .filter_map(|(id, s)| s.as_ref().map(|s| (id, s)))
            .map(|(id, s)| {
                let regions = &s.src.layout.regions;
                let sum = |f: &dyn Fn(&Reg) -> bool| s.regs.iter().zip(regions).filter(|(r, _)| f(r)).map(|(_, x)| x.span).sum();
                SetStats {
                    name: s.name.clone(),
                    nbytes: s.src.layout.nbytes,
                    regions: regions.len() as u32,
                    home: sum(&|r| r.home),
                    resident: sum(&|r| r.map.is_some()),
                    pinned: sum(&|r| r.map.is_some() && r.pins > 0),
                    pins: s.regs.iter().map(|r| r.pins).sum::<u32>() + held.iter().filter(|h| h.0 == id).map(|h| h.1).sum::<u32>(),
                    slot_bytes: held.iter().filter(|h| h.0 == id).map(|h| h.2).sum(),
                    step: s.step,
                    last: s.last,
                    total: s.total,
                }
            })
            .collect();
        Ok(GpuStats {
            device: g.ordinal,
            chunk: g.chunk,
            steps: st.steps,
            mapped: st.mapped,
            idle: st.idle_bytes(),
            slots: st.slots.iter().flatten().count() as u32,
            slot_bytes: st.slots.iter().flatten().map(|s| s.span).sum(),
            slot_span: st.slot_span,
            driver_free: free as u64,
            driver_total: total as u64,
            sets,
        })
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        self.inner.jobs.0.lock().unwrap().stop = true;
        self.inner.jobs.1.notify_all();
        if let Some(t) = self.mover.lock().unwrap().take() {
            let _ = t.join();
        }
    }
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.st.lock().unwrap()
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
                let mut j = self.jobs.0.lock().unwrap();
                loop {
                    if let Some(job) = j.q.pop_front() {
                        break job;
                    }
                    if j.stop {
                        return;
                    }
                    j = self.jobs.1.wait(j).unwrap();
                }
            };
            job.fill.set(FillState::Running);
            let state = match self.run(&job) {
                Ok((start, ready)) => {
                    let mut st = self.lock();
                    let step = st.steps;
                    st.timings.push(Timing { set: job.set, step, kind: Kind::Copy, a: start, b: ready.clone() });
                    FillState::Done(ready)
                }
                Err(e) => FillState::Failed(self.drained(e)),
            };
            job.fill.set(state);
        }
    }

    /// A failure is reported once nothing it queued is in flight.
    fn drained(&self, e: Error) -> Error {
        let synced = cuda::driver().and_then(|d| {
            // SAFETY: our own stream.
            cuda::check("cuStreamSynchronize", unsafe { (d.stream_synchronize)(self.copy.0) })
        });
        match synced {
            Ok(()) => e,
            Err(s) => Error::Io(format!("{e}; and the copy stream could not drain: {s}")),
        }
    }

    fn run(self: &Arc<Self>, job: &Job) -> Result<(EvRef, EvRef)> {
        let s = self.copy.0;
        for w in &job.waits {
            wait_in(s, w)?;
        }
        let start = self.events.get(true)?;
        start.record(s)?;
        let r = job.region;
        let reg = job.src.layout.region(r)?;
        let n = reg.nbytes;
        match job.src.host.as_ref().filter(|h| !h.hole(r)) {
            Some(h) if n > 0 => {
                h.wait_ready(r)?;
                // SAFETY: inside the live mapping the source keeps; the region is complete.
                let p = unsafe { h.base().add(reg.offset as usize) } as *const u8;
                if h.pin_region(r, reg.offset, align_up(n, PART_ALIGN)).0 {
                    // SAFETY: `n` page-locked bytes kept by the source; the destination is mapped.
                    cuda::check("cuMemcpyHtoDAsync", unsafe { (cuda::driver()?.memcpy_htod_async)(job.dst, p as *const c_void, n as usize, s) })?;
                } else {
                    self.staged(job, Some(p as usize))?;
                }
            }
            _ if n > 0 => self.staged(job, None)?,
            _ => {}
        }
        let ready = self.events.get(true)?;
        ready.record(s)?;
        Ok((start, ready))
    }

    /// The region through the staging buffers, a buffer per piece, read on the reader threads:
    /// from host memory at `host`, else from the chunk files through the page cache.
    fn staged(self: &Arc<Self>, job: &Job, host: Option<usize>) -> Result<()> {
        let n = job.src.layout.region(job.region)?.nbytes;
        let pieces = n.div_ceil(STAGING_BYTES);
        let latch = Arc::new(Latch { left: Mutex::new((pieces as usize, None)), cv: Condvar::new() });
        for i in 0..pieces {
            let (g, src, latch, r, dst) = (self.clone(), job.src.clone(), latch.clone(), job.region, job.dst);
            let (at, len) = (i * STAGING_BYTES, STAGING_BYTES.min(n - i * STAGING_BYTES));
            self.readers.submit(true, move || latch.done(g.piece(&src, r, dst, at, len, host)));
        }
        latch.wait()
    }

    fn piece(&self, src: &Src, r: u32, dst: u64, at: u64, len: u64, host: Option<usize>) -> Result<()> {
        let mut buf = self.staging.take()?;
        let res = (|| {
            match host {
                // SAFETY: `len` bytes inside the region the source keeps mapped; the buffer is ours.
                Some(p) => unsafe { std::ptr::copy_nonoverlapping((p as *const u8).add(at as usize), buf.ptr, len as usize) },
                None => {
                    let reg = src.layout.region(r)?;
                    for it in slice_items(&reg.items, reg.offset + at, reg.offset + at + len) {
                        // SAFETY: a buffered read writes `it.len` bytes, inside this buffer.
                        unsafe { src.source.read_item(&it, buf.ptr.add((it.offset - reg.offset - at) as usize), Mode::Buffered, &self.tally)? };
                    }
                }
            }
            let _c = CtxGuard::enter(self.ctx)?;
            // SAFETY: pinned host bytes we filled; the destination is mapped.
            cuda::check("cuMemcpyHtoDAsync", unsafe { (cuda::driver()?.memcpy_htod_async)(dst + at, buf.ptr as *const c_void, len as usize, self.copy.0) })?;
            let e = self.events.get(false)?;
            e.record(self.copy.0)?;
            buf.last = Some(e);
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

    /// Map `span` bytes at `va` in chunks: idle ones first, then new ones. A driver refusal is
    /// `DeviceFull`, with nothing left mapped.
    fn map(&self, st: &mut State, va: u64, span: u64) -> Result<Vec<Chunk>> {
        let mut sizes = vec![self.chunk; (span / self.chunk) as usize];
        if !span.is_multiple_of(self.chunk) {
            sizes.push(span % self.chunk);
        }
        let mut got: Vec<Chunk> = Vec::with_capacity(sizes.len());
        for (i, &size) in sizes.iter().enumerate() {
            if let Some(handle) = st.idle.get_mut(&size).and_then(Vec::pop) {
                got.push(Chunk { handle, size });
                continue;
            }
            let mut h = self.create(size)?;
            if h.is_none() && st.idle_bytes() > 0 {
                self.keep_idle(st, 0)?;
                h = self.create(size)?;
            }
            match h {
                Some(handle) => got.push(Chunk { handle, size }),
                None => {
                    self.give_back(st, got);
                    return Err(Error::DeviceFull { need: span, deficit: sizes[i..].iter().sum() });
                }
            }
        }
        let d = cuda::driver()?;
        let mut at = va;
        let mut failure = None;
        for c in &got {
            // SAFETY: inside a reservation we own; the handle is unmapped.
            if let Err(e) = cuda::check("cuMemMap", unsafe { (d.mem_map)(at, c.size as usize, 0, c.handle, 0) }) {
                failure = Some(e);
                break;
            }
            at += c.size;
        }
        if failure.is_none() {
            let desc = AccessDesc { location: self.prop.location, flags: cuda::MEM_ACCESS_PROT_READWRITE };
            // SAFETY: the span was just mapped.
            failure = cuda::check("cuMemSetAccess", unsafe { (d.mem_set_access)(va, span as usize, &desc, 1) }).err();
        }
        if let Some(e) = failure {
            if at > va {
                // SAFETY: undoing our own mappings.
                cuda::check("cuMemUnmap", unsafe { (d.mem_unmap)(va, (at - va) as usize) })?;
            }
            self.give_back(st, got);
            return Err(e);
        }
        st.mapped += span;
        Ok(got)
    }

    /// Unmap a span nothing in flight touches; its chunks go idle.
    fn unmap(&self, st: &mut State, va: u64, span: u64, chunks: Vec<Chunk>) -> Result<()> {
        // SAFETY: a mapping we made, unpinned, its copies and releases passed.
        cuda::check("cuMemUnmap", unsafe { (cuda::driver()?.mem_unmap)(va, span as usize) })?;
        st.mapped -= span;
        self.give_back(st, chunks);
        Ok(())
    }

    fn give_back(&self, st: &mut State, chunks: Vec<Chunk>) {
        for c in chunks {
            st.idle.entry(c.size).or_default().push(c.handle);
        }
    }

    /// Release idle chunks, largest first, until at most `keep` bytes are idle.
    fn keep_idle(&self, st: &mut State, keep: u64) -> Result<u64> {
        let d = cuda::driver()?;
        let mut freed = 0;
        while st.idle_bytes() > keep {
            let size = *st.idle.keys().next_back().unwrap();
            let h = st.idle.get_mut(&size).unwrap().pop().unwrap();
            if st.idle[&size].is_empty() {
                st.idle.remove(&size);
            }
            // SAFETY: an unmapped handle we created.
            cuda::check("cuMemRelease", unsafe { (d.mem_release)(h) })?;
            freed += size;
        }
        Ok(freed)
    }

    fn new_slot(&self, st: &mut State, span: u64) -> Result<usize> {
        for i in 0..st.slots.len() {
            if st.slots[i].as_mut().is_some_and(|s| s.span < span) && st.slot(i).quiet()? {
                self.drop_slot(st, i)?;
            }
        }
        let va = reserve_va(span)?;
        let chunks = self.map(st, va, span).inspect_err(|_| {
            let _ = free_va(va, span);
        })?;
        let slot = Slot { va, span, chunks, holds: None, fill: None, fresh: false, pins: 0, released: Vec::new(), pending: false, used: 0 };
        let i = st.slots.iter().position(Option::is_none).unwrap_or(st.slots.len());
        if i == st.slots.len() {
            st.slots.push(None);
        }
        st.slots[i] = Some(slot);
        Ok(i)
    }

    fn drop_slot(&self, st: &mut State, i: usize) -> Result<()> {
        let s = st.slots[i].take().expect("a live slot");
        self.unmap(st, s.va, s.span, s.chunks)?;
        free_va(s.va, s.span)
    }

    /// A slot for streamed region `r`: the one holding it, else a settled one (empty, then
    /// consumed, then, for a bind, a prefetch not yet bound), else a new one. A prefetch makes a
    /// new one only below `SLOTS`, and otherwise does nothing (None).
    fn slot_for(&self, st: &mut State, set: usize, r: u32, urgent: bool) -> Result<Option<usize>> {
        let (span, nbytes) = {
            let reg = st.set(set).src.layout.region(r)?;
            (reg.span, reg.nbytes)
        };
        let held = st.slots.iter().position(|s| s.as_ref().is_some_and(|s| s.holds == Some((set, r))));
        let i = match held {
            Some(i) if !(st.slot(i).pins == 0 && st.slot(i).fill.as_ref().is_some_and(|f| f.failed())) => return Ok(Some(i)),
            Some(i) => i,
            None => {
                let pick = st
                    .slots
                    .iter()
                    .enumerate()
                    .filter_map(|(i, s)| s.as_ref().map(|s| (i, s)))
                    .filter(|(_, s)| s.settled() && s.span >= span && (urgent || !s.pending))
                    .min_by_key(|(_, s)| (s.holds.is_some(), s.pending, s.used))
                    .map(|(i, _)| i);
                match pick {
                    Some(i) => i,
                    None if urgent || st.slots.iter().flatten().count() < SLOTS => self.new_slot(st, span.max(st.slot_span))?,
                    None => return Ok(None),
                }
            }
        };
        let clock = st.clock;
        let src = st.set(set).src.clone();
        let s = st.slot(i);
        let fill = Fill::new();
        let waits = std::mem::take(&mut s.released);
        (s.fill, s.fresh, s.holds, s.pending, s.used) = (Some(fill.clone()), true, Some((set, r)), !urgent, clock);
        let dst = s.va;
        st.set(set).count(|c| {
            c.streamed_bytes += nbytes;
            c.faults += urgent as u64;
        });
        self.submit(Job { set, src, region: r, dst, waits, fill }, urgent);
        Ok(Some(i))
    }

    /// Map region `r` at home and queue its fill.
    fn map_home(&self, st: &mut State, set: usize, r: u32, urgent: bool) -> Result<()> {
        let s = st.set(set);
        let reg = s.src.layout.region(r)?;
        let (va, span, nbytes, src) = (s.base + reg.offset, reg.span, reg.nbytes, s.src.clone());
        let t = Instant::now();
        let chunks = self.map(st, va, span)?;
        let ns = t.elapsed().as_nanos() as u64;
        let fill = Fill::new();
        let reg = st.reg(set, r)?;
        let waits = std::mem::take(&mut reg.released);
        reg.map = Some(Mapping { chunks, fill: fill.clone(), fresh: true });
        st.set(set).count(|c| {
            c.map_ns += ns;
            c.paged_in += nbytes;
            c.faults += urgent as u64;
        });
        self.submit(Job { set, src, region: r, dst: va, waits, fill }, urgent);
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
            if let Some(s) = st.slots[i].as_mut() {
                for e in &s.released {
                    let _ = e.sync();
                }
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
    name: String,
    layout: Arc<Layout>,
    parts: Vec<Part>,
}

impl WeightSet {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn layout(&self) -> &Layout {
        &self.layout
    }
    pub fn parts(&self) -> &[Part] {
        &self.parts
    }
    pub fn device(&self) -> i32 {
        self.gpu.inner.ordinal
    }

    /// The watermark: `regions` live at their home VA, the rest stream through slots. A region
    /// left out stays mapped until evicted, and serves binds from home meanwhile.
    pub fn home(&self, regions: &[u32]) -> Result<()> {
        let g = &self.gpu.inner;
        let _c = CtxGuard::enter(g.ctx)?;
        let mut st = g.lock();
        let n = self.layout.regions.len() as u32;
        if let Some(r) = regions.iter().find(|&&r| r >= n) {
            return Err(Error::Invalid(format!("region {r} of {n}")));
        }
        for (r, reg) in st.set(self.id).regs.iter_mut().enumerate() {
            reg.home = regions.contains(&(r as u32));
        }
        st.slot_span = st
            .sets
            .iter()
            .flatten()
            .flat_map(|s| s.regs.iter().zip(&s.src.layout.regions).filter(|(r, _)| !r.home).map(|(_, x)| x.span))
            .max()
            .unwrap_or(0);
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
            st.set(self.id).count(|c| c.binds += 1);
            let reg = st.reg(self.id, region)?;
            if reg.pins == 0 && reg.map.as_ref().is_some_and(|m| m.fill.failed()) {
                let m = reg.map.take().unwrap();
                let (va, span) = self.span(region);
                g.unmap(&mut st, va, span, m.chunks)?;
            }
            let reg = st.reg(self.id, region)?;
            if reg.map.is_some() || reg.home {
                if reg.map.is_none() {
                    g.map_home(&mut st, self.id, region, true)?;
                }
                let reg = st.reg(self.id, region)?;
                reg.pins += 1;
                let m = reg.map.as_mut().unwrap();
                (At::Home, self.span(region).0, m.fill.clone(), std::mem::take(&mut m.fresh))
            } else {
                let i = g.slot_for(&mut st, self.id, region, true)?.expect("a bind always gets a slot");
                let s = st.slot(i);
                s.pins += 1;
                s.pending = false;
                (At::Slot(i), s.va, s.fill.clone().expect("a held slot has a fill"), std::mem::take(&mut s.fresh))
            }
        };
        g.hurry(&fill);
        let ready = match fill.wait() {
            Ok(ready) => ready,
            Err(e) => {
                self.release(at, region, None);
                return Err(e);
            }
        };
        if !ready.done()? {
            let s = stream as CUstream;
            let (a, b) = (g.events.get(true)?, g.events.get(true)?);
            a.record(s)?;
            wait_in(s, &ready)?;
            b.record(s)?;
            let mut st = g.lock();
            let step = st.steps;
            st.timings.push(Timing { set: self.id, step, kind: Kind::Stall, a, b });
        }
        let nbytes = self.layout.regions[region as usize].nbytes;
        Ok(Pinned { ws: self.clone(), at, region, ptr, nbytes, fresh, stream, ready })
    }

    /// Start filling region `r` ahead of its bind: at home, or into a free slot. Never waits.
    pub fn prefetch(&self, region: u32) -> Result<()> {
        let g = &self.gpu.inner;
        let _c = CtxGuard::enter(g.ctx)?;
        let mut st = g.lock();
        let reg = st.reg(self.id, region)?;
        if reg.map.is_some() {
            return Ok(());
        }
        if reg.home {
            g.map_home(&mut st, self.id, region, false)?;
        } else {
            g.slot_for(&mut st, self.id, region, false)?;
        }
        Ok(())
    }

    /// Unmap `regions` (all when None) at home, those unpinned whose fill and release events
    /// completed; returns the bytes unmapped. Their chunks go idle, beyond a small cache back
    /// to the driver.
    pub fn evict(&self, regions: Option<&[u32]>) -> Result<u64> {
        let g = &self.gpu.inner;
        let _c = CtxGuard::enter(g.ctx)?;
        let mut st = g.lock();
        let all: Vec<u32> = (0..self.layout.regions.len() as u32).collect();
        let mut freed = 0;
        for &r in regions.unwrap_or(&all) {
            let reg = st.reg(self.id, r)?;
            let Some(m) = &reg.map else { continue };
            if reg.pins > 0 || !m.fill.landed()? || !passed(&mut reg.released)? {
                continue;
            }
            let m = reg.map.take().unwrap();
            let (va, span) = self.span(r);
            g.unmap(&mut st, va, span, m.chunks)?;
            st.set(self.id).count(|c| c.evicted += span);
            freed += span;
        }
        g.keep_idle(&mut st, IDLE_KEEP)?;
        Ok(freed)
    }

    fn span(&self, r: u32) -> (u64, u64) {
        let reg = &self.layout.regions[r as usize];
        (self.base + reg.offset, reg.span)
    }

    fn release(&self, at: At, region: u32, ev: Option<EvRef>) {
        let mut st = self.gpu.inner.lock();
        let released = match at {
            At::Home => match st.reg(self.id, region) {
                Ok(reg) => {
                    reg.pins -= 1;
                    &mut reg.released
                }
                Err(_) => return,
            },
            At::Slot(i) => {
                let s = st.slot(i);
                s.pins -= 1;
                &mut s.released
            }
        };
        if let Some(e) = ev {
            let _ = passed(released);
            released.push(e);
        }
    }
}

impl Drop for WeightSet {
    fn drop(&mut self) {
        let g = &self.gpu.inner;
        let Ok(_c) = CtxGuard::enter(g.ctx) else { return };
        let set = {
            let mut st = g.lock();
            for s in st.slots.iter_mut().flatten() {
                if s.holds.is_some_and(|(set, _)| set == self.id) {
                    s.holds = None;
                }
            }
            st.sets[self.id].take()
        };
        let Some(mut set) = set else { return };
        for reg in &set.regs {
            if let Some(m) = &reg.map {
                if let Ok(ready) = m.fill.wait() {
                    let _ = ready.sync();
                }
            }
            for e in &reg.released {
                let _ = e.sync();
            }
        }
        let mut st = g.lock();
        for (reg, x) in set.regs.iter_mut().zip(&self.layout.regions) {
            if let Some(m) = reg.map.take() {
                let _ = g.unmap(&mut st, set.base + x.offset, x.span, m.chunks);
            }
        }
        let _ = g.keep_idle(&mut st, IDLE_KEEP);
        let _ = free_va(set.base, self.layout.nbytes.max(REGION_ALIGN));
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
        let ev = CtxGuard::enter(g.ctx).and_then(|_c| {
            let e = g.events.get(false)?;
            e.record(self.stream as CUstream)?;
            Ok(e)
        });
        match ev {
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
    name: String,
    parts: Vec<Part>,
    src: Src,
    regs: Mutex<Vec<(Option<Arc<Anon>>, bool)>>,
}

impl Host {
    #[cfg(feature = "store")]
    pub fn register(&self, name: &str, ck: &Checkpoint, regions: &[Vec<String>], host_fd: Option<RawFd>, files: Option<Vec<(ObjectRef, File)>>) -> Result<Arc<HostSet>> {
        let (layout, source) = open_checkpoint(ck, regions, false, files)?;
        let host = host_fd.map(|fd| HostMem::adopt_sealed(fd, &layout)).transpose()?;
        if host.as_ref().is_some_and(|h| h.window().is_some()) {
            return Err(Error::Invalid("a streamed host layout: register takes a whole sealed layout".into()));
        }
        let parts = parts(ck.header(), &layout);
        let regs = Mutex::new(vec![(None, false); layout.regions.len()]);
        Ok(Arc::new(HostSet { name: name.into(), parts, src: Src { layout: Arc::new(layout), source, host }, regs }))
    }
}

impl HostSet {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn layout(&self) -> &Layout {
        &self.src.layout
    }
    pub fn parts(&self) -> &[Part] {
        &self.parts
    }

    pub fn bind(self: &Arc<Self>, region: u32) -> Result<HostPinned> {
        let reg = self.src.layout.region(region)?;
        let mut regs = self.regs.lock().unwrap();
        let (buf, bound) = &mut regs[region as usize];
        let (ptr, buf) = match self.src.host.as_ref().filter(|h| !h.hole(region)) {
            Some(h) => {
                h.wait_ready(region)?;
                (h.base() as u64 + reg.offset, None)
            }
            None => {
                if buf.is_none() {
                    *buf = Some(Arc::new(Anon::read(&self.src, region)?));
                }
                let b = buf.clone().unwrap();
                (b.ptr as u64, Some(b))
            }
        };
        let fresh = !std::mem::replace(bound, true);
        Ok(HostPinned { _set: self.clone(), _buf: buf, region, ptr, nbytes: reg.nbytes, fresh, writable: self.src.host.as_ref().is_none_or(|h| h.hole(region)) })
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
    fn read(src: &Src, r: u32) -> Result<Anon> {
        let reg = src.layout.region(r)?;
        let len = align_up(reg.nbytes.max(1), PART_ALIGN) as usize;
        // SAFETY: a fresh private anonymous mapping.
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0) };
        if ptr == libc::MAP_FAILED {
            return Err(Error::Io(format!("map {len} bytes: {}", std::io::Error::last_os_error())));
        }
        let a = Anon { ptr: ptr as *mut u8, len };
        let tally = IoTally::default();
        for it in &reg.items {
            // SAFETY: a buffered read writes `it.len` bytes inside the mapping.
            unsafe { src.source.read_item(it, a.ptr.add((it.offset - reg.offset) as usize), Mode::Buffered, &tally)? };
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
/// paths): what `register` maps and what a sealed host layout for it holds. An owner fills one
/// with this (or `seal`), so the two agree by construction; `HostMem::adopt_sealed` checks the
/// digest.
#[cfg(feature = "store")]
pub fn open_checkpoint(ck: &Checkpoint, regions: &[Vec<String>], direct_io: bool, files: Option<Vec<(ObjectRef, File)>>) -> Result<(Layout, Source)> {
    let components: Vec<&str> = regions.iter().flatten().filter_map(|t| t.split_once('/').map(|(c, _)| c)).collect();
    let plan = crate::checkpoint::read_plan(ck.header(), &components);
    let files = match files {
        Some(f) => f,
        None => crate::checkpoint::chunk_files(ck, &plan)?,
    };
    Ok((Layout::build(&plan, regions)?, Source::files(files, direct_io, 0)))
}

/// `regions` of `ck` in a sealed host layout, filled in this process from the chunk files
/// (`files`, else the checkpoint's paths): the memfd `register(host_fd=)` adopts, the same
/// layout by construction.
#[cfg(feature = "store")]
pub fn seal(ck: &Checkpoint, regions: &[Vec<String>], name: &str, threads: usize, files: Option<Vec<(ObjectRef, File)>>) -> Result<OwnedFd> {
    let (layout, source) = open_checkpoint(ck, regions, true, files)?;
    crate::host::fill_sealed(name, &layout, &source, threads, &IoTally::default(), None)
}

fn reserve_va(size: u64) -> Result<u64> {
    let mut base = 0u64;
    // SAFETY: out-param; size is a granularity multiple.
    cuda::check("cuMemAddressReserve", unsafe { (cuda::driver()?.mem_address_reserve)(&mut base, size as usize, REGION_ALIGN as usize, 0, 0) })?;
    Ok(base)
}

fn free_va(base: u64, size: u64) -> Result<()> {
    // SAFETY: a reservation we made, fully unmapped.
    cuda::check("cuMemAddressFree", unsafe { (cuda::driver()?.mem_address_free)(base, size as usize) })
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

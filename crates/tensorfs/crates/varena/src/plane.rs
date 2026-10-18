//! The plane: weight sets, device regions, fills, leases and facts.
//!
//! One lock (`Core`) holds every book; it is held across fast driver calls (map, unmap, record,
//! enqueue) and never across I/O. Waits happen on the condvar or on one targeted event. Nothing
//! here decides what should be resident: callers want, acquire and drop; a region stays mapped
//! until dropped, and only a live lease keeps a drop from unmapping it.

use std::collections::{HashMap, VecDeque};
use std::os::fd::RawFd;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Instant;

use tensorfs_core::store2::Checkpoint;

use crate::cuda::{self, AllocationProp, CUcontext, CUdevice, CUstream, CtxGuard};
use crate::cursor::Cur;
use crate::device::{self, Chunk, CopyJob, CopyPath, CopySrc, Copied, EvRef, EventPool, Mover, MoverCtx};
use crate::host::HostMem;
use crate::io::{Files, IoTally, Readers};
use crate::layout::{Layout, PART_ALIGN, REGION_ALIGN};
use crate::{Error, Result};

/// Everything the plane is told at open.
#[derive(Debug, Clone)]
pub struct PlaneConfig {
    /// CUDA ordinals this plane serves.
    pub devices: Vec<i32>,
    /// Reader threads: chunk reads and host copies into the staging slots.
    pub readers: usize,
    /// Copy streams per device, each with its own staging slots.
    pub copy_streams: usize,
    /// Pinned staging slots per copy stream, and their size: one read request and one copy each.
    pub staging_buffers: usize,
    pub staging_bytes: u64,
    /// Device memory the plane never maps into: the driver's own room (and a display's).
    pub floor_bytes: u64,
}

impl Default for PlaneConfig {
    fn default() -> Self {
        PlaneConfig {
            devices: Vec::new(),
            readers: 16,
            copy_streams: 1,
            staging_buffers: 16,
            staging_bytes: 16 << 20,
            floor_bytes: 256 << 20,
        }
    }
}

/// Where a weight set's host bytes come from.
#[derive(Debug, Clone, Copy)]
pub enum Host {
    /// The chunk files, through the staging slots. `direct`: a chunk the page cache does not
    /// hold reads O_DIRECT; otherwise every read is buffered and stays in the page cache.
    PageCache { direct: bool },
    /// A sealed host layout of this weight set (`host::open_sealed`, its owner's, borrowed for
    /// the call): Ready regions copy from it, page-locked on first copy; holes read the chunk
    /// files, buffered.
    Sealed(RawFd),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WsId(pub u32);

#[derive(Debug, Clone)]
pub struct Event {
    pub kind: &'static str,
    pub ws: Option<u32>,
    pub region: Option<u32>,
    pub device: Option<i32>,
    pub bytes: u64,
    pub detail: String,
}

const EVENT_RING: usize = 4096;

// ---------------------------------------------------------------- books

pub(crate) enum DState {
    Absent,
    /// Submitted; the mover has not reported it yet.
    Queued,
    Filling(EvRef),
    Ready,
    Invalid(Error),
}

pub(crate) struct DReg {
    pub mem: Vec<Chunk>,
    pub state: DState,
    pub live: u32,
    /// Release events of finished leases the device may not have reached.
    pub pending: Vec<EvRef>,
    /// Plane-clock nanoseconds of the last map, acquire or release.
    pub last_use: u64,
}

pub(crate) struct DevWs {
    pub base: u64,
    pub regs: Vec<DReg>,
    pub views: u64,
    /// Bumped before any unmap: a view exported under an older epoch may name unmapped pages.
    pub epoch: u64,
}

pub(crate) struct Ws {
    pub layout: Arc<Layout>,
    pub files: Arc<Files>,
    pub sealed: Option<Arc<HostMem>>,
    pub devs: Vec<Option<DevWs>>,
}

#[derive(Debug, Clone, Default)]
pub struct DevCounters {
    pub copies: u64,
    /// One DMA from page-locked host memory (a sealed region).
    pub host_copy_bytes: u64,
    pub host_copy_ns: u64,
    /// Host memory this process could not page-lock, through the staging slots.
    pub bounce_copy_bytes: u64,
    pub bounce_copy_ns: u64,
    /// The chunk files, through the staging slots.
    pub disk_copy_bytes: u64,
    pub disk_copy_ns: u64,
    /// Acquires that found their region unmapped or failed, and filled it.
    pub misses: u64,
}

pub(crate) struct Timing {
    pub dev: usize,
    pub path: CopyPath,
    pub a: EvRef,
    pub b: EvRef,
    pub bytes: u64,
}

pub(crate) struct Core {
    /// Latched on the first ambiguous driver outcome or lost lease: every later call refuses.
    pub poisoned: Option<String>,
    pub sets: Vec<Option<Ws>>,
    /// Bytes mapped per device: regions and rings.
    pub mapped: Vec<u64>,
    pub cursors: HashMap<u64, Cur>,
    pub next_cursor: u64,
    pub dstats: Vec<DevCounters>,
    pub register_failed: u64,
    pub events: VecDeque<Event>,
    pub events_dropped: u64,
    pub timings: VecDeque<Timing>,
}

pub(crate) struct DeviceRt {
    pub ordinal: i32,
    pub dev: CUdevice,
    pub ctx: CUcontext,
    pub uuid: String,
    pub granularity: u64,
    pub prop: AllocationProp,
    pub movers: Vec<Mover>,
    next_mover: AtomicUsize,
}
unsafe impl Send for DeviceRt {}
unsafe impl Sync for DeviceRt {}

impl DeviceRt {
    pub fn mover(&self) -> usize {
        self.next_mover.fetch_add(1, Ordering::Relaxed) % self.movers.len()
    }
    pub fn bind(&self) -> Result<CtxGuard> {
        CtxGuard::enter(self.ctx)
    }
}

pub(crate) struct Inner {
    pub floor: u64,
    pub devices: Vec<DeviceRt>,
    pub events: Vec<Arc<EventPool>>,
    pub readers: Arc<Readers>,
    pub tally: Arc<IoTally>,
    pub core: Mutex<Core>,
    pub cv: Condvar,
    pub clock: Instant,
}

impl Core {
    pub fn poison(&mut self, cause: String) -> Error {
        Error::Poisoned(self.poisoned.get_or_insert(cause).clone())
    }
    /// Latch a `Poisoned` outcome reported by a copy or a driver call.
    pub fn latch(&mut self, e: &Error) {
        if let Error::Poisoned(p) = e {
            self.poison(p.clone());
        }
    }
    pub fn check(&self) -> Result<()> {
        match &self.poisoned {
            Some(p) => Err(Error::Poisoned(p.clone())),
            None => Ok(()),
        }
    }
    pub fn ws(&self, id: WsId) -> Result<&Ws> {
        self.sets.get(id.0 as usize).and_then(Option::as_ref).ok_or_else(|| closed(id))
    }
    pub fn ws_mut(&mut self, id: WsId) -> Result<&mut Ws> {
        self.sets.get_mut(id.0 as usize).and_then(Option::as_mut).ok_or_else(|| closed(id))
    }
    pub fn event(&mut self, kind: &'static str, ws: WsId, region: u32, device: i32, bytes: u64, detail: String) {
        if self.events.len() == EVENT_RING {
            self.events.pop_front();
            self.events_dropped += 1;
        }
        self.events.push_back(Event { kind, ws: Some(ws.0), region: Some(region), device: Some(device), bytes, detail });
    }
    /// Record a finished copy's timing pair and count it.
    pub fn copied(&mut self, d: usize, cp: &Copied) {
        self.dstats[d].copies += 1;
        if cp.refused {
            self.register_failed += 1;
        }
        self.timings.push_back(Timing { dev: d, path: cp.path, a: cp.start.clone(), b: cp.ready.clone(), bytes: cp.bytes });
    }
}

fn closed(id: WsId) -> Error {
    Error::Closed(format!("weight set {} was closed", id.0))
}

/// Fold copies whose timing pair resolved into the counters; keep the rest.
fn resolve_timings(c: &mut Core) {
    let mut keep = VecDeque::new();
    while let Some(t) = c.timings.pop_front() {
        if !t.b.done().unwrap_or(false) {
            keep.push_back(t);
            continue;
        }
        let ns = (t.b.since(&t.a).unwrap_or(0.0) as f64 * 1e6) as u64;
        let s = &mut c.dstats[t.dev];
        let (bytes, total) = match t.path {
            CopyPath::Pinned => (&mut s.host_copy_bytes, &mut s.host_copy_ns),
            CopyPath::Bounced => (&mut s.bounce_copy_bytes, &mut s.bounce_copy_ns),
            CopyPath::Files => (&mut s.disk_copy_bytes, &mut s.disk_copy_ns),
        };
        *bytes += t.bytes;
        *total += ns;
    }
    c.timings = keep;
}

/// Commit `Filling -> Ready` when the copy event has completed.
fn commit(reg: &mut DReg) -> Result<()> {
    if let DState::Filling(ev) = &reg.state {
        if ev.done()? {
            reg.state = DState::Ready;
        }
    }
    Ok(())
}

pub(crate) fn prune(pending: &mut Vec<EvRef>) {
    pending.retain(|e| !e.done().unwrap_or(false));
}

pub(crate) fn devws(c: &mut Core, ws: WsId, d: usize) -> Result<&mut DevWs> {
    c.check()?;
    let w = c.ws_mut(ws)?;
    if w.devs[d].is_none() {
        let base = device::reserve_va(w.layout.nbytes)?;
        let regs = (0..w.layout.regions.len())
            .map(|_| DReg { mem: Vec::new(), state: DState::Absent, live: 0, pending: Vec::new(), last_use: 0 })
            .collect();
        w.devs[d] = Some(DevWs { base, regs, views: 0, epoch: 1 });
    }
    Ok(w.devs[d].as_mut().unwrap())
}

/// A region's chunks as copy runs: (chunk, offset in the region, length).
pub(crate) fn runs(layout: &Layout, r: u32) -> Vec<(u32, u64, u64)> {
    let region = &layout.regions[r as usize];
    region.chunks.iter().map(|c| (c.index, c.offset - region.offset, c.length)).collect()
}

impl Inner {
    pub(crate) fn now(&self) -> u64 {
        self.clock.elapsed().as_nanos() as u64
    }

    /// Whether `need` more bytes fit on device `d` (its context bound) above the floor:
    /// `Floor` when they never can, `DeviceFull` with what to free when they cannot now.
    pub(crate) fn room(&self, need: u64) -> Result<()> {
        if need == 0 {
            return Ok(());
        }
        let (free, total) = device::mem_info()?;
        let capacity = total.saturating_sub(self.floor);
        if need > capacity {
            return Err(Error::Floor { need, capacity });
        }
        match (need + self.floor).checked_sub(free) {
            Some(deficit) if deficit > 0 => Err(Error::DeviceFull { deficit }),
            _ => Ok(()),
        }
    }

    /// One physical allocation of `span` mapped at `va`. The caller binds the context and counts it.
    pub(crate) fn map_span(&self, d: usize, va: u64, span: u64) -> Result<Vec<Chunk>> {
        let dev = &self.devices[d];
        let mem = device::create(&dev.prop, span)?;
        if let Err(e) = device::map_span(va, &[mem], dev.dev, cuda::MEM_ACCESS_PROT_READWRITE) {
            let _ = device::release(mem.handle);
            return Err(e);
        }
        Ok(vec![mem])
    }

    /// Map every unmapped region of `regions`, all or none: a refusal unmaps what this call
    /// mapped and reports what to free. The caller binds the context.
    pub(crate) fn map_regions(self: &Arc<Self>, c: &mut Core, ws: WsId, d: usize, regions: &[u32]) -> Result<()> {
        let layout = c.ws(ws)?.layout.clone();
        let base = devws(c, ws, d)?.base;
        let todo: Vec<u32> = {
            let dw = devws(c, ws, d)?;
            let mut v: Vec<u32> = regions.iter().copied().filter(|&r| dw.regs[r as usize].mem.is_empty()).collect();
            v.dedup();
            v
        };
        let need: u64 = todo.iter().map(|&r| layout.regions[r as usize].span).sum();
        self.room(need)?;
        let now = self.now();
        for (i, &r) in todo.iter().enumerate() {
            let region = &layout.regions[r as usize];
            match self.map_span(d, base + region.offset, region.span) {
                Ok(mem) => {
                    c.mapped[d] += region.span;
                    let reg = &mut devws(c, ws, d)?.regs[r as usize];
                    reg.mem = mem;
                    reg.last_use = now;
                }
                Err(e) => {
                    for &q in &todo[..i] {
                        self.unmap_region(c, ws, d, q)?;
                    }
                    if let Error::Poisoned(p) = e {
                        return Err(c.poison(p));
                    }
                    let done: u64 = todo[..i].iter().map(|&q| layout.regions[q as usize].span).sum();
                    return Err(match e {
                        Error::DeviceFull { .. } => self.room(need).err().unwrap_or(Error::DeviceFull { deficit: need - done }),
                        e => e,
                    });
                }
            }
        }
        Ok(())
    }

    /// Unmap one unleased region once its copies and readers are past: release its memory.
    pub(crate) fn unmap_region(self: &Arc<Self>, c: &mut Core, ws: WsId, d: usize, r: u32) -> Result<u64> {
        let span = c.ws(ws)?.layout.regions[r as usize].span;
        let offset = c.ws(ws)?.layout.regions[r as usize].offset;
        let dw = c.ws_mut(ws)?.devs[d].as_mut().expect("a mapped region has a reservation");
        let base = dw.base;
        dw.epoch += 1;
        let reg = &mut dw.regs[r as usize];
        let waited = match &reg.state {
            DState::Filling(ev) => ev.sync(),
            _ => Ok(()),
        }
        .and_then(|_| reg.pending.drain(..).try_for_each(|e| e.sync()));
        let mem = std::mem::take(&mut reg.mem);
        reg.state = DState::Absent;
        if let Err(e) = waited.and_then(|_| device::unmap_span(base + offset, span)) {
            return Err(c.poison(format!("unmapping region {r} of weight set {} failed: {e}", ws.0)));
        }
        for m in mem {
            if let Err(e) = device::release(m.handle) {
                return Err(c.poison(format!("releasing region {r} of weight set {} failed: {e}", ws.0)));
            }
        }
        c.mapped[d] -= span;
        Ok(span)
    }

    /// Queue the copy that fills mapped region `r` on device slot `d`: from the sealed layout
    /// when it holds the region, else from the chunk files.
    pub(crate) fn submit_fill(self: &Arc<Self>, c: &mut Core, ws: WsId, d: usize, r: u32) -> Result<()> {
        c.check()?;
        let w = c.ws_mut(ws)?;
        let region = w.layout.regions[r as usize].clone();
        let src = match &w.sealed {
            Some(host) if !host.hole(r) => CopySrc::Sealed { host: host.clone(), region: r, offset: region.offset },
            _ => CopySrc::Files { files: w.files.clone(), runs: runs(&w.layout, r) },
        };
        let dw = w.devs[d].as_mut().unwrap();
        dw.regs[r as usize].state = DState::Queued;
        let inner = self.clone();
        let ordinal = self.devices[d].ordinal;
        let job = CopyJob {
            waits: Vec::new(),
            dst: dw.base + region.offset,
            len: region.nbytes,
            src,
            done: Box::new(move |res: Result<Copied>| {
                let mut c = inner.core.lock().unwrap();
                if let Ok(cp) = &res {
                    c.copied(d, cp);
                }
                let Some(reg) = c.ws_mut(ws).ok().and_then(|w| w.devs[d].as_mut()).map(|dw| &mut dw.regs[r as usize]) else {
                    return;
                };
                match res {
                    Ok(cp) => reg.state = DState::Filling(cp.ready),
                    Err(e) => {
                        reg.state = DState::Invalid(e.clone());
                        c.latch(&e);
                        c.event("fill_failed", ws, r, ordinal, region.nbytes, e.to_string());
                    }
                }
                drop(c);
                inner.cv.notify_all();
            }),
        };
        let m = self.devices[d].mover();
        if let Err(e) = self.devices[d].movers[m].submit(job) {
            c.ws_mut(ws)?.devs[d].as_mut().unwrap().regs[r as usize].state = DState::Invalid(e.clone());
            return Err(e);
        }
        Ok(())
    }

    /// Map and fill `regions` that are not resident on device slot `d`. The caller binds it.
    pub(crate) fn want(self: &Arc<Self>, c: &mut Core, ws: WsId, d: usize, regions: &[u32]) -> Result<()> {
        self.map_regions(c, ws, d, regions)?;
        for &r in regions {
            let reg = &mut devws(c, ws, d)?.regs[r as usize];
            commit(reg)?;
            if matches!(reg.state, DState::Absent | DState::Invalid(_)) {
                self.submit_fill(c, ws, d, r)?;
            }
        }
        Ok(())
    }

    /// A lease on region `r` in place. A region unmapped, or whose last fill failed, is filled
    /// on demand (a miss): a failure is the cause this acquire reports, never the region's state
    /// for the rest of the process. No host block once the fill is queued.
    pub(crate) fn acquire_home(self: &Arc<Self>, ws: WsId, d: usize, r: u32, stream: u64) -> Result<Lease> {
        let _g = self.devices[d].bind()?;
        let mut c = self.core.lock().unwrap();
        let region = c.ws(ws)?.layout.region(r)?.clone();
        {
            let reg = &mut devws(&mut c, ws, d)?.regs[r as usize];
            commit(reg)?;
            if matches!(reg.state, DState::Absent | DState::Invalid(_)) {
                c.dstats[d].misses += 1;
                self.want(&mut c, ws, d, &[r])?;
            }
        }
        let fill = loop {
            match &devws(&mut c, ws, d)?.regs[r as usize].state {
                DState::Queued => c = self.cv.wait(c).unwrap(),
                DState::Invalid(e) => return Err(e.clone()),
                DState::Absent => return Err(Error::Invalid(format!("region {r} was dropped before use"))),
                DState::Filling(ev) => break Some(ev.clone()),
                DState::Ready => break None,
            }
        };
        if let Some(ev) = &fill {
            wait_on(stream, ev)?;
        }
        let now = self.now();
        let dw = devws(&mut c, ws, d)?;
        let reg = &mut dw.regs[r as usize];
        reg.live += 1;
        reg.last_use = now;
        Ok(Lease {
            inner: self.clone(),
            ptr: dw.base + region.offset,
            nbytes: region.nbytes,
            region: r,
            ring_offset: None,
            ws,
            dev: d,
            kind: LeaseKind::Home,
            released: AtomicBool::new(false),
        })
    }

    pub(crate) fn view(self: &Arc<Self>, ws: WsId, d: usize) -> Result<View> {
        let _g = self.devices[d].bind()?;
        let mut c = self.core.lock().unwrap();
        let nbytes = c.ws(ws)?.layout.nbytes;
        let dw = devws(&mut c, ws, d)?;
        dw.views += 1;
        Ok(View { inner: self.clone(), ws, dev: d, ptr: dw.base, nbytes, epoch: dw.epoch })
    }
}

/// Make the consumer `stream` wait for `ev` on the device: no host block.
pub(crate) fn wait_on(stream: u64, ev: &EvRef) -> Result<()> {
    let drv = cuda::driver()?;
    // SAFETY: the consumer's stream in its bound context; the event is live.
    cuda::check("cuStreamWaitEvent", unsafe { (drv.stream_wait_event)(stream as CUstream, ev.raw(), 0) })
}

pub struct Plane {
    pub(crate) inner: Arc<Inner>,
}

impl Plane {
    pub fn open(cfg: PlaneConfig) -> Result<Plane> {
        if cfg.staging_bytes == 0 || !cfg.staging_bytes.is_multiple_of(PART_ALIGN) {
            return Err(Error::Invalid(format!("staging slots of {} bytes: a multiple of {PART_ALIGN}", cfg.staging_bytes)));
        }
        let readers = Arc::new(Readers::new(cfg.readers));
        let mut devices = Vec::new();
        let mut pools = Vec::new();
        for &ordinal in &cfg.devices {
            let (dev, ctx) = cuda::primary_context(ordinal)?;
            let _g = CtxGuard::enter(ctx)?;
            if cuda::attribute(dev, cuda::ATTR_VMM_SUPPORTED)? != 1 {
                return Err(Error::Invalid(format!("device {ordinal} has no virtual memory management")));
            }
            let prop = AllocationProp::device(dev);
            let d = cuda::driver()?;
            let mut g = 0usize;
            // SAFETY: out-param.
            cuda::check("cuMemGetAllocationGranularity", unsafe {
                (d.mem_get_allocation_granularity)(&mut g, &prop, cuda::MEM_GRANULARITY_MINIMUM)
            })?;
            let g = g as u64;
            if g == 0 || !REGION_ALIGN.is_multiple_of(g) {
                return Err(Error::Invalid(format!("device {ordinal}: granularity {g} does not divide {REGION_ALIGN}")));
            }
            let events = Arc::new(EventPool::new(ctx));
            let movers = (0..cfg.copy_streams.max(1))
                .map(|i| {
                    let m = MoverCtx { ctx, events: events.clone(), readers: readers.clone(), staging: (cfg.staging_buffers, cfg.staging_bytes) };
                    Mover::start(format!("tfs-gpu-copy-{ordinal}.{i}"), m)
                })
                .collect::<Result<Vec<_>>>()?;
            devices.push(DeviceRt { ordinal, dev, ctx, uuid: cuda::uuid(dev)?, granularity: g, prop, movers, next_mover: AtomicUsize::new(0) });
            pools.push(events);
        }
        let n = devices.len();
        Ok(Plane {
            inner: Arc::new(Inner {
                floor: cfg.floor_bytes,
                devices,
                events: pools,
                readers,
                tally: Arc::new(IoTally::default()),
                core: Mutex::new(Core {
                    poisoned: None,
                    sets: Vec::new(),
                    mapped: vec![0; n],
                    cursors: HashMap::new(),
                    next_cursor: 1,
                    dstats: vec![DevCounters::default(); n],
                    register_failed: 0,
                    events: VecDeque::new(),
                    events_dropped: 0,
                    timings: VecDeque::new(),
                }),
                cv: Condvar::new(),
                clock: Instant::now(),
            }),
        })
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, Core> {
        self.inner.core.lock().unwrap()
    }

    /// Err(Poisoned) once the plane has latched a poison.
    pub fn check(&self) -> Result<()> {
        self.lock().check()
    }

    pub(crate) fn dev(&self, ordinal: i32) -> Result<usize> {
        self.inner
            .devices
            .iter()
            .position(|d| d.ordinal == ordinal)
            .ok_or_else(|| Error::Invalid(format!("device {ordinal} is not served by this plane")))
    }

    /// Register `regions` of `ck` (`tensorfs_core::client::regions`, or any chunk grouping
    /// `Layout::build` accepts) as one weight set, its host bytes from `host`.
    pub fn register(&self, ck: &Checkpoint, regions: &[Vec<u32>], host: Host) -> Result<WsId> {
        self.check()?;
        let layout = Layout::build(ck.header(), regions)?;
        let (direct, sealed) = match host {
            Host::PageCache { direct } => (direct, None),
            Host::Sealed(fd) => (false, Some(Arc::new(HostMem::adopt_sealed(fd, &layout)?))),
        };
        let files = Arc::new(Files::new(ck.chunk_paths().to_vec(), direct, self.inner.tally.clone()));
        let ws = Ws { layout: Arc::new(layout), files, sealed, devs: (0..self.inner.devices.len()).map(|_| None).collect() };
        let mut c = self.lock();
        c.sets.push(Some(ws));
        Ok(WsId(c.sets.len() as u32 - 1))
    }

    pub fn layout(&self, ws: WsId) -> Result<Arc<Layout>> {
        Ok(self.lock().ws(ws)?.layout.clone())
    }
    pub fn devices(&self) -> Vec<(i32, String)> {
        self.inner.devices.iter().map(|d| (d.ordinal, d.uuid.clone())).collect()
    }

    fn regions_or_all(c: &Core, ws: WsId, regions: Option<&[u32]>) -> Result<Vec<u32>> {
        let n = c.ws(ws)?.layout.regions.len() as u32;
        match regions {
            None => Ok((0..n).collect()),
            Some(r) => match r.iter().find(|&&x| x >= n) {
                Some(bad) => Err(Error::Invalid(format!("region {bad} of {n}"))),
                None => Ok(r.to_vec()),
            },
        }
    }

    /// Make `regions` (all when None) resident on a device: map what is not (all or none:
    /// `DeviceFull`/`Floor` otherwise) and fill it. Returns at once; the ticket reports completion.
    pub fn want(&self, ws: WsId, ordinal: i32, regions: Option<&[u32]>) -> Result<Ticket> {
        let d = self.dev(ordinal)?;
        let _g = self.inner.devices[d].bind()?;
        let mut c = self.lock();
        c.check()?;
        let regions = Self::regions_or_all(&c, ws, regions)?;
        self.inner.want(&mut c, ws, d, &regions)?;
        Ok(Ticket { inner: self.inner.clone(), ws, dev: d, regions })
    }

    /// Unmap `regions` (all when None) on a device and release their memory. A leased region
    /// stays; a region still filling is unmapped once its copy lands. Returns the bytes freed.
    pub fn drop_regions(&self, ws: WsId, ordinal: i32, regions: Option<&[u32]>) -> Result<u64> {
        let d = self.dev(ordinal)?;
        let _g = self.inner.devices[d].bind()?;
        let mut c = self.lock();
        c.check()?;
        let regions = Self::regions_or_all(&c, ws, regions)?;
        let mut freed = 0;
        for r in regions {
            loop {
                let Some(dw) = c.ws_mut(ws)?.devs[d].as_mut() else { return Ok(freed) };
                let reg = &dw.regs[r as usize];
                if !matches!(reg.state, DState::Queued) {
                    break;
                }
                c = self.inner.cv.wait(c).unwrap();
            }
            let reg = &c.ws(ws)?.devs[d].as_ref().unwrap().regs[r as usize];
            if reg.mem.is_empty() || reg.live > 0 {
                continue;
            }
            freed += self.inner.unmap_region(&mut c, ws, d, r)?;
        }
        Ok(freed)
    }

    // ------------------------------------------------------------ access

    /// A stream-scoped lease on region `r` in place; an unmapped region is a miss and is filled
    /// (`DeviceFull`/`Floor` when it cannot map). No host block once the fill is queued.
    pub fn acquire(&self, ws: WsId, ordinal: i32, r: u32, stream: u64) -> Result<Lease> {
        let d = self.dev(ordinal)?;
        self.inner.acquire_home(ws, d, r, stream)
    }

    // ------------------------------------------------------------ lifecycle

    /// Close a weight set: refuses while leases, views or cursors remain. Its device memory is
    /// released. A poisoned plane refuses: copies may still touch its ranges, so they stay mapped.
    pub fn close_ws(&self, ws: WsId) -> Result<()> {
        let mut c = self.lock();
        c.check()?;
        if c.cursors.values().any(|k| k.ws == ws) {
            return Err(Error::Invalid(format!("weight set {} has an open cursor", ws.0)));
        }
        loop {
            let w = c.ws(ws)?;
            let devs = w.devs.iter().flatten();
            if devs.clone().any(|dw| dw.views > 0 || dw.regs.iter().any(|r| r.live > 0)) {
                return Err(Error::Invalid(format!("weight set {} has live leases or views", ws.0)));
            }
            if !devs.clone().any(|dw| dw.regs.iter().any(|r| matches!(r.state, DState::Queued))) {
                break;
            }
            c = self.inner.cv.wait(c).unwrap();
        }
        for d in 0..self.inner.devices.len() {
            if c.ws(ws)?.devs[d].is_none() {
                continue;
            }
            let _g = self.inner.devices[d].bind()?;
            for r in 0..c.ws(ws)?.layout.regions.len() as u32 {
                if !c.ws(ws)?.devs[d].as_ref().unwrap().regs[r as usize].mem.is_empty() {
                    self.inner.unmap_region(&mut c, ws, d, r)?;
                }
            }
            let dw = c.ws_mut(ws)?.devs[d].take().unwrap();
            device::free_va(dw.base, c.ws(ws)?.layout.nbytes)?;
        }
        if let Some(host) = &c.ws(ws)?.sealed {
            let _g = self.inner.devices.first().map(DeviceRt::bind).transpose()?;
            host.unpin_all()?;
        }
        c.sets[ws.0 as usize] = None;
        Ok(())
    }

    pub fn close(&self) -> Result<()> {
        let ids: Vec<WsId> = {
            let c = self.lock();
            (0..c.sets.len() as u32).filter(|&i| c.sets[i as usize].is_some()).map(WsId).collect()
        };
        for id in ids {
            self.close_ws(id)?;
        }
        for dev in &self.inner.devices {
            for m in &dev.movers {
                m.stop();
            }
        }
        self.inner.readers.shutdown();
        Ok(())
    }

    // ------------------------------------------------------------ facts

    pub fn events(&self) -> Vec<Event> {
        self.lock().events.drain(..).collect()
    }

    pub fn stats(&self) -> Stats {
        let mut c = self.lock();
        resolve_timings(&mut c);
        let t = &self.inner.tally;
        let mut devices: Vec<DevStats> = self
            .inner
            .devices
            .iter()
            .enumerate()
            .map(|(d, dev)| DevStats {
                ordinal: dev.ordinal,
                uuid: dev.uuid.clone(),
                granularity: dev.granularity,
                mapped: c.mapped[d],
                counters: c.dstats[d].clone(),
                ..Default::default()
            })
            .collect();
        let mut regions = Vec::new();
        for (wi, w) in c.sets.iter_mut().enumerate() {
            let Some(w) = w.as_mut() else { continue };
            for (d, dw) in w.devs.iter_mut().enumerate() {
                let Some(dw) = dw.as_mut() else { continue };
                for (r, reg) in dw.regs.iter_mut().enumerate() {
                    if reg.mem.is_empty() {
                        continue;
                    }
                    let _ = commit(reg);
                    let span = w.layout.regions[r].span;
                    if reg.live > 0 {
                        devices[d].leased += span;
                    }
                    let state = match reg.state {
                        DState::Absent | DState::Queued | DState::Filling(_) => "filling",
                        DState::Ready => "ready",
                        DState::Invalid(_) => "invalid",
                    };
                    regions.push(RegionStats {
                        ws: wi as u32,
                        device: self.inner.devices[d].ordinal,
                        region: r as u32,
                        bytes: span,
                        state,
                        live: reg.live,
                        last_use: reg.last_use,
                    });
                }
            }
        }
        let io = IoStats {
            direct_bytes: t.direct_bytes.load(Ordering::Relaxed),
            cached_bytes: t.cached_bytes.load(Ordering::Relaxed),
            buffered_bytes: t.buffered_bytes.load(Ordering::Relaxed),
            direct_refused: t.direct_refused.load(Ordering::Relaxed),
            register_failed: c.register_failed,
        };
        Stats { devices, regions, io, now: self.inner.now(), events_dropped: c.events_dropped, poisoned: c.poisoned.clone() }
    }
}

impl Drop for Plane {
    fn drop(&mut self) {
        // Movers and readers hold `Inner`; stop them when the last user handle goes.
        if Arc::strong_count(&self.inner) == 1 {
            let _ = self.close();
        }
    }
}

impl Drop for Inner {
    /// The last reference can go without `close()`: handles collected in any order, a plane
    /// dropped with a fill queued, a poisoned plane. Its threads stop, and once every device has
    /// run all queued work its ranges are unmapped and its memory released, as `close()` would.
    /// A device that cannot be synchronized keeps its ranges for the life of the process.
    fn drop(&mut self) {
        self.readers.stop();
        for dev in &self.devices {
            for m in &dev.movers {
                m.detach();
            }
        }
        let c = self.core.get_mut().unwrap_or_else(|p| p.into_inner());
        if teardown(&self.devices, c).is_err() {
            for w in c.sets.iter().flatten() {
                std::mem::forget(w.sealed.clone());
            }
        }
    }
}

/// Free everything the books still hold, after the devices ran all queued work.
fn teardown(devices: &[DeviceRt], c: &mut Core) -> Result<()> {
    for dev in devices {
        let _g = dev.bind()?;
        // SAFETY: the device's context is bound.
        cuda::check("cuCtxSynchronize", unsafe { (cuda::driver()?.ctx_synchronize)() })?;
    }
    let mut rings: Vec<(usize, u64, u64, Vec<Chunk>)> = c.cursors.drain().map(|(_, k)| k.into_ring()).collect();
    for (d, dev) in devices.iter().enumerate() {
        let _g = dev.bind()?;
        let mut mem = Vec::new();
        for (_, base, len, held) in rings.extract_if(.., |r| r.0 == d && r.2 > 0) {
            device::unmap_span(base, len)?;
            device::free_va(base, len)?;
            mem.extend(held);
        }
        for w in c.sets.iter_mut().flatten() {
            let Some(dw) = w.devs[d].take() else { continue };
            for (reg, region) in dw.regs.into_iter().zip(&w.layout.regions) {
                if !reg.mem.is_empty() {
                    device::unmap_span(dw.base + region.offset, region.span)?;
                    mem.extend(reg.mem);
                }
            }
            device::free_va(dw.base, w.layout.nbytes)?;
        }
        for m in mem {
            device::release(m.handle)?;
        }
    }
    let _g = devices.first().map(DeviceRt::bind).transpose()?;
    for w in c.sets.iter().flatten() {
        if let Some(host) = &w.sealed {
            host.unpin_all()?;
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Default)]
pub struct DevStats {
    pub ordinal: i32,
    pub uuid: String,
    pub granularity: u64,
    /// Bytes mapped: regions and rings.
    pub mapped: u64,
    /// Bytes of regions a live lease holds.
    pub leased: u64,
    pub counters: DevCounters,
}

/// One mapped region on one device: what a caller choosing drops reads.
#[derive(Debug, Clone)]
pub struct RegionStats {
    pub ws: u32,
    pub device: i32,
    pub region: u32,
    /// The mapped span.
    pub bytes: u64,
    /// "filling", "ready" or "invalid".
    pub state: &'static str,
    pub live: u32,
    /// Plane-clock nanoseconds of the last map, acquire or release (`Stats::now` is the clock now).
    pub last_use: u64,
}

#[derive(Debug, Clone, Default)]
pub struct IoStats {
    pub direct_bytes: u64,
    pub cached_bytes: u64,
    pub buffered_bytes: u64,
    pub direct_refused: u64,
    /// Sealed regions this process could not page-lock (they bounce).
    pub register_failed: u64,
}

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub devices: Vec<DevStats>,
    pub regions: Vec<RegionStats>,
    pub io: IoStats,
    pub now: u64,
    pub events_dropped: u64,
    pub poisoned: Option<String>,
}

// ---------------------------------------------------------------- tickets, leases, views

/// Completion of a `want`. Reads the books; holds nothing.
pub struct Ticket {
    inner: Arc<Inner>,
    ws: WsId,
    dev: usize,
    regions: Vec<u32>,
}

impl Ticket {
    /// Ok(true) when every region is resident.
    pub fn done(&self) -> Result<bool> {
        self.check(false)
    }
    /// Host wait until done: the condvar while a copy is queued, then each copy's own event.
    pub fn wait(&self) -> Result<()> {
        self.check(true).map(|_| ())
    }

    fn check(&self, block: bool) -> Result<bool> {
        let mut c = self.inner.core.lock().unwrap();
        loop {
            let w = c.ws_mut(self.ws)?;
            let Some(dw) = w.devs[self.dev].as_mut() else {
                return Err(Error::Invalid(format!("weight set {} left the device", self.ws.0)));
            };
            let (mut busy, mut evs) = (false, Vec::new());
            for &r in &self.regions {
                let reg = &mut dw.regs[r as usize];
                commit(reg)?;
                match &reg.state {
                    DState::Ready => {}
                    DState::Queued => busy = true,
                    DState::Filling(ev) => evs.push(ev.clone()),
                    DState::Invalid(e) => return Err(e.clone()),
                    DState::Absent => return Err(Error::Invalid(format!("region {r} was dropped"))),
                }
            }
            if !block {
                return Ok(!busy && evs.is_empty());
            }
            if busy {
                c = self.inner.cv.wait(c).unwrap();
                continue;
            }
            drop(c);
            let _g = self.inner.devices[self.dev].bind()?;
            for e in &evs {
                e.sync()?;
            }
            return Ok(true);
        }
    }
}

pub(crate) enum LeaseKind {
    Home,
    Ring { cursor: u64, k: u64 },
}

/// Use of one region's bytes on one consumer stream: the only pin. `release(stream)` exactly
/// once; dropping an unreleased lease poisons the plane (its consumer may still run on those
/// bytes).
pub struct Lease {
    pub(crate) inner: Arc<Inner>,
    pub ptr: u64,
    pub nbytes: u64,
    pub region: u32,
    /// Where a streamed lease's bytes sit in its cursor's ring; None for a region in place.
    pub ring_offset: Option<u64>,
    pub(crate) ws: WsId,
    pub(crate) dev: usize,
    pub(crate) kind: LeaseKind,
    pub(crate) released: AtomicBool,
}

impl Lease {
    pub fn device(&self) -> i32 {
        self.inner.devices[self.dev].ordinal
    }

    /// Record the release event on `stream`: the bytes may be unmapped or overwritten once the
    /// device passes it. Fails only on a second release: a driver failure never leaves the lease
    /// counted. Without an event the stream is synchronized instead (`release_synced`); when that
    /// fails too the bytes are never reused (`release_unconfirmed`): a region in place turns
    /// Invalid (refilled on its next acquire), a ring slot stays held until the cursor closes.
    pub fn release(&self, stream: u64) -> Result<()> {
        if self.released.swap(true, Ordering::SeqCst) {
            return Err(Error::Invalid("lease released twice".into()));
        }
        let rt = &self.inner.devices[self.dev];
        let s = stream as CUstream;
        let g = rt.bind();
        let recorded = g.as_ref().map_err(Error::clone).and_then(|_| {
            crate::faults::hit("release-event")?;
            let ev = self.inner.events[self.dev].get(false)?;
            ev.record(s)?;
            Ok(ev)
        });
        // As a ring slot's `freed`: Some(Some(ev)) after ev, Some(None) at once, None never.
        let (freed, report) = match recorded {
            Ok(ev) => (Some(Some(ev)), None),
            Err(e) => {
                let synced = crate::faults::hit("release-sync").and_then(|_| {
                    let d = cuda::driver()?;
                    // SAFETY: the consumer's stream, in its bound context.
                    cuda::check("cuStreamSynchronize", unsafe { (d.stream_synchronize)(s) })
                });
                match synced {
                    Ok(()) => (Some(None), Some(("release_synced", e.to_string()))),
                    Err(e2) => (None, Some(("release_unconfirmed", format!("{e}; {e2}")))),
                }
            }
        };
        drop(g);
        let now = self.inner.now();
        let mut c = self.inner.core.lock().unwrap();
        let mut events = Vec::from_iter(report);
        match self.kind {
            LeaseKind::Home => {
                // Always there: a live lease keeps its weight set open and its region mapped.
                if let Some(dw) = c.ws_mut(self.ws).ok().and_then(|w| w.devs[self.dev].as_mut()) {
                    let reg = &mut dw.regs[self.region as usize];
                    reg.live -= 1;
                    reg.last_use = now;
                    match freed {
                        Some(Some(ev)) => {
                            prune(&mut reg.pending);
                            reg.pending.push(ev);
                        }
                        Some(None) => {}
                        None => reg.state = DState::Invalid(Error::Invalid("a release the device could not confirm".into())),
                    }
                }
            }
            LeaseKind::Ring { cursor, k } => {
                if let Err(e) = crate::cursor::release_ring(&self.inner, &mut c, cursor, k, freed) {
                    events.push(("prefetch_failed", e.to_string()));
                }
            }
        }
        for (kind, detail) in events {
            c.event(kind, self.ws, self.region, rt.ordinal, self.nbytes, detail);
        }
        Ok(())
    }

    /// A zero-copy view of this lease's bytes. It keeps the weight set's VA (or the cursor's
    /// ring) alive, not the bytes: they are valid only while a lease covers them.
    pub fn view(&self) -> Result<LeaseView> {
        let hold = match self.kind {
            LeaseKind::Home => Hold::Ws(self.inner.view(self.ws, self.dev)?),
            LeaseKind::Ring { cursor, .. } => Hold::Ring(crate::cursor::ring_view(&self.inner, cursor)?),
        };
        Ok(LeaseView { ptr: self.ptr, nbytes: self.nbytes, device: self.device(), hold })
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        if !self.released.load(Ordering::SeqCst) {
            let cause = format!("lease on region {} dropped without release(stream)", self.region);
            self.inner.core.lock().unwrap().poison(cause);
        }
    }
}

pub enum Hold {
    Ws(View),
    Ring(crate::cursor::RingView),
}

pub struct LeaseView {
    pub ptr: u64,
    pub nbytes: u64,
    pub device: i32,
    hold: Hold,
}

impl LeaseView {
    /// A region's view is stale once any region of its weight set was unmapped; a ring view
    /// never is (the ring stays mapped for the cursor's life).
    pub fn stale(&self) -> bool {
        match &self.hold {
            Hold::Ws(v) => v.stale(),
            Hold::Ring(_) => false,
        }
    }
}

/// A weight set's device VA, for zero-copy views of resident regions.
pub struct View {
    inner: Arc<Inner>,
    ws: WsId,
    dev: usize,
    pub ptr: u64,
    pub nbytes: u64,
    epoch: u64,
}

impl View {
    /// Whether any region was unmapped since this view was taken.
    pub fn stale(&self) -> bool {
        let c = self.inner.core.lock().unwrap();
        c.ws(self.ws).ok().and_then(|w| w.devs[self.dev].as_ref()).is_none_or(|dw| dw.epoch != self.epoch)
    }
}

impl Drop for View {
    fn drop(&mut self) {
        let mut c = self.inner.core.lock().unwrap();
        if let Some(dw) = c.ws_mut(self.ws).ok().and_then(|w| w.devs[self.dev].as_mut()) {
            dw.views -= 1;
        }
    }
}

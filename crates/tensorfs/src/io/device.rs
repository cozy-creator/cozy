//! B7's bounded pinned staging and reuse policy, independent of varena's placement types.

use std::collections::{BTreeMap, VecDeque};
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Instant;

use super::cuda::{self, CUcontext, CUevent, CUstream, CtxGuard};
use super::{host, Copy, Dst, Error, ReadOpts, ReadStats, Src};

const STREAMS: usize = 4;

// LAB ONLY: measurement knobs, never merged.
struct Lab {
    fifo: bool,
    readers: usize,
    staging: usize,
    piece: usize,
    streams: usize,
    check: u8,
    register: usize,
    depth: usize,
    populate: bool,
    trace: bool,
    own: usize,
    affinity: bool,
}

fn local_cpus(device: i32) -> Option<libc::cpu_set_t> {
    let d = cuda::driver().ok()?;
    let mut dev = 0;
    cuda::check("cuDeviceGet", unsafe { (d.device_get)(&mut dev, device) }).ok()?;
    let a = |x| cuda::attribute(dev, x).ok();
    let pci = format!("{:04x}:{:02x}:{:02x}.0", a(cuda::ATTR_PCI_DOMAIN_ID)?, a(cuda::ATTR_PCI_BUS_ID)?, a(cuda::ATTR_PCI_DEVICE_ID)?);
    let list = std::fs::read_to_string(format!("/sys/bus/pci/devices/{pci}/local_cpulist")).ok()?;
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for part in list.trim().split(',') {
        let (lo, hi) = part.split_once('-').unwrap_or((part, part));
        for cpu in lo.parse::<usize>().ok()?..=hi.parse::<usize>().ok()? {
            unsafe { libc::CPU_SET(cpu, &mut set) };
        }
    }
    Some(set)
}

fn pin_here(device: i32) {
    if let Some(set) = local_cpus(device) {
        unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) };
    }
}

fn lab() -> &'static Lab {
    static LAB: OnceLock<Lab> = OnceLock::new();
    LAB.get_or_init(|| {
        let n = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let s = |k: &str| std::env::var(k).unwrap_or_default();
        Lab {
            fifo: s("TFS_LAB_POOL") == "fifo",
            readers: n("TFS_LAB_READERS", 8),
            staging: n("TFS_LAB_STAGING", 16),
            piece: n("TFS_LAB_PIECE", 2 << 20),
            streams: n("TFS_LAB_STREAMS", 0),
            check: match s("TFS_LAB_CHECK").as_str() { "off" => 1, "copy" => 2, "lazy" => 3, _ => 0 },
            register: n("TFS_LAB_REGISTER", 0),
            depth: n("TFS_LAB_DEPTH", 2),
            populate: s("TFS_LAB_POPULATE") == "1",
            trace: s("TFS_LAB_TRACE") == "1",
            own: n("TFS_LAB_OWN", 0),
            affinity: s("TFS_LAB_AFFINITY") == "1",
        }
    })
}

#[derive(Default)]
struct LabTally {
    wait_ns: AtomicU64,
    fill_ns: AtomicU64,
    queue_ns: AtomicU64,
    check_ns: AtomicU64,
    map_ns: AtomicU64,
    reg_ns: AtomicU64,
    unreg_ns: AtomicU64,
    registered: AtomicU64,
}

struct Reg {
    done: Event,
    map: *mut libc::c_void,
    len: usize,
}

fn unreg(r: Reg, t: &LabTally) -> cuda::Result<()> {
    r.done.sync()?;
    let s = Instant::now();
    let rc = unsafe { (cuda::driver()?.mem_host_unregister)(r.map) };
    unsafe { libc::munmap(r.map, r.len) };
    t.unreg_ns.fetch_add(s.elapsed().as_nanos() as u64, Relaxed);
    cuda::check("cuMemHostUnregister", rc)
}

fn failed(copy: usize, error: cuda::Error) -> Error {
    let kind = match error {
        cuda::Error::Cuda {
            code: cuda::ERROR_OUT_OF_MEMORY,
            ..
        } => io::ErrorKind::OutOfMemory,
        cuda::Error::CudaUnavailable(_) => io::ErrorKind::Unsupported,
        _ => io::ErrorKind::Other,
    };
    Error {
        copy,
        source: io::Error::new(kind, error),
    }
}

struct Event {
    ctx: CUcontext,
    raw: CUevent,
}

impl Event {
    fn new(ctx: CUcontext, timing: bool) -> cuda::Result<Event> {
        let _ctx = CtxGuard::enter(ctx)?;
        let mut raw = std::ptr::null_mut();
        cuda::check("cuEventCreate", unsafe {
            (cuda::driver()?.event_create)(
                &mut raw,
                if timing {
                    cuda::EVENT_DEFAULT
                } else {
                    cuda::EVENT_DISABLE_TIMING
                },
            )
        })?;
        Ok(Event { ctx, raw })
    }
    fn record(&self, stream: CUstream) -> cuda::Result<()> {
        let _ctx = CtxGuard::enter(self.ctx)?;
        cuda::check("cuEventRecord", unsafe {
            (cuda::driver()?.event_record)(self.raw, stream)
        })
    }
    fn sync(&self) -> cuda::Result<()> {
        let _ctx = CtxGuard::enter(self.ctx)?;
        cuda::check("cuEventSynchronize", unsafe {
            (cuda::driver()?.event_synchronize)(self.raw)
        })
    }
    fn since(&self, start: &Event) -> cuda::Result<u64> {
        let mut ms = 0.0f32;
        cuda::check("cuEventElapsedTime", unsafe {
            (cuda::driver()?.event_elapsed_time)(&mut ms, start.raw, self.raw)
        })?;
        Ok((ms as f64 * 1e6) as u64)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if let (Ok(d), Ok(_ctx)) = (cuda::driver(), CtxGuard::enter(self.ctx)) {
            unsafe { (d.event_destroy)(self.raw) };
        }
    }
}

struct Slot {
    ptr: *mut u8,
    done: Event,
    pending: bool,
    recorded: bool,
}

impl Slot {
    fn new(ctx: CUcontext) -> cuda::Result<Slot> {
        let done = Event::new(ctx, false)?;
        let mut ptr = std::ptr::null_mut();
        cuda::check("cuMemHostAlloc", unsafe {
            (cuda::driver()?.mem_host_alloc)(&mut ptr, lab().piece, cuda::MEMHOSTALLOC_PORTABLE)
        })?;
        Ok(Slot {
            ptr: ptr.cast(),
            done,
            pending: false,
            recorded: false,
        })
    }
}

unsafe impl Send for Slot {}

impl Drop for Slot {
    fn drop(&mut self) {
        if let (Ok(d), Ok(_ctx)) = (cuda::driver(), CtxGuard::enter(self.done.ctx)) {
            if self.pending && (!self.recorded || self.done.sync().is_err()) {
                // The driver may still touch this allocation. Retain it until process exit
                // rather than free an unproved in-flight stage during unwinding.
                return;
            }
            unsafe { (d.mem_free_host)(self.ptr.cast()) };
        }
    }
}

struct Pool {
    free: Mutex<VecDeque<Slot>>,
    ready: Condvar,
}

impl Pool {
    fn take(&self) -> Slot {
        let mut free = self
            .ready
            .wait_while(self.free.lock().unwrap(), |free| free.is_empty())
            .unwrap();
        if lab().fifo { free.pop_front() } else { free.pop_back() }.unwrap()
    }
    fn put(&self, slot: Slot) {
        self.free.lock().unwrap().push_back(slot);
        self.ready.notify_one();
    }
}

struct Engine {
    device: i32,
    ctx: CUcontext,
    streams: Vec<cuda::Stream>,
    queues: Vec<Mutex<()>>,
    pool: Pool,
    serial: Mutex<()>,
    seen: Mutex<VecDeque<(usize, usize)>>,
    held: Mutex<Vec<Retained>>,
    staged: Mutex<Vec<Slot>>,
}

/// An engine-owned CUDA allocation. DLPack holders share it; the last owner frees it in its
/// primary context, independently of Python checkpoint and component handles.
pub struct DeviceBuffer {
    ptr: u64,
    len: usize,
    engine: Arc<Engine>,
}

impl DeviceBuffer {
    pub fn new(device: i32, len: usize) -> cuda::Result<DeviceBuffer> {
        let engine = Engine::get(device)?;
        let _ctx = CtxGuard::enter(engine.ctx)?;
        let mut ptr = 0;
        if len != 0 {
            cuda::check("cuMemAlloc", unsafe {
                (cuda::driver()?.mem_alloc)(&mut ptr, len)
            })?;
        }
        Ok(DeviceBuffer { ptr, len, engine })
    }
    pub fn ptr(&self) -> u64 {
        self.ptr
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn device(&self) -> i32 {
        self.engine.device
    }
    pub fn context(&self) -> CUcontext {
        self.engine.ctx
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            if let (Ok(d), Ok(_ctx)) = (cuda::driver(), CtxGuard::enter(self.engine.ctx)) {
                unsafe { (d.mem_free)(self.ptr) };
            }
        }
    }
}

#[derive(Default)]
struct Retained {
    slots: Vec<Slot>,
    pins: Vec<cuda::HostPin>,
    events: Vec<Event>,
    streams: Vec<cuda::Stream>,
    _buffers: Vec<DeviceBuffer>,
}

// A failed check leaves the actual owners in place, not just their pointer identities.
fn checked_take<T, E>(
    owners: &mut Vec<T>,
    check: impl FnOnce(&mut [T]) -> Result<(), E>,
) -> Result<Vec<T>, E> {
    check(owners)?;
    Ok(std::mem::take(owners))
}

struct RunHold<'a> {
    engine: &'a Engine,
    owners: Option<Retained>,
}

impl RunHold<'_> {
    fn take(&mut self) -> Retained {
        let mut owners = self.owners.take().expect("run owners taken once");
        owners
            .slots
            .append(&mut self.engine.staged.lock().unwrap_or_else(|e| e.into_inner()));
        owners
    }

    fn finish(mut self, complete: bool) -> cuda::Result<()> {
        let owners = self.take();
        self.engine.release(vec![owners], complete)
    }
}

impl Drop for RunHold<'_> {
    fn drop(&mut self) {
        if self.owners.is_some() {
            let owners = self.take();
            self.engine
                .held
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(owners);
        }
    }
}

fn registry() -> &'static Mutex<BTreeMap<i32, Arc<Engine>>> {
    static ENGINES: OnceLock<Mutex<BTreeMap<i32, Arc<Engine>>>> = OnceLock::new();
    ENGINES.get_or_init(Mutex::default)
}

/// Noninitializing: no driver/context is created when this device has no engine.
pub(super) fn drain(device: i32) -> cuda::Result<()> {
    let engine = registry()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&device)
        .cloned();
    match engine {
        Some(engine) => {
            let _serial = engine.serial.lock().unwrap_or_else(|e| e.into_inner());
            engine.recover()
        }
        None => Ok(()),
    }
}

#[cfg(feature = "store")]
pub(super) fn retain_decode(stream: Option<cuda::Stream>, buffers: Vec<DeviceBuffer>) {
    let engine = buffers
        .first()
        .expect("decode retains its output")
        .engine
        .clone();
    engine
        .held
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(Retained {
            streams: stream.into_iter().collect(),
            _buffers: buffers,
            ..Retained::default()
        });
}

// Handles are used only after entering their primary context; calls on a stream are ordered.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Engine {
    fn get(device: i32) -> cuda::Result<Arc<Engine>> {
        let mut engines = registry().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(engine) = engines.get(&device) {
            return Ok(engine.clone());
        }
        let streams: Vec<_> = (0..STREAMS)
            .map(|_| cuda::Stream::new(device))
            .collect::<cuda::Result<_>>()?;
        let ctx = streams[0].context();
        let _ctx = CtxGuard::enter(ctx)?;
        let staging = if lab().own > 0 { lab().own * lab().readers } else { lab().staging };
        let free = if lab().affinity {
            let ctx_addr = ctx as usize;
            std::thread::scope(|s| {
                s.spawn(|| {
                    pin_here(device);
                    let ctx = ctx_addr as CUcontext;
                    let _ctx = CtxGuard::enter(ctx)?;
                    (0..staging).map(|_| Slot::new(ctx)).collect::<cuda::Result<VecDeque<_>>>()
                })
                .join()
                .unwrap()
            })?
        } else {
            (0..staging).map(|_| Slot::new(ctx)).collect::<cuda::Result<_>>()?
        };
        let engine = Arc::new(Engine {
            device,
            ctx,
            streams,
            queues: (0..STREAMS).map(|_| Mutex::new(())).collect(),
            pool: Pool {
                free: Mutex::new(free),
                ready: Condvar::new(),
            },
            serial: Mutex::new(()),
            seen: Mutex::new(VecDeque::new()),
            held: Mutex::new(Vec::new()),
            staged: Mutex::new(Vec::new()),
        });
        engines.insert(device, engine.clone());
        Ok(engine)
    }

    fn reused(&self, ptr: usize, len: usize) -> bool {
        let mut seen = self.seen.lock().unwrap();
        if seen.contains(&(ptr, len)) {
            return true;
        }
        if seen.len() >= 256 {
            seen.pop_front();
        }
        seen.push_back((ptr, len));
        false
    }

    fn wait(&self, event: &Event, streams: usize) -> cuda::Result<()> {
        let _ctx = CtxGuard::enter(self.ctx)?;
        for stream in &self.streams[..streams] {
            cuda::check("cuStreamWaitEvent", unsafe {
                (cuda::driver()?.stream_wait_event)(stream.raw() as CUstream, event.raw, 0)
            })?;
        }
        Ok(())
    }

    fn synchronize(&self, owners: &[Retained]) -> cuda::Result<()> {
        let mut first = None;
        for stream in self
            .streams
            .iter()
            .chain(owners.iter().flat_map(|r| &r.streams))
        {
            if let Err(error) = stream.drain() {
                first.get_or_insert(error);
            }
        }
        first.map_or(Ok(()), Err)
    }

    fn checked_release(&self, owners: &mut [Retained], complete: bool) -> cuda::Result<()> {
        if !complete {
            self.synchronize(owners)?;
        }
        let mut first = None;
        for pin in owners.iter_mut().flat_map(|r| &mut r.pins) {
            if let Err(error) = pin.release() {
                first.get_or_insert(error);
            }
        }
        first.map_or(Ok(()), Err)
    }

    fn released(&self, owners: Vec<Retained>) {
        for mut owner in owners {
            for mut slot in owner.slots.drain(..) {
                slot.pending = false;
                slot.recorded = false;
                self.pool.put(slot);
            }
            // The remaining event/stream/allocation owners drop only after checked completion.
            drop(owner);
        }
    }

    fn release(&self, mut owners: Vec<Retained>, complete: bool) -> cuda::Result<()> {
        match checked_take(&mut owners, |r| self.checked_release(r, complete)) {
            Ok(released) => {
                self.released(released);
                Ok(())
            }
            Err(error) => {
                self.held
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .extend(owners);
                Err(error)
            }
        }
    }

    // Caller holds serial, including before allowing any slot to be reused.
    fn recover(&self) -> cuda::Result<()> {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        let released = checked_take(&mut held, |r| self.checked_release(r, false))?;
        drop(held);
        // All previous engine work is now complete, including slots already returned to pool.
        for slot in self
            .pool
            .free
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter_mut()
        {
            slot.pending = false;
            slot.recorded = false;
        }
        self.released(released);
        Ok(())
    }

    fn hold_slot(&self, slot: Slot) {
        self.staged
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(slot);
    }

    fn run(
        &self,
        copies: &[(usize, Copy)],
        opts: &ReadOpts,
        staged_host: bool,
    ) -> Result<ReadStats, Error> {
        let _serial = self.serial.lock().unwrap_or_else(|e| e.into_inner());
        let first = copies[0].0;
        if !self
            .held
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
        {
            self.recover().map_err(|e| failed(first, e))?;
        }
        let _ctx = CtxGuard::enter(self.ctx).map_err(|e| failed(first, e))?;
        let streams = if lab().streams > 0 { lab().streams } else { opts.streams }.clamp(1, STREAMS);
        let lt = LabTally::default();
        let lab_start = Instant::now();
        let mut hold = RunHold {
            engine: self,
            owners: Some(Retained::default()),
        };
        let mut complete = false;
        let result = (|| {
            let owners = hold.owners.as_mut().expect("run owners live");
            let mut jobs = Vec::new();
            let tally = Tally::default();
            for (jobs_ci, &(copy, c)) in copies.iter().enumerate() {
                if let Src::Device { device, stream, .. } = c.src {
                    let source = Engine::get(device).map_err(|e| failed(copy, e))?;
                    owners
                        .events
                        .push(Event::new(source.ctx, false).map_err(|e| failed(copy, e))?);
                    let event = owners.events.last().expect("source event inserted");
                    event
                        .record(stream as CUstream)
                        .map_err(|e| failed(copy, e))?;
                    self.wait(event, streams).map_err(|e| failed(copy, e))?;
                }
                if let Dst::Device { stream, .. } = c.dst {
                    if stream != 0 {
                        owners
                            .events
                            .push(Event::new(self.ctx, false).map_err(|e| failed(copy, e))?);
                        let event = owners.events.last().expect("destination event inserted");
                        event
                            .record(stream as CUstream)
                            .map_err(|e| failed(copy, e))?;
                        self.wait(event, streams).map_err(|e| failed(copy, e))?;
                    }
                }
                let mut direct = matches!(
                    (c.src, c.dst),
                    (Src::Device { .. } | Src::Zero, Dst::Device { .. })
                );
                if !staged_host {
                    if let (Src::Host(ptr), Dst::Device { .. }) = (c.src, c.dst) {
                        let reused = self.reused(ptr as usize, c.dst.len());
                        if unsafe { cuda::registered(ptr) }.map_err(|e| failed(copy, e))? {
                            direct = true;
                        } else if reused {
                            let start = Instant::now();
                            let pin =
                                unsafe { cuda::pin_region(ptr.cast_mut(), c.dst.len(), true) };
                            tally
                                .pin_ns
                                .fetch_add(start.elapsed().as_nanos() as u64, Relaxed);
                            if let Ok(pin) = pin {
                                owners.pins.push(pin);
                                direct = true;
                            }
                        }
                    }
                }
                let mut io_direct = opts.direct && lab().check != 1;
                let mut register = false;
                if let (Src::File(file, offset), Dst::Device { .. }, false) = (c.src, c.dst, direct) {
                    if lab().check == 2 || lab().register > 0 {
                        let s = Instant::now();
                        let warm = platform::durable::cached(file, offset, c.dst.len() as u64).unwrap_or(false);
                        lt.check_ns.fetch_add(s.elapsed().as_nanos() as u64, Relaxed);
                        if lab().check == 2 && warm {
                            io_direct = false;
                        }
                        register = warm && lab().register > 0;
                    }
                }
                let step = if direct { c.dst.len() } else if register { lab().register } else { lab().piece };
                for at in (0..c.dst.len()).step_by(step.max(1)) {
                    jobs.push(Job {
                        copy,
                        ci: jobs_ci,
                        c,
                        at,
                        len: step.min(c.dst.len() - at),
                        direct,
                        io_direct,
                        register,
                    });
                }
            }
            let start = owners.events.len();
            for stream in &self.streams[..streams] {
                owners
                    .events
                    .push(Event::new(self.ctx, true).map_err(|e| failed(first, e))?);
                owners
                    .events
                    .last()
                    .expect("start event inserted")
                    .record(stream.raw() as CUstream)
                    .map_err(|e| failed(first, e))?;
            }
            let next = AtomicUsize::new(0);
            let stopped = AtomicBool::new(false);
            let error = Mutex::new(None);
            let warm: Vec<OnceLock<bool>> = copies.iter().map(|_| OnceLock::new()).collect();
            let own = lab().own;
            let work = || {
                if lab().affinity {
                    pin_here(self.device);
                }
                let _ctx = CtxGuard::enter(self.ctx);
                let mut out = VecDeque::new();
                let mut mine: Vec<Slot> = (0..own).map(|_| self.pool.take()).collect();
                let mut n = 0;
                let mut bad = false;
                while !stopped.load(Relaxed) {
                    let at = next.fetch_add(1, Relaxed);
                    let Some(job) = jobs.get(at) else { break };
                    let r = if own > 0 && !job.direct && !job.register {
                        n += 1;
                        self.stage(job, at % streams, opts, &tally, &lt, &warm, &mut mine[n % own])
                    } else {
                        self.piece(job, at % streams, opts, &tally, &lt, &mut out, &warm)
                    };
                    if let Err(failure) = r {
                        bad = true;
                        error.lock().unwrap().get_or_insert(failure);
                        stopped.store(true, Relaxed);
                    }
                }
                for slot in mine {
                    if bad { self.hold_slot(slot) } else { self.pool.put(slot) }
                }
                for r in out {
                    if let Err(e) = unreg(r, &lt) {
                        error.lock().unwrap().get_or_insert(failed(first, e));
                    }
                }
            };
            std::thread::scope(|scope| {
                if own > 0 || lab().affinity {
                    for _ in 0..lab().readers.max(1) {
                        scope.spawn(work);
                    }
                } else {
                    for _ in 1..lab().readers.max(1).min(jobs.len().max(1)) {
                        scope.spawn(work);
                    }
                    work();
                }
            });
            let mut busy_ns = 0;
            let mut all_complete = true;
            for (lane, stream) in self.streams[..streams].iter().enumerate() {
                let drained = (|| {
                    owners.events.push(Event::new(self.ctx, true)?);
                    let end = owners.events.last().expect("end event inserted");
                    end.record(stream.raw() as CUstream)?;
                    end.sync()?;
                    end.since(&owners.events[start + lane])
                })();
                match drained {
                    Ok(ns) => busy_ns += ns,
                    Err(e) => {
                        all_complete = false;
                        error.lock().unwrap().get_or_insert(failed(first, e));
                    }
                }
            }
            complete = all_complete;
            if lab().trace {
                let g = |a: &AtomicU64| a.load(Relaxed) / 1000;
                eprintln!(
                    "lab run jobs={} bytes={} us={} wait={} fill={} queue={} check={} map={} reg={} unreg={} registered={} staged={} cached={} direct={}",
                    jobs.len(), copies.iter().map(|(_, c)| c.dst.len()).sum::<usize>(), lab_start.elapsed().as_micros(),
                    g(&lt.wait_ns), g(&lt.fill_ns), g(&lt.queue_ns), g(&lt.check_ns), g(&lt.map_ns), g(&lt.reg_ns), g(&lt.unreg_ns),
                    lt.registered.load(Relaxed), tally.staged.load(Relaxed), tally.cached.load(Relaxed), tally.direct.load(Relaxed),
                );
            }
            if let Some(error) = error.into_inner().unwrap() {
                return Err(error);
            }
            Ok(ReadStats {
                direct: tally.direct.load(Relaxed),
                cached: tally.cached.load(Relaxed),
                bounced: tally.bounced.load(Relaxed),
                pin_ns: tally.pin_ns.load(Relaxed),
                staged: tally.staged.load(Relaxed),
                busy_ns,
                ..ReadStats::default()
            })
        })();
        let released = hold.finish(complete).map_err(|e| failed(first, e));
        let stats = result?;
        released?;
        Ok(stats)
    }

    fn piece(
        &self,
        job: &Job,
        lane: usize,
        opts: &ReadOpts,
        tally: &Tally,
        lt: &LabTally,
        out: &mut VecDeque<Reg>,
        warm: &[OnceLock<bool>],
    ) -> Result<(), Error> {
        let _ctx = CtxGuard::enter(self.ctx).map_err(|e| failed(job.copy, e))?;
        let d = cuda::driver().map_err(|e| failed(job.copy, e))?;
        let stream = self.streams[lane].raw() as CUstream;
        if job.register {
            let (Src::File(file, offset), Dst::Device { ptr: dst, .. }) = (job.c.src, job.c.dst) else {
                unreachable!()
            };
            let at = offset + job.at as u64;
            let start = at - at % 4096;
            let pre = (at - start) as usize;
            let len = (pre + job.len).next_multiple_of(4096);
            let s = Instant::now();
            let flags = libc::MAP_SHARED | if lab().populate { libc::MAP_POPULATE } else { 0 };
            let map = unsafe {
                libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, flags, std::os::fd::AsRawFd::as_raw_fd(file), start as libc::off_t)
            };
            if map == libc::MAP_FAILED {
                return Err(Error { copy: job.copy, source: io::Error::last_os_error() });
            }
            lt.map_ns.fetch_add(s.elapsed().as_nanos() as u64, Relaxed);
            let s = Instant::now();
            let rc = unsafe {
                (d.mem_host_register)(map, len, cuda::MEMHOSTREGISTER_PORTABLE | cuda::MEMHOSTREGISTER_READ_ONLY)
            };
            lt.reg_ns.fetch_add(s.elapsed().as_nanos() as u64, Relaxed);
            if rc != cuda::SUCCESS {
                unsafe { libc::munmap(map, len) };
                return cuda::check("cuMemHostRegister", rc).map_err(|e| failed(job.copy, e));
            }
            let done = Event::new(self.ctx, false).map_err(|e| failed(job.copy, e))?;
            {
                let _queue = self.queues[lane].lock().unwrap();
                let rc = unsafe {
                    (d.memcpy_htod_async)(dst + job.at as u64, (map as *const u8).add(pre).cast(), job.len, stream)
                };
                cuda::check("cuMemcpyAsync", rc).map_err(|e| failed(job.copy, e))?;
                done.record(stream).map_err(|e| failed(job.copy, e))?;
            }
            lt.registered.fetch_add(job.len as u64, Relaxed);
            out.push_back(Reg { done, map, len });
            while out.len() > lab().depth {
                unreg(out.pop_front().unwrap(), lt).map_err(|e| failed(job.copy, e))?;
            }
            return Ok(());
        }
        if job.direct {
            let _queue = self.queues[lane].lock().unwrap();
            let Dst::Device { ptr: dst, .. } = job.c.dst else {
                unreachable!()
            };
            let result = match job.c.src {
                Src::Zero => unsafe { (d.memset_d8_async)(dst, 0, job.len, stream) },
                Src::Host(ptr) => unsafe {
                    (d.memcpy_htod_async)(dst, ptr.cast(), job.len, stream)
                },
                Src::Device { device, ptr, .. } if device == self.device => unsafe {
                    (d.memcpy_dtod_async)(dst, ptr, job.len, stream)
                },
                Src::Device { device, ptr, .. } => {
                    let source = Engine::get(device).map_err(|e| failed(job.copy, e))?;
                    unsafe {
                        (d.memcpy_peer_async)(dst, self.ctx, ptr, source.ctx, job.len, stream)
                    }
                }
                _ => unreachable!(),
            };
            return cuda::check("cuMemcpyAsync", result).map_err(|e| failed(job.copy, e));
        }
        let mut slot = self.pool.take();
        let result = self.stage(job, lane, opts, tally, lt, warm, &mut slot);
        if result.is_err() {
            self.hold_slot(slot);
        } else {
            self.pool.put(slot);
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn stage(
        &self,
        job: &Job,
        lane: usize,
        opts: &ReadOpts,
        tally: &Tally,
        lt: &LabTally,
        warm: &[OnceLock<bool>],
        slot: &mut Slot,
    ) -> Result<(), Error> {
        let d = cuda::driver().map_err(|e| failed(job.copy, e))?;
        let stream = self.streams[lane].raw() as CUstream;
        let result = (|| {
            if slot.pending {
                let s = Instant::now();
                slot.done.sync().map_err(|e| failed(job.copy, e))?;
                lt.wait_ns.fetch_add(s.elapsed().as_nanos() as u64, Relaxed);
                slot.pending = false;
            }
            let fill_start = Instant::now();
            match job.c.src {
                Src::File(file, offset) => {
                    let copy = Copy {
                        src: Src::File(file, offset + job.at as u64),
                        dst: Dst::Host {
                            ptr: slot.ptr,
                            len: job.len,
                        },
                    };
                    let mut io_direct = job.io_direct;
                    if lab().check == 3 && io_direct {
                        let w = *warm[job.ci].get_or_init(|| {
                            let s = Instant::now();
                            let w = platform::durable::cached(file, offset, job.c.dst.len() as u64).unwrap_or(false);
                            lt.check_ns.fetch_add(s.elapsed().as_nanos() as u64, Relaxed);
                            w
                        });
                        io_direct = !w;
                    }
                    let options = ReadOpts {
                        threads: 1,
                        direct: io_direct,
                        ..opts.clone()
                    };
                    let stats = unsafe { host::fill(&[copy], &options) }.map_err(|e| Error {
                        copy: job.copy,
                        source: e.source,
                    })?;
                    tally.direct.fetch_add(stats.direct, Relaxed);
                    tally.cached.fetch_add(stats.cached, Relaxed);
                    tally.bounced.fetch_add(stats.bounced, Relaxed);
                }
                Src::Host(ptr) => unsafe {
                    std::ptr::copy_nonoverlapping(ptr.add(job.at), slot.ptr, job.len)
                },
                Src::Device { .. } => {}
                Src::Zero => unreachable!("zero uses native device memset"),
            }
            lt.fill_ns.fetch_add(fill_start.elapsed().as_nanos() as u64, Relaxed);
            let queue_start = Instant::now();
            let _queue = self.queues[lane].lock().unwrap();
            let rc = match (job.c.src, job.c.dst) {
                (_, Dst::Device { ptr, .. }) => unsafe {
                    (d.memcpy_htod_async)(ptr + job.at as u64, slot.ptr.cast(), job.len, stream)
                },
                (Src::Device { ptr, .. }, Dst::Host { .. }) => unsafe {
                    (d.memcpy_dtoh_async)(slot.ptr.cast(), ptr + job.at as u64, job.len, stream)
                },
                _ => unreachable!(),
            };
            slot.pending = true;
            slot.recorded = false;
            cuda::check("cuMemcpyAsync", rc).map_err(|e| failed(job.copy, e))?;
            slot.done.record(stream).map_err(|e| failed(job.copy, e))?;
            slot.recorded = true;
            lt.queue_ns.fetch_add(queue_start.elapsed().as_nanos() as u64, Relaxed);
            if let Dst::Host { ptr, .. } = job.c.dst {
                slot.done.sync().map_err(|e| failed(job.copy, e))?;
                slot.pending = false;
                unsafe { std::ptr::copy_nonoverlapping(slot.ptr, ptr.add(job.at), job.len) };
            }
            tally.staged.fetch_add(job.len as u64, Relaxed);
            Ok(())
        })();
        result
    }
}

struct Job<'a> {
    copy: usize,
    ci: usize,
    c: Copy<'a>,
    at: usize,
    len: usize,
    direct: bool,
    io_direct: bool,
    register: bool,
}

#[derive(Default)]
struct Tally {
    direct: AtomicU64,
    cached: AtomicU64,
    bounced: AtomicU64,
    pin_ns: AtomicU64,
    staged: AtomicU64,
}

/// Same bounded B7 staging, with checked completion and retained native resources on error.
pub(super) unsafe fn fill(
    copies: &[(usize, Copy)],
    opts: &ReadOpts,
    staged_host: bool,
) -> Result<ReadStats, Error> {
    let mut groups: BTreeMap<(i32, &str), Vec<(usize, Copy)>> = BTreeMap::new();
    for &(index, copy) in copies {
        if copy.dst.is_empty() {
            continue;
        }
        let device = match (copy.src, copy.dst) {
            (_, Dst::Device { device, .. }) | (Src::Device { device, .. }, _) => device,
            _ => unreachable!(),
        };
        let kind = match (copy.src, copy.dst) {
            (Src::Zero, _) => "zero",
            (Src::Device { .. }, Dst::Host { .. }) => "d2h",
            (Src::Device { device: source, .. }, Dst::Device { device: target, .. })
                if source != target =>
            {
                "peer"
            }
            (Src::Device { .. }, Dst::Device { .. }) => "d2d",
            _ => "h2d",
        };
        groups
            .entry((device, kind))
            .or_default()
            .push((index, copy));
    }
    let mut total = ReadStats::default();
    for ((device, kind), copies) in groups {
        let engine = Engine::get(device).map_err(|e| failed(copies[0].0, e))?;
        let start = Instant::now();
        let stats = engine.run(&copies, opts, staged_host)?;
        #[cfg(unix)]
        {
            let ns = start.elapsed().as_nanos() as u64;
            let plain: Vec<Copy> = copies.iter().map(|&(_, c)| c).collect();
            super::disk_run(&plain, &stats, ns);
            let run = super::metrics::Copy {
                kind,
                gpu: device,
                bytes: copies.iter().map(|(_, c)| c.dst.len() as u64).sum(),
                ns,
                link: super::metrics::device_link(device)
                    .map_or_else(String::new, |link| link.to_string()),
                busy_ns: stats.busy_ns,
                pin_ns: stats.pin_ns,
                staged: stats.staged,
                cached: stats.cached,
                ..Default::default()
            };
            match kind {
                "h2d" => super::metrics::h2d_run(run),
                _ => super::metrics::record(run),
            }
        }
        #[cfg(not(unix))]
        let _ = (kind, start);
        total.direct += stats.direct;
        total.cached += stats.cached;
        total.bounced += stats.bounced;
        total.pin_ns += stats.pin_ns;
        total.staged += stats.staged;
        total.busy_ns += stats.busy_ns;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Owner(Arc<AtomicUsize>);
    impl Drop for Owner {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn failed_completion_checks_retain_actual_owners_until_a_checked_success() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut owners = vec![Owner(dropped.clone()), Owner(dropped.clone())];
        for _ in 0..2 {
            let result = checked_take(&mut owners, |_| Err::<(), _>("injected completion error"));
            assert!(result.is_err());
            assert_eq!(owners.len(), 2);
            assert_eq!(dropped.load(Ordering::SeqCst), 0);
        }
        let released = checked_take(&mut owners, |_| Ok::<(), &str>(())).unwrap();
        assert!(owners.is_empty());
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
        drop(released);
        assert_eq!(dropped.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn draining_a_device_without_an_engine_does_not_initialize_cuda() {
        // No CUDA discovery, context creation, allocation or driver call is needed.
        assert!(!registry().lock().unwrap().contains_key(&i32::MIN));
        drain(i32::MIN).unwrap();
        assert!(!registry().lock().unwrap().contains_key(&i32::MIN));
    }
}

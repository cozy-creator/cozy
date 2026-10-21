//! B7's bounded pinned staging and reuse policy, independent of varena's placement types.

use std::collections::{BTreeMap, VecDeque};
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use super::cuda::{self, CUcontext, CUevent, CUstream, CtxGuard};
use super::{host, Copy, Dst, Error, ReadOpts, ReadStats, Src};

const PIECE: usize = 2 << 20;
const READERS: usize = 8;
/// Slots each reader owns: one fills while the other's copy lands, and both stay in its caches.
/// A shared pool (LIFO or FIFO) measured 8-20% slower warm H2D.
const RING: usize = 2;
const STAGING: usize = READERS * RING;
const STREAMS: usize = 4;

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
            (cuda::driver()?.mem_host_alloc)(&mut ptr, PIECE, cuda::MEMHOSTALLOC_PORTABLE)
        })?;
        Ok(Slot {
            ptr: ptr.cast(),
            done,
            pending: false,
            recorded: false,
        })
    }
}

// SAFETY: pinned host memory and an event of the engine's context, used only with it entered.
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

/// The CPUs of the GPU's own NUMA node that this process may use. Readers run there and staging
/// is allocated there: on a two-socket EPYC, readers and slots left to the scheduler measured
/// 64-94% of the H2D ceiling from run to run, kept local 92-94%.
#[derive(Clone, Copy)]
struct Near(#[cfg(target_os = "linux")] Option<libc::cpu_set_t>);

impl Near {
    #[cfg(target_os = "linux")]
    fn of(device: i32) -> Near {
        let list = super::metrics::pci_address(device).and_then(|pci| {
            std::fs::read_to_string(format!("/sys/bus/pci/devices/{pci}/local_cpulist")).ok()
        });
        // SAFETY: plain C structs and calls on this thread's own mask.
        unsafe {
            let (mut allowed, mut near): (libc::cpu_set_t, libc::cpu_set_t) =
                (std::mem::zeroed(), std::mem::zeroed());
            if list.is_none()
                || libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut allowed)
                    != 0
            {
                return Near(None);
            }
            for range in list.as_deref().unwrap_or_default().trim().split(',') {
                let (lo, hi) = range.split_once('-').unwrap_or((range, range));
                let (Ok(lo), Ok(hi)) = (lo.parse::<usize>(), hi.parse::<usize>()) else {
                    return Near(None);
                };
                for cpu in (lo..=hi).filter(|&cpu| {
                    cpu < libc::CPU_SETSIZE as usize && libc::CPU_ISSET(cpu, &allowed)
                }) {
                    libc::CPU_SET(cpu, &mut near);
                }
            }
            Near((libc::CPU_COUNT(&near) > 0).then_some(near))
        }
    }
    #[cfg(not(target_os = "linux"))]
    fn of(_: i32) -> Near {
        Near()
    }
    /// Moves this thread there (TensorFS's own threads only).
    fn enter(&self) {
        #[cfg(target_os = "linux")]
        if let Some(set) = &self.0 {
            // SAFETY: a valid mask for the calling thread.
            unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), set) };
        }
    }
}

struct Engine {
    device: i32,
    ctx: CUcontext,
    streams: Vec<cuda::Stream>,
    queues: Vec<Mutex<()>>,
    near: Near,
    free: Mutex<Vec<Slot>>,
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
        let near = Near::of(device);
        // Pinned pages land on the allocating thread's node.
        let at = ctx as usize;
        let free = std::thread::scope(|s| {
            s.spawn(|| {
                near.enter();
                let ctx = at as CUcontext;
                let _ctx = CtxGuard::enter(ctx)?;
                (0..STAGING)
                    .map(|_| Slot::new(ctx))
                    .collect::<cuda::Result<Vec<_>>>()
            })
            .join()
            .expect("staging allocation does not panic")
        })?;
        let engine = Arc::new(Engine {
            device,
            ctx,
            streams,
            queues: (0..STREAMS).map(|_| Mutex::new(())).collect(),
            near,
            free: Mutex::new(free),
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
                self.free().push(slot);
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
        // All previous engine work is now complete, including slots already returned to the free list.
        for slot in self.free().iter_mut() {
            slot.pending = false;
            slot.recorded = false;
        }
        self.released(released);
        Ok(())
    }

    fn free(&self) -> std::sync::MutexGuard<'_, Vec<Slot>> {
        self.free.lock().unwrap_or_else(|e| e.into_inner())
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
        let streams = opts.streams.clamp(1, STREAMS);
        let mut hold = RunHold {
            engine: self,
            owners: Some(Retained::default()),
        };
        let mut complete = false;
        let result = (|| {
            let owners = hold.owners.as_mut().expect("run owners live");
            let mut jobs = Vec::new();
            let tally = Tally::default();
            for &(copy, c) in copies {
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
                let step = if direct { c.dst.len() } else { PIECE };
                for at in (0..c.dst.len()).step_by(step.max(1)) {
                    jobs.push(Job {
                        copy,
                        c,
                        at,
                        len: step.min(c.dst.len() - at),
                        direct,
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
            // A failed copy's slots stay retained until checked completion: read with the rest.
            let readers = opts.threads.clamp(1, READERS).min(jobs.len()).min(self.free().len() / RING);
            if readers == 0 && !jobs.is_empty() {
                let lost = io::Error::other("all staging is still retained by failed copies");
                return Err(Error {
                    copy: first,
                    source: lost,
                });
            }
            let work = || {
                self.near.enter();
                let mut ring = {
                    let mut free = self.free();
                    let at = free.len() - RING;
                    free.split_off(at)
                };
                let mut turn = 0;
                while !stopped.load(Relaxed) {
                    let at = next.fetch_add(1, Relaxed);
                    let Some(job) = jobs.get(at) else { break };
                    turn += 1;
                    if let Err(failure) =
                        self.piece(job, at % streams, opts, &tally, &mut ring[turn % RING])
                    {
                        error.lock().unwrap().get_or_insert(failure);
                        stopped.store(true, Relaxed);
                    }
                }
                // After any failure a slot may still be in flight: kept until checked completion.
                for slot in ring {
                    match stopped.load(Relaxed) {
                        true => self.hold_slot(slot),
                        false => self.free().push(slot),
                    }
                }
            };
            // Every reader is TensorFS's own thread, so moving it to the GPU's CPUs moves no caller.
            std::thread::scope(|scope| {
                for _ in 0..readers {
                    scope.spawn(work);
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
        slot: &mut Slot,
    ) -> Result<(), Error> {
        let _ctx = CtxGuard::enter(self.ctx).map_err(|e| failed(job.copy, e))?;
        let d = cuda::driver().map_err(|e| failed(job.copy, e))?;
        let stream = self.streams[lane].raw() as CUstream;
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
        if slot.pending {
            slot.done.sync().map_err(|e| failed(job.copy, e))?;
            slot.pending = false;
        }
        match job.c.src {
            Src::File(file, offset) => {
                let copy = Copy {
                    src: Src::File(file, offset + job.at as u64),
                    dst: Dst::Host {
                        ptr: slot.ptr,
                        len: job.len,
                    },
                };
                let options = ReadOpts {
                    threads: 1,
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
        if let Dst::Host { ptr, .. } = job.c.dst {
            slot.done.sync().map_err(|e| failed(job.copy, e))?;
            slot.pending = false;
            unsafe { std::ptr::copy_nonoverlapping(slot.ptr, ptr.add(job.at), job.len) };
        }
        tally.staged.fetch_add(job.len as u64, Relaxed);
        Ok(())
    }
}

struct Job<'a> {
    copy: usize,
    c: Copy<'a>,
    at: usize,
    len: usize,
    direct: bool,
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

//! B7's bounded pinned staging and reuse policy, independent of varena's placement types.

use std::collections::{BTreeMap, VecDeque};
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Instant;

use super::cuda::{self, CUcontext, CUevent, CUstream, CtxGuard};
use super::{host, Copy, Dst, Error, ReadOpts, ReadStats, Src};

const STAGING: usize = 16;
const PIECE: usize = 2 << 20;
const READERS: usize = 8;
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
        })
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        if let (Ok(d), Ok(_ctx)) = (cuda::driver(), CtxGuard::enter(self.done.ctx)) {
            if self.pending {
                let _ = self.done.sync();
            }
            unsafe { (d.mem_free_host)(self.ptr.cast()) };
        }
    }
}

struct Pool {
    free: Mutex<Vec<Slot>>,
    ready: Condvar,
}

impl Pool {
    fn take(&self) -> Slot {
        self.ready
            .wait_while(self.free.lock().unwrap(), |free| free.is_empty())
            .unwrap()
            .pop()
            .unwrap()
    }
    fn put(&self, slot: Slot) {
        self.free.lock().unwrap().push(slot);
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

// Handles are used only after entering their primary context; calls on a stream are ordered.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Engine {
    fn get(device: i32) -> cuda::Result<Arc<Engine>> {
        static ENGINES: OnceLock<Mutex<BTreeMap<i32, Arc<Engine>>>> = OnceLock::new();
        let mut engines = ENGINES.get_or_init(Mutex::default).lock().unwrap();
        if let Some(engine) = engines.get(&device) {
            return Ok(engine.clone());
        }
        let streams: Vec<_> = (0..STREAMS)
            .map(|_| cuda::Stream::new(device))
            .collect::<cuda::Result<_>>()?;
        let ctx = streams[0].context();
        let _ctx = CtxGuard::enter(ctx)?;
        let free = (0..STAGING)
            .map(|_| Slot::new(ctx))
            .collect::<cuda::Result<_>>()?;
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

    fn run(&self, copies: &[(usize, Copy)], opts: &ReadOpts) -> Result<ReadStats, Error> {
        let _serial = self.serial.lock().unwrap();
        let _ctx = CtxGuard::enter(self.ctx).map_err(|e| failed(copies[0].0, e))?;
        let streams = opts.streams.clamp(1, STREAMS);
        let first = copies[0].0;
        let mut ready = Vec::new();
        let mut pins = Vec::new();
        let mut jobs = Vec::new();
        let tally = Tally::default();
        for &(copy, c) in copies {
            if let Src::Device { device, stream, .. } = c.src {
                let source = Engine::get(device).map_err(|e| failed(copy, e))?;
                let event = Event::new(source.ctx, false).map_err(|e| failed(copy, e))?;
                event
                    .record(stream as CUstream)
                    .map_err(|e| failed(copy, e))?;
                self.wait(&event, streams).map_err(|e| failed(copy, e))?;
                ready.push(event);
            }
            if let Dst::Device { stream, .. } = c.dst {
                if stream != 0 {
                    let event = Event::new(self.ctx, false).map_err(|e| failed(copy, e))?;
                    event
                        .record(stream as CUstream)
                        .map_err(|e| failed(copy, e))?;
                    self.wait(&event, streams).map_err(|e| failed(copy, e))?;
                    ready.push(event);
                }
            }
            let mut direct = matches!((c.src, c.dst), (Src::Device { .. } | Src::Zero, Dst::Device { .. }));
            if let (Src::Host(ptr), Dst::Device { .. }) = (c.src, c.dst) {
                let reused = self.reused(ptr as usize, c.dst.len());
                if unsafe { cuda::registered(ptr) }.map_err(|e| failed(copy, e))? {
                    direct = true;
                } else if reused {
                    let start = Instant::now();
                    let pin = unsafe { cuda::pin_region(ptr.cast_mut(), c.dst.len(), true) };
                    tally
                        .pin_ns
                        .fetch_add(start.elapsed().as_nanos() as u64, Relaxed);
                    if let Ok(pin) = pin {
                        pins.push(pin);
                        direct = true;
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
        let start: Vec<_> = (0..streams)
            .map(|_| Event::new(self.ctx, true))
            .collect::<cuda::Result<_>>()
            .map_err(|e| failed(first, e))?;
        for (event, stream) in start.iter().zip(&self.streams) {
            event
                .record(stream.raw() as CUstream)
                .map_err(|e| failed(first, e))?;
        }
        let next = AtomicUsize::new(0);
        let stopped = AtomicBool::new(false);
        let error = Mutex::new(None);
        let work = || {
            while !stopped.load(Relaxed) {
                let at = next.fetch_add(1, Relaxed);
                let Some(job) = jobs.get(at) else { break };
                if let Err(failure) = self.piece(job, at % streams, opts, &tally) {
                    error.lock().unwrap().get_or_insert(failure);
                    stopped.store(true, Relaxed);
                }
            }
        };
        std::thread::scope(|scope| {
            for _ in 1..opts.threads.clamp(1, READERS).min(jobs.len().max(1)) {
                scope.spawn(work);
            }
            work();
        });
        let mut busy_ns = 0;
        for (begin, stream) in start.iter().zip(&self.streams) {
            let drained = (|| {
                let end = Event::new(self.ctx, true)?;
                end.record(stream.raw() as CUstream)?;
                end.sync()?;
                end.since(begin)
            })();
            match drained {
                Ok(ns) => busy_ns += ns,
                Err(e) => {
                    if let Ok(d) = cuda::driver() {
                        unsafe { (d.stream_synchronize)(stream.raw() as CUstream) };
                    }
                    error.lock().unwrap().get_or_insert(failed(first, e));
                }
            }
        }
        if let Some(error) = error.into_inner().unwrap() {
            return Err(error);
        }
        drop(ready);
        drop(pins);
        Ok(ReadStats {
            direct: tally.direct.load(Relaxed),
            cached: tally.cached.load(Relaxed),
            bounced: tally.bounced.load(Relaxed),
            pin_ns: tally.pin_ns.load(Relaxed),
            staged: tally.staged.load(Relaxed),
            busy_ns,
            ..ReadStats::default()
        })
    }

    fn piece(&self, job: &Job, lane: usize, opts: &ReadOpts, tally: &Tally) -> Result<(), Error> {
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
        let mut slot = self.pool.take();
        let result = (|| {
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
            cuda::check("cuMemcpyAsync", rc).map_err(|e| failed(job.copy, e))?;
            slot.done.record(stream).map_err(|e| failed(job.copy, e))?;
            slot.pending = true;
            if let Dst::Host { ptr, .. } = job.c.dst {
                slot.done.sync().map_err(|e| failed(job.copy, e))?;
                slot.pending = false;
                unsafe { std::ptr::copy_nonoverlapping(slot.ptr, ptr.add(job.at), job.len) };
            }
            tally.staged.fetch_add(job.len as u64, Relaxed);
            Ok(())
        })();
        if result.is_err() {
            unsafe { (d.stream_synchronize)(stream) };
        }
        self.pool.put(slot);
        result
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

/// Each device has B7's reusable pinned buffers; reads finish before any borrowed source is
/// released. A borrowed/owned registration is never retained past its caller's lifetime.
pub(super) unsafe fn fill(copies: &[(usize, Copy)], opts: &ReadOpts) -> Result<ReadStats, Error> {
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
            (Src::Device { device: source, .. }, Dst::Device { device: target, .. }) if source != target => "peer",
            (Src::Device { .. }, Dst::Device { .. }) => "d2d",
            _ => "h2d",
        };
        groups.entry((device, kind)).or_default().push((index, copy));
    }
    let mut total = ReadStats::default();
    for ((device, kind), copies) in groups {
        let engine = Engine::get(device).map_err(|e| failed(copies[0].0, e))?;
        #[cfg(unix)]
        let ceiling = if kind == "h2d" { super::metrics::h2d(device).map_or(0.0, |c| c.0) } else { 0.0 };
        let start = Instant::now();
        let stats = engine.run(&copies, opts)?;
        #[cfg(unix)]
        super::metrics::record(super::metrics::Copy {
            kind, gpu: device, bytes: copies.iter().map(|(_, c)| c.dst.len() as u64).sum(),
            ns: start.elapsed().as_nanos() as u64, ceiling,
            link: super::metrics::device_link(device).map_or_else(String::new, |link| link.to_string()),
            busy_ns: stats.busy_ns, pin_ns: stats.pin_ns, staged: stats.staged, cached: stats.cached,
        });
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

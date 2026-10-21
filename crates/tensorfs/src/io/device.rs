//! B7's bounded pinned staging and reuse policy, independent of varena's placement types.

use std::collections::BTreeMap;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::time::Instant;

use super::cuda::{self, CUcontext, CUevent, CUstream, CtxGuard};
use super::{host, Copy, Dst, Error, ReadOpts, ReadStats, Src};

const PIECE: usize = 2 << 20;
/// Readers a run starts with until runs on the engine have learned better (`Grow`).
const READERS: usize = 8;
/// Slots each reader owns: one fills while the other's copy lands, and both stay in its caches.
/// A shared pool (LIFO or FIFO) measured 8-20% slower warm H2D.
const RING: usize = 2;
const STREAMS: usize = 4;
/// Bytes of a memfd's pages the host copy probe reads per reader count: past any CPU cache.
#[cfg(target_os = "linux")]
const COPY_PROBE: usize = 1 << 30;

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
    /// CPUs readers may run on.
    fn cpus(&self) -> usize {
        #[cfg(target_os = "linux")]
        if let Some(set) = &self.0 {
            // SAFETY: a valid mask.
            return unsafe { libc::CPU_COUNT(set) } as usize;
        }
        std::thread::available_parallelism().map_or(1, |n| n.get())
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
    /// Staging slots between runs; readers allocate what they lack on the GPU's node.
    free: Mutex<Vec<Slot>>,
    learned: Mutex<Learned>,
    serial: Mutex<()>,
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
        let engine = Arc::new(Engine {
            device,
            ctx,
            streams,
            queues: (0..STREAMS).map(|_| Mutex::new(())).collect(),
            near: Near::of(device),
            free: Mutex::new(Vec::new()),
            learned: Mutex::new(Learned::default()),
            serial: Mutex::new(()),
            held: Mutex::new(Vec::new()),
            staged: Mutex::new(Vec::new()),
        });
        engines.insert(device, engine.clone());
        Ok(engine)
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
        match complete {
            true => Ok(()),
            false => self.synchronize(owners),
        }
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

    /// A reader's slots: free ones, else new ones pinned on this thread's (the GPU's) node.
    fn ring(&self) -> cuda::Result<Vec<Slot>> {
        let mut ring = {
            let mut free = self.free();
            let at = free.len().saturating_sub(RING);
            free.split_off(at)
        };
        while ring.len() < RING {
            match Slot::new(self.ctx) {
                Ok(slot) => ring.push(slot),
                Err(e) if ring.is_empty() => return Err(e),
                Err(_) => break,
            }
        }
        Ok(ring)
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
                // A host range its owner keeps registered (`pins::PinnedRanges`) copies directly;
                // any other is staged. Registering one for a single read cost more than it saved
                // (cuMemHostRegister + unregister ~100 ms/GiB: 25% of the link against 70% staged).
                if let (Src::Host(ptr), Dst::Device { len, .. }, false) = (c.src, c.dst, staged_host) {
                    let registered = |p| unsafe { cuda::registered(p) }.map_err(|e| failed(copy, e));
                    direct = len > 0 && registered(ptr)? && registered(ptr.wrapping_add(len - 1))?;
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
            let cap = opts
                .threads
                .clamp(1, self.near.cpus())
                .min(jobs.len().max(1));
            let learned = *self.learned.lock().unwrap_or_else(|e| e.into_inner());
            let h2d = super::metrics::h2d_kept(self.device).unwrap_or(0.0);
            let mut grow = Grow::new(learned, cap, h2d);
            let (allowed, running) = (AtomicUsize::new(grow.readers), AtomicUsize::new(0));
            // Leaves one reader when more run than `allowed`.
            let surplus = || {
                let mut n = running.load(Relaxed);
                while n > allowed.load(Relaxed) {
                    match running.compare_exchange(n, n - 1, Relaxed, Relaxed) {
                        Ok(_) => return true,
                        Err(now) => n = now,
                    }
                }
                false
            };
            // A reader that could pin no slot leaves the jobs to the others, if any.
            let starved = Mutex::new(None);
            let work = |tx: &mpsc::Sender<Read>| {
                self.near.enter();
                let ring = CtxGuard::enter(self.ctx).and_then(|_ctx| self.ring());
                let mut ring = match ring {
                    Ok(ring) => ring,
                    Err(e) => {
                        running.fetch_sub(1, Relaxed);
                        starved.lock().unwrap().get_or_insert(failed(first, e));
                        return;
                    }
                };
                let mut turn = 0;
                loop {
                    if surplus() {
                        break;
                    }
                    let at = next.fetch_add(1, Relaxed);
                    let job = jobs.get(at).filter(|_| !stopped.load(Relaxed));
                    let Some(job) = job else {
                        running.fetch_sub(1, Relaxed);
                        break;
                    };
                    turn += 1;
                    let n = ring.len();
                    let slot = &mut ring[turn % n];
                    match self.piece(job, at % streams, opts, &tally, slot) {
                        Ok(Some(cached)) => {
                            let _ = tx.send(Read::Piece { bytes: job.len as u64, cached });
                        }
                        Ok(None) => {}
                        Err(failure) => {
                            error.lock().unwrap().get_or_insert(failure);
                            stopped.store(true, Relaxed);
                        }
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
            // Every reader is TensorFS's own thread, so moving it to the GPU's CPUs moves no
            // caller. This thread grows them while that raises the staged rate.
            let (tx, rx) = mpsc::channel();
            let clock = Instant::now();
            std::thread::scope(|scope| {
                let spawn = |count: usize| {
                    for _ in 0..count {
                        running.fetch_add(1, Relaxed);
                        let leave = Leave(tx.clone());
                        scope.spawn(move || work(&leave.0));
                    }
                };
                let (mut target, mut alive) = (grow.readers, grow.readers);
                spawn(target);
                while alive > 0 {
                    let change = match rx.recv() {
                        Ok(Read::Done) => {
                            alive -= 1;
                            continue;
                        }
                        Ok(Read::Piece { bytes, cached }) => {
                            grow.piece(bytes, cached, clock.elapsed().as_nanos() as u64)
                        }
                        Err(_) => break,
                    };
                    if let Some(readers) = change {
                        allowed.store(readers, Relaxed);
                        if readers > target {
                            spawn(readers - target);
                            alive += readers - target;
                        }
                        target = readers;
                    }
                }
            });
            grow.learn(&mut self.learned.lock().unwrap_or_else(|e| e.into_inner()));
            if next.load(Relaxed) < jobs.len() {
                if let Some(starved) = starved.into_inner().unwrap() {
                    error.lock().unwrap().get_or_insert(starved);
                }
            }
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
                staged: tally.staged.load(Relaxed),
                busy_ns,
                readers: grow.readers as u64,
                ..ReadStats::default()
            })
        })();
        let released = hold.finish(complete).map_err(|e| failed(first, e));
        let stats = result?;
        released?;
        Ok(stats)
    }

    /// One job; for a staged file piece, whether the page cache held all of it (growth reads only
    /// those).
    fn piece(
        &self,
        job: &Job,
        lane: usize,
        opts: &ReadOpts,
        tally: &Tally,
        slot: &mut Slot,
    ) -> Result<Option<bool>, Error> {
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
            return cuda::check("cuMemcpyAsync", result)
                .map(|()| None)
                .map_err(|e| failed(job.copy, e));
        }
        if slot.pending {
            slot.done.sync().map_err(|e| failed(job.copy, e))?;
            slot.pending = false;
        }
        let mut cached = None;
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
                cached = Some(stats.direct == 0);
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
        Ok(cached)
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
    staged: AtomicU64,
}

/// What a reader tells the run: a staged piece queued to the GPU, or that it left.
enum Read {
    Piece { bytes: u64, cached: bool },
    Done,
}

/// Tells the run its reader left, on unwind too.
struct Leave(mpsc::Sender<Read>);

impl Drop for Leave {
    fn drop(&mut self) {
        let _ = self.0.send(Read::Done);
    }
}

/// Readers per source (0: the page cache, 1: the disk), learned by runs on one engine; a source
/// is settled once growing its readers stopped helping.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Learned {
    readers: [usize; 2],
    settled: [bool; 2],
    last: usize,
}

impl Default for Learned {
    fn default() -> Learned {
        Learned { readers: [READERS; 2], settled: [false; 2], last: 0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Phase {
    /// The window after a change, left to settle; the run's first also names its source.
    Ramp,
    /// Two windows at the kept readers: the base rate and its spread.
    Base(Option<f64>),
    /// `trial` readers started: the window to settle, then the one measured.
    Ramped(usize),
    Trial(usize),
    Done,
}

/// Grows a run's readers while that raises its staged rate (bytes queued to the GPU per ns), in
/// windows of two turns of every ring. A step of a quarter more readers is kept only when it beat
/// the base rate by more than the base's own window-to-window spread, and growth stops at `cap`
/// or the GPU's H2D ceiling. Pure, so the policy is tested without a GPU.
#[derive(Debug)]
struct Grow {
    learned: Learned,
    /// Readers proven (or learned) for this run, and those running now.
    readers: usize,
    target: usize,
    cap: usize,
    h2d: f64,
    phase: Phase,
    source: Option<usize>,
    /// The window so far: bytes, cached bytes, and when it began (ns).
    window: (u64, u64, u64),
    base: f64,
    spread: f64,
}

impl Grow {
    fn new(learned: Learned, cap: usize, h2d: f64) -> Grow {
        let readers = learned.readers[learned.last].clamp(1, cap);
        let (phase, source, window, base, spread) = (Phase::Ramp, None, (0, 0, 0), 0.0, 0.0);
        Grow { learned, readers, target: readers, cap, h2d, phase, source, window, base, spread }
    }

    /// A staged piece queued at `now` ns: the readers to run with, when they change.
    fn piece(&mut self, bytes: u64, cached: bool, now: u64) -> Option<usize> {
        if self.phase == Phase::Done {
            return None;
        }
        let (total, held, since) = &mut self.window;
        (*total, *held) = (*total + bytes, *held + if cached { bytes } else { 0 });
        if *total < (2 * self.target * RING * PIECE) as u64 {
            return None;
        }
        let rate = *total as f64 / now.saturating_sub(*since).max(1) as f64;
        let source = usize::from(*held * 2 < *total);
        self.window = (0, 0, now);
        let next = match (self.phase, self.source) {
            (Phase::Ramp, None) => {
                self.source = Some(source);
                let learned = self.learned.readers[source].clamp(1, self.cap);
                match std::mem::replace(&mut self.readers, learned) == learned {
                    true => self.settled(source),
                    false => Phase::Ramp,
                }
            }
            (Phase::Ramp, Some(source)) => self.settled(source),
            (Phase::Base(None), _) => Phase::Base(Some(rate)),
            (Phase::Base(Some(first)), _) => {
                (self.base, self.spread) = (first.max(rate), (first - rate).abs());
                self.step()
            }
            (Phase::Ramped(trial), _) => Phase::Trial(trial),
            (Phase::Trial(trial), _) if rate > self.base + self.spread => {
                (self.readers, self.base) = (trial, rate);
                self.step()
            }
            (Phase::Trial(_) | Phase::Done, _) => Phase::Done,
        };
        self.phase = next;
        let target = match next {
            Phase::Ramped(trial) | Phase::Trial(trial) => trial,
            _ => self.readers,
        };
        (std::mem::replace(&mut self.target, target) != target).then_some(target)
    }

    /// Growth goes on for a source not yet settled.
    fn settled(&self, source: usize) -> Phase {
        match self.learned.settled[source] {
            true => Phase::Done,
            false => Phase::Base(None),
        }
    }

    /// The next trial, or done at the cap or the link's ceiling.
    fn step(&self) -> Phase {
        let trial = (self.readers + (self.readers / 4).max(1)).min(self.cap);
        match trial == self.readers || (self.h2d > 0.0 && self.base >= self.h2d) {
            true => Phase::Done,
            false => Phase::Ramped(trial),
        }
    }

    /// What the run proved, for the engine's next runs.
    fn learn(&self, learned: &mut Learned) {
        if let Some(source) = self.source {
            learned.readers[source] = self.readers;
            learned.settled[source] |= self.phase == Phase::Done;
            learned.last = source;
        }
    }
}

/// Page cache → pinned staging GB/s on GPU `device`'s readers, and the readers that reached it:
/// a memfd's pages read into each reader's ring as staging reads a cached file, the readers
/// doubling while that helped, up to the CPUs they may use. Slots beyond what the best count
/// (or a run's start) needs are freed again.
#[cfg(target_os = "linux")]
pub(super) fn copy_ceiling(device: i32) -> io::Result<(f64, u32)> {
    use std::os::fd::FromRawFd;
    use std::os::unix::fs::FileExt;
    let engine = Engine::get(device).map_err(io::Error::other)?;
    let _serial = engine.serial.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: a fresh anonymous file, owned below. The syscall, not glibc's wrapper (2.27+): the
    // `tfs` binary links against manylinux2014's glibc 2.17.
    let fd = unsafe { libc::syscall(libc::SYS_memfd_create, c"tensorfs-copy-probe".as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the descriptor was just created and nothing else owns it.
    let file = unsafe { std::fs::File::from_raw_fd(fd as i32) };
    let block = vec![0x5a; PIECE];
    for at in (0..COPY_PROBE).step_by(PIECE) {
        file.write_all_at(&block, at as u64)?;
    }
    let (cpus, mut best, mut readers, mut n) = (engine.near.cpus(), 0.0, 0, 1);
    let measured = loop {
        match engine.copy_rate(&file, n) {
            Ok(gbps) if gbps > best => (best, readers) = (gbps, n),
            Ok(_) => break Ok(()),
            Err(e) => break Err(e),
        }
        match n == cpus {
            true => break Ok(()),
            false => n = (n * 2).min(cpus),
        }
    };
    engine.free().truncate(RING * readers.max(READERS));
    measured.map(|()| (best, readers as u32))
}

#[cfg(not(target_os = "linux"))]
pub(super) fn copy_ceiling(_: i32) -> io::Result<(f64, u32)> {
    Err(io::Error::new(io::ErrorKind::Unsupported, "no host copy probe on this platform"))
}

impl Engine {
    /// GB/s `readers` reach reading all of `file` (cached) into their rings, timed from when every
    /// reader holds its slots. Caller holds `serial`.
    #[cfg(target_os = "linux")]
    fn copy_rate(&self, file: &std::fs::File, readers: usize) -> io::Result<f64> {
        let (next, ready) = (AtomicUsize::new(0), std::sync::Barrier::new(readers + 1));
        let read = || -> io::Result<()> {
            self.near.enter();
            let ring = CtxGuard::enter(self.ctx).and_then(|_ctx| self.ring());
            ready.wait();
            let mut ring = ring.map_err(io::Error::other)?;
            let result = (|| {
                for turn in 0.. {
                    let at = next.fetch_add(1, Relaxed);
                    if at >= COPY_PROBE / PIECE {
                        break;
                    }
                    let n = ring.len();
                    let slot = &mut ring[turn % n];
                    if std::mem::take(&mut slot.pending) {
                        slot.done.sync().map_err(io::Error::other)?;
                    }
                    // SAFETY: the slot's PIECE bytes, with no copy in flight from them.
                    let buf = unsafe { std::slice::from_raw_parts_mut(slot.ptr, PIECE) };
                    host::read_exact_at(file, buf, (at * PIECE) as u64)?;
                }
                Ok(())
            })();
            self.free().extend(ring);
            result
        };
        let start = std::thread::scope(|s| {
            let readers: Vec<_> = (0..readers).map(|_| s.spawn(read)).collect();
            ready.wait();
            let start = Instant::now();
            readers.into_iter().try_for_each(|r| r.join().expect("a probe reader does not panic"))?;
            io::Result::Ok(start)
        })?;
        Ok(COPY_PROBE as f64 / start.elapsed().as_nanos().max(1) as f64)
    }
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
            let disk = super::disk_run(&plain, &stats, ns);
            let run = super::metrics::Copy {
                kind,
                gpu: device,
                bytes: copies.iter().map(|(_, c)| c.dst.len() as u64).sum(),
                ns,
                link: super::metrics::device_link(device)
                    .map_or_else(String::new, |link| link.to_string()),
                busy_ns: stats.busy_ns,
                readers: stats.readers,
                pin_ns: stats.pin_ns,
                staged: stats.staged,
                cached: stats.cached,
                ..Default::default()
            };
            match kind {
                "h2d" => super::metrics::h2d_run(run, stats.direct, disk),
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
        total.readers = total.readers.max(stats.readers);
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

    /// Feeds `grow` staged pieces at `gbps(readers)`: the reader counts it switches to.
    fn drive(
        grow: &mut Grow,
        gbps: impl Fn(usize) -> f64,
        pieces: usize,
        cached: bool,
    ) -> Vec<usize> {
        let (mut now, mut readers, mut changes) = (0, grow.target, Vec::new());
        for _ in 0..pieces {
            now += (PIECE as f64 / gbps(readers)).ceil() as u64;
            if let Some(n) = grow.piece(PIECE as u64, cached, now) {
                (readers, changes) = (n, [changes, vec![n]].concat());
            }
        }
        changes
    }

    #[test]
    fn readers_grow_while_the_staged_rate_rises_and_the_last_step_is_retired() {
        let mut grow = Grow::new(Learned::default(), 64, 0.0);
        let changes = drive(&mut grow, |n| (2.0 * n as f64).min(26.0), 4000, true);
        assert_eq!(changes, [10, 12, 15, 18, 15]);
        let mut learned = Learned::default();
        grow.learn(&mut learned);
        assert_eq!(
            learned,
            Learned {
                readers: [15, READERS],
                settled: [true, false],
                last: 0
            }
        );
    }

    #[test]
    fn a_flat_rate_keeps_the_readers_and_settles_after_one_step() {
        let mut grow = Grow::new(Learned::default(), 64, 0.0);
        assert_eq!(drive(&mut grow, |_| 20.0, 4000, false), [10, READERS]);
        let mut learned = Learned::default();
        grow.learn(&mut learned);
        assert_eq!(
            learned,
            Learned {
                readers: [READERS; 2],
                settled: [false, true],
                last: 1
            }
        );
    }

    #[test]
    fn a_settled_source_switches_once_and_the_link_ceiling_or_cap_stops_growth() {
        let learned = Learned {
            readers: [12, READERS],
            settled: [true, false],
            last: 1,
        };
        let mut grow = Grow::new(learned, 64, 0.0);
        assert_eq!(drive(&mut grow, |n| 2.0 * n as f64, 4000, true), [12]);
        assert_eq!(grow.phase, Phase::Done);
        let mut grow = Grow::new(Learned::default(), 64, 16.0);
        assert!(drive(&mut grow, |n| 2.0 * n as f64, 4000, true).is_empty());
        let mut grow = Grow::new(Learned::default(), 9, 0.0);
        assert_eq!(drive(&mut grow, |n| 2.0 * n as f64, 4000, true), [9]);
        assert_eq!((grow.readers, grow.phase), (9, Phase::Done));
    }

    #[test]
    fn a_step_within_the_base_windows_spread_is_not_kept() {
        let mut grow = Grow::new(Learned::default(), 64, 0.0);
        let mut window = 0;
        let (mut now, mut readers, mut changes) = (0.0, grow.target, Vec::new());
        for _ in 0..4000 {
            // Base windows alternate 16 and 20 GB/s; one more reader adds 2.
            window += 1;
            let noise = if (window / 32) % 2 == 0 { 0.0 } else { 4.0 };
            now += PIECE as f64 / (16.0 + noise + 2.0 * (readers - READERS) as f64 / 2.0);
            if let Some(n) = grow.piece(PIECE as u64, true, now as u64) {
                (readers, changes) = (n, [changes, vec![n]].concat());
            }
        }
        assert_eq!(changes.last(), Some(&READERS));
    }

    #[test]
    fn draining_a_device_without_an_engine_does_not_initialize_cuda() {
        // No CUDA discovery, context creation, allocation or driver call is needed.
        assert!(!registry().lock().unwrap().contains_key(&i32::MIN));
        drain(i32::MIN).unwrap();
        assert!(!registry().lock().unwrap().contains_key(&i32::MIN));
    }
}

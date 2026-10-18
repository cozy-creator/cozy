//! One GPU's mechanisms: its event pool, VMM allocations and mappings, and the copy movers that
//! carry the chunk reader.
//!
//! A mover owns one copy stream and a pool of pinned staging slots. It takes jobs in order and
//! splits each into slot-sized pieces; the reader threads fill the slots (chunk reads, O_DIRECT
//! where they can, or copies from host memory) and each piece moves to the device in one copy
//! as soon as its slot is full. Pieces of later jobs start while earlier ones are still reading,
//! so the disk always has every free slot's request in flight.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use crate::cuda::{self, AccessDesc, AllocationProp, CUcontext, CUevent, CUstream, CtxGuard, Handle};
use crate::host::HostMem;
use crate::io::{Files, Ptr, Readers};
use crate::layout::{align_up, PART_ALIGN, REGION_ALIGN};
use crate::{Error, Result};

// ---------------------------------------------------------------- events

/// A pooled event. Dropping the last reference returns it to the pool: nothing can still
/// query its old recording, and waits already enqueued captured it, so reuse is safe even
/// while the device has not reached it.
pub struct Ev {
    raw: CUevent,
    timing: bool,
    pool: Arc<EventPool>,
}
unsafe impl Send for Ev {}
unsafe impl Sync for Ev {}
pub type EvRef = Arc<Ev>;

impl Ev {
    pub fn raw(&self) -> CUevent {
        self.raw
    }
    /// Has the device passed this event's recording?
    pub fn done(&self) -> Result<bool> {
        let d = cuda::driver()?;
        // SAFETY: a live event of our context.
        match unsafe { (d.event_query)(self.raw) } {
            cuda::SUCCESS => Ok(true),
            cuda::ERROR_NOT_READY => Ok(false),
            rc => cuda::check("cuEventQuery", rc).map(|_| false),
        }
    }
    /// Host wait for this one event: targeted, never a device-wide sync.
    pub fn sync(&self) -> Result<()> {
        let d = cuda::driver()?;
        // SAFETY: a live event.
        cuda::check("cuEventSynchronize", unsafe { (d.event_synchronize)(self.raw) })
    }
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // an opaque driver handle
    pub fn record(&self, stream: CUstream) -> Result<()> {
        let d = cuda::driver()?;
        // SAFETY: event and stream of the same context (the caller binds it).
        cuda::check("cuEventRecord", unsafe { (d.event_record)(self.raw, stream) })
    }
    /// Milliseconds from `start` to `self`, both timing events and both done.
    pub fn since(&self, start: &Ev) -> Result<f32> {
        let d = cuda::driver()?;
        let mut ms = 0f32;
        // SAFETY: two completed timing events.
        cuda::check("cuEventElapsedTime", unsafe {
            (d.event_elapsed_time)(&mut ms, start.raw, self.raw)
        })?;
        Ok(ms)
    }
}

impl Drop for Ev {
    fn drop(&mut self) {
        let mut f = self.pool.free.lock().unwrap();
        if self.timing { &mut f.1 } else { &mut f.0 }.push(self.raw);
    }
}

pub struct EventPool {
    ctx: CUcontext,
    free: Mutex<(Vec<CUevent>, Vec<CUevent>)>,
}
unsafe impl Send for EventPool {}
unsafe impl Sync for EventPool {}

impl EventPool {
    pub fn new(ctx: CUcontext) -> EventPool {
        EventPool {
            ctx,
            free: Mutex::new((Vec::new(), Vec::new())),
        }
    }

    pub fn get(self: &Arc<Self>, timing: bool) -> Result<EvRef> {
        let reuse = {
            let mut f = self.free.lock().unwrap();
            if timing { &mut f.1 } else { &mut f.0 }.pop()
        };
        let raw = match reuse {
            Some(r) => r,
            None => {
                let _g = CtxGuard::enter(self.ctx)?;
                let d = cuda::driver()?;
                let mut e: CUevent = std::ptr::null_mut();
                let flags = if timing { cuda::EVENT_DEFAULT } else { cuda::EVENT_DISABLE_TIMING };
                // SAFETY: out-param, context bound.
                cuda::check("cuEventCreate", unsafe { (d.event_create)(&mut e, flags) })?;
                e
            }
        };
        Ok(Arc::new(Ev {
            raw,
            timing,
            pool: self.clone(),
        }))
    }
}

impl Drop for EventPool {
    fn drop(&mut self) {
        if let Ok(d) = cuda::driver() {
            let f = self.free.get_mut().unwrap();
            for e in f.0.drain(..).chain(f.1.drain(..)) {
                // SAFETY: our events; destroy of a pending event defers its release.
                unsafe { (d.event_destroy)(e) };
            }
        }
    }
}

// ---------------------------------------------------------------- allocations

/// One mapped physical handle and its size.
#[derive(Debug, Clone, Copy)]
pub struct Chunk {
    pub handle: Handle,
    pub size: u64,
}

/// A physical allocation of `size` bytes on `prop`'s device (its context bound). The driver's
/// out-of-memory is `DeviceFull`.
pub fn create(prop: &AllocationProp, size: u64) -> Result<Chunk> {
    let d = cuda::driver()?;
    let mut h: Handle = 0;
    // SAFETY: out-param; prop describes device memory of the bound context's device.
    let rc = unsafe { (d.mem_create)(&mut h, size as usize, prop, 0) };
    if rc == cuda::ERROR_OUT_OF_MEMORY {
        return Err(Error::DeviceFull { deficit: size });
    }
    cuda::check("cuMemCreate", rc)?;
    Ok(Chunk { handle: h, size })
}

/// (free, total) device memory, from the driver (the device's context bound).
pub fn mem_info() -> Result<(u64, u64)> {
    let d = cuda::driver()?;
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: out-params.
    cuda::check("cuMemGetInfo", unsafe { (d.mem_get_info)(&mut free, &mut total) })?;
    Ok((free as u64, total as u64))
}

/// Map `chunks` contiguously at `va` and grant `device` `access` (a `MEM_ACCESS_PROT_*`) over
/// the span. On failure every mapping made here is undone; `Err(Poisoned)` if an undo failed.
pub fn map_span(va: u64, chunks: &[Chunk], device: i32, access: u32) -> Result<()> {
    let d = cuda::driver()?;
    let mut at = va;
    let mut mapped = 0usize;
    let mut failure = None;
    for c in chunks {
        // SAFETY: `va` lies in a reservation we own; the handle is unmapped.
        if let Err(e) = cuda::check("cuMemMap", unsafe { (d.mem_map)(at, c.size as usize, 0, c.handle, 0) }) {
            failure = Some(e);
            break;
        }
        mapped += 1;
        at += c.size;
    }
    if failure.is_none() {
        let desc = AccessDesc {
            location: cuda::MemLocation {
                kind: cuda::MEM_LOCATION_TYPE_DEVICE,
                id: device,
            },
            flags: access,
        };
        // SAFETY: the span was just mapped.
        if let Err(e) = cuda::check("cuMemSetAccess", unsafe {
            (d.mem_set_access)(va, (at - va) as usize, &desc, 1)
        }) {
            failure = Some(e);
        }
    }
    let Some(e) = failure else { return Ok(()) };
    let mut at = va;
    for c in &chunks[..mapped] {
        // SAFETY: undoing our own mapping.
        let rc = unsafe { (d.mem_unmap)(at, c.size as usize) };
        if rc != cuda::SUCCESS {
            return Err(Error::Poisoned(format!("rollback cuMemUnmap({rc}) after {e}")));
        }
        at += c.size;
    }
    Err(e)
}

/// Release one handle reference. A failure is ambiguous driver state: the caller poisons.
pub fn release(h: Handle) -> Result<()> {
    let d = cuda::driver()?;
    // SAFETY: a handle this process created or imported, no longer mapped by us.
    cuda::check("cuMemRelease", unsafe { (d.mem_release)(h) })
}

/// Unmap a mapped span. A failure is ambiguous driver state: the caller poisons.
pub fn unmap_span(va: u64, len: u64) -> Result<()> {
    let d = cuda::driver()?;
    // SAFETY: the span is a mapping we made; no kernel can touch it (leases are clear).
    cuda::check("cuMemUnmap", unsafe { (d.mem_unmap)(va, len as usize) })
}

pub fn reserve_va(size: u64) -> Result<u64> {
    let d = cuda::driver()?;
    let mut base = 0u64;
    // SAFETY: out-param; size is a granularity multiple.
    cuda::check("cuMemAddressReserve", unsafe {
        (d.mem_address_reserve)(&mut base, size as usize, REGION_ALIGN as usize, 0, 0)
    })?;
    Ok(base)
}

pub fn free_va(base: u64, size: u64) -> Result<()> {
    let d = cuda::driver()?;
    // SAFETY: a reservation we made, fully unmapped.
    cuda::check("cuMemAddressFree", unsafe { (d.mem_address_free)(base, size as usize) })
}

// ---------------------------------------------------------------- movers

/// Where a device copy reads from.
pub enum CopySrc {
    /// Page-locked host bytes: one DMA.
    Host(*const u8),
    /// Host bytes this process could not page-lock: through the staging slots, never as pageable
    /// memory (a pageable copy runs at a fraction of the link and stalls the stream).
    Bounce(*const u8),
    /// A sealed layout's region, once its owner marks it Ready: page-locked on its first copy,
    /// else bounced. This reference keeps the layout mapped.
    Sealed { host: Arc<HostMem>, region: u32, offset: u64 },
    /// Chunk files through the staging slots: runs of (chunk, offset in the copy, length).
    Files { files: Arc<Files>, runs: Vec<(u32, u64, u64)> },
}
unsafe impl Send for CopySrc {}

/// Where a copy's bytes came from, for the facts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyPath {
    Pinned,
    Bounced,
    Files,
}

pub struct Copied {
    pub ready: EvRef,
    pub start: EvRef,
    pub bytes: u64,
    pub path: CopyPath,
    /// This copy's region was refused a page lock (counted once per region).
    pub refused: bool,
}

type Done = Box<dyn FnOnce(Result<Copied>) + Send>;

/// `len` bytes to `dst` once `waits` have passed on the copy stream. `done` runs once every
/// byte is queued (Ok, with the event that marks them landed), or after a failure drained.
pub struct CopyJob {
    pub waits: Vec<EvRef>,
    pub dst: u64,
    pub len: u64,
    pub src: CopySrc,
    pub done: Done,
}

/// A pinned staging slot and the copy that last read it.
struct Slot {
    ptr: *mut u8,
    last: Option<EvRef>,
}
unsafe impl Send for Slot {}

struct Slots {
    free: Mutex<Vec<Slot>>,
    cv: Condvar,
}

impl Slots {
    /// A free slot whose last copy has landed.
    fn take(&self) -> Result<Slot> {
        let mut free = self.free.lock().unwrap();
        while free.is_empty() {
            free = self.cv.wait(free).unwrap();
        }
        let mut slot = free.pop().unwrap();
        drop(free);
        if let Some(ev) = slot.last.take() {
            if let Err(e) = ev.sync() {
                self.put(slot);
                return Err(e);
            }
        }
        Ok(slot)
    }
    fn put(&self, slot: Slot) {
        self.free.lock().unwrap().push(slot);
        self.cv.notify_one();
    }
}

#[derive(Clone, Copy)]
struct Stream(CUstream);
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

/// One staged job's pieces in flight: the last to finish reports it.
struct Staged {
    left: AtomicUsize,
    failed: Mutex<Option<Error>>,
    done: Mutex<Option<Done>>,
    stream: Stream,
    start: EvRef,
    ready: EvRef,
    bytes: u64,
    path: CopyPath,
    refused: bool,
    /// The sealed layout a bounced region reads from, mapped until the last piece.
    _kept: Option<Arc<HostMem>>,
}

impl Staged {
    fn fail(&self, e: Error) {
        self.failed.lock().unwrap().get_or_insert(e);
    }
    fn failed(&self) -> bool {
        self.failed.lock().unwrap().is_some()
    }
    /// The last piece out: every copy of the job is queued, so `ready` after them covers all.
    fn finish_one(&self) {
        if self.left.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        let res = match self.failed.lock().unwrap().take() {
            Some(e) => drained(self.stream.0, e),
            None => match self.ready.record(self.stream.0) {
                Ok(()) => Ok(Copied {
                    ready: self.ready.clone(),
                    start: self.start.clone(),
                    bytes: self.bytes,
                    path: self.path,
                    refused: self.refused,
                }),
                Err(e) => drained(self.stream.0, e),
            },
        };
        if let Some(done) = self.done.lock().unwrap().take() {
            done(res);
        }
    }
}

/// A failure reported only once nothing it queued is in flight: no caller unmaps or reuses a
/// range a copy still touches. A drain that itself fails is ambiguous device state: `Poisoned`.
fn drained<T>(stream: CUstream, e: Error) -> Result<T> {
    let d = cuda::driver()?;
    // SAFETY: our own stream.
    let drain = crate::faults::hit("drain").and_then(|_| cuda::check("cuStreamSynchronize", unsafe { (d.stream_synchronize)(stream) }));
    match drain {
        Ok(()) => Err(e),
        Err(drain) => Err(Error::Poisoned(format!("a failed copy ({e}) could not be drained: {drain}"))),
    }
}

/// One copy stream and the thread that dispatches its jobs, in submission order.
pub struct Mover {
    name: String,
    tx: Mutex<Option<Sender<CopyJob>>>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

pub struct MoverCtx {
    pub ctx: CUcontext,
    pub events: Arc<EventPool>,
    pub readers: Arc<Readers>,
    /// (slots, bytes each): bytes is a multiple of 4 KiB.
    pub staging: (usize, u64),
}
unsafe impl Send for MoverCtx {}
unsafe impl Sync for MoverCtx {}

impl Mover {
    pub fn start(name: String, m: MoverCtx) -> Result<Mover> {
        let (tx, rx) = channel::<CopyJob>();
        let (ok_tx, ok_rx) = channel::<Result<()>>();
        let thread = std::thread::Builder::new()
            .name(name.clone())
            .spawn(move || mover_main(Arc::new(m), rx, ok_tx))
            .map_err(|e| Error::Io(format!("spawn mover: {e}")))?;
        ok_rx.recv().map_err(|_| Error::Io("mover died at start".into()))??;
        Ok(Mover { name, tx: Mutex::new(Some(tx)), thread: Mutex::new(Some(thread)) })
    }

    /// Queue a copy. On `Err` the job was dropped without calling `done`.
    pub fn submit(&self, job: CopyJob) -> Result<()> {
        if self.tx.lock().unwrap().as_ref().is_some_and(|tx| tx.send(job).is_ok()) {
            return Ok(());
        }
        Err(Error::Closed(format!("copy stream {} has stopped", self.name)))
    }

    /// Close the queue: the thread finishes its jobs, frees its slots and exits by itself.
    pub fn detach(&self) {
        self.tx.lock().unwrap().take();
    }

    pub fn stop(&self) {
        self.detach();
        if let Some(t) = self.thread.lock().unwrap().take() {
            let _ = t.join();
        }
    }
}

fn mover_main(m: Arc<MoverCtx>, rx: Receiver<CopyJob>, ok: Sender<Result<()>>) {
    let (n, size) = m.staging;
    let init = (|| -> Result<(CUstream, Vec<Slot>)> {
        let d = cuda::driver()?;
        cuda::check("cuCtxSetCurrent", unsafe { (d.ctx_set_current)(m.ctx) })?;
        let mut s: CUstream = std::ptr::null_mut();
        // SAFETY: out-param, context bound.
        cuda::check("cuStreamCreate", unsafe { (d.stream_create)(&mut s, cuda::STREAM_NON_BLOCKING) })?;
        let mut slots = Vec::new();
        for _ in 0..n.max(1) {
            let mut p: *mut libc::c_void = std::ptr::null_mut();
            // SAFETY: out-param; pinned, portable host memory (page aligned).
            cuda::check("cuMemHostAlloc", unsafe { (d.mem_host_alloc)(&mut p, size as usize, cuda::MEMHOSTALLOC_PORTABLE) })?;
            slots.push(Slot { ptr: p as *mut u8, last: None });
        }
        Ok((s, slots))
    })();
    let (stream, slots) = match init {
        Ok((s, slots)) => {
            let _ = ok.send(Ok(()));
            (Stream(s), Arc::new(Slots { free: Mutex::new(slots), cv: Condvar::new() }))
        }
        Err(e) => {
            let _ = ok.send(Err(e));
            return;
        }
    };
    while let Ok(job) = rx.recv() {
        dispatch(&m, stream, &slots, job);
    }
    // Every piece in flight returns its slot; then the stream drains and the slots go.
    let mut free = slots.free.lock().unwrap();
    while free.len() < n.max(1) {
        free = slots.cv.wait(free).unwrap();
    }
    if let Ok(d) = cuda::driver() {
        // SAFETY: our stream and slots; the stream is drained first.
        unsafe {
            (d.stream_synchronize)(stream.0);
            (d.stream_destroy)(stream.0);
            for s in free.drain(..) {
                (d.mem_free_host)(s.ptr as *mut libc::c_void);
            }
        }
    }
}

/// One slot's worth of a staged job: read `n` bytes (a chunk's from `off`, or host memory at
/// `off`) into a slot, then copy them to `at` in the job.
struct Piece {
    chunk: Option<u32>,
    off: u64,
    at: u64,
    n: u64,
}

fn dispatch(m: &Arc<MoverCtx>, stream: Stream, slots: &Arc<Slots>, job: CopyJob) {
    let CopyJob { waits, dst, len, src, done } = job;
    let s = stream.0;
    // A sealed region: never a byte its owner has not finished.
    let (src, kept, refused) = match src {
        CopySrc::Sealed { host, region, offset } => {
            if let Err(e) = host.wait_ready(region) {
                return done(Err(e));
            }
            let (pinned, now) = host.pin_region(region, offset, align_up(len, PART_ALIGN));
            // SAFETY: inside the live mapping `host` keeps; the region is complete.
            let p = unsafe { host.base().add(offset as usize) } as *const u8;
            (if pinned { CopySrc::Host(p) } else { CopySrc::Bounce(p) }, Some(host), now)
        }
        src => (src, None, false),
    };
    let begun = (|| -> Result<(EvRef, EvRef)> {
        let d = cuda::driver()?;
        for w in &waits {
            // SAFETY: events of this context; the stream is ours.
            cuda::check("cuStreamWaitEvent", unsafe { (d.stream_wait_event)(s, w.raw(), 0) })?;
        }
        crate::faults::gate("copy", s)?;
        let start = m.events.get(true)?;
        start.record(s)?;
        Ok((start, m.events.get(true)?))
    })();
    let (start, ready) = match begun {
        Ok(x) => x,
        Err(e) => return done(drained(s, e)),
    };
    let size = m.staging.1;
    let (files, from, path, pieces) = match src {
        CopySrc::Host(p) => {
            let copied = (|| {
                let d = cuda::driver()?;
                // SAFETY: `p` covers `len` page-locked bytes, kept alive by the books.
                cuda::check("cuMemcpyHtoDAsync", unsafe { (d.memcpy_htod_async)(dst, p as *const libc::c_void, len as usize, s) })?;
                crate::faults::hit("host-copy-recorded")?;
                ready.record(s)
            })();
            return done(match copied {
                Ok(()) => Ok(Copied { ready, start, bytes: len, path: CopyPath::Pinned, refused }),
                Err(e) => drained(s, e),
            });
        }
        CopySrc::Bounce(p) => {
            let pieces = (0..len.div_ceil(size)).map(|i| Piece { chunk: None, off: i * size, at: i * size, n: size.min(len - i * size) });
            (None, Some(Ptr(p as *mut u8)), CopyPath::Bounced, pieces.collect::<Vec<_>>())
        }
        CopySrc::Files { files, runs } => {
            let mut pieces = Vec::new();
            for (c, at, n) in runs {
                for i in 0..n.div_ceil(size) {
                    pieces.push(Piece { chunk: Some(c), off: i * size, at: at + i * size, n: size.min(n - i * size) });
                }
            }
            (Some(files), None, CopyPath::Files, pieces)
        }
        CopySrc::Sealed { .. } => unreachable!("resolved above"),
    };
    let job = Arc::new(Staged {
        left: AtomicUsize::new(pieces.len() + 1),
        failed: Mutex::new(None),
        done: Mutex::new(Some(done)),
        stream,
        start,
        ready,
        bytes: len,
        path,
        refused,
        _kept: kept,
    });
    for piece in pieces {
        if job.failed() {
            job.finish_one();
            continue;
        }
        let slot = match slots.take() {
            Ok(slot) => slot,
            Err(e) => {
                job.fail(e);
                job.finish_one();
                continue;
            }
        };
        let (m, job, slots, files) = (m.clone(), job.clone(), slots.clone(), files.clone());
        m.readers.clone().submit(move || {
            // Whole values, not their raw-pointer fields (closures capture fields precisely).
            let (m, mut slot, stream, from) = (m, slot, stream, from);
            let filled = match (&files, from) {
                // SAFETY: the slot is this piece's alone until it is returned below; it holds
                // `size` bytes, a multiple of 4 KiB, page aligned.
                (Some(f), _) => unsafe { f.read(piece.chunk.unwrap(), piece.off, piece.n, slot.ptr) },
                (None, Some(p)) => {
                    // SAFETY: the host range stays mapped (the books, or `_kept`) until done.
                    unsafe { std::ptr::copy_nonoverlapping(p.0.add(piece.off as usize), slot.ptr, piece.n as usize) };
                    Ok(())
                }
                (None, None) => unreachable!("a staged job reads files or host memory"),
            };
            let copied = filled.and_then(|_| {
                if job.failed() {
                    return Ok(());
                }
                let _g = CtxGuard::enter(m.ctx)?;
                crate::faults::hit("staged-copy")?;
                let d = cuda::driver()?;
                // SAFETY: the slot holds the piece; pinned host memory; the destination is mapped.
                cuda::check("cuMemcpyHtoDAsync", unsafe {
                    (d.memcpy_htod_async)(dst + piece.at, slot.ptr as *const libc::c_void, piece.n as usize, stream.0)
                })?;
                let e = m.events.get(false)?;
                e.record(stream.0)?;
                slot.last = Some(e);
                Ok(())
            });
            if let Err(e) = copied {
                job.fail(e);
            }
            slots.put(slot);
            match CtxGuard::enter(m.ctx) {
                Ok(_g) => job.finish_one(),
                Err(e) => {
                    job.fail(e);
                    job.finish_one();
                }
            }
        });
    }
    job.finish_one();
}

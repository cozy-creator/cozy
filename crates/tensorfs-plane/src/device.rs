//! One GPU: its primary context, event pool, VMM arena ledger and copy movers.
//!
//! The arena is varena's core with two changes: a region is backed by slab-sized physical
//! handles (the caller's `slab_bytes`, or one handle per region) instead of one per 2 MiB, and
//! handles are POSIX-fd exportable where the device supports it. Budget-fed, recycle before
//! create, refuse before mutation, else roll back exactly, else poison.

use std::collections::{BTreeMap, VecDeque};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use crate::cuda::{self, AccessDesc, AllocationProp, CUcontext, CUevent, CUstream, CtxGuard, Handle};
use crate::io::{Batch, Dst, IoTally, Mode, Readers, Source};
use crate::layout::{Item, ItemSource, REGION_ALIGN};
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

// ---------------------------------------------------------------- the arena ledger

/// One mapped physical handle and its size.
#[derive(Debug, Clone, Copy)]
pub struct Chunk {
    pub handle: Handle,
    pub size: u64,
}

/// Budget arithmetic plus the idle-handle pool. Never grows itself, never measures the device.
pub struct Arena {
    pub prop: AllocationProp,
    pub budget: u64,
    /// Physical bytes held: mapped plus idle.
    pub committed: u64,
    pub mapped: u64,
    pub slab: u64,
    idle: BTreeMap<u64, Vec<Handle>>,
}
unsafe impl Send for Arena {}

impl Arena {
    pub fn new(prop: AllocationProp, slab: u64) -> Arena {
        Arena {
            prop,
            budget: 0,
            committed: 0,
            mapped: 0,
            slab,
            idle: BTreeMap::new(),
        }
    }

    /// Handle sizes backing a span: slabs, then the remainder (or one handle when slab is 0).
    pub fn chunk_sizes(&self, span: u64) -> Vec<u64> {
        if self.slab == 0 || span <= self.slab {
            return vec![span];
        }
        let mut v = vec![self.slab; (span / self.slab) as usize];
        if !span.is_multiple_of(self.slab) {
            v.push(span % self.slab);
        }
        v
    }

    pub fn idle_bytes(&self) -> u64 {
        self.idle.iter().map(|(s, v)| s * v.len() as u64).sum()
    }

    fn release_one(&mut self, size: u64) -> Result<()> {
        let h = self.idle.get_mut(&size).and_then(Vec::pop).expect("idle handle");
        if self.idle.get(&size).is_some_and(Vec::is_empty) {
            self.idle.remove(&size);
        }
        let d = cuda::driver()?;
        // SAFETY: an unmapped handle we created.
        if let Err(e) = cuda::check("cuMemRelease", unsafe { (d.mem_release)(h) }) {
            self.idle.entry(size).or_default().push(h);
            return Err(e);
        }
        self.committed -= size;
        Ok(())
    }

    /// Handles for `sizes`: exact-size idle handles first; idle handles of other sizes are
    /// released to make headroom; the rest are created. Refuses before any mutation when the
    /// budget cannot cover it. The caller's context must be current.
    pub fn take(&mut self, sizes: &[u64]) -> Result<Vec<Chunk>> {
        let mut avail: BTreeMap<u64, usize> = self.idle.iter().map(|(s, v)| (*s, v.len())).collect();
        let mut create = 0u64;
        for s in sizes {
            match avail.get_mut(s) {
                Some(n) if *n > 0 => *n -= 1,
                _ => create += s,
            }
        }
        let spare: u64 = avail.iter().map(|(s, n)| s * *n as u64).sum();
        let headroom = self.budget.saturating_sub(self.committed);
        if create > headroom + spare {
            return Err(Error::BudgetExceeded {
                pool: "vram",
                requested: sizes.iter().sum(),
                available: sizes.iter().sum::<u64>() - create + headroom + spare,
            });
        }
        // Release non-matching idle handles, largest first, until the creates fit.
        while create > self.budget.saturating_sub(self.committed) {
            let size = *avail.iter().rev().find(|(_, n)| **n > 0).map(|(s, _)| s).unwrap();
            *avail.get_mut(&size).unwrap() -= 1;
            self.release_one(size)?;
        }
        let d = cuda::driver()?;
        let mut out: Vec<Chunk> = Vec::with_capacity(sizes.len());
        for &s in sizes {
            if let Some(h) = self.idle.get_mut(&s).and_then(Vec::pop) {
                out.push(Chunk { handle: h, size: s });
                continue;
            }
            let mut h: Handle = 0;
            // SAFETY: out-param; prop describes device memory of the bound context's device.
            let rc = unsafe { (d.mem_create)(&mut h, s as usize, &self.prop, 0) };
            if rc != cuda::SUCCESS {
                self.put_idle(out);
                // The device itself is full (the budget exceeds what is free): capacity, not
                // an ambiguous driver state.
                if rc == cuda::ERROR_OUT_OF_MEMORY {
                    return Err(Error::BudgetExceeded { pool: "device", requested: s, available: 0 });
                }
                return Err(cuda::check("cuMemCreate", rc).unwrap_err());
            }
            self.committed += s;
            out.push(Chunk { handle: h, size: s });
        }
        for s in self.idle.keys().copied().collect::<Vec<_>>() {
            if self.idle[&s].is_empty() {
                self.idle.remove(&s);
            }
        }
        Ok(out)
    }

    /// Return unmapped handles to the idle pool (still committed).
    pub fn put_idle(&mut self, chunks: Vec<Chunk>) {
        for c in chunks {
            self.idle.entry(c.size).or_default().push(c.handle);
        }
    }

    /// Release idle handles while committed exceeds the budget. Returns bytes released.
    pub fn trim(&mut self) -> Result<u64> {
        let before = self.committed;
        while self.committed > self.budget {
            let Some(size) = self.idle.keys().next_back().copied() else { break };
            self.release_one(size)?;
        }
        Ok(before - self.committed)
    }

    /// Release every idle handle (close).
    pub fn release_idle(&mut self) -> Result<()> {
        while let Some(size) = self.idle.keys().next_back().copied() {
            self.release_one(size)?;
        }
        Ok(())
    }
}

/// Map `chunks` contiguously at `va` and grant `device` read-write over the span. On failure
/// every mapping made here is undone; `Err(Poisoned)` if an undo itself failed.
pub fn map_span(va: u64, chunks: &[Chunk], device: i32) -> Result<()> {
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
            flags: cuda::MEM_ACCESS_PROT_READWRITE,
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

/// Unmap a mapped span. A failure is ambiguous driver state: the caller poisons.
pub fn unmap_span(va: u64, len: u64) -> Result<()> {
    let d = cuda::driver()?;
    // SAFETY: the span is a mapping we made; no kernel can touch it (leases are clear).
    cuda::check("cuMemUnmap", unsafe { (d.mem_unmap)(va, len as usize) })
}

/// What the DRIVER maps at `va`, independent of the plane's books.
pub fn driver_handle(va: u64) -> Result<Option<Handle>> {
    let d = cuda::driver()?;
    let mut h: Handle = 0;
    // SAFETY: out-param; an unmapped VA is an error return, which is the signal read here.
    let rc = unsafe { (d.mem_retain_allocation_handle)(&mut h, va as *mut libc::c_void) };
    if rc == cuda::ERROR_INVALID_VALUE {
        return Ok(None);
    }
    cuda::check("cuMemRetainAllocationHandle", rc)?;
    // SAFETY: balance the retain.
    cuda::check("cuMemRelease retained", unsafe { (d.mem_release)(h) })?;
    Ok(Some(h))
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
    /// Registered (or at worst pageable) host bytes of the pinned tier.
    Host(*const u8),
    /// Disk through the staging buffers: `items` in layout offsets, `base` the layout offset
    /// that lands at `dst`.
    Disk {
        source: Arc<Source>,
        items: Vec<Item>,
        base: u64,
    },
}
unsafe impl Send for CopySrc {}

pub struct Copied {
    pub ready: EvRef,
    pub start: EvRef,
    pub bytes: u64,
    pub from_host: bool,
}

pub struct CopyJob {
    pub waits: Vec<EvRef>,
    pub dst: u64,
    pub len: u64,
    pub src: CopySrc,
    pub done: Box<dyn FnOnce(Result<Copied>) + Send>,
}

struct Bounce {
    ptr: *mut u8,
    last: Option<EvRef>,
}

/// One copy stream and the thread that feeds it, in submission order.
pub struct Mover {
    tx: Mutex<Option<Sender<CopyJob>>>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

pub struct MoverCtx {
    pub ctx: CUcontext,
    pub events: Arc<EventPool>,
    pub readers: Arc<Readers>,
    pub tally: Arc<IoTally>,
    pub staging: (usize, u64),
}
unsafe impl Send for MoverCtx {}

impl Mover {
    pub fn start(name: String, m: MoverCtx) -> Result<Mover> {
        let (tx, rx) = channel::<CopyJob>();
        let (ok_tx, ok_rx) = channel::<Result<()>>();
        let thread = std::thread::Builder::new()
            .name(name)
            .spawn(move || mover_main(m, rx, ok_tx))
            .map_err(|e| Error::Io(format!("spawn mover: {e}")))?;
        ok_rx
            .recv()
            .map_err(|_| Error::Io("mover died at start".into()))??;
        Ok(Mover {
            tx: Mutex::new(Some(tx)),
            thread: Mutex::new(Some(thread)),
        })
    }

    /// Queue a copy. On `Err` the job was dropped without calling `done`.
    pub fn submit(&self, job: CopyJob) -> Result<()> {
        match self.tx.lock().unwrap().as_ref() {
            Some(tx) => tx.send(job).map_err(|_| Error::Closed),
            None => Err(Error::Closed),
        }
    }

    /// Close the queue: the thread drains its stream, frees its buffers and exits by itself.
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

fn mover_main(m: MoverCtx, rx: Receiver<CopyJob>, ok: Sender<Result<()>>) {
    let init = (|| -> Result<(CUstream, Vec<Bounce>)> {
        let d = cuda::driver()?;
        cuda::check("cuCtxSetCurrent", unsafe { (d.ctx_set_current)(m.ctx) })?;
        let mut s: CUstream = std::ptr::null_mut();
        // SAFETY: out-param, context bound.
        cuda::check("cuStreamCreate", unsafe { (d.stream_create)(&mut s, cuda::STREAM_NON_BLOCKING) })?;
        let mut b = Vec::new();
        for _ in 0..m.staging.0 {
            let mut p: *mut libc::c_void = std::ptr::null_mut();
            // SAFETY: out-param; pinned, portable host memory.
            cuda::check("cuMemHostAlloc", unsafe {
                (d.mem_host_alloc)(&mut p, m.staging.1 as usize, cuda::MEMHOSTALLOC_PORTABLE)
            })?;
            b.push(Bounce { ptr: p as *mut u8, last: None });
        }
        Ok((s, b))
    })();
    let (stream, mut bounces) = match init {
        Ok(x) => {
            let _ = ok.send(Ok(()));
            x
        }
        Err(e) => {
            let _ = ok.send(Err(e));
            return;
        }
    };
    while let Ok(job) = rx.recv() {
        let done = job.done;
        let r = run_copy(&m, stream, &mut bounces, job.waits, job.dst, job.len, job.src);
        done(r);
    }
    if let Ok(d) = cuda::driver() {
        // SAFETY: our stream and buffers; the stream is drained first.
        unsafe {
            (d.stream_synchronize)(stream);
            (d.stream_destroy)(stream);
            for b in bounces {
                (d.mem_free_host)(b.ptr as *mut libc::c_void);
            }
        }
    }
}

struct Waiter {
    m: Mutex<Option<Result<()>>>,
    cv: Condvar,
}

/// One job on the mover's stream. `Err` means nothing this job enqueued is still in flight and
/// no reader still writes a bounce: a failed copy is drained before it is reported, so no
/// caller unmaps, unregisters or reuses a range a copy still touches. A drain that itself fails
/// is ambiguous device state: `Poisoned`.
fn run_copy(
    m: &MoverCtx,
    stream: CUstream,
    bounces: &mut [Bounce],
    waits: Vec<EvRef>,
    dst: u64,
    len: u64,
    src: CopySrc,
) -> Result<Copied> {
    let r = enqueue_copy(m, stream, bounces, waits, dst, len, src);
    let Err(e) = r else { return r };
    let d = cuda::driver()?;
    // SAFETY: our own stream.
    let drained = crate::faults::hit("drain").and_then(|_| {
        cuda::check("cuStreamSynchronize", unsafe { (d.stream_synchronize)(stream) })
    });
    match drained {
        Ok(()) => Err(e),
        Err(drain) => Err(Error::Poisoned(format!("a failed copy ({e}) could not be drained: {drain}"))),
    }
}

fn enqueue_copy(
    m: &MoverCtx,
    stream: CUstream,
    bounces: &mut [Bounce],
    waits: Vec<EvRef>,
    dst: u64,
    len: u64,
    src: CopySrc,
) -> Result<Copied> {
    let d = cuda::driver()?;
    let start = m.events.get(true)?;
    let ready = m.events.get(true)?;
    let wait_all = |waits: &[EvRef]| -> Result<()> {
        for w in waits {
            // SAFETY: events of this context; the stream is ours.
            cuda::check("cuStreamWaitEvent", unsafe { (d.stream_wait_event)(stream, w.raw(), 0) })?;
        }
        Ok(())
    };
    match src {
        CopySrc::Host(p) => {
            wait_all(&waits)?;
            crate::faults::gate("copy", stream)?;
            start.record(stream)?;
            // SAFETY: `p` covers `len` bytes of the pinned tier, kept alive by the books.
            cuda::check("cuMemcpyHtoDAsync", unsafe {
                (d.memcpy_htod_async)(dst, p as *const libc::c_void, len as usize, stream)
            })?;
            crate::faults::hit("host-copy-recorded")?;
            ready.record(stream)?;
            Ok(Copied { ready, start, bytes: len, from_host: true })
        }
        CopySrc::Disk { source, items, base } => {
            let cap = m.staging.1;
            if bounces.is_empty() || cap == 0 {
                return Err(Error::Invalid("disk-sourced device fill with no staging buffers".into()));
            }
            let nchunks = len.div_ceil(cap) as usize;
            let mut inflight: VecDeque<(u64, usize, Arc<Waiter>)> = VecDeque::new();
            let mut next = 0usize;
            // The slot's previous readers first: the stream waits, the host keeps reading.
            let mut failed = wait_all(&waits).and_then(|_| crate::faults::gate("copy", stream)).err();
            let mut first = true;
            loop {
                while failed.is_none() && inflight.len() < bounces.len() && next < nchunks {
                    let b = next % bounces.len();
                    if let Some(last) = bounces[b].last.take() {
                        if let Err(e) = last.sync() {
                            failed = Some(e);
                            break;
                        }
                    }
                    let c0 = next as u64 * cap;
                    let c1 = (c0 + cap).min(len);
                    let pieces = slice_items(&items, base + c0, base + c1);
                    let w = Arc::new(Waiter { m: Mutex::new(None), cv: Condvar::new() });
                    let w2 = w.clone();
                    let batch = Batch::new(pieces.len(), move |r| {
                        *w2.m.lock().unwrap() = Some(r);
                        w2.cv.notify_all();
                    });
                    for it in pieces {
                        let into = Dst(unsafe { bounces[b].ptr.add((it.offset - base - c0) as usize) });
                        let (source, batch, tally) = (source.clone(), batch.clone(), m.tally.clone());
                        m.readers.submit(true, move || {
                            let into = into;
                            // SAFETY: the bounce range belongs to this chunk until its copy
                            // event, recorded only after this batch completes.
                            let r = unsafe { source.read_item(&it, into.0, Mode::Buffered, &tally) };
                            // Drop the source before signalling: whoever waits on the batch may
                            // then close the last weight set and end the lease at once.
                            drop(source);
                            batch.finish(r);
                        });
                    }
                    inflight.push_back((c0, b, w));
                    next += 1;
                }
                let Some((c0, b, w)) = inflight.pop_front() else { break };
                let mut g = w.m.lock().unwrap();
                while g.is_none() {
                    g = w.cv.wait(g).unwrap();
                }
                // After any failure keep draining: readers may still write the bounces.
                if let Err(e) = g.take().unwrap() {
                    failed.get_or_insert(e);
                }
                if failed.is_some() {
                    continue;
                }
                let mut copy = || -> Result<()> {
                    if first {
                        start.record(stream)?;
                        first = false;
                    }
                    crate::faults::hit("staged-copy")?;
                    let n = cap.min(len - c0);
                    // SAFETY: the bounce holds the chunk; pinned host memory.
                    cuda::check("cuMemcpyHtoDAsync", unsafe {
                        (d.memcpy_htod_async)(dst + c0, bounces[b].ptr as *const libc::c_void, n as usize, stream)
                    })?;
                    let e = m.events.get(false)?;
                    e.record(stream)?;
                    bounces[b].last = Some(e);
                    Ok(())
                };
                if let Err(e) = copy() {
                    failed = Some(e);
                }
            }
            if let Some(e) = failed {
                return Err(e);
            }
            if first {
                start.record(stream)?;
            }
            ready.record(stream)?;
            Ok(Copied { ready, start, bytes: len, from_host: false })
        }
    }
}

/// The parts of `items` inside layout range `[lo, hi)`, split at the edges. Object ranges are
/// split at the same offsets; inline bytes are sliced.
pub fn slice_items(items: &[Item], lo: u64, hi: u64) -> Vec<Item> {
    let mut out = Vec::new();
    for it in items {
        let (s, e) = (it.offset.max(lo), (it.offset + it.len).min(hi));
        if s >= e {
            continue;
        }
        let skip = s - it.offset;
        let source = match &it.source {
            ItemSource::Object(o) => ItemSource::Object(tensorfs_core::read::ObjectRange {
                obj: o.obj.clone(),
                off: o.off + skip,
                len: e - s,
            }),
            ItemSource::Inline(b) => ItemSource::Inline(b[skip as usize..(skip + e - s) as usize].into()),
        };
        out.push(Item {
            offset: s,
            len: e - s,
            source,
            part_end: it.part_end && e == it.offset + it.len,
        });
    }
    out
}

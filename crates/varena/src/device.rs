//! One GPU: its primary context, event pool, VMM arena ledger and copy movers.
//! These operations run in the caller's process. In Cozy that caller is Runtime's
//! executor, using TensorD's memory grants; TensorD may retain exported allocation fds
//! for reuse while Runtime performs the imports, mappings and tensor construction.
//!
//! The arena is varena's core with two changes: a region is backed by slab-sized physical
//! handles (the caller's `slab_bytes`, or one handle per region) instead of one per 2 MiB, and
//! handles are POSIX-fd exportable where the device supports it. Budget-fed, recycle before
//! create, refuse before mutation, else roll back exactly, else poison.

use std::collections::{BTreeMap, VecDeque};
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use crate::cuda::{self, AccessDesc, AllocationProp, CUcontext, CUevent, CUstream, CtxGuard, Handle};
use crate::io::{Dst, IoTally, Mode, Readers, Source};
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

/// Change `device`'s access to a mapped span: shared bytes become read-only for their filler.
pub fn set_access(va: u64, len: u64, device: i32, access: u32) -> Result<()> {
    let d = cuda::driver()?;
    let desc = AccessDesc {
        location: cuda::MemLocation {
            kind: cuda::MEM_LOCATION_TYPE_DEVICE,
            id: device,
        },
        flags: access,
    };
    // SAFETY: a span we mapped; no use is in flight (the caller checked its leases).
    cuda::check("cuMemSetAccess", unsafe { (d.mem_set_access)(va, len as usize, &desc, 1) })
}

/// The driver can export and import device memory as POSIX fds.
pub fn sharing_symbols() -> bool {
    cuda::driver().is_ok_and(|d| {
        d.mem_export_to_shareable_handle.is_some()
            && d.mem_import_from_shareable_handle.is_some()
            && d.mem_get_allocation_properties.is_some()
    })
}

/// A close-on-exec POSIX fd naming `h`'s physical memory. Every fd, imported handle and
/// mapping is a reference: the memory lives until the last one is gone, in any process.
pub fn export_fd(h: Handle) -> Result<OwnedFd> {
    let d = cuda::driver()?;
    let export = d
        .mem_export_to_shareable_handle
        .ok_or_else(|| Error::CudaUnavailable("cuMemExportToShareableHandle".into()))?;
    let mut raw: i32 = -1;
    // SAFETY: a POSIX-fd handle is one int written through the out-pointer; flags are 0.
    cuda::check("cuMemExportToShareableHandle", unsafe {
        export(&mut raw as *mut i32 as *mut libc::c_void, h, cuda::MEM_HANDLE_TYPE_POSIX_FD, 0)
    })?;
    if raw < 0 {
        return Err(Error::Invalid("the driver exported no fd".into()));
    }
    // SAFETY: the export handed us this new fd exactly once.
    let fd = unsafe { OwnedFd::from_raw_fd(raw) };
    // SAFETY: fcntl on our own fd.
    let flags = unsafe { libc::fcntl(raw, libc::F_GETFD) };
    if flags < 0 || unsafe { libc::fcntl(raw, libc::F_SETFD, flags | libc::FD_CLOEXEC) } < 0 {
        return Err(Error::Io(format!("cloexec on an exported fd: {}", std::io::Error::last_os_error())));
    }
    Ok(fd)
}

/// Import a shared fd as a handle of pinned memory on `dev` (this process's device numbering).
/// The fd stays the caller's; the handle is a separate reference the caller must release.
pub fn import_fd(fd: BorrowedFd<'_>, dev: cuda::CUdevice) -> Result<Handle> {
    let d = cuda::driver()?;
    let (import, props) = match (d.mem_import_from_shareable_handle, d.mem_get_allocation_properties) {
        (Some(i), Some(p)) => (i, p),
        _ => return Err(Error::CudaUnavailable("cuMemImportFromShareableHandle".into())),
    };
    let mut h: Handle = 0;
    // SAFETY: a POSIX fd travels as the fd number cast to a pointer, not a pointer to it.
    cuda::check("cuMemImportFromShareableHandle", unsafe {
        import(&mut h, fd.as_raw_fd() as usize as *mut libc::c_void, cuda::MEM_HANDLE_TYPE_POSIX_FD)
    })?;
    let mut p = AllocationProp::device(0, false);
    // SAFETY: a live imported handle and a correctly sized out-record.
    let rc = unsafe { props(&mut p, h) };
    let wrong = rc != cuda::SUCCESS
        || p.kind != cuda::MEM_ALLOCATION_TYPE_PINNED
        || p.location.kind != cuda::MEM_LOCATION_TYPE_DEVICE
        || p.location.id != dev;
    if wrong {
        let _ = release(h);
        cuda::check("cuMemGetAllocationPropertiesFromHandle", rc)?;
        return Err(Error::Invalid(format!(
            "shared memory is not pinned memory of this device (kind {}, location {}:{})",
            p.kind, p.location.kind, p.location.id
        )));
    }
    Ok(h)
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
    /// Page-locked host bytes (a registered region): one DMA.
    Host(*const u8),
    /// Host bytes this process could not page-lock: copied through the staging buffers, never
    /// as pageable memory (a pageable copy runs at a fraction of the link and stalls the mover).
    Bounce(*const u8),
    /// A sealed layout's region outside the pinned books, once its filler marks it Ready: the
    /// mover page-locks it on its first copy (`HostMem::pin_region`; the layout is the machine's
    /// RAM, so no budget is charged), else bounces it. This reference keeps the layout mapped.
    Sealed { host: Arc<crate::host::HostMem>, region: u32, offset: u64 },
    /// Disk through the staging buffers: `items` in layout offsets, `base` the layout offset
    /// that lands at `dst`.
    Disk {
        source: Arc<Source>,
        items: Vec<Item>,
        base: u64,
    },
}
unsafe impl Send for CopySrc {}

/// Where a copy's bytes came from, for the facts: one DMA from page-locked host memory, host
/// memory through the staging buffers, or the object files through them.
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
    free: Mutex<VecDeque<Slot>>,
    cv: Condvar,
}

impl Slots {
    /// A free slot whose last copy has landed: one already past it if any, else the one
    /// returned longest ago, waited for.
    fn take(&self) -> Result<Slot> {
        let mut free = self.free.lock().unwrap();
        while free.is_empty() {
            free = self.cv.wait(free).unwrap();
        }
        let landed = free.iter().position(|s| s.last.as_ref().is_none_or(|e| e.done().unwrap_or(false)));
        let mut slot = free.remove(landed.unwrap_or(0)).unwrap();
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
        self.free.lock().unwrap().push_back(slot);
        self.cv.notify_one();
    }
}

#[derive(Clone, Copy)]
struct Stream(CUstream);
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

type Done = Box<dyn FnOnce(Result<Copied>) + Send>;

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
    _kept: Option<Arc<crate::host::HostMem>>,
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
                Ok(()) => Ok(Copied { ready: self.ready.clone(), start: self.start.clone(), bytes: self.bytes, path: self.path, refused: self.refused }),
                Err(e) => drained(self.stream.0, e),
            },
        };
        if let Some(done) = self.done.lock().unwrap().take() {
            done(res);
        }
    }
}

/// A failure reported only once nothing it queued is in flight: no caller unmaps, unregisters or
/// reuses a range a copy still touches. A drain that itself fails is ambiguous device state.
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
    pub tally: Arc<IoTally>,
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
    let n = m.staging.0.max(1);
    let init = (|| -> Result<(CUstream, VecDeque<Slot>)> {
        let d = cuda::driver()?;
        cuda::check("cuCtxSetCurrent", unsafe { (d.ctx_set_current)(m.ctx) })?;
        let mut s: CUstream = std::ptr::null_mut();
        // SAFETY: out-param, context bound.
        cuda::check("cuStreamCreate", unsafe { (d.stream_create)(&mut s, cuda::STREAM_NON_BLOCKING) })?;
        let mut slots = VecDeque::new();
        for _ in 0..n {
            let mut p: *mut libc::c_void = std::ptr::null_mut();
            // SAFETY: out-param; pinned, portable host memory (page aligned).
            cuda::check("cuMemHostAlloc", unsafe { (d.mem_host_alloc)(&mut p, m.staging.1 as usize, cuda::MEMHOSTALLOC_PORTABLE) })?;
            slots.push_back(Slot { ptr: p as *mut u8, last: None });
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
    while free.len() < n {
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

/// Copy one slot's bytes to `dst` on the stream and mark the slot busy until they land.
fn copy_slot(m: &MoverCtx, stream: Stream, slot: &mut Slot, dst: u64, n: u64) -> Result<()> {
    crate::faults::hit("staged-copy")?;
    let d = cuda::driver()?;
    // SAFETY: the slot holds the bytes; pinned host memory; the destination is mapped.
    cuda::check("cuMemcpyHtoDAsync", unsafe { (d.memcpy_htod_async)(dst, slot.ptr as *const libc::c_void, n as usize, stream.0) })?;
    let e = m.events.get(false)?;
    e.record(stream.0)?;
    slot.last = Some(e);
    Ok(())
}

/// The object items of `[lo, hi)` (layout offsets), adjacent ranges of one object merged into
/// one read where the layout keeps their spacing (a chunk's 4 KiB padding reads along).
fn coalesced(items: &[Item], lo: u64, hi: u64) -> Vec<Item> {
    let mut out: Vec<Item> = Vec::new();
    for it in slice_items(items, lo, hi) {
        if let (Some(prev), ItemSource::Object(o)) = (out.last_mut(), &it.source) {
            if let ItemSource::Object(p) = &mut prev.source {
                if p.obj == o.obj && o.off >= p.off && it.offset >= prev.offset && o.off - p.off == it.offset - prev.offset {
                    let end = it.offset + it.len;
                    p.len = end - prev.offset;
                    prev.len = end - prev.offset;
                    prev.part_end = it.part_end;
                    continue;
                }
            }
        }
        out.push(it);
    }
    out
}

/// One slot's worth of a staged job: its bytes `[at, at + n)` of the copy.
struct Piece {
    at: u64,
    n: u64,
}

fn dispatch(m: &Arc<MoverCtx>, stream: Stream, slots: &Arc<Slots>, job: CopyJob) {
    let CopyJob { waits, dst, len, src, done } = job;
    let s = stream.0;
    // Never a byte a sealed layout's machine has not finished: wait for the region's Ready word.
    let (src, kept, refused) = match src {
        // A streamed layout's slot is refilled once this copy lets its region go: its bytes are
        // in the staging slots before the claim ends, so they are bounced here, in order.
        CopySrc::Sealed { host, region, .. } if host.window().is_some() => {
            let copied = host.clone().with_region(region, |p| {
                let begun = begin(m, s, &waits)?;
                for at in (0..len).step_by(m.staging.1 as usize) {
                    let n = m.staging.1.min(len - at);
                    let mut slot = slots.take()?;
                    // SAFETY: the slot is ours; the region stays in its slot while we claim it.
                    unsafe { std::ptr::copy_nonoverlapping(p.add(at as usize), slot.ptr, n as usize) };
                    let r = copy_slot(m, stream, &mut slot, dst + at, n);
                    slots.put(slot);
                    r?;
                }
                begun.1.record(s)?;
                Ok(begun)
            });
            return done(match copied {
                Ok((start, ready)) => Ok(Copied { ready, start, bytes: len, path: CopyPath::Bounced, refused: false }),
                Err(e) => drained(s, e),
            });
        }
        CopySrc::Sealed { host, region, offset } => {
            if let Err(e) = host.wait_ready(region) {
                return done(Err(e));
            }
            let span = crate::layout::align_up(len, crate::layout::PART_ALIGN);
            let (pinned, now) = host.pin_region(region, offset, span);
            // SAFETY: inside the live mapping `host` keeps; the region is complete.
            let p = unsafe { host.base().add(offset as usize) } as *const u8;
            (if pinned { CopySrc::Host(p) } else { CopySrc::Bounce(p) }, Some(host), now)
        }
        src => (src, None, false),
    };
    let (start, ready) = match begin(m, s, &waits) {
        Ok(x) => x,
        Err(e) => return done(drained(s, e)),
    };
    let size = m.staging.1;
    let pieces = (0..len.div_ceil(size)).map(|i| Piece { at: i * size, n: size.min(len - i * size) });
    let (disk, from, path) = match src {
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
        CopySrc::Bounce(p) => (None, Some(Dst(p as *mut u8)), CopyPath::Bounced),
        CopySrc::Disk { source, items, base } => (Some((source, Arc::new(items), base)), None, CopyPath::Files),
        CopySrc::Sealed { .. } => unreachable!("resolved above"),
    };
    let pieces: Vec<Piece> = pieces.collect();
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
        let (m, job, slots, disk) = (m.clone(), job.clone(), slots.clone(), disk.clone());
        m.readers.clone().submit(true, move || {
            // Whole values, not their raw-pointer fields (closures capture fields precisely).
            let (m, mut slot, stream, from) = (m, slot, stream, from);
            let filled = match (&disk, from) {
                (Some((source, items, base)), _) => coalesced(items, base + piece.at, base + piece.at + piece.n).iter().try_for_each(|it| {
                    // SAFETY: the slot is this piece's alone until it is returned below; an
                    // O_DIRECT read may run to the next 4 KiB boundary, inside the slot.
                    unsafe { source.read_item(it, slot.ptr.add((it.offset - base - piece.at) as usize), Mode::Cache, &m.tally) }
                }),
                (None, Some(p)) => {
                    // SAFETY: the host range stays mapped (the books, or `_kept`) until done.
                    unsafe { std::ptr::copy_nonoverlapping(p.0.add(piece.at as usize), slot.ptr, piece.n as usize) };
                    Ok(())
                }
                (None, None) => unreachable!("a staged job reads files or host memory"),
            };
            let copied = filled.and_then(|_| {
                if job.failed() {
                    return Ok(());
                }
                let _g = CtxGuard::enter(m.ctx)?;
                copy_slot(&m, stream, &mut slot, dst + piece.at, piece.n)
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

/// Enqueue `waits` on the stream, then a start event; returns (start, ready to record later).
fn begin(m: &MoverCtx, s: CUstream, waits: &[EvRef]) -> Result<(EvRef, EvRef)> {
    let d = cuda::driver()?;
    for w in waits {
        // SAFETY: events of this context; the stream is ours.
        cuda::check("cuStreamWaitEvent", unsafe { (d.stream_wait_event)(s, w.raw(), 0) })?;
    }
    crate::faults::gate("copy", s)?;
    let start = m.events.get(true)?;
    start.record(s)?;
    Ok((start, m.events.get(true)?))
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
            ItemSource::Object(o) => ItemSource::Object(tensorfs::read::ObjectRange {
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

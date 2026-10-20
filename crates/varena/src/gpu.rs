//! GPU placement mechanism. The policy is in `plan`; all byte transfers are TensorFS reads.
//! A mapping is writable only while filling. Pins and CUDA completion events protect it
//! until every consumer has finished; neither an error nor a dropped Python handle is proof
//! that queued GPU work has stopped.
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{c_int, c_void};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::{self, ThreadId};

use crossbeam_queue::SegQueue;
use parking_lot::{Condvar, Mutex};
use tensorfs::io::cuda::{self, AccessDesc, AllocationProp, CtxGuard, Handle};
use tensorfs::io::source::{IoTally, Item, Mode, Source};
use tensorfs::io::{self, Dst, ReadStats};

use crate::{Error, Result};

type PageId = u64;

macro_rules! call {
    ($op:literal, $field:ident($($arg:expr),* $(,)?)) => {
        cuda::check($op, unsafe { (cuda::driver()?.$field)($($arg),*) }).map_err(Error::from)
    };
}

fn invalid(reason: &str) -> Error {
    Error::Invalid(reason.to_owned())
}

struct Device {
    ordinal: i32,
    context: usize,
    prop: AllocationProp,
    granularity: u64,
    stream: cuda::Stream,
}

// CUDA handles are opaque. Every use enters the retained context; no Rust pointer is read.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    fn enter(&self) -> Result<CtxGuard> {
        Ok(CtxGuard::enter(self.context as cuda::CUcontext)?)
    }

    fn access(&self, ptr: u64, bytes: u64, readonly: bool) -> Result<()> {
        let desc = AccessDesc {
            location: self.prop.location,
            flags: if readonly {
                cuda::MEM_ACCESS_PROT_READ
            } else {
                cuda::MEM_ACCESS_PROT_READWRITE
            },
        };
        call!(
            "cuMemSetAccess",
            mem_set_access(ptr, bytes as usize, &desc, 1)
        )
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        if let Ok(driver) = cuda::driver() {
            unsafe {
                (driver.primary_ctx_release)(self.prop.location.id);
            }
        }
    }
}

/// One stable virtual address range for a weight set. Pages can be unmapped independently.
pub struct Reservation {
    device: Arc<Device>,
    pub ptr: u64,
    pub bytes: u64,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if let (Ok(_context), Ok(driver)) = (self.device.enter(), cuda::driver()) {
            unsafe {
                (driver.mem_address_free)(self.ptr, self.bytes as usize);
            }
        }
    }
}

struct Event {
    device: Arc<Device>,
    raw: usize,
}

impl Event {
    fn record(device: &Arc<Device>, stream: usize) -> Result<Self> {
        let _context = device.enter()?;
        let mut raw = std::ptr::null_mut();
        call!(
            "cuEventCreate",
            event_create(&mut raw, cuda::EVENT_DISABLE_TIMING)
        )?;
        let event = Self {
            device: device.clone(),
            raw: raw as usize,
        };
        call!("cuEventRecord", event_record(raw, stream as cuda::CUstream))?;
        Ok(event)
    }

    fn done(&self) -> Result<bool> {
        let _context = self.device.enter()?;
        match unsafe { (cuda::driver()?.event_query)(self.raw as cuda::CUevent) } {
            cuda::SUCCESS => Ok(true),
            cuda::ERROR_NOT_READY => Ok(false),
            code => Err(cuda::check("cuEventQuery", code)
                .err()
                .map(Error::from)
                .unwrap_or_else(|| invalid("unexpected event result"))),
        }
    }

    fn wait(&self, stream: usize) -> Result<()> {
        let _context = self.device.enter()?;
        call!(
            "cuStreamWaitEvent",
            stream_wait_event(stream as cuda::CUstream, self.raw as cuda::CUevent, 0)
        )
    }

    fn synchronize(&self) -> Result<()> {
        let _context = self.device.enter()?;
        call!(
            "cuEventSynchronize",
            event_synchronize(self.raw as cuda::CUevent)
        )
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if let (Ok(_context), Ok(driver)) = (self.device.enter(), cuda::driver()) {
            unsafe {
                (driver.event_destroy)(self.raw as cuda::CUevent);
            }
        }
    }
}

struct Chunk {
    handle: Handle,
    bytes: u64,
}

struct Mapping {
    range: Arc<Reservation>,
    offset: u64,
    bytes: u64,
    chunks: Vec<Chunk>,
    ready: Option<Arc<Event>>,
    releases: Vec<Arc<Event>>,
    pins: usize,
    kept: Option<UnixStream>,
}

impl Mapping {
    fn ptr(&self) -> u64 {
        self.range.ptr + self.offset
    }

    fn allocate(range: Arc<Reservation>, offset: u64, bytes: u64) -> Result<Self> {
        let device = range.device.clone();
        let _context = device.enter()?;
        let mut out = Self {
            range,
            offset,
            bytes: 0,
            chunks: vec![],
            ready: None,
            releases: vec![],
            pins: 0,
            kept: None,
        };
        while out.bytes < bytes {
            let remaining = bytes - out.bytes;
            let mut size = remaining.min((64 << 20).max(device.granularity));
            size -= size % device.granularity;
            let mut handle = 0;
            loop {
                let code = unsafe {
                    (cuda::driver()?.mem_create)(&mut handle, size as usize, &device.prop, 0)
                };
                if code == cuda::SUCCESS {
                    break;
                }
                if code == cuda::ERROR_OUT_OF_MEMORY && size > device.granularity {
                    size = device.granularity;
                    continue;
                }
                if code == cuda::ERROR_OUT_OF_MEMORY {
                    return Err(Error::DeviceFull {
                        need: bytes,
                        deficit: remaining,
                    });
                }
                cuda::check("cuMemCreate", code)?;
            }
            if let Err(error) = call!(
                "cuMemMap",
                mem_map(out.ptr() + out.bytes, size as usize, 0, handle, 0)
            ) {
                let _ = call!("cuMemRelease", mem_release(handle));
                return Err(error);
            }
            out.chunks.push(Chunk {
                handle,
                bytes: size,
            });
            out.bytes += size;
        }
        device.access(out.ptr(), bytes, false)?;
        Ok(out)
    }

    fn idle(&mut self) -> Result<bool> {
        if self.pins != 0 {
            return Ok(false);
        }
        if let Some(event) = &self.ready {
            if !event.done()? {
                return Ok(false);
            }
        }
        let mut pending = Vec::new();
        while let Some(event) = self.releases.pop() {
            match event.done() {
                Ok(true) => {}
                Ok(false) => pending.push(event),
                Err(error) => {
                    pending.push(event);
                    self.releases.extend(pending);
                    return Err(error);
                }
            }
        }
        self.releases = pending;
        Ok(self.releases.is_empty())
    }

    /// Release while retaining all owners on an error. The caller may retry after drain.
    fn release(&mut self, discard_custody: bool) -> Result<()> {
        let device = self.range.device.clone();
        let _context = device.enter()?;
        if self.bytes != 0 {
            call!("cuMemUnmap", mem_unmap(self.ptr(), self.bytes as usize))?;
            self.bytes = 0;
        }
        while let Some(chunk) = self.chunks.last() {
            call!("cuMemRelease", mem_release(chunk.handle))?;
            self.chunks.pop();
        }
        if let Some(container) = self.kept.take().filter(|_| discard_custody) {
            // Nonblocking: a new executor may already have consumed this queued handle.
            let _ = container.set_nonblocking(true);
            while platform::handles::recv_fd(&container).is_ok() {}
        }
        Ok(())
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // A failed CUDA wait must never allow a borrowed source or destination to be reused.
        // Sticky faults end the executor; retaining its allocation is safer than early release.
        if self.bytes == 0 && self.chunks.is_empty() {
            return;
        }
        let safe = self
            .range
            .device
            .enter()
            .and_then(|_context| call!("cuCtxSynchronize", ctx_synchronize()))
            .is_ok();
        if safe {
            let _ = self.release(false);
        }
        if self.bytes != 0 || !self.chunks.is_empty() {
            std::mem::forget(self.range.clone());
        }
    }
}

enum State {
    Empty,
    Filling(ThreadId),
    Ready(Mapping),
    Failed(Mapping, Error),
}

struct Page {
    range: Arc<Reservation>,
    offset: u64,
    bytes: u64,
    source: Arc<Source>,
    items: Vec<Item>,
    origin: u64,
    fingerprint: String,
    state: Mutex<State>,
    retired: AtomicBool,
    write_pins: AtomicUsize,
    // Live owners are distinct from returned leases whose release is queued or
    // being processed by another poller. Both remain protected by write_pins.
    write_live: AtomicUsize,
    write_events: Mutex<Vec<Event>>,
    host_sources: Mutex<Vec<Arc<dyn Send + Sync>>>,
}

struct Release {
    page: Arc<Page>,
    streams: BTreeSet<usize>,
    writable: bool,
}

/// An independently retained fence, or a fill's progress generation. Its wait
/// is separate so placement can publish dequeued pin facts before blocking.
pub(crate) struct Pending {
    event: Option<Arc<Event>>,
    fill_generation: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FillPhase {
    Allocate,
    Read,
    Transform,
    Finish,
}
impl FillPhase {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Allocate => "allocate",
            Self::Read => "read",
            Self::Transform => "transform",
            Self::Finish => "finish",
        }
    }
}
pub(crate) struct FillAttempt {
    pub result: Result<ReadStats>,
    pub phase: FillPhase,
}

/// Actual mapping readiness, without turning a transient state into INVALID.
pub(crate) enum BindState {
    Ready { pin: Pin, pins: usize },
    Empty,
    Filling(Pending),
}

/// Holds mappings, never chooses victims. The policy invokes fill/unmap with page IDs.
pub struct Gpu {
    device: Arc<Device>,
    pages: Mutex<BTreeMap<PageId, Arc<Page>>>,
    releases: SegQueue<Release>,
    generation: Mutex<u64>,
    changed: Condvar,
    wakeup: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
    #[cfg(test)]
    write_waiters: AtomicUsize,
}

impl Gpu {
    pub fn open(ordinal: i32) -> Result<Arc<Self>> {
        let (physical, context) = cuda::primary_context(ordinal)?;
        let prop = AllocationProp::device(physical, cuda::MEM_HANDLE_TYPE_POSIX_FD);
        let setup = (|| {
            let _context = CtxGuard::enter(context)?;
            let mut granularity = 0usize;
            call!(
                "cuMemGetAllocationGranularity",
                mem_get_allocation_granularity(
                    &mut granularity,
                    &prop,
                    cuda::MEM_GRANULARITY_MINIMUM
                )
            )?;
            if granularity == 0 {
                return Err(invalid("zero VMM granularity"));
            }
            let stream = cuda::Stream::new(ordinal)?;
            Ok(Self {
                device: Arc::new(Device {
                    ordinal,
                    context: context as usize,
                    prop,
                    granularity: granularity as u64,
                    stream,
                }),
                pages: Mutex::new(BTreeMap::new()),
                releases: SegQueue::new(),
                generation: Mutex::new(0),
                changed: Condvar::new(),
                wakeup: Mutex::new(None),
                #[cfg(test)]
                write_waiters: AtomicUsize::new(0),
            })
        })();
        if setup.is_err() {
            if let Ok(driver) = cuda::driver() {
                unsafe {
                    (driver.primary_ctx_release)(physical);
                }
            }
        }
        setup.map(Arc::new)
    }

    pub fn device(&self) -> i32 {
        self.device.ordinal
    }

    pub fn uuid(&self) -> Result<String> {
        Ok(cuda::uuid(self.device.prop.location.id)?)
    }

    pub fn capacity(&self) -> Result<u64> {
        let _context = self.device.enter()?;
        let (mut free, mut total) = (0, 0);
        call!("cuMemGetInfo", mem_get_info(&mut free, &mut total))?;
        Ok(total as u64)
    }

    pub fn reserve(&self, bytes: u64) -> Result<Arc<Reservation>> {
        let g = self.device.granularity;
        let bytes = bytes
            .checked_add(g - 1)
            .map(|n| n / g * g)
            .ok_or_else(|| invalid("address range overflows"))?;
        if bytes == 0 || bytes > usize::MAX as u64 {
            return Err(invalid("invalid address range"));
        }
        let _context = self.device.enter()?;
        let mut ptr = 0;
        call!(
            "cuMemAddressReserve",
            mem_address_reserve(&mut ptr, bytes as usize, g as usize, 0, 0)
        )?;
        Ok(Arc::new(Reservation {
            device: self.device.clone(),
            ptr,
            bytes,
        }))
    }

    /// Register already-described byte ranges. Layout calculation belongs to TensorFS.
    #[allow(clippy::too_many_arguments)]
    pub fn register(
        &self,
        id: PageId,
        range: Arc<Reservation>,
        offset: u64,
        bytes: u64,
        source: Arc<Source>,
        items: Vec<Item>,
        origin: u64,
        fingerprint: String,
    ) -> Result<()> {
        let g = self.device.granularity;
        if !Arc::ptr_eq(&range.device, &self.device)
            || bytes == 0
            || !offset.is_multiple_of(g)
            || !bytes.is_multiple_of(g)
            || offset
                .checked_add(bytes)
                .is_none_or(|end| end > range.bytes)
        {
            return Err(invalid(
                "page is outside its device reservation or unaligned",
            ));
        }
        let mut pages = self.pages.lock();
        if pages.contains_key(&id) {
            return Err(invalid("page already registered"));
        }
        if pages.values().any(|page| {
            Arc::ptr_eq(&page.range, &range)
                && offset < page.offset + page.bytes
                && page.offset < offset + bytes
        }) {
            return Err(invalid("page overlaps a registered mapping"));
        }
        pages.insert(
            id,
            Arc::new(Page {
                range,
                offset,
                bytes,
                source,
                items,
                origin,
                fingerprint,
                state: Mutex::new(State::Empty),
                retired: AtomicBool::new(false),
                write_pins: AtomicUsize::new(0),
                write_live: AtomicUsize::new(0),
                write_events: Mutex::new(Vec::new()),
                host_sources: Mutex::new(Vec::new()),
            }),
        );
        Ok(())
    }

    fn page(&self, id: PageId) -> Result<Arc<Page>> {
        self.pages
            .lock()
            .get(&id)
            .cloned()
            .ok_or_else(|| invalid("unknown page"))
    }

    /// Called on an I/O worker, outside the policy lock. A concurrent bind sees Filling.
    pub fn fill(self: &Arc<Self>, id: PageId) -> Result<ReadStats> {
        self.fill_attempt_bytes(id, None, None, |_| Ok(())).result
    }

    /// The RAM-tier lease must remain alive for this call. TensorFS still performs
    /// the transfer; the mechanism only selects a ready source and its destination.
    pub fn fill_from_ram(self: &Arc<Self>, id: PageId, bytes: &[u8]) -> Result<ReadStats> {
        self.fill_attempt_bytes(id, Some(bytes), None, |_| Ok(()))
            .result
    }

    /// A declared page-in transform runs only on a new, private writable mapping.
    /// It must order GPU work onto the supplied stream; completion is checked before
    /// the page becomes read-only and may be shared or exported.
    pub fn fill_with(
        self: &Arc<Self>,
        id: PageId,
        transform: impl FnOnce(FillPin) -> Result<()>,
    ) -> Result<ReadStats> {
        self.fill_attempt_bytes(id, None, None, transform).result
    }

    pub fn fill_from_ram_with(
        self: &Arc<Self>,
        id: PageId,
        bytes: &[u8],
        transform: impl FnOnce(FillPin) -> Result<()>,
    ) -> Result<ReadStats> {
        self.fill_attempt_bytes(id, Some(bytes), None, transform)
            .result
    }

    /// Retain a direct host source through failed CUDA work. The page's complete
    /// owner graph survives until checked recovery, including last-owner teardown.
    pub fn fill_from_ram_owned_with<T: AsRef<[u8]> + Send + Sync + 'static>(
        self: &Arc<Self>,
        id: PageId,
        source: Arc<T>,
        transform: impl FnOnce(FillPin) -> Result<()>,
    ) -> Result<ReadStats> {
        self.fill_attempt_bytes(
            id,
            Some(source.as_ref().as_ref()),
            Some(source.clone()),
            transform,
        )
        .result
    }

    /// Placement observes the actual failure phase without changing the public
    /// fill result or guessing from an absent completed-copy metric.
    pub(crate) fn fill_attempt_with<T: AsRef<[u8]> + Send + Sync + 'static>(
        self: &Arc<Self>,
        id: PageId,
        source: Option<Arc<T>>,
        transform: impl FnOnce(FillPin) -> Result<()>,
    ) -> FillAttempt {
        self.fill_attempt_bytes(
            id,
            source.as_ref().map(|source| source.as_ref().as_ref()),
            source
                .as_ref()
                .map(|source| source.clone() as Arc<dyn Send + Sync>),
            transform,
        )
    }

    fn fill_attempt_bytes(
        self: &Arc<Self>,
        id: PageId,
        ram: Option<&[u8]>,
        host_owner: Option<Arc<dyn Send + Sync>>,
        transform: impl FnOnce(FillPin) -> Result<()>,
    ) -> FillAttempt {
        let mut phase = FillPhase::Allocate;
        let result = self.fill_bytes(id, ram, host_owner, &mut phase, transform);
        FillAttempt { result, phase }
    }

    fn fill_bytes(
        self: &Arc<Self>,
        id: PageId,
        ram: Option<&[u8]>,
        host_owner: Option<Arc<dyn Send + Sync>>,
        phase: &mut FillPhase,
        transform: impl FnOnce(FillPin) -> Result<()>,
    ) -> Result<ReadStats> {
        let page = self.page(id)?;
        if ram.is_some_and(|bytes| bytes.len() as u64 != page.bytes) {
            return Err(invalid("RAM region size differs from the mapped page"));
        }
        {
            let mut state = page.state.lock();
            if page.retired.load(Ordering::Acquire) {
                return Err(invalid("page is retired"));
            }
            if !matches!(*state, State::Empty) {
                return Err(invalid("page is not empty"));
            }
            if let Some(owner) = host_owner {
                page.host_sources.lock().push(owner);
            }
            *state = State::Filling(thread::current().id());
        }
        let allocation = catch_unwind(AssertUnwindSafe(|| {
            Mapping::allocate(page.range.clone(), page.offset, page.bytes)
        }))
        .unwrap_or(Err(Error::Internal("VMM allocation panicked")));
        let mut mapping = match allocation {
            Ok(mapping) => mapping,
            Err(error) => {
                let mut state = page.state.lock();
                *state = State::Empty;
                page.host_sources.lock().clear();
                drop(state);
                self.notify();
                return Err(error);
            }
        };
        let result: Result<ReadStats> = catch_unwind(AssertUnwindSafe(|| {
            *phase = FillPhase::Read;
            let destination = Dst::Device {
                device: self.device.ordinal,
                ptr: mapping.ptr(),
                len: page.bytes as usize,
                stream: self.device.stream.raw(),
            };
            let stats = match ram {
                Some(bytes) => unsafe {
                    io::read(
                        &[io::Copy {
                            src: io::Src::Host(bytes.as_ptr()),
                            dst: destination,
                        }],
                        &io::ReadOpts::default(),
                    )
                }?,
                None => unsafe {
                    page.source.read_items(
                        &page.items,
                        page.origin,
                        destination,
                        2,
                        Mode::Cache,
                        &IoTally::default(),
                    )
                }?,
            };
            *phase = FillPhase::Transform;
            page.write_pins.fetch_add(1, Ordering::AcqRel);
            page.write_live.fetch_add(1, Ordering::AcqRel);
            transform(FillPin {
                gpu: self.clone(),
                page: page.clone(),
                ptr: mapping.ptr(),
                bytes: mapping.bytes,
                stream: self.device.stream.raw(),
                streams: Mutex::new(BTreeSet::from([self.device.stream.raw()])),
            })?;
            *phase = FillPhase::Finish;
            if page.write_live.load(Ordering::Acquire) != 0 {
                return Err(Error::Internal(
                    "a page-in transform retained a writable view",
                ));
            }
            self.wait_write_returns(&page)?;
            let mut events = page.write_events.lock();
            for event in events.iter() {
                event.synchronize()?;
            }
            events.clear();
            self.device.stream.drain()?;
            let _context = self.device.enter()?;
            self.device.access(mapping.ptr(), mapping.bytes, true)?;
            mapping.ready = Some(Arc::new(Event::record(
                &self.device,
                self.device.stream.raw(),
            )?));
            Ok(stats)
        }))
        .unwrap_or(Err(Error::Internal("page fill or transform panicked")));
        let mut state = page.state.lock();
        let result = match result {
            Ok(stats) => {
                *state = State::Ready(mapping);
                page.host_sources.lock().clear();
                Ok(stats)
            }
            Err(error) => {
                *state = State::Failed(mapping, error.clone());
                Err(error)
            }
        };
        drop(state);
        self.notify();
        result
    }

    pub fn bind(self: &Arc<Self>, id: PageId, stream: usize) -> Result<Pin> {
        self.poll()?;
        match self.try_pin(id, stream)? {
            BindState::Ready { pin, .. } => {
                pin.wait_on(stream)?;
                Ok(pin)
            }
            BindState::Empty | BindState::Filling(_) => {
                Err(invalid("page has not finished filling"))
            }
        }
    }

    /// Pin only the actual ready mapping. No poll, stream wait or callback runs
    /// here: placement calls it inside its short decision/publication gate.
    pub(crate) fn try_pin(self: &Arc<Self>, id: PageId, stream: usize) -> Result<BindState> {
        let page = self.page(id)?;
        let generation = self.generation.lock();
        let mut state = page.state.lock();
        if page.retired.load(Ordering::Acquire) {
            return Err(invalid("page is retired"));
        }
        let mapping = match &mut *state {
            State::Ready(mapping) => mapping,
            State::Failed(_, error) => return Err(error.clone()),
            State::Empty => return Ok(BindState::Empty),
            State::Filling(owner) => {
                if *owner == thread::current().id() {
                    return Err(invalid("page fill depends on its own unfinished page"));
                }
                return Ok(BindState::Filling(Pending {
                    event: None,
                    fill_generation: Some(*generation),
                }));
            }
        };
        mapping.pins = mapping
            .pins
            .checked_add(1)
            .ok_or_else(|| invalid("pin count overflows"))?;
        let ptr = mapping.ptr();
        Ok(BindState::Ready {
            pins: mapping.pins,
            pin: Pin {
                gpu: self.clone(),
                page: page.clone(),
                ptr,
                bytes: page.bytes,
                streams: Mutex::new(BTreeSet::from([stream])),
            },
        })
    }

    /// Apply queued pin releases on a CUDA-capable thread, not in a Python finalizer.
    pub fn poll(&self) -> Result<()> {
        let mut changed = false;
        while let Some(release) = self.releases.pop() {
            if release.writable {
                for stream in &release.streams {
                    if *stream == 2 {
                        let done = self
                            .device
                            .enter()
                            .and_then(|_context| call!("cuCtxSynchronize", ctx_synchronize()));
                        if let Err(error) = done {
                            self.releases.push(release);
                            self.write_changed();
                            return Err(error);
                        }
                        continue;
                    }
                    let event = Event::record(&self.device, *stream);
                    match event {
                        Ok(event) => release.page.write_events.lock().push(event),
                        Err(error) => {
                            self.releases.push(release);
                            self.write_changed();
                            return Err(error);
                        }
                    }
                }
                release.page.write_pins.fetch_sub(1, Ordering::AcqRel);
                // Wake an owner waiting for this exact returned lease even if a
                // later unrelated cleanup fails before poll's final notification.
                self.write_changed();
                changed = true;
                continue;
            }
            let mut state = release.page.state.lock();
            if let State::Ready(mapping) = &mut *state {
                for stream in &release.streams {
                    if *stream == 2 {
                        // CUDA's per-thread default sentinel cannot be replayed on
                        // this cleanup thread. A checked context drain covers the
                        // original thread's work before the pin can be released.
                        let completion = self
                            .device
                            .enter()
                            .and_then(|_context| call!("cuCtxSynchronize", ctx_synchronize()));
                        if let Err(error) = completion {
                            drop(state);
                            self.releases.push(release);
                            return Err(error);
                        }
                        continue;
                    }
                    match Event::record(&self.device, *stream) {
                        Ok(event) => mapping.releases.push(Arc::new(event)),
                        Err(error) => {
                            drop(state);
                            // Keep the pin and all stream identities if recording fails.
                            // A later recovery retries; no page becomes evictable early.
                            self.releases.push(release);
                            return Err(error);
                        }
                    }
                }
                mapping.pins = mapping.pins.saturating_sub(1);
                changed = true;
            }
        }
        let candidates: Vec<_> = self
            .pages
            .lock()
            .iter()
            .map(|(id, page)| (*id, page.clone()))
            .collect();
        for (id, page) in candidates {
            let mut state = page.state.lock();
            let retired = page.retired.load(Ordering::Acquire);
            if let State::Failed(mapping, _) = &mut *state {
                if page.write_pins.load(Ordering::Acquire) == 0 {
                    io::drain(self.device.ordinal)?;
                    self.device.stream.drain()?;
                    let mut events = page.write_events.lock();
                    for event in events.iter() {
                        event.synchronize()?;
                    }
                    events.clear();
                    mapping.release(true)?;
                    *state = State::Empty;
                    page.host_sources.lock().clear();
                    changed = true;
                }
            }
            if !retired {
                continue;
            }
            let removable = match &mut *state {
                State::Empty => page.write_pins.load(Ordering::Acquire) == 0,
                State::Ready(mapping) if mapping.pins == 0 => {
                    if mapping.idle()? {
                        mapping.release(true)?;
                        true
                    } else {
                        false
                    }
                }
                _ => false,
            };
            if removable {
                *state = State::Empty;
                self.pages.lock().remove(&id);
                changed = true;
            }
        }
        if changed {
            self.notify();
        }
        Ok(())
    }

    /// The coordinator installs a nonblocking wake (normally an eventfd write).
    /// It must capture the coordinator weakly, and never run a decision inline.
    pub fn set_wakeup(&self, wakeup: Arc<dyn Fn() + Send + Sync>) {
        *self.wakeup.lock() = Some(wakeup);
    }

    fn write_changed(&self) {
        let mut generation = self.generation.lock();
        *generation = generation.wrapping_add(1);
        self.changed.notify_all();
    }

    fn notify(&self) {
        self.write_changed();
        let wakeup = self.wakeup.lock().clone();
        if let Some(wakeup) = wakeup {
            let _ = catch_unwind(AssertUnwindSafe(|| wakeup()));
        }
    }

    fn wait_write_returns(&self, page: &Page) -> Result<()> {
        loop {
            // A different poller may return or requeue the lease while this poll
            // is running. Capture before poll so its notification cannot precede
            // registration unnoticed, including errors that leave write_pins > 0.
            let before = *self.generation.lock();
            self.poll()?;
            let mut generation = self.generation.lock();
            if page.write_pins.load(Ordering::Acquire) == 0 {
                return Ok(());
            }
            if *generation != before {
                continue;
            }
            // Both this predicate check/registration and write_changed's notify
            // hold generation. CUDA work above never runs under that mutex.
            #[cfg(test)]
            self.write_waiters.fetch_add(1, Ordering::AcqRel);
            self.changed.wait(&mut generation);
            #[cfg(test)]
            self.write_waiters.fetch_sub(1, Ordering::AcqRel);
        }
    }

    /// Wait for an actual in-flight fill or CUDA consumer fence. Live pins with
    /// no submitted completion are not guessed at and do not trigger a timer.
    pub fn wait_pending(&self) -> Result<bool> {
        self.wait_prepared(self.prepare_pending()?)
    }

    pub(crate) fn prepare_pending(&self) -> Result<Pending> {
        self.poll()?;
        let generation = self.generation.lock();
        let before = *generation;
        let pages: Vec<_> = self.pages.lock().values().cloned().collect();
        let mut filling = false;
        for page in pages {
            let state = page.state.lock();
            match &*state {
                // A transform may lazily bind another model. It cannot wait for
                // its own outer fill to finish: return the existing relief rung
                // so that dependency can make room, or report exhaustion.
                State::Filling(owner) => filling |= *owner != thread::current().id(),
                State::Ready(mapping) => {
                    for event in mapping.ready.iter().chain(&mapping.releases) {
                        if !event.done()? {
                            return Ok(Pending {
                                event: Some(event.clone()),
                                fill_generation: None,
                            });
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(Pending {
            event: None,
            fill_generation: filling.then_some(before),
        })
    }

    pub(crate) fn wait_prepared(&self, pending: Pending) -> Result<bool> {
        if let Some(event) = pending.event {
            // Mapping and generation guards were released by prepare_pending.
            // The Arc keeps this exact fence alive even if a concurrent poll
            // retires its mapping after observing completion.
            event.synchronize()?;
            return Ok(true);
        }
        if let Some(before) = pending.fill_generation {
            let mut generation = self.generation.lock();
            // A fill may notify between preparation/publication and this wait.
            while *generation == before {
                self.changed.wait(&mut generation);
            }
            return Ok(true);
        }
        Ok(false)
    }

    /// False means the page remains protected by a fill, view, or consumer event.
    pub fn unmap(&self, id: PageId) -> Result<bool> {
        self.poll()?;
        let page = self.page(id)?;
        if page.write_pins.load(Ordering::Acquire) != 0 {
            return Ok(false);
        }
        let mut state = page.state.lock();
        match &mut *state {
            State::Empty => Ok(true),
            State::Filling(_) => Ok(false),
            State::Ready(mapping) => {
                if !mapping.idle()? {
                    return Ok(false);
                }
                mapping.release(true)?;
                *state = State::Empty;
                page.host_sources.lock().clear();
                Ok(true)
            }
            State::Failed(mapping, _) => {
                io::drain(self.device.ordinal)?;
                self.device.stream.drain()?;
                mapping.release(true)?;
                *state = State::Empty;
                Ok(true)
            }
        }
    }

    /// Closing a model prevents new users immediately, but an existing view keeps
    /// its exact source/address until its last consumer fence finishes.
    pub fn unregister(&self, id: PageId) -> Result<bool> {
        let page = self.pages.lock().get(&id).cloned();
        let Some(page) = page else {
            return Ok(true);
        };
        page.retired.store(true, Ordering::Release);
        self.poll()?;
        let done = !self.pages.lock().contains_key(&id);
        self.notify();
        Ok(done)
    }

    /// Each receiving socket contains one exported allocation. Draining it on unmap
    /// invalidates the custodian's duplicate; process death leaves the handle available.
    pub fn export(&self, id: PageId) -> Result<(Vec<u64>, OwnedFd)> {
        self.poll()?;
        let page = self.page(id)?;
        let _context = self.device.enter()?;
        let mut state = page.state.lock();
        let State::Ready(mapping) = &mut *state else {
            return Err(invalid("page is not ready"));
        };
        // Ready pages are immutable. Retaining their allocation does not revoke
        // a view or disturb a consumer stream, so publish custody before first use.
        // This also preserves weights if the executor dies during that first use.
        if mapping.kept.is_none() {
            let (tx, rx) = UnixStream::pair().map_err(|error| Error::Io(error.to_string()))?;
            // A highly fragmented page must not block while filling a socket queue
            // which its custodian cannot receive until this method returns.
            tx.set_nonblocking(true)
                .map_err(|error| Error::Io(error.to_string()))?;
            for chunk in &mapping.chunks {
                let mut raw: c_int = -1;
                call!(
                    "cuMemExportToShareableHandle",
                    mem_export(
                        &mut raw as *mut c_int as *mut c_void,
                        chunk.handle,
                        cuda::MEM_HANDLE_TYPE_POSIX_FD,
                        0
                    )
                )?;
                let fd = unsafe { OwnedFd::from_raw_fd(raw) };
                platform::handles::send_fd(&tx, &fd)
                    .map_err(|error| Error::Io(error.to_string()))?;
            }
            mapping.kept = Some(rx);
        }
        let container = mapping
            .kept
            .as_ref()
            .ok_or_else(|| invalid("missing page container"))?;
        let fd = container
            .try_clone()
            .map_err(|error| Error::Io(error.to_string()))?;
        Ok((
            mapping.chunks.iter().map(|chunk| chunk.bytes).collect(),
            fd.into(),
        ))
    }

    /// Consume restart custody only when its complete identity matches. An absent or
    /// invalidated container is a cache miss; it does not change the disk/RAM source.
    pub fn import(
        &self,
        id: PageId,
        fingerprint: &str,
        spans: &[u64],
        container: OwnedFd,
    ) -> Result<bool> {
        let page = self.page(id)?;
        if fingerprint != page.fingerprint {
            return Ok(false);
        }
        if spans.is_empty()
            || spans
                .iter()
                .any(|n| *n == 0 || !n.is_multiple_of(self.device.granularity))
            || spans.iter().try_fold(0u64, |sum, n| sum.checked_add(*n)) != Some(page.bytes)
        {
            return Ok(false);
        }
        let mut state = page.state.lock();
        if !matches!(*state, State::Empty) {
            return Ok(false);
        }
        let _context = self.device.enter()?;
        let receiver = UnixStream::from(container);
        receiver
            .set_nonblocking(true)
            .map_err(|error| Error::Io(error.to_string()))?;
        let mut mapping = Mapping {
            range: page.range.clone(),
            offset: page.offset,
            bytes: 0,
            chunks: vec![],
            ready: None,
            releases: vec![],
            pins: 0,
            kept: None,
        };
        for bytes in spans {
            let Ok(fd) = platform::handles::recv_fd(&receiver) else {
                return Ok(false);
            };
            let mut handle = 0;
            if let Err(error) = call!(
                "cuMemImportFromShareableHandle",
                mem_import(
                    &mut handle,
                    fd.as_raw_fd() as usize as *mut c_void,
                    cuda::MEM_HANDLE_TYPE_POSIX_FD
                )
            ) {
                if error.is_sticky() {
                    return Err(error);
                }
                return Ok(false);
            }
            if let Err(error) = call!(
                "cuMemMap",
                mem_map(mapping.ptr() + mapping.bytes, *bytes as usize, 0, handle, 0)
            ) {
                let _ = call!("cuMemRelease", mem_release(handle));
                if error.is_sticky() {
                    return Err(error);
                }
                return Ok(false);
            }
            mapping.chunks.push(Chunk {
                handle,
                bytes: *bytes,
            });
            mapping.bytes += bytes;
        }
        self.device.access(mapping.ptr(), mapping.bytes, true)?;
        mapping.ready = Some(Arc::new(Event::record(
            &self.device,
            self.device.stream.raw(),
        )?));
        *state = State::Ready(mapping);
        Ok(true)
    }

    /// Retry failed fills only after the engine and producer stream prove completion.
    /// A sticky CUDA failure is returned; the Runtime then replaces the executor.
    pub fn recover(&self) -> Result<()> {
        io::drain(self.device.ordinal)?;
        self.device.stream.drain()?;
        self.poll()?;
        let pages: Vec<_> = self.pages.lock().values().cloned().collect();
        for page in pages {
            if page.write_live.load(Ordering::Acquire) == 0
                && page.write_pins.load(Ordering::Acquire) != 0
            {
                self.wait_write_returns(&page)?;
            }
            let mut state = page.state.lock();
            if let State::Failed(mapping, _) = &mut *state {
                if page.write_pins.load(Ordering::Acquire) != 0 {
                    continue;
                }
                let mut events = page.write_events.lock();
                for event in events.iter() {
                    event.synchronize()?;
                }
                events.clear();
                mapping.release(true)?;
                *state = State::Empty;
                page.host_sources.lock().clear();
            }
        }
        Ok(())
    }

    /// Derive mechanism facts from mappings; the final flag includes pending consumer
    /// events, so the planner cannot count such a mapping as immediately reclaimable.
    pub fn facts(&self) -> Result<Vec<(PageId, u64, usize, bool)>> {
        self.poll()?;
        self.facts_current()
    }

    /// Capture without polling: placement serializes this capture and its
    /// publication with pin acquisition, after any CUDA cleanup and IPC.
    pub(crate) fn facts_current(&self) -> Result<Vec<(PageId, u64, usize, bool)>> {
        let mut facts = Vec::new();
        let pages: Vec<_> = self
            .pages
            .lock()
            .iter()
            .map(|(id, page)| (*id, page.clone()))
            .collect();
        for (id, page) in pages {
            facts.push(match &mut *page.state.lock() {
                State::Empty => (id, 0, 0, false),
                State::Filling(_) => (id, page.bytes, 0, true),
                State::Ready(mapping) => (id, mapping.bytes, mapping.pins, !mapping.idle()?),
                State::Failed(mapping, _) => (id, mapping.bytes, mapping.pins, true),
            });
        }
        Ok(facts)
    }

    pub fn fingerprint(&self, id: PageId) -> Result<String> {
        Ok(self.page(id)?.fingerprint.clone())
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        let safe = io::drain(self.device.ordinal).is_ok()
            && self.device.stream.drain().is_ok()
            && self
                .device
                .enter()
                .and_then(|_context| {
                    // Last-owner teardown includes consumer streams whose release
                    // is still queued, including per-thread default streams.
                    call!("cuCtxSynchronize", ctx_synchronize())
                })
                .is_ok();
        if safe {
            // An explicit model close retires its pages even if a tensor outlives
            // the coordinator. Last-pin teardown must revoke their queued custody;
            // only unretired pages are intentionally kept for a replacement owner.
            for page in self.pages.get_mut().values() {
                if !page.retired.load(Ordering::Acquire) {
                    continue;
                }
                if let State::Ready(mapping) | State::Failed(mapping, _) = &mut *page.state.lock() {
                    let _ = mapping.release(true);
                }
            }
        } else {
            // An unfinished engine read may still borrow source descriptors and host
            // mappings. Keep the complete owner graph alive until process teardown.
            for (_, page) in std::mem::take(self.pages.get_mut()) {
                std::mem::forget(page);
            }
            while let Some(release) = self.releases.pop() {
                std::mem::forget(release);
            }
        }
    }
}

/// Force the release-dequeued/producer-finishing interleaving on real CUDA.
/// The observer counter exists only in native test binaries; it changes no wait.
#[cfg(test)]
pub(crate) fn returned_write_release_interleaving() -> Result<()> {
    use std::sync::mpsc;
    use std::time::{Duration, Instant};
    let gpu = Gpu::open(0)?;
    let length = crate::layout::REGION_ALIGN;
    gpu.register(
        1,
        gpu.reserve(length)?,
        0,
        length,
        Arc::new(Source::files(vec![], false, 0)),
        vec![Item {
            offset: 0,
            len: 1,
            part_end: true,
            source: io::ItemSource::Inline(vec![7].into()),
        }],
        0,
        "returned-write-interleaving".into(),
    )?;
    let (sent, returned) = mpsc::channel();
    let filling = gpu.clone();
    let worker = thread::spawn(move || {
        filling.fill_with(1, |write| {
            drop(write);
            // This standalone Gpu has no automatic poller. Retain the exact release
            // as another poller would after dequeue, before recording its fences.
            let release = filling.releases.pop().expect("returned writable lease");
            assert!(sent.send(release).is_ok(), "release observer disappeared");
            Ok(())
        })
    });
    let release = returned
        .recv_timeout(Duration::from_secs(10))
        .map_err(|_| Error::Internal("fill did not return its writable lease"))?;
    let page = gpu.page(1)?;
    assert_eq!(page.write_live.load(Ordering::Acquire), 0);
    assert_eq!(page.write_pins.load(Ordering::Acquire), 1);
    let deadline = Instant::now() + Duration::from_secs(10);
    let waited = loop {
        if gpu.write_waiters.load(Ordering::Acquire) != 0 {
            break true;
        }
        if worker.is_finished() || Instant::now() >= deadline {
            break false;
        }
        thread::yield_now();
    };
    gpu.releases.push(release);
    let completion = gpu.poll();
    let filled = worker
        .join()
        .map_err(|_| Error::Internal("fill worker panicked"))?;
    completion?;
    if !waited {
        return Err(Error::Internal(
            "returned write was mistaken for a retained view",
        ));
    }
    filled?;
    let pin = gpu.bind(1, 0)?;
    let mut bytes = [0_u8; 1];
    unsafe {
        io::read(
            &[io::Copy {
                src: io::Src::Device {
                    device: 0,
                    ptr: pin.ptr,
                    stream: 0,
                },
                dst: io::Dst::host(&mut bytes),
            }],
            &io::ReadOpts::default(),
        )
    }?;
    assert_eq!(bytes, [7]);
    drop(pin);
    gpu.wait_pending()?;
    assert!(gpu.unmap(1)?);

    // The opposite ordering must not wait for an already-delivered notification:
    // finish the returned lease inside the callback, before the producer enters
    // wait_write_returns. This uses real event recording and the same page again.
    gpu.fill_with(1, |write| {
        drop(write);
        gpu.poll()?;
        assert_eq!(page.write_pins.load(Ordering::Acquire), 0);
        Ok(())
    })?;
    let pin = gpu.bind(1, 0)?;
    unsafe {
        io::read(
            &[io::Copy {
                src: io::Src::Device {
                    device: 0,
                    ptr: pin.ptr,
                    stream: 0,
                },
                dst: io::Dst::host(&mut bytes),
            }],
            &io::ReadOpts::default(),
        )
    }?;
    assert_eq!(bytes, [7]);
    assert_eq!(gpu.write_waiters.load(Ordering::Acquire), 0);
    drop(pin);
    gpu.wait_pending()?;
    assert!(gpu.unregister(1)?);
    Ok(())
}

/// The storage owner. Every stream that sees its pointer must first call `wait_on`.
pub struct Pin {
    gpu: Arc<Gpu>,
    page: Arc<Page>,
    pub ptr: u64,
    pub bytes: u64,
    streams: Mutex<BTreeSet<usize>>,
}

impl Pin {
    pub fn wait_on(&self, stream: usize) -> Result<()> {
        self.streams.lock().insert(stream);
        let ready = match &*self.page.state.lock() {
            State::Ready(mapping) => mapping.ready.clone(),
            _ => return Err(invalid("pinned page is not ready")),
        };
        ready.as_ref().map_or(Ok(()), |event| event.wait(stream))
    }
}

impl Drop for Pin {
    fn drop(&mut self) {
        self.gpu.releases.push(Release {
            page: self.page.clone(),
            streams: self.streams.get_mut().clone(),
            writable: false,
        });
        self.gpu.notify();
    }
}

/// A writable page view exists only during a page-in transform. An escaped view
/// keeps the allocation and source alive and prevents read-only publication.
pub struct FillPin {
    gpu: Arc<Gpu>,
    page: Arc<Page>,
    pub ptr: u64,
    pub bytes: u64,
    pub stream: usize,
    streams: Mutex<BTreeSet<usize>>,
}
impl FillPin {
    pub fn device(&self) -> i32 {
        self.gpu.device()
    }
    pub fn wait_on(&self, stream: usize) -> Result<()> {
        self.streams.lock().insert(stream);
        Event::record(&self.gpu.device, self.stream)?.wait(stream)
    }
}
impl Drop for FillPin {
    fn drop(&mut self) {
        self.gpu.releases.push(Release {
            page: self.page.clone(),
            streams: self.streams.get_mut().clone(),
            writable: true,
        });
        self.page.write_live.fetch_sub(1, Ordering::AcqRel);
        self.gpu.notify();
    }
}

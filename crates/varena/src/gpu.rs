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
use std::sync::Arc;

use crossbeam_queue::SegQueue;
use parking_lot::Mutex;
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
    ready: Option<Event>,
    releases: Vec<Event>,
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
    Filling,
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
}

struct Release {
    page: Arc<Page>,
    streams: BTreeSet<usize>,
}

/// Holds mappings, never chooses victims. The policy invokes fill/unmap with page IDs.
pub struct Gpu {
    device: Arc<Device>,
    pages: Mutex<BTreeMap<PageId, Arc<Page>>>,
    releases: SegQueue<Release>,
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
            || offset % g != 0
            || bytes % g != 0
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
    pub fn fill(&self, id: PageId) -> Result<ReadStats> {
        self.fill_bytes(id, None)
    }

    /// The RAM-tier lease must remain alive for this call. TensorFS still performs
    /// the transfer; the mechanism only selects a ready source and its destination.
    pub fn fill_from_ram(&self, id: PageId, bytes: &[u8]) -> Result<ReadStats> {
        self.fill_bytes(id, Some(bytes))
    }

    fn fill_bytes(&self, id: PageId, ram: Option<&[u8]>) -> Result<ReadStats> {
        let page = self.page(id)?;
        if ram.is_some_and(|bytes| bytes.len() as u64 != page.bytes) {
            return Err(invalid("RAM region size differs from the mapped page"));
        }
        {
            let mut state = page.state.lock();
            if !matches!(*state, State::Empty) {
                return Err(invalid("page is not empty"));
            }
            *state = State::Filling;
        }
        let mut mapping = match Mapping::allocate(page.range.clone(), page.offset, page.bytes) {
            Ok(mapping) => mapping,
            Err(error) => {
                *page.state.lock() = State::Empty;
                return Err(error);
            }
        };
        let result: Result<ReadStats> = (|| {
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
            let _context = self.device.enter()?;
            self.device.access(mapping.ptr(), mapping.bytes, true)?;
            mapping.ready = Some(Event::record(&self.device, self.device.stream.raw())?);
            Ok(stats)
        })();
        let mut state = page.state.lock();
        match result {
            Ok(stats) => {
                *state = State::Ready(mapping);
                Ok(stats)
            }
            Err(error) => {
                *state = State::Failed(mapping, error.clone());
                Err(error)
            }
        }
    }

    pub fn bind(self: &Arc<Self>, id: PageId, stream: usize) -> Result<Pin> {
        self.poll()?;
        let page = self.page(id)?;
        let mut state = page.state.lock();
        let mapping = match &mut *state {
            State::Ready(mapping) => mapping,
            State::Failed(_, error) => return Err(error.clone()),
            _ => return Err(invalid("page has not finished filling")),
        };
        if let Some(ready) = &mapping.ready {
            ready.wait(stream)?;
        }
        mapping.pins = mapping
            .pins
            .checked_add(1)
            .ok_or_else(|| invalid("pin count overflows"))?;
        let ptr = mapping.ptr();
        Ok(Pin {
            gpu: self.clone(),
            page: page.clone(),
            ptr,
            bytes: page.bytes,
            streams: Mutex::new(BTreeSet::from([stream])),
        })
    }

    /// Apply queued pin releases on a CUDA-capable thread, not in a Python finalizer.
    pub fn poll(&self) -> Result<()> {
        while let Some(release) = self.releases.pop() {
            let mut state = release.page.state.lock();
            if let State::Ready(mapping) = &mut *state {
                for stream in &release.streams {
                    match Event::record(&self.device, *stream) {
                        Ok(event) => mapping.releases.push(event),
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
            }
        }
        Ok(())
    }

    /// False means the page remains protected by a fill, view, or consumer event.
    pub fn unmap(&self, id: PageId) -> Result<bool> {
        self.poll()?;
        let page = self.page(id)?;
        let mut state = page.state.lock();
        match &mut *state {
            State::Empty => Ok(true),
            State::Filling => Ok(false),
            State::Ready(mapping) => {
                if !mapping.idle()? {
                    return Ok(false);
                }
                mapping.release(true)?;
                *state = State::Empty;
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
        if !mapping.idle()? {
            return Err(invalid("page is busy"));
        }
        if mapping.kept.is_none() {
            let (tx, rx) = UnixStream::pair().map_err(|error| Error::Io(error.to_string()))?;
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
                .any(|n| *n == 0 || *n % self.device.granularity != 0)
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
        mapping.ready = Some(Event::record(&self.device, self.device.stream.raw())?);
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
            let mut state = page.state.lock();
            if let State::Failed(mapping, _) = &mut *state {
                mapping.release(true)?;
                *state = State::Empty;
            }
        }
        Ok(())
    }

    /// Derive mechanism facts from mappings; there is no second cached byte counter.
    pub fn facts(&self) -> Result<Vec<(PageId, u64, usize, bool)>> {
        self.poll()?;
        Ok(self
            .pages
            .lock()
            .iter()
            .map(|(id, page)| match &*page.state.lock() {
                State::Empty => (*id, 0, 0, false),
                State::Filling => (*id, page.bytes, 0, true),
                State::Ready(mapping) => (*id, mapping.bytes, mapping.pins, false),
                State::Failed(mapping, _) => (*id, mapping.bytes, mapping.pins, true),
            })
            .collect())
    }

    pub fn fingerprint(&self, id: PageId) -> Result<String> {
        Ok(self.page(id)?.fingerprint.clone())
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        let safe = io::drain(self.device.ordinal).is_ok() && self.device.stream.drain().is_ok();
        if !safe {
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
        let state = self.page.state.lock();
        match &*state {
            State::Ready(mapping) => mapping
                .ready
                .as_ref()
                .map_or(Ok(()), |event| event.wait(stream)),
            _ => Err(invalid("pinned page is not ready")),
        }
    }
}

impl Drop for Pin {
    fn drop(&mut self) {
        self.gpu.releases.push(Release {
            page: self.page.clone(),
            streams: self.streams.get_mut().clone(),
        });
    }
}

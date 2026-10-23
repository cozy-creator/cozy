//! The CUDA VMM [`Backend`]: shareable physical pages (a POSIX fd, a WIN32 handle) mapped into
//! reserved address space, filled pages remapped read-only, and fills through TensorFS's engine
//! into `Dst::Device` ordered on varena's copy stream. Everything runs in the device's primary
//! context, the one torch uses, so a page's pointer is valid there unchanged. Nothing is stored
//! but the device's facts: a page's id is its `CUmemGenericAllocationHandle`, an event's its
//! `CUevent`.

use std::ffi::{c_char, c_int, c_void, CStr};

use platform::handles::Owned;
use tensorfs::io::cuda::{self, AccessDesc, AllocationProp, CUcontext, CUdevice, CUevent, CUstream, CtxGuard};
use tensorfs::io::Dst;

use crate::backend::{Access, Backend, Event, Events, Physical, Va};
use crate::{Error, GpuId, Result};

#[cfg(unix)]
const HANDLES: u32 = cuda::MEM_HANDLE_TYPE_POSIX_FD;
#[cfg(unix)]
const HANDLES_SUPPORTED: c_int = cuda::ATTR_POSIX_FD_SUPPORTED;
#[cfg(windows)]
const HANDLES: u32 = cuda::MEM_HANDLE_TYPE_WIN32;
#[cfg(windows)]
const HANDLES_SUPPORTED: c_int = cuda::ATTR_WIN32_HANDLE_SUPPORTED;

/// One driver call; the caller has the context entered.
macro_rules! call {
    ($op:literal, $f:ident($($arg:expr),* $(,)?)) => {{
        // Each driver call is a trace span of its name: a subscriber can time every one.
        let _call = tracing::trace_span!($op).entered();
        // SAFETY: the documented driver ABI, on live handles of this backend's context; every
        // out-param points at a live local.
        cuda::check($op, unsafe { (cuda::driver()?.$f)($($arg),*) })
    }};
}

/// This process's reference on the device's primary context.
struct Primary {
    device: CUdevice,
    ctx: usize,
}

impl Primary {
    fn enter(&self) -> Result<CtxGuard> {
        Ok(CtxGuard::enter(self.ctx as CUcontext)?)
    }
}

impl Drop for Primary {
    fn drop(&mut self) {
        if let Ok(d) = cuda::driver() {
            // SAFETY: releases the reference `open` retained.
            unsafe { (d.primary_ctx_release)(self.device) };
        }
    }
}

/// `SECURITY_ATTRIBUTES` (minwinbase.h): with no descriptor, an exported page's NT handle gets
/// the creator's default security; it crosses to a peer by `DuplicateHandle`, never by name.
#[cfg(windows)]
#[repr(C)]
struct SecurityAttributes {
    length: u32,
    descriptor: *mut c_void,
    inherit: i32,
}

/// One CUDA device's VMM mechanism.
pub struct CudaBackend {
    ordinal: i32,
    /// Varena's copy stream: fills are ordered on it. Dropped (synchronized) before the context.
    stream: cuda::Stream,
    primary: Primary,
    prop: AllocationProp,
    granularity: u64,
    capacity: u64,
    nvml: Option<platform::gpu::Device>,
    /// The device's used bytes `open` measured across making the context.
    context: u64,
    /// varena's event ids over the driver's handles (`CUevent` as usize): never reused.
    events: Events<usize>,
    /// What `prop`'s WIN32 metadata points at, for as long as pages are created.
    #[cfg(windows)]
    _security: Box<SecurityAttributes>,
    /// This process's WDDM budget on the device's adapter (matched by LUID), where DXGI has one.
    #[cfg(windows)]
    budget: Option<std::sync::Arc<platform::gpu::Budget>>,
}

// SAFETY: driver handles are opaque and every use enters their context; `prop`'s only pointer is
// the WIN32 metadata, owned by the backend and never written.
unsafe impl Send for CudaBackend {}
unsafe impl Sync for CudaBackend {}

impl CudaBackend {
    /// Device `ordinal` as CUDA enumerates it (after `CUDA_VISIBLE_DEVICES`), on its primary
    /// context, retained while the backend lives.
    pub fn open(ordinal: i32) -> Result<CudaBackend> {
        let mut device: CUdevice = 0;
        call!("cuDeviceGet", device_get(&mut device, ordinal))?;
        for (attr, what) in [(cuda::ATTR_VMM_SUPPORTED, "virtual memory management"), (HANDLES_SUPPORTED, "shareable handles")] {
            if cuda::attribute(device, attr)? == 0 {
                return Err(Error::Invalid(format!("CUDA device {ordinal} has no {what}")));
            }
        }
        let nvml = platform::gpu::Device::open(&cuda::uuid(device)?)
            .inspect_err(|e| tracing::warn!(target: "varena", "NVML: {e}: no process bytes measured"))
            .ok();
        let before = used(nvml.as_ref());
        let (_, ctx) = cuda::primary_context(ordinal)?;
        let primary = Primary { device, ctx: ctx as usize };
        let stream = cuda::Stream::new(ordinal)?;
        // A context that existed before `open` (a framework made it) measures 0 here.
        let context = used(nvml.as_ref()).zip(before).map_or(0, |(after, before)| after.saturating_sub(before));
        #[allow(unused_mut)]
        let mut prop = AllocationProp::device(device, HANDLES);
        #[cfg(windows)]
        let security = Box::new(SecurityAttributes { length: std::mem::size_of::<SecurityAttributes>() as u32, descriptor: std::ptr::null_mut(), inherit: 0 });
        #[cfg(windows)]
        {
            prop.win32_handle_meta_data = (&*security as *const SecurityAttributes).cast_mut().cast();
        }
        #[cfg(windows)]
        let budget = budget(device);
        let _ctx = primary.enter()?;
        let mut granularity = 0usize;
        call!("cuMemGetAllocationGranularity", mem_get_allocation_granularity(&mut granularity, &prop, cuda::MEM_GRANULARITY_MINIMUM))?;
        let (mut free, mut total) = (0usize, 0usize);
        call!("cuMemGetInfo", mem_get_info(&mut free, &mut total))?;
        drop(_ctx);
        Ok(CudaBackend {
            ordinal,
            stream,
            primary,
            prop,
            granularity: granularity as u64,
            capacity: total as u64,
            nvml,
            context,
            events: Events::default(),
            #[cfg(windows)]
            _security: security,
            #[cfg(windows)]
            budget,
        })
    }


    pub fn ordinal(&self) -> i32 {
        self.ordinal
    }

    /// The copy stream (a `CUstream`).
    pub fn stream(&self) -> usize {
        self.stream.raw()
    }

    /// The device as the shared table names it.
    pub fn gpu(&self) -> Result<GpuId> {
        let mut name = [0 as c_char; 256];
        call!("cuDeviceGetName", device_get_name(name.as_mut_ptr(), name.len() as c_int, self.primary.device))?;
        // SAFETY: the driver wrote a NUL-terminated name into the buffer.
        let model = unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned();
        Ok(GpuId { uuid: cuda::uuid(self.primary.device)?, model, memory: self.capacity })
    }

    fn aligned(&self, what: &str, n: u64) -> Result<()> {
        match n % self.granularity {
            0 => Ok(()),
            _ => Err(Error::Invalid(format!("{what} {n} is not a multiple of the {} byte granularity", self.granularity))),
        }
    }

    fn access(&self, ptr: u64, len: u64, access: Access) -> Result<()> {
        let flags = match access {
            Access::ReadWrite => cuda::MEM_ACCESS_PROT_READWRITE,
            Access::Read => cuda::MEM_ACCESS_PROT_READ,
        };
        let desc = AccessDesc { location: self.prop.location, flags };
        Ok(call!("cuMemSetAccess", mem_set_access(ptr, len as usize, &desc, 1))?)
    }

    /// `f` in the context; a failure of a call with no result to return is logged.
    fn logged(&self, op: &str, f: impl FnOnce() -> Result<()>) {
        if let Err(e) = self.primary.enter().and_then(|_ctx| f()) {
            tracing::warn!(target: "varena", "{op}: {e}");
        }
    }
}

/// The device's DXGI budget, its adapter found by the device's LUID. None (logged) where DXGI
/// names no such adapter or will not register the change event: no budget signal then.
#[cfg(windows)]
fn budget(device: CUdevice) -> Option<std::sync::Arc<platform::gpu::Budget>> {
    let opened = cuda::luid(device).map_err(Error::from).and_then(|luid| platform::gpu::Budget::open(luid).map_err(Error::from));
    match opened {
        Ok(Some(budget)) => Some(std::sync::Arc::new(budget)),
        Ok(None) => {
            tracing::warn!(target: "varena", "DXGI names no adapter with CUDA device {device}'s LUID: no budget signal");
            None
        }
        Err(e) => {
            tracing::warn!(target: "varena", "DXGI budget: {e}: no budget signal");
            None
        }
    }
}

#[cfg(windows)]
impl crate::backend::Budget for platform::gpu::Budget {
    fn read(&self) -> Result<(u64, u64)> {
        Ok(platform::gpu::Budget::read(self)?)
    }

    fn wait(&self) -> bool {
        platform::gpu::Budget::wait(self)
    }

    fn close(&self) {
        platform::gpu::Budget::close(self)
    }
}

/// The device's used bytes as NVML reads them; `None` when NVML can't.
fn used(nvml: Option<&platform::gpu::Device>) -> Option<u64> {
    let reading = nvml?.read().ok()?;
    Some(reading.total.saturating_sub(reading.free))
}

impl CudaBackend {
    /// `event`'s live driver handle; a retired or unknown id is an error, never a stale handle.
    fn raw(&self, event: Event) -> Result<CUevent> {
        self.events.get(event).map(|h| h as CUevent).ok_or_else(|| Error::Internal(format!("event {} is not live", event.0)))
    }

    /// A queued handle is complete, or unknowable (its query failed): nothing waits on it.
    fn complete(&self, handle: usize) -> bool {
        let Ok(_ctx) = self.primary.enter() else { return false };
        // SAFETY: a handle this backend made and has not destroyed.
        let code = cuda::driver().map(|d| unsafe { (d.event_query)(handle as CUevent) });
        !matches!(code, Ok(cuda::ERROR_NOT_READY))
    }

    fn destroy_event(&self, handle: usize) {
        self.logged("cuEventDestroy", || Ok(call!("cuEventDestroy", event_destroy(handle as CUevent))?));
    }
}

impl Backend for CudaBackend {
    fn capacity(&self) -> u64 {
        self.capacity
    }

    fn free(&self) -> Result<u64> {
        let _ctx = self.primary.enter()?;
        let (mut free, mut total) = (0usize, 0usize);
        call!("cuMemGetInfo", mem_get_info(&mut free, &mut total))?;
        Ok(free as u64)
    }

    /// NVML's figure for this pid. Where the driver names no process of ours (WDDM has no
    /// per-process numbers; a container whose pids are the host's) there is none: an error, and
    /// the context stays as `open` measured it (`opened_context`).
    fn process_used(&self) -> Result<u64> {
        let nvml = self.nvml.as_ref().ok_or_else(|| Error::Io("NVML is unavailable: no process bytes".into()))?;
        let reading = nvml.read()?;
        reading.processes.as_ref().and_then(|p| p.get(&std::process::id()).copied()).ok_or_else(|| Error::Device {
            op: "process_used",
            code: -1,
            name: "the driver names no process of ours (WDDM, or another pid namespace)".into(),
        })
    }

    /// The device's used bytes `open` measured across making the context (0 when a framework
    /// made it first): the context where the driver has no per-process numbers.
    fn opened_context(&self) -> u64 {
        self.context
    }

    fn granularity(&self) -> u64 {
        self.granularity
    }

    fn reserve(&self, len: u64) -> Result<Va> {
        let len = len.checked_next_multiple_of(self.granularity).ok_or_else(|| Error::Invalid(format!("reserve {len} overflows")))?;
        let _ctx = self.primary.enter()?;
        let mut base = 0u64;
        call!("cuMemAddressReserve", mem_address_reserve(&mut base, len as usize, 0, 0, 0))?;
        Ok(Va { base, len })
    }

    fn release(&self, va: Va) {
        self.logged("cuMemAddressFree", || Ok(call!("cuMemAddressFree", mem_address_free(va.base, va.len as usize))?));
    }

    fn create(&self, len: u64) -> Result<Physical> {
        self.aligned("a page of", len)?;
        let _ctx = self.primary.enter()?;
        let mut id = 0;
        call!("cuMemCreate", mem_create(&mut id, len as usize, &self.prop, 0))?;
        Ok(Physical { id, len })
    }

    fn destroy(&self, page: Physical) -> Result<()> {
        let _ctx = self.primary.enter()?;
        Ok(call!("cuMemRelease", mem_release(page.id))?)
    }

    fn map(&self, va: Va, offset: u64, page: &Physical, access: Access) -> Result<()> {
        self.aligned("offset", offset)?;
        if offset.checked_add(page.len).is_none_or(|end| end > va.len) {
            return Err(Error::Invalid(format!("{} bytes at {offset} outside a {} byte reservation", page.len, va.len)));
        }
        let ptr = va.base + offset;
        let _ctx = self.primary.enter()?;
        call!("cuMemMap", mem_map(ptr, page.len as usize, 0, page.id, 0))?;
        if let Err(error) = self.access(ptr, page.len, access) {
            if let Err(release) = call!("cuMemUnmap", mem_unmap(ptr, page.len as usize)) {
                return Err(Error::Ownership(format!("mapping access failed ({error}); rollback failed ({release})")));
            }
            return Err(error);
        }
        Ok(())
    }

    fn protect(&self, va: Va, offset: u64, len: u64, access: Access) -> Result<()> {
        let _ctx = self.primary.enter()?;
        self.access(va.base + offset, len, access)
    }

    fn unmap(&self, va: Va, offset: u64, len: u64) -> Result<()> {
        let _ctx = self.primary.enter()?;
        Ok(call!("cuMemUnmap", mem_unmap(va.base + offset, len as usize))?)
    }

    fn export(&self, page: &Physical) -> Result<Owned> {
        let _ctx = self.primary.enter()?;
        #[cfg(unix)]
        {
            use std::os::fd::FromRawFd;
            let mut fd: c_int = -1;
            call!("cuMemExportToShareableHandle", mem_export((&mut fd as *mut c_int).cast(), page.id, HANDLES, 0))?;
            // SAFETY: the driver made this descriptor for us to own.
            Ok(unsafe { Owned::from_raw_fd(fd) })
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::FromRawHandle;
            let mut handle: *mut c_void = std::ptr::null_mut();
            call!("cuMemExportToShareableHandle", mem_export((&mut handle as *mut *mut c_void).cast(), page.id, HANDLES, 0))?;
            // SAFETY: the driver made this NT handle for us to own.
            Ok(unsafe { Owned::from_raw_handle(handle) })
        }
    }

    fn import(&self, handle: Owned, len: u64) -> Result<Physical> {
        self.aligned("a page of", len)?;
        #[cfg(unix)]
        let os: *mut c_void = {
            use std::os::fd::AsRawFd;
            handle.as_raw_fd() as usize as *mut c_void
        };
        #[cfg(windows)]
        let os: *mut c_void = {
            use std::os::windows::io::AsRawHandle;
            handle.as_raw_handle()
        };
        let _ctx = self.primary.enter()?;
        let mut id = 0;
        call!("cuMemImportFromShareableHandle", mem_import(&mut id, os, HANDLES))?;
        // The allocation holds its own reference now; our handle closes.
        drop(handle);
        Ok(Physical { id, len })
    }

    fn target(&self, va: Va, offset: u64, len: u64) -> Dst {
        Dst::Device { device: self.ordinal, ptr: va.base + offset, len: len as usize, stream: self.stream.raw() }
    }

    fn copy_in(&self, va: Va, offset: u64, src: &[u8]) -> Result<()> {
        crate::backend::inside(va, offset, src.len())?;
        let _ctx = self.primary.enter()?;
        call!("cuMemcpyHtoDAsync", memcpy_htod_async(va.base + offset, src.as_ptr().cast(), src.len(), self.stream.raw() as CUstream))?;
        Ok(self.stream.drain()?)
    }

    fn fence(&self) -> Result<Event> {
        self.record(self.stream.raw())
    }

    fn record(&self, stream: usize) -> Result<Event> {
        let _ctx = self.primary.enter()?;
        let mut event: CUevent = std::ptr::null_mut();
        call!("cuEventCreate", event_create(&mut event, cuda::EVENT_DEFAULT))?;
        if let Err(e) = call!("cuEventRecord", event_record(event, stream as CUstream)) {
            // SAFETY: the event just made; nothing recorded it.
            unsafe { cuda::driver().map(|d| (d.event_destroy)(event)) }.ok();
            return Err(e.into());
        }
        Ok(self.events.add(event as usize))
    }

    fn query(&self, event: Event) -> Result<bool> {
        if event == Event::DONE {
            return Ok(true);
        }
        let _ctx = self.primary.enter()?;
        let _call = tracing::trace_span!("cuEventQuery").entered();
        // SAFETY: a live event of this context.
        match unsafe { (cuda::driver()?.event_query)(self.raw(event)?) } {
            cuda::SUCCESS => Ok(true),
            cuda::ERROR_NOT_READY => Ok(false),
            code => Err(cuda::check("cuEventQuery", code).err().map_or_else(|| Error::Internal("cuEventQuery".into()), Error::from)),
        }
    }

    fn elapsed(&self, from: Event, to: Event) -> Result<Option<u64>> {
        if from == Event::DONE || to == Event::DONE {
            return Ok(Some(0));
        }
        if !(self.query(from)? && self.query(to)?) {
            return Ok(None);
        }
        let _ctx = self.primary.enter()?;
        let mut ms = 0f32;
        call!("cuEventElapsedTime", event_elapsed_time(&mut ms, self.raw(from)?, self.raw(to)?))?;
        Ok(Some((f64::from(ms.max(0.0)) * 1e6) as u64))
    }

    fn wait(&self, event: Event, stream: usize) -> Result<()> {
        if event == Event::DONE {
            return Ok(());
        }
        let _ctx = self.primary.enter()?;
        Ok(call!("cuStreamWaitEvent", stream_wait_event(stream as CUstream, self.raw(event)?, 0))?)
    }

    fn synchronize(&self, stream: usize) -> Result<()> {
        let _ctx = self.primary.enter()?;
        Ok(call!("cuStreamSynchronize", stream_synchronize(stream as CUstream))?)
    }

    fn finish(&self, event: Event) -> Result<()> {
        if event == Event::DONE {
            return Ok(());
        }
        let _ctx = self.primary.enter()?;
        Ok(call!("cuEventSynchronize", event_synchronize(self.raw(event)?))?)
    }

    /// Queued only: destroyed by a sweep (the mechanism or trim thread) once complete.
    fn retire(&self, event: Event) {
        self.events.retire(event);
    }

    fn sweep(&self) {
        self.events.sweep(|h| self.complete(h), |h| self.destroy_event(h));
    }

    #[cfg(windows)]
    fn budget(&self) -> Option<std::sync::Arc<dyn crate::backend::Budget>> {
        self.budget.clone().map(|b| b as std::sync::Arc<dyn crate::backend::Budget>)
    }

    fn drain(&self) -> Result<()> {
        self.stream.drain()?;
        Ok(tensorfs::io::drain(self.ordinal)?)
    }

    fn copy_out(&self, va: Va, offset: u64, dst: &mut [u8], after: Event) -> Result<()> {
        let _ctx = self.primary.enter()?;
        let stream = self.stream.raw() as CUstream;
        if after != Event::DONE {
            call!("cuStreamWaitEvent", stream_wait_event(stream, self.raw(after)?, 0))?;
        }
        call!("cuMemcpyDtoHAsync", memcpy_dtoh_async(dst.as_mut_ptr().cast(), va.base + offset, dst.len(), stream))?;
        Ok(call!("cuStreamSynchronize", stream_synchronize(stream))?)
    }

    fn capturing(&self, stream: usize) -> Result<bool> {
        let _ctx = self.primary.enter()?;
        let mut status: c_int = 0;
        call!("cuStreamIsCapturing", stream_is_capturing(stream as CUstream, &mut status))?;
        // CU_STREAM_CAPTURE_STATUS_NONE is 0; ACTIVE and INVALIDATED still capture.
        Ok(status != 0)
    }

    fn relax(&self) {
        // CU_STREAM_CAPTURE_MODE_RELAXED.
        let mut mode: c_int = 2;
        let rc = cuda::driver().map(|d| {
            // SAFETY: an out-param on the stack; the call needs no context.
            unsafe { (d.thread_exchange_stream_capture_mode)(&mut mode) }
        });
        if !matches!(rc, Ok(cuda::SUCCESS)) {
            tracing::warn!("varena: this thread stays in its capture mode: cuThreadExchangeStreamCaptureMode failed");
        }
    }
}

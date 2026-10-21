//! The CUDA VMM [`Backend`]: shareable physical pages (a POSIX fd, a WIN32 handle) mapped into
//! reserved address space, filled pages remapped read-only, and fills through TensorFS's engine
//! into `Dst::Device` ordered on varena's copy stream. Everything runs in the device's primary
//! context, the one torch uses, so a page's pointer is valid there unchanged. Nothing is stored
//! but the device's facts: a page's id is its `CUmemGenericAllocationHandle`, an event's its
//! `CUevent`.

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr};

use parking_lot::Mutex;
use platform::handles::Owned;
use tensorfs::io::cuda::{self, AccessDesc, AllocationProp, CUcontext, CUdevice, CUevent, CUstream, CtxGuard};
use tensorfs::io::Dst;

use crate::backend::{Access, Backend, Event, Physical, Va};
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
    ($op:literal, $f:ident($($arg:expr),* $(,)?)) => {
        // SAFETY: the documented driver ABI, on live handles of this backend's context; every
        // out-param points at a live local.
        cuda::check($op, unsafe { (cuda::driver()?.$f)($($arg),*) })
    };
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

/// One CUDA device's VMM mechanism.
pub struct CudaBackend {
    /// Host registrations (`register_host`) by address. They go before the context does.
    hosts: Mutex<HashMap<usize, cuda::HostPin>>,
    ordinal: i32,
    /// Varena's copy stream: fills are ordered on it. Dropped (synchronized) before the context.
    stream: cuda::Stream,
    primary: Primary,
    prop: AllocationProp,
    granularity: u64,
    capacity: u64,
    nvml: Option<platform::gpu::Device>,
    /// The device's used bytes just before `open` made the context.
    before: Option<u64>,
}

// SAFETY: driver handles are opaque and every use enters their context; `prop`'s only pointer is
// the null WIN32 metadata.
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
        let prop = AllocationProp::device(device, HANDLES);
        let _ctx = primary.enter()?;
        let mut granularity = 0usize;
        call!("cuMemGetAllocationGranularity", mem_get_allocation_granularity(&mut granularity, &prop, cuda::MEM_GRANULARITY_MINIMUM))?;
        let (mut free, mut total) = (0usize, 0usize);
        call!("cuMemGetInfo", mem_get_info(&mut free, &mut total))?;
        drop(_ctx);
        Ok(CudaBackend { hosts: Mutex::new(HashMap::new()), ordinal, stream, primary, prop, granularity: granularity as u64, capacity: total as u64, nvml, before })
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

/// The device's used bytes as NVML reads them; `None` when NVML can't.
fn used(nvml: Option<&platform::gpu::Device>) -> Option<u64> {
    let reading = nvml?.read().ok()?;
    Some(reading.total.saturating_sub(reading.free))
}

fn raw(event: Event) -> CUevent {
    event.0 as CUevent
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
    /// per-process numbers; a container's pids are the host's), the device's used bytes since
    /// just before `open` made the context, which also move with other processes.
    fn process_used(&self) -> Result<u64> {
        let (Some(nvml), Some(before)) = (&self.nvml, self.before) else {
            return Err(Error::Io("NVML is unavailable: no process bytes".into()));
        };
        let reading = nvml.read()?;
        Ok(match reading.processes.as_ref().and_then(|p| p.get(&std::process::id())) {
            Some(&bytes) => bytes,
            None => reading.total.saturating_sub(reading.free).saturating_sub(before),
        })
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

    fn destroy(&self, page: Physical) {
        self.logged("cuMemRelease", || Ok(call!("cuMemRelease", mem_release(page.id))?));
    }

    fn map(&self, va: Va, offset: u64, page: &Physical, access: Access) -> Result<()> {
        self.aligned("offset", offset)?;
        if offset.checked_add(page.len).is_none_or(|end| end > va.len) {
            return Err(Error::Invalid(format!("{} bytes at {offset} outside a {} byte reservation", page.len, va.len)));
        }
        let ptr = va.base + offset;
        let _ctx = self.primary.enter()?;
        call!("cuMemMap", mem_map(ptr, page.len as usize, 0, page.id, 0))?;
        self.access(ptr, page.len, access).inspect_err(|_| {
            // SAFETY: the mapping just made, with nothing ordered on it yet.
            unsafe { cuda::driver().map(|d| (d.mem_unmap)(ptr, page.len as usize)) }.ok();
        })
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
        Ok(Event(event as u64))
    }

    fn query(&self, event: Event) -> Result<bool> {
        if event == Event::DONE {
            return Ok(true);
        }
        let _ctx = self.primary.enter()?;
        // SAFETY: a live event of this context.
        match unsafe { (cuda::driver()?.event_query)(raw(event)) } {
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
        call!("cuEventElapsedTime", event_elapsed_time(&mut ms, raw(from), raw(to)))?;
        Ok(Some((f64::from(ms.max(0.0)) * 1e6) as u64))
    }

    fn wait(&self, event: Event, stream: usize) -> Result<()> {
        if event == Event::DONE {
            return Ok(());
        }
        let _ctx = self.primary.enter()?;
        Ok(call!("cuStreamWaitEvent", stream_wait_event(stream as CUstream, raw(event), 0))?)
    }

    fn retire(&self, event: Event) {
        if event != Event::DONE {
            // An event still pending is freed by the driver once it completes.
            self.logged("cuEventDestroy", || Ok(call!("cuEventDestroy", event_destroy(raw(event)))?));
        }
    }

    fn drain(&self) -> Result<()> {
        self.stream.drain()?;
        Ok(tensorfs::io::drain(self.ordinal)?)
    }

    /// Read-only (the tier's mapping is), in the primary context, so TensorFS's reads see
    /// registered memory and copy directly.
    fn register_host(&self, ptr: u64, len: u64) -> Result<()> {
        let _ctx = self.primary.enter()?;
        // SAFETY: the tier keeps the range mapped until `unregister_host`.
        let pin = unsafe { cuda::pin_region(ptr as *mut u8, len as usize, true) }?;
        self.hosts.lock().insert(ptr as usize, pin);
        Ok(())
    }

    /// A failed release keeps the registration, so its owner keeps the memory mapped and retries.
    fn unregister_host(&self, ptr: u64) -> Result<()> {
        let mut hosts = self.hosts.lock();
        let pin = hosts.get_mut(&(ptr as usize)).ok_or_else(|| Error::Invalid(format!("{ptr:#x} is not registered")))?;
        pin.release()?;
        hosts.remove(&(ptr as usize));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hosted runners have no NVIDIA driver; run there with `--ignored`, never where a GPU is.
    #[test]
    #[ignore = "a machine without a CUDA driver only"]
    fn no_driver_is_a_typed_error() {
        match CudaBackend::open(0) {
            Err(Error::Device { op: "load the driver", .. }) => {}
            Err(e) => unreachable!("expected no driver, got {e}"),
            Ok(_) => unreachable!("this machine has a CUDA driver"),
        }
    }
}

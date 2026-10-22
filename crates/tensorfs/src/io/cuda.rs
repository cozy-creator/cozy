//! The CUDA VMM [`Backend`]: shareable physical pages (a POSIX fd, a WIN32 handle) mapped into
//! reserved address space, filled pages remapped read-only, and fills through TensorFS's engine
//! into `Dst::Device` ordered on varena's copy stream. Everything runs in the device's primary
//! context, the one torch uses, so a page's pointer is valid there unchanged. Nothing is stored
//! but the device's facts: a page's id is its `CUmemGenericAllocationHandle`, an event's its
//! `CUevent`.

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr};

<<<<<<<< HEAD:crates/tensorfs/src/io/cuda.rs
#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    #[error("CUDA unavailable: {0}")]
    CudaUnavailable(String),
    #[error("{op}: CUDA {code} ({name})")]
    Cuda {
        op: &'static str,
        code: i32,
        name: String,
    },
}
pub type Result<T> = std::result::Result<T, Error>;

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
/// `CUmemGenericAllocationHandle`: one physical VRAM allocation.
pub type Handle = u64;
========
use parking_lot::Mutex;
use platform::handles::Owned;
use tensorfs::io::cuda::{self, AccessDesc, AllocationProp, CUcontext, CUdevice, CUevent, CUstream, CtxGuard};
use tensorfs::io::Dst;

use crate::backend::{Access, Backend, Event, Physical, Va};
use crate::{Error, GpuId, Result};
>>>>>>>> varena-one/integration:crates/varena/src/cuda.rs

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

<<<<<<<< HEAD:crates/tensorfs/src/io/cuda.rs
impl AllocationProp {
    /// Pinned device memory on `device`; `handles`: another process may hold it by a POSIX fd.
    pub fn device(device: CUdevice, handles: u32) -> Self {
        AllocationProp {
            kind: MEM_ALLOCATION_TYPE_PINNED,
            requested_handle_types: handles,
            location: MemLocation {
                kind: MEM_LOCATION_TYPE_DEVICE,
                id: device,
            },
            win32_handle_meta_data: std::ptr::null_mut(),
            alloc_flags: AllocFlags::default(),
========
impl Drop for Primary {
    fn drop(&mut self) {
        if let Ok(d) = cuda::driver() {
            // SAFETY: releases the reference `open` retained.
            unsafe { (d.primary_ctx_release)(self.device) };
>>>>>>>> varena-one/integration:crates/varena/src/cuda.rs
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

<<<<<<<< HEAD:crates/tensorfs/src/io/cuda.rs
type F<Args, R = CUresult> = unsafe extern "C" fn(Args) -> R;

pub struct Driver {
    _lib: libloading::Library,
    pub init: unsafe extern "C" fn(c_uint) -> CUresult,
    pub get_error_name: unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult,
    pub device_get: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult,
    pub device_get_count: F<*mut c_int>,
    pub device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult,
    pub device_get_uuid: unsafe extern "C" fn(*mut [u8; 16], CUdevice) -> CUresult,
    pub primary_ctx_retain: unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult,
    pub ctx_get_current: F<*mut CUcontext>,
    pub ctx_set_current: F<CUcontext>,
    pub ctx_synchronize: unsafe extern "C" fn() -> CUresult,
    pub primary_ctx_release: F<CUdevice>,
    pub mem_get_info: unsafe extern "C" fn(*mut usize, *mut usize) -> CUresult,
    pub stream_create: unsafe extern "C" fn(*mut CUstream, c_uint) -> CUresult,
    pub stream_destroy: F<CUstream>,
    pub stream_synchronize: F<CUstream>,
    pub stream_wait_event: unsafe extern "C" fn(CUstream, CUevent, c_uint) -> CUresult,
    pub event_create: unsafe extern "C" fn(*mut CUevent, c_uint) -> CUresult,
    pub event_destroy: F<CUevent>,
    pub event_record: unsafe extern "C" fn(CUevent, CUstream) -> CUresult,
    pub event_query: F<CUevent>,
    pub event_synchronize: F<CUevent>,
    pub event_elapsed_time: unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult,
    pub memcpy_htod_async:
        unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult,
    pub memcpy_dtoh_async:
        unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult,
    pub memcpy_dtod_async:
        unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUresult,
    pub memcpy_peer_async: unsafe extern "C" fn(
        CUdeviceptr,
        CUcontext,
        CUdeviceptr,
        CUcontext,
        usize,
        CUstream,
    ) -> CUresult,
    pub memset_d8_async: unsafe extern "C" fn(CUdeviceptr, u8, usize, CUstream) -> CUresult,
    pub launch_host_func:
        unsafe extern "C" fn(CUstream, unsafe extern "C" fn(*mut c_void), *mut c_void) -> CUresult,
    pub module_load_data: unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
    pub module_get_function:
        unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,
    pub module_unload: F<CUmodule>,
    pub launch_kernel: unsafe extern "C" fn(
        CUfunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CUstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CUresult,
    pub mem_retain_allocation_handle: unsafe extern "C" fn(*mut Handle, *mut c_void) -> CUresult,
    pub memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    pub mem_host_register: unsafe extern "C" fn(*mut c_void, usize, c_uint) -> CUresult,
    pub mem_host_unregister: F<*mut c_void>,
    pub mem_host_alloc: unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CUresult,
    pub mem_free_host: F<*mut c_void>,
    pub mem_alloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    pub pointer_get_attribute: unsafe extern "C" fn(*mut c_void, c_int, CUdeviceptr) -> CUresult,
    pub mem_free: F<CUdeviceptr>,
    pub mem_get_allocation_granularity:
        unsafe extern "C" fn(*mut usize, *const AllocationProp, u32) -> CUresult,
    pub mem_address_reserve:
        unsafe extern "C" fn(*mut CUdeviceptr, usize, usize, CUdeviceptr, u64) -> CUresult,
    pub mem_address_free: unsafe extern "C" fn(CUdeviceptr, usize) -> CUresult,
    pub mem_create:
        unsafe extern "C" fn(*mut Handle, usize, *const AllocationProp, u64) -> CUresult,
    pub mem_release: F<Handle>,
    pub mem_map: unsafe extern "C" fn(CUdeviceptr, usize, usize, Handle, u64) -> CUresult,
    pub mem_unmap: unsafe extern "C" fn(CUdeviceptr, usize) -> CUresult,
    pub mem_set_access:
        unsafe extern "C" fn(CUdeviceptr, usize, *const AccessDesc, usize) -> CUresult,
    pub mem_export: unsafe extern "C" fn(*mut c_void, Handle, u32, u64) -> CUresult,
    pub mem_import: unsafe extern "C" fn(*mut Handle, *mut c_void, u32) -> CUresult,
}

// The table is immutable after load and the driver entry points are thread-safe.
unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

static DRIVER: OnceLock<Result<Driver>> = OnceLock::new();

/// The process-wide driver. `Err` when there is no usable CUDA driver.
pub fn driver() -> Result<&'static Driver> {
    match DRIVER.get_or_init(Driver::load) {
        Ok(d) => Ok(d),
        Err(error) => Err(error.clone()),
    }
}

pub fn available() -> bool {
    driver().is_ok()
}

macro_rules! load {
    ($lib:expr, $name:literal) => {
        // SAFETY: the field types above are the documented CUDA driver ABI for these names.
        *unsafe { $lib.get::<_>($name) }.map_err(|e| {
            Error::CudaUnavailable(format!("{}: {e}", String::from_utf8_lossy($name)))
        })?
    };
}

impl Driver {
    fn load() -> Result<Driver> {
        // SAFETY: dlopen of the NVIDIA driver library; no initializer of ours runs.
        #[cfg(windows)]
        let names = ["nvcuda.dll"];
        #[cfg(not(windows))]
        let names = ["libcuda.so.1", "libcuda.so"];
        let mut library = None;
        let mut failures = Vec::new();
        for name in names {
            match unsafe { libloading::Library::new(name) } {
                Ok(lib) => {
                    library = Some(lib);
                    break;
                }
                Err(error) => failures.push(error.to_string()),
            }
        }
        let lib = library.ok_or_else(|| Error::CudaUnavailable(failures.join("; ")))?;
        let d = Driver {
            init: load!(lib, b"cuInit"),
            get_error_name: load!(lib, b"cuGetErrorName"),
            device_get: load!(lib, b"cuDeviceGet"),
            device_get_count: load!(lib, b"cuDeviceGetCount"),
            device_get_attribute: load!(lib, b"cuDeviceGetAttribute"),
            device_get_uuid: load!(lib, b"cuDeviceGetUuid_v2"),
            primary_ctx_retain: load!(lib, b"cuDevicePrimaryCtxRetain"),
            ctx_get_current: load!(lib, b"cuCtxGetCurrent"),
            ctx_set_current: load!(lib, b"cuCtxSetCurrent"),
            ctx_synchronize: load!(lib, b"cuCtxSynchronize"),
            primary_ctx_release: load!(lib, b"cuDevicePrimaryCtxRelease_v2"),
            mem_get_info: load!(lib, b"cuMemGetInfo_v2"),
            stream_create: load!(lib, b"cuStreamCreate"),
            stream_destroy: load!(lib, b"cuStreamDestroy_v2"),
            stream_synchronize: load!(lib, b"cuStreamSynchronize"),
            stream_wait_event: load!(lib, b"cuStreamWaitEvent"),
            event_create: load!(lib, b"cuEventCreate"),
            event_destroy: load!(lib, b"cuEventDestroy_v2"),
            event_record: load!(lib, b"cuEventRecord"),
            event_query: load!(lib, b"cuEventQuery"),
            event_synchronize: load!(lib, b"cuEventSynchronize"),
            event_elapsed_time: load!(lib, b"cuEventElapsedTime"),
            memcpy_htod_async: load!(lib, b"cuMemcpyHtoDAsync_v2"),
            memcpy_dtoh_async: load!(lib, b"cuMemcpyDtoHAsync_v2"),
            memcpy_dtod_async: load!(lib, b"cuMemcpyDtoDAsync_v2"),
            memcpy_peer_async: load!(lib, b"cuMemcpyPeerAsync"),
            memset_d8_async: load!(lib, b"cuMemsetD8Async"),
            launch_host_func: load!(lib, b"cuLaunchHostFunc"),
            module_load_data: load!(lib, b"cuModuleLoadData"),
            module_get_function: load!(lib, b"cuModuleGetFunction"),
            module_unload: load!(lib, b"cuModuleUnload"),
            launch_kernel: load!(lib, b"cuLaunchKernel"),
            mem_retain_allocation_handle: load!(lib, b"cuMemRetainAllocationHandle"),
            memcpy_dtoh: load!(lib, b"cuMemcpyDtoH_v2"),
            mem_host_register: load!(lib, b"cuMemHostRegister_v2"),
            mem_host_unregister: load!(lib, b"cuMemHostUnregister"),
            mem_host_alloc: load!(lib, b"cuMemHostAlloc"),
            mem_free_host: load!(lib, b"cuMemFreeHost"),
            mem_alloc: load!(lib, b"cuMemAlloc_v2"),
            pointer_get_attribute: load!(lib, b"cuPointerGetAttribute"),
            mem_free: load!(lib, b"cuMemFree_v2"),
            mem_get_allocation_granularity: load!(lib, b"cuMemGetAllocationGranularity"),
            mem_address_reserve: load!(lib, b"cuMemAddressReserve"),
            mem_address_free: load!(lib, b"cuMemAddressFree"),
            mem_create: load!(lib, b"cuMemCreate"),
            mem_release: load!(lib, b"cuMemRelease"),
            mem_map: load!(lib, b"cuMemMap"),
            mem_unmap: load!(lib, b"cuMemUnmap"),
            mem_set_access: load!(lib, b"cuMemSetAccess"),
            mem_export: load!(lib, b"cuMemExportToShareableHandle"),
            mem_import: load!(lib, b"cuMemImportFromShareableHandle"),
            _lib: lib,
        };
        // SAFETY: documented as safe to call repeatedly; flags must be 0.
        let rc = unsafe { (d.init)(0) };
        if rc != SUCCESS {
            return Err(Error::Cuda {
                op: "cuInit",
                code: rc,
                name: d.error_name(rc),
            });
========
/// The stream pages are saved to the host on: the lowest priority, so a save never goes ahead of a
/// fill. Synchronized and destroyed before the context is released.
struct Saves {
    raw: usize,
    ctx: usize,
}

impl Drop for Saves {
    fn drop(&mut self) {
        if let (Ok(d), Ok(_ctx)) = (cuda::driver(), CtxGuard::enter(self.ctx as CUcontext)) {
            // SAFETY: our stream, in its context; nothing is queued on it after this.
            unsafe {
                (d.stream_synchronize)(self.raw as CUstream);
                (d.stream_destroy)(self.raw as CUstream);
            }
>>>>>>>> varena-one/integration:crates/varena/src/cuda.rs
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
    saves: Saves,
    primary: Primary,
    prop: AllocationProp,
    granularity: u64,
    capacity: u64,
    nvml: Option<platform::gpu::Device>,
    /// The device's used bytes `open` measured across making the context.
    context: u64,
    /// The device page-locks host memory it may only read (attribute 113).
    read_only_register: bool,
    /// What `prop`'s WIN32 metadata points at, for as long as pages are created.
    #[cfg(windows)]
    _security: Box<SecurityAttributes>,
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
        let read_only_register = cuda::attribute(device, cuda::ATTR_READ_ONLY_HOST_REGISTER_SUPPORTED)? != 0;
        #[allow(unused_mut)]
        let mut prop = AllocationProp::device(device, HANDLES);
        #[cfg(windows)]
        let security = Box::new(SecurityAttributes { length: std::mem::size_of::<SecurityAttributes>() as u32, descriptor: std::ptr::null_mut(), inherit: 0 });
        #[cfg(windows)]
        {
            prop.win32_handle_meta_data = (&*security as *const SecurityAttributes).cast_mut().cast();
        }
        let _ctx = primary.enter()?;
        let mut granularity = 0usize;
        call!("cuMemGetAllocationGranularity", mem_get_allocation_granularity(&mut granularity, &prop, cuda::MEM_GRANULARITY_MINIMUM))?;
        let (mut free, mut total) = (0usize, 0usize);
        call!("cuMemGetInfo", mem_get_info(&mut free, &mut total))?;
        let (mut least, mut greatest): (c_int, c_int) = (0, 0);
        call!("cuCtxGetStreamPriorityRange", ctx_get_stream_priority_range(&mut least, &mut greatest))?;
        let mut raw: CUstream = std::ptr::null_mut();
        call!("cuStreamCreateWithPriority", stream_create_with_priority(&mut raw, cuda::STREAM_NON_BLOCKING, least))?;
        let saves = Saves { raw: raw as usize, ctx: primary.ctx };
        drop(_ctx);
        Ok(CudaBackend {
            hosts: Mutex::new(HashMap::new()),
            ordinal,
            stream,
            saves,
            primary,
            prop,
            granularity: granularity as u64,
            capacity: total as u64,
            nvml,
            context,
            read_only_register,
            #[cfg(windows)]
            _security: security,
        })
    }

<<<<<<<< HEAD:crates/tensorfs/src/io/cuda.rs
/// Makes `ctx` current on this thread for the guard's life, then restores the caller's
/// context. Caller threads belong to someone else (the executor's framework keeps its own
/// current device), so TensorFS never leaves a different context bound behind it.
pub struct CtxGuard {
    prev: Option<CUcontext>,
}
========
>>>>>>>> varena-one/integration:crates/varena/src/cuda.rs

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

<<<<<<<< HEAD:crates/tensorfs/src/io/cuda.rs
/// A device's PRIMARY context: the one the CUDA runtime uses, so TensorFS's pointers, streams and
/// events are valid in the executor's framework unchanged. Retained for the process's life.
pub fn primary_context(ordinal: c_int) -> Result<(CUdevice, CUcontext)> {
    let d = driver()?;
    let mut dev: CUdevice = 0;
    let mut ctx: CUcontext = std::ptr::null_mut();
    // SAFETY: out-params; a bad ordinal is a driver error, not UB.
    check("cuDeviceGet", unsafe { (d.device_get)(&mut dev, ordinal) })?;
    // SAFETY: retains (does not create a second) primary context.
    check("cuDevicePrimaryCtxRetain", unsafe {
        (d.primary_ctx_retain)(&mut ctx, dev)
    })?;
    Ok((dev, ctx))
========
/// The device's used bytes as NVML reads them; `None` when NVML can't.
fn used(nvml: Option<&platform::gpu::Device>) -> Option<u64> {
    let reading = nvml?.read().ok()?;
    Some(reading.total.saturating_sub(reading.free))
>>>>>>>> varena-one/integration:crates/varena/src/cuda.rs
}

fn raw(event: Event) -> CUevent {
    event.0 as CUevent
}

<<<<<<<< HEAD:crates/tensorfs/src/io/cuda.rs
pub fn uuid(dev: CUdevice) -> Result<String> {
    let d = driver()?;
    let mut u = [0u8; 16];
    // SAFETY: 16-byte out-param.
    check("cuDeviceGetUuid", unsafe {
        (d.device_get_uuid)(&mut u, dev)
    })?;
    let h: String = u.iter().map(|b| format!("{b:02x}")).collect();
    Ok(format!(
        "GPU-{}-{}-{}-{}-{}",
        &h[0..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..32]
    ))
========
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

    /// One driver copy on the save stream, after `after`, then a wait for that stream alone.
    fn save(&self, va: Va, offset: u64, host: &mut [u8], after: Event) -> Result<()> {
        if offset.checked_add(host.len() as u64).is_none_or(|end| end > va.len) {
            return Err(Error::Invalid(format!("{} bytes at {offset} outside a {} byte reservation", host.len(), va.len)));
        }
        let _ctx = self.primary.enter()?;
        let stream = self.saves.raw as CUstream;
        if after != Event::DONE {
            call!("cuStreamWaitEvent", stream_wait_event(stream, raw(after), 0))?;
        }
        call!("cuMemcpyDtoHAsync", memcpy_dtoh_async(host.as_mut_ptr().cast(), va.base + offset, host.len(), stream))?;
        Ok(call!("cuStreamSynchronize", stream_synchronize(stream))?)
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
        let _call = tracing::trace_span!("cuEventQuery").entered();
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

    fn synchronize(&self, stream: usize) -> Result<()> {
        let _ctx = self.primary.enter()?;
        Ok(call!("cuStreamSynchronize", stream_synchronize(stream as CUstream))?)
    }

    fn finish(&self, event: Event) -> Result<()> {
        if event == Event::DONE {
            return Ok(());
        }
        let _ctx = self.primary.enter()?;
        Ok(call!("cuEventSynchronize", event_synchronize(raw(event)))?)
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

    /// Read-only where the device can (attribute 113), else read-write, which a read-only mapping
    /// refuses: the caller then stages. A range already registered by another owner is refused,
    /// never borrowed; one this backend holds stays as it is.
    fn register_host(&self, ptr: u64, len: u64) -> Result<()> {
        let _ctx = self.primary.enter()?;
        let mut hosts = self.hosts.lock();
        if hosts.contains_key(&(ptr as usize)) {
            return Ok(());
        }
        // SAFETY: a lookup of the driver's registrations; no memory is touched.
        if unsafe { cuda::registered(ptr as *const u8) }? {
            return Err(Error::Invalid(format!("{ptr:#x} is registered by another owner")));
        }
        // SAFETY: the tier keeps the range mapped until `unregister_host`.
        let pin = unsafe { cuda::pin_region(ptr as *mut u8, len as usize, self.read_only_register) }?;
        hosts.insert(ptr as usize, pin);
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
>>>>>>>> varena-one/integration:crates/varena/src/cuda.rs
}

/// A caller-owned consumer stream for a DLPack handoff. The producer orders its work onto
/// `raw()`; the read engine then transfers on that stream or waits for it.
pub struct Stream {
    device: CUdevice,
    ctx: CUcontext,
    raw: CUstream,
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    pub fn new(device: i32) -> Result<Stream> {
        let (device, ctx) = primary_context(device)?;
        let mut raw = std::ptr::null_mut();
        let result = (|| {
            let _ctx = CtxGuard::enter(ctx)?;
            check("cuStreamCreate", unsafe {
                (driver()?.stream_create)(&mut raw, STREAM_NON_BLOCKING)
            })
        })();
        if let Err(error) = result {
            if let Ok(driver) = driver() {
                unsafe { (driver.primary_ctx_release)(device) };
            }
            return Err(error);
        }
        Ok(Stream { device, ctx, raw })
    }

    pub fn raw(&self) -> usize {
        self.raw as usize
    }
    pub fn device(&self) -> i32 {
        self.device
    }
    /// Wait for work previously ordered onto this stream in its retained primary context.
    ///
    /// Only success proves completion. On error, callers must retain this stream and every
    /// producer/allocation owner whose release requires that work to have stopped. Best-effort
    /// destruction is not a substitute for this checked ownership boundary.
    pub fn drain(&self) -> Result<()> {
        let driver = driver()?;
        let _context = CtxGuard::enter(self.ctx)?;
        check("cuStreamSynchronize", unsafe {
            (driver.stream_synchronize)(self.raw)
        })
    }
    pub(crate) fn context(&self) -> CUcontext {
        self.ctx
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if let Ok(driver) = driver() {
            if let Ok(_ctx) = CtxGuard::enter(self.ctx) {
                unsafe {
                    (driver.stream_synchronize)(self.raw);
                    (driver.stream_destroy)(self.raw);
                }
            }
            unsafe { (driver.primary_ctx_release)(self.device) };
        }
    }
}

/// A registered host range. Its memory and context must outlive this guard; copies must be
/// complete before it drops. An existing registration is borrowed and never unregistered.
pub struct HostPin {
    ptr: *mut c_void,
    ctx: CUcontext,
    owned: bool,
}
unsafe impl Send for HostPin {}
unsafe impl Sync for HostPin {}

/// # Safety
/// `ptr..ptr+len` remains mapped until the returned guard drops; the caller retains the
/// current context for the same lifetime and finishes every copy before dropping the guard.
pub unsafe fn pin_region(ptr: *mut u8, len: usize, read_only: bool) -> Result<HostPin> {
    let d = driver()?;
    let mut ctx = std::ptr::null_mut();
    check("cuCtxGetCurrent", unsafe { (d.ctx_get_current)(&mut ctx) })?;
    let flags = MEMHOSTREGISTER_PORTABLE
        | if read_only {
            MEMHOSTREGISTER_READ_ONLY
        } else {
            0
        };
    let rc = unsafe { (d.mem_host_register)(ptr.cast(), len, flags) };
    if rc != ERROR_HOST_MEMORY_ALREADY_REGISTERED {
        check("cuMemHostRegister", rc)?;
    }
    Ok(HostPin {
        ptr: ptr.cast(),
        ctx,
        owned: rc == SUCCESS,
    })
}

/// Whether the current context can copy directly from this page-locked host address.
/// # Safety
/// `ptr` is live host memory; its owner holds the registration during the copy.
pub unsafe fn registered(ptr: *const u8) -> Result<bool> {
    let mut kind: c_int = 0;
    let rc = unsafe {
        (driver()?.pointer_get_attribute)((&mut kind as *mut c_int).cast(), 2, ptr as u64)
    };
    match rc {
        SUCCESS => Ok(kind == 1),
        ERROR_INVALID_VALUE | ERROR_NOT_SUPPORTED => Ok(false),
        _ => check("cuPointerGetAttribute", rc).map(|_| false),
    }
}

impl Drop for HostPin {
    fn drop(&mut self) {
        let _ = self.release();
    }
}

impl HostPin {
    /// Release only this guard's registration. On failure ownership is retained, so a
    /// mapping owner can refuse to punch/unmap it and retry. Borrowed pins are untouched.
    pub fn release(&mut self) -> Result<()> {
        if self.owned {
            let driver = driver()?;
            let _context = CtxGuard::enter(self.ctx)?;
            check("cuMemHostUnregister", unsafe {
                (driver.mem_host_unregister)(self.ptr)
            })?;
            self.owned = false;
        }
        Ok(())
    }
}

//! The CUDA driver API, dlopen'd on first use. The crate builds and its CPU paths run with no
//! driver present; `libcuda.so.1` missing is a typed error, never a panic.
//!
//! One symbol table. Adding a symbol is one field plus one `load!` line.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::sync::OnceLock;

use crate::{Error, Result};

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
/// `CUmemGenericAllocationHandle`: one physical VRAM allocation.
pub type Handle = u64;

pub const SUCCESS: CUresult = 0;
pub const ERROR_INVALID_VALUE: CUresult = 1;
pub const ERROR_OUT_OF_MEMORY: CUresult = 2;
pub const ERROR_HOST_MEMORY_ALREADY_REGISTERED: CUresult = 712;
pub const ERROR_NOT_READY: CUresult = 600;

pub const STREAM_NON_BLOCKING: c_uint = 0x1;
pub const EVENT_DEFAULT: c_uint = 0x0;
pub const EVENT_DISABLE_TIMING: c_uint = 0x2;
pub const MEMHOSTREGISTER_PORTABLE: c_uint = 0x1;
pub const MEMHOSTALLOC_PORTABLE: c_uint = 0x1;

pub const ATTR_VMM_SUPPORTED: c_int = 102;
pub const ATTR_POSIX_FD_SUPPORTED: c_int = 103;

pub const MEM_ALLOCATION_TYPE_PINNED: u32 = 1;
pub const MEM_LOCATION_TYPE_DEVICE: u32 = 1;
pub const MEM_HANDLE_TYPE_NONE: u32 = 0;
pub const MEM_HANDLE_TYPE_POSIX_FD: u32 = 1;
pub const MEM_ACCESS_PROT_READWRITE: u32 = 3;
pub const MEM_GRANULARITY_MINIMUM: u32 = 0;

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct MemLocation {
    pub kind: u32,
    pub id: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct AllocFlags {
    pub compression_type: u8,
    pub gpu_direct_rdma_capable: u8,
    pub usage: u16,
    pub reserved: [u8; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AllocationProp {
    pub kind: u32,
    pub requested_handle_types: u32,
    pub location: MemLocation,
    pub win32_handle_meta_data: *mut c_void,
    pub alloc_flags: AllocFlags,
}

impl AllocationProp {
    /// Pinned device memory on `device`, POSIX-fd exportable when the device supports it.
    pub fn device(device: CUdevice, exportable: bool) -> Self {
        AllocationProp {
            kind: MEM_ALLOCATION_TYPE_PINNED,
            requested_handle_types: if exportable {
                MEM_HANDLE_TYPE_POSIX_FD
            } else {
                MEM_HANDLE_TYPE_NONE
            },
            location: MemLocation {
                kind: MEM_LOCATION_TYPE_DEVICE,
                id: device,
            },
            win32_handle_meta_data: std::ptr::null_mut(),
            alloc_flags: AllocFlags::default(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AccessDesc {
    pub location: MemLocation,
    pub flags: u32,
}

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
    pub mem_get_info: unsafe extern "C" fn(*mut usize, *mut usize) -> CUresult,
    pub stream_create: unsafe extern "C" fn(*mut CUstream, c_uint) -> CUresult,
    pub stream_destroy: F<CUstream>,
    pub stream_synchronize: F<CUstream>,
    pub stream_wait_event: unsafe extern "C" fn(CUstream, CUevent, c_uint) -> CUresult,
    pub launch_host_func:
        unsafe extern "C" fn(CUstream, unsafe extern "C" fn(*mut c_void), *mut c_void) -> CUresult,
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
    pub memset_d8_async: unsafe extern "C" fn(CUdeviceptr, u8, usize, CUstream) -> CUresult,
    pub mem_host_register: unsafe extern "C" fn(*mut c_void, usize, c_uint) -> CUresult,
    pub mem_host_unregister: F<*mut c_void>,
    pub mem_host_alloc: unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CUresult,
    pub mem_free_host: F<*mut c_void>,
    pub mem_get_allocation_granularity:
        unsafe extern "C" fn(*mut usize, *const AllocationProp, u32) -> CUresult,
    pub mem_address_reserve:
        unsafe extern "C" fn(*mut CUdeviceptr, usize, usize, CUdeviceptr, u64) -> CUresult,
    pub mem_address_free: unsafe extern "C" fn(CUdeviceptr, usize) -> CUresult,
    pub mem_create: unsafe extern "C" fn(*mut Handle, usize, *const AllocationProp, u64) -> CUresult,
    pub mem_release: F<Handle>,
    pub mem_map: unsafe extern "C" fn(CUdeviceptr, usize, usize, Handle, u64) -> CUresult,
    pub mem_unmap: unsafe extern "C" fn(CUdeviceptr, usize) -> CUresult,
    pub mem_set_access:
        unsafe extern "C" fn(CUdeviceptr, usize, *const AccessDesc, usize) -> CUresult,
    /// The driver's own answer to "what is mapped here": the one call that reads residency
    /// from the driver instead of the plane's books. Every success needs a `mem_release`.
    pub mem_retain_allocation_handle: unsafe extern "C" fn(*mut Handle, *mut c_void) -> CUresult,
}

// The table is immutable after load and the driver entry points are thread-safe.
unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

static DRIVER: OnceLock<std::result::Result<Driver, String>> = OnceLock::new();

/// The process-wide driver. `Err` when there is no usable CUDA driver.
pub fn driver() -> Result<&'static Driver> {
    match DRIVER.get_or_init(|| Driver::load().map_err(|e| e.to_string())) {
        Ok(d) => Ok(d),
        Err(msg) => Err(Error::CudaUnavailable(msg.clone())),
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
        let lib = unsafe { libloading::Library::new("libcuda.so.1") }
            .or_else(|_| unsafe { libloading::Library::new("libcuda.so") })
            .map_err(|e| Error::CudaUnavailable(format!("dlopen libcuda: {e}")))?;
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
            mem_get_info: load!(lib, b"cuMemGetInfo_v2"),
            stream_create: load!(lib, b"cuStreamCreate"),
            stream_destroy: load!(lib, b"cuStreamDestroy_v2"),
            stream_synchronize: load!(lib, b"cuStreamSynchronize"),
            stream_wait_event: load!(lib, b"cuStreamWaitEvent"),
            launch_host_func: load!(lib, b"cuLaunchHostFunc"),
            event_create: load!(lib, b"cuEventCreate"),
            event_destroy: load!(lib, b"cuEventDestroy_v2"),
            event_record: load!(lib, b"cuEventRecord"),
            event_query: load!(lib, b"cuEventQuery"),
            event_synchronize: load!(lib, b"cuEventSynchronize"),
            event_elapsed_time: load!(lib, b"cuEventElapsedTime"),
            memcpy_htod_async: load!(lib, b"cuMemcpyHtoDAsync_v2"),
            memcpy_dtoh_async: load!(lib, b"cuMemcpyDtoHAsync_v2"),
            memcpy_dtod_async: load!(lib, b"cuMemcpyDtoDAsync_v2"),
            memset_d8_async: load!(lib, b"cuMemsetD8Async"),
            mem_host_register: load!(lib, b"cuMemHostRegister_v2"),
            mem_host_unregister: load!(lib, b"cuMemHostUnregister"),
            mem_host_alloc: load!(lib, b"cuMemHostAlloc"),
            mem_free_host: load!(lib, b"cuMemFreeHost"),
            mem_get_allocation_granularity: load!(lib, b"cuMemGetAllocationGranularity"),
            mem_address_reserve: load!(lib, b"cuMemAddressReserve"),
            mem_address_free: load!(lib, b"cuMemAddressFree"),
            mem_create: load!(lib, b"cuMemCreate"),
            mem_release: load!(lib, b"cuMemRelease"),
            mem_map: load!(lib, b"cuMemMap"),
            mem_unmap: load!(lib, b"cuMemUnmap"),
            mem_set_access: load!(lib, b"cuMemSetAccess"),
            mem_retain_allocation_handle: load!(lib, b"cuMemRetainAllocationHandle"),
            _lib: lib,
        };
        // SAFETY: documented as safe to call repeatedly; flags must be 0.
        let rc = unsafe { (d.init)(0) };
        if rc != SUCCESS {
            return Err(Error::CudaUnavailable(format!("cuInit -> {rc}")));
        }
        Ok(d)
    }

    pub fn error_name(&self, rc: CUresult) -> String {
        let mut p: *const c_char = std::ptr::null();
        // SAFETY: out-param; the driver returns a static string or fails.
        if unsafe { (self.get_error_name)(rc, &mut p) } == SUCCESS && !p.is_null() {
            // SAFETY: a NUL-terminated static string owned by the driver.
            return unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
        }
        format!("CUresult {rc}")
    }
}

/// Map a driver result to the crate error.
#[inline]
pub fn check(op: &'static str, rc: CUresult) -> Result<()> {
    if rc == SUCCESS {
        Ok(())
    } else {
        Err(Error::Cuda {
            op,
            code: rc,
            name: driver().map(|d| d.error_name(rc)).unwrap_or_default(),
        })
    }
}

/// Makes `ctx` current on this thread for the guard's life, then restores the caller's
/// context. Caller threads belong to someone else (the executor's framework keeps its own
/// current device), so the plane never leaves a different context bound behind it.
pub struct CtxGuard {
    prev: Option<CUcontext>,
}

impl CtxGuard {
    // Driver handles are opaque: Rust never dereferences them.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn enter(ctx: CUcontext) -> Result<CtxGuard> {
        let d = driver()?;
        let mut prev: CUcontext = std::ptr::null_mut();
        // SAFETY: out-param of the documented shape.
        check("cuCtxGetCurrent", unsafe { (d.ctx_get_current)(&mut prev) })?;
        if prev == ctx {
            return Ok(CtxGuard { prev: None });
        }
        // SAFETY: `ctx` is a retained primary context.
        check("cuCtxSetCurrent", unsafe { (d.ctx_set_current)(ctx) })?;
        Ok(CtxGuard { prev: Some(prev) })
    }
}

impl Drop for CtxGuard {
    fn drop(&mut self) {
        if let (Some(prev), Ok(d)) = (self.prev, driver()) {
            // SAFETY: restoring the handle the driver reported as current (possibly null).
            unsafe { (d.ctx_set_current)(prev) };
        }
    }
}

/// A device's PRIMARY context: the one the CUDA runtime uses, so plane pointers, streams and
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
}

pub fn attribute(dev: CUdevice, attr: c_int) -> Result<i32> {
    let d = driver()?;
    let mut v: c_int = 0;
    // SAFETY: out-param.
    check("cuDeviceGetAttribute", unsafe {
        (d.device_get_attribute)(&mut v, attr, dev)
    })?;
    Ok(v)
}

pub fn uuid(dev: CUdevice) -> Result<String> {
    let d = driver()?;
    let mut u = [0u8; 16];
    // SAFETY: 16-byte out-param.
    check("cuDeviceGetUuid", unsafe { (d.device_get_uuid)(&mut u, dev) })?;
    let h: String = u.iter().map(|b| format!("{b:02x}")).collect();
    Ok(format!(
        "GPU-{}-{}-{}-{}-{}",
        &h[0..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..32]
    ))
}

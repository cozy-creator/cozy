//! The CUDA driver API, dlopen'd on first use. The crate builds and its CPU paths run with no
//! driver present; `libcuda.so.1` missing is a typed error, never a panic.
//!
//! One symbol table. Adding a symbol is one field plus one `load!` line.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::sync::OnceLock;

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

pub const SUCCESS: CUresult = 0;
pub const ERROR_INVALID_VALUE: CUresult = 1;
pub const ERROR_OUT_OF_MEMORY: CUresult = 2;
pub const ERROR_HOST_MEMORY_ALREADY_REGISTERED: CUresult = 712;
pub const ERROR_NOT_READY: CUresult = 600;
pub const ERROR_NOT_SUPPORTED: CUresult = 801;

pub const STREAM_NON_BLOCKING: c_uint = 0x1;
pub const EVENT_DEFAULT: c_uint = 0x0;
pub const EVENT_DISABLE_TIMING: c_uint = 0x2;
pub const MEMHOSTREGISTER_PORTABLE: c_uint = 0x1;
/// The range is mapped read-only (a sealed layout); the device only reads it.
pub const MEMHOSTREGISTER_READ_ONLY: c_uint = 0x8;
pub const MEMHOSTALLOC_PORTABLE: c_uint = 0x1;

pub const ATTR_VMM_SUPPORTED: c_int = 102;
pub const ATTR_PCI_BUS_ID: c_int = 33;
pub const ATTR_PCI_DEVICE_ID: c_int = 34;
pub const ATTR_PCI_DOMAIN_ID: c_int = 50;
pub const ATTR_POSIX_FD_SUPPORTED: c_int = 103;

pub const MEM_ALLOCATION_TYPE_PINNED: u32 = 1;
pub const MEM_LOCATION_TYPE_DEVICE: u32 = 1;
pub const MEM_HANDLE_TYPE_NONE: u32 = 0;
pub const MEM_HANDLE_TYPE_POSIX_FD: u32 = 1;
pub const MEM_ACCESS_PROT_READ: u32 = 1;
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
/// current device), so TensorFS never leaves a different context bound behind it.
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
            check("cuMemHostUnregister", unsafe { (driver.mem_host_unregister)(self.ptr) })?;
            self.owned = false;
        }
        Ok(())
    }
}

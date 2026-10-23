//! Windows: this process's video-memory budget on one adapter, through DXGI (`dxgi.dll`, opened
//! at run time): what VidMm lets the process keep resident there, its usage, and an event at
//! every change (`IDXGIAdapter3::RegisterVideoMemoryBudgetChangeNotificationEvent`). The adapter
//! is the CUDA device's by LUID (`cuDeviceGetLuid`). windows-sys has no DXGI interfaces, so the
//! few methods used are called through their vtable slots (dxgi1_4.h order).
#![allow(unsafe_code)]
use crate::{Error, Result};
use libloading::Library;
use std::ffi::c_void;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use windows_sys::Win32::Foundation::{HANDLE, WAIT_OBJECT_0};
use windows_sys::Win32::System::Threading::{
    CreateEventW, INFINITE, SetEvent, WaitForMultipleObjects,
};
use windows_sys::core::GUID;

const IID_IDXGI_FACTORY1: GUID = GUID::from_u128(0x770aae78_f26f_4dba_a829_253c83d1b387);
const IID_IDXGI_ADAPTER3: GUID = GUID::from_u128(0x645967a4_1392_4310_a798_8053ce3e93fd);
/// `DXGI_ERROR_NOT_FOUND`: no adapter at that index.
const NOT_FOUND: i32 = 0x887A0002_u32 as i32;
/// `DXGI_MEMORY_SEGMENT_GROUP_LOCAL`: the adapter's own memory.
const LOCAL: i32 = 0;

// Vtable slots: IUnknown 0-2, IDXGIObject 3-6, IDXGIFactory 7-11, IDXGIFactory1 12-13;
// IDXGIAdapter 7-9, IDXGIAdapter1 10, IDXGIAdapter2 11, IDXGIAdapter3 12-17.
const QUERY_INTERFACE: usize = 0;
const RELEASE: usize = 2;
const ENUM_ADAPTERS1: usize = 12;
const GET_DESC1: usize = 10;
const QUERY_VIDEO_MEMORY_INFO: usize = 14;
const REGISTER_BUDGET_EVENT: usize = 16;
const UNREGISTER_BUDGET_EVENT: usize = 17;

type QueryInterface = unsafe extern "system" fn(*mut c_void, *const GUID, *mut *mut c_void) -> i32;
type Release = unsafe extern "system" fn(*mut c_void) -> u32;
type EnumAdapters1 = unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> i32;
type GetDesc1 = unsafe extern "system" fn(*mut c_void, *mut AdapterDesc1) -> i32;
type QueryVideoMemoryInfo = unsafe extern "system" fn(*mut c_void, u32, i32, *mut VideoMemoryInfo) -> i32;
type RegisterBudgetEvent = unsafe extern "system" fn(*mut c_void, HANDLE, *mut u32) -> i32;
type UnregisterBudgetEvent = unsafe extern "system" fn(*mut c_void, u32);
type CreateFactory1 = unsafe extern "system" fn(*const GUID, *mut *mut c_void) -> i32;

/// `DXGI_ADAPTER_DESC1`.
#[repr(C)]
struct AdapterDesc1 {
    description: [u16; 128],
    vendor: u32,
    device: u32,
    subsystem: u32,
    revision: u32,
    dedicated_video: usize,
    dedicated_system: usize,
    shared_system: usize,
    luid_low: u32,
    luid_high: i32,
    flags: u32,
}

/// `DXGI_QUERY_VIDEO_MEMORY_INFO`.
#[repr(C)]
#[derive(Default)]
struct VideoMemoryInfo {
    budget: u64,
    usage: u64,
    available_for_reservation: u64,
    reservation: u64,
}

/// Method `index` of COM object `this`, as `F`.
///
/// # Safety
/// `this` is a live COM object whose vtable has slot `index` of signature `F`.
unsafe fn method<F: Copy>(this: *mut c_void, index: usize) -> F {
    // SAFETY: the caller's promise: a COM object starts with its vtable pointer.
    unsafe {
        let vtable = *(this as *const *const usize);
        std::mem::transmute_copy(&*vtable.add(index))
    }
}

/// One COM reference, released on drop.
struct Com(*mut c_void);

impl Drop for Com {
    fn drop(&mut self) {
        // SAFETY: a reference this value owns.
        unsafe { method::<Release>(self.0, RELEASE)(self.0) };
    }
}

fn failed(op: &'static str, hr: i32) -> Error {
    Error::Gpu { op, detail: format!("HRESULT {:#010x}", hr as u32) }
}

/// One DXGI adapter: its name, its LUID as `cuDeviceGetLuid` writes it, its dedicated memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Adapter {
    pub name: String,
    pub luid: [u8; 8],
    pub dedicated: u64,
}

fn factory() -> Result<(Library, Com)> {
    // SAFETY: DXGI's library, loaded by its documented name.
    let dxgi = unsafe { Library::new("dxgi.dll") }.map_err(|e| Error::Gpu { op: "load DXGI", detail: e.to_string() })?;
    // SAFETY: the export's documented signature (dxgi.h).
    let create = *unsafe { dxgi.get::<CreateFactory1>(b"CreateDXGIFactory1\0") }.map_err(|e| Error::Gpu { op: "load DXGI", detail: e.to_string() })?;
    let mut factory = std::ptr::null_mut();
    // SAFETY: an IID and a writable interface pointer.
    let hr = unsafe { create(&IID_IDXGI_FACTORY1, &mut factory) };
    if hr < 0 {
        return Err(failed("CreateDXGIFactory1", hr));
    }
    Ok((dxgi, Com(factory)))
}

/// Every adapter DXGI enumerates, with its own reference.
fn each(factory: &Com) -> Result<Vec<(Adapter, Com)>> {
    let mut out = vec![];
    for index in 0.. {
        let mut adapter = std::ptr::null_mut();
        // SAFETY: a live IDXGIFactory1 and a writable interface pointer.
        let hr = unsafe { method::<EnumAdapters1>(factory.0, ENUM_ADAPTERS1)(factory.0, index, &mut adapter) };
        if hr == NOT_FOUND {
            break;
        }
        if hr < 0 {
            return Err(failed("IDXGIFactory1::EnumAdapters1", hr));
        }
        let adapter = Com(adapter);
        // SAFETY: plain-old-data the call fills.
        let mut desc: AdapterDesc1 = unsafe { std::mem::zeroed() };
        // SAFETY: a live IDXGIAdapter1 and a description of the documented layout.
        let hr = unsafe { method::<GetDesc1>(adapter.0, GET_DESC1)(adapter.0, &mut desc) };
        if hr < 0 {
            return Err(failed("IDXGIAdapter1::GetDesc1", hr));
        }
        let len = desc.description.iter().position(|c| *c == 0).unwrap_or(desc.description.len());
        let mut luid = [0u8; 8];
        luid[..4].copy_from_slice(&desc.luid_low.to_le_bytes());
        luid[4..].copy_from_slice(&desc.luid_high.to_le_bytes());
        let name = String::from_utf16_lossy(&desc.description[..len]);
        out.push((Adapter { name, luid, dedicated: desc.dedicated_video as u64 }, adapter));
    }
    Ok(out)
}

/// Every adapter DXGI names (the GPUs, and Microsoft's software adapter).
pub fn adapters() -> Result<Vec<Adapter>> {
    let (_dxgi, factory) = factory()?;
    Ok(each(&factory)?.into_iter().map(|(a, _)| a).collect())
}

/// This process's budget on one adapter, and its change event.
pub struct Budget {
    adapter: Com,
    cookie: u32,
    changed: OwnedHandle,
    /// Set by [`Budget::close`]: every wait ends.
    closed: OwnedHandle,
    _dxgi: Library,
}

// SAFETY: DXGI's objects are free-threaded; the events are plain kernel objects.
unsafe impl Send for Budget {}
// SAFETY: as above.
unsafe impl Sync for Budget {}

fn event(manual: bool) -> Result<OwnedHandle> {
    // SAFETY: an unnamed event, not set, with default security.
    let h = unsafe { CreateEventW(std::ptr::null(), i32::from(manual), 0, std::ptr::null()) };
    if h.is_null() {
        return Err(Error::os("create an event")(std::io::Error::last_os_error()));
    }
    // SAFETY: a handle we just made.
    Ok(unsafe { OwnedHandle::from_raw_handle(h) })
}

impl Budget {
    /// The budget on the adapter whose LUID is `luid`; `None` when DXGI names no such adapter.
    pub fn open(luid: [u8; 8]) -> Result<Option<Budget>> {
        let (dxgi, factory) = factory()?;
        let Some((_, adapter)) = each(&factory)?.into_iter().find(|(a, _)| a.luid == luid) else { return Ok(None) };
        let mut third = std::ptr::null_mut();
        // SAFETY: a live adapter and a writable interface pointer.
        let hr = unsafe { method::<QueryInterface>(adapter.0, QUERY_INTERFACE)(adapter.0, &IID_IDXGI_ADAPTER3, &mut third) };
        if hr < 0 {
            return Err(failed("QueryInterface(IDXGIAdapter3)", hr));
        }
        let adapter = Com(third);
        let changed = event(false)?;
        let mut cookie = 0u32;
        // SAFETY: a live IDXGIAdapter3, an event we hold and a writable cookie.
        let hr = unsafe { method::<RegisterBudgetEvent>(adapter.0, REGISTER_BUDGET_EVENT)(adapter.0, changed.as_raw_handle(), &mut cookie) };
        if hr < 0 {
            return Err(failed("IDXGIAdapter3::RegisterVideoMemoryBudgetChangeNotificationEvent", hr));
        }
        Ok(Some(Budget { adapter, cookie, changed, closed: event(true)?, _dxgi: dxgi }))
    }

    /// This process's budget and usage on the adapter's own memory, bytes.
    pub fn read(&self) -> Result<(u64, u64)> {
        let mut info = VideoMemoryInfo::default();
        // SAFETY: a live IDXGIAdapter3, node 0, and an info of the documented layout.
        let hr = unsafe { method::<QueryVideoMemoryInfo>(self.adapter.0, QUERY_VIDEO_MEMORY_INFO)(self.adapter.0, 0, LOCAL, &mut info) };
        if hr < 0 {
            return Err(failed("IDXGIAdapter3::QueryVideoMemoryInfo", hr));
        }
        Ok((info.budget, info.usage))
    }

    /// Blocks until the budget or usage changed (true) or [`Budget::close`] was called (false);
    /// once closed, always false (the lowest signalled index wins, so `closed` comes first).
    pub fn wait(&self) -> bool {
        let handles = [self.closed.as_raw_handle() as HANDLE, self.changed.as_raw_handle() as HANDLE];
        // SAFETY: live events this value holds.
        unsafe { WaitForMultipleObjects(2, handles.as_ptr(), 0, INFINITE) == WAIT_OBJECT_0 + 1 }
    }

    pub fn close(&self) {
        // SAFETY: an event this value holds.
        unsafe { SetEvent(self.closed.as_raw_handle() as HANDLE) };
    }
}

impl Drop for Budget {
    fn drop(&mut self) {
        // SAFETY: the registration `open` made, on the adapter this value holds.
        unsafe { method::<UnregisterBudgetEvent>(self.adapter.0, UNREGISTER_BUDGET_EVENT)(self.adapter.0, self.cookie) };
    }
}

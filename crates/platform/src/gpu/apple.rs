//! The GPU of an Apple Silicon Mac, as Metal reports it (Metal and the Objective-C runtime, opened
//! at run time). Its capability is its newest Apple GPU family (`apple9` on M3 and M4), the row
//! the lane rule reads as it reads a CUDA card's compute capability; a virtualized Mac's GPU
//! reports no family (`apple`). Its memory is Metal's recommended working set: what one process
//! may keep on it without the system paging.
//!
//! A GPU with no family on macOS 26 or later is not offered: there Torch's Metal 4 kernels
//! compute wrong results on it (pytorch#196582; measured on hosted macOS 26 and 27), and Torch
//! 2.15 drops it.
#![allow(unsafe_code)]
use super::{Gpu, MemoryClass};
use libloading::Library;
use std::ffi::{CStr, c_char, c_void};

const METAL: &str = "/System/Library/Frameworks/Metal.framework/Metal";
const OBJC: &str = "/usr/lib/libobjc.A.dylib";
/// `MTLGPUFamilyApple7` (M1) up to Apple 12, newest first.
const FAMILIES: [i64; 6] = [1012, 1011, 1010, 1009, 1008, 1007];

type Id = *mut c_void;
type Sel = *const c_void;

/// The Mac's GPU (this module is Apple Silicon's alone); `None` where Metal answers none.
pub fn probe() -> Option<Gpu> {
    // SAFETY: system libraries at their documented paths; each symbol is copied out under the
    // signature its header declares, and every Objective-C message below is sent with the
    // argument and return types of the method it names (`MTLDevice`, `NSString`).
    unsafe {
        let (metal, objc) = (Library::new(METAL).ok()?, Library::new(OBJC).ok()?);
        let create = *metal
            .get::<unsafe extern "C" fn() -> Id>(b"MTLCreateSystemDefaultDevice\0")
            .ok()?;
        let sel = *objc
            .get::<unsafe extern "C" fn(*const c_char) -> Sel>(b"sel_registerName\0")
            .ok()?;
        let send = *objc.get::<unsafe extern "C" fn()>(b"objc_msgSend\0").ok()?;
        let object: unsafe extern "C" fn(Id, Sel) -> Id = std::mem::transmute(send);
        let text: unsafe extern "C" fn(Id, Sel) -> *const c_char = std::mem::transmute(send);
        let count: unsafe extern "C" fn(Id, Sel) -> u64 = std::mem::transmute(send);
        let family: unsafe extern "C" fn(Id, Sel, i64) -> bool = std::mem::transmute(send);
        let device = create();
        if device.is_null() {
            return None;
        }
        let name = object(device, sel(c"name".as_ptr()));
        let name = match name.is_null() {
            true => "Apple GPU".to_owned(),
            false => CStr::from_ptr(text(name, sel(c"UTF8String".as_ptr())))
                .to_string_lossy()
                .into_owned(),
        };
        let newest = FAMILIES
            .into_iter()
            .find(|f| family(device, sel(c"supportsFamily:".as_ptr()), *f));
        let os = os_version();
        if newest.is_none()
            && os
                .split('.')
                .next()
                .and_then(|m| m.parse::<u32>().ok())
                .is_none_or(|m| m >= 26)
        {
            tracing::warn!(
                "{name} names no Apple GPU family (a virtual GPU) on macOS {os}: this machine offers no GPU"
            );
            return None;
        }
        let memory_bytes = count(device, sel(c"recommendedMaxWorkingSetSize".as_ptr()));
        // The device lives with the process; Metal keeps one system default device anyway.
        std::mem::forget(metal);
        Some(Gpu {
            index: 0,
            uuid: format!("APPLE-{}", name.replace(' ', "-")),
            name,
            pci_bus_id: String::new(),
            memory_bytes,
            driver: format!("macOS {os}"),
            capability: newest.map_or("apple".into(), |f| format!("apple{}", f - 1000)),
            memory_class: Some(MemoryClass::Unified),
        })
    }
}

fn os_version() -> String {
    let mut buffer = [0u8; 32];
    let mut len = buffer.len();
    // SAFETY: a NUL-terminated name and a buffer whose length is passed with it.
    let read = unsafe {
        libc::sysctlbyname(
            c"kern.osproductversion".as_ptr(),
            buffer.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    match read {
        0 => CStr::from_bytes_until_nul(&buffer)
            .map_or_else(|_| "unread".into(), |s| s.to_string_lossy().into_owned()),
        _ => "unread".into(),
    }
}

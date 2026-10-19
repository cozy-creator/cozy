//! Each GPU's memory class from the CUDA driver's device attributes (`libcuda.so.1`,
//! `nvcuda.dll`, opened at run time). A computer with no CUDA driver answers nothing.
use super::{Gpu, MemoryClass};
use crate::{Error, Result};
use libloading::Library;
use std::ffi::{CString, c_char, c_int, c_uint};

const INTEGRATED: c_int = 18;
const PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES: c_int = 100;

#[cfg(windows)]
const LIBRARY: &str = "nvcuda.dll";
#[cfg(not(windows))]
const LIBRARY: &str = "libcuda.so.1";

/// Fills each GPU's `memory_class`: integrated → `Unified`, else pageable memory through the
/// host's page tables → `Coherent`, else `Discrete`. No CUDA driver, no device or a bus id CUDA
/// does not know leaves it `None`; only an attribute a matched device refuses is an error.
///
/// It initializes CUDA in the calling process: call it only where no fork without exec follows
/// (TensorD's service after its stable parent's fork; executors are started with exec).
pub fn memory_classes(gpus: &mut [Gpu]) -> Result<()> {
    if gpus.is_empty() {
        return Ok(());
    }
    // SAFETY: the CUDA driver library, loaded by its documented name.
    let Ok(library) = (unsafe { Library::new(LIBRARY) }) else {
        return Ok(());
    };
    type Init = unsafe extern "C" fn(c_uint) -> c_int;
    type ByBusId = unsafe extern "C" fn(*mut c_int, *const c_char) -> c_int;
    type Attribute = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int;
    // SAFETY: the driver API's exported signatures (cuda.h), copied out as plain fn pointers.
    let symbols = unsafe {
        (
            library.get::<Init>(b"cuInit\0").map(|s| *s),
            library.get::<ByBusId>(b"cuDeviceGetByPCIBusId\0").map(|s| *s),
            library.get::<Attribute>(b"cuDeviceGetAttribute\0").map(|s| *s),
        )
    };
    let (Ok(init), Ok(by_bus_id), Ok(attribute)) = symbols else {
        return Ok(());
    };
    // SAFETY: cuInit takes flags 0; any failure (no device, a driver too old) is no answer.
    if unsafe { init(0) } != 0 {
        return Ok(());
    }
    // The driver stays loaded: it registered exit handlers inside itself at cuInit.
    std::mem::forget(library);
    for gpu in gpus {
        let Ok(bus_id) = CString::new(gpu.pci_bus_id.as_str()) else {
            continue;
        };
        let mut device: c_int = 0;
        // SAFETY: an out-param and a NUL-terminated string that outlives the call.
        if unsafe { by_bus_id(&mut device, bus_id.as_ptr()) } != 0 {
            continue;
        }
        let has = |attr: c_int, name: &'static str| read(attribute, device, attr, name);
        gpu.memory_class = Some(if has(INTEGRATED, "integrated")? {
            MemoryClass::Unified
        } else if has(PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, "host page tables")? {
            MemoryClass::Coherent
        } else {
            MemoryClass::Discrete
        });
    }
    Ok(())
}

fn read(
    attribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int,
    device: c_int,
    attr: c_int,
    name: &'static str,
) -> Result<bool> {
    let mut value: c_int = 0;
    // SAFETY: an out-param for a device the driver just named.
    match unsafe { attribute(&mut value, attr, device) } {
        0 => Ok(value != 0),
        code => Err(Error::Gpu {
            op: "CUDA device attribute",
            detail: format!("{name}: CUresult {code}"),
        }),
    }
}

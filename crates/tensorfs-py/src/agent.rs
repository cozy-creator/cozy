//! varena's agent as torch's CUDA allocator: the C ABI the shim (shim/varena_torch.cpp) calls,
//! over each device's Memory (design/varena-leases.md §11). No Python and no GIL. Every entry
//! catches a panic: an allocation refuses with its reason, anything else logs and returns.

use std::collections::BTreeMap;
use std::ffi::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, RwLock};

use varena::{Error, Memory, PoolId};

/// The Memory serving each CUDA device's torch allocations, from `Memory.open` on.
static SERVING: RwLock<BTreeMap<i32, Arc<Memory>>> = RwLock::new(BTreeMap::new());

/// `memory` serves torch's allocations on CUDA device `device` from now on.
pub fn serve(device: i32, memory: Arc<Memory>) {
    SERVING.write().unwrap_or_else(|p| p.into_inner()).insert(device, memory);
}

fn serving(device: c_int) -> Option<Arc<Memory>> {
    SERVING.read().unwrap_or_else(|p| p.into_inner()).get(&device).cloned()
}

/// `what` without unwinding into C: a panic is logged.
fn guarded(what: &str, device: Option<c_int>, f: impl FnOnce()) {
    if catch_unwind(AssertUnwindSafe(f)).is_err() {
        let memories: Vec<_> = match device {
            Some(device) => serving(device).into_iter().collect(),
            None => SERVING.read().unwrap_or_else(|p| p.into_inner()).values().cloned().collect(),
        };
        for memory in memories { memory.allocator_failed(); }
        tracing::error!("varena: torch's {what} panicked; the executor must retire");
    }
}

/// `text` into the shim's buffer of `len` bytes (its last stays the NUL).
fn write(why: *mut c_char, len: usize, text: &str) {
    if why.is_null() || len == 0 {
        return;
    }
    let n = text.len().min(len);
    // SAFETY: the shim passes a buffer of `len` + 1 bytes, zeroed.
    unsafe { std::ptr::copy_nonoverlapping(text.as_ptr().cast::<c_char>(), why, n) };
}

/// A torch block, or 0 with diagnostic text and a code: 0 capacity; a positive CUresult device
/// fault (999 for unknown ownership); -1 usage/I/O error. Only capacity becomes torch OOM.
///
/// # Safety
/// `why` names `why_len + 1` zeroed writable bytes and `code` names a writable int, or either
/// pointer is null. Both remain valid for this call. The versioned shim enforces this ABI.
#[no_mangle]
pub unsafe extern "C" fn varena_agent_alloc_v2(bytes: u64, device: c_int, stream: usize, routed: c_int, pool_a: u64, pool_b: u64, why: *mut c_char, why_len: usize, code: *mut c_int) -> u64 {
    let served = catch_unwind(AssertUnwindSafe(|| {
        let memory = serving(device).ok_or_else(|| Error::Invalid(format!("no Memory is open on CUDA device {device}")))?;
        memory.torch_alloc(bytes, stream, (routed != 0).then_some(PoolId(pool_a, pool_b)))
    }));
    match served.unwrap_or_else(|_| {
        if let Some(memory) = serving(device) { memory.allocator_failed(); }
        Err(Error::Internal("the allocator panicked".into()))
    }) {
        Ok(ptr) => ptr,
        Err(error) => {
            let result = match &error {
                Error::Exhausted(_) | Error::Device { code: 2, .. } => 0,
                Error::Device { code, .. } if *code > 0 => *code,
                Error::Device { .. } | Error::Ownership(_) | Error::Internal(_) | Error::Closed(_) => 999,
                _ => -1,
            };
            if !code.is_null() {
                // SAFETY: the versioned shim passes a writable code cell.
                unsafe { code.write(result) };
            }
            write(why, why_len, &error.to_string());
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn varena_agent_free(ptr: u64, bytes: u64, device: c_int, stream: usize) {
    guarded("free", Some(device), || match serving(device) {
        Some(memory) => memory.torch_free(ptr, bytes, stream),
        None => tracing::error!("varena: torch freed {ptr:#x} on CUDA device {device}, which no Memory serves"),
    });
}

#[no_mangle]
pub extern "C" fn varena_agent_record_stream(ptr: u64, stream: usize) {
    guarded("record_stream", None, || {
        let memories: Vec<Arc<Memory>> = SERVING.read().unwrap_or_else(|p| p.into_inner()).values().cloned().collect();
        // torch names no device here; a pointer is one Memory's.
        for memory in memories {
            memory.record_stream(ptr, stream);
        }
    });
}

#[no_mangle]
pub extern "C" fn varena_agent_pool_begin(device: c_int, pool_a: u64, pool_b: u64) {
    guarded("pool_begin", Some(device), || {
        if let Some(memory) = serving(device) {
            memory.pool_begin(PoolId(pool_a, pool_b), None); // the shim cannot name the capture stream
        }
    });
}

#[no_mangle]
pub extern "C" fn varena_agent_pool_end(device: c_int, pool_a: u64, pool_b: u64) {
    guarded("pool_end", Some(device), || {
        if let Some(memory) = serving(device) {
            memory.pool_end(PoolId(pool_a, pool_b));
        }
    });
}

#[no_mangle]
pub extern "C" fn varena_agent_pool_release(device: c_int, pool_a: u64, pool_b: u64) {
    guarded("pool_release", Some(device), || {
        if let Some(memory) = serving(device) {
            memory.pool_release(PoolId(pool_a, pool_b));
        }
    });
}

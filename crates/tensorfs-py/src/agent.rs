//! varena's agent as torch's CUDA allocator: the C ABI the shim (shim/varena_torch.cpp) calls,
//! over each device's Memory (design/varena-leases.md §11). No Python and no GIL. Every entry
//! catches a panic: an allocation refuses with its reason, anything else logs and returns.

use std::collections::BTreeMap;
use std::ffi::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, RwLock};

use varena::{Memory, PoolId};

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
fn guarded(what: &str, f: impl FnOnce()) {
    if catch_unwind(AssertUnwindSafe(f)).is_err() {
        tracing::error!("varena: torch's {what} panicked; the block is leaked, not reused");
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

/// A torch block: its device pointer, or 0 with the reason written (the shim throws it as
/// `c10::OutOfMemoryError`). `routed`: the stream's capture routes to pool `(pool_a, pool_b)`.
#[no_mangle]
pub extern "C" fn varena_agent_alloc(bytes: u64, device: c_int, stream: usize, routed: c_int, pool_a: u64, pool_b: u64, why: *mut c_char, why_len: usize) -> u64 {
    let served = catch_unwind(AssertUnwindSafe(|| {
        let memory = serving(device).ok_or_else(|| format!("varena: no Memory is open on CUDA device {device}: open it before any torch CUDA allocation"))?;
        memory.torch_alloc(bytes, stream, (routed != 0).then_some(PoolId(pool_a, pool_b))).map_err(|e| e.to_string())
    }));
    match served.unwrap_or_else(|_| Err("varena: the allocator panicked".into())) {
        Ok(ptr) => ptr,
        Err(reason) => {
            write(why, why_len, &reason);
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn varena_agent_free(ptr: u64, bytes: u64, device: c_int, stream: usize) {
    guarded("free", || match serving(device) {
        Some(memory) => memory.torch_free(ptr, bytes, stream),
        None => tracing::error!("varena: torch freed {ptr:#x} on CUDA device {device}, which no Memory serves"),
    });
}

#[no_mangle]
pub extern "C" fn varena_agent_record_stream(ptr: u64, stream: usize) {
    guarded("record_stream", || {
        let memories: Vec<Arc<Memory>> = SERVING.read().unwrap_or_else(|p| p.into_inner()).values().cloned().collect();
        // torch names no device here; a pointer is one Memory's.
        for memory in memories {
            memory.record_stream(ptr, stream);
        }
    });
}

#[no_mangle]
pub extern "C" fn varena_agent_pool_begin(device: c_int, pool_a: u64, pool_b: u64) {
    guarded("pool_begin", || {
        if let Some(memory) = serving(device) {
            memory.pool_begin(PoolId(pool_a, pool_b), None); // the shim cannot name the capture stream
        }
    });
}

#[no_mangle]
pub extern "C" fn varena_agent_pool_end(device: c_int, pool_a: u64, pool_b: u64) {
    guarded("pool_end", || {
        if let Some(memory) = serving(device) {
            memory.pool_end(PoolId(pool_a, pool_b));
        }
    });
}

#[no_mangle]
pub extern "C" fn varena_agent_pool_release(device: c_int, pool_a: u64, pool_b: u64) {
    guarded("pool_release", || {
        if let Some(memory) = serving(device) {
            memory.pool_release(PoolId(pool_a, pool_b));
        }
    });
}

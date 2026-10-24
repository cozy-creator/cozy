//! varena's agent as torch's CUDA allocator: the C ABI the shim (shim/varena_torch.cpp) calls,
//! over each device's Memory (design/varena-leases.md §11). No Python and no GIL. Every entry
//! catches a panic: an allocation refuses with its reason, anything else logs and returns.

use std::collections::BTreeMap;
use std::ffi::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, RwLock};

use varena::{Error, Memory, PoolId};

/// The v2 allocator ABI: semantic class plus the real driver code when there is one. The shim
/// chooses an exception type from this record, never from the diagnostic message.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Failure {
    kind: c_int,
    cuda_code: c_int,
}

impl Failure {
    fn of(error: &Error) -> Self {
        let cuda_code = match error { Error::Device { code, .. } => *code, _ => -1 };
        let kind = match error {
            Error::Exhausted(_) | Error::Device { code: 2, .. } => 1,
            Error::Device { .. } => 2,
            Error::Ownership(_) => 3,
            Error::Internal(_) => 4,
            Error::Invalid(_) => 5,
            _ => 6,
        };
        Self { kind, cuda_code }
    }
}

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
        let memories: Vec<Arc<Memory>> = match device {
            Some(device) => serving(device).into_iter().collect(),
            None => SERVING.read().unwrap_or_else(|p| p.into_inner()).values().cloned().collect(),
        };
        for memory in memories { memory.quarantine_allocator(); }
        tracing::error!("varena: torch's {what} panicked; its device memory is quarantined");
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

/// A torch block: its device pointer, or 0 with a typed failure and diagnostic reason.
/// `routed`: the stream's capture routes to pool `(pool_a, pool_b)`. Only the versioned shim
/// may call this ABI; `failure` points to its writable record for this call.
///
/// # Safety
/// `why` names `why_len + 1` writable, initially zeroed bytes and `failure` names one writable
/// Failure, or either pointer is null. Both buffers remain valid for the entire call.
#[no_mangle]
pub unsafe extern "C" fn varena_agent_alloc_v2(bytes: u64, device: c_int, stream: usize, routed: c_int, pool_a: u64, pool_b: u64, why: *mut c_char, why_len: usize, failure: *mut Failure) -> u64 {
    let served = catch_unwind(AssertUnwindSafe(|| {
        let memory = serving(device).ok_or_else(|| Error::Invalid(format!("varena: no Memory is open on CUDA device {device}: open it before any torch CUDA allocation")))?;
        memory.torch_alloc(bytes, stream, (routed != 0).then_some(PoolId(pool_a, pool_b)))
    }));
    match served.unwrap_or_else(|_| {
        if let Some(memory) = serving(device) { memory.quarantine_allocator(); }
        Err(Error::Ownership("varena: the allocator panicked".into()))
    }) {
        Ok(ptr) => ptr,
        Err(error) => {
            if !failure.is_null() {
                // SAFETY: the v2 shim supplies one writable Failure for this call.
                unsafe { failure.write(Failure::of(&error)) };
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

#[cfg(test)]
mod tests {
    use super::*;
    use varena::arena::{Chunk, Kind, Source};
    use varena::{Exhausted, GpuId, HostBackend, Va};

    struct Refusal(Error);
    impl Source for Refusal {
        fn map(&self, _: Va, _: Kind, _: &mut Vec<Chunk>) -> varena::Result<()> { Err(self.0.clone()) }
        fn unmap(&self, _: Va, chunks: Vec<Chunk>) -> Vec<(u64, Chunk)> {
            assert!(chunks.is_empty());
            Vec::new()
        }
    }

    #[test]
    fn allocator_abi_keeps_failure_category_and_driver_code() -> varena::Result<()> {
        let cases = [
            (Error::Exhausted(Box::new(Exhausted { need: 4096, site: None, stats: Default::default(), foreign: None })), 1, -1),
            (Error::Device { op: "create", code: 2, name: "OOM".into() }, 1, 2),
            (Error::Device { op: "map", code: 700, name: "ILLEGAL_ADDRESS".into() }, 2, 700),
            (Error::Ownership("mapping rollback failed".into()), 3, -1),
            (Error::Invalid("invalid allocation".into()), 5, -1),
            // An allocator invariant failure poisons its ownership before it crosses the ABI.
            (Error::Internal("broken arena".into()), 3, -1),
            (Error::Io("source disconnected".into()), 6, -1),
        ];
        for (index, (error, kind, cuda_code)) in cases.into_iter().enumerate() {
            let device = 10_000 + index as i32;
            let memory = Arc::new(Memory::open(GpuId { uuid: format!("abi-{device}"), model: "host".into(), memory: 64 << 20 }, Box::new(HostBackend::new(64 << 20)))?);
            memory.serve_torch(Arc::new(Refusal(error)))?;
            serve(device, memory.clone());
            let mut why = [0; 512];
            let mut failure = Failure { kind: 0, cuda_code: 0 };
            // SAFETY: both output buffers are local and sized for the v2 ABI.
            let ptr = unsafe { varena_agent_alloc_v2(4096, device, 0, 0, 0, 0, why.as_mut_ptr(), why.len() - 1, &mut failure) };
            assert_eq!(ptr, 0);
            assert_eq!(failure, Failure { kind, cuda_code });
            assert_ne!(why[0], 0);
            assert_eq!(why[511], 0);
            if matches!(kind, 2 | 3) {
                assert!(matches!(memory.request("after-fault", Default::default()), Err(Error::Ownership(_))));
                assert!(matches!(memory.torch_alloc(512, 0, None), Err(Error::Ownership(_))));
            }
        }
        Ok(())
    }
}

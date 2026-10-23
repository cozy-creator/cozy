//! Step 6 over leases (varena-leases §9 step 3, design/step6-allocator-design.md): varena is
//! torch's only GPU allocator, carving torch blocks out of leased chunks in per-request, process
//! and graph-pool arenas. Signatures agreed with S3's shim; the bodies come in step 3.

use super::{Memory, Request};
use crate::Result;

/// torch's `MempoolId_t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PoolId(pub u64, pub u64);

/// torch's figures, varena's now: process-wide and per open request.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TorchStats {
    pub allocated: u64,
    pub mapped: u64,
    pub peak: u64,
    /// Request id → (allocated, peak).
    pub requests: Vec<(String, u64, u64)>,
}

/// The calling thread allocates into a request's arena until this drops; owned, so a Python
/// request object keeps it across calls.
pub struct Allocating {
    _private: (),
}

impl Memory {
    /// A torch block on `stream`: from `pool`'s arena while a capture routes there, else the
    /// current request's arena, else the process arena. Err is `Error::Exhausted`, whose text is
    /// torch's OOM reason. A capturing thread is served from free bytes or refused (R6).
    pub fn torch_alloc(&self, _bytes: u64, _stream: usize, _pool: Option<PoolId>) -> Result<u64> {
        todo!("step 3")
    }

    /// Never fails and never waits on the device.
    pub fn torch_free(&self, _ptr: u64, _bytes: u64, _stream: usize) {
        todo!("step 3")
    }

    pub fn record_stream(&self, _ptr: u64, _stream: usize) {
        todo!("step 3")
    }

    /// Before capture begins, from Python: maps `expected` bytes into `pool`, making room here.
    pub fn pool_prepare(&self, _pool: PoolId, _expected: u64) -> Result<()> {
        todo!("step 3")
    }

    /// torch's begin-allocate-to-pool hook, on the capturing thread (the caller): routes its
    /// allocations to `pool` (each names its capture stream). Makes no room.
    pub fn pool_begin(&self, _pool: PoolId) {
        todo!("step 3")
    }

    /// Stops routing; frees made during capture apply here (R2).
    pub fn pool_end(&self, _pool: PoolId) {
        todo!("step 3")
    }

    /// One use fewer; the pool's chunks go once none is left and its release event completed
    /// (R4/R5).
    pub fn pool_release(&self, _pool: PoolId) {
        todo!("step 3")
    }

    pub fn torch_stats(&self) -> TorchStats {
        todo!("step 3")
    }

    pub fn allocating(&self, _request: &Request) -> Allocating {
        todo!("step 3")
    }
}

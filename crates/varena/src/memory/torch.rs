//! Step 6 over leases (varena-leases §11): the hooks S3's shim calls, on the device's
//! [`Torch`] (arena.rs). The lease agent serves its chunks ([`Memory::serve_torch`]); with none
//! served, the device's own pages up to the card do ([`Direct`]).

use std::sync::Arc;

use super::{Memory, Request};
use crate::arena::{Direct, Owner, Route, Source, Torch};
use crate::{Error, Result};

pub use crate::arena::{Ended, PoolId};

/// torch's figures, varena's now: process-wide, per open request and per pool.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TorchStats {
    pub allocated: u64,
    pub mapped: u64,
    pub peak: u64,
    /// Request id → (allocated, peak).
    pub requests: Vec<(String, u64, u64)>,
    /// Pool → (mapped, allocated): torch's `MemPool.snapshot()` raises under a pluggable allocator.
    pub pools: Vec<(PoolId, u64, u64)>,
}

/// The calling thread allocates into an arena until this drops (on any thread); owned, so a
/// Python object keeps it across calls.
pub struct Allocating {
    _route: Option<Route>,
}

impl Memory {
    /// Serves torch's allocations from `source` (the lease agent), before any is made.
    pub fn serve_torch(&self, source: Arc<dyn Source>) -> Result<Arc<Torch>> {
        let torch = Torch::new(self.inner.backend.clone(), source)?;
        self.inner.torch.set(torch.clone()).map_err(|_| Error::Invalid("torch's allocations are served already".into()))?;
        Ok(torch)
    }

    /// The device's torch allocator: standalone over its own pages when nothing served it.
    pub fn torch(&self) -> Result<Arc<Torch>> {
        if let Some(t) = self.inner.torch.get() {
            return Ok(t.clone());
        }
        let backend = self.inner.backend.clone();
        let torch = Torch::new(backend.clone(), Arc::new(Direct::new(backend.clone(), backend.capacity())))?;
        Ok(self.inner.torch.get_or_init(|| torch).clone())
    }

    /// A torch block on `stream`: from `pool`'s arena while a capture routes there, else the
    /// thread's routed arena (a request's or a region's), else the process's. Err is
    /// `Error::Exhausted`, whose text is torch's OOM reason.
    pub fn torch_alloc(&self, bytes: u64, stream: usize, pool: Option<PoolId>) -> Result<u64> {
        self.torch()?.alloc(bytes, stream, pool)
    }

    /// Never fails and never waits on the device.
    pub fn torch_free(&self, ptr: u64, bytes: u64, stream: usize) {
        match self.inner.torch.get() {
            Some(t) => t.free(ptr, bytes, stream),
            None => tracing::error!("varena: torch freed {ptr:#x} on a Memory that never allocated for it"),
        }
    }

    pub fn record_stream(&self, ptr: u64, stream: usize) {
        if let Some(t) = self.inner.torch.get() {
            t.record_stream(ptr, stream);
        }
    }

    /// Before capture begins: maps `expected` bytes into `pool`, making room here.
    pub fn pool_prepare(&self, pool: PoolId, expected: u64) -> Result<()> {
        self.torch()?.pool_prepare(pool, expected)
    }

    /// torch's begin-allocate-to-pool hook, on the capturing thread before the capture starts.
    pub fn pool_begin(&self, pool: PoolId) {
        match self.torch() {
            Ok(t) => t.pool_begin(pool),
            Err(e) => tracing::error!("varena: no torch allocator for a capture: {e}"),
        }
    }

    /// After the capture ended, or failed (torch never ends a failed one: the Runtime does).
    pub fn pool_end(&self, pool: PoolId) {
        if let Some(t) = self.inner.torch.get() {
            t.pool_end(pool);
        }
    }

    /// One use fewer; the pool's chunks go once none is left and its fences completed (R4/R5).
    pub fn pool_release(&self, pool: PoolId) {
        if let Some(t) = self.inner.torch.get() {
            t.pool_release(pool);
        }
    }

    /// An event on `stream` after the pool's use there (a replay and its readers).
    pub fn pool_fence(&self, pool: PoolId, stream: usize) -> Result<()> {
        self.torch()?.pool_fence(pool, stream)
    }

    pub fn torch_stats(&self) -> TorchStats {
        let Some(t) = self.inner.torch.get() else { return TorchStats::default() };
        let s = t.stats();
        let mut out = TorchStats { allocated: s.allocated, mapped: s.mapped, peak: s.peak, ..TorchStats::default() };
        for (owner, a) in s.arenas {
            match owner {
                Owner::Request(id) => out.requests.push((id.to_string(), a.allocated, a.peak)),
                Owner::Pool(p) => out.pools.push((p, a.mapped, a.allocated)),
                Owner::Process | Owner::Region(_) => {}
            }
        }
        out
    }

    /// The calling thread's blocks go to the request's arena until the guard drops.
    pub fn allocating(&self, request: &Request) -> Allocating {
        Allocating { _route: self.torch().ok().map(|t| t.route(Owner::Request(request.id))) }
    }

    /// The calling thread's blocks go to cache region `region`'s own arena (M2) until it drops.
    pub fn region_allocating(&self, region: u64) -> Allocating {
        Allocating { _route: self.torch().ok().map(|t| t.route(Owner::Region(region))) }
    }

    /// The region is gone: its spans leave as they come whole free; the record names survivors.
    pub fn region_close(&self, region: u64) -> Ended {
        self.inner.torch.get().map(|t| t.end(Owner::Region(region))).unwrap_or_default()
    }
}

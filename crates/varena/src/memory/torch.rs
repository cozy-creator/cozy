//! Step 6 over leases (varena-leases §11): the hooks S3's shim calls, on the device's
//! [`Torch`] (arena.rs). Its chunks come from the lease agent ([`Memory::serve_leased_torch`]),
//! or, for a Memory explicitly opened standalone, from the device's own pages
//! ([`Memory::serve_standalone_torch`]). Until one of them is served, every torch allocation is
//! refused: nothing is ever made outside the manager's books.

#[cfg(unix)]
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[cfg(unix)]
use parking_lot::Mutex;

use super::{Memory, Request};
use crate::arena::{Direct, Owner, Route, Source, Torch};
#[cfg(unix)]
use crate::{
    arena::{Chunk, Kind},
    backend::{Access, Backend, Physical, Va},
    lease::{Advice, Agent, Ticket},
};
use crate::{Error, Result};

pub use crate::arena::{Ended, PoolId};

/// torch's figures, varena's now: process-wide, per open request and per pool.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TorchStats {
    pub allocated: u64,
    pub mapped: u64,
    pub peak: u64,
    /// The allocator's own overhead: mapped less live torch bytes.
    pub fragmentation: u64,
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
    /// Serves torch's allocations from `source`, once, before any is made.
    pub fn serve_torch(&self, source: Arc<dyn Source>) -> Result<Arc<Torch>> {
        let torch = Torch::new(self.inner.backend.clone(), source)?;
        self.inner.torch.set(torch.clone()).map_err(|_| Error::Invalid("torch's allocations are served already".into()))?;
        Ok(torch)
    }

    /// Torch's chunks from the manager this Memory leases from ([`Memory::lease`]).
    #[cfg(unix)]
    pub fn serve_leased_torch(&self) -> Result<Arc<Torch>> {
        let agent = self.inner.lease.get().cloned().ok_or_else(|| Error::Invalid("this Memory holds no lease".into()))?;
        let mech = self.inner.mech.clone();
        let wake = Box::new(move || {
            mech.post(super::Mech::Revoke);
        });
        self.serve_torch(Arc::new(Leased { agent, backend: self.inner.backend.clone(), held: Mutex::new(HashMap::new()), wake }))
    }

    /// Standalone, chosen explicitly when no compatible manager serves this GPU: the device's
    /// own pages, within the allowance its weight pages share (rev 6).
    pub fn serve_standalone_torch(&self) -> Result<Arc<Torch>> {
        self.serve_torch(Arc::new(Direct::new(self.inner.backend.clone(), self.inner.allowance.clone())))
    }

    /// The device's torch allocator: refused until one is served.
    pub fn torch(&self) -> Result<Arc<Torch>> {
        self.inner.torch.get().cloned().ok_or_else(|| {
            Error::Invalid("varena serves no torch allocation on this GPU yet: lease it and serve torch first (Memory::serve_leased_torch), or open it standalone (Memory::serve_standalone_torch)".into())
        })
    }

    /// A torch block on `stream`: from `pool`'s arena while a capture routes there, else the
    /// thread's routed arena (a request's or a region's), else the process's. Err is
    /// `Error::Exhausted`, whose text is torch's OOM reason, or `Invalid` before torch is served.
    pub fn torch_alloc(&self, bytes: u64, stream: usize, pool: Option<PoolId>) -> Result<u64> {
        self.torch()?.alloc(bytes, stream, pool)
    }

    /// Never fails and never waits on the device.
    pub fn torch_free(&self, ptr: u64, bytes: u64, stream: usize) {
        match self.inner.torch.get() {
            Some(t) => t.free(ptr, bytes, stream),
            None => tracing::error!("varena: torch freed {ptr:#x} on a GPU whose torch allocations are not served"),
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

    /// torch's begin-allocate-to-pool hook, on the capturing thread before the capture starts on
    /// `stream` (its current stream, when the caller knows it; else the first routed allocation
    /// names it).
    pub fn pool_begin(&self, pool: PoolId, stream: Option<usize>) {
        match self.torch() {
            Ok(t) => t.pool_begin(pool, stream),
            Err(e) => tracing::error!("varena: a capture began with {e}"),
        }
    }

    /// After the capture ended, or failed: torch's reset (any thread) or the Runtime's
    /// capture_failed ends one whose begin threw.
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
        let mut out = TorchStats { allocated: s.allocated, mapped: s.mapped, peak: s.peak, fragmentation: s.fragmentation, ..TorchStats::default() };
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
        Allocating { _route: self.inner.torch.get().map(|t| t.route(Owner::Request(request.id))) }
    }

    /// The calling thread's blocks go to cache region `region`'s own arena (M2) until it drops.
    pub fn region_allocating(&self, region: u64) -> Allocating {
        Allocating { _route: self.inner.torch.get().map(|t| t.route(Owner::Region(region))) }
    }

    /// The region is gone: its spans leave as they come whole free; the record names survivors.
    pub fn region_close(&self, region: u64) -> Ended {
        self.inner.torch.get().map(|t| t.end(Owner::Region(region))).unwrap_or_default()
    }
}

impl Request {
    /// A section's torch peak (a package split's window): the request arena's high-water since the
    /// last section or pass start, which restarts here; the open pass's peak still covers it.
    pub fn section_peak(&self) -> u64 {
        self.inner.torch.get().map_or(0, |t| t.reset_peak(Owner::Request(self.id)))
    }
}

impl super::Inner {
    /// A pass opens (`Request::pass`): its request arena's peak restarts at what is live.
    pub(super) fn torch_pass(&self, req: u64) {
        if let Some(t) = self.torch.get() {
            t.pass_start(Owner::Request(req));
        }
    }

    /// A pass ends (`Pass::end`): the most its request arena held since the pass opened, the
    /// sections measured inside it included.
    pub(super) fn torch_peak(&self, req: u64) -> u64 {
        self.torch.get().map_or(0, |t| t.pass_peak(Owner::Request(req)))
    }

    /// The manager asked these back (`Inner::revoke`, no state lock held): the torch chunks among
    /// them that are idle and quiet go now; one in use stays pinned, one whose epoch is pending
    /// goes at the poller's next wake ([`Source::quiet`]).
    #[cfg(unix)]
    pub(super) fn revoke_torch(&self, asked: &HashSet<Ticket>) {
        if let Some(t) = self.torch.get() {
            let ids: Vec<u64> = asked.iter().map(|t| id(*t)).collect();
            t.reclaim_chunks(&ids);
        }
    }
}

#[cfg(unix)]
fn id(t: Ticket) -> u64 {
    u64::from(t.slot) << 32 | u64::from(t.seq)
}

/// The lease agent as the arenas' source: the manager's chunks, mapped at the arenas' VAs. A
/// chunk in an arena is pinned in the table, so the manager never takes one in use; an idle one
/// is unpinned and hinted idle, and goes when asked ([`super::Inner::revoke_torch`]).
#[cfg(unix)]
struct Leased {
    agent: Arc<Agent>,
    backend: Arc<dyn Backend>,
    held: Mutex<HashMap<u64, Held>>,
    /// Runs the mechanism thread's revoke.
    wake: Box<dyn Fn() + Send + Sync>,
}

/// A chunk's grant and import, whether it is pinned, and whether its give-back committed (a
/// retry then only unmaps and releases).
#[cfg(unix)]
struct Held {
    ticket: Ticket,
    phys: Physical,
    pinned: bool,
    committed: bool,
}

#[cfg(unix)]
impl Leased {
    /// One chunk back: unpinned, committed, unmapped, released. Err leaves it mapped (still
    /// pinned, or the unmap failed), its lease kept for the retry.
    fn give(&self, at: u64, c: &Chunk) -> Result<()> {
        let held = self.held.lock().remove(&c.id);
        let Some(mut h) = held else { return Err(Error::Internal(format!("chunk {:#x} was never granted here", c.id))) };
        let step = (|| {
            if h.pinned {
                self.agent.unpin(h.ticket)?;
                h.pinned = false;
            }
            if !h.committed {
                if !self.agent.commit(h.ticket)? {
                    return Err(Error::Internal("still pinned".into()));
                }
                h.committed = true;
            }
            self.backend.unmap(Va { base: at, len: c.len }, 0, c.len)
        })();
        match step {
            Ok(()) => {
                // Unmapped: its addresses are safe to reuse whatever the release says.
                if let Err(e) = self.agent.release(h.ticket, h.phys) {
                    tracing::error!("varena: releasing torch chunk {:#x} failed: {e}", c.id);
                }
                Ok(())
            }
            Err(e) => {
                self.held.lock().insert(c.id, h);
                Err(e)
            }
        }
    }
}

#[cfg(unix)]
impl Source for Leased {
    fn slot(&self, chunk: &Chunk) -> Option<u32> {
        u32::try_from(chunk.id >> 32).ok()
    }

    fn map(&self, va: Va, _: Kind, into: &mut Vec<Chunk>) -> Result<()> {
        let (mut tickets, mut off, mut failed) = (Vec::new(), 0, None);
        for lease in self.agent.take(va.len, crate::lease::Kind::Torch, None)? {
            let len = lease.phys.len;
            let mapped = if failed.is_some() || off + len > va.len {
                Err(Error::Internal(format!("a grant of {len} bytes past its {}-byte span", va.len)))
            } else {
                self.backend.map(va, off, &lease.phys, Access::ReadWrite)
            };
            match mapped {
                Ok(()) => {
                    tickets.push(lease.ticket);
                    into.push(Chunk { id: id(lease.ticket), len });
                    self.held.lock().insert(id(lease.ticket), Held { ticket: lease.ticket, phys: lease.phys, pinned: false, committed: false });
                    off += len;
                }
                Err(e) => {
                    let _ = self.agent.refused(lease.ticket, lease.phys);
                    failed.get_or_insert(e);
                }
            }
        }
        self.agent.mapped(&tickets)?;
        if off < va.len && failed.is_none() {
            failed = Some(Error::Internal(format!("a grant of {off} bytes for a {}-byte span", va.len)));
        }
        if let Some(e) = failed {
            return Err(e);
        }
        self.idle(into, false);
        Ok(())
    }

    fn unmap(&self, va: Va, chunks: Vec<Chunk>) -> Vec<(u64, Chunk)> {
        let (mut at, mut stuck) = (va.base, Vec::new());
        for c in chunks {
            let here = at;
            at += c.len;
            if let Err(e) = self.give(here, &c) {
                tracing::error!("varena: torch chunk {:#x} could not go back yet: {e}", c.id);
                stuck.push((here, c));
            }
        }
        stuck
    }

    fn idle(&self, chunks: &[Chunk], idle: bool) {
        let mut held = self.held.lock();
        for c in chunks {
            let Some(h) = held.get_mut(&c.id) else { continue };
            if idle && h.pinned {
                h.pinned = self.agent.unpin(h.ticket).is_err();
            } else if !idle && !h.pinned {
                h.pinned = self.agent.pin(h.ticket).unwrap_or(false);
            }
            // Whole free: revocable at once, ahead of any executor's idle weights.
            let advice = if idle { Advice::Free { since: super::monotonic_ns() } } else { Advice::Unset };
            let _ = self.agent.advise(h.ticket, advice);
        }
    }

    fn quiet(&self) {
        if !self.agent.asked_back().is_empty() {
            (self.wake)();
        }
    }
}

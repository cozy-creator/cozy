//! The mechanism under [`crate::Memory`], in the shape of CUDA's VMM: reserve address space,
//! create shareable physical pages, map them (read-only once filled), export and import them,
//! fill them through TensorFS's `io::read` on a copy stream, and order consumers on events.
//! [`HostBackend`] is the same mechanism over host pages (a memfd, an `shm_open` object, a
//! pagefile section): CPU and MPS executors and the CPU tests use it; [`crate::CudaBackend`] is CUDA's.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use platform::handles::Owned;
use platform::memory;
use tensorfs::io::Dst;

use crate::{Error, Result};

/// Reserved address space: `len` bytes from `base`, nothing mapped until [`Backend::map`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Va {
    pub base: u64,
    pub len: u64,
}

/// A physical page of `len` bytes; `id` is the backend's handle. Not `Clone`: one owner, given
/// back through [`Backend::destroy`].
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Physical {
    pub id: u64,
    pub len: u64,
}

/// An event on a stream (a `CUevent`); [`Event::DONE`] is already complete.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Event(pub u64);

impl Event {
    pub const DONE: Event = Event(0);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    ReadWrite,
    Read,
}

/// The code [`Backend::create`] fails with when the device has no memory for the page
/// (`CUDA_ERROR_OUT_OF_MEMORY`; the host backend answers the same at its limit).
pub const OUT_OF_MEMORY: i32 = 2;

/// One device's mechanism. Every call is short: no call waits on another process or a fill.
pub trait Backend: Send + Sync {
    /// The device's total bytes (the card, or a host device's limit).
    fn capacity(&self) -> u64;
    /// Bytes the driver needs free at all times (its lazy kernel loads, cr-215): never leased,
    /// never filled, in any mode.
    fn headroom(&self) -> u64 {
        crate::table::headroom(self.capacity())
    }
    /// Free bytes as the driver measures them now. Only foreign bytes derive from it (the
    /// device's used bytes less every row's); placement never sizes from it.
    fn free(&self) -> Result<u64>;
    /// Device bytes this process holds now, measured (NVML per process). Its context is this
    /// less torch's reserved bytes and its mapped pages. Errs where it cannot be measured
    /// (WDDM, a foreign pid namespace).
    fn process_used(&self) -> Result<u64>;
    /// The device's used bytes across this process's context creation, at open: the context
    /// where `process_used` cannot be measured.
    fn opened_context(&self) -> u64;
    /// Physical pages and their offsets are multiples of this.
    fn granularity(&self) -> u64;
    fn reserve(&self, len: u64) -> Result<Va>;
    /// Gives back a reservation; nothing in it is mapped.
    fn release(&self, va: Va);
    /// New physical memory that [`Backend::export`] can share. Fails when the device is full.
    fn create(&self, len: u64) -> Result<Physical>;
    /// Drops this process's reference; the memory is freed when no mapping or handle holds it.
    fn destroy(&self, page: Physical) -> Result<()>;
    fn map(&self, va: Va, offset: u64, page: &Physical, access: Access) -> Result<()>;
    /// Changes the access of a mapped range (filled pages become read-only).
    fn protect(&self, va: Va, offset: u64, len: u64, access: Access) -> Result<()>;
    fn unmap(&self, va: Va, offset: u64, len: u64) -> Result<()>;
    /// A handle another process can import (a POSIX fd, a WIN32 handle).
    fn export(&self, page: &Physical) -> Result<Owned>;
    /// A handle another process maps read-write: a lease (a host page's export is read-only).
    fn lend(&self, page: &Physical) -> Result<Owned> {
        self.export(page)
    }
    fn import(&self, handle: Owned, len: u64) -> Result<Physical>;
    /// Where `io::read` writes `len` mapped bytes at `va + offset`: device memory ordered on the
    /// backend's copy stream, or host memory.
    fn target(&self, va: Va, offset: u64, len: u64) -> Dst;
    /// An event after everything the copy stream holds so far.
    fn fence(&self) -> Result<Event>;
    /// An event (with timing) after everything `stream` holds so far.
    fn record(&self, stream: usize) -> Result<Event>;
    fn query(&self, event: Event) -> Result<bool>;
    /// Nanoseconds between two completed events, measured on the device; `None` until both are
    /// complete (never a host wait).
    fn elapsed(&self, from: Event, to: Event) -> Result<Option<u64>>;
    /// Orders `stream` (a `CUstream`; 0 the default) after `event`, without a host wait.
    fn wait(&self, event: Event, stream: usize) -> Result<()>;
    /// Blocks until `stream`'s work so far completes: only where recording an event failed.
    fn synchronize(&self, stream: usize) -> Result<()>;
    /// Blocks until `event` completes: only the mechanism thread, before it unmaps a page whose
    /// consumers may still read it, and a planned handover ([`crate::Memory::settle`]).
    fn finish(&self, event: Event) -> Result<()>;
    /// Gives `event` up, pending or not, from any thread (the allocator hook's too): it only
    /// queues it, no driver call. Retiring one event twice destroys it once.
    fn retire(&self, event: Event);
    /// Destroys the retired events that completed (never waiting on one): the mechanism and
    /// trim threads only, since a destroy can wait on a driver lock another thread holds.
    fn sweep(&self) {}
    /// This process's device budget, where the OS sets one (Windows); `None` elsewhere.
    fn budget(&self) -> Option<Arc<dyn Budget>> {
        None
    }
    /// After a device error: proves the copy stream's work complete or fails.
    fn drain(&self) -> Result<()>;
}

/// Pages in host memory (a memfd, an `shm_open` object or a pagefile section each), mapped into
/// a reserved range. Its export maps read-only, its capacity a given limit. Fills are synchronous, so its events
/// are always complete.
pub struct HostBackend {
    limit: u64,
    headroom: u64,
    pages: Mutex<Pages>,
}

#[derive(Default)]
struct Pages {
    next: u64,
    /// id → (handle, bytes it counts against the limit: 0 for an imported page).
    live: HashMap<u64, (Owned, u64)>,
}

impl Pages {
    fn created(&self) -> u64 {
        self.live.values().map(|(_, n)| n).sum()
    }

    fn insert(&mut self, handle: Owned, counted: u64, len: u64) -> Physical {
        self.next += 1;
        self.live.insert(self.next, (handle, counted));
        Physical { id: self.next, len }
    }
}

impl HostBackend {
    /// Host pages load no kernels: no headroom.
    pub fn new(limit: u64) -> HostBackend {
        HostBackend::with_headroom(limit, 0)
    }

    /// A host device that keeps `headroom` free, as a card's driver does.
    pub fn with_headroom(limit: u64, headroom: u64) -> HostBackend {
        HostBackend { limit, headroom, pages: Mutex::new(Pages::default()) }
    }
}

fn page_err(e: platform::Error) -> Error {
    let e = std::io::Error::from(e);
    Error::Device { op: "host page", code: e.raw_os_error().unwrap_or(-1), name: e.to_string() }
}

/// One implementation on every OS, over cozy-platform's host-page seams (memfd, `shm_open`, or
/// pagefile sections in placeholders).
impl Backend for HostBackend {
    fn capacity(&self) -> u64 {
        self.limit
    }

    fn free(&self) -> Result<u64> {
        Ok(self.limit.saturating_sub(self.pages.lock().created()))
    }

    /// The pages it created, as NVML charges a process: an imported page is its exporter's.
    fn process_used(&self) -> Result<u64> {
        Ok(self.pages.lock().created())
    }

    fn opened_context(&self) -> u64 {
        0
    }

    fn granularity(&self) -> u64 {
        memory::granularity()
    }

    fn reserve(&self, len: u64) -> Result<Va> {
        Ok(Va { base: memory::reserve(len).map_err(page_err)?, len })
    }

    fn release(&self, va: Va) {
        memory::release(va.base, va.len);
    }

    fn create(&self, len: u64) -> Result<Physical> {
        let mut pages = self.pages.lock();
        if pages.created().saturating_add(len) > self.limit {
            // As a card answers.
            return Err(Error::Device { op: "create", code: OUT_OF_MEMORY, name: "out of memory".into() });
        }
        let page = memory::page(len).map_err(page_err)?;
        // A holder's ftruncate must never SIGBUS another mapper (a lease's adopter).
        #[cfg(target_os = "linux")]
        platform::shm::seal_with(std::os::fd::AsFd::as_fd(&page), platform::shm::RESIZE_SEALS).map_err(page_err)?;
        Ok(pages.insert(page, len, len))
    }

    fn destroy(&self, page: Physical) -> Result<()> {
        self.pages.lock().live.remove(&page.id);
        Ok(())
    }

    fn map(&self, va: Va, offset: u64, page: &Physical, access: Access) -> Result<()> {
        let pages = self.pages.lock();
        let (handle, _) = pages.live.get(&page.id).ok_or_else(|| Error::Invalid(format!("no page {}", page.id)))?;
        if offset.saturating_add(page.len) > va.len {
            return Err(Error::Invalid("map outside the reservation".into()));
        }
        memory::map_page(va.base + offset, handle, page.len, access == Access::ReadWrite).map_err(page_err)
    }

    fn protect(&self, va: Va, offset: u64, len: u64, access: Access) -> Result<()> {
        memory::protect(va.base + offset, len, access == Access::ReadWrite).map_err(page_err)
    }

    fn unmap(&self, va: Va, offset: u64, len: u64) -> Result<()> {
        memory::unmap_page(va.base + offset, len).map_err(page_err)
    }

    fn export(&self, page: &Physical) -> Result<Owned> {
        let pages = self.pages.lock();
        let (handle, _) = pages.live.get(&page.id).ok_or_else(|| Error::Invalid(format!("no page {}", page.id)))?;
        memory::export(handle).map_err(page_err)
    }

    fn headroom(&self) -> u64 {
        self.headroom
    }

    fn lend(&self, page: &Physical) -> Result<Owned> {
        let pages = self.pages.lock();
        let (handle, _) = pages.live.get(&page.id).ok_or_else(|| Error::Invalid(format!("no page {}", page.id)))?;
        memory::lend(handle).map_err(page_err)
    }

    fn import(&self, handle: Owned, len: u64) -> Result<Physical> {
        Ok(self.pages.lock().insert(handle, 0, len))
    }

    fn target(&self, va: Va, offset: u64, len: u64) -> Dst {
        Dst::Host { ptr: (va.base + offset) as *mut u8, len: len as usize }
    }

    fn fence(&self) -> Result<Event> {
        Ok(Event::DONE)
    }

    fn record(&self, _: usize) -> Result<Event> {
        Ok(Event::DONE)
    }

    fn query(&self, _: Event) -> Result<bool> {
        Ok(true)
    }

    fn elapsed(&self, _: Event, _: Event) -> Result<Option<u64>> {
        Ok(Some(0))
    }

    fn wait(&self, _: Event, _: usize) -> Result<()> {
        Ok(())
    }

    fn synchronize(&self, _: usize) -> Result<()> {
        Ok(())
    }

    /// Host events are complete when made.
    fn finish(&self, _: Event) -> Result<()> {
        Ok(())
    }

    fn retire(&self, _: Event) {}

    fn drain(&self) -> Result<()> {
        Ok(())
    }
}

/// This process's device budget where the OS sets one per process (Windows: DXCore/DXGI) and its
/// changes; a game in front shrinks it.
pub trait Budget: Send + Sync {
    /// This process's budget and usage, bytes.
    fn read(&self) -> Result<(u64, u64)>;
    /// Blocks until the budget or usage changed (true) or [`Budget::close`] was called (false).
    fn wait(&self) -> bool;
    fn close(&self);
}

/// The driver's events behind varena's ids. An id is never reused, so retiring one twice, or a
/// stale copy of it after the driver handed its handle out again, reaches nothing: a handle is
/// queued for destruction once, by construction. Destroyed only once complete and only by a
/// sweep, on a thread that may wait (a destroy can block on its work or on a driver lock another
/// thread holds).
pub struct Events<H> {
    live: parking_lot::RwLock<HashMap<u64, H>>,
    next: std::sync::atomic::AtomicU64,
    retired: Mutex<Vec<H>>,
}

impl<H> Default for Events<H> {
    fn default() -> Self {
        Events { live: parking_lot::RwLock::new(HashMap::new()), next: std::sync::atomic::AtomicU64::new(Event::DONE.0 + 1), retired: Mutex::new(Vec::new()) }
    }
}

impl<H: Copy> Events<H> {
    /// A new id for `handle`.
    pub fn add(&self, handle: H) -> Event {
        let id = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.live.write().insert(id, handle);
        Event(id)
    }

    /// The live handle of `event`; `None` once retired (or never made here).
    pub fn get(&self, event: Event) -> Option<H> {
        self.live.read().get(&event.0).copied()
    }

    /// Gives `event` up: its handle is queued for a sweep, no driver call. Again: nothing.
    pub fn retire(&self, event: Event) {
        if let Some(handle) = self.live.write().remove(&event.0) {
            self.retired.lock().push(handle);
        }
    }

    /// Destroys the queued handles `done` reports complete; the others stay queued. The queue is
    /// not held across the driver calls, so a retire never waits on a sweep.
    pub fn sweep(&self, done: impl Fn(H) -> bool, destroy: impl Fn(H)) {
        let taken = std::mem::take(&mut *self.retired.lock());
        let pending: Vec<H> = taken.into_iter().filter(|h| {
            let complete = done(*h);
            if complete {
                destroy(*h);
            }
            !complete
        }).collect();
        if !pending.is_empty() {
            self.retired.lock().extend(pending);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashSet;

    use super::*;

    /// A retire never destroys; an event given up twice, while pending and after, is destroyed
    /// once by a sweep, when complete. A stale id retired after its handle was handed out again
    /// (the driver reuses handles) reaches nothing: the new event lives on.
    #[test]
    fn a_retired_event_is_destroyed_once_and_only_once_complete() {
        let (events, complete, destroyed) = (Events::<u64>::default(), RefCell::new(HashSet::new()), RefCell::new(Vec::new()));
        let done = |h: u64| complete.borrow().contains(&h);
        let destroy = |h: u64| destroyed.borrow_mut().push(h);
        let first = events.add(7);
        events.retire(first);
        events.retire(first);
        events.retire(Event::DONE);
        assert!(destroyed.borrow().is_empty(), "a retire destroyed");
        events.sweep(done, destroy);
        assert!(destroyed.borrow().is_empty(), "a pending event was destroyed");
        complete.borrow_mut().insert(7);
        events.sweep(done, destroy);
        events.sweep(done, destroy);
        assert_eq!(*destroyed.borrow(), vec![7], "destroyed once, once complete");
        // The driver hands handle 7 out again: a stale retire of the first id cannot reach it.
        complete.borrow_mut().remove(&7);
        let second = events.add(7);
        events.retire(first);
        events.sweep(done, destroy);
        assert_eq!((events.get(second), destroyed.borrow().len()), (Some(7), 1), "a stale id reached a live handle");
    }
}

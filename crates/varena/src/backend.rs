//! The mechanism under [`crate::Memory`], in the shape of CUDA's VMM: reserve address space,
//! create shareable physical pages, map them (read-only once filled), export and import them,
//! fill them through TensorFS's `io::read` on a copy stream, and order consumers on events.
//! [`HostBackend`] is the same mechanism over host pages (memfd): CPU and MPS executors and the
//! CPU tests use it. The CUDA backend lives with its driver (S2).

use std::collections::HashMap;
use std::os::fd::{AsFd, AsRawFd, OwnedFd};

use parking_lot::Mutex;
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

/// An event on the backend's copy stream (a `CUevent`); [`Event::DONE`] is already complete.
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

/// One device's mechanism. Every call is short: no call waits on another process or a fill.
pub trait Backend: Send + Sync {
    /// The device's total bytes (the card, or a host device's limit).
    fn capacity(&self) -> u64;
    /// Free bytes as the driver measures them now. Only foreign bytes derive from it (the device's
    /// used bytes less every row's); placement never sizes from it.
    fn free(&self) -> Result<u64>;
    /// Bytes this process's driver context holds outside pages and the allocator, measured.
    fn context(&self) -> u64;
    /// Physical pages and their offsets are multiples of this.
    fn granularity(&self) -> u64;
    fn reserve(&self, len: u64) -> Result<Va>;
    /// Gives back a reservation; nothing in it is mapped.
    fn release(&self, va: Va);
    /// New physical memory that [`Backend::export`] can share. Fails when the device is full.
    fn create(&self, len: u64) -> Result<Physical>;
    /// Drops this process's reference; the memory is freed when no mapping or handle holds it.
    fn destroy(&self, page: Physical);
    fn map(&self, va: Va, offset: u64, page: &Physical, access: Access) -> Result<()>;
    /// Changes the access of a mapped range (filled pages become read-only).
    fn protect(&self, va: Va, offset: u64, len: u64, access: Access) -> Result<()>;
    fn unmap(&self, va: Va, offset: u64, len: u64) -> Result<()>;
    /// A handle another process can import (a POSIX fd; a WIN32 handle on Windows).
    fn export(&self, page: &Physical) -> Result<OwnedFd>;
    fn import(&self, handle: OwnedFd, len: u64) -> Result<Physical>;
    /// Where `io::read` writes `len` mapped bytes at `va + offset`: device memory ordered on the
    /// backend's copy stream, or host memory.
    fn target(&self, va: Va, offset: u64, len: u64) -> Dst;
    /// An event after everything the copy stream holds so far.
    fn fence(&self) -> Result<Event>;
    /// An event after everything `stream` holds so far (a pin's release).
    fn record(&self, stream: usize) -> Result<Event>;
    fn query(&self, event: Event) -> Result<bool>;
    /// Orders `stream` (a `CUstream`; 0 the default) after `event`, without a host wait.
    fn wait(&self, event: Event, stream: usize) -> Result<()>;
    fn retire(&self, event: Event);
    /// After a device error: proves the copy stream's work complete or fails.
    fn drain(&self) -> Result<()>;
}

/// Pages in host memory: a memfd each, mapped into a reserved range. Its export is a read-only
/// description of the memfd, its capacity a given limit.
pub struct HostBackend {
    limit: u64,
    pages: Mutex<Pages>,
}

#[derive(Default)]
struct Pages {
    next: u64,
    /// id → (fd, bytes it counts against the limit: 0 for an imported page).
    live: HashMap<u64, (OwnedFd, u64)>,
}

impl HostBackend {
    pub fn new(limit: u64) -> HostBackend {
        HostBackend { limit, pages: Mutex::new(Pages::default()) }
    }

    fn created(pages: &Pages) -> u64 {
        pages.live.values().map(|(_, n)| n).sum()
    }

    fn insert(&self, fd: OwnedFd, counted: u64, len: u64) -> Physical {
        let mut pages = self.pages.lock();
        pages.next += 1;
        let id = pages.next;
        pages.live.insert(id, (fd, counted));
        Physical { id, len }
    }
}

fn prot(access: Access) -> libc::c_int {
    match access {
        Access::ReadWrite => libc::PROT_READ | libc::PROT_WRITE,
        Access::Read => libc::PROT_READ,
    }
}

fn host_err(op: &'static str) -> Error {
    let e = std::io::Error::last_os_error();
    Error::Device { op, code: e.raw_os_error().unwrap_or(-1), name: e.to_string() }
}

impl Backend for HostBackend {
    fn capacity(&self) -> u64 {
        self.limit
    }

    fn free(&self) -> Result<u64> {
        Ok(self.limit.saturating_sub(Self::created(&self.pages.lock())))
    }

    fn context(&self) -> u64 {
        0
    }

    fn granularity(&self) -> u64 {
        platform::shm::page_size() as u64
    }

    fn reserve(&self, len: u64) -> Result<Va> {
        let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE;
        // SAFETY: a new anonymous inaccessible mapping; nothing aliases it.
        let base = unsafe { libc::mmap(std::ptr::null_mut(), len as usize, libc::PROT_NONE, flags, -1, 0) };
        if base == libc::MAP_FAILED {
            return Err(host_err("reserve"));
        }
        Ok(Va { base: base as u64, len })
    }

    fn release(&self, va: Va) {
        // SAFETY: a reservation this backend made.
        unsafe { libc::munmap(va.base as *mut libc::c_void, va.len as usize) };
    }

    fn create(&self, len: u64) -> Result<Physical> {
        let pages = self.pages.lock();
        if Self::created(&pages).saturating_add(len) > self.limit {
            // CUDA_ERROR_OUT_OF_MEMORY, as a card answers.
            return Err(Error::Device { op: "create", code: 2, name: "out of memory".into() });
        }
        drop(pages);
        let fd = platform::shm::memfd_named("varena-page")?;
        // SAFETY: sizing a memfd only we hold.
        if unsafe { libc::ftruncate(fd.as_raw_fd(), len as libc::off_t) } != 0 {
            return Err(host_err("create"));
        }
        Ok(self.insert(fd, len, len))
    }

    fn destroy(&self, page: Physical) {
        self.pages.lock().live.remove(&page.id);
    }

    fn map(&self, va: Va, offset: u64, page: &Physical, access: Access) -> Result<()> {
        let pages = self.pages.lock();
        let (fd, _) = pages.live.get(&page.id).ok_or_else(|| Error::Invalid(format!("no page {}", page.id)))?;
        if offset.saturating_add(page.len) > va.len {
            return Err(Error::Invalid("map outside the reservation".into()));
        }
        let at = (va.base + offset) as *mut libc::c_void;
        // SAFETY: a fixed shared mapping inside our own reservation.
        let got = unsafe { libc::mmap(at, page.len as usize, prot(access), libc::MAP_SHARED | libc::MAP_FIXED, fd.as_raw_fd(), 0) };
        if got == libc::MAP_FAILED {
            return Err(host_err("map"));
        }
        Ok(())
    }

    fn protect(&self, va: Va, offset: u64, len: u64, access: Access) -> Result<()> {
        // SAFETY: a range of our own reservation.
        if unsafe { libc::mprotect((va.base + offset) as *mut libc::c_void, len as usize, prot(access)) } != 0 {
            return Err(host_err("protect"));
        }
        Ok(())
    }

    fn unmap(&self, va: Va, offset: u64, len: u64) -> Result<()> {
        let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE | libc::MAP_FIXED;
        let at = (va.base + offset) as *mut libc::c_void;
        // SAFETY: replaces a range of our own reservation with inaccessible anonymous memory.
        if unsafe { libc::mmap(at, len as usize, libc::PROT_NONE, flags, -1, 0) } == libc::MAP_FAILED {
            return Err(host_err("unmap"));
        }
        Ok(())
    }

    fn export(&self, page: &Physical) -> Result<OwnedFd> {
        let pages = self.pages.lock();
        let (fd, _) = pages.live.get(&page.id).ok_or_else(|| Error::Invalid(format!("no page {}", page.id)))?;
        Ok(platform::shm::reopen(fd.as_fd(), false)?)
    }

    fn import(&self, handle: OwnedFd, len: u64) -> Result<Physical> {
        Ok(self.insert(handle, 0, len))
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

    fn wait(&self, _: Event, _: usize) -> Result<()> {
        Ok(())
    }

    fn retire(&self, _: Event) {}

    fn drain(&self) -> Result<()> {
        Ok(())
    }
}

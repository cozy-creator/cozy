//! The pinned host tier for one weight set: a sparse memfd laid out exactly like the weight
//! set, mapped `MAP_SHARED`. Only filled regions consume RAM; eviction punches the region's
//! hole. An embedding owner such as TensorD passes the fd between processes: each adopter
//! sees the same prepared CPU bytes and the same per-region Ready words. Runtime uses those
//! bytes to populate model tensors; the host layout itself is not a constructed model.
//!
//! Cross-process safety rests on two facts. Contents are immutable (content-addressed), so two
//! processes filling one region write identical bytes. And every process that has a region
//! filling or ready holds a shared OFD lock on the region's word (its claim) until it lets the
//! region go; the only destructive act, the punch, needs the lock exclusive, so the last holder
//! to let go frees the RAM and no process ever loses pages it still registers. The kernel drops
//! a dead holder's claims.
//!
//! A sealed layout (`fill_sealed`) is filled through TensorFS's verified read path by its
//! owner (TensorD in Cozy). It is sealed against new writable mappings, resizing and punching;
//! the filler's existing mapping can finish the regions. Executors adopt it read-only
//! (`adopt_sealed`) and wait for each region's Ready state. Its CPU RAM goes when the last
//! descriptor and mapping do; GPU allocations have separate lifetimes.
//!
//! A streamed layout (`window::open_window`) is a sealed memfd too, for a weight set larger than
//! the host may hold: a few slots, each holding whichever region its machine staged there, and
//! the same state page plus a word per slot. An adopter reads region `r` only while it holds a
//! shared OFD lock on `r`'s word (`with_region`): that is the want the machine sees, and while it
//! is held the machine refills no slot holding `r`.

use std::collections::HashMap;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd, RawFd};
use std::sync::atomic::{fence, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use tensorfs_core::ids::ObjectRef;
use cozy_platform::shm::{self, Region, RESIZE_SEALS, WRITE_SEALS};

use crate::ceiling::Copy;
use crate::cuda;
use crate::io::{IoTally, Mode, Source};
use crate::layout::{ItemSource, Layout, PART_ALIGN};
use crate::{Error, Result};

const MAGIC: u64 = u64::from_le_bytes(*b"TFSPLN03");
const MAGIC_WINDOW: u64 = u64::from_le_bytes(*b"TFSWIN01");
/// The state page, at `data` (the layout's nbytes): magic, region count, digest (64 hex bytes),
/// one u32 word per region, each region's (offset, span), then the regions the layout was built
/// from (`names_off`); the file's last 8 bytes repeat `data`, so a holder without the layout
/// (`release_hold`, `sealed_regions`) finds the page.
const WORDS_OFF: usize = 8 + 8 + 64;
const READY: u32 = 1;
/// A sealed layout's region its filler could not fill: whoever waits for it gets an error.
const FAILED: u32 = 2;
/// A sealed layout's region its machine has not staged (`open_sealed`'s `staged` set): no
/// adopter waits for it; it reads the weight set's object files instead. The machine may stage
/// it later (`OpenFill::stage`), which clears the bit before the first byte lands.
const HOLE: u32 = 4;

/// A region's page lock in this process (`HostMem::pin_region`): never copied, registered,
/// refused, copied once unlocked.
const PIN_NONE: u8 = 0;
const PIN_YES: u8 = 1;
const PIN_REFUSED: u8 = 2;
const PIN_ONCE: u8 = 3;

fn ext_off(regions: usize) -> usize {
    (WORDS_OFF + 4 * regions).next_multiple_of(8)
}

/// The names section: its byte length, then per region a u32 name count and each name as a u32
/// length and its bytes.
fn names_off(regions: usize) -> usize {
    ext_off(regions) + 16 * regions
}

fn meta_len(regions: usize, names: usize) -> usize {
    (names_off(regions) + 8 + names + 8).next_multiple_of(PART_ALIGN as usize)
}

fn encode_names(names: &[Vec<String>]) -> Vec<u8> {
    let mut out = Vec::new();
    for region in names {
        out.extend((region.len() as u32).to_le_bytes());
        for n in region {
            out.extend((n.len() as u32).to_le_bytes());
            out.extend(n.as_bytes());
        }
    }
    out
}

fn decode_names(mut b: &[u8], regions: usize) -> Option<Vec<Vec<String>>> {
    let u32_of = |b: &mut &[u8]| -> Option<usize> {
        let (n, rest) = b.split_first_chunk::<4>()?;
        *b = rest;
        Some(u32::from_le_bytes(*n) as usize)
    };
    let mut out = Vec::with_capacity(regions.min(b.len() / 4));
    for _ in 0..regions {
        let count = u32_of(&mut b)?;
        let mut region = Vec::with_capacity(count.min(b.len() / 4));
        for _ in 0..count {
            let len = u32_of(&mut b)?;
            let (name, rest) = b.split_at_checked(len)?;
            region.push(String::from_utf8(name.to_vec()).ok()?);
            b = rest;
        }
        out.push(region);
    }
    b.is_empty().then_some(out)
}

fn read_at(fd: RawFd, buf: &mut [u8], at: usize) -> bool {
    // SAFETY: a read into our own buffer.
    unsafe { libc::pread(fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len(), at as libc::off_t) == buf.len() as isize }
}

fn u64_at(fd: RawFd, at: usize) -> Option<u64> {
    let mut b = [0u8; 8];
    read_at(fd, &mut b, at).then(|| u64::from_le_bytes(b))
}

/// The state page's names section length, when `data + meta_len` is exactly `size`.
fn names_len(fd: RawFd, data: usize, regions: usize, size: usize) -> Option<usize> {
    let len = usize::try_from(u64_at(fd, data.checked_add(names_off(regions))?)?).ok()?;
    (len < size && data.checked_add(meta_len(regions, len)) == Some(size)).then_some(len)
}

/// The regions a sealed layout (`fill_sealed`, `open_sealed`) was built from, read from its own
/// state page: what rebuilds its layout without the module (`Gpu::adopt`). Adoption checks the
/// rebuilt layout's digest, so the names are trusted no further than that.
pub fn sealed_regions(fd: RawFd) -> Result<Vec<Vec<String>>> {
    let bad = || Error::Invalid("host fd is not a sealed weight-set layout".into());
    let size = file_size(fd)?;
    let data = size.checked_sub(8).and_then(|t| u64_at(fd, t)).and_then(|d| usize::try_from(d).ok()).ok_or_else(bad)?;
    let (magic, n) = (u64_at(fd, data).ok_or_else(bad)?, u64_at(fd, data + 8).ok_or_else(bad)?);
    let n = usize::try_from(n).ok().filter(|&n| magic == MAGIC && n < size).ok_or_else(bad)?;
    let len = names_len(fd, data, n, size).ok_or_else(bad)?;
    let mut buf = vec![0u8; len];
    if !read_at(fd, &mut buf, data + names_off(n) + 8) {
        return Err(bad());
    }
    decode_names(&buf, n).ok_or_else(bad)
}

/// A streamed layout's window section, after the extents: slot count, slot span, then one u64
/// word per slot: `(region + 1) << 32 | READY` once staged, the region without READY while it
/// is refilled, 0 empty.
fn win_off(regions: usize) -> usize {
    ext_off(regions) + 16 * regions
}

fn meta_len_window(regions: usize, slots: usize) -> usize {
    (win_off(regions) + 16 + 8 * slots + 8).next_multiple_of(PART_ALIGN as usize)
}

/// The slot word of region `r` staged and complete.
pub(crate) fn staged(r: u32) -> u64 {
    (r as u64 + 1) << 32 | READY as u64
}

#[derive(Clone, Copy, Debug)]
pub struct Window {
    pub slots: usize,
    pub slot_span: u64,
}

/// An OFD byte-range lock on `fd`; Ok(false) when a non-waiting request conflicts.
fn ofd_lock(fd: BorrowedFd<'_>, start: usize, len: usize, region: Region, wait: bool) -> Result<bool> {
    shm::region_lock(fd, start, len, region, wait).map_err(crate::os("OFD lock"))
}

fn punch(fd: BorrowedFd<'_>, off: u64, span: u64) -> Result<()> {
    shm::punch(fd, off, span).map_err(crate::os("punch host region"))
}

/// A new memfd named for what it holds; the name may not carry a NUL.
fn memfd(name: String) -> Result<OwnedFd> {
    if name.contains('\0') {
        return Err(Error::Invalid("weight set name contains NUL".into()));
    }
    shm::memfd_named(&name).map_err(crate::os("memfd_create"))
}

fn os(what: &str) -> Error {
    Error::Io(format!("{what}: {}", std::io::Error::last_os_error()))
}

/// The magic of the state page the file's trailer points at (0 when there is none).
fn state_magic(fd: RawFd) -> Result<u64> {
    let size = file_size(fd)?;
    let read = |at: usize| {
        let mut b = [0u8; 8];
        // SAFETY: a read into our own buffer.
        let got = unsafe { libc::pread(fd, b.as_mut_ptr() as *mut libc::c_void, 8, at as libc::off_t) };
        (got == 8).then(|| u64::from_le_bytes(b))
    };
    let Some(data) = size.checked_sub(8).and_then(read) else { return Ok(0) };
    Ok(usize::try_from(data).ok().filter(|&d| d + 8 <= size).and_then(read).unwrap_or(0))
}

fn file_size(fd: RawFd) -> Result<usize> {
    // SAFETY: fstat into a zeroed struct.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut st) } != 0 {
        return Err(os("fstat host fd"));
    }
    Ok(st.st_size as usize)
}

/// A new open file description of the file behind `fd` (a dup would share `fd`'s, and with it
/// its OFD locks).
pub(crate) fn reopen(fd: RawFd) -> Result<OwnedFd> {
    reopen_as(fd, true)
}

fn reopen_as(fd: RawFd, write: bool) -> Result<OwnedFd> {
    // SAFETY: the caller holds `fd` open for the call.
    shm::reopen(unsafe { BorrowedFd::borrow_raw(fd) }, write).map_err(crate::os("reopen host fd"))
}

/// A claim on every region of the memfd behind `fd` for a process that keeps a tier without
/// a plane (for example, TensorD between executors): a new description holding one shared OFD lock over
/// the whole file. While it is held, no plane punches any region (an eviction there only ends
/// that plane's claim). End it with `release_hold`, which punches what no plane claims; merely
/// closing it leaves those regions resident until the memfd's last descriptor and mapping go.
/// Waits only while a punch is in progress.
pub fn hold(fd: RawFd) -> Result<OwnedFd> {
    let own = reopen(fd)?;
    ofd_lock(own.as_fd(), 0, 0, Region::Read, true)?;
    Ok(own)
}

/// End a `hold`: every Ready region no plane claims is punched (the hold was its last
/// claimant), then the description closes. Returns the bytes punched. A memfd of another
/// state-page format is only let go.
pub fn release_hold(hold: OwnedFd) -> Result<u64> {
    let fd = hold.as_raw_fd();
    let locks = hold.as_fd();
    // SAFETY: fstat into a zeroed struct.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut st) } != 0 {
        return Err(os("fstat held tier"));
    }
    let size = st.st_size as usize;
    let mut tail = [0u8; 8];
    // SAFETY: reads into our own buffers.
    let read = |buf: &mut [u8], at: usize| unsafe { libc::pread(fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len(), at as libc::off_t) } == buf.len() as isize;
    if size < 8 || !read(&mut tail, size - 8) {
        return Ok(0);
    }
    let data = u64::from_le_bytes(tail) as usize;
    let mut head = [0u8; 16];
    if data >= size || !read(&mut head, data) {
        return Ok(0);
    }
    let n = u64::from_le_bytes(head[8..].try_into().unwrap()) as usize;
    if u64::from_le_bytes(head[..8].try_into().unwrap()) != MAGIC || n >= size || names_len(fd, data, n, size).is_none() {
        return Ok(0);
    }
    let mut words = vec![0u8; 4 * n];
    let mut ext = vec![0u8; 16 * n];
    if !read(&mut words, data + WORDS_OFF) || !read(&mut ext, data + ext_off(n)) {
        return Err(os("read held tier's state page"));
    }
    let mut punched = 0;
    for r in 0..n {
        let w = u32::from_le_bytes(words[4 * r..4 * r + 4].try_into().unwrap());
        let at = data + WORDS_OFF + 4 * r;
        // Exclusive over our own shared hold: granted only when no plane claims the region.
        if w & READY == 0 || !ofd_lock(locks, at, 4, Region::Write, false)? {
            continue;
        }
        let cleared = (w & !READY).to_le_bytes();
        // SAFETY: a 4-byte write into the word we hold exclusively.
        if unsafe { libc::pwrite(fd, cleared.as_ptr() as *const libc::c_void, 4, at as libc::off_t) } != 4 {
            return Err(os("clear Ready"));
        }
        let off = u64::from_le_bytes(ext[16 * r..16 * r + 8].try_into().unwrap());
        let span = u64::from_le_bytes(ext[16 * r + 8..16 * r + 16].try_into().unwrap());
        punch(locks, off, span)?;
        punched += span;
    }
    Ok(punched)
}

pub struct HostMem {
    /// This description's own open file description: the mapping and every claim.
    fd: OwnedFd,
    /// A second description of the same memfd, never locked: the handle `fd()` hands to other
    /// processes, so a receiver that keeps it does not keep this process's claims alive.
    share: OwnedFd,
    ptr: *mut u8,
    /// Data span (the layout's `nbytes`); the state page follows it.
    data: usize,
    map_len: usize,
    regions: usize,
    /// A sealed layout, mapped read-only: never written, never punched by this process.
    sealed: bool,
    /// Registered ranges (offset, context as an address): unregistered at drop at the latest,
    /// since unmapping a registered range leaves the driver a stale pinned range at that VA.
    registered: Mutex<Vec<(u64, usize)>>,
    /// Each region's page lock (`pin_region`); taken under `pinning`.
    pinned: Vec<AtomicU8>,
    pinning: Mutex<()>,
    /// A streamed layout: its slots hold the regions; None, the regions sit at their offsets.
    window: Option<Window>,
    /// This process's readers per region of a streamed layout: one claim per description, held
    /// while any of them reads.
    readers: Mutex<HashMap<u32, usize>>,
}

// The mapping is shared memory addressed by offsets; all mutation goes through the plane's
// books or through atomics.
unsafe impl Send for HostMem {}
unsafe impl Sync for HostMem {}

impl HostMem {
    /// A fresh memfd for `layout`.
    pub fn create(name: &str, layout: &Layout) -> Result<HostMem> {
        Self::adopt_owned(memfd(format!("tfs-plane:{name}"))?, layout, false, true)
    }

    /// A layout's memory in a memfd `reserve` made: sized to the layout now (pages past it are
    /// freed, pages it lacks come on first write), its state page written fresh.
    pub fn create_in(reserved: OwnedFd, layout: &Layout) -> Result<HostMem> {
        Self::adopt_owned(reserved, layout, false, true)
    }

    /// Adopt a memfd another process (such as TensorD) created. Reopening through /proc gives this
    /// process its own open file description, so its OFD locks conflict with other holders'
    /// (a dup or an inherited fd would share theirs).
    pub fn adopt(fd: RawFd, layout: &Layout) -> Result<HostMem> {
        Self::adopt_owned(reopen(fd)?, layout, false, false)
    }

    /// Adopt a sealed layout (`fill_sealed`, `open_sealed`) read-only: its seals are the proof
    /// nobody else can change a byte; a region is used only once its filler marks it Ready.
    pub fn adopt_sealed(fd: RawFd, layout: &Layout) -> Result<HostMem> {
        // SAFETY: the caller holds `fd` open for the call.
        let seals = shm::seals(unsafe { BorrowedFd::borrow_raw(fd) }).map_err(crate::os("read seals"))?;
        if seals & RESIZE_SEALS != RESIZE_SEALS || seals & WRITE_SEALS == 0 {
            return Err(Error::Invalid("a sealed host layout must carry write, resize and seal seals".into()));
        }
        let own = reopen_as(fd, false)?;
        // The state page's magic says which (a size can match both: one region, one slot).
        let host = if state_magic(own.as_raw_fd())? == MAGIC_WINDOW {
            Self::adopt_window(own, layout)?
        } else {
            Self::adopt_owned(own, layout, true, false)?
        };
        if (0..host.regions as u32).any(|r| host.failed(r)) {
            return Err(Error::Invalid("a sealed host layout's fill failed".into()));
        }
        Ok(host)
    }

    pub fn failed(&self, r: u32) -> bool {
        self.word(r).load(Ordering::Acquire) & FAILED != 0
    }

    /// Region `r` is not staged in this sealed layout: read it from the object files.
    pub fn hole(&self, r: u32) -> bool {
        self.word(r).load(Ordering::Acquire) & (HOLE | READY) == HOLE
    }

    /// Until region `r` of a sealed layout is Ready: its filler sets the word after the last
    /// byte lands (or marks it failed). Its filler is this executor's machine, so a filler that
    /// dies takes the waiter with it; there is no deadline to guess.
    pub fn wait_ready(&self, r: u32) -> Result<()> {
        let mut pause = std::time::Duration::from_micros(20);
        loop {
            let w = self.word(r).load(Ordering::Acquire);
            if w & READY != 0 {
                return Ok(());
            }
            if w & FAILED != 0 {
                return Err(Error::Invalid(format!("sealed layout region {r}: its fill failed")));
            }
            if w & HOLE != 0 {
                return Err(Error::Invalid(format!("sealed layout region {r} is not staged")));
            }
            std::thread::sleep(pause);
            pause = (pause * 2).min(std::time::Duration::from_millis(1));
        }
    }

    pub fn sealed(&self) -> bool {
        self.sealed
    }

    pub fn window(&self) -> Option<Window> {
        self.window
    }

    /// A streamed layout's memfd: `slots` slots of the layout's largest region span, then the
    /// state page with the window section, every slot empty.
    pub(crate) fn create_window(name: &str, layout: &Layout, slots: usize) -> Result<HostMem> {
        let n = layout.regions.len();
        let slot_span = layout.regions.iter().map(|r| r.span).max().unwrap_or(0).max(PART_ALIGN);
        let slots = slots.max(1);
        let data = slots * slot_span as usize;
        let map_len = data + meta_len_window(n, slots);
        let fd = memfd(format!("tfs-window:{name}"))?;
        // SAFETY: sizing a descriptor we own; a sparse file costs no RAM until written.
        if unsafe { libc::ftruncate(fd.as_raw_fd(), map_len as libc::off_t) } != 0 {
            return Err(os("ftruncate window"));
        }
        let h = Self::map(fd, data, map_len, n, false, Some(Window { slots, slot_span }))?;
        // SAFETY: every write lies inside the state page of our writable mapping.
        unsafe {
            let page = h.ptr.add(data);
            for (r, reg) in layout.regions.iter().enumerate() {
                let e = page.add(ext_off(n) + 16 * r) as *mut u64;
                e.write_unaligned(reg.offset.to_le());
                e.add(1).write_unaligned(reg.span.to_le());
            }
            let w = page.add(win_off(n)) as *mut u64;
            w.write_unaligned((slots as u64).to_le());
            w.add(1).write_unaligned(slot_span.to_le());
            (page.add(meta_len_window(n, slots) - 8) as *mut u64).write_unaligned((data as u64).to_le());
        }
        let head = h.head();
        head.1.copy_from_slice(layout.digest.as_bytes());
        head.0[1] = n as u64;
        head.0[0] = MAGIC_WINDOW;
        Ok(h)
    }

    /// A streamed layout, read-only: its geometry from the state page (found through the
    /// trailer), checked against `layout` before anything is mapped at that size.
    fn adopt_window(fd: OwnedFd, layout: &Layout) -> Result<HostMem> {
        let n = layout.regions.len();
        let size = file_size(fd.as_raw_fd())?;
        let read_u64 = |at: usize| -> Result<u64> {
            let mut b = [0u8; 8];
            // SAFETY: a read into our own buffer.
            let got = unsafe { libc::pread(fd.as_raw_fd(), b.as_mut_ptr() as *mut libc::c_void, 8, at as libc::off_t) };
            if got != 8 {
                return Err(Error::Invalid("host fd is neither this layout nor a window of it".into()));
            }
            Ok(u64::from_le_bytes(b))
        };
        let data = read_u64(size.checked_sub(8).ok_or_else(|| Error::Invalid("empty host fd".into()))?)? as usize;
        let (magic, count) = (read_u64(data)?, read_u64(data + 8)?);
        let (slots, slot_span) = (read_u64(data + win_off(n))? as usize, read_u64(data + win_off(n) + 8)?);
        let largest = layout.regions.iter().map(|r| r.span).max().unwrap_or(0);
        if magic != MAGIC_WINDOW
            || count != n as u64
            || slots == 0
            || slot_span < largest
            || slots.checked_mul(slot_span as usize) != Some(data)
            || data.checked_add(meta_len_window(n, slots)) != Some(size)
        {
            return Err(Error::Invalid("host fd is neither this layout nor a window of it".into()));
        }
        let h = Self::map(fd, data, size, n, true, Some(Window { slots, slot_span }))?;
        // SAFETY: the digest lies inside the state page of our mapping.
        let digest = unsafe { std::slice::from_raw_parts(h.ptr.add(data + 16), 64) };
        if digest != layout.digest.as_bytes() {
            return Err(Error::Invalid("host fd holds a different weight-set layout".into()));
        }
        Ok(h)
    }

    fn map(fd: OwnedFd, data: usize, map_len: usize, regions: usize, sealed: bool, window: Option<Window>) -> Result<HostMem> {
        // SAFETY: shared mapping of a descriptor we hold, sized by the caller's checks.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                map_len,
                if sealed { libc::PROT_READ } else { libc::PROT_READ | libc::PROT_WRITE },
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(os("mmap host fd"));
        }
        let share = reopen_as(fd.as_raw_fd(), !sealed)?;
        Ok(HostMem {
            fd,
            share,
            ptr: ptr as *mut u8,
            data,
            map_len,
            regions,
            sealed,
            registered: Mutex::new(Vec::new()),
            pinned: (0..regions).map(|_| AtomicU8::new(PIN_NONE)).collect(),
            pinning: Mutex::new(()),
            window,
            readers: Mutex::new(HashMap::new()),
        })
    }

    pub(crate) fn raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }
    pub(crate) fn map_len(&self) -> usize {
        self.map_len
    }
    pub(crate) fn regions_len(&self) -> usize {
        self.regions
    }

    pub(crate) fn slot_word(&self, s: usize) -> &AtomicU64 {
        let w = self.window.expect("a streamed layout");
        assert!(s < w.slots);
        // SAFETY: in-bounds, 8-byte aligned (the window section is), lives as long as the mapping.
        unsafe { &*(self.ptr.add(self.data + win_off(self.regions) + 16 + 8 * s) as *const AtomicU64) }
    }

    pub(crate) fn mark_failed(&self, r: u32) {
        self.word(r).fetch_or(FAILED, Ordering::Release);
    }

    /// Region `r`'s offset in the layout (its extent in the state page).
    fn offset_of(&self, r: u32) -> u64 {
        assert!((r as usize) < self.regions);
        // SAFETY: the extent lies inside the state page of our mapping.
        u64::from_le(unsafe { (self.ptr.add(self.data + ext_off(self.regions) + 16 * r as usize) as *const u64).read_unaligned() })
    }

    /// Run `f` on region `r`'s bytes once they are in: for a layout, after its Ready word; for a
    /// streamed layout, in whichever slot holds `r`, which no refill touches until `f` returns
    /// (this process's claim on `r` is the machine's signal to stage it, and its guard).
    pub fn with_region<T>(&self, r: u32, f: impl FnOnce(*const u8) -> Result<T>) -> Result<T> {
        let Some(w) = self.window else {
            self.wait_ready(r)?;
            // SAFETY: inside the live mapping; a sealed layout's Ready region never changes.
            return f(unsafe { self.ptr.add(self.offset_of(r) as usize) });
        };
        self.read_begin(r)?;
        // The claim is visible to the machine before any slot word is read (its refill clears
        // the word, then looks for claims: one of the two sees the other).
        fence(Ordering::SeqCst);
        let tag = staged(r);
        let mut pause = std::time::Duration::from_micros(20);
        let res = loop {
            if let Some(s) = (0..w.slots).find(|&s| self.slot_word(s).load(Ordering::SeqCst) == tag) {
                // SAFETY: slot `s` lies inside the live mapping and holds `r` while we claim it.
                break f(unsafe { self.ptr.add(s * w.slot_span as usize) });
            }
            if self.failed(r) {
                break Err(Error::Invalid(format!("streamed layout region {r}: its machine could not stage it")));
            }
            std::thread::sleep(pause);
            pause = (pause * 2).min(std::time::Duration::from_millis(1));
        };
        self.read_end(r);
        res
    }

    fn read_begin(&self, r: u32) -> Result<()> {
        let mut readers = self.readers.lock().unwrap();
        let n = readers.entry(r).or_insert(0);
        if *n == 0 {
            self.lock(r, Region::Read, true)?;
        }
        *n += 1;
        Ok(())
    }

    fn read_end(&self, r: u32) {
        let mut readers = self.readers.lock().unwrap();
        if let Some(n) = readers.get_mut(&r) {
            *n -= 1;
            if *n == 0 {
                readers.remove(&r);
                self.unlock(r);
            }
        }
    }

    /// The regions some description claims (a reader's want), ascending, from the lock table.
    /// A query returns one conflicting lock, not necessarily the lowest, so the words on either
    /// side of it are asked again: about two queries per claimed run of words.
    pub(crate) fn claimed(&self) -> Result<Vec<u32>> {
        let lo = self.data + WORDS_OFF;
        let mut ranges = vec![(lo, lo + 4 * self.regions)];
        let mut out = Vec::new();
        while let Some((a, b)) = ranges.pop() {
            if a >= b {
                continue;
            }
            let Some((start, len)) = shm::region_holder(self.fd.as_fd(), a, b - a).map_err(crate::os("query region claims"))? else {
                continue;
            };
            let s = start.clamp(a, b);
            let e = if len == 0 { b } else { start + len }.clamp(s, b);
            out.extend(((s - lo) / 4..(e - lo).div_ceil(4)).map(|r| r as u32));
            ranges.push((a, s));
            ranges.push((e.max(s + 1), b));
        }
        out.sort_unstable();
        out.dedup();
        Ok(out)
    }

    /// `init`: this description sizes the memfd to the layout and writes its state page (a
    /// fresh or reserved memfd); otherwise an empty memfd is initialized and any other must
    /// already hold this layout.
    fn adopt_owned(fd: OwnedFd, layout: &Layout, sealed: bool, init: bool) -> Result<HostMem> {
        let regions = layout.regions.len();
        let data = layout.nbytes as usize;
        let size = file_size(fd.as_raw_fd())?;
        let fresh = (init || size == 0) && !sealed;
        let names = if fresh { encode_names(&layout.names) } else { Vec::new() };
        let map_len = if fresh {
            let map_len = data + meta_len(regions, names.len());
            // SAFETY: sizing our own descriptor; a sparse file costs no RAM until written.
            if unsafe { libc::ftruncate(fd.as_raw_fd(), map_len as libc::off_t) } != 0 {
                return Err(os("ftruncate host fd"));
            }
            map_len
        } else if names_len(fd.as_raw_fd(), data, regions, size).is_some() {
            size
        } else {
            return Err(Error::Invalid(format!("host fd is {size} bytes, not this layout's {data} and its state page")));
        };
        // SAFETY: shared mapping of a descriptor we hold, length checked above.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                map_len,
                if sealed { libc::PROT_READ } else { libc::PROT_READ | libc::PROT_WRITE },
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(os("mmap host fd"));
        }
        // Huge pages cut registration and TLB cost where shmem THP is enabled; advisory.
        shm::advise_huge(ptr as *mut u8, data);
        let share = reopen_as(fd.as_raw_fd(), !sealed)?;
        let h = HostMem {
            fd,
            share,
            ptr: ptr as *mut u8,
            data,
            map_len,
            regions,
            sealed,
            registered: Mutex::new(Vec::new()),
            pinned: (0..regions).map(|_| AtomicU8::new(PIN_NONE)).collect(),
            pinning: Mutex::new(()),
            window: None,
            readers: Mutex::new(HashMap::new()),
        };
        if fresh {
            let head = h.head();
            // SAFETY: the extents and the trailer lie inside the state page of our mapping.
            unsafe {
                let page = h.ptr.add(data);
                for (r, reg) in layout.regions.iter().enumerate() {
                    let e = page.add(ext_off(regions) + 16 * r) as *mut u64;
                    e.write_unaligned(reg.offset.to_le());
                    e.add(1).write_unaligned(reg.span.to_le());
                }
                let at = page.add(names_off(regions));
                (at as *mut u64).write_unaligned((names.len() as u64).to_le());
                std::ptr::copy_nonoverlapping(names.as_ptr(), at.add(8), names.len());
                (page.add(map_len - data - 8) as *mut u64).write_unaligned((data as u64).to_le());
            }
            head.1.copy_from_slice(layout.digest.as_bytes());
            head.0[1] = regions as u64;
            head.0[0] = MAGIC;
        } else {
            // Read through raw pointers: a sealed mapping is read-only.
            // SAFETY: the state page lies inside our mapping, 8-byte aligned (page aligned).
            let (magic, count, digest) = unsafe {
                let page = h.ptr.add(data);
                (
                    (page as *const u64).read(),
                    (page.add(8) as *const u64).read(),
                    std::slice::from_raw_parts(page.add(16), 64),
                )
            };
            if magic != MAGIC || count != regions as u64 || digest != layout.digest.as_bytes() {
                return Err(Error::Invalid("host fd holds a different weight-set layout".into()));
            }
        }
        Ok(h)
    }

    #[allow(clippy::mut_from_ref)]
    fn head(&self) -> (&mut [u64; 2], &mut [u8]) {
        // SAFETY: the state page lies inside our mapping, 8-byte aligned (page aligned).
        unsafe {
            let base = self.ptr.add(self.data);
            (
                &mut *(base as *mut [u64; 2]),
                std::slice::from_raw_parts_mut(base.add(16), 64),
            )
        }
    }

    fn word(&self, r: u32) -> &AtomicU32 {
        assert!((r as usize) < self.regions);
        // SAFETY: in-bounds, 4-byte aligned, lives as long as the mapping.
        unsafe { &*(self.ptr.add(self.data + WORDS_OFF + 4 * r as usize) as *const AtomicU32) }
    }

    /// The memfd to hand to another process (an unlocked description; see `share`).
    pub fn fd(&self) -> RawFd {
        self.share.as_raw_fd()
    }
    pub fn base(&self) -> *mut u8 {
        self.ptr
    }
    pub fn ready(&self, r: u32) -> bool {
        self.word(r).load(Ordering::Acquire) & READY != 0
    }
    pub fn set_ready(&self, r: u32, ready: bool) {
        assert!(!self.sealed, "a sealed layout's Ready words never change");
        if ready {
            self.word(r).fetch_or(READY, Ordering::Release);
        } else {
            self.word(r).fetch_and(!READY, Ordering::Release);
        }
    }

    /// Bytes of this memfd resident in RAM, from the kernel (`st_blocks`).
    pub fn resident_bytes(&self) -> u64 {
        // SAFETY: fstat into a zeroed struct.
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(self.fd.as_raw_fd(), &mut st) } != 0 {
            return 0;
        }
        st.st_blocks as u64 * 512
    }

    fn lock(&self, r: u32, region: Region, wait: bool) -> Result<bool> {
        assert!((r as usize) < self.regions);
        ofd_lock(self.fd.as_fd(), self.data + WORDS_OFF + 4 * r as usize, 4, region, wait)
    }

    /// Claim region `r` (shared): held while this description has it filling or ready. Waits
    /// only while another description is punching it (microseconds).
    pub fn claim(&self, r: u32) -> Result<()> {
        self.lock(r, Region::Read, true).map(|_| ())
    }

    /// Let region `r` go. The last claimant clears Ready and punches the region's RAM (true);
    /// while another description still claims it, only this claim ends (false).
    pub fn release(&self, r: u32, off: u64, span: u64) -> Result<bool> {
        if self.sealed {
            self.unlock(r);
            return Ok(false);
        }
        let last = self.lock(r, Region::Write, false)?;
        let punched = if last {
            self.set_ready(r, false);
            punch(self.fd.as_fd(), off, span).inspect_err(|_| self.unlock(r))?;
            true
        } else {
            false
        };
        self.unlock(r);
        Ok(punched)
    }

    fn unlock(&self, r: u32) {
        let _ = self.lock(r, Region::Unlock, false);
    }

    /// Page-lock `[off, off+len)` for DMA from every context (PORTABLE). The caller binds a
    /// context first.
    pub fn register(&self, off: u64, len: u64) -> Result<()> {
        let d = cuda::driver()?;
        // SAFETY: the range lies inside our live mapping and is page aligned.
        let rc = unsafe {
            (d.mem_host_register)(
                self.ptr.add(off as usize) as *mut libc::c_void,
                len as usize,
                cuda::MEMHOSTREGISTER_PORTABLE | if self.sealed { cuda::MEMHOSTREGISTER_READ_ONLY } else { 0 },
            )
        };
        if rc == cuda::ERROR_HOST_MEMORY_ALREADY_REGISTERED {
            return Ok(());
        }
        cuda::check("cuMemHostRegister", rc)?;
        let mut ctx: cuda::CUcontext = std::ptr::null_mut();
        // SAFETY: out-param.
        unsafe { (d.ctx_get_current)(&mut ctx) };
        self.registered.lock().unwrap().push((off, ctx as usize));
        Ok(())
    }

    /// Whether region `r` (`[off, off + len)`) copies to a device as one DMA: page-locked on
    /// its second copy, and kept until `unpin_region` or the layout closes. A first copy (a
    /// refill) goes through the staging buffers: locking 4 KiB pages runs at about 15 GB/s and
    /// does not scale with threads (4090 host, run 11942), below a GPU's H2D ceiling; a region
    /// copied again (streamed) pays it once. A refused region is never asked again. A sealed
    /// layout is mapped read-only, so only a driver that locks read-only memory can lock it (an
    /// RTX 4070 or 4090 answers NOT_SUPPORTED): that answer holds for the whole layout. Only a
    /// region whose bytes exist (Ready). The caller binds a context first.
    pub fn pin_region(&self, r: u32, off: u64, len: u64) -> bool {
        let state = &self.pinned[r as usize];
        let _g = self.pinning.lock().unwrap();
        match state.load(Ordering::Acquire) {
            PIN_YES => true,
            PIN_REFUSED => false,
            PIN_NONE => {
                state.store(PIN_ONCE, Ordering::Release);
                false
            }
            _ => match crate::faults::hit("pin").and_then(|_| self.register(off, len)) {
                Ok(()) => {
                    state.store(PIN_YES, Ordering::Release);
                    true
                }
                Err(Error::Cuda { code: cuda::ERROR_NOT_SUPPORTED, .. }) if self.sealed => {
                    self.pinned.iter().for_each(|s| s.store(PIN_REFUSED, Ordering::Release));
                    false
                }
                Err(_) => {
                    state.store(PIN_REFUSED, Ordering::Release);
                    false
                }
            },
        }
    }

    /// Unlock region `r` before its RAM is punched; a refill locks it again. The caller binds a
    /// context first.
    pub fn unpin_region(&self, r: u32, off: u64) -> Result<()> {
        let _g = self.pinning.lock().unwrap();
        if self.pinned[r as usize].swap(PIN_NONE, Ordering::AcqRel) == PIN_YES {
            self.unregister(off)?;
        }
        Ok(())
    }

    /// Unlock every region this process locked (a plane closing, while its context lives).
    pub fn unpin_all(&self) -> Result<()> {
        let _g = self.pinning.lock().unwrap();
        let offsets: Vec<u64> = self.registered.lock().unwrap().iter().map(|r| r.0).collect();
        for off in offsets {
            self.unregister(off)?;
        }
        for state in &self.pinned {
            state.store(PIN_NONE, Ordering::Release);
        }
        Ok(())
    }

    pub fn unregister(&self, off: u64) -> Result<()> {
        let d = cuda::driver()?;
        // SAFETY: `off` is the start of a range this process registered.
        cuda::check("cuMemHostUnregister", unsafe {
            (d.mem_host_unregister)(self.ptr.add(off as usize) as *mut libc::c_void)
        })?;
        self.registered.lock().unwrap().retain(|r| r.0 != off);
        Ok(())
    }
}

impl Drop for HostMem {
    fn drop(&mut self) {
        // A plane that never closed (poisoned) still owns registrations: end them first.
        for (off, ctx) in std::mem::take(self.registered.get_mut().unwrap()) {
            if let Ok(_g) = cuda::CtxGuard::enter(ctx as cuda::CUcontext) {
                let _ = self.unregister(off);
            }
        }
        // SAFETY: unmapping our own mapping, now unregistered.
        unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.map_len) };
    }
}

/// Memory for a layout still to be planned: a memfd of `bytes` whose pages are allocated now,
/// on `threads` threads. Allocation (the kernel zeroes and charges each 4 KiB page; shmem huge
/// pages are usually off) is most of a fill from the page cache; `fill_sealed` into the result
/// then mostly copies. Pages past the layout it receives are freed then.
pub fn reserve(name: &str, bytes: u64, threads: usize) -> Result<OwnedFd> {
    let len = crate::layout::align_up(bytes.max(1), PART_ALIGN) as usize;
    let fd = memfd(format!("tfs-reserve:{name}"))?;
    // SAFETY: plain syscalls on a descriptor we own; the mapping is ours and is unmapped
    // before returning (the pages stay in the memfd).
    unsafe {
        if libc::ftruncate(fd.as_raw_fd(), len as libc::off_t) != 0 {
            return Err(os("size reservation"));
        }
        let ptr = libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd.as_raw_fd(), 0);
        if ptr == libc::MAP_FAILED {
            return Err(os("map reservation"));
        }
        shm::advise_huge(ptr as *mut u8, len);
        shm::populate(ptr as *mut u8, len, threads);
        libc::munmap(ptr, len);
    }
    Ok(fd)
}

/// A sealed layout handed out before its bytes: sealed against resize, sealing and every new
/// write from the start, while this owner's mapping fills it (`run`, `fill`) and stages more of
/// it later (`stage`). Adopters use each region only once its Ready word is set, after its last
/// byte landed; a region left a hole they read from the object files.
pub struct OpenFill {
    host: HostMem,
}

/// Create `layout`'s memfd (in `reserved` when given: its pages exist, so the copy only maps
/// them) and seal it now; returns the memfd to hand out and the fill to run. `staged`: the
/// only regions this layout will hold (the rest are holes until staged); None: every region.
pub fn open_sealed(name: &str, layout: &Layout, reserved: Option<OwnedFd>, staged: Option<&[u32]>) -> Result<(OwnedFd, OpenFill)> {
    let host = match reserved {
        Some(fd) => HostMem::create_in(fd, layout)?,
        None => HostMem::create(name, layout)?,
    };
    if let Some(staged) = staged {
        for r in 0..layout.regions.len() as u32 {
            if !staged.contains(&r) {
                host.word(r).fetch_or(HOLE, Ordering::Release);
            }
        }
    }
    let fd = reopen(host.fd.as_raw_fd())?;
    // Future-write leaves this mapping writable.
    shm::seal(fd.as_fd()).map_err(crate::os("seal host layout"))?;
    Ok((fd, OpenFill { host }))
}

/// `open_sealed` and `OpenFill::run` in one: returns once every byte is in.
pub fn fill_sealed(name: &str, layout: &Layout, source: &Source, threads: usize, tally: &IoTally, reserved: Option<OwnedFd>) -> Result<OwnedFd> {
    let (fd, open) = open_sealed(name, layout, reserved, None)?;
    open.run(layout, source, threads, tally)?;
    Ok(fd)
}

impl OpenFill {
    /// Region `r` is a hole: not staged (yet).
    pub fn hole(&self, r: u32) -> bool {
        self.host.hole(r)
    }

    /// Fill every region through the verified read path (`Source::read_item`: page-cache
    /// copies, O_DIRECT for the rest) on `threads` threads, in layout order, setting each
    /// region's Ready word as its last item lands. Each object's residency is checked once (a
    /// check maps the file on overlayfs, which stalls every thread's page faults), and its
    /// descriptors close when its last item lands. The mapping goes when this returns.
    pub fn run(self, layout: &Layout, source: &Source, threads: usize, tally: &IoTally) -> Result<()> {
        self.fill(layout, source, threads, tally)
    }

    /// `run` without giving up the mapping: the owner keeps it to `stage` holes later.
    pub fn fill(&self, layout: &Layout, source: &Source, threads: usize, tally: &IoTally) -> Result<()> {
        let wanted: Vec<u32> = (0..layout.regions.len() as u32).filter(|&r| !self.host.hole(r)).collect();
        self.copy(layout, &wanted, source, threads, tally)
    }

    /// Take `regions` that are holes for staging: each stops being one at once, so an adopter
    /// asking from now on waits for its Ready word instead of reading the files. Returns those
    /// taken (a region already staged or staging is left alone); `stage` must follow.
    pub fn take(&self, regions: &[u32]) -> Vec<u32> {
        let holes = |&r: &u32| r < self.host.regions as u32 && self.host.word(r).fetch_and(!HOLE, Ordering::AcqRel) & HOLE != 0;
        regions.iter().copied().filter(holes).collect()
    }

    /// Fill the regions `take` returned; each turns Ready, or failed if this errors. Returns
    /// the bytes staged.
    pub fn stage(&self, taken: &[u32], layout: &Layout, source: &Source, threads: usize, tally: &IoTally) -> Result<u64> {
        self.copy(layout, taken, source, threads, tally).inspect_err(|_| {
            for &r in taken {
                if !self.host.ready(r) {
                    self.host.word(r).fetch_or(FAILED, Ordering::Release);
                }
            }
        })?;
        Ok(taken.iter().map(|&r| layout.regions[r as usize].span).sum())
    }

    /// Logged as a `disk` copy run against the volume's ceiling.
    fn copy(&self, layout: &Layout, wanted: &[u32], source: &Source, threads: usize, tally: &IoTally) -> Result<()> {
        let (t, host) = (Instant::now(), &self.host);
        let counts = || [&tally.cached_bytes, &tally.direct_bytes, &tally.buffered_bytes, &tally.inline_bytes].map(|c| c.load(Ordering::Relaxed));
        let before = counts();
        let mut region_of = Vec::new();
        for &r in wanted {
            region_of.extend(std::iter::repeat_n(r as usize, layout.regions[r as usize].items.len()));
        }
        let pending: Vec<AtomicUsize> = layout.regions.iter().map(|r| AtomicUsize::new(r.items.len())).collect();
        for &r in wanted {
            if layout.regions[r as usize].items.is_empty() {
                host.set_ready(r, true);
            }
        }
        let items: Vec<_> = wanted.iter().flat_map(|&r| layout.regions[r as usize].items.iter()).collect();
        let mut index: HashMap<&str, usize> = HashMap::new();
        let mut objects: Vec<&ObjectRef> = Vec::new();
        let owner: Vec<Option<usize>> = items
            .iter()
            .map(|it| match &it.source {
                ItemSource::Object(o) => Some(*index.entry(o.obj.sha256.as_str()).or_insert_with(|| {
                    objects.push(&o.obj);
                    objects.len() - 1
                })),
                ItemSource::Inline(_) => None,
            })
            .collect();
        let left: Vec<AtomicUsize> = (0..objects.len()).map(|_| AtomicUsize::new(0)).collect();
        for o in owner.iter().flatten() {
            left[*o].fetch_add(1, Ordering::Relaxed);
        }
        let resident: Vec<OnceLock<bool>> = (0..objects.len()).map(|_| OnceLock::new()).collect();
        let next = AtomicUsize::new(0);
        let failed: Mutex<Option<Error>> = Mutex::new(None);
        std::thread::scope(|s| {
            for _ in 0..threads.clamp(1, items.len().max(1)) {
                s.spawn(|| loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some(it) = items.get(k) else { return };
                    let mode = match owner[k] {
                        Some(o) if *resident[o].get_or_init(|| source.resident(objects[o], tally).unwrap_or(false)) => Mode::Cached,
                        Some(_) => Mode::Cache,
                        None => Mode::Cached,
                    };
                    // SAFETY: each item's range (and its padding) is this thread's alone; the layout
                    // keeps items disjoint inside the live mapping.
                    let dst = unsafe { host.base().add(it.offset as usize) };
                    if let Err(e) = unsafe { source.read_item(it, dst, mode, tally) } {
                        failed.lock().unwrap().get_or_insert(e);
                        next.store(items.len(), Ordering::Relaxed);
                        return;
                    }
                    if let Some(o) = owner[k] {
                        if left[o].fetch_sub(1, Ordering::AcqRel) == 1 {
                            source.close_object(&objects[o].sha256);
                        }
                    }
                    // The region's last item: its bytes are all in, so adopters may use it.
                    let r = region_of[k];
                    if pending[r].fetch_sub(1, Ordering::AcqRel) == 1 {
                        host.set_ready(r as u32, true);
                    }
                });
            }
        });
        if let Some(e) = failed.into_inner().unwrap() {
            return Err(e);
        }
        let moved = counts().iter().zip(before).map(|(a, b)| a - b).collect::<Vec<_>>();
        crate::ceiling::record(Copy {
            kind: "disk",
            gpu: -1,
            bytes: moved.iter().sum(),
            ns: t.elapsed().as_nanos() as u64,
            ceiling: source.disk_ceiling(&objects, tally).unwrap_or(0.0),
            cached: moved[0],
            ..Default::default()
        });
        Ok(())
    }
}

impl Drop for OpenFill {
    /// What the filler leaves unfinished (an error, a panic, a filler that never ran) is marked
    /// failed: no adopter waits forever for it or reads it. Holes stay holes.
    fn drop(&mut self) {
        for r in 0..self.host.regions as u32 {
            if !self.host.ready(r) && !self.host.hole(r) {
                self.host.word(r).fetch_or(FAILED, Ordering::Release);
            }
        }
    }
}

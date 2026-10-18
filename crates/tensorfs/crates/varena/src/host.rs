//! Sealed host layouts: a weight set's bytes in a memfd laid out exactly like the weight set,
//! with a state page after the data. Its owner (TensorD in Cozy) creates it with `open_sealed`,
//! seals it against every new writable mapping, resize and punch, and fills it through its own
//! mapping, setting each region's Ready word after the region's last byte. A plane adopts it
//! read-only (`Host::Sealed`), copies a region once its word is Ready, and reads a region left a
//! hole from the chunk files. The RAM goes when the last descriptor and mapping do.

use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd, RawFd};
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::Mutex;

use cozy_platform::shm::{self, RESIZE_SEALS, WRITE_SEALS};

use crate::cuda;
use crate::layout::{Layout, PART_ALIGN};
use crate::{Error, Result};

const MAGIC: u64 = u64::from_le_bytes(*b"TFSGPU01");
/// The state page, at `data` (the layout's nbytes): magic, region count, digest (64 hex bytes),
/// then one u32 word per region.
const WORDS_OFF: usize = 8 + 8 + 64;
const READY: u32 = 1;
/// A region its owner could not fill: whoever waits for it gets an error.
const FAILED: u32 = 2;
/// A region its owner does not hold: adopters read it from the chunk files. The owner may take
/// it later (`SealedFill::take`), which clears the bit before the first byte lands.
const HOLE: u32 = 4;

/// A region's page lock in this process (`HostMem::pin_region`): not tried, registered, refused.
const PIN_NONE: u8 = 0;
const PIN_YES: u8 = 1;
const PIN_REFUSED: u8 = 2;

fn meta_len(regions: usize) -> usize {
    (WORDS_OFF + 4 * regions).next_multiple_of(PART_ALIGN as usize)
}

fn os(what: &str) -> Error {
    Error::Io(format!("{what}: {}", std::io::Error::last_os_error()))
}

/// One mapping of a layout's memfd: the owner's writable one, or an adopter's read-only one.
pub struct HostMem {
    fd: OwnedFd,
    ptr: *mut u8,
    data: usize,
    map_len: usize,
    regions: usize,
    writable: bool,
    /// Registered ranges (offset, context as an address): unregistered at drop at the latest,
    /// since unmapping a registered range leaves the driver a stale pinned range at that VA.
    registered: Mutex<Vec<(u64, usize)>>,
    pinned: Vec<AtomicU8>,
    pinning: Mutex<()>,
}

// The mapping is shared memory addressed by offsets; mutation goes through atomics or the
// owner's exclusive fill.
unsafe impl Send for HostMem {}
unsafe impl Sync for HostMem {}

impl HostMem {
    fn map(fd: OwnedFd, layout: &Layout, writable: bool) -> Result<HostMem> {
        let regions = layout.regions.len();
        let data = layout.nbytes as usize;
        let map_len = data + meta_len(regions);
        // SAFETY: fstat into a zeroed struct.
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd.as_raw_fd(), &mut st) } != 0 {
            return Err(os("fstat host layout"));
        }
        if writable {
            // SAFETY: sizing a fresh descriptor we own; a sparse file costs no RAM until written.
            if unsafe { libc::ftruncate(fd.as_raw_fd(), map_len as libc::off_t) } != 0 {
                return Err(os("size host layout"));
            }
        } else if st.st_size as usize != map_len {
            return Err(Error::Invalid(format!("host layout is {} bytes, this layout needs {map_len}", st.st_size)));
        }
        let prot = if writable { libc::PROT_READ | libc::PROT_WRITE } else { libc::PROT_READ };
        // SAFETY: shared mapping of a descriptor we hold, length checked above.
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), map_len, prot, libc::MAP_SHARED, fd.as_raw_fd(), 0) };
        if ptr == libc::MAP_FAILED {
            return Err(os("mmap host layout"));
        }
        // Huge pages cut registration and TLB cost where shmem THP is enabled; advisory.
        shm::advise_huge(ptr as *mut u8, data);
        Ok(HostMem {
            fd,
            ptr: ptr as *mut u8,
            data,
            map_len,
            regions,
            writable,
            registered: Mutex::new(Vec::new()),
            pinned: (0..regions).map(|_| AtomicU8::new(PIN_NONE)).collect(),
            pinning: Mutex::new(()),
        })
    }

    /// Adopt a sealed layout read-only: its seals are the proof nobody else changes a byte; its
    /// state page must name this layout.
    pub fn adopt_sealed(fd: RawFd, layout: &Layout) -> Result<HostMem> {
        // SAFETY: the caller holds `fd` open for the call.
        let fd = unsafe { BorrowedFd::borrow_raw(fd) };
        let seals = shm::seals(fd).map_err(crate::os("read seals"))?;
        if seals & RESIZE_SEALS != RESIZE_SEALS || seals & WRITE_SEALS == 0 {
            return Err(Error::Invalid("a sealed host layout must carry write, resize and seal seals".into()));
        }
        let h = Self::map(shm::reopen(fd, false).map_err(crate::os("reopen host layout"))?, layout, false)?;
        // SAFETY: the state page lies inside our mapping, 8-byte aligned (page aligned).
        let (magic, count, digest) = unsafe {
            let page = h.ptr.add(h.data);
            ((page as *const u64).read(), (page.add(8) as *const u64).read(), std::slice::from_raw_parts(page.add(16), 64))
        };
        if magic != MAGIC || count != h.regions as u64 || digest != layout.digest.as_bytes() {
            return Err(Error::Invalid("the host layout holds a different weight-set layout".into()));
        }
        if (0..h.regions as u32).any(|r| h.failed(r)) {
            return Err(Error::Invalid("the host layout's fill failed".into()));
        }
        Ok(h)
    }

    fn word(&self, r: u32) -> &AtomicU32 {
        assert!((r as usize) < self.regions);
        // SAFETY: in-bounds, 4-byte aligned, lives as long as the mapping.
        unsafe { &*(self.ptr.add(self.data + WORDS_OFF + 4 * r as usize) as *const AtomicU32) }
    }

    pub fn base(&self) -> *mut u8 {
        self.ptr
    }
    pub fn ready(&self, r: u32) -> bool {
        self.word(r).load(Ordering::Acquire) & READY != 0
    }
    pub fn failed(&self, r: u32) -> bool {
        self.word(r).load(Ordering::Acquire) & FAILED != 0
    }
    /// Region `r` is not held here: read it from the chunk files.
    pub fn hole(&self, r: u32) -> bool {
        self.word(r).load(Ordering::Acquire) & (HOLE | READY) == HOLE
    }

    /// Until region `r` is Ready: its owner sets the word after the last byte lands (or marks it
    /// failed). The owner is this executor's machine, so an owner that dies takes the waiter with
    /// it; there is no deadline to guess.
    pub fn wait_ready(&self, r: u32) -> Result<()> {
        let mut pause = std::time::Duration::from_micros(20);
        loop {
            let w = self.word(r).load(Ordering::Acquire);
            if w & READY != 0 {
                return Ok(());
            }
            if w & FAILED != 0 {
                return Err(Error::Invalid(format!("host layout region {r}: its fill failed")));
            }
            if w & HOLE != 0 {
                return Err(Error::Invalid(format!("host layout region {r} is a hole")));
            }
            std::thread::sleep(pause);
            pause = (pause * 2).min(std::time::Duration::from_millis(1));
        }
    }

    /// Page-lock region `r` (`[off, off + len)`) once, so its copies are one DMA, and keep it until
    /// the layout closes. Returns (locked, refused now): a refused region is never asked again and
    /// copies through the staging buffers. A read-only mapping locks only where the driver locks
    /// read-only memory (an RTX 4070 answers NOT_SUPPORTED): that answer holds for the whole
    /// layout. The caller binds a context first.
    pub fn pin_region(&self, r: u32, off: u64, len: u64) -> (bool, bool) {
        let state = &self.pinned[r as usize];
        let _g = self.pinning.lock().unwrap();
        match state.load(Ordering::Acquire) {
            PIN_YES => (true, false),
            PIN_REFUSED => (false, false),
            _ => match crate::faults::hit("pin").and_then(|_| self.register(off, len)) {
                Ok(()) => {
                    state.store(PIN_YES, Ordering::Release);
                    (true, false)
                }
                Err(Error::Cuda { code: cuda::ERROR_NOT_SUPPORTED, .. }) => {
                    for state in &self.pinned {
                        let _ = state.compare_exchange(PIN_NONE, PIN_REFUSED, Ordering::AcqRel, Ordering::Acquire);
                    }
                    (false, true)
                }
                Err(_) => {
                    state.store(PIN_REFUSED, Ordering::Release);
                    (false, true)
                }
            },
        }
    }

    fn register(&self, off: u64, len: u64) -> Result<()> {
        let d = cuda::driver()?;
        let flags = cuda::MEMHOSTREGISTER_PORTABLE | if self.writable { 0 } else { cuda::MEMHOSTREGISTER_READ_ONLY };
        // SAFETY: the range lies inside our live mapping and is page aligned.
        let rc = unsafe { (d.mem_host_register)(self.ptr.add(off as usize) as *mut libc::c_void, len as usize, flags) };
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

    /// Unlock every region this process locked (a plane closing, while its context lives).
    pub fn unpin_all(&self) -> Result<()> {
        let _g = self.pinning.lock().unwrap();
        let d = cuda::driver()?;
        for (off, _) in std::mem::take(&mut *self.registered.lock().unwrap()) {
            // SAFETY: `off` starts a range this process registered.
            cuda::check("cuMemHostUnregister", unsafe { (d.mem_host_unregister)(self.ptr.add(off as usize) as *mut libc::c_void) })?;
        }
        for state in &self.pinned {
            state.store(PIN_NONE, Ordering::Release);
        }
        Ok(())
    }
}

impl Drop for HostMem {
    fn drop(&mut self) {
        // A plane that never closed (poisoned) still owns registrations: end them first.
        for (off, ctx) in std::mem::take(self.registered.get_mut().unwrap()) {
            if let (Ok(_g), Ok(d)) = (cuda::CtxGuard::enter(ctx as cuda::CUcontext), cuda::driver()) {
                // SAFETY: a range this process registered.
                unsafe { (d.mem_host_unregister)(self.ptr.add(off as usize) as *mut libc::c_void) };
            }
        }
        // SAFETY: unmapping our own mapping, now unregistered.
        unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.map_len) };
    }
}

/// The owner's side of a sealed layout: its writable mapping. Regions it leaves unfinished when
/// dropped are marked failed (holes stay holes), so no adopter waits for them forever.
pub struct SealedFill {
    host: HostMem,
}

/// Create `layout`'s memfd and seal it now: returns the memfd to hand out (sealed against every
/// new writable mapping, resize and punch) and the fill, whose mapping stays writable. `held`:
/// the only regions it will hold, the rest holes; None: every region.
pub fn open_sealed(name: &str, layout: &Layout, held: Option<&[u32]>) -> Result<(OwnedFd, SealedFill)> {
    if name.contains('\0') {
        return Err(Error::Invalid("a layout name contains NUL".into()));
    }
    let fd = shm::memfd_named(&format!("tfs-gpu:{name}")).map_err(crate::os("memfd_create"))?;
    let host = HostMem::map(fd, layout, true)?;
    // SAFETY: the state page lies inside our writable mapping, page aligned.
    unsafe {
        let page = host.ptr.add(host.data);
        std::slice::from_raw_parts_mut(page.add(16), 64).copy_from_slice(layout.digest.as_bytes());
        (page.add(8) as *mut u64).write(layout.regions.len() as u64);
        (page as *mut u64).write(MAGIC);
    }
    for r in 0..layout.regions.len() as u32 {
        if held.is_some_and(|h| !h.contains(&r)) {
            host.word(r).fetch_or(HOLE, Ordering::Release);
        }
    }
    let out = shm::reopen(host.fd.as_fd(), true).map_err(crate::os("reopen host layout"))?;
    // Future-write: this mapping stays writable, no new one can be.
    shm::seal(out.as_fd()).map_err(crate::os("seal host layout"))?;
    Ok((out, SealedFill { host }))
}

impl SealedFill {
    /// The writable mapping: region `r`'s chunk `c` goes at `layout.regions[r].chunks[c].offset`.
    pub fn base(&self) -> *mut u8 {
        self.host.ptr
    }
    pub fn hole(&self, r: u32) -> bool {
        self.host.hole(r)
    }
    /// Region `r`'s bytes are all in: adopters may copy it.
    pub fn set_ready(&self, r: u32) {
        self.host.word(r).fetch_or(READY, Ordering::Release);
    }
    pub fn fail(&self, r: u32) {
        self.host.word(r).fetch_or(FAILED, Ordering::Release);
    }
    /// Take holes for filling: each stops being one at once, so an adopter asking from now on
    /// waits for its Ready word. Returns those taken (a region not a hole is left alone).
    pub fn take(&self, regions: &[u32]) -> Vec<u32> {
        let h = &self.host;
        regions.iter().copied().filter(|&r| (r as usize) < h.regions && h.word(r).fetch_and(!HOLE, Ordering::AcqRel) & HOLE != 0).collect()
    }
}

impl Drop for SealedFill {
    fn drop(&mut self) {
        for r in 0..self.host.regions as u32 {
            if !self.host.ready(r) && !self.host.hole(r) {
                self.fail(r);
            }
        }
    }
}

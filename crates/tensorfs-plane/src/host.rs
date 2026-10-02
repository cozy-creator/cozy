//! The pinned host tier for one weight set: a sparse memfd laid out exactly like the weight
//! set, mapped `MAP_SHARED`. Only filled regions consume RAM; eviction punches the region's
//! hole. The fd is the handle a Worker passes between processes: any process that adopts it
//! sees the same bytes and the same per-region Ready words.
//!
//! Cross-process safety rests on two facts. Contents are immutable (content-addressed), so two
//! processes filling one region write identical bytes. And every process that has a region
//! filling or ready holds a shared OFD lock on the region's word (its claim) until it lets the
//! region go; the only destructive act, the punch, needs the lock exclusive, so the last holder
//! to let go frees the RAM and no process ever loses pages it still registers. The kernel drops
//! a dead holder's claims.

use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

use crate::cuda;
use crate::layout::{Layout, PART_ALIGN};
use crate::{Error, Result};

const MAGIC: u64 = u64::from_le_bytes(*b"TFSPLN02");
/// The state page, at `data` (the layout's nbytes): magic, region count, digest (64 hex bytes),
/// one u32 word per region, then each region's (offset, span); the file's last 8 bytes repeat
/// `data`, so a holder without the layout (`release_hold`) finds the page.
const WORDS_OFF: usize = 8 + 8 + 64;
const READY: u32 = 1;

fn ext_off(regions: usize) -> usize {
    (WORDS_OFF + 4 * regions).next_multiple_of(8)
}

fn meta_len(regions: usize) -> usize {
    (ext_off(regions) + 16 * regions + 8).next_multiple_of(PART_ALIGN as usize)
}

/// An OFD byte-range lock on `fd`; Ok(false) when a non-waiting request conflicts.
fn ofd_lock(fd: RawFd, start: usize, len: usize, kind: libc::c_short, wait: bool) -> Result<bool> {
    let fl = libc::flock {
        l_type: kind,
        l_whence: libc::SEEK_SET as libc::c_short,
        l_start: start as libc::off_t,
        l_len: len as libc::off_t,
        l_pid: 0,
    };
    let cmd = if wait { libc::F_OFD_SETLKW } else { libc::F_OFD_SETLK };
    // SAFETY: OFD byte-range lock on a descriptor the caller holds.
    if unsafe { libc::fcntl(fd, cmd, &fl) } == 0 {
        return Ok(true);
    }
    let e = std::io::Error::last_os_error();
    match e.raw_os_error() {
        Some(libc::EAGAIN) | Some(libc::EACCES) if !wait => Ok(false),
        _ => Err(Error::Io(format!("OFD lock at {start}: {e}"))),
    }
}

fn punch(fd: RawFd, off: u64, span: u64) -> Result<()> {
    // SAFETY: punching a hole in a memfd the caller holds.
    let rc = unsafe {
        libc::fallocate(fd, libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE, off as libc::off_t, span as libc::off_t)
    };
    if rc != 0 {
        return Err(os("punch host region"));
    }
    Ok(())
}

fn os(what: &str) -> Error {
    Error::Io(format!("{what}: {}", std::io::Error::last_os_error()))
}

/// A new open file description of the file behind `fd` (a dup would share `fd`'s, and with it
/// its OFD locks).
fn reopen(fd: RawFd) -> Result<OwnedFd> {
    let path = CString::new(format!("/proc/self/fd/{fd}")).expect("no NUL");
    // SAFETY: plain open of a magic link to a descriptor the caller holds.
    let own = unsafe { libc::open(path.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
    if own < 0 {
        return Err(os("reopen host fd"));
    }
    // SAFETY: fresh descriptor from the kernel.
    Ok(unsafe { OwnedFd::from_raw_fd(own) })
}

/// A claim on every region of the memfd behind `fd` for a process that keeps a tier without
/// a plane (the Worker between executors): a new description holding one shared OFD lock over
/// the whole file. While it is held, no plane punches any region (an eviction there only ends
/// that plane's claim). End it with `release_hold`, which punches what no plane claims; merely
/// closing it leaves those regions resident until the memfd's last descriptor and mapping go.
/// Waits only while a punch is in progress.
pub fn hold(fd: RawFd) -> Result<OwnedFd> {
    let own = reopen(fd)?;
    ofd_lock(own.as_raw_fd(), 0, 0, libc::F_RDLCK as _, true)?;
    Ok(own)
}

/// End a `hold`: every Ready region no plane claims is punched (the hold was its last
/// claimant), then the description closes. Returns the bytes punched. A memfd of another
/// state-page format is only let go.
pub fn release_hold(hold: OwnedFd) -> Result<u64> {
    let fd = hold.as_raw_fd();
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
    if u64::from_le_bytes(head[..8].try_into().unwrap()) != MAGIC || data.checked_add(meta_len(n)) != Some(size) {
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
        if w & READY == 0 || !ofd_lock(fd, at, 4, libc::F_WRLCK as _, false)? {
            continue;
        }
        let cleared = (w & !READY).to_le_bytes();
        // SAFETY: a 4-byte write into the word we hold exclusively.
        if unsafe { libc::pwrite(fd, cleared.as_ptr() as *const libc::c_void, 4, at as libc::off_t) } != 4 {
            return Err(os("clear Ready"));
        }
        let off = u64::from_le_bytes(ext[16 * r..16 * r + 8].try_into().unwrap());
        let span = u64::from_le_bytes(ext[16 * r + 8..16 * r + 16].try_into().unwrap());
        punch(fd, off, span)?;
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
    /// Registered ranges (offset, context as an address): unregistered at drop at the latest,
    /// since unmapping a registered range leaves the driver a stale pinned range at that VA.
    registered: Mutex<Vec<(u64, usize)>>,
}

// The mapping is shared memory addressed by offsets; all mutation goes through the plane's
// books or through atomics.
unsafe impl Send for HostMem {}
unsafe impl Sync for HostMem {}

impl HostMem {
    /// A fresh memfd for `layout`.
    pub fn create(name: &str, layout: &Layout) -> Result<HostMem> {
        let cname = CString::new(format!("tfs-plane:{name}"))
            .map_err(|_| Error::Invalid("weight set name contains NUL".into()))?;
        // SAFETY: plain syscall; the name is NUL-terminated.
        let fd = unsafe { libc::memfd_create(cname.as_ptr(), libc::MFD_CLOEXEC) };
        if fd < 0 {
            return Err(os("memfd_create"));
        }
        // SAFETY: we own the fresh descriptor.
        Self::adopt_owned(unsafe { OwnedFd::from_raw_fd(fd) }, layout)
    }

    /// Adopt a memfd another process (the Worker) created. Reopening through /proc gives this
    /// process its own open file description, so its OFD locks conflict with other holders'
    /// (a dup or an inherited fd would share theirs).
    pub fn adopt(fd: RawFd, layout: &Layout) -> Result<HostMem> {
        Self::adopt_owned(reopen(fd)?, layout)
    }

    fn adopt_owned(fd: OwnedFd, layout: &Layout) -> Result<HostMem> {
        let regions = layout.regions.len();
        let data = layout.nbytes as usize;
        let map_len = data + meta_len(regions);
        // SAFETY: fstat into a zeroed struct.
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd.as_raw_fd(), &mut st) } != 0 {
            return Err(os("fstat host fd"));
        }
        let fresh = st.st_size == 0;
        if fresh {
            // SAFETY: sizing our own descriptor; a sparse file costs no RAM until written.
            if unsafe { libc::ftruncate(fd.as_raw_fd(), map_len as libc::off_t) } != 0 {
                return Err(os("ftruncate host fd"));
            }
        } else if st.st_size as usize != map_len {
            return Err(Error::Invalid(format!(
                "host fd is {} bytes, this layout needs {map_len}",
                st.st_size
            )));
        }
        // SAFETY: shared mapping of a descriptor we hold, length checked above.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                map_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(os("mmap host fd"));
        }
        // Huge pages cut registration and TLB cost where shmem THP is enabled; advisory.
        // SAFETY: advice on our own mapping.
        unsafe { libc::madvise(ptr, data, libc::MADV_HUGEPAGE) };
        let share = reopen(fd.as_raw_fd())?;
        let h = HostMem {
            fd,
            share,
            ptr: ptr as *mut u8,
            data,
            map_len,
            regions,
            registered: Mutex::new(Vec::new()),
        };
        let head = h.head();
        if fresh {
            // SAFETY: the extents and the trailer lie inside the state page of our mapping.
            unsafe {
                let page = h.ptr.add(data);
                for (r, reg) in layout.regions.iter().enumerate() {
                    let e = page.add(ext_off(regions) + 16 * r) as *mut u64;
                    e.write_unaligned(reg.offset.to_le());
                    e.add(1).write_unaligned(reg.span.to_le());
                }
                (page.add(meta_len(regions) - 8) as *mut u64).write_unaligned((data as u64).to_le());
            }
            head.1.copy_from_slice(layout.digest.as_bytes());
            head.0[1] = regions as u64;
            head.0[0] = MAGIC;
        } else if head.0[0] != MAGIC
            || head.0[1] != regions as u64
            || head.1 != layout.digest.as_bytes()
        {
            return Err(Error::Invalid(
                "host fd holds a different weight-set layout".into(),
            ));
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

    fn lock(&self, r: u32, kind: libc::c_short, wait: bool) -> Result<bool> {
        assert!((r as usize) < self.regions);
        ofd_lock(self.fd.as_raw_fd(), self.data + WORDS_OFF + 4 * r as usize, 4, kind, wait)
    }

    /// Claim region `r` (shared): held while this description has it filling or ready. Waits
    /// only while another description is punching it (microseconds).
    pub fn claim(&self, r: u32) -> Result<()> {
        self.lock(r, libc::F_RDLCK as _, true).map(|_| ())
    }

    /// Let region `r` go. The last claimant clears Ready and punches the region's RAM (true);
    /// while another description still claims it, only this claim ends (false).
    pub fn release(&self, r: u32, off: u64, span: u64) -> Result<bool> {
        let last = self.lock(r, libc::F_WRLCK as _, false)?;
        let punched = if last {
            self.set_ready(r, false);
            punch(self.fd.as_raw_fd(), off, span).inspect_err(|_| self.unlock(r))?;
            true
        } else {
            false
        };
        self.unlock(r);
        Ok(punched)
    }

    fn unlock(&self, r: u32) {
        let _ = self.lock(r, libc::F_UNLCK as _, false);
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
                cuda::MEMHOSTREGISTER_PORTABLE,
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

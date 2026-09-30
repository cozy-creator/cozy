//! The pinned host tier for one weight set: a sparse memfd laid out exactly like the weight
//! set, mapped `MAP_SHARED`. Only filled regions consume RAM; eviction punches the region's
//! hole. The fd is the handle a Worker passes between processes: any process that adopts it
//! sees the same bytes and the same per-region Ready words.
//!
//! Cross-process safety rests on two facts. Contents are immutable (content-addressed), so two
//! processes filling one region write identical bytes. And the only destructive act, the
//! punch, needs an exclusive OFD lock on the region's word while a filler holds a shared one;
//! the kernel drops both when their holder dies.

use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

use crate::cuda;
use crate::layout::{Layout, PART_ALIGN};
use crate::{Error, Result};

const MAGIC: u64 = u64::from_le_bytes(*b"TFSPLN01");
/// Header: magic, region count, digest (64 hex bytes), then one u32 word per region.
const WORDS_OFF: usize = 8 + 8 + 64;
const READY: u32 = 1;

fn os(what: &str) -> Error {
    Error::Io(format!("{what}: {}", std::io::Error::last_os_error()))
}

pub struct HostMem {
    fd: OwnedFd,
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
    fn meta_len(regions: usize) -> usize {
        (WORDS_OFF + 4 * regions).div_ceil(PART_ALIGN as usize) * PART_ALIGN as usize
    }

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
        let path = CString::new(format!("/proc/self/fd/{fd}")).expect("no NUL");
        // SAFETY: plain open of a magic link to a descriptor the caller owns.
        let own = unsafe { libc::open(path.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
        if own < 0 {
            return Err(os("reopen host fd"));
        }
        // SAFETY: fresh descriptor from the kernel.
        Self::adopt_owned(unsafe { OwnedFd::from_raw_fd(own) }, layout)
    }

    fn adopt_owned(fd: OwnedFd, layout: &Layout) -> Result<HostMem> {
        let regions = layout.regions.len();
        let data = layout.nbytes as usize;
        let map_len = data + Self::meta_len(regions);
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
        let h = HostMem {
            fd,
            ptr: ptr as *mut u8,
            data,
            map_len,
            regions,
            registered: Mutex::new(Vec::new()),
        };
        let head = h.head();
        if fresh {
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

    pub fn fd(&self) -> RawFd {
        self.fd.as_raw_fd()
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
        let fl = libc::flock {
            l_type: kind,
            l_whence: libc::SEEK_SET as libc::c_short,
            l_start: (self.data + WORDS_OFF + 4 * r as usize) as libc::off_t,
            l_len: 4,
            l_pid: 0,
        };
        let cmd = if wait { libc::F_OFD_SETLKW } else { libc::F_OFD_SETLK };
        // SAFETY: OFD byte-range lock on our own descriptor.
        if unsafe { libc::fcntl(self.fd.as_raw_fd(), cmd, &fl) } == 0 {
            return Ok(true);
        }
        let e = std::io::Error::last_os_error();
        match e.raw_os_error() {
            Some(libc::EAGAIN) | Some(libc::EACCES) if !wait => Ok(false),
            _ => Err(Error::Io(format!("OFD lock region {r}: {e}"))),
        }
    }

    /// Shared hold for the duration of a fill; blocks only while a punch of the same region is
    /// in progress (microseconds).
    pub fn fill_lock(&self, r: u32) -> Result<()> {
        self.lock(r, libc::F_RDLCK as _, true).map(|_| ())
    }
    pub fn unlock(&self, r: u32) {
        let _ = self.lock(r, libc::F_UNLCK as _, false);
    }

    /// Clear Ready and free the region's RAM. Refuses (false) while any process is filling it.
    pub fn punch(&self, r: u32, off: u64, span: u64) -> Result<bool> {
        if !self.lock(r, libc::F_WRLCK as _, false)? {
            return Ok(false);
        }
        self.set_ready(r, false);
        // SAFETY: punching a hole in our own memfd; the range lies inside the data span.
        let rc = unsafe {
            libc::fallocate(
                self.fd.as_raw_fd(),
                libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                off as libc::off_t,
                span as libc::off_t,
            )
        };
        let err = (rc != 0).then(|| os("punch host region"));
        self.unlock(r);
        err.map_or(Ok(true), Err)
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

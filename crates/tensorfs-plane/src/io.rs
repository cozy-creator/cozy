//! Source bytes: a TensorFS read lease (the GC hold and the verified door) plus per-object
//! descriptors the plane opens through the store's own verification (`open_verified`). A
//! pinned-tier fill copies what the page cache already holds and reads the rest by O_DIRECT
//! into the destination (no second host copy), else through `tensorfs_core::read::read_into`.
//! Each mode's bytes are counted.

use std::collections::{HashMap, VecDeque};
use std::ffi::CString;
use std::fs::File;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
#[cfg(feature = "store")]
use std::sync::RwLock;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use tensorfs_core::ids::ObjectRef;
#[cfg(feature = "store")]
use tensorfs_core::{meta::Meta, read::ReadLease, store::Store};

use crate::layout::{Item, ItemSource, PART_ALIGN};
use crate::{Error, Result};

#[derive(Debug, Default)]
pub struct IoTally {
    pub cached_bytes: AtomicU64,
    pub direct_bytes: AtomicU64,
    pub buffered_bytes: AtomicU64,
    pub inline_bytes: AtomicU64,
    /// Objects whose filesystem refused O_DIRECT: their bytes went buffered.
    pub direct_refused: AtomicU64,
}

/// How a fill wants its bytes: `Cache` for the pinned tier (O_DIRECT for what the page cache
/// lacks), `Buffered` when no pinned tier exists and the page cache is the host cache,
/// `Cached` for an object its caller found resident (`Source::resident`): copied, no check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Cache,
    Buffered,
    Cached,
}

struct Fds {
    buffered: File,
    direct: Option<File>,
    /// On overlayfs (a container's root): the file reports no page cache of its own.
    overlay: bool,
}

impl Fds {
    /// A buffered descriptor and, with `direct_io`, a second O_DIRECT description of the SAME
    /// inode through /proc: no pathname is resolved again.
    fn of(buffered: File, direct_io: bool, tally: &IoTally) -> Fds {
        let direct = if direct_io {
            let p = CString::new(format!("/proc/self/fd/{}", buffered.as_raw_fd())).unwrap();
            // SAFETY: plain open of a magic link to our own descriptor.
            let fd = unsafe { libc::open(p.as_ptr(), libc::O_RDONLY | libc::O_DIRECT | libc::O_CLOEXEC) };
            if fd < 0 {
                tally.direct_refused.fetch_add(1, Ordering::Relaxed);
                None
            } else {
                // SAFETY: fresh descriptor from the kernel.
                Some(unsafe { File::from_raw_fd(fd) })
            }
        } else {
            None
        };
        Fds { overlay: overlay(&buffered), buffered, direct }
    }
}

/// Where a weight set's bytes come from when its plane fills them: a TensorFS read lease (the
/// GC hold and the verified door) with descriptors opened through the store's verification;
/// or object files a machine verified and handed over (`files`: a plane-only executor reads
/// only those); or nothing, for a sealed layout that holds every byte (`sealed`).
pub struct Source {
    #[cfg(feature = "store")]
    lease: RwLock<Option<ReadLease>>,
    #[cfg(feature = "store")]
    meta: Option<Arc<Meta>>,
    #[cfg(feature = "store")]
    store: Option<Arc<Store>>,
    fds: Mutex<HashMap<String, Arc<Fds>>>,
    /// Handed-over object files: the only objects this source reads, kept open until it goes
    /// (a read opens its working descriptors from them, and closes those after the fill).
    handed: Option<HashMap<String, File>>,
    direct_io: bool,
    plane: usize,
}

/// Whether the page cache holds every page of `[off, off+len)`, from `cachestat(2)`: a pure
/// query (unlike an `RWF_NOWAIT` read, which starts readahead on a miss). Where that cannot
/// answer, the pages are counted through a mapping: a kernel without it (< 6.5), a container
/// whose seccomp filter refuses it, and overlayfs (a container's root), whose files report no
/// pages of their own.
fn cached(f: &Fds, off: u64, len: u64) -> bool {
    #[repr(C)]
    struct Range {
        off: u64,
        len: u64,
    }
    #[repr(C)]
    #[derive(Default)]
    struct Stat {
        nr_cache: u64,
        nr_dirty: u64,
        nr_writeback: u64,
        nr_evicted: u64,
        nr_recently_evicted: u64,
    }
    const SYS_CACHESTAT: libc::c_long = 451;
    let r = Range { off, len };
    let mut st = Stat::default();
    // SAFETY: the documented cachestat ABI; both structs outlive the call.
    let rc = unsafe { libc::syscall(SYS_CACHESTAT, f.buffered.as_raw_fd(), &r, &mut st, 0) };
    let page = 4096u64;
    if rc == 0 && st.nr_cache >= (off + len).div_ceil(page) - off / page {
        return true;
    }
    (rc != 0 || f.overlay) && resident(&f.buffered, off, len)
}

fn overlay(f: &File) -> bool {
    const OVERLAYFS_SUPER_MAGIC: i64 = 0x794c7630;
    // SAFETY: out-param of the right type on our own descriptor.
    let mut fs: libc::statfs = unsafe { std::mem::zeroed() };
    #[allow(clippy::unnecessary_cast)] // f_type is not an i64 on every target
    let magic = unsafe { libc::fstatfs(f.as_raw_fd(), &mut fs) == 0 }.then_some(fs.f_type as i64);
    magic == Some(OVERLAYFS_SUPER_MAGIC)
}

/// Every page of `[off, off+len)` is in core, by `mincore(2)` over a mapping (never faults).
fn resident(f: &File, off: u64, len: u64) -> bool {
    let start = off - off % 4096;
    let n = (off + len - start) as usize;
    if n == 0 {
        return true;
    }
    // SAFETY: a read-only shared mapping of our descriptor, unmapped below.
    let p = unsafe { libc::mmap(std::ptr::null_mut(), n, libc::PROT_READ, libc::MAP_SHARED, f.as_raw_fd(), start as libc::off_t) };
    if p == libc::MAP_FAILED {
        return false;
    }
    let mut pages = vec![0u8; n.div_ceil(4096)];
    // SAFETY: the vector holds one byte per page of the mapping.
    let ok = unsafe { libc::mincore(p, n, pages.as_mut_ptr()) } == 0 && pages.iter().all(|b| b & 1 == 1);
    unsafe { libc::munmap(p, n) };
    ok
}

fn pread_exact(f: &File, buf: &mut [u8], at: u64, what: &str) -> Result<()> {
    f.read_exact_at(buf, at).map_err(|e| Error::Io(format!("{what}: read at {at}: {e}")))
}

impl Source {
    #[cfg(feature = "store")]
    pub fn new(store: Arc<Store>, meta: Arc<Meta>, lease: ReadLease, direct_io: bool, plane: usize) -> Source {
        Source {
            lease: RwLock::new(Some(lease)),
            meta: Some(meta),
            store: Some(store),
            fds: Mutex::new(HashMap::new()),
            handed: None,
            direct_io,
            plane,
        }
    }

    /// The source of a weight set over a sealed layout: the object files its machine verified
    /// and handed over for the layout's holes, read through the same descriptors (O_DIRECT
    /// through a second description of each, where the filesystem allows). No lease, no store;
    /// with no files (a whole layout holds every byte) any read is refused.
    pub fn files(files: Vec<(ObjectRef, File)>, direct_io: bool, plane: usize) -> Source {
        Source {
            #[cfg(feature = "store")]
            lease: RwLock::new(None),
            #[cfg(feature = "store")]
            meta: None,
            #[cfg(feature = "store")]
            store: None,
            fds: Mutex::new(HashMap::new()),
            handed: Some(files.into_iter().map(|(obj, file)| (obj.sha256, file)).collect()),
            direct_io,
            plane,
        }
    }

    /// Close one object's working descriptors; the next read of it reopens them (through the
    /// store, or from the file handed over, which stays).
    pub fn close_object(&self, sha256: &str) {
        self.fds.lock().unwrap().remove(sha256);
    }

    /// Whether the page cache holds every byte of `obj` (one check for all its items).
    pub fn resident(&self, obj: &ObjectRef, tally: &IoTally) -> Result<bool> {
        let fds = self.fds(obj, tally)?;
        Ok(cached(&fds, 0, obj.length))
    }

    pub(crate) fn made_by(&self, plane: usize) -> bool {
        self.plane == plane
    }

    /// End the TensorFS lease. Runs when the last weight set and handle drop the source.
    fn release(&self) -> Result<()> {
        self.fds.lock().unwrap().clear();
        #[cfg(feature = "store")]
        if let Some(l) = self.lease.write().unwrap().take() {
            l.release(self.meta.as_ref().ok_or(Error::Closed)?)?;
        }
        Ok(())
    }

    fn fds(&self, obj: &ObjectRef, tally: &IoTally) -> Result<Arc<Fds>> {
        if let Some(f) = self.fds.lock().unwrap().get(&obj.sha256) {
            return Ok(f.clone());
        }
        #[cfg(feature = "store")]
        if let Some(store) = self.store.as_ref() {
            let vf = store.open_verified(&obj.sha256)?;
            if vf.len() != obj.length {
                return Err(Error::Invalid(format!(
                    "{}: verified descriptor has {} bytes, the plan says {}",
                    obj.sha256,
                    vf.len(),
                    obj.length
                )));
            }
            let f = Arc::new(Fds::of(vf.into_file(), self.direct_io, tally));
            self.fds.lock().unwrap().insert(obj.sha256.clone(), f.clone());
            return Ok(f);
        }
        let handed = self.handed.as_ref().and_then(|h| h.get(&obj.sha256));
        let Some(file) = handed else {
            return Err(Error::Invalid(format!("{}: no file for this object was handed over", obj.sha256)));
        };
        let file = file.try_clone().map_err(|e| Error::Io(format!("{}: {e}", obj.sha256)))?;
        let f = Arc::new(Fds::of(file, self.direct_io, tally));
        self.fds.lock().unwrap().insert(obj.sha256.clone(), f.clone());
        Ok(f)
    }

    /// Fill `item.len` bytes at `dst`. `Mode::Cache` reads a range the page cache fully
    /// holds by copy and everything else by O_DIRECT when alignment allows, which may DMA up
    /// to the next `PART_ALIGN` boundary past the item: the layout reserves that as padding.
    ///
    /// # Safety
    /// `dst` must be valid for writes of `align_up(item.len, PART_ALIGN)` bytes and not be
    /// read or written by anyone else until this returns.
    pub unsafe fn read_item(&self, item: &Item, dst: *mut u8, mode: Mode, tally: &IoTally) -> Result<()> {
        // SAFETY: per the contract.
        let buf = unsafe { std::slice::from_raw_parts_mut(dst, item.len as usize) };
        let o = match &item.source {
            ItemSource::Inline(b) => {
                buf.copy_from_slice(b);
                tally.inline_bytes.fetch_add(item.len, Ordering::Relaxed);
                return Ok(());
            }
            ItemSource::Object(o) => o,
        };
        #[cfg(feature = "store")]
        let guard = self.lease.read().unwrap();
        #[cfg(feature = "store")]
        if let Some(lease) = guard.as_ref() {
            lease.covers(o)?;
        } else if self.handed.is_none() {
            return Err(Error::Closed);
        }
        let fds = self.fds(&o.obj, tally)?;
        let a = PART_ALIGN;
        let end = o.off + item.len;
        let aligned = (dst as u64).wrapping_sub(o.off).is_multiple_of(a) && (end.is_multiple_of(a) || item.part_end);
        let direct = match (mode, &fds.direct) {
            (Mode::Cache, Some(f)) if aligned && !cached(&fds, o.off, item.len) => f,
            _ => {
                #[cfg(feature = "store")]
                if let Some(lease) = guard.as_ref() {
                    tensorfs_core::read::read_into(lease, o, buf)?;
                } else {
                    pread_exact(&fds.buffered, buf, o.off, &o.obj.sha256)?;
                }
                #[cfg(not(feature = "store"))]
                pread_exact(&fds.buffered, buf, o.off, &o.obj.sha256)?;
                let t = if mode == Mode::Buffered { &tally.buffered_bytes } else { &tally.cached_bytes };
                t.fetch_add(item.len, Ordering::Relaxed);
                return Ok(());
            }
        };
        let from = o.off - o.off % a;
        let need = (end - from) as usize;
        // SAFETY: [dst - (o.off - from), dst + align_up(end) - o.off) lies inside this part
        // plus its padding (`aligned` above; parts start on `a`).
        let dbuf = unsafe {
            std::slice::from_raw_parts_mut(dst.sub((o.off - from) as usize), (end.div_ceil(a) * a - from) as usize)
        };
        let mut done = 0usize;
        while done < need {
            // SAFETY: aligned destination, offset and length (the O_DIRECT contract).
            let n = unsafe {
                libc::pread(
                    direct.as_raw_fd(),
                    dbuf[done..].as_mut_ptr() as *mut libc::c_void,
                    dbuf.len() - done,
                    (from + done as u64) as libc::off_t,
                )
            };
            if n < 0 {
                let e = std::io::Error::last_os_error();
                if e.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(Error::Io(format!("O_DIRECT pread {}: {e}", o.obj.sha256)));
            }
            // A short read before `need` resumes at an aligned offset; EOF before it is short.
            let next = done + n as usize;
            let next = if next < need { next - next % a as usize } else { next };
            if next <= done {
                return Err(Error::Io(format!("{}: O_DIRECT read ended at {}", o.obj.sha256, from + done as u64)));
            }
            done = next;
        }
        tally.direct_bytes.fetch_add(item.len, Ordering::Relaxed);
        Ok(())
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        if let Err(e) = self.release() {
            eprintln!("tensorfs-plane: read lease release failed: {e}");
        }
    }
}

type Job = Box<dyn FnOnce() + Send>;

#[derive(Default)]
struct Queues {
    urgent: VecDeque<Job>,
    background: VecDeque<Job>,
    stop: bool,
}

/// Persistent reader threads. Urgent jobs (staging for a device copy someone waits on) run
/// before background jobs (pinned-tier prefetch), so a large prefetch never starves a stream.
pub struct Readers {
    q: Arc<(Mutex<Queues>, Condvar)>,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl Readers {
    /// The pool's threads: how many pieces of one job can run at once.
    pub fn threads(&self) -> usize {
        self.threads.lock().unwrap().len().max(1)
    }

    pub fn new(n: usize) -> Readers {
        let q = Arc::new((Mutex::new(Queues::default()), Condvar::new()));
        let threads = (0..n.max(1))
            .map(|i| {
                let q = q.clone();
                std::thread::Builder::new()
                    .name(format!("tfs-plane-read-{i}"))
                    .spawn(move || loop {
                        let job = {
                            let mut g = q.0.lock().unwrap();
                            loop {
                                if let Some(j) = g.urgent.pop_front().or_else(|| g.background.pop_front()) {
                                    break j;
                                }
                                if g.stop {
                                    return;
                                }
                                g = q.1.wait(g).unwrap();
                            }
                        };
                        job();
                    })
                    .expect("spawn reader")
            })
            .collect();
        Readers {
            q,
            threads: Mutex::new(threads),
        }
    }

    pub fn submit(&self, urgent: bool, job: impl FnOnce() + Send + 'static) {
        let mut g = self.q.0.lock().unwrap();
        if urgent { &mut g.urgent } else { &mut g.background }.push_back(Box::new(job));
        drop(g);
        self.q.1.notify_one();
    }

    /// Let the threads exit once the queue is empty; waits for nothing (the caller may be one
    /// of them).
    pub fn stop(&self) {
        self.q.0.lock().unwrap().stop = true;
        self.q.1.notify_all();
    }

    /// Finish queued jobs, then stop the threads.
    pub fn shutdown(&self) {
        self.stop();
        for t in self.threads.lock().unwrap().drain(..) {
            let _ = t.join();
        }
    }
}

type Done = Box<dyn FnOnce(Result<()>) + Send>;

/// A set of jobs with one completion: the last finisher calls `done` with the first error.
pub struct Batch {
    left: AtomicUsize,
    err: Mutex<Option<Error>>,
    done: Mutex<Option<Done>>,
}

impl Batch {
    pub fn new(jobs: usize, done: impl FnOnce(Result<()>) + Send + 'static) -> Arc<Batch> {
        let b = Arc::new(Batch {
            left: AtomicUsize::new(jobs),
            err: Mutex::new(None),
            done: Mutex::new(Some(Box::new(done))),
        });
        if jobs == 0 {
            b.fire();
        }
        b
    }

    pub fn finish(&self, r: Result<()>) {
        if let Err(e) = r {
            self.err.lock().unwrap().get_or_insert(e);
        }
        if self.left.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.fire();
        }
    }

    fn fire(&self) {
        let r = match self.err.lock().unwrap().take() {
            Some(e) => Err(e),
            None => Ok(()),
        };
        if let Some(f) = self.done.lock().unwrap().take() {
            f(r);
        }
    }
}

/// A raw destination pointer handed to reader threads. The plane's books guarantee exclusive
/// access to the range for the fill's duration.
#[derive(Clone, Copy)]
pub struct Dst(pub *mut u8);
unsafe impl Send for Dst {}

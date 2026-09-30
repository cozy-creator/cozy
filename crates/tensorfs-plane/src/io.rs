//! Source bytes: a TensorFS read lease (the GC hold and the verified door) plus per-object
//! descriptors the plane opens through the store's own verification (`open_verified`). Every
//! read first takes what the page cache already holds (`RWF_NOWAIT`); the rest comes by
//! O_DIRECT into the destination when the item's alignment allows (cache fills: no second host
//! copy), else through `tensorfs_core::read::read_into`. Each mode's bytes are counted.

use std::collections::HashMap;
use std::ffi::CString;
use std::fs::File;
use std::os::fd::{AsRawFd, FromRawFd};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;

use tensorfs_core::ids::ObjectRef;
use tensorfs_core::meta::Meta;
use tensorfs_core::read::{self, ObjectRange, ReadLease};
use tensorfs_core::store::Store;

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
/// lacks), `Buffered` when no pinned tier exists and the page cache is the host cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Cache,
    Buffered,
}

struct Fds {
    buffered: File,
    direct: Option<File>,
}

pub struct Source {
    lease: RwLock<Option<ReadLease>>,
    meta: Arc<Meta>,
    store: Arc<Store>,
    fds: Mutex<HashMap<String, Arc<Fds>>>,
    direct_io: bool,
}

fn pread_nowait(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
    let iov = libc::iovec {
        iov_base: buf.as_mut_ptr() as *mut libc::c_void,
        iov_len: buf.len(),
    };
    // SAFETY: one iovec over a live caller buffer.
    let n = unsafe { libc::preadv2(f.as_raw_fd(), &iov, 1, off as libc::off_t, libc::RWF_NOWAIT) };
    if n < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(n as usize)
    }
}

impl Source {
    pub fn new(store: Arc<Store>, meta: Arc<Meta>, lease: ReadLease, direct_io: bool) -> Source {
        Source {
            lease: RwLock::new(Some(lease)),
            meta,
            store,
            fds: Mutex::new(HashMap::new()),
            direct_io,
        }
    }

    /// End the TensorFS lease. Only after every fill that reads through it has finished.
    pub fn release(&self) -> Result<()> {
        self.fds.lock().unwrap().clear();
        match self.lease.write().unwrap().take() {
            Some(l) => Ok(l.release(&self.meta)?),
            None => Ok(()),
        }
    }

    fn fds(&self, obj: &ObjectRef, tally: &IoTally) -> Result<Arc<Fds>> {
        if let Some(f) = self.fds.lock().unwrap().get(&obj.sha256) {
            return Ok(f.clone());
        }
        let vf = self.store.open_verified(&obj.sha256)?;
        if vf.len() != obj.length {
            return Err(Error::Invalid(format!(
                "{}: verified descriptor has {} bytes, the plan says {}",
                obj.sha256,
                vf.len(),
                obj.length
            )));
        }
        let buffered = vf.into_file();
        // A second open file description of the SAME verified inode, through /proc: no
        // pathname in the store is resolved again.
        let direct = if self.direct_io {
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
        let f = Arc::new(Fds { buffered, direct });
        self.fds.lock().unwrap().insert(obj.sha256.clone(), f.clone());
        Ok(f)
    }

    /// Fill `item.len` bytes at `dst`. With `Mode::Cache` an aligned item may DMA up to the
    /// next `PART_ALIGN` boundary past its end, which the layout reserves as padding.
    ///
    /// # Safety
    /// `dst` must be valid for writes of `align_up(item.len, PART_ALIGN)` bytes and not be
    /// read or written by anyone else until this returns.
    pub unsafe fn read_item(&self, item: &Item, dst: *mut u8, mode: Mode, tally: &IoTally) -> Result<()> {
        let len = item.len as usize;
        // SAFETY: per the contract.
        let buf = unsafe { std::slice::from_raw_parts_mut(dst, len) };
        let o = match &item.source {
            ItemSource::Inline(b) => {
                buf.copy_from_slice(b);
                tally.inline_bytes.fetch_add(item.len, Ordering::Relaxed);
                return Ok(());
            }
            ItemSource::Object(o) => o,
        };
        let guard = self.lease.read().unwrap();
        let lease = guard.as_ref().ok_or(Error::Closed)?;
        lease.covers(o)?;
        let fds = self.fds(&o.obj, tally)?;

        let mut got = 0usize;
        while got < len {
            match pread_nowait(&fds.buffered, &mut buf[got..], o.off + got as u64) {
                Ok(0) => break,
                Ok(n) => got += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(_) => break, // EAGAIN (not cached) or no NOWAIT support: read the rest.
            }
        }
        tally.cached_bytes.fetch_add(got as u64, Ordering::Relaxed);
        if got == len {
            return Ok(());
        }

        let a = PART_ALIGN;
        let start = o.off + got as u64;
        let skip = start % a;
        let end = o.off + item.len;
        let congruent = (dst as u64).wrapping_sub(o.off) % a == 0;
        let tail_ok = end % a == 0 || item.part_end;
        if let (Mode::Cache, Some(direct), true, true) = (mode, &fds.direct, congruent, tail_ok) {
            let from = start - skip;
            let to = end.div_ceil(a) * a;
            // SAFETY: [dst+got-skip, dst+to-o.off) lies inside the part plus its padding:
            // `skip` bytes before `start` are this item's (already filled, same content).
            let dbuf = unsafe {
                std::slice::from_raw_parts_mut(dst.add(got - skip as usize), (to - from) as usize)
            };
            let mut done = 0usize;
            let need = (end - from) as usize;
            while done < need {
                // SAFETY: aligned destination, offset and length (O_DIRECT contract).
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
                if n == 0 {
                    return Err(Error::Io(format!(
                        "{}: ended at {} of {end} bytes",
                        o.obj.sha256,
                        from + done as u64
                    )));
                }
                // A short read before `need` continues from an aligned offset (O_DIRECT).
                let next = done + n as usize;
                let next = if next < need { next - next % a as usize } else { next };
                if next <= done {
                    return Err(Error::Io(format!("{}: O_DIRECT made no progress", o.obj.sha256)));
                }
                done = next;
            }
            tally.direct_bytes.fetch_add((len - got) as u64, Ordering::Relaxed);
            return Ok(());
        }
        read::read_into(
            lease,
            &ObjectRange {
                obj: o.obj.clone(),
                off: start,
                len: (len - got) as u64,
            },
            &mut buf[got..],
        )?;
        tally.buffered_bytes.fetch_add((len - got) as u64, Ordering::Relaxed);
        Ok(())
    }
}

type Job = Box<dyn FnOnce() + Send>;

/// Persistent reader threads fed by one queue.
pub struct Readers {
    tx: Mutex<Option<Sender<Job>>>,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl Readers {
    pub fn new(n: usize) -> Readers {
        let (tx, rx) = channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        let threads = (0..n.max(1))
            .map(|i| {
                let rx = rx.clone();
                std::thread::Builder::new()
                    .name(format!("tfs-plane-read-{i}"))
                    .spawn(move || loop {
                        let job = match rx.lock().unwrap().recv() {
                            Ok(j) => j,
                            Err(_) => return,
                        };
                        job();
                    })
                    .expect("spawn reader")
            })
            .collect();
        Readers {
            tx: Mutex::new(Some(tx)),
            threads: Mutex::new(threads),
        }
    }

    pub fn submit(&self, job: impl FnOnce() + Send + 'static) {
        if let Some(tx) = self.tx.lock().unwrap().as_ref() {
            let _ = tx.send(Box::new(job));
        }
    }

    pub fn shutdown(&self) {
        self.tx.lock().unwrap().take();
        for t in self.threads.lock().unwrap().drain(..) {
            let _ = t.join();
        }
    }
}

/// A set of jobs with one completion: the last finisher calls `done` with the first error.
pub struct Batch {
    left: AtomicUsize,
    err: Mutex<Option<Error>>,
    done: Mutex<Option<Box<dyn FnOnce(Result<()>) + Send>>>,
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

//! The chunk reader's disk half: a weight set's chunk files, and the reader threads.
//!
//! A chunk opens on its first read, with a second O_DIRECT description of the same inode where
//! the filesystem allows. A chunk the page cache holds whole is copied from it; any other is read
//! with O_DIRECT (`direct`), or buffered so it stays in the page cache. Each mode's bytes are
//! counted.

use std::collections::VecDeque;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;

use cozy_platform::durable;

use crate::layout::PART_ALIGN;
use crate::{Error, Result};

#[derive(Debug, Default)]
pub struct IoTally {
    /// Read with O_DIRECT.
    pub direct_bytes: AtomicU64,
    /// Copied from the page cache (the chunk was resident whole).
    pub cached_bytes: AtomicU64,
    /// Read buffered from a chunk the page cache did not hold.
    pub buffered_bytes: AtomicU64,
    /// Chunks whose filesystem refused O_DIRECT: their bytes went buffered.
    pub direct_refused: AtomicU64,
}

struct Fds {
    buffered: File,
    /// Present when this chunk reads O_DIRECT: it is not resident and the filesystem allows it.
    direct: Option<File>,
    resident: bool,
}

/// A weight set's chunk files, by chunk index.
pub struct Files {
    paths: Vec<PathBuf>,
    open: Vec<OnceLock<std::result::Result<Fds, Error>>>,
    direct: bool,
    pub tally: Arc<IoTally>,
}

impl Files {
    /// `direct`: chunks the page cache does not hold read O_DIRECT; otherwise buffered.
    pub fn new(paths: Vec<PathBuf>, direct: bool, tally: Arc<IoTally>) -> Files {
        let open = paths.iter().map(|_| OnceLock::new()).collect();
        Files { paths, open, direct, tally }
    }

    fn fds(&self, c: u32) -> Result<&Fds> {
        let path = self.paths.get(c as usize).ok_or_else(|| Error::Invalid(format!("no chunk {c}")))?;
        self.open[c as usize]
            .get_or_init(|| {
                let buffered = File::open(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
                let len = buffered.metadata().map_err(|e| Error::Io(format!("{}: {e}", path.display())))?.len();
                let resident = durable::cached(&buffered, 0, len).unwrap_or(false);
                let direct = (self.direct && !resident).then(|| durable::open_direct(&buffered).ok().flatten()).flatten();
                if self.direct && !resident && direct.is_none() {
                    self.tally.direct_refused.fetch_add(1, Ordering::Relaxed);
                }
                Ok(Fds { buffered, direct, resident })
            })
            .as_ref()
            .map_err(Error::clone)
    }

    /// Read `len` bytes of chunk `c` from `off` into `dst`.
    ///
    /// # Safety
    /// `dst` must be valid for writes of `align_up(len, PART_ALIGN)` bytes, `PART_ALIGN` aligned,
    /// and touched by no one else until this returns. `off` is `PART_ALIGN` aligned.
    pub unsafe fn read(&self, c: u32, off: u64, len: u64, dst: *mut u8) -> Result<()> {
        let fds = self.fds(c)?;
        let what = || self.paths[c as usize].display().to_string();
        let Some(direct) = &fds.direct else {
            // SAFETY: per the contract.
            let buf = unsafe { std::slice::from_raw_parts_mut(dst, len as usize) };
            fds.buffered.read_exact_at(buf, off).map_err(|e| Error::Io(format!("{}: read at {off}: {e}", what())))?;
            let t = if fds.resident { &self.tally.cached_bytes } else { &self.tally.buffered_bytes };
            t.fetch_add(len, Ordering::Relaxed);
            return Ok(());
        };
        // The last request of a chunk may end mid-page: read up to the next page, short at EOF.
        let room = len.div_ceil(PART_ALIGN) * PART_ALIGN;
        let mut done = 0u64;
        while done < len {
            // SAFETY: aligned destination, offset and length (the O_DIRECT contract).
            let n = unsafe {
                libc::pread(
                    direct.as_raw_fd(),
                    dst.add(done as usize) as *mut libc::c_void,
                    (room - done) as usize,
                    (off + done) as libc::off_t,
                )
            };
            if n < 0 {
                let e = std::io::Error::last_os_error();
                if e.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(Error::Io(format!("{}: O_DIRECT read at {}: {e}", what(), off + done)));
            }
            // A short read resumes at an aligned offset; EOF before `len` is short.
            let next = done + n as u64;
            let next = if next < len { next - next % PART_ALIGN } else { next };
            if next <= done {
                return Err(Error::Io(format!("{}: ends at {}, short of {}", what(), off + done, off + len)));
            }
            done = next;
        }
        self.tally.direct_bytes.fetch_add(len, Ordering::Relaxed);
        Ok(())
    }
}

type Job = Box<dyn FnOnce() + Send>;
/// The jobs and whether to stop once they are done.
type Queue = Arc<(Mutex<(VecDeque<Job>, bool)>, Condvar)>;

/// Persistent reader threads, first come first served.
pub struct Readers {
    q: Queue,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl Readers {
    pub fn new(n: usize) -> Readers {
        let q: Queue = Arc::new((Mutex::new((VecDeque::new(), false)), Condvar::new()));
        let threads = (0..n.max(1))
            .map(|i| {
                let q = q.clone();
                std::thread::Builder::new()
                    .name(format!("tfs-gpu-read-{i}"))
                    .spawn(move || loop {
                        let job = {
                            let mut g = q.0.lock().unwrap();
                            loop {
                                if let Some(j) = g.0.pop_front() {
                                    break j;
                                }
                                if g.1 {
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
        Readers { q, threads: Mutex::new(threads) }
    }

    pub fn submit(&self, job: impl FnOnce() + Send + 'static) {
        self.q.0.lock().unwrap().0.push_back(Box::new(job));
        self.q.1.notify_one();
    }

    /// Let the threads exit once the queue is empty; waits for nothing (the caller may be one
    /// of them).
    pub fn stop(&self) {
        self.q.0.lock().unwrap().1 = true;
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

/// A raw pointer handed to reader threads; the plane's books keep its range exclusive.
#[derive(Clone, Copy)]
pub struct Ptr(pub *mut u8);
unsafe impl Send for Ptr {}

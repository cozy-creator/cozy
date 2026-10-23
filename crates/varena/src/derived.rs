//! Derived weights on disk (varena-leases §6, review M1/M4; S5's store): a page a GPU transform
//! produced (a LoRA bake, a quantization from another source) is written to the entry TensorD
//! hands this executor, write-behind at idle, and a later bind of that identity refills from it
//! through the page cache instead of deriving it again.
//! - M4: only where deriving again takes longer than writing the bytes and reading them back.
//! - M1: a revoke never waits on a write: the page is pinned only for its device-to-host copy,
//!   and a write still queued when its page goes is dropped.
//!
//! The holder (Memory) owns the hooks: it pins pages, says when no request runs, and keeps each
//! written entry's fd (the holder set is the open fds) for refills.

use std::collections::VecDeque;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::{Condvar, Mutex};
use sha2::{Digest, Sha256};
use tensorfs::format::ObjectRef;
use tensorfs::io::source::{IoTally, Mode, Placement, Source};
use tensorfs::io::{Dst, ItemSource, ObjectRange};

use crate::backend::{Backend, Event, Va};
use crate::layout::Item;
use crate::{Error, Result};

/// One page's derived bytes: its set's fingerprint, its region, and its fill identity's digest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Key {
    pub fingerprint: u64,
    pub region: u32,
    pub fill: [u8; 16],
}

impl Key {
    /// `identity` is the Fill::Declared transform identity (LoRA digests, scales, recipe, lane).
    pub fn new(fingerprint: u64, region: u32, identity: &str) -> Key {
        let mut fill = [0u8; 16];
        fill.copy_from_slice(Sha256::digest(identity.as_bytes()).get(..16).unwrap_or(&[0; 16]));
        Key { fingerprint, region, fill }
    }

    /// The store's name for it (S5): opaque to TensorD.
    pub fn store(&self) -> String {
        let hex: String = self.fill.iter().map(|b| format!("{b:02x}")).collect();
        format!("{:016x}/{}/{hex}", self.fingerprint, self.region)
    }
}

/// M4: deriving again (`derive_ns`) takes longer than writing `bytes` and reading them back.
pub fn pays(derive_ns: u64, bytes: u64, write_bps: f64, read_bps: f64) -> bool {
    if write_bps <= 0.0 || read_bps <= 0.0 {
        return false;
    }
    let cost_s = bytes as f64 / write_bps + bytes as f64 / read_bps;
    derive_ns as f64 / 1e9 > cost_s
}

/// What the writer asks of its holder.
pub trait Hold: Send + Sync {
    /// Pins the page for its device-to-host copy; dropping the guard unpins. `None` once it is
    /// gone or holds other bytes: its write is dropped (M1).
    fn pin(&self, key: &Key) -> Option<Box<dyn Send>>;
    /// No request runs: writes go only now.
    fn idle(&self) -> bool;
    /// Written whole: the holder keeps `entry` for refills and tells TensorD.
    fn written(&self, key: Key, entry: File);
}

/// One page to write: TensorD's entry for it, where its bytes are mapped, and the event its
/// transform completed by.
pub struct Job {
    pub key: Key,
    pub entry: File,
    pub va: Va,
    pub ready: Event,
}

#[derive(Default)]
struct Queue {
    jobs: VecDeque<Job>,
    writing: bool,
    /// Bumped by every offer and wake: a waiter that read it before its check misses none.
    gen: u64,
    stop: bool,
}

struct Shared {
    backend: Arc<dyn Backend>,
    hold: Arc<dyn Hold>,
    queue: Mutex<Queue>,
    woke: Condvar,
    /// Measured device-to-disk bytes per second; 0 until a write lands.
    bps: AtomicU64,
}

/// The write-behind thread.
pub struct Writer {
    sh: Arc<Shared>,
}

impl Writer {
    pub fn new(backend: Arc<dyn Backend>, hold: Arc<dyn Hold>) -> Result<Writer> {
        let sh = Arc::new(Shared { backend, hold, queue: Mutex::new(Queue::default()), woke: Condvar::new(), bps: AtomicU64::new(0) });
        let run = sh.clone();
        std::thread::Builder::new().name("varena-derived".into()).spawn(move || run.run()).map_err(|e| Error::io("spawn the derived writer", e))?;
        Ok(Writer { sh })
    }

    /// Queues a page; it is written at the next idle moment.
    pub fn offer(&self, job: Job) {
        let mut q = self.sh.queue.lock();
        q.jobs.push_back(job);
        q.gen += 1;
        self.sh.woke.notify_all();
    }

    /// The page goes (a revoke, an identity change): a write still queued for it is dropped. One
    /// past its copy finishes, from host memory.
    pub fn abandon(&self, key: &Key) {
        self.sh.queue.lock().jobs.retain(|j| j.key != *key);
    }

    /// The holder went idle.
    pub fn wake(&self) {
        let mut q = self.sh.queue.lock();
        q.gen += 1;
        self.sh.woke.notify_all();
    }

    /// Writes queued or in progress.
    pub fn pending(&self) -> usize {
        let q = self.sh.queue.lock();
        q.jobs.len() + usize::from(q.writing)
    }

    /// Measured device-to-disk bytes per second (M4's write side), once a write landed.
    pub fn write_bps(&self) -> Option<f64> {
        let bps = self.sh.bps.load(Relaxed);
        (bps > 0).then_some(bps as f64)
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        let mut q = self.sh.queue.lock();
        q.stop = true;
        self.sh.woke.notify_all();
    }
}

impl Shared {
    fn run(&self) {
        loop {
            let job = {
                let mut q = self.queue.lock();
                loop {
                    if q.stop {
                        return;
                    }
                    let gen = q.gen;
                    if !q.jobs.is_empty() {
                        // The holder's idle test runs with no lock of ours held.
                        drop(q);
                        let idle = self.hold.idle();
                        q = self.queue.lock();
                        if idle {
                            if let Some(job) = q.jobs.pop_front() {
                                q.writing = true;
                                break job;
                            }
                        }
                    }
                    while q.gen == gen && !q.stop {
                        self.woke.wait(&mut q);
                    }
                }
            };
            if let Err(e) = self.write(job) {
                tracing::warn!("varena: a derived page's write failed: {e}");
            }
            self.queue.lock().writing = false;
        }
    }

    fn write(&self, job: Job) -> Result<()> {
        let Some(pin) = self.hold.pin(&job.key) else { return Ok(()) };
        let start = Instant::now();
        let mut bytes = vec![0u8; usize::try_from(job.va.len).map_err(|_| Error::Invalid("a page beyond the address space".into()))?];
        self.backend.copy_out(job.va, 0, &mut bytes, job.ready)?;
        // The bytes are ours: a revoke may take the page now, while the disk write runs.
        drop(pin);
        job.entry.write_all_at(&bytes, 0).map_err(|e| Error::io("write a derived entry", e))?;
        let secs = start.elapsed().as_secs_f64().max(1e-9);
        let now = (job.va.len as f64 / secs) as u64;
        let was = self.bps.load(Relaxed);
        self.bps.store(if was == 0 { now } else { (was * 3 + now) / 4 }, Relaxed);
        self.hold.written(job.key, job.entry);
        Ok(())
    }
}

/// A refill from a written entry: one item over the whole page, read through the page cache
/// exactly as a store fill reads.
pub struct Refill {
    source: Source,
    items: Vec<Item>,
}

impl Refill {
    pub fn new(key: &Key, entry: File, len: u64) -> Refill {
        let obj = ObjectRef { sha256: key.store(), length: len };
        let items = vec![Item { offset: 0, len, source: ItemSource::Object(ObjectRange { obj: obj.clone(), off: 0, len }), part_end: true }];
        Refill { source: Source::files(vec![(obj, entry)], false, 0), items }
    }

    /// Reads the entry into `dst`.
    ///
    /// # Safety
    /// `dst` is mapped, as long as the entry, and only this refill writes it until it returns.
    pub unsafe fn read(&self, dst: Dst, tally: &IoTally) -> Result<()> {
        let placement = Placement { items: &self.items, origin: 0, dst };
        // SAFETY: the caller's contract on `dst`.
        unsafe { self.source.read_placements(&[placement], tensorfs::io::ReadOpts::default().threads, Mode::Buffered, tally, None) }?;
        Ok(())
    }
}

//! The GPU's shared table, named per GPU and user so a standalone engine opens it without
//! TensorD. One file of shared memory: a header, then a heap of records on 256-byte cells that
//! grows with the file. Each record's word carries its size, a generation, its kind and state,
//! and a record is only ever freed by comparing that word under the heap lock, so a stale
//! reader can never free a newer record at the same offset. A row (one per process) is held by
//! an OFD lock that dies with it; its bytes by class are a two-copy latch, published from its
//! own table's derivation. Asks go to a row's own ask thread over a socket named by the row's
//! offset and generation: the answer or the victim's death ends the wait, and a reused offset
//! reads as gone. Facts (peaks, rooms, dispatches) are immutable records, a newer version
//! superseding an older one; a write merges with the newest under the heap lock.
//!
//! Locks: the heap lock (this process's threads by a mutex, other processes by an OFD lock) and
//! a row's ask and latch locks are leaves: nothing else is locked while one is held (the heap
//! maps the table afresh rather than through the shared handle). Every process with the table
//! open holds a read lock on [`OPEN`]; the last row to leave unlinks the name only when it can
//! take that lock for writing, so nobody is left holding a table nobody else can find.
//! The admission gate orders metadata snapshots/commits across processes; its callers sample
//! drivers before taking it and never keep it through driver work or peer/service waits.

use std::collections::HashMap;
use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release, SeqCst};
use std::sync::atomic::{fence, AtomicBool, AtomicU32, AtomicU64};
use std::sync::{Arc, Weak};

use parking_lot::{Mutex, MutexGuard, RwLock};
use platform::contain::host_memory;
use platform::handles::Owned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::custody;
use crate::decide::{self, Admitted};
use crate::os::{self, Listener, Map};
use crate::table::{Class, Remote};
use crate::{Error, GpuId, Result};

const MAGIC: u64 = u64::from_le_bytes(*b"vledgr15");
/// The header's page: the heap starts after it, and grows from there.
const HEADER: u64 = 4096;
/// The heap's unit: rows take two, a fact as many as its bytes need.
const CELL: u64 = 256;
const ROW_CELLS: u64 = 2;
const FACT_HEAD: u64 = 24;
/// The header bytes every opener read-locks while it has the table open (the heap lock is the
/// eight before them).
const OPEN: u64 = 8;
/// Serializes authoritative admission snapshots and their metadata commits.
const ADMISSION: u64 = 16;

const FREE: u64 = 0;
const CLAIMING: u64 = 1;
const READY: u64 = 2;
const ROW: u64 = 1;
const FACT: u64 = 2;

/// The latch: context, torch, caches, classes (7), remaining, unmeasured, running, outside,
/// yielded (family, W) × YIELDED, host_peak, host_unmeasured, returning, physical,
/// physical_epoch.
const WORDS: usize = 19 + 2 * decide::YIELDED;

#[repr(C)]
struct Header {
    magic: u64,
    capacity: AtomicU64,
    /// The heap's end, a byte offset; the file is at least this long.
    end: AtomicU64,
    /// Moves on every allocation or free; its low bits are a new record's generation.
    shape: AtomicU64,
    /// Device bytes no row accounts for: TensorD's sample while it has the table (`sampled`),
    /// else as the last executor decision measured them.
    foreign: AtomicU64,
    /// 0: shared (the machine's owner uses it too), 1: dedicated.
    mode: AtomicU32,
    /// 1 once TensorD's ledger sampled foreign bytes, until it goes: executors read them, never
    /// write them.
    sampled: AtomicU32,
}

#[repr(C)]
struct RowRec {
    word: AtomicU64,
    pid: AtomicU32,
    /// Its ask thread's id: TensorD's `stalled` reads that thread's state.
    tid: AtomicU32,
    /// The row this one is asking now (its id; 0: none) and the bytes: the asker's own record,
    /// never the victim's.
    asking: AtomicU64,
    asked: AtomicU64,
    /// Kept pages no live process maps (the custody service's row).
    orphans: AtomicU64,
    /// Its process waits for room elsewhere: every change of the table signals its endpoint.
    waiting: AtomicU64,
    /// Full heap generation, including the bits not carried by the compact row id.
    born: AtomicU64,
    /// Changes before every physical operation, including operations whose net change is zero.
    physical_epoch: AtomicU64,
    /// Concurrent driver operations/publications still outstanding. A failed or abandoned
    /// operation stays active: uncertain physical ownership is never subtraction evidence.
    physical_active: AtomicU64,
    /// The orphan count's publication tag (ordinary physical counts use the facts latch).
    orphan_epoch: AtomicU64,
    seq: AtomicU64,
    latch: [[AtomicU64; WORDS]; 2],
}

#[repr(C)]
struct FactRec {
    word: AtomicU64,
    version: u64,
    klen: u32,
    vlen: u32,
}

const _: () = assert!(size_of::<RowRec>() as u64 <= ROW_CELLS * CELL && size_of::<Header>() as u64 <= HEADER);
const _: () = assert!(size_of::<FactRec>() as u64 == FACT_HEAD);

fn word(cells: u64, gen: u64, kind: u64, state: u64) -> u64 {
    cells << 32 | (gen & 0xffff) << 16 | kind << 8 | state
}

fn cells(w: u64) -> u64 {
    w >> 32
}

fn gen(w: u64) -> u64 {
    (w >> 16) & 0xffff
}

fn kind(w: u64) -> u64 {
    (w >> 8) & 0xff
}

fn state(w: u64) -> u64 {
    w & 0xff
}

/// A row's id in [`Remote::row`]: its offset and generation.
fn row_id(off: u64, w: u64) -> u64 {
    (off / CELL) << 16 | gen(w)
}

/// The machine's sharing mode, as TensorD sets it ([`Ledger::set_mode`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Someone else uses the machine too: speculation never grows a process's footprint.
    Shared,
    Dedicated,
}

/// One row's published facts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Facts {
    pub context: u64,
    pub torch: u64,
    pub caches: u64,
    pub classes: [u64; 7],
    pub remaining: u64,
    pub unmeasured: u64,
    pub running: u64,
    pub outside: u64,
    /// The remaining W of its requests waiting on child calls, per family they wait for (a
    /// hash, W): each family's child is admitted beside its own parents as if they were gone,
    /// never beside another family's (review 3, R4). Family 0: any, counted for every child.
    pub yielded: [(u64, u64); decide::YIELDED],
    /// Its running requests' largest measured anonymous host peak, and how many have none
    /// measured: TensorD's host rule keeps their growth free.
    pub host_peak: u64,
    pub host_unmeasured: u64,
    /// Subset of P already retiring: charged until completion, then returned without an ask.
    pub returning: u64,
    /// Actual unique resident bytes. Future admission promises are never physical usage;
    /// orphans remain custody's separate charge, not duplicated in their keeper's facts.
    pub physical: u64,
    /// Epoch sampled BEFORE collecting `physical`; zero means unobserved. A publication
    /// must retain the observation's tag rather than tagging old counts with the current epoch.
    pub physical_epoch: u64,
}

impl Facts {
    fn words(&self) -> [u64; WORDS] {
        let c = self.classes;
        let y = self.yielded;
        [self.context, self.torch, self.caches, c[0], c[1], c[2], c[3], c[4], c[5], c[6], self.remaining, self.unmeasured, self.running, self.outside, y[0].0, y[0].1, y[1].0, y[1].1, y[2].0, y[2].1, self.host_peak, self.host_unmeasured, self.returning, self.physical, self.physical_epoch]
    }

    fn of(w: [u64; WORDS]) -> Facts {
        let [context, torch, caches, c0, c1, c2, c3, c4, c5, c6, remaining, unmeasured, running, outside, f0, y0, f1, y1, f2, y2, host_peak, host_unmeasured, returning, physical, physical_epoch] = w;
        let yielded = [(f0, y0), (f1, y1), (f2, y2)];
        Facts { context, torch, caches, classes: [c0, c1, c2, c3, c4, c5, c6], remaining, unmeasured, running, outside, yielded, host_peak, host_unmeasured, returning, physical, physical_epoch }
    }

    fn held(&self) -> u64 {
        self.context + self.torch + self.classes.iter().sum::<u64>()
    }
}

/// A row as read.
#[derive(Clone, Debug)]
pub(crate) struct RowView {
    pub off: u64,
    pub word: u64,
    pub pid: u32,
    pub live: bool,
    pub facts: Facts,
    pub orphans: u64,
    /// The row it is asking now (0: none) and the bytes.
    pub asking: u64,
    pub asked: u64,
    pub tid: u32,
    pub waiting: bool,
    physical: PhysicalStamp,
}

impl RowView {
    pub fn id(&self) -> u64 {
        row_id(self.off, self.word)
    }
}

/// What a row's ask thread is asked.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum Ask {
    /// Device bytes under a ceiling (Class::N, D or W, as its index).
    Room { ceiling: usize, bytes: u64 },
}

#[derive(Debug, Serialize, Deserialize)]
struct Answer {
    freed: u64,
}

/// The table mapped in this process.
pub(crate) struct Shared {
    /// This process's description: mapped and used for the heap and open locks, never for a
    /// row lock.
    fd: Owned,
    map: RwLock<Arc<Map>>,
    /// The heap lock's half for this process's threads: an OFD lock is per description, so it
    /// excludes only other processes.
    heap: Mutex<()>,
    admission: Mutex<()>,
    pub name: String,
    /// The table owner's uid: its services (custody, the wake, the asks) are named
    /// in that user's namespace.
    pub ns: u32,
    /// Signals the GPU's wake endpoint (TensorD's, by name).
    wake: os::Notifier,
}

/// The heap lock, held.
struct Heap<'a> {
    fd: &'a Owned,
    _threads: MutexGuard<'a, ()>,
}

impl Drop for Heap<'_> {
    fn drop(&mut self) {
        let _ = os::unlock(self.fd, 0, 8);
    }
}

/// The shared admission gate. Its local half excludes threads sharing one file description;
/// the OFD half excludes independently opened descriptions, including those in this process.
pub(crate) struct Admission<'a> {
    shared: &'a Shared,
    _threads: MutexGuard<'a, ()>,
}

impl Admission<'_> {
    pub fn rows(&self) -> Result<Vec<RowView>> {
        self.shared.rows()
    }
}

impl Drop for Admission<'_> {
    fn drop(&mut self) {
        let _ = os::unlock(&self.shared.fd, ADMISSION, 8);
    }
}

#[derive(Clone, Copy, Debug)]
struct PhysicalStamp {
    born: u64,
    epoch: u64,
    facts: u64,
    orphans: u64,
    clean: bool,
}

/// Physical attribution across a driver observation. Admission promises are read afresh
/// under the gate; only physical subtraction is filtered by this earlier observation.
pub(crate) struct PhysicalSample(HashMap<u64, PhysicalStamp>);

impl PhysicalSample {
    /// Whether every other live row's physical counts held across the sample (row `own`, the
    /// caller's, it counts exactly itself): only then is the foreign figure derived from it
    /// evidence others may use.
    pub fn complete(&self, rows: &[RowView], own: u64) -> bool {
        rows.iter().filter(|r| r.live && r.id() != own).all(|row| {
            let Some(before) = self.0.get(&row.id()) else { return false };
            let after = row.physical;
            before.clean && after.clean && before.born == after.born && before.epoch == after.epoch && before.facts == before.epoch && after.facts == after.epoch && before.orphans == before.epoch && after.orphans == after.epoch
        })
    }

    /// The other live rows' physical bytes proven across the sample (row `own` excepted).
    pub fn known(&self, rows: &[RowView], own: u64) -> u64 {
        rows.iter().filter(|r| r.live && r.id() != own).fold(0u64, |known, row| {
            let Some(before) = self.0.get(&row.id()) else { return known };
            let after = row.physical;
            if !before.clean || !after.clean || before.born != after.born || before.epoch != after.epoch {
                return known;
            }
            let physical = if before.facts == before.epoch && after.facts == after.epoch { row.facts.physical } else { 0 };
            let orphans = if before.orphans == before.epoch && after.orphans == after.epoch { row.orphans } else { 0 };
            known.saturating_add(physical).saturating_add(orphans)
        })
    }
}

/// Takes the table's name for a new user: read-locked open, so the last row leaving does not
/// unlink it under them; a name unlinked meanwhile is taken again.
fn open_named(name: &str) -> Result<Owned> {
    loop {
        let table = os::shared(name, true)?;
        os::share(&table, OPEN, 8, true)?;
        if os::linked(&table)? {
            return Ok(table);
        }
    }
}

impl Drop for Shared {
    /// Its descriptor's locks go with it; a duplicate of a grant's description leaves the
    /// grant's open mark held (an OFD lock is the description's).
    fn drop(&mut self) {
        os::closing(&self.fd);
    }
}

/// The processes that have the file behind `fd` open, this one aside (Linux: their
/// descriptors; elsewhere none are named).
fn openers(fd: &Owned) -> Vec<u32> {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        let Ok(path) = std::fs::read_link(format!("/proc/self/fd/{}", fd.as_raw_fd())) else { return vec![] };
        let me = std::process::id();
        let Ok(procs) = std::fs::read_dir("/proc") else { return vec![] };
        procs
            .flatten()
            .filter_map(|p| p.file_name().to_str().and_then(|n| n.parse::<u32>().ok()))
            .filter(|pid| *pid != me)
            .filter(|pid| std::fs::read_dir(format!("/proc/{pid}/fd")).is_ok_and(|fds| fds.flatten().any(|f| std::fs::read_link(f.path()).is_ok_and(|l| l == path))))
            .collect()
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = fd;
        vec![]
    }
}

impl Shared {
    /// The table behind `fd` (a grant's), initialised at `capacity` (the card's) if it is new.
    /// Only TensorD changes it after ([`Ledger::create`]): an opener's limit is its own row's.
    pub fn adopt(fd: Owned, gpu: &GpuId, capacity: u64) -> Result<Shared> {
        // The description handed over is this process's own (a granted executor of another uid
        // could not open the file again).
        let ns = os::own(&fd)?;
        os::accept_uid(ns);
        os::share(&fd, OPEN, 8, true)?;
        if !os::linked(&fd)? {
            return Err(Error::Closed(format!("the table of {} was unlinked", gpu.uuid)));
        }
        // Initialisation is one writer's: the heap lock serialises it.
        os::lock(&fd, 0, 8, true)?;
        let init = (|| -> Result<Map> {
            if os::size(&fd)? < HEADER {
                os::resize(&fd, HEADER)?;
            }
            let mut map = os::map(&fd, os::size(&fd)? as usize, true)?;
            // SAFETY: the header page is mapped and aligned; its fields are atomics or written
            // once under the heap lock before `magic`.
            let magic = unsafe { (*(map.base as *const Header)).magic };
            if magic != 0 && magic != MAGIC {
                // Another build's table (S3's gpu12 and gpu14: a wheel's A/B on one pod). With
                // its owners still open it waits for them; with none it is this build's now.
                let format = String::from_utf8_lossy(&magic.to_le_bytes()).into_owned();
                if os::held(&fd, OPEN, 8)? {
                    return Err(Error::Closed(format!(
                        "the varena table of {} is another build's (format {format}; this build's is {}), and its owners have it open (pids {:?}): it is reopened once they finish",
                        gpu.uuid,
                        String::from_utf8_lossy(&MAGIC.to_le_bytes()),
                        openers(&fd)
                    )));
                }
                tracing::warn!("varena: the table of {} was another build's (format {format}) with no owner open: reclaimed", gpu.uuid);
                drop(map);
                os::resize(&fd, 0)?;
                os::resize(&fd, HEADER)?;
                map = os::map(&fd, os::size(&fd)? as usize, true)?;
            }
            // SAFETY: as above, the header page of the (possibly reclaimed) mapping.
            let h = unsafe { &mut *(map.base as *mut Header) };
            if h.magic == 0 {
                h.capacity.store(capacity, Relaxed);
                h.end.store(HEADER, Relaxed);
                fence(Release);
                h.magic = MAGIC;
            }
            Ok(map)
        })();
        os::unlock(&fd, 0, 8)?;
        let map = init?;
        Ok(Shared { fd, map: RwLock::new(Arc::new(map)), heap: Mutex::new(()), admission: Mutex::new(()), name: gpu.uuid.clone(), ns, wake: os::wake_sender()? })
    }

    pub fn descriptor(&self) -> Result<Owned> {
        os::reopen(&self.fd, true)
    }

    fn heap(&self) -> Result<Heap<'_>> {
        let threads = self.heap.lock();
        os::lock(&self.fd, 0, 8, true)?;
        Ok(Heap { fd: &self.fd, _threads: threads })
    }

    /// A metadata-only transaction: resnapshot rows and publish admission before releasing.
    /// Sample driver state BEFORE taking this guard. Never hold it across driver calls, IPC,
    /// asks, or waits; native allocation callbacks may only consume already-owned quota.
    pub fn admission(&self) -> Result<Admission<'_>> {
        let threads = self.admission.lock();
        os::lock(&self.fd, ADMISSION, 8, true)?;
        Ok(Admission { shared: self, _threads: threads })
    }

    /// Physical identities immediately BEFORE a driver sample. Only unchanged, fully
    /// published identities in the authoritative post-sample snapshot may be subtracted.
    pub fn physical_sample(&self) -> Result<PhysicalSample> {
        Ok(PhysicalSample(self.rows()?.into_iter().filter(|r| r.live).map(|r| (r.id(), r.physical)).collect()))
    }

    /// `m`, or under the heap lock a mapping of its own when `m` ends before `len`.
    fn covering(&self, m: Arc<Map>, len: u64) -> Result<Arc<Map>> {
        if m.len as u64 >= len {
            return Ok(m);
        }
        Ok(Arc::new(os::map(&self.fd, os::size(&self.fd)? as usize, true)?))
    }

    /// The current mapping, remapped when the heap outgrew it.
    fn map(&self) -> Result<Arc<Map>> {
        let m = self.map.read().clone();
        if header(&m).end.load(Acquire) <= m.len as u64 {
            return Ok(m);
        }
        let mut slot = self.map.write();
        if header(&slot).end.load(Acquire) > slot.len as u64 {
            *slot = Arc::new(os::map(&self.fd, os::size(&self.fd)? as usize, true)?);
        }
        Ok(slot.clone())
    }

    /// TensorD's: the capacity is exactly this (a lifted limit raises it).
    fn set_capacity(&self, capacity: u64) -> Result<()> {
        let m = self.map()?;
        header(&m).capacity.store(capacity, Release);
        Ok(())
    }

    pub fn capacity(&self) -> Result<u64> {
        let m = self.map()?;
        Ok(header(&m).capacity.load(Acquire))
    }

    pub fn foreign(&self) -> Result<u64> {
        let m = self.map()?;
        Ok(header(&m).foreign.load(Acquire))
    }

    pub fn set_foreign(&self, bytes: u64) -> Result<()> {
        let m = self.map()?;
        header(&m).foreign.store(bytes, Release);
        Ok(())
    }

    /// TensorD samples foreign bytes (its ledger is live): an executor reads them.
    pub fn sampled(&self) -> Result<bool> {
        let m = self.map()?;
        Ok(header(&m).sampled.load(Acquire) == 1)
    }

    fn set_sampled(&self, on: bool) -> Result<()> {
        let m = self.map()?;
        header(&m).sampled.store(u32::from(on), Release);
        Ok(())
    }

    /// The largest foreign process on the device (its name and bytes), as TensorD last saw it.
    pub fn foreign_holder(&self) -> Result<Option<(String, u64)>> {
        let Some(v) = self.fact(FOREIGN_HOLDER)? else { return Ok(None) };
        let (bytes, name) = v.split_at_checked(8).ok_or_else(|| Error::Internal("foreign holder fact".into()))?;
        Ok(Some((String::from_utf8_lossy(name).into_owned(), u64::from_le_bytes(bytes.try_into().map_err(|_| Error::Internal("foreign holder fact".into()))?))))
    }

    pub fn shared_mode(&self) -> Result<bool> {
        let m = self.map()?;
        Ok(header(&m).mode.load(Acquire) == 0)
    }

    fn set_mode(&self, mode: Mode) -> Result<()> {
        let m = self.map()?;
        header(&m).mode.store(u32::from(mode == Mode::Dedicated), Release);
        Ok(())
    }

    /// Room freed, a row died, an ask went out or a peak was recorded: TensorD's event, from
    /// every participant (granted or standalone) through the GPU's named wake.
    pub fn ring(&self) {
        self.event();
        // And every row waiting for room.
        if let Ok(rows) = self.waiting() {
            rows.into_iter().for_each(|id| os::wake(&self.wake, self.ns, &row_wake(&self.name, id)));
        }
    }

    /// TensorD's event alone.
    fn event(&self) {
        os::wake(&self.wake, self.ns, &self.name);
    }

    /// The rows waiting for room (their ids).
    fn waiting(&self) -> Result<Vec<u64>> {
        fence(SeqCst);
        self.records(|m, at, w| (kind(w) == ROW && row(m, at)?.waiting.load(SeqCst) > 0).then(|| row_id(at, w)))
    }

    /// Every READY record, read again if the heap's shape moved meanwhile.
    fn records<T>(&self, mut f: impl FnMut(&Map, u64, u64) -> Option<T>) -> Result<Vec<T>> {
        loop {
            let m = self.map()?;
            let h = header(&m);
            let shape = h.shape.load(Acquire);
            let (out, torn, at) = scan(&m, &mut f);
            if h.shape.load(Acquire) == shape {
                // A record is written before the end moves past it: torn at a steady shape is a
                // broken table, not a race.
                return match torn {
                    false => Ok(out),
                    true => Err(Error::Internal(format!("ledger {} is torn at {at}", self.name))),
                };
            }
        }
    }

    /// Allocates `n` cells under the heap lock: the first free run that fits, else the end.
    /// Returns the record's offset and its READY word.
    fn alloc(&self, n: u64, kind: u64, init: impl FnOnce(&Map, u64)) -> Result<(u64, u64)> {
        let m = self.map()?;
        let _heap = self.heap()?;
        self.alloc_locked(m, n, kind, init).map(|(at, ready, _)| (at, ready))
    }

    /// Under the heap lock; also returns a mapping that covers the record.
    fn alloc_locked(&self, m: Arc<Map>, n: u64, kind: u64, init: impl FnOnce(&Map, u64)) -> Result<(u64, u64, Arc<Map>)> {
        let m = self.covering(m.clone(), header(&m).end.load(Acquire))?;
        let end = header(&m).end.load(Acquire);
        // First fit over free runs, merging neighbours as it walks.
        let (mut at, mut run_start, mut run) = (HEADER, None, 0u64);
        let mut found = None;
        while at < end {
            let w = atomic(&m, at).ok_or_else(|| Error::Internal("ledger heap outside its map".into()))?.load(Acquire);
            let size = cells(w).max(1);
            if state(w) == FREE {
                let start = *run_start.get_or_insert(at);
                run += size;
                if run >= n {
                    found = Some(start);
                    break;
                }
            } else {
                run_start = None;
                run = 0;
            }
            at += size * CELL;
        }
        let mut grown = None;
        let m = match found {
            Some(_) => m,
            None => {
                let start = run_start.unwrap_or(end);
                let need = start + n * CELL;
                let size = os::size(&self.fd)?;
                if need > size {
                    // Doubling keeps growth to a logarithmic number of remaps.
                    os::resize(&self.fd, need.max(size * 2))?;
                }
                found = Some(start);
                grown = (need > end).then_some(need);
                self.covering(m, need)?
            }
        };
        let start = found.unwrap_or(end);
        let h = header(&m);
        let tail = run.saturating_sub(n);
        if tail > 0 {
            let w = atomic(&m, start + n * CELL).ok_or_else(|| Error::Internal("ledger tail".into()))?;
            w.store(word(tail, 0, 0, FREE), Release);
        }
        let g = h.shape.load(Acquire) % 0xffff + 1;
        let w = atomic(&m, start).ok_or_else(|| Error::Internal("ledger record".into()))?;
        w.store(word(n, g, kind, CLAIMING), Release);
        init(&m, start);
        if let Some(end) = grown {
            h.end.store(end, Release);
        }
        h.shape.fetch_add(1, AcqRel);
        Ok((start, word(n, g, kind, READY), m))
    }

    /// Frees the record at `at` if its word is still `seen` (and, `dead`, no one holds its row
    /// lock): a stale reader frees nothing newer.
    fn free(&self, at: u64, seen: u64, dead: bool) -> Result<bool> {
        let m = self.map()?;
        let _heap = self.heap()?;
        self.free_locked(m, at, seen, dead)
    }

    fn free_locked(&self, m: Arc<Map>, at: u64, seen: u64, dead: bool) -> Result<bool> {
        let m = self.covering(m, at + cells(seen) * CELL)?;
        let Some(w) = atomic(&m, at) else { return Ok(false) };
        if w.load(Acquire) != seen || (dead && os::held(&self.fd, at, ROW_CELLS * CELL)?) {
            return Ok(false);
        }
        w.store(word(cells(seen), 0, 0, FREE), Release);
        header(&m).shape.fetch_add(1, AcqRel);
        Ok(true)
    }

    /// Retire the row lock before its heap record becomes reusable, under the same heap
    /// lock as allocation. The Row value may live until its destructor finishes afterward.
    fn release_row(&self, row: &Row) -> Result<bool> {
        let m = self.map()?;
        let _heap = self.heap()?;
        os::unlock(&row._lock, row.at, ROW_CELLS * CELL)?;
        self.free_locked(m, row.at, row.word, false)
    }

    /// After a row left: with no row left and nobody else holding the table open, its name
    /// goes (a standalone engine's table, its last user gone).
    fn leave(&self) -> Result<()> {
        // A Shared can have several rows and callers in this process. Its OPEN lock is one
        // file description, so only the heap mutex excludes a concurrent local claim/leave.
        let _heap = self.heap()?;
        let m = self.map()?;
        let end = header(&m).end.load(Acquire);
        let mut at = HEADER;
        while at < end {
            let w = atomic(&m, at).ok_or_else(|| Error::Internal("ledger leave outside its map".into()))?.load(Acquire);
            if cells(w) == 0 || at + cells(w) * CELL > end {
                return Err(Error::Internal("ledger leave found a torn record".into()));
            }
            // A claim owns the generation before it publishes READY or takes its row lock.
            if kind(w) == ROW && state(w) != FREE {
                return Ok(());
            }
            at += cells(w) * CELL;
        }
        if !os::lock(&self.fd, OPEN, 8, false)? {
            return Ok(());
        }
        // A delayed second leave on the old description must never unlink a replacement
        // created under the same name after the first leave retired this generation.
        let linked = os::linked(&self.fd);
        if matches!(linked, Ok(true)) {
            os::unlink(&self.name);
        }
        os::share(&self.fd, OPEN, 8, false)?;
        linked.map(|_| ())
    }

    // ---- rows

    /// A row for `pid`, held by an OFD lock on a description of its own.
    pub fn claim(self: &Arc<Self>, pid: u32) -> Result<Row> {
        self.claim_with(pid, os::reopen(&self.fd, true)?)
    }

    /// A row held by a lock on `lock`, a description of this table no one else uses (a grant's).
    pub fn claim_with(self: &Arc<Self>, pid: u32, lock: Owned) -> Result<Row> {
        let (at, ready) = self.alloc(ROW_CELLS, ROW, |m, at| {
            if let Some(r) = row(m, at) {
                r.pid.store(pid, Relaxed);
                r.tid.store(0, Relaxed);
                r.born.store(header(m).shape.load(Relaxed).wrapping_add(1), Relaxed);
                r.physical_epoch.store(1, Relaxed);
                r.physical_active.store(0, Relaxed);
                r.orphan_epoch.store(0, Relaxed);
                [&r.asking, &r.asked, &r.orphans, &r.waiting, &r.seq].iter().for_each(|a| a.store(0, Relaxed));
                r.latch.iter().flatten().for_each(|a| a.store(0, Relaxed));
            }
        })?;
        let claiming = ready & !0xff | CLAIMING;
        if !matches!(os::lock(&lock, at, ROW_CELLS * CELL, false), Ok(true)) {
            self.free(at, claiming, false)?;
            return Err(Error::Internal("ledger row lock refused".into()));
        }
        let m = self.map()?;
        let rec = row(&m, at).ok_or_else(|| Error::Internal("ledger row".into()))?;
        rec.word.store(ready, Release);
        Ok(Row { shared: self.clone(), at, word: ready, map: m, _lock: lock, latch: Mutex::new(()), asking: Mutex::new(()) })
    }

    pub fn rows(&self) -> Result<Vec<RowView>> {
        let views = self.records(|m, at, w| {
            if kind(w) != ROW {
                return None;
            }
            let r = row(m, at)?;
            // Epoch/active surround both publications. An operation starting or finishing
            // during this read invalidates the snapshot; a fresh one can validate later.
            let epoch = r.physical_epoch.load(SeqCst);
            let active = r.physical_active.load(SeqCst);
            let facts = Facts::of(latch_read(&r.seq, &r.latch));
            let orphans = r.orphans.load(Acquire);
            let orphan_epoch = r.orphan_epoch.load(Acquire);
            let physical = PhysicalStamp {
                born: r.born.load(Acquire), epoch, facts: facts.physical_epoch, orphans: orphan_epoch,
                clean: active == 0 && r.physical_active.load(SeqCst) == 0 && r.physical_epoch.load(SeqCst) == epoch,
            };
            Some(RowView { off: at, word: w, pid: r.pid.load(Acquire), live: true, facts, orphans, physical, asking: r.asking.load(Acquire), asked: r.asked.load(Acquire), tid: r.tid.load(Acquire), waiting: r.waiting.load(Acquire) > 0 })
        })?;
        views
            .into_iter()
            .map(|mut v| -> Result<RowView> {
                v.live = os::held(&self.fd, v.off, ROW_CELLS * CELL)?;
                Ok(v)
            })
            .collect()
    }

    /// Frees dead rows (of `pid`, or any): their holders are gone, so their locks are free.
    pub fn reap(&self, pid: Option<u32>) -> Result<bool> {
        let mut any = false;
        let rows = self.rows()?;
        #[cfg(feature = "faults")]
        crate::faults::hit(crate::faults::Point::Reap);
        for r in rows {
            if !r.live && pid.is_none_or(|p| p == r.pid) {
                any |= self.free(r.off, r.word, true)?;
            }
        }
        if any {
            self.ring();
        }
        Ok(any)
    }

    /// Whether `id` still names a live row record (a reused offset is another row).
    fn names(&self, id: u64) -> Result<bool> {
        let at = (id >> 16) * CELL;
        let m = self.map()?;
        Ok(row(&m, at).is_some_and(|r| {
            let w = r.word.load(Acquire);
            state(w) == READY && kind(w) == ROW && row_id(at, w) == id
        }))
    }

    // ---- facts

    /// The newest value of each key under `prefix`.
    pub fn facts(&self, prefix: &str) -> Result<Vec<(String, Vec<u8>)>> {
        let found = self.records(|m, at, w| {
            if kind(w) != FACT {
                return None;
            }
            let (key, value, version) = fact(m, at)?;
            key.starts_with(prefix).then_some((key, value, version))
        })?;
        let mut newest: HashMap<String, (u64, Vec<u8>)> = HashMap::new();
        for (key, value, version) in found {
            if newest.get(&key).is_none_or(|(v, _)| *v < version) {
                newest.insert(key, (version, value));
            }
        }
        let mut out: Vec<_> = newest.into_iter().map(|(k, (_, v))| (k, v)).collect();
        out.sort();
        Ok(out)
    }

    pub fn fact(&self, key: &str) -> Result<Option<Vec<u8>>> {
        Ok(self.facts(key)?.into_iter().find(|(k, _)| k == key).map(|(_, v)| v))
    }

    /// Writes `key` (a newer version supersedes the old record); `None` deletes it.
    pub fn put(&self, key: &str, value: Option<&[u8]>) -> Result<()> {
        self.merge(key, |_| value.map(<[u8]>::to_vec))
    }

    /// Writes `key` as `f` makes it from its newest value, atomically across processes (under
    /// the heap lock): concurrent writers merge rather than overwrite. `None` deletes it.
    pub fn merge(&self, key: &str, f: impl FnOnce(Option<&[u8]>) -> Option<Vec<u8>>) -> Result<()> {
        let m = self.map()?;
        let _heap = self.heap()?;
        let m = self.covering(m.clone(), header(&m).end.load(Acquire))?;
        // The heap's shape cannot move under its lock: one scan is a steady one.
        let (old, ..) = scan(&m, &mut |m: &Map, at, w| {
            if kind(w) != FACT {
                return None;
            }
            let (k, value, version) = fact(m, at)?;
            (k == key).then_some((at, w, version, value))
        });
        let newest = old.iter().max_by_key(|(_, _, v, _)| *v);
        let value = f(newest.map(|(.., v)| v.as_slice()));
        if value.as_ref() == newest.map(|(.., v)| v) && old.len() <= 1 {
            return Ok(());
        }
        if let Some(value) = value {
            let version = newest.map_or(0, |(_, _, v, _)| *v) + 1;
            let bytes = FACT_HEAD + key.len() as u64 + value.len() as u64;
            let (at, ready, m) = self.alloc_locked(m.clone(), bytes.div_ceil(CELL), FACT, |m, at| {
                // SAFETY: a CLAIMING record of `bytes.div_ceil(CELL)` cells inside the map: no
                // reader parses it.
                unsafe {
                    let rec = m.base.add(at as usize) as *mut FactRec;
                    (*rec).version = version;
                    (*rec).klen = key.len() as u32;
                    (*rec).vlen = value.len() as u32;
                    let body = m.base.add((at + FACT_HEAD) as usize);
                    std::ptr::copy_nonoverlapping(key.as_ptr(), body, key.len());
                    std::ptr::copy_nonoverlapping(value.as_ptr(), body.add(key.len()), value.len());
                }
            })?;
            atomic(&m, at).ok_or_else(|| Error::Internal("ledger fact".into()))?.store(ready, Release);
        }
        for (at, w, ..) in old {
            self.free_locked(m.clone(), at, w, false)?;
        }
        Ok(())
    }
}

/// A request's anonymous host peak at `host/<package>|<shape>`: the larger stays.
pub(crate) fn record_host(shared: &Shared, key: &str, bytes: u64) -> Result<()> {
    record_peak(shared, &format!("host/{key}"), bytes)
}

/// A process running requests on a GPU, as TensorD's host rule reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostCall {
    pub pid: u32,
    /// The anonymous peak its running requests were measured to reach; `None`: one is unmeasured.
    pub peak: Option<u64>,
    /// Its anonymous bytes now.
    pub anon: u64,
    /// Its requests running now.
    pub running: u64,
}

/// The processes running requests on this GPU; `None`: its table cannot be read.
pub(crate) fn host_calls(shared: &Shared) -> Option<Vec<HostCall>> {
    let rows = shared.rows().ok()?;
    let calls = rows.iter().filter(|r| r.live && r.facts.running > 0).map(|r| HostCall {
        pid: r.pid,
        peak: (r.facts.host_unmeasured == 0).then_some(r.facts.host_peak),
        anon: host_memory::resident(r.pid).map_or(0, |now| now.anon),
        running: r.facts.running,
    });
    Some(calls.collect())
}

/// This process's anonymous peak while it runs requests. Linux keeps only a resident high-water
/// (`VmHWM`), so each look bounds the anonymous peak since the last as that high-water less the
/// smaller of its file bytes then and now and less its shmem bytes now (the host pages it maps:
/// counted once, in the budget, never as need), then starts it over.
pub(crate) struct Window {
    peak: u64,
    file: u64,
}

impl Window {
    /// Measured from now; `None` where it cannot be read.
    pub(crate) fn start() -> Option<Window> {
        let now = host_memory::resident(std::process::id()).ok()?;
        restart();
        Some(Window { peak: now.anon, file: now.file })
    }

    /// The anonymous peak so far; `None` where it cannot be read.
    pub(crate) fn look(&mut self) -> Option<u64> {
        let now = host_memory::resident(std::process::id()).ok()?;
        let bound = now.peak.saturating_sub(self.file.min(now.file) + now.shmem);
        self.peak = self.peak.max(bound).max(now.anon);
        self.file = now.file;
        restart();
        Some(self.peak)
    }
}

/// This process's resident high-water starts over (where it may not, the next look bounds from
/// further back).
fn restart() {
    if let Err(e) = host_memory::reset_peak(std::process::id()) {
        tracing::debug!("varena: host high-water not reset: {e}");
    }
}

/// A request's compute bytes (beside its weights) at `compute/<package>|<shape>`: the latest
/// run's replaces what was there (one large run never pins it).
pub(crate) fn record_compute(shared: &Shared, key: &str, bytes: u64) -> Result<()> {
    shared.put(&format!("compute/{key}"), Some(&bytes.to_le_bytes()))
}

/// Merges a request's working set into `peak/<package>|<shape>`: the larger stays, whoever
/// writes last.
pub(crate) fn record_peak(shared: &Shared, key: &str, bytes: u64) -> Result<()> {
    shared.merge(key, |old| {
        let known = old.and_then(|v| v.try_into().ok()).map(u64::from_le_bytes);
        Some(known.map_or(bytes, |k| k.max(bytes)).to_le_bytes().to_vec())
    })
}

/// A set's resident bytes at `set/<name>`, from its layout: the larger stays.
pub(crate) fn record_set(shared: &Shared, name: &str, bytes: u64) -> Result<()> {
    shared.merge(&format!("set/{name}"), |old| {
        let known = old.and_then(|v| v.try_into().ok()).map(u64::from_le_bytes);
        Some(known.map_or(bytes, |k| k.max(bytes)).to_le_bytes().to_vec())
    })
}

/// Writes `key` only where the table has nothing newer (a restore: what this run measured
/// wins over what an earlier one saved).
fn restore_fact(shared: &Shared, key: &str, value: Vec<u8>) -> Result<()> {
    shared.merge(key, |old| Some(old.map_or(value, <[u8]>::to_vec)))
}

/// A family's id as rows publish it (never 0).
pub(crate) fn family_id(family: &str) -> u64 {
    let h = Sha256::digest(family.as_bytes());
    u64::from_le_bytes(std::array::from_fn(|i| h.get(i).copied().unwrap_or(0))).max(1)
}

/// The wake endpoint of row `id`, signalled while it waits for room.
fn row_wake(gpu: &str, id: u64) -> String {
    format!("{gpu}-row-{id}")
}

fn ask_name(gpu: &str, id: u64) -> String {
    format!("ask-{gpu}-{id}")
}

/// The fact naming the largest foreign process: its bytes (8, little-endian), then its name.
const FOREIGN_HOLDER: &str = "foreign/holder";

/// Asks row `id` for `bytes` under `ceiling` over its ask socket; what it freed (a dead or reused
/// row frees nothing).
fn ask_row(shared: &Shared, id: u64, ceiling: Class, bytes: u64) -> Result<u64> {
    if !shared.names(id)? {
        return Ok(0);
    }
    Ok(match os::connect(shared.ns, &ask_name(&shared.name, id)) {
        Ok(Some(stream)) => os::call::<_, Answer>(&stream, &Ask::Room { ceiling: ceiling.index(), bytes }, None).map_or(0, |(a, _)| a.freed),
        _ => 0,
    })
}

/// One process's row. Dropping it publishes zeros, closes its asks and frees it; a killed
/// process's row reads dead until reaped.
pub(crate) struct Row {
    shared: Arc<Shared>,
    at: u64,
    word: u64,
    /// This mapping covers the row for its entire lifetime, even if the heap grows. Native
    /// callbacks touch these atomics without the map RwLock, remapping, or a latch wait.
    map: Arc<Map>,
    _lock: Owned,
    /// One latch writer at a time.
    latch: Mutex<()>,
    /// One ask out at a time: the record names one victim.
    asking: Mutex<()>,
}

impl Row {
    pub fn at(&self) -> u64 {
        self.at
    }

    pub fn id(&self) -> u64 {
        row_id(self.at, self.word)
    }

    fn rec<T>(&self, f: impl FnOnce(&RowRec) -> T) -> Option<T> {
        row(&self.map, self.at).filter(|r| r.word.load(Acquire) == self.word).map(f)
    }

    pub fn publish(&self, facts: &Facts) {
        let Some(r) = row(&self.map, self.at) else { return };
        let _one = self.latch.lock();
        if r.word.load(Acquire) == self.word {
            latch_write(&r.seq, &r.latch, &facts.words());
        }
    }

    /// A stamp taken before collecting authoritative physical counts. Keep it alongside
    /// those counts through deferred publication; taking a new stamp afterward is unsafe.
    pub fn physical_epoch(&self) -> u64 {
        self.rec(|r| r.physical_epoch.load(SeqCst)).unwrap_or(0)
    }

    /// Mark BEFORE touching physical ownership. This is only pinned-map atomic work and is
    /// usable inside native allocation callbacks. Complete only after publishing the change.
    pub fn physical_change(self: &Arc<Self>) -> PhysicalChange {
        self.begin_physical();
        PhysicalChange { row: self.clone() }
    }

    fn begin_physical(&self) {
        self.rec(|r| {
            r.physical_active.fetch_add(1, SeqCst);
            r.physical_epoch.fetch_add(1, SeqCst);
        });
    }

    fn end_physical(&self) {
        self.rec(|r| { r.physical_active.fetch_sub(1, SeqCst); });
    }

    /// The custody service's orphans. The caller brackets earlier driver work with a
    /// PhysicalChange and orders ownership transfers under admission; this setter itself
    /// neither acquires that gate nor proves completion of the earlier work.
    pub fn set_orphans(&self, bytes: u64) {
        self.begin_physical();
        let changed = self.rec(|r| {
            let epoch = r.physical_epoch.load(SeqCst);
            let changed = r.orphans.swap(bytes, AcqRel) != bytes;
            r.orphan_epoch.store(epoch, Release);
            changed
        }).unwrap_or(false);
        self.end_physical();
        if changed { self.shared.ring(); }
    }

    /// Asks row `id` for `bytes` under `ceiling`: its ask thread answers from its pages at once.
    /// Returns what it freed; a dead or reused row frees nothing. What is asked is in this row's
    /// own record while the ask is out; one ask at a time per row.
    pub fn ask(&self, id: u64, ceiling: Class, bytes: u64) -> Result<u64> {
        if !self.shared.names(id)? {
            return Ok(0);
        }
        // Mapped first: the ask lock is a leaf (held across the call, it locks nothing else).
        let m = self.shared.map()?;
        let rec = row(&m, self.at).ok_or_else(|| Error::Internal("ledger row".into()))?;
        let _one = self.asking.lock();
        rec.asked.store(bytes, Release);
        rec.asking.store(id, Release);
        self.shared.ring();
        let freed = ask_row(&self.shared, id, ceiling, bytes);
        rec.asking.store(0, Release);
        rec.asked.store(0, Release);
        freed
    }

    /// Binds this row's ask socket: its owner's ask thread serves it.
    pub fn listen(&self) -> Result<Listener> {
        os::listen(self.shared.ns, &self.ask_name())?.ok_or_else(|| Error::Internal("a row's ask socket is taken".into()))
    }

    /// Its ask socket's name.
    pub fn ask_name(&self) -> String {
        ask_name(&self.shared.name, self.id())
    }

    /// The calling thread is this row's ask thread.
    pub fn ask_thread(&self) {
        let tid = os::tid();
        self.rec(|r| r.tid.store(tid, Release));
    }

    /// Wakes an ask thread blocked in its accept (to stop it).
    pub fn knock(&self) {
        let _ = os::connect(self.shared.ns, &ask_name(&self.shared.name, self.id()));
    }

    /// Serves this row's asks until `stop`: every asker (of this user only) at once, each
    /// answered with what `answer` freed. A silent or stopped asker holds up nobody's answer.
    pub fn serve_asks(listener: os::Listener, stop: &AtomicBool, row: Weak<Row>, answer: impl Fn(Ask) -> u64 + Send + Sync + 'static) {
        os::serve_all(listener, stop, move |body| {
            let freed = os::decode::<Ask>(body).map_or(0, &answer);
            if let Some(r) = row.upgrade() {
                r.shared.ring();
            }
            let reply = os::encode(&Answer { freed }).unwrap_or_default();
            let mut framed = (reply.len() as u32).to_le_bytes().to_vec();
            framed.push(0);
            framed.extend_from_slice(&reply);
            framed
        });
    }
}

/// An outstanding physical operation. Drop deliberately does NOT clear the dirty marker:
/// without a checked publication/completion, old physical counts cannot become evidence.
#[must_use = "finish only once the operation's outcome is certain and counted"]
pub struct PhysicalChange {
    row: Arc<Row>,
}

impl PhysicalChange {
    pub fn finish(self) {
        self.row.end_physical();
    }
}

impl Drop for Row {
    fn drop(&mut self) {
        self.publish(&Facts::default());
        self.set_orphans(0);
        if self.shared.release_row(self).unwrap_or(false) {
            let _ = self.shared.leave();
            self.shared.ring();
        }
        os::closing(&self._lock);
    }
}

/// The READY records of `m` through `f`, whether the walk was torn, and where it stopped.
fn scan<T>(m: &Map, f: &mut impl FnMut(&Map, u64, u64) -> Option<T>) -> (Vec<T>, bool, u64) {
    let end = header(m).end.load(Acquire).min(m.len as u64);
    let mut out = vec![];
    let mut at = HEADER;
    while at < end {
        let Some(w) = atomic(m, at) else { break };
        let w = w.load(Acquire);
        if cells(w) == 0 || at + cells(w) * CELL > end {
            return (out, true, at);
        }
        if state(w) == READY {
            out.extend(f(m, at, w));
        }
        at += cells(w) * CELL;
    }
    fence(Acquire);
    (out, false, at)
}

fn header(m: &Map) -> &Header {
    // SAFETY: the header page is always mapped and aligned; it is read through atomics and
    // fields written once before `magic`.
    unsafe { &*(m.base as *const Header) }
}

fn atomic(m: &Map, at: u64) -> Option<&AtomicU64> {
    (at.is_multiple_of(8) && at + 8 <= m.len as u64).then(|| {
        // SAFETY: inside the map and aligned.
        unsafe { &*(m.base.add(at as usize) as *const AtomicU64) }
    })
}

fn row(m: &Map, at: u64) -> Option<&RowRec> {
    (at.is_multiple_of(CELL) && at + ROW_CELLS * CELL <= m.len as u64).then(|| {
        // SAFETY: inside the map, aligned, and every field is atomic.
        unsafe { &*(m.base.add(at as usize) as *const RowRec) }
    })
}

/// A READY fact's key, value and version.
fn fact(m: &Map, at: u64) -> Option<(String, Vec<u8>, u64)> {
    if at + FACT_HEAD > m.len as u64 {
        return None;
    }
    // SAFETY: inside the map; a READY fact's fields are immutable until it is freed, and the
    // caller rereads when the heap's shape moved.
    let (version, klen, vlen) = unsafe {
        let rec = m.base.add(at as usize) as *const FactRec;
        ((*rec).version, (*rec).klen as u64, (*rec).vlen as u64)
    };
    if at + FACT_HEAD + klen + vlen > m.len as u64 {
        return None;
    }
    // SAFETY: as above, bounds checked.
    let body = unsafe { std::slice::from_raw_parts(m.base.add((at + FACT_HEAD) as usize), (klen + vlen) as usize) };
    let (k, v) = body.split_at(klen as usize);
    Some((String::from_utf8_lossy(k).into_owned(), v.to_vec(), version))
}

/// Copy 0 is written while `seq` is odd, copy 1 while even; a reader reads the other.
fn latch_write(seq: &AtomicU64, copies: &[[AtomicU64; WORDS]; 2], words: &[u64; WORDS]) {
    let s = seq.load(Relaxed);
    seq.store(s + 1, Relaxed);
    fence(Release);
    let [first, second] = copies;
    first.iter().zip(words).for_each(|(a, v)| a.store(*v, Relaxed));
    seq.store(s + 2, Release);
    fence(Release);
    second.iter().zip(words).for_each(|(a, v)| a.store(*v, Relaxed));
}

fn latch_read(seq: &AtomicU64, copies: &[[AtomicU64; WORDS]; 2]) -> [u64; WORDS] {
    loop {
        let s = seq.load(Acquire);
        let [a, b] = copies;
        let copy = if s & 1 == 0 { a } else { b };
        let words: [u64; WORDS] = std::array::from_fn(|i| copy.get(i).map_or(0, |w| w.load(Relaxed)));
        fence(Acquire);
        if seq.load(Relaxed) == s {
            return words;
        }
    }
}

/// The other rows as the decision procedure reads them, and the orphan bytes.
pub(crate) fn remotes(rows: &[RowView], own: u64) -> (Vec<Remote>, u64) {
    let orphans = rows.iter().map(|r| r.orphans).sum();
    let others = rows
        .iter()
        .filter(|r| r.off != own)
        .map(|r| Remote {
            row: r.id(),
            running: r.facts.running > 0 || !r.live,
            context: r.facts.context,
            torch: r.facts.torch,
            physical: r.facts.physical,
            caches: r.facts.caches,
            classes: r.facts.classes,
            returning: if r.live { r.facts.returning } else { 0 },
        })
        .collect();
    (others, orphans)
}

/// Bytes the live rows are asking of row `id` now (each asker's own record).
fn asked_of(rows: &[RowView], id: u64) -> u64 {
    rows.iter().filter(|r| r.live && r.asking == id).map(|r| r.asked).sum()
}

/// Every row's admission charge, for status/free. These include promises, not just physical bytes.
pub(crate) fn held(rows: &[RowView]) -> u64 {
    rows.iter().map(|r| r.facts.held() + r.orphans).sum()
}

// ---- TensorD's side

/// A process TensorD started whose memory protocol has no row of its own (a standalone
/// executor, varena-leases R2-2): its measured bytes, held as context in a row TensorD holds by
/// its pid. Context is no class an ask reaches, so it is never asked; its physical bytes are
/// known, so no executor reads them as foreign. Dropped (at its reap), the row goes.
pub struct FixedRow(Arc<Row>);

impl FixedRow {
    /// Its device bytes now.
    pub fn set(&self, bytes: u64) {
        let change = self.0.physical_change();
        let physical_epoch = self.0.physical_epoch();
        self.0.publish(&Facts { context: bytes, physical: bytes, physical_epoch, ..Facts::default() });
        change.finish();
        self.0.shared.ring();
    }
}

/// What TensorD hands an executor for its [`crate::Memory`]: two descriptions of the table,
/// one for the table and one for the executor's row lock (an executor of another uid cannot
/// open the file itself). Fields are public so a spawner can pass them across exec.
pub struct Grant {
    pub table: Owned,
    pub row: Owned,
    /// A standalone opener's cap on its own row (never the table's: v8, a test's 1 GiB capped
    /// every executor on the card).
    pub limit: Option<u64>,
}

impl Grant {
    /// `gpu`'s table by name, created at the card's capacity when absent, for a standalone
    /// engine; `limit` caps only this opener's row.
    pub fn open(gpu: &GpuId, limit: Option<u64>) -> Result<Grant> {
        // Read-locked open while the grant lives.
        let table = open_named(&gpu.uuid)?;
        drop(Shared::adopt(os::dup(&table)?, gpu, gpu.memory)?);
        let row = os::reopen(&table, true)?;
        Ok(Grant { table, row, limit })
    }
}

/// One row as [`Ledger::status`] shows it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RowStatus {
    pub pid: u32,
    pub live: bool,
    pub context: u64,
    /// Bytes by class (F unused, O, S, N, D, W, P); O is on the custody service's row.
    pub classes: [u64; 7],
    /// The largest remaining working set of each running request, summed.
    pub remaining: u64,
    /// Running requests with no measured working set.
    pub unmeasured: u32,
    /// Torch's bytes as the row counts them: rooms, and claims while its asks are out.
    pub torch: u64,
    /// Bytes asked of it and not yet answered.
    pub asked: u64,
    /// Bytes it is giving back: its retiring pages and its caches' queued shrinks.
    pub returning: u64,
    /// Waiting for room: a request's wait on a child call ended and its room is not back yet.
    pub waiting: bool,
}

/// The table at a glance (display only; reading it changes nothing).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Status {
    pub capacity: u64,
    pub free: u64,
    /// Bytes on the device no row accounts for, beyond what the card leaves outside the cap.
    pub foreign: u64,
    pub rows: Vec<RowStatus>,
    /// Request id → `<package>|<shape>`.
    pub dispatched: Vec<(String, String)>,
}

/// A measured peak, as TensorD persists it per GPU model: an opaque typed record, restored
/// exactly as it was read.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Peak {
    /// A request's largest working set at `<package>|<shape>`.
    Request { key: String, bytes: u64 },
    /// A request's largest allocations beside its weights (its compute bytes) at
    /// `<package>|<shape>`: what placement keeps free on GPU 0.
    Compute { key: String, bytes: u64 },
    /// A set's resident bytes (its layout's span), by set name.
    Set { key: String, bytes: u64 },
    /// A request's host high-water at `<package>|<shape>` (anonymous bytes; file mappings
    /// aside).
    Host { key: String, bytes: u64 },
}

/// TensorD's handle on one GPU's table.
pub struct Ledger {
    shared: Arc<Shared>,
    /// The card's bytes: the capacity again once this TensorD goes (its limit never outlives it).
    card: u64,
    /// The GPU's wake endpoint, bound by name: every participant signals it.
    events: os::Endpoint,
    custody: custody::Client,
    /// Rows that answered a foreign reading's ask with nothing, by ceiling, and their classes
    /// then: not asked again until those change (at the floor every reading would ask them).
    quiet: Mutex<HashMap<(u64, usize), [u64; 7]>>,
}

impl Drop for Ledger {
    fn drop(&mut self) {
        let _ = self.shared.set_capacity(self.card);
        let _ = self.shared.set_sampled(false);
    }
}

impl Ledger {
    /// The GPU's table, named per GPU and user; capacity = min(card, limit), whatever an earlier
    /// opener chose, and the card's again when this ledger goes. A restarted TensorD reopens the
    /// same table and drops every dispatch it finds (TensorD is the only dispatcher: they are a
    /// dead one's). TensorD serves custody when it holds the service lock (it does when it
    /// starts first).
    pub fn create(gpu: &GpuId, limit: Option<u64>) -> Result<Ledger> {
        let capacity = limit.map_or(gpu.memory, |l| l.min(gpu.memory));
        let table = open_named(&gpu.uuid)?;
        let shared = Arc::new(Shared::adopt(table, gpu, capacity)?);
        // TensorD is the capacity's authority: min(card, limit) exactly (an opener's limit is
        // its own row's).
        shared.set_capacity(capacity)?;
        let events = os::wake_bind(shared.ns, &gpu.uuid)?.ok_or_else(|| Error::Invalid(format!("another TensorD holds {}", gpu.uuid)))?;
        shared.reap(None)?;
        for (key, _) in shared.facts("dispatch/")? {
            shared.put(&key, None)?;
        }
        let custody = custody::Client::open(&shared)?;
        Ok(Ledger { shared, card: gpu.memory, events, custody, quiet: Mutex::default() })
    }

    /// The machine's sharing mode (shared until TensorD says dedicated).
    pub fn set_mode(&self, mode: Mode) -> Result<()> {
        self.shared.set_mode(mode)
    }

    /// The largest foreign process on the device (TensorD's sample: its name and bytes; `None`
    /// when none holds any): a request refused at its floor names it.
    pub fn set_foreign_holder(&self, holder: Option<(&str, u64)>) -> Result<()> {
        let value = holder.map(|(name, bytes)| [&bytes.to_le_bytes()[..], name.as_bytes()].concat());
        self.shared.put(FOREIGN_HOLDER, value.as_deref())
    }

    /// TensorD's sample of the device bytes no row accounts for, once per new reading. Shared
    /// mode charges H: short of H free (the rows hold more than capacity − H − `foreign`), it wants
    /// the deficit and H more (2H − free), from orphans, then idle rows' pages (ceiling N), then
    /// running rows (ceiling W: they stream), until met. `fresh` (a new foreign process crossed
    /// its threshold) releases every orphan and asks every live row for its idle pages (N: never
    /// a running set). What it freed; a run is never ended for it.
    pub fn set_foreign(&self, foreign: u64, fresh: bool) -> Result<u64> {
        self.shared.set_foreign(foreign)?;
        self.shared.set_sampled(true)?;
        let rows = self.shared.rows()?;
        let h = if self.shared.shared_mode()? { crate::table::headroom(self.card) } else { 0 };
        let deficit = held(&rows).saturating_sub(self.shared.capacity()?.saturating_sub(foreign + h));
        let want = if deficit > 0 { deficit + h } else { 0 };
        let mut freed = match (fresh, want) {
            (true, _) => self.custody.release(u64::MAX),
            (false, 0) => 0,
            (false, want) => self.custody.release(want),
        };
        let live: Vec<&RowView> = rows.iter().filter(|r| r.live).collect();
        if fresh {
            for r in &live {
                freed += self.ask(r, Class::N, u64::MAX)?;
            }
        }
        for (r, ceiling) in live.iter().filter(|r| r.facts.running == 0).map(|r| (r, Class::N)).chain(live.iter().filter(|r| r.facts.running > 0).map(|r| (r, Class::W))) {
            if freed >= want {
                break;
            }
            freed += self.ask(r, ceiling, want - freed)?;
        }
        if freed > 0 {
            self.shared.ring();
        }
        Ok(freed)
    }

    /// TensorD stopped sampling (nothing held on the device): executors measure foreign bytes
    /// themselves again.
    pub fn stop_sampling(&self) -> Result<()> {
        self.shared.set_sampled(false)
    }

    /// Row `r`'s answer to an ask, unless it answered the same ceiling with nothing and its
    /// classes have not changed since.
    fn ask(&self, r: &RowView, ceiling: Class, bytes: u64) -> Result<u64> {
        let key = (r.id(), ceiling.index());
        if self.quiet.lock().get(&key) == Some(&r.facts.classes) {
            return Ok(0);
        }
        let freed = ask_row(&self.shared, r.id(), ceiling, bytes)?;
        match freed {
            0 => self.quiet.lock().insert(key, r.facts.classes),
            _ => self.quiet.lock().remove(&key),
        };
        Ok(freed)
    }

    /// The descriptors executor `pid` opens its memory with. Its uid joins the uids this
    /// process's services accept (a granted set, never any peer).
    pub fn grant(&self, pid: u32) -> Result<Grant> {
        os::accept_uid(os::uid_of(pid)?);
        let table = self.shared.descriptor()?;
        // A transferred grant owns this generation even before the child creates its row.
        os::share(&table, OPEN, 8, true)?;
        Ok(Grant { table, row: self.shared.descriptor()?, limit: None })
    }

    /// A [`FixedRow`] for `pid`, held by this process.
    pub fn fixed(&self, pid: u32) -> Result<FixedRow> {
        Ok(FixedRow(Arc::new(self.shared.claim(pid)?)))
    }

    /// After `waitpid`: `pid`'s row goes, and the pages it kept become orphans.
    pub fn reap(&self, pid: u32) {
        let _ = self.shared.reap(Some(pid));
        let _ = self.custody.reap(pid);
    }

    #[allow(clippy::type_complexity)]
    fn admitted(&self) -> Result<(u64, u64, Vec<Admitted>, Vec<Option<u64>>)> {
        let rows = self.shared.rows()?;
        let admitted = rows
            .iter()
            .filter(|r| r.live)
            .map(|r| Admitted { context: r.facts.context, outside: r.facts.outside, remaining: r.facts.remaining, running: r.facts.running as u32, unmeasured: r.facts.unmeasured as u32, yielded: r.facts.yielded })
            .collect();
        let dispatched = self.shared.facts("dispatch/")?.into_iter().map(|(_, key)| self.peak(&String::from_utf8_lossy(&key))).collect::<Result<_>>()?;
        Ok((self.shared.capacity()?, self.shared.foreign()?, admitted, dispatched))
    }

    /// A request's measured peak at `<package>|<shape>`; an empty shape means the package's
    /// widest measured one.
    fn peak(&self, key: &str) -> Result<Option<u64>> {
        self.figure("peak", key)
    }

    /// A request's recorded figure (`peak` or `compute`) at `<package>|<shape>`; an empty shape
    /// means the package's widest measured one.
    fn figure(&self, kind: &str, key: &str) -> Result<Option<u64>> {
        let bytes = |v: Vec<u8>| v.try_into().ok().map(u64::from_le_bytes);
        if key.ends_with('|') {
            let all = self.shared.facts(&format!("{kind}/{key}"))?;
            return Ok(all.into_iter().filter_map(|(_, v)| bytes(v)).max());
        }
        Ok(self.shared.fact(&format!("{kind}/{key}"))?.and_then(bytes))
    }

    /// Whether a request of `package` at `shape` (empty: its widest measured shape) can start
    /// now beside everything running.
    pub fn fits(&self, package: &str, shape: &str, family: Option<&str>) -> bool {
        let fit = || -> Result<bool> {
            let (capacity, foreign, rows, dispatched) = self.admitted()?;
            Ok(decide::fits(capacity, foreign, &rows, &dispatched, self.peak(&format!("{package}|{shape}"))?, family.map(family_id)))
        };
        fit().unwrap_or(false)
    }

    /// A multi-GPU group: dispatches `request` on every GPU of `group` if it fits on all, else
    /// on none (an error part-way undoes the dispatches made).
    pub fn admit(group: &[&Ledger], request: &str, package: &str, shape: &str, family: Option<&str>) -> Result<bool> {
        if !group.iter().all(|l| l.fits(package, shape, family)) {
            return Ok(false);
        }
        for (i, l) in group.iter().enumerate() {
            if let Err(e) = l.dispatch(request, package, shape) {
                group.iter().take(i).for_each(|done| done.undispatch(request));
                return Err(e);
            }
        }
        Ok(true)
    }

    /// The GPU of `group` (an index) that keeps each set of `package`'s request at `shape`
    /// ([`decide::place`]). A set's bytes are its layout's when the caller has it (`Some`),
    /// else as its load recorded them on any GPU of the group, else unknown. Each GPU's room is
    /// its capacity less foreign bytes and every live row's unmovable bytes (context; torch, W
    /// and P, at least its running requests' peaks; a yielded parent's W unless it waits for
    /// `family`); GPU 0's less the
    /// package's measured compute room too (its room beside its weights, so where its sets
    /// lived when it was measured does not move them).
    pub fn place(group: &[&Ledger], package: &str, shape: &str, family: Option<&str>, sets: &[(&str, Option<u64>)]) -> Result<Vec<u32>> {
        let family = family.map(family_id);
        let mut rooms = Vec::with_capacity(group.len());
        for (g, l) in group.iter().enumerate() {
            // A yielded parent's W comes back, except to its own family's child.
            // A running request holds its recorded peak (its pages and allocations) at least.
            let unmovable = |f: &Facts| f.context + (f.torch + [Class::W, Class::P].iter().filter_map(|c| f.classes.get(c.index())).sum::<u64>()).max(f.remaining) + decide::yielded_beside(&f.yielded, family);
            let held: u64 = l.shared.rows()?.iter().filter(|r| r.live).map(|r| unmovable(&r.facts)).sum();
            // GPU 0 keeps the package's compute room free: its measured room beside its
            // weights, wherever those lived when it was measured.
            let working = match g {
                0 => l.figure("compute", &format!("{package}|{shape}"))?.unwrap_or(0),
                _ => 0,
            };
            rooms.push(l.shared.capacity()?.saturating_sub(l.shared.foreign()? + held + working));
        }
        let mut bytes = Vec::with_capacity(sets.len());
        for (name, known) in sets {
            let recorded = || -> Result<Option<u64>> {
                let mut most = None;
                for l in group {
                    let b = l.shared.fact(&format!("set/{name}"))?.and_then(|v| v.try_into().ok()).map(u64::from_le_bytes);
                    most = most.max(b);
                }
                Ok(most)
            };
            bytes.push(match known {
                Some(b) => Some(*b),
                None => recorded()?,
            });
        }
        Ok(decide::place(&rooms, &bytes))
    }

    /// The processes running requests on this GPU, as TensorD's host rule reads them; `None`:
    /// its table cannot be read.
    pub fn host_calls(&self) -> Option<Vec<HostCall>> {
        host_calls(&self.shared)
    }

    /// Names the `<package>|<shape>` a request's peak is recorded under, before it runs. The
    /// executor consumes it when the request opens.
    pub fn dispatch(&self, request: &str, package: &str, shape: &str) -> Result<()> {
        self.shared.put(&format!("dispatch/{request}"), Some(format!("{package}|{shape}").as_bytes()))
    }

    pub fn undispatch(&self, request: &str) {
        let _ = self.shared.put(&format!("dispatch/{request}"), None);
    }

    /// The measured peaks, to persist per (GPU model, package release).
    pub fn peaks(&self) -> Vec<Peak> {
        let mut out = vec![];
        for (key, v) in self.shared.facts("peak/").unwrap_or_default() {
            if let Ok(bytes) = v.try_into().map(u64::from_le_bytes) {
                out.push(Peak::Request { key: key.trim_start_matches("peak/").to_string(), bytes });
            }
        }
        for (key, v) in self.shared.facts("host/").unwrap_or_default() {
            if let Ok(bytes) = v.try_into().map(u64::from_le_bytes) {
                out.push(Peak::Host { key: key.trim_start_matches("host/").to_string(), bytes });
            }
        }
        for (key, v) in self.shared.facts("compute/").unwrap_or_default() {
            if let Ok(bytes) = v.try_into().map(u64::from_le_bytes) {
                out.push(Peak::Compute { key: key.trim_start_matches("compute/").to_string(), bytes });
            }
        }
        for (key, v) in self.shared.facts("set/").unwrap_or_default() {
            if let Ok(bytes) = v.try_into().map(u64::from_le_bytes) {
                out.push(Peak::Set { key: key.trim_start_matches("set/").to_string(), bytes });
            }
        }
        out
    }

    /// Peaks persisted earlier on this machine, for this GPU model, by recency: a key the table
    /// already measured keeps its newer value; restoring what was saved changes nothing.
    pub fn restore(&self, peaks: &[Peak]) -> Result<()> {
        for p in peaks {
            match p {
                Peak::Request { key, bytes } => restore_fact(&self.shared, &format!("peak/{key}"), bytes.to_le_bytes().to_vec())?,
                Peak::Set { key, bytes } => record_set(&self.shared, key, *bytes)?,
                Peak::Host { key, bytes } => record_host(&self.shared, key, *bytes)?,
                Peak::Compute { key, bytes } => restore_fact(&self.shared, &format!("compute/{key}"), bytes.to_le_bytes().to_vec())?,
            }
        }
        Ok(())
    }

    pub fn status(&self) -> Status {
        let Ok(rows) = self.shared.rows() else { return Status::default() };
        let capacity = self.shared.capacity().unwrap_or(0);
        let foreign = self.shared.foreign().unwrap_or(0);
        let dispatched = self.shared.facts("dispatch/").unwrap_or_default();
        let classes = |r: &RowView| {
            let mut c = r.facts.classes;
            if let Some(o) = c.get_mut(Class::O.index()) {
                *o += r.orphans;
            }
            c
        };
        Status {
            capacity,
            free: capacity.saturating_sub(foreign + held(&rows)),
            foreign,
            rows: rows
                .iter()
                .map(|r| RowStatus { pid: r.pid, live: r.live, context: r.facts.context, classes: classes(r), remaining: r.facts.remaining, unmeasured: r.facts.unmeasured as u32, torch: r.facts.torch, asked: asked_of(&rows, r.id()), returning: r.facts.returning, waiting: r.waiting })
                .collect(),
            dispatched: dispatched.into_iter().map(|(k, v)| (k.trim_start_matches("dispatch/").to_string(), String::from_utf8_lossy(&v).into_owned())).collect(),
        }
    }

    /// Executors with an ask pending to them whose ask thread cannot answer, by its measured
    /// state: stopped (`T`, `t`) or its process a zombie. A running or sleeping thread is never
    /// stalled, nor a stopped one nobody is asking; nothing is counted or timed. Off Linux
    /// nothing is reported. TensorD ends what it returns.
    pub fn stalled(&self) -> Vec<u32> {
        let rows = self.shared.rows().unwrap_or_default();
        rows.iter().filter(|r| r.tid != 0 && asked_of(&rows, r.id()) > 0 && os::stalled(r.pid, r.tid)).map(|r| r.pid).collect()
    }

    /// Readable when room frees, a row dies, an ask goes out or a peak is recorded, in any
    /// process on the GPU; [`Ledger::clear_events`] clears it.
    pub fn events(&self) -> &Owned {
        self.events.handle()
    }

    /// Clears every event so far; true when one came from an accepted user (another user's
    /// signals are not events).
    pub fn clear_events(&self) -> bool {
        os::wake_clear(&self.events)
    }

    /// Blocks until the next event from an accepted user, and clears every one so far.
    pub fn await_event(&self) -> Result<()> {
        os::wake_await(&self.events)
    }

    /// An event for TensorD (the lease manager's: a grant, a revoke answered, an exit).
    pub fn ring(&self) {
        self.shared.ring();
    }

    /// A GPU fault or sticky error: the GPU's kept pages go.
    pub fn fault(&self) {
        let _ = self.custody.fault();
        self.shared.ring();
    }
}

// Named-table retirement belongs to Shared::leave, under the OPEN lock. Dropping TensorD's
// handle cannot unlink while same-process grants or a custody row still own this generation:
// a delayed old row could otherwise unlink a replacement table with the same GPU name.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]

    use super::*;

    /// S3's gpu12 and gpu14 (two wheels' A/B on one pod): another build's table names its format
    /// and owners while one has it open, and is reclaimed once none does.
    #[test]
    fn another_builds_table_is_named_while_open_and_reclaimed_after() {
        let gpu = GpuId { uuid: format!("unit-{}-format", std::process::id()), model: "host".into(), memory: 1 << 30 };
        let old = os::shared(&gpu.uuid, true).unwrap();
        os::resize(&old, HEADER).unwrap();
        let map = os::map(&old, HEADER as usize, true).unwrap();
        // SAFETY: the header page, mapped writable, this test's alone.
        unsafe { (*(map.base as *mut Header)).magic = u64::from_le_bytes(*b"vledgr11") };
        drop(map);
        // An owner of the other build has it open.
        os::share(&old, OPEN, 8, true).unwrap();
        let refused = Shared::adopt(open_named(&gpu.uuid).unwrap(), &gpu, gpu.memory).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(refused.contains("format vledgr11") && refused.contains("have it open"), "another build's open table: {refused}");
        drop(old);
        let shared = Shared::adopt(open_named(&gpu.uuid).unwrap(), &gpu, gpu.memory).unwrap();
        assert!(shared.rows().unwrap().is_empty(), "reclaimed: this build's, empty");
        drop(shared);
        os::remove_table(&gpu.uuid);
    }

    /// Review 1, finding 1: a reaper or writer acting on a stale view frees nothing once the
    /// offset holds a newer record: the free compares the whole word, generation included.
    #[test]
    fn a_stale_word_frees_nothing_newer() {
        let gpu = GpuId { uuid: format!("unit-{}-stale", std::process::id()), model: "host".into(), memory: 1 << 30 };
        let shared = Arc::new(Shared::adopt(os::shared(&gpu.uuid, true).unwrap(), &gpu, gpu.memory).unwrap());
        let first = shared.claim(1).unwrap();
        let (at, stale) = (first.at, first.word);
        drop(first);
        let second = shared.claim(2).unwrap();
        assert_eq!(second.at, at, "the offset is reused");
        assert_ne!(second.word, stale, "with a new generation");
        assert!(!shared.free(at, stale, true).unwrap(), "a stale reaper frees nothing");
        assert!(!shared.free(at, stale, false).unwrap(), "nor a stale writer");
        assert!(shared.rows().unwrap().iter().any(|r| r.off == at && r.pid == 2 && r.live));
        // Facts: a superseded version is freed by its own word only.
        shared.put("k", Some(b"1")).unwrap();
        shared.put("k", Some(b"2")).unwrap();
        assert_eq!(shared.fact("k").unwrap(), Some(b"2".to_vec()));
        // An ask to the freed row's id reads it as gone, never a hang.
        assert_eq!(second.ask(row_id(at, stale), Class::N, 1).unwrap(), 0);
        drop(second);
        os::unlink(&gpu.uuid);
    }

    #[test]
    fn a_retired_row_releases_its_lock_before_the_offset_is_reused() {
        let gpu = GpuId { uuid: format!("unit-{}-row-release", std::process::id()), model: "host".into(), memory: 64 << 20 };
        let shared = Arc::new(Shared::adopt(open_named(&gpu.uuid).unwrap(), &gpu, gpu.memory).unwrap());
        let old = shared.claim(std::process::id()).unwrap();
        assert!(shared.release_row(&old).unwrap());
        // Keep the old Row and its descriptor alive, as while its destructor is still doing
        // table cleanup. The new row must already be able to lock this exact offset.
        let next = shared.claim(std::process::id()).unwrap();
        assert_eq!(next.at, old.at);
        assert_ne!(next.word, old.word);
        next.publish(&Facts { torch: 123, ..Facts::default() });
        drop(old);
        assert!(shared.names(next.id()).unwrap());
        assert_eq!(shared.rows().unwrap().iter().find(|r| r.id() == next.id()).unwrap().facts.torch, 123, "a stale row cannot publish over its successor");
    }

    #[test]
    fn a_delayed_leave_cannot_unlink_the_next_table_generation() {
        let gpu = GpuId { uuid: format!("unit-{}-leave-generation", std::process::id()), model: "host".into(), memory: 64 << 20 };
        let old = Arc::new(Shared::adopt(open_named(&gpu.uuid).unwrap(), &gpu, 12 << 20).unwrap());
        drop(old.claim(std::process::id()).unwrap());
        assert!(!os::linked(&old.fd).unwrap());
        let next = Arc::new(Shared::adopt(open_named(&gpu.uuid).unwrap(), &gpu, 8 << 20).unwrap());
        let row = next.claim(std::process::id()).unwrap();
        old.leave().unwrap();
        assert!(os::linked(&next.fd).unwrap(), "a delayed leave owns only its old generation");
        let reopened = Shared::adopt(open_named(&gpu.uuid).unwrap(), &gpu, gpu.memory).unwrap();
        assert_eq!(reopened.capacity().unwrap(), 8 << 20, "the cap survives reopening by name");
        drop(reopened);
        drop(row);
    }

    #[test]
    fn a_claiming_row_keeps_the_table_linked() {
        let gpu = GpuId { uuid: format!("unit-{}-claim-generation", std::process::id()), model: "host".into(), memory: 64 << 20 };
        let shared = Arc::new(Shared::adopt(open_named(&gpu.uuid).unwrap(), &gpu, 8 << 20).unwrap());
        // Stop precisely between heap allocation and publishing the new row READY.
        let (at, ready) = shared.alloc(ROW_CELLS, ROW, |_, _| {}).unwrap();
        shared.leave().unwrap();
        assert!(os::linked(&shared.fd).unwrap(), "the unpublished claim already owns this generation");
        assert!(shared.free(at, ready & !0xff | CLAIMING, false).unwrap());
        shared.leave().unwrap();
        assert!(!os::linked(&shared.fd).unwrap());
    }

    #[test]
    fn an_incompatible_table_is_refused_without_resetting_live_rows() {
        let gpu = GpuId { uuid: format!("unit-{}-version", std::process::id()), model: "host".into(), memory: 64 << 20 };
        let shared = Arc::new(Shared::adopt(open_named(&gpu.uuid).unwrap(), &gpu, 8 << 20).unwrap());
        let live = shared.claim(std::process::id()).unwrap();
        let m = shared.map().unwrap();
        // SAFETY: no other thread opens or reads this test's private header while it is changed.
        unsafe { (*(m.base as *mut Header)).magic = u64::from_le_bytes(*b"vledger9"); }
        assert!(matches!(Shared::adopt(open_named(&gpu.uuid).unwrap(), &gpu, gpu.memory), Err(Error::Closed(_))));
        assert_eq!(shared.capacity().unwrap(), 8 << 20);
        assert!(shared.names(live.id()).unwrap());
        // SAFETY: as above; restore the fixture's header for normal cleanup.
        unsafe { (*(m.base as *mut Header)).magic = MAGIC; }
    }

    /// A table's namespace is its owner's; a process's own uid is what `grant` reads for it.
    #[test]
    fn a_table_is_named_for_its_owner() {
        let gpu = GpuId { uuid: format!("unit-{}-owner", std::process::id()), model: "host".into(), memory: 1 << 30 };
        let shared = Shared::adopt(os::shared(&gpu.uuid, true).unwrap(), &gpu, gpu.memory).unwrap();
        assert_eq!(shared.ns, os::euid());
        assert_eq!(os::uid_of(std::process::id()).unwrap(), os::euid());
        os::unlink(&gpu.uuid);
    }

    /// Review 2, finding 14: concurrent writers of one fact merge (the larger stays), whatever
    /// their order.
    #[test]
    fn concurrent_peaks_merge_to_the_largest() {
        let gpu = GpuId { uuid: format!("unit-{}-merge", std::process::id()), model: "host".into(), memory: 1 << 30 };
        let shared = Arc::new(Shared::adopt(os::shared(&gpu.uuid, true).unwrap(), &gpu, gpu.memory).unwrap());
        let writers: Vec<_> = (1..=8u64)
            .map(|i| {
                let shared = shared.clone();
                std::thread::spawn(move || (0..16).for_each(|_| record_peak(&shared, "peak/pkg|1x1", i << 20).unwrap()))
            })
            .collect();
        writers.into_iter().for_each(|w| w.join().unwrap());
        let bytes = shared.fact("peak/pkg|1x1").unwrap().map(|v| u64::from_le_bytes(v.try_into().unwrap()));
        assert_eq!(bytes, Some(8 << 20));
        assert_eq!(shared.facts("peak/").unwrap().len(), 1, "one record per key");
        os::unlink(&gpu.uuid);
    }

    /// Review 2, finding 14: a standalone engine's table is unlinked when its last row leaves
    /// (custody stops once nobody is connected and nothing is kept).
    #[cfg(target_os = "linux")]
    #[test]
    fn a_standalone_table_goes_with_its_last_row() {
        let gpu = GpuId { uuid: format!("unit-{}-unlink", std::process::id()), model: "host".into(), memory: 1 << 30 };
        // TensorD's wake, without a TensorD holding the table open.
        let wake = os::wake_bind(os::euid(), &gpu.uuid).unwrap().unwrap();
        let mem = crate::Memory::open(gpu.clone(), None, Box::new(crate::HostBackend::new(gpu.memory))).unwrap();
        // SAFETY: geteuid reads a process attribute.
        let path = format!("/dev/shm/cozy-varena-{}-{}", unsafe { libc::geteuid() }, gpu.uuid);
        assert!(std::path::Path::new(&path).exists());
        drop(mem);
        while std::path::Path::new(&path).exists() {
            os::wake_await(&wake).unwrap();
        }
    }

    #[test]
    fn physical_facts_are_published_without_turning_promises_into_device_usage() {
        let gpu = GpuId { uuid: format!("unit-{}-physical-facts", std::process::id()), model: "host".into(), memory: 64 << 20 };
        let shared = Arc::new(Shared::adopt(open_named(&gpu.uuid).unwrap(), &gpu, gpu.memory).unwrap());
        let row = shared.claim(std::process::id()).unwrap();
        let facts = Facts { context: 2, torch: 30, physical: 12, returning: 3, ..Facts::default() };
        assert_eq!(Facts::of(facts.words()), facts);
        row.publish(&facts);
        let rows = shared.rows().unwrap();
        let (peers, _) = remotes(&rows, u64::MAX);
        let peer = peers.iter().find(|r| r.row == row.id()).unwrap();
        assert_eq!(peer.physical, 12);
        assert_eq!(peer.held(), 32, "admission includes the reserved envelope");
        assert_eq!(held(&rows), 32, "status.free must continue counting promises");
    }

    fn physical_fixture(name: &str) -> (Arc<Shared>, Arc<Row>) {
        let gpu = GpuId { uuid: format!("unit-{}-{name}", std::process::id()), model: "host".into(), memory: 64 << 20 };
        let shared = Arc::new(Shared::adopt(open_named(&gpu.uuid).unwrap(), &gpu, gpu.memory).unwrap());
        let row = Arc::new(shared.claim(std::process::id()).unwrap());
        (shared, row)
    }

    fn publish_physical(row: &Row, physical: u64, promised: u64) {
        let physical_epoch = row.physical_epoch();
        row.publish(&Facts { physical, physical_epoch, torch: promised, ..Facts::default() });
    }

    #[test]
    fn admission_excludes_independent_descriptions_and_local_threads() {
        let (shared, _row) = physical_fixture("admission-gate");
        let other = shared.descriptor().unwrap();
        let gate = shared.admission().unwrap();
        assert!(shared.admission.try_lock().is_none(), "the OFD lock alone cannot exclude threads on one description");
        assert!(!os::lock(&other, ADMISSION, 8, false).unwrap(), "independent descriptions contend on the kernel lock");
        assert_eq!(gate.rows().unwrap().len(), 1);
        drop(gate);
        assert!(os::lock(&other, ADMISSION, 8, false).unwrap());
        os::unlock(&other, ADMISSION, 8).unwrap();
    }

    #[test]
    fn stable_physical_facts_allow_promises_to_change_without_inventing_device_usage() {
        let (shared, row) = physical_fixture("stable-physical");
        publish_physical(&row, 4, 4);
        let before = shared.physical_sample().unwrap();
        let gate = shared.admission().unwrap();
        publish_physical(&row, 4, 12);
        let rows = gate.rows().unwrap();
        assert_eq!(before.known(&rows, u64::MAX), 4);
        assert_eq!(held(&rows), 12);
    }

    #[test]
    fn a_free_before_counter_publication_cannot_hide_foreign_bytes() {
        let (shared, row) = physical_fixture("free-observation");
        publish_physical(&row, 8, 8);
        let before = shared.physical_sample().unwrap();
        let freeing = row.physical_change();
        // The driver has returned eight bytes, but the row still conservatively charges them.
        let rows = shared.rows().unwrap();
        assert_eq!(before.known(&rows, u64::MAX), 0);
        assert_eq!(held(&rows), 8);
        let during = shared.physical_sample().unwrap();
        publish_physical(&row, 0, 8);
        freeing.finish();
        assert_eq!(during.known(&shared.rows().unwrap(), u64::MAX), 0);
        let after = shared.physical_sample().unwrap();
        assert_eq!(after.known(&shared.rows().unwrap(), u64::MAX), 0);
    }

    #[test]
    fn deferred_callbacks_and_overlapping_changes_stay_dirty_until_tagged_publication() {
        let (shared, row) = physical_fixture("deferred-physical");
        publish_physical(&row, 4, 12);
        let before = shared.physical_sample().unwrap();
        let a = {
            // Markers must work when native callbacks cannot acquire either ordinary lock.
            let _map = shared.map.write();
            let _latch = row.latch.lock();
            row.physical_change()
        };
        let stale_epoch = row.physical_epoch();
        let b = row.physical_change();
        row.publish(&Facts { physical: 12, physical_epoch: stale_epoch, torch: 12, ..Facts::default() });
        a.finish();
        assert_eq!(before.known(&shared.rows().unwrap(), u64::MAX), 0);
        b.finish();
        let stale = shared.physical_sample().unwrap();
        assert_eq!(stale.known(&shared.rows().unwrap(), u64::MAX), 0, "inactive is not proof that a stale publication is current");
        publish_physical(&row, 12, 12);
        let current = shared.physical_sample().unwrap();
        assert_eq!(current.known(&shared.rows().unwrap(), u64::MAX), 12);
    }

    #[test]
    fn physical_aba_and_abandoned_operations_never_validate_an_old_sample() {
        let (shared, row) = physical_fixture("physical-aba");
        publish_physical(&row, 4, 4);
        let before = shared.physical_sample().unwrap();
        for physical in [8, 4] {
            let change = row.physical_change();
            publish_physical(&row, physical, 8);
            change.finish();
        }
        assert_eq!(before.known(&shared.rows().unwrap(), u64::MAX), 0, "equal counts do not make an overlapping driver sample consistent");
        let current = shared.physical_sample().unwrap();
        assert_eq!(current.known(&shared.rows().unwrap(), u64::MAX), 4);
        let uncertain = row.physical_change();
        drop(uncertain);
        publish_physical(&row, 4, 8);
        let abandoned = shared.physical_sample().unwrap();
        assert_eq!(abandoned.known(&shared.rows().unwrap(), u64::MAX), 0, "dropping an uncompleted marker cannot clear uncertain ownership");
        assert_eq!(held(&shared.rows().unwrap()), 8);
    }

    #[test]
    fn reused_rows_and_custody_transfers_invalidate_prior_physical_identity() {
        let (shared, row) = physical_fixture("physical-generation");
        // Keep the table generation open while its source row is retired and reused.
        let custody = Arc::new(shared.claim(std::process::id()).unwrap());
        publish_physical(&row, 8, 8);
        let before = shared.physical_sample().unwrap();
        let old_at = row.at;
        let source_change = row.physical_change();
        let custody_change = custody.physical_change();
        {
            let _gate = shared.admission().unwrap();
            custody.set_orphans(8); // Destination charge precedes source release.
            publish_physical(&row, 0, 0);
            source_change.finish();
            custody_change.finish();
        }
        assert_eq!(before.known(&shared.rows().unwrap(), u64::MAX), 0);
        let transferred = shared.physical_sample().unwrap();
        assert_eq!(transferred.known(&shared.rows().unwrap(), u64::MAX), 8);
        drop(row);
        let replacement = shared.claim(std::process::id()).unwrap();
        assert_eq!(replacement.at, old_at);
        publish_physical(&replacement, 8, 8);
        assert_eq!(before.known(&shared.rows().unwrap(), u64::MAX), 0, "same offset and count are not the same owner");
        assert_eq!(transferred.known(&shared.rows().unwrap(), u64::MAX), 8, "only the unchanged custody owner remains attributable");
    }

}

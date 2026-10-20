//! Shared placement facts and handle custody. No eviction policy or CUDA calls live here.
//!
//! A named table survives a TensorD restart. Each executor holds its row's OFD lock on a
//! private file description, so even SIGKILL drops the claim. Snapshot copies are atomics:
//! an interrupted publisher leaves the preceding complete copy readable.

use std::collections::{BTreeMap, HashMap};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
#[cfg(target_os = "linux")]
use std::os::fd::FromRawFd;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{fence, AtomicBool, AtomicU32, AtomicU64, Ordering::*};
use std::sync::{Arc, Weak};
use std::thread::JoinHandle;

use parking_lot::{Condvar, Mutex};
use platform::shm::{self, Region};

use crate::plan::Class;
use crate::{Error, Result};
use tensorfs::format::ObjectRef;
use tensorfs::io::{
    source::{IoTally, Item, Mode, Source},
    Dst, ItemSource, ObjectRange,
};

pub const ROWS: usize = 64;
const RUNNING: usize = 8;
const CLASSES: usize = 7;
const WORDS: usize = CLASSES + 2 + RUNNING * 2;
const KEYS: usize = 256;
const KEY_WORDS: usize = 24;
const MAGIC: u64 = u64::from_le_bytes(*b"vslim001");
const LIVE: u64 = 1;
const REAPING: u64 = 2;

#[repr(C, align(64))]
struct Header {
    magic: AtomicU64,
    capacity: AtomicU64,
    changed: AtomicU32,
}
#[repr(C, align(64))]
struct Slot {
    state: AtomicU64,
    pid: AtomicU32,
    bell: AtomicU32,
    seq: AtomicU64,
    copies: [[AtomicU64; WORDS]; 2],
    asked: [AtomicU64; CLASSES],
    handled: [AtomicU64; CLASSES],
    wakeups: AtomicU64,
    progress: AtomicU64,
    // Protocol extension in the old Slot tail padding. Existing field offsets
    // and the mapped table size stay unchanged. Signals name an owner generation.
    cache_release: AtomicU64,
    cache_inflight: AtomicU64,
}
#[repr(C)]
struct Peak {
    state: AtomicU64,
    key: [AtomicU64; KEY_WORDS],
    bytes: AtomicU64,
}
#[repr(C)]
struct Shared {
    header: Header,
    rows: [Slot; ROWS],
    peaks: [Peak; KEYS],
}

/// A stable row reference; a reused slot never answers an old generation's ask.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Owner {
    pub row: u32,
    pub generation: u64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Running {
    pub id: u64,
    /// Largest remaining working set, excluding the separately counted context and pins.
    pub remaining: Option<u64>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Snapshot {
    /// Derived by plan::Table; transport only, never a second set of accounting counters.
    pub classes: [u64; CLASSES],
    pub context: u64,
    pub requests: Vec<Running>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Live,
    Dead,
    Reaping,
}
#[derive(Clone, Debug)]
pub struct RowStatus {
    pub owner: Owner,
    pub pid: u32,
    pub state: State,
    pub snapshot: Snapshot,
    pub progress: u64,
    pub wakeups: u64,
}
#[derive(Clone, Debug)]
pub struct Status {
    pub capacity: u64,
    pub rows: Vec<RowStatus>,
    pub peaks: Vec<(String, u64)>,
    pub kept: Vec<(PageKey, u64)>,
    pub ram_bytes: u64,
    /// Live private-host reservations, including allocation still in progress.
    pub private_host_bytes: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ask {
    pub ceiling: Class,
    pub bytes: u64,
    pub sequence: u64,
}

fn io(what: &str, error: impl std::fmt::Display) -> Error {
    Error::Io(format!("{what}: {error}"))
}
fn last(what: &str) -> Error {
    io(what, std::io::Error::last_os_error())
}
fn words(text: &str) -> Result<[u64; KEY_WORDS]> {
    if text.len() > KEY_WORDS * 8 || text.contains('\0') {
        return Err(Error::Invalid(
            "ledger key is too long or contains NUL".into(),
        ));
    }
    let mut out = [0; KEY_WORDS];
    for (word, bytes) in out.iter_mut().zip(text.as_bytes().chunks(8)) {
        let mut padded = [0; 8];
        for (dst, src) in padded.iter_mut().zip(bytes) {
            *dst = *src;
        }
        *word = u64::from_le_bytes(padded);
    }
    Ok(out)
}
fn text(words: &[u64]) -> String {
    let bytes: Vec<_> = words
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .take_while(|b| *b != 0)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}
fn sum(values: impl IntoIterator<Item = u64>) -> u64 {
    values.into_iter().fold(0, u64::saturating_add)
}
#[cfg(target_os = "linux")]
fn wait(word: &AtomicU32, before: u32) {
    // SAFETY: the shared mapping owns this aligned atomic until the waiter returns.
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            word.as_ptr(),
            libc::FUTEX_WAIT,
            before,
            std::ptr::null::<libc::timespec>(),
        );
    }
}
#[cfg(target_os = "linux")]
fn wake(word: &AtomicU32) {
    // SAFETY: FUTEX_WAKE does not access memory beyond this atomic word.
    unsafe {
        libc::syscall(libc::SYS_futex, word.as_ptr(), libc::FUTEX_WAKE, i32::MAX);
    }
}
#[cfg(not(target_os = "linux"))]
fn wait(_: &AtomicU32, _: u32) {}
#[cfg(not(target_os = "linux"))]
fn wake(_: &AtomicU32) {}
fn eventfd() -> Result<OwnedFd> {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: eventfd has only scalar arguments and returns a fresh descriptor.
        let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
        if fd < 0 {
            return Err(last("create ledger eventfd"));
        }
        // SAFETY: owned descriptor from the successful call above.
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }
    #[cfg(not(target_os = "linux"))]
    Err(Error::Unsupported("varena ledger requires Linux".into()))
}
fn signal(fd: BorrowedFd<'_>) {
    let one = 1u64.to_ne_bytes();
    // SAFETY: eight readable bytes and a live eventfd. EAGAIN means it is already signalled.
    unsafe {
        libc::write(fd.as_raw_fd(), one.as_ptr().cast(), one.len());
    }
}

struct Table {
    file: File,
    identity: (u64, u64),
    base: *mut Shared,
}
// SAFETY: only atomics are shared; references cannot outlive this mapping's Arc.
unsafe impl Send for Table {}
unsafe impl Sync for Table {}
impl Drop for Table {
    fn drop(&mut self) {
        // SAFETY: this is the mapping created by Table::map, with no remaining users.
        unsafe {
            libc::munmap(self.base.cast(), size_of::<Shared>());
        }
    }
}
impl Table {
    fn map(file: File) -> Result<Self> {
        let metadata = file.metadata().map_err(|e| io("stat ledger", e))?;
        if metadata.len() != size_of::<Shared>() as u64 {
            return Err(Error::Invalid(
                "ledger size differs from this protocol".into(),
            ));
        }
        // SAFETY: a correctly sized shared file, with atomically accessed fields.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size_of::<Shared>(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return Err(last("map ledger"));
        }
        Ok(Self {
            file,
            identity: (metadata.dev(), metadata.ino()),
            base: base.cast(),
        })
    }
    fn shared(&self) -> &Shared {
        // SAFETY: Table owns a valid aligned mapping for its entire lifetime.
        unsafe { &*self.base }
    }
    fn slot(&self, owner: Owner) -> Result<&Slot> {
        let slot = self
            .shared()
            .rows
            .get(owner.row as usize)
            .ok_or_else(|| Error::Invalid("ledger row out of range".into()))?;
        if slot.state.load(Acquire) >> 2 != owner.generation {
            return Err(Error::Closed("ledger row was reused".into()));
        }
        Ok(slot)
    }
    fn alive(&self, row: usize) -> Result<bool> {
        shm::region_holder(
            self.file.as_fd(),
            size_of::<Header>() + row * size_of::<Slot>(),
            size_of::<Slot>(),
        )
        .map(|v| v.is_some())
        .map_err(crate::os("probe ledger claim"))
    }
    fn changed(&self) {
        let changed = &self.shared().header.changed;
        changed.fetch_add(1, Release);
        wake(changed);
    }
    fn signal(&self, owner: Owner) -> Result<()> {
        let slot = self.slot(owner)?;
        slot.bell.fetch_add(1, Release);
        wake(&slot.bell);
        self.changed();
        Ok(())
    }
    fn rows(&self) -> Result<Vec<RowStatus>> {
        let mut out = Vec::new();
        for (row, slot) in self.shared().rows.iter().enumerate() {
            loop {
                let before = slot.state.load(Acquire);
                if before & 3 == 0 {
                    break;
                }
                let state = if before & 3 == REAPING {
                    State::Reaping
                } else if self.alive(row)? {
                    State::Live
                } else {
                    State::Dead
                };
                let data = loop {
                    let seq = slot.seq.load(Acquire);
                    let data = slot
                        .copies
                        .get((seq & 1) as usize)
                        .ok_or_else(|| Error::Invalid("ledger latch index".into()))?
                        .each_ref()
                        .map(|v| v.load(Relaxed));
                    fence(Acquire);
                    if slot.seq.load(Relaxed) == seq {
                        break data;
                    }
                };
                let context = data.get(CLASSES).copied().unwrap_or(0);
                let count = data.get(CLASSES + 1).copied().unwrap_or(0) as usize;
                let requests = data
                    .get(CLASSES + 2..)
                    .unwrap_or(&[])
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .take(count.min(RUNNING))
                    .map(|[id, bytes]| Running {
                        id: *id,
                        remaining: (*bytes != u64::MAX).then_some(*bytes),
                    })
                    .collect();
                let snapshot = Snapshot {
                    classes: std::array::from_fn(|i| data.get(i).copied().unwrap_or(0)),
                    context,
                    requests,
                };
                let found = RowStatus {
                    owner: Owner {
                        row: row as u32,
                        generation: before >> 2,
                    },
                    pid: slot.pid.load(Relaxed),
                    state,
                    snapshot,
                    progress: slot.progress.load(Acquire),
                    wakeups: slot.wakeups.load(Acquire),
                };
                if slot.state.load(Acquire) == before {
                    out.push(found);
                    break;
                }
            }
        }
        Ok(out)
    }
}

struct Watcher {
    stop: Arc<AtomicBool>,
    table: Arc<Table>,
    join: Option<JoinHandle<()>>,
}
impl Drop for Watcher {
    fn drop(&mut self) {
        self.stop.store(true, Release);
        self.table.changed();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

/// Descriptors passed to one executor. The row claim is made by that executor, not TensorD.
/// TensorD must drop its copy after descriptor transfer, especially the service channel.
pub struct Grant {
    pub table: OwnedFd,
    pub events: OwnedFd,
    pub channel: OwnedFd,
    pub pid: u32,
}

/// One GPU table. It does not choose victims: the owning executor applies plan::decide.
pub struct Ledger {
    table: Arc<Table>,
    events: Arc<OwnedFd>,
    _watcher: Option<Watcher>,
    custody: Arc<Mutex<Custody>>,
    host: Arc<HostCache>,
    observations: Mutex<HashMap<Owner, (u64, u64)>>,
    admission_releases: Mutex<HashMap<Owner, (u64, Snapshot)>>,
}
impl Ledger {
    /// Open the machine/user's existing ledger, or initialize it under the file's OFD lock.
    pub fn create(gpu_uuid: &str, capacity: u64) -> Result<Self> {
        Self::create_with_host(gpu_uuid, capacity, Arc::new(HostCache::default()))
    }
    /// TensorD supplies one host cache for all of its device ledgers. CUDA page
    /// custody and the shared per-device placement table remain independent.
    pub fn create_with_host(gpu_uuid: &str, capacity: u64, host: Arc<HostCache>) -> Result<Self> {
        host.check_process()?;
        if gpu_uuid.is_empty()
            || gpu_uuid.len() > 96
            || !gpu_uuid
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_'))
        {
            return Err(Error::Invalid("GPU UUID is not a safe ledger name".into()));
        }
        // SAFETY: geteuid reads the process's numeric user ID.
        let uid = unsafe { libc::geteuid() };
        let name = format!("/dev/shm/varena-{gpu_uuid}-{uid}");
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(name)
            .map_err(|e| io("open named ledger", e))?;
        let meta = file.metadata().map_err(|e| io("stat named ledger", e))?;
        if meta.uid() != uid || !meta.is_file() {
            return Err(Error::Invalid("ledger owner or file type differs".into()));
        }
        shm::region_lock(file.as_fd(), 0, size_of::<Header>(), Region::Write, true)
            .map_err(crate::os("initialize ledger"))?;
        let fresh = file
            .metadata()
            .map_err(|e| io("restat named ledger", e))?
            .len()
            == 0;
        if fresh {
            file.set_len(size_of::<Shared>() as u64)
                .map_err(|e| io("size ledger", e))?;
        }
        let table = Arc::new(Table::map(file)?);
        if fresh {
            table.shared().header.capacity.store(capacity, Relaxed);
            table.shared().header.magic.store(MAGIC, Release);
        }
        if table.shared().header.magic.load(Acquire) != MAGIC {
            return Err(Error::Invalid("ledger protocol differs".into()));
        }
        let current = table.shared().header.capacity.load(Acquire);
        if current != capacity {
            let live = table.rows()?.iter().any(|row| row.state == State::Live);
            if live && capacity < current {
                return Err(Error::Invalid(
                    "cannot lower a GPU cap while an executor owns a row".into(),
                ));
            }
            // A child may have discovered less CUDA-usable memory than NVML's
            // inventory total. A coordinator reopening beside live rows retains
            // that correction; raising a cap still requires a drained generation.
            if !live {
                table.shared().header.capacity.store(capacity, Release);
                // Cuts and resident targets were measured under the old limit.
                // The first request under the new cap must establish its own peak.
                for peak in &table.shared().peaks {
                    peak.state.store(0, Release);
                }
                table.changed();
            }
        }
        shm::region_lock(
            table.file.as_fd(),
            0,
            size_of::<Header>(),
            Region::Unlock,
            false,
        )
        .map_err(crate::os("unlock ledger initialization"))?;
        let events = Arc::new(eventfd()?);
        let stop = Arc::new(AtomicBool::new(false));
        let (watched, stopping, bell) = (table.clone(), stop.clone(), events.clone());
        let mut before = table.shared().header.changed.load(Acquire);
        let join = std::thread::Builder::new()
            .name("varena-events".into())
            .spawn(move || {
                while !stopping.load(Acquire) {
                    wait(&watched.shared().header.changed, before);
                    let after = watched.shared().header.changed.load(Acquire);
                    if after != before {
                        signal(bell.as_fd());
                        before = after;
                    }
                }
            })
            .map_err(|e| io("start ledger event bridge", e))?;
        let out = Self {
            table: table.clone(),
            events,
            _watcher: Some(Watcher {
                stop,
                table,
                join: Some(join),
            }),
            custody: Arc::default(),
            host,
            observations: Mutex::default(),
            admission_releases: Mutex::default(),
        };
        // A new coordinator has no process history for a previous daemon's rows.
        out.reap_dead()?;
        Ok(out)
    }
    /// Adopt the descriptors in a grant. No named-file lookup or CUDA initialization occurs.
    pub fn adopt(table: BorrowedFd<'_>, events: BorrowedFd<'_>) -> Result<Self> {
        let table = Arc::new(Table::map(File::from(
            shm::reopen(table, true).map_err(crate::os("reopen ledger"))?,
        ))?);
        if table.shared().header.magic.load(Acquire) != MAGIC {
            return Err(Error::Invalid("ledger protocol differs".into()));
        }
        let events = Arc::new(
            events
                .try_clone_to_owned()
                .map_err(|e| io("duplicate ledger eventfd", e))?,
        );
        Ok(Self {
            table,
            events,
            _watcher: None,
            custody: Arc::default(),
            host: Arc::new(HostCache::default()),
            observations: Mutex::default(),
            admission_releases: Mutex::default(),
        })
    }
    pub fn fd(&self) -> BorrowedFd<'_> {
        self.table.file.as_fd()
    }
    pub fn events(&self) -> BorrowedFd<'_> {
        self.events.as_fd()
    }
    pub fn capacity(&self) -> u64 {
        self.table.shared().header.capacity.load(Acquire)
    }
    /// Reduce the shared ceiling to this device's usable total. CUDA is queried by
    /// the child before this call; no per-process reserve or second authority is stored.
    pub fn limit_capacity(&self, capacity: u64) -> Result<u64> {
        if capacity == 0 {
            return Err(Error::Invalid("GPU usable capacity is zero".into()));
        }
        // Serialize against coordinator initialization, which may replace an idle
        // table's cap. Placement already owns its row before calling this method.
        let lock = shm::reopen(self.fd(), true).map_err(crate::os("open capacity lock"))?;
        shm::region_lock(lock.as_fd(), 0, size_of::<Header>(), Region::Write, true)
            .map_err(crate::os("lock capacity"))?;
        let before = self
            .table
            .shared()
            .header
            .capacity
            .fetch_min(capacity, AcqRel);
        drop(lock);
        if capacity < before {
            // Owner workers sleep on their row bells; TensorD's eventfd bridge
            // sleeps on the header. Wake both without manufacturing reclaim asks.
            for slot in &self.table.shared().rows {
                if slot.state.load(Acquire) & 3 == LIVE {
                    slot.bell.fetch_add(1, Release);
                    wake(&slot.bell);
                }
            }
            self.table.changed();
        }
        Ok(before.min(capacity))
    }
    /// A private rank channel created before the rank process exists. Its first bind pins
    /// one live row generation; no later message may name a different owner.
    pub fn grant_unbound(&self) -> Result<Grant> {
        self.grant(0)
    }
    pub fn grant(&self, pid: u32) -> Result<Grant> {
        self.host.check_process()?;
        let (server, client) = UnixStream::pair().map_err(|e| io("create grant channel", e))?;
        let (table, custody, host) = (self.table.clone(), self.custody.clone(), self.host.clone());
        std::thread::Builder::new()
            .name("varena-custody".into())
            .spawn(move || serve(server, pid, table, custody, host))
            .map_err(|e| io("start grant service", e))?;
        Ok(Grant {
            table: self
                .fd()
                .try_clone_to_owned()
                .map_err(|e| io("duplicate table", e))?,
            events: self
                .events()
                .try_clone_to_owned()
                .map_err(|e| io("duplicate eventfd", e))?,
            channel: client.into(),
            pid,
        })
    }
    pub fn claim(&self, pid: u32) -> Result<Row> {
        if pid != std::process::id() {
            return Err(Error::Invalid(
                "only the executor may claim its own row".into(),
            ));
        }
        // Serialize admission with an idle-only cap change. This private open
        // file description releases the initialization lock on every return.
        let initialization = shm::reopen(self.fd(), true).map_err(crate::os("open claim gate"))?;
        shm::region_lock(
            initialization.as_fd(),
            0,
            size_of::<Header>(),
            Region::Write,
            true,
        )
        .map_err(crate::os("lock claim gate"))?;
        if self
            .table
            .rows()?
            .iter()
            .any(|row| row.pid == pid && row.state == State::Live)
        {
            return Err(Error::Invalid("process already owns a ledger row".into()));
        }
        let lock = shm::reopen(self.fd(), true).map_err(crate::os("open private ledger claim"))?;
        for (row, slot) in self.table.shared().rows.iter().enumerate() {
            if slot.state.load(Acquire) & 3 != 0 {
                continue;
            }
            let start = size_of::<Header>() + row * size_of::<Slot>();
            if !shm::region_lock(lock.as_fd(), start, size_of::<Slot>(), Region::Write, false)
                .map_err(crate::os("claim row"))?
            {
                continue;
            }
            let before = slot.state.load(Acquire);
            if before & 3 != 0 {
                shm::region_lock(
                    lock.as_fd(),
                    start,
                    size_of::<Slot>(),
                    Region::Unlock,
                    false,
                )
                .map_err(crate::os("unclaim row"))?;
                continue;
            }
            slot.pid.store(pid, Relaxed);
            slot.seq.store(0, Relaxed);
            for word in slot
                .copies
                .iter()
                .flatten()
                .chain(&slot.asked)
                .chain(&slot.handled)
            {
                word.store(0, Relaxed);
            }
            slot.progress.store(0, Relaxed);
            slot.wakeups.store(0, Relaxed);
            slot.cache_release.store(0, Relaxed);
            slot.cache_inflight.store(0, Relaxed);
            let owner = Owner {
                row: row as u32,
                generation: (before >> 2) + 1,
            };
            slot.state.store(owner.generation << 2 | LIVE, Release);
            self.table.changed();
            return Ok(Row {
                table: self.table.clone(),
                owner,
                _claim: Some(lock),
                write: Mutex::new(()),
            });
        }
        Err(Error::Invalid("ledger has no free row".into()))
    }
    pub fn status(&self) -> Result<Status> {
        self.host.check_process()?;
        let kept = self
            .custody
            .lock()
            .pages
            .iter()
            .filter(|(_, page)| queued(page))
            .map(|(key, page)| (*key, sum(page.spans.iter().copied())))
            .collect();
        let (ram_bytes, private_host_bytes) = {
            let mut state = self.host.state.lock();
            state.prune_private();
            (state.retained_bytes(), state.private_bytes())
        };
        Ok(Status {
            capacity: self.capacity(),
            rows: self.table.rows()?,
            peaks: self.peaks(),
            kept,
            ram_bytes,
            private_host_bytes,
        })
    }
    /// Reap only kernel-released claims. Their custody remains available until consumed or
    /// explicitly dropped; no GPU context is needed to close the queued handle containers.
    pub fn reap(&self, pid: u32) -> Result<()> {
        self.host.check_process()?;
        for row in self.table.rows()?.into_iter().filter(|r| r.pid == pid) {
            if row.state == State::Live {
                return Err(Error::Invalid("cannot reap a live row".into()));
            }
            self.reap_owner(row.owner)?;
        }
        Ok(())
    }
    fn reap_dead(&self) -> Result<()> {
        self.host.check_process()?;
        for row in self.table.rows()? {
            if row.state != State::Live {
                self.reap_owner(row.owner)?;
            }
        }
        Ok(())
    }
    /// The claim lock, not PID existence or a prior snapshot, proves death. A
    /// fresh description conflicts even with a live claim in this same process.
    fn reap_owner(&self, owner: Owner) -> Result<bool> {
        let claim = shm::reopen(self.fd(), true).map_err(crate::os("open reap claim"))?;
        let start = size_of::<Header>() + owner.row as usize * size_of::<Slot>();
        if !shm::region_lock(
            claim.as_fd(),
            start,
            size_of::<Slot>(),
            Region::Write,
            false,
        )
        .map_err(crate::os("lock reap claim"))?
        {
            return Ok(false);
        }
        // No claimant can appear while this guard lives. A concurrent reaper
        // may already have freed/reused the slot after our original snapshot.
        let slot = match self.table.slot(owner) {
            Ok(slot) => slot,
            Err(Error::Closed(_)) => return Ok(false),
            Err(error) => return Err(error),
        };
        if slot.state.load(Acquire) & 3 == 0 {
            return Ok(false);
        }
        slot.state.store(owner.generation << 2 | REAPING, Release);
        self.custody.lock().orphan(owner);
        for held in self.host.state.lock().ram.values_mut() {
            held.observers.retain(|subscriber, _| {
                subscriber.table != self.table.identity || subscriber.owner != owner
            });
        }
        slot.state.store(owner.generation << 2, Release);
        drop(claim); // make the freed row claimable before notifying admission
        self.table.changed();
        Ok(true)
    }
    /// An exact-class ask; plan::decide, not the ledger, orders victims and ceilings.
    pub fn ask(&self, owner: Owner, bytes: u64, ceiling: Class) -> Result<Ask> {
        let slot = self.table.slot(owner)?;
        if slot.state.load(Acquire) & 3 != LIVE || !self.table.alive(owner.row as usize)? {
            return Err(Error::Closed("asked row is not live".into()));
        }
        let counter = slot
            .asked
            .get(ceiling.as_usize())
            .ok_or_else(|| Error::Invalid("invalid class ceiling".into()))?;
        let sequence = counter.fetch_add(bytes, AcqRel).saturating_add(bytes);
        slot.bell.fetch_add(1, Release);
        wake(&slot.bell);
        Ok(Ask {
            bytes,
            ceiling,
            sequence,
        })
    }
    /// Request only idle cache callbacks, never a byte eviction. The pending
    /// generation remains visible until the owner finishes its callbacks/events.
    fn release_idle_caches(&self, owner: Owner) -> Result<()> {
        let slot = self.table.slot(owner)?;
        if slot.state.load(Acquire) & 3 != LIVE || !self.table.alive(owner.row as usize)? {
            return Err(Error::Closed("cache-release owner is not live".into()));
        }
        if slot.cache_inflight.load(Acquire) == owner.generation {
            return Ok(());
        }
        if slot.cache_release.fetch_max(owner.generation, AcqRel) < owner.generation {
            slot.bell.fetch_add(1, Release);
            wake(&slot.bell);
        }
        Ok(())
    }
    pub fn answered(&self, owner: Owner, ask: Ask) -> Result<bool> {
        let slot = match self.table.slot(owner) {
            Ok(slot) => slot,
            Err(Error::Closed(_)) => return Ok(true),
            Err(error) => return Err(error),
        };
        Ok(slot
            .handled
            .get(ask.ceiling.as_usize())
            .is_some_and(|v| v.load(Acquire) >= ask.sequence)
            || !self.table.alive(owner.row as usize)?)
    }
    pub fn wait_answer(&self, owner: Owner, ask: Ask, stop: &AtomicBool) -> Result<bool> {
        loop {
            let before = self.table.shared().header.changed.load(Acquire);
            if stop.load(Acquire) {
                return Ok(false);
            }
            if self.answered(owner, ask)? {
                return Ok(true);
            }
            wait(&self.table.shared().header.changed, before);
        }
    }
    pub fn measured(&self, package: &str, shape: &str) -> Option<u64> {
        let key = format!("{package}|{shape}");
        self.peaks()
            .into_iter()
            .find_map(|(k, v)| (k == key).then_some(v))
    }
    /// Only a victim's own wakeups count as observations; repeated status reads cannot kill it.
    pub fn stalled(&self) -> Result<Vec<u32>> {
        let mut observations = self.observations.lock();
        let mut stalled = Vec::new();
        for row in self.table.rows()? {
            let slot = self.table.slot(row.owner)?;
            let pending = slot.cache_release.load(Acquire) == row.owner.generation
                || slot.cache_inflight.load(Acquire) == row.owner.generation
                || slot
                    .asked
                    .iter()
                    .zip(&slot.handled)
                    .any(|(a, h)| a.load(Acquire) > h.load(Acquire));
            if row.state == State::Live && pending {
                match observations.get(&row.owner).copied() {
                    Some((progress, wakeups)) if row.progress == progress => {
                        if row.wakeups >= wakeups.saturating_add(2) {
                            stalled.push(row.pid);
                        }
                    }
                    _ => {
                        observations.insert(row.owner, (row.progress, row.wakeups));
                    }
                }
            } else {
                observations.remove(&row.owner);
            }
        }
        Ok(stalled)
    }
    /// Active remaining work stays conservative. An observed peak is not an idle
    /// request's irreducible minimum: its own placement/cuts enforce actual capacity.
    pub fn fits(&self, package: &str, shape: &str) -> Result<bool> {
        self.reap_dead()?;
        let rows = self.table.rows()?;
        // Keep the release stamp and its pending-generation publication atomic
        // to other admission checks. This gate never waits for owner callbacks.
        let mut attempted = self.admission_releases.lock();
        attempted.retain(|owner, _| {
            rows.iter()
                .any(|row| row.owner == *owner && row.state == State::Live)
        });
        if rows.iter().any(|r| r.state != State::Live) {
            return Ok(false);
        }
        for row in &rows {
            let slot = self.table.slot(row.owner)?;
            if slot.cache_release.load(Acquire) == row.owner.generation
                || slot.cache_inflight.load(Acquire) == row.owner.generation
            {
                return Ok(false);
            }
        }
        let running: Vec<_> = rows.iter().flat_map(|r| &r.snapshot.requests).collect();
        if running.iter().any(|r| r.remaining.is_none()) {
            return Ok(false);
        }
        let key = format!("{package}|{shape}");
        let peaks = self.peaks();
        let peak = if shape.is_empty() {
            let prefix = format!("{package}|");
            peaks
                .into_iter()
                .filter(|(k, _)| k.starts_with(&prefix))
                .map(|(_, v)| v)
                .max()
        } else {
            peaks.into_iter().find_map(|(k, v)| (k == key).then_some(v))
        };
        let Some(peak) = peak else {
            return Ok(running.is_empty());
        };
        let fixed = sum(rows.iter().map(|r| {
            r.snapshot.context.saturating_add(
                r.snapshot
                    .classes
                    .get(Class::Pinned.as_usize())
                    .copied()
                    .unwrap_or(0),
            )
        }));
        let fits = fixed
            .saturating_add(sum(running.iter().filter_map(|r| r.remaining)))
            .saturating_add(peak)
            <= self.capacity();
        let mut releasing = false;
        if !fits {
            // A learned peak already includes the model's resident weights. An
            // idle graph can keep those same weights pinned, so ask its owner to
            // drop discretionary cache refs before retrying this conservative sum.
            let release: Vec<_> = {
                rows.iter()
                    .filter(|row| {
                        row.snapshot.requests.is_empty()
                            && row.snapshot.classes[Class::Pinned.as_usize()] != 0
                    })
                    .filter_map(|row| {
                        let stamp = (row.progress, row.snapshot.clone());
                        if attempted.get(&row.owner) == Some(&stamp) {
                            None
                        } else {
                            attempted.insert(row.owner, stamp);
                            Some(row.owner)
                        }
                    })
                    .collect()
            };
            for owner in release {
                releasing = true;
                match self.release_idle_caches(owner) {
                    Ok(()) | Err(Error::Closed(_)) => {}
                    Err(error) => return Err(error),
                }
            }
        }
        // Let an otherwise idle executor adapt after its discretionary caches
        // acknowledged the attempt, even if they could free nothing. Never turn
        // that historical peak into a hard minimum or erase actual context/pins.
        Ok(fits || (running.is_empty() && !releasing))
    }
    pub fn record_peak(&self, package: &str, shape: &str, bytes: u64) -> Result<()> {
        let key = words(&format!("{package}|{shape}"))?;
        let lock = shm::reopen(self.fd(), true).map_err(crate::os("open peak lock"))?;
        shm::region_lock(lock.as_fd(), 0, size_of::<Header>(), Region::Write, true)
            .map_err(crate::os("lock peaks"))?;
        // Only the peak writer holds this lock. A dead writer's incomplete slot is reusable.
        for peak in &self.table.shared().peaks {
            if peak.state.load(Acquire) == 1 {
                peak.state.store(0, Release);
            }
        }
        for peak in &self.table.shared().peaks {
            let state = peak.state.load(Acquire);
            if state == 2 && peak.key.each_ref().map(|v| v.load(Relaxed)) == key {
                peak.bytes.fetch_max(bytes, Release);
                self.table.changed();
                return Ok(());
            }
            if state == 0 && peak.state.compare_exchange(0, 1, AcqRel, Acquire).is_ok() {
                for (word, value) in peak.key.iter().zip(key) {
                    word.store(value, Relaxed);
                }
                peak.bytes.store(bytes, Relaxed);
                peak.state.store(2, Release);
                self.table.changed();
                return Ok(());
            }
        }
        Err(Error::Invalid("ledger peak table is full".into()))
    }
    pub fn peaks(&self) -> Vec<(String, u64)> {
        self.table
            .shared()
            .peaks
            .iter()
            .filter(|p| p.state.load(Acquire) == 2)
            .map(|p| {
                (
                    text(&p.key.each_ref().map(|v| v.load(Relaxed))),
                    p.bytes.load(Acquire),
                )
            })
            .collect()
    }
    /// Register a RAM layout once. A library worker reads its disjoint regions using TensorFS;
    /// a consumer sees only completed regions and otherwise immediately falls back to disk.
    pub fn cache_ram(
        &self,
        layout: [u8; 32],
        source: Arc<Source>,
        regions: Vec<RamRegion>,
    ) -> Result<bool> {
        self.host.check_process()?;
        register_ram(&self.table, &self.host, layout, source, regions, None)
    }
    pub fn ram_status(&self, layout: [u8; 32]) -> Option<RamStatus> {
        self.host.check_process().ok()?;
        ram_status(&self.host.state.lock(), layout)
    }
    /// Shared native allocations not yet reflected in live host memory. Executor
    /// forecasts are counted by TensorD; committed buffers are already physical.
    pub fn pending_host_bytes(&self) -> Result<u64> {
        self.host.check_process()?;
        let mut state = self.host.state.lock();
        state.prune_private();
        state
            .pending_host()
            .ok_or_else(|| Error::Invalid("pending host bytes overflow".into()))
    }
    /// Tell the cache about a starting executor before admitting it. Unknown peaks
    /// suppress optional warming and shed every idle/queued region; measured peaks
    /// shed only until their room is available. The caller always admits demand.
    pub fn prepare_host(&self, reservation: u64, need: Option<u64>) {
        if self.host.check_process().is_err() {
            return;
        }
        {
            let mut state = self.host.state.lock();
            state.claims.insert(reservation, need);
        }
        while !self
            .host
            .can_warm(platform::contain::host_memory::read().available, 0)
        {
            if !self.release_unused_ram() {
                break;
            }
        }
    }
    pub fn finish_host(&self, reservation: u64) {
        if self.host.check_process().is_err() {
            return;
        }
        let mut state = self.host.state.lock();
        state.claims.remove(&reservation);
    }
    /// Explicit layout invalidation, only when every region is idle. Pressure
    /// instead calls release_unused_ram, whose unit is a single region.
    pub fn release_ram(&self, layout: [u8; 32]) -> bool {
        if self.host.check_process().is_err() {
            return false;
        }
        let removed = {
            let mut state = self.host.state.lock();
            let idle = state.ram.get_mut(&layout).is_some_and(|held| {
                held.slots
                    .iter_mut()
                    .all(|slot| slot.buffer.is_none() || slot.idle())
            });
            if idle {
                state.ram.remove(&layout)
            } else {
                None
            }
        };
        if let Some(held) = removed {
            for slot in &held.slots {
                if let Some(ram) = &slot.buffer {
                    ram.cancelled.store(true, Release);
                }
            }
            self.host.wake_warm();
            notify_owners(held.observers);
            self.table.changed();
            true
        } else {
            false
        }
    }
    /// Reclaim one least-recently-used idle region. A live lease or actual byte
    /// fill protects only that region; other regions of its layout can give way.
    pub fn release_unused_ram(&self) -> bool {
        if self.host.check_process().is_err() {
            return false;
        }
        self.host.release_unused_ram(&self.table, u64::MAX)
    }
    pub fn drop_kept(&self) {
        self.custody.lock().pages.clear();
        self.table.changed();
    }
}

pub struct Row {
    table: Arc<Table>,
    owner: Owner,
    _claim: Option<OwnedFd>,
    write: Mutex<()>,
}

/// Keeps the generation's pending evidence visible through callbacks and CUDA
/// completion. A failed callback is still one attempt, never an automatic retry.
pub(crate) struct CacheRelease<'a>(&'a Row);
impl Drop for CacheRelease<'_> {
    fn drop(&mut self) {
        let owner = self.0.owner;
        if let Ok(slot) = self.0.table.slot(owner) {
            let _ = slot
                .cache_release
                .compare_exchange(owner.generation, 0, AcqRel, Acquire);
            let _ = slot
                .cache_inflight
                .compare_exchange(owner.generation, 0, AcqRel, Acquire);
            self.0.table.changed();
        }
    }
}

impl Row {
    pub(crate) fn cache_release(&self) -> Result<Option<CacheRelease<'_>>> {
        let slot = self.table.slot(self.owner)?;
        if slot.cache_release.load(Acquire) != self.owner.generation {
            return Ok(None);
        }
        if slot
            .cache_inflight
            .compare_exchange(0, self.owner.generation, AcqRel, Acquire)
            .is_err()
        {
            return Ok(None);
        }
        slot.wakeups.fetch_add(1, Release);
        self.table.changed();
        Ok(Some(CacheRelease(self)))
    }
    pub fn owner(&self) -> Owner {
        self.owner
    }
    pub fn publish(&self, snapshot: &Snapshot, progress: u64) -> Result<()> {
        if snapshot.requests.len() > RUNNING {
            return Err(Error::Invalid(
                "too many running requests in ledger row".into(),
            ));
        }
        let _write = self.write.lock();
        let slot = self.table.slot(self.owner)?;
        let mut data = [0; WORDS];
        for (dst, value) in data.iter_mut().zip(snapshot.classes) {
            *dst = value;
        }
        if let Some(dst) = data.get_mut(CLASSES) {
            *dst = snapshot.context;
        }
        if let Some(dst) = data.get_mut(CLASSES + 1) {
            *dst = snapshot.requests.len() as u64;
        }
        if let Some(requests) = data.get_mut(CLASSES + 2..) {
            for ([id, left], request) in requests
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(&snapshot.requests)
            {
                *id = request.id;
                *left = request.remaining.unwrap_or(u64::MAX);
            }
        }
        let seq = slot.seq.load(Relaxed);
        slot.seq.store(seq.wrapping_add(1), Release);
        fence(Release);
        if let Some(copy) = slot.copies.first() {
            for (dst, value) in copy.iter().zip(data) {
                dst.store(value, Relaxed);
            }
        }
        slot.seq.store(seq.wrapping_add(2), Release);
        fence(Release);
        if let Some(copy) = slot.copies.get(1) {
            for (dst, value) in copy.iter().zip(data) {
                dst.store(value, Relaxed);
            }
        }
        slot.progress.store(progress, Release);
        self.table.changed();
        Ok(())
    }
    /// Wait for one changed bell. Non-ask wakes let the owner poll completed GPU events
    /// and pin releases on the same worker; `seen` must persist across calls.
    pub fn wait_event(&self, seen: &mut u32, stop: &AtomicBool) -> Result<Option<Ask>> {
        let slot = self.table.slot(self.owner)?;
        if stop.load(Acquire) {
            return Ok(None);
        }
        let current = slot.bell.load(Acquire);
        if current == *seen {
            wait(&slot.bell, current);
        }
        *seen = slot.bell.load(Acquire);
        if stop.load(Acquire) {
            return Ok(None);
        }
        for ceiling in Class::ALL {
            let index = ceiling.as_usize();
            let (asked, handled) = slot
                .asked
                .get(index)
                .zip(slot.handled.get(index))
                .ok_or_else(|| Error::Invalid("invalid class ceiling".into()))?;
            let (sequence, done) = (asked.load(Acquire), handled.load(Acquire));
            if sequence > done {
                slot.wakeups.fetch_add(1, Release);
                self.table.changed();
                return Ok(Some(Ask {
                    ceiling,
                    bytes: sequence - done,
                    sequence,
                }));
            }
        }
        Ok(None)
    }
    pub fn wait_ask(&self, stop: &AtomicBool) -> Result<Option<Ask>> {
        let mut seen = 0;
        loop {
            if let Some(ask) = self.wait_event(&mut seen, stop)? {
                return Ok(Some(ask));
            }
            if stop.load(Acquire) {
                return Ok(None);
            }
        }
    }
    /// Wake a blocked reclaim worker after its close flag changes.
    pub fn wake(&self) -> Result<()> {
        self.table.signal(self.owner)
    }
    pub fn ack(&self, ask: Ask) -> Result<()> {
        let slot = self.table.slot(self.owner)?;
        slot.handled
            .get(ask.ceiling.as_usize())
            .ok_or_else(|| Error::Invalid("invalid class ceiling".into()))?
            .fetch_max(ask.sequence, Release);
        self.table.changed();
        Ok(())
    }
}
impl Drop for Row {
    fn drop(&mut self) {
        drop(self._claim.take());
        if let Ok(slot) = self.table.slot(self.owner) {
            // A normal close needs no waitpid/reap of this still-live process.
            // Preserve the generation so a stale grant cannot target its next row.
            let _ = slot.state.compare_exchange(
                self.owner.generation << 2 | LIVE,
                self.owner.generation << 2,
                AcqRel,
                Acquire,
            );
        }
        self.table.changed();
    }
}

/// Fingerprint includes the TensorFS layout, encoding, transform, rank shard and GPU UUID.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PageKey {
    pub fingerprint: [u8; 32],
    pub page: u64,
}
struct Kept {
    owner: Option<Owner>,
    spans: Vec<u64>,
    container: UnixStream,
}
#[derive(Clone, Debug)]
pub struct RamRegion {
    pub offset: u64,
    pub len: u64,
    pub items: Vec<Item>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamStatus {
    pub generation: u64,
    pub filling: bool,
    pub ready: Vec<bool>,
}
fn validate_regions(regions: &[RamRegion]) -> Result<()> {
    let mut end = 0u64;
    for region in regions {
        if region.len == 0 || region.offset < end {
            return Err(Error::Invalid(
                "RAM regions must be ordered and nonoverlapping".into(),
            ));
        }
        end = region
            .offset
            .checked_add(region.len)
            .ok_or_else(|| Error::Invalid("RAM region overflow".into()))?;
        let mut intervals = Vec::new();
        for item in &region.items {
            let item_end = item
                .offset
                .checked_add(item.len)
                .ok_or_else(|| Error::Invalid("RAM item overflow".into()))?;
            if item.offset < region.offset || item_end > end {
                return Err(Error::Invalid("RAM item exceeds its region".into()));
            }
            match &item.source {
                ItemSource::Object(range)
                    if range.len != item.len
                        || range
                            .off
                            .checked_add(range.len)
                            .is_none_or(|n| n > range.obj.length) =>
                {
                    return Err(Error::Invalid("RAM object range exceeds source".into()))
                }
                ItemSource::Inline(bytes) if bytes.len() as u64 != item.len => {
                    return Err(Error::Invalid("RAM inline length differs".into()))
                }
                _ => (),
            }
            intervals.push((item.offset, item_end));
        }
        intervals.sort_unstable();
        let mut previous = region.offset;
        for (start, finish) in intervals {
            if start < previous {
                return Err(Error::Invalid("RAM items overlap".into()));
            }
            previous = finish;
        }
    }
    if regions.is_empty() {
        return Err(Error::Invalid("empty RAM layout".into()));
    }
    Ok(())
}
struct Ram {
    file: File,
    base: usize,
    len: usize,
    region: RamRegion,
    ready: AtomicBool,
    pending: AtomicBool,
    filling: AtomicBool,
    cancelled: AtomicBool,
}
// SAFETY: one worker owns writes, then ready publishes an immutable region.
// Readers receive read-only descriptions and hold independent region leases.
unsafe impl Send for Ram {}
unsafe impl Sync for Ram {}
impl Ram {
    fn new(region: RamRegion) -> Result<Self> {
        let len = usize::try_from(region.len)
            .map_err(|_| Error::Invalid("RAM region too large".into()))?;
        let file = File::from(
            shm::memfd_named("varena-ram-region").map_err(crate::os("create RAM region"))?,
        );
        file.set_len(region.len)
            .map_err(|e| io("size RAM region", e))?;
        // SAFETY: a sized, nonempty region; this is its sole writable mapping.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return Err(last("map RAM region"));
        }
        let ram = Self {
            file,
            base: base as usize,
            len,
            region,
            ready: AtomicBool::new(false),
            pending: AtomicBool::new(true),
            filling: AtomicBool::new(false),
            cancelled: AtomicBool::new(false),
        };
        shm::seal(ram.file.as_fd()).map_err(crate::os("seal RAM region"))?;
        Ok(ram)
    }
}
impl Drop for Ram {
    fn drop(&mut self) {
        // SAFETY: active writers/readers retain their own Arc or file/lease owners.
        unsafe {
            libc::munmap(self.base as *mut libc::c_void, self.len);
        }
    }
}
struct RamSlot {
    buffer: Option<Arc<Ram>>,
    readers: Vec<UnixStream>,
    used: u64,
}
impl RamSlot {
    fn missing(&self) -> bool {
        self.buffer
            .as_ref()
            .is_none_or(|ram| !ram.pending.load(Acquire) && !ram.ready.load(Acquire))
    }
    fn idle(&mut self) -> bool {
        self.readers
            .retain(|reader| !platform::handles::hung_up(reader));
        self.readers.is_empty()
            && self
                .buffer
                .as_ref()
                .is_some_and(|ram| !ram.filling.load(Acquire))
    }
}
struct RamHeld {
    regions: Vec<RamRegion>,
    slots: Vec<RamSlot>,
    observers: BTreeMap<Subscriber, Weak<Table>>,
    identity: u64,
    generation: u64,
}
impl RamHeld {
    fn prune(&mut self) {
        self.observers.retain(|subscriber, table| {
            table.upgrade().is_some_and(|table| {
                table
                    .slot(subscriber.owner)
                    .is_ok_and(|slot| slot.state.load(Acquire) & 3 == LIVE)
            })
        });
    }
}
#[derive(Default)]
struct Custody {
    pages: BTreeMap<PageKey, Kept>,
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Subscriber {
    table: (u64, u64),
    owner: Owner,
    token: u64,
}
/// A machine service's explicitly owned host tier. It is shared by that service's
/// per-GPU ledgers, never by a static registry, and drops with its last owner.
pub struct HostCache {
    process: u32,
    state: Mutex<HostState>,
    warming: Mutex<bool>,
    next_warm: Condvar,
}

/// A private allocation's capability, retained by its actual buffer/consumer owners.
/// Closing a request, row or plane must not release it before those owners finish.
pub struct HostReservation {
    lease: UnixStream,
    committed: AtomicBool,
}

impl HostReservation {
    /// The full reserved span is now physically pinned and reflected in live host
    /// memory. Keep its lease, but do not charge these bytes twice as pending.
    pub fn commit(&self) -> Result<()> {
        if !self.committed.load(Acquire) {
            let mut lease = &self.lease;
            lease
                .write_all(&[1])
                .map_err(|error| io("commit private host allocation", error))?;
            self.committed.store(true, Release);
        }
        Ok(())
    }
}

struct PrivateHost {
    bytes: u64,
    lease: UnixStream,
    pending: bool,
}

impl PrivateHost {
    fn live(&mut self) -> bool {
        if platform::handles::hung_up(&self.lease) {
            return false;
        }
        if self.pending {
            let mut committed = [0_u8];
            // Leave commit readiness for the dedicated service observer, even
            // if a budget query runs before that observer begins its wait.
            let read = unsafe {
                libc::recv(
                    self.lease.as_raw_fd(),
                    committed.as_mut_ptr().cast(),
                    committed.len(),
                    libc::MSG_PEEK | libc::MSG_DONTWAIT,
                )
            };
            if read == 0 {
                return false;
            }
            if read == 1 && committed == [1] {
                self.pending = false;
            }
            // A malformed marker or interrupted probe keeps the reservation
            // conservative until close.
        }
        true
    }
}

enum HostAdmission {
    Granted(UnixStream, u64, UnixStream),
    Reclaimed(u64),
    Refused,
}

#[derive(Default)]
struct HostState {
    ram: BTreeMap<[u8; 32], RamHeld>,
    clock: u64,
    claims: BTreeMap<u64, Option<u64>>,
    private: Vec<PrivateHost>,
}
impl HostState {
    fn prune_private(&mut self) {
        self.private.retain_mut(PrivateHost::live);
    }

    fn private_bytes(&self) -> u64 {
        sum(self.private.iter().map(|allocation| allocation.bytes))
    }

    fn pending_private(&self) -> Option<u64> {
        self.private
            .iter()
            .filter(|allocation| allocation.pending)
            .try_fold(0_u64, |sum, allocation| sum.checked_add(allocation.bytes))
    }

    fn pending_host(&self) -> Option<u64> {
        self.ram
            .values()
            .flat_map(|held| &held.slots)
            .filter_map(|slot| slot.buffer.as_ref())
            .filter(|ram| ram.filling.load(Acquire))
            .try_fold(self.pending_private()?, |sum, ram| {
                sum.checked_add(ram.len as u64)
            })
    }

    fn can_reserve(&self, available: i64, bytes: u64) -> bool {
        // Executor forecasts suppress optional warming, including unknown peaks.
        // Exact demand is bounded by current room and in-flight native allocations.
        self.pending_host()
            .and_then(|pending| pending.checked_add(bytes))
            .zip(u64::try_from(available).ok())
            .is_some_and(|(need, available)| need <= available)
    }

    fn retained_bytes(&self) -> u64 {
        sum(self
            .ram
            .values()
            .flat_map(|held| &held.slots)
            .filter_map(|slot| slot.buffer.as_ref())
            .filter(|ram| ram.ready.load(Acquire) || ram.filling.load(Acquire))
            .map(|ram| ram.len as u64))
    }
    fn can_warm(&self, available: i64, bytes: u64) -> bool {
        let Some(claimed) = self
            .claims
            .values()
            .try_fold(0_u64, |sum, need| need.map(|need| sum.saturating_add(need)))
        else {
            return false;
        };
        self.pending_private()
            .and_then(|pending| pending.checked_add(bytes)?.checked_add(claimed))
            .zip(u64::try_from(available).ok())
            .is_some_and(|(need, available)| need <= available)
    }
}
impl Default for HostCache {
    fn default() -> Self {
        Self {
            process: std::process::id(),
            state: Mutex::default(),
            warming: Mutex::new(false),
            next_warm: Condvar::new(),
        }
    }
}
impl HostCache {
    fn reserve_private(
        &self,
        table: &Table,
        bytes: u64,
        worker_room: i64,
        through: Option<u64>,
    ) -> Result<HostAdmission> {
        self.check_process()?;
        if bytes == 0 {
            return Err(Error::Invalid("empty private host reservation".into()));
        }
        if worker_room < 0 {
            return Ok(HostAdmission::Refused);
        }
        let through = {
            let mut state = self.state.lock();
            state.prune_private();
            // Couple this live proc/cgroup metadata reading with pending claims.
            // No array allocation, page I/O, CUDA or descriptor exchange holds this lock.
            let broker_room = platform::contain::host_memory::read().available;
            if broker_room < 0 {
                return Ok(HostAdmission::Refused);
            }
            let available = worker_room.min(broker_room);
            if state.can_reserve(available, bytes) {
                let pending = state
                    .pending_host()
                    .ok_or_else(|| Error::Invalid("private host reservation overflow".into()))?;
                let (server, client) =
                    UnixStream::pair().map_err(|error| io("private host lease", error))?;
                server
                    .set_nonblocking(true)
                    .map_err(|error| io("nonblocking private host lease", error))?;
                let observer = server
                    .try_clone()
                    .map_err(|error| io("observe private host lease", error))?;
                state.private.push(PrivateHost {
                    bytes,
                    lease: server,
                    pending: true,
                });
                return Ok(HostAdmission::Granted(client, pending, observer));
            }
            through.unwrap_or(state.clock).min(state.clock)
        };
        // Each retry removes one actually idle region from the original fixed
        // set. New warming cannot create an unbounded admission/reclaim loop.
        Ok(if self.release_unused_ram(table, through) {
            HostAdmission::Reclaimed(through)
        } else {
            HostAdmission::Refused
        })
    }

    fn release_unused_ram(&self, table: &Table, through: u64) -> bool {
        let mut state = self.state.lock();
        let retained_before = state.retained_bytes();
        let victim = state
            .ram
            .iter_mut()
            .flat_map(|(key, held)| {
                held.slots
                    .iter_mut()
                    .enumerate()
                    .filter_map(move |(index, slot)| {
                        (slot.used <= through && slot.idle()).then_some((*key, index, slot.used))
                    })
            })
            .min_by_key(|(_, _, used)| *used)
            .map(|(key, index, _)| (key, index));
        let Some((key, index)) = victim else {
            return false;
        };
        state.clock = state.clock.saturating_add(1);
        let generation = state.clock;
        let Some(held) = state.ram.get_mut(&key) else {
            return false;
        };
        let removed = held
            .slots
            .get_mut(index)
            .and_then(|slot| slot.buffer.take());
        let ready_bytes = removed.as_ref().map_or(0, |ram| {
            if ram.ready.load(Acquire) {
                ram.len as u64
            } else {
                0
            }
        });
        if let Some(ram) = &removed {
            ram.cancelled.store(true, Release);
        }
        held.generation = generation;
        let observers = held.observers.clone();
        if held.slots.iter().all(|slot| slot.buffer.is_none()) {
            state.ram.remove(&key);
        }
        let retained_after = state.retained_bytes();
        drop(state);
        // Actually unmap/free this region outside the table lock.
        drop(removed);
        // These are charged cache spans, not process RSS. Reader leases and
        // in-flight fills were excluded by idle() before choosing this region.
        tracing::info!(target: "varena::host", region = index, ready_bytes,
            retained_before, retained_after, eligibility = "unleased_not_filling",
            "released idle RAM cache region");
        self.wake_warm();
        notify_owners(observers);
        table.changed();
        true
    }
    fn wake_warm(&self) {
        let _active = self.warming.lock();
        self.next_warm.notify_all();
    }
    fn can_warm(&self, available: i64, bytes: u64) -> bool {
        let mut state = self.state.lock();
        state.prune_private();
        let live = platform::contain::host_memory::read().available;
        state.can_warm(available.min(live), bytes)
    }
    fn begin_fill(
        &self,
        layout: [u8; 32],
        identity: u64,
        index: usize,
        ram: &Arc<Ram>,
        available: i64,
    ) -> bool {
        let mut state = self.state.lock();
        state.prune_private();
        let live = platform::contain::host_memory::read().available;
        if !state.can_warm(available.min(live), ram.len as u64) || ram.cancelled.load(Acquire) {
            return false;
        }
        let current = state
            .ram
            .get(&layout)
            .filter(|held| held.identity == identity)
            .and_then(|held| held.slots.get(index))
            .and_then(|slot| slot.buffer.as_ref())
            .is_some_and(|current| Arc::ptr_eq(current, ram));
        if current {
            ram.filling.store(true, Release);
        }
        current
    }
    fn check_process(&self) -> Result<()> {
        if self.process != std::process::id() {
            return Err(Error::Invalid(
                "host cache must be created after fork; adopt a grant instead".into(),
            ));
        }
        Ok(())
    }
    fn warm_slot(&self, ram: &Ram) -> Option<WarmSlot<'_>> {
        let mut active = self.warming.lock();
        while *active {
            if ram.cancelled.load(Acquire) {
                return None;
            }
            self.next_warm.wait(&mut active);
        }
        if ram.cancelled.load(Acquire) {
            return None;
        }
        *active = true;
        Some(WarmSlot(self))
    }
}
/// Serialize optional cache fills so they cannot spend the same observed host
/// headroom twice. The coordination mutex is released before any byte I/O.
struct WarmSlot<'a>(&'a HostCache);
impl Drop for WarmSlot<'_> {
    fn drop(&mut self) {
        *self.0.warming.lock() = false;
        self.0.next_warm.notify_all();
    }
}
impl Custody {
    fn orphan(&mut self, owner: Owner) {
        for page in self.pages.values_mut().filter(|p| p.owner == Some(owner)) {
            page.owner = None;
        }
    }
}

fn queued(page: &Kept) -> bool {
    let mut bytes = 0i32;
    // SAFETY: FIONREAD writes one integer and never consumes SCM_RIGHTS. Each chunk is
    // queued as the platform's one-byte descriptor frame. A partial drain is not a page.
    (unsafe { libc::ioctl(page.container.as_raw_fd(), libc::FIONREAD, &mut bytes) }) == 0
        && usize::try_from(bytes).ok() == Some(page.spans.len())
}

fn orphan(table: &Table, page: &Kept) -> bool {
    page.owner.is_none_or(|owner| match table.slot(owner) {
        Err(Error::Closed(_)) => true,
        Ok(_) => table.alive(owner.row as usize).is_ok_and(|live| !live),
        Err(_) => false,
    })
}

fn notify_owners(observers: BTreeMap<Subscriber, Weak<Table>>) {
    for (subscriber, table) in observers {
        if let Some(table) = table.upgrade() {
            let _ = table.signal(subscriber.owner);
        }
    }
}
fn ram_status(state: &HostState, layout: [u8; 32]) -> Option<RamStatus> {
    let held = state.ram.get(&layout)?;
    Some(RamStatus {
        generation: held.generation,
        filling: held.slots.iter().any(|slot| {
            slot.buffer
                .as_ref()
                .is_some_and(|ram| ram.pending.load(Acquire))
        }),
        ready: held
            .slots
            .iter()
            .map(|slot| {
                slot.buffer
                    .as_ref()
                    .is_some_and(|ram| ram.ready.load(Acquire))
            })
            .collect(),
    })
}

/// Reuse the reserve request's dedicated service thread through the result's
/// lifetime. The server endpoint never keeps a client alias alive. Observe one
/// commit, then only hangup, so an unread marker cannot create a wake loop.
fn watch_private_host(lease: UnixStream, host: &HostCache, table: &Table) -> Result<()> {
    #[cfg(target_os = "linux")]
    let closed = libc::POLLHUP | libc::POLLERR | libc::POLLRDHUP;
    #[cfg(not(target_os = "linux"))]
    let closed = libc::POLLHUP | libc::POLLERR;
    let mut descriptor = libc::pollfd {
        fd: lease.as_raw_fd(),
        events: libc::POLLIN | closed,
        revents: 0,
    };
    loop {
        // SAFETY: a live, independently owned descriptor; an event-driven wait,
        // never under the host-state lock and with no timeout or status polling.
        let ready = unsafe { libc::poll(&mut descriptor, 1, -1) };
        if ready < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(io("wait for private host lease", error));
        }
        if descriptor.revents & libc::POLLNVAL != 0 {
            return Err(Error::Io("private host lease descriptor closed".into()));
        }
        let ended = descriptor.revents & closed != 0;
        if ended || descriptor.revents & libc::POLLIN != 0 {
            host.state.lock().prune_private();
            host.wake_warm();
            table.changed();
        }
        if ended {
            return Ok(());
        }
        descriptor.events = closed;
    }
}

struct RamFill {
    ram: Arc<Ram>,
    index: usize,
    table: Arc<Table>,
    host: Arc<HostCache>,
    layout: [u8; 32],
    identity: u64,
    complete: bool,
}
impl Drop for RamFill {
    fn drop(&mut self) {
        let available = platform::contain::host_memory::read().available;
        let observers = {
            let mut state = self.host.state.lock();
            let room = state.can_warm(available, 0);
            self.ram.ready.store(self.complete, Release);
            self.ram.filling.store(false, Release);
            self.ram.pending.store(false, Release);
            state.clock = state.clock.saturating_add(1);
            let generation = state.clock;
            let observers = state
                .ram
                .get_mut(&self.layout)
                .filter(|held| held.identity == self.identity)
                .filter(|held| {
                    held.slots
                        .get(self.index)
                        .and_then(|slot| slot.buffer.as_ref())
                        .is_some_and(|ram| Arc::ptr_eq(ram, &self.ram))
                })
                .map(|held| {
                    if !room {
                        if let Some(slot) = held.slots.get_mut(self.index) {
                            if slot.idle() {
                                if let Some(ram) = slot.buffer.take() {
                                    ram.cancelled.store(true, Release);
                                }
                            }
                        }
                    }
                    held.generation = generation;
                    held.observers.clone()
                })
                .unwrap_or_default();
            if state
                .ram
                .get(&self.layout)
                .is_some_and(|held| held.slots.iter().all(|slot| slot.buffer.is_none()))
            {
                state.ram.remove(&self.layout);
            }
            observers
        };
        notify_owners(observers);
        self.table.changed();
    }
}
fn same_regions(a: &[RamRegion], b: &[RamRegion]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(a, b)| a.offset == b.offset && a.len == b.len)
}
fn register_ram(
    table: &Arc<Table>,
    host: &Arc<HostCache>,
    layout: [u8; 32],
    source: Arc<Source>,
    regions: Vec<RamRegion>,
    observer: Option<(Owner, u64)>,
) -> Result<bool> {
    validate_regions(&regions)?;
    let missing = {
        let state = host.state.lock();
        if let Some(held) = state.ram.get(&layout) {
            if !same_regions(&held.regions, &regions) {
                return Err(Error::Invalid(
                    "RAM fingerprint has a different layout".into(),
                ));
            }
            held.slots
                .iter()
                .enumerate()
                .filter_map(|(index, slot)| slot.missing().then_some(index))
                .collect::<Vec<_>>()
        } else {
            (0..regions.len()).collect()
        }
    };
    // Creating/sealing mappings never holds the host table lock.
    let created = missing
        .into_iter()
        .map(|index| {
            let region = regions
                .get(index)
                .ok_or_else(|| Error::Invalid("RAM region absent".into()))?
                .clone();
            Ram::new(region).map(|ram| (index, Arc::new(ram)))
        })
        .collect::<Result<Vec<_>>>()?;
    let (identity, work, retired) = {
        let mut state = host.state.lock();
        if !state.ram.contains_key(&layout) {
            state.clock = state
                .clock
                .checked_add(1)
                .ok_or_else(|| Error::Invalid("RAM generation exhausted".into()))?;
            let identity = state.clock;
            state.ram.insert(
                layout,
                RamHeld {
                    regions: regions.clone(),
                    slots: regions
                        .iter()
                        .map(|_| RamSlot {
                            buffer: None,
                            readers: vec![],
                            used: identity,
                        })
                        .collect(),
                    observers: BTreeMap::new(),
                    identity,
                    generation: identity,
                },
            );
        }
        let held = state
            .ram
            .get_mut(&layout)
            .ok_or_else(|| Error::Invalid("RAM layout absent".into()))?;
        if !same_regions(&held.regions, &regions) {
            return Err(Error::Invalid(
                "RAM fingerprint has a different layout".into(),
            ));
        }
        held.prune();
        if let Some((owner, token)) = observer {
            held.observers.insert(
                Subscriber {
                    table: table.identity,
                    owner,
                    token,
                },
                Arc::downgrade(table),
            );
        }
        let mut work = Vec::new();
        let mut retired = Vec::new();
        for (index, ram) in created {
            if let Some(slot) = held.slots.get_mut(index).filter(|slot| slot.missing()) {
                if let Some(old) = slot.buffer.replace(ram.clone()) {
                    retired.push(old);
                }
                work.push((index, ram));
            }
        }
        (held.identity, work, retired)
    };
    drop(retired);
    if work.is_empty() {
        return Ok(false);
    }
    let (table, host) = (table.clone(), host.clone());
    let (failed_host, failed_table) = (host.clone(), table.clone());
    let started = std::thread::Builder::new()
        .name("varena-ram".into())
        .spawn(move || {
            let active = work
                .iter()
                .find(|(_, ram)| !ram.cancelled.load(Acquire))
                .map(|(_, ram)| ram.clone());
            let permit = active.as_ref().and_then(|ram| host.warm_slot(ram));
            for (index, ram) in work {
                let mut finished = RamFill {
                    ram: ram.clone(),
                    index,
                    table: table.clone(),
                    host: host.clone(),
                    layout,
                    identity,
                    complete: false,
                };
                let available = platform::contain::host_memory::read().available;
                if permit.is_none() || !host.begin_fill(layout, identity, index, &ram, available) {
                    continue;
                }
                // SAFETY: this worker exclusively writes this independent region mapping.
                let result = unsafe {
                    source.read_items(
                        &ram.region.items,
                        ram.region.offset,
                        Dst::Host {
                            ptr: ram.base as *mut u8,
                            len: ram.len,
                        },
                        1,
                        Mode::Cache,
                        &IoTally::default(),
                    )
                };
                match result {
                    Ok(_) => finished.complete = true,
                    Err(error) => tracing::warn!(%error, "RAM region remains on disk"),
                }
            }
        });
    if let Err(error) = started {
        let mut removed = Vec::new();
        let observers = {
            let mut state = failed_host.state.lock();
            state
                .ram
                .get_mut(&layout)
                .filter(|held| held.identity == identity)
                .map(|held| {
                    for slot in &mut held.slots {
                        if slot.buffer.as_ref().is_some_and(|ram| {
                            ram.pending.load(Acquire) && !ram.filling.load(Acquire)
                        }) {
                            if let Some(ram) = slot.buffer.take() {
                                ram.cancelled.store(true, Release);
                                removed.push(ram);
                            }
                        }
                    }
                    held.observers.clone()
                })
                .unwrap_or_default()
        };
        drop(removed);
        notify_owners(observers);
        failed_table.changed();
        return Err(io("start RAM fill", error));
    }
    Ok(true)
}

/// The grant channel carries only placement custody, never model headers or data bytes.
/// Packet: op/u8, row/u32, generation/u64, fingerprint/32 bytes, page/u64, count/u32,
/// spans/count*u64; an offer then sends one SCM_RIGHTS container. Imports consume it once.
fn serve(
    mut stream: UnixStream,
    pid: u32,
    table: Arc<Table>,
    custody: Arc<Mutex<Custody>>,
    host: Arc<HostCache>,
) {
    let result = (|| -> Result<()> {
        let mut bound = None;
        loop {
            let mut header = [0u8; 57];
            match stream.read_exact(&mut header) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
                Err(e) => return Err(io("read custody command", e)),
            }
            let read = |start: usize, len: usize| {
                header
                    .get(start..start + len)
                    .ok_or_else(|| Error::Invalid("custody header".into()))
            };
            let owner = Owner {
                row: u32::from_le_bytes(
                    read(1, 4)?
                        .try_into()
                        .map_err(|_| Error::Invalid("row".into()))?,
                ),
                generation: u64::from_le_bytes(
                    read(5, 8)?
                        .try_into()
                        .map_err(|_| Error::Invalid("generation".into()))?,
                ),
            };
            let slot = table.slot(owner)?;
            if (pid != 0 && slot.pid.load(Acquire) != pid)
                || !table.alive(owner.row as usize)?
                || bound.is_some_and(|previous| previous != owner)
            {
                return Err(Error::Invalid("grant used by another row owner".into()));
            }
            if pid == 0 && bound.is_none() && header.first() != Some(&0) {
                return Err(Error::Invalid("rank grant must bind before use".into()));
            }
            bound = Some(owner);
            let key = PageKey {
                fingerprint: read(13, 32)?
                    .try_into()
                    .map_err(|_| Error::Invalid("fingerprint".into()))?,
                page: u64::from_le_bytes(
                    read(45, 8)?
                        .try_into()
                        .map_err(|_| Error::Invalid("page".into()))?,
                ),
            };
            let count = u32::from_le_bytes(
                read(53, 4)?
                    .try_into()
                    .map_err(|_| Error::Invalid("count".into()))?,
            ) as usize;
            if count > 4096 {
                return Err(Error::Invalid("too many custody chunks".into()));
            }
            match header.first().copied() {
                Some(0) => {
                    stream
                        .write_all(&[1])
                        .map_err(|e| io("bind rank grant", e))?;
                }
                Some(1) => {
                    let mut spans = Vec::with_capacity(count);
                    for _ in 0..count {
                        let mut bytes = [0; 8];
                        stream
                            .read_exact(&mut bytes)
                            .map_err(|e| io("read chunk span", e))?;
                        spans.push(u64::from_le_bytes(bytes));
                    }
                    let container = platform::handles::recv_fd(&stream)
                        .map_err(crate::os("receive kept container"))?;
                    let mut held = custody.lock();
                    let replace = held
                        .pages
                        .get(&key)
                        .is_none_or(|p| !queued(p) || p.owner == Some(owner) || orphan(&table, p));
                    if replace {
                        held.pages.insert(
                            key,
                            Kept {
                                owner: Some(owner),
                                spans,
                                container: UnixStream::from(container),
                            },
                        );
                    }
                    stream
                        .write_all(&[u8::from(replace)])
                        .map_err(|e| io("acknowledge kept page", e))?;
                }
                Some(2) => {
                    let mut held = custody.lock();
                    let available = held.pages.get(&key).is_some_and(|p| orphan(&table, p));
                    let page = if available {
                        held.pages.remove(&key)
                    } else {
                        None
                    };
                    drop(held);
                    let mut handles = Vec::new();
                    if let Some(page) = &page {
                        page.container
                            .set_nonblocking(true)
                            .map_err(|e| io("read kept page without waiting", e))?;
                        for _ in &page.spans {
                            match platform::handles::recv_fd(&page.container) {
                                Ok(fd) => handles.push(fd),
                                Err(_) => {
                                    handles.clear();
                                    break;
                                }
                            }
                        }
                    }
                    let spans = page
                        .filter(|p| handles.len() == p.spans.len())
                        .map(|p| p.spans)
                        .unwrap_or_default();
                    stream
                        .write_all(&(spans.len() as u32).to_le_bytes())
                        .map_err(|e| io("send kept page size", e))?;
                    for span in spans {
                        stream
                            .write_all(&span.to_le_bytes())
                            .map_err(|e| io("send kept chunk span", e))?;
                    }
                    for handle in handles {
                        platform::handles::send_fd(&stream, &handle)
                            .map_err(crate::os("send kept handle"))?;
                    }
                }
                Some(3) => {
                    let mut held = custody.lock();
                    if held.pages.get(&key).is_some_and(|p| p.owner == Some(owner)) {
                        held.pages.remove(&key);
                    }
                    stream
                        .write_all(&[1])
                        .map_err(|e| io("acknowledge release", e))?;
                }
                Some(4) => {
                    let ready = {
                        let held = host.state.lock();
                        held.ram.get(&key.fingerprint).and_then(|held| {
                            let index = usize::try_from(key.page).ok()?;
                            let slot = held.slots.get(index)?;
                            let ram = slot.buffer.as_ref()?;
                            ram.ready
                                .load(Acquire)
                                .then(|| (ram.clone(), held.identity, index))
                        })
                    };
                    let ready = if let Some((ram, identity, index)) = ready {
                        let (ours, theirs) =
                            UnixStream::pair().map_err(|e| io("create RAM reader lease", e))?;
                        let file = shm::reopen(ram.file.as_fd(), false)
                            .map_err(crate::os("open read-only RAM"))?;
                        let mut state = host.state.lock();
                        state.clock = state.clock.saturating_add(1);
                        let used = state.clock;
                        state
                            .ram
                            .get_mut(&key.fingerprint)
                            .filter(|held| held.identity == identity)
                            .and_then(|held| held.slots.get_mut(index))
                            .filter(|slot| {
                                slot.buffer
                                    .as_ref()
                                    .is_some_and(|current| Arc::ptr_eq(current, &ram))
                            })
                            .map(|slot| {
                                slot.used = used;
                                slot.readers.push(ours);
                                (file, theirs, ram.len as u64)
                            })
                    } else {
                        None
                    };
                    if let Some((file, lease, len)) = ready {
                        stream
                            .write_all(&[1])
                            .and_then(|_| stream.write_all(&0_u64.to_le_bytes()))
                            .and_then(|_| stream.write_all(&len.to_le_bytes()))
                            .map_err(|e| io("send RAM region", e))?;
                        platform::handles::send_fd(&stream, &file)
                            .map_err(crate::os("send RAM file"))?;
                        platform::handles::send_fd(&stream, &lease)
                            .map_err(crate::os("send RAM lease"))?;
                    } else {
                        stream.write_all(&[0]).map_err(|e| io("send RAM miss", e))?;
                    }
                }
                Some(5) => {
                    let mut held = custody.lock();
                    let released = held
                        .pages
                        .get(&key)
                        .is_some_and(|page| orphan(&table, page));
                    if released {
                        held.pages.remove(&key);
                    }
                    stream
                        .write_all(&[u8::from(released)])
                        .map_err(|e| io("release orphan", e))?;
                }
                Some(6) => {
                    let held = custody.lock();
                    let pages: Vec<_> = held
                        .pages
                        .iter()
                        .filter(|(_, page)| orphan(&table, page) && queued(page))
                        .map(|(key, page)| (*key, sum(page.spans.iter().copied())))
                        .collect();
                    stream
                        .write_all(&(pages.len() as u32).to_le_bytes())
                        .map_err(|e| io("send orphan count", e))?;
                    for (key, bytes) in pages {
                        stream
                            .write_all(&key.fingerprint)
                            .and_then(|_| stream.write_all(&key.page.to_le_bytes()))
                            .and_then(|_| stream.write_all(&bytes.to_le_bytes()))
                            .map_err(|e| io("send orphan page", e))?;
                    }
                }
                Some(7) => {
                    let objects = (0..count)
                        .map(|_| read_object(&mut stream))
                        .collect::<Result<Vec<_>>>()?;
                    let regions = read_regions(&mut stream)?;
                    let files = objects
                        .into_iter()
                        .map(|object| {
                            let fd = platform::handles::recv_fd(&stream)
                                .map_err(crate::os("receive RAM source"))?;
                            // SAFETY: F_GETFL reads flags from this owned descriptor.
                            let flags = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFL) };
                            if flags < 0 {
                                return Err(last("inspect RAM source"));
                            }
                            if flags & libc::O_ACCMODE != libc::O_RDONLY {
                                return Err(Error::Invalid(
                                    "RAM source descriptor must be read-only".into(),
                                ));
                            }
                            let file = File::from(fd);
                            if file.metadata().map_err(|e| io("stat RAM source", e))?.len()
                                != object.length
                            {
                                return Err(Error::Invalid("RAM source length differs".into()));
                            }
                            Ok((object, file))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let accepted = register_ram(
                        &table,
                        &host,
                        key.fingerprint,
                        Arc::new(Source::files(files, true, 0)),
                        regions,
                        Some((owner, key.page)),
                    )?;
                    stream
                        .write_all(&[u8::from(accepted)])
                        .map_err(|e| io("ack RAM registration", e))?;
                }
                Some(8) => {
                    let status = ram_status(&host.state.lock(), key.fingerprint);
                    stream
                        .write_all(&[u8::from(status.is_some())])
                        .map_err(|e| io("send RAM status", e))?;
                    if let Some(status) = status {
                        stream
                            .write_all(&status.generation.to_le_bytes())
                            .and_then(|_| stream.write_all(&[u8::from(status.filling)]))
                            .and_then(|_| {
                                stream.write_all(&(status.ready.len() as u64).to_le_bytes())
                            })
                            .and_then(|_| {
                                stream.write_all(
                                    &status.ready.into_iter().map(u8::from).collect::<Vec<_>>(),
                                )
                            })
                            .map_err(|e| io("send RAM readiness", e))?;
                    }
                }
                Some(9) => {
                    let mut held = host.state.lock();
                    let remove = held.ram.get_mut(&key.fingerprint).is_some_and(|ram| {
                        ram.observers.remove(&Subscriber {
                            table: table.identity,
                            owner,
                            token: key.page,
                        });
                        ram.prune();
                        ram.observers.is_empty()
                    });
                    let removed = if remove {
                        held.ram.remove(&key.fingerprint)
                    } else {
                        None
                    };
                    drop(held);
                    if let Some(ram) = removed {
                        for slot in &ram.slots {
                            if let Some(buffer) = &slot.buffer {
                                buffer.cancelled.store(true, Release);
                            }
                        }
                    }
                    host.wake_warm();
                    stream
                        .write_all(&[1])
                        .map_err(|e| io("ack RAM unsubscribe", e))?;
                }
                Some(10) => {
                    let (server, client) =
                        UnixStream::pair().map_err(|e| io("fork grant channel", e))?;
                    let (table, custody, host, pid) = (
                        table.clone(),
                        custody.clone(),
                        host.clone(),
                        slot.pid.load(Acquire),
                    );
                    std::thread::Builder::new()
                        .name("varena-custody".into())
                        .spawn(move || serve(server, pid, table, custody, host))
                        .map_err(|e| io("start forked grant", e))?;
                    platform::handles::send_fd(&stream, &client)
                        .map_err(crate::os("send forked grant"))?;
                }
                Some(11) => {
                    if count > 1 {
                        return Err(Error::Invalid("private host admission phase".into()));
                    }
                    let mut room = [0; 8];
                    let mut through = [0; 8];
                    stream
                        .read_exact(&mut room)
                        .and_then(|_| stream.read_exact(&mut through))
                        .map_err(|error| io("read private host admission", error))?;
                    match host.reserve_private(
                        &table,
                        key.page,
                        i64::from_le_bytes(room),
                        (count == 1).then_some(u64::from_le_bytes(through)),
                    )? {
                        HostAdmission::Granted(lease, pending, observer) => {
                            // Publish the grant before handing its client to the
                            // worker; subsequent wakes name actual commit/free.
                            table.changed();
                            stream
                                .write_all(&[1])
                                .and_then(|_| stream.write_all(&pending.to_le_bytes()))
                                .map_err(|error| io("admit private host allocation", error))?;
                            platform::handles::send_fd(&stream, &lease)
                                .map_err(crate::os("send private host lease"))?;
                            drop(lease);
                            drop(stream);
                            // Only reserve_host's dedicated fork uses this command.
                            // Preserve the lease after its row/request has gone,
                            // and observe SIGKILL via the same kernel hangup.
                            return watch_private_host(observer, &host, &table);
                        }
                        HostAdmission::Reclaimed(through) => {
                            stream
                                .write_all(&[2])
                                .and_then(|_| stream.write_all(&through.to_le_bytes()))
                                .map_err(|error| io("refresh private host room", error))?;
                        }
                        HostAdmission::Refused => {
                            stream
                                .write_all(&[0])
                                .map_err(|error| io("refuse private host allocation", error))?;
                        }
                    }
                }
                _ => return Err(Error::Invalid("unknown custody command".into())),
            }
            table.changed();
        }
    })();
    if let Err(error) = result {
        tracing::warn!(%error, "varena custody channel ended");
    }
    table.changed();
}

/// Executor half of a grant. Operations are serialized, with no background CUDA work.
pub struct Channel(Mutex<UnixStream>);
impl Channel {
    /// Exact private bytes, before allocating/pinning them. The lease is a
    /// separate capability so a request/plane may close while its result lives.
    pub fn reserve_host(&self, owner: Owner, bytes: u64) -> Result<Option<Arc<HostReservation>>> {
        if bytes == 0 {
            return Err(Error::Invalid("empty private host reservation".into()));
        }
        // Admission failure or an older service cannot destroy the demand channel.
        let channel = self.fork(owner)?;
        let mut through = None;
        loop {
            let room = platform::contain::host_memory::read().available;
            if room < 0 {
                return Ok(None);
            }
            let mut stream = channel.0.lock();
            Self::header(
                &mut stream,
                11,
                owner,
                PageKey {
                    fingerprint: [0; 32],
                    page: bytes,
                },
                usize::from(through.is_some()),
            )?;
            stream
                .write_all(&room.to_le_bytes())
                .and_then(|_| stream.write_all(&through.unwrap_or(0_u64).to_le_bytes()))
                .map_err(|error| io("request private host room", error))?;
            let mut reply = [0];
            if let Err(error) = stream.read_exact(&mut reply) {
                if error.kind() == std::io::ErrorKind::UnexpectedEof {
                    return Ok(None);
                }
                return Err(io("read private host admission", error));
            }
            match reply {
                [0] => return Ok(None),
                [1] => {
                    let mut pending = [0; 8];
                    stream
                        .read_exact(&mut pending)
                        .map_err(|error| io("read private host pending bytes", error))?;
                    let lease = platform::handles::recv_fd(&stream)
                        .map_err(crate::os("receive private host lease"))?;
                    drop(stream);
                    // Re-read this worker's tightest cgroup after broker reclaim
                    // and before allocation. Commit cannot hide earlier pending
                    // allocations: their spans accompany this grant too.
                    let room = platform::contain::host_memory::read().available;
                    if bytes
                        .checked_add(u64::from_le_bytes(pending))
                        .zip(u64::try_from(room).ok())
                        .is_none_or(|(need, room)| need > room)
                    {
                        return Ok(None);
                    }
                    return Ok(Some(Arc::new(HostReservation {
                        lease: UnixStream::from(lease),
                        committed: AtomicBool::new(false),
                    })));
                }
                [2] => {
                    let mut changed = [0; 8];
                    stream
                        .read_exact(&mut changed)
                        .map_err(|error| io("read reclaimed host snapshot", error))?;
                    through = Some(u64::from_le_bytes(changed));
                    // Each retry follows an actual region removal, not a timer
                    // or a status poll, and uses the original fixed victim set.
                }
                _ => return Err(Error::Invalid("private host admission reply".into())),
            }
        }
    }

    /// A separate stream to the same service custody, including for adopted ledgers.
    /// Background cache registration cannot occupy the GPU demand stream.
    pub fn fork(&self, owner: Owner) -> Result<Self> {
        let mut stream = self.0.lock();
        Self::header(
            &mut stream,
            10,
            owner,
            PageKey {
                fingerprint: [0; 32],
                page: 0,
            },
            0,
        )?;
        let fd = platform::handles::recv_fd(&stream).map_err(crate::os("receive forked grant"))?;
        drop(stream);
        let channel = Self::new(fd);
        channel.bind(owner)?;
        Ok(channel)
    }
    pub fn new(fd: OwnedFd) -> Self {
        Self(Mutex::new(UnixStream::from(fd)))
    }
    pub fn bind(&self, owner: Owner) -> Result<()> {
        let mut stream = self.0.lock();
        Self::header(
            &mut stream,
            0,
            owner,
            PageKey {
                fingerprint: [0; 32],
                page: 0,
            },
            0,
        )?;
        let mut ack = [0];
        stream
            .read_exact(&mut ack)
            .map_err(|e| io("read rank binding", e))?;
        if ack != [1] {
            return Err(Error::Invalid("rank grant refused".into()));
        }
        Ok(())
    }
    fn header(
        stream: &mut UnixStream,
        op: u8,
        owner: Owner,
        key: PageKey,
        count: usize,
    ) -> Result<()> {
        stream
            .write_all(&[op])
            .and_then(|_| stream.write_all(&owner.row.to_le_bytes()))
            .and_then(|_| stream.write_all(&owner.generation.to_le_bytes()))
            .and_then(|_| stream.write_all(&key.fingerprint))
            .and_then(|_| stream.write_all(&key.page.to_le_bytes()))
            .and_then(|_| stream.write_all(&(count as u32).to_le_bytes()))
            .map_err(|e| io("write custody command", e))
    }
    pub fn keep(
        &self,
        owner: Owner,
        key: PageKey,
        spans: &[u64],
        container: BorrowedFd<'_>,
    ) -> Result<bool> {
        if spans.is_empty() || spans.len() > 4096 || spans.contains(&0) {
            return Err(Error::Invalid("invalid kept chunk spans".into()));
        }
        let mut stream = self.0.lock();
        Self::header(&mut stream, 1, owner, key, spans.len())?;
        for span in spans {
            stream
                .write_all(&span.to_le_bytes())
                .map_err(|e| io("write kept span", e))?;
        }
        platform::handles::send_fd(&stream, &container)
            .map_err(crate::os("send kept container"))?;
        let mut ack = [0];
        stream
            .read_exact(&mut ack)
            .map_err(|e| io("read keep acknowledgement", e))?;
        Ok(ack == [1])
    }
    pub fn take(&self, owner: Owner, key: PageKey) -> Result<Vec<(u64, OwnedFd)>> {
        let mut stream = self.0.lock();
        Self::header(&mut stream, 2, owner, key, 0)?;
        let mut size = [0; 4];
        stream
            .read_exact(&mut size)
            .map_err(|e| io("read imported size", e))?;
        let count = u32::from_le_bytes(size) as usize;
        if count > 4096 {
            return Err(Error::Invalid("too many imported chunks".into()));
        }
        let mut spans = Vec::with_capacity(count);
        for _ in 0..count {
            let mut value = [0; 8];
            stream
                .read_exact(&mut value)
                .map_err(|e| io("read imported span", e))?;
            spans.push(u64::from_le_bytes(value));
        }
        spans
            .into_iter()
            .map(|span| {
                platform::handles::recv_fd(&stream)
                    .map(|fd| (span, fd))
                    .map_err(crate::os("receive imported page"))
            })
            .collect()
    }
    pub fn release(&self, owner: Owner, key: PageKey) -> Result<()> {
        let mut stream = self.0.lock();
        Self::header(&mut stream, 3, owner, key, 0)?;
        let mut ack = [0];
        stream
            .read_exact(&mut ack)
            .map_err(|e| io("read release acknowledgement", e))?;
        Ok(())
    }
}

#[cfg(all(test, target_os = "linux"))]
mod admission_tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    fn private(state: &mut HostState, bytes: u64) -> Arc<HostReservation> {
        let (server, client) = UnixStream::pair().unwrap();
        server.set_nonblocking(true).unwrap();
        state.private.push(PrivateHost {
            bytes,
            lease: server,
            pending: true,
        });
        Arc::new(HostReservation {
            lease: client,
            committed: AtomicBool::new(false),
        })
    }

    #[test]
    fn private_pending_commit_and_alias_lifetime_share_one_host_budget() {
        let mut state = HostState::default();
        state.claims.insert(1, None);
        let allocation = private(&mut state, 80);
        assert!(state.can_reserve(100, 20));
        assert!(!state.can_reserve(100, 21));
        assert!(
            !state.can_warm(1000, 1),
            "unknown forecasts suppress optional warming"
        );
        assert!(!state.can_reserve(-1, 1));
        assert!(!state.can_reserve(i64::MAX, u64::MAX));
        state.claims.clear();
        assert!(!state.can_warm(100, 21));
        allocation.commit().unwrap();
        allocation.commit().unwrap();
        state.prune_private();
        assert_eq!(state.pending_private(), Some(0));
        assert_eq!(state.private_bytes(), 80);
        // The allocator has now spent 80 of the original 100 physical bytes.
        // A fresh reading of20 must not subtract that committed span again.
        assert!(state.can_reserve(20, 20));
        let alias = allocation.clone();
        drop(allocation);
        state.prune_private();
        assert_eq!(state.private_bytes(), 80);
        drop(alias);
        state.prune_private();
        assert_eq!(state.private_bytes(), 0);
    }

    #[test]
    fn private_admission_preserves_live_leases_across_row_reuse() {
        let fixture = Fixture::new();
        let row = fixture.ledger.claim(std::process::id()).unwrap();
        let old = row.owner();
        let channel = Channel::new(fixture.ledger.grant(std::process::id()).unwrap().channel);
        let guard = channel
            .reserve_host(old, 4096)
            .unwrap()
            .expect("small host grant");
        assert_eq!(
            unsafe { libc::fcntl(guard.lease.as_raw_fd(), libc::F_GETFD) } & libc::FD_CLOEXEC,
            libc::FD_CLOEXEC,
            "a lease must not leak into an unrelated exec"
        );
        drop(row);
        let next = fixture.ledger.claim(std::process::id()).unwrap();
        assert_eq!(next.owner().row, old.row);
        assert!(next.owner().generation > old.generation);
        assert_eq!(fixture.ledger.status().unwrap().private_host_bytes, 4096);
        assert_eq!(fixture.ledger.pending_host_bytes().unwrap(), 4096);
        guard.commit().unwrap();
        assert_eq!(fixture.ledger.pending_host_bytes().unwrap(), 0);
        assert_eq!(fixture.ledger.status().unwrap().private_host_bytes, 4096);
        drop(guard);
        assert_eq!(fixture.ledger.status().unwrap().private_host_bytes, 0);
    }

    #[test]
    fn private_reclaim_uses_a_fixed_set_of_idle_regions() {
        let fixture = Fixture::new();
        let host = &fixture.ledger.host;
        let put = |layout, used| {
            let region = RamRegion {
                offset: 0,
                len: 4,
                items: vec![],
            };
            let ram = Arc::new(Ram::new(region.clone()).unwrap());
            ram.pending.store(false, Release);
            ram.ready.store(true, Release);
            let mut state = host.state.lock();
            state.clock = used;
            state.ram.insert(
                layout,
                RamHeld {
                    regions: vec![region],
                    slots: vec![RamSlot {
                        buffer: Some(ram),
                        readers: vec![],
                        used,
                    }],
                    observers: BTreeMap::new(),
                    identity: used,
                    generation: used,
                },
            );
        };
        put([1; 32], 1);
        let through = match host
            .reserve_private(&fixture.ledger.table, 4096, 0, None)
            .unwrap()
        {
            HostAdmission::Reclaimed(through) => through,
            _ => panic!("insufficient room must first shed the idle region"),
        };
        assert!(host.state.lock().ram.is_empty());
        put([2; 32], through + 1);
        assert!(matches!(
            host.reserve_private(&fixture.ledger.table, 4096, 0, Some(through))
                .unwrap(),
            HostAdmission::Refused
        ));
        assert!(host.state.lock().ram.contains_key(&[2; 32]));
    }

    #[test]
    fn cache_signals_use_only_the_existing_tail_padding() {
        assert_eq!((size_of::<Header>(), align_of::<Header>()), (64, 64));
        assert_eq!(
            (
                offset_of!(Header, magic),
                offset_of!(Header, capacity),
                offset_of!(Header, changed)
            ),
            (0, 8, 16)
        );
        assert_eq!((size_of::<Slot>(), align_of::<Slot>()), (576, 64));
        assert_eq!(
            [
                offset_of!(Slot, state),
                offset_of!(Slot, pid),
                offset_of!(Slot, bell),
                offset_of!(Slot, seq),
                offset_of!(Slot, copies),
                offset_of!(Slot, asked),
                offset_of!(Slot, handled),
                offset_of!(Slot, wakeups),
                offset_of!(Slot, progress),
            ],
            [0, 8, 12, 16, 24, 424, 480, 536, 544]
        );
        assert_eq!(
            (
                offset_of!(Slot, cache_release),
                offset_of!(Slot, cache_inflight)
            ),
            (552, 560)
        );
        assert_eq!((size_of::<Peak>(), align_of::<Peak>()), (208, 8));
        assert_eq!(
            (
                offset_of!(Peak, state),
                offset_of!(Peak, key),
                offset_of!(Peak, bytes)
            ),
            (0, 8, 200)
        );
        assert_eq!((size_of::<Shared>(), align_of::<Shared>()), (90176, 64));
        assert_eq!(
            (
                offset_of!(Shared, header),
                offset_of!(Shared, rows),
                offset_of!(Shared, peaks)
            ),
            (0, 64, 36928)
        );
    }

    struct Fixture {
        ledger: Ledger,
        name: String,
    }
    impl Fixture {
        fn new() -> Self {
            Self::with_capacity(1024)
        }
        fn with_capacity(capacity: u64) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let name = format!(
                "admission-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Relaxed)
            );
            Self {
                ledger: Ledger::create(&name, capacity).unwrap(),
                name,
            }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let uid = unsafe { libc::geteuid() };
            let _ = std::fs::remove_file(format!("/dev/shm/varena-{}-{uid}", self.name));
        }
    }

    #[test]
    fn unchanged_admission_does_not_repeat_or_erase_an_inflight_cache_release() {
        let fixture = Fixture::new();
        let row = fixture.ledger.claim(std::process::id()).unwrap();
        fixture.ledger.record_peak("package", "shape", 700).unwrap();
        let mut snapshot = Snapshot {
            context: 20,
            ..Default::default()
        };
        snapshot.classes[Class::Pinned.as_usize()] = 600;
        row.publish(&snapshot, 7).unwrap();
        assert!(!fixture.ledger.fits("package", "shape").unwrap());
        let slot = fixture.ledger.table.slot(row.owner()).unwrap();
        let bell = slot.bell.load(Acquire);
        let release = row
            .cache_release()
            .unwrap()
            .expect("explicit cache release");
        assert_eq!(slot.cache_release.load(Acquire), row.owner().generation);
        assert_eq!(slot.cache_inflight.load(Acquire), row.owner().generation);
        for _ in 0..100 {
            assert!(!fixture.ledger.fits("package", "shape").unwrap());
        }
        assert_eq!(slot.bell.load(Acquire), bell);
        assert!(row.cache_release().unwrap().is_none());
        drop(release);
        assert_eq!(
            (
                slot.cache_release.load(Acquire),
                slot.cache_inflight.load(Acquire)
            ),
            (0, 0)
        );
        assert!(fixture.ledger.fits("package", "shape").unwrap());
        assert_eq!(
            slot.bell.load(Acquire),
            bell,
            "a no-progress callback must not cause a wake loop"
        );
        snapshot.classes[Class::Pinned.as_usize()] = 0;
        snapshot.classes[Class::Unused.as_usize()] = 600;
        row.publish(&snapshot, 8).unwrap();
        assert!(fixture.ledger.fits("package", "shape").unwrap());
    }

    #[test]
    fn an_idle_adaptive_request_can_try_after_noop_cache_release_despite_its_observed_peak() {
        let fixture = Fixture::with_capacity(25_153_044_480);
        let row = fixture.ledger.claim(std::process::id()).unwrap();
        let mut snapshot = Snapshot {
            context: 331_350_016,
            ..Default::default()
        };
        snapshot.classes[Class::Pinned.as_usize()] = 100_949_504;
        fixture
            .ledger
            .record_peak("h3", "shape", 24_777_107_456)
            .unwrap();
        row.publish(&snapshot, 10998).unwrap();
        assert!(!fixture.ledger.fits("h3", "shape").unwrap());
        let release = row
            .cache_release()
            .unwrap()
            .expect("idle cache release requested");
        assert!(
            !fixture.ledger.fits("h3", "shape").unwrap(),
            "in-flight cleanup is not quiescence"
        );
        drop(release); // A real no-op acknowledgement: context and pins stay exactly as published.
        assert!(
            fixture.ledger.fits("h3", "shape").unwrap(),
            "observed peak permanently blocked an idle adaptive call"
        );
        assert_eq!(fixture.ledger.table.rows().unwrap()[0].snapshot, snapshot);
        assert_eq!(fixture.ledger.capacity(), 25_153_044_480);
        assert_eq!(fixture.ledger.measured("h3", "shape"), Some(24_777_107_456));
        snapshot.requests = vec![Running {
            id: 2,
            remaining: Some(1),
        }];
        row.publish(&snapshot, 10999).unwrap();
        assert!(
            !fixture.ledger.fits("h3", "shape").unwrap(),
            "running work lost conservative admission"
        );
        snapshot.requests[0].remaining = None;
        row.publish(&snapshot, 11000).unwrap();
        assert!(
            !fixture.ledger.fits("h3", "shape").unwrap(),
            "unknown running work was admitted beside"
        );
    }

    #[test]
    fn admission_waits_for_cache_ack_even_if_the_published_bytes_already_fit() {
        let fixture = Fixture::new();
        let row = fixture.ledger.claim(std::process::id()).unwrap();
        row.publish(&Snapshot::default(), 1).unwrap();
        fixture.ledger.record_peak("package", "shape", 1).unwrap();
        fixture.ledger.release_idle_caches(row.owner()).unwrap();
        assert!(!fixture.ledger.fits("package", "shape").unwrap());
        let release = row.cache_release().unwrap().unwrap();
        assert!(!fixture.ledger.fits("package", "shape").unwrap());
        drop(release);
        assert!(fixture.ledger.fits("package", "shape").unwrap());
    }

    #[test]
    fn a_reaper_with_an_old_snapshot_cannot_clear_a_reused_generation() {
        let fixture = Fixture::new();
        let mut old = fixture.ledger.claim(std::process::id()).unwrap();
        let owner = old.owner();
        // Kernel release without Row::drop's state update, as after SIGKILL.
        drop(old._claim.take());
        let (ready, snapshot) = std::sync::mpsc::channel();
        let (resume, waiting) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let ledger = &fixture.ledger;
            let reaper = scope.spawn(move || {
                ready.send(()).unwrap();
                waiting.recv().unwrap();
                ledger.reap_owner(owner).unwrap()
            });
            snapshot.recv().unwrap();
            fixture.ledger.reap(std::process::id()).unwrap();
            let mut next = fixture.ledger.claim(std::process::id()).unwrap();
            assert_eq!(owner.row, next.owner().row);
            assert!(next.owner().generation > owner.generation);
            resume.send(()).unwrap();
            assert!(
                !reaper.join().unwrap(),
                "a fresh OFD guard must respect the new live claim"
            );
            let expected = next.owner().generation << 2 | LIVE;
            let slot = fixture.ledger.table.slot(next.owner()).unwrap();
            assert_eq!(slot.state.load(Acquire), expected);
            // The old snapshot must also fail its generation recheck after the
            // new owner's kernel claim disappears, when taking the guard succeeds.
            drop(next._claim.take());
            assert!(!fixture.ledger.reap_owner(owner).unwrap());
            assert_eq!(slot.state.load(Acquire), expected);
            assert!(fixture.ledger.reap_owner(next.owner()).unwrap());
        });
    }

    #[test]
    fn a_late_cache_signal_cannot_target_a_reused_row_generation() {
        let fixture = Fixture::new();
        let row = fixture.ledger.claim(std::process::id()).unwrap();
        let old = row.owner();
        fixture.ledger.release_idle_caches(old).unwrap();
        drop(row.cache_release().unwrap().unwrap());
        drop(row);
        let next = fixture.ledger.claim(std::process::id()).unwrap();
        assert_eq!(old.row, next.owner().row);
        assert!(next.owner().generation > old.generation);
        let slot = fixture.ledger.table.slot(next.owner()).unwrap();
        assert_eq!(
            (
                slot.cache_release.load(Acquire),
                slot.cache_inflight.load(Acquire)
            ),
            (0, 0)
        );
        assert!(fixture.ledger.release_idle_caches(old).is_err());
        // A producer that passed the old generation check before its owner died.
        slot.cache_release.fetch_max(old.generation, AcqRel);
        assert!(next.cache_release().unwrap().is_none());
        fixture.ledger.release_idle_caches(next.owner()).unwrap();
        let release = next
            .cache_release()
            .unwrap()
            .expect("new generation signal");
        slot.cache_release.fetch_max(old.generation, AcqRel);
        assert_eq!(slot.cache_release.load(Acquire), next.owner().generation);
        drop(release);
        assert_eq!(
            (
                slot.cache_release.load(Acquire),
                slot.cache_inflight.load(Acquire)
            ),
            (0, 0)
        );
    }
}

/// Retain the lease as long as the CUDA host registration or any reader still uses this file.
pub struct RamView {
    pub file: OwnedFd,
    pub lease: OwnedFd,
    pub offset: u64,
    pub len: u64,
}
impl Channel {
    pub fn ram(&self, owner: Owner, layout: [u8; 32], region: u64) -> Result<Option<RamView>> {
        let mut stream = self.0.lock();
        Self::header(
            &mut stream,
            4,
            owner,
            PageKey {
                fingerprint: layout,
                page: region,
            },
            0,
        )?;
        let mut ready = [0];
        stream
            .read_exact(&mut ready)
            .map_err(|e| io("read RAM readiness", e))?;
        if ready == [0] {
            return Ok(None);
        }
        let mut offset = [0; 8];
        let mut len = [0; 8];
        stream
            .read_exact(&mut offset)
            .and_then(|_| stream.read_exact(&mut len))
            .map_err(|e| io("read RAM range", e))?;
        let file = platform::handles::recv_fd(&stream).map_err(crate::os("receive RAM file"))?;
        let lease = platform::handles::recv_fd(&stream).map_err(crate::os("receive RAM lease"))?;
        Ok(Some(RamView {
            file,
            lease,
            offset: u64::from_le_bytes(offset),
            len: u64::from_le_bytes(len),
        }))
    }
}

impl Channel {
    /// Withdraw a closed model's interest. Existing reader FDs remain valid; an
    /// unobserved cache stops warming and disappears from subsequent snapshots.
    pub fn forget_ram(&self, owner: Owner, layout: [u8; 32], token: u64) -> Result<()> {
        let mut stream = self.0.lock();
        Self::header(
            &mut stream,
            9,
            owner,
            PageKey {
                fingerprint: layout,
                page: token,
            },
            0,
        )?;
        let mut ack = [0];
        stream
            .read_exact(&mut ack)
            .map_err(|e| io("read RAM unsubscribe", e))?;
        if ack != [1] {
            return Err(Error::Invalid("invalid RAM unsubscribe".into()));
        }
        Ok(())
    }
    /// Snapshot only. Completion and eviction also wake subscribed owner rows.
    pub fn ram_status(&self, owner: Owner, layout: [u8; 32]) -> Result<Option<RamStatus>> {
        let mut stream = self.0.lock();
        Self::header(
            &mut stream,
            8,
            owner,
            PageKey {
                fingerprint: layout,
                page: 0,
            },
            0,
        )?;
        let mut present = [0];
        stream
            .read_exact(&mut present)
            .map_err(|e| io("read RAM status", e))?;
        if present == [0] {
            return Ok(None);
        }
        if present != [1] {
            return Err(Error::Invalid("invalid RAM status".into()));
        }
        let generation = read_u64(&mut stream)?;
        let mut filling = [0];
        stream
            .read_exact(&mut filling)
            .map_err(|e| io("read RAM fill status", e))?;
        let count = usize::try_from(read_u64(&mut stream)?)
            .map_err(|_| Error::Invalid("RAM region count overflow".into()))?;
        if count > MAX_RAM_ITEMS || filling[0] > 1 {
            return Err(Error::Invalid("invalid RAM readiness".into()));
        }
        let mut ready = vec![0; count];
        stream
            .read_exact(&mut ready)
            .map_err(|e| io("read RAM readiness", e))?;
        if ready.iter().any(|value| *value > 1) {
            return Err(Error::Invalid("invalid RAM ready flag".into()));
        }
        Ok(Some(RamStatus {
            generation,
            filling: filling[0] == 1,
            ready: ready.into_iter().map(|v| v == 1).collect(),
        }))
    }
    pub fn release_orphan(&self, owner: Owner, key: PageKey) -> Result<bool> {
        let mut stream = self.0.lock();
        Self::header(&mut stream, 5, owner, key, 0)?;
        let mut released = [0];
        stream
            .read_exact(&mut released)
            .map_err(|e| io("read orphan release", e))?;
        Ok(released == [1])
    }
    pub fn orphans(&self, owner: Owner) -> Result<Vec<(PageKey, u64)>> {
        let mut stream = self.0.lock();
        Self::header(
            &mut stream,
            6,
            owner,
            PageKey {
                fingerprint: [0; 32],
                page: 0,
            },
            0,
        )?;
        let mut count = [0; 4];
        stream
            .read_exact(&mut count)
            .map_err(|e| io("read orphan count", e))?;
        let count = u32::from_le_bytes(count);
        if count > 1_000_000 {
            return Err(Error::Invalid("too many orphan pages".into()));
        }
        (0..count)
            .map(|_| {
                let mut fingerprint = [0; 32];
                let mut page = [0; 8];
                let mut bytes = [0; 8];
                stream
                    .read_exact(&mut fingerprint)
                    .and_then(|_| stream.read_exact(&mut page))
                    .and_then(|_| stream.read_exact(&mut bytes))
                    .map_err(|e| io("read orphan page", e))?;
                Ok((
                    PageKey {
                        fingerprint,
                        page: u64::from_le_bytes(page),
                    },
                    u64::from_le_bytes(bytes),
                ))
            })
            .collect()
    }
}

const MAX_RAM_ITEMS: usize = 1 << 20;
const MAX_INLINE_BYTES: usize = 16 << 20;
fn read_u64(stream: &mut UnixStream) -> Result<u64> {
    let mut bytes = [0; 8];
    stream
        .read_exact(&mut bytes)
        .map_err(|e| io("read RAM metadata", e))?;
    Ok(u64::from_le_bytes(bytes))
}
fn read_count(stream: &mut UnixStream, maximum: usize) -> Result<usize> {
    let value = usize::try_from(read_u64(stream)?)
        .map_err(|_| Error::Invalid("RAM count overflow".into()))?;
    if value > maximum {
        return Err(Error::Invalid("RAM metadata exceeds bounds".into()));
    }
    Ok(value)
}
fn read_object(stream: &mut UnixStream) -> Result<ObjectRef> {
    let mut hash = [0; 64];
    stream
        .read_exact(&mut hash)
        .map_err(|e| io("read object hash", e))?;
    if !hash
        .iter()
        .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(Error::Invalid("noncanonical object hash".into()));
    }
    Ok(ObjectRef {
        sha256: String::from_utf8_lossy(&hash).into_owned(),
        length: read_u64(stream)?,
    })
}
fn write_object(stream: &mut UnixStream, object: &ObjectRef) -> Result<()> {
    if object.sha256.len() != 64
        || !object
            .sha256
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(Error::Invalid("noncanonical object hash".into()));
    }
    stream
        .write_all(object.sha256.as_bytes())
        .and_then(|_| stream.write_all(&object.length.to_le_bytes()))
        .map_err(|e| io("write RAM object", e))
}
fn read_regions(stream: &mut UnixStream) -> Result<Vec<RamRegion>> {
    let count = read_count(stream, MAX_RAM_ITEMS)?;
    let (mut left, mut inline) = (MAX_RAM_ITEMS, MAX_INLINE_BYTES);
    (0..count)
        .map(|_| {
            let offset = read_u64(stream)?;
            let len = read_u64(stream)?;
            let count = read_count(stream, left)?;
            left -= count;
            let items = (0..count)
                .map(|_| {
                    let offset = read_u64(stream)?;
                    let len = read_u64(stream)?;
                    let mut flags = [0; 2];
                    stream
                        .read_exact(&mut flags)
                        .map_err(|e| io("read RAM item", e))?;
                    let source = match flags.first().copied() {
                        Some(0) => ItemSource::Object(ObjectRange {
                            obj: read_object(stream)?,
                            off: read_u64(stream)?,
                            len,
                        }),
                        Some(1) => {
                            let size = read_count(stream, inline)?;
                            inline -= size;
                            let mut bytes = vec![0; size];
                            stream
                                .read_exact(&mut bytes)
                                .map_err(|e| io("read inline source", e))?;
                            ItemSource::Inline(bytes.into())
                        }
                        _ => return Err(Error::Invalid("unknown RAM source kind".into())),
                    };
                    Ok(Item {
                        offset,
                        len,
                        source,
                        part_end: flags.get(1) == Some(&1),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(RamRegion { offset, len, items })
        })
        .collect()
}
impl Channel {
    /// Supply already verified read-only objects. The server retains every descriptor for
    /// the asynchronous TensorFS fill; only typed region spans cross this control channel.
    pub fn cache_ram(
        &self,
        owner: Owner,
        layout: [u8; 32],
        files: &[(ObjectRef, OwnedFd)],
        regions: &[RamRegion],
    ) -> Result<bool> {
        self.cache_ram_tagged(owner, layout, files, regions, 0)
    }
    /// A model-specific subscription prevents a closing registration from withdrawing
    /// a newer model's interest in the same shared layout.
    pub fn cache_ram_tagged(
        &self,
        owner: Owner,
        layout: [u8; 32],
        files: &[(ObjectRef, OwnedFd)],
        regions: &[RamRegion],
        token: u64,
    ) -> Result<bool> {
        if files.len() > 4096 || regions.len() > MAX_RAM_ITEMS {
            return Err(Error::Invalid("RAM registration is too large".into()));
        }
        let canonical = |object: &ObjectRef| {
            object.sha256.len() == 64
                && object
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        };
        if files.iter().any(|(object, _)| !canonical(object))
            || regions.iter().flat_map(|r| &r.items).any(
                |item| matches!(&item.source,ItemSource::Object(range) if !canonical(&range.obj)),
            )
        {
            return Err(Error::Invalid("noncanonical RAM source hash".into()));
        }
        let (mut items, mut inline) = (0usize, 0usize);
        for region in regions {
            items = items.saturating_add(region.items.len());
            for item in &region.items {
                if let ItemSource::Inline(bytes) = &item.source {
                    inline = inline.saturating_add(bytes.len());
                }
            }
        }
        if items > MAX_RAM_ITEMS || inline > MAX_INLINE_BYTES {
            return Err(Error::Invalid("RAM metadata exceeds bounds".into()));
        }
        let mut stream = self.0.lock();
        Self::header(
            &mut stream,
            7,
            owner,
            PageKey {
                fingerprint: layout,
                page: token,
            },
            files.len(),
        )?;
        for (object, _) in files {
            write_object(&mut stream, object)?;
        }
        stream
            .write_all(&(regions.len() as u64).to_le_bytes())
            .map_err(|e| io("write RAM regions", e))?;
        for region in regions {
            for value in [region.offset, region.len, region.items.len() as u64] {
                stream
                    .write_all(&value.to_le_bytes())
                    .map_err(|e| io("write RAM region", e))?;
            }
            for item in &region.items {
                for value in [item.offset, item.len] {
                    stream
                        .write_all(&value.to_le_bytes())
                        .map_err(|e| io("write RAM item", e))?;
                }
                let tag = if matches!(item.source, ItemSource::Object(_)) {
                    0
                } else {
                    1
                };
                stream
                    .write_all(&[tag, u8::from(item.part_end)])
                    .map_err(|e| io("write RAM item kind", e))?;
                match &item.source {
                    ItemSource::Object(range) => {
                        write_object(&mut stream, &range.obj)?;
                        stream
                            .write_all(&range.off.to_le_bytes())
                            .map_err(|e| io("write object range", e))?;
                    }
                    ItemSource::Inline(bytes) => {
                        stream
                            .write_all(&(bytes.len() as u64).to_le_bytes())
                            .and_then(|_| stream.write_all(bytes))
                            .map_err(|e| io("write inline RAM source", e))?;
                    }
                }
            }
        }
        for (_, fd) in files {
            platform::handles::send_fd(&stream, fd).map_err(crate::os("send RAM source"))?;
        }
        let mut accepted = [0];
        stream
            .read_exact(&mut accepted)
            .map_err(|e| io("read RAM registration", e))?;
        Ok(accepted == [1])
    }
}

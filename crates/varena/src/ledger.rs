//! Shared placement facts and handle custody. No eviction policy or CUDA calls live here.
//!
//! A named table survives a TensorD restart. Each executor holds its row's OFD lock on a
//! private file description, so even SIGKILL drops the claim. Snapshot copies are atomics:
//! an interrupted publisher leaves the preceding complete copy readable.

use std::collections::{BTreeMap, HashMap};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{fence, AtomicBool, AtomicU32, AtomicU64, Ordering::*};
use std::sync::Arc;
use std::thread::JoinHandle;

use parking_lot::Mutex;
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
        if file.metadata().map_err(|e| io("stat ledger", e))?.len() != size_of::<Shared>() as u64 {
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
    observations: Mutex<HashMap<Owner, (u64, u64)>>,
}
impl Ledger {
    /// Open the machine/user's existing ledger, or initialize it under the file's OFD lock.
    pub fn create(gpu_uuid: &str, capacity: u64) -> Result<Self> {
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
        if table.shared().header.capacity.load(Acquire) != capacity {
            if table.rows()?.iter().any(|row| row.state == State::Live) {
                return Err(Error::Invalid(
                    "cannot change a GPU cap while an executor owns a row".into(),
                ));
            }
            table.shared().header.capacity.store(capacity, Release);
            // Cuts and resident targets were measured under the old limit.
            // The first request under the new cap must establish its own peak.
            for peak in &table.shared().peaks {
                peak.state.store(0, Release);
            }
            table.changed();
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
        Ok(Self {
            table: table.clone(),
            events,
            _watcher: Some(Watcher {
                stop,
                table,
                join: Some(join),
            }),
            custody: Arc::default(),
            observations: Mutex::default(),
        })
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
            observations: Mutex::default(),
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
    /// A private rank channel created before the rank process exists. Its first bind pins
    /// one live row generation; no later message may name a different owner.
    pub fn grant_unbound(&self) -> Result<Grant> {
        self.grant(0)
    }
    pub fn grant(&self, pid: u32) -> Result<Grant> {
        let (server, client) = UnixStream::pair().map_err(|e| io("create grant channel", e))?;
        let (table, custody) = (self.table.clone(), self.custody.clone());
        std::thread::Builder::new()
            .name("varena-custody".into())
            .spawn(move || serve(server, pid, table, custody))
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
        let kept = self
            .custody
            .lock()
            .pages
            .iter()
            .filter(|(_, page)| queued(page))
            .map(|(key, page)| (*key, sum(page.spans.iter().copied())))
            .collect();
        Ok(Status {
            capacity: self.capacity(),
            rows: self.table.rows()?,
            peaks: self.peaks(),
            kept,
            ram_bytes: sum(self
                .custody
                .lock()
                .ram
                .values()
                .map(|r| r.layout.len as u64)),
        })
    }
    /// Reap only kernel-released claims. Their custody remains available until consumed or
    /// explicitly dropped; no GPU context is needed to close the queued handle containers.
    pub fn reap(&self, pid: u32) -> Result<()> {
        for row in self.table.rows()?.into_iter().filter(|r| r.pid == pid) {
            if row.state == State::Live {
                return Err(Error::Invalid("cannot reap a live row".into()));
            }
            let slot = self.table.slot(row.owner)?;
            slot.state
                .store(row.owner.generation << 2 | REAPING, Release);
            self.custody.lock().orphan(row.owner);
            slot.state.store(row.owner.generation << 2, Release);
        }
        self.table.changed();
        Ok(())
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
            let pending = slot
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
    /// Measured remaining working sets are declarations, not a learned usage prediction.
    pub fn fits(&self, package: &str, shape: &str) -> Result<bool> {
        let rows = self.table.rows()?;
        if rows.iter().any(|r| r.state != State::Live) {
            return Ok(false);
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
        Ok(fixed
            .saturating_add(sum(running.iter().filter_map(|r| r.remaining)))
            .saturating_add(peak)
            <= self.capacity())
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
        register_ram(&self.table, &self.custody, layout, source, regions)
    }
    /// Explicit pressure reclamation: never drops a layout with a live reader lease.
    pub fn release_ram(&self, layout: [u8; 32]) -> bool {
        let mut custody = self.custody.lock();
        let idle = custody.ram.get_mut(&layout).is_some_and(|held| {
            held.readers
                .retain(|reader| !platform::handles::hung_up(reader));
            held.readers.is_empty() && !held.layout.filling.load(Acquire)
        });
        if idle {
            custody.ram.remove(&layout);
            self.table.changed();
        }
        idle
    }
    /// On real host pressure, release the oldest completed RAM layout with no live view.
    /// No GPU pages or running-request buffers are considered here.
    pub fn release_unused_ram(&self) -> bool {
        let mut custody = self.custody.lock();
        let victim = custody
            .ram
            .iter_mut()
            .filter_map(|(key, held)| {
                held.readers
                    .retain(|reader| !platform::handles::hung_up(reader));
                (held.readers.is_empty() && !held.layout.filling.load(Acquire))
                    .then_some((*key, held.used))
            })
            .min_by_key(|(_, used)| *used)
            .map(|(key, _)| key);
        if let Some(key) = victim {
            custody.ram.remove(&key);
            self.table.changed();
            true
        } else {
            false
        }
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
impl Row {
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
        let slot = self.table.slot(self.owner)?;
        slot.bell.fetch_add(1, Release);
        wake(&slot.bell);
        self.table.changed();
        Ok(())
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
struct Ram {
    file: File,
    base: usize,
    len: usize,
    regions: Vec<RamRegion>,
    ready: Vec<AtomicBool>,
    filling: AtomicBool,
}
// SAFETY: regions are disjoint, one worker writes them, and a region is published only after
// its TensorFS read finishes. Receivers get read-only file descriptions, never this pointer.
unsafe impl Send for Ram {}
unsafe impl Sync for Ram {}
impl Ram {
    fn new(regions: Vec<RamRegion>) -> Result<Self> {
        let mut end = 0u64;
        for region in &regions {
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
        let len =
            usize::try_from(end).map_err(|_| Error::Invalid("RAM layout too large".into()))?;
        if len == 0 {
            return Err(Error::Invalid("empty RAM layout".into()));
        }
        let file =
            File::from(shm::memfd_named("varena-ram").map_err(crate::os("create RAM layout"))?);
        file.set_len(end).map_err(|e| io("size RAM layout", e))?;
        // SAFETY: a sized memfd; this mapping is the sole writer and is retained until drop.
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
            return Err(last("map RAM layout"));
        }
        let ram = Self {
            file,
            base: base as usize,
            len,
            ready: regions.iter().map(|_| AtomicBool::new(false)).collect(),
            filling: AtomicBool::new(true),
            regions,
        };
        shm::seal(ram.file.as_fd()).map_err(crate::os("seal RAM layout"))?;
        Ok(ram)
    }
}
impl Drop for Ram {
    fn drop(&mut self) {
        // SAFETY: the worker and service retain an Arc until they finish using this mapping.
        unsafe {
            libc::munmap(self.base as *mut libc::c_void, self.len);
        }
    }
}
struct RamHeld {
    layout: Arc<Ram>,
    readers: Vec<UnixStream>,
    used: u64,
}
#[derive(Default)]
struct Custody {
    pages: BTreeMap<PageKey, Kept>,
    ram: BTreeMap<[u8; 32], RamHeld>,
    clock: u64,
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

fn register_ram(
    table: &Arc<Table>,
    custody: &Mutex<Custody>,
    layout: [u8; 32],
    source: Arc<Source>,
    regions: Vec<RamRegion>,
) -> Result<bool> {
    let ram = Arc::new(Ram::new(regions)?);
    let mut custody = custody.lock();
    if custody.ram.contains_key(&layout) {
        return Ok(false);
    }
    let (fill, table) = (ram.clone(), table.clone());
    std::thread::Builder::new()
        .name("varena-ram".into())
        .spawn(move || {
            for (region, ready) in fill.regions.iter().zip(&fill.ready) {
                // SAFETY: validated nonoverlapping ranges in this worker's writable mapping.
                let result = unsafe {
                    source.read_items(
                        &region.items,
                        region.offset,
                        Dst::Host {
                            ptr: (fill.base + region.offset as usize) as *mut u8,
                            len: region.len as usize,
                        },
                        1,
                        Mode::Cache,
                        &IoTally::default(),
                    )
                };
                match result {
                    Ok(_) => {
                        ready.store(true, Release);
                        table.changed();
                    }
                    Err(error) => tracing::warn!(%error, "RAM region remains on disk"),
                }
            }
            fill.filling.store(false, Release);
            table.changed();
        })
        .map_err(|e| io("start RAM fill", e))?;
    custody.clock = custody.clock.saturating_add(1);
    let used = custody.clock;
    custody.ram.insert(
        layout,
        RamHeld {
            layout: ram,
            readers: Vec::new(),
            used,
        },
    );
    Ok(true)
}

/// The grant channel carries only placement custody, never model headers or data bytes.
/// Packet: op/u8, row/u32, generation/u64, fingerprint/32 bytes, page/u64, count/u32,
/// spans/count*u64; an offer then sends one SCM_RIGHTS container. Imports consume it once.
fn serve(mut stream: UnixStream, pid: u32, table: Arc<Table>, custody: Arc<Mutex<Custody>>) {
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
                    let replace = held.pages.get(&key).is_none_or(|p| {
                        !queued(p)
                            || p.owner == Some(owner)
                            || p.owner
                                .is_none_or(|o| !table.alive(o.row as usize).unwrap_or(true))
                    });
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
                    let available = held.pages.get(&key).is_some_and(|p| {
                        p.owner
                            .is_none_or(|o| !table.alive(o.row as usize).unwrap_or(true))
                    });
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
                    let mut held = custody.lock();
                    held.clock = held.clock.saturating_add(1);
                    let used = held.clock;
                    let ready = held.ram.get_mut(&key.fingerprint).and_then(|ram| {
                        let index = usize::try_from(key.page).ok()?;
                        let region = ram.layout.regions.get(index)?;
                        let (offset, len) = (region.offset, region.len);
                        ram.layout
                            .ready
                            .get(index)?
                            .load(Acquire)
                            .then_some((ram, offset, len))
                    });
                    if let Some((ram, offset, len)) = ready {
                        ram.used = used;
                        let (ours, theirs) =
                            UnixStream::pair().map_err(|e| io("create RAM reader lease", e))?;
                        let file = shm::reopen(ram.layout.file.as_fd(), false)
                            .map_err(crate::os("open read-only RAM"))?;
                        stream
                            .write_all(&[1])
                            .and_then(|_| stream.write_all(&offset.to_le_bytes()))
                            .and_then(|_| stream.write_all(&len.to_le_bytes()))
                            .map_err(|e| io("send RAM region", e))?;
                        platform::handles::send_fd(&stream, &file)
                            .map_err(crate::os("send RAM file"))?;
                        platform::handles::send_fd(&stream, &theirs)
                            .map_err(crate::os("send RAM lease"))?;
                        ram.readers.push(ours);
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
                        &custody,
                        key.fingerprint,
                        Arc::new(Source::files(files, true, 0)),
                        regions,
                    )?;
                    stream
                        .write_all(&[u8::from(accepted)])
                        .map_err(|e| io("ack RAM registration", e))?;
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
                page: 0,
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

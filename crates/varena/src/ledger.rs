//! The per-GPU ledger (design memory-varena §5.6): a memfd of rows, one per process, each held
//! by an OFD lock that dies with it; then measured peaks (`<package>|<shape>` → bytes) and
//! TensorD's dispatches (request id → that key). Rows are atomics; facts are a two-copy latch,
//! so readers never tear nor wait on a dead writer. Asks are cumulative counters, frees bump a
//! futex generation and an eventfd. No CUDA, no clocks: stalls are progress that did not move.

use std::collections::HashMap;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd};
use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release};
use std::sync::atomic::{fence, AtomicU32, AtomicU64};
use std::sync::{Arc, Mutex};

use platform::shm::{self, Region, RESIZE_SEALS};

use crate::{lock, Error, Result};

const MAGIC: u64 = u64::from_le_bytes(*b"vledger3");
pub const ROWS: u32 = 64;
pub const REQUEST_ID: usize = 64;
pub const PEAK_KEY: usize = 120;
pub const RUNNING: usize = 8;
const HEADER: usize = 256;
const ROW: usize = 1536;
const PEAKS: usize = 512;
const DISPATCH: usize = 64;
const FACTS: usize = 12;
const ID: usize = REQUEST_ID / 8;
const KEY: usize = PEAK_KEY / 8;
/// The latch: the facts, the running count, then (id, remaining) per request.
const WORDS: usize = FACTS + 1 + RUNNING * (ID + 1);
const LEN: usize = HEADER + ROWS as usize * ROW + PEAKS * size_of::<Peak>() + DISPATCH * size_of::<Dispatch>();

const FREE: u64 = 0;
const LIVE: u64 = 1;
const REAPING: u64 = 2;
const CLAIMING: u64 = 1;
const READY: u64 = 2;

#[repr(C)]
struct Header {
    magic: u64,
    /// The free generation: a futex word `freed` bumps.
    freed: AtomicU32,
    capacity: u64,
    uuid: [u64; 8],
}

#[repr(C, align(256))]
struct Slot {
    /// `gen << 2 | state`: unchanged around a read, the read saw one row.
    word: AtomicU64,
    pid: AtomicU32,
    /// A futex word rung by asks, acks, reaps and returns.
    bell: AtomicU32,
    seq: AtomicU64,
    asked: AtomicU64,
    handled: AtomicU64,
    freed: AtomicU64,
    progress: AtomicU64,
    reaping: AtomicU64,
    latch: [[AtomicU64; WORDS]; 2],
}

/// Filled in a prefix, never freed: 0, then `CLAIMING` while its words are written, then `READY`.
#[repr(C)]
struct Peak {
    word: AtomicU64,
    key: [AtomicU64; KEY],
    peak: AtomicU64,
}

#[repr(C)]
struct Dispatch {
    /// `gen << 1 | 1` while it names a request; each fill and clear moves it.
    state: AtomicU64,
    id: [AtomicU64; ID],
    key: [AtomicU64; KEY],
}

const _: () = assert!(size_of::<Slot>() == ROW && size_of::<Header>() <= HEADER);

/// `text` as zero-padded words; None when longer than they hold or holding a NUL.
fn pack<const N: usize>(text: &str) -> Option<[u64; N]> {
    if text.len() > N * 8 || text.contains('\0') {
        return None;
    }
    let mut words = [0u64; N];
    for (w, chunk) in words.iter_mut().zip(text.as_bytes().chunks(8)) {
        let mut b = [0u8; 8];
        b[..chunk.len()].copy_from_slice(chunk);
        *w = u64::from_le_bytes(b);
    }
    Some(words)
}

fn packed<const N: usize>(what: &str, text: &str) -> Result<[u64; N]> {
    pack(text).ok_or_else(|| Error::Invalid(format!("{what} {text:?}: at most {} bytes, no NUL", N * 8)))
}

fn unpack(words: &[u64]) -> String {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    String::from_utf8_lossy(&bytes[..bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len())]).into_owned()
}

fn load<const N: usize>(words: &[AtomicU64; N]) -> [u64; N] {
    std::array::from_fn(|i| words[i].load(Relaxed))
}

fn store(atomics: &[AtomicU64], words: &[u64]) {
    atomics.iter().zip(words).for_each(|(a, v)| a.store(*v, Relaxed));
}

fn sum(xs: impl IntoIterator<Item = u64>) -> u64 {
    xs.into_iter().fold(0, u64::saturating_add)
}

/// A row's bytes on this GPU, and `last_use` in CLOCK_MONOTONIC ns. `cold`: resident pages an
/// ask evicts, `kept` the warm set's share of them (asked last). `cache`: torch bytes shed on an
/// ask. `idle`: varena's idle chunks. `active_peak`: the active requests' measured peak.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Facts {
    pub context: u64,
    pub resident: u64,
    pub cold: u64,
    pub kept: u64,
    pub pinned: u64,
    pub torch_reserved: u64,
    pub torch_allocated: u64,
    pub cache: u64,
    pub idle: u64,
    pub active_peak: u64,
    pub last_use: u64,
    pub waiting: u64,
}

impl Facts {
    fn words(&self) -> [u64; FACTS] {
        let f = self;
        [f.context, f.resident, f.cold, f.kept, f.pinned, f.torch_reserved, f.torch_allocated, f.cache, f.idle, f.active_peak, f.last_use, f.waiting]
    }

    fn from_words(w: &[u64]) -> Facts {
        let [context, resident, cold, kept, pinned, torch_reserved, torch_allocated, cache, idle, active_peak, last_use, waiting] =
            w[..FACTS].try_into().unwrap();
        Facts { context, resident, cold, kept, pinned, torch_reserved, torch_allocated, cache, idle, active_peak, last_use, waiting }
    }

    /// Torch bytes no ask frees: live tensors outside caches, or the active peak if higher.
    fn activations(&self) -> u64 {
        self.torch_allocated.saturating_sub(self.cache).max(self.active_peak)
    }

    /// Every byte the process holds now or will for its active requests.
    pub fn held(&self) -> u64 {
        sum([self.context, self.resident, self.idle, self.torch_reserved.max(self.activations().saturating_add(self.cache))])
    }

    /// What no ask frees: the context, weights in use or pinned, and activations.
    pub fn unevictable(&self) -> u64 {
        sum([self.context, self.resident.saturating_sub(self.cold), self.activations()])
    }

    /// What an ask frees, in order: idle chunks, caches and slack; cold weights; kept ones.
    fn reclaim(&self) -> [u64; 3] {
        let cold = self.cold.min(self.resident);
        let kept = self.kept.min(cold);
        [self.held().saturating_sub(self.unevictable()).saturating_sub(cold), cold - kept, kept]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Live,
    /// Its process is gone; its bytes count until reaped and returned.
    Dead,
    Reaping,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowFacts {
    pub slot: u32,
    pub gen: u64,
    pub pid: u32,
    pub state: State,
    pub facts: Facts,
    pub want: u64,
    pub acked: u64,
    pub progress: u64,
    pub reaping: u64,
    /// (request id, remaining measured peak; -1 never measured).
    pub running: Vec<(String, i64)>,
}

impl RowFacts {
    pub fn held(&self) -> u64 {
        if self.state == State::Reaping { self.reaping } else { self.facts.held() }
    }

    /// Bytes an ask can get from this row: only a live one answers.
    pub fn reclaimable(&self) -> u64 {
        if self.state == State::Live { self.facts.held().saturating_sub(self.facts.unevictable()) } else { 0 }
    }

    pub fn unevictable(&self) -> u64 {
        self.held() - self.reclaimable()
    }
}

/// A row asked (or, from `admit`, to ask) for `bytes`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Victim {
    pub slot: u32,
    pub gen: u64,
    pub pid: u32,
    pub bytes: u64,
    /// The row's `handled` that answers this ask.
    until: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Admit {
    Now,
    /// `victims` (least recently used first) cover the gap but `short`, which no ask frees.
    After { victims: Vec<Victim>, short: u64 },
}

/// A table at a glance (TensorD's `status`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Status {
    pub capacity: u64,
    pub freed: u64,
    pub rows: Vec<RowFacts>,
    pub peaks: Vec<(String, u64)>,
    pub dispatched: Vec<(String, String)>,
}

fn io(what: &str) -> Error {
    Error::Io(format!("{what}: {}", std::io::Error::last_os_error()))
}

/// Copy 0 is written while `seq` is odd, copy 1 while even; a reader reads the other.
fn latch_write(seq: &AtomicU64, copies: &[[AtomicU64; WORDS]; 2], words: &[u64; WORDS]) {
    let s = seq.load(Relaxed);
    seq.store(s + 1, Relaxed);
    fence(Release);
    store(&copies[0], words);
    seq.store(s + 2, Release);
    fence(Release);
    store(&copies[1], words);
}

fn latch_read(seq: &AtomicU64, copies: &[[AtomicU64; WORDS]; 2]) -> [u64; WORDS] {
    loop {
        let s = seq.load(Acquire);
        let words = load(&copies[(s & 1) as usize]);
        fence(Acquire);
        if seq.load(Relaxed) == s {
            return words;
        }
    }
}

#[cfg(target_os = "linux")]
fn wait(word: &AtomicU32, seen: u32) {
    // SAFETY: a futex wait on a live shared word; it returns on a change, a wake or a signal.
    unsafe { libc::syscall(libc::SYS_futex, word.as_ptr(), libc::FUTEX_WAIT, seen, std::ptr::null::<libc::timespec>()) };
}

#[cfg(target_os = "linux")]
fn wake(word: &AtomicU32) {
    // SAFETY: a futex wake on a live shared word.
    unsafe { libc::syscall(libc::SYS_futex, word.as_ptr(), libc::FUTEX_WAKE, i32::MAX) };
}

// Elsewhere no ledger can be created (no memfd), so these are never reached.
#[cfg(not(target_os = "linux"))]
fn wait(_: &AtomicU32, _: u32) {}

#[cfg(not(target_os = "linux"))]
fn wake(_: &AtomicU32) {}

/// Blocks on futex `word` until `done` answers.
fn until<T>(word: &AtomicU32, mut done: impl FnMut() -> Option<T>) -> T {
    loop {
        let seen = word.load(Acquire);
        if let Some(x) = done() {
            return x;
        }
        wait(word, seen);
    }
}

#[cfg(target_os = "linux")]
fn eventfd() -> Result<OwnedFd> {
    // SAFETY: eventfd(2) with no shared state; a negative return is an error.
    let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
    if fd < 0 {
        return Err(io("eventfd"));
    }
    // SAFETY: a descriptor we just created and own.
    Ok(unsafe { <OwnedFd as std::os::fd::FromRawFd>::from_raw_fd(fd) })
}

#[cfg(not(target_os = "linux"))]
fn eventfd() -> Result<OwnedFd> {
    Err(Error::Invalid("the ledger's eventfd is Linux's".into()))
}

impl Slot {
    fn ring(&self) {
        self.bell.fetch_add(1, Release);
        wake(&self.bell);
    }
}

struct Table {
    /// This process's own description: mapped and probed, never locked.
    fd: OwnedFd,
    base: *mut u8,
}

// SAFETY: the mapping is only reached through atomics (rows) or read-only data (the header).
unsafe impl Send for Table {}
// SAFETY: as above.
unsafe impl Sync for Table {}

impl Drop for Table {
    fn drop(&mut self) {
        // SAFETY: our mapping; no reference into it outlives the Table (rows hold an Arc).
        unsafe { libc::munmap(self.base.cast(), LEN) };
    }
}

impl Table {
    fn map(fd: OwnedFd) -> Result<Table> {
        // SAFETY: a shared mapping of a memfd whose size is (or is about to be) sealed.
        let base = unsafe { libc::mmap(std::ptr::null_mut(), LEN, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd.as_raw_fd(), 0) };
        if base == libc::MAP_FAILED {
            return Err(io("map ledger"));
        }
        Ok(Table { fd, base: base.cast() })
    }

    fn header(&self) -> &Header {
        // SAFETY: the header is written once, before the memfd is handed out.
        unsafe { &*(self.base as *const Header) }
    }

    /// `n` records at byte `at`: the rows, then the peaks, then the dispatches.
    fn array<T>(&self, at: usize, n: usize) -> &[T] {
        // SAFETY: inside the mapping and aligned for `T`, whose fields are all atomic.
        unsafe { std::slice::from_raw_parts(self.base.add(at) as *const T, n) }
    }

    fn slot(&self, i: u32) -> &Slot {
        &self.array(HEADER, ROWS as usize)[i as usize]
    }

    fn peaks(&self) -> &[Peak] {
        self.array(HEADER + ROWS as usize * ROW, PEAKS)
    }

    fn dispatch(&self) -> &[Dispatch] {
        self.array(HEADER + ROWS as usize * ROW + PEAKS * size_of::<Peak>(), DISPATCH)
    }

    fn alive(&self, i: u32) -> Result<bool> {
        let holder = shm::region_holder(self.fd.as_fd(), HEADER + i as usize * ROW, ROW).map_err(crate::os("probe ledger row"))?;
        Ok(holder.is_some())
    }

    fn read(&self, i: u32) -> Result<Option<RowFacts>> {
        let s = self.slot(i);
        loop {
            let w = s.word.load(Acquire);
            let state = match w & 3 {
                FREE => return Ok(None),
                REAPING => State::Reaping,
                _ if self.alive(i)? => State::Live,
                _ => State::Dead,
            };
            let words = latch_read(&s.seq, &s.latch);
            let (pid, asked, handled) = (s.pid.load(Relaxed), s.asked.load(Relaxed), s.handled.load(Relaxed));
            let (acked, progress, reaping) = (s.freed.load(Relaxed), s.progress.load(Relaxed), s.reaping.load(Relaxed));
            fence(Acquire);
            if s.word.load(Relaxed) != w {
                continue;
            }
            let running = words[FACTS + 1..].chunks(ID + 1).take((words[FACTS] as usize).min(RUNNING));
            let running = running.map(|r| (unpack(&r[..ID]), r[ID] as i64)).collect();
            let (facts, want) = (Facts::from_words(&words), asked.saturating_sub(handled));
            return Ok(Some(RowFacts { slot: i, gen: w >> 2, pid, state, facts, want, acked, progress, reaping, running }));
        }
    }
}

/// One process's row. Dropping it drops the lock: the row reads dead until reaped and returned.
pub struct Row {
    t: Arc<Table>,
    slot: u32,
    gen: u64,
    pid: u32,
    _lock: OwnedFd,
    seen: AtomicU64,
    /// The latch's words as last written; one writer at a time.
    words: Mutex<[u64; WORDS]>,
}

impl Row {
    pub fn id(&self) -> u32 {
        self.slot
    }

    pub fn gen(&self) -> u64 {
        self.gen
    }

    pub fn pid(&self) -> u32 {
        self.pid
    }

    fn write(&self, at: usize, part: &[u64]) {
        let mut words = lock(&self.words);
        words[at..at + part.len()].copy_from_slice(part);
        let s = self.t.slot(self.slot);
        latch_write(&s.seq, &s.latch, &words);
        s.progress.fetch_add(1, Release);
    }

    pub fn publish(&self, f: &Facts) {
        self.write(0, &f.words());
    }

    /// Its facts less `freed` bytes an answer gave in Rust (cold weights, then idle chunks), so
    /// later asks see what is left before the process publishes again.
    pub fn shed(&self, freed: u64) {
        let mut f = Facts::from_words(&lock(&self.words)[..FACTS]);
        let weights = freed.min(f.cold);
        f.cold -= weights;
        f.resident = f.resident.saturating_sub(weights);
        f.kept = f.kept.min(f.cold);
        f.idle = f.idle.saturating_sub(freed - weights);
        self.publish(&f);
    }

    /// Its running requests and the most each still needs beyond what it holds (-1: unmeasured).
    pub fn running(&self, requests: &[(String, i64)]) -> Result<()> {
        if requests.len() > RUNNING {
            return Err(Error::Invalid(format!("{} running requests: at most {RUNNING}", requests.len())));
        }
        let mut words = [0u64; WORDS - FACTS];
        words[0] = requests.len() as u64;
        for ((id, left), w) in requests.iter().zip(words[1..].chunks_mut(ID + 1)) {
            w[..ID].copy_from_slice(&packed::<ID>("request id", id)?);
            w[ID] = *left as u64;
        }
        self.write(FACTS, &words);
        Ok(())
    }

    /// Bytes others asked of this row and it has not answered.
    pub fn want(&self) -> u64 {
        let s = self.t.slot(self.slot);
        let asked = s.asked.load(Acquire);
        self.seen.fetch_max(asked, Relaxed);
        asked.saturating_sub(s.handled.load(Relaxed))
    }

    /// Blocks until something is asked of this row; returns `want()`.
    pub fn wait_want(&self) -> u64 {
        until(&self.t.slot(self.slot).bell, || Some(self.want()).filter(|w| *w > 0))
    }

    /// Answers every ask `want()` saw, having freed `freed` bytes for them.
    pub fn ack(&self, freed: u64) {
        let s = self.t.slot(self.slot);
        s.handled.fetch_max(self.seen.load(Relaxed), Release);
        s.freed.fetch_add(freed, Release);
        s.progress.fetch_add(1, Release);
        s.ring();
    }

    /// Answers asks on a thread of its own with `answer(want) -> freed`; each answer is progress,
    /// so a process busy in one long kernel is never stalled. A panicking answer frees nothing.
    pub fn serve(self: &Arc<Self>, answer: impl Fn(u64) -> u64 + Send + 'static) -> Result<std::thread::JoinHandle<()>> {
        let row = self.clone();
        let serve = move || loop {
            let want = row.wait_want();
            row.ack(crate::caught(|| Ok(answer(want))).unwrap_or(0));
        };
        std::thread::Builder::new().name("varena-ask".into()).spawn(serve).map_err(|e| Error::Io(format!("spawn the ask thread: {e}")))
    }
}

/// One GPU's table, mapped in this process, and the eventfd beside it.
pub struct Ledger {
    t: Arc<Table>,
    /// `stalled`'s last look: (slot, gen) -> progress.
    looks: Mutex<HashMap<(u32, u64), u64>>,
    events: Option<OwnedFd>,
    /// One writer of the dispatch table in this process (TensorD is its only writer).
    dispatching: Mutex<()>,
}

impl Ledger {
    fn new(t: Table, events: Option<OwnedFd>) -> Ledger {
        Ledger { t: Arc::new(t), looks: Mutex::default(), events, dispatching: Mutex::default() }
    }

    /// A new table and eventfd; returns it and the memfd to hand to other processes.
    pub fn create(gpu_uuid: &str, capacity: u64) -> Result<(Ledger, OwnedFd)> {
        let uuid = packed("gpu uuid", gpu_uuid)?;
        let fd = shm::memfd_named(&format!("varena-ledger:{gpu_uuid}")).map_err(crate::os("memfd_create"))?;
        // SAFETY: sizing a memfd only we hold.
        if unsafe { libc::ftruncate(fd.as_raw_fd(), LEN as libc::off_t) } != 0 {
            return Err(io("size ledger"));
        }
        let t = Table::map(shm::reopen(fd.as_fd(), true).map_err(crate::os("reopen ledger"))?)?;
        // SAFETY: nobody else holds this memfd yet.
        unsafe { (t.base as *mut Header).write(Header { magic: MAGIC, freed: AtomicU32::new(0), capacity, uuid }) };
        shm::seal_with(fd.as_fd(), RESIZE_SEALS).map_err(crate::os("seal ledger"))?;
        Ok((Ledger::new(t, Some(eventfd()?)), fd))
    }

    /// The table behind `fd`, and the eventfd `freed` signals when the caller has it.
    pub fn adopt(fd: BorrowedFd<'_>, events: Option<BorrowedFd<'_>>) -> Result<Ledger> {
        if shm::seals(fd).map_err(crate::os("read ledger seals"))? & RESIZE_SEALS != RESIZE_SEALS {
            return Err(Error::Invalid("not a ledger: its size is not sealed".into()));
        }
        let own = std::fs::File::from(shm::reopen(fd, true).map_err(crate::os("reopen ledger"))?);
        if own.metadata().map_err(|e| Error::Io(format!("stat ledger: {e}")))?.len() != LEN as u64 {
            return Err(Error::Invalid("not a varena ledger: its size".into()));
        }
        let t = Table::map(own.into())?;
        if t.header().magic != MAGIC {
            return Err(Error::Invalid("not a varena ledger".into()));
        }
        let events = events.map(|e| e.try_clone_to_owned()).transpose().map_err(|e| Error::Io(format!("dup eventfd: {e}")))?;
        Ok(Ledger::new(t, events))
    }

    pub fn events(&self) -> Option<BorrowedFd<'_>> {
        self.events.as_ref().map(AsFd::as_fd)
    }

    pub fn fd(&self) -> BorrowedFd<'_> {
        self.t.fd.as_fd()
    }

    pub fn gpu_uuid(&self) -> String {
        unpack(&self.t.header().uuid)
    }

    pub fn capacity(&self) -> u64 {
        self.t.header().capacity
    }

    /// A free row for `pid`, held by an OFD lock on a description of this process's own.
    pub fn claim(&self, pid: u32) -> Result<Row> {
        if let Some(r) = self.snapshot()?.iter().find(|r| r.pid == pid && r.state == State::Live) {
            return Err(Error::Invalid(format!("pid {pid} already holds row {}", r.slot)));
        }
        let lock = shm::reopen(self.t.fd.as_fd(), true).map_err(crate::os("reopen ledger"))?;
        let range = |i: u32, region| shm::region_lock(lock.as_fd(), HEADER + i as usize * ROW, ROW, region, false).map_err(crate::os("lock ledger row"));
        for i in 0..ROWS {
            let s = self.t.slot(i);
            let w = s.word.load(Acquire);
            if w & 3 != FREE || !range(i, Region::Write)? {
                continue;
            }
            if s.word.load(Acquire) != w {
                range(i, Region::Unlock)?;
                continue;
            }
            fence(Release);
            let gen = (w >> 2) + 1;
            s.pid.store(pid, Relaxed);
            // A writer that died mid-write left `seq` odd: start the new one even.
            s.seq.store((s.seq.load(Relaxed) | 1) + 1, Relaxed);
            s.latch.iter().flatten().for_each(|a| a.store(0, Relaxed));
            [&s.asked, &s.handled, &s.freed, &s.progress, &s.reaping].iter().for_each(|a| a.store(0, Relaxed));
            if s.word.compare_exchange(w, gen << 2 | LIVE, Release, Relaxed).is_err() {
                range(i, Region::Unlock)?;
                continue;
            }
            let (t, seen, words) = (self.t.clone(), AtomicU64::new(0), Mutex::new([0; WORDS]));
            return Ok(Row { t, slot: i, gen, pid, _lock: lock, seen, words });
        }
        Err(Error::Invalid(format!("ledger {} is full: {ROWS} rows", self.gpu_uuid())))
    }

    pub fn snapshot(&self) -> Result<Vec<RowFacts>> {
        (0..ROWS).filter_map(|i| self.t.read(i).transpose()).collect()
    }

    /// Asks other live rows for up to `need` bytes, as `pick` orders them, and rings each.
    pub fn ask(&self, from: Option<u32>, need: u64) -> Result<Vec<Victim>> {
        let (mut victims, _) = pick(&self.snapshot()?, from, need);
        victims.retain_mut(|v| {
            let s = self.t.slot(v.slot);
            if s.word.load(Acquire) != v.gen << 2 | LIVE {
                return false;
            }
            v.until = s.asked.fetch_add(v.bytes, AcqRel) + v.bytes;
            s.ring();
            true
        });
        Ok(victims)
    }

    /// The victim answered the ask, or its row is dead or gone.
    pub fn answered(&self, v: &Victim) -> Result<bool> {
        let s = self.t.slot(v.slot);
        Ok(s.word.load(Acquire) != v.gen << 2 | LIVE || s.handled.load(Acquire) >= v.until || !self.t.alive(v.slot)?)
    }

    /// Blocks until `answered`. A victim that dies without answering ends the wait when reaped.
    pub fn wait(&self, v: &Victim) -> Result<()> {
        until(&self.t.slot(v.slot).bell, || self.answered(v).map(|done| done.then_some(())).transpose())
    }

    /// `Now` only if `need` bytes are free beside every byte the rows hold. Never below need.
    pub fn admit(&self, need: u64) -> Result<Admit> {
        let rows = self.snapshot()?;
        let free = self.capacity().saturating_sub(sum(rows.iter().map(RowFacts::held)));
        if free >= need {
            return Ok(Admit::Now);
        }
        let (victims, short) = pick(&rows, None, need - free);
        Ok(Admit::After { victims, short })
    }

    /// Marks `pid`'s dead rows reaping; their bytes count until `returned`. Refuses a live pid.
    pub fn reap(&self, pid: u32) -> Result<u64> {
        let rows: Vec<RowFacts> = self.snapshot()?.into_iter().filter(|r| r.pid == pid).collect();
        let dead: Vec<&RowFacts> = rows.iter().filter(|r| r.state == State::Dead).collect();
        if let Some(r) = rows.iter().find(|r| r.state == State::Live).filter(|_| dead.is_empty()) {
            return Err(Error::Invalid(format!("pid {pid} is alive: it holds row {}", r.slot)));
        }
        let mut held = 0;
        for r in dead {
            let s = self.t.slot(r.slot);
            s.reaping.store(r.facts.held(), Relaxed);
            if s.word.compare_exchange(r.gen << 2 | LIVE, r.gen << 2 | REAPING, AcqRel, Relaxed).is_ok() {
                held += r.facts.held();
                s.ring();
            }
        }
        Ok(held)
    }

    /// `pid`'s reaped bytes are back on the GPU (NVML shows them): its rows are free.
    pub fn returned(&self, pid: u32) -> Result<()> {
        let mut back = false;
        for s in (0..ROWS).map(|i| self.t.slot(i)) {
            let w = s.word.load(Acquire);
            if w & 3 == REAPING && s.pid.load(Relaxed) == pid && s.word.compare_exchange(w, w & !3, Release, Relaxed).is_ok() {
                s.ring();
                back = true;
            }
        }
        if back {
            self.freed();
        }
        Ok(())
    }

    /// Asked rows whose progress has not moved since this handle's last look.
    pub fn stalled(&self) -> Result<Vec<u32>> {
        let rows = self.snapshot()?;
        let mut looks = lock(&self.looks);
        let asked: Vec<&RowFacts> = rows.iter().filter(|r| r.state == State::Live && r.want > 0).collect();
        let stalled = asked.iter().filter(|r| looks.get(&(r.slot, r.gen)) == Some(&r.progress)).map(|r| r.pid).collect();
        *looks = asked.iter().map(|r| ((r.slot, r.gen), r.progress)).collect();
        Ok(stalled)
    }

    /// Bytes freed on this GPU: the free generation moves and the eventfd is signalled.
    pub fn freed(&self) {
        let word = &self.t.header().freed;
        word.fetch_add(1, Release);
        wake(word);
        if let Some(events) = &self.events {
            let one = 1u64.to_ne_bytes();
            // SAFETY: an 8-byte write to an eventfd we hold; a full counter (EAGAIN) is a signal already.
            unsafe { libc::write(events.as_raw_fd(), one.as_ptr().cast(), 8) };
        }
    }

    pub fn freed_gen(&self) -> u64 {
        self.t.header().freed.load(Acquire) as u64
    }

    /// Blocks until the free generation is other than `seen`; returns it.
    pub fn wait_freed(&self, seen: u64) -> u64 {
        until(&self.t.header().freed, || Some(self.freed_gen()).filter(|g| *g != seen))
    }

    /// The measured peak of `key` (`<package>|<shape>`), if any process recorded one.
    pub fn peak(&self, key: &str) -> Option<u64> {
        self.peaks().into_iter().find(|(k, _)| k == key).map(|(_, peak)| peak)
    }

    /// Records `key`'s measured peak (any process; the latest write stands).
    pub fn set_peak(&self, key: &str, bytes: u64) -> Result<()> {
        let (words, peaks): ([u64; KEY], _) = (packed("peak key", key)?, self.t.peaks());
        let mut i = 0;
        while let Some(slot) = peaks.get(i) {
            match slot.word.load(Acquire) {
                READY if load(&slot.key) == words => {
                    slot.peak.store(bytes, Release);
                    return Ok(());
                }
                0 if slot.word.compare_exchange(0, CLAIMING, AcqRel, Acquire).is_ok() => {
                    store(&slot.key, &words);
                    slot.peak.store(bytes, Relaxed);
                    slot.word.store(READY, Release);
                    return Ok(());
                }
                0 => {}
                _ => i += 1,
            }
        }
        Err(Error::Invalid(format!("ledger peaks are full: {PEAKS} keys")))
    }

    pub fn peaks(&self) -> Vec<(String, u64)> {
        let ready = self.t.peaks().iter().filter(|s| s.word.load(Acquire) == READY);
        ready.map(|s| (unpack(&load(&s.key)), s.peak.load(Acquire))).collect()
    }

    /// Names the key a request's peak is recorded under (TensorD's write, before it runs).
    pub fn dispatch(&self, request_id: &str, key: &str) -> Result<()> {
        let (id, key) = (packed::<ID>("request id", request_id)?, packed::<KEY>("peak key", key)?);
        let _one = lock(&self.dispatching);
        self.clear(&id);
        let free = self.t.dispatch().iter().find(|s| s.state.load(Acquire) & 1 == 0);
        let slot = free.ok_or_else(|| Error::Invalid(format!("ledger dispatch table is full: {DISPATCH} requests")))?;
        // Its words change only after the clear that freed it is visible.
        fence(Release);
        store(&slot.id, &id);
        store(&slot.key, &key);
        slot.state.fetch_add(3, Release);
        Ok(())
    }

    /// The request's entry goes (TensorD's write, once it ended).
    pub fn undispatch(&self, request_id: &str) {
        let _one = lock(&self.dispatching);
        if let Some(id) = pack(request_id) {
            self.clear(&id);
        }
    }

    fn clear(&self, id: &[u64; ID]) {
        for slot in self.t.dispatch().iter().filter(|s| s.state.load(Acquire) & 1 == 1 && load(&s.id) == *id) {
            slot.state.fetch_sub(1, Release);
        }
    }

    /// The key TensorD named for `request_id`, if it is dispatched.
    pub fn dispatched(&self, request_id: &str) -> Option<String> {
        self.dispatched_all().into_iter().find(|(id, _)| id == request_id).map(|(_, key)| key)
    }

    fn dispatched_all(&self) -> Vec<(String, String)> {
        let mut entries = vec![];
        for slot in self.t.dispatch() {
            let state = slot.state.load(Acquire);
            let (id, key) = (load(&slot.id), load(&slot.key));
            fence(Acquire);
            if state & 1 == 1 && slot.state.load(Relaxed) == state {
                entries.push((unpack(&id), unpack(&key)));
            }
        }
        entries
    }

    /// Whether a request can start now (design §5.6): never beside a dead row or an unmeasured
    /// request, unmeasured itself only alone; else its peak (at `shape`, or the widest) beside all
    /// unevictable bytes and remaining peaks, plus the largest context if it `spawn`s a process.
    pub fn fits(&self, package: &str, shape: Option<&str>, spawn: bool) -> Result<bool> {
        let rows = self.snapshot()?;
        let left: Vec<i64> = rows.iter().flat_map(|r| r.running.iter().map(|(_, left)| *left)).collect();
        if rows.iter().any(|r| r.state != State::Live) || left.iter().any(|l| *l < 0) {
            return Ok(false);
        }
        let prefix = format!("{package}|");
        let peak = match shape {
            Some(shape) => self.peak(&format!("{prefix}{shape}")),
            None => self.peaks().into_iter().filter(|(key, _)| key.starts_with(&prefix)).map(|(_, peak)| peak).max(),
        };
        let Some(peak) = peak else {
            return Ok(left.is_empty());
        };
        let context = if spawn { rows.iter().map(|r| r.facts.context).max().unwrap_or(0) } else { 0 };
        let reserved = rows.iter().map(RowFacts::unevictable).chain(left.iter().map(|l| *l as u64));
        Ok(sum(reserved.chain([context, peak])) <= self.capacity())
    }

    pub fn status(&self) -> Result<Status> {
        let (capacity, freed, peaks, dispatched) = (self.capacity(), self.freed_gen(), self.peaks(), self.dispatched_all());
        Ok(Status { capacity, freed, rows: self.snapshot()?, peaks, dispatched })
    }
}

/// Victims for `need` among live rows but `from`: idle bytes, then cold, then kept, each least
/// recently used first, net of what each was already asked; and what they leave short.
fn pick(rows: &[RowFacts], from: Option<u32>, need: u64) -> (Vec<Victim>, u64) {
    let mut lru: Vec<&RowFacts> = rows.iter().filter(|r| r.state == State::Live && Some(r.slot) != from).collect();
    lru.sort_by_key(|r| r.facts.last_use);
    let avail: Vec<[u64; 3]> = lru
        .iter()
        .map(|r| {
            let mut asked = r.want;
            r.facts.reclaim().map(|t| {
                let taken = asked.min(t);
                asked -= taken;
                t - taken
            })
        })
        .collect();
    let (mut asked, mut left) = (vec![0u64; lru.len()], need);
    for tier in 0..3 {
        for (a, avail) in asked.iter_mut().zip(&avail) {
            let take = left.min(avail[tier]);
            *a += take;
            left -= take;
        }
    }
    let victims = lru.iter().zip(asked).filter(|(_, bytes)| *bytes > 0);
    (victims.map(|(r, bytes)| Victim { slot: r.slot, gen: r.gen, pid: r.pid, bytes, until: 0 }).collect(), left)
}

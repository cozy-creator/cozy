//! The per-GPU ledger (design memory-varena §5.6): one memfd table, one row per process
//! holding bytes on the GPU.
//!
//! A row has one writer, its process, which claims it with an OFD lock on the row's bytes. The
//! kernel drops the lock at death, so anyone can tell a dead row by probing the lock. Rows hold
//! atomics only. Facts are a latch (two copies under one sequence): a reader never tears and
//! never waits on a writer that died mid-publish. Asks are cumulative counters. An asker adds
//! to `asked`; the owner answers by raising `handled` to what it saw and adding what it freed.
//! Nothing here touches CUDA or keeps time: stalls are judged by progress counters.

use std::collections::HashMap;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd};
use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release};
use std::sync::atomic::{fence, AtomicU32, AtomicU64};
use std::sync::{Arc, Mutex};

use cozy_platform::shm::{self, Region, RESIZE_SEALS};

use crate::{Error, Result};

const MAGIC: u64 = u64::from_le_bytes(*b"vledger1");
pub const ROWS: u32 = 64;
const HEADER: usize = 256;
const ROW: usize = 256;
const UUID: usize = 64;
const FACTS: usize = 11;

const FREE: u64 = 0;
const LIVE: u64 = 1;
const REAPING: u64 = 2;

#[repr(C)]
struct Header {
    magic: u64,
    rows: u32,
    _pad: u32,
    capacity: u64,
    uuid: [u8; UUID],
}

#[repr(C, align(256))]
struct Slot {
    /// `gen << 2 | state`. Every claim, reap and return changes it, so a reader that sees it
    /// unchanged around its reads read one row.
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
    facts: [[AtomicU64; FACTS]; 2],
}

const _: () = assert!(size_of::<Slot>() == ROW && size_of::<Header>() <= HEADER);

macro_rules! facts {
    ($($(#[$doc:meta])* $field:ident),* $(,)?) => {
        /// What a row's process publishes: bytes on this GPU, and `last_use` in CLOCK_MONOTONIC
        /// ns (`time.monotonic_ns()`).
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct Facts {
            $($(#[$doc])* pub $field: u64,)*
        }

        impl Facts {
            fn words(&self) -> [u64; FACTS] {
                [$(self.$field),*]
            }

            fn from_words([$($field),*]: [u64; FACTS]) -> Facts {
                Facts { $($field),* }
            }
        }
    };
}

facts! {
    /// CUDA context, NCCL buffers and library handles.
    context,
    /// Weight pages mapped.
    resident,
    /// Resident pages no `use` holds and nothing pins: an ask evicts them.
    cold,
    /// The warm set's (`keep`) part of `cold`: asked for after every other row's cold bytes.
    kept,
    /// Resident pages live tensors pin.
    pinned,
    torch_reserved,
    torch_allocated,
    /// Torch bytes in caches shed on an ask (part of `torch_allocated`).
    cache,
    /// varena's idle chunks.
    idle,
    /// The measured torch-allocated peak of the active requests; 0 when idle.
    active_peak,
    last_use,
}

fn sum(xs: &[u64]) -> u64 {
    xs.iter().fold(0, |s, x| s.saturating_add(*x))
}

impl Facts {
    /// Torch bytes no ask frees: live tensors outside caches, or the active peak if higher.
    fn activations(&self) -> u64 {
        self.torch_allocated.saturating_sub(self.cache).max(self.active_peak)
    }

    /// Every byte the process holds now or will for its active requests.
    pub fn held(&self) -> u64 {
        let torch = self.torch_reserved.max(self.activations().saturating_add(self.cache));
        sum(&[self.context, self.resident, self.idle, torch])
    }

    /// What no ask frees: the context, weights in use or pinned, and activations.
    pub fn unevictable(&self) -> u64 {
        sum(&[self.context, self.resident.saturating_sub(self.cold), self.activations()])
    }

    /// What an ask frees, in the order the owner frees it: idle chunks, caches and slack; cold
    /// weights outside the warm set; the warm set's cold weights.
    fn reclaim(&self) -> [u64; 3] {
        let cold = self.cold.min(self.resident);
        let kept = self.kept.min(cold);
        [self.held().saturating_sub(self.unevictable()).saturating_sub(cold), cold - kept, kept]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Live,
    /// Its process is gone (the lock dropped); its bytes count as held until reaped and returned.
    Dead,
    /// Reaped: held until the caller confirms its bytes are back.
    Reaping,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowFacts {
    pub slot: u32,
    pub gen: u64,
    pub pid: u32,
    pub state: State,
    pub facts: Facts,
    /// Bytes asked of this row and not yet answered.
    pub want: u64,
    /// Bytes this row freed for asks, cumulative.
    pub acked: u64,
    pub progress: u64,
    /// What a reaped row held when its process died.
    pub reaping: u64,
}

impl RowFacts {
    pub fn held(&self) -> u64 {
        match self.state {
            State::Reaping => self.reaping,
            _ => self.facts.held(),
        }
    }

    /// Bytes an ask can get from this row: only a live one answers.
    pub fn reclaimable(&self) -> u64 {
        match self.state {
            State::Live => self.facts.held().saturating_sub(self.facts.unevictable()),
            _ => 0,
        }
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
    /// Not free now. `victims` hold reclaimable bytes that cover the gap (least recently used
    /// first); `short` is what no ask can free: bytes in use, pinned, in activations, or in dead
    /// and reaping rows. `short == 0` once `ask` and its answers make it fit.
    After { victims: Vec<Victim>, short: u64 },
}

struct Table {
    /// This process's own description: mapped and probed, never locked.
    fd: OwnedFd,
    base: *mut u8,
    len: usize,
    rows: u32,
}

// SAFETY: the mapping is only reached through atomics (rows) or read-only data (the header).
unsafe impl Send for Table {}
// SAFETY: as above.
unsafe impl Sync for Table {}

impl Drop for Table {
    fn drop(&mut self) {
        // SAFETY: our mapping; no reference into it outlives the Table (rows hold an Arc).
        unsafe { libc::munmap(self.base.cast(), self.len) };
    }
}

fn io(what: &str) -> Error {
    Error::Io(format!("{what}: {}", std::io::Error::last_os_error()))
}

impl Table {
    fn map(fd: OwnedFd, len: usize) -> Result<Table> {
        // SAFETY: a shared mapping of a memfd whose size is (or is about to be) sealed.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return Err(io("map ledger"));
        }
        Ok(Table { fd, base: base.cast(), len, rows: 0 })
    }

    fn header(&self) -> &Header {
        // SAFETY: the header is written once, before the memfd is handed out.
        unsafe { &*(self.base as *const Header) }
    }

    fn slot(&self, i: u32) -> &Slot {
        debug_assert!(i < self.rows);
        // SAFETY: inside the mapping, 256-aligned; every field is atomic.
        unsafe { &*(self.base.add(HEADER + i as usize * ROW) as *const Slot) }
    }

    fn alive(&self, i: u32) -> Result<bool> {
        Ok(shm::region_holder(self.fd.as_fd(), HEADER + i as usize * ROW, ROW)
            .map_err(crate::os("probe ledger row"))?
            .is_some())
    }

    fn read(&self, i: u32) -> Result<Option<RowFacts>> {
        let s = self.slot(i);
        loop {
            let w = s.word.load(Acquire);
            if w & 3 == FREE {
                return Ok(None);
            }
            let state = match w & 3 {
                REAPING => State::Reaping,
                _ if self.alive(i)? => State::Live,
                _ => State::Dead,
            };
            let facts = s.facts();
            let (pid, asked, handled) = (s.pid.load(Relaxed), s.asked.load(Relaxed), s.handled.load(Relaxed));
            let (freed, progress, reaping) = (s.freed.load(Relaxed), s.progress.load(Relaxed), s.reaping.load(Relaxed));
            fence(Acquire);
            if s.word.load(Relaxed) == w {
                return Ok(Some(RowFacts {
                    slot: i,
                    gen: w >> 2,
                    pid,
                    state,
                    facts,
                    want: asked.saturating_sub(handled),
                    acked: freed,
                    progress,
                    reaping,
                }));
            }
        }
    }
}

impl Slot {
    /// The latch: copy 0 while `seq` is odd, copy 1 while it is even; readers read the other.
    fn publish(&self, f: &Facts) {
        let w = f.words();
        let s = self.seq.load(Relaxed);
        self.seq.store(s + 1, Relaxed);
        fence(Release);
        self.facts[0].iter().zip(w).for_each(|(a, v)| a.store(v, Relaxed));
        self.seq.store(s + 2, Release);
        fence(Release);
        self.facts[1].iter().zip(w).for_each(|(a, v)| a.store(v, Relaxed));
    }

    fn facts(&self) -> Facts {
        loop {
            let s = self.seq.load(Acquire);
            let copy = &self.facts[(s & 1) as usize];
            let w = std::array::from_fn(|i| copy[i].load(Relaxed));
            fence(Acquire);
            if self.seq.load(Relaxed) == s {
                return Facts::from_words(w);
            }
        }
    }

    fn ring(&self) {
        self.bell.fetch_add(1, Release);
        wake(&self.bell);
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
fn wait(_: &AtomicU32, _: u32) {
    std::thread::yield_now();
}

#[cfg(not(target_os = "linux"))]
fn wake(_: &AtomicU32) {}

/// One process's row. Dropping it releases the lock, so the row reads dead until reaped: its
/// bytes stay counted until its process is gone and the caller confirms them back.
pub struct Row {
    t: Arc<Table>,
    slot: u32,
    gen: u64,
    pid: u32,
    _lock: OwnedFd,
    seen: AtomicU64,
    writer: Mutex<()>,
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

    pub fn publish(&self, f: &Facts) {
        let _one = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let s = self.t.slot(self.slot);
        s.publish(f);
        s.progress.fetch_add(1, Release);
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
        let s = self.t.slot(self.slot);
        loop {
            let bell = s.bell.load(Acquire);
            let w = self.want();
            if w > 0 {
                return w;
            }
            wait(&s.bell, bell);
        }
    }

    /// Answers every ask `want()` saw, having freed `freed` bytes for them.
    pub fn ack(&self, freed: u64) {
        let s = self.t.slot(self.slot);
        s.handled.fetch_max(self.seen.load(Relaxed), Release);
        s.freed.fetch_add(freed, Release);
        s.progress.fetch_add(1, Release);
        s.ring();
    }
}

/// One GPU's table, mapped in this process.
pub struct Ledger {
    t: Arc<Table>,
    /// `stalled`'s last look: (slot, gen) -> progress.
    looks: Mutex<HashMap<(u32, u64), u64>>,
}

impl Ledger {
    /// A new table in a sealed-size memfd; returns it and the memfd to hand to other processes.
    pub fn create(gpu_uuid: &str, capacity: u64) -> Result<(Ledger, OwnedFd)> {
        if gpu_uuid.len() > UUID || gpu_uuid.contains('\0') {
            return Err(Error::Invalid(format!("gpu uuid {gpu_uuid:?}: at most {UUID} bytes, no NUL")));
        }
        let fd = shm::memfd_named(&format!("varena-ledger:{gpu_uuid}")).map_err(crate::os("memfd_create"))?;
        let len = HEADER + ROWS as usize * ROW;
        // SAFETY: sizing a memfd only we hold.
        if unsafe { libc::ftruncate(fd.as_raw_fd(), len as libc::off_t) } != 0 {
            return Err(io("size ledger"));
        }
        let own = shm::reopen(fd.as_fd(), true).map_err(crate::os("reopen ledger"))?;
        let mut t = Table::map(own, len)?;
        let mut uuid = [0u8; UUID];
        uuid[..gpu_uuid.len()].copy_from_slice(gpu_uuid.as_bytes());
        // SAFETY: nobody else holds this memfd yet.
        unsafe {
            (t.base as *mut Header).write(Header { magic: MAGIC, rows: ROWS, _pad: 0, capacity, uuid });
        }
        t.rows = ROWS;
        shm::seal_with(fd.as_fd(), RESIZE_SEALS).map_err(crate::os("seal ledger"))?;
        Ok((Ledger { t: Arc::new(t), looks: Mutex::default() }, fd))
    }

    /// The table behind `fd` (any description of it), mapped through a description of its own.
    pub fn adopt(fd: BorrowedFd<'_>) -> Result<Ledger> {
        let seals = shm::seals(fd).map_err(crate::os("read ledger seals"))?;
        if seals & RESIZE_SEALS != RESIZE_SEALS {
            return Err(Error::Invalid("not a ledger: its size is not sealed".into()));
        }
        let own = shm::reopen(fd, true).map_err(crate::os("reopen ledger"))?;
        // SAFETY: fstat into a zeroed struct.
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(own.as_raw_fd(), &mut st) } != 0 {
            return Err(io("stat ledger"));
        }
        let len = st.st_size as usize;
        if len < HEADER {
            return Err(Error::Invalid(format!("not a ledger: {len} bytes")));
        }
        let mut t = Table::map(own, len)?;
        let h = t.header();
        if h.magic != MAGIC || len != HEADER + h.rows as usize * ROW {
            return Err(Error::Invalid("not a varena ledger".into()));
        }
        t.rows = h.rows;
        Ok(Ledger { t: Arc::new(t), looks: Mutex::default() })
    }

    /// This process's description of the table, to pass on.
    pub fn fd(&self) -> BorrowedFd<'_> {
        self.t.fd.as_fd()
    }

    pub fn gpu_uuid(&self) -> String {
        let u = &self.t.header().uuid;
        String::from_utf8_lossy(&u[..u.iter().position(|&b| b == 0).unwrap_or(UUID)]).into_owned()
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
        let range = |i: u32, region| {
            shm::region_lock(lock.as_fd(), HEADER + i as usize * ROW, ROW, region, false)
                .map_err(crate::os("lock ledger row"))
        };
        for i in 0..self.t.rows {
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
            // A writer that died mid-publish left `seq` odd: start the new one even.
            s.seq.store((s.seq.load(Relaxed) | 1) + 1, Relaxed);
            s.facts.iter().flatten().for_each(|a| a.store(0, Relaxed));
            for a in [&s.asked, &s.handled, &s.freed, &s.progress, &s.reaping] {
                a.store(0, Relaxed);
            }
            if s.word.compare_exchange(w, gen << 2 | LIVE, Release, Relaxed).is_err() {
                range(i, Region::Unlock)?;
                continue;
            }
            let t = self.t.clone();
            return Ok(Row { t, slot: i, gen, pid, _lock: lock, seen: AtomicU64::new(0), writer: Mutex::new(()) });
        }
        Err(Error::Invalid(format!("ledger {} is full: {} rows", self.gpu_uuid(), self.t.rows)))
    }

    /// Every row in use: live, dead or reaping.
    pub fn snapshot(&self) -> Result<Vec<RowFacts>> {
        (0..self.t.rows).filter_map(|i| self.t.read(i).transpose()).collect()
    }

    /// Asks other live rows for `need` bytes: idle bytes, then cold weights outside warm sets, then
    /// warm sets' cold weights, each least recently used first, net of what others already asked
    /// of them. Sets each victim's want and rings
    /// it. The victims may cover less than `need`.
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
        Ok(s.word.load(Acquire) != v.gen << 2 | LIVE
            || s.handled.load(Acquire) >= v.until
            || !self.t.alive(v.slot)?)
    }

    /// Blocks until `answered`. A victim that dies without answering ends the wait when it is
    /// reaped.
    pub fn wait(&self, v: &Victim) -> Result<()> {
        let s = self.t.slot(v.slot);
        loop {
            let bell = s.bell.load(Acquire);
            if self.answered(v)? {
                return Ok(());
            }
            wait(&s.bell, bell);
        }
    }

    /// `Now` only if `need` bytes are free beside every byte the rows hold. Never below need.
    pub fn admit(&self, need: u64) -> Result<Admit> {
        let rows = self.snapshot()?;
        let held = rows.iter().fold(0u64, |s, r| s.saturating_add(r.held()));
        let free = self.capacity().saturating_sub(held);
        if free >= need {
            return Ok(Admit::Now);
        }
        let (victims, short) = pick(&rows, None, need - free);
        Ok(Admit::After { victims, short })
    }

    /// Marks `pid`'s dead rows reaping and returns the bytes they held, which stay counted until
    /// `returned`. Refuses a pid whose only rows are live.
    pub fn reap(&self, pid: u32) -> Result<u64> {
        let rows: Vec<_> = self.snapshot()?.into_iter().filter(|r| r.pid == pid).collect();
        let dead: Vec<_> = rows.iter().filter(|r| r.state == State::Dead).collect();
        if dead.is_empty() {
            if let Some(r) = rows.iter().find(|r| r.state == State::Live) {
                return Err(Error::Invalid(format!("pid {pid} is alive: it holds row {}", r.slot)));
            }
        }
        let mut held = 0;
        for r in dead {
            let s = self.t.slot(r.slot);
            s.reaping.store(r.facts.held(), Relaxed);
            let live = r.gen << 2 | LIVE;
            if s.word.compare_exchange(live, r.gen << 2 | REAPING, AcqRel, Relaxed).is_ok() {
                held += r.facts.held();
                s.ring();
            }
        }
        Ok(held)
    }

    /// `pid`'s reaped bytes are back on the GPU (NVML shows them): its rows are free.
    pub fn returned(&self, pid: u32) -> Result<()> {
        for i in 0..self.t.rows {
            let s = self.t.slot(i);
            let w = s.word.load(Acquire);
            if w & 3 == REAPING && s.pid.load(Relaxed) == pid && s.word.compare_exchange(w, w & !3, Release, Relaxed).is_ok() {
                s.ring();
            }
        }
        Ok(())
    }

    /// Pids of live rows asked for bytes whose progress has not moved since this handle's last
    /// look. The first look at an ask only records it.
    pub fn stalled(&self) -> Result<Vec<u32>> {
        let rows = self.snapshot()?;
        let mut looks = self.looks.lock().unwrap_or_else(|e| e.into_inner());
        let mut next = HashMap::new();
        let mut stalled = Vec::new();
        for r in rows.iter().filter(|r| r.state == State::Live && r.want > 0) {
            if looks.get(&(r.slot, r.gen)) == Some(&r.progress) {
                stalled.push(r.pid);
            }
            next.insert((r.slot, r.gen), r.progress);
        }
        *looks = next;
        Ok(stalled)
    }
}

/// Victims for `need` bytes among live rows other than `from`, and what they leave short.
fn pick(rows: &[RowFacts], from: Option<u32>, need: u64) -> (Vec<Victim>, u64) {
    let mut lru: Vec<&RowFacts> = rows.iter().filter(|r| r.state == State::Live && Some(r.slot) != from).collect();
    lru.sort_by_key(|r| r.facts.last_use);
    // What a victim was already asked takes its tiers in the order it frees them.
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
    let mut asked = vec![0u64; lru.len()];
    let mut left = need;
    for tier in 0..3 {
        for (a, avail) in asked.iter_mut().zip(&avail) {
            let take = left.min(avail[tier]);
            *a += take;
            left -= take;
        }
    }
    let victims = lru
        .iter()
        .zip(asked)
        .filter(|(_, bytes)| *bytes > 0)
        .map(|(r, bytes)| Victim { slot: r.slot, gen: r.gen, pid: r.pid, bytes, until: 0 })
        .collect();
    (victims, left)
}

//! The GPU's shared table, named per GPU and user so a standalone engine opens it without
//! TensorD. One file of shared memory: a header, then a heap of records on 256-byte cells that
//! grows with the file. A row (one per process) is held by an OFD lock that dies with it; its
//! bytes by class are a two-copy latch, published from its own table's derivation; asks are
//! cumulative counters per class ceiling, answered by the row's Rust thread. Facts (peaks,
//! rooms, dispatches) are immutable records, a newer version superseding an older one. Every
//! allocation or free moves the header's `shape`, so a reader that saw it move reads again.

use std::collections::HashMap;
use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release};
use std::sync::atomic::{fence, AtomicU32, AtomicU64};
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};
use platform::handles::Owned;

use crate::custody;
use crate::decide::{self, Admitted};
use crate::os::{self, Map};
use crate::table::{Class, Remote};
use crate::{Error, GpuId, Result};

const MAGIC: u64 = u64::from_le_bytes(*b"vledger5");
/// The header's page: the heap starts after it.
const HEADER: u64 = 4096;
/// The heap's unit: rows take two, a fact as many as its bytes need.
const CELL: u64 = 256;
const ROW_CELLS: u64 = 2;
const FACT_HEAD: u64 = 24;

const FREE: u64 = 0;
const CLAIMING: u64 = 1;
const READY: u64 = 2;
const ROW: u64 = 1;
const FACT: u64 = 2;

/// The latch: context, torch, caches, classes (7), remaining, unmeasured, running, outside.
const WORDS: usize = 14;

#[repr(C)]
struct Header {
    magic: u64,
    capacity: AtomicU64,
    /// The heap's end, a byte offset; the file is at least this long.
    end: AtomicU64,
    /// Moves on every allocation or free.
    shape: AtomicU64,
    /// Device bytes no row accounts for, as the last executor decision measured them.
    foreign: AtomicU64,
    /// A futex word rung when room frees, a row dies or a peak is recorded.
    freed: AtomicU32,
    _pad: u32,
    uuid_len: u64,
    uuid: [u8; 64],
}

#[repr(C)]
struct RowRec {
    word: AtomicU64,
    pid: AtomicU32,
    /// The ask thread's wait word, rung by askers.
    bell: AtomicU32,
    _pad: [u32; 2],
    /// Asks rung, and the ask thread's wakes.
    rung: AtomicU64,
    ran: AtomicU64,
    /// `ran + 1` as an asker saw it when it rang again and the thread had not run since its
    /// first ring; the row is stalled while `ran` has not moved past it.
    suspect: AtomicU64,
    /// Cumulative bytes asked and handled, per ceiling (N, D).
    asked: [AtomicU64; 2],
    handled: [AtomicU64; 2],
    /// Cumulative bytes freed by answers.
    freed: AtomicU64,
    /// Kept pages no live process maps (the custody holder's row).
    orphans: AtomicU64,
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

fn word(cells: u64, kind: u64, state: u64) -> u64 {
    cells << 32 | kind << 8 | state
}

/// The counter of `pair` for asks under `c`: N's, or D's.
fn by_ceiling(pair: &[AtomicU64; 2], c: Class) -> &AtomicU64 {
    let [n, d] = pair;
    if c >= Class::D { d } else { n }
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
}

impl Facts {
    fn words(&self) -> [u64; WORDS] {
        let c = self.classes;
        [self.context, self.torch, self.caches, c[0], c[1], c[2], c[3], c[4], c[5], c[6], self.remaining, self.unmeasured, self.running, self.outside]
    }

    fn of(w: [u64; WORDS]) -> Facts {
        let [context, torch, caches, c0, c1, c2, c3, c4, c5, c6, remaining, unmeasured, running, outside] = w;
        Facts { context, torch, caches, classes: [c0, c1, c2, c3, c4, c5, c6], remaining, unmeasured, running, outside }
    }

    fn held(&self) -> u64 {
        self.context + self.torch + self.classes.iter().sum::<u64>()
    }
}

/// A row as read.
#[derive(Clone, Debug)]
pub(crate) struct RowView {
    pub off: u64,
    pub pid: u32,
    pub live: bool,
    pub facts: Facts,
    pub orphans: u64,
    pub want: u64,
    pub stalled: bool,
}

/// The table mapped in this process.
pub(crate) struct Shared {
    /// This process's description: mapped and used for the heap lock, never for a row lock.
    fd: Owned,
    map: RwLock<Arc<Map>>,
    pub name: String,
    events: Option<Owned>,
}

impl Shared {
    /// The table behind `fd` (a grant's), initialised if it is new. The first opener sets the
    /// capacity; a later one can only lower it.
    pub fn adopt(fd: Owned, gpu: &GpuId, capacity: u64, events: Option<Owned>) -> Result<Shared> {
        let fd = os::reopen(&fd, true)?;
        // Initialisation is one writer's: the heap lock serialises it.
        os::lock(&fd, 0, 8, true)?;
        let init = (|| -> Result<Map> {
            if os::size(&fd)? < HEADER + CELL {
                os::resize(&fd, HEADER + 64 * CELL)?;
            }
            let map = os::map(&fd, os::size(&fd)? as usize, true)?;
            // SAFETY: the header page is mapped, aligned, and its fields are atomics or written
            // once under the heap lock below.
            let h = unsafe { &mut *(map.base as *mut Header) };
            if h.magic != MAGIC {
                let uuid = gpu.uuid.as_bytes();
                h.uuid.iter_mut().zip(uuid).for_each(|(d, s)| *d = *s);
                h.uuid_len = uuid.len().min(64) as u64;
                h.capacity.store(capacity, Relaxed);
                h.end.store(HEADER, Relaxed);
                fence(Release);
                h.magic = MAGIC;
            }
            h.capacity.fetch_min(capacity, AcqRel);
            Ok(map)
        })();
        os::unlock(&fd, 0, 8)?;
        let map = init?;
        Ok(Shared { fd, map: RwLock::new(Arc::new(map)), name: gpu.uuid.clone(), events })
    }

    pub fn descriptor(&self) -> Result<Owned> {
        os::reopen(&self.fd, true)
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

    /// Room freed, a row died or a peak was recorded: wake waiters and TensorD.
    pub fn ring(&self) {
        if let Ok(m) = self.map() {
            os::ring(&header(&m).freed);
        }
        if let Some(e) = &self.events {
            os::signal(e);
        }
    }

    /// Every record: (offset, word), read again if the heap's shape moved meanwhile.
    fn records<T>(&self, mut f: impl FnMut(&Map, u64, u64) -> Option<T>) -> Result<Vec<T>> {
        loop {
            let m = self.map()?;
            let h = header(&m);
            let shape = h.shape.load(Acquire);
            let end = h.end.load(Acquire).min(m.len as u64);
            let mut out = vec![];
            let mut at = HEADER;
            let mut torn = false;
            while at < end {
                let Some(w) = atomic(&m, at) else { break };
                let w = w.load(Acquire);
                let cells = w >> 32;
                if cells == 0 || at + cells * CELL > end {
                    torn = true;
                    break;
                }
                if w & 0xff == READY {
                    out.extend(f(&m, at, w));
                }
                at += cells * CELL;
            }
            fence(Acquire);
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

    /// Allocates `cells` under the heap lock: the first free run that fits, else the end.
    fn alloc(&self, cells: u64, kind: u64, init: impl FnOnce(&Map, u64)) -> Result<u64> {
        os::lock(&self.fd, 0, 8, true)?;
        let at = self.alloc_locked(cells, kind, init);
        os::unlock(&self.fd, 0, 8)?;
        at
    }

    fn alloc_locked(&self, cells: u64, kind: u64, init: impl FnOnce(&Map, u64)) -> Result<u64> {
        let m = self.map()?;
        let h = header(&m);
        let end = h.end.load(Acquire);
        // First fit over free runs, merging neighbours as it walks.
        let (mut at, mut run_start, mut run) = (HEADER, None, 0u64);
        let mut found = None;
        while at < end {
            let w = atomic(&m, at).ok_or_else(|| Error::Internal("ledger heap outside its map".into()))?.load(Acquire);
            let n = (w >> 32).max(1);
            if w & 0xff == FREE {
                let start = *run_start.get_or_insert(at);
                run += n;
                if run >= cells {
                    found = Some(start);
                    break;
                }
            } else {
                run_start = None;
                run = 0;
            }
            at += n * CELL;
        }
        let mut grown = None;
        let m = match found {
            Some(_) => m,
            None => {
                let start = run_start.unwrap_or(end);
                let need = start + cells * CELL;
                let size = os::size(&self.fd)?;
                if need > size {
                    os::resize(&self.fd, need.max(size * 2))?;
                }
                found = Some(start);
                grown = (need > end).then_some(need);
                if m.len as u64 >= need {
                    m
                } else {
                    let mut slot = self.map.write();
                    *slot = Arc::new(os::map(&self.fd, os::size(&self.fd)? as usize, true)?);
                    slot.clone()
                }
            }
        };
        let start = found.unwrap_or(end);
        let h = header(&m);
        // A free run longer than asked keeps its tail free.
        let tail = run.saturating_sub(cells);
        if tail > 0 {
            let w = atomic(&m, start + cells * CELL).ok_or_else(|| Error::Internal("ledger tail".into()))?;
            w.store(word(tail, 0, FREE), Release);
        }
        let w = atomic(&m, start).ok_or_else(|| Error::Internal("ledger record".into()))?;
        w.store(word(cells, kind, CLAIMING), Release);
        init(&m, start);
        if let Some(end) = grown {
            h.end.store(end, Release);
        }
        h.shape.fetch_add(1, AcqRel);
        Ok(start)
    }

    fn free(&self, at: u64) -> Result<()> {
        os::lock(&self.fd, 0, 8, true)?;
        let m = self.map();
        if let Ok(m) = &m {
            if let Some(w) = atomic(m, at) {
                let cells = w.load(Acquire) >> 32;
                w.store(word(cells, 0, FREE), Release);
                header(m).shape.fetch_add(1, AcqRel);
            }
        }
        os::unlock(&self.fd, 0, 8)?;
        m.map(|_| ())
    }

    // ---- rows

    /// A row for `pid`, held by an OFD lock on a description of its own.
    pub fn claim(self: &Arc<Self>, pid: u32) -> Result<Row> {
        let lock = os::reopen(&self.fd, true)?;
        let at = self.alloc(ROW_CELLS, ROW, |m, at| {
            if let Some(r) = row(m, at) {
                r.pid.store(pid, Relaxed);
                [&r.rung, &r.ran, &r.suspect, &r.freed, &r.orphans, &r.seq].iter().for_each(|a| a.store(0, Relaxed));
                r.asked.iter().chain(&r.handled).chain(r.latch.iter().flatten()).for_each(|a| a.store(0, Relaxed));
            }
        })?;
        let locked = os::lock(&lock, at, ROW_CELLS * CELL, false);
        let m = self.map()?;
        let rec = row(&m, at).ok_or_else(|| Error::Internal("ledger row".into()))?;
        if !matches!(locked, Ok(true)) {
            self.free(at)?;
            return Err(Error::Internal(format!("ledger row lock: {:?}", locked.err())));
        }
        rec.word.store(word(ROW_CELLS, ROW, READY), Release);
        Ok(Row { shared: self.clone(), at, _lock: lock, words: Mutex::new([0; WORDS]) })
    }

    pub fn rows(&self) -> Result<Vec<RowView>> {
        let views = self.records(|m, at, w| {
            if (w >> 8) & 0xff != ROW {
                return None;
            }
            let r = row(m, at)?;
            let facts = Facts::of(latch_read(&r.seq, &r.latch));
            let want = r.asked.iter().zip(&r.handled).map(|(a, h)| a.load(Acquire).saturating_sub(h.load(Acquire))).sum();
            let suspect = r.suspect.load(Acquire);
            let stalled = suspect != 0 && suspect == r.ran.load(Acquire) + 1;
            Some(RowView { off: at, pid: r.pid.load(Acquire), live: true, facts, orphans: r.orphans.load(Acquire), want, stalled })
        })?;
        views.into_iter().map(|mut v| -> Result<RowView> {
            v.live = os::held(&self.fd, v.off, ROW_CELLS * CELL)?;
            Ok(v)
        }).collect()
    }

    /// Frees dead rows (of `pid`, or any): their holders are gone, so the lock is free.
    pub fn reap(&self, pid: Option<u32>) -> Result<bool> {
        let mut any = false;
        for r in self.rows()? {
            if !r.live && pid.is_none_or(|p| p == r.pid) {
                let m = self.map()?;
                let _ = m;
                self.free(r.off)?;
                any = true;
            }
        }
        if any {
            self.ring();
        }
        Ok(any)
    }

    /// Asks the row at `at` for `bytes` under `ceiling`; returns what its `handled` must reach.
    pub fn ask(&self, at: u64, ceiling: Class, bytes: u64) -> Result<u64> {
        let m = self.map()?;
        let r = row(&m, at).ok_or_else(|| Error::Invalid("no such row".into()))?;
        let until = by_ceiling(&r.asked, ceiling).fetch_add(bytes, AcqRel) + bytes;
        r.rung.fetch_add(1, AcqRel);
        os::ring(&r.bell);
        Ok(until)
    }

    /// Waits for the row's answer (its Rust thread, never Python), or its death, woken by any
    /// ledger event. Each wake without an answer rings again; when the thread has not run
    /// across two such rings the row is marked, for TensorD's `stalled`.
    pub fn answered(&self, at: u64, ceiling: Class, until: u64) -> Result<u64> {
        let m = self.map()?;
        let r = row(&m, at).ok_or_else(|| Error::Invalid("no such row".into()))?;
        let freed = r.freed.load(Acquire);
        let mut ran = r.ran.load(Acquire);
        let (mut woke, mut misses) = (false, 0);
        os::until(&header(&m).freed, || {
            let done = by_ceiling(&r.handled, ceiling).load(Acquire) >= until || r.word.load(Acquire) & 0xff != READY;
            if done || !os::held(&self.fd, at, ROW_CELLS * CELL).unwrap_or(false) {
                return Some(());
            }
            if std::mem::replace(&mut woke, true) {
                let now = r.ran.load(Acquire);
                misses = if now == ran { misses + 1 } else { 0 };
                ran = now;
                if misses >= 2 {
                    r.suspect.store(ran + 1, Release);
                }
                r.rung.fetch_add(1, AcqRel);
                os::ring(&r.bell);
                if let Some(e) = &self.events {
                    os::signal(e);
                }
            }
            None
        });
        Ok(r.freed.load(Acquire).saturating_sub(freed))
    }

    // ---- facts

    /// The newest value of each key under `prefix`.
    pub fn facts(&self, prefix: &str) -> Result<Vec<(String, Vec<u8>)>> {
        let found = self.records(|m, at, w| {
            if (w >> 8) & 0xff != FACT {
                return None;
            }
            let (key, value, version) = fact(m, at)?;
            key.starts_with(prefix).then_some((key, value, version, at))
        })?;
        let mut newest: HashMap<String, (u64, Vec<u8>)> = HashMap::new();
        for (key, value, version, _) in found {
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
        let old: Vec<(u64, u64)> = self.records(|m, at, w| {
            if (w >> 8) & 0xff != FACT {
                return None;
            }
            let (k, _, version) = fact(m, at)?;
            (k == key).then_some((at, version))
        })?;
        if let Some(value) = value {
            let version = old.iter().map(|(_, v)| *v).max().unwrap_or(0) + 1;
            let bytes = FACT_HEAD + key.len() as u64 + value.len() as u64;
            let at = self.alloc(bytes.div_ceil(CELL), FACT, |m, at| {
                // SAFETY: a CLAIMING record of `bytes.div_ceil(CELL)` cells inside the map: no reader parses it.
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
            let m = self.map()?;
            let w = atomic(&m, at).ok_or_else(|| Error::Internal("ledger fact".into()))?;
            w.store(word(bytes.div_ceil(CELL), FACT, READY), Release);
        }
        for (at, _) in old {
            self.free(at)?;
        }
        Ok(())
    }
}

/// One process's row. Dropping it drops the lock: the row reads dead until reaped.
pub(crate) struct Row {
    shared: Arc<Shared>,
    at: u64,
    _lock: Owned,
    /// The latch's words as last written; one writer at a time.
    words: Mutex<[u64; WORDS]>,
}

impl Row {
    pub fn at(&self) -> u64 {
        self.at
    }

    fn rec<T>(&self, f: impl FnOnce(&RowRec) -> T) -> Option<T> {
        let m = self.shared.map().ok()?;
        row(&m, self.at).map(f)
    }

    pub fn publish(&self, facts: &Facts) {
        let mut words = self.words.lock();
        *words = facts.words();
        self.rec(|r| latch_write(&r.seq, &r.latch, &words));
    }

    pub fn set_orphans(&self, bytes: u64) {
        self.rec(|r| r.orphans.store(bytes, Release));
    }

    /// Blocks until something is asked of this row or `stop` says so; returns (ceiling, bytes)
    /// per ceiling asked.
    pub fn wait_asked(&self, stop: &std::sync::atomic::AtomicBool) -> Vec<(Class, u64, u64)> {
        let Ok(m) = self.shared.map() else { return vec![] };
        let Some(r) = row(&m, self.at) else { return vec![] };
        os::until(&r.bell, || {
            r.ran.fetch_add(1, AcqRel);
            if stop.load(Acquire) {
                return Some(vec![]);
            }
            let wants: Vec<(Class, u64, u64)> = [Class::N, Class::D]
                .into_iter()
                .filter_map(|c| {
                    let asked = by_ceiling(&r.asked, c).load(Acquire);
                    let want = asked.saturating_sub(by_ceiling(&r.handled, c).load(Acquire));
                    (want > 0).then_some((c, want, asked))
                })
                .collect();
            (!wants.is_empty()).then_some(wants)
        })
    }

    /// Answers asks up to `asked` under `ceiling`, having freed `freed` bytes.
    pub fn answer(&self, ceiling: Class, asked: u64, freed: u64) {
        self.rec(|r| {
            r.freed.fetch_add(freed, AcqRel);
            by_ceiling(&r.handled, ceiling).fetch_max(asked, AcqRel);
        });
        // Askers wait on the ledger's events.
        self.shared.ring();
    }

    /// Wakes the ask thread (to stop it).
    pub fn wake(&self) {
        self.rec(|r| os::ring(&r.bell));
    }
}

impl Drop for Row {
    fn drop(&mut self) {
        // A clean exit frees the row now; a killed process leaves it for a reap.
        if self.shared.free(self.at).is_ok() {
            self.shared.ring();
        }
    }
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

/// The other rows as the decision procedure reads them, plus orphan and held totals.
pub(crate) fn remotes(rows: &[RowView], own: u64) -> (Vec<Remote>, u64) {
    let orphans = rows.iter().map(|r| r.orphans).sum();
    let others = rows
        .iter()
        .filter(|r| r.off != own)
        .map(|r| Remote {
            row: (r.off / CELL) as u32,
            running: r.facts.running > 0 || !r.live,
            context: r.facts.context,
            torch: r.facts.torch,
            caches: r.facts.caches,
            classes: r.facts.classes,
        })
        .collect();
    (others, orphans)
}

/// A row's offset from its id in [`Remote::row`].
pub(crate) fn row_offset(id: u32) -> u64 {
    id as u64 * CELL
}

/// Every row's held bytes, for foreign bytes.
pub(crate) fn held(rows: &[RowView]) -> u64 {
    rows.iter().map(|r| r.facts.held() + r.orphans).sum()
}

// ---- TensorD's side

/// What TensorD hands an executor for its [`crate::Memory`]: the table and TensorD's eventfd.
/// Fields are public so a spawner can pass them across exec.
pub struct Grant {
    pub table: Owned,
    pub events: Owned,
}

impl Grant {
    /// `gpu`'s table by name, created when absent, for a standalone engine: no TensorD reads
    /// its events. `limit` lowers the capacity (the first opener sets it, a later one can only
    /// lower it).
    pub fn open(gpu: &GpuId, limit: Option<u64>) -> Result<Grant> {
        let capacity = limit.map_or(gpu.memory, |l| l.min(gpu.memory));
        let (table, _) = os::shared(&gpu.uuid, true)?;
        let events = os::event()?;
        let shared = Shared::adopt(os::dup(&table)?, gpu, capacity, None)?;
        drop(shared);
        Ok(Grant { table, events })
    }
}

/// One row as [`Ledger::status`] shows it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RowStatus {
    pub pid: u32,
    pub live: bool,
    pub context: u64,
    /// Bytes by class (F unused, O, S, N, D, W, P).
    pub classes: [u64; 7],
    /// The largest remaining working set of each running request, summed.
    pub remaining: u64,
    /// Running requests with no measured working set.
    pub unmeasured: u32,
    /// Bytes asked of it and not yet answered.
    pub asked: u64,
}

/// The table at a glance (display only).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Status {
    pub capacity: u64,
    pub free: u64,
    /// Bytes on the device no row accounts for, measured.
    pub foreign: u64,
    pub rows: Vec<RowStatus>,
    /// Request id → `<package>|<shape>`.
    pub dispatched: Vec<(String, String)>,
    /// Rows whose ask thread did not run when rung.
    pub stalled: Vec<u32>,
}

/// A measured peak, as TensorD persists it per GPU model.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Peak {
    /// `<package>|<shape>` (a request's largest working set) or `<set>|<shape>|<cuts>` (a
    /// pass's room and its sites).
    pub key: String,
    pub bytes: u64,
    /// Each cut site's peak, for a pass.
    pub sites: Vec<(String, u64)>,
}

/// A pass's measured room, as the ledger keeps it under `room/<set>|<shape>|<cuts>`.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Room {
    /// Allocated peak less what was allocated at the pass's start.
    pub delta: u64,
    pub slack: u64,
    pub context: u64,
    pub sites: Vec<(String, u64)>,
}

impl Room {
    pub fn bytes(&self) -> u64 {
        self.delta + self.slack + self.context
    }
}

/// TensorD's handle on one GPU's table.
pub struct Ledger {
    shared: Arc<Shared>,
    /// TensorD's own row: custody's orphans are published in it.
    _row: Arc<Row>,
    events: Owned,
    custody: custody::Client,
}

impl Ledger {
    /// The GPU's table, named per GPU and user; capacity = min(card, limit). TensorD serves
    /// custody when it holds the service lock (it does when it starts first).
    pub fn create(gpu: &GpuId, limit: Option<u64>) -> Result<Ledger> {
        let capacity = limit.map_or(gpu.memory, |l| l.min(gpu.memory));
        let events = os::event()?;
        let (table, _) = os::shared(&gpu.uuid, true)?;
        let shared = Arc::new(Shared::adopt(table, gpu, capacity, Some(os::dup(&events)?))?);
        shared.reap(None)?;
        let row = Arc::new(shared.claim(std::process::id())?);
        let custody = custody::Client::open(&shared.name, &row)?;
        Ok(Ledger { shared, _row: row, events, custody })
    }

    /// The descriptors executor `pid` opens its memory with.
    pub fn grant(&self, _pid: u32) -> Result<Grant> {
        Ok(Grant { table: self.shared.descriptor()?, events: os::dup(&self.events)? })
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
            .map(|r| Admitted { context: r.facts.context, outside: r.facts.outside, remaining: r.facts.remaining, running: r.facts.running as u32, unmeasured: r.facts.unmeasured as u32 })
            .collect();
        let dispatched = self.shared.facts("dispatch/")?.into_iter().map(|(_, key)| self.peak(&String::from_utf8_lossy(&key))).collect::<Result<_>>()?;
        Ok((self.shared.capacity()?, self.shared.foreign()?, admitted, dispatched))
    }

    fn peak(&self, key: &str) -> Result<Option<u64>> {
        let v = self.shared.fact(&format!("peak/{key}"))?;
        Ok(v.and_then(|v| v.try_into().ok()).map(u64::from_le_bytes))
    }

    /// Whether a request of `package` at `shape` can start now beside everything running.
    pub fn fits(&self, package: &str, shape: &str) -> bool {
        let fit = || -> Result<bool> {
            let (capacity, foreign, rows, dispatched) = self.admitted()?;
            Ok(decide::fits(capacity, foreign, &rows, &dispatched, self.peak(&format!("{package}|{shape}"))?))
        };
        fit().unwrap_or(false)
    }

    /// A multi-GPU group: dispatches `request` on every GPU of `group` if it fits on all, else
    /// on none.
    pub fn admit(group: &[&Ledger], request: &str, package: &str, shape: &str) -> Result<bool> {
        if !group.iter().all(|l| l.fits(package, shape)) {
            return Ok(false);
        }
        for l in group {
            l.dispatch(request, package, shape)?;
        }
        Ok(true)
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
            let bytes = v.try_into().map(u64::from_le_bytes).unwrap_or(0);
            out.push(Peak { key: key.trim_start_matches("peak/").to_string(), bytes, sites: vec![] });
        }
        for (key, v) in self.shared.facts("room/").unwrap_or_default() {
            if let Ok(room) = serde_json::from_slice::<Room>(&v) {
                out.push(Peak { key: key.trim_start_matches("room/").to_string(), bytes: room.bytes(), sites: room.sites });
            }
        }
        out
    }

    /// Peaks persisted earlier on this machine, for this GPU model: a request's (`a|b`) as its
    /// largest W, a pass's (`a|b|c`) as its room.
    pub fn restore(&self, peaks: &[Peak]) -> Result<()> {
        for p in peaks {
            match p.key.matches('|').count() {
                1 => self.shared.put(&format!("peak/{}", p.key), Some(&p.bytes.to_le_bytes()))?,
                _ => {
                    let room = Room { delta: p.bytes, slack: 0, context: 0, sites: p.sites.clone() };
                    self.shared.put(&format!("room/{}", p.key), Some(&os::encode(&room)?))?;
                }
            }
        }
        Ok(())
    }

    pub fn status(&self) -> Status {
        let Ok(rows) = self.shared.rows() else { return Status::default() };
        let capacity = self.shared.capacity().unwrap_or(0);
        let foreign = self.shared.foreign().unwrap_or(0);
        let stalled = rows.iter().filter(|r| r.live && r.stalled).map(|r| r.pid).collect();
        let dispatched = self.shared.facts("dispatch/").unwrap_or_default();
        Status {
            capacity,
            free: capacity.saturating_sub(foreign + held(&rows)),
            foreign,
            rows: rows
                .iter()
                .map(|r| RowStatus { pid: r.pid, live: r.live, context: r.facts.context, classes: r.facts.classes, remaining: r.facts.remaining, unmeasured: r.facts.unmeasured as u32, asked: r.want })
                .collect(),
            dispatched: dispatched.into_iter().map(|(k, v)| (k.trim_start_matches("dispatch/").to_string(), String::from_utf8_lossy(&v).into_owned())).collect(),
            stalled,
        }
    }

    /// Executors whose ask thread did not run when an asker rang it twice: TensorD ends them.
    /// No side effect: any caller may read it at any time.
    pub fn stalled(&self) -> Vec<u32> {
        self.shared.rows().unwrap_or_default().into_iter().filter(|r| r.live && r.stalled).map(|r| r.pid).collect()
    }

    /// Readable when room frees, a row dies or a peak is recorded; reading it clears it.
    pub fn events(&self) -> &Owned {
        &self.events
    }

    /// A GPU fault or sticky error: the GPU's kept pages go.
    pub fn fault(&self) {
        let _ = self.custody.fault();
        self.shared.ring();
    }
}

impl Drop for Ledger {
    fn drop(&mut self) {
        os::unlink(&self.shared.name);
    }
}

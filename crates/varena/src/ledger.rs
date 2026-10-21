//! The GPU's shared table, named per GPU and user so a standalone engine opens it without
//! TensorD. One file of shared memory: a header, then a heap of records on 256-byte cells that
//! grows with the file. Each record's word carries its size, a generation, its kind and state,
//! and a record is only ever freed by comparing that word under the heap lock, so a stale
//! reader can never free a newer record at the same offset. A row (one per process) is held by
//! an OFD lock that dies with it; its bytes by class are a two-copy latch, published from its
//! own table's derivation. Asks go to a row's own ask thread over a socket named by the row's
//! offset and generation: the answer or the victim's death ends the wait, and a reused offset
//! reads as gone. Facts (peaks, rooms, dispatches) are immutable records, a newer version
//! superseding an older one.

use std::collections::HashMap;
use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release};
use std::sync::atomic::{fence, AtomicU32, AtomicU64};
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};
use platform::handles::Owned;
use serde::{Deserialize, Serialize};

use crate::custody;
use crate::decide::{self, Admitted};
use crate::os::{self, Listener, Map};
use crate::table::{Class, Remote};
use crate::{Error, GpuId, Result};

const MAGIC: u64 = u64::from_le_bytes(*b"vledger6");
/// The header's page: the heap starts after it, and grows from there.
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
    /// Moves on every allocation or free; its low bits are a new record's generation.
    shape: AtomicU64,
    /// Device bytes no row accounts for, as the last executor decision measured them.
    foreign: AtomicU64,
    /// 0: shared (the machine's owner uses it too), 1: dedicated.
    mode: AtomicU32,
}

#[repr(C)]
struct RowRec {
    word: AtomicU64,
    pid: AtomicU32,
    /// Its ask thread's id: TensorD's `stalled` reads that thread's state.
    tid: AtomicU32,
    /// Bytes asked of it and not yet answered (display).
    asked: AtomicU64,
    /// Kept pages no live process maps (the custody service's row).
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
    pub word: u64,
    pub pid: u32,
    pub live: bool,
    pub facts: Facts,
    pub orphans: u64,
    pub asked: u64,
    pub tid: u32,
}

impl RowView {
    pub fn id(&self) -> u64 {
        row_id(self.off, self.word)
    }
}

/// What a row's ask thread is asked.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum Ask {
    /// Device bytes under a ceiling (Class::N or Class::D, as its index).
    Room { ceiling: usize, bytes: u64 },
    /// The RAM tier gives back a host layout: unpin and unmap it.
    Host { digest: String },
}

#[derive(Debug, Serialize, Deserialize)]
struct Answer {
    freed: u64,
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
        os::own(&fd)?;
        // Initialisation is one writer's: the heap lock serialises it.
        os::lock(&fd, 0, 8, true)?;
        let init = (|| -> Result<Map> {
            if os::size(&fd)? < HEADER {
                os::resize(&fd, HEADER)?;
            }
            let map = os::map(&fd, os::size(&fd)? as usize, true)?;
            // SAFETY: the header page is mapped and aligned; its fields are atomics or written
            // once under the heap lock before `magic`.
            let h = unsafe { &mut *(map.base as *mut Header) };
            if h.magic != MAGIC {
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

    pub fn shared_mode(&self) -> Result<bool> {
        let m = self.map()?;
        Ok(header(&m).mode.load(Acquire) == 0)
    }

    fn set_mode(&self, mode: Mode) -> Result<()> {
        let m = self.map()?;
        header(&m).mode.store(u32::from(mode == Mode::Dedicated), Release);
        Ok(())
    }

    /// Room freed, a row died or a peak was recorded: TensorD's event.
    pub fn ring(&self) {
        if let Some(e) = &self.events {
            os::signal(e);
        }
    }

    /// Every READY record, read again if the heap's shape moved meanwhile.
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
                if cells(w) == 0 || at + cells(w) * CELL > end {
                    torn = true;
                    break;
                }
                if state(w) == READY {
                    out.extend(f(&m, at, w));
                }
                at += cells(w) * CELL;
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

    /// Allocates `n` cells under the heap lock: the first free run that fits, else the end.
    /// Returns the record's offset and its READY word.
    fn alloc(&self, n: u64, kind: u64, init: impl FnOnce(&Map, u64)) -> Result<(u64, u64)> {
        os::lock(&self.fd, 0, 8, true)?;
        let at = self.alloc_locked(n, kind, init);
        os::unlock(&self.fd, 0, 8)?;
        at
    }

    fn alloc_locked(&self, n: u64, kind: u64, init: impl FnOnce(&Map, u64)) -> Result<(u64, u64)> {
        let m = self.map()?;
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
        Ok((start, word(n, g, kind, READY)))
    }

    /// Frees the record at `at` if its word is still `seen` (and, `dead`, no one holds its row
    /// lock): a stale reader frees nothing newer.
    fn free(&self, at: u64, seen: u64, dead: bool) -> Result<bool> {
        os::lock(&self.fd, 0, 8, true)?;
        let freed = (|| -> Result<bool> {
            let m = self.map()?;
            let Some(w) = atomic(&m, at) else { return Ok(false) };
            if w.load(Acquire) != seen || (dead && os::held(&self.fd, at, ROW_CELLS * CELL)?) {
                return Ok(false);
            }
            w.store(word(cells(seen), 0, 0, FREE), Release);
            header(&m).shape.fetch_add(1, AcqRel);
            Ok(true)
        })();
        os::unlock(&self.fd, 0, 8)?;
        freed
    }

    // ---- rows

    /// A row for `pid`, held by an OFD lock on a description of its own.
    pub fn claim(self: &Arc<Self>, pid: u32) -> Result<Row> {
        let lock = os::reopen(&self.fd, true)?;
        let (at, ready) = self.alloc(ROW_CELLS, ROW, |m, at| {
            if let Some(r) = row(m, at) {
                r.pid.store(pid, Relaxed);
                r.tid.store(0, Relaxed);
                [&r.asked, &r.orphans, &r.seq].iter().for_each(|a| a.store(0, Relaxed));
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
        Ok(Row { shared: self.clone(), at, word: ready, _lock: lock, words: Mutex::new([0; WORDS]) })
    }

    pub fn rows(&self) -> Result<Vec<RowView>> {
        let views = self.records(|m, at, w| {
            if kind(w) != ROW {
                return None;
            }
            let r = row(m, at)?;
            let facts = Facts::of(latch_read(&r.seq, &r.latch));
            Some(RowView { off: at, word: w, pid: r.pid.load(Acquire), live: true, facts, orphans: r.orphans.load(Acquire), asked: r.asked.load(Acquire), tid: r.tid.load(Acquire) })
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
        for r in self.rows()? {
            if !r.live && pid.is_none_or(|p| p == r.pid) {
                any |= self.free(r.off, r.word, true)?;
            }
        }
        if any {
            self.ring();
        }
        Ok(any)
    }

    /// Asks row `id` for `bytes` under `ceiling`: its ask thread answers from its pages at once.
    /// Returns what it freed; a dead or reused row frees nothing.
    pub fn ask(&self, id: u64, ceiling: Class, bytes: u64) -> Result<u64> {
        let at = (id >> 16) * CELL;
        let m = self.map()?;
        let Some(r) = row(&m, at) else { return Ok(0) };
        let w = r.word.load(Acquire);
        if state(w) != READY || kind(w) != ROW || row_id(at, w) != id {
            return Ok(0);
        }
        r.asked.fetch_add(bytes, AcqRel);
        if let Some(e) = &self.events {
            os::signal(e);
        }
        let freed = match os::connect(&ask_name(&self.name, id))? {
            Some(stream) => os::call::<_, Answer>(&stream, &Ask::Room { ceiling: ceiling.index(), bytes }, None).map_or(0, |(a, _)| a.freed),
            None => 0,
        };
        r.asked.fetch_sub(bytes, AcqRel);
        Ok(freed)
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
        let old: Vec<(u64, u64, u64)> = self.records(|m, at, w| {
            if kind(w) != FACT {
                return None;
            }
            let (k, _, version) = fact(m, at)?;
            (k == key).then_some((at, w, version))
        })?;
        if let Some(value) = value {
            let version = old.iter().map(|(.., v)| *v).max().unwrap_or(0) + 1;
            let bytes = FACT_HEAD + key.len() as u64 + value.len() as u64;
            let (at, ready) = self.alloc(bytes.div_ceil(CELL), FACT, |m, at| {
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
            let m = self.map()?;
            atomic(&m, at).ok_or_else(|| Error::Internal("ledger fact".into()))?.store(ready, Release);
        }
        for (at, w, _) in old {
            self.free(at, w, false)?;
        }
        Ok(())
    }
}

fn ask_name(gpu: &str, id: u64) -> String {
    format!("ask-{gpu}-{id}")
}

/// Asks the row whose ask socket is `name` to give back host layout `digest`; returns the bytes
/// it unpinned and unmapped. Its reply or its death ends the wait.
pub(crate) fn release_host(name: &str, digest: &str) -> u64 {
    match os::connect(name) {
        Ok(Some(stream)) => os::call::<_, Answer>(&stream, &Ask::Host { digest: digest.into() }, None).map_or(0, |(a, _)| a.freed),
        _ => 0,
    }
}

/// One process's row. Dropping it publishes zeros, closes its asks and frees it; a killed
/// process's row reads dead until reaped.
pub(crate) struct Row {
    shared: Arc<Shared>,
    at: u64,
    word: u64,
    _lock: Owned,
    /// The latch's words as last written; one writer at a time.
    words: Mutex<[u64; WORDS]>,
}

impl Row {
    pub fn at(&self) -> u64 {
        self.at
    }

    pub fn id(&self) -> u64 {
        row_id(self.at, self.word)
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

    /// The custody service's orphans; a change is a ledger event.
    pub fn set_orphans(&self, bytes: u64) {
        if self.rec(|r| r.orphans.swap(bytes, AcqRel) != bytes).unwrap_or(false) {
            self.shared.ring();
        }
    }

    /// Binds this row's ask socket: its owner's ask thread serves it.
    pub fn listen(&self) -> Result<Listener> {
        os::listen(&self.ask_name())?.ok_or_else(|| Error::Internal("a row's ask socket is taken".into()))
    }

    /// Its ask socket's name, for services that ask it (the RAM tier).
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
        let _ = os::connect(&ask_name(&self.shared.name, self.id()));
    }

    /// Serves one asker: reads its ask and replies with what `answer` freed.
    pub fn serve(&self, stream: &os::Stream, answer: impl FnOnce(Ask) -> u64) -> Result<()> {
        let Some((bytes, _)) = os::recv(stream)? else { return Ok(()) };
        let freed = answer(os::decode(&bytes)?);
        os::send(stream, &(os::encode(&Answer { freed })?, None))?;
        self.shared.ring();
        Ok(())
    }
}

impl Drop for Row {
    fn drop(&mut self) {
        self.publish(&Facts::default());
        self.set_orphans(0);
        if self.shared.free(self.at, self.word, false).unwrap_or(false) {
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
            caches: r.facts.caches,
            classes: r.facts.classes,
        })
        .collect();
    (others, orphans)
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
        let table = os::shared(&gpu.uuid, true)?;
        let events = os::event()?;
        drop(Shared::adopt(os::dup(&table)?, gpu, capacity, None)?);
        Ok(Grant { table, events })
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
    /// Bytes asked of it and not yet answered.
    pub asked: u64,
}

/// The table at a glance (display only; reading it changes nothing).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Status {
    pub capacity: u64,
    pub free: u64,
    /// Bytes on the device no row accounts for, measured.
    pub foreign: u64,
    pub rows: Vec<RowStatus>,
    /// Request id → `<package>|<shape>`.
    pub dispatched: Vec<(String, String)>,
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
    events: Owned,
    custody: custody::Client,
}

impl Ledger {
    /// The GPU's table, named per GPU and user; capacity = min(card, limit), never raised by a
    /// later opener. A restarted TensorD reopens the same table. TensorD serves custody when
    /// it holds the service lock (it does when it starts first).
    pub fn create(gpu: &GpuId, limit: Option<u64>) -> Result<Ledger> {
        let capacity = limit.map_or(gpu.memory, |l| l.min(gpu.memory));
        let events = os::event()?;
        let table = os::shared(&gpu.uuid, true)?;
        let shared = Arc::new(Shared::adopt(table, gpu, capacity, Some(os::dup(&events)?))?);
        shared.reap(None)?;
        let custody = custody::Client::open(&shared)?;
        Ok(Ledger { shared, events, custody })
    }

    /// The machine's sharing mode (shared until TensorD says dedicated).
    pub fn set_mode(&self, mode: Mode) -> Result<()> {
        self.shared.set_mode(mode)
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

    /// A request's measured peak at `<package>|<shape>`; an empty shape means the package's
    /// widest measured one.
    fn peak(&self, key: &str) -> Result<Option<u64>> {
        let bytes = |v: Vec<u8>| v.try_into().ok().map(u64::from_le_bytes);
        if key.ends_with('|') {
            let all = self.shared.facts(&format!("peak/{key}"))?;
            return Ok(all.into_iter().filter_map(|(_, v)| bytes(v)).max());
        }
        Ok(self.shared.fact(&format!("peak/{key}"))?.and_then(bytes))
    }

    /// Whether a request of `package` at `shape` (empty: its widest measured shape) can start
    /// now beside everything running.
    pub fn fits(&self, package: &str, shape: &str) -> bool {
        let fit = || -> Result<bool> {
            let (capacity, foreign, rows, dispatched) = self.admitted()?;
            Ok(decide::fits(capacity, foreign, &rows, &dispatched, self.peak(&format!("{package}|{shape}"))?))
        };
        fit().unwrap_or(false)
    }

    /// A multi-GPU group: dispatches `request` on every GPU of `group` if it fits on all, else
    /// on none (an error part-way undoes the dispatches made).
    pub fn admit(group: &[&Ledger], request: &str, package: &str, shape: &str) -> Result<bool> {
        if !group.iter().all(|l| l.fits(package, shape)) {
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
                .map(|r| RowStatus { pid: r.pid, live: r.live, context: r.facts.context, classes: classes(r), remaining: r.facts.remaining, unmeasured: r.facts.unmeasured as u32, asked: r.asked })
                .collect(),
            dispatched: dispatched.into_iter().map(|(k, v)| (k.trim_start_matches("dispatch/").to_string(), String::from_utf8_lossy(&v).into_owned())).collect(),
        }
    }

    /// Executors whose ask thread cannot answer, by its measured state: stopped (`T`, `t`) or
    /// its process a zombie. A running or sleeping thread is never stalled; nothing is counted
    /// or timed. Off Linux nothing is reported. TensorD ends what it returns.
    pub fn stalled(&self) -> Vec<u32> {
        let rows = self.shared.rows().unwrap_or_default();
        rows.iter().filter(|r| r.tid != 0 && os::stalled(r.pid, r.tid)).map(|r| r.pid).collect()
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
    /// A table with a live row of another process stays: a restarted TensorD reopens it.
    fn drop(&mut self) {
        let me = std::process::id();
        if self.shared.rows().is_ok_and(|rows| rows.iter().all(|r| !r.live || r.pid == me)) {
            os::unlink(&self.shared.name);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]

    use super::*;

    /// Review 1, finding 1: a reaper or writer acting on a stale view frees nothing once the
    /// offset holds a newer record: the free compares the whole word, generation included.
    #[test]
    fn a_stale_word_frees_nothing_newer() {
        let gpu = GpuId { uuid: format!("unit-{}-stale", std::process::id()), model: "host".into(), memory: 1 << 30 };
        let shared = Arc::new(Shared::adopt(os::shared(&gpu.uuid, true).unwrap(), &gpu, gpu.memory, None).unwrap());
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
        assert_eq!(shared.ask(row_id(at, stale), Class::N, 1).unwrap(), 0);
        drop(second);
        os::unlink(&gpu.uuid);
    }
}

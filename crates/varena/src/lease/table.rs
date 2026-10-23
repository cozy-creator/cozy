//! The lease table (varena-leases §11): one per executor and GPU, in shared memory the manager
//! creates and both map. A two-line header, then one line per chunk. A slot's `word` (state, pins,
//! grant seq) changes only by CAS: the manager moves Free→Granted and Released/Refused→Free, the
//! agent everything between. The manager's `ask` is advisory; the agent's CAS decides a revoke.
//! The table is untrusted input to the manager: atomics only, every index bounds-checked, no lock
//! either side could hold.

use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release};
use std::sync::atomic::{AtomicU32, AtomicU64};

use super::policy::Advice;
use crate::os::Map;
use crate::{Error, Result};

pub const MAGIC: u64 = u64::from_le_bytes(*b"vlease01");
pub const VERSION: u32 = 1;
const LINE: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Free = 0,
    Granted = 1,
    Mapped = 2,
    Revoking = 3,
    Released = 4,
    Refused = 5,
}

impl State {
    fn of(bits: u64) -> State {
        match bits & 0xf {
            1 => State::Granted,
            2 => State::Mapped,
            3 => State::Revoking,
            4 => State::Released,
            5 => State::Refused,
            _ => State::Free,
        }
    }
}

/// A slot's word: state (4 bits), pins (28), grant seq (32).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Word {
    pub state: State,
    pub pins: u32,
    pub seq: u32,
}

const PINS: u64 = (1 << 28) - 1;

impl Word {
    fn pack(self) -> u64 {
        self.state as u64 | (u64::from(self.pins) & PINS) << 4 | u64::from(self.seq) << 32
    }

    fn unpack(bits: u64) -> Word {
        Word { state: State::of(bits), pins: ((bits >> 4) & PINS) as u32, seq: (bits >> 32) as u32 }
    }
}

#[repr(C, align(128))]
struct Header {
    magic: AtomicU64,
    version: AtomicU32,
    slots: AtomicU32,
    pid: AtomicU32,
    _pad: AtomicU32,
    birth: AtomicU64,
    /// The manager bumps it at every attach: an agent re-attaches when it moves.
    epoch: AtomicU64,
    /// Grants and revokes the agent answered.
    acked: AtomicU64,
    /// GPU work of this process seen complete (its streams' events): a fence its revoke waits
    /// on moving counts as progress.
    progress: AtomicU64,
    /// The manager's figures, rewritten at every grant and release (`room` reads them, no IPC):
    /// bytes it could create now, other executors' unpinned and retained bytes it could revoke,
    /// the headroom it keeps, and foreign bytes as last sampled.
    free: AtomicU64,
    revocable: AtomicU64,
    headroom: AtomicU64,
    foreign: AtomicU64,
    /// 1 while GPU work the agent's holder recorded is not complete: busy, never stalled (a
    /// kernel longer than the looks is no stall; a stream that never completes is a GPU hang,
    /// the Runtime's health path).
    busy: AtomicU64,
}

/// The manager's figures in a table's header.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct Figures {
    pub free: u64,
    pub revocable: u64,
    pub headroom: u64,
    pub foreign: u64,
}

/// What a chunk holds: a region of a set's layout (`fingerprint` its digest's first 8 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Content {
    pub fingerprint: u64,
    pub region: u32,
    pub index: u32,
}

#[repr(C, align(64))]
struct Slot {
    word: AtomicU64,
    /// A multiple of the device's allocation granularity.
    bytes: AtomicU64,
    fingerprint: AtomicU64,
    next_use: AtomicU64,
    ask: AtomicU32,
    region: AtomicU32,
    index: AtomicU32,
    /// 1 once its fill landed: a dead executor's landed weights are retained.
    landed: AtomicU32,
    class: AtomicU32,
}

const HEAD: usize = 2 * LINE;
const _: () = assert!(std::mem::size_of::<Header>() == HEAD && std::mem::size_of::<Slot>() == LINE);

/// The bytes a table of `slots` chunks takes.
pub fn bytes(slots: u32) -> usize {
    HEAD + LINE * slots as usize
}

/// A mapped table. Every field is an atomic in the shared mapping.
pub struct Table {
    map: Map,
    slots: u32,
}

impl Table {
    /// The manager's: a new table over `map` (zeroed, `bytes(slots)` long).
    pub fn create(map: Map, slots: u32, pid: u32, birth: u64) -> Result<Table> {
        if map.len < bytes(slots) {
            return Err(Error::Invalid(format!("a lease table of {slots} slots needs {} bytes", bytes(slots))));
        }
        let table = Table { map, slots };
        let h = table.header();
        h.version.store(VERSION, Relaxed);
        h.slots.store(slots, Relaxed);
        h.pid.store(pid, Relaxed);
        h.birth.store(birth, Relaxed);
        h.magic.store(MAGIC, Release);
        Ok(table)
    }

    /// Either side's: an existing table (the agent at spawn, a restarted manager by name).
    pub fn open(map: Map) -> Result<Table> {
        if map.len < HEAD {
            return Err(Error::Invalid("a lease table shorter than its header".into()));
        }
        // SAFETY: the mapping is at least one line, page-aligned, and the header is atomics only.
        let h = unsafe { &*(map.base as *const Header) };
        if h.magic.load(Acquire) != MAGIC {
            return Err(Error::Invalid("not a lease table".into()));
        }
        let version = h.version.load(Relaxed);
        if version != VERSION {
            return Err(Error::Invalid(format!("lease table version {version}, this side speaks {VERSION}")));
        }
        let slots = h.slots.load(Relaxed);
        if map.len < bytes(slots) {
            return Err(Error::Invalid("a lease table shorter than its slots".into()));
        }
        Ok(Table { map, slots })
    }

    fn header(&self) -> &Header {
        // SAFETY: checked at create/open: the mapping holds the header line, all atomics.
        unsafe { &*(self.map.base as *const Header) }
    }

    fn slot(&self, i: u32) -> Option<&Slot> {
        // SAFETY: i < slots, and the mapping holds every slot line, all atomics.
        (i < self.slots).then(|| unsafe { &*(self.map.base.add(HEAD + LINE * i as usize) as *const Slot) })
    }

    fn slot_or(&self, i: u32) -> Result<&Slot> {
        self.slot(i).ok_or_else(|| Error::Invalid(format!("no lease slot {i} of {}", self.slots)))
    }

    pub fn slots(&self) -> u32 {
        self.slots
    }

    pub fn pid(&self) -> u32 {
        self.header().pid.load(Relaxed)
    }

    pub fn birth(&self) -> u64 {
        self.header().birth.load(Relaxed)
    }

    pub fn epoch(&self) -> u64 {
        self.header().epoch.load(Acquire)
    }

    pub fn attach(&self) -> u64 {
        self.header().epoch.fetch_add(1, AcqRel) + 1
    }

    pub fn acked(&self) -> u64 {
        self.header().acked.load(Acquire)
    }

    pub fn figures(&self) -> Figures {
        let h = self.header();
        Figures { free: h.free.load(Acquire), revocable: h.revocable.load(Acquire), headroom: h.headroom.load(Acquire), foreign: h.foreign.load(Acquire) }
    }

    pub fn set_figures(&self, f: Figures) {
        let h = self.header();
        h.free.store(f.free, Release);
        h.revocable.store(f.revocable, Release);
        h.headroom.store(f.headroom, Release);
        h.foreign.store(f.foreign, Release);
    }

    pub fn progress(&self) -> u64 {
        self.header().progress.load(Acquire)
    }

    pub fn progressed(&self) {
        self.header().progress.fetch_add(1, Release);
    }

    pub fn busy(&self) -> bool {
        self.header().busy.load(Acquire) == 1
    }

    pub fn set_busy(&self, busy: bool) {
        self.header().busy.store(u64::from(busy), Release);
    }

    fn ack(&self) {
        self.header().acked.fetch_add(1, Release);
    }

    pub fn word(&self, i: u32) -> Option<Word> {
        self.slot(i).map(|s| Word::unpack(s.word.load(Acquire)))
    }

    pub fn bytes_of(&self, i: u32) -> u64 {
        self.slot(i).map_or(0, |s| s.bytes.load(Relaxed))
    }

    pub fn content(&self, i: u32) -> Option<Content> {
        let s = self.slot(i)?;
        let fingerprint = s.fingerprint.load(Relaxed);
        (fingerprint != 0).then(|| Content { fingerprint, region: s.region.load(Relaxed), index: s.index.load(Relaxed) })
    }

    pub fn landed(&self, i: u32) -> bool {
        self.slot(i).is_some_and(|s| s.landed.load(Acquire) == 1)
    }

    pub fn ask(&self, i: u32) -> u32 {
        self.slot(i).map_or(0, |s| s.ask.load(Acquire))
    }

    /// One CAS from `from` to `to`; the word as found when it fails.
    fn swap(&self, i: u32, from: Word, to: Word) -> Result<std::result::Result<(), Word>> {
        let s = self.slot_or(i)?;
        Ok(s.word.compare_exchange(from.pack(), to.pack(), AcqRel, Acquire).map(drop).map_err(Word::unpack))
    }

    // The manager's moves.

    /// A Free slot, filled and granted under `seq`; false when it is not Free.
    pub fn grant(&self, i: u32, seq: u32, bytes: u64, content: Option<Content>, landed: bool) -> Result<bool> {
        let s = self.slot_or(i)?;
        let now = Word::unpack(s.word.load(Acquire));
        if now.state != State::Free || bytes == 0 {
            return Ok(false);
        }
        s.bytes.store(bytes, Relaxed);
        let c = content.unwrap_or(Content { fingerprint: 0, region: 0, index: 0 });
        s.fingerprint.store(c.fingerprint, Relaxed);
        s.region.store(c.region, Relaxed);
        s.index.store(c.index, Relaxed);
        s.landed.store(u32::from(landed), Relaxed);
        s.ask.store(0, Relaxed);
        Ok(self.swap(i, now, Word { state: State::Granted, pins: 0, seq })?.is_ok())
    }

    /// The manager wants `seq` back (advisory: the agent's CAS decides).
    pub fn request(&self, i: u32, seq: u32) -> Result<()> {
        self.slot_or(i)?.ask.store(seq, Release);
        Ok(())
    }

    /// A Released or Refused slot (or any slot of a dead executor, `dead`) is Free again: its
    /// word as it was.
    pub fn reclaim(&self, i: u32, dead: bool) -> Result<Option<Word>> {
        let s = self.slot_or(i)?;
        loop {
            let now = Word::unpack(s.word.load(Acquire));
            let done = matches!(now.state, State::Released | State::Refused);
            if now.state == State::Free || !(done || dead) {
                return Ok(None);
            }
            if self.swap(i, now, Word { state: State::Free, pins: 0, seq: now.seq })?.is_ok() {
                s.ask.store(0, Release);
                return Ok(Some(now));
            }
        }
    }

    // The agent's moves.

    /// The grant imported and mapped (`mapped`) or refused.
    pub fn answer(&self, i: u32, seq: u32, mapped: bool) -> Result<bool> {
        let to = Word { state: if mapped { State::Mapped } else { State::Refused }, pins: 0, seq };
        let ok = self.swap(i, Word { state: State::Granted, pins: 0, seq }, to)?.is_ok();
        if ok {
            self.ack();
        }
        Ok(ok)
    }

    /// One pin more; false while it is not Mapped (being revoked, or never ours).
    pub fn pin(&self, i: u32) -> Result<bool> {
        let s = self.slot_or(i)?;
        let mut now = Word::unpack(s.word.load(Acquire));
        loop {
            if now.state != State::Mapped || u64::from(now.pins) == PINS {
                return Ok(false);
            }
            match self.swap(i, now, Word { pins: now.pins + 1, ..now })? {
                Ok(()) => return Ok(true),
                Err(found) => now = found,
            }
        }
    }

    pub fn unpin(&self, i: u32) -> Result<()> {
        let s = self.slot_or(i)?;
        let mut now = Word::unpack(s.word.load(Acquire));
        loop {
            if now.state != State::Mapped || now.pins == 0 {
                return Err(Error::Invalid(format!("lease slot {i}: unpin of {now:?}")));
            }
            match self.swap(i, now, Word { pins: now.pins - 1, ..now })? {
                Ok(()) => return Ok(()),
                Err(found) => now = found,
            }
        }
    }

    /// The revoke's commit: Mapped with no pins under `seq` → Revoking. False when pinned, moved
    /// on, or never this grant.
    pub fn commit(&self, i: u32, seq: u32) -> Result<bool> {
        Ok(self.swap(i, Word { state: State::Mapped, pins: 0, seq }, Word { state: State::Revoking, pins: 0, seq })?.is_ok())
    }

    /// Unmapped and released: the manager may destroy or retain it.
    pub fn released(&self, i: u32, seq: u32) -> Result<bool> {
        let ok = self.swap(i, Word { state: State::Revoking, pins: 0, seq }, Word { state: State::Released, pins: 0, seq })?.is_ok();
        if ok {
            self.ack();
        }
        Ok(ok)
    }

    pub fn set_landed(&self, i: u32) -> Result<()> {
        self.set_landed_to(i, true)
    }

    pub fn set_landed_to(&self, i: u32, landed: bool) -> Result<()> {
        self.slot_or(i)?.landed.store(u32::from(landed), Release);
        Ok(())
    }

    /// The agent publishes a chunk's advice; the manager reads it back with [`Table::advice`].
    pub fn advise(&self, i: u32, advice: Advice) -> Result<()> {
        let (class, value) = advice.encode();
        let s = self.slot_or(i)?;
        s.class.store(class, Relaxed);
        s.next_use.store(value, Relaxed);
        Ok(())
    }

    pub fn advice(&self, i: u32) -> Advice {
        self.slot(i).map_or(Advice::Unset, |s| Advice::decode(s.class.load(Relaxed), s.next_use.load(Relaxed)))
    }
}

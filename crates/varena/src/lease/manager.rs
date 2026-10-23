//! The lease manager (varena-leases §1, §2, §2a, §11): it creates every physical chunk, grants
//! chunks to executors' agents, revokes for room, and takes an exited executor's leases back
//! once its pidfd shows the exit: landed weights are retained for the next executor of that
//! trust domain and region (no copy), the rest destroyed. Only the manager exports.
//!
//! One manager per GPU: TensorD makes it where it made the ledger, hands it the GPU's listener,
//! and reacts to its events fd. Until step 4 the admission and placement facts stay the ledger's,
//! which the manager owns ([`Manager::ledger`]).
//!
//! The table is untrusted input: identity, pid, a chunk's index and whether it landed are the
//! manager's own records (peer credentials, grant time, fill-ack time), never read back at exit.
//!
//! Locks: `books` is the only one, and nothing blocks under it (sends and records are made after
//! it drops).

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::os::fd::AsFd;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;
use platform::handles::Peer;
use platform::process::{Birth, Exact};

use super::table::{self, Content, State, Table};
use super::wire::{End, Region, ToAgent, ToManager};
use crate::backend::{Backend, Physical};
use crate::ledger::{Ledger, Mode};
use crate::os::{self, Listener, Owned, Stream};
use crate::{Error, GpuId, Result};

/// The largest chunk: one handle each (§3).
pub const CHUNK: u64 = 32 << 20;
/// The lease protocol an agent speaks in its Hello (1 is the ledger's).
pub const PROTOCOL: u32 = 2;

/// Who a connecting peer is, as its host knows it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    /// Its executor group (a multi-GPU executor's followers share it).
    pub executor: String,
    /// The trust domain retained weights are adopted within.
    pub domain: String,
}

/// What the manager did, for its host's memory log.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Record {
    Grant { pid: u32, ask: u64, chunks: usize, bytes: u64, adopted: bool, create_us: u64, export_us: u64, total_us: u64 },
    Adopt { pid: u32, region: Region, chunks: usize, bytes: u64 },
    Refuse { pid: u32, ask: u64, why: String },
    Revoke { pid: u32, slots: Vec<u32> },
    Answered { pid: u32, released: usize, us: u64 },
    Reclaim { pid: u32, destroyed_bytes: u64, retained_bytes: u64, us: u64 },
}

/// What TensorD answers for the manager.
pub trait Hosting: Send + Sync {
    /// The identity of a peer on the GPU's listener (any uid: the host judges it); `None`
    /// refuses it.
    fn identify(&self, peer: &Peer) -> Option<Identity>;

    /// A record of what the manager did, after its books dropped.
    fn record(&self, _record: Record) {}
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct ExecutorHolding {
    pub pid: u32,
    pub executor: String,
    pub chunks: usize,
    pub bytes: u64,
}

/// What a manager holds, for its host's status and the tests: free = capacity − created.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct Holdings {
    pub capacity: u64,
    pub created: u64,
    pub created_chunks: usize,
    pub leased: u64,
    pub retained: u64,
    pub retained_chunks: usize,
    /// Chunks a previous manager created that agents still map (0 until re-attach exists).
    pub books_only: u64,
    pub executors: Vec<ExecutorHolding>,
}

pub struct Manager {
    backend: Arc<dyn Backend>,
    capacity: u64,
    granularity: u64,
    ledger: Ledger,
    hosting: Arc<dyn Hosting>,
    retired: AtomicBool,
    books: Mutex<Books>,
}

#[derive(Default)]
struct Books {
    created: u64,
    created_chunks: usize,
    seq: u32,
    next: u64,
    executors: BTreeMap<u64, Exec>,
    /// Landed weights of exited executors, by trust domain and region: a region's chunks by
    /// index, oldest first in `order`.
    retained: BTreeMap<Kept, BTreeMap<u32, Physical>>,
    order: VecDeque<Kept>,
}

type Kept = (String, Region);

struct Exec {
    /// From the peer's credentials, never the table.
    pid: u32,
    identity: Identity,
    table: Arc<Table>,
    end: Arc<End>,
    free: BTreeSet<u32>,
    leased: BTreeMap<u32, Leased>,
    waiting: VecDeque<Waiting>,
    /// Slots asked back and not yet released.
    asked: BTreeSet<u32>,
    /// Releases the manager reclaimed itself: the progress a stall is judged by.
    progress: u64,
    /// `progress` at the last stall look while unpinned asks were out.
    looked: Option<u64>,
}

struct Leased {
    phys: Physical,
    seq: u32,
    region: Option<Region>,
    index: u32,
    /// The agent acked its fill (recorded at that ack, never read at exit).
    landed: bool,
}

struct Waiting {
    id: u64,
    bytes: u64,
    region: Option<Region>,
}

/// What to send and record once `books` dropped.
#[derive(Default)]
struct Out {
    sends: Vec<(Arc<End>, ToAgent, Option<Owned>)>,
    records: Vec<Record>,
}

impl Out {
    fn append(&mut self, mut other: Out) {
        self.sends.append(&mut other.sends);
        self.records.append(&mut other.records);
    }
}

/// A region's chunk sizes: `CHUNK`s, the tail rounded up to the device's `granularity`.
pub fn chunks(bytes: u64, granularity: u64) -> Vec<u64> {
    let unit = granularity.max(1);
    let mut out = vec![];
    let mut left = bytes.div_ceil(unit) * unit;
    while left > 0 {
        let take = left.min(CHUNK);
        out.push(take);
        left -= take;
    }
    out
}

enum Room {
    /// Enough is free or coming: retained bytes dropped, or revokes of chunks still unpinned or
    /// already being revoked.
    Coming,
    /// Only other executors' pinned chunks could give it: asked now, they give when unpinned.
    Pinned,
    /// No room can come (the only room left is the asker's own pins).
    Refused,
}

fn micros(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_micros()).unwrap_or(u64::MAX)
}

impl Manager {
    /// The GPU's manager: its ledger made, `listener` served on the manager's own thread until
    /// [`Manager::retire`]. It creates at most `limit` (a `--gpu-memory-limit`, else the card),
    /// less its own context and, in shared mode, the headroom H.
    pub fn create(gpu: &GpuId, limit: Option<u64>, mode: Mode, backend: Arc<dyn Backend>, listener: Listener, hosting: Arc<dyn Hosting>) -> Result<Arc<Manager>> {
        let ledger = Ledger::create(gpu, limit)?;
        ledger.set_mode(mode)?;
        let card = limit.unwrap_or(gpu.memory).min(backend.capacity());
        let headroom = if mode == Mode::Shared { crate::table::headroom(gpu.memory) } else { 0 };
        let capacity = card.saturating_sub(headroom).saturating_sub(backend.opened_context());
        let granularity = backend.granularity();
        let manager = Arc::new(Manager { backend, capacity, granularity, ledger, hosting, retired: AtomicBool::new(false), books: Mutex::new(Books::default()) });
        let serving = manager.clone();
        std::thread::Builder::new()
            .name("lease-listener".into())
            .spawn(move || {
                for stream in listener.incoming() {
                    if serving.retired.load(Relaxed) {
                        break;
                    }
                    match stream {
                        Ok(stream) => serving.serve(stream),
                        Err(error) => tracing::warn!("lease: accepting an agent: {error}"),
                    }
                }
            })
            .map_err(|e| Error::io("spawn lease-listener", e))?;
        Ok(manager)
    }

    /// The admission and placement facts, the ledger's until step 4.
    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    /// Readable on any lease or ledger event; [`Manager::clear_events`] clears it.
    pub fn events(&self) -> &Owned {
        self.ledger.events()
    }

    pub fn clear_events(&self) -> bool {
        self.ledger.clear_events()
    }

    /// The ledger's stalled rows, and executors with unpinned chunks asked back that released
    /// nothing since the last look (progress the manager observed itself, never a clock).
    pub fn stalled(&self) -> Vec<u32> {
        let mut pids = self.ledger.stalled();
        let mut b = self.books.lock();
        for e in b.executors.values_mut() {
            let unpinned = e.asked.iter().any(|s| e.table.word(*s).is_some_and(|w| w.state == State::Mapped && w.pins == 0));
            if !unpinned {
                e.looked = None;
                continue;
            }
            if e.looked == Some(e.progress) {
                pids.push(e.pid);
            }
            e.looked = Some(e.progress);
        }
        pids.sort_unstable();
        pids.dedup();
        pids
    }

    pub fn holdings(&self) -> Holdings {
        let b = self.books.lock();
        let executors: Vec<ExecutorHolding> = b
            .executors
            .values()
            .map(|e| ExecutorHolding { pid: e.pid, executor: e.identity.executor.clone(), chunks: e.leased.len(), bytes: e.leased.values().map(|l| l.phys.len).sum() })
            .collect();
        Holdings {
            capacity: self.capacity,
            created: b.created,
            created_chunks: b.created_chunks,
            leased: executors.iter().map(|e| e.bytes).sum(),
            retained: b.retained.values().flat_map(|v| v.values()).map(|p| p.len).sum(),
            retained_chunks: b.retained.values().map(BTreeMap::len).sum(),
            books_only: 0,
            executors,
        }
    }

    pub fn set_mode(&self, mode: Mode) -> Result<()> {
        self.ledger.set_mode(mode)
    }

    /// Stops accepting agents; connected ones keep their leases until they exit.
    pub fn retire(&self) {
        self.retired.store(true, Relaxed);
    }

    /// One executor's connection, on its own thread until the executor has exited and its
    /// leases are back.
    fn serve(self: &Arc<Self>, stream: Stream) {
        let manager = self.clone();
        let spawned = std::thread::Builder::new().name("lease-executor".into()).spawn(move || {
            if let Err(error) = manager.executor(stream) {
                tracing::warn!("lease: an executor's connection ended: {error}");
            }
        });
        if let Err(error) = spawned {
            tracing::warn!("lease: serving an agent: {error}");
        }
    }

    fn executor(self: &Arc<Self>, stream: Stream) -> Result<()> {
        let peer = platform::handles::peer(&stream).map_err(|e| Error::io("lease peer", e))?;
        let end = Arc::new(End::new(stream));
        let Some(identity) = self.hosting.identify(&peer) else {
            return end.send(&ToAgent::Refuse { id: 0, why: "not an executor of this machine".into() }, None);
        };
        let Some((ToManager::Hello { protocol, birth }, again)) = end.recv::<ToManager>()? else {
            return Err(Error::Invalid("a lease connection opened without Hello".into()));
        };
        if protocol != PROTOCOL || again.is_some() {
            // A re-attach after a manager restart comes after slice 1: that agent runs standalone.
            let why = if again.is_some() { "re-attach is not served yet".into() } else { format!("lease protocol {protocol}; this manager speaks {PROTOCOL}") };
            return end.send(&ToAgent::Refuse { id: 0, why }, None);
        }
        let found = Birth::of(peer.pid).map_err(|e| Error::Io(e.to_string()))?;
        if found.start_ticks != birth {
            return Err(Error::Invalid(format!("lease Hello for birth {birth} from pid {} (birth {})", peer.pid, found.start_ticks)));
        }
        let process = Exact::open(&found).map_err(|e| Error::Io(e.to_string()))?.ok_or_else(|| Error::Closed(format!("process {}", peer.pid)))?;
        let slots = u32::try_from(self.capacity / self.granularity.max(1)).unwrap_or(u32::MAX).max(1);
        // Sealed against resizing before the agent sees it: its ftruncate could SIGBUS us.
        let file = platform::memory::page(table::bytes(slots) as u64).map_err(|e| Error::Io(e.to_string()))?;
        platform::shm::seal_with(file.as_fd(), platform::shm::RESIZE_SEALS).map_err(|e| Error::Io(e.to_string()))?;
        let table = Arc::new(Table::create(os::map(&file, table::bytes(slots), true)?, slots, peer.pid, birth, [0, 0])?);
        let epoch = table.attach();
        let id = {
            let mut b = self.books.lock();
            b.next += 1;
            let id = b.next;
            let exec = Exec { pid: peer.pid, identity, table, end: end.clone(), free: (0..slots).collect(), leased: BTreeMap::new(), waiting: VecDeque::new(), asked: BTreeSet::new(), progress: 0, looked: None };
            b.executors.insert(id, exec);
            id
        };
        end.send(&ToAgent::Welcome { epoch }, Some(os::dup(&file)?))?;
        drop(file);
        loop {
            match end.recv::<ToManager>() {
                Ok(Some((message, _))) => {
                    let out = match message {
                        ToManager::Ask { id: ask, bytes, region } => self.ask(id, Waiting { id: ask, bytes, region }),
                        ToManager::Answered => self.answered(id),
                        ToManager::Hello { .. } => Out::default(),
                    };
                    self.emit(out);
                }
                Ok(None) => break,
                Err(error) => {
                    tracing::warn!("lease: executor {}: {error}", peer.pid);
                    break;
                }
            }
        }
        // The socket closed: its leases come back once the process's exit shows (its mappings
        // are gone then), never before.
        process.wait().map_err(|e| Error::Io(e.to_string()))?;
        let out = self.exited(id);
        self.emit(out);
        Ok(())
    }

    fn emit(&self, out: Out) {
        for (end, message, handle) in out.sends {
            if let Err(error) = end.send(&message, handle) {
                tracing::debug!("lease: a send to an executor failed (it is going): {error}");
            }
        }
        for record in out.records {
            self.hosting.record(record);
        }
        self.ledger.ring();
    }

    fn ask(&self, exec: u64, ask: Waiting) -> Out {
        let mut b = self.books.lock();
        if let Some(e) = b.executors.get_mut(&exec) {
            e.waiting.push_back(ask);
        }
        self.settle(&mut b)
    }

    /// The agent moved slots: fills acked landed are recorded; Released and Refused slots are
    /// reclaimed, their chunks destroyed; then the waiting asks (their victims picked again).
    fn answered(&self, exec: u64) -> Out {
        let start = Instant::now();
        let mut b = self.books.lock();
        let mut gone = vec![];
        let mut out = Out::default();
        if let Some(e) = b.executors.get_mut(&exec) {
            let mut done = vec![];
            for (slot, l) in e.leased.iter_mut() {
                match e.table.word(*slot).map(|w| (w.state, w.seq)) {
                    Some((State::Mapped, seq)) if seq == l.seq && e.table.landed(*slot) => l.landed = true,
                    Some((State::Released | State::Refused, seq)) if seq == l.seq => done.push(*slot),
                    _ => {}
                }
            }
            for slot in done {
                if let (Ok(Some(_)), Some(l)) = (e.table.reclaim(slot, false), e.leased.remove(&slot)) {
                    e.free.insert(slot);
                    e.asked.remove(&slot);
                    e.progress += 1;
                    gone.push(l.phys);
                }
            }
            if !gone.is_empty() {
                out.records.push(Record::Answered { pid: e.pid, released: gone.len(), us: 0 });
            }
        }
        for phys in gone {
            self.destroy(&mut b, phys);
        }
        if let Some(Record::Answered { us, .. }) = out.records.last_mut() {
            *us = micros(start);
        }
        out.append(self.settle(&mut b));
        out
    }

    /// An exited executor's every slot: landed weights (by the manager's records) retained in
    /// its domain, the rest destroyed.
    fn exited(&self, exec: u64) -> Out {
        let start = Instant::now();
        let mut b = self.books.lock();
        let Some(e) = b.executors.remove(&exec) else { return Out::default() };
        let (mut gone, mut retained) = (vec![], 0);
        for l in e.leased.into_values() {
            match l.region {
                Some(region) if l.landed => {
                    retained += l.phys.len;
                    let key = (e.identity.domain.clone(), region);
                    let kept = b.retained.entry(key.clone()).or_default();
                    let first = kept.is_empty();
                    gone.extend(kept.insert(l.index, l.phys));
                    if first {
                        b.order.push_back(key);
                    }
                }
                _ => gone.push(l.phys),
            }
        }
        let destroyed = gone.iter().map(|p| p.len).sum();
        for phys in gone {
            self.destroy(&mut b, phys);
        }
        let mut out = self.settle(&mut b);
        out.records.insert(0, Record::Reclaim { pid: e.pid, destroyed_bytes: destroyed, retained_bytes: retained, us: micros(start) });
        out
    }

    fn destroy(&self, b: &mut Books, phys: Physical) {
        b.created = b.created.saturating_sub(phys.len);
        b.created_chunks = b.created_chunks.saturating_sub(1);
        if let Err(error) = self.backend.destroy(phys) {
            tracing::warn!("lease: destroying a chunk: {error}");
        }
    }

    fn drop_retained(&self, b: &mut Books, key: &Kept) {
        b.order.retain(|k| k != key);
        for phys in b.retained.remove(key).unwrap_or_default().into_values() {
            self.destroy(b, phys);
        }
    }

    /// Answers waiting asks in order while room allows; for the first that does not fit, makes
    /// room and waits for it, or refuses it when none can come.
    fn settle(&self, b: &mut Books) -> Out {
        let mut out = Out::default();
        while let Some((exec, ask)) = b.executors.iter_mut().find_map(|(id, e)| e.waiting.pop_front().map(|w| (*id, w))) {
            let pid = b.executors.get(&exec).map_or(0, |e| e.pid);
            let refuse = |b: &Books, out: &mut Out, why: String| {
                if let Some(e) = b.executors.get(&exec) {
                    out.sends.push((e.end.clone(), ToAgent::Refuse { id: ask.id, why: why.clone() }, None));
                }
                out.records.push(Record::Refuse { pid, ask: ask.id, why });
            };
            match self.answer(b, exec, &ask) {
                Ok(Some(granted)) => out.append(granted),
                Ok(None) => {
                    let short = (b.created + chunks(ask.bytes, self.granularity).iter().sum::<u64>()).saturating_sub(self.capacity);
                    match self.make_room(b, exec, short, &mut out) {
                        Room::Coming | Room::Pinned => {
                            if let Some(e) = b.executors.get_mut(&exec) {
                                e.waiting.push_front(ask);
                            }
                            break;
                        }
                        Room::Refused => refuse(b, &mut out, format!("nothing revocable for {} bytes", ask.bytes)),
                    }
                }
                Err(error) => refuse(b, &mut out, error.to_string()),
            }
        }
        out
    }

    /// The grants for `ask` when it fits now; `None` when room must be made first. A failure
    /// partway rolls the whole ask back: no slot stays granted, no chunk leaks.
    fn answer(&self, b: &mut Books, exec: u64, ask: &Waiting) -> Result<Option<Out>> {
        let start = Instant::now();
        let sizes = chunks(ask.bytes, self.granularity);
        let Some(pid) = b.executors.get(&exec).map(|e| e.pid) else { return Err(Error::Closed("the executor".into())) };
        let domain = b.executors.get(&exec).map(|e| e.identity.domain.clone()).unwrap_or_default();
        // Adopted only whole: every chunk, by index, the size this ask cuts.
        let whole = |kept: &BTreeMap<u32, Physical>| kept.len() == sizes.len() && kept.iter().zip(&sizes).enumerate().all(|(i, ((index, p), len))| *index as usize == i && p.len == *len);
        let adopt = ask.region.map(|r| (domain, r)).filter(|k| b.retained.get(k).is_some_and(whole));
        let phys: Vec<Physical> = if let Some(key) = &adopt {
            b.order.retain(|k| k != key);
            b.retained.remove(key).unwrap_or_default().into_values().collect()
        } else {
            if b.created + sizes.iter().sum::<u64>() > self.capacity {
                return Ok(None);
            }
            let mut made = vec![];
            for &len in &sizes {
                match self.backend.create(len) {
                    Ok(p) => made.push(p),
                    Err(error) => {
                        made.into_iter().for_each(|p| drop(self.backend.destroy(p)));
                        return Err(error);
                    }
                }
            }
            b.created += sizes.iter().sum::<u64>();
            b.created_chunks += sizes.len();
            made
        };
        let create_us = micros(start);
        let exporting = Instant::now();
        let n = phys.len();
        let bytes: u64 = phys.iter().map(|p| p.len).sum();
        let mut out = Out::default();
        let mut granted: Vec<u32> = vec![];
        let mut failed = None;
        let mut spare = vec![];
        for (index, p) in phys.into_iter().enumerate() {
            if failed.is_some() {
                spare.push(p);
                continue;
            }
            b.seq = b.seq.wrapping_add(1).max(1);
            let seq = b.seq;
            let Some(e) = b.executors.get_mut(&exec) else {
                spare.push(p);
                failed = Some(Error::Closed("the executor".into()));
                continue;
            };
            let Some(slot) = e.free.pop_first() else {
                spare.push(p);
                failed = Some(Error::Invalid("the executor's lease table is full".into()));
                continue;
            };
            let content = ask.region.map(|r| Content { fingerprint: r.fingerprint, region: r.region, index: index as u32 });
            let exported = e.table.grant(slot, seq, p.len, content, adopt.is_some()).and_then(|ok| if ok { self.backend.lend(&p).map(Some) } else { Ok(None) });
            match exported {
                Ok(Some(handle)) => {
                    out.sends.push((e.end.clone(), ToAgent::Grant { id: ask.id, slot, seq, bytes: p.len, adopted: adopt.is_some(), last: index + 1 == n }, Some(handle)));
                    e.leased.insert(slot, Leased { phys: p, seq, region: ask.region, index: index as u32, landed: adopt.is_some() });
                    granted.push(slot);
                }
                Ok(None) => {
                    e.free.insert(slot);
                    spare.push(p);
                    failed = Some(Error::Invalid(format!("lease slot {slot} was not free")));
                }
                Err(error) => {
                    let _ = e.table.reclaim(slot, true);
                    e.free.insert(slot);
                    spare.push(p);
                    failed = Some(error);
                }
            }
        }
        let Some(error) = failed else {
            let adopted = adopt.is_some();
            out.records.push(Record::Grant { pid, ask: ask.id, chunks: n, bytes, adopted, create_us, export_us: micros(exporting), total_us: micros(start) });
            if let (true, Some(region)) = (adopted, ask.region) {
                out.records.push(Record::Adopt { pid, region, chunks: n, bytes });
            }
            return Ok(Some(out));
        };
        // Roll back: the slots granted so far are Free again, their chunks with the rest.
        if let Some(e) = b.executors.get_mut(&exec) {
            for slot in granted {
                let _ = e.table.reclaim(slot, true);
                e.free.insert(slot);
                if let Some(l) = e.leased.remove(&slot) {
                    spare.insert(l.index as usize, l.phys);
                }
            }
        }
        match adopt {
            // Adopted chunks go back to retention whole, by index: nothing was copied.
            Some(key) if spare.len() == n => {
                b.retained.insert(key.clone(), spare.into_iter().enumerate().map(|(i, p)| (i as u32, p)).collect());
                b.order.push_front(key);
            }
            _ => spare.into_iter().for_each(|p| self.destroy(b, p)),
        }
        Err(error)
    }

    /// Room for `short` bytes: retained regions first (oldest), then asks back of unpinned
    /// chunks (idle class first, then furthest next use). Asked slots count only while they can
    /// still give (unpinned, or being revoked). When only other executors' pinned chunks could
    /// give it, they are asked too and the ask waits for their unpin; when only the asker's own
    /// pins could, it is refused.
    fn make_room(&self, b: &mut Books, asker: u64, short: u64, out: &mut Out) -> Room {
        let mut room = 0;
        while room < short {
            let Some(key) = b.order.front().cloned() else { break };
            room += b.retained.get(&key).map_or(0, |v| v.values().map(|p| p.len).sum());
            self.drop_retained(b, &key);
        }
        if room >= short {
            return Room::Coming;
        }
        for e in b.executors.values() {
            for s in &e.asked {
                if e.table.word(*s).is_some_and(|w| (w.state == State::Mapped && w.pins == 0) || w.state == State::Revoking) {
                    room += e.table.bytes_of(*s);
                }
            }
        }
        let (mut unpinned, mut pinned) = (vec![], vec![]);
        for (id, e) in &b.executors {
            for (slot, l) in &e.leased {
                let Some(w) = e.table.word(*slot) else { continue };
                if w.state != State::Mapped || w.seq != l.seq || e.asked.contains(slot) {
                    continue;
                }
                if w.pins == 0 {
                    let (class, next) = e.table.class(*slot);
                    unpinned.push((class, std::cmp::Reverse(next), *id, *slot, l.seq));
                } else if *id != asker {
                    pinned.push((*id, *slot, l.seq));
                }
            }
        }
        unpinned.sort();
        let mut asked: BTreeMap<u64, Vec<u32>> = BTreeMap::new();
        let mut ask_back = |b: &mut Books, id: u64, slot: u32, seq: u32| -> u64 {
            let Some(e) = b.executors.get_mut(&id) else { return 0 };
            if e.table.request(slot, seq).is_err() {
                return 0;
            }
            e.asked.insert(slot);
            asked.entry(id).or_default().push(slot);
            e.table.bytes_of(slot)
        };
        for (_, _, id, slot, seq) in unpinned {
            if room >= short {
                break;
            }
            room += ask_back(b, id, slot, seq);
        }
        let verdict = if room >= short {
            Room::Coming
        } else {
            for (id, slot, seq) in pinned {
                if room >= short {
                    break;
                }
                room += ask_back(b, id, slot, seq);
            }
            if room >= short { Room::Pinned } else { Room::Refused }
        };
        for (id, slots) in asked {
            if let Some(e) = b.executors.get(&id) {
                out.sends.push((e.end.clone(), ToAgent::Revoke, None));
                out.records.push(Record::Revoke { pid: e.pid, slots });
            }
        }
        verdict
    }
}

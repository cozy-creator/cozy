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
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};
use platform::handles::Peer;
use platform::process::Exact;

use super::table::{self, Content, Figures, State, Table};
use super::wire::{End, Refusal, Region, ToAgent, ToManager};
use crate::backend::{Backend, Physical};
use crate::ledger::{FixedRow, Ledger, Mode};
use crate::os::{self, Listener, Owned, Stream};
use crate::{Error, GpuId, Result};

/// The largest chunk: one handle each (§3).
pub const CHUNK: u64 = 32 << 20;
/// The lease protocol an agent speaks in its Hello (1 is the ledger's).
pub const PROTOCOL: u32 = 2;
/// The stall look's cadence (TensorD's sampler's): a period, never a verdict.
const LOOK: Duration = Duration::from_millis(500);
/// Looks in a row with neither a release nor any of its GPU work completing, while an unpinned
/// ask of it waits, before an executor is stalled.
const LOOKS: u32 = 3;

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
    Refuse { pid: u32, ask: u64, code: Refusal, why: String },
    Revoke { pid: u32, slots: Vec<u32> },
    Answered { pid: u32, released: usize, us: u64 },
    Reclaim { pid: u32, destroyed_bytes: u64, retained_bytes: u64, us: u64 },
    /// Judged stalled: `looks` looks with no release and no GPU progress, its oldest unpinned
    /// ask `waited_ms` old.
    Stalled { pid: u32, looks: u32, waited_ms: u64 },
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
    /// Of `bytes`, what its fills acked landed: what its exit would retain.
    pub landed: u64,
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

/// Opens the GPU's backend (TensorD's CUDA context), at the first ask.
pub type Opener = Box<dyn Fn() -> Result<Arc<dyn Backend>> + Send + Sync>;

/// The backend while chunks or executors need it, with what it measured at open.
struct Device {
    backend: Arc<dyn Backend>,
    capacity: u64,
    granularity: u64,
    headroom: u64,
    context: u64,
}

/// A table's slots: one per smallest chunk the card could hold.
const SLOT_UNIT: u64 = 2 << 20;

pub struct Manager {
    open: Opener,
    /// One open at a time (never under `books`).
    opening: Mutex<()>,
    gpu_memory: u64,
    limit: Option<u64>,
    mode: Mode,
    ledger: Ledger,
    /// Its own bytes on the ledger: retained chunks no executor maps and its context, so no
    /// executor counts them foreign.
    row: FixedRow,
    hosting: Arc<dyn Hosting>,
    retired: AtomicBool,
    books: Mutex<Books>,
    /// Wakes the stall look when asks go out.
    looking: Condvar,
}

#[derive(Default)]
struct Books {
    /// Opened at the first ask, dropped once no chunk and no executor is left (§2a: TensorD's
    /// context keeps a laptop's GPU awake and costs VRAM).
    device: Option<Arc<Device>>,
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
    /// Slots asked back and not yet released, and when each was asked.
    asked: BTreeMap<u32, Instant>,
    /// Releases the manager reclaimed itself, and with the agent's GPU progress, what a stall
    /// is judged by.
    progress: u64,
    /// The stall look's: (releases, GPU progress) at the last look that saw either move, and
    /// looks since.
    seen: (u64, u64),
    still: u32,
    stalled: bool,
    /// The executors whose pins its waiting ask waits on.
    waits_on: BTreeSet<u64>,
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

/// What to send and record once `books` dropped, and a backend nothing needs any more.
#[derive(Default)]
struct Out {
    sends: Vec<(Arc<End>, ToAgent, Option<Owned>)>,
    records: Vec<Record>,
    idle: Option<Arc<Device>>,
}

impl Out {
    fn append(&mut self, mut other: Out) {
        self.sends.append(&mut other.sends);
        self.records.append(&mut other.records);
        self.idle = self.idle.take().or(other.idle);
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

/// An ask's chunks: a weights region is one chunk sized to its page (§3, never split); torch
/// and cache bytes split at [`CHUNK`].
fn sizes(ask: &Waiting, granularity: u64) -> Vec<u64> {
    match ask.region {
        Some(_) => {
            let unit = granularity.max(1);
            vec![ask.bytes.div_ceil(unit) * unit]
        }
        None => chunks(ask.bytes, granularity),
    }
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
    /// [`Manager::retire`]. Its backend opens at the first ask (`open`); it then creates at most
    /// `limit` (a `--gpu-memory-limit`, else the card), less its own context and, in shared
    /// mode, the headroom H.
    pub fn create(gpu: &GpuId, limit: Option<u64>, mode: Mode, open: Opener, listener: Listener, hosting: Arc<dyn Hosting>) -> Result<Arc<Manager>> {
        let ledger = Ledger::create(gpu, limit)?;
        ledger.set_mode(mode)?;
        let row = ledger.fixed(std::process::id())?;
        let manager = Arc::new(Manager { open, opening: Mutex::new(()), gpu_memory: gpu.memory, limit, mode, ledger, row, hosting, retired: AtomicBool::new(false), books: Mutex::new(Books::default()), looking: Condvar::new() });
        let weak = Arc::downgrade(&manager);
        std::thread::Builder::new().name("lease-look".into()).spawn(move || look_loop(&weak)).map_err(|e| Error::io("spawn lease-look", e))?;
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

    /// The ledger's stalled rows, and executors the stall look judged stalled: an unpinned
    /// chunk asked back, and neither a release nor any GPU progress of it across [`LOOKS`]
    /// looks. A getter: any caller, any rate.
    pub fn stalled(&self) -> Vec<u32> {
        let mut pids = self.ledger.stalled();
        pids.extend(self.books.lock().executors.values().filter(|e| e.stalled).map(|e| e.pid));
        pids.sort_unstable();
        pids.dedup();
        pids
    }

    /// One stall look (every [`LOOK`] while asks are out): whether any ask is out, and the
    /// executors newly judged stalled.
    fn look(&self) -> (bool, Vec<Record>) {
        let mut b = self.books.lock();
        let (mut out, mut records) = (false, vec![]);
        for e in b.executors.values_mut() {
            out |= !e.asked.is_empty();
            let now = (e.progress, e.table.progress());
            let oldest = e.asked.iter().filter(|(s, _)| e.table.word(**s).is_some_and(|w| w.state == State::Mapped && w.pins == 0)).map(|(_, at)| *at).min();
            let Some(oldest) = oldest.filter(|_| now == e.seen) else {
                (e.seen, e.still, e.stalled) = (now, 0, false);
                continue;
            };
            e.still += 1;
            if e.still >= LOOKS && !e.stalled {
                e.stalled = true;
                records.push(Record::Stalled { pid: e.pid, looks: e.still, waited_ms: u64::try_from(oldest.elapsed().as_millis()).unwrap_or(u64::MAX) });
            }
        }
        (out, records)
    }

    pub fn holdings(&self) -> Holdings {
        let b = self.books.lock();
        let executors: Vec<ExecutorHolding> = b
            .executors
            .values()
            .map(|e| ExecutorHolding {
                pid: e.pid,
                executor: e.identity.executor.clone(),
                chunks: e.leased.len(),
                bytes: e.leased.values().map(|l| l.phys.len).sum(),
                landed: e.leased.values().filter(|l| l.landed && l.region.is_some()).map(|l| l.phys.len).sum(),
            })
            .collect();
        Holdings {
            capacity: b.device.as_ref().map_or(0, |d| d.capacity),
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
            return end.send(&ToAgent::Refuse { id: 0, code: Refusal::NotRegistered, why: format!("pid {} is no executor of this machine", peer.pid) }, None);
        };
        let Some((ToManager::Hello { protocol, birth }, again)) = end.recv::<ToManager>()? else {
            return Err(Error::Invalid("a lease connection opened without Hello".into()));
        };
        if protocol != PROTOCOL || again.is_some() {
            // A re-attach after a manager restart comes after slice 1: that agent runs standalone.
            let why = if again.is_some() { "re-attach is not served yet".into() } else { format!("lease protocol {protocol}; this manager speaks {PROTOCOL}") };
            return end.send(&ToAgent::Refuse { id: 0, code: Refusal::Incompatible, why }, None);
        }
        // The birth of the process handles will be duplicated into (Windows: the one the pipe
        // keeps open), so a pid reused after the stream was handed on never passes.
        let found = platform::handles::peer_birth(end.stream()).map_err(|e| Error::Io(e.to_string()))?;
        if found.start_ticks != birth {
            return Err(Error::Invalid(format!("lease Hello for birth {birth} from pid {} (birth {})", peer.pid, found.start_ticks)));
        }
        // A peer no handle can reach (Windows: of a higher integrity level) runs standalone.
        if let Err(error) = platform::handles::accepts_handles(end.stream()) {
            return end.send(&ToAgent::Refuse { id: 0, code: Refusal::Incompatible, why: error.to_string() }, None);
        }
        let process = Exact::open(&found).map_err(|e| Error::Io(e.to_string()))?.ok_or_else(|| Error::Closed(format!("process {}", peer.pid)))?;
        let slots = u32::try_from(self.limit.unwrap_or(self.gpu_memory) / SLOT_UNIT).unwrap_or(u32::MAX).max(1);
        // Sealed against resizing before the agent sees it.
        let file = os::sealed_page(table::bytes(slots) as u64)?;
        let table = Arc::new(Table::create(os::map(&file, table::bytes(slots), true)?, slots, peer.pid, birth)?);
        let epoch = table.attach();
        let id = {
            let mut b = self.books.lock();
            b.next += 1;
            let id = b.next;
            let exec = Exec { pid: peer.pid, identity, table, end: end.clone(), free: (0..slots).collect(), leased: BTreeMap::new(), waiting: VecDeque::new(), asked: BTreeMap::new(), progress: 0, seen: (0, 0), still: 0, stalled: false, waits_on: BTreeSet::new() };
            b.executors.insert(id, exec);
            id
        };
        end.send(&ToAgent::Welcome { epoch }, Some(os::lend(&file)?))?;
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
        drop(out.idle);
        self.ledger.ring();
    }

    /// The backend, opened now if it is not (outside `books`); the open's failure refuses.
    fn device(&self) -> Result<Arc<Device>> {
        if let Some(d) = self.books.lock().device.clone() {
            return Ok(d);
        }
        let _one = self.opening.lock();
        if let Some(d) = self.books.lock().device.clone() {
            return Ok(d);
        }
        let backend = (self.open)()?;
        let card = self.limit.unwrap_or(self.gpu_memory).min(backend.capacity());
        let headroom = if self.mode == Mode::Shared { crate::table::headroom(self.gpu_memory) } else { 0 };
        let context = backend.opened_context();
        let capacity = card.saturating_sub(headroom).saturating_sub(context);
        let device = Arc::new(Device { capacity, granularity: backend.granularity().max(1), headroom, context, backend });
        self.books.lock().device = Some(device.clone());
        Ok(device)
    }

    /// The backend goes once nothing needs it; its drop (a context's destroy) runs unlocked.
    fn idle(b: &mut Books) -> Option<Arc<Device>> {
        (b.created == 0 && b.executors.is_empty()).then(|| b.device.take()).flatten()
    }

    fn ask(&self, exec: u64, ask: Waiting) -> Out {
        if let Err(error) = self.device() {
            let b = self.books.lock();
            let mut out = Out::default();
            if let Some(e) = b.executors.get(&exec) {
                out.sends.push((e.end.clone(), ToAgent::Refuse { id: ask.id, code: Refusal::Failed, why: error.to_string() }, None));
                out.records.push(Record::Refuse { pid: e.pid, ask: ask.id, code: Refusal::Failed, why: error.to_string() });
            }
            return out;
        }
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
                    Some((State::Mapped, seq)) if seq == l.seq => l.landed = e.table.landed(*slot),
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
        out.idle = Self::idle(&mut b);
        out
    }

    /// An exited executor's every slot: landed weights (by the manager's records) retained in
    /// its domain, the rest destroyed.
    fn exited(&self, exec: u64) -> Out {
        let start = Instant::now();
        let mut b = self.books.lock();
        let Some(e) = b.executors.remove(&exec) else { return Out::default() };
        let (mut gone, mut retained) = (vec![], 0);
        for (slot, l) in e.leased {
            // Both say landed: the manager's record of this grant's ack, and the table now (an
            // unland whose doorbell was still queued wins).
            match l.region {
                Some(region) if l.landed && e.table.landed(slot) && e.table.word(slot).is_some_and(|w| w.seq == l.seq) => {
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
        out.idle = Self::idle(&mut b);
        out
    }

    fn destroy(&self, b: &mut Books, phys: Physical) {
        b.created = b.created.saturating_sub(phys.len);
        b.created_chunks = b.created_chunks.saturating_sub(1);
        let Some(d) = &b.device else { return };
        if let Err(error) = d.backend.destroy(phys) {
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
    /// Each executor's asks in order (one waiting holds up only its own later asks): first
    /// every ask that fits now, then room made for each executor's first that does not, which
    /// then waits for it or is refused when none can come.
    fn settle(&self, b: &mut Books) -> Out {
        let mut out = Out::default();
        let ids: Vec<u64> = b.executors.keys().copied().collect();
        for exec in ids.iter().copied() {
            while let Some(ask) = b.executors.get_mut(&exec).and_then(|e| e.waiting.pop_front()) {
                match self.answer(b, exec, &ask) {
                    Ok(Some(granted)) => {
                        out.append(granted);
                        if let Some(e) = b.executors.get_mut(&exec) {
                            e.waits_on.clear();
                        }
                    }
                    Ok(None) => {
                        if let Some(e) = b.executors.get_mut(&exec) {
                            e.waiting.push_front(ask);
                        }
                        break;
                    }
                    Err(error) => refuse(b, &mut out, exec, ask.id, Refusal::Failed, error.to_string()),
                }
            }
        }
        for exec in ids {
            let Some(ask) = b.executors.get_mut(&exec).and_then(|e| e.waiting.pop_front()) else { continue };
            let (capacity, granularity) = b.device.as_ref().map_or((0, 1), |d| (d.capacity, d.granularity));
            let short = (b.created + sizes(&ask, granularity).iter().sum::<u64>()).saturating_sub(capacity);
            match self.make_room(b, exec, short, &mut out) {
                Room::Coming | Room::Pinned => {
                    if let Some(e) = b.executors.get_mut(&exec) {
                        e.waiting.push_front(ask);
                    }
                    self.looking.notify_all();
                }
                Room::Refused => refuse(b, &mut out, exec, ask.id, Refusal::NothingRevocable, format!("nothing revocable for {} bytes", ask.bytes)),
            }
        }
        self.publish(b);
        out
    }

    /// Every table's figures, and the manager's own row, as the books stand now.
    fn publish(&self, b: &Books) {
        let retained: u64 = b.retained.values().flat_map(|v| v.values()).map(|p| p.len).sum();
        let (free, headroom, context) = b.device.as_ref().map_or((0, 0, 0), |d| (d.capacity.saturating_sub(b.created), d.headroom, d.context));
        let unpinned: BTreeMap<u64, u64> = b
            .executors
            .iter()
            .map(|(id, e)| (*id, e.leased.iter().filter(|(s, l)| e.table.word(**s).is_some_and(|w| w.state == State::Mapped && w.pins == 0 && w.seq == l.seq)).map(|(_, l)| l.phys.len).sum()))
            .collect();
        let total: u64 = unpinned.values().sum();
        for (id, e) in &b.executors {
            let revocable = retained + total - unpinned.get(id).copied().unwrap_or(0);
            e.table.set_figures(Figures { free, revocable, headroom, foreign: 0 });
        }
        self.row.set(retained + context);
    }

    /// The grants for `ask` when it fits now; `None` when room must be made first. A failure
    /// partway rolls the whole ask back: no slot stays granted, no chunk leaks.
    fn answer(&self, b: &mut Books, exec: u64, ask: &Waiting) -> Result<Option<Out>> {
        let start = Instant::now();
        let Some(dev) = b.device.clone() else { return Err(Error::Closed("the GPU's backend".into())) };
        let sizes = sizes(ask, dev.granularity);
        let Some(pid) = b.executors.get(&exec).map(|e| e.pid) else { return Err(Error::Closed("the executor".into())) };
        let domain = b.executors.get(&exec).map(|e| e.identity.domain.clone()).unwrap_or_default();
        // Adopted only whole: every chunk, by index, the size this ask cuts.
        let whole = |kept: &BTreeMap<u32, Physical>| kept.len() == sizes.len() && kept.iter().zip(&sizes).enumerate().all(|(i, ((index, p), len))| *index as usize == i && p.len == *len);
        let adopt = ask.region.map(|r| (domain, r)).filter(|k| b.retained.get(k).is_some_and(whole));
        let phys: Vec<Physical> = if let Some(key) = &adopt {
            b.order.retain(|k| k != key);
            b.retained.remove(key).unwrap_or_default().into_values().collect()
        } else {
            if b.created + sizes.iter().sum::<u64>() > dev.capacity {
                return Ok(None);
            }
            let mut made = vec![];
            for &len in &sizes {
                match dev.backend.create(len) {
                    Ok(p) => made.push(p),
                    Err(error) => {
                        made.into_iter().for_each(|p| drop(dev.backend.destroy(p)));
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
            let exported = e.table.grant(slot, seq, p.len, content, adopt.is_some()).and_then(|ok| if ok { dev.backend.lend(&p).map(Some) } else { Ok(None) });
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
        // An executor waiting (transitively) on the asker's pins is never waited on: a cycle.
        let cyclic = waiting_on(b, asker);
        let (mut later, mut unpinned, mut pinned) = (0, vec![], vec![]);
        for (id, e) in &b.executors {
            for (slot, l) in &e.leased {
                let Some(w) = e.table.word(*slot) else { continue };
                if w.seq != l.seq {
                    continue;
                }
                let free_of_cycle = *id != asker && !cyclic.contains(id);
                match (w.state, w.pins, e.asked.contains_key(slot)) {
                    (State::Revoking, ..) | (State::Mapped, 0, true) => room += w_bytes(e, *slot),
                    (State::Mapped, _, true) if free_of_cycle => later += w_bytes(e, *slot),
                    (State::Mapped, 0, false) => {
                        let (class, next) = e.table.class(*slot);
                        unpinned.push((class, std::cmp::Reverse(next), *id, *slot, l.seq));
                    }
                    (State::Mapped, _, false) if free_of_cycle => pinned.push((*id, *slot, l.seq)),
                    _ => {}
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
            e.asked.insert(slot, Instant::now());
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
            let mut on = BTreeSet::new();
            for (id, slot, seq) in pinned {
                if room + later >= short {
                    break;
                }
                later += ask_back(b, id, slot, seq);
                on.insert(id);
            }
            if room + later >= short {
                // Whose pins it waits on: every executor holding an asked, still-pinned chunk.
                for (id, e) in &b.executors {
                    if *id != asker && e.asked.keys().any(|s| e.table.word(*s).is_some_and(|w| w.state == State::Mapped && w.pins > 0)) {
                        on.insert(*id);
                    }
                }
                if let Some(e) = b.executors.get_mut(&asker) {
                    e.waits_on = on;
                }
                Room::Pinned
            } else {
                Room::Refused
            }
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

fn w_bytes(e: &Exec, slot: u32) -> u64 {
    e.table.bytes_of(slot)
}

/// Every executor whose waiting ask waits, directly or through others, on `of`'s pins.
fn waiting_on(b: &Books, of: u64) -> BTreeSet<u64> {
    let mut found = BTreeSet::new();
    let mut frontier = vec![of];
    while let Some(x) = frontier.pop() {
        for (id, e) in &b.executors {
            if !e.waiting.is_empty() && e.waits_on.contains(&x) && found.insert(*id) {
                frontier.push(*id);
            }
        }
    }
    found
}

fn refuse(b: &mut Books, out: &mut Out, exec: u64, ask: u64, code: Refusal, why: String) {
    let Some(e) = b.executors.get_mut(&exec) else { return };
    e.waits_on.clear();
    out.sends.push((e.end.clone(), ToAgent::Refuse { id: ask, code, why: why.clone() }, None));
    out.records.push(Record::Refuse { pid: e.pid, ask, code, why });
}

/// The stall look: every [`LOOK`] while asks are out, else asleep until one goes out.
fn look_loop(manager: &Weak<Manager>) {
    loop {
        let Some(m) = manager.upgrade() else { return };
        let (out, records) = m.look();
        for record in records {
            m.hosting.record(record);
        }
        let mut b = m.books.lock();
        if out {
            m.looking.wait_for(&mut b, LOOK);
        } else {
            m.looking.wait_for(&mut b, LOOK * 20);
        }
    }
}

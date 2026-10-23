//! The lease manager (varena-leases §1, §2, §11): it creates every physical chunk, grants chunks
//! to executors' agents, revokes for room, and takes a dead executor's leases back once its exit
//! shows: landed weights are retained for the next executor of that layout (no copy), the rest
//! destroyed. Only the manager exports. TensorD hosts it; tests host it in-process.
//!
//! One manager per GPU (§2a): TensorD makes it where it made the ledger, hands it the GPU's
//! listener, and reacts to its events fd. Until step 4 the admission and placement facts stay the
//! ledger's, which the manager owns ([`Manager::ledger`]).
//!
//! Locks: `books` is the only one, and nothing blocks under it (sends are made after it drops).

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::Arc;

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
    /// The trust domain retained weights are shared within: publisher, package, version, user.
    pub domain: String,
}

/// What TensorD answers for the manager.
pub trait Hosting: Send + Sync {
    /// The identity of a peer on the GPU's listener; `None` refuses it (not an executor of ours).
    fn identify(&self, peer: &Peer) -> Option<Identity>;
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
    seq: u32,
    next: u64,
    executors: BTreeMap<u64, Exec>,
    /// Landed weights of exited executors: a region's chunks by index, oldest region first in
    /// `order`.
    retained: BTreeMap<Region, BTreeMap<u32, Physical>>,
    order: VecDeque<Region>,
}

struct Exec {
    table: Arc<Table>,
    end: Arc<End>,
    free: BTreeSet<u32>,
    leased: BTreeMap<u32, Leased>,
    waiting: VecDeque<Waiting>,
    /// Revokes asked and not yet answered: no new ask is made while one is out.
    asked: BTreeSet<u32>,
    /// `acked` when the oldest outstanding ask went out: no growth since is a stall.
    acked_at_ask: u64,
}

struct Leased {
    phys: Physical,
    seq: u32,
    region: Option<Region>,
}

struct Waiting {
    id: u64,
    bytes: u64,
    region: Option<Region>,
}

/// What to send once `books` is dropped.
type Sends = Vec<(Arc<End>, ToAgent, Option<Owned>)>;

/// What a manager holds, for its host's status and the tests.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Holdings {
    pub created: u64,
    pub leased: u64,
    pub retained: u64,
    pub executors: usize,
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

impl Manager {
    /// The GPU's manager: its ledger made, `listener` served on the manager's own thread until
    /// [`Manager::retire`]. `limit` caps the bytes it creates (a `--gpu-memory-limit`), else the
    /// card's.
    pub fn create(gpu: &GpuId, limit: Option<u64>, mode: Mode, backend: Arc<dyn Backend>, listener: Listener, hosting: Arc<dyn Hosting>) -> Result<Arc<Manager>> {
        let ledger = Ledger::create(gpu, limit)?;
        ledger.set_mode(mode)?;
        let capacity = limit.unwrap_or(gpu.memory).min(backend.capacity());
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
                        Ok(stream) => {
                            if let Err(error) = serving.serve(stream) {
                                tracing::warn!("lease: serving an agent: {error}");
                            }
                        }
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

    /// Executors with revokes outstanding that answer none (judged by progress, never a clock),
    /// and the ledger's stalled rows.
    pub fn stalled(&self) -> Vec<u32> {
        let mut pids = self.ledger.stalled();
        let b = self.books.lock();
        for e in b.executors.values() {
            if !e.asked.is_empty() && e.table.acked() == e.acked_at_ask {
                pids.push(e.table.pid());
            }
        }
        pids.sort_unstable();
        pids.dedup();
        pids
    }

    pub fn set_mode(&self, mode: Mode) -> Result<()> {
        self.ledger.set_mode(mode)
    }

    /// Stops accepting agents; connected ones keep their leases until they exit.
    pub fn retire(&self) {
        self.retired.store(true, Relaxed);
    }

    pub fn holdings(&self) -> Holdings {
        let b = self.books.lock();
        Holdings {
            created: b.created,
            leased: b.executors.values().flat_map(|e| e.leased.values()).map(|l| l.phys.len).sum(),
            retained: b.retained.values().flat_map(|v| v.values()).map(|p| p.len).sum(),
            executors: b.executors.len(),
        }
    }

    /// Serves one executor's connection on its own thread until the executor has exited and its
    /// leases are back.
    fn serve(self: &Arc<Self>, stream: Stream) -> Result<()> {
        let manager = self.clone();
        std::thread::Builder::new()
            .name("lease-executor".into())
            .spawn(move || {
                if let Err(error) = manager.executor(stream) {
                    tracing::warn!("lease: an executor's connection ended: {error}");
                }
            })
            .map(drop)
            .map_err(|e| Error::io("spawn lease-executor", e))
    }

    fn executor(self: &Arc<Self>, stream: Stream) -> Result<()> {
        let peer = platform::handles::peer_credentials(&stream).map_err(|e| Error::Io(e.to_string()))?;
        let end = Arc::new(End::new(stream));
        let Some(_identity) = self.hosting.identify(&peer) else {
            end.send(&ToAgent::Refuse { id: 0, why: "not an executor of this machine".into() }, None)?;
            return Ok(());
        };
        let Some((ToManager::Hello { protocol, birth }, again)) = end.recv::<ToManager>()? else {
            return Err(Error::Invalid("a lease connection opened without Hello".into()));
        };
        if protocol != PROTOCOL {
            end.send(&ToAgent::Refuse { id: 0, why: format!("lease protocol {protocol}; this manager speaks {PROTOCOL}") }, None)?;
            return Ok(());
        }
        let found = Birth::of(peer.pid).map_err(|e| Error::Io(e.to_string()))?;
        if found.start_ticks != birth {
            return Err(Error::Invalid(format!("lease Hello for birth {birth} from pid {} (birth {})", peer.pid, found.start_ticks)));
        }
        let process = Exact::open(&found).map_err(|e| Error::Io(e.to_string()))?.ok_or_else(|| Error::Closed(format!("process {}", peer.pid)))?;
        if again.is_some() {
            // A re-attach after a manager restart: every chunk it holds is books-only (§2).
            return Err(Error::Invalid("lease re-attach after a manager restart is not built yet (slice 1)".into()));
        }
        let slots = u32::try_from(self.capacity / self.granularity.max(1)).unwrap_or(u32::MAX).max(1);
        let file = platform::memory::page(table::bytes(slots) as u64).map_err(|e| Error::Io(e.to_string()))?;
        let table = Arc::new(Table::create(os::map(&file, table::bytes(slots), true)?, slots, peer.pid, birth, [0, 0])?);
        let epoch = table.attach();
        let id = {
            let mut b = self.books.lock();
            b.next += 1;
            let id = b.next;
            b.executors.insert(id, Exec { table: table.clone(), end: end.clone(), free: (0..slots).collect(), leased: BTreeMap::new(), waiting: VecDeque::new(), asked: BTreeSet::new(), acked_at_ask: 0 });
            id
        };
        end.send(&ToAgent::Welcome { epoch }, Some(os::dup(&file)?))?;
        drop(file);
        while let Some((message, _)) = end.recv::<ToManager>()? {
            let sends = match message {
                ToManager::Ask { id: ask, bytes, region } => self.ask(id, Waiting { id: ask, bytes, region }),
                ToManager::Answered => self.answered(id),
                ToManager::Hello { .. } => vec![],
            };
            send(sends);
        }
        // The socket closed: its leases come back once the process's exit shows (its mappings
        // are gone then), never before.
        process.wait().map_err(|e| Error::Io(e.to_string()))?;
        send(self.exited(id));
        self.ledger.ring();
        Ok(())
    }

    fn ask(&self, exec: u64, ask: Waiting) -> Sends {
        let mut b = self.books.lock();
        if let Some(e) = b.executors.get_mut(&exec) {
            e.waiting.push_back(ask);
        }
        self.settle(&mut b)
    }

    /// The agent moved slots: Released ones are destroyed, Refused ones too; then waiting asks.
    fn answered(&self, exec: u64) -> Sends {
        let mut b = self.books.lock();
        let mut gone = vec![];
        if let Some(e) = b.executors.get_mut(&exec) {
            let done: Vec<u32> = e.leased.keys().copied().filter(|&s| matches!(e.table.word(s).map(|w| w.state), Some(State::Released | State::Refused))).collect();
            for slot in done {
                if let (Ok(Some(_)), Some(l)) = (e.table.reclaim(slot, false), e.leased.remove(&slot)) {
                    e.free.insert(slot);
                    e.asked.remove(&slot);
                    gone.push(l.phys);
                }
            }
        }
        for phys in gone {
            self.destroy(&mut b, phys);
        }
        self.settle(&mut b)
    }

    /// An exited executor's every slot: landed weights retained, the rest destroyed.
    fn exited(&self, exec: u64) -> Sends {
        let mut b = self.books.lock();
        let Some(e) = b.executors.remove(&exec) else { return vec![] };
        let mut gone = vec![];
        for (slot, l) in e.leased {
            let landed = e.table.landed(slot) && e.table.word(slot).is_some_and(|w| w.state == State::Mapped);
            match (l.region, e.table.content(slot)) {
                (Some(region), Some(content)) if landed => {
                    let kept = b.retained.entry(region).or_default();
                    let first = kept.is_empty();
                    gone.extend(kept.insert(content.index, l.phys));
                    if first {
                        b.order.push_back(region);
                    }
                }
                _ => gone.push(l.phys),
            }
        }
        for phys in gone {
            self.destroy(&mut b, phys);
        }
        self.settle(&mut b)
    }

    fn destroy(&self, b: &mut Books, phys: Physical) {
        b.created = b.created.saturating_sub(phys.len);
        if let Err(error) = self.backend.destroy(phys) {
            tracing::warn!("lease: destroying a chunk: {error}");
        }
    }

    fn drop_retained(&self, b: &mut Books, region: Region) {
        b.order.retain(|r| *r != region);
        for phys in b.retained.remove(&region).unwrap_or_default().into_values() {
            self.destroy(b, phys);
        }
    }

    /// Answers waiting asks in order while room allows; makes room (retained first, then revokes
    /// of unpinned chunks, furthest next use first) for the first one that does not fit; refuses
    /// it when nothing is revocable.
    fn settle(&self, b: &mut Books) -> Sends {
        let mut sends = vec![];
        loop {
            let Some((exec, ask)) = b.executors.iter_mut().find_map(|(id, e)| e.waiting.pop_front().map(|w| (*id, w))) else { break };
            match self.answer(b, exec, &ask) {
                Ok(Some(mut out)) => sends.append(&mut out),
                Ok(None) => {
                    // Not now: room is coming (asked) or it is refused.
                    let short = (b.created + chunks(ask.bytes, self.granularity).iter().sum::<u64>()).saturating_sub(self.capacity);
                    let coming = self.make_room(b, short, &mut sends);
                    let Some(e) = b.executors.get_mut(&exec) else { continue };
                    if coming {
                        e.waiting.push_front(ask);
                        break;
                    }
                    sends.push((e.end.clone(), ToAgent::Refuse { id: ask.id, why: format!("nothing revocable for {} bytes", ask.bytes) }, None));
                }
                Err(error) => {
                    if let Some(e) = b.executors.get(&exec) {
                        sends.push((e.end.clone(), ToAgent::Refuse { id: ask.id, why: error.to_string() }, None));
                    }
                }
            }
        }
        sends
    }

    /// The grants for `ask` when it fits now; `None` when room must be made first.
    fn answer(&self, b: &mut Books, exec: u64, ask: &Waiting) -> Result<Option<Sends>> {
        let sizes = chunks(ask.bytes, self.granularity);
        // Adopted only whole: every chunk, by index, the size this ask cuts.
        let whole = |kept: &BTreeMap<u32, Physical>| kept.len() == sizes.len() && kept.iter().zip(&sizes).enumerate().all(|(i, ((index, p), len))| *index as usize == i && p.len == *len);
        let adopt = ask.region.filter(|r| b.retained.get(r).is_some_and(whole));
        let phys: Vec<Physical> = if let Some(region) = adopt {
            b.order.retain(|r| *r != region);
            b.retained.remove(&region).unwrap_or_default().into_values().collect()
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
            made
        };
        let n = phys.len();
        let mut sends = vec![];
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
                continue;
            };
            let Some(slot) = e.free.pop_first() else {
                spare.push(p);
                failed = Some(Error::Invalid("the executor's lease table is full".into()));
                continue;
            };
            let content = ask.region.map(|r| Content { fingerprint: r.fingerprint, region: r.region, index: index as u32 });
            let granted = e.table.grant(slot, seq, p.len, content, adopt.is_some()).and_then(|ok| if ok { self.backend.export(&p).map(Some) } else { Ok(None) });
            match granted {
                Ok(Some(handle)) => {
                    sends.push((e.end.clone(), ToAgent::Grant { id: ask.id, slot, seq, bytes: p.len, adopted: adopt.is_some(), last: index + 1 == n }, Some(handle)));
                    e.leased.insert(slot, Leased { phys: p, seq, region: ask.region });
                }
                Ok(None) => {
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
        for p in spare {
            self.destroy(b, p);
        }
        match failed {
            Some(error) => Err(error),
            None => Ok(Some(sends)),
        }
    }

    /// Frees retained regions, oldest first, then asks revokes of unpinned chunks (idle class
    /// first, then furthest next use) until `short` bytes are free or coming. True when room is
    /// coming or made.
    fn make_room(&self, b: &mut Books, short: u64, sends: &mut Sends) -> bool {
        let mut room = 0;
        while room < short {
            let Some(region) = b.order.front().copied() else { break };
            room += b.retained.get(&region).map_or(0, |v| v.values().map(|p| p.len).sum());
            self.drop_retained(b, region);
        }
        if room >= short {
            return true;
        }
        let coming: u64 = b.executors.values().flat_map(|e| e.asked.iter().map(|s| e.table.bytes_of(*s))).sum();
        room += coming;
        let mut victims: Vec<(u32, std::cmp::Reverse<u64>, u64, u32, u32)> = vec![];
        for (id, e) in &b.executors {
            for (slot, l) in &e.leased {
                let Some(w) = e.table.word(*slot) else { continue };
                if w.state == State::Mapped && w.pins == 0 && !e.asked.contains(slot) {
                    let (class, next) = e.table.class(*slot);
                    victims.push((class, std::cmp::Reverse(next), *id, *slot, l.seq));
                }
            }
        }
        victims.sort();
        let mut rang = BTreeSet::new();
        for (_, _, id, slot, seq) in victims {
            if room >= short {
                break;
            }
            let Some(e) = b.executors.get_mut(&id) else { continue };
            if e.table.request(slot, seq).is_ok() {
                if e.asked.is_empty() {
                    e.acked_at_ask = e.table.acked();
                }
                e.asked.insert(slot);
                room += e.table.bytes_of(slot);
                rang.insert(id);
            }
        }
        for id in rang {
            if let Some(e) = b.executors.get(&id) {
                sends.push((e.end.clone(), ToAgent::Revoke, None));
            }
        }
        room >= short && room > 0
    }
}

fn send(sends: Sends) {
    for (end, message, handle) in sends {
        if let Err(error) = end.send(&message, handle) {
            tracing::debug!("lease: a send to an executor failed (it is going): {error}");
        }
    }
}

//! The lease agent (varena-leases §1, §11), one per model process and GPU: it asks the manager
//! for chunks and imports them; its holder (varena's memory) maps them at its own VAs, pins them,
//! and gives them back when asked once they are quiet (no pins, the stream epoch passed, fills
//! landed). It never exports and makes no policy decision. Slice 1 leases weights only.
//!
//! An ask is answered on the reader thread ([`Agent::ask`]): the holder's mechanism thread never
//! waits on the manager, which may be waiting on this process's own revokes.
//!
//! Locks: `pending` and `held` are leaves; the reader thread imports outside them.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use parking_lot::{Condvar, Mutex};
use platform::process::Birth;

use super::policy::Advice;
use super::table::{State, Table};
use super::wire::{End, Kind, Measured, Refusal, Region, RegionFacts, ToAgent, ToManager};
use crate::backend::{Backend, Physical};
use crate::os::{self, Stream};
use crate::{Error, Exhausted, Result};

/// One grant: its table slot and the sequence it was granted under.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Ticket {
    pub slot: u32,
    pub seq: u32,
}

/// One imported chunk: the holder maps `phys` and answers it with [`Agent::mapped`] or
/// [`Agent::refused`].
#[derive(Debug)]
pub struct Lease {
    pub ticket: Ticket,
    pub phys: Physical,
    /// Its bytes are its region's already (retained by the manager): no fill.
    pub adopted: bool,
    /// A derived entry kept for the ask's fill key (on its first lease): the bytes refill from
    /// it, with no transform.
    pub entry: Option<std::fs::File>,
}

/// What runs on the reader thread once an ask is answered.
pub type Done = Box<dyn FnOnce(Result<Vec<Lease>>) + Send>;

/// What runs on the reader thread once a derive is answered: the entry to write, or none kept.
pub type Deriving = Box<dyn FnOnce(Option<super::DerivedEntry>) + Send>;

struct Asking {
    bytes: u64,
    got: Vec<Lease>,
    entry: Option<std::fs::File>,
    done: Done,
}

#[derive(Default)]
struct Pending {
    next: u64,
    asks: HashMap<u64, Asking>,
    /// Asks a failed import ended: their later chunks are refused unimported.
    dead: HashSet<u64>,
    /// Derives by key, oldest first.
    derives: HashMap<String, Vec<Deriving>>,
    closed: bool,
}

/// What an attach came to.
pub enum Attach {
    Leased(Arc<Agent>),
    /// `NotRegistered`: redial once registered; `Incompatible`: run that GPU standalone.
    Refused { code: Refusal, why: String },
}

pub struct Agent {
    end: Arc<End>,
    table: Arc<Table>,
    backend: Arc<dyn Backend>,
    pending: Mutex<Pending>,
    /// Mapped grants this process holds.
    held: Mutex<BTreeSet<Ticket>>,
    /// The holder's wake for asks back (the doorbell's, and a grant asked back before its map).
    revoked: Box<dyn Fn() + Send + Sync>,
    /// The holder's wake for a region's reclaim ([`Agent::on_reclaim`]).
    reclaim: OnceLock<Box<dyn Fn(u64, bool) + Send + Sync>>,
}

impl Agent {
    /// Attaches to the manager serving `stream`; `revoked` runs on the reader thread at every
    /// Revoke doorbell (the holder wakes its mechanism thread, never works there). A refusal
    /// comes back typed: the caller redials or runs that GPU standalone, never fails for it.
    pub fn attach(stream: Stream, backend: Arc<dyn Backend>, revoked: impl Fn() + Send + Sync + 'static) -> Result<Attach> {
        let end = Arc::new(End::new(stream));
        let me = Birth::own().map_err(|e| Error::Io(e.to_string()))?;
        // A refusal can land before the Hello does: whatever the send says, read the answer.
        let hello = end.send(&ToManager::Hello { protocol: super::manager::PROTOCOL, birth: me.start_ticks }, None);
        let answer = match end.recv::<ToAgent>() {
            Ok(answer) => answer,
            Err(error) => return Err(hello.err().unwrap_or(error)),
        };
        let table = match answer {
            Some((ToAgent::Welcome { .. }, Some(file))) => {
                let len = os::size(&file)? as usize;
                Arc::new(Table::open(os::map(&file, len, true)?)?)
            }
            Some((ToAgent::Refuse { code, why, .. }, _)) => return Ok(Attach::Refused { code, why }),
            _ => return Err(Error::Closed("the lease manager answered Hello with no table".into())),
        };
        let agent = Arc::new(Agent { end, table, backend, pending: Mutex::new(Pending::default()), held: Mutex::new(BTreeSet::new()), revoked: Box::new(revoked), reclaim: OnceLock::new() });
        let reader = agent.clone();
        std::thread::Builder::new()
            .name("lease-agent".into())
            .spawn(move || {
                if let Err(error) = reader.read() {
                    tracing::warn!("lease: the manager's connection failed: {error}");
                }
                // Every ask still out fails, every derive keeps nothing: the manager is gone.
                let (asks, derives): (Vec<Asking>, Vec<Deriving>) = {
                    let mut p = reader.pending.lock();
                    p.closed = true;
                    (p.asks.drain().map(|(_, a)| a).collect(), p.derives.drain().flat_map(|(_, d)| d).collect())
                };
                for a in asks {
                    reader.give_back(a.got);
                    (a.done)(Err(Error::Closed("the lease manager".into())));
                }
                derives.into_iter().for_each(|d| d(None));
            })
            .map_err(|e| Error::io("spawn lease-agent", e))?;
        Ok(Attach::Leased(agent))
    }

    fn read(&self) -> Result<()> {
        while let Some((message, handle)) = self.end.recv::<ToAgent>()? {
            match message {
                ToAgent::Grant { id, slot, seq, bytes, adopted, last } => self.granted(id, Ticket { slot, seq }, bytes, adopted, last, handle)?,
                ToAgent::Refuse { id, code, why } => {
                    let asking = self.pending.lock().asks.remove(&id);
                    if let Some(a) = asking {
                        self.give_back(a.got);
                        (a.done)(Err(match code {
                            Refusal::NothingRevocable => exhausted(a.bytes, &why),
                            _ => Error::Closed(format!("the lease manager: {why}")),
                        }));
                    }
                }
                ToAgent::Revoke => (self.revoked)(),
                ToAgent::Reclaim { region, swap } => match self.reclaim.get() {
                    Some(wake) => wake(region, swap),
                    None => tracing::warn!("lease: region {region}'s reclaim reached an agent with no region holder"),
                },
                ToAgent::Entry { id: 0, key, entry } => {
                    let deriving = self.pending.lock().derives.get_mut(&key).and_then(|d| (!d.is_empty()).then(|| d.remove(0)));
                    match deriving {
                        Some(done) => done(handle.map(|h| super::DerivedEntry { id: entry, file: std::fs::File::from(h) })),
                        None => tracing::warn!("lease: an entry for {key} nobody derives"),
                    }
                }
                ToAgent::Entry { id, .. } => {
                    if let Some(a) = self.pending.lock().asks.get_mut(&id) {
                        a.entry = handle.map(std::fs::File::from);
                    }
                }
                ToAgent::Welcome { .. } => {}
            }
        }
        Ok(())
    }

    fn granted(&self, id: u64, ticket: Ticket, bytes: u64, adopted: bool, last: bool, handle: Option<os::Owned>) -> Result<()> {
        let dead = {
            let mut p = self.pending.lock();
            let dead = p.dead.contains(&id) || !p.asks.contains_key(&id);
            if dead && last {
                p.dead.remove(&id);
            }
            dead
        };
        if dead {
            drop(handle);
            self.table.answer(ticket.slot, ticket.seq, false)?;
            return self.end.send(&ToManager::Answered, None);
        }
        match handle.ok_or_else(|| Error::Invalid("a grant with no handle".into())).and_then(|h| self.backend.import(h, bytes)) {
            Ok(phys) => {
                let done = {
                    let mut p = self.pending.lock();
                    let Some(a) = p.asks.get_mut(&id) else { return Ok(()) };
                    let entry = a.entry.take();
                    a.got.push(Lease { ticket, phys, adopted, entry });
                    if last { p.asks.remove(&id) } else { None }
                };
                if let Some(a) = done {
                    (a.done)(Ok(a.got));
                }
                Ok(())
            }
            Err(error) => {
                // The whole ask fails: this chunk and every one imported for it go back.
                let asking = {
                    let mut p = self.pending.lock();
                    if !last {
                        p.dead.insert(id);
                    }
                    p.asks.remove(&id)
                };
                self.table.answer(ticket.slot, ticket.seq, false)?;
                if let Some(a) = asking {
                    self.give_back(a.got);
                    (a.done)(Err(error));
                }
                self.end.send(&ToManager::Answered, None)
            }
        }
    }

    /// Imported leases nobody will map: refused, their imports gone.
    fn give_back(&self, leases: Vec<Lease>) {
        if leases.is_empty() {
            return;
        }
        for lease in leases {
            let _ = self.backend.destroy(lease.phys);
            let _ = self.table.answer(lease.ticket.slot, lease.ticket.seq, false);
        }
        let _ = self.end.send(&ToManager::Answered, None);
    }

    /// Asks for `bytes` of weights for `region` (`fill`: its declared fill key); `done` runs on
    /// the reader thread with the imported leases (the holder maps each and answers it) or the
    /// refusal (Exhausted), once the manager answers: after room is made, which may take other
    /// processes' revokes.
    pub fn ask(&self, bytes: u64, kind: Kind, region: Option<Region>, fill: Option<String>, done: Done) -> Result<()> {
        let id = {
            let mut p = self.pending.lock();
            if p.closed {
                drop(p);
                done(Err(Error::Closed("the lease manager".into())));
                return Ok(());
            }
            p.next += 1;
            let id = p.next;
            p.asks.insert(id, Asking { bytes, got: vec![], entry: None, done });
            id
        };
        self.end.send(&ToManager::Ask { id, bytes, kind, region, fill }, None)
    }

    /// Asks the host for a disk entry to write `key`'s derived bytes into; `done` runs on the
    /// reader thread with it, or with none (none kept, or a manager that keeps none). Once it
    /// is written whole, [`Agent::derived`].
    pub fn derive(&self, key: String, bytes: u64, done: Deriving) -> Result<()> {
        {
            let mut p = self.pending.lock();
            if p.closed {
                drop(p);
                done(None);
                return Ok(());
            }
            p.derives.entry(key.clone()).or_default().push(done);
        }
        self.end.send(&ToManager::Derive { key, bytes }, None)
    }

    /// Entry `entry` (its [`super::DerivedEntry::id`]) of `key` is written whole: grants refill
    /// from it from now on.
    pub fn derived(&self, key: String, entry: u64) -> Result<()> {
        self.end.send(&ToManager::Derived { key, entry }, None)
    }

    /// [`Agent::ask`], waiting for the answer here: never on a thread that answers revokes.
    pub fn take(&self, bytes: u64, kind: Kind, region: Option<Region>) -> Result<Vec<Lease>> {
        let answer = Arc::new((Mutex::new(None), Condvar::new()));
        let set = answer.clone();
        self.ask(
            bytes,
            kind,
            region,
            None,
            Box::new(move |r| {
                *set.0.lock() = Some(r);
                set.1.notify_all();
            }),
        )?;
        let mut got = answer.0.lock();
        loop {
            if let Some(r) = got.take() {
                return r;
            }
            answer.1.wait(&mut got);
        }
    }

    /// The holder mapped these: they are its to pin.
    pub fn mapped(&self, tickets: &[Ticket]) -> Result<()> {
        for t in tickets {
            self.table.answer(t.slot, t.seq, true)?;
            self.held.lock().insert(*t);
        }
        self.end.send(&ToManager::Answered, None)?;
        // Asked back while in flight (L3: the revoke follows the map): the holder hears it now.
        if tickets.iter().any(|t| self.asked(*t)) {
            (self.revoked)();
        }
        Ok(())
    }

    /// The holder could not map it: its import goes, and the manager reclaims the chunk.
    pub fn refused(&self, ticket: Ticket, phys: Physical) -> Result<()> {
        self.backend.destroy(phys)?;
        self.table.answer(ticket.slot, ticket.seq, false)?;
        self.end.send(&ToManager::Answered, None)
    }

    /// Its fill landed: the manager records it at this ack (a landed weights chunk outlives
    /// this process in retention).
    pub fn landed(&self, ticket: Ticket) -> Result<()> {
        self.table.set_landed(ticket.slot)?;
        self.end.send(&ToManager::Answered, None)
    }

    /// GPU work of this process completed (an event its holder waits on): a revoke behind a
    /// long kernel is progressing, never stalled.
    pub fn progressed(&self) {
        self.table.progressed();
    }

    /// Its bytes are no longer its region's weights (a streamed slot moved): never retained.
    pub fn unland(&self, ticket: Ticket) -> Result<()> {
        self.table.set_landed_to(ticket.slot, false)?;
        self.end.send(&ToManager::Answered, None)
    }

    /// GPU work its holder recorded is (not) yet complete: a busy agent is never stalled.
    pub fn busy(&self, busy: bool) {
        self.table.set_busy(busy);
    }

    /// A request (not idle work) is (not) open in its holder: only then are its pins waited on.
    pub fn running(&self, running: bool) {
        self.table.set_running(running);
    }

    pub fn pin(&self, ticket: Ticket) -> Result<bool> {
        self.table.pin(ticket.slot)
    }

    pub fn unpin(&self, ticket: Ticket) -> Result<()> {
        self.table.unpin(ticket.slot)
    }

    pub fn advise(&self, ticket: Ticket, advice: Advice) -> Result<()> {
        self.table.advise(ticket.slot, advice)
    }

    /// Whether the manager asked for this lease back.
    pub fn asked(&self, ticket: Ticket) -> bool {
        self.table.ask(ticket.slot) == ticket.seq
    }

    /// Every held lease the manager asked back.
    pub fn asked_back(&self) -> Vec<Ticket> {
        self.held.lock().iter().copied().filter(|t| self.asked(*t)).collect()
    }

    /// Commits a give-back (asked or not) when the holder decides it, before its fence: false
    /// while pinned. The holder then waits for the fence, unmaps and calls [`Agent::release`].
    pub fn commit(&self, ticket: Ticket) -> Result<bool> {
        self.table.commit(ticket.slot, ticket.seq)
    }

    /// The holder decided to give it back (before its fence): its pins in the table go and the
    /// return is committed, so the manager sees it progressing. A grant not yet mapped is left.
    pub fn decide(&self, ticket: Ticket) -> Result<()> {
        while let Some(w) = self.table.word(ticket.slot).filter(|w| w.seq == ticket.seq && w.state == State::Mapped) {
            if w.pins == 0 {
                self.table.commit(ticket.slot, ticket.seq)?;
                break;
            }
            self.table.unpin(ticket.slot)?;
        }
        Ok(())
    }

    /// The holder is done with it (unmapped, or never mapped): back to the manager, whatever
    /// its state.
    pub fn give(&self, ticket: Ticket, phys: Physical) -> Result<()> {
        match self.table.word(ticket.slot).filter(|w| w.seq == ticket.seq).map(|w| w.state) {
            Some(State::Granted) => self.refused(ticket, phys),
            Some(State::Mapped) => {
                self.decide(ticket)?;
                self.release(ticket, phys)
            }
            Some(State::Revoking) => self.release(ticket, phys),
            _ => self.backend.destroy(phys),
        }
    }

    /// The committed lease is unmapped: its import goes, and the manager frees the chunk.
    pub fn release(&self, ticket: Ticket, phys: Physical) -> Result<()> {
        let committed = self.table.word(ticket.slot).is_some_and(|w| w.state == State::Revoking && w.seq == ticket.seq);
        if !committed {
            return Err(Error::Invalid(format!("lease slot {} released before its commit", ticket.slot)));
        }
        self.held.lock().remove(&ticket);
        self.backend.destroy(phys)?;
        self.table.released(ticket.slot, ticket.seq)?;
        self.end.send(&ToManager::Answered, None)
    }

    /// The holder's wake for the manager's Reclaim of a region, on the reader thread (it posts to
    /// its mechanism thread, never works there). Set once.
    pub fn on_reclaim(&self, wake: Box<dyn Fn(u64, bool) + Send + Sync>) -> Result<()> {
        self.reclaim.set(wake).map_err(|_| Error::Invalid("the agent's reclaim holder is set already".into()))
    }

    /// A region's facts for the manager's policy, at its create and its changes.
    pub fn region(&self, facts: RegionFacts) -> Result<()> {
        self.end.send(&ToManager::Region(facts), None)
    }

    pub fn region_gone(&self, id: u64) -> Result<()> {
        self.end.send(&ToManager::RegionGone { id }, None)
    }

    pub fn table(&self) -> &Table {
        &self.table
    }

    /// A request's peaks at its end, or a set's bytes at its load.
    pub fn measured(&self, m: Measured) -> Result<()> {
        self.end.send(&ToManager::Measured(m), None)
    }
}

fn exhausted(bytes: u64, why: &str) -> Error {
    Error::Exhausted(Box::new(Exhausted { need: bytes, site: Some(format!("lease: {why}")), stats: crate::memory::Stats::default(), foreign: None }))
}

//! The lease agent (varena-leases §1, §11), one per model process and GPU: it asks the manager
//! for chunks and imports them; its holder (varena's memory) maps them at its own VAs, pins them,
//! and gives them back when asked once they are quiet (no pins, the stream epoch passed, fills
//! landed). It never exports and makes no policy decision. Slice 1 leases weights only.
//!
//! Locks: `pending` only, a leaf; the reader thread imports outside it.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};
use platform::process::Birth;

use super::table::{State, Table};
use super::wire::{End, Region, ToAgent, ToManager};
use crate::backend::{Backend, Physical};
use crate::os::{self, Stream};
use crate::{Error, Exhausted, Result};

/// One imported chunk: the holder maps `phys` and answers it with [`Agent::mapped`] or
/// [`Agent::refused`].
#[derive(Debug)]
pub struct Lease {
    pub slot: u32,
    pub seq: u32,
    pub phys: Physical,
    /// Its bytes are its region's already (retained by the manager): no fill.
    pub adopted: bool,
}

enum Answer {
    Granting(Vec<Lease>),
    Done(Vec<Lease>),
    Refused(String),
}

#[derive(Default)]
struct Pending {
    next: u64,
    asks: HashMap<u64, Answer>,
    closed: bool,
}

pub struct Agent {
    end: Arc<End>,
    table: Arc<Table>,
    backend: Arc<dyn Backend>,
    pending: Mutex<Pending>,
    answered: Condvar,
}

impl Agent {
    /// Attaches to the manager serving `stream`; `revoked` runs on the reader thread at every
    /// Revoke doorbell (the holder wakes its mechanism thread, never works there).
    pub fn attach(stream: Stream, backend: Arc<dyn Backend>, revoked: impl Fn() + Send + 'static) -> Result<Arc<Agent>> {
        let end = Arc::new(End::new(stream));
        let me = Birth::own().map_err(|e| Error::Io(e.to_string()))?;
        end.send(&ToManager::Hello { protocol: super::manager::PROTOCOL, birth: me.start_ticks }, None)?;
        let table = match end.recv::<ToAgent>()? {
            Some((ToAgent::Welcome { .. }, Some(file))) => {
                let len = os::size(&file)? as usize;
                Arc::new(Table::open(os::map(&file, len, true)?)?)
            }
            Some((ToAgent::Refuse { why, .. }, _)) => return Err(Error::Closed(format!("the lease manager: {why}"))),
            _ => return Err(Error::Closed("the lease manager answered Hello with no table".into())),
        };
        let agent = Arc::new(Agent { end, table, backend, pending: Mutex::new(Pending::default()), answered: Condvar::new() });
        let reader = agent.clone();
        std::thread::Builder::new()
            .name("lease-agent".into())
            .spawn(move || {
                if let Err(error) = reader.read(&revoked) {
                    tracing::warn!("lease: the manager's connection failed: {error}");
                }
                reader.pending.lock().closed = true;
                reader.answered.notify_all();
            })
            .map_err(|e| Error::io("spawn lease-agent", e))?;
        Ok(agent)
    }

    fn read(&self, revoked: &dyn Fn()) -> Result<()> {
        while let Some((message, handle)) = self.end.recv::<ToAgent>()? {
            match message {
                ToAgent::Grant { id, slot, seq, bytes, adopted, last } => {
                    let imported = handle.ok_or_else(|| Error::Invalid("a grant with no handle".into())).and_then(|h| self.backend.import(h, bytes));
                    let mut p = self.pending.lock();
                    match imported {
                        Ok(phys) => {
                            let lease = Lease { slot, seq, phys, adopted };
                            let answer = p.asks.remove(&id);
                            let mut got = match answer {
                                Some(Answer::Granting(got)) => got,
                                _ => vec![],
                            };
                            got.push(lease);
                            p.asks.insert(id, if last { Answer::Done(got) } else { Answer::Granting(got) });
                        }
                        Err(error) => {
                            drop(p);
                            self.table.answer(slot, seq, false)?;
                            self.end.send(&ToManager::Answered, None)?;
                            self.pending.lock().asks.insert(id, Answer::Refused(error.to_string()));
                        }
                    }
                    self.answered.notify_all();
                }
                ToAgent::Refuse { id, why } => {
                    self.pending.lock().asks.insert(id, Answer::Refused(why));
                    self.answered.notify_all();
                }
                ToAgent::Revoke => revoked(),
                ToAgent::Welcome { .. } => {}
            }
        }
        Ok(())
    }

    /// `bytes` of weights for `region`, imported (the caller maps each and answers it). Waits for
    /// the manager's answer, which comes once room is made; refused, the request ends Exhausted.
    pub fn take(&self, bytes: u64, region: Option<Region>) -> Result<Vec<Lease>> {
        let id = {
            let mut p = self.pending.lock();
            p.next += 1;
            p.next
        };
        self.end.send(&ToManager::Ask { id, bytes, region }, None)?;
        let mut p = self.pending.lock();
        loop {
            match p.asks.remove(&id) {
                Some(Answer::Done(got)) => return Ok(got),
                Some(Answer::Refused(why)) => {
                    return Err(Error::Exhausted(Box::new(Exhausted { need: bytes, site: Some(format!("lease: {why}")), stats: crate::memory::Stats::default(), foreign: None })));
                }
                Some(granting) => {
                    p.asks.insert(id, granting);
                }
                None => {}
            }
            if p.closed {
                return Err(Error::Closed("the lease manager".into()));
            }
            self.answered.wait(&mut p);
        }
    }

    /// The holder mapped these: they are its to pin.
    pub fn mapped(&self, leases: &[Lease]) -> Result<()> {
        for lease in leases {
            self.table.answer(lease.slot, lease.seq, true)?;
        }
        self.end.send(&ToManager::Answered, None)
    }

    pub fn landed(&self, lease: &Lease) -> Result<()> {
        self.table.set_landed(lease.slot)
    }

    pub fn pin(&self, lease: &Lease) -> Result<bool> {
        self.table.pin(lease.slot)
    }

    pub fn unpin(&self, lease: &Lease) -> Result<()> {
        self.table.unpin(lease.slot)
    }

    pub fn hint(&self, lease: &Lease, class: u32, next_use: u64) -> Result<()> {
        self.table.hint(lease.slot, class, next_use)
    }

    /// Whether the manager asked for this lease back.
    pub fn asked(&self, lease: &Lease) -> bool {
        self.table.ask(lease.slot) == lease.seq
    }

    /// Commits a give-back (asked or not): false while pinned. The holder then unmaps and calls
    /// [`Agent::release`].
    pub fn commit(&self, lease: &Lease) -> Result<bool> {
        self.table.commit(lease.slot, lease.seq)
    }

    /// The committed lease is unmapped: its import goes, and the manager frees the chunk.
    pub fn release(&self, lease: Lease) -> Result<()> {
        let mapped = self.table.word(lease.slot).is_some_and(|w| w.state == State::Revoking && w.seq == lease.seq);
        if !mapped {
            return Err(Error::Invalid(format!("lease slot {} released before its commit", lease.slot)));
        }
        self.backend.destroy(lease.phys)?;
        self.table.released(lease.slot, lease.seq)?;
        self.end.send(&ToManager::Answered, None)
    }

    pub fn table(&self) -> &Table {
        &self.table
    }
}

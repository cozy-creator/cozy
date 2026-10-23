//! The lease socket's messages (varena-leases §11): handles and doorbells only; every state the
//! two sides share is in the table. One stream per executor and GPU; each side serializes its own
//! sends.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::os::{self, Owned, Stream};
use crate::Result;

/// A weights region a grant fills: its set's fingerprint and region index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Region {
    pub fingerprint: u64,
    pub region: u32,
}

/// Why the manager refused: an agent redials, runs standalone or fails the request by it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Refusal {
    /// The peer is no executor TensorD registered (yet: a follower may redial once it is).
    NotRegistered,
    /// No compatible manager (another protocol, or a re-attach it does not serve): standalone.
    Incompatible,
    /// No room can come for the ask.
    NothingRevocable,
    /// The manager failed serving the ask (its backend would not open, a chunk would not export).
    Failed,
    /// Nobody serves this GPU's leases (no listener): never sent, an agent's own finding.
    Absent,
}

/// What an ask is for: a weights page is one chunk sized to it (§3, never split, retainable or
/// not); torch and cache bytes split at the largest chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Weights,
    Torch,
    Cache,
}

/// A region as the manager's policy sees it (§5, §6): sent at its create and its changes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionFacts {
    pub id: u64,
    pub kind: crate::region::Reclaimable,
    /// The chunks it holds now: a cache region's arena chunks; a graph's weight pages and pool chunks.
    pub slots: Vec<u32>,
    /// Its measured production (or capture) time.
    pub cost_ns: u64,
    /// Its last use, CLOCK_MONOTONIC ns: one clock for every process on the machine.
    pub last_use: u64,
    /// Advice: when it is used next, ordered within the executor as its chunks' advice is
    /// (`u64::MAX`: no use declared).
    pub next_use: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ToManager {
    /// Attach. The peer's credentials name the process; `birth` (its start ticks) pins that
    /// birth. A re-attach after a manager restart sends its table's fd back with it.
    Hello { protocol: u32, birth: u64 },
    /// `bytes` of weights, for `region` when given (a retained region answers with no copy).
    Ask { id: u64, bytes: u64, kind: Kind, region: Option<Region> },
    /// Doorbell: slots moved to Mapped, Refused or Released.
    Answered,
    Region(RegionFacts),
    RegionGone { id: u64 },
    /// What it measured: a request's peaks at its end, a set's bytes at its load.
    Measured(Measured),
}

/// What an agent measured.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Measured {
    /// A request's peaks at its end (keyed by its dispatch): its working set, its bytes beside its
    /// weights, its anonymous host high-water.
    Request { request: String, working: u64, compute: u64, host: u64 },
    /// A set's resident bytes, at its load.
    Set { name: String, bytes: u64 },
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ToAgent {
    /// The table's fd attached (a memfd), on a first attach.
    Welcome { epoch: u64 },
    /// One chunk of ask `id`, its handle attached; `last` closes the ask. `adopted`: its bytes are
    /// the region's already (retained), no fill.
    Grant { id: u64, slot: u32, seq: u32, bytes: u64, adopted: bool, last: bool },
    Refuse { id: u64, code: Refusal, why: String },
    /// Doorbell: `ask` set on some slots.
    Revoke,
    /// Reclaim region `region` whole: a graph drops; a cache region swaps (`swap`) or is discarded.
    Reclaim { region: u64, swap: bool },
}

/// One side's end: sends serialized, receives on one thread.
pub struct End {
    stream: Stream,
    sending: Mutex<()>,
}

impl End {
    pub fn new(stream: Stream) -> End {
        End { stream, sending: Mutex::new(()) }
    }

    pub fn send<T: Serialize>(&self, m: &T, handle: Option<Owned>) -> Result<()> {
        let bytes = os::encode(m)?;
        let _one = self.sending.lock();
        os::send(&self.stream, &(bytes, handle))
    }

    /// The next message; `None` once the peer closed.
    pub fn recv<T: serde::de::DeserializeOwned>(&self) -> Result<Option<(T, Option<Owned>)>> {
        let Some((bytes, handle)) = os::recv(&self.stream)? else { return Ok(None) };
        Ok(Some((os::decode(&bytes)?, handle)))
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }
}

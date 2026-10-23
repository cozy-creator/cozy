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
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ToManager {
    /// Attach. The peer's credentials name the process; `birth` (its start ticks) pins that
    /// birth. A re-attach after a manager restart sends its table's fd back with it.
    Hello { protocol: u32, birth: u64 },
    /// `bytes` of weights, for `region` when given (a retained region answers with no copy).
    Ask { id: u64, bytes: u64, region: Option<Region> },
    /// Doorbell: slots moved to Mapped, Refused or Released.
    Answered,
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

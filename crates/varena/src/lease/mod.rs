//! Leases (varena-leases): the manager, hosted by TensorD, creates every GPU chunk and leases it
//! to each model process's agent over one shared table and one socket. Both roles live here so
//! they are tested together against one backend.

pub mod agent;
pub mod manager;
pub mod policy;
#[allow(unsafe_code)]
pub mod table;
pub mod wire;

pub use agent::{Agent, Attach, Deriving, Done, Lease, Ticket};
pub use policy::Advice;
pub use manager::{accepts, chunks, ExecutorHolding, Holdings, Hosting, Identity, Manager, Mode, Opener, Peak, Pressure, Record, CHUNK, MARKER, PROTOCOL};
pub use table::{Content, Figures, State, Table, Word};
pub use wire::{Kind, Measured, Refusal, Region};

use crate::os::{self, Listener, Stream};
use crate::{GpuId, Result};

/// A derived-weights disk entry to write (§6): its own id, which [`Agent::derived`] names back
/// (a second writer of one key gets its own: neither marks the other's written), and its file.
pub struct DerivedEntry {
    pub id: u64,
    pub file: std::fs::File,
}

fn service(uuid: &str) -> String {
    format!("lease-{uuid}")
}

/// The GPU's lease listener for its manager; `None` while another process serves it.
pub fn listen(gpu: &GpuId) -> Result<Option<Listener>> {
    os::listen(os::euid(), &service(&gpu.uuid))
}

/// An agent's connection to the manager of GPU `uuid` in namespace `ns` (the uid TensorD
/// serves as); `None` when nobody serves it: that GPU runs standalone.
pub fn connect(ns: u32, uuid: &str) -> Result<Option<Stream>> {
    os::connect(ns, &service(uuid))
}

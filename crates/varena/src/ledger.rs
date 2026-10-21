//! The GPU's shared table: one row per process, held by an OFD lock that dies with it; each
//! row's bytes by class, published from its own table's derivation; asks with a class ceiling
//! and their answers; measured peaks; TensorD's dispatches. Named per GPU and user, so a
//! standalone engine opens it without TensorD.

use std::os::fd::{AsFd, BorrowedFd, OwnedFd};

use crate::{Error, GpuId, Result};

/// What TensorD hands an executor for its [`crate::Memory`]: the table and TensorD's eventfd.
/// Fields are public so a spawner can pass them across exec.
pub struct Grant {
    pub table: OwnedFd,
    pub events: OwnedFd,
}

/// One row as [`Ledger::status`] shows it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RowStatus {
    pub pid: u32,
    pub live: bool,
    pub context: u64,
    /// Bytes by class (F unused, O, S, N, D, W, P).
    pub classes: [u64; 7],
    /// The largest remaining working set of each running request, summed.
    pub remaining: u64,
    /// Running requests with no measured working set.
    pub unmeasured: u32,
    /// Bytes asked of it and not yet answered.
    pub asked: u64,
}

/// The table at a glance (display only).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Status {
    pub capacity: u64,
    pub free: u64,
    /// Bytes on the device no row accounts for, measured.
    pub foreign: u64,
    pub rows: Vec<RowStatus>,
    /// Request id → `<package>|<shape>`.
    pub dispatched: Vec<(String, String)>,
    /// Rows whose ask thread did not run when rung.
    pub stalled: Vec<u32>,
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

/// TensorD's handle on one GPU's table.
pub struct Ledger {
    events: OwnedFd,
}

impl Ledger {
    /// The GPU's table, named per GPU and user; capacity = min(card, limit). TensorD runs the
    /// custody service when it holds the service lock (it does when it starts first).
    pub fn create(gpu: &GpuId, limit: Option<u64>) -> Result<Ledger> {
        let _ = (gpu, limit);
        Err(Error::Internal("not built yet".into()))
    }

    /// The descriptors executor `pid` opens its memory with.
    pub fn grant(&self, pid: u32) -> Result<Grant> {
        let _ = pid;
        Err(Error::Internal("not built yet".into()))
    }

    /// After `waitpid`: `pid`'s row goes, and the pages it kept become orphans.
    pub fn reap(&self, pid: u32) {
        let _ = pid;
    }

    /// Whether a request of `package` at `shape` can start now beside everything running.
    pub fn fits(&self, package: &str, shape: &str) -> bool {
        let _ = (package, shape);
        false
    }

    /// A multi-GPU group: dispatches `request` on every GPU of `group` if it fits on all, else
    /// on none.
    pub fn admit(group: &[&Ledger], request: &str, package: &str, shape: &str) -> Result<bool> {
        let _ = (group, request, package, shape);
        Err(Error::Internal("not built yet".into()))
    }

    /// Names the `<package>|<shape>` a request's peak is recorded under, before it runs.
    pub fn dispatch(&self, request: &str, package: &str, shape: &str) -> Result<()> {
        let _ = (request, package, shape);
        Err(Error::Internal("not built yet".into()))
    }

    pub fn undispatch(&self, request: &str) {
        let _ = request;
    }

    /// The measured peaks, to persist per (GPU model, package release).
    pub fn peaks(&self) -> Vec<Peak> {
        vec![]
    }

    /// Peaks persisted earlier on this machine, for this GPU model.
    pub fn restore(&self, peaks: &[Peak]) -> Result<()> {
        let _ = peaks;
        Err(Error::Internal("not built yet".into()))
    }

    pub fn status(&self) -> Status {
        Status::default()
    }

    /// Readable when room frees, a row dies or a peak is recorded; reading it clears it.
    pub fn events(&self) -> BorrowedFd<'_> {
        self.events.as_fd()
    }

    /// A GPU fault or sticky error: the GPU's kept pages go.
    pub fn fault(&self) {}
}

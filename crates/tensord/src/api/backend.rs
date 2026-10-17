//! One adapter into the owned execution engine. This module owns no journal.
use super::{auth::VerifiedActor, domain, v1};
use std::collections::BTreeMap;
use cozy_proto::error::Error;

pub trait MachineBackend: Send + Sync + 'static {
    /// `cozy.machine.v1` Run sources and Write (`runs`).
    fn runs(&self) -> Option<std::sync::Arc<crate::runs::Runs>> {
        None
    }
    fn list_packages(
        &self,
        _: VerifiedActor,
        _: domain::PackageListQuery,
    ) -> Result<domain::PackageList, Error> {
        unsupported()
    }
    /// What each of the caller's installations holds now (`Environment.level`), by
    /// installation id.
    fn levels(&self, _: VerifiedActor) -> Result<BTreeMap<String, &'static str>, Error> {
        Ok(BTreeMap::new())
    }
    /// The caller's warm set, each member with what it holds now (`StatusFrame.warm`).
    fn warm_set(&self, _: VerifiedActor) -> Result<Vec<v1::WarmItem>, Error> {
        Ok(vec![])
    }
    fn workspace(
        &self,
        _: VerifiedActor,
        _: domain::MachineExecutionWorkspaceQuery,
    ) -> Result<domain::MachineExecutionWorkspace, Error>;
    fn get(
        &self,
        _: VerifiedActor,
        _: domain::MachineExecutionQuery,
    ) -> Result<domain::MachineExecutionState, Error> {
        unsupported()
    }
    fn events(
        &self,
        _: VerifiedActor,
        _: domain::MachineExecutionEventsQuery,
    ) -> Result<domain::MachineExecutionEventPage, Error> {
        unsupported()
    }
    /// What a run's executor measured (canonical JSON), when it ran on a device.
    fn measurements(
        &self,
        _: VerifiedActor,
        _: domain::MachineExecutionQuery,
    ) -> Result<Option<Vec<u8>>, Error> {
        Ok(None)
    }
    /// A run's own time running its callable, every attempt summed (0: unknown).
    fn execution_ms(&self, _: VerifiedActor, _: domain::MachineExecutionQuery) -> Result<u64, Error> {
        Ok(0)
    }
    /// How the machine memo met a run (`call_memo`).
    fn memo(&self, _: VerifiedActor, _: domain::MachineExecutionQuery) -> Result<crate::journal::CallMemo, Error> {
        Ok(Default::default())
    }
    fn control(
        &self,
        _: VerifiedActor,
        _: domain::MachineExecutionControl,
    ) -> Result<domain::MachineExecutionState, Error> {
        unsupported()
    }
    fn list(
        &self,
        _: VerifiedActor,
        _: domain::MachineExecutionListQuery,
    ) -> Result<domain::MachineExecutionList, Error> {
        unsupported()
    }
    /// One run output's current bytes (`GET /v1/runs/{run}/outputs/{output}[/{index}]`).
    /// `index` is a list item's 1-based index. The caller has verified `actor`'s capability;
    /// a run another actor submitted is absent.
    fn open_output(
        &self,
        _: VerifiedActor,
        _run: u64,
        _output: &str,
        _index: Option<u32>,
    ) -> Result<OutputSnapshot, Error> {
        unsupported()
    }
    /// One file of a tree output (`member`, its manifest path), at the output's revision.
    fn open_member(
        &self,
        _: VerifiedActor,
        _run: u64,
        _output: &str,
        _index: Option<u32>,
        _member: &str,
    ) -> Result<OutputSnapshot, Error> {
        unsupported()
    }
    /// One kept log's bytes, oldest first (wire 72). A log not written yet is empty.
    /// A failed attempt's retained triage bundle.
    fn read_triage(
        &self,
        _: VerifiedActor,
        _: domain::MachineExecutionTriageQuery,
    ) -> Result<domain::MachineExecutionTriage, Error> {
        unsupported()
    }
    fn read_machine_log(
        &self,
        _: VerifiedActor,
        _: domain::MachineLogQuery,
    ) -> Result<Vec<u8>, Error> {
        unsupported()
    }
    /// A complete content-addressed object this signer wrote with `Write`, verified against
    /// `sha256` (hex).
    fn object(&self, _: VerifiedActor, _sha256: &str) -> Result<std::fs::File, Error> {
        unsupported()
    }
}
/// One consistent view of an output's current bytes: its parts in order.
pub struct OutputSnapshot {
    pub parts: Vec<(std::fs::File, u64)>,
    pub length: u64,
    /// The 1-based ordinal of this (output, index)'s products in the run's log.
    pub rev: u64,
    pub media_type: String,
    /// `sha256:<hex>`, set once the run is terminal (the output is final).
    pub sha256: Option<String>,
}
/// One bounded native-input transfer. The store/journal owner validates identity,
/// membership, contiguous offsets, native custody and durable commit/abort semantics.
/// Dropping a disconnected transfer never supplies execution cancellation authority.
pub trait InputTreeReceiver: Send {
    fn blob(&mut self, blob: domain::InputTreeImportBlob) -> Result<(), Error>;
    fn commit(
        self: Box<Self>,
        commit: domain::InputTreeImportCommit,
    ) -> Result<domain::NativeByteRetentionResult, Error>;
}

fn unsupported<T>() -> Result<T, Error> {
    Err(Error::not_found(
        "capability_unavailable",
        "this machine backend does not implement this operation",
    ))
}

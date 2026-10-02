//! The ONE transport (tfs-048/tfs-049). TensorFS knows manifests, objects, digests and
//! the store; before this module it could not fetch anything, so every consumer
//! reimplemented transport — cozy-runtime in Python, cozy-creator and tensorhub in Go.
//! One job, three implementations, two languages. This module ends that: closure lookup,
//! presign, credential presentation, the concurrent retrying download, and the upload
//! half — one implementation, reachable from Rust, Python (`tensorfs-py`) and Go (the
//! `tfs` binary), and every byte still enters through `Store::put_stream`.
//!
//! What this module owns is the SOCKET and nothing about identity: `FetchPlan::of` still
//! answers residency, `put_stream` is still the only admission door, and the digest is
//! still the only authority. A transport that answered any of those itself would be a
//! second opinion about identity, and the two would disagree exactly once, silently, on
//! the artifact that mattered.
//!
//! Three behaviours are PORTED from cozy-runtime's `pull.py`/`egress.py` with their
//! evidence, not reinvented (they were measured, and paid for on a rented pod):
//!
//! - **per-object resume across pulls** (`pull`): the next pull re-plans against the store
//!   and asks only for the objects still missing;
//! - **the liveness ledger, never a timeout** (`ledger`): a deadline arrives from the
//!   caller or there is none; a stream is judged only against the longest gap this pull
//!   has actually shown;
//! - **the chunked download** (`download`): the wanted objects as fixed ranged chunks on
//!   one queue, and a request that fails, goes silent or runs far slower than its peers
//!   asked again on a fresh connection. A Hub pull and one object from a source URL
//!   (`fetch_ranged`) are the same downloader.
//!
//! And the address predicate travels WITH the transport (`policy`), so there is still
//! exactly one downloader with the check — cr-012's fence, retargeted here by cr-090.
//!
//! `tensorfs-core` still builds and tests with no network and no hub: every test speaks
//! to a loopback listener it starts itself, and nothing here reads the environment.

mod api;
mod decisions;
mod download;
mod http;
mod hub;
mod ledger;
mod memory;
mod policy;
mod pull;
mod push;
pub(crate) mod sources;

#[cfg(test)]
mod tests;

pub use api::{api_get, fetch_prefix, probe_length, Prefix};
pub use download::{fetch_ranged, Ranged, CHUNK_BYTES, PART_BYTES};
pub use http::{trust_roots, Client};
pub use hub::{
    closure, closure_all, credential_from_spec, presign, Anonymous, Closure, CredentialProvider,
    HostToken, Presigned, ScopedHeaders,
};
pub use ledger::{
    Deadline, Ledger, PullCancellation, FETCH_ATTEMPTS, PULL_STREAMS, SAMPLE_SECONDS, STALL_FACTOR,
    STILL_SAMPLES, STREAMS,
};
pub use memory::{memory_available, streams_within, transfer_streams, STREAM_MEMORY};
pub use policy::{base_host, blocked, is_loopback_origin, resolve, CheckedUrl, SourcePolicy};
pub use pull::{
    fetch_wanted, pull, resolve_closure, FetchObserver, Fetched, ObjectSource, ObjectUrls,
    PullReport, PullRequest, WalkOutcome, RETRYABLE_STATUS,
};
pub use push::{push_object, Pushed, UploadGrant, MAX_GRANT_BYTES, PUT_ATTEMPTS};

pub use sources::{SourceDownload, SourceProgress};

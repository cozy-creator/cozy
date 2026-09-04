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
//! - **per-object resume** (`pull`): the recovery unit is an object; no byte-range resume
//!   inside one — the artifact is already cut on the object grid;
//! - **the liveness ledger, never a timeout** (`ledger`): a deadline arrives from the
//!   caller or there is none; a stream is judged only against the longest gap this pull
//!   has actually shown;
//! - **the N-way walk** (`pull::fetch_wanted`): 49.8 MB/s serial vs 117.6 MB/s at 16
//!   streams on 92.42 GiB, same pods, same presigned GETs, 141 MB/s ceiling.
//!
//! And the address predicate travels WITH the transport (`policy`), so there is still
//! exactly one downloader with the check — cr-012's fence, retargeted here by cr-090.
//!
//! `tensorfs-core` still builds and tests with no network and no hub: every test speaks
//! to a loopback listener it starts itself, and nothing here reads the environment.

mod api;
mod http;
mod hub;
mod ledger;
mod policy;
mod pull;
mod push;

#[cfg(test)]
mod tests;

pub use api::{api_get, probe_length};
pub use http::Client;
pub use hub::{
    closure, closure_all, credential_from_spec, presign, Anonymous, Closure, CredentialProvider,
    HostToken, Presigned, ScopedHeaders,
};
pub use ledger::{
    Deadline, Ledger, FETCH_ATTEMPTS, SAMPLE_SECONDS, STALL_FACTOR, STILL_SAMPLES, STREAMS,
};
pub use policy::{base_host, blocked, is_loopback_origin, resolve, CheckedUrl, SourcePolicy};
pub use pull::{
    fetch_authorized, fetch_url, fetch_wanted, pull, FetchObserver, Fetched, ObjectSource,
    ObjectUrls, PullReport, PullRequest, WalkOutcome, RETRYABLE_STATUS,
};
pub use push::{push_object, Pushed, UploadGrant, MAX_GRANT_BYTES, PUT_ATTEMPTS};

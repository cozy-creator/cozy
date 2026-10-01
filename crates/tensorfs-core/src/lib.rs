//! tensorfs-core — canonical documents, tensor rules, the fit function, and the encoding
//! registry (tfs-013). See tracker-v2/cozytensors.md and tensorfs.md.

// DENY, not forbid, for exactly one reason: `sha256.rs` reaches the CPU's SHA-256
// instructions through intrinsics, and `forbid` cannot be relaxed even module-wide
// (tfs-068). Two `#[allow(unsafe_code)]` sit in that file and nowhere else; anywhere
// else in this crate an `unsafe` is still a hard error, and `git grep -n unsafe` over
// crates/tensorfs-core/src is the check that it stays that way.
#![deny(unsafe_code)]

pub mod asset_update;
pub mod b64;
mod cache_roots;
pub mod canon;
pub mod capability;
pub mod catalog;
mod cbor;
pub mod checkpoint;
pub mod checkpoint_root;
pub mod corpus;
pub mod derived;
pub mod descriptors;
pub mod disk;
pub mod dtype;
pub mod durability;
pub mod ensure;
pub mod err;
pub mod fetch;
mod filesystem_identity;
pub mod fill;
pub mod fit;
pub mod gc;
pub mod header;
pub mod home;
pub mod ids;
pub mod ingest;
pub mod jcs;
pub mod keyed_roots;
pub mod limits;
pub mod machine;
pub mod manifest;
pub mod meta;
pub mod project;
pub mod providers;
pub mod read;
pub mod receipt;
pub mod reclaim;
pub mod registry;
pub mod relation;
pub mod repo_cache;
pub mod repository;
pub mod sha256;
pub mod source_artifact;
pub mod spec;
pub mod staging;
pub mod stats;
pub mod storage;
pub mod store;
pub mod transport;
pub mod unsynced;
pub mod vectors;

/// This TensorFS's release version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// What callers may feature-detect by name instead of by version: `ensure`, ranged `get`
/// under a lease, delivered-product marks, the pressure pass, and keyed cache roots.
pub const CAPABILITIES: &[&str] = &[
    "deliver/1",
    "ensure/1",
    "get-range/1",
    "keyed-roots/1",
    "pressure/1",
];

/// Conservative identity of the actual native converter build, not its version label.
pub const IMPLEMENTATION_SHA256: &str = env!("TENSORFS_IMPLEMENTATION_SHA256");

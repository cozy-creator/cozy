//! tensorfs-core — canonical documents, tensor rules, the fit function, and the encoding
//! registry (tfs-013). See tracker-v2/cozytensors.md and tensorfs.md.

// DENY, not forbid, for exactly one reason: `sha256.rs` reaches the CPU's SHA-256
// instructions through intrinsics, and `forbid` cannot be relaxed even module-wide
// (tfs-068). Two `#[allow(unsafe_code)]` sit in that file and nowhere else; anywhere
// else in this crate an `unsafe` is still a hard error, and `git grep -n unsafe` over
// crates/tensorfs-core/src is the check that it stays that way.
#![deny(unsafe_code)]

pub mod b64;
mod cache_roots;
pub mod canon;
pub mod capability;
pub mod catalog;
mod cbor;
pub mod checkpoint;
pub mod corpus;
pub mod derived;
pub mod dtype;
pub mod durability;
pub mod err;
pub mod fetch;
pub mod fill;
pub mod fit;
pub mod gc;
pub mod header;
pub mod home;
pub mod ids;
pub mod ingest;
pub mod jcs;
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
pub mod storage;
pub mod store;
pub mod transport;
pub mod vectors;

/// Conservative identity of the actual native converter build, not its version label.
pub const IMPLEMENTATION_SHA256: &str = env!("TENSORFS_IMPLEMENTATION_SHA256");

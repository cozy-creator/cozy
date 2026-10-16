//! tensorfs-core — canonical documents, tensor rules, the fit function, and the encoding
//! registry (tfs-013). See tracker-v2/cozytensors.md and tensorfs.md.

#![forbid(unsafe_code)]

pub mod b64;
#[cfg(feature = "store")]
mod cache_roots;
pub mod canon;
pub mod capability;
#[cfg(feature = "store")]
pub mod contract;
#[cfg(feature = "store")]
pub mod catalog;
mod cbor;
#[cfg(feature = "store")]
pub mod checkpoint;
#[cfg(feature = "store")]
pub mod checkpoint_root;
#[cfg(feature = "store")]
pub mod corpus;
#[cfg(feature = "store")]
pub mod derived;
#[cfg(feature = "store")]
pub mod descriptors;
#[cfg(feature = "store")]
pub mod disk;
pub mod dtype;
#[cfg(feature = "store")]
pub mod durability;
#[cfg(feature = "store")]
pub mod admission;
pub mod err;
#[cfg(feature = "store")]
pub mod fetch;
#[cfg(feature = "store")]
mod filesystem_identity;
#[cfg(feature = "store")]
pub mod fill;
pub mod fit;
#[cfg(feature = "store")]
pub mod gc;
pub mod header;
#[cfg(feature = "store")]
pub mod home;
pub mod ids;
#[cfg(feature = "store")]
pub mod ingest;
pub mod jcs;
pub mod limits;
pub mod machine;
pub mod manifest;
#[cfg(feature = "store")]
pub mod meta;
#[cfg(feature = "store")]
pub mod object_roots;
#[cfg(feature = "store")]
pub mod origin;
#[cfg(feature = "store")]
pub mod project;
#[cfg(feature = "store")]
pub mod providers;
pub mod read;
#[cfg(feature = "store")]
pub mod receipt;
#[cfg(feature = "store")]
pub mod reclaim;
pub mod registry;
pub mod relation;
#[cfg(feature = "store")]
pub mod repository;
pub mod sha256;
#[cfg(feature = "store")]
pub mod source_artifact;
#[cfg(feature = "store")]
pub mod source_model;
pub mod spec;
#[cfg(feature = "store")]
pub mod stats;
#[cfg(feature = "store")]
pub mod storage;
#[cfg(feature = "store")]
pub mod store;
#[cfg(feature = "store")]
pub mod transport;
#[cfg(feature = "store")]
pub mod unsynced;
pub mod vectors;

/// This TensorFS's release version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The on-disk Store this build reads and writes: layout, catalog and object encoding. The
/// `cozy` binary and the `tensorfs` wheel share every Store ([`contract`]); a change an older
/// build would misread bumps it, and the wheel's minor.
pub const STORE_FORMAT: u32 = 1;

/// The cross-process protocol over one Store: owner claim, writer guards, read leases and
/// holds. Bumped like [`STORE_FORMAT`] by a change a process speaking the old one would break.
pub const CLAIM_PROTOCOL: u32 = 1;

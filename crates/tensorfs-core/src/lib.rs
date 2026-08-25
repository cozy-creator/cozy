//! tensorfs-core — canonical documents, tensor rules, the structure function, and the
//! encoding registry (tfs-013). See tracker-v2/cozytensors.md and tensorfs.md.

#![forbid(unsafe_code)]

pub mod b64;
pub mod canon;
pub mod dtype;
pub mod err;
pub mod header;
pub mod ids;
pub mod ingest;
pub mod limits;
pub mod manifest;
pub mod registry;
pub mod relation;
pub mod sha256;
pub mod spec;
pub mod vectors;

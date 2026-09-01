//! tensorfs-core — canonical documents, tensor rules, the structure function, and the
//! encoding registry (tfs-013). See tracker-v2/cozytensors.md and tensorfs.md.

#![forbid(unsafe_code)]

pub mod b64;
pub mod canon;
pub mod capability;
pub mod catalog;
mod cbor;
pub mod checkpoint;
pub mod compat;
pub mod corpus;
pub mod derived;
pub mod dtype;
pub mod err;
pub mod fetch;
pub mod fill;
#[cfg(feature = "grade")]
pub mod grade;
pub mod header;
pub mod ids;
pub mod ingest;
pub mod limits;
pub mod machine;
pub mod manifest;
pub mod meta;
pub mod project;
pub mod read;
pub mod receipt;
pub mod registry;
pub mod relation;
pub mod repository;
pub mod sha256;
pub mod spec;
pub mod storage;
pub mod store;
pub mod structure;
pub mod vectors;
#[cfg(feature = "grade")]
pub mod verifier;

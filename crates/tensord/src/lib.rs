// The machine API's error envelope is TensorD's error type; refusals are cold paths, so its size
// costs nothing worth boxing it for.
#![allow(clippy::result_large_err)]
pub mod access;
mod archive;
pub mod api;
mod boundary_json;
mod call_memo;
pub mod catalog;
pub mod config;
pub mod derive;
pub mod child_launcher;
pub mod device_executor;
pub mod execution;
pub mod frame;
mod gpu_allocation;
pub mod gpu_service;
pub mod held_models;
#[cfg(target_os = "linux")]
pub mod host_tier;
/// Off where the OS has no sealed shared memory: weights are read from the page cache.
#[cfg(not(target_os = "linux"))]
#[path = "host_tier_off.rs"]
pub mod host_tier;
pub mod hub;
pub mod calls;
pub mod journal;
pub mod launch_identity;
pub mod local_source;
pub mod machine;
pub mod machine_api;
pub mod memo;
pub mod memory;
pub mod model_sources;
pub mod native_inputs;
pub mod objects;
pub mod origin;
pub mod output_capacity;
pub mod precision;
pub mod process;
pub mod products;
pub mod protocol;
pub mod published;
pub mod reclaim;
#[cfg(target_os = "linux")]
pub mod resident_custody;
/// Off where the OS cannot share an executor's device memory by handle.
#[cfg(not(target_os = "linux"))]
#[path = "resident_custody_off.rs"]
pub mod resident_custody;
pub mod runs;
pub mod weights;
#[cfg(test)]
mod scratch;
#[cfg(test)]
mod sdk_selection_tests;
pub mod serve;
pub mod service;
pub mod stop;
pub mod storage;
pub mod triage;
pub mod trees;

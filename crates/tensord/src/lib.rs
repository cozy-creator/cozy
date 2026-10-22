// The machine API's error envelope is TensorD's error type; refusals are cold paths, so its size
// costs nothing worth boxing it for.
#![allow(clippy::result_large_err)]
pub mod access;
pub mod adapter;
mod archive;
pub mod api;
mod boundary_json;
mod call_memo;
pub mod cas;
pub mod catalog;
pub mod config;
pub mod child_launcher;
pub mod device_executor;
pub mod execution;
mod gpu_allocation;
pub mod gpu_service;
pub mod held_models;
pub mod hub;
pub mod calls;
pub mod journal;
pub mod launch_identity;
pub mod local_source;
pub mod machine;
pub mod machine_api;
pub mod memo;
pub mod memory;
pub mod native_inputs;
pub mod objects;
pub mod origin;
pub mod output_capacity;
pub mod precision;
pub mod process;
pub mod products;
pub mod seam;
pub mod published;
pub mod reclaim;
pub mod runs;
#[cfg(test)]
mod scratch;
#[cfg(test)]
mod sdk_selection_tests;
pub mod serve;
mod queue;
pub mod service;
pub mod stop;
pub mod storage;
pub mod triage;
pub mod trees;

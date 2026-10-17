//! The wire types every cozy process shares.

pub mod calcifer;
pub mod capability;
pub mod config;
pub mod error;
pub mod ids;

/// `cozy.machine.v1`: Calcifer ↔ machine. Unchanged on the wire at the cutover from Go.
pub mod machine_json;

pub mod machine {
    tonic::include_proto!("cozy.machine.v1");
}

pub mod auth;
pub mod backend;
pub mod domain;
mod cozy1;
pub mod identity;
pub mod install;
mod machine_status;
mod machine_update;
mod machine_v1;
pub mod retired;
mod server;
pub mod workspaces;

/// `cozy.machine.v1` and Cozy-Cap live in `cozy-proto`, shared with Calcifer and the CLI.
pub use cozy_proto::{capability, machine as v1};

pub use backend::MachineBackend;
pub use machine_status::CAPABILITIES;
pub use server::{serve, MachineIdentity};

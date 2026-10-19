pub mod auth;
pub mod backend;
pub mod domain;
mod cozy1;
mod http;
pub mod identity;
pub mod install;
pub(crate) mod machine_runs;
mod machine_status;
mod machine_update;
mod server;
pub mod workspaces;

/// The machine API's types and Cozy-Cap live in `cozy-proto`, shared with Calcifer and the CLI.
pub use cozy_proto::{capability, machine as v1};

pub use backend::MachineBackend;
pub(crate) use http::refused;
pub use machine_status::CAPABILITIES;
pub use server::{serve, MachineIdentity};

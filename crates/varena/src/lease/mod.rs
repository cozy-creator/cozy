//! Leases (varena-leases): the manager, hosted by TensorD, creates every GPU chunk and leases it
//! to each model process's agent over one shared table and one socket. Both roles live here so
//! they are tested together against one backend.

pub mod agent;
pub mod manager;
pub mod table;
pub mod wire;

pub use agent::{Agent, Lease};
pub use manager::{chunks, Holdings, Hosting, Identity, Manager, CHUNK, PROTOCOL};
pub use table::{Content, State, Table, Word};
pub use wire::Region;

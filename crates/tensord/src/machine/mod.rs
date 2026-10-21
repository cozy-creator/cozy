//! The machine role on a pod or this computer: its launch grant, lifetime identity,
//! readiness receipt and rental lifecycle. Its contracts are the Hub's.
pub mod binary;
pub mod client;
pub mod endpoint;
pub mod grant;
pub mod hub;
pub mod identity;
pub mod lifecycle;
pub mod net;
pub mod player;
pub mod probe;
pub mod provider;
pub mod pypi;
pub mod receipt;
pub mod ssh;
pub mod supervise;
pub mod toolchain;
pub mod update;

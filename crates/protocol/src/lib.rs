//! The wire types every cozy process shares.

pub mod calcifer;
pub mod capability;
pub mod config;
pub mod error;
pub mod ids;
pub mod machine;
pub mod names;

/// The commit this build is from (`build.rs`).
pub const COMMIT: &str = env!("COZY_COMMIT");

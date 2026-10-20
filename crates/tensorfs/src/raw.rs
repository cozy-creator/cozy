//! Advanced interop shared by language bindings and placement integrations.
//! Ordinary Rust clients drive async work on their own runtime.

use std::sync::Arc;

use crate::{Failure, Result};

/// The existing process-owned driver for synchronous language bindings.
/// Merely importing a binding does not call this or start native workers.
pub fn runtime() -> Result<Arc<tokio::runtime::Runtime>> {
    crate::fetch::runtime().map_err(|error| {
        let detail = error.detail.clone();
        Failure { detail, ..Failure::from(error) }
    })
}

//! Advanced interop shared by language bindings and placement integrations.
//! Ordinary Rust clients drive async work on their own runtime.

use std::sync::Arc;

use crate::{Failure, Result};

pub use crate::session::PartData;

/// Group tensor metadata for placement without opening or fetching payload files.
pub fn regions(
    header: &crate::format::Header,
    max_bytes: u64,
    boundaries: &[String],
    only: &[String],
) -> Vec<Vec<String>> {
    crate::client::regions(header, max_bytes, boundaries, only)
}

/// Fetch exactly these objects after checking that they belong to the checkpoint.
pub async fn fetch_objects(
    checkpoint: &crate::Checkpoint,
    objects: &[crate::format::ObjectRef],
    want: &crate::Fetch<'_>,
) -> Result<crate::Fetched> {
    checkpoint.fetch_objects_exact(objects, want).await
}

/// Fill caller-owned host or device destinations through the existing byte engine.
///
/// # Safety
/// Every destination must be valid for the complete transfer and remain owned until
/// this call completes, including its error path. Device pointers and streams must
/// satisfy the same requirements as [`crate::io::read`].
#[allow(unsafe_code)]
pub unsafe fn read_into(
    checkpoint: &crate::Checkpoint,
    targets: &[crate::io::Target<'_>],
    options: &crate::io::ReadOpts,
) -> Result<crate::io::ReadStats> {
    // SAFETY: the caller supplies and retains every target under the contract above.
    unsafe { checkpoint.read_into_raw(targets, options) }
}

/// The existing process-owned driver for synchronous language bindings.
/// Merely importing a binding does not call this or start native workers.
pub fn runtime() -> Result<Arc<tokio::runtime::Runtime>> {
    crate::fetch::runtime().map_err(|error| {
        let detail = error.detail.clone();
        Failure { detail, ..Failure::from(error) }
    })
}

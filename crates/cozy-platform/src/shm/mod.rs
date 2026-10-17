//! Sealed shared memory for weight layouts (TensorD `os.rs`): a Linux memfd with file seals. macOS
//! has neither memfd nor seals; the Apple tier (AP) designs its own, and until then every call
//! answers `Unsupported`.
#[cfg_attr(target_os = "linux", path = "linux.rs")]
#[cfg_attr(not(target_os = "linux"), path = "unsupported.rs")]
mod imp;

pub use imp::*;

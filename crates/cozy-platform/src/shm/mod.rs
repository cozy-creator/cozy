//! Sealed shared memory for weight layouts (TensorD `os.rs`): a Linux memfd with file seals. macOS
//! has neither memfd nor seals; the Apple tier (AP) designs its own, and until then every call
//! answers `Unsupported`.
#[cfg_attr(target_os = "linux", path = "linux.rs")]
#[cfg_attr(not(target_os = "linux"), path = "unsupported.rs")]
mod imp;

pub use imp::*;

const SEAL_SEAL: i32 = 0x1;
const SEAL_SHRINK: i32 = 0x2;
const SEAL_GROW: i32 = 0x4;
const SEAL_WRITE: i32 = 0x8;
const SEAL_FUTURE_WRITE: i32 = 0x10;

/// What makes a memfd immutable to everyone else: no write, resize, punch or unsealing.
/// Future-write (Linux 5.1) refuses new writable mappings and writes; the filler's own mapping
/// may finish (TensorFS's host tier).
pub const FULL_SEALS: i32 = SEAL_SEAL | SEAL_SHRINK | SEAL_GROW | SEAL_FUTURE_WRITE;
/// What an adopter of a sealed memfd requires besides one of [`WRITE_SEALS`].
pub const RESIZE_SEALS: i32 = SEAL_SEAL | SEAL_SHRINK | SEAL_GROW;
pub const WRITE_SEALS: i32 = SEAL_WRITE | SEAL_FUTURE_WRITE;
/// No write at all, and no writable mapping may exist when it is sealed: the set TensorD seals
/// with and requires of a plan it receives.
pub const NO_WRITE_SEALS: i32 = RESIZE_SEALS | SEAL_WRITE;

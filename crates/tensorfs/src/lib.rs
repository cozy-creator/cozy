//! TensorFS: the cozytensors format, its store, fetch and convert, and the read engine.

// Unsafe code lives only in platform, the OS seam.
#![deny(unsafe_code)]

pub mod b64;
#[cfg(feature = "store")]
pub mod descriptors;
pub mod err;
#[cfg(feature = "store")]
pub mod export;
pub mod failure;
pub use failure::{Failure, Kind, Result};
#[cfg(feature = "store")]
pub mod fetch;
pub mod format;
#[cfg(feature = "store")]
pub mod home;
#[cfg(feature = "store")]
pub mod import;
/// The read engine: byte ranges of files into host or device destinations (xs-055).
pub mod io;
pub mod kernels;
/// The A11 0.12 -> D10 header migration.
#[cfg(feature = "store")]
pub mod migrate;
pub mod sha256;
#[cfg(feature = "store")]
pub mod store;
#[cfg(feature = "store")]
pub mod writer;
#[cfg(feature = "store")]
mod verification;
#[cfg(feature = "store")]
pub use verification::VerificationReport;
/// Advanced engine and synchronous-language integration.
#[cfg(feature = "store")]
pub mod raw;

/// The public session API; networking uses the caller's async runtime.
#[cfg(feature = "store")]
mod session;
#[cfg(feature = "store")]
pub use session::{
    bearer, Checkpoint, Client, Component, ComponentWriter, Config, Convert, ConvertProgress,
    Download, Endpoint, Fetched, Format, HeaderMap, Open, Progress, Read, ReadInto, Ref, Request,
    Selection, TokenProvider, Writer, WriterOptions,
};

/// This TensorFS's release version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The on-disk store this build reads and writes, shared by every process on a machine. A
/// change an older build would misread bumps it (`tensorfs.store/3`: trees and D10 headers).
pub const STORE_FORMAT: u32 = 3;

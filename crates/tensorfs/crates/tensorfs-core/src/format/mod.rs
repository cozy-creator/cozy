//! cozytensors/2 and the TensorFS v2 documents: types, strict reads, deterministic writes.
//! Normative spec: `crates/tensorfs/spec/cozytensors-2.md`. Depends on serde, serde_json and
//! sha2 only, so it builds without the `store` feature.

mod cbor;
mod decode;
mod doc;
mod dtype;
mod json;
pub mod layout;
mod value;

pub use decode::{decode, PartBytes, SCHEMES};
pub use doc::{
    repo_key, Body, Component, Encoding, Header, Location, Manifest, Metadata, ObjectRef, Part,
    Release, Repo, Span, Tensor,
};
pub use dtype::Dtype;
pub use value::Value;

pub const HEADER_FORMAT: &str = "cozytensors/2";
pub const MANIFEST_FORMAT: &str = "tensorfs.manifest/2";
pub const REPO_FORMAT: &str = "tensorfs.repo/2";
/// The manifest path of a checkpoint's header.
pub const HEADER_PATH: &str = "model.cozytensors";

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// The bytes are not a valid document of the expected kind.
    #[error("invalid document: {0}")]
    Invalid(String),
    /// The `format` tag names another document kind or major version.
    #[error("format {found:?} is not {expected} (any minor)")]
    Format {
        found: String,
        expected: &'static str,
    },
    /// The tensor reads, but this reader has no decoder for its scheme.
    #[error("unknown scheme {0:?}: its parts read raw, decode is refused")]
    UnknownScheme(String),
    /// A known scheme whose params, parts or logical dtype break its definition.
    #[error("cannot decode: {0}")]
    Decode(String),
}

impl Error {
    pub fn code(&self) -> &'static str {
        match self {
            Error::Invalid(_) => "invalid",
            Error::Format { .. } => "format",
            Error::UnknownScheme(_) => "unknown_scheme",
            Error::Decode(_) => "decode",
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn invalid<T>(detail: impl Into<String>) -> Result<T> {
    Err(Error::Invalid(detail.into()))
}

/// `<name>/<major>` or `<name>/<major>.<minor>`: a reader takes any minor of its major.
pub(crate) fn check_format(found: &str, expected: &'static str) -> Result<()> {
    let ok = found.strip_prefix(expected).is_some_and(|rest| {
        rest.is_empty()
            || rest
                .strip_prefix('.')
                .is_some_and(|m| !m.is_empty() && m.bytes().all(|c| c.is_ascii_digit()))
    });
    if ok {
        Ok(())
    } else {
        Err(Error::Format {
            found: found.into(),
            expected,
        })
    }
}

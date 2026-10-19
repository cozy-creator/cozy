//! cozytensors/1 and the TensorFS documents: types, strict reads, deterministic writes.
//! Normative spec: `crates/tensorfs/spec/cozytensors.md`. Depends on serde, serde_json and
//! sha2 only, so it builds without the `store` feature.

mod cbor;
pub mod decode;
mod doc;
mod dtype;
pub mod encode;
mod json;
pub mod layout;
mod value;

pub use decode::{check, decode, grid::ScaleLayout, PartBytes, SCHEMES};
pub use doc::{
    repo_key, Body, Component, Encoding, Header, Location, Manifest, Metadata, ObjectRef, Part,
    Release, Repo, Span, Tensor,
};
pub use dtype::Dtype;
pub use encode::{GlobalScale, NibbleOrder};
pub use value::Value;

pub const HEADER_FORMAT: &str = "cozytensors/1";
pub const MANIFEST_FORMAT: &str = "tensorfs.manifest/1";
pub const REPO_FORMAT: &str = "tensorfs.repo/1";
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
    /// A header from before the cut to this format: same tag, older layout.
    #[error("pre-cut cozytensors header (old layout); re-import or migrate")]
    PreCut,
    /// The tensor reads, but this reader has no decoder for its scheme.
    #[error("unknown scheme {0:?}: its parts read raw, decode is refused")]
    UnknownScheme(String),
    /// A known scheme whose params, parts or logical dtype break its definition.
    #[error("cannot decode: {0}")]
    Decode(String),
    /// Values or arrays a scheme's encoder cannot write.
    #[error("cannot encode: {0}")]
    Encode(String),
}

impl Error {
    pub fn code(&self) -> &'static str {
        match self {
            Error::Invalid(_) => "invalid",
            Error::Format { .. } => "format",
            Error::PreCut => "pre_cut",
            Error::UnknownScheme(_) => "unknown_scheme",
            Error::Decode(_) => "decode",
            Error::Encode(_) => "encode",
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

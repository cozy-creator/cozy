//! The golden-vector dispatch (tfs-013): doc kind + bytes -> outcome.
//!
//! It lives in the library rather than in `tfs` because there are now TWO harnesses over
//! one corpus — the Rust CLI and the Python facade (tfs-007) — and a second dispatch table
//! would be a place for the two languages to disagree about which reader a vector names.
//! There is exactly one reader per document either way; this says which one.

use crate::err::{refuse, Code, Refusal, Result};
use crate::header::{Closure, Header};
use crate::ids::{Doc, Plain};
use crate::ingest::transaction::IngestSession;
use crate::ingest::{IngestProfile, IngestSubject, IngestVerificationReceipt};
use crate::manifest::Manifest;
use crate::registry;
use crate::spec::EncodingSpec;
use crate::vectors::EncodingVectors;

/// What a vector's bytes turned into. `id` is the universal identity law
/// (`sha256(stored_bytes)`); the others are present only for the documents that project
/// them, and their absence is itself part of the frozen verdict.
///
#[derive(Debug, Clone)]
pub struct DocOutcome {
    pub id: String,
    pub tensor_schema: Option<String>,
    pub manifest: Option<String>,
}

pub const DOC_KINDS: &[&str] = &[
    "header",
    "header-serving",
    "encoding",
    "vectors",
    "manifest",
    "profile",
    "subject",
    "session",
    "receipt",
    "stamp",
    "machine_capability",
];

pub fn run_doc(
    doc: &str,
    bytes: &[u8],
    closure: &Closure,
    platform: &[String],
) -> Result<DocOutcome> {
    let id = crate::ids::object_id(bytes);
    let (tensor_schema, manifest) = match doc {
        "header" => Header::parse(bytes).and_then(|h| {
            h.validate(closure)?;
            Ok((Some(h.tensor_schema_digest()), None))
        })?,
        // Serving admission is a SECOND gate over the same bytes: `header` says the document
        // is valid data, `header-serving` says this runtime may execute it.
        "header-serving" => Header::parse(bytes).and_then(|h| {
            h.admit_for_serving(closure, platform)?;
            Ok((Some(h.tensor_schema_digest()), None))
        })?,
        "encoding" => EncodingSpec::decode_nested(bytes).map(|_| (None, None))?,
        "vectors" => EncodingVectors::parse_fixture(bytes).and_then(|v| {
            let plain = registry::seeds()
                .into_iter()
                .find(|s| s.alias == "plain/1")
                .expect("the plain/1 seed is compiled in")
                .spec;
            v.check(&plain)?;
            Ok((None, None))
        })?,
        // A decode IS the path law now: `Manifest` cannot exist unvalidated, so
        // parsing is the whole check and there is no second validate() to forget.
        "manifest" => Manifest::parse(bytes).map(|m| (None, Some(m.manifest_id())))?,
        "profile" => IngestProfile::parse(bytes).and_then(|p| {
            p.validate(platform)?;
            Ok((None, None))
        })?,
        "subject" => IngestSubject::parse(bytes).map(|_| (None, None))?,
        "session" => IngestSession::parse(bytes).map(|_| (None, None))?,
        "receipt" => IngestVerificationReceipt::parse(bytes).map(|_| (None, None))?,
        "stamp" => crate::ingest::stamp::Stamp::parse(bytes).map(|_| (None, None))?,
        "machine_capability" => {
            crate::machine::check_surface(doc, bytes)?;
            (None, None)
        }
        other => {
            return refuse(
                Code::UNKNOWN_FORMAT,
                format!("{other:?} is not one of the corpus's document kinds: {DOC_KINDS:?}"),
            )
        }
    };
    Ok(DocOutcome {
        id,
        tensor_schema,
        manifest,
    })
}

/// The receipt's binding check over five documents that arrive as bytes — the cross-vector
/// assertion, reachable from either harness.
pub fn receipt_binds(
    receipt: &[u8],
    subject: &[u8],
    manifest: &[u8],
    header: &[u8],
    stamp: &[u8],
) -> std::result::Result<(), Refusal> {
    let r = IngestVerificationReceipt::parse(receipt)?;
    let s = IngestSubject::parse(subject)?;
    let m = Manifest::parse(manifest)?;
    let h = Header::parse(header)?;
    let t = crate::ingest::stamp::Stamp::parse(stamp)?;
    r.binds(&s, &m, &h, &t)
}

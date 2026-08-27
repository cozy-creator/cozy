//! The golden-vector dispatch (tfs-013): doc kind + bytes -> outcome.
//!
//! It lives in the library rather than in `tfs` because there are now TWO harnesses over
//! one corpus — the Rust CLI and the Python facade (tfs-007) — and a second dispatch table
//! would be a place for the two languages to disagree about which reader a vector names.
//! There is exactly one reader per document either way; this says which one.

use crate::capability::CapabilityRecords;
use crate::err::{refuse, Code, Refusal, Result};
use crate::header::{Closure, Header};
use crate::ids::Doc;
use crate::ingest::fingerprint::FingerprintRegistry;
use crate::ingest::transaction::IngestRoot;
use crate::ingest::{IngestProfile, IngestSubject, IngestVerificationReceipt};
use crate::manifest::SnapshotManifest;
use crate::meta::State;
use crate::receipt::Receipt;
use crate::registry;
use crate::spec::EncodingSpec;
use crate::structure::StructureFixture;
use crate::vectors::EncodingVectors;

/// What a vector's bytes turned into. `id` is the universal identity law
/// (`sha256(stored_bytes)`); the other two are present only for the documents that project
/// them, and their absence is itself part of the frozen verdict.
#[derive(Debug, Clone)]
pub struct DocOutcome {
    pub id: String,
    pub topology: Option<String>,
    pub checkpoint: Option<String>,
}

pub const DOC_KINDS: &[&str] = &[
    "header",
    "header-serving",
    "spec",
    "vectors",
    "structure",
    "capability",
    "manifest",
    "profile",
    "subject",
    "registry",
    "root",
    "receipt",
    "stamp",
    "meta",
    "operation_receipt",
    "code_topology",
    "compat_report",
    "conversion_registration",
];

pub fn run_doc(
    doc: &str,
    bytes: &[u8],
    closure: &Closure,
    platform: &[String],
) -> Result<DocOutcome> {
    let id = crate::ids::object_id(bytes);
    let (topology, checkpoint) = match doc {
        "header" => Header::parse(bytes).and_then(|h| {
            h.validate(closure)?;
            Ok((Some(h.topology_digest()), None))
        })?,
        // Serving admission is a SECOND gate over the same bytes: `header` says the document
        // is valid data, `header-serving` says this runtime may execute it.
        "header-serving" => Header::parse(bytes).and_then(|h| {
            h.admit_for_serving(closure, platform)?;
            Ok((Some(h.topology_digest()), None))
        })?,
        "spec" => EncodingSpec::parse(bytes).map(|_| (None, None))?,
        "vectors" => EncodingVectors::parse(bytes).and_then(|v| {
            let plain = registry::seeds()
                .into_iter()
                .find(|s| s.alias == "plain/1")
                .expect("the plain/1 seed is compiled in")
                .spec;
            v.check(&plain)?;
            Ok((None, None))
        })?,
        // A structure fixture's banked key is its topology_digest, so that is what the
        // corpus freezes: a fixture whose projection drifts announces itself here.
        "structure" => StructureFixture::parse(bytes).map(|f| (Some(f.topology_digest()), None))?,
        "capability" => CapabilityRecords::parse(bytes).map(|_| (None, None))?,
        // A decode IS the path law now: `SnapshotManifest` cannot exist unvalidated, so
        // parsing is the whole check and there is no second validate() to forget.
        "manifest" => SnapshotManifest::parse(bytes).map(|m| (None, m.checkpoint_id().ok()))?,
        "profile" => IngestProfile::parse(bytes).and_then(|p| {
            p.validate(platform)?;
            Ok((None, None))
        })?,
        "subject" => IngestSubject::parse(bytes).map(|_| (None, None))?,
        "registry" => FingerprintRegistry::parse(bytes).map(|_| (None, None))?,
        "root" => IngestRoot::parse(bytes).map(|_| (None, None))?,
        "receipt" => IngestVerificationReceipt::parse(bytes).map(|_| (None, None))?,
        "stamp" => crate::ingest::stamp::Stamp::parse(bytes).map(|_| (None, None))?,
        "meta" => State::parse(bytes).map(|_| (None, None))?,
        "operation_receipt" => Receipt::parse(bytes).map(|_| (None, None))?,
        // The layout plane's documents (tfs-009). A code topology projects to the SAME
        // frozen projection a header does, so its digest is reported here too.
        "code_topology" => {
            crate::compat::CodeTopology::parse(bytes).map(|c| (Some(c.topology_digest()), None))?
        }
        "compat_report" => {
            crate::compat::CompatibilityReport::parse(bytes).map(|_| (None, None))?
        }
        #[cfg(feature = "grade")]
        "conversion_registration" => {
            crate::grade::ConversionRegistration::parse(bytes).map(|_| (None, None))?
        }
        #[cfg(not(feature = "grade"))]
        "conversion_registration" => {
            return refuse(
                Code::UNKNOWN_FORMAT,
                "conversion_registration is a hub-side grading document; this build carries \
                 no grading vocabulary (the `grade` feature is off in the worker wheel)",
            )
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
        topology,
        checkpoint,
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
    let m = SnapshotManifest::parse(manifest)?;
    let h = Header::parse(header)?;
    let t = crate::ingest::stamp::Stamp::parse(stamp)?;
    r.binds(&s, &m, &h, &t)
}

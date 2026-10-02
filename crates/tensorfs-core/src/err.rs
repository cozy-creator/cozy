//! Typed refusals. A refusal names its code and the first disagreeing thing with both sides.

use std::fmt;

macro_rules! codes {
    ($($v:ident),* $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[allow(non_camel_case_types)]
        pub enum Code { $($v),* }
        impl Code {
            pub fn as_str(self) -> &'static str {
                match self { $(Code::$v => stringify!($v)),* }
            }
            /// Every code, in declaration order. The binding layer builds one Python
            /// exception class per entry from THIS list, so the two languages cannot
            /// drift: a code added here appears in Python without anyone editing Python.
            pub const ALL: &'static [Code] = &[$(Code::$v),*];
        }
    };
}

codes![
    // canonical-byte layer
    NONCANONICAL_ENCODING,
    DUPLICATE_KEY,
    UNKNOWN_FIELD,
    MISSING_FIELD,
    WRONG_TYPE,
    NON_INTEGER_NUMBER,
    NUMBER_RANGE,
    NON_ASCII_FIELD,
    MALFORMED_JSON,
    MALFORMED_CBOR,
    TRAILING_BYTES,
    DEPTH_CAP,
    SIZE_CAP,
    COUNT_CAP,
    KEY_GRAMMAR,
    // identity layer
    MALFORMED_DIGEST,
    UNKNOWN_FORMAT,
    OBJECT_ID_MISMATCH,
    LENGTH_MISMATCH,
    // arithmetic / geometry
    ARITH_OVERFLOW,
    DIVISION_REMAINDER,
    ZERO_ELEMENT,
    RANK_CAP,
    SHAPE_MISMATCH,
    DTYPE_MISMATCH,
    DTYPE_UNKNOWN,
    BYTE_LENGTH_MISMATCH,
    // header
    EMPTY_COMPONENTS,
    MISSING_TENSOR,
    ROLE_SET_MISMATCH,
    UNCITED_ENCODING,
    UNKNOWN_ENCODING,
    ENCODING_MISMATCH,
    INLINE_EXCLUSIVE,
    INLINE_THRESHOLD,
    BAD_BASE64,
    SORT_ORDER,
    TOTAL_REFS_CAP,
    TOTAL_BYTES_CAP,
    UNKNOWN_TO_LOCAL_CAPABILITY,
    // encoding spec
    OPTIONAL_ROLE_FORBIDDEN,
    UNKNOWN_RELATION,
    VECTORS_REQUIRED,
    // manifest
    PATH_ILLEGAL,
    PATH_ORDER,
    PATH_PARENT_MISSING,
    PATH_CASE_COLLISION,
    PATH_RESERVED,
    PATH_DIGEST_MISMATCH,
    ATTACHMENT_CARDINALITY,
    WHOLE_DIGEST_MISMATCH,
    NOT_CONTAINED,
    // registry / qualification
    CAPABILITY_UNQUALIFIED,
    VECTOR_MISMATCH,
    // object store
    STORE_ERA,
    STORE_ROOT_ABSENT,
    // There is no room: either the Store's recorded disk budget would be exceeded by an
    // admission that has not started yet, or the filesystem itself answered ENOSPC. It is a
    // VERDICT, never weather: a caller that retries it fails at the same byte forever, which
    // is exactly what the generic IO_FAILED an out-of-space write used to raise bought on
    // rented hardware (tfs-064).
    CAPACITY_EXHAUSTED,
    OBJECT_ABSENT,
    OBJECT_CORRUPT,
    RANGE_BOUNDS,
    NOT_REGULAR_FILE,
    PERMISSION_DENIED,
    IO_FAILED,
    // ingest
    PICKLE_REFUSED,
    UNREGISTERED_ENCODING,
    EVIDENCE_ARM_AMBIGUOUS,
    CROSS_SUBJECT_REPLAY,
    // ingest border (tfs-003): carrier
    ARCHIVE_REFUSED,
    GGUF_REFUSED,
    CARRIER_HEADER_CAP,
    CARRIER_GEOMETRY,
    CARRIER_TRUNCATED,
    CARRIER_OVERLAP,
    // the gguf-v1 planner (tfs-011): a planner's refusals, never a second door's
    GGUF_MAGIC_ABSENT,
    GGUF_VERSION_UNSUPPORTED,
    GGUF_METADATA_TRUNCATED,
    GGUF_TYPE_UNKNOWN,
    GGUF_GEOMETRY,
    GGUF_ARCH_UNDECLARED,
    GGUF_ARCH_MIXED,
    GGUF_SHAPE_UNRECOVERABLE,
    GGUF_CARRIER_TRUNCATED,
    GGUF_ENCODING_NOT_REVIEWED,
    // ingest border: classification and authorization
    UNREGISTERED_FINGERPRINT,
    AMBIGUOUS_CLASSIFICATION,
    PROFILE_REQUIRED,
    TENSOR_SCHEMA_MISMATCH,
    DIALECT_MISMATCH,
    // ingest border: conversion
    CONVERTS_NOTHING,
    MISSING_COMPANION_ROLE,
    DOUBLE_ENCODE,
    REMEDY_FROM_TARGET,
    GOLDEN_MISMATCH,
    PLACEMENT_UNFIT,
    CONVERT_SIZE_CAP,
    SEAM_UNBANKED,
    // An adapter factor targets a module the reviewed converter cannot map; dropping it
    // would change the adapter.
    ADAPTER_TARGET_UNSUPPORTED,
    // ingest border: the temporary candidate transaction
    QUOTA_EXHAUSTED,
    ROOT_EXPIRED,
    ROOT_ABSENT,
    TRANSACTION_CONFLICT,
    TRANSACTION_CLOSED,
    WRITER_FENCED,
    ARTIFACT_INCOMPLETE,
    DISPOSITION_CONFLICT,
    // reads, leases and fill (tfs-005)
    BUFFER_SIZE,
    SHORT_READ,
    LEASE_NOT_COVERED,
    LEASE_REVOKED,
    TRAVERSAL_INCOMPLETE,
    TRAVERSAL_ORDER_MISMATCH,
    CONSTRUCTION_ORDER_REQUIRED,
    PERMUTE_NOT_BLOCK_GRANULAR,
    SLOT_STARVED,
    FD_HEADROOM,
    PLATFORM_UNSUPPORTED,
    PROJECTION_WEIGHT_PATH,
    BLOB_NOT_RETAINED,
    // request-time fit
    MISSING_COMPONENT,
    EXTRA_COMPONENT,
    EXTRA_TENSOR,
    UNCLASSIFIED_EXTRAS,
    // Stamp and the hermetic verifier
    STAMP_MISMATCH,
    EVIDENCE_NOT_RESIDENT,
    VERIFIER_OUTPUT_INCOMPLETE,
    // roots, holds and GC (tfs-006)
    LOCK_CONTENDED,
    ROOT_GENERATION,
    CENSUS_INCOMPLETE,
    SWEEP_SET_INCOMPLETE,
    SNAPSHOT_PINNED,
    STORE_BUSY,
    DURABILITY_UNPROVEN,
    // reclamation (tfs-065): choosing what to stop naming, so a collection can free it
    // This store was never given a disk budget, and having one is the whole opt-in.
    RECLAIM_NOT_PERMITTED,
    // Dropping everything droppable still would not free what was asked for. The honest
    // answer to "this does not fit" is that it does not fit -- never a plan that empties
    // the store and fails anyway.
    RECLAIM_INSUFFICIENT,
    // the ingest durability journal (tfs-057): a chain that is not this conversion's is
    // discarded whole, never partially adopted
    JOURNAL_STALE,
    // transport (tfs-049): the one byte mover; every byte it moves still enters put_stream
    SOURCE_NOT_ALLOWED,
    REDIRECT_REFUSED,
    HUB_UNREACHABLE,
    HUB_REFUSED,
    TLS_UNTRUSTED,
    REF_NOT_FOUND,
    CREDENTIAL_REQUIRED,
    TRANSFER_FAILED,
    DEADLINE_EXCEEDED,
    // ensure: an attempt that landed nothing. Retrying an attempt that made progress is
    // resumption; retrying one that made none is a loop.
    STALLED,
    // repository graph
    REPOSITORY_FULL,
    REPOSITORY_CONFLICT,
    REPOSITORY_ABSENT,
    CHECKPOINT_CONFLICT,
    CHECKPOINT_ABSENT,
    CHECKPOINT_REFERENCED,
    RELEASE_CONFLICT,
    RELEASE_ABSENT,
];

#[derive(Debug, Clone)]
pub struct Refusal {
    pub code: Code,
    pub detail: String,
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code.as_str(), self.detail)
    }
}

impl std::error::Error for Refusal {}

pub type Result<T> = std::result::Result<T, Refusal>;

pub fn refuse<T>(code: Code, detail: impl Into<String>) -> Result<T> {
    Err(Refusal {
        code,
        detail: detail.into(),
    })
}

//! The error families every v2 surface reports (the `tfs` CLI's `--json`, Python): twelve, each
//! with a stable code. The cause stays in the detail.

use crate::err::{Code, Refusal};
use crate::format;
#[cfg(feature = "store")]
use crate::store;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A document, argument or tensor breaks the format or a grammar.
    Invalid,
    /// Another document kind or major version.
    Format,
    /// No such ref, release, lane, component, tensor, file or object.
    NotFound,
    /// Bytes that do not match their digest or length.
    Integrity,
    /// The tensor reads, but its scheme has no decoder here.
    UnknownScheme,
    /// A known scheme whose parts or params break its definition.
    Decode,
    Io,
    /// Storage, device memory or a quota is full.
    Full,
    /// A transfer failed or stalled.
    Network,
    /// Refused: egress policy, credentials, TLS trust or the server.
    Denied,
    /// A source or target this build cannot represent.
    Unsupported,
    /// A revision, conditional write or server policy conflicts with this operation.
    Conflict,
}

impl Kind {
    pub const ALL: [Kind; 12] = [
        Kind::Invalid,
        Kind::Format,
        Kind::NotFound,
        Kind::Integrity,
        Kind::UnknownScheme,
        Kind::Decode,
        Kind::Io,
        Kind::Full,
        Kind::Network,
        Kind::Denied,
        Kind::Unsupported,
        Kind::Conflict,
    ];

    pub fn code(self) -> &'static str {
        match self {
            Kind::Invalid => "invalid",
            Kind::Format => "format",
            Kind::NotFound => "not_found",
            Kind::Integrity => "integrity",
            Kind::UnknownScheme => "unknown_scheme",
            Kind::Decode => "decode",
            Kind::Io => "io",
            Kind::Full => "full",
            Kind::Network => "network",
            Kind::Denied => "denied",
            Kind::Unsupported => "unsupported",
            Kind::Conflict => "conflict",
        }
    }
}

#[derive(Debug, Clone, thiserror::Error)]
#[error("{}: {detail}", kind.code())]
pub struct Failure {
    pub kind: Kind,
    pub detail: String,
}

pub type Result<T> = std::result::Result<T, Failure>;

pub fn fail<T>(kind: Kind, detail: impl Into<String>) -> Result<T> {
    Err(Failure {
        kind,
        detail: detail.into(),
    })
}

impl From<Refusal> for Failure {
    fn from(r: Refusal) -> Failure {
        use Code::*;
        let kind = match r.code {
            UNKNOWN_FORMAT | STORE_ERA => Kind::Format,
            OBJECT_ABSENT | REF_NOT_FOUND | REVISION_ABSENT | REPOSITORY_ABSENT
            | CHECKPOINT_ABSENT | MISSING_TENSOR | MISSING_COMPONENT | STORE_ROOT_ABSENT
            | ROOT_ABSENT => Kind::NotFound,
            OBJECT_ID_MISMATCH
            | OBJECT_CORRUPT
            | LENGTH_MISMATCH
            | WHOLE_DIGEST_MISMATCH
            | PATH_DIGEST_MISMATCH
            | SHORT_READ => Kind::Integrity,
            UNKNOWN_ENCODING | UNREGISTERED_ENCODING => Kind::UnknownScheme,
            ENCODING_MISMATCH => Kind::Decode,
            IO_FAILED | NOT_REGULAR_FILE | FD_HEADROOM | STORE_BUSY => Kind::Io,
            CAPACITY_EXHAUSTED | QUOTA_EXHAUSTED => Kind::Full,
            TRANSFER_FAILED | HUB_UNREACHABLE | STALLED | DEADLINE_EXCEEDED => Kind::Network,
            SOURCE_NOT_ALLOWED | REDIRECT_REFUSED | HUB_REFUSED | TLS_UNTRUSTED
            | CREDENTIAL_REQUIRED | PERMISSION_DENIED => Kind::Denied,
            CHECKPOINT_CONFLICT | RELEASE_CONFLICT | REPOSITORY_CONFLICT | TRANSACTION_CONFLICT
            | DISPOSITION_CONFLICT => Kind::Conflict,
            PICKLE_REFUSED
            | ARCHIVE_REFUSED
            | GGUF_REFUSED
            | AMBIGUOUS_CLASSIFICATION
            | UNREGISTERED_FINGERPRINT
            | CONVERTS_NOTHING
            | MISSING_COMPANION_ROLE
            | ADAPTER_TARGET_UNSUPPORTED
            | CARRIER_HEADER_CAP
            | CARRIER_GEOMETRY
            | CARRIER_TRUNCATED
            | CARRIER_OVERLAP => Kind::Unsupported,
            _ => Kind::Invalid,
        };
        Failure {
            kind,
            detail: format!("{}: {}", r.code.as_str(), r.detail),
        }
    }
}

impl From<format::Error> for Failure {
    fn from(e: format::Error) -> Failure {
        let kind = match e {
            format::Error::Invalid(_) => Kind::Invalid,
            format::Error::Format { .. } | format::Error::PreCut => Kind::Format,
            format::Error::UnknownScheme(_) => Kind::UnknownScheme,
            format::Error::Decode(_) | format::Error::Encode(_) => Kind::Decode,
        };
        Failure {
            kind,
            detail: e.to_string(),
        }
    }
}

#[cfg(feature = "store")]
impl From<crate::io::cuda::Error> for Failure {
    fn from(error: crate::io::cuda::Error) -> Failure {
        use crate::io::cuda;
        let kind = match &error {
            cuda::Error::CudaUnavailable(_) => Kind::Unsupported,
            cuda::Error::Cuda { code, .. } if *code == cuda::ERROR_OUT_OF_MEMORY => Kind::Full,
            _ => Kind::Io,
        };
        Failure { kind, detail: error.to_string() }
    }
}

#[cfg(feature = "store")]
impl From<store::Error> for Failure {
    fn from(e: store::Error) -> Failure {
        let kind = match &e {
            store::Error::Format(f) => return f.clone().into(),
            store::Error::Io { source, .. } => {
                if let Some(refusal) = source.get_ref().and_then(|cause| cause.downcast_ref::<Refusal>()) {
                    return refusal.clone().into();
                }
                if let Some(cuda) = source.get_ref().and_then(|cause| cause.downcast_ref::<crate::io::cuda::Error>()) {
                    return cuda.clone().into();
                }
                Kind::Io
            }
            store::Error::Full { .. } => Kind::Full,
            store::Error::Absent(_) => Kind::NotFound,
            store::Error::Mismatch { .. } => Kind::Integrity,
            store::Error::Invalid { .. } => Kind::Invalid,
        };
        Failure {
            kind,
            detail: e.to_string(),
        }
    }
}

impl From<std::io::Error> for Failure {
    fn from(e: std::io::Error) -> Failure {
        #[cfg(feature = "store")]
        if let Some(cuda) = e.get_ref().and_then(|cause| cause.downcast_ref::<crate::io::cuda::Error>()) {
            return cuda.clone().into();
        }
        let kind = match e.kind() {
            std::io::ErrorKind::StorageFull | std::io::ErrorKind::QuotaExceeded => Kind::Full,
            std::io::ErrorKind::NotFound => Kind::NotFound,
            _ => Kind::Io,
        };
        Failure {
            kind,
            detail: e.to_string(),
        }
    }
}

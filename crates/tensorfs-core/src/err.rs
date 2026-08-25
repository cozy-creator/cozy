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
    // snapshot manifest
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
    OBJECT_ABSENT,
    OBJECT_CORRUPT,
    RANGE_BOUNDS,
    NOT_REGULAR_FILE,
    IO_FAILED,
    // ingest
    PICKLE_REFUSED,
    UNREGISTERED_ENCODING,
    EVIDENCE_ARM_AMBIGUOUS,
    CROSS_SUBJECT_REPLAY,
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

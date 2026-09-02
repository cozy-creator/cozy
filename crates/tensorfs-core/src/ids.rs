//! Identity: ObjectId = sha256 of the exact resident canonical bytes, universally.

use crate::canon::{self, Fields, Value};
use crate::err::{refuse, Code, Result};
use crate::sha256;

/// Bare lowercase 64-hex digest (ObjectRef form).
pub fn hex64(what: &str, s: &str) -> Result<String> {
    if s.len() != 64
        || !s
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return refuse(
            Code::MALFORMED_DIGEST,
            format!("{what}: {s:?} is not 64 lowercase hex"),
        );
    }
    Ok(s.to_string())
}

/// Prefixed `sha256:<hex>` form (the `encoding` field and every id we print).
pub fn prefixed(what: &str, s: &str) -> Result<String> {
    match s.strip_prefix("sha256:") {
        Some(h) => {
            hex64(what, h)?;
            Ok(s.to_string())
        }
        None => refuse(
            Code::MALFORMED_DIGEST,
            format!("{what}: {s:?} lacks the lowercase `sha256:` prefix"),
        ),
    }
}

pub fn object_id(bytes: &[u8]) -> String {
    format!("sha256:{}", sha256::hex_digest(bytes))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectRef {
    pub sha256: String,
    pub length: u64,
}

impl ObjectRef {
    pub fn of(bytes: &[u8]) -> ObjectRef {
        ObjectRef {
            sha256: sha256::hex_digest(bytes),
            length: bytes.len() as u64,
        }
    }
    pub fn id(&self) -> String {
        format!("sha256:{}", self.sha256)
    }
    pub fn from_value(what: &'static str, v: &Value) -> Result<ObjectRef> {
        let mut f = Fields::new(what, v)?;
        let length = f.req_uint("length")?;
        let sha256 = hex64(what, f.req_str("sha256")?)?;
        f.done()?;
        Ok(ObjectRef { sha256, length })
    }
    pub fn to_value(&self) -> Value {
        Value::obj(vec![
            ("length", Value::uint(self.length)),
            ("sha256", Value::str(self.sha256.clone())),
        ])
    }
}

/// Every identity-bearing tensorfs document: one canonical encoding, one id.
pub trait Doc: Sized {
    const FORMAT: &'static str;
    const MAX_BYTES: usize;

    fn from_value(v: &Value) -> Result<Self>;
    fn to_value(&self) -> Value;

    fn canonical_bytes(&self) -> Vec<u8> {
        canon::write(&self.to_value())
    }
    fn object_id(&self) -> String {
        object_id(&self.canonical_bytes())
    }
    fn object_ref(&self) -> ObjectRef {
        ObjectRef::of(&self.canonical_bytes())
    }
    /// The universal law: stored bytes are the canonical bytes.
    fn parse(bytes: &[u8]) -> Result<Self> {
        let v = canon::parse_canonical(bytes, Self::MAX_BYTES)?;
        let doc = Self::from_value(&v)?;
        if doc.canonical_bytes() != bytes {
            return refuse(
                Code::NONCANONICAL_ENCODING,
                "re-emitted bytes differ from stored bytes",
            );
        }
        Ok(doc)
    }
}

/// Strict canonical JSON used only as a bounded command/hash-preimage struct. Plain values
/// have no embedded format tag, no CAS namespace, and no compatibility reader.
pub trait Plain: Sized {
    const MAX_BYTES: usize;
    fn from_value(value: &Value) -> Result<Self>;
    fn to_value(&self) -> Value;

    fn parse(bytes: &[u8]) -> Result<Self> {
        let value = crate::canon::parse_canonical(bytes, Self::MAX_BYTES)?;
        let parsed = Self::from_value(&value)?;
        if parsed.canonical_bytes() != bytes {
            return refuse(Code::NONCANONICAL_ENCODING, "plain value is not canonical");
        }
        Ok(parsed)
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        crate::canon::write(&self.to_value())
    }

    fn object_id(&self) -> String {
        object_id(&self.canonical_bytes())
    }

    fn object_ref(&self) -> ObjectRef {
        ObjectRef::of(&self.canonical_bytes())
    }
}

/// Exact stored-byte documents. Most TensorFS documents use [`Doc`]'s bounded canonical
/// JSON implementation; the CozyTensors header implements this trait directly because its
/// one stored representation is restricted deterministic CBOR.
pub trait StoredDoc: Sized {
    const FORMAT: &'static str;
    const MAX_BYTES: usize;

    fn canonical_bytes(&self) -> Result<Vec<u8>>;
    fn parse(bytes: &[u8]) -> Result<Self>;

    fn object_id(&self) -> Result<String> {
        Ok(object_id(&self.canonical_bytes()?))
    }
    fn object_ref(&self) -> Result<ObjectRef> {
        Ok(ObjectRef::of(&self.canonical_bytes()?))
    }
}

impl<T: Doc> StoredDoc for T {
    const FORMAT: &'static str = T::FORMAT;
    const MAX_BYTES: usize = T::MAX_BYTES;

    fn canonical_bytes(&self) -> Result<Vec<u8>> {
        Ok(<T as Doc>::canonical_bytes(self))
    }
    fn parse(bytes: &[u8]) -> Result<Self> {
        <T as Doc>::parse(bytes)
    }
}

pub fn check_format(what: &'static str, got: &str, want: &str) -> Result<()> {
    if got != want {
        return refuse(
            Code::UNKNOWN_FORMAT,
            format!("{what}: format {got:?} != {want:?}"),
        );
    }
    Ok(())
}

/// ASCII name grammar for component/role names and tenant/dialect ids.
pub fn ascii_name(what: &str, s: &str, max: usize) -> Result<()> {
    if s.is_empty() || s.len() > max {
        return refuse(
            Code::KEY_GRAMMAR,
            format!("{what}: name length {} out of bounds", s.len()),
        );
    }
    if !s
        .bytes()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b'.' | b'/'))
    {
        return refuse(
            Code::KEY_GRAMMAR,
            format!("{what}: {s:?} leaves the ASCII name grammar"),
        );
    }
    Ok(())
}

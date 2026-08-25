//! Capability records (tfs-008): what an (encoding, device) pair is QUALIFIED to execute.
//!
//! Hardware capability never lives on the spec — a spec is an immutable byte contract and
//! hardware is not. It lives here, keyed by the spec DIGEST (never an alias, which can be
//! renamed without changing a single artifact id).
//!
//! The rule is FAIL-CLOSED, and it is one rule with no default branch: a pair with no
//! qualified record refuses. This is the DTYPE_MIN_SM class, kept with force under its new
//! owner — v1 carried a minimum-SM floor that fell OPEN when the device was unrecognised,
//! so an unknown card silently got the fast path and produced plausible wrong outputs. An
//! unknown device is not a permissive case here; it is the absence of evidence.
//!
//! A record cannot exist without naming the exact vector set its implementation passed, so
//! a vectorless spec is structurally unqualifiable rather than merely discouraged.

use crate::canon::{as_arr, as_str, Fields, Value};
use crate::err::{refuse, Code, Result};
use crate::ids::{ascii_name, check_format, prefixed, Doc, ObjectRef};
use crate::limits;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityRecord {
    /// `sha256:<hex>` of the EncodingSpec object. Never an alias.
    pub encoding: String,
    /// Device class, e.g. `cuda.sm89`, `cuda.sm100`, `cpu`.
    pub device: String,
    /// What was reviewed and qualified.
    pub implementation: String,
    /// The exact vector set this implementation passed BIT-FOR-BIT. Mandatory: there is no
    /// review-only qualification, so a vectorless spec can never appear here.
    pub vectors: ObjectRef,
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CapabilityRecords {
    pub records: Vec<CapabilityRecord>,
}

impl CapabilityRecords {
    /// The ONE admission decision. No default, no fallback, no "unknown means allowed".
    pub fn admit(&self, encoding: &str, device: &str) -> Result<&CapabilityRecord> {
        match self
            .records
            .iter()
            .find(|r| r.encoding == encoding && r.device == device)
        {
            Some(r) => Ok(r),
            None => {
                let elsewhere: Vec<&str> = self
                    .records
                    .iter()
                    .filter(|r| r.encoding == encoding)
                    .map(|r| r.device.as_str())
                    .collect();
                refuse(
                    Code::CAPABILITY_UNQUALIFIED,
                    format!(
                        "no qualified record for ({encoding}, {device}). Qualified devices for \
                         this encoding: {elsewhere:?}. An unrecognised device is the ABSENCE \
                         of evidence, never permission — remedy: qualify an implementation \
                         against this spec's vectors on this device."
                    ),
                )
            }
        }
    }
}

impl Doc for CapabilityRecords {
    const FORMAT: &'static str = "tensorfs.capability_records/1";
    const MAX_BYTES: usize = limits::DOC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("CapabilityRecords", v)?;
        check_format(
            "CapabilityRecords",
            as_str("CapabilityRecords", "format", f.req("format")?)?,
            Self::FORMAT,
        )?;
        let mut records = Vec::new();
        for rv in as_arr("CapabilityRecords", "records", f.req("records")?)? {
            let mut rf = Fields::new("CapabilityRecord", rv)?;
            let device = rf.req_str("device")?.to_string();
            let encoding = prefixed(
                "CapabilityRecord.encoding",
                as_str("CapabilityRecord", "encoding", rf.req("encoding")?)?,
            )?;
            let implementation = rf.req_str("implementation")?.to_string();
            let note = rf.req_str("note")?.to_string();
            let vectors = ObjectRef::from_value("CapabilityRecord.vectors", rf.req("vectors")?)?;
            rf.done()?;
            ascii_name("CapabilityRecord.device", &device, limits::MAX_NAME_BYTES)?;
            ascii_name(
                "CapabilityRecord.implementation",
                &implementation,
                limits::MAX_NAME_BYTES,
            )?;
            if note.len() > limits::MAX_DOC_TEXT_BYTES {
                return refuse(Code::SIZE_CAP, "CapabilityRecord.note over the byte cap");
            }
            records.push(CapabilityRecord {
                encoding,
                device,
                implementation,
                vectors,
                note,
            });
        }
        f.done()?;
        // One (encoding, device) pair, one verdict: two records for one pair is a registry
        // that can answer the same question two ways.
        let mut keys: Vec<(&str, &str)> = records
            .iter()
            .map(|r| (r.encoding.as_str(), r.device.as_str()))
            .collect();
        let given = keys.clone();
        keys.sort_unstable();
        keys.dedup();
        if keys.len() != given.len() {
            return refuse(
                Code::DUPLICATE_KEY,
                "two capability records name one (encoding, device) pair",
            );
        }
        Ok(CapabilityRecords { records })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            ("format", Value::str(Self::FORMAT)),
            (
                "records",
                Value::arr(
                    self.records
                        .iter()
                        .map(|r| {
                            Value::obj(vec![
                                ("device", Value::str(r.device.clone())),
                                ("encoding", Value::str(r.encoding.clone())),
                                ("implementation", Value::str(r.implementation.clone())),
                                ("note", Value::str(r.note.clone())),
                                ("vectors", r.vectors.to_value()),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    }
}

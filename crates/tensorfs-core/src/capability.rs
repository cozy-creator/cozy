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
//!
//! The record set is a compiled-in registry constant (`registry::capability_records`), never
//! written, stored, or digested — a plain struct, not a canonical document (proto-007).

use crate::canon::{self, Value};
use crate::err::{refuse, Code, Result};
use crate::ids::ObjectRef;

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

impl CapabilityRecords {
    pub fn to_value(&self) -> Value {
        Value::obj(vec![(
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
        )])
    }
    pub fn canonical_bytes(&self) -> Vec<u8> {
        canon::write(&self.to_value())
    }
}

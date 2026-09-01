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

use crate::canon::{self, Fields, Value};
use crate::err::{refuse, Code, Result};
use crate::ids::ObjectRef;
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
                         of evidence, never permission -- remedy: qualify an implementation \
                         against this spec's vectors on this device."
                    ),
                )
            }
        }
    }
}

impl CapabilityRecords {
    /// Read an OBSERVED record set -- one produced by a component that actually holds the
    /// card. TensorFS owns no device and cannot measure one, so an accelerator record can
    /// only ever arrive from outside; what stays here is the RULE that judges it.
    ///
    /// The observer says what it qualified and where. It does NOT say which vector set that
    /// qualification passed, because that is not an observer's fact to state -- the spec
    /// pins it, and a record whose vector set the observer could choose would let an
    /// observer qualify a spec against bytes the spec never carried. So this binds the
    /// vector set from the registry and refuses a record whose encoding is not a VECTORED
    /// spec here at all. The vectorless law is enforced at the border rather than trusted:
    /// without it, "observed" would mean nothing stronger than "asserted".
    pub fn parse_observed(bytes: &[u8]) -> Result<Self> {
        Self::observed_from_value(&canon::parse_canonical(bytes, limits::DOC_MAX_BYTES)?)
    }

    pub fn observed_from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("ObservedCapabilityRecords", v)?;
        let rows = canon::as_arr("ObservedCapabilityRecords", "records", f.req("records")?)?;
        f.done()?;
        if rows.len() > limits::MAX_ENTRIES {
            return refuse(
                Code::COUNT_CAP,
                format!("{} observed capability records exceeds the cap", rows.len()),
            );
        }
        let vectored = crate::registry::vector_refs();
        let mut records = Vec::with_capacity(rows.len());
        for row in rows {
            let mut rf = Fields::new("ObservedCapabilityRecord", row)?;
            let device = rf.req_str("device")?.to_string();
            let encoding =
                crate::ids::prefixed("ObservedCapabilityRecord.encoding", rf.req_str("encoding")?)?;
            let implementation = rf.req_str("implementation")?.to_string();
            let note = rf.req_str("note")?.to_string();
            rf.done()?;
            if device.is_empty() || implementation.is_empty() {
                return refuse(
                    Code::MISSING_FIELD,
                    "a capability record names a device and the implementation it qualified",
                );
            }
            let vectors = match vectored.iter().find(|(id, _)| *id == encoding) {
                Some((_, v)) => v.clone(),
                None => {
                    return refuse(
                        Code::VECTORS_REQUIRED,
                        format!(
                            "observed record for {encoding} on {device}: no VECTORED spec by \
                             that digest is in the platform namespace, so there are no bytes \
                             an implementation could have been qualified against"
                        ),
                    )
                }
            };
            records.push(CapabilityRecord {
                encoding,
                device,
                implementation,
                vectors,
                note,
            });
        }
        Ok(CapabilityRecords { records })
    }

    /// The compiled-in rule set, WIDENED by an observation. Observed records are appended,
    /// never substituted: `admit` still finds one exact pair, and the compiled-in
    /// reference-decoder records keep answering for the device they were measured on.
    pub fn with_observed(mut self, observed: CapabilityRecords) -> CapabilityRecords {
        for r in observed.records {
            if !self
                .records
                .iter()
                .any(|x| x.encoding == r.encoding && x.device == r.device)
            {
                self.records.push(r);
            }
        }
        self
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry;

    fn plain() -> String {
        registry::seeds()
            .iter()
            .find(|s| s.alias == "plain/1")
            .expect("plain/1 is entry zero")
            .spec
            .object_id()
    }

    fn observed(device: &str, encoding: &str) -> Vec<u8> {
        canon::write(&Value::obj(vec![(
            "records",
            Value::arr(vec![Value::obj(vec![
                ("device", Value::str(device)),
                ("encoding", Value::str(encoding)),
                ("implementation", Value::str("cozy-runtime:verbatim@0")),
                ("note", Value::str("probe observation")),
            ])]),
        )]))
    }

    /// The compiled-in set speaks for the reference decoders on cpu and for NOTHING else.
    /// An accelerator answer exists only once something that held the card said so.
    #[test]
    fn an_accelerator_refuses_until_an_observation_arrives() {
        let enc = plain();
        let base = registry::capability_records();
        assert!(base.admit(&enc, "cpu").is_ok());
        assert_eq!(
            base.admit(&enc, "cuda.sm89").unwrap_err().code,
            Code::CAPABILITY_UNQUALIFIED
        );
        let widened = registry::capability_records().with_observed(
            CapabilityRecords::parse_observed(&observed("cuda.sm89", &enc)).unwrap(),
        );
        assert_eq!(
            widened.admit(&enc, "cuda.sm89").unwrap().implementation,
            "cozy-runtime:verbatim@0"
        );
        // One observation qualifies ONE pair. A neighbouring card is still the absence of
        // evidence, which is the whole point of a fail-closed rule.
        assert!(widened.admit(&enc, "cuda.sm100").is_err());
    }

    /// The observer states what it qualified, never which bytes it was qualified against.
    /// The vector set comes from the spec, so a record naming a spec with no vectors has
    /// nothing it could have passed and is refused at the border.
    #[test]
    fn a_vectorless_spec_cannot_be_observed_into_qualification() {
        let vectorless = registry::seeds()
            .iter()
            .find(|s| s.spec.vectors.is_none())
            .expect("the registry seeds at least one vectorless spec")
            .spec
            .object_id();
        let e = CapabilityRecords::parse_observed(&observed("cuda.sm89", &vectorless)).unwrap_err();
        assert_eq!(e.code, Code::VECTORS_REQUIRED);
    }

    /// The bound vector set is the SPEC's, not a copy the observer chose.
    #[test]
    fn the_bound_vector_set_is_the_specs_own() {
        let enc = plain();
        let want = registry::seeds()
            .iter()
            .find(|s| s.spec.object_id() == enc)
            .and_then(|s| s.spec.vectors.clone())
            .expect("plain/1 is vectored");
        let parsed = CapabilityRecords::parse_observed(&observed("cuda.sm89", &enc)).unwrap();
        assert_eq!(parsed.records[0].vectors, want);
    }

    /// An observation never overwrites what TensorFS measured for itself. A CPU worker's
    /// honest report is readable -- it is simply not authoritative over the compiled-in
    /// record for the same pair, which keeps answering.
    #[test]
    fn an_observation_never_displaces_a_compiled_in_record() {
        let enc = plain();
        let widened = registry::capability_records()
            .with_observed(CapabilityRecords::parse_observed(&observed("cpu", &enc)).unwrap());
        assert_eq!(
            widened.admit(&enc, "cpu").unwrap().implementation,
            "tfs-conform-reference"
        );
    }

    /// An unknown digest is not a new spec, it is an unqualifiable one.
    #[test]
    fn an_unknown_encoding_digest_is_refused() {
        let e = CapabilityRecords::parse_observed(&observed(
            "cuda.sm89",
            &format!("sha256:{}", "0".repeat(64)),
        ))
        .unwrap_err();
        assert_eq!(e.code, Code::VECTORS_REQUIRED);
    }
}

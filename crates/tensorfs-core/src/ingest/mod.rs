//! The ingest border (tfs-003) — the ONLY place in this repo that parses a foreign
//! carrier. Everything under `ingest/` is inside the fourth fence; nothing outside it may
//! name a foreign container.
//!
//! This file holds the ingest DOCUMENTS: profile (dialect + converter + limits), subject
//! (tenant + exact source generations + proposed header), and the ONE merged verification
//! receipt. Receipts are evidence, never publication authority.
//!
//! The border's four stages, each its own file:
//!   carrier     — strict bounded foreign-carrier decode (refuse before allocation)
//!   fingerprint — classification into a reviewed registry verdict; ambiguity refuses
//!   convert     — the PLAN (pure function of headers + target) and the reviewed converters
//!   transaction — temporary candidate write through the real CAS, receipt, install/reap
//!
//! And one stage that runs BEFORE all of them:
//!   preflight   — the plan, decided from headers over sparse holes, before a byte moves

pub mod carrier;
pub mod convert;
pub mod custody;
pub mod fingerprint;
pub mod gguf;
mod h3_lora;
pub mod journal;
mod narrow;
pub mod preflight;
pub mod routes;
pub mod source;
pub mod stamp;
pub mod transaction;

// The gguf-v1 PLANNER (tfs-011) is inside this fence because it parses a foreign
// container, and NOT because it is a door: `carrier::sniff` still refuses GGUF on the
// normal path. Re-exported so a caller outside the border can name the entry point
// without naming the container.
pub use gguf::{check_twin as gguf_check_twin, plan as gguf_plan, GgufPlan, ShapeSource};

use crate::canon::{as_arr, as_str, Fields, Value};
use crate::err::{refuse, Code, Result};
use crate::ids::{ascii_name, prefixed, Doc, ObjectRef, Plain};
use crate::limits;

const DIALECTS: [&str; 5] = [
    "cozytensors",
    "safetensors.diffusers",
    "safetensors.single_file",
    "gguf",
    "pickle",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestProfile {
    /// `normal` or `authorized_one_shot`
    pub class: String,
    pub source_dialect: String,
    pub converter_build: String,
    pub converter_recipe: ObjectRef,
    pub expected_tensor_schema_digest: String,
    pub allowed_encodings: Vec<ObjectRef>,
    pub max_source_bytes: u64,
    pub max_tensors: u64,
}

impl IngestProfile {
    /// Border policy: a normal profile can name neither pickle nor an unregistered encoding.
    pub fn validate(&self, platform_aliased: &[String]) -> Result<()> {
        if self.class == "normal" && self.source_dialect == "pickle" {
            return refuse(
                Code::PICKLE_REFUSED,
                "pickle always refuses normal ingest; only a separately authorized one-shot converter may read it",
            );
        }
        for e in &self.allowed_encodings {
            if !platform_aliased.contains(&e.id()) {
                return refuse(
                    Code::UNREGISTERED_ENCODING,
                    format!(
                        "{}: not a platform-aliased spec digest (launch border policy)",
                        e.id()
                    ),
                );
            }
        }
        Ok(())
    }
}

impl Plain for IngestProfile {
    const MAX_BYTES: usize = limits::SPEC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("IngestProfile", v)?;
        let class = f.req_str("class")?.to_string();
        if class != "normal" && class != "authorized_one_shot" {
            return refuse(
                Code::UNKNOWN_FIELD,
                format!("class {class:?} is outside the closed set"),
            );
        }
        let source_dialect = f.req_str("source_dialect")?.to_string();
        if !DIALECTS.contains(&source_dialect.as_str())
            && !convert::CONVERTERS
                .iter()
                .any(|converter| converter.dialect == source_dialect)
        {
            return refuse(
                Code::UNKNOWN_FIELD,
                format!("source_dialect {source_dialect:?} is unknown"),
            );
        }
        let mut cf = Fields::new("IngestProfile.converter", f.req("converter")?)?;
        let converter_build = prefixed("converter.build", cf.req_str("build")?)?;
        let converter_recipe = ObjectRef::from_value("converter.recipe", cf.req("recipe")?)?;
        cf.done()?;
        let expected_tensor_schema_digest = prefixed(
            "expected_tensor_schema_digest",
            f.req_str("expected_tensor_schema_digest")?,
        )?;
        let mut allowed_encodings = Vec::new();
        for e in as_arr(
            "IngestProfile",
            "allowed_encodings",
            f.req("allowed_encodings")?,
        )? {
            allowed_encodings.push(ObjectRef::from_value("allowed_encodings", e)?);
        }
        let mut lf = Fields::new("IngestProfile.limits", f.req("limits")?)?;
        let max_source_bytes = lf.req_uint("max_source_bytes")?;
        let max_tensors = lf.req_uint("max_tensors")?;
        lf.done()?;
        f.done()?;
        Ok(IngestProfile {
            class,
            source_dialect,
            converter_build,
            converter_recipe,
            expected_tensor_schema_digest,
            allowed_encodings,
            max_source_bytes,
            max_tensors,
        })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            (
                "allowed_encodings",
                Value::arr(
                    self.allowed_encodings
                        .iter()
                        .map(|e| e.to_value())
                        .collect(),
                ),
            ),
            ("class", Value::str(self.class.clone())),
            (
                "converter",
                Value::obj(vec![
                    ("build", Value::str(self.converter_build.clone())),
                    ("recipe", self.converter_recipe.to_value()),
                ]),
            ),
            (
                "expected_tensor_schema_digest",
                Value::str(self.expected_tensor_schema_digest.clone()),
            ),
            (
                "limits",
                Value::obj(vec![
                    ("max_source_bytes", Value::uint(self.max_source_bytes)),
                    ("max_tensors", Value::uint(self.max_tensors)),
                ]),
            ),
            ("source_dialect", Value::str(self.source_dialect.clone())),
        ])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestSubject {
    pub profile: ObjectRef,
    pub tenant: String,
    /// exact source generations: (digest of the source identity, generation)
    pub sources: Vec<(String, u64)>,
    pub proposed_header: ObjectRef,
    pub candidates: Vec<ObjectRef>,
}

impl Plain for IngestSubject {
    // The subject's `candidates` is EVERY object the manifest reaches, so it scales with the
    // artifact exactly as the header and the manifest do — 2,601 refs for a 6.9 GB SDXL
    // tree, 244 KiB of canonical bytes. It shared the fixed 64 KiB spec cap and therefore
    // WROTE fine and REFUSED SIZE_CAP on the way back in, which made `install` impossible
    // for any artifact over ~700 objects. Same cap as the other object-set documents.
    const MAX_BYTES: usize = limits::DOC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("IngestSubject", v)?;
        let mut candidates = Vec::new();
        for c in as_arr("IngestSubject", "candidates", f.req("candidates")?)? {
            candidates.push(ObjectRef::from_value("candidates", c)?);
        }
        let profile = ObjectRef::from_value("IngestSubject.profile", f.req("profile")?)?;
        let proposed_header =
            ObjectRef::from_value("IngestSubject.proposed_header", f.req("proposed_header")?)?;
        let mut sources = Vec::new();
        for s in as_arr("IngestSubject", "sources", f.req("sources")?)? {
            let mut sf = Fields::new("IngestSubject.source", s)?;
            let generation = sf.req_uint("generation")?;
            let digest = prefixed("source.source_digest", sf.req_str("source_digest")?)?;
            sf.done()?;
            sources.push((digest, generation));
        }
        let tenant = f.req_str("tenant")?.to_string();
        ascii_name("IngestSubject.tenant", &tenant, limits::MAX_NAME_BYTES)?;
        f.done()?;
        if sources.is_empty() {
            return refuse(Code::MISSING_FIELD, "IngestSubject.sources is empty");
        }
        Ok(IngestSubject {
            profile,
            tenant,
            sources,
            proposed_header,
            candidates,
        })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            (
                "candidates",
                Value::arr(self.candidates.iter().map(|c| c.to_value()).collect()),
            ),
            ("profile", self.profile.to_value()),
            ("proposed_header", self.proposed_header.to_value()),
            (
                "sources",
                Value::arr(
                    self.sources
                        .iter()
                        .map(|(d, g)| {
                            Value::obj(vec![
                                ("generation", Value::uint(*g)),
                                ("source_digest", Value::str(d.clone())),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("tenant", Value::str(self.tenant.clone())),
        ])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Evidence {
    ApprovedProducer {
        implementation: String,
        recipe: ObjectRef,
    },
    IndependentNumerical {
        verifier_build: String,
        report: ObjectRef,
    },
}

/// The ONE ingest receipt (the former separate LayoutReceipt is merged in).
/// Signer/observations live in the envelope — OUTSIDE every subject identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestVerificationReceipt {
    pub subject: ObjectRef,
    pub manifest: ObjectRef,
    pub header: ObjectRef,
    /// The ledger of what Stamp actually verified (tfs-009). The receipt attests; the
    /// stamp records the facts a consumer would otherwise have to re-derive — which is how
    /// a second decision layer gets built by accident.
    pub stamp: ObjectRef,
    pub evidence: Evidence,
    pub signer: String,
    pub signature: String,
    pub observations: Vec<(String, i64)>,
}

impl IngestVerificationReceipt {
    /// A receipt is evidence about ONE exact subject/manifest/header/stamp tuple.
    pub fn binds(
        &self,
        subject: &IngestSubject,
        manifest: &crate::manifest::Manifest,
        header: &crate::header::Header,
        stamp: &stamp::Stamp,
    ) -> Result<()> {
        for (what, got, want) in [
            ("subject", self.subject.clone(), subject.object_ref()),
            ("manifest", self.manifest.clone(), manifest.object_ref()),
            ("header", self.header.clone(), header.object_ref()?),
            ("stamp", self.stamp.clone(), stamp.object_ref()),
        ] {
            if got != want {
                return refuse(
                    Code::CROSS_SUBJECT_REPLAY,
                    format!("receipt {what} {} != presented {}", got.id(), want.id()),
                );
            }
        }
        Ok(())
    }
}

impl Plain for IngestVerificationReceipt {
    const MAX_BYTES: usize = limits::SPEC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("IngestVerificationReceipt", v)?;
        let arms = as_arr("IngestVerificationReceipt", "evidence", f.req("evidence")?)?;
        if arms.len() != 1 {
            return refuse(
                Code::EVIDENCE_ARM_AMBIGUOUS,
                format!(
                    "{} evidence arms: a receipt carries exactly one",
                    arms.len()
                ),
            );
        }
        let mut af = Fields::new("evidence", &arms[0])?;
        let t = as_str("evidence", "t", af.req("t")?)?.to_string();
        let evidence = match t.as_str() {
            "approved_producer" => Evidence::ApprovedProducer {
                implementation: prefixed("evidence.implementation", af.req_str("implementation")?)?,
                recipe: ObjectRef::from_value("evidence.recipe", af.req("recipe")?)?,
            },
            "independent_numerical" => Evidence::IndependentNumerical {
                verifier_build: prefixed("evidence.verifier_build", af.req_str("verifier_build")?)?,
                report: ObjectRef::from_value("evidence.report", af.req("report")?)?,
            },
            other => {
                return refuse(
                    Code::UNKNOWN_FIELD,
                    format!("evidence arm {other:?} is unknown"),
                )
            }
        };
        af.done()?;
        let header = ObjectRef::from_value("receipt.header", f.req("header")?)?;
        let mut observations = Vec::new();
        for (k, ov) in crate::canon::as_obj("receipt", "observations", f.req("observations")?)? {
            observations.push((
                k.clone(),
                match ov {
                    Value::Int(i) => *i,
                    o => {
                        return refuse(
                            Code::WRONG_TYPE,
                            format!("observation {k:?}: expected integer, got {}", o.kind()),
                        )
                    }
                },
            ));
        }
        let signature = f.req_str("signature")?.to_string();
        let signer = f.req_str("signer")?.to_string();
        let manifest = ObjectRef::from_value("receipt.manifest", f.req("manifest")?)?;
        let stamp = ObjectRef::from_value("receipt.stamp", f.req("stamp")?)?;
        let subject = ObjectRef::from_value("receipt.subject", f.req("subject")?)?;
        f.done()?;
        Ok(IngestVerificationReceipt {
            subject,
            manifest,
            header,
            stamp,
            evidence,
            signer,
            signature,
            observations,
        })
    }

    fn to_value(&self) -> Value {
        let arm = match &self.evidence {
            Evidence::ApprovedProducer {
                implementation,
                recipe,
            } => Value::obj(vec![
                ("implementation", Value::str(implementation.clone())),
                ("recipe", recipe.to_value()),
                ("t", Value::str("approved_producer")),
            ]),
            Evidence::IndependentNumerical {
                verifier_build,
                report,
            } => Value::obj(vec![
                ("report", report.to_value()),
                ("t", Value::str("independent_numerical")),
                ("verifier_build", Value::str(verifier_build.clone())),
            ]),
        };
        Value::obj(vec![
            ("evidence", Value::arr(vec![arm])),
            ("header", self.header.to_value()),
            (
                "observations",
                Value::map(
                    self.observations
                        .iter()
                        .map(|(k, i)| (k.clone(), Value::Int(*i)))
                        .collect(),
                ),
            ),
            ("signature", Value::str(self.signature.clone())),
            ("signer", Value::str(self.signer.clone())),
            ("manifest", self.manifest.to_value()),
            ("stamp", self.stamp.to_value()),
            ("subject", self.subject.to_value()),
        ])
    }
}

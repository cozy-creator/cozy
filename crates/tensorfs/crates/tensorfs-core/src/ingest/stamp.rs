//! `Stamp` — the layout plane's FIRST operation, and the only thing that can mint an
//! IngestVerificationReceipt.
//!
//! Stamp runs INSIDE isolated ingest and verifies, against live bytes rather than against
//! what the run reported: the exact IngestSubject/IngestProfile pair, the source
//! generations, the converter build/lock/recipe, the source observations, the canonical
//! CozyTensorsHeader, the computed `tensor_schema_digest`, the encoding part roles, the candidate
//! object set, the golden-suite result and the bounded resource accounting.
//!
//! It emits ONE receipt carrying exactly ONE evidence arm — approved-producer
//! recipe/implementation, or independent numerical verification. **A receipt is not a
//! publication**: `transaction::install` is the coordinator's separate act, and nothing
//! here writes a root.
//!
//! The ledger is its own canonical document. The receipt ATTESTS; the Stamp records WHAT
//! WAS VERIFIED, and the non-integer facts (tensor schema digest, converter recipe digest, the
//! candidate-set digest, the golden verdict, the isolation tier) have nowhere else to live.
//! Without it every consumer re-derives them — which is a second decision layer wearing a
//! consumer's clothes. th-002 verifies precisely this set.

use crate::canon::{as_arr, as_obj, Fields, Value};
use crate::checkpoint as ck;
use crate::err::{refuse, Code, Result};
use crate::header::Header;
use crate::ids::{ascii_name, prefixed, Doc, ObjectRef, Plain};
use crate::limits;
use crate::manifest::Manifest;
use crate::store::Store;

use super::transaction::Outcome;
use super::{Evidence, IngestProfile, IngestSubject, IngestVerificationReceipt};

/// The golden-suite verdict vocabulary. Closed: "the converter ran" is not a verdict.
pub const GOLDEN_PASS: &str = "pass";
pub const GOLDEN_FAIL: &str = "fail";
pub const GOLDEN_ABSENT: &str = "absent";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stamp {
    pub subject: ObjectRef,
    pub profile: ObjectRef,
    pub header: ObjectRef,
    pub manifest: ObjectRef,
    /// COMPUTED from the header that was actually written — never copied from the profile.
    pub tensor_schema_digest: String,
    pub converter_build: String,
    pub converter_recipe: ObjectRef,
    /// sha256 over the sorted candidate id list: the candidate SET as one comparable fact.
    pub candidate_set: String,
    pub golden: String,
    pub isolation_tier: u64,
    pub sources: Vec<(String, u64)>,
    pub accounting: Vec<(String, i64)>,
}

/// The digest of a candidate SET, order-independent and cardinality-sensitive.
pub fn candidate_set_digest(ids: &[String]) -> String {
    let mut v: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
    v.sort_unstable();
    v.dedup();
    crate::ids::object_id(&crate::canon::write(&Value::arr(
        v.into_iter().map(Value::str).collect(),
    )))
}

impl Plain for Stamp {
    const MAX_BYTES: usize = limits::SPEC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("Stamp", v)?;
        let mut accounting = Vec::new();
        for (k, av) in as_obj("Stamp", "accounting", f.req("accounting")?)? {
            ascii_name("Stamp.accounting", k, limits::MAX_NAME_BYTES)?;
            accounting.push((
                k.clone(),
                match av {
                    Value::Int(i) => *i,
                    o => {
                        return refuse(
                            Code::WRONG_TYPE,
                            format!("accounting {k:?}: expected integer, got {}", o.kind()),
                        )
                    }
                },
            ));
        }
        let candidate_set = prefixed("Stamp.candidate_set", f.req_str("candidate_set")?)?;
        let converter_build = prefixed("Stamp.converter_build", f.req_str("converter_build")?)?;
        let converter_recipe =
            ObjectRef::from_value("Stamp.converter_recipe", f.req("converter_recipe")?)?;
        let golden = f.req_str("golden")?.to_string();
        let header = ObjectRef::from_value("Stamp.header", f.req("header")?)?;
        let isolation_tier = f.req_uint("isolation_tier")?;
        let profile = ObjectRef::from_value("Stamp.profile", f.req("profile")?)?;
        let manifest = ObjectRef::from_value("Stamp.manifest", f.req("manifest")?)?;
        let mut sources = Vec::new();
        for s in as_arr("Stamp", "sources", f.req("sources")?)? {
            let mut sf = Fields::new("Stamp.source", s)?;
            let generation = sf.req_uint("generation")?;
            let digest = prefixed("Stamp.source_digest", sf.req_str("source_digest")?)?;
            sf.done()?;
            sources.push((digest, generation));
        }
        let subject = ObjectRef::from_value("Stamp.subject", f.req("subject")?)?;
        let tensor_schema_digest = prefixed(
            "Stamp.tensor_schema_digest",
            f.req_str("tensor_schema_digest")?,
        )?;
        f.done()?;
        if ![GOLDEN_PASS, GOLDEN_FAIL, GOLDEN_ABSENT].contains(&golden.as_str()) {
            return refuse(
                Code::UNKNOWN_FIELD,
                format!("golden {golden:?} is outside the closed verdict set"),
            );
        }
        if sources.is_empty() {
            return refuse(Code::MISSING_FIELD, "Stamp.sources is empty");
        }
        Ok(Stamp {
            subject,
            profile,
            header,
            manifest,
            tensor_schema_digest,
            converter_build,
            converter_recipe,
            candidate_set,
            golden,
            isolation_tier,
            sources,
            accounting,
        })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            (
                "accounting",
                Value::map(
                    self.accounting
                        .iter()
                        .map(|(k, i)| (k.clone(), Value::Int(*i)))
                        .collect(),
                ),
            ),
            ("candidate_set", Value::str(self.candidate_set.clone())),
            ("converter_build", Value::str(self.converter_build.clone())),
            ("converter_recipe", self.converter_recipe.to_value()),
            ("golden", Value::str(self.golden.clone())),
            ("header", self.header.to_value()),
            ("isolation_tier", Value::uint(self.isolation_tier)),
            ("profile", self.profile.to_value()),
            ("manifest", self.manifest.to_value()),
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
            ("subject", self.subject.to_value()),
            (
                "tensor_schema_digest",
                Value::str(self.tensor_schema_digest.clone()),
            ),
        ])
    }
}

/// What a stamped ingest produced. Both documents, or neither.
#[derive(Debug)]
pub struct Stamped {
    pub stamp: Stamp,
    pub receipt: IngestVerificationReceipt,
}

/// Everything the caller must hand over to be stamped. Named rather than positional: a
/// twelve-argument verification function is one transposition away from stamping the wrong
/// thing.
pub struct Presented<'a> {
    pub subject: &'a IngestSubject,
    pub profile: &'a IngestProfile,
    pub profile_ref: &'a ObjectRef,
    pub outcome: &'a Outcome,
    pub golden: &'a str,
    pub isolation_tier: u64,
    pub evidence: Evidence,
    pub signer: &'a str,
    pub signature: &'a str,
    /// The registry pin the border admits against (platform-aliased spec digests).
    pub pin: &'a [String],
}

fn disagree(what: &str, got: &str, want: &str) -> crate::err::Refusal {
    crate::err::Refusal {
        code: Code::STAMP_MISMATCH,
        detail: format!("{what}: stamped {got}, presented {want}"),
    }
}

/// The verification. Everything is re-derived from bytes the store holds; nothing is taken
/// on the run's word. A failure anywhere means NO receipt, which is the whole mechanism:
/// there is no path from an unverified candidate to an installed checkpoint.
pub fn stamp(store: &Store, p: Presented) -> Result<Stamped> {
    // 1. subject <-> profile. A subject that names a different profile than the one whose
    //    limits were applied is two runs wearing one identity.
    if p.subject.profile != *p.profile_ref {
        return Err(disagree(
            "subject.profile",
            &p.subject.profile.id(),
            &p.profile_ref.id(),
        ));
    }
    if p.profile.object_ref() != *p.profile_ref {
        return Err(disagree(
            "profile bytes",
            &p.profile.object_ref().id(),
            &p.profile_ref.id(),
        ));
    }
    // 2. border policy on the profile itself, re-run against the pin.
    p.profile.validate(p.pin)?;

    // 3. the proposed header IS the header that was written.
    if p.subject.proposed_header != p.outcome.header_ref {
        return Err(disagree(
            "subject.proposed_header",
            &p.subject.proposed_header.id(),
            &p.outcome.header_ref.id(),
        ));
    }

    // 4. canonical header, read back OUT OF THE STORE and strictly re-parsed. `Doc::parse`
    //    re-emits and compares, so a non-canonical twin cannot pass here.
    let header: Header = ck::load_header(store, &p.outcome.header_ref)?;
    if header != p.outcome.header {
        return Err(disagree(
            "header",
            &header.object_ref()?.id(),
            &p.outcome.header.object_ref()?.id(),
        ));
    }
    let closure = ck::load_closure(store, &header)?;
    header.validate(&closure)?;

    // 5. the computed tensor schema digest against the profile's expectation. COMPUTED from the
    //    header just re-read, never carried over from the plan.
    let tensor_schema_digest = header.tensor_schema_digest();
    if tensor_schema_digest != p.profile.expected_tensor_schema_digest {
        return refuse(
            Code::TENSOR_SCHEMA_MISMATCH,
            format!(
                "computed tensor schema {tensor_schema_digest} != the profile's expected {}",
                p.profile.expected_tensor_schema_digest
            ),
        );
    }

    // 6. encoding part roles: every cited spec is one the profile allowed, and its exact
    //    role set is what the tensors carry (header.validate proved the second half).
    let allowed: Vec<String> = p.profile.allowed_encodings.iter().map(|e| e.id()).collect();
    for r in &header.encodings {
        if !allowed.contains(&r.object_id()) {
            return refuse(
                Code::UNREGISTERED_ENCODING,
                format!(
                    "{}: cited by the candidate header but outside the profile's allowed set",
                    r.object_id()
                ),
            );
        }
    }

    // 7. the candidate object set, EVERY AND ONLY. The walk is the authority; the subject's
    //    list is the claim.
    let manifest: Manifest = ck::load_manifest(store, &p.outcome.manifest_ref)?;
    if manifest != p.outcome.manifest {
        return Err(disagree(
            "manifest",
            &manifest.object_ref().id(),
            &p.outcome.manifest.object_ref().id(),
        ));
    }
    let walk = ck::walk(store, &manifest)?;
    let reached: Vec<String> = walk.distinct().iter().map(|o| o.id()).collect();
    let claimed: Vec<String> = p.subject.candidates.iter().map(|c| c.id()).collect();
    let reached_set = candidate_set_digest(&reached);
    let claimed_set = candidate_set_digest(&claimed);
    if reached_set != claimed_set {
        let first_missing = reached.iter().find(|r| !claimed.contains(r));
        let first_surplus = claimed.iter().find(|c| !reached.contains(c));
        return refuse(
            Code::STAMP_MISMATCH,
            format!(
                "candidate set disagrees: the manifest reaches {} object(s), the subject claims \
                 {}. First reached-but-unclaimed {:?}; first claimed-but-unreached {:?}",
                reached.len(),
                claimed.len(),
                first_missing,
                first_surplus
            ),
        );
    }

    // 8. bounded resource accounting against the profile's own limits.
    let tensors = p.outcome.get("tensors");
    if tensors as u64 > p.profile.max_tensors {
        return refuse(
            Code::QUOTA_EXHAUSTED,
            format!(
                "{tensors} tensors over the profile's max_tensors {}",
                p.profile.max_tensors
            ),
        );
    }
    let moved = p.outcome.get("bytes_written") + p.outcome.get("bytes_inherited");
    if moved as u64 > p.profile.max_source_bytes {
        return refuse(
            Code::QUOTA_EXHAUSTED,
            format!(
                "{moved} B moved over the profile's max_source_bytes {}",
                p.profile.max_source_bytes
            ),
        );
    }

    // 9. the golden-suite result. A converter whose golden fixtures did not pass mints
    //    nothing, and "absent" is not "pass".
    if ![GOLDEN_PASS, GOLDEN_FAIL, GOLDEN_ABSENT].contains(&p.golden) {
        return refuse(
            Code::UNKNOWN_FIELD,
            format!(
                "golden {:?} is outside the closed verdict set: an unrecognised verdict is \
                 not a lenient one",
                p.golden
            ),
        );
    }
    if p.golden != GOLDEN_PASS {
        return refuse(
            Code::GOLDEN_MISMATCH,
            format!(
                "golden suite {:?}: only {GOLDEN_PASS:?} mints a receipt",
                p.golden
            ),
        );
    }

    // 10. the ONE evidence arm, and the rule that keeps it honest: evidence must NAME BYTES
    //     THIS STORE HOLDS. A source's own `__metadata__` restating "quantized by X" is
    //     header-invisible and cannot mint anything — the arm has to resolve to an object.
    match &p.evidence {
        Evidence::ApprovedProducer {
            implementation,
            recipe,
        } => {
            if *implementation != p.profile.converter_build {
                return Err(disagree(
                    "evidence.implementation",
                    implementation,
                    &p.profile.converter_build,
                ));
            }
            if *recipe != p.profile.converter_recipe {
                return Err(disagree(
                    "evidence.recipe",
                    &recipe.id(),
                    &p.profile.converter_recipe.id(),
                ));
            }
            require_resident(store, recipe, "approved-producer recipe")?;
        }
        Evidence::IndependentNumerical { report, .. } => {
            require_resident(store, report, "independent-numerical report")?;
        }
    }

    let stamp = Stamp {
        subject: p.subject.object_ref(),
        profile: p.profile_ref.clone(),
        header: p.outcome.header_ref.clone(),
        manifest: p.outcome.manifest_ref.clone(),
        tensor_schema_digest,
        converter_build: p.profile.converter_build.clone(),
        converter_recipe: p.profile.converter_recipe.clone(),
        candidate_set: reached_set,
        golden: p.golden.to_string(),
        isolation_tier: p.isolation_tier,
        sources: p.subject.sources.clone(),
        accounting: p.outcome.obs.clone(),
    };
    let receipt = IngestVerificationReceipt {
        subject: stamp.subject.clone(),
        manifest: stamp.manifest.clone(),
        header: stamp.header.clone(),
        stamp: stamp.object_ref(),
        evidence: p.evidence,
        signer: p.signer.to_string(),
        signature: p.signature.to_string(),
        observations: p.outcome.obs.clone(),
    };
    Ok(Stamped { stamp, receipt })
}

/// The store must actually hold the bytes an evidence arm names, at the exact length it
/// declares. A digest is a claim until something resolves it.
fn require_resident(store: &Store, r: &ObjectRef, what: &str) -> Result<()> {
    match ck::read_object(store, r, limits::DOC_MAX_BYTES) {
        Ok(b) if b.len() as u64 == r.length => Ok(()),
        Ok(b) => refuse(
            Code::EVIDENCE_NOT_RESIDENT,
            format!(
                "{what} {} resolves to {} B, the arm declares {}",
                r.id(),
                b.len(),
                r.length
            ),
        ),
        Err(e) => refuse(
            Code::EVIDENCE_NOT_RESIDENT,
            format!(
                "{what} {} is not resident in this store ({}) — an evidence arm names bytes, \
                 never a restated source claim",
                r.id(),
                e.code.as_str()
            ),
        ),
    }
}

//! The complete durable catalog format: one mutable repository document points at immutable
//! manifests, which point at immutable blobs.  Fixed namespace locations provide the types;
//! neither stored JSON shape carries a format tag.

use std::cmp::Ordering;

use crate::canon::{as_arr, Fields, Value};
use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::{ascii_name, hex64, prefixed, Doc, ObjectRef};

pub const MAX_REPOSITORY_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_RELEASES: usize = 4096;
pub const MAX_EVIDENCE_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepositoryName {
    pub org: String,
    pub name: String,
}

impl RepositoryName {
    pub fn new(org: impl Into<String>, name: impl Into<String>) -> Result<Self> {
        let value = Self {
            org: org.into(),
            name: name.into(),
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<()> {
        ascii_name("Repository.org", &self.org, 128)?;
        ascii_name("Repository.name", &self.name, 128)?;
        if self.org.contains('/')
            || self.name.contains('/')
            || matches!(self.org.as_str(), "." | "..")
            || matches!(self.name.as_str(), "." | "..")
        {
            return refuse(
                Code::KEY_GRAMMAR,
                "repository org/name may not contain '/' or be a dot component",
            );
        }
        Ok(())
    }

    fn from_value(v: &Value) -> Result<Self> {
        let mut fields = Fields::new("Repository.repo", v)?;
        let name = fields.req_str("name")?.to_string();
        let org = fields.req_str("org")?.to_string();
        fields.done()?;
        Self::new(org, name)
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            ("name", Value::str(self.name.clone())),
            ("org", Value::str(self.org.clone())),
        ])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestProvenance {
    pub source_dialect: String,
    pub converter_build: String,
    pub converter_recipe: ObjectRef,
    pub sources: Vec<(String, u64)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationEvidence {
    ApprovedProducer {
        implementation: String,
        recipe: ObjectRef,
    },
    IndependentNumerical {
        verifier_build: String,
        report: ObjectRef,
    },
}

/// Rust always derives classification from the exact header and closure. When a release
/// came through the ingest border, its bounded provenance and one receipt verdict ride here
/// instead of becoming separately stored documents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseEvidence {
    pub classification_digest: String,
    pub ingest: Option<IngestProvenance>,
    pub verification: Option<VerificationEvidence>,
}

impl ReleaseEvidence {
    pub fn required_files(&self) -> Result<Vec<(&'static str, ObjectRef)>> {
        let Some(ingest) = &self.ingest else {
            return Ok(Vec::new());
        };
        let mut files = vec![("evidence/converter-recipe", ingest.converter_recipe.clone())];
        match self.verification.as_ref().expect("validated evidence pair") {
            VerificationEvidence::ApprovedProducer { recipe, .. } => {
                if recipe != &ingest.converter_recipe {
                    return refuse(
                        Code::OBJECT_ID_MISMATCH,
                        "approved-producer recipe differs from ingest converter recipe",
                    );
                }
            }
            VerificationEvidence::IndependentNumerical { report, .. } => {
                files.push(("evidence/verification-report", report.clone()));
            }
        }
        Ok(files)
    }

    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let value = crate::canon::parse_canonical(bytes, MAX_EVIDENCE_BYTES)?;
        Self::from_value(&value)
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        crate::canon::write(&self.to_value())
    }

    fn from_value(v: &Value) -> Result<Self> {
        let mut fields = Fields::new("Release.evidence", v)?;
        let classification_digest = prefixed(
            "Release.evidence.classification_digest",
            fields.req_str("classification_digest")?,
        )?;
        let ingest = fields
            .opt("ingest")
            .map(|value| {
                let mut ingest = Fields::new("Release.evidence.ingest", value)?;
                let converter_build = prefixed(
                    "Release.evidence.ingest.converter_build",
                    ingest.req_str("converter_build")?,
                )?;
                let converter_recipe = ObjectRef::from_value(
                    "Release.evidence.ingest.converter_recipe",
                    ingest.req("converter_recipe")?,
                )?;
                let source_dialect = ingest.req_str("source_dialect")?.to_string();
                ascii_name(
                    "Release.evidence.ingest.source_dialect",
                    &source_dialect,
                    128,
                )?;
                let mut sources = Vec::new();
                for value in as_arr("Release.evidence.ingest", "sources", ingest.req("sources")?)? {
                    let mut source = Fields::new("Release.evidence.ingest.source", value)?;
                    let generation = source.req_uint("generation")?;
                    let digest = prefixed(
                        "Release.evidence.ingest.source.digest",
                        source.req_str("digest")?,
                    )?;
                    source.done()?;
                    sources.push((digest, generation));
                }
                ingest.done()?;
                if sources.is_empty() || sources.len() > 256 {
                    return refuse(
                        Code::COUNT_CAP,
                        "ingest provenance source count is outside 1..=256",
                    );
                }
                if sources.windows(2).any(|pair| pair[0] >= pair[1]) {
                    return refuse(
                        Code::SORT_ORDER,
                        "ingest provenance sources must be sorted and unique",
                    );
                }
                Ok(IngestProvenance {
                    source_dialect,
                    converter_build,
                    converter_recipe,
                    sources,
                })
            })
            .transpose()?;
        let verification = fields
            .opt("verification")
            .map(|value| {
                let mut verification = Fields::new("Release.evidence.verification", value)?;
                let kind = verification.req_str("kind")?;
                let result = match kind {
                    "approved_producer" => VerificationEvidence::ApprovedProducer {
                        implementation: prefixed(
                            "Release.evidence.verification.implementation",
                            verification.req_str("implementation")?,
                        )?,
                        recipe: ObjectRef::from_value(
                            "Release.evidence.verification.recipe",
                            verification.req("recipe")?,
                        )?,
                    },
                    "independent_numerical" => VerificationEvidence::IndependentNumerical {
                        verifier_build: prefixed(
                            "Release.evidence.verification.verifier_build",
                            verification.req_str("verifier_build")?,
                        )?,
                        report: ObjectRef::from_value(
                            "Release.evidence.verification.report",
                            verification.req("report")?,
                        )?,
                    },
                    _ => {
                        return refuse(Code::UNKNOWN_FIELD, "release verification kind is unknown")
                    }
                };
                verification.done()?;
                Ok(result)
            })
            .transpose()?;
        if ingest.is_some() != verification.is_some() {
            return refuse(
                Code::MISSING_FIELD,
                "ingest provenance and verification must be present together",
            );
        }
        fields.done()?;
        let evidence = Self {
            classification_digest,
            ingest,
            verification,
        };
        if crate::canon::write(&evidence.to_value()).len() > MAX_EVIDENCE_BYTES {
            return refuse(Code::SIZE_CAP, "release evidence exceeds 64 KiB");
        }
        Ok(evidence)
    }

    pub fn to_value(&self) -> Value {
        let mut fields = vec![(
            "classification_digest",
            Value::str(self.classification_digest.clone()),
        )];
        if let Some(ingest) = &self.ingest {
            fields.push((
                "ingest",
                Value::obj(vec![
                    (
                        "converter_build",
                        Value::str(ingest.converter_build.clone()),
                    ),
                    ("converter_recipe", ingest.converter_recipe.to_value()),
                    ("source_dialect", Value::str(ingest.source_dialect.clone())),
                    (
                        "sources",
                        Value::arr(
                            ingest
                                .sources
                                .iter()
                                .map(|(digest, generation)| {
                                    Value::obj(vec![
                                        ("digest", Value::str(digest.clone())),
                                        ("generation", Value::uint(*generation)),
                                    ])
                                })
                                .collect(),
                        ),
                    ),
                ]),
            ));
        }
        if let Some(verification) = &self.verification {
            fields.push((
                "verification",
                match verification {
                    VerificationEvidence::ApprovedProducer {
                        implementation,
                        recipe,
                    } => Value::obj(vec![
                        ("implementation", Value::str(implementation.clone())),
                        ("kind", Value::str("approved_producer")),
                        ("recipe", recipe.to_value()),
                    ]),
                    VerificationEvidence::IndependentNumerical {
                        verifier_build,
                        report,
                    } => Value::obj(vec![
                        ("kind", Value::str("independent_numerical")),
                        ("report", report.to_value()),
                        ("verifier_build", Value::str(verifier_build.clone())),
                    ]),
                },
            ));
        }
        Value::obj(fields)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Release {
    pub evidence: ReleaseEvidence,
    pub lane: String,
    pub manifest: ObjectRef,
    pub version: String,
}

impl Release {
    fn key(&self) -> (&str, &str) {
        (&self.version, &self.lane)
    }

    fn validate(&self) -> Result<()> {
        ascii_name("Release.version", &self.version, 128)?;
        ascii_name("Release.lane", &self.lane, 128)?;
        hex64("Release.manifest", &self.manifest.sha256)?;
        if self.manifest.length == 0 {
            return refuse(Code::LENGTH_MISMATCH, "release manifest has zero length");
        }
        // Round through the strict evidence reader so programmatic construction has the
        // same caps as stored bytes.
        ReleaseEvidence::from_value(&self.evidence.to_value())?;
        Ok(())
    }

    fn from_value(v: &Value) -> Result<Self> {
        let mut fields = Fields::new("Repository.release", v)?;
        let evidence = ReleaseEvidence::from_value(fields.req("evidence")?)?;
        let lane = fields.req_str("lane")?.to_string();
        let manifest =
            ObjectRef::from_value("Repository.release.manifest", fields.req("manifest")?)?;
        let version = fields.req_str("version")?.to_string();
        fields.done()?;
        let release = Self {
            evidence,
            lane,
            manifest,
            version,
        };
        release.validate()?;
        Ok(release)
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            ("evidence", self.evidence.to_value()),
            ("lane", Value::str(self.lane.clone())),
            ("manifest", self.manifest.to_value()),
            ("version", Value::str(self.version.clone())),
        ])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Repository {
    pub releases: Vec<Release>,
    pub repo: RepositoryName,
}

impl Repository {
    pub fn empty(repo: RepositoryName) -> Self {
        Self {
            releases: Vec::new(),
            repo,
        }
    }

    fn validate(&self) -> Result<()> {
        self.repo.validate()?;
        if self.releases.len() > MAX_RELEASES {
            return refuse(Code::REPOSITORY_FULL, "repository exceeds 4,096 releases");
        }
        let mut previous: Option<(&str, &str)> = None;
        for release in &self.releases {
            release.validate()?;
            if previous.is_some_and(|key| key >= release.key()) {
                return refuse(
                    Code::SORT_ORDER,
                    "repository releases must be sorted and unique by (version,lane)",
                );
            }
            previous = Some(release.key());
        }
        if self.canonical_bytes().len() > MAX_REPOSITORY_BYTES {
            return refuse(Code::REPOSITORY_FULL, "repository exceeds 8 MiB");
        }
        Ok(())
    }

    pub fn put_release(&self, release: Release) -> Result<Self> {
        release.validate()?;
        let mut next = self.clone();
        match next
            .releases
            .binary_search_by(|candidate| candidate.key().cmp(&release.key()))
        {
            Ok(index) if next.releases[index] == release => return Ok(next),
            Ok(_) => {
                return refuse(
                    Code::RELEASE_CONFLICT,
                    format!(
                        "release ({},{}) already exists with different facts",
                        release.version, release.lane
                    ),
                )
            }
            Err(index) => next.releases.insert(index, release),
        }
        next.validate()?;
        Ok(next)
    }

    pub fn remove_release(&self, version: &str, lane: &str) -> Result<Self> {
        let mut next = self.clone();
        let key = (version, lane);
        let index = next
            .releases
            .binary_search_by(|candidate| candidate.key().cmp(&key))
            .map_err(|_| crate::err::Refusal {
                code: Code::RELEASE_ABSENT,
                detail: format!("release ({version},{lane}) does not exist"),
            })?;
        next.releases.remove(index);
        next.validate()?;
        Ok(next)
    }

    pub fn document_sha256(&self) -> String {
        crate::sha256::hex_digest(&self.canonical_bytes())
    }
}

impl Doc for Repository {
    // Location supplies the type. This constant is intentionally not a stored format id.
    const FORMAT: &'static str = "";
    const MAX_BYTES: usize = MAX_REPOSITORY_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut fields = Fields::new("Repository", v)?;
        let raw = as_arr("Repository", "releases", fields.req("releases")?)?;
        if raw.len() > MAX_RELEASES {
            return refuse(Code::REPOSITORY_FULL, "repository exceeds 4,096 releases");
        }
        let mut releases = Vec::with_capacity(raw.len());
        for value in raw {
            releases.push(Release::from_value(value)?);
        }
        let repo = RepositoryName::from_value(fields.req("repo")?)?;
        fields.done()?;
        let repository = Self { releases, repo };
        repository.validate()?;
        Ok(repository)
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            (
                "releases",
                Value::arr(self.releases.iter().map(Release::to_value).collect()),
            ),
            ("repo", self.repo.to_value()),
        ])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mutation {
    PutRelease {
        repo: RepositoryName,
        lane: String,
        manifest: ObjectRef,
        version: String,
        evidence: Box<ReleaseEvidence>,
    },
    RemoveRelease {
        repo: RepositoryName,
        version: String,
        lane: String,
    },
    DeleteRepository {
        repo: RepositoryName,
    },
}

impl Mutation {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let value = crate::canon::parse_canonical(bytes, MAX_EVIDENCE_BYTES + 4096)?;
        let mut fields = Fields::new("RepositoryMutation", &value)?;
        let action = fields.req_str("action")?.to_string();
        let result = match action.as_str() {
            "put_release" => {
                let evidence = Box::new(ReleaseEvidence::parse(&crate::b64::decode(
                    fields.req_str("evidence_base64")?,
                )?)?);
                let lane = fields.req_str("lane")?.to_string();
                let manifest =
                    ObjectRef::from_value("RepositoryMutation.manifest", fields.req("manifest")?)?;
                let repo = RepositoryName::from_value(fields.req("repo")?)?;
                let version = fields.req_str("version")?.to_string();
                Mutation::PutRelease {
                    repo,
                    lane,
                    manifest,
                    version,
                    evidence,
                }
            }
            "remove_release" => Mutation::RemoveRelease {
                lane: fields.req_str("lane")?.to_string(),
                repo: RepositoryName::from_value(fields.req("repo")?)?,
                version: fields.req_str("version")?.to_string(),
            },
            "delete_repository" => Mutation::DeleteRepository {
                repo: RepositoryName::from_value(fields.req("repo")?)?,
            },
            _ => {
                return refuse(
                    Code::UNKNOWN_FIELD,
                    format!("unknown repository action {action:?}"),
                )
            }
        };
        fields.done()?;
        Ok(result)
    }

    pub fn repo(&self) -> &RepositoryName {
        match self {
            Mutation::PutRelease { repo, .. }
            | Mutation::RemoveRelease { repo, .. }
            | Mutation::DeleteRepository { repo } => repo,
        }
    }

    pub fn put_ingested(
        repo: RepositoryName,
        lane: impl Into<String>,
        manifest: ObjectRef,
        version: impl Into<String>,
        classification_digest: impl Into<String>,
        ingest: IngestProvenance,
        verification: VerificationEvidence,
    ) -> Self {
        Self::PutRelease {
            repo,
            lane: lane.into(),
            manifest,
            version: version.into(),
            evidence: Box::new(ReleaseEvidence {
                classification_digest: classification_digest.into(),
                ingest: Some(ingest),
                verification: Some(verification),
            }),
        }
    }
}

/// Apply against the exact bytes the caller observed. The returned `None` means conditional
/// repository deletion; `Some` is the exact replacement body. The backend supplies the
/// observed bytes/ETag as its CAS precondition, never an embedded generation number.
pub fn apply(
    current: Option<&[u8]>,
    mutation: &Mutation,
    classification_digest: Option<&str>,
) -> Result<Option<Repository>> {
    let existing = current.map(Repository::parse).transpose()?;
    if existing
        .as_ref()
        .is_some_and(|repo| &repo.repo != mutation.repo())
    {
        return refuse(
            Code::REPOSITORY_CONFLICT,
            "mutation repository differs from observed bytes",
        );
    }
    match mutation {
        Mutation::PutRelease {
            repo,
            lane,
            manifest,
            version,
            evidence,
        } => {
            let classification_digest = classification_digest.ok_or_else(|| Refusal {
                code: Code::MISSING_FIELD,
                detail: "put_release requires the header-derived topology digest".into(),
            })?;
            let derived = prefixed("classification_digest", classification_digest)?;
            if evidence.classification_digest != derived {
                return refuse(
                    Code::TOPOLOGY_MISMATCH,
                    format!(
                        "release evidence classification {} differs from derived {derived}",
                        evidence.classification_digest
                    ),
                );
            }
            Ok(Some(
                existing
                    .unwrap_or_else(|| Repository::empty(repo.clone()))
                    .put_release(Release {
                        evidence: (**evidence).clone(),
                        lane: lane.clone(),
                        manifest: manifest.clone(),
                        version: version.clone(),
                    })?,
            ))
        }
        Mutation::RemoveRelease { version, lane, .. } => {
            let repository = existing.ok_or_else(|| crate::err::Refusal {
                code: Code::REPOSITORY_ABSENT,
                detail: "cannot remove a release from an absent repository".into(),
            })?;
            Ok(Some(repository.remove_release(version, lane)?))
        }
        Mutation::DeleteRepository { .. } => {
            if existing.is_none() {
                return refuse(Code::REPOSITORY_ABSENT, "repository is already absent");
            }
            Ok(None)
        }
    }
}

pub fn compare_release_key(left: &Release, right: &Release) -> Ordering {
    left.key().cmp(&right.key())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(seed: u8) -> ReleaseEvidence {
        ReleaseEvidence {
            classification_digest: format!("sha256:{}", format!("{seed:02x}").repeat(32)),
            ingest: None,
            verification: None,
        }
    }

    fn release(version: &str, lane: &str, seed: u8) -> Release {
        Release {
            evidence: evidence(seed),
            lane: lane.into(),
            manifest: ObjectRef {
                sha256: format!("{:02x}", seed + 1).repeat(32),
                length: 12,
            },
            version: version.into(),
        }
    }

    #[test]
    fn location_typed_repository_is_canonical_and_idempotent() {
        let name = RepositoryName::new("org", "model").unwrap();
        let one = Repository::empty(name.clone())
            .put_release(release("1.0.0", "X", 1))
            .unwrap();
        assert!(!String::from_utf8(one.canonical_bytes())
            .unwrap()
            .contains("format"));
        assert_eq!(Repository::parse(&one.canonical_bytes()).unwrap(), one);
        assert_eq!(one.put_release(release("1.0.0", "X", 1)).unwrap(), one);
        assert_eq!(
            one.put_release(release("1.0.0", "X", 2)).unwrap_err().code,
            Code::RELEASE_CONFLICT
        );
    }

    #[test]
    fn evidence_digests_require_the_sha256_prefix_and_repo_names_reject_dot_components() {
        assert_eq!(
            RepositoryName::new(".", "model").unwrap_err().code,
            Code::KEY_GRAMMAR
        );
        let good = Repository::empty(RepositoryName::new("org", "model").unwrap())
            .put_release(release("1.0.0", "X", 1))
            .unwrap()
            .canonical_bytes();
        let bad = String::from_utf8(good)
            .unwrap()
            .replace("sha256:0101", "0101010101");
        assert_eq!(
            Repository::parse(bad.as_bytes()).unwrap_err().code,
            Code::MALFORMED_DIGEST
        );
    }

    #[test]
    fn repository_and_evidence_caps_are_enforced_at_the_exact_writer() {
        let repo = RepositoryName::new("org", "model").unwrap();
        let releases: Vec<Release> = (0..MAX_RELEASES)
            .map(|index| release(&format!("{index:04}"), "X", (index % 200) as u8))
            .collect();
        Repository {
            releases: releases.clone(),
            repo: repo.clone(),
        }
        .validate()
        .unwrap();
        let mut too_many = releases;
        too_many.push(release("zzzz", "X", 201));
        assert_eq!(
            Repository {
                releases: too_many,
                repo: repo.clone(),
            }
            .validate()
            .unwrap_err()
            .code,
            Code::REPOSITORY_FULL
        );

        assert!(evidence(1).to_value().kind() == "object");
        assert!(
            Repository {
                releases: (0..MAX_RELEASES)
                    .map(|index| {
                        let mut value = release(
                            &format!("{index:04}{}", "v".repeat(124)),
                            &"l".repeat(128),
                            (index % 200) as u8,
                        );
                        value.version.truncate(128);
                        value
                    })
                    .collect(),
                repo,
            }
            .canonical_bytes()
            .len()
                < MAX_REPOSITORY_BYTES
        );
    }
}

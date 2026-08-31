//! The complete durable catalog format: one mutable repository document points at immutable
//! manifests, which point at immutable blobs.  Fixed namespace locations provide the types;
//! neither stored JSON shape carries a format tag.

use std::cmp::Ordering;

use crate::canon::{as_arr, as_str, Fields, Value};
use crate::err::{refuse, Code, Result};
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
    /// Canonical credential-free origin, for example hf://org/repo@<commit> or
    /// civitai://<version>. Never a signed/access URL or a local path.
    pub source_uri: Option<String>,
    /// Upstream declaration retained as evidence, never a TensorFS eligibility verdict.
    pub declared_license: Option<String>,
}

fn bounded_fact(what: &str, value: &str, max: usize) -> Result<()> {
    if value.is_empty()
        || value.len() > max
        || value.chars().any(|character| character.is_control())
    {
        return refuse(
            Code::SIZE_CAP,
            format!("{what} must be 1..={max} non-control UTF-8 bytes"),
        );
    }
    Ok(())
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
                let source_uri = ingest
                    .opt("source_uri")
                    .map(|value| {
                        as_str("Release.evidence.ingest", "source_uri", value).map(str::to_string)
                    })
                    .transpose()?;
                if let Some(value) = &source_uri {
                    bounded_fact("Release.evidence.ingest.source_uri", value, 2048)?;
                    if value.contains('?') || value.contains('#') {
                        return refuse(
                            Code::KEY_GRAMMAR,
                            "ingest source_uri must be canonical and query/fragment-free",
                        );
                    }
                }
                let declared_license = ingest
                    .opt("declared_license")
                    .map(|value| {
                        as_str("Release.evidence.ingest", "declared_license", value)
                            .map(str::to_string)
                    })
                    .transpose()?;
                if let Some(value) = &declared_license {
                    bounded_fact("Release.evidence.ingest.declared_license", value, 512)?;
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
                    source_uri,
                    declared_license,
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
            let mut ingest_fields = vec![
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
            ];
            if let Some(value) = &ingest.source_uri {
                ingest_fields.push(("source_uri", Value::str(value.clone())));
            }
            if let Some(value) = &ingest.declared_license {
                ingest_fields.push(("declared_license", Value::str(value.clone())));
            }
            fields.push(("ingest", Value::obj(ingest_fields)));
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
    /// Present only for a CozyTensors model release. Ordinary package, dataset, and user-file
    /// repositories need no fabricated model-classification payload.
    pub evidence: Option<ReleaseEvidence>,
    pub lane: String,
    pub manifest: ObjectRef,
    /// Optional domain metadata stored as an ordinary blob (for example PackageRelease).
    pub metadata: Option<ObjectRef>,
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
        if let Some(metadata) = &self.metadata {
            hex64("Release.metadata", &metadata.sha256)?;
            if metadata.length == 0 {
                return refuse(Code::LENGTH_MISMATCH, "release metadata has zero length");
            }
        }
        if let Some(evidence) = &self.evidence {
            // Round through the strict evidence reader so programmatic construction has the
            // same caps as stored bytes.
            ReleaseEvidence::from_value(&evidence.to_value())?;
        }
        Ok(())
    }

    fn from_value(v: &Value) -> Result<Self> {
        let mut fields = Fields::new("Repository.release", v)?;
        let evidence = fields
            .opt("evidence")
            .map(ReleaseEvidence::from_value)
            .transpose()?;
        let lane = fields.req_str("lane")?.to_string();
        let manifest =
            ObjectRef::from_value("Repository.release.manifest", fields.req("manifest")?)?;
        let metadata = fields
            .opt("metadata")
            .map(|value| ObjectRef::from_value("Repository.release.metadata", value))
            .transpose()?;
        let version = fields.req_str("version")?.to_string();
        fields.done()?;
        let release = Self {
            evidence,
            lane,
            manifest,
            metadata,
            version,
        };
        release.validate()?;
        Ok(release)
    }

    fn to_value(&self) -> Value {
        let mut fields = vec![
            ("lane", Value::str(self.lane.clone())),
            ("manifest", self.manifest.to_value()),
            ("version", Value::str(self.version.clone())),
        ];
        if let Some(evidence) = &self.evidence {
            fields.push(("evidence", evidence.to_value()));
        }
        if let Some(metadata) = &self.metadata {
            fields.push(("metadata", metadata.to_value()));
        }
        Value::obj(fields)
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

    /// Creator's local alias is an ordinary Repository at a reserved location with one
    /// hidden release. Keep this invariant in the canonical reader, not in every client.
    pub fn local_release(&self) -> Result<&Release> {
        if self.repo.org != "local" || self.releases.len() != 1 {
            return refuse(
                Code::ATTACHMENT_CARDINALITY,
                "a local alias must be org local with exactly one release",
            );
        }
        let release = &self.releases[0];
        if release.lane != "local" {
            return refuse(
                Code::KEY_GRAMMAR,
                "a local alias release must use lane local",
            );
        }
        hex64("local source selection", &release.version)?;
        Ok(release)
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
        metadata: Option<ObjectRef>,
        version: String,
        evidence: Option<Box<ReleaseEvidence>>,
    },
    RemoveRelease {
        repo: RepositoryName,
        version: String,
        lane: String,
    },
    /// Device-local mutable alias. The Repository still contains an ordinary immutable
    /// release row, but replacement atomically swaps the complete one-row document.
    ReplaceLocal {
        repo: RepositoryName,
        manifest: ObjectRef,
        version: String,
        evidence: Box<ReleaseEvidence>,
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
                let evidence = fields
                    .opt("evidence_base64")
                    .map(|value| {
                        let encoded =
                            crate::canon::as_str("RepositoryMutation", "evidence_base64", value)?;
                        Ok(Box::new(ReleaseEvidence::parse(&crate::b64::decode(
                            encoded,
                        )?)?))
                    })
                    .transpose()?;
                let lane = fields.req_str("lane")?.to_string();
                let manifest =
                    ObjectRef::from_value("RepositoryMutation.manifest", fields.req("manifest")?)?;
                let metadata = fields
                    .opt("metadata")
                    .map(|value| ObjectRef::from_value("RepositoryMutation.metadata", value))
                    .transpose()?;
                let repo = RepositoryName::from_value(fields.req("repo")?)?;
                let version = fields.req_str("version")?.to_string();
                Mutation::PutRelease {
                    repo,
                    lane,
                    manifest,
                    metadata,
                    version,
                    evidence,
                }
            }
            "remove_release" => Mutation::RemoveRelease {
                lane: fields.req_str("lane")?.to_string(),
                repo: RepositoryName::from_value(fields.req("repo")?)?,
                version: fields.req_str("version")?.to_string(),
            },
            "replace_local" => {
                let encoded = crate::canon::as_str(
                    "RepositoryMutation",
                    "evidence_base64",
                    fields.req("evidence_base64")?,
                )?;
                Mutation::ReplaceLocal {
                    evidence: Box::new(ReleaseEvidence::parse(&crate::b64::decode(encoded)?)?),
                    manifest: ObjectRef::from_value(
                        "RepositoryMutation.manifest",
                        fields.req("manifest")?,
                    )?,
                    repo: RepositoryName::from_value(fields.req("repo")?)?,
                    version: fields.req_str("version")?.to_string(),
                }
            }
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
            | Mutation::ReplaceLocal { repo, .. }
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
            metadata: None,
            version: version.into(),
            evidence: Some(Box::new(ReleaseEvidence {
                classification_digest: classification_digest.into(),
                ingest: Some(ingest),
                verification: Some(verification),
            })),
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
            metadata,
            version,
            evidence,
        } => {
            if repo.org == "local" {
                return refuse(
                    Code::KEY_GRAMMAR,
                    "local repositories require replace_local and exactly one reserved lane",
                );
            }
            match (evidence.as_deref(), classification_digest) {
                (Some(evidence), Some(classification_digest)) => {
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
                }
                (None, None) => {}
                (Some(_), None) => {
                    return refuse(
                        Code::ATTACHMENT_CARDINALITY,
                        "model evidence requires one CozyTensors manifest entry",
                    )
                }
                (None, Some(_)) => {
                    return refuse(
                        Code::MISSING_FIELD,
                        "a CozyTensors release requires model evidence",
                    )
                }
            }
            Ok(Some(
                existing
                    .unwrap_or_else(|| Repository::empty(repo.clone()))
                    .put_release(Release {
                        evidence: evidence.as_deref().cloned(),
                        lane: lane.clone(),
                        manifest: manifest.clone(),
                        metadata: metadata.clone(),
                        version: version.clone(),
                    })?,
            ))
        }
        Mutation::RemoveRelease {
            repo,
            version,
            lane,
        } => {
            if repo.org == "local" {
                return refuse(
                    Code::KEY_GRAMMAR,
                    "local aliases are removed as a complete repository",
                );
            }
            let repository = existing.ok_or_else(|| crate::err::Refusal {
                code: Code::REPOSITORY_ABSENT,
                detail: "cannot remove a release from an absent repository".into(),
            })?;
            Ok(Some(repository.remove_release(version, lane)?))
        }
        Mutation::ReplaceLocal {
            repo,
            manifest,
            version,
            evidence,
        } => {
            if repo.org != "local" {
                return refuse(Code::KEY_GRAMMAR, "replace_local requires org local");
            }
            hex64("local source selection", version)?;
            let derived = prefixed(
                "classification_digest",
                classification_digest.ok_or_else(|| crate::err::Refusal {
                    code: Code::MISSING_FIELD,
                    detail: "a local CozyTensors alias requires model classification".into(),
                })?,
            )?;
            if evidence.classification_digest != derived {
                return refuse(
                    Code::TOPOLOGY_MISMATCH,
                    format!(
                        "local evidence classification {} differs from derived {derived}",
                        evidence.classification_digest
                    ),
                );
            }
            let repository = Repository {
                releases: vec![Release {
                    evidence: Some((**evidence).clone()),
                    lane: "local".into(),
                    manifest: manifest.clone(),
                    metadata: None,
                    version: version.clone(),
                }],
                repo: repo.clone(),
            };
            repository.validate()?;
            Ok(Some(repository))
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
            evidence: Some(evidence(seed)),
            lane: lane.into(),
            manifest: ObjectRef {
                sha256: format!("{:02x}", seed + 1).repeat(32),
                length: 12,
            },
            metadata: None,
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

    #[test]
    fn ingest_origin_is_bounded_credential_free_evidence() {
        let recipe = ObjectRef {
            sha256: "44".repeat(32),
            length: 12,
        };
        let source_uri = format!("hf://org/model@{}", "88".repeat(20));
        let evidence = ReleaseEvidence {
            classification_digest: format!("sha256:{}", "55".repeat(32)),
            ingest: Some(IngestProvenance {
                source_dialect: "safetensors.diffusers".into(),
                converter_build: format!("sha256:{}", "66".repeat(32)),
                converter_recipe: recipe.clone(),
                sources: vec![(format!("sha256:{}", "77".repeat(32)), 1)],
                source_uri: Some(source_uri.clone()),
                declared_license: Some("apache-2.0".into()),
            }),
            verification: Some(VerificationEvidence::ApprovedProducer {
                implementation: format!("sha256:{}", "66".repeat(32)),
                recipe,
            }),
        };
        assert_eq!(
            ReleaseEvidence::parse(&evidence.canonical_bytes()).unwrap(),
            evidence
        );
        let with_query = String::from_utf8(evidence.canonical_bytes())
            .unwrap()
            .replace(&source_uri, &format!("{source_uri}?token=secret"));
        assert_eq!(
            ReleaseEvidence::parse(with_query.as_bytes())
                .unwrap_err()
                .code,
            Code::KEY_GRAMMAR
        );
    }
}

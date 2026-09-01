//! Backend-neutral repository census, projection rebuild, and GC mark planning.
//!
//! A cloud caller stages exact repository and manifest bytes plus only typed CozyTensors
//! headers. `blobs.jsonl` is the complete key/length inventory, so model bodies are never
//! downloaded for GC. The local adapter builds the identical census shape.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::canon::{Fields, Value};
use crate::err::{refuse, Code, Refusal, Result};
use crate::header::{Body, Header};
use crate::ids::{hex64, Doc, ObjectRef};
use crate::manifest::{Entry, Manifest};
use crate::repository::{Release, ReleaseLane, Repository, RepositoryName};

fn io(what: impl AsRef<str>, error: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("{}: {error}", what.as_ref()),
    }
}

pub fn blob_key(sha256: &str) -> Result<String> {
    let digest = hex64("blob key", sha256)?;
    Ok(format!(
        "blobs/{}/{}/{}",
        &digest[..2],
        &digest[2..4],
        digest
    ))
}

pub fn manifest_key(sha256: &str) -> Result<String> {
    let digest = hex64("manifest key", sha256)?;
    Ok(format!(
        "manifests/{}/{}/{}.json",
        &digest[..2],
        &digest[2..4],
        digest
    ))
}

pub fn repo_key(repo: &RepositoryName) -> Result<String> {
    // `new` applies the path-component grammar to programmatically constructed values.
    let checked = RepositoryName::new(repo.org.clone(), repo.name.clone())?;
    Ok(format!("repos/{}/{}.json", checked.org, checked.name))
}

fn digest_from_blob_key(key: &str) -> Result<String> {
    let parts: Vec<&str> = key.split('/').collect();
    if parts.len() != 4 || parts[0] != "blobs" {
        return refuse(Code::KEY_GRAMMAR, format!("invalid blob key {key:?}"));
    }
    let digest = hex64("blob key", parts[3])?;
    if parts[1] != &digest[..2] || parts[2] != &digest[2..4] {
        return refuse(
            Code::PATH_DIGEST_MISMATCH,
            "blob key fanout disagrees with digest",
        );
    }
    Ok(digest)
}

fn digest_from_manifest_key(key: &str) -> Result<String> {
    let parts: Vec<&str> = key.split('/').collect();
    if parts.len() != 4 || parts[0] != "manifests" {
        return refuse(Code::KEY_GRAMMAR, format!("invalid manifest key {key:?}"));
    }
    let Some(stem) = parts[3].strip_suffix(".json") else {
        return refuse(Code::KEY_GRAMMAR, "manifest key must end in .json");
    };
    let digest = hex64("manifest key", stem)?;
    if parts[1] != &digest[..2] || parts[2] != &digest[2..4] {
        return refuse(
            Code::PATH_DIGEST_MISMATCH,
            "manifest key fanout disagrees with digest",
        );
    }
    Ok(digest)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InventoryEntry {
    pub key: String,
    pub length: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeldKey {
    pub key: String,
    pub kind: String,
    pub length: u64,
    pub sha256: String,
}

impl HeldKey {
    pub fn parse_line(bytes: &[u8]) -> Result<Self> {
        let value = crate::canon::parse_canonical(bytes, 16 * 1024)?;
        let mut fields = Fields::new("HeldKey", &value)?;
        let key = fields.req_str("key")?.to_string();
        let kind = fields.req_str("kind")?.to_string();
        let length = fields.req_uint("length")?;
        fields.done()?;
        let sha256 = match kind.as_str() {
            "blob" => digest_from_blob_key(&key)?,
            "manifest" => digest_from_manifest_key(&key)?,
            _ => return refuse(Code::UNKNOWN_FIELD, "hold kind must be blob or manifest"),
        };
        Ok(Self {
            key,
            kind,
            length,
            sha256,
        })
    }
}

impl InventoryEntry {
    pub fn parse_line(bytes: &[u8]) -> Result<Self> {
        let value = crate::canon::parse_canonical(bytes, 16 * 1024)?;
        let mut fields = Fields::new("InventoryEntry", &value)?;
        let key = fields.req_str("key")?.to_string();
        let length = fields.req_uint("length")?;
        fields.done()?;
        let sha256 = digest_from_blob_key(&key)?;
        Ok(Self {
            key,
            length,
            sha256,
        })
    }

    pub fn line(&self) -> Vec<u8> {
        crate::canon::write(&Value::obj(vec![
            ("key", Value::str(self.key.clone())),
            ("length", Value::uint(self.length)),
        ]))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoRow {
    pub document_sha256: String,
    pub name: String,
    pub org: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointRow {
    pub evidence_json: Option<Vec<u8>>,
    pub manifest_length: u64,
    pub manifest_sha256: String,
    pub name: String,
    pub org: String,
    pub published: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseRow {
    pub evidence_json: Option<Vec<u8>>,
    pub header_length: Option<u64>,
    pub header_sha256: Option<String>,
    pub lane: String,
    pub manifest_length: u64,
    pub manifest_sha256: String,
    pub name: String,
    pub object_bytes: u64,
    pub object_count: u64,
    pub org: String,
    pub release_revision: u64,
    pub release_yanked: bool,
    pub topology_digest: Option<String>,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseObjectRow {
    pub lane: String,
    pub length: u64,
    pub name: String,
    pub object_kind: String,
    pub org: String,
    pub sha256: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseEncodingRow {
    pub alias: Option<String>,
    pub encoding_id: String,
    pub lane: String,
    pub name: String,
    pub org: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseComponentRow {
    pub component: String,
    pub lane: String,
    pub name: String,
    pub org: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseConfigRow {
    pub config: String,
    pub lane: String,
    pub length: u64,
    pub media_type: String,
    pub name: String,
    pub org: String,
    pub sha256: String,
    pub version: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Projection {
    pub checkpoints: Vec<CheckpointRow>,
    pub components: Vec<ReleaseComponentRow>,
    pub configs: Vec<ReleaseConfigRow>,
    pub encodings: Vec<ReleaseEncodingRow>,
    pub repos: Vec<RepoRow>,
    pub releases: Vec<ReleaseRow>,
    pub objects: Vec<ReleaseObjectRow>,
}

impl Projection {
    pub fn json_lines(&self) -> Vec<Vec<u8>> {
        let mut lines = Vec::new();
        for row in &self.repos {
            lines.push(crate::canon::write(&Value::obj(vec![
                ("document_sha256", Value::str(row.document_sha256.clone())),
                ("kind", Value::str("repo")),
                ("name", Value::str(row.name.clone())),
                ("org", Value::str(row.org.clone())),
            ])));
        }
        for row in &self.checkpoints {
            let mut fields = vec![
                ("kind", Value::str("checkpoint")),
                ("manifest_length", Value::uint(row.manifest_length)),
                ("manifest_sha256", Value::str(row.manifest_sha256.clone())),
                ("name", Value::str(row.name.clone())),
                ("org", Value::str(row.org.clone())),
                ("published", Value::Bool(row.published)),
            ];
            if let Some(evidence) = &row.evidence_json {
                fields.push((
                    "evidence",
                    crate::canon::parse_canonical(evidence, crate::repository::MAX_EVIDENCE_BYTES)
                        .expect("validated evidence bytes"),
                ));
            }
            lines.push(crate::canon::write(&Value::obj(fields)));
        }
        for row in &self.releases {
            let mut fields = vec![
                ("kind", Value::str("release")),
                ("lane", Value::str(row.lane.clone())),
                ("manifest_length", Value::uint(row.manifest_length)),
                ("manifest_sha256", Value::str(row.manifest_sha256.clone())),
                ("name", Value::str(row.name.clone())),
                ("object_bytes", Value::uint(row.object_bytes)),
                ("object_count", Value::uint(row.object_count)),
                ("org", Value::str(row.org.clone())),
                ("release_revision", Value::uint(row.release_revision)),
                ("release_yanked", Value::Bool(row.release_yanked)),
                ("version", Value::str(row.version.clone())),
            ];
            if let Some(evidence) = &row.evidence_json {
                fields.push((
                    "evidence",
                    crate::canon::parse_canonical(evidence, crate::repository::MAX_EVIDENCE_BYTES)
                        .expect("validated evidence bytes"),
                ));
            }
            if let (Some(sha256), Some(length), Some(topology)) =
                (&row.header_sha256, row.header_length, &row.topology_digest)
            {
                fields.push(("header_length", Value::uint(length)));
                fields.push(("header_sha256", Value::str(sha256.clone())));
                fields.push(("topology_digest", Value::str(topology.clone())));
            }
            lines.push(crate::canon::write(&Value::obj(fields)));
        }
        for row in &self.objects {
            lines.push(crate::canon::write(&Value::obj(vec![
                ("kind", Value::str("release_object")),
                ("lane", Value::str(row.lane.clone())),
                ("length", Value::uint(row.length)),
                ("name", Value::str(row.name.clone())),
                ("object_kind", Value::str(row.object_kind.clone())),
                ("org", Value::str(row.org.clone())),
                ("sha256", Value::str(row.sha256.clone())),
                ("version", Value::str(row.version.clone())),
            ])));
        }
        for row in &self.encodings {
            let mut fields = vec![
                ("encoding_id", Value::str(row.encoding_id.clone())),
                ("kind", Value::str("release_encoding")),
                ("lane", Value::str(row.lane.clone())),
                ("name", Value::str(row.name.clone())),
                ("org", Value::str(row.org.clone())),
                ("version", Value::str(row.version.clone())),
            ];
            if let Some(alias) = &row.alias {
                fields.push(("alias", Value::str(alias.clone())));
            }
            lines.push(crate::canon::write(&Value::obj(fields)));
        }
        for row in &self.components {
            lines.push(crate::canon::write(&Value::obj(vec![
                ("component", Value::str(row.component.clone())),
                ("kind", Value::str("release_component")),
                ("lane", Value::str(row.lane.clone())),
                ("name", Value::str(row.name.clone())),
                ("org", Value::str(row.org.clone())),
                ("version", Value::str(row.version.clone())),
            ])));
        }
        for row in &self.configs {
            lines.push(crate::canon::write(&Value::obj(vec![
                ("config", Value::str(row.config.clone())),
                ("kind", Value::str("release_config")),
                ("lane", Value::str(row.lane.clone())),
                ("length", Value::uint(row.length)),
                ("media_type", Value::str(row.media_type.clone())),
                ("name", Value::str(row.name.clone())),
                ("org", Value::str(row.org.clone())),
                ("sha256", Value::str(row.sha256.clone())),
                ("version", Value::str(row.version.clone())),
            ])));
        }
        lines
    }
}

#[derive(Debug, Clone)]
pub struct Census {
    root: PathBuf,
    repositories: Vec<(Repository, Vec<u8>)>,
    manifests: BTreeMap<String, (Manifest, Vec<u8>)>,
    inventory: BTreeMap<String, InventoryEntry>,
}

#[derive(Debug, Clone)]
pub struct CheckpointFacts {
    pub header: Option<ObjectRef>,
    pub topology_digest: Option<String>,
    pub object_count: u64,
    pub object_bytes: u64,
    pub encodings: Vec<(String, Option<String>)>,
    pub components: Vec<String>,
    pub configs: Vec<(String, String, u64, String)>,
}

impl Census {
    /// Validate one retained checkpoint against its exact staged Manifest, optional
    /// CozyTensors header, and complete blob inventory. Checkpoint retention has no
    /// release label or lane, but it carries the same closed object graph.
    pub fn validate_checkpoint_inputs(
        manifest_ref: &ObjectRef,
        manifest_bytes: &[u8],
        header_bytes: Option<&[u8]>,
        inventory: &BTreeMap<String, InventoryEntry>,
    ) -> Result<CheckpointFacts> {
        if ObjectRef::of(manifest_bytes) != *manifest_ref {
            return refuse(
                Code::OBJECT_ID_MISMATCH,
                "staged manifest bytes disagree with the checkpoint reference",
            );
        }
        let manifest = Manifest::parse(manifest_bytes)?;
        let header = match (manifest.header(), header_bytes) {
            (Some(header_ref), Some(bytes)) => {
                if ObjectRef::of(bytes) != *header_ref {
                    return refuse(
                        Code::OBJECT_ID_MISMATCH,
                        "staged header bytes disagree with the manifest reference",
                    );
                }
                Some(Header::parse(bytes)?)
            }
            (None, None) => None,
            (Some(_), None) => {
                return refuse(
                    Code::MISSING_FIELD,
                    "model checkpoint requires staged header bytes",
                )
            }
            (None, Some(_)) => {
                return refuse(
                    Code::ATTACHMENT_CARDINALITY,
                    "ordinary checkpoint cannot supply CozyTensors header bytes",
                )
            }
        };
        let mut objects: BTreeMap<String, (&'static str, ObjectRef)> = BTreeMap::new();
        for (_, entry) in manifest.entries() {
            insert_ref(
                &mut objects,
                if matches!(entry, Entry::CozyTensors(_)) {
                    "header"
                } else {
                    "file"
                },
                entry.blob(),
            )?;
        }
        if let Some(header) = &header {
            for (_, config) in &header.configs {
                insert_ref(
                    &mut objects,
                    "config",
                    &ObjectRef {
                        sha256: config.sha256.clone(),
                        length: config.length,
                    },
                )?;
            }
            for (_, asset) in &header.assets {
                for object in &asset.segments {
                    insert_ref(&mut objects, "model_asset", object)?;
                }
            }
            for (_, _, tensor) in header.tensors() {
                for (_, part) in &tensor.parts {
                    if let Body::Segments(segments) = &part.body {
                        for object in segments {
                            insert_ref(&mut objects, "part", object)?;
                        }
                    }
                }
            }
        }
        let mut object_bytes = 0u64;
        for (kind, object) in objects.values() {
            if inventory.get(&object.sha256).map(|entry| entry.length) != Some(object.length) {
                return refuse(
                    Code::OBJECT_ABSENT,
                    format!(
                        "{} {} is absent or length-mismatched in the blob inventory",
                        kind,
                        object.id()
                    ),
                );
            }
            object_bytes = object_bytes
                .checked_add(object.length)
                .ok_or_else(|| Refusal {
                    code: Code::ARITH_OVERFLOW,
                    detail: "checkpoint object bytes overflow".into(),
                })?;
        }
        let mut facts = CheckpointFacts {
            header: manifest.header().cloned(),
            topology_digest: header.as_ref().map(Header::topology_digest),
            object_count: objects.len() as u64,
            object_bytes,
            encodings: Vec::new(),
            components: Vec::new(),
            configs: Vec::new(),
        };
        if let Some(header) = header {
            for spec in &header.encodings {
                let encoding_id = spec.object_id();
                let aliases: Vec<String> = crate::registry::seeds()
                    .into_iter()
                    .filter(|seed| seed.spec.object_id() == encoding_id)
                    .map(|seed| seed.alias.to_string())
                    .collect();
                if aliases.is_empty() {
                    facts.encodings.push((encoding_id, None));
                } else {
                    for alias in aliases {
                        facts.encodings.push((encoding_id.clone(), Some(alias)));
                    }
                }
            }
            facts.components = header
                .components
                .iter()
                .map(|(component, _)| component.clone())
                .collect();
            facts.configs = header
                .configs
                .iter()
                .map(|(config, value)| {
                    (
                        config.clone(),
                        value.sha256.clone(),
                        value.length,
                        value.media_type.clone(),
                    )
                })
                .collect();
        }
        Ok(facts)
    }

    /// Build the complete SQL projection for one release from the exact bytes staged by a
    /// publisher. This is the pre-CAS admission seam used by local and cloud publication.
    pub fn project_release_inputs(
        repository: &Repository,
        repository_bytes: &[u8],
        release: &Release,
        lane: &ReleaseLane,
        manifest_bytes: &[u8],
        header_bytes: Option<&[u8]>,
        inventory: BTreeMap<String, InventoryEntry>,
    ) -> Result<Projection> {
        if repository.canonical_bytes() != repository_bytes {
            return refuse(
                Code::NONCANONICAL_ENCODING,
                "repository projection bytes are not the canonical repository",
            );
        }
        if !repository.releases.contains(release) || !release.lanes.contains(lane) {
            return refuse(
                Code::RELEASE_ABSENT,
                "projected release is not present in replacement repository",
            );
        }
        if ObjectRef::of(manifest_bytes) != lane.manifest {
            return refuse(
                Code::OBJECT_ID_MISMATCH,
                "staged manifest bytes disagree with the release reference",
            );
        }
        let manifest = Manifest::parse(manifest_bytes)?;
        match (manifest.header(), header_bytes) {
            (Some(header_ref), Some(bytes)) => {
                if ObjectRef::of(bytes) != *header_ref {
                    return refuse(
                        Code::OBJECT_ID_MISMATCH,
                        "staged header bytes disagree with the manifest reference",
                    );
                }
                Header::parse(bytes)?;
            }
            (None, None) => {}
            (Some(_), None) => {
                return refuse(
                    Code::MISSING_FIELD,
                    "model release requires staged header bytes",
                )
            }
            (None, Some(_)) => {
                return refuse(
                    Code::ATTACHMENT_CARDINALITY,
                    "ordinary release cannot supply CozyTensors header bytes",
                )
            }
        }

        let mut manifests = BTreeMap::new();
        manifests.insert(
            lane.manifest.sha256.clone(),
            (manifest, manifest_bytes.to_vec()),
        );
        let census = Self {
            root: PathBuf::new(),
            repositories: Vec::new(),
            manifests,
            inventory,
        };
        let mut projection = Projection::default();
        projection.repos.push(RepoRow {
            document_sha256: crate::sha256::hex_digest(repository_bytes),
            name: repository.repo.name.clone(),
            org: repository.repo.org.clone(),
        });
        projection
            .checkpoints
            .extend(repository.checkpoints.iter().map(|checkpoint| {
                CheckpointRow {
                    evidence_json: checkpoint
                        .evidence
                        .as_ref()
                        .map(|evidence| crate::canon::write(&evidence.to_value())),
                    manifest_length: checkpoint.manifest.length,
                    manifest_sha256: checkpoint.manifest.sha256.clone(),
                    name: repository.repo.name.clone(),
                    org: repository.repo.org.clone(),
                    published: checkpoint.published,
                }
            }));
        census.project_release_lane_with_header(
            repository,
            release,
            lane,
            &mut projection,
            header_bytes,
        )?;
        Ok(projection)
    }

    pub fn open(root: &Path) -> Result<Self> {
        let repositories = read_repositories(root)?;
        let manifests = read_manifests(root)?;
        let inventory = if root.join("blobs.jsonl").is_file() {
            read_inventory(&root.join("blobs.jsonl"))?
        } else {
            inventory_from_tree(root)?
        };
        Ok(Self {
            root: root.to_path_buf(),
            repositories,
            manifests,
            inventory,
        })
    }

    pub fn open_metadata(root: &Path) -> Result<Self> {
        Ok(Self {
            root: root.to_path_buf(),
            repositories: read_repositories(root)?,
            manifests: read_manifests(root)?,
            inventory: BTreeMap::new(),
        })
    }

    pub fn header_requests(&self) -> Result<Vec<InventoryEntry>> {
        let mut requests = BTreeMap::new();
        for (repository, _) in &self.repositories {
            for checkpoint in &repository.checkpoints {
                let (manifest, _) = self.manifest(&checkpoint.manifest)?;
                if let Some(header) = manifest.header() {
                    insert_ref(&mut requests, "header", header)?;
                }
            }
        }
        requests
            .into_iter()
            .map(|(_, (_, object))| {
                Ok(InventoryEntry {
                    key: blob_key(&object.sha256)?,
                    length: object.length,
                    sha256: object.sha256,
                })
            })
            .collect()
    }

    pub fn projection(&self) -> Result<Projection> {
        let mut projection = Projection::default();
        for (repository, bytes) in &self.repositories {
            projection.repos.push(RepoRow {
                document_sha256: crate::sha256::hex_digest(bytes),
                name: repository.repo.name.clone(),
                org: repository.repo.org.clone(),
            });
            projection
                .checkpoints
                .extend(repository.checkpoints.iter().map(|checkpoint| {
                    CheckpointRow {
                        evidence_json: checkpoint
                            .evidence
                            .as_ref()
                            .map(|evidence| crate::canon::write(&evidence.to_value())),
                        manifest_length: checkpoint.manifest.length,
                        manifest_sha256: checkpoint.manifest.sha256.clone(),
                        name: repository.repo.name.clone(),
                        org: repository.repo.org.clone(),
                        published: checkpoint.published,
                    }
                }));
            for release in &repository.releases {
                self.project_release(repository, release, &mut projection)?;
            }
        }
        Ok(projection)
    }

    pub fn projection_for(&self, org: &str, name: &str) -> Result<Projection> {
        let mut projection = Projection::default();
        let Some((repository, bytes)) = self
            .repositories
            .iter()
            .find(|(repository, _)| repository.repo.org == org && repository.repo.name == name)
        else {
            return Ok(projection);
        };
        projection.repos.push(RepoRow {
            document_sha256: crate::sha256::hex_digest(bytes),
            name: name.to_string(),
            org: org.to_string(),
        });
        projection
            .checkpoints
            .extend(repository.checkpoints.iter().map(|checkpoint| {
                CheckpointRow {
                    evidence_json: checkpoint
                        .evidence
                        .as_ref()
                        .map(|evidence| crate::canon::write(&evidence.to_value())),
                    manifest_length: checkpoint.manifest.length,
                    manifest_sha256: checkpoint.manifest.sha256.clone(),
                    name: repository.repo.name.clone(),
                    org: repository.repo.org.clone(),
                    published: checkpoint.published,
                }
            }));
        for release in &repository.releases {
            self.project_release(repository, release, &mut projection)?;
        }
        Ok(projection)
    }

    pub fn gc_plan(&self, holds: &[HeldKey]) -> Result<Vec<GcDelete>> {
        let projection = self.projection()?;
        let mut live_blobs = BTreeSet::new();
        for object in &projection.objects {
            live_blobs.insert(object.sha256.clone());
        }
        let mut live_manifests = BTreeSet::new();
        for repository in &self.repositories {
            for checkpoint in &repository.0.checkpoints {
                live_manifests.insert(checkpoint.manifest.sha256.clone());
            }
        }
        for hold in holds {
            match hold.kind.as_str() {
                "blob" => {
                    if self
                        .inventory
                        .get(&hold.sha256)
                        .is_some_and(|entry| entry.length != hold.length)
                    {
                        return refuse(
                            Code::LENGTH_MISMATCH,
                            "blob hold length disagrees with census",
                        );
                    }
                    live_blobs.insert(hold.sha256.clone());
                }
                "manifest" => {
                    if self
                        .manifests
                        .get(&hold.sha256)
                        .is_some_and(|(_, bytes)| bytes.len() as u64 != hold.length)
                    {
                        return refuse(
                            Code::LENGTH_MISMATCH,
                            "manifest hold length disagrees with census",
                        );
                    }
                    live_manifests.insert(hold.sha256.clone());
                }
                _ => unreachable!("HeldKey parser closes the kind"),
            }
        }
        for sha256 in &live_manifests {
            let Some((manifest, bytes)) = self.manifests.get(sha256) else {
                // A held manifest may not have reached its final key yet. Its separately
                // held blobs remain protected; there are no manifest bytes to traverse.
                continue;
            };
            let reference = ObjectRef {
                sha256: sha256.clone(),
                length: bytes.len() as u64,
            };
            for object in self.manifest_blob_refs(manifest, &reference)? {
                live_blobs.insert(object.sha256);
            }
        }
        let mut plan = Vec::new();
        for entry in self.inventory.values() {
            if !live_blobs.contains(&entry.sha256) {
                plan.push(GcDelete {
                    key: entry.key.clone(),
                    kind: "blob",
                    length: entry.length,
                    sha256: entry.sha256.clone(),
                });
            }
        }
        for (sha256, (_, bytes)) in &self.manifests {
            if !live_manifests.contains(sha256) {
                plan.push(GcDelete {
                    key: manifest_key(sha256)?,
                    kind: "manifest",
                    length: bytes.len() as u64,
                    sha256: sha256.clone(),
                });
            }
        }
        plan.sort_by(|left, right| left.key.cmp(&right.key));
        Ok(plan)
    }

    fn manifest(&self, reference: &ObjectRef) -> Result<&(Manifest, Vec<u8>)> {
        let found = self
            .manifests
            .get(&reference.sha256)
            .ok_or_else(|| Refusal {
                code: Code::OBJECT_ABSENT,
                detail: format!("manifest {} is absent from the census", reference.id()),
            })?;
        if found.1.len() as u64 != reference.length {
            return refuse(
                Code::LENGTH_MISMATCH,
                format!("manifest {} length differs from release", reference.id()),
            );
        }
        Ok(found)
    }

    fn header(&self, reference: &ObjectRef) -> Result<Header> {
        let path = self.root.join(blob_key(&reference.sha256)?);
        let bytes = fs::read(&path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                Refusal {
                    code: Code::OBJECT_ABSENT,
                    detail: format!("staged CozyTensors header {} is absent", reference.id()),
                }
            } else {
                io(format!("read {}", path.display()), error)
            }
        })?;
        if bytes.len() as u64 != reference.length
            || crate::sha256::hex_digest(&bytes) != reference.sha256
        {
            return refuse(
                Code::OBJECT_ID_MISMATCH,
                format!(
                    "staged CozyTensors header {} disagrees with its reference",
                    reference.id()
                ),
            );
        }
        Header::parse(&bytes)
    }

    fn project_release(
        &self,
        repository: &Repository,
        release: &Release,
        projection: &mut Projection,
    ) -> Result<()> {
        for lane in &release.lanes {
            self.project_release_lane_with_header(repository, release, lane, projection, None)?;
        }
        Ok(())
    }

    fn project_release_lane_with_header(
        &self,
        repository: &Repository,
        release: &Release,
        lane: &ReleaseLane,
        projection: &mut Projection,
        header_bytes: Option<&[u8]>,
    ) -> Result<()> {
        let (manifest, _) = self.manifest(&lane.manifest)?;
        let checkpoint = repository
            .checkpoints
            .iter()
            .find(|checkpoint| checkpoint.manifest == lane.manifest)
            .ok_or_else(|| Refusal {
                code: Code::CHECKPOINT_ABSENT,
                detail: "release checkpoint is absent from repository roots".into(),
            })?;
        let model = match (manifest.header(), checkpoint.evidence.as_ref()) {
            (Some(header_ref), Some(evidence)) => {
                let header = match header_bytes {
                    Some(bytes) => {
                        if ObjectRef::of(bytes) != *header_ref {
                            return refuse(
                                Code::OBJECT_ID_MISMATCH,
                                "staged header bytes disagree with manifest",
                            );
                        }
                        Header::parse(bytes)?
                    }
                    None => self.header(header_ref)?,
                };
                let topology_digest = header.topology_digest();
                if evidence.classification_digest != topology_digest {
                    return refuse(
                        Code::TOPOLOGY_MISMATCH,
                        format!(
                            "release classification {} differs from header topology {topology_digest}",
                            evidence.classification_digest
                        ),
                    );
                }
                for (path, required) in evidence.required_files()? {
                    let found = manifest
                        .entries()
                        .iter()
                        .find(|(candidate, _)| candidate == path);
                    match found {
                        Some((_, Entry::File(actual))) if actual == &required => {}
                        _ => {
                            return refuse(
                                Code::NOT_CONTAINED,
                                format!(
                                "release evidence {} must be the ordinary manifest file {path:?}",
                                required.id()
                            ),
                            )
                        }
                    }
                }
                Some((header_ref, header, topology_digest))
            }
            (None, None) if header_bytes.is_none() => None,
            (Some(_), None) => {
                return refuse(
                    Code::MISSING_FIELD,
                    "a CozyTensors release requires model evidence",
                )
            }
            (None, Some(_)) => {
                return refuse(
                    Code::ATTACHMENT_CARDINALITY,
                    "model evidence requires one CozyTensors manifest entry",
                )
            }
            (None, None) => {
                return refuse(
                    Code::ATTACHMENT_CARDINALITY,
                    "ordinary release cannot supply CozyTensors header bytes",
                )
            }
        };
        let mut objects: BTreeMap<String, (&'static str, ObjectRef)> = BTreeMap::new();
        for (_, entry) in manifest.entries() {
            insert_ref(
                &mut objects,
                if matches!(entry, Entry::CozyTensors(_)) {
                    "header"
                } else {
                    "file"
                },
                entry.blob(),
            )?;
        }
        if let Some(metadata) = &lane.metadata {
            insert_ref(&mut objects, "metadata", metadata)?;
        }
        if let Some((_, header, _)) = &model {
            for (_, config) in &header.configs {
                insert_ref(
                    &mut objects,
                    "config",
                    &ObjectRef {
                        sha256: config.sha256.clone(),
                        length: config.length,
                    },
                )?;
            }
            for (_, asset) in &header.assets {
                for object in &asset.segments {
                    insert_ref(&mut objects, "model_asset", object)?;
                }
            }
            for (_, _, tensor) in header.tensors() {
                for (_, part) in &tensor.parts {
                    if let Body::Segments(segments) = &part.body {
                        for object in segments {
                            insert_ref(&mut objects, "part", object)?;
                        }
                    }
                }
            }
        }
        let mut object_bytes = 0u64;
        for (kind, object) in objects.values() {
            let observed_length = match self.inventory.get(&object.sha256) {
                Some(inventory) => Some(inventory.length),
                None if self.inventory.is_empty() => {
                    fs::metadata(self.root.join(blob_key(&object.sha256)?))
                        .ok()
                        .map(|metadata| metadata.len())
                }
                None => None,
            };
            if observed_length != Some(object.length) {
                return refuse(
                    Code::OBJECT_ABSENT,
                    format!(
                        "{} {} is absent or length-mismatched in the blob inventory",
                        kind,
                        object.id()
                    ),
                );
            }
            object_bytes = object_bytes
                .checked_add(object.length)
                .ok_or_else(|| Refusal {
                    code: Code::ARITH_OVERFLOW,
                    detail: "release object bytes overflow".into(),
                })?;
            projection.objects.push(ReleaseObjectRow {
                lane: lane.lane.clone(),
                length: object.length,
                name: repository.repo.name.clone(),
                object_kind: (*kind).to_string(),
                org: repository.repo.org.clone(),
                sha256: object.sha256.clone(),
                version: release.version.clone(),
            });
        }
        projection.releases.push(ReleaseRow {
            evidence_json: checkpoint
                .evidence
                .as_ref()
                .map(|evidence| crate::canon::write(&evidence.to_value())),
            header_length: model.as_ref().map(|(reference, _, _)| reference.length),
            header_sha256: model
                .as_ref()
                .map(|(reference, _, _)| reference.sha256.clone()),
            lane: lane.lane.clone(),
            manifest_length: lane.manifest.length,
            manifest_sha256: lane.manifest.sha256.clone(),
            name: repository.repo.name.clone(),
            object_bytes,
            object_count: objects.len() as u64,
            org: repository.repo.org.clone(),
            release_revision: release.revision,
            release_yanked: release.yanked,
            topology_digest: model.as_ref().map(|(_, _, digest)| digest.clone()),
            version: release.version.clone(),
        });
        let Some((_, header, _)) = model else {
            return Ok(());
        };
        for spec in &header.encodings {
            let encoding_id = spec.object_id();
            let aliases: Vec<String> = crate::registry::seeds()
                .into_iter()
                .filter(|seed| seed.spec.object_id() == encoding_id)
                .map(|seed| seed.alias.to_string())
                .collect();
            if aliases.is_empty() {
                projection.encodings.push(ReleaseEncodingRow {
                    alias: None,
                    encoding_id,
                    lane: lane.lane.clone(),
                    name: repository.repo.name.clone(),
                    org: repository.repo.org.clone(),
                    version: release.version.clone(),
                });
            } else {
                for alias in aliases {
                    projection.encodings.push(ReleaseEncodingRow {
                        alias: Some(alias),
                        encoding_id: encoding_id.clone(),
                        lane: lane.lane.clone(),
                        name: repository.repo.name.clone(),
                        org: repository.repo.org.clone(),
                        version: release.version.clone(),
                    });
                }
            }
        }
        for (component, _) in &header.components {
            projection.components.push(ReleaseComponentRow {
                component: component.clone(),
                lane: lane.lane.clone(),
                name: repository.repo.name.clone(),
                org: repository.repo.org.clone(),
                version: release.version.clone(),
            });
        }
        for (config, reference) in &header.configs {
            projection.configs.push(ReleaseConfigRow {
                config: config.clone(),
                lane: lane.lane.clone(),
                length: reference.length,
                media_type: reference.media_type.clone(),
                name: repository.repo.name.clone(),
                org: repository.repo.org.clone(),
                sha256: reference.sha256.clone(),
                version: release.version.clone(),
            });
        }
        Ok(())
    }

    fn manifest_blob_refs(
        &self,
        manifest: &Manifest,
        _reference: &ObjectRef,
    ) -> Result<Vec<ObjectRef>> {
        let mut objects = BTreeMap::new();
        for (_, entry) in manifest.entries() {
            insert_ref(&mut objects, "file", entry.blob())?;
        }
        if let Some(header_ref) = manifest.header() {
            let header = self.header(header_ref)?;
            for (_, config) in &header.configs {
                insert_ref(
                    &mut objects,
                    "config",
                    &ObjectRef {
                        sha256: config.sha256.clone(),
                        length: config.length,
                    },
                )?;
            }
            for (_, asset) in &header.assets {
                for object in &asset.segments {
                    insert_ref(&mut objects, "model_asset", object)?;
                }
            }
            for (_, _, tensor) in header.tensors() {
                for (_, part) in &tensor.parts {
                    if let Body::Segments(segments) = &part.body {
                        for object in segments {
                            insert_ref(&mut objects, "part", object)?;
                        }
                    }
                }
            }
        }
        for (kind, object) in objects.values() {
            if self.inventory.get(&object.sha256).map(|entry| entry.length) != Some(object.length) {
                return refuse(
                    Code::OBJECT_ABSENT,
                    format!(
                        "{kind} {} is absent or length-mismatched in inventory",
                        object.id()
                    ),
                );
            }
        }
        Ok(objects.into_values().map(|(_, object)| object).collect())
    }
}

fn insert_ref(
    target: &mut BTreeMap<String, (&'static str, ObjectRef)>,
    kind: &'static str,
    object: &ObjectRef,
) -> Result<()> {
    match target.get(&object.sha256) {
        Some((_, existing)) if existing.length != object.length => refuse(
            Code::LENGTH_MISMATCH,
            format!("{} appears at conflicting lengths", object.id()),
        ),
        Some(_) => Ok(()),
        None => {
            target.insert(object.sha256.clone(), (kind, object.clone()));
            Ok(())
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GcDelete {
    pub key: String,
    pub kind: &'static str,
    pub length: u64,
    pub sha256: String,
}

impl GcDelete {
    pub fn line(&self) -> Vec<u8> {
        crate::canon::write(&Value::obj(vec![
            ("key", Value::str(self.key.clone())),
            ("kind", Value::str(self.kind)),
            ("length", Value::uint(self.length)),
        ]))
    }
}

fn read_repositories(root: &Path) -> Result<Vec<(Repository, Vec<u8>)>> {
    let mut result = Vec::new();
    for org_path in read_dir(&root.join("repos"))? {
        if !org_path.is_dir() {
            return refuse(
                Code::CENSUS_INCOMPLETE,
                "repos/ contains a non-directory org entry",
            );
        }
        let org = org_path.file_name().unwrap().to_string_lossy().to_string();
        for path in read_dir(&org_path)? {
            if !path.is_file() || path.extension().and_then(|value| value.to_str()) != Some("json")
            {
                return refuse(
                    Code::CENSUS_INCOMPLETE,
                    "repository census contains a non-json file",
                );
            }
            let bytes = fs::read(&path).map_err(|error| io("read repository", error))?;
            let repository = <Repository as Doc>::parse(&bytes)?;
            let name = path.file_stem().unwrap().to_string_lossy();
            if repository.repo.org != org || repository.repo.name != name {
                return refuse(
                    Code::PATH_DIGEST_MISMATCH,
                    format!("repository bytes disagree with path {}", path.display()),
                );
            }
            result.push((repository, bytes));
        }
    }
    result.sort_by(|left, right| {
        (&left.0.repo.org, &left.0.repo.name).cmp(&(&right.0.repo.org, &right.0.repo.name))
    });
    Ok(result)
}

fn read_manifests(root: &Path) -> Result<BTreeMap<String, (Manifest, Vec<u8>)>> {
    let mut result = BTreeMap::new();
    let base = root.join("manifests");
    for first in read_dir(&base)? {
        for second in read_dir(&first)? {
            for path in read_dir(&second)? {
                if path.extension().and_then(|value| value.to_str()) != Some("json") {
                    return refuse(
                        Code::CENSUS_INCOMPLETE,
                        "manifest census contains a non-json file",
                    );
                }
                let digest = path.file_stem().unwrap().to_string_lossy().to_string();
                hex64("manifest filename", &digest)?;
                if path.strip_prefix(root).ok().and_then(Path::to_str)
                    != Some(&manifest_key(&digest)?)
                {
                    return refuse(
                        Code::PATH_DIGEST_MISMATCH,
                        "manifest fanout path is not canonical",
                    );
                }
                let bytes = fs::read(&path).map_err(|error| io("read manifest", error))?;
                if crate::sha256::hex_digest(&bytes) != digest {
                    return refuse(
                        Code::OBJECT_ID_MISMATCH,
                        "manifest bytes disagree with filename",
                    );
                }
                let manifest = <Manifest as Doc>::parse(&bytes)?;
                if result.insert(digest, (manifest, bytes)).is_some() {
                    return refuse(Code::CENSUS_INCOMPLETE, "duplicate manifest in census");
                }
            }
        }
    }
    Ok(result)
}

pub fn read_inventory(path: &Path) -> Result<BTreeMap<String, InventoryEntry>> {
    let bytes = fs::read(path).map_err(|error| io(format!("read {}", path.display()), error))?;
    let mut result = BTreeMap::new();
    let mut previous: Option<String> = None;
    for line in bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
    {
        let entry = InventoryEntry::parse_line(line)?;
        if previous.as_ref().is_some_and(|key| key >= &entry.key) {
            return refuse(
                Code::SORT_ORDER,
                "blob inventory must be strictly sorted by key",
            );
        }
        previous = Some(entry.key.clone());
        if result.insert(entry.sha256.clone(), entry).is_some() {
            return refuse(Code::CENSUS_INCOMPLETE, "duplicate blob inventory digest");
        }
    }
    Ok(result)
}

pub fn read_holds(path: &Path) -> Result<Vec<HeldKey>> {
    let bytes = fs::read(path).map_err(|error| io(format!("read {}", path.display()), error))?;
    let mut result = Vec::new();
    let mut previous: Option<String> = None;
    for line in bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
    {
        let hold = HeldKey::parse_line(line)?;
        if previous.as_ref().is_some_and(|key| key >= &hold.key) {
            return refuse(Code::SORT_ORDER, "holds must be strictly sorted by key");
        }
        previous = Some(hold.key.clone());
        result.push(hold);
    }
    Ok(result)
}

fn inventory_from_tree(root: &Path) -> Result<BTreeMap<String, InventoryEntry>> {
    let mut result = BTreeMap::new();
    for first in read_dir(&root.join("blobs"))? {
        for second in read_dir(&first)? {
            for path in read_dir(&second)? {
                if !path.is_file() {
                    return refuse(Code::CENSUS_INCOMPLETE, "blob census contains a non-file");
                }
                let sha256 = path.file_name().unwrap().to_string_lossy().to_string();
                let key = blob_key(&sha256)?;
                if path.strip_prefix(root).ok().and_then(Path::to_str) != Some(&key) {
                    return refuse(
                        Code::PATH_DIGEST_MISMATCH,
                        "blob fanout path is not canonical",
                    );
                }
                let length = fs::metadata(&path)
                    .map_err(|error| io("stat blob", error))?
                    .len();
                let entry = InventoryEntry {
                    key,
                    length,
                    sha256: sha256.clone(),
                };
                if result.insert(sha256, entry).is_some() {
                    return refuse(Code::CENSUS_INCOMPLETE, "duplicate blob in census");
                }
            }
        }
    }
    Ok(result)
}

fn read_dir(path: &Path) -> Result<Vec<PathBuf>> {
    let mut result = Vec::new();
    let entries =
        fs::read_dir(path).map_err(|error| io(format!("read_dir {}", path.display()), error))?;
    for entry in entries {
        result.push(
            entry
                .map_err(|error| io("read directory entry", error))?
                .path(),
        );
    }
    result.sort();
    Ok(result)
}

pub fn write_json_lines(path: &Path, lines: &[Vec<u8>]) -> Result<()> {
    let mut bytes = Vec::new();
    for line in lines {
        bytes.extend_from_slice(line);
        bytes.push(b'\n');
    }
    fs::write(path, bytes).map_err(|error| io(format!("write {}", path.display()), error))
}

pub fn repository_release_lines(root: &Path) -> Result<Vec<Vec<u8>>> {
    let mut lines = Vec::new();
    for (repository, _) in read_repositories(root)? {
        for release in &repository.releases {
            for lane in &release.lanes {
                let checkpoint = repository
                    .checkpoints
                    .iter()
                    .find(|checkpoint| checkpoint.manifest == lane.manifest)
                    .expect("validated repository release checkpoint");
                let mut fields = vec![
                    ("lane", Value::str(lane.lane.clone())),
                    ("manifest_length", Value::uint(lane.manifest.length)),
                    ("manifest_sha256", Value::str(lane.manifest.sha256.clone())),
                    ("name", Value::str(repository.repo.name.clone())),
                    ("org", Value::str(repository.repo.org.clone())),
                    ("release_revision", Value::uint(release.revision)),
                    ("release_yanked", Value::Bool(release.yanked)),
                    ("version", Value::str(release.version.clone())),
                ];
                if let Some(evidence) = &checkpoint.evidence {
                    fields.push(("evidence", evidence.to_value()));
                }
                if let Some(metadata) = &lane.metadata {
                    fields.push(("metadata", metadata.to_value()));
                }
                lines.push(crate::canon::write(&Value::obj(fields)));
            }
        }
    }
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtype;
    use crate::header::{Closure, Header, Part, Tensor};
    use crate::manifest::{Draft, Entry};
    use crate::repository::{
        Checkpoint, CheckpointEvidence, IngestProvenance, Mutation, Release, RepositoryName,
        VerificationEvidence,
    };
    use crate::store::{Fault, Store};

    fn temporary(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tensorfs-{name}-{}-{}",
            std::process::id(),
            crate::meta::now_secs()
        ))
    }

    fn fixture(root: &Path) -> (Store, RepositoryName, ObjectRef) {
        let store = Store::init(root).unwrap();
        let spec = crate::registry::seeds()
            .into_iter()
            .find(|seed| seed.alias == "plain/1")
            .unwrap()
            .spec;
        let tensor = Tensor {
            dtype: Dtype::F32,
            shape: vec![1],
            encoding: spec.object_id(),
            parts: vec![("value".into(), Part::plan(Dtype::F32, vec![1], &[0; 4]))],
        };
        let header = Header {
            configs: Vec::new(),
            assets: Vec::new(),
            encodings: vec![spec],
            components: vec![("model".into(), vec![("weight".into(), tensor)])],
        };
        header.validate(&Closure::default()).unwrap();
        let header_bytes = header.canonical_bytes().unwrap();
        let header_ref = store
            .put_stream(
                &mut header_bytes.as_slice(),
                Some(&ObjectRef::of(&header_bytes)),
                &Fault::default(),
            )
            .unwrap()
            .obj;
        let manifest = Draft {
            entries: vec![("model.cozytensors".into(), Entry::CozyTensors(header_ref))],
        }
        .seal()
        .unwrap();
        let manifest_ref = store.put_manifest(&manifest).unwrap().obj;
        (
            store,
            RepositoryName::new("org", "model").unwrap(),
            manifest_ref,
        )
    }

    fn evidence(store: &Store, manifest: &ObjectRef) -> CheckpointEvidence {
        let manifest = store.read_manifest(manifest).unwrap();
        let header = crate::checkpoint::load_header(store, manifest.header().unwrap()).unwrap();
        CheckpointEvidence {
            classification_digest: header.topology_digest(),
            ingest: None,
            verification: None,
        }
    }

    fn apply_release(
        store: &Store,
        observed: Option<&[u8]>,
        release: &Mutation,
    ) -> Result<Option<Repository>> {
        let Mutation::UpdateRelease { repo, set, .. } = release else {
            panic!("apply_release requires UpdateRelease")
        };
        let manifest = &set.first().expect("release test sets one lane").manifest;
        let checkpointed = store
            .apply_repository(
                observed,
                &Mutation::PutCheckpoint {
                    repo: repo.clone(),
                    manifest: manifest.clone(),
                    evidence: Some(Box::new(evidence(store, manifest))),
                },
                &Fault::default(),
            )?
            .expect("put_checkpoint retains the repository");
        store.apply_repository(
            Some(&checkpointed.canonical_bytes()),
            release,
            &Fault::default(),
        )
    }

    fn release_mutation(
        repo: &RepositoryName,
        version: &str,
        lane: &str,
        manifest: &ObjectRef,
        expected_revision: u64,
    ) -> Mutation {
        Mutation::UpdateRelease {
            expected_revision,
            repo: repo.clone(),
            remove: Vec::new(),
            set: vec![ReleaseLane {
                lane: lane.into(),
                manifest: manifest.clone(),
                metadata: None,
            }],
            version: version.into(),
        }
    }

    #[test]
    fn local_repo_commit_rebuild_and_gc_follow_only_the_repo_graph() {
        let root = temporary("repo-graph");
        let _ = fs::remove_dir_all(&root);
        let (store, repo, manifest) = fixture(&root);
        let put = release_mutation(&repo, "1.0.0", "cpu", &manifest, 0);
        apply_release(&store, None, &put).unwrap();
        let bytes = fs::read(store.repository_path(&repo)).unwrap();
        assert_eq!(
            store
                .apply_repository(Some(&bytes), &put, &Fault::default())
                .unwrap()
                .unwrap()
                .canonical_bytes(),
            bytes
        );
        let connection = rusqlite::Connection::open(root.join("tensorfs.sqlite")).unwrap();
        let releases: u64 = connection
            .query_row(
                "SELECT count(*) FROM tensorfs_released_manifests",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(releases, 1);

        let census = Census::open(&root).unwrap();
        assert!(census.gc_plan(&[]).unwrap().is_empty());
        let delete = Mutation::DeleteRepository { repo: repo.clone() };
        store
            .apply_repository(Some(&bytes), &delete, &Fault::default())
            .unwrap();
        let after_delete = Census::open(&root).unwrap();
        let held = HeldKey {
            key: manifest_key(&manifest.sha256).unwrap(),
            kind: "manifest".into(),
            length: manifest.length,
            sha256: manifest.sha256.clone(),
        };
        assert!(after_delete.gc_plan(&[held]).unwrap().is_empty());
        let plan = after_delete.gc_plan(&[]).unwrap();
        assert!(plan
            .iter()
            .any(|row| row.kind == "manifest" && row.sha256 == manifest.sha256));
        assert!(plan.iter().any(|row| row.kind == "blob"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn same_repo_concurrent_create_has_one_winner_and_one_typed_conflict() {
        let root = temporary("repo-conflict");
        let _ = fs::remove_dir_all(&root);
        let (store, repo, manifest) = fixture(&root);
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
        let mut threads = Vec::new();
        for version in ["1.0.0", "2.0.0"] {
            let store = store.clone();
            let repo = repo.clone();
            let manifest = manifest.clone();
            let barrier = barrier.clone();
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                apply_release(
                    &store,
                    None,
                    &release_mutation(&repo, version, "cpu", &manifest, 0),
                )
            }));
        }
        barrier.wait();
        let results: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(
            results
                .iter()
                .filter_map(|result| result.as_ref().err())
                .filter(|error| error.code == Code::REPOSITORY_CONFLICT)
                .count(),
            1
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn release_projection_requires_exact_manifest_header_and_complete_inventory() {
        let root = temporary("release-inputs");
        let _ = fs::remove_dir_all(&root);
        let (store, repo, manifest_ref) = fixture(&root);
        let manifest_bytes = fs::read(store.manifest_path(&manifest_ref.sha256)).unwrap();
        let manifest = Manifest::parse(&manifest_bytes).unwrap();
        let header_ref = manifest.header().unwrap();
        let header_bytes = fs::read(store.blob_path(&header_ref.sha256)).unwrap();
        let checkpoint_evidence = CheckpointEvidence {
            classification_digest: Header::parse(&header_bytes).unwrap().topology_digest(),
            ingest: None,
            verification: None,
        };
        let lane = ReleaseLane {
            lane: "cpu".into(),
            manifest: manifest_ref.clone(),
            metadata: None,
        };
        let release = Release {
            lanes: vec![lane.clone()],
            revision: 1,
            version: "1.0.0".into(),
            yanked: false,
        };
        let repository = Repository::empty(repo)
            .put_checkpoint(Checkpoint {
                manifest: manifest_ref.clone(),
                evidence: Some(checkpoint_evidence),
                published: false,
                source_selection: None,
            })
            .unwrap()
            .update_release("1.0.0", 0, vec![lane.clone()], Vec::new())
            .unwrap();
        let repository_bytes = repository.canonical_bytes();
        let inventory = inventory_from_tree(&root).unwrap();
        let projection = Census::project_release_inputs(
            &repository,
            &repository_bytes,
            &release,
            &lane,
            &manifest_bytes,
            Some(&header_bytes),
            inventory.clone(),
        )
        .unwrap();
        assert_eq!(projection.releases.len(), 1);
        assert_eq!(projection.objects.len(), 1);
        assert!(!projection.encodings.is_empty());
        assert!(!projection.components.is_empty());

        let recipe_bytes = b"resident evidence outside the manifest";
        let recipe = store
            .put_stream(
                &mut recipe_bytes.as_slice(),
                Some(&ObjectRef::of(recipe_bytes)),
                &Fault::default(),
            )
            .unwrap()
            .obj;
        let evidence_checkpoint = Checkpoint {
            manifest: lane.manifest.clone(),
            evidence: Some(CheckpointEvidence {
                classification_digest: Header::parse(&header_bytes).unwrap().topology_digest(),
                ingest: Some(IngestProvenance {
                    source_dialect: "cozytensors".into(),
                    converter_build: format!("sha256:{}", "22".repeat(32)),
                    converter_recipe: recipe.clone(),
                    sources: vec![(format!("sha256:{}", "33".repeat(32)), 1)],
                    source_uri: None,
                    declared_license: None,
                }),
                verification: Some(VerificationEvidence::ApprovedProducer {
                    implementation: format!("sha256:{}", "22".repeat(32)),
                    recipe,
                }),
            }),
            published: false,
            source_selection: None,
        };
        let evidence_repository = Repository::empty(repository.repo.clone())
            .put_checkpoint(evidence_checkpoint)
            .unwrap()
            .update_release("1.0.0", 0, vec![lane.clone()], Vec::new())
            .unwrap();
        assert_eq!(
            Census::project_release_inputs(
                &evidence_repository,
                &evidence_repository.canonical_bytes(),
                &release,
                &lane,
                &manifest_bytes,
                Some(&header_bytes),
                inventory_from_tree(&root).unwrap(),
            )
            .unwrap_err()
            .code,
            Code::NOT_CONTAINED
        );

        let mut missing_header = inventory;
        missing_header.remove(&header_ref.sha256);
        assert_eq!(
            Census::project_release_inputs(
                &repository,
                &repository_bytes,
                &release,
                &lane,
                &manifest_bytes,
                Some(&header_bytes),
                missing_header,
            )
            .unwrap_err()
            .code,
            Code::OBJECT_ABSENT
        );
        assert_eq!(
            Census::project_release_inputs(
                &repository,
                &repository_bytes,
                &release,
                &lane,
                &manifest_bytes,
                Some(b"not the header"),
                BTreeMap::new(),
            )
            .unwrap_err()
            .code,
            Code::OBJECT_ID_MISMATCH
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn stale_append_cannot_undo_release_or_repository_removal() {
        let root = temporary("remove-ordering");
        let _ = fs::remove_dir_all(&root);
        let (store, repo, manifest) = fixture(&root);
        apply_release(
            &store,
            None,
            &release_mutation(&repo, "1.0.0", "cpu", &manifest, 0),
        )
        .unwrap();
        let before_remove = fs::read(store.repository_path(&repo)).unwrap();
        store
            .apply_repository(
                Some(&before_remove),
                &Mutation::YankRelease {
                    expected_revision: 1,
                    repo: repo.clone(),
                    version: "1.0.0".into(),
                },
                &Fault::default(),
            )
            .unwrap();
        let after_remove = fs::read(store.repository_path(&repo)).unwrap();
        let append = release_mutation(&repo, "2.0.0", "cpu", &manifest, 0);
        assert_eq!(
            store
                .apply_repository(Some(&before_remove), &append, &Fault::default())
                .unwrap_err()
                .code,
            Code::REPOSITORY_CONFLICT
        );
        store
            .apply_repository(
                Some(&after_remove),
                &Mutation::DeleteRepository { repo: repo.clone() },
                &Fault::default(),
            )
            .unwrap();
        assert_eq!(
            store
                .apply_repository(Some(&after_remove), &append, &Fault::default())
                .unwrap_err()
                .code,
            Code::REPOSITORY_CONFLICT
        );
        assert!(!store.repository_path(&repo).exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn ingest_holds_bridge_every_final_key_to_repository_commit_against_gc() {
        let root = temporary("acquisition-gc");
        let _ = fs::remove_dir_all(&root);
        let store = Store::init(&root).unwrap();
        let repo = RepositoryName::new("org", "held-model").unwrap();
        let catalog = crate::catalog::Catalog::open(&root).unwrap();
        let guard = catalog
            .begin_operation("acquire-1", &repo.org, &repo.name, 10)
            .unwrap();
        let spec = crate::registry::seeds()
            .into_iter()
            .find(|seed| seed.alias == "plain/1")
            .unwrap()
            .spec;
        let header = Header {
            configs: Vec::new(),
            assets: Vec::new(),
            encodings: vec![spec.clone()],
            components: vec![(
                "model".into(),
                vec![(
                    "weight".into(),
                    Tensor {
                        dtype: Dtype::F32,
                        shape: vec![1],
                        encoding: spec.object_id(),
                        parts: vec![("value".into(), Part::plan(Dtype::F32, vec![1], &[0; 4]))],
                    },
                )],
            )],
        };
        header.validate(&Closure::default()).unwrap();
        let header_bytes = header.canonical_bytes().unwrap();
        let header_ref = store
            .put_stream_held(
                &mut header_bytes.as_slice(),
                Some(&ObjectRef::of(&header_bytes)),
                &Fault::default(),
                Some("acquire-1"),
            )
            .unwrap()
            .obj;
        let manifest = Draft {
            entries: vec![("model.cozytensors".into(), Entry::CozyTensors(header_ref))],
        }
        .seal()
        .unwrap();
        let manifest_ref = store
            .put_manifest_held(&manifest, Some("acquire-1"))
            .unwrap()
            .obj;

        let root_for_gc = root.clone();
        let gc = std::thread::spawn(move || {
            let catalog = crate::catalog::Catalog::open(&root_for_gc).unwrap();
            let holds = catalog.held_keys().unwrap();
            Census::open(&root_for_gc).unwrap().gc_plan(&holds).unwrap()
        });
        assert!(gc.join().unwrap().is_empty());

        apply_release(
            &store,
            None,
            &release_mutation(&repo, "1.0.0", "cpu", &manifest_ref, 0),
        )
        .unwrap();
        catalog.complete_operation("acquire-1").unwrap();
        drop(guard);
        assert!(catalog.held_keys().unwrap().is_empty());
        assert!(Census::open(&root)
            .unwrap()
            .gc_plan(&[])
            .unwrap()
            .is_empty());
        let _ = fs::remove_dir_all(root);
    }
}

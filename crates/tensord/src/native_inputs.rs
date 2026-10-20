//! Native input custody in the run objects' store (`cas`) and the service's sole journal.
use crate::api::{auth::VerifiedActor, backend::InputTreeReceiver, domain};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{self, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::Arc,
};
use crate::{cas::Cas, trees};
use tensorfs::{
    format::ObjectRef,
    sha256,
};
use protocol::error::Error;

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct IntakeSpec {
    pub actor: String,
    pub request_id: String,
    pub input_id: String,
    pub manifest_sha256: String,
    pub manifest_length: u64,
    pub manifest_canonical_bytes: Vec<u8>,
    pub content_bytes: u64,
    pub retention_id: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct IntakeState {
    pub spec: IntakeSpec,
    pub released: bool,
    pub receipt: Option<Vec<u8>>,
}
/// Engine implements these in its existing SQLite journal. Finish registers the
/// exact NativeByteRetentionRequest in the same transaction as the result receipt.
pub trait IntakeJournal: Send + Sync {
    fn begin_intake(&self, spec: IntakeSpec) -> io::Result<IntakeState>;
    fn finish_intake(
        &self,
        actor: &str,
        retention: &str,
        receipt: Vec<u8>,
    ) -> io::Result<IntakeState>;
    fn abort_intake(&self, actor: &str, retention: &str) -> io::Result<IntakeState>;
}
struct Partial {
    reference: ObjectRef,
    path: PathBuf,
    file: File,
    received: u64,
}
pub struct SourceIntake {
    cas: Cas,
    journal: Arc<dyn IntakeJournal>,
    state: IntakeState,
    manifest: BTreeMap<String, ObjectRef>,
    directory: PathBuf,
    objects: BTreeMap<String, ObjectRef>,
    partials: BTreeMap<String, Partial>,
}
impl Drop for SourceIntake {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

impl SourceIntake {
    pub fn begin(
        cas: Cas,
        journal: Arc<dyn IntakeJournal>,
        staging: &Path,
        workspace: &str,
        actor: VerifiedActor,
        header: domain::InputTreeImportHeader,
    ) -> Result<Box<dyn InputTreeReceiver>, Error> {
        if header.request_id.is_empty()
            || header.request_id.len() > 128
            || !header
                .request_id
                .bytes()
                .all(|b| (0x20..=0x7e).contains(&b))
            || header.input_id.is_empty()
            || header.input_id.len() > 1024
            || !header.input_id.bytes().all(|b| (0x20..=0x7e).contains(&b))
        {
            return Err(Error::validation("invalid_request",
                "input subject is outside its typed identifier bound",
            ));
        }
        let reference = header
            .manifest
            .as_ref()
            .ok_or_else(|| Error::validation("invalid_request", "input manifest reference is absent"))?;
        if reference.digest.len() != 32
            || reference.length != header.manifest_canonical_bytes.len() as u64
            || header.manifest_canonical_bytes.is_empty()
            || header.manifest_canonical_bytes.len() > 1 << 20
            || sha256::digest(&header.manifest_canonical_bytes).as_slice() != reference.digest
        {
            return Err(Error::validation("invalid_request",
                "input manifest bytes do not verify",
            ));
        }
        let manifest = trees::parse(&header.manifest_canonical_bytes)
            .map_err(|_| Error::validation("invalid_request", "input tree must contain ordinary files"))?;
        let content = manifest
            .values()
            .try_fold(0u64, |sum, file| sum.checked_add(file.length))
            .ok_or_else(|| Error::validation("invalid_request", "input content length overflow"))?;
        if content != header.content_bytes {
            return Err(Error::validation("invalid_request",
                "input content count differs from manifest",
            ));
        }
        let mut objects = BTreeMap::new();
        for reference in manifest.values() {
            if objects
                .insert(reference.sha256.clone(), reference.clone())
                .is_some_and(|prior: ObjectRef| prior.length != reference.length)
            {
                return Err(Error::validation("invalid_request",
                    "one input object cannot declare multiple lengths",
                ));
            }
        }
        let actor = sha256::hex(&actor.public_key);
        let subject = serde_json_canonicalizer::to_vec(&serde_json::json!({
            "format": "cozy.machine.input-intake/1", "workspace": workspace, "actor": actor,
            "request": header.request_id, "input": header.input_id,
        }))
        .map_err(|e| Error::internal("custody_corrupt", e.to_string()))?;
        let retention_id = format!("sha256:{}", sha256::hex_digest(&subject));
        let spec = IntakeSpec {
            actor,
            request_id: header.request_id,
            input_id: header.input_id,
            manifest_sha256: sha256::hex(&reference.digest),
            manifest_length: reference.length,
            manifest_canonical_bytes: header.manifest_canonical_bytes,
            content_bytes: header.content_bytes,
            retention_id,
        };
        let state = journal.begin_intake(spec.clone()).map_err(database)?;
        if state.spec != spec {
            return Err(Error::conflict("precondition_failed",
                "input intake changed its immutable subject",
            ));
        }
        fs::create_dir_all(staging).map_err(database)?;
        let directory = staging.join(
            uuid::Uuid::new_v4().to_string(),
        );
        fs::create_dir(&directory).map_err(database)?;
        platform::durable::set_mode(&directory, 0o700).map_err(|error| database(error.into()))?;
        Ok(Box::new(Self {
            cas,
            journal,
            state,
            manifest,
            directory,
            objects,
            partials: BTreeMap::new(),
        }))
    }
    fn result(&self, released: bool) -> Result<domain::NativeByteRetentionResult, Error> {
        let tree = trees::read(&self.cas, &self.state.spec.retention_id).map_err(native)?;
        let source = match tree {
            Some(tree) => Some(domain::NativeByteTreeRef {
                producer_root_id: tree.hold,
                receipt_digest: decode_hex(&tree.manifest.sha256)?,
                manifest: Some(domain::Ref {
                    digest: decode_hex(&tree.manifest.sha256)?,
                    length: tree.manifest.length,
                }),
                content_bytes: self.state.spec.content_bytes,
            }),
            None => None,
        };
        Ok(domain::NativeByteRetentionResult {
            source,
            retention_id: self.state.spec.retention_id.clone(),
            released,
        })
    }
    fn release_native(&self) -> Result<(), Error> {
        trees::release(&self.cas, &self.state.spec.retention_id).map_err(native)?;
        Ok(())
    }
}
impl InputTreeReceiver for SourceIntake {
    fn blob(&mut self, blob: domain::InputTreeImportBlob) -> Result<(), Error> {
        if self.state.released {
            return Err(Error::conflict("precondition_failed",
                "input intake is permanently released",
            ));
        }
        let object = blob
            .object
            .ok_or_else(|| Error::validation("invalid_request", "input object reference absent"))?;
        if object.digest.len() != 32 {
            return Err(Error::validation("invalid_request", "input object digest is invalid"));
        }
        let hash = sha256::hex(&object.digest);
        let expected = self
            .objects
            .get(&hash)
            .ok_or_else(|| Error::validation("invalid_request", "input object is outside declared manifest"))?;
        if object.length != expected.length || blob.data.len() > 1 << 20 {
            return Err(Error::validation("invalid_request",
                "input object identity or chunk bound differs",
            ));
        }
        let partial = match self.partials.entry(hash.clone()) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => {
                let path = self.directory.join(&hash);
                let file = platform::durable::open_nofollow(
                    &path,
                    platform::durable::Mode::mode(OpenOptions::new().create_new(true).read(true).write(true), 0o600),
                )
                .map_err(|error| database(error.into()))?;
                entry.insert(Partial {
                    reference: expected.clone(),
                    path,
                    file,
                    received: 0,
                })
            }
        };
        if blob.offset != partial.received
            || blob.data.len() as u64 > partial.reference.length - partial.received
            || (blob.data.is_empty() && partial.reference.length != 0)
        {
            return Err(Error::conflict("precondition_failed",
                "input object chunks must be contiguous within one stream",
            ));
        }
        partial
            .file
            .seek(SeekFrom::Start(partial.received))
            .map_err(database)?;
        partial.file.write_all(&blob.data).map_err(database)?;
        partial.received += blob.data.len() as u64;
        Ok(())
    }
    fn commit(
        self: Box<Self>,
        commit: domain::InputTreeImportCommit,
    ) -> Result<domain::NativeByteRetentionResult, Error> {
        if commit.abort || self.state.released {
            self.journal
                .abort_intake(&self.state.spec.actor, &self.state.spec.retention_id)
                .map_err(database)?;
            let result = self.result(true)?;
            self.release_native()?;
            return Ok(result);
        }
        for reference in self.objects.values() {
            if self.cas.reuse(reference).map_err(native)? {
                continue;
            }
            let partial = self
                .partials
                .get(&reference.sha256)
                .ok_or_else(|| Error::conflict("precondition_failed", "input object is incomplete"))?;
            if partial.received != reference.length {
                return Err(Error::conflict("precondition_failed", "input object is incomplete"));
            }
            partial.file.sync_all().map_err(database)?;
        }
        let files: Vec<(String, &Path, ObjectRef)> = self
            .manifest
            .iter()
            .map(|(path, file)| {
                let local = self.partials.get(&file.sha256).map_or(self.directory.as_path(), |p| p.path.as_path());
                (path.clone(), local, file.clone())
            })
            .collect();
        trees::admit(&self.cas, &files).map_err(native)?;
        // The tree is the client's own manifest, whose digest the run names.
        let manifest = self.cas.put_bytes(&self.state.spec.manifest_canonical_bytes).map_err(native)?;
        trees::hold(&self.cas, &self.state.spec.retention_id, manifest, &self.manifest).map_err(native)?;
        let result = self.result(false)?;
        let committed = self
            .journal
            .finish_intake(
                &self.state.spec.actor,
                &self.state.spec.retention_id,
                crate::archive::encode_native_byte_retention_result(&result),
            )
            .map_err(database)?;
        if committed.released {
            self.release_native()?;
            return Ok(domain::NativeByteRetentionResult {
                released: true,
                ..result
            });
        }
        if let Some(receipt) = committed.receipt {
            return crate::archive::decode_native_byte_retention_result(receipt.as_slice())
                .map_err(|_| Error::internal("custody_corrupt", "committed intake receipt is invalid"));
        }
        Err(Error::internal("custody_corrupt",
            "journal acknowledged input without durable receipt",
        ))
    }
}
fn decode_hex(spelling: &str) -> Result<Vec<u8>, Error> {
    (0..32)
        .map(|index| {
            u8::from_str_radix(&spelling[index * 2..index * 2 + 2], 16)
                .map_err(|_| Error::internal("custody_corrupt", "native digest is invalid"))
        })
        .collect()
}
fn native(error: impl std::fmt::Display) -> Error {
    Error::conflict("precondition_failed", format!("native input custody: {error}"))
}
fn database(error: io::Error) -> Error {
    match error.kind() {
        io::ErrorKind::AlreadyExists => Error::conflict("already_exists", error.to_string()),
        io::ErrorKind::PermissionDenied => Error::forbidden("capability_scope", error.to_string()),
        io::ErrorKind::InvalidInput => Error::validation("invalid_request", error.to_string()),
        _ => Error::unavailable("unavailable", error.to_string()),
    }
}

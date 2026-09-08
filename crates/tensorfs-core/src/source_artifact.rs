//! Retained ordinary-file trees. Source location is provenance, never content identity.
//! Roots are filesystem authority and independent owners have permanent release tombstones.
use crate::{
    canon::{as_arr, Fields, Value},
    catalog::WriterGuard,
    err::{refuse, Code, Refusal, Result},
    fetch::FetchPlan,
    ids::{prefixed, Doc, ObjectRef, Plain},
    manifest::Manifest,
    storage::HeldKey,
    store::Store,
};
use fs2::FileExt;
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::PathBuf,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeRoot {
    pub owner: String,
    pub producer: String,
    pub manifest: ObjectRef,
    /// GC liveness intent while incomplete; complete roots additionally verify every member.
    pub objects: Vec<ObjectRef>,
    pub complete: bool,
    pub released: bool,
}
impl Plain for TreeRoot {
    const MAX_BYTES: usize = 1 << 20;
    fn from_value(value: &Value) -> Result<Self> {
        let mut f = Fields::new("TreeRoot", value)?;
        let owner = prefixed("tree owner", f.req_str("owner")?)?;
        let producer = prefixed("tree producer", f.req_str("producer")?)?;
        let manifest = ObjectRef::from_value("tree manifest", f.req("manifest")?)?;
        let objects = as_arr("TreeRoot", "objects", f.req("objects")?)?
            .iter()
            .map(|v| ObjectRef::from_value("tree object", v))
            .collect::<Result<Vec<_>>>()?;
        let boolean = |v: &Value| match v {
            Value::Bool(b) => Ok(*b),
            _ => refuse(Code::WRONG_TYPE, "tree state must be boolean"),
        };
        let complete = boolean(f.req("complete")?)?;
        let released = boolean(f.req("released")?)?;
        f.done()?;
        Ok(Self {
            owner,
            producer,
            manifest,
            objects,
            complete,
            released,
        })
    }
    fn to_value(&self) -> Value {
        Value::obj(vec![
            ("owner", Value::str(&self.owner)),
            ("producer", Value::str(&self.producer)),
            ("manifest", self.manifest.to_value()),
            (
                "objects",
                Value::arr(self.objects.iter().map(ObjectRef::to_value).collect()),
            ),
            ("complete", Value::Bool(self.complete)),
            ("released", Value::Bool(self.released)),
        ])
    }
}
impl TreeRoot {
    /// Immutable native receipt. Current custody/release never changes producer history.
    pub fn receipt(&self) -> Result<Vec<u8>> {
        if !self.complete {
            return refuse(Code::TRANSACTION_CLOSED, "incomplete source has no receipt");
        }
        Ok(crate::canon::write(&Value::obj(vec![
            ("format", Value::str("tensorfs.source-result/1")),
            ("producer", Value::str(&self.producer)),
            ("manifest", self.manifest.to_value()),
        ])))
    }
}

fn io(e: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("source artifact: {e}"),
    }
}
fn directory(store: &Store) -> PathBuf {
    store.root().join("roots/trees")
}
fn path(store: &Store, owner: &str) -> Result<PathBuf> {
    prefixed("tree owner", owner)?;
    Ok(directory(store).join(format!("{}.json", &owner[7..])))
}
fn lock(store: &Store) -> Result<File> {
    fs::create_dir_all(directory(store)).map_err(io)?;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(directory(store).join("lock"))
        .map_err(io)?;
    file.lock_exclusive().map_err(io)?;
    Ok(file)
}
pub fn read(store: &Store, owner: &str) -> Result<Option<TreeRoot>> {
    let file = match File::open(path(store, owner)?) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(io(e)),
    };
    let mut bytes = Vec::new();
    file.take(TreeRoot::MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(io)?;
    let root = TreeRoot::parse(&bytes)?;
    if root.owner != owner {
        return refuse(Code::CROSS_SUBJECT_REPLAY, "tree root names another owner");
    }
    Ok(Some(root))
}
fn write(store: &Store, root: &TreeRoot) -> Result<()> {
    let bytes = root.canonical_bytes();
    if bytes.len() > TreeRoot::MAX_BYTES {
        return refuse(Code::SIZE_CAP, "retained tree root exceeds metadata bound");
    }
    let path = path(store, &root.owner)?;
    let tmp = path.with_extension(format!("tmp-{}", crate::meta::now_nanos_unique()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .map_err(io)?;
    file.write_all(&bytes)
        .and_then(|()| file.sync_all())
        .map_err(io)?;
    fs::rename(tmp, path).map_err(io)?;
    for dir in [
        directory(store),
        store.root().join("roots"),
        store.root().to_path_buf(),
    ] {
        File::open(dir).and_then(|f| f.sync_all()).map_err(io)?;
    }
    Ok(())
}
fn verify(store: &Store, manifest: &ObjectRef) -> Result<Manifest> {
    let tree = store.read_manifest(manifest)?;
    if tree.header().is_some() {
        return refuse(
            Code::WRONG_TYPE,
            "source artifact must be an ordinary-file tree",
        );
    }
    for (_, entry) in tree.entries() {
        match store.record_valid(&entry.blob().sha256) {
            Ok(record) if record.length == entry.blob().length => {}
            _ => {
                return refuse(
                    Code::DURABILITY_UNPROVEN,
                    "tree member has no standing verification record",
                )
            }
        }
    }
    Ok(tree)
}
/// Start/resume a source writer. The caller holds this guard across every body transfer.
pub struct Writer<'a> {
    store: &'a Store,
    root: TreeRoot,
    _writer: WriterGuard,
    _owner_lock: File,
}
impl<'a> Writer<'a> {
    pub fn open(store: &'a Store, owner: &str, manifest: ObjectRef) -> Result<Self> {
        let writer = WriterGuard::acquire(store.root())?;
        let _lock = lock(store)?;
        let owner_lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path(store, owner)?.with_extension("lock"))
            .map_err(io)?;
        owner_lock.try_lock_exclusive().map_err(|_| Refusal {
            code: Code::STORE_BUSY,
            detail: "source owner already has a live writer".into(),
        })?;
        let root = match read(store, owner)? {
            Some(root) => {
                if root.released {
                    return refuse(Code::TRANSACTION_CLOSED, "source owner was released");
                }
                if root.manifest != manifest {
                    return refuse(
                        Code::TRANSACTION_CONFLICT,
                        "source owner names a different member roster",
                    );
                }
                root
            }
            None => {
                let root = TreeRoot {
                    owner: owner.into(),
                    producer: owner.into(),
                    manifest,
                    objects: vec![],
                    complete: false,
                    released: false,
                };
                write(store, &root)?;
                root
            }
        };
        Ok(Self {
            store,
            root,
            _writer: writer,
            _owner_lock: owner_lock,
        })
    }
    pub fn completed(&self) -> Result<bool> {
        if self.root.complete {
            verify(self.store, &self.root.manifest)?;
        }
        Ok(self.root.complete)
    }
    pub fn landed(&mut self, object: &ObjectRef) -> Result<()> {
        if !matches!(self.store.record_valid(&object.sha256), Ok(row) if row.length == object.length)
        {
            return refuse(Code::DURABILITY_UNPROVEN, "source member is not admitted");
        }
        self.hold_expected(object)
    }
    /// Publish exact-member liveness before admission may unlink its durable prefix.
    /// This is neither verification nor a reusable result: completion verifies the roster.
    pub(crate) fn hold_expected(&mut self, object: &ObjectRef) -> Result<()> {
        if !self.root.objects.contains(object) {
            self.root.objects.push(object.clone());
            self.root.objects.sort_by(|a, b| a.sha256.cmp(&b.sha256));
            let _lock = lock(self.store)?;
            write(self.store, &self.root)?;
        }
        Ok(())
    }
    pub fn finish(mut self, tree: &Manifest) -> Result<TreeRoot> {
        if tree.object_ref() != self.root.manifest {
            return refuse(
                Code::OBJECT_ID_MISMATCH,
                "source manifest differs from accepted roster",
            );
        }
        self.store.put_manifest(tree)?;
        verify(self.store, &self.root.manifest)?;
        self.root.complete = true;
        let _lock = lock(self.store)?;
        write(self.store, &self.root)?;
        Ok(self.root.clone())
    }
}
/// Retain already-admitted ordinary-file bytes as one producer result.
pub fn create(store: &Store, owner: &str, tree: &Manifest) -> Result<TreeRoot> {
    check_roster(owner, tree)?;
    let mut writer = Writer::open(store, owner, tree.object_ref())?;
    for (_, entry) in tree.entries() {
        writer.landed(entry.blob())?;
    }
    writer.finish(tree)
}
/// Adopt only from a still-retained complete owner, before releasing the source owner.
pub fn retain(store: &Store, source: &str, owner: &str) -> Result<TreeRoot> {
    let _writer = WriterGuard::acquire(store.root())?;
    let _lock = lock(store)?;
    let original = read(store, source)?.ok_or_else(|| Refusal {
        code: Code::ROOT_ABSENT,
        detail: "source tree root is absent".into(),
    })?;
    if let Some(root) = read(store, owner)? {
        if root.manifest != original.manifest || root.producer != original.producer {
            return refuse(
                Code::CROSS_SUBJECT_REPLAY,
                "recipient root names another tree",
            );
        }
        if root.released {
            return refuse(
                Code::TRANSACTION_CLOSED,
                "recipient tree owner was released",
            );
        }
        if !root.complete {
            return refuse(
                Code::TRANSACTION_CONFLICT,
                "recipient has an incomplete producer",
            );
        }
        verify(store, &root.manifest)?;
        return Ok(root);
    }
    if original.released || !original.complete {
        return refuse(
            Code::TRANSACTION_CLOSED,
            "source tree has no complete retained owner",
        );
    }
    verify(store, &original.manifest)?;
    let root = TreeRoot {
        owner: prefixed("recipient owner", owner)?,
        ..original
    };
    write(store, &root)?;
    Ok(root)
}
pub fn release(store: &Store, owner: &str) -> Result<()> {
    let _writer = WriterGuard::acquire(store.root())?;
    let _lock = lock(store)?;
    let Some(mut root) = read(store, owner)? else {
        return refuse(Code::ROOT_ABSENT, "tree owner is absent");
    };
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path(store, owner)?.with_extension("lock"))
        .map_err(io)?;
    file.try_lock_exclusive().map_err(|_| Refusal {
        code: Code::STORE_BUSY,
        detail: "source writer must stop before release".into(),
    })?;
    root.released = true;
    root.objects.clear();
    write(store, &root)?;
    crate::transport::prefix::discard_owner(store, owner)
}
pub(crate) fn held_objects(store: &Store) -> Result<Vec<HeldKey>> {
    let entries = match fs::read_dir(directory(store)) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
        Err(e) => return Err(io(e)),
    };
    let mut result = vec![];
    for entry in entries {
        let path = entry.map_err(io)?.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let owner = format!("sha256:{}", path.file_stem().unwrap().to_string_lossy());
        let root = read(store, &owner)?.unwrap();
        if root.released {
            continue;
        }
        for (kind, object) in root
            .objects
            .iter()
            .map(|o| ("blob", o))
            .chain(root.complete.then_some(("manifest", &root.manifest)))
        {
            let key = if kind == "blob" {
                crate::storage::blob_key(&object.sha256)?
            } else {
                crate::storage::manifest_key(&object.sha256)?
            };
            result.push(HeldKey::parse_line(&crate::canon::write(&Value::obj(
                vec![
                    ("key", Value::str(key)),
                    ("kind", Value::str(kind)),
                    ("length", Value::uint(object.length)),
                ],
            )))?);
        }
    }
    Ok(result)
}

/// Fence a stopped predecessor, retaining its completed objects and verified prefixes
/// independently. Existing recipient checkpoints win; adoption never transfers ownership.
pub fn adopt_partial(store: &Store, source: &str, owner: &str) -> Result<TreeRoot> {
    if source == owner {
        return refuse(
            Code::TRANSACTION_CONFLICT,
            "partial adoption needs independent owners",
        );
    }
    let original = read(store, source)?.ok_or_else(|| Refusal {
        code: Code::ROOT_ABSENT,
        detail: "partial source root is absent".into(),
    })?;
    let source_writer = Writer::open(store, source, original.manifest.clone())?;
    let original = source_writer.root.clone();
    if original.complete {
        return refuse(
            Code::TRANSACTION_CLOSED,
            "complete source should use retained-result adoption",
        );
    }
    let mut recipient = Writer::open(store, owner, original.manifest)?;
    if recipient.root.complete {
        return refuse(
            Code::TRANSACTION_CONFLICT,
            "recipient source is already complete",
        );
    }
    if !original.objects.is_empty() {
        // Incomplete roots reserve expected members before downloading them. Only
        // present, natively verified objects can become completed recipient work.
        let (plan, _) =
            FetchPlan::of_objects(store, "source-progress-adoption", &original.objects)?;
        for object in plan.held {
            recipient.landed(&object)?;
        }
    }
    crate::transport::prefix::adopt_owner(store, source, owner, |object| recipient.landed(object))?;
    Ok(recipient.root.clone())
}

/// Build the existing converter's roster from verified retained raw members. Paths stay
/// native implementation detail; they are never part of the client artifact capability.
#[allow(clippy::type_complexity)]
pub fn conversion_roster(
    store: &Store,
    owner: &str,
) -> Result<(
    ObjectRef,
    Vec<crate::ingest::source::VerifiedModelSourceFile>,
    Vec<String>,
)> {
    let _writer = WriterGuard::acquire(store.root())?;
    let _lock = lock(store)?;
    let root = read(store, owner)?.ok_or_else(|| Refusal {
        code: Code::ROOT_ABSENT,
        detail: "source tree root is absent".into(),
    })?;
    if !root.complete || root.released {
        return refuse(
            Code::TRANSACTION_CLOSED,
            "conversion requires a complete retained source tree",
        );
    }
    let tree = verify(store, &root.manifest)?;
    let mut roster = Vec::new();
    for (member, entry) in tree.entries() {
        let object = entry.blob();
        let mut file = store.open_nofollow(&object.sha256)?;
        let header = if member.ends_with(".safetensors.index.json") {
            if object.length > 32 << 20 {
                return refuse(
                    Code::SIZE_CAP,
                    "source index exceeds converter header bound",
                );
            }
            let mut bytes = Vec::new();
            file.take(object.length)
                .read_to_end(&mut bytes)
                .map_err(io)?;
            bytes
        } else if member.ends_with(".safetensors") {
            let mut prefix = [0u8; 8];
            file.read_exact(&mut prefix).map_err(io)?;
            let length = u64::from_le_bytes(prefix);
            if length > crate::limits::CARRIER_HEADER_MAX_BYTES as u64
                || length.checked_add(8).is_none_or(|n| n > object.length)
            {
                return refuse(Code::SIZE_CAP, "source tensor header exceeds bound");
            }
            let mut bytes = prefix.to_vec();
            file.take(length).read_to_end(&mut bytes).map_err(io)?;
            bytes
        } else {
            continue;
        };
        roster.push(crate::ingest::source::VerifiedModelSourceFile {
            member: member.clone(),
            object_id: object.id(),
            length: object.length,
            path: store.object_path(&object.sha256),
            header,
        });
    }
    let landed = roster.iter().map(|row| row.member.clone()).collect();
    Ok((root.manifest, roster, landed))
}

/// Prove the largest completed root fits before any member payload is transferred.
pub fn check_roster(owner: &str, tree: &Manifest) -> Result<()> {
    prefixed("source owner", owner)?;
    let mut objects: Vec<_> = tree
        .entries()
        .iter()
        .map(|(_, entry)| entry.blob().clone())
        .collect();
    objects.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    objects.dedup();
    let root = TreeRoot {
        owner: owner.into(),
        producer: owner.into(),
        manifest: tree.object_ref(),
        objects,
        complete: true,
        released: false,
    };
    if root.canonical_bytes().len() > TreeRoot::MAX_BYTES {
        return refuse(
            Code::SIZE_CAP,
            "selected source root exceeds metadata bound before transfer",
        );
    }
    Ok(())
}

/// Cancel a recipient before or after adoption. The original retained receipt supplies
/// immutable producer/manifest identity; a durable tombstone fences any later retain.
pub fn release_retention(
    store: &Store,
    source: &str,
    receipt_digest: &str,
    owner: &str,
) -> Result<()> {
    let _writer = WriterGuard::acquire(store.root())?;
    let _lock = lock(store)?;
    let original = read(store, source)?.ok_or_else(|| Refusal {
        code: Code::ROOT_ABSENT,
        detail: "source receipt root is absent".into(),
    })?;
    prefixed("source receipt", receipt_digest)?;
    if crate::ids::object_id(&original.receipt()?) != receipt_digest {
        return refuse(
            Code::CROSS_SUBJECT_REPLAY,
            "tree release differs from original native receipt",
        );
    }
    let mut root = match read(store, owner)? {
        Some(root) => {
            if root.producer != original.producer || root.manifest != original.manifest {
                return refuse(
                    Code::CROSS_SUBJECT_REPLAY,
                    "tree release names another recipient receipt",
                );
            }
            root
        }
        None => TreeRoot {
            owner: prefixed("tree recipient", owner)?,
            ..original
        },
    };
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path(store, owner)?.with_extension("lock"))
        .map_err(io)?;
    file.try_lock_exclusive().map_err(|_| Refusal {
        code: Code::STORE_BUSY,
        detail: "source writer must stop before recipient release".into(),
    })?;
    root.released = true;
    root.objects.clear();
    write(store, &root)?;
    crate::transport::prefix::discard_owner(store, owner)
}

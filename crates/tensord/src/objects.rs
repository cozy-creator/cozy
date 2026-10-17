//! Objects a signer uploads (run inputs, local package sources), content-addressed. An upload
//! lands byte ranges in any order, several side by side, and resumes from the ranges held. Bytes
//! stage per signer and enter the TensorFS store only once every byte is held and the digest
//! verifies. A run names only objects its own signer wrote, so another signer's object is neither
//! readable nor reported as held.
use crate::{
    execution::Engine,
    journal::{Artifact, Execution, HeldMemo, ResultRecord},
};
use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, MutexGuard, Weak,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tensorfs_core::{
    catalog::WriterGuard, ids::ObjectRef, object_roots, sha256, store::Fault, store::Store,
};

/// The largest memoized result record this machine holds for reuse.
const MEMO_RESULT_BYTES: usize = 48 << 10;

/// Byte ranges `[start, end)`, sorted and apart.
pub type Ranges = Vec<(u64, u64)>;

pub struct Objects {
    root: PathBuf,
    store: Arc<Store>,
    engine: Arc<Engine>,
    /// Uploads under way, by part file.
    uploads: Mutex<HashMap<PathBuf, Weak<Upload>>>,
}

/// Why a write was refused, as the API's typed reason.
#[derive(Debug)]
pub struct Refused {
    pub code: &'static str,
    pub message: String,
}
fn refused(code: &'static str, message: impl Into<String>) -> Refused {
    Refused {
        code,
        message: message.into(),
    }
}
impl From<io::Error> for Refused {
    fn from(error: io::Error) -> Self {
        refused("object_storage_failed", error.to_string())
    }
}

/// One object's upload, shared by the writes that carry its ranges.
pub struct Upload {
    actor: String,
    object: ObjectRef,
    part: PathBuf,
    file: File,
    held: Mutex<Ranges>,
    /// One settle at a time records what landed.
    settling: Mutex<()>,
    /// Stored or dropped: later writes to it land nowhere, and a new upload starts afresh.
    ended: AtomicBool,
}

impl Upload {
    /// Lands `data` at `offset`.
    pub fn write_at(&self, offset: u64, data: &[u8]) -> Result<(), Refused> {
        let end = offset + data.len() as u64;
        if end > self.object.length {
            return Err(refused(
                "object_length_exceeded",
                "more bytes than the object's declared length",
            ));
        }
        if self.ended.load(Ordering::Acquire) {
            return Ok(());
        }
        self.file.write_all_at(data, offset)?;
        insert(&mut self.held.lock().unwrap(), (offset, end));
        Ok(())
    }

    /// What has landed, synced or not.
    pub fn held(&self) -> Ranges {
        self.held.lock().unwrap().clone()
    }
}

/// Adds `[start, end)` to `ranges`, merging what it touches.
fn insert(ranges: &mut Ranges, (mut start, mut end): (u64, u64)) {
    if start >= end {
        return;
    }
    ranges.retain(|&(s, e)| {
        let apart = e < start || s > end;
        if !apart {
            (start, end) = (start.min(s), end.max(e));
        }
        apart
    });
    let at = ranges.partition_point(|&(s, _)| s < start);
    ranges.insert(at, (start, end));
}

fn whole(length: u64) -> Ranges {
    match length {
        0 => vec![],
        length => vec![(0, length)],
    }
}

/// The record of what a part holds: its object's length and the ranges synced.
fn held_path(part: &Path) -> PathBuf {
    part.with_extension("held")
}
fn read_held(part: &Path) -> io::Result<Option<(u64, Ranges)>> {
    match fs::read(held_path(part)) {
        Ok(raw) => Ok(serde_json::from_slice(&raw).ok()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}
fn write_held(part: &Path, length: u64, held: &Ranges) -> io::Result<()> {
    let path = held_path(part);
    let next = path.with_extension("held.next");
    fs::write(&next, serde_json::to_vec(&(length, held))?)?;
    fs::rename(next, path)
}

impl Objects {
    pub fn new(root: &Path, store: Arc<Store>, engine: Arc<Engine>) -> io::Result<Self> {
        fs::create_dir_all(root)?;
        Ok(Self {
            root: root.to_path_buf(),
            store,
            engine,
            uploads: Mutex::default(),
        })
    }

    /// Excludes root release while a Write or an acceptance takes references.
    pub(crate) fn guard(&self) -> MutexGuard<'_, ()> {
        self.engine.object_custody.lock().unwrap()
    }
    /// Excludes the store's GC between an object's put and its root.
    pub(crate) fn writer_guard(&self) -> io::Result<WriterGuard> {
        WriterGuard::acquire(self.store.root()).map_err(io::Error::other)
    }
    /// Roots `objects` in the store; called with `guard` and `writer_guard` held.
    pub(crate) fn retain(&self, objects: &[ObjectRef]) -> io::Result<()> {
        for object in objects {
            object_roots::retain(&self.store, object).map_err(io::Error::other)?;
        }
        Ok(())
    }

    /// Releases the roots of objects no unfinished run names and nothing used for the TTL,
    /// or for `IDLE` while the store's disk is low. Their bytes are then the store GC's.
    pub fn sweep(&self) -> io::Result<usize> {
        let disk =
            tensorfs_core::admission::Disk::measure(self.store.root()).map_err(io::Error::other)?;
        self.release(match disk.short() {
            Some(_) => crate::reclaim::IDLE,
            None => crate::reclaim::TTL,
        })
    }

    /// `sweep` over roots unused for `unused`. A root written just before a crash has no
    /// journal row yet: it ages from its own mtime.
    pub fn release(&self, unused: Duration) -> io::Result<usize> {
        let _custody = self.guard();
        let before = SystemTime::now().checked_sub(unused).unwrap_or(UNIX_EPOCH);
        let before_ms = before
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        let mut released = 0;
        for object in object_roots::list(&self.store).map_err(io::Error::other)? {
            let root = self
                .store
                .root()
                .join("roots/objects")
                .join(format!("{}.json", object.sha256));
            if fs::metadata(root)?.modified()? >= before {
                continue;
            }
            if self
                .engine
                .with_journal(|j| j.object_releasable(&object.sha256, before_ms))?
                && object_roots::remove(&self.store, &object.sha256).map_err(io::Error::other)?
            {
                // A memo answered from this file cannot be answered any more.
                self.engine.with_journal(|j| j.drop_held_memos(&object.sha256))?;
                released += 1;
            }
        }
        Ok(released)
    }

    /// The stored object this signer wrote, or None.
    pub fn path(&self, actor: &str, digest: &str) -> io::Result<Option<(PathBuf, u64)>> {
        let Some(hex) = hex(digest) else {
            return Ok(None);
        };
        let Some(length) = self.engine.with_journal(|j| j.object(actor, &hex))? else {
            return Ok(None);
        };
        let path = self.store.object_path(&hex);
        Ok(path.is_file().then_some((path, length)))
    }

    /// The success a memoized computation left for this signer, its files linked into `run`'s
    /// custody as that run's own result; None when there is none or a file is gone (the memo
    /// goes with it).
    pub fn recall(&self, actor: &str, computation: &str, run: &str) -> io::Result<Option<ResultRecord>> {
        let _custody = self.guard();
        let Some(raw) = self.engine.with_journal(|j| j.held_memo(actor, computation))? else {
            return Ok(None);
        };
        let Ok(mut record) = serde_json::from_slice::<ResultRecord>(&raw) else {
            return Ok(None);
        };
        let custody = self.engine.root.join("results").join(run);
        for (index, artifact) in record.artifacts.iter_mut().enumerate() {
            let Some((held, _)) = self
                .path(actor, &format!("sha256:{}", artifact.sha256))?
                .filter(|(_, length)| *length == artifact.length)
            else {
                self.engine.with_journal(|j| j.drop_held_memos(&artifact.sha256))?;
                return Ok(None);
            };
            fs::create_dir_all(&custody)?;
            let name = format!("{index}-{}", artifact.sha256);
            if let Err(error) = fs::hard_link(&held, custody.join(&name)) {
                if error.kind() != io::ErrorKind::AlreadyExists {
                    fs::copy(&held, custody.join(&name))?;
                }
            }
            artifact.path = format!("results/{run}/{name}");
        }
        Ok(Some(record))
    }

    /// A memoizable run's success `result`, its files kept for its signer, to hold under its
    /// computation while this machine holds every file it names. A tree's members are not held
    /// as one result, nor is a record over `MEMO_RESULT_BYTES`.
    pub fn keep(&self, record: &Execution, result: &ResultRecord) -> io::Result<Option<HeldMemo>> {
        let Some(submission) = &record.submission else {
            return Ok(None);
        };
        if result.asset_bindings.iter().any(|b| b.media_type == crate::gpu_service::TREE_MEDIA) {
            return Ok(None);
        }
        let bound: Vec<_> = result
            .artifacts
            .iter()
            .filter(|a| result.asset_bindings.iter().any(|b| b.relative_path == a.name))
            .collect();
        let held = ResultRecord {
            value: result.value.clone(),
            artifacts: bound.iter().map(|a| Artifact { path: String::new(), ..(*a).clone() }).collect(),
            asset_bindings: result.asset_bindings.clone(),
        };
        let raw = serde_json::to_vec(&held)?;
        if raw.len() > MEMO_RESULT_BYTES {
            return Ok(None);
        }
        for artifact in &bound {
            let object = ObjectRef { sha256: artifact.sha256.clone(), length: artifact.length };
            self.adopt(&submission.actor, &record.id, &self.engine.root.join(&artifact.path), &object)?;
        }
        Ok(Some(HeldMemo {
            actor: submission.actor.clone(),
            computation: record.memo.computation.clone(),
            record: raw,
            objects: bound.iter().map(|a| a.sha256.clone()).collect(),
        }))
    }

    /// A file this machine produced for the signer (a child run's result) as its object, so a
    /// later run of that signer may be handed it.
    pub fn adopt(&self, actor: &str, parent: &str, file: &Path, object: &ObjectRef) -> io::Result<()> {
        let _custody = self.guard();
        let _writer = self.writer_guard()?;
        if !crate::products::held(&self.store, object) {
            self.store
                .put_file(file, Some(object), &Fault::default())
                .map_err(io::Error::other)?;
        }
        self.retain(std::slice::from_ref(object))?;
        self.engine
            .with_journal(|j| j.adopt_object(actor, parent, object))?;
        Ok(())
    }

    fn part(&self, actor: &str, hex: &str) -> PathBuf {
        let dir = self.root.join(&sha256::hex_digest(actor.as_bytes())[..32]);
        dir.join(format!("{hex}.part"))
    }

    /// The stored object this signer wrote, its roots renewed so a run naming it soon finds it.
    fn renew(&self, actor: &str, digest: &str) -> io::Result<Option<u64>> {
        let _custody = self.guard();
        let Some((_, length)) = self.path(actor, digest)? else {
            return Ok(None);
        };
        let object = ObjectRef {
            sha256: digest["sha256:".len()..].to_string(),
            length,
        };
        let _writer = self.writer_guard()?;
        self.retain(std::slice::from_ref(&object))?;
        self.engine.with_journal(|j| j.bind_object(actor, &object))?;
        Ok(Some(length))
    }

    /// The ranges of `digest` (`sha256:<hex>`) this signer holds: all of it once stored, else
    /// what its upload has landed. An upload whose ranges cover it is settled first, so whole
    /// always means stored; bytes that differ from the digest are dropped, and held no more.
    pub fn held(&self, actor: &str, digest: &str) -> Result<Ranges, Refused> {
        let hex = valid(digest)?;
        if let Some(length) = self.renew(actor, digest)? {
            return Ok(whole(length));
        }
        let part = self.part(actor, &hex);
        let upload = self.uploads.lock().unwrap().get(&part).and_then(Weak::upgrade);
        if let Some(upload) = upload {
            return match upload.held() {
                held if held != whole(upload.object.length) => Ok(held),
                _ => match self.settle(&upload) {
                    Err(refusal) if refusal.code == "object_digest_mismatch" => Ok(vec![]),
                    settled => settled,
                },
            };
        }
        Ok(read_held(&part)?.map(|(_, held)| held).unwrap_or_default())
    }

    /// The upload of `digest`, `length` bytes, for this signer: joined while under way, else
    /// resumed from the ranges an earlier one synced. None once the object is stored.
    pub fn upload(&self, actor: &str, digest: &str, length: u64) -> Result<Option<Arc<Upload>>, Refused> {
        let hex = valid(digest)?;
        if self.renew(actor, digest)? == Some(length) {
            return Ok(None);
        }
        let part = self.part(actor, &hex);
        let mut uploads = self.uploads.lock().unwrap();
        uploads.retain(|_, upload| upload.strong_count() > 0);
        if let Some(upload) = uploads.get(&part).and_then(Weak::upgrade) {
            if upload.object.length != length {
                return Err(refused(
                    "invalid_request",
                    format!("{digest} is being written as {} bytes", upload.object.length),
                ));
            }
            return Ok(Some(upload));
        }
        fs::create_dir_all(part.parent().expect("a signer's directory"))?;
        let file = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&part)?;
        let held = match read_held(&part)? {
            Some((was, held)) if was == length => held,
            _ => {
                file.set_len(0)?;
                vec![]
            }
        };
        file.set_len(length)?;
        let missing = length - held.iter().map(|(s, e)| e - s).sum::<u64>();
        self.store
            .admits(missing)
            .map_err(|e| refused("machine_disk_full", format!("no room for the object: {e}")))?;
        let upload = Arc::new(Upload {
            actor: actor.to_string(),
            object: ObjectRef { sha256: hex, length },
            part: part.clone(),
            file,
            held: Mutex::new(held),
            settling: Mutex::default(),
            ended: AtomicBool::new(false),
        });
        uploads.insert(part, Arc::downgrade(&upload));
        Ok(Some(upload))
    }

    /// Records what `upload` landed, durably. Once it holds every byte, the object is verified
    /// against its digest and stored for its signer; bytes that differ are dropped. Answers the
    /// ranges held.
    pub fn settle(&self, upload: &Upload) -> Result<Ranges, Refused> {
        let digest = format!("sha256:{}", upload.object.sha256);
        let length = upload.object.length;
        let _settling = upload.settling.lock().unwrap();
        if upload.ended.load(Ordering::Acquire) {
            return self.held(&upload.actor, &digest);
        }
        let held = upload.held();
        upload.file.sync_data()?;
        if held != whole(length) {
            write_held(&upload.part, length, &held)?;
            return Ok(held);
        }
        let _custody = self.guard();
        let _writer = self.writer_guard()?;
        upload.ended.store(true, Ordering::Release);
        self.uploads.lock().unwrap().remove(&upload.part);
        let stored = (self.store).put_file(&upload.part, Some(&upload.object), &Fault::default());
        let _ = fs::remove_file(&upload.part);
        let _ = fs::remove_file(held_path(&upload.part));
        stored.map_err(|e| {
            refused(
                match e.code {
                    tensorfs_core::err::Code::CAPACITY_EXHAUSTED => "machine_disk_full",
                    tensorfs_core::err::Code::OBJECT_ID_MISMATCH
                    | tensorfs_core::err::Code::LENGTH_MISMATCH => "object_digest_mismatch",
                    _ => "object_storage_failed",
                },
                e.to_string(),
            )
        })?;
        self.retain(std::slice::from_ref(&upload.object))?;
        self.engine
            .with_journal(|j| j.bind_object(&upload.actor, &upload.object))?;
        Ok(held)
    }

    /// Writes all of `bytes` as one upload.
    #[cfg(test)]
    pub fn write(&self, actor: &str, digest: &str, bytes: &[u8]) -> Result<Ranges, Refused> {
        match self.upload(actor, digest, bytes.len() as u64)? {
            None => Ok(whole(bytes.len() as u64)),
            Some(upload) => {
                upload.write_at(0, bytes)?;
                self.settle(&upload)
            }
        }
    }
}

fn valid(digest: &str) -> Result<String, Refused> {
    hex(digest).ok_or_else(|| {
        refused(
            "invalid_request",
            "an object is named by sha256:<64 lowercase hex>",
        )
    })
}

fn hex(digest: &str) -> Option<String> {
    digest
        .strip_prefix("sha256:")
        .filter(|h| h.len() == 64 && h.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
        .map(String::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn objects(root: &Path) -> Objects {
        let service =
            crate::service::Service::open(&root.join("state"), &root.join("g"), 1).unwrap();
        let store = Arc::new(Store::ensure(&root.join("tensorfs")).unwrap());
        Objects::new(&root.join("writes"), store, service.engine.clone()).unwrap()
    }

    /// Ranges land in any order, side by side, survive the machine's restart, verify once whole,
    /// and belong to their signer.
    #[test]
    fn an_upload_lands_ranges_resumes_verifies_and_belongs_to_its_signer() {
        let root = std::env::temp_dir().join(format!("cm-objects-{}", uuid::Uuid::new_v4()));
        let objects = objects(&root);
        let bytes: Vec<u8> = (0..300_000u32).map(|i| (i % 251) as u8).collect();
        let digest = format!("sha256:{}", sha256::hex_digest(&bytes));
        let length = bytes.len() as u64;

        let (first, second) = (
            objects.upload("alice", &digest, length).unwrap().unwrap(),
            objects.upload("alice", &digest, length).unwrap().unwrap(),
        );
        assert!(Arc::ptr_eq(&first, &second), "one upload, joined");
        second.write_at(200_000, &bytes[200_000..250_000]).unwrap();
        first.write_at(0, &bytes[..100_000]).unwrap();
        assert_eq!(objects.held("alice", &digest).unwrap(), [(0, 100_000), (200_000, 250_000)]);
        assert_eq!(objects.settle(&first).unwrap(), [(0, 100_000), (200_000, 250_000)]);
        drop((first, second));

        // What was synced is held after a restart; the rest completes the object.
        let objects = Objects::new(&root.join("writes"), objects.store.clone(), objects.engine.clone()).unwrap();
        assert_eq!(objects.held("alice", &digest).unwrap(), [(0, 100_000), (200_000, 250_000)]);
        let upload = objects.upload("alice", &digest, length).unwrap().unwrap();
        upload.write_at(50_000, &bytes[50_000..200_000]).unwrap();
        upload.write_at(250_000, &bytes[250_000..]).unwrap();
        assert_eq!(
            upload.write_at(250_000, &[0; 50_001]).unwrap_err().code,
            "object_length_exceeded"
        );
        // Asked while its ranges cover it, the upload is settled first: whole means stored.
        assert_eq!(objects.held("alice", &digest).unwrap(), [(0, length)]);
        assert_eq!(objects.settle(&upload).unwrap(), [(0, length)]);
        let (path, held) = objects.path("alice", &digest).unwrap().unwrap();
        assert_eq!((fs::read(path).unwrap(), held), (bytes.clone(), length));
        assert!(objects.upload("alice", &digest, length).unwrap().is_none());
        assert_eq!(objects.held("alice", &digest).unwrap(), [(0, length)]);

        // Another signer learns nothing of it and must send the bytes itself.
        assert!(objects.path("bob", &digest).unwrap().is_none());
        assert_eq!(objects.held("bob", &digest).unwrap(), []);

        // Bytes that differ from their digest are dropped, not stored.
        let wrong = format!("sha256:{}", "0".repeat(64));
        assert_eq!(objects.write("alice", &wrong, b"abc").unwrap_err().code, "object_digest_mismatch");
        assert_eq!(objects.held("alice", &wrong).unwrap(), []);
        let _ = fs::remove_dir_all(root);
    }

    /// A held memo answers with its file in the answered run's custody while its signer holds
    /// the file; when reclaim releases the file, the memo goes with it.
    #[test]
    fn a_held_memo_answers_while_its_file_is_held_and_goes_with_it() {
        let root = std::env::temp_dir().join(format!("cm-objects-{}", uuid::Uuid::new_v4()));
        let objects = objects(&root);
        let bytes = b"note 5".to_vec();
        let hex = sha256::hex_digest(&bytes);
        objects.write("alice", &format!("sha256:{hex}"), &bytes).unwrap();
        let record = crate::journal::ResultRecord {
            value: serde_json::json!({"file": {"asset_ref": "attempt:1/file/0001"}}),
            artifacts: vec![crate::journal::Artifact { name: "file-0001".into(), path: String::new(), sha256: hex.clone(), length: 6 }],
            asset_bindings: vec![crate::journal::AssetBinding {
                relative_path: "file-0001".into(), asset_ref: "attempt:1/file/0001".into(), media_type: "text/plain".into(),
                length: 6,
            }],
        };
        let engine = objects.engine.clone();
        let held = crate::journal::HeldMemo { actor: "alice".into(), computation: "sha256:c".into(),
            record: serde_json::to_vec(&record).unwrap(), objects: vec![hex.clone()] };
        engine.with_journal(|j| j.hold_memo(&held)).unwrap();
        let held = objects.recall("alice", "sha256:c", "7").unwrap().unwrap();
        assert_eq!(held.artifacts[0].path, format!("results/7/0-{hex}"));
        assert_eq!(fs::read(engine.root.join(&held.artifacts[0].path)).unwrap(), bytes);
        assert_eq!(held.value, record.value);
        assert!(objects.recall("bob", "sha256:c", "8").unwrap().is_none(), "a memo is its signer's");
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(objects.release(Duration::ZERO).unwrap(), 1);
        assert!(engine.with_journal(|j| j.held_memo("alice", "sha256:c")).unwrap().is_none(), "reclaim dropped the memo");
        let _ = fs::remove_dir_all(root);
    }
}

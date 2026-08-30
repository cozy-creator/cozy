//! Immutable blobs at `blobs/xx/yy/<hex>` plus immutable manifests at
//! `manifests/xx/yy/<hex>.json`.
//!
//! Tensor parts, blob segments and documents admit through ONE put path; the store knows
//! no shapes (tensorfs.md §1). Identity is `sha256(stored_bytes)`, so an identity-bearing
//! document admits only as its exact canonical bytes — a pretty twin lands at a different
//! id and refuses `put_expected`.
//!
//! Admission is `writing → verified-temp → installed`: temp in `tmp/`, hash while writing,
//! fsync, 0444, then a no-clobber `link()` into place plus a parent-directory fsync. A
//! crash at any stage leaves a temp, never a partial object under `objects/`.
//!
//! `contains` is a PRESENCE HINT and nothing more. Trusting bytes requires a durable
//! verification record bound to object id, length, file identity (dev, ino), size and
//! change token (mtime). Any mismatch invalidates the record and
//! forces a rehash; a rehash that disagrees quarantines.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::{hex64, Doc, ObjectRef};
use crate::sha256::{self, Sha256};

const BUF: usize = 1 << 20;

#[cfg(target_os = "linux")]
const O_NOFOLLOW: i32 = 0o400_000;
#[cfg(target_os = "macos")]
const O_NOFOLLOW: i32 = 0x0100;

fn io(what: impl AsRef<str>, e: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("{}: {e}", what.as_ref()),
    }
}

// ---------------------------------------------------- durable verification record

/// `<root>/verified/xx/yy/<hex>` — the durable fact that these exact bytes were hashed
/// once, on admission. Bound to everything that could change underneath it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VRecord {
    pub sha256: String,
    pub length: u64,
    pub dev: u64,
    pub ino: u64,
    pub mtime_s: u64,
    pub mtime_ns: u64,
}

impl VRecord {
    fn of(hex: &str, md: &fs::Metadata) -> VRecord {
        VRecord {
            sha256: hex.to_string(),
            length: md.len(),
            dev: md.dev(),
            ino: md.ino(),
            mtime_s: md.mtime().max(0) as u64,
            mtime_ns: md.mtime_nsec().max(0) as u64,
        }
    }
    /// The first disagreeing bound fact, or None when the record still stands.
    fn disagreement(&self, hex: &str, md: &fs::Metadata) -> Option<String> {
        if self.dev != md.dev() {
            return Some(format!("file device {} != {}", self.dev, md.dev()));
        }
        self.portable_disagreement(hex, md)
    }

    /// Everything stable across a remount of the same NFS object. Used only by the explicit
    /// record migration; normal reads continue to require exact mount-local device identity.
    fn portable_disagreement(&self, hex: &str, md: &fs::Metadata) -> Option<String> {
        let now = VRecord::of(hex, md);
        if self.sha256 != now.sha256 {
            return Some(format!("object id {} != {}", self.sha256, now.sha256));
        }
        if self.ino != now.ino {
            return Some(format!("file inode {} != {}", self.ino, now.ino));
        }
        if self.length != now.length {
            return Some(format!("size {} != {}", self.length, now.length));
        }
        if self.mtime_s != now.mtime_s || self.mtime_ns != now.mtime_ns {
            return Some(format!(
                "change token {}.{:09} != {}.{:09}",
                self.mtime_s, self.mtime_ns, now.mtime_s, now.mtime_ns
            ));
        }
        None
    }
}

// ---------------------------------------------------------------- fault injection

/// Dev-only kill points. `tfs put --fault <stage>` reaches the stage, announces it, and
/// waits to be killed from outside — a real SIGKILL to a real process, never a simulation.
#[derive(Debug, Clone, Default)]
pub struct Fault {
    pub stage: Option<String>,
    pub ready: Option<PathBuf>,
}

pub const FAULT_STAGES: &[&str] = &[
    "open",    // temp created, no bytes
    "partial", // first chunk written, temp is a partial object
    "written", // all bytes written, not yet durable
    "fsync",   // durable temp, still 0600, not linked
    "chmod",   // 0444 verified temp, one link() from installed
    "linked",  // installed, parent dir not yet fsynced, temp not yet unlinked
    "record",  // verification record written; nothing left to lose
];

impl Fault {
    fn hit(&self, stage: &str) {
        if self.stage.as_deref() != Some(stage) {
            return;
        }
        if let Some(p) = &self.ready {
            let _ = fs::write(p, format!("{stage} {}\n", std::process::id()));
        }
        eprintln!("FAULT-READY {stage} pid={}", std::process::id());
        loop {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
}

// ---------------------------------------------------------------- the store

#[derive(Debug, Clone)]
pub struct Store {
    root: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Put {
    pub obj: ObjectRef,
    /// This process won the no-clobber admission (vs. found the id already installed).
    pub admitted: bool,
}

/// Bytes proven to be `sha256` THROUGH THE DESCRIPTOR THIS HOLDS.
///
/// The point of the type is the fd, not the fields: a `VerifiedFile` cannot be built from a
/// pathname and a promise, so there is no way to spell "verified over there, read over
/// here". Reads start at offset 0 and the descriptor keeps the inode alive for the handle's
/// whole life, so a replacement at the path — same length or not — cannot reach them.
#[derive(Debug)]
pub struct VerifiedFile {
    file: File,
    sha256: String,
    len: u64,
}

impl VerifiedFile {
    /// The id these bytes were proven to have.
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// The length observed on the verified descriptor (`fstat`, never a second `stat`).
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The pinned descriptor. Whatever a caller does with it, it is still the file that was
    /// verified — that is the whole guarantee, and it travels with the handle.
    pub fn into_file(self) -> File {
        self.file
    }

    /// Positional read through the pinned descriptor — the fd, never the name. Shared-ref
    /// on purpose: `pread` moves no cursor, so concurrent readers need no lock.
    pub fn read_exact_at(&self, buf: &mut [u8], off: u64) -> std::io::Result<()> {
        use std::os::unix::fs::FileExt;
        self.file.read_exact_at(buf, off)
    }

    /// `fstat` on the pinned descriptor. A truncation of THIS inode is visible here; a
    /// replacement at the pathname is not — by design, that swap cannot reach this handle.
    pub fn current_len(&self) -> std::io::Result<u64> {
        self.file.metadata().map(|m| m.len())
    }
}

impl Read for VerifiedFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.file.read(buf)
    }
}

impl Seek for VerifiedFile {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.file.seek(pos)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Verified { rehashed: bool },
    Invalidated { why: String },
    Quarantined { to: PathBuf, why: String },
}

#[derive(Debug, Default, Clone)]
pub struct Scrub {
    pub scanned: usize,
    pub sampled: usize,
    pub rehashed: usize,
    pub quarantined: usize,
    pub bytes_hashed: u64,
}

impl Store {
    pub fn init(root: &Path) -> Result<Store> {
        for d in [
            "repos",
            "manifests",
            "blobs",
            "checkouts",
            ".tensorfs/tmp",
            ".tensorfs/quarantine",
        ] {
            fs::create_dir_all(root.join(d)).map_err(|e| io(format!("mkdir {d}"), e))?;
        }
        crate::catalog::Catalog::initialize(root)?;
        Ok(Store {
            root: root.to_path_buf(),
        })
    }

    /// The three authoritative namespaces identify a TensorFS store. Old mixed-object and
    /// state.json layouts are a prelaunch hardcut and never dual-read.
    pub fn open(root: &Path) -> Result<Store> {
        if !["repos", "manifests", "blobs"]
            .iter()
            .all(|name| root.join(name).is_dir())
        {
            return refuse(
                Code::STORE_ERA,
                format!(
                    "{} lacks the repos/manifests/blobs layout; old stores require an offline migration",
                    root.display()
                ),
            );
        }
        Ok(Store {
            root: root.to_path_buf(),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// A malformed id has NO path in this store. `hex[0..2]` panics on a short string, and a
    /// panic is not a refusal — an empty argument from a shell pipeline reached this and
    /// aborted the process instead of producing `OBJECT_ABSENT`. The sentinel never exists
    /// and never embeds the caller's string, so `..` or a leading `/` cannot travel with it;
    /// the caller's own absent-object refusal then names the id it was handed.
    fn shard(&self, base: &str, hex: &str) -> PathBuf {
        if crate::ids::hex64("object id", hex).is_err() {
            return self.root.join(base).join(".malformed-id");
        }
        self.root
            .join(base)
            .join(&hex[0..2])
            .join(&hex[2..4])
            .join(hex)
    }
    pub fn object_path(&self, hex: &str) -> PathBuf {
        self.shard("blobs", hex)
    }
    pub fn blob_path(&self, hex: &str) -> PathBuf {
        self.object_path(hex)
    }
    pub fn manifest_path(&self, hex: &str) -> PathBuf {
        let mut path = self.shard("manifests", hex);
        path.set_extension("json");
        path
    }

    pub fn read_manifest(&self, reference: &ObjectRef) -> Result<crate::manifest::Manifest> {
        let path = self.manifest_path(&reference.sha256);
        let bytes = fs::read(&path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                Refusal {
                    code: Code::OBJECT_ABSENT,
                    detail: format!("manifest {} is absent", reference.id()),
                }
            } else {
                io(format!("read {}", path.display()), error)
            }
        })?;
        if bytes.len() as u64 != reference.length {
            return refuse(
                Code::LENGTH_MISMATCH,
                format!(
                    "manifest {} is {} bytes, expected {}",
                    reference.id(),
                    bytes.len(),
                    reference.length
                ),
            );
        }
        if crate::sha256::hex_digest(&bytes) != reference.sha256 {
            return refuse(
                Code::OBJECT_ID_MISMATCH,
                format!("manifest {} bytes disagree with its key", reference.id()),
            );
        }
        crate::manifest::Manifest::parse(&bytes)
    }

    pub fn put_manifest(&self, manifest: &crate::manifest::Manifest) -> Result<Put> {
        self.put_manifest_held(manifest, None)
    }

    pub fn put_manifest_held(
        &self,
        manifest: &crate::manifest::Manifest,
        operation: Option<&str>,
    ) -> Result<Put> {
        let catalog = crate::catalog::Catalog::open(&self.root)?;
        let _writer = crate::catalog::WriterGuard::acquire(&self.root)?;
        let bytes = manifest.canonical_bytes();
        let object = ObjectRef::of(&bytes);
        if let Some(operation) = operation {
            catalog.hold(
                operation,
                &crate::storage::manifest_key(&object.sha256)?,
                object.length,
            )?;
        }
        let path = self.manifest_path(&object.sha256);
        fs::create_dir_all(path.parent().unwrap()).map_err(|error| io("mkdir manifest", error))?;
        if path.exists() {
            self.read_manifest(&object)?;
            return Ok(Put {
                obj: object,
                admitted: false,
            });
        }
        let temporary = path.parent().unwrap().join(format!(
            ".{}.{}.{}",
            path.file_name().unwrap().to_string_lossy(),
            std::process::id(),
            now_nanos_unique()
        ));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temporary)
            .map_err(|error| io("create manifest temp", error))?;
        file.write_all(&bytes)
            .map_err(|error| io("write manifest", error))?;
        file.sync_all()
            .map_err(|error| io("fsync manifest", error))?;
        fs::set_permissions(&temporary, fs::Permissions::from_mode(0o444))
            .map_err(|error| io("chmod manifest", error))?;
        let admitted = match fs::hard_link(&temporary, &path) {
            Ok(()) => true,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => false,
            Err(error) => return cleanup(&temporary, io("install manifest", error)),
        };
        let _ = fs::remove_file(&temporary);
        if admitted {
            fsync_dir(path.parent().unwrap())?;
        }
        self.read_manifest(&object)?;
        Ok(Put {
            obj: object,
            admitted,
        })
    }

    pub fn repository_path(&self, repo: &crate::repository::RepositoryName) -> PathBuf {
        self.root
            .join("repos")
            .join(&repo.org)
            .join(format!("{}.json", repo.name))
    }

    /// Local repository CAS. `observed` is exactly the body the caller read, or `None` for
    /// create. The durable rename/delete happens before the SQLite projection commit.
    pub fn apply_repository(
        &self,
        observed: Option<&[u8]>,
        mutation: &crate::repository::Mutation,
        fault: &Fault,
    ) -> Result<Option<crate::repository::Repository>> {
        let catalog = crate::catalog::Catalog::open(&self.root)?;
        let _writer = crate::catalog::WriterGuard::acquire(&self.root)?;
        let path = self.repository_path(mutation.repo());
        let current = match fs::read(&path) {
            Ok(bytes) => Some(bytes),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(io(format!("read {}", path.display()), error)),
        };
        if current.as_deref() != observed {
            return refuse(
                Code::REPOSITORY_CONFLICT,
                "repository bytes changed since the caller observed them",
            );
        }
        let classification = match mutation {
            crate::repository::Mutation::PutRelease { manifest, .. } => {
                let parsed = self.read_manifest(manifest)?;
                let header_ref = parsed.header().ok_or_else(|| Refusal {
                    code: Code::ATTACHMENT_CARDINALITY,
                    detail: "release manifest has no cozytensors entry".into(),
                })?;
                let header = crate::checkpoint::load_header(self, header_ref)?;
                crate::checkpoint::walk(self, &parsed)?.require_resident(self)?;
                Some(header.topology_digest())
            }
            crate::repository::Mutation::RemoveRelease { .. }
            | crate::repository::Mutation::DeleteRepository { .. } => None,
        };
        let replacement = crate::repository::apply(observed, mutation, classification.as_deref())?;
        if let Some(repository) = &replacement {
            for release in &repository.releases {
                self.read_manifest(&release.manifest)?;
            }
        }
        let org = mutation.repo().org.clone();
        let name = mutation.repo().name.clone();
        let expected = current;
        let replacement_for_write = replacement.clone();
        let path_for_write = path.clone();
        catalog.commit_repository(&org, &name, || {
            let actual = match fs::read(&path_for_write) {
                Ok(bytes) => Some(bytes),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => return Err(io("read repository under coordination", error)),
            };
            if actual != expected {
                return refuse(
                    Code::REPOSITORY_CONFLICT,
                    "repository changed while acquiring SQLite coordination",
                );
            }
            fault.hit("repo-before-replace");
            match &replacement_for_write {
                Some(repository) => {
                    let bytes = repository.canonical_bytes();
                    if actual.as_deref() != Some(bytes.as_slice()) {
                        let directory = path_for_write.parent().unwrap();
                        fs::create_dir_all(directory)
                            .map_err(|error| io("mkdir repository org", error))?;
                        let temporary = directory.join(format!(
                            ".{}.{}.{}",
                            path_for_write.file_name().unwrap().to_string_lossy(),
                            std::process::id(),
                            now_nanos_unique()
                        ));
                        let mut file = OpenOptions::new()
                            .write(true)
                            .create_new(true)
                            .mode(0o600)
                            .open(&temporary)
                            .map_err(|error| io("create repository temp", error))?;
                        if let Err(error) = file.write_all(&bytes) {
                            return cleanup(&temporary, io("write repository", error));
                        }
                        if let Err(error) = file.sync_all() {
                            return cleanup(&temporary, io("fsync repository", error));
                        }
                        drop(file);
                        if let Err(error) = fs::rename(&temporary, &path_for_write) {
                            return cleanup(&temporary, io("rename repository", error));
                        }
                        fsync_dir(directory)?;
                    }
                }
                None => {
                    fs::remove_file(&path_for_write)
                        .map_err(|error| io("delete repository", error))?;
                    fsync_dir(path_for_write.parent().unwrap())?;
                }
            }
            fault.hit("repo-after-replace");
            crate::storage::Census::open_metadata(&self.root)?.projection_for(&org, &name)
        })?;
        Ok(replacement)
    }
    /// PRESENCE HINT ONLY. True says an object file exists at that id; it says nothing
    /// about its bytes. Resume must consult `record` or rehash.
    pub fn contains(&self, hex: &str) -> bool {
        self.object_path(hex).is_file()
    }

    // ------------------------------------------------------------ admission

    pub fn put_stream<R: Read>(
        &self,
        r: &mut R,
        expect: Option<&ObjectRef>,
        fault: &Fault,
    ) -> Result<Put> {
        self.put_stream_held(r, expect, fault, None)
    }

    pub fn put_stream_held<R: Read>(
        &self,
        r: &mut R,
        expect: Option<&ObjectRef>,
        fault: &Fault,
        operation: Option<&str>,
    ) -> Result<Put> {
        let catalog = crate::catalog::Catalog::open(&self.root)?;
        let _writer = crate::catalog::WriterGuard::acquire(&self.root)?;
        let tmp = self.root.join(".tensorfs/tmp").join(format!(
            "put-{}-{}",
            std::process::id(),
            now_nanos_unique()
        ));
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp)
            .map_err(|e| io(format!("create temp {}", tmp.display()), e))?;
        fault.hit("open");

        let mut h = Sha256::new();
        let mut buf = vec![0u8; BUF];
        let mut len: u64 = 0;
        let mut first = true;
        loop {
            let n = match r.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) => return cleanup(&tmp, io("read source", e)),
            };
            h.update(&buf[..n]);
            if let Err(e) = f.write_all(&buf[..n]) {
                return cleanup(&tmp, io("write temp", e));
            }
            len += n as u64;
            if first {
                first = false;
                fault.hit("partial");
            }
        }
        fault.hit("written");
        if let Err(e) = f.sync_all() {
            return cleanup(&tmp, io("fsync temp", e));
        }
        drop(f);
        fault.hit("fsync");

        let hex = sha256::hex(&h.finish());
        if let Some(want) = expect {
            if want.length != len {
                return cleanup(
                    &tmp,
                    Refusal {
                        code: Code::LENGTH_MISMATCH,
                        detail: format!("expected {} bytes, streamed {len}", want.length),
                    },
                );
            }
            if want.sha256 != hex {
                return cleanup(
                    &tmp,
                    Refusal {
                        code: Code::OBJECT_ID_MISMATCH,
                        detail: format!("expected sha256:{}, streamed sha256:{hex}", want.sha256),
                    },
                );
            }
        }
        if let Some(operation) = operation {
            catalog.hold(operation, &crate::storage::blob_key(&hex)?, len)?;
        }

        if let Err(e) = fs::set_permissions(&tmp, fs::Permissions::from_mode(0o444)) {
            return cleanup(&tmp, io("chmod 0444", e));
        }
        fault.hit("chmod");

        let dst = self.object_path(&hex);
        let dir = dst.parent().unwrap().to_path_buf();
        if let Err(e) = fs::create_dir_all(&dir) {
            return cleanup(&tmp, io("mkdir fan-out", e));
        }
        // No-clobber admission: link() refuses to replace an installed object.
        let admitted = match fs::hard_link(&tmp, &dst) {
            Ok(()) => true,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => false,
            Err(e) => return cleanup(&tmp, io(format!("link {}", dst.display()), e)),
        };
        fault.hit("linked");
        if admitted {
            fsync_dir(&dir)?;
        }
        let _ = fs::remove_file(&tmp);

        // A record is written only for bytes THIS process hashed. Losing the race means we
        // never saw the installed file's bytes, so we make no claim about them.
        if admitted {
            let md = stat(&dst)?;
            self.write_record(&VRecord::of(&hex, &md))?;
        }
        fault.hit("record");
        Ok(Put {
            obj: ObjectRef {
                sha256: hex,
                length: len,
            },
            admitted,
        })
    }

    pub fn put_file(&self, src: &Path, expect: Option<&ObjectRef>, fault: &Fault) -> Result<Put> {
        let mut f = File::open(src).map_err(|e| io(format!("open {}", src.display()), e))?;
        self.put_stream(&mut f, expect, fault)
    }

    /// Copy one exact object between stores through the same verified read and admission
    /// doors used everywhere else. Store layout never escapes this module: the source is
    /// opened once through `open_verified`, and the destination hashes, length-checks and
    /// publishes without clobbering through `put_stream`.
    pub fn transfer_object(&self, destination: &Store, want: &ObjectRef) -> Result<Put> {
        // A retried transfer streams NOTHING: the destination's own predicate
        // (`record_valid` — may these bytes be trusted without hashing them again?, the
        // same question `FetchPlan::of` asks) answers before any byte moves, so a retried
        // 99 GB store-to-store materialization is a stat, not a re-copy.
        if let Ok(rec) = destination.record_valid(&want.sha256) {
            if rec.length == want.length {
                return Ok(Put {
                    obj: want.clone(),
                    admitted: false,
                });
            }
            return refuse(
                Code::LENGTH_MISMATCH,
                format!(
                    "{} is {} bytes in the destination store, expected {}",
                    want.id(),
                    rec.length,
                    want.length
                ),
            );
        }
        let mut source = self.open_verified(&want.sha256)?;
        if source.len() != want.length {
            return refuse(
                Code::LENGTH_MISMATCH,
                format!(
                    "{} is {} bytes in the source store, expected {}",
                    want.id(),
                    source.len(),
                    want.length
                ),
            );
        }
        let put = destination.put_stream(&mut source, Some(want), &Fault::default())?;
        if !put.admitted {
            let installed = destination.open_verified(&want.sha256)?;
            if installed.len() != want.length {
                return refuse(
                    Code::LENGTH_MISMATCH,
                    format!(
                        "{} is {} bytes in the destination store, expected {}",
                        want.id(),
                        installed.len(),
                        want.length
                    ),
                );
            }
        }
        Ok(put)
    }

    pub fn transfer_manifest(&self, destination: &Store, want: &ObjectRef) -> Result<Put> {
        let manifest = self.read_manifest(want)?;
        let put = destination.put_manifest(&manifest)?;
        if put.obj != *want {
            return refuse(
                Code::OBJECT_ID_MISMATCH,
                "transferred manifest identity changed during admission",
            );
        }
        Ok(put)
    }

    // ------------------------------------------------------------ records

    fn write_record(&self, rec: &VRecord) -> Result<()> {
        crate::catalog::Catalog::open(&self.root)?.write_verification(rec)
    }

    fn record(&self, hex: &str) -> Option<VRecord> {
        crate::catalog::Catalog::open(&self.root)
            .and_then(|catalog| catalog.verification(hex))
            .ok()
            .flatten()
    }

    fn drop_record(&self, hex: &str) {
        if let Ok(catalog) = crate::catalog::Catalog::open(&self.root) {
            catalog.drop_verification(hex);
        }
    }

    /// The one question the read and resume paths ask: may these bytes be trusted without
    /// hashing them again?
    pub fn record_valid(&self, hex: &str) -> std::result::Result<VRecord, String> {
        let md = match stat(&self.object_path(hex)) {
            Ok(md) => md,
            Err(e) => return Err(e.detail),
        };
        let rec = self
            .record(hex)
            .ok_or_else(|| "no verification record".to_string())?;
        match rec.disagreement(hex, &md) {
            Some(why) => Err(why),
            None => Ok(rec),
        }
    }

    // ------------------------------------------------------------ reads

    /// Open for reading without following a symlink at the final component, and refuse
    /// anything that is not a regular file.
    pub fn open_nofollow(&self, hex: &str) -> Result<File> {
        let p = self.object_path(hex);
        let f = OpenOptions::new()
            .read(true)
            .custom_flags(O_NOFOLLOW)
            .open(&p)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    Refusal {
                        code: Code::OBJECT_ABSENT,
                        detail: format!("sha256:{hex} not in {}", self.root.display()),
                    }
                } else {
                    io(format!("open {}", p.display()), e)
                }
            })?;
        let md = f.metadata().map_err(|e| io("stat object", e))?;
        if !md.is_file() {
            return refuse(
                Code::NOT_REGULAR_FILE,
                format!("{} is not a regular file", p.display()),
            );
        }
        Ok(f)
    }

    /// A verified handle: trusted from a valid record, otherwise rehashed. Bytes that
    /// disagree with their id are quarantined, never returned.
    ///
    /// THE FILE THAT WAS VERIFIED IS THE FILE THAT IS READ. v1 verified a PATHNAME and then
    /// re-opened that pathname to read it — two resolutions of one name, with a window
    /// between them in which a same-length replacement passes every check the record binds
    /// (size, and an mtime the attacker controls). The handle is opened ONCE here; the
    /// fstat, the rehash and every subsequent read go through that one descriptor, which on
    /// Linux pins the inode. A rename over the path after this point does not reach these
    /// bytes at all — the swap is not caught, it is INEFFECTIVE.
    pub fn open_verified(&self, hex: &str) -> Result<VerifiedFile> {
        Ok(self.open_verified_reporting(hex)?.0)
    }

    /// `open_verified` plus the fact a caller may need to report: whether trusting these
    /// bytes cost a rehash (missing or disagreeing record) rather than a record lookup.
    pub fn open_verified_reporting(&self, hex: &str) -> Result<(VerifiedFile, bool)> {
        let (verdict, file) = self.verify_pinned(hex, false)?;
        let rehashed = !matches!(verdict, Verdict::Verified { rehashed: false });
        match (verdict, file) {
            (Verdict::Quarantined { to, why }, _) => refuse(
                Code::OBJECT_CORRUPT,
                format!("sha256:{hex} quarantined to {}: {why}", to.display()),
            ),
            (_, Some(mut f)) => {
                let len = f.metadata().map_err(|e| io("stat object", e))?.len();
                f.seek(SeekFrom::Start(0)).map_err(|e| io("rewind", e))?;
                Ok((
                    VerifiedFile {
                        file: f,
                        sha256: hex.to_string(),
                        len,
                    },
                    rehashed,
                ))
            }
            (v, None) => refuse(
                Code::OBJECT_CORRUPT,
                format!("sha256:{hex}: {v:?} left no verified descriptor"),
            ),
        }
    }

    /// Bounded range read over verified bytes. Out-of-bounds refuses instead of clamping.
    pub fn read_range(&self, hex: &str, off: u64, len: u64) -> Result<Vec<u8>> {
        let mut vf = self.open_verified(hex)?;
        let size = vf.len;
        let end = off.checked_add(len).ok_or(Refusal {
            code: Code::RANGE_BOUNDS,
            detail: format!("offset {off} + length {len} overflows"),
        })?;
        if end > size {
            return refuse(
                Code::RANGE_BOUNDS,
                format!("range [{off}, {end}) leaves object of {size} bytes"),
            );
        }
        vf.file
            .seek(SeekFrom::Start(off))
            .map_err(|e| io("seek", e))?;
        let mut out = vec![0u8; len as usize];
        vf.file
            .read_exact(&mut out)
            .map_err(|e| io("read range", e))?;
        Ok(out)
    }

    /// Stream a verified object out. The read path does not hash; the pinned verification
    /// decided already, and it decided about THIS descriptor.
    pub fn read_into<W: Write>(&self, hex: &str, w: &mut W) -> Result<u64> {
        let mut vf = self.open_verified(hex)?;
        let mut buf = vec![0u8; BUF];
        let mut n = 0u64;
        loop {
            let k = vf.file.read(&mut buf).map_err(|e| io("read object", e))?;
            if k == 0 {
                return Ok(n);
            }
            w.write_all(&buf[..k]).map_err(|e| io("write out", e))?;
            n += k as u64;
        }
    }

    // ------------------------------------------------------------ verify / quarantine

    /// A valid verification record IS the answer; a rehash happens exactly when the record
    /// is missing or any bound fact moved. That rehash is CORRECTNESS, not paranoia, and it
    /// is not optional.
    ///
    /// PARANOID MODE IS DELETED (owner ruling 2026-08-25). The per-read full-rehash option
    /// was 16.6x slower, had no consumer, and gave integrity checking a second home.
    /// `scrub` — the rate-knobbed whole-store sweep — is now the ONLY deliberate
    /// full-rehash door, and it reaches the forced path through `rehash` below rather than
    /// through a flag anyone can pass on a read.
    pub fn verify(&self, hex: &str) -> Result<Verdict> {
        Ok(self.verify_pinned(hex, false)?.0)
    }

    /// The scrub's door: rehash this object whatever its record says. Not reachable from
    /// the read path, the CLI, or the wheel — only from `scrub`.
    fn rehash(&self, hex: &str) -> Result<Verdict> {
        Ok(self.verify_pinned(hex, true)?.0)
    }

    /// Verify through ONE descriptor and hand it back with the verdict.
    ///
    /// Every fact this consults comes from the open fd — `fstat` for the record binding,
    /// the fd itself for the rehash — so the pathname is resolved exactly once. The
    /// returned `File` is the object those facts are about; `None` only on quarantine,
    /// where there is nothing to hand out.
    fn verify_pinned(&self, hex: &str, force: bool) -> Result<(Verdict, Option<File>)> {
        let mut f = self.open_nofollow(hex)?;
        let md = f.metadata().map_err(|e| io("fstat object", e))?;
        let disagreement = match self.record(hex) {
            None => Some("no verification record".to_string()),
            Some(rec) => rec.disagreement(hex, &md),
        };
        if disagreement.is_none() && !force {
            return Ok((Verdict::Verified { rehashed: false }, Some(f)));
        }
        let got = hash_fd(&mut f)?;
        if got != hex {
            let why = disagreement.unwrap_or_else(|| "scrub rehash".to_string());
            // Take the NAME: whatever is at the path now must stop satisfying reads. The
            // descriptor is dropped with `f` — corrupt bytes are never handed back.
            let to = self.quarantine(hex)?;
            return Ok((
                Verdict::Quarantined {
                    to,
                    why: format!("{why}; rehash gave sha256:{got}"),
                },
                None,
            ));
        }
        let md = f.metadata().map_err(|e| io("fstat object", e))?;
        self.write_record(&VRecord::of(hex, &md))?;
        Ok((
            match disagreement {
                None => Verdict::Verified { rehashed: true },
                Some(why) => Verdict::Invalidated { why },
            },
            Some(f),
        ))
    }

    pub fn hash_object(&self, hex: &str) -> Result<String> {
        hash_fd(&mut self.open_nofollow(hex)?)
    }

    /// Take the NAME first: rename out of `blobs/` so exactly one remover wins and the
    /// bad bytes can never satisfy another read.
    pub fn quarantine(&self, hex: &str) -> Result<PathBuf> {
        let dir = self.root.join(".tensorfs/quarantine");
        fs::create_dir_all(&dir).map_err(|e| io("mkdir quarantine", e))?;
        let mut to = dir.join(hex);
        let mut n = 1;
        while to.exists() {
            to = dir.join(format!("{hex}.{n}"));
            n += 1;
        }
        match fs::rename(self.object_path(hex), &to) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // RACING QUARANTINERS: the rename loser finds the name already taken.
                // What quarantine promises — nothing at this id satisfies another read —
                // is already true, so losing the race is the promise kept, not IO_FAILED;
                // `open_verified` then reports the honest OBJECT_CORRUPT.
                self.drop_record(hex);
                let winner = dir.join(hex);
                return Ok(if winner.exists() { winner } else { to });
            }
            Err(e) => return Err(io("quarantine rename", e)),
        }
        self.drop_record(hex);
        Ok(to)
    }

    pub fn quarantine_manifest(&self, hex: &str) -> Result<PathBuf> {
        crate::ids::hex64("manifest", hex)?;
        let _writer = crate::catalog::WriterGuard::acquire(&self.root)?;
        let directory = self.root.join(".tensorfs/quarantine/manifests");
        fs::create_dir_all(&directory).map_err(|error| io("mkdir manifest quarantine", error))?;
        let mut destination = directory.join(hex);
        let mut suffix = 1u64;
        while destination.exists() {
            destination = directory.join(format!("{hex}.{suffix}"));
            suffix += 1;
        }
        fs::rename(self.manifest_path(hex), &destination)
            .map_err(|error| io("quarantine manifest", error))?;
        fsync_dir(&directory)?;
        Ok(destination)
    }

    // ------------------------------------------------------------ scrub / reap

    pub fn objects(&self) -> Result<Vec<String>> {
        let mut out = Vec::new();
        let base = self.root.join("blobs");
        for a in read_dir(&base)? {
            for b in read_dir(&a)? {
                for o in read_dir(&b)? {
                    if let Some(n) = o.file_name().and_then(|s| s.to_str()) {
                        if hex64("object", n).is_ok() {
                            out.push(n.to_string());
                        }
                    }
                }
            }
        }
        out.sort();
        Ok(out)
    }

    /// ONE knob, and since the 2026-08-25 owner ruling the ONE deliberate full-rehash door.
    /// `rate` is the fraction of objects rehashed this pass, chosen deterministically from
    /// (pass, id). Everything else gets its record checked, and a record that disagrees is
    /// rehashed anyway.
    pub fn scrub(&self, rate: f64, pass: u64) -> Result<Scrub> {
        let mut s = Scrub::default();
        for hex in self.objects()? {
            s.scanned += 1;
            let sampled = sample(pass, &hex) < rate;
            if sampled {
                s.sampled += 1;
            }
            let size = stat(&self.object_path(&hex)).map(|m| m.len()).unwrap_or(0);
            match if sampled {
                self.rehash(&hex)?
            } else {
                self.verify(&hex)?
            } {
                Verdict::Verified { rehashed } => {
                    if rehashed {
                        s.rehashed += 1;
                        s.bytes_hashed += size;
                    }
                }
                Verdict::Invalidated { .. } => {
                    s.rehashed += 1;
                    s.bytes_hashed += size;
                }
                Verdict::Quarantined { .. } => {
                    s.rehashed += 1;
                    s.bytes_hashed += size;
                    s.quarantined += 1;
                }
            }
        }
        Ok(s)
    }

    /// Age-bounded reap of abandoned admission temps. A temp is scratch by construction:
    /// nothing references it, so removing an old one can never lose an object.
    pub fn reap(&self, max_age_secs: u64) -> Result<(usize, usize)> {
        let (mut gone, mut kept) = (0, 0);
        for p in read_dir(&self.root.join(".tensorfs/tmp"))? {
            let age = fs::metadata(&p)
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| SystemTime::now().duration_since(t).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            if age >= max_age_secs && fs::remove_file(&p).is_ok() {
                gone += 1;
            } else {
                kept += 1;
            }
        }
        Ok((gone, kept))
    }
}

// ---------------------------------------------------------------- helpers

fn cleanup<T>(tmp: &Path, e: Refusal) -> Result<T> {
    let _ = fs::set_permissions(tmp, fs::Permissions::from_mode(0o600));
    let _ = fs::remove_file(tmp);
    Err(e)
}

/// Hash an OPEN object from the top. Takes the descriptor, never the name — the caller has
/// already resolved the path and everything after it is about this inode.
fn hash_fd(f: &mut File) -> Result<String> {
    f.seek(SeekFrom::Start(0)).map_err(|e| io("rewind", e))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; BUF];
    loop {
        let n = f.read(&mut buf).map_err(|e| io("read object", e))?;
        if n == 0 {
            return Ok(sha256::hex(&h.finish()));
        }
        h.update(&buf[..n]);
    }
}

fn stat(p: &Path) -> Result<fs::Metadata> {
    let md = fs::symlink_metadata(p).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            Refusal {
                code: Code::OBJECT_ABSENT,
                detail: format!("{} absent", p.display()),
            }
        } else {
            io(format!("stat {}", p.display()), e)
        }
    })?;
    if !md.is_file() {
        return refuse(
            Code::NOT_REGULAR_FILE,
            format!("{} is not a regular file", p.display()),
        );
    }
    Ok(md)
}

fn read_dir(p: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let rd = match fs::read_dir(p) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(io(format!("read_dir {}", p.display()), e)),
    };
    for e in rd {
        out.push(e.map_err(|e| io("dir entry", e))?.path());
    }
    out.sort();
    Ok(out)
}

fn fsync_dir(dir: &Path) -> Result<()> {
    File::open(dir)
        .and_then(|d| d.sync_all())
        .map_err(|e| io(format!("fsync dir {}", dir.display()), e))
}

fn now_nanos_unique() -> u128 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
        .wrapping_add(N.fetch_add(1, Ordering::Relaxed) as u128)
}

/// Deterministic [0,1) sample position for (pass, object) — the one scrub knob's input.
fn sample(pass: u64, hex: &str) -> f64 {
    let d = sha256::digest(format!("{pass}:{hex}").as_bytes());
    let v = u32::from_be_bytes([d[0], d[1], d[2], d[3]]);
    v as f64 / u32::MAX as f64
}

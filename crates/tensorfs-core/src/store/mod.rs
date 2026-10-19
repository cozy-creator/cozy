//! The store (design §4): one host, the filesystem only.
//!
//! `<root>/{format, blobs/<aa>/<bb>/<sha256>, refs/<org>/<name>.json, pins/<id>.json, tmp/}`
//!
//! - Objects are trees, cozytensors headers, chunks, assets and repo files. A ref names a
//!   model's releases of trees; a pin keeps trees or headers. A cozytensors is opened by its
//!   header ([`Store::checkpoint`]) or a tree's variant ([`Store::variant`]); a plain folder
//!   enters by [`Store::admit_folder`] and leaves by [`Store::snapshot`].
//! - One door admits objects ([`Store::admit`], [`Store::put`]): streamed into `tmp/`, hashed,
//!   checked, flushed per [`Durability`] and renamed into `blobs/`. An object is verified there
//!   once and trusted after; [`Store::verify`] rehashes on demand.
//! - Refs name models; pins are the one retention record (`records.rs`).
//! - GC marks from refs and pins and sweeps unmarked objects older than the grace (`gc.rs`). So
//!   whatever newly references an object makes it young first: `put` and `reuse` touch it, and
//!   `pin` and `update_ref` touch their new roots' closures.
//! - Processes share a store with no leases, markers or boot-time rehash: a torn write never has
//!   a name, and an open descriptor outlives an unlink.

mod checkpoint;
mod folder;
mod gc;
mod records;
pub(crate) mod retention;

pub use checkpoint::{wanted, Checkpoint, ChunkRead, Piece, Placed, ReadPlan};
pub use gc::{Collected, PinUsage, Usage};
pub(crate) use records::is_name;
pub use records::Pin;

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use platform::durable::{self, DiskClass, FileId};

use crate::err::{Code, Refusal};
use crate::fetch;
use crate::format::{self, ObjectRef};
use crate::sha256::{hex, Sha256};

/// [`crate::STORE_FORMAT`] as the store's `format` record.
const FORMAT: &str = "tensorfs.store/3";
const LAYOUT: [&str; 4] = ["blobs", "refs", "pins", "tmp"];

/// How long an unreferenced object outlives its last write or reuse.
pub const GRACE: Duration = Duration::from_secs(3600);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Durability {
    /// Each object and record is flushed before it is named (local machines).
    Durable,
    /// Nothing is flushed until one `syncfs`, which runs before a record names objects this
    /// handle admitted (pods; owner ruling 2026-09-27, tfs-122). Linux only, else `Durable`.
    Ephemeral,
}

impl Durability {
    /// What the filesystem under `path` calls for: ephemeral on tmpfs and overlay, else durable.
    pub fn of(path: &Path) -> Durability {
        match DiskClass::of(path) {
            DiskClass::Ephemeral => Durability::Ephemeral.supported(),
            _ => Durability::Durable,
        }
    }

    fn supported(self) -> Durability {
        if cfg!(target_os = "linux") {
            self
        } else {
            Durability::Durable
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{op} {}: {source}", path.display())]
    Io {
        op: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("{op} {}: the disk is full", path.display())]
    Full { op: &'static str, path: PathBuf },
    #[error("object {0} is absent")]
    Absent(String),
    #[error("expected {} ({} bytes), got {got}", expected.sha256, expected.length)]
    Mismatch { expected: ObjectRef, got: String },
    #[error("{}: {why}", path.display())]
    Invalid { path: PathBuf, why: String },
    /// A document, or a tensor written or read, breaks cozytensors/1.
    #[error(transparent)]
    Format(#[from] format::Error),
}

impl Error {
    pub fn code(&self) -> &'static str {
        match self {
            Error::Io { .. } => "io",
            Error::Full { .. } => "full",
            Error::Absent(_) => "absent",
            Error::Mismatch { .. } => "mismatch",
            Error::Invalid { .. } => "invalid",
            Error::Format(error) => error.code(),
        }
    }
}

impl Error {
    /// The refusal the fetch and import layers speak.
    pub fn refusal(self) -> Refusal {
        let code = match &self {
            Error::Io { .. } => Code::IO_FAILED,
            Error::Full { .. } => Code::CAPACITY_EXHAUSTED,
            Error::Absent(_) => Code::OBJECT_ABSENT,
            Error::Mismatch { .. } => Code::OBJECT_ID_MISMATCH,
            Error::Invalid { .. } => Code::UNKNOWN_FORMAT,
            Error::Format(format::Error::UnknownScheme(_)) => Code::UNKNOWN_ENCODING,
            Error::Format(format::Error::Decode(_) | format::Error::Encode(_)) => Code::ENCODING_MISMATCH,
            Error::Format(format::Error::Format { .. } | format::Error::PreCut) => {
                Code::UNKNOWN_FORMAT
            }
            Error::Format(format::Error::Invalid(_)) => Code::TENSOR_SCHEMA_MISMATCH,
        };
        Refusal {
            code,
            detail: self.to_string(),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn fail(op: &'static str, path: impl Into<PathBuf>) -> impl FnOnce(io::Error) -> Error {
    let path = path.into();
    move |source| match source.kind() {
        io::ErrorKind::StorageFull | io::ErrorKind::QuotaExceeded => Error::Full { op, path },
        _ => Error::Io { op, path, source },
    }
}

fn platform(
    op: &'static str,
    path: impl Into<PathBuf>,
) -> impl FnOnce(platform::Error) -> Error {
    let path = path.into();
    move |error| fail(op, path)(error.into())
}

pub(crate) fn invalid(path: impl Into<PathBuf>, why: impl ToString) -> Error {
    Error::Invalid {
        path: path.into(),
        why: why.to_string(),
    }
}

fn is_digest(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// `family/major[.minor]`: the same family and major as `want`.
pub(crate) fn same_major(tag: &str, want: &str) -> bool {
    fn split(tag: &str) -> Option<(&str, &str)> {
        let (family, version) = tag.split_once('/')?;
        Some((family, version.split('.').next()?))
    }
    split(tag).is_some() && split(tag) == split(want)
}

pub(crate) fn same_family(tag: &str, want: &str) -> bool {
    tag.split_once('/').map(|t| t.0) == want.split_once('/').map(|w| w.0)
}

pub(crate) fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

#[derive(Clone, Debug)]
pub struct Store {
    root: PathBuf,
    durability: Durability,
    grace: Duration,
    /// This handle admitted objects that no `syncfs` covers yet (ephemeral only).
    unsynced: Arc<AtomicBool>,
}

impl Store {
    /// Opens the store at `root`, creating it when `root` is absent or empty.
    pub fn open(root: impl Into<PathBuf>) -> Result<Store> {
        let root = root.into();
        fs::create_dir_all(&root).map_err(fail("create", &root))?;
        let record = root.join("format");
        match fs::read(&record) {
            Ok(bytes) => {
                #[derive(serde::Deserialize)]
                struct Record {
                    format: String,
                }
                match serde_json::from_slice::<Record>(&bytes) {
                    Ok(found) if same_major(&found.format, FORMAT) => {}
                    Ok(found) if same_family(&found.format, FORMAT) => {
                        let why = format!(
                            "is a {} store from an older TensorFS, which this one cannot read: \
                             remove it and pull again (tensorfs.store/3 holds trees and D10 headers)",
                            found.format
                        );
                        return Err(invalid(&record, why));
                    }
                    _ => {
                        let found = String::from_utf8_lossy(&bytes);
                        return Err(invalid(&record, format!("is {found:?}, not {FORMAT}")));
                    }
                }
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                for entry in fs::read_dir(&root).map_err(fail("list", &root))? {
                    let name = entry.map_err(fail("list", &root))?.file_name();
                    let name = name.to_string_lossy();
                    if !LAYOUT.contains(&&*name) && !name.starts_with(".format") {
                        return Err(invalid(&root, "is neither empty nor a TensorFS store"));
                    }
                }
                let bytes = format!("{{\"format\":\"{FORMAT}\"}}");
                durable::replace(&record, bytes.as_bytes(), 0o644)
                    .map_err(platform("write", &record))?;
            }
            Err(error) => return Err(fail("read", &record)(error)),
        }
        for dir in LAYOUT {
            let path = root.join(dir);
            fs::create_dir_all(&path).map_err(fail("create", &path))?;
        }
        Ok(Store {
            durability: Durability::of(&root),
            grace: GRACE,
            root,
            unsynced: Arc::default(),
        })
    }

    pub fn with_durability(mut self, durability: Durability) -> Store {
        self.durability = durability.supported();
        self
    }

    pub fn with_grace(mut self, grace: Duration) -> Store {
        self.grace = grace;
        self
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn durability(&self) -> Durability {
        self.durability
    }

    /// How long an unreferenced object outlives its last write or reuse.
    pub fn grace(&self) -> Duration {
        self.grace
    }

    /// Where `object` lives. A malformed digest has no path.
    pub fn path(&self, object: &ObjectRef) -> Result<PathBuf> {
        self.blob(&object.sha256)
    }

    fn blob(&self, sha256: &str) -> Result<PathBuf> {
        if !is_digest(sha256) {
            return Err(invalid(
                &self.root,
                format!("{sha256:?} is not a sha256 digest"),
            ));
        }
        Ok(self.object_path(sha256))
    }

    /// Where the object `sha256`, a digest already checked, lives.
    pub(crate) fn object_path(&self, sha256: &str) -> PathBuf {
        (self.root.join("blobs").join(&sha256[..2]).join(&sha256[2..4])).join(sha256)
    }

    /// The one door. Bytes written to the [`Admission`] become an object at
    /// [`Admission::finish`]; with `expect`, any other bytes are refused.
    pub fn admit(&self, expect: Option<&ObjectRef>) -> Result<Admission> {
        if let Some(expect) = expect {
            self.path(expect)?;
        }
        let (temp, file) = self.temp("put")?;
        Ok(Admission {
            store: self.clone(),
            expect: expect.cloned(),
            temp,
            file: Some(file),
            hash: Sha256::new(),
            length: 0,
            over: false,
        })
    }

    /// Admits everything `src` yields. With `expect`, an object already here is reused without
    /// reading `src`.
    pub fn put(&self, mut src: impl Read, expect: Option<&ObjectRef>) -> Result<ObjectRef> {
        if let Some(expect) = expect {
            if self.reuse(expect)? {
                return Ok(expect.clone());
            }
        }
        let mut admission = self.admit(expect)?;
        let mut buffer = vec![0; 1 << 20];
        loop {
            let n = match src.read(&mut buffer) {
                Ok(0) => break,
                Ok(n) => n,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(fail("read source for", &admission.temp)(error)),
            };
            if let Err(error) = admission.write_all(&buffer[..n]) {
                return Err(admission.refusal(error));
            }
        }
        admission.finish()
    }

    pub fn put_bytes(&self, bytes: &[u8]) -> Result<ObjectRef> {
        self.put(bytes, None)
    }

    /// True when `object` is here, and then it is young for GC: call it before referencing an
    /// object this process did not just put.
    pub fn reuse(&self, object: &ObjectRef) -> Result<bool> {
        let Some(file) = self.hold(&object.sha256)? else {
            return Ok(false);
        };
        let length = file
            .metadata()
            .map_err(fail("stat", self.path(object)?))?
            .len();
        if length != object.length {
            return Err(Error::Mismatch {
                expected: object.clone(),
                got: format!("{length} bytes on disk"),
            });
        }
        Ok(true)
    }

    /// Opens `object` to read. Its bytes were verified when admitted.
    pub fn file(&self, object: &ObjectRef) -> Result<File> {
        let path = self.path(object)?;
        // One a sweep is moving right now still reads, from GC's trash.
        let opened = File::open(&path)
            .or_else(|_| File::open(self.trash(&object.sha256)))
            .or_else(|_| File::open(&path));
        let file = match opened {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Err(Error::Absent(object.sha256.clone()))
            }
            Err(error) => return Err(fail("open", &path)(error)),
        };
        let length = file.metadata().map_err(fail("stat", &path))?.len();
        if length != object.length {
            return Err(Error::Mismatch {
                expected: object.clone(),
                got: format!("{length} bytes on disk"),
            });
        }
        Ok(file)
    }

    pub fn read(&self, object: &ObjectRef) -> Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(object.length as usize);
        let path = self.path(object)?;
        self.file(object)?
            .read_to_end(&mut bytes)
            .map_err(fail("read", &path))?;
        Ok(bytes)
    }

    /// Rehashes `object`. A copy that no longer matches is removed, to be fetched again.
    pub fn verify(&self, object: &ObjectRef) -> Result<bool> {
        let path = self.path(object)?;
        let mut file = match File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(fail("open", &path)(error)),
        };
        let mut hash = Sha256::new();
        let mut length = 0u64;
        let mut buffer = vec![0; 1 << 20];
        loop {
            match file.read(&mut buffer) {
                Ok(0) => break,
                Ok(n) => {
                    hash.update(&buffer[..n]);
                    length += n as u64;
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(fail("read", &path)(error)),
            }
        }
        if length == object.length && hex(&hash.finish()) == object.sha256 {
            return Ok(true);
        }
        match fs::remove_file(&path) {
            Ok(()) => Ok(false),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(fail("remove", &path)(error)),
        }
    }

    /// Makes every object this handle admitted durable. Ephemeral: one `syncfs`, which the
    /// next record commit also runs. Durable objects already are.
    pub fn sync(&self) -> Result<()> {
        if !self.unsynced.swap(false, Ordering::SeqCst) {
            return Ok(());
        }
        let synced = syncfs(&self.root);
        if synced.is_err() {
            self.unsynced.store(true, Ordering::SeqCst);
        }
        synced.map_err(fail("syncfs", &self.root))
    }

    pub(crate) fn flushes(&self) -> bool {
        self.durability == Durability::Durable
    }

    fn temp(&self, kind: &str) -> Result<(PathBuf, File)> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = self
                .root
                .join("tmp")
                .join(format!("{kind}-{}-{n}", std::process::id()));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => return Ok((path, file)),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(fail("create", &path)(error)),
            }
        }
    }

    /// Where a sweep moves an object before it decides (`gc.rs`).
    fn trash(&self, sha256: &str) -> PathBuf {
        self.root.join("tmp").join("gc").join(sha256)
    }

    /// Opens object `sha256` and makes it young; `None` when it is gone. One a sweep is moving
    /// right now is taken back from GC's trash. It is held only once, after the touch, its name
    /// names it again: a sweep that moves it from then on finds it young and puts it back.
    fn hold(&self, sha256: &str) -> Result<Option<File>> {
        let blob = self.blob(sha256)?;
        let trash = self.trash(sha256);
        loop {
            let file = match touch(&blob)? {
                Some(file) => file,
                None => match touch(&trash)? {
                    Some(file) => match fs::rename(&trash, &blob) {
                        Err(error) if error.kind() != io::ErrorKind::NotFound => {
                            return Err(fail("restore", &blob)(error))
                        }
                        _ => file,
                    },
                    None => match touch(&blob)? {
                        Some(file) => file,
                        None => return Ok(None),
                    },
                },
            };
            let named = match FileId::at(&blob).map_err(io::Error::from) {
                Ok(id) => FileId::of(&file).map_err(platform("stat", &blob))? == id,
                Err(error) if error.kind() == io::ErrorKind::NotFound => false,
                Err(error) => return Err(fail("stat", &blob)(error)),
            };
            if named {
                return Ok(Some(file));
            }
        }
    }

    /// Names a finished temp `blob`, unless a copy is there already, which is kept young.
    fn install(&self, temp: &Path, blob: &Path, sha256: &str) -> Result<()> {
        if self.hold(sha256)?.is_some() {
            let _ = fs::remove_file(temp);
            return Ok(());
        }
        let shard = blob.parent().expect("a blob has a shard");
        let created = !shard.is_dir();
        if created {
            fs::create_dir_all(shard).map_err(fail("create", shard))?;
        }
        fs::rename(temp, blob).map_err(fail("rename into", blob))?;
        fault::point("put.renamed");
        if !self.flushes() {
            self.unsynced.store(true, Ordering::SeqCst);
            return Ok(());
        }
        if created {
            let parent = shard.parent().expect("a shard has a parent");
            sync_dir(parent.parent().expect("blobs/<aa> has a parent"))?;
            sync_dir(parent)?;
        }
        sync_dir(shard)
    }

    /// Replaces the record at `path` by a temp and a rename; `flush` makes it durable first.
    fn commit(&self, path: &Path, bytes: &[u8], flush: bool) -> Result<()> {
        let flush = flush && self.flushes();
        let (temp, mut file) = self.temp("record")?;
        let written = file
            .write_all(bytes)
            .and_then(|()| if flush { file.sync_data() } else { Ok(()) });
        drop(file);
        if let Err(error) = written {
            let _ = fs::remove_file(&temp);
            return Err(fail("write", &temp)(error));
        }
        fault::point("record.written");
        let dir = path.parent().expect("a record has a directory");
        let created = !dir.is_dir();
        if created {
            fs::create_dir_all(dir).map_err(fail("create", dir))?;
        }
        if let Err(error) = fs::rename(&temp, path) {
            let _ = fs::remove_file(&temp);
            return Err(fail("rename into", path)(error));
        }
        fault::point("record.renamed");
        if flush {
            if created {
                sync_dir(dir.parent().expect("a record directory has a parent"))?;
            }
            sync_dir(dir)?;
        }
        Ok(())
    }

    fn remove_record(&self, path: &Path) -> Result<bool> {
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(fail("remove", path)(error)),
        }
        if self.flushes() {
            sync_dir(path.parent().expect("a record has a directory"))?;
        }
        Ok(true)
    }
}

/// Bytes on their way into the store. Dropped unfinished, it leaves nothing behind.
pub struct Admission {
    store: Store,
    expect: Option<ObjectRef>,
    temp: PathBuf,
    file: Option<File>,
    hash: Sha256,
    length: u64,
    over: bool,
}

impl Admission {
    /// Hashes, checks, flushes and names the bytes written.
    pub fn finish(mut self) -> Result<ObjectRef> {
        let got = ObjectRef {
            length: self.length,
            sha256: hex(&std::mem::take(&mut self.hash).finish()),
        };
        if let Some(expect) = &self.expect {
            if *expect != got {
                return Err(Error::Mismatch {
                    expected: expect.clone(),
                    got: format!("{} ({} bytes)", got.sha256, got.length),
                });
            }
        }
        let file = self
            .file
            .take()
            .expect("an admission has its file until it finishes");
        if self.store.flushes() {
            file.sync_data().map_err(fail("sync", &self.temp))?;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(fs::Permissions::from_mode(0o444))
                .map_err(fail("protect", &self.temp))?;
        }
        drop(file);
        fault::point("put.written");
        let blob = self.store.blob(&got.sha256)?;
        self.store.install(&self.temp, &blob, &got.sha256)?;
        Ok(got)
    }

    fn refusal(&self, error: io::Error) -> Error {
        match (&self.expect, self.over) {
            (Some(expect), true) => Error::Mismatch {
                expected: expect.clone(),
                got: "more bytes".into(),
            },
            _ => fail("write", &self.temp)(error),
        }
    }
}

impl Write for Admission {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if let Some(expect) = &self.expect {
            if self.length + bytes.len() as u64 > expect.length {
                self.over = true;
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "longer than the expected object",
                ));
            }
        }
        let file = self
            .file
            .as_mut()
            .expect("an admission has its file until it finishes");
        let n = file.write(bytes)?;
        self.hash.update(&bytes[..n]);
        self.length += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Drop for Admission {
    fn drop(&mut self) {
        self.file.take();
        let _ = fs::remove_file(&self.temp);
    }
}

pub(crate) fn sync_dir(dir: &Path) -> Result<()> {
    durable::sync_dir(dir).map_err(platform("sync directory", dir))
}

/// The fetcher admits through the store's one door (design §5).
impl fetch::ObjectStore for Store {
    fn has(&self, object: &ObjectRef) -> bool {
        self.reuse(object).unwrap_or(false)
    }

    fn open(&self, object: &ObjectRef) -> io::Result<File> {
        self.file(object).map_err(|error| match error {
            Error::Absent(_) => io::Error::new(io::ErrorKind::NotFound, error.to_string()),
            Error::Io { source, .. } => source,
            other => io::Error::other(other.to_string()),
        })
    }

    fn admit(&self, object: &ObjectRef) -> crate::err::Result<Box<dyn fetch::Admission>> {
        let admission = Store::admit(self, Some(object)).map_err(Error::refusal)?;
        Ok(Box::new(admission))
    }
}

impl fetch::Admission for Admission {
    fn write(&mut self, bytes: &[u8]) -> crate::err::Result<()> {
        self.write_all(bytes)
            .map_err(|error| self.refusal(error).refusal())
    }

    fn commit(self: Box<Self>) -> crate::err::Result<()> {
        self.finish().map(drop).map_err(Error::refusal)
    }
}

/// Makes an existing file young for GC; `None` when it is absent.
fn touch(path: &Path) -> Result<Option<File>> {
    #[cfg(windows)]
    let opened = {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_WRITE_ATTRIBUTES: u32 = 0x100;
        OpenOptions::new()
            .access_mode(FILE_WRITE_ATTRIBUTES | 0x80000000)
            .open(path)
    };
    #[cfg(not(windows))]
    let opened = File::open(path);
    let file = match opened {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(fail("open", path)(error)),
    };
    file.set_modified(SystemTime::now())
        .map_err(fail("touch", path))?;
    Ok(Some(file))
}

#[cfg(target_os = "linux")]
fn syncfs(root: &Path) -> io::Result<()> {
    rustix::fs::syncfs(File::open(root)?).map_err(io::Error::from)
}

#[cfg(not(target_os = "linux"))]
fn syncfs(_: &Path) -> io::Result<()> {
    Ok(())
}

/// Kill points for crash tests (feature `faults`). An armed point fires once: it prints
/// `FAULT <point>` and waits for a line on stdin, so the test can SIGKILL the process exactly
/// there or let it go on.
#[cfg(feature = "faults")]
pub mod fault {
    use std::io::{BufRead, Write};
    use std::sync::{Mutex, PoisonError};

    static ARMED: Mutex<Option<String>> = Mutex::new(None);

    pub fn arm(point: &str) {
        *ARMED.lock().unwrap_or_else(PoisonError::into_inner) = Some(point.to_owned());
    }

    pub(crate) fn point(name: &str) {
        {
            let mut armed = ARMED.lock().unwrap_or_else(PoisonError::into_inner);
            if armed.as_deref() != Some(name) {
                return;
            }
            armed.take();
        }
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "FAULT {name}");
        let _ = out.flush();
        drop(out);
        let _ = std::io::stdin().lock().read_line(&mut String::new());
    }
}

#[cfg(not(feature = "faults"))]
pub(crate) mod fault {
    #[inline(always)]
    pub(crate) fn point(_: &str) {}
}

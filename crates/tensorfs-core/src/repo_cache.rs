//! Optional immutable cache for repository manifests and blobs.
//!
//! This is deliberately not a [`Store`]: it has no repositories, SQLite, holds,
//! operations, verification records, or garbage collector. Every hit is checked against
//! the caller's exact digest and length while bytes move into an ordinary local Store.
//! Cache failure is weather, never authority.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use crate::err::{Code, Result};
use crate::ids::{hex64, Doc, ObjectRef};
use crate::manifest::Manifest;
use crate::sha256::{self, Sha256};
use crate::store::{self, Fault, Store};

/// The two immutable repository object namespaces a cache may contain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheKind {
    Blob,
    Manifest,
}

impl CacheKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "blob" => Some(Self::Blob),
            "manifest" => Some(Self::Manifest),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheRead {
    Hit,
    Missing,
    Corrupt,
    Unavailable,
}

impl CacheRead {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hit => "hit",
            Self::Missing => "missing",
            Self::Corrupt => "corrupt",
            Self::Unavailable => "unavailable",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheWrite {
    Stored,
    Present,
    Unavailable,
}

impl CacheWrite {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stored => "stored",
            Self::Present => "present",
            Self::Unavailable => "unavailable",
        }
    }
}

#[derive(Debug, Clone)]
pub struct RepoObjectCache {
    root: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SoftFailure {
    Missing,
    Corrupt,
    Unavailable,
}

impl RepoObjectCache {
    /// Name an optional cache. Construction performs no I/O and creates no metadata.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The one cross-language layout contract. `digest` is bare lowercase SHA-256.
    pub fn path(&self, kind: CacheKind, digest: &str) -> Result<PathBuf> {
        let digest = hex64("repo cache object", digest)?;
        let namespace = match kind {
            CacheKind::Blob => "blobs",
            CacheKind::Manifest => "manifests",
        };
        let mut path = self
            .root
            .join(namespace)
            .join(&digest[..2])
            .join(&digest[2..4])
            .join(&digest);
        if kind == CacheKind::Manifest {
            path.set_extension("json");
        }
        Ok(path)
    }

    /// Stream a cache hit through the local Store's ordinary verified admission path.
    pub fn admit(
        &self,
        destination: &Store,
        kind: CacheKind,
        want: &ObjectRef,
    ) -> Result<CacheRead> {
        let path = self.path(kind, &want.sha256)?;
        match kind {
            CacheKind::Blob => {
                let mut source = match open_exact(&path, want.length) {
                    Ok(source) => source,
                    Err(error) => return Ok(read_status(error)),
                };
                match destination.put_stream(&mut source, Some(want), &Fault::default()) {
                    Ok(put) if put.admitted => Ok(CacheRead::Hit),
                    Ok(_) => match destination.open_verified(&want.sha256) {
                        Ok(file) if file.len() == want.length => Ok(CacheRead::Hit),
                        Ok(_) | Err(_) => Ok(CacheRead::Unavailable),
                    },
                    Err(error)
                        if matches!(
                            error.code,
                            Code::OBJECT_ID_MISMATCH | Code::LENGTH_MISMATCH | Code::OBJECT_CORRUPT
                        ) =>
                    {
                        Ok(CacheRead::Corrupt)
                    }
                    Err(_) => Ok(CacheRead::Unavailable),
                }
            }
            CacheKind::Manifest => {
                let manifest = match read_manifest(&path, want) {
                    Ok(manifest) => manifest,
                    Err(error) => return Ok(read_status(error)),
                };
                match destination.put_manifest(&manifest) {
                    Ok(put) if put.obj == *want => Ok(CacheRead::Hit),
                    Ok(_) => Ok(CacheRead::Corrupt),
                    Err(_) => Ok(CacheRead::Unavailable),
                }
            }
        }
    }

    /// Best-effort publication from an already admitted local Store object.
    pub fn backfill_store(
        &self,
        source: &Store,
        kind: CacheKind,
        want: &ObjectRef,
    ) -> Result<CacheWrite> {
        let destination = self.path(kind, &want.sha256)?;
        if destination.exists() {
            return Ok(match verify_path(&destination, kind, want) {
                Ok(()) => CacheWrite::Present,
                Err(_) => CacheWrite::Unavailable,
            });
        }
        let result = match kind {
            CacheKind::Blob => match source.open_verified(&want.sha256) {
                Ok(mut file) if file.len() == want.length => {
                    publish_stream(&mut file, &destination, want)
                }
                Ok(_) | Err(_) => Err(SoftFailure::Unavailable),
            },
            CacheKind::Manifest => match source.read_manifest(want) {
                Ok(manifest) => publish_bytes(&manifest.canonical_bytes(), &destination),
                Err(_) => Err(SoftFailure::Unavailable),
            },
        };
        Ok(match result {
            Ok(true) => CacheWrite::Stored,
            Ok(false) => match verify_path(&destination, kind, want) {
                Ok(()) => CacheWrite::Present,
                Err(_) => CacheWrite::Unavailable,
            },
            Err(_) => CacheWrite::Unavailable,
        })
    }
}

fn read_status(error: SoftFailure) -> CacheRead {
    match error {
        SoftFailure::Missing => CacheRead::Missing,
        SoftFailure::Corrupt => CacheRead::Corrupt,
        SoftFailure::Unavailable => CacheRead::Unavailable,
    }
}

fn open_exact(path: &Path, length: u64) -> std::result::Result<File, SoftFailure> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(store::O_NOFOLLOW)
        .open(path)
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                SoftFailure::Missing
            } else {
                SoftFailure::Unavailable
            }
        })?;
    let metadata = file.metadata().map_err(|_| SoftFailure::Unavailable)?;
    if !metadata.is_file() {
        return Err(SoftFailure::Unavailable);
    }
    if metadata.len() != length {
        return Err(SoftFailure::Corrupt);
    }
    Ok(file)
}

fn read_manifest(path: &Path, want: &ObjectRef) -> std::result::Result<Manifest, SoftFailure> {
    let mut file = open_exact(path, want.length)?;
    let max = Manifest::MAX_BYTES as u64;
    if want.length > max {
        return Err(SoftFailure::Corrupt);
    }
    let mut bytes = Vec::with_capacity(want.length as usize);
    file.read_to_end(&mut bytes)
        .map_err(|_| SoftFailure::Unavailable)?;
    if sha256::hex_digest(&bytes) != want.sha256 {
        return Err(SoftFailure::Corrupt);
    }
    Manifest::parse(&bytes).map_err(|_| SoftFailure::Corrupt)
}

fn verify_path(
    path: &Path,
    kind: CacheKind,
    want: &ObjectRef,
) -> std::result::Result<(), SoftFailure> {
    match kind {
        CacheKind::Blob => {
            let mut file = open_exact(path, want.length)?;
            let mut sink = std::io::sink();
            stream_exact(&mut file, &mut sink, want)
        }
        CacheKind::Manifest => read_manifest(path, want).map(|_| ()),
    }
}

fn publish_bytes(bytes: &[u8], destination: &Path) -> std::result::Result<bool, SoftFailure> {
    let mut source = bytes;
    let want = ObjectRef::of(bytes);
    publish_stream(&mut source, destination, &want)
}

fn publish_stream(
    source: &mut impl Read,
    destination: &Path,
    want: &ObjectRef,
) -> std::result::Result<bool, SoftFailure> {
    let Some(directory) = destination.parent() else {
        return Err(SoftFailure::Unavailable);
    };
    fs::create_dir_all(directory).map_err(|_| SoftFailure::Unavailable)?;
    let name = destination
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("object");
    let temporary = directory.join(format!(
        ".{name}.{}.{}.tmp",
        std::process::id(),
        store::now_nanos_unique()
    ));
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&temporary)
        .map_err(|_| SoftFailure::Unavailable)?;
    if let Err(error) = stream_exact(source, &mut output, want) {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    if output.sync_all().is_err()
        || fs::set_permissions(&temporary, fs::Permissions::from_mode(0o444)).is_err()
    {
        let _ = fs::remove_file(&temporary);
        return Err(SoftFailure::Unavailable);
    }
    let admitted = match fs::hard_link(&temporary, destination) {
        Ok(()) => true,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => false,
        Err(_) => {
            let _ = fs::remove_file(&temporary);
            return Err(SoftFailure::Unavailable);
        }
    };
    let _ = fs::remove_file(&temporary);
    if admitted && store::fsync_dir(directory).is_err() {
        return Err(SoftFailure::Unavailable);
    }
    Ok(admitted)
}

fn stream_exact(
    source: &mut impl Read,
    destination: &mut impl Write,
    want: &ObjectRef,
) -> std::result::Result<(), SoftFailure> {
    let mut digest = Sha256::new();
    let mut length = 0u64;
    let mut buffer = vec![0u8; store::BUF];
    loop {
        let read = source
            .read(&mut buffer)
            .map_err(|_| SoftFailure::Unavailable)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
        destination
            .write_all(&buffer[..read])
            .map_err(|_| SoftFailure::Unavailable)?;
        length = length
            .checked_add(read as u64)
            .ok_or(SoftFailure::Corrupt)?;
    }
    if length != want.length || sha256::hex(&digest.finish()) != want.sha256 {
        return Err(SoftFailure::Corrupt);
    }
    Ok(())
}

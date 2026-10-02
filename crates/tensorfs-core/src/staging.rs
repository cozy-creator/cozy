//! `<store>/staging/<sha256>` — a downloaded source carrier, which is NOT a store object.
//!
//! Owner, 2026-09-04: *"The downloaded files are not part of CAS however right? they
//! shouldn't be. They should just be something in tensorfs' /tmp/ folder or wherever it
//! temporarily stores safetensors while it's downloading and ingesting them"* … *"it
//! download into its /tmp/, convert, then discard / delete the file that was converted
//! after it's done."*
//!
//! Before this module a HuggingFace shard was admitted by `tfs fetch url` into `blobs/`
//! with a `tensorfs_verified_blobs` row, indistinguishable in kind from a converted repo
//! segment. A temporary INPUT and a durable OUTPUT shared one namespace and one lifetime,
//! and every symptom followed from that: `gc` is a whole-store sweep and could not tell a
//! spent carrier from one that had landed and not yet been read, so it could not run
//! mid-ingest; unlinking a blob by hand left its verification record behind; and an H3
//! ingest had to provision 596 GB so that 210.3 GB of source and the output built from it
//! could coexist. Staged-and-dropped takes that to 406 GB and raises the largest
//! ingestible source from 1.77 TB to 2.94 TB.
//!
//! **There is nothing to be atomic about.** A targeted CAS-retirement verb (`tfs forget`)
//! was specified for this and withdrawn: it is a sophisticated answer to a problem that
//! exists only because the carrier is in the CAS. Deleting a file that was never a store
//! object needs no hold check, no chain check, no catalog surgery and no crash-ordering
//! argument. [`remove`] is one `unlink` and it is idempotent.
//!
//! ## Why a sibling of `blobs/` rather than a subdirectory of `tmp/`
//!
//! `tmp/` is the store's own scratch — `put-`/`stage-` temps that [`crate::store::Store::reap`]
//! sweeps, `tmp/leases`, `tmp/writers`, `tmp/ingest/<session>`. A carrier is none of those:
//! it outlives the process that fetched it and it is read by a different one. The decisive
//! property of a top-level `staging/` is structural rather than conventional: `Census`
//! walks `repos/`, `manifests/` and `blobs/` only, so **a staged carrier is invisible to
//! every liveness question anyone can ask, by construction rather than by a rule someone
//! has to remember.**
//!
//! It is FLAT and content-addressed — `staging/<hex>`, the path th-158 hands the pod — so a
//! re-ask finds its own file with no side index, two byte-identical members stage once, and
//! there are no fan-out directories to create or to prune. That last one matters: `gc`
//! prunes emptied fan-out directories safely because it holds the exclusive recovery lock,
//! and staging runs beside live writers, where removing a directory a concurrent
//! `create_dir_all`/`hard_link` pair is between would fail that writer with `ENOENT`.
//! `staging/` itself is created once and never removed.
//!
//! ## One door, one hash
//!
//! [`stage`] streams through [`crate::store::Store::stream_verified`], the same core
//! `put_stream` uses: bytes are hashed as they are written and refused if they are not
//! exactly the `ObjectRef` the record owner named. A narrower path must not become a weaker
//! one; a second hashing loop would be a second opinion about identity, and the two would
//! disagree exactly once, silently, on the artifact that mattered.
//!
//! ## No catalog, no writer marker
//!
//! Staging never opens the catalog and never takes a `WriterGuard`. It touches nothing any
//! census reads, so excluding `gc` for the ten to fifteen minutes a 5 GiB shard takes would
//! buy nothing. The one interaction is `reap`, whose decision is the kernel lock on a temp
//! and not the store's exclusivity — already correct beside a live staging writer.
//!
//! ## A staged name is never a partial, and residency is answered by rehash
//!
//! The bytes are hashed into a locked temp and only then linked to `staging/<hex>`, exactly
//! as `put_stream` installs an object. That is what lets a restart trust a name it finds
//! there enough to spend a rehash on it. A record would be the wrong instrument: a record
//! is a durable claim about a file the store KEEPS, and a carrier is a file the store is
//! about to throw away. So [`staged`] rehashes — match, answer staged and move zero bytes;
//! mismatch, remove it and fetch. That preserves a behaviour `tfs fetch url` has today (a
//! re-ask moves nothing) and costs seconds of local read against tens of seconds of link
//! and real money.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::{hex64, ObjectRef};
use crate::store::{Fault, Store, STAGE_TEMP};

fn io(what: impl AsRef<str>, error: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("{}: {error}", what.as_ref()),
    }
}

/// The staging area of one Store. Not a namespace: nothing walks it for liveness.
pub fn area(store: &Store) -> PathBuf {
    store.root().join("staging")
}

/// Where one carrier lives, or a refusal if the id is not a lowercase sha256.
///
/// The validation is what keeps a caller's string out of the path: `..` and a leading `/`
/// cannot travel with an id that had to be 64 hex characters first.
pub fn path(store: &Store, sha256: &str) -> Result<PathBuf> {
    let hex = hex64("staged carrier", sha256)?;
    Ok(area(store).join(hex))
}

/// Create `staging/` if it is absent, at TensorFS's own mode.
///
/// Pre-creating at 0o755 is run-185's lesson: a directory born under whichever uid touched
/// the Store first is a directory the Runtime's unprivileged reader cannot enter. `init`
/// and `prepare_readers` do this too; this is the path for a Store that predates tfs-067.
pub fn ensure(store: &Store) -> Result<PathBuf> {
    let dir = area(store);
    match fs::create_dir(&dir) {
        Ok(()) => {
            fs::set_permissions(&dir, std::os::unix::fs::PermissionsExt::from_mode(0o755))
                .map_err(|error| io(format!("chmod {}", dir.display()), error))?;
            crate::store::fsync_dir(store.root())?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(io(format!("mkdir {}", dir.display()), error)),
    }
    if !real_dir(&dir)? {
        return refuse(
            Code::STORE_ERA,
            format!("{} is not a real staging directory", dir.display()),
        );
    }
    Ok(dir)
}

/// A carrier on disk, and the absolute path the caller is about to hand to a converter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Carrier {
    pub object: ObjectRef,
    pub path: PathBuf,
    /// Whether this call moved the bytes, rather than finding them already staged.
    pub fetched: bool,
}

/// Is this carrier already here, provably?
///
/// A file at the right length is REHASHED, because a carrier has no verification record and
/// a length is not an identity. A mismatch is not a refusal: the wrong bytes are removed and
/// the answer is `None`, so the caller fetches — the same repair `FetchPlan` performs for a
/// corrupt blob.
pub fn staged(store: &Store, want: &ObjectRef) -> Result<Option<PathBuf>> {
    let path = path(store, &want.sha256)?;
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(io(format!("stat {}", path.display()), error)),
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return refuse(
            Code::NOT_REGULAR_FILE,
            format!("{} is not one plain staged file", path.display()),
        );
    }
    if metadata.len() != want.length {
        // A length that disagrees cannot be the object, and no rehash is worth paying to
        // find that out.
        remove(store, &want.sha256)?;
        return Ok(None);
    }
    let mut file =
        fs::File::open(&path).map_err(|error| io(format!("open {}", path.display()), error))?;
    let mut hasher = crate::sha256::Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let n = file
            .read(&mut buffer)
            .map_err(|error| io(format!("read {}", path.display()), error))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
        store.moved(n as u64);
    }
    if crate::sha256::hex(&hasher.finish()) != want.sha256 {
        remove(store, &want.sha256)?;
        return Ok(None);
    }
    Ok(Some(path))
}

/// Stream one carrier in through the shared hashing core and link it to `staging/<hex>`.
///
/// The binding is the object id itself and `stream_verified` enforces it by digest, so a
/// URL that serves the wrong bytes cannot poison this area — it can only fail, leaving
/// nothing staged and no temp behind.
pub fn stage<R: Read>(store: &Store, r: &mut R, want: &ObjectRef) -> Result<Carrier> {
    ensure(store)?;
    let verified = store.stream_verified(r, Some(want), &Fault::default(), STAGE_TEMP)?;
    adopt(store, verified)
}

/// Link a temp file already hashed to its id to `staging/<hex>`, and drop the temp.
pub(crate) fn adopt(store: &Store, verified: crate::store::Verified) -> Result<Carrier> {
    let dir = ensure(store)?;
    let destination = dir.join(&verified.sha256);
    // 0444 for the same reason an object is: the converter reads it and nothing writes it
    // again. A carrier is removed by unlink, which a read-only mode does not prevent.
    if let Err(error) = fs::set_permissions(
        &verified.tmp,
        std::os::unix::fs::PermissionsExt::from_mode(0o444),
    ) {
        let _ = fs::remove_file(&verified.tmp);
        return Err(io("chmod 0444", error));
    }
    // No-clobber, exactly as admission is: two concurrent fetches of one id both succeed
    // and one file exists. There is no record to write, so there is nothing to race on
    // afterwards either.
    match fs::hard_link(&verified.tmp, &destination) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => {
            let _ = fs::remove_file(&verified.tmp);
            return Err(io(format!("link {}", destination.display()), error));
        }
    }
    if store.flushes() {
        crate::store::fsync_dir(&dir)?;
    }
    let _ = fs::remove_file(&verified.tmp);
    drop(verified.file);
    Ok(Carrier {
        object: ObjectRef {
            sha256: verified.sha256,
            length: verified.length,
        },
        path: destination,
        fetched: true,
    })
}

/// Retire one spent carrier. IDEMPOTENT: an absent id is success, because the caller's
/// intent — "these bytes are not here any more" — is already true.
///
/// Returns whether this call removed the file. `staging/` is never removed.
pub fn remove(store: &Store, sha256: &str) -> Result<bool> {
    let path = path(store, sha256)?;
    match fs::remove_file(&path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(io(format!("remove {}", path.display()), error)),
    }
}

/// What a sweep removed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Cleared {
    pub carriers: u64,
    pub bytes: u64,
    /// Orphan `stage-` temps: a fetch that died mid-stream, whose lock the kernel released
    /// with it.
    pub temps: u64,
}

/// Sweep the whole area, plus the orphan temps of fetches that died before linking.
///
/// This is the end-of-operation call, and it is what keeps a crash from leaking 210 GB into
/// the next run on the same pod: [`remove`] retires a carrier the converter says is spent,
/// and this retires everything the converter never got to say anything about.
///
/// A live fetch's temp is KEPT, on the same lock test `reap` uses — the kernel holds it for
/// a live writer whatever its mtime says.
pub fn clear(store: &Store) -> Result<Cleared> {
    let mut cleared = Cleared::default();
    let dir = area(store);
    if real_dir(&dir)? {
        for entry in
            fs::read_dir(&dir).map_err(|error| io(format!("read {}", dir.display()), error))?
        {
            let entry = entry.map_err(|error| io(format!("read {}", dir.display()), error))?;
            let path = entry.path();
            let metadata = match fs::symlink_metadata(&path) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return Err(io(format!("stat {}", path.display()), error)),
            };
            if !metadata.is_file() || metadata.file_type().is_symlink() {
                continue;
            }
            match fs::remove_file(&path) {
                Ok(()) => {
                    cleared.carriers += 1;
                    cleared.bytes = cleared.bytes.saturating_add(metadata.len());
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(io(format!("remove {}", path.display()), error)),
            }
        }
    }
    cleared.temps = reap_stage_temps(store)?;
    Ok(cleared)
}

/// Every carrier on disk, sorted by id. Whatever is not a plain file named by a valid
/// digest is not a carrier and is not reported.
pub fn list(store: &Store) -> Result<Vec<ObjectRef>> {
    let mut carriers = Vec::new();
    let dir = area(store);
    if !real_dir(&dir)? {
        return Ok(carriers);
    }
    for entry in fs::read_dir(&dir).map_err(|error| io(format!("read {}", dir.display()), error))? {
        let entry = entry.map_err(|error| io(format!("read {}", dir.display()), error))?;
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if hex64("staged carrier", &name).is_err() {
            continue;
        }
        let metadata = match fs::symlink_metadata(entry.path()) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(io("stat staged carrier", error)),
        };
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            continue;
        }
        carriers.push(ObjectRef {
            sha256: name,
            length: metadata.len(),
        });
    }
    carriers.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    Ok(carriers)
}

/// What the staging area is holding, for [`crate::store::Store::occupancy`].
pub fn bytes(store: &Store) -> Result<u64> {
    let mut total = 0u64;
    for carrier in list(store)? {
        total = total.checked_add(carrier.length).ok_or_else(|| Refusal {
            code: Code::ARITH_OVERFLOW,
            detail: "staging occupancy overflow".into(),
        })?;
    }
    Ok(total)
}

/// One line per carrier: the id and the absolute path a converter opens.
pub fn line(store: &Store, carrier: &ObjectRef) -> Result<String> {
    Ok(format!(
        "{} {} {}",
        carrier.id(),
        carrier.length,
        path(store, &carrier.sha256)?.display()
    ))
}

fn reap_stage_temps(store: &Store) -> Result<u64> {
    use fs2::FileExt;
    let mut gone = 0;
    let tmp = store.root().join("tmp");
    if !real_dir(&tmp)? {
        return Ok(0);
    }
    for entry in fs::read_dir(&tmp).map_err(|error| io("read tmp", error))? {
        let entry = entry.map_err(|error| io("read tmp", error))?;
        let name = entry.file_name();
        if !name.to_str().is_some_and(|n| n.starts_with(STAGE_TEMP)) {
            continue;
        }
        let path = entry.path();
        if !fs::symlink_metadata(&path).is_ok_and(|m| m.is_file()) {
            continue;
        }
        let file = match fs::File::options().read(true).open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(io(format!("open temp {}", path.display()), error)),
        };
        // The lock, never the clock: a temp the kernel still holds has a live fetch behind
        // it, and a 5 GiB shard's writer is older than any age a sweep could pick.
        if file.try_lock_exclusive().is_err() {
            continue;
        }
        match fs::remove_file(&path) {
            Ok(()) => gone += 1,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(io(format!("remove temp {}", path.display()), error)),
        }
    }
    Ok(gone)
}

fn real_dir(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(metadata.is_dir() && !metadata.file_type().is_symlink()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(io(format!("stat {}", path.display()), error)),
    }
}

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
use crate::limits;
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

    /// Offer optional replication after local acquisition. This reports no durability
    /// and never waits for filesystem I/O or queue space. Ingest's explicit Mirror
    /// retains its separate completion/watermark contract.
    pub(crate) fn offer_backfill(
        &self,
        source: &Store,
        kind: CacheKind,
        object: &ObjectRef,
    ) -> bool {
        static BACKFILLS: std::sync::OnceLock<Backfills> = std::sync::OnceLock::new();
        let pool = BACKFILLS.get_or_init(Backfills::start);
        pool.sender
            .try_send(Backfill {
                cache: self.clone(),
                source: source.root().to_path_buf(),
                kind,
                object: object.clone(),
            })
            .is_ok()
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

    /// Read one cached blob's exact bytes, verified against the digest and length that
    /// named it. Bounded by the caller's own declared length, which is why this is only for
    /// the small canonical documents the cache carries beside tensor bytes — a journal link
    /// — and never for a tensor object, which streams through `admit`.
    pub fn read_blob(&self, want: &ObjectRef) -> Result<Vec<u8>> {
        let path = self.path(CacheKind::Blob, &want.sha256)?;
        if want.length > limits::DOC_MAX_BYTES as u64 {
            return crate::err::refuse(
                Code::SIZE_CAP,
                format!("{}: {} B is past the document cap", want.id(), want.length),
            );
        }
        let mut file = open_exact(&path, want.length).map_err(|failure| crate::err::Refusal {
            code: match failure {
                SoftFailure::Missing => Code::OBJECT_ABSENT,
                SoftFailure::Corrupt => Code::OBJECT_CORRUPT,
                SoftFailure::Unavailable => Code::IO_FAILED,
            },
            detail: format!("{}: the cache cannot answer for it", want.id()),
        })?;
        let mut bytes = Vec::with_capacity(want.length as usize);
        file.read_to_end(&mut bytes)
            .map_err(|error| crate::err::Refusal {
                code: Code::IO_FAILED,
                detail: format!("read cached {}: {error}", want.id()),
            })?;
        if sha256::hex_digest(&bytes) != want.sha256 {
            return crate::err::refuse(
                Code::OBJECT_CORRUPT,
                format!("{}: the cached bytes are not the object named", want.id()),
            );
        }
        Ok(bytes)
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

// Reference-only memory policy for optional acquisition backfill, not a protocol bound.
// Typical jobs are a few hundred bytes; even two 4 KiB paths per job stay near 1 MiB.
const BACKFILL_QUEUE: usize = 128;

struct Backfill {
    cache: RepoObjectCache,
    source: PathBuf,
    kind: CacheKind,
    object: ObjectRef,
}

struct Backfills {
    sender: std::sync::mpsc::SyncSender<Backfill>,
}

impl Backfills {
    fn start() -> Self {
        let (sender, receiver) = std::sync::mpsc::sync_channel::<Backfill>(BACKFILL_QUEUE);
        let receiver = std::sync::Arc::new(std::sync::Mutex::new(receiver));
        // One pool for the process, never a new blocked thread set for every pull.
        // Workers are detached: process exit may leave cache temps, never canonical
        // partial objects. Dropping a pull must not join a stalled FUSE syscall.
        for _ in 0..STREAMS {
            let receiver = receiver.clone();
            let _ = std::thread::Builder::new()
                .name("tfs-cache-write".into())
                .spawn(move || loop {
                    let work = {
                        let queue = receiver.lock().unwrap_or_else(|error| error.into_inner());
                        match queue.recv() {
                            Ok(work) => work,
                            Err(_) => return,
                        }
                    };
                    // Reopen without a caller's progress observer. Queued work owns no GC
                    // hold: a source reclaimed before it opens is simply a missed cache fill.
                    // Once opened, the verified descriptor pins that inode through the copy.
                    if let Ok(store) = Store::open(&work.source) {
                        let _ = work.cache.backfill_store(&store, work.kind, &work.object);
                    }
                });
        }
        Self { sender }
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

// ---------------------------------------------------------------- write-through

/// How many objects may be waiting for the cache at once.
///
/// It is `MAX_TOTAL_REFS` on purpose and not a tuned number. A legal checkpoint may not
/// reference more objects than that — the header refuses above it — so one conversion
/// cannot offer more than that, so an offer can never find this queue full and can never
/// block the producer. The bound is the DOCUMENT'S OWN bound, which makes it a derived
/// fact rather than a guess, and 65,536 `ObjectRef`s is a few megabytes against artifacts
/// measured in hundreds of gigabytes.
const QUEUE: usize = limits::MAX_TOTAL_REFS;

/// How many objects the mirror publishes at once. A mounted network filesystem answers a
/// single writer well below its own ceiling; more than one write in flight is what makes
/// the publication overlap the work that produced the bytes instead of trailing it.
const STREAMS: usize = 4;

/// How many times ONE object's publication is retried before the mount is called
/// unavailable. It is a COUNT OF ANSWERS and never a clock — nothing in this file asks how
/// long anything took: each attempt is a complete publish that either produced the final
/// path or did not, and a mount that refuses the same object this many times running is not
/// slow, it is not working.
const ATTEMPTS: usize = 3;

/// What the write-through published, counted per object. `stored` is what this run put on
/// the cache; `present` was already there — the ordinary answer when two pods run the same
/// deterministic plan — and `unavailable` is what the mount would not take.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MirrorReport {
    pub offered: u64,
    pub stored: u64,
    pub present: u64,
    pub unavailable: u64,
    pub bytes_offered: u64,
    pub bytes_stored: u64,
    pub bytes_present: u64,
    pub durable: Durable,
}

/// The WATERMARK, and the reason it is a prefix rather than a total.
///
/// `objects` counts offers, IN OFFER ORDER, that are all now on the cache. A later object
/// landing first does not advance it, because a journal naming an object whose predecessor
/// never made it would claim a durability the cache cannot honour. It is a count and a byte
/// total; it makes no statement about time, and every decision built on it — checkpoint
/// here, this run is making progress, that one is not — is a decision about observed work.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Durable {
    pub objects: u64,
    pub bytes: u64,
    /// Set once a publication has exhausted [`ATTEMPTS`], or once an offer could not be
    /// queued at all. The prefix stops here for the rest of the run and no later checkpoint
    /// happens. The conversion is untouched: it simply stops being resumable past this
    /// point, which is cost and latency, never authority or availability.
    pub stalled: bool,
}

struct Work {
    seq: u64,
    kind: CacheKind,
    object: ObjectRef,
}

#[derive(Default)]
struct Prefix {
    /// The next sequence that must land for the durable prefix to advance.
    next: u64,
    /// Sequences that landed AHEAD of `next`, and what each was worth. With several writers
    /// in flight completion order is not offer order; this is the reorder buffer that turns
    /// one into the other. It holds at most what is genuinely out of order.
    ahead: std::collections::BTreeMap<u64, u64>,
    bytes: u64,
    stalled: bool,
}

/// What the producer has said about its own progress, and the door a journal waits at.
/// There is no timeout here and no sampling cadence: the producer says "another unit of
/// work is done", or it says "I am finished", and nothing else wakes a waiter.
#[derive(Default)]
struct Pulse {
    units: u64,
    closed: bool,
}

#[derive(Default)]
struct MirrorState {
    offered: std::sync::atomic::AtomicU64,
    bytes_offered: std::sync::atomic::AtomicU64,
    stored: std::sync::atomic::AtomicU64,
    present: std::sync::atomic::AtomicU64,
    unavailable: std::sync::atomic::AtomicU64,
    bytes_stored: std::sync::atomic::AtomicU64,
    bytes_present: std::sync::atomic::AtomicU64,
    prefix: std::sync::Mutex<Prefix>,
    /// Every offer in offer order. A journal link names objects by their place in this
    /// list, so the list is the mirror's and not a second copy kept by the producer. Bounded
    /// by [`QUEUE`] for the same reason the queue is.
    offers: std::sync::Mutex<Vec<(CacheKind, ObjectRef)>>,
    pulse: std::sync::Mutex<Pulse>,
    woken: std::sync::Condvar,
    /// The newest snapshot of the producer's own progress document, and the offer sequence
    /// it was handed over at. A link may name it only once that sequence is inside the
    /// durable prefix — the document is subject to exactly the rule its objects are.
    progress: std::sync::Mutex<Option<(u64, ObjectRef)>>,
    /// Durable bytes when the last progress snapshot was offered.
    progress_at: std::sync::atomic::AtomicU64,
    /// Durable bytes between snapshots — the same interval the journal links at, held here
    /// so the producer asks one question (`progress_due`) instead of carrying a policy.
    interval: u64,
}

impl MirrorState {
    fn prefix(&self) -> std::sync::MutexGuard<'_, Prefix> {
        self.prefix.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn stall(&self) {
        self.prefix().stalled = true;
    }

    fn record(&self, seq: u64, length: u64, outcome: CacheWrite) {
        use std::sync::atomic::Ordering::Relaxed;
        match outcome {
            CacheWrite::Stored => {
                self.stored.fetch_add(1, Relaxed);
                self.bytes_stored.fetch_add(length, Relaxed);
            }
            CacheWrite::Present => {
                self.present.fetch_add(1, Relaxed);
                self.bytes_present.fetch_add(length, Relaxed);
            }
            CacheWrite::Unavailable => {
                self.unavailable.fetch_add(1, Relaxed);
            }
        }
        let mut prefix = self.prefix();
        if outcome == CacheWrite::Unavailable {
            prefix.stalled = true;
            return;
        }
        if prefix.stalled {
            return;
        }
        prefix.ahead.insert(seq, length);
        loop {
            let next = prefix.next;
            let Some(bytes) = prefix.ahead.remove(&next) else {
                break;
            };
            prefix.bytes += bytes;
            prefix.next = next + 1;
        }
    }

    fn durable(&self) -> Durable {
        let prefix = self.prefix();
        Durable {
            objects: prefix.next,
            bytes: prefix.bytes,
            stalled: prefix.stalled,
        }
    }
}

/// Write-through publication of already-admitted local objects, on threads of its own.
///
/// **The producer never waits for the cache, and that is the whole design.** Cache I/O
/// blocking the worker is a named falsifier of the minimal-local-state decision, and a
/// mounted network filesystem is precisely the thing that stops answering without saying
/// so. An offer is therefore a push onto a queue that provably cannot fill (see [`QUEUE`]),
/// and every byte the cache costs is paid on these threads.
///
/// **What the mirror publishes is already proved.** It goes through
/// [`RepoObjectCache::backfill_store`], which reads the local Store's VERIFIED descriptor
/// and re-hashes on the way to the final path, so the cache can only receive bytes this pod
/// already admitted at the digest that names them. There is no second admission path and no
/// way for an unproved byte to reach the mount.
pub struct Mirror {
    sender: std::sync::Mutex<Option<std::sync::mpsc::SyncSender<Work>>>,
    threads: std::sync::Mutex<Vec<std::thread::JoinHandle<()>>>,
    state: std::sync::Arc<MirrorState>,
}

impl Mirror {
    /// Start publishing onto `cache` from the Store at `store_root`.
    ///
    /// The mirror opens its OWN Store handle rather than sharing the caller's, for one
    /// reason: the caller's handle usually carries a progress observer, and bytes this
    /// publication reads are not bytes the conversion moved. A root that will not open is
    /// weather like any other — every offer is counted, none is durable, and the producer
    /// is unaffected.
    pub fn start(cache: RepoObjectCache, store_root: &Path, interval: u64) -> Mirror {
        let state = std::sync::Arc::new(MirrorState {
            interval,
            ..MirrorState::default()
        });
        let (sender, receiver) = std::sync::mpsc::sync_channel::<Work>(QUEUE);
        let Ok(store) = Store::open(store_root) else {
            state.stall();
            return Mirror {
                sender: std::sync::Mutex::new(None),
                threads: std::sync::Mutex::new(Vec::new()),
                state,
            };
        };
        let receiver = std::sync::Arc::new(std::sync::Mutex::new(receiver));
        let mut threads = Vec::with_capacity(STREAMS);
        for _ in 0..STREAMS {
            let receiver = std::sync::Arc::clone(&receiver);
            let state = std::sync::Arc::clone(&state);
            let cache = cache.clone();
            let store = store.clone();
            threads.push(std::thread::spawn(move || loop {
                let work = {
                    let queue = receiver.lock().unwrap_or_else(|e| e.into_inner());
                    match queue.recv() {
                        Ok(work) => work,
                        Err(_) => return,
                    }
                };
                let mut outcome = CacheWrite::Unavailable;
                for _ in 0..ATTEMPTS {
                    outcome = cache
                        .backfill_store(&store, work.kind, &work.object)
                        .unwrap_or(CacheWrite::Unavailable);
                    if outcome != CacheWrite::Unavailable {
                        break;
                    }
                }
                state.record(work.seq, work.object.length, outcome);
                state.wake();
            }));
        }
        Mirror {
            sender: std::sync::Mutex::new(Some(sender)),
            threads: std::sync::Mutex::new(threads),
            state,
        }
    }

    /// Hand one already-admitted object to the cache. Returns its sequence, which is the
    /// number a watermark is read against. Never blocks, never fails, never refuses.
    pub fn offer(&self, kind: CacheKind, object: &ObjectRef) -> u64 {
        use std::sync::atomic::Ordering::Relaxed;
        // The sequence is the offer list's INDEX, taken under the same lock that appends to
        // it. Numbering outside the lock is the version of this that reads correctly and is
        // wrong: two producers could take 5 and 6 and then append in the other order, and a
        // journal link would name the objects of a different slice than the one the durable
        // prefix vouched for.
        let seq = {
            let mut offers = self.state.offers.lock().unwrap_or_else(|e| e.into_inner());
            offers.push((kind, object.clone()));
            (offers.len() - 1) as u64
        };
        self.state.offered.fetch_add(1, Relaxed);
        self.state.bytes_offered.fetch_add(object.length, Relaxed);
        let work = Work {
            seq,
            kind,
            object: object.clone(),
        };
        let sender = self.sender.lock().unwrap_or_else(|e| e.into_inner());
        match sender.as_ref() {
            // The only ways here are a queue past the document's own reference cap and a
            // writer set that is gone. The object stays local and proved either way; the
            // durable prefix simply stops at this sequence.
            Some(sender) if sender.try_send(work).is_err() => self.state.stall(),
            Some(_) => {}
            None => self.state.stall(),
        }
        seq
    }

    /// Offer every segment of one part. An inline body lives in the header and has no
    /// object of its own, so a part with no segments offers nothing.
    pub fn offer_part(&self, part: &crate::header::Part) {
        for segment in part.segments() {
            self.offer(CacheKind::Blob, segment);
        }
    }

    /// Whether enough has become durable for the producer to snapshot its progress document
    /// again. Reading it costs one lock and no I/O, so a converter may ask it every op.
    pub fn progress_due(&self) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let durable = self.state.durable();
        durable
            .bytes
            .saturating_sub(self.state.progress_at.load(Relaxed))
            >= self.state.interval
    }

    /// Hand over a snapshot of the producer's own progress document. It is an ordinary blob
    /// and travels the ordinary way; what makes it special is only that a journal link may
    /// point at it, and only once it is inside the durable prefix like everything else.
    pub fn offer_progress(&self, object: &ObjectRef) {
        use std::sync::atomic::Ordering::Relaxed;
        let seq = self.offer(CacheKind::Blob, object);
        self.state
            .progress_at
            .store(self.state.durable().bytes, Relaxed);
        *self
            .state
            .progress
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some((seq, object.clone()));
    }

    /// The newest progress snapshot that is inside a durable prefix of `objects` offers, or
    /// `None`. A link that named a snapshot the cache had not taken would send a resuming pod
    /// to an object that is not there.
    pub fn progress_within(&self, objects: u64) -> Option<ObjectRef> {
        let progress = self
            .state
            .progress
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        progress
            .as_ref()
            .filter(|(seq, _)| *seq < objects)
            .map(|(_, object)| object.clone())
    }

    /// The producer says one unit of its own work is finished. This is the ONLY thing that
    /// wakes a journal, and it is an observation about work rather than about time. It
    /// takes a lock and a notify and does no I/O, so calling it in a conversion's inner
    /// loop costs the conversion nothing.
    pub fn at_unit(&self) {
        self.state.wake();
    }

    /// The durable prefix as it stands right now. One lock, no I/O.
    pub fn durable(&self) -> Durable {
        self.state.durable()
    }

    /// What has been handed over, durable or not — the other half of the watermark, and the
    /// number that says whether the mirror is keeping up with the producer.
    pub fn offered(&self) -> (u64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        (
            self.state.offered.load(Relaxed),
            self.state.bytes_offered.load(Relaxed),
        )
    }

    /// The objects at offer positions `range`, in offer order. A journal link names exactly
    /// a slice of this list, so the producer never keeps a second copy that could disagree.
    pub fn offers(&self, from: u64, to: u64) -> Vec<(CacheKind, ObjectRef)> {
        let offers = self.state.offers.lock().unwrap_or_else(|e| e.into_inner());
        let from = (from as usize).min(offers.len());
        let to = (to as usize).min(offers.len());
        offers[from..to].to_vec()
    }

    /// Block until the producer reports more work than `seen`, or until it is finished.
    /// Returns the new unit count, or `None` once the producer has closed. There is no
    /// timeout: the only two things that end this wait are progress and completion.
    pub fn awaited(&self, seen: u64) -> Option<u64> {
        let mut pulse = self.state.pulse.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if pulse.units != seen {
                return Some(pulse.units);
            }
            if pulse.closed {
                return None;
            }
            pulse = self
                .state
                .woken
                .wait(pulse)
                .unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Stop taking offers and wait for every one already taken to have its answer. After
    /// this the durable prefix is final. Idempotent.
    pub fn drain(&self) {
        {
            let mut sender = self.sender.lock().unwrap_or_else(|e| e.into_inner());
            *sender = None;
        }
        let threads = std::mem::take(&mut *self.threads.lock().unwrap_or_else(|e| e.into_inner()));
        for thread in threads {
            let _ = thread.join();
        }
        self.state.wake();
    }

    /// Tell every waiter the producer is finished. A journal wakes one last time, takes its
    /// final link over the drained prefix, and returns.
    pub fn close(&self) {
        let mut pulse = self.state.pulse.lock().unwrap_or_else(|e| e.into_inner());
        pulse.closed = true;
        drop(pulse);
        self.state.woken.notify_all();
    }

    pub fn report(&self) -> MirrorReport {
        use std::sync::atomic::Ordering::Relaxed;
        MirrorReport {
            offered: self.state.offered.load(Relaxed),
            stored: self.state.stored.load(Relaxed),
            present: self.state.present.load(Relaxed),
            unavailable: self.state.unavailable.load(Relaxed),
            bytes_offered: self.state.bytes_offered.load(Relaxed),
            bytes_stored: self.state.bytes_stored.load(Relaxed),
            bytes_present: self.state.bytes_present.load(Relaxed),
            durable: self.state.durable(),
        }
    }
}

impl MirrorState {
    fn wake(&self) {
        let mut pulse = self.pulse.lock().unwrap_or_else(|e| e.into_inner());
        pulse.units += 1;
        drop(pulse);
        self.woken.notify_all();
    }
}

impl Drop for Mirror {
    fn drop(&mut self) {
        self.close();
        self.drain();
    }
}

#[cfg(test)]
mod backfill_tests {
    use super::*;

    fn temporary(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tensorfs-backfill-{name}-{}-{}",
            std::process::id(),
            store::now_nanos_unique()
        ))
    }

    #[test]
    fn active_cache_copy_allows_gc_and_keeps_its_open_verified_bytes() {
        let root = temporary("gc");
        let source = Store::init(&root.join("source")).unwrap();
        let bytes = vec![37; store::BUF * 2 + 17];
        let object = source
            .put_stream(&mut bytes.as_slice(), None, &Fault::default())
            .unwrap()
            .obj;
        let mut file = source.open_verified(&object.sha256).unwrap();
        let cache = RepoObjectCache::new(root.join("cache"));
        let destination = cache.path(CacheKind::Blob, &object.sha256).unwrap();
        struct Reclaim<F> {
            file: F,
            root: PathBuf,
            first: bool,
            length: u64,
        }
        impl<F: Read> Read for Reclaim<F> {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                let read = self.file.read(output)?;
                if self.first {
                    self.first = false;
                    // The actual copy has already opened its temporary and read its
                    // first buffer. No global hold may defer GC until cache I/O ends.
                    let report = crate::gc::collect(&self.root, false).unwrap();
                    assert_eq!(report.reclaimed_blobs, 1);
                    assert_eq!(report.reclaimed_bytes, self.length);
                }
                Ok(read)
            }
        }
        let mut input = Reclaim {
            file: &mut file,
            root: source.root().to_path_buf(),
            first: true,
            length: object.length,
        };
        assert_eq!(publish_stream(&mut input, &destination, &object), Ok(true));
        assert!(!source.contains(&object.sha256));
        assert_eq!(fs::read(&destination).unwrap(), bytes);
        assert_eq!(verify_path(&destination, CacheKind::Blob, &object), Ok(()));
        assert_eq!(
            fs::read_dir(destination.parent().unwrap()).unwrap().count(),
            1
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn failed_cache_copy_removes_only_its_temporary_and_never_publishes() {
        let root = temporary("failure");
        let cache = RepoObjectCache::new(&root);
        let object = ObjectRef::of(&[19; 1024]);
        let destination = cache.path(CacheKind::Blob, &object.sha256).unwrap();
        fs::create_dir_all(destination.parent().unwrap()).unwrap();
        let other = destination.parent().unwrap().join(".other-process.tmp");
        fs::write(&other, b"owned elsewhere").unwrap();
        struct Fails(bool);
        impl Read for Fails {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                if self.0 {
                    return Err(std::io::Error::other("injected source read failure"));
                }
                self.0 = true;
                output[..17].fill(19);
                Ok(17)
            }
        }
        assert_eq!(
            publish_stream(&mut Fails(false), &destination, &object),
            Err(SoftFailure::Unavailable)
        );
        assert!(!destination.exists());
        assert_eq!(fs::read(&other).unwrap(), b"owned elsewhere");
        assert_eq!(
            fs::read_dir(destination.parent().unwrap()).unwrap().count(),
            1
        );
        fs::remove_dir_all(root).unwrap();
    }
}

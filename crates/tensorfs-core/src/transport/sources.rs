//! Selected-source downloads (HuggingFace, Civitai) on the pull's downloader, resumable
//! across processes. Called only while the selected-source writer holds its owner fence.
//!
//! Each wanted object downloads into `roots/source-downloads/<owner>/<object>/bytes.part`
//! as ranged chunks. A chunk that is home is fsynced, hashed and appended to `chunks`, the
//! object's record: its identity on the first line, then `<first> <end> <sha256>` per
//! chunk. A process that resumes re-hashes every recorded chunk, forgets any that no longer
//! match, and asks only for what is not covered. The finished file is hashed whole and
//! committed to the Store from where it lies; bytes that hash wrong start the object over.
use super::download::{Download, PART_BYTES};
use super::{CredentialProvider, Deadline, Ledger, PullCancellation, SourcePolicy, STREAMS};
use crate::err::{refuse, Code, Refusal, Result};
use crate::fetch::{DeliveryGrant, FetchPlan};
use crate::ids::{prefixed, ObjectRef};
use crate::sha256::{hex, Sha256};
use crate::store::{Fault, Store};
use fs2::FileExt;
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{FileExt as _, MetadataExt};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Clone, Debug)]
pub struct SourceDownload {
    /// One request's bytes, and the unit a resumed download keeps.
    pub checkpoint_bytes: u64,
    /// Requests in flight across the whole call.
    pub streams: usize,
    pub fault: Fault,
    pub cancellation: Option<PullCancellation>,
}

impl Default for SourceDownload {
    fn default() -> Self {
        Self {
            checkpoint_bytes: PART_BYTES,
            streams: STREAMS,
            fault: Fault::default(),
            cancellation: None,
        }
    }
}

/// Bytes of the call's distinct objects present locally (held before it, or recorded by
/// it) and their total. Present drops only when retained bytes are discarded.
pub type SourceProgress<'a> = &'a (dyn Fn(u64, u64) + Sync);

/// One wanted object: its grant and the source URL it is fetched from.
pub(crate) struct SourceJob {
    pub grant: DeliveryGrant,
    pub url: String,
}

fn io(error: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("source download: {error}"),
    }
}

fn area(store: &Store) -> PathBuf {
    store.root().join("roots/source-downloads")
}

fn owner_dir(store: &Store, owner: &str) -> Result<PathBuf> {
    prefixed("source download owner", owner)?;
    Ok(area(store).join(&owner[7..]))
}

/// Chunks home, by first byte: their end and the sha256 of their bytes.
type Chunks = BTreeMap<u64, (u64, String)>;

/// Recorded ranges, merged into the bytes they cover.
fn covered(done: &BTreeMap<u64, u64>) -> u64 {
    let (mut total, mut reached) = (0, 0);
    for (first, end) in done {
        let first = (*first).max(reached);
        if *end > first {
            total += end - first;
            reached = *end;
        }
    }
    total
}

/// The ranges `done` does not cover, in order.
pub(super) fn missing(done: &BTreeMap<u64, u64>, length: u64) -> Vec<(u64, u64)> {
    let (mut out, mut reached) = (Vec::new(), 0);
    for (first, end) in done {
        if *first > reached {
            out.push((reached, *first));
        }
        reached = reached.max(*end);
    }
    if reached < length {
        out.push((reached, length));
    }
    out
}

fn digest(file: &File, first: u64, end: u64) -> std::io::Result<Option<String>> {
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; (end - first).min(1 << 20) as usize];
    let mut at = first;
    while at < end {
        let want = buffer.len().min((end - at) as usize);
        match file.read_at(&mut buffer[..want], at)? {
            0 => return Ok(None),
            read => {
                hash.update(&buffer[..read]);
                at += read as u64;
            }
        }
    }
    Ok(Some(hex(&hash.finish())))
}

/// One object's download area: the bytes, the record of chunks home, and the lock that
/// fences every other writer of it.
pub(super) struct Part {
    dir: PathBuf,
    identity: String,
    pub file: File,
    record: Mutex<(File, Chunks)>,
    _lock: File,
}

impl Part {
    pub fn bytes(&self) -> PathBuf {
        self.dir.join("bytes.part")
    }

    /// Lock it, read its record and re-hash every chunk it names. A chunk that no longer
    /// matches is forgotten and fetched again; a read-only open (a donor) refuses on it.
    pub fn open(store: &Store, owner: &str, object: &ObjectRef, writable: bool) -> Result<Self> {
        crate::ids::hex64("source download object", &object.sha256)?;
        let dir = owner_dir(store, owner)?.join(&object.sha256);
        let created = !dir.is_dir();
        fs::create_dir_all(&dir).map_err(io)?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(dir.join("lock"))
            .map_err(io)?;
        lock.try_lock_exclusive().map_err(|_| Refusal {
            code: Code::STORE_BUSY,
            detail: "this source object has a live writer".into(),
        })?;
        let open = |name: &str, options: &mut OpenOptions| {
            options
                .read(true)
                .create(writable)
                .open(dir.join(name))
                .map_err(io)
        };
        let file = open("bytes.part", OpenOptions::new().write(writable))?;
        let mut record = open("chunks", OpenOptions::new().append(writable))?;
        let mut text = String::new();
        record.read_to_string(&mut text).map_err(io)?;
        let identity = format!("{} {}", object.sha256, object.length);
        let mut lines = text.split_inclusive('\n');
        match lines.next() {
            Some(line) if line.trim_end() == identity => {}
            None if writable => record
                .write_all(format!("{identity}\n").as_bytes())
                .and_then(|()| record.sync_data())
                .map_err(io)?,
            _ => {
                return refuse(
                    Code::TRANSACTION_CONFLICT,
                    "source download record names another object",
                )
            }
        }
        // A line cut short by a crash is not a claim.
        let (mut done, mut forgot) = (Chunks::new(), false);
        for line in lines.filter(|line| line.ends_with('\n')) {
            let fields: Vec<&str> = line.split_whitespace().collect();
            let [first, end, sha256] = fields[..] else {
                continue;
            };
            let (Ok(first), Ok(end)) = (first.parse::<u64>(), end.parse::<u64>()) else {
                continue;
            };
            if first >= end || end > object.length {
                continue;
            }
            match digest(&file, first, end).map_err(io)? {
                Some(sha) if sha == sha256 => {
                    done.insert(first, (end, sha));
                }
                _ => forgot = true,
            }
        }
        if forgot && !writable {
            return refuse(
                Code::OBJECT_CORRUPT,
                "retained source bytes no longer hash to their record",
            );
        }
        if created && writable {
            // Newly created ancestors are durable before any progress is acknowledged.
            for dir in dir.ancestors().take(4) {
                File::open(dir).and_then(|f| f.sync_all()).map_err(io)?;
            }
        }
        let part = Part {
            dir,
            identity,
            file,
            record: Mutex::new((record, done)),
            _lock: lock,
        };
        if forgot {
            part.rewrite()?;
        }
        Ok(part)
    }

    /// The ranges recorded home.
    pub fn done(&self) -> BTreeMap<u64, u64> {
        let record = self.record.lock().unwrap();
        record
            .1
            .iter()
            .map(|(first, (end, _))| (*first, *end))
            .collect()
    }

    pub fn covered(&self) -> u64 {
        covered(&self.done())
    }

    /// Bytes `first..end` are home: data first, then the record line. A crash leaves the
    /// record without it, and the range is asked for again.
    pub fn record(&self, first: u64, end: u64, fault: &Fault) -> Result<()> {
        fault.hit(if first == 0 {
            "chunk-written"
        } else {
            "later-chunk-written"
        });
        self.file.sync_data().map_err(io)?;
        fault.hit("chunk-fsync");
        let sha256 = digest(&self.file, first, end)
            .map_err(io)?
            .ok_or_else(|| Refusal {
                code: Code::LENGTH_MISMATCH,
                detail: "a source chunk is past the end of its file".into(),
            })?;
        let mut record = self.record.lock().unwrap();
        record
            .0
            .write_all(format!("{first} {end} {sha256}\n").as_bytes())
            .and_then(|()| record.0.sync_data())
            .map_err(io)?;
        record.1.insert(first, (end, sha256));
        drop(record);
        fault.hit("chunk-recorded");
        Ok(())
    }

    /// Forget every range: the bytes hashed wrong as a whole.
    pub fn reset(&self) -> Result<()> {
        let mut record = self.record.lock().unwrap();
        record.1.clear();
        record
            .0
            .set_len(self.identity.len() as u64 + 1)
            .and_then(|()| record.0.sync_data())
            .and_then(|()| self.file.set_len(0))
            .map_err(io)
    }

    /// The record as it stands, replaced whole: a crash leaves the old one, whose
    /// forgotten chunks are forgotten again.
    fn rewrite(&self) -> Result<()> {
        let mut record = self.record.lock().unwrap();
        let mut text = format!("{}\n", self.identity);
        for (first, (end, sha256)) in &record.1 {
            text.push_str(&format!("{first} {end} {sha256}\n"));
        }
        let temp = self.dir.join("chunks.tmp");
        fs::write(&temp, text)
            .and_then(|()| File::open(&temp)?.sync_all())
            .and_then(|()| fs::rename(&temp, self.dir.join("chunks")))
            .map_err(io)?;
        record.0 = OpenOptions::new()
            .append(true)
            .open(self.dir.join("chunks"))
            .map_err(io)?;
        Ok(())
    }

    /// The object is in the Store. Its lock stays, so an old process never locks an
    /// unlinked predecessor.
    pub fn finished(&self) {
        let _ = fs::remove_file(self.dir.join("chunks"));
    }
}

/// What a source download reports and keeps, shared by its objects.
pub(super) struct Keep<'a> {
    pub owner: &'a str,
    pub fault: &'a Fault,
    progress: SourceProgress<'a>,
    total: u64,
    present: Mutex<u64>,
}

impl Keep<'_> {
    /// Present bytes grew or shrank by `delta`; the report leaves under the same lock.
    pub fn landed(&self, delta: i64) {
        let mut present = self.present.lock().unwrap();
        *present = present.saturating_add_signed(delta);
        (self.progress)(*present, self.total);
    }
}

/// Fetch every job into the store, returning the bytes this call moved per job.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fetch_sources(
    store: &Store,
    owner: &str,
    jobs: &[SourceJob],
    policy: &SourcePolicy,
    credentials: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &Ledger,
    options: &SourceDownload,
    held: u64,
    progress: SourceProgress<'_>,
) -> Result<Vec<u64>> {
    // The whole remaining download against the disk budget, before any byte moves: bytes
    // already retained under this owner are part of the store's occupancy.
    let mut adding = 0u64;
    for job in jobs {
        let part = owner_dir(store, owner)?
            .join(&job.grant.object.sha256)
            .join("bytes.part");
        let retained = fs::metadata(part).map_or(0, |md| allocated(&md));
        adding = adding.saturating_add(job.grant.object.length.saturating_sub(retained));
    }
    store.admits(adding)?;
    let keep = Keep {
        owner,
        fault: &options.fault,
        progress,
        total: held + jobs.iter().map(|job| job.grant.object.length).sum::<u64>(),
        present: Mutex::new(held),
    };
    keep.landed(0);
    let wanted: Vec<ObjectRef> = jobs.iter().map(|job| job.grant.object.clone()).collect();
    let urls: BTreeMap<String, String> = jobs
        .iter()
        .map(|job| (job.grant.object.sha256.clone(), job.url.clone()))
        .collect();
    let moved = Mutex::new(BTreeMap::<String, u64>::new());
    let on_object = |object: &ObjectRef, bytes: u64, _| {
        moved.lock().unwrap().insert(object.sha256.clone(), bytes);
    };
    Download {
        store,
        wanted: &wanted,
        grant: &|object| {
            let at = jobs.iter().position(|job| job.grant.object == *object);
            Ok(jobs[at.expect("a wanted object has a job")].grant.clone())
        },
        urls: &urls,
        chunk: options.checkpoint_bytes.clamp(1 << 20, 1 << 30),
        ranged: false,
        policy,
        credential: credentials,
        deadline,
        ledger,
        on_object: Some(&on_object),
        backfill: None,
        keep: Some(&keep),
    }
    .run(super::transfer_streams(options.streams.clamp(1, 128)), None)?;
    let moved = moved.into_inner().unwrap();
    Ok(jobs
        .iter()
        .map(|job| moved.get(&job.grant.object.sha256).copied().unwrap_or(0))
        .collect())
}

/// Allocated bytes: a sparse part holds only what its chunks wrote.
fn allocated(md: &fs::Metadata) -> u64 {
    md.len().min(md.blocks().saturating_mul(512))
}

pub(crate) fn discard_owner(store: &Store, owner: &str) -> Result<()> {
    match fs::remove_dir_all(owner_dir(store, owner)?) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(io(e)),
        _ => Ok(()),
    }
}

pub(crate) fn bytes(store: &Store) -> Result<u64> {
    let owners = match fs::read_dir(area(store)) {
        Ok(rows) => rows,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(io(e)),
    };
    let mut total = 0u64;
    for owner in owners {
        for object in fs::read_dir(owner.map_err(io)?.path()).map_err(io)? {
            match fs::symlink_metadata(object.map_err(io)?.path().join("bytes.part")) {
                Ok(md) if md.is_file() => {
                    total = total.checked_add(allocated(&md)).ok_or_else(|| Refusal {
                        code: Code::ARITH_OVERFLOW,
                        detail: "source download bytes overflow".into(),
                    })?
                }
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(io(e)),
            }
        }
    }
    Ok(total)
}

/// The identity a record's first line names.
fn recorded_object(dir: &Path) -> Result<Option<ObjectRef>> {
    let mut text = String::new();
    match File::open(dir.join("chunks")) {
        Ok(file) => file.take(256).read_to_string(&mut text).map_err(io)?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(io(e)),
    };
    let line = text.lines().next().unwrap_or_default();
    let mut fields = line.split_whitespace();
    match (fields.next(), fields.next().and_then(|n| n.parse().ok())) {
        (Some(sha256), Some(length))
            if dir.file_name().and_then(|n| n.to_str()) == Some(sha256) =>
        {
            Ok(Some(ObjectRef {
                sha256: sha256.into(),
                length,
            }))
        }
        _ => refuse(
            Code::TRANSACTION_CONFLICT,
            "source download record names another object",
        ),
    }
}

/// Copy an independently owned download's recorded chunks while both owners are fenced.
/// A donor object that is whole is admitted through the digest-checked door instead.
pub(crate) fn adopt_owner(
    store: &Store,
    source: &str,
    owner: &str,
    mut landed: impl FnMut(&ObjectRef) -> Result<()>,
) -> Result<()> {
    let entries = match fs::read_dir(owner_dir(store, source)?) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(io(e)),
    };
    for entry in entries {
        let dir = entry.map_err(io)?.path();
        let Some(object) = recorded_object(&dir)? else {
            continue;
        };
        // Admission may have linked the bytes before the donor recorded this member: the
        // Store's own verified object is reused, and the donor's files are left untouched.
        let (plan, _) = FetchPlan::of_objects(
            store,
            "source-prefix-adoption",
            std::slice::from_ref(&object),
        )?;
        if !plan.held.is_empty() {
            landed(&object)?;
            continue;
        }
        let from = Part::open(store, source, &object, false)?;
        let done = from.done();
        if missing(&done, object.length).is_empty() {
            let mut reader = Read::take(&from.file, object.length);
            store.put_stream(&mut reader, Some(&object), &Fault::default())?;
            landed(&object)?;
            continue;
        }
        let to = Part::open(store, owner, &object, true)?;
        let have = to.done();
        let wanted: Vec<(u64, u64)> = done
            .into_iter()
            .filter(|(first, end)| have.get(first) != Some(end))
            .collect();
        store.admits(wanted.iter().map(|(first, end)| end - first).sum())?;
        let mut buffer = vec![0u8; 1 << 20];
        for (first, end) in wanted {
            let mut at = first;
            while at < end {
                let want = buffer.len().min((end - at) as usize);
                let read = from.file.read_at(&mut buffer[..want], at).map_err(io)?;
                if read == 0 {
                    return refuse(
                        Code::LENGTH_MISMATCH,
                        "an adopted source range was truncated",
                    );
                }
                to.file.write_all_at(&buffer[..read], at).map_err(io)?;
                at += read as u64;
            }
            to.record(first, end, &Fault::default())?;
        }
    }
    Ok(())
}

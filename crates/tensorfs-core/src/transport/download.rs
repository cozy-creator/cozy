//! The pull's downloader, the way an S3 client does it. The wanted objects are cut into
//! fixed ranged chunks on one queue, `streams` workers each GET one chunk into its
//! object's temp file, and a finished file is hashed and committed to the Store.
//!
//! One rule per request: one that fails, or waits or moves far slower than the requests
//! running beside it over the same time, is dropped, and the rest of its chunk is asked
//! again on fresh connections, in 1 MiB pieces. A pod whose every request is slow restarts
//! none (vast 5090s, 2026-10-10: 885 restarts in 42 s against a median of the few early
//! asks that came home). The pull ends on an answer asking again cannot change, or when
//! nothing new has landed for twice its measured patience.
//!
//! Why chunks: a cold R2 object answers each MiB after its own delay, usually under a
//! second and sometimes a minute. One GET for the object pays every delay in a row (runs
//! 2322, sasori, dietfried, chieri: a pull's last objects at 30-180 KB/s for minutes).
//! Separate requests pay them side by side, and a late one is simply asked again.
//!
//! `fetch_ranged` is the same downloader for one object from a URL the caller holds, and
//! `sources` for a selected source's members, kept on disk across processes. Such an
//! origin may not answer ranges, so an object's first chunk goes alone: a 206 queues the
//! rest, a 200 is the whole object. Where it redirects to is asked for the rest directly.

use super::decisions;
use super::ledger::{Attempt, Deadline, Ledger, STALL_FACTOR};
use super::pull::{
    open_object, shared_client, FetchObserver, Fetched, ObjectObserver, ObjectSource, ObjectUrls,
    Opened, WalkOutcome,
};
use super::sources::{missing, Keep, Part};
use super::STREAMS;
use super::{CredentialProvider, SourcePolicy};
use crate::err::{refuse, Code, Refusal, Result};
use crate::fetch::{DeliveryGrant, Destination};
use crate::ids::ObjectRef;
use crate::repo_cache::{BackfillBatch, CacheKind, CacheRead, CacheTrust};
use crate::store::{Fault, Store, Verified, PUT_TEMP};
use fs2::FileExt;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::os::unix::fs::{FileExt as _, OpenOptionsExt};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// One request's bytes. Measured on a pod at 128 workers: 4, 8 and 16 MiB chunks all move
/// what whole 64 MiB GETs move (1.1-1.3 GB/s); 1 MiB chunks move half.
pub const CHUNK_BYTES: u64 = 8 << 20;

/// The most writes into temp files at once, whatever the requests in flight. A buffered
/// write is a copy into the page cache: the kernel's writeback writes the disk, and one
/// file's writes take turns on its inode lock, so more writers add no bytes, only threads
/// stalled in the kernel. With one writer per request (512) a 6.5 GiB pull on a laptop left
/// 180 threads in uninterruptible sleep, load at 396 and the disk's IO pressure full for ten
/// minutes. Requests beyond this wait their turn holding their read buffer, so in-flight
/// unwritten bytes stay at most one buffer per request.
pub const WRITERS: usize = 8;

/// Turns at writing, `WRITERS` at once; `peak` is the most that ever wrote together.
struct Writers {
    free: Mutex<usize>,
    freed: Condvar,
    peak: AtomicUsize,
}

impl Writers {
    fn new() -> Self {
        Writers {
            free: Mutex::new(WRITERS),
            freed: Condvar::new(),
            peak: AtomicUsize::new(0),
        }
    }

    fn write(&self, file: &File, data: &[u8], at: u64) -> std::io::Result<()> {
        let mut free = self.free.lock().unwrap();
        while *free == 0 {
            free = self.freed.wait(free).unwrap();
        }
        *free -= 1;
        self.peak.fetch_max(WRITERS - *free, Ordering::Relaxed);
        drop(free);
        let written = file.write_all_at(data, at);
        *self.free.lock().unwrap() += 1;
        self.freed.notify_one();
        written
    }
}

/// One request's bytes from a source origin. Larger, because every ask walks the origin's
/// redirect again (0.41 s against HuggingFace, and its resolve endpoint is rate-limited).
pub const PART_BYTES: u64 = 64 << 20;

/// A dropped request's unfinished range is cut into pieces, an eighth of a chunk each and
/// never under this: a slow object answers small asks side by side, and a chunk asked again
/// whole would pay its slow MiBs in a row. Also what a body is given time to move, at its
/// peers' pace, before it is judged.
const PIECE_BYTES: u64 = 1 << 20;

/// An object being downloaded: its file, and what is still out.
struct Object {
    /// Its place in `wanted`, which is how the URL supply names it.
    at: usize,
    grant: DeliveryGrant,
    cache: Option<(CacheKind, CacheTrust)>,
    path: PathBuf,
    file: File,
    /// A source download's kept file and record, in place of a temp file.
    part: Option<Part>,
    /// Pieces not yet home.
    left: AtomicUsize,
    /// Chunks not yet asked for: an origin that may not range has not yet answered one.
    rest: Mutex<Vec<(u64, u64)>>,
    /// Each chunk not yet home, by its first byte: its end and the pieces still out.
    units: Mutex<HashMap<u64, (u64, usize)>>,
    /// Where the origin last redirected this object's URL, asked directly until refused.
    delivery: Mutex<Option<String>>,
    /// Bytes requests wrote for it, bytes asked twice included.
    moved: AtomicU64,
}

impl Drop for Object {
    fn drop(&mut self) {
        if self.part.is_none() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Bytes `first..end` of one object. An empty range is the object's commit.
struct Chunk {
    object: Arc<Object>,
    first: u64,
    end: u64,
    /// Not asked before this: an origin's stated wait, or one sample after an ask that
    /// moved nothing. `stated` says the wait is the origin's own.
    due: Instant,
    stated: bool,
    /// The origin refused the URL the last ask held, with this status.
    stale: Option<u16>,
    /// Times this range was dropped. Each doubles what the next ask is given before it is
    /// judged, so an origin that is honestly slow still answers.
    drops: u32,
    /// The object's only chunk so far: its origin has not yet shown that it answers ranges.
    lead: bool,
    /// The chunk this is (a piece of): the range a source download records when it is home.
    unit: (u64, u64),
}

impl Chunk {
    /// This chunk's fields for a piece cut from its front.
    fn take(&self) -> Chunk {
        Chunk {
            object: Arc::clone(&self.object),
            ..*self
        }
    }
}

/// A request on the wire, as the watch sees it.
struct Flight {
    attempt: Arc<Attempt>,
    subject: String,
    range: (u64, u64),
    began: Instant,
    /// How many fair windows this request gets before it is judged.
    patience: u32,
    /// When its body was last judged, and with how many bytes.
    judged: Option<Instant>,
    seen: u64,
}

/// A request as the others see it: ended, or still running (`open`) as of the watch's tick.
struct Peer {
    began: Instant,
    first: Option<Instant>,
    ended: Instant,
    open: bool,
    bytes: u64,
    /// It asked for less than a piece: judged by its time in all, not by a rate.
    small: bool,
}

enum Outcome {
    Home,
    Again(Vec<Chunk>),
    Fatal(Refusal),
}

/// How a request ended short of its chunk.
enum Ended {
    /// Ask again, after the wait the origin stated if it stated one.
    Retry(Refusal, Option<Duration>),
    /// An answer asking again cannot change.
    Fatal(Refusal),
}

impl From<Refusal> for Ended {
    fn from(refusal: Refusal) -> Ended {
        match refusal.code {
            // A connection that failed, stalled or was dropped, a name the resolver did not
            // answer, and a full descriptor table.
            Code::TRANSFER_FAILED | Code::HUB_UNREACHABLE | Code::FD_HEADROOM => {
                Ended::Retry(refusal, None)
            }
            _ => Ended::Fatal(refusal),
        }
    }
}

#[derive(Default)]
struct State {
    queue: VecDeque<Chunk>,
    /// The next index of `plan.wanted` nobody has opened, and ones a full descriptor table
    /// sent back.
    next: usize,
    again: VecDeque<usize>,
    /// Workers holding a chunk or opening an object.
    busy: usize,
    /// The last requests that ended, as many as there are streams, oldest first.
    shown: VecDeque<Peer>,
    /// The last refusal a retry answered, quoted if the pull stops moving.
    last: Option<Refusal>,
    failure: Option<Refusal>,
    done: bool,
}

pub(super) struct Download<'a> {
    pub store: &'a Store,
    pub wanted: &'a [ObjectRef],
    /// The door's grant for one wanted object.
    pub grant: &'a (dyn Fn(&ObjectRef) -> Result<DeliveryGrant> + Sync),
    pub urls: &'a dyn ObjectUrls,
    /// One request's bytes, and whether the origin is known to answer ranges.
    pub chunk: u64,
    pub ranged: bool,
    pub policy: &'a SourcePolicy,
    pub credential: &'a dyn CredentialProvider,
    pub deadline: Deadline,
    pub ledger: &'a Ledger,
    pub on_object: Option<ObjectObserver<'a>>,
    pub backfill: Option<&'a BackfillBatch<'a>>,
    /// A source download: objects kept on disk under their owner, resumable.
    pub keep: Option<&'a Keep<'a>>,
}

struct Run<'a> {
    job: &'a Download<'a>,
    state: Mutex<State>,
    /// Workers wait here for a chunk; the watch waits on `over` for the end.
    wake: Condvar,
    over: Condvar,
    flights: Vec<Mutex<Option<Flight>>>,
    fetched: AtomicU64,
    bytes_moved: AtomicU64,
    cached: AtomicU64,
    bytes_cached: AtomicU64,
    requests: AtomicU64,
    restarts: AtomicU64,
    began: Instant,
    /// When each worker began waiting on the hub for a URL, in ms since `began` plus one.
    /// A wait of a sample or more is the hub's to judge, and is not the pull standing still.
    minting: Vec<AtomicU64>,
    /// The latest moment such a wait ended.
    minted: AtomicU64,
    /// Bytes home in temp files and from the cache. A whole answer to a resumed range takes
    /// its object's bytes back out, so asking the same bytes forever is not progress.
    landed: AtomicU64,
    writers: Writers,
}

impl Download<'_> {
    /// Download `wanted` with `streams` requests in flight. `observer` hears the bytes home
    /// so far, on this thread.
    pub fn run(
        &self,
        streams: usize,
        mut observer: Option<&mut dyn FetchObserver>,
    ) -> Result<WalkOutcome> {
        let began = Instant::now();
        let chunks: usize = self
            .wanted
            .iter()
            .map(|object| object.length.div_ceil(self.chunk).max(1) as usize)
            .sum();
        let streams = streams.min(chunks).max(1);
        let run = Run {
            job: self,
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
            over: Condvar::new(),
            flights: (0..streams).map(|_| Mutex::new(None)).collect(),
            fetched: AtomicU64::new(0),
            bytes_moved: AtomicU64::new(0),
            cached: AtomicU64::new(0),
            bytes_cached: AtomicU64::new(0),
            requests: AtomicU64::new(0),
            restarts: AtomicU64::new(0),
            began,
            minting: (0..streams).map(|_| AtomicU64::new(0)).collect(),
            minted: AtomicU64::new(0),
            landed: AtomicU64::new(0),
            writers: Writers::new(),
        };
        std::thread::scope(|scope| {
            for slot in 0..streams {
                let run = &run;
                scope.spawn(move || run.work(slot));
            }
            run.watch(&mut observer);
        });
        if let Some(observer) = observer.as_mut() {
            observer.moved(run.landed.load(Ordering::Relaxed));
        }
        let failure = run.state.lock().unwrap().failure.take();
        let outcome = WalkOutcome {
            fetched: run.fetched.load(Ordering::Relaxed),
            bytes_moved: run.bytes_moved.load(Ordering::Relaxed),
            cached: run.cached.load(Ordering::Relaxed),
            bytes_cached: run.bytes_cached.load(Ordering::Relaxed),
            streams,
            writers: run.writers.peak.load(Ordering::Relaxed),
            requests: run.requests.load(Ordering::Relaxed),
            restarts: run.restarts.load(Ordering::Relaxed),
        };
        decisions::record(
            self.store,
            "walk",
            &format!(
                "{} {}",
                self.wanted.len(),
                self.wanted.iter().map(|object| object.length).sum::<u64>()
            ),
            &format!(
                "fetched={} cached={} moved={} seconds={:.1} streams={streams} writers={} \
                 requests={} restarts={} outcome={}",
                outcome.fetched,
                outcome.cached,
                outcome.bytes_moved,
                began.elapsed().as_secs_f64(),
                outcome.writers,
                outcome.requests,
                outcome.restarts,
                failure.as_ref().map_or("ok", |r| r.code.as_str()),
            ),
        );
        match failure {
            Some(refusal) => Err(refusal),
            None => Ok(outcome),
        }
    }
}

impl Run<'_> {
    fn work(&self, slot: usize) {
        let mut buffer = vec![0u8; 256 << 10];
        while let Some(chunk) = self.take() {
            let outcome = match chunk.first == chunk.end {
                true => self.commit(&chunk.object),
                false => self.request(slot, chunk, &mut buffer),
            };
            let mut state = self.state.lock().unwrap();
            state.busy -= 1;
            match outcome {
                Outcome::Home => {}
                Outcome::Again(chunks) => {
                    for chunk in chunks.into_iter().rev() {
                        state.queue.push_front(chunk);
                    }
                }
                Outcome::Fatal(refusal) => self.fail(&mut state, refusal),
            }
            self.wake.notify_all();
        }
    }

    /// The next chunk that is due. With none, the next object is opened and its chunks
    /// queued; with nothing left to open, the worker waits for one to come due or back.
    fn take(&self) -> Option<Chunk> {
        let wanted = self.job.wanted;
        let mut state = self.state.lock().unwrap();
        loop {
            if state.failure.is_some() {
                return None;
            }
            let now = Instant::now();
            if let Some(index) = state.queue.iter().position(|chunk| chunk.due <= now) {
                state.busy += 1;
                return state.queue.remove(index);
            }
            let open = state.again.pop_front().or_else(|| {
                (state.next < wanted.len()).then(|| {
                    state.next += 1;
                    state.next - 1
                })
            });
            if let Some(at) = open {
                state.busy += 1;
                drop(state);
                let opened = self.open(at);
                state = self.state.lock().unwrap();
                state.busy -= 1;
                match opened {
                    Ok(chunks) => state.queue.extend(chunks),
                    // A full descriptor table is backpressure: open it again a sample on.
                    Err(refusal) if refusal.code == Code::FD_HEADROOM => {
                        state.again.push_front(at);
                        state.last = Some(refusal);
                        let sample = self.job.ledger.sample();
                        state = self.wake.wait_timeout(state, sample).unwrap().0;
                    }
                    Err(refusal) => self.fail(&mut state, refusal),
                }
                self.wake.notify_all();
                continue;
            }
            if state.queue.is_empty() && state.busy == 0 {
                state.done = true;
                self.wake.notify_all();
                self.over.notify_all();
                return None;
            }
            let wait = state
                .queue
                .iter()
                .map(|chunk| chunk.due.saturating_duration_since(now))
                .min()
                .map_or(self.job.ledger.sample(), |due| {
                    due.min(self.job.ledger.sample())
                })
                .max(Duration::from_millis(1));
            state = self.wake.wait_timeout(state, wait).unwrap().0;
        }
    }

    fn fail(&self, state: &mut State, refusal: Refusal) {
        if state.failure.is_none() {
            // Keep the first failure before waking peers, so their cancellation errors
            // cannot replace its diagnosis.
            state.failure = Some(refusal);
            self.job.ledger.cancel();
            self.over.notify_all();
        }
    }

    /// One object off the wanted list: answered by the mounted cache, or given a temp file
    /// and cut into chunks.
    fn open(&self, at: usize) -> Result<Vec<Chunk>> {
        let job = self.job;
        let wanted = &job.wanted[at];
        let grant = (job.grant)(wanted)?;
        let kind = Some(match grant.destination {
            Destination::Manifest => CacheKind::Manifest,
            Destination::Blob => CacheKind::Blob,
        });
        let mut cache = None;
        let mounted = job.store.repo_cache().filter(|_| job.keep.is_none());
        if let (Some(kind), Some(mounted)) = (kind, mounted) {
            let mut trust = CacheTrust::Name;
            match mounted.admit(job.store, kind, wanted)? {
                CacheRead::Hit => {
                    self.cached.fetch_add(1, Ordering::Relaxed);
                    self.bytes_cached
                        .fetch_add(wanted.length, Ordering::Relaxed);
                    // Cached bytes never reach a socket, and they are progress too.
                    job.ledger.moved(wanted.length);
                    self.landed.fetch_add(wanted.length, Ordering::Relaxed);
                    if let Some(observer) = job.on_object {
                        observer(wanted, wanted.length, ObjectSource::Cache);
                    }
                    return Ok(Vec::new());
                }
                // This read proved the cached bytes wrong: the backfill replaces them.
                CacheRead::Corrupt => trust = CacheTrust::Rehash,
                CacheRead::Missing | CacheRead::Unavailable => {}
            }
            cache = Some((kind, trust));
        }
        let length = wanted.length;
        let (path, file, part, chunks) = match job.keep {
            Some(keep) => {
                let part = Part::open(job.store, keep.owner, wanted, true)?;
                let done = part.done();
                keep.landed(part.covered() as i64);
                let file = part.file.try_clone().map_err(|error| crate::store::classify_io("source download", error))?;
                (part.bytes(), file, Some(part), missing(&done, length))
            }
            None => {
                let (path, file) = self.temp(length)?;
                (path, file, None, vec![(0, length)])
            }
        };
        let chunks: Vec<(u64, u64)> = chunks
            .into_iter()
            .flat_map(|(first, end)| {
                (first..end)
                    .step_by(job.chunk as usize)
                    .map(move |at| (at, (at + job.chunk).min(end)))
            })
            .collect();
        // An origin that may not range is asked for the first chunk alone.
        let asked = if job.ranged { chunks.len() } else { 1 };
        let object = Arc::new(Object {
            at,
            grant,
            cache,
            path,
            file,
            part,
            left: AtomicUsize::new(chunks.len()),
            rest: Mutex::new(chunks.iter().skip(asked).rev().copied().collect()),
            units: Mutex::new(
                chunks
                    .iter()
                    .map(|(first, end)| (*first, (*end, 1)))
                    .collect(),
            ),
            delivery: Mutex::new(None),
            moved: AtomicU64::new(0),
        });
        let chunk = |(first, end)| Chunk {
            object: Arc::clone(&object),
            first,
            end,
            due: Instant::now(),
            stated: false,
            stale: None,
            drops: 0,
            lead: !job.ranged,
            unit: (first, end),
        };
        if chunks.is_empty() {
            return Ok(vec![chunk((length, length))]);
        }
        Ok(chunks.into_iter().take(asked).map(chunk).collect())
    }

    /// A locked temp file of `length` bytes, under the Store's own temp name so its reaper
    /// knows a live writer from an orphan.
    fn temp(&self, length: u64) -> Result<(PathBuf, File)> {
        let path = self.job.store.root().join("tmp").join(format!(
            "{PUT_TEMP}chunks-{}-{}",
            std::process::id(),
            crate::meta::now_nanos_unique()
        ));
        let io = |what: &str, error: std::io::Error| Refusal {
            code: match crate::descriptors::exhausted(&error) {
                true => Code::FD_HEADROOM,
                false => Code::IO_FAILED,
            },
            detail: format!("{what} {}: {error}", path.display()),
        };
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)
            .map_err(|error| io("create", error))?;
        let sized = file
            .try_lock_exclusive()
            .and_then(|()| file.set_len(length));
        if let Err(error) = sized {
            let _ = std::fs::remove_file(&path);
            return Err(io("size", error));
        }
        Ok((path, file))
    }

    /// The origin answered a range: the object's other chunks join the queue.
    fn rest(&self, lead: &Chunk) {
        let rest = std::mem::take(&mut *lead.object.rest.lock().unwrap());
        let mut state = self.state.lock().unwrap();
        for (first, end) in rest.into_iter().rev() {
            state.queue.push_back(Chunk {
                object: Arc::clone(&lead.object),
                first,
                end,
                due: Instant::now(),
                stated: false,
                stale: None,
                drops: 0,
                lead: false,
                unit: (first, end),
            });
        }
        self.wake.notify_all();
    }

    /// A piece is home. When it was its chunk's last, a source download records the chunk.
    fn unit_home(&self, chunk: &Chunk) -> Result<()> {
        let object = &chunk.object;
        let mut units = object.units.lock().unwrap();
        let Some((end, out)) = units.get_mut(&chunk.unit.0) else {
            return Ok(());
        };
        *out -= 1;
        if *out > 0 {
            return Ok(());
        }
        let end = *end;
        units.remove(&chunk.unit.0);
        drop(units);
        let (Some(part), Some(keep)) = (&object.part, self.job.keep) else {
            return Ok(());
        };
        keep.landed(part.record(chunk.unit.0, end, keep.fault)? as i64);
        Ok(())
    }

    /// One GET for a chunk, written at its offsets. Whatever ends it early, the bytes that
    /// landed stay and the rest goes back on the queue.
    fn request(&self, slot: usize, mut chunk: Chunk, buffer: &mut [u8]) -> Outcome {
        let job = self.job;
        let object = Arc::clone(&chunk.object);
        let wanted = &object.grant.object;
        // The URL first: a wait on the hub is the hub's to judge, not this request's.
        let stale = chunk.stale.take();
        let asked = self.began.elapsed();
        self.minting[slot].store(asked.as_millis() as u64 + 1, Ordering::Release);
        let delivery = object.delivery.lock().unwrap().clone();
        let url = match delivery.clone() {
            Some(delivery) => Ok(delivery),
            None => job
                .urls
                .url_for_while(object.at, wanted, stale.is_some(), job.ledger),
        };
        self.minting[slot].store(0, Ordering::Release);
        if self.began.elapsed() - asked >= job.ledger.sample() {
            let now = self.began.elapsed().as_millis() as u64;
            self.minted.fetch_max(now, Ordering::AcqRel);
        }
        let url = match (url, stale) {
            (Ok(url), _) => url,
            // A refused re-authorization still names the origin's answer.
            (Err(refusal), Some(status)) => {
                return Outcome::Fatal(Refusal {
                    code: Code::TRANSFER_FAILED,
                    detail: format!(
                        "{}: the object store answered {status} and re-authorizing was \
                         refused — {}",
                        wanted.id(),
                        refusal.detail
                    ),
                })
            }
            (Err(refusal), None) => return Outcome::Fatal(refusal),
        };
        let attempt = Arc::new(Attempt::default());
        let ledger = job.ledger.for_request(Arc::clone(&attempt), self.tick());
        let began = Instant::now();
        let asked = (chunk.first, chunk.end);
        *self.flights[slot].lock().unwrap() = Some(Flight {
            attempt: Arc::clone(&attempt),
            subject: decisions::object(wanted),
            range: asked,
            began,
            patience: 1 << chunk.drops.min(6),
            judged: None,
            seen: 0,
        });
        self.requests.fetch_add(1, Ordering::Relaxed);
        let ended = self.stream(&mut chunk, &url, delivery.is_some(), &ledger, buffer);
        *self.flights[slot].lock().unwrap() = None;
        self.show(&attempt, began, asked);
        // What this ask added: a whole answer to a resumed range starts the object over.
        let moved = chunk.first.saturating_sub(asked.0);
        let (refusal, stated) = match ended {
            Ok(()) => {
                if let Err(refusal) = self.unit_home(&chunk) {
                    return Outcome::Fatal(named(refusal, &wanted.id()));
                }
                if object.left.fetch_sub(1, Ordering::AcqRel) > 1 {
                    return Outcome::Home;
                }
                chunk.first = wanted.length;
                chunk.end = wanted.length;
                chunk.due = Instant::now();
                return Outcome::Again(vec![chunk]);
            }
            Err(Ended::Retry(refusal, stated)) => (refusal, stated),
            Err(Ended::Fatal(refusal)) => {
                // The first failure is the diagnosis; a stopped pull's own refusal is not.
                let stopped = job.ledger.check_running().err();
                return Outcome::Fatal(stopped.unwrap_or_else(|| named(refusal, &wanted.id())));
            }
        };
        // The pull itself was stopped, or its deadline passed: nothing is asked again.
        if let Err(stopped) = job
            .ledger
            .check_running()
            .and_then(|()| job.deadline.remaining().map(|_| ()))
        {
            return Outcome::Fatal(stopped);
        }
        let dropped = attempt.stopped.load(Ordering::Acquire);
        chunk.drops += dropped as u32;
        // An ask that moved nothing waits one sample, or as long as the origin stated.
        let wait = match (stated, moved) {
            (Some(stated), _) => stated,
            (None, 0) if !dropped => job.ledger.sample(),
            _ => Duration::ZERO,
        };
        chunk.due = Instant::now() + wait;
        chunk.stated = stated.is_some();
        self.state.lock().unwrap().last = Some(named(refusal, &wanted.id()));
        // A dropped request's unfinished range goes back as pieces, asked side by side.
        let mut again = Vec::new();
        let piece = (job.chunk / 8).max(PIECE_BYTES);
        while dropped && !chunk.lead && chunk.end - chunk.first > piece {
            let end = chunk.first + piece;
            object.left.fetch_add(1, Ordering::AcqRel);
            if let Some((_, out)) = object.units.lock().unwrap().get_mut(&chunk.unit.0) {
                *out += 1;
            }
            again.push(Chunk {
                object: Arc::clone(&object),
                end,
                ..chunk.take()
            });
            chunk.first = end;
        }
        again.push(chunk);
        Outcome::Again(again)
    }

    /// An ended request that had a first byte, for the ones still running to be judged by.
    fn show(&self, attempt: &Attempt, began: Instant, asked: (u64, u64)) {
        let Some(first) = attempt.first.get().copied() else {
            return;
        };
        let mut state = self.state.lock().unwrap();
        if state.shown.len() >= self.flights.len() {
            state.shown.pop_front();
        }
        state.shown.push_back(Peer {
            began,
            first: Some(first),
            ended: Instant::now(),
            open: false,
            bytes: attempt.bytes.load(Ordering::Relaxed),
            small: asked.1 - asked.0 < PIECE_BYTES,
        });
    }

    /// The watch's cadence, and a request's own: a fifth of the pull's sample.
    fn tick(&self) -> Duration {
        (self.job.ledger.sample() / 5).max(Duration::from_millis(1))
    }

    /// Ask for `chunk` and write what arrives, advancing `chunk.first` as bytes land.
    fn stream(
        &self,
        chunk: &mut Chunk,
        url: &str,
        delivered: bool,
        ledger: &Ledger,
        buffer: &mut [u8],
    ) -> std::result::Result<(), Ended> {
        let job = self.job;
        let object = &chunk.object;
        let wanted = &object.grant.object;
        // A Hub object that is one chunk is one plain GET. A source request always names its
        // range: a 206 is how the origin says it ranges, and its stand-ins answer only ranges.
        let whole = chunk.first == 0 && chunk.end == wanted.length;
        let range = [(
            "range".to_string(),
            format!("bytes={}-{}", chunk.first, chunk.end - 1),
        )];
        let mut hops = Vec::new();
        let opened = open_object(
            shared_client(),
            url,
            job.policy,
            job.credential,
            job.deadline,
            ledger,
            if whole && job.ranged { &[] } else { &range },
            &mut |checked, _| hops.push(checked.url()),
        )?;
        // A redirect target that refuses is forgotten, and the object's URL asked again.
        if delivered && !matches!(opened, Opened::Body(_) | Opened::Retryable(..)) {
            *object.delivery.lock().unwrap() = None;
            let answered = Refusal {
                code: Code::TRANSFER_FAILED,
                detail: "the origin's redirect target refused this ask".into(),
            };
            return Err(Ended::Retry(answered, Some(Duration::ZERO)));
        }
        let response = match opened {
            Opened::Body(response) => {
                if hops.len() > 1 {
                    *object.delivery.lock().unwrap() = hops.pop();
                }
                response
            }
            Opened::Retryable(status, stated) => {
                let answered = Refusal {
                    code: Code::TRANSFER_FAILED,
                    detail: format!("the object store answered {status}"),
                };
                // A wait no clock can hold is no statement.
                let stated = stated.filter(|wait| Instant::now().checked_add(*wait).is_some());
                return Err(Ended::Retry(answered, stated));
            }
            // The signature was refused: the next ask re-authorizes, if that is honest.
            Opened::Expired(status) => {
                chunk.stale = Some(status);
                let answered = Refusal {
                    code: Code::TRANSFER_FAILED,
                    detail: format!("the object store answered {status}"),
                };
                return Err(Ended::Retry(answered, Some(Duration::ZERO)));
            }
            // An origin's own answer no retry can change is a fact, not weather.
            Opened::Refused(refusal) => return Err(Ended::Fatal(refusal)),
        };
        // A ranged ask must be answered with that range of this object.
        let ranged = content_range(response.header("content-range"));
        // An object of another length than the plan declares is a fact no ask changes.
        let stated = match response.status {
            206 => ranged.map(|(_, _, total)| total),
            _ => response
                .header("content-length")
                .and_then(|length| length.trim().parse().ok()),
        };
        if let Some(total) = stated.filter(|total| *total != wanted.length) {
            return Err(Ended::Fatal(Refusal {
                code: Code::LENGTH_MISMATCH,
                detail: format!(
                    "the plan declares {} B and the origin says {total} B",
                    wanted.length
                ),
            }));
        }
        match (response.status, ranged) {
            (206, Some((first, _, _))) if first == chunk.first => {
                if std::mem::take(&mut chunk.lead) {
                    self.rest(chunk);
                }
            }
            (200, _) if whole => {}
            // A range answered with the whole object: take it from its first byte. An origin
            // that does not range gives the object to this one request.
            (200, _) if chunk.lead || wanted.length <= job.chunk => {
                // What this chunk's earlier asks moved is moved again.
                let again = chunk.first - chunk.unit.0;
                job.store.discarded(again);
                self.landed.fetch_sub(again, Ordering::Relaxed);
                chunk.first = 0;
                if chunk.lead {
                    chunk.end = wanted.length;
                    chunk.unit = (0, wanted.length);
                    object.left.store(1, Ordering::Release);
                    object.rest.lock().unwrap().clear();
                    *object.units.lock().unwrap() = HashMap::from([(0, (wanted.length, 1))]);
                }
            }
            (status, ranged) => {
                return Err(Ended::Fatal(Refusal {
                    code: Code::TRANSFER_FAILED,
                    detail: format!(
                        "asked for bytes {}-{} of {} B and the origin answered {status} with \
                         Content-Range {ranged:?}",
                        chunk.first,
                        chunk.end - 1,
                        wanted.length
                    ),
                }));
            }
        }
        let mut body = response.into_body();
        while chunk.first < chunk.end {
            let want = (chunk.end - chunk.first).min(buffer.len() as u64) as usize;
            let read = body.read_refusing(&mut buffer[..want])?;
            if read == 0 {
                return Err(Ended::Retry(
                    Refusal {
                        code: Code::TRANSFER_FAILED,
                        detail: format!(
                            "the origin closed its answer {} B short",
                            chunk.end - chunk.first
                        ),
                    },
                    None,
                ));
            }
            if let Err(error) = self
                .writers
                .write(&object.file, &buffer[..read], chunk.first)
            {
                return Err(Ended::Fatal(crate::store::classify_io(format!("write {}", object.path.display()), error)));
            }
            chunk.first += read as u64;
            object.moved.fetch_add(read as u64, Ordering::Relaxed);
            self.landed.fetch_add(read as u64, Ordering::Relaxed);
            job.store.moved(read as u64);
        }
        Ok(())
    }

    /// Every chunk is home: hash the file and commit it at the digest it hashed to.
    fn commit(&self, object: &Object) -> Outcome {
        let job = self.job;
        let wanted = &object.grant.object;
        let moved = object.moved.load(Ordering::Relaxed);
        let fault = job
            .keep
            .map_or(&Fault::default(), |keep| keep.fault)
            .clone();
        let admitted = (|| {
            if job.store.flushes() {
                object.file.sync_all().map_err(|error| crate::store::classify_io(format!("fsync {}", object.path.display()), error))?;
            }
            let mut reader = Verifying::new(&object.file, wanted.length);
            if object.grant.destination == Destination::Manifest {
                return object.grant.admit(job.store, &mut reader).map(|_| ());
            }
            let sha256 = reader.digest()?;
            if sha256 != wanted.sha256 {
                return refuse(
                    Code::OBJECT_ID_MISMATCH,
                    format!("expected sha256:{}, fetched sha256:{sha256}", wanted.sha256),
                );
            }
            let verified = Verified {
                tmp: object.path.clone(),
                file: object.file.try_clone().map_err(|error| crate::store::classify_io(format!("clone {}", object.path.display()), error))?,
                sha256,
                length: wanted.length,
            };
            fault.hit("verified");
            job.store.admit_verified(verified, &fault, None).map(|_| ())
        })();
        // Wrong bytes are the door's verdict on the origin: nothing is asked again, and a
        // kept download starts over the next time it is asked for.
        if let Err(refusal) = admitted {
            job.store.discarded(moved);
            if let (Some(part), Some(keep), Code::OBJECT_ID_MISMATCH) =
                (&object.part, job.keep, refusal.code)
            {
                match part.reset() {
                    Ok(forgot) => keep.landed(-(forgot as i64)),
                    Err(reset) => return Outcome::Fatal(named(reset, &wanted.id())),
                }
            }
            return Outcome::Fatal(named(refusal, &wanted.id()));
        }
        if let (Some(part), Some(keep)) = (&object.part, job.keep) {
            fault.hit("admitted");
            part.finished();
            keep.landed((wanted.length - part.covered()) as i64);
        }
        job.ledger.answered();
        self.fetched.fetch_add(1, Ordering::Relaxed);
        self.bytes_moved.fetch_add(moved, Ordering::Relaxed);
        if let (Some((kind, trust)), Some(backfill)) = (object.cache, job.backfill) {
            // Local admission is complete; replication never delays this object's report.
            backfill.offer(kind, wanted, trust);
        }
        if let Some(observer) = job.on_object {
            observer(wanted, moved, ObjectSource::Origin);
        }
        Outcome::Home
    }

    /// The one rule, five times a sample: a request that has waited far longer for its first
    /// byte than the requests beside it, or whose body runs far slower than theirs over the
    /// same time, is dropped. Peers are the requests still running and those that ended
    /// since its window began, never history, so a pod whose every request is slow drops
    /// none. And the pull's own end: nothing new has landed for twice its measured patience.
    fn watch(&self, observer: &mut Option<&mut dyn FetchObserver>) {
        let job = self.job;
        let sample = self.tick();
        // The most bytes ever home, and when that last grew while requests were being asked.
        let (mut most, mut grew) = (0, Instant::now());
        let mut state = self.state.lock().unwrap();
        loop {
            if state.done || state.failure.is_some() {
                return;
            }
            state = self.over.wait_timeout(state, sample).unwrap().0;
            if let Some(observer) = observer.as_mut() {
                // Told outside the lock: an observer may be slow, and workers must not wait.
                drop(state);
                observer.moved(self.landed.load(Ordering::Relaxed));
                state = self.state.lock().unwrap();
            }
            if state.done || state.failure.is_some() {
                return;
            }
            let stopped = job
                .ledger
                .check_running()
                .and_then(|()| job.deadline.remaining().map(|_| ()));
            if let Err(refusal) = stopped {
                self.fail(&mut state, refusal);
                self.wake.notify_all();
                return;
            }
            let floor = sample.as_secs_f64() * 2.0;
            let now = Instant::now();
            let live: Vec<(usize, Peer)> = (self.flights.iter().enumerate())
                .filter_map(|(slot, flight)| {
                    let flight = flight.lock().unwrap();
                    let flight = flight.as_ref()?;
                    let peer = Peer {
                        began: flight.began,
                        first: flight.attempt.first.get().copied(),
                        ended: now,
                        open: true,
                        bytes: flight.attempt.bytes.load(Ordering::Relaxed),
                        small: flight.range.1 - flight.range.0 < PIECE_BYTES,
                    };
                    Some((slot, peer))
                })
                .collect();
            // A peer's figure, and whether it is only a floor because the peer still runs.
            let secs =
                |from: Instant, to: Instant| to.saturating_duration_since(from).as_secs_f64();
            let wait = |peer: &Peer| {
                Some(
                    peer.first
                        .map_or((secs(peer.began, peer.ended), peer.open), |first| {
                            (secs(peer.began, first), false)
                        }),
                )
            };
            let took = |peer: &Peer| {
                peer.small
                    .then(|| (secs(peer.began, peer.ended), peer.open))
            };
            let rate = |peer: &Peer| {
                let first = peer.first.filter(|_| !peer.small)?;
                let body = secs(first, peer.ended);
                (peer.bytes >= PIECE_BYTES || body >= sample.as_secs_f64())
                    .then(|| (peer.bytes as f64 / body.max(1e-6), false))
            };
            // The middle of the others running now and those ended since `since`. A floor
            // under `least` says nothing yet, and is left out.
            let shown = &state.shown;
            let peers = |slot: usize, since: Instant, least: f64, figure: &Figure| {
                let ended = shown.range(shown.partition_point(|peer| peer.ended < since)..);
                let values = (live.iter())
                    .filter(|(other, _)| *other != slot)
                    .map(|(_, peer)| peer)
                    .chain(ended)
                    .filter_map(figure)
                    .filter_map(|(value, floor)| (!floor || value >= least).then_some(value))
                    .collect();
                median(values).filter(|value| *value > 0.0)
            };
            let mut flying = false;
            for (slot, flight) in self.flights.iter().enumerate() {
                let mut flight = flight.lock().unwrap();
                let Some(flight) = flight.as_mut() else {
                    continue;
                };
                flying = true;
                let bytes = flight.attempt.bytes.load(Ordering::Relaxed);
                let (began, waited) = (flight.began, flight.began.elapsed().as_secs_f64());
                let patience = flight.patience as f64;
                // Far longer at it than its peers took: for its first byte, or in all.
                let longer = |figure: &Figure| {
                    let least = waited / (STALL_FACTOR * patience);
                    (waited > floor * patience)
                        .then(|| peers(slot, began, least, figure))
                        .flatten()
                        .filter(|typical| waited > (typical * STALL_FACTOR).max(floor) * patience)
                };
                let slow = match flight.attempt.first.get() {
                    // Less than a piece to fetch, against its small peers' time in all.
                    _ if flight.range.1 - flight.range.0 < PIECE_BYTES => longer(&took)
                        .map(|took| (bytes as f64 / waited, format!("peers_small_s={took:.2}"))),
                    None => longer(&wait).map(|wait| (0.0, format!("peers_wait_s={wait:.2}"))),
                    // A body is judged once its peers' pace would have brought a piece of it
                    // home, and over no less than two ticks: then far slower than theirs.
                    Some(first) => {
                        let since = flight.judged.unwrap_or(began);
                        let window = flight.judged.unwrap_or(*first).elapsed().as_secs_f64();
                        (window >= floor * patience)
                            .then(|| peers(slot, since, 0.0, &rate))
                            .flatten()
                            .filter(|typical| window >= (PIECE_BYTES as f64 / typical) * patience)
                            .and_then(|typical| {
                                let rate = (bytes - flight.seen) as f64 / window;
                                (flight.judged, flight.seen) = (Some(Instant::now()), bytes);
                                (rate * STALL_FACTOR < typical)
                                    .then(|| (rate, format!("peers_bps={typical:.0}")))
                            })
                    }
                };
                let Some((rate, peers)) = slow else {
                    continue;
                };
                flight.attempt.stopped.store(true, Ordering::Release);
                self.restarts.fetch_add(1, Ordering::Relaxed);
                decisions::record(
                    job.store,
                    "restart",
                    &flight.subject,
                    &format!(
                        "bytes={}-{} got={bytes} rate_bps={rate:.0} {peers} after_s={waited:.1} \
                         patience={}",
                        flight.range.0,
                        flight.range.1 - 1,
                        flight.patience,
                    ),
                );
            }
            // The pull's own end: requests are being asked and nothing new has landed for
            // twice its patience, so every silent one was asked again and that came to
            // nothing too. A wait the hub or the origin asked for is theirs, not judged here.
            let now = Instant::now();
            let asking = flying
                || state
                    .queue
                    .iter()
                    .any(|chunk| chunk.first < chunk.end && !(chunk.stated && chunk.due > now));
            let landed = self.landed.load(Ordering::Relaxed);
            let elapsed = self.began.elapsed();
            let on_hub = self.minting.iter().any(|since| {
                let since = since.load(Ordering::Acquire);
                since != 0 && elapsed - Duration::from_millis(since - 1) >= sample
            });
            if landed > most || !asking || on_hub {
                (most, grew) = (most.max(landed), now);
                continue;
            }
            let minted = self.began + Duration::from_millis(self.minted.load(Ordering::Acquire));
            grew = grew.max(minted);
            let patience = job.ledger.stillness() * 2;
            if grew.elapsed() > patience {
                let last = state.last.take();
                let refusal = Refusal {
                    code: Code::TRANSFER_FAILED,
                    detail: format!(
                        "the pull stopped: nothing landed for {:.1} s, twice the {:.1} s its \
                         own measured pace allows, with {most} B home{}",
                        grew.elapsed().as_secs_f64(),
                        patience.as_secs_f64() / 2.0,
                        last.map_or(String::new(), |last| format!(
                            "; last answer: {}",
                            last.detail
                        )),
                    ),
                };
                self.fail(&mut state, refusal);
                self.wake.notify_all();
                return;
            }
        }
    }
}

/// A file read from its first byte as the door reads it, and its digest.
struct Verifying<'a> {
    file: &'a File,
    at: u64,
    length: u64,
}

impl<'a> Verifying<'a> {
    fn new(file: &'a File, length: u64) -> Self {
        Verifying {
            file,
            at: 0,
            length,
        }
    }

    fn digest(&mut self) -> Result<String> {
        let mut hash = crate::sha256::Sha256::new();
        let mut buffer = vec![0u8; 1 << 20];
        loop {
            let read = self.read(&mut buffer).map_err(|error| crate::store::classify_io("read back the downloaded file", error))?;
            if read == 0 {
                return Ok(crate::sha256::hex(&hash.finish()));
            }
            hash.update(&buffer[..read]);
        }
    }
}

impl Read for Verifying<'_> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let want = (self.length - self.at).min(out.len() as u64) as usize;
        let read = self.file.read_at(&mut out[..want], self.at)?;
        self.at += read as u64;
        Ok(read)
    }
}

/// One figure of a peer, and whether it is only a floor.
type Figure<'a> = dyn Fn(&Peer) -> Option<(f64, bool)> + 'a;

/// The middle of what peers showed; none with no peer to compare with.
fn median(mut values: Vec<f64>) -> Option<f64> {
    let middle = values.len() / 2;
    (!values.is_empty()).then(|| *values.select_nth_unstable_by(middle, f64::total_cmp).1)
}

fn named(refusal: Refusal, id: &str) -> Refusal {
    if refusal.detail.contains(id) {
        return refusal;
    }
    Refusal {
        detail: format!("{id}: {}", refusal.detail),
        ..refusal
    }
}

/// `bytes <first>-<last>/<total>`; a `*` total is no answer.
pub(super) fn content_range(header: Option<&str>) -> Option<(u64, u64, u64)> {
    let (range, total) = header?.trim().strip_prefix("bytes ")?.split_once('/')?;
    let (first, last) = range.split_once('-')?;
    Some((
        first.trim().parse().ok()?,
        last.trim().parse().ok()?,
        total.trim().parse().ok()?,
    ))
}

/// How one source object is asked for: requests in flight, and one request's bytes.
#[derive(Debug, Clone, Copy)]
pub struct Ranged {
    pub streams: usize,
    pub part: u64,
}

impl Default for Ranged {
    fn default() -> Ranged {
        Ranged {
            streams: STREAMS,
            part: PART_BYTES,
        }
    }
}

/// One granted object from a URL the caller holds, through the pull's downloader. Nothing
/// re-authorizes the URL: an origin that refuses it ends the fetch. A fetch that fails
/// stops `ledger` with it.
#[allow(clippy::too_many_arguments)]
pub fn fetch_ranged(
    store: &Store,
    grant: &DeliveryGrant,
    url: &str,
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &Ledger,
    ranged: Ranged,
    observer: Option<&mut dyn FetchObserver>,
) -> Result<Fetched> {
    let urls = BTreeMap::from([(grant.object.sha256.clone(), url.to_string())]);
    let outcome = Download {
        store,
        wanted: std::slice::from_ref(&grant.object),
        grant: &|_| Ok(grant.clone()),
        urls: &urls,
        chunk: ranged.part.clamp(1 << 20, 1 << 30),
        ranged: false,
        policy,
        credential,
        deadline,
        ledger,
        on_object: None,
        backfill: None,
        keep: None,
    }
    .run(ranged.streams.clamp(1, 64), observer)?;
    Ok(Fetched {
        object: grant.object.clone(),
        transferred: outcome.bytes_moved,
    })
}

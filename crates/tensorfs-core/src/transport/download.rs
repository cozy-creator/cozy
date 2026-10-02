//! The pull's downloader, the way an S3 client does it. The wanted objects are cut into
//! fixed ranged chunks on one queue, `streams` workers each GET one chunk into its
//! object's temp file, and a finished file is hashed and committed to the Store.
//!
//! One rule per request: one that fails, or moves bytes far slower than the first asks that
//! came home (or none), is dropped, and the rest of its chunk is asked again on fresh
//! connections, in 1 MiB pieces. Nothing counts asks. The pull ends on an answer asking
//! again cannot change, or when nothing new has landed for twice its measured patience.
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

/// One request's bytes from a source origin. Larger, because every ask walks the origin's
/// redirect again (0.41 s against HuggingFace, and its resolve endpoint is rate-limited).
pub const PART_BYTES: u64 = 64 << 20;

/// A dropped request's unfinished range is cut into pieces, an eighth of a chunk each and
/// never under this: a slow object answers small asks side by side, and a chunk asked again
/// whole would pay its slow MiBs in a row. Also the least a request must carry for its body
/// rate to be compared with its peers'.
const PIECE_BYTES: u64 = 1 << 20;

/// Completed first asks a running request is compared with: how long each waited for its
/// first byte, and how fast its body then came.
const PEERS: usize = 64;

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
    /// Asks in a row dropped with no byte moved. Each doubles the time the next ask gets
    /// before it is judged, so an origin that is honestly slow to answer still does.
    silent: u32,
    /// Never dropped or failed: a first ask, whose rate is what peers are compared with.
    fresh: bool,
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
    /// How many times its peers' wait this request gets for its first byte.
    patience: u32,
    /// When its body was last judged, and with how many bytes.
    judged: Option<Instant>,
    seen: u64,
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
    /// The last first asks that came home. Those that carried a piece's worth of bytes:
    /// seconds to their first byte, and their body's rate. The smaller ones: seconds in all.
    waits: VecDeque<f64>,
    rates: VecDeque<f64>,
    small: VecDeque<f64>,
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
                "fetched={} cached={} moved={} seconds={:.1} streams={streams} requests={} \
                 restarts={} outcome={}",
                outcome.fetched,
                outcome.cached,
                outcome.bytes_moved,
                began.elapsed().as_secs_f64(),
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
        let kind = match grant.destination {
            Destination::Manifest => Some(CacheKind::Manifest),
            Destination::Blob => Some(CacheKind::Blob),
            Destination::Staging => None,
        };
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
                let file = part.file.try_clone().map_err(|error| Refusal {
                    code: Code::IO_FAILED,
                    detail: format!("source download: {error}"),
                })?;
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
            silent: 0,
            fresh: true,
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
                silent: 0,
                fresh: true,
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
        let before = part.covered();
        part.record(chunk.unit.0, end, keep.fault)?;
        keep.landed((part.covered() - before) as i64);
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
            patience: 1 << chunk.silent.min(6),
            judged: None,
            seen: 0,
        });
        self.requests.fetch_add(1, Ordering::Relaxed);
        let ended = self.stream(&mut chunk, &url, delivery.is_some(), &ledger, buffer);
        *self.flights[slot].lock().unwrap() = None;
        // What this ask added: a whole answer to a resumed range starts the object over.
        let moved = chunk.first.saturating_sub(asked.0);
        let (refusal, stated) = match ended {
            Ok(()) => {
                if let (true, Some(first)) = (chunk.fresh, attempt.first.get()) {
                    let wait = first.duration_since(began).as_secs_f64();
                    let body = first.elapsed().as_secs_f64().max(1e-6);
                    let mut state = self.state.lock().unwrap();
                    if moved >= PIECE_BYTES {
                        remember(&mut state.waits, wait);
                        remember(&mut state.rates, moved as f64 / body);
                    } else {
                        remember(&mut state.small, wait + body);
                    }
                }
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
        chunk.fresh = false;
        chunk.silent = match (dropped, moved) {
            (true, 0) => chunk.silent + 1,
            _ => 0,
        };
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
        // An object that is one chunk is one plain GET.
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
            if whole { &[] } else { &range },
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
            if let Err(error) = object.file.write_all_at(&buffer[..read], chunk.first) {
                return Err(Ended::Fatal(Refusal {
                    code: Code::IO_FAILED,
                    detail: format!("write {}: {error}", object.path.display()),
                }));
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
                object.file.sync_all().map_err(|error| Refusal {
                    code: Code::IO_FAILED,
                    detail: format!("fsync {}: {error}", object.path.display()),
                })?;
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
                file: object.file.try_clone().map_err(|error| Refusal {
                    code: Code::IO_FAILED,
                    detail: format!("clone {}: {error}", object.path.display()),
                })?,
                sha256,
                length: wanted.length,
            };
            if object.grant.destination == Destination::Staging {
                return crate::staging::adopt(job.store, verified).map(|_| ());
            }
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
                keep.landed(-(part.covered() as i64));
                if let Err(reset) = part.reset() {
                    return Outcome::Fatal(named(reset, &wanted.id()));
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
    /// byte than its peers did, or whose body runs far slower than theirs, is dropped. And
    /// the pull's own end: nothing new has landed for twice its measured patience.
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
            let (wait, typical) = (median(&state.waits), median(&state.rates));
            let small = median(&state.small);
            let floor = sample.as_secs_f64() * 2.0;
            let mut flying = false;
            for flight in &self.flights {
                let mut flight = flight.lock().unwrap();
                let Some(flight) = flight.as_mut() else {
                    continue;
                };
                flying = true;
                let bytes = flight.attempt.bytes.load(Ordering::Relaxed);
                let waited = flight.began.elapsed().as_secs_f64();
                let patience = flight.patience as f64;
                let slow = match flight.attempt.first.get() {
                    // Less than a piece to fetch, and far longer at it than its small peers
                    // took in all. Never sooner than the two ticks a body is judged over.
                    _ if flight.range.1 - flight.range.0 < PIECE_BYTES && small > 0.0 => {
                        let allowed = (small * STALL_FACTOR).max(floor) * patience;
                        (waited > allowed).then_some(bytes as f64 / waited)
                    }
                    // No byte yet, after far longer than its peers waited for theirs.
                    None => {
                        let allowed = (wait * STALL_FACTOR).max(floor) * patience;
                        (wait > 0.0 && waited > allowed).then_some(0.0)
                    }
                    // A body judged over two ticks, from its first byte.
                    Some(first) => {
                        let since = *flight.judged.get_or_insert(*first);
                        let window = since.elapsed();
                        (window >= sample * 2)
                            .then(|| {
                                let rate = (bytes - flight.seen) as f64 / window.as_secs_f64();
                                (flight.judged, flight.seen) = (Some(Instant::now()), bytes);
                                rate
                            })
                            .filter(|rate| rate * STALL_FACTOR < typical)
                    }
                };
                let Some(rate) = slow else {
                    continue;
                };
                flight.attempt.stopped.store(true, Ordering::Release);
                self.restarts.fetch_add(1, Ordering::Relaxed);
                decisions::record(
                    job.store,
                    "restart",
                    &flight.subject,
                    &format!(
                        "bytes={}-{} got={bytes} rate_bps={rate:.0} peers_bps={typical:.0} \
                         peers_wait_s={wait:.2} peers_small_s={small:.2} after_s={waited:.1}",
                        flight.range.0,
                        flight.range.1 - 1,
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
            let read = self.read(&mut buffer).map_err(|error| Refusal {
                code: Code::IO_FAILED,
                detail: format!("read back the downloaded file: {error}"),
            })?;
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

fn remember(peers: &mut VecDeque<f64>, value: f64) {
    if peers.len() == PEERS {
        peers.pop_front();
    }
    peers.push_back(value);
}

/// The middle of what the last peers showed; zero with none to compare with.
fn median(peers: &VecDeque<f64>) -> f64 {
    let mut sorted: Vec<f64> = peers.iter().copied().collect();
    sorted.sort_by(f64::total_cmp);
    sorted.get(sorted.len() / 2).copied().unwrap_or(0.0)
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

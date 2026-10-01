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

use super::decisions;
use super::ledger::{Attempt, Deadline, Ledger, STALL_FACTOR};
use super::pull::{
    open_object, shared_client, ObjectObserver, ObjectSource, ObjectUrls, Opened, WalkOutcome,
};
use super::{CredentialProvider, SourcePolicy};
use crate::err::{refuse, Code, Refusal, Result};
use crate::fetch::{DeliveryGrant, Destination, FetchPlan};
use crate::repo_cache::{BackfillBatch, CacheKind, CacheRead, CacheTrust};
use crate::store::{Fault, Store, Verified, PUT_TEMP};
use fs2::FileExt;
use std::collections::VecDeque;
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

/// What a dropped request's unfinished range is cut into: a slow object answers small
/// asks side by side, and a chunk asked again whole would pay its slow MiBs in a row.
const PIECE_BYTES: u64 = 1 << 20;

/// Completed requests whose rates are the peers a running one is compared with.
const PEERS: usize = 64;

/// An object being downloaded: its temp file, and what is still out.
struct Object {
    /// Its place in `plan.wanted`, which is how the URL supply names it.
    at: usize,
    grant: DeliveryGrant,
    cache: Option<(CacheKind, CacheTrust)>,
    path: PathBuf,
    file: File,
    /// Chunks not yet home.
    left: AtomicUsize,
    /// Bytes requests wrote for it, bytes asked twice included.
    moved: AtomicU64,
}

impl Drop for Object {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
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
    /// Ticks a verdict is taken over, and when and with how many bytes the last one was.
    grace: u32,
    judged: Instant,
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
            // A connection that failed, stalled or was dropped, and a full descriptor table.
            Code::TRANSFER_FAILED | Code::FD_HEADROOM => Ended::Retry(refusal, None),
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
    /// Rates of the last requests that came home with a chunk's worth of bytes.
    peers: VecDeque<f64>,
    /// The last refusal a retry answered, quoted if the pull stops moving.
    last: Option<Refusal>,
    failure: Option<Refusal>,
    done: bool,
}

pub(super) struct Download<'a> {
    pub store: &'a Store,
    pub plan: &'a FetchPlan,
    pub urls: &'a dyn ObjectUrls,
    pub policy: &'a SourcePolicy,
    pub credential: &'a dyn CredentialProvider,
    pub deadline: Deadline,
    pub ledger: &'a Ledger,
    pub on_object: Option<ObjectObserver<'a>>,
    pub backfill: Option<&'a BackfillBatch<'a>>,
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
    /// Download `plan.wanted` with `streams` requests in flight.
    pub fn run(&self, streams: usize) -> Result<WalkOutcome> {
        let began = Instant::now();
        let chunks: usize = self
            .plan
            .wanted
            .iter()
            .map(|object| object.length.div_ceil(CHUNK_BYTES).max(1) as usize)
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
            run.watch();
        });
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
            &format!("{} {}", self.plan.wanted.len(), self.plan.wanted_bytes()),
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
        let wanted = &self.job.plan.wanted;
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
        let wanted = &job.plan.wanted[at];
        let grant = DeliveryGrant::mint(job.plan, wanted)?;
        let kind = match grant.destination {
            Destination::Manifest => Some(CacheKind::Manifest),
            Destination::Blob => Some(CacheKind::Blob),
            Destination::Staging => None,
        };
        let mut cache = None;
        if let (Some(kind), Some(mounted)) = (kind, job.store.repo_cache()) {
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
        let path = job.store.root().join("tmp").join(format!(
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
        // The Store's own temp name and lock, so its reaper knows a live writer from an orphan.
        let sized = file
            .try_lock_exclusive()
            .and_then(|()| file.set_len(length));
        if let Err(error) = sized {
            let _ = std::fs::remove_file(&path);
            return Err(io("size", error));
        }
        let firsts: Vec<u64> = (0..length).step_by(CHUNK_BYTES as usize).collect();
        let object = Arc::new(Object {
            at,
            grant,
            cache,
            path,
            file,
            left: AtomicUsize::new(firsts.len()),
            moved: AtomicU64::new(0),
        });
        let chunk = |first, end| Chunk {
            object: Arc::clone(&object),
            first,
            end,
            due: Instant::now(),
            stated: false,
            stale: None,
            silent: 0,
            fresh: true,
        };
        if firsts.is_empty() {
            return Ok(vec![chunk(0, 0)]);
        }
        Ok(firsts
            .into_iter()
            .map(|first| chunk(first, (first + CHUNK_BYTES).min(length)))
            .collect())
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
        let url = job
            .urls
            .url_for_while(object.at, wanted, stale.is_some(), job.ledger);
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
            // Two ticks, so the time to a first byte is not read as a slow body.
            grace: 2 << chunk.silent.min(6),
            judged: began,
            seen: 0,
        });
        self.requests.fetch_add(1, Ordering::Relaxed);
        let ended = self.stream(&mut chunk, &url, &ledger, buffer);
        *self.flights[slot].lock().unwrap() = None;
        // What this ask added: a whole answer to a resumed range starts the object over.
        let moved = chunk.first.saturating_sub(asked.0);
        let (refusal, stated) = match ended {
            Ok(()) => {
                if chunk.fresh && moved >= PIECE_BYTES {
                    let rate = moved as f64 / began.elapsed().as_secs_f64().max(1e-6);
                    let mut state = self.state.lock().unwrap();
                    if state.peers.len() == PEERS {
                        state.peers.pop_front();
                    }
                    state.peers.push_back(rate);
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
        while dropped && chunk.end - chunk.first > PIECE_BYTES {
            let end = chunk.first + PIECE_BYTES;
            object.left.fetch_add(1, Ordering::AcqRel);
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
        let response = match open_object(
            shared_client(),
            url,
            job.policy,
            job.credential,
            job.deadline,
            ledger,
            if whole { &[] } else { &range },
            &mut |_, _| {},
        )? {
            Opened::Body(response) => response,
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
        let ranged = super::ranged::content_range(response.header("content-range"));
        match (response.status, ranged) {
            (206, Some((first, _, total))) if first == chunk.first && total == wanted.length => {}
            (200, _) if whole => {}
            // A resumed range answered with the whole object: take it from its first byte.
            (200, _) if wanted.length <= CHUNK_BYTES => {
                job.store.discarded(chunk.first);
                self.landed.fetch_sub(chunk.first, Ordering::Relaxed);
                chunk.first = 0;
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
            job.store
                .admit_verified(verified, &Fault::default(), None)
                .map(|_| ())
        })();
        // Wrong bytes are the door's verdict on the origin: nothing is asked again.
        if let Err(refusal) = admitted {
            job.store.discarded(moved);
            return Outcome::Fatal(named(refusal, &wanted.id()));
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

    /// The one rule, five times a sample: a request moving bytes far slower than the first
    /// asks that came home, or none at all, is dropped. And the pull's own end: nothing new
    /// has landed for twice its measured patience.
    fn watch(&self) {
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
            let mut peers: Vec<f64> = state.peers.iter().copied().collect();
            peers.sort_by(f64::total_cmp);
            let typical = peers.get(peers.len() / 2).copied().unwrap_or(0.0);
            let mut flying = false;
            for flight in &self.flights {
                let mut flight = flight.lock().unwrap();
                let Some(flight) = flight.as_mut() else {
                    continue;
                };
                flying = true;
                let window = flight.judged.elapsed();
                if window < sample * flight.grace {
                    continue;
                }
                let bytes = flight.attempt.bytes.load(Ordering::Relaxed);
                let rate = (bytes - flight.seen) as f64 / window.as_secs_f64();
                (flight.judged, flight.seen) = (Instant::now(), bytes);
                if rate * STALL_FACTOR >= typical {
                    continue;
                }
                flight.attempt.stopped.store(true, Ordering::Release);
                self.restarts.fetch_add(1, Ordering::Relaxed);
                decisions::record(
                    job.store,
                    "restart",
                    &flight.subject,
                    &format!(
                        "bytes={}-{} got={bytes} rate_bps={rate:.0} peers_bps={typical:.0} \
                         after_s={:.1}",
                        flight.range.0,
                        flight.range.1 - 1,
                        flight.began.elapsed().as_secs_f64()
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

fn named(refusal: Refusal, id: &str) -> Refusal {
    if refusal.detail.contains(id) {
        return refusal;
    }
    Refusal {
        detail: format!("{id}: {}", refusal.detail),
        ..refusal
    }
}

//! The pull's downloader, the way an S3 client does it. The wanted objects are cut into
//! fixed ranged chunks on one queue, `streams` workers each GET one chunk into its
//! object's temp file, and a finished file is hashed and committed to the Store. A worker is
//! a task on the network runtime (`net`), and its writes, hashes and blocking calls go to two
//! small pools: a run's threads are as many with 512 requests in flight as with one, and
//! every run in the process shares one budget of requests in flight (`in_flight`).
//!
//! One rule per request: one that fails, or waits or moves far slower than the requests
//! running beside it over the same time, is dropped, and the rest of its chunk is asked
//! again on fresh connections, in 1 MiB pieces. A pod whose every request is slow restarts
//! none (vast 5090s, 2026-10-10: 885 restarts in 42 s against a median of the few early
//! asks that came home). A batch ends on an answer asking again cannot change, or when
//! nothing new has landed for twice its measured patience. A lasting feed then asks again
//! with URLs minted afresh, for as long as any download into its Store moves, and once when
//! none does; its objects end only on the origin refusing a fresh URL, wrong bytes, or that.
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
//!
//! Objects come from a [`Feed`]: a fixed [`Batch`] (a pull, a source download, one ranged
//! fetch), or [`crate::fetch::Fetcher`]'s queue, which grows while the download runs. Each
//! open object has a [`Cell`]: its urgency orders the queue, and a request for a lower class
//! (Now, Next, Behind) gives way to waiting work of a higher one. A batch's refusal ends it;
//! a lasting feed's refusal ends only its object.

use super::credential::{Answer, Ask};
use super::decisions;
use super::ledger::{Attempt, Deadline, Ledger, STALL_FACTOR};
use super::net;
use super::pull::{
    absolute_location, shared_client, FetchObserver, Fetched, ObjectObserver, ObjectSource,
    ObjectUrls, WalkOutcome, RETRYABLE_STATUS,
};
use super::sources::{missing, Keep, Part};
use super::STREAMS;
use super::http::Judge;
use super::{CredentialProvider, SourcePolicy};
use crate::err::{refuse, Code, Refusal, Result};
use crate::fetch::{DeliveryGrant, Destination};
use crate::ids::ObjectRef;
use crate::store::{Fault, Store, Verified, PUT_TEMP};
use fs2::FileExt;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::os::unix::fs::{FileExt as _, OpenOptionsExt};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use futures::FutureExt;
use std::panic::AssertUnwindSafe;
use std::sync::{mpsc, Arc, Mutex, PoisonError, RwLock};
use std::time::{Duration, Instant};

/// One request's bytes. Measured on a pod at 128 workers: 4, 8 and 16 MiB chunks all move
/// what whole 64 MiB GETs move (1.1-1.3 GB/s); 1 MiB chunks move half.
pub const CHUNK_BYTES: u64 = 8 << 20;

/// The most writes into temp files at once, whatever the requests in flight: the write
/// pool's threads. A buffered write is a copy into the page cache: the kernel's writeback
/// writes the disk, and one file's writes take turns on its inode lock, so more writers add no
/// bytes, only threads stalled in the kernel. With one writer per request (512) a 6.5 GiB pull
/// on a laptop left 180 threads in uninterruptible sleep, load at 396 and the disk's IO
/// pressure full for ten minutes. Requests beyond this wait their turn holding what they
/// gathered, so in-flight unwritten bytes stay at most one write per request.
pub const WRITERS: usize = 8;

/// Jobs the async workers hand to threads: file writes, hashing and commits, and the hub,
/// DNS and admission calls that block. A few threads serve every request in flight.
type Job<'s> = Box<dyn FnOnce() + Send + 's>;

/// A download's threads for work that blocks: at most `threads` of them, started for the run
/// and ended with it. A thread the container refuses is backpressure; with none, a job runs
/// on its caller.
struct Offload<'s> {
    jobs: Mutex<Option<mpsc::Sender<Job<'s>>>>,
    started: usize,
    busy: Arc<AtomicUsize>,
    /// The most jobs that ever ran together.
    peak: Arc<AtomicUsize>,
}

impl<'s> Offload<'s> {
    fn start<'e>(scope: &'s std::thread::Scope<'s, 'e>, threads: usize, name: &str) -> Offload<'s> {
        let (sender, receiver) = mpsc::channel::<Job<'s>>();
        let receiver = Arc::new(Mutex::new(receiver));
        let started = (0..threads)
            .take_while(|_| {
                let receiver = Arc::clone(&receiver);
                let serve = move || loop {
                    let job = receiver.lock().unwrap_or_else(PoisonError::into_inner).recv();
                    match job {
                        Ok(job) => job(),
                        Err(_) => return,
                    }
                };
                std::thread::Builder::new().name(name.into()).spawn_scoped(scope, serve).is_ok()
            })
            .count();
        Offload {
            jobs: Mutex::new(Some(sender)),
            started,
            busy: Arc::default(),
            peak: Arc::default(),
        }
    }

    /// `job` on one of the threads, awaited. A job that panics resumes its panic here.
    async fn run<T: Send + 's>(&self, job: impl FnOnce() -> T + Send + 's) -> T {
        let (answer, answered) = futures::channel::oneshot::channel();
        let (busy, peak) = (Arc::clone(&self.busy), Arc::clone(&self.peak));
        let job: Job<'s> = Box::new(move || {
            peak.fetch_max(busy.fetch_add(1, Ordering::AcqRel) + 1, Ordering::Relaxed);
            let _ = answer.send(std::panic::catch_unwind(AssertUnwindSafe(job)));
            busy.fetch_sub(1, Ordering::AcqRel);
        });
        let refused = match (self.started, self.jobs.lock().unwrap_or_else(PoisonError::into_inner).as_ref()) {
            (1.., Some(jobs)) => jobs.send(job).err().map(|refused| refused.0),
            _ => Some(job),
        };
        if let Some(job) = refused {
            job();
        }
        match answered.await {
            Ok(Ok(value)) => value,
            Ok(Err(panic)) => std::panic::resume_unwind(panic),
            Err(_) => panic!("an offloaded job ended without its answer"),
        }
    }

    /// No more jobs: the threads end once the queued ones are done.
    fn close(&self) {
        self.jobs.lock().unwrap_or_else(PoisonError::into_inner).take();
    }
}

/// A download's two pools: writes into temp files apart from the control work, so a slow
/// hub or a hashing commit never holds a request's bytes back.
struct Pools<'s> {
    writes: Offload<'s>,
    control: Offload<'s>,
}

/// Threads for control work: a URL mint, a DNS check, an admission, a commit's hash and fsync.
const CONTROL: usize = 8;

/// The requests this process keeps in flight, every download together: one budget for the
/// machine (TensorD owns its Store), never a fresh 512 per pull, and no more than its
/// descriptor table holds.
fn in_flight() -> &'static (usize, tokio::sync::Semaphore) {
    static BUDGET: std::sync::OnceLock<(usize, tokio::sync::Semaphore)> = std::sync::OnceLock::new();
    BUDGET.get_or_init(|| {
        let requests = crate::descriptors::requests(super::PULL_STREAMS);
        (requests, tokio::sync::Semaphore::new(requests))
    })
}

/// Bytes every download into `store` (the machine's) has brought home in this process, net
/// of bytes moved again. A run that stands still is asked again for as long as another moves.
fn moved(store: &Store) -> Arc<AtomicU64> {
    static MOVED: std::sync::LazyLock<Mutex<HashMap<PathBuf, Arc<AtomicU64>>>> =
        std::sync::LazyLock::new(Mutex::default);
    Arc::clone(MOVED.lock().unwrap().entry(store.root().to_path_buf()).or_default())
}

/// What a request gathers before handing a write to its pool: one write per this many bytes,
/// not one per network frame.
const WRITE_BYTES: usize = 256 << 10;

/// One request's bytes from a source origin. Larger, because every ask walks the origin's
/// redirect again (0.41 s against HuggingFace, and its resolve endpoint is rate-limited).
pub const PART_BYTES: u64 = 64 << 20;

/// A dropped request's unfinished range is cut into pieces, an eighth of a chunk each and
/// never under this: a slow object answers small asks side by side, and a chunk asked again
/// whole would pay its slow MiBs in a row. Also what a body is given time to move, at its
/// peers' pace, before it is judged.
const PIECE_BYTES: u64 = 1 << 20;

/// The classes of urgency, as a [`Cell`] key's top two bits.
pub(crate) const NOW: u64 = 0;
pub(crate) const NEXT: u64 = 1 << 62;
pub(crate) const BEHIND: u64 = 2 << 62;

/// What a feed and the download share about one open object: its urgency (a smaller key is
/// sooner; [`NOW`], [`NEXT`] or [`BEHIND`] in the top two bits) and whether it is still wanted.
pub(crate) struct Cell {
    key: AtomicU64,
    gone: AtomicBool,
}

impl Cell {
    pub fn new(key: u64) -> Arc<Cell> {
        Arc::new(Cell {
            key: AtomicU64::new(key),
            gone: AtomicBool::new(false),
        })
    }
    pub fn key(&self) -> u64 {
        self.key.load(Ordering::Acquire)
    }
    pub fn set(&self, key: u64) {
        self.key.store(key, Ordering::Release);
    }
    fn class(&self) -> u64 {
        self.key() >> 62
    }
    pub fn gone(&self) -> bool {
        self.gone.load(Ordering::Acquire)
    }
    /// Nothing wants the object any more: its chunks are dropped and its requests stopped.
    pub fn leave(&self) {
        self.gone.store(true, Ordering::Release);
    }
}

/// Wakes a download's workers and its watch: anything that changes what they may do beats.
#[derive(Default)]
pub(crate) struct Pulse {
    beats: Mutex<u64>,
    beaten: tokio::sync::Notify,
}

impl Pulse {
    pub fn seen(&self) -> u64 {
        *self.beats.lock().unwrap()
    }
    pub fn beat(&self) {
        *self.beats.lock().unwrap() += 1;
        self.beaten.notify_waiters();
    }
    /// Until a beat after `seen`, for at most `at_most`.
    async fn changed(&self, seen: u64, at_most: Duration) {
        let beaten = self.beaten.notified();
        tokio::pin!(beaten);
        beaten.as_mut().enable();
        if self.seen() != seen {
            return;
        }
        let _ = tokio::time::timeout(at_most, beaten).await;
    }
}

/// Where one object's URL, egress policy and credential come from.
pub(crate) trait Supply: Send + Sync {
    /// See [`ObjectUrls::url_for_while`]; `at` is the object's place in its batch.
    fn url(&self, at: usize, object: &ObjectRef, stale: bool, ledger: &Ledger) -> Result<String>;
    fn policy(&self) -> &SourcePolicy;
    fn credential(&self) -> &dyn CredentialProvider;
}

/// One object a feed hands its download.
pub(crate) struct Opening<'a> {
    pub at: usize,
    pub grant: DeliveryGrant,
    pub supply: Arc<dyn Supply + 'a>,
    pub cell: Arc<Cell>,
}

/// Where a download's objects come from, most urgent first, and where their ends are told.
pub(crate) trait Feed<'a>: Sync {
    /// The most urgent object waiting to be opened, if its key is below `before`.
    fn open(&self, before: u64) -> Result<Option<Opening<'a>>>;
    /// The key of the most urgent object waiting to be opened.
    fn waiting(&self) -> Option<u64>;
    /// One object (opened with `cell`) landed, from which door and with the bytes this
    /// download moved, or was refused.
    fn ended(&self, object: &ObjectRef, cell: &Cell, outcome: Result<(ObjectSource, u64)>);
    /// A refused object ends only itself; a feed that does not last ends at its first refusal.
    fn lasting(&self) -> bool;
    fn pulse(&self) -> &Pulse;
    /// Held shared while an object is admitted, so a collection can exclude admissions.
    fn gate(&self) -> Option<&RwLock<()>> {
        None
    }
}

/// URLs, policy and credential a caller holds for a whole batch.
struct Given<'a> {
    urls: &'a dyn ObjectUrls,
    policy: &'a SourcePolicy,
    credential: &'a dyn CredentialProvider,
}

impl Supply for Given<'_> {
    fn url(&self, at: usize, object: &ObjectRef, stale: bool, ledger: &Ledger) -> Result<String> {
        self.urls.url_for_while(at, object, stale, ledger)
    }
    fn policy(&self) -> &SourcePolicy {
        self.policy
    }
    fn credential(&self) -> &dyn CredentialProvider {
        self.credential
    }
}

/// A fixed list of objects opened in order, all equally urgent: its first refusal ends it.
pub(super) struct Batch<'a> {
    wanted: &'a [ObjectRef],
    grant: &'a (dyn Fn(&ObjectRef) -> Result<DeliveryGrant> + Sync),
    supply: Arc<dyn Supply + 'a>,
    on_object: Option<ObjectObserver<'a>>,
    next: Mutex<usize>,
    cell: Arc<Cell>,
    pulse: Pulse,
}

impl<'a> Batch<'a> {
    pub fn new(
        wanted: &'a [ObjectRef],
        grant: &'a (dyn Fn(&ObjectRef) -> Result<DeliveryGrant> + Sync),
        urls: &'a dyn ObjectUrls,
        policy: &'a SourcePolicy,
        credential: &'a dyn CredentialProvider,
        on_object: Option<ObjectObserver<'a>>,
    ) -> Batch<'a> {
        Batch {
            wanted,
            grant,
            supply: Arc::new(Given {
                urls,
                policy,
                credential,
            }),
            on_object,
            next: Mutex::new(0),
            cell: Cell::new(NEXT),
            pulse: Pulse::default(),
        }
    }

    /// `streams`, or fewer when the batch has fewer chunks of `chunk` bytes.
    pub fn streams(&self, streams: usize, chunk: u64) -> usize {
        let chunks: usize = (self.wanted.iter())
            .map(|object| object.length.div_ceil(chunk).max(1) as usize)
            .sum();
        streams.min(chunks).max(1)
    }
}

impl<'a> Feed<'a> for Batch<'a> {
    fn open(&self, _before: u64) -> Result<Option<Opening<'a>>> {
        let at = {
            let mut next = self.next.lock().unwrap();
            if *next >= self.wanted.len() {
                return Ok(None);
            }
            *next += 1;
            *next - 1
        };
        Ok(Some(Opening {
            at,
            grant: (self.grant)(&self.wanted[at])?,
            supply: Arc::clone(&self.supply),
            cell: Arc::clone(&self.cell),
        }))
    }
    fn waiting(&self) -> Option<u64> {
        (*self.next.lock().unwrap() < self.wanted.len()).then_some(NEXT)
    }
    fn ended(&self, object: &ObjectRef, _cell: &Cell, outcome: Result<(ObjectSource, u64)>) {
        if let (Some(observer), Ok((door, moved))) = (self.on_object, outcome) {
            observer(object, moved, door);
        }
    }
    fn lasting(&self) -> bool {
        false
    }
    fn pulse(&self) -> &Pulse {
        &self.pulse
    }
}

/// An object being downloaded: its file, and what is still out.
struct Object<'a> {
    /// Its place in its batch, which is how a batch's URL supply names it.
    at: usize,
    grant: DeliveryGrant,
    supply: Arc<dyn Supply + 'a>,
    cell: Arc<Cell>,
    /// Its end was told to the feed.
    told: AtomicBool,
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
    /// The pull stood still on it: its next ask holds a URL minted afresh.
    remint: AtomicBool,
    /// Bytes requests wrote for it, bytes asked twice included.
    moved: AtomicU64,
}

impl Drop for Object<'_> {
    fn drop(&mut self) {
        if self.part.is_none() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Bytes `first..end` of one object. An empty range is the object's commit.
struct Chunk<'a> {
    object: Arc<Object<'a>>,
    first: u64,
    end: u64,
    /// Not asked before this: an origin's stated wait, or one sample after an ask that
    /// moved nothing. `stated` says the wait is the origin's own.
    due: Instant,
    stated: bool,
    /// The origin refused the URL the last ask held, with this status.
    stale: Option<u16>,
    /// This ask's URL was minted afresh for it: the origin's refusal of it is final.
    fresh: bool,
    /// Times this range was dropped. Each doubles what the next ask is given before it is
    /// judged, so an origin that is honestly slow still answers.
    drops: u32,
    /// The object's only chunk so far: its origin has not yet shown that it answers ranges.
    lead: bool,
    /// The chunk this is (a piece of): the range a source download records when it is home.
    unit: (u64, u64),
}

impl<'a> Chunk<'a> {
    /// This chunk's fields for a piece cut from its front.
    fn take(&self) -> Chunk<'a> {
        Chunk {
            object: Arc::clone(&self.object),
            ..*self
        }
    }
}

/// A request on the wire, as the watch sees it.
/// How far one request may go and how far it has come. An idle worker takes the back half of
/// what a running request still has to bring, so the last objects of a download are many
/// requests side by side, not one slow one: no byte is asked twice.
struct Reach {
    end: AtomicU64,
    received: AtomicU64,
}

struct Flight<'a> {
    attempt: Arc<Attempt>,
    subject: String,
    range: (u64, u64),
    reach: Arc<Reach>,
    /// The chunk the request serves, and whether its origin has yet to show it ranges.
    unit: (u64, u64),
    lead: bool,
    began: Instant,
    /// How many fair windows this request gets before it is judged.
    patience: u32,
    /// When its body was last judged, and with how many bytes.
    judged: Option<Instant>,
    seen: u64,
    object: Arc<Object<'a>>,
    /// Stopped for more urgent work: its range goes back whole, and it is no slow request.
    preempted: bool,
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

enum Outcome<'a> {
    Home,
    Again(Vec<Chunk<'a>>),
    /// This object is refused.
    Fatal(Refusal),
    /// The download itself was stopped.
    Stop(Refusal),
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

/// A worker's next step, decided under the run's lock and taken outside it.
enum Next<'a> {
    Take(Option<Chunk<'a>>),
    Open(Option<Opening<'a>>, u64),
    Wait(u64, Duration),
}

struct State<'a> {
    queue: VecDeque<Chunk<'a>>,
    /// Objects a full descriptor table sent back, opened again first.
    again: VecDeque<Opening<'a>>,
    /// Workers holding a chunk or opening an object.
    busy: usize,
    /// The last requests that ended, as many as there are streams, oldest first.
    shown: VecDeque<Peer>,
    /// The last refusal a retry answered, quoted if the pull stops moving.
    last: Option<Refusal>,
    failure: Option<Refusal>,
    done: bool,
}

pub(crate) struct Download<'a> {
    pub store: &'a Store,
    pub feed: &'a dyn Feed<'a>,
    /// One request's bytes, and whether the origin is known to answer ranges.
    pub chunk: u64,
    pub ranged: bool,
    pub deadline: Deadline,
    pub ledger: &'a Ledger,
    /// A source download: objects kept on disk under their owner, resumable.
    pub keep: Option<&'a Keep<'a>>,
}

struct Run<'r, 'a> {
    job: &'r Download<'a>,
    state: Mutex<State<'a>>,
    flights: Vec<Mutex<Option<Flight<'a>>>>,
    fetched: AtomicU64,
    bytes_moved: AtomicU64,
    requests: AtomicU64,
    restarts: AtomicU64,
    opened: AtomicU64,
    opened_bytes: AtomicU64,
    began: Instant,
    /// When each worker began waiting on the hub for a URL, in ms since `began` plus one.
    /// A wait of a sample or more is the hub's to judge, and is not the pull standing still.
    minting: Vec<AtomicU64>,
    /// The latest moment such a wait ended.
    minted: AtomicU64,
    /// Bytes home in temp files and from the cache. A whole answer to a resumed range takes
    /// its object's bytes back out, so asking the same bytes forever is not progress.
    landed: AtomicU64,
    /// Disk work under way: commits, admissions and writes.
    local: AtomicUsize,
    /// Bytes home in every download into this Store (`moved`).
    everywhere: Arc<AtomicU64>,
}

impl<'a> Download<'a> {
    /// Download what the feed hands out with `streams` requests in flight, until it has
    /// nothing more. `observer` hears the bytes home so far, on this thread. The requests are
    /// tasks on the network runtime, their blocking work on two small pools: the threads are
    /// as many with 512 in flight as with one.
    pub fn run(
        &self,
        streams: usize,
        mut observer: Option<&mut dyn FetchObserver>,
    ) -> Result<WalkOutcome> {
        let began = Instant::now();
        // More workers than the machine's budget would only hold objects open while they wait.
        let streams = streams.clamp(1, in_flight().0);
        let run = Run {
            job: self,
            state: Mutex::new(State {
                queue: VecDeque::new(),
                again: VecDeque::new(),
                busy: 0,
                shown: VecDeque::new(),
                last: None,
                failure: None,
                done: false,
            }),
            flights: (0..streams).map(|_| Mutex::new(None)).collect(),
            fetched: AtomicU64::new(0),
            bytes_moved: AtomicU64::new(0),
            requests: AtomicU64::new(0),
            restarts: AtomicU64::new(0),
            opened: AtomicU64::new(0),
            opened_bytes: AtomicU64::new(0),
            began,
            minting: (0..streams).map(|_| AtomicU64::new(0)).collect(),
            minted: AtomicU64::new(0),
            landed: AtomicU64::new(0),
            local: AtomicUsize::new(0),
            everywhere: moved(self.store),
        };
        let writers = std::thread::scope(|scope| {
            let pools = Pools {
                writes: Offload::start(scope, WRITERS, "tensorfs-write"),
                control: Offload::start(scope, CONTROL, "tensorfs-work"),
            };
            let ran = net::block_on(async {
                // A stream that panics ends the walk: its chunk would keep its peers waiting.
                let streams = (0..streams).map(|slot| {
                    AssertUnwindSafe(run.work(&pools, slot)).catch_unwind().map(|worked| {
                        if worked.is_err() {
                            let refusal = Refusal { code: Code::IO_FAILED, detail: "a stream panicked".into() };
                            run.fail(&mut run.state.lock().unwrap_or_else(PoisonError::into_inner), refusal);
                        }
                    })
                });
                futures::future::join(futures::future::join_all(streams), run.watch(&mut observer)).await;
            });
            if let Err(refusal) = ran {
                run.fail(&mut run.state.lock().unwrap(), refusal);
            }
            pools.writes.close();
            pools.control.close();
            pools.writes.peak.load(Ordering::Relaxed)
        });
        if let Some(observer) = observer.as_mut() {
            observer.moved(run.landed.load(Ordering::Relaxed));
        }
        let failure = run.state.lock().unwrap().failure.take();
        let outcome = WalkOutcome {
            fetched: run.fetched.load(Ordering::Relaxed),
            bytes_moved: run.bytes_moved.load(Ordering::Relaxed),
            streams,
            writers,
            requests: run.requests.load(Ordering::Relaxed),
            restarts: run.restarts.load(Ordering::Relaxed),
        };
        decisions::record(
            self.store,
            "walk",
            &format!(
                "{} {}",
                run.opened.load(Ordering::Relaxed),
                run.opened_bytes.load(Ordering::Relaxed)
            ),
            &format!(
                "fetched={} moved={} seconds={:.1} streams={streams} writers={} requests={} \
                 restarts={} outcome={}",
                outcome.fetched,
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

impl<'a> Run<'_, 'a> {
    async fn work<'s>(&'s self, pools: &Pools<'s>, slot: usize)
    where
        'a: 's,
    {
        while let Some(chunk) = self.take(pools).await {
            let object = Arc::clone(&chunk.object);
            let outcome = match chunk.first == chunk.end {
                true => {
                    let committing = Arc::clone(&object);
                    match self.locally(pools.control.run(move || self.commit(&committing))).await {
                        // A full descriptor table is backpressure: the commit waits a sample.
                        Outcome::Fatal(refusal) if refusal.code == Code::FD_HEADROOM => {
                            net::shed();
                            let mut chunk = chunk;
                            chunk.due = Instant::now() + self.job.ledger.sample();
                            self.state.lock().unwrap().last = Some(refusal);
                            Outcome::Again(vec![chunk])
                        }
                        outcome => outcome,
                    }
                }
                false => self.request(pools, slot, chunk).await,
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
                Outcome::Fatal(refusal) => self.refused(&mut state, &object, refusal),
                Outcome::Stop(refusal) => self.fail(&mut state, refusal),
            }
            drop(state);
            self.job.feed.pulse().beat();
        }
    }

    /// One object is refused: a lasting feed loses that object alone, a batch ends.
    fn refused(&self, state: &mut State<'a>, object: &Object<'a>, refusal: Refusal) {
        if !self.job.feed.lasting() {
            return self.fail(state, refusal);
        }
        object.cell.leave();
        if !object.told.swap(true, Ordering::AcqRel) {
            self.job.feed.ended(&object.grant.object, &object.cell, Err(refusal));
        }
    }

    /// The next chunk that is due, most urgent first. An object waiting to be opened that is
    /// more urgent than every due chunk is opened first and its chunks queued; with nothing
    /// to take or open, the worker waits for one to come due, back, or fed.
    async fn take<'s>(&'s self, pools: &Pools<'s>) -> Option<Chunk<'a>>
    where
        'a: 's,
    {
        let feed = self.job.feed;
        loop {
            match self.next() {
                Next::Take(chunk) => return chunk,
                Next::Wait(seen, wait) => feed.pulse().changed(seen, wait).await,
                Next::Open(again, before) => {
                    let opened = self.locally(pools.control.run(move || {
                            let opening = match again {
                                Some(again) => Ok(Some(again)),
                                None => feed.open(before),
                            };
                            opening.map(|opening| opening.map(|opening| {
                                let chunks = self.open(&opening);
                                (opening, chunks)
                            }))
                        }))
                        .await;
                    if let Some(backoff) = self.opened(opened) {
                        feed.pulse().changed(feed.pulse().seen(), backoff).await;
                    }
                }
            }
        }
    }

    /// Bytes home (negative: moved again), in this run and in its Store's count.
    fn home(&self, bytes: i64) {
        self.landed.fetch_add(bytes as u64, Ordering::Relaxed);
        self.everywhere.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    /// `work` awaited as this run's own disk work: the watch does not count it as stillness.
    async fn locally<T>(&self, work: impl std::future::Future<Output = T>) -> T {
        self.local.fetch_add(1, Ordering::AcqRel);
        let done = work.await;
        self.local.fetch_sub(1, Ordering::AcqRel);
        done
    }

    /// What a worker does next, decided under the run's lock: take a due chunk, open the
    /// next object, or wait for a beat.
    fn next(&self) -> Next<'a> {
        let feed = self.job.feed;
        // Read before looking: whatever changes after this beats, so no wake is missed.
        let seen = feed.pulse().seen();
        let mut state = self.state.lock().unwrap();
        if state.failure.is_some() || state.done {
            return Next::Take(None);
        }
        state.queue.retain(|chunk| !chunk.object.cell.gone());
        let now = Instant::now();
        let due = (state.queue.iter().enumerate())
            .filter(|(_, chunk)| chunk.due <= now)
            .min_by_key(|(at, chunk)| (chunk.object.cell.key(), *at))
            .map(|(at, chunk)| (at, chunk.object.cell.key()));
        let before = due.map_or(u64::MAX, |(_, key)| key);
        let again = state.again.front().map(|opening| opening.cell.key());
        let waiting = again.or_else(|| feed.waiting()).filter(|key| *key < before);
        // Behind work waits while anything more urgent is out.
        let next = waiting.or(due.map(|(_, key)| key));
        let held = next.is_some_and(|key| key >> 62 == BEHIND >> 62) && self.urgent(&state);
        if let (Some((at, _)), None, false) = (due, waiting, held) {
            state.busy += 1;
            return Next::Take(state.queue.remove(at));
        }
        if waiting.is_some() && !held {
            state.busy += 1;
            return Next::Open(state.again.pop_front(), before);
        }
        if let (None, false, Some(stolen)) = (due, held, self.steal()) {
            state.busy += 1;
            return Next::Take(Some(stolen));
        }
        if state.queue.is_empty() && state.busy == 0 && feed.waiting().is_none() {
            state.done = true;
            drop(state);
            feed.pulse().beat();
            return Next::Take(None);
        }
        let wait = (state.queue.iter())
            .map(|chunk| chunk.due.saturating_duration_since(now))
            .min()
            .map_or(self.job.ledger.sample(), |due| due.min(self.job.ledger.sample()))
            .max(Duration::from_millis(1));
        Next::Wait(seen, wait)
    }

    /// With nothing else to do, the back half of the running request with the most still to
    /// come: two requests then bring it side by side. A request with under two pieces left, an
    /// origin not yet known to range, and an object nothing wants are left alone.
    fn steal(&self) -> Option<Chunk<'a>> {
        let (mut best, mut most) = (None, 2 * PIECE_BYTES);
        for (slot, flight) in self.flights.iter().enumerate() {
            let flight = flight.lock().unwrap();
            let Some(flight) = flight.as_ref().filter(|f| !f.lead && !f.preempted && !f.object.cell.gone()) else {
                continue;
            };
            let left = (flight.reach.end.load(Ordering::Acquire))
                .saturating_sub(flight.reach.received.load(Ordering::Acquire));
            if left >= most {
                (best, most) = (Some(slot), left);
            }
        }
        let flight = self.flights[best?].lock().unwrap();
        let flight = flight.as_ref()?;
        let end = flight.reach.end.load(Ordering::Acquire);
        let received = flight.reach.received.load(Ordering::Acquire);
        let split = (received + (end - received) / 2).next_multiple_of(64 << 10).min(end);
        if split <= received || split >= end {
            return None;
        }
        (flight.reach.end).compare_exchange(end, split, Ordering::AcqRel, Ordering::Acquire).ok()?;
        let object = Arc::clone(&flight.object);
        object.left.fetch_add(1, Ordering::AcqRel);
        if let Some((_, out)) = object.units.lock().unwrap().get_mut(&flight.unit.0) {
            *out += 1;
        }
        Some(Chunk {
            object,
            first: split,
            end,
            due: Instant::now(),
            stated: false,
            stale: None,
            fresh: false,
            drops: 0,
            lead: false,
            unit: flight.unit,
        })
    }

    /// What opening an object gave, filed: its chunks queued, or its refusal told. A full
    /// descriptor table is backpressure: the object is opened again after the backoff this
    /// answers.
    #[allow(clippy::type_complexity)]
    fn opened(&self, opened: Result<Option<(Opening<'a>, Result<Vec<Chunk<'a>>>)>>) -> Option<Duration> {
        let feed = self.job.feed;
        let mut state = self.state.lock().unwrap();
        state.busy -= 1;
        match opened {
            Ok(None) => {}
            Ok(Some((_, Ok(chunks)))) => state.queue.extend(chunks),
            Ok(Some((opening, Err(refusal)))) if refusal.code == Code::FD_HEADROOM => {
                net::shed();
                state.again.push_front(opening);
                state.last = Some(refusal);
                return Some(self.job.ledger.sample());
            }
            Ok(Some((opening, Err(refusal)))) => {
                let refusal = named(refusal, &opening.grant.object.id());
                if feed.lasting() {
                    opening.cell.leave();
                    drop(state);
                    feed.ended(&opening.grant.object, &opening.cell, Err(refusal));
                    feed.pulse().beat();
                    return None;
                }
                self.fail(&mut state, refusal);
            }
            Err(refusal) => self.fail(&mut state, refusal),
        }
        drop(state);
        feed.pulse().beat();
        None
    }

    fn fail(&self, state: &mut State, refusal: Refusal) {
        if state.failure.is_none() {
            // Keep the first failure before waking peers, so their cancellation errors
            // cannot replace its diagnosis.
            state.failure = Some(refusal);
            self.job.ledger.cancel();
            self.job.feed.pulse().beat();
        }
    }

    /// One object off the feed, given a temp file (a source download's kept file) and cut into
    /// chunks.
    fn open(&self, opening: &Opening<'a>) -> Result<Vec<Chunk<'a>>> {
        let job = self.job;
        let wanted = &opening.grant.object;
        let grant = opening.grant.clone();
        self.opened.fetch_add(1, Ordering::Relaxed);
        self.opened_bytes.fetch_add(wanted.length, Ordering::Relaxed);
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
            at: opening.at,
            grant,
            supply: Arc::clone(&opening.supply),
            cell: Arc::clone(&opening.cell),
            told: AtomicBool::new(false),
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
            remint: AtomicBool::new(false),
            moved: AtomicU64::new(0),
        });
        let chunk = |(first, end)| Chunk {
            object: Arc::clone(&object),
            first,
            end,
            due: Instant::now(),
            stated: false,
            stale: None,
            fresh: false,
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
    fn rest(&self, lead: &Chunk<'a>) {
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
                fresh: false,
                drops: 0,
                lead: false,
                unit: (first, end),
            });
        }
        drop(state);
        self.job.feed.pulse().beat();
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

    /// The download's own stop, or its deadline, as a refusal.
    fn stopped(&self) -> Option<Refusal> {
        let job = self.job;
        (job.ledger.check_running())
            .and_then(|()| job.deadline.remaining().map(|_| ()))
            .err()
    }

    /// One GET for a chunk, written at its offsets. Whatever ends it early, the bytes that
    /// landed stay and the rest goes back on the queue.
    async fn request<'s>(&'s self, pools: &Pools<'s>, slot: usize, mut chunk: Chunk<'a>) -> Outcome<'a>
    where
        'a: 's,
    {
        let job = self.job;
        let object = Arc::clone(&chunk.object);
        let wanted = &object.grant.object;
        // The URL first: a wait on the hub is the hub's to judge, not this request's, and a
        // wait for a free control thread is neither.
        let stale = chunk.stale.take();
        let delivery = object.delivery.lock().unwrap().clone();
        chunk.fresh = false;
        let url = match delivery.clone() {
            Some(delivery) => Ok(delivery),
            None => {
                let minting = Arc::clone(&object);
                let stale = stale.is_some() || object.remint.swap(false, Ordering::AcqRel);
                chunk.fresh = stale;
                pools
                    .control
                    .run(move || {
                        let asked = self.began.elapsed();
                        self.minting[slot].store(asked.as_millis() as u64 + 1, Ordering::Release);
                        let url = minting.supply.url(minting.at, &minting.grant.object, stale, job.ledger);
                        self.minting[slot].store(0, Ordering::Release);
                        let now = self.began.elapsed();
                        if now - asked >= job.ledger.sample() {
                            self.minted.fetch_max(now.as_millis() as u64, Ordering::AcqRel);
                        }
                        url
                    })
                    .await
            }
        };
        let url = match (url, stale) {
            (Ok(url), _) => Ok(url),
            (Err(_), _) if self.stopped().is_some() => {
                return Outcome::Stop(self.stopped().unwrap())
            }
            // A refused re-authorization still names the origin's answer.
            (Err(refusal), Some(status)) => Err(Refusal {
                code: Code::TRANSFER_FAILED,
                detail: format!(
                    "{}: the object store answered {status} and re-authorizing was refused — {}",
                    wanted.id(),
                    refusal.detail
                ),
            }),
            (Err(refusal), None) => Err(refusal),
        };
        let url = match url {
            Ok(url) => url,
            // The link's or the hub's weather is asked again a sample on.
            Err(refusal) if job.feed.lasting() && matches!(refusal.code, Code::TRANSFER_FAILED | Code::HUB_UNREACHABLE) => {
                chunk.due = Instant::now() + job.ledger.sample();
                self.state.lock().unwrap().last = Some(named(refusal, &wanted.id()));
                return Outcome::Again(vec![chunk]);
            }
            Err(refusal) => return Outcome::Fatal(refusal),
        };
        // One of the machine's requests in flight, held to this request's end.
        let Ok(_flying) = in_flight().1.acquire().await else {
            return Outcome::Stop(Refusal { code: Code::TRANSFER_FAILED, detail: "the request budget closed".into() });
        };
        let attempt = Arc::new(Attempt::default());
        let ledger = job.ledger.for_request(Arc::clone(&attempt), self.tick());
        let began = Instant::now();
        let asked = (chunk.first, chunk.end);
        let reach = Arc::new(Reach { end: AtomicU64::new(chunk.end), received: AtomicU64::new(chunk.first) });
        *self.flights[slot].lock().unwrap() = Some(Flight {
            attempt: Arc::clone(&attempt),
            subject: decisions::object(wanted),
            range: asked,
            reach: Arc::clone(&reach),
            unit: chunk.unit,
            lead: chunk.lead,
            began,
            patience: 1 << chunk.drops.min(6),
            judged: None,
            seen: 0,
            object: Arc::clone(&object),
            preempted: false,
        });
        self.requests.fetch_add(1, Ordering::Relaxed);
        let ended = self.stream(pools, &mut chunk, &url, delivery.is_some(), &ledger, &reach).await;
        let preempted = (self.flights[slot].lock().unwrap().take()).is_some_and(|f| f.preempted);
        self.show(&attempt, began, asked);
        // Nothing wants it any more: what landed goes with its temp file.
        if object.cell.gone() {
            return Outcome::Home;
        }
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
            // The first failure is the diagnosis; a stopped pull's own refusal is not.
            Err(Ended::Fatal(refusal)) => {
                return match self.stopped() {
                    Some(stopped) => Outcome::Stop(stopped),
                    None => Outcome::Fatal(named(refusal, &wanted.id())),
                }
            }
        };
        // The pull itself was stopped, or its deadline passed: nothing is asked again.
        if let Some(stopped) = self.stopped() {
            return Outcome::Stop(stopped);
        }
        // Stopped for more urgent work: the rest goes back whole, due at once.
        if preempted {
            return Outcome::Again(vec![chunk]);
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
    async fn stream<'s>(
        &'s self,
        pools: &Pools<'s>,
        chunk: &mut Chunk<'a>,
        url: &str,
        delivered: bool,
        ledger: &Ledger,
        reach: &Reach,
    ) -> std::result::Result<(), Ended>
    where
        'a: 's,
    {
        let job = self.job;
        let object = Arc::clone(&chunk.object);
        let wanted = &object.grant.object;
        let lead = chunk.lead;
        // A Hub object that is one chunk is one plain GET. A source request always names its
        // range: a 206 is how the origin says it ranges, and its stand-ins answer only ranges.
        let whole = chunk.first == 0 && chunk.end == wanted.length;
        let range = [(
            "range".to_string(),
            format!("bytes={}-{}", chunk.first, chunk.end - 1),
        )];
        let mut hops = Vec::new();
        let asked: &[(String, String)] = if whole && job.ranged { &[] } else { &range };
        let opened = open_object(pools, url, &object, ledger, asked, &mut hops).await?;
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
            // Refused: the next ask holds a URL minted afresh, and a refusal of that is the
            // origin's verdict.
            Opened::Expired(status) => {
                let answered = Refusal {
                    code: Code::TRANSFER_FAILED,
                    detail: format!("the object store answered {status}"),
                };
                if chunk.fresh {
                    return Err(Ended::Fatal(Refusal {
                        detail: format!("{} to a URL minted afresh", answered.detail),
                        ..answered
                    }));
                }
                chunk.stale = Some(status);
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
                self.home(-(again as i64));
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
        let mut body = response.body;
        // A lead chunk the origin answered whole now reaches the object's end; no other range
        // is reset, as an idle worker may already have taken its back (`Run::steal`).
        if lead {
            reach.end.store(chunk.end, Ordering::Release);
        }
        let asked_end = chunk.end;
        // Frames gather into one write of up to `WRITE_BYTES`, handed to the write pool.
        let mut gathered: Vec<u8> = Vec::new();
        while chunk.first < reach.end.load(Ordering::Acquire) {
            // A body that fails still keeps what it gave: the next ask is for the rest.
            let (data, failed) = match body.chunk().await {
                Ok(data) => (data, None),
                Err(refusal) => (None, Some(refusal)),
            };
            let ended = data.is_none();
            if let Some(data) = &data {
                let left = reach.end.load(Ordering::Acquire).saturating_sub(chunk.first) as usize;
                let taken = data.len().min(left.saturating_sub(gathered.len()));
                gathered.extend_from_slice(&data[..taken]);
                reach.received.store(chunk.first + gathered.len() as u64, Ordering::Release);
                // A steal that raced this frame: the bytes past its new end are the thief's.
                let end = reach.end.load(Ordering::Acquire);
                gathered.truncate(end.saturating_sub(chunk.first) as usize);
                self.home(taken as i64);
            }
            chunk.end = reach.end.load(Ordering::Acquire);
            let full = chunk.first + gathered.len() as u64 >= chunk.end;
            if !gathered.is_empty() && (gathered.len() >= WRITE_BYTES || full || ended) {
                let bytes = std::mem::take(&mut gathered);
                let (file, offset, length) = (Arc::clone(&object), chunk.first, bytes.len() as u64);
                let written = self.locally(pools.writes.run(move || file.file.write_all_at(&bytes, offset))).await;
                written.map_err(|error| {
                    Ended::Fatal(crate::store::classify_io(format!("write {}", object.path.display()), error))
                })?;
                self.wrote(chunk, &object, length);
            }
            if let Some(refusal) = failed {
                return Err(Ended::from(refusal));
            }
            if ended {
                break;
            }
        }
        chunk.end = reach.end.load(Ordering::Acquire);
        if chunk.first < chunk.end {
            return Err(Ended::Retry(
                Refusal {
                    code: Code::TRANSFER_FAILED,
                    detail: format!("the origin closed its answer {} B short", chunk.end - chunk.first),
                },
                None,
            ));
        }
        // Its end read, the connection serves the next ask; a range cut short leaves it.
        if chunk.end == asked_end {
            let _ = body.chunk().await;
        }
        Ok(())
    }

    /// `bytes` more of `chunk` are written: the chunk advances.
    fn wrote(&self, chunk: &mut Chunk<'a>, object: &Object<'a>, bytes: u64) {
        chunk.first += bytes;
        object.moved.fetch_add(bytes, Ordering::Relaxed);
        self.job.store.moved(bytes);
    }

    /// Every chunk is home: hash the file and commit it at the digest it hashed to.
    fn commit(&self, object: &Object<'a>) -> Outcome<'a> {
        let job = self.job;
        let wanted = &object.grant.object;
        let moved = object.moved.load(Ordering::Relaxed);
        if object.cell.gone() {
            return Outcome::Home;
        }
        // A collection the feed runs excludes admissions while it plans and deletes.
        let _gate = job.feed.gate().map(|gate| gate.read().unwrap());
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
            if refusal.code == Code::FD_HEADROOM {
                return Outcome::Fatal(refusal);
            }
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
        if !object.told.swap(true, Ordering::AcqRel) {
            job.feed.ended(wanted, &object.cell, Ok((ObjectSource::Origin, moved)));
        }
        Outcome::Home
    }

    /// Whether a request or a due chunk of a class more urgent than Behind is out: Behind work
    /// runs only while there is none.
    fn urgent(&self, state: &State<'a>) -> bool {
        let behind = BEHIND >> 62;
        state.queue.iter().any(|chunk| chunk.object.cell.class() < behind)
            || self.job.feed.waiting().is_some_and(|key| key >> 62 < behind)
            || (self.flights.iter()).any(|flight| {
                (flight.lock().unwrap().as_ref()).is_some_and(|f| f.object.cell.class() < behind)
            })
    }

    /// Requests of a lower class give way to waiting work of a higher one, one for each
    /// waiting chunk no free worker will take, least urgent first; Behind requests stop
    /// whenever more urgent work is out. A request whose object nothing wants stops too.
    fn preempt(&self, state: &State<'a>) {
        let now = Instant::now();
        let due = || state.queue.iter().filter(|chunk| chunk.due <= now);
        let fed = self.job.feed.waiting().map(|key| key >> 62);
        let again = state.again.iter().map(|opening| opening.cell.key() >> 62);
        let waiting = due().map(|chunk| chunk.object.cell.class()).chain(fed).chain(again).min();
        let urgent = self.urgent(state);
        let mut need = waiting.map_or(0, |class| {
            due().filter(|chunk| chunk.object.cell.class() == class).count()
                + usize::from(fed == Some(class))
                + state.again.len()
        });
        need = need.saturating_sub(self.flights.len().saturating_sub(state.busy));
        let mut victims = Vec::new();
        for (slot, flight) in self.flights.iter().enumerate() {
            let mut flight = flight.lock().unwrap();
            let Some(flight) = flight.as_mut() else {
                continue;
            };
            let class = flight.object.cell.class();
            if flight.object.cell.gone() || (urgent && class == BEHIND >> 62) {
                flight.preempted = true;
                flight.attempt.stopped.store(true, Ordering::Release);
            } else if waiting.is_some_and(|waiting| class > waiting) {
                match flight.preempted {
                    true => need = need.saturating_sub(1),
                    false => victims.push((class, slot)),
                }
            }
        }
        victims.sort_by(|a, b| b.cmp(a));
        for (_, slot) in victims.into_iter().take(need) {
            if let Some(flight) = self.flights[slot].lock().unwrap().as_mut() {
                flight.preempted = true;
                flight.attempt.stopped.store(true, Ordering::Release);
            }
        }
    }

    /// The one rule, five times a sample: a request that has waited far longer for its first
    /// byte than the requests beside it, or whose body runs far slower than theirs over the
    /// same time, is dropped. Peers are the requests still running and those that ended
    /// since its window began, never history, so a pod whose every request is slow drops
    /// none. And the pull's own end: nothing new has landed for twice its measured patience.
    /// Between those, every beat gives waiting work its way ([`Run::preempt`]).
    async fn watch(&self, observer: &mut Option<&mut dyn FetchObserver>) {
        let job = self.job;
        let pulse = job.feed.pulse();
        let sample = self.tick();
        // The most bytes ever home, and when that last grew while requests were being asked;
        // what the process and this run then had home, and whether this run has since asked
        // again with fresh URLs.
        let (mut most, mut grew) = (0, Instant::now());
        let (mut since, mut reasked) = ((self.everywhere.load(Ordering::Relaxed), 0), false);
        let mut judge = Instant::now() + sample;
        loop {
            let seen = pulse.seen();
            {
                let state = self.state.lock().unwrap();
                if state.done || state.failure.is_some() {
                    return;
                }
                self.preempt(&state);
            }
            let left = judge.saturating_duration_since(Instant::now());
            if !left.is_zero() {
                pulse.changed(seen, left).await;
                continue;
            }
            judge = Instant::now() + sample;
            if let Some(observer) = observer.as_mut() {
                // Told outside the lock: an observer may be slow, and workers must not wait.
                observer.moved(self.landed.load(Ordering::Relaxed));
            }
            let mut state = self.state.lock().unwrap();
            if state.done || state.failure.is_some() {
                return;
            }
            if let Some(refusal) = self.stopped() {
                self.fail(&mut state, refusal);
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
                if flight.preempted {
                    continue;
                }
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
                reasked &= landed <= most;
                (most, grew, since) = (most.max(landed), now, (self.everywhere.load(Ordering::Relaxed), landed));
                continue;
            }
            // This run's own disk work (a commit's hash and fsync, an admission, a write) is
            // not the origin standing still: the verdict waits for it, and still counts.
            if self.local.load(Ordering::Acquire) > 0 {
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
                if !job.feed.lasting() {
                    self.fail(&mut state, refusal);
                    return;
                }
                let elsewhere = self.everywhere.load(Ordering::Relaxed).wrapping_sub(since.0) as i64
                    > landed.wrapping_sub(since.1) as i64;
                let verdict = if elsewhere || !reasked { "reask" } else { "stalled" };
                decisions::record(
                    job.store,
                    verdict,
                    "run",
                    &format!(
                        "still_s={:.1} patience_s={:.1} home={most} elsewhere={elsewhere}",
                        grew.elapsed().as_secs_f64(),
                        patience.as_secs_f64(),
                    ),
                );
                if elsewhere || !reasked {
                    // Asked again for what it was asking for, each URL minted afresh, for as
                    // long as anything else moves, and once when nothing does.
                    state.last = Some(refusal);
                    let now = Instant::now();
                    for chunk in state.queue.iter_mut().filter(|chunk| chunk.first < chunk.end) {
                        (chunk.due, chunk.stated) = (now, false);
                        chunk.object.remint.store(true, Ordering::Release);
                    }
                    for flight in &self.flights {
                        if let Some(flight) = flight.lock().unwrap().as_mut() {
                            flight.attempt.stopped.store(true, Ordering::Release);
                            *flight.object.delivery.lock().unwrap() = None;
                            flight.object.remint.store(true, Ordering::Release);
                        }
                    }
                    job.ledger.condemned();
                    reasked = true;
                } else {
                    // Nothing in the process moves, even on fresh URLs: what this run was
                    // asking for is refused, and the download goes on.
                    let refusal = Refusal { code: Code::STALLED, ..refusal };
                    let mut asked: Vec<Arc<Object<'a>>> =
                        (state.queue.iter()).map(|chunk| Arc::clone(&chunk.object)).collect();
                    for flight in &self.flights {
                        if let Some(flight) = flight.lock().unwrap().as_ref() {
                            asked.push(Arc::clone(&flight.object));
                        }
                    }
                    for object in asked {
                        self.refused(&mut state, &object, named(refusal.clone(), &object.grant.object.id()));
                    }
                    reasked = false;
                }
                drop(state);
                pulse.beat();
                (most, grew, since) = (landed, Instant::now(), (self.everywhere.load(Ordering::Relaxed), landed));
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

enum Opened<'l> {
    Body(net::Response<'l>),
    /// A transient answer, and the wait the origin stated for it, if any.
    Retryable(u16, Option<Duration>),
    /// The origin refused the URL (401, 403, 404): asked again with one minted afresh.
    Expired(u16),
    Refused(Refusal),
}

/// One GET of `object`, redirect hops walked BY HAND so every hop faces the same predicate
/// and the credential is re-asked per host (a scoped credential cannot leak onto a CDN).
/// `extra` (a range) follows the credential. Each hop's URL joins `hops`.
async fn open_object<'l, 's, 'a: 's>(
    pools: &Pools<'s>,
    url: &str,
    object: &Arc<Object<'a>>,
    ledger: &'l Ledger,
    extra: &[(String, String)],
    hops: &mut Vec<String>,
) -> Result<Opened<'l>> {
    let mut target = url.to_string();
    let redirects = object.supply.policy().max_redirects;
    for _hop in 0..=redirects {
        let (supply, asked) = (Arc::clone(&object.supply), target.clone());
        let checked = Arc::new(pools.control.run(move || supply.policy().check(&asked)).await?);
        let mut retried = false;
        let response = loop {
            let (supply, url) = (Arc::clone(&object.supply), Arc::clone(&checked));
            let presented = pools
                .control
                .run(move || supply.credential().headers(&Ask { method: "GET", url: &url }))
                .await;
            let headers: Vec<(String, String)> = presented.iter().chain(extra).cloned().collect();
            let response = net::get(shared_client(), &checked, &headers, ledger, Judge::Stream, Code::TRANSFER_FAILED).await?;
            let again = !retried && {
                let (supply, url, status) = (Arc::clone(&object.supply), Arc::clone(&checked), response.status);
                let answered = response.headers().to_vec();
                pools
                    .control
                    .run(move || {
                        let answer = Answer::of(status, &answered, &presented);
                        supply.credential().answered(&Ask { method: "GET", url: &url }, &answer)
                    })
                    .await
            };
            if !again {
                break response;
            }
            retried = true;
            response.discard().await;
        };
        let status = response.status;
        hops.push(checked.url());
        if (300..400).contains(&status) {
            let Some(location) = response.header("location").map(str::to_string) else {
                return refuse(Code::REDIRECT_REFUSED, "origin redirected with no Location");
            };
            let Some(absolute) = absolute_location(&checked, &location) else {
                return refuse(Code::REDIRECT_REFUSED, "origin redirected with an unusable Location");
            };
            if redirects == 0 {
                return refuse(
                    Code::REDIRECT_REFUSED,
                    "a presigned GET has no business redirecting, and following one would open \
                     a URL the predicate never saw",
                );
            }
            target = absolute;
            response.discard().await;
            continue;
        }
        if status == 200 || status == 206 {
            return Ok(Opened::Body(response));
        }
        if RETRYABLE_STATUS.contains(&status) {
            let wait = response.stated_wait();
            response.discard().await;
            return Ok(Opened::Retryable(status, wait));
        }
        if matches!(status, 401 | 403 | 404) {
            return Ok(Opened::Expired(status));
        }
        return Ok(Opened::Refused(Refusal {
            code: Code::TRANSFER_FAILED,
            detail: format!("the object store answered {status}"),
        }));
    }
    refuse(
        Code::REDIRECT_REFUSED,
        format!("more than {redirects} redirect hop(s); a longer chain is a loop or an evasion"),
    )
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
    let wanted = std::slice::from_ref(&grant.object);
    let granted = |_: &ObjectRef| Ok(grant.clone());
    let batch = Batch::new(wanted, &granted, &urls, policy, credential, None);
    let chunk = ranged.part.clamp(1 << 20, 1 << 30);
    let outcome = Download {
        store,
        feed: &batch,
        chunk,
        ranged: false,
        deadline,
        ledger,
        keep: None,
    }
    .run(batch.streams(ranged.streams.clamp(1, 64), chunk), observer)?;
    Ok(Fetched {
        object: grant.object.clone(),
        transferred: outcome.bytes_moved,
    })
}

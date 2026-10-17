//! The downloader, the way an S3 client does it: one queue of whole-object GETs (objects are
//! chunks of at most 64 MiB), a paced number of workers, and any request that fails, stalls
//! or runs far slower than its peers restarted on a fresh connection from the byte it reached.
//!
//! - **Order.** The queue is a plain number, smaller first; equal numbers go in arrival order.
//!   An object wanted twice is downloaded once.
//! - **Tiers.** The local store, then the volume (when one is bound), then the origin. Every
//!   byte enters the store's admission, which hashes it: wrong bytes are never admitted.
//! - **Judging a request** is relative and measured, never a timeout. Each request measures
//!   the rate of every stretch of body it receives between two writes into the store; every
//!   tick the watch drops one with no answer after 8× the fetch's median time to answer, or
//!   whose body moved under an eighth of the median stretch rate over its window, and it is
//!   asked again from the byte it reached. A request that was dropped gets twice the window
//!   next time, so an origin that is honestly slow still answers, and a fetch whose every
//!   request is slow restarts none. A cold R2 object stalls at a MiB boundary for whoever asks
//!   first while a fresh ranged ask is answered at once (pod measurements, 2026-10-01).
//! - **Ending.** An object ends on an answer asking again cannot change (absent, refused after
//!   a fresh location, the same wrong bytes twice), or when nothing in the whole fetch has
//!   landed for twice its measured patience; the refusal quotes the last answer.
//! - **Pace.** Workers start at 16 and double each window while the queue waits and goodput
//!   grows by 1.2×, up to what the descriptor table holds (tfs-122); a full table halves them.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::future::Future;
use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use tokio::sync::{watch, Notify, Semaphore};

use super::net::Client;
use super::origin::{Location, Origin};
use super::volume::{blocking, VolumeOrigin, Wrote, STEP};
use super::{Admission, ObjectStore, Tier};
use crate::err::{refuse, Code, Refusal, Result};
use crate::format::ObjectRef;

/// How many times slower than its peers a request may run, or longer than their answer it may
/// wait, before it is dropped. Also how much longer than the fetch's longest recovered stillness
/// it may stand still before an object ends.
const STALL_FACTOR: f64 = 8.0;
/// The noise floor, in ticks, below which a wait for an answer or a silence is no evidence.
const HEAD_TICKS: u32 = 8;
/// Ticks over which a body's rate is judged.
const JUDGE_TICKS: usize = 4;
/// The floor, in ticks, of the stillness an object's retries are allowed.
const STILL_TICKS: u32 = 120;
/// Rate samples needed before any request is judged slow.
const MIN_SAMPLES: usize = 8;
/// The fewest bytes a stretch of body needs to be a rate sample.
const STRETCH_BYTES: u64 = 64 << 10;
/// Rate and answer-time samples kept.
const SAMPLES: usize = 256;
const MAX_DOUBLINGS: u32 = 6;
/// Bytes a request gathers before handing a write to the store.
const WRITE_BYTES: usize = 1 << 20;
/// Writes into the store at once: more add threads stalled in the kernel, not bytes.
const WRITERS: usize = 8;
const VOLUME_READS: usize = 8;
const WRITE_BACKS: usize = 2;
/// Ticks per pace window, and the gain that earns a doubling.
const PACE_TICKS: u32 = 8;
const PACE_GAIN: f64 = 1.2;

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One object being fetched, and everyone waiting for it.
pub(crate) struct Task {
    pub object: ObjectRef,
    origin: Arc<dyn Origin>,
    priority: AtomicU64,
    waiters: AtomicUsize,
    started: AtomicBool,
    /// When its transfer began, in nanoseconds since the fetcher's epoch.
    since: AtomicU64,
    /// The most bytes any of its requests reached: only bytes past it are progress.
    reached: AtomicU64,
    done: watch::Sender<Option<std::result::Result<Tier, Refusal>>>,
}

impl Task {
    fn wanted(&self) -> bool {
        self.waiters.load(Ordering::Acquire) > 0
    }
}

/// A place in a task's line of waiters; dropping it gives the place up.
pub(crate) struct Waiter {
    task: Arc<Task>,
    answer: watch::Receiver<Option<std::result::Result<Tier, Refusal>>>,
}

impl Waiter {
    pub async fn wait(mut self) -> Result<Tier> {
        let answer = self.answer.wait_for(Option::is_some).await;
        match answer.as_deref() {
            Ok(Some(answer)) => answer.clone(),
            _ => refuse(
                Code::TRANSFER_FAILED,
                format!("{}: the fetcher closed", self.task.object.sha256),
            ),
        }
    }
}

impl Drop for Waiter {
    fn drop(&mut self) {
        self.task.waiters.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Default)]
struct Queue {
    tasks: HashMap<String, Arc<Task>>,
    heap: BinaryHeap<Reverse<(u64, u64, String)>>,
    seq: u64,
}

/// Why the watch dropped a request.
#[derive(Clone)]
enum Dropped {
    Judged(String),
    Unwanted,
}

/// A request in flight, as the watch sees it.
struct Flight {
    task: Arc<Task>,
    file: bool,
    began: Instant,
    doublings: u32,
    answered: AtomicBool,
    /// Bytes received, and when the request last moved (nanoseconds since `began`).
    bytes: AtomicU64,
    moved: AtomicU64,
    /// Waiting on the store, not the network: not judged meanwhile.
    local: AtomicBool,
    /// Where the current stretch of body began: bytes, and nanoseconds since `began`.
    stretch: Mutex<(u64, u64)>,
    /// Bytes at each recent tick, newest last.
    window: Mutex<VecDeque<u64>>,
    dropped: Mutex<Option<Dropped>>,
    abort: Notify,
}

impl Flight {
    fn moved(&self) {
        self.moved
            .store(self.began.elapsed().as_nanos() as u64, Ordering::Release);
    }

    /// The stretch of body since the last write ends: its rate is a sample.
    fn stretch_ends(&self, measured: &Mutex<Measured>) {
        let (bytes, now) = (
            self.bytes.load(Ordering::Acquire),
            self.began.elapsed().as_nanos() as u64,
        );
        let (from, since) = *lock(&self.stretch);
        if bytes >= from + STRETCH_BYTES && now > since {
            push(
                &mut lock(measured).rates,
                (bytes - from) as f64 / Duration::from_nanos(now - since).as_secs_f64(),
            );
        }
    }

    fn stretch_begins(&self) {
        *lock(&self.stretch) = (
            self.bytes.load(Ordering::Acquire),
            self.began.elapsed().as_nanos() as u64,
        );
    }

    fn silent(&self) -> Duration {
        self.began
            .elapsed()
            .saturating_sub(Duration::from_nanos(self.moved.load(Ordering::Acquire)))
    }

    fn drop_with(&self, why: Dropped) {
        let mut dropped = lock(&self.dropped);
        if dropped.is_none() {
            *dropped = Some(why);
            self.abort.notify_one();
        }
    }

    /// Resolves once the watch drops this request.
    fn aborted(&self) -> impl Future<Output = ()> + '_ {
        self.abort.notified()
    }

    fn why(&self) -> Dropped {
        lock(&self.dropped).clone().unwrap_or(Dropped::Unwanted)
    }
}

/// A flight registered with the watch for as long as this lives.
struct Flying<'s> {
    shared: &'s Shared,
    flight: Arc<Flight>,
}

impl Drop for Flying<'_> {
    fn drop(&mut self) {
        lock(&self.shared.flights).retain(|f| !Arc::ptr_eq(f, &self.flight));
    }
}

#[derive(Default)]
struct Measured {
    /// Seconds from asking to the answer's head.
    heads: VecDeque<f64>,
    /// Bytes per second of one stretch of body.
    rates: VecDeque<f64>,
    /// The longest one volume step took.
    file_step: Duration,
}

fn push(samples: &mut VecDeque<f64>, value: f64) {
    if samples.len() >= SAMPLES {
        samples.pop_front();
    }
    samples.push_back(value);
}

fn median(samples: &VecDeque<f64>) -> Option<f64> {
    let mut sorted: Vec<f64> = samples.iter().copied().collect();
    sorted.sort_by(f64::total_cmp);
    sorted.get(sorted.len() / 2).copied()
}

struct Pace {
    level: usize,
    max: usize,
    /// Goodput at the last level, while growing.
    last: Option<f64>,
    settled: bool,
    /// Permits to retire as they come back.
    debt: usize,
}

/// What the Fetcher shares with its dispatcher, watch and transfers.
pub(crate) struct Shared {
    pub store: Arc<dyn ObjectStore>,
    pub client: Client,
    pub volume: Option<VolumeOrigin>,
    pub tick: Duration,
    epoch: Instant,
    queue: Mutex<Queue>,
    queued: Notify,
    closed: AtomicBool,
    closing: Notify,
    permits: Arc<Semaphore>,
    pace: Mutex<Pace>,
    flights: Mutex<Vec<Arc<Flight>>>,
    measured: Mutex<Measured>,
    /// Transfers running; the stillness clock restarts when the first begins.
    active: AtomicUsize,
    /// When any object last gained bytes, in nanoseconds since `epoch`.
    advanced: AtomicU64,
    /// The longest the fetch stood still and then moved again, in nanoseconds.
    worst_still: AtomicU64,
    /// Tasks and write-backs not yet ended.
    busy: AtomicUsize,
    settle: Notify,
    writes: Semaphore,
    volume_reads: Semaphore,
    write_backs: Semaphore,
    pub stats: Counters,
}

#[derive(Default)]
pub(crate) struct Counters {
    pub requests: AtomicU64,
    pub restarts: AtomicU64,
    pub landed: AtomicU64,
    pub volume_written: AtomicU64,
    pub volume_skipped: AtomicU64,
}

enum Attempt {
    Done,
    /// Ask again: after `wait` when the origin stated one; `dropped` when the watch judged it.
    Again {
        why: String,
        wait: Option<Duration>,
        dropped: bool,
    },
    /// 401/403: the location may be stale.
    Refused(Refusal),
    Fatal(Refusal),
}

impl From<Refusal> for Attempt {
    fn from(refusal: Refusal) -> Attempt {
        match refusal.code {
            Code::TRANSFER_FAILED | Code::HUB_UNREACHABLE | Code::FD_HEADROOM => Attempt::Again {
                why: refusal.to_string(),
                wait: None,
                dropped: false,
            },
            _ => Attempt::Fatal(refusal),
        }
    }
}

/// How a file read ended.
enum FileRead {
    Admitted,
    Absent,
    /// The bytes were not the object's.
    Corrupt(Refusal),
    /// Abandoned as silent; the operation may still be blocked.
    Stuck(String),
    Failed(Refusal),
}

/// One object's bytes on their way into the store's admission.
struct Sink<'s> {
    shared: &'s Shared,
    task: &'s Task,
    admission: Option<Box<dyn Admission>>,
    buffer: Vec<u8>,
    reached: u64,
}

impl<'s> Sink<'s> {
    fn open(shared: &'s Shared, task: &'s Task) -> Result<Sink<'s>> {
        let admission = Some(shared.store.admit(&task.object)?);
        Ok(Sink {
            shared,
            task,
            admission,
            buffer: Vec::with_capacity(WRITE_BYTES),
            reached: 0,
        })
    }

    /// Start the object over.
    fn fresh(&mut self) -> Result<()> {
        self.admission = Some(self.shared.store.admit(&self.task.object)?);
        self.buffer.clear();
        self.reached = 0;
        Ok(())
    }

    async fn push(&mut self, flight: &Flight, data: &[u8]) -> Result<()> {
        let length = self.task.object.length;
        if self.reached + data.len() as u64 > length {
            return refuse(
                Code::LENGTH_MISMATCH,
                format!(
                    "{}: the origin sent more than its {length} B",
                    self.task.object.sha256
                ),
            );
        }
        self.buffer.extend_from_slice(data);
        self.reached += data.len() as u64;
        self.shared.advance(self.task, self.reached);
        if self.buffer.len() >= WRITE_BYTES {
            self.flush(Some(flight)).await?;
        }
        Ok(())
    }

    async fn flush(&mut self, flight: Option<&Flight>) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let Some(mut admission) = self.admission.take() else {
            return refuse(Code::IO_FAILED, "the admission is gone");
        };
        let buffer = std::mem::take(&mut self.buffer);
        if let Some(flight) = flight {
            flight.local.store(true, Ordering::Release);
            flight.stretch_ends(&self.shared.measured);
        }
        let _writing = self.shared.writes.acquire().await;
        let written = tokio::task::spawn_blocking(move || {
            let written = admission.write(&buffer);
            (admission, buffer, written)
        })
        .await;
        if let Some(flight) = flight {
            flight.moved();
            flight.stretch_begins();
            lock(&flight.window).clear();
            flight.local.store(false, Ordering::Release);
        }
        let (admission, mut buffer, written) = written.map_err(|error| Refusal {
            code: Code::IO_FAILED,
            detail: error.to_string(),
        })?;
        buffer.clear();
        (self.admission, self.buffer) = (Some(admission), buffer);
        written
    }

    /// Everything written, hashed and admitted; the sink is spent until `fresh`.
    async fn commit(&mut self) -> Result<()> {
        self.flush(None).await?;
        let Some(admission) = self.admission.take() else {
            return refuse(Code::IO_FAILED, "the admission is gone");
        };
        let _writing = self.shared.writes.acquire().await;
        tokio::task::spawn_blocking(move || admission.commit())
            .await
            .map_err(|error| Refusal {
                code: Code::IO_FAILED,
                detail: error.to_string(),
            })?
    }
}

impl Shared {
    pub fn new(
        store: Arc<dyn ObjectStore>,
        client: Client,
        volume: Option<VolumeOrigin>,
        tick: Duration,
        streams: usize,
        max_streams: usize,
    ) -> Shared {
        let max = crate::descriptors::requests(max_streams.max(1));
        let level = streams.clamp(1, max);
        Shared {
            store,
            client,
            volume,
            tick: tick.max(Duration::from_millis(1)),
            epoch: Instant::now(),
            queue: Mutex::default(),
            queued: Notify::new(),
            closed: AtomicBool::new(false),
            closing: Notify::new(),
            permits: Arc::new(Semaphore::new(level)),
            pace: Mutex::new(Pace {
                level,
                max,
                last: None,
                settled: false,
                debt: 0,
            }),
            flights: Mutex::default(),
            measured: Mutex::default(),
            active: AtomicUsize::new(0),
            advanced: AtomicU64::new(0),
            worst_still: AtomicU64::new(0),
            busy: AtomicUsize::new(0),
            settle: Notify::new(),
            writes: Semaphore::new(WRITERS),
            volume_reads: Semaphore::new(VOLUME_READS),
            write_backs: Semaphore::new(WRITE_BACKS),
            stats: Counters::default(),
        }
    }

    fn now(&self) -> u64 {
        self.epoch.elapsed().as_nanos() as u64
    }

    pub fn streams(&self) -> usize {
        lock(&self.pace).level
    }

    /// Stops the dispatcher and the watch, and every request in flight.
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.closing.notify_waiters();
        self.queued.notify_waiters();
        for flight in lock(&self.flights).iter() {
            flight.drop_with(Dropped::Unwanted);
        }
    }

    async fn closed(&self) {
        loop {
            let notified = self.closing.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.closed.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }

    /// Resolves once no task or write-back is left.
    pub async fn settled(&self) {
        loop {
            let notified = self.settle.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.busy.load(Ordering::Acquire) == 0 {
                return;
            }
            notified.await;
        }
    }

    fn unbusy(&self) {
        if self.busy.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.settle.notify_waiters();
        }
    }

    /// A place in line for `object`, joining its task when one is open.
    pub fn want(&self, object: &ObjectRef, origin: &Arc<dyn Origin>, priority: u64) -> Waiter {
        let mut queue = lock(&self.queue);
        let task = match queue.tasks.get(&object.sha256).cloned() {
            Some(task) => {
                task.waiters.fetch_add(1, Ordering::AcqRel);
                if priority < task.priority.load(Ordering::Acquire)
                    && !task.started.load(Ordering::Acquire)
                {
                    task.priority.store(priority, Ordering::Release);
                    queue.seq += 1;
                    let seq = queue.seq;
                    queue
                        .heap
                        .push(Reverse((priority, seq, object.sha256.clone())));
                }
                task
            }
            None => {
                let task = Arc::new(Task {
                    object: object.clone(),
                    origin: Arc::clone(origin),
                    priority: AtomicU64::new(priority),
                    waiters: AtomicUsize::new(1),
                    started: AtomicBool::new(false),
                    since: AtomicU64::new(0),
                    reached: AtomicU64::new(0),
                    done: watch::channel(None).0,
                });
                queue.seq += 1;
                let seq = queue.seq;
                queue
                    .heap
                    .push(Reverse((priority, seq, object.sha256.clone())));
                queue.tasks.insert(object.sha256.clone(), Arc::clone(&task));
                self.busy.fetch_add(1, Ordering::AcqRel);
                self.queued.notify_one();
                task
            }
        };
        Waiter {
            answer: task.done.subscribe(),
            task,
        }
    }

    /// The most urgent task nobody has started that someone still wants.
    fn pop(&self) -> Option<Arc<Task>> {
        let mut queue = lock(&self.queue);
        while let Some(Reverse((priority, _, sha256))) = queue.heap.pop() {
            let Some(task) = queue.tasks.get(&sha256).cloned() else {
                continue;
            };
            if task.started.load(Ordering::Acquire)
                || task.priority.load(Ordering::Acquire) != priority
            {
                continue;
            }
            if !task.wanted() {
                queue.tasks.remove(&sha256);
                self.unbusy();
                continue;
            }
            task.started.store(true, Ordering::Release);
            return Some(task);
        }
        None
    }

    fn waiting(&self) -> bool {
        !lock(&self.queue).heap.is_empty()
    }

    fn finish(&self, task: &Arc<Task>, outcome: Result<Tier>) {
        let mut queue = lock(&self.queue);
        if queue
            .tasks
            .get(&task.object.sha256)
            .is_some_and(|t| Arc::ptr_eq(t, task))
        {
            queue.tasks.remove(&task.object.sha256);
        }
        drop(queue);
        task.done.send_replace(Some(outcome));
        self.unbusy();
    }

    /// Starts tasks as workers come free, most urgent first.
    pub async fn dispatch(self: Arc<Self>) {
        loop {
            let permit = tokio::select! {
                permit = Arc::clone(&self.permits).acquire_owned() => match permit {
                    Ok(permit) => permit,
                    Err(_) => return,
                },
                () = self.closed() => return,
            };
            let task = loop {
                let notified = self.queued.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                if self.closed.load(Ordering::Acquire) {
                    return;
                }
                if let Some(task) = self.pop() {
                    break task;
                }
                notified.await;
            };
            let shared = Arc::clone(&self);
            tokio::spawn(async move {
                let outcome = shared.run(&task).await;
                let mut pace = lock(&shared.pace);
                if pace.debt > 0 {
                    pace.debt -= 1;
                    permit.forget();
                }
                drop(pace);
                shared.finish(&task, outcome);
            });
        }
    }

    async fn run(self: &Arc<Self>, task: &Arc<Task>) -> Result<Tier> {
        struct Active<'a>(&'a AtomicUsize);
        impl Drop for Active<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::AcqRel);
            }
        }
        if self.active.fetch_add(1, Ordering::AcqRel) == 0 {
            self.advanced.store(self.now(), Ordering::Release);
        }
        let _active = Active(&self.active);
        task.since.store(self.now(), Ordering::Release);
        let outcome = self.fetch(task).await;
        if let Err(refusal) = &outcome {
            tracing::info!(object = %task.object.sha256, %refusal, "fetch refused");
        }
        outcome
    }

    /// `task`'s object from the first tier that has it.
    async fn fetch(self: &Arc<Self>, task: &Arc<Task>) -> Result<Tier> {
        if self.store.has(&task.object) {
            return Ok(Tier::Local);
        }
        let mut replace = false;
        if let Some(volume) = self.volume.clone().filter(VolumeOrigin::answering) {
            let _reading = self.volume_reads.acquire().await;
            if volume.answering() {
                match self
                    .read_file(task, &volume, volume.blob(&task.object), 0)
                    .await
                {
                    FileRead::Admitted => {
                        let (touching, object) = (volume.clone(), task.object.clone());
                        tokio::task::spawn_blocking(move || touching.touch(&object));
                        return Ok(Tier::Volume);
                    }
                    FileRead::Corrupt(refusal) => {
                        tracing::info!(object = %task.object.sha256, %refusal, "volume copy refused");
                        replace = true;
                    }
                    FileRead::Absent | FileRead::Stuck(_) => {}
                    FileRead::Failed(refusal) => return Err(refusal),
                }
            }
        }
        if !task.wanted() {
            return refuse(Code::TRANSFER_FAILED, "nothing wants it");
        }
        self.from_origin(task).await?;
        if task.origin.public() {
            self.write_back(task.object.clone(), replace);
        }
        Ok(Tier::Origin)
    }

    /// The origin's copy, asked again from the byte reached until it lands or ends.
    async fn from_origin(self: &Arc<Self>, task: &Arc<Task>) -> Result<()> {
        let mut sink = Sink::open(self, task)?;
        let (mut location, mut delivery, mut relocated) = (None::<Location>, None::<String>, false);
        let (mut doublings, mut wrong) = (0, None::<String>);
        loop {
            if !task.wanted() {
                return refuse(Code::TRANSFER_FAILED, "nothing wants it");
            }
            let here = match location.take().filter(|l| !l.expired()) {
                Some(here) => here,
                None => {
                    delivery = None;
                    let located = task.origin.locate(std::slice::from_ref(&task.object)).await;
                    match located.map(|mut located| located.pop()) {
                        Ok(Some(here)) => here,
                        Ok(None) => {
                            let what = format!("{}: the origin located nothing", task.object.sha256);
                            return refuse(Code::REF_NOT_FOUND, what);
                        }
                        // The origin's weather is asked again, under the same stillness rule.
                        Err(refusal) => match Attempt::from(refusal) {
                            Attempt::Again { why, .. } => {
                                if let Some(refusal) = self.given_up(task, &why) {
                                    return Err(refusal);
                                }
                                tokio::time::sleep(self.tick).await;
                                continue;
                            }
                            Attempt::Fatal(refusal) | Attempt::Refused(refusal) => {
                                return Err(refusal)
                            }
                            Attempt::Done => unreachable!("a refusal is never done"),
                        },
                    }
                }
            };
            let reached = task.reached.load(Ordering::Acquire);
            let attempt = match &here {
                Location::File(path) => match self
                    .read_file_into(task, &mut sink, path.clone(), doublings, None)
                    .await
                {
                    FileRead::Admitted => return Ok(()),
                    FileRead::Absent => {
                        return refuse(Code::REF_NOT_FOUND, format!("{}: absent", path.display()))
                    }
                    FileRead::Corrupt(refusal) => Attempt::Fatal(refusal),
                    FileRead::Stuck(why) => Attempt::Again {
                        why,
                        wait: None,
                        dropped: true,
                    },
                    FileRead::Failed(refusal) => Attempt::Fatal(refusal),
                },
                Location::Url { url, headers, .. } => {
                    let (delivered, asked) = (
                        delivery.is_some(),
                        delivery.clone().unwrap_or_else(|| url.clone()),
                    );
                    self.request(
                        task,
                        &mut sink,
                        &asked,
                        headers,
                        doublings,
                        delivered,
                        &mut delivery,
                    )
                    .await
                }
            };
            location = Some(here);
            match attempt {
                Attempt::Done => match sink.commit().await {
                    Ok(()) => return Ok(()),
                    Err(refusal) if refusal.code == Code::OBJECT_ID_MISMATCH => {
                        // Different wrong bytes may be the link's; the same twice are the origin's.
                        if wrong.as_ref() == Some(&refusal.detail) {
                            return Err(refusal);
                        }
                        tracing::info!(object = %task.object.sha256, %refusal, "refused, asked again");
                        wrong = Some(refusal.detail.clone());
                        sink.fresh()?;
                    }
                    Err(refusal) => return Err(refusal),
                },
                Attempt::Again { why, wait, dropped } => {
                    if dropped {
                        self.stats.restarts.fetch_add(1, Ordering::Relaxed);
                        doublings = (doublings + 1).min(MAX_DOUBLINGS);
                        tracing::debug!(object = %task.object.sha256, at = sink.reached, %why, "restart");
                    }
                    if let Some(refusal) = self.given_up(task, &why) {
                        return Err(refusal);
                    }
                    let moved = task.reached.load(Ordering::Acquire) > reached;
                    let pause = wait.or((!moved && !dropped).then_some(self.tick));
                    if let Some(pause) = pause {
                        tokio::time::sleep(pause).await;
                    }
                }
                Attempt::Refused(refusal) => {
                    if relocated {
                        return Err(refusal);
                    }
                    (relocated, location) = (true, None);
                }
                Attempt::Fatal(refusal) => return Err(refusal),
            }
        }
    }

    /// The refusal that ends `task` when nothing anywhere has landed for twice the patience the
    /// fetch has measured.
    fn given_up(&self, task: &Task, last: &str) -> Option<Refusal> {
        let since = self
            .advanced
            .load(Ordering::Acquire)
            .max(task.since.load(Ordering::Acquire));
        let still = Duration::from_nanos(self.now().saturating_sub(since));
        let worst = Duration::from_nanos(self.worst_still.load(Ordering::Acquire));
        let patience = (self.tick * STILL_TICKS).max(worst.mul_f64(STALL_FACTOR));
        (still > patience * 2).then(|| Refusal {
            code: Code::STALLED,
            detail: format!(
                "{}: nothing landed for {:.1} s against the {:.1} s this fetch's own pace allows; the last answer: {last}",
                task.object.sha256,
                still.as_secs_f64(),
                patience.as_secs_f64()
            ),
        })
    }

    /// `task` reached `reached` bytes: what is past its earlier best is progress.
    fn advance(&self, task: &Task, reached: u64) {
        let before = task.reached.fetch_max(reached, Ordering::AcqRel);
        if reached > before {
            self.stats
                .landed
                .fetch_add(reached - before, Ordering::Relaxed);
            let now = self.now();
            let gap = now.saturating_sub(self.advanced.swap(now, Ordering::AcqRel));
            self.worst_still.fetch_max(gap, Ordering::AcqRel);
        }
    }

    fn fly(&self, task: &Arc<Task>, file: bool, doublings: u32) -> Flying<'_> {
        let flight = Arc::new(Flight {
            task: Arc::clone(task),
            file,
            began: Instant::now(),
            doublings,
            answered: AtomicBool::new(false),
            bytes: AtomicU64::new(0),
            moved: AtomicU64::new(0),
            local: AtomicBool::new(false),
            stretch: Mutex::default(),
            window: Mutex::default(),
            dropped: Mutex::new(None),
            abort: Notify::new(),
        });
        lock(&self.flights).push(Arc::clone(&flight));
        Flying {
            shared: self,
            flight,
        }
    }

    fn dropped(flight: &Flight) -> Attempt {
        match flight.why() {
            Dropped::Judged(why) => Attempt::Again {
                why,
                wait: None,
                dropped: true,
            },
            Dropped::Unwanted => Attempt::Fatal(Refusal {
                code: Code::TRANSFER_FAILED,
                detail: "nothing wants it".into(),
            }),
        }
    }

    /// One GET of what `sink` still lacks. `delivered`: `url` is where the origin redirected an
    /// earlier ask, which a refusal sends back to the origin.
    #[allow(clippy::too_many_arguments)]
    async fn request(
        &self,
        task: &Arc<Task>,
        sink: &mut Sink<'_>,
        url: &str,
        headers: &[(String, String)],
        doublings: u32,
        delivered: bool,
        delivery: &mut Option<String>,
    ) -> Attempt {
        let object = &task.object;
        let flying = self.fly(task, false, doublings);
        let flight = &flying.flight;
        self.stats.requests.fetch_add(1, Ordering::Relaxed);
        let mut headers = headers.to_vec();
        if sink.reached > 0 {
            headers.push((
                "range".into(),
                format!("bytes={}-{}", sink.reached, object.length - 1),
            ));
        }
        let answer = tokio::select! {
            answer = self.client.get(url, &headers) => answer,
            () = flight.aborted() => return Shared::dropped(flight),
        };
        let mut answer = match answer {
            Ok(answer) => answer,
            Err(refusal) => {
                if refusal.code == Code::FD_HEADROOM {
                    self.backoff();
                }
                return refusal.into();
            }
        };
        let head = flight.began.elapsed();
        push(&mut lock(&self.measured).heads, head.as_secs_f64());
        flight.answered.store(true, Ordering::Release);
        flight.moved();
        flight.stretch_begins();
        if answer.url != url {
            *delivery = Some(answer.url.clone());
        }
        let status = answer.status();
        let what = || format!("{}: {status} from {url}", object.sha256);
        match status {
            200 => {
                if sink.reached > 0 {
                    // The origin ignored the range: the object starts over from this answer.
                    if let Err(refusal) = sink.fresh() {
                        return Attempt::Fatal(refusal);
                    }
                }
                let length = answer
                    .header("content-length")
                    .and_then(|v| v.parse::<u64>().ok());
                if length.is_some_and(|length| length != object.length) {
                    return Attempt::Fatal(Refusal {
                        code: Code::LENGTH_MISMATCH,
                        detail: format!("{}: {length:?} B, not {}", what(), object.length),
                    });
                }
            }
            206 => {
                let range = answer
                    .header("content-range")
                    .unwrap_or_default()
                    .to_string();
                let first = range
                    .strip_prefix("bytes ")
                    .and_then(|r| r.split_once('-'))
                    .and_then(|(a, _)| a.parse::<u64>().ok());
                let total = range.rsplit_once('/').map(|(_, total)| total);
                if first != Some(sink.reached)
                    || total.is_some_and(|t| t != "*" && t != object.length.to_string())
                {
                    return Attempt::Fatal(Refusal {
                        code: Code::LENGTH_MISMATCH,
                        detail: format!(
                            "{}: content-range {range:?} for byte {}",
                            what(),
                            sink.reached
                        ),
                    });
                }
            }
            401 | 403 | 404 | 410 if delivered => {
                *delivery = None;
                return Attempt::Again {
                    why: what(),
                    wait: None,
                    dropped: false,
                };
            }
            401 | 403 => {
                return Attempt::Refused(Refusal {
                    code: Code::CREDENTIAL_REQUIRED,
                    detail: what(),
                })
            }
            404 | 410 => {
                return Attempt::Fatal(Refusal {
                    code: Code::REF_NOT_FOUND,
                    detail: what(),
                })
            }
            416 => {
                return Attempt::Fatal(Refusal {
                    code: Code::LENGTH_MISMATCH,
                    detail: what(),
                })
            }
            408 | 425 | 429 | 500..=599 => {
                return Attempt::Again {
                    why: what(),
                    wait: answer.stated_wait(),
                    dropped: false,
                }
            }
            _ => {
                return Attempt::Fatal(Refusal {
                    code: Code::HUB_REFUSED,
                    detail: what(),
                })
            }
        }
        loop {
            let frame = tokio::select! {
                frame = answer.frame() => frame,
                () = flight.aborted() => return Shared::dropped(flight),
            };
            match frame {
                Ok(Some(data)) => {
                    flight.bytes.fetch_add(data.len() as u64, Ordering::AcqRel);
                    flight.moved();
                    if let Err(refusal) = sink.push(flight, &data).await {
                        return refusal.into();
                    }
                }
                Ok(None) => {
                    flight.stretch_ends(&self.measured);
                    break;
                }
                Err(refusal) => {
                    let why = format!(
                        "{}: {} at {} of {} B",
                        what(),
                        refusal.detail,
                        sink.reached,
                        object.length
                    );
                    return Attempt::Again {
                        why,
                        wait: None,
                        dropped: false,
                    };
                }
            }
        }
        match sink.reached == object.length {
            true => Attempt::Done,
            false => Attempt::Again {
                why: format!(
                    "{}: the body ended at {} of {} B",
                    what(),
                    sink.reached,
                    object.length
                ),
                wait: None,
                dropped: false,
            },
        }
    }

    /// `object` from a volume, judged by its silence.
    async fn read_file(
        &self,
        task: &Arc<Task>,
        volume: &VolumeOrigin,
        path: PathBuf,
        doublings: u32,
    ) -> FileRead {
        let mut sink = match Sink::open(self, task) {
            Ok(sink) => sink,
            Err(refusal) => return FileRead::Failed(refusal),
        };
        self.read_file_into(task, &mut sink, path, doublings, Some(volume))
            .await
    }

    async fn read_file_into(
        &self,
        task: &Arc<Task>,
        sink: &mut Sink<'_>,
        path: PathBuf,
        doublings: u32,
        volume: Option<&VolumeOrigin>,
    ) -> FileRead {
        let flying = self.fly(task, true, doublings);
        let flight = &flying.flight;
        if sink.reached > 0 {
            if let Err(refusal) = sink.fresh() {
                return FileRead::Failed(refusal);
            }
        }
        let length = task.object.length;
        let opened = blocking(
            volume,
            move || -> std::io::Result<std::fs::File> {
                let file = std::fs::File::open(&path)?;
                let metadata = file.metadata()?;
                match metadata.is_file() && metadata.len() == length {
                    true => Ok(file),
                    false => Err(std::io::ErrorKind::NotFound.into()),
                }
            },
            flight.aborted(),
        )
        .await;
        let mut file = match opened {
            None => return self.file_dropped(flight),
            Some(Err(_)) => return FileRead::Absent,
            Some(Ok(file)) => file,
        };
        flight.answered.store(true, Ordering::Release);
        flight.moved();
        while sink.reached < length {
            let want = (length - sink.reached).min(STEP as u64) as usize;
            let began = Instant::now();
            let step = blocking(
                volume,
                move || {
                    let mut buffer = vec![0; want];
                    let read = file.read_exact(&mut buffer);
                    (file, buffer, read)
                },
                flight.aborted(),
            )
            .await;
            let Some((back, buffer, read)) = step else {
                return self.file_dropped(flight);
            };
            if read.is_err() {
                return FileRead::Absent;
            }
            file = back;
            {
                let mut measured = lock(&self.measured);
                measured.file_step = measured.file_step.max(began.elapsed());
            }
            flight
                .bytes
                .fetch_add(buffer.len() as u64, Ordering::AcqRel);
            flight.moved();
            if let Err(refusal) = sink.push(flight, &buffer).await {
                return FileRead::Failed(refusal);
            }
        }
        match sink.commit().await {
            Ok(()) => FileRead::Admitted,
            Err(refusal) if refusal.code == Code::OBJECT_ID_MISMATCH => FileRead::Corrupt(refusal),
            Err(refusal) => FileRead::Failed(refusal),
        }
    }

    fn file_dropped(&self, flight: &Flight) -> FileRead {
        match flight.why() {
            Dropped::Judged(why) => FileRead::Stuck(why),
            Dropped::Unwanted => FileRead::Failed(Refusal {
                code: Code::TRANSFER_FAILED,
                detail: "nothing wants it".into(),
            }),
        }
    }

    /// A public object onto the volume, behind everything else.
    fn write_back(self: &Arc<Self>, object: ObjectRef, replace: bool) {
        let Some(volume) = self.volume.clone() else {
            return;
        };
        self.busy.fetch_add(1, Ordering::AcqRel);
        let shared = Arc::clone(self);
        tokio::spawn(async move {
            if let Ok(_writing) = shared.write_backs.acquire().await {
                if volume.answering() {
                    let store = Arc::clone(&shared.store);
                    let wrote = tokio::task::spawn_blocking(move || {
                        volume.write(&*store, &object, replace)
                    })
                    .await;
                    match wrote {
                        Ok(Wrote::Written) => {
                            shared.stats.volume_written.fetch_add(1, Ordering::Relaxed)
                        }
                        Ok(Wrote::Skipped) => {
                            shared.stats.volume_skipped.fetch_add(1, Ordering::Relaxed)
                        }
                        _ => 0,
                    };
                }
            }
            shared.unbusy();
        });
    }

    /// The descriptor table is full: half the workers go.
    fn backoff(&self) {
        let mut pace = lock(&self.pace);
        let gone = pace.level / 2;
        if gone == 0 {
            return;
        }
        (pace.level, pace.settled, pace.last) = (pace.level - gone, true, None);
        let forgotten = self.permits.forget_permits(gone);
        pace.debt += gone - forgotten;
        tracing::info!(streams = pace.level, "descriptor table full: fewer streams");
    }

    /// Samples and judges every request in flight, each tick, and paces the workers.
    pub async fn watch(self: Arc<Self>) {
        let (mut ticks, mut landed) = (0u32, 0u64);
        loop {
            tokio::select! {
                () = tokio::time::sleep(self.tick) => {}
                () = self.closed() => return,
            }
            self.judge();
            ticks += 1;
            if ticks.is_multiple_of(PACE_TICKS) {
                let now = self.stats.landed.load(Ordering::Relaxed);
                let goodput = (now - landed) as f64 / (self.tick * PACE_TICKS).as_secs_f64();
                landed = now;
                self.pace(goodput);
            }
        }
    }

    fn pace(&self, goodput: f64) {
        let saturated = self.permits.available_permits() == 0 && self.waiting();
        let mut pace = lock(&self.pace);
        if self.active.load(Ordering::Acquire) == 0 && !self.waiting() {
            (pace.last, pace.settled) = (None, false);
            return;
        }
        if !saturated || pace.settled || pace.level >= pace.max || pace.debt > 0 {
            return;
        }
        if pace.last.is_some_and(|last| goodput < last * PACE_GAIN) {
            pace.settled = true;
            return;
        }
        let grow = (pace.level * 2).min(pace.max) - pace.level;
        (pace.last, pace.level) = (Some(goodput), pace.level + grow);
        self.permits.add_permits(grow);
        tracing::debug!(streams = pace.level, goodput, "more streams");
    }

    fn judge(&self) {
        let flights: Vec<Arc<Flight>> = lock(&self.flights).clone();
        let tick = self.tick.as_secs_f64();
        for flight in flights.iter().filter(|f| {
            !f.file && f.answered.load(Ordering::Acquire) && !f.local.load(Ordering::Acquire)
        }) {
            let mut window = lock(&flight.window);
            window.push_back(flight.bytes.load(Ordering::Acquire));
            if window.len() > (JUDGE_TICKS << flight.doublings) + 1 {
                window.pop_front();
            }
        }
        let measured = lock(&self.measured);
        let floor = self.tick * HEAD_TICKS;
        let head = median(&measured.heads).map_or(floor, |h| {
            floor.max(Duration::from_secs_f64(h * STALL_FACTOR))
        });
        let reference = (measured.rates.len() >= MIN_SAMPLES)
            .then(|| median(&measured.rates))
            .flatten();
        let file_patience = floor.max(measured.file_step.mul_f64(STALL_FACTOR));
        drop(measured);
        for flight in flights {
            if lock(&flight.dropped).is_some() {
                continue;
            }
            if !flight.task.wanted() {
                flight.drop_with(Dropped::Unwanted);
                continue;
            }
            if flight.local.load(Ordering::Acquire) {
                continue;
            }
            let scale = 1u32 << flight.doublings;
            let silent = flight.silent();
            let judged =
                |why: String| Dropped::Judged(format!("{}: {why}", flight.task.object.sha256));
            if flight.file {
                if silent > file_patience * scale {
                    flight.drop_with(judged(format!(
                        "the volume was silent {:.1} s against {:.1} s",
                        silent.as_secs_f64(),
                        (file_patience * scale).as_secs_f64()
                    )));
                }
                continue;
            }
            if !flight.answered.load(Ordering::Acquire) {
                let waited = flight.began.elapsed();
                if waited > head * scale {
                    flight.drop_with(judged(format!(
                        "no answer after {:.2} s against the {:.2} s answers take here",
                        waited.as_secs_f64(),
                        (head * scale).as_secs_f64()
                    )));
                }
                continue;
            }
            let window = lock(&flight.window).clone();
            let full = (JUDGE_TICKS << flight.doublings) + 1;
            match reference {
                Some(reference) if window.len() >= full => {
                    let rate =
                        (window[window.len() - 1] - window[0]) as f64 / (tick * (full - 1) as f64);
                    if rate * STALL_FACTOR < reference {
                        flight.drop_with(judged(format!(
                            "{:.0} KB/s against the {:.0} KB/s requests move here",
                            rate / 1e3,
                            reference / 1e3
                        )));
                    }
                }
                None if silent > head * scale => {
                    flight.drop_with(judged(format!(
                        "silent {:.2} s against {:.2} s",
                        silent.as_secs_f64(),
                        (head * scale).as_secs_f64()
                    )));
                }
                _ => {}
            }
        }
    }
}

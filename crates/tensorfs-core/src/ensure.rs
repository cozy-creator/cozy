//! `ensure`: make one model resident. The whole-checkpoint download for `tfs ensure` and
//! `tensorfs.ensure`; what it lacks is fetched by the Store's [`Fetcher`].
//!
//! Resolve the closure; lead or join the Store's one flight for its manifest; admit the new
//! bytes against the disk, running the GC policy when short; want what is absent at Now and
//! wait for it; prove the closure here and record it. Events sample bytes and rate while it
//! runs; the answer is an [`Ensured`] or a typed refusal. Nothing here is a timer: the
//! fetcher's downloader gives up only on measured stillness.
//!
//! The Hub CA is process trust ([`crate::transport::trust_roots`]); callers name it once.

mod admission;
mod flight;

pub use admission::{relieve, Disk, Relief};
pub(crate) use admission::{free, Space};
pub(crate) use flight::held_objects;

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::err::{Code, Result};
use crate::fetch::{Access, FetchPlan, Fetcher, Origin, Priority};
use crate::store::Store;
use crate::transport::{self, PullCancellation, PullRequest};

use flight::{Flight, Record};

/// Events are samples at this resolution. It decides nothing.
pub const EVENT_SECONDS: f64 = 1.0;

/// One model to make resident.
pub struct Request<'a> {
    pub store: &'a Store,
    /// The hub's base URL.
    pub hub: &'a str,
    /// `org/name[@release][@sha256:<hex>]`.
    pub refspec: &'a str,
    pub lane: &'a str,
    pub access: Arc<dyn Access>,
    /// The step this download serves; echoed on every event.
    pub step: &'a str,
    /// Manifests the GC policy must not evict: the caller's serving set.
    pub keep: &'a [String],
    pub cancellation: Option<PullCancellation>,
    /// The most requests in flight, and the transport ledger's resolution: configuration,
    /// never a verdict. Other than the defaults, the ensure fetches on a fetcher of its own.
    pub streams: usize,
    pub sample_seconds: f64,
    pub on_event: Option<&'a (dyn Fn(&Event) + Sync)>,
}

impl<'a> Request<'a> {
    pub fn new(store: &'a Store, hub: &'a str, refspec: &'a str, access: Arc<dyn Access>) -> Request<'a> {
        Request {
            store,
            hub,
            refspec,
            lane: "",
            access,
            step: "",
            keep: &[],
            cancellation: None,
            streams: transport::PULL_STREAMS,
            sample_seconds: transport::SAMPLE_SECONDS,
            on_event: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Asking the hub (or the Store) which manifest the ref names.
    Resolving,
    /// Another process leads this manifest's flight; its progress is reported.
    Waiting,
    /// The disk is short; the GC policy is running.
    Collecting,
    Fetching,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Resolving => "resolving",
            Phase::Waiting => "waiting",
            Phase::Collecting => "collecting",
            Phase::Fetching => "fetching",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    pub model: String,
    pub step: String,
    pub phase: Phase,
    pub bytes_done: u64,
    pub bytes_total: u64,
    /// Bytes per second over the last sample.
    pub rate: f64,
}

/// What `ensure` did. Every figure is counted, never estimated.
#[derive(Debug, Clone, PartialEq)]
pub struct Ensured {
    pub model: String,
    pub release: String,
    pub lane: String,
    pub scope: String,
    pub manifest: crate::ids::ObjectRef,
    pub bytes_total: u64,
    /// Resident before this ensure moved anything.
    pub bytes_held: u64,
    /// Landed from the origin.
    pub bytes_fetched: u64,
    /// Landed from the Store's mounted repo cache.
    pub bytes_cached: u64,
    /// Freed by the GC policy to admit this download.
    pub collected_bytes: u64,
    /// Another process's flight for this manifest was running; this one waited for it.
    pub joined: bool,
    pub seconds: f64,
}

/// Whether asking again later can succeed. Integrity, trust and policy verdicts cannot.
pub fn resumable(code: Code) -> bool {
    !matches!(
        code,
        Code::OBJECT_ID_MISMATCH
            | Code::LENGTH_MISMATCH
            | Code::WHOLE_DIGEST_MISMATCH
            | Code::OBJECT_CORRUPT
            | Code::MALFORMED_DIGEST
            | Code::SOURCE_NOT_ALLOWED
            | Code::REDIRECT_REFUSED
            | Code::PERMISSION_DENIED
            | Code::STORE_ERA
            | Code::TLS_UNTRUSTED
            | Code::CREDENTIAL_REQUIRED
    )
}

pub fn ensure(request: &Request<'_>) -> Result<Ensured> {
    let started = Instant::now();
    let store = request.store;
    let say = events(request);
    let tick = Duration::from_secs_f64(EVENT_SECONDS);
    // Resolving says so once per tick: a hub that takes its time is time passing on the
    // caller's display, never silence, and the caller's cancellation still ends it.
    let resolving = || say(Phase::Resolving, 0, 0, 0.0);
    resolving();
    let closure = ticking(tick, &resolving, || {
        let mut pull = PullRequest::new(store, request.hub, request.refspec, request.access.credential(), request.access.policy());
        pull.lane = request.lane;
        pull.cancellation = request.cancellation.clone();
        pull.sample_seconds = request.sample_seconds;
        transport::resolve_closure(&pull)
    })?;
    let model = request.refspec.split('@').next().unwrap_or_default();
    let declared = closure.manifest.length + closure.objects.iter().map(|o| o.length).sum::<u64>();
    let flight = Flight::lead(store, &closure.manifest.sha256, request.cancellation.as_ref(), tick, &|record| match record {
        // Before the leader has written its record the closure still says how big this is.
        Some(record) => say(Phase::Waiting, record.bytes_done, record.bytes_total, 0.0),
        None => say(Phase::Waiting, 0, declared, 0.0),
    })?;
    // A killed download strands its partial `put-` temps, and GC, the only other reaper,
    // never runs while a model is leased. Each temp carries its writer's lock.
    let _ = store.reap();
    let mut closure_objects = closure.objects.clone();
    closure_objects.push(closure.manifest.clone());
    flight.hold(&closure_objects)?;
    let session = format!("ensure-{}", &closure.manifest.sha256[..12]);
    let (plan, _) = FetchPlan::of(store, &session, &closure.manifest, &closure.objects)?;
    let total = plan.declared_bytes();
    let record = Record {
        model: model.to_string(),
        step: request.step.to_string(),
        bytes_done: plan.held_bytes(),
        bytes_total: total,
        wanted_bytes: plan.wanted_bytes(),
        wanted_objects: plan.wanted.len() as u64,
        ..Record::default()
    };
    let collected = admission::admit(store, &flight, &record, request.keep, &|| {
        say(Phase::Collecting, plan.held_bytes(), total, 0.0)
    })?;
    let origin = Origin {
        hub: request.hub.to_string(),
        refspec: request.refspec.to_string(),
        lane: request.lane.to_string(),
        manifest: closure.manifest.clone(),
    };
    say(Phase::Fetching, plan.held_bytes(), total, 0.0);
    let (mut fetched, mut cached) = (0, 0);
    if !plan.wanted.is_empty() {
        let fetcher = match (request.streams, request.sample_seconds) {
            (transport::PULL_STREAMS, transport::SAMPLE_SECONDS) => Fetcher::of(store)?,
            (streams, sample) => Fetcher::new(store, streams, Duration::from_secs_f64(sample))?,
        };
        if closure.presign_max_digests > 0 {
            fetcher.seed(&origin, closure.clone());
        }
        let ticket = fetcher.want(&origin, Arc::clone(&request.access), &plan.wanted, Priority::Now);
        let stop = request.cancellation.clone().unwrap_or_default();
        let (mut at, mut last) = (Instant::now(), 0);
        let sample = || {
            let done = (plan.held_bytes() + ticket.progress().0).min(total);
            let rate = done.saturating_sub(last) as f64 / at.elapsed().as_secs_f64();
            (at, last) = (Instant::now(), done);
            say(Phase::Fetching, done, total, rate);
            let _ = flight.record(&Record { bytes_done: done, landed_bytes: done.saturating_sub(plan.held_bytes()), ..record.clone() });
        };
        ticking_mut(tick, sample, || ticket.wait(&[], &stop))?;
        (fetched, cached) = ticket.landed();
    }
    // A warm checkpoint already recorded is proved by the plan's presence pass alone.
    if !plan.wanted.is_empty() || !transport::records(store, &closure) {
        match closure.scope.as_str() {
            "runtime" => plan.complete_cozytensors(store),
            _ => plan.complete(store),
        }?;
        transport::record(store, &closure)?;
    }
    // Whole and recorded: no longer a checkpoint held in part.
    crate::origin::retire(store, &closure.manifest)?;
    say(Phase::Fetching, total, total, fetched as f64 / started.elapsed().as_secs_f64());
    Ok(Ensured {
        model: closure.model.clone(),
        release: closure.release.clone(),
        lane: closure.lane.clone(),
        scope: closure.scope.clone(),
        manifest: closure.manifest.clone(),
        bytes_total: total,
        bytes_held: plan.held_bytes(),
        bytes_fetched: fetched,
        bytes_cached: cached,
        collected_bytes: collected,
        joined: flight.joined,
        seconds: started.elapsed().as_secs_f64(),
    })
}

/// What a request's events say, as they happen.
fn events<'r>(request: &'r Request<'_>) -> impl Fn(Phase, u64, u64, f64) + Sync + 'r {
    let model = request.refspec.split('@').next().unwrap_or_default();
    move |phase, bytes_done, bytes_total, rate| {
        if let Some(on_event) = request.on_event {
            on_event(&Event {
                model: model.to_string(),
                step: request.step.to_string(),
                phase,
                bytes_done,
                bytes_total,
                rate,
            });
        }
    }
}

/// Run `work`, calling `tick` once per `every` until it returns.
fn ticking<T>(every: Duration, tick: &(dyn Fn() + Sync), work: impl FnOnce() -> T) -> T {
    ticking_mut(every, || tick(), work)
}

fn ticking_mut<T>(every: Duration, mut tick: impl FnMut() + Send, work: impl FnOnce() -> T) -> T {
    let stop = (Mutex::new(false), Condvar::new());
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let mut stopped = stop.0.lock().unwrap();
            loop {
                stopped = stop.1.wait_timeout(stopped, every).unwrap().0;
                if *stopped {
                    return;
                }
                tick();
            }
        });
        let _stop = Stop(&stop);
        work()
    })
}

/// Stops a ticker as its work ends, by return or by unwind alike: a panic in the work (a
/// thread the container refused) would otherwise leave its scope waiting forever on a ticker
/// that never stops (TensorD 0.5.30 prepare wedge, 2026-10-15).
struct Stop<'a>(&'a (Mutex<bool>, Condvar));

impl Drop for Stop<'_> {
    fn drop(&mut self) {
        *self.0 .0.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = true;
        self.0 .1.notify_all();
    }
}

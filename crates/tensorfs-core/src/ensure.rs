//! `ensure`: make one model resident. The one entry the machine daemon (`tfs ensure`) and
//! the Runtime (`tensorfs.ensure`) use for model download.
//!
//! Resolve the closure; lead or join the Store's one flight for its manifest; admit the new
//! bytes against the disk, running the GC policy when short; pull, re-attempting on the
//! link's weather until no object has landed for the ledger's floor. Events sample bytes and
//! rate while it runs; the answer is an [`Ensured`] or a typed refusal. Nothing here is a
//! timer: giving up is earned by measured stillness, and a stalled stream is judged by the
//! transport's own ledger.
//!
//! The Hub CA is process trust ([`crate::transport::trust_roots`]); callers name it once.

mod admission;
mod flight;

pub use admission::{relieve, Relief};
pub(crate) use flight::held_objects;

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::err::{Code, Refusal, Result};
use crate::fetch::FetchPlan;
use crate::ids::ObjectRef;
use crate::store::{self, Store};
use crate::transport::{
    self, CredentialProvider, ObjectSource, PullCancellation, PullRequest, SourcePolicy,
};

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
    pub credential: &'a dyn CredentialProvider,
    /// The egress allowlist: object-storage hosts beside the hub.
    pub policy: &'a SourcePolicy,
    /// The step this download serves; echoed on every event.
    pub step: &'a str,
    /// Manifests the GC policy must not evict: the caller's serving set.
    pub keep: &'a [String],
    pub cancellation: Option<PullCancellation>,
    pub streams: usize,
    /// The transport ledger's resolution; configuration, never a verdict.
    pub sample_seconds: f64,
    pub on_event: Option<&'a (dyn Fn(&Event) + Sync)>,
}

impl<'a> Request<'a> {
    pub fn new(
        store: &'a Store,
        hub: &'a str,
        refspec: &'a str,
        credential: &'a dyn CredentialProvider,
        policy: &'a SourcePolicy,
    ) -> Request<'a> {
        Request {
            store,
            hub,
            refspec,
            lane: "",
            credential,
            policy,
            step: "",
            keep: &[],
            cancellation: None,
            streams: transport::PULL_STREAMS,
            sample_seconds: transport::SAMPLE_SECONDS,
            on_event: None,
        }
    }

    fn pull(&self, store: &'a Store) -> PullRequest<'a> {
        let mut pull =
            PullRequest::new(store, self.hub, self.refspec, self.credential, self.policy);
        pull.lane = self.lane;
        pull.cancellation = self.cancellation.clone();
        pull.streams = self.streams;
        pull.sample_seconds = self.sample_seconds;
        pull
    }

    fn cancelled(&self) -> bool {
        self.cancellation
            .as_ref()
            .is_some_and(PullCancellation::is_cancelled)
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
    pub attempt: u32,
}

/// What `ensure` did. Every figure is counted, never estimated.
#[derive(Debug, Clone, PartialEq)]
pub struct Ensured {
    pub model: String,
    pub release: String,
    pub lane: String,
    pub scope: String,
    pub manifest: ObjectRef,
    pub bytes_total: u64,
    /// Resident before this ensure moved anything.
    pub bytes_held: u64,
    /// Landed from the origin.
    pub bytes_fetched: u64,
    /// Landed from the Store's mounted repo cache.
    pub bytes_cached: u64,
    /// Written onward into the repo cache.
    pub cache_written_bytes: u64,
    /// Freed by the GC policy to admit this download.
    pub collected_bytes: u64,
    pub attempts: u32,
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

/// Refusals about the link rather than the request: another attempt may land more.
fn weather(code: Code) -> bool {
    matches!(code, Code::TRANSFER_FAILED | Code::HUB_UNREACHABLE)
}

/// Bytes the Store streams in and objects it admits, as they happen.
#[derive(Default)]
struct Meter {
    moved: AtomicU64,
}

impl store::Progress for Meter {
    fn moved(&self, bytes: u64) {
        self.moved.fetch_add(bytes, Ordering::Relaxed);
    }
    fn admitted(&self) {}
    fn discarded(&self, bytes: u64) {
        let _ = (self.moved).fetch_update(Ordering::Relaxed, Ordering::Relaxed, |moved| {
            Some(moved.saturating_sub(bytes))
        });
    }
}

/// The fetch's live counters, shared by the pull's callbacks and the sampler.
#[derive(Default)]
struct Counters {
    attempt: AtomicU32,
    /// Resident at the current attempt's plan, and the meter reading at that moment.
    held: AtomicU64,
    moved_at_plan: AtomicU64,
    landed_objects: AtomicU64,
    origin_bytes: AtomicU64,
    cache_bytes: AtomicU64,
}

pub fn ensure(request: &Request<'_>) -> Result<Ensured> {
    let started = Instant::now();
    let model = request.refspec.split('@').next().unwrap_or_default();
    let say = |phase: Phase, bytes_done: u64, bytes_total: u64, rate: f64, attempt: u32| {
        if let Some(on_event) = request.on_event {
            on_event(&Event {
                model: model.to_string(),
                step: request.step.to_string(),
                phase,
                bytes_done,
                bytes_total,
                rate,
                attempt,
            });
        }
    };
    let tick = Duration::from_secs_f64(EVENT_SECONDS);
    // Resolving says so once per tick: a hub that takes its time is time passing on the
    // caller's display, never silence, and the caller's cancellation still ends it.
    let resolving = || say(Phase::Resolving, 0, 0, 0.0, 0);
    resolving();
    let closure = ticking(tick, &resolving, || {
        transport::resolve_closure(&request.pull(request.store))
    })?;
    let declared = closure.manifest.length + closure.objects.iter().map(|o| o.length).sum::<u64>();
    let flight = Flight::lead(
        request.store,
        &closure.manifest.sha256,
        request.cancellation.as_ref(),
        tick,
        // Before the leader has written its record (it is still planning or admitting), the
        // closure still says how big this is: never a 0/0 bar.
        &|record| match record {
            Some(record) => say(
                Phase::Waiting,
                record.bytes_done,
                record.bytes_total,
                0.0,
                0,
            ),
            None => say(Phase::Waiting, 0, declared, 0.0, 0),
        },
    )?;
    // A killed pull strands its partial `put-` temps, and GC, the only other reaper, never
    // runs while a model is leased. Each temp carries its writer's lock, so this takes only
    // the dead ones.
    let _ = request.store.reap();
    let mut closure_objects = closure.objects.clone();
    closure_objects.push(closure.manifest.clone());
    flight.hold(&closure_objects)?;
    let session = format!("ensure-{}", &closure.manifest.sha256[..12]);
    let (plan, _) = FetchPlan::of(request.store, &session, &closure.manifest, &closure.objects)?;
    let total = plan.declared_bytes();
    let mut record = Record {
        model: model.to_string(),
        step: request.step.to_string(),
        bytes_done: plan.held_bytes(),
        bytes_total: total,
        wanted_bytes: plan.wanted_bytes(),
        wanted_objects: plan.wanted.len() as u64,
        ..Record::default()
    };
    let collected = admission::admit(request.store, &flight, &record, request.keep, &|| {
        say(Phase::Collecting, plan.held_bytes(), total, 0.0, 0)
    })?;

    let meter = Arc::new(Meter::default());
    let store = request.store.clone().observed(meter.clone());
    let counters = Counters::default();
    counters.held.store(plan.held_bytes(), Ordering::Relaxed);
    let on_plan = |plan: &FetchPlan| {
        counters.held.store(plan.held_bytes(), Ordering::Relaxed);
        counters
            .moved_at_plan
            .store(meter.moved.load(Ordering::Relaxed), Ordering::Relaxed);
    };
    let on_object = |object: &ObjectRef, _moved: u64, door: ObjectSource| {
        counters.landed_objects.fetch_add(1, Ordering::Relaxed);
        let bytes = match door {
            ObjectSource::Origin => &counters.origin_bytes,
            ObjectSource::Cache => &counters.cache_bytes,
        };
        bytes.fetch_add(object.length, Ordering::Relaxed);
    };
    let done = || {
        let moved = meter.moved.load(Ordering::Relaxed);
        let since_plan = moved.saturating_sub(counters.moved_at_plan.load(Ordering::Relaxed));
        (counters.held.load(Ordering::Relaxed) + since_plan).min(total)
    };
    say(Phase::Fetching, done(), total, 0.0, 1);

    let stop = (Mutex::new(false), Condvar::new());
    let pulled = std::thread::scope(|scope| {
        scope.spawn(|| {
            let (mut at, mut last) = (Instant::now(), meter.moved.load(Ordering::Relaxed));
            let mut stopped = stop.0.lock().unwrap();
            while !*stopped {
                stopped = stop.1.wait_timeout(stopped, tick).unwrap().0;
                let moved = meter.moved.load(Ordering::Relaxed);
                let rate = moved.saturating_sub(last) as f64 / at.elapsed().as_secs_f64();
                (at, last) = (Instant::now(), moved);
                let attempt = counters.attempt.load(Ordering::Relaxed);
                say(Phase::Fetching, done(), total, rate, attempt);
                let _ = flight.record(&Record {
                    bytes_done: done(),
                    landed_bytes: moved,
                    landed_objects: counters.landed_objects.load(Ordering::Relaxed),
                    ..record.clone()
                });
            }
        });
        let pulled = attempts(request, &store, &closure, &counters, &on_plan, &on_object);
        *stop.0.lock().unwrap() = true;
        stop.1.notify_all();
        pulled
    });
    let report = pulled?;
    record.bytes_done = total;
    say(
        Phase::Fetching,
        total,
        total,
        counters.origin_bytes.load(Ordering::Relaxed) as f64 / started.elapsed().as_secs_f64(),
        counters.attempt.load(Ordering::Relaxed),
    );
    Ok(Ensured {
        model: report.model,
        release: report.release,
        lane: report.lane,
        scope: report.scope,
        manifest: closure.manifest.clone(),
        bytes_total: total,
        bytes_held: plan.held_bytes(),
        bytes_fetched: counters.origin_bytes.load(Ordering::Relaxed),
        bytes_cached: counters.cache_bytes.load(Ordering::Relaxed),
        cache_written_bytes: report.cache_writes.bytes(),
        collected_bytes: collected,
        attempts: counters.attempt.load(Ordering::Relaxed),
        joined: flight.joined,
        seconds: started.elapsed().as_secs_f64(),
    })
}

/// Run `work`, calling `tick` once per `every` until it returns.
fn ticking<T>(every: Duration, tick: &(dyn Fn() + Sync), work: impl FnOnce() -> T) -> T {
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
        let out = work();
        *stop.0.lock().unwrap() = true;
        stop.1.notify_all();
        out
    })
}

/// Pull until done. A pull that failed on the link is re-attempted; one that landed nothing
/// waits a sample first. The ensure is `STALLED` only once no object has landed for the
/// ledger's own floor: a blip that spent one attempt's asks in milliseconds measured nothing.
fn attempts(
    request: &Request<'_>,
    store: &Store,
    closure: &transport::Closure,
    counters: &Counters,
    on_plan: &(dyn Fn(&FetchPlan) + Sync),
    on_object: &(dyn Fn(&ObjectRef, u64, ObjectSource) + Sync),
) -> Result<transport::PullReport> {
    let sample = Duration::from_secs_f64(request.sample_seconds);
    let floor = sample * transport::STILL_SAMPLES;
    let mut waits = transport::Ledger::with_resolution(sample);
    if let Some(cancellation) = &request.cancellation {
        waits = waits.with_cancellation(cancellation.clone());
    }
    let mut advanced = Instant::now();
    loop {
        let attempt = counters.attempt.fetch_add(1, Ordering::Relaxed) + 1;
        let landed = counters.landed_objects.load(Ordering::Relaxed);
        let mut pull = request.pull(store);
        pull.closure = Some(closure);
        pull.on_plan = Some(on_plan);
        pull.on_object = Some(on_object);
        match transport::pull(&pull) {
            Ok(report) => return Ok(report),
            Err(refusal) if request.cancelled() || !weather(refusal.code) => return Err(refusal),
            Err(_) if counters.landed_objects.load(Ordering::Relaxed) != landed => {
                advanced = Instant::now();
            }
            Err(refusal) if advanced.elapsed() > floor => {
                return Err(Refusal {
                    code: Code::STALLED,
                    detail: format!(
                        "no object landed for {:.1} s over {attempt} attempt(s), past the \
                         {:.1} s floor: {refusal}",
                        advanced.elapsed().as_secs_f64(),
                        floor.as_secs_f64()
                    ),
                })
            }
            Err(_) => waits.pause(sample, transport::Deadline::none())?,
        }
    }
}

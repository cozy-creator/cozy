//! The pull's liveness ledger and the caller's deadline — cozy-runtime `pull.py`'s
//! `Deadline`/`_Streams`, ported with their evidence (tfs-049).
//!
//! **No magic timeouts, and no unbounded silence either.** A deadline arrives from the
//! caller or there is none; nothing here invents a number of seconds to fail on. But a
//! presigned GET that accepts a connection and then stops sending held a pool slot until
//! the deadline expired, so one black-holed stream cost the whole pull. This ledger is the
//! answer, and its rule is MEASURED rather than invented: every chunk is counted where it
//! crosses into this process, the longest gap the pull has ACTUALLY shown — the origin's
//! own time-to-answer included — is remembered, and a stream silent for far longer than
//! that is stalled, closed, and its object re-asked, which the plan already made cost one
//! object. A slow link teaches the ledger its own pace and is left alone. The same rule,
//! lifted out for processes, is cozy-runtime's `internal/liveness.py`, and the constants
//! below are shared with it so a stream and a process are judged at the same resolution.
//!
//! The Python original watched from a sampling thread and woke blocked reads by shutting
//! the socket down. Here each stream samples ITSELF: its socket read timeout is the
//! sampling resolution — resolution, not a verdict — and on each empty tick it consults
//! the deadline and this ledger. Same rule, same granularity, no watchdog thread.
//!
//! **Two planes, one rule (tfs-106).** The paragraph above judges a STREAM by its own
//! chunks, which is the byte plane. It cannot judge the CONTROL plane, because a hub call
//! has no bytes of its own worth judging: a mint that takes a minute while sixteen streams
//! keep landing objects is a call, not a wedge. What makes a hub call a wedge is that the
//! TRANSFER it is serving has itself stopped — every stream parked on `Leases::minting`
//! behind one thread that will never return. So this ledger keeps a second measurement,
//! `still()`: what the whole pull has achieved and how long ago. A control-plane wait is
//! judged by THAT, never by its own duration, against a term (`stillness()`) measured the
//! same way and from the same evidence — the longest this pull has itself gone still and
//! recovered — and never shorter than the byte plane's own `patience()`. On 2026-09-08 two
//! paid H100s sat at exactly zero for fifteen minutes with neither plane complaining.

use crate::err::{refuse, Code, Result};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How many objects are in flight by default. A MEASURED number, not a taste: #566b swept
/// the same pods, the same bucket and the same presigned GETs and found the per-stream
/// rate flat and the aggregate saturating by 16 (49.8 MB/s at 1, 117.6 MB/s at 16, against
/// the 141 MB/s ceiling the pod's own transport probe reached). It is a parameter because
/// the right number is a property of the LINK, and this module cannot measure the caller's.
pub const STREAMS: usize = 16;

/// The most requests a pull keeps in flight, before the memory and descriptor bounds.
/// Measured on a pod: 8 MiB requests move the same 1.1-1.2 GB/s at 64, 128 and 256, so
/// the link is not what more buys. A cold object answers each MiB in its own time, and
/// 2 GiB of cold objects asked a MiB at a time was 90% home in 12.0 s at 128, 4.7 s at
/// 512 and 3.2 s at 1,024.
pub const PULL_STREAMS: usize = 512;

/// How often a waiting stream SAMPLES — the runtime's one sampling cadence, shared with
/// cozy-runtime `internal/liveness.py::SAMPLE_SECONDS`. Resolution, not a verdict: nothing
/// is ended because this many seconds passed.
pub const SAMPLE_SECONDS: f64 = 5.0;

/// The NOISE FLOOR, in samples, below which silence is not evidence of anything — shared
/// with `liveness.py::STILL_SAMPLES`. A pull whose chunks came back in milliseconds must
/// not condemn a stream that is merely a second slow.
pub const STILL_SAMPLES: u32 = 6;

/// How many times the pull's OWN worst measured gap a stream may be silent for before it
/// is stalled. This is the term that carries the verdict, and it is MEASURED: a link that
/// has shown 30 s between chunks — or an origin that took 30 s to start answering — earns
/// 240 s of patience, and one that has never paused earns the floor.
pub const STALL_FACTOR: f64 = 8.0;

/// How many times one object may be re-asked for. A grant is finite and the plan is
/// per-object resumable, so a retry costs one object, not the pull. A stall or a transient
/// object-store answer that survives this many attempts is the link, not a blip.
pub const FETCH_ATTEMPTS: u32 = 4;

/// A caller-supplied wall deadline, or none at all. `remaining()` is what every socket
/// operation is bounded by, so a slow hub and a slow object share one budget instead of
/// each inventing its own. `None` means the caller set no deadline and the pull blocks
/// until the transport itself gives up — which is honest: a module that substituted a
/// number here would be deciding, on the caller's behalf, that a large artifact on a slow
/// link is a failure.
#[derive(Debug, Clone, Copy)]
pub struct Deadline {
    at: Option<Instant>,
}

impl Deadline {
    pub fn none() -> Deadline {
        Deadline { at: None }
    }

    pub fn after_seconds(seconds: Option<f64>) -> Deadline {
        Deadline {
            at: seconds.map(|s| Instant::now() + Duration::from_secs_f64(s.clamp(0.0, 1e9))),
        }
    }

    /// What is LEFT, or the typed refusal once nothing is.
    pub fn remaining(&self) -> Result<Option<Duration>> {
        match self.at {
            None => Ok(None),
            Some(at) => {
                let now = Instant::now();
                if now >= at {
                    refuse(
                        Code::DEADLINE_EXCEEDED,
                        "the caller's deadline passed with objects still to move — re-run: \
                         a pull resumes per object and re-fetches only what is missing",
                    )
                } else {
                    Ok(Some(at - now))
                }
            }
        }
    }
}

/// What the WHOLE pull has achieved, and when. `moved` is every object byte that has
/// crossed into this process; `advanced` is the last instant this pull got anywhere at all
/// — a byte landed, or a hub call it was waiting on answered.
///
/// A hub answer counts as an advance and NOT as bytes, and both halves of that matter. It
/// counts, because a closure page that just came back is a pull that is getting somewhere
/// and the next page must be judged from there rather than from the start of the walk. It
/// is not bytes, because the figure a refusal quotes has to be what actually landed.
struct Progress {
    moved: u64,
    advanced: Instant,
    /// The longest this pull has ever gone between advances AND THEN ADVANCED AGAIN. It is
    /// the control plane's whole measured term, and it is kept apart from `worst_gap`
    /// deliberately: that one is one stream's chunk rhythm, this one is the pull's, and a
    /// pull legitimately goes still between objects for as long as its own store takes to
    /// admit one. Folding them would make each rule wrong in the other's units.
    worst_still: Duration,
}

/// What the pull has moved and how long it has been still. This is the CONTROL plane's
/// whole evidence, and it is the pair a refusal quotes: a number a reader can check against
/// the transfer's own report, and an interval they can check against `patience()`.
#[derive(Debug, Clone, Copy)]
pub struct Still {
    pub moved: u64,
    pub silent: Duration,
}

/// One caller's cooperative stop signal. Clones name the same pull intent;
/// a resumed operation uses a new token and retains existing Store bytes.
#[derive(Clone, Debug, Default)]
pub struct PullCancellation(Arc<AtomicBool>);

impl PullCancellation {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// The one liveness ledger a pull shares across its streams.
pub struct Ledger {
    caller_cancellation: Option<PullCancellation>,
    cancelled: Arc<AtomicBool>,
    sample: Duration,
    /// How often a wait on this ledger wakes to look. The sample, unless a request asked
    /// to notice its own stop sooner; it never changes a verdict.
    tick: Duration,
    worst_gap: Arc<Mutex<Duration>>,
    progress: Arc<Mutex<Progress>>,
    attempt: Option<Arc<Attempt>>,
    #[cfg(test)]
    clock: Option<Arc<dyn Fn() -> Instant + Send + Sync>>,
}

impl Ledger {
    pub fn new() -> Ledger {
        Ledger::with_resolution(Duration::from_secs_f64(SAMPLE_SECONDS))
    }

    /// The cadence is configuration — resolution, never a verdict — so a test may sample
    /// in milliseconds and judge by exactly the production rule.
    pub fn with_resolution(sample: Duration) -> Ledger {
        Ledger {
            caller_cancellation: None,
            cancelled: Arc::new(AtomicBool::new(false)),
            sample: sample.max(Duration::from_millis(1)),
            tick: sample.max(Duration::from_millis(1)),
            worst_gap: Arc::new(Mutex::new(Duration::ZERO)),
            progress: Arc::new(Mutex::new(Progress {
                moved: 0,
                advanced: Instant::now(),
                worst_still: Duration::ZERO,
            })),
            attempt: None,
            #[cfg(test)]
            clock: None,
        }
    }

    pub fn with_cancellation(mut self, cancellation: PullCancellation) -> Self {
        self.caller_cancellation = Some(cancellation);
        self
    }

    #[cfg(test)]
    pub(crate) fn with_clock(
        sample: Duration,
        clock: impl Fn() -> Instant + Send + Sync + 'static,
    ) -> Self {
        let mut ledger = Self::with_resolution(sample);
        ledger.progress.lock().unwrap().advanced = clock();
        ledger.clock = Some(Arc::new(clock));
        ledger
    }

    fn now(&self) -> Instant {
        #[cfg(test)]
        if let Some(clock) = &self.clock {
            return clock();
        }
        Instant::now()
    }

    /// A terminal object failure stops the whole pull, including already-open
    /// HTTP streams. Completed objects remain admitted for the next pull.
    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub(crate) fn check_running(&self) -> Result<()> {
        if self
            .caller_cancellation
            .as_ref()
            .is_some_and(PullCancellation::is_cancelled)
        {
            refuse(
                Code::TRANSFER_FAILED,
                "the pull was cancelled by its caller",
            )
        } else if self.cancelled.load(Ordering::Acquire) {
            refuse(
                Code::TRANSFER_FAILED,
                "the pull stopped after another object failed",
            )
        } else if self
            .attempt
            .as_ref()
            .is_some_and(|a| a.stopped.load(Ordering::Acquire))
        {
            refuse(Code::TRANSFER_FAILED, "this object attempt was stopped")
        } else {
            Ok(())
        }
    }

    pub fn sample(&self) -> Duration {
        self.sample
    }

    pub(super) fn tick(&self) -> Duration {
        self.tick
    }

    /// The smallest patience ever granted: `STILL_SAMPLES` intervals of the resolution.
    pub fn floor(&self) -> Duration {
        self.sample * STILL_SAMPLES
    }

    /// One observed gap between a stream's chunks — or an origin's time-to-first-answer,
    /// which is the first gap a stream has shown. A stream a caller has already judged
    /// stalled must not report here: counting post-verdict silence as the link's honest
    /// pace would let every stall raise the patience the next stall is judged by.
    pub fn taught(&self, gap: Duration) {
        let mut worst = self.worst_gap.lock().unwrap();
        if gap > *worst {
            *worst = gap;
        }
    }

    /// The verdict term: silence beyond BOTH the measured term and the floor is a stall.
    pub fn patience(&self) -> Duration {
        let worst = *self.worst_gap.lock().unwrap();
        worst.mul_f64(STALL_FACTOR).max(self.floor())
    }

    /// Object bytes that just crossed into this process. Counted where they arrive, on
    /// whichever of the pull's streams arrived first, so `still()` describes the pull and
    /// not whichever stream happens to ask.
    pub fn moved(&self, bytes: u64) {
        self.advance(bytes);
        if let Some(attempt) = &self.attempt {
            attempt.first.get_or_init(Instant::now);
            attempt.bytes.fetch_add(bytes, Ordering::Relaxed);
        }
    }

    /// This pull's ledger for one request: its own stop and byte count, and the cadence it
    /// wakes at to notice that stop.
    pub(super) fn for_request(&self, attempt: Arc<Attempt>, tick: Duration) -> Self {
        Self {
            caller_cancellation: self.caller_cancellation.clone(),
            cancelled: Arc::clone(&self.cancelled),
            sample: self.sample,
            tick,
            worst_gap: Arc::clone(&self.worst_gap),
            progress: Arc::clone(&self.progress),
            attempt: Some(attempt),
            #[cfg(test)]
            clock: self.clock.clone(),
        }
    }

    /// A call this pull was waiting on came back. It moved no artifact bytes and says so,
    /// but the pull did get somewhere, and the next wait must be measured from here.
    pub fn answered(&self) {
        self.advance(0);
    }

    /// The transfer starts NOW. Everything before it — opening the store, splitting the
    /// closure against what is already local, the disk-budget check — is local work whose
    /// duration this pull has just DEMONSTRATED, so it is taught rather than discarded: the
    /// first mint of the walk is then judged by a term that has seen how still this pull
    /// goes while it is doing something that is not moving bytes.
    pub fn transfer_begins(&self) {
        self.advance(0);
    }

    /// One advance of the whole pull, teaching the gap it just closed.
    ///
    /// EVERY closed gap is evidence, including a long one. A gap that ended in an advance
    /// is by definition not a stall — the pull recovered from it — and it is precisely the
    /// pull saying how still it can legitimately be. Silence that never ends teaches
    /// nothing, because nothing arrives to teach it, which is the same reason a condemned
    /// stream cannot raise the bar the next stream is judged by.
    fn advance(&self, bytes: u64) {
        let mut progress = self.progress.lock().unwrap();
        let now = self.now();
        let gap = now.saturating_duration_since(progress.advanced);
        if gap > progress.worst_still {
            progress.worst_still = gap;
        }
        progress.moved = progress.moved.saturating_add(bytes);
        progress.advanced = now;
    }

    /// The CONTROL plane's verdict term: the longest this pull has ever gone between
    /// advances, times the same factor the byte plane uses — and never less patient than
    /// the byte plane itself. A coarser plane that condemned sooner than the finer one
    /// beneath it would kill transfers the finer one had already judged healthy.
    pub fn stillness(&self) -> Duration {
        let worst = self.progress.lock().unwrap().worst_still;
        worst.mul_f64(STALL_FACTOR).max(self.patience())
    }

    /// What a retryable answer costs before the re-ask: the origin's own stated wait,
    /// which is not a failed attempt, or — when it states none — this pull's patience,
    /// which is (`true`). The hub's rule, applied to every origin.
    pub(crate) fn retry_wait(&self, stated: Option<Duration>) -> (Duration, bool) {
        // A wait no clock can hold is no statement (and `Instant + wait` would panic).
        let representable = |wait: &Duration| Instant::now().checked_add(*wait).is_some();
        match stated.filter(|wait| !wait.is_zero() && representable(wait)) {
            Some(wait) => (wait, false),
            None => (self.patience(), true),
        }
    }

    /// Sleep `wait` in sampling ticks; cancellation and the caller's deadline end it.
    pub(crate) fn pause(&self, wait: Duration, deadline: Deadline) -> Result<()> {
        let until = Instant::now()
            .checked_add(wait)
            .unwrap_or_else(|| Instant::now() + self.patience());
        loop {
            self.check_running()?;
            let now = Instant::now();
            if now >= until {
                return Ok(());
            }
            let mut tick = (until - now).min(self.sample);
            if let Some(left) = deadline.remaining()? {
                tick = tick.min(left);
            }
            std::thread::sleep(tick);
        }
    }

    /// What this pull has moved and how long it has been still.
    pub fn still(&self) -> Still {
        let progress = self.progress.lock().unwrap();
        Still {
            moved: progress.moved,
            silent: self.now().saturating_duration_since(progress.advanced),
        }
    }
}

impl Default for Ledger {
    fn default() -> Ledger {
        Ledger::new()
    }
}

/// One request's own stop, when its first byte came, and its byte count. Dropping a
/// request never stops the pull.
#[derive(Default)]
pub(super) struct Attempt {
    pub stopped: AtomicBool,
    pub first: std::sync::OnceLock<Instant>,
    pub bytes: AtomicU64,
}

//! The machine's idle clock. A rental releases itself after 15 minutes with no job queued for it
//! or running on it, used or not (owner ruling 2026-10-10); every run asked of it is a job (a
//! call, a job, a warm-up, an upload); an explicit keepalive restarts that clock once. The clock
//! is durable state, never a sample held in memory: the journal's last job end and the ledger's
//! idle start (this rental's first boot, or a keepalive), so restarts and updates neither shorten
//! nor lose it. This computer's machine counts the same clock from its start: after 5 minutes its
//! executors exit and give back every byte they held, after 15 it exits itself (Calcifer starts
//! it again when something uses it), unless another computer's controller is authorized to use
//! it. A client following a run or a log, reading an output or uploading an object, holds either
//! clock. Also the Hub lease of authorized keys.
use super::hub::{Hub, Refusal};
use crate::api::auth::Keys;
use std::{
    io,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use protocol::error::Error;

/// The idle window, `RentalIdleTimeoutSeconds` in the CLI.
pub const IDLE_GRACE_MS: i64 = 900_000;
/// This computer's machine's executors exit after this part of its idle window (5 minutes).
const EXECUTORS_PART: i64 = 3;

/// The journal's jobs: one queued or running, and when the last one stopped.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Jobs {
    pub working: bool,
    pub ended_ms: i64,
}

#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
struct Ledger {
    /// When the idle clock last started without a job: first boot, or a keepalive.
    idle_since_ms: i64,
    released: bool,
}

struct State {
    ledger: Ledger,
    jobs: Jobs,
    /// The admission epoch `jobs` was read after: a later admission may have queued a job.
    sampled: u64,
    /// An update is activating: no new work is admitted until it exits or fails.
    activating: bool,
}

pub struct Lifecycle {
    rental: bool,
    /// The idle window: `IDLE_GRACE_MS` unless the grant names another.
    grace: i64,
    path: PathBuf,
    state: Mutex<State>,
    admissions: AtomicUsize,
    epoch: AtomicU64,
    /// Clients following a run's events or a log now.
    following: AtomicUsize,
    /// This computer's machine stays up: another computer's controller is authorized to use it.
    kept: AtomicBool,
    /// When an output read or an object upload last moved bytes (unix ms).
    transferred: AtomicI64,
    /// Called when an activation starts or ends without its restart (the runs it holds change),
    /// and when an admission or a follower lets go: the idle clock may start.
    changed: std::sync::OnceLock<Box<dyn Fn() + Send + Sync>>,
}

/// Holds the release while a call that may queue a job is in flight.
pub struct Admission(Arc<Lifecycle>);

/// Holds the release while a client follows a run's events or a log.
pub struct Follow(Arc<Lifecycle>);
impl Drop for Follow {
    fn drop(&mut self) {
        self.0.following.fetch_sub(1, Ordering::AcqRel);
        self.0.changed();
    }
}

/// Closes admission while an update waits for admitted work to drain and replaces the
/// service. Dropping it (a failed activation) reopens admission.
pub struct Activation(Arc<Lifecycle>);
impl Drop for Activation {
    fn drop(&mut self) {
        self.0.state.lock().unwrap().activating = false;
        self.0.changed();
    }
}
impl Drop for Admission {
    fn drop(&mut self) {
        self.0.admissions.fetch_sub(1, Ordering::AcqRel);
        self.0.changed();
    }
}

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

impl Lifecycle {
    /// `fresh` starts this root's first idle clock; a rental's restart keeps the persisted
    /// ledger. Each start of this computer's machine is a use: its clock starts again, and an
    /// exit it took is over.
    pub fn open(path: PathBuf, rental: bool, fresh: bool, grace: i64) -> io::Result<Arc<Self>> {
        let fresh = fresh || !rental;
        let mut ledger: Ledger = match std::fs::read(&path) {
            Ok(raw) if !fresh => serde_json::from_slice(&raw)
                .map_err(|e| io::Error::other(format!("the idle ledger is unreadable: {e}")))?,
            Ok(_) => Ledger::default(),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ledger::default(),
            Err(e) => return Err(e),
        };
        if ledger.idle_since_ms == 0 {
            ledger.idle_since_ms = now_ms();
        }
        let lifecycle = Arc::new(Self {
            rental,
            grace,
            path,
            state: Mutex::new(State { ledger, jobs: Jobs::default(), sampled: 0, activating: false }),
            admissions: AtomicUsize::new(0),
            epoch: AtomicU64::new(0),
            following: AtomicUsize::new(0),
            kept: AtomicBool::new(false),
            transferred: AtomicI64::new(0),
            changed: Default::default(),
        });
        lifecycle.save(&lifecycle.state.lock().unwrap().ledger)?;
        Ok(lifecycle)
    }

    fn save(&self, ledger: &Ledger) -> io::Result<()> {
        super::identity::write_atomic(&self.path, &serde_json::to_vec(ledger)?, 0o600)
    }

    /// When the idle clock runs out: 15 minutes after the latest of its start, the last job's end
    /// and the last bytes an output read or an upload moved.
    fn idle_end(&self, state: &State) -> i64 {
        let used = state.jobs.ended_ms.max(self.transferred.load(Ordering::Acquire));
        state.ledger.idle_since_ms.max(used) + self.grace
    }

    /// An output read or an object upload moved bytes: the rental is in use, and its idle clock
    /// starts again now. A transfer that stalls moves nothing, so it never holds the rental past one
    /// window.
    pub fn transferred(&self) {
        self.transferred.fetch_max(now_ms(), Ordering::AcqRel);
    }

    fn due(&self, state: &State, now: i64) -> bool {
        state.ledger.released || (!state.jobs.working && now >= self.idle_end(state))
    }

    /// A job is queued or running, a call that may queue one is in flight (on this computer, a
    /// follower too: a rental bills, so only jobs hold it), or a call was admitted since the
    /// journal was last read.
    fn working(&self, state: &State) -> bool {
        state.jobs.working
            || self.admitted() > 0
            || (!self.rental && self.following.load(Ordering::Acquire) > 0)
            || state.sampled != self.epoch()
    }

    /// The deadline a caller is told: none (0) while working.
    fn deadline(&self, state: &State) -> i64 {
        match self.working(state) && !state.ledger.released {
            true => 0,
            false => self.idle_end(state),
        }
    }

    /// When the idle clock started, as last read: none while working, activating or released.
    fn idle_since(&self) -> Option<i64> {
        let state = self.state.lock().unwrap();
        let idle = !self.working(&state) && !state.activating && !state.ledger.released;
        idle.then(|| self.idle_end(&state) - self.grace)
    }

    /// A client follows a run's events or a log: the machine is in use until it lets go.
    pub fn follow(self: &Arc<Self>) -> Follow {
        self.following.fetch_add(1, Ordering::AcqRel);
        Follow(self.clone())
    }

    pub fn admit(self: &Arc<Self>) -> Result<Admission, Error> {
        let state = self.state.lock().unwrap();
        if state.activating {
            return Err(Error::unavailable(
                "machine_updating",
                "the machine is activating an update; submit again once it is back",
            ));
        }
        // This computer's machine past its deadline but not yet exiting takes the call.
        if state.ledger.released || (self.rental && self.due(&state, now_ms())) {
            return Err(Error::unavailable(
                "machine_released",
                "this machine released itself after 15 minutes idle",
            ));
        }
        self.admissions.fetch_add(1, Ordering::AcqRel);
        self.epoch.fetch_add(1, Ordering::AcqRel);
        Ok(Admission(self.clone()))
    }

    /// Closes admission under the lock `admit` takes: a call is either admitted (and counted)
    /// before activation or refused after it, never accepted while the service exits.
    pub fn activate(self: &Arc<Self>) -> Result<Activation, Error> {
        let mut state = self.state.lock().unwrap();
        if state.activating || state.ledger.released {
            return Err(Error::unavailable(
                "machine_updating",
                "the machine is already activating or released",
            ));
        }
        state.activating = true;
        drop(state);
        self.changed();
        Ok(Activation(self.clone()))
    }

    /// An update is activating: admission is closed until it exits or fails.
    pub fn activating(&self) -> bool {
        self.state.lock().unwrap().activating
    }
    pub fn on_change(&self, changed: impl Fn() + Send + Sync + 'static) {
        let _ = self.changed.set(Box::new(changed));
    }
    fn changed(&self) {
        if let Some(changed) = self.changed.get() {
            changed();
        }
    }

    /// The admission epoch, read before the journal is.
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// The journal's jobs as read after admission epoch `epoch`.
    pub fn observe(&self, jobs: Jobs, epoch: u64) {
        let mut state = self.state.lock().unwrap();
        state.jobs = jobs;
        state.sampled = epoch;
    }

    /// The last job end this lifecycle has read.
    pub fn ended_ms(&self) -> i64 {
        self.state.lock().unwrap().jobs.ended_ms
    }

    /// `POST /v1/keepalive`: restarts the idle clock once; answers the new deadline.
    pub fn renew(&self) -> Result<i64, Error> {
        if !self.rental {
            return Ok(0);
        }
        let now = now_ms();
        let mut state = self.state.lock().unwrap();
        if self.due(&state, now) {
            return Err(Error::unavailable(
                "machine_released",
                "this rental's idle release is already due or committed",
            ));
        }
        let mut ledger = state.ledger.clone();
        ledger.idle_since_ms = ledger.idle_since_ms.max(now);
        self.save(&ledger).map_err(|e| {
            Error::unavailable("idle_ledger_unwritable", format!("the idle ledger could not be written: {e}"))
        })?;
        state.ledger = ledger;
        Ok(self.deadline(&state))
    }

    /// Calls that may start work, in flight now.
    pub fn admitted(&self) -> usize {
        self.admissions.load(Ordering::Acquire)
    }

    /// When this machine releases itself or exits if no job comes first. 0: no deadline,
    /// because it is working, or because another computer's controller keeps it up.
    pub fn deadline_ms(&self) -> i64 {
        match self.kept.load(Ordering::Acquire) {
            true => 0,
            false => self.deadline(&self.state.lock().unwrap()),
        }
    }

    pub fn released(&self) -> bool {
        self.state.lock().unwrap().ledger.released
    }

    /// Work came between this computer's machine's claim and its stop: it serves on.
    fn reopen(&self) {
        self.state.lock().unwrap().ledger.released = false;
    }

    /// Takes the release once the idle clock ran out on a current reading. Irreversible but for
    /// `reopen`.
    fn claim(&self) -> io::Result<bool> {
        let mut state = self.state.lock().unwrap();
        if state.ledger.released {
            return Ok(true);
        }
        if state.activating || self.working(&state) || !self.due(&state, now_ms()) {
            return Ok(false);
        }
        let mut ledger = state.ledger.clone();
        ledger.released = true;
        self.save(&ledger)?;
        state.ledger = ledger;
        Ok(true)
    }
}

/// Keeps the Hub's authorized keys: asked again at half the lease it names. While the Hub is
/// unreachable or not yet ready the keys stay as they were, and it is asked again after 1 s,
/// doubling to 5 s (the Hub reads a pod's readiness every 5 s).
pub async fn keep_authority(hub: Arc<Hub>, keys: Keys) {
    const UNANSWERED: (Duration, Duration) = (Duration::from_secs(1), Duration::from_secs(5));
    let mut unanswered = UNANSWERED.0;
    loop {
        let delay = match hub.authorized_keys().await {
            Ok((current, lease)) => {
                keys.renew(current);
                unanswered = UNANSWERED.0;
                lease / 2
            }
            Err(Refusal::Denied) => {
                tracing::error!(
                    "the Hub denied rental authority; no new control is admitted"
                );
                keys.revoke();
                Duration::from_secs(1)
            }
            Err(Refusal::Transport(_)) => {
                let delay = unanswered;
                unanswered = (unanswered * 2).min(UNANSWERED.1);
                delay
            }
        };
        tokio::time::sleep(delay).await;
    }
}

/// Reads the journal's jobs each second and releases the rental once 15 minutes passed with
/// none. Returns when the Hub accepted the release, or when the rental ended itself: idle past
/// its deadline with no answer from the Hub through that whole idle window, the Hub cannot end
/// it, so it ends itself at its provider (`provider`, else by stopping) and billing stops.
pub async fn release_when_idle(
    lifecycle: Arc<Lifecycle>,
    hub: Arc<Hub>,
    provider: Option<super::provider::ProviderSelf>,
    jobs: impl Fn(i64) -> io::Result<Jobs>,
) {
    let mut next_ask = 0;
    loop {
        let epoch = lifecycle.epoch();
        match jobs(lifecycle.ended_ms()) {
            Ok(read) => lifecycle.observe(read, epoch),
            Err(error) => {
                tracing::warn!("cannot read this rental's jobs: {error}");
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            }
        }
        let now = now_ms();
        if now >= next_ask {
            match lifecycle.claim() {
                Ok(true) => match hub.release().await {
                    Ok(()) => {
                        tracing::info!("15 minutes without a job; Tensorhub accepted the release");
                        return;
                    }
                    Err(error) => {
                        tracing::warn!("idle release not accepted; retrying without extending the deadline: {error}");
                        next_ask = now + 5_000;
                        if unheard(hub.contact_ms(), lifecycle.deadline_ms(), lifecycle.grace) {
                            end_without_hub(provider.as_ref()).await;
                            return;
                        }
                    }
                },
                Ok(false) => (),
                Err(error) => {
                    tracing::warn!("idle release could not be recorded: {error}")
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

/// This computer's machine through its idle clock: after a third of the window with no work its
/// executors exit (`release`), after the whole window the machine stops (`stop`, false when work
/// came meanwhile) and this process exits. While `kept` (another computer's controller is
/// authorized to use it) only the executors go. It sleeps until activity or the next deadline.
pub fn exit_when_idle(
    lifecycle: Arc<Lifecycle>,
    engine: Arc<crate::execution::Engine>,
    release: impl Fn(),
    stop: impl Fn() -> io::Result<bool>,
    kept: impl Fn() -> bool,
) {
    let mut released = None;
    loop {
        let (seen, epoch) = (engine.activity_epoch(), lifecycle.epoch());
        match engine.jobs(lifecycle.ended_ms()) {
            Ok((working, ended_ms)) => lifecycle.observe(Jobs { working, ended_ms }, epoch),
            Err(error) => tracing::warn!("cannot read this machine's jobs: {error}"),
        }
        let Some(since) = lifecycle.idle_since() else {
            engine.wait_activity(seen, None);
            continue;
        };
        let (now, grace) = (now_ms(), lifecycle.grace);
        if now >= since + grace / EXECUTORS_PART && released != Some(since) {
            tracing::info!("idle {} s: its executors end and give back what they hold", (now - since) / 1000);
            release();
            released = Some(since);
        }
        let held = kept();
        lifecycle.kept.store(held, Ordering::Release);
        if now >= since + grace && !held {
            match lifecycle.claim().and_then(|claimed| Ok(claimed && stop()?)) {
                Ok(true) => {
                    tracing::info!("idle {} s: this machine exits until it is used", (now - since) / 1000);
                    std::process::exit(0);
                }
                Ok(false) => lifecycle.reopen(),
                Err(error) => tracing::warn!("this machine's idle exit: {error}"),
            }
        }
        // Kept up past its window, it reads its authorization again a window later.
        let next = match released == Some(since) {
            false => since + grace / EXECUTORS_PART,
            true if now < since + grace => since + grace,
            true => now + grace,
        };
        let wait = (next - now).max(1000);
        engine.wait_activity(seen, Some(Duration::from_millis(wait as u64)));
    }
}

/// The Hub has not answered since this idle window began: it cannot hear the release.
fn unheard(contact_ms: i64, deadline_ms: i64, grace: i64) -> bool {
    contact_ms < deadline_ms - grace
}

async fn end_without_hub(provider: Option<&super::provider::ProviderSelf>) {
    tracing::warn!("idle past the deadline with no answer from Tensorhub through the whole idle window; ending this rental itself");
    match provider {
        Some(provider) => match provider.end().await {
            Ok(()) => tracing::info!("the provider accepted this pod's end"),
            Err(error) => {
                tracing::error!("the provider did not end this pod ({error}); stopping")
            }
        },
        None => tracing::error!("no provider credential; stopping"),
    }
}

//! How many objects a pull keeps in flight, decided by what the link answers.
//!
//! A fixed 16 was measured on a pod capped at 141 MB/s, where 16 saturated the link. On an
//! uncapped pod the aggregate is 16 × one GET's rate: the same AP-IN-1 datacenter moved the
//! H3 lane at 378-400 MB/s on 2026-09-08 and at 112 MB/s on 2026-09-27 (run 1225), same
//! client, same bucket. When one GET is slow, only more GETs buy bytes.
//!
//! The walk starts at `start`, and after each sampling window doubles the streams it
//! admits while doing so bought at least `GAIN` more aggregate goodput. A doubling that
//! bought less settles the walk; one that bought less than nothing returns to the level
//! before it. Nothing is ended or failed here: this only admits workers to the next object.
//!
//! Two signals halve the level and stop its growth for the rest of the walk: a stream the
//! ledger judged stalled while the level stood, and a full descriptor table (`shrink`).
//! Both say the pull is asking more of the link or the process than either will give.
use super::ledger::Ledger;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// The most objects a pull may keep in flight, before the memory bound.
pub const PULL_STREAMS: usize = 128;

/// The aggregate-goodput ratio a doubling must buy to justify the next one. Below it the
/// link, the disk or the origin is the limit and more streams only add partial objects.
pub const GAIN: f64 = 1.2;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Level {
    pub streams: usize,
    /// Aggregate object bytes per second landed while this level was admitted.
    pub bytes_per_second: f64,
}

struct State {
    limit: usize,
    active: usize,
    settled: bool,
    /// Times the level was halved on backpressure.
    backoffs: u32,
    stalls_seen: u64,
    window_started: Instant,
    moved_at_window: u64,
    history: Vec<Level>,
}

pub(super) struct Pace {
    ceiling: usize,
    state: Mutex<State>,
    /// Workers waiting for a slot.
    admitted: Condvar,
    /// The steering thread, woken early only when a worker leaves.
    wake: Condvar,
    live: AtomicUsize,
}

pub(super) struct Permit<'a>(&'a Pace);

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        self.0.state.lock().unwrap().active -= 1;
        self.0.admitted.notify_one();
    }
}

impl Pace {
    /// `workers` threads will call `admit`, each calling `worker_finished` as it leaves.
    pub fn new(start: usize, ceiling: usize, workers: usize, ledger: &Ledger) -> Self {
        let ceiling = ceiling.max(1);
        let limit = start.clamp(1, ceiling);
        Self {
            ceiling,
            state: Mutex::new(State {
                limit,
                active: 0,
                settled: limit == ceiling,
                backoffs: 0,
                stalls_seen: ledger.stalls(),
                window_started: Instant::now(),
                moved_at_window: ledger.still().moved,
                history: Vec::new(),
            }),
            admitted: Condvar::new(),
            wake: Condvar::new(),
            live: AtomicUsize::new(workers),
        }
    }

    /// Streams admitted now.
    pub fn limit(&self) -> usize {
        self.state.lock().unwrap().limit
    }

    /// The level the walk ended at, every level it measured on the way, and how many
    /// times backpressure halved it.
    pub fn outcome(&self) -> (usize, Vec<Level>, u32) {
        let state = self.state.lock().unwrap();
        (state.limit, state.history.clone(), state.backoffs)
    }

    /// Times backpressure has halved the level so far. A caller reads it before an attempt
    /// and hands it to `back_off`, so one episode that fails many streams at once halves
    /// the level once, not once per stream.
    pub fn backoffs(&self) -> u32 {
        self.state.lock().unwrap().backoffs
    }

    /// Halve the level unless it was already halved since `seen`.
    pub fn back_off(&self, seen: u32) {
        if self.state.lock().unwrap().backoffs == seen {
            self.shrink();
        }
    }

    /// Halve the level and stop growing it. `false` when it is already one stream: there is
    /// nothing left to give back.
    pub fn shrink(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        state.settled = true;
        if state.limit <= 1 {
            return false;
        }
        state.limit /= 2;
        state.backoffs += 1;
        true
    }

    pub fn worker_finished(&self) {
        self.live.fetch_sub(1, Ordering::AcqRel);
        self.wake.notify_all();
    }

    /// The walk is stopping: release every waiting worker and the steering thread now.
    pub fn halt(&self) {
        self.admitted.notify_all();
        self.wake.notify_all();
    }

    /// Wait for a stream slot. `None` once the walk is stopping; the caller owns why.
    pub fn admit(&self, stop: &AtomicBool, ledger: &Ledger) -> Option<Permit<'_>> {
        let mut state = self.state.lock().unwrap();
        while state.active >= state.limit {
            if stop.load(Ordering::Relaxed) || ledger.check_running().is_err() {
                return None;
            }
            state = self
                .admitted
                .wait_timeout(state, ledger.sample())
                .unwrap()
                .0;
        }
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        state.active += 1;
        Some(Permit(self))
    }

    /// Measure one window and move the level. Runs until the walk's workers are gone.
    /// `unclaimed` is how many objects no worker has taken yet: growth past it buys nothing.
    pub fn steer(&self, ledger: &Ledger, stop: &AtomicBool, unclaimed: &dyn Fn() -> usize) {
        let tick = ledger.sample();
        loop {
            let mut state = self.state.lock().unwrap();
            loop {
                if stop.load(Ordering::Relaxed) || self.live.load(Ordering::Acquire) == 0 {
                    return;
                }
                let waited = state.window_started.elapsed();
                if waited >= tick {
                    break;
                }
                state = self.wake.wait_timeout(state, tick - waited).unwrap().0;
            }
            let stalls = ledger.stalls();
            let stalled = stalls > state.stalls_seen;
            state.stalls_seen = stalls;
            if state.settled && !stalled {
                state.window_started = Instant::now();
                continue;
            }
            drop(state);
            if stalled {
                self.shrink();
                self.state.lock().unwrap().window_started = Instant::now();
                continue;
            }
            self.observe(ledger.still().moved, Instant::now(), unclaimed());
        }
    }

    fn observe(&self, moved: u64, now: Instant, unclaimed: usize) {
        let mut state = self.state.lock().unwrap();
        let elapsed = now.saturating_duration_since(state.window_started);
        if state.settled || elapsed < Duration::from_millis(1) {
            return;
        }
        let rate = moved.saturating_sub(state.moved_at_window) as f64 / elapsed.as_secs_f64();
        let level = Level {
            streams: state.limit,
            bytes_per_second: rate,
        };
        let previous = state.history.last().copied();
        state.history.push(level);
        state.window_started = now;
        state.moved_at_window = moved;
        match previous {
            Some(before) if rate < before.bytes_per_second => {
                state.limit = before.streams;
                state.settled = true;
            }
            Some(before) if rate < before.bytes_per_second * GAIN => state.settled = true,
            _ if state.limit >= self.ceiling || unclaimed <= state.limit => state.settled = true,
            _ => {
                state.limit = (state.limit * 2).min(self.ceiling);
                drop(state);
                self.admitted.notify_all();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paced(start: usize, ceiling: usize) -> (Pace, Instant) {
        let pace = Pace::new(start, ceiling, 1, &Ledger::new());
        let at = pace.state.lock().unwrap().window_started;
        (pace, at)
    }

    #[test]
    fn doubles_while_goodput_follows_and_settles_when_it_stops() {
        let (pace, t0) = paced(16, 128);
        let second = Duration::from_secs(1);
        pace.observe(100, t0 + second, 1000);
        assert_eq!(pace.limit(), 32);
        pace.observe(300, t0 + second * 2, 1000);
        assert_eq!(pace.limit(), 64, "2x the goodput earns the next doubling");
        pace.observe(530, t0 + second * 3, 1000);
        assert_eq!(pace.limit(), 64, "1.15x does not");
        pace.observe(10_000, t0 + second * 4, 1000);
        let (level, history, _) = pace.outcome();
        assert_eq!(level, 64, "a settled walk stays settled");
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn a_doubling_that_loses_goodput_returns_to_the_level_before_it() {
        let (pace, t0) = paced(16, 128);
        let second = Duration::from_secs(1);
        pace.observe(100, t0 + second, 1000);
        pace.observe(180, t0 + second * 2, 1000);
        assert_eq!(pace.limit(), 16);
    }

    #[test]
    fn growth_stops_at_the_ceiling_and_at_the_work_left() {
        let (pace, t0) = paced(16, 32);
        pace.observe(100, t0 + Duration::from_secs(1), 1000);
        pace.observe(300, t0 + Duration::from_secs(2), 1000);
        assert_eq!(pace.limit(), 32);
        assert!(pace.state.lock().unwrap().settled);

        let (few, t0) = paced(16, 128);
        few.observe(100, t0 + Duration::from_secs(1), 10);
        assert_eq!(
            few.limit(),
            16,
            "ten unclaimed objects cannot use 32 streams"
        );
    }

    #[test]
    fn a_fixed_request_never_probes() {
        let (fixed, t0) = paced(16, 16);
        fixed.observe(100, t0 + Duration::from_secs(1), 1000);
        assert_eq!(fixed.limit(), 16);
        assert!(fixed.outcome().1.is_empty());
    }
}

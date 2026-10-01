//! Hedged recovery of slow object streams. An object's original GET may race one hedge
//! on the walk's idle lanes; the first verified admission wins and stops the other. A
//! hedge is earned by measurement, never by a timer: a stream is slow when,
//! at its own measured network rate, what it has left would take more than
//! `STALL_FACTOR` times the pull's best completed transfer of a comparable object.
//!
//! A large object is hedged as ranged parts on its share of the idle lanes, so a link
//! whose individual answers are slow still fills; a small one as one fresh whole GET.
//! Lanes are re-lent every sample: as the walk drains, the lanes it frees go to the
//! ranged hedges still on the wire, so none sits idle while their bytes remain.
//! Run 2322: R2 answered eight 64 MiB objects application-limited at 6-50 KB/s while the
//! rest of the pull moved 400 MB/s, and nothing re-asked them for fifteen minutes.
use super::{
    decisions,
    ledger::{Attempt, AttemptData, Deadline, Ledger, FETCH_ATTEMPTS, STALL_FACTOR, STILL_SAMPLES},
    pull::{fetch_with_attempts, shared_client, Fetched},
    ranged, CredentialProvider, SourcePolicy,
};
use crate::{
    err::Result,
    fetch::{DeliveryGrant, Destination},
    store::Store,
};
use std::{
    collections::{BTreeMap, VecDeque},
    sync::{
        atomic::{AtomicU32, AtomicUsize, Ordering},
        mpsc, Arc, Mutex,
    },
    time::{Duration, Instant},
};

/// What the walk can lend a hedge right now: the level it admits, the objects it has in
/// flight, and whether every object has been claimed.
#[derive(Clone, Copy, Debug)]
pub(super) struct Lanes {
    pub level: usize,
    pub busy: usize,
    pub tail: bool,
}

struct Reference {
    length: u64,
    elapsed: Duration,
}

#[derive(Default)]
pub(super) struct Recovery {
    references: Mutex<BTreeMap<String, Reference>>,
    /// Hedges launched and won across the walk, for its one summary line.
    pub hedges: AtomicUsize,
    pub won: AtomicUsize,
    /// Lanes hedges hold across the whole walk.
    racing: AtomicUsize,
    /// Ranged hedges on the wire, which share the lanes the walk is not using.
    ranged: AtomicUsize,
}

/// A hold's lanes once its hedge has returned: nothing more may be lent to it.
const CLOSED: usize = usize::MAX;

/// Lanes one hedge holds, returned however its thread leaves.
struct Hold<'a> {
    recovery: &'a Recovery,
    lanes: Arc<AtomicUsize>,
    ranged: bool,
}

impl Drop for Hold<'_> {
    fn drop(&mut self) {
        let lanes = self.lanes.swap(CLOSED, Ordering::AcqRel);
        self.recovery.racing.fetch_sub(lanes, Ordering::AcqRel);
        if self.ranged {
            self.recovery.ranged.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

struct Racer {
    state: Arc<Attempt>,
    lanes: Arc<AtomicUsize>,
    ranged: bool,
    live: bool,
    window: Window,
}

/// The most lanes a ranged copy of `length` bytes can use: one per part it can be cut into.
fn parts(length: u64) -> usize {
    (length / ranged::HEDGE_PART).max(2) as usize
}

impl Recovery {
    fn learn(&self, data: &AttemptData, length: u64, elapsed: Duration) {
        if data.origin.is_empty() || elapsed.is_zero() {
            return;
        }
        // A complete, verified transfer span includes startup, hashing and local
        // admission. Socket-wait time alone would invent enormous reference rates.
        let mut references = self.references.lock().unwrap();
        let reference = references
            .entry(data.origin.clone())
            .or_insert(Reference { length, elapsed });
        if length > reference.length {
            // Keep size and duration from the same evidence. Slow tiny metadata
            // must not become the reference for a later large-object sample.
            *reference = Reference { length, elapsed };
        } else if length == reference.length {
            reference.elapsed = reference.elapsed.min(elapsed);
        }
    }

    /// Whether this stream's measured remaining time dwarfs what a replacement has been
    /// shown to cost. Without a comparable completed transfer the basis is the time the
    /// stream has itself already spent waiting.
    fn judge(&self, data: &AttemptData, window: &Window, length: u64, floor: Duration) -> Verdict {
        if !data.waiting || data.bytes >= length {
            return Verdict::Unmeasured;
        }
        let Some(first) = window.0.front() else {
            return Verdict::Unmeasured;
        };
        let waited = data.waited.saturating_sub(first.waited);
        let bytes = data.bytes.saturating_sub(first.bytes);
        if waited < floor || bytes == 0 {
            return Verdict::Unmeasured; // silence has its own policy; local work is not network delay
        }
        let left = (length - data.bytes) as f64 * waited.as_secs_f64() / bytes as f64;
        let references = self.references.lock().unwrap();
        // Fast small metadata cannot qualify a large-body retry.
        match references.get(&data.origin).filter(|r| r.length >= length) {
            Some(reference) if left > STALL_FACTOR * reference.elapsed.as_secs_f64() => {
                Verdict::Slow
            }
            Some(_) => Verdict::Healthy,
            None if left > STALL_FACTOR * data.waited.as_secs_f64() => Verdict::Unreferenced,
            None => Verdict::Healthy,
        }
    }

    /// Reserve this object's share of the idle lanes, if the disk has room for the copy.
    /// A stream measured slow against a comparable transfer is barely using its own lane,
    /// so that lane counts as idle for its hedge even when the walk has none to spare. A
    /// ranged hedge always gets two lanes, so one can re-ask a part the other is stuck on.
    /// The caller hands the count to a `Hold`, which returns it.
    fn reserve(
        &self,
        lanes: Lanes,
        length: u64,
        store: &Store,
        whole: bool,
        own: bool,
    ) -> Option<usize> {
        // The original's object beside the hedge's copies: a ranged hedge assembles in
        // scratch before its door writes.
        let copies = if whole { 2 } else { 3 };
        if store.admits(length.saturating_mul(copies)).is_err() {
            return None;
        }
        let share = (lanes.level / lanes.busy.max(1)).max(1);
        let want = if whole { 1 } else { share.min(parts(length)) };
        let taken = self.borrow(lanes, want, own);
        if taken == 0 {
            return None;
        }
        if !whole && taken < 2 {
            self.racing.fetch_add(2 - taken, Ordering::AcqRel);
            return Some(2);
        }
        Some(taken)
    }

    /// Up to `want` of the lanes the walk and every hedge leave idle.
    fn borrow(&self, lanes: Lanes, want: usize, own: bool) -> usize {
        loop {
            let racing = self.racing.load(Ordering::Acquire);
            let spare = (lanes.level + usize::from(own)).saturating_sub(lanes.busy + racing);
            let take = want.min(spare);
            if take == 0 {
                return 0;
            }
            if self
                .racing
                .compare_exchange(racing, racing + take, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return take;
            }
        }
    }

    /// Raise a running ranged hedge to its share of the lanes the walk is not using, split
    /// evenly over every ranged hedge on the wire. Asked every sample, so the lanes a
    /// draining walk frees reach the bytes that remain. Dietfried: hedges fired with two
    /// lanes while 128 were busy kept two with 125 idle, and the last cold objects took up
    /// to twelve minutes at about 1 MB/s. Answers the hedge's new lane count.
    fn grow(&self, lanes: Lanes, length: u64, held: &AtomicUsize) -> Option<usize> {
        let current = held.load(Ordering::Acquire);
        let hedges = self.ranged.load(Ordering::Acquire).max(1);
        let target = (lanes.level.saturating_sub(lanes.busy) / hedges).min(parts(length));
        if current == CLOSED || target <= current {
            return None;
        }
        let taken = self.borrow(lanes, target - current, false);
        if taken == 0 {
            return None;
        }
        if held
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n != CLOSED).then_some(n + taken)
            })
            .is_err()
        {
            self.racing.fetch_sub(taken, Ordering::AcqRel);
            return None;
        }
        Some(current + taken)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fetch(
        &self,
        store: &Store,
        grant: &DeliveryGrant,
        authorize: &(dyn Fn(bool, &Ledger) -> Result<String> + Sync),
        policy: &SourcePolicy,
        credential: &dyn CredentialProvider,
        deadline: Deadline,
        ledger: &Ledger,
        lanes: &(dyn Fn() -> Lanes + Sync),
        received: Option<&(dyn Fn() + Sync)>,
    ) -> Result<Fetched> {
        // Original asks and hedges share one budget, so a broken object is not re-bought
        // once per idle lane.
        let attempts = AtomicU32::new(0);
        if grant.destination != Destination::Blob {
            return fetch_with_attempts(
                store,
                grant,
                authorize,
                policy,
                credential,
                deadline,
                ledger,
                None,
                &attempts,
                true,
                shared_client(),
                received,
            );
        }
        let length = grant.object.length;
        std::thread::scope(|scope| {
            let (sender, receiver) = mpsc::channel();
            let attempts = &attempts;
            let launch = |lanes: usize| {
                let state = Arc::new(Attempt::default());
                let local = ledger.for_attempt(Arc::clone(&state));
                let sender = sender.clone();
                let index_state = Arc::clone(&state);
                let held = Arc::new(AtomicUsize::new(lanes));
                if lanes > 1 {
                    self.ranged.fetch_add(1, Ordering::AcqRel);
                }
                let hold = Hold {
                    recovery: self,
                    lanes: Arc::clone(&held),
                    ranged: lanes > 1,
                };
                scope.spawn(move || {
                    let hold = hold;
                    let started = Instant::now();
                    let result = if lanes > 1 {
                        ranged::hedge(
                            store,
                            grant,
                            authorize,
                            policy,
                            credential,
                            deadline,
                            &local,
                            &hold.lanes,
                        )
                    } else {
                        // A whole-object hedge starts a fresh TCP and TLS session; the
                        // original keeps connection reuse.
                        let fresh = (lanes == 1).then(super::http::Client::new);
                        fetch_with_attempts(
                            store,
                            grant,
                            authorize,
                            policy,
                            credential,
                            deadline,
                            &local,
                            None,
                            attempts,
                            lanes == 0,
                            fresh.as_ref().unwrap_or_else(|| shared_client()),
                            received,
                        )
                    };
                    let _ = sender.send((index_state, result, started.elapsed()));
                });
                Racer {
                    state,
                    lanes: held,
                    ranged: lanes > 1,
                    live: true,
                    window: Window::default(),
                }
            };
            let mut racers = vec![launch(0)];
            let mut failure = None;
            let result = loop {
                match receiver.recv_timeout(ledger.sample()) {
                    Ok((state, Ok(found), elapsed)) => {
                        self.learn(&state.data.lock().unwrap(), length, elapsed);
                        if racers.len() > 1 {
                            self.won.fetch_add(
                                usize::from(!Arc::ptr_eq(&racers[0].state, &state)),
                                Ordering::Relaxed,
                            );
                            let winner = racers
                                .iter()
                                .position(|racer| Arc::ptr_eq(&racer.state, &state))
                                .unwrap_or(0);
                            let kind = match winner {
                                0 => "original",
                                _ if racers[winner].ranged => "ranged",
                                _ => "whole",
                            };
                            decisions::record(
                                store,
                                "won",
                                &decisions::object(&grant.object),
                                &format!(
                                    "{kind} racers={} after_s={:.1}",
                                    racers.len(),
                                    elapsed.as_secs_f64()
                                ),
                            );
                        }
                        if let Some(received) = received {
                            received();
                        }
                        break Ok(found);
                    }
                    Ok((state, Err(error), _)) => {
                        let racer = racers
                            .iter_mut()
                            .find(|racer| Arc::ptr_eq(&racer.state, &state))
                            .expect("every racer is recorded");
                        racer.live = false;
                        // The original's refusal names the object; a hedge's names a hedge.
                        if Arc::ptr_eq(&racers[0].state, &state) || failure.is_none() {
                            failure = Some(error);
                        }
                        if racers.iter().all(|racer| !racer.live) {
                            break Err(failure.take().expect("a racer failed"));
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        if let Err(error) = ledger
                            .check_running()
                            .and_then(|()| deadline.remaining().map(|_| ()))
                        {
                            break Err(error);
                        }
                        let first = racers.len() == 1;
                        let now = lanes();
                        // A hedge condemns nothing, so two samples of measured network wait
                        // are evidence enough; the stall floor guards verdicts, not hedges.
                        let evidence = ledger.sample() * 2;
                        let mut every_slow = true;
                        let mut measured = true;
                        let mut slowest: Option<(usize, f64)> = None;
                        let mut ranged_live = false;
                        for (index, racer) in racers.iter_mut().enumerate() {
                            if !racer.live {
                                continue;
                            }
                            // A ranged hedge heals its own slow parts on its own lanes, and
                            // takes its share of whatever lanes have come free.
                            if racer.ranged {
                                ranged_live = true;
                                if let Some(lanes) = self.grow(now, length, &racer.lanes) {
                                    decisions::record(
                                        store,
                                        "grow",
                                        &decisions::object(&grant.object),
                                        &format!(
                                            "lanes={lanes} busy={} level={} hedges={}",
                                            now.busy,
                                            now.level,
                                            self.ranged.load(Ordering::Acquire)
                                        ),
                                    );
                                }
                                continue;
                            }
                            let data = racer.state.data.lock().unwrap().clone();
                            racer.window.observe(&data);
                            // With no comparable evidence, only an idle tail probes, once.
                            let verdict = self.judge(&data, &racer.window, length, evidence);
                            measured &= verdict == Verdict::Slow;
                            every_slow &= match verdict {
                                Verdict::Slow => true,
                                Verdict::Unreferenced => first && now.tail,
                                Verdict::Healthy | Verdict::Unmeasured => false,
                            };
                            let rate = racer.window.rate(&data);
                            if slowest.is_none_or(|(_, worst)| rate < worst) {
                                slowest = Some((index, rate));
                            }
                        }
                        if ranged_live || !every_slow || slowest.is_none() {
                            continue;
                        }
                        let (index, rate) = slowest.expect("a live racer was judged");
                        let left =
                            length.saturating_sub(racers[index].state.data.lock().unwrap().bytes);
                        if length >= 2 * ranged::HEDGE_PART {
                            // A cold R2 read stalls at a chunk boundary, and every whole GET
                            // of that object then streams in lockstep behind the same stall
                            // (sasori, fern). Fresh ranged asks are served apart from it, so a
                            // large object is only ever hedged as ranged parts, on at least
                            // two lanes so one can re-ask whatever the other is stuck on.
                            let Some(lanes) = self.reserve(now, length, store, false, measured)
                            else {
                                continue;
                            };
                            let budgeted = attempts
                                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                                    (n < FETCH_ATTEMPTS).then_some(n + 1)
                                })
                                .is_ok();
                            if !budgeted {
                                self.racing.fetch_sub(lanes, Ordering::AcqRel);
                                continue;
                            }
                            decisions::record(
                                store,
                                "hedge",
                                &decisions::object(&grant.object),
                                &format!(
                                    "ranged lanes={lanes} rate_bps={rate:.0} left={left} \
                                     measured={measured} busy={} level={}",
                                    now.busy, now.level
                                ),
                            );
                            self.hedges.fetch_add(1, Ordering::Relaxed);
                            racers.push(launch(lanes));
                        } else if first {
                            if self.reserve(now, length, store, true, measured).is_some() {
                                decisions::record(
                                    store,
                                    "hedge",
                                    &decisions::object(&grant.object),
                                    &format!("whole rate_bps={rate:.0} left={left}"),
                                );
                                self.hedges.fetch_add(1, Ordering::Relaxed);
                                racers.push(launch(1));
                            }
                        } else if measured
                            && racers.iter().filter(|racer| racer.live).count() > 1
                            && attempts.load(Ordering::Acquire) < FETCH_ATTEMPTS
                        {
                            // A small object's racers are all slow: the slowest gives its
                            // lane to a fresh GET. A sole survivor is never stopped.
                            racers[index].state.stopped.store(true, Ordering::Release);
                            self.racing.fetch_add(1, Ordering::AcqRel);
                            decisions::record(
                                store,
                                "replace",
                                &decisions::object(&grant.object),
                                &format!("whole rate_bps={rate:.0} left={left}"),
                            );
                            racers.push(launch(1));
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        unreachable!("the local sender remains live")
                    }
                }
            };
            // Ok exists only after exact length/hash verification and CAS admission.
            // Errors cancel this object's attempts only; the caller owns pull failure.
            for racer in &racers {
                racer.state.stopped.store(true, Ordering::Release);
            }
            result
        })
    }
}

#[derive(Debug, PartialEq)]
enum Verdict {
    Unmeasured,
    Healthy,
    Slow,
    Unreferenced,
}

#[derive(Default)]
struct Window(VecDeque<AttemptData>);
impl Window {
    /// Bytes per second of network wait across the window.
    fn rate(&self, data: &AttemptData) -> f64 {
        let Some(first) = self.0.front() else {
            return f64::INFINITY;
        };
        let waited = data.waited.saturating_sub(first.waited).as_secs_f64();
        if waited == 0.0 {
            return f64::INFINITY;
        }
        data.bytes.saturating_sub(first.bytes) as f64 / waited
    }

    fn observe(&mut self, data: &AttemptData) {
        if self
            .0
            .back()
            .is_some_and(|old| data.bytes < old.bytes || data.waited < old.waited)
        {
            self.0.clear(); // a normal retry starts a new measurement
        }
        self.0.push_back(data.clone());
        while self.0.len() > STILL_SAMPLES as usize + 2 {
            self.0.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measured_reference_keeps_comparable_size_and_does_not_learn_local_pauses() {
        let recovery = Recovery::default();
        let mut data = AttemptData {
            bytes: 4 << 20,
            waited: Duration::from_secs(30),
            origin: "origin".into(),
            waiting: true,
        };
        let window = Window(VecDeque::from([AttemptData::default()]));
        let size = 64 << 20;
        let floor = Duration::from_secs(30);
        let judge =
            |recovery: &Recovery, data: &AttemptData| recovery.judge(data, &window, size, floor);
        assert_eq!(
            judge(&recovery, &data),
            Verdict::Unreferenced,
            "15x its own wait left"
        );
        recovery.learn(&data, 1024, Duration::from_millis(1));
        assert_eq!(
            judge(&recovery, &data),
            Verdict::Unreferenced,
            "tiny fast metadata is not large-object evidence"
        );
        recovery.learn(&data, size, Duration::from_secs(3));
        assert_eq!(judge(&recovery, &data), Verdict::Slow);
        recovery.learn(&data, size, Duration::from_secs(900));
        assert_eq!(
            judge(&recovery, &data),
            Verdict::Slow,
            "a completed straggler cannot erase the demonstrated replacement cost"
        );
        data.waited = Duration::from_millis(1);
        assert_eq!(
            judge(&recovery, &data),
            Verdict::Unmeasured,
            "consumer time supplies no evidence of network degradation"
        );
        data.waited = floor;
        data.bytes = 60 << 20;
        assert_eq!(
            judge(&recovery, &data),
            Verdict::Healthy,
            "nearly done at its own rate"
        );
    }

    #[test]
    fn a_hedge_takes_its_fair_share_of_idle_lanes_and_returns_them() {
        let root = std::env::temp_dir().join(format!(
            "tensorfs-hedge-lanes-{}-{}",
            std::process::id(),
            crate::meta::now_nanos_unique()
        ));
        let store = Store::init(&root).unwrap();
        let recovery = Recovery::default();
        let tail = Lanes {
            level: 128,
            busy: 8,
            tail: true,
        };
        let lanes = recovery
            .reserve(tail, 64 << 20, &store, false, false)
            .unwrap();
        assert_eq!(lanes, 16, "eight stragglers share 120 idle lanes");
        recovery.ranged.fetch_add(1, Ordering::AcqRel);
        let hold = Hold {
            recovery: &recovery,
            lanes: Arc::new(AtomicUsize::new(lanes)),
            ranged: true,
        };
        assert_eq!(
            recovery.reserve(tail, 64 << 20, &store, true, false),
            Some(1)
        );
        recovery.racing.fetch_sub(1, Ordering::AcqRel);
        assert_eq!(recovery.grow(tail, 64 << 20, &hold.lanes), Some(64));
        let drained = Lanes {
            level: 128,
            busy: 1,
            tail: true,
        };
        assert_eq!(
            recovery.grow(drained, 64 << 20, &hold.lanes),
            None,
            "one lane per part is all a ranged copy can use"
        );
        let lanes = Arc::clone(&hold.lanes);
        drop(hold);
        assert_eq!(recovery.racing.load(Ordering::Acquire), 0);
        assert_eq!(recovery.ranged.load(Ordering::Acquire), 0);
        assert_eq!(
            recovery.grow(drained, 64 << 20, &lanes),
            None,
            "a returned hedge is lent nothing"
        );
        assert_eq!(recovery.racing.load(Ordering::Acquire), 0);
        let full = Lanes {
            level: 4,
            busy: 4,
            tail: false,
        };
        assert_eq!(recovery.reserve(full, 64 << 20, &store, false, false), None);
        assert_eq!(
            recovery.reserve(full, 64 << 20, &store, false, true),
            Some(2),
            "a measured stall lends its own lane to its hedge"
        );
        let _ = std::fs::remove_dir_all(root);
    }
}

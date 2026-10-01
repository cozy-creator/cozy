//! Hedged recovery of slow object streams. An object's original GET may race one hedge
//! on the walk's idle lanes; the first verified admission wins and stops the other. A
//! hedge is earned by measurement, never by a timer: a stream is slow when,
//! at its own measured network rate, what it has left would take more than
//! `STALL_FACTOR` times the pull's best completed transfer of a comparable object.
//!
//! A large object is hedged as ranged parts on its share of the idle lanes, so a link
//! whose individual answers are slow still fills; a small one as one fresh whole GET.
//! Run 2322: R2 answered eight 64 MiB objects application-limited at 6-50 KB/s while the
//! rest of the pull moved 400 MB/s, and nothing re-asked them for fifteen minutes.
use super::{
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
    /// Lanes hedges hold across the whole walk.
    racing: AtomicUsize,
}

/// Lanes one hedge holds, returned however its thread leaves.
struct Hold<'a> {
    racing: &'a AtomicUsize,
    lanes: usize,
}

impl Drop for Hold<'_> {
    fn drop(&mut self) {
        self.racing.fetch_sub(self.lanes, Ordering::AcqRel);
    }
}

struct Racer {
    state: Arc<Attempt>,
    ranged: bool,
    live: bool,
    window: Window,
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
    /// so that lane counts as idle for its hedge even when the walk has none to spare: a
    /// cold-object stall mid-pull is hedged then, not only once the walk drains.
    /// The caller hands the count to a `Hold`, which returns it.
    fn reserve(
        &self,
        lanes: Lanes,
        length: u64,
        store: &Store,
        whole: bool,
        own: bool,
    ) -> Option<usize> {
        let share = (lanes.level / lanes.busy.max(1)).max(1);
        let parts = if whole {
            1
        } else {
            (length / ranged::HEDGE_PART).max(1) as usize
        };
        loop {
            let racing = self.racing.load(Ordering::Acquire);
            let spare = (lanes.level + usize::from(own)).saturating_sub(lanes.busy + racing);
            let mut want = share.min(spare).min(parts);
            // Room for the original's object beside the hedge's copies: a ranged hedge
            // assembles in scratch before its door writes, so a disk without room for
            // both still takes one whole GET.
            if want > 1 && store.admits(length.saturating_mul(3)).is_err() {
                want = 1;
            }
            if want == 0 || store.admits(length.saturating_mul(2)).is_err() {
                return None;
            }
            if self
                .racing
                .compare_exchange(racing, racing + want, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Some(want);
            }
        }
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
                scope.spawn(move || {
                    let _hold = Hold {
                        racing: &self.racing,
                        lanes,
                    };
                    let started = Instant::now();
                    let result = if lanes > 1 {
                        ranged::hedge(
                            store, grant, authorize, policy, credential, deadline, &local, lanes,
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
                        // One hedge per object. A ranged hedge the origin refused (one that
                        // will not answer ranges, say) may be followed by one whole GET.
                        let first = racers.len() == 1;
                        let fallback = racers.len() == 2 && racers[1].ranged && !racers[1].live;
                        let now = lanes();
                        // A hedge condemns nothing, so two samples of measured network wait
                        // are evidence enough; the stall floor guards verdicts, not hedges.
                        let evidence = ledger.sample() * 2;
                        let mut every_slow = true;
                        let mut measured = true;
                        let mut slowest: Option<(usize, f64)> = None;
                        for (index, racer) in racers.iter_mut().enumerate() {
                            if !racer.live {
                                continue;
                            }
                            let data = racer.state.data.lock().unwrap().clone();
                            racer.window.observe(&data);
                            // A ranged hedge heals its own slow parts on its own lanes. With no
                            // comparable evidence, only an idle tail probes, once per object.
                            let verdict = if racer.ranged {
                                Verdict::Healthy
                            } else {
                                self.judge(&data, &racer.window, length, evidence)
                            };
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
                        if !every_slow {
                            continue;
                        }
                        if first || fallback {
                            if let Some(lanes) =
                                self.reserve(now, length, store, fallback, measured)
                            {
                                // A whole hedge spends its ask itself; a ranged one spends
                                // one here, its parts resuming within their own asks.
                                let budgeted = if lanes > 1 {
                                    attempts
                                        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                                            (n < FETCH_ATTEMPTS).then_some(n + 1)
                                        })
                                        .is_ok()
                                } else {
                                    attempts.load(Ordering::Acquire) < FETCH_ATTEMPTS
                                };
                                if budgeted {
                                    racers.push(launch(lanes));
                                } else {
                                    self.racing.fetch_sub(lanes, Ordering::AcqRel);
                                }
                            }
                        } else if measured
                            && racers.iter().filter(|racer| racer.live).count() > 1
                            && attempts.load(Ordering::Acquire) < FETCH_ATTEMPTS
                        {
                            // Every racer is measured slow. A cold R2 read stalls at a chunk
                            // boundary and a fresh ask moves 50-100x faster, so the slowest
                            // racer gives its lane to a fresh whole GET rather than waiting.
                            // A sole survivor is never stopped: its hedge already failed.
                            let (index, _) = slowest.expect("a live racer was judged");
                            racers[index].state.stopped.store(true, Ordering::Release);
                            self.racing.fetch_add(1, Ordering::AcqRel);
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
        let hold = Hold {
            racing: &recovery.racing,
            lanes,
        };
        assert_eq!(
            recovery.reserve(tail, 64 << 20, &store, true, false),
            Some(1)
        );
        recovery.racing.fetch_sub(1, Ordering::AcqRel);
        drop(hold);
        assert_eq!(recovery.racing.load(Ordering::Acquire), 0);
        let full = Lanes {
            level: 4,
            busy: 4,
            tail: false,
        };
        assert_eq!(recovery.reserve(full, 64 << 20, &store, false, false), None);
        assert_eq!(
            recovery.reserve(full, 64 << 20, &store, false, true),
            Some(1),
            "a measured stall lends its own lane to its hedge"
        );
        let _ = std::fs::remove_dir_all(root);
    }
}

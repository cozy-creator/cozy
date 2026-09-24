//! One spare stream may race a measured slow object; only verified completion
//! cancels its peer. This is recovery from observed degradation, not a deadline.
use super::{
    ledger::{Attempt, AttemptData, Deadline, Ledger, STALL_FACTOR, STILL_SAMPLES},
    pull::{fetch_authorized, fetch_with_attempts, shared_client, Fetched},
    CredentialProvider, SourcePolicy,
};
use crate::{
    err::Result,
    fetch::{DeliveryGrant, Destination, FetchPlan},
    store::Store,
};
use std::{
    collections::{BTreeMap, VecDeque},
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        mpsc, Arc, Condvar, Mutex,
    },
    time::{Duration, Instant},
};

struct Reference {
    length: u64,
    elapsed: Duration,
}

pub(super) struct Recovery {
    enabled: bool,
    occupied: AtomicBool,
    unhelpful: AtomicBool,
    references: Mutex<BTreeMap<String, Reference>>,
    slots: Slots,
}

impl Recovery {
    pub fn new(
        store: &Store,
        plan: &FetchPlan,
        streams: usize,
        allowance: Option<(u64, u64)>,
    ) -> Self {
        let largest = plan.wanted.iter().map(|o| o.length).max().unwrap_or(0);
        let extra = plan.wanted_bytes().checked_add(largest);
        let enabled = streams > 2
            && !plan.wanted.is_empty()
            && extra.is_some_and(|bytes| {
                allowance.is_some_and(|(limit, objects)| {
                    bytes <= limit && (plan.wanted.len() as u64) < objects
                }) && store.admits(bytes).is_ok()
            });
        Self {
            enabled,
            occupied: AtomicBool::new(false),
            unhelpful: AtomicBool::new(false),
            references: Mutex::new(BTreeMap::new()),
            slots: Slots::new(streams.saturating_sub(1)),
        }
    }

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

    fn worth_asking(
        &self,
        data: &AttemptData,
        window: &Window,
        length: u64,
        floor: Duration,
    ) -> bool {
        if self.unhelpful.load(Ordering::Acquire) || !data.waiting || data.bytes >= length {
            return false;
        }
        let Some(first) = window.0.front() else {
            return false;
        };
        let waited = data.waited.saturating_sub(first.waited);
        let bytes = data.bytes.saturating_sub(first.bytes);

        if waited < floor || bytes == 0 {
            return false; // silence already has its own policy; local work is not network delay
        }
        let references = self.references.lock().unwrap();
        let Some(reference) = references.get(&data.origin) else {
            return false;
        };
        if reference.length < length {
            return false; // fast small metadata cannot qualify a large-body retry
        }
        // Excluding consumer time makes the candidate rate optimistic. Even that
        // measured remaining time must dwarf a whole replacement's proven cost.
        let rate = bytes as f64 / waited.as_secs_f64();
        (length - data.bytes) as f64 / rate > STALL_FACTOR * reference.elapsed.as_secs_f64()
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
        tail: &(dyn Fn() -> bool + Sync),
    ) -> Result<Fetched> {
        if !self.enabled {
            return fetch_authorized(
                store,
                grant,
                &|stale| authorize(stale, ledger),
                policy,
                credential,
                deadline,
                ledger,
                None,
            );
        }
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
                &AtomicU32::new(0),
                true,
                Some(&self.slots),
                shared_client(),
            );
        }
        let attempts = AtomicU32::new(0);
        let states = [Arc::new(Attempt::default()), Arc::new(Attempt::default())];
        let mut permit = None;
        std::thread::scope(|scope| {
            let (sender, receiver) = mpsc::channel();
            let launch = |index, retry, send: mpsc::Sender<_>| {
                let local = ledger.for_attempt(Arc::clone(&states[index]));
                let attempts = &attempts;
                scope.spawn(move || {
                    let started = Instant::now();
                    // A rare contender starts a fresh TLS session context as well
                    // as a fresh TCP connection; normal requests retain reuse.
                    let fresh = (index == 1).then(super::http::Client::new);
                    let client = fresh.as_ref().unwrap_or_else(|| shared_client());
                    let result = fetch_with_attempts(
                        store,
                        grant,
                        authorize,
                        policy,
                        credential,
                        deadline,
                        &local,
                        None,
                        attempts,
                        retry,
                        if index == 0 { Some(&self.slots) } else { None },
                        client,
                    );
                    let _ = send.send((index, result, started.elapsed()));
                });
            };
            launch(0, true, sender.clone());
            let mut window = Window::default();
            let mut hedged = false;
            let mut contender_live = false;
            let mut original_error = None;
            let result = loop {
                match receiver.recv_timeout(ledger.sample()) {
                    Ok((index, Ok(found), elapsed)) => {
                        self.learn(
                            &states[index].data.lock().unwrap(),
                            grant.object.length,
                            elapsed,
                        );
                        if hedged && index == 0 {
                            // A uniformly slower route earns continued progress, not
                            // one speculative download of every remaining object.
                            self.unhelpful.store(true, Ordering::Release);
                        }
                        break Ok(found);
                    }
                    Ok((index, Err(error), _)) => {
                        if index == 0 {
                            original_error = Some(error);
                        } else {
                            contender_live = false;
                            self.unhelpful.store(true, Ordering::Release);
                        }
                        if !contender_live {
                            if let Some(error) = original_error.take() {
                                break Err(error);
                            }
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        if let Err(error) = ledger
                            .check_running()
                            .and_then(|()| deadline.remaining().map(|_| ()))
                        {
                            break Err(error);
                        }
                        let data = states[0].data.lock().unwrap().clone();
                        window.observe(&data);
                        // A bounded tail may have no newly completed reference
                        // (a resumed pull can contain only one missing object).
                        // Spare capacity permits one extra ask, never a failure
                        // verdict or cancellation before a verified winner.
                        let idle_tail = tail()
                            && data.waiting
                            && !data.origin.is_empty()
                            && data.bytes < grant.object.length;
                        if !hedged
                            && (idle_tail
                                || self.worth_asking(
                                    &data,
                                    &window,
                                    grant.object.length,
                                    ledger.floor(),
                                ))
                            && self
                                .occupied
                                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                                .is_ok()
                        {
                            let reserved = Permit(&self.occupied);
                            if !idle_tail && self.unhelpful.load(Ordering::Acquire) {
                                drop(reserved);
                                continue;
                            }
                            permit = Some(reserved);
                            hedged = true;
                            contender_live = true;
                            launch(1, false, sender.clone());
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        unreachable!("the local sender remains live")
                    }
                }
            };
            // Ok exists only after exact length/hash verification and CAS admission.
            // Errors cancel this object's attempts only; the caller owns pull failure.
            for state in &states {
                state.stopped.store(true, Ordering::Release);
            }
            result
        })
        // The permit outlives both scoped threads and their temporary-file cleanup.
    }
}

/// Normal requests may prepare their grant concurrently, but only N-1 may own
/// object sockets/temporary writers. The one contender uses the reserved slot.
pub(super) struct Slots {
    limit: usize,
    used: Mutex<usize>,
    ready: Condvar,
}
impl Slots {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            used: Mutex::new(0),
            ready: Condvar::new(),
        }
    }

    pub(super) fn acquire(&self, ledger: &Ledger, deadline: Deadline) -> Result<Slot<'_>> {
        let mut used = self.used.lock().unwrap();
        let mut waited = false;
        while *used >= self.limit {
            waited = true;
            ledger.check_running()?;
            let tick = deadline
                .remaining()?
                .map_or(ledger.sample(), |left| left.min(ledger.sample()));
            used = self.ready.wait_timeout(used, tick).unwrap().0;
        }
        ledger.check_running()?;
        deadline.remaining()?;
        *used += 1;
        Ok(Slot {
            slots: self,
            waited,
        })
    }
}

pub(super) struct Slot<'a> {
    slots: &'a Slots,
    pub waited: bool,
}
impl Drop for Slot<'_> {
    fn drop(&mut self) {
        *self.slots.used.lock().unwrap() -= 1;
        self.slots.ready.notify_one();
    }
}

struct Permit<'a>(&'a AtomicBool);
impl Drop for Permit<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

#[derive(Default)]
struct Window(VecDeque<AttemptData>);
impl Window {
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
        let recovery = Recovery {
            enabled: true,
            occupied: AtomicBool::new(false),
            unhelpful: AtomicBool::new(false),
            references: Mutex::new(BTreeMap::new()),
            slots: Slots::new(1),
        };
        let mut data = AttemptData {
            bytes: 4 << 20,
            waited: Duration::from_secs(30),
            origin: "origin".into(),
            waiting: true,
        };
        let window = Window(VecDeque::from([AttemptData::default()]));
        let size = 64 << 20;
        let floor = Duration::from_secs(30);
        recovery.learn(&data, 1024, Duration::from_millis(1));
        assert!(
            !recovery.worth_asking(&data, &window, size, floor),
            "tiny fast metadata is not large-object evidence"
        );
        recovery.learn(&data, 1024, Duration::from_secs(10));
        recovery.learn(&data, size, Duration::from_secs(3));
        assert!(
            recovery.worth_asking(&data, &window, size, floor),
            "slow small metadata must not poison the later large sample"
        );
        recovery.learn(&data, size, Duration::from_secs(900));
        assert!(
            recovery.worth_asking(&data, &window, size, floor),
            "a completed straggler cannot erase the demonstrated replacement cost"
        );
        data.waited = Duration::from_millis(1);
        assert!(
            !recovery.worth_asking(&data, &window, size, floor),
            "consumer time supplies no evidence of network degradation"
        );
        data.waited = floor;
        recovery.unhelpful.store(true, Ordering::Release);
        assert!(
            !recovery.worth_asking(&data, &window, size, floor),
            "an unhelpful contender must prevent further speculative traffic"
        );
    }
}

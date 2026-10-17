//! Ranged transfers side by side, as an S3 client makes them: each range on its own lane (its own
//! connection), judged by its measured bytes against its peers' over the same window. A range that
//! fails, or whose bytes come `SLOW` times slower than its peers', goes again from where its bytes
//! stopped on a fresh connection. A transfer gives up only when every lane has failed since a byte
//! last moved. No flat timeouts: only lack of measured progress ends anything.
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use tokio::{
    task::{AbortHandle, Id, JoinSet},
    time::Instant,
};

/// A range whose bytes come this many times slower than its peers' is asked again.
const SLOW: f64 = 8.0;
/// How often ranges in flight are measured, and the shortest window a rate is judged over.
const SAMPLE: Duration = Duration::from_secs(1);
const WINDOW: Duration = Duration::from_secs(2);
/// A rate is judged over at least this many round trips (a range's typical wait for its first
/// byte), so TCP's own loss recovery, which holds bytes back a round trip or two, never reads as a
/// slow range on a far link.
const ROUND_TRIPS: u32 = 4;

/// Why a range's transfer stopped.
pub(crate) enum Ranged<E> {
    /// Asked again from where its bytes stopped, on a fresh connection.
    Again(String),
    /// The whole transfer stops.
    Stop(E),
}

/// What a lane is given to move one range: its lane, whether its connection must be fresh, the
/// range `[first, end)`, and `at`, which it moves past each byte it carries.
pub(crate) struct Leg {
    pub lane: usize,
    pub fresh: bool,
    pub first: u64,
    pub end: u64,
    pub at: Arc<AtomicU64>,
}

/// A range in flight: its task and how far its bytes are.
struct Flight {
    lane: usize,
    first: u64,
    end: u64,
    at: Arc<AtomicU64>,
    began: Instant,
    /// Since its first byte: `at` as each sample found it, back to one a window ago.
    seen: VecDeque<(Instant, u64)>,
    slow: bool,
    task: AbortHandle,
}

/// Moves `ranges` over at most `lanes` side by side, each by `start`. A range that answers
/// `Again` (or is judged slow) goes again from where its bytes stopped, on a fresh connection.
/// Gives up (`Again`, with the last failure) only when every lane has failed since a byte last
/// moved.
pub(crate) async fn transfer<E, F, Fut>(
    ranges: impl IntoIterator<Item = (u64, u64)>,
    lanes: usize,
    mut start: F,
) -> Result<(), Ranged<E>>
where
    E: Send + 'static,
    F: FnMut(Leg) -> Fut,
    Fut: Future<Output = Result<(), Ranged<E>>> + Send + 'static,
{
    let mut queue: VecDeque<(u64, u64)> = ranges.into_iter().filter(|(a, b)| a < b).collect();
    let (mut tasks, mut flights) = (JoinSet::new(), HashMap::<Id, Flight>::new());
    let (mut rates, mut waits) = (vec![], vec![]);
    let mut fresh = vec![false; lanes];
    let (mut failures, mut moved, mut moved_then) = (0, 0, 0);
    let mut sample = tokio::time::interval(SAMPLE);
    loop {
        while flights.len() < lanes
            && let Some((first, end)) = queue.pop_front()
        {
            let lane = (0..lanes)
                .find(|lane| !flights.values().any(|f| f.lane == *lane))
                .expect("a free lane");
            let at = Arc::new(AtomicU64::new(first));
            let leg = Leg {
                lane,
                fresh: std::mem::take(&mut fresh[lane]),
                first,
                end,
                at: at.clone(),
            };
            let task = tasks.spawn(start(leg));
            let flight = Flight {
                lane,
                first,
                end,
                at,
                began: Instant::now(),
                seen: VecDeque::new(),
                slow: false,
                task,
            };
            flights.insert(flight.task.id(), flight);
        }
        if flights.is_empty() {
            return Ok(());
        }
        tokio::select! {
            Some(joined) = tasks.join_next_with_id() => {
                let (id, done) = match joined {
                    Ok((id, done)) => (id, done),
                    Err(e) => (e.id(), Err(Ranged::Again(e.to_string()))),
                };
                let flight = flights.remove(&id).expect("a flight per task");
                let at = flight.at.load(Ordering::Relaxed);
                moved += at - flight.first;
                match done {
                    Ok(()) => {
                        let seconds = flight.began.elapsed().as_secs_f64().max(1e-3);
                        rates.push((flight.end - flight.first) as f64 / seconds);
                    }
                    Err(Ranged::Stop(e)) => return Err(Ranged::Stop(e)),
                    Err(Ranged::Again(message)) => {
                        fresh[flight.lane] = true;
                        queue.push_front((at, flight.end));
                        if !flight.slow {
                            if moved > moved_then {
                                (failures, moved_then) = (0, moved);
                            }
                            failures += 1;
                            if failures > lanes {
                                return Err(Ranged::Again(message));
                            }
                        }
                    }
                }
            }
            _ = sample.tick() => judge(&mut flights, &rates, &mut waits),
        }
    }
}

/// Measures the ranges in flight and drops the one whose first byte waits `SLOW` times longer
/// than its peers' did, or whose bytes come `SLOW` times slower than its peers' over the same
/// window of `ROUND_TRIPS` round trips; its rest is asked again. Peers share the path, so one
/// lane's smaller share is TCP's doing, not a bad connection: a range is held to ranges in flight
/// beside it, and to finished ones only when it is alone, a straggler at the tail. A range whose
/// bytes have all moved is waiting for its answer and is not judged.
fn judge(flights: &mut HashMap<Id, Flight>, finished: &[f64], waits: &mut Vec<Duration>) {
    let now = Instant::now();
    waits.sort();
    let wait = waits.get(waits.len() / 2).copied();
    let window = wait.map_or(WINDOW, |wait| (wait * ROUND_TRIPS).max(WINDOW));
    let alone = flights.len() == 1;
    let mut judged = vec![];
    for flight in flights.values_mut() {
        let at = flight.at.load(Ordering::Relaxed);
        if at >= flight.end {
            continue;
        }
        if flight.seen.is_empty() {
            if at > flight.first {
                waits.push(now - flight.began);
                flight.seen.push_back((now, at));
            } else {
                let typical = wait.map(|wait| wait.mul_f64(SLOW));
                flight.slow |= typical.is_some_and(|t| now - flight.began > t.max(WINDOW));
            }
            continue;
        }
        flight.seen.push_back((now, at));
        while flight
            .seen
            .get(1)
            .is_some_and(|(then, _)| now - *then >= window)
        {
            flight.seen.pop_front();
        }
        let (then, from) = flight.seen[0];
        if now - then >= window {
            judged.push((
                flight.task.id(),
                (at - from) as f64 / (now - then).as_secs_f64(),
            ));
        }
    }
    let mut rates: Vec<f64> = match alone {
        true => finished.to_vec(),
        false => judged.iter().map(|(_, rate)| *rate).collect(),
    };
    rates.sort_by(f64::total_cmp);
    let enough = if alone { 1 } else { 2 };
    let typical = (rates.len() >= enough).then(|| rates[rates.len() / 2]);
    for (id, rate) in judged {
        if let (Some(typical), Some(flight)) = (typical, flights.get_mut(&id)) {
            flight.slow |= rate * SLOW < typical;
        }
    }
    for flight in flights.values().filter(|flight| flight.slow) {
        flight.task.abort();
    }
}

/// What `held` leaves of `[0, length)`, cut into parts of `clamp(missing / lanes, 1 MiB, most)`
/// bytes, so every lane ramps on its own connection.
pub(crate) fn parts(held: &[(u64, u64)], length: u64, lanes: usize, most: u64) -> Vec<(u64, u64)> {
    let missing = cozy_proto::machine::missing(held, length);
    let total = cozy_proto::machine::held_bytes(&missing);
    let size = (total / lanes as u64).clamp(1 << 20, most);
    missing
        .into_iter()
        .flat_map(|(a, b)| {
            (a..b)
                .step_by(size as usize)
                .map(move |x| (x, (x + size).min(b)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    #[test]
    fn missing_bytes_are_cut_into_parts_a_lane_each() {
        assert_eq!(
            parts(&[], 20 * MIB, 4, 64 * MIB),
            (0..4)
                .map(|i| (i * 5 * MIB, (i + 1) * 5 * MIB))
                .collect::<Vec<_>>()
        );
        assert_eq!(parts(&[(0, 100)], 100, 4, 64 * MIB), vec![], "all held");
        assert_eq!(
            parts(&[(0, 10), (20, 30)], 3 * MIB, 4, 64 * MIB),
            vec![
                (10, 20),
                (30, MIB + 30),
                (MIB + 30, 2 * MIB + 30),
                (2 * MIB + 30, 3 * MIB)
            ],
            "gaps, then 1 MiB parts at least"
        );
        assert_eq!(
            parts(&[], 1024 * MIB, 4, 64 * MIB).len(),
            16,
            "at most `most`"
        );
    }

    /// A range that fails resumes from where its bytes stopped, its lane redialed; one that never
    /// moves a byte while its peers finish is judged slow and asked again.
    #[tokio::test(start_paused = true)]
    async fn a_failed_or_stalled_range_resumes_where_it_stopped() {
        let legs = Arc::new(std::sync::Mutex::new(vec![]));
        let seen = legs.clone();
        let done = transfer::<(), _, _>([(0, 100), (100, 200), (200, 300)], 3, move |leg| {
            let mut seen = seen.lock().unwrap();
            seen.push((leg.first, leg.end, leg.fresh));
            let attempt = seen.len();
            async move {
                match (leg.first, attempt) {
                    // Fails half way through.
                    (0, 1) => {
                        leg.at.store(50, Ordering::Relaxed);
                        Err(Ranged::Again("dropped".into()))
                    }
                    // Never moves a byte.
                    (200, 3) => std::future::pending().await,
                    _ => {
                        leg.at.store(leg.first + 1, Ordering::Relaxed);
                        tokio::time::sleep(Duration::from_millis(1500)).await;
                        leg.at.store(leg.end, Ordering::Relaxed);
                        Ok(())
                    }
                }
            }
        })
        .await;
        assert!(done.is_ok());
        let legs = legs.lock().unwrap().clone();
        assert!(
            legs.contains(&(50, 100, true)),
            "on its lane, redialed: {legs:?}"
        );
        let stalled = legs.iter().filter(|(first, ..)| *first == 200).count();
        assert_eq!(stalled, 2, "asked again on a free lane: {legs:?}");
    }

    /// Every lane failing with no byte moved gives up; a byte moving starts the count over.
    #[tokio::test(start_paused = true)]
    async fn it_gives_up_only_when_every_lane_failed_since_a_byte_moved() {
        let tries = Arc::new(AtomicU64::new(0));
        let counted = tries.clone();
        let done = transfer::<(), _, _>([(0, 10)], 2, move |leg| {
            let attempt = counted.fetch_add(1, Ordering::Relaxed);
            async move {
                if attempt == 1 {
                    leg.at.store(leg.first + 1, Ordering::Relaxed);
                }
                Err(Ranged::Again(format!("try {attempt}")))
            }
        })
        .await;
        assert!(matches!(done, Err(Ranged::Again(m)) if m == "try 3"));
        assert_eq!(tries.load(Ordering::Relaxed), 4);
    }

    /// A leg moving `per_second` bytes a second, a tenth of a second at a time.
    async fn moving(leg: Leg, per_second: u64) -> Result<(), Ranged<()>> {
        let mut at = leg.first;
        while at < leg.end {
            tokio::time::sleep(Duration::from_millis(100)).await;
            at = (at + per_second / 10).min(leg.end);
            leg.at.store(at, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Lanes that won slow start finish fast and leave two that shared the path with them at a
    /// tenth of their rate: neither is slower than the other, so neither is asked again (they
    /// were, when finished ranges' rates stood in for peers'). A lone range at the tail that is
    /// far slower than the finished ones is a straggler, and is.
    #[tokio::test(start_paused = true)]
    async fn ranges_are_judged_against_peers_in_flight_and_a_lone_straggler_against_finished_ones()
    {
        let legs = Arc::new(std::sync::Mutex::new(vec![]));
        let seen = legs.clone();
        let ranges = (0..4).map(|i| (i * 10_000, (i + 1) * 10_000));
        let done = transfer::<(), _, _>(ranges, 4, move |leg| {
            seen.lock().unwrap().push((leg.first, leg.fresh));
            let per_second = if leg.first < 20_000 { 10_000 } else { 1_000 };
            moving(leg, per_second)
        })
        .await;
        assert!(done.is_ok());
        let legs = legs.lock().unwrap().clone();
        assert_eq!(legs.len(), 4, "no range asked again: {legs:?}");

        let legs = Arc::new(std::sync::Mutex::new(vec![]));
        let seen = legs.clone();
        let done = transfer::<(), _, _>([(0, 10_000), (10_000, 20_000)], 2, move |leg| {
            let mut seen = seen.lock().unwrap();
            seen.push((leg.first, leg.fresh));
            let per_second = match seen.len() {
                // The straggler, until it is asked again.
                2 => 500,
                _ => 10_000,
            };
            moving(leg, per_second)
        })
        .await;
        assert!(done.is_ok());
        let legs = legs.lock().unwrap().clone();
        let again = legs.iter().skip(2).find(|(first, _)| *first > 10_000);
        assert!(
            again.is_some(),
            "the straggler was asked again from where it stopped: {legs:?}"
        );
    }
}

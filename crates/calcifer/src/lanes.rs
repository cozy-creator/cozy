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
    let missing = protocol::machine::missing(held, length);
    let total = protocol::machine::held_bytes(&missing);
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

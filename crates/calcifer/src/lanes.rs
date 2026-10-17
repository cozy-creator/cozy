//! Ranged transfers side by side, as an S3 client makes them: each range on its own lane (its own
//! connection), judged by its measured bytes against its peers'. A range that fails, or whose
//! bytes come `SLOW` times slower than its peers', goes again from where its bytes stopped on a
//! fresh connection. A transfer gives up only when every lane has failed since a byte last moved.
//! No flat timeouts: only lack of measured progress ends anything.
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use tokio::task::{AbortHandle, Id, JoinSet};

/// A range whose bytes come this many times slower than its peers' is asked again.
const SLOW: f64 = 8.0;
/// How often ranges in flight are measured, and the shortest window a rate is judged over.
const SAMPLE: Duration = Duration::from_secs(1);
const WINDOW: Duration = Duration::from_secs(2);

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
    /// Since its first byte: the start of the window its rate is measured over, and `at` then.
    window: Option<(Instant, u64)>,
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
                window: None,
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
/// than its peers' did, or whose bytes come `SLOW` times slower than its peers' (finished ranges
/// included) over its latest window; its rest is asked again. A range whose bytes have all moved
/// is waiting for its answer and is not judged.
fn judge(flights: &mut HashMap<Id, Flight>, finished: &[f64], waits: &mut Vec<Duration>) {
    let now = Instant::now();
    let mut rates = finished.to_vec();
    let mut judged = vec![];
    for flight in flights.values_mut() {
        let at = flight.at.load(Ordering::Relaxed);
        match flight.window {
            _ if at >= flight.end => {}
            None if at > flight.first => {
                waits.push(now - flight.began);
                flight.window = Some((now, at));
            }
            None => {
                waits.sort();
                let typical = waits.get(waits.len() / 2).map(|wait| wait.mul_f64(SLOW));
                flight.slow |= typical.is_some_and(|t| now - flight.began > t.max(WINDOW));
            }
            Some((since, seen)) if now - since >= WINDOW => {
                let rate = (at - seen) as f64 / (now - since).as_secs_f64();
                flight.window = Some((now, at));
                rates.push(rate);
                judged.push((flight.task.id(), rate));
            }
            Some(_) => {}
        }
    }
    rates.sort_by(f64::total_cmp);
    let typical = (rates.len() >= 2).then(|| rates[rates.len() / 2]);
    for (id, rate) in judged {
        if let (Some(typical), Some(flight)) = (typical, flights.get_mut(&id)) {
            flight.slow |= rate * SLOW < typical;
        }
    }
    for flight in flights.values().filter(|flight| flight.slow) {
        flight.task.abort();
    }
}

/// What `held` leaves of `[0, length)`, cut into parts of `clamp(missing / lanes, MIN, MAX)`
/// bytes.
pub(crate) fn parts(held: &[(u64, u64)], length: u64, lanes: usize) -> Vec<(u64, u64)> {
    const MIN: u64 = 1 << 20;
    const MAX: u64 = 64 << 20;
    let missing = cozy_proto::machine::missing(held, length);
    let total = cozy_proto::machine::held_bytes(&missing);
    let size = (total / lanes as u64).clamp(MIN, MAX);
    missing
        .into_iter()
        .flat_map(|(a, b)| (a..b).step_by(size as usize).map(move |x| (x, (x + size).min(b))))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    #[test]
    fn missing_bytes_are_cut_into_parts_a_lane_each() {
        assert_eq!(
            parts(&[], 20 * MIB, 4),
            (0..4).map(|i| (i * 5 * MIB, (i + 1) * 5 * MIB)).collect::<Vec<_>>()
        );
        assert_eq!(parts(&[(0, 100)], 100, 4), vec![], "all held");
        assert_eq!(
            parts(&[(0, 10), (20, 30)], 3 * MIB, 4),
            vec![(10, 20), (30, MIB + 30), (MIB + 30, 2 * MIB + 30), (2 * MIB + 30, 3 * MIB)],
            "gaps, then 1 MiB parts at least"
        );
        assert_eq!(
            parts(&[], 1024 * MIB, 4).len(),
            16,
            "64 MiB parts at most"
        );
    }

    /// A range that fails resumes from where its bytes stopped on a fresh lane; one that never
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
        assert!(legs.contains(&(50, 100, true)), "{legs:?}");
        assert!(legs.contains(&(200, 300, true)), "{legs:?}");
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
}

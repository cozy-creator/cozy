//! The one rule for asking a remote again (tfs-049), the Endpoint's exchanges' and every body
//! GET outside the Fetcher's: weather and transport failures are asked again at `Retry-After` or a
//! doubling wait, a request is dropped once nothing has moved for the stall rule, and the asking
//! ends once nothing has moved for it at all. No flat timeout: every verdict scales with what was
//! measured.

use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::net::{Client, GetError, HeaderMap};
use super::token::TokenProvider;
use crate::err::{Code, Refusal};

/// The stall rule's sampling resolution.
pub(crate) const TICK: Duration = Duration::from_millis(250);
/// Stillness under 120 ticks is no evidence; for 8× the slowest answer seen, it is.
const STILL_TICKS: u32 = 120;
const STALL_FACTOR: u32 = 8;

/// An answer that means "not now". 501 and 505 say the ask can never be answered.
pub(crate) fn weather(status: u16) -> bool {
    matches!(status, 408 | 425 | 429 | 500..=599) && !matches!(status, 501 | 505)
}

/// What the stall rule scales: its tick, and the slowest answer seen.
pub(crate) struct Patience {
    tick: Duration,
    slowest: AtomicU64,
}

impl Patience {
    pub(crate) fn new(tick: Duration) -> Patience {
        Patience { tick, slowest: AtomicU64::new(0) }
    }

    fn get(&self) -> Duration {
        let slowest = Duration::from_nanos(self.slowest.load(Ordering::Relaxed));
        (self.tick * STILL_TICKS).max(slowest * STALL_FACTOR)
    }
}

impl Default for Patience {
    fn default() -> Patience {
        Patience::new(TICK)
    }
}

/// An ask answered short of a refusal: done, or not now (at `wait` when the answer stated one).
pub(crate) enum Tried<T> {
    Done(T),
    Again { why: String, wait: Option<Duration> },
}

/// `ask` until it is done. Each ask gets the counter it bumps as anything moves and watches its
/// exchanges with it ([`watched`]). Weather and transport failures are asked again; bytes that
/// moved before a failure are progress, a "not now" is not. Ends `HUB_UNREACHABLE` once nothing
/// has moved for the stall rule (or twice the longest stated wait); the caller's own verdict and
/// any other refusal end it at once.
pub(crate) async fn again<T, A, F>(patience: &Patience, mut ask: A) -> Result<T, GetError>
where
    A: FnMut(Arc<AtomicU64>) -> F,
    F: Future<Output = Result<Tried<T>, GetError>>,
{
    let (mut still, mut wait, mut stated_most) = (Instant::now(), patience.tick, Duration::ZERO);
    loop {
        let (moved, began) = (Arc::new(AtomicU64::new(0)), Instant::now());
        let (why, stated) = match ask(Arc::clone(&moved)).await {
            Ok(tried) => {
                patience.slowest.fetch_max(began.elapsed().as_nanos() as u64, Ordering::Relaxed);
                match tried {
                    Tried::Done(done) => return Ok(done),
                    Tried::Again { why, wait } => (why, wait),
                }
            }
            Err(GetError::Exchange(e)) if matches!(e.code, Code::TRANSFER_FAILED | Code::FD_HEADROOM | Code::STALLED) => {
                if moved.load(Ordering::Relaxed) > 0 {
                    still = Instant::now();
                }
                (e.detail, None)
            }
            Err(e) => return Err(e),
        };
        stated_most = stated_most.max(stated.unwrap_or_default());
        let patience = patience.get().max(stated_most * 2);
        if still.elapsed() > patience {
            let detail = format!("{why}; nothing moved for {:.1} s", still.elapsed().as_secs_f64());
            return Err(GetError::Exchange(Refusal { code: Code::HUB_UNREACHABLE, detail }));
        }
        tracing::debug!(%why, "asked again");
        tokio::time::sleep(stated.unwrap_or(wait)).await;
        wait = (wait * 2).min(patience / 8);
    }
}

/// `exchange`, dropped as `STALLED` once `moved` stands still for the stall rule.
pub(crate) async fn watched<T, E: From<Refusal>>(
    patience: &Patience,
    moved: &AtomicU64,
    exchange: impl Future<Output = Result<T, E>>,
) -> Result<T, E> {
    tokio::pin!(exchange);
    let (mut seen, mut since) = (moved.load(Ordering::Relaxed), Instant::now());
    loop {
        tokio::select! {
            done = &mut exchange => return done,
            () = tokio::time::sleep(patience.tick) => {
                let now = moved.load(Ordering::Relaxed);
                if now != seen {
                    (seen, since) = (now, Instant::now());
                } else if since.elapsed() > patience.get() {
                    let detail = format!("nothing moved for {:.1} s", since.elapsed().as_secs_f64());
                    return Err(Refusal { code: Code::STALLED, detail }.into());
                }
            }
        }
    }
}

/// Where [`get`] puts a body: each frame with the byte it starts at.
pub(crate) type Sink<'a> = dyn FnMut(u64, &[u8]) -> Result<(), Refusal> + Send + 'a;

/// GET `url` (`range`: from, and through) into `sink`, which takes each body frame with the byte it
/// starts at, asked [`again`] from the byte reached; a 200 to an open-ended ask starts the body
/// over. The answer's headers, or the status of an answer neither data nor weather.
pub(crate) async fn get(
    client: &Client,
    url: &str,
    headers: &[(String, String)],
    provider: Option<&dyn TokenProvider>,
    patience: &Patience,
    range: Option<(u64, Option<u64>)>,
    sink: &mut Sink<'_>,
) -> Result<Result<HeaderMap, u16>, GetError> {
    let (from, through) = range.unwrap_or((0, None));
    let (reached, sink) = (&AtomicU64::new(from), &Mutex::new(sink));
    again(patience, move |moved| async move {
        let mut at = reached.load(Ordering::Relaxed);
        let mut headers = headers.to_vec();
        if at > 0 || through.is_some() {
            let through = through.map_or(String::new(), |t| t.to_string());
            headers.push(("range".into(), format!("bytes={at}-{through}")));
        }
        watched(patience, &moved, async {
            let mut answer = client.get_authorized(url, &headers, provider).await?;
            moved.fetch_add(1, Ordering::Relaxed);
            let status = answer.status();
            match status {
                _ if weather(status) => return Ok(Tried::Again { why: format!("GET {url}: {status}"), wait: answer.stated_wait() }),
                200 if through.is_none() => at = 0,
                206 if answer.content_range().is_some_and(|(first, _)| first == at) => {}
                _ => return Ok(Tried::Done(Err(status))),
            }
            while let Some(frame) = answer.frame().await? {
                (sink.lock().unwrap())(at, &frame).map_err(GetError::Caller)?;
                at += frame.len() as u64;
                reached.store(at, Ordering::Relaxed);
                moved.fetch_add(frame.len() as u64, Ordering::Relaxed);
            }
            Ok(Tried::Done(Ok(answer.response.headers().clone())))
        })
        .await
    })
    .await
}

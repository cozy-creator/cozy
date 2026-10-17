//! One machine's API through `cozy_proto::machine::client`: TLS that admits only the machine's
//! pinned leaf, every request carrying a fresh machine-scope Cozy-Cap signed by the key the
//! machine scopes its runs to. Control and SSE share one HTTP/2 connection; inputs upload as
//! ranged PUTs side by side on HTTP/1.1 lanes of their own.
use std::{
    path::Path,
    pin::Pin,
    sync::{Arc, Mutex, atomic::Ordering},
    time::{Duration, Instant},
};

use cozy_proto::{
    ids::Digest,
    machine::{
        self as v1,
        client::{Auth, Bulk, Error, Machine},
    },
};
use ed25519_dalek::SigningKey;
use futures_util::{Stream, StreamExt as _, TryStreamExt as _};
use tokio::io::{AsyncReadExt as _, AsyncSeekExt as _};

use crate::{
    lanes::{self, Leg, Ranged},
    store,
};

/// The refusal of a machine from before this API (cozy v0.9's gRPC), which is never driven.
pub const OUTDATED: &str = "machine_outdated";
/// Parts of an upload in flight side by side, each on its own connection.
const LANES: usize = 4;
/// How often an upload asks the machine what it holds, which is its progress.
const TELL_EVERY: Duration = Duration::from_secs(1);
/// What an upload reads from its file at a time.
const FRAME: usize = 64 << 10;

#[derive(Clone, Debug, thiserror::Error)]
pub enum LinkError {
    #[error("the machine has no such run")]
    NotFound,
    /// A refusal the same request cannot change: the machine will not take it.
    #[error("the machine refused it ({code}): {message}")]
    Refused { code: String, message: String },
    /// The machine released itself (a rental's idle end): it is gone.
    #[error("the machine released itself: {0}")]
    Released(String),
    /// The machine does not admit the key (yet).
    #[error("the machine does not admit this key: {0}")]
    Unauthorized(String),
    /// Unreachable, starting, updating, or not admitting the key yet: try again.
    #[error("{0}")]
    Unreachable(String),
}

impl From<Error> for LinkError {
    fn from(e: Error) -> Self {
        match e {
            Error::Refused { status: 404, .. } => LinkError::NotFound,
            Error::Refused { error, .. } if error.code == "machine_released" => {
                LinkError::Released(error.message)
            }
            Error::Refused { status: 401, error } => LinkError::Unauthorized(error.message),
            Error::Refused { status, error } if status < 500 => LinkError::Refused {
                code: error.code,
                message: error.message,
            },
            Error::Refused { error, .. } => {
                LinkError::Unreachable(format!("{}: {}", error.code, error.message))
            }
            Error::Transport(message) => LinkError::Unreachable(message),
            Error::NotMachineApi(message) => LinkError::Refused {
                code: OUTDATED.into(),
                message,
            },
        }
    }
}

/// What a machine streams: a run's events after its cursor, or its Status frames.
pub type Live<T> = Pin<Box<dyn Stream<Item = Result<T, LinkError>> + Send>>;

#[derive(Clone)]
pub struct Link {
    machine: Machine,
    /// The upload lanes, kept across objects: a project's many small files reuse connections.
    lanes: Arc<Mutex<Vec<Option<Bulk>>>>,
}

impl Link {
    /// The machine at `endpoint`, connected on first use and again whenever it breaks.
    pub fn new(endpoint: &store::Endpoint, key: SigningKey) -> Result<Self, LinkError> {
        let auth = Auth::Key {
            key: Arc::new(key),
            worker: endpoint.worker_id.clone(),
        };
        Ok(Self {
            machine: Machine::new(&endpoint.address, endpoint.pin, auth)?,
            lanes: Arc::new(Mutex::new(vec![None; LANES])),
        })
    }

    /// A bulk lane of its own: HTTP/1.1 connections apart from the control one.
    pub fn bulk(&self) -> Result<Bulk, LinkError> {
        Ok(self.machine.bulk()?)
    }

    /// Upload lane `i`, connected again when `fresh`.
    fn lane(&self, i: usize, fresh: bool) -> Option<Bulk> {
        let mut lanes = self.lanes.lock().unwrap_or_else(|e| e.into_inner());
        if fresh || lanes[i].is_none() {
            lanes[i] = self.machine.bulk().ok();
        }
        lanes[i].clone()
    }

    /// Sends the run (with `spec`) or attaches to it, and answers its state with its events after
    /// `after`. Closing them never cancels.
    pub async fn run(
        &self,
        id: &str,
        after: u64,
        spec: Option<v1::RunSpec>,
    ) -> Result<(v1::RunState, Live<v1::RunEvent>), LinkError> {
        if let Some(spec) = spec {
            let request = v1::RunRequest {
                id: id.into(),
                spec,
            };
            self.machine.submit(&request).await?;
        }
        let mut events = Box::pin(self.machine.events(id, after).await?.map_err(LinkError::from));
        match events.next().await {
            Some(Ok(v1::RunEvent {
                event: v1::Event::State(state),
                ..
            })) => Ok((state, events)),
            Some(Err(e)) => Err(e),
            _ => Err(LinkError::Unreachable(
                "the machine's stream did not open with the run's state".into(),
            )),
        }
    }

    /// The machine's whole picture now.
    pub async fn status(&self) -> Result<v1::StatusFrame, LinkError> {
        Ok(self.machine.status().await?)
    }

    /// The machine's picture, then again whenever it changes.
    pub async fn watch(&self) -> Result<Live<v1::StatusFrame>, LinkError> {
        Ok(Box::pin(self.machine.watch().await?.map_err(LinkError::from)))
    }

    /// Starts a rental's idle clock over; answers the picture with its new deadline.
    pub async fn keepalive(&self) -> Result<v1::StatusFrame, LinkError> {
        Ok(self.machine.keepalive().await?)
    }

    /// The identity frame the machine at `address` gives anyone who reaches it: its worker id,
    /// boot and version.
    pub async fn identity(address: &str, pin: [u8; 32]) -> Result<v1::StatusFrame, LinkError> {
        Ok(Machine::new(address, pin, Auth::None)?.status().await?)
    }

    /// The last `tail` bytes (0: all) of a log the machine keeps.
    pub async fn read_log(&self, log: &str, tail: u64) -> Result<Vec<u8>, LinkError> {
        Ok(self.machine.log(log, tail).await?.to_vec())
    }

    /// Asks the machine to cancel the run. A machine that never received it records it canceled,
    /// so a late send of it is answered with that.
    pub async fn cancel(&self, id: &str) -> Result<v1::RunState, LinkError> {
        Ok(self.machine.cancel(id).await?)
    }

    /// How many bytes of the object the machine holds for this signer.
    pub async fn held(&self, digest: &Digest, length: u64) -> Result<u64, LinkError> {
        let held = self.machine.held(&digest.to_string()).await?;
        Ok(v1::held_bytes(&held).min(length))
    }

    /// Uploads the object at `path`: what the machine does not hold yet, in parts side by side
    /// (`lanes`). `progress` hears what the machine holds, from a HEAD about once a second while
    /// parts fly, with the rate; an object the machine already holds is not written and
    /// `progress` hears nothing. A failed or stalled part resumes on a fresh connection; the
    /// upload gives up only when every lane has failed since a byte last moved, or a whole pass
    /// landed nothing.
    pub async fn write(
        &self,
        digest: &Digest,
        length: u64,
        path: &Path,
        mut progress: impl FnMut(u64, Option<f64>),
    ) -> Result<(), LinkError> {
        let name = digest.to_string();
        let mut held = self.machine.held(&name).await?;
        if v1::held_bytes(&held) >= length {
            return Ok(());
        }
        let (from, started) = (v1::held_bytes(&held), Instant::now());
        let rate = |now: u64| {
            let seconds = started.elapsed().as_secs_f64().max(1e-3);
            Some(now.saturating_sub(from) as f64 / seconds)
        };
        progress(from, None);
        loop {
            let before = v1::held_bytes(&held);
            let parts = lanes::parts(&held, length, LANES);
            let sent = lanes::transfer(parts, LANES, |leg| {
                let lane = self.lane(leg.lane, leg.fresh);
                put(lane, name.clone(), path.to_owned(), length, leg)
            });
            let tell = async {
                let mut tick = tokio::time::interval(TELL_EVERY);
                tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                tick.tick().await;
                loop {
                    tick.tick().await;
                    if let Ok(now) = self.machine.held(&name).await {
                        let now = v1::held_bytes(&now);
                        progress(now, rate(now));
                    }
                }
            };
            let sent = tokio::select! {
                sent = sent => sent,
                _ = tell => unreachable!("it tells until the parts end"),
            };
            held = self.machine.held(&name).await?;
            let now = v1::held_bytes(&held);
            if now >= length {
                progress(length, rate(length));
                return Ok(());
            }
            match sent {
                Err(Ranged::Stop(e)) => return Err(e),
                Err(Ranged::Again(message)) if now == before => {
                    return Err(LinkError::Unreachable(message));
                }
                _ if now == before => {
                    return Err(LinkError::Unreachable(format!(
                        "the machine holds {now} of the {length} bytes of {digest} and took no more"
                    )));
                }
                _ => {}
            }
        }
    }
}

/// Lands one part from `path` on `lane`, moving `leg.at` past each byte handed to it.
async fn put(
    lane: Option<Bulk>,
    digest: String,
    path: std::path::PathBuf,
    total: u64,
    leg: Leg,
) -> Result<(), Ranged<LinkError>> {
    let lane = lane.ok_or_else(|| Ranged::Again("no connection to the machine".into()))?;
    let unreadable = |e: std::io::Error| {
        Ranged::Stop(LinkError::Refused {
            code: "input_unreadable".into(),
            message: format!("{}: {e}", path.display()),
        })
    };
    let mut file = tokio::fs::File::open(&path).await.map_err(unreadable)?;
    file.seek(std::io::SeekFrom::Start(leg.first))
        .await
        .map_err(unreadable)?;
    let length = leg.end - leg.first;
    let at = leg.at.clone();
    let body = tokio_util::io::ReaderStream::with_capacity(file.take(length), FRAME)
        .inspect_ok(move |bytes| {
            at.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        });
    match lane
        .put(&digest, (leg.first, length, total), reqwest::Body::wrap_stream(body))
        .await
    {
        Ok(_) => Ok(()),
        Err(e) => match LinkError::from(e) {
            LinkError::Unreachable(message) => Err(Ranged::Again(message)),
            e => Err(Ranged::Stop(e)),
        },
    }
}

/// The TLS leaf (DER) the machine at `address` presents, whatever it is.
pub async fn presented_leaf(address: &str) -> Result<Vec<u8>, LinkError> {
    Ok(cozy_proto::machine::client::presented_leaf(address).await?)
}

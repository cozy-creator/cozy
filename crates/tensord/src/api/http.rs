//! The machine API over HTTP (G/API.md, design/machine-http-api.md): one axum Router behind the
//! pinned-TLS listener, every call authorized by one `Cozy-Cap`, every refusal an
//! `error::Envelope`.
use super::{
    auth::{StreamAuthority, VerifiedActor},
    backend::{MachineBackend, OutputSnapshot},
    capability::{self, Grant},
    domain, v1, MachineIdentity,
};
use platform::tail as logs;
use axum::{
    body::{Body, Bytes},
    extract::{rejection::QueryRejection, DefaultBodyLimit, Path, Query, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post, put},
    Json, Router,
};
use protocol::error::{Envelope, Error};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    io::{Read, Seek, SeekFrom},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio_stream::{Stream, StreamExt};
use v1::path;

/// A refusal as its envelope, at the status its kind names unless the protocol names another.
#[derive(Debug)]
pub struct Refusal(pub Error, Option<StatusCode>);
impl From<Error> for Refusal {
    fn from(error: Error) -> Self {
        Self(error, None)
    }
}
impl From<crate::objects::Refused> for Error {
    fn from(refusal: crate::objects::Refused) -> Self {
        refused(refusal.code, refusal.message)
    }
}
impl IntoResponse for Refusal {
    fn into_response(self) -> Response {
        let status = self.1.unwrap_or_else(|| {
            StatusCode::from_u16(self.0.kind.http_status()).unwrap_or(StatusCode::BAD_REQUEST)
        });
        (status, Json(Envelope { error: self.0 })).into_response()
    }
}

/// A typed refusal of the run layer (`runs`, `objects`) as the error its code names.
pub(crate) fn refused(code: &str, message: impl Into<String>) -> Error {
    match code {
        "run_id_conflict" => Error::conflict(code, message),
        "machine_disk_full" => Error::capacity(code, message),
        "object_storage_failed" | "machine_released" | "machine_updating" => {
            Error::unavailable(code, message)
        }
        c if c.starts_with("invalid") || c.starts_with("object_") => Error::validation(code, message),
        _ => Error::conflict(code, message),
    }
}

type Answer = Result<Response, Refusal>;

pub(super) struct Machine<B> {
    identity: Arc<MachineIdentity>,
    backend: Arc<B>,
}
impl<B> Clone for Machine<B> {
    fn clone(&self) -> Self {
        Self { identity: self.identity.clone(), backend: self.backend.clone() }
    }
}

pub(super) fn router<B: MachineBackend>(identity: Arc<MachineIdentity>, backend: Arc<B>) -> Router {
    Router::new()
        .route(path::STATUS, get(status::<B>))
        .route(path::KEEPALIVE, post(keepalive::<B>))
        .route(path::RUNS, post(submit::<B>).layer(DefaultBodyLimit::max(16 << 20)))
        .route(path::EVENTS, get(events::<B>))
        .route(path::CANCEL, post(cancel::<B>))
        .route(path::OUTPUT, get(output::<B>))
        .route(path::TRIAGE, get(triage::<B>))
        .route(path::LOG, get(log::<B>))
        .route(path::OBJECT, put(put_object::<B>).head(head_object::<B>).get(follow_object::<B>))
        .route(path::OBJECTS_HELD, post(held_objects::<B>).layer(DefaultBodyLimit::max(16 << 20)))
        .route(path::BEARER, put(bearer::<B>))
        .route(path::HEALTH, get(health))
        .fallback(|| async { Refusal::from(Error::not_found("route_not_found", "this machine serves no such path")) })
        .method_not_allowed_fallback(|| async {
            Refusal(Error::validation("method_not_allowed", "this path takes another method"), Some(StatusCode::METHOD_NOT_ALLOWED))
        })
        .with_state(Machine { identity, backend })
}

/// Who may do what: a machine-scope cap does everything as its signer; a run-scope cap (a play
/// link) may only read its run's events and its granted outputs, until its expiry.
pub(super) struct Caller {
    pub actor: VerifiedActor,
    grant: Grant,
}
impl Caller {
    fn machine(&self) -> Result<(), Error> {
        match self.grant.action == capability::MACHINE {
            true => Ok(()),
            false => Err(Error::forbidden("capability_scope", "this call needs a machine-scope Cozy-Cap")),
        }
    }
    /// An output-limited run grant (`run play`'s link), which sees only those outputs.
    pub(super) fn limit(&self) -> Option<Grant> {
        let limited = self.grant.action != capability::MACHINE && !self.grant.outputs.is_empty();
        limited.then(|| self.grant.clone())
    }
    fn run(&self, id: &str, output: Option<(&str, Option<u32>)>) -> Result<(), Error> {
        let allowed = self.grant.action == capability::MACHINE
            || match output {
                Some((name, index)) => self.grant.allows(id, name, index),
                None => self.grant.run == id,
            };
        match allowed {
            true => Ok(()),
            false => Err(Error::forbidden("capability_scope", "the capability does not grant this")),
        }
    }
}

impl<B: MachineBackend> Machine<B> {
    fn caller(&self, headers: &HeaderMap) -> Result<Caller, Error> {
        let token = headers
            .get(header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix(capability::SCHEME))
            .ok_or_else(|| Error::credential("capability_absent", "the call carries no Cozy-Cap"))?;
        let authority = &self.identity.authority;
        let (grant, signer) = authority
            .keys
            .check(|keys| capability::verify(token, &authority.worker_id, keys, unix_now(), ""))
            .map_err(|refusal| Error::credential("capability_invalid", refusal.to_string()))?;
        Ok(Caller { actor: VerifiedActor { public_key: signer.to_bytes() }, grant })
    }

    /// A response stream lives until its key stops authorizing; a play link's also until its
    /// expiry. A machine cap's expiry held at open: a long run, upload or read outlives it.
    pub(super) fn authority(&self, caller: &Caller) -> StreamAuthority {
        let expires = (caller.grant.action != capability::MACHINE).then_some(caller.grant.expires);
        StreamAuthority::new(self.identity.authority.keys.clone(), caller.actor, expires)
    }

    /// Holds idle release, and an update's activation, while a call that may start work runs.
    fn admit(&self) -> Result<Option<crate::machine::lifecycle::Admission>, Error> {
        self.identity.lifecycle.as_ref().map(|l| l.admit()).transpose()
    }

    pub(super) async fn call<T: Send + 'static>(
        &self,
        f: impl FnOnce(&B) -> Result<T, Error> + Send + 'static,
    ) -> Result<T, Error> {
        blocking(&self.backend, f).await
    }
}

/// `f` on the blocking pool: the backend's journal and files are synchronous.
pub(super) async fn blocking<B: Send + Sync + 'static, T: Send + 'static>(
    backend: &Arc<B>,
    f: impl FnOnce(&B) -> Result<T, Error> + Send + 'static,
) -> Result<T, Error> {
    let backend = backend.clone();
    tokio::task::spawn_blocking(move || f(&backend))
        .await
        .map_err(|_| Error::internal("machine_operation_stopped", "machine operation stopped"))?
}

fn unix_now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |since| since.as_secs() as i64)
}

fn query<T: DeserializeOwned>(query: Result<Query<T>, QueryRejection>) -> Result<T, Error> {
    query.map(|Query(query)| query).map_err(|e| Error::validation("invalid_request", e.body_text()))
}

fn json<T: Serialize>(status: StatusCode, value: &T) -> Response {
    (status, Json(value)).into_response()
}

/// An SSE response of `events`, a comment every 15 s so idle NAT mappings stay open, ended (not
/// closed cleanly) once the caller's authority lapses.
pub(super) fn sse<S>(authority: StreamAuthority, events: S) -> Response
where
    S: Stream<Item = Result<Event, Error>> + Send + 'static,
{
    Sse::new(authority.wrap(Box::pin(events)))
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response()
}

fn wants_events(headers: &HeaderMap) -> bool {
    headers
        .get(header::ACCEPT)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|accept| accept.contains("text/event-stream"))
}

/// `GET /v1/status`: anyone gets the identity frame with the sealed readiness receipt; a
/// machine-scope cap gets the whole picture, as SSE (sent again on change) when it asks.
async fn status<B: MachineBackend>(State(m): State<Machine<B>>, headers: HeaderMap) -> Answer {
    let caller = match headers.contains_key(header::AUTHORIZATION) {
        false => None,
        true => Some(m.caller(&headers)?).filter(|c| c.machine().is_ok()),
    };
    let Some(caller) = caller else {
        return Ok(json(StatusCode::OK, &super::machine_status::identity_frame(&m.identity)));
    };
    if !wants_events(&headers) {
        let frame = super::machine_status::frame(&m.identity, &m.backend, caller.actor).await?;
        return Ok(json(StatusCode::OK, &frame));
    }
    let frames = super::machine_status::watch(m.identity.clone(), m.backend.clone(), caller.actor)
        .map(|frame| frame.and_then(|frame| event("status", &frame)));
    Ok(sse(m.authority(&caller), frames))
}

/// One SSE event named `name` whose data is `value`.
pub(super) fn event<T: Serialize>(name: &str, value: &T) -> Result<Event, Error> {
    Event::default()
        .event(name)
        .json_data(value)
        .map_err(|e| Error::internal("event_unencodable", e.to_string()))
}

/// `POST /v1/keepalive`: restarts the idle clock once; answers the frame with the new deadline.
async fn keepalive<B: MachineBackend>(State(m): State<Machine<B>>, headers: HeaderMap) -> Answer {
    let caller = m.caller(&headers)?;
    caller.machine()?;
    m.identity
        .lifecycle
        .as_ref()
        .ok_or_else(|| Error::conflict("idle_deadline_absent", "this machine has no idle deadline"))?
        .renew()?;
    let frame = super::machine_status::frame(&m.identity, &m.backend, caller.actor).await?;
    Ok(json(StatusCode::OK, &frame))
}

/// `PUT /v1/bearer`: the signer's newest ordinary bearer at the Hub `endpoint` names. Every later
/// read there under its owner's access presents it, until one that expires later comes.
async fn bearer<B: MachineBackend>(State(m): State<Machine<B>>, headers: HeaderMap, body: Bytes) -> Answer {
    let caller = m.caller(&headers)?;
    caller.machine()?;
    let given: v1::TensorFSOptions = serde_json::from_slice(&body)
        .map_err(|e| Error::validation("invalid_request", format!("the bearer is not JSON: {e}")))?;
    let actor = crate::machine_api::actor_id(caller.actor);
    crate::hub::valid_origin(&given.endpoint)
        .then(|| crate::hub::Bearer::offer(&actor, &given.endpoint, &given.token, given.expires_unix_ms))
        .flatten()
        .ok_or_else(|| Error::validation("invalid_bearer", "a bearer names its Hub's origin and a token"))?;
    Ok(StatusCode::NO_CONTENT.into_response())
}

fn run_id(id: &str) -> Result<(), Error> {
    match id.is_empty() || id.len() > 256 {
        true => Err(Error::validation("invalid_run_id", "a run id is 1-256 bytes")),
        false => Ok(()),
    }
}

/// `POST /v1/runs`: accepts a run (201), or answers the one the same id and spec name (200).
/// A new run is admitted before it is accepted: an update's activation and a rental's idle
/// release wait for it.
async fn submit<B: MachineBackend>(State(m): State<Machine<B>>, headers: HeaderMap, body: Bytes) -> Answer {
    let caller = m.caller(&headers)?;
    caller.machine()?;
    let request: v1::RunRequest = serde_json::from_slice(&body)
        .map_err(|e| Error::validation("invalid_request", format!("the run request is not JSON: {e}")))?;
    run_id(&request.id)?;
    let (created, state) = match request.spec.kind {
        v1::RunKind::Update => {
            super::machine_update::submit(m.identity.clone(), m.backend.clone(), caller.actor, request).await?
        }
        _ => {
            let admitted = m.admit()?;
            let accepted = super::machine_runs::submit(&m.backend, caller.actor, request).await?;
            drop(admitted);
            accepted
        }
    };
    let status = if created { StatusCode::CREATED } else { StatusCode::OK };
    Ok(json(status, &state))
}

#[derive(Deserialize)]
struct After {
    #[serde(default)]
    after: u64,
}

/// `GET /v1/runs/{id}/events`: the run's log as SSE after `Last-Event-ID` (or `?after=`), its
/// snapshot first. Closing it never cancels; it holds no admission.
async fn events<B: MachineBackend>(
    State(m): State<Machine<B>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    after: Result<Query<After>, QueryRejection>,
) -> Answer {
    let caller = m.caller(&headers)?;
    caller.run(&id, None)?;
    let last = headers.get("last-event-id").and_then(|v| v.to_str().ok()?.trim().parse().ok());
    let after = last.unwrap_or(query(after)?.after);
    let (identity, actor, owned) = (m.identity.clone(), caller.actor, id.clone());
    if m.call(move |backend| super::machine_update::owns(&identity, backend, actor, &owned)).await? {
        caller.machine()?;
        let events = super::machine_update::events(m.identity.clone(), id, after).await?;
        return Ok(sse(m.authority(&caller), events.map(|e| e.and_then(|e| run_event(&e)))));
    }
    let events = super::machine_runs::follow(m.backend.clone(), caller.actor, id, after, caller.limit()).await?;
    let events = tokio_stream::wrappers::ReceiverStream::new(events);
    Ok(sse(m.authority(&caller), events.map(|e| e.and_then(|e| run_event(&e)))))
}

/// A run event as SSE: named by its variant, its sequence as the id when it is durable.
pub(super) fn run_event(event: &v1::RunEvent) -> Result<Event, Error> {
    let name = event.event.name();
    let durable = !matches!(name, "state" | "progress" | "gpu") && event.sequence > 0;
    let sse = self::event(name, event)?;
    Ok(if durable { sse.id(event.sequence.to_string()) } else { sse })
}

/// `POST /v1/runs/{id}/cancel`. An unknown id is tombstoned as canceled.
async fn cancel<B: MachineBackend>(State(m): State<Machine<B>>, Path(id): Path<String>, headers: HeaderMap) -> Answer {
    let caller = m.caller(&headers)?;
    caller.machine()?;
    run_id(&id)?;
    let (identity, actor, owned) = (m.identity.clone(), caller.actor, id.clone());
    if m.call(move |backend| super::machine_update::owns(&identity, backend, actor, &owned)).await? {
        return Ok(json(StatusCode::OK, &super::machine_update::cancel(&m.identity, &id).await?));
    }
    let run = id.clone();
    let changed = m
        .call(move |backend| {
            let execution = Some(super::machine_runs::query(backend, actor, &run)?);
            let action = domain::MachineExecutionAction::Cancel as i32;
            backend.control(actor, domain::MachineExecutionControl { execution, action })
        })
        .await?;
    Ok(json(StatusCode::OK, &super::machine_runs::state(&id, &changed)))
}

#[derive(Deserialize)]
struct Item {
    #[serde(default)]
    index: u32,
    #[serde(default)]
    member: String,
}

/// `GET /v1/runs/{id}/outputs/{output}`: an output's current bytes, ranged, readable while it
/// grows. `ETag` is its revision; `If-Match` refuses once the output moved past it.
async fn output<B: MachineBackend>(
    State(m): State<Machine<B>>,
    Path((id, output)): Path<(String, String)>,
    item: Result<Query<Item>, QueryRejection>,
    headers: HeaderMap,
) -> Answer {
    let caller = m.caller(&headers)?;
    let Item { index, member } = query(item)?;
    let index = (index > 0).then_some(index);
    caller.run(&id, Some((&output, index)))?;
    let actor = caller.actor;
    let snapshot = m
        .call(move |backend| {
            let run = backend.get(actor, super::machine_runs::query(backend, actor, &id)?)?;
            match member.as_str() {
                "" => backend.open_output(actor, run.number, &output, index),
                member => backend.open_member(actor, run.number, &output, index, member),
            }
        })
        .await?;
    let tag = format!("\"{}\"", snapshot.rev);
    let wanted = headers.get(header::IF_MATCH).and_then(|v| v.to_str().ok());
    if wanted.is_some_and(|wanted| wanted.trim() != "*" && !wanted.split(',').any(|t| t.trim() == tag)) {
        let moved = Error::conflict("output_revision_moved", format!("the output is at revision {}", snapshot.rev));
        return Err(Refusal(moved, Some(StatusCode::PRECONDITION_FAILED)));
    }
    let OutputSnapshot { parts, length, media_type, .. } = snapshot;
    let mut response = ranged(&headers, parts, length, m.authority(&caller), m.identity.lifecycle.clone())?;
    let headers = response.headers_mut();
    headers.insert(header::ETAG, HeaderValue::from_str(&tag).expect("a revision is ASCII"));
    if let Ok(media_type) = HeaderValue::from_str(&media_type) {
        headers.insert(header::CONTENT_TYPE, media_type);
    }
    Ok(response)
}

/// An output's bytes on disk: its parts in order, each with its length.
type Parts = Vec<(std::fs::File, u64)>;

/// `Range: bytes=a-b` (or `a-`, `-n`) of `length` bytes as `[start, end)`; None for the whole.
fn range(headers: &HeaderMap, length: u64) -> Result<Option<(u64, u64)>, Refusal> {
    let Some(spec) = headers.get(header::RANGE).and_then(|v| v.to_str().ok()) else {
        return Ok(None);
    };
    let unsatisfiable = || {
        let error = Error::validation("range_unsatisfiable", format!("the output is {length} bytes"));
        Refusal(error, Some(StatusCode::RANGE_NOT_SATISFIABLE))
    };
    let Some((first, last)) = spec.trim().strip_prefix("bytes=").and_then(|r| r.split_once('-')) else {
        return Err(unsatisfiable());
    };
    let number = |n: &str| n.trim().parse::<u64>().map_err(|_| unsatisfiable());
    let (start, end) = match (first.trim(), last.trim()) {
        ("", suffix) => (length.saturating_sub(number(suffix)?), length),
        (start, "") => (number(start)?, length),
        (start, last) => (number(start)?, number(last)?.saturating_add(1).min(length)),
    };
    match start < end {
        true => Ok(Some((start, end))),
        false => Err(unsatisfiable()),
    }
}

/// `parts` as 200, or the range the request names as 206, streamed from the blocking pool.
/// Each chunk the reader takes counts as the rental's use (`Lifecycle::transferred`).
fn ranged(
    headers: &HeaderMap,
    parts: Parts,
    length: u64,
    authority: StreamAuthority,
    lifecycle: Option<Arc<crate::machine::lifecycle::Lifecycle>>,
) -> Answer {
    let range = range(headers, length)?;
    let (start, end) = range.unwrap_or((0, length));
    let reader = from_offset(parts, start)
        .map_err(|e| Error::internal("output_unreadable", e.to_string()))?
        .take(end - start);
    let (sender, receiver) = tokio::sync::mpsc::channel(4);
    tokio::task::spawn_blocking(move || {
        let (mut reader, mut buffer, mut left) = (reader, vec![0; 1 << 20], end - start);
        while left > 0 {
            // Bytes that end before the revision's length fail the body, never a short success.
            let read = match reader.read(&mut buffer) {
                Ok(0) => Err(Error::internal("output_short", "the output's bytes ended before its length")),
                Ok(n) => Ok(Bytes::copy_from_slice(&buffer[..n])),
                Err(error) => Err(Error::internal("output_unreadable", error.to_string())),
            };
            left = match &read {
                Ok(bytes) => left - bytes.len() as u64,
                Err(_) => 0,
            };
            if sender.blocking_send(read).is_err() {
                return;
            }
        }
    });
    let chunks = tokio_stream::wrappers::ReceiverStream::new(receiver).map(move |chunk| {
        if let (Ok(_), Some(lifecycle)) = (&chunk, &lifecycle) {
            lifecycle.transferred();
        }
        chunk
    });
    let body = authority.wrap(Box::pin(chunks));
    let mut response = Response::new(Body::from_stream(body));
    let headers = response.headers_mut();
    headers.insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    headers.insert(header::CONTENT_LENGTH, HeaderValue::from(end - start));
    if range.is_some() {
        *response.status_mut() = StatusCode::PARTIAL_CONTENT;
        let span = format!("bytes {start}-{}/{length}", end - 1);
        response.headers_mut().insert(header::CONTENT_RANGE, HeaderValue::from_str(&span).expect("ASCII"));
    }
    Ok(response)
}

/// The bytes from `offset`. Parts before it are skipped and the part it falls in is seeked
/// into, so a resumed download costs the machine only its tail.
fn from_offset(parts: Parts, mut offset: u64) -> std::io::Result<Box<dyn Read + Send>> {
    let mut reader: Box<dyn Read + Send> = Box::new(std::io::empty());
    for (mut file, length) in parts {
        if offset >= length {
            offset -= length;
            continue;
        }
        file.seek(SeekFrom::Current(offset as i64))?;
        reader = Box::new(reader.chain(file.take(length - offset)));
        offset = 0;
    }
    Ok(reader)
}

/// `GET /v1/runs/{id}/triage`: a failed attempt's retained bundle, the run owner's only.
async fn triage<B: MachineBackend>(State(m): State<Machine<B>>, Path(id): Path<String>, headers: HeaderMap) -> Answer {
    let caller = m.caller(&headers)?;
    caller.run(&id, None)?;
    if caller.limit().is_some() {
        return Err(Error::forbidden("capability_scope", "triage is the run owner's").into());
    }
    let actor = caller.actor;
    let triage = m
        .call(move |backend| {
            let execution = Some(super::machine_runs::query(backend, actor, &id)?);
            backend.read_triage(actor, domain::MachineExecutionTriageQuery { execution, attempt_ordinal: 0 })
        })
        .await?;
    Ok(([(header::CONTENT_TYPE, "application/json")], triage.bundle_canonical_bytes).into_response())
}

#[derive(Deserialize)]
struct Last {
    #[serde(default)]
    lines: u64,
}

/// `GET /v1/logs/{name}?lines=N`: a machine log's newest N whole lines (0: all it keeps). As SSE,
/// those and then each batch as it is appended, each event's id the place after it: a
/// `Last-Event-ID` resumes there.
async fn log<B: MachineBackend>(
    State(m): State<Machine<B>>,
    Path(name): Path<String>,
    last: Result<Query<Last>, QueryRejection>,
    headers: HeaderMap,
) -> Answer {
    let caller = m.caller(&headers)?;
    caller.machine()?;
    let log = m.backend.log(&name)?;
    let start = match headers.get("last-event-id") {
        None => logs::Start::Last(query(last)?.lines),
        Some(id) => logs::Start::After(
            id.to_str()
                .ok()
                .and_then(|id| id.trim().parse().ok())
                .ok_or_else(|| Error::validation("invalid_request", "Last-Event-ID is not a log cursor"))?,
        ),
    };
    let unreadable = |e: std::io::Error| Error::internal("log_unreadable", e.to_string());
    if !wants_events(&headers) {
        let text = m.call(move |_| log.read(start).map_err(unreadable)).await?;
        return Ok(([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], text).into_response());
    }
    let follow = m
        .call(move |_| match log.follow(start) {
            Ok(follow) => Ok(follow),
            Err(e) if e.is_unsupported() => Err(Error::unavailable("log_follow_unsupported", e.to_string())),
            Err(e) => Err(Error::internal("log_unreadable", e.to_string())),
        })
        .await?;
    let (sender, lines) = tokio::sync::mpsc::channel(16);
    tokio::task::spawn_blocking(move || {
        let read = follow.run(|lines| sender.blocking_send(Ok(lines)).is_ok(), || !sender.is_closed());
        if let Err(e) = read {
            let _ = sender.blocking_send(Err(e));
        }
    });
    let lines = tokio_stream::wrappers::ReceiverStream::new(lines).map(move |lines| match lines {
        Ok(lines) => {
            let after = lines.after.to_string();
            Ok(event("lines", &v1::LogLines { after: after.clone(), text: lines.text })?.id(after))
        }
        Err(e) => Err(unreadable(e)),
    });
    Ok(sse(m.authority(&caller), lines))
}

fn runs_of<B: MachineBackend>(m: &Machine<B>) -> Result<Arc<crate::runs::Runs>, Error> {
    m.backend.runs().ok_or_else(|| Error::not_found("capability_unavailable", "this machine takes no objects"))
}

fn held(ranges: &[(u64, u64)]) -> HeaderValue {
    HeaderValue::from_str(&v1::spell_held(ranges)).expect("ranges are ASCII")
}

/// `HEAD /v1/objects/sha256:<hex>`: the ranges this signer holds (all of it once stored, its
/// roots renewed so a run sent next finds it).
async fn head_object<B: MachineBackend>(State(m): State<Machine<B>>, Path(digest): Path<String>, headers: HeaderMap) -> Answer {
    let caller = m.caller(&headers)?;
    caller.machine()?;
    let (runs, actor) = (runs_of(&m)?, crate::machine_api::actor_id(caller.actor));
    let ranges = blocking(&runs, move |runs| runs.objects.held(&actor, &digest).map_err(Into::into)).await?;
    Ok((StatusCode::OK, [(v1::HELD, held(&ranges))]).into_response())
}

#[derive(Deserialize)]
struct Length {
    length: u64,
}

/// `GET /v1/objects/sha256:<hex>?length=N` as SSE: `held` events, the bytes this signer's upload of
/// the object holds as they land (after each MiB more), ending once it holds all N. A stored object
/// answers N at once.
async fn follow_object<B: MachineBackend>(
    State(m): State<Machine<B>>,
    Path(digest): Path<String>,
    length: Result<Query<Length>, QueryRejection>,
    headers: HeaderMap,
) -> Answer {
    let caller = m.caller(&headers)?;
    caller.machine()?;
    let length = query(length)?.length;
    let (runs, actor) = (runs_of(&m)?, crate::machine_api::actor_id(caller.actor));
    let upload = blocking(&runs, move |runs| runs.objects.upload(&actor, &digest, length).map_err(Into::into)).await?;
    let mut landed = match &upload {
        Some(upload) => upload.follow(),
        None => tokio::sync::watch::channel(length).1,
    };
    let (sender, held) = tokio::sync::mpsc::channel(4);
    tokio::spawn(async move {
        // Joined while it is followed; let go once its follower is.
        let _upload = upload;
        let mut sent = None;
        loop {
            let now = *landed.borrow_and_update();
            if sent.is_none_or(|sent| now >= length || now >= sent + (1 << 20)) {
                if sender.send(event("held", &v1::HeldBytes { held: now })).await.is_err() || now >= length {
                    return;
                }
                sent = Some(now);
            }
            tokio::select! {
                changed = landed.changed() => if changed.is_err() { return },
                () = sender.closed() => return,
            }
        }
    });
    Ok(sse(m.authority(&caller), tokio_stream::wrappers::ReceiverStream::new(held)))
}

/// `POST /v1/objects/held`: `HEAD`'s answer for many objects in one request, in the order asked.
async fn held_objects<B: MachineBackend>(State(m): State<Machine<B>>, headers: HeaderMap, body: Bytes) -> Answer {
    let caller = m.caller(&headers)?;
    caller.machine()?;
    let ask: v1::HeldRequest = serde_json::from_slice(&body)
        .map_err(|e| Error::validation("invalid_request", format!("the held request is not JSON: {e}")))?;
    let (runs, actor) = (runs_of(&m)?, crate::machine_api::actor_id(caller.actor));
    let held = blocking(&runs, move |runs| {
        let held = ask.digests.into_iter().map(|digest| {
            let ranges = runs.objects.held(&actor, &digest)?;
            Ok(v1::HeldObject { digest, ranges })
        });
        held.collect::<Result<Vec<_>, crate::objects::Refused>>().map_err(Into::into)
    })
    .await?;
    Ok(json(StatusCode::OK, &v1::HeldAnswer { held }))
}

/// `PUT /v1/objects/sha256:<hex>`: lands the body at the offset its `Content-Range` names (the
/// whole object without one). What landed is synced and recorded when the body ends, however it
/// ends; the object is verified and stored once every byte is held. Uploads hold no admission.
async fn put_object<B: MachineBackend>(
    State(m): State<Machine<B>>,
    Path(digest): Path<String>,
    headers: HeaderMap,
    body: Body,
) -> Answer {
    let caller = m.caller(&headers)?;
    caller.machine()?;
    drop(m.admit()?);
    let (start, end, length) = span(&headers)?;
    let (runs, actor) = (runs_of(&m)?, crate::machine_api::actor_id(caller.actor));
    let objects = runs.clone();
    let Some(upload) = blocking(&objects, move |runs| runs.objects.upload(&actor, &digest, length).map_err(Into::into)).await? else {
        return Ok((StatusCode::NO_CONTENT, [(v1::HELD, held(&[(0, length)]))]).into_response());
    };
    let (sender, mut receiver) = tokio::sync::mpsc::channel::<Bytes>(8);
    let landing = upload.clone();
    let writer = tokio::task::spawn_blocking(move || -> Result<Vec<(u64, u64)>, crate::objects::Refused> {
        let mut at = start;
        let mut wrote = Ok(());
        while let Some(data) = receiver.blocking_recv() {
            if at + data.len() as u64 > end {
                wrote = Err(crate::objects::Refused {
                    code: "object_length_exceeded",
                    message: "the body is longer than its Content-Range".into(),
                });
                break;
            }
            if let Err(error) = landing.write_at(at, &data) {
                wrote = Err(error);
                break;
            }
            at += data.len() as u64;
        }
        drop(receiver);
        let held = runs.objects.settle(&landing)?;
        wrote.map(|()| held)
    });
    let mut frames = m.authority(&caller).wrap(Box::pin(body.into_data_stream().map(|frame| {
        frame.map_err(|e| Error::validation("request_body_failed", e.to_string()))
    })));
    let mut failed = None;
    while let Some(frame) = frames.next().await {
        match frame {
            // A refused writer drops its receiver; its refusal is the answer.
            Ok(data) => {
                if sender.send(data).await.is_err() {
                    break;
                }
            }
            Err(error) => {
                failed = Some(error);
                break;
            }
        }
    }
    drop(sender);
    let held = writer
        .await
        .map_err(|_| Error::internal("machine_operation_stopped", "machine operation stopped"))?
        .map_err(Error::from)?;
    if let Some(error) = failed {
        return Err(error.into());
    }
    Ok((StatusCode::NO_CONTENT, [(v1::HELD, self::held(&held))]).into_response())
}

/// A PUT's `[start, end)` of an object of `length` bytes: its `Content-Range`, else the whole
/// object its `Content-Length` measures.
fn span(headers: &HeaderMap) -> Result<(u64, u64, u64), Error> {
    let text = |name| headers.get(name).and_then(|v: &HeaderValue| v.to_str().ok());
    let invalid = || Error::validation("invalid_range", "a part is `Content-Range: bytes a-b/length`");
    match text(header::CONTENT_RANGE) {
        Some(range) => {
            let (span, length) = range.trim().strip_prefix("bytes ").and_then(|r| r.split_once('/')).ok_or_else(invalid)?;
            let (first, last) = span.split_once('-').ok_or_else(invalid)?;
            let number = |n: &str| n.trim().parse::<u64>().map_err(|_| invalid());
            let (start, end, length) = (number(first)?, number(last)? + 1, number(length)?);
            match start < end && end <= length {
                true => Ok((start, end, length)),
                false => Err(invalid()),
            }
        }
        None => {
            let length = text(header::CONTENT_LENGTH).and_then(|l| l.parse().ok()).ok_or_else(|| {
                Error::validation("invalid_range", "a whole object's PUT carries its Content-Length")
            })?;
            Ok((0, length, length))
        }
    }
}

/// `/v1/health`: answers a bare probe, refuses one that carries a credential.
async fn health(headers: HeaderMap) -> Result<StatusCode, Refusal> {
    match headers.contains_key(header::AUTHORIZATION) {
        true => Err(Error::credential("health_takes_no_credential", "health answers a bare probe").into()),
        false => Ok(StatusCode::NO_CONTENT),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every path and method the committed spec names is routed: none answers the router's
    /// fallbacks (`route_not_found`, 405).
    #[tokio::test]
    async fn every_path_and_method_in_the_spec_is_routed() {
        use tower::ServiceExt;
        struct Bare;
        impl MachineBackend for Bare {
            fn workspace(
                &self,
                _: VerifiedActor,
                _: domain::MachineExecutionWorkspaceQuery,
            ) -> Result<domain::MachineExecutionWorkspace, Error> {
                Ok(Default::default())
            }
        }
        let key = ed25519_dalek::SigningKey::from_bytes(&[1; 32]).verifying_key();
        let identity = MachineIdentity::ephemeral("m".into(), vec![key], vec![7; 32]).unwrap();
        let router = router(Arc::new(identity), Arc::new(Bare));
        let spec: serde_json::Value = serde_json::from_str(&v1::api::spec()).unwrap();
        let paths = spec["paths"].as_object().unwrap();
        let mut routed = 0;
        for (path, item) in paths {
            let uri = path.split('/').map(|s| if s.starts_with('{') { "x" } else { s }).collect::<Vec<_>>().join("/");
            for method in item.as_object().unwrap().keys().filter(|m| ["get", "put", "post", "head"].contains(&m.as_str())) {
                let request = axum::http::Request::builder().method(method.to_uppercase().as_str()).uri(&uri);
                let response = router.clone().oneshot(request.body(Body::empty()).unwrap()).await.unwrap();
                let status = response.status();
                let body = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
                let fallback = String::from_utf8_lossy(&body).contains("route_not_found");
                assert!(status != StatusCode::METHOD_NOT_ALLOWED && !fallback, "{method} {path}: {status}");
                routed += 1;
            }
        }
        assert_eq!(routed, 14, "the spec's routes");
    }

    /// A read from any offset answers exactly the bytes after it, across parts, and reads
    /// none before it: the parts it skips are not read at all.
    #[test]
    fn a_read_from_an_offset_reads_only_the_tail() {
        let root = std::env::temp_dir().join(format!("cm-read-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let lengths = [3u64, 0, 4 << 20, 5];
        let mut whole = vec![];
        for (i, length) in lengths.iter().enumerate() {
            let bytes: Vec<u8> = (0..*length).map(|b| (b % 251) as u8 ^ i as u8).collect();
            std::fs::write(root.join(i.to_string()), &bytes).unwrap();
            whole.extend(bytes);
        }
        let parts = || {
            let open = |i: usize| std::fs::File::open(root.join(i.to_string())).unwrap();
            lengths
                .iter()
                .enumerate()
                .map(|(i, l)| (open(i), *l))
                .collect()
        };
        let total = whole.len() as u64;
        for offset in [0, 1, 3, 4, 2 << 20, total - 5, total - 1, total] {
            let mut read = vec![];
            from_offset(parts(), offset)
                .unwrap()
                .read_to_end(&mut read)
                .unwrap();
            assert_eq!(read, whole[offset as usize..], "from {offset}");
        }
        // This thread's read syscalls carry the tail, not the 4 MiB before it (Linux counts
        // them per thread).
        if cfg!(target_os = "linux") {
            let read_bytes = || {
                let io = std::fs::read_to_string("/proc/thread-self/io").unwrap();
                let line = io.lines().find(|l| l.starts_with("rchar:")).unwrap();
                line[6..].trim().parse::<u64>().unwrap()
            };
            let opened = parts();
            let before = read_bytes();
            let mut tail = vec![];
            from_offset(opened, total - 5)
                .unwrap()
                .read_to_end(&mut tail)
                .unwrap();
            let read = read_bytes() - before;
            assert!(read < 4096, "a read of 5 bytes read {read}");
            assert_eq!(tail, whole[whole.len() - 5..]);
        }
        std::fs::remove_dir_all(root).unwrap();
    }

}

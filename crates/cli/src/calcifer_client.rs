//! The CLI's client of Calcifer: loopback HTTP/1.1 with the bearer from Calcifer's lock record,
//! typed refusals from its envelope, and a run's event stream. Any call starts Calcifer when none
//! runs, from the home's installed `cozy` only: as this user's service where a service manager
//! answers, else as a detached `cozy calcifer`. Nothing here gives up because time passed: a start
//! is waited on until Calcifer publishes its record or its refusal, and a long wait is reported.
use std::{
    fs::OpenOptions,
    io::{self, Write as _},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use platform::{
    lock::{Kind as LockKind, LockFile},
    paths::Home,
    service::{ServiceSpec, Services},
    tools,
};
use protocol::{
    calcifer::{Down, Event, Holder, MachineList, REFUSED, Run, RunList, SubmitRun, Submitted},
    error::{Envelope, Error, Kind},
    machine::LogLines,
};
use reqwest::{RequestBuilder, Response, StatusCode, header};
use serde::{Serialize, de::DeserializeOwned};

use crate::{
    Context,
    launch::{self, Ended, Launch},
    output,
};

/// How often a wait looks again: a sampling period, never a deadline.
const SAMPLE: Duration = Duration::from_millis(50);
/// How often a wait that goes on says so.
const REPORT: Duration = Duration::from_secs(5);
/// Calcifer stopping under a call, or a stream closing with nothing in it, this many times in a
/// row is a Calcifer that is not going to answer.
const AGAIN: u32 = 3;

/// One command's connection to Calcifer. Cheap to clone; clones share the connection.
#[derive(Clone)]
pub struct Client(Arc<Inner>);

struct Inner {
    http: reqwest::Client,
    home: Home,
    user_home: PathBuf,
    json: bool,
    holder: Mutex<Option<Holder>>,
}

impl Client {
    /// Calcifer, started if none runs.
    pub async fn connect(ctx: &Context) -> Result<Self, Error> {
        let client = Self::new(ctx)?;
        client.ready().await?;
        Ok(client)
    }

    /// The running Calcifer, or `None`: never starts one.
    pub async fn running(ctx: &Context) -> Result<Option<Self>, Error> {
        let client = Self::new(ctx)?;
        Ok(client.live().await.map(|holder| {
            client.remember(Some(holder));
            client
        }))
    }

    fn new(ctx: &Context) -> Result<Self, Error> {
        let http = reqwest::Client::builder()
            .no_proxy()
            .build()
            .map_err(|e| Error::new(Kind::Internal, "http_client", e.to_string()))?;
        Ok(Self(Arc::new(Inner {
            http,
            home: ctx.home.clone(),
            user_home: ctx.user_home.clone(),
            json: ctx.json,
            holder: Mutex::new(None),
        })))
    }

    /// The record of the Calcifer this client last reached.
    pub fn holder(&self) -> Option<Holder> {
        self.0.holder.lock().expect("holder").clone()
    }

    pub async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T, Error> {
        decode(self.send(|c, base| c.get(format!("{base}{path}"))).await?).await
    }

    pub async fn get_query<Q, T>(&self, path: &str, query: &Q) -> Result<T, Error>
    where
        Q: Serialize + ?Sized,
        T: DeserializeOwned,
    {
        let answer = self.send(|c, base| c.get(format!("{base}{path}")).query(query));
        decode(answer.await?).await
    }

    /// A POST; `key` is its `Idempotency-Key`, so a retry is the same ask.
    pub async fn post<B, T>(&self, path: &str, body: &B, key: Option<&str>) -> Result<T, Error>
    where
        B: Serialize + ?Sized,
        T: DeserializeOwned,
    {
        let answer = self.send(|c, base| {
            let request = c.post(format!("{base}{path}")).json(body);
            match key {
                Some(key) => request.header("Idempotency-Key", key),
                None => request,
            }
        });
        decode(answer.await?).await
    }

    pub async fn delete<T: DeserializeOwned>(&self, path: &str) -> Result<T, Error> {
        decode(
            self.send(|c, base| c.delete(format!("{base}{path}")))
                .await?,
        )
        .await
    }

    /// A plain-text answer, such as a log.
    pub async fn text(&self, path: &str) -> Result<String, Error> {
        let answer = ok(self.send(|c, base| c.get(format!("{base}{path}"))).await?).await?;
        answer.text().await.map_err(unreachable)
    }

    pub async fn submit(&self, key: &str, run: &SubmitRun) -> Result<Submitted, Error> {
        self.post("/v1/runs", run, Some(key)).await
    }

    /// A run by its id or number.
    pub async fn run(&self, id: &str) -> Result<Run, Error> {
        self.get(&path(&["v1", "runs", id])).await
    }

    /// Runs, newest first; `query` is the route's filters.
    pub async fn runs<Q: Serialize + ?Sized>(&self, query: &Q) -> Result<RunList, Error> {
        self.get_query("/v1/runs", query).await
    }

    /// Asks for a run's cancel; its machine's own outcome ends it.
    pub async fn cancel(&self, id: &str) -> Result<Run, Error> {
        let path = path(&["v1", "runs", id, "cancel"]);
        self.post(&path, &serde_json::json!({}), None).await
    }

    pub async fn machines(&self) -> Result<MachineList, Error> {
        self.get("/v1/machines").await
    }

    /// A machine log's newest `lines` whole lines (0: all) at `path`, then each batch as the
    /// machine appends it.
    pub fn follow_log(&self, path: &str, lines: u64) -> LogFollow {
        LogFollow {
            client: self.clone(),
            path: path.to_owned(),
            lines,
            after: None,
            stream: None,
            sse: Sse::default(),
            received: false,
            empty: 0,
        }
    }

    /// A run's events after `starting_after`, to its terminal one.
    pub fn events(&self, id: &str, starting_after: u64) -> Events {
        Events {
            client: self.clone(),
            path: path(&["v1", "runs", id, "events"]),
            cursor: starting_after,
            stream: None,
            sse: Sse::default(),
            received: false,
            empty: 0,
            landing: false,
            ended: false,
            done: false,
        }
    }

    /// A run's events after `starting_after` past its terminal one, until Calcifer ends the
    /// stream: once each output's last revision landed or failed (or Calcifer stopped).
    pub fn events_until_landed(&self, id: &str, starting_after: u64) -> Events {
        Events {
            landing: true,
            ..self.events(id, starting_after)
        }
    }

    /// A list-level stream (`/v1/runs/events`, `/v1/machines/events`), subscribed once this
    /// answers: it carries no backlog, so a caller lists after it. It ends when its connection
    /// closes; the caller subscribes and lists again.
    pub async fn watch<T: DeserializeOwned>(&self, path: &str) -> Result<Watch<T>, Error> {
        let answer = self.send(|c, base| {
            c.get(format!("{base}{path}"))
                .header(header::ACCEPT, "text/event-stream")
        });
        Ok(Watch {
            stream: ok(answer.await?).await?,
            sse: Sse::default(),
            of: std::marker::PhantomData,
        })
    }

    /// Asks Calcifer to stop and waits until that process has let go of the home. Runs on
    /// machines continue; the next Calcifer resumes what it owed.
    pub async fn stop(&self) -> Result<Down, Error> {
        let holder = self.endpoint().await?;
        let base = format!("http://{}", holder.address);
        let answer = self
            .0
            .http
            .post(format!("{base}/v1/down"))
            .bearer_auth(&holder.token)
            .json(&serde_json::json!({}))
            .send()
            .await
            .map_err(unreachable)?;
        let down: Down = decode(answer).await?;
        self.wait("Calcifer to stop", || async { !self.holds(&holder) })
            .await;
        self.remember(None);
        Ok(down)
    }

    /// Reaches Calcifer, starting it if none runs.
    pub async fn ready(&self) -> Result<Holder, Error> {
        self.endpoint().await
    }

    async fn endpoint(&self) -> Result<Holder, Error> {
        if let Some(holder) = self.holder() {
            return Ok(holder);
        }
        let holder = match self.live().await {
            Some(holder) => holder,
            None => self.start().await?,
        };
        self.remember(Some(holder.clone()));
        Ok(holder)
    }

    fn remember(&self, holder: Option<Holder>) {
        *self.0.holder.lock().expect("holder") = holder;
    }

    /// The published record, when its Calcifer answers. Asked over the pooled connection, which
    /// stays open for the command: Calcifer stays up while a client is connected.
    async fn live(&self) -> Option<Holder> {
        let holder = self.record()?;
        let probe = format!("http://{}/v1/runs", holder.address);
        let request = self.0.http.get(probe).query(&[("limit", 1)]);
        request.bearer_auth(&holder.token).send().await.ok()?;
        Some(holder)
    }

    fn record(&self) -> Option<Holder> {
        let raw = LockFile::new(self.0.home.calcifer_lock()).holder().ok()??;
        serde_json::from_slice(&raw).ok()
    }

    /// Whether the Calcifer that published `holder` still holds the home. A record its holder
    /// was killed before erasing names nobody once the lock is free.
    fn holds(&self, holder: &Holder) -> bool {
        self.record().as_ref() == Some(holder) && !self.lock_free()
    }

    fn lock_free(&self) -> bool {
        let lock = LockFile::new(self.0.home.calcifer_lock());
        matches!(lock.try_lock(LockKind::Exclusive), Ok(Some(_)))
    }

    /// Starts Calcifer and waits for its record or its refusal.
    async fn start(&self) -> Result<Holder, Error> {
        let home = &self.0.home;
        let installed = installed(home)?;
        let failed = |e: platform::Error| {
            Error::new(Kind::Unavailable, "calcifer_start_failed", e.to_string())
        };
        home.open().map_err(failed)?;
        platform::durable::private_dir(&home.calcifer()).map_err(failed)?;
        let refused = home.calcifer().join(REFUSED);
        if let Err(e) = std::fs::remove_file(&refused)
            && e.kind() != io::ErrorKind::NotFound
        {
            return Err(failed(platform::Error::Io {
                op: "remove",
                path: refused,
                source: e,
            }));
        }
        let services = Services::new(home, &self.0.user_home);
        let mut steps = Steps::new(home.calcifer_log());
        // Another cozy (a build, a copy) never writes the service: it starts it only as already
        // defined for the installed cozy, else runs the installed one detached.
        let exe = std::env::current_exe().and_then(|exe| exe.canonicalize());
        let ours = matches!((exe, installed.canonicalize()), (Ok(exe), Ok(it)) if exe == it);
        let managed = services.available() && (ours || services.current(&self.spec()));
        // Started detached for want of systemctl: no login starts it again.
        let path: Vec<_> =
            std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()).collect();
        if !managed
            && cfg!(target_os = "linux")
            && tools::on_path("systemctl", &path).is_none()
            && let Some(warning) = tools::warn_once("systemctl")
        {
            output::warn(&warning);
        }
        if managed && ours {
            let wrote = services.ensure(&self.spec()).map_err(failed)?;
            steps.done(match wrote {
                true => "wrote Calcifer's service and reloaded its manager",
                false => "found Calcifer's service current",
            });
        }
        let spec = self.spec();
        let mut launch = Launch::new(managed.then_some(&services), &spec);
        let mut asked = false;
        launch::until(
            self.0.json,
            "Calcifer",
            "`cozy logs calcifer`",
            async || {
                // Nobody holds the home: start one. Again too when the last start lost to a Calcifer
                // that was stopping, which held the lock until it went.
                if self.lock_free() {
                    launch.start().map_err(failed)?;
                    if !std::mem::replace(&mut asked, true) {
                        steps.done("started Calcifer");
                    }
                }
                if let Some(error) = refusal(&refused) {
                    return Err(error);
                }
                if let Some(holder) = self.live().await {
                    steps.done("Calcifer answered");
                    return Ok(Some(holder));
                }
                match launch.ended() {
                    // Another Calcifer held the lock: wait for its record, or for it to let go.
                    None | Some(Ended::Yielded) => Ok(None),
                    Some(Ended::Failed(why)) => Err(refusal(&refused).unwrap_or_else(|| {
                        Error::new(
                            Kind::Unavailable,
                            "calcifer_start_failed",
                            format!("Calcifer: {why}"),
                        )
                        .remedy("the next cozy command starts it again")
                        .next("cozy logs calcifer")
                    })),
                }
            },
        )
        .await
    }

    fn spec(&self) -> ServiceSpec {
        ServiceSpec {
            name: "calcifer".into(),
            program: self.0.home.binary(),
            args: vec!["calcifer".into()],
            log: Some(self.0.home.calcifer_log()),
        }
    }

    /// Sends one request, starting Calcifer when it is gone, reading its record again when a
    /// restart minted a new token, and waiting out a Calcifer that is stopping.
    async fn send(
        &self,
        build: impl Fn(&reqwest::Client, &str) -> RequestBuilder,
    ) -> Result<Response, Error> {
        let mut again = 0;
        loop {
            let holder = self.endpoint().await?;
            let base = format!("http://{}", holder.address);
            let sent = build(&self.0.http, &base)
                .bearer_auth(&holder.token)
                .send()
                .await;
            // Nothing answered: Calcifer is gone, or one that is stopping closed the connection
            // under the request. Every request is safe to send again: reads, keyed asks, cancels.
            // A 401: a restart minted a new token.
            let retry = match &sent {
                Err(_) => true,
                Ok(answer) => answer.status() == StatusCode::UNAUTHORIZED,
            };
            if retry && again < AGAIN {
                again += 1;
                self.remember(None);
                continue;
            }
            let answer = sent.map_err(unreachable)?;
            if answer.status() != StatusCode::SERVICE_UNAVAILABLE || again >= AGAIN {
                return Ok(answer);
            }
            let refusal = envelope(answer).await;
            if refusal.code != "calcifer_stopping" {
                return Err(refusal);
            }
            again += 1;
            self.wait("Calcifer to stop", || async { !self.holds(&holder) })
                .await;
            self.remember(None);
        }
    }

    /// Samples `done` until it holds, saying what it waits for every few seconds.
    async fn wait<F, Fut>(&self, what: &str, done: F)
    where
        F: Fn() -> Fut,
        Fut: Future<Output = bool>,
    {
        let began = Instant::now();
        let mut reported = began;
        while !done().await {
            if reported.elapsed() >= REPORT {
                reported = Instant::now();
                let waited = began.elapsed().as_secs();
                self.say(&format!("waiting for {what} ({waited}s so far)"));
            }
            tokio::time::sleep(SAMPLE).await;
        }
    }

    fn say(&self, message: &str) {
        if !self.0.json {
            anstream::eprintln!("{message}");
        }
    }
}

/// A run's events, resumed from the last durable one whenever the connection closes before the
/// terminal event, across a Calcifer restart too. Dropping it cancels nothing.
pub struct Events {
    client: Client,
    path: String,
    cursor: u64,
    stream: Option<Response>,
    sse: Sse,
    /// Whether this connection has carried an event.
    received: bool,
    /// Connections in a row that closed carrying none.
    empty: u32,
    /// Reads past the terminal event until Calcifer ends the stream.
    landing: bool,
    /// The terminal event was read.
    ended: bool,
    done: bool,
}

impl Events {
    /// Reads on after `None` from where it stopped: a landing follower whose run still owes an
    /// output (its Calcifer stopped) waits again.
    pub fn again(&mut self) {
        self.done = false;
    }

    /// The next event; `None` after the terminal one, or for a landing stream once Calcifer ends
    /// it after that. Cancel-safe: a dropped call loses nothing
    /// a later one would not read again.
    pub async fn next(&mut self) -> Option<Result<Event, Error>> {
        while !self.done {
            if let Some(event) = self.parsed() {
                if let Some(n) = event.sequence_number() {
                    self.cursor = self.cursor.max(n);
                }
                self.received = true;
                self.empty = 0;
                self.ended |= event.is_terminal();
                self.done = self.ended && !self.landing;
                return Some(Ok(event));
            }
            let Some(stream) = self.stream.as_mut() else {
                match self.open().await {
                    Ok(stream) => self.stream = Some(stream),
                    Err(error) => {
                        self.done = true;
                        return Some(Err(error));
                    }
                }
                continue;
            };
            match stream.chunk().await {
                Ok(Some(bytes)) => self.sse.push(&bytes),
                // An ended run's stream ends once its outputs settled; its follower checks them.
                Ok(None) if self.ended => {
                    self.stream = None;
                    self.done = true;
                }
                // A close is not a verdict: reconnect from the cursor.
                Ok(None) | Err(_) => {
                    self.stream = None;
                    self.sse = Sse::default();
                    if !std::mem::take(&mut self.received) {
                        self.empty += 1;
                    }
                    if self.empty > AGAIN {
                        self.done = true;
                        return Some(Err(Error::new(
                            Kind::Unavailable,
                            "events_closed",
                            format!(
                                "the event stream closed {} times without an event",
                                self.empty
                            ),
                        )
                        .remedy("the run is still recorded")
                        .next("cozy run list")));
                    }
                }
            }
        }
        None
    }

    async fn open(&self) -> Result<Response, Error> {
        let cursor = self.cursor;
        let answer = self.client.send(|c, base| {
            c.get(format!("{base}{}", self.path))
                .query(&[("starting_after", cursor)])
                .header(header::ACCEPT, "text/event-stream")
        });
        ok(answer.await?).await
    }

    /// The next event this build reads; event types it does not know are skipped.
    fn parsed(&mut self) -> Option<Event> {
        while let Some(data) = self.sse.next() {
            if let Ok(event) = serde_json::from_str(&data) {
                return Some(event);
            }
        }
        None
    }
}

/// One list-level stream (`Client::watch`).
pub struct Watch<T> {
    stream: Response,
    sse: Sse,
    of: std::marker::PhantomData<T>,
}

impl<T: DeserializeOwned> Watch<T> {
    /// The next item, skipping types this build does not read; `None` once the stream closed.
    pub async fn next(&mut self) -> Option<T> {
        loop {
            while let Some(data) = self.sse.next() {
                if let Ok(item) = serde_json::from_str(&data) {
                    return Some(item);
                }
            }
            match self.stream.chunk().await {
                Ok(Some(bytes)) => self.sse.push(&bytes),
                Ok(None) | Err(_) => return None,
            }
        }
    }
}

/// A followed machine log. A closed stream is opened again after the last batch's place, across
/// a Calcifer restart too, so no line arrives twice.
pub struct LogFollow {
    client: Client,
    path: String,
    lines: u64,
    after: Option<String>,
    stream: Option<Response>,
    sse: Sse,
    /// Whether this connection has carried lines.
    received: bool,
    /// Connections in a row that closed carrying none.
    empty: u32,
}

impl LogFollow {
    pub async fn next(&mut self) -> Result<LogLines, Error> {
        loop {
            if let Some(data) = self.sse.next() {
                if let Ok(lines) = serde_json::from_str::<LogLines>(&data) {
                    self.after = Some(lines.after.clone());
                    (self.received, self.empty) = (true, 0);
                    return Ok(lines);
                }
                continue;
            }
            let Some(stream) = self.stream.as_mut() else {
                self.stream = Some(self.open().await?);
                continue;
            };
            match stream.chunk().await {
                Ok(Some(bytes)) => self.sse.push(&bytes),
                Ok(None) | Err(_) => {
                    (self.stream, self.sse) = (None, Sse::default());
                    if !std::mem::take(&mut self.received) {
                        self.empty += 1;
                    }
                    if self.empty > AGAIN {
                        return Err(Error::new(
                            Kind::Unavailable,
                            "log_closed",
                            format!("the log stream closed {} times without a line", self.empty),
                        ));
                    }
                }
            }
        }
    }

    async fn open(&self) -> Result<Response, Error> {
        let answer = self.client.send(|c, base| {
            let request = c
                .get(format!("{base}{}", self.path))
                .header(header::ACCEPT, "text/event-stream");
            match &self.after {
                Some(after) => request.query(&[("after", after)]),
                None => request.query(&[("lines", self.lines)]),
            }
        });
        ok(answer.await?).await
    }
}

/// Server-sent events: the `data:` lines of each event, up to its blank line.
#[derive(Default)]
struct Sse {
    buffer: Vec<u8>,
    data: String,
}

impl Sse {
    fn push(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    fn next(&mut self) -> Option<String> {
        while let Some(end) = self.buffer.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.buffer.drain(..=end).collect();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim_end_matches(['\n', '\r']);
            if line.is_empty() {
                if !self.data.is_empty() {
                    return Some(std::mem::take(&mut self.data));
                }
            } else if let Some(value) = line.strip_prefix("data:") {
                if !self.data.is_empty() {
                    self.data.push('\n');
                }
                self.data.push_str(value.strip_prefix(' ').unwrap_or(value));
            }
        }
        None
    }
}

/// A path from segments, each percent-encoded.
pub fn path(segments: &[&str]) -> String {
    let mut url = url::Url::parse("http://calcifer/").expect("a base URL");
    url.path_segments_mut()
        .expect("a base URL")
        .extend(segments);
    url.path().to_owned()
}

/// The home's installed `cozy`, the only one Calcifer and the machine run.
pub(crate) fn installed(home: &Home) -> Result<PathBuf, Error> {
    let binary = home.binary();
    let message = format!("Calcifer runs only {}, which is missing", binary.display());
    match binary.is_file() {
        true => Ok(binary),
        false => Err(Error::new(Kind::NotFound, "cozy_not_installed", message).next(INSTALL)),
    }
}

const INSTALL: &str = match cfg!(windows) {
    true => "irm https://tensorhub.com/install.ps1 | iex",
    false => "curl -LsSf https://tensorhub.com/install.sh | sh",
};

/// The refusal a Calcifer that could not start left.
fn refusal(path: &Path) -> Option<Error> {
    let raw = std::fs::read(path).ok()?;
    serde_json::from_slice::<Envelope>(&raw)
        .ok()
        .map(|e| e.error)
}

fn unreachable(error: reqwest::Error) -> Error {
    Error::new(
        Kind::Unavailable,
        "calcifer_unreachable",
        format!("Calcifer stopped answering: {error}"),
    )
    .remedy("it may have stopped mid-request; run the command again")
}

async fn ok(answer: Response) -> Result<Response, Error> {
    if answer.status().is_success() {
        Ok(answer)
    } else {
        Err(envelope(answer).await)
    }
}

async fn decode<T: DeserializeOwned>(answer: Response) -> Result<T, Error> {
    let answer = ok(answer).await?;
    let body = answer.bytes().await.map_err(unreachable)?;
    serde_json::from_slice(&body).map_err(|e| {
        Error::new(
            Kind::Internal,
            "untyped_answer",
            format!("Calcifer's answer is not what this cozy reads: {e}"),
        )
    })
}

/// A refusal's typed envelope; a body that is not one is reported as what it was.
async fn envelope(answer: Response) -> Error {
    let status = answer.status();
    let body = answer.bytes().await.unwrap_or_default();
    match serde_json::from_slice::<Envelope>(&body) {
        Ok(envelope) => envelope.error,
        Err(_) => {
            let text = String::from_utf8_lossy(&body);
            Error::new(
                Kind::Internal,
                "untyped_answer",
                format!(
                    "Calcifer answered {status} with no typed refusal: {}",
                    text.trim()
                ),
            )
        }
    }
}

/// The steps of a start, each with its time, appended to Calcifer's log (`cozy logs calcifer`)
/// beside Calcifer's own: a slow start names its slow step.
struct Steps {
    log: PathBuf,
    since: Instant,
}

impl Steps {
    fn new(log: PathBuf) -> Self {
        Self {
            log,
            since: Instant::now(),
        }
    }

    fn done(&mut self, step: &str) {
        let ms = self.since.elapsed().as_millis();
        let line = format!("{}  INFO cozy: {step} in {ms} ms\n", jiff::Timestamp::now());
        let log = OpenOptions::new().create(true).append(true).open(&self.log);
        if let Ok(mut log) = log {
            let _ = log.write_all(line.as_bytes());
        }
        self.since = Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_survive_any_chunking() {
        let stream = b": hi\r\nid: 1\r\ndata: {\"a\":\r\ndata: 1}\r\n\r\nevent: x\ndata: 2\n\n";
        for cut in 0..stream.len() {
            let mut sse = Sse::default();
            let mut got = vec![];
            for part in [&stream[..cut], &stream[cut..]] {
                sse.push(part);
                got.extend(std::iter::from_fn(|| sse.next()));
            }
            assert_eq!(got, ["{\"a\":\n1}", "2"], "cut at {cut}");
        }
    }

    #[test]
    fn paths_escape_each_segment() {
        assert_eq!(path(&["v1", "runs", "a b/c"]), "/v1/runs/a%20b%2Fc");
    }
}

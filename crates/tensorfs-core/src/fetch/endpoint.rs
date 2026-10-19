//! The distribution client (`spec/distribution.md`): one endpoint `B` serving
//! `B/refs/<org>/<name>.json` and `B/blobs/<aa>/<bb>/<sha256>`, over HTTP(S) or a directory.
//!
//! - **Reads.** The repo document with its ETag (conditional GET); objects as Fetcher locations,
//!   whose redirects the client follows without credentials across origins.
//! - **Uploads.** Per object HEAD → PUT (to the HEAD's `Location`, else here) → HEAD; `412` means
//!   present. [`upload_tree`] sends chunks, assets and repo files, then headers, then the tree.
//! - **Publish.** [`Endpoint::update_repo`] PUTs the whole document under `If-Match` or
//!   `If-None-Match: *` and, on `412`, reads it again and re-applies the change.
//! - **A directory** writes through a store over it: objects verified, temp and rename; refs under
//!   the store's record lock, read and replaced inside it.
//! - **Weather** (transport failures, 408/425/429/5xx) is asked again at `Retry-After` or a
//!   doubling wait, until nothing has moved for the Fetcher's stall rule.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use hyper::Uri;
use serde_json::Value;

use super::net::{join, same_origin, Body, Client, HeaderMap};
use super::origin::{release_of, split_name, Location, Origin};
use super::policy::is_loopback_origin;
use super::token::{default_for, s3_base, Request, TokenProvider};
use crate::err::{refuse, Code, Refusal, Result};
use crate::format::{repo_key, Header, ObjectRef, Repo, Tree};
use crate::sha256::{digest, hex, hex_digest};
use crate::store2::Store;

/// The most bytes a repo document may have.
const REPO_BYTES: usize = 16 << 20;
/// The Fetcher's tick, and its floor for stillness (120 ticks): nothing moving for less is no
/// evidence; for 8× the slowest answer seen, it is.
const TICK: Duration = Duration::from_millis(250);
const STILL: Duration = Duration::from_secs(30);
const STALL_FACTOR: u32 = 8;
const WEATHER: [u16; 7] = [408, 425, 429, 500, 502, 503, 504];
/// Objects an upload sends at once.
const STREAMS: usize = 16;

/// An endpoint of the distribution protocol.
pub struct Endpoint {
    url: String,
    at: At,
    client: Client,
    token: Option<Arc<dyn TokenProvider>>,
    /// The slowest answer seen (nanoseconds): what the stall rule scales.
    slowest: AtomicU64,
}

enum At {
    Http(String),
    Dir(PathBuf, OnceLock<Store>),
}

/// A repo document as read: unchanged since the ETag asked with, absent, or it and its ETag.
#[derive(Debug, Clone, PartialEq)]
pub enum Fresh<T = Repo> {
    Unchanged,
    Absent,
    Doc(T, String),
}

/// An object's bytes for an upload.
pub enum Payload {
    Bytes(Bytes),
    File(fs::File),
}

/// What [`upload_tree`] did: the repo as published, the closure's objects, and those it sent.
#[derive(Debug, Clone, PartialEq)]
pub struct Uploaded {
    pub repo: Repo,
    pub objects: usize,
    pub sent: usize,
    pub bytes: u64,
}

/// An answer read whole.
struct Reply {
    url: String,
    status: u16,
    headers: HeaderMap,
    body: Vec<u8>,
}

impl Reply {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name)?.to_str().ok()
    }
}

/// One exchange: its method, URL, headers, body, and whether the token provider authorizes it.
struct Ask<'a> {
    method: &'a str,
    url: &'a str,
    headers: Vec<(String, String)>,
    body: Option<&'a (dyn Fn() -> Result<Payload> + Send + Sync)>,
    payload: Option<&'a str>,
    auth: bool,
}

impl Endpoint {
    /// The endpoint at `url`: `https://…`, `http://` on loopback, `s3://<bucket>[/<prefix>]`
    /// (SigV4), or a directory (`file:///dir` or a path). With no `token`, the default chain's
    /// provider for it, if any.
    pub fn new(client: &Client, url: &str, token: Option<Arc<dyn TokenProvider>>) -> Result<Endpoint> {
        let url = url.trim_end_matches('/').to_owned();
        let at = if url.starts_with("https://") || url.starts_with("http://") || url.starts_with("s3://") {
            let base = match url.starts_with("s3://") {
                true => s3_base(&url)?,
                false => url.clone(),
            };
            if base.starts_with("http://") && !is_loopback_origin(&base) {
                return refuse(Code::SOURCE_NOT_ALLOWED, format!("{base}: plain http only on loopback"));
            }
            client.policy().admit(&base)?;
            At::Http(base)
        } else {
            let path = url.strip_prefix("file://").unwrap_or(&url);
            if !path.starts_with('/') {
                return refuse(Code::SOURCE_NOT_ALLOWED, format!("{url}: not an https://, loopback http:// or file:/// endpoint"));
            }
            At::Dir(PathBuf::from(path), OnceLock::new())
        };
        let token = match at {
            At::Http(_) => token.or_else(|| default_for(&url)),
            At::Dir(..) => None,
        };
        Ok(Endpoint { url, at, client: client.clone(), token, slowest: AtomicU64::new(0) })
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    /// Public reads only, even when the environment configures credentials for this host.
    pub fn anonymous(mut self) -> Self {
        self.token = None;
        self
    }

    /// The repo document `name` and its ETag; `None` when there is none.
    pub async fn repo(&self, name: &str) -> Result<Option<(Repo, String)>> {
        Ok(match self.repo_since(name, None).await? {
            Fresh::Doc(repo, etag) => Some((repo, etag)),
            _ => None,
        })
    }

    /// The repo document `name`, or `Unchanged` while it still has ETag `etag`.
    pub async fn repo_since(&self, name: &str, etag: Option<&str>) -> Result<Fresh> {
        let (bytes, etag) = match self.repo_bytes(name, etag).await? {
            Fresh::Doc(bytes, etag) => (bytes, etag),
            Fresh::Unchanged => return Ok(Fresh::Unchanged),
            Fresh::Absent => return Ok(Fresh::Absent),
        };
        let what = format!("{}/{}", self.url, repo_key(name));
        let repo = Repo::from_json(&bytes).map_err(|e| Refusal { code: Code::UNKNOWN_FORMAT, detail: format!("{what}: {e}") })?;
        match repo.name.eq_ignore_ascii_case(name) {
            true => Ok(Fresh::Doc(repo, etag)),
            false => refuse(Code::UNKNOWN_FORMAT, format!("{what} names {:?}", repo.name)),
        }
    }

    /// The repo document `name` as its bytes, unparsed, or `Unchanged` while it has ETag `etag`.
    pub async fn repo_bytes(&self, name: &str, etag: Option<&str>) -> Result<Fresh<Vec<u8>>> {
        split_name(name)?;
        let key = repo_key(name);
        let (bytes, tag) = match &self.at {
            At::Dir(root, _) => match fs::read(root.join(&key)) {
                Ok(bytes) => {
                    let tag = etag_of(&bytes);
                    (bytes, tag)
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Fresh::Absent),
                Err(e) => return Err(crate::err::classify_io(root.join(&key).display().to_string(), e)),
            },
            At::Http(base) => {
                let url = format!("{base}/{key}");
                let headers = etag.map(|e| ("if-none-match".to_owned(), e.to_owned())).into_iter().collect();
                let reply = self.exchange(Ask { method: "GET", url: &url, headers, body: None, payload: None, auth: true }).await?;
                match reply.status {
                    200 => {
                        let tag = reply.header("etag").map_or_else(|| etag_of(&reply.body), str::to_owned);
                        (reply.body, tag)
                    }
                    304 => return Ok(Fresh::Unchanged),
                    404 | 410 => return Ok(Fresh::Absent),
                    _ => return Err(refusal(&reply, "GET")),
                }
            }
        };
        Ok(match etag == Some(tag.as_str()) {
            true => Fresh::Unchanged,
            false => Fresh::Doc(bytes, tag),
        })
    }

    /// Writes `bytes` as the repo document `name` while it still has ETag `etag` (`None`: while
    /// there is none); false when it has changed (`412`). A server checks the bytes.
    pub async fn put_repo_bytes(&self, name: &str, bytes: Vec<u8>, etag: Option<&str>) -> Result<bool> {
        split_name(name)?;
        let key = repo_key(name);
        let base = match &self.at {
            At::Http(base) => base,
            At::Dir(root, _) => {
                let (root, etag) = (root.clone(), etag.map(str::to_owned));
                let put = tokio::task::spawn_blocking(move || put_dir(&root, &key, &bytes, etag.as_deref()));
                return put.await.map_err(|e| Refusal { code: Code::IO_FAILED, detail: e.to_string() })?;
            }
        };
        let url = format!("{base}/{key}");
        let precondition = match etag {
            Some(etag) => ("if-match".to_owned(), etag.to_owned()),
            None => ("if-none-match".to_owned(), "*".to_owned()),
        };
        let headers = vec![("content-type".to_owned(), "application/json".to_owned()), precondition];
        let payload = hex_digest(&bytes);
        let bytes = Bytes::from(bytes);
        let body = move || Ok(Payload::Bytes(bytes.clone()));
        let reply = self.exchange(Ask { method: "PUT", url: &url, headers, body: Some(&body), payload: Some(&payload), auth: true }).await?;
        match reply.status {
            200 | 201 | 204 => Ok(true),
            // S3 answers a race of conditional writes 409 ConditionalRequestConflict.
            412 => Ok(false),
            409 if envelope(&reply.body).is_none() => Ok(false),
            _ => Err(refusal(&reply, "PUT")),
        }
    }

    /// Changes the repo document `name` to what `change` makes of it (`None`: a document with
    /// no releases and no checkpoints, which deletes the model): a conditional write, with
    /// `change` applied again to the document as it is whenever another writer won.
    pub async fn update_repo(
        &self,
        name: &str,
        mut change: impl FnMut(Option<Repo>) -> Result<Option<Repo>>,
    ) -> Result<Repo> {
        let mut sent: Option<Repo> = None;
        loop {
            let (old, etag) = match self.repo(name).await? {
                Some((repo, etag)) => (Some(repo), Some(etag)),
                None => (None, None),
            };
            // A lost answer, or another writer that wrote the same: it is there.
            if sent.is_some() && old == sent {
                return Ok(sent.expect("compared"));
            }
            let new = change(old.clone())?.unwrap_or_else(|| Repo::new(name));
            if !new.name.eq_ignore_ascii_case(name) {
                return refuse(Code::KEY_GRAMMAR, format!("repo {:?} cannot be {name}", new.name));
            }
            let empty = new.releases.is_empty() && new.checkpoints.is_empty();
            if old.as_ref() == Some(&new) || (old.is_none() && empty) {
                return Ok(new);
            }
            let bytes = new.to_json().map_err(|e| Refusal { code: Code::UNKNOWN_FORMAT, detail: e.to_string() })?;
            if self.put_repo_bytes(name, bytes, etag.as_deref()).await? {
                return Ok(new);
            }
            sent = Some(new);
        }
    }

    /// An object named by digest, including its length, or `None` when absent or unreadable.
    pub async fn object(&self, sha256: &str) -> Result<Option<ObjectRef>> {
        let sha256 = sha256.strip_prefix("sha256:").unwrap_or(sha256);
        if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
            return refuse(Code::MALFORMED_DIGEST, "an object digest must be 64 lowercase hex digits");
        }
        let mut object = ObjectRef { sha256: sha256.to_owned(), length: 0 };
        object.length = match &self.at {
            At::Dir(root, _) => match fs::metadata(root.join(object.key())) {
                Ok(metadata) => metadata.len(),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(error) => return Err(crate::err::classify_io(self.url(), error)),
            },
            At::Http(base) => {
                let url = format!("{base}/{}", object.key());
                let reply = self.exchange(Ask { method: "HEAD", url: &url, headers: vec![], body: None, payload: None, auth: true }).await?;
                match reply.status {
                    200 => reply.header("content-length").and_then(|length| length.parse().ok())
                        .ok_or_else(|| Refusal { code: Code::LENGTH_MISMATCH, detail: format!("HEAD {url}: no valid Content-Length") })?,
                    404 | 410 => return Ok(None),
                    _ => return Err(refusal(&reply, "HEAD")),
                }
            }
        };
        Ok(Some(object))
    }

    /// Whether the endpoint holds `object`.
    pub async fn has(&self, object: &ObjectRef) -> Result<bool> {
        match &self.at {
            At::Dir(root, _) => Ok(fs::metadata(root.join(object.key())).is_ok_and(|m| m.len() == object.length)),
            At::Http(base) => {
                let url = format!("{base}/{}", object.key());
                let reply = self.exchange(Ask { method: "HEAD", url: &url, headers: vec![], body: None, payload: None, auth: true }).await?;
                match reply.status {
                    200 => held(&reply, object).map(|()| true),
                    404 | 410 => Ok(false),
                    _ => Err(refusal(&reply, "HEAD")),
                }
            }
        }
    }

    /// Sends `object` unless the endpoint holds it: true when it sent it. `payload` gives its
    /// bytes, only when they are needed (once per attempt).
    pub async fn upload(&self, object: &ObjectRef, payload: &(dyn Fn() -> Result<Payload> + Send + Sync)) -> Result<bool> {
        let base = match &self.at {
            At::Http(base) => base,
            At::Dir(..) => {
                if self.has(object).await? {
                    return Ok(false);
                }
                let (store, payload, object) = (self.store()?.clone(), payload()?, object.clone());
                let put = tokio::task::spawn_blocking(move || match payload {
                    Payload::Bytes(bytes) => store.put(&bytes[..], Some(&object)),
                    Payload::File(file) => store.put(file, Some(&object)),
                });
                let put = put.await.map_err(|e| Refusal { code: Code::IO_FAILED, detail: e.to_string() })?;
                put.map_err(crate::store2::Error::refusal)?;
                return Ok(true);
            }
        };
        let url = format!("{base}/{}", object.key());
        let head = |method| Ask { method, url: &url, headers: vec![], body: None, payload: None, auth: true };
        let reply = self.exchange(head("HEAD")).await?;
        match reply.status {
            200 => return held(&reply, object).map(|()| false),
            404 | 410 => {}
            _ => return Err(refusal(&reply, "HEAD")),
        }
        // A writer's HEAD of an absent object may say where to PUT it (a presigned URL).
        let target = reply.header("location").map(|at| join(&url.parse::<Uri>().expect("sent"), at));
        let auth = target.as_deref().is_none_or(|t| same_origin(t, &url));
        let target = target.unwrap_or_else(|| url.clone());
        let checksum = crate::b64::encode(&decode_hex(&object.sha256));
        let headers = vec![
            ("content-length".to_owned(), object.length.to_string()),
            ("x-amz-checksum-sha256".to_owned(), checksum),
            ("if-none-match".to_owned(), "*".to_owned()),
        ];
        let ask = Ask { method: "PUT", url: &target, headers, body: Some(payload), payload: Some(&object.sha256), auth };
        let put = self.exchange(ask).await?;
        if !matches!(put.status, 200 | 201 | 204 | 412) {
            return Err(refusal(&put, "PUT"));
        }
        // The confirming HEAD: where a redirecting server verifies what it was sent.
        let confirm = self.exchange(head("HEAD")).await?;
        match confirm.status {
            200 => held(&confirm, object).map(|()| true),
            _ => Err(refusal(&confirm, "the confirming HEAD")),
        }
    }

    /// One authorized exchange at `key` under an HTTP endpoint (`None` for a directory): its
    /// status, headers and body. What the conformance suite checks servers with.
    pub(crate) async fn raw(&self, method: &str, key: &str, headers: Vec<(String, String)>, body: Option<Bytes>) -> Result<Option<(u16, HeaderMap, Vec<u8>)>> {
        let At::Http(base) = &self.at else { return Ok(None) };
        let url = format!("{base}/{key}");
        let payload = body.as_ref().map(|b| hex_digest(b));
        let make = body.map(|b| move || Ok(Payload::Bytes(b.clone())));
        let body = make.as_ref().map(|m| m as &(dyn Fn() -> Result<Payload> + Send + Sync));
        let reply = self.exchange(Ask { method, url: &url, headers, body, payload: payload.as_deref(), auth: true }).await?;
        Ok(Some((reply.status, reply.headers, reply.body)))
    }

    /// The store over a directory endpoint, which its writes go through.
    fn store(&self) -> Result<&Store> {
        let At::Dir(root, store) = &self.at else { unreachable!("a directory endpoint") };
        if let Some(store) = store.get() {
            return Ok(store);
        }
        let opened = Store::open(root).map_err(crate::store2::Error::refusal)?;
        Ok(store.get_or_init(|| opened))
    }

    /// The provider's headers for `method url`, carrying `headers` with `payload`.
    async fn authorize(&self, method: &str, url: &str, headers: &[(String, String)], payload: Option<&str>) -> Result<Vec<(String, String)>> {
        match &self.token {
            Some(token) => token.headers(&Request { method, url, headers, payload }).await,
            None => Ok(vec![]),
        }
    }

    /// The stall rule: nothing moved for 8× the slowest answer, never under [`STILL`].
    fn patience(&self) -> Duration {
        let slowest = Duration::from_nanos(self.slowest.load(Ordering::Relaxed));
        STILL.max(slowest * STALL_FACTOR)
    }

    /// One exchange, its answer read whole. Weather is asked again until nothing has moved for
    /// the stall rule; the provider may ask once more (a nonce, a refresh).
    async fn exchange(&self, ask: Ask<'_>) -> Result<Reply> {
        let (mut still, mut wait, mut asked_again) = (Instant::now(), TICK, false);
        let mut stated_most = Duration::ZERO;
        loop {
            let mut headers = ask.headers.clone();
            let (moved, began) = (Arc::new(AtomicU64::new(0)), Instant::now());
            let authorized = match ask.auth {
                true => self.authorize(ask.method, ask.url, &ask.headers, ask.payload).await,
                false => Ok(vec![]),
            };
            let outcome = match authorized {
                Ok(auth) => {
                    headers.extend(auth);
                    let body = match ask.body {
                        Some(make) => Some(match make()? {
                            Payload::Bytes(bytes) => Body::Bytes(bytes),
                            Payload::File(file) => {
                                let length = file.metadata().map_err(|e| crate::err::classify_io(ask.url, e))?.len();
                                Body::File { file, length, sent: Arc::clone(&moved) }
                            }
                        }),
                        None => None,
                    };
                    self.watched(self.round_trip(ask.method, ask.url, &headers, body, &moved), &moved).await
                }
                Err(e) => Err(e),
            };
            // Bytes that moved before a transport failure are progress; a "not now" is not.
            let progressed = outcome.is_err() && moved.load(Ordering::Relaxed) > 0;
            let (why, stated) = match outcome {
                Ok(reply) => {
                    self.slowest.fetch_max(began.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    let sent = Request { method: ask.method, url: ask.url, headers: &headers, payload: ask.payload };
                    let again = ask.auth && (self.token.as_ref()).is_some_and(|t| t.retry(&sent, reply.status, &reply.headers));
                    if again && !asked_again {
                        asked_again = true;
                        continue;
                    }
                    if !WEATHER.contains(&reply.status) {
                        return Ok(reply);
                    }
                    let stated = (reply.header("retry-after")).and_then(|s| s.trim().parse().ok()).map(Duration::from_secs);
                    (format!("{} {}: {}", ask.method, ask.url, reply.status), stated)
                }
                Err(e) if matches!(e.code, Code::TRANSFER_FAILED | Code::FD_HEADROOM | Code::STALLED) => (e.detail, None),
                Err(e) => return Err(e),
            };
            if progressed {
                still = Instant::now();
            }
            stated_most = stated_most.max(stated.unwrap_or_default());
            let patience = self.patience().max(stated_most * 2);
            if still.elapsed() > patience {
                return refuse(Code::HUB_UNREACHABLE, format!("{why}; nothing moved for {:.0} s", still.elapsed().as_secs_f64()));
            }
            tracing::debug!(%why, "asked again");
            tokio::time::sleep(stated.unwrap_or(wait)).await;
            wait = (wait * 2).min(patience / 8);
        }
    }

    async fn round_trip(&self, method: &str, url: &str, headers: &[(String, String)], body: Option<Body>, moved: &AtomicU64) -> Result<Reply> {
        let mut answer = self.client.send(method, url, headers, body).await?;
        let (status, headers) = (answer.status(), answer.response.headers().clone());
        moved.fetch_add(1, Ordering::Relaxed);
        let mut body = Vec::new();
        while let Some(frame) = answer.frame().await? {
            body.extend_from_slice(&frame);
            moved.fetch_add(frame.len() as u64, Ordering::Relaxed);
            if body.len() > REPO_BYTES {
                return refuse(Code::SIZE_CAP, format!("{method} {url}: an answer over {REPO_BYTES} B"));
            }
        }
        Ok(Reply { url: url.to_owned(), status, headers, body })
    }

    /// `exchange` dropped once nothing moves (no body bytes out, no answer) for the stall rule.
    async fn watched<T>(&self, exchange: impl std::future::Future<Output = Result<T>>, moved: &AtomicU64) -> Result<T> {
        tokio::pin!(exchange);
        let (mut seen, mut since) = (0, Instant::now());
        loop {
            tokio::select! {
                done = &mut exchange => return done,
                () = tokio::time::sleep(TICK) => {
                    let now = moved.load(Ordering::Relaxed);
                    if now != seen {
                        (seen, since) = (now, Instant::now());
                    } else if since.elapsed() > self.patience() {
                        return refuse(Code::STALLED, format!("nothing moved for {:.0} s", since.elapsed().as_secs_f64()));
                    }
                }
            }
        }
    }
}

#[async_trait]
impl Origin for Endpoint {
    async fn resolve(&self, name: &str, release: Option<&str>, _: Option<&str>) -> Result<ObjectRef> {
        match self.repo(name).await? {
            Some((repo, _)) => Ok(release_of(&repo, release)?.1),
            None => refuse(Code::REF_NOT_FOUND, format!("no repo {name} at {}", self.url)),
        }
    }

    async fn locate(&self, objects: &[ObjectRef]) -> Result<Vec<Location>> {
        let mut out = Vec::with_capacity(objects.len());
        for object in objects {
            out.push(match &self.at {
                At::Dir(root, _) => Location::File(root.join(object.key())),
                At::Http(base) => {
                    let url = format!("{base}/{}", object.key());
                    match &self.token {
                        Some(provider) => Location::Authenticated { url, provider: Arc::clone(provider) },
                        None => Location::Url { url, headers: vec![], expires: None },
                    }
                }
            });
        }
        Ok(out)
    }

    fn public(&self) -> bool {
        self.token.is_none()
    }
}

/// `repo` (`name`'s, when there is none) naming `tree`: as release `version`, whose revision
/// counts each change, or with none among the unreleased checkpoints.
pub fn with_tree(repo: Option<Repo>, name: &str, tree: &ObjectRef, version: Option<&str>) -> Repo {
    let mut repo = repo.unwrap_or_else(|| Repo::new(name));
    match version {
        Some(version) => {
            match repo.releases.iter_mut().find(|r| r.version == version) {
                Some(r) if r.tree == *tree => {}
                Some(r) => (r.tree, r.revision) = (tree.clone(), r.revision + 1),
                None => repo.set_release(version, tree.clone()),
            }
            repo.checkpoints.retain(|c| c != tree);
        }
        None if repo.roots().any(|r| r == tree) => {}
        None => repo.checkpoints.push(tree.clone()),
    }
    repo
}

/// Sends what `endpoint` lacks of `tree`'s closure in `store` (chunks, assets and repo files,
/// then headers, then the tree), then names it in repo `name`: as release `version`, or with
/// none as an unreleased checkpoint.
pub async fn upload_tree(endpoint: &Endpoint, name: &str, tree: &ObjectRef, store: &Store, version: Option<&str>) -> Result<Uploaded> {
    let refusal = crate::store2::Error::refusal;
    let parsed = store.tree(tree).map_err(refusal)?;
    let mut leaves = Vec::new();
    let mut headers = Vec::new();
    for (_, header) in parsed.variants() {
        let bytes = store.read(header).map_err(refusal)?;
        let parsed = Header::from_cbor(&bytes).map_err(|e| Refusal { code: Code::UNKNOWN_FORMAT, detail: format!("{}: {e}", header.sha256) })?;
        leaves.extend(parsed.objects().cloned());
        headers.push(header.clone());
    }
    leaves.extend(parsed.repo_files().map(|(_, file)| file.clone()));
    leaves.sort();
    leaves.dedup();
    let (mut sent, mut bytes) = (0, 0);
    let objects = leaves.len() + headers.len() + 1;
    for stage in [leaves, headers, vec![tree.clone()]] {
        let mut sending = futures::stream::iter(stage)
            .map(|object| async move {
                let (from, at) = (store.clone(), object.clone());
                let payload = move || from.file(&at).map(Payload::File).map_err(crate::store2::Error::refusal);
                endpoint.upload(&object, &payload).await.map(|s| (s, object.length))
            })
            .buffer_unordered(STREAMS);
        while let Some(done) = sending.next().await {
            if let (true, length) = done? {
                (sent, bytes) = (sent + 1, bytes + length);
            }
        }
    }
    let repo = endpoint.update_repo(name, |old| Ok(Some(with_tree(old, name, tree, version)))).await?;
    Ok(Uploaded { repo, objects, sent, bytes })
}

/// Replaces `<root>/<key>` with `bytes` under the store's record lock while it has ETag `etag`.
fn put_dir(root: &Path, key: &str, bytes: &[u8], etag: Option<&str>) -> Result<bool> {
    let (path, refs) = (root.join(key), root.join("refs"));
    let io = |what: &Path| {
        let what = what.display().to_string();
        move |e: std::io::Error| crate::err::classify_io(what, e)
    };
    fs::create_dir_all(path.parent().expect("a ref has a directory")).map_err(io(&path))?;
    let _lock = platform::lock::dir(&refs).map_err(|e| io(&refs)(e.into()))?;
    let now = match fs::read(&path) {
        Ok(bytes) => Some(etag_of(&bytes)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(io(&path)(e)),
    };
    if now.as_deref() != etag {
        return Ok(false);
    }
    platform::durable::replace(&path, bytes, 0o644).map_err(|e| io(&path)(e.into()))?;
    Ok(true)
}

/// A writer's objects on their way to an endpoint (`Writer(repo_id=…)`): each is sent (HEAD →
/// PUT → HEAD) as soon as it is sealed while the writer goes on, at most [`IN_FLIGHT`] at once;
/// at commit the header, then the tree, then the repo's new checkpoint.
/// Its calls block: a writer runs outside the runtime that sends.
pub struct Stream {
    endpoint: Arc<Endpoint>,
    runtime: tokio::runtime::Handle,
    sent: Mutex<HashSet<ObjectRef>>,
    sending: Mutex<Vec<tokio::task::JoinHandle<Result<()>>>>,
    failed: Mutex<Option<Refusal>>,
    permits: Arc<tokio::sync::Semaphore>,
}

impl std::fmt::Debug for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Stream({})", self.endpoint.url)
    }
}

/// Objects a [`Stream`] holds in memory while they are sent (each at most 64 MiB).
const IN_FLIGHT: usize = 4;

impl Stream {
    pub fn new(endpoint: Arc<Endpoint>, runtime: tokio::runtime::Handle) -> Stream {
        Stream {
            endpoint,
            runtime,
            sent: Mutex::default(),
            sending: Mutex::default(),
            failed: Mutex::default(),
            permits: Arc::new(tokio::sync::Semaphore::new(IN_FLIGHT)),
        }
    }

    /// Whether `object` was given to [`Stream::put`].
    pub fn has(&self, object: &ObjectRef) -> bool {
        self.sent.lock().unwrap().contains(object)
    }

    /// Sends `bytes` as an object, once; its ref at once.
    pub fn put(&self, bytes: Vec<u8>) -> Result<ObjectRef> {
        let object = ObjectRef::of(&bytes);
        let bytes = Bytes::from(bytes);
        self.send(&object, move || Ok(Payload::Bytes(bytes.clone())))?;
        Ok(object)
    }

    /// Queues an admitted object without copying its bytes into an upload buffer.
    pub(crate) fn put_stored(&self, store: &Store, object: &ObjectRef) -> Result<()> {
        let (store, at) = (store.clone(), object.clone());
        self.send(object, move || store.file(&at).map(Payload::File).map_err(crate::store2::Error::refusal))
    }

    fn send(&self, object: &ObjectRef, make: impl Fn() -> Result<Payload> + Send + Sync + 'static) -> Result<()> {
        if !self.sent.lock().unwrap().insert(object.clone()) {
            return Ok(());
        }
        let permit = self.runtime.block_on(Arc::clone(&self.permits).acquire_owned()).expect("never closed");
        let (endpoint, at) = (Arc::clone(&self.endpoint), object.clone());
        let task = self.runtime.spawn(async move {
            let _permit = permit;
            endpoint.upload(&at, &make).await.map(drop)
        });
        self.sending.lock().unwrap().push(task);
        Ok(())
    }

    /// Waits for every object put so far: the first failure.
    pub fn flush(&self) -> Result<()> {
        let tasks = std::mem::take(&mut *self.sending.lock().unwrap());
        let mut first = self.failed.lock().unwrap().clone().map_or(Ok(()), Err);
        for task in tasks {
            let done = match self.runtime.block_on(task) {
                Ok(done) => done,
                Err(e) => Err(Refusal { code: Code::TRANSFER_FAILED, detail: e.to_string() }),
            };
            if first.is_ok() {
                first = done;
            }
        }
        if let Err(ref failure) = first {
            *self.failed.lock().unwrap() = Some(failure.clone());
        }
        first
    }

    /// A writer's commit: its `header` as `variant` of a tree with `files`, each sent after
    /// what it cites, then the tree added to repo `name`'s checkpoints. The header's ref.
    pub fn commit(&self, name: &str, variant: &str, header: Vec<u8>, files: &BTreeMap<String, ObjectRef>) -> Result<ObjectRef> {
        self.flush()?;
        let header = self.put(header)?;
        self.flush()?;
        let mut tree = Tree { files: files.clone() };
        tree.set_variant(variant, header.clone());
        let tree = self.put(tree.to_json().map_err(|e| Refusal { code: Code::UNKNOWN_FORMAT, detail: e.to_string() })?)?;
        self.flush()?;
        let named = self.endpoint.update_repo(name, |old| Ok(Some(with_tree(old, name, &tree, None))));
        self.runtime.block_on(named)?;
        Ok(header)
    }
}

/// The ETag a directory (or a server that sends none) gives a document: `"sha256:<hex>"`.
fn etag_of(bytes: &[u8]) -> String {
    format!("\"sha256:{}\"", hex(&digest(bytes)))
}

fn held(reply: &Reply, object: &ObjectRef) -> Result<()> {
    match reply.header("content-length").and_then(|v| v.parse::<u64>().ok()) {
        Some(length) if length != object.length => refuse(
            Code::LENGTH_MISMATCH,
            format!("{}: {length} B, not {}", reply.url, object.length),
        ),
        _ => Ok(()),
    }
}

fn decode_hex(text: &str) -> Vec<u8> {
    (0..text.len() / 2).map(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).unwrap_or(0)).collect()
}

/// The protocol's error envelope: `{"error": {"code", "message", "detail"?}}`.
fn envelope(body: &[u8]) -> Option<(String, String, Value)> {
    let parsed: Value = serde_json::from_slice(body).ok()?;
    let error = parsed.get("error")?;
    let code = error.get("code")?.as_str()?.to_owned();
    let message = error.get("message").and_then(Value::as_str).unwrap_or_default().to_owned();
    Some((code, message, error.get("detail").cloned().unwrap_or(Value::Null)))
}

/// A refusal for `reply`: the envelope's code when there is one, else the status's.
fn refusal(reply: &Reply, what: &str) -> Refusal {
    let (named, message, detail) = envelope(&reply.body).unwrap_or_else(|| {
        let text = String::from_utf8_lossy(&reply.body[..reply.body.len().min(200)]).trim().to_owned();
        (String::new(), text, Value::Null)
    });
    let code = match (named.as_str(), reply.status) {
        ("digest_mismatch", _) => Code::OBJECT_ID_MISMATCH,
        ("closure_incomplete", _) => Code::CHECKPOINT_CONFLICT,
        ("precondition_failed" | "precondition_required" | "policy", _) | (_, 409 | 412 | 428) => Code::REPOSITORY_CONFLICT,
        ("not_found", _) | (_, 404 | 410) => Code::REF_NOT_FOUND,
        ("unauthorized" | "denied", _) | (_, 401 | 403) => Code::CREDENTIAL_REQUIRED,
        ("too_large", _) | (_, 413) => Code::SIZE_CAP,
        ("rate_limited" | "unavailable", _) => Code::HUB_UNREACHABLE,
        _ => Code::HUB_REFUSED,
    };
    let named = if named.is_empty() { String::new() } else { format!(" {named}") };
    let detail = if detail.is_null() { String::new() } else { format!(" {detail}") };
    Refusal { code, detail: format!("{what} {}: {}{named}: {message}{detail}", reply.url, reply.status) }
}

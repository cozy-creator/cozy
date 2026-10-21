//! The distribution client (`spec/distribution.md`): one endpoint `B` serving
//! `B/refs/<org>/<name>.json` and `B/blobs/<aa>/<bb>/<sha256>`, over HTTP(S) or a directory.
//!
//! - **Reads.** The repo document with its ETag (conditional GET); objects as Fetcher locations,
//!   whose redirects the client follows without credentials across origins.
//! - **Uploads.** Per object HEAD → PUT (to the HEAD's `Location`, else here) → HEAD; `412` means
//!   present. [`upload_tree`] sends chunks, assets and repo files, then headers, then the tree
//!   (the commit), then moves a tag to it.
//! - **Refs.** [`Endpoint::update_repo`] PUTs the whole document under `If-Match` or
//!   `If-None-Match: *` and, on `412`, reads it again: another writer's progress is re-applied
//!   onto; a `412` over a document that has not changed is refused (no one made progress).
//! - **A directory** writes through a store over it: objects verified, temp and rename; refs under
//!   the store's record lock, read and replaced inside it.
//! - **Weather** (transport failures, 408/425/429/5xx) is asked again at `Retry-After` or a
//!   doubling wait, until nothing has moved for the stall rule ([`super::ask`]).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use base64::engine::{general_purpose::STANDARD, Engine as _};
use super::net::Uri;
use serde_json::Value;

use super::ask::{self, Patience, Tried};
use super::net::{join, same_origin, Body, Client, GetError, HeaderMap};
use super::origin::{Location, Origin};
use super::policy::is_loopback_origin;
use super::token::{default_for, s3_base, Request, TokenProvider};
use crate::err::{refuse, Code, Refusal, Result};
use crate::format::{is_digest, repo_key, repo_name, Header, ObjectRef, Repo, Tree};
use crate::sha256::{digest, hex, hex_digest};
use crate::store::Store;

/// The tag an upload moves when none is named.
pub const MAIN: &str = "main";
/// The most bytes a repo document may have.
const REPO_BYTES: usize = 16 << 20;
/// Objects an upload sends at once.
const STREAMS: usize = 16;

/// An endpoint of the distribution protocol.
pub struct Endpoint {
    url: String,
    at: At,
    client: Client,
    token: Option<Arc<dyn TokenProvider>>,
    patience: Patience,
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
    /// The commit the tag now names: its tree before, with the upload's files on top.
    pub commit: ObjectRef,
    pub objects: usize,
    pub sent: usize,
    pub bytes: u64,
    /// The requested repo update was already present, independent of blob deduplication.
    pub converged: bool,
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
        Ok(Endpoint { url, at, client: client.clone(), token, patience: Patience::default() })
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
        self.repo_attempt(name).await.map_err(GetError::refusal)
    }

    /// Preserve credential callback origin until the caller's cache fallback decision.
    pub(crate) async fn repo_attempt(&self, name: &str) -> std::result::Result<Option<(Repo, String)>, GetError> {
        Ok(match self.repo_since_attempt(name, None).await? {
            Fresh::Doc(repo, etag) => Some((repo, etag)),
            _ => None,
        })
    }

    /// The repo document `name`, or `Unchanged` while it still has ETag `etag`.
    pub async fn repo_since(&self, name: &str, etag: Option<&str>) -> Result<Fresh> {
        self.repo_since_attempt(name, etag).await.map_err(GetError::refusal)
    }

    async fn repo_since_attempt(&self, name: &str, etag: Option<&str>) -> std::result::Result<Fresh, GetError> {
        let (bytes, etag) = match self.repo_bytes_attempt(name, etag).await? {
            Fresh::Doc(bytes, etag) => (bytes, etag),
            Fresh::Unchanged => return Ok(Fresh::Unchanged),
            Fresh::Absent => return Ok(Fresh::Absent),
        };
        let what = format!("{}/{}", self.url, repo_key(name));
        let repo = Repo::from_json(&bytes).map_err(|e| Refusal { code: Code::UNKNOWN_FORMAT, detail: format!("{what}: {e}") })?;
        match repo.name.eq_ignore_ascii_case(name) {
            true => Ok(Fresh::Doc(repo, etag)),
            false => refuse(Code::UNKNOWN_FORMAT, format!("{what} names {:?}", repo.name)).map_err(GetError::Exchange),
        }
    }

    /// The repo document `name` as its bytes, unparsed, or `Unchanged` while it has ETag `etag`.
    pub async fn repo_bytes(&self, name: &str, etag: Option<&str>) -> Result<Fresh<Vec<u8>>> {
        self.repo_bytes_attempt(name, etag).await.map_err(GetError::refusal)
    }

    async fn repo_bytes_attempt(&self, name: &str, etag: Option<&str>) -> std::result::Result<Fresh<Vec<u8>>, GetError> {
        check_repo_name(name)?;
        let key = repo_key(name);
        let (bytes, tag) = match &self.at {
            At::Dir(root, _) => match fs::read(root.join(&key)) {
                Ok(bytes) => {
                    let tag = etag_of(&bytes);
                    (bytes, tag)
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Fresh::Absent),
                Err(e) => return Err(crate::err::classify_io(root.join(&key).display().to_string(), e).into()),
            },
            At::Http(base) => {
                let url = repo_url(base, &key);
                let headers = etag.map(|e| ("if-none-match".to_owned(), e.to_owned())).into_iter().collect();
                let reply = self.exchange_attempt(Ask { method: "GET", url: &url, headers, body: None, payload: None, auth: true }).await?;
                match reply.status {
                    200 => {
                        let tag = reply.header("etag").map_or_else(|| etag_of(&reply.body), str::to_owned);
                        (reply.body, tag)
                    }
                    304 => return Ok(Fresh::Unchanged),
                    404 | 410 => return Ok(Fresh::Absent),
                    _ => return Err(refusal(&reply, "GET").into()),
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
        check_repo_name(name)?;
        let key = repo_key(name);
        let base = match &self.at {
            At::Http(base) => base,
            At::Dir(root, _) => {
                let (root, etag) = (root.clone(), etag.map(str::to_owned));
                let put = tokio::task::spawn_blocking(move || put_dir(&root, &key, &bytes, etag.as_deref()));
                return put.await.map_err(|e| Refusal { code: Code::IO_FAILED, detail: e.to_string() })?;
            }
        };
        let url = repo_url(base, &key);
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
    /// no tags, which deletes the model): a conditional write, with `change` applied again to
    /// the document as it is whenever another writer won. A refusal (`412`) while the document is
    /// still the one the write was based on is no one's progress: it fails with that reason.
    pub async fn update_repo(
        &self,
        name: &str,
        mut change: impl FnMut(Option<Repo>) -> Result<Option<Repo>>,
    ) -> Result<Repo> {
        let (mut sent, mut based): (Option<Repo>, Option<Option<String>>) = (None, None);
        loop {
            let (old, etag) = match self.repo(name).await? {
                Some((repo, etag)) => (Some(repo), Some(etag)),
                None => (None, None),
            };
            // A lost answer, or another writer that wrote the same: it is there.
            if let Some(applied) = sent.as_ref().filter(|applied| old.as_ref() == Some(*applied)) {
                return Ok(applied.clone());
            }
            if based.as_ref() == Some(&etag) {
                let at = etag.as_deref().unwrap_or("absent");
                return refuse(Code::REPOSITORY_CONFLICT, format!("{name}: the write was refused (412) though the document is still {at}: no writer made progress"));
            }
            let new = change(old.clone())?.unwrap_or_else(|| Repo::new(name));
            if !new.name.eq_ignore_ascii_case(name) {
                return refuse(Code::KEY_GRAMMAR, format!("repo {:?} cannot be {name}", new.name));
            }
            if old.as_ref() == Some(&new) || (old.is_none() && new.tags.is_empty()) {
                return Ok(new);
            }
            let bytes = new.to_json().map_err(|e| Refusal { code: Code::UNKNOWN_FORMAT, detail: e.to_string() })?;
            if self.put_repo_bytes(name, bytes, etag.as_deref()).await? {
                return Ok(new);
            }
            (sent, based) = (Some(new), Some(etag));
        }
    }

    /// An object named by digest, including its length, or `None` when absent or unreadable.
    pub async fn object(&self, sha256: &str) -> Result<Option<ObjectRef>> {
        let sha256 = sha256.strip_prefix("sha256:").unwrap_or(sha256);
        if !is_digest(sha256) {
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

    /// A small object (a tree) read whole, following redirects, and verified.
    pub async fn document(&self, object: &ObjectRef) -> Result<Vec<u8>> {
        if object.length > REPO_BYTES as u64 {
            return refuse(Code::SIZE_CAP, format!("{}: {} B is not a document", object.id(), object.length));
        }
        let bytes = match &self.at {
            At::Dir(root, _) => {
                let path = root.join(object.key());
                fs::read(&path).map_err(|e| crate::err::classify_io(path.display().to_string(), e))?
            }
            At::Http(base) => {
                let url = format!("{base}/{}", object.key());
                let mut body = Vec::new();
                let mut sink = |at: u64, frame: &[u8]| {
                    body.truncate(at as usize);
                    body.extend_from_slice(frame);
                    match body.len() as u64 > object.length {
                        true => refuse(Code::OBJECT_ID_MISMATCH, format!("GET {url}: more than {} B", object.length)),
                        false => Ok(()),
                    }
                };
                let got = ask::get(&self.client, &url, &[], self.token.as_deref(), &self.patience, None, &mut sink).await;
                match got.map_err(GetError::refusal)? {
                    Ok(_) => body,
                    Err(status @ (404 | 410)) => return refuse(Code::OBJECT_ABSENT, format!("GET {url}: {status}")),
                    Err(status) => return refuse(Code::HUB_REFUSED, format!("GET {url}: {status}")),
                }
            }
        };
        match ObjectRef::of(&bytes) == *object {
            true => Ok(bytes),
            false => refuse(Code::OBJECT_ID_MISMATCH, format!("{} does not match its bytes", object.id())),
        }
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
                put.map_err(crate::store::Error::refusal)?;
                return Ok(true);
            }
        };
        let url = format!("{base}/{}", object.key());
        let head = || Ask { method: "HEAD", url: &url, headers: vec![], body: None, payload: None, auth: true };
        let mut reply = self.exchange(head()).await?;
        // Places the bytes went, without their signatures. A confirming HEAD (where a redirecting
        // server verifies what it was sent) may name another place, as when another account's
        // copy landed first and this writer proves it holds the bytes: they go there too.
        let mut sent: Vec<String> = vec![];
        loop {
            match reply.status {
                200 => return held(&reply, object).map(|()| !sent.is_empty()),
                404 | 410 => {}
                _ => return Err(refusal(&reply, if sent.is_empty() { "HEAD" } else { "the confirming HEAD" })),
            }
            // A writer's HEAD of an absent object may say where to PUT it (a presigned URL).
            let target = match reply.header("location") {
                Some(at) => join(&url.parse::<Uri>().expect("sent"), at),
                None => url.clone(),
            };
            let place = target.split('?').next().unwrap_or_default().to_owned();
            if sent.contains(&place) {
                return Err(refusal(&reply, "the confirming HEAD"));
            }
            let checksum = STANDARD.encode(decode_hex(&object.sha256));
            let headers = vec![
                ("content-length".to_owned(), object.length.to_string()),
                ("x-amz-checksum-sha256".to_owned(), checksum),
                ("if-none-match".to_owned(), "*".to_owned()),
            ];
            let auth = same_origin(&target, &url);
            let ask = Ask { method: "PUT", url: &target, headers, body: Some(payload), payload: Some(&object.sha256), auth };
            let put = self.exchange(ask).await?;
            if !matches!(put.status, 200 | 201 | 204 | 412) {
                return Err(refusal(&put, "PUT"));
            }
            sent.push(place);
            reply = self.exchange(head()).await?;
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
        let opened = Store::open(root).map_err(crate::store::Error::refusal)?;
        Ok(store.get_or_init(|| opened))
    }

    /// The provider's headers for `method url`, carrying `headers` with `payload`.
    async fn authorize(&self, method: &str, url: &str, headers: &[(String, String)], payload: Option<&str>) -> Result<Vec<(String, String)>> {
        match &self.token {
            Some(token) => token.headers(&Request { method, url, headers, payload }).await,
            None => Ok(vec![]),
        }
    }

    /// One exchange, its answer read whole, asked again by the one rule ([`ask::again`]); the
    /// provider may ask once more (a nonce, a refresh).
    async fn exchange(&self, ask: Ask<'_>) -> Result<Reply> {
        self.exchange_attempt(ask).await.map_err(GetError::refusal)
    }

    async fn exchange_attempt(&self, ask: Ask<'_>) -> std::result::Result<Reply, GetError> {
        let (ask, challenged, patience) = (&ask, &AtomicBool::new(false), &self.patience);
        ask::again(patience, move |moved| async move {
            let mut headers = ask.headers.clone();
            if ask.auth {
                // A provider's failure is the caller's verdict, not failed network traffic.
                let authorized = ask::watched(patience, &moved, self.authorize(ask.method, ask.url, &ask.headers, ask.payload));
                headers.extend(authorized.await.map_err(GetError::Caller)?);
            }
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
            let reply = ask::watched(patience, &moved, self.round_trip(ask.method, ask.url, &headers, body, &moved)).await?;
            let sent = Request { method: ask.method, url: ask.url, headers: &headers, payload: ask.payload };
            let again = match (ask.auth, &self.token) {
                (true, Some(token)) => token.retry(&sent, reply.status, &reply.headers).map_err(GetError::Caller)?,
                _ => false,
            };
            let why = format!("{} {}: {}", ask.method, ask.url, reply.status);
            Ok(if again && !challenged.swap(true, Ordering::Relaxed) {
                Tried::Again { why, wait: Some(Duration::ZERO) }
            } else if ask::weather(reply.status) {
                let wait = (reply.header("retry-after")).and_then(|s| s.trim().parse().ok()).map(Duration::from_secs);
                Tried::Again { why, wait }
            } else {
                Tried::Done(reply)
            })
        })
        .await
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
}

#[async_trait]
impl Origin for Endpoint {
    async fn repo_of(&self, name: &str) -> Result<Repo> {
        match self.repo(name).await? {
            Some((repo, _)) => Ok(repo),
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

/// `repo` (`name`'s, when there is none) with tag `tag` moved to `commit`: an existing tag keeps
/// its place and flags, a new one is private and last.
pub fn with_tag(repo: Option<Repo>, name: &str, tag: &str, commit: &ObjectRef) -> Repo {
    let mut repo = repo.unwrap_or_else(|| Repo::new(name));
    repo.set_tag(tag, commit.clone());
    repo
}

/// What one commit made: its tree and ref, the repo document naming it, whether the tag already
/// named that tree, and whether the tree's object was sent (the endpoint lacked it).
pub struct Committed {
    pub commit: ObjectRef,
    pub tree: Tree,
    pub repo: Repo,
    pub converged: bool,
    pub sent: bool,
}

/// The one commit path (git's rule, as Hugging Face's `upload_folder`): tag `tag` of repo `name`
/// moves to its current tree (none: empty) with the paths a `delete` pattern matches removed and
/// `files` added or replaced. Every object `files` cite must be at the endpoint already. When
/// another writer moved the tag first, its tree is the new parent and the change is applied
/// again, for as long as writers make progress ([`Endpoint::update_repo`]).
pub async fn commit_onto(endpoint: &Endpoint, name: &str, tag: &str, files: &BTreeMap<String, ObjectRef>, delete: &[&str]) -> Result<Committed> {
    if !crate::format::tag_name(tag) {
        return refuse(Code::KEY_GRAMMAR, format!("{tag:?} is not a tag name"));
    }
    loop {
        let parent = endpoint.repo(name).await?.and_then(|(r, _)| Some(r.tag(tag)?.commit.clone()));
        let base = match &parent {
            Some(commit) => Tree::from_json(&endpoint.document(commit).await?).map_err(|e| invalid_tree(commit, e))?,
            None => Tree::default(),
        };
        let tree = base.commit(files, delete);
        let bytes = Bytes::from(tree.to_json().map_err(|e| Refusal { code: Code::UNKNOWN_FORMAT, detail: e.to_string() })?);
        let commit = ObjectRef::of(&bytes);
        let sent = endpoint.upload(&commit, &move || Ok(Payload::Bytes(bytes.clone()))).await?;
        // Other tags' edits are re-applied by update_repo; this tag moving on is a new parent.
        let mut moved = false;
        let named = endpoint.update_repo(name, |old| {
            let now = old.as_ref().and_then(|r| r.tag(tag)).map(|t| &t.commit);
            moved = now != parent.as_ref() && now != Some(&commit);
            match moved {
                true => refuse(Code::REPOSITORY_CONFLICT, format!("{name}@{tag} moved")),
                false => Ok(Some(with_tag(old, name, tag, &commit))),
            }
        });
        match named.await {
            Ok(repo) => return Ok(Committed { converged: parent.as_ref() == Some(&commit), commit, tree, repo, sent }),
            Err(_) if moved => continue,
            Err(e) => return Err(e),
        }
    }
}

fn invalid_tree(commit: &ObjectRef, e: crate::format::Error) -> Refusal {
    Refusal { code: Code::UNKNOWN_FORMAT, detail: format!("{}: {e}", commit.id()) }
}

/// Sends what `endpoint` lacks of the closure of `tree`'s files in `store` (chunks, assets and
/// repo files, then headers), then commits those files onto tag `tag` of repo `name`
/// ([`commit_onto`]).
pub async fn upload_tree(endpoint: &Endpoint, name: &str, tree: &ObjectRef, store: &Store, tag: &str) -> Result<Uploaded> {
    upload_tree_with(endpoint, name, tree, store, tag, &[], &UploadControl {
        cancelled: &|| false,
        progress: &|_, _| {},
        commit: &|| true,
    }).await
}

/// Cooperative control of one upload. `commit` is called once after the objects land,
/// before the atomic ref update. Once it accepts, cancellation no longer interrupts the
/// ref's conditional-update and lost-response recovery loop. Progress counts verified bytes,
/// including objects the endpoint already holds.
pub struct UploadControl<'a> {
    pub cancelled: &'a (dyn Fn() -> bool + Sync),
    pub progress: &'a (dyn Fn(u64, u64) + Sync),
    pub commit: &'a (dyn Fn() -> bool + Sync),
}

/// [`upload_tree`] deleting the paths `delete` patterns match, with cancellation before the ref
/// update, progress, and a final commit gate.
pub async fn upload_tree_with(endpoint: &Endpoint, name: &str, tree: &ObjectRef, store: &Store, tag: &str, delete: &[&str], control: &UploadControl<'_>) -> Result<Uploaded> {
    if !crate::format::tag_name(tag) {
        return refuse(Code::KEY_GRAMMAR, format!("{tag:?} is not a tag name"));
    }
    let stopped = || Refusal { code: Code::TRANSFER_FAILED, detail: "upload stopped before its ref update; admitted objects remain available".into() };
    if (control.cancelled)() {
        return Err(stopped());
    }
    let refusal = crate::store::Error::refusal;
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
    let total = leaves.iter().chain(&headers)
        .try_fold(0u64, |sum, object| sum.checked_add(object.length))
        .ok_or_else(|| Refusal { code: Code::ARITH_OVERFLOW, detail: "upload closure byte count overflows".into() })?;
    let mut verified = 0;
    (control.progress)(0, total);
    for stage in [leaves, headers] {
        let mut sending = futures::stream::iter(stage)
            .map(|object| async move {
                let (from, at) = (store.clone(), object.clone());
                let payload = move || from.file(&at).map(Payload::File).map_err(crate::store::Error::refusal);
                endpoint.upload(&object, &payload).await.map(|s| (s, object.length))
            })
            .buffer_unordered(STREAMS);
        loop {
            if (control.cancelled)() {
                return Err(stopped());
            }
            let done = tokio::select! {
                done = sending.next() => done,
                () = tokio::time::sleep(ask::TICK) => continue,
            };
            let Some(done) = done else { break };
            let (pushed, length) = done?;
            if pushed {
                (sent, bytes) = (sent + 1, bytes + length);
            }
            verified += length;
            (control.progress)(verified, total);
        }
    }
    if (control.cancelled)() || !(control.commit)() {
        return Err(stopped());
    }
    let done = commit_onto(endpoint, name, tag, &parsed.files, delete).await?;
    if done.sent {
        (sent, bytes) = (sent + 1, bytes + done.commit.length);
    }
    Ok(Uploaded { repo: done.repo, commit: done.commit, objects, sent, bytes, converged: done.converged })
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
/// at commit the header, then the tree (the commit), then the tag moved to it.
/// File-backed objects queue without blocking the caller's runtime. Byte-backed puts and
/// finalization block, so the finalizer runs outside the runtime that sends.
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

/// Concurrent transfers; byte-backed objects are also bounded by this limit.
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
        if !self.sent.lock().unwrap().insert(object.clone()) {
            return Ok(());
        }
        let (store, at) = (store.clone(), object.clone());
        let (endpoint, permits) = (Arc::clone(&self.endpoint), Arc::clone(&self.permits));
        let task = self.runtime.spawn(async move {
            let _permit = permits.acquire_owned().await.expect("never closed");
            let make = || store.file(&at).map(Payload::File).map_err(crate::store::Error::refusal);
            endpoint.upload(&at, &make).await.map(drop)
        });
        self.sending.lock().unwrap().push(task);
        Ok(())
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

    /// A writer's commit: its `header`, sent after what it cites, as `variant` with `files`
    /// committed onto tag `tag` of repo `name` ([`commit_onto`]).
    pub fn commit(&self, name: &str, tag: &str, variant: &str, header: Vec<u8>, files: &BTreeMap<String, ObjectRef>) -> Result<Committed> {
        self.flush()?;
        let header = self.put(header)?;
        self.flush()?;
        let mut tree = Tree { files: files.clone() };
        tree.set_variant(variant, header);
        self.runtime.block_on(commit_onto(&self.endpoint, name, tag, &tree.files, &[]))
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        for task in self.sending.get_mut().unwrap_or_else(std::sync::PoisonError::into_inner).drain(..) {
            task.abort();
        }
    }
}

// Generic endpoints use format segments, independent of a Hub's catalog name policy.
fn check_repo_name(name: &str) -> Result<()> {
    match repo_name(name) {
        true => Ok(()),
        false => refuse(Code::KEY_GRAMMAR, format!("{name:?} is not <org>/<name>")),
    }
}

// repo_key has already folded ASCII case. Escape its segments for HTTP only; on disk the
// same key stays raw. Keeping only the separators and URI-unreserved bytes avoids interpreting
// literal percent signs, query/fragment delimiters or UTF-8 bytes as URL structure.
fn repo_url(base: &str, key: &str) -> String {
    const HEX: &[u8] = b"0123456789ABCDEF";
    let mut url = format!("{base}/");
    for byte in key.bytes() {
        if byte.is_ascii_alphanumeric() || b"-._~/".contains(&byte) {
            url.push(char::from(byte));
        } else {
            url.push('%');
            url.push(char::from(HEX[usize::from(byte >> 4)]));
            url.push(char::from(HEX[usize::from(byte & 15)]));
        }
    }
    url
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
        ("policy", _) if detail.get("reason").and_then(Value::as_str) == Some("capability_exceeded") => Code::CREDENTIAL_REQUIRED,
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

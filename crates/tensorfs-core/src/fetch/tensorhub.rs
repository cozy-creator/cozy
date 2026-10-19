//! Tensorhub, the default origin ([`TENSORHUB`]). The Hub names a model's tree by name, release
//! and variant (its wire's `lane`; its `manifest` is the tree) and lists its closure
//! (`POST /v1/tensorfs/closure`), then presigns its objects (`POST /v1/tensorfs/presign`); the
//! Fetcher reads them from the presigned URLs, which see no credential. Public releases are
//! read anonymously; a bearer token, or a caller's own [`Credential`], reads the private trees
//! it grants.
//!
//! Wire: closure `{"ref":"org/name[@release][@sha256:<hex>]","lane","after"?}` →
//! `{complete, lane, manifest, model, objects[], presign_max_digests, public, release, scope}`,
//! paged by digest; presign `{"digests":[hex…],"manifest":hex}` (the checkpoint they belong
//! to) → `{urls:{hex:url}, expires_at_unix, server_time_unix}`; a refusal is
//! `{"error":{"code","message","remedy"}}`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use super::net::{Client, HeaderMap};
use super::origin::{split_name, Location, Origin};
use crate::err::{refuse, Code, Refusal, Result};
use crate::format::ObjectRef;

/// The origin a pull reads when none is named.
pub const TENSORHUB: &str = "https://tensorhub.com";

/// Asks of one Hub call while it answers "not now".
const ATTEMPTS: u32 = 4;
const RETRYABLE: [u16; 7] = [408, 425, 429, 500, 502, 503, 504];
/// The most bytes one Hub answer may have.
const ANSWER_BYTES: u64 = 16 << 20;

/// What a request to the Hub presents: a bearer token, or a proof made per request (DPoP).
#[async_trait]
pub trait Credential: Send + Sync {
    /// Headers for `method url`, a Hub URL.
    async fn present(&self, method: &str, url: &str) -> Vec<(String, String)>;

    /// The Hub answered `status` and `headers` to a request that sent `sent`: true asks it once
    /// more (a fresh nonce).
    fn observe(&self, url: &str, sent: &[(String, String)], status: u16, headers: &HeaderMap) -> bool {
        let _ = (url, sent, status, headers);
        false
    }
}

struct Bearer(String);

#[async_trait]
impl Credential for Bearer {
    async fn present(&self, _: &str, _: &str) -> Vec<(String, String)> {
        vec![("authorization".into(), format!("Bearer {}", self.0))]
    }
}

/// What the Hub named: the model, the release and variant it chose (both empty for a tree
/// asked by hash) and their tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    pub model: String,
    pub release: String,
    pub variant: String,
    pub tree: ObjectRef,
    /// Anyone may read it.
    pub public: bool,
}

#[derive(Deserialize)]
struct Page {
    complete: bool,
    lane: String,
    manifest: ObjectRef,
    model: String,
    objects: Vec<ObjectRef>,
    presign_max_digests: usize,
    #[serde(default)]
    public: bool,
    release: String,
    scope: String,
}

#[derive(Deserialize)]
struct Presigned {
    urls: HashMap<String, String>,
    expires_at_unix: u64,
    server_time_unix: u64,
}

/// The checkpoint `locate` presigns: its pinned reference, lane and manifest.
#[derive(Clone, PartialEq, Eq)]
struct At {
    reference: String,
    lane: String,
    manifest: ObjectRef,
}

#[derive(Default)]
struct State {
    at: Option<At>,
    /// Its objects, once listed, and the Hub's presign bound.
    listed: Option<(Vec<ObjectRef>, usize)>,
    /// Presigned URLs, each until half its Hub-stated life.
    urls: HashMap<String, (String, SystemTime)>,
}

pub struct TensorhubOrigin {
    client: Client,
    base: String,
    credential: Option<Arc<dyn Credential>>,
    public: AtomicBool,
    state: tokio::sync::Mutex<State>,
}

impl TensorhubOrigin {
    /// The Hub at `base`, anonymous.
    pub fn new(client: &Client, base: &str) -> Result<TensorhubOrigin> {
        let base = base.trim_end_matches('/').to_string();
        client.policy().admit(&base)?;
        Ok(TensorhubOrigin {
            client: client.clone(),
            base,
            credential: None,
            public: AtomicBool::new(false),
            state: tokio::sync::Mutex::default(),
        })
    }

    /// Sends `Authorization: Bearer <token>` to the Hub.
    pub fn bearer(self, token: impl Into<String>) -> TensorhubOrigin {
        self.credential(Arc::new(Bearer(token.into())))
    }

    /// Presents `credential` on every Hub call.
    pub fn credential(mut self, credential: Arc<dyn Credential>) -> TensorhubOrigin {
        self.credential = Some(credential);
        self
    }

    /// Binds `locate` to `name`'s `tree` at `release` (none: the owner's tree by hash) and
    /// `variant`, without asking the Hub.
    pub fn at(mut self, name: &str, release: Option<&str>, variant: &str, tree: &ObjectRef) -> TensorhubOrigin {
        self.state.get_mut().at = Some(At {
            reference: pinned(name, release.unwrap_or(""), tree),
            lane: variant.into(),
            manifest: tree.clone(),
        });
        self
    }

    /// `name` at `release` (the newest when `None`; `sha256:<hex>` names the owner's tree by
    /// hash) and `variant` (the tree's only one when `None`), and binds `locate` to it.
    pub async fn named(&self, name: &str, release: Option<&str>, lane: Option<&str>) -> Result<Resolved> {
        split_name(name)?;
        let reference = match release {
            Some(release) => format!("{name}@{release}"),
            None => name.to_string(),
        };
        let asked = lane.unwrap_or("");
        let page = self.page(&reference, asked, "").await?;
        self.public.store(page.public, Ordering::Relaxed);
        let lane = match page.lane.is_empty() {
            true => asked.to_string(),
            false => page.lane.clone(),
        };
        let at = At {
            reference: pinned(&page.model, &page.release, &page.manifest),
            lane,
            manifest: page.manifest.clone(),
        };
        let mut state = self.state.lock().await;
        if state.at.as_ref() != Some(&at) {
            state.listed = match page.complete {
                true => Some((closure(page.objects, &page.manifest), page.presign_max_digests)),
                false => None,
            };
            state.at = Some(at);
        }
        Ok(Resolved {
            model: page.model,
            release: page.release,
            variant: page.lane,
            tree: page.manifest,
            public: page.public,
        })
    }

    /// Every object of the checkpoint at `at`, however many pages it takes, and the presign
    /// bound.
    async fn list(&self, At { reference, lane, .. }: &At) -> Result<(Vec<ObjectRef>, usize)> {
        let first = self.page(reference, lane, "").await?;
        self.public.store(first.public, Ordering::Relaxed);
        let (manifest, bound) = (first.manifest.clone(), first.presign_max_digests);
        let mut objects = first.objects;
        let mut complete = first.complete;
        while !complete {
            let Some(last) = objects.last().map(|o| o.sha256.clone()) else {
                return refuse(Code::HUB_REFUSED, "the closure is incomplete and its page is empty");
            };
            let next = self.page(reference, lane, &last).await?;
            if next.manifest != manifest || (!next.complete && next.objects.last().is_none_or(|o| o.sha256 <= last)) {
                return refuse(Code::HUB_REFUSED, format!("the closure page after {last} changed tree or did not advance"));
            }
            complete = next.complete;
            objects.extend(next.objects);
        }
        Ok((closure(objects, &manifest), bound))
    }

    async fn page(&self, reference: &str, lane: &str, after: &str) -> Result<Page> {
        let mut body = json!({"lane": lane, "ref": reference});
        if !after.is_empty() {
            body["after"] = json!(after);
        }
        let page: Page = serde_json::from_value(self.call("/v1/tensorfs/closure", &body).await?)
            .map_err(|e| Refusal { code: Code::HUB_REFUSED, detail: format!("the closure answer: {e}; update Tensorhub") })?;
        if page.scope != "runtime" {
            return refuse(Code::HUB_REFUSED, format!("closure scope {:?} is not runtime", page.scope));
        }
        if page.presign_max_digests == 0 {
            return refuse(Code::HUB_REFUSED, "the Hub declared presign_max_digests = 0");
        }
        Ok(page)
    }

    async fn presign(&self, digests: &[String], manifest: &ObjectRef) -> Result<Presigned> {
        let body = json!({"digests": digests, "manifest": manifest.sha256});
        serde_json::from_value(self.call("/v1/tensorfs/presign", &body).await?)
            .map_err(|e| Refusal { code: Code::HUB_REFUSED, detail: format!("the presign answer: {e}; update Tensorhub") })
    }

    /// One Hub call, asked again only while the Hub (or the link to it) says "not now", at the
    /// wait it states or a doubling backoff.
    async fn call(&self, path: &str, body: &Value) -> Result<Value> {
        let url = format!("{}{path}", self.base);
        let body = serde_json::to_vec(body).expect("a JSON value serializes");
        let mut wait = Duration::from_secs(1);
        let mut last = String::new();
        for attempt in 1..=ATTEMPTS {
            match self.ask(&url, &body).await {
                Ok(value) => return Ok(value),
                Err(Ask::Final(refusal)) => return Err(refusal),
                Err(Ask::Weather(detail, stated)) => {
                    last = detail;
                    if attempt < ATTEMPTS {
                        tokio::time::sleep(stated.unwrap_or(wait)).await;
                        wait *= 2;
                    }
                }
            }
        }
        refuse(Code::HUB_UNREACHABLE, format!("{last} — on all {ATTEMPTS} asks"))
    }

    async fn ask(&self, url: &str, body: &[u8]) -> std::result::Result<Value, Ask> {
        let mut retried = false;
        loop {
            let sent = match &self.credential {
                Some(credential) => credential.present("POST", url).await,
                None => Vec::new(),
            };
            let answer = match self.client.post(url, &sent, body.to_vec(), ANSWER_BYTES).await {
                Ok(answer) => answer,
                Err(e) if matches!(e.code, Code::TRANSFER_FAILED | Code::FD_HEADROOM) => return Err(Ask::Weather(e.detail, None)),
                Err(e) => return Err(Ask::Final(e)),
            };
            let again = (self.credential.as_ref()).is_some_and(|c| c.observe(url, &sent, answer.status, &answer.headers));
            if again && !retried {
                retried = true;
                continue;
            }
            let status = answer.status;
            if (200..300).contains(&status) {
                return serde_json::from_slice(&answer.body)
                    .map_err(|e| Ask::Final(Refusal { code: Code::HUB_REFUSED, detail: format!("{url} answered unreadable JSON: {e}") }));
            }
            let Some(refusal) = envelope(status, &answer.body, url) else {
                return Err(Ask::Weather(format!("{url} answered HTTP {status} with no Hub envelope"), None));
            };
            if !RETRYABLE.contains(&status) {
                return Err(Ask::Final(refusal));
            }
            let parsed: Value = serde_json::from_slice(&answer.body).unwrap_or_default();
            let stated = (parsed["retry_after_unix"].as_u64())
                .zip(parsed["server_time_unix"].as_u64())
                .map(|(at, now)| Duration::from_secs(at.saturating_sub(now)));
            return Err(Ask::Weather(refusal.detail, stated));
        }
    }
}

#[async_trait]
impl Origin for TensorhubOrigin {
    async fn resolve(&self, name: &str, release: Option<&str>, variant: Option<&str>) -> Result<ObjectRef> {
        Ok(self.named(name, release, variant).await?.tree)
    }

    async fn locate(&self, objects: &[ObjectRef]) -> Result<Vec<Location>> {
        let mut state = self.state.lock().await;
        let now = SystemTime::now();
        let fresh = |state: &State, sha: &str| state.urls.get(sha).is_some_and(|(_, until)| *until > now);
        let mut ask: Vec<String> = (objects.iter())
            .filter(|o| !fresh(&state, &o.sha256))
            .map(|o| o.sha256.clone())
            .collect();
        ask.sort();
        ask.dedup();
        if !ask.is_empty() {
            let Some(at) = state.at.clone() else {
                return refuse(Code::REF_NOT_FOUND, "no checkpoint named: resolve one before locating its objects");
            };
            if state.listed.is_none() {
                state.listed = Some(self.list(&at).await?);
            }
            let (listed, bound) = state.listed.clone().expect("listed");
            // One round trip presigns the asked objects and as many of the rest as fit.
            for object in &listed {
                if ask.len() >= bound {
                    break;
                }
                if !fresh(&state, &object.sha256) && !ask.contains(&object.sha256) {
                    ask.push(object.sha256.clone());
                }
            }
            for batch in ask.chunks(bound) {
                let mut digests = batch.to_vec();
                digests.sort();
                let presigned = self.presign(&digests, &at.manifest).await?;
                let life = Duration::from_secs(presigned.expires_at_unix.saturating_sub(presigned.server_time_unix));
                let until = SystemTime::now() + life / 2;
                for digest in digests {
                    let Some(url) = presigned.urls.get(&digest) else {
                        return refuse(Code::HUB_REFUSED, format!("sha256:{digest}: asked for and not presigned"));
                    };
                    state.urls.insert(digest, (url.clone(), until));
                }
            }
        }
        Ok((objects.iter())
            .map(|o| {
                let (url, until) = state.urls[&o.sha256].clone();
                Location::Url { url, headers: vec![], expires: Some(until) }
            })
            .collect())
    }

    fn public(&self) -> bool {
        self.public.load(Ordering::Relaxed)
    }
}

/// The reference that pins `manifest`: a page walk names the tree it started on.
fn pinned(name: &str, release: &str, manifest: &ObjectRef) -> String {
    match release.is_empty() {
        true => format!("{name}@sha256:{}", manifest.sha256),
        false => format!("{name}@{release}@sha256:{}", manifest.sha256),
    }
}

/// A closure's objects with its manifest, sorted.
fn closure(mut objects: Vec<ObjectRef>, manifest: &ObjectRef) -> Vec<ObjectRef> {
    objects.push(manifest.clone());
    objects.sort();
    objects.dedup();
    objects
}

enum Ask {
    Final(Refusal),
    Weather(String, Option<Duration>),
}

/// The Hub's error envelope as a refusal, verbatim.
fn envelope(status: u16, raw: &[u8], url: &str) -> Option<Refusal> {
    let parsed: Value = serde_json::from_slice(raw).ok()?;
    let error = &parsed["error"];
    let code = error["code"].as_str().filter(|c| !c.is_empty())?;
    let message = (error["message"].as_str())
        .filter(|m| !m.is_empty())
        .map_or_else(|| format!("HTTP {status} to {url}"), str::to_string);
    let detail = match error["remedy"].as_str().filter(|r| !r.is_empty()) {
        Some(remedy) => format!("{code}: {message} — {remedy}"),
        None => format!("{code}: {message}"),
    };
    let code = match status {
        404 => Code::REF_NOT_FOUND,
        401 | 403 => Code::CREDENTIAL_REQUIRED,
        _ => Code::HUB_REFUSED,
    };
    Some(Refusal { code, detail })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hubs_envelope_crosses_verbatim_with_its_class() {
        let raw = br#"{"error":{"code":"model_not_found","message":"no such model","remedy":"check the name"}}"#;
        let refusal = envelope(404, raw, "u").unwrap();
        assert_eq!((refusal.code, refusal.detail.as_str()), (Code::REF_NOT_FOUND, "model_not_found: no such model — check the name"));
        assert_eq!(envelope(403, br#"{"error":{"code":"x"}}"#, "u").unwrap().code, Code::CREDENTIAL_REQUIRED);
        assert!(envelope(502, b"<html>bad gateway</html>", "u").is_none(), "a proxy's page is weather, not the Hub");
    }
}

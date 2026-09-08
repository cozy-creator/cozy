//! The hub's two calls (tfs-048 protocol, th-129's server half): `closure` and `presign`.
//!
//! One call used to do everything — a hub-computed plan minting a URL for every file
//! whether held or not, which is why the pod had to pre-emptively describe its own disk.
//! Two calls replace it, and the store stops being the hub's business:
//!
//! 1. `POST {base}/v1/tensorfs/closure`, body
//!    `{"lane":"..","ref":"org/name[@release]","after":hex?,"limit":N?}`
//!    → `{"complete":bool,"lane":..,"manifest":{"length":N,"sha256":hex},"model":"org/name",
//!       "objects":[{"length":N,"sha256":hex},...],"presign_max_digests":N,"release":..,
//!       "scope":"runtime","stamps":{key:[values]}}` — NO URLs. `objects` is sorted
//!    unique by digest and EXCLUDES the manifest row, which travels in `manifest` on
//!    EVERY page. `scope` names the walk the closure covers (`runtime` today); `stamps`
//!    is the resolved lane's stamped-fact set (#567d) and may be absent.
//! 2. `POST {base}/v1/tensorfs/presign`, body `{"digests":[hex,...]}` (sorted unique, at
//!    most the `presign_max_digests` the closure declared) →
//!    `{"expires_at_unix":N,"server_time_unix":N,"urls":{hex:"https://...",...}}` — URLs
//!    only for what was asked, the manifest digest included when asked. The two clock
//!    fields are the URL's declared life stated in the HUB's own time, which is what lets
//!    a caller judge a URL's age without trusting its own clock; the pull mints against
//!    them per object rather than once for a whole closure.
//!
//! **NO REQUEST OR RESPONSE HERE GROWS WITH THE MODEL (proto-037).** The closure is
//! paginated on its own digests — the rows are sorted unique, so the cursor IS the last
//! digest and needs no server session — and the presign ask is bounded by a ceiling the
//! HUB derives from its own body cap and declares on every closure. This module holds no
//! copy of that number. It held one once: 4,096, copied from the hub's grant-subject cap,
//! which weighs 274,445 B against a route that reads 131,072. A 2,598-object closure
//! passed every bound this side knew about and arrived as a 174,146 B body.
//!
//! Errors travel in the hub's own envelope `{"error":{"code","message","remedy"}}` and
//! cross verbatim — the hub knows why it refused better than anything restated here.
//! Responses are I-JSON, ASCII, no HTML escaping (Go: `SetEscapeHTML(false)`).
//!
//! **The credential arrives as a provider trait, never a credential type**, so one
//! implementation serves whatever a caller holds. It is asked per host, so a credential
//! stays scoped to the hosts its provider names and cannot leak onto a presigned object
//! host or a redirect target. A rented pod holds nothing at all: the hub's closure and
//! presign routes are PUBLIC (tensorhub owner ruling 2026-09-03, which deleted the signed
//! DownloadDelegation these calls used to present), so the pod fetches as `Anonymous`.

use crate::canon::{self, as_arr, as_obj, as_str, as_uint, Value};
use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::{hex64, ObjectRef};
use crate::limits;
use crate::transport::http::Client;
use crate::transport::ledger::{Deadline, Ledger};
use crate::transport::policy::SourcePolicy;

/// The headers that present a credential to `host`. Asked per request and per redirect
/// hop; an empty answer presents nothing, which is what a presigned URL wants.
pub trait CredentialProvider: Sync {
    fn headers(&self, host: &str) -> Vec<(String, String)>;
}

/// No credential at all.
pub struct Anonymous;

impl CredentialProvider for Anonymous {
    fn headers(&self, _host: &str) -> Vec<(String, String)> {
        Vec::new()
    }
}

/// One bearer token, presented only to the hosts it names (tfs-052's shape).
pub struct HostToken {
    pub hosts: Vec<String>,
    pub token: String,
}

impl CredentialProvider for HostToken {
    fn headers(&self, host: &str) -> Vec<(String, String)> {
        if self.hosts.iter().any(|h| h.eq_ignore_ascii_case(host)) {
            vec![(
                "authorization".to_string(),
                format!("Bearer {}", self.token),
            )]
        } else {
            Vec::new()
        }
    }
}

/// Arbitrary presentation headers, scoped to named hosts — the generic impl a caller
/// builds when its credential is not a bearer.
pub struct ScopedHeaders {
    pub hosts: Vec<String>,
    pub headers: Vec<(String, String)>,
}

impl CredentialProvider for ScopedHeaders {
    fn headers(&self, host: &str) -> Vec<(String, String)> {
        if self.hosts.iter().any(|h| h.eq_ignore_ascii_case(host)) {
            self.headers.clone()
        } else {
            Vec::new()
        }
    }
}

/// The ONE credential spelling every surface passes (tfs-050): exactly one scoped value,
/// handed explicitly — a file, one deliberately-set variable — never inherited.
///
/// - `bearer <token>` → `authorization: Bearer <token>`
/// - empty            → anonymous, which is what a rented pod uses: the hub's closure and
///   presign routes take no credential.
pub fn credential_from_spec(spec: &str, hosts: Vec<String>) -> Result<ScopedHeaders> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Ok(ScopedHeaders {
            hosts,
            headers: Vec::new(),
        });
    }
    let mut parts = spec.split_whitespace();
    let headers = match (parts.next(), parts.next()) {
        (Some("bearer"), Some(token)) if parts.next().is_none() => {
            vec![("authorization".to_string(), format!("Bearer {token}"))]
        }
        _ => {
            return refuse(
                Code::CREDENTIAL_REQUIRED,
                "a credential is `bearer <token>` or empty — one scoped value, passed \
                 explicitly",
            )
        }
    };
    Ok(ScopedHeaders { hosts, headers })
}

/// What `closure` answered: one exact manifest, the object rows it reaches, and the
/// resolved identity the local index row needs.
#[derive(Debug, Clone)]
pub struct Closure {
    pub model: String,
    pub release: String,
    pub lane: String,
    /// `runtime` — the selected CozyTensors runtime closure — is the one scope served
    /// today; the field exists so a snapshot-scoped answer is a new value, not a silent
    /// reinterpretation.
    pub scope: String,
    pub manifest: ObjectRef,
    /// Sorted unique by digest; EXCLUDES the manifest row.
    pub objects: Vec<ObjectRef>,
    pub stamps: Vec<(String, Vec<String>)>,
    /// False when more object rows follow this page.
    pub complete: bool,
    /// The largest presign ask the HUB will read, as the hub itself stated it. `None` when
    /// it declared nothing — there is then no bound this side may invent, so the caller
    /// falls back to something derived from a fact it owns.
    pub presign_max_digests: Option<usize>,
}

/// ONE PAGE of a closure. `after` is the last digest of the page before it, exclusive;
/// the empty string asks for the first page. Callers that want the whole closure use
/// [`closure_all`], which is what a pull does.
#[allow(clippy::too_many_arguments)]
pub fn closure(
    client: &Client,
    base: &str,
    refspec: &str,
    lane: &str,
    after: &str,
    credential: &dyn CredentialProvider,
    policy: &SourcePolicy,
    deadline: Deadline,
    ledger: &Ledger,
) -> Result<Closure> {
    // No `limit`: a page size is the hub's to choose, and a client that named one would be
    // guessing at a bound it does not own. `after` is omitted entirely on the first page so
    // the request is the one an unpaginated hub already understands.
    let mut fields = vec![("lane", Value::str(lane)), ("ref", Value::str(refspec))];
    if !after.is_empty() {
        fields.insert(0, ("after", Value::str(after)));
    }
    let body = canon::write(&Value::obj(fields));
    let value = call(
        client,
        base,
        "/v1/tensorfs/closure",
        &body,
        credential,
        policy,
        deadline,
        ledger,
    )?;
    let obj = as_obj("closure", "response", &value)?;
    let field = |name: &str| {
        obj.iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v)
            .ok_or_else(|| Refusal {
                code: Code::HUB_REFUSED,
                detail: format!("the closure response carries no {name:?} field"),
            })
    };
    let manifest = object_row("closure.manifest", field("manifest")?)?;
    let mut objects = Vec::new();
    for row in as_arr("closure", "objects", field("objects")?)? {
        objects.push(object_row("closure.objects", row)?);
    }
    if objects
        .windows(2)
        .any(|pair| pair[0].sha256 >= pair[1].sha256)
    {
        return refuse(
            Code::HUB_REFUSED,
            "closure.objects must be sorted and unique by digest — the answer is the \
             content-addressed manifest and has one spelling",
        );
    }
    if objects.iter().any(|o| o.sha256 == manifest.sha256) {
        return refuse(
            Code::HUB_REFUSED,
            "closure.objects repeats the manifest row; the manifest travels once",
        );
    }
    let scope = match obj.iter().find(|(k, _)| k == "scope") {
        Some((_, v)) => as_str("closure", "scope", v)?.to_string(),
        None => "runtime".to_string(),
    };
    if scope != "runtime" {
        return refuse(
            Code::HUB_REFUSED,
            format!("closure scope {scope:?} is not one this build proves; runtime only"),
        );
    }
    let mut stamps = Vec::new();
    if let Some((_, raw)) = obj.iter().find(|(k, _)| k == "stamps") {
        for (key, values) in as_obj("closure", "stamps", raw)? {
            let mut out = Vec::new();
            for v in as_arr("closure.stamps", key, values)? {
                out.push(as_str("closure.stamps", key, v)?.to_string());
            }
            if !out.is_empty() {
                stamps.push((key.clone(), out));
            }
        }
    }
    // An absent `complete` is a hub that does not paginate, and everything it sent IS the
    // closure. A hub that paginates says so on every page.
    let complete = match obj.iter().find(|(k, _)| k == "complete") {
        Some((_, Value::Bool(done))) => *done,
        Some((_, other)) => {
            return refuse(
                Code::HUB_REFUSED,
                format!("closure.complete is {other:?}, not a boolean"),
            )
        }
        None => true,
    };
    let presign_max_digests = match obj.iter().find(|(k, _)| k == "presign_max_digests") {
        Some((_, raw)) => {
            let declared = as_uint("closure", "presign_max_digests", raw)?;
            if declared == 0 {
                return refuse(
                    Code::HUB_REFUSED,
                    "the hub declared presign_max_digests = 0, which authorizes no ask at all",
                );
            }
            Some(declared as usize)
        }
        None => None,
    };
    Ok(Closure {
        model: as_str("closure", "model", field("model")?)?.to_string(),
        release: as_str("closure", "release", field("release")?)?.to_string(),
        lane: as_str("closure", "lane", field("lane")?)?.to_string(),
        scope,
        manifest,
        objects,
        stamps,
        complete,
        presign_max_digests,
    })
}

/// THE WHOLE closure, however many pages it takes.
///
/// Page one resolves the ref; every page after it pins the manifest page one returned, so a
/// release re-pointed mid-walk cannot stitch two trees into one closure — the hub answers
/// about the identity that was asked for, not about whatever the lane names now. The pages
/// are additionally checked against each other, because a hub that paginates correctly and
/// a hub that has been restarted onto a different catalog look the same from one page.
#[allow(clippy::too_many_arguments)]
pub fn closure_all(
    client: &Client,
    base: &str,
    refspec: &str,
    lane: &str,
    credential: &dyn CredentialProvider,
    policy: &SourcePolicy,
    deadline: Deadline,
    ledger: &Ledger,
) -> Result<Closure> {
    let mut whole = closure(
        client, base, refspec, lane, "", credential, policy, deadline, ledger,
    )?;
    if whole.complete {
        return Ok(whole);
    }
    // The digest-pinned spelling of what page one resolved. `resolve` on the hub treats a
    // digest as identity and settles on it alone.
    let pinned = if whole.release.is_empty() {
        format!("{}@sha256:{}", whole.model, whole.manifest.sha256)
    } else {
        format!(
            "{}@{}@sha256:{}",
            whole.model, whole.release, whole.manifest.sha256
        )
    };
    // A closure may not exceed what this build will parse as one document's reference
    // total. The bound is the artifact's, not the protocol's: a page count would be a
    // number about the wire, and the wire is exactly what pagination made irrelevant.
    while !whole.complete {
        let Some(last) = whole.objects.last().map(|o| o.sha256.clone()) else {
            return refuse(
                Code::HUB_REFUSED,
                "the closure is incomplete and its page is empty — paging cannot advance",
            );
        };
        if whole.objects.len() >= limits::MAX_TOTAL_REFS {
            return refuse(
                Code::TOTAL_REFS_CAP,
                format!(
                    "the closure reached {} objects, this build's whole-document reference \
                     total, and is still not complete",
                    limits::MAX_TOTAL_REFS
                ),
            );
        }
        let next = closure(
            client, base, &pinned, lane, &last, credential, policy, deadline, ledger,
        )?;
        if next.manifest != whole.manifest || next.model != whole.model || next.scope != whole.scope
        {
            return refuse(
                Code::HUB_REFUSED,
                format!(
                    "the closure changed identity between pages: {} {} became {} {} — a \
                     paged answer describes ONE tree",
                    whole.model,
                    whole.manifest.id(),
                    next.model,
                    next.manifest.id()
                ),
            );
        }
        if next
            .objects
            .first()
            .is_some_and(|first| first.sha256 <= last)
        {
            return refuse(
                Code::HUB_REFUSED,
                format!(
                    "the closure page after {last} starts at {} — the rows are sorted unique \
                     by digest and a page must continue the one before it",
                    next.objects[0].sha256
                ),
            );
        }
        whole.complete = next.complete;
        whole.objects.extend(next.objects);
    }
    Ok(whole)
}

/// URLs for exactly the asked digests. A digest the hub does not answer is a typed
/// refusal here, not a KeyError three calls later.
#[allow(clippy::too_many_arguments)]
/// What one presign answered: the URLs, and the LIFETIME the hub declared for them.
///
/// The lifetime is read from the hub's OWN clock — `expires_at_unix - server_time_unix` —
/// never from this machine's, so a skewed local clock cannot make a live URL look dead or a
/// dead one look live. `None` when the hub declared neither field: there is then nothing to
/// judge a URL's age against, and the caller's mint window is the only bound on it.
#[derive(Debug, Clone, Default)]
pub struct Presigned {
    pub urls: Vec<(String, String)>,
    pub lifetime: Option<std::time::Duration>,
}

/// Mint URLs for exactly `digests` — sorted unique bare hex, which is what the hub accepts
/// and what `FetchPlan` already produces. The hub answers every digest it was asked or
/// refuses; a silent subset is a refusal here, because a walk that discovers a missing URL
/// halfway through has already spent the link.
pub fn presign(
    client: &Client,
    base: &str,
    digests: &[String],
    credential: &dyn CredentialProvider,
    policy: &SourcePolicy,
    deadline: Deadline,
    ledger: &Ledger,
) -> Result<Presigned> {
    let body = canon::write(&Value::obj(vec![(
        "digests",
        Value::arr(digests.iter().map(Value::str).collect()),
    )]));
    let value = call(
        client,
        base,
        "/v1/tensorfs/presign",
        &body,
        credential,
        policy,
        deadline,
        ledger,
    )?;
    let obj = as_obj("presign", "response", &value)?;
    let urls = obj
        .iter()
        .find(|(k, _)| k == "urls")
        .map(|(_, v)| v)
        .ok_or_else(|| Refusal {
            code: Code::HUB_REFUSED,
            detail: "the presign response carries no \"urls\" field".into(),
        })?;
    let rows = as_obj("presign", "urls", urls)?;
    let field = |name: &str| obj.iter().find(|(k, _)| k == name).map(|(_, v)| v);
    let lifetime = match (field("expires_at_unix"), field("server_time_unix")) {
        (Some(expires), Some(server_now)) => {
            let expires = as_uint("presign", "expires_at_unix", expires)?;
            let server_now = as_uint("presign", "server_time_unix", server_now)?;
            Some(std::time::Duration::from_secs(
                expires.saturating_sub(server_now),
            ))
        }
        _ => None,
    };
    let mut out = Vec::new();
    for digest in digests {
        match rows.iter().find(|(k, _)| k == digest) {
            Some((_, v)) => out.push((
                digest.clone(),
                as_str("presign.urls", digest, v)?.to_string(),
            )),
            None => {
                return refuse(
                    Code::HUB_REFUSED,
                    format!(
                        "sha256:{digest}: asked for and not answered — presign authorizes \
                         every digest it was asked or refuses, never a silent subset"
                    ),
                )
            }
        }
    }
    Ok(Presigned {
        urls: out,
        lifetime,
    })
}

fn object_row(what: &'static str, value: &Value) -> Result<ObjectRef> {
    let obj = as_obj(what, "row", value)?;
    let sha = obj
        .iter()
        .find(|(k, _)| k == "sha256")
        .map(|(_, v)| v)
        .ok_or_else(|| Refusal {
            code: Code::HUB_REFUSED,
            detail: format!("{what}: row carries no sha256"),
        })?;
    let length = obj
        .iter()
        .find(|(k, _)| k == "length")
        .map(|(_, v)| v)
        .ok_or_else(|| Refusal {
            code: Code::HUB_REFUSED,
            detail: format!("{what}: row carries no length"),
        })?;
    Ok(ObjectRef {
        sha256: hex64(what, as_str(what, "sha256", sha)?)?,
        length: as_uint(what, "length", length)?,
    })
}

/// The hub answers a mint mid-walk, so a hub blip must not discard a transfer that has
/// already moved gigabytes. `call` re-asks while re-asking is HONEST — proto-035's polarity:
/// an allow-list of statuses that could genuinely answer differently later, everything else
/// terminal on the first refusal.
///
/// What it waits is never a duration invented here. A 429 carries `retry_after_unix` beside
/// the hub's own `server_time_unix`, so the wait is the hub's stated instant read off the
/// hub's own clock; when nothing is stated the wait is the ledger's `patience()`, which is a
/// multiple of the gap this pull has ACTUALLY shown. Both are debounces over an observed
/// fact. The caller's deadline bounds either one.
#[allow(clippy::too_many_arguments)]
fn call(
    client: &Client,
    base: &str,
    path: &str,
    body: &[u8],
    credential: &dyn CredentialProvider,
    policy: &SourcePolicy,
    deadline: Deadline,
    ledger: &Ledger,
) -> Result<Value> {
    let mut last: Option<Refusal> = None;
    for attempt in 1..=crate::transport::ledger::FETCH_ATTEMPTS {
        let final_attempt = attempt == crate::transport::ledger::FETCH_ATTEMPTS;
        match call_once(
            client, base, path, body, credential, policy, deadline, ledger,
        ) {
            Ok(value) => return Ok(value),
            // NOT WEATHER, AND NOT ABOUT THE HUB (tfs-106). `TRANSFER_FAILED` out of an
            // exchange means the PULL has stopped moving while this call was outstanding.
            // Re-asking reaches the identical verdict on the next attempt's first tick —
            // nothing this loop does can restart a transfer — and the only thing three more
            // asks would achieve is replacing a refusal that names the response head with
            // one that names a request write. Answer with what was measured.
            Err(Retryable { refusal, .. }) if refusal.code == Code::TRANSFER_FAILED => {
                return Err(refusal)
            }
            Err(Retryable { wait, refusal }) if !final_attempt => {
                // A wait is spent out of the caller's budget, never past it.
                let wait = match deadline.remaining()? {
                    Some(left) => wait.min(left),
                    None => wait,
                };
                last = Some(refusal);
                std::thread::sleep(wait);
            }
            Err(Retryable { refusal, .. }) => return Err(refusal),
        }
    }
    Err(last.unwrap_or(Refusal {
        code: Code::HUB_UNREACHABLE,
        detail: format!("the hub at {base} did not answer {path}"),
    }))
}

/// One refusal plus how long to wait before re-asking. A zero wait means "do not re-ask".
struct Retryable {
    wait: std::time::Duration,
    refusal: Refusal,
}

impl From<Refusal> for Retryable {
    fn from(refusal: Refusal) -> Retryable {
        Retryable {
            wait: std::time::Duration::ZERO,
            refusal,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn call_once(
    client: &Client,
    base: &str,
    path: &str,
    body: &[u8],
    credential: &dyn CredentialProvider,
    policy: &SourcePolicy,
    deadline: Deadline,
    ledger: &Ledger,
) -> std::result::Result<Value, Retryable> {
    let url = format!("{}{}", base.trim_end_matches('/'), path);
    let checked = policy.check(&url).map_err(Retryable::from)?;
    let mut headers = vec![("content-type".to_string(), "application/json".to_string())];
    headers.extend(credential.headers(&checked.host));
    let response = crate::transport::http::request(
        client,
        &checked,
        "POST",
        &headers,
        Some(body),
        deadline,
        ledger,
        // THE CONTROL PLANE (tfs-106). Every wait in this exchange answers to the pull's own
        // progress. It is not a bound on how long a mint may take — a hub is entitled to be
        // slow, and a mint outstanding while sixteen streams keep landing objects is left
        // entirely alone. It is a bound on a call that is serving a transfer which has
        // itself stopped, which is the only shape `Leases::minting` can wedge in, and the
        // shape that cost `gayong` and `weed` fifteen billed minutes at zero bytes.
        crate::transport::http::Judge::Transfer,
        None,
        Code::HUB_UNREACHABLE,
    )
    .map_err(|refusal| match refusal.code {
        // A hub that did not answer at all is weather, not a verdict: the same request may
        // reach it on the next connection.
        Code::HUB_UNREACHABLE => Retryable {
            wait: ledger.patience(),
            refusal: Refusal {
                code: Code::HUB_UNREACHABLE,
                detail: format!(
                    "the hub at {base} did not answer {path} — {}; artifacts already in the \
                     local store are usable offline",
                    refusal.detail
                ),
            },
        },
        _ => Retryable::from(refusal),
    })?;
    let status = response.status;
    let raw = response
        .read_capped(limits::DOC_MAX_BYTES)
        .map_err(Retryable::from)?;
    if (200..300).contains(&status) {
        return canon::parse(&raw, limits::DOC_MAX_BYTES).map_err(|refusal| {
            Retryable::from(Refusal {
                code: Code::HUB_REFUSED,
                detail: format!("the hub's {path} answer is not readable I-JSON: {refusal}"),
            })
        });
    }
    let Some(refusal) = envelope(status, &raw, path) else {
        // NOT THE HUB. Every refusal the hub can write carries its typed envelope, so a
        // status that carries none was written by something in front of it — a tunnel
        // whose agent is reconnecting, a load balancer with no backend, a captive
        // gateway. That is weather over an immutable request, exactly like a connection
        // that was never answered, and it is classified as such.
        //
        // Reading it as a verdict cost a paid H100 on 2026-09-08 (tfs-101): the public
        // origin's tunnel agent was seventeen seconds out of a reconnect loop, its edge
        // answered `POST /v1/tensorfs/presign` with its own 404 page, this function
        // called that REF_NOT_FOUND, the pod supervisor's poisoned set made it terminal,
        // and the pod journaled a permanent refusal for a placement set whose every
        // object the hub was holding and would have signed.
        return Err(Retryable {
            wait: ledger.patience(),
            refusal: Refusal {
                code: Code::HUB_UNREACHABLE,
                detail: format!(
                    "{base}{path} answered HTTP {status} with no hub envelope, so the hub                      did not write it; artifacts already in the local store are usable offline"
                ),
            },
        });
    };
    if !crate::transport::pull::RETRYABLE_STATUS.contains(&status) {
        return Err(Retryable::from(refusal));
    }
    Err(Retryable {
        wait: stated_wait(&raw).unwrap_or_else(|| ledger.patience()),
        refusal,
    })
}

/// How long the HUB says to wait, read off the pair of clocks it sent so a skewed local
/// clock cannot shorten or lengthen it. `None` when the hub stated nothing.
fn stated_wait(raw: &[u8]) -> Option<std::time::Duration> {
    let Ok(Value::Obj(fields)) = canon::parse(raw, 64 * 1024) else {
        return None;
    };
    let uint = |name: &str| {
        fields
            .iter()
            .find(|(k, _)| k == name)
            .and_then(|(_, v)| match v {
                Value::Int(n) if *n >= 0 => Some(*n as u64),
                _ => None,
            })
    };
    let (retry_after, server_now) = (uint("retry_after_unix")?, uint("server_time_unix")?);
    Some(std::time::Duration::from_secs(
        retry_after.saturating_sub(server_now),
    ))
}

/// The hub's own typed envelope, crossed verbatim: its code and remedy are BETTER than
/// anything invented here — the hub knows why it refused.
///
/// `None` when the answer carried no envelope, and that is the point of the return type.
/// The STATUS ALONE is not evidence about the catalog: a 404 means "the published catalog
/// does not name this" only when the hub said so, and the presence of `error.code` is the
/// only thing that distinguishes the hub's verdict from a proxy's opinion. Mapping status
/// without that test is how a tunnel's error page came to mean a model does not exist —
/// permanently, on a pod that had already paid to boot.
fn envelope(status: u16, raw: &[u8], path: &str) -> Option<Refusal> {
    let mut code = String::new();
    let mut message = String::new();
    let mut remedy = String::new();
    if let Ok(Value::Obj(fields)) = canon::parse(raw, 64 * 1024) {
        if let Some((_, Value::Obj(error))) = fields.into_iter().find(|(k, _)| k == "error") {
            for (key, val) in error {
                if let Value::Str(text) = val {
                    match key.as_str() {
                        "code" => code = text,
                        "message" => message = text,
                        "remedy" => remedy = text,
                        _ => {}
                    }
                }
            }
        }
    }
    if code.is_empty() {
        return None;
    }
    if message.is_empty() {
        message = format!("the hub answered HTTP {status} to {path}");
    }
    let detail = if remedy.is_empty() {
        format!("{code}: {message}")
    } else {
        format!("{code}: {message} — {remedy}")
    };
    Some(match status {
        404 => Refusal {
            code: Code::REF_NOT_FOUND,
            detail,
        },
        401 | 403 => Refusal {
            code: Code::CREDENTIAL_REQUIRED,
            detail: if remedy.is_empty() {
                format!("{code}: {message} — supply the credential and re-run; never a prompt")
            } else {
                detail
            },
        },
        _ => Refusal {
            code: Code::HUB_REFUSED,
            detail,
        },
    })
}

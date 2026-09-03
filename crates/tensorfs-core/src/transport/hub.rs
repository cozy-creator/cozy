//! The hub's two calls (tfs-048 protocol, th-129's server half): `closure` and `presign`.
//!
//! One call used to do everything — a hub-computed plan minting a URL for every file
//! whether held or not, which is why the pod had to pre-emptively describe its own disk.
//! Two calls replace it, and the store stops being the hub's business:
//!
//! 1. `POST {base}/v1/tensorfs/closure`, body `{"lane":"...","ref":"org/name[@release]"}`
//!    → `{"lane":..,"manifest":{"length":N,"sha256":hex},"model":"org/name",
//!       "objects":[{"length":N,"sha256":hex},...],"release":..,"scope":"runtime",
//!       "stamps":{key:[values]}}` — permission-checked, NO URLs. `objects` is sorted
//!    unique by digest and EXCLUDES the manifest row, which travels in `manifest`.
//!    `scope` names the walk the closure covers (`runtime` today); `stamps` is the
//!    resolved lane's stamped-fact set (#567d) and may be absent.
//! 2. `POST {base}/v1/tensorfs/presign`, body `{"digests":[hex,...]}` (sorted unique, at
//!    most `MAX_PRESIGN_DIGESTS`) → `{"expires_at_unix":N,"server_time_unix":N,
//!    "urls":{hex:"https://...",...}}` — credential again, URLs only for what was asked,
//!    the manifest digest included when asked. The two clock fields are the URL's declared
//!    life stated in the HUB's own time, which is what lets a caller judge a URL's age
//!    without trusting its own clock; the pull mints against them per object rather than
//!    once for a whole closure.
//!
//! Errors travel in the hub's own envelope `{"error":{"code","message","remedy"}}` and
//! cross verbatim — the hub knows why it refused better than anything restated here.
//! Responses are I-JSON, ASCII, no HTML escaping (Go: `SetEscapeHTML(false)`).
//!
//! **The credential arrives as a provider trait, never a credential type**, so one
//! implementation serves a worker delegation on a pod and a user token on a laptop. It is
//! asked per host, so a credential stays scoped to the hosts its provider names and
//! cannot leak onto a presigned object host or a redirect target.

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
/// builds when its credential is not a bearer (a worker delegation is two headers).
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
/// - `bearer <token>`                       → `authorization: Bearer <token>`
/// - `delegation <payload-b64url> <sig-b64url>` → the worker delegation pair
/// - empty                                  → anonymous
pub fn credential_from_spec(spec: &str, hosts: Vec<String>) -> Result<ScopedHeaders> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Ok(ScopedHeaders {
            hosts,
            headers: Vec::new(),
        });
    }
    let mut parts = spec.split_whitespace();
    let headers = match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some("bearer"), Some(token), None, _) => {
            vec![("authorization".to_string(), format!("Bearer {token}"))]
        }
        (Some("delegation"), Some(payload), Some(signature), None) => vec![
            ("x-tensorfs-delegation".to_string(), payload.to_string()),
            (
                "x-tensorfs-delegation-signature".to_string(),
                signature.to_string(),
            ),
        ],
        _ => {
            return refuse(
                Code::CREDENTIAL_REQUIRED,
                "a credential is `bearer <token>` or `delegation <payload> <signature>` — \
                 one scoped value, passed explicitly",
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
}

#[allow(clippy::too_many_arguments)]
pub fn closure(
    client: &Client,
    base: &str,
    refspec: &str,
    lane: &str,
    credential: &dyn CredentialProvider,
    policy: &SourcePolicy,
    deadline: Deadline,
    ledger: &Ledger,
) -> Result<Closure> {
    let body = canon::write(&Value::obj(vec![
        ("lane", Value::str(lane)),
        ("ref", Value::str(refspec)),
    ]));
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
    Ok(Closure {
        model: as_str("closure", "model", field("model")?)?.to_string(),
        release: as_str("closure", "release", field("release")?)?.to_string(),
        lane: as_str("closure", "lane", field("lane")?)?.to_string(),
        scope,
        manifest,
        objects,
        stamps,
    })
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
    let url = format!("{}{}", base.trim_end_matches('/'), path);
    let checked = policy.check(&url)?;
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
        false,
        None,
        Code::HUB_UNREACHABLE,
    )
    .map_err(|refusal| match refusal.code {
        Code::HUB_UNREACHABLE => Refusal {
            code: Code::HUB_UNREACHABLE,
            detail: format!(
                "the hub at {base} did not answer {path} — {}; artifacts already in the \
                 local store are usable offline",
                refusal.detail
            ),
        },
        _ => refusal,
    })?;
    let status = response.status;
    let raw = response.read_capped(limits::DOC_MAX_BYTES)?;
    if (200..300).contains(&status) {
        return canon::parse(&raw, limits::DOC_MAX_BYTES).map_err(|refusal| Refusal {
            code: Code::HUB_REFUSED,
            detail: format!("the hub's {path} answer is not readable I-JSON: {refusal}"),
        });
    }
    Err(envelope(status, &raw, path))
}

/// The hub's own typed envelope, crossed verbatim: its code and remedy are BETTER than
/// anything invented here — the hub knows why it refused.
fn envelope(status: u16, raw: &[u8], path: &str) -> Refusal {
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
        code = format!("http.{status}");
    }
    if message.is_empty() {
        message = format!("the hub answered HTTP {status} to {path}");
    }
    let detail = if remedy.is_empty() {
        format!("{code}: {message}")
    } else {
        format!("{code}: {message} — {remedy}")
    };
    match status {
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
    }
}

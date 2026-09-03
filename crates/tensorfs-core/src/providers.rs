//! Provider handlers (tfs-052): HuggingFace and Civitai live IN TensorFS.
//!
//! Provider knowledge — address grammar, API shapes, immutable-URL rules, origin
//! allowlists — belongs with the provider handler, not with whichever daemon happens to
//! hold a socket today. A handler LISTS what a pinned source reaches (member paths with
//! whatever identity the provider declares), then PULLS it: the fetch plane splits the
//! declared set against the real store, the one transport moves only `wanted`, and every
//! byte enters through `Store::put_stream`. A foreign origin is not a second admission
//! path.
//!
//! Credentials arrive through `transport::CredentialProvider` — never the environment —
//! and are asked per host on every redirect hop, so an owner's key cannot leak onto a
//! CDN host. Where a handler RUNS is deployment: ingest belongs where the credential
//! already lives (the owner's box or the hub); a pod fetching a foreign origin directly
//! takes a short-lived scoped token through the same trait.

use crate::err::{refuse, Code, Refusal, Result};
use crate::fetch::{DeliveryGrant, FetchPlan};
use crate::ids::ObjectRef;
use crate::jcs::{self, Json};
use crate::store::{Fault, Store};
use crate::transport::{
    api_get, fetch_url, probe_length, CredentialProvider, Deadline, Ledger, SourcePolicy,
};

/// Listing documents are bounded like any other untrusted input.
const API_DOC_MAX_BYTES: u64 = 8 << 20;
const API_PAGE_MAX: usize = 64;
/// A member the provider does not identify by digest is fetched whole and hashed; the
/// class is HuggingFace non-LFS files (configs, tokenizers), which git itself keeps small.
const UNPINNED_MEMBER_MAX_BYTES: u64 = 64 << 20;
const MEMBER_MAX: usize = 4096;

/// One pinned foreign source, in the URI grammar the rest of cozy already speaks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceUri {
    /// `hf://<org>/<repo>@<40-hex commit>` — a repository pinned to one revision.
    HuggingFace {
        org: String,
        repo: String,
        revision: String,
    },
    /// `civitai://<model-version-id>` — one published model version.
    Civitai { version: u64 },
}

fn hf_name(value: &str) -> bool {
    let bytes = value.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= 128
        && bytes[0].is_ascii_alphanumeric()
        && bytes
            .iter()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

fn member_path(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.is_empty() || bytes.len() > 1024 || value.starts_with(['.', '/']) {
        return false;
    }
    if !bytes
        .iter()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-' | b'/' | b'+' | b' '))
    {
        return false;
    }
    value
        .split('/')
        .all(|segment| !segment.is_empty() && segment != "." && segment != "..")
}

impl SourceUri {
    pub fn parse(value: &str) -> Result<SourceUri> {
        if let Some(rest) = value.strip_prefix("hf://") {
            let (org, tail) = rest.split_once('/').ok_or(Refusal {
                code: Code::SOURCE_NOT_ALLOWED,
                detail: "hf source is hf://<org>/<repo>@<40-hex revision>".into(),
            })?;
            let (repo, revision) = tail.split_once('@').ok_or(Refusal {
                code: Code::SOURCE_NOT_ALLOWED,
                detail: "hf source must pin a 40-hex revision with @".into(),
            })?;
            if !hf_name(org)
                || !hf_name(repo)
                || revision.len() != 40
                || !revision
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            {
                return refuse(
                    Code::SOURCE_NOT_ALLOWED,
                    "hf source names or revision are outside the grammar",
                );
            }
            return Ok(SourceUri::HuggingFace {
                org: org.to_string(),
                repo: repo.to_string(),
                revision: revision.to_string(),
            });
        }
        if let Some(rest) = value.strip_prefix("civitai://") {
            let version: u64 = rest.parse().map_err(|_| Refusal {
                code: Code::SOURCE_NOT_ALLOWED,
                detail: "civitai source is civitai://<model-version-id>".into(),
            })?;
            if version == 0 || rest.starts_with('0') {
                return refuse(
                    Code::SOURCE_NOT_ALLOWED,
                    "civitai model version id is invalid",
                );
            }
            return Ok(SourceUri::Civitai { version });
        }
        refuse(
            Code::SOURCE_NOT_ALLOWED,
            "a foreign source is hf://… or civitai://…",
        )
    }
}

/// Provider origins. The defaults are the real services; a test points both at its own
/// loopback listener and sets `allow_local`.
#[derive(Debug, Clone)]
pub struct Endpoints {
    pub huggingface: String,
    pub civitai: String,
    pub allow_local: bool,
}

impl Default for Endpoints {
    fn default() -> Self {
        Endpoints {
            huggingface: "https://huggingface.co".into(),
            civitai: "https://civitai.com".into(),
            allow_local: false,
        }
    }
}

impl Endpoints {
    /// The redirect fence for one source: the provider's API host and its known delivery
    /// origins (a leading dot admits subdomains — the policy's own grammar), enforced by
    /// the transport on the first URL and on every hop. An overridden base (a loopback
    /// test origin) joins the allowlist explicitly; `allow_local` only unlocks its
    /// address class, never the host check.
    pub fn policy(&self, uri: &SourceUri) -> SourcePolicy {
        let mut allowed_hosts = match uri {
            SourceUri::HuggingFace { .. } => vec![
                "huggingface.co".into(),
                ".huggingface.co".into(),
                "hf.co".into(),
                ".hf.co".into(),
            ],
            SourceUri::Civitai { .. } => vec![
                "civitai.com".into(),
                ".civitai.com".into(),
                "civitai-delivery-worker-prod.5ac0637cfd0766c97916cefa3764fbdf.r2.cloudflarestorage.com"
                    .into(),
            ],
        };
        if let Ok(host) = crate::transport::base_host(self.base(uri)) {
            if !allowed_hosts.contains(&host) {
                allowed_hosts.push(host);
            }
        }
        SourcePolicy {
            allowed_hosts,
            allow_local: self.allow_local,
            max_redirects: 5,
        }
    }

    fn base(&self, uri: &SourceUri) -> &str {
        match uri {
            SourceUri::HuggingFace { .. } => self.huggingface.trim_end_matches('/'),
            SourceUri::Civitai { .. } => self.civitai.trim_end_matches('/'),
        }
    }
}

/// One member of a pinned source, as the provider declares it. `sha256`/`length` are the
/// provider's claims — the admission door judges the bytes either way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceMember {
    pub member: String,
    pub length: Option<u64>,
    pub sha256: Option<String>,
    pub url: String,
}

fn json_get<'j>(value: &'j Json, key: &str) -> Option<&'j Json> {
    match value {
        Json::Obj(pairs) => pairs
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value),
        _ => None,
    }
}

fn json_str<'j>(value: &'j Json, key: &str) -> Option<&'j str> {
    match json_get(value, key) {
        Some(Json::Str(s)) => Some(s),
        _ => None,
    }
}

fn json_u64(value: &Json, key: &str) -> Option<u64> {
    match json_get(value, key) {
        Some(Json::Num(n)) if n.fract() == 0.0 && *n >= 0.0 => Some(*n as u64),
        _ => None,
    }
}

fn digest64(value: &str) -> Option<String> {
    let lower = value.to_ascii_lowercase();
    (lower.len() == 64 && lower.bytes().all(|b| b.is_ascii_hexdigit())).then_some(lower)
}

/// List what one pinned source reaches: every member with the identity the provider
/// declares. Bounded, policy-fenced, credentialed through the trait.
pub fn list(
    uri: &SourceUri,
    endpoints: &Endpoints,
    credentials: &dyn CredentialProvider,
    deadline: Deadline,
) -> Result<Vec<SourceMember>> {
    let ledger = Ledger::new();
    list_observed(uri, endpoints, credentials, deadline, &ledger)
}

fn list_observed(
    uri: &SourceUri,
    endpoints: &Endpoints,
    credentials: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &Ledger,
) -> Result<Vec<SourceMember>> {
    let policy = endpoints.policy(uri);
    let base = endpoints.base(uri);
    let mut members = match uri {
        SourceUri::HuggingFace {
            org,
            repo,
            revision,
        } => {
            let mut page = format!("{base}/api/models/{org}/{repo}/tree/{revision}?recursive=true");
            let mut members = Vec::new();
            let mut listed_everything = false;
            for _ in 0..API_PAGE_MAX {
                let (body, next) = api_get(
                    &page,
                    &policy,
                    credentials,
                    API_DOC_MAX_BYTES,
                    deadline,
                    ledger,
                )?;
                let rows = match jcs::parse(&body, API_DOC_MAX_BYTES as usize)? {
                    Json::Arr(rows) => rows,
                    _ => {
                        return refuse(
                            Code::MALFORMED_JSON,
                            "the HuggingFace tree endpoint did not answer an array",
                        )
                    }
                };
                for row in &rows {
                    if json_str(row, "type") != Some("file") {
                        continue;
                    }
                    let Some(path) = json_str(row, "path") else {
                        return refuse(Code::MISSING_FIELD, "a tree row names no path");
                    };
                    if !member_path(path) {
                        return refuse(
                            Code::KEY_GRAMMAR,
                            format!("member path {path:?} is outside the grammar"),
                        );
                    }
                    // LFS rows carry an exact sha256 and length; plain git rows carry a
                    // git oid, which is not an object identity here.
                    let (sha256, length) = match json_get(row, "lfs") {
                        Some(lfs) => (
                            json_str(lfs, "oid").and_then(digest64),
                            json_u64(lfs, "size").or_else(|| json_u64(row, "size")),
                        ),
                        None => (None, json_u64(row, "size")),
                    };
                    members.push(SourceMember {
                        member: path.to_string(),
                        length,
                        sha256,
                        url: format!("{base}/{org}/{repo}/resolve/{revision}/{path}"),
                    });
                }
                match next {
                    Some(next_page) => page = next_page,
                    None => {
                        listed_everything = true;
                        break;
                    }
                }
            }
            if !listed_everything {
                return refuse(
                    Code::COUNT_CAP,
                    format!("the tree listing paginated past {API_PAGE_MAX} pages"),
                );
            }
            members
        }
        SourceUri::Civitai { version } => {
            let (body, _) = api_get(
                &format!("{base}/api/v1/model-versions/{version}"),
                &policy,
                credentials,
                API_DOC_MAX_BYTES,
                deadline,
                ledger,
            )?;
            let document = jcs::parse(&body, API_DOC_MAX_BYTES as usize)?;
            let Some(Json::Arr(files)) = json_get(&document, "files") else {
                return refuse(
                    Code::MISSING_FIELD,
                    "the Civitai model-version document lists no files",
                );
            };
            let mut members = Vec::new();
            for file in files {
                let Some(name) = json_str(file, "name") else {
                    return refuse(Code::MISSING_FIELD, "a Civitai file names no name");
                };
                if !member_path(name) {
                    return refuse(
                        Code::KEY_GRAMMAR,
                        format!("member name {name:?} is outside the grammar"),
                    );
                }
                let sha256 = json_get(file, "hashes")
                    .and_then(|hashes| json_str(hashes, "SHA256"))
                    .and_then(digest64);
                let url = match json_str(file, "downloadUrl") {
                    Some(explicit) => explicit.to_string(),
                    None => format!("{base}/api/download/models/{version}"),
                };
                // sizeKB is kilobytes as a float — an approximation, not a length.
                // The exact length is probed from the origin at pull time.
                members.push(SourceMember {
                    member: name.to_string(),
                    length: None,
                    sha256,
                    url,
                });
            }
            members
        }
    };
    if members.is_empty() {
        return refuse(Code::MISSING_FIELD, "the source reaches no member");
    }
    if members.len() > MEMBER_MAX {
        return refuse(
            Code::COUNT_CAP,
            format!("the source lists more than {MEMBER_MAX} members"),
        );
    }
    members.sort_by(|left, right| left.member.cmp(&right.member));
    if members
        .windows(2)
        .any(|pair| pair[0].member == pair[1].member)
    {
        return refuse(Code::DUPLICATE_KEY, "the source lists one member twice");
    }
    Ok(members)
}

/// What pulling one member produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pulled {
    pub member: String,
    pub object: ObjectRef,
    pub transferred: u64,
    pub held: bool,
}

/// Fetch and ingest one pinned foreign source: list, resolve exact identities, split the
/// declared set against the real store, move only `wanted` through the one transport, and
/// hash-then-admit the members the provider does not identify. The warm half of th-124:
/// a store that already holds the weights answers with zero transferred bytes.
pub fn pull(
    store: &Store,
    uri: &SourceUri,
    endpoints: &Endpoints,
    credentials: &dyn CredentialProvider,
    deadline: Deadline,
    progress: &mut dyn FnMut(&str, u64),
) -> Result<Vec<Pulled>> {
    let policy = endpoints.policy(uri);
    // One liveness ledger for the whole pull: the stall judgment learns this link's own
    // pace across listings, probes and members alike.
    let ledger = Ledger::new();
    let members = list_observed(uri, endpoints, credentials, deadline, &ledger)?;

    // Exact identities first: a Civitai listing declares a digest but only approximate
    // kilobytes, so the exact length comes from a one-byte ranged probe.
    let mut pinned: Vec<(SourceMember, ObjectRef)> = Vec::new();
    let mut unpinned: Vec<SourceMember> = Vec::new();
    for member in members {
        match (&member.sha256, member.length) {
            (Some(sha256), Some(length)) if length > 0 => {
                let object = ObjectRef {
                    sha256: sha256.clone(),
                    length,
                };
                pinned.push((member, object));
            }
            (Some(sha256), _) => {
                let length = probe_length(&member.url, &policy, credentials, deadline, &ledger)?;
                if length == 0 {
                    return refuse(
                        Code::LENGTH_MISMATCH,
                        format!("{}: origin reports a zero-length member", member.member),
                    );
                }
                let object = ObjectRef {
                    sha256: sha256.clone(),
                    length,
                };
                pinned.push((member, object));
            }
            (None, _) => unpinned.push(member),
        }
    }

    let mut declared: Vec<ObjectRef> = pinned.iter().map(|(_, object)| object.clone()).collect();
    declared.sort_by(|left, right| left.sha256.cmp(&right.sha256));
    declared.dedup_by(|left, right| left.sha256 == right.sha256);
    let mut pulled = Vec::new();
    if !declared.is_empty() {
        let (plan, _) = FetchPlan::of_objects(store, "source-pull", &declared)?;
        for (member, object) in &pinned {
            if plan.wants(&object.sha256) {
                let grant = DeliveryGrant::mint(&plan, object)?;
                let fetched = fetch_url(
                    store,
                    &grant,
                    &member.url,
                    &policy,
                    credentials,
                    deadline,
                    &ledger,
                    Some(&mut |current| progress(&member.member, current)),
                )?;
                pulled.push(Pulled {
                    member: member.member.clone(),
                    object: object.clone(),
                    transferred: fetched.transferred,
                    held: false,
                });
            } else {
                pulled.push(Pulled {
                    member: member.member.clone(),
                    object: object.clone(),
                    transferred: 0,
                    held: true,
                });
            }
        }
        plan.complete(store)?;
    }

    // Members without a provider digest (HuggingFace non-LFS: configs, tokenizers) are
    // fetched whole, hashed, and admitted at the identity the bytes computed. They cannot
    // be planned HELD without an identity, so they re-fetch — bounded and small by git's
    // own rules.
    for member in &unpinned {
        let cap = member
            .length
            .map_or(UNPINNED_MEMBER_MAX_BYTES, |declared_length| declared_length);
        if cap > UNPINNED_MEMBER_MAX_BYTES {
            return refuse(
                Code::SIZE_CAP,
                format!(
                    "{}: an unidentified member may not exceed {UNPINNED_MEMBER_MAX_BYTES} bytes",
                    member.member
                ),
            );
        }
        let (body, _) = api_get(&member.url, &policy, credentials, cap, deadline, &ledger)?;
        if member
            .length
            .is_some_and(|declared_length| declared_length != body.len() as u64)
        {
            return refuse(
                Code::LENGTH_MISMATCH,
                format!("{}: body length disagrees with the listing", member.member),
            );
        }
        let object = ObjectRef::of(&body);
        store.put_stream(&mut body.as_slice(), Some(&object), &Fault::default())?;
        progress(&member.member, object.length);
        pulled.push(Pulled {
            member: member.member.clone(),
            object,
            transferred: body.len() as u64,
            held: false,
        });
    }
    pulled.sort_by(|left, right| left.member.cmp(&right.member));
    Ok(pulled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::{Anonymous, Deadline, HostToken};
    use std::io::{BufRead, Write as IoWrite};
    use std::net::TcpListener;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    fn temporary(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tensorfs-providers-{name}-{}-{}",
            std::process::id(),
            crate::meta::now_nanos_unique()
        ))
    }

    struct Request {
        path: String,
        range: Option<String>,
    }

    type RequestLog = Arc<Mutex<Vec<(String, Option<String>)>>>;

    /// A loopback provider origin: `connections` requests routed through one handler.
    fn origin(
        connections: usize,
        handler: impl Fn(&Request) -> Vec<u8> + Send + 'static,
    ) -> (String, RequestLog, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        let handle = std::thread::spawn(move || {
            for _ in 0..connections {
                let (mut socket, _) = listener.accept().unwrap();
                let mut reader = std::io::BufReader::new(socket.try_clone().unwrap());
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let path = line.split_whitespace().nth(1).unwrap_or("/").to_string();
                let mut authorization = None;
                let mut range = None;
                loop {
                    let mut header = String::new();
                    reader.read_line(&mut header).unwrap();
                    let header = header.trim_end().to_string();
                    if header.is_empty() {
                        break;
                    }
                    let lower = header.to_ascii_lowercase();
                    if let Some(value) = lower.strip_prefix("authorization: ") {
                        authorization = Some(value.to_string());
                    }
                    if let Some(value) = lower.strip_prefix("range: ") {
                        range = Some(value.to_string());
                    }
                }
                log.lock().unwrap().push((path.clone(), authorization));
                let response = handler(&Request { path, range });
                socket.write_all(&response).unwrap();
                let _ = socket.flush();
            }
        });
        (base, seen, handle)
    }

    fn ok_body(body: &[u8], content_type: &str) -> Vec<u8> {
        let mut out = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: {content_type}\r\nETag: \"tag\"\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        out.extend_from_slice(body);
        out
    }

    #[test]
    fn source_uri_grammar_holds() {
        assert_eq!(
            SourceUri::parse(&format!("hf://Org-1/repo.x@{}", "ab".repeat(20))).unwrap(),
            SourceUri::HuggingFace {
                org: "Org-1".into(),
                repo: "repo.x".into(),
                revision: "ab".repeat(20),
            }
        );
        assert_eq!(
            SourceUri::parse("civitai://12345").unwrap(),
            SourceUri::Civitai { version: 12345 }
        );
        for bad in [
            "hf://org-only",
            "hf://org/repo",                               // unpinned
            &format!("hf://org/repo@{}", "AB".repeat(20)), // uppercase revision
            "hf://org/repo@abc",
            "civitai://0",
            "civitai://01",
            "civitai://x",
            "s3://bucket/key",
        ] {
            assert_eq!(
                SourceUri::parse(bad).unwrap_err().code,
                Code::SOURCE_NOT_ALLOWED,
                "{bad}"
            );
        }
    }

    #[test]
    fn huggingface_lists_pulls_and_then_holds() {
        let root = temporary("hf");
        let store = Store::init(&root).unwrap();
        let weights: Vec<u8> = (0u32..80_000).flat_map(|i| i.to_le_bytes()).collect();
        let weights_ref = ObjectRef::of(&weights);
        let config = b"{\"model_type\":\"test\"}".to_vec();
        let revision = "ab".repeat(20);
        let tree = format!(
            r#"[
  {{"type":"file","path":"model.safetensors","size":{},"oid":"deadbeef","lfs":{{"oid":"{}","size":{}}}}},
  {{"type":"file","path":"config.json","size":{},"oid":"deadbeef"}},
  {{"type":"directory","path":"scheduler"}}
]"#,
            weights_ref.length,
            weights_ref.sha256,
            weights_ref.length,
            config.len()
        );
        let weights_body = weights.clone();
        let config_body = config.clone();
        let revision_for_routes = revision.clone();
        // list: tree; pull #1: tree + weights + config; pull #2: tree + config (HELD weights).
        let (base, seen, handle) = origin(6, move |request| {
            let r = &revision_for_routes;
            if request.path == format!("/api/models/org/repo/tree/{r}?recursive=true") {
                ok_body(tree.as_bytes(), "application/json")
            } else if request.path == format!("/org/repo/resolve/{r}/model.safetensors") {
                ok_body(&weights_body, "application/octet-stream")
            } else if request.path == format!("/org/repo/resolve/{r}/config.json") {
                ok_body(&config_body, "application/json")
            } else {
                panic!("unrouted path {}", request.path)
            }
        });
        let uri = SourceUri::parse(&format!("hf://org/repo@{revision}")).unwrap();
        let endpoints = Endpoints {
            huggingface: base.clone(),
            allow_local: true,
            ..Endpoints::default()
        };
        let credentials = HostToken {
            hosts: vec!["127.0.0.1".into()],
            token: "hf_secret".into(),
        };

        let listed = list(&uri, &endpoints, &credentials, Deadline::none()).unwrap();
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].member, "config.json");
        assert_eq!(listed[0].sha256, None);
        assert_eq!(
            listed[1].sha256.as_deref(),
            Some(weights_ref.sha256.as_str())
        );

        let first = pull(
            &store,
            &uri,
            &endpoints,
            &credentials,
            Deadline::none(),
            &mut |_, _| {},
        )
        .unwrap();
        assert_eq!(first.len(), 2);
        let weights_row = first
            .iter()
            .find(|row| row.member == "model.safetensors")
            .unwrap();
        assert_eq!(weights_row.object, weights_ref);
        assert_eq!(weights_row.transferred, weights_ref.length);
        assert!(!weights_row.held);
        let config_row = first
            .iter()
            .find(|row| row.member == "config.json")
            .unwrap();
        assert_eq!(config_row.object, ObjectRef::of(&config));
        assert!(store.record_valid(&weights_ref.sha256).is_ok());

        // Warm: the weights are HELD and move nothing; the digestless config re-fetches.
        let second = pull(
            &store,
            &uri,
            &endpoints,
            &credentials,
            Deadline::none(),
            &mut |_, _| {},
        )
        .unwrap();
        let weights_row = second
            .iter()
            .find(|row| row.member == "model.safetensors")
            .unwrap();
        assert!(weights_row.held);
        assert_eq!(weights_row.transferred, 0);

        // The bearer reached every loopback request, because the token is scoped to the
        // host that was actually asked.
        let log = seen.lock().unwrap();
        assert_eq!(log.len(), 6);
        assert!(log
            .iter()
            .all(|(_, authorization)| authorization.as_deref() == Some("bearer hf_secret")));
        drop(log);
        handle.join().unwrap();
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn civitai_probes_exact_length_and_holds_when_warm() {
        let root = temporary("civitai");
        let store = Store::init(&root).unwrap();
        let body: Vec<u8> = (0u32..70_000).flat_map(|i| i.to_le_bytes()).collect();
        let object = ObjectRef::of(&body);
        // No downloadUrl in the fixture: the handler must fall back to the provider's
        // canonical download route under the configured base.
        let listing = format!(
            r#"{{"id":777,"files":[{{"name":"model.safetensors","sizeKB":273.4,"hashes":{{"SHA256":"{}"}}}}]}}"#,
            object.sha256.to_ascii_uppercase()
        );
        let payload = body.clone();
        let length = object.length;
        // pull #1: listing + probe + body; pull #2: listing + probe (object HELD).
        let (base, _, handle) = origin(5, move |request| {
            if request.path == "/api/v1/model-versions/777" {
                ok_body(listing.as_bytes(), "application/json")
            } else if request.path == "/api/download/models/777" {
                if request.range.as_deref() == Some("bytes=0-0") {
                    format!(
                        "HTTP/1.1 206 Partial Content\r\nContent-Length: 1\r\nContent-Range: bytes 0-0/{length}\r\nETag: \"tag\"\r\nConnection: close\r\n\r\n\0"
                    )
                    .into_bytes()
                } else {
                    ok_body(&payload, "application/octet-stream")
                }
            } else {
                panic!("unrouted path {}", request.path)
            }
        });
        let uri = SourceUri::parse("civitai://777").unwrap();
        let endpoints = Endpoints {
            civitai: base.clone(),
            allow_local: true,
            ..Endpoints::default()
        };
        // pull #1: listing + one-byte length probe + full body. The listing carried a
        // digest and kilobytes; the probe turned that into an exact ObjectRef.
        let pulled = pull(
            &store,
            &uri,
            &endpoints,
            &Anonymous,
            Deadline::none(),
            &mut |_, _| {},
        )
        .unwrap();
        assert_eq!(pulled[0].object, object);
        assert_eq!(pulled[0].transferred, object.length);
        assert!(store.record_valid(&object.sha256).is_ok());
        // Warm: listing + probe again, zero bytes moved.
        let warm = pull(
            &store,
            &uri,
            &endpoints,
            &Anonymous,
            Deadline::none(),
            &mut |_, _| {},
        )
        .unwrap();
        assert!(warm[0].held);
        assert_eq!(warm[0].transferred, 0);
        handle.join().unwrap();
        let _ = std::fs::remove_dir_all(root);
    }
}

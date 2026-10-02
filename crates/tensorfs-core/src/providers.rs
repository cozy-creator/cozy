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
    api_get, fetch_prefix, fetch_ranged, probe_length, CredentialProvider, Deadline, Ledger,
    Ranged, SourcePolicy,
};

/// Listing documents are bounded like any other untrusted input.
const API_DOC_MAX_BYTES: u64 = 8 << 20;
const API_PAGE_MAX: usize = 64;
/// A member the provider does not identify by digest is fetched whole and hashed; the
/// class is HuggingFace non-LFS files (configs, tokenizers), which git itself keeps small.
const UNPINNED_MEMBER_MAX_BYTES: u64 = 64 << 20;
/// An explicit ordinary-file selection is a bounded working set, not a weight download.
const FILE_SELECTION_MAX_BYTES: u64 = 64 << 20;
const MEMBER_MAX: usize = 4096;
/// A `*.safetensors.index.json` is a map from tensor name to shard file. It is JSON that
/// git itself keeps small, and the Go resolver bounded it at the same 32 MiB.
const INDEX_MAX_BYTES: u64 = 32 << 20;
/// H3's largest index names 535 tensors. The bound is Go's, three orders above the fact.
const WEIGHT_MAP_MAX: usize = 200_000;

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
            ..Default::default()
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
    /// The provider's own designation of "this is the file to take", already narrowed to a
    /// SafeTensors carrier — Civitai's `primary` flag together with `metadata.format`.
    /// HuggingFace declares no such thing (a repository is a tree, and the selection rule
    /// is the file suffix), so its rows leave this false.
    pub primary: bool,
    /// Another SafeTensors file of the same Civitai version with a declared digest (a text
    /// encoder, a VAE): a carrier only a reviewed profile composes the model from.
    pub companion: bool,
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
    // A provider may spell its digest with the algorithm on the front; the Go resolver
    // strips exactly this prefix and so does this one.
    let lower = value.to_ascii_lowercase();
    let lower = lower.strip_prefix("sha256:").unwrap_or(&lower);
    (lower.len() == 64 && lower.bytes().all(|b| b.is_ascii_hexdigit())).then(|| lower.to_string())
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
    list_selected_observed(uri, endpoints, credentials, deadline, ledger, None, None)
}

/// `unnameable`, when given, collects tensor carriers whose paths are outside the member
/// grammar instead of refusing the listing: the caller names its carriers exactly, so a
/// file no selection can name cannot shrink it.
#[allow(clippy::too_many_arguments)]
fn list_selected_observed(
    uri: &SourceUri,
    endpoints: &Endpoints,
    credentials: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &Ledger,
    files: Option<&[String]>,
    mut unnameable: Option<&mut Vec<String>>,
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
                    if files.is_some_and(|files| !files.iter().any(|file| file == path)) {
                        continue;
                    }
                    if !member_path(path) {
                        // Every HuggingFace repository carries `.gitattributes`, and a
                        // leading dot is outside the member grammar. Refusing the whole
                        // listing over a file nothing would ever select made `source list`
                        // fail on every real repository; the Go resolver skips such rows
                        // and always has.
                        //
                        // The skip is safe in exactly one direction: a member that is not
                        // a tensor carrier cannot change a selection by being absent. One
                        // that IS a carrier would silently shrink the model, so it refuses
                        // instead — a smaller model that still downloads is the failure
                        // shape worth refusing loudly.
                        if is_safetensors(path) || is_index(path) {
                            let Some(skipped) = unnameable.as_mut() else {
                                return refuse(
                                    Code::KEY_GRAMMAR,
                                    format!(
                                        "tensor carrier {path:?} has a path outside the member \
                                         grammar, so the selection cannot name it"
                                    ),
                                );
                            };
                            skipped.push(path.to_string());
                        }
                        continue;
                    }
                    // LFS rows carry an exact sha256 and length; plain git rows carry a
                    // git oid, which is not an object identity here.
                    let (sha256, length) = match json_get(row, "lfs") {
                        Some(lfs) => {
                            let oid = json_str(lfs, "oid");
                            // A GATED repository answers an unauthenticated listing with
                            // the oid MASKED — sixty-four asterisks where the digest goes.
                            // Nothing downstream can recover from that, and without saying
                            // so here the failure surfaces much later as "this member has
                            // no digest", which reads as a size or format problem and sends
                            // the reader looking in the wrong place entirely.
                            if oid.is_some_and(|oid| {
                                !oid.is_empty() && oid.bytes().all(|b| b == b'*')
                            }) {
                                return refuse(
                                    Code::CREDENTIAL_REQUIRED,
                                    format!(
                                        "{path}: the provider masked this member's digest, \
                                         which is what a gated repository returns to a \
                                         caller it has not authorized — supply the owner's \
                                         token with --credential-file"
                                    ),
                                );
                            }
                            (
                                oid.and_then(digest64),
                                json_u64(lfs, "size").or_else(|| json_u64(row, "size")),
                            )
                        }
                        None => (None, json_u64(row, "size")),
                    };
                    members.push(SourceMember {
                        member: path.to_string(),
                        length,
                        sha256,
                        url: format!(
                            "{base}/{org}/{repo}/resolve/{revision}/{}",
                            path.replace(' ', "%20")
                        ),
                        primary: false,
                        companion: false,
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
                let file_id = json_u64(file, "id")
                    .filter(|id| *id > 0 && *id < (1u64 << 53))
                    .ok_or_else(|| Refusal {
                        code: Code::KEY_GRAMMAR,
                        detail: "a Civitai file requires a positive exact integer id".into(),
                    })?;
                let sha256 = json_get(file, "hashes")
                    .and_then(|hashes| json_str(hashes, "SHA256"))
                    .and_then(digest64);
                let url = match json_str(file, "downloadUrl") {
                    Some(explicit) => explicit.to_string(),
                    None => format!("{base}/api/download/models/{version}?fileId={file_id}"),
                };
                // A Civitai version is a bag of files (weights, a VAE, a config); the
                // one to take is the `primary` SafeTensors file, exactly as the Go
                // resolver selects it. `metadata.format` must agree or be silent.
                let format_ok = match json_get(file, "metadata").and_then(|m| json_str(m, "format"))
                {
                    Some(format) => format.eq_ignore_ascii_case("SafeTensor"),
                    None => true,
                };
                let safetensors = format_ok && name.to_ascii_lowercase().ends_with(".safetensors");
                let flagged = matches!(json_get(file, "primary"), Some(Json::Bool(true)));
                let primary = flagged && safetensors;
                let companion = !flagged && safetensors && sha256.is_some();
                // sizeKB is kilobytes as a float — an approximation, not a length.
                // The exact length is probed from the origin at pull time.
                members.push(SourceMember {
                    // Display filenames may change; the provider file ID is the reviewed
                    // profile member identity, shared with Creator and retained source trees.
                    member: format!("civitai/files/{file_id}"),
                    length: None,
                    sha256,
                    url,
                    primary,
                    companion,
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
    members.dedup();
    if members
        .windows(2)
        .any(|pair| pair[0].member == pair[1].member)
    {
        return refuse(
            Code::DUPLICATE_KEY,
            "the source lists one member twice with different facts",
        );
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
                // A source member is whatever the uploader pushed, up to gigabytes: it is
                // asked for as parallel ranges, or whole if the origin will not range.
                let fetched = fetch_ranged(
                    store,
                    &grant,
                    &member.url,
                    &policy,
                    credentials,
                    deadline,
                    &ledger,
                    Ranged::default(),
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

// ---------------------------------------------------------------------------- resolution

/// Where a member's object identity came from.
///
/// This distinction is the whole reason resolution is not a pure metadata read. A
/// HuggingFace LFS row carries a **content sha256** (`lfs.oid`), which IS the object id.
/// A plain git row carries a **git sha1**, which identifies a blob under git's own
/// `blob <len>\0` framing and is not a digest of these bytes at all. Treating the second
/// as the first produces a member that can never verify, and a member that can never
/// verify is one nobody notices until the 210 GB pull is already paid for. So the answer
/// says, per member, which of the two it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provenance {
    /// The provider stated a content sha256 and it was taken as given.
    Declared,
    /// Computed here, from bytes this resolution actually read.
    Computed,
}

impl Provenance {
    pub fn as_str(self) -> &'static str {
        match self {
            Provenance::Declared => "declared",
            Provenance::Computed => "computed",
        }
    }
}

/// One member of a pinned source with an EXACT identity — the fact a caller can size a
/// disk against, plan a fetch from, or refuse on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedMember {
    pub member: String,
    pub object: ObjectRef,
    pub url: String,
    pub provenance: Provenance,
    /// True for a member the selection names in its own right — a standalone
    /// `.safetensors`, or an index — as opposed to a shard reached only through an index's
    /// `weight_map`. A caller selects carriers; the shards come along because the index
    /// says they must.
    pub carrier: bool,
    /// For an index member, the shard members its `weight_map` names. Sorted and unique.
    pub requires: Vec<String>,
    /// A Civitai companion carrier (see [`SourceMember::companion`]): selected only when the
    /// provider's own selection plans as nothing reviewed and it composes one that does.
    pub companion: bool,
}

/// What one pinned source resolves to.
///
/// **`members.len()` is not `objects().len()`.** Two members of one repository can be the
/// same bytes — MiniMax-H3 ships `FL2VA/transformer/model.safetensors.index.json` and
/// `Ref2VA/transformer/model.safetensors.index.json` as one 38,323-byte object — and a
/// content-addressed store holds that object once. Any caller that assumes member→object
/// is injective will double-count a disk budget and mis-plan a fetch, so the two counts
/// are separate methods here rather than one number anyone can read the wrong way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolution {
    pub canonical: String,
    pub members: Vec<ResolvedMember>,
    /// sha256 over the canonical source name and every (member, digest, length) triple, in
    /// member order — one value naming this exact selection (the Go resolver's
    /// `SelectionSHA256`, byte-for-byte the same construction).
    pub selection_sha256: String,
}

impl Resolution {
    /// The DISTINCT objects this source reaches, sorted by digest. This is what a store
    /// admits and what a disk must hold.
    pub fn objects(&self) -> Vec<ObjectRef> {
        let mut objects: Vec<ObjectRef> = self.members.iter().map(|m| m.object.clone()).collect();
        objects.sort_by(|left, right| left.sha256.cmp(&right.sha256));
        objects.dedup_by(|left, right| left.sha256 == right.sha256);
        objects
    }

    /// Summed over MEMBERS — duplicates counted once per member. This is what a tree on an
    /// ordinary filesystem costs.
    pub fn member_bytes(&self) -> u64 {
        self.members.iter().map(|m| m.object.length).sum()
    }

    /// Summed over DISTINCT OBJECTS. This is the container-disk floor: a CAS store holds
    /// one copy of shared bytes however many members name them.
    pub fn object_bytes(&self) -> u64 {
        self.objects().iter().map(|o| o.length).sum()
    }

    /// NARROW this resolution to exactly the carriers named, plus the shards those
    /// carriers' indexes require. The Go resolver's `Plan.Select`, same rules.
    ///
    /// **This step is not optional, and omitting it fails silently.** A repository is not
    /// a model: `MiniMaxAI/MiniMax-H3@42ed227e` resolves to 112 tensor carriers totalling
    /// **498.3 GB**, while the two reviewed profiles that actually name a model select 5
    /// carriers which expand to **48 members / 47 objects / 210.3 GB**. A caller that
    /// resolves without narrowing does not get an error — it gets a download 2.4x larger
    /// that completes successfully, which is the worst shape a regression can take. So
    /// `select` lives beside `resolve`, and the CLI's `--source-profile` is the only way
    /// the narrowing can be forgotten by accident.
    ///
    /// A name that is not a carrier is refused rather than ignored: selecting a shard
    /// directly would silently drop the index that gives it meaning.
    pub fn select(&self, carriers: &[String]) -> Result<Resolution> {
        if carriers.is_empty() {
            return refuse(
                Code::MISSING_FIELD,
                "a selection names at least one carrier member",
            );
        }
        let mut wanted: Vec<String> = Vec::new();
        for name in carriers {
            let Some(row) = self.members.iter().find(|row| &row.member == name) else {
                return refuse(
                    Code::MISSING_FIELD,
                    format!("the source does not contain carrier member {name:?}"),
                );
            };
            if !row.carrier {
                return refuse(
                    Code::AMBIGUOUS_CLASSIFICATION,
                    format!(
                        "{name:?} is a shard reached through an index, not a carrier to                          select on its own"
                    ),
                );
            }
            wanted.push(row.member.clone());
            wanted.extend(row.requires.iter().cloned());
        }
        wanted.sort();
        wanted.dedup();
        let members: Vec<ResolvedMember> = self
            .members
            .iter()
            .filter(|row| wanted.contains(&row.member))
            .cloned()
            .collect();
        Ok(Resolution {
            selection_sha256: selection_digest(&self.canonical, &members),
            canonical: self.canonical.clone(),
            members,
        })
    }
}

/// Origin-independent identity of the exact selected logical file tree.
pub fn content_manifest(resolution: &Resolution) -> Result<crate::manifest::Manifest> {
    if resolution.members.is_empty() || resolution.members.len() > MEMBER_MAX {
        return refuse(
            Code::COUNT_CAP,
            "selected source member count is outside the bound",
        );
    }
    let mut lengths = std::collections::BTreeMap::new();
    for member in &resolution.members {
        if lengths
            .insert(&member.object.sha256, member.object.length)
            .is_some_and(|length| length != member.object.length)
        {
            return refuse(
                Code::LENGTH_MISMATCH,
                "selected members disagree on the same content length",
            );
        }
        for required in &member.requires {
            if !resolution.members.iter().any(|row| &row.member == required) {
                return refuse(
                    Code::MISSING_FIELD,
                    "selected source is missing an index shard",
                );
            }
        }
    }
    crate::manifest::Manifest::from_files(
        resolution
            .members
            .iter()
            .map(|member| (member.member.clone(), member.object.clone()))
            .collect(),
    )
}

/// Materialize precisely an accepted selection; never re-list or broaden its roster.
/// `progress` hears (present bytes, total bytes) of the selection's distinct objects.
#[allow(clippy::too_many_arguments)]
pub fn materialize_selected(
    store: &Store,
    owner: &str,
    resolution: &Resolution,
    policy: &SourcePolicy,
    credentials: &dyn CredentialProvider,
    deadline: Deadline,
    progress: crate::transport::SourceProgress<'_>,
) -> Result<(crate::source_artifact::TreeRoot, Vec<Pulled>)> {
    materialize_selected_with_options(
        store,
        owner,
        resolution,
        policy,
        credentials,
        deadline,
        progress,
        &crate::transport::SourceDownload::default(),
    )
}

/// Stream count, checkpoint sizing, cancellation and development fault points.
#[allow(clippy::too_many_arguments)]
pub fn materialize_selected_with_options(
    store: &Store,
    owner: &str,
    resolution: &Resolution,
    policy: &SourcePolicy,
    credentials: &dyn CredentialProvider,
    deadline: Deadline,
    progress: crate::transport::SourceProgress<'_>,
    options: &crate::transport::SourceDownload,
) -> Result<(crate::source_artifact::TreeRoot, Vec<Pulled>)> {
    use crate::ids::Doc;
    let tree = content_manifest(resolution)?;
    crate::source_artifact::check_roster(owner, &tree)?;
    let mut writer = crate::source_artifact::Writer::open(store, owner, tree.object_ref())?;
    let objects = resolution.objects();
    let (plan, _) = FetchPlan::of_objects(store, "selected-source", &objects)?;
    let mut ledger = Ledger::new();
    if let Some(cancellation) = &options.cancellation {
        ledger = ledger.with_cancellation(cancellation.clone());
    }
    // Every member's liveness is published before any of their bytes move.
    let mut jobs: Vec<crate::transport::sources::SourceJob> = vec![];
    let (mut seen, mut held_bytes) = (std::collections::BTreeSet::new(), 0);
    for member in &resolution.members {
        writer.hold_expected(&member.object)?;
        let object = &member.object;
        if !seen.insert(&object.sha256) {
            continue;
        }
        if matches!(store.record_valid(&object.sha256), Ok(row) if row.length == object.length) {
            held_bytes += object.length;
            continue;
        }
        jobs.push(crate::transport::sources::SourceJob {
            grant: DeliveryGrant::mint(&plan, object)?,
            url: member.url.clone(),
        });
    }
    let transferred = crate::transport::sources::fetch_sources(
        store,
        owner,
        &jobs,
        policy,
        credentials,
        deadline,
        &ledger,
        options,
        held_bytes,
        progress,
    )?;
    let mut pulled = vec![];
    let mut counted = std::collections::BTreeSet::new();
    for member in &resolution.members {
        writer.landed(&member.object)?;
        let first = counted.insert(&member.object.sha256);
        let fetched = jobs
            .iter()
            .position(|job| job.grant.object == member.object)
            .filter(|_| first);
        pulled.push(Pulled {
            member: member.member.clone(),
            object: member.object.clone(),
            transferred: fetched.map_or(0, |at| transferred[at]),
            held: fetched.is_none(),
        });
    }
    plan.complete(store)?;
    crate::transport::sources::discard_owner(store, owner)?;
    let result = writer.finish(&tree)?;
    Ok((result, pulled))
}

/// The Go resolver's `finishPlan` digest, same construction: the canonical source name and
/// every (member, digest, length) triple, each field NUL-terminated, in member order. One
/// value naming this exact selection.
fn selection_digest(canonical: &str, members: &[ResolvedMember]) -> String {
    let mut selection = Vec::new();
    selection.extend_from_slice(canonical.as_bytes());
    selection.push(0);
    for row in members {
        selection.extend_from_slice(row.member.as_bytes());
        selection.push(0);
        selection.extend_from_slice(row.object.sha256.as_bytes());
        selection.push(0);
        selection.extend_from_slice(row.object.length.to_string().as_bytes());
        selection.push(0);
    }
    crate::sha256::hex_digest(&selection)
}

fn lower_ends_with(value: &str, suffix: &str) -> bool {
    value.to_ascii_lowercase().ends_with(suffix)
}

/// How much of a member is asked for on the first ask, when all that is wanted is its
/// header.
///
/// MEASURED against the model this exists for: MiniMax-H3's 44 shard headers are 4-6 KB
/// each and its four indexes are 12-38 KB, so 64 KiB covers every member of a 210.3 GB
/// selection in ONE request. The size is a trade against latency and not against
/// bandwidth: this crate's client sends `Connection: close`, so every ask pays a fresh
/// connect plus TLS plus time-to-first-byte, measured at 0.41 s against HuggingFace's CDN.
/// A two-step read — eight bytes for the declared length, then the header — would double
/// that for every member and save kilobytes. So the first ask is speculative and generous,
/// and a header that does not fit costs exactly one more.
pub const HEAD_PROBE_BYTES: u64 = 64 << 10;

/// One member's header, and the length of the object it was cut from.
///
/// **`length` is the FULL object length, never `head.len()`.** A carrier header declares
/// tensor regions and every reader checks those regions against the file's real size, so a
/// header divorced from its length can be geometrically checked against nothing. The two
/// travel together for that reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemberHead {
    pub member: String,
    pub length: u64,
    pub head: Vec<u8>,
}

/// A member whose header could not be read cheaply, and why. NOT a refusal: an origin that
/// will not serve a range is a real origin, and the honest verdict there is that the plan
/// is undecided rather than that the model is bad.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnreadMember {
    pub member: String,
    pub why: String,
}

/// What one header read cost and reached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeadRead {
    pub heads: Vec<MemberHead>,
    pub unread: Vec<UnreadMember>,
    /// Bytes actually pulled off the wire.
    pub bytes: u64,
    /// Ranged GETs actually issued. The rate-limit unit, and the number a caller sizes a
    /// provider's budget against.
    pub requests: usize,
}

/// Read every member's HEADER, and nothing else.
///
/// This is the whole of tfs-076: the facts that decide whether a selection can convert are
/// in the members' headers, so they can be read for kilobytes before anything is rented and
/// before a byte of payload moves. MiniMax-H3's 48 members total 210.3 GB and their headers
/// total 275 KB; three separate H3 runs moved the 210.3 GB and then refused on facts inside
/// the 275 KB.
///
/// The read is per-member and BEST EFFORT. Two answers are possible and they mean different
/// things: a member whose header arrives is a member the plan can be decided from, and a
/// member whose header will not arrive cheaply lands in `unread`, which makes the plan
/// UNDECIDED rather than refused. Only a disagreement about identity — the origin's own
/// length contradicting the length this selection is pinned to — refuses here, because that
/// is a fault in the selection itself and no amount of transfer improves it.
pub fn read_heads(
    resolution: &Resolution,
    uri: &SourceUri,
    endpoints: &Endpoints,
    credentials: &dyn CredentialProvider,
    deadline: Deadline,
) -> Result<HeadRead> {
    let policy = endpoints.policy(uri);
    let ledger = Ledger::new();
    let mut out = HeadRead {
        heads: Vec::with_capacity(resolution.members.len()),
        unread: Vec::new(),
        bytes: 0,
        requests: 0,
    };
    for row in &resolution.members {
        // An index is a whole small document — its `weight_map` IS the fact wanted — while
        // a tensor carrier is a prefix of a file that can be 5 GiB. Both are bounded.
        let whole = is_index(&row.member) || lower_ends_with(&row.member, ".json");
        if whole && row.object.length > INDEX_MAX_BYTES {
            out.unread.push(UnreadMember {
                member: row.member.clone(),
                why: format!(
                    "index is {} B, over the {INDEX_MAX_BYTES} B bound",
                    row.object.length
                ),
            });
            continue;
        }
        let first = if whole {
            row.object.length
        } else {
            row.object.length.min(HEAD_PROBE_BYTES)
        };
        let prefix = fetch_prefix(&row.url, &policy, credentials, first, deadline, &ledger)?;
        out.requests += 1;
        // The two halves of an identity may not disagree. A selection pinned to one length
        // and an origin serving another is a fault in the SELECTION, decidable here for one
        // request and never improved by transferring the object.
        if let Some(total) = prefix.total {
            if total != row.object.length {
                return refuse(
                    Code::LENGTH_MISMATCH,
                    format!(
                        "{}: the selection pins {} B and the origin says {total} B",
                        row.member, row.object.length
                    ),
                );
            }
        }
        let Some(mut head) = prefix.bytes else {
            out.unread.push(UnreadMember {
                member: row.member.clone(),
                why: "origin would not serve a range for an object over the ask".into(),
            });
            continue;
        };
        out.bytes += head.len() as u64;
        // What the first ask learned about how much was actually needed. A safetensors
        // carrier says so in its own first eight bytes; anything the cap or the file
        // refuses is NOT re-asked, because the short head already decides it — that is run
        // 309's `carrier_header_cap`, reached for 64 KiB instead of 210.3 GB.
        let need = if whole {
            row.object.length
        } else if head.len() >= 8 {
            let declared = u64::from_le_bytes(head[0..8].try_into().expect("eight bytes"));
            match declared.checked_add(8) {
                Some(need)
                    if declared <= crate::limits::CARRIER_HEADER_MAX_BYTES as u64
                        && need <= row.object.length =>
                {
                    need
                }
                _ => head.len() as u64,
            }
        } else {
            head.len() as u64
        };
        // The first ask is speculative, so it usually OVERSHOOTS. Trim to what the header
        // actually is: `bytes` is what the wire cost and `head` is what the plan is built
        // from, and reporting one as the other would misstate the guard in either direction.
        if need < head.len() as u64 {
            head.truncate(need as usize);
        }
        if need > head.len() as u64 {
            let more = fetch_prefix(&row.url, &policy, credentials, need, deadline, &ledger)?;
            out.requests += 1;
            match more.bytes {
                Some(bytes) => {
                    out.bytes += bytes.len() as u64;
                    head = bytes;
                }
                None => {
                    out.unread.push(UnreadMember {
                        member: row.member.clone(),
                        why: format!("origin would not serve the {need} B header range"),
                    });
                    continue;
                }
            }
        }
        out.heads.push(MemberHead {
            member: row.member.clone(),
            length: row.object.length,
            head,
        });
    }
    Ok(out)
}

fn is_index(member: &str) -> bool {
    lower_ends_with(member, ".safetensors.index.json")
}

fn is_safetensors(member: &str) -> bool {
    lower_ends_with(member, ".safetensors")
}

/// Civitai carrier members have provider IDs rather than filename extensions.
pub(crate) fn is_civitai_member(member: &str) -> bool {
    member.strip_prefix("civitai/files/").is_some_and(|id| {
        id.parse::<u64>()
            .is_ok_and(|value| value > 0 && value < (1u64 << 53) && value.to_string() == id)
    })
}

/// Resolve `relative` against the directory of `member`, the way the Go resolver's
/// `path.Clean(path.Join(path.Dir(member), shard))` does. `None` when the reference walks
/// above the repository root — an index may not name its way out of its own tree.
///
/// **Shared with `ingest::carrier::read_sharded` on purpose.** The resolver uses this to
/// decide which shards a selection must FETCH; the ingest uses it to decide which carrier
/// a `weight_map` value NAMES. Two joins would be two answers to one question, and the
/// disagreement would only ever be found by a run that downloaded everything first.
pub(crate) fn join_member(member: &str, relative: &str) -> Option<String> {
    let directory = member.rsplit_once('/').map_or("", |(head, _)| head);
    let mut parts: Vec<&str> = Vec::new();
    for segment in directory.split('/').chain(relative.split('/')) {
        match segment {
            "" | "." => {}
            ".." => {
                parts.pop()?;
            }
            other => parts.push(other),
        }
    }
    (!parts.is_empty()).then(|| parts.join("/"))
}

/// Resolve one pinned source to its exact member list: what to fetch, at what identity,
/// from where — WITHOUT moving a tensor byte.
///
/// This is the owner-side question. A container disk is sized from the total here before a
/// rental is requested, so resolution must be answerable anywhere the credential lives and
/// must not need a `Store`: it takes none, and returns facts rather than admitting any.
///
/// Three steps, all through the fenced transport:
///
/// 1. **List** what the pinned revision reaches (`list_observed`).
/// 2. **Select** the tensor carriers — `*.safetensors` and `*.safetensors.index.json` for
///    HuggingFace, the one primary SafeTensors file for Civitai plus its companions.
/// 3. **Expand and identify.** Every index is read (it is JSON, tens of kilobytes) and its
///    `weight_map` expanded into the shards it names, which must be present in the
///    listing. Reading it is also what gives it an identity: an index is a plain git blob,
///    so its digest exists only once these bytes have been hashed.
///
/// The bytes read are bounded by construction and not by hope. After step 2 the ONLY
/// members without a provider-declared content digest are the indexes — HuggingFace keeps
/// every tensor file in LFS — so "hash what the provider did not identify" costs kilobytes
/// on a repository whose weights are hundreds of gigabytes. Measured on
/// `MiniMaxAI/MiniMax-H3@42ed227e`: 280 files, 112 selected, 104 LFS and 8 plain, and all
/// 8 of the plain ones are indexes.
pub fn resolve(
    uri: &SourceUri,
    endpoints: &Endpoints,
    credentials: &dyn CredentialProvider,
    deadline: Deadline,
) -> Result<Resolution> {
    resolve_with(uri, endpoints, credentials, deadline, None)
}

/// [`resolve`] for a caller that then selects carriers BY NAME. A tensor carrier whose path
/// is outside the member grammar cannot be one of them, so it is skipped and returned for
/// the caller to report, where `resolve` refuses: a repository of 949 LoRAs with one
/// `…(QQQQ4413).safetensors` refused every upload from it.
pub fn resolve_named(
    uri: &SourceUri,
    endpoints: &Endpoints,
    credentials: &dyn CredentialProvider,
    deadline: Deadline,
) -> Result<(Resolution, Vec<String>)> {
    let mut skipped = Vec::new();
    let resolution = resolve_with(uri, endpoints, credentials, deadline, Some(&mut skipped))?;
    Ok((resolution, skipped))
}

fn resolve_with(
    uri: &SourceUri,
    endpoints: &Endpoints,
    credentials: &dyn CredentialProvider,
    deadline: Deadline,
    unnameable: Option<&mut Vec<String>>,
) -> Result<Resolution> {
    let ledger = Ledger::new();
    let policy = endpoints.policy(uri);
    let listed = list_selected_observed(
        uri,
        endpoints,
        credentials,
        deadline,
        &ledger,
        None,
        unnameable,
    )?;

    let selected: Vec<SourceMember> = match uri {
        SourceUri::HuggingFace { .. } => listed
            .into_iter()
            .filter(|member| is_safetensors(&member.member) || is_index(&member.member))
            .collect(),
        SourceUri::Civitai { .. } => {
            let primary = listed.iter().filter(|member| member.primary).count();
            if primary != 1 {
                return refuse(
                    Code::COUNT_CAP,
                    format!(
                        "the Civitai version declares {primary} primary SafeTensors files; \
                         exactly one is required"
                    ),
                );
            }
            listed
                .into_iter()
                .filter(|member| member.primary || member.companion)
                .collect()
        }
    };
    if selected.is_empty() {
        return refuse(
            Code::MISSING_FIELD,
            "the source reaches no tensor carrier (.safetensors or .safetensors.index.json)",
        );
    }
    if selected.len() > MEMBER_MAX {
        return refuse(
            Code::COUNT_CAP,
            format!("the selection names more than {MEMBER_MAX} members"),
        );
    }

    // Pass one: an exact identity for every selected member, and the indexes' bodies.
    let mut resolved: Vec<ResolvedMember> = Vec::with_capacity(selected.len());
    let mut indexes: Vec<(String, Vec<u8>)> = Vec::new();
    for member in &selected {
        let (object, provenance) = if is_index(&member.member) {
            // An index must be READ regardless of what the provider declared: its
            // `weight_map` is the expansion, and reading it is also the only way its
            // identity exists at all when it is a plain git blob.
            let (body, _) = api_get(
                &member.url,
                &policy,
                credentials,
                INDEX_MAX_BYTES,
                deadline,
                &ledger,
            )?;
            if body.is_empty() {
                return refuse(
                    Code::LENGTH_MISMATCH,
                    format!("{}: the origin served an empty index", member.member),
                );
            }
            let object = ObjectRef::of(&body);
            // A declared digest is not needed here, but if one exists the two opinions
            // about these bytes must agree.
            if let Some(declared) = &member.sha256 {
                if declared != &object.sha256 {
                    return refuse(
                        Code::OBJECT_ID_MISMATCH,
                        format!(
                            "{}: the provider declared sha256:{declared} but the bytes are {}",
                            member.member,
                            object.id()
                        ),
                    );
                }
            }
            indexes.push((member.member.clone(), body));
            (object, Provenance::Computed)
        } else {
            match &member.sha256 {
                Some(sha256) => {
                    // Declared identity. A Civitai listing states kilobytes as a float,
                    // which is an approximation and never a length, so the exact one comes
                    // from a one-byte ranged probe through the same fence.
                    let length = match member.length {
                        Some(length) if length > 0 => length,
                        _ => probe_length(&member.url, &policy, credentials, deadline, &ledger)?,
                    };
                    if length == 0 {
                        return refuse(
                            Code::LENGTH_MISMATCH,
                            format!("{}: the origin reports a zero-length member", member.member),
                        );
                    }
                    (
                        ObjectRef {
                            sha256: sha256.clone(),
                            length,
                        },
                        Provenance::Declared,
                    )
                }
                None => {
                    // A tensor carrier the provider does not identify. On HuggingFace this
                    // means a `.safetensors` committed outside LFS, which git keeps small;
                    // it is hashed here, under the same bound the pull path uses for the
                    // class, and refused rather than guessed at if it is larger.
                    if member
                        .length
                        .is_some_and(|length| length > UNPINNED_MEMBER_MAX_BYTES)
                    {
                        return refuse(
                            Code::SIZE_CAP,
                            format!(
                                "{}: the provider identifies no digest for a carrier larger \
                                 than {UNPINNED_MEMBER_MAX_BYTES} bytes, so resolving it \
                                 would mean downloading it",
                                member.member
                            ),
                        );
                    }
                    let (body, _) = api_get(
                        &member.url,
                        &policy,
                        credentials,
                        UNPINNED_MEMBER_MAX_BYTES,
                        deadline,
                        &ledger,
                    )?;
                    if body.is_empty() {
                        return refuse(
                            Code::LENGTH_MISMATCH,
                            format!("{}: the origin served an empty member", member.member),
                        );
                    }
                    (ObjectRef::of(&body), Provenance::Computed)
                }
            }
        };
        resolved.push(ResolvedMember {
            member: member.member.clone(),
            object,
            url: member.url.clone(),
            provenance,
            carrier: true,
            requires: Vec::new(),
            companion: member.companion,
        });
    }

    // Pass two: expand every index into the shards it names. A shard the listing does not
    // contain is a broken selection, not a member to invent.
    let present: Vec<String> = resolved.iter().map(|row| row.member.clone()).collect();
    let mut referenced: Vec<String> = Vec::new();
    for (member, body) in &indexes {
        let document = jcs::parse(body, INDEX_MAX_BYTES as usize)?;
        let Some(Json::Obj(weight_map)) = json_get(&document, "weight_map") else {
            return refuse(
                Code::MISSING_FIELD,
                format!("{member}: the tensor index declares no weight_map object"),
            );
        };
        if weight_map.is_empty() || weight_map.len() > WEIGHT_MAP_MAX {
            return refuse(
                Code::COUNT_CAP,
                format!(
                    "{member}: the tensor index names {} tensors",
                    weight_map.len()
                ),
            );
        }
        let mut requires: Vec<String> = Vec::new();
        for (tensor, shard) in weight_map {
            let Json::Str(shard) = shard else {
                return refuse(
                    Code::WRONG_TYPE,
                    format!("{member}: weight_map entry {tensor:?} does not name a file"),
                );
            };
            let Some(resolved_shard) = join_member(member, shard) else {
                return refuse(
                    Code::PATH_ILLEGAL,
                    format!("{member}: weight_map names {shard:?}, which leaves the repository"),
                );
            };
            if !member_path(&resolved_shard) || !present.contains(&resolved_shard) {
                return refuse(
                    Code::MISSING_FIELD,
                    format!(
                        "{member}: weight_map names shard {resolved_shard:?}, which the \
                         revision does not contain"
                    ),
                );
            }
            requires.push(resolved_shard);
        }
        requires.sort();
        requires.dedup();
        referenced.extend(requires.iter().cloned());
        for row in resolved.iter_mut() {
            if &row.member == member {
                row.requires = requires.clone();
            }
        }
    }
    // A shard reached only through an index is not itself a carrier: a caller selects the
    // index, and the shards follow from what it says.
    for row in resolved.iter_mut() {
        if referenced.contains(&row.member) {
            row.carrier = false;
        }
    }

    resolved.sort_by(|left, right| left.member.cmp(&right.member));
    if resolved
        .windows(2)
        .any(|pair| pair[0].member == pair[1].member)
    {
        return refuse(Code::DUPLICATE_KEY, "the selection names one member twice");
    }

    let canonical = match uri {
        SourceUri::HuggingFace {
            org,
            repo,
            revision,
        } => format!("hf://{org}/{repo}@{revision}"),
        SourceUri::Civitai { version } => format!("civitai://{version}"),
    };
    Ok(Resolution {
        selection_sha256: selection_digest(&canonical, &resolved),
        canonical,
        members: resolved,
    })
}

/// Resolve only explicitly named ordinary files at one immutable HF revision.
/// No suffix has special meaning here: selecting Python bytes never executes them,
/// and selecting an index does not expand it into tensor shards. Tensor carriers
/// continue to use `resolve` and reviewed profile/carrier selection.
///
/// A name ending in `/` selects every ordinary file under that folder (a tokenizer or
/// processor, whose file set varies by class); tensor carriers there are never selected.
pub fn resolve_files(
    uri: &SourceUri,
    files: &[String],
    endpoints: &Endpoints,
    credentials: &dyn CredentialProvider,
    deadline: Deadline,
) -> Result<Resolution> {
    let SourceUri::HuggingFace {
        org,
        repo,
        revision,
    } = uri
    else {
        return refuse(
            Code::SOURCE_NOT_ALLOWED,
            "ordinary file selection requires HuggingFace",
        );
    };
    if files.is_empty() || files.len() > MEMBER_MAX {
        return refuse(
            Code::COUNT_CAP,
            "ordinary file selection count is outside the bound",
        );
    }
    let (folders, mut names): (Vec<String>, Vec<String>) =
        files.iter().cloned().partition(|name| name.ends_with('/'));
    names.sort();
    names.dedup();
    if names
        .iter()
        .chain(&folders)
        .map(|name| name.trim_end_matches('/'))
        .any(|name| !member_path(name) || crate::manifest::check_path(name).is_err())
    {
        return refuse(
            Code::PATH_ILLEGAL,
            "ordinary file selection requires exact member or folder paths",
        );
    }
    let ledger = Ledger::new();
    let policy = endpoints.policy(uri);
    let filter = folders.is_empty().then_some(names.as_slice());
    let listed =
        list_selected_observed(uri, endpoints, credentials, deadline, &ledger, filter, None)?;
    for folder in &folders {
        let before = names.len();
        names.extend(
            listed
                .iter()
                .filter(|m| m.member.starts_with(folder.as_str()))
                .filter(|m| !is_safetensors(&m.member) && !is_index(&m.member))
                .map(|m| m.member.clone()),
        );
        if names.len() == before {
            return refuse(
                Code::MISSING_FIELD,
                format!("the source has no ordinary file under {folder:?}"),
            );
        }
    }
    names.sort();
    names.dedup();
    if names.len() > MEMBER_MAX {
        return refuse(
            Code::COUNT_CAP,
            "ordinary file selection count is outside the bound",
        );
    }
    // Establish the whole roster before fetching any body. A missing later member
    // must not turn a partial requested tree into a successful smaller selection.
    let selected = names
        .iter()
        .map(|name| {
            listed
                .iter()
                .find(|member| &member.member == name)
                .ok_or_else(|| Refusal {
                    code: Code::MISSING_FIELD,
                    detail: format!("the source does not contain requested file {name:?}"),
                })
        })
        .collect::<Result<Vec<_>>>()?;
    let listed_bytes = selected.iter().try_fold(0u64, |total, member| {
        total.checked_add(member.length.unwrap_or(0))
    });
    if listed_bytes.is_none_or(|bytes| bytes > FILE_SELECTION_MAX_BYTES) {
        return refuse(
            Code::SIZE_CAP,
            "ordinary file selection exceeds its byte bound",
        );
    }
    let mut remaining = FILE_SELECTION_MAX_BYTES;
    let mut resolved = Vec::with_capacity(selected.len());
    for member in selected {
        if member
            .length
            .map_or(remaining == 0, |length| length > remaining)
        {
            return refuse(
                Code::SIZE_CAP,
                "ordinary file selection exceeds its byte bound",
            );
        }
        let (object, provenance) = if let Some(sha256) = &member.sha256 {
            let length = match member.length {
                Some(length) => length,
                None => probe_length(&member.url, &policy, credentials, deadline, &ledger)?,
            };
            (
                ObjectRef {
                    sha256: sha256.clone(),
                    length,
                },
                Provenance::Declared,
            )
        } else {
            let (body, _) = api_get(
                &member.url,
                &policy,
                credentials,
                remaining,
                deadline,
                &ledger,
            )?;
            if member
                .length
                .is_some_and(|length| length != body.len() as u64)
            {
                return refuse(
                    Code::LENGTH_MISMATCH,
                    "ordinary file body disagrees with listed length",
                );
            }
            (ObjectRef::of(&body), Provenance::Computed)
        };
        remaining = remaining
            .checked_sub(object.length)
            .ok_or_else(|| Refusal {
                code: Code::SIZE_CAP,
                detail: "ordinary file selection exceeds its byte bound".into(),
            })?;
        resolved.push(ResolvedMember {
            member: member.member.clone(),
            object,
            url: member.url.clone(),
            provenance,
            carrier: false,
            requires: Vec::new(),
            companion: false,
        });
    }
    let canonical = format!("hf://{org}/{repo}@{revision}");
    Ok(Resolution {
        selection_sha256: selection_digest(&canonical, &resolved),
        canonical,
        members: resolved,
    })
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

    // ------------------------------------------------------------------ resolution (tfs-067)

    /// One HTTP response with a redirect whose `Location` is RELATIVE, which is the form
    /// HuggingFace answers for every plain git blob.
    fn relative_redirect(location: &str) -> Vec<u8> {
        format!(
            "HTTP/1.1 307 Temporary Redirect\r\nLocation: {location}\r\nContent-Length: 0\r\n\
             Connection: close\r\n\r\n"
        )
        .into_bytes()
    }

    /// MiniMax-H3's exact shape, small enough to serve from one thread.
    ///
    /// Two sibling directories each hold a sharded transformer, and their two
    /// `model.safetensors.index.json` members are BYTE-IDENTICAL — the real repository
    /// ships `FL2VA/transformer/…` and `Ref2VA/transformer/…` as one 38,323-byte object,
    /// `sha256:fb457a26…`. Because each index names its shards RELATIVELY, the same bytes
    /// expand to different shards in each directory. That is the case a resolver assuming
    /// member→object is injective gets wrong, and it is why `members` and `objects` are
    /// separate counts.
    struct H3Shape {
        tree: String,
        index_body: Vec<u8>,
        shards: Vec<(String, ObjectRef)>,
        vae: (String, ObjectRef),
    }

    fn h3_shape() -> H3Shape {
        let index_body = br#"{"weight_map":{"a.weight":"model-00001-of-00002.safetensors","b.weight":"model-00002-of-00002.safetensors"}}"#.to_vec();
        let mut shards = Vec::new();
        for directory in ["FL2VA", "Ref2VA"] {
            for shard in 1..=2u32 {
                let member =
                    format!("{directory}/transformer/model-0000{shard}-of-00002.safetensors");
                // Seeded from the member path so no two shards collide by accident: the
                // duplicate object in this fixture must be the one the test is about.
                let bytes: Vec<u8> = member
                    .as_bytes()
                    .iter()
                    .copied()
                    .cycle()
                    .take(8_000)
                    .collect();
                shards.push((member, ObjectRef::of(&bytes)));
            }
        }
        let vae_bytes: Vec<u8> = (0u32..1_500).flat_map(|i| (i * 7).to_le_bytes()).collect();
        let vae = (
            "vae/diffusion_pytorch_model.safetensors".to_string(),
            ObjectRef::of(&vae_bytes),
        );
        let mut rows: Vec<String> = Vec::new();
        for (member, object) in shards.iter().chain(std::iter::once(&vae)) {
            rows.push(format!(
                r#"{{"type":"file","path":"{member}","size":{},"oid":"gitsha1","lfs":{{"oid":"{}","size":{}}}}}"#,
                object.length, object.sha256, object.length
            ));
        }
        for directory in ["FL2VA", "Ref2VA"] {
            // Plain git blobs: a git sha1 and NO lfs object. Their content digest does not
            // exist until these bytes are read.
            rows.push(format!(
                r#"{{"type":"file","path":"{directory}/transformer/model.safetensors.index.json","size":{},"oid":"89d6b0862f66080854b239ad7a0b50c31a6de986"}}"#,
                index_body.len()
            ));
        }
        // Not a tensor carrier: the selection must drop it.
        rows.push(r#"{"type":"file","path":"config.json","size":604,"oid":"gitsha1"}"#.to_string());
        rows.push(r#"{"type":"directory","path":"vae"}"#.to_string());
        H3Shape {
            tree: format!("[{}]", rows.join(",")),
            index_body,
            shards,
            vae,
        }
    }

    /// The whole of phase one, on the shape that cost a 210 GB run: the selection drops
    /// non-carriers, the indexes are read and hashed, `weight_map` expands relatively, the
    /// shards keep their LFS identity untouched, and 7 members resolve to 6 objects.
    ///
    /// One index is served behind a RELATIVE redirect, which is what HuggingFace really
    /// answers for a plain git blob (tfs-060). Before the `api.rs` half of that fix this
    /// test fails with `REDIRECT_REFUSED`.
    #[test]
    fn resolve_expands_shards_and_identifies_every_member_class() {
        let shape = h3_shape();
        let revision = "ab".repeat(20);
        let tree = shape.tree.clone();
        let index_body = shape.index_body.clone();
        let index_object = ObjectRef::of(&shape.index_body);
        let revision_for_routes = revision.clone();
        // tree(1) + FL2VA index 307 + its 200 + Ref2VA index 200 = 4. No tensor byte moves:
        // every shard is identified by the listing alone.
        let (base, seen, handle) = origin(4, move |request| {
            let r = &revision_for_routes;
            if request.path == format!("/api/models/org/repo/tree/{r}?recursive=true") {
                ok_body(tree.as_bytes(), "application/json")
            } else if request.path
                == format!("/org/repo/resolve/{r}/FL2VA/transformer/model.safetensors.index.json")
            {
                relative_redirect(&format!(
                    "/api/resolve-cache/models/org/repo/{r}/FL2VA.json"
                ))
            } else if request.path == format!("/api/resolve-cache/models/org/repo/{r}/FL2VA.json")
                || request.path
                    == format!(
                        "/org/repo/resolve/{r}/Ref2VA/transformer/model.safetensors.index.json"
                    )
            {
                ok_body(&index_body, "application/json")
            } else {
                panic!(
                    "unrouted path {} — resolution read something it should not",
                    request.path
                )
            }
        });
        let uri = SourceUri::parse(&format!("hf://org/repo@{revision}")).unwrap();
        let endpoints = Endpoints {
            huggingface: base,
            allow_local: true,
            ..Endpoints::default()
        };
        let resolved = resolve(&uri, &endpoints, &Anonymous, Deadline::none()).unwrap();

        // The selection is the tensor carriers and nothing else: config.json is gone.
        assert_eq!(resolved.members.len(), 7);
        assert!(!resolved
            .members
            .iter()
            .any(|row| row.member == "config.json"));

        // NON-INJECTIVE: 7 members, 6 distinct objects. The two indexes are one object.
        assert_eq!(resolved.objects().len(), 6);
        assert_eq!(
            resolved.member_bytes() - resolved.object_bytes(),
            index_object.length,
            "the duplicated index must be counted once against a disk and twice against a tree"
        );

        // Digest provenance, per member class.
        for (member, object) in shape.shards.iter().chain(std::iter::once(&shape.vae)) {
            let row = resolved
                .members
                .iter()
                .find(|row| &row.member == member)
                .unwrap_or_else(|| panic!("{member} missing"));
            assert_eq!(&row.object, object, "{member}: LFS identity must survive");
            assert_eq!(
                row.provenance,
                Provenance::Declared,
                "{member}: an LFS row states a content sha256"
            );
        }
        for directory in ["FL2VA", "Ref2VA"] {
            let member = format!("{directory}/transformer/model.safetensors.index.json");
            let row = resolved
                .members
                .iter()
                .find(|row| row.member == member)
                .unwrap();
            assert_eq!(
                row.object, index_object,
                "a plain git blob's identity is the digest of its bytes, never its git sha1"
            );
            assert_eq!(row.provenance, Provenance::Computed);
            // The SAME index bytes expand to the shards of their OWN directory.
            assert_eq!(
                row.requires,
                vec![
                    format!("{directory}/transformer/model-00001-of-00002.safetensors"),
                    format!("{directory}/transformer/model-00002-of-00002.safetensors"),
                ]
            );
            assert!(row.carrier, "an index is a carrier");
        }

        // A shard reached through an index is not itself a carrier; a standalone one is.
        for (member, _) in &shape.shards {
            let row = resolved
                .members
                .iter()
                .find(|row| &row.member == member)
                .unwrap();
            assert!(!row.carrier, "{member} is reached through its index");
        }
        assert!(
            resolved
                .members
                .iter()
                .find(|row| row.member == shape.vae.0)
                .unwrap()
                .carrier,
            "an unsharded .safetensors is its own carrier"
        );

        assert_eq!(resolved.canonical, format!("hf://org/repo@{revision}"));
        assert_eq!(resolved.selection_sha256.len(), 64);
        assert_eq!(seen.lock().unwrap().len(), 4, "resolution is four reads");
        handle.join().unwrap();
    }

    /// A gated repository answers an unauthenticated listing with the LFS oid replaced by
    /// sixty-four asterisks. Verified live against `meta-llama/Llama-3.2-1B`, whose
    /// `model.safetensors` lists `size: 2471645608` and `oid: "****…"`. Read naively that
    /// member simply has no digest, and the failure surfaces as a size complaint about a
    /// 2.4 GB file — pointing at the wrong problem entirely. The credential is the answer,
    /// so the refusal says so.
    #[test]
    fn a_gated_repository_names_the_credential_and_not_the_size() {
        let revision = "56".repeat(20);
        let tree = format!(
            r#"[{{"type":"file","path":"model.safetensors","size":2471645608,"oid":"gitsha1","lfs":{{"oid":"{}","size":2471645608}}}}]"#,
            "*".repeat(64)
        );
        let revision_for_routes = revision.clone();
        let (base, _, handle) = origin(1, move |request| {
            let r = &revision_for_routes;
            assert_eq!(
                request.path,
                format!("/api/models/org/gated/tree/{r}?recursive=true")
            );
            ok_body(tree.as_bytes(), "application/json")
        });
        let uri = SourceUri::parse(&format!("hf://org/gated@{revision}")).unwrap();
        let endpoints = Endpoints {
            huggingface: base,
            allow_local: true,
            ..Endpoints::default()
        };
        let refusal = resolve(&uri, &endpoints, &Anonymous, Deadline::none()).unwrap_err();
        assert_eq!(refusal.code, Code::CREDENTIAL_REQUIRED);
        assert!(refusal.detail.contains("--credential-file"));
        handle.join().unwrap();
    }

    /// Every HuggingFace repository carries `.gitattributes`, whose leading dot is outside
    /// the member grammar. Refusing the whole listing over it made `source list` fail on
    /// every real repository; a non-carrier is skipped, a carrier is still refused.
    #[test]
    fn an_unnameable_non_carrier_is_skipped_and_an_unnameable_carrier_is_not() {
        let revision = "78".repeat(20);
        let shard: Vec<u8> = (0u32..1_000).flat_map(|i| i.to_le_bytes()).collect();
        let object = ObjectRef::of(&shard);
        let good = format!(
            r#"[{{"type":"file","path":".gitattributes","size":1519,"oid":"gitsha1"}},
                {{"type":"file","path":"model.safetensors","size":{},"oid":"gitsha1","lfs":{{"oid":"{}","size":{}}}}}]"#,
            object.length, object.sha256, object.length
        );
        let revision_for_routes = revision.clone();
        let (base, _, handle) = origin(1, move |_request| {
            let _ = &revision_for_routes;
            ok_body(good.as_bytes(), "application/json")
        });
        let uri = SourceUri::parse(&format!("hf://org/repo@{revision}")).unwrap();
        let endpoints = Endpoints {
            huggingface: base,
            allow_local: true,
            ..Endpoints::default()
        };
        let resolved = resolve(&uri, &endpoints, &Anonymous, Deadline::none()).unwrap();
        assert_eq!(resolved.members.len(), 1);
        assert_eq!(resolved.members[0].member, "model.safetensors");
        handle.join().unwrap();

        // The same path, but on a member the selection WOULD have taken: refused, because
        // silently dropping it would shrink the model and still succeed.
        let bad = r#"[{"type":"file","path":".hidden/model.safetensors","size":10,"oid":"g"}]"#;
        let (base, _, handle) = origin(1, move |_| ok_body(bad.as_bytes(), "application/json"));
        let endpoints = Endpoints {
            huggingface: base,
            allow_local: true,
            ..Endpoints::default()
        };
        assert_eq!(
            resolve(&uri, &endpoints, &Anonymous, Deadline::none())
                .unwrap_err()
                .code,
            Code::KEY_GRAMMAR
        );
        handle.join().unwrap();
    }

    /// NARROWING is the difference between 210 GB and 498 GB, and its absence is a
    /// SUCCESSFUL download of the wrong size. Selecting one carrier keeps that carrier and
    /// exactly the shards its own index names — not its sibling's, though the two indexes
    /// are the same object.
    #[test]
    fn select_narrows_to_one_carrier_and_the_shards_it_requires() {
        let shape = h3_shape();
        let revision = "34".repeat(20);
        let tree = shape.tree.clone();
        let index_body = shape.index_body.clone();
        let revision_for_routes = revision.clone();
        let (base, _, handle) = origin(3, move |request| {
            let r = &revision_for_routes;
            if request.path == format!("/api/models/org/repo/tree/{r}?recursive=true") {
                ok_body(tree.as_bytes(), "application/json")
            } else {
                ok_body(&index_body, "application/json")
            }
        });
        let uri = SourceUri::parse(&format!("hf://org/repo@{revision}")).unwrap();
        let endpoints = Endpoints {
            huggingface: base,
            allow_local: true,
            ..Endpoints::default()
        };
        let whole = resolve(&uri, &endpoints, &Anonymous, Deadline::none()).unwrap();
        assert_eq!(whole.members.len(), 7);

        let narrowed = whole
            .select(&["FL2VA/transformer/model.safetensors.index.json".to_string()])
            .unwrap();
        assert_eq!(
            narrowed
                .members
                .iter()
                .map(|row| row.member.as_str())
                .collect::<Vec<_>>(),
            vec![
                "FL2VA/transformer/model-00001-of-00002.safetensors",
                "FL2VA/transformer/model-00002-of-00002.safetensors",
                "FL2VA/transformer/model.safetensors.index.json",
            ],
            "the sibling directory's shards are NOT reached, though its index is the same \
             object"
        );
        assert!(narrowed.object_bytes() < whole.object_bytes());
        // A narrowing is a different selection, and says so.
        assert_ne!(narrowed.selection_sha256, whole.selection_sha256);
        // Selecting the same carriers twice is the same answer.
        assert_eq!(
            narrowed.selection_sha256,
            whole
                .select(&["FL2VA/transformer/model.safetensors.index.json".to_string()])
                .unwrap()
                .selection_sha256
        );

        // A shard is not a carrier: selecting one directly would silently drop the index
        // that gives it meaning, so it is refused rather than ignored.
        assert_eq!(
            whole
                .select(&["FL2VA/transformer/model-00001-of-00002.safetensors".to_string()])
                .unwrap_err()
                .code,
            Code::AMBIGUOUS_CLASSIFICATION
        );
        assert_eq!(
            whole
                .select(&["absent/model.safetensors".to_string()])
                .unwrap_err()
                .code,
            Code::MISSING_FIELD
        );
        assert_eq!(whole.select(&[]).unwrap_err().code, Code::MISSING_FIELD);
        handle.join().unwrap();
    }

    /// An index naming a shard the revision does not contain is a broken selection. The
    /// Go resolver refuses it; so does this one, rather than inventing a member.
    #[test]
    fn resolve_refuses_an_index_naming_an_absent_shard() {
        let revision = "cd".repeat(20);
        let index_body =
            br#"{"weight_map":{"a.weight":"model-00009-of-00009.safetensors"}}"#.to_vec();
        let tree = format!(
            r#"[{{"type":"file","path":"transformer/model.safetensors.index.json","size":{},"oid":"gitsha1"}}]"#,
            index_body.len()
        );
        let revision_for_routes = revision.clone();
        let (base, _, handle) = origin(2, move |request| {
            let r = &revision_for_routes;
            if request.path == format!("/api/models/org/repo/tree/{r}?recursive=true") {
                ok_body(tree.as_bytes(), "application/json")
            } else {
                ok_body(&index_body, "application/json")
            }
        });
        let uri = SourceUri::parse(&format!("hf://org/repo@{revision}")).unwrap();
        let endpoints = Endpoints {
            huggingface: base,
            allow_local: true,
            ..Endpoints::default()
        };
        let refusal = resolve(&uri, &endpoints, &Anonymous, Deadline::none()).unwrap_err();
        assert_eq!(refusal.code, Code::MISSING_FIELD);
        assert!(refusal.detail.contains("model-00009-of-00009.safetensors"));
        handle.join().unwrap();
    }

    /// tfs#298: a repository of LoRAs holds one whose name no selection can spell. A caller
    /// that names its carrier resolves and hears which file was skipped; a whole-repository
    /// resolve, which that file would silently shrink, still refuses.
    #[test]
    fn a_named_selection_skips_a_sibling_no_selection_can_name() {
        let revision = "ce".repeat(20);
        let odd = "living/Clothes-H3(QQQQ4413).safetensors";
        let lfs = |path: &str, seed: char| {
            let oid = seed.to_string().repeat(64);
            format!(
                r#"{{"type":"file","path":"{path}","size":64,"lfs":{{"oid":"{oid}","size":64}}}}"#
            )
        };
        let tree = format!(
            "[{},{}]",
            lfs("retro/GOLDENBOY_H3_V1.safetensors", 'a'),
            lfs(odd, 'b')
        );
        let (base, _, handle) = origin(2, move |_| ok_body(tree.as_bytes(), "application/json"));
        let uri = SourceUri::parse(&format!("hf://org/repo@{revision}")).unwrap();
        let endpoints = Endpoints {
            huggingface: base,
            allow_local: true,
            ..Endpoints::default()
        };
        let (resolution, skipped) =
            resolve_named(&uri, &endpoints, &Anonymous, Deadline::none()).unwrap();
        assert_eq!(skipped, vec![odd.to_string()]);
        let names: Vec<&str> = resolution
            .members
            .iter()
            .map(|m| m.member.as_str())
            .collect();
        assert_eq!(names, vec!["retro/GOLDENBOY_H3_V1.safetensors"]);
        let selected = resolution
            .select(&["retro/GOLDENBOY_H3_V1.safetensors".to_string()])
            .unwrap();
        assert_eq!(selected.members.len(), 1);

        let refusal = resolve(&uri, &endpoints, &Anonymous, Deadline::none()).unwrap_err();
        assert_eq!(refusal.code, Code::KEY_GRAMMAR);
        assert!(refusal.detail.contains("QQQQ4413"), "{refusal}");
        handle.join().unwrap();
    }

    /// An index may not name its way out of the repository.
    #[test]
    fn resolve_refuses_an_index_escaping_the_repository() {
        let revision = "ef".repeat(20);
        let index_body = br#"{"weight_map":{"a.weight":"../../../etc/passwd"}}"#.to_vec();
        let tree = format!(
            r#"[{{"type":"file","path":"transformer/model.safetensors.index.json","size":{},"oid":"gitsha1"}}]"#,
            index_body.len()
        );
        let revision_for_routes = revision.clone();
        let (base, _, handle) = origin(2, move |request| {
            let r = &revision_for_routes;
            if request.path == format!("/api/models/org/repo/tree/{r}?recursive=true") {
                ok_body(tree.as_bytes(), "application/json")
            } else {
                ok_body(&index_body, "application/json")
            }
        });
        let uri = SourceUri::parse(&format!("hf://org/repo@{revision}")).unwrap();
        let endpoints = Endpoints {
            huggingface: base,
            allow_local: true,
            ..Endpoints::default()
        };
        let refusal = resolve(&uri, &endpoints, &Anonymous, Deadline::none()).unwrap_err();
        assert_eq!(refusal.code, Code::PATH_ILLEGAL);
        handle.join().unwrap();
    }

    /// Civitai: one version is a bag of files. The primary SafeTensors is the model and the
    /// version's other declared SafeTensors are companions a reviewed profile may compose it
    /// with. Digests are declared but lengths only approximate kilobytes, so each exact
    /// length comes from a ranged probe through the same fence.
    #[test]
    fn resolve_takes_the_primary_civitai_safetensors_and_its_companions() {
        let header =
            br#"{"model.diffusion_model.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut body = (header.len() as u64).to_le_bytes().to_vec();
        body.extend_from_slice(header);
        body.extend_from_slice(&1f32.to_le_bytes());
        let object = ObjectRef::of(&body);
        let listing = format!(
            r#"{{"id":777,"files":[
               {{"id":92696,"name":"model.safetensors","primary":true,"sizeKB":35.15,"hashes":{{"SHA256":"{}"}},"metadata":{{"format":"SafeTensor"}}}},
               {{"id":92697,"name":"extra.safetensors","primary":false,"sizeKB":1.0,"hashes":{{"SHA256":"{}"}}}}
            ]}}"#,
            object.sha256.to_ascii_uppercase(),
            "11".repeat(32)
        );
        let length = object.length;
        let (base, _, handle) = origin(3, move |request| {
            if request.path == "/api/v1/model-versions/777" {
                ok_body(listing.as_bytes(), "application/json")
            } else {
                format!(
                    "HTTP/1.1 206 Partial Content\r\nContent-Length: 1\r\nContent-Range: bytes 0-0/{length}\r\nETag: \"tag\"\r\nConnection: close\r\n\r\n\0"
                )
                .into_bytes()
            }
        });
        let uri = SourceUri::parse("civitai://777").unwrap();
        let endpoints = Endpoints {
            civitai: base,
            allow_local: true,
            ..Endpoints::default()
        };
        let resolved = resolve(&uri, &endpoints, &Anonymous, Deadline::none()).unwrap();
        let rows: Vec<(&str, bool)> = resolved
            .members
            .iter()
            .map(|row| (row.member.as_str(), row.companion))
            .collect();
        assert_eq!(
            rows,
            [
                ("civitai/files/92696", false),
                ("civitai/files/92697", true)
            ]
        );
        let resolved = resolved.select(&["civitai/files/92696".into()]).unwrap();
        assert_eq!(resolved.members[0].member, "civitai/files/92696");
        assert_eq!(resolved.members[0].object, object);
        assert_eq!(resolved.members[0].provenance, Provenance::Declared);
        assert!(resolved.members[0].carrier);
        assert_eq!(resolved.canonical, "civitai://777");
        handle.join().unwrap();

        // The real source tree must keep the carrier visible to conversion despite its
        // lack of a filename extension, under the member the reviewed profile selects.
        let root = temporary("civitai-member");
        let store = Store::init(&root).unwrap();
        store
            .put_stream(&mut body.as_slice(), Some(&object), &Fault::default())
            .unwrap();
        let tree = content_manifest(&resolved).unwrap();
        let owner = ObjectRef::of(b"civitai-member-owner").id();
        crate::source_artifact::create(&store, &owner, &tree).unwrap();
        let (_, roster, landed) =
            crate::source_artifact::conversion_roster(&store, &owner).unwrap();
        assert_eq!(roster.len(), 1);
        assert_eq!(roster[0].member, "civitai/files/92696");
        assert_eq!(roster[0].header, body[..8 + header.len()]);
        assert_eq!(landed, vec!["civitai/files/92696"]);
        let registry = crate::ingest::fingerprint::FingerprintRegistry::parse(
            crate::ingest::source::BUILTIN_REGISTRY_BYTES,
        )
        .unwrap();
        let profile = registry
            .source_profiles
            .iter()
            .find(|profile| profile.name == "civitai/101055/128078/single-file-fp16")
            .unwrap();
        assert!(
            profile
                .components
                .iter()
                .all(|component| component.source_member.as_deref()
                    == Some(roster[0].member.as_str()))
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn civitai_refuses_missing_invalid_and_duplicate_file_ids() {
        for id in ["null", "0", "-1", "1.5", "9007199254740992", "\"92696\""] {
            let listing = format!(r#"{{"files":[{{"id":{id},"name":"model.safetensors"}}]}}"#);
            let (base, _, handle) =
                origin(1, move |_| ok_body(listing.as_bytes(), "application/json"));
            let endpoints = Endpoints {
                civitai: base,
                allow_local: true,
                ..Endpoints::default()
            };
            let refusal = list(
                &SourceUri::Civitai { version: 777 },
                &endpoints,
                &Anonymous,
                Deadline::none(),
            )
            .unwrap_err();
            let expected = if id == "9007199254740992" {
                Code::NUMBER_RANGE // The shared JSON decoder refuses before provider parsing.
            } else {
                Code::KEY_GRAMMAR
            };
            assert_eq!(refusal.code, expected, "{id}");
            handle.join().unwrap();
        }
        for (listing, code) in [
            (
                r#"{"files":[{"name":"missing-id.safetensors"}]}"#,
                Code::KEY_GRAMMAR,
            ),
            (
                r#"{"files":[{"id":92696,"name":"a.safetensors","hashes":{"SHA256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}},{"id":92696,"name":"b.safetensors","hashes":{"SHA256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}}]}"#,
                Code::DUPLICATE_KEY,
            ),
        ] {
            let (base, _, handle) =
                origin(1, move |_| ok_body(listing.as_bytes(), "application/json"));
            let endpoints = Endpoints {
                civitai: base,
                allow_local: true,
                ..Endpoints::default()
            };
            let refusal = list(
                &SourceUri::Civitai { version: 777 },
                &endpoints,
                &Anonymous,
                Deadline::none(),
            )
            .unwrap_err();
            assert_eq!(refusal.code, code);
            handle.join().unwrap();
        }
        // A provider that lists one file twice with the same facts lists it once.
        let listing = r#"{"files":[{"id":92696,"name":"a.safetensors"},{"id":92696,"name":"a.safetensors"}]}"#;
        let (base, _, handle) = origin(1, move |_| ok_body(listing.as_bytes(), "application/json"));
        let endpoints = Endpoints {
            civitai: base,
            allow_local: true,
            ..Endpoints::default()
        };
        let members = list(
            &SourceUri::Civitai { version: 777 },
            &endpoints,
            &Anonymous,
            Deadline::none(),
        )
        .unwrap();
        assert_eq!(members.len(), 1);
        handle.join().unwrap();
    }

    /// The credential is scoped to the host that holds it. A relative redirect stays on
    /// that host and keeps the bearer; the fence re-checks it either way.
    #[test]
    fn resolve_presents_the_credential_only_to_the_scoped_host() {
        let revision = "12".repeat(20);
        let index_body =
            br#"{"weight_map":{"a.weight":"model-00001-of-00001.safetensors"}}"#.to_vec();
        let shard: Vec<u8> = (0u32..1_000).flat_map(|i| i.to_le_bytes()).collect();
        let shard_object = ObjectRef::of(&shard);
        let tree = format!(
            r#"[{{"type":"file","path":"model.safetensors.index.json","size":{},"oid":"gitsha1"}},
                {{"type":"file","path":"model-00001-of-00001.safetensors","size":{},"oid":"gitsha1","lfs":{{"oid":"{}","size":{}}}}}]"#,
            index_body.len(),
            shard_object.length,
            shard_object.sha256,
            shard_object.length
        );
        let revision_for_routes = revision.clone();
        let (base, seen, handle) = origin(2, move |request| {
            let r = &revision_for_routes;
            if request.path == format!("/api/models/org/repo/tree/{r}?recursive=true") {
                ok_body(tree.as_bytes(), "application/json")
            } else {
                ok_body(&index_body, "application/json")
            }
        });
        let uri = SourceUri::parse(&format!("hf://org/repo@{revision}")).unwrap();
        let endpoints = Endpoints {
            huggingface: base,
            allow_local: true,
            ..Endpoints::default()
        };
        // Scoped to a host this resolution never speaks to: nothing may present it.
        let elsewhere = HostToken {
            hosts: vec!["cdn.example.invalid".into()],
            token: "hf_secret".into(),
        };
        let resolved = resolve(&uri, &endpoints, &elsewhere, Deadline::none()).unwrap();
        assert_eq!(resolved.members.len(), 2);
        assert!(
            seen.lock()
                .unwrap()
                .iter()
                .all(|(_, authorization)| authorization.is_none()),
            "a token scoped to another host must never be presented"
        );
        handle.join().unwrap();
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
            r#"{{"id":777,"files":[{{"id":92696,"name":"model.safetensors","sizeKB":273.4,"hashes":{{"SHA256":"{}"}}}}]}}"#,
            object.sha256.to_ascii_uppercase()
        );
        let payload = body.clone();
        let length = object.length;
        // pull #1: listing + probe + body; pull #2: listing + probe (object HELD).
        let (base, _, handle) = origin(5, move |request| {
            if request.path == "/api/v1/model-versions/777" {
                ok_body(listing.as_bytes(), "application/json")
            } else if request.path == "/api/download/models/777?fileId=92696" {
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

    #[test]
    fn explicit_files_resolve_only_named_bytes_and_retain_across_restart() {
        let config = b"{\"model_type\":\"test\"}".to_vec();
        let tokenizer = b"these are tokenizer bytes".to_vec();
        let expected = vec![ObjectRef::of(&config), ObjectRef::of(&tokenizer)];
        let listing = format!(
            r#"[
            {{"type":"file","path":"model.safetensors","size":100000000000,"lfs":{{"oid":"{}","size":100000000000}}}},
            {{"type":"file","path":"config.json","size":{}}},
            {{"type":"file","path":"tokenizer files/tokenizer.json","size":{}}},
            {{"type":"file","path":"unused.py","size":99}}
        ]"#,
            "*".repeat(64),
            config.len(),
            tokenizer.len()
        );
        // Two order-independent resolutions each read list + two small bodies;
        // materialization reads the bodies once, and restart/warm admission reads none.
        let (base, seen, handle) = origin(8, move |request| {
            if request.path.contains("/tree/") {
                ok_body(listing.as_bytes(), "application/json")
            } else if request.path.ends_with("/config.json") {
                ok_body(&config, "application/json")
            } else if request.path.ends_with("/tokenizer.json") {
                ok_body(&tokenizer, "application/json")
            } else {
                panic!("unselected file requested: {}", request.path);
            }
        });
        let uri = SourceUri::parse(&format!("hf://org/repo@{}", "ab".repeat(20))).unwrap();
        let endpoints = Endpoints {
            huggingface: base,
            allow_local: true,
            ..Endpoints::default()
        };
        let mut names = vec![
            "config.json".into(),
            "tokenizer files/tokenizer.json".into(),
        ];
        let selected =
            resolve_files(&uri, &names, &endpoints, &Anonymous, Deadline::none()).unwrap();
        names.reverse();
        assert_eq!(
            selected,
            resolve_files(&uri, &names, &endpoints, &Anonymous, Deadline::none()).unwrap()
        );
        assert_eq!(
            selected
                .members
                .iter()
                .map(|m| m.object.clone())
                .collect::<Vec<_>>(),
            expected
        );
        assert!(selected
            .members
            .iter()
            .all(|m| m.provenance == Provenance::Computed && !m.carrier && m.requires.is_empty()));
        let root = temporary("selected-files");
        let store = Store::init(&root).unwrap();
        let producer = ObjectRef::of(b"files-producer").id();
        let recipient = ObjectRef::of(b"files-recipient").id();
        let (cold, transfers) = materialize_selected(
            &store,
            &producer,
            &selected,
            &endpoints.policy(&uri),
            &Anonymous,
            Deadline::none(),
            &|_, _| {},
        )
        .unwrap();
        assert_eq!(
            transfers.iter().map(|row| row.transferred).sum::<u64>(),
            selected.member_bytes()
        );
        crate::source_artifact::retain(&store, &producer, &recipient).unwrap();
        drop(store);
        let store = Store::open(&root).unwrap();
        let (warm, transfers) = materialize_selected(
            &store,
            &producer,
            &selected,
            &endpoints.policy(&uri),
            &Anonymous,
            Deadline::none(),
            &|_, _| {},
        )
        .unwrap();
        assert_eq!(warm, cold);
        assert!(transfers.iter().all(|row| row.held && row.transferred == 0));
        crate::source_artifact::release(&store, &producer).unwrap();
        crate::gc::collect(&root, false).unwrap();
        let retained = crate::source_artifact::read(&store, &recipient)
            .unwrap()
            .unwrap();
        assert!(retained.complete && !retained.released);
        for object in &expected {
            assert!(store.record_valid(&object.sha256).is_ok());
        }
        crate::source_artifact::release(&store, &recipient).unwrap();
        crate::gc::collect(&root, false).unwrap();
        for object in expected {
            assert!(!store.contains(&object.sha256));
        }
        handle.join().unwrap();
        assert_eq!(seen.lock().unwrap().len(), 8);
        assert!(seen
            .lock()
            .unwrap()
            .iter()
            .any(|(path, _)| path.contains("tokenizer%20files/")));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn explicit_files_refuse_missing_and_over_budget_before_fetching_bodies() {
        for (names, listing, code) in [
            (
                vec!["absent.json".into(), "config.json".into()],
                r#"[{"type":"file","path":"config.json","size":2}]"#.to_string(),
                Code::MISSING_FIELD,
            ),
            (
                vec!["config.json".into()],
                format!(
                    r#"[{{"type":"file","path":"config.json","size":{}}}]"#,
                    FILE_SELECTION_MAX_BYTES + 1
                ),
                Code::SIZE_CAP,
            ),
        ] {
            let (base, _, handle) =
                origin(1, move |_| ok_body(listing.as_bytes(), "application/json"));
            let uri = SourceUri::parse(&format!("hf://org/repo@{}", "ab".repeat(20))).unwrap();
            let endpoints = Endpoints {
                huggingface: base,
                allow_local: true,
                ..Endpoints::default()
            };
            assert_eq!(
                resolve_files(&uri, &names, &endpoints, &Anonymous, Deadline::none())
                    .unwrap_err()
                    .code,
                code
            );
            handle.join().unwrap();
        }
        let uri = SourceUri::parse(&format!("hf://org/repo@{}", "ab".repeat(20))).unwrap();
        for names in [
            vec!["../config.json".into()],
            vec!["*.json".into()],
            vec!["/config.json".into()],
        ] {
            assert_eq!(
                resolve_files(
                    &uri,
                    &names,
                    &Endpoints::default(),
                    &Anonymous,
                    Deadline::none()
                )
                .unwrap_err()
                .code,
                Code::PATH_ILLEGAL
            );
        }
    }

    #[test]
    fn a_folder_selects_its_ordinary_files_and_never_a_carrier() {
        let listing = br#"[
            {"type":"file","path":"tokenizer/vocab.json","size":2},
            {"type":"file","path":"tokenizer/merges.txt","size":2},
            {"type":"file","path":"tokenizer/model.safetensors","size":2},
            {"type":"file","path":"tokenizer_2/vocab.json","size":2},
            {"type":"file","path":"unet/config.json","size":2}
        ]"#;
        let (base, seen, handle) = origin(5, move |request| {
            if request.path.contains("/tree/") {
                ok_body(listing, "application/json")
            } else {
                ok_body(b"{}", "application/json")
            }
        });
        let uri = SourceUri::parse(&format!("hf://org/repo@{}", "ab".repeat(20))).unwrap();
        let endpoints = Endpoints {
            huggingface: base,
            allow_local: true,
            ..Endpoints::default()
        };
        let selected = resolve_files(
            &uri,
            &["tokenizer/".into(), "unet/config.json".into()],
            &endpoints,
            &Anonymous,
            Deadline::none(),
        )
        .unwrap();
        let members: Vec<&str> = selected.members.iter().map(|m| m.member.as_str()).collect();
        assert_eq!(
            members,
            [
                "tokenizer/merges.txt",
                "tokenizer/vocab.json",
                "unet/config.json"
            ]
        );
        assert_eq!(
            resolve_files(
                &uri,
                &["vae/".into()],
                &endpoints,
                &Anonymous,
                Deadline::none()
            )
            .unwrap_err()
            .code,
            Code::MISSING_FIELD
        );
        handle.join().unwrap();
        assert!(!seen
            .lock()
            .unwrap()
            .iter()
            .any(|(path, _)| path.ends_with(".safetensors")));
    }

    #[test]
    fn explicit_files_refuse_a_changed_listing_length() {
        let (base, _, handle) = origin(2, move |request| {
            if request.path.contains("/tree/") {
                ok_body(
                    br#"[{"type":"file","path":"config.json","size":2}]"#,
                    "application/json",
                )
            } else {
                ok_body(b"changed", "application/json")
            }
        });
        let uri = SourceUri::parse(&format!("hf://org/repo@{}", "ab".repeat(20))).unwrap();
        let endpoints = Endpoints {
            huggingface: base,
            allow_local: true,
            ..Endpoints::default()
        };
        assert_eq!(
            resolve_files(
                &uri,
                &["config.json".into()],
                &endpoints,
                &Anonymous,
                Deadline::none()
            )
            .unwrap_err()
            .code,
            Code::LENGTH_MISMATCH
        );
        handle.join().unwrap();
    }

    #[test]
    fn explicit_files_changed_after_resolution_never_finish_a_native_root() {
        let replies = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (base, _, handle) = origin(3, move |request| {
            if request.path.contains("/tree/") {
                ok_body(
                    br#"[{"type":"file","path":"config.json","size":2}]"#,
                    "application/json",
                )
            } else {
                let attempt = replies.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                ok_body(if attempt == 0 { b"AA" } else { b"BB" }, "application/json")
            }
        });
        let uri = SourceUri::parse(&format!("hf://org/repo@{}", "ab".repeat(20))).unwrap();
        let endpoints = Endpoints {
            huggingface: base,
            allow_local: true,
            ..Endpoints::default()
        };
        let selected = resolve_files(
            &uri,
            &["config.json".into()],
            &endpoints,
            &Anonymous,
            Deadline::none(),
        )
        .unwrap();
        let root = temporary("changed-files");
        let store = Store::init(&root).unwrap();
        let producer = ObjectRef::of(b"changed-files-producer").id();
        let error = materialize_selected(
            &store,
            &producer,
            &selected,
            &endpoints.policy(&uri),
            &Anonymous,
            Deadline::none(),
            &|_, _| {},
        )
        .unwrap_err();
        assert_eq!(error.code, Code::OBJECT_ID_MISMATCH);
        assert!(!store.contains(&selected.members[0].object.sha256));
        assert!(
            !crate::source_artifact::read(&store, &producer)
                .unwrap()
                .unwrap()
                .complete
        );
        handle.join().unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn explicit_files_can_retain_an_empty_file() {
        let (base, _, handle) = origin(2, move |request| {
            if request.path.contains("/tree/") {
                ok_body(
                    br#"[{"type":"file","path":"empty.py","size":0}]"#,
                    "application/json",
                )
            } else {
                ok_body(b"", "text/plain")
            }
        });
        let uri = SourceUri::parse(&format!("hf://org/repo@{}", "ab".repeat(20))).unwrap();
        let endpoints = Endpoints {
            huggingface: base,
            allow_local: true,
            ..Endpoints::default()
        };
        let selected = resolve_files(
            &uri,
            &["empty.py".into()],
            &endpoints,
            &Anonymous,
            Deadline::none(),
        )
        .unwrap();
        assert_eq!(selected.members[0].object, ObjectRef::of(b""));
        let root = temporary("empty-files");
        let store = Store::init(&root).unwrap();
        let producer = ObjectRef::of(b"empty-files-producer").id();
        let (retained, transfers) = materialize_selected(
            &store,
            &producer,
            &selected,
            &endpoints.policy(&uri),
            &Anonymous,
            Deadline::none(),
            &|_, _| {},
        )
        .unwrap();
        assert!(retained.complete && !retained.released);
        assert_eq!(transfers[0].transferred, 0);
        assert!(store
            .record_valid(&selected.members[0].object.sha256)
            .is_ok());
        handle.join().unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }
}

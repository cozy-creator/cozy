//! `tfs fetch` (tfs-003 row 220) — the pull plane's document half, driven from the CLI.
//!
//! Core links no HTTP and holds no credential, by fence, so this emits what a transport
//! adapter must move and CONSUMES the bytes it moved. The adapter — cozy-runtime's, per
//! boundaries.md §9 — holds the URL and the token; `fetch admit` reads the body on stdin.
//! That is the whole seam: `curl "$url" | tfs fetch admit <store> <sha256> <length>` and a
//! worker's Python transport calling `Store.put_reader` are the SAME door, because both end
//! inside `Store::put_stream`.

use std::fs;
use std::io;
use std::path::Path;
use std::process::ExitCode;

use tensorfs_core::fetch::{self, DeliveryGrant, FetchPlan};
use tensorfs_core::ids::{Doc, ObjectRef};
use tensorfs_core::manifest::Manifest;
use tensorfs_core::repo_cache::RepoObjectCache;
use tensorfs_core::store::Store;

use crate::{bail, flag, Flags};

macro_rules! ok {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => return bail(e),
        }
    };
}

fn rd(p: &Path) -> Result<Vec<u8>, ExitCode> {
    fs::read(p).map_err(|e| {
        eprintln!("REFUSED IO_FAILED: read {}: {e}", p.display());
        ExitCode::FAILURE
    })
}

macro_rules! read {
    ($p:expr) => {
        match rd($p) {
            Ok(b) => b,
            Err(c) => return c,
        }
    };
}

fn emit_bytes(bytes: &[u8], flags: &Flags) {
    if let Some(p) = flag(flags, "out") {
        if let Err(e) = fs::write(p, bytes) {
            eprintln!("REFUSED IO_FAILED: write {p}: {e}");
        }
    }
}

/// PLAN FIRST, against a real store. The declaration is the remote catalog's claim about
/// what the checkpoint reaches; the split into held/wanted is this machine's own answer, and
/// it is where idempotency lives — an object already installed and vouched for is HELD and
/// no grant can be minted for it.
///
/// The manifest bytes and the sorted `manifest walk --refs` JSONL are the declaration. No
/// wrapper document duplicates them.
pub fn cmd_plan(root: &Path, manifest_p: &Path, session: &str, flags: &Flags) -> ExitCode {
    let store = ok!(Store::open(root));
    let manifest_bytes = read!(manifest_p);
    let _manifest = ok!(Manifest::parse(&manifest_bytes));
    let manifest = ObjectRef::of(&manifest_bytes);
    let Some(refs_path) = flag(flags, "refs") else {
        eprintln!("--refs <jsonl> is required");
        return ExitCode::from(2);
    };
    let refs = ok!(fetch::parse_refs_jsonl(&read!(Path::new(refs_path))));
    let (plan, why) = ok!(FetchPlan::of(&store, session, &manifest, &refs));
    emit_bytes(&plan.canonical_bytes(), flags);
    println!(
        "plan         {}",
        tensorfs_core::ids::object_id(&plan.canonical_bytes())
    );
    println!("session      {}", plan.session);
    println!("manifest     {}", manifest.id());
    println!(
        "held         {} object(s), {} B — NOT fetched",
        plan.held.len(),
        plan.held_bytes()
    );
    println!(
        "wanted       {} object(s), {} B",
        plan.wanted.len(),
        plan.wanted_bytes()
    );
    if crate::flag_on(flags, "why") {
        for (o, p) in &why {
            println!("             {} {}", p.as_str(), o.id());
        }
    }
    ExitCode::SUCCESS
}

/// THE DOOR. Bytes arrive on stdin and leave as an installed object or as nothing.
///
/// There is no `--force`, no `--skip-verify` and no path that writes at an id it did not
/// compute: `put_stream` hashes what it reads and refuses at the first disagreement with the
/// grant, having already removed the partial temp file.
pub fn cmd_admit(root: &Path, plan_p: &Path, id: &str, flags: &Flags) -> ExitCode {
    let store = ok!(Store::open(root));
    let plan = ok!(FetchPlan::parse(&read!(plan_p)));
    let hex = id.trim_start_matches("sha256:");
    let length = match plan.wanted.iter().find(|o| o.sha256 == hex) {
        Some(o) => o.length,
        // Not in the wanted set: mint anyway so `mint` produces the exact typed refusal
        // (already-held vs never-declared) rather than this function inventing one.
        None => crate::flag_num(flags, "length", 0u64),
    };
    let object = ObjectRef {
        sha256: hex.to_string(),
        length,
    };
    let grant = ok!(DeliveryGrant::mint(&plan, &object));
    let mut stdin = io::stdin().lock();
    let admitted = ok!(grant.admit(&store, &mut stdin));
    println!("admitted     {}", admitted.object.id());
    println!("length       {} B", admitted.object.length);
    println!("key          {}", grant.key());
    println!(
        "writer       {}",
        if admitted.first_writer {
            "this process won the no-clobber admission"
        } else {
            "already installed by a concurrent puller; bytes verified either way"
        }
    );
    ExitCode::SUCCESS
}

/// The PROOF. Every object the plan declared is installed and this store's own record
/// vouches for it. A transfer's success report is not residency and does not appear here.
pub fn cmd_complete(root: &Path, plan_p: &Path) -> ExitCode {
    let store = ok!(Store::open(root));
    let plan = ok!(FetchPlan::parse(&read!(plan_p)));
    ok!(plan.complete(&store));
    match &plan.manifest {
        Some(manifest) => println!("complete     {}", manifest.id()),
        None => println!(
            "complete     {} object(s), no manifest anchor",
            plan.held.len() + plan.wanted.len()
        ),
    }
    println!(
        "resident     {} object(s), {} B, every one verifiably local",
        plan.held.len() + plan.wanted.len(),
        plan.declared_bytes()
    );
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------- the one transport (tfs-050)

use tensorfs_core::providers::{self, Endpoints, SourceUri};
use tensorfs_core::transport::{
    self, credential_from_spec, Anonymous, Deadline, PullRequest, SourcePolicy,
};

fn policy_of(flags: &Flags) -> SourcePolicy {
    SourcePolicy {
        allowed_hosts: crate::flag(flags, "allow-hosts")
            .map(|hosts| {
                hosts
                    .split(',')
                    .map(str::trim)
                    .filter(|h| !h.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default(),
        allow_local: crate::flag_on(flags, "allow-local"),
        max_redirects: crate::flag_num(flags, "max-redirects", 0u32),
    }
}

fn deadline_of(flags: &Flags) -> Deadline {
    Deadline::after_seconds(crate::flag(flags, "timeout").and_then(|t| t.parse::<f64>().ok()))
}

/// The ONE credential spelling, reaching the child explicitly (tfs-050): `--credential-file
/// <path>` or the deliberately-set `TFS_CREDENTIAL` — never argv (world-readable in /proc),
/// never the inherited environment at large. The value is `bearer <token>`, scoped to
/// exactly the named hosts, or empty for anonymous — which is what a rented pod passes,
/// the hub's closure and presign routes taking no credential at all.
fn credential_of(
    flags: &Flags,
    hosts: Vec<String>,
) -> Result<tensorfs_core::transport::ScopedHeaders, ExitCode> {
    let spec = match flag(flags, "credential-file") {
        Some(path) => match fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) => {
                eprintln!("REFUSED IO_FAILED: read {path}: {error}");
                return Err(ExitCode::FAILURE);
            }
        },
        None => std::env::var("TFS_CREDENTIAL").unwrap_or_default(),
    };
    match credential_from_spec(spec.trim(), hosts) {
        Ok(credential) => Ok(credential),
        Err(refusal) => {
            eprintln!("REFUSED {refusal}");
            Err(ExitCode::FAILURE)
        }
    }
}

/// `tfs fetch <root> <ref>` — closure, plan against the REAL store, the N-way walk
/// presigning `wanted` as it reaches it, the completion proof. A warm store asks one
/// question and moves nothing.
///
/// `--repo-cache <root>` names an optional mounted immutable object cache to consult
/// between the local Store and the origin, and to publish origin bytes onward into. It is
/// a PATH and never a mode: an absent flag is a pull with no cache, and a cache that is
/// missing, cold, corrupt, read-only or unmounted degrades to exactly that pull.
pub fn cmd_pull(root: &Path, refspec: &str, flags: &Flags) -> ExitCode {
    let store = ok!(Store::open(root));
    let Some(base) = flag(flags, "hub") else {
        eprintln!("--hub <base-url> is required");
        return ExitCode::from(2);
    };
    let hub_host = match transport::base_host(base) {
        Ok(host) => host,
        Err(refusal) => return bail(refusal),
    };
    let credential = match credential_of(flags, vec![hub_host]) {
        Ok(credential) => credential,
        Err(code) => return code,
    };
    let policy = policy_of(flags);
    let mut request = PullRequest::new(&store, base, refspec, &credential, &policy);
    request.lane = flag(flags, "lane").unwrap_or("");
    request.session = flag(flags, "session").unwrap_or("");
    request.streams = crate::flag_num(flags, "streams", transport::STREAMS);
    request.deadline = deadline_of(flags);
    let cache = flag(flags, "repo-cache").map(RepoObjectCache::new);
    request.cache = cache.as_ref();
    let report = ok!(transport::pull(&request));
    println!(
        "model        {}@{} lane {}",
        report.model, report.release, report.lane
    );
    println!("manifest     {}", report.manifest);
    println!(
        "held         {} object(s), {} B — NOT fetched",
        report.held, report.bytes_held
    );
    println!(
        "cached       {} object(s), {} B — from the mounted cache, NOT the origin",
        report.cached, report.bytes_cached
    );
    println!(
        "fetched      {} object(s), {} B, {} stream(s)",
        report.fetched, report.bytes_moved, report.streams
    );
    println!("complete     every declared object verifiably local");
    ExitCode::SUCCESS
}

/// `tfs fetch plan-objects <root> <session> --refs <jsonl>` — the HELD/WANTED split over a
/// bare object set (th-124: raw foreign-source members have no Manifest until preparation
/// authors one).
pub fn cmd_plan_objects(root: &Path, session: &str, flags: &Flags) -> ExitCode {
    let store = ok!(Store::open(root));
    let Some(refs_path) = flag(flags, "refs") else {
        eprintln!("--refs <jsonl> is required");
        return ExitCode::from(2);
    };
    let refs = ok!(fetch::parse_refs_jsonl(&read!(Path::new(refs_path))));
    let (plan, why) = ok!(FetchPlan::of_objects(&store, session, &refs));
    emit_bytes(&plan.canonical_bytes(), flags);
    println!(
        "plan         {}",
        tensorfs_core::ids::object_id(&plan.canonical_bytes())
    );
    println!("session      {}", plan.session);
    println!(
        "held         {} object(s), {} B — NOT fetched",
        plan.held.len(),
        plan.held_bytes()
    );
    println!(
        "wanted       {} object(s), {} B",
        plan.wanted.len(),
        plan.wanted_bytes()
    );
    if crate::flag_on(flags, "why") {
        for (o, p) in &why {
            println!("             {} {}", p.as_str(), o.id());
        }
    }
    ExitCode::SUCCESS
}

/// `tfs fetch url <root> <sha256> <length> <url|->` — plan-of-one, then the one transport
/// into `put_stream`. A held object answers `held` and moves nothing. A url of `-` reads
/// the real URL as one line on stdin: a capability URL can embed a signature, and argv is
/// world-readable in /proc.
pub fn cmd_url(root: &Path, id: &str, length: &str, url: &str, flags: &Flags) -> ExitCode {
    use std::io::BufRead;
    let store = ok!(Store::open(root));
    let stdin_url;
    let url = if url == "-" {
        let mut line = String::new();
        if io::stdin().lock().read_line(&mut line).is_err() {
            eprintln!("REFUSED SOURCE_NOT_ALLOWED: could not read the URL from stdin");
            return ExitCode::FAILURE;
        }
        stdin_url = line.trim().to_string();
        stdin_url.as_str()
    } else {
        url
    };
    let sha256 = ok!(tensorfs_core::ids::hex64("fetch url object", id));
    let Ok(length) = length.parse::<u64>() else {
        eprintln!("length must be a byte count");
        return ExitCode::from(2);
    };
    let object = ObjectRef { sha256, length };
    let session = flag(flags, "session").unwrap_or("fetch-url");
    let (plan, _) = ok!(FetchPlan::of_objects(
        &store,
        session,
        std::slice::from_ref(&object)
    ));
    if plan.held.iter().any(|held| held.sha256 == object.sha256) {
        println!("held         {}", object.id());
        println!("transferred  0 B");
        return ExitCode::SUCCESS;
    }
    let grant = ok!(fetch::DeliveryGrant::mint(&plan, &object));
    let ledger = transport::Ledger::new();
    let fetched = ok!(transport::fetch_url(
        &store,
        &grant,
        url,
        &policy_of(flags),
        &Anonymous,
        deadline_of(flags),
        &ledger,
        None,
    ));
    println!("admitted     {}", fetched.admitted.object.id());
    println!("transferred  {} B", fetched.transferred);
    ExitCode::SUCCESS
}

/// The push destination, reaching the child exactly the way the credential does (tfs-050) —
/// never argv, which is world-readable in `/proc`, and a presigned PUT is a bearer secret.
/// A minted grant is a destination AND the headers its signature covers, so the two travel
/// as ONE document and there is no spelling that spends the URL without its conditions:
/// `--grant-file <path>` or one deliberately-set `TFS_PUSH_GRANT`, the URL on the first
/// line then one `name: value` per signed header. `--url <presigned|->` remains for a bare
/// destination that signs nothing, `-` reading it from stdin. Exactly one of the three.
fn grant_of(flags: &Flags) -> Result<transport::UploadGrant, ExitCode> {
    use std::io::BufRead;
    let file = flag(flags, "grant-file");
    let environment = std::env::var("TFS_PUSH_GRANT").unwrap_or_default();
    let url = flag(flags, "url");
    let given = [
        file.is_some(),
        !environment.trim().is_empty(),
        url.is_some(),
    ];
    if given.iter().filter(|named| **named).count() != 1 {
        eprintln!(
            "exactly one of --grant-file <path>, TFS_PUSH_GRANT, or --url <presigned|-> \
             names the destination"
        );
        return Err(ExitCode::from(2));
    }
    let document = match file {
        Some(path) => match fs::read_to_string(path) {
            Ok(text) => Some(text),
            Err(error) => {
                eprintln!("REFUSED IO_FAILED: read {path}: {error}");
                return Err(ExitCode::FAILURE);
            }
        },
        None if !environment.trim().is_empty() => Some(environment),
        None => None,
    };
    if let Some(document) = document {
        return match transport::UploadGrant::parse(&document) {
            Ok(grant) => Ok(grant),
            Err(refusal) => {
                eprintln!("REFUSED {refusal}");
                Err(ExitCode::FAILURE)
            }
        };
    }
    let url = url.unwrap_or_default();
    if url != "-" {
        return Ok(transport::UploadGrant::to(url));
    }
    let mut line = String::new();
    if io::stdin().lock().read_line(&mut line).is_err() {
        eprintln!("REFUSED SOURCE_NOT_ALLOWED: could not read the URL from stdin");
        return Err(ExitCode::FAILURE);
    }
    Ok(transport::UploadGrant::to(line.trim()))
}

/// `tfs push <root> <sha256> --grant-file <path> [--manifest]` — the upload half: one
/// verified store object to the one destination a grant names, under the exact headers
/// its signature covers, 412 a success under `If-None-Match: *`.
pub fn cmd_push(root: &Path, id: &str, flags: &Flags) -> ExitCode {
    let store = ok!(Store::open(root));
    let sha256 = ok!(tensorfs_core::ids::hex64("push object", id));
    let grant = match grant_of(flags) {
        Ok(grant) => grant,
        Err(code) => return code,
    };
    let ledger = transport::Ledger::new();
    let pushed = ok!(transport::push_object(
        &store,
        &sha256,
        crate::flag_on(flags, "manifest"),
        &grant,
        &policy_of(flags),
        &Anonymous,
        deadline_of(flags),
        &ledger,
        flag(flags, "content-type").unwrap_or(""),
    ));
    println!("pushed       {}", pushed.object.id());
    println!("status       HTTP {}", pushed.http_status);
    println!(
        "sent         {} B{}",
        pushed.bytes_sent,
        if pushed.landed_precondition {
            " (the immutable key already held these exact bytes)"
        } else {
            ""
        }
    );
    ExitCode::SUCCESS
}

fn source_setup(
    uri: &str,
    flags: &Flags,
) -> Result<
    (
        SourceUri,
        Endpoints,
        tensorfs_core::transport::ScopedHeaders,
    ),
    ExitCode,
> {
    let uri = match SourceUri::parse(uri) {
        Ok(parsed) => parsed,
        Err(refusal) => {
            eprintln!("REFUSED {refusal}");
            return Err(ExitCode::FAILURE);
        }
    };
    let mut endpoints = Endpoints {
        allow_local: crate::flag_on(flags, "allow-local"),
        ..Endpoints::default()
    };
    if let Some(base) = flag(flags, "api") {
        match uri {
            SourceUri::HuggingFace { .. } => endpoints.huggingface = base.to_string(),
            SourceUri::Civitai { .. } => endpoints.civitai = base.to_string(),
        }
    }
    // The credential is scoped to the provider's own API host: a resolve/download
    // redirect onto a delivery host deliberately presents nothing.
    let api_host = match tensorfs_core::transport::base_host(match uri {
        SourceUri::HuggingFace { .. } => &endpoints.huggingface,
        SourceUri::Civitai { .. } => &endpoints.civitai,
    }) {
        Ok(host) => host,
        Err(refusal) => {
            eprintln!("REFUSED {refusal}");
            return Err(ExitCode::FAILURE);
        }
    };
    let credential = credential_of(flags, vec![api_host])?;
    Ok((uri, endpoints, credential))
}

/// `tfs source list <uri>` — what one pinned foreign source reaches, as the provider
/// declares it (tfs-052). No store, no bytes moved.
pub fn cmd_source_list(uri: &str, flags: &Flags) -> ExitCode {
    let (uri, endpoints, credential) = match source_setup(uri, flags) {
        Ok(setup) => setup,
        Err(code) => return code,
    };
    let members = ok!(providers::list(
        &uri,
        &endpoints,
        &credential,
        deadline_of(flags)
    ));
    for member in &members {
        println!(
            "{}  {}  {}",
            match &member.sha256 {
                Some(sha256) => format!("sha256:{sha256}"),
                None => format!("{:>71}", "(provider declares no digest)"),
            },
            match member.length {
                Some(length) => format!("{length:>14} B"),
                None => format!("{:>16}", "?"),
            },
            member.member
        );
    }
    println!("members      {}", members.len());
    ExitCode::SUCCESS
}

/// `tfs source pull <root> <uri>` — fetch and ingest one pinned foreign source through
/// the plan/transport/put_stream path (tfs-052). Held members move nothing.
pub fn cmd_source_pull(root: &Path, uri: &str, flags: &Flags) -> ExitCode {
    let (uri, endpoints, credential) = match source_setup(uri, flags) {
        Ok(setup) => setup,
        Err(code) => return code,
    };
    let store = ok!(Store::open(root));
    let pulled = ok!(providers::pull(
        &store,
        &uri,
        &endpoints,
        &credential,
        deadline_of(flags),
        &mut |_, _| {},
    ));
    let mut transferred = 0u64;
    let mut held = 0usize;
    for row in &pulled {
        transferred += row.transferred;
        held += usize::from(row.held);
        println!(
            "{}  {}  {} B {}",
            if row.held { "held    " } else { "admitted" },
            row.object.id(),
            row.object.length,
            row.member
        );
    }
    println!("members      {}", pulled.len());
    println!("held         {held} member(s) — NOT fetched");
    println!("transferred  {transferred} B");
    ExitCode::SUCCESS
}

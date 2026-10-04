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
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tensorfs_core::canon;
use tensorfs_core::fetch::FetchPlan;
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::store::Store;
use tensorfs_core::transport::ObjectSource;


use crate::{bail, flag, Flags};

// ---------------------------------------------------------------- the one transport (tfs-050)

use tensorfs_core::transport::{
    self, credential_from_spec, Anonymous, Deadline, PullRequest, SourcePolicy,
};

pub(crate) fn policy_of(flags: &Flags) -> SourcePolicy {
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
        ..Default::default()
    }
}

fn deadline_of(flags: &Flags) -> Deadline {
    Deadline::after_seconds(crate::flag(flags, "timeout").and_then(|t| t.parse::<f64>().ok()))
}

/// The ONE credential spelling, reaching the child explicitly (tfs-050): `--credential-file
/// <path>` or the deliberately-set `TFS_CREDENTIAL` — never argv (world-readable in /proc),
/// never the inherited environment at large. The value is `bearer <token>` or
/// `worker <id> <token>`, scoped to exactly the named hosts, or empty for anonymous.
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
/// The mounted immutable object cache — consulted between the local Store and the origin,
/// and published onward into — is the STORE's, recorded by `tfs store ensure --repo-cache`
/// and read off the handle this opens. There is no flag: a fetch does not get to name a
/// cache, because the cache is a fact about the deployment and not about this pull. An
/// unbound Store is a pull with no cache, and a cache that is missing, cold, corrupt,
/// read-only or unmounted degrades to exactly that pull.
pub fn cmd_pull(root: &Path, refspec: &str, flags: &Flags) -> ExitCode {
    // Before the open, which is where an unclean shutdown's recovery runs.
    let before = tensorfs_core::stats::snapshot();
    let store = ok!(Store::open(root));
    let Some(base) = flag(flags, "hub") else {
        eprintln!("--hub <base-url> is required");
        return ExitCode::from(2);
    };
    let hub_host = ok!(transport::base_host(base));
    let credential = match credential_of(flags, vec![hub_host]) {
        Ok(credential) => credential,
        Err(code) => return code,
    };
    let policy = policy_of(flags);
    let mut request = PullRequest::new(&store, base, refspec, &credential, &policy);
    request.lane = flag(flags, "lane").unwrap_or("");
    request.session = flag(flags, "session").unwrap_or("");
    request.streams = crate::flag_num(flags, "streams", transport::PULL_STREAMS);
    request.deadline = deadline_of(flags);
    // THE WALK REPORTS WHILE IT WALKS.
    //
    // Until now `tfs fetch` printed one report at the end, the pod's supervisor buffered
    // that output to EOF, and the owner logged the resulting stage change once. So a
    // 210 GB materialization was, to every layer above it, a single silent edge lasting
    // hours: an operator could not tell a healthy pull from a wedged one without reading
    // the daemon's raw log, and on 2026-09-03 a person had to.
    //
    // What is printed is what was MEASURED — bytes through each door and the instant they
    // landed — and nothing derived. A rate is the consumer's division of two of these
    // samples, which is the only place it can be computed honestly: this walk has sixteen
    // streams and no opinion about what interval anyone wants to average over. The
    // denominator is the native FetchPlan's declared byte count. The plan callback also
    // supplies already-held bytes, so a resumed or warm pull starts at its real position.
    //
    // The two doors are separate counters because they cost different things and only one
    // of them is an invoice. A cache that has silently gone cold produces a complete
    // closure, correct totals, and a bill — and looks exactly like a warm one unless the
    // split is reported while it happens.
    let progress = PullProgress::new(request.sample_seconds);
    let planned = |plan: &FetchPlan| progress.planned(plan);
    let observe = |_object: &ObjectRef, moved: u64, door: ObjectSource| progress.saw(moved, door);
    request.on_plan = Some(&planned);
    request.on_object = Some(&observe);
    let started = std::time::Instant::now();
    let report = ok!(transport::pull(&request));
    let seconds = started.elapsed().as_secs_f64();
    let after = tensorfs_core::stats::snapshot();
    progress.settle();
    if !tensorfs_core::repo_cache::finish_optional_backfills() {
        eprintln!("repo cache replication incomplete; local fetch remains complete");
    }
    // Exactly one model pull owns this receipt; a shared library process may
    // concurrently finish other pulls without contaminating this observation.
    let quoted = |value: &str| String::from_utf8(canon::write(&canon::Value::str(value))).unwrap();
    eprintln!(
        "{{\"event\":\"pull.cache\",\"model\":{},\"manifest\":{},\"scope\":{},\"cache_written_bytes\":{},\"cache_bytes\":{},\"total_bytes\":{}}}",
        quoted(&report.model), quoted(&report.manifest), quoted(&report.scope),
        report.cache_writes.bytes(), report.bytes_cached, report.bytes_total,
    );
    // What the requests cost and what admitting them cost: a pull whose batches take most
    // of the wall time is held by the disk, not the link. `durability` says which syncs the
    // disk the Store lives on made it pay, and what an unclean shutdown before this pull
    // cost to repair.
    let durability = &report.durability;
    eprintln!(
        "{{\"event\":\"pull.streams\",\"model\":{},\"seconds\":{:.3},\"bytes_moved\":{},\"streams\":{},\"requests\":{},\"restarts\":{},\"admission\":{{\"objects\":{},\"batches\":{},\"seconds\":{:.3}}},\"durability\":{{\"class\":{},\"filesystem\":{},\"syncs\":{},\"sync_seconds\":{:.3},\"recovery\":{{\"rehashed\":{},\"removed\":{},\"seconds\":{:.3}}}}}}}",
        quoted(&report.model),
        seconds,
        report.bytes_moved,
        report.streams,
        report.requests,
        report.restarts,
        after.admitted_objects - before.admitted_objects,
        after.admission_batches - before.admission_batches,
        (after.admission_nanos - before.admission_nanos) as f64 / 1e9,
        quoted(durability.class.as_str()),
        quoted(durability.filesystem),
        durability.syncs,
        durability.seconds,
        after.recovery_rehashed - before.recovery_rehashed,
        after.recovery_removed - before.recovery_removed,
        (after.recovery_nanos - before.recovery_nanos) as f64 / 1e9,
    );
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

// ------------------------------------------------------------ the observation channel

// ------------------------------------------------------------ the staging area

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

/// PullProgress is the walk's own reporting counter: totals through each door and the
/// cadence they are printed at.
///
/// `resolution` is the ledger's sampling cadence, reused rather than reinvented. It is a
/// REPORTING RESOLUTION and decides nothing: no refusal, no reclaim and no verdict depends
/// on it, and a walk that never crosses it still prints its terminal sample. Sixteen
/// streams complete objects concurrently, so without a resolution a closure of a hundred
/// thousand small objects would print a hundred thousand lines to say what one line says.
struct PullProgress {
    state: Mutex<PullProgressState>,
    resolution: Duration,
}

struct PullProgressState {
    total_bytes: u64,
    held_bytes: u64,
    wanted_objects: u64,
    origin_bytes: u64,
    cache_bytes: u64,
    origin_objects: u64,
    cache_objects: u64,
    last: Option<Instant>,
}

impl PullProgress {
    fn new(sample_seconds: f64) -> PullProgress {
        PullProgress {
            state: Mutex::new(PullProgressState {
                total_bytes: 0,
                held_bytes: 0,
                wanted_objects: 0,
                origin_bytes: 0,
                cache_bytes: 0,
                origin_objects: 0,
                cache_objects: 0,
                last: None,
            }),
            resolution: Duration::from_secs_f64(sample_seconds.clamp(0.001, 3600.0)),
        }
    }

    fn planned(&self, plan: &FetchPlan) {
        let mut state = self.state.lock().unwrap();
        state.total_bytes = plan.declared_bytes();
        state.held_bytes = plan.held_bytes();
        state.wanted_objects = plan.wanted.len() as u64;
        state.last = Some(Instant::now());
        print_progress(&state);
    }

    fn saw(&self, moved: u64, door: ObjectSource) {
        let mut state = self.state.lock().unwrap();
        match door {
            ObjectSource::Origin => {
                state.origin_bytes += moved;
                state.origin_objects += 1;
            }
            ObjectSource::Cache => {
                state.cache_bytes += moved;
                state.cache_objects += 1;
            }
        }
        let now = Instant::now();
        let due = match state.last {
            None => true,
            Some(last) => now.duration_since(last) >= self.resolution,
        };
        if due {
            state.last = Some(now);
            print_progress(&state);
        }
    }

    /// settle prints the final sample unconditionally, so the last bytes of a walk that
    /// ended inside one resolution window are still reported. A consumer that computes a
    /// rate from the gap between samples needs the last one to exist.
    fn settle(&self) {
        let mut state = self.state.lock().unwrap();
        state.last = Some(Instant::now());
        print_progress(&state);
    }
}

/// THE EVENT CHANNEL, and the one rule that governs everything on it.
///
/// Events are STDERR, one JSON object per line, `event` first. stdout carries the report a
/// caller parses for its result and nothing else, so **no event can ever be parsed as a
/// completion proof** — the guarantee is which stream it arrived on, not a convention about
/// its contents. `parseFetchReport` reads stdout and requires the `cached` row, and nothing
/// printed here can satisfy that.
///
/// Reserved on this channel, for the metadata-first ordering that lands beside it:
///
///   {"event":"pull.metadata","objects":N,"bytes":B}
///
/// emitted AT MOST ONCE, at the instant every metadata object of the closure — the
/// manifest, its CozyTensors header, and the header-declared spec and asset objects — is
/// verified local, and before the tensor bodies finish. It names no closure and proves no
/// completion; it is a permission to start reading, which is what lets a consumer overlap
/// the compatibility check with the rest of the walk instead of waiting for the whole
/// materialization. The instant belongs to the ordering change that produces it; only the
/// spelling is fixed here.
///
/// One sample, on STDERR as one JSON object per line. stdout carries the report a caller
/// parses for its result; progress is a side channel and must never be mistaken for one.
///
/// Every counter is always present, including a zero. A consumer that reads an absent key
/// as zero cannot tell a cache that answered nothing from a build that never reported the
/// door at all, and those are the two states worth telling apart. `wanted_objects` is the
/// plan's count of objects to land; a pod host sizes this pull's storage reservation from it.
fn print_progress(state: &PullProgressState) {
    eprintln!(
        "{{\"event\":\"pull.progress\",\"total_bytes\":{},\"held_bytes\":{},\"wanted_objects\":{},\"origin_bytes\":{},\"cache_bytes\":{},\"origin_objects\":{},\"cache_objects\":{}}}",
        state.total_bytes, state.held_bytes, state.wanted_objects, state.origin_bytes, state.cache_bytes,
        state.origin_objects, state.cache_objects
    );
}

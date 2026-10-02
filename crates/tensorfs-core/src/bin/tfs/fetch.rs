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
use tensorfs_core::fetch::{self, DeliveryGrant, FetchPlan};
use tensorfs_core::ids::{Doc, ObjectRef};
use tensorfs_core::manifest::Manifest;
use tensorfs_core::staging;
use tensorfs_core::store::Store;
use tensorfs_core::transport::ObjectSource;

use tensorfs_core::err::{Code, Refusal};
use tensorfs_core::ingest::fingerprint::FingerprintRegistry;
use tensorfs_core::ingest::preflight;

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
    pull_with_budget(root, refspec, flags, None)
}

/// A distinct verb makes an older CLI refuse instead of ignoring new quota flags.
pub fn cmd_bounded_pull(root: &Path, refspec: &str, flags: &Flags) -> ExitCode {
    let bound = |name: &str| {
        let values = crate::flag_all(flags, name);
        if values.len() != 1 {
            return None;
        }
        values[0].parse::<u64>().ok()
    };
    let (Some(bytes), Some(objects)) = (bound("max-download-bytes"), bound("max-download-objects"))
    else {
        eprintln!(
            "bounded fetch requires one unsigned --max-download-bytes and --max-download-objects"
        );
        return ExitCode::from(2);
    };
    pull_with_budget(root, refspec, flags, Some((bytes, objects)))
}

fn pull_with_budget(
    root: &Path,
    refspec: &str,
    flags: &Flags,
    budget: Option<(u64, u64)>,
) -> ExitCode {
    // Before the open, which is where an unclean shutdown's recovery runs.
    let before = tensorfs_core::stats::snapshot();
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
    request.streams = crate::flag_num(flags, "streams", transport::PULL_STREAMS);
    request.deadline = deadline_of(flags);
    request.download_budget = budget;
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

// ------------------------------------------------------------ the observation channel

/// The RESOLUTION of `--progress`, never a verdict about it.
///
/// A record every `SAMPLE_BYTES` keeps the channel's cost proportional to the BYTES MOVED
/// rather than to the number of fetches running: the pod runs several `fetch url` children
/// at once over one link, and the link is what bounds their sum, so the whole transfer
/// emits `moved / SAMPLE_BYTES` records however many children there are. At the 117.6 MB/s
/// #566b measured for a saturated link that is about two records a second; at the 6 MB/s
/// the H3 pod degraded to, one every ten seconds.
///
/// `SAMPLE_GAP` floors it so a crawling link still says so several times a minute, and it
/// is the ledger's own `SAMPLE_SECONDS` rather than a second number invented here — the
/// same cadence `cmd_pull`'s walk reports at. It is not a deadline and cannot fire on its
/// own: a record is only ever emitted by a read that ACTUALLY MOVED BYTES, so silence on
/// this channel means silence on the wire, which is the fact the parent is watching for.
/// Judging that silence is the parent's job and it does it by observation. Nothing in this
/// file can end a transfer.
const SAMPLE_BYTES: u64 = 64 << 20;

fn sample_gap() -> std::time::Duration {
    std::time::Duration::from_secs_f64(transport::SAMPLE_SECONDS)
}

/// `--progress`: this command's half of the event channel `cmd_pull` opened.
///
/// EVENTS ARE STDERR, one JSON object per line, `event` first; stdout carries the report a
/// caller parses for its result and nothing else. That is what makes "no event can be
/// parsed as a completion proof" structural rather than conventional — the guarantee is
/// which stream it arrived on. `fetch url`'s result is `held|admitted <id>` and
/// `transferred <n> B`, and nothing printed here can be mistaken for either.
///
/// One event, and the counter names are `pull.progress`'s so that one consumer reads both
/// verbs. `origin_bytes` is what is home so far; it goes back if an origin that does not
/// range starts the object over. `origin_objects` goes to one when the object is admitted:
///
///   {"event":"fetch.progress","origin_bytes":N,"origin_objects":0|1}
struct Reporter {
    moved: u64,
    reported: u64,
    /// The byte delta between records. `--progress-bytes` states it, because resolution is
    /// configuration and nothing here is a verdict; production states nothing and gets
    /// SAMPLE_BYTES.
    every: u64,
    gap: std::time::Duration,
    last: std::time::Instant,
}

impl Reporter {
    fn new(every: u64) -> Self {
        Reporter {
            moved: 0,
            reported: 0,
            every: every.max(1),
            gap: sample_gap(),
            last: std::time::Instant::now(),
        }
    }

    fn say(&mut self, event: &str) {
        use std::io::Write;
        let mut out = io::stderr().lock();
        // A closed channel is not this transfer's business. The bytes are wanted by
        // whoever asked for them, not by whoever was watching.
        let _ = writeln!(out, "{event}").and_then(|()| out.flush());
        self.last = std::time::Instant::now();
    }

    fn announce(&mut self, admitted: u64) {
        self.reported = self.moved;
        let event = format!(
            "{{\"event\":\"fetch.progress\",\"origin_bytes\":{},\"origin_objects\":{}}}",
            self.moved, admitted
        );
        self.say(&event);
    }
}

impl transport::FetchObserver for Reporter {
    fn moved(&mut self, total: u64) {
        // A count that goes back is reported at once: those bytes are not on this machine.
        let due = total < self.moved
            || total.saturating_sub(self.reported) >= self.every
            || self.last.elapsed() >= self.gap;
        self.moved = total;
        if due {
            self.announce(0);
        }
    }
}

/// `tfs fetch url <root> <sha256> <length> <url|-> [--progress] [--streams N] [--part-mib N]`
/// — plan-of-one, then the pull's downloader. A held object answers `held` and moves
/// nothing. A url of `-` reads the real URL as one line on stdin: a capability URL can
/// embed a signature, and argv is world-readable in /proc.
///
/// `--streams` is how many ranged requests of THIS object are in flight and `--part-mib`
/// one request's size. An origin that will not answer a range gives the object to one
/// request whatever these say.
///
/// `--progress` opens the observation channel above. Without it this command says nothing
/// until it is finished, which for a 5 GiB shard is ten to fifteen minutes of silence a
/// supervisor cannot read: the H3 ingest moved 210 GB over 2h56m and its only signal was a
/// cumulative byte total at each file's COMPLETION, so "slow" and "wedged" looked identical
/// for the whole run and the run was lost.
/// A url of `-` reads the real URL as one line on stdin: a capability URL can embed a
/// signature, and argv is world-readable in /proc.
fn url_of(url: &str) -> Result<String, ExitCode> {
    use std::io::BufRead;
    if url != "-" {
        return Ok(url.to_string());
    }
    let mut line = String::new();
    if io::stdin().lock().read_line(&mut line).is_err() {
        eprintln!("REFUSED SOURCE_NOT_ALLOWED: could not read the URL from stdin");
        return Err(ExitCode::FAILURE);
    }
    Ok(line.trim().to_string())
}

fn object_of(what: &str, id: &str, length: &str) -> Result<ObjectRef, ExitCode> {
    let sha256 = match tensorfs_core::ids::hex64(what, id) {
        Ok(sha256) => sha256,
        Err(refusal) => {
            eprintln!("REFUSED {refusal}");
            return Err(ExitCode::FAILURE);
        }
    };
    let Ok(length) = length.parse::<u64>() else {
        eprintln!("length must be a byte count");
        return Err(ExitCode::from(2));
    };
    Ok(ObjectRef { sha256, length })
}

fn ranged_of(flags: &Flags) -> transport::Ranged {
    transport::Ranged {
        streams: crate::flag_num(flags, "streams", transport::STREAMS),
        part: crate::flag_num::<u64>(flags, "part-mib", transport::PART_BYTES >> 20) << 20,
    }
}

/// Enumerate one locally admitted recovery link; no tensor bytes or NFS access are needed.
pub fn cmd_checkpoint_page(root: &Path, id: &str, length: &str, flags: &Flags) -> ExitCode {
    use tensorfs_core::canon::Value;
    use tensorfs_core::repo_cache::CacheKind;
    let store = ok!(Store::open(root));
    let object = match object_of("checkpoint head", id.trim_start_matches("sha256:"), length) {
        Ok(object) => object,
        Err(code) => return code,
    };
    let tensorfs_core::durability::Window {
        link,
        objects,
        next,
    } = ok!(tensorfs_core::durability::local_window(
        &store,
        &object,
        crate::flag_num(flags, "offset", 0),
        crate::flag_num(flags, "limit", tensorfs_core::durability::CHECKPOINT_REFS),
    ));
    let reference = |r: &ObjectRef| Value::arr(vec![Value::str(r.id()), Value::uint(r.length)]);
    let mut fields = vec![
        ("operation", Value::str(link.operation)),
        ("plan_digest", Value::str(link.plan)),
        ("index", Value::uint(link.index)),
        ("bytes", Value::uint(link.bytes)),
        (
            "objects",
            Value::arr(
                objects
                    .into_iter()
                    .map(|(kind, object)| {
                        Value::obj(vec![
                            (
                                "kind",
                                Value::str(match kind {
                                    CacheKind::Blob => "blob",
                                    CacheKind::Manifest => "manifest",
                                }),
                            ),
                            ("object_id", Value::str(object.id())),
                            ("length", Value::uint(object.length)),
                        ])
                    })
                    .collect(),
            ),
        ),
    ];
    if let Some(previous) = &link.prev {
        fields.push(("previous", reference(previous)));
    }
    if let Some(progress) = &link.progress {
        fields.push(("progress", reference(progress)));
    }
    if let Some(next) = next {
        fields.push(("next_offset", Value::uint(next as u64)));
    }
    let value = Value::obj(fields);
    println!(
        "{}",
        String::from_utf8(tensorfs_core::canon::write(&value)).expect("canonical JSON")
    );
    ExitCode::SUCCESS
}

pub fn cmd_url(root: &Path, id: &str, length: &str, url: &str, flags: &Flags) -> ExitCode {
    let store = ok!(Store::open(root));
    let url = match url_of(url) {
        Ok(url) => url,
        Err(code) => return code,
    };
    let url = url.as_str();
    let object = match object_of("fetch url object", id, length) {
        Ok(object) => object,
        Err(code) => return code,
    };
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
    let mut reporter = crate::flag_on(flags, "progress")
        .then(|| Reporter::new(crate::flag_num(flags, "progress-bytes", SAMPLE_BYTES)));
    let fetched = transport::fetch_ranged(
        &store,
        &grant,
        url,
        &policy_of(flags),
        &Anonymous,
        deadline_of(flags),
        &ledger,
        ranged_of(flags),
        reporter
            .as_mut()
            .map(|r| r as &mut dyn transport::FetchObserver),
    );
    let fetched = match fetched {
        Ok(fetched) => fetched,
        Err(refusal) => return bail(refusal),
    };
    // The last record is the one that closes the channel: every byte moved, and the object
    // admitted. A reader that saw nothing else still sees this.
    if let Some(reporter) = reporter.as_mut() {
        reporter.moved = fetched.transferred;
        reporter.announce(1);
    }
    println!("admitted     {}", fetched.object.id());
    println!("transferred  {} B", fetched.transferred);
    ExitCode::SUCCESS
}

// ------------------------------------------------------------ the staging area

/// `tfs staging fetch <root> <sha256> <length> <url|-> [--progress] [--streams N] …` — the
/// SOURCE-CARRIER fetch, and the one verb the pod's source lane calls.
///
/// It is `fetch url`'s transport exactly: the same policy fence on the first URL and every
/// redirect hop, the same ranged resume, the same event channel, and the same digest
/// enforced by the same hashing core. What differs is where the bytes come to rest — a
/// carrier lands at `<root>/staging/<hex>`, outside every namespace a census walks, with no
/// catalog record and nothing to garbage-collect. `tfs fetch url` is UNCHANGED and still
/// admits repo objects to the CAS; there is deliberately no flag that could put a 5 GB
/// carrier in `blobs/` or a repo object in `staging/`.
///
/// Two dispositions, and the second is what makes a restart cheap:
///
///   present <id> — a staged file at the right length REHASHED to the id asked for. Zero
///                  bytes moved, which is the behaviour `fetch url` has today for a HELD
///                  object and which a port could silently have dropped.
///   staged  <id> — this call moved the bytes.
///
/// `path` is on stdout in both cases, because the point of this issue is that the caller
/// stops deriving TensorFS's internal layout by hand.
pub fn cmd_staging_fetch(
    root: &Path,
    id: &str,
    length: &str,
    url: &str,
    flags: &Flags,
) -> ExitCode {
    let store = ok!(Store::open(root));
    let url = match url_of(url) {
        Ok(url) => url,
        Err(code) => return code,
    };
    let object = match object_of("staging fetch object", id, length) {
        Ok(object) => object,
        Err(code) => return code,
    };
    // Residency by REHASH, never by a record: a carrier is a file the store is about to
    // throw away, and a durable claim about one would outlive it.
    if let Some(path) = ok!(staging::staged(&store, &object)) {
        println!("present      {}", object.id());
        println!("path         {}", path.display());
        println!("transferred  0 B");
        return ExitCode::SUCCESS;
    }
    let session = flag(flags, "session").unwrap_or("staging");
    let grant = ok!(fetch::DeliveryGrant::stage(session, &object));
    let ledger = transport::Ledger::new();
    let mut reporter = crate::flag_on(flags, "progress")
        .then(|| Reporter::new(crate::flag_num(flags, "progress-bytes", SAMPLE_BYTES)));
    let fetched = transport::fetch_ranged(
        &store,
        &grant,
        &url,
        &policy_of(flags),
        &Anonymous,
        deadline_of(flags),
        &ledger,
        ranged_of(flags),
        reporter
            .as_mut()
            .map(|r| r as &mut dyn transport::FetchObserver),
    );
    let fetched = match fetched {
        Ok(fetched) => fetched,
        Err(refusal) => return bail(refusal),
    };
    if let Some(reporter) = reporter.as_mut() {
        reporter.moved = fetched.transferred;
        reporter.announce(1);
    }
    let path = ok!(staging::path(&store, &fetched.object.sha256));
    println!("staged       {}", fetched.object.id());
    println!("path         {}", path.display());
    println!("transferred  {} B", fetched.transferred);
    ExitCode::SUCCESS
}

/// `tfs staging drop <root> <sha256>` — retire one spent carrier. One unlink, IDEMPOTENT:
/// an id that is not there is success, because the caller's intent is already true.
///
/// This is the consumer of `Progress.spent` — the carriers no remaining conversion op will
/// read. It needs no hold check, no chain check and no catalog surgery, because there is
/// nothing to be atomic about.
pub fn cmd_staging_drop(root: &Path, id: &str) -> ExitCode {
    let store = ok!(Store::open(root));
    let sha256 = ok!(tensorfs_core::ids::hex64("staging drop object", id));
    let removed = ok!(staging::remove(&store, &sha256));
    println!(
        "{}      sha256:{sha256}",
        if removed { "dropped" } else { "absent " }
    );
    ExitCode::SUCCESS
}

/// `tfs staging clear <root>` — sweep the whole area, plus the orphan `stage-` temps of
/// fetches that died before linking.
///
/// The end-of-operation call. `drop` retires what the converter says is spent; this retires
/// everything the converter never got to say anything about, so a crash cannot leak 210 GB
/// into the next run on the same pod. A live fetch's temp is kept, on the kernel lock.
pub fn cmd_staging_clear(root: &Path) -> ExitCode {
    let store = ok!(Store::open(root));
    let cleared = ok!(staging::clear(&store));
    println!("cleared      {} carrier(s)", cleared.carriers);
    println!("reclaimed    {} B", cleared.bytes);
    println!("temps        {} orphan(s)", cleared.temps);
    ExitCode::SUCCESS
}

/// `tfs staging list <root>` — every carrier on disk, its length and its absolute path.
pub fn cmd_staging_list(root: &Path) -> ExitCode {
    let store = ok!(Store::open(root));
    let carriers = ok!(staging::list(&store));
    let mut bytes = 0u64;
    for carrier in &carriers {
        bytes = bytes.saturating_add(carrier.length);
        println!("{}", ok!(staging::line(&store, carrier)));
    }
    println!("staged       {} carrier(s), {bytes} B", carriers.len());
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

/// `tfs source resolve <uri>` — what one pinned foreign source resolves to, EXACTLY:
/// every tensor carrier, its object id, its length and its URL, with sharded repositories
/// expanded through their `weight_map`. No store, no tensor bytes.
///
/// This is the verb an owner runs BEFORE renting anything: the container-disk floor is
/// `objects` bytes, and it has to exist before the disk it sizes. Nothing here takes a
/// `Store` for exactly that reason.
///
/// `--json` is the machine answer. It prints `members` and `objects` as separate counts
/// and separate totals, because they differ whenever two members are the same bytes and a
/// caller that conflates them will size a disk for weights it will only store once.
pub fn cmd_source_resolve(uri: &str, flags: &Flags) -> ExitCode {
    let (uri, endpoints, credential) = match source_setup(uri, flags) {
        Ok(setup) => setup,
        Err(code) => return code,
    };
    let resolution = ok!(providers::resolve(
        &uri,
        &endpoints,
        &credential,
        deadline_of(flags)
    ));
    // NARROWING. A repository is not a model. MiniMax-H3 resolves to 112 carriers /
    // 498.3 GB; the reviewed profiles that name an actual model select 5 carriers, which
    // expand through their indexes to 48 members / 47 objects / 210.3 GB. Skipping this
    // step does not fail — it downloads 2.4x more, successfully — so the profile names and
    // the member list they narrow to are both printed, and `resolved`/`selected` counts
    // are reported separately whenever a narrowing actually happened.
    let profiles: Vec<String> = crate::flag_all(flags, "source-profile")
        .into_iter()
        .map(str::to_string)
        .collect();
    let full = resolution.members.len();
    let resolution = if profiles.is_empty() {
        resolution
    } else {
        let registry = match crate::ingest::load_registry(flags) {
            Ok(registry) => registry,
            Err(code) => return code,
        };
        let carriers = ok!(registry.exact_source_members(&profiles));
        ok!(resolution.select(&carriers))
    };
    if crate::flag_on(flags, "json") {
        let rows: Vec<canon::Value> = resolution
            .members
            .iter()
            .map(|row| {
                canon::Value::obj(vec![
                    ("carrier", canon::Value::Bool(row.carrier)),
                    ("length", canon::Value::uint(row.object.length)),
                    ("member", canon::Value::str(row.member.clone())),
                    ("object", canon::Value::str(row.object.id())),
                    ("provenance", canon::Value::str(row.provenance.as_str())),
                    (
                        "requires",
                        canon::Value::arr(
                            row.requires
                                .iter()
                                .map(|need| canon::Value::str(need.clone()))
                                .collect(),
                        ),
                    ),
                    ("url", canon::Value::str(row.url.clone())),
                ])
            })
            .collect();
        let document = canon::Value::obj(vec![
            ("canonical", canon::Value::str(resolution.canonical.clone())),
            (
                "source_profiles",
                canon::Value::arr(
                    profiles
                        .iter()
                        .map(|profile| canon::Value::str(profile.clone()))
                        .collect(),
                ),
            ),
            ("resolved_members", canon::Value::uint(full as u64)),
            (
                "member_bytes",
                canon::Value::uint(resolution.member_bytes()),
            ),
            ("members", canon::Value::arr(rows)),
            (
                "object_bytes",
                canon::Value::uint(resolution.object_bytes()),
            ),
            (
                "objects",
                canon::Value::uint(resolution.objects().len() as u64),
            ),
            (
                "selection_sha256",
                canon::Value::str(resolution.selection_sha256.clone()),
            ),
        ]);
        println!("{}", String::from_utf8_lossy(&canon::write(&document)));
        return ExitCode::SUCCESS;
    }
    for row in &resolution.members {
        println!(
            "{}  {:>14} B  {:9}  {}{}",
            row.object.id(),
            row.object.length,
            row.provenance.as_str(),
            if row.carrier { "" } else { "  " },
            row.member
        );
    }
    println!("canonical    {}", resolution.canonical);
    if profiles.is_empty() {
        println!(
            "profiles     (none) — THE WHOLE REPOSITORY. A model is a narrowing of this; \n\
             \x20            pass --source-profile <name> to select one."
        );
        println!("members      {}", resolution.members.len());
    } else {
        println!("profiles     {}", profiles.join(", "));
        println!(
            "members      {} of {} resolved carriers",
            resolution.members.len(),
            full
        );
    }
    println!(
        "objects      {} distinct — what a store admits",
        resolution.objects().len()
    );
    println!("member bytes {}", resolution.member_bytes());
    println!(
        "object bytes {} — the container-disk floor",
        resolution.object_bytes()
    );
    println!("selection    sha256:{}", resolution.selection_sha256);
    ExitCode::SUCCESS
}

/// `tfs source plan <uri> --source-profile <name>...` — decide the conversion plan from
/// HEADERS, before a byte of payload moves, before a pod is rented (tfs-076).
///
/// > *"why does conversion have to download all 200GBs of files and then refuse? shouldn't
/// > it refuse right away?"*
///
/// It should, and this is where it does. Resolve, narrow, read each member's header by
/// ranged GET, and run the pod's OWN planner over them. No Store, no tensor bytes, no
/// rental. On MiniMax-H3 that is 275 KB of header standing in for 210.3 GB of payload.
///
/// Three exits, and they mean three different things:
///
/// - **0, `PLAN OK`.** Every requested profile plans. The session id printed for each is
///   the conversion journal's key, and the pod will compute the same one.
/// - **1, `REFUSED <code>`.** A guard refused on the headers. This selection cannot convert
///   and no amount of transfer changes that — run 290, 294 and 309 all end here, for free.
/// - **2, `UNDECIDED`.** One or more headers could not be read cheaply (an origin that will
///   not serve a range). NOT a refusal: a caller that put this in front of a rental should
///   proceed to the ordinary path, because "I could not look" is never "this is bad".
pub fn cmd_source_plan(uri: &str, flags: &Flags) -> ExitCode {
    let (uri, endpoints, credential) = match source_setup(uri, flags) {
        Ok(setup) => setup,
        Err(code) => return code,
    };
    // The narrowing is not optional and omitting it fails INTERESTINGLY: H3's repository
    // resolves to 112 carriers that match no reviewed profile, so an unnarrowed preflight
    // would refuse a model that converts perfectly well. A profile is what makes the
    // question answerable at all.
    let profiles: Vec<String> = crate::flag_all(flags, "source-profile")
        .into_iter()
        .map(str::to_string)
        .collect();
    if profiles.is_empty() {
        return bail(Refusal {
            code: Code::MISSING_FIELD,
            detail: "source plan requires --source-profile <name>: a repository is not a \
                     model, and only a reviewed profile says which model this is"
                .into(),
        });
    }
    let (registry_locator, registry_bytes) = match crate::ingest::registry_bytes(flags) {
        Ok(registry) => registry,
        Err(code) => return code,
    };
    let registry = ok!(FingerprintRegistry::parse(&registry_bytes));
    let wanted = ok!(registry.exact_source_members(&profiles));

    let full = ok!(providers::resolve(
        &uri,
        &endpoints,
        &credential,
        deadline_of(flags)
    ));
    let resolved = full.members.len();
    let selection = ok!(full.select(&wanted));
    let heads = ok!(providers::read_heads(
        &selection,
        &uri,
        &endpoints,
        &credential,
        deadline_of(flags)
    ));

    let json = crate::flag_on(flags, "json");
    if !heads.unread.is_empty() {
        if json {
            let document = canon::Value::obj(vec![
                (
                    "unread",
                    canon::Value::arr(
                        heads
                            .unread
                            .iter()
                            .map(|row| {
                                canon::Value::obj(vec![
                                    ("member", canon::Value::str(row.member.clone())),
                                    ("why", canon::Value::str(row.why.clone())),
                                ])
                            })
                            .collect(),
                    ),
                ),
                ("verdict", canon::Value::str("undecided".to_string())),
            ]);
            println!("{}", String::from_utf8_lossy(&canon::write(&document)));
        } else {
            for row in &heads.unread {
                eprintln!("unread  {}  {}", row.member, row.why);
            }
            eprintln!(
                "UNDECIDED — {} of {} member header(s) could not be read cheaply. This is \
                 not a refusal: proceed to the ordinary path.",
                heads.unread.len(),
                selection.members.len()
            );
        }
        return ExitCode::from(2);
    }

    let scratch = std::env::temp_dir().join(format!(
        "tfs-source-plan-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|since| since.as_nanos())
            .unwrap_or(0)
    ));
    let preflight = ok!(preflight::plan(
        &registry_locator,
        &registry_bytes,
        &heads.heads,
        &profiles,
        &scratch,
    ));

    if json {
        let mut fields = match preflight.value() {
            canon::Value::Obj(fields) => fields,
            _ => Vec::new(),
        };
        fields.push((
            "requests".to_string(),
            canon::Value::uint(heads.requests as u64),
        ));
        fields.push((
            "resolved_members".to_string(),
            canon::Value::uint(resolved as u64),
        ));
        fields.push((
            "selection_sha256".to_string(),
            canon::Value::str(selection.selection_sha256.clone()),
        ));
        fields.sort_by(|left, right| left.0.cmp(&right.0));
        println!(
            "{}",
            String::from_utf8_lossy(&canon::write(&canon::Value::Obj(fields)))
        );
        return ExitCode::SUCCESS;
    }

    for plan in &preflight.plans {
        println!("profile      {}", plan.profile);
        println!("converter    {}", plan.converter);
        println!("target       {}", plan.target);
        println!(
            "session      {} — the conversion journal's key",
            plan.session
        );
        println!("constructs   {} tensor(s)", plan.constructs);
        for component in &plan.components {
            println!(
                "  {:<14} {}{}",
                component.component,
                component.member.as_deref().unwrap_or("(unnamed carrier)"),
                if component.projected {
                    "  [projected]"
                } else {
                    ""
                }
            );
        }
    }
    println!("canonical    {}", selection.canonical);
    println!(
        "members      {} of {} resolved carriers",
        preflight.members, resolved
    );
    println!("registry     {}", preflight.registry_sha256);
    println!("selection    sha256:{}", selection.selection_sha256);
    println!(
        "header bytes {} read in {} ranged GET(s)",
        preflight.header_bytes, heads.requests
    );
    println!(
        "member bytes {} NOT moved — this plan was decided without them",
        preflight.member_bytes
    );
    println!("PLAN OK — this selection can convert.");
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

//! tfs: the Hub's TensorFS verifier, fixture writer and maintainer tool. `USAGE` lists the
//! verbs; the library is the product path (TensorD links it).
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use tensorfs_core::canon;
use tensorfs_core::durability;
use tensorfs_core::err::Refusal;
use tensorfs_core::header::Header;
use tensorfs_core::ids::{Doc, ObjectRef};
use tensorfs_core::repo_cache::{CacheKind, RepoObjectCache};
use tensorfs_core::store::{Fault, Store, FAULT_STAGES};

type TfsResult<T> = tensorfs_core::err::Result<T>;

macro_rules! ok {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => return bail(e),
        }
    };
}

const USAGE: &str = "usage: tfs <verb> ...
  version
  key blob|manifest <sha256> | repo <org> <name>
  cbor encode|decode|check <file> [--out <file>]
  manifest build <entries.jsonl> --out <manifest.json>
  manifest inspect <manifest.json> --refs <jsonl> [--header <h> --inventory <i> --closure <c>]
  manifest get|walk <root> <sha256> ...
  repo inspect <repo.json> --rows <jsonl>
  repo apply <current.json|-> <mutation.json> --out <file> --rows <jsonl> [--manifest <m> --header <h> --inventory <i>]
  store init <root> [--repo-cache <cache> | --no-repo-cache]
  store rebuild <census> --rows <jsonl>
  gc headers <census> --out <jsonl>
  gc plan <census> --holds <jsonl> --out <jsonl>
  put|get <root> ..., reap <root>
  checkpoint write <root> [--plain] [--blocks <n>] [--no-configs]
  fetch <root> <ref> --hub <base>
  ensure <root> <ref> --hub <base>
  repo-cache restore|resume|chain ...
  ingest bank <dialect> <converter> ..., ingest source-members|source-plan|plan|run|install|reap ...";

/// Flags that never take a value.
const SWITCHES: &[&str] = &[
    "allow-local", "dry-run", "keys", "no-configs",
    "no-disk-budget", "no-journal", "no-netns", "no-repo-cache", "plain", "progress",
];
/// Flags that take the next token as their value; given last, or before another flag, they
/// are present without one.
const VALUED: &[&str] = &[
    "allow-hosts", "blocks", "ca-file", "carrier", "checkpoint-bytes", "closure",
    "config", "credential-file", "disk-budget", "expect", "expect-length", "fault",
    "header", "holds", "hub", "inventory", "isolation-tier", "keep", "lane",
    "manifest", "max-download-bytes", "max-download-objects", "max-redirects", "member",
    "observed", "order", "out", "part-mib", "plan", "progress-bytes", "range", "ready", "refs",
    "registry", "repo-cache", "rlimit-as-mb", "rlimit-fsize-mb", "rows", "sample-seconds",
    "scale", "session", "slow-ms", "source", "source-name", "source-plan", "source-profile",
    "spec-variant", "step", "streams", "target", "tenant", "timeout",
];

fn usage() -> ExitCode {
    eprintln!("{USAGE}");
    ExitCode::from(2)
}

/// ONE spelling in both directions, on the POSITIONAL surface. Every id this tool prints is
/// `sha256:<hex>`, and a positional digest used to be bare hex only, so an id copied out of
/// one command could not be pasted into the next (job-001 stripped the prefix at every
/// hand-off). A positional argument that is exactly a printed id is now accepted as itself.
///
/// FLAG values are deliberately untouched: `--pin` and its kind already take the prefixed
/// spelling on the way in, so they were never the inconsistent half, and rewriting them
/// broke a flag-carried digest the first time this was applied to the whole argv. And only
/// the CLI is loosened either way — a canonical DOCUMENT still holds each field to its one
/// spelling, which is what keeps digests stable.
fn accept_printed_ids(args: Vec<String>) -> Vec<String> {
    args.into_iter()
        .map(|t| match t.strip_prefix("sha256:") {
            Some(h) if tensorfs_core::ids::hex64("argument", h).is_ok() => h.to_string(),
            _ => t,
        })
        .collect()
}

fn main() -> ExitCode {
    let (args, flags) = split_flags(std::env::args().skip(1).collect());
    // `ensure` names its CA itself so a bad file is a refusal on its own channel.
    let ensure = args.first().is_some_and(|verb| verb == "ensure");
    if let (Some(path), false) = (flag(&flags, "ca-file"), ensure) {
        let trusted = fs::read(path)
            .map_err(|e| format!("IO_FAILED: read {path}: {e}"))
            .and_then(|pem| {
                tensorfs_core::transport::trust_roots(&pem).map_err(|r| format!("{r} ({path})"))
            });
        if let Err(refusal) = trusted {
            eprintln!("REFUSED {refusal}");
            return ExitCode::FAILURE;
        }
    }
    let args = accept_printed_ids(args);
    let a: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    match a.as_slice() {
        ["version"] => cmd_version(),
        ["ensure", r, refspec] => ensure::cmd_ensure(Path::new(r), refspec, &flags),
        ["store", "init", r] => cmd_store_init(Path::new(r), &flags),
        ["store", "rebuild", r] => repo::cmd_rebuild(Path::new(r), &flags),
        ["repo-cache", "restore", store, object, length] => {
            cmd_repo_cache_restore(Path::new(store), object, length, &flags)
        }
        ["repo-cache", "resume", store, object, length] => {
            cmd_repo_cache_resume(Path::new(store), object, length, &flags)
        }
        ["repo-cache", "chain", cache, object, length] => {
            cmd_repo_cache_chain(Path::new(cache), object, length)
        }
        ["put", r, f] => cmd_put(Path::new(r), Path::new(f), &flags),
        ["get", r, h] => cmd_get(Path::new(r), h, &flags),
        ["reap", r] => cmd_reap(Path::new(r)),
        ["checkpoint", "write", r] => ckpt::cmd_write(Path::new(r), &flags),
        ["manifest", "get", r, h] => snap::cmd_get(Path::new(r), h, &flags),
        ["manifest", "walk", r, h] => snap::cmd_walk(Path::new(r), h, &flags),
        ["manifest", "build", entries] => repo::cmd_manifest_build(Path::new(entries), &flags),
        ["manifest", "inspect", p] => repo::cmd_manifest_inspect(Path::new(p), &flags),
        ["repo", "inspect", p] => repo::cmd_repo_inspect(Path::new(p), &flags),
        ["repo", "apply", current, mutation] => {
            repo::cmd_repo_apply(current, Path::new(mutation), &flags)
        }
        ["key", kind, rest @ ..] => repo::cmd_key(kind, rest),
        ["gc", "headers", census] => repo::cmd_gc_headers(Path::new(census), &flags),
        ["gc", "plan", census] => repo::cmd_gc_plan(Path::new(census), &flags),
        ["ingest", "bank", dia, conv] => ingest::cmd_bank(dia, conv, &flags),
        ["ingest", "source-members", profiles @ ..] => ingest::cmd_source_members(profiles, &flags),
        ["ingest", "source-plan"] => ingest::cmd_source_plan(&flags),
        ["ingest", "plan"] => match flag(&flags, "source-plan") {
            Some(path) => ingest::cmd_plan_source(Path::new(path)),
            None => {
                eprintln!("ingest plan requires a target or --source-plan <json>");
                ExitCode::from(2)
            }
        },
        ["ingest", "plan", t] => ingest::cmd_plan(t, &flags),
        ["ingest", "run", r] => match flag(&flags, "source-plan") {
            Some(path) => ingest::cmd_run_source(Path::new(r), Path::new(path), &flags),
            None => {
                eprintln!("ingest run requires a target or --source-plan <json>");
                ExitCode::from(2)
            }
        },
        ["ingest", "run", r, t] => ingest::cmd_run(Path::new(r), t, &flags),
        ["ingest-worker", r, t] => ingest::cmd_worker(Path::new(r), t, &flags),
        ["ingest", "install", r, s, org, name, version, lane] => {
            ingest::cmd_install(Path::new(r), s, org, name, version, lane, &flags)
        }
        ["ingest", "reap", r] => ingest::cmd_reap(Path::new(r), &flags),
        ["fetch", r, refspec] => fetch::cmd_pull(Path::new(r), refspec, &flags),
        ["cbor", "encode", f] => cmd_cbor("encode", Path::new(f), &flags),
        ["cbor", "decode", f] => cmd_cbor("decode", Path::new(f), &flags),
        ["cbor", "check", f] => cmd_cbor("check", Path::new(f), &flags),
        _ => usage(),
    }
}

/// `tfs <version> sha256:<digest of this binary's own bytes>` — one line, machine-parsed by
/// tensorhub and cozy-creator at startup (tfs-051). Byte-identical builds print the same
/// digest; two builds stamped with one version cannot hide behind it.
fn cmd_version() -> ExitCode {
    let bytes = match std::env::current_exe().and_then(fs::read) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("REFUSED IO_FAILED: read own binary: {error}");
            return ExitCode::from(1);
        }
    };
    println!(
        "tfs {} sha256:{}",
        env!("CARGO_PKG_VERSION"),
        tensorfs_core::sha256::hex_digest(&bytes)
    );
    ExitCode::SUCCESS
}

fn cache_args(kind: &str, object: &str, length: &str) -> TfsResult<(CacheKind, ObjectRef)> {
    let kind = CacheKind::parse(kind).ok_or_else(|| Refusal {
        code: tensorfs_core::err::Code::WRONG_TYPE,
        detail: format!("repo cache kind {kind:?} is not blob or manifest"),
    })?;
    let sha256 = tensorfs_core::ids::hex64("repo cache object", object)?;
    let length = length.parse::<u64>().map_err(|_| Refusal {
        code: tensorfs_core::err::Code::NUMBER_RANGE,
        detail: format!("repo cache length {length:?} is not an unsigned integer"),
    })?;
    Ok((kind, ObjectRef { sha256, length }))
}

/// `tfs repo-cache restore` — bring one whole checkpoint back off the cache.
///
/// This is the ordinary answer to "the pod that converted this is gone." The Manifest is
/// admitted, its closure walked, and every object the local Store does not already hold is
/// admitted from the cache through the SAME door a pull uses. There is no adoption on trust
/// and no second admission path: a cached object whose bytes disagree with its id is removed
/// and reported, and the completion proof at the end is the Store's own local walk.
fn cmd_repo_cache_restore(store: &Path, object: &str, length: &str, flags: &Flags) -> ExitCode {
    let (_, manifest) = ok!(cache_args("manifest", object, length));
    let store = ok!(Store::open(store));
    let session = flag(flags, "session").unwrap_or("restore");
    let report = ok!(durability::restore_checkpoint(&store, session, &manifest));
    print_restore(&report)
}

/// `tfs repo-cache resume` — bring back everything a conversion had already made durable,
/// and the map of what those bytes MEAN.
///
/// The chain head names links that live on the cache; each is read at its exact digest, and
/// the chain is DISCARDED WHOLE unless every link was written under the plan `--plan` names
/// — the plan a resuming run re-derives from the source headers alone, for zero tensor
/// bytes. A discarded chain restores nothing and succeeds: the run converts afresh.
///
/// The objects come back exactly as a checkpoint's closure does. The conversion journal the
/// newest link points at is installed beside them under `--session`, so the next
/// `tfs ingest run` reuses the ops it describes instead of re-converting them — after
/// `transaction::convert` re-checks every journalled part against the Store's own admission
/// law, which is why a journal restored beside objects that did not survive costs a
/// re-conversion and never a wrong artifact. `--no-journal` restores the bytes alone.
fn cmd_repo_cache_resume(store: &Path, object: &str, length: &str, flags: &Flags) -> ExitCode {
    let Some(plan) = flag(flags, "plan") else {
        eprintln!(
            "REFUSED MISSING_FIELD: resume requires --plan <sha256:...>, the digest of the plan \
             re-derived from these sources; without it a chain cannot be shown to be this \
             conversion's"
        );
        return ExitCode::FAILURE;
    };
    let (_, head) = ok!(cache_args("blob", object, length));
    let store = ok!(Store::open(store));
    let session = flag(flags, "session").unwrap_or("resume");
    let cache = match store.repo_cache() {
        Some(cache) => cache.clone(),
        None => {
            eprintln!(
                "REFUSED MISSING_FIELD: {} is bound to no repo cache; bind one with \
                 `tfs store ensure --repo-cache <root>`",
                store.root().display()
            );
            return ExitCode::FAILURE;
        }
    };
    let chain = ok!(durability::chain(&cache, &head));
    let set = match durability::fold(&chain, plan) {
        Ok(set) => set,
        Err(error) if error.code == tensorfs_core::err::Code::IO_FAILED => return bail(error),
        Err(error) => {
            println!(
                "chain        DISCARDED {}: {} — nothing restored; the conversion starts a new chain",
                error.code.as_str(),
                error.detail
            );
            return ExitCode::SUCCESS;
        }
    };
    println!(
        "chain        {} link(s) over operation {} — {} B declared durable",
        set.links, set.operation, set.bytes
    );
    let report = ok!(durability::restore_objects(&store, session, &set.blobs, &set.manifests));
    if !flag_on(flags, "no-journal") {
        match set.progress.last() {
            Some(snapshot) => {
                let digest = plan.trim_start_matches("sha256:");
                let tenant = flag(flags, "tenant").unwrap_or("dev");
                match durability::restore_conversion(&store, session, tenant, digest, snapshot) {
                    Ok(()) => println!(
                        "conversion   {} installed under session {session} — the ops it names \
                         are re-checked, never trusted",
                        snapshot.id()
                    ),
                    // The bytes are back either way; without the map they are re-converted,
                    // which costs CPU and never correctness.
                    Err(error) => println!(
                        "conversion   REFUSED {}: {} — the objects stand and the conversion \
                         re-derives what they mean",
                        error.code.as_str(),
                        error.detail
                    ),
                }
            }
            None => println!("conversion   no journal snapshot is durable; every op re-converts"),
        }
    }
    print_restore(&report)
}

/// `tfs repo-cache chain` — the links a head names, root first, reading nothing but the
/// cache. It is how an operator sees whether a run's watermark is ADVANCING, which is the
/// only honest question to ask about a long conversion: not how many minutes it has taken,
/// but whether more of it is durable than the last time anyone looked.
fn cmd_repo_cache_chain(cache: &Path, object: &str, length: &str) -> ExitCode {
    let (_, head) = ok!(cache_args("blob", object, length));
    let cache = RepoObjectCache::new(cache);
    let chain = ok!(durability::chain(&cache, &head));
    for link in &chain {
        let object = link.object_ref();
        println!(
            "{:<6} {} {} B durable, {} blob(s), {} manifest(s), plan {}, length {}",
            link.index,
            object.id(),
            link.bytes,
            link.blobs.len(),
            link.manifests.len(),
            link.plan,
            object.length
        );
    }
    println!("operation    {}", chain[0].operation);
    ExitCode::SUCCESS
}

fn print_restore(report: &tensorfs_core::durability::Restored) -> ExitCode {
    println!(
        "held         {} object(s), {} B — NOT restored",
        report.held, report.bytes_held
    );
    println!(
        "restored     {} object(s), {} B — from the cache, verified into this Store",
        report.admitted, report.bytes_admitted
    );
    println!(
        "absent       {} object(s) the cache could not answer for, {} corrupt",
        report.missing, report.corrupt
    );
    if report.complete {
        println!("complete     every declared object is resident and verified");
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "REFUSED OBJECT_ABSENT: the cache did not hold the whole declared set; what it \
             held is now local and proved, and the rest has to be produced again"
        );
        ExitCode::FAILURE
    }
}

// ------------------------------------------------------------------ store CLI

/// `--key value` pairs and bare `--flag` switches, split out of the positionals.
type Flags = Vec<(String, String)>;

/// A switch never takes the next token. An unknown flag warns, and takes the next token as
/// its value as every flag once did, so a caller's typo is visible and nothing it passes moves.
fn split_flags(argv: Vec<String>) -> (Vec<String>, Flags) {
    let (mut pos, mut flags) = (Vec::new(), Flags::new());
    let mut it = argv.into_iter().peekable();
    while let Some(t) = it.next() {
        match t.strip_prefix("--") {
            None => pos.push(t),
            Some(k) => {
                let switch = SWITCHES.contains(&k);
                if !switch && !VALUED.contains(&k) {
                    eprintln!("tfs: warning: unknown flag --{k}");
                }
                let v = match it.peek() {
                    Some(n) if !switch && !n.starts_with("--") => it.next().unwrap(),
                    _ => "1".to_string(),
                };
                flags.push((k.to_string(), v));
            }
        }
    }
    (pos, flags)
}

fn flag<'a>(f: &'a Flags, k: &str) -> Option<&'a str> {
    f.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str())
}
/// Every value given for a REPEATABLE flag, in the order they were typed.
fn flag_all<'a>(f: &'a Flags, k: &str) -> Vec<&'a str> {
    f.iter()
        .filter(|(a, _)| a == k)
        .map(|(_, v)| v.as_str())
        .collect()
}
fn flag_on(f: &Flags, k: &str) -> bool {
    flag(f, k).is_some()
}
/// `--k <number>`, or `d` when absent. A value that is not a number is a usage error.
fn flag_num<T: std::str::FromStr>(f: &Flags, k: &str, d: T) -> T {
    match flag(f, k) {
        None => d,
        Some(v) => v.parse().unwrap_or_else(|_| {
            eprintln!("tfs: --{k} {v:?} is not a number");
            std::process::exit(2)
        }),
    }
}

fn bail(e: Refusal) -> ExitCode {
    eprintln!("REFUSED {e}");
    ExitCode::FAILURE
}

mod ckpt;
mod ensure;
mod fetch;
mod ingest;
mod repo;
mod snap;
mod watch;

/// `tfs store init|ensure|info <root>` — and, on the two SETUP verbs, the one place a
/// deployment says where its repo cache is.
///
/// `--repo-cache <root>` records the mount ON THE STORE, so every later handle finds it and
/// no operation is ever handed one. `--no-repo-cache` records that there is none. Neither
/// flag leaves whatever the Store already carried; the pair exists so that a caller whose
/// binding must track an environment — the pod supervisor, whose volume is bound per
/// datacenter — passes exactly one on EVERY boot and can never inherit a mount that has
/// gone away. A human running `tfs store init ~/.tensorfs` passes neither and gets what
/// they had.
///
/// Neither flag touches the cache path. See `Store::bind_repo_cache`: a stat on a hard NFS
/// mount that has stopped answering never returns, and boot is not allowed to hang on the
/// thing that may only cost latency.
fn cmd_store_init(root: &Path, flags: &Flags) -> ExitCode {
    let bind = match (flag(flags, "repo-cache"), flag_on(flags, "no-repo-cache")) {
        (Some(_), true) => {
            eprintln!("--repo-cache and --no-repo-cache say opposite things; pass one");
            return ExitCode::from(2);
        }
        (Some(path), false) => Some(Some(PathBuf::from(path))),
        (None, true) => Some(None),
        (None, false) => None,
    };
    let budget = match (flag(flags, "disk-budget"), flag_on(flags, "no-disk-budget")) {
        (Some(_), true) => {
            eprintln!("--disk-budget and --no-disk-budget say opposite things; pass one");
            return ExitCode::from(2);
        }
        (Some(value), false) => match value.parse::<u64>() {
            Ok(bytes) => Some(Some(bytes)),
            Err(_) => {
                eprintln!("--disk-budget takes a byte count, not {value:?}");
                return ExitCode::from(2);
            }
        },
        (None, true) => Some(None),
        (None, false) => None,
    };
    let mut s = ok!(Store::init(root));
    if let Some(cache) = bind {
        s = ok!(s.bind_repo_cache(cache.as_deref()));
    }
    if let Some(budget) = budget {
        s = ok!(s.bind_disk_budget(budget));
    }
    println!("root:       {}", s.root().display());
    println!("blobs:      {}", ok!(s.objects()).len());
    println!(
        "repo-cache: {}",
        match s.repo_cache() {
            Some(cache) => cache.root().display().to_string(),
            None => "(none)".to_string(),
        }
    );
    // Printed in the same readback shape as repo-cache, and for the same reason: a
    // supervisor fences the line to prove the tfs it was baked with implements the flag it
    // just passed, rather than trusting a zero exit (tensorhub pod-supervisor
    // supervise.go:229-234).
    println!(
        "disk-budget: {}",
        match s.disk_budget() {
            Some(bytes) => format!("{bytes} B, holding {} B", ok!(s.occupancy())),
            None => "(none)".to_string(),
        }
    );
    println!(
        "durability: {} ({})",
        s.disk_class().as_str(),
        s.disk().kind
    );
    ExitCode::SUCCESS
}

fn fault_of(f: &Flags) -> Fault {
    Fault {
        stage: flag(f, "fault").map(|s| s.to_string()),
        ready: flag(f, "ready").map(PathBuf::from),
    }
}

/// `<off>:<len>` or `<off>:` (to the end) inside an object of `length` bytes.
fn byte_range(range: &str, length: u64) -> TfsResult<(u64, u64)> {
    let bad = || Refusal {
        code: tensorfs_core::err::Code::RANGE_BOUNDS,
        detail: format!("--range {range:?} is not <off>:<len> or <off>: inside {length} B"),
    };
    let (off, len) = range.split_once(':').ok_or_else(bad)?;
    let off: u64 = off.parse().map_err(|_| bad())?;
    let len: u64 = match len {
        "" => length.checked_sub(off).ok_or_else(bad)?,
        len => len.parse().map_err(|_| bad())?,
    };
    match off.checked_add(len) {
        Some(end) if end <= length => Ok((off, len)),
        _ => Err(bad()),
    }
}

/// Stream a range of one verified object to stdout under a read lease, so a GC pass cannot
/// take the object mid-read.
fn get_leased(store: &Store, hex: &str, range: Option<&str>) -> TfsResult<()> {
    use std::io::Write;
    use tensorfs_core::read::{self, ObjectRange};
    let length = store.open_verified(hex)?.len();
    let (mut off, mut left) = match range {
        Some(range) => byte_range(range, length)?,
        None => (0, length),
    };
    let meta = tensorfs_core::meta::Meta::open(store)?;
    let obj = ObjectRef {
        sha256: hex.to_string(),
        length,
    };
    let (lease, _) = read::acquire(store, &meta, "", vec![obj.clone()])?;
    let mut out = std::io::stdout().lock();
    let mut buf = vec![0u8; 1 << 20];
    let streamed = (|| {
        while left > 0 {
            let n = left.min(buf.len() as u64);
            let chunk = &mut buf[..n as usize];
            let range = ObjectRange {
                obj: obj.clone(),
                off,
                len: n,
            };
            read::read_into(&lease, &range, chunk)?;
            out.write_all(chunk).map_err(|e| Refusal {
                code: tensorfs_core::err::Code::IO_FAILED,
                detail: format!("write stdout: {e}"),
            })?;
            (off, left) = (off + n, left - n);
        }
        out.flush().map_err(|e| Refusal {
            code: tensorfs_core::err::Code::IO_FAILED,
            detail: format!("write stdout: {e}"),
        })
    })();
    lease.release(&meta)?;
    streamed
}

fn cmd_put(root: &Path, src: &Path, f: &Flags) -> ExitCode {
    let s = ok!(Store::open(root));
    let expect = match (flag(f, "expect"), flag(f, "expect-length")) {
        (None, _) => None,
        (Some(_), None) => {
            eprintln!("--expect requires --expect-length: an ObjectRef is length-bearing");
            return ExitCode::from(2);
        }
        (Some(h), Some(l)) => Some(ObjectRef {
            sha256: h.trim_start_matches("sha256:").to_string(),
            length: l.parse().unwrap_or(u64::MAX),
        }),
    };
    let fault = fault_of(f);
    if let Some(st) = &fault.stage {
        if !FAULT_STAGES.contains(&st.as_str()) {
            eprintln!("--fault {st}: stages are {FAULT_STAGES:?}");
            return ExitCode::from(2);
        }
    }
    let p = ok!(s.put_file(src, expect.as_ref(), &fault));
    println!(
        "{} sha256:{} length={}",
        if p.admitted {
            "admitted"
        } else {
            "present (lost the admission race; this process claims nothing about its bytes)"
        },
        p.obj.sha256,
        p.obj.length
    );
    ExitCode::SUCCESS
}

fn cmd_get(root: &Path, hex: &str, f: &Flags) -> ExitCode {
    let s = ok!(Store::open(root));
    if flag(f, "out") == Some("-") {
        ok!(get_leased(&s, hex, flag(f, "range")));
        return ExitCode::SUCCESS;
    }
    let t = Instant::now();
    if let Some(r) = flag(f, "range") {
        let (o, l) = r.split_once(':').unwrap_or((r, "1"));
        let bytes = ok!(s.read_range(hex, o.parse().unwrap_or(0), l.parse().unwrap_or(1)));
        println!(
            "range {} bytes in {:.3} s, sha256:{}",
            bytes.len(),
            t.elapsed().as_secs_f64(),
            tensorfs_core::sha256::hex_digest(&bytes)
        );
        return ExitCode::SUCCESS;
    }
    let mut sink: Box<dyn std::io::Write> = match flag(f, "out") {
        Some(p) => Box::new(match fs::File::create(p) {
            Ok(fh) => fh,
            Err(e) => {
                eprintln!("{p}: {e}");
                return ExitCode::FAILURE;
            }
        }),
        None => Box::new(Counting(0)),
    };
    let n = ok!(s.read_into(hex, &mut sink));
    let secs = t.elapsed().as_secs_f64();
    println!(
        "read {n} verified bytes in {secs:.3} s ({:.1} MiB/s)",
        n as f64 / secs / 1048576.0
    );
    ExitCode::SUCCESS
}

struct Counting(u64);
impl std::io::Write for Counting {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0 += b.len() as u64;
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn cmd_reap(root: &Path) -> ExitCode {
    let s = ok!(Store::open(root));
    let (gone, kept) = ok!(s.reap());
    println!("reap: removed {gone} orphaned temps, kept {kept} with a live writer");
    ExitCode::SUCCESS
}

fn rss_kb(field: &str) -> u64 {
    fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find(|line| line.starts_with(field))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}


// ------------------------------------------------------------------ id

fn write_cli(bytes: &[u8], flags: &Flags) -> ExitCode {
    if let Some(path) = flag(flags, "out") {
        match fs::write(path, bytes) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("{path}: {error}");
                ExitCode::FAILURE
            }
        }
    } else {
        use std::io::Write;
        match std::io::stdout().write_all(bytes) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("stdout: {error}");
                ExitCode::FAILURE
            }
        }
    }
}

fn cmd_cbor(action: &str, path: &Path, flags: &Flags) -> ExitCode {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("{}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };
    match action {
        "encode" => {
            let value =
                ok!(canon::parse_with_depth(&bytes, tensorfs_core::limits::DOC_MAX_BYTES, 16));
            let header = ok!(Header::from_diagnostic_value(&value));
            if let Err(error) = header.validate_structure() {
                return bail(error);
            }
            let encoded = ok!(header.canonical_bytes());
            if let Err(error) = Header::parse(&encoded) {
                return bail(error);
            }
            write_cli(&encoded, flags)
        }
        "decode" => {
            let header = ok!(Header::parse(&bytes));
            let value = ok!(header.diagnostic_value());
            let mut rendered = canon::pretty(&value, 0).into_bytes();
            rendered.push(b'\n');
            write_cli(&rendered, flags)
        }
        "check" => {
            let header = ok!(Header::parse(&bytes));
            println!("header_digest {}", tensorfs_core::ids::object_id(&bytes));
            println!("bytes {}", bytes.len());
            println!("components {}", header.components.len());
            println!("tensors {}", header.tensors().count());
            for (component, tensors) in &header.components {
                println!("component {component} {}", tensors.len());
                for (index, (key, _)) in tensors.iter().enumerate() {
                    println!("order {component} {index} {key}");
                }
            }
            ExitCode::SUCCESS
        }
        _ => unreachable!(),
    }
}

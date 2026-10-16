//! tfs: the Hub's TensorFS verifier, fixture writer and maintainer tool. `USAGE` lists the
//! verbs; the library is the product path (TensorD links it).
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use tensorfs_core::canon;
use tensorfs_core::err::Refusal;
use tensorfs_core::header::Header;
use tensorfs_core::ids::ObjectRef;
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
  version [--json]
  key blob|manifest <sha256> | repo <org> <name>
  cbor encode|decode|check <file> [--out <file>]
  manifest build <entries.jsonl> --out <manifest.json>
  manifest inspect <manifest.json> --refs <jsonl> [--header <h> --inventory <i> --closure <c>]
  manifest get|walk <root> <sha256> ...
  repo inspect <repo.json> --rows <jsonl>
  repo apply <current.json|-> <mutation.json> --out <file> --rows <jsonl> [--manifest <m> --header <h> --inventory <i>]
  store init <root> [--disk-budget <bytes> | --no-disk-budget]
  store rebuild <census> --rows <jsonl>
  gc headers <census> --out <jsonl>
  gc plan <census> --holds <jsonl> --out <jsonl>
  put|get <root> ..., reap <root>
  fetch <root> <ref> --hub <base> [--lane <lane>]
  checkpoint write <root> [--plain] [--blocks <n>] [--no-configs]
  ingest bank <dialect> <converter> ..., ingest source-members|source-plan|plan|run|install|reap ...";

/// Flags that never take a value.
const SWITCHES: &[&str] = &[
    "allow-local", "dry-run", "json", "keys", "no-configs",
    "no-disk-budget", "no-netns", "plain", "progress",
];
/// Flags that take the next token as their value; given last, or before another flag, they
/// are present without one.
const VALUED: &[&str] = &[
    "allow-hosts", "blocks", "ca-file", "carrier", "closure",
    "config", "credential-file", "disk-budget", "expect", "expect-length", "fault",
    "header", "holds", "hub", "inventory", "isolation-tier", "keep", "lane",
    "manifest", "max-download-bytes", "max-download-objects", "max-redirects", "member",
    "observed", "order", "out", "part-mib", "progress-bytes", "range", "ready", "refs",
    "registry", "rlimit-as-mb", "rlimit-fsize-mb", "rows", "sample-seconds",
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
    if let Some(path) = flag(&flags, "ca-file") {
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
        ["version"] => cmd_version(flag_on(&flags, "json")),
        ["store", "init", r] => cmd_store_init(Path::new(r), &flags),
        ["store", "rebuild", r] => repo::cmd_rebuild(Path::new(r), &flags),
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
            Some(path) => ingest::cmd_run_source(Path::new(r), Path::new(path)),
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
        ["fetch", r, refspec] => fetch::cmd_fetch(Path::new(r), refspec, &flags),
        ["cbor", "encode", f] => cmd_cbor("encode", Path::new(f), &flags),
        ["cbor", "decode", f] => cmd_cbor("decode", Path::new(f), &flags),
        ["cbor", "check", f] => cmd_cbor("check", Path::new(f), &flags),
        _ => usage(),
    }
}

/// `tfs <version> sha256:<digest of this binary's own bytes>` — one line, machine-parsed by
/// tensorhub and cozy-creator at startup (tfs-051). Byte-identical builds print the same
/// digest; two builds stamped with one version cannot hide behind it.
/// `tfs <version> sha256:<own digest>`, or with `--json` that and the Store contract.
fn cmd_version(json: bool) -> ExitCode {
    let bytes = match std::env::current_exe().and_then(fs::read) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("REFUSED IO_FAILED: read own binary: {error}");
            return ExitCode::from(1);
        }
    };
    let digest = tensorfs_core::sha256::hex_digest(&bytes);
    if json {
        let record = serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
            "sha256": digest,
            "store_format": tensorfs_core::STORE_FORMAT,
            "claim_protocol": tensorfs_core::CLAIM_PROTOCOL,
        });
        println!("{record}");
    } else {
        println!("tfs {} sha256:{digest}", env!("CARGO_PKG_VERSION"));
    }
    ExitCode::SUCCESS
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
mod fetch;
mod ingest;
mod repo;
mod snap;
mod watch;

/// `tfs store init <root>`, recording or clearing its disk budget.
fn cmd_store_init(root: &Path, flags: &Flags) -> ExitCode {
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
    if let Some(budget) = budget {
        s = ok!(s.bind_disk_budget(budget));
    }
    println!("root:       {}", s.root().display());
    println!("blobs:      {}", ok!(s.objects()).len());
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

//! tfs — TensorFS store and CozyTensors inspection tool.
//!
//!   tfs registry                 print the seeded platform namespace (alias -> digest)
//!   tfs id <file>                object id (+ derived ids) of a stored document
//!   tfs cbor encode|decode|check <file>  header conformance seam; JSON is diagnostic only
//!
//! The object store (tfs-002). Flags: --expect <hex> --expect-length <n> --out <path>
//! --range <off>:<len> --rate <f> --pass <n> --max-age <s> --slow-ms <n>
//! --size <bytes> --small <n>, plus the dev-only kill points --fault <stage> --ready <path>.
//!
//!   tfs store init|info <root>
//!   tfs store rebuild <census> --rows <jsonl> [--sqlite <tensorfs.sqlite>]
//!   tfs put <root> <file>        stream one object in (any shape)
//!   tfs get <root> <hex>         verified read, whole or ranged
//!   tfs contains <root> <hex>    the presence HINT, and what it is worth
//!   tfs verify <root> <hex>      verification record or rehash; remove corrupt bytes
//!   tfs scrub <root>             one rate knob over the whole store
//!   tfs reap <root>              age-bounded temp reap
//!   tfs fill <root> <plan>       resumable fill: skip only verified objects
//!   tfs bench-store <root>       put/read/range/rehash throughput + peak RSS
//!
//! Structure fixtures (tfs-008). Flags: --provenance real|synthetic --source <name>
//! --source-sha256 <hex> --note <text>
//!
//!   tfs structure bank <rows> <family> <variant>        bank TRANSCRIBED evidence rows
//!   tfs structure bank-header <root> <hex> <fam> <var>  bank from a live header
//!   tfs structure list                                  the bank, with provenance
//!   tfs structure match <root> <hex>                    does this checkpoint match one?
//!
//! Qualification (tfs-008): what an (encoding, device) pair is QUALIFIED to execute.
//! FAIL-CLOSED -- an absent pair refuses, because an unrecognised device is the absence of
//! evidence and never permission. The compiled-in records speak only for the reference
//! decoders in `tfs conform`, on the CPU they run on; an accelerator answer exists only
//! once a component that HELD the card hands one in through `--records`.
//!
//!   tfs conform [<what>]                             qualify the reference decoders
//!   tfs vectors mine <rows> <name>                   mine a producer's vectors
//!   tfs capability <encoding> <device> [--alias <a>] [--records <observed.json>] [--json]
//!
//! The repository/manifest/blob store (tfs-034).
//!
//!   tfs manifest build <entries.jsonl> --out <manifest.json>
//!   tfs manifest admit|get|show|walk|verify <root> ...
//!   tfs manifest inspect <manifest.json> --refs <jsonl>
//!   tfs repo inspect <repo.json> --rows <jsonl>
//!   tfs repo apply <current.json|-> <mutation.json> --out <replacement> --rows <jsonl>
//!                  put_checkpoint requires --manifest, --header and --inventory
//!   tfs repo commit <store> <current.json|-> <mutation.json>
//!   tfs local resolve <store> <name>
//!   tfs local replace <store> <name> <source-sha256> <manifest> <length>
//!     [--expected-topology <sha256>] [--expected-encoding <sha256>]...
//!                     --observed <repository-sha256|absent> --evidence <evidence.json>
//!   tfs local remove <store> <name> --observed <repository-sha256>
//!   tfs key blob|manifest <sha256> | repo <org> <name>
//!   tfs gc headers <census> --out <requests.jsonl>
//!   tfs gc plan <census> --holds <holds.jsonl> --out <plan.jsonl>
//!
//! The checkpoint plane (tfs-015). Flags: --blocks <n> --pin <alias|empty> --seed <s>
//! `write`/`read` also accept `--plain`, the closed all-`plain/1` fixture in PyTorch
//! `named_parameters()` traversal order.
//!
//!   tfs checkpoint write <root> [--plain]    objectize a synthetic tree, emit the header
//!   tfs checkpoint read <root> <hex>         reconstruct every part, byte-identity checked
//!   tfs checkpoint info <root> <hex>         component/scope query, zero tensor-byte reads
//!   tfs checkpoint locate <root> <hex> <component> <key> <role> <off> <len>
//!   tfs checkpoint admit <root> <hex>        serving admission against the local pin
//!   tfs checkpoint reproduce <root> <header.json> <entries.json>
//!                                            verified refs -> canonical header + manifest
//!   tfs checkpoint caps                      whole-document total caps, armed live
//!   tfs checkpoint size [n]                  uniform-form header cost at H3 cardinality
//!   tfs ingest source-members <profile>... [--registry <operator-registry.json>]
//!   tfs ingest source-plan --carrier <path>... [--registry <operator-registry.json>]
//!
//! Reads and projection (tfs-005). Flags: --workers <n> --window <b> --ring <b>
//! --hold-secs <s> --drop-unreleased --no-symlink
//!
//!   tfs read lease <root> <manifest>         acquire/verify/release one ReadLease
//!   tfs read plan <root> <manifest>          stored construction order + drift refusal
//!   tfs read fill <root> <manifest>          the pooled fill into a caller-owned ring
//!   tfs read verify-fill <root> <manifest>   the same bytes through the sequential path
//!   tfs read span <root> <manifest> <c> <k> <role> <off> <len>
//!   tfs read arms <root> <manifest>          the read/projection red arms, planted live
//!   tfs read tear <root> <manifest>          tear a verified blob UNDER a live lease
//!   tfs read prove <root> <manifest>         delivered bytes vs header digests
//!   tfs read transform                       run decomposition + the block-granularity rule
//!   tfs checkout <root> <manifest> <dest>    manifest tree; header included, payload CAS-only
//!   tfs materialize <root> <manifest> <path> <dest>  independent ordinary-file bytes
//!   tfs render <root> <hex>                  pretty PROJECTION of a canonical document
//!
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use tensorfs_core::canon;
use tensorfs_core::compat::{self, Accepts, CodeTopology, CompatibilityReport, Lane, Requirement};
use tensorfs_core::dtype::Dtype;
use tensorfs_core::err::Refusal;
use tensorfs_core::grade::{self, ConversionRegistration, Mapping};
use tensorfs_core::header::{Asset, Body, Closure, Header, Part, Tensor};
use tensorfs_core::ids::{ConfigRef, Doc, ObjectRef, Plain};
use tensorfs_core::ingest::stamp::Stamp;
use tensorfs_core::ingest::{Evidence, IngestProfile, IngestSubject, IngestVerificationReceipt};
use tensorfs_core::manifest::{Draft, Entry, Manifest};
use tensorfs_core::registry;
use tensorfs_core::spec::EncodingSpec;
use tensorfs_core::store::{Fault, Store, Verdict, FAULT_STAGES};

type TfsResult<T> = tensorfs_core::err::Result<T>;

/// ONE spelling in both directions, on the POSITIONAL surface. Every id this tool prints is
/// `sha256:<hex>`, and a positional digest used to be bare hex only, so an id copied out of
/// one command could not be pasted into the next (job-001 stripped the prefix at every
/// hand-off). A positional argument that is exactly a printed id is now accepted as itself.
///
/// FLAG values are deliberately untouched: `--pin` and its kind already take the prefixed
/// spelling on the way in, so they were never the inconsistent half, and rewriting them
/// broke `verify-compat` the first time this was applied to the whole argv. And only the
/// CLI is loosened either way — a canonical DOCUMENT still holds each field to its one
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
    let args = accept_printed_ids(args);
    let a: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    match a.as_slice() {
        ["store", "init", r] => cmd_store(Path::new(r), "init"),
        ["store", "info", r] => cmd_store(Path::new(r), "info"),
        ["store", "rebuild", r] => repo::cmd_rebuild(Path::new(r), &flags),
        ["put", r, f] => cmd_put(Path::new(r), Path::new(f), &flags),
        ["get", r, h] => cmd_get(Path::new(r), h, &flags),
        ["contains", r, h] => cmd_contains(Path::new(r), h),
        ["verify", r, h] => cmd_verify_object(Path::new(r), h, &flags),
        ["scrub", r] => cmd_scrub(Path::new(r), &flags),
        ["reap", r] => cmd_reap(Path::new(r), &flags),
        ["fill", r, p] => cmd_fill(Path::new(r), Path::new(p), &flags),
        ["bench-store", r] => cmd_bench_store(Path::new(r), &flags),
        ["checkpoint", "write", r] => ckpt::cmd_write(Path::new(r), &flags),
        ["checkpoint", "read", r, h] => ckpt::cmd_read(Path::new(r), h, &flags),
        ["checkpoint", "info", r, h] => ckpt::cmd_info(Path::new(r), h),
        ["checkpoint", "locate", r, h, c, k, role, off, len] => ckpt::cmd_locate(
            Path::new(r),
            h,
            [c, k, role],
            off.parse().unwrap_or(0),
            len.parse().unwrap_or(0),
            &flags,
        ),
        ["checkpoint", "admit", r, h] => ckpt::cmd_admit(Path::new(r), h, &flags),
        ["conform"] => conform::cmd_conform(None, &flags),
        ["conform", w] => conform::cmd_conform(Some(w), &flags),
        ["capability", e, d] => conform::cmd_capability(e, d, &flags),
        ["vectors", "mine", rows, name] => conform::cmd_mine(Path::new(rows), name, &flags),
        ["structure", "bank", rows, fam, var] => {
            fixtures::cmd_bank_rows(Path::new(rows), fam, var, &flags)
        }
        ["structure", "bank-header", r, h, fam, var] => {
            fixtures::cmd_bank_header(Path::new(r), h, fam, var, &flags)
        }
        ["structure", "list"] => fixtures::cmd_list(),
        ["structure", "match", r, h] => fixtures::cmd_match(Path::new(r), h),
        ["manifest", "show", r, h] => snap::cmd_show(Path::new(r), h),
        ["manifest", "get", r, h] => snap::cmd_get(Path::new(r), h, &flags),
        ["manifest", "admit", r, p] => snap::cmd_admit(Path::new(r), Path::new(p), &flags),
        ["manifest", "walk", r, h] => snap::cmd_walk(Path::new(r), h, &flags),
        ["manifest", "verify", r, h] => snap::cmd_verify(Path::new(r), h),
        ["manifest", "build", entries] => repo::cmd_manifest_build(Path::new(entries), &flags),
        ["checkpoint", "reproduce", r, h, e] => {
            snap::cmd_reproduce(Path::new(r), Path::new(h), Path::new(e), &flags)
        }
        ["manifest", "inspect", p] => repo::cmd_manifest_inspect(Path::new(p), &flags),
        ["repo", "inspect", p] => repo::cmd_repo_inspect(Path::new(p), &flags),
        ["repo", "list", r] => repo::cmd_repo_list(Path::new(r), &flags),
        ["repo", "get", r, org, name] => repo::cmd_repo_get(Path::new(r), org, name, &flags),
        ["repo", "apply", current, mutation] => {
            repo::cmd_repo_apply(current, Path::new(mutation), &flags)
        }
        ["repo", "commit", store, current, mutation] => {
            repo::cmd_repo_commit(Path::new(store), current, Path::new(mutation), &flags)
        }
        ["local", "resolve", store, name] => repo::cmd_local_resolve(Path::new(store), name),
        ["local", "replace", store, name, source, manifest, length] => {
            repo::cmd_local_replace(Path::new(store), name, source, manifest, length, &flags)
        }
        ["local", "remove", store, name] => repo::cmd_local_remove(Path::new(store), name, &flags),
        ["key", kind, rest @ ..] => repo::cmd_key(kind, rest),
        ["gc", "headers", census] => repo::cmd_gc_headers(Path::new(census), &flags),
        ["gc", "holds", store] => repo::cmd_gc_holds(Path::new(store), &flags),
        ["gc", "plan", census] => repo::cmd_gc_plan(Path::new(census), &flags),
        ["ingest", "inspect"] => ingest::cmd_inspect(&flags),
        ["gguf", "plan", f] => ingest::cmd_gguf_plan(Path::new(f), false, &flags),
        ["gguf", "admit", f] => ingest::cmd_gguf_plan(Path::new(f), true, &flags),
        ["ingest", "converters"] => ingest::cmd_converters(),
        ["ingest", "golden"] => ingest::cmd_golden(),
        ["ingest", "bank", dia, fam, var, conv] => ingest::cmd_bank(dia, fam, var, conv, &flags),
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
        ["ingest", "reingest", r, h, t] => ingest::cmd_reingest(Path::new(r), h, t, &flags),
        ["ingest", "install", r, s, org, name, version, lane] => {
            ingest::cmd_install(Path::new(r), s, org, name, version, lane, &flags)
        }
        ["ingest", "restamp", r, s] => ingest::cmd_restamp(Path::new(r), s, &flags),
        ["ingest", "reap", r] => ingest::cmd_reap(Path::new(r), &flags),
        ["ingest", "sessions", r] => ingest::cmd_sessions(Path::new(r)),
        ["checkpoint", "caps"] => ckpt::cmd_caps(),
        ["checkpoint", "size"] => ckpt::cmd_size(3699),
        ["checkpoint", "size", n] => ckpt::cmd_size(n.parse().unwrap_or(3699)),
        ["read", "lease", r, h] => rd::cmd_lease(Path::new(r), h, &flags),
        ["read", "plan", r, h] => rd::cmd_plan(Path::new(r), h, &flags),
        ["read", "fill", r, h] => rd::cmd_fill(Path::new(r), h, &flags),
        ["read", "stream", r, h] => rd::cmd_stream(Path::new(r), h, &flags),
        ["read", "verify-fill", r, h] => rd::cmd_verify_fill(Path::new(r), h, &flags),
        ["read", "span", r, h, c, k, role, off, len] => rd::cmd_span(
            Path::new(r),
            h,
            [c, k, role],
            off.parse().unwrap_or(0),
            len.parse().unwrap_or(0),
        ),
        ["read", "arms", r, h] => rd::cmd_arms(Path::new(r), h),
        ["read", "tear", r, h] => rd::cmd_tear(Path::new(r), h),
        ["read", "swap", r, h] => rd::cmd_swap(Path::new(r), h),
        ["read", "prove", r, h] => rd::cmd_prove(Path::new(r), h, &flags),
        ["read", "transform"] => rd::cmd_transform(),
        ["checkout", r, h, d] => rd::cmd_checkout(Path::new(r), h, Path::new(d), &flags),
        ["materialize", r, h, p, d] => rd::cmd_materialize(Path::new(r), h, p, Path::new(d)),
        ["render", r, h] => rd::cmd_render(Path::new(r), h),
        ["fetch", "plan", r, d, s] => fetch::cmd_plan(Path::new(r), Path::new(d), s, &flags),
        ["fetch", "admit", r, p, i] => fetch::cmd_admit(Path::new(r), Path::new(p), i, &flags),
        ["fetch", "complete", r, p] => fetch::cmd_complete(Path::new(r), Path::new(p)),
        ["compat", "import", r] => layout::cmd_import(Path::new(r), &flags),
        ["compat", "topology", r, h] => layout::cmd_topology(Path::new(r), h, &flags),
        ["compat", "check", c, h] => layout::cmd_check(Path::new(c), Path::new(h), &flags),
        ["compat", "release-cut", r] => layout::cmd_release_cut(Path::new(r)),
        ["grade", "decision", d] => layout::cmd_decision(Path::new(d)),
        // The BATCH surface (tfs-039). Grading is microseconds; the wall was the process
        // around it. `batch` reads a file of canonical reports, `serve` reads them from
        // stdin and flushes per line, and both emit exactly the bytes `grade --json` emits
        // for the same document — so the single-pair form below is untouched.
        ["grade", "batch", p] => layout::cmd_grade_stream(Some(Path::new(p)), &flags),
        ["grade", "serve"] => layout::cmd_grade_stream(None, &flags),
        ["grade", r] => layout::cmd_grade(Path::new(r), &flags),
        ["grade", "register", n, k, s] => layout::cmd_register(n, k, s, &flags),
        ["verify-request", c, h] => layout::cmd_request(Path::new(c), Path::new(h), &flags),
        ["verify-compat", "serve"] => layout::cmd_verify_stream(None),
        ["verify-compat", "batch", p] => layout::cmd_verify_stream(Some(Path::new(p))),
        ["verify-compat", q, o] => layout::cmd_verify(Path::new(q), Path::new(o), &flags),
        ["verify-consume", q, o] => layout::cmd_consume(Path::new(q), Path::new(o)),
        ["registry"] => cmd_registry(),
        ["vectors", "gen"] => cmd_gen(&default_dir()),
        ["vectors", "gen", dir] => cmd_gen(Path::new(dir)),
        ["vectors", "verify"] => cmd_verify(&default_dir()),
        ["vectors", "verify", dir] => cmd_verify(Path::new(dir)),
        ["cbor", "encode", f] => cmd_cbor("encode", Path::new(f), &flags),
        ["cbor", "decode", f] => cmd_cbor("decode", Path::new(f), &flags),
        ["cbor", "check", f] => cmd_cbor("check", Path::new(f), &flags),
        ["cbor", "plant", f, kind] => cmd_cbor_plant(Path::new(f), kind, &flags),
        ["id", f] => cmd_id(Path::new(f)),
        _ => {
            eprintln!("usage: see the header of {}", file!());
            ExitCode::from(2)
        }
    }
}

// ------------------------------------------------------------------ store CLI

/// `--key value` pairs and bare `--flag` switches, split out of the positionals.
type Flags = Vec<(String, String)>;

fn split_flags(argv: Vec<String>) -> (Vec<String>, Flags) {
    let (mut pos, mut flags) = (Vec::new(), Flags::new());
    let mut it = argv.into_iter().peekable();
    while let Some(t) = it.next() {
        match t.strip_prefix("--") {
            None => pos.push(t),
            Some(k) => {
                let v = match it.peek() {
                    Some(n) if !n.starts_with("--") => it.next().unwrap(),
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
fn flag_num<T: std::str::FromStr>(f: &Flags, k: &str, d: T) -> T {
    flag(f, k).and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn bail(e: Refusal) -> ExitCode {
    eprintln!("REFUSED {e}");
    ExitCode::FAILURE
}

macro_rules! ok {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => return bail(e),
        }
    };
}

mod ckpt;
mod conform;
mod fetch;
mod fixtures;
mod ingest;
mod layout;
mod rd;
mod repo;
mod snap;

fn cmd_store(root: &Path, verb: &str) -> ExitCode {
    let s = match verb {
        "init" => ok!(Store::init(root)),
        _ => ok!(Store::open(root)),
    };
    println!("root:       {}", s.root().display());
    println!("blobs:      {}", ok!(s.objects()).len());
    ExitCode::SUCCESS
}

fn fault_of(f: &Flags) -> Fault {
    Fault {
        stage: flag(f, "fault").map(|s| s.to_string()),
        ready: flag(f, "ready").map(PathBuf::from),
    }
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

fn cmd_contains(root: &Path, hex: &str) -> ExitCode {
    let s = ok!(Store::open(root));
    let present = s.contains(hex);
    println!("contains:  {present}   (presence HINT only)");
    match s.record_valid(hex) {
        Ok(r) => println!(
            "record:    valid (dev:ino {}:{}, {} bytes)",
            r.dev, r.ino, r.length
        ),
        Err(why) => println!("record:    INVALID — {why}"),
    }
    ExitCode::SUCCESS
}

fn cmd_verify_object(root: &Path, hex: &str, f: &Flags) -> ExitCode {
    let s = ok!(Store::open(root));
    // Deleted 2026-08-25 (owner ruling). A flag that silently does nothing is worse than
    // one that is gone, so passing it says where the door moved to.
    if flag_on(f, "paranoid") {
        eprintln!(
            "REFUSED UNKNOWN_FIELD: --paranoid is DELETED (owner ruling 2026-08-25). A read \
             rehashes exactly when its verification record is missing or stale, which is \
             correctness and is not optional; the deliberate full-rehash door is \
             `tfs scrub <root> --rate 1.0`."
        );
        return ExitCode::FAILURE;
    }
    let v = ok!(s.verify(hex));
    match &v {
        Verdict::Verified { rehashed } => {
            println!("VERIFIED sha256:{hex} (rehashed: {rehashed})");
            ExitCode::SUCCESS
        }
        Verdict::Invalidated { why, .. } => {
            println!("RECORD INVALIDATED ({why}) — rehashed, bytes are good, record refreshed");
            ExitCode::SUCCESS
        }
        Verdict::CorruptRemoved { why } => {
            eprintln!("CORRUPT REMOVED sha256:{hex} ({why})");
            ExitCode::FAILURE
        }
    }
}

fn cmd_get(root: &Path, hex: &str, f: &Flags) -> ExitCode {
    let s = ok!(Store::open(root));
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

fn cmd_scrub(root: &Path, f: &Flags) -> ExitCode {
    let s = ok!(Store::open(root));
    let (rate, pass) = (flag_num(f, "rate", 0.1f64), flag_num(f, "pass", 0u64));
    let t = Instant::now();
    let r = ok!(s.scrub(rate, pass));
    let secs = t.elapsed().as_secs_f64();
    println!(
        "scrub rate={rate} pass={pass}: scanned {} sampled {} rehashed {} corrupt-removed {} ({:.1} MiB hashed, {secs:.3} s, {:.1} MiB/s)",
        r.scanned,
        r.sampled,
        r.rehashed,
        r.removed,
        r.bytes_hashed as f64 / 1048576.0,
        r.bytes_hashed as f64 / secs.max(1e-9) / 1048576.0
    );
    ExitCode::from(if r.removed > 0 { 1 } else { 0 })
}

fn cmd_reap(root: &Path, f: &Flags) -> ExitCode {
    let s = ok!(Store::open(root));
    let age = flag_num(f, "max-age", 3600u64);
    let (gone, kept) = ok!(s.reap(age));
    println!("reap max-age={age}s: removed {gone} temps, kept {kept}");
    ExitCode::SUCCESS
}

/// Plan lines are `<hex> <length> <path>`. Resume skips an object ONLY under a valid
/// verification record — presence alone is never enough.
fn cmd_fill(root: &Path, plan: &Path, f: &Flags) -> ExitCode {
    let s = ok!(Store::open(root));
    let text = match fs::read_to_string(plan) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{}: {e}", plan.display());
            return ExitCode::FAILURE;
        }
    };
    let slow = flag_num(f, "slow-ms", 0u64);
    let (mut put, mut skipped, mut refused) = (0, 0, 0);
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let p: Vec<&str> = line.split_whitespace().collect();
        let (hex, len, src) = (p[0], p[1].parse::<u64>().unwrap_or(0), p[2]);
        match s.record_valid(hex) {
            Ok(_) => {
                skipped += 1;
                println!("skip   sha256:{hex} (valid verification record)");
                continue;
            }
            // Present but unverified. Presence alone never resumes, and no-clobber
            // admission means a corrupt squatter would silently survive a re-put — so the
            // rehash happens HERE: good bytes resume, bad bytes are removed and the
            // way and the object is admitted afresh.
            Err(why) if s.contains(hex) => match ok!(s.verify(hex)) {
                Verdict::CorruptRemoved { .. } => {
                    println!("requeue sha256:{hex} ({why}; corrupt bytes removed)")
                }
                _ => {
                    skipped += 1;
                    println!("skip   sha256:{hex} ({why}; rehash agreed, record refreshed)");
                    continue;
                }
            },
            Err(_) => {}
        }
        let want = ObjectRef {
            sha256: hex.to_string(),
            length: len,
        };
        match s.put_file(Path::new(src), Some(&want), &fault_of(f)) {
            Ok(_) => {
                put += 1;
                println!("put    sha256:{hex}");
            }
            Err(e) => {
                refused += 1;
                println!("REFUSE sha256:{hex} {e}");
            }
        }
        if slow > 0 {
            std::thread::sleep(std::time::Duration::from_millis(slow));
        }
    }
    println!("fill: put {put}, skipped {skipped}, refused {refused}");
    ExitCode::from(if refused > 0 { 1 } else { 0 })
}

fn cmd_bench_store(root: &Path, f: &Flags) -> ExitCode {
    let s = ok!(Store::open(root));
    let size: u64 = flag_num(f, "size", 1u64 << 30);
    let small: usize = flag_num(f, "small", 2000);
    let src = root.join("bench.src");
    print!("synthesizing {:.0} MiB … ", size as f64 / 1048576.0);
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let t = Instant::now();
    {
        let mut w = std::io::BufWriter::new(fs::File::create(&src).unwrap());
        let mut block = vec![0u8; 1 << 20];
        let mut x: u64 = 0x243f6a8885a308d3;
        for b in block.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *b = (x >> 24) as u8;
        }
        let mut left = size;
        while left > 0 {
            let n = left.min(block.len() as u64) as usize;
            block[0] = (left >> 20) as u8; // defeat whole-file dedup across runs
            std::io::Write::write_all(&mut w, &block[..n]).unwrap();
            left -= n as u64;
        }
    }
    println!("{:.2} s", t.elapsed().as_secs_f64());

    let mib = |b: u64, s: f64| b as f64 / s / 1048576.0;
    let row = |what: &str, bytes: u64, secs: f64| {
        println!("  {what:<34} {secs:8.3} s   {:9.1} MiB/s", mib(bytes, secs));
    };

    let t = Instant::now();
    let p = ok!(s.put_file(&src, None, &Fault::default()));
    let put_s = t.elapsed().as_secs_f64();
    row("streaming put (hash+fsync)", size, put_s);
    let hex = p.obj.sha256.clone();

    let t = Instant::now();
    let n = ok!(s.read_into(&hex, &mut Counting(0)));
    row("verified read (record hit)", n, t.elapsed().as_secs_f64());

    let t = Instant::now();
    let mut got = 0u64;
    for i in 0..256u64 {
        let off = (i * 7_919_393) % (size - (1 << 20));
        got += ok!(s.read_range(&hex, off, 1 << 20)).len() as u64;
    }
    row("range reads (256 x 1 MiB)", got, t.elapsed().as_secs_f64());

    let t = Instant::now();
    ok!(s.hash_object(&hex));
    row("rehash (scrub rate)", size, t.elapsed().as_secs_f64());

    let t = Instant::now();
    let mut blob = vec![0u8; 4096];
    for i in 0..small {
        blob[0..8].copy_from_slice(&(i as u64).to_le_bytes());
        ok!(s.put_stream(&mut &blob[..], None, &Fault::default()));
    }
    let secs = t.elapsed().as_secs_f64();
    println!(
        "  {:<34} {secs:8.3} s   {:9.1} put/s   ({:.2} ms each)",
        format!("small puts ({small} x 4 KiB)"),
        small as f64 / secs,
        secs * 1000.0 / small as f64
    );
    println!("  peak RSS {:.1} MiB", rss_kb("VmHWM") as f64 / 1024.0);
    let _ = fs::remove_file(&src);
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

fn default_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vectors")
}

// ------------------------------------------------------------------ registry

fn cmd_registry() -> ExitCode {
    let seeds = registry::seeds();
    println!(
        "platform namespace: {} seed specs (identity = digest; names are aliases)",
        seeds.len()
    );
    for s in &seeds {
        let roles: Vec<&str> = s.spec.roles.iter().map(|(n, _)| n.as_str()).collect();
        println!(
            "  {:<28} {}  roles={:?} {}",
            s.alias,
            s.spec.object_id(),
            roles,
            match s.spec.require_executable() {
                Ok(v) => format!("vectors={}", v.id()),
                Err(e) => format!("VECTORLESS ({} — store/inspect only)", e.code.as_str()),
            }
        );
    }
    ExitCode::SUCCESS
}

// ------------------------------------------------------------------ corpus construction

fn seed_spec(alias: &str, nth: usize) -> EncodingSpec {
    registry::seeds()
        .into_iter()
        .filter(|s| s.alias == alias)
        .nth(nth)
        .expect("seed")
        .spec
}

/// A legal spec whose geometry overflows u64 on a modest logical shape: the corpus's
/// checked-arithmetic probe. Never a platform alias.
fn probe_spec() -> EncodingSpec {
    use tensorfs_core::relation::{Carrier, Dim, Relation};
    EncodingSpec {
        logical_dtypes: vec![Dtype::Bf16],
        logical_rank: Some(2),
        roles: vec![(
            "value".to_string(),
            tensorfs_core::spec::Role {
                carrier: Carrier::Set(vec![Dtype::U8]),
                shape: Relation::Dims(vec![
                    Dim::CeilBlock {
                        axis: 0,
                        block: 1,
                        mul: 9007199254740991,
                    },
                    Dim::Axis { axis: 1 },
                ]),
            },
        )],
        doc: Some(
            "limit probe: legal geometry that overflows checked 64-bit arithmetic".to_string(),
        ),
        vectors: None,
    }
}

fn pattern(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i * 31 + 7) as u8).collect()
}

fn part_of(dt: Dtype, shape: Vec<u64>) -> Part {
    let bytes = pattern(shape.iter().product::<u64>() as usize * dt.size() as usize);
    Part::plan(dt, shape, &bytes)
}

fn config_ref() -> ConfigRef {
    let bytes = br#"{"hidden_size":1536}"#;
    ConfigRef {
        sha256: tensorfs_core::sha256::hex_digest(bytes),
        length: bytes.len() as u64,
        media_type: "application/json".to_string(),
    }
}

fn encodings_of(specs: &[&EncodingSpec]) -> Vec<EncodingSpec> {
    let mut v: Vec<EncodingSpec> = specs.iter().map(|spec| (*spec).clone()).collect();
    v.sort_by_key(EncodingSpec::object_id);
    v
}

/// Positive header: plain inline tensor + an fp8-rowwise tensor (segments AND inline).
fn header_a(config: Option<ConfigRef>) -> (Header, Closure) {
    let plain = seed_spec("plain/1", 0);
    let fp8 = seed_spec("fp8-rowwise/1", 0);
    let mut closure = Closure::default();
    let h = Header {
        configs: config
            .map(|c| vec![("transformer".to_string(), c)])
            .unwrap_or_default(),
        assets: vec![(
            "tokenizer/vocab.txt".to_string(),
            Asset {
                logical_sha256: tensorfs_core::sha256::hex_digest(b"one\ntwo\n"),
                logical_length: 8,
                media_type: "text/plain".to_string(),
                segments: vec![ObjectRef::of(b"one\ntwo\n")],
            },
        )],
        encodings: encodings_of(&[&plain, &fp8]),
        components: vec![(
            "transformer".to_string(),
            vec![
                (
                    "blocks.0.attn.to_q.weight".to_string(),
                    Tensor {
                        dtype: Dtype::Bf16,
                        shape: vec![64, 128],
                        encoding: fp8.object_id(),
                        parts: vec![
                            ("data".to_string(), part_of(Dtype::F8E4M3FN, vec![64, 128])),
                            ("scale".to_string(), part_of(Dtype::F32, vec![64])),
                        ],
                    },
                ),
                (
                    "blocks.0.norm.weight".to_string(),
                    Tensor {
                        dtype: Dtype::Bf16,
                        shape: vec![128],
                        encoding: plain.object_id(),
                        parts: vec![("value".to_string(), part_of(Dtype::Bf16, vec![128]))],
                    },
                ),
            ],
        )],
    };
    closure.insert(plain);
    closure.insert(fp8);
    (h, closure)
}

/// Positive header: nvfp4 W4A4 base variant (packed nibble pairs + block scales + scalar).
fn header_nvfp4() -> (Header, EncodingSpec) {
    let spec = seed_spec("nvfp4-w4a4/1", 0);
    let h = Header {
        configs: vec![],
        assets: vec![],
        encodings: encodings_of(&[&spec]),
        components: vec![(
            "transformer".to_string(),
            vec![(
                "blocks.0.mlp.fc1.weight".to_string(),
                Tensor {
                    dtype: Dtype::Bf16,
                    shape: vec![256, 512],
                    encoding: spec.object_id(),
                    parts: vec![
                        ("weight".to_string(), part_of(Dtype::U8, vec![256, 256])),
                        (
                            "weight_scale".to_string(),
                            part_of(Dtype::F8E4M3FN, vec![256, 32]),
                        ),
                        ("weight_scale_2".to_string(), part_of(Dtype::F32, vec![])),
                    ],
                },
            )],
        )],
    };
    (h, spec)
}

fn manifest_checkpoint(header: &Header, extra_file: bool) -> TfsResult<Manifest> {
    let mut entries = vec![
        (
            "model.cozytensors".to_string(),
            Entry::CozyTensors(header.object_ref()?),
        ),
        (
            "tokenizer.model".to_string(),
            Entry::File(ObjectRef::of(&pattern(4096))),
        ),
        (
            "transformer/config.json".to_string(),
            Entry::File(ObjectRef::of(br#"{"hidden_size":1536}"#)),
        ),
        (
            "assets/prompt.txt".to_string(),
            Entry::File(ObjectRef::of(b"one\ntwo\n")),
        ),
    ];
    if extra_file {
        entries.push((
            "transformer/generation.json".to_string(),
            Entry::File(ObjectRef::of(br#"{"steps":30}"#)),
        ));
    }
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    Draft { entries }.seal()
}

fn profile_normal(allowed: Vec<ObjectRef>, dialect: &str) -> IngestProfile {
    IngestProfile {
        class: "normal".to_string(),
        source_dialect: dialect.to_string(),
        converter_build: tensorfs_core::ids::object_id(b"converter-build-v1"),
        converter_recipe: ObjectRef::of(b"recipe"),
        expected_topology_digest: tensorfs_core::ids::object_id(b"topology"),
        allowed_encodings: allowed,
        max_source_bytes: 1 << 40,
        max_tensors: 100_000,
    }
}

fn subject(profile: &IngestProfile, header: &Header, tenant: &str) -> TfsResult<IngestSubject> {
    Ok(IngestSubject {
        profile: profile.object_ref(),
        tenant: tenant.to_string(),
        sources: vec![(tensorfs_core::ids::object_id(b"hf://acme/model@rev1"), 1)],
        proposed_header: header.object_ref()?,
        candidates: vec![ObjectRef::of(b"candidate")],
    })
}

/// The corpus twin of what `ingest::stamp::stamp` mints. The vector exists so the LEDGER's
/// canonical bytes are frozen alongside the receipt that names them.
fn stamp_doc(
    prof: &IngestProfile,
    sub: &IngestSubject,
    snap: &Manifest,
    header: &Header,
) -> TfsResult<Stamp> {
    Ok(Stamp {
        subject: sub.object_ref(),
        profile: prof.object_ref(),
        header: header.object_ref()?,
        manifest: snap.object_ref(),
        topology_digest: header.topology_digest(),
        converter_build: prof.converter_build.clone(),
        converter_recipe: prof.converter_recipe.clone(),
        candidate_set: tensorfs_core::ingest::stamp::candidate_set_digest(
            &sub.candidates.iter().map(|c| c.id()).collect::<Vec<_>>(),
        ),
        golden: tensorfs_core::ingest::stamp::GOLDEN_PASS.to_string(),
        isolation_tier: 1,
        sources: sub.sources.clone(),
        accounting: vec![
            ("bytes_written".to_string(), 1 << 20),
            ("tensors".to_string(), 2),
        ],
    })
}

/// The layout plane's three frozen documents, all derived from the corpus header so the
/// vector and the live path cannot describe different things. The code topology is the
/// header's OWN topology (a contract that asks for exactly what is there is the Satisfied
/// case), and the registration is run, not asserted.
#[allow(clippy::type_complexity)]
fn layout_plane_docs(
    h: &Header,
    closure: &Closure,
) -> TfsResult<(
    CodeTopology,
    CompatibilityReport,
    ConversionRegistration,
    CompatibilityReport,
    CompatibilityReport,
    CodeTopology,
    CompatibilityReport,
)> {
    let code = CodeTopology {
        contract: tensorfs_core::ids::object_id(b"ModelConstructionContract:corpus"),
        components: h
            .components
            .iter()
            .map(|(c, ts)| {
                (
                    c.clone(),
                    ts.iter()
                        .map(|(k, t)| {
                            (
                                k.clone(),
                                Requirement {
                                    dtype: t.dtype,
                                    shape: t.shape.clone(),
                                    accepts: Accepts::Any,
                                },
                            )
                        })
                        .collect(),
                )
            })
            .collect(),
    };
    let pin: Vec<String> = h.encodings.iter().map(EncodingSpec::object_id).collect();
    let out = compat::check_compatibility(
        &code,
        h,
        closure,
        &pin,
        Lane::Canonical,
        &[],
        compat::Depth::Full,
    )?;
    let report = CompatibilityReport::of(&code, h, Lane::Canonical, &out)?;
    let reg = ConversionRegistration::register(
        "bf16-to-f32",
        Mapping::Widen {
            from: Dtype::Bf16,
            to: Dtype::F32,
        },
        &grade::complete_domain_16(),
    )?;

    // THE RED ARM (tfs-037), as a pair of frozen reports over ONE contract.
    //
    // The contract's first key asks for f8_e4m3fn where the checkpoint holds bf16 — a
    // downward cast, PRODUCIBLE on its own — and its last key asks for a shape the
    // checkpoint does not have, which is a different model and INCOMPATIBLE on its own.
    // Graded on the FIRST fact the pair reads PRODUCIBLE and a catalog would offer the
    // conversion; graded on the WHOLE set it is INCOMPATIBLE. Both reports are frozen with
    // their verdicts so the two answers cannot silently become one.
    let mut worst = code.clone();
    let keys: usize = worst.components.iter().map(|(_, k)| k.len()).sum();
    if keys < 2 {
        return Err(tensorfs_core::err::Refusal {
            code: tensorfs_core::err::Code::MISSING_FIELD,
            detail: "the worst-grade arm needs a contract with at least two keys".into(),
        });
    }
    let last = worst.components.len() - 1;
    worst.components[0].1[0].1.dtype = Dtype::F8E4M3FN;
    let tail = worst.components[last].1.len() - 1;
    worst.components[last].1[tail].1.shape = vec![7];

    let mut reports = Vec::new();
    for depth in [compat::Depth::Full, compat::Depth::FirstMismatch] {
        let out =
            compat::check_compatibility(&worst, h, closure, &pin, Lane::Canonical, &[], depth)?;
        reports.push(CompatibilityReport::of(&worst, h, Lane::Canonical, &out)?);
    }
    let first_fact = reports.pop().unwrap();
    let whole_set = reports.pop().unwrap();

    // THE SHAPE-BUCKET ARM (tfs-038). The same contract with every required dtype recast
    // and NOTHING else touched. Keys and shapes are untouched, so the shape-only projection
    // is byte-identical to the header's while the exact projection is not — which is the
    // whole point: an exact-index miss that is a shape-index HIT is the CONVERTIBLE and
    // PRODUCIBLE half of a catalog, and grading it is worth the microseconds. Frozen as a
    // pair so the day the shape projection stops eliding the dtype, or starts eliding
    // something else, one of these two digests moves and the corpus says so.
    let mut recast = code.clone();
    for (_, keys) in recast.components.iter_mut() {
        for (_, r) in keys.iter_mut() {
            r.dtype = Dtype::F8E4M3FN;
        }
    }
    let out = compat::check_compatibility(
        &recast,
        h,
        closure,
        &pin,
        Lane::Canonical,
        &[],
        compat::Depth::Full,
    )?;
    let recast_report = CompatibilityReport::of(&recast, h, Lane::Canonical, &out)?;
    Ok((
        code,
        report,
        reg,
        whole_set,
        first_fact,
        recast,
        recast_report,
    ))
}

fn receipt(
    sub: &IngestSubject,
    snap: &Manifest,
    header: &Header,
    stamp: &Stamp,
    signer: &str,
) -> TfsResult<IngestVerificationReceipt> {
    Ok(IngestVerificationReceipt {
        subject: sub.object_ref(),
        manifest: snap.object_ref(),
        header: header.object_ref()?,
        stamp: stamp.object_ref(),
        evidence: Evidence::ApprovedProducer {
            implementation: tensorfs_core::ids::object_id(b"converter-impl"),
            recipe: ObjectRef::of(b"recipe"),
        },
        signer: signer.to_string(),
        signature: "ed25519:AAAA".to_string(),
        observations: vec![
            ("source_bytes".to_string(), 1 << 20),
            ("tensors".to_string(), 2),
        ],
    })
}

// ------------------------------------------------------------------ vector files

struct Case {
    group: &'static str,
    name: String,
    doc: &'static str,
    bytes: Vec<u8>,
    expect: &'static str,
    code: Option<String>,
    note: String,
}

fn ok_case(group: &'static str, name: &str, doc: &'static str, bytes: Vec<u8>, note: &str) -> Case {
    Case {
        group,
        name: name.to_string(),
        doc,
        bytes,
        expect: "ok",
        code: None,
        note: note.to_string(),
    }
}

fn red_case(
    group: &'static str,
    name: &str,
    doc: &'static str,
    bytes: Vec<u8>,
    code: &str,
    note: &str,
) -> Case {
    Case {
        group,
        name: name.to_string(),
        doc,
        bytes,
        expect: "refuse",
        code: Some(code.to_string()),
        note: note.to_string(),
    }
}

fn sub(bytes: &[u8], from: &str, to: &str) -> Vec<u8> {
    let hay = String::from_utf8(bytes.to_vec()).expect("utf8");
    assert!(hay.contains(from), "surgery anchor {from:?} absent");
    hay.replacen(from, to, 1).into_bytes()
}

fn binary_sub(bytes: &[u8], from: &[u8], to: &[u8]) -> Vec<u8> {
    let at = bytes
        .windows(from.len())
        .position(|window| window == from)
        .unwrap_or_else(|| panic!("binary surgery anchor {from:?} absent"));
    let mut out = Vec::with_capacity(bytes.len() + to.len() - from.len());
    out.extend_from_slice(&bytes[..at]);
    out.extend_from_slice(to);
    out.extend_from_slice(&bytes[at + from.len()..]);
    out
}

fn build_corpus() -> TfsResult<(Vec<Case>, Vec<EncodingSpec>)> {
    let mut cases: Vec<Case> = Vec::new();
    let (ha, closure) = header_a(Some(config_ref()));
    let ha_bytes = ha.conformance_bytes()?;
    let (hn, _) = header_nvfp4();
    let plain = seed_spec("plain/1", 0);
    let fp8 = seed_spec("fp8-rowwise/1", 0);
    let mxfp8 = seed_spec("mxfp8/1", 0);
    let svdq = seed_spec("svdq-microscale/1", 0);
    // the corpus closure ships the WHOLE platform namespace, not only the cited specs
    let mut specs: Vec<EncodingSpec> = registry::seeds().into_iter().map(|s| s.spec).collect();
    for s in closure
        .specs
        .iter()
        .map(|(_, s)| s.clone())
        .chain([probe_spec()])
    {
        if !specs.iter().any(|x| x.object_id() == s.object_id()) {
            specs.push(s);
        }
    }

    // --- positives
    cases.push(ok_case(
        "header",
        "plain-fp8",
        "header",
        ha_bytes.clone(),
        "inline at the 256 B cap + segmented data",
    ));
    cases.push(ok_case(
        "header",
        "nvfp4-base",
        "header",
        hn.conformance_bytes()?,
        "nvfp4 W4A4 base variant, rank-0 scalar role",
    ));
    let (ha2, _) = header_a(None);
    cases.push(ok_case(
        "header",
        "plain-fp8-noconfig",
        "header",
        ha2.conformance_bytes()?,
        "same tensors, no configs: same topology, different header id",
    ));
    let mut reordered = ha.clone();
    reordered.components[0].1.reverse();
    cases.push(ok_case(
        "header",
        "plain-fp8-reordered",
        "header",
        reordered.conformance_bytes()?,
        "same topology and objects, opposite identity-bearing tensor order",
    ));
    cases.push(ok_case(
        "encoding",
        "plain",
        "encoding",
        plain.canonical_bytes(),
        "entry zero; carries vectors",
    ));
    cases.push(ok_case(
        "encoding",
        "mxfp8",
        "encoding",
        mxfp8.canonical_bytes(),
        "ceil-block scale relation; vectorless",
    ));
    cases.push(ok_case(
        "encoding",
        "svdq-microscale",
        "encoding",
        svdq.canonical_bytes(),
        "rank-7 permuted scale grid mined from v1",
    ));
    cases.push(ok_case(
        "encoding_vectors",
        "plain",
        "vectors",
        registry::plain_vectors().fixture_bytes(),
        "raw bit patterns: signed zero, NaN payload, infinity, rank-0",
    ));

    let snap = manifest_checkpoint(&ha, false)?;
    let snap2 = manifest_checkpoint(&ha, true)?;
    cases.push(ok_case(
        "manifest",
        "checkpoint",
        "manifest",
        snap.canonical_bytes(),
        "manifest identity is its exact virtual-tree bytes",
    ));
    cases.push(ok_case(
        "manifest",
        "checkpoint-extra-file",
        "manifest",
        snap2.canonical_bytes(),
        "one more sibling file: new manifest_id, same header id",
    ));

    let prof = profile_normal(
        vec![plain.object_ref(), fp8.object_ref()],
        "safetensors.diffusers",
    );
    let sub_doc = subject(&prof, &ha, "acme")?;
    let stmp = stamp_doc(&prof, &sub_doc, &snap, &ha)?;
    let rcpt = receipt(&sub_doc, &snap, &ha, &stmp, "platform-key-1")?;
    cases.push(ok_case(
        "ingest",
        "profile-normal",
        "profile",
        prof.canonical_bytes(),
        "normal profile over platform-aliased digests",
    ));
    cases.push(ok_case(
        "ingest",
        "subject",
        "subject",
        sub_doc.canonical_bytes(),
        "tenant + exact source generations",
    ));
    cases.push(ok_case(
        "ingest",
        "stamp",
        "stamp",
        stmp.canonical_bytes(),
        "the verified ledger: topology, candidate set, converter, golden, accounting",
    ));
    cases.push(ok_case(
        "ingest",
        "receipt",
        "receipt",
        rcpt.canonical_bytes(),
        "one evidence arm; signer in the envelope; binds its stamp",
    ));

    // --- the layout plane (tfs-009): code topology, report, registration
    let (code, report, reg, worst_full, worst_first, recast, recast_report) =
        layout_plane_docs(&ha, &closure)?;
    cases.push(ok_case(
        "layout",
        "code-topology",
        "code_topology",
        code.canonical_bytes(),
        "the bounded projection of a ModelConstructionContract",
    ));
    cases.push(ok_case(
        "layout",
        "report-satisfied",
        "compat_report",
        report.canonical_bytes(),
        "Satisfied, with the component-granular encoding facts grading needs",
    ));
    // tfs-037. Two reports, ONE contract, one checkpoint: the only difference is how far
    // the walk went, and it is the difference between offering a conversion and refusing
    // one. Their frozen `grade:` lines are what notices if the ranking path ever regresses
    // to reading the first fact.
    cases.push(ok_case(
        "layout",
        "report-worst-full",
        "compat_report",
        worst_full.canonical_bytes(),
        "a dtype cast FIRST and a shape mismatch SECOND: the whole set grades INCOMPATIBLE",
    ));
    cases.push(ok_case(
        "layout",
        "report-worst-first-fact",
        "compat_report",
        worst_first.canonical_bytes(),
        "the same pair stopped at the first fact: PRODUCIBLE, which is why ranking may not \
         stop there",
    ));
    cases.push(ok_case(
        "layout",
        "code-topology-recast",
        "code_topology",
        recast.canonical_bytes(),
        "the same contract with only the dtypes recast: a DIFFERENT exact topology digest \
         and the SAME shape digest",
    ));
    cases.push(ok_case(
        "layout",
        "report-recast",
        "compat_report",
        recast_report.canonical_bytes(),
        "an exact-index MISS that is a shape-index HIT: differs by dtype alone, so it is a \
         candidate to grade rather than a pair to skip",
    ));
    cases.push(red_case(
        "layout-red",
        "worst-graded-partial",
        "compat_report",
        sub(
            &worst_full.canonical_bytes(),
            "\"depth\":\"full\"",
            "\"depth\":\"first-mismatch\"",
        ),
        "PARTIAL_MISMATCH_SET",
        "a full mismatch set relabelled as a stopped walk: ranking refuses rather than \
         reading a truncated set",
    ));
    cases.push(red_case(
        "layout-red",
        "third-verdict",
        "compat_report",
        sub(
            &report.canonical_bytes(),
            "\"t\":\"satisfied\"",
            "\"t\":\"derivable\"",
        ),
        "UNKNOWN_FIELD",
        "a third satisfaction verdict below the hub",
    ));
    cases.push(red_case(
        "layout-red",
        "empty-accepts",
        "code_topology",
        sub(
            &code.canonical_bytes(),
            "\"accepts\":\"any\"",
            "\"accepts\":[]",
        ),
        "MISSING_FIELD",
        "an empty accepts list admits nothing and says nothing",
    ));
    // The grading plane's documents. The registration above was RUN, not asserted: these
    // bytes exist only because bf16 -> f32 -> bf16 came back identical over all 65,536
    // representable inputs, and freezing them puts that document's canonical form under
    // `tfs vectors verify` and the cross-language facade diff.
    cases.push(ok_case(
        "layout",
        "conversion-registration",
        "conversion_registration",
        reg.canonical_bytes(),
        "a lossless widening, proved over the complete 16-bit domain",
    ));
    for (name, note, graded) in [
        (
            "grade-compatible",
            "the contract asks for exactly what is there",
            grade::Grade::Compatible,
        ),
        (
            "grade-convertible",
            "a registered lossless cast names the registration that closes it",
            grade::Grade::Convertible {
                registration: reg.name.clone(),
                mapping: reg.mapping.describe(),
            },
        ),
        (
            "grade-producible",
            "the remedy names the CONTRACT's target, never the observed encoding",
            grade::Grade::Producible {
                target: "f8_e4m3fn at transformer/blocks.0.attn.to_q.weight".to_string(),
                source: "this checkpoint's bf16 tensors".to_string(),
            },
        ),
        (
            "grade-incompatible",
            "a different model, not a different packaging",
            grade::Grade::Incompatible {
                reason: "transformer/blocks.0.attn.to_q.weight: logical shapes disagree"
                    .to_string(),
            },
        ),
    ] {
        cases.push(ok_case(
            "layout",
            name,
            "grade_decision",
            grade::GradeDecision {
                report: report.object_id(),
                grade: graded,
            }
            .canonical_bytes(),
            note,
        ));
    }
    cases.push(red_case(
        "layout-red",
        "fifth-grade",
        "grade_decision",
        sub(
            &grade::GradeDecision {
                report: report.object_id(),
                grade: grade::Grade::Compatible,
            }
            .canonical_bytes(),
            "\"t\":\"COMPATIBLE\"",
            "\"t\":\"PROBABLY\"",
        ),
        "UNKNOWN_FIELD",
        "a fifth grade below the hub: the four levels are the closed set",
    ));
    // --- restricted deterministic CBOR representations (all red)
    let g = "canonical";
    cases.push(red_case(
        g,
        "old-json-header",
        "header",
        canon::write(&ha.diagnostic_value()?),
        "MALFORMED_CBOR",
        "the pre-hardcut JSON header has no reader",
    ));
    cases.push(red_case(
        g,
        "nonpreferred-root-length",
        "header",
        binary_sub(&ha_bytes, &[0x85], &[0x98, 0x05]),
        "NONCANONICAL_ENCODING",
        "root arity five encoded with an unnecessary width byte",
    ));
    cases.push(red_case(
        g,
        "indefinite-root",
        "header",
        binary_sub(&ha_bytes, &[0x85], &[0x9f]),
        "NONCANONICAL_ENCODING",
        "indefinite containers are outside the profile",
    ));
    cases.push(red_case(
        g,
        "forbidden-map",
        "header",
        binary_sub(&ha_bytes, &[0x85], &[0xa5]),
        "MALFORMED_CBOR",
        "maps are outside the all-array profile",
    ));
    cases.push(red_case(
        g,
        "forbidden-tag",
        "header",
        [vec![0xc0], ha_bytes.clone()].concat(),
        "MALFORMED_CBOR",
        "tags are outside the profile",
    ));
    cases.push(red_case(
        g,
        "forbidden-float",
        "header",
        [vec![0xfb], 1f64.to_be_bytes().to_vec()].concat(),
        "MALFORMED_CBOR",
        "floats are outside the profile",
    ));
    cases.push(red_case(
        g,
        "forbidden-null",
        "header",
        vec![0xf6],
        "MALFORMED_CBOR",
        "null is not admitted",
    ));
    cases.push(red_case(
        g,
        "truncated",
        "header",
        ha_bytes[..ha_bytes.len() - 1].to_vec(),
        "MALFORMED_CBOR",
        "the final byte is absent",
    ));
    cases.push(red_case(
        g,
        "trailing-bytes",
        "header",
        [ha_bytes.clone(), vec![0x80]].concat(),
        "TRAILING_BYTES",
        "a second value after the document",
    ));

    // --- header schema/geometry reds
    let g = "header-red";
    cases.push(red_case(
        g,
        "unknown-field",
        "header",
        binary_sub(&ha_bytes, &[0x85], &[0x86]),
        "WRONG_TYPE",
        "the fixed root has no sixth field",
    ));
    cases.push(red_case(
        g,
        "unknown-major",
        "header",
        binary_sub(&ha_bytes, b"cozytensors/1", b"cozytensors/2"),
        "UNKNOWN_FORMAT",
        "unknown format major",
    ));
    let mut h = ha.clone();
    if let Body::Segments(segments) = &mut h.components[0].1[0].1.parts[0].1.body {
        segments[0].length -= 1;
    }
    cases.push(red_case(
        g,
        "anchor-equation",
        "header",
        h.conformance_bytes()?,
        "BYTE_LENGTH_MISMATCH",
        "segment total off by one byte",
    ));
    let mut unknown_dtype = ha_bytes.clone();
    let dtype_at = unknown_dtype
        .windows(b"blocks.0.attn.to_q.weight".len())
        .position(|window| window == b"blocks.0.attn.to_q.weight")
        .unwrap()
        + b"blocks.0.attn.to_q.weight".len();
    unknown_dtype[dtype_at] = 12;
    cases.push(red_case(
        g,
        "dtype-outside-enum",
        "header",
        unknown_dtype,
        "DTYPE_UNKNOWN",
        "carrier dtype outside the closed enum",
    ));

    let mut h = ha.clone();
    h.components[0].1[0].1.parts.retain(|(n, _)| n != "scale");
    cases.push(red_case(
        g,
        "role-missing",
        "header",
        h.conformance_bytes()?,
        "ROLE_SET_MISMATCH",
        "spec role set is EXACT",
    ));
    let mut h = ha.clone();
    h.components[0].1[0]
        .1
        .parts
        .push(("input_scale".to_string(), part_of(Dtype::F32, vec![])));
    cases.push(red_case(
        g,
        "role-extra",
        "header",
        h.conformance_bytes()?,
        "ROLE_SET_MISMATCH",
        "an extra role is a different spec digest",
    ));
    let mut h = ha.clone();
    h.components[0].1[0].1.parts[1].1.shape = vec![32];
    h.components[0].1[0].1.parts[1].1.body = Body::Inline(pattern(128));
    cases.push(red_case(
        g,
        "relation-shape",
        "header",
        h.conformance_bytes()?,
        "SHAPE_MISMATCH",
        "scale shape != relation result",
    ));
    let mut h = ha.clone();
    h.components[0].1[1].1.shape = vec![0];
    cases.push(red_case(
        g,
        "zero-element",
        "header",
        h.conformance_bytes()?,
        "ZERO_ELEMENT",
        "zero-element tensor",
    ));
    let mut h = ha.clone();
    h.components[0].1[1].1.parts[0].1 = Part {
        dtype: Dtype::Bf16,
        shape: vec![128],
        body: Body::Segments(vec![ObjectRef::of(&pattern(256))]),
    };
    cases.push(red_case(
        g,
        "inline-threshold-under",
        "header",
        h.conformance_bytes()?,
        "INLINE_THRESHOLD",
        "256 B stored as a segment",
    ));
    let mut h = ha.clone();
    h.components[0].1[1].1.shape = vec![130];
    h.components[0].1[1].1.parts[0].1 = Part {
        dtype: Dtype::Bf16,
        shape: vec![130],
        body: Body::Inline(pattern(260)),
    };
    cases.push(red_case(
        g,
        "inline-threshold-over",
        "header",
        h.conformance_bytes()?,
        "INLINE_THRESHOLD",
        "260 B stored inline",
    ));
    let mut h = ha.clone();
    h.components[0].1[1].1.parts[0].1.body = Body::Segments(vec![]);
    cases.push(red_case(
        g,
        "empty-segments",
        "header",
        h.conformance_bytes()?,
        "COUNT_CAP",
        "segmented body is a non-empty ObjectRef array",
    ));
    let mut h = ha.clone();
    h.encodings.push(probe_spec());
    h.encodings.sort_by_key(EncodingSpec::object_id);
    cases.push(red_case(
        g,
        "uncited-encoding",
        "header",
        h.conformance_bytes()?,
        "UNCITED_ENCODING",
        "derivable-fact law",
    ));
    let mut h = ha_bytes.clone();
    let key = b"blocks.0.norm.weight";
    let at = h
        .windows(key.len())
        .position(|window| window == key)
        .unwrap()
        + key.len();
    h[at + 4] = 2;
    cases.push(red_case(
        g,
        "encoding-not-listed",
        "header",
        h,
        "ENCODING_MISMATCH",
        "cited digest missing from `encodings`",
    ));
    let mut h = ha.clone();
    h.components.clear();
    cases.push(red_case(
        g,
        "empty-components",
        "header",
        h.conformance_bytes()?,
        "EMPTY_COMPONENTS",
        "no components",
    ));

    // ---- tfs-015: hostile native bytes, keys, totals, capability ----
    cases.push(red_case(
        g,
        "digest-31-bytes",
        "header",
        binary_sub(&ha_bytes, &[0x58, 0x20], &[0x58, 0x1f]),
        "MALFORMED_DIGEST",
        "a digest is exactly 32 native bytes",
    ));
    cases.push(red_case(
        g,
        "digest-33-bytes",
        "header",
        binary_sub(&ha_bytes, &[0x58, 0x20], &[0x58, 0x21]),
        "MALFORMED_DIGEST",
        "a digest is exactly 32 native bytes",
    ));
    let mut h = ha.clone();
    h.configs.push(("aaa".to_string(), config_ref()));
    cases.push(red_case(
        g,
        "configs-unsorted",
        "header",
        h.conformance_bytes()?,
        "SORT_ORDER",
        "configs are a sorted set, not construction order",
    ));
    let mut h = ha.clone();
    h.encodings.reverse();
    cases.push(red_case(
        g,
        "encodings-unsorted",
        "header",
        h.conformance_bytes()?,
        "SORT_ORDER",
        "EncodingSpec references sort by raw digest",
    ));

    // Tensor keys are data-shaped and therefore the hostile-input surface.
    for (name, key, code, note) in [
        ("key-empty", "", "NON_ASCII_FIELD", "the empty key"),
        (
            "key-non-ascii",
            "blocks.0.attn.tö_q.weight",
            "NON_ASCII_FIELD",
            "a non-ASCII tensor key",
        ),
    ] {
        let mut header = ha.clone();
        header.components[0].1[0].0 = key.to_string();
        cases.push(red_case(
            g,
            name,
            "header",
            header.conformance_bytes()?,
            code,
            note,
        ));
    }

    // A zero-length segment is a fetch that can never satisfy the anchor equation.
    let mut h = ha.clone();
    if let Body::Segments(segments) = &mut h.components[0].1[0].1.parts[0].1.body {
        segments[0].length = 0;
    }
    cases.push(red_case(
        g,
        "segment-zero-length",
        "header",
        h.conformance_bytes()?,
        "ZERO_ELEMENT",
        "an empty segment record",
    ));

    // Whole-DOCUMENT byte total, not just this part's.
    let huge = Header {
        configs: vec![],
        assets: vec![],
        encodings: encodings_of(&[&plain]),
        components: vec![(
            "transformer".to_string(),
            vec![(
                "huge.weight".to_string(),
                Tensor {
                    dtype: Dtype::F64,
                    shape: vec![1 << 48],
                    encoding: plain.object_id(),
                    parts: vec![(
                        "value".to_string(),
                        Part {
                            dtype: Dtype::F64,
                            shape: vec![1 << 48],
                            body: Body::Segments(vec![ObjectRef {
                                sha256: "0".repeat(64),
                                length: 1u64 << 51,
                            }]),
                        },
                    )],
                },
            )],
        )],
    };
    cases.push(red_case(
        g,
        "total-bytes-cap",
        "header",
        huge.conformance_bytes()?,
        "TOTAL_BYTES_CAP",
        "one legal tensor over the whole-document byte total",
    ));

    // A spec object present in the closure that this runtime's pin does not alias: valid
    // DATA, unsupported EXECUTION. `header` accepts the same bytes; only serving refuses.
    let foreign = {
        let mut s = seed_spec("plain/1", 0);
        s.doc = Some("a spec object no platform alias names".to_string());
        s
    };
    let hf = Header {
        configs: vec![],
        assets: vec![],
        encodings: encodings_of(&[&foreign]),
        components: vec![(
            "transformer".to_string(),
            vec![(
                "blocks.0.norm.weight".to_string(),
                Tensor {
                    dtype: Dtype::Bf16,
                    shape: vec![128],
                    encoding: foreign.object_id(),
                    parts: vec![("value".to_string(), part_of(Dtype::Bf16, vec![128]))],
                },
            )],
        )],
    };
    cases.push(ok_case(
        g,
        "unaliased-spec-stores",
        "header",
        hf.conformance_bytes()?,
        "a pin-absent spec object validates as data",
    ));
    cases.push(red_case(
        g,
        "unaliased-spec-serving",
        "header-serving",
        hf.conformance_bytes()?,
        "UNKNOWN_TO_LOCAL_CAPABILITY",
        "the same bytes refused for SERVING against the local pin",
    ));
    specs.push(foreign);
    let mut h = hn.clone();
    h.components[0].1[0].1.shape = vec![256, 511];
    cases.push(red_case(
        g,
        "division-remainder",
        "header",
        h.conformance_bytes()?,
        "DIVISION_REMAINDER",
        "in/2 with an odd in",
    ));
    let mut h = ha.clone();
    h.components[0].1[1].1.shape = vec![67108864, 67108864];
    cases.push(red_case(
        g,
        "element-count-cap",
        "header",
        h.conformance_bytes()?,
        "COUNT_CAP",
        "aggregate element limit",
    ));
    // the element cap fires before any product can overflow, so the checked-arithmetic arm
    // is reached through a legal-but-extreme relation instead (probe spec, in the closure)
    let probe = probe_spec();
    let h = Header {
        configs: vec![],
        assets: vec![],
        encodings: encodings_of(&[&probe]),
        components: vec![(
            "transformer".to_string(),
            vec![(
                "probe.weight".to_string(),
                Tensor {
                    dtype: Dtype::Bf16,
                    shape: vec![4096, 8],
                    encoding: probe.object_id(),
                    parts: vec![(
                        "value".to_string(),
                        Part {
                            dtype: Dtype::U8,
                            shape: vec![1],
                            body: Body::Inline(vec![0u8]),
                        },
                    )],
                },
            )],
        )],
    };
    cases.push(red_case(
        g,
        "arith-overflow",
        "header",
        h.conformance_bytes()?,
        "ARITH_OVERFLOW",
        "checked 64-bit geometry arithmetic in the relation evaluator",
    ));
    let mut h = ha.clone();
    h.components[0].1[1].1.dtype = Dtype::I8;
    cases.push(red_case(
        g,
        "logical-dtype-carrier",
        "header",
        h.conformance_bytes()?,
        "DTYPE_MISMATCH",
        "carrier no longer equals the logical dtype",
    ));

    // --- location-typed Manifest reds
    let manifest_bytes = snap.canonical_bytes();
    cases.push(red_case(
        "manifest-red",
        "entry-kind-symlink",
        "manifest",
        sub(&manifest_bytes, "\"kind\":\"file\"", "\"kind\":\"symlink\""),
        "UNKNOWN_FIELD",
        "only file and cozytensors entries exist",
    ));
    let mut draft = Draft::of(&snap);
    draft.entries.clear();
    cases.push(red_case(
        "manifest-red",
        "entries-empty",
        "manifest",
        draft.canonical_bytes(),
        "MISSING_FIELD",
        "a manifest lists at least one path",
    ));
    let mut draft = Draft::of(&snap);
    draft.entries.push((
        "../escape".to_string(),
        Entry::File(ObjectRef::of(b"escape")),
    ));
    draft.entries.sort_by(|left, right| left.0.cmp(&right.0));
    cases.push(red_case(
        "manifest-red",
        "path-dotdot",
        "manifest",
        draft.canonical_bytes(),
        "PATH_ILLEGAL",
        "dot components are unrepresentable",
    ));
    let mut draft = Draft::of(&snap);
    draft.entries.push((
        "model2.cozytensors".to_string(),
        Entry::CozyTensors(ha.object_ref()?),
    ));
    draft.entries.sort_by(|left, right| left.0.cmp(&right.0));
    cases.push(red_case(
        "manifest-red",
        "cozytensors-twice",
        "manifest",
        draft.canonical_bytes(),
        "ATTACHMENT_CARDINALITY",
        "one manifest carries at most one CozyTensors header",
    ));

    // --- the border's own documents (tfs-003)
    let g = "ingest";
    let session = tensorfs_core::ingest::transaction::IngestSession {
        session: "e3b0c44298fc1c14".to_string(),
        tenant: "acme".to_string(),
        candidates: vec![ObjectRef::of(b"a candidate header")],
    };
    cases.push(ok_case(
        g,
        "ingest-session",
        "session",
        session.canonical_bytes(),
        "a temporary session: a candidate is never a release",
    ));

    // --- ingest reds
    let g = "ingest-red";
    let mut future = session.clone();
    future.session = "not a session name".to_string();
    cases.push(red_case(
        g,
        "session-grammar",
        "session",
        future.canonical_bytes(),
        "KEY_GRAMMAR",
        "a session id is an ASCII name, never free text",
    ));
    let refused_dialect = profile_normal(vec![plain.object_ref()], "pickle");
    cases.push(red_case(
        g,
        "pickle-normal-profile",
        "profile",
        refused_dialect.canonical_bytes(),
        "PICKLE_REFUSED",
        "pickle always refuses normal ingest",
    ));
    let foreign = profile_normal(
        vec![ObjectRef::of(b"a spec the platform never aliased")],
        "safetensors.diffusers",
    );
    cases.push(red_case(
        g,
        "unregistered-encoding",
        "profile",
        foreign.canonical_bytes(),
        "UNREGISTERED_ENCODING",
        "launch border admits platform-aliased digests only",
    ));
    let two_arms = sub(
        &rcpt.canonical_bytes(),
        "\"evidence\":[",
        "\"evidence\":[{\"report\":{\"length\":6,\"sha256\":\"0000000000000000000000000000000000000000000000000000000000000000\"},\"t\":\"independent_numerical\",\"verifier_build\":\"sha256:0000000000000000000000000000000000000000000000000000000000000000\"},",
    );
    cases.push(red_case(
        g,
        "two-evidence-arms",
        "receipt",
        two_arms,
        "EVIDENCE_ARM_AMBIGUOUS",
        "exactly one evidence arm",
    ));

    // --- one-byte mutants of every positive
    let positives: Vec<(String, &'static str, Vec<u8>, String)> = cases
        .iter()
        .filter(|c| c.expect == "ok")
        .map(|c| {
            (
                format!("{}-{}", c.group, c.name),
                c.doc,
                c.bytes.clone(),
                tensorfs_core::ids::object_id(&c.bytes),
            )
        })
        .collect();
    for (name, doc, bytes, id) in positives {
        let mut m = bytes.clone();
        let off = m.len() / 3;
        m[off] ^= 0x01;
        cases.push(Case {
            group: "onebyte",
            name,
            doc,
            bytes: m,
            expect: "mutant",
            code: None,
            note: format!("one byte flipped at offset {off} of {id}"),
        });
    }

    Ok((cases, specs))
}

// ------------------------------------------------------------------ gen

fn cmd_gen(dir: &Path) -> ExitCode {
    let (cases, specs) = match build_corpus() {
        Ok(corpus) => corpus,
        Err(error) => return bail(error),
    };
    // Regenerate only what the corpus OWNS. `mined/` (producer-mined vectors) and
    // `structure/` (banked fixtures) are evidence with their own provenance and their own
    // banking commands; a corpus rebuild must never be able to delete them.
    let _ = fs::remove_dir_all(dir.join("cases"));
    let _ = fs::remove_dir_all(dir.join("closure"));
    fs::create_dir_all(dir.join("closure")).unwrap();
    for s in &specs {
        let id = s.object_id();
        fs::write(
            dir.join("closure").join(format!("{}.cbor", &id[7..19])),
            s.canonical_bytes(),
        )
        .unwrap();
    }
    let mut n = 0;
    for c in &cases {
        let d = dir.join("cases").join(c.group);
        fs::create_dir_all(&d).unwrap();
        let extension = if matches!(c.doc, "header" | "header-serving" | "encoding") {
            "cbor"
        } else {
            "json"
        };
        fs::write(d.join(format!("{}.{}", c.name, extension)), &c.bytes).unwrap();
        let mut v = format!("doc: {}\nexpect: {}\n", c.doc, c.expect);
        if let Some(code) = &c.code {
            v += &format!("code: {code}\n");
        }
        if c.expect == "ok" {
            v += &format!("id: {}\n", tensorfs_core::ids::object_id(&c.bytes));
            if c.doc == "header" {
                let h = Header::parse(&c.bytes).unwrap();
                v += &format!("topology: {}\n", h.topology_digest());
                v += &format!("shape: {}\n", h.topology_shape_digest());
            }
            // tfs-038: the code side freezes BOTH projections too, so a dtype-only recast
            // of one contract is provably the same shape bucket and a different exact one.
            if c.doc == "code_topology" {
                let code = tensorfs_core::compat::CodeTopology::parse(&c.bytes).unwrap();
                v += &format!("topology: {}\n", code.topology_digest());
                v += &format!("shape: {}\n", code.topology_shape_digest());
            }
            if c.doc == "manifest" {
                let m = Manifest::parse(&c.bytes).unwrap();
                v += &format!("manifest: {}\n", m.manifest_id());
            }
            // The hub verdict a report earns, frozen alongside its bytes: the corpus holds
            // the DECISION, not merely the canonical form the decision was read from.
            if c.doc == "compat_report" {
                let closure = Closure::default();
                let outcome = tensorfs_core::corpus::run_doc(
                    c.doc,
                    &c.bytes,
                    &closure,
                    &registry::platform_digests(),
                )
                .unwrap();
                v += &format!("grade: {}\n", outcome.grade.unwrap());
            }
        }
        v += &format!("note: {}\n", c.note);
        fs::write(d.join(format!("{}.verdict", c.name)), v).unwrap();
        n += 1;
    }
    println!(
        "wrote {n} vectors and {} closure spec objects to {}",
        specs.len(),
        dir.display()
    );
    ExitCode::SUCCESS
}

// ------------------------------------------------------------------ verify

enum Outcome {
    Ok {
        id: String,
        topology: Option<String>,
        shape: Option<String>,
        manifest: Option<String>,
        grade: Option<String>,
    },
    Refused(Refusal),
}

fn run_doc(doc: &str, bytes: &[u8], closure: &Closure, platform: &[String]) -> Outcome {
    // ONE dispatch, shared with the Python facade's harness (tensorfs_core::corpus).
    match tensorfs_core::corpus::run_doc(doc, bytes, closure, platform) {
        Ok(o) => Outcome::Ok {
            id: o.id,
            topology: o.topology,
            shape: o.shape,
            manifest: o.manifest,
            grade: o.grade,
        },
        Err(e) => Outcome::Refused(e),
    }
}

fn read_verdict(p: &Path) -> Vec<(String, String)> {
    fs::read_to_string(p)
        .unwrap()
        .lines()
        .filter_map(|l| {
            l.split_once(": ")
                .map(|(a, b)| (a.to_string(), b.to_string()))
        })
        .collect()
}

fn field(v: &[(String, String)], k: &str) -> Option<String> {
    v.iter().find(|(a, _)| a == k).map(|(_, b)| b.clone())
}

fn cmd_verify(dir: &Path) -> ExitCode {
    let started = Instant::now();
    let mut closure = Closure::default();
    let mut closure_ok = true;
    for e in fs::read_dir(dir.join("closure")).expect("closure dir") {
        let p = e.unwrap().path();
        let bytes = fs::read(&p).unwrap();
        match EncodingSpec::decode_nested(&bytes) {
            Ok(s) => closure.insert(s),
            Err(err) => {
                println!("FAIL closure/{}: {err}", p.display());
                closure_ok = false;
            }
        }
    }
    let platform = registry::platform_digests();

    let mut groups: Vec<PathBuf> = fs::read_dir(dir.join("cases"))
        .expect("cases dir")
        .map(|e| e.unwrap().path())
        .collect();
    groups.sort();
    let (mut pass, mut fail) = (0, 0);
    let mut ids: Vec<(String, Outcome)> = Vec::new();

    for g in groups {
        let mut files: Vec<PathBuf> = fs::read_dir(&g)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| {
                p.extension()
                    .map(|x| x == "json" || x == "cbor")
                    .unwrap_or(false)
            })
            .collect();
        files.sort();
        for f in files {
            let name = format!(
                "{}/{}",
                g.file_name().unwrap().to_string_lossy(),
                f.file_stem().unwrap().to_string_lossy()
            );
            let v = read_verdict(&f.with_extension("verdict"));
            let bytes = fs::read(&f).unwrap();
            let doc = field(&v, "doc").expect("doc kind");
            let expect = field(&v, "expect").expect("expect");
            let outcome = run_doc(&doc, &bytes, &closure, &platform);
            let verdict = match (expect.as_str(), &outcome) {
                (
                    "ok",
                    Outcome::Ok {
                        id,
                        topology,
                        shape,
                        manifest,
                        grade,
                    },
                ) => {
                    let mut bad = Vec::new();
                    if field(&v, "id").as_deref() != Some(id.as_str()) {
                        bad.push(format!("id {id} != frozen {:?}", field(&v, "id")));
                    }
                    if let Some(t) = field(&v, "topology") {
                        if topology.as_deref() != Some(t.as_str()) {
                            bad.push(format!("topology {topology:?} != frozen {t}"));
                        }
                    }
                    if let Some(t) = field(&v, "shape") {
                        if shape.as_deref() != Some(t.as_str()) {
                            bad.push(format!("shape {shape:?} != frozen {t}"));
                        }
                    }
                    if let Some(expected) = field(&v, "manifest") {
                        if manifest.as_deref() != Some(expected.as_str()) {
                            bad.push(format!("manifest {manifest:?} != frozen {expected}"));
                        }
                    }
                    if let Some(expected) = field(&v, "grade") {
                        if grade.as_deref() != Some(expected.as_str()) {
                            bad.push(format!("grade {grade:?} != frozen {expected}"));
                        }
                    }
                    if bad.is_empty() {
                        Ok(format!("id={}", &id[7..19]))
                    } else {
                        Err(bad.join("; "))
                    }
                }
                ("ok", Outcome::Refused(e)) => Err(format!("expected ok, refused {e}")),
                ("refuse", Outcome::Refused(e)) => {
                    let want = field(&v, "code").unwrap_or_default();
                    if e.code.as_str() == want {
                        Ok(format!("{} ({})", e.code.as_str(), truncate(&e.detail, 72)))
                    } else {
                        Err(format!(
                            "refused {} but the frozen code is {want}",
                            e.code.as_str()
                        ))
                    }
                }
                ("refuse", Outcome::Ok { .. }) => Err(format!(
                    "expected {} but it passed",
                    field(&v, "code").unwrap_or_default()
                )),
                ("mutant", Outcome::Refused(e)) => Ok(format!("refused {}", e.code.as_str())),
                ("mutant", Outcome::Ok { id, .. }) => {
                    let from = field(&v, "note").unwrap_or_default();
                    if from.contains(id.as_str()) {
                        Err("a flipped byte kept the same object id".to_string())
                    } else {
                        Ok(format!("distinct id={}", &id[7..19]))
                    }
                }
                (e, _) => Err(format!("unknown expectation {e:?}")),
            };
            match verdict {
                Ok(d) => {
                    pass += 1;
                    println!("PASS {name:<44} {expect:<7} {d}");
                }
                Err(d) => {
                    fail += 1;
                    println!("FAIL {name:<44} {expect:<7} {d}");
                }
            }
            ids.push((name, outcome));
        }
    }

    // cross-vector identity assertions
    let get = |n: &str| -> Option<&Outcome> { ids.iter().find(|(k, _)| k == n).map(|(_, o)| o) };
    let idof = |n: &str| -> String {
        match get(n) {
            Some(Outcome::Ok { id, .. }) => id.clone(),
            _ => String::new(),
        }
    };
    let topof = |n: &str| -> String {
        match get(n) {
            Some(Outcome::Ok {
                topology: Some(t), ..
            }) => t.clone(),
            _ => String::new(),
        }
    };
    let shapeof = |n: &str| -> String {
        match get(n) {
            Some(Outcome::Ok { shape: Some(s), .. }) => s.clone(),
            _ => String::new(),
        }
    };
    let cpof = |n: &str| -> String {
        match get(n) {
            Some(Outcome::Ok {
                manifest: Some(c), ..
            }) => c.clone(),
            _ => String::new(),
        }
    };
    let short = |value: String| value.chars().take(19).collect::<String>();
    let mut cross: Vec<(String, bool, String)> = Vec::new();
    cross.push((
        "manifest_id == manifest_id (unconditional)".into(),
        cpof("manifest/checkpoint") == idof("manifest/checkpoint")
            && !cpof("manifest/checkpoint").is_empty(),
        cpof("manifest/checkpoint"),
    ));
    cross.push((
        "header ObjectId != manifest_id".into(),
        idof("header/plain-fp8") != cpof("manifest/checkpoint"),
        format!(
            "{} vs {}",
            short(idof("header/plain-fp8")),
            short(cpof("manifest/checkpoint"))
        ),
    ));
    cross.push((
        "a config change flips header id, not topology_digest".into(),
        idof("header/plain-fp8") != idof("header/plain-fp8-noconfig")
            && topof("header/plain-fp8") == topof("header/plain-fp8-noconfig"),
        topof("header/plain-fp8"),
    ));
    cross.push((
        "tensor reorder flips header id, not topology_digest".into(),
        idof("header/plain-fp8") != idof("header/plain-fp8-reordered")
            && topof("header/plain-fp8") == topof("header/plain-fp8-reordered"),
        topof("header/plain-fp8-reordered"),
    ));
    // tfs-038's red arm, as a CROSS assertion over two frozen documents: recasting only the
    // dtypes must move the exact digest and leave the shape digest exactly where it was. If
    // the shape projection ever stopped eliding the dtype the first half fails; if it began
    // eliding a shape the second half fails.
    cross.push((
        "a dtype-only recast flips topology_digest, not the shape digest".into(),
        topof("layout/code-topology") != topof("layout/code-topology-recast")
            && shapeof("layout/code-topology") == shapeof("layout/code-topology-recast")
            && !shapeof("layout/code-topology").is_empty(),
        shapeof("layout/code-topology-recast"),
    ));
    cross.push((
        "the shape projection is not the exact one".into(),
        topof("layout/code-topology") != shapeof("layout/code-topology")
            && !shapeof("layout/code-topology").is_empty(),
        short(shapeof("layout/code-topology")),
    ));
    cross.push((
        "one more sibling file flips manifest_id only".into(),
        cpof("manifest/checkpoint") != cpof("manifest/checkpoint-extra-file"),
        cpof("manifest/checkpoint-extra-file"),
    ));
    let (cha, _) = header_a(Some(config_ref()));
    let csnap = ok!(manifest_checkpoint(&cha, false));
    let cprof = profile_normal(
        vec![seed_spec("plain/1", 0).object_ref()],
        "safetensors.diffusers",
    );
    let csub = ok!(subject(&cprof, &cha, "acme"));
    let cstamp = ok!(stamp_doc(&cprof, &csub, &csnap, &cha));
    let crcpt = ok!(receipt(&csub, &csnap, &cha, &cstamp, "platform-key-1"));
    cross.push((
        "receipt binds its exact subject/manifest/header/stamp".into(),
        crcpt.binds(&csub, &csnap, &cha, &cstamp).is_ok(),
        String::new(),
    ));
    let other = ok!(subject(&cprof, &cha, "other-tenant"));
    let replay = crcpt.binds(&other, &csnap, &cha, &cstamp);
    cross.push((
        "cross-subject replay refuses".into(),
        replay.as_ref().err().map(|e| e.code)
            == Some(tensorfs_core::err::Code::CROSS_SUBJECT_REPLAY),
        replay.err().map(|e| e.detail).unwrap_or_default(),
    ));
    let seeds_match = registry::seeds()
        .iter()
        .all(|s| closure.get(&s.spec.object_id()).is_some());
    cross.push((
        "every seed digest resolves in the corpus closure".into(),
        seeds_match,
        String::new(),
    ));

    for (what, ok, detail) in &cross {
        if *ok {
            pass += 1;
            println!("PASS cross: {what:<52} {detail}");
        } else {
            fail += 1;
            println!("FAIL cross: {what:<52} {detail}");
        }
    }

    println!(
        "\n{pass} pass, {fail} fail, {} closure specs, {:.3}s",
        closure.specs.len(),
        started.elapsed().as_secs_f64()
    );
    if fail == 0 && closure_ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n])
    }
}

// ------------------------------------------------------------------ id

fn cmd_id(p: &Path) -> ExitCode {
    let bytes = match fs::read(p) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{}: {e}", p.display());
            return ExitCode::FAILURE;
        }
    };
    println!("object_id: {}", tensorfs_core::ids::object_id(&bytes));
    println!("length:    {}", bytes.len());
    if let Ok(h) = Header::parse(&bytes) {
        println!("kind:      cozytensors/1 header");
        println!("topology:  {}", h.topology_digest());
        println!("tensors:   {}", h.tensors().count());
        match h.diagnostic_value() {
            Ok(value) => println!("pretty:\n{}", canon::pretty(&value, 0)),
            Err(error) => return bail(error),
        }
    } else if let Ok(m) = Manifest::parse(&bytes) {
        println!("kind:      manifest");
        println!("manifest_id: {}", m.manifest_id());
    } else {
        println!(
            "kind:      NOT a tensorfs document — {}",
            Header::parse(&bytes).err().unwrap()
        );
    }
    ExitCode::SUCCESS
}

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
                match canon::parse_with_depth(&bytes, tensorfs_core::limits::DOC_MAX_BYTES, 16) {
                    Ok(value) => value,
                    Err(error) => return bail(error),
                };
            let header = match Header::from_diagnostic_value(&value) {
                Ok(header) => header,
                Err(error) => return bail(error),
            };
            if let Err(error) = header.validate_structure() {
                return bail(error);
            }
            let encoded = match header.canonical_bytes() {
                Ok(bytes) => bytes,
                Err(error) => return bail(error),
            };
            if let Err(error) = Header::parse(&encoded) {
                return bail(error);
            }
            write_cli(&encoded, flags)
        }
        "decode" => {
            let header = match Header::parse(&bytes) {
                Ok(header) => header,
                Err(error) => return bail(error),
            };
            let value = match header.diagnostic_value() {
                Ok(value) => value,
                Err(error) => return bail(error),
            };
            let mut rendered = canon::pretty(&value, 0).into_bytes();
            rendered.push(b'\n');
            write_cli(&rendered, flags)
        }
        "check" => {
            let header = match Header::parse(&bytes) {
                Ok(header) => header,
                Err(error) => return bail(error),
            };
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

fn cmd_cbor_plant(path: &Path, kind: &str, flags: &Flags) -> ExitCode {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("{}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let mut header = match Header::parse(&bytes) {
        Ok(header) => header,
        Err(error) => return bail(error),
    };
    match kind {
        "anchor-length" => {
            let segment = header
                .components
                .iter_mut()
                .flat_map(|(_, tensors)| tensors.iter_mut())
                .flat_map(|(_, tensor)| tensor.parts.iter_mut())
                .find_map(|(_, part)| match &mut part.body {
                    Body::Segments(segments) => segments.first_mut(),
                    Body::Inline(_) => None,
                });
            match segment {
                Some(segment) if segment.length > 1 => segment.length -= 1,
                _ => {
                    eprintln!("REFUSED MISSING_TENSOR: header has no plantable segment");
                    return ExitCode::FAILURE;
                }
            }
        }
        "asset-logical-digest" => match header.assets.first_mut() {
            Some((_, asset)) => {
                asset.logical_sha256 = tensorfs_core::sha256::hex_digest(b"wrong asset bytes")
            }
            None => {
                eprintln!("REFUSED MISSING_FIELD: header has no model asset");
                return ExitCode::FAILURE;
            }
        },
        other => {
            eprintln!("REFUSED UNKNOWN_FIELD: unknown CBOR plant {other:?}");
            return ExitCode::FAILURE;
        }
    }
    match header.conformance_bytes() {
        Ok(bytes) => write_cli(&bytes, flags),
        Err(error) => bail(error),
    }
}

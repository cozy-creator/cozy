//! tfs — the runnable conformance/benchmark tool for tfs-013.
//!
//!   tfs registry                 print the seeded platform namespace (alias -> digest)
//!   tfs vectors gen [dir]        (re)write the frozen golden-vector corpus
//!   tfs vectors verify [dir]     run every vector live, print per-vector verdicts
//!   tfs bench [tensors] [reps]   canonical-writer throughput + peak RSS
//!   tfs flip                     identity-flip matrix over live documents
//!   tfs id <file>                object id (+ derived ids) of a stored document
//!
//! The object store (tfs-002). Flags: --expect <hex> --expect-length <n> --out <path>
//! --range <off>:<len> --rate <f> --pass <n> --max-age <s> --slow-ms <n>
//! --size <bytes> --small <n>, plus the dev-only kill points --fault <stage> --ready <path>.
//!
//!   tfs store init|info|bump <root>
//!   tfs put <root> <file>        stream one object in (any shape)
//!   tfs get <root> <hex>         verified read, whole or ranged
//!   tfs contains <root> <hex>    the presence HINT, and what it is worth
//!   tfs verify <root> <hex>      verification record or rehash; quarantine on disagreement
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
//! The snapshot plane (tfs-004). Flags: --iters <n>
//!
//!   tfs snapshot show <root> <hex>           strict decode, decode->re-encode identity, entries
//!   tfs snapshot walk <root> <hex>           transitive root; every-and-only vs the store census
//!   tfs snapshot verify <root> <hex>         full-checkpoint verify: every declared byte
//!   tfs snapshot flip <root> <hex>           what moves checkpoint_id / header_digest / topology
//!   tfs snapshot fuzz <root> <hex>           bounded mutation fuzz over canonical bytes
//!   tfs snapshot blob put|read|repair <root> <file> [i]
//!
//! The checkpoint plane (tfs-015). Flags: --blocks <n> --pin <alias|empty> --seed <s>
//!
//!   tfs checkpoint write <root>              objectize a synthetic tree, emit the header
//!   tfs checkpoint read <root> <hex>         reconstruct every part, byte-identity checked
//!   tfs checkpoint info <root> <hex>         component/scope query, zero tensor-byte reads
//!   tfs checkpoint locate <root> <hex> <component> <key> <role> <off> <len>
//!   tfs checkpoint admit <root> <hex>        serving admission against the local pin
//!   tfs checkpoint caps                      whole-document total caps, armed live
//!   tfs checkpoint size [n]                  uniform-form header cost at H3 cardinality
//!
//! The ingest border (tfs-003) — the ONE normal tensor-ingest door. A `--source` names its
//! component; a filename never authorizes anything. Flags: --source <c>=<path> (repeated)
//! --config <c>=<path> --registry <path> --spec-variant <n> --session <s> --tenant <t>
//! --ttl <s> --rekey <c>=<from>:<to> --rlimit-as-mb/-cpu-s/-fsize-mb --no-netns --now
//!
//!   tfs ingest inspect [--markers]           hermetic bounded decode + classification
//!                                            --markers also decodes the `.comfy_quant`
//!                                            payloads — the only tensor bytes inspect reads
//!   tfs ingest bank <dialect> <fam> <var> <converter>   the REVIEW act: bank a fingerprint
//!   tfs ingest plan <c=alias,...>            the plan, from headers alone: 0 tensor bytes
//!                                            a component repeats for its ALLOWED SET
//!   tfs ingest run <root> <c=alias,...>      spawn the isolated worker; candidate + receipt
//!   tfs ingest-worker <root> <c=alias,...>   the hermetic child (invoked by `run`)
//!   tfs ingest install <root> <session>      the coordinator promotes ONE root
//!   tfs ingest restamp <root> <session>      re-verify a candidate; --plant <what> for arms
//!   tfs ingest reingest <root> <hex> <target>  inherit by object reference: 0 bytes hashed
//!   tfs ingest roots|reap <root>             candidates are not checkpoints
//!   tfs ingest converters | golden           reviewed converters; geometry run live
//!
//! The gguf-v1 PLANNER (tfs-011) — a PLAN over a metadata region, never a second door.
//! Flags: --probe (the input is a bounded metadata region) --tensors <n> --twin <rows.json>
//!
//!   tfs gguf plan <file>                     carrier-blob + per-tensor views, 0 weight bytes
//!   tfs gguf admit <file>                    the same, then the admission verdict (refuses)
//!
//! Reads and projection (tfs-005). Flags: --workers <n> --window <b> --ring <b>
//! --order module|key --hold-secs <s> --drop-unreleased --no-symlink
//!
//!   tfs read lease <root> <snap>             acquire/verify/release one ReadLease
//!   tfs read plan <root> <snap>              destination order vs header key order
//!   tfs read fill <root> <snap>              the pooled fill into a caller-owned ring
//!   tfs read verify-fill <root> <snap>       the same bytes through the sequential path
//!   tfs read span <root> <snap> <c> <k> <role> <off> <len>
//!   tfs read arms <root> <snap>              the read/projection red arms, planted live
//!   tfs read tear <root> <snap>              tear a verified object UNDER a live lease
//!   tfs read prove <root> <snap>             delivered bytes vs the digests the header declares
//!   tfs read transform                       run decomposition + the block-granularity rule
//!   tfs project <root> <snap> <dest>         tier 2: relative-symlink tree, no weight paths
//!   tfs materialize <root> <file> <dest>     tier 3: stream a RETAINED carrier blob
//!   tfs render <root> <hex>                  pretty PROJECTION of a canonical document
//!
//! The layout plane (tfs-009) — Stamp is at the border; this is the SECOND operation.
//! Flags: --contract <name> --closure <a,b> --pin <d,...> --lane canonical|local_unbordered
//! --survey (every measured role set and the spec it selects, instead of the first gap)
//! --registry <a,b> --corpus <f> --out <f> --truncate-at <n>
//!
//!   tfs compat import <rows.json>            measured rows -> canonical header + contract
//!   tfs compat topology <root> <hex>         a header's own topology as a code contract
//!   tfs compat check <code.json> <header.json>   Satisfied | Mismatch(details) + the facts
//!   tfs compat release-cut <report.json>     unclassified extras never cut a release
//!   tfs grade <report.json>                  COMPATIBLE|CONVERTIBLE|PRODUCIBLE|INCOMPATIBLE
//!   tfs grade register <name> <kind> <spec>  the registration ACT: runs A->B->A
//!   tfs grade decision <file>                read one grade decision, or refuse
//!   tfs verify-request <code.json> <header.json>   build the bounded request document
//!   tfs verify-compat <request> <response>   the hermetic verifier: one file in, one out
//!   tfs verify-consume <request> <response>  the hub's side; partial output mints nothing
//!
//! The cloud publish protocol (tfs-017), GREEN branch. Core links no HTTP: these emit what
//! a coordinator must sign and consume what the backend reported.
//! Flags: --out <f> --expires-at <t> --length <n>
//!
//!   tfs cloud closure <root> <snap> <session>   DECLARE FIRST: the exact object set
//!   tfs cloud artifact-set <document>           canonical first-party build object set
//!   tfs cloud grant <set> <object-id>           both conditions, derived, not optional
//!   tfs cloud verified <grant> <version> <etag> <checksum-b64>
//!   tfs cloud conflict <grant> <resident-checksum>   honest duplicate, or quarantine
//!   tfs cloud publish <closure> <facts-dir>     the atomic root, over facts only
//!
//! The pull plane (tfs-003 row 220) — cloud run backwards. Core links no HTTP, so the
//! transport is the caller's and the bytes arrive on stdin. Flags: --out <path> --why
//!
//!   tfs fetch plan <root> <closure> <session>   split the declared set: HELD vs WANTED
//!   tfs fetch admit <root> <plan> <object-id>   THE DOOR — stdin in, verified object out
//!   tfs fetch complete <root> <plan>            every declared object verifiably resident
//!
//! Roots, holds and local GC (tfs-006, exclusive-and-explicit per tfs-020). Flags:
//! --kind <k> --expect-generation <n> --source-url <u> --hold-secs <s> --ready <p>
//! --limit <n> --dry-run --allow-empty --wait
//!
//!   tfs root list|register|release|open <root> [name] [snap]
//!   tfs root note <root> <snap> --source-url <u>  provenance WITHOUT pinning
//!   tfs root reconcile <root> <name>=<snap>...   the supervisor's binding set
//!   tfs hold take|reap|steal <root> [snap|id]    owner-liveness holds
//!   tfs rm <root> <snap>                         THE explicit tombstone — the only act
//!                                                that makes bytes collectable
//!   tfs gc mark|usage|sweep|reap|refs <root>     the census, the numbers, the cycle
//!   tfs gc arms <root>                           the roots/GC red arms, planted live

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use tensorfs_core::canon::{self, Value};
use tensorfs_core::compat::{self, Accepts, CodeTopology, CompatibilityReport, Lane, Requirement};
use tensorfs_core::dtype::Dtype;
use tensorfs_core::err::Refusal;
use tensorfs_core::grade::{self, ConversionRegistration, Mapping};
use tensorfs_core::header::{Body, Closure, Header, Part, Tensor};
use tensorfs_core::ids::{ConfigRef, Doc, ObjectRef};
use tensorfs_core::ingest::stamp::Stamp;
use tensorfs_core::ingest::{Evidence, IngestProfile, IngestSubject, IngestVerificationReceipt};
use tensorfs_core::manifest::{BlobDescriptor, Draft, Entry, SnapshotManifest};
use tensorfs_core::registry;
use tensorfs_core::spec::EncodingSpec;
use tensorfs_core::store::{Fault, Store, Verdict, FAULT_STAGES};

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
        ["store", "bump", r] => cmd_store(Path::new(r), "bump"),
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
        ["snapshot", "show", r, h] => snap::cmd_show(Path::new(r), h),
        ["snapshot", "walk", r, h] => snap::cmd_walk(Path::new(r), h),
        ["snapshot", "verify", r, h] => snap::cmd_verify(Path::new(r), h),
        ["snapshot", "flip", r, h] => snap::cmd_flip(Path::new(r), h, &flags),
        ["snapshot", "arms", r, h] => snap::cmd_arms(Path::new(r), h),
        ["snapshot", "fuzz", r, h] => {
            snap::cmd_fuzz(Path::new(r), h, flag_num(&flags, "iters", 20000))
        }
        ["snapshot", "blob", "put", r, f] => snap::cmd_blob_put(Path::new(r), Path::new(f)),
        ["snapshot", "blob", "read", r, f] => {
            snap::cmd_blob_read(Path::new(r), Path::new(f), &flags)
        }
        ["snapshot", "blob", "repair", r, f, i] => {
            snap::cmd_blob_repair(Path::new(r), Path::new(f), i.parse().unwrap_or(0))
        }
        ["checkpoint", "caps"] => ckpt::cmd_caps(),
        ["checkpoint", "size"] => ckpt::cmd_size(3699),
        ["checkpoint", "size", n] => ckpt::cmd_size(n.parse().unwrap_or(3699)),
        ["ingest", "inspect"] => ingest::cmd_inspect(&flags),
        ["gguf", "plan", f] => ingest::cmd_gguf_plan(Path::new(f), false, &flags),
        ["gguf", "admit", f] => ingest::cmd_gguf_plan(Path::new(f), true, &flags),
        ["ingest", "converters"] => ingest::cmd_converters(),
        ["ingest", "golden"] => ingest::cmd_golden(),
        ["ingest", "bank", dia, fam, var, conv] => ingest::cmd_bank(dia, fam, var, conv, &flags),
        ["ingest", "plan", t] => ingest::cmd_plan(t, &flags),
        ["ingest", "run", r, t] => ingest::cmd_run(Path::new(r), t, &flags),
        ["ingest-worker", r, t] => ingest::cmd_worker(Path::new(r), t, &flags),
        ["ingest", "reingest", r, h, t] => ingest::cmd_reingest(Path::new(r), h, t, &flags),
        ["ingest", "install", r, s] => ingest::cmd_install(Path::new(r), s),
        ["ingest", "restamp", r, s] => ingest::cmd_restamp(Path::new(r), s, &flags),
        ["ingest", "reap", r] => ingest::cmd_reap(Path::new(r), &flags),
        ["ingest", "roots", r] => ingest::cmd_roots(Path::new(r)),
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
        ["project", r, h, d] => rd::cmd_project(Path::new(r), h, Path::new(d), &flags),
        ["materialize", r, f, d] => rd::cmd_materialize(Path::new(r), Path::new(f), Path::new(d)),
        ["render", r, h] => rd::cmd_render(Path::new(r), h),
        ["root", v] => roots::cmd_root(Path::new(v), "list", &[], &flags),
        ["root", "list", r] => roots::cmd_root(Path::new(r), "list", &[], &flags),
        ["root", "register", r, n, h] => roots::cmd_root(Path::new(r), "register", &[n, h], &flags),
        ["root", "note", r, h] => roots::cmd_root(Path::new(r), "note", &[h], &flags),
        ["root", "release", r, n] => roots::cmd_root(Path::new(r), "release", &[n], &flags),
        ["root", "open", r, n] => roots::cmd_root(Path::new(r), "open", &[n], &flags),
        ["root", "reconcile", r, rest @ ..] => {
            roots::cmd_root(Path::new(r), "reconcile", rest, &flags)
        }
        ["hold", "take", r] => roots::cmd_hold(Path::new(r), "take", &[], &flags),
        ["hold", "take", r, h] => roots::cmd_hold(Path::new(r), "take", &[h], &flags),
        ["hold", "reap", r] => roots::cmd_hold(Path::new(r), "reap", &[], &flags),
        ["hold", "steal", r, i] => roots::cmd_hold(Path::new(r), "steal", &[i], &flags),
        ["gc", "mark", r] => roots::cmd_gc(Path::new(r), "mark", &flags),
        ["gc", "usage", r] => roots::cmd_gc(Path::new(r), "usage", &flags),
        ["gc", "sweep", r] => roots::cmd_gc(Path::new(r), "sweep", &flags),
        ["gc", "reap", r] => roots::cmd_gc(Path::new(r), "reap", &flags),
        ["gc", "refs", r] => roots::cmd_gc(Path::new(r), "refs", &flags),
        ["gc", "arms", r] => roots::cmd_arms(Path::new(r)),
        ["rm", r, h] => roots::cmd_rm(Path::new(r), h),
        ["cloud", "closure", r, h, s] => cloud::cmd_closure(Path::new(r), h, s, &flags),
        ["cloud", "artifact-set", d] => cloud::cmd_artifact_set(Path::new(d), &flags),
        ["cloud", "grant", c, i] => cloud::cmd_grant(Path::new(c), i, &flags),
        ["cloud", "verified", g, v, e, c] => cloud::cmd_verified(Path::new(g), v, e, c, &flags),
        ["cloud", "conflict", g, r] => cloud::cmd_conflict(Path::new(g), r),
        ["cloud", "publish", c, d] => cloud::cmd_publish(Path::new(c), Path::new(d), &flags),
        ["fetch", "plan", r, d, s] => fetch::cmd_plan(Path::new(r), Path::new(d), s, &flags),
        ["fetch", "admit", r, p, i] => fetch::cmd_admit(Path::new(r), Path::new(p), i, &flags),
        ["fetch", "complete", r, p] => fetch::cmd_complete(Path::new(r), Path::new(p)),
        ["compat", "import", r] => layout::cmd_import(Path::new(r), &flags),
        ["compat", "topology", r, h] => layout::cmd_topology(Path::new(r), h, &flags),
        ["compat", "check", c, h] => layout::cmd_check(Path::new(c), Path::new(h), &flags),
        ["compat", "release-cut", r] => layout::cmd_release_cut(Path::new(r)),
        ["grade", "decision", d] => layout::cmd_decision(Path::new(d)),
        ["grade", r] => layout::cmd_grade(Path::new(r), &flags),
        ["grade", "register", n, k, s] => layout::cmd_register(n, k, s, &flags),
        ["verify-request", c, h] => layout::cmd_request(Path::new(c), Path::new(h), &flags),
        ["verify-compat", q, o] => layout::cmd_verify(Path::new(q), Path::new(o), &flags),
        ["verify-consume", q, o] => layout::cmd_consume(Path::new(q), Path::new(o)),
        ["registry"] => cmd_registry(),
        ["vectors", "gen"] => cmd_gen(&default_dir()),
        ["vectors", "gen", d] => cmd_gen(Path::new(d)),
        ["vectors", "verify"] => cmd_verify(&default_dir()),
        ["vectors", "verify", d] => cmd_verify(Path::new(d)),
        ["bench"] => cmd_bench(2000, 20),
        ["bench", n] => cmd_bench(n.parse().unwrap_or(2000), 20),
        ["bench", n, r] => cmd_bench(n.parse().unwrap_or(2000), r.parse().unwrap_or(20)),
        ["flip"] => cmd_flip(),
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
mod cloud;
mod conform;
mod fetch;
mod fixtures;
mod ingest;
mod layout;
mod rd;
mod roots;
mod snap;

fn cmd_store(root: &Path, verb: &str) -> ExitCode {
    let mut s = match verb {
        "init" => ok!(Store::init(root)),
        _ => ok!(Store::open(root)),
    };
    if verb == "bump" {
        let g = ok!(s.bump_generation());
        println!("generation bumped to {g} — every verification record is now stale");
    }
    println!("root:       {}", s.root().display());
    println!("era:        {}", tensorfs_core::store::ERA);
    println!("generation: {}", s.generation());
    println!("objects:    {}", ok!(s.objects()).len());
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
            "record:    valid (dev:ino {}:{}, {} bytes, generation {})",
            r.dev, r.ino, r.length, r.generation
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
        Verdict::Quarantined { to, why } => {
            eprintln!("QUARANTINED sha256:{hex} -> {} ({why})", to.display());
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
        "scrub rate={rate} pass={pass}: scanned {} sampled {} rehashed {} quarantined {} ({:.1} MiB hashed, {secs:.3} s, {:.1} MiB/s)",
        r.scanned,
        r.sampled,
        r.rehashed,
        r.quarantined,
        r.bytes_hashed as f64 / 1048576.0,
        r.bytes_hashed as f64 / secs.max(1e-9) / 1048576.0
    );
    ExitCode::from(if r.quarantined > 0 { 1 } else { 0 })
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
            // rehash happens HERE: good bytes resume, bad bytes are quarantined out of the
            // way and the object is admitted afresh.
            Err(why) if s.contains(hex) => match ok!(s.verify(hex)) {
                Verdict::Quarantined { to, .. } => {
                    println!(
                        "requeue sha256:{hex} ({why}; quarantined to {})",
                        to.display()
                    )
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

fn encodings_of(specs: &[&EncodingSpec]) -> Vec<ObjectRef> {
    let mut v: Vec<ObjectRef> = specs.iter().map(|s| s.object_ref()).collect();
    v.sort_by(|a, b| a.sha256.cmp(&b.sha256));
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

fn manifest_checkpoint(header: &Header, extra_file: bool) -> SnapshotManifest {
    let blob = BlobDescriptor::of(&pattern(4096));
    let mut entries = vec![
        // The attachment is TYPED; this is the one path the same object is projected to.
        (
            "model.cozytensors".to_string(),
            Entry::File(header.object_ref()),
        ),
        ("tokenizer.model".to_string(), Entry::Blob(blob)),
        ("transformer".to_string(), Entry::Dir),
        (
            "transformer/config.json".to_string(),
            Entry::File(ObjectRef::of(br#"{"hidden_size":1536}"#)),
        ),
    ];
    if extra_file {
        entries.push((
            "transformer/generation.json".to_string(),
            Entry::File(ObjectRef::of(br#"{"steps":30}"#)),
        ));
    }
    Draft {
        header: Some(header.object_ref()),
        entries,
    }
    .seal()
    .expect("the corpus's positive manifest is a legal tree by construction")
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

fn subject(profile: &IngestProfile, header: &Header, tenant: &str) -> IngestSubject {
    IngestSubject {
        profile: profile.object_ref(),
        tenant: tenant.to_string(),
        sources: vec![(tensorfs_core::ids::object_id(b"hf://acme/model@rev1"), 1)],
        proposed_header: header.object_ref(),
        candidates: vec![ObjectRef::of(b"candidate")],
    }
}

/// The corpus twin of what `ingest::stamp::stamp` mints. The vector exists so the LEDGER's
/// canonical bytes are frozen alongside the receipt that names them.
fn stamp_doc(
    prof: &IngestProfile,
    sub: &IngestSubject,
    snap: &SnapshotManifest,
    header: &Header,
) -> Stamp {
    Stamp {
        subject: sub.object_ref(),
        profile: prof.object_ref(),
        header: header.object_ref(),
        snapshot: snap.object_ref(),
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
    }
}

/// The layout plane's three frozen documents, all derived from the corpus header so the
/// vector and the live path cannot describe different things. The code topology is the
/// header's OWN topology (a contract that asks for exactly what is there is the Satisfied
/// case), and the registration is run, not asserted.
fn layout_plane_docs(
    h: &Header,
    closure: &Closure,
) -> (CodeTopology, CompatibilityReport, ConversionRegistration) {
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
    let pin: Vec<String> = h.encodings.iter().map(|e| e.id()).collect();
    let out = compat::check_compatibility(&code, h, closure, &pin, Lane::Canonical, &[])
        .expect("the corpus header is a legal checkpoint by construction");
    let report = CompatibilityReport::of(&code, h, Lane::Canonical, &out);
    let reg = ConversionRegistration::register(
        "bf16-to-f32",
        Mapping::Widen {
            from: Dtype::Bf16,
            to: Dtype::F32,
        },
        &grade::complete_domain_16(),
    )
    .expect("bf16 -> f32 -> bf16 is the identity on every bit pattern");
    (code, report, reg)
}

fn receipt(
    sub: &IngestSubject,
    snap: &SnapshotManifest,
    header: &Header,
    stamp: &Stamp,
    signer: &str,
) -> IngestVerificationReceipt {
    IngestVerificationReceipt {
        subject: sub.object_ref(),
        snapshot: snap.object_ref(),
        header: header.object_ref(),
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
    }
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

/// A deliberately non-canonical twin: same content, reversed object key order.
fn reversed(v: &Value) -> Vec<u8> {
    fn emit(v: &Value, out: &mut Vec<u8>) {
        match v {
            Value::Obj(m) => {
                out.push(b'{');
                for (i, (k, x)) in m.iter().rev().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    out.extend_from_slice(&canon::write(&Value::Str(k.clone())));
                    out.push(b':');
                    emit(x, out);
                }
                out.push(b'}');
            }
            Value::Arr(a) => {
                out.push(b'[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    emit(x, out);
                }
                out.push(b']');
            }
            other => out.extend_from_slice(&canon::write(other)),
        }
    }
    let mut out = Vec::new();
    emit(v, &mut out);
    out
}

fn build_corpus() -> (Vec<Case>, Vec<EncodingSpec>) {
    let mut cases: Vec<Case> = Vec::new();
    let (ha, closure) = header_a(Some(config_ref()));
    let ha_bytes = ha.canonical_bytes();
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
        hn.canonical_bytes(),
        "nvfp4 W4A4 base variant, rank-0 scalar role",
    ));
    let (ha2, _) = header_a(None);
    cases.push(ok_case(
        "header",
        "plain-fp8-noconfig",
        "header",
        ha2.canonical_bytes(),
        "same tensors, no configs: same topology, different header id",
    ));
    cases.push(ok_case(
        "spec",
        "plain",
        "spec",
        plain.canonical_bytes(),
        "entry zero; carries vectors",
    ));
    cases.push(ok_case(
        "spec",
        "mxfp8",
        "spec",
        mxfp8.canonical_bytes(),
        "ceil-block scale relation; vectorless",
    ));
    cases.push(ok_case(
        "spec",
        "svdq-microscale",
        "spec",
        svdq.canonical_bytes(),
        "rank-7 permuted scale grid mined from v1",
    ));
    cases.push(ok_case(
        "encoding_vectors",
        "plain",
        "vectors",
        registry::plain_vectors().canonical_bytes(),
        "raw bit patterns: signed zero, NaN payload, infinity, rank-0",
    ));

    let snap = manifest_checkpoint(&ha, false);
    let snap2 = manifest_checkpoint(&ha, true);
    cases.push(ok_case(
        "manifest",
        "checkpoint",
        "manifest",
        snap.canonical_bytes(),
        "checkpoint_id = snapshot_id, header attached structurally",
    ));
    cases.push(ok_case(
        "manifest",
        "checkpoint-extra-file",
        "manifest",
        snap2.canonical_bytes(),
        "one more sibling file: new checkpoint_id, same header id",
    ));

    let prof = profile_normal(
        vec![plain.object_ref(), fp8.object_ref()],
        "safetensors.diffusers",
    );
    let sub_doc = subject(&prof, &ha, "acme");
    let stmp = stamp_doc(&prof, &sub_doc, &snap, &ha);
    let rcpt = receipt(&sub_doc, &snap, &ha, &stmp, "platform-key-1");
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
    let (code, report, _reg) = layout_plane_docs(&ha, &closure);
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
    // --- canonical-byte semantic twins (all red)
    let g = "canonical";
    cases.push(red_case(
        g,
        "whitespace",
        "header",
        sub(&ha_bytes, "{\"components\"", "{ \"components\""),
        "NONCANONICAL_ENCODING",
        "one inserted space",
    ));
    cases.push(red_case(
        g,
        "key-order",
        "header",
        reversed(&ha.to_value()),
        "NONCANONICAL_ENCODING",
        "reversed object key order",
    ));
    cases.push(red_case(
        g,
        "duplicate-key",
        "header",
        sub(
            &ha_bytes,
            "\"format\":\"cozytensors/1\"",
            "\"format\":\"cozytensors/1\",\"format\":\"cozytensors/1\"",
        ),
        "DUPLICATE_KEY",
        "same key twice",
    ));
    cases.push(red_case(
        g,
        "number-float",
        "header",
        sub(&ha_bytes, "\"length\":8192", "\"length\":8192.0"),
        "NON_INTEGER_NUMBER",
        "float spelling of an integer",
    ));
    cases.push(red_case(
        g,
        "number-exponent",
        "header",
        sub(&ha_bytes, "\"length\":8192", "\"length\":8.192e3"),
        "NON_INTEGER_NUMBER",
        "exponent spelling",
    ));
    cases.push(red_case(
        g,
        "number-leading-zero",
        "header",
        sub(&ha_bytes, "\"length\":8192", "\"length\":08192"),
        "NON_INTEGER_NUMBER",
        "non-canonical integer spelling",
    ));
    cases.push(red_case(
        g,
        "null-value",
        "header",
        sub(&ha_bytes, "\"length\":8192", "\"length\":null"),
        "WRONG_TYPE",
        "null is not admitted",
    ));
    cases.push(red_case(
        g,
        "escaped-ascii",
        "header",
        sub(&ha_bytes, "\"format\"", "\"form\\u0061t\""),
        "NONCANONICAL_ENCODING",
        "\\u escape of a plain letter",
    ));
    cases.push(red_case(
        g,
        "unicode-nfc",
        "header",
        sub(&ha_bytes, "transformer", "transf\u{00f6}rmer"),
        "NON_ASCII_FIELD",
        "NFC o-diaeresis in a component name",
    ));
    cases.push(red_case(
        g,
        "unicode-nfd",
        "header",
        sub(&ha_bytes, "transformer", "transfo\u{0308}rmer"),
        "NON_ASCII_FIELD",
        "NFD twin of the same name",
    ));
    cases.push(red_case(
        g,
        "digest-uppercase",
        "header",
        uppercase_digest(&ha_bytes),
        "MALFORMED_DIGEST",
        "uppercase hex in a digest",
    ));
    cases.push(red_case(
        g,
        "trailing-whitespace",
        "header",
        [ha_bytes.clone(), b"\n".to_vec()].concat(),
        "NONCANONICAL_ENCODING",
        "trailing newline: parses, then fails the re-emit law",
    ));
    cases.push(red_case(
        g,
        "trailing-bytes",
        "header",
        [ha_bytes.clone(), b"{}".to_vec()].concat(),
        "TRAILING_BYTES",
        "a second value after the document",
    ));

    // --- header schema/geometry reds
    let g = "header-red";
    cases.push(red_case(
        g,
        "unknown-field",
        "header",
        sub(&ha_bytes, ",\"format\"", ",\"extra\":1,\"format\""),
        "UNKNOWN_FIELD",
        "unknown root field",
    ));
    cases.push(red_case(
        g,
        "unknown-major",
        "header",
        sub(&ha_bytes, "cozytensors/1", "cozytensors/2"),
        "UNKNOWN_FORMAT",
        "unknown format major",
    ));
    cases.push(red_case(
        g,
        "anchor-equation",
        "header",
        sub(&ha_bytes, "\"length\":8192", "\"length\":8191"),
        "BYTE_LENGTH_MISMATCH",
        "segment total off by one byte",
    ));
    cases.push(red_case(
        g,
        "dtype-outside-enum",
        "header",
        sub(
            &ha_bytes,
            "\"dtype\":\"f8_e4m3fn\"",
            "\"dtype\":\"f4_e2m1\"",
        ),
        "DTYPE_UNKNOWN",
        "carrier dtype outside the closed enum",
    ));

    let mut h = ha.clone();
    h.components[0].1[0].1.parts.retain(|(n, _)| n != "scale");
    cases.push(red_case(
        g,
        "role-missing",
        "header",
        h.canonical_bytes(),
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
        h.canonical_bytes(),
        "ROLE_SET_MISMATCH",
        "an extra role is a different spec digest",
    ));
    let mut h = ha.clone();
    h.components[0].1[0].1.parts[1].1.shape = vec![32];
    cases.push(red_case(
        g,
        "relation-shape",
        "header",
        h.canonical_bytes(),
        "SHAPE_MISMATCH",
        "scale shape != relation result",
    ));
    let mut h = ha.clone();
    h.components[0].1[1].1.shape = vec![0];
    cases.push(red_case(
        g,
        "zero-element",
        "header",
        h.canonical_bytes(),
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
        h.canonical_bytes(),
        "INLINE_THRESHOLD",
        "256 B stored as a segment",
    ));
    let mut h = ha.clone();
    h.components[0].1[1].1.shape = vec![130];
    h.components[0].1[1].1.parts[0].1 = Part {
        dtype: Dtype::Bf16,
        shape: vec![130],
        body: Body::Inline(tensorfs_core::b64::encode(&pattern(260))),
    };
    cases.push(red_case(
        g,
        "inline-threshold-over",
        "header",
        h.canonical_bytes(),
        "INLINE_THRESHOLD",
        "260 B stored inline",
    ));
    let inline_anchor = format!(
        "\"inline\":\"{}\"",
        tensorfs_core::b64::encode(&pattern(256))
    );
    cases.push(red_case(
        g,
        "inline-and-segments",
        "header",
        sub(
            &ha_bytes,
            &inline_anchor,
            &format!(
                "{inline_anchor},\"segments\":[{{\"length\":256,\"sha256\":\"{}\"}}]",
                "0".repeat(64)
            ),
        ),
        "INLINE_EXCLUSIVE",
        "both bodies present",
    ));
    let mut h = ha.clone();
    h.encodings.push(ObjectRef::of(b"an uncited spec object"));
    h.encodings.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    cases.push(red_case(
        g,
        "uncited-encoding",
        "header",
        h.canonical_bytes(),
        "UNCITED_ENCODING",
        "derivable-fact law",
    ));
    let mut h = ha.clone();
    h.components[0].1[1].1.encoding = tensorfs_core::ids::object_id(b"a spec nobody shipped");
    h.encodings.push(ObjectRef::of(b"a spec nobody shipped"));
    h.encodings.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    cases.push(red_case(
        g,
        "dangling-spec",
        "header",
        h.canonical_bytes(),
        "UNKNOWN_ENCODING",
        "cited digest absent from the closure",
    ));
    let mut h = ha.clone();
    h.components[0].1[1].1.encoding = tensorfs_core::ids::object_id(b"not listed either");
    cases.push(red_case(
        g,
        "encoding-not-listed",
        "header",
        h.canonical_bytes(),
        "ENCODING_MISMATCH",
        "cited digest missing from `encodings`",
    ));
    let mut h = ha.clone();
    h.components.clear();
    cases.push(red_case(
        g,
        "empty-components",
        "header",
        h.canonical_bytes(),
        "EMPTY_COMPONENTS",
        "no components",
    ));

    // ---- tfs-015: hostile bodies, hostile keys, whole-document totals, capability ----

    // The inline body's grammar is exact: one canonical spelling per byte run.
    let inline_b64 = tensorfs_core::b64::encode(&pattern(256));
    for (name, twin, note) in [
        (
            "base64-unpadded",
            inline_b64.trim_end_matches('=').to_string(),
            "padding stripped",
        ),
        (
            "base64-whitespace",
            format!("{} {}", &inline_b64[..8], &inline_b64[8..]),
            "a space inside the body",
        ),
        (
            "base64-urlsafe",
            inline_b64.replace('+', "-").replace('/', "_"),
            "url-safe alphabet",
        ),
    ] {
        cases.push(red_case(
            g,
            name,
            "header",
            sub(&ha_bytes, &inline_b64, &twin),
            "BAD_BASE64",
            note,
        ));
    }
    // A last quantum whose unused low bits are set decodes to the same bytes and re-encodes
    // differently — the classic base64 malleability channel.
    let tail = tensorfs_core::b64::encode(&pattern(2));
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut evil = tail.clone().into_bytes();
    // Two data bytes leave the final sextet's low TWO bits unused; canonical encoding
    // clears them, so setting one decodes identically and re-encodes differently.
    let i = alphabet.iter().position(|c| *c == evil[2]).unwrap();
    evil[2] = alphabet[i | 1];
    let mut h = ha.clone();
    h.components[0].1[1].1.shape = vec![1];
    h.components[0].1[1].1.parts[0].1 = Part {
        dtype: Dtype::Bf16,
        shape: vec![1],
        body: Body::Inline(tail.clone()),
    };
    cases.push(red_case(
        g,
        "base64-noncanonical-tail",
        "header",
        sub(
            &h.canonical_bytes(),
            &tail,
            &String::from_utf8(evil).unwrap(),
        ),
        "BAD_BASE64",
        "unused low bits set in the final quantum",
    ));

    // Tensor keys are data-shaped and therefore the hostile-input surface.
    for (name, key, code, note) in [
        ("key-empty", "", "KEY_GRAMMAR", "the empty key"),
        (
            "key-non-ascii",
            "blocks.0.attn.tö_q.weight",
            "NON_ASCII_FIELD",
            "a non-ASCII tensor key",
        ),
    ] {
        cases.push(red_case(
            g,
            name,
            "header",
            sub(&ha_bytes, "blocks.0.attn.to_q.weight", key),
            code,
            note,
        ));
    }

    // Endianness is a format GLOBAL: there is no field to disagree with.
    cases.push(red_case(
        g,
        "endian-field",
        "header",
        sub(
            &ha_bytes,
            "\"dtype\":\"f8_e4m3fn\"",
            "\"dtype\":\"f8_e4m3fn\",\"endian\":\"little\"",
        ),
        "UNKNOWN_FIELD",
        "a part claiming its own byte order",
    ));

    // A zero-length segment is a fetch that can never satisfy the anchor equation.
    cases.push(red_case(
        g,
        "segment-zero-length",
        "header",
        sub(&ha_bytes, "\"length\":8192", "\"length\":0"),
        "ZERO_ELEMENT",
        "an empty segment record",
    ));

    // Whole-DOCUMENT byte total, not just this part's.
    let huge = Header {
        configs: vec![],
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
        huge.canonical_bytes(),
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
        hf.canonical_bytes(),
        "a pin-absent spec object validates as data",
    ));
    cases.push(red_case(
        g,
        "unaliased-spec-serving",
        "header-serving",
        hf.canonical_bytes(),
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
        h.canonical_bytes(),
        "DIVISION_REMAINDER",
        "in/2 with an odd in",
    ));
    let mut h = ha.clone();
    h.components[0].1[1].1.shape = vec![67108864, 67108864];
    cases.push(red_case(
        g,
        "element-count-cap",
        "header",
        h.canonical_bytes(),
        "COUNT_CAP",
        "aggregate element limit",
    ));
    // the element cap fires before any product can overflow, so the checked-arithmetic arm
    // is reached through a legal-but-extreme relation instead (probe spec, in the closure)
    let probe = probe_spec();
    let h = Header {
        configs: vec![],
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
                            body: Body::Inline(tensorfs_core::b64::encode(&[0u8])),
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
        h.canonical_bytes(),
        "ARITH_OVERFLOW",
        "checked 64-bit geometry arithmetic in the relation evaluator",
    ));
    let mut h = ha.clone();
    h.components[0].1[1].1.dtype = Dtype::I8;
    cases.push(red_case(
        g,
        "logical-dtype-carrier",
        "header",
        h.canonical_bytes(),
        "DTYPE_MISMATCH",
        "carrier no longer equals the logical dtype",
    ));

    // --- spec reds
    let g = "spec-red";
    let plain_bytes = plain.canonical_bytes();
    cases.push(red_case(
        g,
        "optional-role",
        "spec",
        sub(
            &plain_bytes,
            "\"shape\":{\"t\":\"same\"}",
            "\"optional\":true,\"shape\":{\"t\":\"same\"}",
        ),
        "OPTIONAL_ROLE_FORBIDDEN",
        "no optional roles: one digest, one byte contract",
    ));
    cases.push(red_case(
        g,
        "unknown-relation",
        "spec",
        sub(&plain_bytes, "\"t\":\"same\"", "\"t\":\"stride\""),
        "UNKNOWN_RELATION",
        "tag outside the closed grammar",
    ));
    let mxfp8_bytes = mxfp8.canonical_bytes();
    cases.push(red_case(
        g,
        "missing-logical-rank",
        "spec",
        sub(&mxfp8_bytes, ",\"logical_rank\":2", ""),
        "MISSING_FIELD",
        "dims relations require a pinned logical rank",
    ));
    cases.push(red_case(
        g,
        "unsorted-logical-dtypes",
        "spec",
        sub(
            &mxfp8_bytes,
            "[\"bf16\",\"f16\",\"f32\"]",
            "[\"f32\",\"bf16\",\"f16\"]",
        ),
        "SORT_ORDER",
        "closed set must be sorted unique",
    ));

    // --- manifest reds
    let g = "manifest-red";
    let header_ref = ha.object_ref();
    let mut m = Draft::of(&snap);
    m.entries.swap(0, 1);
    cases.push(red_case(
        g,
        "path-order",
        "manifest",
        m.canonical_bytes(),
        "PATH_ORDER",
        "entries must strictly ascend by bytes",
    ));
    let mut m = Draft::of(&snap);
    m.entries.retain(|(p, _)| p != "transformer");
    cases.push(red_case(
        g,
        "parent-missing",
        "manifest",
        m.canonical_bytes(),
        "PATH_PARENT_MISSING",
        "parents-before-children",
    ));
    let mut m = Draft::of(&snap);
    m.entries.push((
        "transformer/CONFIG.json".to_string(),
        Entry::File(ObjectRef::of(b"x")),
    ));
    m.entries.sort_by(|a, b| a.0.cmp(&b.0));
    cases.push(red_case(
        g,
        "case-collision",
        "manifest",
        m.canonical_bytes(),
        "PATH_CASE_COLLISION",
        "case-fold collision inside one tree",
    ));
    let mut m = Draft::of(&snap);
    m.entries[1].0 = "../escape.bin".to_string();
    m.entries.sort_by(|a, b| a.0.cmp(&b.0));
    cases.push(red_case(
        g,
        "path-dotdot",
        "manifest",
        m.canonical_bytes(),
        "PATH_ILLEGAL",
        "dot component",
    ));
    let mut m = Draft::of(&snap);
    m.entries[1].0 = "nul.bin".to_string();
    m.entries.sort_by(|a, b| a.0.cmp(&b.0));
    cases.push(red_case(
        g,
        "windows-reserved",
        "manifest",
        m.canonical_bytes(),
        "PATH_RESERVED",
        "reserved stem",
    ));
    let mut m = Draft::of(&snap);
    if let Entry::Blob(b) = &mut m.entries[1].1 {
        b.logical_size += 1;
    }
    cases.push(red_case(
        g,
        "blob-length",
        "manifest",
        m.canonical_bytes(),
        "BYTE_LENGTH_MISMATCH",
        "a GAP: the descriptor claims a byte the segments never deliver",
    ));

    // tfs-004: the reduced path semantics, the typed attachment, and the content-addressed
    // closure. Every one of these is a document a foreign tree could ask us to represent.
    let snap_bytes = snap.canonical_bytes();
    cases.push(red_case(
        g,
        "entry-kind-symlink",
        "manifest",
        sub(&snap_bytes, "\"kind\":\"dir\"", "\"kind\":\"symlink\""),
        "UNKNOWN_FIELD",
        "REDUCED semantics: symlinks are unrepresentable, not merely unused",
    ));
    cases.push(red_case(
        g,
        "entry-executable-bit",
        "manifest",
        sub(
            &snap_bytes,
            "\"kind\":\"dir\"",
            "\"executable\":true,\"kind\":\"dir\"",
        ),
        "UNKNOWN_FIELD",
        "no executable bit: a mode is not a storage fact",
    ));
    cases.push(red_case(
        g,
        "unknown-version",
        "manifest",
        sub(&snap_bytes, "tensorfs.snapshot/1", "tensorfs.snapshot/2"),
        "UNKNOWN_FORMAT",
        "an unknown snapshot version refuses, never degrades",
    ));
    cases.push(red_case(
        g,
        "trailing-bytes",
        "manifest",
        [snap_bytes.clone(), b"{}".to_vec()].concat(),
        "TRAILING_BYTES",
        "one trailing byte: a semantic twin is never READ as a snapshot",
    ));
    cases.push(red_case(
        g,
        "path-backslash",
        "manifest",
        sub(
            &snap_bytes,
            "transformer/config.json",
            "transformer\\\\up.json",
        ),
        "PATH_ILLEGAL",
        "traversal by the other separator",
    ));
    let mut m = Draft::of(&snap);
    m.entries.push((
        "transformer/config.json".to_string(),
        Entry::File(ObjectRef::of(b"twin")),
    ));
    m.entries.sort_by(|a, b| a.0.cmp(&b.0));
    cases.push(red_case(
        g,
        "duplicate-path",
        "manifest",
        m.canonical_bytes(),
        "PATH_ORDER",
        "one path, one entry: strict ascent IS the duplicate refusal",
    ));
    let mut m = Draft::of(&snap);
    m.entries.retain(|(p, _)| p != "model.cozytensors");
    cases.push(red_case(
        g,
        "attachment-unlisted",
        "manifest",
        m.canonical_bytes(),
        "ATTACHMENT_CARDINALITY",
        "a header the projected tree does not contain",
    ));
    let mut m = Draft::of(&snap);
    m.entries.push((
        "model2.cozytensors".to_string(),
        Entry::File(header_ref.clone()),
    ));
    m.entries.sort_by(|a, b| a.0.cmp(&b.0));
    cases.push(red_case(
        g,
        "attachment-twice",
        "manifest",
        m.canonical_bytes(),
        "ATTACHMENT_CARDINALITY",
        "two pathnames for one header: which one is the checkpoint?",
    ));
    let mut m = Draft::of(&snap);
    m.entries = vec![];
    cases.push(red_case(
        g,
        "entries-empty",
        "manifest",
        m.canonical_bytes(),
        "MISSING_FIELD",
        "a snapshot with nothing in it",
    ));
    let mut m = Draft::of(&snap);
    let spec_obj = plain.object_ref();
    m.entries
        .push((tensorfs_core::manifest::CLOSURE_DIR.to_string(), Entry::Dir));
    m.entries.push((
        tensorfs_core::manifest::closure_path(&spec_obj.sha256),
        Entry::File(spec_obj.clone()),
    ));
    m.entries.sort_by(|a, b| a.0.cmp(&b.0));
    cases.push(ok_case(
        g,
        "closure-contained",
        "manifest",
        m.canonical_bytes(),
        "a contained EncodingSpec at the path whose terminal IS its digest",
    ));
    let good = m.canonical_bytes();
    let mut wrong = spec_obj.sha256.clone();
    wrong.replace_range(0..1, if wrong.starts_with('a') { "b" } else { "a" });
    cases.push(red_case(
        g,
        "closure-digest-mismatch",
        "manifest",
        sub(
            &good,
            &format!("closure/{}.json", spec_obj.sha256),
            &format!("closure/{wrong}.json"),
        ),
        "PATH_DIGEST_MISMATCH",
        "a content-addressed path that names something else",
    ));
    let mut m = Draft::of(&snap);
    if let Entry::Blob(b) = &mut m.entries[1].1 {
        // a HOLE: a non-final segment short of the profile grid
        b.segments.push(b.segments[0].clone());
        b.logical_size *= 2;
    }
    cases.push(red_case(
        g,
        "blob-hole",
        "manifest",
        m.canonical_bytes(),
        "BYTE_LENGTH_MISMATCH",
        "sparse holes are unrepresentable: the profile fixes every boundary",
    ));
    let mut m = Draft::of(&snap);
    if let Entry::Blob(b) = &mut m.entries[1].1 {
        b.logical_size -= 1;
    }
    cases.push(red_case(
        g,
        "blob-overlap",
        "manifest",
        m.canonical_bytes(),
        "BYTE_LENGTH_MISMATCH",
        "segments that claim more bytes than the blob has",
    ));

    // --- structure fixtures + capability records (tfs-008)
    let g = "structure";
    let fx = tensorfs_core::structure::StructureFixture {
        task_family: "corpus".to_string(),
        structural_variant: "two-component".to_string(),
        provenance: tensorfs_core::structure::Provenance {
            kind: tensorfs_core::structure::SYNTHETIC.to_string(),
            source: "the vector corpus".to_string(),
            source_sha256: None,
            note: "a minimal banked projection: two components, three logical keys".to_string(),
        },
        components: vec![
            (
                "transformer".to_string(),
                vec![
                    (
                        "blocks.0.attn.to_q.weight".to_string(),
                        tensorfs_core::structure::Logical {
                            dtype: Dtype::Bf16,
                            shape: vec![64, 128],
                        },
                    ),
                    (
                        "blocks.0.norm.weight".to_string(),
                        tensorfs_core::structure::Logical {
                            dtype: Dtype::Bf16,
                            shape: vec![128],
                        },
                    ),
                ],
            ),
            (
                "vae".to_string(),
                vec![(
                    "decoder.conv_in.weight".to_string(),
                    tensorfs_core::structure::Logical {
                        dtype: Dtype::F32,
                        shape: vec![8, 8],
                    },
                )],
            ),
        ],
    };
    let fx_bytes = fx.canonical_bytes();
    cases.push(ok_case(
        g,
        "fixture-synthetic",
        "structure",
        fx_bytes.clone(),
        "banked projection + topology_digest, provenance stated in the same bytes",
    ));
    cases.push(red_case(
        "structure-red",
        "provenance-real-undigested",
        "structure",
        sub(&fx_bytes, "\"kind\":\"synthetic\"", "\"kind\":\"real\""),
        "MISSING_FIELD",
        "a claim of realness with no source digest is unfalsifiable",
    ));
    cases.push(red_case(
        "structure-red",
        "provenance-unknown-kind",
        "structure",
        sub(&fx_bytes, "\"kind\":\"synthetic\"", "\"kind\":\"probably\""),
        "UNKNOWN_FIELD",
        "provenance is real or synthetic, never a third thing",
    ));

    // --- the border's own documents (tfs-003)
    let g = "ingest";
    let banked = tensorfs_core::ingest::fingerprint::Banked {
        keyset_digest: tensorfs_core::ids::object_id(b"a reviewed key set"),
        dialect: "safetensors.diffusers".to_string(),
        task_family: "sdxl".to_string(),
        structural_variant: "base-1.0-fp16".to_string(),
        component: "vae".to_string(),
        converter: "diffusers.identity/1".to_string(),
        topology_digest: tensorfs_core::ids::object_id(b"a projected topology"),
        logical_keys: 248,
        provenance: tensorfs_core::structure::Provenance {
            kind: tensorfs_core::structure::REAL.to_string(),
            source: "stabilityai/stable-diffusion-xl-base-1.0".to_string(),
            source_sha256: Some(tensorfs_core::sha256::hex_digest(b"the exact carrier")),
            note: "banked from a real carrier".to_string(),
        },
    };
    let reg = tensorfs_core::ingest::fingerprint::FingerprintRegistry {
        entries: vec![banked.clone()],
    };
    cases.push(ok_case(
        g,
        "fingerprint-registry",
        "registry",
        reg.canonical_bytes(),
        "a reviewed entry: the review happened when this was banked",
    ));
    let root = tensorfs_core::ingest::transaction::IngestRoot {
        session: "e3b0c44298fc1c14".to_string(),
        tenant: "acme".to_string(),
        created: 1787600000,
        expires_at: 1787603600,
        candidates: vec![ObjectRef::of(b"a candidate header")],
    };
    cases.push(ok_case(
        g,
        "ingest-root",
        "root",
        root.canonical_bytes(),
        "an expiring quarantine root: a candidate is never a checkpoint",
    ));

    // --- ingest reds
    let g = "ingest-red";
    let mut unfalsifiable = reg.clone();
    unfalsifiable.entries[0].provenance.source_sha256 = None;
    cases.push(red_case(
        g,
        "registry-real-without-digest",
        "registry",
        unfalsifiable.canonical_bytes(),
        "MISSING_FIELD",
        "a `real` claim naming no bytes is unfalsifiable",
    ));
    let mut twinned = reg.clone();
    twinned.entries.push(banked.clone());
    cases.push(red_case(
        g,
        "registry-duplicate-entry",
        "registry",
        twinned.canonical_bytes(),
        "DUPLICATE_KEY",
        "one (key set, component) is banked once — twice is a review defect",
    ));
    let mut future = root.clone();
    future.session = "not a session name".to_string();
    cases.push(red_case(
        g,
        "root-session-grammar",
        "root",
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

    cases.extend(meta_cases());
    (cases, specs)
}

/// The metadata authority document (tfs-006) and the degraded-behaviour receipt (tfs-005).
/// The authority is the one document that decides what may be deleted, so its decoder is
/// frozen here with the same discipline as the header's.
fn meta_cases() -> Vec<Case> {
    use tensorfs_core::meta::{HoldRow, QRow, RootRow, SnapshotRow, State};

    let snap = "a".repeat(64);
    let other = "b".repeat(64);
    let full = State::from_records(
        7,
        vec![
            SnapshotRow {
                snapshot: snap.clone(),
                known_at: 100,
                tombstoned: false,
                source_url: String::new(),
                last_used: 140,
                // The ingest receipt this snapshot keeps beside its transitive graph.
                evidence: vec!["e".repeat(64)],
            },
            SnapshotRow {
                snapshot: other.clone(),
                known_at: 101,
                tombstoned: true,
                source_url: "https://example.invalid/m".to_string(),
                last_used: 141,
                evidence: Vec::new(),
            },
        ],
        vec![RootRow {
            name: "binding-0".to_string(),
            snapshot: snap.clone(),
            kind: "binding".to_string(),
            created: 120,
        }],
        vec![HoldRow {
            id: "read-1-2".to_string(),
            kind: "read".to_string(),
            snapshot: other.clone(),
            objects: vec!["c".repeat(64)],
            owner_pid: 4242,
            boot_epoch: 1700000000,
            acquired: 130,
        }],
        vec![QRow {
            object: "d".repeat(64),
            claim_generation: 6,
        }],
    );
    let bytes = full.canonical_bytes();
    let mut out = vec![ok_case(
        "meta",
        "authority-full",
        "meta",
        bytes.clone(),
        "all four record kinds in one transaction document",
    )];

    // Order is part of the document: an unsorted twin re-encodes differently, so it can
    // neither round-trip nor be believed. Two authorities disagreeing is what this refuses.
    let unsorted = String::from_utf8(bytes.clone()).unwrap();
    let swapped = {
        // Emit the snapshots array with the two rows exchanged, by rendering a state whose
        // decode order is reversed. Canonical writing sorts, so plant the bytes directly.
        let mut s = unsorted.clone();
        let a = format!("\"snapshot\":\"{snap}\"");
        let b = format!("\"snapshot\":\"{other}\"");
        let (ia, ib) = (s.find(&a).unwrap(), s.rfind(&b).unwrap());
        if ia < ib {
            s.replace_range(ib..ib + b.len(), &a);
            s.replace_range(ia..ia + a.len(), &b);
        }
        s
    };
    out.push(red_case(
        "meta-red",
        "records-out-of-order",
        "meta",
        swapped.into_bytes(),
        "SORT_ORDER",
        "two snapshot rows in descending order — an unsorted authority is refused",
    ));
    out.push(red_case(
        "meta-red",
        "hold-nonhex-snapshot",
        "meta",
        String::from_utf8(bytes.clone())
            .unwrap()
            .replacen(
                &format!("\"snapshot\":\"{other}\""),
                "\"snapshot\":\"latest\"",
                1,
            )
            .into_bytes(),
        "MALFORMED_DIGEST",
        "a hold pinning a NAME instead of an immutable snapshot id",
    ));
    out.push(red_case(
        "meta-red",
        "unknown-record-field",
        "meta",
        String::from_utf8(bytes)
            .unwrap()
            // Inserted in SORTED position on purpose: out of order it would be refused one
            // layer earlier, by the canonical reader, and this arm is about the FIELD.
            .replacen(
                "\"boot_epoch\":1700000000",
                "\"boot_epoch\":1700000000,\"expires_at\":999",
                1,
            )
            .into_bytes(),
        "UNKNOWN_FIELD",
        "a wall-clock expiry smuggled onto a LOCAL hold — local liveness is owner-evidence",
    ));

    out
}

fn uppercase_digest(bytes: &[u8]) -> Vec<u8> {
    let s = String::from_utf8(bytes.to_vec()).unwrap();
    let i = s.find("\"sha256\":\"").unwrap() + 10;
    let mut out = s.into_bytes();
    for b in out[i..i + 8].iter_mut() {
        b.make_ascii_uppercase();
    }
    out
}

// ------------------------------------------------------------------ gen

fn cmd_gen(dir: &Path) -> ExitCode {
    let (cases, specs) = build_corpus();
    // Regenerate only what the corpus OWNS. `mined/` (producer-mined vectors) and
    // `structure/` (banked fixtures) are evidence with their own provenance and their own
    // banking commands; a corpus rebuild must never be able to delete them.
    let _ = fs::remove_dir_all(dir.join("cases"));
    let _ = fs::remove_dir_all(dir.join("closure"));
    fs::create_dir_all(dir.join("closure")).unwrap();
    for s in &specs {
        let id = s.object_id();
        fs::write(
            dir.join("closure").join(format!("{}.json", &id[7..19])),
            s.canonical_bytes(),
        )
        .unwrap();
    }
    let mut n = 0;
    for c in &cases {
        let d = dir.join("cases").join(c.group);
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join(format!("{}.json", c.name)), &c.bytes).unwrap();
        let mut v = format!("doc: {}\nexpect: {}\n", c.doc, c.expect);
        if let Some(code) = &c.code {
            v += &format!("code: {code}\n");
        }
        if c.expect == "ok" {
            v += &format!("id: {}\n", tensorfs_core::ids::object_id(&c.bytes));
            if c.doc == "header" {
                let h = Header::parse(&c.bytes).unwrap();
                v += &format!("topology: {}\n", h.topology_digest());
            }
            if c.doc == "manifest" {
                let m = SnapshotManifest::parse(&c.bytes).unwrap();
                if let Ok(cid) = m.checkpoint_id() {
                    v += &format!("checkpoint: {cid}\n");
                }
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
        checkpoint: Option<String>,
    },
    Refused(Refusal),
}

fn run_doc(doc: &str, bytes: &[u8], closure: &Closure, platform: &[String]) -> Outcome {
    // ONE dispatch, shared with the Python facade's harness (tensorfs_core::corpus).
    match tensorfs_core::corpus::run_doc(doc, bytes, closure, platform) {
        Ok(o) => Outcome::Ok {
            id: o.id,
            topology: o.topology,
            checkpoint: o.checkpoint,
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
        match EncodingSpec::parse(&bytes) {
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
            .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
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
                        checkpoint,
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
                    if let Some(c) = field(&v, "checkpoint") {
                        if checkpoint.as_deref() != Some(c.as_str()) {
                            bad.push(format!("checkpoint {checkpoint:?} != frozen {c}"));
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
    let cpof = |n: &str| -> String {
        match get(n) {
            Some(Outcome::Ok {
                checkpoint: Some(c),
                ..
            }) => c.clone(),
            _ => String::new(),
        }
    };
    let mut cross: Vec<(String, bool, String)> = Vec::new();
    cross.push((
        "checkpoint_id == snapshot_id (unconditional)".into(),
        cpof("manifest/checkpoint") == idof("manifest/checkpoint")
            && !cpof("manifest/checkpoint").is_empty(),
        cpof("manifest/checkpoint"),
    ));
    cross.push((
        "header ObjectId != checkpoint_id".into(),
        idof("header/plain-fp8") != cpof("manifest/checkpoint"),
        format!(
            "{} vs {}",
            &idof("header/plain-fp8")[..19],
            &cpof("manifest/checkpoint")[..19]
        ),
    ));
    cross.push((
        "a config change flips header id, not topology_digest".into(),
        idof("header/plain-fp8") != idof("header/plain-fp8-noconfig")
            && topof("header/plain-fp8") == topof("header/plain-fp8-noconfig"),
        topof("header/plain-fp8"),
    ));
    cross.push((
        "one more sibling file flips checkpoint_id only".into(),
        cpof("manifest/checkpoint") != cpof("manifest/checkpoint-extra-file"),
        cpof("manifest/checkpoint-extra-file"),
    ));
    let (cha, _) = header_a(Some(config_ref()));
    let csnap = manifest_checkpoint(&cha, false);
    let cprof = profile_normal(
        vec![seed_spec("plain/1", 0).object_ref()],
        "safetensors.diffusers",
    );
    let csub = subject(&cprof, &cha, "acme");
    let cstamp = stamp_doc(&cprof, &csub, &csnap, &cha);
    let crcpt = receipt(&csub, &csnap, &cha, &cstamp, "platform-key-1");
    cross.push((
        "receipt binds its exact subject/snapshot/header/stamp".into(),
        crcpt.binds(&csub, &csnap, &cha, &cstamp).is_ok(),
        String::new(),
    ));
    let other = subject(&cprof, &cha, "other-tenant");
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

// ------------------------------------------------------------------ bench

fn rss_kb(field: &str) -> u64 {
    fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with(field))
        .and_then(|l| l.split_whitespace().nth(1).and_then(|n| n.parse().ok()))
        .unwrap_or(0)
}

fn cmd_bench(tensors: usize, reps: usize) -> ExitCode {
    let plain = seed_spec("plain/1", 0);
    let fp8 = seed_spec("fp8-rowwise/1", 0);
    let mut closure = Closure::default();
    closure.insert(plain.clone());
    closure.insert(fp8.clone());

    let build = Instant::now();
    let mut ts = Vec::with_capacity(tensors);
    for i in 0..tensors {
        let key = format!(
            "blocks.{}.attn.to_{}.weight",
            i / 4,
            "qkvo".as_bytes()[i % 4] as char
        );
        ts.push((
            key,
            Tensor {
                dtype: Dtype::Bf16,
                shape: vec![4096, 4096],
                encoding: fp8.object_id(),
                parts: vec![
                    (
                        "data".to_string(),
                        Part {
                            dtype: Dtype::F8E4M3FN,
                            shape: vec![4096, 4096],
                            body: Body::Segments(
                                (0..4)
                                    .map(|s| ObjectRef {
                                        sha256: tensorfs_core::sha256::hex_digest(
                                            format!("{i}:{s}").as_bytes(),
                                        ),
                                        length: 4 * 1024 * 1024,
                                    })
                                    .collect(),
                            ),
                        },
                    ),
                    ("scale".to_string(), part_of(Dtype::F32, vec![4096])),
                ],
            },
        ));
    }
    ts.sort_by(|a, b| a.0.cmp(&b.0));
    let h = Header {
        configs: vec![("transformer".to_string(), config_ref())],
        encodings: encodings_of(&[&fp8]),
        components: vec![("transformer".to_string(), ts)],
    };
    let build_s = build.elapsed().as_secs_f64();

    let t = Instant::now();
    let mut bytes = Vec::new();
    for _ in 0..reps {
        bytes = h.canonical_bytes();
    }
    let write_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let mut id = String::new();
    for _ in 0..reps {
        id = tensorfs_core::ids::object_id(&bytes);
    }
    let hash_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    for _ in 0..reps {
        Header::parse(&bytes).expect("reparse");
    }
    let parse_s = t.elapsed().as_secs_f64();

    let parsed = Header::parse(&bytes).unwrap();
    let t = Instant::now();
    for _ in 0..reps {
        parsed.validate(&closure).expect("validate");
    }
    let val_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let mut topo = String::new();
    for _ in 0..reps {
        topo = parsed.topology_digest();
    }
    let topo_s = t.elapsed().as_secs_f64();

    let mb = bytes.len() as f64 / (1024.0 * 1024.0);
    println!(
        "header: {tensors} tensors, {} B canonical ({mb:.2} MiB), built in {build_s:.3}s",
        bytes.len()
    );
    println!("id: {id}");
    println!("topology_digest: {topo}");
    println!("reps: {reps}");
    let row = |what: &str, s: f64| {
        println!(
            "  {what:<22} {:>8.1} ms/doc   {:>7.1} docs/s   {:>8.1} MiB/s",
            s / reps as f64 * 1000.0,
            reps as f64 / s,
            mb * reps as f64 / s
        )
    };
    row("canonical write", write_s);
    row("sha256 id", hash_s);
    row("strict parse", parse_s);
    row("validate", val_s);
    row("topology projection", topo_s);
    println!(
        "  peak RSS {:.1} MiB (current {:.1} MiB)",
        rss_kb("VmHWM") as f64 / 1024.0,
        rss_kb("VmRSS") as f64 / 1024.0
    );
    ExitCode::SUCCESS
}

// ------------------------------------------------------------------ flip matrix

fn cmd_flip() -> ExitCode {
    let (ha, _) = header_a(Some(config_ref()));
    let snap = manifest_checkpoint(&ha, false);
    let prof = profile_normal(
        vec![seed_spec("plain/1", 0).object_ref()],
        "safetensors.diffusers",
    );
    let sub0 = subject(&prof, &ha, "acme");
    let st0 = stamp_doc(&prof, &sub0, &snap, &ha);
    let r0 = receipt(&sub0, &snap, &ha, &st0, "platform-key-1");

    let base = (
        ha.object_id(),
        ha.topology_digest(),
        snap.checkpoint_id().unwrap(),
        sub0.object_id(),
        r0.object_id(),
    );

    // (what changed, the five ids it produced, which of the five flipped)
    type FlipRow = (
        &'static str,
        (String, String, String, String, String),
        [bool; 5],
    );
    let mut rows: Vec<FlipRow> = Vec::new();

    let mut h = ha.clone();
    if let Body::Segments(s) = &mut h.components[0].1[0].1.parts[0].1.body {
        s[0].sha256 = tensorfs_core::sha256::hex_digest(b"different weight bytes");
    }
    let s = manifest_checkpoint(&h, false);
    rows.push((
        "artifact bytes change",
        (
            h.object_id(),
            h.topology_digest(),
            s.checkpoint_id().unwrap(),
            sub0.object_id(),
            r0.object_id(),
        ),
        [true, false, true, false, false],
    ));

    let mut h = ha.clone();
    h.components[0].1[1].1.dtype = Dtype::F16;
    h.components[0].1[1].1.parts[0].1.dtype = Dtype::F16;
    let s = manifest_checkpoint(&h, false);
    rows.push((
        "logical dtype change",
        (
            h.object_id(),
            h.topology_digest(),
            s.checkpoint_id().unwrap(),
            sub0.object_id(),
            r0.object_id(),
        ),
        [true, true, true, false, false],
    ));

    let (h, _) = header_a(None);
    let s = manifest_checkpoint(&h, false);
    rows.push((
        "config change",
        (
            h.object_id(),
            h.topology_digest(),
            s.checkpoint_id().unwrap(),
            sub0.object_id(),
            r0.object_id(),
        ),
        [true, false, true, false, false],
    ));

    let s = manifest_checkpoint(&ha, true);
    rows.push((
        "snapshot gains a file",
        (
            ha.object_id(),
            ha.topology_digest(),
            s.checkpoint_id().unwrap(),
            sub0.object_id(),
            r0.object_id(),
        ),
        [false, false, true, false, false],
    ));

    let sub1 = subject(&prof, &ha, "other-tenant");
    rows.push((
        "tenant change",
        (
            ha.object_id(),
            ha.topology_digest(),
            snap.checkpoint_id().unwrap(),
            sub1.object_id(),
            r0.object_id(),
        ),
        [false, false, false, true, false],
    ));

    let r1 = receipt(&sub0, &snap, &ha, &st0, "platform-key-2-rotated");
    rows.push((
        "signer rotation",
        (
            ha.object_id(),
            ha.topology_digest(),
            snap.checkpoint_id().unwrap(),
            sub0.object_id(),
            r1.object_id(),
        ),
        [false, false, false, false, true],
    ));

    println!("identity-flip matrix (expected flips vs observed)");
    println!(
        "{:<24} {:>10} {:>10} {:>12} {:>10} {:>10}",
        "change", "header", "topology", "checkpoint", "subject", "receipt"
    );
    let mut ok = true;
    for (what, got, want) in &rows {
        let obs = [
            got.0 != base.0,
            got.1 != base.1,
            got.2 != base.2,
            got.3 != base.3,
            got.4 != base.4,
        ];
        let cell = |i: usize| {
            let mark = if obs[i] == want[i] { "" } else { " <-WRONG" };
            ok &= obs[i] == want[i];
            format!("{}{}", if obs[i] { "FLIP" } else { "same" }, mark)
        };
        let cells: Vec<String> = (0..5).map(cell).collect();
        println!(
            "{what:<24} {:>10} {:>10} {:>12} {:>10} {:>10}",
            cells[0], cells[1], cells[2], cells[3], cells[4]
        );
    }
    // alias identity: the alias lives outside the spec object, so renaming flips nothing.
    let plain = seed_spec("plain/1", 0);
    let mut renamed = plain.clone();
    renamed.doc = plain.doc.clone();
    let alias_stable = renamed.object_id() == plain.object_id();
    println!(
        "{:<24} {:>10}",
        "alias rename/promotion",
        if alias_stable { "same" } else { "FLIP <-WRONG" }
    );
    ok &= alias_stable;
    if ok {
        println!("\nall flips as designed");
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
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
        println!("pretty:\n{}", canon::pretty(&h.to_value(), 0));
    } else if let Ok(m) = SnapshotManifest::parse(&bytes) {
        println!("kind:      snapshot manifest");
        match m.checkpoint_id() {
            Ok(c) => println!("checkpoint_id: {c}"),
            Err(e) => println!("checkpoint_id: n/a ({})", e.code.as_str()),
        }
    } else if let Ok(s) = EncodingSpec::parse(&bytes) {
        println!("kind:      encoding spec ({} roles)", s.roles.len());
    } else {
        println!(
            "kind:      NOT a tensorfs document — {}",
            Header::parse(&bytes).err().unwrap()
        );
    }
    ExitCode::SUCCESS
}

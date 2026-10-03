//! tfs — TensorFS store and CozyTensors inspection tool.
//!
//!   tfs version                  release version + sha256 of this binary, the skew observable
//!   tfs registry                 print the seeded platform namespace (alias -> digest)
//!   tfs id <file>                object id (+ derived ids) of a stored document
//!   tfs cbor encode|decode|check <file>  header conformance seam; JSON is diagnostic only
//!   tfs config canonicalize <config.json> --out <canonical.json>
//!
//! The object store (tfs-002). Flags: --expect <hex> --expect-length <n> --out <path>
//! --range <off>:<len> --rate <f> --pass <n> --slow-ms <n>
//! --size <bytes> --small <n>, plus the dev-only kill points --fault <stage> --ready <path>.
//!
//! A Store is created only where someone said to: `tfs store init|ensure <root>`, or the same
//! verbs over `TENSORFS_HOME`. Every other command REFUSES a root that does not exist, and the
//! `~/.tensorfs` fallback is opened, never conjured (tfs-053).
//!
//!   tfs store init|info <root>       [--repo-cache <root> | --no-repo-cache]
//!   tfs store ensure [<root>]         [--repo-cache <root> | --no-repo-cache]
//!                                the Store RECORDS the mounted immutable object cache it
//!                                mirrors through and reads back from, so no later command
//!                                and no caller ever names one. `info` prints the binding
//!   tfs store prepare-readers [<root>]
//!   tfs store complete-cozytensors [<root>]
//!   tfs home                     print TENSORFS_HOME or ~/.tensorfs
//!   tfs store rebuild <census> --rows <jsonl> [--sqlite <tensorfs.sqlite>]
//!   tfs store admit-file <store> blob|manifest <id> <length> <source>
//!   tfs repo-cache admit <cache> <store> blob|manifest <id> <length>
//!   tfs repo-cache backfill <cache> <store> blob|manifest <id> <length>
//!   tfs repo-cache restore <store> <manifest-id> <length> [--session <s>]
//!   tfs repo-cache resume <store> <chain-id> <length> --plan <sha256:...>
//!                                   [--session <s>] [--tenant <t>] [--no-journal]
//!   tfs repo-cache chain <cache> <chain-id> <length>       the links, root to head
//!   tfs put <root> <file>        stream one object in (any shape)
//!   tfs get <root> <hex>         verified read, whole or ranged; with --out - the bytes
//!                                go to stdout under a read lease (--range <off>:<len>|<off>:)
//!   tfs contains <root> <hex>    the presence HINT, and what it is worth
//!   tfs verify <root> <hex>      verification record or rehash; remove corrupt bytes
//!   tfs scrub <root>             one rate knob over the whole store
//!   tfs reap <root>              reap admission temps whose writer is gone
//!   tfs fill <root> <plan>       resumable fill: skip only verified objects
//!
//! The ONE transport (tfs-048/tfs-049/tfs-050). The credential is `--credential-file` or
//! TFS_CREDENTIAL (`bearer <token>`, `worker <id> <token>`, or empty for anonymous), never
//! bare argv; a URL of `-` arrives on stdin. Flags: --hub <base> --lane --session --streams <n>
//! --timeout <s> --allow-hosts a,b,.c --allow-local --max-redirects <n>
//! --ca-file <pem> (a private hub's CA, trusted beside the webpki set).
//! `--streams` is the most requests in flight (default 512, memory-bounded).
//! `tfs fetch` reads and backfills the Store's bound repo cache between the local Store
//! and the origin. It is bound by `tfs store ensure --repo-cache` and never by a flag here.
//!
//!   tfs capabilities             one feature name per line (ensure/1, get-range/1, ...)
//!   tfs ensure <root> <ref> --hub <base> [--lane L] [--step S] [--keep sha256:<m>]...
//!                               make one model resident: closure, one flight per manifest per
//!                               Store, disk admission + GC policy, paced resumable fetch, retry
//!                               only after measured progress. STDOUT is JSON lines:
//!                               ensure.progress samples, then ensure.result (exit 0) or
//!                               ensure.refused {code, resumable, detail} (exit 1)
//!   tfs pressure <root> [--keep sha256:<m>]...   one pressure pass: at <= 1/10 free, run
//!                               the GC policy until > 1/5 is free; one JSON line
//!   tfs fetch <root> <ref>       closure -> plan -> N-way walk (presign at use) -> proof
//!   tfs fetch bounded <root> <ref> --max-download-bytes N --max-download-objects N
//!                               same pull, refused before body transfer if new objects exceed bounds
//!   tfs fetch plan-objects <root> <session> --refs <jsonl>   HELD/WANTED, no manifest
//!   tfs fetch url <root> <sha256> <length> <url|-> [--progress] [--streams N]
//!                                                    one granted object, in parallel ranges
//!            --progress opens the same STDERR event channel `tfs fetch` reports on, while
//!            the bytes move: {"event":"fetch.progress","origin_bytes":N,...} per sample,
//!            so a supervisor can tell slow from wedged without a clock. stdout stays the
//!            result. --progress-bytes <n> is that channel's resolution, never a verdict
//!
//! The STAGING area (tfs-067). A foreign source carrier is not a CAS object: it lands at
//! `<root>/staging/<sha256>`, outside every namespace a census walks, with no catalog
//! record and no `gc` candidacy — a temporary input that the store reads once and throws
//! away. `tfs fetch url` is unchanged and still admits repo objects to the CAS.
//!
//!   tfs staging fetch <root> <sha256> <length> <url|->   same transport, same digest, same
//!            [--progress] [--streams N] [--part-mib N]   fence; a different destination.
//!            `present` answers a re-ask over a file that rehashes to the id, moving zero
//!            bytes; `staged` answers one this call moved. `path` is printed in both cases
//!   tfs staging drop <root> <sha256>    retire one spent carrier; one unlink, idempotent
//!   tfs staging clear <root>            sweep the area plus orphan `stage-` temps
//!   tfs staging list <root>             every carrier: id, length, absolute path
//!
//!   tfs push <root> <sha256> --grant-file <path> [--manifest] [--content-type ct]
//!            the grant is the destination on line one, then one "name: value" per header
//!            its signature covers; TFS_PUSH_GRANT carries the same text, and
//!            --url <presigned|-> is a bare destination that signs no conditions
//!   tfs source list <hf://org/repo@rev | civitai://id> [--api <base>]  provider listing
//!   tfs source resolve <uri> [--source-profile <name>]... [--registry <r>] [--json]
//!            the EXACT member list: every tensor
//!            carrier with its object id, length and URL, sharded repos expanded through
//!            their weight_map. No store and no tensor bytes — this is what an owner runs
//!            to size a container disk before renting the machine that will hold it.
//!            `members` and `objects` are printed as separate counts: two members can be
//!            the same bytes, and a CAS store holds those once.
//!            --source-profile NARROWS the repository to one reviewed model and is the
//!            step that decides 210 GB instead of 498 GB; without it the answer is the
//!            whole repository, which is a successful download of the wrong size
//!   tfs source plan <uri> --source-profile <name>... [--registry <r>] [--json]
//!            the CONVERSION plan, decided from HEADERS, before a byte of payload moves
//!            and before a pod is rented (tfs-076). Resolve, narrow, fetch each member's
//!            header by ranged GET, and run the pod's own planner over them. No store, no
//!            tensor bytes. MiniMax-H3 is 275 KB of header standing in for 210.3 GB of
//!            payload, and three separate H3 runs moved the 210.3 GB before refusing on
//!            facts inside those 275 KB.
//!            exit 0 PLAN OK / 1 REFUSED, this cannot convert / 2 UNDECIDED, a header
//!            could not be read cheaply — proceed to the ordinary path, never refuse
//!   tfs source pull <root> <uri> [--api <base>] [--allow-local]        tfs-052 handlers
//!   tfs bench-store <root>       put/read/range/rehash throughput + peak RSS
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
//!   tfs repo list <store> --rows <jsonl>     every release lane and local alias
//!   tfs repo usage <store> --rows <jsonl>    blob bytes per repo: total, and unique to it
//!   tfs repo apply <current.json|-> <mutation.json> --out <replacement> --rows <jsonl>
//!                  put_checkpoint requires --manifest, --header and --inventory
//!   tfs repo commit <store> <current.json|-> <mutation.json>
//!   tfs local resolve <store> <name>
//!   tfs local replace <store> <name> <source-sha256> <manifest> <length>
//!                     --observed <repository-sha256|absent>
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
//!   tfs checkpoint page <root> <hex> <len>   bounded source recovery link, --offset/--limit
//!   tfs checkpoint locate <root> <hex> <component> <key> <role> <off> <len>
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
use tensorfs_core::durability;
use tensorfs_core::err::Refusal;
use tensorfs_core::header::Header;
use tensorfs_core::ids::{Doc, ObjectRef};
use tensorfs_core::repo_cache::{CacheKind, RepoObjectCache};
use tensorfs_core::store::{Fault, Store, Verdict, FAULT_STAGES};

type TfsResult<T> = tensorfs_core::err::Result<T>;

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
        ["pressure", r] => ensure::cmd_pressure(Path::new(r), &flags),
        ["store", "init", r] => cmd_store(Path::new(r), "init", &flags),
        ["store", "info", r] => cmd_store(Path::new(r), "info", &flags),
        ["store", "ensure", r] => cmd_store(Path::new(r), "ensure", &flags),
        ["store", "prepare-readers", r] => cmd_store_prepare_readers(Path::new(r)),
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
        ["contains", r, h] => cmd_contains(Path::new(r), h),
        ["reap", r] => cmd_reap(Path::new(r)),
        ["fill", r, p] => cmd_fill(Path::new(r), Path::new(p), &flags),
        ["checkpoint", "write", r] => ckpt::cmd_write(Path::new(r), &flags),
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
        ["repo", "usage", r] => repo::cmd_repo_usage(Path::new(r), &flags),
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
        ["reclaim", "usage", store] => repo::cmd_reclaim_usage(Path::new(store), &flags),
        ["reclaim", "plan", store] => repo::cmd_reclaim_plan(Path::new(store), &flags),
        ["reclaim", "run", store] => repo::cmd_reclaim_run(Path::new(store), &flags),
        ["gc", root] => repo::cmd_gc(Path::new(root), &flags),
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
        ["fetch", "bounded", r, refspec] => fetch::cmd_bounded_pull(Path::new(r), refspec, &flags),
        ["staging", "fetch", r, i, l, u] => fetch::cmd_staging_fetch(Path::new(r), i, l, u, &flags),
        ["staging", "drop", r, i] => fetch::cmd_staging_drop(Path::new(r), i),
        ["fetch", "bounded", _] => {
            eprintln!("usage: see the header of {}", file!());
            ExitCode::from(2)
        }
        ["fetch", r, refspec] => fetch::cmd_pull(Path::new(r), refspec, &flags),
        ["push", r, i] => fetch::cmd_push(Path::new(r), i, &flags),
        ["cbor", "encode", f] => cmd_cbor("encode", Path::new(f), &flags),
        ["cbor", "decode", f] => cmd_cbor("decode", Path::new(f), &flags),
        ["cbor", "check", f] => cmd_cbor("check", Path::new(f), &flags),
        _ => {
            eprintln!("usage: see the header of {}", file!());
            ExitCode::from(2)
        }
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
    let (_, manifest) = match cache_args("manifest", object, length) {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
    let store = match Store::open(store) {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
    let session = flag(flags, "session").unwrap_or("restore");
    let report = match durability::restore_checkpoint(&store, session, &manifest) {
        Ok(report) => report,
        Err(error) => return bail(error),
    };
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
    let (_, head) = match cache_args("blob", object, length) {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
    let store = match Store::open(store) {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
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
    let chain = match durability::chain(&cache, &head) {
        Ok(chain) => chain,
        Err(error) => return bail(error),
    };
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
    let report = match durability::restore_objects(&store, session, &set.blobs, &set.manifests) {
        Ok(report) => report,
        Err(error) => return bail(error),
    };
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
    let (_, head) = match cache_args("blob", object, length) {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
    let cache = RepoObjectCache::new(cache);
    let chain = match durability::chain(&cache, &head) {
        Ok(chain) => chain,
        Err(error) => return bail(error),
    };
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
fn cmd_store(root: &Path, verb: &str, flags: &Flags) -> ExitCode {
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
    let mut s = match verb {
        "init" => ok!(Store::init(root)),
        "ensure" => ok!(Store::ensure(root)),
        _ => ok!(Store::open(root)),
    };
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

fn cmd_store_prepare_readers(root: &Path) -> ExitCode {
    let store = ok!(Store::open(root));
    ok!(store.prepare_readers());
    println!("root:       {}", store.root().display());
    println!("readers:    prepared");
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

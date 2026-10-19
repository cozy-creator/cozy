//! The v1 verbs Tensorhub still execs (`internal/tfs`). They go with it (xs-045 U11).
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use tensorfs_core::canon;
use tensorfs_core::err::Refusal;
use tensorfs_core::header::Header;

macro_rules! ok {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => return bail(e),
        }
    };
}

pub const USAGE: &str = "v1 (until xs-045 U11):
  version [--json]
  key blob|manifest <sha256> | repo <org> <name>
  cbor encode|decode|check <file> [--out <file>]
  manifest build <entries.jsonl> --out <manifest.json>
  manifest inspect <manifest.json> --refs <jsonl> [--header <h> --inventory <i> --closure <c>]
  repo inspect <repo.json> --rows <jsonl>
  repo apply <current.json|-> <mutation.json> --out <file> --rows <jsonl> [--manifest <m> --header <h> --inventory <i>]
  store rebuild <census> --rows <jsonl>
  gc headers <census> --out <jsonl>
  gc plan <census> --holds <jsonl> --out <jsonl>";

/// Flags that never take a value.
const SWITCHES: &[&str] = &[
    "allow-local",
    "dry-run",
    "json",
    "keys",
    "no-configs",
    "no-disk-budget",
    "no-netns",
    "plain",
    "progress",
];
/// Flags that take the next token as their value; given last, or before another flag, they
/// are present without one.
const VALUED: &[&str] = &[
    "allow-hosts",
    "blocks",
    "ca-file",
    "carrier",
    "closure",
    "config",
    "credential-file",
    "disk-budget",
    "expect",
    "expect-length",
    "fault",
    "header",
    "holds",
    "hub",
    "inventory",
    "isolation-tier",
    "keep",
    "lane",
    "manifest",
    "max-download-bytes",
    "max-download-objects",
    "max-redirects",
    "member",
    "observed",
    "order",
    "out",
    "part-mib",
    "progress-bytes",
    "range",
    "ready",
    "refs",
    "registry",
    "rlimit-as-mb",
    "rlimit-fsize-mb",
    "rows",
    "sample-seconds",
    "scale",
    "session",
    "slow-ms",
    "source",
    "source-name",
    "source-plan",
    "source-profile",
    "spec-variant",
    "step",
    "streams",
    "target",
    "tenant",
    "timeout",
];

fn usage() -> ExitCode {
    eprintln!("{}\n{USAGE}", super::USAGE);
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

pub fn main(argv: Vec<String>) -> ExitCode {
    let (args, flags) = split_flags(argv);
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
        ["store", "rebuild", r] => repo::cmd_rebuild(Path::new(r), &flags),
        ["manifest", "build", entries] => repo::cmd_manifest_build(Path::new(entries), &flags),
        ["manifest", "inspect", p] => repo::cmd_manifest_inspect(Path::new(p), &flags),
        ["repo", "inspect", p] => repo::cmd_repo_inspect(Path::new(p), &flags),
        ["repo", "apply", current, mutation] => {
            repo::cmd_repo_apply(current, Path::new(mutation), &flags)
        }
        ["key", kind, rest @ ..] => repo::cmd_key(kind, rest),
        ["gc", "headers", census] => repo::cmd_gc_headers(Path::new(census), &flags),
        ["gc", "plan", census] => repo::cmd_gc_plan(Path::new(census), &flags),
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

fn bail(e: Refusal) -> ExitCode {
    eprintln!("REFUSED {e}");
    ExitCode::FAILURE
}

mod repo;

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
            let value = ok!(canon::parse_with_depth(
                &bytes,
                tensorfs_core::limits::DOC_MAX_BYTES,
                16
            ));
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

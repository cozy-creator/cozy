//! tfs: TensorFS v2 on the command line. With `--json` every command prints one JSON document,
//! a failure as `{"error": {"code", "detail"}}`. The v1 verbs Tensorhub still execs live in
//! `v1/` until it moves to v2 (xs-045 U11).

mod dev;
mod v1;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value as Json};
use tensorfs_core::client::{self, ImportOptions, Named};
use tensorfs_core::export;
use tensorfs_core::failure::{fail, Failure, Kind, Result};
use tensorfs_core::format::{Body, Dtype, ObjectRef};
use tensorfs_core::migrate::{self, v1 as old};
use tensorfs_core::store2::Store;

const USAGE: &str = "usage: tfs [--store <dir>] [--json] <verb> ...
  import <src>... [--name <org/name>] [--converter <table>] [--variant <variant>]
      src: a file, a directory, hf://<org>/<repo>[@<rev>][/<path>], civitai://<version>[/<file>], a URL
  pull [<origin>] <org/name[@release]> [--lane <lane>] [--hub <url>]
      from Tensorhub (--hub, else https://tensorhub.com; $TENSORFS_HUB_TOKEN reads private
      checkpoints), or from a static origin's URL or directory
  export <ref> <dir> --safetensors [--dequantize bf16|f16|f32]
  export <ref> <dir> --origin
  inspect <ref>
  verify [<ref>]
  gc [--older-than <7d|12h|30m|90s>]
  dev ckpt [--name <org/name>] [--blocks <n>] [--scale <n>] [--plain] [--no-configs]
  dev migrate <v1-dir> [<manifest-sha256>|<org/name>]... [--dry-run | --plan]
      v1-dir: a v1 store (read only) or a directory of a v1 manifest and its objects;
      --plan: the census, names and new bytes from manifests and headers alone, no store
  dev headers <hub-url> <dir> <org/name@release/lane>...   a released lane's v1 manifest and header
<ref> is org/name[@release] (--lane names a lane, else the release's only one) or a manifest
sha256. The store is --store, else $TENSORFS_HOME, else ~/.tensorfs; it is created when absent.";

const SWITCHES: &[&str] = &[
    "json",
    "safetensors",
    "origin",
    "plain",
    "no-configs",
    "dry-run",
    "plan",
];
const VALUED: &[&str] = &[
    "store",
    "hub",
    "name",
    "lane",
    "dequantize",
    "older-than",
    "converter",
    "variant",
    "blocks",
    "scale",
];
const VERBS: &[&str] = &["import", "pull", "export", "inspect", "verify", "dev"];

struct Args {
    pos: Vec<String>,
    flags: Vec<(String, String)>,
}

impl Args {
    fn parse(argv: &[String]) -> std::result::Result<Args, String> {
        let (mut pos, mut flags) = (Vec::new(), Vec::new());
        let mut it = argv.iter();
        while let Some(token) = it.next() {
            let Some(name) = token.strip_prefix("--") else {
                pos.push(token.clone());
                continue;
            };
            let (name, inline) = match name.split_once('=') {
                Some((n, v)) => (n, Some(v.to_owned())),
                None => (name, None),
            };
            let value = if SWITCHES.contains(&name) {
                String::new()
            } else if VALUED.contains(&name) {
                match inline.or_else(|| it.next().cloned()) {
                    Some(v) => v,
                    None => return Err(format!("--{name} takes a value")),
                }
            } else {
                return Err(format!("unknown flag --{name}"));
            };
            flags.push((name.to_owned(), value));
        }
        Ok(Args { pos, flags })
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.flags
            .iter()
            .rev()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }

    fn on(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    fn number(&self, name: &str, default: u64) -> Result<u64> {
        match self.get(name) {
            None => Ok(default),
            Some(v) => v
                .parse()
                .or_else(|_| fail(Kind::Invalid, format!("--{name} {v:?} is not a number"))),
        }
    }
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    // The positionals as v2 reads them; `gc headers|plan` and every other verb are v1's.
    let mut pos = Vec::new();
    let mut it = argv.iter();
    while let Some(token) = it.next() {
        match token.strip_prefix("--") {
            Some(name) if VALUED.contains(&name) => drop(it.next()),
            Some(_) => {}
            None => pos.push(token.as_str()),
        }
    }
    let v2 = match pos.as_slice() {
        ["gc"] => true,
        [verb, ..] => VERBS.contains(verb),
        [] => false,
    };
    if !v2 {
        return v1::main(argv);
    }
    let args = match Args::parse(&argv) {
        Ok(args) => args,
        Err(why) => {
            eprintln!("tfs: {why}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let json = args.on("json");
    match run(&args) {
        Ok(Some((doc, human))) => {
            if json {
                println!("{doc}");
            } else {
                println!("{human}");
            }
            ExitCode::SUCCESS
        }
        Ok(None) => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
        Err(Failed(f, doc)) => {
            if json {
                let mut out = json!({"error": {"code": f.kind.code(), "detail": f.detail}});
                if let Some(doc) = doc {
                    out.as_object_mut().expect("an object").extend(doc);
                }
                println!("{out}");
            } else {
                eprintln!("tfs: {f}");
            }
            ExitCode::FAILURE
        }
    }
}

/// A failure, with whatever the command found before it.
struct Failed(Failure, Option<Map<String, Json>>);

impl From<Failure> for Failed {
    fn from(f: Failure) -> Failed {
        Failed(f, None)
    }
}

impl From<tensorfs_core::store2::Error> for Failed {
    fn from(e: tensorfs_core::store2::Error) -> Failed {
        Failed(e.into(), None)
    }
}

type Done = std::result::Result<Option<(Json, String)>, Failed>;

fn store_root(args: &Args) -> Result<PathBuf> {
    if let Some(dir) = args.get("store") {
        return Ok(PathBuf::from(dir));
    }
    if let Some(dir) = std::env::var_os("TENSORFS_HOME").filter(|d| !d.is_empty()) {
        return Ok(PathBuf::from(dir));
    }
    match std::env::var_os("HOME").filter(|d| !d.is_empty()) {
        Some(home) => Ok(Path::new(&home).join(".tensorfs")),
        None => fail(
            Kind::Invalid,
            "no store: pass --store, or set TENSORFS_HOME or HOME",
        ),
    }
}

fn object(o: &ObjectRef) -> Json {
    json!({"sha256": o.sha256, "length": o.length})
}

fn named(n: &Named) -> Map<String, Json> {
    let doc = json!({"manifest": object(&n.manifest), "name": n.name, "release": n.release, "lane": n.lane});
    doc.as_object().expect("an object").clone()
}

fn label(n: &Named) -> String {
    match (&n.name, &n.release, &n.lane) {
        (Some(name), Some(release), Some(lane)) => {
            format!("{name}@{release} lane {lane}: sha256:{}", n.manifest.sha256)
        }
        _ => format!("sha256:{}", n.manifest.sha256),
    }
}

fn run(args: &Args) -> Done {
    let pos: Vec<&str> = args.pos.iter().map(String::as_str).collect();
    let lane = args.get("lane");
    let store = || Store::open(store_root(args)?).map_err(Failure::from);
    match pos.as_slice() {
        ["import", sources @ ..] if !sources.is_empty() => {
            let sources: Vec<String> = sources.iter().map(|s| s.to_string()).collect();
            let quiet = args.on("json");
            let last = Mutex::new((String::new(), Instant::now()));
            let progress = |phase: &str, done: u64, total: u64| {
                let mut last = last.lock().unwrap_or_else(|e| e.into_inner());
                if quiet || (last.0 == phase && last.1.elapsed() < Duration::from_secs(2)) {
                    return;
                }
                *last = (phase.to_owned(), Instant::now());
                eprintln!("{phase} {done}/{total}");
            };
            let opts = ImportOptions {
                name: args.get("name"),
                converter: args.get("converter"),
                variant: args.get("variant"),
                progress: &progress,
            };
            let n = store()?.import_named(&sources, &opts)?;
            Ok(Some((
                Json::Object(named(&n)),
                format!("imported {}", label(&n)),
            )))
        }
        ["pull", .., reference] if pos.len() <= 3 => {
            let token = std::env::var("TENSORFS_HUB_TOKEN").ok().filter(|t| !t.is_empty());
            let from = match pos.as_slice() {
                [_, origin, _] => client::PullFrom::Origin(origin),
                _ => client::PullFrom::Hub {
                    base: args.get("hub"),
                    token: token.as_deref(),
                },
            };
            let p = store()?.pull(from, reference, lane)?;
            let mut doc = named(&p.named);
            doc.extend([
                ("objects".into(), p.objects.into()),
                ("fetched".into(), p.fetched.into()),
                ("held".into(), p.held.into()),
            ]);
            let human = format!(
                "pulled {} ({} objects, {} bytes fetched, {} held)",
                label(&p.named),
                p.objects,
                p.fetched,
                p.held
            );
            Ok(Some((Json::Object(doc), human)))
        }
        ["export", reference, dir] => {
            let store = store()?;
            let n = store.resolve(reference, lane)?;
            let dir = Path::new(dir);
            let files = match (
                args.on("safetensors"),
                args.on("origin"),
                args.get("dequantize"),
            ) {
                (true, false, dequantize) => {
                    let dtype = dequantize.map(|d| Dtype::parse(&d.to_ascii_uppercase()));
                    export::safetensors(&store.checkpoint(&n.manifest)?, dir, dtype.as_ref())?
                }
                (false, true, None) => client::export_origin(&store, &n, dir)?,
                _ => return Ok(None),
            };
            let human = format!(
                "exported {} into {}: {} files",
                label(&n),
                dir.display(),
                files.len()
            );
            let mut doc = named(&n);
            doc.insert("files".into(), files.into());
            Ok(Some((Json::Object(doc), human)))
        }
        ["inspect", reference] => inspect(&store()?, reference, lane),
        ["verify"] => verify(&store()?, None),
        ["verify", reference] => verify(&store()?, Some((reference, lane))),
        ["gc"] => {
            let store = store()?;
            let collected = match args.get("older-than") {
                None => store.gc()?,
                Some(text) => store.collect(u64::MAX, duration(text)?)?,
            };
            let doc = json!({"objects": collected.objects, "bytes": collected.bytes, "pins": collected.pins});
            let human = format!(
                "collected {} objects, {} bytes; dropped {} pins",
                collected.objects,
                collected.bytes,
                collected.pins.len()
            );
            Ok(Some((doc, human)))
        }
        ["dev", "ckpt"] => {
            let store = store()?;
            let plain = args.on("plain");
            let scale = args.number("scale", 1)?.max(1);
            let manifest = dev::ckpt(
                &store,
                args.number("blocks", 12)?,
                scale,
                plain,
                !args.on("no-configs"),
            )?;
            let name = args.get("name").unwrap_or("dev/ckpt");
            let n = store.set_lane(
                name,
                "1",
                Some(if plain { "bf16" } else { "mixed" }),
                &manifest,
            )?;
            store.unpin(&manifest.sha256)?;
            let ck = store.checkpoint(&manifest)?;
            let mut objects: Vec<ObjectRef> = vec![manifest.clone()];
            objects.extend(ck.manifest().files.values().cloned());
            objects.extend(ck.header().chunks.iter().cloned());
            objects.sort();
            objects.dedup();
            let mut doc = named(&n);
            doc.insert("objects".into(), objects.iter().map(object).collect());
            let human = format!("wrote {} ({} objects)", label(&n), objects.len());
            Ok(Some((Json::Object(doc), human)))
        }
        ["dev", "migrate", root, select @ ..] => {
            let store = match args.on("plan") {
                true => None,
                false => Some(store()?),
            };
            migrate_v1(store.as_ref(), root, select, args.on("dry-run"))
        }
        ["dev", "headers", hub, dir, lanes @ ..] if !lanes.is_empty() => headers(hub, dir, lanes),
        _ => Ok(None),
    }
}

/// Migrates the v1 checkpoints `select` names (a manifest digest or a repo), or every one the
/// v1 store's repos name; one report per checkpoint and the totals.
fn migrate_v1(store: Option<&Store>, root: &str, select: &[&str], dry_run: bool) -> Done {
    let src = old::Source::new(root);
    let named = match src.root().join("repos").is_dir() {
        true => src.lanes()?,
        false => vec![],
    };
    let mut lanes: Vec<old::Lane> = Vec::new();
    for s in select {
        let repo = |n: &String| n == s || n.starts_with(&format!("{s}@"));
        let found: Vec<old::Lane> = match s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
        {
            true => match named.iter().find(|l| l.manifest.sha256 == *s) {
                Some(lane) => vec![lane.clone()],
                None => vec![old::Lane {
                    manifest: src.manifest_ref(s)?,
                    names: vec![],
                }],
            },
            false => named
                .iter()
                .filter(|l| l.names.iter().any(repo))
                .cloned()
                .collect(),
        };
        if found.is_empty() {
            return Err(Failure {
                kind: Kind::NotFound,
                detail: format!("no v1 checkpoint {s}"),
            }
            .into());
        }
        for lane in found {
            if !lanes.iter().any(|l| l.manifest == lane.manifest) {
                lanes.push(lane);
            }
        }
    }
    if select.is_empty() {
        lanes = named;
    }
    let mut reports = Vec::new();
    for (i, lane) in lanes.iter().enumerate() {
        eprintln!(
            "[{}/{}] {} {:?}",
            i + 1,
            lanes.len(),
            lane.manifest.sha256,
            lane.names
        );
        let report = migrate::lane(&src, store, lane, dry_run);
        if let Some(error) = report.doc.get("error") {
            eprintln!("  {error}");
        }
        reports.push(report);
    }
    let totals = migrate::totals(&reports);
    let human = format!(
        "{} checkpoints, {} passed; {} new objects, {} new bytes",
        reports.len(),
        totals["passed"],
        totals["new"]["objects"],
        totals["new"]["bytes"]
    );
    let checkpoints: Vec<Json> = reports
        .iter()
        .map(|r| Json::Object(r.doc.clone()))
        .collect();
    // Each model's refs file from its own rows, once every checkpoint it names has migrated.
    let mut refs = Vec::new();
    if let (Some(store), false) = (store, dry_run) {
        let wanted: Vec<&ObjectRef> = lanes.iter().map(|l| &l.manifest).collect();
        let repos = match src.root().join("repos").is_dir() {
            true => src.repos()?,
            false => vec![],
        };
        for repo in repos {
            let named = repo
                .releases
                .iter()
                .flat_map(|r| r.lanes.iter().map(|l| &l.1));
            if !named.chain(&repo.checkpoints).any(|m| wanted.contains(&m)) {
                continue;
            }
            refs.push(match migrate::refs(&repo, &reports) {
                Ok(v2) => {
                    store.update_ref(&v2.name, |_| Some(v2.clone()))?;
                    json!({"name": v2.name, "written": true, "releases": v2.releases.len(), "checkpoints": v2.checkpoints.len()})
                }
                Err(missing) => json!({"name": repo.name, "written": false, "not_migrated": missing}),
            });
        }
    }
    let doc = json!({"checkpoints": checkpoints, "refs": refs, "totals": totals});
    match reports.iter().all(|r| r.pass) {
        true => Ok(Some((doc, human))),
        false => Err(Failed(
            Failure {
                kind: Kind::Integrity,
                detail: format!("{human}; some did not migrate"),
            },
            doc.as_object().cloned(),
        )),
    }
}

/// Each released lane's v1 manifest and header from the Hub at `hub`, into `dir`.
fn headers(hub: &str, dir: &str, lanes: &[&str]) -> Done {
    fs::create_dir_all(dir).map_err(|e| Failure {
        kind: Kind::Io,
        detail: format!("{dir}: {e}"),
    })?;
    let mut docs = Vec::new();
    for spec in lanes {
        let Some((refspec, lane)) = spec.rsplit_once('/').filter(|(r, _)| r.contains('@')) else {
            return Ok(None);
        };
        let got = migrate::hub::headers(hub, refspec, lane, Path::new(dir))?;
        eprintln!("{spec}: {} ({} bytes)", got.manifest.sha256, got.bytes);
        docs.push(json!({"lane": spec, "release": got.release, "manifest": object(&got.manifest), "bytes": got.bytes}));
    }
    let bytes: u64 = docs.iter().filter_map(|d| d["bytes"].as_u64()).sum();
    let human = format!(
        "{} lanes, {bytes} bytes of manifests and headers",
        docs.len()
    );
    Ok(Some((json!({"lanes": docs, "bytes": bytes}), human)))
}

/// `7d`, `12h`, `30m`, `90s` or seconds.
fn duration(text: &str) -> Result<Duration> {
    let (number, unit) = text.split_at(text.trim_end_matches(char::is_alphabetic).len());
    let scale = match unit {
        "" | "s" => 1,
        "m" => 60,
        "h" => 3600,
        "d" => 86400,
        _ => 0,
    };
    match number.parse::<u64>() {
        Ok(n) if scale > 0 => Ok(Duration::from_secs(n * scale)),
        _ => fail(
            Kind::Invalid,
            format!("--older-than {text:?}: a number of s, m, h or d"),
        ),
    }
}

fn inspect(store: &Store, reference: &str, lane: Option<&str>) -> Done {
    let n = store.resolve(reference, lane)?;
    let ck = store.checkpoint(&n.manifest)?;
    let header = ck.header();
    let mut human = vec![label(&n)];
    // Arrays in header (construction) order.
    let mut components = Vec::new();
    for c in header.components() {
        let mut tensors = Vec::new();
        let mut bytes = 0;
        // Each distinct encoding (`plain` for none) and its tensor count, in order of first use.
        let mut encodings: Vec<(String, usize)> = Vec::new();
        for t in &c.tensors {
            let mut doc = json!({"key": t.key, "dtype": t.dtype.name(), "shape": t.shape});
            let shown = match &t.body {
                Body::Data(at) => {
                    bytes += at.length;
                    "plain".to_string()
                }
                Body::Encoded { encoding, parts } => {
                    bytes += parts.iter().map(|p| p.length).sum::<u64>();
                    doc["encoding"] = encoding.to_string().into();
                    encoding.to_string()
                }
            };
            match encodings.iter_mut().find(|(e, _)| *e == shown) {
                Some((_, n)) => *n += 1,
                None => encodings.push((shown, 1)),
            }
            tensors.push(doc);
        }
        human.push(format!(
            "  {}: {} tensors, {bytes} bytes",
            c.name,
            c.tensors.len()
        ));
        human.extend(encodings.iter().map(|(e, n)| format!("    {n} × {e}")));
        components.push(json!({"name": c.name, "bytes": bytes, "tensors": tensors}));
    }
    let chunk_bytes: u64 = header.chunks.iter().map(|c| c.length).sum();
    human.push(format!(
        "  {} chunks, {chunk_bytes} bytes; {} files",
        header.chunks.len(),
        ck.manifest().files.len()
    ));
    let files: Map<String, Json> = ck
        .manifest()
        .files
        .iter()
        .map(|(p, o)| (p.clone(), object(o)))
        .collect();
    let mut doc = named(&n);
    doc.extend([
        ("files".into(), Json::Object(files)),
        ("chunks".into(), header.chunks.len().into()),
        ("bytes".into(), chunk_bytes.into()),
        ("metadata".into(), json!(header.metadata)),
        ("components".into(), Json::Array(components)),
    ]);
    Ok(Some((Json::Object(doc), human.join("\n"))))
}

/// Rehashes every object (or a checkpoint's): a corrupt one is removed, to be fetched again.
fn verify(store: &Store, reference: Option<(&str, Option<&str>)>) -> Done {
    let objects: Vec<ObjectRef> = match reference {
        Some((reference, lane)) => {
            let n = store.resolve(reference, lane)?;
            let ck = store.checkpoint(&n.manifest)?;
            let mut all = vec![n.manifest.clone()];
            all.extend(ck.manifest().files.values().cloned());
            all.extend(ck.header().chunks.iter().cloned());
            all.sort();
            all.dedup();
            all
        }
        None => blobs(store)?,
    };
    let (mut corrupt, mut missing, mut bytes) = (Vec::new(), Vec::new(), 0);
    for o in &objects {
        if !store.path(o)?.exists() {
            missing.push(o.sha256.clone());
        } else if store.verify(o)? {
            bytes += o.length;
        } else {
            corrupt.push(o.sha256.clone());
        }
    }
    let mut doc = Map::new();
    doc.extend([
        ("objects".into(), objects.len().into()),
        ("bytes".into(), bytes.into()),
        ("corrupt".into(), corrupt.clone().into()),
        ("missing".into(), missing.clone().into()),
    ]);
    if corrupt.is_empty() && missing.is_empty() {
        let human = format!("verified {} objects, {bytes} bytes", objects.len());
        return Ok(Some((Json::Object(doc), human)));
    }
    let detail = format!(
        "{} corrupt (removed) and {} missing of {} objects",
        corrupt.len(),
        missing.len(),
        objects.len()
    );
    Err(Failed(
        Failure {
            kind: Kind::Integrity,
            detail,
        },
        Some(doc),
    ))
}

/// Every object file in the store, at its length on disk.
fn blobs(store: &Store) -> Result<Vec<ObjectRef>> {
    let mut out = Vec::new();
    let list = |dir: &Path| -> Result<Vec<fs::DirEntry>> {
        match fs::read_dir(dir) {
            Ok(entries) => Ok(entries.flatten().collect()),
            Err(e) => fail(Kind::Io, format!("{}: {e}", dir.display())),
        }
    };
    for top in list(&store.root().join("blobs"))? {
        for shard in list(&top.path())? {
            for blob in list(&shard.path())? {
                let sha256 = blob.file_name().to_string_lossy().into_owned();
                if let Ok(m) = blob.metadata() {
                    out.push(ObjectRef {
                        sha256,
                        length: m.len(),
                    });
                }
            }
        }
    }
    out.sort();
    Ok(out)
}

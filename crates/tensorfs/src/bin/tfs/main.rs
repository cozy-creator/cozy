//! `tfs`: the Python verbs on the command line (`convert`, `download`, `upload`, `export`,
//! `inspect`, `verify`), plus `ceiling`, `conform endpoint`, `dev ckpt` and `version`.

mod dev;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value as Json};
use tensorfs::failure::{fail, Failure, Kind, Result};
use tensorfs::format::{Body, ClassRef, ObjectRef, Tree};
use tensorfs::store::Store;
use tensorfs::{fetch, Checkpoint, Client, Config, Convert, Download, Format, Open, Ref, Selection};

const VERBS: &[&str] = &[
    "convert <source>... [--converter <c>] [--source-variant <v>] [--name local/<name>[@tag]]",
    "download <ref> [--dir <dir>]",
    "upload <folder> <org/name>[@tag] [--delete <glob>]...  commits the folder onto the tag (main); the token is $TENSORFS_TOKEN",
    "export <ref> <dir> [--safetensors [--decode] | --gguf]",
    "inspect <ref>",
    "verify <ref>",
    "ceiling [<path>] [--gpu <n>]              measure what reads from <path>'s volume (and to GPU n) reach",
    "conform endpoint <url> [--write] [--token-env <VAR>]",
    "dev ckpt [--name local/<name>[@tag]] [--blocks <n>] [--scale <n>] [--plain] [--no-configs]",
    "version",
];

fn known(verb: &str) -> bool {
    VERBS.iter().any(|l| l.split(' ').next() == Some(verb))
}

/// The usage of `verb`, or of every verb when it names none.
fn usage(verb: Option<&str>) -> String {
    let verb = verb.filter(|v| known(v));
    let mut text = String::from("usage: tfs [--store <dir>] [--endpoint <url>] [--json] <verb>");
    for line in VERBS.iter().filter(|l| verb.is_none_or(|v| l.split(' ').next() == Some(v))) {
        text.push_str("\n  ");
        text.push_str(line);
    }
    text.push_str(
        "\n<ref> is org/name[@revision][/variant], sha256:<hex>, a .cozytensors file or a folder; a revision is
a tag, or sha256:<hex> of a commit or a variant header, and none is the highest public version;
--variant picks a variant; --source-variant a source's weights variant (fp16). The store is --store,
else $TENSORFS_HOME, else ~/.cache/tensorfs.",
    );
    text
}

const SWITCHES: &[&str] = &["help", "json", "safetensors", "decode", "gguf", "plain", "no-configs", "write"];
const VALUED: &[&str] = &[
    "store", "endpoint", "name", "converter", "variant", "source-variant", "blocks", "scale",
    "token-env", "dir", "gpu", "delete",
];

struct Args {
    pos: Vec<String>,
    flags: Vec<(String, String)>,
}

impl Args {
    fn parse(argv: &[String]) -> std::result::Result<Args, String> {
        let (mut pos, mut flags) = (Vec::new(), Vec::new());
        let mut it = argv.iter();
        while let Some(token) = it.next() {
            let Some(name) = token.strip_prefix("--").or((token == "-h").then_some("help")) else {
                if token.len() > 1 && token.starts_with('-') {
                    return Err(format!("unknown flag {token}"));
                }
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
        self.flags.iter().rev().find(|(n, _)| n == name).map(|(_, v)| v.as_str())
    }

    fn all(&self, name: &str) -> Vec<&str> {
        self.flags.iter().filter(|(n, _)| n == name).map(|(_, v)| v.as_str()).collect()
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
    tracing_subscriber::fmt().with_writer(std::io::stderr).init();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match Args::parse(&argv) {
        Ok(args) => args,
        Err(why) => {
            eprintln!("tfs: {why}\n{}", usage(None));
            return ExitCode::from(2);
        }
    };
    let verb = args.pos.first().map(String::as_str);
    if args.on("help") || verb == Some("help") {
        let topic = args.pos.get(usize::from(verb == Some("help"))).map(String::as_str);
        println!("{}", usage(topic));
        return ExitCode::SUCCESS;
    }
    // Before the store opens, so a typo never depends on its state.
    let Some(verb) = verb.filter(|v| known(v)) else {
        let why = verb.map_or("no verb".to_owned(), |v| format!("unknown verb {v:?}"));
        eprintln!("tfs: {why}\n{}", usage(None));
        return ExitCode::from(2);
    };
    let json = args.on("json");
    let ran = fetch::runtime().map_err(Failure::from).and_then(|rt| rt.block_on(run(&args)));
    match ran {
        Ok(Some((doc, human))) => {
            println!("{}", if json { doc.to_string() } else { human });
            ExitCode::SUCCESS
        }
        Ok(None) => {
            eprintln!("tfs: {verb}: wrong arguments\n{}", usage(Some(verb)));
            ExitCode::from(2)
        }
        Err(f) => {
            match json {
                true => println!("{}", json!({"error": {"code": f.kind.code(), "detail": f.detail}})),
                false => eprintln!("tfs: {f}"),
            }
            ExitCode::FAILURE
        }
    }
}

type Done = Result<Option<(Json, String)>>;

fn config(args: &Args) -> Result<Config> {
    let mut config = Config::from_env();
    if let Some(dir) = args.get("store") {
        config.cache_dir = PathBuf::from(dir);
    }
    if let Some(url) = args.get("endpoint") {
        config.endpoint = url.to_owned();
    }
    Ok(config)
}

fn object(o: &ObjectRef) -> Json {
    json!({"sha256": o.sha256, "length": o.length})
}

fn identity(ck: &Checkpoint) -> Map<String, Json> {
    let doc = json!({"header": object(ck.id()), "tree": ck.tree().map(object), "variant": ck.variant()});
    doc.as_object().expect("an object").clone()
}

async fn open(client: &Client, args: &Args, reference: &str) -> Result<Checkpoint> {
    let options = Open { variant: args.get("variant").map(str::to_owned), ..Open::default() };
    client.open(reference, &options).await
}

async fn run(args: &Args) -> Done {
    let pos: Vec<&str> = args.pos.iter().map(String::as_str).collect();
    match pos.as_slice() {
        ["version"] => return version(args.on("json")).map(Some),
        ["conform", "endpoint", url] => {
            return conform(url, args.on("write"), args.get("token-env"), args.on("json")).await
        }
        _ => {}
    }
    let config = config(args)?;
    let root = config.cache_dir.clone();
    let client = Client::new(config)?;
    match pos.as_slice() {
        ["convert", sources @ ..] if !sources.is_empty() => {
            let sources: Vec<String> = sources.iter().map(|s| s.to_string()).collect();
            let quiet = args.on("json");
            let last = Mutex::new((String::new(), Instant::now()));
            let progress = |stage: &str, done: u64, total: u64| -> Result<()> {
                let mut last = last.lock().unwrap_or_else(|e| e.into_inner());
                if !quiet && (last.0 != stage || last.1.elapsed() >= Duration::from_secs(2)) {
                    *last = (stage.to_owned(), Instant::now());
                    eprintln!("{stage} {done}/{total}");
                }
                Ok(())
            };
            let options = Convert {
                converter: args.get("converter"),
                source_variant: args.get("source-variant"),
                progress: Some(&progress),
            };
            let ck = client.convert_to_cozytensors(&sources, &options).await?;
            if let Some(name) = args.get("name") {
                let tree = ck.tree().expect("a conversion names its tree").clone();
                name_locally(&root, &Store::open(&root)?, name, &tree).await?;
            }
            Ok(Some((Json::Object(identity(&ck)), format!("converted sha256:{}", ck.id().sha256))))
        }
        ["download", reference] => {
            let reference: Ref = reference.parse()?;
            let dir = args.get("dir").map(Path::new);
            let folder = client.materialize(&reference, &Download::default(), dir, false).await?;
            Ok(Some((json!({"path": folder}), format!("downloaded into {}", folder.display()))))
        }
        ["upload", folder, reference] => {
            let reference: Ref = reference.parse()?;
            let commit = client.upload_folder(&reference, Path::new(folder), &args.all("delete"), None).await?;
            let tag = reference.revision.as_deref().unwrap_or(fetch::MAIN);
            let human = format!("uploaded sha256:{}, {}@{tag}", commit.sha256, reference.repo_id);
            Ok(Some((json!({"commit": object(&commit), "tag": tag}), human)))
        }
        ["export", reference, dir] => {
            let ck = open(&client, args, reference).await?;
            let dir = Path::new(dir);
            ck.download(&Download::default()).await?;
            let files: Vec<String> = match (args.on("safetensors"), args.on("gguf"), args.on("decode")) {
                (true, false, decode) => {
                    ck.materialize(dir, Format::Safetensors, decode).await?;
                    listing(dir)?
                }
                (false, true, false) => tensorfs::export::gguf(&Store::open(&root)?.checkpoint(ck.id())?, dir)?,
                (false, false, false) => {
                    ck.materialize(dir, Format::Cozytensors, false).await?;
                    listing(dir)?
                }
                _ => return Ok(None),
            };
            let human = format!("exported into {}: {} files", dir.display(), files.len());
            let mut doc = identity(&ck);
            doc.insert("files".into(), files.into());
            Ok(Some((Json::Object(doc), human)))
        }
        ["inspect", reference] => inspect(&open(&client, args, reference).await?),
        ["verify", reference] => {
            let ck = open(&client, args, reference).await?;
            let report = ck.verify(&Selection::All)?;
            let ids = |v: &[ObjectRef]| v.iter().map(|o| o.sha256.clone()).collect::<Vec<_>>();
            let mut doc = identity(&ck);
            doc.extend([("corrupt".into(), ids(&report.corrupt).into()), ("missing".into(), ids(&report.missing).into())]);
            match report.is_ok() {
                true => Ok(Some((Json::Object(doc), "verified".into()))),
                false => fail(
                    Kind::Integrity,
                    format!("{} corrupt and {} missing objects: {}", report.corrupt.len(), report.missing.len(), Json::Object(doc)),
                ),
            }
        }
        ["ceiling", path @ ..] if path.len() <= 1 => {
            let path = path.first().map_or(root, PathBuf::from);
            let gpu = args.get("gpu").map(|_| args.number("gpu", 0)).transpose()?.map(|n| n as i32);
            let c = tensorfs::io::ceiling(gpu, &path).map_err(|e| Failure {
                kind: if e.kind() == std::io::ErrorKind::Unsupported { Kind::Unsupported } else { Kind::Io },
                detail: format!("{}: {e}", path.display()),
            })?;
            let doc = json!({"disk_gbps": c.disk_gbps, "disk_direct": c.disk_direct, "h2d_gbps": c.h2d_gbps, "streams": c.streams});
            let mut human = format!("disk {:.2} GB/s{}", c.disk_gbps, if c.disk_direct { "" } else { " (through the page cache)" });
            if let Some(h2d) = c.h2d_gbps {
                human += &format!("\nh2d {h2d:.2} GB/s on {} streams", c.streams.unwrap_or(1));
            }
            Ok(Some((doc, human)))
        }
        ["dev", "ckpt"] => {
            let store = Store::open(&root)?;
            let plain = args.on("plain");
            let scale = args.number("scale", 1)?.max(1);
            let header = dev::ckpt(&store, args.number("blocks", 12)?, scale, plain, !args.on("no-configs"))?;
            let mut tree = Tree::default();
            tree.set_variant(if plain { "bf16" } else { "mixed" }, header.clone());
            let tree = store.put_tree(&tree)?;
            let name = args.get("name").unwrap_or("local/ckpt@1");
            let commit = name_locally(&root, &store, name, &tree).await?;
            let objects = store.closure(&tree)?;
            let doc = json!({"header": object(&header), "tree": object(&tree), "commit": object(&commit), "name": name, "objects": objects.iter().map(object).collect::<Vec<_>>()});
            Ok(Some((doc, format!("wrote {name}: sha256:{} ({} objects)", header.sha256, objects.len()))))
        }
        _ => Ok(None),
    }
}

/// Commits `tree`'s files on top of tag `@tag` (none: `main`) of the local repo
/// `local/<name>`, through the store's own endpoint: the one commit path. The commit.
async fn name_locally(root: &Path, store: &Store, name: &str, tree: &ObjectRef) -> Result<ObjectRef> {
    let named: Ref = name.parse()?;
    let tag = named.revision.as_deref().unwrap_or(fetch::MAIN);
    if !named.repo_id.starts_with("local/") || named.variant.is_some() || !tensorfs::format::tag_name(tag) {
        return fail(Kind::Invalid, format!("--name {name:?}: a local name is local/<name>[@tag]"));
    }
    let root = std::path::absolute(root).map_err(Failure::from)?;
    let local = fetch::Endpoint::new(&fetch::Client::new(fetch::SourcePolicy::default()), &root.to_string_lossy(), None)?;
    Ok(fetch::upload_tree(&local, &named.repo_id, tree, store, tag).await?.commit)
}

fn listing(dir: &Path) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(at) = stack.pop() {
        let entries = fs::read_dir(&at).map_err(|e| Failure { kind: Kind::Io, detail: format!("{}: {e}", at.display()) })?;
        for entry in entries.flatten() {
            let path = entry.path();
            match path.is_dir() {
                true => stack.push(path),
                false => out.push(path.strip_prefix(dir).unwrap_or(&path).to_string_lossy().into_owned()),
            }
        }
    }
    out.sort();
    Ok(out)
}

/// `tfs <version> sha256:<digest of this binary>`: the skew observable callers refuse on.
fn version(json: bool) -> Result<(Json, String)> {
    let bytes = std::env::current_exe()
        .and_then(fs::read)
        .map_err(|e| Failure { kind: Kind::Io, detail: format!("read this binary: {e}") })?;
    let digest = tensorfs::sha256::hex_digest(&bytes);
    let human = format!("tfs {} sha256:{digest}", env!("CARGO_PKG_VERSION"));
    let doc = json!({"version": env!("CARGO_PKG_VERSION"), "sha256": digest, "store_format": tensorfs::STORE_FORMAT});
    Ok(match json {
        true => (doc, String::new()),
        false => (doc, human),
    })
}

fn inspect(ck: &Checkpoint) -> Done {
    let header = ck.header();
    let mut human = vec![format!("sha256:{}", ck.id().sha256)];
    let mut components = Vec::new();
    for c in header.components() {
        let (mut tensors, mut bytes) = (Vec::new(), 0);
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
        human.push(format!("  {}: {} tensors, {bytes} bytes", c.name, c.tensors.len()));
        human.extend(encodings.iter().map(|(e, n)| format!("    {n} × {e}")));
        let assets: Map<String, Json> = c.assets.iter().map(|(p, o)| (p.clone(), object(o))).collect();
        components.push(json!({"name": c.name, "class": class(&c.class), "config": c.config, "assets": assets, "bytes": bytes, "tensors": tensors}));
    }
    let chunk_bytes: u64 = header.chunks.iter().map(|c| c.length).sum();
    human.push(format!("  {} chunks, {chunk_bytes} bytes", header.chunks.len()));
    let mut doc = identity(ck);
    doc.extend([
        ("pipeline".into(), json!(header.pipeline().map(|p| json!({"class": class(&p.class), "config": p.config})))),
        ("chunks".into(), header.chunks.len().into()),
        ("bytes".into(), chunk_bytes.into()),
        ("metadata".into(), json!(header.metadata)),
        ("components".into(), Json::Array(components)),
    ]);
    Ok(Some((Json::Object(doc), human.join("\n"))))
}

/// A class as the header writes it, `[library, version, name]`.
fn class(c: &ClassRef) -> Json {
    json!([c.library, c.version, c.name])
}

/// `tfs conform endpoint`: every case, and a failure when a MUST failed.
async fn conform(url: &str, write: bool, token_env: Option<&str>, json: bool) -> Done {
    let token = match token_env {
        Some(var) => match std::env::var(var).ok().filter(|t| !t.is_empty()) {
            Some(token) => Some(fetch::bearer(token)),
            None => return fail(Kind::Invalid, format!("${var} is unset")),
        },
        None => None,
    };
    let scratch = std::env::temp_dir().join(format!("tfs-conform-{}", std::process::id()));
    let _ = fs::remove_dir_all(&scratch);
    let policy = fetch::SourcePolicy {
        any_host: true,
        allow_local: fetch::policy::is_loopback_origin(url),
        max_redirects: 5,
        ..Default::default()
    };
    let client = fetch::Client::new(policy);
    let cases = async {
        let endpoint = std::sync::Arc::new(fetch::Endpoint::new(&client, url, token)?);
        fetch::conform::endpoint(endpoint, &client, write, &scratch).await
    }
    .await;
    let _ = fs::remove_dir_all(&scratch);
    let cases = cases.map_err(Failure::from)?;
    let mark = |c: &fetch::conform::Case| match (c.ok, c.must) {
        (None, _) => "n/a ",
        (Some(true), _) => "pass",
        (Some(false), true) => "FAIL",
        (Some(false), false) => "warn",
    };
    let human: Vec<String> = cases.iter().map(|c| format!("{}  {}  {}", mark(c), c.name, c.detail).trim_end().to_owned()).collect();
    let docs: Vec<Json> = cases.iter().map(|c| json!({"case": c.name, "must": c.must, "ok": c.ok, "detail": c.detail})).collect();
    let failed = cases.iter().filter(|c| c.must && c.ok == Some(false)).count();
    let doc = json!({"endpoint": url, "cases": docs, "failed": failed});
    if failed == 0 {
        return Ok(Some((doc, human.join("\n"))));
    }
    if !json {
        eprintln!("{}", human.join("\n"));
    }
    fail(Kind::Invalid, format!("{url}: {failed} MUST case(s) failed: {doc}"))
}

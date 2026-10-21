//! The model ingest cycle on a real Tensorhub with real machines: Tensorhub's dev-build `cozy_ref`
//! job (TENSORHUB_URL, TENSORHUB_LOG; MinIO storage; its public origin behind a private CA) and the
//! TensorD COZY_TEST_TENSORD names (this workspace's). The machines trust that CA
//! only as each run's Hub access names it, never through their environment, so every Hub call a
//! run makes proves it: the upload, a private checkpoint's read under the owner's bearer, a
//! package's card and install plan. The machine pins, makes, quantizes
//! and uploads every byte; this computer only asks.
//!
//! upload → private download elsewhere → publish → info → family → quantize (a package's
//! quantizer) → move the release to the commit that adds its lane → yank → delete.
#![cfg(unix)]
mod common;

use std::path::{Path, PathBuf};

use common::{Home, Out};
use serde_json::Value;
use std::sync::Arc;

use tensorfs::{
    err::{refuse, Code, Result},
    fetch::{bearer, Client, Config, Endpoint, Fetcher, Origin, SourcePolicy, Want},
    format::ObjectRef,
    store::Store,
};

/// `model`'s tree `tree` (`sha256:<hex>`) from the Hub's distribution endpoint through TensorFS's
/// fetcher: the library's own pull. The repo document names it only to its owner.
fn pull(store: &Store, client: &Client, hub: Endpoint, model: &str, tree: &str) -> Result<ObjectRef> {
    let hub = Arc::new(hub);
    tensorfs::fetch::runtime()?.block_on(async {
        let repo = hub.repo(model).await?.map(|(repo, _)| repo);
        let Some(named) = repo.iter().flat_map(|repo| repo.roots()).find(|root| root.id() == tree).cloned() else {
            return refuse(Code::REF_NOT_FOUND, format!("{model} names no {tree} here"));
        };
        let origin: Arc<dyn Origin> = hub.clone();
        let fetcher = Fetcher::new(Arc::new(store.clone()), client.clone(), None, Config::default());
        fetcher.pull_tree(&origin, &named, None, &Want::default()).await?;
        fetcher.settled().await;
        Ok(named)
    })
}

/// `cozy args` in `dir`. Each step is named, timestamped and flushed before it starts, and its
/// output streams into the log as it comes, so the job's own timeout shows the step it stopped in.
fn cozy(home: &Home, dir: Option<&Path>, args: &[&str]) -> Out {
    use std::io::{BufRead as _, Write as _};
    let at = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
    println!("step {at}: cozy {}", args.join(" "));
    let _ = std::io::stdout().flush();
    let mut command = home.command();
    if let Some(dir) = dir {
        command.current_dir(dir);
    }
    let mut child = command
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .unwrap();
    let mut stdout = String::new();
    for line in std::io::BufReader::new(child.stdout.take().unwrap()).lines() {
        let line = line.unwrap();
        println!("  | {line}");
        stdout.push_str(&line);
        stdout.push('\n');
    }
    let code = child.wait().unwrap().code().unwrap_or(-1);
    Out { code, stdout, stderr: String::new() }
}

/// What Calcifer and its machines say now: printed when a step fails.
fn diagnose(home: &Home) {
    for args in [&["run", "list", "--limit", "5"][..], &["machine", "list"]] {
        let out = common::out(home.command().args(args).output().unwrap());
        println!("cozy {}:\n{}{}", args.join(" "), out.stdout, out.stderr);
    }
    let log = std::fs::read_to_string(home.root().join("calcifer/calcifer.log")).unwrap_or_default();
    let tail: Vec<&str> = log.lines().rev().take(50).collect();
    println!("calcifer.log (last 50 lines):\n{}", tail.into_iter().rev().collect::<Vec<_>>().join("\n"));
}

/// A step that succeeds, and its JSON document.
fn json(home: &Home, args: &[&str]) -> Value {
    let out = cozy(home, None, args);
    if out.code != 0 {
        diagnose(home);
    }
    assert_eq!(out.code, 0, "{args:?}: {out:?}");
    serde_json::from_str(&out.stdout).unwrap_or_else(|e| panic!("{e}: {out:?}"))
}

/// Pinned, so a released machine takes it; `model_machine` covers a machine pinning it.
const TINY: &str = "hf://hf-internal-testing/tiny-random-gpt2@71034c5d8bde858ff824298bdedc65515b97d2b9";

/// The last NDJSON event a followed `--json` run printed: its end.
fn end(out: &Out) -> Value {
    let line = out.stdout.lines().last().unwrap_or_else(|| panic!("no events: {out:?}"));
    serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {out:?}"))
}

fn completed(home: &Home, out: &Out) -> Value {
    let ended = end(out);
    if ended["type"] != "run.completed" {
        diagnose(home);
    }
    assert_eq!(ended["type"], "run.completed", "{out:?}");
    ended["run"].clone()
}

fn copy(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        match entry.file_type().unwrap().is_dir() {
            true => copy(&entry.path(), &to.join(entry.file_name())),
            false => drop(std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap()),
        }
    }
}

/// `tests/fixtures/model/tiny-quantizer`, locked and published by this home's account.
fn publish_quantizer(home: &Home) {
    let project: PathBuf = home.path().join("src/tiny-quantizer");
    copy(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/model/tiny-quantizer"), &project);
    for verb in ["lock", "publish"] {
        let out = cozy(home, Some(&project), &["--json", "package", verb]);
        if out.code != 0 {
            diagnose(home);
        }
        assert_eq!(out.code, 0, "package {verb}: {out:?}");
    }
}

/// A machine added to this home; every Hub call it makes trusts the CA through the run's access.
fn machine(home: &Home, name: &str) -> common::tensord::Tensord {
    common::tensord::added(home, name)
}

/// Every object `manifest` reaches in the store at `root` (manifest, files, header,
/// chunks), with its bytes.
fn closure(root: &Path, manifest: &str) -> Vec<(String, Vec<u8>)> {
    let store = Store::open(root).unwrap();
    let sha256 = manifest.trim_start_matches("sha256:").to_string();
    let length = std::fs::metadata(store.path(&ObjectRef { sha256: sha256.clone(), length: 0 }).unwrap()).unwrap().len();
    let mut objects: Vec<_> = (store.closure(&ObjectRef { sha256, length }).unwrap().into_iter())
        .map(|object| (object.sha256.clone(), store.read(&object).unwrap()))
        .collect();
    objects.sort();
    objects
}

/// Asset `name` of the root `model` component of checkpoint `header` in the store at `root`.
fn asset(root: &Path, header: &str, name: &str) -> Vec<u8> {
    let store = Store::open(root).unwrap();
    let sha256 = header.trim_start_matches("sha256:").to_string();
    let length = std::fs::metadata(store.path(&ObjectRef { sha256: sha256.clone(), length: 0 }).unwrap()).unwrap().len();
    let checkpoint = store.checkpoint(&ObjectRef { sha256, length }).unwrap();
    checkpoint.asset("model", name).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// TensorFS end to end (xs-045 U9a): one machine imports a Hugging Face model into its store
/// and puts it in the Hub's custody; a second, fresh machine pulls it by its digest under the
/// run's owner bearer, and the library's distribution `Endpoint` pulls it into a third store
/// with that bearer (and is refused without one). Both machines hold the same checkpoint, and
/// the library's pull the same tree, byte for byte.
#[test]
fn a_model_uploaded_by_one_machine_is_pulled_bit_identical_by_another() {
    let Some(home) = Home::with_hub() else { return };
    let account = home.sign_in("tfscycle@cozy.test");
    let first = machine(&home, "m1");
    let model = format!("{account}/tiny-cycle");
    let run = completed(&home, &cozy(&home, None, &["--json", "model", "upload", TINY, &model, "--on", "m1"]));
    let made = &run["result"]["models"][0];
    assert_eq!(made["published"]["destination"], format!("{model}@main"), "{run}");
    // The commit (the tree `main` moved to) and the checkpoint (its header) it names.
    let source = made["published"]["tree"].as_str().unwrap().to_owned();
    let header = made["published"]["checkpoint"].as_str().unwrap().to_owned();

    // A machine pulls the commit's checkpoint (its header and chunks), not its repository files.
    let second = machine(&home, "m2");
    let checkpoint = format!("{model}#{source}");
    let run = completed(&home, &cozy(&home, None, &["--json", "model", "download", &checkpoint, "--on", "m2"]));
    let pulled = &run["result"]["models"][0];
    assert_eq!((pulled["tree"].as_str(), pulled["checkpoint"].as_str()), (Some(source.as_str()), Some(header.as_str())), "{run}");
    let (made, pulled) = (closure(&first.store(), &header), closure(&second.store(), &header));
    assert!(made.len() >= 2, "a header and its chunks: {}", made.len());
    assert!(made == pulled, "the pulled checkpoint differs from the uploaded one");
    // The tokenizer files are the root component's assets: they travel with the checkpoint.
    for (name, sha256) in [
        ("tokenizer.json", "e62fc5f642751cf5d0bf542adb6aba88624b696f26f5f40aff7e4a53788eb85c"),
        ("vocab.json", "eda8bd11b5763cec815aa93b5524cf7aeed42b49c772134355451460f46c299c"),
        ("merges.txt", "150ce8950a91d89c2711cc5d0ca54895236a479a36913137409ac254806bf468"),
    ] {
        let (made, pulled) = (asset(&first.store(), &header, name), asset(&second.store(), &header, name));
        assert!(made == pulled && tensorfs::sha256::hex_digest(&pulled) == sha256, "{name} differs");
    }
    let made = closure(&first.store(), &source);

    let ca = std::fs::read(std::env::var("TENSORHUB_CA").expect("the Hub's CA")).unwrap();
    let url = std::env::var("TENSORHUB_ORIGIN").unwrap();
    let policy = SourcePolicy { any_host: true, allow_local: true, max_redirects: 5, ..SourcePolicy::default() };
    let client = Client::with_roots(policy, &ca).unwrap();
    let third = tempfile::tempdir().unwrap();
    let store = Store::open(third.path()).unwrap();
    let anonymous = pull(&store, &client, Endpoint::new(&client, &url, None).unwrap().anonymous(), &model, &source);
    assert!(anonymous.is_err(), "an owner's checkpoint read anonymously: {anonymous:?}");
    let hub = Endpoint::new(&client, &url, Some(bearer(home.token()))).unwrap();
    assert_eq!(pull(&store, &client, hub, &model, &source).unwrap().id(), source);
    assert!(closure(third.path(), &source) == made, "the library's pull differs from the uploaded checkpoint");
}

#[test]
fn a_model_is_uploaded_quantized_and_published_by_its_machine() {
    let Some(home) = Home::with_hub() else { return };
    let account = home.sign_in("modelcycle@cozy.test");
    let _machine = machine(&home, "m1");
    let model = format!("{account}/tiny");

    // The machine makes the source and uploads a private checkpoint.
    let run = completed(&home, &cozy(&home, None, &["--json", "model", "upload", TINY, &model, "--on", "m1"]));
    let made = &run["result"]["models"][0];
    assert_eq!(made["resolved"], TINY, "{run}");
    assert_eq!(made["published"]["destination"], format!("{model}@main"), "{run}");
    let source = made["published"]["tree"].as_str().unwrap().to_owned();
    let header = made["published"]["checkpoint"].as_str().unwrap().to_owned();

    // Another machine reads the private checkpoint by its header's digest, under the run's owner bearer.
    let _other = machine(&home, "m2");
    let checkpoint = format!("{model}#{header}");
    let run = completed(&home, &cozy(&home, None, &["--json", "model", "download", &checkpoint, "--on", "m2"]));
    assert_eq!(run["result"]["models"][0]["checkpoint"], header.as_str(), "{run}");

    // A release names it; info shows the lane and its checkpoint.
    let release = json(&home, &["--json", "model", "publish", &model, "--release", "1.0.0", "--commit", &source]);
    assert_eq!((release["commit"].as_str(), &release["changed"]), (Some(source.as_str()), &Value::Bool(true)), "{release}");
    let info = json(&home, &["--json", "model", "info", &model]);
    assert_eq!(info["releases"][0]["lanes"][0]["commit"], format!("{model}#{source}"), "{info}");

    json(&home, &["--json", "model", "family", &model, "cycle-test"]);
    let found = json(&home, &["--json", "model", "search", "--family", "cycle-test"]);
    assert_eq!(found["models"][0]["model"], model.as_str(), "{found}");

    // The package's quantizer runs on the machine and uploads the fp8 lane to the model.
    publish_quantizer(&home);
    let package = format!("{account}/tiny-quantizer");
    // The release's only lane: the import's unnamed variant.
    let pinned = format!("{model}@1.0.0");
    let run = completed(&home, &cozy(&home, None, &["--json", "model", "quantize", &pinned, "--fp8", "--package", &package, "--on", "m1"]));
    println!("quantize run: {run}");
    let written = run["output"].as_array().into_iter().flatten().find(|item| item["name"] == "fp8");
    let fp8 = written.and_then(|item| item["checkpoint"]["revision"].as_str()).unwrap_or_else(|| panic!("no tree in {run}")).to_owned();

    // A person sees the checkpoint the machine wrote and how to publish it.
    let human = cozy(&home, None, &["model", "quantize", &pinned, "--fp8", "--package", &package, "--on", "m1"]);
    assert_eq!(human.code, 0, "{human:?}");
    assert!(human.stdout.contains(&format!("{model}@{fp8}")), "{human:?}");
    // The release it was made from moves to the tag the variant went onto.
    let next = format!("cozy model publish {model} --release 1.0.0 --commit main");
    assert!(human.stdout.contains(&next), "{human:?}");

    // The machine holds the package now: quantize finds its quantizer there, and none for int8.
    let none = cozy(&home, None, &["--json", "model", "quantize", &pinned, "--int8", "--on", "m1"]);
    assert_eq!((none.code, none.refusal()["code"].as_str()), (1, Some("model_quantizer_not_found")), "{none:?}");

    // The release moves to the commit that adds the fp8 variant.
    let moved = json(&home, &["--json", "model", "publish", &model, "--release", "1.0.0", "--commit", &fp8]);
    assert_eq!(moved["commit"].as_str(), Some(fp8.as_str()), "{moved}");

    let yanked = json(&home, &["--json", "model", "yank", &model, "--release", "1.0.0"]);
    assert_eq!(yanked["yanked"], true, "{yanked}");
    let deleted = json(&home, &["--json", "model", "delete", &model, "--yes"]);
    assert_eq!(deleted["model"], model.as_str(), "{deleted}");
}

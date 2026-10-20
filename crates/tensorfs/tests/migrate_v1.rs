//! xs-053 A11, v1 → D10 directly (owner, 2026-10-20: no public 0.12). Six real v1 lanes cut to a
//! few tensors each (`fixtures/v1`): SDXL fp8 and int8 from 8819, H3 nvfp4 from tensorhub.com,
//! and three small ones whole. A kept tensor keeps its
//! key, dtype, spec, roles and bytes; a cut one its first rows in every part. The fixture holds
//! the objects of some tensors; `lanes.json` names every tensor's D10 name and scheme and, for
//! those held, each part's sha256. Each lane migrates through `d10::migrate` against a directory
//! endpoint, cites no new chunk but the pack, and opens through the public API.
#![cfg(feature = "store")]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{json, Value as Json};
use tensorfs::fetch::{Client as FetchClient, Endpoint, SourcePolicy};
use tensorfs::format::{Body, Config, ObjectRef, Repo, Value};
use tensorfs::import::Access;
use tensorfs::migrate::d10::{self, Migrator, UNKNOWN, UPSTREAMS};
use tensorfs::store::Store;
use tensorfs::{Client, Download, Open, Read, Selection};

#[derive(Deserialize)]
struct Lane {
    name: String,
    manifest: ObjectRef,
    /// The v1 objects it cites, each distinct run once, and the pack when it has inline parts.
    chunks: usize,
    configs: BTreeMap<String, Json>,
    /// By D10 `component/key`.
    tensors: BTreeMap<String, Want>,
}

#[derive(Deserialize)]
struct Want {
    scheme: Option<String>,
    /// Each part's sha256 by D10 role (`data` for a plain tensor), when the fixture holds it.
    read: Option<BTreeMap<String, String>>,
}

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn sha256(bytes: &[u8]) -> String {
    ObjectRef::of(bytes).sha256
}

#[test]
fn real_v1_lanes_become_d10_citing_their_bytes_where_they_are() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v1");
    let lanes: Vec<Lane> = serde_json::from_slice(&fs::read(fixture.join("lanes.json")).unwrap()).unwrap();
    let dir = Scratch(Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("migrate-v1-{}", std::process::id())));
    let _ = fs::remove_dir_all(&dir.0);
    // The endpoint: a store holding every fixture object, and each lane as an unreleased
    // checkpoint, as the Hub shows a v1 model: a private tag whose commit is the v1 manifest.
    let root = dir.0.join("endpoint");
    Store::open(&root).unwrap();
    for entry in fs::read_dir(&fixture).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        if name.len() == 64 {
            let at = root.join(ObjectRef { sha256: name, length: 0 }.key());
            fs::create_dir_all(at.parent().unwrap()).unwrap();
            fs::copy(&path, at).unwrap();
        }
    }
    for lane in &lanes {
        let at = root.join(format!("refs/{}.json", lane.name));
        fs::create_dir_all(at.parent().unwrap()).unwrap();
        let tag = format!("v1-{}", &lane.manifest.sha256[..12]);
        let doc = json!({"format": "cozytensors.repo/1", "name": lane.name, "tags": [{"name": tag, "commit": lane.manifest}]});
        fs::write(at, serde_json::to_vec(&doc).unwrap()).unwrap();
    }

    let blobs = |root: &Path| -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        for a in fs::read_dir(root.join("blobs")).unwrap() {
            for b in fs::read_dir(a.unwrap().path()).unwrap() {
                for o in fs::read_dir(b.unwrap().path()).unwrap() {
                    out.insert(o.unwrap().file_name().to_string_lossy().to_string());
                }
            }
        }
        out
    };
    let held = blobs(&root);
    let fetch = FetchClient::new(SourcePolicy::default());
    let endpoint = Arc::new(Endpoint::new(&fetch, &root.to_string_lossy(), None).unwrap());
    let mut migrator = Migrator::new(Access::from_env(), UPSTREAMS);
    let mut trees = BTreeMap::new();
    for lane in &lanes {
        let scratch = Store::open(dir.0.join("scratch").join(lane.name.replace('/', "__"))).unwrap();
        let done = d10::migrate(&endpoint, &fetch, &lane.name, &scratch, &mut migrator).unwrap();
        println!("{:#}", done.report);
        let report = &done.report["checkpoints"][0];
        assert_eq!(report["old"], lane.manifest.id().as_str());
        let repo = Repo::from_json(&fs::read(root.join(format!("refs/{}.json", lane.name))).unwrap()).unwrap();
        assert_eq!(repo.tags.len(), 1);
        let tag = &repo.tags[0];
        assert_eq!((tag.name.clone(), tag.public), (format!("v1-{}", &lane.manifest.sha256[..12]), false));
        trees.insert(lane.name.clone(), (tag.commit.clone(), report.clone()));
    }

    let uploaded: BTreeSet<String> = blobs(&root).difference(&held).cloned().collect();
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let client = Client::new(tensorfs::Config { cache_dir: dir.0.join("cache"), ..tensorfs::Config::from_env() }).unwrap();
    for lane in &lanes {
        let (tree, report) = &trees[&lane.name];
        let open = Open { revision: Some(tree.id()), endpoint: Some(root.to_string_lossy().into()), ..Open::default() };
        let ck = rt.block_on(client.open(&lane.name, &open)).unwrap();

        // The v1 objects where they are, and one new chunk at most: the pack of the inline parts
        // (D10 cites only 4 KiB-aligned ranges).
        let v1 = &report["lane"]["from_v1"]["v1"];
        assert_eq!(ck.chunks().len(), lane.chunks, "{}", lane.name);
        let new: Vec<String> = ck.chunks().iter().filter(|c| uploaded.contains(&c.sha256)).map(ObjectRef::id).collect();
        let pack: Vec<String> = v1["pack"].as_str().map(str::to_owned).into_iter().collect();
        assert_eq!(new, pack, "{}", lane.name);
        assert_eq!(v1["inline_parts"].as_u64().unwrap() > 0, !pack.is_empty(), "{}", lane.name);

        // Every tensor at its D10 name (U12's renames), each v1 spec as its D10 scheme.
        let schemes: BTreeMap<String, Option<String>> = (ck.components().iter())
            .flat_map(|c| c.tensors().iter().map(move |t| (c.name().to_owned(), t)))
            .map(|(c, t)| {
                let scheme = match &t.body {
                    Body::Encoded { encoding, .. } => Some(encoding.scheme.clone()),
                    Body::Data(_) => None,
                };
                (format!("{c}/{}", t.key), scheme)
            })
            .collect();
        let want: BTreeMap<String, Option<String>> = lane.tensors.iter().map(|(k, w)| (k.clone(), w.scheme.clone())).collect();
        assert_eq!(schemes, want, "{}", lane.name);

        // Each held tensor reads back its v1 parts byte for byte, under D10's roles.
        for (name, w) in &lane.tensors {
            let Some(want) = &w.read else { continue };
            let (component, key) = name.split_once('/').unwrap();
            let selection = Selection::Tensors { component: component.into(), keys: vec![key.into()] };
            rt.block_on(ck.download(&Download { selection, priority: 0, progress: None })).unwrap();
            let read: BTreeMap<String, String> = match ck.component(component).unwrap().read(key).unwrap() {
                Read::Plain(bytes) => BTreeMap::from([("data".to_owned(), sha256(&bytes))]),
                Read::Encoded { parts, .. } => parts.iter().map(|(role, bytes)| (role.clone(), sha256(bytes))).collect(),
            };
            assert_eq!(&read, want, "{}: {name}", lane.name);
        }

        // Configs: each v1 config is its component's, H3's one `model` config split per component.
        let renamed = &report["lane"]["from_v1"]["renames"]["components"];
        let config = |name: &str| ck.component(name).map(|c| c.config().clone()).ok();
        let as_config = |v: &Json| serde_json::from_value::<Config>(v.clone()).unwrap();
        for (name, v1) in &lane.configs {
            let d10 = |c: &str| renamed[c].as_str().unwrap_or(c).to_owned();
            match name.as_str() {
                "model" if ck.component("model").is_err() && v1.as_object().is_some_and(|m| m.keys().all(|c| ck.component(&d10(c)).is_ok())) => {
                    for (c, per) in v1.as_object().unwrap() {
                        assert_eq!(config(&d10(c)), Some(as_config(per)), "{}: {c}", lane.name);
                    }
                }
                "model" if ck.components().len() == 1 => assert_eq!(Some(ck.components()[0].config().clone()), Some(as_config(v1)), "{}", lane.name),
                "model" => assert_eq!(config("model"), Some(as_config(v1)), "{}", lane.name),
                c if ck.component(&d10(c)).is_ok() => assert_eq!(config(&d10(c)), Some(as_config(v1)), "{}: {c}", lane.name),
                _ => {}
            }
        }

        // Sources: the table's row where one names the model, else `unknown`, never filled.
        let source = ck.metadata().get("source").cloned();
        if lane.name == "paul/d1-gate5-sdxl-b" {
            assert_eq!(source, Some(Value::Str(UPSTREAMS[0].from.into())));
            assert_eq!(ck.pipeline().unwrap().class, "diffusers.StableDiffusionXLPipeline");
            assert!(ck.component("tokenizer").is_ok() && ck.component("tokenizer_2").is_ok());
        } else {
            assert_eq!(source, Some(Value::Str(UNKNOWN.into())), "{}", lane.name);
            assert!(!ck.metadata().contains_key("filled"));
        }
    }

    // Again: nothing changes.
    let before: Vec<Vec<u8>> = lanes.iter().map(|l| fs::read(root.join(format!("refs/{}.json", l.name))).unwrap()).collect();
    for lane in &lanes {
        let scratch = Store::open(dir.0.join("again").join(lane.name.replace('/', "__"))).unwrap();
        let done = d10::migrate(&endpoint, &fetch, &lane.name, &scratch, &mut migrator).unwrap();
        assert!(done.report.get("sent").is_none(), "{}", lane.name);
    }
    let after: Vec<Vec<u8>> = lanes.iter().map(|l| fs::read(root.join(format!("refs/{}.json", l.name))).unwrap()).collect();
    assert_eq!(before, after);
}

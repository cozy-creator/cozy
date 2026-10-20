//! xs-053 A11, v1 → D10 directly (owner, 2026-10-20: no public 0.12). Six real v1 lanes copied
//! read-only from 8819's repo CAS (`fixtures/v1`: their manifests, headers, assets and the
//! objects of a few small tensors; `lanes.json` holds the sha256 of each selected part's v1
//! bytes). Each migrates through `d10::migrate` against a directory endpoint, cites no new chunk,
//! and opens through the public API: its components, configs and stored tensor bytes are v1's.
#![cfg(feature = "store")]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{json, Value as Json};
use tensorfs::fetch::{Client as FetchClient, Endpoint, SourcePolicy};
use tensorfs::format::{Config, ObjectRef, Repo, Value};
use tensorfs::import::Access;
use tensorfs::migrate::d10::{self, Migrator, UNKNOWN, UPSTREAMS};
use tensorfs::store::Store;
use tensorfs::{Client, Download, Open, Read, Selection};

#[derive(Deserialize)]
struct Lane {
    name: String,
    manifest: ObjectRef,
    parts: BTreeSet<String>,
    tensors: Vec<String>,
    configs: BTreeMap<String, Json>,
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
    // The endpoint: a store holding every fixture object, and each lane as an unreleased checkpoint.
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
        let doc = json!({"format": "tensorfs.repo/1", "name": lane.name, "releases": [], "checkpoints": [lane.manifest]});
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
        assert_eq!(repo.checkpoints.len(), 1);
        trees.insert(lane.name.clone(), (repo.checkpoints[0].clone(), report.clone()));
    }

    let uploaded: BTreeSet<String> = blobs(&root).difference(&held).cloned().collect();
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let client = Client::new(tensorfs::Config { cache_dir: dir.0.join("cache"), ..tensorfs::Config::from_env() }).unwrap();
    for lane in &lanes {
        let (tree, report) = &trees[&lane.name];
        let open = Open { revision: Some(tree.id()), endpoint: Some(root.to_string_lossy().into()), ..Open::default() };
        let ck = rt.block_on(client.open(&lane.name, &open)).unwrap();

        // One new chunk at most: the pack of the inline parts (D10 cites only 4 KiB-aligned
        // ranges). Every other chunk is a v1 object, cited where it is.
        let v1 = &report["lane"]["from_v1"]["v1"];
        assert_eq!(ck.chunks().len() as u64, v1["chunks"].as_u64().unwrap());
        let new: Vec<String> = ck.chunks().iter().filter(|c| uploaded.contains(&c.sha256)).map(ObjectRef::id).collect();
        let pack: Vec<String> = v1["pack"].as_str().map(str::to_owned).into_iter().collect();
        assert_eq!(new, pack, "{}", lane.name);
        assert_eq!(v1["inline_parts"].as_u64().unwrap() > 0, !pack.is_empty(), "{}", lane.name);

        // Every tensor whose chunks the endpoint holds (the fixture's objects and the pack) reads
        // back exactly the selected v1 bytes.
        let mut found = BTreeSet::new();
        for component in ck.components() {
            for tensor in component.tensors() {
                let here = tensor.locations().iter().all(|(_, at)| {
                    ck.header().spans(at).iter().all(|s| root.join(ck.chunks()[s.chunk].key()).exists())
                });
                if !here || tensor.locations().iter().all(|(_, at)| at.length == 0) {
                    if lane.tensors.contains(&format!("{}/{}", component.name(), tensor.key)) {
                        eprintln!("{}: {}/{} not here: {:?}", lane.name, component.name(), tensor.key, tensor.locations());
                    }
                    continue;
                }
                let want = Selection::Tensors { component: component.name().into(), keys: vec![tensor.key.clone()] };
                rt.block_on(ck.download(&Download { selection: want, priority: 0, progress: None })).unwrap();
                let parts: Vec<Vec<u8>> = match component.read(&tensor.key).unwrap() {
                    Read::Plain(bytes) => vec![bytes],
                    Read::Encoded { parts, .. } => parts.into_values().collect(),
                };
                for bytes in parts {
                    let digest = sha256(&bytes);
                    if lane.tensors.contains(&format!("{}/{}", component.name(), tensor.key)) && !lane.parts.contains(&digest) {
                        eprintln!("{}: {}/{} read {} ({} bytes), not a selected v1 part", lane.name, component.name(), tensor.key, digest, bytes.len());
                    }
                    if lane.parts.contains(&digest) {
                        found.insert(digest);
                    }
                }
            }
        }
        assert_eq!(found, lane.parts, "{}: every selected v1 part reads back byte-exact", lane.name);

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
            assert!(ck.metadata().get("filled").is_none());
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

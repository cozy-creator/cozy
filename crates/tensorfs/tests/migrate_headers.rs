//! xs-053 A11: a store TensorFS 0.12 wrote (`fixtures/store-0.12`: tiny SDXL and tiny GPT-2
//! imported by the 0.12 wheel at c6410cb2c, their chunks left out since the migration never
//! reads one) moves to D10 in place. Every lane keeps its chunks and tensors byte for byte, its
//! configs equal parsed, its other files byte-exact as assets or repo files; run data a lane
//! lacked comes from the pinned upstream (live Hugging Face) and is recorded.
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use minicbor::Decoder;
use serde_json::Value as Json;
use tensorfs::import::Access;
use tensorfs::format::{Config, Header, ObjectRef, Repo, Tree, Value};
use tensorfs::migrate::d10::{self, Migrator, Upstream};
use tensorfs::store::Store;

const SDXL: &str = "hf://hf-internal-testing/tiny-sdxl-pipe@20594cbc343cfcfe447af5c87cdaf6c436b453f2";
const GPT2: &str = "hf://hf-internal-testing/tiny-random-gpt2@71034c5d8bde858ff824298bdedc65515b97d2b9";
const TABLE: &[Upstream] = &[
    Upstream { models: &["acme/tiny"], from: SDXL, files: &["model_index.json", "tokenizer/", "tokenizer_2/", "README"], source: None, note: "test" },
    Upstream { models: &["acme/gpt2"], from: GPT2, files: &["special_tokens_map.json", "README"], source: None, note: "test" },
];
/// The fixture's manifests: tiny-sdxl-pipe whole, without its run data and README (`ft`), the
/// same with another README, tiny-random-gpt2 whole and without special_tokens_map.json.
const FULL: &str = "90bac0063e68b5ac3726751c2bd5f04f616032cdf78fcdd6e31df68aed59d664";
const FT: &str = "bc44adc5638e46c5aec21b1a264790429082065cda1c159557e19df4969c5403";
const GPT2_FULL: &str = "8e06be82f22744ed935767b71e82ac8ea8aeb030f6fd3952eee3a0d3e0c2c0cf";

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

fn blob(root: &Path, sha256: &str) -> Vec<u8> {
    std::fs::read(root.join("blobs").join(&sha256[..2]).join(&sha256[2..4]).join(sha256)).unwrap()
}

/// An old manifest's files, its header apart.
fn manifest(root: &Path, sha256: &str) -> (BTreeMap<String, ObjectRef>, Vec<u8>) {
    let doc: Json = serde_json::from_slice(&blob(root, sha256)).unwrap();
    let mut files: BTreeMap<String, ObjectRef> = serde_json::from_value(doc["files"].clone()).unwrap();
    let header = files.remove("model.cozytensors").unwrap();
    (files, blob(root, &header.sha256))
}

/// Byte spans of a CBOR array's items.
fn items<'a>(bytes: &'a [u8], d: &mut Decoder<'a>) -> Vec<&'a [u8]> {
    let n = d.array().unwrap().unwrap();
    (0..n)
        .map(|_| {
            let at = d.position();
            d.skip().unwrap();
            &bytes[at..d.position()]
        })
        .collect()
}

/// A header's encodings, and each component's name and tensors, as bytes.
type Raw = (Vec<u8>, Vec<(Vec<u8>, Vec<u8>)>);

fn raw(header: &[u8], root_len: usize, (encodings, components, tensors): (usize, usize, usize)) -> Raw {
    let mut d = Decoder::new(header);
    let root = items(header, &mut d);
    assert_eq!(root.len(), root_len);
    let mut d = Decoder::new(root[components]);
    let comps = (0..d.array().unwrap().unwrap())
        .map(|_| {
            let at = d.position();
            d.skip().unwrap();
            let one = &root[components][at..d.position()];
            let fields = items(one, &mut Decoder::new(one));
            (fields[0].to_vec(), fields[tensors].to_vec())
        })
        .collect();
    (root[encodings].to_vec(), comps)
}

fn parse(bytes: &[u8]) -> Config {
    serde_json::from_slice(bytes).unwrap()
}

/// `lane` (old manifest `sha256` in `old`) against its D10 header and tree in `store`.
fn same(old: &Path, store: &Store, sha256: &str, header: &ObjectRef, tree: &Tree) -> Header {
    let (files, old_header) = manifest(old, sha256);
    let bytes = store.read(header).unwrap();
    let new = Header::from_cbor(&bytes).unwrap();
    // The chunks, encodings and every component's tensors, byte for byte.
    let mut d = Decoder::new(&old_header);
    let root = items(&old_header, &mut d);
    let chunks: Vec<&[u8]> = items(root[1], &mut Decoder::new(root[1]));
    let mut d = Decoder::new(&bytes);
    let new_root = items(&bytes, &mut d);
    assert_eq!(chunks, items(new_root[1], &mut Decoder::new(new_root[1])), "the chunks moved");
    let (enc, comps) = raw(&old_header, 5, (2, 3, 1));
    let (new_enc, new_comps) = raw(&bytes, 6, (2, 4, 3));
    assert_eq!(enc, new_enc);
    assert_eq!(comps.as_slice(), &new_comps[..comps.len()], "a tensor changed");
    // Each file: a config equal parsed, else an asset or a repo file, byte-exact.
    let root_component = new.component("model").is_some() && !files.keys().any(|p| p.starts_with("model/"));
    for (path, object) in &files {
        let old_bytes = blob(old, &object.sha256);
        if path == "model_index.json" {
            assert_eq!(new.pipeline().unwrap().config, parse(&old_bytes), "{path}");
            continue;
        }
        let (name, rel) = match path.split_once('/') {
            Some((c, rel)) if new.component(c).is_some() => (c, rel),
            _ if root_component && !path.contains('/') => ("model", path.as_str()),
            _ => {
                assert_eq!(tree.files.get(path), Some(object), "{path}: not a repo file");
                continue;
            }
        };
        let c = new.component(name).unwrap();
        if rel == c.config_name() {
            assert_eq!(c.config, parse(&old_bytes), "{path}");
        } else if let Some(asset) = c.assets.get(rel) {
            assert_eq!(asset, object, "{path}");
            assert_eq!(store.read(asset).unwrap(), old_bytes, "{path}");
        } else {
            assert_eq!(tree.files.get(path), Some(object), "{path}: not a repo file");
        }
    }
    new
}

fn filled(h: &Header) -> Option<(String, BTreeSet<String>)> {
    let Value::Map(f) = h.metadata.get("filled")? else { panic!("filled is a map") };
    let files = match &f["files"] {
        Value::List(l) => l.iter().map(|v| v.as_str().unwrap().to_owned()).collect(),
        other => panic!("{other:?}"),
    };
    Some((f["from"].as_str().unwrap().to_owned(), files))
}

#[test]
fn a_store_from_0_12_moves_to_d10_in_place() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/store-0.12");
    let dir: PathBuf = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("migrate-headers-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let root = dir.join("store");
    copy(&fixture, &root);
    let mut migrator = Migrator::new(Access::from_env(), TABLE);
    let report = d10::store(&root, &mut migrator).unwrap();
    println!("{report:#}");

    let store = Store::open(&root).unwrap();
    let repos: BTreeMap<String, Repo> = store.refs().unwrap().into_iter().map(|r| (r.name.clone(), r)).collect();
    assert_eq!(repos.keys().collect::<Vec<_>>(), ["acme/gpt2", "acme/tiny", "local/tiny-gpt2", "local/tiny-sdxl"]);
    let variants = |tree: &ObjectRef| {
        let tree = store.tree(tree).unwrap();
        let headers: BTreeMap<String, ObjectRef> = tree.variants().map(|(v, h)| (v.to_owned(), h.clone())).collect();
        (tree, headers)
    };

    // A release of lanes is one tree at its next revision, yanked or not.
    let tiny = &repos["acme/tiny"];
    let releases: Vec<(&str, bool, u64)> = tiny.releases.iter().map(|r| (r.version.as_str(), r.yanked, r.revision)).collect();
    assert_eq!(releases, [("0.9.0", true, 3), ("1.0.0", false, 4), ("1.1.0", false, 2)]);
    assert!(tiny.checkpoints.is_empty(), "every old checkpoint is in a release");
    let (tree, headers) = variants(&tiny.releases[1].tree);
    assert_eq!(headers.keys().collect::<Vec<_>>(), ["ft", "full"]);
    let full = same(&fixture, &store, FULL, &headers["full"], &tree);
    let ft = same(&fixture, &store, FT, &headers["ft"], &tree);
    assert_eq!(full.metadata.get("source"), Some(&Value::Str(SDXL.into())));
    assert_eq!(filled(&full), None, "the whole pipe lacks nothing");
    // ft lacked its pipeline, tokenizers and README: the upstream's, recorded.
    let (from, files) = filled(&ft).unwrap();
    assert_eq!(from, SDXL);
    let (full_files, _) = manifest(&fixture, FULL);
    let expect: BTreeSet<String> = (full_files.keys())
        .filter(|p| p.starts_with("tokenizer") || *p == "model_index.json" || *p == "README.md")
        .cloned()
        .collect();
    assert_eq!(files, expect);
    assert_eq!(ft.pipeline(), full.pipeline());
    assert_eq!(ft.pipeline().unwrap().class, "diffusers.StableDiffusionXLPipeline");
    for name in ["tokenizer", "tokenizer_2", "scheduler", "unet"] {
        assert_eq!(ft.component(name), full.component(name), "{name}");
    }
    assert_eq!(tree.files["README.md"], full_files["README.md"]);
    assert!(!full.metadata.contains_key("sources") && !full.metadata.contains_key("normalized"));
    assert!(full.metadata.contains_key("safetensors_metadata"));

    // 1.1.0's lanes disagree on README.md: the first lane's stays, and the report says so.
    let (tree, _) = variants(&tiny.releases[2].tree);
    assert_eq!(tree.files["README.md"], full_files["README.md"]);
    let conflicts = report["repos"].as_array().unwrap().iter().find(|r| r["model"] == "acme/tiny").unwrap()["releases"][2]["conflicts"].clone();
    assert_eq!(conflicts[0]["path"], "README.md");
    assert_eq!(conflicts[0]["dropped"], "b");

    // An unreleased checkpoint is a tree of its own; the root component takes the filled file.
    let gpt2 = &repos["acme/gpt2"];
    assert!(gpt2.releases.is_empty());
    let (tree, headers) = variants(&gpt2.checkpoints[0]);
    assert_eq!(headers.keys().collect::<Vec<_>>(), [""]);
    let (gpt2_files, _) = manifest(&fixture, GPT2_FULL);
    let h = Header::from_cbor(&store.read(&headers[""]).unwrap()).unwrap();
    assert_eq!(h.component("model").unwrap().assets["special_tokens_map.json"], gpt2_files["special_tokens_map.json"]);
    assert_eq!(filled(&h).unwrap().1, BTreeSet::from(["special_tokens_map.json".to_owned()]));
    assert!(tree.repo_files().next().is_none());

    // local/ models: their own lanes, no upstream, the old pinned source kept.
    let local = &repos["local/tiny-sdxl"];
    assert_eq!((local.releases[0].version.as_str(), local.releases[0].revision), ("1", 1));
    let (tree, headers) = variants(&local.releases[0].tree);
    let h = same(&fixture, &store, FULL, &headers["f32"], &tree);
    assert_eq!(h.metadata.get("source"), Some(&Value::Str(SDXL.into())));
    assert_eq!(filled(&h), None);
    let local = &repos["local/tiny-gpt2"];
    let (tree, headers) = variants(&local.releases[0].tree);
    let h = same(&fixture, &store, GPT2_FULL, &headers["f32"], &tree);
    assert_eq!(h.component("model").unwrap().config["model_type"], "gpt2");

    // Again: nothing changes.
    let refs = |root: &Path| -> Vec<Vec<u8>> {
        ["acme/gpt2", "acme/tiny", "local/tiny-gpt2", "local/tiny-sdxl"].map(|n| std::fs::read(root.join(format!("refs/{n}.json"))).unwrap()).to_vec()
    };
    let before = refs(&root);
    d10::store(&root, &mut migrator).unwrap();
    assert_eq!(refs(&root), before);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn partial_tokenizer_keeps_existing_files_and_fills_missing_files() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/store-0.12");
    let unique = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("migrate-partial-{}-{unique}", std::process::id()));
    copy(&fixture, &dir);
    let mut manifest: Json = serde_json::from_slice(&blob(&fixture, FULL)).unwrap();
    let missing = "tokenizer/vocab.json";
    let expected: ObjectRef = serde_json::from_value(manifest["files"].as_object_mut().unwrap().remove(missing).unwrap()).unwrap();
    assert!(manifest["files"].get("tokenizer/tokenizer_config.json").is_some());
    let bytes = serde_json::to_vec(&manifest).unwrap();
    let object = ObjectRef::of(&bytes);
    let path = dir.join(object.key());
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, bytes).unwrap();
    let doc = serde_json::to_vec(&serde_json::json!({
        "name": "acme/tiny", "releases": [{"version": "1", "lanes": {"full": object}}]
    })).unwrap();
    let mut migrator = Migrator::new(Access::from_env(), TABLE);
    let migrated = migrator.repo(&doc, &|o| Ok(std::fs::read(dir.join(o.key()))?)).unwrap();
    let read = |o: &ObjectRef| migrated.objects.iter().find(|(at, _)| at == o).unwrap().1.clone();
    let tree = Tree::from_json(&read(&migrated.repo.releases[0].tree)).unwrap();
    let header = Header::from_cbor(&read(tree.variant(Some("full")).unwrap().1)).unwrap();
    assert_eq!(header.component("tokenizer").unwrap().assets["vocab.json"], expected);
    assert_eq!(filled(&header).unwrap().1, BTreeSet::from([missing.to_owned()]));
    std::fs::remove_dir_all(dir).unwrap();
}

//! Migration-only TEXT-class D10 metadata. Chunks/assets deliberately are not available to
//! the planner: every original byte reference must survive without a payload read.
#![cfg(feature = "store")]

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use minicbor::Decoder;
use serde_json::{json, Value as Json};
use tensorfs::fetch::{Client, Endpoint, SourcePolicy};
use tensorfs::format::{Body, ClassRef, Component, Config, Dtype, Encoding, Header, Location,
                       ObjectRef, Part, Pipeline, Tree, Value};
use tensorfs::import::Access;
use tensorfs::migrate::d10::{self, Migrator};
use tensorfs::store::Store;

type Objects = HashMap<ObjectRef, Vec<u8>>;

fn config(value: Json) -> Config { serde_json::from_value(value).unwrap() }

fn header() -> Header {
    let chunks = vec![ObjectRef::of(&[7; 8192]), ObjectRef::of(&[1; 100])];
    Header::new(chunks, Some(Pipeline {
        class: ClassRef::new("diffusers", "0.35.1", "StableDiffusionXLPipeline"),
        config: config(json!({"_class_name": "StableDiffusionXLPipeline", "_diffusers_version": "0.35.1",
            "unet": ["diffusers", "UNet2DConditionModel"], "tokenizer": ["transformers", "CLIPTokenizer"]})),
    }), vec![Component {
        name: "unet".into(), class: ClassRef::new("diffusers", "0.35.1", "UNet2DConditionModel"),
        config: config(json!({"_class_name": "UNet2DConditionModel", "norm_eps": 1e-5, "upcast": 1.0})),
        tensors: vec![Tensor::plain(), Tensor::encoded()], ..Default::default()
    }, Component {
        name: "tokenizer".into(), class: ClassRef::new("transformers", "4.57.0", "CLIPTokenizer"),
        config: config(json!({"transformers_version": "4.57.0", "model_max_length": 77})),
        assets: BTreeMap::from([("vocab.json".into(), ObjectRef::of(b"original vocab"))]),
        config_file: Some("tokenizer_config.json".into()), ..Default::default()
    }, Component { name: "unknown".into(), ..Default::default() }],
    BTreeMap::from([("source".into(), Value::Str("local".into())), ("retained".into(), Value::Int(17))])).unwrap()
}

struct Tensor;
impl Tensor {
    fn plain() -> tensorfs::format::Tensor {
        tensorfs::format::Tensor { key: "plain".into(), dtype: Dtype::BF16, shape: vec![2, 3],
            body: Body::Data(Location { chunk: 0, offset: 0, length: 12 }) }
    }
    fn encoded() -> tensorfs::format::Tensor {
        tensorfs::format::Tensor { key: "encoded".into(), dtype: Dtype::F32, shape: vec![4, 1024],
            body: Body::Encoded { encoding: Encoding::fp8_rowwise(0), parts: vec![
                Part { role: "weight".into(), dtype: Dtype::F8_E4M3, shape: vec![4, 1024], chunk: 0, offset: 4096, length: 4096 },
                Part { role: "scale".into(), dtype: Dtype::F32, shape: vec![4], chunk: 1, offset: 0, length: 16 },
            ] } }
    }
}

fn fields(bytes: &[u8]) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut d = Decoder::new(bytes);
    let n = d.array().unwrap().unwrap();
    let head = bytes[..d.position()].to_vec();
    let items = (0..n).map(|_| {
        let at = d.position(); d.skip().unwrap(); bytes[at..d.position()].to_vec()
    }).collect();
    assert_eq!(d.position(), bytes.len());
    (head, items)
}

fn joined(head: Vec<u8>, values: Vec<Vec<u8>>) -> Vec<u8> {
    head.into_iter().chain(values.into_iter().flatten()).collect()
}

/// Reproduce the pre-cut D10 TEXT class layout, or erase only class fields for a byte oracle.
fn class_fields(bytes: &[u8], erase: bool) -> Vec<u8> {
    let text = |bytes: &mut Vec<u8>| {
        if erase { *bytes = vec![0x80]; return; }
        let (_, values) = fields(bytes);
        let name = Decoder::new(&values[2]).str().unwrap().to_owned();
        let mut e = minicbor::Encoder::new(Vec::new());
        e.str(&name).unwrap(); *bytes = e.into_writer();
    };
    let (head, mut root) = fields(bytes);
    let (phead, mut pipeline) = fields(&root[3]);
    if !pipeline.is_empty() { text(&mut pipeline[0]); }
    root[3] = joined(phead, pipeline);
    let (chead, components) = fields(&root[4]);
    root[4] = joined(chead, components.into_iter().map(|c| {
        let (head, mut values) = fields(&c); text(&mut values[1]); joined(head, values)
    }).collect());
    joined(head, root)
}

fn put(objects: &mut Objects, bytes: Vec<u8>) -> ObjectRef {
    let at = ObjectRef::of(&bytes); objects.insert(at.clone(), bytes); at
}

fn tree(objects: &mut Objects, header: &Header, text: bool, note: &[u8]) -> ObjectRef {
    let bytes = header.to_cbor().unwrap();
    let at = put(objects, if text { class_fields(&bytes, false) } else { bytes });
    let mut tree = Tree { files: BTreeMap::from([("README.md".into(), ObjectRef::of(note)),
                                                ("LICENSE".into(), ObjectRef::of(b"source license"))]) };
    tree.set_variant("int8", at.clone()); tree.set_variant("bf16", at);
    put(objects, tree.to_json().unwrap())
}

#[test]
fn released_and_private_d10_trees_convert_only_class_fields_and_plan_idempotently() {
    let expected = header();
    let mut objects = Objects::new();
    let old = tree(&mut objects, &expected, true, b"original README");
    let original = objects.clone();
    let doc = serde_json::to_vec(&json!({"format": "tensorfs.repo/1", "name": "acme/d10",
        "releases": [{"version": "1.0.0", "tree": old, "yanked": true}],
        "checkpoints": [old], "tags": [{"name": "main", "commit": old}]})).unwrap();
    let mut migrator = Migrator::new(Access::default(), &[]);
    let planned = migrator.repo(&doc, &|at| Ok(objects[at].clone())).unwrap();
    assert_eq!(objects, original, "planning must not write source data");
    assert_eq!(planned.repo.tags.len(), 2, "a released tree is not another unnamed checkpoint");
    assert_eq!((planned.repo.tags[0].name.as_str(), planned.repo.tags[0].public, planned.repo.tags[0].yanked), ("1.0.0", true, true));
    assert_eq!((planned.repo.tags[1].name.as_str(), planned.repo.tags[1].public), ("main", false));
    assert_eq!(planned.repo.tags[0].commit, planned.repo.tags[1].commit);
    assert_eq!(planned.objects.len(), 2, "one shared header and one tree, no chunks/assets");
    for (at, bytes) in &planned.objects { objects.insert(at.clone(), bytes.clone()); }
    let before = Tree::from_json(&original[&old]).unwrap();
    let after = Tree::from_json(&objects[&planned.repo.tags[0].commit]).unwrap();
    assert_eq!(before.repo_files().collect::<Vec<_>>(), after.repo_files().collect::<Vec<_>>());
    for (variant, old_header) in before.variants() {
        let new_bytes = &objects[after.header(variant).unwrap()];
        assert!(Header::from_cbor(&original[old_header]).is_err(), "ordinary reader stays ARRAY3-only");
        let new = Header::from_cbor(new_bytes).unwrap();
        assert_eq!(new, expected, "class versions, tensors, encodings, locations and metadata");
        assert_eq!(class_fields(&original[old_header], true), class_fields(new_bytes, true), "every non-class CBOR byte");
    }
    let again = migrator.repo(&planned.repo.to_json().unwrap(), &|at| Ok(objects[at].clone())).unwrap();
    assert_eq!(again.repo, planned.repo);
    assert!(again.objects.is_empty(), "canonical headers/trees are reused unchanged");
}

#[test]
fn unnamed_private_tree_names_avoid_all_existing_names_and_preserve_flags() {
    let mut objects = Objects::new();
    let old = tree(&mut objects, &header(), true, b"private original");
    let other = tree(&mut objects, &header(), false, b"named original");
    let names = (12..=64).step_by(4).map(|width| format!("v2-{}", &old.sha256[..width])).collect::<Vec<_>>();
    let tags = names.iter().map(|name| json!({"name": name, "commit": other, "yanked": true})).collect::<Vec<_>>();
    let doc = serde_json::to_vec(&json!({"name": "acme/private", "checkpoints": [old], "tags": tags})).unwrap();
    let mut migrator = Migrator::new(Access::default(), &[]);
    let first = migrator.repo(&doc, &|at| Ok(objects[at].clone())).unwrap();
    let second = migrator.repo(&doc, &|at| Ok(objects[at].clone())).unwrap();
    assert_eq!(first.repo, second.repo);
    assert_eq!(first.repo.tags[0].name, format!("v2-{}-1", old.sha256));
    assert!(!first.repo.tags[0].public && !first.repo.tags[0].yanked);
    for (tag, name) in first.repo.tags[1..].iter().zip(names) {
        assert_eq!((&tag.name, &tag.commit, tag.public, tag.yanked), (&name, &other, false, true));
    }
}

#[test]
fn ambiguous_text_class_refuses_before_any_endpoint_write() {
    let header = Header::new(vec![], None, vec![Component { name: "model".into(),
        class: ClassRef::new("not-recorded", "", "Ambiguous"), ..Default::default() }], Default::default()).unwrap();
    let mut objects = Objects::new();
    let old = tree(&mut objects, &header, true, b"private");
    let dir = std::env::temp_dir().join(format!("d10-ambiguous-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let root = dir.join("endpoint"); Store::open(&root).unwrap();
    for (at, bytes) in &objects {
        let path = root.join(at.key()); std::fs::create_dir_all(path.parent().unwrap()).unwrap(); std::fs::write(path, bytes).unwrap();
    }
    let doc = serde_json::to_vec(&json!({"name": "acme/ambiguous", "checkpoints": [old]})).unwrap();
    std::fs::create_dir_all(root.join("refs/acme")).unwrap();
    std::fs::write(root.join("refs/acme/ambiguous.json"), &doc).unwrap();
    fn files(root: &std::path::Path) -> BTreeMap<std::path::PathBuf, Vec<u8>> {
        let mut found = BTreeMap::new();
        for entry in std::fs::read_dir(root).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                found.extend(files(&path));
            } else { found.insert(path.clone(), std::fs::read(path).unwrap()); }
        }
        found
    }
    let before = files(&root);
    let scratch = Store::open(dir.join("scratch")).unwrap();
    let client = Client::new(SourcePolicy::default());
    let endpoint = Arc::new(Endpoint::new(&client, &root.to_string_lossy(), None).unwrap());
    let error = d10::migrate(&endpoint, &client, "acme/ambiguous", &scratch, &mut Migrator::new(Access::default(), &[])).err().unwrap();
    assert_eq!(error.kind, tensorfs::failure::Kind::Unsupported);
    assert!(error.detail.contains("Ambiguous") && error.detail.contains("verified library"), "{error}");
    assert_eq!(std::fs::read(root.join("refs/acme/ambiguous.json")).unwrap(), doc);
    assert_eq!(files(&root), before, "no endpoint object or ref was written");
    let store = Store::open(&root).unwrap();
    for (at, bytes) in objects { assert_eq!(store.read(&at).unwrap(), bytes); }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn approved_family_and_qualified_names_work_without_inventing_versions() {
    for (model, class, metadata, want) in [
        ("acme/birefnet", "BiRefNet", json!({}), ClassRef::new("birefnet", "0.4.0", "BiRefNet")),
        ("acme/birefnet", "BiRefNet", json!({"architectures": ["BiRefNet"]}), ClassRef::new("transformers", "", "BiRefNet")),
        ("acme/configured", "PlainModel", json!({"architectures": ["PlainModel"]}), ClassRef::new("transformers", "", "PlainModel")),
        ("acme/qualified", "example.deep.Model", json!({"_class_name": "example.deep.Model", "_diffusers_version": "0.41.0"}), ClassRef::new("example", "", "deep.Model")),
        ("acme/versioned", "example.deep.Model", json!({"_example_version": "1.2"}), ClassRef::new("example", "1.2", "deep.Model")),
        ("acme/unknown", "", json!({}), ClassRef::default()),
    ] {
        let header = Header::new(vec![], None, vec![Component { name: "model".into(),
            class: if class.is_empty() { ClassRef::default() } else { ClassRef::new("source", "", class) },
            config: config(metadata), ..Default::default() }], Default::default()).unwrap();
        let mut objects = Objects::new(); let old = tree(&mut objects, &header, true, b"source");
        let doc = serde_json::to_vec(&json!({"name": model, "checkpoints": [old]})).unwrap();
        let planned = Migrator::new(Access::default(), &[]).repo(&doc, &|at| Ok(objects[at].clone())).unwrap();
        for (at, bytes) in planned.objects { objects.insert(at, bytes); }
        let tree = Tree::from_json(&objects[&planned.repo.tags[0].commit]).unwrap();
        let new = Header::from_cbor(&objects[tree.header("int8").unwrap()]).unwrap();
        assert_eq!(new.component("model").unwrap().class, want);
    }
}

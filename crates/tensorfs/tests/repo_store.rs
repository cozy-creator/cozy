use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use tensorfs::canon::Value;
use tensorfs::dtype::Dtype;
use tensorfs::header::{Asset, Closure, Header, Part, Tensor};
use tensorfs::ids::{Doc, ObjectRef};
use tensorfs::manifest::{Draft, Entry, Manifest};
use tensorfs::meta::Meta;
use tensorfs::read;
use tensorfs::repository::{Mutation, ReleaseLane, Repository};
use tensorfs::store::{Fault, Store, Verdict};

fn temporary(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tensorfs-cli-{name}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

#[test]
fn one_tmp_tree_is_recreatable_and_corruption_is_not_retained() {
    let root = temporary("tmp-layout");
    let store = Store::init(&root).unwrap();
    assert!(root.join("tmp").is_dir());

    let bytes = b"known-good";
    let object = ObjectRef::of(bytes);
    store
        .put_stream(&mut bytes.as_slice(), Some(&object), &Fault::default())
        .unwrap();
    let path = store.blob_path(&object.sha256);
    platform::durable::set_mode(&path, 0o644).unwrap();
    fs::write(&path, b"corruption").unwrap();
    assert!(matches!(
        store.verify(&object.sha256).unwrap(),
        Verdict::CorruptRemoved { .. }
    ));
    assert!(!path.exists());

    let _ = fs::remove_dir_all(root.join("tmp"));
    Store::open(&root).unwrap();
    assert!(root.join("tmp").is_dir());
    let _ = fs::remove_dir_all(root);
}

fn model_fixture(store: &Store, value: [u8; 4]) -> ObjectRef {
    let spec = tensorfs::registry::seeds()
        .into_iter()
        .find(|seed| seed.alias == "plain/1")
        .unwrap()
        .spec;
    let header = Header {
        configs: Vec::new(),
        assets: Vec::new(),
        encodings: vec![spec.clone()],
        components: vec![(
            "model".into(),
            vec![(
                "weight".into(),
                Tensor {
                    dtype: Dtype::F32,
                    shape: vec![1],
                    encoding: spec.object_id(),
                    parts: vec![("value".into(), Part::plan(Dtype::F32, vec![1], &value))],
                },
            )],
        )],
    };
    header.validate(&Closure::default()).unwrap();
    let bytes = header.canonical_bytes().unwrap();
    let header = store
        .put_stream(
            &mut bytes.as_slice(),
            Some(&ObjectRef::of(&bytes)),
            &Fault::default(),
        )
        .unwrap()
        .obj;
    store
        .put_manifest(
            &Draft {
                entries: vec![("model.cozytensors".into(), Entry::CozyTensors(header))],
            }
            .seal()
            .unwrap(),
        )
        .unwrap()
        .obj
}

#[test]
fn a_model_asset_reads_through_the_cozytensors_lease() {
    let root = temporary("model-asset");
    let store = Store::init(&root).unwrap();
    let meta = Meta::open(&store).unwrap();
    let parts: [&[u8]; 2] = [b"vocab\n", b"merges\n"];
    let segments = parts
        .iter()
        .map(|bytes| {
            let mut source = *bytes;
            store
                .put_stream(&mut source, Some(&ObjectRef::of(bytes)), &Fault::default())
                .unwrap()
                .obj
        })
        .collect::<Vec<_>>();
    let bytes = parts.concat();
    let spec = tensorfs::registry::seeds()
        .into_iter()
        .find(|seed| seed.alias == "plain/1")
        .unwrap()
        .spec;
    let header = Header {
        configs: Vec::new(),
        assets: vec![(
            "tokenizer/vocab.txt".into(),
            Asset {
                logical_sha256: tensorfs::sha256::hex_digest(&bytes),
                logical_length: bytes.len() as u64,
                media_type: "text/plain".into(),
                segments,
            },
        )],
        encodings: vec![spec.clone()],
        components: vec![(
            "model".into(),
            vec![(
                "weight".into(),
                Tensor {
                    dtype: Dtype::F32,
                    shape: vec![1],
                    encoding: spec.object_id(),
                    parts: vec![("value".into(), Part::plan(Dtype::F32, vec![1], &[0; 4]))],
                },
            )],
        )],
    };
    let header_bytes = header.canonical_bytes().unwrap();
    let header_ref = store
        .put_stream(
            &mut header_bytes.as_slice(),
            Some(&ObjectRef::of(&header_bytes)),
            &Fault::default(),
        )
        .unwrap()
        .obj;
    let manifest = Draft {
        entries: vec![("model.cozytensors".into(), Entry::CozyTensors(header_ref))],
    }
    .seal()
    .unwrap();
    store.put_manifest(&manifest).unwrap();
    let (lease, _) = read::acquire_cozytensors(&store, &meta, &manifest).unwrap();
    let asset = &header.assets[0].1;
    assert_eq!(
        read::read_asset(&lease, "tokenizer/vocab.txt", asset, bytes.len() as u64).unwrap(),
        bytes
    );
    assert_eq!(
        read::read_asset(&lease, "tokenizer/vocab.txt", asset, 1)
            .unwrap_err()
            .code,
        tensorfs::err::Code::SIZE_CAP
    );
    lease.release(&meta).unwrap();
    let _ = fs::remove_dir_all(root);
}

fn apply_release(
    store: &Store,
    observed: Option<&[u8]>,
    release: &Mutation,
) -> tensorfs::err::Result<Option<Repository>> {
    let Mutation::UpdateRelease { repo, set, .. } = release else {
        panic!("apply_release requires UpdateRelease")
    };
    let manifest = &set.first().expect("test release sets one lane").manifest;
    let checkpointed = store
        .apply_repository(
            observed,
            &Mutation::PutCheckpoint {
                repo: repo.clone(),
                manifest: manifest.clone(),
            },
            &Fault::default(),
        )?
        .expect("put_checkpoint retains the repository");
    store.apply_repository(
        Some(&checkpointed.canonical_bytes()),
        release,
        &Fault::default(),
    )
}

#[test]
fn local_alias_is_one_atomic_replaceable_checkpoint() {
    let root = temporary("local-alias");
    let store = Store::init(&root).unwrap();
    let repo = tensorfs::repository::RepositoryName::new("local", "h3-dev").unwrap();
    let manifest_a = model_fixture(&store, [0, 0, 0, 0]);
    let manifest_b = model_fixture(&store, [0, 0, 128, 63]);
    assert_ne!(manifest_a, manifest_b);
    let mutation_a = Mutation::ReplaceLocal {
        repo: repo.clone(),
        manifest: manifest_a.clone(),
        version: "11".repeat(32),
    };
    let alias_a = store
        .apply_repository(None, &mutation_a, &Fault::default())
        .unwrap()
        .unwrap();
    let bytes_a = alias_a.canonical_bytes();
    assert!(alias_a.releases.is_empty());
    assert_eq!(alias_a.checkpoints.len(), 1);
    assert_eq!(alias_a.checkpoints[0].manifest, manifest_a);

    let mutation_b = Mutation::ReplaceLocal {
        repo: repo.clone(),
        manifest: manifest_b.clone(),
        version: "22".repeat(32),
    };
    let alias_b = store
        .apply_repository(Some(&bytes_a), &mutation_b, &Fault::default())
        .unwrap()
        .unwrap();
    assert!(alias_b.releases.is_empty());
    assert_eq!(alias_b.checkpoints.len(), 1);
    assert_eq!(
        alias_b.checkpoints[0].source_selection,
        Some("22".repeat(32))
    );
    assert_eq!(alias_b.checkpoints[0].manifest, manifest_b);
    assert_eq!(
        store
            .apply_repository(Some(&bytes_a), &mutation_a, &Fault::default())
            .unwrap_err()
            .code,
        tensorfs::err::Code::REPOSITORY_CONFLICT
    );
    assert_eq!(
        store
            .apply_repository(
                Some(&alias_b.canonical_bytes()),
                &Mutation::UpdateRelease {
                    expected_revision: 0,
                    repo: repo.clone(),
                    remove: Vec::new(),
                    set: vec![ReleaseLane {
                        extra: Default::default(),
                        lane: "other".into(),
                        manifest: manifest_b.clone(),
                    }],
                    version: "1.0.0".into(),
                },
                &Fault::default(),
            )
            .unwrap_err()
            .code,
        tensorfs::err::Code::KEY_GRAMMAR
    );

    // Replacing with an observed body swaps the one checkpoint; the alias has no release and
    // is still the one row a consumer can find a local model by.
    let alias_c = store
        .apply_repository(
            Some(&alias_b.canonical_bytes()),
            &Mutation::ReplaceLocal {
                repo: repo.clone(),
                manifest: manifest_b.clone(),
                version: "33".repeat(32),
            },
            &Fault::default(),
        )
        .unwrap()
        .unwrap();
    let list = || -> String {
        tensorfs::storage::repository_release_lines(&root)
            .unwrap()
            .into_iter()
            .map(|line| String::from_utf8(line).unwrap() + "\n")
            .collect()
    };
    assert_eq!(
        list(),
        format!(
            "{{\"kind\":\"local\",\"manifest_length\":{},\"manifest_sha256\":\"{}\",\"name\":\"h3-dev\",\"org\":\"local\",\"source_selection\":\"sha256:{}\"}}\n",
            manifest_b.length,
            manifest_b.sha256,
            "33".repeat(32)
        )
    );
    store
        .apply_repository(
            Some(&alias_c.canonical_bytes()),
            &Mutation::DeleteRepository { repo: repo.clone() },
            &Fault::default(),
        )
        .unwrap();
    assert_eq!(list(), "");
    let _ = fs::remove_dir_all(root);
}

fn checkpoint_mutation(manifest: &ObjectRef, org: &str, name: &str) -> Vec<u8> {
    tensorfs::canon::write(&Value::obj(vec![
        ("action", Value::str("put_checkpoint")),
        ("manifest", manifest.to_value()),
        (
            "repo",
            Value::obj(vec![("name", Value::str(name)), ("org", Value::str(org))]),
        ),
    ]))
}

#[test]
fn manifest_build_is_the_single_ordinary_file_producer() {
    let root = temporary("manifest-build");
    fs::create_dir_all(&root).unwrap();
    let entries = root.join("entries.jsonl");
    let output = root.join("manifest.json");
    let empty = ObjectRef::of(b"");
    let source = ObjectRef::of(b"print('polo')\n");
    let rows = [
        tensorfs::canon::write(&Value::obj(vec![
            ("blob", empty.to_value()),
            ("path", Value::str("marco/__init__.py")),
        ])),
        tensorfs::canon::write(&Value::obj(vec![
            ("blob", source.clone().to_value()),
            ("path", Value::str("marco/package.py")),
        ])),
    ];
    let mut bytes = Vec::new();
    for row in rows {
        bytes.extend_from_slice(&row);
        bytes.push(b'\n');
    }
    fs::write(&entries, bytes).unwrap();

    let result = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args([
            "manifest",
            "build",
            entries.to_str().unwrap(),
            "--out",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );

    let expected = Manifest::from_files(vec![
        ("marco/__init__.py".into(), empty),
        ("marco/package.py".into(), source),
    ])
    .unwrap();
    let stored = fs::read(&output).unwrap();
    assert_eq!(stored, expected.canonical_bytes());
    assert_eq!(
        result.stdout,
        [
            tensorfs::canon::write(&ObjectRef::of(&stored).to_value()),
            b"\n".to_vec(),
        ]
        .concat()
    );

    fs::write(
        &entries,
        [
            tensorfs::canon::write(&Value::obj(vec![
                ("blob", ObjectRef::of(b"later").to_value()),
                ("path", Value::str("z.py")),
            ])),
            b"\n".to_vec(),
            tensorfs::canon::write(&Value::obj(vec![
                ("blob", ObjectRef::of(b"earlier").to_value()),
                ("path", Value::str("a.py")),
            ])),
        ]
        .concat(),
    )
    .unwrap();
    let refused = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args([
            "manifest",
            "build",
            entries.to_str().unwrap(),
            "--out",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!refused.status.success());
    assert!(String::from_utf8_lossy(&refused.stderr).contains("PATH_ORDER"));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn both_cas_domains_rebuild_their_projection_from_the_filesystem() {
    let root = temporary("cas-domain-rebuild");
    for domain in ["repo_cas", "dataset_cas"] {
        let store_root = root.join(domain);
        let store = Store::init(&store_root).unwrap();
        let payload = format!("{domain} durable bytes\n").into_bytes();
        let blob = store
            .put_stream(
                &mut payload.as_slice(),
                Some(&ObjectRef::of(&payload)),
                &Fault::default(),
            )
            .unwrap()
            .obj;
        let manifest = Manifest::from_files(vec![("content.bin".into(), blob)]).unwrap();
        let manifest_ref = store.put_manifest(&manifest).unwrap().obj;
        let repo = tensorfs::repository::RepositoryName::new("recovery", domain).unwrap();
        apply_release(
            &store,
            None,
            &Mutation::UpdateRelease {
                expected_revision: 0,
                repo: repo.clone(),
                remove: Vec::new(),
                set: vec![ReleaseLane {
                    extra: Default::default(),
                    lane: "main".into(),
                    manifest: manifest_ref,
                }],
                version: "1.0.0".into(),
            },
        )
        .unwrap();
        let repository_bytes = fs::read(store.repository_path(&repo)).unwrap();
        assert!(
            !String::from_utf8_lossy(&repository_bytes).contains("evidence"),
            "ordinary repositories must not fabricate model evidence"
        );
        let rows = store_root.join("rebuild.jsonl");
        let result = Command::new(env!("CARGO_BIN_EXE_tfs"))
            .args([
                "store",
                "rebuild",
                store_root.to_str().unwrap(),
                "--rows",
                rows.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{domain}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let rows = fs::read_to_string(rows).unwrap();
        assert!(rows.contains("\"kind\":\"repo\""));
        assert!(rows.contains("\"kind\":\"release\""));
        assert!(rows.contains("\"kind\":\"checkpoint_object\""));
    }
    let _ = fs::remove_dir_all(root);
}

#[test]
fn ordinary_dataset_repo_apply_needs_no_model_header_or_evidence() {
    let root = temporary("ordinary-repo-apply");
    fs::create_dir_all(&root).unwrap();
    let content = ObjectRef::of(b"dataset source");
    let manifest = Manifest::from_files(vec![("src/package.py".into(), content.clone())]).unwrap();
    let manifest_bytes = manifest.canonical_bytes();
    let manifest_ref = ObjectRef::of(&manifest_bytes);
    let manifest_path = root.join("manifest.json");
    fs::write(&manifest_path, &manifest_bytes).unwrap();
    let inventory = root.join("blobs.jsonl");
    let mut inventory_rows = [content.clone()]
        .into_iter()
        .map(|object| {
            tensorfs::canon::write(&Value::obj(vec![
                (
                    "key",
                    Value::str(tensorfs::storage::blob_key(&object.sha256).unwrap()),
                ),
                ("length", Value::uint(object.length)),
            ]))
        })
        .collect::<Vec<_>>();
    inventory_rows.sort();
    fs::write(
        &inventory,
        inventory_rows
            .into_iter()
            .flat_map(|mut row| {
                row.push(b'\n');
                row
            })
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let checkpoint_mutation_path = root.join("checkpoint-mutation.json");
    fs::write(
        &checkpoint_mutation_path,
        checkpoint_mutation(&manifest_ref, "cozy", "marco-polo"),
    )
    .unwrap();
    let checkpointed = root.join("checkpointed.json");
    let checkpoint_rows = root.join("checkpoint-rows.jsonl");
    let checkpoint_result = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args([
            "repo",
            "apply",
            "-",
            checkpoint_mutation_path.to_str().unwrap(),
            "--manifest",
            manifest_path.to_str().unwrap(),
            "--inventory",
            inventory.to_str().unwrap(),
            "--out",
            checkpointed.to_str().unwrap(),
            "--rows",
            checkpoint_rows.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        checkpoint_result.status.success(),
        "{}",
        String::from_utf8_lossy(&checkpoint_result.stderr)
    );
    let mutation = root.join("mutation.json");
    fs::write(
        &mutation,
        tensorfs::canon::write(&Value::obj(vec![
            ("action", Value::str("update_release")),
            ("expected_revision", Value::uint(0)),
            ("remove", Value::arr(Vec::new())),
            (
                "repo",
                Value::obj(vec![
                    ("name", Value::str("marco-polo")),
                    ("org", Value::str("cozy")),
                ]),
            ),
            (
                "set",
                Value::arr(vec![Value::obj(vec![
                    ("lane", Value::str("source")),
                    ("manifest", manifest_ref.to_value()),
                ])]),
            ),
            ("version", Value::str("1.0.0")),
        ])),
    )
    .unwrap();
    let replacement = root.join("repository.json");
    let rows = root.join("rows.jsonl");
    let result = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args([
            "repo",
            "apply",
            checkpointed.to_str().unwrap(),
            mutation.to_str().unwrap(),
            "--manifest",
            manifest_path.to_str().unwrap(),
            "--inventory",
            inventory.to_str().unwrap(),
            "--out",
            replacement.to_str().unwrap(),
            "--rows",
            rows.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let repository = Repository::parse(&fs::read(replacement).unwrap()).unwrap();
    assert_eq!(repository.releases.len(), 1);
    let rows = fs::read_to_string(rows).unwrap();
    assert!(rows.contains("\"release_revision\":1"));
    assert!(!rows.contains("header_sha256"));
    assert!(!rows.contains("tensor_schema_digest"));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn committed_repo_apply_fixture_is_the_cross_process_projection_seam() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/repo_apply");
    let root = temporary("repo-apply-fixture");
    fs::create_dir_all(&root).unwrap();
    let manifest_bytes = fs::read(fixture.join("manifest.json")).unwrap();
    let manifest = ObjectRef::of(&manifest_bytes);
    let checkpoint_mutation_path = root.join("checkpoint-mutation.json");
    fs::write(
        &checkpoint_mutation_path,
        checkpoint_mutation(&manifest, "tensorhub", "marco-polo"),
    )
    .unwrap();
    let checkpointed = root.join("checkpointed.json");
    let checkpoint_rows = root.join("checkpoint-rows.jsonl");
    let checkpoint_status = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args([
            "repo",
            "apply",
            "-",
            checkpoint_mutation_path.to_str().unwrap(),
            "--manifest",
            fixture.join("manifest.json").to_str().unwrap(),
            "--header",
            fixture.join("header.cbor").to_str().unwrap(),
            "--inventory",
            fixture.join("blobs.jsonl").to_str().unwrap(),
            "--out",
            checkpointed.to_str().unwrap(),
            "--rows",
            checkpoint_rows.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(checkpoint_status.success());
    let release_mutation = root.join("release-mutation.json");
    let mut release_mutation_bytes = fs::read(fixture.join("mutation.json")).unwrap();
    if release_mutation_bytes.last() == Some(&b'\n') {
        release_mutation_bytes.pop();
    }
    fs::write(&release_mutation, release_mutation_bytes).unwrap();
    let repository = root.join("repository.json");
    let rows = root.join("rows.jsonl");
    let status = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args([
            "repo",
            "apply",
            checkpointed.to_str().unwrap(),
            release_mutation.to_str().unwrap(),
            "--manifest",
            fixture.join("manifest.json").to_str().unwrap(),
            "--header",
            fixture.join("header.cbor").to_str().unwrap(),
            "--inventory",
            fixture.join("blobs.jsonl").to_str().unwrap(),
            "--out",
            repository.to_str().unwrap(),
            "--rows",
            rows.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success());
    let mut expected_repository = fs::read(fixture.join("expected_repository.json")).unwrap();
    if expected_repository.last() == Some(&b'\n') {
        expected_repository.pop();
    }
    assert_eq!(fs::read(repository).unwrap(), expected_repository);
    assert_eq!(
        fs::read(rows).unwrap(),
        fs::read(fixture.join("expected_rows.jsonl")).unwrap()
    );

    let empty_inventory = root.join("empty.jsonl");
    fs::write(&empty_inventory, b"").unwrap();
    let refused = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args([
            "repo",
            "apply",
            "-",
            checkpoint_mutation_path.to_str().unwrap(),
            "--manifest",
            fixture.join("manifest.json").to_str().unwrap(),
            "--header",
            fixture.join("header.cbor").to_str().unwrap(),
            "--inventory",
            empty_inventory.to_str().unwrap(),
            "--out",
            root.join("refused.json").to_str().unwrap(),
            "--rows",
            root.join("refused.jsonl").to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(!refused.success());
    let _ = fs::remove_dir_all(root);
}

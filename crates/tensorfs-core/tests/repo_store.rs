use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tensorfs_core::canon::Value;
use tensorfs_core::dtype::Dtype;
use tensorfs_core::header::{Closure, Header, Part, Tensor};
use tensorfs_core::ids::{Doc, ObjectRef};
use tensorfs_core::manifest::{Draft, Entry, Manifest};
use tensorfs_core::repository::{CheckpointEvidence, Mutation, ReleaseLane, Repository};
use tensorfs_core::store::{ExpectedCheckpointContract, Fault, Store, Verdict};

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
    let mut permissions = fs::metadata(&path).unwrap().permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut permissions, 0o644);
    fs::set_permissions(&path, permissions).unwrap();
    fs::write(&path, b"corruption").unwrap();
    assert!(matches!(
        store.verify(&object.sha256).unwrap(),
        Verdict::CorruptRemoved { .. }
    ));
    assert!(!path.exists());

    fs::remove_dir_all(root.join("tmp")).unwrap();
    Store::open(&root).unwrap();
    assert!(root.join("tmp").is_dir());
    let _ = fs::remove_dir_all(root);
}

fn fixture(root: &Path) -> (ObjectRef, CheckpointEvidence) {
    let store = Store::init(root).unwrap();
    model_fixture(&store, [0, 0, 0, 0])
}

fn model_fixture(store: &Store, value: [u8; 4]) -> (ObjectRef, CheckpointEvidence) {
    let spec = tensorfs_core::registry::seeds()
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
    let evidence = CheckpointEvidence {
        classification_digest: header.topology_digest(),
        ingest: None,
        verification: None,
    };
    let bytes = header.canonical_bytes().unwrap();
    let header = store
        .put_stream(
            &mut bytes.as_slice(),
            Some(&ObjectRef::of(&bytes)),
            &Fault::default(),
        )
        .unwrap()
        .obj;
    let manifest = store
        .put_manifest(
            &Draft {
                entries: vec![("model.cozytensors".into(), Entry::CozyTensors(header))],
            }
            .seal()
            .unwrap(),
        )
        .unwrap()
        .obj;
    (manifest, evidence)
}

fn apply_release(
    store: &Store,
    observed: Option<&[u8]>,
    release: &Mutation,
) -> tensorfs_core::err::Result<Option<Repository>> {
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
                evidence: store
                    .read_manifest(manifest)?
                    .header()
                    .map(|header| {
                        tensorfs_core::checkpoint::load_header(store, header).map(|header| {
                            Box::new(CheckpointEvidence {
                                classification_digest: header.topology_digest(),
                                ingest: None,
                                verification: None,
                            })
                        })
                    })
                    .transpose()?,
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
    let repo = tensorfs_core::repository::RepositoryName::new("local", "h3-dev").unwrap();
    let (manifest_a, evidence_a) = model_fixture(&store, [0, 0, 0, 0]);
    let (manifest_b, evidence_b) = model_fixture(&store, [0, 0, 128, 63]);
    assert_ne!(manifest_a, manifest_b);
    let mutation_a = Mutation::ReplaceLocal {
        repo: repo.clone(),
        manifest: manifest_a.clone(),
        version: "11".repeat(32),
        evidence: Box::new(evidence_a),
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
        evidence: Box::new(evidence_b.clone()),
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
        tensorfs_core::err::Code::REPOSITORY_CONFLICT
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
                        lane: "other".into(),
                        manifest: manifest_b.clone(),
                    }],
                    version: "1.0.0".into(),
                },
                &Fault::default(),
            )
            .unwrap_err()
            .code,
        tensorfs_core::err::Code::KEY_GRAMMAR
    );

    let evidence_path = root.join("evidence.json");
    fs::write(&evidence_path, evidence_b.canonical_bytes()).unwrap();
    let header = tensorfs_core::checkpoint::load_header(
        &store,
        store.read_manifest(&manifest_b).unwrap().header().unwrap(),
    )
    .unwrap();
    let topology = header.topology_digest();
    let encoding = header.encodings[0].object_id();
    let replace = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args([
            "local",
            "replace",
            root.to_str().unwrap(),
            "h3-dev",
            &format!("sha256:{}", "33".repeat(32)),
            &manifest_b.id(),
            &manifest_b.length.to_string(),
            "--observed",
            &format!("sha256:{}", alias_b.document_sha256()),
            "--evidence",
            evidence_path.to_str().unwrap(),
            "--expected-topology",
            &topology,
            "--expected-encoding",
            &encoding,
        ])
        .output()
        .unwrap();
    assert!(
        replace.status.success(),
        "{}",
        String::from_utf8_lossy(&replace.stderr)
    );
    assert!(String::from_utf8_lossy(&replace.stdout).contains(&format!(
        "\"source_selection\":\"sha256:{}\"",
        "33".repeat(32)
    )));
    let resolve = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["local", "resolve", root.to_str().unwrap(), "h3-dev"])
        .output()
        .unwrap();
    assert!(resolve.status.success());
    assert_eq!(resolve.stdout, replace.stdout);
    let current = Repository::parse(&fs::read(store.repository_path(&repo)).unwrap()).unwrap();
    let digest = format!("sha256:{}", current.document_sha256());
    let remove = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args([
            "local",
            "remove",
            root.to_str().unwrap(),
            "h3-dev",
            "--observed",
            &digest,
        ])
        .output()
        .unwrap();
    assert!(
        remove.status.success(),
        "{}",
        String::from_utf8_lossy(&remove.stderr)
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn expected_local_contract_gates_positive_mismatch_replay_and_cas() {
    let root = temporary("local-contract");
    let store = Store::init(&root).unwrap();
    let repo = tensorfs_core::repository::RepositoryName::new("local", "contract").unwrap();
    let (manifest_a, evidence_a) = model_fixture(&store, [0, 0, 0, 0]);
    let (manifest_b, evidence_b) = model_fixture(&store, [0, 0, 128, 63]);
    let topology = evidence_b.classification_digest.clone();
    let encoding = tensorfs_core::registry::seeds()
        .into_iter()
        .find(|seed| seed.alias == "plain/1")
        .unwrap()
        .spec
        .object_id();
    let expected =
        ExpectedCheckpointContract::optional(Some(&topology), Some(vec![encoding.clone()]))
            .unwrap()
            .unwrap();
    let mutation_a = Mutation::ReplaceLocal {
        repo: repo.clone(),
        manifest: manifest_a,
        version: "11".repeat(32),
        evidence: Box::new(evidence_a),
    };
    let alias_a = store
        .apply_repository(None, &mutation_a, &Fault::default())
        .unwrap()
        .unwrap()
        .canonical_bytes();

    let mutation_b = Mutation::ReplaceLocal {
        repo: repo.clone(),
        manifest: manifest_b.clone(),
        version: "22".repeat(32),
        evidence: Box::new(evidence_b),
    };
    let alias_b = store
        .apply_repository_with_expected_local(
            Some(&alias_a),
            &mutation_b,
            Some(&expected),
            &Fault::default(),
        )
        .unwrap()
        .unwrap()
        .canonical_bytes();
    let alias_path = store.repository_path(&repo);
    assert_eq!(fs::read(&alias_path).unwrap(), alias_b);

    // Exact replay is the same Repository bytes and does not invent a generation.
    let replay = store
        .apply_repository_with_expected_local(
            Some(&alias_b),
            &mutation_b,
            Some(&expected),
            &Fault::default(),
        )
        .unwrap()
        .unwrap()
        .canonical_bytes();
    assert_eq!(replay, alias_b);

    let rejects =
        |observed: &[u8], contract: &ExpectedCheckpointContract, code: tensorfs_core::err::Code| {
            assert_eq!(
                store
                    .apply_repository_with_expected_local(
                        Some(observed),
                        &mutation_b,
                        Some(contract),
                        &Fault::default(),
                    )
                    .unwrap_err()
                    .code,
                code
            );
            assert_eq!(fs::read(&alias_path).unwrap(), alias_b);
        };

    let wrong_topology = ExpectedCheckpointContract::optional(
        Some(&format!("sha256:{}", "aa".repeat(32))),
        Some(vec![encoding]),
    )
    .unwrap()
    .unwrap();
    rejects(
        &alias_b,
        &wrong_topology,
        tensorfs_core::err::Code::TOPOLOGY_MISMATCH,
    );

    let wrong_encoding = ExpectedCheckpointContract::optional(
        Some(&topology),
        Some(vec![format!("sha256:{}", "bb".repeat(32))]),
    )
    .unwrap()
    .unwrap();
    rejects(
        &alias_b,
        &wrong_encoding,
        tensorfs_core::err::Code::ENCODING_MISMATCH,
    );

    assert_eq!(
        ExpectedCheckpointContract::optional(Some(&topology), Some(vec!["plain/1".into()]),)
            .unwrap_err()
            .code,
        tensorfs_core::err::Code::MALFORMED_DIGEST
    );
    for encoding_ids in [
        vec![
            format!("sha256:{}", "bb".repeat(32)),
            format!("sha256:{}", "aa".repeat(32)),
        ],
        vec![
            format!("sha256:{}", "aa".repeat(32)),
            format!("sha256:{}", "aa".repeat(32)),
        ],
    ] {
        assert_eq!(
            ExpectedCheckpointContract::optional(Some(&topology), Some(encoding_ids))
                .unwrap_err()
                .code,
            tensorfs_core::err::Code::SORT_ORDER
        );
    }

    // A correct artifact contract never weakens the caller's observed-bytes CAS.
    rejects(
        &alias_a,
        &expected,
        tensorfs_core::err::Code::REPOSITORY_CONFLICT,
    );
    let _ = fs::remove_dir_all(root);
}

fn mutation(manifest: &ObjectRef, evidence: &CheckpointEvidence) -> Vec<u8> {
    mutation_from_evidence(
        manifest,
        &evidence.canonical_bytes(),
        "1.0.0",
        "cpu",
        "org",
        "model",
    )
}

fn checkpoint_mutation(
    manifest: &ObjectRef,
    org: &str,
    name: &str,
    evidence: Option<&CheckpointEvidence>,
) -> Vec<u8> {
    checkpoint_mutation_from_evidence(
        manifest,
        org,
        name,
        evidence.map(CheckpointEvidence::canonical_bytes).as_deref(),
    )
}

fn checkpoint_mutation_from_evidence(
    manifest: &ObjectRef,
    org: &str,
    name: &str,
    evidence: Option<&[u8]>,
) -> Vec<u8> {
    let mut fields = vec![
        ("action", Value::str("put_checkpoint")),
        ("manifest", manifest.to_value()),
        (
            "repo",
            Value::obj(vec![("name", Value::str(name)), ("org", Value::str(org))]),
        ),
    ];
    if let Some(evidence) = evidence {
        fields.push((
            "evidence_base64",
            Value::str(tensorfs_core::b64::encode(evidence)),
        ));
    }
    tensorfs_core::canon::write(&Value::obj(fields))
}

fn mutation_from_evidence(
    manifest: &ObjectRef,
    _evidence: &[u8],
    version: &str,
    lane: &str,
    org: &str,
    name: &str,
) -> Vec<u8> {
    tensorfs_core::canon::write(&Value::obj(vec![
        ("action", Value::str("update_release")),
        ("expected_revision", Value::uint(0)),
        ("remove", Value::arr(Vec::new())),
        (
            "repo",
            Value::obj(vec![("name", Value::str(name)), ("org", Value::str(org))]),
        ),
        (
            "set",
            Value::arr(vec![Value::obj(vec![
                ("lane", Value::str(lane)),
                ("manifest", manifest.to_value()),
            ])]),
        ),
        ("version", Value::str(version)),
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
        tensorfs_core::canon::write(&Value::obj(vec![
            ("blob", empty.to_value()),
            ("path", Value::str("marco/__init__.py")),
        ])),
        tensorfs_core::canon::write(&Value::obj(vec![
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
            tensorfs_core::canon::write(&ObjectRef::of(&stored).to_value()),
            b"\n".to_vec(),
        ]
        .concat()
    );

    fs::write(
        &entries,
        [
            tensorfs_core::canon::write(&Value::obj(vec![
                ("blob", ObjectRef::of(b"later").to_value()),
                ("path", Value::str("z.py")),
            ])),
            b"\n".to_vec(),
            tensorfs_core::canon::write(&Value::obj(vec![
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

fn projection_snapshot(root: &Path) -> Vec<String> {
    let db = rusqlite::Connection::open(root.join("tensorfs.sqlite")).unwrap();
    let mut statement = db
        .prepare(
            "SELECT org||'|'||name||'|'||manifest_sha256
             FROM tensorfs_released_manifests ORDER BY 1",
        )
        .unwrap();
    statement
        .query_map([], |row| row.get::<_, String>(0))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

#[test]
fn both_cas_domains_rebuild_after_projection_tables_are_emptied() {
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
        let repo = tensorfs_core::repository::RepositoryName::new("recovery", domain).unwrap();
        apply_release(
            &store,
            None,
            &Mutation::UpdateRelease {
                expected_revision: 0,
                repo: repo.clone(),
                remove: Vec::new(),
                set: vec![ReleaseLane {
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
        let before = projection_snapshot(&store_root);
        assert_eq!(
            before.len(),
            1,
            "only released manifest lookup is projected"
        );

        let db = rusqlite::Connection::open(store_root.join("tensorfs.sqlite")).unwrap();
        db.execute("DELETE FROM tensorfs_released_manifests", [])
            .unwrap();
        drop(db);
        assert!(projection_snapshot(&store_root).is_empty());

        let rows = store_root.join("rebuild.jsonl");
        let result = Command::new(env!("CARGO_BIN_EXE_tfs"))
            .args([
                "store",
                "rebuild",
                store_root.to_str().unwrap(),
                "--rows",
                rows.to_str().unwrap(),
                "--sqlite",
                store_root.join("tensorfs.sqlite").to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{domain}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(projection_snapshot(&store_root), before, "{domain}");
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
            tensorfs_core::canon::write(&Value::obj(vec![
                (
                    "key",
                    Value::str(tensorfs_core::storage::blob_key(&object.sha256).unwrap()),
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
        checkpoint_mutation(&manifest_ref, "cozy", "marco-polo", None),
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
        tensorfs_core::canon::write(&Value::obj(vec![
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
    assert!(repository.checkpoints[0].evidence.is_none());
    let rows = fs::read_to_string(rows).unwrap();
    assert!(rows.contains("\"release_revision\":1"));
    assert!(!rows.contains("header_sha256"));
    assert!(!rows.contains("topology_digest"));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn crash_after_repo_replace_rebuilds_sql_from_files() {
    let root = temporary("crash-rebuild");
    let (manifest, evidence) = fixture(&root);
    let store = Store::open(&root).unwrap();
    let repo = tensorfs_core::repository::RepositoryName::new("org", "model").unwrap();
    let checkpointed = store
        .apply_repository(
            None,
            &Mutation::PutCheckpoint {
                repo,
                manifest: manifest.clone(),
                evidence: Some(Box::new(evidence.clone())),
            },
            &Fault::default(),
        )
        .unwrap()
        .unwrap();
    let checkpointed_path = root.join("checkpointed.json");
    fs::write(&checkpointed_path, checkpointed.canonical_bytes()).unwrap();
    let mutation_path = root.join("mutation.json");
    fs::write(&mutation_path, mutation(&manifest, &evidence)).unwrap();
    let ready = root.join("ready");
    let mut child = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args([
            "repo",
            "commit",
            root.to_str().unwrap(),
            checkpointed_path.to_str().unwrap(),
            mutation_path.to_str().unwrap(),
            "--fault",
            "repo-after-replace",
            "--ready",
            ready.to_str().unwrap(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let started = Instant::now();
    while !ready.is_file() && started.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(ready.is_file(), "repo writer never reached its crash point");
    let blocked_rows = root.join("blocked-rows.jsonl");
    assert!(
        !Command::new(env!("CARGO_BIN_EXE_tfs"))
            .args([
                "store",
                "rebuild",
                root.to_str().unwrap(),
                "--rows",
                blocked_rows.to_str().unwrap(),
            ])
            .status()
            .unwrap()
            .success(),
        "rebuild must refuse while the writer process retains its marker lock"
    );
    assert!(Command::new("kill")
        .args(["-9", &child.id().to_string()])
        .status()
        .unwrap()
        .success());
    let _ = child.wait();
    assert!(root.join("repos/org/model.json").is_file());
    let db = rusqlite::Connection::open(root.join("tensorfs.sqlite")).unwrap();
    let before: u64 = db
        .query_row(
            "SELECT count(*) FROM tensorfs_released_manifests",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(before, 0, "crash must not let SQL lead or half-commit");
    drop(db);

    let rows = root.join("rows.jsonl");
    let status = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args([
            "store",
            "rebuild",
            root.to_str().unwrap(),
            "--rows",
            rows.to_str().unwrap(),
            "--sqlite",
            root.join("tensorfs.sqlite").to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success());
    let db = rusqlite::Connection::open(root.join("tensorfs.sqlite")).unwrap();
    let after: u64 = db
        .query_row(
            "SELECT count(*) FROM tensorfs_released_manifests",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(after, 1);
    assert!(fs::read_to_string(rows)
        .unwrap()
        .contains("checkpoint_object"));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn committed_repo_apply_fixture_is_the_cross_process_projection_seam() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/repo_apply");
    let root = temporary("repo-apply-fixture");
    fs::create_dir_all(&root).unwrap();
    let manifest_bytes = fs::read(fixture.join("manifest.json")).unwrap();
    let manifest = ObjectRef::of(&manifest_bytes);
    let checkpoint_evidence = CheckpointEvidence {
        classification_digest:
            "sha256:070967dd3fa6e33a6993da22055d1216e6592903ac8d7b40f0299997b2518914".into(),
        ingest: None,
        verification: None,
    };
    let checkpoint_mutation_path = root.join("checkpoint-mutation.json");
    fs::write(
        &checkpoint_mutation_path,
        checkpoint_mutation(
            &manifest,
            "tensorhub",
            "marco-polo",
            Some(&checkpoint_evidence),
        ),
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

#[test]
fn repo_list_evidence_round_trips_as_an_opaque_raw_message() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/repo_apply");
    let root = temporary("repo-list-evidence");
    let store = Store::init(&root).unwrap();

    let header_bytes = fs::read(fixture.join("header.cbor")).unwrap();
    store
        .put_stream(
            &mut header_bytes.as_slice(),
            Some(&ObjectRef::of(&header_bytes)),
            &Fault::default(),
        )
        .unwrap();
    let manifest_bytes = fs::read(fixture.join("manifest.json")).unwrap();
    let manifest = Manifest::parse(&manifest_bytes).unwrap();
    let manifest_ref = store.put_manifest(&manifest).unwrap().obj;
    let mut first_mutation_bytes = fs::read(fixture.join("mutation.json")).unwrap();
    if first_mutation_bytes.last() == Some(&b'\n') {
        first_mutation_bytes.pop();
    }
    let first_mutation = Mutation::parse(&first_mutation_bytes).unwrap();
    apply_release(&store, None, &first_mutation).unwrap();

    let rows = root.join("list.jsonl");
    assert!(Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args([
            "repo",
            "list",
            root.to_str().unwrap(),
            "--rows",
            rows.to_str().unwrap(),
        ])
        .status()
        .unwrap()
        .success());
    let listed = fs::read(&rows).unwrap();
    assert_eq!(
        listed,
        fs::read(fixture.join("expected_list.jsonl")).unwrap()
    );

    let evidence = fs::read(fixture.join("evidence.json")).unwrap();
    let prefix = b"{\"evidence\":";
    assert!(listed.starts_with(prefix));
    let raw_message = &listed[prefix.len()..prefix.len() + evidence.len()];
    assert_eq!(
        raw_message, evidence,
        "repo list must preserve exact evidence bytes"
    );

    let current = root.join("current.json");
    assert!(Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args([
            "repo",
            "get",
            root.to_str().unwrap(),
            "tensorhub",
            "marco-polo",
            "--out",
            current.to_str().unwrap(),
        ])
        .status()
        .unwrap()
        .success());
    let second_mutation = root.join("second.json");
    fs::write(
        &second_mutation,
        mutation_from_evidence(
            &manifest_ref,
            raw_message,
            "2.0.0",
            "cpu",
            "tensorhub",
            "marco-polo",
        ),
    )
    .unwrap();
    assert!(Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args([
            "repo",
            "commit",
            root.to_str().unwrap(),
            current.to_str().unwrap(),
            second_mutation.to_str().unwrap(),
        ])
        .status()
        .unwrap()
        .success());
    let repository_bytes = fs::read(store.repository_path(first_mutation.repo())).unwrap();
    let repository = Repository::parse(&repository_bytes).unwrap();
    assert_eq!(repository.releases.len(), 2);
    assert_eq!(repository.checkpoints.len(), 1);
    assert!(
        repository.checkpoints[0]
            .evidence
            .as_ref()
            .unwrap()
            .canonical_bytes()
            == raw_message
    );

    let before_red = repository_bytes;
    let current_red = root.join("current-red.json");
    fs::write(&current_red, &before_red).unwrap();
    let mut noncanonical = raw_message.to_vec();
    noncanonical.push(b' ');
    let red_mutation = root.join("red.json");
    fs::write(
        &red_mutation,
        checkpoint_mutation_from_evidence(
            &manifest_ref,
            "tensorhub",
            "marco-polo",
            Some(&noncanonical),
        ),
    )
    .unwrap();
    let red = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args([
            "repo",
            "commit",
            root.to_str().unwrap(),
            current_red.to_str().unwrap(),
            red_mutation.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!red.status.success());
    assert!(String::from_utf8_lossy(&red.stderr).contains("NONCANONICAL_ENCODING"));
    assert_eq!(
        fs::read(store.repository_path(first_mutation.repo())).unwrap(),
        before_red,
        "refused evidence must not mutate the repository",
    );

    let _ = fs::remove_dir_all(root);
}

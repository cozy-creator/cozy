use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tensorfs_core::canon::Value;
use tensorfs_core::dtype::Dtype;
use tensorfs_core::header::{Closure, Header, Part, Tensor};
use tensorfs_core::ids::{Doc, ObjectRef};
use tensorfs_core::manifest::{Draft, Entry, Manifest};
use tensorfs_core::repository::{Mutation, ReleaseEvidence, Repository};
use tensorfs_core::store::{Fault, Store};

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

fn fixture(root: &Path) -> (ObjectRef, ReleaseEvidence) {
    let store = Store::init(root).unwrap();
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
                    parts: vec![(
                        "value".into(),
                        Part::plan(Dtype::F32, vec![1], &[0, 0, 0, 0]),
                    )],
                },
            )],
        )],
    };
    header.validate(&Closure::default()).unwrap();
    let evidence = ReleaseEvidence {
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

fn mutation(manifest: &ObjectRef, evidence: &ReleaseEvidence) -> Vec<u8> {
    mutation_from_evidence(
        manifest,
        &evidence.canonical_bytes(),
        "1.0.0",
        "cpu",
        "org",
        "model",
    )
}

fn mutation_from_evidence(
    manifest: &ObjectRef,
    evidence: &[u8],
    version: &str,
    lane: &str,
    org: &str,
    name: &str,
) -> Vec<u8> {
    tensorfs_core::canon::write(&Value::obj(vec![
        ("action", Value::str("put_release")),
        (
            "evidence_base64",
            Value::str(tensorfs_core::b64::encode(evidence)),
        ),
        ("lane", Value::str(lane)),
        ("manifest", manifest.to_value()),
        (
            "repo",
            Value::obj(vec![("name", Value::str(name)), ("org", Value::str(org))]),
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
            ("path", Value::str("marco/endpoint.py")),
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
        ("marco/endpoint.py".into(), source),
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
    let mut out = Vec::new();
    for sql in [
        "SELECT org||'|'||name||'|'||document_sha256 FROM tensorfs_repos ORDER BY 1",
        "SELECT org||'|'||name||'|'||version||'|'||lane||'|'||manifest_sha256||'|'||manifest_length||'|'||hex(evidence_json)||'|'||header_sha256||'|'||header_length||'|'||topology_digest||'|'||object_count||'|'||object_bytes FROM tensorfs_releases ORDER BY 1",
        "SELECT org||'|'||name||'|'||version||'|'||lane||'|'||sha256||'|'||length||'|'||object_kind FROM tensorfs_release_objects ORDER BY 1",
    ] {
        let mut statement = db.prepare(sql).unwrap();
        out.extend(
            statement
                .query_map([], |row| row.get::<_, String>(0))
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
        );
    }
    out
}

#[test]
fn four_domain_ordinary_repositories_rebuild_after_projection_tables_are_emptied() {
    let root = temporary("four-domain-rebuild");
    for domain in ["repo_cas", "package_source", "dataset_cas", "user_media"] {
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
        let metadata = if domain == "package_source" {
            let bytes = b"{\"format\":\"cozy.package.release/1\"}";
            Some(
                store
                    .put_stream(
                        &mut bytes.as_slice(),
                        Some(&ObjectRef::of(bytes)),
                        &Fault::default(),
                    )
                    .unwrap()
                    .obj,
            )
        } else {
            None
        };
        let repo = tensorfs_core::repository::RepositoryName::new("recovery", domain).unwrap();
        store
            .apply_repository(
                None,
                &Mutation::PutRelease {
                    repo: repo.clone(),
                    lane: "main".into(),
                    manifest: manifest_ref,
                    metadata: metadata.clone(),
                    version: "1.0.0".into(),
                    evidence: None,
                },
                &Fault::default(),
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
            if metadata.is_some() { 4 } else { 3 },
            "one repo, release, and every reached blob"
        );

        let db = rusqlite::Connection::open(store_root.join("tensorfs.sqlite")).unwrap();
        db.execute("DELETE FROM tensorfs_repos", []).unwrap();
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
        assert!(rows.contains("\"kind\":\"release_object\""));
    }
    let _ = fs::remove_dir_all(root);
}

#[test]
fn ordinary_repo_apply_needs_no_model_header_or_evidence() {
    let root = temporary("ordinary-repo-apply");
    fs::create_dir_all(&root).unwrap();
    let content = ObjectRef::of(b"package source");
    let metadata = ObjectRef::of(b"package release");
    let manifest = Manifest::from_files(vec![("src/package.py".into(), content.clone())]).unwrap();
    let manifest_bytes = manifest.canonical_bytes();
    let manifest_ref = ObjectRef::of(&manifest_bytes);
    let manifest_path = root.join("manifest.json");
    fs::write(&manifest_path, &manifest_bytes).unwrap();
    let inventory = root.join("blobs.jsonl");
    let mut inventory_rows = [content.clone(), metadata.clone()]
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
    let mutation = root.join("mutation.json");
    fs::write(
        &mutation,
        tensorfs_core::canon::write(&Value::obj(vec![
            ("action", Value::str("put_release")),
            ("lane", Value::str("source")),
            ("manifest", manifest_ref.to_value()),
            ("metadata", metadata.to_value()),
            (
                "repo",
                Value::obj(vec![
                    ("name", Value::str("marco-polo")),
                    ("org", Value::str("cozy")),
                ]),
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
            "-",
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
    assert!(repository.releases[0].evidence.is_none());
    assert_eq!(repository.releases[0].metadata.as_ref(), Some(&metadata));
    let rows = fs::read_to_string(rows).unwrap();
    assert!(rows.contains("\"object_kind\":\"file\""));
    assert!(rows.contains("\"object_kind\":\"metadata\""));
    assert!(!rows.contains("header_sha256"));
    assert!(!rows.contains("topology_digest"));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn crash_after_repo_replace_rebuilds_sql_from_files() {
    let root = temporary("crash-rebuild");
    let (manifest, evidence) = fixture(&root);
    let mutation_path = root.join("mutation.json");
    fs::write(&mutation_path, mutation(&manifest, &evidence)).unwrap();
    let ready = root.join("ready");
    let mut child = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args([
            "repo",
            "commit",
            root.to_str().unwrap(),
            "-",
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
        .query_row("SELECT count(*) FROM tensorfs_releases", [], |row| {
            row.get(0)
        })
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
        .query_row("SELECT count(*) FROM tensorfs_releases", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(after, 1);
    assert!(fs::read_to_string(rows).unwrap().contains("release_object"));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn committed_repo_apply_fixture_is_the_cross_process_projection_seam() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/repo_apply");
    let root = temporary("repo-apply-fixture");
    fs::create_dir_all(&root).unwrap();
    let repository = root.join("repository.json");
    let rows = root.join("rows.jsonl");
    let status = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args([
            "repo",
            "apply",
            "-",
            fixture.join("mutation.json").to_str().unwrap(),
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
    assert_eq!(
        fs::read(repository).unwrap(),
        fs::read(fixture.join("expected_repository.json")).unwrap()
    );
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
            fixture.join("mutation.json").to_str().unwrap(),
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
    let first_mutation =
        Mutation::parse(&fs::read(fixture.join("mutation.json")).unwrap()).unwrap();
    store
        .apply_repository(None, &first_mutation, &Fault::default())
        .unwrap();

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
    assert!(repository.releases.iter().all(|release| release
        .evidence
        .as_ref()
        .unwrap()
        .canonical_bytes()
        == raw_message));

    let before_red = repository_bytes;
    let current_red = root.join("current-red.json");
    fs::write(&current_red, &before_red).unwrap();
    let mut noncanonical = raw_message.to_vec();
    noncanonical.push(b' ');
    let red_mutation = root.join("red.json");
    fs::write(
        &red_mutation,
        mutation_from_evidence(
            &manifest_ref,
            &noncanonical,
            "3.0.0",
            "cpu",
            "tensorhub",
            "marco-polo",
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

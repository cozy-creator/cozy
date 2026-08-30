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
    assert!(repository
        .releases
        .iter()
        .all(|release| release.evidence.canonical_bytes() == raw_message));

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

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tensorfs_core::canon::Value;
use tensorfs_core::dtype::Dtype;
use tensorfs_core::header::{Closure, Header, Part, Tensor};
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::manifest::{Draft, Entry};
use tensorfs_core::repository::ReleaseEvidence;
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
    tensorfs_core::canon::write(&Value::obj(vec![
        ("action", Value::str("put_release")),
        (
            "evidence_base64",
            Value::str(tensorfs_core::b64::encode(&evidence.canonical_bytes())),
        ),
        ("lane", Value::str("cpu")),
        ("manifest", manifest.to_value()),
        (
            "repo",
            Value::obj(vec![
                ("name", Value::str("model")),
                ("org", Value::str("org")),
            ]),
        ),
        ("version", Value::str("1.0.0")),
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

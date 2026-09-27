use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::Barrier;

use tensorfs_core::checkpoint_root;
use tensorfs_core::dtype::Dtype;
use tensorfs_core::err::Code;
use tensorfs_core::header::{Header, Part, Tensor};
use tensorfs_core::ids::{object_id, Doc, ObjectRef};
use tensorfs_core::manifest::{Draft, Entry};
use tensorfs_core::repository::{Mutation, RepositoryName};
use tensorfs_core::store::{Fault, Store};

const REPOSITORY: &str = "models/checkpoint";

struct Fixture {
    root: PathBuf,
    store: Store,
    manifest: ObjectRef,
    data: ObjectRef,
    optional: ObjectRef,
    repository: Vec<u8>,
}

impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "tensorfs-checkpoint-roots-{}-{}",
            std::process::id(),
            tensorfs_core::meta::now_nanos_unique(),
        ));
        let store = Store::init(&root).unwrap();
        let bytes = vec![0x31; 2048];
        let data = store
            .put_stream(&mut bytes.as_slice(), None, &Fault::default())
            .unwrap()
            .obj;
        let plain = tensorfs_core::registry::seeds()
            .into_iter()
            .find(|seed| seed.alias == "plain/1")
            .unwrap()
            .spec;
        let header = Header {
            configs: vec![],
            assets: vec![],
            encodings: vec![plain.clone()],
            components: vec![(
                "model".into(),
                vec![(
                    "weight".into(),
                    Tensor {
                        dtype: Dtype::U8,
                        shape: vec![2048],
                        encoding: plain.object_id(),
                        parts: vec![("value".into(), Part::plan(Dtype::U8, vec![2048], &bytes))],
                    },
                )],
            )],
        };
        let header = store
            .put_stream(
                &mut header.canonical_bytes().unwrap().as_slice(),
                None,
                &Fault::default(),
            )
            .unwrap()
            .obj;
        let optional = store
            .put_stream(
                &mut &b"optional source documentation"[..],
                None,
                &Fault::default(),
            )
            .unwrap()
            .obj;
        let model = Draft {
            entries: vec![
                ("cozytensors".into(), Entry::CozyTensors(header)),
                ("optional.txt".into(), Entry::File(optional.clone())),
                ("payload".into(), Entry::File(data.clone())),
            ],
        }
        .seal()
        .unwrap();
        let manifest = store.put_manifest(&model).unwrap().obj;
        let repository = store
            .apply_repository(
                None,
                &Mutation::PutCheckpoint {
                    repo: RepositoryName::new("models", "checkpoint").unwrap(),
                    manifest: manifest.clone(),
                },
                &Fault::default(),
            )
            .unwrap()
            .unwrap()
            .canonical_bytes();
        Self {
            root,
            store,
            manifest,
            data,
            optional,
            repository,
        }
    }

    fn remove_repository(&self) {
        self.store
            .apply_repository(
                Some(&self.repository),
                &Mutation::DeleteRepository {
                    repo: RepositoryName::new("models", "checkpoint").unwrap(),
                },
                &Fault::default(),
            )
            .unwrap();
    }

    fn gc(&self) {
        tensorfs_core::gc::collect(&self.root, false).unwrap();
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn independent_checkpoint_roots_survive_repository_removal_and_gc() {
    let fixture = Fixture::new();
    let first = object_id(b"first");
    let second = object_id(b"second");
    let root =
        checkpoint_root::retain(&fixture.store, &first, REPOSITORY, fixture.manifest.clone())
            .unwrap();
    checkpoint_root::retain(
        &fixture.store,
        &second,
        REPOSITORY,
        fixture.manifest.clone(),
    )
    .unwrap();
    fixture.remove_repository();
    fixture.gc();
    assert!(!fixture.store.contains(&fixture.optional.sha256));
    let reopened = Store::open(&fixture.root).unwrap();
    assert_eq!(
        checkpoint_root::retain(&reopened, &first, REPOSITORY, fixture.manifest.clone()).unwrap(),
        root
    );
    checkpoint_root::release(&reopened, &first, REPOSITORY, fixture.manifest.clone()).unwrap();
    fixture.gc();
    assert!(fixture.store.contains(&fixture.data.sha256));
    checkpoint_root::release(&reopened, &second, REPOSITORY, fixture.manifest.clone()).unwrap();
    fixture.gc();
    assert!(!fixture.store.contains(&fixture.data.sha256));
    assert!(
        checkpoint_root::read(&reopened, &first)
            .unwrap()
            .unwrap()
            .released
    );
}

#[test]
fn runtime_closure_pin_does_not_require_optional_snapshot_payload() {
    let fixture = Fixture::new();
    fs::remove_file(fixture.store.blob_path(&fixture.optional.sha256)).unwrap();
    checkpoint_root::check_source(&fixture.store, REPOSITORY, &fixture.manifest).unwrap();
    assert_eq!(
        fixture
            .store
            .apply_repository(
                None,
                &Mutation::ReplaceLocal {
                    repo: RepositoryName::new("local", "requires-whole-snapshot").unwrap(),
                    version: "ab".repeat(32),
                    manifest: fixture.manifest.clone(),
                },
                &Fault::default()
            )
            .unwrap_err()
            .code,
        Code::OBJECT_ABSENT
    );
    let owner = object_id(b"runtime-only");
    checkpoint_root::retain(&fixture.store, &owner, REPOSITORY, fixture.manifest.clone()).unwrap();
    fixture.remove_repository();
    fixture.gc();
    assert!(fixture.store.contains(&fixture.data.sha256));
    checkpoint_root::release(&fixture.store, &owner, REPOSITORY, fixture.manifest.clone()).unwrap();
    fixture.gc();
    assert!(!fixture.store.contains(&fixture.data.sha256));
}

#[test]
fn a_full_snapshot_hold_wins_over_a_runtime_only_hold() {
    let fixture = Fixture::new();
    fixture.remove_repository();
    let full = tensorfs_core::storage::HeldKey {
        key: tensorfs_core::storage::manifest_key(&fixture.manifest.sha256).unwrap(),
        kind: "manifest".into(),
        length: fixture.manifest.length,
        sha256: fixture.manifest.sha256.clone(),
    };
    let runtime = tensorfs_core::storage::HeldKey {
        kind: "cozytensors".into(),
        ..full.clone()
    };
    let census = tensorfs_core::storage::Census::open(&fixture.root).unwrap();
    let only_runtime = census.gc_plan(std::slice::from_ref(&runtime)).unwrap();
    assert!(only_runtime
        .iter()
        .any(|row| row.sha256 == fixture.optional.sha256));
    for holds in [
        [runtime.clone(), full.clone()],
        [full.clone(), runtime.clone()],
    ] {
        let plan = census.gc_plan(&holds).unwrap();
        assert!(!plan.iter().any(|row| row.sha256 == fixture.optional.sha256));
    }
}

#[test]
fn source_identity_records_and_release_fences_are_native() {
    let fixture = Fixture::new();
    let owner = object_id(b"control");
    assert_eq!(
        checkpoint_root::retain(
            &fixture.store,
            &owner,
            "models/missing",
            fixture.manifest.clone()
        )
        .unwrap_err()
        .code,
        Code::REPOSITORY_ABSENT
    );
    let wrong = ObjectRef {
        length: fixture.manifest.length + 1,
        ..fixture.manifest.clone()
    };
    assert_eq!(
        checkpoint_root::retain(&fixture.store, &owner, REPOSITORY, wrong.clone())
            .unwrap_err()
            .code,
        Code::ROOT_ABSENT
    );
    assert!(checkpoint_root::read(&fixture.store, &owner)
        .unwrap()
        .is_none());
    checkpoint_root::retain(&fixture.store, &owner, REPOSITORY, fixture.manifest.clone()).unwrap();
    assert_eq!(
        checkpoint_root::release(&fixture.store, &owner, REPOSITORY, wrong)
            .unwrap_err()
            .code,
        Code::CROSS_SUBJECT_REPLAY
    );
    checkpoint_root::release(&fixture.store, &owner, REPOSITORY, fixture.manifest.clone()).unwrap();
    assert_eq!(
        checkpoint_root::retain(&fixture.store, &owner, REPOSITORY, fixture.manifest.clone())
            .unwrap_err()
            .code,
        Code::TRANSACTION_CLOSED
    );
    let canceled = object_id(b"canceled-before-acquire");
    checkpoint_root::release(
        &fixture.store,
        &canceled,
        REPOSITORY,
        fixture.manifest.clone(),
    )
    .unwrap();
    assert_eq!(
        checkpoint_root::retain(
            &fixture.store,
            &canceled,
            REPOSITORY,
            fixture.manifest.clone()
        )
        .unwrap_err()
        .code,
        Code::TRANSACTION_CLOSED
    );
    let absent = object_id(b"missing-payload");
    fs::remove_file(fixture.store.blob_path(&fixture.data.sha256)).unwrap();
    assert_eq!(
        checkpoint_root::retain(
            &fixture.store,
            &absent,
            REPOSITORY,
            fixture.manifest.clone()
        )
        .unwrap_err()
        .code,
        Code::OBJECT_ABSENT
    );
    assert!(checkpoint_root::read(&fixture.store, &absent)
        .unwrap()
        .is_none());
}

#[test]
fn corrupt_payload_cannot_become_an_input_root() {
    let fixture = Fixture::new();
    let owner = object_id(b"corrupt");
    let path = fixture.store.blob_path(&fixture.data.sha256);
    fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
    fs::write(path, vec![0x32; 2048]).unwrap();
    assert_eq!(
        checkpoint_root::retain(&fixture.store, &owner, REPOSITORY, fixture.manifest.clone())
            .unwrap_err()
            .code,
        Code::OBJECT_CORRUPT
    );
    assert!(checkpoint_root::read(&fixture.store, &owner)
        .unwrap()
        .is_none());
}

#[test]
fn concurrent_release_permanently_fences_acquisition() {
    let fixture = Fixture::new();
    for index in 0..8 {
        let owner = object_id(format!("race-{index}").as_bytes());
        let barrier = Barrier::new(3);
        std::thread::scope(|scope| {
            let barrier = &barrier;
            let store = &fixture.store;
            let manifest = &fixture.manifest;
            let owner = &owner;
            let acquire = scope.spawn(move || {
                barrier.wait();
                checkpoint_root::retain(store, owner, REPOSITORY, manifest.clone())
            });
            let release = scope.spawn(move || {
                barrier.wait();
                checkpoint_root::release(store, owner, REPOSITORY, manifest.clone()).unwrap()
            });
            barrier.wait();
            if let Err(error) = acquire.join().unwrap() {
                assert_eq!(error.code, Code::TRANSACTION_CLOSED);
            }
            assert!(release.join().unwrap().released);
        });
        assert!(
            checkpoint_root::read(&fixture.store, &owner)
                .unwrap()
                .unwrap()
                .released
        );
    }
}

#[test]
fn checkpoint_pin_process_child() {
    let Some(root) = std::env::var_os("TFS_CHECKPOINT_PIN_CHILD") else {
        return;
    };
    let root = PathBuf::from(root);
    let store = Store::open(&root).unwrap();
    let reference = fs::read_to_string(root.join("test-subject.json")).unwrap();
    let value = tensorfs_core::canon::parse(reference.as_bytes(), 4096).unwrap();
    let manifest = ObjectRef::from_value("test manifest", &value).unwrap();
    checkpoint_root::retain(&store, &object_id(b"process-root"), REPOSITORY, manifest).unwrap();
    std::process::exit(73);
}

#[test]
fn completed_pin_survives_actual_process_exit() {
    let fixture = Fixture::new();
    fs::write(
        fixture.root.join("test-subject.json"),
        tensorfs_core::canon::write(&fixture.manifest.to_value()),
    )
    .unwrap();
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "checkpoint_pin_process_child", "--nocapture"])
        .env("TFS_CHECKPOINT_PIN_CHILD", &fixture.root)
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(73));
    fixture.remove_repository();
    fixture.gc();
    let reopened = Store::open(&fixture.root).unwrap();
    assert!(
        !checkpoint_root::read(&reopened, &object_id(b"process-root"))
            .unwrap()
            .unwrap()
            .released
    );
    assert!(reopened.contains(&fixture.data.sha256));
}

#[test]
fn a_root_written_by_a_newer_tensorfs_is_refused_naming_that_version() {
    let fixture = Fixture::new();
    let owner = object_id(b"newer-writer");
    checkpoint_root::retain(&fixture.store, &owner, REPOSITORY, fixture.manifest.clone()).unwrap();
    let path = fixture
        .root
        .join("roots/checkpoints")
        .join(format!("{}.json", &owner[7..]));
    let text = fs::read_to_string(&path).unwrap();
    let ours = format!("\"tensorfs\":\"{}\"}}", tensorfs_core::VERSION);
    assert!(text.ends_with(&ours), "{text}");
    fs::write(
        &path,
        text.replace(&ours, "\"tensorfs\":\"9.9.9\",\"zones\":[]}"),
    )
    .unwrap();
    let refusal = checkpoint_root::read(&fixture.store, &owner).unwrap_err();
    assert_eq!(refusal.code, Code::UNKNOWN_FIELD);
    assert!(
        refusal.detail.contains("TensorFS 9.9.9") && refusal.detail.contains("tensorfs>=9.9.9"),
        "{}",
        refusal.detail
    );
}

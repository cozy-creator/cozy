use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use tensorfs_core::checkpoint_root;
use tensorfs_core::dtype::Dtype;
use tensorfs_core::err::Code;
use tensorfs_core::header::{Header, Part, Tensor};
use tensorfs_core::ids::{Doc, ObjectRef};
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
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn a_checkpoint_input_does_not_require_optional_snapshot_payload() {
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
fn a_corrupt_payload_is_not_a_valid_input() {
    let fixture = Fixture::new();
    let path = fixture.store.blob_path(&fixture.data.sha256);
    fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
    fs::write(path, vec![0x32; 2048]).unwrap();
    assert_eq!(
        checkpoint_root::check_source(&fixture.store, REPOSITORY, &fixture.manifest)
            .unwrap_err()
            .code,
        Code::OBJECT_CORRUPT
    );
}

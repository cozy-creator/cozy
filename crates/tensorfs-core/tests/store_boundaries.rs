use std::fs;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use tensorfs_core::err::Code;
use tensorfs_core::header::{Body, Closure, Header, Part, Tensor};
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::manifest::{Draft, Entry, Manifest};
use tensorfs_core::store::{Fault, Store};
use tensorfs_core::{dtype::Dtype, registry};

fn temporary(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tensorfs-boundary-{name}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn mode(path: &Path) -> u32 {
    fs::symlink_metadata(path).unwrap().permissions().mode() & 0o7777
}

fn model_manifest(store: &Store, body: Body, shape: Vec<u64>, note: &[u8]) -> ObjectRef {
    let spec = registry::seeds()
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
                    shape: shape.clone(),
                    encoding: spec.object_id(),
                    parts: vec![(
                        "value".into(),
                        Part {
                            dtype: Dtype::F32,
                            shape,
                            body,
                        },
                    )],
                },
            )],
        )],
    };
    header.validate(&Closure::default()).unwrap();
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
        entries: vec![
            ("README.txt".into(), Entry::File(ObjectRef::of(note))),
            ("model.cozytensors".into(), Entry::CozyTensors(header_ref)),
        ],
    }
    .seal()
    .unwrap();
    store.put_manifest(&manifest).unwrap().obj
}

#[test]
fn ensure_initializes_only_absent_or_empty_real_roots() {
    let root = temporary("ensure");
    fs::create_dir_all(&root).unwrap();

    let absent = root.join("absent");
    let output = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["store", "ensure", absent.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8(output.stdout)
        .unwrap()
        .contains(&format!("root:       {}", absent.display())));
    Store::ensure(&absent).unwrap();
    Store::ensure(&absent).unwrap();

    let empty = root.join("empty");
    fs::create_dir(&empty).unwrap();
    Store::ensure(&empty).unwrap();

    let foreign = root.join("foreign");
    fs::create_dir(&foreign).unwrap();
    fs::write(foreign.join("somebody-elses-state"), b"keep").unwrap();
    assert_eq!(Store::ensure(&foreign).unwrap_err().code, Code::STORE_ERA);
    assert!(!foreign.join("repos").exists());

    let partial = root.join("partial");
    fs::create_dir(&partial).unwrap();
    fs::create_dir(partial.join("repos")).unwrap();
    fs::create_dir(partial.join("manifests")).unwrap();
    assert_eq!(Store::ensure(&partial).unwrap_err().code, Code::STORE_ERA);
    assert!(!partial.join("blobs").exists());

    let real = root.join("real");
    Store::init(&real).unwrap();
    let link = root.join("link");
    symlink(&real, &link).unwrap();
    assert_eq!(Store::ensure(&link).unwrap_err().code, Code::STORE_ERA);
    assert_eq!(Store::open(&link).unwrap_err().code, Code::STORE_ERA);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn prepare_readers_owns_exact_modes_and_refuses_internal_symlinks() {
    let root = temporary("readers");
    let store_root = root.join("store");
    fs::create_dir_all(&root).unwrap();
    Store::ensure(&store_root).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["store", "prepare-readers", store_root.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    for path in [
        "",
        "repos",
        "manifests",
        "blobs",
        "checkouts",
        "tmp",
        "tmp/writers",
    ] {
        assert_eq!(mode(&store_root.join(path)), 0o755, "{path}");
    }
    assert_eq!(mode(&store_root.join("tmp/leases")), 0o1733);
    assert_eq!(mode(&store_root.join("tmp/writers/recovery.lock")), 0o666);

    let outside = root.join("outside");
    fs::create_dir(&outside).unwrap();
    fs::set_permissions(&outside, fs::Permissions::from_mode(0o700)).unwrap();
    fs::remove_dir(store_root.join("tmp/leases")).unwrap();
    symlink(&outside, store_root.join("tmp/leases")).unwrap();
    let store = Store::open(&store_root).unwrap();
    assert!(store.prepare_readers().is_err());
    assert_eq!(mode(&outside), 0o700);

    fs::remove_file(store_root.join("tmp/leases")).unwrap();
    fs::create_dir(store_root.join("tmp/leases")).unwrap();
    let outside_file = root.join("outside-lock");
    fs::write(&outside_file, b"not the lock").unwrap();
    fs::set_permissions(&outside_file, fs::Permissions::from_mode(0o600)).unwrap();
    fs::remove_file(store_root.join("tmp/writers/recovery.lock")).unwrap();
    symlink(&outside_file, store_root.join("tmp/writers/recovery.lock")).unwrap();
    assert!(store.prepare_readers().is_err());
    assert_eq!(mode(&outside_file), 0o600);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn complete_cozytensors_census_is_sorted_verified_and_runtime_only() {
    let root = temporary("census");
    let store = Store::ensure(&root).unwrap();
    let complete_a = model_manifest(&store, Body::Inline(vec![0; 4]), vec![1], b"missing-a");
    let complete_b = model_manifest(&store, Body::Inline(vec![1; 4]), vec![1], b"missing-b");
    let missing_part = ObjectRef::of(&vec![2; 260]);
    let _incomplete = model_manifest(
        &store,
        Body::Segments(vec![missing_part]),
        vec![65],
        b"missing-c",
    );
    let ordinary = Manifest::from_files(vec![(
        "dataset.jsonl".into(),
        ObjectRef::of(b"not resident"),
    )])
    .unwrap();
    store.put_manifest(&ordinary).unwrap();
    let corrupt = "11".repeat(32);
    let corrupt_path = store.manifest_path(&corrupt);
    fs::create_dir_all(corrupt_path.parent().unwrap()).unwrap();
    fs::write(corrupt_path, b"not canonical manifest bytes").unwrap();

    let mut expected = vec![complete_a.id(), complete_b.id()];
    expected.sort();
    assert_eq!(store.complete_cozytensors_manifests().unwrap(), expected);

    let output = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["store", "complete-cozytensors", root.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect::<Vec<_>>(),
        expected
    );
    fs::remove_dir_all(root).unwrap();
}

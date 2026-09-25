//! proto-061: landed bytes are hashed once and trusted from one per-process catalog load.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use tensorfs_core::catalog::connections_opened;
use tensorfs_core::err::Code;
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::store::{Fault, Store, Verdict};

fn temporary(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "tensorfs-trust-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = fs::remove_dir_all(&root);
    root
}

/// Land `bytes` through another process, so this one has never seen its row.
fn put_elsewhere(root: &Path, scratch: &Path, bytes: &[u8]) -> ObjectRef {
    let file = scratch.join(format!("{}.bin", ObjectRef::of(bytes).sha256));
    fs::write(&file, bytes).unwrap();
    let put = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["put", root.to_str().unwrap(), file.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        put.status.success(),
        "{}",
        String::from_utf8_lossy(&put.stderr)
    );
    ObjectRef::of(bytes)
}

fn body(index: usize) -> Vec<u8> {
    let mut bytes = vec![(index % 251) as u8; 4096 + index];
    bytes[..8].copy_from_slice(&(index as u64).to_le_bytes());
    bytes
}

fn rewrite_in_place(path: &Path, keep_mtime: bool) {
    let before = fs::metadata(path).unwrap().modified().unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o644)).unwrap();
    let length = fs::metadata(path).unwrap().len() as usize;
    fs::write(path, vec![0xA5; length]).unwrap();
    if keep_mtime {
        fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(before)
            .unwrap();
    }
}

#[test]
fn trust_checks_load_the_catalog_once_and_bad_bytes_are_still_caught() {
    let root = temporary("store");
    let scratch = temporary("scratch");
    fs::create_dir_all(&scratch).unwrap();
    Store::init(&root).unwrap();
    let objects: Vec<ObjectRef> = (0..64)
        .map(|index| put_elsewhere(&root, &scratch, &body(index)))
        .collect();

    // Three full passes, as a placement switch takes: one catalog open in total.
    let store = Store::open(&root).unwrap();
    let opened = connections_opened(&root);
    for _ in 0..3 {
        for object in &objects {
            assert_eq!(
                store.verify(&object.sha256).unwrap(),
                Verdict::Verified { rehashed: false }
            );
            store.record_valid(&object.sha256).unwrap();
        }
    }
    assert_eq!(connections_opened(&root) - opened, 1);

    // A row another process writes after the load is found on the kept connection.
    let late = put_elsewhere(&root, &scratch, &body(1000));
    assert!(
        !store.open_verified_reporting(&late.sha256).unwrap().1,
        "a recorded object must not be rehashed"
    );
    assert_eq!(connections_opened(&root) - opened, 1);

    // Missing bytes refuse.
    fs::remove_file(store.blob_path(&objects[0].sha256)).unwrap();
    assert_eq!(
        store.verify(&objects[0].sha256).unwrap_err().code,
        Code::OBJECT_ABSENT
    );

    // Changed bytes break the record's binding, are rehashed, and are removed.
    let changed = store.blob_path(&objects[1].sha256);
    rewrite_in_place(&changed, false);
    assert!(matches!(
        store.verify(&objects[1].sha256).unwrap(),
        Verdict::CorruptRemoved { .. }
    ));
    assert!(!changed.exists());

    // Bytes changed under an intact binding are the scrub's to find, and it finds them.
    let hidden = store.blob_path(&objects[2].sha256);
    rewrite_in_place(&hidden, true);
    assert_eq!(
        store.verify(&objects[2].sha256).unwrap(),
        Verdict::Verified { rehashed: false }
    );
    let scrub = store.scrub(1.0, 0).unwrap();
    assert_eq!(scrub.removed, 1);
    assert!(!hidden.exists());

    // Landing hashes: bytes that are not the declared object are never installed.
    let declared = ObjectRef::of(&[8u8; 5000]);
    let refused = store
        .put_stream(
            &mut [7u8; 5000].as_slice(),
            Some(&declared),
            &Fault::default(),
        )
        .unwrap_err();
    assert_eq!(refused.code, Code::OBJECT_ID_MISMATCH);
    assert!(!store.contains(&declared.sha256));

    fs::remove_dir_all(root).unwrap();
    fs::remove_dir_all(scratch).unwrap();
}

#[test]
fn a_new_handle_validates_and_trusts_on_one_connection() {
    let root = temporary("handle");
    let scratch = temporary("handle-scratch");
    fs::create_dir_all(&scratch).unwrap();
    Store::init(&root).unwrap();
    let object = put_elsewhere(&root, &scratch, &body(7));

    // What the Python `Store.open`/`Store.ensure` do: open, validate, then trust reads.
    let opened = connections_opened(&root);
    let store = Store::ensure(&root).unwrap();
    tensorfs_core::meta::Meta::open(&store).unwrap();
    for _ in 0..3 {
        store.record_valid(&object.sha256).unwrap();
    }
    assert_eq!(connections_opened(&root) - opened, 1);

    fs::remove_dir_all(root).unwrap();
    fs::remove_dir_all(scratch).unwrap();
}

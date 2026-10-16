use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::Duration,
};
use tensorfs_core::{
    err::Code,
    gc,
    ids::{Doc, ObjectRef},
    manifest::{Draft, Entry, Manifest},
    source_artifact,
    store::{Fault, Store},
};

fn directory(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tfs-import-{label}-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ))
}
fn owner() -> String {
    ObjectRef::of(b"local import").id()
}
fn tree() -> Manifest {
    Manifest::from_files(vec![
        ("a.json".into(), ObjectRef::of(b"first member")),
        ("nested/b.bin".into(), ObjectRef::of(b"second member")),
    ])
    .unwrap()
}
fn files(dir: &Path) -> Vec<(String, PathBuf)> {
    vec![
        ("a.json".into(), dir.join("a")),
        ("nested/b.bin".into(), dir.join("b")),
    ]
}

#[test]
fn refuses_roster_and_typed_header_before_starting_custody() {
    let dir = directory("roster");
    let store = Store::init(&dir).unwrap();
    let paths = files(&dir);
    for bad in [
        paths[..1].to_vec(),
        vec![paths[0].clone(), paths[0].clone()],
        vec![paths[0].clone(), ("extra".into(), dir.join("extra"))],
    ] {
        assert_eq!(
            source_artifact::import_tree(&store, &owner(), &tree(), &bad, &Fault::default())
                .unwrap_err()
                .code,
            Code::TRANSACTION_CONFLICT
        );
        assert!(source_artifact::read(&store, &owner()).unwrap().is_none());
    }
    let typed = Draft {
        entries: vec![(
            "model.ct".into(),
            Entry::CozyTensors(ObjectRef::of(b"header")),
        )],
    }
    .seal()
    .unwrap();
    assert_eq!(
        source_artifact::import_tree(&store, &owner(), &typed, &paths, &Fault::default())
            .unwrap_err()
            .code,
        Code::WRONG_TYPE
    );
    assert!(source_artifact::read(&store, &owner()).unwrap().is_none());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn verifies_bytes_and_resumes_only_missing_members() {
    let dir = directory("verify");
    let store = Store::init(&dir).unwrap();
    fs::write(dir.join("a"), b"first member").unwrap();
    fs::write(dir.join("b"), b"short").unwrap();
    assert_eq!(
        source_artifact::import_tree(&store, &owner(), &tree(), &files(&dir), &Fault::default())
            .unwrap_err()
            .code,
        Code::LENGTH_MISMATCH
    );
    fs::write(dir.join("b"), b"wrong! member").unwrap();
    assert_eq!(
        source_artifact::import_tree(&store, &owner(), &tree(), &files(&dir), &Fault::default())
            .unwrap_err()
            .code,
        Code::OBJECT_ID_MISMATCH
    );
    assert!(
        !source_artifact::read(&store, &owner())
            .unwrap()
            .unwrap()
            .complete
    );
    fs::remove_file(dir.join("a")).unwrap();
    gc::collect(&dir, false).unwrap();
    fs::write(dir.join("b"), b"second member").unwrap();
    let result =
        source_artifact::import_tree(&store, &owner(), &tree(), &files(&dir), &Fault::default())
            .unwrap();
    assert!(result.complete);
    assert_eq!(result.manifest, tree().object_ref());
    fs::remove_file(dir.join("b")).unwrap();
    assert_eq!(
        source_artifact::import_tree(&store, &owner(), &tree(), &files(&dir), &Fault::default())
            .unwrap(),
        result
    );
    let changed = Manifest::from_files(vec![("a.json".into(), ObjectRef::of(b"other"))]).unwrap();
    assert_eq!(
        source_artifact::import_tree(
            &store,
            &owner(),
            &changed,
            &files(&dir)[..1],
            &Fault::default()
        )
        .unwrap_err()
        .code,
        Code::TRANSACTION_CONFLICT
    );
    let recipient = ObjectRef::of(b"recipient").id();
    source_artifact::retain(&store, &owner(), &recipient).unwrap();
    source_artifact::release(&store, &owner()).unwrap();
    gc::collect(&dir, false).unwrap();
    assert_eq!(store.read_manifest(&result.manifest).unwrap(), tree());
    for (_, entry) in tree().entries() {
        assert!(store.contains(&entry.blob().sha256));
    }
    assert_eq!(
        source_artifact::import_tree(&store, &owner(), &tree(), &files(&dir), &Fault::default())
            .unwrap_err()
            .code,
        Code::TRANSACTION_CLOSED
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn import_child() {
    let Some(dir) = std::env::var_os("TFS_IMPORT_TREE_CHILD") else {
        return;
    };
    let dir = PathBuf::from(dir);
    let store = Store::open(&dir).unwrap();
    source_artifact::import_tree(
        &store,
        &owner(),
        &tree(),
        &files(&dir),
        &Fault {
            stage: Some("record".into()),
            ready: Some(dir.join("ready")),
        },
    )
    .unwrap();
    panic!("fault should wait for the parent to kill this process");
}

#[test]
fn symlink_member_is_refused_before_reading_its_target() {
    let dir = directory("symlink");
    let store = Store::init(&dir).unwrap();
    fs::write(dir.join("target"), b"first member").unwrap();
    std::os::unix::fs::symlink(dir.join("target"), dir.join("a")).unwrap();
    assert_eq!(
        source_artifact::import_tree(&store, &owner(), &tree(), &files(&dir), &Fault::default())
            .unwrap_err()
            .code,
        Code::WRONG_TYPE
    );
    assert!(!store.contains(&ObjectRef::of(b"first member").sha256));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn killed_after_admission_before_landed_keeps_member_live_for_gc_and_resume() {
    let dir = directory("kill");
    let store = Store::init(&dir).unwrap();
    fs::write(dir.join("a"), b"first member").unwrap();
    fs::write(dir.join("b"), b"second member").unwrap();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "import_child", "--nocapture"])
        .env("TFS_IMPORT_TREE_CHILD", &dir)
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    for _ in 0..1000 {
        if dir.join("ready").is_file() {
            break;
        }
        assert!(child.try_wait().unwrap().is_none());
        thread::sleep(Duration::from_millis(5));
    }
    assert!(dir.join("ready").is_file());
    child.kill().unwrap();
    child.wait().unwrap();
    let partial = source_artifact::read(&store, &owner()).unwrap().unwrap();
    assert!(!partial.complete);
    assert_eq!(partial.objects, vec![ObjectRef::of(b"first member")]);
    gc::collect(&dir, false).unwrap();
    fs::remove_file(dir.join("a")).unwrap();
    let completed =
        source_artifact::import_tree(&store, &owner(), &tree(), &files(&dir), &Fault::default())
            .unwrap();
    assert!(completed.complete);
    gc::collect(&dir, false).unwrap();
    assert_eq!(store.read_manifest(&completed.manifest).unwrap(), tree());
    fs::remove_dir_all(dir).unwrap();
}

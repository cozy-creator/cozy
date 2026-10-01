use std::{fs, path::PathBuf, thread, time::Duration};
use tensorfs_core::{
    err::Code,
    gc,
    ids::{Doc, ObjectRef},
    keyed_roots,
    manifest::{Draft, Entry, Manifest},
    store::Store,
};

const SPACE: &str = "encoder.t5-v1";

fn key(byte: char) -> String {
    byte.to_string().repeat(64)
}

struct Fixture {
    dir: PathBuf,
    store: Store,
}

impl Fixture {
    fn new(label: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "tfs-keyed-{label}-{}-{}",
            std::process::id(),
            tensorfs_core::meta::now_nanos_unique()
        ));
        let store = Store::init(&dir).unwrap();
        Self { dir, store }
    }

    /// One tree whose members hold `bodies`, with their local files written.
    fn tree(&self, bodies: &[&str]) -> (Manifest, Vec<(String, PathBuf)>) {
        let mut entries = vec![];
        let mut files = vec![];
        for body in bodies {
            let path = self.dir.join(format!("src-{body}"));
            fs::write(&path, body).unwrap();
            entries.push((format!("{body}.bin"), ObjectRef::of(body.as_bytes())));
            files.push((format!("{body}.bin"), path));
        }
        (Manifest::from_files(entries).unwrap(), files)
    }

    fn put(&self, key: &str, bodies: &[&str]) -> keyed_roots::KeyedRoot {
        let (tree, files) = self.tree(bodies);
        keyed_roots::put(&self.store, SPACE, key, &tree, &files).unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

#[test]
fn put_get_list_idempotent_and_conflicting() {
    let f = Fixture::new("put");
    let (tree, files) = f.tree(&["first", "second"]);
    let root = keyed_roots::put(&f.store, SPACE, &key('b'), &tree, &files).unwrap();
    assert_eq!(root.manifest, tree.object_ref());
    assert_eq!(root.bytes, tree.object_ref().length + 5 + 6);
    assert_eq!(
        keyed_roots::get(&f.store, SPACE, &key('b')).unwrap(),
        Some(root.clone())
    );

    // The same tree again is the standing root, needing no source bytes.
    for (_, path) in &files {
        fs::remove_file(path).unwrap();
    }
    thread::sleep(Duration::from_millis(2));
    assert_eq!(
        keyed_roots::put(&f.store, SPACE, &key('b'), &tree, &files).unwrap(),
        root
    );
    let (other, other_files) = f.tree(&["other"]);
    assert_eq!(
        keyed_roots::put(&f.store, SPACE, &key('b'), &other, &other_files)
            .unwrap_err()
            .code,
        Code::TRANSACTION_CONFLICT
    );

    let newer = keyed_roots::put(&f.store, SPACE, &key('a'), &other, &other_files).unwrap();
    assert!(newer.created_unix_ms > root.created_unix_ms);
    assert_eq!(
        keyed_roots::list(&f.store, SPACE).unwrap(),
        vec![root, newer]
    );
    assert!(keyed_roots::list(&f.store, "absent").unwrap().is_empty());
    assert_eq!(
        keyed_roots::get(&f.store, "absent", &key('a')).unwrap(),
        None
    );
}

#[test]
fn refuses_bad_spellings_and_typed_trees() {
    let f = Fixture::new("grammar");
    let (tree, files) = f.tree(&["body"]);
    for space in [
        "",
        "Encoder",
        ".hidden",
        "a/b",
        "-x",
        "a".repeat(65).as_str(),
    ] {
        assert_eq!(
            keyed_roots::put(&f.store, space, &key('a'), &tree, &files)
                .unwrap_err()
                .code,
            Code::KEY_GRAMMAR,
            "{space:?}"
        );
        assert_eq!(
            keyed_roots::list(&f.store, space).unwrap_err().code,
            Code::KEY_GRAMMAR
        );
    }
    for bad in [
        key('A'),
        key('a')[..63].to_string(),
        format!("sha256:{}", key('a')),
    ] {
        assert_eq!(
            keyed_roots::get(&f.store, SPACE, &bad).unwrap_err().code,
            Code::MALFORMED_DIGEST
        );
    }
    assert_eq!(
        keyed_roots::put(&f.store, SPACE, &key('a'), &tree, &files[..0])
            .unwrap_err()
            .code,
        Code::TRANSACTION_CONFLICT
    );
    let typed = Draft {
        entries: vec![("m.ct".into(), Entry::CozyTensors(ObjectRef::of(b"header")))],
    }
    .seal()
    .unwrap();
    assert_eq!(
        keyed_roots::put(&f.store, SPACE, &key('a'), &typed, &files)
            .unwrap_err()
            .code,
        Code::WRONG_TYPE
    );
    assert!(keyed_roots::list(&f.store, SPACE).unwrap().is_empty());
}

#[test]
fn gc_keeps_live_roots_and_reclaims_dropped_ones_leaving_nothing() {
    let f = Fixture::new("gc");
    let dropped = f.put(&key('1'), &["dropped one", "dropped two"]);
    let live = f.put(&key('2'), &["live"]);
    let present = |root: &keyed_roots::KeyedRoot, bodies: &[&str]| {
        f.store.manifest_path(&root.manifest.sha256).is_file()
            && bodies
                .iter()
                .all(|b| f.store.contains(&ObjectRef::of(b.as_bytes()).sha256))
    };
    assert_eq!(gc::collect(&f.dir, false).unwrap().reclaimed_bytes, 0);
    assert!(present(&dropped, &["dropped one", "dropped two"]));

    assert!(keyed_roots::remove(&f.store, SPACE, &key('1')).unwrap());
    assert!(!keyed_roots::remove(&f.store, SPACE, &key('1')).unwrap());
    let names: Vec<_> = fs::read_dir(f.dir.join("roots/keyed").join(SPACE))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    assert_eq!(names, vec![format!("{}.json", key('2'))]);

    let report = gc::collect(&f.dir, false).unwrap();
    assert_eq!(report.reclaimed_bytes, dropped.bytes);
    assert_eq!((report.reclaimed_blobs, report.reclaimed_manifests), (2, 1));
    assert!(!f.store.contains(&ObjectRef::of(b"dropped one").sha256));
    assert!(present(&live, &["live"]));

    let again = f.put(&key('1'), &["dropped one", "dropped two"]);
    assert_eq!(again.manifest, dropped.manifest);
    assert!(present(&again, &["dropped one", "dropped two"]));
}

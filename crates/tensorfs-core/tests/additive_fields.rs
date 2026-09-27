//! A document written by a newer TensorFS keeps working here: fields this build does not read
//! are carried through every read and rewrite unchanged, and GC keeps what they name.
use std::fs;
use std::path::PathBuf;

use tensorfs_core::canon;
use tensorfs_core::err::Code;
use tensorfs_core::gc;
use tensorfs_core::ids::{Doc, ObjectRef};
use tensorfs_core::manifest::Manifest;
use tensorfs_core::repository::{Mutation, Repository, RepositoryName};
use tensorfs_core::store::{Fault, Store};

fn temporary(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tensorfs-additive-{name}-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ))
}

fn blob(store: &Store, body: &[u8]) -> ObjectRef {
    let want = ObjectRef::of(body);
    store
        .put_stream(&mut &body[..], Some(&want), &Fault::default())
        .unwrap();
    want
}

fn object(r: &ObjectRef) -> String {
    format!(r#"{{"length":{},"sha256":"{}"}}"#, r.length, r.sha256)
}

/// Canonical bytes exactly as a newer writer would emit them: sorted keys, no whitespace.
fn newer_manifest(file: &ObjectRef, sidecar: &ObjectRef) -> Vec<u8> {
    let text = format!(
        r#"{{"entries":[{{"blob":{},"kind":"file","mode":"0755","path":"a.txt"}}],"sidecars":[{}]}}"#,
        object(file),
        object(sidecar)
    );
    assert_eq!(
        canon::write(&canon::parse(text.as_bytes(), 1 << 20).unwrap()),
        text.as_bytes()
    );
    text.into_bytes()
}

#[test]
fn a_newer_manifest_reads_keeps_its_identity_and_its_named_objects() {
    let root = temporary("manifest");
    let store = Store::init(&root).unwrap();
    let file = blob(&store, b"ordinary file");
    let sidecar = blob(&store, b"named only by a field this build does not read");
    let bytes = newer_manifest(&file, &sidecar);

    let manifest = Manifest::parse(&bytes).unwrap();
    assert_eq!(manifest.canonical_bytes(), bytes);
    assert_eq!(manifest.object_ref(), ObjectRef::of(&bytes));
    assert_eq!(manifest.unread_refs(), vec![sidecar.clone()]);
    let put = store.put_manifest(&manifest).unwrap();
    assert_eq!(put.obj, ObjectRef::of(&bytes));

    let repo = RepositoryName::new("acme", "model").unwrap();
    store
        .apply_repository(
            None,
            &Mutation::PutCheckpoint {
                repo: repo.clone(),
                manifest: put.obj.clone(),
            },
            &Fault::default(),
        )
        .unwrap();
    gc::collect(&root, false).unwrap();
    assert!(store.contains(&file.sha256));
    assert!(
        store.contains(&sidecar.sha256),
        "GC deleted an object a newer manifest names"
    );

    // Once nothing names it, the sidecar is ordinary garbage again.
    let observed = fs::read(root.join("repos/acme/model.json")).unwrap();
    store
        .apply_repository(
            Some(&observed),
            &Mutation::RemoveCheckpoint {
                repo,
                manifest_sha256: put.obj.sha256.clone(),
            },
            &Fault::default(),
        )
        .unwrap();
    gc::collect(&root, false).unwrap();
    assert!(!store.contains(&sidecar.sha256));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn a_newer_repository_survives_an_older_writers_mutation() {
    let root = temporary("repository");
    let store = Store::init(&root).unwrap();
    let first = store
        .put_manifest(&Manifest::from_files(vec![("a".into(), blob(&store, b"a"))]).unwrap())
        .unwrap()
        .obj;
    let second = store
        .put_manifest(&Manifest::from_files(vec![("b".into(), blob(&store, b"b"))]).unwrap())
        .unwrap()
        .obj;
    let pinned = blob(&store, b"pinned by a newer field");
    let text = format!(
        r#"{{"checkpoints":[{{"manifest":{},"note":"imported","published":true}}],"labels":{{"pin":{}}},"releases":[{{"channel":"beta","lanes":[{{"lane":"default","manifest":{},"precision":"bf16"}}],"revision":1,"version":"v1","yanked":false}}],"repo":{{"name":"model","org":"acme"}}}}"#,
        object(&first),
        object(&pinned),
        object(&first)
    );
    let parsed = Repository::parse(text.as_bytes()).unwrap();
    assert_eq!(parsed.canonical_bytes(), text.as_bytes());
    assert_eq!(parsed.unread_refs(), vec![pinned.clone()]);
    fs::create_dir_all(root.join("repos/acme")).unwrap();
    fs::write(root.join("repos/acme/model.json"), &text).unwrap();

    let repo = RepositoryName::new("acme", "model").unwrap();
    let next = store
        .apply_repository(
            Some(text.as_bytes()),
            &Mutation::PutCheckpoint {
                repo: repo.clone(),
                manifest: second.clone(),
            },
            &Fault::default(),
        )
        .unwrap()
        .unwrap();
    let written = String::from_utf8(next.canonical_bytes()).unwrap();
    for kept in [
        r#""labels":"#,
        r#""note":"imported""#,
        r#""channel":"beta""#,
        r#""precision":"bf16""#,
    ] {
        assert!(written.contains(kept), "{kept} dropped from {written}");
    }
    // Re-asserting a checkpoint it already holds is a no-op, whatever a newer writer added.
    let again = Repository::parse(written.as_bytes())
        .unwrap()
        .put_checkpoint(next.checkpoints[0].clone())
        .unwrap();
    assert_eq!(again.canonical_bytes(), written.as_bytes());

    gc::collect(&root, false).unwrap();
    assert!(store.contains(&pinned.sha256));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn an_unknown_entry_kind_is_carried_and_kept_and_refuses_only_to_materialize() {
    let root = temporary("opaque-entry");
    let store = Store::init(&root).unwrap();
    let file = blob(&store, b"ordinary");
    let link = blob(&store, b"target of an entry kind this build does not know");
    let text = format!(
        r#"{{"entries":[{{"blob":{},"kind":"file","path":"a.txt"}},{{"blob":{},"kind":"symlink","path":"b"}}]}}"#,
        object(&file),
        object(&link)
    );
    let manifest = Manifest::parse(text.as_bytes()).unwrap();
    assert_eq!(manifest.canonical_bytes(), text.as_bytes());
    let put = store.put_manifest(&manifest).unwrap();
    store
        .apply_repository(
            None,
            &Mutation::PutCheckpoint {
                repo: RepositoryName::new("acme", "opaque").unwrap(),
                manifest: put.obj.clone(),
            },
            &Fault::default(),
        )
        .unwrap();
    gc::collect(&root, false).unwrap();
    assert!(
        store.contains(&link.sha256),
        "GC deleted an opaque entry's object"
    );

    assert_eq!(
        tensorfs_core::project::ordinary_file(&manifest, "a.txt").unwrap(),
        &file
    );
    assert_eq!(
        tensorfs_core::project::ordinary_file(&manifest, "b")
            .unwrap_err()
            .code,
        Code::UNKNOWN_FIELD
    );
    let _ = fs::remove_dir_all(root);
}

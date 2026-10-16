use std::{
    fs,
    io::{Read, Write},
    net::TcpListener,
    process::{Command, Stdio},
    thread,
    time::Duration,
};
use tensorfs_core::{
    err::Code,
    gc,
    ids::{Doc, ObjectRef},
    manifest::Manifest,
    providers::{self, Provenance, Resolution, ResolvedMember},
    source_artifact,
    store::{Fault, Store},
    transport::{Anonymous, Deadline, SourcePolicy},
};
fn root(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "tfs-source-{name}-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ))
}
fn owner(label: &str) -> String {
    ObjectRef::of(label.as_bytes()).id()
}
fn tree(store: &Store) -> Manifest {
    let bytes = b"raw source bytes";
    let object = ObjectRef::of(bytes);
    store
        .put_stream(&mut bytes.as_slice(), Some(&object), &Fault::default())
        .unwrap();
    Manifest::from_files(vec![("model.safetensors".into(), object)]).unwrap()
}

#[test]
fn retained_tree_distinguishes_missing_corrupt_and_stale_verified_members() {
    let dir = root("member-verification");
    let store = Store::init(&dir).unwrap();
    let manifest = tree(&store);
    let member = manifest.entries()[0].1.blob();
    let producer = owner("producer");
    let recipient = owner("recipient");
    source_artifact::create(&store, &producer, &manifest).unwrap();
    source_artifact::retain(&store, &producer, &recipient).unwrap();
    let blob = store.object_path(&member.sha256);
    let saved = dir.join("temporarily-missing-member");
    fs::rename(&blob, &saved).unwrap();
    drop(store);

    let store = Store::open(&dir).unwrap();
    // Both a new recipient and a replayed recipient must expose the same missing object.
    for target in [&recipient, &owner("new-recipient")] {
        let error = source_artifact::retain(&store, &producer, target).unwrap_err();
        assert_eq!(error.code, Code::OBJECT_ABSENT, "{error}");
    }
    assert!(source_artifact::read(&store, &owner("new-recipient"))
        .unwrap()
        .is_none());

    // The restored inode is the one that was hashed: its record still stands.
    fs::rename(&saved, &blob).unwrap();
    assert!(store.record_valid(&member.sha256).is_ok());
    source_artifact::retain(&store, &producer, &recipient).unwrap();
    assert!(!store.open_verified_reporting(&member.sha256).unwrap().1);

    // Replacing the inode with the exact same content invalidates metadata, not identity.
    fs::rename(&blob, &saved).unwrap();
    fs::copy(&saved, &blob).unwrap();
    assert!(store.record_valid(&member.sha256).is_err());
    source_artifact::retain(&store, &producer, &recipient).unwrap();
    assert!(store.record_valid(&member.sha256).is_ok());
    fs::remove_file(saved).unwrap();

    // A changed object is a corruption refusal, never the missing-byte cache result.
    fs::remove_file(&blob).unwrap();
    fs::write(&blob, vec![0u8; member.length as usize]).unwrap();
    let error = source_artifact::retain(&store, &producer, &recipient).unwrap_err();
    assert_eq!(error.code, Code::OBJECT_CORRUPT, "{error}");
    assert!(!blob.exists());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn retained_tree_rejects_nonregular_members_and_false_lengths() {
    let dir = root("member-shape");
    let store = Store::init(&dir).unwrap();
    let manifest = tree(&store);
    let member = manifest.entries()[0].1.blob();
    let producer = owner("producer");
    source_artifact::create(&store, &producer, &manifest).unwrap();
    let incorrect = Manifest::from_files(vec![(
        "model.safetensors".into(),
        ObjectRef {
            sha256: member.sha256.clone(),
            length: member.length + 1,
        },
    )])
    .unwrap();
    let error = source_artifact::create(&store, &owner("wrong-length"), &incorrect).unwrap_err();
    assert_eq!(error.code, Code::DURABILITY_UNPROVEN, "{error}");
    let blob = store.object_path(&member.sha256);
    fs::remove_file(&blob).unwrap();
    fs::create_dir(&blob).unwrap();
    let error = source_artifact::retain(&store, &producer, &owner("nonregular")).unwrap_err();
    assert_eq!(error.code, Code::NOT_REGULAR_FILE, "{error}");
    fs::remove_dir_all(dir).unwrap();
}
fn resolution(base: &str, name: &str, bytes: &[u8]) -> Resolution {
    Resolution {
        canonical: base.into(),
        selection_sha256: "old accepted provenance preserved".into(),
        members: vec![ResolvedMember {
            member: name.into(),
            object: ObjectRef::of(bytes),
            url: format!("{base}/{name}"),
            provenance: Provenance::Declared,
            carrier: true,
            requires: vec![],
            companion: false,
        }],
    }
}
#[test]
fn content_ignores_origins_but_binds_names_exact_lengths_and_indexes() {
    let a = resolution("hf://a/model@revision", "model.safetensors", b"data");
    let b = resolution("civitai://42", "model.safetensors", b"data");
    assert_eq!(
        providers::content_manifest(&a).unwrap(),
        providers::content_manifest(&b).unwrap()
    );
    for mutation in 0..3 {
        let mut b = b.clone();
        match mutation {
            0 => b.members[0].member = "different.safetensors".into(),
            1 => b.members[0].object.length += 1,
            _ => b.members[0].object = ObjectRef::of(b"other"),
        };
        assert_ne!(
            providers::content_manifest(&a).unwrap(),
            providers::content_manifest(&b).unwrap()
        );
    }
    let mut missing = a.clone();
    missing.members[0].requires.push("missing-shard".into());
    assert!(providers::content_manifest(&missing).is_err());
}
#[test]
fn accepted_selection_moves_only_its_objects_and_replays_without_network() {
    let dir = root("selected");
    let store = Store::init(&dir).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let body = b"selected carrier only";
    let server = thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        let mut request = [0; 4096];
        let n = socket.read(&mut request).unwrap();
        let request = String::from_utf8_lossy(&request[..n]);
        assert!(request.starts_with("GET /chosen.safetensors "), "{request}");
        write!(
            socket,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .unwrap();
        socket.write_all(body).unwrap();
    });
    let selected = resolution(&base, "chosen.safetensors", body);
    let policy = SourcePolicy {
        allowed_hosts: vec!["127.0.0.1".into()],
        allow_local: true,
        max_redirects: 5,
        ..Default::default()
    };
    let (first, rows) = providers::materialize_selected(
        &store,
        &owner("producer"),
        &selected,
        &policy,
        &Anonymous,
        Deadline::none(),
        &|_, _| {},
    )
    .unwrap();
    assert_eq!(rows[0].transferred, body.len() as u64);
    assert!(first.complete);
    server.join().unwrap();
    let (again, rows) = providers::materialize_selected(
        &store,
        &owner("producer"),
        &selected,
        &policy,
        &Anonymous,
        Deadline::none(),
        &|_, _| {},
    )
    .unwrap();
    assert_eq!(first, again);
    assert!(rows[0].held);
    assert_eq!(rows[0].transferred, 0);
    source_artifact::retain(&store, &owner("producer"), &owner("recipient")).unwrap();
    source_artifact::release(&store, &owner("producer")).unwrap();
    gc::collect(&dir, false).unwrap();
    assert_eq!(
        store
            .read_range(&selected.members[0].object.sha256, 0, body.len() as u64)
            .unwrap(),
        body
    );
    source_artifact::release(&store, &owner("recipient")).unwrap();
    gc::collect(&dir, false).unwrap();
    assert!(!store.contains(&selected.members[0].object.sha256));
    assert!(source_artifact::retain(&store, &owner("producer"), &owner("late")).is_err());
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn owned_partial_objects_survive_gc_and_live_writer_is_fenced() {
    let dir = root("partial");
    let store = Store::init(&dir).unwrap();
    let tree = tree(&store);
    let id = owner("partial");
    let mut writer = source_artifact::Writer::open(&store, &id, tree.object_ref()).unwrap();
    writer.landed(tree.entries()[0].1.blob()).unwrap();
    assert!(source_artifact::Writer::open(&store, &id, tree.object_ref()).is_err());
    assert!(source_artifact::release(&store, &id).is_err());
    drop(writer);
    gc::collect(&dir, false).unwrap();
    assert!(store.contains(&tree.entries()[0].1.blob().sha256));
    assert!(source_artifact::retain(&store, &id, &owner("recipient")).is_err());
    source_artifact::create(&store, &id, &tree).unwrap();
    source_artifact::release(&store, &id).unwrap();
    assert!(source_artifact::create(&store, &id, &tree).is_err());
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn committed_child() {
    let Some(dir) = std::env::var_os("TFS_SOURCE_CHILD") else {
        return;
    };
    let store = Store::open(std::path::Path::new(&dir)).unwrap();
    let tree = tree(&store);
    source_artifact::create(&store, &owner("killed"), &tree).unwrap();
    fs::write(store.root().join("committed"), tree.object_ref().id()).unwrap();
    loop {
        thread::park();
    }
}
#[test]
fn killed_after_commit_before_reply_replays_and_adopts() {
    let dir = root("killed");
    Store::init(&dir).unwrap();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "committed_child", "--nocapture"])
        .env("TFS_SOURCE_CHILD", &dir)
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    for _ in 0..1000 {
        if dir.join("committed").is_file() {
            break;
        }
        assert!(child.try_wait().unwrap().is_none());
        thread::sleep(Duration::from_millis(5));
    }
    assert!(dir.join("committed").is_file());
    child.kill().unwrap();
    child.wait().unwrap();
    let store = Store::open(&dir).unwrap();
    gc::collect(&dir, false).unwrap();
    let first = source_artifact::read(&store, &owner("killed"))
        .unwrap()
        .unwrap();
    assert!(first.complete);
    let adopted = source_artifact::retain(&store, &owner("killed"), &owner("after-kill")).unwrap();
    assert_eq!(adopted.manifest, first.manifest);
    source_artifact::release(&store, &owner("killed")).unwrap();
    gc::collect(&dir, false).unwrap();
    let tree = store.read_manifest(&adopted.manifest).unwrap();
    assert!(store.contains(&tree.entries()[0].1.blob().sha256));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn recipient_adoption_and_producer_release_have_one_serializable_outcome() {
    use std::sync::{Arc, Barrier};
    for _ in 0..12 {
        let dir = root("adopt-race");
        let store = Store::init(&dir).unwrap();
        let tree = tree(&store);
        source_artifact::create(&store, &owner("producer"), &tree).unwrap();
        let barrier = Arc::new(Barrier::new(3));
        let acquire_store = store.clone();
        let acquire_barrier = barrier.clone();
        let acquire = thread::spawn(move || {
            acquire_barrier.wait();
            source_artifact::retain(&acquire_store, &owner("producer"), &owner("recipient"))
        });
        let release_store = store.clone();
        let release_barrier = barrier.clone();
        let release = thread::spawn(move || {
            release_barrier.wait();
            source_artifact::release(&release_store, &owner("producer"))
        });
        barrier.wait();
        let acquired = acquire.join().unwrap();
        release.join().unwrap().unwrap();
        gc::collect(&dir, false).unwrap();
        assert_eq!(
            store.contains(&tree.entries()[0].1.blob().sha256),
            acquired.is_ok()
        );
        fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn receipt_replay_refuses_another_producer_of_identical_bytes() {
    let dir = root("receipt-producer");
    let store = Store::init(&dir).unwrap();
    let tree = tree(&store);
    source_artifact::create(&store, &owner("first"), &tree).unwrap();
    source_artifact::create(&store, &owner("second"), &tree).unwrap();
    source_artifact::retain(&store, &owner("first"), &owner("recipient")).unwrap();
    let refusal =
        source_artifact::retain(&store, &owner("second"), &owner("recipient")).unwrap_err();
    assert_eq!(refusal.code, tensorfs_core::err::Code::CROSS_SUBJECT_REPLAY);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn recipient_cancel_before_adoption_leaves_a_permanent_exact_receipt_tombstone() {
    let dir = root("cancel-before-adopt");
    let store = Store::init(&dir).unwrap();
    let tree = tree(&store);
    let producer = source_artifact::create(&store, &owner("producer"), &tree).unwrap();
    let receipt = ObjectRef::of(&producer.receipt().unwrap()).id();
    source_artifact::release_retention(&store, &owner("producer"), &receipt, &owner("recipient"))
        .unwrap();
    let refusal =
        source_artifact::retain(&store, &owner("producer"), &owner("recipient")).unwrap_err();
    assert_eq!(refusal.code, tensorfs_core::err::Code::TRANSACTION_CLOSED);
    source_artifact::release(&store, &owner("producer")).unwrap();
    source_artifact::release_retention(&store, &owner("producer"), &receipt, &owner("recipient"))
        .unwrap();
    gc::collect(&dir, false).unwrap();
    assert!(!store.contains(&tree.entries()[0].1.blob().sha256));
    fs::remove_dir_all(dir).unwrap();
}

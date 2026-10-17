use super::*;
use std::sync::{Arc, Barrier};

fn fixture(label: &str) -> (std::path::PathBuf, Store, Meta, String) {
    let root = std::env::temp_dir().join(format!(
        "tensorfs-next-{label}-{}",
        crate::meta::now_nanos_unique()
    ));
    let store = Store::init(&root).unwrap();
    let meta = Meta::open(&store).unwrap();
    let id = crate::ids::object_id(label.as_bytes());
    (root, store, meta, id)
}

fn release(meta: &Meta, begun: Begin) {
    release_guards(meta, Some(begun.writer_hold), begun.source_leases);
}

#[test]
fn next_attempt_survives_a_large_old_epoch_fence_and_restart_with_progress() {
    let (root, store, meta, id) = fixture("restart");
    let old_epoch = 9_000_000_000_000;
    let declaration = tests::created_declaration();
    let first = begin(&store, &meta, &id, old_epoch, declaration.clone(), None).unwrap();
    let bytes = vec![0x51; 2048];
    add_part(
        &store,
        &meta,
        &id,
        old_epoch,
        ("model", "weight", "value"),
        &mut bytes.as_slice(),
    )
    .unwrap();
    let (_, head) =
        checkpoint_progress(&store, &meta, &id, old_epoch, "run", "output", None).unwrap();
    let roots_before = roots::read(&store, &id).unwrap().unwrap();
    assert!(fence(&meta, &id, old_epoch).unwrap());
    release(&meta, first);
    drop(meta);
    drop(store);

    let store = Store::open(&root).unwrap();
    let meta = Meta::open(&store).unwrap();
    let next = begin_next(&store, &meta, &id, declaration, head.head.as_ref()).unwrap();
    assert_eq!(next.writer_session_id, old_epoch + 1);
    assert!(
        !fence(&meta, &id, old_epoch).unwrap(),
        "a late old channel cannot fence its replacement"
    );
    check_writer(&meta, &id, next.writer_session_id).unwrap();
    assert_eq!(
        completed(&meta, &id, next.writer_session_id).unwrap().0,
        vec![("model".into(), "weight".into(), "value".into())]
    );
    let roots_after = roots::read(&store, &id).unwrap().unwrap();
    assert_eq!(roots_before.blobs, roots_after.blobs);
    assert_eq!(roots_before.sources, roots_after.sources);
    assert_eq!(roots_before.checkpoint, roots_after.checkpoint);
    let result = commit(&store, &meta, &id, next.writer_session_id).unwrap();
    assert_eq!(result.declaration.max_new_bytes, 2048);
    release(&meta, next.begun);
    assert_eq!(
        begin_next(&store, &meta, &id, tests::created_declaration(), None)
            .err()
            .unwrap()
            .code,
        Code::TRANSACTION_CLOSED
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn concurrent_next_attempts_allocate_distinct_durable_sessions() {
    let (root, store, meta, id) = fixture("concurrent");
    let peers = 4;
    let ready = Arc::new(Barrier::new(peers));
    let threads: Vec<_> = (0..peers)
        .map(|_| {
            let (root, id, ready) = (root.clone(), id.clone(), ready.clone());
            std::thread::spawn(move || {
                let store = Store::open(&root).unwrap();
                let meta = Meta::open(&store).unwrap();
                ready.wait();
                let next =
                    begin_next(&store, &meta, &id, tests::created_declaration(), None).unwrap();
                release(&meta, next.begun);
                next.writer_session_id
            })
        })
        .collect();
    let mut sessions: Vec<_> = threads
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .collect();
    sessions.sort_unstable();
    assert_eq!(sessions, vec![1, 2, 3, 4]);
    for old in &sessions[..3] {
        assert_eq!(
            check_writer(&meta, &id, *old).unwrap_err().code,
            Code::WRITER_FENCED
        );
        assert!(!fence(&meta, &id, *old).unwrap());
    }
    check_writer(&meta, &id, 4).unwrap();
    assert!(fence(&meta, &id, 4).unwrap());
    let after = begin_next(&store, &meta, &id, tests::created_declaration(), None).unwrap();
    assert_eq!(after.writer_session_id, 5);
    release(&meta, after.begun);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn next_attempt_preserves_conflicting_declarations_and_terminal_states() {
    let (root, store, meta, id) = fixture("conflict");
    let first = begin_next(&store, &meta, &id, tests::created_declaration(), None).unwrap();
    let before = meta.derived_row(&id).unwrap();
    let roots_before = roots::read(&store, &id).unwrap();
    let mut changed = tests::created_declaration();
    changed.max_new_bytes += 1;
    assert_eq!(
        begin_next(&store, &meta, &id, changed, None)
            .err()
            .unwrap()
            .code,
        Code::TRANSACTION_CONFLICT
    );
    assert_eq!(meta.derived_row(&id).unwrap(), before);
    assert_eq!(roots::read(&store, &id).unwrap(), roots_before);
    check_writer(&meta, &id, first.writer_session_id).unwrap();
    assert!(fence(&meta, &id, first.writer_session_id).unwrap());
    release(&meta, first.begun);
    assert!(abandon(&store, &meta, &id).unwrap().is_none());
    assert_eq!(
        begin_next(&store, &meta, &id, tests::created_declaration(), None)
            .err()
            .unwrap()
            .code,
        Code::TRANSACTION_CLOSED
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn next_attempt_honors_retained_restore_fences_and_refuses_session_exhaustion() {
    let (root, store, meta, id) = fixture("restore");
    let first = begin_next(&store, &meta, &id, tests::created_declaration(), None).unwrap();
    assert!(fence(&meta, &id, first.writer_session_id).unwrap());
    release(&meta, first.begun);
    // A restore can preserve a newer root fence than the recovered transaction row.
    let mut retained = roots::read(&store, &id).unwrap().unwrap();
    retained.restore_epoch = Some(999);
    roots::write(&store, &retained).unwrap();
    let next = begin_next(&store, &meta, &id, tests::created_declaration(), None).unwrap();
    assert_eq!(next.writer_session_id, 1000);
    assert!(fence(&meta, &id, next.writer_session_id).unwrap());
    release(&meta, next.begun);
    let last = begin(
        &store,
        &meta,
        &id,
        limits::INT_MAX as u64,
        tests::created_declaration(),
        None,
    )
    .unwrap();
    let before = meta.derived_row(&id).unwrap();
    let roots_before = roots::read(&store, &id).unwrap();
    assert_eq!(
        begin_next(&store, &meta, &id, tests::created_declaration(), None)
            .err()
            .unwrap()
            .code,
        Code::NUMBER_RANGE
    );
    assert_eq!(meta.derived_row(&id).unwrap(), before);
    assert_eq!(roots::read(&store, &id).unwrap(), roots_before);
    release(&meta, last);
    assert!(fence(&meta, &id, limits::INT_MAX as u64).unwrap());
    abandon(&store, &meta, &id).unwrap();
    assert_eq!(
        begin_next(&store, &meta, &id, tests::created_declaration(), None)
            .err()
            .unwrap()
            .code,
        Code::TRANSACTION_CLOSED
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn closing_an_old_native_channel_cannot_fence_its_replacement() {
    let (root, store, meta, id) = fixture("channel");
    let meta = Arc::new(meta);
    let first = channel::Writer::begin_next(
        &store,
        meta.clone(),
        &id,
        tests::created_declaration(),
        None,
    )
    .unwrap();
    let first_session = meta
        .derived_row(&id)
        .unwrap()
        .unwrap()
        .current_session
        .unwrap();
    let second = channel::Writer::begin_next(
        &store,
        meta.clone(),
        &id,
        tests::created_declaration(),
        None,
    )
    .unwrap();
    let second_session = meta
        .derived_row(&id)
        .unwrap()
        .unwrap()
        .current_session
        .unwrap();
    assert_eq!(second_session, first_session + 1);
    let (client, server) = cozy_platform::ipc::pair().unwrap();
    drop(client); // The stale executor disconnects after its replacement was admitted.
    channel::serve_derived(
        &first,
        server,
        "request",
        "model",
        None,
        &channel::Hooks {
            check_current: &|| Ok(()),
            record_checkpoint: &|_| panic!("closed peer checkpointed"),
            record_receipt: &|_| panic!("closed peer committed"),
            fetch: None,
        },
    )
    .unwrap();
    check_writer(&meta, &id, second_session).unwrap();
    assert!(
        matches!(lookup(&store, &meta, &id).unwrap(), Lookup::Open { writer_session: Some(value) } if value == second_session)
    );
    drop(second);
    drop(first);
    let _ = std::fs::remove_dir_all(root);
}

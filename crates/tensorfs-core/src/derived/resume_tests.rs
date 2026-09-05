use super::*;
use crate::ids::Doc;

fn fixture(label: &str) -> (Store, Meta, String, Declaration) {
    let root = std::env::temp_dir().join(format!(
        "tensorfs-derived-{label}-{}",
        crate::meta::now_nanos_unique()
    ));
    let store = Store::init(&root).unwrap();
    let meta = Meta::open(&store).unwrap();
    let mut declaration = super::tests::created_declaration();
    let tensor = &mut declaration.components[0].add[0];
    tensor.dtype = Dtype::Bf16;
    tensor.shape = vec![16, 32];
    tensor.encoding = registry::seeds()
        .into_iter()
        .find(|seed| seed.alias == "fp8-rowwise/1")
        .unwrap()
        .spec
        .object_id();
    tensor.parts = vec![
        PartDeclaration {
            role: "data".into(),
            dtype: Dtype::F8E4M3FN,
            shape: vec![16, 32],
        },
        PartDeclaration {
            role: "scale".into(),
            dtype: Dtype::F32,
            shape: vec![16],
        },
    ];
    declaration.configs = vec![ConfigDeclaration::Add {
        target: "model.json".into(),
    }];
    (
        store,
        meta,
        format!("sha256:{}", "71".repeat(32)),
        declaration,
    )
}

fn release(meta: &Meta, writer: Begin) {
    release_guards(meta, Some(writer.writer_hold), writer.source_leases);
}

fn data(store: &Store, meta: &Meta, id: &str, epoch: u64, byte: u8) -> crate::checkpoint::Written {
    add_part(
        store,
        meta,
        id,
        epoch,
        ("model", "weight", "data"),
        &mut vec![byte; 512].as_slice(),
    )
    .unwrap()
}

fn finish(store: &Store, meta: &Meta, id: &str, epoch: u64) -> ReceiptFacts {
    add_part(
        store,
        meta,
        id,
        epoch,
        ("model", "weight", "scale"),
        &mut vec![0; 64].as_slice(),
    )
    .unwrap();
    add_config(meta, id, epoch, "model.json", &mut b"{\"a\":1}".as_slice()).unwrap();
    commit(store, meta, id, epoch).unwrap()
}

/// The same bounded page/read/admit path used by the host; no SQLite or private root is copied.
fn transfer(from: &Store, into: &Store, head: &ObjectRef) {
    let mut cursor = Some(head.clone());
    while let Some(object) = cursor {
        let mut bytes = Vec::new();
        from.read_into(&object.sha256, &mut bytes).unwrap();
        into.put_stream(&mut bytes.as_slice(), Some(&object), &Default::default())
            .unwrap();
        let mut offset = 0;
        loop {
            let page = crate::durability::local_window(from, &object, offset, 1).unwrap();
            assert!(page.objects.len() <= 1);
            for (kind, object) in page.objects {
                assert_eq!(kind, crate::repo_cache::CacheKind::Blob);
                let mut bytes = Vec::new();
                from.read_into(&object.sha256, &mut bytes).unwrap();
                into.put_stream(&mut bytes.as_slice(), Some(&object), &Default::default())
                    .unwrap();
            }
            cursor = page.link.prev;
            if let Some(next) = page.next {
                offset = next;
            } else {
                break;
            }
        }
    }
}

#[test]
fn role_interruption_epoch_fence_gc_and_empty_store_recovery_match_clean_commit() {
    let (store, meta, id, declaration) = fixture("resume");
    let first = begin(&store, &meta, &id, 7, declaration.clone(), None).unwrap();
    let payload = data(&store, &meta, &id, 7, 0x31).part.segments()[0].clone();
    add_config(&meta, &id, 7, "model.json", &mut b"{\"a\":1}".as_slice()).unwrap();
    let (plan, exported) =
        checkpoint_progress(&store, &meta, &id, 7, "request", "output", None).unwrap();
    let head = exported.head.as_ref().unwrap();
    assert_eq!(
        commit(&store, &meta, &id, 7).unwrap_err().code,
        Code::ARTIFACT_INCOMPLETE
    );
    assert!(fence(&meta, &id, 7).unwrap());
    release(&meta, first);
    assert_eq!(
        check_writer(&meta, &id, 7).unwrap_err().code,
        Code::WRITER_FENCED
    );

    // GC reads the private files even while the derived SQL row is undecodable.
    let db = rusqlite::Connection::open(crate::catalog::Catalog::path(store.root())).unwrap();
    let saved: Vec<u8> = db
        .query_row(
            "SELECT bytes FROM tensorfs_derived_transactions WHERE id=?1",
            [&id],
            |row| row.get(0),
        )
        .unwrap();
    db.execute(
        "UPDATE tensorfs_derived_transactions SET bytes=?1 WHERE id=?2",
        rusqlite::params![b"invalid".as_slice(), &id],
    )
    .unwrap();
    crate::gc::collect(store.root(), false).unwrap();
    assert!(store.contains(&payload.sha256) && store.contains(&head.sha256));
    db.execute(
        "UPDATE tensorfs_derived_transactions SET bytes=?1 WHERE id=?2",
        rusqlite::params![saved, &id],
    )
    .unwrap();
    drop(db);

    let second = begin(&store, &meta, &id, 8, declaration.clone(), None).unwrap();
    let accepted = completed(&meta, &id, 8).unwrap();
    assert_eq!(
        accepted.0,
        vec![("model".into(), "weight".into(), "data".into())]
    );
    assert_eq!(accepted.1, vec!["model.json"]);
    data(&store, &meta, &id, 8, 0x31);
    assert_eq!(
        add_part(
            &store,
            &meta,
            &id,
            8,
            ("model", "weight", "data"),
            &mut vec![0x32; 512].as_slice()
        )
        .unwrap_err()
        .code,
        Code::TRANSACTION_CONFLICT
    );
    assert_eq!(
        add_config(&meta, &id, 8, "model.json", &mut b"{\"a\":2}".as_slice())
            .unwrap_err()
            .code,
        Code::TRANSACTION_CONFLICT
    );
    assert_eq!(
        checkpoint_progress(&store, &meta, &id, 8, "request", "output", None).unwrap(),
        (plan.clone(), exported.clone())
    );

    let replacement = Store::init(&store.root().join("replacement")).unwrap();
    transfer(&store, &replacement, head);
    let new_meta = Meta::open(&replacement).unwrap();
    assert!(matches!(
        lookup(&replacement, &new_meta, &id).unwrap(),
        Lookup::Absent
    ));
    let canonical = declaration.clone().canonical_work_bytes().unwrap();
    validate_checkpoint(
        &replacement,
        &new_meta,
        &id,
        &canonical,
        head,
        "request",
        "output",
        &plan,
    )
    .unwrap();
    assert!(new_meta.derived_rows().unwrap().is_empty());
    assert!(roots::read(&replacement, &id).unwrap().is_none());
    let restored = begin(
        &replacement,
        &new_meta,
        &id,
        1,
        declaration.clone(),
        Some(head),
    )
    .unwrap();
    assert_eq!(completed(&new_meta, &id, 1).unwrap(), accepted);
    assert_eq!(
        checkpoint_progress(&replacement, &new_meta, &id, 1, "request", "output", None).unwrap(),
        (plan, exported.clone())
    );
    let recovered = finish(&replacement, &new_meta, &id, 1);
    let local = finish(&store, &meta, &id, 8);
    assert_eq!(recovered.manifest, local.manifest);
    assert_eq!(recovered.header, local.header);
    release(&new_meta, restored);
    release(&meta, second);
    dispose(&replacement, &new_meta, &id).unwrap();
    dispose(&store, &meta, &id).unwrap();
    crate::gc::collect(store.root(), false).unwrap();
    assert!(!store.contains(&payload.sha256) && !store.contains(&head.sha256));
    std::fs::remove_dir_all(store.root()).unwrap();
}

#[test]
fn checkpoint_cursor_extends_without_ack_and_refuses_forks_or_other_slots() {
    let (store, meta, id, declaration) = fixture("cursor");
    let writer = begin(&store, &meta, &id, 1, declaration, None).unwrap();
    data(&store, &meta, &id, 1, 0x31);
    let (_, first) = checkpoint_progress(&store, &meta, &id, 1, "request", "output", None).unwrap();
    add_config(&meta, &id, 1, "model.json", &mut b"{\"a\":1}".as_slice()).unwrap();
    let (plan, second) =
        checkpoint_progress(&store, &meta, &id, 1, "request", "output", None).unwrap();
    assert!(second.links > first.links);
    assert_eq!(
        checkpoint_progress(
            &store,
            &meta,
            &id,
            1,
            "request",
            "output",
            first.head.as_ref()
        )
        .unwrap()
        .1,
        second
    );
    assert_eq!(
        checkpoint_progress(&store, &meta, &id, 1, "request", "other", None)
            .unwrap_err()
            .code,
        Code::CROSS_SUBJECT_REPLAY
    );
    let fork = crate::durability::checkpoint_bytes(
        &store,
        &crate::durability::Policy {
            chain: "request/output",
            operation: &id,
            plan: &plan,
            interval: crate::durability::INTERVAL_BYTES,
        },
        None,
        b"fork",
        Vec::new(),
        None,
    )
    .unwrap()
    .head;
    assert_eq!(
        checkpoint_progress(
            &store,
            &meta,
            &id,
            1,
            "request",
            "output",
            fork.head.as_ref()
        )
        .unwrap_err()
        .code,
        Code::TRANSACTION_CONFLICT
    );
    fence(&meta, &id, 1).unwrap();
    release(&meta, writer);
    abandon(&store, &meta, &id).unwrap();
    crate::gc::collect(store.root(), false).unwrap();
    assert!(!store.contains(&second.head.unwrap().sha256));
    std::fs::remove_dir_all(store.root()).unwrap();
}

#[test]
fn missing_corrupt_and_changed_work_cannot_be_reused() {
    for corrupt in [false, true] {
        let (store, meta, id, declaration) = fixture("invalid");
        let first = begin(&store, &meta, &id, 1, declaration.clone(), None).unwrap();
        let payload = data(&store, &meta, &id, 1, 0x31).part.segments()[0].clone();
        let (_, checkpoint) =
            checkpoint_progress(&store, &meta, &id, 1, "request", "output", None).unwrap();
        fence(&meta, &id, 1).unwrap();
        release(&meta, first);
        if corrupt {
            std::fs::remove_file(store.blob_path(&payload.sha256)).unwrap();
            std::fs::write(store.blob_path(&payload.sha256), vec![0x99; 512]).unwrap();
        } else {
            std::fs::remove_file(store.blob_path(&payload.sha256)).unwrap();
        }
        let mut changed = declaration.clone();
        changed.work_fingerprint = Some(format!("sha256:{}", "45".repeat(32)));
        assert_eq!(
            begin(&store, &meta, &id, 2, changed, checkpoint.head.as_ref())
                .err()
                .unwrap()
                .code,
            Code::TRANSACTION_CONFLICT
        );
        let mut missing = declaration.clone();
        missing.work_fingerprint = None;
        assert_eq!(
            begin(&store, &meta, &id, 2, missing, None)
                .err()
                .unwrap()
                .code,
            Code::MISSING_FIELD
        );
        let canonical = declaration.clone().canonical_work_bytes().unwrap();
        let head = checkpoint.head.as_ref().unwrap();
        let link = crate::durability::local_link(&store, head).unwrap();
        assert_eq!(
            validate_checkpoint(
                &store, &meta, &id, &canonical, head, "request", "output", &link.plan
            )
            .unwrap_err()
            .code,
            Code::OBJECT_CORRUPT
        );
        let resumed = begin(&store, &meta, &id, 2, declaration, checkpoint.head.as_ref()).unwrap();
        assert!(completed(&meta, &id, 2).unwrap().0.is_empty());
        assert_eq!(
            commit(&store, &meta, &id, 2).unwrap_err().code,
            Code::ARTIFACT_INCOMPLETE
        );
        fence(&meta, &id, 2).unwrap();
        release(&meta, resumed);
        abandon(&store, &meta, &id).unwrap();
        std::fs::remove_dir_all(store.root()).unwrap();
    }
}

#[test]
fn recovery_refuses_old_writer_authority_and_malformed_part_geometry() {
    let (store, meta, id, declaration) = fixture("authority");
    let writer = begin(&store, &meta, &id, 9, declaration.clone(), None).unwrap();
    data(&store, &meta, &id, 9, 0x31);
    let row = meta.derived_rows().unwrap().remove(0);
    let plan = progress_plan(&row);
    let mut geometry = progress_row(&row);
    geometry.added_parts[0].part.shape = vec![16, 31];
    for snapshot in [row, geometry] {
        let bytes = crate::canon::write(&snapshot.to_value());
        let head = crate::durability::checkpoint_bytes(
            &store,
            &crate::durability::Policy {
                chain: "request/output",
                operation: &id,
                plan: &plan,
                interval: crate::durability::INTERVAL_BYTES,
            },
            None,
            &bytes,
            Vec::new(),
            None,
        )
        .unwrap()
        .head
        .head
        .unwrap();
        assert!(begin(&store, &meta, &id, 10, declaration.clone(), Some(&head)).is_err());
        check_writer(&meta, &id, 9).unwrap();
    }
    fence(&meta, &id, 9).unwrap();
    release(&meta, writer);
    abandon(&store, &meta, &id).unwrap();
    std::fs::remove_dir_all(store.root()).unwrap();
}

#[test]
fn source_identity_is_locked_and_source_closure_stays_rooted_after_writer_death() {
    use crate::repository::{Mutation, ReleaseLane, RepositoryName};
    let (store, meta, id, mut declaration) = fixture("sources");
    let mut sources = Vec::new();
    for byte in [0x72, 0x73] {
        let source_id = format!("sha256:{}", format!("{byte:02x}").repeat(32));
        let writer = begin(
            &store,
            &meta,
            &source_id,
            1,
            super::tests::created_declaration(),
            None,
        )
        .unwrap();
        let payload = add_part(
            &store,
            &meta,
            &source_id,
            1,
            ("model", "weight", "value"),
            &mut vec![byte; 2048].as_slice(),
        )
        .unwrap()
        .part
        .segments()[0]
            .clone();
        let receipt = commit(&store, &meta, &source_id, 1).unwrap();
        release(&meta, writer);
        let repo = RepositoryName::new("fixture", format!("source-{byte}")).unwrap();
        let checkpointed = store
            .apply_repository(
                None,
                &Mutation::PutCheckpoint {
                    repo: repo.clone(),
                    manifest: receipt.manifest.clone(),
                },
                &Default::default(),
            )
            .unwrap()
            .unwrap();
        let document = store
            .apply_repository(
                Some(&checkpointed.canonical_bytes()),
                &Mutation::UpdateRelease {
                    expected_revision: 0,
                    repo: repo.clone(),
                    remove: Vec::new(),
                    set: vec![ReleaseLane {
                        lane: "main".into(),
                        manifest: receipt.manifest.clone(),
                    }],
                    version: "1.0.0".into(),
                },
                &Default::default(),
            )
            .unwrap()
            .unwrap();
        dispose(&store, &meta, &source_id).unwrap();
        sources.push((receipt.manifest, payload, repo, document));
    }
    declaration.sources = vec![Source {
        alias: "base".into(),
        manifest: sources[0].0.clone(),
    }];
    declaration.components[0].source = Some("base".into());
    declaration.components[0].source_component = Some("model".into());
    declaration.components[0].drop = vec!["weight".into()];
    let writer = begin(&store, &meta, &id, 1, declaration.clone(), None).unwrap();
    data(&store, &meta, &id, 1, 0x31);
    let (_, checkpoint) =
        checkpoint_progress(&store, &meta, &id, 1, "request", "output", None).unwrap();
    fence(&meta, &id, 1).unwrap();
    release(&meta, writer);
    declaration.sources[0].manifest = sources[1].0.clone();
    assert_eq!(
        begin(&store, &meta, &id, 2, declaration, checkpoint.head.as_ref())
            .err()
            .unwrap()
            .code,
        Code::TRANSACTION_CONFLICT
    );
    let (manifest, payload, repo, document) = &sources[0];
    store
        .apply_repository(
            Some(&document.canonical_bytes()),
            &Mutation::DeleteRepository { repo: repo.clone() },
            &Default::default(),
        )
        .unwrap();
    crate::gc::collect(store.root(), false).unwrap();
    store.read_manifest(manifest).unwrap();
    assert!(store.contains(&payload.sha256));
    abandon(&store, &meta, &id).unwrap();
    crate::gc::collect(store.root(), false).unwrap();
    assert!(!store.contains(&payload.sha256));
    std::fs::remove_dir_all(store.root()).unwrap();
}

#[test]
fn wrong_document_lengths_refuse_before_parsing_or_importing_work() {
    let (store, meta, id, declaration) = fixture("lengths");
    let writer = begin(&store, &meta, &id, 1, declaration.clone(), None).unwrap();
    data(&store, &meta, &id, 1, 0x31);
    let (plan, checkpoint) =
        checkpoint_progress(&store, &meta, &id, 1, "request", "output", None).unwrap();
    let head = checkpoint.head.unwrap();
    let mut short_head = head.clone();
    short_head.length = 1;
    assert_eq!(
        crate::durability::local_link(&store, &short_head)
            .unwrap_err()
            .code,
        Code::LENGTH_MISMATCH
    );
    let mut link = crate::durability::local_link(&store, &head).unwrap();
    let progress = link.progress.as_mut().unwrap();
    progress.length = 1;
    for object in &mut link.blobs {
        if object.sha256 == progress.sha256 {
            object.length = 1;
        }
    }
    let bytes = link.canonical_bytes();
    let forged = store
        .put_stream(
            &mut bytes.as_slice(),
            Some(&ObjectRef::of(&bytes)),
            &Default::default(),
        )
        .unwrap()
        .obj;
    let canonical = declaration.clone().canonical_work_bytes().unwrap();
    assert_eq!(
        validate_checkpoint(&store, &meta, &id, &canonical, &forged, "request", "output", &plan)
            .unwrap_err()
            .code,
        Code::LENGTH_MISMATCH
    );
    assert_eq!(
        begin(&store, &meta, &id, 2, declaration, Some(&forged))
            .err()
            .unwrap()
            .code,
        Code::LENGTH_MISMATCH
    );
    check_writer(&meta, &id, 1).unwrap();
    fence(&meta, &id, 1).unwrap();
    release(&meta, writer);
    abandon(&store, &meta, &id).unwrap();
    std::fs::remove_dir_all(store.root()).unwrap();
}

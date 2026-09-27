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
            source: None,
            role: "data".into(),
            dtype: Dtype::F8E4M3FN,
            shape: vec![16, 32],
        },
        PartDeclaration {
            source: None,
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
fn recovery_discards_old_writer_authority_and_malformed_part_geometry() {
    let (store, meta, id, declaration) = fixture("authority");
    let writer = begin(&store, &meta, &id, 9, declaration.clone(), None).unwrap();
    data(&store, &meta, &id, 9, 0x31);
    let row = meta.derived_rows().unwrap().remove(0);
    let plan = progress_plan(&row);
    let mut geometry = progress_row(&row);
    geometry.added_parts[0].part.shape = vec![16, 31];
    let mut writers = vec![writer];
    for (epoch, snapshot) in [(10, row), (11, geometry)] {
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
        // The checkpoint cannot be continued: it is discarded, nothing of it is imported,
        // and the new writer proceeds from the work this Store already holds.
        writers.push(begin(&store, &meta, &id, epoch, declaration.clone(), Some(&head)).unwrap());
        check_writer(&meta, &id, epoch).unwrap();
        let current = meta.derived_rows().unwrap().remove(0);
        assert!(current
            .added_parts
            .iter()
            .all(|added| added.part.shape != vec![16, 31]));
    }
    fence(&meta, &id, 11).unwrap();
    for writer in writers {
        release(&meta, writer);
    }
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
                        extra: Default::default(),
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
fn wrong_document_lengths_refuse_to_validate_and_are_discarded_on_resume() {
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
    // A forged checkpoint is discarded, not imported; this begin then meets the live writer.
    // Resuming from it discards it instead of refusing: the next writer proceeds.
    let next = begin(&store, &meta, &id, 2, declaration, Some(&forged)).unwrap();
    check_writer(&meta, &id, 2).unwrap();
    fence(&meta, &id, 2).unwrap();
    release(&meta, writer);
    release(&meta, next);
    abandon(&store, &meta, &id).unwrap();
    std::fs::remove_dir_all(store.root()).unwrap();
}

#[test]
fn restored_inherited_source_custody_survives_origin_release_and_writer_restart() {
    let (store, meta, id, mut declaration) = fixture("inherited-restart");
    let source_id = format!("sha256:{}", "75".repeat(32));
    let source_writer = begin(
        &store,
        &meta,
        &source_id,
        1,
        super::tests::created_declaration(),
        None,
    )
    .unwrap();
    let source_part = add_part(
        &store,
        &meta,
        &source_id,
        1,
        ("model", "weight", "value"),
        &mut vec![0x61; 2048].as_slice(),
    )
    .unwrap();
    let source = commit(&store, &meta, &source_id, 1).unwrap();
    release(&meta, source_writer);
    declaration.sources.push(Source {
        alias: "base".into(),
        manifest: source.manifest.clone(),
    });
    declaration.components.push(ComponentDeclaration {
        target: "retained".into(),
        source: Some("base".into()),
        source_component: Some("model".into()),
        drop: Vec::new(),
        add: Vec::new(),
    });
    declaration.order.push(("retained".into(), "weight".into()));
    let writer = begin(&store, &meta, &id, 11, declaration.clone(), None).unwrap();
    let added = data(&store, &meta, &id, 11, 0x31).part.segments()[0].clone();
    add_config(&meta, &id, 11, "model.json", &mut b"{\"a\":1}".as_slice()).unwrap();
    let (plan, exported) =
        checkpoint_progress(&store, &meta, &id, 11, "request", "output", None).unwrap();
    let head = exported.head.unwrap();
    let chain = crate::durability::local_chain(&store, &head, &plan).unwrap();
    // Inputs stay under source custody; a role checkpoint exports only its added payload
    // and native progress. It does not force another upload of the source closure.
    assert!(chain.blobs.contains(&added));
    for object in source_part.part.segments() {
        assert!(!chain.blobs.contains(object));
    }
    assert!(chain.manifests.is_empty());
    fence(&meta, &id, 11).unwrap();
    release(&meta, writer);
    dispose(&store, &meta, &source_id).unwrap();
    crate::gc::collect(store.root(), false).unwrap();

    let (restored, restored_meta, _, _) = fixture("inherited-empty");
    transfer(&store, &restored, &head);
    let canonical = declaration.canonical_work_bytes().unwrap();
    let validate = || {
        validate_checkpoint(
            &restored,
            &restored_meta,
            &id,
            &canonical,
            &head,
            "request",
            "output",
            &plan,
        )
    };
    assert_eq!(validate().unwrap_err().code, Code::ROOT_ABSENT);
    assert!(matches!(
        lookup(&restored, &restored_meta, &id).unwrap(),
        Lookup::Absent
    ));

    // Ordinary exact source acquisition into the new Store, without copying SQLite,
    // private derived roots, a receipt, or the original writer's authority.
    let manifest = checkpoint::load_manifest(&store, &source.manifest).unwrap();
    let walk = checkpoint::walk(&store, &manifest).unwrap();
    for reached in &walk.objects {
        let mut bytes = Vec::new();
        store.read_into(&reached.obj.sha256, &mut bytes).unwrap();
        restored
            .put_stream(
                &mut bytes.as_slice(),
                Some(&reached.obj),
                &Default::default(),
            )
            .unwrap();
    }
    restored.put_manifest(&manifest).unwrap();
    // Resident bytes alone never self-authorize begin or read-only validation.
    assert_eq!(validate().unwrap_err().code, Code::ROOT_ABSENT);
    assert_eq!(
        begin(
            &restored,
            &restored_meta,
            &id,
            1,
            declaration.clone(),
            Some(&head)
        )
        .err()
        .unwrap()
        .code,
        Code::ROOT_ABSENT
    );
    assert!(roots::read(&restored, &id).unwrap().is_none());

    let operation = "restored-source";
    let request = format!("sha256:{}", "76".repeat(32));
    let catalog = crate::catalog::Catalog::open(restored.root()).unwrap();
    let crate::catalog::SourcePreparation::Open(source_guard) = catalog
        .begin_source_preparation(operation, &request)
        .unwrap()
    else {
        panic!("already prepared")
    };
    crate::ingest::transaction::ensure_session_root(restored.root(), operation, "_tensorfs")
        .unwrap();
    crate::ingest::transaction::name_candidates(
        restored.root(),
        operation,
        std::slice::from_ref(&source.manifest),
    )
    .unwrap();
    catalog
        .commit_source_preparation(
            operation,
            &request,
            &[crate::catalog::SourcePreparationResult {
                slot: "source".into(),
                profile: "proof".into(),
                manifest_sha256: source.manifest.sha256.clone(),
                manifest_length: source.manifest.length,
            }],
        )
        .unwrap();
    drop(source_guard);
    validate().unwrap();
    assert!(matches!(
        lookup(&restored, &restored_meta, &id).unwrap(),
        Lookup::Absent
    ));
    let first = begin(
        &restored,
        &restored_meta,
        &id,
        1,
        declaration.clone(),
        Some(&head),
    )
    .unwrap();
    assert_eq!(
        completed(&restored_meta, &id, 1).unwrap(),
        (
            vec![("model".into(), "weight".into(), "data".into())],
            vec!["model.json".into()]
        )
    );
    crate::ingest::source::release_model_source(&restored, operation).unwrap();
    assert!(catalog.model_source_operations().unwrap().is_empty());
    fence(&restored_meta, &id, 1).unwrap();
    release(&restored_meta, first);
    crate::gc::collect(restored.root(), false).unwrap();
    validate().unwrap();
    let next = begin(&restored, &restored_meta, &id, 2, declaration, None).unwrap();
    let (_, replay) =
        checkpoint_progress(&restored, &restored_meta, &id, 2, "request", "output", None).unwrap();
    assert_eq!(replay.head.as_ref(), Some(&head));
    let receipt = finish(&restored, &restored_meta, &id, 2);
    assert_eq!(receipt.inherited_payload_objects, 1);
    assert_eq!(receipt.inherited_payload_bytes, 2048);
    assert_eq!(receipt.added_objects, vec![added.clone()]);
    release(&restored_meta, next);
    adopt(&restored, &restored_meta, &id, "restored-output").unwrap();
    crate::gc::collect(restored.root(), false).unwrap();
    checkpoint::walk(
        &restored,
        &checkpoint::load_manifest(&restored, &receipt.manifest).unwrap(),
    )
    .unwrap()
    .require_resident(&restored)
    .unwrap();
    dispose(&restored, &restored_meta, &id).unwrap();
    crate::gc::collect(restored.root(), false).unwrap();
    assert!(!restored.contains(&added.sha256));
    assert!(!restored.contains(&source_part.part.segments()[0].sha256));
    abandon(&store, &meta, &id).unwrap();
    std::fs::remove_dir_all(store.root()).unwrap();
    std::fs::remove_dir_all(restored.root()).unwrap();
}

#[test]
fn per_tensor_checkpoints_retain_linear_metadata() {
    let (store, meta, id, mut declaration) = fixture("linear-progress");
    let prototype = declaration.components[0].add[0].clone();
    const COUNT: usize = 128;
    declaration.components[0].add = (0..COUNT)
        .map(|index| {
            let mut tensor = prototype.clone();
            tensor.key = format!("weight_{index:04}");
            tensor
        })
        .collect();
    declaration.order = declaration.components[0]
        .add
        .iter()
        .map(|tensor| ("model".into(), tensor.key.clone()))
        .collect();
    declaration.max_new_bytes = (COUNT * 576) as u64;
    let writer = begin(&store, &meta, &id, 1, declaration.clone(), None).unwrap();
    let context_bytes =
        crate::canon::write(&progress::context(&meta.derived_rows().unwrap()[0]).to_value()).len()
            as u64;
    let mut samples = Vec::new();
    let mut last = None;
    for index in 0..COUNT {
        let key = format!("weight_{index:04}");
        add_part(
            &store,
            &meta,
            &id,
            1,
            ("model", &key, "data"),
            &mut vec![0x31; 512].as_slice(),
        )
        .unwrap();
        add_part(
            &store,
            &meta,
            &id,
            1,
            ("model", &key, "scale"),
            &mut vec![0; 64].as_slice(),
        )
        .unwrap();
        let (plan, head) =
            checkpoint_progress(&store, &meta, &id, 1, "request", "model", None).unwrap();
        if [32, 64, 128].contains(&(index + 1)) {
            let root = head.head.as_ref().unwrap();
            let chain = crate::durability::local_chain(&store, root, &plan).unwrap();
            assert_eq!(chain.progress.len(), index + 1);
            let mut refs = std::collections::BTreeMap::new();
            for object in chain.blobs.iter().chain(
                crate::durability::local_documents(&store, root)
                    .unwrap()
                    .iter(),
            ) {
                refs.insert(object.sha256.clone(), object.length);
            }
            refs.remove(&ObjectRef::of(&vec![0x31; 512]).sha256);
            let metadata: u64 = refs.values().sum();
            assert!(
                metadata < context_bytes + ((index + 1) * 2048) as u64,
                "cumulative snapshots returned: tensors={} metadata={metadata}",
                index + 1
            );
            samples.push(metadata);
            let restored = read_progress(
                &store,
                root,
                &id,
                &String::from_utf8(declaration.clone().canonical_work_bytes().unwrap()).unwrap(),
                &[],
            )
            .unwrap();
            assert_eq!(restored.added_parts.len(), (index + 1) * 2);
            eprintln!("derived metadata: tensors={} retained_bytes={metadata} context_bytes={context_bytes}",index+1);
        }
        last = Some((plan, head));
    }
    assert!(samples[2] - samples[1] <= 21 * (samples[1] - samples[0]) / 10);
    assert_eq!(
        checkpoint_progress(&store, &meta, &id, 1, "request", "model", None).unwrap(),
        last.unwrap()
    );
    fence(&meta, &id, 1).unwrap();
    release(&meta, writer);
    abandon(&store, &meta, &id).unwrap();
    std::fs::remove_dir_all(store.root()).unwrap();
}

#[test]
fn source_free_restore_holds_survive_gc_until_begin_or_explicit_abandon() {
    let (source, source_meta, id, declaration) = fixture("restore-hold");
    let writer = begin(&source, &source_meta, &id, 1, declaration.clone(), None).unwrap();
    data(&source, &source_meta, &id, 1, 0x31);
    let (plan, head) =
        checkpoint_progress(&source, &source_meta, &id, 1, "request", "output", None).unwrap();
    let head = head.head.unwrap();
    let target = Store::init(&source.root().join("restore-target")).unwrap();
    let meta = Meta::open(&target).unwrap();
    let admit = |object: &ObjectRef| {
        crate::durability::restore_derived(&target, &meta, &id, 3, object, || {
            let mut body = Vec::new();
            source.read_into(&object.sha256, &mut body)?;
            target.put_stream(&mut body.as_slice(), Some(object), &Default::default())?;
            Ok(())
        })
        .unwrap();
        crate::gc::collect(target.root(), false).unwrap();
        assert!(target.contains(&object.sha256));
    };
    admit(&head);
    let page = crate::durability::local_window(&target, &head, 0, 128).unwrap();
    for (kind, object) in page.objects {
        assert_eq!(kind, crate::repo_cache::CacheKind::Blob);
        admit(&object);
    }
    assert!(matches!(
        lookup(&target, &meta, &id).unwrap(),
        Lookup::Absent
    ));
    let canonical = declaration.clone().canonical_work_bytes().unwrap();
    validate_checkpoint(
        &target, &meta, &id, &canonical, &head, "request", "output", &plan,
    )
    .unwrap();
    // Ready is read-only: persistent retention remains after its transient guard ends.
    crate::gc::collect(target.root(), false).unwrap();
    assert_eq!(
        roots::read(&target, &id).unwrap().unwrap().restore_epoch,
        Some(3)
    );
    assert_eq!(
        begin(&target, &meta, &id, 2, declaration.clone(), Some(&head))
            .err()
            .unwrap()
            .code,
        Code::WRITER_FENCED
    );
    let restored = begin(&target, &meta, &id, 3, declaration, Some(&head)).unwrap();
    assert!(roots::read(&target, &id)
        .unwrap()
        .unwrap()
        .restore_epoch
        .is_none());
    assert_eq!(completed(&meta, &id, 3).unwrap().0.len(), 1);
    fence(&meta, &id, 3).unwrap();
    release(&meta, restored);
    abandon(&target, &meta, &id).unwrap();
    assert_eq!(
        crate::durability::restore_derived(&target, &meta, &id, 4, &head, || -> Result<()> {
            panic!("abandoned restore performed IO")
        })
        .unwrap_err()
        .code,
        Code::TRANSACTION_CLOSED
    );
    crate::gc::collect(target.root(), false).unwrap();
    assert!(!target.contains(&head.sha256));

    let late_id = format!("sha256:{}", "79".repeat(32));
    let late = ObjectRef::of(b"late immutable object");
    assert_eq!(
        crate::durability::restore_derived(&target, &meta, &late_id, 1, &late, || {
            target.put_stream(
                &mut b"late immutable object".as_slice(),
                Some(&late),
                &Default::default(),
            )?;
            abandon(&target, &meta, &late_id)?;
            Ok(())
        })
        .unwrap_err()
        .code,
        Code::TRANSACTION_CLOSED
    );
    assert!(roots::read(&target, &late_id).unwrap().is_none());
    crate::gc::collect(target.root(), false).unwrap();
    assert!(!target.contains(&late.sha256));
    fence(&source_meta, &id, 1).unwrap();
    release(&source_meta, writer);
    std::fs::remove_dir_all(source.root()).unwrap();
}

/// Opt-in generated-data profile. Payloads are tiny so metadata scaling is visible.
/// Run in release mode; elapsed time is evidence, never a flaky unit-test threshold.
#[test]
#[ignore]
fn profile_derived_transaction_scaling() {
    use std::time::Instant;
    let scenarios = [
        (219, 1, 1),
        (219, 1, 10),
        (219, 4, 10),
        (219, 32, 10),
        (219, 128, 10),
        (3968, 1, 10),
        (3968, 4, 10),
        (219, 1, 219),
    ];
    for (keys, histories, writes) in scenarios {
        let (store, meta, id, _) = fixture("row-profile");
        let mut declaration = super::tests::created_declaration();
        let prototype = declaration.components[0].add[0].clone();
        declaration.components[0].add = (0..keys)
            .map(|index| {
                let mut tensor = prototype.clone();
                tensor.key = format!("weight_{index:04}");
                tensor.shape = vec![4];
                tensor.parts[0].shape = vec![4];
                tensor
            })
            .collect();
        declaration.order = declaration.components[0]
            .add
            .iter()
            .map(|tensor| ("model".into(), tensor.key.clone()))
            .collect();
        declaration.max_new_bytes = (keys * 16) as u64;
        let writer = begin(&store, &meta, &id, 1, declaration, None).unwrap();
        // Populate representative unrelated transactions without timing fixture creation.
        let mut txn = meta.txn().unwrap();
        let prototype = txn.rows[0].clone();
        let row_bytes = crate::canon::write(&prototype.to_value()).len();
        for index in 1..histories {
            let mut row = prototype.clone();
            row.id = ObjectRef::of(format!("history-{index}").as_bytes()).id();
            txn.rows.push(row);
        }
        txn.commit().unwrap();
        let started = Instant::now();
        for _ in 0..20 {
            check_writer(&meta, &id, 1).unwrap();
        }
        let fence_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut part_ms = 0.0;
        let mut checkpoint_ms = 0.0;
        let mut prior = None;
        let mut cursor = None;
        for index in 0..writes {
            let key = format!("weight_{index:04}");
            let started = Instant::now();
            add_indexed_part(
                &store,
                &meta,
                &id,
                1,
                &writer.additions,
                cursor.as_mut(),
                ("model", &key, "value"),
                &mut [index as u8; 16].as_slice(),
            )
            .unwrap();
            part_ms += started.elapsed().as_secs_f64() * 1000.0;
            let started = Instant::now();
            let (_, head) = checkpoint_indexed(
                &store,
                &meta,
                &id,
                1,
                "profile",
                "model",
                prior.as_ref(),
                &mut cursor,
            )
            .unwrap();
            prior = head.head;
            checkpoint_ms += started.elapsed().as_secs_f64() * 1000.0;
        }
        eprintln!("PROFILE {{\"keys\":{keys},\"histories\":{histories},\"writes\":{writes},\"row_bytes\":{row_bytes},\"fence20_ms\":{fence_ms:.3},\"add_ms\":{part_ms:.3},\"checkpoint_ms\":{checkpoint_ms:.3}}}");
        fence(&meta, &id, 1).unwrap();
        release(&meta, writer);
        std::fs::remove_dir_all(store.root()).unwrap();
    }
}

#[test]
fn named_writer_operations_leave_other_rows_and_writer_lock_alone() {
    use rusqlite::{params, Connection};
    let (store, meta, first_id, declaration) = fixture("scoped-row");
    let second_id = ObjectRef::of(b"independent writer").id();
    let first = begin(&store, &meta, &first_id, 1, declaration.clone(), None).unwrap();
    let second = begin(&store, &meta, &second_id, 1, declaration.clone(), None).unwrap();
    let connection = Connection::open(crate::catalog::Catalog::path(store.root())).unwrap();
    let untouched: Vec<u8> = connection
        .query_row(
            "SELECT bytes FROM tensorfs_derived_transactions WHERE id=?1",
            [&second_id],
            |row| row.get(0),
        )
        .unwrap();
    // Corrupt unrelated history makes an accidental whole-table read fail, while
    // a scoped mutation must preserve it byte-for-byte and mutate only this writer.
    let damaged = ObjectRef::of(b"unrelated damaged history").id();
    connection
        .execute(
            "INSERT INTO tensorfs_derived_transactions(id,bytes) VALUES(?1,?2)",
            params![damaged, b"broken".as_slice()],
        )
        .unwrap();
    data(&store, &meta, &first_id, 1, 0x31);
    let after: Vec<u8> = connection
        .query_row(
            "SELECT bytes FROM tensorfs_derived_transactions WHERE id=?1",
            [&second_id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(untouched, after);
    // A read fence must succeed while an unrelated SQLite writer is active. The old
    // BEGIN IMMEDIATE observation instead hit the five-second busy timeout.
    connection.execute_batch("BEGIN IMMEDIATE").unwrap();
    check_writer(&meta, &second_id, 1).unwrap();
    assert!(completed(&meta, &second_id, 1).unwrap().0.is_empty());
    connection.execute_batch("ROLLBACK").unwrap();
    // A named operation does not parse unrelated history. Global inventory still
    // reports the damaged row rather than silently skipping it.
    check_writer(&meta, &first_id, 1).unwrap();
    assert!(meta.derived_rows().is_err());
    connection
        .execute(
            "DELETE FROM tensorfs_derived_transactions WHERE id=?1",
            [&damaged],
        )
        .unwrap();
    assert!(fence(&meta, &first_id, 1).unwrap());
    assert_eq!(
        check_writer(&meta, &first_id, 1).unwrap_err().code,
        Code::WRITER_FENCED
    );
    check_writer(&meta, &second_id, 1).unwrap();
    release(&meta, first);
    let resumed = begin(&store, &meta, &first_id, 2, declaration, None).unwrap();
    assert_eq!(completed(&meta, &first_id, 2).unwrap().0.len(), 1);
    assert_eq!(
        check_writer(&meta, &first_id, 1).unwrap_err().code,
        Code::WRITER_FENCED
    );
    fence(&meta, &first_id, 2).unwrap();
    fence(&meta, &second_id, 1).unwrap();
    release(&meta, resumed);
    release(&meta, second);
    std::fs::remove_dir_all(store.root()).unwrap();
}

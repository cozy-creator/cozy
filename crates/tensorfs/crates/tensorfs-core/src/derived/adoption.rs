//! Independent custody of checkpointed work for another exact derived declaration.

use super::*;

/// Adopt a stopped writer's verified progress without copying tensor payloads or
/// transferring its writer session. The caller owns authorization between requests;
/// native code binds the complete declaration and independently roots the result.
#[allow(clippy::too_many_arguments)]
pub fn adopt_checkpoint(
    store: &Store,
    meta: &Meta,
    transaction: &str,
    source_transaction: &str,
    declaration_bytes: &[u8],
    source_head: &ObjectRef,
    operation: &str,
    slot: &str,
) -> Result<(String, crate::durability::Head)> {
    let transaction = transaction_id(transaction)?;
    let source_transaction = transaction_id(source_transaction)?;
    if transaction == source_transaction {
        return refuse(
            Code::TRANSACTION_CONFLICT,
            "partial adoption requires a new transaction",
        );
    }
    let mut declaration = Declaration::from_value(&crate::canon::parse_canonical(
        declaration_bytes,
        limits::DOC_MAX_BYTES,
    )?)?;
    let canonical = declaration.canonical_work_bytes()?;
    if canonical != declaration_bytes {
        return refuse(
            Code::TRANSACTION_CONFLICT,
            "partial adoption changed its declaration",
        );
    }
    let text = String::from_utf8(canonical).expect("canonical JSON is UTF-8");
    let hold = meta.acquire_hold("derived-adoption")?;
    let result = (|| {
        require_source_roots(store, &declaration)?;
        let sources = load_sources(store, &declaration)?;
        let facts: Vec<_> = sources.iter().map(|source| source.fact.clone()).collect();
        let rows = meta.derived_rows()?;
        let existing = rows.iter().find(|row| row.id == transaction);
        if existing.is_some_and(|row| {
            row.state != TransactionState::Open
                || row.declaration.as_deref() != Some(&text)
                || row.source_facts != facts
        }) {
            return refuse(
                Code::TRANSACTION_CONFLICT,
                "adoption recipient changed or closed",
            );
        }

        let recipient_root = roots::read(store, &transaction)?;
        if recipient_root
            .as_ref()
            .is_some_and(|root| root.restore_epoch.is_some())
        {
            return refuse(
                Code::WRITER_FENCED,
                "adoption recipient has an active restore fence",
            );
        }
        // The filesystem root is committed before the metadata row. Recover that
        // boundary even when the original request has since released its custody.
        if let Some(head) = recipient_root
            .as_ref()
            .and_then(|root| root.checkpoint.as_ref())
        {
            let recovered = read_progress(store, head, &transaction, &text, &facts)?;
            validate_chain(store, head, &recovered, operation, slot)?;
            for object in added_objects(&recovered) {
                require_payload_record(store, &object, "adopted progress")?;
            }
            build_header(
                &declaration,
                &sources,
                &recovered.added_parts,
                &recovered.added_configs,
                true,
            )?;
            let result = observation(store, head.clone())?;
            let mut txn = meta.txn()?;
            if txn.rows.iter().find(|row| row.id == transaction) != existing
                || roots::read(store, &transaction)? != recipient_root
            {
                return refuse(
                    Code::TRANSACTION_CONFLICT,
                    "adoption recipient changed during recovery",
                );
            }
            if existing.is_none() {
                retain_progress(store, &recovered, Some(head))?;
                txn.rows.push(recovered);
                txn.commit()?;
            }
            return Ok(result);
        }
        if existing.is_some() || recipient_root.is_some() {
            return refuse(
                Code::TRANSACTION_CONFLICT,
                "adoption recipient already owns different work",
            );
        }
        let source = rows
            .iter()
            .find(|row| row.id == source_transaction)
            .ok_or_else(|| Refusal {
                code: Code::ROOT_ABSENT,
                detail: "partial source has no retained transaction".into(),
            })?;
        if source.state != TransactionState::Open || source.current_session.is_some() {
            return refuse(
                Code::WRITER_FENCED,
                "partial source writer must be stopped before adoption",
            );
        }
        if source.declaration.as_deref() != Some(&text) || source.source_facts != facts {
            return refuse(
                Code::TRANSACTION_CONFLICT,
                "partial source has different work or inputs",
            );
        }
        let root = roots::read(store, &source_transaction)?.ok_or_else(|| Refusal {
            code: Code::ROOT_ABSENT,
            detail: "partial source no longer has native custody".into(),
        })?;
        if root.restore_epoch.is_some() {
            return refuse(
                Code::WRITER_FENCED,
                "partial source has an active restore fence",
            );
        }
        let current = root.checkpoint.as_ref().ok_or_else(|| Refusal {
            code: Code::ROOT_ABSENT,
            detail: "partial source has no completed checkpoint".into(),
        })?;
        let link = crate::durability::local_link(store, source_head)?;
        let plan = progress_plan(source);
        let selected = crate::durability::checkpoint_predecessor(
            store,
            &crate::durability::Policy {
                chain: &link.operation,
                operation: &source_transaction,
                plan: &plan,
                interval: crate::durability::INTERVAL_BYTES,
            },
            Some(current),
            Some(source_head),
        )?
        .expect("source checkpoint supplied");
        let incoming = read_progress(store, &selected, &source_transaction, &text, &facts)?;
        let mut recipient = incoming.clone();
        recipient.id = transaction.clone();
        recipient.added_parts.clear();
        recipient.added_configs.clear();
        merge_progress(store, &mut recipient, Some(&incoming))?;
        build_header(
            &declaration,
            &sources,
            &recipient.added_parts,
            &recipient.added_configs,
            true,
        )?;
        let (plan, exported) = export(store, &recipient, operation, slot)?;
        let mut txn = meta.txn()?;
        if txn.rows.iter().find(|row| row.id == source_transaction) != Some(source)
            || txn.rows.iter().any(|row| row.id == transaction)
            || roots::read(store, &source_transaction)?.as_ref() != Some(&root)
            || roots::read(store, &transaction)? != recipient_root
        {
            return refuse(
                Code::TRANSACTION_CONFLICT,
                "partial source or recipient changed during adoption",
            );
        }
        retain_progress(store, &recipient, exported.head.as_ref())?;
        txn.rows.push(recipient);
        txn.commit()?;
        Ok((plan, exported))
    })();
    hold.release(meta)?;
    result
}

fn validate_chain(
    store: &Store,
    head: &ObjectRef,
    row: &TransactionRow,
    operation: &str,
    slot: &str,
) -> Result<()> {
    let link = crate::durability::local_link(store, head)?;
    if link.operation != format!("{operation}/{slot}") || link.plan != progress_plan(row) {
        return refuse(
            Code::TRANSACTION_CONFLICT,
            "adopted checkpoint changed its recipient binding",
        );
    }
    Ok(())
}

fn observation(store: &Store, head: ObjectRef) -> Result<(String, crate::durability::Head)> {
    let link = crate::durability::local_link(store, &head)?;
    let chain = crate::durability::local_chain(store, &head, &link.plan)?;
    Ok((
        link.plan,
        crate::durability::Head {
            head: Some(head),
            links: chain.links,
            bytes: chain.bytes,
        },
    ))
}

fn export(
    store: &Store,
    row: &TransactionRow,
    operation: &str,
    slot: &str,
) -> Result<(String, crate::durability::Head)> {
    let plan = progress_plan(row);
    let chain = format!("{operation}/{slot}");
    ascii_name("checkpoint operation", &chain, limits::MAX_NAME_BYTES)?;
    let work_bytes = crate::canon::write(&progress::context(row).to_value());
    let work = ObjectRef::of(&work_bytes);
    store.put_stream(&mut work_bytes.as_slice(), Some(&work), &Default::default())?;
    let delta = progress::Delta::between(row, None, work.clone())?;
    let mut objects = added_objects(row);
    objects.push(work);
    let exported = crate::durability::checkpoint_bytes(
        store,
        &crate::durability::Policy {
            chain: &chain,
            operation: &row.id,
            plan: &plan,
            interval: crate::durability::INTERVAL_BYTES,
        },
        None,
        &crate::canon::write(&delta.to_value()),
        objects,
        None,
    )?;
    Ok((plan, exported.head))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    #[derive(Default)]
    struct MeasuredIO {
        moved: AtomicU64,
        admitted: AtomicU64,
    }

    impl crate::store::Progress for MeasuredIO {
        fn moved(&self, bytes: u64) {
            self.moved.fetch_add(bytes, Ordering::Relaxed);
        }
        fn admitted(&self) {
            self.admitted.fetch_add(1, Ordering::Relaxed);
        }
    }

    impl MeasuredIO {
        fn take(&self) -> (u64, u64) {
            (
                self.moved.swap(0, Ordering::Relaxed),
                self.admitted.swap(0, Ordering::Relaxed),
            )
        }
    }

    #[test]
    fn multiple_checkpoint_groups_separate_rehash_from_adoption_metadata_io() {
        let (store, meta, mut declaration, source, target) = fixture();
        let mut third = declaration.components[0].add[0].clone();
        third.key = "third".into();
        declaration.components[0].add.push(third);
        declaration.order.push(("model".into(), "third".into()));
        declaration.max_new_bytes += 2048;
        let raw = declaration.canonical_work_bytes().unwrap();
        let writer = begin(&store, &meta, &source, 1, declaration.clone(), None).unwrap();
        let first = write(&store, &meta, &source, "weight", 3);
        checkpoint_progress(&store, &meta, &source, 1, "source", "model", None).unwrap();
        let second = write(&store, &meta, &source, "next", 4);
        let (_, checkpoint) =
            checkpoint_progress(&store, &meta, &source, 1, "source", "model", None).unwrap();
        fence(&meta, &source, 1).unwrap();
        release_guards(&meta, Some(writer.writer_hold), writer.source_leases);

        let progress = Arc::new(MeasuredIO::default());
        let store = store.observed(progress.clone());
        eprintln!("CR108_IO_ROOT {}", store.root().display());
        eprintln!("CR108_IO_PAYLOAD {} {}", first.sha256, second.sha256);
        // Lose only the verification receipts in this fixture, not the actual
        // bytes. The real verifier must rehash them before the warm adoption arm.
        let catalog = crate::catalog::Catalog::open(store.root()).unwrap();
        catalog.drop_verification(&first.sha256);
        catalog.drop_verification(&second.sha256);
        drop(catalog);
        eprintln!("CR108_IO_PHASE verification");
        assert!(matches!(
            store.verify(&first.sha256).unwrap(),
            crate::store::Verdict::Invalidated { .. }
        ));
        assert!(matches!(
            store.verify(&second.sha256).unwrap(),
            crate::store::Verdict::Invalidated { .. }
        ));
        let verification = progress.take();
        assert_eq!(verification, (4096, 0));

        eprintln!("CR108_IO_PHASE inspect_before");
        let before: BTreeSet<_> = store.objects().unwrap().into_iter().collect();
        let before_payloads: Vec<_> = [&first, &second]
            .iter()
            .map(|object| {
                let path = store.blob_path(&object.sha256);
                (
                    std::fs::read(&path).unwrap(),
                    std::fs::metadata(path).unwrap().modified().unwrap(),
                )
            })
            .collect();
        eprintln!("CR108_IO_PHASE adoption");
        let (_, adopted) = adopt_checkpoint(
            &store,
            &meta,
            &target,
            &source,
            &raw,
            checkpoint.head.as_ref().unwrap(),
            "target",
            "model",
        )
        .unwrap();
        let adoption = progress.take();
        eprintln!("CR108_IO_PHASE inspect_after");
        let fresh: Vec<_> = store
            .objects()
            .unwrap()
            .into_iter()
            .filter(|id| !before.contains(id))
            .collect();
        let metadata_bytes: u64 = fresh
            .iter()
            .map(|id| std::fs::metadata(store.blob_path(id)).unwrap().len())
            .sum();
        assert!(metadata_bytes > 0);
        assert_eq!(adoption, (metadata_bytes, fresh.len() as u64));
        for (object, prior) in [&first, &second].iter().zip(before_payloads) {
            let path = store.blob_path(&object.sha256);
            assert_eq!(std::fs::read(&path).unwrap(), prior.0);
            assert_eq!(
                std::fs::metadata(path).unwrap().modified().unwrap(),
                prior.1
            );
        }
        eprintln!("CR108_IO_PHASE restore");
        let next = begin(
            &store,
            &meta,
            &target,
            1,
            declaration,
            adopted.head.as_ref(),
        )
        .unwrap();
        assert_eq!(
            completed(&meta, &target, 1).unwrap().0,
            vec![
                ("model".into(), "next".into(), "value".into()),
                ("model".into(), "weight".into(), "value".into())
            ]
        );
        let restore = progress.take();
        assert_eq!(restore, (0, 0));
        eprintln!("CR108_IO_PHASE finish");
        write(&store, &meta, &target, "third", 5);
        commit(&store, &meta, &target, 1).unwrap();
        release_guards(&meta, Some(next.writer_hold), next.source_leases);
        eprintln!("CR108_IO_PHASE done");
        eprintln!("CR108_IO_COUNTS verification={verification:?} adoption={adoption:?} restore={restore:?} adoption_metadata_bytes={metadata_bytes}");
    }

    fn fixture() -> (Store, Meta, Declaration, String, String) {
        let path = std::env::temp_dir().join(format!(
            "tensorfs-partial-adopt-{}",
            crate::meta::now_nanos_unique()
        ));
        let store = Store::init(&path).unwrap();
        let meta = Meta::open(&store).unwrap();
        let mut declaration = super::super::tests::created_declaration();
        let mut second = declaration.components[0].add[0].clone();
        second.key = "next".into();
        declaration.components[0].add.push(second);
        declaration.order.push(("model".into(), "next".into()));
        declaration.max_new_bytes *= 2;
        (
            store,
            meta,
            declaration,
            format!("sha256:{}", "a1".repeat(32)),
            format!("sha256:{}", "a2".repeat(32)),
        )
    }

    fn write(store: &Store, meta: &Meta, id: &str, key: &str, byte: u8) -> ObjectRef {
        let written = add_part(
            store,
            meta,
            id,
            1,
            ("model", key, "value"),
            &mut vec![byte; 2048].as_slice(),
        )
        .unwrap();
        written.part.segments()[0].clone()
    }

    #[test]
    fn independent_partial_adoption_survives_producer_release_and_finishes_exactly() {
        let (store, meta, mut declaration, source, target) = fixture();
        let bytes = declaration.canonical_work_bytes().unwrap();
        let writer = begin(&store, &meta, &source, 1, declaration.clone(), None).unwrap();
        let object = write(&store, &meta, &source, "weight", 3);
        let (_, progress) =
            checkpoint_progress(&store, &meta, &source, 1, "old", "output", None).unwrap();
        // An add after the durable checkpoint is deliberately not adopted.
        write(&store, &meta, &source, "next", 9);
        let head = progress.head.unwrap();
        assert_eq!(
            adopt_checkpoint(&store, &meta, &target, &source, &bytes, &head, "new", "output")
                .unwrap_err()
                .code,
            Code::WRITER_FENCED
        );
        fence(&meta, &source, 1).unwrap();
        release_guards(&meta, Some(writer.writer_hold), writer.source_leases);
        let (plan, adopted) = adopt_checkpoint(
            &store, &meta, &target, &source, &bytes, &head, "new", "output",
        )
        .unwrap();
        assert_ne!(adopted.head.as_ref(), Some(&head));
        validate_checkpoint(
            &store,
            &meta,
            &target,
            &bytes,
            adopted.head.as_ref().unwrap(),
            "new",
            "output",
            &plan,
        )
        .unwrap();
        abandon(&store, &meta, &source).unwrap();
        crate::gc::collect(store.root(), false).unwrap();
        assert!(store.contains(&object.sha256));
        let (_, replay) = adopt_checkpoint(
            &store, &meta, &target, &source, &bytes, &head, "new", "output",
        )
        .unwrap();
        assert_eq!(replay.head, adopted.head);
        let next = begin(
            &store,
            &meta,
            &target,
            1,
            declaration.clone(),
            adopted.head.as_ref(),
        )
        .unwrap();
        assert_eq!(
            completed(&meta, &target, 1).unwrap().0,
            vec![("model".into(), "weight".into(), "value".into())]
        );
        write(&store, &meta, &target, "next", 4);
        let result = commit(&store, &meta, &target, 1).unwrap();
        release_guards(&meta, Some(next.writer_hold), next.source_leases);
        let control = format!("sha256:{}", "a3".repeat(32));
        let clean = begin(&store, &meta, &control, 1, declaration, None).unwrap();
        write(&store, &meta, &control, "weight", 3);
        write(&store, &meta, &control, "next", 4);
        assert_eq!(
            commit(&store, &meta, &control, 1).unwrap().manifest,
            result.manifest
        );
        release_guards(&meta, Some(clean.writer_hold), clean.source_leases);
    }

    #[test]
    fn different_implementation_or_closed_recipient_cannot_adopt() {
        let (store, meta, mut declaration, source, target) = fixture();
        let bytes = declaration.canonical_work_bytes().unwrap();
        let writer = begin(&store, &meta, &source, 1, declaration.clone(), None).unwrap();
        write(&store, &meta, &source, "weight", 3);
        let (_, progress) =
            checkpoint_progress(&store, &meta, &source, 1, "old", "output", None).unwrap();
        fence(&meta, &source, 1).unwrap();
        release_guards(&meta, Some(writer.writer_hold), writer.source_leases);
        let head = progress.head.unwrap();
        declaration.work_fingerprint = Some(format!("sha256:{}", "ff".repeat(32)));
        assert_eq!(
            adopt_checkpoint(
                &store,
                &meta,
                &target,
                &source,
                &declaration.canonical_work_bytes().unwrap(),
                &head,
                "new",
                "output"
            )
            .unwrap_err()
            .code,
            Code::TRANSACTION_CONFLICT
        );
        assert!(matches!(
            lookup(&store, &meta, &target).unwrap(),
            Lookup::Absent
        ));
        abandon(&store, &meta, &target).unwrap();
        assert_eq!(
            adopt_checkpoint(&store, &meta, &target, &source, &bytes, &head, "new", "output")
                .unwrap_err()
                .code,
            Code::TRANSACTION_CONFLICT
        );
    }

    #[test]
    fn root_before_metadata_commit_recovery_does_not_need_the_original_owner() {
        let (store, meta, mut declaration, source, target) = fixture();
        let bytes = declaration.canonical_work_bytes().unwrap();
        let writer = begin(&store, &meta, &source, 1, declaration.clone(), None).unwrap();
        let object = write(&store, &meta, &source, "weight", 3);
        let (_, progress) =
            checkpoint_progress(&store, &meta, &source, 1, "old", "output", None).unwrap();
        fence(&meta, &source, 1).unwrap();
        release_guards(&meta, Some(writer.writer_hold), writer.source_leases);
        let head = progress.head.unwrap();
        let (_, adopted) = adopt_checkpoint(
            &store, &meta, &target, &source, &bytes, &head, "new", "output",
        )
        .unwrap();
        // Reproduce the durable root / uncommitted SQL boundary exactly.
        let db = rusqlite::Connection::open(crate::catalog::Catalog::path(store.root())).unwrap();
        db.execute(
            "DELETE FROM tensorfs_derived_transactions WHERE id=?1",
            [&target],
        )
        .unwrap();
        drop(db);
        abandon(&store, &meta, &source).unwrap();
        crate::gc::collect(store.root(), false).unwrap();
        let (_, replay) = adopt_checkpoint(
            &store, &meta, &target, &source, &bytes, &head, "new", "output",
        )
        .unwrap();
        assert_eq!(replay.head, adopted.head);
        assert!(store.contains(&object.sha256));
        let writer = begin(&store, &meta, &target, 1, declaration, replay.head.as_ref()).unwrap();
        assert_eq!(completed(&meta, &target, 1).unwrap().0.len(), 1);
        release_guards(&meta, Some(writer.writer_hold), writer.source_leases);
    }

    #[test]
    fn replay_refuses_missing_or_corrupt_payload_and_preserves_restore_fence() {
        for mode in ["missing", "corrupt", "restore"] {
            let (store, meta, mut declaration, source, target) = fixture();
            let bytes = declaration.canonical_work_bytes().unwrap();
            let writer = begin(&store, &meta, &source, 1, declaration, None).unwrap();
            let object = write(&store, &meta, &source, "weight", 3);
            let (_, progress) =
                checkpoint_progress(&store, &meta, &source, 1, "old", "output", None).unwrap();
            fence(&meta, &source, 1).unwrap();
            release_guards(&meta, Some(writer.writer_hold), writer.source_leases);
            let head = progress.head.unwrap();
            adopt_checkpoint(
                &store, &meta, &target, &source, &bytes, &head, "new", "output",
            )
            .unwrap();
            match mode {
                "missing" => std::fs::remove_file(store.blob_path(&object.sha256)).unwrap(),
                "corrupt" => {
                    std::fs::remove_file(store.blob_path(&object.sha256)).unwrap();
                    std::fs::write(store.blob_path(&object.sha256), vec![7; 2048]).unwrap();
                }
                _ => retain_restore(&store, &meta, &target, 2, None).unwrap(),
            }
            let error = adopt_checkpoint(
                &store, &meta, &target, &source, &bytes, &head, "new", "output",
            )
            .unwrap_err();
            if mode == "restore" {
                assert_eq!(error.code, Code::WRITER_FENCED);
                assert_eq!(
                    roots::read(&store, &target).unwrap().unwrap().restore_epoch,
                    Some(2)
                );
            }
        }
    }
}

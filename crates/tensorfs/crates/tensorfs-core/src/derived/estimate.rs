//! Remaining added-role geometry without opening or fencing a native writer.

use super::*;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct WriteEstimate {
    pub payload_bytes: u64,
    pub largest_part_bytes: u64,
    pub parts: u64,
}

/// Inspect exactly the progress that begin would adopt. The observation is not a
/// reservation or receipt: begin revalidates it, and per-role admission still
/// guards every actual write. All inspection holds are released before return.
pub fn estimate_write(
    store: &Store,
    meta: &Meta,
    transaction: &str,
    writer_session: u64,
    mut declaration: Declaration,
    checkpoint: Option<&ObjectRef>,
) -> Result<WriteEstimate> {
    let transaction = transaction_id(transaction)?;
    if writer_session == 0 || writer_session > limits::INT_MAX as u64 {
        return refuse(
            Code::NUMBER_RANGE,
            "WriterSessionId must be positive and inside the canonical integer range",
        );
    }
    let text =
        String::from_utf8(declaration.canonical_work_bytes()?).expect("canonical JSON is UTF-8");
    let hold = meta.acquire_hold("derived-write-estimate")?;
    let estimate = (|| {
        require_source_roots(store, &declaration)?;
        let sources = load_sources(store, &declaration)?;
        let facts: Vec<_> = sources.iter().map(|source| source.fact.clone()).collect();
        let incoming = checkpoint
            .map(|head| read_progress(store, head, &transaction, &text, &facts))
            .transpose()?;
        let initial = meta
            .derived_rows()?
            .into_iter()
            .find(|row| row.id == transaction);
        let mut row = planned_row(
            initial.as_ref(),
            &transaction,
            writer_session,
            &text,
            &facts,
        )?;
        merge_progress(store, &mut row, incoming.as_ref())?;
        build_header(
            &declaration,
            &sources,
            &row.added_parts,
            &row.added_configs,
            true,
        )?;
        if roots::read(store, &transaction)?
            .and_then(|root| root.restore_epoch)
            .is_some_and(|epoch| epoch > writer_session)
        {
            return refuse(
                Code::WRITER_FENCED,
                "writer is behind the retained restore epoch",
            );
        }
        let completed: HashSet<_> = row
            .added_parts
            .iter()
            .map(|part| {
                (
                    part.component.as_str(),
                    part.key.as_str(),
                    part.role.as_str(),
                )
            })
            .collect();
        let mut result = WriteEstimate::default();
        for (_, data) in &declaration.files {
            let length = data.len() as u64;
            result.payload_bytes += length;
            result.largest_part_bytes = result.largest_part_bytes.max(length);
            result.parts += 1;
        }
        for component in &declaration.components {
            for tensor in &component.add {
                for part in &tensor.parts {
                    if part.source.is_some()
                        || completed.contains(&(
                            component.target.as_str(),
                            tensor.key.as_str(),
                            part.role.as_str(),
                        ))
                    {
                        continue;
                    }
                    let length = checked_bytes("remaining derived part", &part.shape, part.dtype)?;
                    result.payload_bytes = result
                        .payload_bytes
                        .checked_add(length)
                        .ok_or_else(|| arithmetic("remaining derived payload overflows"))?;
                    result.largest_part_bytes = result.largest_part_bytes.max(length);
                    result.parts += 1;
                }
            }
        }
        Ok(result)
    })();
    let released = hold.release(meta);
    estimate.and_then(|value| released.map(|()| value))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(label: &str) -> (Store, Meta, String, Declaration) {
        let path = std::env::temp_dir().join(format!(
            "tensorfs-derived-estimate-{label}-{}",
            crate::meta::now_nanos_unique()
        ));
        let store = Store::init(&path).unwrap();
        let meta = Meta::open(&store).unwrap();
        (
            store,
            meta,
            format!("sha256:{}", "78".repeat(32)),
            super::super::tests::created_declaration(),
        )
    }

    fn release(meta: &Meta, begun: Begin) {
        release_guards(meta, Some(begun.writer_hold), begun.source_leases);
    }

    #[test]
    fn fresh_estimate_does_not_claim_writer_or_block_collection() {
        let (store, meta, id, declaration) = fixture("fresh");
        let before = meta.derived_rows().unwrap();
        assert_eq!(
            estimate_write(&store, &meta, &id, 1, declaration.clone(), None).unwrap(),
            WriteEstimate {
                payload_bytes: 2048,
                largest_part_bytes: 2048,
                parts: 1
            }
        );
        assert_eq!(meta.derived_rows().unwrap(), before);
        assert!(matches!(
            lookup(&store, &meta, &id).unwrap(),
            Lookup::Absent
        ));
        crate::gc::collect(store.root(), false).unwrap();
        let writer = begin(&store, &meta, &id, 1, declaration, None).unwrap();
        fence(&meta, &id, 1).unwrap();
        release(&meta, writer);
        std::fs::remove_dir_all(store.root()).unwrap();
    }

    #[test]
    fn interrupted_large_writer_only_charges_its_small_remaining_tail() {
        let (store, meta, id, mut declaration) = fixture("large-partial");
        const LARGE: u64 = 64 << 20;
        let mut tail = declaration.components[0].add[0].clone();
        tail.key = "tail".into();
        tail.shape = vec![1];
        tail.parts[0].shape = vec![1];
        declaration.components[0].add[0].shape = vec![LARGE / 4];
        declaration.components[0].add[0].parts[0].shape = vec![LARGE / 4];
        declaration.components[0].add.push(tail);
        declaration.order.push(("model".into(), "tail".into()));
        declaration.max_new_bytes = LARGE + 4;
        let writer = begin(&store, &meta, &id, 1, declaration.clone(), None).unwrap();
        let written = add_part(
            &store,
            &meta,
            &id,
            1,
            ("model", "weight", "value"),
            &mut std::io::repeat(0x31).take(LARGE),
        )
        .unwrap();
        let (_, checkpoint) =
            checkpoint_progress(&store, &meta, &id, 1, "request", "model", None).unwrap();
        assert!(checkpoint.head.is_some());
        fence(&meta, &id, 1).unwrap();
        release(&meta, writer);
        let before = meta.derived_rows().unwrap();
        assert_eq!(
            estimate_write(
                &store,
                &meta,
                &id,
                1,
                declaration.clone(),
                checkpoint.head.as_ref()
            )
            .unwrap_err()
            .code,
            Code::WRITER_FENCED
        );
        assert_eq!(
            estimate_write(
                &store,
                &meta,
                &id,
                2,
                declaration.clone(),
                checkpoint.head.as_ref()
            )
            .unwrap(),
            WriteEstimate {
                payload_bytes: 4,
                largest_part_bytes: 4,
                parts: 1
            }
        );
        assert_eq!(meta.derived_rows().unwrap(), before);
        crate::gc::collect(store.root(), false).unwrap();
        assert!(written
            .part
            .segments()
            .iter()
            .all(|part| store.contains(&part.sha256)));
        let resumed = begin(&store, &meta, &id, 2, declaration, checkpoint.head.as_ref()).unwrap();
        assert_eq!(
            completed(&meta, &id, 2).unwrap().0,
            vec![("model".into(), "weight".into(), "value".into())]
        );
        add_part(
            &store,
            &meta,
            &id,
            2,
            ("model", "tail", "value"),
            &mut [0u8; 4].as_slice(),
        )
        .unwrap();
        commit(&store, &meta, &id, 2).unwrap();
        release(&meta, resumed);
        std::fs::remove_dir_all(store.root()).unwrap();
    }

    #[test]
    fn grafts_charge_no_new_payload_and_preserve_source_custody() {
        let (store, meta, id, mut declaration) = fixture("graft");
        let writer = begin(&store, &meta, &id, 1, declaration.clone(), None).unwrap();
        add_part(
            &store,
            &meta,
            &id,
            1,
            ("model", "weight", "value"),
            &mut [0x31u8; 2048].as_slice(),
        )
        .unwrap();
        let original = commit(&store, &meta, &id, 1).unwrap();
        release(&meta, writer);
        declaration.sources.push(Source {
            alias: "original".into(),
            manifest: original.manifest.clone(),
        });
        declaration.components[0].add[0].parts[0].source = Some(PartSource {
            source: "original".into(),
            component: "model".into(),
            tensor: "weight".into(),
            role: "value".into(),
        });
        declaration.max_new_bytes = 0;
        let target = format!("sha256:{}", "79".repeat(32));
        let before = meta.derived_rows().unwrap();
        assert_eq!(
            estimate_write(&store, &meta, &target, 1, declaration.clone(), None).unwrap(),
            WriteEstimate::default()
        );
        assert_eq!(meta.derived_rows().unwrap(), before);
        crate::gc::collect(store.root(), false).unwrap();
        let target_writer = begin(&store, &meta, &target, 1, declaration, None).unwrap();
        assert_eq!(
            commit(&store, &meta, &target, 1).unwrap().manifest,
            original.manifest
        );
        release(&meta, target_writer);
        std::fs::remove_dir_all(store.root()).unwrap();
    }

    #[test]
    fn changed_work_and_missing_checkpoint_refuse_without_mutating_progress() {
        let (store, meta, id, declaration) = fixture("refuse");
        let writer = begin(&store, &meta, &id, 1, declaration.clone(), None).unwrap();
        fence(&meta, &id, 1).unwrap();
        release(&meta, writer);
        let before = meta.derived_rows().unwrap();
        let mut changed = declaration.clone();
        changed.work_fingerprint = Some(format!("sha256:{}", "65".repeat(32)));
        assert_eq!(
            estimate_write(&store, &meta, &id, 2, changed, None)
                .unwrap_err()
                .code,
            Code::TRANSACTION_CONFLICT
        );
        let absent = ObjectRef::of(b"not a checkpoint");
        assert!(estimate_write(&store, &meta, &id, 2, declaration, Some(&absent)).is_err());
        assert_eq!(meta.derived_rows().unwrap(), before);
        crate::gc::collect(store.root(), false).unwrap();
        std::fs::remove_dir_all(store.root()).unwrap();
    }
}

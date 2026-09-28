//! Accepted parts are rows of their own: accepting one never reads or rewrites the others.

use super::*;
use rusqlite::params;

fn fixture(label: &str) -> (Store, Meta, String, Declaration) {
    let root = std::env::temp_dir().join(format!(
        "tensorfs-part-rows-{label}-{}",
        crate::meta::now_nanos_unique()
    ));
    let store = Store::init(&root).unwrap();
    let meta = Meta::open(&store).unwrap();
    let mut declaration = super::tests::created_declaration();
    let prototype = declaration.components[0].add[0].clone();
    declaration.components[0].add = (0..3)
        .map(|index| {
            let mut tensor = prototype.clone();
            tensor.key = format!("weight_{index}");
            tensor.shape = vec![128];
            tensor.parts[0].shape = vec![128];
            tensor
        })
        .collect();
    declaration.order = (0..3)
        .map(|index| ("model".into(), format!("weight_{index}")))
        .collect();
    declaration.max_new_bytes = 3 * 512;
    (
        store,
        meta,
        format!("sha256:{}", "7a".repeat(32)),
        declaration,
    )
}

fn catalog(store: &Store) -> rusqlite::Connection {
    rusqlite::Connection::open(crate::catalog::Catalog::path(store.root())).unwrap()
}

fn add(store: &Store, meta: &Meta, id: &str, writer: &Begin, key: &str, byte: u8) -> Result<()> {
    add_indexed_part(
        store,
        meta,
        id,
        1,
        &writer.additions,
        None,
        ("model", key, "value"),
        &mut [byte; 512].as_slice(),
    )
    .map(|_| ())
}

fn stored_rows(store: &Store, id: &str) -> (usize, Option<i64>, i64) {
    let connection = catalog(store);
    let (bytes, open): (Vec<u8>, Option<i64>) = connection
        .query_row(
            "SELECT bytes,open_session FROM tensorfs_derived_transactions WHERE id=?1",
            [id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    let value = crate::canon::parse_canonical(&bytes, limits::DOC_MAX_BYTES).unwrap();
    let inline = TransactionRow::from_value(&value)
        .unwrap()
        .added_parts
        .len();
    let parts = connection
        .query_row(
            "SELECT count(*) FROM tensorfs_derived_parts WHERE id=?1",
            [id],
            |row| row.get(0),
        )
        .unwrap();
    (inline, open, parts)
}

#[test]
fn parts_are_rows_that_resume_refuse_conflicts_and_stay_retained() {
    let (store, meta, id, declaration) = fixture("rows");
    let writer = begin(&store, &meta, &id, 1, declaration.clone(), None).unwrap();
    add(&store, &meta, &id, &writer, "weight_0", 1).unwrap();
    add(&store, &meta, &id, &writer, "weight_1", 2).unwrap();
    // The row itself names no part; each part is one row; the writer's session is indexed.
    assert_eq!(stored_rows(&store, &id), (0, Some(1), 2));
    // The same bytes again are the same accepted part; other bytes conflict.
    add(&store, &meta, &id, &writer, "weight_0", 1).unwrap();
    assert_eq!(stored_rows(&store, &id).2, 2);
    assert_eq!(
        add(&store, &meta, &id, &writer, "weight_0", 9)
            .unwrap_err()
            .code,
        Code::TRANSACTION_CONFLICT
    );
    // A fenced writer cannot add. Once it is gone, GC keeps its accepted bytes through
    // the append-only journal alone, and its successor resumes with both parts.
    fence(&meta, &id, 1).unwrap();
    assert_eq!(
        add(&store, &meta, &id, &writer, "weight_2", 3)
            .unwrap_err()
            .code,
        Code::WRITER_FENCED
    );
    release_guards(&meta, Some(writer.writer_hold), writer.source_leases);
    crate::gc::collect(store.root(), false).unwrap();
    assert!(store.contains(&ObjectRef::of(&[1u8; 512]).sha256));
    assert!(store.contains(&ObjectRef::of(&[2u8; 512]).sha256));
    let resumed = begin(&store, &meta, &id, 2, declaration, None).unwrap();
    assert_eq!(completed(&meta, &id, 2).unwrap().0.len(), 2);
    add_indexed_part(
        &store,
        &meta,
        &id,
        2,
        &resumed.additions,
        None,
        ("model", "weight_2", "value"),
        &mut [3u8; 512].as_slice(),
    )
    .unwrap();
    let receipt = commit(&store, &meta, &id, 2).unwrap();
    assert_eq!(receipt.added_objects.len(), 3);
    assert_eq!(stored_rows(&store, &id).1, None);
    fence(&meta, &id, 2).unwrap();
    release_guards(&meta, Some(resumed.writer_hold), resumed.source_leases);
    std::fs::remove_dir_all(store.root()).unwrap();
}

#[test]
fn an_older_builds_inline_row_and_a_torn_journal_line_are_read() {
    let (store, meta, id, declaration) = fixture("legacy");
    let writer = begin(&store, &meta, &id, 1, declaration, None).unwrap();
    add(&store, &meta, &id, &writer, "weight_0", 1).unwrap();
    // An older build rewrites the row with its parts inline and no open session.
    let row = meta.derived_row(&id).unwrap().unwrap();
    let connection = catalog(&store);
    connection
        .execute("DELETE FROM tensorfs_derived_parts WHERE id=?1", [&id])
        .unwrap();
    connection
        .execute(
            "UPDATE tensorfs_derived_transactions SET bytes=?2, open_session=NULL WHERE id=?1",
            params![id, crate::canon::write(&row.to_value())],
        )
        .unwrap();
    // A writer that died mid-append left a torn journal line.
    let journal = store
        .root()
        .join("roots/derived")
        .join(format!("{}.parts", id.trim_start_matches("sha256:")));
    let mut torn = std::fs::read(&journal).unwrap();
    torn.extend_from_slice(b"[{\"length\":512,\"sha");
    std::fs::write(&journal, torn).unwrap();

    assert_eq!(
        add(&store, &meta, &id, &writer, "weight_0", 9)
            .unwrap_err()
            .code,
        Code::TRANSACTION_CONFLICT
    );
    add(&store, &meta, &id, &writer, "weight_1", 2).unwrap();
    assert_eq!(completed(&meta, &id, 1).unwrap().0.len(), 2);
    fence(&meta, &id, 1).unwrap();
    release_guards(&meta, Some(writer.writer_hold), writer.source_leases);
    crate::gc::collect(store.root(), false).unwrap();
    assert!(store.contains(&ObjectRef::of(&[1u8; 512]).sha256));
    assert!(store.contains(&ObjectRef::of(&[2u8; 512]).sha256));
    abandon(&store, &meta, &id).unwrap();
    assert!(!journal.exists());
    std::fs::remove_dir_all(store.root()).unwrap();
}

#[test]
fn indexed_checkpoints_write_the_chain_full_checkpoints_write() {
    const COUNT: usize = 40;
    let mut stores = Vec::new();
    for label in ["full", "indexed"] {
        let (store, meta, id, mut declaration) = fixture(label);
        let prototype = declaration.components[0].add[0].clone();
        declaration.components[0].add = (0..COUNT)
            .map(|index| {
                let mut tensor = prototype.clone();
                tensor.key = format!("weight_{index:03}");
                tensor
            })
            .collect();
        declaration.order = (0..COUNT)
            .map(|index| ("model".into(), format!("weight_{index:03}")))
            .collect();
        declaration.configs = vec![ConfigDeclaration::Add {
            target: "model.json".into(),
        }];
        declaration.max_new_bytes = (COUNT * 512 + 64) as u64;
        let writer = begin(&store, &meta, &id, 1, declaration.clone(), None).unwrap();
        stores.push((store, meta, id, declaration, writer));
    }
    let mut cursor = None;
    let (mut full_prior, mut indexed_prior) = (None, None);
    let mut heads: Vec<ObjectRef> = Vec::new();
    for index in 0..COUNT {
        let key = format!("weight_{index:03}");
        let payload = [index as u8; 512];
        let (store, meta, id, _, writer) = &stores[0];
        add_indexed_part(
            store,
            meta,
            id,
            1,
            &writer.additions,
            None,
            ("model", &key, "value"),
            &mut payload.as_slice(),
        )
        .unwrap();
        let full = checkpoint_progress(store, meta, id, 1, "request", "model", full_prior.as_ref())
            .unwrap();
        let (store, meta, id, _, writer) = &stores[1];
        add_indexed_part(
            store,
            meta,
            id,
            1,
            &writer.additions,
            cursor.as_mut(),
            ("model", &key, "value"),
            &mut payload.as_slice(),
        )
        .unwrap();
        // An older acknowledged head sends the writer down the full path, which re-seeds it.
        let previous = if index == 25 {
            heads.get(heads.len() - 3).cloned()
        } else {
            indexed_prior.clone()
        };
        let indexed = checkpoint_indexed(
            store,
            meta,
            id,
            1,
            "request",
            "model",
            previous.as_ref(),
            &mut cursor,
        )
        .unwrap();
        assert_eq!(full, indexed, "checkpoint {index} diverged");
        assert_eq!(cursor.as_ref().unwrap().tip.head, indexed.1);
        full_prior = full.1.head;
        indexed_prior = indexed.1.head;
        heads.push(indexed_prior.clone().unwrap());
        if index == 20 {
            for (store, meta, id, _, _) in &stores {
                add_config(meta, id, 1, "model.json", &mut b"{}".as_slice()).unwrap();
                let _ = store;
            }
            cursor = None;
        }
    }
    let (store, meta, id, declaration, writer) = stores.pop().unwrap();
    let head = indexed_prior.unwrap();
    let canonical = String::from_utf8(declaration.clone().canonical_work_bytes().unwrap()).unwrap();
    let restored = read_progress(&store, &head, &id, &canonical, &[]).unwrap();
    assert_eq!(restored.added_parts.len(), COUNT);
    assert_eq!(restored.added_configs.len(), 1);
    fence(&meta, &id, 1).unwrap();
    release_guards(&meta, Some(writer.writer_hold), writer.source_leases);
    let resumed = begin(&store, &meta, &id, 2, declaration, Some(&head)).unwrap();
    assert_eq!(completed(&meta, &id, 2).unwrap().0.len(), COUNT);
    fence(&meta, &id, 2).unwrap();
    release_guards(&meta, Some(resumed.writer_hold), resumed.source_leases);
    std::fs::remove_dir_all(store.root()).unwrap();
    for (store, meta, id, _, writer) in stores {
        fence(&meta, &id, 1).unwrap();
        release_guards(&meta, Some(writer.writer_hold), writer.source_leases);
        std::fs::remove_dir_all(store.root()).unwrap();
    }
}

#[test]
fn an_indexed_checkpoint_does_not_reread_its_chain() {
    let (store, meta, id, declaration) = fixture("no-reread");
    let writer = begin(&store, &meta, &id, 1, declaration, None).unwrap();
    let mut cursor = None;
    let mut prior: Option<ObjectRef> = None;
    let mut first = None;
    for (index, key) in ["weight_0", "weight_1", "weight_2"].into_iter().enumerate() {
        add_indexed_part(
            &store,
            &meta,
            &id,
            1,
            &writer.additions,
            cursor.as_mut(),
            ("model", key, "value"),
            &mut [index as u8 + 1; 512].as_slice(),
        )
        .unwrap();
        let (_, head) = checkpoint_indexed(
            &store,
            &meta,
            &id,
            1,
            "request",
            "model",
            prior.as_ref(),
            &mut cursor,
        )
        .unwrap();
        prior = head.head;
        first.get_or_insert(prior.clone().unwrap());
        if index == 0 {
            // The chain's root link is gone: a checkpoint that walked the chain now fails.
            let root = first.clone().unwrap();
            std::fs::remove_file(store.object_path(root.sha256.trim_start_matches("sha256:")))
                .unwrap();
            assert!(
                checkpoint_progress(&store, &meta, &id, 1, "request", "model", prior.as_ref())
                    .is_err()
            );
        }
    }
    assert_eq!(cursor.as_ref().unwrap().parts, 3);
    fence(&meta, &id, 1).unwrap();
    release_guards(&meta, Some(writer.writer_hold), writer.source_leases);
    std::fs::remove_dir_all(store.root()).unwrap();
}

#[test]
fn a_digest_receipt_names_the_declaration_the_intent_bound() {
    let (store, meta, id, declaration) = fixture("receipt-digest");
    let intent = declaration.clone().canonical_work_bytes().unwrap();
    let writer = begin(&store, &meta, &id, 1, declaration, None).unwrap();
    for (index, key) in ["weight_0", "weight_1", "weight_2"].into_iter().enumerate() {
        add(&store, &meta, &id, &writer, key, index as u8 + 1).unwrap();
    }
    let receipt = commit(&store, &meta, &id, 1).unwrap().to_value();
    let field = |name: &str| {
        as_obj("receipt", "facts", &receipt)
            .unwrap()
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.clone())
    };
    assert!(field("declaration").is_none());
    assert_eq!(
        field("declaration_digest"),
        Some(Value::str(ObjectRef::of(&intent).id()))
    );
    assert_eq!(
        field("declaration_length"),
        Some(Value::uint(intent.len() as u64))
    );
    let _ = fence(&meta, &id, 1);
    release_guards(&meta, Some(writer.writer_hold), writer.source_leases);
    std::fs::remove_dir_all(store.root()).unwrap();
}

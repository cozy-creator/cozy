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

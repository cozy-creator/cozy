use tensorfs_core::{
    catalog::Catalog,
    checkpoint,
    dtype::Dtype,
    ids::ObjectRef,
    limits, sha256,
    store::{Fault, Store, Verdict},
};

fn store(label: &str) -> Store {
    Store::init(&std::env::temp_dir().join(format!(
        "tfs-resident-{label}-{}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    )))
    .unwrap()
}

#[test]
fn held_admission_rolls_back_both_catalog_facts_on_failure() {
    let store = store("atomic");
    let catalog = Catalog::open(store.root()).unwrap();
    let body = vec![0x31; 512];
    let object = ObjectRef::of(&body);
    assert!(store
        .put_stream_held(
            &mut body.as_slice(),
            Some(&object),
            &Fault::default(),
            Some("absent")
        )
        .is_err());
    assert!(!store.contains(&object.sha256));
    let _writer = catalog
        .begin_operation("resident", "proof", "atomic")
        .unwrap();
    let db = rusqlite::Connection::open(Catalog::path(store.root())).unwrap();
    db.execute_batch("CREATE TRIGGER refuse_verification BEFORE INSERT ON tensorfs_verified_blobs BEGIN SELECT RAISE(ABORT,'admission proof'); END").unwrap();
    let error = store
        .put_stream_held(
            &mut body.as_slice(),
            Some(&object),
            &Fault::default(),
            Some("resident"),
        )
        .unwrap_err();
    assert!(error.detail.contains("admission proof"));
    assert!(store.contains(&object.sha256));
    assert!(store.record_valid(&object.sha256).is_err());
    assert!(catalog.held_keys().unwrap().is_empty());
    db.execute_batch("DROP TRIGGER refuse_verification")
        .unwrap();
    // The failed writer may leave an immutable object, but never a false verification.
    // A no-clobber loser must not endorse a different inode's bytes either.
    std::fs::remove_file(store.blob_path(&object.sha256)).unwrap();
    std::fs::write(store.blob_path(&object.sha256), vec![0x32; 512]).unwrap();
    let lost = store
        .put_stream_held(
            &mut body.as_slice(),
            Some(&object),
            &Fault::default(),
            Some("resident"),
        )
        .unwrap();
    assert!(!lost.admitted && store.record_valid(&object.sha256).is_err());
    assert!(matches!(
        store.verify(&object.sha256).unwrap(),
        Verdict::CorruptRemoved { .. }
    ));
    let admitted = store
        .put_stream_held(
            &mut body.as_slice(),
            Some(&object),
            &Fault::default(),
            Some("resident"),
        )
        .unwrap();
    assert!(admitted.admitted);
    assert_eq!(store.record_valid(&object.sha256).unwrap().length, 512);
    assert_eq!(catalog.held_keys().unwrap().len(), 1);
    drop(db);
    drop(_writer);
    std::fs::remove_dir_all(store.root()).unwrap();
}

#[test]
fn public_whole_digest_and_segment_ids_remain_exact() {
    let store = store("whole");
    for length in [1024, limits::GRID_BYTES as usize + 4096] {
        let bytes = (0..length)
            .map(|i| ((i * 17) % 251) as u8)
            .collect::<Vec<_>>();
        let written = checkpoint::objectize(
            &store,
            "public whole",
            Dtype::U8,
            vec![length as u64],
            &mut bytes.as_slice(),
        )
        .unwrap();
        assert_eq!(written.whole, sha256::hex_digest(&bytes));
        let expected = bytes
            .chunks(limits::GRID_BYTES as usize)
            .map(ObjectRef::of)
            .collect::<Vec<_>>();
        assert_eq!(written.part.segments(), expected);
        assert_eq!(written.segments, expected.len());
        for object in expected {
            assert_eq!(
                store.record_valid(&object.sha256).unwrap().length,
                object.length
            );
        }
    }
    std::fs::remove_dir_all(store.root()).unwrap();
}

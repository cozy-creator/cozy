//! A live writer on a persistent disk syncs at its checkpoints, not per part, and still
//! resumes from its last checkpoint after the machine loses its page cache.

use super::*;
use crate::disk::DiskClass;
use crate::unsynced::testing::{from_an_earlier_boot, markers, persistent_root, tear};

const TENSORS: usize = 6;

fn declaration() -> Declaration {
    let mut declaration = super::tests::created_declaration();
    let prototype = declaration.components[0].add[0].clone();
    declaration.components[0].add = (0..TENSORS)
        .map(|index| {
            let mut tensor = prototype.clone();
            tensor.key = format!("weight_{index}");
            tensor.shape = vec![128];
            tensor.parts[0].shape = vec![128];
            tensor
        })
        .collect();
    declaration.order = (0..TENSORS)
        .map(|index| ("model".into(), format!("weight_{index}")))
        .collect();
    declaration.max_new_bytes = (TENSORS * 512) as u64;
    declaration
}

fn store(name: &str) -> (Store, Meta) {
    let store = Store::init(&persistent_root(name)).unwrap();
    assert_eq!(store.disk_class(), DiskClass::Persistent);
    let meta = Meta::open(&store).unwrap();
    (store, meta)
}

fn add(store: &Store, meta: &Meta, id: &str, session: u64, writer: &Begin, index: usize) -> Part {
    add_indexed_part(
        store,
        meta,
        id,
        session,
        &writer.additions,
        None,
        ("model", &format!("weight_{index}"), "value"),
        &mut [index as u8 + 1; 512].as_slice(),
    )
    .unwrap()
    .part
}

#[test]
fn a_power_loss_after_a_checkpoint_resumes_from_it_to_the_same_manifest() {
    let id = format!("sha256:{}", "6c".repeat(32));

    // The reference: one uninterrupted writer.
    let (clean, clean_meta) = store("group-commit-clean");
    let writer = begin(&clean, &clean_meta, &id, 1, declaration(), None).unwrap();
    let live = writer_store(&clean).unwrap();
    for index in 0..TENSORS {
        add(&live, &clean_meta, &id, 1, &writer, index);
    }
    let expected = commit(&live, &clean_meta, &id, 1).unwrap();
    finish_writer(&live).unwrap();
    release_guards(&clean_meta, Some(writer.writer_hold), writer.source_leases);
    assert!(
        markers(clean.root()).is_empty(),
        "a committed writer retires its epoch"
    );

    // A checkpoint after three parts; three more admitted unsynced; then the machine goes
    // down and the page cache holding those three is lost.
    let (crashed, meta) = store("group-commit-crash");
    let root = crashed.root().to_path_buf();
    let writer = begin(&crashed, &meta, &id, 1, declaration(), None).unwrap();
    let live = writer_store(&crashed).unwrap();
    assert_eq!(
        markers(&root).len(),
        1,
        "a persistent writer admits under one epoch"
    );
    for index in 0..3 {
        add(&live, &meta, &id, 1, &writer, index);
    }
    let (_, head) = checkpoint_progress(&live, &meta, &id, 1, "request", "model", None).unwrap();
    let lost: Vec<Part> = (3..TENSORS)
        .map(|index| add(&live, &meta, &id, 1, &writer, index))
        .collect();
    fence(&meta, &id, 1).unwrap();
    release_guards(&meta, Some(writer.writer_hold), writer.source_leases);
    drop(live);
    from_an_earlier_boot(&markers(&root).pop().unwrap());
    for segment in lost.iter().flat_map(Part::segments) {
        tear(&crashed.blob_path(&segment.sha256));
    }

    // Opening repairs the torn objects; the resumed writer keeps what its checkpoint made
    // durable and recomputes the rest.
    let reopened = Store::open(&root).unwrap();
    let meta = Meta::open(&reopened).unwrap();
    let head = head.head.expect("the checkpoint names a head");
    let writer = begin(&reopened, &meta, &id, 2, declaration(), Some(&head)).unwrap();
    assert_eq!(completed(&meta, &id, 2).unwrap().0.len(), 3);
    let live = writer_store(&reopened).unwrap();
    for index in 3..TENSORS {
        add(&live, &meta, &id, 2, &writer, index);
    }
    let resumed = commit(&live, &meta, &id, 2).unwrap();
    finish_writer(&live).unwrap();
    release_guards(&meta, Some(writer.writer_hold), writer.source_leases);
    assert_eq!(resumed.manifest, expected.manifest);
    assert!(markers(&root).is_empty());
    std::fs::remove_dir_all(clean.root()).unwrap();
    std::fs::remove_dir_all(&root).unwrap();
}

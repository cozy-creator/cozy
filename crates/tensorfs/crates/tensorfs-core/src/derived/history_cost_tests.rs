//! A derivation costs the same however many derivations the Store already retains (finding
//! 50): proving its source rooted walked every retained checkpoint chain, so each lane on a
//! conversion pod took longer than the one before.

use std::time::{Duration, Instant};

use super::tests::created_declaration;
use super::*;
use crate::repository::{Mutation, RepositoryName};
use crate::store::Fault;

const PARTS: usize = 32;
const CONVERSIONS: usize = 60;
const WINDOW: usize = 10;

/// A source-inheriting derivation that adds PARTS tensors and checkpoints after each, as a
/// quantization lane does: its source inspection, writer, chain and pending root.
fn convert(store: &Store, meta: &Meta, index: usize, source: &ObjectRef) -> Duration {
    let started = Instant::now();
    inspect_source(
        store,
        meta,
        source.clone(),
        vec!["model".into()],
        Vec::new(),
    )
    .unwrap();
    let prototype = created_declaration().components.remove(0).add.remove(0);
    let add: Vec<_> = (0..PARTS)
        .map(|part| TensorDeclaration {
            key: format!("added_{part}"),
            ..prototype.clone()
        })
        .collect();
    let mut order = vec![("model".to_string(), "weight".to_string())];
    order.extend(
        add.iter()
            .map(|tensor| ("model".to_string(), tensor.key.clone())),
    );
    let declaration = Declaration {
        sources: vec![Source {
            alias: "source".into(),
            manifest: source.clone(),
        }],
        components: vec![ComponentDeclaration {
            target: "model".into(),
            source: Some("source".into()),
            source_component: Some("model".into()),
            drop: Vec::new(),
            add,
        }],
        order,
        max_new_bytes: (PARTS * 2048) as u64,
        ..created_declaration()
    };
    let transaction = format!("sha256:{:064x}", index + 1);
    let writer = begin(store, meta, &transaction, 1, declaration, None).unwrap();
    let (mut cursor, mut head) = (None, None);
    for part in 0..PARTS {
        add_indexed_part(
            store,
            meta,
            &transaction,
            1,
            &writer.additions,
            cursor.as_mut(),
            ("model", &format!("added_{part}"), "value"),
            &mut [(index * PARTS + part) as u8; 2048].as_slice(),
        )
        .unwrap();
        let (_, next) = checkpoint_indexed(
            store,
            meta,
            &transaction,
            1,
            "lane",
            "model",
            head.as_ref(),
            &mut cursor,
        )
        .unwrap();
        head = next.head;
    }
    commit(store, meta, &transaction, 1).unwrap();
    release_guards(meta, Some(writer.writer_hold), writer.source_leases);
    started.elapsed()
}

#[test]
fn a_conversion_costs_the_same_after_many_retained_conversions() {
    let scratch = std::path::Path::new("/dev/shm");
    let root = if scratch.is_dir() {
        scratch.to_path_buf()
    } else {
        std::env::temp_dir()
    }
    .join(format!(
        "tensorfs-history-cost-{}-{}",
        std::process::id(),
        crate::meta::now_nanos_unique()
    ));
    let store = Store::init(&root).unwrap();
    let meta = Meta::open(&store).unwrap();

    // The source is a model checkpoint, rooted by its repository as a pulled model is.
    let source_id = format!("sha256:{}", "5a".repeat(32));
    let writer = begin(&store, &meta, &source_id, 1, created_declaration(), None).unwrap();
    add_part(
        &store,
        &meta,
        &source_id,
        1,
        ("model", "weight", "value"),
        &mut [0x21; 2048].as_slice(),
    )
    .unwrap();
    let source = commit(&store, &meta, &source_id, 1).unwrap().manifest;
    release_guards(&meta, Some(writer.writer_hold), writer.source_leases);
    store
        .apply_repository(
            None,
            &Mutation::PutCheckpoint {
                repo: RepositoryName::new("org", "source").unwrap(),
                manifest: source.clone(),
            },
            &Fault::default(),
        )
        .unwrap();

    let costs: Vec<Duration> = (0..CONVERSIONS)
        .map(|index| convert(&store, &meta, index, &source))
        .collect();
    let median = |window: &[Duration]| {
        let mut sorted = window.to_vec();
        sorted.sort();
        sorted[sorted.len() / 2]
    };
    let (first, last) = (
        median(&costs[..WINDOW]),
        median(&costs[CONVERSIONS - WINDOW..]),
    );
    let _ = std::fs::remove_dir_all(&root);
    // Retained history grew six-fold between the windows; the cost of one conversion may
    // not follow it. Before the fix the last window cost several times the first.
    assert!(
        last.as_secs_f64() < 2.0 * first.as_secs_f64(),
        "conversion cost grew with retained history: first {first:?}, last {last:?}"
    );
}

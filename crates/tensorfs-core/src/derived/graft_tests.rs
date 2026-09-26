use super::*;

fn produced(store: &Store, meta: &Meta, id: &str, byte: u8) -> ReceiptFacts {
    let writer = begin(
        store,
        meta,
        id,
        1,
        super::tests::created_declaration(),
        None,
    )
    .unwrap();
    add_part(
        store,
        meta,
        id,
        1,
        ("model", "weight", "value"),
        &mut vec![byte; 2048].as_slice(),
    )
    .unwrap();
    let receipt = commit(store, meta, id, 1).unwrap();
    release_guards(meta, Some(writer.writer_hold), writer.source_leases);
    receipt
}

fn fixture() -> (Store, Meta, ReceiptFacts, ReceiptFacts, Declaration) {
    let root = std::env::temp_dir().join(format!(
        "tensorfs-graft-{}",
        crate::meta::now_nanos_unique()
    ));
    let store = Store::init(&root).unwrap();
    let meta = Meta::open(&store).unwrap();
    let body = produced(&store, &meta, &format!("sha256:{}", "a1".repeat(32)), 0x11);
    let bank = produced(&store, &meta, &format!("sha256:{}", "a2".repeat(32)), 0x22);
    let mut declaration = super::tests::created_declaration();
    declaration.max_new_bytes = 0;
    declaration.sources = vec![
        Source {
            alias: "body".into(),
            manifest: body.manifest.clone(),
        },
        Source {
            alias: "bank".into(),
            manifest: bank.manifest.clone(),
        },
    ];
    let component = &mut declaration.components[0];
    component.source = Some("body".into());
    component.source_component = Some("model".into());
    component.add[0].key = "table".into();
    component.add[0].parts[0].source = Some(PartSource {
        source: "bank".into(),
        component: "model".into(),
        tensor: "weight".into(),
        role: "value".into(),
    });
    declaration.order.push(("model".into(), "table".into()));
    (store, meta, body, bank, declaration)
}

#[test]
fn graft_receipt_recovery_and_independent_custody_preserve_exact_segments() {
    let (store, meta, body, bank, mut declaration) = fixture();
    let encoded = declaration.canonical_work_bytes().unwrap();
    assert_eq!(
        Declaration::parse_text(std::str::from_utf8(&encoded).unwrap()).unwrap(),
        declaration
    );
    let id = format!("sha256:{}", "a3".repeat(32));
    let writer = begin(&store, &meta, &id, 1, declaration.clone(), None).unwrap();
    let sources = load_sources(&store, &declaration).unwrap();
    assert_eq!(source(&sources, "bank").unwrap().fact.components, ["model"]);
    let part = &source(&sources, "bank").unwrap().header.components[0].1[0]
        .1
        .parts[0]
        .1;
    let exact_part = part.clone();
    let objects = part.segments().to_vec();
    assert_eq!(
        add_part(
            &store,
            &meta,
            &id,
            1,
            ("model", "table", "value"),
            &mut [].as_slice()
        )
        .unwrap_err()
        .code,
        Code::TRANSACTION_CONFLICT
    );
    checkpoint_progress(&store, &meta, &id, 1, "graft", "result", None).unwrap();
    assert!(fence(&meta, &id, 1).unwrap());
    release_guards(&meta, Some(writer.writer_hold), writer.source_leases);
    dispose(&store, &meta, &body.transaction).unwrap();
    dispose(&store, &meta, &bank.transaction).unwrap();
    crate::gc::collect(store.root(), false).unwrap();
    let writer = begin(&store, &meta, &id, 2, declaration.clone(), None).unwrap();
    assert_eq!(
        commit(&store, &meta, &id, 1).unwrap_err().code,
        Code::WRITER_FENCED
    );
    // Crash window between durable manifest/root and native transaction acknowledgement.
    let connection =
        rusqlite::Connection::open(crate::catalog::Catalog::path(store.root())).unwrap();
    connection.execute_batch("CREATE TRIGGER fail_graft_commit BEFORE DELETE ON tensorfs_derived_transactions BEGIN SELECT RAISE(ABORT,'graft commit window'); END").unwrap();
    assert_eq!(
        commit(&store, &meta, &id, 2).unwrap_err().code,
        Code::IO_FAILED
    );
    connection
        .execute_batch("DROP TRIGGER fail_graft_commit")
        .unwrap();
    drop(connection);
    let receipt = commit(&store, &meta, &id, 2).unwrap();
    assert!(receipt.added_objects.is_empty());
    assert_eq!(receipt.inherited_payload_bytes, 4096);
    assert_eq!(receipt.inherited_payload_objects, 2);
    assert_eq!(receipt.declaration, declaration);
    let header = checkpoint::load_header(&store, &receipt.header).unwrap();
    assert_eq!(header.components[0].1[1].1.parts[0].1, exact_part);
    release_guards(&meta, Some(writer.writer_hold), writer.source_leases);
    let owner = format!("sha256:{}", "a4".repeat(32));
    let digest = ObjectRef::of(&crate::canon::write(&receipt.to_value())).id();
    retain_result(&store, &meta, &id, &digest, &owner).unwrap();
    dispose(&store, &meta, &id).unwrap();
    crate::gc::collect(store.root(), false).unwrap();
    store.read_manifest(&receipt.manifest).unwrap();
    for object in &objects {
        store.open_verified(&object.sha256).unwrap();
    }
    release_retention(&store, &meta, &id, &digest, &owner).unwrap();
    crate::gc::collect(store.root(), false).unwrap();
    for object in objects {
        assert!(!store.contains(&object.sha256));
    }
    std::fs::remove_dir_all(store.root()).unwrap();
}

#[test]
fn graft_refuses_semantic_substitution_and_undeclared_sources() {
    let (store, meta, _, _, declaration) = fixture();
    let id = format!("sha256:{}", "b1".repeat(32));
    for field in [
        "alias",
        "component",
        "tensor",
        "role",
        "missing_role",
        "dtype",
        "shape",
        "logical",
        "encoding",
    ] {
        let mut invalid = declaration.clone();
        let tensor = &mut invalid.components[0].add[0];
        let part = &mut tensor.parts[0];
        let selected = part.source.as_mut().unwrap();
        match field {
            "alias" => selected.source = "undeclared".into(),
            "component" => selected.component = "absent".into(),
            "tensor" => selected.tensor = "absent".into(),
            "role" => selected.role = "scale".into(),
            "missing_role" => {
                selected.role = "scale".into();
                part.role = "scale".into();
            }
            "dtype" => part.dtype = Dtype::Bf16,
            "shape" => part.shape = vec![256],
            "logical" => tensor.shape = vec![256],
            "encoding" => tensor.encoding = format!("sha256:{}", "ff".repeat(32)),
            _ => unreachable!(),
        }
        assert!(
            begin(&store, &meta, &id, 1, invalid, None).is_err(),
            "accepted {field}"
        );
        assert!(matches!(
            lookup(&store, &meta, &id).unwrap(),
            Lookup::Absent
        ));
    }
    let mut different = declaration.clone();
    different.components[0].add[0].parts[0]
        .source
        .as_mut()
        .unwrap()
        .tensor = "another".into();
    assert_ne!(
        different.canonical_work_bytes().unwrap(),
        declaration.clone().canonical_work_bytes().unwrap()
    );
    std::fs::remove_dir_all(store.root()).unwrap();
}

#[test]
fn source_projection_inherits_only_selected_tensors_without_new_bytes() {
    let (store, meta, _, bank, mut declaration) = fixture();
    declaration.sources.retain(|source| source.alias == "bank");
    declaration.components[0].source = None;
    declaration.components[0].source_component = None;
    declaration.order = vec![("model".into(), "table".into())];
    let id = format!("sha256:{}", "b2".repeat(32));
    let writer = begin(&store, &meta, &id, 1, declaration, None).unwrap();
    let receipt = commit(&store, &meta, &id, 1).unwrap();
    assert!(receipt.added_objects.is_empty());
    assert_eq!(receipt.inherited_payload_bytes, 2048);
    assert_eq!(receipt.sources[0].manifest, bank.manifest);
    release_guards(&meta, Some(writer.writer_hold), writer.source_leases);
    std::fs::remove_dir_all(store.root()).unwrap();
}

#[test]
fn pressure_keeps_cached_inputs_of_a_paused_derive_until_explicit_abandon() {
    use crate::ids::Doc;
    use crate::repository::{Mutation, RepositoryName};
    let (store, meta, body, bank, declaration) = fixture();
    let mut repos = Vec::new();
    for (name, receipt) in [("body", &body), ("bank", &bank)] {
        let repo = RepositoryName::new("downloaded", name).unwrap();
        let recorded = store
            .apply_cached_repository(
                None,
                &[Mutation::PutCheckpoint {
                    repo: repo.clone(),
                    manifest: receipt.manifest.clone(),
                }],
                &Default::default(),
            )
            .unwrap()
            .unwrap();
        assert!(crate::cache_roots::matches(&store, &repo, &recorded.canonical_bytes()).unwrap());
        repos.push(repo);
    }
    let id = format!("sha256:{}", "a4".repeat(32));
    let writer = begin(&store, &meta, &id, 1, declaration, None).unwrap();
    fence(&meta, &id, 1).unwrap();
    release_guards(&meta, Some(writer.writer_hold), writer.source_leases);
    // The producer's original obligations end; only the paused derive still needs inputs.
    dispose(&store, &meta, &body.transaction).unwrap();
    dispose(&store, &meta, &bank.transaction).unwrap();
    let held = crate::gc::collect_cached(store.root(), &[]).unwrap();
    assert_eq!(held.reclaimed_bytes, 0);
    assert!(repos
        .iter()
        .all(|repo| store.repository_path(repo).exists()));
    abandon(&store, &meta, &id).unwrap();
    assert!(
        crate::gc::collect_cached(store.root(), &[])
            .unwrap()
            .reclaimed_bytes
            > 0
    );
    assert!(repos
        .iter()
        .any(|repo| !store.repository_path(repo).exists()));
    std::fs::remove_dir_all(store.root()).unwrap();
}

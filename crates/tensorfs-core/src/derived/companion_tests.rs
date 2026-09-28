use super::*;
use crate::repository::Mutation;
use crate::repository::RepositoryName;
use crate::store::Fault;

fn fixture() -> (Store, Meta, String) {
    let root = std::env::temp_dir().join(format!(
        "tensorfs-companions-{}",
        crate::meta::now_nanos_unique()
    ));
    let store = Store::init(&root).unwrap();
    let meta = Meta::open(&store).unwrap();
    (store, meta, format!("sha256:{}", "c1".repeat(32)))
}

fn release(meta: &Meta, writer: Begin) {
    release_guards(meta, Some(writer.writer_hold), writer.source_leases);
}

fn declaration() -> Declaration {
    let mut d = super::tests::created_declaration();
    d.files = vec![
        ("LICENSE".into(), b"research licence\n".to_vec()),
        ("Notice".into(), b"Exact attribution\n".to_vec()),
    ];
    d.max_new_bytes += d
        .files
        .iter()
        .map(|(_, data)| data.len() as u64)
        .sum::<u64>();
    d
}

fn write_weight(store: &Store, meta: &Meta, id: &str) -> Part {
    add_part(
        store,
        meta,
        id,
        1,
        ("model", "weight", "value"),
        &mut vec![0x31; 2048].as_slice(),
    )
    .unwrap()
    .part
}

#[test]
fn companion_intent_resumes_and_published_checkpoint_checks_out_real_files() {
    let (store, meta, id) = fixture();
    let mut d = declaration();
    let encoded = d.canonical_work_bytes().unwrap();
    assert_eq!(
        Declaration::parse_text(std::str::from_utf8(&encoded).unwrap()).unwrap(),
        d
    );
    let writer = begin(&store, &meta, &id, 1, d.clone(), None).unwrap();
    let part = write_weight(&store, &meta, &id);
    let (_, progress) =
        checkpoint_progress(&store, &meta, &id, 1, "request", "model", None).unwrap();
    fence(&meta, &id, 1).unwrap();
    release(&meta, writer);
    crate::gc::collect(store.root(), false).unwrap();
    let mut changed = d.clone();
    changed.files[0].1 = b"changed licence\n".to_vec();
    assert_eq!(
        begin(&store, &meta, &id, 2, changed, progress.head.as_ref())
            .err()
            .unwrap()
            .code,
        Code::TRANSACTION_CONFLICT
    );
    let writer = begin(&store, &meta, &id, 2, d.clone(), progress.head.as_ref()).unwrap();
    assert_eq!(completed(&meta, &id, 2).unwrap().0.len(), 1);
    let receipt = commit(&store, &meta, &id, 2).unwrap();
    release(&meta, writer);
    assert!(receipt
        .added_objects
        .contains(&ObjectRef::of(&d.files[0].1)));
    let manifest = checkpoint::load_manifest(&store, &receipt.manifest).unwrap();
    assert_eq!(
        manifest
            .entries()
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>(),
        ["LICENSE", "Notice", "model.cozytensors"]
    );
    let header = checkpoint::load_header(&store, &receipt.header).unwrap();
    assert_eq!(header.components[0].1[0].1.parts[0].1, part);
    // Documents participate in normal model-closure fetching as well as ordinary checkout.
    let walk = checkpoint::walk_cozytensors(&store, &manifest).unwrap();
    assert!(walk
        .objects
        .iter()
        .any(|r| r.kind == "model_asset" && r.obj == ObjectRef::of(&d.files[1].1)));
    store
        .apply_repository(
            None,
            &Mutation::PutCheckpoint {
                repo: RepositoryName::new("research", "reference-image").unwrap(),
                manifest: receipt.manifest.clone(),
            },
            &Fault::default(),
        )
        .unwrap();
    dispose(&store, &meta, &id).unwrap();
    crate::gc::collect(store.root(), false).unwrap();
    let out = store.root().join("published-readback");
    crate::project::checkout(&store, &manifest, &out, true).unwrap();
    assert_eq!(std::fs::read(out.join("LICENSE")).unwrap(), d.files[0].1);
    assert_eq!(std::fs::read(out.join("Notice")).unwrap(), d.files[1].1);
    std::fs::remove_dir_all(store.root()).unwrap();
}

#[test]
fn ordinary_companions_survive_derivation_without_copying_tensor_backing_files() {
    let (store, meta, id) = fixture();
    let mut source_declaration = declaration();
    // Same exact ObjectRef as the tensor: the named asset role must survive deduplication.
    source_declaration
        .files
        .push(("Tensor-shaped.txt".into(), vec![0x31; 2048]));
    source_declaration.max_new_bytes += 2048;
    let writer = begin(&store, &meta, &id, 1, source_declaration, None).unwrap();
    let part = write_weight(&store, &meta, &id);
    let source = commit(&store, &meta, &id, 1).unwrap();
    release(&meta, writer);
    let mut manifest = Draft::of(&checkpoint::load_manifest(&store, &source.manifest).unwrap());
    manifest.entries.push((
        "tensor-backing.bin".into(),
        Entry::File(part.segments()[0].clone()),
    ));
    manifest.entries.sort_by(|a, b| a.0.cmp(&b.0));
    let source_manifest = store.put_manifest(&manifest.seal().unwrap()).unwrap().obj;
    store
        .apply_repository(
            None,
            &Mutation::PutCheckpoint {
                repo: RepositoryName::new("research", "source").unwrap(),
                manifest: source_manifest.clone(),
            },
            &Fault::default(),
        )
        .unwrap();
    let next = format!("sha256:{}", "c2".repeat(32));
    let mut d = super::tests::created_declaration();
    d.sources = vec![Source {
        alias: "source".into(),
        manifest: source_manifest,
    }];
    d.components[0].source = Some("source".into());
    d.components[0].source_component = Some("model".into());
    d.components[0].add.clear();
    d.max_new_bytes = 0;
    let writer = begin(&store, &meta, &next, 1, d.clone(), None).unwrap();
    let derived = commit(&store, &meta, &next, 1).unwrap();
    release(&meta, writer);
    let manifest = checkpoint::load_manifest(&store, &derived.manifest).unwrap();
    assert_eq!(
        manifest
            .entries()
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>(),
        [
            "LICENSE",
            "Notice",
            "Tensor-shaped.txt",
            "model.cozytensors"
        ]
    );
    assert!(derived.added_objects.is_empty());
    let header = checkpoint::load_header(&store, &derived.header).unwrap();
    assert_eq!(header.components[0].1[0].1.parts[0].1, part);
    // Two source documents may only disagree when the caller explicitly chooses new bytes.
    let mut sources = load_sources(&store, &d).unwrap();
    let mut other = SourceData {
        fact: sources[0].fact.clone(),
        manifest: sources[0].manifest.clone(),
        header: sources[0].header.clone(),
    };
    let mut draft = Draft::of(&other.manifest);
    draft.entries[0].1 = Entry::File(ObjectRef::of(b"different licence"));
    other.manifest = draft.seal().unwrap();
    sources.push(other);
    assert_eq!(
        companion_files(&d, &sources).unwrap_err().code,
        Code::TRANSACTION_CONFLICT
    );
    d.files
        .push(("LICENSE".into(), b"explicit licence".to_vec()));
    assert_eq!(
        companion_files(&d, &sources).unwrap()["LICENSE"],
        Entry::File(ObjectRef::of(b"explicit licence"))
    );
    std::fs::remove_dir_all(store.root()).unwrap();
}

#[test]
fn a_verified_store_object_is_a_companion_by_reference_and_never_a_new_byte() {
    let (store, meta, id) = fixture();
    let vocab = vec![0x7b; 256 << 10]; // over the inline bound: references have none
    let object = store
        .put_stream(
            &mut vocab.as_slice(),
            Some(&ObjectRef::of(&vocab)),
            &Fault::default(),
        )
        .unwrap()
        .obj;
    let mut d = super::tests::created_declaration();
    d.objects = vec![("tokenizer/vocab.json".into(), object.clone())];
    let encoded = d.canonical_work_bytes().unwrap();
    assert_eq!(
        Declaration::parse_text(std::str::from_utf8(&encoded).unwrap()).unwrap(),
        d
    );
    let writer = begin(&store, &meta, &id, 1, d.clone(), None).unwrap();
    write_weight(&store, &meta, &id);
    let receipt = commit(&store, &meta, &id, 1).unwrap();
    release(&meta, writer);
    assert!(!receipt.added_objects.contains(&object));
    let header = checkpoint::load_header(&store, &receipt.header).unwrap();
    let (name, asset) = &header.assets[0];
    assert_eq!(
        (name.as_str(), &asset.segments),
        ("tokenizer/vocab.json", &vec![object])
    );
    let mut read = Vec::new();
    checkpoint::read_asset(&store, name, asset, &mut read).unwrap();
    assert_eq!(read, vocab);

    // An object this Store does not hold is refused at commit, never published.
    let next = format!("sha256:{}", "c3".repeat(32));
    d.objects = vec![("tokenizer/vocab.json".into(), ObjectRef::of(b"absent"))];
    let writer = begin(&store, &meta, &next, 1, d, None).unwrap();
    write_weight(&store, &meta, &next);
    assert!(commit(&store, &meta, &next, 1).is_err());
    release(&meta, writer);
    std::fs::remove_dir_all(store.root()).unwrap();
}

#[test]
fn companion_paths_size_count_and_quotas_are_native_refusals() {
    for path in [
        "../LICENSE",
        "/LICENSE",
        "model.cozytensors",
        "model.cozytensors/LICENSE",
        "a//b",
        "a\\b",
    ] {
        let mut d = declaration();
        d.files = vec![(path.into(), b"x".to_vec())];
        assert!(d.normalize_and_validate().is_err(), "accepted {path}");
    }
    let mut d = declaration();
    d.files = vec![
        ("a".into(), vec![]),
        ("a-other".into(), vec![]),
        ("a/b".into(), vec![]),
    ];
    assert_eq!(
        d.normalize_and_validate().unwrap_err().code,
        Code::PATH_ILLEGAL
    );
    d.files = vec![("a".into(), vec![1; MAX_COMPANION_BYTES + 1])];
    assert_eq!(d.normalize_and_validate().unwrap_err().code, Code::SIZE_CAP);
    d.files = (0..17).map(|i| (format!("doc{i}"), vec![])).collect();
    assert_eq!(
        d.normalize_and_validate().unwrap_err().code,
        Code::COUNT_CAP
    );
    d.files = (0..9)
        .map(|i| (format!("doc{i}"), vec![1; MAX_COMPANION_BYTES]))
        .collect();
    assert_eq!(d.normalize_and_validate().unwrap_err().code, Code::SIZE_CAP);
    d = declaration();
    d.max_new_bytes = 2048;
    assert_eq!(
        d.normalize_and_validate().unwrap_err().code,
        Code::QUOTA_EXHAUSTED
    );
    d = super::tests::created_declaration();
    let bytes = d.canonical_work_bytes().unwrap();
    assert!(!String::from_utf8(bytes).unwrap().contains("\"files\""));
}

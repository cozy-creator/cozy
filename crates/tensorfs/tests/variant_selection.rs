#![cfg(feature = "store")]
use std::{fs,path::Path};
use tensorfs::{Client,Config,Open};

#[tokio::test]
async fn local_and_static_origin_select_nested_headers_without_tensor_payloads() {
    struct Scratch(std::path::PathBuf);
    impl Scratch { fn path(&self)->&Path { &self.0 } }
    impl Drop for Scratch { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
    let scratch = Scratch(Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("variant-selection-{}",std::process::id())));
    fs::create_dir_all(scratch.path()).unwrap();
    let folder = scratch.path().join("model");
    fs::create_dir_all(folder.join("nested")).unwrap();
    let vectors = Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance/vectors/select");
    fs::copy(vectors.join("int8.cbor"),folder.join("nested/banana.cozytensors")).unwrap();
    fs::copy(vectors.join("bf16.cbor"),folder.join("model.int8.cozytensors")).unwrap();
    let mut config = Config::from_env();
    config.cache_dir = scratch.path().join("cache"); config.endpoint = scratch.path().join("origin").to_string_lossy().into(); config.token=None; config.mirror=None;
    let client = Client::new(config).unwrap();
    let gpu = Open { bits:Some(8),gpus:Some(vec!["8.6".into()]),..Default::default() };
    let selected = client.open(folder.to_str().unwrap(),&gpu).await.unwrap();
    assert_eq!(selected.variant(),Some("nested/banana.cozytensors"));
    let cpu = Open {gpus:Some(vec![]),bits:Some(16),..Default::default()};
    let selected = client.open(folder.to_str().unwrap(),&cpu).await.unwrap();
    assert_eq!(selected.variant(),Some("model.int8.cozytensors"));
    let legacy = Open {variant:Some("int8".into()),..cpu.clone()};
    assert_eq!(client.open(folder.to_str().unwrap(),&legacy).await.unwrap().id(),selected.id());
    // A static repo has only the tree and headers. No chunks exist anywhere.
    let store = tensorfs::store::Store::open(scratch.path().join("origin")).unwrap();
    let mut tree=tensorfs::format::Tree::default();
    for name in ["nested/banana.cozytensors","model.int8.cozytensors"] {
        tree.files.insert(name.into(),store.put_bytes(&fs::read(folder.join(name)).unwrap()).unwrap());
    }
    let root=store.put_tree(&tree).unwrap();
    let mut repo=tensorfs::format::Repo::new("acme/profile");
    repo.tags.push(tensorfs::format::Tag{name:"1.0".into(),commit:root.clone(),public:true,yanked:false});
    let at=store.root().join(repo.key());fs::create_dir_all(at.parent().unwrap()).unwrap();fs::write(at,repo.to_json().unwrap()).unwrap();
    let selected=client.open("acme/profile",&gpu).await.unwrap();
    assert_eq!(selected.variant(),Some("nested/banana.cozytensors"));
    let projected=scratch.path().join("projected");
    let reference="acme/profile@1.0/int8".parse().unwrap();
    let metadata=tensorfs::Download{selection:tensorfs::Selection::Components(vec![]),..Default::default()};
    client.materialize(&reference,&metadata,Some(&projected),false).await.unwrap();
    assert!(projected.join("model.int8.cozytensors").is_file());
    assert!(!projected.join("nested/banana.cozytensors").exists());
    let snapshot=scratch.path().join("snapshot");
    store.snapshot(&root,&snapshot,Some(&["int8".into()])).unwrap();
    assert!(snapshot.join("model.int8.cozytensors").is_file());
    let invalid=Open {bits:Some(7),..gpu};
    assert!(client.open("acme/profile",&invalid).await.is_err());
}

#[test]
fn pinned_header_aliases_resolve_before_agreement_is_checked() {
    use tensorfs::format::{Tree,ObjectRef};
    let first=ObjectRef::of(b"first");let second=ObjectRef::of(b"second");
    let mut tree=Tree::default();tree.files.insert("model.z.cozytensors".into(),first);
    let cited=Some("model.z.cozytensors".into());
    assert_eq!(tensorfs::fetch::pick_variant(&tree,None,cited.clone(),Some("z")).unwrap(),cited);
    tree.files.insert("z.cozytensors".into(),second);
    assert!(tensorfs::fetch::pick_variant(&tree,None,cited,Some("z")).is_err(),"an exact competing path must not be redirected to the legacy alias");
}

#[tokio::test]
async fn stored_variant_inspection_does_not_require_an_execution_route() {
    struct Scratch(std::path::PathBuf);
    impl Drop for Scratch { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
    let root = Scratch(Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("variant-inspection-{}", std::process::id())));
    let folder = root.0.join("model");
    fs::create_dir_all(folder.join("nested")).unwrap();
    let vector = Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance/vectors/select/future.cbor");
    fs::copy(vector, folder.join("nested/unknown.cozytensors")).unwrap();
    let mut config = Config::from_env();
    config.cache_dir = root.0.join("cache"); config.token = None; config.mirror = None;
    let client = Client::new(config).unwrap();
    let cpu = Open { gpus: Some(vec![]), ..Default::default() };
    assert!(client.open(folder.to_str().unwrap(), &cpu).await.is_err(), "unknown execution remains refused");
    let inspected = client.open_variant(folder.to_str().unwrap(), &Open::default()).await.unwrap();
    assert_eq!(inspected.variant(), Some("nested/unknown.cozytensors"));
    assert!(client.open_variant(folder.to_str().unwrap(), &cpu).await.is_err(), "explicit constraints cannot be ignored");
    fs::copy(folder.join("nested/unknown.cozytensors"), folder.join("second.cozytensors")).unwrap();
    assert!(client.open_variant(folder.to_str().unwrap(), &Open::default()).await.is_err(), "data inspection must not guess among variants");
    let named = Open { variant: Some("nested/unknown.cozytensors".into()), ..Default::default() };
    assert_eq!(client.open_variant(folder.to_str().unwrap(), &named).await.unwrap().id(), inspected.id());
}

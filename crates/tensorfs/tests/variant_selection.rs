#![cfg(feature = "store")]
use std::{fs,path::Path};
use tensorfs::{Client,Config,Kind,Open};

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
    let gpu = Open { lane:Some(8),gpus:Some(vec!["8.6".into()]),..Default::default() };
    let selected = client.open(folder.to_str().unwrap(),&gpu).await.unwrap();
    assert_eq!(selected.variant(),Some("nested/banana.cozytensors"));
    let cpu = Open {gpus:Some(vec![]),lane:Some(16),..Default::default()};
    let selected = client.open(folder.to_str().unwrap(),&cpu).await.unwrap();
    assert_eq!(selected.variant(),Some("model.int8.cozytensors"));
    let inspect = std::process::Command::new(env!("CARGO_BIN_EXE_tfs"))
        .env("CUDA_VISIBLE_DEVICES", "")
        .args(["--json", "--store", scratch.path().join("cli-cache").to_str().unwrap(), "inspect", folder.to_str().unwrap()])
        .output().unwrap();
    assert!(inspect.status.success(), "{}", String::from_utf8_lossy(&inspect.stdout));
    let inspected: serde_json::Value = serde_json::from_slice(&inspect.stdout).unwrap();
    assert_eq!(inspected["variant"], "model.int8.cozytensors", "data commands retain the shared choice when it fits");
    let exact = Open {variant:Some("model.int8.cozytensors".into()),..cpu.clone()};
    assert_eq!(client.open(folder.to_str().unwrap(),&exact).await.unwrap().id(),selected.id());
    let word = Open {variant:Some("int8".into()),..cpu.clone()};
    assert!(client.open(folder.to_str().unwrap(),&word).await.is_err(),"a lane word is no path");
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
    let reference="acme/profile@1.0/model.int8.cozytensors".parse().unwrap();
    let metadata=tensorfs::Download{selection:tensorfs::Selection::Components(vec![]),..Default::default()};
    client.materialize(&reference,&metadata,Some(&projected),false).await.unwrap();
    assert!(projected.join("model.int8.cozytensors").is_file());
    assert!(!projected.join("nested/banana.cozytensors").exists());
    let snapshot=scratch.path().join("snapshot");
    store.snapshot(&root,&snapshot,Some(&["model.int8.cozytensors".into()])).unwrap();
    assert!(snapshot.join("model.int8.cozytensors").is_file());
    let invalid=Open {lane:Some(7),..gpu};
    assert!(client.open("acme/profile",&invalid).await.is_err());
}

/// A pin is the variant's exact path: a bare lane word names no variant, whatever files exist.
#[test]
fn a_pin_is_an_exact_path() {
    use tensorfs::format::{Tree,ObjectRef};
    let first=ObjectRef::of(b"first");
    let mut tree=Tree::default();tree.files.insert("model.z.cozytensors".into(),first);
    let cited=Some("model.z.cozytensors".into());
    assert_eq!(tensorfs::fetch::pick_variant(&tree,None,cited.clone(),Some("model.z.cozytensors")).unwrap(),cited);
    assert!(tensorfs::fetch::pick_variant(&tree,None,None,Some("z")).is_err(),"a lane word is no path");
    assert!(tree.header("z").is_none() && tree.header("model.z").is_none());
}

/// A sole 8-bit release (spec §7): a CPU read takes it; a GPU that runs none of it, and 16 bits
/// asked, are refused naming what it publishes. `tfs` reads on the CPU: it inspects even an
/// encoding no GPU runs, which a GPU read refuses.
#[tokio::test]
async fn a_sole_8_bit_release_serves_the_cpu_and_refusals_name_it() {
    struct Scratch(std::path::PathBuf);
    impl Drop for Scratch { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
    let root = Scratch(Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("variant-sole-{}", std::process::id())));
    let vectors = Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance/vectors/select");
    let origin = tensorfs::store::Store::open(root.0.join("origin")).unwrap();
    let mut tree = tensorfs::format::Tree::default();
    tree.files.insert("weights/q.cozytensors".into(), origin.put_bytes(&fs::read(vectors.join("int8.cbor")).unwrap()).unwrap());
    let commit = origin.put_tree(&tree).unwrap();
    let mut repo = tensorfs::format::Repo::new("acme/int8");
    repo.tags.push(tensorfs::format::Tag { name: "1.0".into(), commit, public: true, yanked: false });
    let at = origin.root().join(repo.key());
    fs::create_dir_all(at.parent().unwrap()).unwrap();
    fs::write(at, repo.to_json().unwrap()).unwrap();
    let mut config = Config::from_env();
    config.cache_dir = root.0.join("cache"); config.endpoint = root.0.join("origin").to_string_lossy().into(); config.token = None; config.mirror = None;
    let client = &Client::new(config).unwrap();
    let cpu = Open { gpus: Some(vec![]), ..Default::default() };
    assert_eq!(client.open("acme/int8", &cpu).await.unwrap().variant(), Some("weights/q.cozytensors"));
    let refused = |gpus: &[&str], lane| {
        let open = Open { gpus: Some(gpus.iter().map(|cc| cc.to_string()).collect()), lane, ..Default::default() };
        async move { client.open("acme/int8", &open).await.err().map(|e| (e.kind, e.detail)) }
    };
    let published = "acme/int8@1.0 has no variant that works; it publishes weights/q.cozytensors (int8)";
    assert_eq!(refused(&["7.5"], None).await, Some((Kind::NotFound,
        format!("precision 8 on sm75: {published} needs sm80+ except sm103 for int8.rowwise"))));
    assert_eq!(refused(&[], Some(16)).await, Some((Kind::NotFound, format!("precision 16 on no GPU: {published} is not 16-bit"))));
    assert_eq!(refused(&["8.9"], None).await, None, "a GPU that runs it takes it");
    // Its header by id is its sole candidate: a GPU-less read opens it, asked bits or GPUs that
    // cannot use it refuse.
    let header = format!("sha256:{}", tree.files["weights/q.cozytensors"].sha256);
    let by_id = |gpus: Vec<String>, lane| Open { gpus: Some(gpus), lane, ..Default::default() };
    assert_eq!(client.open(&header, &by_id(vec![], None)).await.unwrap().id().id(), header);
    assert_eq!(client.open(&header, &by_id(vec!["7.5".into()], None)).await.err().map(|e| e.kind), Some(Kind::NotFound));
    assert_eq!(client.open(&header, &by_id(vec![], Some(16))).await.err().map(|e| e.kind), Some(Kind::NotFound));

    let folder = root.0.join("model");
    fs::create_dir_all(folder.join("nested")).unwrap();
    fs::copy(vectors.join("future.cbor"), folder.join("nested/unknown.cozytensors")).unwrap();
    let gpu = Open { gpus: Some(vec!["9.0".into()]), ..Default::default() };
    assert_eq!(client.open(folder.to_str().unwrap(), &gpu).await.err().map(|e| e.kind), Some(Kind::NotFound));
    let inspect = std::process::Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["--json", "--store", root.0.join("cli-cache").to_str().unwrap(), "inspect", folder.to_str().unwrap()])
        .output().unwrap();
    assert!(inspect.status.success(), "{}", String::from_utf8_lossy(&inspect.stdout));
    let inspected: serde_json::Value = serde_json::from_slice(&inspect.stdout).unwrap();
    assert_eq!(inspected["variant"], "nested/unknown.cozytensors");
}

/// An exact id opens by the one rule on a GPU-less host (cozy-runtime run 39735798532): a commit
/// whose sole variant is fp8 opens offline by its commit id and by its header id. A present GPU
/// that runs no fp8 is refused, and so are 16 bits asked.
#[tokio::test]
async fn an_exact_ids_sole_variant_opens_on_a_gpu_less_host() {
    struct Scratch(std::path::PathBuf);
    impl Drop for Scratch { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
    let root = Scratch(Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("variant-exact-{}", std::process::id())));
    let cache = tensorfs::store::Store::open(root.0.join("cache")).unwrap();
    let vectors = Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance/vectors/select");
    let header = cache.put_bytes(&fs::read(vectors.join("fp8.cbor")).unwrap()).unwrap();
    let mut tree = tensorfs::format::Tree::default();
    tree.files.insert("out/quantized.cozytensors".into(), header.clone());
    let commit = cache.put_tree(&tree).unwrap();
    let mut config = Config::from_env();
    config.cache_dir = root.0.join("cache"); config.endpoint = "http://127.0.0.1:1".into(); config.token = None; config.mirror = None;
    let client = Client::new(config).unwrap();
    let open = |gpus: &[&str], lane| Open { gpus: Some(gpus.iter().map(|cc| cc.to_string()).collect()), lane, local_files_only: true, ..Default::default() };
    for id in [commit.id(), header.id()] {
        let opened = client.open(&id, &open(&[], None)).await.unwrap_or_else(|e| panic!("{id}: {e}"));
        assert_eq!(opened.id(), &header, "{id}");
        assert_eq!(client.open(&id, &open(&["7.5"], None)).await.err().map(|e| e.kind), Some(Kind::NotFound), "{id} on sm75");
        assert_eq!(client.open(&id, &open(&[], Some(16))).await.err().map(|e| e.kind), Some(Kind::NotFound), "{id} at 16 bits");
    }
}

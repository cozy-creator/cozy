//! Normal checkpoint access stays in the shared store; directory exports are explicit.
#![cfg(feature = "store")]

use std::fs::{self, File, TryLockError};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use tensorfs::format::{Body, Component, Dtype, Header, Location, ObjectRef, Tensor, Tree};
use tensorfs::store2::{Checkpoint, Store};
use tensorfs::{Client, Config, Fetch, Open};

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
                "primary-store-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create scratch: {error}"),
            }
        }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn seed(store: &Store) -> (ObjectRef, ObjectRef) {
    let bytes = store.put_bytes(b"shared bytes").unwrap();
    let asset = store.put_bytes(b"{\"tokenizer\":true}").unwrap();
    let header = Header::new(
        vec![bytes.clone()],
        None,
        vec![Component {
            name: "model".into(),
            tensors: vec![Tensor {
                key: "weight".into(),
                dtype: Dtype::U8,
                shape: vec![bytes.length],
                body: Body::Data(Location {
                    chunk: 0,
                    offset: 0,
                    length: bytes.length,
                }),
            }],
            assets: [("tokenizer.json".into(), asset)].into(),
            ..Default::default()
        }],
        Default::default(),
    )
    .unwrap();
    let header = store.put_bytes(&header.to_cbor().unwrap()).unwrap();
    let mut tree = Tree::default();
    tree.set_variant("main", header.clone());
    (header, store.put_tree(&tree).unwrap())
}

fn no_compatibility_directories(store: &Store) {
    for area in ["views", "snapshots"] {
        assert!(
            !store.root().join(area).exists(),
            "normal checkpoint access created {area}"
        );
    }
}

#[tokio::test]
async fn normal_open_read_and_fetch_do_not_create_compatibility_directories() {
    let scratch = Scratch::new();
    let store = Store::open(scratch.0.join("store")).unwrap().with_cache();
    let (header, tree) = seed(&store);
    no_compatibility_directories(&store);

    let raw = store.checkpoint(&header).unwrap();
    assert_eq!(raw.read("model", "weight").unwrap(), b"shared bytes");
    no_compatibility_directories(&store);
    let variant = store.variant(&tree, Some("main")).unwrap();
    assert_eq!(
        variant.asset("model", "tokenizer.json").unwrap(),
        b"{\"tokenizer\":true}"
    );
    no_compatibility_directories(&store);

    let mut config = Config::from_env();
    config.cache_dir = store.root().to_path_buf();
    config.endpoint = format!("file://{}", scratch.0.join("unused-endpoint").display());
    config.token = None;
    config.mirror = None;
    let client = Client::new(config).unwrap();
    let checkpoint = client
        .open(
            &tree.id(),
            &Open {
                variant: Some("main".into()),
                local_files_only: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    checkpoint.fetch(&Fetch::default()).await.unwrap();
    assert_eq!(
        checkpoint
            .component("model")
            .unwrap()
            .read("weight")
            .unwrap(),
        b"shared bytes"
    );
    assert!(checkpoint.missing(&Default::default()).unwrap().is_empty());
    no_compatibility_directories(&store);
}

#[test]
fn explicit_cached_folder_retains_its_existing_directory() {
    let scratch = Scratch::new();
    let store = Store::open(scratch.0.join("store")).unwrap().with_cache();
    let (_, tree) = seed(&store);
    let directory = store.root().join("snapshots").join(&tree.sha256);
    store.snapshot(&tree, &directory, None).unwrap();
    assert!(!fs::read(directory.join("model.main.cozytensors"))
        .unwrap()
        .is_empty());
    let checkpoint = Checkpoint::folder(&directory, Some("main"), Some(&store)).unwrap();
    let fence = directory
        .parent()
        .unwrap()
        .join(format!(".{}.tensorfs-lock", tree.sha256));
    let file = File::open(fence).unwrap();
    assert!(matches!(file.try_lock(), Err(TryLockError::WouldBlock)));
    assert_eq!(checkpoint.read("model", "weight").unwrap(), b"shared bytes");
    drop(checkpoint);
    file.try_lock().unwrap();
    assert!(
        !store.root().join("views").exists(),
        "opening an explicit snapshot created an unused views area"
    );
}

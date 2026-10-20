//! Normal checkpoint access stays in the shared store; directory exports are explicit.
#![cfg(feature = "store")]

use std::fs::{self, File, TryLockError};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use tensorfs::format::{Body, Component, Dtype, Header, Location, ObjectRef, Tensor, Tree};
use tensorfs::store2::{Checkpoint, Store};
use tensorfs::{Client, Config, Download, Format, Kind, Open, Ref};

const README: &[u8] = b"known repository file outside the checkpoint";

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
    tree.files.insert("README.md".into(), ObjectRef::of(README));
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
                lazy: false,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    // Normal checkpoint access must not reopen the enclosing tree or consult its README.
    fs::remove_file(store.path(&tree).unwrap()).unwrap();
    checkpoint.download(&Download::default()).await.unwrap();
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
    store.put_bytes(README).unwrap();
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

struct Http {
    url: String,
    requests: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Http {
    fn new(root: PathBuf, repo_status: u16) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let (ending, seen) = (Arc::clone(&stop), Arc::clone(&requests));
        let worker = thread::spawn(move || {
            while !ending.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => Self::serve(stream, &root, repo_status, &seen),
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(error) => panic!("accept: {error}"),
                }
            }
        });
        Self {
            url,
            requests,
            stop,
            worker: Some(worker),
        }
    }

    fn serve(mut stream: TcpStream, root: &Path, repo_status: u16, seen: &Mutex<Vec<String>>) {
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut request = Vec::new();
        let mut byte = [0];
        while !request.ends_with(b"\r\n\r\n") {
            if stream.read(&mut byte).unwrap_or(0) == 0 {
                return;
            }
            request.push(byte[0]);
            assert!(request.len() < 32_768);
        }
        let request = String::from_utf8(request).unwrap();
        let mut first = request.lines().next().unwrap().split_whitespace();
        let method = first.next().unwrap();
        let path = first.next().unwrap();
        seen.lock().unwrap().push(path.to_owned());
        assert!(!path.contains(".."));
        let (status, bytes) =
            if path.trim_start_matches('/') == ObjectRef::of(README).key() && repo_status != 200 {
                (repo_status, Vec::new())
            } else {
                match fs::read(root.join(path.trim_start_matches('/'))) {
                    Ok(bytes) => (200, bytes),
                    Err(_) => (404, Vec::new()),
                }
            };
        let response = format!(
            "HTTP/1.1 {status} test\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            bytes.len()
        );
        let _ = stream.write_all(response.as_bytes());
        if method != "HEAD" {
            let _ = stream.write_all(&bytes);
        }
    }

    fn repo_requests(&self) -> usize {
        let path = format!("/{}", ObjectRef::of(README).key());
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|seen| **seen == path)
            .count()
    }
}

impl Drop for Http {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.worker.take().unwrap().join().unwrap();
    }
}

#[tokio::test]
async fn explicit_export_fetches_known_repo_files_and_never_omits_refused_files() {
    for status in [200, 404, 403] {
        let scratch = Scratch::new();
        let source = Store::open(scratch.0.join("source")).unwrap();
        let (header, tree) = seed(&source);
        source.put_bytes(README).unwrap();
        let server = Http::new(source.root().to_path_buf(), status);
        let cache = scratch.0.join("cache");
        let mut config = Config::from_env();
        config.cache_dir = cache.clone();
        config.endpoint = server.url.clone();
        config.token = None;
        config.mirror = None;
        let client = Client::new(config).unwrap();
        let checkpoint = client
            .open(
                &tree.id(),
                &Open {
                    variant: Some("main".into()),
                    lazy: false,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(checkpoint.missing(&Default::default()).unwrap().is_empty());
        assert_eq!(
            server.repo_requests(),
            0,
            "ordinary eager open fetched a repo file"
        );
        assert!(!cache.join(ObjectRef::of(README).key()).exists());
        for area in ["views", "snapshots"] {
            assert!(!cache.join(area).exists());
        }

        let invalid = scratch.0.join("invalid-export");
        assert_eq!(
            checkpoint
                .materialize(&invalid, Format::Cozytensors, Some(Dtype::F32))
                .await
                .unwrap_err()
                .kind,
            Kind::Invalid
        );
        assert_eq!(server.repo_requests(), 0);
        assert!(!invalid.exists());
        let requests_before = server.requests.lock().unwrap().len();
        let invalid = scratch.0.join("invalid-safetensors");
        let error = checkpoint
            .materialize(&invalid, Format::Safetensors, Some(Dtype::I8))
            .await
            .unwrap_err();
        assert_eq!(error.kind, Kind::Invalid);
        assert_eq!(error.detail, "dequantize to I8: one of F32, BF16, F16");
        assert_eq!(server.requests.lock().unwrap().len(), requests_before);
        assert!(!invalid.exists());

        let destination = scratch.0.join("export");
        let result = checkpoint
            .materialize(&destination, Format::Cozytensors, None)
            .await;
        assert!(
            server.repo_requests() > 0,
            "explicit export did not prepare its repo file"
        );
        if status != 200 {
            assert_eq!(
                result.unwrap_err().kind,
                if status == 403 {
                    Kind::Denied
                } else {
                    Kind::NotFound
                }
            );
            assert!(
                !destination.exists(),
                "failed export wrote a partial directory"
            );
            continue;
        }
        result.unwrap();
        assert_eq!(fs::read(destination.join("README.md")).unwrap(), README);
        assert_eq!(
            ObjectRef::of(&fs::read(destination.join("model.main.cozytensors")).unwrap()),
            header
        );
        let snapshot = scratch.0.join("snapshot");
        let reference = Ref {
            repo_id: tree.id(),
            revision: None,
            variant: Some("main".into()),
            endpoint: None,
        };
        client
            .materialize(&reference, &Download::default(), Some(&snapshot), false)
            .await
            .unwrap();
        assert_eq!(fs::read(snapshot.join("README.md")).unwrap(), README);
        assert_eq!(
            ObjectRef::of(&fs::read(snapshot.join("model.main.cozytensors")).unwrap()),
            header
        );
        for object in checkpoint.header().objects() {
            assert_eq!(
                ObjectRef::of(&fs::read(snapshot.join("chunks").join(&object.sha256)).unwrap()),
                *object
            );
        }
    }
}

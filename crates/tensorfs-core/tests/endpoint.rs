//! The distribution client against real endpoints: a directory (`file://`), and the same
//! directory served read-only by `python -m http.server` (no Range, no ETag, no writes).

use std::collections::BTreeMap;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;

use tensorfs_core::err::Code;
use tensorfs_core::fetch::{
    upload_tree, Client, Config, Endpoint, Fetcher, Fresh, ObjectStore, Origin, Payload, SourcePolicy, Want,
};
use tensorfs_core::format::{ConfigValue, Dtype, ObjectRef, Tree};
use tensorfs_core::store2::Store;
use tensorfs_core::writer::{Data, Source};

const MIB: usize = 1 << 20;

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("tfs-endpoint-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn noise(seed: u64, n: usize) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

fn client() -> Client {
    Client::new(SourcePolicy { any_host: true, allow_local: true, max_redirects: 5, ..Default::default() })
}

/// A two-component checkpoint (a grid-sized tensor, small ones, an asset) as variant `bf16` of
/// a tree with a README: (tree, header, the tensors by component/key).
fn checkpoint(store: &Store, seed: u64) -> (ObjectRef, ObjectRef, BTreeMap<(String, String), Vec<u8>>) {
    let mut tensors = BTreeMap::new();
    let mut w = store.writer(None, None).unwrap();
    let config = BTreeMap::from([("layers".to_owned(), ConfigValue::Int(2))]);
    for (component, key, n) in [("unet", "big.weight", 40 * MIB), ("unet", "norm.weight", 3000), ("vae", "conv.weight", 2 * MIB)] {
        let bytes = noise(seed + n as u64, n);
        let mut c = w.component(component, "diffusers.Thing", config.clone(), None).unwrap();
        c.put(key, &Dtype::U8, &[n as u64], Data::Plain(Source::Read(&mut &bytes[..]))).unwrap();
        tensors.insert((component.to_owned(), key.to_owned()), bytes);
    }
    w.get("tokenizer").asset("vocab.json", &noise(seed, 5000)).unwrap();
    let header = w.commit().unwrap();
    let mut tree = Tree::default();
    tree.set_variant("bf16", header.clone());
    tree.files.insert("README.md".into(), store.put_bytes(format!("# model {seed}\n").as_bytes()).unwrap());
    (store.put_tree(&tree).unwrap(), header, tensors)
}

/// `python -m http.server` over `dir` on a free loopback port, killed when dropped.
struct HttpServer(Child, String);

impl HttpServer {
    fn start(dir: &Path) -> HttpServer {
        let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let child = Command::new("python3")
            .args(["-m", "http.server", "--bind", "127.0.0.1", &port.to_string(), "--directory"])
            .arg(dir)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        while std::net::TcpStream::connect(("127.0.0.1", port)).is_err() {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        HttpServer(child, format!("http://127.0.0.1:{port}"))
    }
}

impl Drop for HttpServer {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Pulls `name@1.0` variant `bf16` from `origin` into `store` and checks every tensor.
async fn pull_and_check(origin: Arc<dyn Origin>, store: &Store, header: &ObjectRef, tensors: &BTreeMap<(String, String), Vec<u8>>) {
    let fetcher = Fetcher::new(Arc::new(store.clone()) as Arc<dyn ObjectStore>, client(), None, Config::default());
    let pulled = fetcher.pull(&origin, "Acme/Tiny", Some("1.0"), Some("bf16"), &Want::default()).await.unwrap();
    fetcher.settled().await;
    assert_eq!(pulled.header, *header);
    let ck = store.checkpoint(header).unwrap();
    for ((component, key), bytes) in tensors {
        assert!(ck.read(component, key).unwrap() == *bytes, "{component}/{key}");
    }
    for object in ck.header().objects() {
        assert!(store.has(object), "{object:?}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_tree_goes_up_to_a_directory_and_down_over_http() {
    let scratch = Scratch::new("roundtrip");
    let source = Store::open(scratch.0.join("source")).unwrap();
    let (tree, header, tensors) = checkpoint(&source, 1);
    let dir = scratch.0.join("endpoint");
    let endpoint = Endpoint::new(&client(), &format!("file://{}", dir.display()), None).unwrap();

    let up = upload_tree(&endpoint, "Acme/Tiny", &tree, &source, Some("1.0")).await.unwrap();
    let closure = source.closure(&tree).unwrap().len();
    assert_eq!((up.sent, up.objects), (closure, closure));
    assert_eq!(up.repo.release(Some("1.0")).unwrap().tree, tree);
    assert!(dir.join("refs/acme/tiny.json").is_file(), "keys are lowercased");
    let again = upload_tree(&endpoint, "acme/tiny", &tree, &source, Some("1.0")).await.unwrap();
    assert_eq!((again.sent, again.repo), (0, up.repo.clone()), "a second upload sends nothing and changes nothing");
    let notes = Tree { files: BTreeMap::from([("README.md".to_owned(), source.put_bytes(b"notes").unwrap())]) };
    let notes = source.put_tree(&notes).unwrap();
    let unreleased = upload_tree(&endpoint, "acme/tiny", &notes, &source, None).await.unwrap().repo;
    assert_eq!((unreleased.checkpoints, unreleased.releases), (vec![notes.clone()], up.repo.releases.clone()));
    let moved = upload_tree(&endpoint, "acme/tiny", &notes, &source, Some("1.0")).await.unwrap().repo;
    assert_eq!((moved.release(Some("1.0")).unwrap().revision, moved.checkpoints.len()), (1, 0), "a changed release counts");
    endpoint.update_repo("acme/tiny", |_| Ok(Some(up.repo.clone()))).await.unwrap();

    let (_, etag) = endpoint.repo("acme/tiny").await.unwrap().unwrap();
    assert_eq!(endpoint.repo_since("acme/tiny", Some(&etag)).await.unwrap(), Fresh::Unchanged);
    assert_eq!(endpoint.repo_since("acme/none", None).await.unwrap(), Fresh::Absent);

    pull_and_check(Arc::new(endpoint), &Store::open(scratch.0.join("by-dir")).unwrap(), &header, &tensors).await;

    let server = HttpServer::start(&dir);
    let http = Arc::new(Endpoint::new(&client(), &server.1, None).unwrap());
    let (repo, etag) = http.repo("acme/tiny").await.unwrap().unwrap();
    assert_eq!(repo, up.repo);
    assert_eq!(http.repo_since("acme/tiny", Some(&etag)).await.unwrap(), Fresh::Unchanged);
    assert!(http.has(&tree).await.unwrap());
    pull_and_check(http.clone(), &Store::open(scratch.0.join("by-http")).unwrap(), &header, &tensors).await;

    // A static server takes no writes.
    let readme = ObjectRef::of(b"never sent");
    let payload = || Ok(Payload::Bytes(b"never sent"[..].into()));
    let refused = http.upload(&readme, &payload).await.unwrap_err();
    assert_eq!(refused.code, Code::HUB_REFUSED, "{refused}");
    let refused = http.update_repo("acme/tiny", |_| Ok(None)).await.unwrap_err();
    assert_eq!(refused.code, Code::HUB_REFUSED, "{refused}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_writers_all_land_and_wrong_bytes_never_do() {
    let scratch = Scratch::new("writers");
    let source = Store::open(scratch.0.join("source")).unwrap();
    let dir = scratch.0.join("endpoint");
    let endpoint = Arc::new(Endpoint::new(&client(), dir.to_str().unwrap(), None).unwrap());

    let mut trees = Vec::new();
    for i in 0..8u8 {
        let readme = source.put_bytes(&[b'r', i]).unwrap();
        let tree = Tree { files: BTreeMap::from([("README.md".to_owned(), readme.clone())]) };
        let tree = source.put_tree(&tree).unwrap();
        for object in [&readme, &tree] {
            let (from, at) = (source.clone(), object.clone());
            endpoint.upload(object, &move || Ok(Payload::File(from.file(&at).unwrap()))).await.unwrap();
        }
        trees.push(tree);
    }
    let writers = trees.iter().cloned().map(|tree| {
        let endpoint = Arc::clone(&endpoint);
        tokio::spawn(async move {
            endpoint.update_repo("acme/many", |old| Ok(Some(tensorfs_core::fetch::with_tree(old, "acme/many", &tree, None)))).await
        })
    });
    for writer in futures::future::join_all(writers).await {
        writer.unwrap().unwrap();
    }
    let (repo, _) = endpoint.repo("acme/many").await.unwrap().unwrap();
    let mut landed = repo.checkpoints.clone();
    landed.sort();
    trees.sort();
    assert_eq!(landed, trees);

    let claimed = ObjectRef::of(b"what was promised");
    let wrong = endpoint.upload(&claimed, &|| Ok(Payload::Bytes(b"something else"[..].into()))).await.unwrap_err();
    assert_eq!(wrong.code, Code::OBJECT_ID_MISMATCH, "{wrong}");
    assert!(!endpoint.has(&claimed).await.unwrap());

    let deleted = endpoint.update_repo("acme/many", |_| Ok(None)).await.unwrap();
    assert!(deleted.checkpoints.is_empty() && endpoint.repo("acme/many").await.unwrap().is_none());
}

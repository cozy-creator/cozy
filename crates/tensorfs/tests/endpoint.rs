//! The distribution client against real endpoints: a directory (`file://`), and the same
//! directory served read-only by `python -m http.server` (no Range, no ETag, no writes).

use std::collections::BTreeMap;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;

use tensorfs::format::ClassRef;
use tensorfs::err::Code;
use tensorfs::fetch::{
    upload_tree, Client, Config, Endpoint, Fetcher, Fresh, ObjectStore, Origin, Payload, SourcePolicy, Stream, Want,
};
use tensorfs::format::{Config as HeaderConfig, Dtype, ObjectRef, Tree};
use tensorfs::store::Store;
use tensorfs::writer::{Data, Source};

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
    let config: HeaderConfig = serde_json::from_str(r#"{"layers":2}"#).unwrap();
    for (component, key, n) in [("unet", "big.weight", 40 * MIB), ("unet", "norm.weight", 3000), ("vae", "conv.weight", 2 * MIB)] {
        let bytes = noise(seed + n as u64, n);
        let mut c = w.component(component, ClassRef::new("diffusers", "0.35.1", "Thing"), config.clone(), None).unwrap();
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

    let up = upload_tree(&endpoint, "Acme/Tiny", &tree, &source, "1.0").await.unwrap();
    let closure = source.closure(&tree).unwrap().len();
    assert_eq!((up.sent, up.objects), (closure, closure));
    let tag = up.repo.tag("1.0").unwrap();
    assert_eq!((&tag.commit, tag.public), (&tree, false), "a new tag is private");
    assert!(dir.join("refs/acme/tiny.json").is_file(), "keys are lowercased");
    let again = upload_tree(&endpoint, "acme/tiny", &tree, &source, "1.0").await.unwrap();
    assert_eq!((again.sent, again.repo), (0, up.repo.clone()), "a second upload sends nothing and changes nothing");
    let published = endpoint.update_repo("acme/tiny", |old| Ok(old.map(|mut r| (r.tag_mut("1.0").unwrap().public = true, r).1))).await.unwrap();
    let notes = Tree { files: BTreeMap::from([("README.md".to_owned(), source.put_bytes(b"notes").unwrap())]) };
    let notes = source.put_tree(&notes).unwrap();
    let drafted = upload_tree(&endpoint, "acme/tiny", &notes, &source, "main").await.unwrap().repo;
    let tags: Vec<(&str, &ObjectRef, bool)> = drafted.tags.iter().map(|t| (t.name.as_str(), &t.commit, t.public)).collect();
    assert_eq!(tags, [("1.0", &tree, true), ("main", &notes, false)], "a new tag goes last");
    let moved = upload_tree(&endpoint, "acme/tiny", &notes, &source, "1.0").await.unwrap().repo;
    let tag = moved.tag("1.0").unwrap();
    assert_eq!((&tag.commit, tag.public, moved.tags.len()), (&notes, true, 2), "the tag moves and keeps its flags");
    endpoint.update_repo("acme/tiny", |_| Ok(Some(published.clone()))).await.unwrap();

    let (_, etag) = endpoint.repo("acme/tiny").await.unwrap().unwrap();
    assert_eq!(endpoint.repo_since("acme/tiny", Some(&etag)).await.unwrap(), Fresh::Unchanged);
    assert_eq!(endpoint.repo_since("acme/none", None).await.unwrap(), Fresh::Absent);

    pull_and_check(Arc::new(endpoint), &Store::open(scratch.0.join("by-dir")).unwrap(), &header, &tensors).await;

    let server = HttpServer::start(&dir);
    let http = Arc::new(Endpoint::new(&client(), &server.1, None).unwrap());
    let (repo, etag) = http.repo("acme/tiny").await.unwrap().unwrap();
    assert_eq!(repo, published);
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
    let writers = trees.iter().cloned().enumerate().map(|(i, tree)| {
        let endpoint = Arc::clone(&endpoint);
        tokio::spawn(async move {
            endpoint.update_repo("acme/many", |old| Ok(Some(tensorfs::fetch::with_tag(old, "acme/many", &format!("w{i}"), &tree)))).await
        })
    });
    for writer in futures::future::join_all(writers).await {
        writer.unwrap().unwrap();
    }
    let (repo, _) = endpoint.repo("acme/many").await.unwrap().unwrap();
    let mut landed: Vec<ObjectRef> = repo.tags.iter().map(|t| t.commit.clone()).collect();
    landed.sort();
    trees.sort();
    assert_eq!(landed, trees);

    let claimed = ObjectRef::of(b"what was promised");
    let wrong = endpoint.upload(&claimed, &|| Ok(Payload::Bytes(b"something else"[..].into()))).await.unwrap_err();
    assert_eq!(wrong.code, Code::OBJECT_ID_MISMATCH, "{wrong}");
    assert!(!endpoint.has(&claimed).await.unwrap());

    // Raw bytes, for documents of another layout: a stale ETag is refused, the current one lands.
    let Fresh::Doc(bytes, etag) = endpoint.repo_bytes("acme/many", None).await.unwrap() else { panic!("present") };
    assert!(!endpoint.put_repo_bytes("acme/many", b"{}".to_vec(), Some("\"sha256:stale\"")).await.unwrap());
    assert!(!endpoint.put_repo_bytes("acme/many", b"{}".to_vec(), None).await.unwrap(), "a create finds it there");
    assert!(endpoint.put_repo_bytes("acme/many", bytes.clone(), Some(&etag)).await.unwrap());

    let deleted = endpoint.update_repo("acme/many", |_| Ok(None)).await.unwrap();
    assert_eq!(endpoint.repo("acme/many").await.unwrap().unwrap().0, deleted);
    assert!(deleted.tags.is_empty(), "deleting is a document of no tags");
}

#[test]
fn a_streaming_writer_tees_to_the_store_before_publishing() {
    let scratch = Scratch::new("stream");
    let runtime = tensorfs::fetch::runtime().unwrap();
    let dir = scratch.0.join("endpoint");
    let endpoint = Arc::new(Endpoint::new(&client(), dir.to_str().unwrap(), None).unwrap());
    let local = Store::open(scratch.0.join("local")).unwrap();
    let stream = Arc::new(Stream::new(Arc::clone(&endpoint), runtime.handle().clone()));
    let mut w = local.streaming_writer(Arc::clone(&stream), "acme/streamed", "run-7", "int8", None).unwrap();
    let big = noise(9, 70 * MIB);
    let mut c = w.component("unet", ClassRef::new("diffusers", "0.35.1", "Thing"), Default::default(), None).unwrap();
    c.put("big", &Dtype::U8, &[big.len() as u64], Data::Plain(Source::Read(&mut &big[..]))).unwrap();
    c.put("small", &Dtype::U8, &[5], Data::Plain(Source::Read(&mut &b"small"[..]))).unwrap();
    stream.flush().unwrap();
    assert!(runtime.block_on(endpoint.has(&ObjectRef::of(&big[..64 * MIB]))).unwrap());
    assert!(runtime.block_on(endpoint.repo("acme/streamed")).unwrap().is_none(), "no checkpoint before commit");
    w.repo_file("README.md", b"# streamed").unwrap();
    let header = w.commit().unwrap();

    assert!(local.checkpoint(&header).unwrap().read("unet", "big").unwrap() == big);
    assert_eq!(runtime.block_on(endpoint.object(&header.id())).unwrap(), Some(header.clone()));
    assert!(runtime.block_on(endpoint.object("not-a-digest")).is_err());
    let (repo, _) = runtime.block_on(endpoint.repo("acme/streamed")).unwrap().unwrap();
    let tag = &repo.tags[..];
    assert_eq!((tag.len(), tag[0].name.as_str(), tag[0].public), (1, "run-7", false), "a private tag");
    assert_eq!(w.published(), Some(&tag[0].commit), "this writer's exact committed tree");
    let into = Store::open(scratch.0.join("into")).unwrap();
    runtime.block_on(async {
        let fetcher = Fetcher::new(Arc::new(into.clone()) as Arc<dyn ObjectStore>, client(), None, Config::default());
        let origin: Arc<dyn Origin> = endpoint.clone();
        let pulled = fetcher.pull(&origin, "acme/streamed", Some(&header.id()), None, &Want::default()).await.unwrap();
        fetcher.settled().await;
        assert_eq!((pulled.header, pulled.variant.as_str()), (header.clone(), "int8"));
    });
    assert!(into.checkpoint(&header).unwrap().read("unet", "big").unwrap() == big);
}

/// Synchronous puts must also work from the caller's Tokio runtime. The finalizer may
/// block off-thread, but sealed files should already be uploading before it is called.
#[tokio::test(flavor = "current_thread")]
async fn stored_stream_puts_work_inside_the_callers_runtime() {
    let scratch = Scratch::new("async-stored");
    let dir = scratch.0.join("endpoint");
    let endpoint = Arc::new(Endpoint::new(&client(), dir.to_str().unwrap(), None).unwrap());
    let local = Store::open(scratch.0.join("local")).unwrap();
    let stream = Arc::new(Stream::new(Arc::clone(&endpoint), tokio::runtime::Handle::current()));
    let mut writer = local.streaming_writer(Arc::clone(&stream), "acme/async-stored", "main", "bf16", None).unwrap();
    let mut files = Vec::new();
    for index in 0..9 {
        let bytes = format!("stored file {index}").into_bytes();
        writer.repo_file(&format!("file-{index}.txt"), &bytes).unwrap();
        files.push((ObjectRef::of(&bytes), bytes));
    }
    writer.component("model", Default::default(), Default::default(), None).unwrap()
        .put("weight", &Dtype::U8, &[7], Data::Plain(Source::Read(&mut &b"weights"[..]))).unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            let mut complete = true;
            for (object, _) in &files {
                complete &= endpoint.has(object).await.unwrap();
            }
            if complete { break; }
            tokio::task::yield_now().await;
        }
    }).await.expect("sealed files upload before commit");
    assert!(endpoint.repo("acme/async-stored").await.unwrap().is_none());
    for (object, bytes) in &files {
        assert_eq!(&std::fs::read(dir.join(object.key())).unwrap(), bytes);
    }
    let (header, tree) = tokio::task::spawn_blocking(move || {
        let header = writer.commit().unwrap();
        (header, writer.published().unwrap().clone())
    }).await.unwrap();
    assert_eq!(local.checkpoint(&header).unwrap().read("model", "weight").unwrap(), b"weights");
    assert_eq!(endpoint.repo("acme/async-stored").await.unwrap().unwrap().0.tag("main").unwrap().commit, tree);
}

#[tokio::test(flavor = "current_thread")]
async fn dropping_a_stored_stream_cancels_unstarted_uploads() {
    let scratch = Scratch::new("async-abandon");
    let endpoint = Arc::new(Endpoint::new(&client(), scratch.0.join("endpoint").to_str().unwrap(), None).unwrap());
    let local = Store::open(scratch.0.join("local")).unwrap();
    let stream = Arc::new(Stream::new(Arc::clone(&endpoint), tokio::runtime::Handle::current()));
    let mut writer = local.streaming_writer(Arc::clone(&stream), "acme/abandoned", "main", "bf16", None).unwrap();
    let bytes = b"never uploaded";
    writer.repo_file("README.md", bytes).unwrap();
    writer.abandon();
    drop(stream);
    tokio::task::yield_now().await;
    assert!(!endpoint.has(&ObjectRef::of(bytes)).await.unwrap());
    assert!(endpoint.repo("acme/abandoned").await.unwrap().is_none());
}

#[test]
fn an_upload_failure_keeps_the_local_checkpoint_and_stays_failed() {
    let scratch = Scratch::new("stream-failure");
    let served = scratch.0.join("static");
    Store::open(&served).unwrap();
    let server = HttpServer::start(&served);
    let runtime = tensorfs::fetch::runtime().unwrap();
    let endpoint = Arc::new(Endpoint::new(&client(), &server.1, None).unwrap());
    let local = Store::open(scratch.0.join("local")).unwrap();
    let stream = Arc::new(Stream::new(Arc::clone(&endpoint), runtime.handle().clone()));
    let mut w = local.streaming_writer(Arc::clone(&stream), "acme/failed", "main", "int8", None).unwrap();
    w.component("unet", ClassRef::new("diffusers", "0.35.1", "Thing"), Default::default(), None).unwrap()
        .put("w", &Dtype::U8, &[5], Data::Plain(Source::Read(&mut &b"saved"[..]))).unwrap();
    assert!(w.commit().is_err(), "the real read-only server refuses PUT");
    assert!(w.published().is_none(), "no remote commit was made");
    let header = w.committed().expect("the local commit precedes upload").clone();
    assert_eq!(local.checkpoint(&header).unwrap().read("unet", "w").unwrap(), b"saved");
    assert!(stream.flush().is_err(), "the first upload failure is sticky");
    assert!(runtime.block_on(endpoint.repo("acme/failed")).unwrap().is_none());
}

/// `tfs conform endpoint <url> [--write] --json`: (its exit success, its failed MUST count, its cases).
fn conform(url: &str, write: bool) -> (bool, u64, Vec<serde_json::Value>) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_tfs"));
    cmd.args(["conform", "endpoint", url, "--json"]);
    if write {
        cmd.arg("--write");
    }
    let out = cmd.output().unwrap();
    let doc: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap_or_else(|_| panic!("{}", String::from_utf8_lossy(&out.stderr)));
    (out.status.success(), doc["failed"].as_u64().unwrap(), doc["cases"].as_array().unwrap().clone())
}

#[test]
fn a_directory_and_a_static_server_conform() {
    let scratch = Scratch::new("conform");
    let dir = scratch.0.join("endpoint");
    let (ok, failed, cases) = conform(&format!("file://{}", dir.display()), true);
    assert!(ok && failed == 0, "{cases:#?}");
    assert!(cases.len() >= 12);
    let server = HttpServer::start(&dir);
    let runtime = tensorfs::fetch::runtime().unwrap();
    let endpoint = Endpoint::new(&client(), &server.1, None).unwrap();
    let fixture = runtime.block_on(endpoint.repo(tensorfs::fetch::conform::FIXTURE)).unwrap().unwrap().0;
    let tree = &fixture.tag("1").unwrap().commit;
    assert_eq!(runtime.block_on(endpoint.object(&tree.sha256)).unwrap(), Some(tree.clone()));
    let (ok, failed, cases) = conform(&server.1, false);
    assert!(ok && failed == 0, "{cases:#?}");
    // python -m http.server sends no ETag and ignores Range: SHOULDs it fails.
    let warned: Vec<&str> = (cases.iter()).filter(|c| c["ok"] == false).filter_map(|c| c["case"].as_str()).collect();
    assert_eq!(warned, ["refs: an ETag", "refs: If-None-Match is 304", "blobs: Range gives 206 with that byte (or redirects)"]);
}

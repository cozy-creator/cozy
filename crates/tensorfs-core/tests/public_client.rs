//! An external consumer of the public session on real files and a loopback HTTP
//! origin. No raw Store, private downloader, pretrained data or GPU is used.
#![cfg(feature = "store")]

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use async_trait::async_trait;
use tensorfs_core::format::{self, Body, Dtype, Header, Location, ObjectRef, Tensor};
use tensorfs_core::{
    bearer, Client, Config, Endpoint, Failure, Fetch, Kind, Open, Ref, Request, Selection,
    TokenProvider,
};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
                "public-client-{}-{}",
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

fn config(root: &Path, endpoint: &str) -> Config {
    let mut config = Config::from_env();
    config.cache_dir = root.to_path_buf();
    config.endpoint = endpoint.to_owned();
    config.mirror = None;
    config.token = None;
    config.ca_der = None;
    config
}

struct Fixture {
    folder: PathBuf,
    chunks: Vec<ObjectRef>,
    asset: ObjectRef,
    header: ObjectRef,
}

fn fixture(root: &Path) -> Fixture {
    let folder = root.join("fixture");
    fs::create_dir_all(folder.join("chunks")).unwrap();
    let mut chunks = Vec::new();
    for bytes in [b"abcdefgh", b"ijklmnop", b"QRSTUVWX"] {
        let object = ObjectRef::of(bytes);
        fs::write(folder.join("chunks").join(&object.sha256), bytes).unwrap();
        chunks.push(object);
    }
    let asset = ObjectRef::of(b"{\"vocab\": [\"one\", \"two\"]}");
    fs::write(
        folder.join("chunks").join(&asset.sha256),
        b"{\"vocab\": [\"one\", \"two\"]}",
    )
    .unwrap();
    let header = Header::new(
        chunks.clone(),
        None,
        vec![
            format::Component {
                name: "model".into(),
                class: "test.Tiny".into(),
                config: serde_json::from_str("{\"width\":16,\"eps\":0.001,\"activation\":null}")
                    .unwrap(),
                tensors: vec![Tensor {
                    key: "w".into(),
                    dtype: Dtype::U8,
                    shape: vec![16],
                    body: Body::Data(Location {
                        chunk: 0,
                        offset: 0,
                        length: 16,
                    }),
                }],
                ..format::Component::default()
            },
            format::Component {
                name: "other".into(),
                tensors: vec![Tensor {
                    key: "v".into(),
                    dtype: Dtype::U8,
                    shape: vec![8],
                    body: Body::Data(Location {
                        chunk: 2,
                        offset: 0,
                        length: 8,
                    }),
                }],
                ..format::Component::default()
            },
            format::Component {
                name: "tokenizer".into(),
                assets: BTreeMap::from([("tokenizer.json".into(), asset.clone())]),
                ..format::Component::default()
            },
        ],
        BTreeMap::new(),
    )
    .unwrap()
    .to_cbor()
    .unwrap();
    let id = ObjectRef::of(&header);
    fs::write(folder.join("model.cozytensors"), header).unwrap();
    fs::write(
        folder.join("README.md"),
        b"local fixture, no downloaded model\n",
    )
    .unwrap();
    Fixture {
        folder,
        chunks,
        asset,
        header: id,
    }
}

struct Http {
    url: String,
    requests: Arc<Mutex<Vec<(String, String, bool)>>>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Http {
    fn new(root: PathBuf) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let (thread_stop, record) = (Arc::clone(&stop), Arc::clone(&requests));
        let worker = thread::spawn(move || {
            while !thread_stop.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => Self::serve(stream, &root, &record),
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2))
                    }
                    Err(error) => panic!("HTTP accept: {error}"),
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

    fn serve(mut stream: TcpStream, root: &Path, record: &Mutex<Vec<(String, String, bool)>>) {
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
        record.lock().unwrap().push((
            method.to_owned(),
            path.to_owned(),
            request
                .to_ascii_lowercase()
                .contains("authorization: bearer public-client-proof"),
        ));
        assert!(!path.contains(".."));
        let Ok(bytes) = fs::read(root.join(path.trim_start_matches('/'))) else {
            let _ = stream.write_all(
                b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            );
            return;
        };
        let range = request.lines().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            if !name.eq_ignore_ascii_case("range") {
                return None;
            }
            let (lo, hi) = value.trim().strip_prefix("bytes=")?.split_once('-')?;
            Some((
                lo.parse::<usize>().unwrap(),
                if hi.is_empty() {
                    bytes.len()
                } else {
                    hi.parse::<usize>().unwrap() + 1
                },
            ))
        });
        let (lo, hi, status) = range.map_or((0, bytes.len(), "200 OK"), |(lo, hi)| {
            (lo, hi.min(bytes.len()), "206 Partial Content")
        });
        assert!(lo <= hi);
        let range_header = if range.is_some() {
            format!("Content-Range: bytes {}-{}/{}\r\n", lo, hi - 1, bytes.len())
        } else {
            String::new()
        };
        let header = format!("HTTP/1.1 {status}\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nETag: \"{}\"\r\n{range_header}Connection: close\r\n\r\n", hi - lo, ObjectRef::of(&bytes).id());
        stream.write_all(header.as_bytes()).unwrap();
        if method != "HEAD" {
            let _ = stream.write_all(&bytes[lo..hi]);
        }
    }

    fn count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
    fn fetched(&self, object: &ObjectRef) -> bool {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .any(|(_, path, _)| path.ends_with(&object.sha256))
    }
}
impl Drop for Http {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            worker.join().unwrap();
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn actual_public_client_selects_fetches_and_retains_children() {
    let scratch = Scratch::new();
    let data = fixture(&scratch.0);
    let origin = scratch.0.join("origin");
    let file_origin = format!("file://{}", origin.display());
    let producer = Client::new(config(&scratch.0.join("producer"), &file_origin)).unwrap();
    let reference: Ref = "account/model@1".parse().unwrap();
    let tree = producer
        .upload_folder(&reference, &data.folder, None)
        .await
        .unwrap();
    drop(producer);
    let http = Http::new(origin);
    let cache = scratch.0.join("consumer");
    let mut options = config(&cache, &http.url);
    options.token = Some(bearer("public-client-proof"));
    let client = Client::new(options).unwrap();
    let checkpoint = client
        .open(
            "account/model",
            &Open {
                revision: Some("1".into()),
                ..Open::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(checkpoint.id(), &data.header);
    assert_eq!(checkpoint.tree(), Some(&tree));
    assert!(!http.fetched(&data.chunks[0]) && !http.fetched(&data.chunks[2]));
    assert_eq!(checkpoint.components().len(), 3);

    checkpoint
        .fetch(&Fetch {
            selection: Selection::Assets {
                component: "tokenizer".into(),
                names: None,
            },
            ..Fetch::default()
        })
        .await
        .unwrap();
    let tokenizer = checkpoint.component("tokenizer").unwrap().clone();
    assert_eq!(
        fs::read(tokenizer.folder().unwrap().join("tokenizer.json")).unwrap(),
        tokenizer.asset("tokenizer.json").unwrap()
    );
    assert!(http.fetched(&data.asset));
    assert!(!http.fetched(&data.chunks[0]) && !http.fetched(&data.chunks[2]));

    checkpoint
        .fetch(&Fetch {
            selection: Selection::Part {
                component: "model".into(),
                key: "w".into(),
                role: Some(String::new()),
                range: Some(0..4),
            },
            ..Fetch::default()
        })
        .await
        .unwrap();
    assert!(http.fetched(&data.chunks[0]) && !http.fetched(&data.chunks[1]));
    let component = checkpoint.component("model").unwrap().clone();
    let mut prefix = [0; 4];
    component.read_part_into("w", "", 0, &mut prefix).unwrap();
    assert_eq!(&prefix, b"abcd");
    let invalid = checkpoint
        .fetch(&Fetch {
            selection: Selection::Part {
                component: "model".into(),
                key: "w".into(),
                role: Some(String::new()),
                range: Some(15..17),
            },
            ..Fetch::default()
        })
        .await
        .unwrap_err();
    assert_eq!(invalid.kind, Kind::Invalid);

    checkpoint.fetch(&Fetch::default()).await.unwrap();
    assert!(checkpoint.missing(&Selection::All).unwrap().is_empty());
    let selected = Fetch {
        selection: Selection::Components(vec!["model".into()]),
        ..Fetch::default()
    };
    let snapshot = client
        .snapshot_download(&reference, &selected, None, false)
        .await
        .unwrap();
    assert!(snapshot
        .join("chunks")
        .join(&data.chunks[0].sha256)
        .is_file());
    assert!(
        !snapshot
            .join("chunks")
            .join(&data.chunks[2].sha256)
            .exists(),
        "warm unselected chunk was materialized"
    );
    assert!(!snapshot.join("chunks").join(&data.asset.sha256).exists());
    assert_eq!(
        fs::read(snapshot.join("README.md")).unwrap(),
        b"local fixture, no downloaded model\n"
    );
    let user_dir = cache.join("snapshots").join("user-owned");
    fs::create_dir_all(&user_dir).unwrap();
    fs::write(user_dir.join("mine"), b"owned by caller").unwrap();
    assert_eq!(
        client
            .snapshot_download(&reference, &selected, Some(&user_dir), false)
            .await
            .unwrap(),
        user_dir
    );
    assert_eq!(fs::read(user_dir.join("mine")).unwrap(), b"owned by caller");
    let readme = client
        .hub_download(
            &reference,
            "README.md",
            Some(&scratch.0.join("readme")),
            false,
        )
        .await
        .unwrap();
    assert_eq!(
        fs::read(readme).unwrap(),
        b"local fixture, no downloaded model\n"
    );

    let before = http.count();
    let warm = client
        .open("account/model@1", &Open::default())
        .await
        .unwrap();
    warm.fetch(&Fetch::default()).await.unwrap();
    assert_eq!(
        http.count(),
        before,
        "warm explicit revision contacted the origin"
    );
    assert!(http
        .requests
        .lock()
        .unwrap()
        .iter()
        .all(|(_, _, authorized)| *authorized));
    let view = checkpoint.view();
    let mut file = checkpoint.chunk(0).unwrap();
    drop(warm);
    drop(tokenizer);
    drop(checkpoint);
    drop(client);
    assert!(view.has_source());
    assert_eq!(component.read("w").unwrap(), b"abcdefghijklmnop");
    let mut bytes = [0; 8];
    file.read_exact(&mut bytes).unwrap();
    assert_eq!(&bytes, b"abcdefgh");
    drop(component);
    drop(view);
    drop(file);
    #[cfg(target_os = "linux")]
    {
        let owned_fds = || {
            fs::read_dir("/proc/self/fd")
                .unwrap()
                .flatten()
                .filter(|entry| {
                    fs::read_link(entry.path()).is_ok_and(|target| target.starts_with(&cache))
                })
                .count()
        };
        for _ in 0..100 {
            if owned_fds() == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(
            owned_fds(),
            0,
            "final child/view drop retained cache ownership descriptors"
        );
    }
}

struct Denied;
#[async_trait]
impl TokenProvider for Denied {
    async fn headers(
        &self,
        _request: &Request<'_>,
    ) -> tensorfs_core::Result<Vec<(String, String)>> {
        Err(Failure {
            kind: Kind::Denied,
            detail: "exact consumer refusal".into(),
        })
    }
}

#[tokio::test(flavor = "current_thread")]
async fn supplied_provider_error_and_local_unreleased_semantics_are_public() {
    let scratch = Scratch::new();
    let data = fixture(&scratch.0);
    let mut options = config(&scratch.0.join("cache"), "http://127.0.0.1:1");
    options.token = Some(Arc::new(Denied));
    let endpoint = Endpoint::with_config(&options.endpoint, &options).unwrap();
    let error = endpoint.repo("account/model").await.unwrap_err();
    assert_eq!(
        (error.kind, error.detail.as_str()),
        (Kind::Denied, "exact consumer refusal")
    );
    let client = Client::new(options).unwrap();
    let reference: Ref = "local/unreleased".parse().unwrap();
    let tree = client
        .upload_folder(&reference, &data.folder, None)
        .await
        .unwrap();
    let local = Endpoint::with_config(
        "local",
        &config(&scratch.0.join("cache"), "https://tensorhub.com"),
    )
    .unwrap();
    let repo = local.repo("local/unreleased").await.unwrap().unwrap().0;
    assert!(repo.releases.is_empty() && repo.checkpoints == [tree.clone()]);
    assert_eq!(
        client
            .open("local/unreleased", &Open::default())
            .await
            .unwrap_err()
            .kind,
        Kind::NotFound
    );
    let checkpoint = client
        .open(
            &tree.id(),
            &Open {
                local_files_only: true,
                ..Open::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(checkpoint.id(), &data.header);
    assert_eq!(
        checkpoint.component("other").unwrap().read("v").unwrap(),
        b"QRSTUVWX"
    );
}

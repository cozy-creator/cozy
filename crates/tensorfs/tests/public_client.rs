//! An external consumer of the public session on real files and a loopback HTTP
//! origin. No raw Store, private downloader, pretrained data or GPU is used.
#![cfg(feature = "store")]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use async_trait::async_trait;
use tensorfs::format::{self, Body, Dtype, Header, Location, ObjectRef, Tensor};
use tensorfs::{
    bearer, Client, Config, Endpoint, Failure, Download, Kind, Open, Ref, Request, Selection,
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

#[cfg(target_os = "linux")]
fn owned_fds(root: &Path) -> usize {
    fs::read_dir("/proc/self/fd")
        .unwrap()
        .flatten()
        .filter(|entry| fs::read_link(entry.path()).is_ok_and(|target| target.starts_with(root)))
        .count()
}

#[tokio::test(flavor = "current_thread")]
async fn eager_open_downloads_checkpoint_bytes_without_materializing_a_repository() {
    assert!(Open::default().lazy);
    let scratch = Scratch::new();
    let data = fixture(&scratch.0);
    let origin = scratch.0.join("origin");
    let producer = Client::new(config(
        &scratch.0.join("producer"),
        &format!("file://{}", origin.display()),
    ))
    .unwrap();
    let tree = producer
        .upload_folder(&"account/eager@1".parse().unwrap(), &data.folder, None)
        .await
        .unwrap();
    let http = Http::new(origin);
    let cache = scratch.0.join("consumer");
    let client = Client::new(config(&cache, &http.url)).unwrap();
    let options = Open {
        revision: Some("1".into()),
        ..Open::default()
    };
    let lazy = client.open("account/eager", &options).await.unwrap();
    assert!(!lazy.missing(&Selection::All).unwrap().is_empty());
    assert!(!http.fetched(&data.chunks[0]) && !http.fetched(&data.asset));
    let eager = client
        .open("account/eager", &Open { lazy: false, ..options.clone() })
        .await
        .unwrap();
    assert!(eager.missing(&Selection::All).unwrap().is_empty());
    for object in data.chunks.iter().chain([&data.asset]) {
        assert!(http.fetched(object));
    }
    assert!(!cache.join("snapshots").join(&tree.sha256).exists());
    let readme = ObjectRef::of(b"local fixture, no downloaded model\n");
    assert!(!http.fetched(&readme), "eager open downloaded a repository file");
    let requests = http.requests.lock().unwrap().len();
    assert!(client
        .open("account/eager", &Open { lazy: false, local_files_only: true, ..options.clone() })
        .await
        .is_ok());
    fs::remove_file(cache.join(data.chunks[0].key())).unwrap();
    assert_eq!(client
        .open("account/eager", &Open { lazy: false, local_files_only: true, ..options })
        .await
        .unwrap_err()
        .kind, Kind::NotFound);
    assert_eq!(http.requests.lock().unwrap().len(), requests);
}

#[tokio::test(flavor = "current_thread")]
async fn eager_local_header_honors_offline_before_using_its_configured_origin() {
    let scratch = Scratch::new();
    let data = fixture(&scratch.0);
    let origin = scratch.0.join("origin");
    let producer = Client::new(config(
        &scratch.0.join("producer"),
        &format!("file://{}", origin.display()),
    ))
    .unwrap();
    producer
        .upload_folder(&"account/local@1".parse().unwrap(), &data.folder, None)
        .await
        .unwrap();
    let http = Http::new(origin);
    let header = scratch.0.join("local.cozytensors");
    fs::copy(data.folder.join("model.cozytensors"), &header).unwrap();
    let client = Client::new(config(&scratch.0.join("consumer"), &http.url)).unwrap();
    let source = header.to_str().unwrap();
    assert!(client.open(source, &Open::default()).await.is_ok());
    assert_eq!(client
        .open(source, &Open { lazy: false, local_files_only: true, ..Open::default() })
        .await
        .unwrap_err()
        .kind, Kind::NotFound);
    assert!(http.requests.lock().unwrap().is_empty());
    let eager = client
        .open(source, &Open { lazy: false, ..Open::default() })
        .await
        .unwrap();
    assert_eq!(eager.id(), &data.header);
    assert!(eager.missing(&Selection::All).unwrap().is_empty());
    assert_eq!(eager.component("model").unwrap().read_decoded("w").unwrap(), b"abcdefghijklmnop");
    assert!(!scratch.0.join("chunks").exists());
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

    let support = scratch.0.join("support-files");
    assert_eq!(
        checkpoint
            .component("tokenizer")
            .unwrap()
            .export_files(Some(&support))
            .unwrap_err()
            .kind,
        Kind::NotFound
    );
    assert!(
        !support.exists(),
        "unfetched support-file export created output"
    );
    checkpoint
        .download(&Download {
            selection: Selection::Assets {
                component: "tokenizer".into(),
                names: None,
            },
            ..Download::default()
        })
        .await
        .unwrap();
    let tokenizer = checkpoint.component("tokenizer").unwrap().clone();
    assert_eq!(
        fs::read(
            tokenizer
                .export_files(Some(&support))
                .unwrap()
                .join("tokenizer.json")
        )
        .unwrap(),
        tokenizer.asset("tokenizer.json").unwrap()
    );
    let support_names: BTreeMap<_, _> = fs::read_dir(&support)
        .unwrap()
        .map(|e| {
            let e = e.unwrap();
            (e.file_name(), e.metadata().unwrap().len())
        })
        .collect();
    assert_eq!(
        support_names.len(),
        2,
        "support export must contain only config and the asset"
    );
    assert!(support.join("config.json").is_file());
    assert!(!support.join("chunks").exists());
    assert!(http.fetched(&data.asset));
    assert!(!http.fetched(&data.chunks[0]) && !http.fetched(&data.chunks[2]));

    checkpoint
        .download(&Download {
            selection: Selection::Part {
                component: "model".into(),
                key: "w".into(),
                role: Some(String::new()),
                range: Some(0..4),
            },
            ..Download::default()
        })
        .await
        .unwrap();
    assert!(http.fetched(&data.chunks[0]) && !http.fetched(&data.chunks[1]));
    let component = checkpoint.component("model").unwrap().clone();
    let mut prefix = [0; 4];
    component.read_into("w", &mut prefix, &Default::default()).unwrap();
    assert_eq!(&prefix, b"abcd");
    let invalid = checkpoint
        .download(&Download {
            selection: Selection::Part {
                component: "model".into(),
                key: "w".into(),
                role: Some(String::new()),
                range: Some(15..17),
            },
            ..Download::default()
        })
        .await
        .unwrap_err();
    assert_eq!(invalid.kind, Kind::Invalid);

    checkpoint.download(&Download::default()).await.unwrap();
    assert!(checkpoint.missing(&Selection::All).unwrap().is_empty());
    let selected = Download {
        selection: Selection::Components(vec!["model".into()]),
        ..Download::default()
    };
    let snapshot = client
        .materialize(&reference, &selected, None, false)
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
            .materialize(&reference, &selected, Some(&user_dir), false)
            .await
            .unwrap(),
        user_dir
    );
    assert_eq!(fs::read(user_dir.join("mine")).unwrap(), b"owned by caller");
    let readme = client
        .download_file(
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
    warm.download(&Download::default()).await.unwrap();
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
    #[cfg(unix)]
    let source = checkpoint.source(false);
    drop(warm);
    drop(tokenizer);
    drop(checkpoint);
    drop(client);
    assert_eq!(component.read_decoded("w").unwrap(), b"abcdefghijklmnop");
    #[cfg(target_os = "linux")]
    assert!(
        owned_fds(&cache) > 0,
        "last children must still own retention"
    );
    drop(component);
    #[cfg(unix)]
    drop(source);
    #[cfg(target_os = "linux")]
    {
        for _ in 0..100 {
            if owned_fds(&cache) == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(
            owned_fds(&cache),
            0,
            "final child/view drop retained cache ownership descriptors"
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn thin_folder_and_offline_opens_fetch_only_selected_local_mirror_objects() {
    let scratch = Scratch::new();
    let data = fixture(&scratch.0);
    let mirror = scratch.0.join("mirror");
    let producer = Client::new(config(
        &scratch.0.join("producer"),
        &format!("file://{}", mirror.display()),
    ))
    .unwrap();
    let tree = producer
        .upload_folder(&"account/model@1".parse().unwrap(), &data.folder, None)
        .await
        .unwrap();
    drop(producer);
    let http = Http::new(mirror.clone());
    let thin = scratch.0.join("thin");
    fs::create_dir_all(thin.join("chunks")).unwrap();
    fs::copy(
        data.folder.join("model.cozytensors"),
        thin.join("model.cozytensors"),
    )
    .unwrap();
    fs::copy(
        data.folder.join("chunks").join(&data.chunks[0].sha256),
        thin.join("chunks").join(&data.chunks[0].sha256),
    )
    .unwrap();
    let cache = scratch.0.join("thin-cache");
    let mut options = config(&cache, &http.url);
    options.mirror = Some(mirror.clone());
    options.token = Some(Arc::new(Denied));
    let client = Client::new(options).unwrap();
    let checkpoint = client
        .open(
            thin.to_str().unwrap(),
            &Open {
                local_files_only: true,
                ..Open::default()
            },
        )
        .await
        .unwrap();
    let progress = Mutex::new(Vec::new());
    let report = |held, total| {
        progress.lock().unwrap().push((held, total));
        Ok(())
    };
    let landed = checkpoint
        .download(&Download {
            selection: Selection::Tensors {
                component: "model".into(),
                keys: vec!["w".into()],
            },
            progress: Some(&report),
            ..Download::default()
        })
        .await
        .unwrap();
    assert_eq!(
        landed.local, 8,
        "the folder's existing chunk must not be fetched"
    );
    assert_eq!(landed.origin + landed.volume, 8);
    assert_eq!(
        checkpoint.component("model").unwrap().read_decoded("w").unwrap(),
        b"abcdefghijklmnop"
    );
    assert_eq!(progress.lock().unwrap().last(), Some(&(16, 16)));
    assert_eq!(
        checkpoint
            .missing(&Selection::All)
            .unwrap()
            .into_iter()
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([data.chunks[2].clone(), data.asset.clone()])
    );
    assert!(
        !cache.join(data.chunks[0].key()).exists(),
        "existing folder payload was copied to cache"
    );
    assert!(!cache.join(data.chunks[2].key()).exists());
    assert!(!cache.join(data.asset.key()).exists());

    let offline_cache = scratch.0.join("offline-cache");
    let mut options = config(&offline_cache, &http.url);
    options.mirror = Some(mirror.clone());
    options.token = Some(Arc::new(Denied));
    let offline = Client::new(options).unwrap();
    // Cold immutable metadata must also be discoverable on the mounted mirror.
    let immutable = offline
        .open(
            &tree.id(),
            &Open {
                local_files_only: true,
                ..Open::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(immutable.id(), &data.header);
    let expected: BTreeSet<_> = data
        .chunks
        .iter()
        .cloned()
        .chain(std::iter::once(data.asset.clone()))
        .collect();
    assert_eq!(
        immutable
            .missing(&Selection::All)
            .unwrap()
            .into_iter()
            .collect::<BTreeSet<_>>(),
        expected,
        "All names the exact checkpoint chunks and component assets"
    );
    let named = offline
        .open(
            "account/model",
            &Open {
                local_files_only: true,
                ..Open::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(named.named().unwrap().revision.as_deref(), Some("1"));
    named
        .download(&Download {
            selection: Selection::Assets {
                component: "tokenizer".into(),
                names: None,
            },
            ..Download::default()
        })
        .await
        .unwrap();
    assert_eq!(
        named
            .component("tokenizer")
            .unwrap()
            .asset("tokenizer.json")
            .unwrap(),
        b"{\"vocab\": [\"one\", \"two\"]}"
    );
    assert!(!offline_cache.join(data.chunks[0].key()).exists());
    assert!(!offline_cache.join(data.chunks[2].key()).exists());

    // The mounted mirror is untrusted and must use normal hash admission.
    let corrupt = mirror.join(data.chunks[2].key());
    // A published object is read-only. Replace this fixture's directory entry,
    // rather than mutating an immutable inode also owned by the producer store.
    fs::remove_file(&corrupt).unwrap();
    fs::write(corrupt, b"XXXXXXXX").unwrap();
    let error = checkpoint
        .download(&Download {
            selection: Selection::Components(vec!["other".into()]),
            ..Download::default()
        })
        .await
        .unwrap_err();
    assert_eq!(error.kind, Kind::Integrity);
    assert!(!cache.join(data.chunks[2].key()).exists());
    assert_eq!(
        http.count(),
        0,
        "local-files-only work reached the HTTP origin"
    );
}

struct Denied;
#[async_trait]
impl TokenProvider for Denied {
    async fn headers(&self, _request: &Request<'_>) -> tensorfs::Result<Vec<(String, String)>> {
        Err(Failure {
            kind: Kind::Denied,
            detail: "exact consumer refusal".into(),
        })
    }
}

struct CallbackFailure {
    headers: bool,
    active: AtomicBool,
}

impl CallbackFailure {
    fn error(&self) -> Failure {
        Failure {
            kind: Kind::Conflict,
            detail: "CREDENTIAL_REQUIRED: literal consumer text".into(),
        }
    }
}

#[async_trait]
impl TokenProvider for CallbackFailure {
    async fn headers(&self, _request: &Request<'_>) -> tensorfs::Result<Vec<(String, String)>> {
        if self.headers && self.active.load(Ordering::Relaxed) {
            Err(self.error())
        } else {
            Ok(Vec::new())
        }
    }

    fn retry(
        &self,
        _sent: &Request<'_>,
        _status: u16,
        _headers: &tensorfs::HeaderMap,
    ) -> tensorfs::Result<bool> {
        if !self.headers && self.active.load(Ordering::Relaxed) {
            Err(self.error())
        } else {
            Ok(false)
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn public_provider_headers_and_retry_failures_survive_metadata_and_payload_paths() {
    let scratch = Scratch::new();
    let data = fixture(&scratch.0);
    let origin = scratch.0.join("origin");
    let producer = Client::new(config(
        &scratch.0.join("producer"),
        &format!("file://{}", origin.display()),
    ))
    .unwrap();
    producer
        .upload_folder(&"account/model@1".parse().unwrap(), &data.folder, None)
        .await
        .unwrap();
    let http = Http::new(origin);
    for headers in [true, false] {
        let provider = Arc::new(CallbackFailure {
            headers,
            active: AtomicBool::new(false),
        });
        let mut options = config(&scratch.0.join(format!("cache-{headers}")), &http.url);
        options.token = Some(provider.clone());
        let endpoint = Endpoint::with_config(&http.url, &options).unwrap();
        let client = Client::new(options).unwrap();
        let checkpoint = client
            .open("account/model@1", &Open::default())
            .await
            .unwrap();
        provider.active.store(true, Ordering::Relaxed);
        let expected = provider.error();
        let error = endpoint.repo("account/model").await.unwrap_err();
        assert_eq!(
            (error.kind, error.detail.as_str()),
            (expected.kind, expected.detail.as_str())
        );
        let error = checkpoint
            .download(&Download {
                selection: Selection::Tensors {
                    component: "model".into(),
                    keys: vec!["w".into()],
                },
                ..Download::default()
            })
            .await
            .unwrap_err();
        assert_eq!(
            (error.kind, error.detail.as_str()),
            (expected.kind, expected.detail.as_str())
        );
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
        checkpoint.component("other").unwrap().read_decoded("v").unwrap(),
        b"QRSTUVWX"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn public_names_match_the_generic_format_with_unicode_reserved_characters_and_hidden_names() {
    let scratch = Scratch::new();
    let data = fixture(&scratch.0);
    let origin = scratch.0.join("origin");
    let client = Client::new(config(&scratch.0.join("cache"), origin.to_str().unwrap())).unwrap();
    let offline = Client::new(config(&scratch.0.join("cache"), "http://127.0.0.1:1")).unwrap();
    let endpoint = Endpoint::new(origin.to_str().unwrap(), None).unwrap();
    for name in ["AcMe Space/MoDEL%?#É", "Org%2F/.Hidden", "組/名"] {
        let mut reference: Ref = name.parse().unwrap();
        assert_eq!(reference.repo_id, name);
        reference.revision = Some("1".into());
        let tree = client
            .upload_folder(&reference, &data.folder, None)
            .await
            .unwrap();
        let (repo, _) = endpoint.repo(name).await.unwrap().unwrap();
        assert_eq!(repo.release(Some("1")).unwrap().tree, tree);
        let checkpoint = client
            .open(
                name,
                &Open {
                    revision: Some("1".into()),
                    ..Open::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(checkpoint.id(), &data.header);
        let cached = offline
            .open(
                name,
                &Open {
                    revision: Some("1".into()),
                    local_files_only: true,
                    ..Open::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(cached.id(), &data.header);
        checkpoint.download(&Download::default()).await.unwrap();
        assert_eq!(
            checkpoint.component("model").unwrap().read_decoded("w").unwrap(),
            b"abcdefghijklmnop"
        );
    }
    for invalid in ["./model", "org/..", "org/a\\b", "org/a\0b", "org/a/b"] {
        assert_eq!(invalid.parse::<Ref>().unwrap_err().kind, Kind::Invalid);
    }
    // '@' belongs to the shorthand grammar. A literal '@' identity remains expressible by
    // the ordinary typed Ref and Endpoint doors without a second parser.
    let typed = Ref {
        repo_id: "org/literal@name".into(),
        revision: Some("1".into()),
        variant: None,
        endpoint: None,
    };
    let tree = client
        .upload_folder(&typed, &data.folder, None)
        .await
        .unwrap();
    assert_eq!(
        endpoint
            .repo(&typed.repo_id)
            .await
            .unwrap()
            .unwrap()
            .0
            .release(Some("1"))
            .unwrap()
            .tree,
        tree
    );
}

#[tokio::test(flavor = "current_thread")]
async fn advanced_exact_object_fetch_does_not_widen_the_window_and_uses_the_existing_engine() {
    let scratch = Scratch::new();
    let data = fixture(&scratch.0);
    let origin = scratch.0.join("origin");
    let producer = Client::new(config(
        &scratch.0.join("source-cache"),
        origin.to_str().unwrap(),
    ))
    .unwrap();
    let tree = producer
        .upload_folder(&"acme/exact".parse().unwrap(), &data.folder, None)
        .await
        .unwrap();
    let http = Http::new(origin);
    let cache = scratch.0.join("reader-cache");
    let reader = Client::new(config(&cache, &http.url)).unwrap();
    let checkpoint = reader.open(&tree.id(), &Open::default()).await.unwrap();
    let readme = fs::read(data.folder.join("README.md")).unwrap();
    let repo_object = ObjectRef::of(&readme);
    assert!(!checkpoint
        .missing(&Selection::All)
        .unwrap()
        .contains(&repo_object));
    assert!(!checkpoint
        .verify(&Selection::All)
        .unwrap()
        .missing
        .contains(&repo_object));
    let before_invalid = http.requests.lock().unwrap().len();
    assert_eq!(
        tensorfs::raw::fetch_objects(
            &checkpoint,
            &[ObjectRef::of(b"unrelated")],
            &Download::default()
        )
        .await
        .unwrap_err()
        .kind,
        Kind::Invalid
    );
    let wrong_length = ObjectRef {
        length: data.chunks[0].length + 1,
        ..data.chunks[0].clone()
    };
    assert_eq!(
        tensorfs::raw::fetch_objects(&checkpoint, &[wrong_length], &Download::default())
            .await
            .unwrap_err()
            .kind,
        Kind::Invalid
    );
    let wrong_repo_length = ObjectRef {
        length: repo_object.length + 1,
        ..repo_object.clone()
    };
    assert_eq!(
        tensorfs::raw::fetch_objects(&checkpoint, &[wrong_repo_length], &Download::default())
            .await
            .unwrap_err()
            .kind,
        Kind::Invalid
    );
    assert_eq!(
        http.requests.lock().unwrap().len(),
        before_invalid,
        "an invalid exact reference started transport I/O"
    );
    tensorfs::raw::fetch_objects(&checkpoint, &data.chunks[..1], &Download::default())
        .await
        .unwrap();
    assert!(cache.join(data.chunks[0].key()).exists());
    assert!(!cache.join(data.chunks[1].key()).exists());
    assert!(!cache.join(data.chunks[2].key()).exists());
    assert!(!cache.join(data.asset.key()).exists());
    let mut bytes = [0; 4];
    let target = tensorfs::io::Target {
        component: "model",
        key: "w",
        role: "",
        offset: 2,
        dst: tensorfs::io::Dst::Host {
            ptr: bytes.as_mut_ptr(),
            len: bytes.len(),
        },
    };
    // SAFETY: bytes is exclusively borrowed and remains valid until the synchronous engine returns.
    unsafe {
        tensorfs::raw::read_into(&checkpoint, &[target], &tensorfs::io::ReadOpts::default())
            .unwrap();
    }
    assert_eq!(&bytes, b"cdef");
    // Only the explicitly named repository object moves; it never widens the tensor window.
    tensorfs::raw::fetch_objects(
        &checkpoint,
        std::slice::from_ref(&repo_object),
        &Download::default(),
    )
    .await
    .unwrap();
    assert_eq!(fs::read(cache.join(repo_object.key())).unwrap(), readme);
    assert!(http.fetched(&repo_object));
    assert!(!cache.join(data.chunks[1].key()).exists());
    assert!(!cache.join(data.chunks[2].key()).exists());
    assert!(!cache.join(data.asset.key()).exists());

    fs::remove_file(cache.join(repo_object.key())).unwrap();
    checkpoint.download(&Download::default()).await.unwrap();
    assert!(checkpoint.missing(&Selection::All).unwrap().is_empty());
    assert!(checkpoint.verify(&Selection::All).unwrap().is_ok());
    assert!(
        !cache.join(repo_object.key()).exists(),
        "checkpoint All downloaded a repository object"
    );

    // Explicit tree permission survives missing metadata; All and verification ignore it.
    fs::remove_file(cache.join(tree.key())).unwrap();
    assert!(checkpoint.missing(&Selection::All).unwrap().is_empty());
    assert!(checkpoint.verify(&Selection::All).unwrap().is_ok());
    tensorfs::raw::fetch_objects(
        &checkpoint,
        std::slice::from_ref(&tree),
        &Download::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        ObjectRef::of(&fs::read(cache.join(tree.key())).unwrap()),
        tree
    );
    tensorfs::raw::fetch_objects(
        &checkpoint,
        std::slice::from_ref(&repo_object),
        &Download::default(),
    )
    .await
    .unwrap();
    assert_eq!(fs::read(cache.join(repo_object.key())).unwrap(), readme);
}

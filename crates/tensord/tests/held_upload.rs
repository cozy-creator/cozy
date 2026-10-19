//! Real no-code warm preparation uploads an exact held TREE with ordinary TensorFS access.
use axum::{
    body::Bytes,
    extract::State as Extract,
    http::{HeaderMap, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    io,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU8, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};
use tensord::{
    api::domain,
    journal::{Execution, State},
    objects::Objects,
    published::Publisher,
    runs::{Runs, Source as RunSource, Spec},
    service::Service,
};
use tensorfs_core::{
    fetch::{Client, Endpoint, SourcePolicy},
    format::{Dtype, ObjectRef, Repo, Tree},
    store2::Store,
    writer::{Data, Source},
};

const TOKEN: &str = "held-upload-test-bearer";

struct Fixture {
    _root: tempfile::TempDir,
    runs: Arc<Runs>,
    store: Arc<Store>,
    tree: ObjectRef,
    tree_digest: Vec<u8>,
    header: ObjectRef,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let store = tensord::storage::open(&root.path().join("store")).unwrap();
        let header = |tag: u8| {
            let mut writer = store.writer(None, None).unwrap();
            let data = vec![tag; 16384];
            writer
                .get("model")
                .put(
                    "weight",
                    &Dtype::U8,
                    &[data.len() as u64],
                    Data::Plain(Source::Read(&mut &data[..])),
                )
                .unwrap();
            writer.commit().unwrap()
        };
        let selected = header(19);
        let other = header(23);
        let mut tree = Tree::default();
        tree.set_variant("f32", selected.clone());
        tree.set_variant("other", other.clone());
        tree.files.insert(
            "README.md".into(),
            store.put_bytes(b"retained repo file").unwrap(),
        );
        let tree = store.put_tree(&tree).unwrap();
        let tree_digest = tensorfs_core::sha256::digest(&store.read(&tree).unwrap()).to_vec();
        // The newest local name is deliberately different from the exact retained output.
        store
            .set_variant(
                "local/derived",
                "newer",
                Some("f32"),
                &other,
                &BTreeMap::new(),
            )
            .unwrap();
        let service = Service::open(
            &root.path().join("state"),
            &root.path().join("generations"),
            1,
        )
        .unwrap();
        let publisher = Publisher::new(
            &root.path().join("published"),
            Default::default(),
            tensord::origin::Machine::new(store.clone(), None).unwrap(),
        )
        .unwrap();
        let objects = Arc::new(
            Objects::new(
                &root.path().join("objects"),
                store.clone(),
                service.engine.clone(),
            )
            .unwrap(),
        );
        let runs = Arc::new(Runs {
            service,
            objects,
            publisher: Some(publisher),
            local: None,
            own_hub: None,
            leaf: None,
            contexts: Default::default(),
        });
        Self {
            _root: root,
            runs,
            store,
            tree,
            tree_digest,
            header: selected,
        }
    }

    fn submit(&self, id: &str, endpoint: &str, variant: &str) -> String {
        let digest = self.tree_digest.clone();
        let spec = Spec {
            warm: true,
            remove: false,
            set: None,
            parent: String::new(),
            source: RunSource::Models,
            entrypoint: String::new(),
            input: json!({}),
            inputs: vec![],
            models: vec![domain::ModelChoice {
                parameter: "model".into(),
                repository: "local/derived".into(),
                manifest: Some(domain::Ref { digest, length: 0 }),
                lane: variant.into(),
                ..Default::default()
            }],
            attention_kernel: String::new(),
            gpus: 0,
            bits: 0,
            hub: None,
            providers: Default::default(),
            tensorfs: Some(protocol::machine::TensorFSOptions {
                endpoint: endpoint.into(),
                token: TOKEN.into(),
                ..Default::default()
            }),
            weights_destination: "owner/retry".into(),
            owner: "owner".into(),
            catalog_revision: String::new(),
            application: String::new(),
            digest: id.into(),
            journaled: vec![],
        };
        self.runs.submit("owner", id, spec).unwrap().id
    }

    fn settled(&self, id: &str) -> Execution {
        let began = Instant::now();
        let mut seen = self.runs.service.engine.activity_epoch();
        loop {
            let record = self.runs.service.engine.get(id).unwrap();
            if record.state.terminal() {
                return record;
            }
            assert!(
                began.elapsed() < Duration::from_secs(15),
                "warm upload did not settle"
            );
            seen = self
                .runs
                .service
                .engine
                .wait_activity(seen, Some(Duration::from_millis(100)));
        }
    }
}

struct HttpState {
    store: Store,
    mode: AtomicU8, // 1: hold blob PUT, 2: hold ref PUT, 3: refuse ref PUT.
    entered: mpsc::Sender<()>,
    released: tokio::sync::watch::Sender<bool>,
    puts: Mutex<Vec<String>>,
}

async fn answer(
    Extract(state): Extract<Arc<HttpState>>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    assert_eq!(
        headers.get("authorization").and_then(|v| v.to_str().ok()),
        Some("Bearer held-upload-test-bearer")
    );
    let key = uri.path().trim_start_matches('/');
    let is_ref = key.starts_with("refs/");
    if method == Method::PUT {
        let mode = state.mode.load(Ordering::Acquire);
        if (mode == 1 && !is_ref) || (mode == 2 && is_ref) {
            let mut released = state.released.subscribe();
            state.entered.send(()).unwrap();
            while !*released.borrow() {
                released.changed().await.unwrap();
            }
        }
        if is_ref && mode == 3 {
            return StatusCode::FORBIDDEN.into_response();
        }
        if is_ref {
            Repo::from_json(&body).unwrap();
            let path = state.store.root().join(key);
            let old = std::fs::read(&path).ok();
            let etag = old
                .as_ref()
                .map(|bytes| format!("\"{}\"", tensorfs_core::sha256::hex_digest(bytes)));
            if headers
                .get("if-match")
                .and_then(|v| v.to_str().ok())
                .is_some_and(|asked| etag.as_deref() != Some(asked))
                || (headers.contains_key("if-none-match") && old.is_some())
            {
                return StatusCode::PRECONDITION_FAILED.into_response();
            }
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            platform::durable::replace(&path, &body, 0o644).unwrap();
        } else {
            let object = ObjectRef {
                sha256: key.rsplit('/').next().unwrap().into(),
                length: body.len() as u64,
            };
            assert_eq!(key, object.key());
            state.store.put(&body[..], Some(&object)).unwrap();
            state.puts.lock().unwrap().push(object.id());
        }
        return StatusCode::NO_CONTENT.into_response();
    }
    match std::fs::read(state.store.root().join(key)) {
        Ok(bytes) => {
            let etag = format!("\"{}\"", tensorfs_core::sha256::hex_digest(&bytes));
            (
                [("etag", etag), ("content-length", bytes.len().to_string())],
                bytes,
            )
                .into_response()
        }
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

struct Http {
    _root: tempfile::TempDir,
    origin: String,
    state: Arc<HttpState>,
    entered: mpsc::Receiver<()>,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Http {
    fn new(mode: u8) -> Self {
        let root = tempfile::tempdir().unwrap();
        let (entered, receiver) = mpsc::channel();
        let state = Arc::new(HttpState {
            store: Store::open(root.path()).unwrap(),
            mode: AtomicU8::new(mode),
            entered,
            released: tokio::sync::watch::channel(false).0,
            puts: Mutex::new(vec![]),
        });
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let app = axum::Router::new()
            .fallback(answer)
            .with_state(state.clone());
        let (shutdown, stopped) = tokio::sync::oneshot::channel();
        let thread = std::thread::spawn(move || {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap()
                .block_on(async {
                    axum::serve(tokio::net::TcpListener::from_std(listener).unwrap(), app)
                        .with_graceful_shutdown(async {
                            let _ = stopped.await;
                        })
                        .await
                        .unwrap()
                })
        });
        Self {
            _root: root,
            origin,
            state,
            entered: receiver,
            shutdown: Some(shutdown),
            thread: Some(thread),
        }
    }

    fn release(&self) {
        self.state.released.send_replace(true);
    }
    fn wait(&self) {
        self.entered.recv_timeout(Duration::from_secs(10)).unwrap();
    }
    fn published(&self) -> bool {
        self.state
            .store
            .root()
            .join("refs/owner/retry.json")
            .exists()
    }
}

impl Drop for Http {
    fn drop(&mut self) {
        self.release();
        let _ = self.shutdown.take().unwrap().send(());
        self.thread.take().unwrap().join().unwrap();
    }
}

#[test]
fn a_failed_ref_update_retries_the_exact_held_tree_without_uploading_admitted_blobs() {
    let fixture = Fixture::new();
    let http = Http::new(3);
    let failed = fixture.submit("first", &http.origin, "f32");
    let failure = fixture.settled(&failed);
    assert_eq!(failure.state, State::Failed);
    assert_eq!(
        tensord::journal::Failure::decode(failure.failure.as_deref().unwrap())
            .unwrap()
            .code,
        "upload_failed"
    );
    assert!(!http.published());
    assert_eq!(
        fixture
            .store
            .checkpoint(&fixture.header)
            .unwrap()
            .read("model", "weight")
            .unwrap(),
        vec![19; 16384]
    );
    let sent = http.state.puts.lock().unwrap().len();
    assert!(sent > 0);
    http.state.mode.store(0, Ordering::Release);
    let retry = fixture.submit("retry", &http.origin, "f32");
    let result = fixture.settled(&retry);
    assert_eq!(result.state, State::Completed, "{:?}", result.failure);
    assert_eq!(
        http.state.puts.lock().unwrap().len(),
        sent,
        "admitted blobs were uploaded again"
    );
    let row = &result.result.as_ref().unwrap().value["models"][0];
    assert_eq!(row["tree"], fixture.tree.id());
    assert_eq!(row["lane"], "f32");
    assert_eq!(row["published"]["tree"], fixture.tree.id());
    assert_eq!(row["published"]["checkpoint"], fixture.header.id());
    assert_eq!(row["published"]["converged"], false);
    let repo = Repo::from_json(
        &std::fs::read(http.state.store.root().join("refs/owner/retry.json")).unwrap(),
    )
    .unwrap();
    assert!(repo.checkpoints.contains(&fixture.tree));
    assert_eq!(
        http.state.store.tree(&fixture.tree).unwrap(),
        fixture.store.tree(&fixture.tree).unwrap(),
        "all variants and repo files remain"
    );
    let client = Client::new(SourcePolicy {
        any_host: true,
        allow_local: true,
        ..Default::default()
    });
    let endpoint = Endpoint::new(
        &client,
        &http.origin,
        Some(tensorfs_core::fetch::bearer(TOKEN)),
    )
    .unwrap();
    assert!(
        tensorfs_core::fetch::runtime()
            .unwrap()
            .block_on(endpoint.has(&fixture.header))
            .unwrap()
    );
    assert_eq!(
        fixture.settled(&failed).state,
        State::Failed,
        "retry does not rewrite the original failure"
    );
    let repeat = fixture.submit("repeat", &http.origin, "f32");
    let repeat = fixture.settled(&repeat);
    assert_eq!(repeat.state, State::Completed);
    assert_eq!(
        repeat.result.unwrap().value["models"][0]["published"]["converged"],
        true
    );
    assert_eq!(http.state.puts.lock().unwrap().len(), sent);
}

#[test]
fn held_tree_variant_errors_do_not_publish_or_fall_back_to_the_hub() {
    let fixture = Fixture::new();
    let http = Http::new(0);
    for (id, variant) in [("missing", "absent"), ("ambiguous", "")] {
        let run = fixture.submit(id, &http.origin, variant);
        let failure = fixture.settled(&run);
        assert_eq!(failure.state, State::Failed);
        assert_eq!(
            tensord::journal::Failure::decode(failure.failure.as_deref().unwrap())
                .unwrap()
                .code,
            "checkpoint_variant_absent"
        );
    }
    std::fs::remove_file(fixture.store.path(&fixture.tree).unwrap()).unwrap();
    let absent = fixture.submit("missing-local-tree", &http.origin, "f32");
    let failure = fixture.settled(&absent);
    assert_eq!(failure.state, State::Failed);
    assert_eq!(
        tensord::journal::Failure::decode(failure.failure.as_deref().unwrap())
            .unwrap()
            .code,
        "checkpoint_absent"
    );
    assert!(http.state.puts.lock().unwrap().is_empty());
    assert!(!http.published());
}

#[test]
fn cancellation_stops_before_ref_finalization_and_is_refused_after_commit() {
    let fixture = Fixture::new();
    let before = Http::new(1);
    let run = fixture.submit("cancel-before", &before.origin, "f32");
    before.wait();
    fixture.runs.service.engine.cancel(&run, "owner").unwrap();
    before.release();
    assert_eq!(fixture.settled(&run).state, State::Canceled);
    assert!(!before.published());

    let after = Http::new(2);
    let run = fixture.submit("cancel-after", &after.origin, "f32");
    after.wait();
    assert_eq!(
        fixture
            .runs
            .service
            .engine
            .cancel(&run, "owner")
            .unwrap_err()
            .kind(),
        io::ErrorKind::ResourceBusy
    );
    after.release();
    assert_eq!(fixture.settled(&run).state, State::Completed);
    assert!(after.published());
}

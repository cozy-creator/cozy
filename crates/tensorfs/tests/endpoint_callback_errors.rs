//! Provider failures are caller verdicts, even when their code also describes a transport error.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tensorfs::err::{Code, Refusal, Result};
use tensorfs::fetch::token::{HeaderMap, Request};
use tensorfs::fetch::{
    Client, Config, Endpoint, Fetcher, ObjectStore, Origin, Payload, SourcePolicy, TokenProvider,
    Want,
};
use tensorfs::format::ObjectRef;
use tensorfs::store2::Store;

const DETAIL: &str = "the credential callback's literal network failure";
const PAYLOAD: &[u8] = b"ordinary endpoint payload";

struct Server {
    child: Child,
    root: PathBuf,
    url: String,
}

impl Server {
    fn new(broken_first: bool) -> Self {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("tfs-callback-{}-{unique}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let mut child = Command::new("python3").args(["-u", "-c", r#"
import http.server, pathlib, sys
root, broken_first = pathlib.Path(sys.argv[1]), sys.argv[2] == 'true'
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def answer(self):
        log = root / 'requests'
        previous = len(log.read_text().splitlines()) if log.exists() else 0
        with log.open('a') as out: out.write(self.command + ' ' + self.path + '\n')
        if broken_first and previous == 0:
            self.wfile.write(b'not an HTTP response\r\n\r\n')
            self.wfile.flush()
            self.close_connection = True
            return
        body = b'ordinary endpoint payload' if self.path.startswith('/blobs/') and self.command == 'GET' else b''
        self.send_response(200 if body else 404)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.end_headers()
        if self.command != 'HEAD': self.wfile.write(body)
    do_GET = do_HEAD = do_PUT = answer
server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
print(server.server_port, flush=True)
server.serve_forever()
"#]).arg(&root).arg(broken_first.to_string()).stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().unwrap();
        let mut port = String::new();
        BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut port)
            .unwrap();
        Self {
            child,
            root,
            url: format!("http://127.0.0.1:{}", port.trim()),
        }
    }

    fn requests(&self) -> Vec<String> {
        std::fs::read_to_string(self.root.join("requests"))
            .unwrap_or_default()
            .lines()
            .map(str::to_owned)
            .collect()
    }

    fn endpoint(&self, provider: Arc<Provider>) -> Endpoint {
        Endpoint::new(&client(), &self.url, Some(provider)).unwrap()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

#[derive(Clone, Copy)]
enum FailureAt {
    Headers(Code),
    Put(Code),
    Retry(Code),
}

#[derive(Default)]
struct Provider {
    failure: Option<FailureAt>,
    headers: Mutex<Vec<(String, Option<String>)>>,
    retries: AtomicUsize,
}

impl Provider {
    fn failing(failure: FailureAt) -> Arc<Self> {
        Arc::new(Self {
            failure: Some(failure),
            ..Default::default()
        })
    }

    fn calls(&self) -> usize {
        self.headers.lock().unwrap().len()
    }
}

#[async_trait]
impl TokenProvider for Provider {
    async fn headers(&self, request: &Request<'_>) -> Result<Vec<(String, String)>> {
        self.headers
            .lock()
            .unwrap()
            .push((request.method.into(), request.payload.map(str::to_owned)));
        match self.failure {
            Some(FailureAt::Headers(code)) => Err(Refusal {
                code,
                detail: DETAIL.into(),
            }),
            Some(FailureAt::Put(code)) if request.method == "PUT" => Err(Refusal {
                code,
                detail: DETAIL.into(),
            }),
            _ => Ok(vec![]),
        }
    }

    fn retry(&self, _: &Request<'_>, _: u16, _: &HeaderMap) -> Result<bool> {
        self.retries.fetch_add(1, Ordering::Relaxed);
        match self.failure {
            Some(FailureAt::Retry(code)) => Err(Refusal {
                code,
                detail: DETAIL.into(),
            }),
            _ => Ok(false),
        }
    }
}

fn client() -> Client {
    Client::new(SourcePolicy {
        any_host: true,
        allow_local: true,
        ..Default::default()
    })
}

fn literal(error: Refusal, code: Code) {
    assert_eq!(error.code, code);
    assert_eq!(error.detail, DETAIL);
}

async fn promptly<T>(future: impl std::future::Future<Output = Result<T>>) -> Result<T> {
    tokio::time::timeout(Duration::from_secs(3), future)
        .await
        .expect("provider failure was retried instead of returned")
}

#[tokio::test]
async fn metadata_headers_fail_once_without_an_http_request() {
    for code in [
        Code::TRANSFER_FAILED,
        Code::HUB_UNREACHABLE,
        Code::FD_HEADROOM,
        Code::STALLED,
        Code::CREDENTIAL_REQUIRED,
        Code::TRANSACTION_CONFLICT,
        Code::MALFORMED_JSON,
    ] {
        for method in ["GET", "HEAD", "PUT"] {
            let server = Server::new(false);
            let provider = Provider::failing(FailureAt::Headers(code));
            let endpoint = server.endpoint(provider.clone());
            let error = match method {
                "GET" => promptly(endpoint.repo("ordinary/model")).await.unwrap_err(),
                "HEAD" => promptly(endpoint.has(&ObjectRef::of(PAYLOAD)))
                    .await
                    .unwrap_err(),
                _ => promptly(endpoint.put_repo_bytes("ordinary/model", b"{}".to_vec(), None))
                    .await
                    .unwrap_err(),
            };
            literal(error, code);
            assert_eq!(provider.calls(), 1, "{method}: {code:?}");
            assert_eq!(provider.retries.load(Ordering::Relaxed), 0);
            assert!(server.requests().is_empty());
        }
    }
}

#[tokio::test]
async fn upload_headers_failure_preserves_the_payload_without_sending_it() {
    let server = Server::new(false);
    let provider = Provider::failing(FailureAt::Put(Code::TRANSFER_FAILED));
    let endpoint = server.endpoint(provider.clone());
    let reads = AtomicUsize::new(0);
    let payload = || {
        reads.fetch_add(1, Ordering::Relaxed);
        Ok(Payload::Bytes(PAYLOAD.into()))
    };
    literal(
        promptly(endpoint.upload(&ObjectRef::of(PAYLOAD), &payload))
            .await
            .unwrap_err(),
        Code::TRANSFER_FAILED,
    );
    assert_eq!(reads.load(Ordering::Relaxed), 0);
    assert_eq!(
        provider.calls(),
        2,
        "one successful HEAD then the refused PUT"
    );
    assert_eq!(server.requests().len(), 1);
    assert!(server.requests()[0].starts_with("HEAD "));
    let seen = provider.headers.lock().unwrap();
    assert_eq!(seen[1], ("PUT".into(), Some(ObjectRef::of(PAYLOAD).sha256)));
}

#[tokio::test]
async fn retry_callback_failure_is_literal_for_metadata() {
    let server = Server::new(false);
    let provider = Provider::failing(FailureAt::Retry(Code::TRANSFER_FAILED));
    literal(
        promptly(server.endpoint(provider.clone()).repo("ordinary/model"))
            .await
            .unwrap_err(),
        Code::TRANSFER_FAILED,
    );
    assert_eq!(provider.calls(), 1);
    assert_eq!(provider.retries.load(Ordering::Relaxed), 1);
    assert_eq!(server.requests().len(), 1);
}

#[tokio::test]
async fn payload_headers_and_retry_failures_are_terminal() {
    for failure in [
        FailureAt::Headers(Code::TRANSFER_FAILED),
        FailureAt::Retry(Code::TRANSFER_FAILED),
    ] {
        let server = Server::new(false);
        let provider = Provider::failing(failure);
        let origin: Arc<dyn Origin> = Arc::new(server.endpoint(provider.clone()));
        let store = Store::open(server.root.join("store")).unwrap();
        let fetcher = Fetcher::new(
            Arc::new(store.clone()) as Arc<dyn ObjectStore>,
            client(),
            None,
            Config::default(),
        );
        literal(
            promptly(fetcher.objects(&origin, &[ObjectRef::of(PAYLOAD)], &Want::default()))
                .await
                .unwrap_err(),
            Code::TRANSFER_FAILED,
        );
        fetcher.settled().await;
        assert_eq!(provider.calls(), 1);
        let expected = usize::from(matches!(failure, FailureAt::Retry(_)));
        assert_eq!(provider.retries.load(Ordering::Relaxed), expected);
        assert_eq!(server.requests().len(), expected);
        assert!(!store.has(&ObjectRef::of(PAYLOAD)));
    }
}

#[tokio::test]
async fn real_transport_failures_still_retry_metadata_and_payload() {
    let server = Server::new(true);
    let provider = Arc::new(Provider::default());
    assert!(
        promptly(server.endpoint(provider.clone()).repo("ordinary/model"))
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(provider.calls(), 2);
    assert_eq!(server.requests().len(), 2);

    let server = Server::new(true);
    let provider = Arc::new(Provider::default());
    let origin: Arc<dyn Origin> = Arc::new(server.endpoint(provider.clone()));
    let store = Store::open(server.root.join("store")).unwrap();
    let fetcher = Fetcher::new(
        Arc::new(store.clone()) as Arc<dyn ObjectStore>,
        client(),
        None,
        Config::default(),
    );
    promptly(fetcher.objects(&origin, &[ObjectRef::of(PAYLOAD)], &Want::default()))
        .await
        .unwrap();
    fetcher.settled().await;
    assert_eq!(store.read(&ObjectRef::of(PAYLOAD)).unwrap(), PAYLOAD);
    assert_eq!(provider.calls(), 2);
    assert_eq!(server.requests().len(), 2);
}

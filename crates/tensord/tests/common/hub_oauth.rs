//! A real AuthKit authorization server and DPoP resource server (`tests/hub-oauth`, Go) in
//! front of a test Hub (th-241). The harness signs run capabilities with an owner's enrolled
//! device key, as the CLI does; the machine trades them through AuthKit's JWT-bearer grant, and
//! AuthKit verifies every proof. It plays the Hub: its grant decision (which may narrow) and
//! its resource checks (the token's operations, the device key still live). Needs `go`, and
//! AUTHKIT_TEST_DATABASE_URL naming a PostgreSQL the harness may create schemas in.
#![allow(dead_code)]
use serde_json::Value;
use std::{
    collections::BTreeMap,
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio},
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};

pub struct AuthKit {
    child: Child,
    stdin: Option<ChildStdin>,
    pub issuer: String,
    pub resource: String,
    /// Where a capability is traded, as the CLI hands it over beside it.
    pub token_endpoint: String,
    /// The Hub origin a machine reaches: AuthKit's resource server, forwarding to the test Hub.
    pub hub: String,
    control: String,
    dir: PathBuf,
}

fn harness() -> &'static PathBuf {
    static BUILT: OnceLock<PathBuf> = OnceLock::new();
    BUILT.get_or_init(|| {
        let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("hub-oauth-harness");
        let built = Command::new("go")
            .args(["test", "-c", "-o"])
            .arg(&out)
            .arg(".")
            .current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/hub-oauth"))
            .status()
            .expect("go builds the AuthKit harness");
        assert!(built.success(), "building tests/hub-oauth");
        out
    })
}

impl AuthKit {
    /// Serves AuthKit in front of `upstream` until dropped.
    pub fn start(upstream: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("cm-authkit-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let ready = dir.join("ready.json");
        let log = std::fs::File::create(dir.join("harness.log")).unwrap();
        let mut child = Command::new(harness())
            .args(["-test.run", "^TestHarness$", "-test.v", "-control"])
            .arg(&ready)
            .args(["-upstream", upstream])
            .stdin(Stdio::piped())
            .stdout(log.try_clone().unwrap())
            .stderr(log)
            .spawn()
            .unwrap();
        let started = Instant::now();
        while !ready.exists() {
            if let Some(status) = child.try_wait().unwrap() {
                panic!(
                    "the AuthKit harness exited ({status}): {}",
                    dir.join("harness.log").display()
                );
            }
            assert!(
                started.elapsed() < Duration::from_secs(120),
                "the AuthKit harness never became ready"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
        let ready: Value = serde_json::from_slice(&std::fs::read(&ready).unwrap()).unwrap();
        let field = |name: &str| ready[name].as_str().unwrap().to_string();
        Self {
            stdin: child.stdin.take(),
            child,
            issuer: field("issuer"),
            resource: field("resource"),
            token_endpoint: field("token_endpoint"),
            hub: field("hub"),
            control: field("control"),
            dir,
        }
    }

    /// One form to the harness's control door, from any thread (a test's runtime included).
    fn post(&self, path: &str, form: &[(&str, &str)]) -> (u16, Vec<u8>) {
        let url = format!("{}{path}", self.control);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
                runtime.block_on(async {
                    let answer = reqwest::Client::new().post(&url).form(form).send().await.unwrap();
                    (answer.status().as_u16(), answer.bytes().await.unwrap().to_vec())
                })
            })
            .join()
            .unwrap()
        })
    }

    /// A capability the owner's device key signs for the machine key `jkt`: `ops` are its
    /// `authorization_details`, and it expires in `seconds`.
    pub fn capability(&self, jkt: &str, ops: &Value, seconds: u64) -> String {
        let (status, body) = self.post(
            "/capability",
            &[("jkt", jkt), ("ops", &ops.to_string()), ("seconds", &seconds.to_string())],
        );
        assert_eq!(status, 200, "{}", String::from_utf8_lossy(&body));
        String::from_utf8(body).unwrap()
    }

    /// The owner signs out of that device: its key, and every capability it signed, end.
    pub fn revoke_device_key(&self) {
        assert_eq!(self.post("/revoke-device-key", &[]).0, 204);
    }

    /// The owner signs in on a new device: later capabilities are signed by its key.
    pub fn enroll_device_key(&self) {
        assert_eq!(self.post("/enroll-device-key", &[]).0, 204);
    }

    /// What the Hub saw: `verified` (each token-bearing request AuthKit's resource server
    /// verified: method, path, owner, actor, token digest), `exchanges` (grant decisions),
    /// `owner` (the owner's user id) and `last_token`.
    pub fn seen(&self) -> Value {
        let url = format!("{}/seen", self.control);
        let body = std::thread::scope(|scope| {
            scope.spawn(|| {
                let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
                runtime.block_on(async { reqwest::get(&url).await.unwrap().bytes().await.unwrap().to_vec() })
            })
            .join()
            .unwrap()
        });
        serde_json::from_slice(&body).unwrap()
    }

    pub fn verified(&self) -> Vec<Value> {
        self.seen()["verified"].as_array().cloned().unwrap_or_default()
    }

    pub fn exchanges(&self) -> u64 {
        self.seen()["exchanges"].as_u64().unwrap()
    }

    pub fn owner(&self) -> String {
        self.seen()["owner"].as_str().unwrap().to_string()
    }

    pub fn last_token(&self) -> String {
        self.seen()["last_token"].as_str().unwrap().to_string()
    }
}

impl Drop for AuthKit {
    fn drop(&mut self) {
        // Closing its stdin ends the harness, which drops its scratch schema.
        drop(self.stdin.take());
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// What the test Hub saw. Every route but uploads sits behind AuthKit's resource server.
#[derive(Default)]
pub struct Hub {
    pub reads: usize,
    pub declared: Vec<String>,
    pub uploaded: BTreeMap<String, Vec<u8>>,
    /// Uploads that carried a token or proof: object hosts must see none.
    pub credentialed_uploads: usize,
    pub finalized: Vec<String>,
}

/// The Hub routes a catalog read and a checkpoint publication use. Object uploads go to
/// `localhost`, another host than the Hub's `127.0.0.1`, as presigned URLs do.
pub fn test_hub() -> (String, Arc<Mutex<Hub>>) {
    use axum::{
        body::Bytes,
        extract::State,
        http::{HeaderMap, Method, StatusCode, Uri},
        response::IntoResponse,
    };
    type Held = Arc<Mutex<Hub>>;
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let port = listener.local_addr().unwrap().port();
    let held: Held = Default::default();
    let base = format!("http://localhost:{port}");
    let answer = move |State(held): State<Held>,
                       method: Method,
                       uri: Uri,
                       headers: HeaderMap,
                       body: Bytes| {
        let base = base.clone();
        async move {
            let json = |value: Value| {
                (StatusCode::OK, serde_json::to_vec(&value).unwrap()).into_response()
            };
            let ids = |field: &str| -> Vec<String> {
                let body: Value = serde_json::from_slice(&body).unwrap();
                body[field]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|row| {
                        row.get("object_id")
                            .unwrap_or(row)
                            .as_str()
                            .unwrap()
                            .to_string()
                    })
                    .collect()
            };
            let mut held = held.lock().unwrap();
            let path = uri.path().to_string();
            if let Some(hex) = path.strip_prefix("/upload/") {
                let id = format!("sha256:{hex}");
                assert_eq!(
                    id,
                    format!("sha256:{}", tensorfs::sha256::hex_digest(&body))
                );
                if headers.contains_key("authorization") || headers.contains_key("dpop") {
                    held.credentialed_uploads += 1;
                }
                held.uploaded.insert(id, body.to_vec());
                return StatusCode::OK.into_response();
            }
            // Package reads are public and anonymous; everything else came with a token.
            let owner = headers.get("x-verified-owner").and_then(|v| v.to_str().ok());
            if path.starts_with("/v1/packages/") {
                assert_eq!(owner, Some("anonymous"), "{path} is a public read");
            } else {
                assert!(owner.is_some_and(|o| o != "anonymous"), "{path} reached the Hub unverified");
            }
            let publication = "/v1/models/acme/tiny/publications/";
            match (method, path.as_str()) {
                (Method::GET, "/v1/packages/acme/pkg") => {
                    held.reads += 1;
                    json(serde_json::json!({"package": "acme/pkg"}))
                }
                (Method::GET, p) if p.starts_with("/v1/models/acme/tiny/checkpoints/") => (
                    StatusCode::NOT_FOUND,
                    r#"{"error":{"code":"model.checkpoint_not_found","message":"absent","remedy":""}}"#,
                )
                    .into_response(),
                // The Hub's own answers, as api/openapi.yaml spells them.
                (Method::PUT, p) if p.starts_with(publication) => {
                    let declared: Value = serde_json::from_slice(&body).unwrap();
                    let rows: Vec<Value> = declared["objects"].as_array().unwrap().iter()
                        .map(|row| serde_json::json!({"object_id": row["object_id"], "length": row["length"], "state": "pending"}))
                        .collect();
                    held.declared = ids("objects");
                    let operation = &p[publication.len()..];
                    json(serde_json::json!({"created": true,
                        "publication": {"operation": operation, "state": "open", "objects": rows}}))
                }
                (Method::POST, p) if p.starts_with(publication) && p.ends_with("/grants") => {
                    let grants: Vec<_> = ids("object_ids")
                        .into_iter()
                        .map(|id| {
                            serde_json::json!({"object_id": id, "length": 0,
                            "url": format!("{base}/upload/{}", &id[7..]), "required_headers": {},
                            "expires_at_unix": 0})
                        })
                        .collect();
                    json(serde_json::json!({ "grants": grants, "held": [], "server_time_unix": 0 }))
                }
                (Method::POST, p) if p.starts_with(publication) && p.ends_with("/verify") => {
                    json(serde_json::json!({"publication": {"operation": "op", "state": "open", "objects": []}}))
                }
                (Method::POST, p) if p.starts_with(publication) && p.ends_with("/finalize") => {
                    let body: Value = serde_json::from_slice(&body).unwrap();
                    held.finalized
                        .push(body["manifest_id"].as_str().unwrap().to_string());
                    // Queued, as the Hub answers a finalize (202).
                    (StatusCode::ACCEPTED, [("content-type", "application/json")],
                        serde_json::json!({"operation": "op", "state": "completed",
                            "status_url": format!("{p}ation")}).to_string())
                        .into_response()
                }
                _ => (
                    StatusCode::NOT_FOUND,
                    r#"{"error":{"code":"route.absent","message":"no such route","remedy":""}}"#,
                )
                    .into_response(),
            }
        }
    };
    let state = held.clone();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let router = axum::Router::new().fallback(answer).with_state(state);
            let listener = tokio::net::TcpListener::from_std(listener).unwrap();
            axum::serve(listener, router).await.unwrap()
        })
    });
    (format!("http://127.0.0.1:{port}"), held)
}

/// A one-tensor checkpoint in `store`: what a machine publishes.
pub fn checkpoint(store: &tensorfs::store2::Store, weights: &[u8]) -> tensorfs::format::ObjectRef {
    tensors(store, &[weights])
}

/// A checkpoint in `store` with one F32 tensor per entry of `weights` (16 MiB or more: its own
/// chunks).
pub fn tensors(store: &tensorfs::store2::Store, weights: &[&[u8]]) -> tensorfs::format::ObjectRef {
    use tensorfs::{format::Dtype, writer::{Data, Source}};
    let mut writer = store.writer(None, None).unwrap();
    for (at, weights) in weights.iter().enumerate() {
        assert!(!weights.is_empty() && weights.len().is_multiple_of(4));
        let name = if at == 0 { "weight".to_string() } else { format!("weight{at}") };
        let shape = [(weights.len() / 4) as u64];
        writer.put("model", &name, &Dtype::F32, &shape, Data::Plain(Source::Read(&mut &weights[..]))).unwrap();
    }
    writer.commit().unwrap()
}

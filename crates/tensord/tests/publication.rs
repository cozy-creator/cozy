//! A checkpoint put in a Hub repository's custody (`Catalog::publish`, on hub-client) against a
//! loopback Hub that answers as api/openapi.yaml spells it: only the objects the Hub lacks are
//! pushed, each under a just-in-time singleton grant that is renewed only when stale or
//! expired and kept alive while a long PUT moves; then verified and finalized. A stop before
//! finalizing commits nothing and the same operation resumes; a repository that holds the
//! checkpoint converges with no transfer.
#[path = "common/hub_oauth.rs"]
mod hub_oauth;

use axum::{
    body::Bytes,
    extract::State,
    http::{HeaderMap, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tensord::{
    hub::{Capability, Catalog, Op, Source},
    origin::Publication,
    stop::Stop,
};
use tensorfs_core::{err::Code, format::ObjectRef, store2::Store};

/// The patience a stopped publication is held to.
const SAMPLE_SECONDS: f64 = 1.0;

/// The most objects this Hub takes in one grant or verify request.
const SELECTION: usize = 3;
const REPO: &str = "/v1/models/acme/model";

#[derive(Default)]
struct Hub {
    declared: Vec<String>,
    uploaded: HashMap<String, Vec<u8>>,
    verified: Vec<String>,
    finalized: usize,
    holds: bool,
    /// Grant or verify requests refused for their selection's size.
    refused: usize,
    mode: &'static str,
    grants: HashMap<String, usize>,
    puts: HashMap<String, usize>,
    max_selection: usize,
    blocked: String,
    released_after_refill: bool,
    max_unverified_grants: usize,
    /// Ends a `held` PUT of the blocked object, unanswered.
    release: bool,
}

type Held = Arc<Mutex<Hub>>;

fn json(status: u16, value: Value) -> Response {
    (StatusCode::from_u16(status).unwrap(), [("content-type", "application/json")], value.to_string()).into_response()
}

fn refusal(status: u16, code: &str) -> Response {
    json(status, json!({"error": {"code": code, "message": code, "remedy": ""}}))
}

fn ids(body: &[u8], key: &str) -> Vec<String> {
    let body: Value = serde_json::from_slice(body).unwrap();
    body[key].as_array().unwrap().iter().map(|row| row.get("object_id").unwrap_or(row).as_str().unwrap().to_string()).collect()
}

async fn answer(State((held, base)): State<(Held, String)>, method: Method, uri: Uri, headers: HeaderMap, body: Bytes) -> Response {
    let path = uri.path().to_string();
    if path == "/v1/auth/oauth2/token" {
        return json(200, json!({"access_token": "token", "token_type": "DPoP", "expires_in": 3600}));
    }
    if let Some(hex) = path.strip_prefix("/upload/") {
        let id = format!("sha256:{hex}");
        let mode = {
            let mut hub = held.lock().unwrap();
            *hub.puts.entry(id.clone()).or_default() += 1;
            hub.mode
        };
        if mode == "held" && id == held.lock().unwrap().blocked && !held.lock().unwrap().release {
            // Its body arrived; its answer waits until the test lets it go, unrecorded.
            while !held.lock().unwrap().release {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            return refusal(500, "never_answered");
        }
        if mode == "rolling" && id == held.lock().unwrap().blocked {
            let start = Instant::now();
            loop {
                {
                    let mut hub = held.lock().unwrap();
                    if hub.uploaded.len() == hub.declared.len() - 2 && hub.grants.get(&id).copied().unwrap_or(0) >= 2 {
                        // Every later small object landed AND the long PUT's hold renewed,
                        // before this answer releases its upload slot.
                        hub.released_after_refill = true;
                        break;
                    }
                }
                if start.elapsed() > Duration::from_secs(5) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
        let generation = headers.get("x-grant-generation").and_then(|v| v.to_str().ok());
        if generation == Some("1") && matches!(mode, "expired403" | "landed" | "heartbeat") {
            tokio::time::sleep(Duration::from_millis(1200)).await;
            if mode == "landed" {
                held.lock().unwrap().uploaded.insert(id.clone(), body.to_vec());
            }
            if mode != "heartbeat" {
                return (StatusCode::FORBIDDEN, "expired grant").into_response();
            }
        }
        if mode == "denied" {
            return (StatusCode::FORBIDDEN, "permission denied").into_response();
        }
        held.lock().unwrap().uploaded.insert(id, body.to_vec());
        return StatusCode::OK.into_response();
    }
    let mut hub = held.lock().unwrap();
    let operation = path
        .strip_prefix(&format!("{REPO}/publications/"))
        .map(|rest| rest.split('/').next().unwrap().to_string())
        .unwrap_or_default();
    let view = |state: &str| json!({"operation": operation, "state": state, "status_url": format!("{REPO}/publications/{operation}/finalization")});
    let publication = |hub: &Hub| {
        let rows: Vec<Value> = hub.declared.iter().enumerate().map(|(i, id)| {
            let state = if i == 0 || hub.uploaded.contains_key(id) { "accepted" } else { "pending" };
            json!({"object_id": id, "length": 0, "state": state})
        }).collect();
        json!({"operation": operation, "state": "open", "objects": rows})
    };
    let posted = |end: &str| method == Method::POST && path.starts_with(&format!("{REPO}/publications/")) && path.ends_with(end);
    if method == Method::GET && path.starts_with(&format!("{REPO}/checkpoints/")) {
        return match hub.holds {
            true => json(200, json!({"held": true})),
            false => refusal(404, "model.checkpoint_not_found"),
        };
    }
    if method == Method::PUT && path.starts_with(&format!("{REPO}/publications/")) {
        hub.declared = ids(&body, "objects");
        if matches!(hub.mode, "rolling" | "held") {
            let blocked = hub.declared.iter().position(|id| *id == hub.blocked).unwrap();
            hub.declared.swap(1, blocked);
        }
        // The Hub already holds the first object: it is never pushed.
        return json(200, json!({"created": true, "publication": publication(&hub)}));
    }
    // As Tensorhub: a bounded selection per grant or verify request (128 there).
    if (posted("/grants") || posted("/verify")) && !(1..=SELECTION).contains(&ids(&body, "object_ids").len()) {
        hub.refused += 1;
        return refusal(422, "publication.object_selection_invalid");
    }
    if posted("/grants") {
        let selected = ids(&body, "object_ids");
        hub.max_selection = hub.max_selection.max(selected.len());
        let (mut grants, mut holds) = (vec![], vec![]);
        for id in selected {
            let (mode, blocked) = (hub.mode, id == hub.blocked);
            let generation = {
                let generation = hub.grants.entry(id.clone()).or_default();
                *generation += 1;
                *generation
            };
            if hub.uploaded.contains_key(&id) {
                holds.push(json!({"transfer_id": "t", "object_id": id, "length": 0, "state": "accepted"}));
                continue;
            }
            let lifetime = match mode {
                "always-stale" => 0,
                "stale" if generation == 1 => 0,
                "rolling" if blocked => 1,
                "expired403" | "landed" | "heartbeat" if generation == 1 => 1,
                _ => 600,
            };
            grants.push(json!({"object_id": id, "length": 0, "url": format!("{base}/upload/{}", &id[7..]),
                "expires_at_unix": 1_000_000_000 + lifetime,
                "required_headers": {"if-none-match": "*", "x-grant-generation": generation.to_string()}}));
        }
        let unverified = hub.grants.keys().filter(|id| !hub.verified.contains(id)).count();
        hub.max_unverified_grants = hub.max_unverified_grants.max(unverified);
        let server_time = if hub.mode == "legacy" { 0 } else { 1_000_000_000 };
        return json(200, json!({"grants": grants, "held": holds, "server_time_unix": server_time}));
    }
    if posted("/verify") {
        hub.verified.extend(ids(&body, "object_ids"));
        return json(200, json!({"publication": publication(&hub)}));
    }
    if posted("/finalize") {
        hub.finalized += 1;
        return json(202, view("running"));
    }
    if method == Method::GET && path.ends_with("/finalization") {
        return json(200, view("completed"));
    }
    refusal(404, "route.absent")
}

/// The Hub, and a source whose capability publishes to acme/model there.
fn hub() -> (Source, Held) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let origin = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let held: Held = Default::default();
    let state = (held.clone(), origin.clone());
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread().worker_threads(4).enable_all().build().unwrap();
        runtime.block_on(async move {
            let router = axum::Router::new().fallback(answer).with_state(state);
            let listener = tokio::net::TcpListener::from_std(listener).unwrap();
            axum::serve(listener, router).await.unwrap()
        })
    });
    let pem = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).unwrap().serialize_pem();
    let leaf = hub_client::P256Key::from_pem(&pem).unwrap();
    let part = |value: Value| URL_SAFE_NO_PAD.encode(value.to_string());
    let exp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() + 3600;
    let jws = format!("{}.{}.sig", part(json!({"alg": "EdDSA"})), part(json!({"aud": origin, "cnf": {"jkt": leaf.thumbprint()},
        "exp": exp, "authorization_details": [{"type": "tensorhub_model_upload", "model": "acme/model"}]})));
    let capability = Capability::new(&jws, &format!("{origin}/v1/auth/oauth2/token"), &leaf.private_jwk()).unwrap();
    (Source::new(&origin, None, vec![], Some(capability)).unwrap(), held)
}

fn catalog(source: &Source) -> Catalog {
    Catalog::for_op(source, Op::Publish { model: "acme/model" }).unwrap()
}

/// Tensor `at`'s distinct bytes.
fn weight(at: usize, size: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; size];
    bytes[..4].copy_from_slice(&(at as u32 + 1).to_le_bytes());
    bytes
}

/// A store holding a tree whose closure has one object per entry of `sizes` (a repo file each),
/// besides its header and chunk.
fn store(name: &str, sizes: &[usize]) -> (std::path::PathBuf, Store, ObjectRef) {
    use tensorfs_core::{format::{Dtype, Tree}, writer::{Data, Source}};
    let root = std::env::temp_dir().join(format!("td-publication-{name}-{}", uuid::Uuid::new_v4()));
    let store = Store::open(&root).unwrap();
    let mut writer = store.writer(None, None).unwrap();
    writer.get("model").put("weight", &Dtype::F32, &[1], Data::Plain(Source::Read(&mut &[0u8; 4][..]))).unwrap();
    for (at, size) in sizes.iter().enumerate() {
        writer.repo_file(&format!("files/{at}"), &weight(at, *size)).unwrap();
    }
    let header = writer.commit().unwrap();
    let mut tree = Tree { files: writer.repo_files().clone() };
    tree.set_variant("", header);
    let tree = store.put_tree(&tree).unwrap();
    (root, store, tree)
}

fn publication<'a>(store: &'a Store, tree: &'a ObjectRef, operation: &'a str) -> Publication<'a> {
    Publication {
        store,
        destination: "acme/model",
        tree,
        operation,
        progress: &|_, _| {},
        streams: 4,
        stop: None,
        commit: &|| true,
    }
}

#[test]
fn publish_pushes_what_the_hub_lacks_and_follows_its_finalization() {
    let (root, store, tree) = store("basic", &[4096; 6]);
    let (source, state) = hub();
    let landed = Mutex::new(Vec::new());
    let progress = |done: u64, total: u64| landed.lock().unwrap().push((done, total));
    let catalog = catalog(&source);
    let published = catalog.publish(&Publication { destination: "Acme/Model", progress: &progress, ..publication(&store, &tree, "op-1") }).unwrap();
    assert_eq!(published.checkpoint, tree.id());
    assert!(!published.converged);
    let held = state.lock().unwrap();
    let pushed: Vec<_> = held.declared[1..].to_vec();
    let mut uploaded: Vec<_> = held.uploaded.keys().cloned().collect();
    uploaded.sort();
    let mut sorted = pushed.clone();
    sorted.sort();
    assert_eq!(uploaded, sorted);
    for (id, body) in &held.uploaded {
        assert_eq!(&ObjectRef::of(body).id(), id);
    }
    let mut verified = held.verified.clone();
    verified.sort();
    assert_eq!(verified, sorted);
    assert_eq!(held.finalized, 1);
    let (done, total) = *landed.lock().unwrap().last().unwrap();
    assert_eq!(done, total);
    drop(held);

    state.lock().unwrap().holds = true;
    let again = catalog.publish(&publication(&store, &tree, "op-2")).unwrap();
    assert!(again.converged);
    assert_eq!(state.lock().unwrap().finalized, 1);
    let _ = std::fs::remove_dir_all(root);
}

/// A closure larger than one request's selection (an SDXL LoRA's) is asked for and verified in
/// requests the Hub accepts: just-in-time singleton grants fit a Hub whose selection limit is
/// below the worker count.
#[test]
fn publish_asks_for_a_large_closure_in_requests_the_hub_accepts() {
    let (root, store, tree) = store("large", &[4096; 300]);
    let (source, state) = hub();
    let published = catalog(&source).publish(&publication(&store, &tree, "op-large")).unwrap();
    assert_eq!(published.checkpoint, tree.id());
    let held = state.lock().unwrap();
    assert_eq!(held.uploaded.len(), held.declared.len() - 1);
    assert_eq!(held.verified.len(), held.declared.len() - 1);
    assert_eq!(held.finalized, 1);
    assert_eq!(held.refused, 0);
    assert!(held.max_selection <= SELECTION);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn publish_renews_only_expired_authority_and_keeps_long_puts_live() {
    for mode in ["stale", "expired403", "landed", "heartbeat", "denied", "always-stale", "legacy"] {
        let (root, store, tree) = store(mode, &[512]);
        let (source, state) = hub();
        state.lock().unwrap().mode = mode;
        let progress = Mutex::new((0, 0));
        let published = catalog(&source).publish(&Publication {
            progress: &|done, total| *progress.lock().unwrap() = (done, total),
            streams: 2,
            ..publication(&store, &tree, mode)
        });
        let state = state.lock().unwrap();
        assert!(state.max_selection <= 2);
        if mode == "denied" {
            assert_eq!(published.unwrap_err().code, Code::TRANSFER_FAILED);
            assert!(state.grants.values().all(|n| *n == 1));
            assert!(state.puts.values().all(|n| *n == 1));
            assert_eq!(state.finalized, 0);
        } else if mode == "always-stale" {
            assert_eq!(published.unwrap_err().code, Code::HUB_REFUSED);
            assert!(state.grants.values().all(|n| *n <= 3));
            assert!(state.puts.is_empty());
        } else {
            assert_eq!(published.unwrap().checkpoint, tree.id(), "{mode}");
            if mode == "legacy" {
                assert!(state.grants.values().all(|n| *n == 1));
            } else {
                assert!(state.grants.values().all(|n| *n >= 2), "{mode}");
            }
            assert_eq!(state.uploaded.len(), state.declared.len() - 1);
            if mode != "expired403" {
                assert!(state.puts.values().all(|n| *n == 1), "{mode}");
            }
            let (done, total) = *progress.lock().unwrap();
            assert_eq!(done, total, "held-on-renewal bytes count exactly once");
        }
        drop(state);
        let _ = std::fs::remove_dir_all(root);
    }
}

#[test]
fn publish_refills_slots_while_a_large_put_waits_and_renews() {
    let mut sizes = vec![512; 24];
    sizes[0] = 1 << 20;
    let (root, store, tree) = store("rolling", &sizes);
    let blocked = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&weight(0, 1 << 20)));
    let (source, state) = hub();
    {
        let mut state = state.lock().unwrap();
        state.mode = "rolling";
        state.blocked = blocked.clone();
    }
    let published = catalog(&source).publish(&Publication { streams: 8, ..publication(&store, &tree, "rolling") }).unwrap();
    let state = state.lock().unwrap();
    assert!(state.released_after_refill, "later objects must finish before the slow PUT is released");
    assert!(state.max_unverified_grants <= 8, "only dispatchable objects obtain grants");
    assert!(state.grants[&blocked] >= 2, "a waiting PUT retains its publication hold");
    assert_eq!(state.puts[&blocked], 1, "heartbeat does not restart a moving PUT");
    assert_eq!(state.verified.len(), state.declared.len() - 1);
    assert_eq!(published.checkpoint, tree.id());
    drop(state);
    let _ = std::fs::remove_dir_all(root);
}

/// A publication its caller stops mid-push ends within a sampling tick of the stop, finalizes
/// nothing, and the same operation resumes it: only the object whose PUT was cut short is
/// pushed again.
#[test]
fn a_stopped_publication_finalizes_nothing_and_resumes() {
    let mut sizes = vec![512; 8];
    sizes[0] = 1 << 20;
    let (root, store, tree) = store("stopped", &sizes);
    let blocked = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&weight(0, 1 << 20)));
    let (source, state) = hub();
    {
        let mut state = state.lock().unwrap();
        state.mode = "held";
        state.blocked = blocked.clone();
    }
    let catalog = catalog(&source);
    let stop = Stop::default();
    let (stopped, took) = std::thread::scope(|scope| {
        let running = scope.spawn(|| catalog.publish(&Publication { stop: Some(&stop), ..publication(&store, &tree, "stopped") }));
        let started = Instant::now();
        while !{
            let state = state.lock().unwrap();
            state.puts.contains_key(&blocked) && state.uploaded.len() + 2 == state.declared.len()
        } {
            assert!(started.elapsed() < Duration::from_secs(30), "the push never reached the held PUT");
            std::thread::sleep(Duration::from_millis(10));
        }
        let asked = Instant::now();
        stop.cancel();
        (running.join().unwrap(), asked.elapsed())
    });
    assert_eq!(stopped.unwrap_err().code, Code::TRANSFER_FAILED);
    assert!(took < Duration::from_secs_f64(2.0 * SAMPLE_SECONDS), "stopped after {took:?}");
    {
        let mut state = state.lock().unwrap();
        assert_eq!(state.finalized, 0, "a stopped publication finalized");
        assert!(!state.uploaded.contains_key(&blocked));
        state.release = true;
    }
    let resumed = catalog.publish(&publication(&store, &tree, "stopped")).unwrap();
    assert_eq!(resumed.checkpoint, tree.id());
    let state = state.lock().unwrap();
    assert_eq!(state.finalized, 1);
    assert_eq!(state.puts[&blocked], 2);
    assert!(state.puts.iter().all(|(id, n)| *id == blocked || *n == 1), "resumed pushes repeat landed objects: {:?}", state.puts);
    assert_eq!(state.uploaded.len() + 1, state.declared.len());
    drop(state);
    let _ = std::fs::remove_dir_all(root);
}

/// Declined at its commit point, a publication with every object held finalizes nothing; asked
/// again it pushes nothing and finalizes once.
#[test]
fn a_publication_declined_at_its_commit_point_finalizes_nothing() {
    let (root, store, tree) = store("declined", &[512; 4]);
    let (source, state) = hub();
    let catalog = catalog(&source);
    let declined = catalog.publish(&Publication { commit: &|| false, ..publication(&store, &tree, "declined") });
    assert_eq!(declined.unwrap_err().code, Code::TRANSFER_FAILED);
    assert_eq!(state.lock().unwrap().finalized, 0);
    catalog.publish(&publication(&store, &tree, "declined")).unwrap();
    let state = state.lock().unwrap();
    assert_eq!(state.finalized, 1);
    assert!(state.puts.values().all(|n| *n == 1));
    drop(state);
    let _ = std::fs::remove_dir_all(root);
}

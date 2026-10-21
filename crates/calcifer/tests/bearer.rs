//! Calcifer gives the machines it sent its owner's Hub bearer each one its session mints after, at
//! the session's own refresh point: a real TensorD (`COZY_TEST_TENSORD`) keeps reading the owner's
//! checkpoints mid-run after the bearer its run was sent with is revoked. The Hub is a loopback
//! stand-in serving the distribution layout TensorFS writes; no Runtime is involved.
mod common;

use std::{
    collections::HashSet,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::State,
    http::{HeaderMap, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use calcifer::{Calcifer, machines::Machines};
use common::{Tensord, until};
use platform::{durable, paths::Home};
use protocol::{
    calcifer::{Event, Kind, On, SubmitRun},
    config, machine as v1,
};
use tensorfs::{
    fetch::{Client, Endpoint, SourcePolicy, upload_tree},
    format::{Dtype, ObjectRef, Tree},
    store::Store,
    writer::{Data, Source},
};

/// The stand-in Hub: its account and environment, and its owner's unreleased checkpoints, each
/// object answered after `pace` and only under an admitted bearer. It records each object read
/// with the bearer it carried.
struct Hub {
    origin: String,
    layout: PathBuf,
    admitted: Mutex<HashSet<String>>,
    heard: Mutex<Vec<(String, String)>>,
    pace: Duration,
}

async fn answer(
    State(hub): State<Arc<Hub>>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
) -> Response {
    let json = |value: serde_json::Value| {
        (StatusCode::OK, serde_json::to_vec(&value).unwrap()).into_response()
    };
    match uri.path() {
        "/v1/accounts/current" => {
            return json(serde_json::json!({"name": "owner", "user_id": "u1"}));
        }
        "/v1/execution-environment" => {
            let origin = &hub.origin;
            return json(
                serde_json::json!({"environment": {"TENSORHUB_ORIGIN": origin, "TENSORHUB_PUBLIC_ORIGIN": origin}}),
            );
        }
        path if path.starts_with("/blobs/") => {}
        _ => return StatusCode::NOT_FOUND.into_response(),
    }
    let presented = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    let bearer = presented.unwrap_or_default().to_owned();
    hub.heard
        .lock()
        .unwrap()
        .push((format!("{method} {}", uri.path()), bearer.clone()));
    if !hub.admitted.lock().unwrap().contains(&bearer) {
        return (
            StatusCode::UNAUTHORIZED,
            r#"{"error":{"code":"unauthorized","message":"no"}}"#,
        )
            .into_response();
    }
    let Ok(bytes) = std::fs::read(hub.layout.join(&uri.path()[1..])) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    if method == Method::GET {
        tokio::time::sleep(hub.pace).await;
    }
    bytes.into_response()
}

/// `repository`'s unreleased checkpoint of one F32 tensor, uploaded into `layout`: its tree.
async fn checkpoint(
    store: &Store,
    layout: &std::path::Path,
    repository: &str,
    weight: u8,
) -> ObjectRef {
    let mut writer = store.writer(None, None).unwrap();
    let weights = [weight; 8192];
    writer
        .get("model")
        .put(
            "weight",
            &Dtype::F32,
            &[2048],
            Data::Plain(Source::Read(&mut &weights[..])),
        )
        .unwrap();
    let mut tree = Tree::default();
    tree.set_variant("bf16", writer.commit().unwrap());
    let tree = store.put_tree(&tree).unwrap();
    let client = Client::new(SourcePolicy::default());
    let endpoint = Endpoint::new(&client, &format!("file://{}", layout.display()), None).unwrap();
    upload_tree(&endpoint, repository, &tree, store, tensorfs::fetch::MAIN)
        .await
        .unwrap();
    tree
}

/// This computer's sign-in at `hub` (a synthetic device key) and its kept `token`, live until
/// `expires` (Unix seconds): the session uses it with no network until its refresh point.
fn keep(home: &Home, hub: &config::Hub, token: &str, expires: u64) {
    durable::private_dir(&home.keys()).unwrap();
    let key = serde_json::json!({"version": 1, "hub": hub.origin.as_str(), "email": "owner@example.test",
        "device_key_id": "bearer-test-device", "private_key": URL_SAFE_NO_PAD.encode([17u8; 32])});
    durable::replace(
        &hub.device_key(home),
        &serde_json::to_vec(&key).unwrap(),
        0o600,
    )
    .unwrap();
    let bearer = hub_client::Bearer {
        token: token.into(),
        expires,
        device_key_id: "bearer-test-device".into(),
    };
    durable::replace(
        &hub.bearer(home),
        &serde_json::to_vec(&bearer).unwrap(),
        0o600,
    )
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_refreshed_bearer_reaches_the_machine_mid_run() {
    let temp = tempfile::tempdir().unwrap();
    let store = Store::open(temp.path().join("source")).unwrap();
    let layout = temp.path().join("layout");
    let mut trees = vec![];
    for (at, repository) in ["owner/first", "owner/second", "owner/third"]
        .into_iter()
        .enumerate()
    {
        trees.push((
            repository,
            checkpoint(&store, &layout, repository, at as u8).await,
        ));
    }
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let hub = Arc::new(Hub {
        origin: origin.clone(),
        layout,
        admitted: Mutex::new(["t1".to_owned()].into()),
        heard: Mutex::default(),
        // Each run reads its models one after another, each object answered after 3 s: the run
        // outlasts the refresh point of the bearer it is sent with.
        pace: Duration::from_secs(3),
    });
    let router = axum::Router::new().fallback(answer).with_state(hub.clone());
    let serving = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

    let home = Home::at(temp.path());
    let configured = config::Hub {
        name: None,
        origin: config::Origin::parse(&origin).unwrap(),
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    // The session mints its next bearer 30 s before this one expires: 6 s from now.
    keep(&home, &configured, "t1", now + 36);
    let machine = Tensord::start();
    let calcifer =
        Calcifer::open(&temp.path().join("calcifer"), &temp.path().join("outputs")).unwrap();
    calcifer
        .put_machine(&machine.row("m", &common::key_file(temp.path())))
        .unwrap();
    let config = config::Config {
        hubs: [("dev".to_owned(), origin.clone())].into(),
        ..Default::default()
    };
    let machines = Machines::start(calcifer.clone(), home.clone(), &config);
    let mut events = calcifer.subscribe();
    let spec = v1::RunSpec {
        kind: v1::RunKind::Warm,
        models: trees
            .iter()
            .map(|(repository, tree)| v1::ModelChoice {
                parameter: repository.to_string(),
                repository: repository.to_string(),
                manifest: tree.id(),
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    };
    let submit = SubmitRun {
        agreed: None,
        on: On::Machine("m".into()),
        hub: origin.clone(),
        kind: Kind::Warm,
        target: "owner/models".into(),
        spec,
        inputs: vec![],
        out: None,
        warnings: vec![],
    };
    let id = calcifer.submit("k-bearer", submit).await.unwrap().run.id;
    let presented = |token: &str| {
        hub.heard
            .lock()
            .unwrap()
            .iter()
            .any(|(_, bearer)| bearer == token)
    };
    // Test harness bound on a send or refresh that never comes; the product has no such limit.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    let wait = |token: &'static str| async move {
        while !presented(token) {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the machine never presented {token}"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    };
    wait("t1").await;
    // Another process of this computer refreshed the session since (as `cozy` does); the Hub
    // honours both until the machine presents the new one, then revokes the first.
    keep(&home, &configured, "t2", now + 3600);
    hub.admitted.lock().unwrap().insert("t2".into());
    wait("t2").await;
    hub.admitted.lock().unwrap().remove("t1");
    let revoked = hub.heard.lock().unwrap().len();
    let done = until(&mut events, &id, |event| {
        matches!(
            event,
            Event::Completed { .. } | Event::Failed { .. } | Event::Canceled { .. }
        )
    })
    .await;
    assert!(matches!(done, Event::Completed { .. }), "{done:?}");
    let heard = hub.heard.lock().unwrap().clone();
    let after = &heard[revoked..];
    assert!(
        !after.is_empty() && after.iter().all(|(_, bearer)| bearer == "t2"),
        "{heard:?}"
    );
    assert!(
        heard[..revoked].iter().any(|(_, bearer)| bearer == "t1"),
        "{heard:?}"
    );
    machines.close();
    calcifer.close();
    serving.abort();
}

//! The client's transport against a listener built as a machine's is: one pinned TLS leaf with
//! ALPN `h2, http/1.1`, hyper-util's auto builder and an axum router. Control calls ride HTTP/2,
//! bulk calls HTTP/1.1; SSE, held ranges, ranged reads and refusals parse as the API says.
use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
};

use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::Request,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response, sse},
    routing::{get, post, put},
};
use cozy_proto::{
    error::{Envelope, Error as Refusal},
    machine::{
        self, Event, HELD, OutputTarget, Product, RunEvent, RunRequest, RunState, StatusFrame,
        client::{Auth, Error, Machine, presented_leaf},
        path,
    },
};
use futures_util::{StreamExt as _, stream};
use hyper_util::rt::{TokioExecutor, TokioIo};
use sha2::{Digest as _, Sha256};

/// What the listener saw: each request's method, path, HTTP version and headers it names.
type Seen = Arc<Mutex<Vec<String>>>;

struct Listener {
    address: String,
    leaf: Vec<u8>,
    seen: Seen,
}

async fn listen(router: Router) -> Listener {
    let rcgen::CertifiedKey { cert, signing_key } =
        rcgen::generate_simple_self_signed(vec!["cozy-worker".into()]).unwrap();
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let mut config = rustls::ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(
            vec![cert.der().clone()],
            rustls::pki_types::PrivateKeyDer::Pkcs8(signing_key.serialize_der().into()),
        )
        .unwrap();
    config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(config));
    let tcp = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = tcp.local_addr().unwrap().to_string();
    let seen = Seen::default();
    let router = router.layer(axum::middleware::from_fn_with_state(seen.clone(), note));
    tokio::spawn(async move {
        loop {
            let (stream, _) = tcp.accept().await.unwrap();
            let (acceptor, router) = (acceptor.clone(), router.clone());
            tokio::spawn(async move {
                let Ok(tls) = acceptor.accept(stream).await else {
                    return;
                };
                let service = hyper_util::service::TowerToHyperService::new(router);
                let _ = hyper_util::server::conn::auto::Builder::new(TokioExecutor::new())
                    .serve_connection(TokioIo::new(tls), service)
                    .await;
            });
        }
    });
    Listener {
        address,
        leaf: cert.der().to_vec(),
        seen,
    }
}

async fn note(
    axum::extract::State(seen): axum::extract::State<Seen>,
    request: Request,
    next: axum::middleware::Next,
) -> Response {
    let headers = request.headers();
    let named = |name: &str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("-")
    };
    let line = format!(
        "{} {} {:?} auth={} range={} content-range={} if-match={} last-event-id={}",
        request.method(),
        request.uri().path_and_query().map_or("", |p| p.as_str()),
        request.version(),
        named("authorization").starts_with("Cozy-Cap "),
        named("range"),
        named("content-range"),
        named("if-match"),
        named("last-event-id"),
    );
    seen.lock().unwrap().push(line);
    next.run(request).await
}

fn refusal(status: StatusCode, error: Refusal) -> Response {
    (status, Json(Envelope { error })).into_response()
}

fn router() -> Router {
    let status = get(|headers: HeaderMap| async move {
        let frame = |worker: &str| StatusFrame {
            worker_id: worker.into(),
            receipt: b"sealed".to_vec(),
            ..Default::default()
        };
        let wants_sse = headers
            .get(header::ACCEPT)
            .is_some_and(|v| v.as_bytes() == b"text/event-stream");
        if !wants_sse {
            return Json(frame("w")).into_response();
        }
        let events = ["w1", "w2"].map(|w| {
            Ok::<_, Infallible>(
                sse::Event::default()
                    .event("status")
                    .json_data(frame(w))
                    .unwrap(),
            )
        });
        let future = Ok(sse::Event::default()
            .event("from-a-newer-machine")
            .data("{}"));
        sse::Sse::new(stream::iter([events[0].clone(), future, events[1].clone()])).into_response()
    });
    let events = get(|| async {
        let event = |sequence, event: Event| {
            let name = event.name();
            let data = RunEvent {
                sequence,
                at_ms: 1,
                event,
            };
            let event = sse::Event::default().event(name).json_data(data).unwrap();
            Ok::<_, Infallible>(match sequence {
                0 => event,
                n => event.id(n.to_string()),
            })
        };
        sse::Sse::new(stream::iter([
            event(0, Event::State(RunState::default())),
            Ok(sse::Event::default().event("future").data("{\"x\":1}")),
            event(
                5,
                Event::Product(Product {
                    output: "image".into(),
                    rev: 2,
                    ..Default::default()
                }),
            ),
        ]))
    });
    let output = get(|| async {
        Response::builder()
            .status(StatusCode::PARTIAL_CONTENT)
            .header(header::ETAG, "\"3\"")
            .header(header::CONTENT_RANGE, "bytes 4-7/10")
            .header(header::CONTENT_TYPE, "image/png")
            .body(Body::from("4567"))
            .unwrap()
    });
    let objects = put(|body: Bytes| async move {
        assert_eq!(&body[..], b"0123");
        (StatusCode::NO_CONTENT, [(HELD, "0-3,8-9")])
    })
    .head(|| async { (StatusCode::OK, [(HELD, "")]) });
    Router::new()
        .route(path::STATUS, status)
        .route(path::EVENTS, events)
        .route(path::OUTPUT, output)
        .route(path::OBJECT, objects)
        .route(
            path::RUNS,
            post(|Json(run): Json<RunRequest>| async move {
                let message = format!("{} names another spec", run.id);
                refusal(
                    StatusCode::CONFLICT,
                    Refusal::conflict("run_id_conflict", message),
                )
            }),
        )
        .route(
            path::CANCEL,
            post(|| async { (StatusCode::BAD_GATEWAY, "<html>bad gateway</html>") }),
        )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_client_pins_the_leaf_and_speaks_every_shape_of_the_api() {
    let listener = listen(router()).await;
    let pin: [u8; 32] = Sha256::digest(&listener.leaf).into();
    assert_eq!(
        presented_leaf(&listener.address).await.unwrap(),
        listener.leaf
    );

    // Another leaf than the pinned one is refused before anything is sent.
    let impostor = Machine::new(&listener.address, [7; 32], Auth::None).unwrap();
    assert!(matches!(impostor.status().await, Err(Error::Transport(_))));

    let anonymous = Machine::new(&listener.address, pin, Auth::None).unwrap();
    let frame = anonymous.status().await.unwrap();
    assert_eq!(
        (frame.worker_id.as_str(), &frame.receipt[..]),
        ("w", &b"sealed"[..])
    );

    let key = ed25519_dalek::SigningKey::from_bytes(&[9; 32]);
    let machine = anonymous.with_auth(Auth::Key {
        key: Arc::new(key),
        worker: "w".into(),
    });
    let watched: Vec<_> = machine.watch().await.unwrap().collect().await;
    let workers: Vec<_> = watched.into_iter().map(|f| f.unwrap().worker_id).collect();
    assert_eq!(workers, ["w1", "w2"], "an unknown event is skipped");

    let events: Vec<_> = machine.events("run 1", 4).await.unwrap().collect().await;
    let events: Vec<_> = events.into_iter().map(Result::unwrap).collect();
    assert_eq!(events.len(), 2, "{events:?}");
    assert!(matches!(&events[1].event, Event::Product(p) if p.rev == 2 && events[1].sequence == 5));

    let refused = machine
        .submit(&RunRequest {
            id: "r".into(),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(
        (refused.status(), refused.code()),
        (Some(409), Some("run_id_conflict"))
    );
    assert!(matches!(machine.cancel("r").await, Err(Error::Answer(_))));

    assert_eq!(
        machine.held("sha256:ab").await.unwrap(),
        machine::Ranges::new()
    );
    let bulk = machine.bulk().unwrap();
    let held = bulk.put("sha256:ab", (0, 4, 10), "0123").await.unwrap();
    assert_eq!(held, [(0, 4), (8, 10)]);
    let target = OutputTarget {
        run: "run 1".into(),
        output: "image".into(),
        index: 2,
        member: "a/b c.png".into(),
    };
    let output = bulk.get(&target, (4, 8), 3).await.unwrap();
    assert_eq!((output.rev, output.length, output.range), (3, 10, (4, 8)));
    assert_eq!(output.media_type, "image/png");
    let body: Vec<_> = output
        .body
        .map(|piece| piece.unwrap().to_vec())
        .collect()
        .await;
    assert_eq!(body.concat(), b"4567");

    let seen = listener.seen.lock().unwrap().clone();
    let line = |prefix: &str| {
        seen.iter()
            .find(|l| l.starts_with(prefix))
            .unwrap_or_else(|| panic!("no {prefix} in {seen:#?}"))
            .clone()
    };
    assert!(
        line("GET /v1/status").contains("HTTP/2.0 auth=false"),
        "{seen:#?}"
    );
    assert!(
        line("GET /v1/runs/run%201/events").contains("HTTP/2.0 auth=true"),
        "{seen:#?}"
    );
    assert!(
        line("GET /v1/runs/run%201/events").contains("last-event-id=4"),
        "{seen:#?}"
    );
    assert!(line("HEAD /v1/objects/sha256:ab").contains("HTTP/2.0 auth=true"));
    let put = line("PUT /v1/objects/sha256:ab");
    assert!(
        put.contains("HTTP/1.1 auth=true") && put.contains("content-range=bytes 0-3/10"),
        "{put}"
    );
    let get = line("GET /v1/runs/run%201/outputs/image?index=2&member=a%2Fb+c.png");
    assert!(
        get.contains("HTTP/1.1")
            && get.contains("range=bytes=4-7 ")
            && get.contains("if-match=\"3\""),
        "{get}"
    );
}

//! hub-client against a real Tensorhub (Tensorhub's dev-build `cozy_ref` job starts one):
//! device-key enrollment and login, the account, model info and search, a rental quote, a
//! closure read, a run capability traded for a DPoP-bound token over real TLS, and (with the
//! job's docker provider) a rental that boots to ready and ends.
//!
//! TENSORHUB_URL is where the hub listens (plain HTTP on loopback), TENSORHUB_ORIGIN its public
//! https origin (SSL_CERT_FILE trusts its CA), TENSORHUB_LOG the hub's log, where the development
//! mail sender writes codes.
use hub_client::{
    Client, Credential, DeviceKey, Error, P256Key, RunCapability, RunToken, UserSession, testing,
    types,
};
use serde_json::json;
use sha2::{Digest as _, Sha256};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

struct Hub {
    url: String,
    origin: String,
    log: String,
}

fn hub() -> Option<Hub> {
    let var = |name| std::env::var(name).ok().filter(|v: &String| !v.is_empty());
    Some(Hub {
        url: var("TENSORHUB_URL")?,
        origin: var("TENSORHUB_ORIGIN")?,
        log: var("TENSORHUB_LOG")?,
    })
}

fn refused<T: std::fmt::Debug>(answer: Result<T, Error>) -> (u16, String) {
    match answer {
        Err(Error::Hub { status, code, .. }) => (status, code),
        other => panic!("expected a typed refusal, got {other:?}"),
    }
}

#[tokio::test]
async fn a_user_and_a_machine_at_a_real_hub() {
    let Some(hub) = hub() else {
        eprintln!("skipped: TENSORHUB_URL, TENSORHUB_ORIGIN and TENSORHUB_LOG name no hub");
        return;
    };
    let nonce = format!("{:x}", Sha256::digest(format!("{:?}", Instant::now())))[..10].to_string();
    let http = reqwest::Client::new();
    let anonymous = Client::connect(&hub.url, http.clone(), Credential::Anonymous);

    // Enrollment by emailed code; then a fresh session logs in from the key file alone.
    let email = format!("live-{nonce}@example.test");
    let (key, _) = testing::sign_up(&anonymous, &email, Path::new(&hub.log))
        .await
        .unwrap();
    let dir = std::env::temp_dir().join(format!("hub-client-live-{nonce}"));
    key.save(&dir.join("key.json")).unwrap();
    let session = Arc::new(UserSession::new(
        DeviceKey::load(&dir.join("key.json")).unwrap(),
        Some(dir.join("session.json")),
    ));
    let user = Client::connect(&hub.url, http.clone(), Credential::User(session.clone()));

    // The account.
    assert_eq!(
        refused(user.get_current_account().await.map_err(Error::from)),
        (409, "account.name_required".into())
    );
    let account = format!("live{nonce}");
    user.register_account(&account, &types::Empty(Default::default()))
        .await
        .unwrap();
    let me = user.get_current_account().await.unwrap().into_inner();
    assert_eq!(me.name, account);
    assert!(
        dir.join("session.json").exists(),
        "the Bearer is kept beside the key"
    );

    // A model, created by opening its first publication; its card and the searches.
    let object = |bytes: &[u8]| types::DeclaredObject {
        object_id: format!("sha256:{:x}", Sha256::digest(bytes)),
        length: bytes.len() as i64,
    };
    let declared = types::PublicationDeclaration {
        objects: vec![object(b"live")],
    };
    let opened = user
        .open_model_publication(&account, "model", "op-user", &declared)
        .await
        .unwrap()
        .into_inner();
    assert_eq!(opened.publication.objects.len(), 1);
    let card = anonymous
        .get_model(&account, "model")
        .await
        .unwrap()
        .into_inner();
    assert_eq!(
        (card.model.org.as_str(), card.releases.len()),
        (account.as_str(), 0)
    );
    anonymous
        .search_models(None, None, Some("model"))
        .await
        .unwrap();
    anonymous.search_packages(None, Some("sdxl")).await.unwrap();

    // The catalog and a quote: nothing is bought.
    let skus = anonymous.list_rental_skus(None).await.unwrap().into_inner();
    let sku = skus
        .first()
        .map(|sku| sku.name.clone())
        .unwrap_or_else(|| "rtx4090".into());
    let ask = types::RentalRequest {
        accelerator_count: None,
        container_disk_gb: None,
        creator_public_key: hub_client::P256Key::generate().thumbprint(),
        development: None,
        excluded_provider_machines: vec![],
        image: None,
        media_token_sha256: format!("{:x}", Sha256::digest(b"media")),
        name: format!("live-{nonce}"),
        planned_source_bytes: None,
        provider: None,
        serving_models: vec![],
        sku,
    };
    match user.quote_rental(&ask).await.map_err(Error::from) {
        Ok(_) | Err(Error::Hub { .. }) => {}
        Err(other) => panic!("a quote neither priced nor refused: {other}"),
    }

    // Placement for bare `--on new`: this account publishes no package, and the refusal is typed.
    let placement = types::RentalPlacementRequest {
        package: format!("{account}/absent"),
        release: None,
        function: "generate".into(),
        bits: None,
        gpus: None,
    };
    let (status, _) = refused(
        anonymous
            .place_rental(&placement)
            .await
            .map_err(Error::from),
    );
    assert_eq!(status, 404);

    // A closure read: the model has no release yet, and the refusal is typed.
    let closure = types::TensorfsClosureRequest {
        after: None,
        lane: Some("main".into()),
        lanes: vec![],
        limit: None,
        ref_: format!("{account}/model"),
    };
    assert_eq!(
        refused(
            anonymous
                .resolve_tensorfs_closure(&closure)
                .await
                .map_err(Error::from)
        )
        .0,
        404
    );

    // A run capability: signed offline by the device key, traded by the machine's leaf for a
    // DPoP-bound token over TLS, and good for exactly what it names.
    let endpoint = user.token_endpoint().await.unwrap();
    assert!(endpoint.starts_with(&hub.origin), "{endpoint}");
    let leaf = Arc::new(P256Key::generate());
    let capability = session.key().sign_run_capability(&RunCapability {
        user_id: me.user_id.clone(),
        hub: hub.origin.clone(),
        machine: leaf.thumbprint(),
        run: format!("run-{nonce}"),
        operations: vec![
            json!({"type": "tensorhub_model_publish", "model": format!("{account}/model")}),
        ],
        lifetime: Duration::from_secs(300),
    });
    let run = Arc::new(RunToken::new(&capability, &endpoint, leaf).unwrap());
    let machine = Client::connect(
        &hub.origin,
        reqwest::Client::new(),
        Credential::Run(run.clone()),
    );
    let declared = types::PublicationDeclaration {
        objects: vec![object(b"machine")],
    };
    let opened = machine
        .open_model_publication(&account, "model", "op-machine", &declared)
        .await
        .unwrap()
        .into_inner();
    assert_eq!(opened.publication.operation, "op-machine");
    let absent = format!("sha256:{:x}", Sha256::digest(b"absent"));
    assert_eq!(
        refused(
            machine
                .get_model_checkpoint(&account, "model", &absent)
                .await
                .map_err(Error::from)
        )
        .0,
        404
    );
    let other = machine
        .open_model_publication(&account, "other", "op", &declared)
        .await
        .map_err(Error::from);
    assert_eq!(
        refused(other),
        (403, "capability.operation_forbidden".into())
    );
    assert!(
        run.ended().is_none(),
        "a refused operation does not end the capability"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// With the docker provider (`hub_set=provider.kind=docker provider.docker.network=host`), the
/// hub sells its CPU machine: a rental of it boots the registered worker image, which reaches the
/// hub at its loopback origin through the CA the hub grants it, turns ready, and ends.
#[tokio::test]
async fn a_rental_boots_to_ready_and_ends() {
    let Some(hub) = hub() else {
        eprintln!("skipped: no hub");
        return;
    };
    let http = reqwest::Client::new();
    let anonymous = Client::connect(&hub.url, http.clone(), Credential::Anonymous);
    let skus = anonymous.list_rental_skus(None).await.unwrap().into_inner();
    if !skus.iter().any(|sku| sku.name == "cpu") {
        eprintln!("skipped: the hub sells no cpu machine (hub_set provider.kind=docker)");
        return;
    }
    let nonce = format!("{:x}", Sha256::digest(format!("{:?}", Instant::now())))[..10].to_string();
    let email = format!("rental-{nonce}@example.test");
    let (key, bearer) = testing::sign_up(&anonymous, &email, Path::new(&hub.log))
        .await
        .unwrap();
    let user = Client::connect(&hub.url, http, Credential::Bearer(bearer.token));
    user.register_account(&format!("rent{nonce}"), &types::Empty(Default::default()))
        .await
        .unwrap();
    let ask = types::RentalRequest {
        accelerator_count: None,
        container_disk_gb: None,
        creator_public_key: key.public_key(),
        development: None,
        excluded_provider_machines: vec![],
        image: None,
        media_token_sha256: format!("{:x}", Sha256::digest(nonce.as_bytes())),
        name: format!("live-{nonce}"),
        planned_source_bytes: None,
        provider: None,
        serving_models: vec![],
        sku: "cpu".into(),
    };
    let idempotency = types::CreateRentalIdempotencyKey::try_from(format!("live-{nonce}")).unwrap();
    let rental = user
        .create_rental(&idempotency, &ask)
        .await
        .unwrap()
        .into_inner();
    let state = |want: &'static [&'static str]| {
        let user = &user;
        let id = rental.rental_id.clone();
        async move {
            // The image pull dominates; a rental that leaves the path fails at once.
            let started = Instant::now();
            loop {
                let now = user.get_rental(&id).await.unwrap().into_inner();
                if want.contains(&now.state.as_str()) {
                    return now;
                }
                assert!(
                    [
                        "pending_acquisition",
                        "acquiring",
                        "booting",
                        "release_requested"
                    ]
                    .contains(&now.state.as_str()),
                    "rental {id} left the path: {now:?}"
                );
                assert!(
                    started.elapsed() < Duration::from_secs(1200),
                    "rental {id} stuck in {}",
                    now.state
                );
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }
    };
    state(&["ready"]).await;
    user.release_rental(&rental.rental_id).await.unwrap();
    state(&["released"]).await;
}

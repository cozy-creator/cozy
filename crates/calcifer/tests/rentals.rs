//! Rentals and run capabilities at a real Tensorhub, which Tensorhub's dev-build `cozy_ref` job
//! starts: TENSORHUB_ORIGIN is its https origin (the job's SSL_CERT_FILE trusts it), TENSORHUB_LOG
//! where it writes emailed codes; COZY_TEST_TENSORD the machine. No rental is bought here, and
//! nothing is spent.
mod common;

use std::{path::Path, sync::Arc};

use base64::Engine as _;
use calcifer::{Calcifer, machines::Machines};
use cozy_platform::paths::Home;
use cozy_proto::{
    calcifer::{Kind, On, SubmitRun},
    config, machine as v1,
};
use hub_client::{Client, Credential, DeviceKey, P256Key, RunToken, UserSession, testing, types};
use prost::Message as _;
use sha2::{Digest as _, Sha256};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_hub_is_listed_signed_in_and_a_rent_it_will_not_price_records_nothing() {
    let var = |name| std::env::var(name).ok().filter(|v: &String| !v.is_empty());
    // The job's SSL_CERT_FILE trusts the Hub's origin.
    let (Some(url), Some(log)) = (var("TENSORHUB_ORIGIN"), var("TENSORHUB_LOG")) else {
        eprintln!("skipped: TENSORHUB_ORIGIN and TENSORHUB_LOG name no hub");
        return;
    };
    let home = tempfile::tempdir().unwrap();
    let origin = config::origin(&url).unwrap();
    let hub = config::Hub {
        name: Some("ci".into()),
        origin: origin.clone(),
    };
    let nonce = format!(
        "{:x}",
        Sha256::digest(home.path().to_string_lossy().as_bytes())
    );
    let anonymous = Client::connect(&url, reqwest::Client::new(), Credential::Anonymous);
    let email = format!("calcifer-{}@example.test", &nonce[..10]);
    let (key, _) = testing::sign_up(&anonymous, &email, Path::new(&log))
        .await
        .unwrap();
    let key_path = hub.device_key(&Home::at(home.path()));
    key.save(&key_path).unwrap();
    let session = UserSession::new(DeviceKey::load(&key_path).unwrap(), None);
    Client::connect(
        &url,
        reqwest::Client::new(),
        Credential::User(Arc::new(session)),
    )
    .register_account(
        &format!("calcifer{}", &nonce[..10]),
        &types::Empty(Default::default()),
    )
    .await
    .unwrap();

    let calcifer =
        Calcifer::open(&home.path().join("calcifer"), &home.path().join("outputs")).unwrap();
    let config = config::Config {
        hubs: [("ci".to_owned(), origin.clone())].into(),
        ..Default::default()
    };
    let machines = Machines::start(calcifer.clone(), Home::at(home.path()), &config);
    let listed = machines.list(true).await.unwrap();
    assert!(listed.hubs.iter().any(|h| h.hub == origin), "{listed:?}");
    assert!(
        listed.unanswered.is_empty() && !listed.signed_out.contains(&origin),
        "{listed:?}"
    );
    assert!(listed.machines.is_empty(), "this account rents nothing");

    // The Hub prices the exact ask before anything is recorded; it sells nothing here, so the
    // rent is refused and nothing is recorded or kept.
    let refused = machines
        .rent("calcifer-key-1", &origin, "rtx4090", None, None)
        .await
        .unwrap_err();
    assert!(
        ["hub_refused", "hub_unavailable"].contains(&refused.code.as_str()),
        "{refused:?}"
    );
    assert!(calcifer.store().ask("calcifer-key-1").unwrap().is_none());
    assert!(calcifer.store().machines().unwrap().is_empty());
    let keys = std::fs::read_dir(Home::at(home.path()).keys()).unwrap();
    assert!(
        keys.flatten()
            .all(|key| !key.file_name().to_string_lossy().starts_with("rental-")),
        "no rental key was kept"
    );

    // A bare `--on new` asks the Hub's placement; nothing holds this run on a Hub with no
    // provider, so the run fails typed and nothing is rented.
    let run = calcifer
        .submit(
            "calcifer-run-1",
            SubmitRun {
                on: On::New(None),
                hub: origin.clone(),
                kind: Kind::Call,
                target: "cozy-test/held/generate".into(),
                spec: v1::RunSpec {
                    source: Some(v1::run_spec::Source::Release(v1::Release {
                        package: "cozy-test/held".into(),
                        release: String::new(),
                    })),
                    entrypoint: "generate".into(),
                    ..Default::default()
                }
                .encode_to_vec(),
                inputs: vec![],
                out: None,
                warnings: vec![],
            },
        )
        .await
        .unwrap()
        .run
        .id;
    // A hang bound only; the product waits without one.
    let mut failed = None;
    for _ in 0..1200 {
        let row = calcifer.store().run(run.as_str()).unwrap().unwrap();
        if row.status.is_some() {
            failed = Some(row);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    let failed = failed.expect("the run ended");
    let reason = failed.reason.unwrap();
    assert!(
        ["hub_refused", "placement_none_fits"].contains(&reason.code.as_str()),
        "{reason:?}"
    );
    assert!(
        calcifer
            .store()
            .ask(&format!("run-{run}"))
            .unwrap()
            .is_none()
    );
    assert!(calcifer.store().machines().unwrap().is_empty());
    assert_eq!(
        machines.end("kiyui").await.unwrap_err().code,
        "machine_not_found"
    );
    machines.close();
    calcifer.close();
}

/// A run that publishes carries its owner's capability, minted at the send for its machine's TLS
/// leaf: the machine (here the test, holding that leaf's key) trades it at the Hub's token
/// endpoint and publishes into the run's destination, and nowhere else.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_runs_capability_is_bound_to_its_machine_and_traded_at_its_hub() {
    let var = |name| std::env::var(name).ok().filter(|v: &String| !v.is_empty());
    // The job's SSL_CERT_FILE trusts the Hub's origin, which machines read too.
    let (Some(url), Some(log)) = (var("TENSORHUB_ORIGIN"), var("TENSORHUB_LOG")) else {
        eprintln!("skipped: TENSORHUB_ORIGIN and TENSORHUB_LOG name no hub");
        return;
    };
    let home = tempfile::tempdir().unwrap();
    let origin = config::origin(&url).unwrap();
    let nonce = format!(
        "{:x}",
        Sha256::digest(home.path().to_string_lossy().as_bytes())
    );
    let account = format!("calcifer{}", &nonce[..10]);
    let anonymous = Client::connect(&url, reqwest::Client::new(), Credential::Anonymous);
    let email = format!("calcifer-{}@example.test", &nonce[..10]);
    let (key, _) = testing::sign_up(&anonymous, &email, Path::new(&log))
        .await
        .unwrap();
    let hub = config::Hub {
        name: None,
        origin: origin.clone(),
    };
    let key_path = hub.device_key(&Home::at(home.path()));
    key.save(&key_path).unwrap();
    let session = UserSession::new(DeviceKey::load(&key_path).unwrap(), None);
    Client::connect(
        &url,
        reqwest::Client::new(),
        Credential::User(Arc::new(session)),
    )
    .register_account(&account, &types::Empty(Default::default()))
    .await
    .unwrap();

    // The machine: a real TensorD, whose leaf key the test reads to act as it at the Hub.
    let machine = common::Tensord::start();
    let calcifer =
        Calcifer::open(&home.path().join("calcifer"), &home.path().join("outputs")).unwrap();
    let key = common::key_file(home.path());
    calcifer.put_machine(&machine.row("m", &key)).unwrap();
    let machines = Machines::start(calcifer.clone(), Home::at(home.path()), &Default::default());
    let submit = |key: &'static str, destination: String| {
        let spec = v1::RunSpec {
            kind: v1::RunKind::Warm as i32,
            weights_destination: destination,
            ..Default::default()
        };
        calcifer.submit(
            key,
            SubmitRun {
                on: On::Machine("m".into()),
                hub: origin.clone(),
                kind: Kind::Warm,
                target: "upload".into(),
                spec: spec.encode_to_vec(),
                inputs: vec![],
                out: None,
                warnings: vec![],
            },
        )
    };
    let publishes = submit("k-publish", format!("{account}/model"))
        .await
        .unwrap()
        .run
        .id;
    let access = machines.hub_access(&publishes, "m").await.unwrap().unwrap();
    assert!(!access.capability.is_empty() && !access.token_endpoint.is_empty());
    assert_eq!(
        access.origin, origin,
        "where machines read this Hub, as it says"
    );
    let public_only = submit("k-public", String::new()).await.unwrap().run.id;
    let public_access = machines
        .hub_access(&public_only, "m")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(public_access.origin, access.origin);
    assert!(public_access.capability.is_empty() && public_access.token_endpoint.is_empty());

    // As the machine: trade it with the leaf's key, then publish where it names, and only there.
    let identity: serde_json::Value = serde_json::from_slice(
        &std::fs::read(machine.root.path().join("config/identity/identity.json")).unwrap(),
    )
    .unwrap();
    let pem = identity["private_key_pem"].as_str().unwrap();
    let body: String = pem.lines().filter(|l| !l.starts_with("-----")).collect();
    let der = base64::engine::general_purpose::STANDARD
        .decode(body)
        .unwrap();
    let leaf = Arc::new(P256Key::from_pkcs8(&der).unwrap());
    let token = RunToken::new(&access.capability, &access.token_endpoint, leaf).unwrap();
    let machine = Client::connect(
        &origin,
        reqwest::Client::new(),
        Credential::Run(Arc::new(token)),
    );
    let declared = types::PublicationDeclaration {
        objects: vec![types::DeclaredObject {
            object_id: format!("sha256:{:x}", Sha256::digest(b"weights")),
            length: 7,
        }],
    };
    machine
        .open_model_publication(&account, "model", "op-calcifer", &declared)
        .await
        .unwrap();
    let elsewhere = machine
        .open_model_publication(&account, "other", "op-other", &declared)
        .await
        .map_err(hub_client::Error::from);
    assert!(
        matches!(elsewhere, Err(hub_client::Error::Hub { status: 403, .. })),
        "{elsewhere:?}"
    );
    machines.close();
    calcifer.close();
}

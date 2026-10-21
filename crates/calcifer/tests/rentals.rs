//! Rentals and runs' Hub access at a real Tensorhub, which Tensorhub's dev-build `cozy_ref` job
//! starts: TENSORHUB_ORIGIN is its https origin (the job's SSL_CERT_FILE trusts it), TENSORHUB_LOG
//! where it writes emailed codes; COZY_TEST_TENSORD the machine. No rental is bought here, and
//! nothing is spent.
mod common;

use std::{path::Path, sync::Arc};

use calcifer::{Calcifer, machines::Machines};
use platform::paths::Home;
use protocol::{
    calcifer::{Agreed, Kind, On, SubmitRun},
    config, machine as v1,
};
use hub_client::{Client, Credential, DeviceKey, UserSession, testing, types};
use sha2::{Digest as _, Sha256};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_hub_is_listed_signed_in_and_a_rent_it_will_not_price_records_nothing() {
    let var = |name| std::env::var(name).ok().filter(|v: &String| !v.is_empty());
    // The job's SSL_CERT_FILE trusts the Hub's origin.
    let (Some(url), Some(log)) = (var("TENSORHUB_ORIGIN"), var("TENSORHUB_LOG")) else {
        return no_hub();
    };
    let home = tempfile::tempdir().unwrap();
    let origin = config::Origin::parse(&url).unwrap().to_string();
    let hub = config::Hub {
        name: Some("ci".into()),
        origin: config::Origin::parse(&origin).unwrap(),
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
        .rent("calcifer-key-1", &origin, "rtx4090", Default::default())
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
    // provider, so the run fails typed and nothing is rented. Without an agreed quote it is
    // refused before it is recorded.
    let held = |agreed| SubmitRun {
        agreed,
        on: On::New(None),
        hub: origin.clone(),
        kind: Kind::Call,
        target: "cozy-test/held/generate".into(),
        spec: v1::RunSpec {
            source: Some(v1::Source::Release(v1::Release {
                package: "cozy-test/held".into(),
                release: String::new(),
            })),
            entrypoint: "generate".into(),
            ..Default::default()
        },
        inputs: vec![],
        out: None,
        warnings: vec![],
    };
    let unagreed = calcifer.submit("calcifer-run-0", held(None)).await;
    let Err(calcifer::Error::Refused(refused)) = unagreed else { panic!("{unagreed:?}") };
    assert_eq!(refused.code, "quote_unagreed");
    let agreed = Agreed { usd_micros_per_hour: 1_000_000_000, usd_micros_per_gb: 1_000_000 };
    let run = calcifer
        .submit("calcifer-run-1", held(Some(agreed)))
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
        machines.end("kiyui", false).await.unwrap_err().code,
        "machine_not_found"
    );
    machines.close();
    calcifer.close();
}

/// A run reads its Hub where the Hub says machines read it, under any spelling of that Hub, and a
/// run that publishes gets its owner's access there. Local code's org-relative references name
/// this computer's account at the Hub.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_run_reads_its_hub_where_the_hub_says() {
    let var = |name| std::env::var(name).ok().filter(|v: &String| !v.is_empty());
    // The job's SSL_CERT_FILE trusts the Hub's origin, which machines read too.
    let (Some(url), Some(log)) = (var("TENSORHUB_ORIGIN"), var("TENSORHUB_LOG")) else {
        return no_hub();
    };
    let home = tempfile::tempdir().unwrap();
    let origin = config::Origin::parse(&url).unwrap().to_string();
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
        origin: config::Origin::parse(&origin).unwrap(),
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

    let tensord = common::Machine::start();
    let calcifer =
        Calcifer::open(&home.path().join("calcifer"), &home.path().join("outputs")).unwrap();
    let key = common::key_file(home.path());
    calcifer.put_machine(&tensord.row("m", &key)).unwrap();
    let machines = Machines::start(calcifer.clone(), Home::at(home.path()), &Default::default());
    let submit = |key: &'static str, destination: String| {
        let spec = v1::RunSpec {
            kind: v1::RunKind::Warm,
            weights_destination: destination,
            ..Default::default()
        };
        calcifer.submit(
            key,
            SubmitRun {
                agreed: None,
                on: On::Machine("m".into()),
                hub: origin.clone(),
                kind: Kind::Warm,
                target: "upload".into(),
                spec,
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
    let access = machines.hub_access(&publishes).await.unwrap().unwrap();
    assert_eq!(
        access.origin, origin,
        "where machines read this Hub, as it says"
    );
    let public_only = submit("k-public", String::new()).await.unwrap().run.id;
    let public_access = machines.hub_access(&public_only).await.unwrap().unwrap();
    assert_eq!(public_access, access);

    // A machine writes its installations' Hub its own way (`https://tensorhub.com:443`), and
    // `package update` submits under that spelling: the run is still this Hub's, read as a run
    // under the configured spelling reads it.
    let respelled = calcifer
        .submit(
            "k-respelled",
            SubmitRun {
                agreed: None,
                on: On::Machine("m".into()),
                hub: machine_spelling(&origin),
                kind: Kind::Warm,
                target: "update".into(),
                spec: v1::RunSpec {
                    kind: v1::RunKind::Warm,
                    ..Default::default()
                },
                inputs: vec![],
                out: None,
                warnings: vec![],
            },
        )
        .await
        .unwrap()
        .run;
    assert_eq!(respelled.hub, origin);
    let row = calcifer
        .store()
        .run(respelled.id.as_str())
        .unwrap()
        .unwrap();
    assert_eq!(row.hub.as_deref(), Some(origin.as_str()));
    let respelled_access = machines.hub_access(&respelled.id).await.unwrap();
    assert_eq!(
        respelled_access,
        Some(public_access),
        "the run reads its Hub as a run under the configured spelling does"
    );

    // Local code's org-relative references name this account: the machine got the run so.
    let (staged, local) = common::local_spec(home.path());
    let run = SubmitRun {
        agreed: None,
        on: On::Machine("m".into()),
        hub: origin.clone(),
        kind: Kind::Warm,
        target: "local/hello".into(),
        spec: local.clone(),
        inputs: vec![staged],
        out: None,
        warnings: vec![],
    };
    let id = calcifer.submit("k-local", run).await.unwrap().run.id;
    for _ in 0..1200 {
        let row = calcifer.store().run(id.as_str()).unwrap().unwrap();
        if row.accepted_at_ms.is_some() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    let sent = |owner: &str| v1::RunSpec {
        owner: owner.into(),
        hub: Some(v1::HubAccess {
            origin: access.origin.clone(),
            ..Default::default()
        }),
        ..local.clone()
    };
    let link = calcifer::Link::new(&tensord.endpoint, common::key()).unwrap();
    assert!(common::holds_as(&link, &id, sent(&account)).await);
    assert!(!common::holds_as(&link, &id, sent("")).await);
    machines.close();
    calcifer.close();
}

/// `origin` as a machine may write it: the scheme in capitals, the default port spelled out, and
/// a trailing slash.
fn machine_spelling(origin: &str) -> String {
    let (scheme, rest) = origin.split_once("://").unwrap();
    let port = match (rest.contains(':'), scheme) {
        (true, _) => "",
        (false, "https") => ":443",
        (false, _) => ":80",
    };
    format!("{}://{rest}{port}/", scheme.to_uppercase())
}

/// A hosted job without a Hub fails the test, so a green run means it ran: tensorhub's cozy_ref
/// job names one. Elsewhere it skips, saying why.
fn no_hub() {
    assert!(
        std::env::var_os("GITHUB_ACTIONS").is_none(),
        "TENSORHUB_ORIGIN and TENSORHUB_LOG are unset; the job must provide a Tensorhub: \
         run it in tensorhub's dev-build cozy_ref job"
    );
    eprintln!("skipped: TENSORHUB_ORIGIN and TENSORHUB_LOG name no hub");
}

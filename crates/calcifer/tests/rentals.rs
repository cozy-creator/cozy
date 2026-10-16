//! Rentals at a real Tensorhub, which Tensorhub's dev-build `cozy_ref` job starts (TENSORHUB_URL
//! is where it listens, TENSORHUB_LOG where it writes emailed codes). It has no provider:
//! nothing can be bought, so nothing is spent.
use std::{path::Path, sync::Arc};

use calcifer::{Calcifer, machines::Machines};
use cozy_platform::paths::Home;
use cozy_proto::config;
use hub_client::{Client, Credential, DeviceKey, UserSession, testing, types};
use sha2::{Digest as _, Sha256};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_hub_is_listed_signed_in_and_a_rent_it_will_not_price_records_nothing() {
    let var = |name| std::env::var(name).ok().filter(|v: &String| !v.is_empty());
    let (Some(url), Some(log)) = (var("TENSORHUB_URL"), var("TENSORHUB_LOG")) else {
        eprintln!("skipped: TENSORHUB_URL and TENSORHUB_LOG name no hub");
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
    let machines = Machines::start(calcifer.clone(), Home::at(home.path()), vec![hub]);
    let listed = machines.list(true).await.unwrap();
    assert!(listed.hubs.iter().any(|h| h.hub == origin), "{listed:?}");
    assert!(
        listed.unanswered.is_empty() && listed.signed_out.is_empty(),
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
    assert_eq!(
        machines.end("kiyui").await.unwrap_err().code,
        "machine_not_found"
    );
    machines.close();
    calcifer.close();
}

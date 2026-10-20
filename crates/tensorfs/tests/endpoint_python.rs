//! Public Python API and Runtime library publication against the real Hub fixture.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use hub_client::{Credential, types};
use serde_json::json;
use tensorfs::sha256::hex_digest;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn public_python_api_uses_an_ordinary_owner_credential() {
    let var = |name| std::env::var(name).ok().filter(|v: &String| !v.is_empty());
    if var("TENSORHUB_URL").is_none() && var("TENSORFS_TEST_REQUIRE_ENDPOINT").is_none() {
        eprintln!("skipped: the real Hub fixture and installed TensorFS wheel are required");
        return;
    }
    let required = |name| var(name).unwrap_or_else(|| panic!("the endpoint proof requires {name}"));
    let (reach, ca, log, python) = (
        required("TENSORHUB_URL"),
        required("TENSORHUB_CA"),
        required("TENSORHUB_LOG"),
        required("TENSORFS_TEST_PYTHON"),
    );
    let anonymous_endpoint = required("TENSORHUB_ORIGIN");
    assert!(
        anonymous_endpoint.starts_with("https://"),
        "the anonymous endpoint/CA proof requires the fixture's TLS origin"
    );
    let native_source = required("TENSORFS_TEST_NATIVE_SOURCE_SHA");
    let native_sha256 = required("TENSORFS_TEST_NATIVE_SHA256");
    assert!(
        native_source.len() == 40 && native_source.bytes().all(|b| b.is_ascii_hexdigit()),
        "the native source must be a full commit SHA"
    );
    assert!(
        native_sha256.len() == 64 && native_sha256.bytes().all(|b| b.is_ascii_hexdigit()),
        "the native wheel must have an expected SHA256"
    );
    let http = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    let nonce =
        &hex_digest(format!("{}{:?}", std::process::id(), std::time::SystemTime::now()).as_bytes())
            [..10];
    let anonymous = hub_client::Client::connect(&reach, http.clone(), Credential::Anonymous);
    let (_, bearer) = hub_client::testing::sign_up(
        &anonymous,
        &format!("tfs-python-{nonce}@example.test"),
        Path::new(&log),
    )
    .await
    .unwrap();
    let (_, rival_bearer) = hub_client::testing::sign_up(
        &anonymous,
        &format!("tfs-python-rival-{nonce}@example.test"),
        Path::new(&log),
    )
    .await
    .unwrap();
    let user = hub_client::Client::connect(
        &reach,
        http.clone(),
        Credential::Bearer(bearer.token.clone()),
    );
    let rival =
        hub_client::Client::connect(&reach, http, Credential::Bearer(rival_bearer.token.clone()));
    let account = format!("tfspython{nonce}");
    let rival_account = format!("tfsrival{nonce}");
    user.register_account(&account, &types::Empty(Default::default()))
        .await
        .unwrap();
    rival
        .register_account(&rival_account, &types::Empty(Default::default()))
        .await
        .unwrap();
    let scratch = std::env::temp_dir().join(format!("tfs-python-api-{nonce}"));
    let request = json!({
        "endpoint": reach, "anonymous_endpoint": anonymous_endpoint,
        "ca": ca, "model": format!("{account}/model"),
        "rival_model": format!("{rival_account}/model"), "runtime_model": format!("{account}/runtime"),
        "token": bearer.token, "rival_token": rival_bearer.token, "cache_dir": scratch,
        "creator_source": Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."),
        "native_source": native_source, "native_sha256": native_sha256,
        "native_wheel": required("COZY_TEST_TENSORFS_WHEEL"),
        "runtime_source": required("COZY_TEST_RUNTIME_SOURCE"), "runtime_wheel": required("COZY_TEST_RUNTIME_WHEEL"),
    });
    let script =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../tensorfs-py/scripts/endpoint_user_api.py");
    let mut child = Command::new(python)
        .arg("-I")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&serde_json::to_vec(&request).unwrap())
        .unwrap();
    let started = Instant::now();
    while child.try_wait().unwrap().is_none() {
        if started.elapsed() > Duration::from_secs(120) {
            child.kill().unwrap();
            let _ = child.wait();
            panic!("the owned Python endpoint proof exceeded 120 seconds");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let output = child.wait_with_output().unwrap();
    let redact = |bytes: &[u8]| {
        String::from_utf8_lossy(bytes)
            .replace(&bearer.token, "[owner credential]")
            .replace(&rival_bearer.token, "[rival credential]")
    };
    let stdout = redact(&output.stdout);
    assert!(
        output.status.success(),
        "Python endpoint proof: {stdout}\n{}",
        redact(&output.stderr)
    );
    let receipt: serde_json::Value =
        serde_json::from_str(&stdout).expect("the endpoint proof must return its JSON receipt");
    assert_eq!(receipt["tensorfs"]["rival_denied"], true);
    assert_eq!(receipt["tensorfs"]["retained_upload_without_rerun"], true);
    assert_eq!(
        receipt["runtime_library"]["publication_replay_without_rewrite"],
        true
    );
    assert_eq!(receipt["cold_anonymous_read"]["cold_store"], true);
    assert_eq!(receipt["cold_anonymous_read"]["anonymous"], true);
    assert_eq!(
        receipt["cold_anonymous_read"]["metadata_only_then_payload"],
        true
    );
    println!("{stdout}");
    let _ = std::fs::remove_dir_all(scratch);
}

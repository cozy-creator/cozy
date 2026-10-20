//! Public Python API against the CI Hub/AuthKit/MinIO fixture with ordinary owner credentials.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use hub_client::{types, Credential};
use serde_json::json;
use tensorfs::sha256::hex_digest;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn public_python_api_uses_an_ordinary_owner_credential() {
    let var = |name| std::env::var(name).ok().filter(|v: &String| !v.is_empty());
    let (Some(reach), Some(ca), Some(log), Some(python)) = (var("TENSORHUB_URL"), var("TENSORHUB_CA"), var("TENSORHUB_LOG"), var("TENSORFS_TEST_PYTHON")) else {
        eprintln!("skipped: the real Hub fixture and installed TensorFS wheel are required");
        return;
    };
    let http = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()).build().unwrap();
    let nonce = &hex_digest(format!("{}{:?}", std::process::id(), std::time::SystemTime::now()).as_bytes())[..10];
    let anonymous = hub_client::Client::connect(&reach, http.clone(), Credential::Anonymous);
    let (_, bearer) = hub_client::testing::sign_up(&anonymous, &format!("tfs-python-{nonce}@example.test"), Path::new(&log)).await.unwrap();
    let user = hub_client::Client::connect(&reach, http, Credential::Bearer(bearer.token.clone()));
    let account = format!("tfspython{nonce}");
    user.register_account(&account, &types::Empty(Default::default())).await.unwrap();
    let scratch = std::env::temp_dir().join(format!("tfs-python-api-{nonce}"));
    let request = json!({"endpoint": reach, "ca": ca, "model": format!("{account}/model"), "token": bearer.token, "cache_dir": scratch});
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tensorfs/scripts/endpoint_user_api.py");
    let mut child = Command::new(python).arg("-I").arg(script).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    child.stdin.take().unwrap().write_all(&serde_json::to_vec(&request).unwrap()).unwrap();
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
    assert!(output.status.success(), "Python owner endpoint proof: {}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    println!("{}", String::from_utf8_lossy(&output.stdout));
    let _ = std::fs::remove_dir_all(scratch);
}

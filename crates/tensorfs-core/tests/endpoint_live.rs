//! The DPoP provider against a real Tensorhub and its AuthKit: tensorhub's dev-build `cozy_ref`
//! job (`scripts/live-hub-client.sh`) runs it with TENSORHUB_ORIGIN, TENSORHUB_CA and
//! TENSORHUB_LOG; anywhere else it is skipped. A user signs up, mints a capability for a fresh
//! P-256 key offline with their device key, and the provider trades it at AuthKit's token
//! endpoint once (JWT-bearer, DPoP-proved) and proves every Hub request with it.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use aws_lc_rs::encoding::AsBigEndian;
use aws_lc_rs::signature::{EcdsaKeyPair, KeyPair, ECDSA_P256_SHA256_FIXED_SIGNING};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use hub_client::{types, Credential, RunCapability};
use serde_json::{json, Map};
use tensorfs_core::fetch::token::Request;
use tensorfs_core::fetch::{Client, DPoP, JwtBearer, SourcePolicy, TokenProvider};
use tensorfs_core::sha256::hex_digest;

/// A fresh P-256 key as a private JWK, and its thumbprint.
fn key() -> (String, String) {
    let pair = EcdsaKeyPair::generate(&ECDSA_P256_SHA256_FIXED_SIGNING).unwrap();
    let b64 = |bytes: &[u8]| URL_SAFE_NO_PAD.encode(bytes);
    let point = pair.public_key().as_ref();
    let (x, y) = (b64(&point[1..33]), b64(&point[33..65]));
    let d = b64(pair.private_key().as_be_bytes().unwrap().as_ref());
    let jwk = json!({"kty": "EC", "crv": "P-256", "x": x, "y": y, "d": d}).to_string();
    (jwk, hub_client::thumbprint(&x, &y))
}

/// `PUT url` with `body`, proved by `provider`, asked once more when it says so: the status.
async fn put(http: &reqwest::Client, provider: &DPoP, url: &str, body: &serde_json::Value) -> u16 {
    let mut retried = false;
    loop {
        let headers = provider.headers(&Request { method: "PUT", url, headers: &[], payload: None }).await.unwrap();
        let mut request = http.put(url).json(body);
        for (name, value) in &headers {
            request = request.header(name, value);
        }
        let answer = request.send().await.unwrap();
        let sent = Request { method: "PUT", url, headers: &headers, payload: None };
        if !retried && provider.retry(&sent, answer.status().as_u16(), answer.headers()) {
            retried = true;
            continue;
        }
        return answer.status().as_u16();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_capability_is_traded_once_and_proves_each_request() {
    let var = |name| std::env::var(name).ok().filter(|v: &String| !v.is_empty());
    let (Some(origin), Some(ca), Some(log)) = (var("TENSORHUB_ORIGIN"), var("TENSORHUB_CA"), var("TENSORHUB_LOG")) else {
        eprintln!("skipped: no Tensorhub (TENSORHUB_ORIGIN, TENSORHUB_CA, TENSORHUB_LOG)");
        return;
    };
    let http = reqwest::Client::new();
    let nonce = &hex_digest(format!("{}{:?}", std::process::id(), std::time::SystemTime::now()).as_bytes())[..10];
    let anonymous = hub_client::Client::connect(&origin, http.clone(), Credential::Anonymous);
    let (device, bearer) = hub_client::testing::sign_up(&anonymous, &format!("tfs-dpop-{nonce}@example.test"), Path::new(&log))
        .await
        .unwrap();
    let user = hub_client::Client::connect(&origin, http.clone(), Credential::Bearer(bearer.token));
    let account = format!("tfsdpop{nonce}");
    user.register_account(&account, &types::Empty(Default::default())).await.unwrap();
    let user_id = user.get_current_account().await.unwrap().into_inner().user_id;
    let token_endpoint = user.token_endpoint().await.unwrap();

    let (jwk, thumbprint) = key();
    let capability = device.sign_run_capability(&RunCapability {
        user_id,
        hub: origin.clone(),
        key_jkt: thumbprint.clone(),
        run: format!("r-{nonce}"),
        operations: vec![json!({"type": "tensorhub_model_publish", "model": format!("{account}/model")})],
        lifetime: Duration::from_secs(600),
    });
    let policy = SourcePolicy { any_host: true, allow_local: true, ..Default::default() };
    let client = Client::with_roots(policy, &std::fs::read(&ca).unwrap()).unwrap();
    let grant = JwtBearer {
        token_endpoint,
        client_id: "tensord".into(),
        resource: origin.clone(),
        claims: Map::from_iter([("capability".to_owned(), json!(capability))]),
    };
    let provider = Arc::new(DPoP::grant(&client, &jwk, grant).unwrap());
    assert_eq!(provider.thumbprint(), thumbprint);

    let elsewhere = Request { method: "GET", url: "https://elsewhere.example/x", headers: &[], payload: None };
    assert!(provider.headers(&elsewhere).await.unwrap().is_empty(), "only the resource's origin sees the token");
    let declared = json!({"objects": [{"object_id": format!("sha256:{}", hex_digest(b"weights")), "length": 7}]});
    let publication = |model: &str| format!("{origin}/v1/models/{account}/{model}/publications/op-{nonce}");
    assert_eq!(put(&http, &provider, &publication("model"), &declared).await, 200, "the capability's own model");
    // The same token again (a second trade of one capability is refused): denied the model it does not name.
    assert_eq!(put(&http, &provider, &publication("other"), &declared).await, 403);
}

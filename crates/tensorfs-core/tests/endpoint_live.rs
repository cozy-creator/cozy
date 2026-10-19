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
use tensorfs_core::err::Code;
use tensorfs_core::fetch::{bearer as bearer_provider, upload_tree, with_tree, Client, Config, DPoP, Endpoint, Fetcher, JwtBearer, ObjectStore, Origin, SourcePolicy, TokenProvider, Want};
use tensorfs_core::format::{Dtype, Tree};
use tensorfs_core::sha256::hex_digest;
use tensorfs_core::store2::Store;
use tensorfs_core::writer::{Data, Source};

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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_capability_is_traded_once_and_proves_each_request() {
    let var = |name| std::env::var(name).ok().filter(|v: &String| !v.is_empty());
    let (Some(origin), Some(reach), Some(ca), Some(log)) = (var("TENSORHUB_ORIGIN"), var("TENSORHUB_URL"), var("TENSORHUB_CA"), var("TENSORHUB_LOG")) else {
        eprintln!("skipped: no Tensorhub (TENSORHUB_ORIGIN, TENSORHUB_URL, TENSORHUB_CA, TENSORHUB_LOG)");
        return;
    };
    let http = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()).build().unwrap();
    let nonce = &hex_digest(format!("{}{:?}", std::process::id(), std::time::SystemTime::now()).as_bytes())[..10];
    let anonymous = hub_client::Client::connect(&reach, http.clone(), Credential::Anonymous);
    let (device, bearer) = hub_client::testing::sign_up(&anonymous, &format!("tfs-dpop-{nonce}@example.test"), Path::new(&log))
        .await
        .unwrap();
    let user = hub_client::Client::connect(&reach, http.clone(), Credential::Bearer(bearer.token.clone()));
    let account = format!("tfsdpop{nonce}");
    user.register_account(&account, &types::Empty(Default::default())).await.unwrap();
    let user_id = user.get_current_account().await.unwrap().into_inner().user_id;
    let token_endpoint = user.token_endpoint().await.unwrap();

    let policy = SourcePolicy { any_host: true, allow_local: true, ..Default::default() };
    let client = Client::with_roots(policy, &std::fs::read(&ca).unwrap()).unwrap();
    let scratch = std::env::temp_dir().join(format!("tfs-endpoint-live-{nonce}"));
    let store = Store::open(scratch.join("source")).unwrap();
    let mut writer = store.writer(None, None).unwrap();
    let weights: Vec<u8> = (0..300_000u32).map(|i| i.wrapping_mul(2_654_435_761) as u8).collect();
    writer.component("model", "conformance.Fixture", Default::default(), None).unwrap()
        .put("w", &Dtype::U8, &[weights.len() as u64], Data::Plain(Source::Read(&mut &weights[..]))).unwrap();
    let header = writer.commit().unwrap();
    let mut tree = Tree::default();
    tree.set_variant("int8", header.clone());
    let base_tree = store.put_tree(&tree).unwrap();
    let model = format!("{account}/model");
    let owner = Endpoint::new(&client, &reach, Some(bearer_provider(bearer.token))).unwrap();
    upload_tree(&owner, &model, &base_tree, &store, None).await.unwrap();
    tree.files.insert("README.md".into(), store.put_bytes(b"worker output\n").unwrap());
    let output_tree = store.put_tree(&tree).unwrap();

    let (jwk, thumbprint) = key();
    let capability = device.sign_run_capability(&RunCapability {
        user_id,
        hub: origin.clone(),
        key_jkt: thumbprint.clone(),
        run: format!("r-{nonce}"),
        operations: vec![
            json!({"type": "tensorhub_model_read", "model": model, "revision": base_tree.id()}),
            json!({"type": "tensorhub_model_upload", "model": model}),
        ],
        lifetime: Duration::from_secs(600),
    });
    let grant = JwtBearer {
        token_endpoint,
        client_id: "tensord".into(),
        resource: origin.clone(),
        claims: Map::from_iter([("capability".to_owned(), json!(capability))]),
    };
    let provider = Arc::new(DPoP::grant(&client, &jwk, grant).unwrap().at(&reach));
    assert_eq!(provider.thumbprint(), thumbprint);
    assert!(provider.issued_token().await.is_none(), "a snapshot does not trigger a trade");

    let elsewhere = Request { method: "GET", url: "https://elsewhere.example/x", headers: &[], payload: None };
    assert!(provider.headers(&elsewhere).await.unwrap().is_empty(), "only the resource's origin sees the token");
    let endpoint = Arc::new(Endpoint::new(&client, &reach, Some(provider.clone())).unwrap());
    let first = upload_tree(&endpoint, &model, &output_tree, &store, None).await.unwrap();
    assert_eq!(first.sent, 2, "the new README and output tree; the base weights deduplicate");
    let issued = provider.issued_token().await.expect("the first upload traded once");
    let resumed = Arc::new(DPoP::token(&jwk, &issued, Some(&origin)).unwrap().at(&reach));
    let resumed_endpoint = Endpoint::new(&client, &reach, Some(resumed)).unwrap();
    assert_eq!(resumed_endpoint.object(&header.sha256).await.unwrap(), Some(header.clone()), "a token snapshot resumes under the same key without another trade");
    // This would fail if each request tried to trade the single-use capability again.
    assert_eq!(upload_tree(&endpoint, &model, &output_tree, &store, None).await.unwrap().sent, 0);

    let mut raced = Vec::new();
    for i in 0..8 {
        let mut variant = Tree::default();
        variant.set_variant("int8", header.clone());
        variant.files.insert("README.md".into(), store.put_bytes(format!("concurrent writer {i}\n").as_bytes()).unwrap());
        raced.push(store.put_tree(&variant).unwrap());
    }
    let writers = raced.iter().cloned().map(|tree| {
        let (endpoint, store, model) = (Arc::clone(&endpoint), store.clone(), model.clone());
        tokio::spawn(async move { upload_tree(&endpoint, &model, &tree, &store, None).await })
    });
    for result in futures::future::join_all(writers).await {
        result.unwrap().unwrap();
    }
    let named = endpoint.repo(&model).await.unwrap().unwrap().0;
    assert!(raced.iter().all(|tree| named.checkpoints.contains(tree)), "all concurrent appends survived CAS rebases");

    let denied = endpoint.update_repo(&model, |old| {
        let mut repo = old.expect("the seeded repo");
        repo.set_release("99.0.0", base_tree.clone());
        Ok(Some(repo))
    }).await.unwrap_err();
    assert_eq!(denied.code, Code::CREDENTIAL_REQUIRED, "capability_exceeded is denied, despite its policy/409 status");
    let conflict = owner.update_repo(&model, |old| {
        let mut repo = old.expect("the seeded repo");
        repo.set_release("1.0.0", base_tree.clone());
        repo.releases[0].revision = 7;
        Ok(Some(repo))
    }).await.unwrap_err();
    assert_eq!(conflict.code, Code::REPOSITORY_CONFLICT, "ordinary revision policy/409 remains a conflict");

    let other = format!("{account}/other");
    let refused = endpoint.update_repo(&other, |old| Ok(Some(with_tree(old, &other, &output_tree, None)))).await.unwrap_err();
    assert_eq!(refused.code, Code::CREDENTIAL_REQUIRED, "the token cannot upload to another repo");

    let url = format!("{reach}/{}", header.key());
    let headers = provider.headers(&Request { method: "GET", url: &url, headers: &[], payload: None }).await.unwrap();
    let mut request = http.get(&url);
    for (name, value) in &headers {
        request = request.header(name, value);
    }
    let redirect = request.send().await.unwrap();
    assert!(redirect.status().is_redirection(), "the Hub redirects blobs to object storage");
    let location = redirect.headers().get("location").unwrap().to_str().unwrap();
    assert_ne!(reqwest::Url::parse(location).unwrap().origin(), reqwest::Url::parse(&reach).unwrap().origin());
    let authorization = headers.iter().find(|(name, _)| name.eq_ignore_ascii_case("authorization")).unwrap().1.clone();
    let unproved = http.get(&url).header("authorization", authorization).send().await.unwrap();
    assert_eq!(unproved.status().as_u16(), 401, "a stolen token without K is unusable");

    // The real MinIO presigned URL accepts the redirected download. Forwarded Hub DPoP
    // credentials would conflict with its SigV4 query authentication.
    let into = Store::open(scratch.join("pulled")).unwrap();
    let fetcher = Fetcher::new(Arc::new(into.clone()) as Arc<dyn ObjectStore>, client.clone(), None, Config::default());
    // Go's real ServeMux cleans /./ with a same-origin redirect first. Each hop must get a
    // new proof naming its actual path before the Hub redirects to MinIO.
    let source: Arc<dyn Origin> = Arc::new(Endpoint::new(&client, &format!("{reach}/."), Some(provider.clone())).unwrap());
    let pulled = fetcher.pull_tree(&source, &base_tree, Some("int8"), &Want::default()).await.unwrap();
    fetcher.settled().await;
    assert_eq!(pulled.header, header);
    assert_eq!(into.checkpoint(&header).unwrap().read("model", "w").unwrap(), weights);
    let _ = std::fs::remove_dir_all(scratch);
}

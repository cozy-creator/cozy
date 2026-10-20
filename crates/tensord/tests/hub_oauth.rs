//! A run's capability against a real AuthKit (th-241): public reads go anonymous; the first
//! private operation trades the owner's device-key-signed capability, once, through the
//! JWT-bearer grant (an assertion the leaf signs, DPoP from the same key); every proof is
//! checked by AuthKit's resource verifier, and the Hub's narrowing, a revoked device key and
//! the capability's expiry each end the run's private work with a typed reason.
#[path = "common/hub_oauth.rs"]
mod hub_oauth;

use tensord::hub::{reason, Capability, Catalog, Leaf, Op, Source};
use serde_json::json;
use std::{sync::Arc, thread::sleep, time::Duration};

/// A key as the machine's identity holds it: a P-256 PKCS#8 PEM.
fn leaf() -> Arc<Leaf> {
    let pem = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256)
        .unwrap()
        .serialize_pem();
    Arc::new(Leaf::from_pem(&pem).unwrap())
}

/// The owner's checkpoint the capability names, read by hash.
const TINY: &str = "sha256:7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d";

fn tiny() -> Op<'static> {
    Op::Read { model: "acme/tiny", manifest: TINY }
}

/// A private read of the owner's checkpoint: it reaches the Hub only verified, which answers
/// that it holds no such checkpoint.
fn read_tiny(catalog: &Catalog) -> Result<serde_json::Value, String> {
    get(catalog, &format!("/v1/models/acme/tiny/checkpoints/{TINY}"))
}

/// One GET at `url` with `headers`, on a runtime of its own: the status and body.
fn fetch(url: &str, headers: Vec<(String, String)>) -> (u16, Vec<u8>) {
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
            runtime.block_on(async {
                let mut request = reqwest::Client::new().get(url);
                for (name, value) in headers {
                    request = request.header(name, value);
                }
                let answer = request.send().await.unwrap();
                (answer.status().as_u16(), answer.bytes().await.unwrap().to_vec())
            })
        })
        .join()
        .unwrap()
    })
}

/// One JSON read at the catalog's Hub, presented as the catalog presents.
fn get(catalog: &Catalog, path: &str) -> Result<serde_json::Value, String> {
    let url = format!("{}{path}", catalog.origin());
    match fetch(&url, catalog.present("GET", &url)) {
        (200, body) => serde_json::from_slice(&body).map_err(|e| e.to_string()),
        (status, body) => Err(format!("HTTP {status}: {}", String::from_utf8_lossy(&body))),
    }
}

#[test]
#[ignore = "real AuthKit: needs go and AUTHKIT_TEST_DATABASE_URL"]
fn a_run_capability_trades_once_for_exactly_what_it_names() {
    let (upstream, hub) = hub_oauth::test_hub();
    let authkit = hub_oauth::AuthKit::start(&upstream);
    let leaf = leaf();
    let jkt = leaf.thumbprint();
    let hosts = vec!["localhost".to_string()];

    // A public-only run: its reads carry nothing, and AuthKit hears of nothing.
    let public = Source::new(&authkit.hub, None, hosts.clone(), None).unwrap();
    let catalog = Catalog::new(&public).unwrap();
    assert_eq!(get(&catalog, "/v1/packages/acme/pkg").unwrap()["package"], "acme/pkg");
    assert_eq!(hub.lock().unwrap().reads, 1);
    assert_eq!(authkit.exchanges(), 0);
    assert!(authkit.verified().is_empty());

    // A run that reads the owner's checkpoint: the capability names it and one read the Hub
    // narrows away. Nothing is traded until the first private operation.
    let narrowed = format!("sha256:{}", "ab".repeat(32));
    let ops = json!([
        {"type": "tensorhub_model_read", "model": "acme/tiny", "manifest": TINY},
        {"type": "tensorhub_model_read", "model": "acme/narrowed", "manifest": narrowed},
    ]);
    let signed = authkit.capability(&jkt, &ops, 600);
    let source = Source::new(&authkit.hub, None, hosts.clone(), Some(Capability::new(&signed, &authkit.token_endpoint, leaf.clone()).unwrap())).unwrap();
    assert_eq!(authkit.exchanges(), 0);
    assert!(read_tiny(&Catalog::for_op(&source, tiny()).unwrap()).unwrap_err().contains("404"));
    assert!(read_tiny(&Catalog::for_op(&source, tiny()).unwrap()).unwrap_err().contains("404"));
    assert_eq!(authkit.exchanges(), 1, "one trade per run");
    let verified = authkit.verified();
    assert!(verified.iter().all(|s| s["owner"] == authkit.owner().as_str() && s["actor"] == jkt.as_str()), "{verified:?}");
    // Public reads of the same run stay anonymous.
    get(&Catalog::new(&source).unwrap(), "/v1/packages/acme/pkg").unwrap();
    assert_eq!(authkit.verified().len(), verified.len());

    // An operation the capability does not name never reaches the Hub; one the Hub narrowed
    // away is refused there.
    let other = Op::Publish { model: "acme/other" };
    assert_eq!(Catalog::for_op(&source, other).err().unwrap().0, reason::REQUIRED);
    let read = Catalog::for_op(&source, Op::Read { model: "acme/narrowed", manifest: &narrowed }).unwrap();
    let refused = get(&read, &format!("/v1/models/acme/narrowed/checkpoints/{narrowed}")).unwrap_err();
    assert_eq!(read.reason("catalog_read_failed", &refused).0, reason::EXCEEDED, "{refused:?}");

    // The token is bound to the leaf: presented with another key's proof, it reads nothing.
    let (key, token) = (hub_client::P256Key::generate(), authkit.last_token());
    let path = format!("/v1/models/acme/tiny/checkpoints/{TINY}");
    let proof = key.proof("GET", &format!("{}{path}", authkit.resource), Some(&token), None);
    let headers = vec![("authorization".into(), format!("DPoP {token}")), ("dpop".into(), proof)];
    let (status, _) = fetch(&format!("{}{path}", authkit.hub), headers);
    assert!(status == 401 || status == 403, "a stolen token without the key is refused: {status}");

    // Revoking the device key ends the run's private work at its next request, with no
    // second trade.
    authkit.revoke_device_key();
    let refused = read_tiny(&Catalog::for_op(&source, tiny()).unwrap()).unwrap_err();
    assert!(!refused.contains("404"), "{refused}");
    assert_eq!(authkit.exchanges(), 1);

    // A capability that expires ends typed too, before the Hub hears of it.
    authkit.enroll_device_key();
    let brief = authkit.capability(&jkt, &ops, 2);
    let source = Source::new(&authkit.hub, None, hosts, Some(Capability::new(&brief, &authkit.token_endpoint, leaf).unwrap())).unwrap();
    sleep(Duration::from_secs(3));
    assert_eq!(Catalog::for_op(&source, tiny()).err().unwrap().0, reason::EXPIRED);
    assert_eq!(authkit.exchanges(), 1);
}

//! A run's capability against a real AuthKit (th-241): public reads go anonymous; the first
//! private operation trades the owner's device-key-signed capability, once, through the
//! JWT-bearer grant (an assertion the request key signs, DPoP from the same key); every proof is
//! checked by AuthKit's resource verifier, and the Hub's narrowing, a revoked device key and
//! the capability's expiry each end the run's private work with a typed reason.
#[path = "common/hub_oauth.rs"]
mod hub_oauth;

use tensord::hub::{reason, Capability, Catalog, Op, Source};
use serde_json::json;
use std::time::Duration;

fn request_key() -> hub_client::P256Key { hub_client::P256Key::generate() }

const TINY: Op = Op::Publish { model: "acme/tiny" };

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
    match fetch(&url, vec![]) {
        (200, body) => serde_json::from_slice(&body).map_err(|e| e.to_string()),
        (status, body) => Err(format!("HTTP {status}: {}", String::from_utf8_lossy(&body))),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real AuthKit: needs go and AUTHKIT_TEST_DATABASE_URL"]
async fn a_request_key_trades_once_without_the_machine_leaf() {
    let (upstream, hub) = hub_oauth::test_hub();
    let authkit = hub_oauth::AuthKit::start(&upstream);
    let run_key = request_key();
    let jkt = run_key.thumbprint();
    let hosts = vec!["localhost".to_string()];

    // A public-only run: its reads carry nothing, and AuthKit hears of nothing.
    let public = Source::new(&authkit.hub, None, hosts.clone(), None).unwrap();
    let catalog = Catalog::new(&public).unwrap();
    assert_eq!(get(&catalog, "/v1/packages/acme/pkg").unwrap()["package"], "acme/pkg");
    assert_eq!(hub.lock().unwrap().reads, 1);
    assert_eq!(authkit.exchanges(), 0);
    assert!(authkit.verified().is_empty());

    // A run that publishes: the capability names acme/tiny's checkpoints and one read the
    // Hub narrows away. Nothing is traded until the first private operation.
    let narrowed = format!("sha256:{}", "ab".repeat(32));
    let ops = json!([
        {"type": "tensorhub_model_upload", "model": "acme/tiny"},
        {"type": "tensorhub_model_read", "model": "acme/narrowed", "revision": narrowed},
    ]);
    let signed = authkit.capability(&jkt, &ops, 600);
    let source = Source::new(&authkit.hub, None, hosts.clone(), Some(Capability::new(&signed, &authkit.token_endpoint, &run_key.private_jwk()).unwrap())).unwrap();
    assert_eq!(source.runtime_access().key, run_key.private_jwk());
    assert_eq!(source.runtime_access().capability, signed);
    assert_eq!(authkit.exchanges(), 0);
    let tiny = Catalog::for_op(&source, TINY).unwrap();
    assert_eq!(authkit.exchanges(), 0, "a scoped provider does not trade until its first I/O");
    let path = "/refs/acme/tiny.json";
    // The real verifier asks a DPoP nonce; Endpoint retries once, then the resource is absent.
    assert!(tiny.endpoint().unwrap().repo("acme/tiny").await.unwrap().is_none());
    assert!(Catalog::for_op(&source, TINY).unwrap().endpoint().unwrap().repo("acme/tiny").await.unwrap().is_none());
    assert_eq!(authkit.exchanges(), 1, "one trade per run");
    let handoff = Capability::new(&signed, &authkit.token_endpoint, &run_key.private_jwk()).unwrap();
    handoff.restore_token(&authkit.last_token());
    let handed = Source::new(&authkit.hub, None, hosts.clone(), Some(handoff)).unwrap();
    assert!(Catalog::for_op(&handed, TINY).unwrap().endpoint().unwrap().repo("acme/tiny").await.unwrap().is_none());
    assert_eq!(authkit.exchanges(), 1, "a journaled token handoff must not redeem the capability again");
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
    let refused = read.endpoint().unwrap().repo("acme/narrowed").await.unwrap_err();
    assert_eq!(read.reason("catalog_read_failed", &refused).0, reason::EXCEEDED, "{refused:?}");

    // A token without its request key is rejected, rather than treated as an anonymous read.
    let token = authkit.last_token();
    let status = fetch(&format!("{}{path}", authkit.hub), vec![("authorization".into(), format!("DPoP {token}"))]).0;
    assert_eq!(status, 401);

    // Revoking the device key ends the run's private work at its next request, typed, with
    // no second trade.
    authkit.revoke_device_key();
    let tiny = Catalog::for_op(&source, TINY).unwrap();
    let refused = tiny.endpoint().unwrap().repo("acme/tiny").await.unwrap_err();
    assert_eq!(tiny.reason("catalog_read_failed", refused).0, reason::REFUSED);

    assert_eq!(authkit.exchanges(), 1);

    // A capability that expires ends typed too, before the Hub hears of it.
    authkit.enroll_device_key();
    let brief = authkit.capability(&jkt, &ops, 2);
    let source = Source::new(&authkit.hub, None, hosts, Some(Capability::new(&brief, &authkit.token_endpoint, &run_key.private_jwk()).unwrap())).unwrap();
    tokio::time::sleep(Duration::from_secs(3)).await;
    assert_eq!(Catalog::for_op(&source, TINY).err().unwrap().0, reason::EXPIRED);
    assert_eq!(authkit.exchanges(), 1);
}

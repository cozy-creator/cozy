//! Stage beside tests/common as tests/h3_payload_transport_contract.rs in hosted Product25.
//! Existing legacy Hub fixture + current Origin API; no object bodies are requested.
mod common;

use std::sync::Arc;
use std::time::Duration;

use tensorfs::err::Code;
use tensorfs::fetch::{self, Client, Location, Origin, SourcePolicy, TensorhubOrigin};
use tensorfs::format::ObjectRef;

fn object(old: &tensorfs::ids::ObjectRef) -> ObjectRef {
    ObjectRef {
        sha256: old.sha256.clone(),
        length: old.length,
    }
}

#[test]
fn current_origin_resolves_and_presigns_old_manifest_without_reading_payloads() {
    let checkpoint = Arc::new(common::components("acme/old-checkpoint", &[1], 16, 1, &[]));
    let hub = common::hub(
        vec![checkpoint.clone()],
        None,
        Duration::ZERO,
        common::serve_all(),
    );
    let client = Client::new(SourcePolicy {
        allow_local: true,
        ..SourcePolicy::default()
    });
    let origin = TensorhubOrigin::new(&client, &hub.base).unwrap();
    let old_manifest = object(&checkpoint.manifest);
    let chunk = object(&checkpoint.blobs[0]);
    // The origin must not assume its resolved identity is a current-format tree.
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&checkpoint.bodies[&old_manifest.sha256])
            .unwrap()["format"],
        "tensorfs.manifest/1"
    );
    fetch::runtime().unwrap().block_on(async {
        let resolved = origin
            .named(
                &checkpoint.model,
                Some(&checkpoint.release),
                Some(&checkpoint.lane),
            )
            .await
            .unwrap();
        assert_eq!(resolved.tree, old_manifest);
        assert_eq!(resolved.release, checkpoint.release);
        assert_eq!(resolved.variant, checkpoint.lane);
        let locations = origin.locate(std::slice::from_ref(&chunk)).await.unwrap();
        assert_eq!(locations.len(), 1);
        let Location::Url {
            url,
            headers,
            expires,
        } = &locations[0]
        else {
            panic!("expected presigned HTTP location")
        };
        assert_eq!(url, &format!("{}/obj/{}", hub.base, chunk.sha256));
        assert!(headers.is_empty() && expires.is_some());
    });
    assert_eq!(hub.gets("closure"), 1);
    assert_eq!(hub.gets("presign"), 1);
    assert_eq!(
        hub.total_gets(),
        0,
        "metadata preflight fetched a chunk or old header"
    );
}

#[test]
fn wrong_old_manifest_refuses_before_presigning_or_fetching() {
    let checkpoint = Arc::new(common::components("acme/old-checkpoint", &[1], 16, 2, &[]));
    let hub = common::hub(
        vec![checkpoint.clone()],
        None,
        Duration::ZERO,
        common::serve_all(),
    );
    let client = Client::new(SourcePolicy {
        allow_local: true,
        ..SourcePolicy::default()
    });
    let origin = TensorhubOrigin::new(&client, &hub.base).unwrap().at(
        &checkpoint.model,
        Some(&checkpoint.release),
        &checkpoint.lane,
        &ObjectRef::of(b"a different manifest"),
    );
    let error = fetch::runtime()
        .unwrap()
        .block_on(origin.locate(&[object(&checkpoint.blobs[0])]))
        .unwrap_err();
    assert_eq!(error.code, Code::REF_NOT_FOUND);
    assert_eq!(hub.gets("closure"), 1);
    assert_eq!(hub.gets("presign"), 0);
    assert_eq!(hub.total_gets(), 0);
}

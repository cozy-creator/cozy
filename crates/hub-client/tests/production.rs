//! Answers tensorhub.com sent (fetched read-only, 2026-10-16) decode into the generated types.
//! The live Hub predates some fixes the API document holds it to, and the client must still
//! read every answer: `fidika/sdxl@1.0.0` carries `"cut_at": ""`.
use hub_client::types;
use serde::de::DeserializeOwned;

fn decode<T: DeserializeOwned>(name: &str) -> T {
    let path = format!(
        "{}/tests/production/{name}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let raw = std::fs::read(&path).unwrap();
    serde_json::from_slice(&raw).unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
fn production_answers_decode() {
    let card: types::ModelCard = decode("model-fidika-sdxl");
    assert_eq!(card.releases[0].cut_at.as_deref(), Some(""));
    assert!(
        card.releases[0]
            .lanes
            .iter()
            .any(|lane| lane.lane == "int8")
    );
    decode::<types::ModelSearch>("search-models-sdxl");
    decode::<types::PackageSearch>("search-packages-sdxl");
    decode::<types::ModelThroughputs>("model-throughput-fidika-sdxl");
    decode::<types::Software>("software");
    decode::<types::RentalSkuList>("rental-skus");
    decode::<types::CreditDeposit>("credit-deposit");
    decode::<types::AuthProtectedResourceMetadata>("protected-resource");
    decode::<types::AuthServerMetadata>("openid-configuration");
}

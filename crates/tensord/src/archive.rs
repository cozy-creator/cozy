//! The journal's stored values: the domain values themselves as JSON, bytes in base64. Measured
//! on the stored kinds (a model choice, a product, an intake receipt, a terminal page) JSON is
//! 3–5× protobuf's bytes, a few hundred bytes a record.
use crate::api::domain;
use serde::{de::DeserializeOwned, Serialize};

fn encode<T: Serialize>(value: &T) -> Vec<u8> {
    serde_json::to_vec(value).expect("a domain value serializes")
}

fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, serde_json::Error> {
    serde_json::from_slice(bytes)
}

pub fn encode_machine_execution_event_page(value: &domain::MachineExecutionEventPage) -> Vec<u8> {
    encode(value)
}

pub fn decode_machine_execution_event_page(
    bytes: &[u8],
) -> Result<domain::MachineExecutionEventPage, serde_json::Error> {
    decode(bytes)
}

pub fn encode_model_choice(value: &domain::ModelChoice) -> Vec<u8> {
    encode(value)
}

pub fn encode_native_byte_retention_request(value: &domain::NativeByteRetentionRequest) -> Vec<u8> {
    encode(value)
}

pub fn encode_native_byte_retention_result(value: &domain::NativeByteRetentionResult) -> Vec<u8> {
    encode(value)
}

pub fn decode_native_byte_retention_result(
    bytes: &[u8],
) -> Result<domain::NativeByteRetentionResult, serde_json::Error> {
    decode(bytes)
}

pub fn encode_run_product(value: &domain::RunProduct) -> Vec<u8> {
    encode(value)
}

pub fn decode_run_product(bytes: &[u8]) -> Result<domain::RunProduct, serde_json::Error> {
    decode(bytes)
}

pub fn encode_attempt_outcome(value: &domain::AttemptOutcome) -> Vec<u8> {
    encode(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stored values read back as written, their bytes exact; a field a newer machine added is
    /// ignored and one an older machine never wrote reads as its default.
    #[test]
    fn stored_values_round_trip_and_tolerate_other_layouts() {
        let digest = domain::Ref { digest: vec![0, 1, 2, 255], length: 7 };
        let source = domain::NativeByteRetentionRequest {
            source: Some(domain::NativeByteTreeRef {
                producer_root_id: "producer".into(),
                receipt_digest: vec![4, 5],
                manifest: Some(digest.clone()),
                content_bytes: 8,
            }),
            retention_id: "retention".into(),
        };
        let product = domain::RunProduct {
            output: "files".into(),
            checkpoint: None,
            op: domain::RunProductOp::Append as i32,
            index: 2,
            content: Some(digest.clone()),
            media_type: "application/octet-stream".into(),
            label: "kept".into(),
            source: Some(source.clone()),
            parts: vec![domain::RunProductPart { content: Some(digest), source: None, duration_us: 3 }],
        };
        assert_eq!(decode_run_product(&encode_run_product(&product)).unwrap(), product);
        let page = domain::MachineExecutionEventPage {
            events: vec![domain::MachineExecutionEvent {
                sequence: 4,
                kind: "outcome".into(),
                outcome: Some(domain::AttemptOutcome {
                    request_id: "run".into(),
                    outcome_canonical_bytes: br#"{"status":4}"#.to_vec(),
                    ..Default::default()
                }),
                product: Some(product),
                ..Default::default()
            }],
            head_sequence: 4,
            ..Default::default()
        };
        let bytes = encode_machine_execution_event_page(&page);
        assert_eq!(decode_machine_execution_event_page(&bytes).unwrap(), page);
        assert!(String::from_utf8(bytes).unwrap().contains(r#""outcome_canonical_bytes":"eyJzdGF0dXMiOjR9""#));
        let result = decode_native_byte_retention_result(br#"{"retention_id":"r","from_newer":1}"#).unwrap();
        assert_eq!((result.retention_id.as_str(), result.released, result.source), ("r", false, None));
    }
}

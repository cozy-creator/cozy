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

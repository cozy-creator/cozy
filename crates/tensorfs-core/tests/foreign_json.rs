//! Foreign carrier JSON is read by RFC 8259: publisher metadata TensorFS does not consume
//! never refuses a carrier.
use std::{
    fs,
    path::{Path, PathBuf},
};
use tensorfs_core::{
    err::Code,
    ingest::{carrier, fingerprint},
};

fn root(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "tensorfs-foreign-json-{name}-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ));
    fs::create_dir_all(&p).unwrap();
    p
}

/// A safetensors file whose header is exactly `header` (UTF-8 JSON as its writer spelled it).
fn safetensors(path: &Path, header: &str, data: &[u8]) {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(data);
    fs::write(path, bytes).unwrap();
}

// What kohya/Civitai and `safetensors.torch.save_file` really write: raw UTF-8, `\n` and
// `\uXXXX` escapes, numbers with fractions, `null`, nested objects.
const KOHYA_METADATA: &str = r#""__metadata__": {
    "ss_tag_frequency": "{\"img\": {\"少女\": 3, \"caf\u00e9\": 1}}",
    "ss_training_comment": "line one\nline two",
    "ss_learning_rate": 0.0001,
    "ss_network_alpha": 16,
    "ss_seed": null,
    "modelspec.architecture": "stable-diffusion-xl-v1-base/lora",
    "nested": {"a": [1, 2.5]}
  }"#;

#[test]
fn publisher_metadata_in_any_json_spelling_reads() {
    let dir = root("kohya");
    let path = dir.join("lora.safetensors");
    let header = format!(
        "{{\n  {KOHYA_METADATA},\n  \"lora_down.weight\": {{\"dtype\": \"F32\", \"shape\": [2, 2], \"data_offsets\": [0, 16]}}\n}}"
    );
    safetensors(&path, &header, &[0u8; 16]);

    let h = carrier::read_header(&path).unwrap();
    assert_eq!(h.tensors.len(), 1);
    assert_eq!(h.tensors[0].key, "lora_down.weight");
    assert_eq!(h.tensors[0].shape, vec![2, 2]);
    assert_eq!(h.meta("ss_training_comment"), Some("line one\nline two"));
    assert_eq!(h.meta("ss_network_alpha"), Some("16"));
    assert!(h.meta("ss_tag_frequency").unwrap().contains("少女"));
    // Values TensorFS cannot carry as text are dropped, not refused.
    assert_eq!(h.meta("ss_learning_rate"), None);
    assert_eq!(h.meta("ss_seed"), None);
    assert_eq!(h.meta("nested"), None);
    assert_eq!(
        fingerprint::evidence_stamps(&h).unwrap(),
        vec![(
            "modelspec.architecture".to_string(),
            "stable-diffusion-xl-v1-base/lora".to_string()
        )]
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn a_tensor_key_cozytensors_cannot_name_still_refuses() {
    let dir = root("key");
    let path = dir.join("model.safetensors");
    safetensors(
        &path,
        r#"{"blocks.0.tö_q.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}"#,
        &[0u8; 4],
    );
    assert_eq!(
        carrier::read_header(&path).unwrap_err().code,
        Code::KEY_GRAMMAR
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn an_index_with_extra_fields_and_per_shard_metadata_reads() {
    let dir = root("index");
    for (name, key, format) in [
        ("model-00001-of-00002.safetensors", "a.weight", "pt"),
        (
            "model-00002-of-00002.safetensors",
            "b.weight",
            "pt; shard 2",
        ),
    ] {
        safetensors(
            &dir.join(name),
            &format!(
                r#"{{"__metadata__": {{"format": "{format}"}}, "{key}": {{"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}}}"#
            ),
            &[0u8; 4],
        );
    }
    let index = dir.join("model.safetensors.index.json");
    fs::write(
        &index,
        r#"{
  "metadata": {"total_size": 8, "total_parameters": 2.0, "note": null},
  "generator": {"name": "transformers", "version": "4.57.0"},
  "weight_map": {
    "a.weight": "model-00001-of-00002.safetensors",
    "b.weight": "model-00002-of-00002.safetensors"
  }
}"#,
    )
    .unwrap();
    let (h, _) = carrier::read_sharded(&index, carrier::Shards::Siblings).unwrap();
    assert_eq!(h.tensors.len(), 2);
    assert_eq!(h.meta("total_size"), Some("8"));
    let formats: Vec<&str> = h
        .metadata
        .iter()
        .filter(|(k, _)| k == "format")
        .map(|(_, v)| v.as_str())
        .collect();
    assert_eq!(formats, vec!["pt", "pt; shard 2"]);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn a_comfy_quant_marker_with_fields_this_build_does_not_read_decodes() {
    let marker = br#"{"format": "float8_e4m3fn", "full_precision_matrix_mult": true, "scale": 0.5, "note": "caf\u00e9"}"#;
    let mut padded = marker.to_vec();
    padded.resize(marker.len() + 5, 0);
    let decoded = fingerprint::decode_comfy_quant("x.comfy_quant", &padded).unwrap();
    assert_eq!(decoded.format, "float8_e4m3fn");
    assert!(decoded.full_precision_matrix_mult);
}

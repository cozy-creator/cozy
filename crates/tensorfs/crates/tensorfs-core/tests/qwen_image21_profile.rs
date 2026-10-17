//! Real header-only source evidence: no tensor payload is downloaded or read.
use std::{fs, path::PathBuf};
use tensorfs_core::{
    canon::{self, Value},
    dtype::Dtype,
    err::Code,
    ids::ObjectRef,
    ingest::{
        fingerprint, preflight,
        source::{self, BUILTIN_REGISTRY, BUILTIN_REGISTRY_BYTES},
    },
    limits,
    providers::MemberHead,
};

const PROFILE: &str = "hf/qwen/qwen-image-2.1/original/1";
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vectors/qwen-image21-headers")
}
fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    let Value::Obj(rows) = value else {
        panic!("object")
    };
    &rows.iter().find(|(key, _)| key == name).unwrap().1
}
fn text(value: &Value) -> &str {
    let Value::Str(value) = value else {
        panic!("string")
    };
    value
}
fn number(value: &Value) -> u64 {
    let Value::Int(value) = value else {
        panic!("integer")
    };
    (*value).try_into().unwrap()
}
fn heads() -> Vec<MemberHead> {
    let manifest = canon::parse(
        &fs::read(root().join("MANIFEST.json")).unwrap(),
        limits::DOC_MAX_BYTES,
    )
    .unwrap();
    let Value::Arr(rows) = field(&manifest, "members") else {
        panic!("members")
    };
    rows.iter()
        .map(|row| {
            let head = fs::read(root().join(text(field(row, "header_file")))).unwrap();
            assert_eq!(
                ObjectRef::of(&head).sha256,
                text(field(row, "header_sha256"))
            );
            assert_eq!(head.len() as u64, number(field(row, "header_bytes")));
            MemberHead {
                member: text(field(row, "member")).into(),
                length: number(field(row, "full_length")),
                head,
            }
        })
        .collect()
}
fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "qwen21-{name}-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ))
}

#[test]
fn qwen_image21_source_matches_measured_constructor_without_payloads() {
    let heads = heads();
    let path = scratch("plan");
    let decision = preflight::plan(
        BUILTIN_REGISTRY,
        BUILTIN_REGISTRY_BYTES,
        &heads,
        &[PROFILE.into()],
        &path,
    )
    .unwrap();
    assert!(!path.exists());
    assert_eq!(decision.plans.len(), 1);
    assert_eq!(decision.plans[0].constructs, 1285);
    assert_eq!(decision.plans[0].components.len(), 3);
    assert_eq!(decision.plans[0].converter, "diffusers.identity/1");
    let staged = preflight::stage(&path, &heads).unwrap();
    let native = source::plan_source(
        BUILTIN_REGISTRY,
        BUILTIN_REGISTRY_BYTES,
        staged.carriers(),
        Some(PROFILE),
        None,
    )
    .unwrap();
    assert_eq!(native.session, decision.plans[0].session);
    let measured = canon::parse(
        &fs::read(root().join("construction_order.json")).unwrap(),
        limits::DOC_MAX_BYTES,
    )
    .unwrap();
    let actual = Value::arr(
        native
            .construction_order
            .iter()
            .map(|(component, key)| Value::arr(vec![Value::str(component), Value::str(key)]))
            .collect(),
    );
    assert_eq!(actual, measured);
    let carriers = source::CarrierSet::of(staged.carriers());
    for (component, member, dtype, count) in [
        (
            "transformer",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            Dtype::Bf16,
            297,
        ),
        (
            "text_encoder",
            "text_encoder/model.safetensors.index.json",
            Dtype::Bf16,
            750,
        ),
        (
            "vae",
            "vae/diffusion_pytorch_model.safetensors",
            Dtype::F32,
            238,
        ),
    ] {
        let carrier = staged
            .carriers()
            .iter()
            .find(|row| row.member.as_deref() == Some(member))
            .unwrap();
        let (header, _) = source::read_carrier(carrier, &carriers).unwrap();
        let observed = fingerprint::fingerprint(component, &header).unwrap();
        assert_eq!(observed.dtypes, vec![(dtype, count)]);
    }
    drop(staged);
    assert!(!path.exists());
}

#[test]
fn qwen_image21_refuses_missing_shard_before_payload_transfer() {
    let mut heads = heads();
    heads.retain(|row| row.member != "text_encoder/model-00004-of-00004.safetensors");
    let path = scratch("missing");
    let error = preflight::plan(
        BUILTIN_REGISTRY,
        BUILTIN_REGISTRY_BYTES,
        &heads,
        &[PROFILE.into()],
        &path,
    )
    .unwrap_err();
    assert_eq!(error.code, Code::MISSING_FIELD);
    assert!(!path.exists());
}

#[test]
fn qwen_image21_same_keys_at_another_precision_classify_by_key_set() {
    let mut heads = heads();
    for row in &mut heads {
        if row.member.starts_with("transformer/") && row.member.ends_with(".safetensors") {
            let mut header = canon::parse(&row.head[8..], limits::DOC_MAX_BYTES).unwrap();
            let Value::Obj(tensors) = &mut header else {
                panic!("header")
            };
            for (_, value) in tensors {
                if let Value::Obj(fields) = value {
                    if let Some((_, dtype)) = fields.iter_mut().find(|(name, _)| name == "dtype") {
                        *dtype = Value::str("F16");
                    }
                }
            }
            let mut encoded = canon::write(&header);
            // Preserve original carrier/header lengths and source offsets.
            encoded.resize(row.head.len() - 8, b' ');
            row.head[8..].copy_from_slice(&encoded);
        }
    }
    // Owner ruling: the reviewed key set and component classify; conversion validates dtypes.
    let path = scratch("precision");
    let decision = preflight::plan(
        BUILTIN_REGISTRY,
        BUILTIN_REGISTRY_BYTES,
        &heads,
        &[PROFILE.into()],
        &path,
    )
    .unwrap();
    assert_eq!(decision.plans[0].constructs, 1285);
    assert!(!path.exists());
}

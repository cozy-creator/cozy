//! Real header-only evidence for the Hugging Face Diffusers release of Anima Base v1.0: its four
//! components classify against the entries already banked from Civitai 2945208, so the source
//! profile is the only new data. No tensor payload is downloaded or read.
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

const PROFILE: &str = "hf/circlestone-labs/anima-base-v1.0-diffusers/bf16/1";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vectors/anima-hf-headers")
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
        "anima-hf-{name}-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ))
}

#[test]
fn anima_hf_diffusers_plans_against_the_banked_keysets_without_payloads() {
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
    assert_eq!(decision.plans[0].constructs, 1189);
    assert_eq!(decision.plans[0].components.len(), 4);
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
    let recorded = canon::parse(
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
    assert_eq!(actual, recorded);

    let registry = fingerprint::FingerprintRegistry::parse(BUILTIN_REGISTRY_BYTES).unwrap();
    let carriers = source::CarrierSet::of(staged.carriers());
    for (component, member, count) in [
        (
            "transformer",
            "transformer/diffusion_pytorch_model.safetensors",
            567,
        ),
        ("text_encoder", "text_encoder/model.safetensors", 310),
        (
            "text_conditioner",
            "text_conditioner/diffusion_pytorch_model.safetensors",
            118,
        ),
        ("vae", "vae/diffusion_pytorch_model.safetensors", 194),
    ] {
        let carrier = staged
            .carriers()
            .iter()
            .find(|row| row.member.as_deref() == Some(member))
            .unwrap();
        let (header, _) = source::read_carrier(carrier, &carriers).unwrap();
        let observed = fingerprint::fingerprint(component, &header).unwrap();
        assert_eq!(observed.dtypes, vec![(Dtype::Bf16, count)]);
        let banked = registry.authorize(component, &observed).unwrap();
        assert!(banked.provenance.source.contains("2945208"), "{component}");
    }
    drop(staged);
    assert!(!path.exists());
}

#[test]
fn anima_hf_refuses_a_missing_component_before_payload_transfer() {
    let mut heads = heads();
    heads.retain(|row| row.member != "vae/diffusion_pytorch_model.safetensors");
    let path = scratch("missing");
    let error = preflight::plan(
        BUILTIN_REGISTRY,
        BUILTIN_REGISTRY_BYTES,
        &heads,
        &[PROFILE.into()],
        &path,
    )
    .unwrap_err();
    assert_ne!(error.code, Code::IO_FAILED, "{error:?}");
    assert!(!path.exists());
}

#[test]
fn anima_hf_selects_its_profile_without_being_named() {
    let path = scratch("select");
    assert_eq!(
        preflight::select_profile(BUILTIN_REGISTRY, BUILTIN_REGISTRY_BYTES, &heads(), &path)
            .unwrap(),
        PROFILE
    );
    assert!(!path.exists());
}

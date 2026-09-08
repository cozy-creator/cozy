//! Real instrument metadata plans through the production native source path.
//! The only file data staged here is a header/index; sparse payload holes are never read.

use std::{fs, path::PathBuf};
use tensorfs_core::{
    canon::{self, Value},
    err::Code,
    ids::ObjectRef,
    ingest::{
        fingerprint::{self, FingerprintRegistry},
        preflight,
        source::{self, BUILTIN_REGISTRY, BUILTIN_REGISTRY_BYTES},
    },
    limits,
    providers::MemberHead,
};

struct Fixture {
    name: String,
    component: String,
    profile: String,
    carrier: String,
    keys: usize,
    order_digest: String,
    heads: Vec<MemberHead>,
}

fn field<'a>(value: &'a Value, key: &str) -> &'a Value {
    let Value::Obj(fields) = value else {
        panic!("fixture object")
    };
    &fields.iter().find(|(name, _)| name == key).unwrap().1
}
fn text(value: &Value, key: &str) -> String {
    let Value::Str(value) = field(value, key) else {
        panic!("fixture string")
    };
    value.clone()
}
fn count(value: &Value, key: &str) -> u64 {
    let Value::Int(value) = field(value, key) else {
        panic!("fixture integer")
    };
    (*value).try_into().unwrap()
}
fn fixtures() -> Vec<Fixture> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vectors/instrument-headers");
    let Value::Arr(rows) = canon::parse(
        &fs::read(root.join("MANIFEST.json")).unwrap(),
        limits::DOC_MAX_BYTES,
    )
    .unwrap() else {
        panic!("fixture array")
    };
    rows.iter()
        .map(|row| {
            let Value::Arr(members) = field(row, "members") else {
                panic!("members array")
            };
            let heads = members
                .iter()
                .map(|member| {
                    let head = fs::read(root.join(text(member, "header_file"))).unwrap();
                    assert_eq!(ObjectRef::of(&head).sha256, text(member, "header_sha256"));
                    assert_eq!(head.len() as u64, count(member, "header_bytes"));
                    MemberHead {
                        member: text(member, "member"),
                        length: count(member, "full_length"),
                        head,
                    }
                })
                .collect();
            Fixture {
                name: text(row, "name"),
                component: text(row, "component"),
                profile: text(row, "profile"),
                carrier: text(row, "carrier"),
                keys: count(row, "logical_keys") as usize,
                order_digest: text(row, "construction_order_sha256"),
                heads,
            }
        })
        .collect()
}
fn temporary(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tfs-instrument-{name}-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ))
}

#[test]
fn real_instrument_headers_match_native_profiles_without_payloads() {
    let registry = FingerprintRegistry::parse(BUILTIN_REGISTRY_BYTES).unwrap();
    for fixture in fixtures() {
        let scratch = temporary(&fixture.name);
        let decision = preflight::plan(
            BUILTIN_REGISTRY,
            BUILTIN_REGISTRY_BYTES,
            &fixture.heads,
            std::slice::from_ref(&fixture.profile),
            &scratch,
        )
        .unwrap();
        assert!(!scratch.exists());
        assert_eq!(decision.plans.len(), 1);
        let plan = &decision.plans[0];
        assert_eq!(plan.constructs, fixture.keys);
        assert_eq!(plan.converter, "diffusers.identity/1");
        assert_eq!(plan.target, format!("{}=plain/1", fixture.component));
        assert_eq!(plan.components.len(), 1);
        assert_eq!(plan.components[0].component, fixture.component);
        assert_eq!(
            plan.components[0].member.as_deref(),
            Some(fixture.carrier.as_str())
        );
        assert!(!plan.components[0].projected);

        let staged = preflight::stage(&scratch, &fixture.heads).unwrap();
        let native = source::plan_source(
            BUILTIN_REGISTRY,
            BUILTIN_REGISTRY_BYTES,
            staged.carriers(),
            Some(&fixture.profile),
            None,
        )
        .unwrap();
        assert_eq!(
            native.session, plan.session,
            "preflight and native plan are one operation"
        );
        assert!(native
            .construction_order
            .iter()
            .all(|(component, _)| component == &fixture.component));
        let order = Value::arr(
            native
                .construction_order
                .iter()
                .map(|(_, key)| Value::str(key.clone()))
                .collect(),
        );
        assert_eq!(
            ObjectRef::of(&canon::write(&order)).sha256,
            fixture.order_digest
        );
        let carrier = staged
            .carriers()
            .iter()
            .find(|input| input.member.as_deref() == Some(fixture.carrier.as_str()))
            .unwrap();
        let set = source::CarrierSet::of(staged.carriers());
        let (header, _) = source::read_carrier(carrier, &set).unwrap();
        let observed = fingerprint::fingerprint(&fixture.component, &header).unwrap();
        let entry = registry
            .entries
            .iter()
            .find(|entry| {
                entry.component == fixture.component
                    && entry.keyset_digest == observed.keyset_digest
            })
            .unwrap();
        assert_eq!(entry.tensor_schema_digest, observed.tensor_schema_digest);
        assert_eq!(observed.logical_keys, fixture.keys);
        drop(staged);
        assert!(!scratch.exists());
    }
}

#[test]
fn profiles_refuse_a_foreign_model_or_missing_member() {
    let all = fixtures();
    let qwen2 = all.iter().find(|fixture| fixture.name == "qwen2b").unwrap();
    let qwen8 = all.iter().find(|fixture| fixture.name == "qwen8b").unwrap();
    let error = preflight::plan(
        BUILTIN_REGISTRY,
        BUILTIN_REGISTRY_BYTES,
        &qwen2.heads,
        std::slice::from_ref(&qwen8.profile),
        &temporary("wrong-model"),
    )
    .unwrap_err();
    assert_eq!(error.code, Code::UNREGISTERED_FINGERPRINT);
    let absent = "model-00001-of-00004.safetensors";
    let heads: Vec<_> = qwen8
        .heads
        .iter()
        .filter(|head| head.member != absent)
        .cloned()
        .collect();
    let error = preflight::plan(
        BUILTIN_REGISTRY,
        BUILTIN_REGISTRY_BYTES,
        &heads,
        std::slice::from_ref(&qwen8.profile),
        &temporary("missing-shard"),
    )
    .unwrap_err();
    assert_eq!(error.code, Code::MISSING_FIELD);
    assert!(error.detail.contains(absent));
    let mut renamed = qwen2.heads.clone();
    renamed[0].member = "other.safetensors".into();
    let error = preflight::plan(
        BUILTIN_REGISTRY,
        BUILTIN_REGISTRY_BYTES,
        &renamed,
        std::slice::from_ref(&qwen2.profile),
        &temporary("wrong-member"),
    )
    .unwrap_err();
    assert_eq!(error.code, Code::UNREGISTERED_FINGERPRINT);
}

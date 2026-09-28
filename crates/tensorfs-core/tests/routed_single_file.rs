//! A routed single-file converter: one ingest pass from a single-file carrier to the target
//! pipeline's keys, with row-block splits, a transpose, a squeeze and f16 narrowing.
use std::{
    fs,
    path::{Path, PathBuf},
};
use tensorfs_core::{
    canon::{self, Value},
    catalog::Catalog,
    err::Code,
    header::Body,
    ingest::{
        carrier,
        convert::{self, Plan},
        routes::{self, Component, Kind, Route, Table},
        transaction,
    },
    registry,
    spec::EncodingSpec,
    store::Store,
};

fn root(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "tensorfs-routed-{name}-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ));
    fs::create_dir_all(&p).unwrap();
    p
}

fn specs() -> Vec<(String, EncodingSpec)> {
    registry::seeds()
        .into_iter()
        .filter(|s| s.alias == "plain/1")
        .map(|s| (s.alias.to_string(), s.spec))
        .collect()
}

const PREFIX: &str = "conditioner.embedders.1.model.";

/// `(key, dtype, shape, payload)` rows as one safetensors carrier.
fn carrier_file(dir: &Path, rows: &[(&str, &str, Vec<u64>, Vec<u8>)]) -> transaction::SourceFile {
    let mut tensors = Vec::new();
    let mut payload = Vec::new();
    for (key, dtype, shape, bytes) in rows {
        let begin = payload.len() as u64;
        payload.extend_from_slice(bytes);
        tensors.push((
            format!("{PREFIX}{key}"),
            Value::obj(vec![
                ("dtype", Value::str(*dtype)),
                (
                    "shape",
                    Value::arr(shape.iter().copied().map(Value::uint).collect()),
                ),
                (
                    "data_offsets",
                    Value::arr(vec![Value::uint(begin), Value::uint(payload.len() as u64)]),
                ),
            ]),
        ));
    }
    let header = canon::write(&Value::map(tensors));
    let path = dir.join("model.safetensors");
    let mut data = (header.len() as u64).to_le_bytes().to_vec();
    data.extend(header);
    data.extend(payload);
    fs::write(&path, data).unwrap();
    let header = carrier::read_header(&path).unwrap();
    transaction::SourceFile { path, header }
}

fn f32s(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}

fn f16s(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    // Every value here is exactly representable, so narrowing is exact.
    values
        .into_iter()
        .flat_map(|v| {
            let bits = v.to_bits();
            let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
            let half = if v == 0.0 {
                0u16
            } else {
                ((bits >> 16) & 0x8000) as u16
                    | ((exp as u16) << 10)
                    | ((bits >> 13) & 0x3ff) as u16
            };
            half.to_le_bytes()
        })
        .collect()
}

fn route(target: &str, source: &str, kind: Kind) -> Route {
    Route {
        target: target.into(),
        source: source.into(),
        kind,
    }
}

fn table() -> Table {
    let split = |block| Kind::Split { block, blocks: 3 };
    Table {
        converter: routes::SDXL.into(),
        reference: String::new(),
        components: vec![Component {
            name: "text_encoder_2".into(),
            prefix: PREFIX.into(),
            drop: vec!["logit_scale".into()],
            optional: vec!["position_ids".into()],
            routes: vec![
                route("q.weight", "in_proj_weight", split(0)),
                route("k.weight", "in_proj_weight", split(1)),
                route("v.weight", "in_proj_weight", split(2)),
                route("v.bias", "in_proj_bias", split(2)),
                route("projection.weight", "text_projection", Kind::Transpose),
                route("attention.weight", "conv.weight", Kind::Squeeze),
                route("norm.weight", "ln.weight", Kind::Rekey),
            ],
        }],
    }
}

fn plan(file: &transaction::SourceFile) -> tensorfs_core::err::Result<Plan> {
    let conv = convert::converter(routes::SDXL)?;
    let plain = specs().remove(0).1;
    let mut plan = Plan::default();
    plan.converter = conv.name.to_string();
    plan.class_name = conv.class.name().to_string();
    routes::plan(
        conv,
        &table(),
        "text_encoder_2",
        0,
        &file.header,
        &["plain/1"],
        &plain,
        &mut plan,
    )?;
    let order: Vec<_> = plan
        .ops
        .iter()
        .map(|o| (o.component.clone(), o.out_key.clone()))
        .collect();
    plan.apply_order(&order)?;
    Ok(plan)
}

fn source(in_proj: Vec<f32>, extra: bool) -> Vec<(&'static str, &'static str, Vec<u64>, Vec<u8>)> {
    let bf16 = |values: &[f32]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    };
    let mut rows = vec![
        ("in_proj_weight", "F32", vec![6, 2], f32s(in_proj)),
        (
            "in_proj_bias",
            "F32",
            vec![6],
            f32s((0..6).map(|i| i as f32 * 0.5)),
        ),
        (
            "text_projection",
            "F32",
            vec![2, 3],
            f32s([1., 2., 3., 4., 5., 6.]),
        ),
        (
            "conv.weight",
            "BF16",
            vec![2, 2, 1, 1],
            bf16(&[1., -2., 0.5, 8.]),
        ),
        ("ln.weight", "F16", vec![2], f16s([0.25, -1.])),
        ("logit_scale", "F32", vec![], f32s([4.6])),
    ];
    if extra {
        rows.push(("unreviewed", "F32", vec![1], f32s([0.])));
    }
    rows
}

#[test]
fn one_pass_splits_transposes_squeezes_and_narrows_to_the_canonical_keys() {
    let dir = root("convert");
    let file = carrier_file(&dir, &source((1..=12).map(|i| i as f32).collect(), false));
    let plan = plan(&file).unwrap();
    let store = Store::init(&dir.join("store")).unwrap();
    let catalog = Catalog::open(store.root()).unwrap();
    let _writer = catalog
        .resume_operation("convert", "proof", "routed")
        .unwrap();
    let files = [file];
    let run = || {
        transaction::advance(
            &store,
            &plan,
            &files,
            &specs(),
            &[],
            "convert",
            transaction::Carriers::All,
            None,
        )
        .unwrap()
    };
    let (progress, outcome) = run();
    assert_eq!(progress.converted_roles, 7);
    let outcome = outcome.unwrap();
    let (component, tensors) = &outcome.header.components[0];
    assert_eq!(component, "text_encoder_2");
    let expected = [
        ("q.weight", vec![2, 2], f16s([1., 2., 3., 4.])),
        ("k.weight", vec![2, 2], f16s([5., 6., 7., 8.])),
        ("v.weight", vec![2, 2], f16s([9., 10., 11., 12.])),
        ("v.bias", vec![2], f16s([2., 2.5])),
        (
            "projection.weight",
            vec![3, 2],
            f16s([1., 4., 2., 5., 3., 6.]),
        ),
        ("attention.weight", vec![2, 2], f16s([1., -2., 0.5, 8.])),
        ("norm.weight", vec![2], f16s([0.25, -1.])),
    ];
    assert_eq!(tensors.len(), expected.len());
    for ((key, tensor), (want_key, shape, bytes)) in tensors.iter().zip(expected) {
        assert_eq!(key, want_key);
        assert_eq!(tensor.dtype.name(), "f16", "{key}");
        assert_eq!(tensor.shape, shape, "{key}");
        assert_eq!(tensor.parts[0].1.body, Body::Inline(bytes), "{key}");
    }
    let (name, record) = &outcome.header.configs[0];
    assert_eq!(name, "normalization");
    assert_eq!(
        String::from_utf8(record.clone()).unwrap(),
        format!(
            "{{\"converter\":\"{}\",\"dialect\":\"diffusers\",\"reference\":\"{}\",\"state\":\"normalized\"}}",
            routes::SDXL,
            routes::table(routes::SDXL).unwrap().reference
        )
    );
    let (again, replay) = run();
    assert_eq!((again.resumed_roles, again.written), (7, 0));
    assert_eq!(replay.unwrap().manifest_ref, outcome.manifest_ref);
    drop((_writer, catalog, store));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_value_f16_cannot_hold_refuses_and_an_unreviewed_layout_never_plans() {
    let dir = root("refuse");
    let mut in_proj: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    in_proj[7] = 1.0e6;
    let file = carrier_file(&dir, &source(in_proj, false));
    let plan = plan(&file).unwrap();
    let store = Store::init(&dir.join("store")).unwrap();
    let catalog = Catalog::open(store.root()).unwrap();
    let _writer = catalog
        .resume_operation("convert", "proof", "routed")
        .unwrap();
    let files = [file];
    let refusal = transaction::advance(
        &store,
        &plan,
        &files,
        &specs(),
        &[],
        "convert",
        transaction::Carriers::All,
        None,
    )
    .unwrap_err();
    assert_eq!(refusal.code, Code::NUMBER_RANGE, "{}", refusal.detail);
    assert!(refusal.detail.contains("k.weight"), "{}", refusal.detail);

    let other = root("unreviewed");
    let extra = carrier_file(&other, &source((1..=12).map(|i| i as f32).collect(), true));
    assert_eq!(plan_code(&extra), Code::MISSING_TENSOR);
    let mut missing = source((1..=12).map(|i| i as f32).collect(), false);
    missing.retain(|row| row.0 != "logit_scale");
    let missing = carrier_file(&other, &missing);
    assert_eq!(plan_code(&missing), Code::MISSING_TENSOR);
    drop((_writer, catalog, store));
    fs::remove_dir_all(dir).unwrap();
    fs::remove_dir_all(other).unwrap();
}

fn plan_code(file: &transaction::SourceFile) -> Code {
    plan(file).unwrap_err().code
}

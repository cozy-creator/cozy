//! Actual carrier -> native plan -> streamed conversion -> resumed artifact regressions.
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};
use tensorfs_core::{
    canon::{self, Value},
    catalog::Catalog,
    err::Code,
    header::Body,
    ingest::{
        carrier,
        convert::{self, Bytes, Source, Target},
        journal, transaction,
    },
    registry,
    spec::EncodingSpec,
    store::Store,
};

fn root() -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "tensorfs-swiglu-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&p).unwrap();
    p
}
fn carrier_file(
    root: &std::path::Path,
    prefix: &str,
    rows: u64,
    first: bool,
    second: bool,
    bias: bool,
) -> transaction::SourceFile {
    let mut tensors = Vec::new();
    let mut payload = Vec::new();
    for (leaf, shape, enabled) in [
        ("fc1.weight", vec![rows, 2], first),
        ("fc2.weight", vec![2, 2], second),
        ("fc1.bias", vec![rows], bias),
    ] {
        if !enabled {
            continue;
        }
        let begin = payload.len();
        for n in 0..shape.iter().product::<u64>() {
            payload.extend_from_slice(&(n as f32 + 1.0).to_le_bytes());
        }
        tensors.push((
            format!("{prefix}.mlp.{leaf}"),
            Value::obj(vec![
                ("dtype", Value::str("F32")),
                (
                    "shape",
                    Value::arr(shape.into_iter().map(Value::uint).collect()),
                ),
                (
                    "data_offsets",
                    Value::arr(vec![
                        Value::uint(begin as u64),
                        Value::uint(payload.len() as u64),
                    ]),
                ),
            ]),
        ));
    }
    let header = canon::write(&Value::map(tensors));
    let path = root.join("input.safetensors");
    let mut data = (header.len() as u64).to_le_bytes().to_vec();
    data.extend(header);
    data.extend(payload);
    fs::write(&path, data).unwrap();
    let header = carrier::read_header(&path).unwrap();
    transaction::SourceFile { path, header }
}
fn specs() -> Vec<(String, EncodingSpec)> {
    registry::seeds()
        .into_iter()
        .filter(|s| s.alias == "plain/1")
        .map(|s| (s.alias.to_string(), s.spec))
        .collect()
}
fn plan(
    file: &transaction::SourceFile,
    component: &str,
) -> tensorfs_core::err::Result<convert::Plan> {
    let mut plan = convert::plan(
        convert::converter("h3.native/2")?,
        &[Source::Carrier {
            component: component.to_string(),
            file: 0,
            header: &file.header,
        }],
        &Target {
            components: vec![(component.to_string(), "plain/1".to_string())],
        },
        &specs(),
        &Default::default(),
    )?;
    let order = plan
        .ops
        .iter()
        .map(|o| (o.component.clone(), o.out_key.clone()))
        .collect::<Vec<_>>();
    plan.apply_order(&order)?;
    Ok(plan)
}
fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn fused_gate_value_rows_convert_and_resume_for_both_dits_and_refiners() {
    for (component, prefix) in [
        ("fl2va_dit", "blocks.0"),
        ("ref2va_dit", "blocks.49"),
        ("fl2va_dit", "token_refiner.blocks.0"),
        ("ref2va_dit", "token_refiner.blocks.1"),
    ] {
        let root = root();
        let file = carrier_file(&root, prefix, 4, true, true, true);
        let plan = plan(&file, component).unwrap();
        let store = Store::init(&root.join("store")).unwrap();
        let catalog = Catalog::open(store.root()).unwrap();
        let _writer = catalog
            .resume_operation("convert", "proof", "swiglu")
            .unwrap();
        let files = [file];
        let (progress, outcome) = transaction::advance(
            &store,
            &plan,
            &files,
            &specs(),
            &[],
            "convert",
            transaction::Carriers::All,
            None,
        )
        .unwrap();
        assert_eq!(progress.converted_roles, 3);
        let outcome = outcome.unwrap();
        for (_, tensors) in &outcome.header.components {
            for (key, tensor) in tensors {
                let expected = if key.ends_with("ff.net.0.proj.weight") {
                    floats(&[5., 6., 7., 8., 1., 2., 3., 4.])
                } else if key.ends_with("ff.net.0.proj.bias") {
                    floats(&[3., 4., 1., 2.])
                } else {
                    floats(&[1., 2., 3., 4.])
                };
                assert_eq!(
                    tensor.parts[0].1.body,
                    Body::Inline(expected),
                    "{component}/{key}"
                );
            }
        }
        let (again, replay) = transaction::advance(
            &store,
            &plan,
            &files,
            &specs(),
            &[],
            "convert",
            transaction::Carriers::All,
            None,
        )
        .unwrap();
        assert_eq!(again.resumed_roles, 3);
        assert_eq!(again.written, 0);
        assert_eq!(again.hashed, 0);
        assert_eq!(replay.unwrap().manifest_ref, outcome.manifest_ref);
        drop(_writer);
        drop(catalog);
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn fused_mlp_refuses_odd_rows_or_missing_companions() {
    for (rows, first, second, bias, code) in [
        (3, true, true, false, Code::SHAPE_MISMATCH),
        (4, true, false, false, Code::MISSING_COMPANION_ROLE),
        (4, false, true, true, Code::MISSING_COMPANION_ROLE),
    ] {
        let root = root();
        let source = carrier_file(&root, "blocks.0", rows, first, second, bias);
        assert_eq!(plan(&source, "ref2va_dit").unwrap_err().code, code);
        fs::remove_dir_all(root).unwrap();
    }
    assert!(
        convert::converter("h3.native/1").is_err(),
        "old wrong recipe cannot remain selectable"
    );
}

#[test]
fn old_identity_journal_cannot_satisfy_swapped_plan() {
    let root = root();
    let file = carrier_file(&root, "blocks.0", 4, true, true, false);
    let plan = plan(&file, "ref2va_dit").unwrap();
    let mut old = plan.clone();
    for op in &mut old.ops {
        for role in &mut op.roles {
            if let Bytes::Permute { file, key, .. } = &role.bytes {
                role.bytes = Bytes::Stream {
                    file: *file,
                    key: key.clone(),
                };
            }
        }
    }
    let headers = [(file.path.clone(), &file.header)];
    let old_digest = journal::plan_digest(&old, &headers);
    let new_digest = journal::plan_digest(&plan, &headers);
    assert_ne!(
        old_digest, new_digest,
        "transform geometry itself must bind the journal, even before the recipe version changes"
    );
    old.converter = "h3.native/1".to_string();
    let old_digest = journal::plan_digest(&old, &headers);
    assert_ne!(old_digest, new_digest);
    let store = Store::init(&root.join("store")).unwrap();
    let catalog = Catalog::open(store.root()).unwrap();
    drop(
        catalog
            .begin_source_preparation("same-source", &format!("sha256:{old_digest}"))
            .unwrap(),
    );
    let failure = catalog.begin_source_preparation("same-source", &format!("sha256:{new_digest}"));
    assert!(matches!(failure, Err(ref e) if e.code == Code::TRANSACTION_CONFLICT));
    drop(catalog);
    drop(store);
    fs::remove_dir_all(root).unwrap();
}

//! Byte-level LoRA conversion through the real resumable native ingest writer.
use std::{
    fs,
    path::{Path, PathBuf},
};
use tensorfs_core::{
    canon::{self, Value},
    catalog::Catalog,
    err::Code,
    header::{Body, Part},
    ingest::{
        carrier,
        convert::{self, Bytes, Source, Target},
        journal, transaction,
    },
    registry,
    spec::EncodingSpec,
    store::Store,
};

fn specs() -> Vec<(String, EncodingSpec)> {
    registry::seeds()
        .into_iter()
        .filter(|s| s.alias == "plain/1")
        .map(|s| (s.alias.into(), s.spec))
        .collect()
}
fn write_source(root: &Path, kohya: bool, prefix: &str) -> transaction::SourceFile {
    let mut payload = Vec::new();
    let mut tensors = Vec::new();
    for (native, flat, input, output) in [
        ("attn.qkv_proj", "attn_qkv_proj", 5376, 21504),
        ("attn.out_proj", "attn_out_proj", 7168, 5376),
        ("mlp.fc1", "mlp_fc1", 5376, 28672),
        ("mlp.fc2", "mlp_fc2", 14336, 5376),
    ] {
        let stem = if kohya {
            format!("lora_unet_blocks_0_{flat}")
        } else {
            format!("diffusion_model.{prefix}.{native}")
        };
        let roles = if kohya {
            ["lora_down.weight", "lora_up.weight", "alpha"]
        } else {
            ["lora_A.weight", "lora_B.weight", "alpha"]
        };
        for (role, shape) in roles
            .into_iter()
            .zip([vec![2, input], vec![output, 2], vec![]])
        {
            let start = payload.len();
            for i in 0..shape.iter().product::<u64>() {
                payload.extend_from_slice(&(i as f32 + 1.).to_le_bytes());
            }
            tensors.push((
                format!("{stem}.{role}"),
                Value::obj(vec![
                    ("dtype", Value::str("F32")),
                    (
                        "shape",
                        Value::arr(shape.into_iter().map(Value::uint).collect()),
                    ),
                    (
                        "data_offsets",
                        Value::arr(vec![
                            Value::uint(start as u64),
                            Value::uint(payload.len() as u64),
                        ]),
                    ),
                ]),
            ));
        }
    }
    let header = canon::write(&Value::map(tensors));
    let path = root.join("source.safetensors");
    let mut data = (header.len() as u64).to_le_bytes().to_vec();
    data.extend(header);
    data.extend(payload);
    fs::write(&path, data).unwrap();
    transaction::SourceFile {
        header: carrier::read_header(&path).unwrap(),
        path,
    }
}
fn plan(file: &transaction::SourceFile) -> tensorfs_core::err::Result<convert::Plan> {
    let mut p = convert::plan(
        convert::converter("h3.lora/2")?,
        &[Source::Carrier {
            component: "adapter".into(),
            file: 0,
            header: &file.header,
        }],
        &Target {
            components: vec![("adapter".into(), "plain/1".into())],
        },
        &specs(),
        &Default::default(),
    )?;
    let order = p
        .ops
        .iter()
        .map(|op| (op.component.clone(), op.out_key.clone()))
        .collect::<Vec<_>>();
    p.apply_order(&order)?;
    Ok(p)
}
fn bytes(store: &Store, part: &Part) -> Vec<u8> {
    match &part.body {
        Body::Inline(b) => b.clone(),
        Body::Segments(s) => s
            .iter()
            .flat_map(|r| store.read_range(&r.sha256, 0, r.length).unwrap())
            .collect(),
    }
}
fn root() -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "h3-lora-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ));
    fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn trainer_fused_projection_and_converted_qkv_have_the_same_heads() {
    // Upstream Musubi and Comfy compute B(A(x)) before chunking the output into
    // three contiguous projections. Derive that result independently of Xform
    // and compare every head element after the actual resumable ingest writer.
    for kohya in [false, true] {
        let root = root();
        let file = write_source(&root, kohya, "blocks.0");
        let carrier_bytes = fs::read(&file.path).unwrap();
        let read = |suffix: &str| {
            let tensor = file
                .header
                .tensors
                .iter()
                .find(|tensor| tensor.key.ends_with(suffix))
                .unwrap();
            let start = (file.header.data_start + tensor.begin) as usize;
            let end = (file.header.data_start + tensor.end) as usize;
            carrier_bytes[start..end]
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()) as f64)
                .collect::<Vec<_>>()
        };
        let (a, b) = if kohya {
            (
                read("attn_qkv_proj.lora_down.weight"),
                read("attn_qkv_proj.lora_up.weight"),
            )
        } else {
            (
                read("attn.qkv_proj.lora_A.weight"),
                read("attn.qkv_proj.lora_B.weight"),
            )
        };
        // A sparse but nontrivial input; both rank directions contribute.
        let projected = a
            .chunks_exact(5376)
            .map(|row| row[0] * 0.25 - row[5375] * 0.125)
            .collect::<Vec<_>>();
        let fused = b
            .chunks_exact(2)
            .map(|row| row.iter().zip(&projected).map(|(b, a)| b * a).sum::<f64>())
            .collect::<Vec<_>>();
        let store = Store::init(&root.join("store")).unwrap();
        let catalog = Catalog::open(store.root()).unwrap();
        let writer = catalog
            .resume_operation("projection-proof", "proof", "lora")
            .unwrap();
        let (_, out) = transaction::advance(
            &store,
            &plan(&file).unwrap(),
            &[file],
            &specs(),
            &[],
            "projection-proof",
            transaction::Carriers::All,
            None,
        )
        .unwrap();
        let out = out.unwrap();
        let tensors = &out.header.components[0].1;
        for (part, expected) in ["q", "k", "v"].into_iter().zip(fused.chunks_exact(7168)) {
            let key = format!("transformer_blocks.0.attn.to_{part}.lora_B.weight");
            let tensor = &tensors.iter().find(|(name, _)| name == &key).unwrap().1;
            let converted = bytes(&store, &tensor.parts[0].1);
            for (head, rows) in converted.chunks_exact(128 * 2 * 4).enumerate() {
                for (dim, row) in rows.chunks_exact(2 * 4).enumerate() {
                    let actual = row
                        .chunks_exact(4)
                        .map(|v| f32::from_le_bytes(v.try_into().unwrap()) as f64)
                        .zip(&projected)
                        .map(|(b, a)| b * a)
                        .sum::<f64>();
                    assert_eq!(
                        actual,
                        expected[head * 128 + dim],
                        "{key} head {head}, dim {dim}"
                    );
                }
            }
        }
        drop(writer);
        drop(catalog);
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn both_dialects_permute_only_b_rows_and_resume_without_new_bytes() {
    for (kohya, prefix, out_prefix) in [
        (true, "blocks.0", "transformer_blocks.0"),
        (
            false,
            "token_refiner.blocks.1",
            "token_refiner.refiner_blocks.1",
        ),
    ] {
        let root = root();
        let file = write_source(&root, kohya, prefix);
        let plan = plan(&file).unwrap();
        assert_eq!(plan.ops.len(), 18);
        let mut wrong = plan.clone();
        for op in &mut wrong.ops {
            for role in &mut op.roles {
                if let Bytes::Permute { file, key, .. } = &role.bytes {
                    role.bytes = Bytes::Stream {
                        file: *file,
                        key: key.clone(),
                    };
                }
            }
        }
        assert_ne!(
            journal::plan_digest(&plan, &[(file.path.clone(), &file.header)]),
            journal::plan_digest(&wrong, &[(file.path.clone(), &file.header)])
        );
        let store = Store::init(&root.join("store")).unwrap();
        let catalog = Catalog::open(store.root()).unwrap();
        let writer = catalog
            .resume_operation("convert", "proof", "lora")
            .unwrap();
        let files = [file];
        let (_, out) = transaction::advance(
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
        let out = out.unwrap();
        for (_, tensors) in &out.header.components {
            for (key, tensor) in tensors {
                assert!(key.starts_with(out_prefix));
                let actual = bytes(&store, &tensor.parts[0].1);
                let values = actual
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>();
                for (i, actual) in values.iter().enumerate() {
                    let source_index = if key.ends_with(".lora_B.weight")
                        && (key.contains(".attn.to_q.")
                            || key.contains(".attn.to_k.")
                            || key.contains(".attn.to_v."))
                    {
                        let take = if key.contains(".to_q.") {
                            0
                        } else if key.contains(".to_k.") {
                            1
                        } else {
                            2
                        };
                        let row = i / 2;
                        (take * 7168 + row) * 2 + i % 2
                    } else if key.ends_with(".ff.net.0.proj.lora_B.weight") {
                        (i + 28672) % (28672 * 2)
                    } else {
                        i
                    };
                    assert_eq!(*actual, source_index as f32 + 1., "{key} at {i}");
                }
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
        assert_eq!(again.written, 0);
        assert_eq!(again.hashed, 0);
        assert_eq!(replay.unwrap().manifest_ref, out.manifest_ref);
        drop(writer);
        drop(catalog);
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn bad_pair_shape_unknown_tensor_and_unreviewed_geometry_refuse_before_payload() {
    // /1 encoded the wrong trainer layout. An old plan must refuse rather than
    // silently reinterpret its immutable converter identity under new code.
    assert_eq!(
        convert::converter("h3.lora/1").unwrap_err().code,
        Code::UNKNOWN_FIELD
    );
    let root = root();
    let file = write_source(&root, false, "blocks.0");
    for (kind, want) in [
        ("missing", Code::MISSING_COMPANION_ROLE),
        ("rank", Code::SHAPE_MISMATCH),
        ("unknown", Code::UNKNOWN_FIELD),
        ("geometry", Code::SHAPE_MISMATCH),
    ] {
        let mut header = file.header.clone();
        match kind {
            "missing" => {
                header
                    .tensors
                    .retain(|t| !t.key.ends_with("attn.qkv_proj.lora_B.weight"));
            }
            "rank" => {
                header
                    .tensors
                    .iter_mut()
                    .find(|t| t.key.ends_with("attn.qkv_proj.lora_A.weight"))
                    .unwrap()
                    .shape[0] = 3;
            }
            "unknown" => {
                header.tensors[0].key = "unclaimed.weight".into();
            }
            _ => {
                for t in &mut header.tensors {
                    t.key = t.key.replace("blocks.0.", "blocks.50.");
                }
            }
        }
        let file = transaction::SourceFile {
            path: file.path.clone(),
            header,
        };
        assert_eq!(plan(&file).unwrap_err().code, want, "{kind}");
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn real_headers_match_by_structure_and_multiple_versions_require_file_selection() {
    use tensorfs_core::{
        ingest::{preflight, source},
        providers::MemberHead,
    };
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vectors/h3-lora-headers");
    let rows = canon::parse(&fs::read(fixtures.join("MANIFEST.json")).unwrap(), 1 << 20).unwrap();
    let Value::Arr(rows) = rows else {
        panic!("array")
    };
    let mut heads = Vec::new();
    for row in rows {
        let Value::Obj(fields) = row else {
            panic!("object")
        };
        let get = |key: &str| &fields.iter().find(|(k, _)| k == key).unwrap().1;
        let Value::Str(member) = get("member") else {
            panic!("member")
        };
        let Value::Int(length) = get("full_length") else {
            panic!("length")
        };
        let head = fs::read(fixtures.join(member)).unwrap();
        let candidate = MemberHead {
            member: format!("renamed/{member}"),
            length: *length as u64,
            head,
        };
        let root = root();
        let result = preflight::plan(
            source::BUILTIN_REGISTRY,
            source::BUILTIN_REGISTRY_BYTES,
            std::slice::from_ref(&candidate),
            &[if member.contains("spatial") {
                "h3/native-lora-a-b/2"
            } else {
                "h3/native-lora-kohya/2"
            }
            .into()],
            &root.join("sparse"),
        )
        .unwrap();
        assert_eq!(result.plans.len(), 1);
        assert_eq!(result.plans[0].converter, "h3.lora/2");
        assert_eq!(
            result.plans[0].constructs,
            if member.contains("spatial") { 624 } else { 900 }
        );
        assert!(!root.join("sparse").exists());
        fs::remove_dir_all(root).unwrap();
        heads.push(candidate);
    }
    let root = root();
    let error = preflight::plan(
        source::BUILTIN_REGISTRY,
        source::BUILTIN_REGISTRY_BYTES,
        &heads[..2],
        &["h3/native-lora-a-b/2".into()],
        &root.join("sparse"),
    )
    .unwrap_err();
    assert_eq!(error.code, Code::AMBIGUOUS_CLASSIFICATION);
    assert!(error.detail.contains(&heads[0].member));
    assert!(error.detail.contains(&heads[1].member));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn ambiguous_fused_exporter_needs_a_declared_layout_before_preflight_or_planning() {
    use std::cell::Cell;
    use tensorfs_core::{
        ingest::{preflight, source},
        providers::{MemberHead, Provenance, Resolution, ResolvedMember},
    };
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vectors/h3-lora-headers");
    let Value::Arr(rows) =
        canon::parse(&fs::read(fixtures.join("MANIFEST.json")).unwrap(), 1 << 20).unwrap()
    else {
        panic!("manifest array")
    };
    for row in rows.into_iter().filter(|row| {
        let Value::Obj(fields) = row else { return false };
        matches!(fields.iter().find(|(name, _)| name == "member").map(|(_, value)| value), Some(Value::Str(name)) if name.contains("2000step") || name == "wushu_spatial_physics_clean_3000_pruned.safetensors")
    }) {
        let Value::Obj(fields) = row else { unreachable!() };
        let Value::Str(member) = &fields.iter().find(|(name, _)| name == "member").unwrap().1 else { panic!("member") };
        let Value::Int(full_length) = fields.iter().find(|(name, _)| name == "full_length").unwrap().1 else { panic!("length") };
        let original = fs::read(fixtures.join(member)).unwrap();
        let Value::Obj(mut tensors) = canon::parse(&original[8..], 1 << 20).unwrap() else { panic!("header") };
        tensors.retain(|(key, _)| key != "__metadata__");
        // Contiguous QKV and per-head interleaved QKV have these exact same keys,
        // dtypes and shapes. This unknown exporter provides no layout declaration.
        let header = canon::write(&Value::map(tensors));
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        let length = full_length as u64 - original.len() as u64 + bytes.len() as u64;
        let head = MemberHead { member: "unknown-exporter.safetensors".into(), length, head: bytes.clone() };
        let root = root();
        let refused = preflight::select_profile(source::BUILTIN_REGISTRY, source::BUILTIN_REGISTRY_BYTES, std::slice::from_ref(&head), &root.join("preflight")).unwrap_err();
        assert_eq!(refused.code, Code::AMBIGUOUS_CLASSIFICATION);
        assert!(refused.detail.contains("source layout"));
        assert!(!root.join("preflight").exists());
        let path = root.join("unknown.safetensors");
        fs::write(&path, &bytes).unwrap();
        fs::OpenOptions::new().write(true).open(&path).unwrap().set_len(length).unwrap();
        let carrier = source::CarrierInput { member: Some(head.member.clone()), path };
        assert_eq!(source::plan_source(source::BUILTIN_REGISTRY, source::BUILTIN_REGISTRY_BYTES, std::slice::from_ref(&carrier), None, None).unwrap_err().code, Code::AMBIGUOUS_CLASSIFICATION);
        // Intentional raw preservation remains available; it performs no guessed
        // normalization. Known independently checked /2 imports are tested above.
        assert_eq!(source::plan_source(source::BUILTIN_REGISTRY, source::BUILTIN_REGISTRY_BYTES, std::slice::from_ref(&carrier), Some(source::AS_IS), None).unwrap().converter, convert::IDENTITY);
        let resolution = Resolution { canonical: "hf://unknown/exporter@revision".into(), selection_sha256: "11".repeat(32), members: vec![ResolvedMember { member: head.member.clone(), object: tensorfs_core::ids::ObjectRef { sha256: "22".repeat(32), length }, url: "https://example.invalid/model".into(), provenance: Provenance::Declared, carrier: true, requires: vec![], companion: false }] };
        let reads = Cell::new(0);
        let refusal = preflight::select_members(source::BUILTIN_REGISTRY, source::BUILTIN_REGISTRY_BYTES, &resolution, &[], |_| { reads.set(reads.get() + 1); Ok(vec![head.clone()]) }, &root.join("selection")).unwrap_err();
        assert_eq!(refusal.code, Code::AMBIGUOUS_CLASSIFICATION);
        assert_eq!(reads.get(), 1, "a single unknown carrier must be checked before payload transfer");
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn full_cli_conversion_mints_a_receipt_after_the_live_golden_suite() {
    use std::process::Command;
    let root = root();
    let file = write_source(&root, false, "blocks.0");
    let plan = plan(&file).unwrap();
    let order = root.join("order.json");
    fs::write(
        &order,
        canon::write(&Value::arr(
            plan.ops
                .iter()
                .map(|op| {
                    Value::arr(vec![
                        Value::str(op.component.clone()),
                        Value::str(op.out_key.clone()),
                    ])
                })
                .collect(),
        )),
    )
    .unwrap();
    let registry = root.join("registry.json");
    let source_plan = root.join("source-plan.json");
    let store = root.join("store");
    let bin = env!("CARGO_BIN_EXE_tfs");
    for args in [
        vec!["store".into(), "init".into(), store.display().to_string()],
        vec![
            "ingest".into(),
            "bank".into(),
            "safetensors.h3_lora".into(),
            "h3.lora/2".into(),
            "--source".into(),
            format!("adapter={}", file.path.display()),
            "--registry".into(),
            registry.display().to_string(),
            "--source-name".into(),
            "trusted-test-fixture".into(),
            "--source-profile".into(),
            "fixture/h3-lora".into(),
            "--target".into(),
            "adapter=plain/1".into(),
            "--order".into(),
            order.display().to_string(),
        ],
        vec![
            "ingest".into(),
            "source-plan".into(),
            "--registry".into(),
            registry.display().to_string(),
            "--carrier".into(),
            file.path.display().to_string(),
            "--out".into(),
            source_plan.display().to_string(),
        ],
        vec![
            "ingest".into(),
            "run".into(),
            store.display().to_string(),
            "--source-plan".into(),
            source_plan.display().to_string(),
        ],
    ] {
        let output = Command::new(bin).args(&args).output().unwrap();
        assert!(
            output.status.success(),
            "{args:?}\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let Value::Obj(fields) = canon::parse(&fs::read(&source_plan).unwrap(), 1 << 20).unwrap()
    else {
        panic!("source plan")
    };
    let Value::Str(session) = &fields.iter().find(|(k, _)| k == "session").unwrap().1 else {
        panic!("session")
    };
    let receipt = fs::read(store.join("tmp/ingest").join(session).join("receipt.json")).unwrap();
    assert!(String::from_utf8_lossy(&receipt).contains("\"golden_cases\":2"));
    let rejected = Command::new(bin)
        .args([
            "ingest",
            "restamp",
            &store.display().to_string(),
            session,
            "--plant",
            "quota-bytes",
        ])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("QUOTA_EXHAUSTED"));
    let replay = Command::new(bin)
        .args([
            "ingest",
            "run",
            &store.display().to_string(),
            "--source-plan",
            &source_plan.display().to_string(),
        ])
        .output()
        .unwrap();
    assert!(
        replay.status.success(),
        "{}",
        String::from_utf8_lossy(&replay.stderr)
    );
    let replay_receipt =
        fs::read(store.join("tmp/ingest").join(session).join("receipt.json")).unwrap();
    assert!(String::from_utf8_lossy(&replay_receipt).contains("\"roles_resumed\":18"));
    assert!(String::from_utf8_lossy(&replay_receipt).contains("\"bytes_written\":0"));
    fs::remove_dir_all(root).unwrap();
}

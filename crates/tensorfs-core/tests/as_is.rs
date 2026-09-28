//! A source no reviewed profile recognizes is stored as it is, through the path an upload
//! takes: the profile selected from provider heads, the selection narrowed by its plan, then
//! `prepare_selected_source` over the landed bodies. Malformed carriers still refuse.

use std::path::PathBuf;
use tensorfs_core::{
    canon::{self, Value},
    checkpoint,
    err::Code,
    header::Header,
    ids::ObjectRef,
    ingest::{
        preflight,
        source::{self, ModelSourceProfile, SelectedSourceMember, AS_IS},
        source::{BUILTIN_REGISTRY, BUILTIN_REGISTRY_BYTES},
    },
    limits,
    providers::MemberHead,
    store::Store,
};

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tensorfs-as-is-{name}-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ))
}

/// A safetensors body: `(key, dtype, elements, element bytes)`, payload bytes from the key.
fn safetensors(tensors: &[(&str, &str, u64, u64)]) -> Vec<u8> {
    let (mut fields, mut data) = (Vec::new(), Vec::new());
    for (key, dtype, elements, width) in tensors {
        let start = data.len();
        data.extend((0..elements * width).map(|i| key.len() as u8 ^ i as u8));
        fields.push(format!(
            r#""{key}":{{"dtype":"{dtype}","shape":[{elements}],"data_offsets":[{start},{}]}}"#,
            data.len()
        ));
    }
    let header = format!("{{{}}}", fields.join(","));
    let mut body = (header.len() as u64).to_le_bytes().to_vec();
    body.extend_from_slice(header.as_bytes());
    body.extend_from_slice(&data);
    body
}

fn head(member: &str, body: &[u8]) -> MemberHead {
    let head = if member.ends_with(".json") {
        body.to_vec()
    } else {
        body[..8 + u64::from_le_bytes(body[..8].try_into().unwrap()) as usize].to_vec()
    };
    MemberHead {
        member: member.into(),
        length: body.len() as u64,
        head,
    }
}

/// Select, narrow, land and prepare exactly as an upload does; the selected members and the
/// prepared model's header.
fn upload(name: &str, members: &[(&str, Vec<u8>)]) -> (Vec<String>, Header) {
    let heads: Vec<MemberHead> = members.iter().map(|(m, b)| head(m, b)).collect();
    let profile = preflight::select_profile(
        BUILTIN_REGISTRY,
        BUILTIN_REGISTRY_BYTES,
        &heads,
        &scratch(name),
    )
    .unwrap();
    assert_eq!(profile, AS_IS);
    let plan = preflight::plan(
        BUILTIN_REGISTRY,
        BUILTIN_REGISTRY_BYTES,
        &heads,
        std::slice::from_ref(&profile),
        &scratch(name),
    )
    .unwrap();
    let mut selected: Vec<String> = plan.plans[0]
        .components
        .iter()
        .filter_map(|component| component.member.clone())
        .collect();
    selected.sort();

    let root = scratch(name);
    let store = Store::init(&root).unwrap();
    let mut rows = Vec::new();
    // The selection's shards come with it, as `Resolution::select` adds an index's requires.
    for (member, body) in members {
        if !selected.contains(&member.to_string()) && !member.contains("-of-") {
            continue;
        }
        let object = ObjectRef::of(body);
        store
            .put_stream(&mut body.as_slice(), Some(&object), &Default::default())
            .unwrap();
        rows.push(SelectedSourceMember {
            member: member.to_string(),
            object,
            header: head(member, body).head,
        });
    }
    let (prepared, _) = source::prepare_selected_source(
        &store,
        &format!("upload-{}", "ab".repeat(28)),
        &format!("sha256:{}", "11".repeat(32)),
        vec![ModelSourceProfile {
            slot: "model".into(),
            profile,
        }],
        &rows,
        BUILTIN_REGISTRY,
        BUILTIN_REGISTRY_BYTES,
        None,
    )
    .unwrap();
    assert!(prepared.complete);
    let manifest = &prepared.sources[0];
    let manifest = store
        .read_manifest(&ObjectRef {
            sha256: manifest.manifest_digest["sha256:".len()..].into(),
            length: manifest.manifest_length,
        })
        .unwrap();
    let header = checkpoint::load_header(&store, manifest.header().unwrap()).unwrap();
    let _ = std::fs::remove_dir_all(root);
    (selected, header)
}

fn keys(header: &Header) -> Vec<(String, Vec<String>)> {
    header
        .components
        .iter()
        .map(|(name, tensors)| {
            (
                name.clone(),
                tensors.iter().map(|(k, _)| k.clone()).collect(),
            )
        })
        .collect()
}

const RAW: &str = r#"{"converter":"identity/1","dialect":"safetensors","state":"raw"}"#;

#[test]
fn an_unrecognized_single_file_is_stored_as_it_is() {
    // Civitai-shaped: one extensionless member, quant roles and a marker beside a weight.
    let body = safetensors(&[
        ("model.diffusion_model.out.weight", "F16", 4, 2),
        ("blocks.0.weight", "F8_E4M3", 8, 1),
        ("blocks.0.weight_scale", "F32", 1, 4),
        ("blocks.0.comfy_quant", "U8", 3, 1),
    ]);
    let (selected, header) = upload("single", &[("civitai/files/2718327", body)]);
    assert_eq!(selected, ["civitai/files/2718327"]);
    let mut stored = keys(&header);
    stored[0].1.sort();
    assert_eq!(
        stored,
        [(
            "model".to_string(),
            vec![
                "blocks.0.comfy_quant".to_string(),
                "blocks.0.weight".into(),
                "blocks.0.weight_scale".into(),
                "model.diffusion_model.out.weight".into(),
            ]
        )]
    );
    assert_eq!(
        header.configs,
        [("normalization".to_string(), RAW.as_bytes().to_vec())]
    );
}

#[test]
fn an_unrecognized_repository_keeps_one_default_file_per_folder() {
    let weights = |dtype: &str, width| safetensors(&[("conv.weight", dtype, 4, width)]);
    let index = br#"{"metadata":{"total_size":16},"weight_map":{"a.weight":"model-00001-of-00002.safetensors","b.weight":"model-00002-of-00002.safetensors"}}"#;
    let (selected, header) = upload(
        "repository",
        &[
            ("single.safetensors", weights("F32", 4)),
            (
                "unet/diffusion_pytorch_model.fp16.safetensors",
                weights("F16", 2),
            ),
            (
                "unet/diffusion_pytorch_model.safetensors",
                weights("F32", 4),
            ),
            (
                "vae/diffusion_pytorch_model.fp32.safetensors",
                weights("F32", 4),
            ),
            (
                "vae/diffusion_pytorch_model.fp16.safetensors",
                weights("F16", 2),
            ),
            ("text/model.safetensors.index.json", index.to_vec()),
            (
                "text/model-00001-of-00002.safetensors",
                safetensors(&[("a.weight", "F32", 2, 4)]),
            ),
            (
                "text/model-00002-of-00002.safetensors",
                safetensors(&[("b.weight", "F32", 2, 4)]),
            ),
        ],
    );
    // Folders win over the loose root file; non-variant, else lowest precision.
    assert_eq!(
        selected,
        [
            "text/model.safetensors.index.json",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.fp16.safetensors",
        ]
    );
    let mut stored = keys(&header);
    stored.sort();
    stored[0].1.sort();
    assert_eq!(
        stored,
        [
            (
                "text".to_string(),
                vec!["a.weight".to_string(), "b.weight".into()]
            ),
            ("unet".into(), vec!["conv.weight".into()]),
            ("vae".into(), vec!["conv.weight".into()]),
        ]
    );
    let unet = &header
        .components
        .iter()
        .find(|(c, _)| c == "unet")
        .unwrap()
        .1[0]
        .1;
    assert_eq!(unet.dtype.name(), "f32");
    assert_eq!(header.configs[0].1, RAW.as_bytes());
}

#[test]
fn a_reviewed_source_is_never_stored_as_is_unless_asked() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vectors/civitai-sdxl-headers");
    let manifest = canon::parse(
        &std::fs::read(root.join("MANIFEST.json")).unwrap(),
        limits::DOC_MAX_BYTES,
    )
    .unwrap();
    let Value::Obj(fields) = manifest else {
        panic!("manifest object")
    };
    let field = |name: &str| {
        fields
            .iter()
            .find(|(key, _)| key == name)
            .unwrap()
            .1
            .clone()
    };
    let (Value::Str(file), Value::Str(member), Value::Int(length)) =
        (field("file"), field("member"), field("full_bytes"))
    else {
        panic!("manifest fields")
    };
    let heads = vec![MemberHead {
        member,
        length: length as u64,
        head: std::fs::read(root.join(file)).unwrap(),
    }];
    let selected = preflight::select_profile(
        BUILTIN_REGISTRY,
        BUILTIN_REGISTRY_BYTES,
        &heads,
        &scratch("r"),
    )
    .unwrap();
    assert_ne!(selected, AS_IS);
    let forced = preflight::plan(
        BUILTIN_REGISTRY,
        BUILTIN_REGISTRY_BYTES,
        &heads,
        &[AS_IS.to_string()],
        &scratch("forced"),
    )
    .unwrap();
    assert_eq!(forced.plans[0].converter, "identity/1");
    assert_eq!(forced.plans[0].components[0].component, "model");
}

#[test]
fn a_malformed_carrier_still_refuses() {
    let good = safetensors(&[("weight", "F32", 4, 4)]);
    let refused = |name: &str, head: Vec<u8>, length: u64| {
        let member = MemberHead {
            member: "model.safetensors".into(),
            length,
            head,
        };
        preflight::select_profile(
            BUILTIN_REGISTRY,
            BUILTIN_REGISTRY_BYTES,
            &[member],
            &scratch(name),
        )
        .unwrap_err()
        .code
    };
    let prefix = head("model.safetensors", &good).head;
    // Truncated: the header declares more data than the pinned member holds.
    assert_eq!(
        refused("truncated", prefix.clone(), good.len() as u64 - 1),
        Code::CARRIER_TRUNCATED
    );
    // A header that is not a JSON object, and a length prefix past the file.
    let mut garbled = prefix.clone();
    garbled[8] = b'[';
    let mut overlong = prefix.clone();
    overlong[..8].copy_from_slice(&(1u64 << 40).to_le_bytes());
    assert_eq!(
        refused("garbled", garbled, good.len() as u64),
        Code::MALFORMED_JSON
    );
    assert_eq!(
        refused("overlong", overlong, good.len() as u64),
        Code::CARRIER_HEADER_CAP
    );
}

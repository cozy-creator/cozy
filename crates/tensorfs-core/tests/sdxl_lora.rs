//! Two real Civitai SDXL LoRAs (header vectors: 1071060 writes the LDM UNet spelling and both
//! CLIP towers, 145907 the Diffusers UNet spelling) plan under `sdxl/lora-kohya/1` onto the
//! SDXL route table's module paths as PEFT factors. Conv (LoCon) factors and keys outside the
//! reviewed targets refuse.
use std::{collections::BTreeMap, fs, path::PathBuf};
use tensorfs_core::{
    canon::{self, Value},
    err::Code,
    ingest::{
        preflight, routes,
        source::{self, BUILTIN_REGISTRY, BUILTIN_REGISTRY_BYTES},
    },
    limits,
    providers::MemberHead,
};

const PROFILE: &str = "sdxl/lora-kohya/1";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vectors/sdxl-lora-headers")
}

/// Each banked header with its full length and expected per-component key counts.
fn vectors() -> Vec<(MemberHead, BTreeMap<String, u64>)> {
    let manifest = canon::parse(
        &fs::read(root().join("MANIFEST.json")).unwrap(),
        limits::DOC_MAX_BYTES,
    )
    .unwrap();
    let Value::Arr(rows) = manifest else {
        panic!("rows")
    };
    rows.iter()
        .map(|row| {
            let Value::Obj(fields) = row else {
                panic!("row")
            };
            let field = |name: &str| fields.iter().find(|(k, _)| k == name).unwrap().1.clone();
            let (Value::Str(member), Value::Int(length), Value::Obj(components)) =
                (field("member"), field("full_length"), field("components"))
            else {
                panic!("fields")
            };
            let counts = components
                .iter()
                .map(|(name, count)| match count {
                    Value::Int(n) => (name.clone(), *n as u64),
                    _ => panic!("count"),
                })
                .collect();
            let head = MemberHead {
                member: member.clone(),
                length: length as u64,
                head: fs::read(root().join(&member)).unwrap(),
            };
            (head, counts)
        })
        .collect()
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "sdxl-lora-{name}-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ))
}

#[test]
fn real_kohya_loras_plan_onto_the_sdxl_module_paths() {
    let table = routes::table(routes::SDXL).unwrap();
    let targets: Vec<(String, String)> = table
        .components
        .iter()
        .flat_map(|c| c.routes.iter().map(|r| (c.name.clone(), r.target.clone())))
        .collect();
    for (head, counts) in vectors() {
        let heads = vec![head];
        let path = scratch("select");
        let selected =
            preflight::select_profile(BUILTIN_REGISTRY, BUILTIN_REGISTRY_BYTES, &heads, &path)
                .unwrap();
        assert_eq!(selected, PROFILE, "{}", heads[0].member);
        let staged = preflight::stage(&path, &heads).unwrap();
        let plan = source::plan_source(
            BUILTIN_REGISTRY,
            BUILTIN_REGISTRY_BYTES,
            staged.carriers(),
            Some(PROFILE),
            None,
        )
        .unwrap();
        let mut seen: BTreeMap<String, u64> = BTreeMap::new();
        for (component, key) in &plan.construction_order {
            *seen.entry(component.clone()).or_default() += 1;
            let module = [".lora_A.weight", ".lora_B.weight", ".alpha"]
                .iter()
                .find_map(|suffix| key.strip_suffix(suffix))
                .unwrap_or_else(|| panic!("{key} is not a PEFT factor"));
            // Every factor lands on a base weight the SDXL checkpoint has.
            assert!(
                targets.contains(&(component.clone(), format!("{module}.weight"))),
                "{component} {key}"
            );
        }
        assert_eq!(seen, counts, "{}", heads[0].member);
        drop(staged);
    }
}

/// A one-tensor-per-key SafeTensors header (zeros), as its first bytes.
fn header(tensors: &[(&str, &[u64])]) -> MemberHead {
    let mut offset = 0u64;
    let mut entries = vec![];
    for (key, shape) in tensors {
        let bytes = shape.iter().product::<u64>() * 2;
        entries.push(format!(
            "\"{key}\":{{\"dtype\":\"F16\",\"shape\":{shape:?},\"data_offsets\":[{offset},{}]}}",
            offset + bytes
        ));
        offset += bytes;
    }
    let json = format!("{{{}}}", entries.join(","));
    let mut head = (json.len() as u64).to_le_bytes().to_vec();
    head.extend(json.as_bytes());
    MemberHead {
        member: "lora.safetensors".into(),
        length: head.len() as u64 + offset,
        head,
    }
}

fn refusal(tensors: &[(&str, &[u64])]) -> Code {
    let path = scratch("refuse");
    let heads = vec![header(tensors)];
    let staged = preflight::stage(&path, &heads).unwrap();
    source::plan_source(
        BUILTIN_REGISTRY,
        BUILTIN_REGISTRY_BYTES,
        staged.carriers(),
        Some(PROFILE),
        None,
    )
    .unwrap_err()
    .code
}

#[test]
fn conv_factors_unknown_targets_and_half_pairs_refuse() {
    let q = "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q";
    let conv = "lora_unet_down_blocks_1_resnets_0_conv1";
    assert_eq!(
        refusal(&[
            (&format!("{conv}.lora_down.weight"), &[4, 320, 3, 3]),
            (&format!("{conv}.lora_up.weight"), &[640, 4, 1, 1])
        ]),
        Code::ADAPTER_TARGET_UNSUPPORTED
    );
    assert_eq!(
        refusal(&[
            (
                "lora_unet_time_embedding_linear_1.lora_down.weight",
                &[4, 320]
            ),
            (
                "lora_unet_time_embedding_linear_1.lora_up.weight",
                &[1280, 4]
            )
        ]),
        Code::KEY_GRAMMAR
    );
    assert_eq!(
        refusal(&[(&format!("{q}.lora_down.weight"), &[4, 640])]),
        Code::MISSING_COMPANION_ROLE
    );
    assert_eq!(
        refusal(&[
            (&format!("{q}.lora_down.weight"), &[4, 640]),
            (&format!("{q}.lora_up.weight"), &[320, 4])
        ]),
        Code::SHAPE_MISMATCH
    );
}

/// The committed model's tensors as one SafeTensors file, `<component>.<key>` each.
fn export(
    store: &tensorfs_core::store::Store,
    manifest: &tensorfs_core::ids::ObjectRef,
    to: &std::path::Path,
) {
    use tensorfs_core::{dtype::Dtype, header::Body};
    let header = tensorfs_core::checkpoint::load_header(
        store,
        store.read_manifest(manifest).unwrap().header().unwrap(),
    )
    .unwrap();
    let (mut entries, mut data) = (vec![], vec![]);
    for (component, tensors) in &header.components {
        for (key, tensor) in tensors {
            let start = data.len();
            for (_, part) in &tensor.parts {
                match &part.body {
                    Body::Inline(bytes) => data.extend(bytes),
                    Body::Segments(objects) => {
                        for object in objects {
                            data.extend(
                                store.read_range(&object.sha256, 0, object.length).unwrap(),
                            );
                        }
                    }
                }
            }
            let dtype = match tensor.dtype {
                Dtype::F16 => "F16",
                Dtype::Bf16 => "BF16",
                Dtype::F32 => "F32",
                other => panic!("{other:?}"),
            };
            entries.push(format!(
                "\"{component}.{key}\":{{\"dtype\":\"{dtype}\",\"shape\":{:?},\"data_offsets\":[{start},{}]}}",
                tensor.shape,
                data.len()
            ));
        }
    }
    let json = format!("{{{}}}", entries.join(","));
    let mut file = (json.len() as u64).to_le_bytes().to_vec();
    file.extend(json.as_bytes());
    file.extend(data);
    fs::write(to, file).unwrap();
}

/// Real Civitai LoRAs made into PEFT models through `source_model` (real network; 1071060
/// needs a Civitai token in CIVITAI_TOKEN). With SDXL_LORA_EXPORT set, each model's tensors
/// are written there for the numeric check against diffusers' kohya loader.
#[test]
#[ignore = "real network: civitai.com"]
fn real_civitai_loras_become_peft_models() {
    use tensorfs_core::source_model::{make, Access, Request};
    let root = scratch("civitai");
    let store = tensorfs_core::store::Store::init(&root).unwrap();
    let token = std::env::var("CIVITAI_TOKEN").unwrap_or_default();
    let access = Access {
        credential: if token.is_empty() {
            String::new()
        } else {
            format!("bearer {token}")
        },
        ..Default::default()
    };
    for version in ["145907", "1071060"] {
        let source = format!("civitai://{version}");
        let request = Request {
            source: &source,
            profiles: &[],
            name: &format!("lora-{version}"),
            access: &access,
            registry: None,
            progress: &|_, _, _| {},
            cancellation: None,
        };
        let made = make(&store, &request).unwrap();
        assert_eq!(made.profiles, [PROFILE], "{version}");
        if let Ok(dir) = std::env::var("SDXL_LORA_EXPORT") {
            export(
                &store,
                &made.manifest,
                &PathBuf::from(dir).join(format!("{version}.safetensors")),
            );
        }
    }
    let _ = fs::remove_dir_all(root);
}

//! Civitai 2945208 is Anima's three ComfyUI files: the `net.` DiT with its `net.llm_adapter.`
//! text conditioner, the Qwen3 text encoder and the Qwen-Image VAE. From their real headers
//! the unnamed selection is the reviewed Anima profile, whose plan is the census of the
//! reference repository's own save: every key, in constructor order, at its shape, in bf16.
//! The routed ingest moves each source tensor's bytes to its canonical key.
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
use tensorfs_core::{
    canon::{self, Value},
    catalog::Catalog,
    checkpoint,
    ids::ObjectRef,
    ingest::{
        carrier,
        fingerprint::FingerprintRegistry,
        preflight, routes,
        source::{self, CarrierInput, CarrierSet, AS_IS, BUILTIN_REGISTRY, BUILTIN_REGISTRY_BYTES},
        transaction,
    },
    limits,
    providers::{MemberHead, Provenance, Resolution, ResolvedMember},
    registry,
    store::Store,
};

const PROFILE: &str = "civitai/anima/single-file/1";
const MODEL: &str = "civitai/files/2824391";
const TEXT: &str = "civitai/files/2824387";
const VAE: &str = "civitai/files/2004692";

/// A safetensors head (length prefix plus JSON) as the carrier reader parses it.
fn parse(head: &[u8]) -> carrier::SourceHeader {
    let length = u64::from_le_bytes(head[..8].try_into().unwrap()) as usize;
    carrier::parse_header(&head[8..8 + length], 8 + length as u64, None).unwrap()
}

fn vectors() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vectors")
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "civitai-anima-{name}-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ))
}

fn field<'v>(value: &'v Value, name: &str) -> &'v Value {
    let Value::Obj(rows) = value else {
        panic!("object")
    };
    &rows.iter().find(|(key, _)| key == name).unwrap().1
}

/// The three members' real heads, from the reviewed MANIFEST.
fn heads(dir: &str) -> Vec<MemberHead> {
    let root = vectors().join(dir);
    let manifest = canon::parse(
        &fs::read(root.join("MANIFEST.json")).unwrap(),
        limits::DOC_MAX_BYTES,
    )
    .unwrap();
    let Value::Arr(members) = field(&manifest, "members") else {
        panic!("members")
    };
    members
        .iter()
        .map(|row| {
            let (Value::Str(member), Value::Str(file), Value::Int(length)) = (
                field(row, "member"),
                field(row, "file"),
                field(row, "full_bytes"),
            ) else {
                panic!("member fields")
            };
            MemberHead {
                member: member.clone(),
                length: *length as u64,
                head: fs::read(root.join(file)).unwrap(),
            }
        })
        .collect()
}

/// `{component: {key: (dtype, shape)}}` of the reference repository's own save.
fn reference_census() -> BTreeMap<String, BTreeMap<String, (String, Vec<u64>)>> {
    let root = vectors().join("anima-hf-headers");
    ["text_conditioner", "text_encoder", "transformer", "vae"]
        .into_iter()
        .map(|component| {
            let file = fs::read_dir(root.join(component))
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .find(|path| path.extension().is_some_and(|e| e == "header"))
                .unwrap();
            let header = parse(&fs::read(file).unwrap());
            let rows = header
                .tensors
                .iter()
                .map(|t| (t.key.clone(), (t.dtype.name().to_string(), t.shape.clone())))
                .collect();
            (component.to_string(), rows)
        })
        .collect()
}

fn table_order() -> Vec<(String, String)> {
    routes::table(routes::ANIMA)
        .unwrap()
        .components
        .iter()
        .flat_map(|c| c.routes.iter().map(|r| (c.name.clone(), r.target.clone())))
        .collect()
}

#[test]
fn the_real_headers_select_the_anima_profile_and_plan_the_constructor_census() {
    let heads = heads("anima-civitai-headers");
    let path = scratch("select");
    assert_eq!(
        preflight::select_profile(BUILTIN_REGISTRY, BUILTIN_REGISTRY_BYTES, &heads, &path).unwrap(),
        PROFILE
    );
    let staged = preflight::stage(&path, &heads).unwrap();
    let plan = source::plan_source(
        BUILTIN_REGISTRY,
        BUILTIN_REGISTRY_BYTES,
        staged.carriers(),
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        (plan.profile.as_str(), plan.converter.as_str()),
        (PROFILE, routes::ANIMA)
    );
    assert_eq!(plan.construction_order, table_order());
    let sources: Vec<(&str, &str, bool)> = plan
        .sources
        .iter()
        .map(|s| {
            (
                s.component.as_str(),
                s.source_member.as_deref().unwrap(),
                s.projected,
            )
        })
        .collect();
    assert_eq!(
        sources,
        [
            ("text_conditioner", MODEL, true),
            ("text_encoder", TEXT, true),
            ("transformer", MODEL, true),
            ("vae", VAE, true),
        ]
    );

    // Every planned tensor is the reference save's tensor: same key set per component,
    // same shape, bf16, and nothing else.
    let registry = FingerprintRegistry::parse(BUILTIN_REGISTRY_BYTES).unwrap();
    let inputs: Vec<(String, CarrierInput)> = plan
        .sources
        .iter()
        .map(|s| {
            (
                s.component.clone(),
                CarrierInput {
                    member: s.source_member.clone(),
                    path: s.path.clone(),
                },
            )
        })
        .collect();
    let prepared = source::prepare(
        &plan.target,
        &inputs,
        &CarrierSet::of(staged.carriers()),
        source::Classify::Banked(&registry),
        None,
    )
    .unwrap();
    let mut planned: BTreeMap<String, BTreeMap<String, (String, Vec<u64>)>> = BTreeMap::new();
    for op in &prepared.plan.ops {
        planned.entry(op.component.clone()).or_default().insert(
            op.out_key.clone(),
            (
                op.logical_dtype.name().to_string(),
                op.logical_shape.clone(),
            ),
        );
    }
    assert_eq!(planned, reference_census());
    assert_eq!(prepared.plan.ops.len(), 1189);
    drop(staged);
    assert!(!path.exists());
}

fn resolution(members: &[(&str, bool)], heads: &[MemberHead]) -> Resolution {
    Resolution {
        canonical: "civitai://2945208".into(),
        members: members
            .iter()
            .map(|(member, companion)| {
                let head = heads.iter().find(|h| h.member == *member).unwrap();
                ResolvedMember {
                    member: member.to_string(),
                    object: ObjectRef {
                        sha256: tensorfs_core::sha256::hex_digest(member.as_bytes()),
                        length: head.length,
                    },
                    url: format!("https://civitai.invalid/{member}"),
                    provenance: Provenance::Declared,
                    carrier: true,
                    requires: vec![],
                    companion: *companion,
                }
            })
            .collect(),
        selection_sha256: String::new(),
    }
}

fn select(members: &[(&str, bool)], heads: &[MemberHead], profiles: &[String]) -> Vec<String> {
    preflight::select_members(
        BUILTIN_REGISTRY,
        BUILTIN_REGISTRY_BYTES,
        &resolution(members, heads),
        profiles,
        |selection| {
            Ok(heads
                .iter()
                .filter(|h| selection.members.iter().any(|row| row.member == h.member))
                .cloned()
                .collect())
        },
        &scratch("members"),
    )
    .unwrap()
}

#[test]
fn companions_join_only_to_compose_a_reviewed_model_with_the_primary() {
    let anima = heads("anima-civitai-headers");
    let all = [VAE, TEXT, MODEL].map(String::from).to_vec();
    assert_eq!(
        select(&[(MODEL, false), (TEXT, true), (VAE, true)], &anima, &[]),
        all
    );
    assert_eq!(
        select(
            &[(MODEL, false), (TEXT, true), (VAE, true)],
            &anima,
            &[PROFILE.to_string()]
        ),
        all
    );
    // Without its text encoder the version composes nothing reviewed: its primary, as-is.
    assert_eq!(
        select(&[(MODEL, false), (VAE, true)], &anima, &[]),
        [MODEL.to_string()]
    );
    let alone = &anima[..1];
    let path = scratch("alone");
    assert_eq!(
        preflight::select_profile(BUILTIN_REGISTRY, BUILTIN_REGISTRY_BYTES, alone, &path).unwrap(),
        AS_IS
    );

    // An SDXL primary that plans on its own stays exactly that, even beside a companion that
    // would plan the same profile (an fp32 twin), named or not.
    let sdxl =
        fs::read(vectors().join("civitai-sdxl-headers/128078-92696.safetensors.header")).unwrap();
    let length = 6_938_078_334;
    let pair = vec![
        MemberHead {
            member: "civitai/files/92696".into(),
            length,
            head: sdxl.clone(),
        },
        MemberHead {
            member: "civitai/files/92697".into(),
            length,
            head: sdxl,
        },
    ];
    let members = [
        ("civitai/files/92696", false),
        ("civitai/files/92697", true),
    ];
    let primary = ["civitai/files/92696".to_string()];
    assert_eq!(select(&members, &pair, &[]), primary);
    assert_eq!(
        select(&members, &pair, &["civitai/sdxl/single-file/1".into()]),
        primary
    );
}

/// Each dimension clamped to 4: the real layout at a size a test can move.
fn small(shape: &[u64]) -> Vec<u64> {
    shape.iter().map(|d| (*d).min(4)).collect()
}

/// The member's real keys, order and dtypes at `small` shapes, with distinct deterministic
/// bytes per tensor.
fn synthetic(dir: &Path, head: &MemberHead) -> (CarrierInput, BTreeMap<String, Vec<u8>>) {
    let (mut fields, mut payload, mut values) = (Vec::new(), Vec::new(), BTreeMap::new());
    for (index, t) in parse(&head.head).tensors.iter().enumerate() {
        let shape = small(&t.shape);
        let count: u64 = shape.iter().product::<u64>() * t.dtype.size();
        let bytes: Vec<u8> = (0..count)
            .map(|i| (index as u64 * 131 + i * 7 + head.member.len() as u64) as u8)
            .collect();
        let begin = payload.len();
        payload.extend_from_slice(&bytes);
        fields.push(format!(
            r#""{}":{{"dtype":"{}","shape":{:?},"data_offsets":[{begin},{}]}}"#,
            t.key,
            t.dtype.name().to_uppercase(),
            shape,
            payload.len()
        ));
        values.insert(t.key.clone(), bytes);
    }
    let json = format!("{{{}}}", fields.join(","));
    let path = dir.join(head.member.replace('/', "-"));
    let mut body = (json.len() as u64).to_le_bytes().to_vec();
    body.extend_from_slice(json.as_bytes());
    body.extend_from_slice(&payload);
    fs::write(&path, body).unwrap();
    let carrier = CarrierInput {
        member: Some(head.member.clone()),
        path,
    };
    (carrier, values)
}

#[test]
fn a_synthetic_carrier_with_the_real_layout_ingests_to_the_constructor_census() {
    let dir = scratch("ingest");
    fs::create_dir_all(&dir).unwrap();
    let heads = heads("anima-civitai-headers");
    let mut carriers = Vec::new();
    let mut values = BTreeMap::new();
    for head in &heads {
        let (carrier, written) = synthetic(&dir, head);
        carriers.push(carrier);
        values.insert(head.member.clone(), written);
    }
    let plan = source::plan_source(
        BUILTIN_REGISTRY,
        BUILTIN_REGISTRY_BYTES,
        &carriers,
        None,
        None,
    )
    .unwrap();
    assert_eq!(plan.profile, PROFILE);
    let inputs: Vec<(String, CarrierInput)> = plan
        .sources
        .iter()
        .map(|s| {
            (
                s.component.clone(),
                CarrierInput {
                    member: s.source_member.clone(),
                    path: s.path.clone(),
                },
            )
        })
        .collect();
    let registry = FingerprintRegistry::parse(BUILTIN_REGISTRY_BYTES).unwrap();
    let mut prepared = source::prepare(
        &plan.target,
        &inputs,
        &CarrierSet::of(&carriers),
        source::Classify::Banked(&registry),
        None,
    )
    .unwrap();
    prepared.plan.apply_order(&plan.construction_order).unwrap();
    let store = Store::init(&dir.join("store")).unwrap();
    let catalog = Catalog::open(store.root()).unwrap();
    let _writer = catalog
        .resume_operation("convert", "proof", "anima")
        .unwrap();
    let specs: Vec<_> = registry::seeds()
        .into_iter()
        .filter(|s| s.alias == "plain/1")
        .map(|s| (s.alias.to_string(), s.spec))
        .collect();
    let (_, outcome) = transaction::advance(
        &store,
        &prepared.plan,
        &prepared.files,
        &specs,
        &[],
        "convert",
        transaction::Carriers::All,
        None,
    )
    .unwrap();
    let header = outcome.unwrap().header;

    let census = reference_census();
    let order: Vec<(String, String)> = header
        .components
        .iter()
        .flat_map(|(c, tensors)| tensors.iter().map(|(k, _)| (c.clone(), k.clone())))
        .collect();
    assert_eq!(order, table_order());
    let table = routes::table(routes::ANIMA).unwrap();
    let member = |component: &str| match component {
        "text_encoder" => TEXT,
        "vae" => VAE,
        _ => MODEL,
    };
    for (component, tensors) in &header.components {
        let routes = table
            .components
            .iter()
            .find(|c| &c.name == component)
            .unwrap();
        for (key, tensor) in tensors {
            let (dtype, shape) = &census[component][key];
            assert_eq!(
                (tensor.dtype.name(), tensor.shape.clone()),
                (dtype.as_str(), small(shape)),
                "{component}.{key}"
            );
            let route = routes.routes.iter().find(|r| &r.target == key).unwrap();
            let source = format!("{}{}", routes.prefix, route.source);
            let mut got = Vec::new();
            checkpoint::materialize(&store, key, &tensor.parts[0].1, &mut got).unwrap();
            assert_eq!(
                got,
                values[member(component)][&source],
                "{component}.{key} <- {source}"
            );
        }
    }
    let (name, record) = &header.configs[0];
    assert_eq!(name, "normalization");
    assert_eq!(
        String::from_utf8(record.clone()).unwrap(),
        format!(
            "{{\"converter\":\"{}\",\"dialect\":\"diffusers\",\"reference\":\"{}\",\"state\":\"normalized\"}}",
            routes::ANIMA,
            table.reference
        )
    );
    drop((_writer, catalog, store));
    fs::remove_dir_all(dir).unwrap();
}

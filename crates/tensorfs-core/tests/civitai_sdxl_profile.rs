//! The Civitai SDXL base checkpoint (version 128078) plans under two reviewed profiles, a
//! version-bound one and the generic SDXL single-file one, and both plan identically. That is
//! one answer, so no profile has to be named. Both plan the routed SDXL converter straight
//! to the Diffusers constructor's keys, in its order.
use std::{fs, path::PathBuf};
use tensorfs_core::{
    canon::{self, Value},
    ingest::{
        preflight, routes,
        source::{self, BUILTIN_REGISTRY, BUILTIN_REGISTRY_BYTES},
    },
    limits,
    providers::MemberHead,
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vectors/civitai-sdxl-headers")
}

fn head() -> MemberHead {
    let manifest = canon::parse(
        &fs::read(root().join("MANIFEST.json")).unwrap(),
        limits::DOC_MAX_BYTES,
    )
    .unwrap();
    let field = |name: &str| {
        let Value::Obj(rows) = &manifest else {
            panic!("object")
        };
        rows.iter().find(|(key, _)| key == name).unwrap().1.clone()
    };
    let (Value::Str(file), Value::Str(member), Value::Int(length)) =
        (field("file"), field("member"), field("full_bytes"))
    else {
        panic!("manifest fields")
    };
    MemberHead {
        member,
        length: length as u64,
        head: fs::read(root().join(file)).unwrap(),
    }
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "civitai-sdxl-{name}-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ))
}

#[test]
fn equivalent_profiles_select_the_most_specific_without_being_named() {
    let heads = vec![head()];
    let path = scratch("select");
    assert_eq!(
        preflight::select_profile(BUILTIN_REGISTRY, BUILTIN_REGISTRY_BYTES, &heads, &path).unwrap(),
        "civitai/101055/128078/single-file-fp16"
    );
    assert!(!path.exists());

    // Naming either profile still wins, and both plan the same bytes.
    let staged = preflight::stage(&path, &heads).unwrap();
    let plans: Vec<_> = [
        "civitai/101055/128078/single-file-fp16",
        "civitai/sdxl/single-file/1",
    ]
    .into_iter()
    .map(|name| {
        source::plan_source(
            BUILTIN_REGISTRY,
            BUILTIN_REGISTRY_BYTES,
            staged.carriers(),
            Some(name),
            None,
        )
        .unwrap()
    })
    .collect();
    assert_eq!(plans[1].profile, "civitai/sdxl/single-file/1");
    assert_eq!(plans[0].session, plans[1].session);
    assert_eq!(plans[0].construction_order, plans[1].construction_order);
    let table = routes::table(routes::SDXL).unwrap();
    let canonical: Vec<(String, String)> = table
        .components
        .iter()
        .flat_map(|c| c.routes.iter().map(|r| (c.name.clone(), r.target.clone())))
        .collect();
    for plan in &plans {
        assert_eq!(plan.converter, routes::SDXL);
        assert_eq!(plan.construction_order, canonical);
    }
    assert_eq!(canonical.len(), 2641);
    assert!(canonical.contains(&(
        "text_encoder_2".into(),
        "text_model.encoder.layers.31.self_attn.v_proj.weight".into()
    )));
    drop(staged);
    assert!(!path.exists());
}

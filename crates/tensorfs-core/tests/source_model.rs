//! A real Hugging Face source made into a local model through the one entry point: resolve,
//! profile, land, prepare, compose with its Diffusers configs and tokenizers, commit. Real
//! network (`--ignored`): `hf-internal-testing/tiny-sdxl-pipe`, a few MiB.
use std::sync::Mutex;
use tensorfs_core::{
    checkpoint::load_header,
    source_model::{make, Access, Request},
    store::Store,
};

const SOURCE: &str = "hf://hf-internal-testing/tiny-sdxl-pipe@20594cbc343cfcfe447af5c87cdaf6c436b453f2/unet/diffusion_pytorch_model.safetensors";

#[test]
#[ignore = "real network: huggingface.co"]
fn a_real_hugging_face_carrier_becomes_a_local_model_with_its_pipeline_configs() {
    let root = std::env::temp_dir().join(format!(
        "source-model-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ));
    let store = Store::init(&root).unwrap();
    let stages = Mutex::new(Vec::new());
    let progress = |stage: &str, _: u64, _: u64| {
        let mut seen = stages.lock().unwrap();
        if seen.last().map(String::as_str) != Some(stage) {
            seen.push(stage.to_string());
        }
    };
    let access = Access::default();
    let request = Request {
        source: SOURCE,
        profiles: &[],
        name: "tiny-sdxl-unet",
        access: &access,
        registry: None,
        progress: &progress,
        cancellation: None,
    };
    let made = make(&store, &request).unwrap();
    assert_eq!(made.repository, "local/tiny-sdxl-unet");
    assert_eq!(made.profiles.len(), 1, "{made:?}");
    let header = load_header(
        &store,
        store
            .read_manifest(&made.manifest)
            .unwrap()
            .header()
            .unwrap(),
    )
    .unwrap();
    assert!(!header.components.is_empty());
    let configs: Vec<_> = header
        .configs
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    for name in ["model_index", "unet", "vae", "scheduler", "text_encoder"] {
        assert!(configs.contains(&name), "{configs:?}");
    }
    let unet = &header.configs.iter().find(|(n, _)| n == "unet").unwrap().1;
    assert!(!String::from_utf8_lossy(unet).contains("_name_or_path"));
    assert!(
        header
            .assets
            .iter()
            .any(|(name, _)| name.starts_with("tokenizer/")),
        "{:?}",
        header.assets
    );
    let stages = stages.lock().unwrap().clone();
    assert!(
        stages.iter().any(|s| s.starts_with("downloading")),
        "{stages:?}"
    );

    // The same source again: everything is held, the same model is committed.
    let again = make(&store, &request).unwrap();
    assert_eq!(again.manifest, made.manifest);
    let _ = std::fs::remove_dir_all(root);
}

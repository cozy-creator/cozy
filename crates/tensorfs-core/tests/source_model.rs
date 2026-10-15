//! A real Hugging Face source made into a local model through the one entry point: resolve,
//! profile, land, prepare, compose with its Diffusers configs and tokenizers, commit. Real
//! network (`--ignored`): `hf-internal-testing/tiny-sdxl-pipe`, a few MiB.
use std::sync::Mutex;
use tensorfs_core::{
    checkpoint::load_header,
    source_model::{make, Access, Request},
    store::{Fault, Store},
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

/// A file a client wrote into the Store (`object://sha256:<hex>/<file>`) is made like a
/// provider's single file, with no provider: its configs come from its converters' pinned
/// reference repository. Real network for the fixture and the reference configs.
#[test]
#[ignore = "real network: huggingface.co"]
fn a_held_file_becomes_a_local_model_with_no_provider() {
    let root = std::env::temp_dir().join(format!(
        "source-model-held-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ));
    let store = Store::init(&root).unwrap();
    let file = root.join("unet.safetensors");
    let url = "https://huggingface.co/hf-internal-testing/tiny-sdxl-pipe/resolve/20594cbc343cfcfe447af5c87cdaf6c436b453f2/unet/diffusion_pytorch_model.safetensors";
    let fetched = std::process::Command::new("curl")
        .args(["-sSfL", "-o"])
        .arg(&file)
        .arg(url)
        .status()
        .unwrap();
    assert!(fetched.success());
    let held = store.put_file(&file, None, &Fault::default()).unwrap().obj;
    let source = format!("object://sha256:{}/unet.safetensors", held.sha256);
    let progress = |_: &str, _: u64, _: u64| {};
    let access = Access::default();
    let request = Request {
        source: &source,
        profiles: &[],
        name: "held-unet",
        access: &access,
        registry: None,
        progress: &progress,
        cancellation: None,
    };
    let made = make(&store, &request).unwrap();
    assert_eq!(made.repository, "local/held-unet");
    assert_eq!(made.resolved, source);
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

    // A file the Store does not hold is refused, never fetched.
    let absent = format!("object://sha256:{}/unet.safetensors", "0".repeat(64));
    let refused = make(
        &store,
        &Request {
            source: &absent,
            name: "absent",
            ..request
        },
    )
    .unwrap_err();
    assert!(!refused.detail.is_empty());
    let _ = std::fs::remove_dir_all(root);
}

fn temporary(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "source-model-{name}-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ))
}

fn made_header(store: &Store, made: &tensorfs_core::source_model::Made) -> tensorfs_core::header::Header {
    load_header(
        store,
        store.read_manifest(&made.manifest).unwrap().header().unwrap(),
    )
    .unwrap()
}

/// A plain Transformers repository is servable as uploaded: its root model with the config
/// as the construction config and the tokenizer, processor, generation config and chat
/// template as assets, never the README.
#[test]
#[ignore = "real network: huggingface.co"]
fn a_transformers_repository_carries_its_config_tokenizer_and_processor() {
    let root = temporary("transformers");
    let store = Store::init(&root).unwrap();
    let progress = |_: &str, _: u64, _: u64| {};
    let access = Access::default();
    let made = make(
        &store,
        &Request {
            source: "hf://trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration@62f13fe1c31492fb4195c72ec52cbb736f72fefb",
            profiles: &[],
            name: "tiny-qwen-vl",
            access: &access,
            registry: None,
            progress: &progress,
            cancellation: None,
        },
    )
    .unwrap();
    assert_eq!(made.profiles, ["as-is/1"]);
    let header = made_header(&store, &made);
    let components: Vec<_> = header.components.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(components, ["model"]);
    let configs: Vec<_> = header.configs.iter().map(|(name, _)| name.as_str()).collect();
    assert!(configs.contains(&"model"), "{configs:?}");
    let config = &header.configs.iter().find(|(name, _)| name == "model").unwrap().1;
    assert!(String::from_utf8_lossy(config).contains("\"model_type\""));
    let assets: Vec<_> = header.assets.iter().map(|(name, _)| name.as_str()).collect();
    for file in [
        "chat_template.jinja",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
    ] {
        assert!(assets.contains(&file), "{file} not in {assets:?}");
    }
    assert!(!assets.contains(&"README.md"), "{assets:?}");
    let _ = std::fs::remove_dir_all(root);
}

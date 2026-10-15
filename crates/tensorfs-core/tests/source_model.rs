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

/// A real Hugging Face repository at a pinned revision, mirrored whole (real network) and
/// served from loopback with `extra` files added: the repository as its owner might have
/// grown it, read through the real provider, download, conversion and composition path.
struct Mirror {
    base: String,
}

impl Mirror {
    fn of(repo: &str, revision: &str, extra: Vec<(String, Vec<u8>)>) -> (Mirror, String) {
        let tree = format!("https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true");
        let listing = std::process::Command::new("curl").args(["-sSfL", &tree]).output().unwrap();
        assert!(listing.status.success(), "{tree}");
        let rows: Vec<serde_json::Value> = serde_json::from_slice(&listing.stdout).unwrap();
        let mut files: Vec<(String, Vec<u8>)> = rows
            .iter()
            .filter(|row| row["type"] == "file")
            .map(|row| {
                let path = row["path"].as_str().unwrap().to_string();
                let url = format!("https://huggingface.co/{repo}/resolve/{revision}/{path}");
                let body = std::process::Command::new("curl").args(["-sSfL", &url]).output().unwrap();
                assert!(body.status.success(), "{url}");
                (path, body.stdout)
            })
            .collect();
        files.extend(extra);
        let files: std::sync::Arc<std::collections::BTreeMap<String, Vec<u8>>> =
            std::sync::Arc::new(files.into_iter().collect());
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for socket in listener.incoming().flatten() {
                let files = files.clone();
                std::thread::spawn(move || serve(socket, &files));
            }
        });
        (Mirror { base }, format!("hf://{repo}@{revision}"))
    }

    fn access(&self) -> Access {
        Access {
            credential: String::new(),
            endpoints: tensorfs_core::providers::Endpoints {
                huggingface: self.base.clone(),
                allow_local: true,
                ..Default::default()
            },
        }
    }
}

/// One keep-alive connection of the mirror: the tree listing (LFS rows for SafeTensors
/// files) and ranged `resolve` reads.
fn serve(mut socket: std::net::TcpStream, files: &std::collections::BTreeMap<String, Vec<u8>>) {
    use std::io::{Read, Write};
    loop {
        let mut head = Vec::new();
        let mut byte = [0u8];
        while !head.ends_with(b"\r\n\r\n") {
            if !matches!(socket.read(&mut byte), Ok(1)) {
                return;
            }
            head.push(byte[0]);
        }
        let text = String::from_utf8_lossy(&head).to_string();
        let mut first = text.lines().next().unwrap().split_whitespace();
        let (method, target) = (first.next().unwrap().to_string(), first.next().unwrap().to_string());
        let range = text.lines().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            let (a, b) = name.eq_ignore_ascii_case("range").then_some(value.trim())?
                .strip_prefix("bytes=")?
                .split_once('-')?;
            Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?))
        });
        let path = target.split('?').next().unwrap().replace("%20", " ");
        let body = if path.contains("/tree/") {
            let rows: Vec<_> = files
                .iter()
                .map(|(path, bytes)| match path.ends_with(".safetensors") {
                    true => serde_json::json!({"type": "file", "path": path, "size": bytes.len(),
                        "lfs": {"oid": tensorfs_core::sha256::hex_digest(bytes), "size": bytes.len()}}),
                    false => serde_json::json!({"type": "file", "path": path, "size": bytes.len()}),
                })
                .collect();
            Some(serde_json::to_vec(&rows).unwrap())
        } else {
            path.splitn(6, '/').nth(5).and_then(|name| files.get(name)).cloned()
        };
        let Some(body) = body else {
            let _ = socket.write_all(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\n\r\n");
            continue;
        };
        let (status, slice) = match range {
            Some((a, b)) => {
                let b = b.min(body.len() - 1);
                (format!("206 Partial Content\r\ncontent-range: bytes {a}-{b}/{}", body.len()), &body[a..=b])
            }
            None => ("200 OK".to_string(), &body[..]),
        };
        let head = format!("HTTP/1.1 {status}\r\ncontent-length: {}\r\netag: \"m\"\r\n\r\n", slice.len());
        let sent = socket.write_all(head.as_bytes()).is_ok()
            && (method == "HEAD" || socket.write_all(slice).is_ok());
        if !sent {
            return;
        }
    }
}

fn names<T>(rows: &[(String, T)]) -> Vec<&str> {
    rows.iter().map(|(name, _)| name.as_str()).collect()
}

/// A Diffusers repository is the pipeline its index names. A folder beside it (BFL's
/// `variants/` on FLUX.3 Klein, here grown onto a real tiny Flux pipeline) is not a
/// component, is never downloaded, and does not reach the model.
#[test]
#[ignore = "real network: huggingface.co"]
fn a_diffusers_repository_is_its_index_components_and_nothing_else() {
    let tensors = |name: &str| {
        let url = format!(
            "https://huggingface.co/hf-internal-testing/tiny-flux-pipe/resolve/a98bc4ec6a80c47c477ed22d7e81c79872d9c723/{name}"
        );
        std::process::Command::new("curl").args(["-sSfL", &url]).output().unwrap().stdout
    };
    let variant = tensors("transformer/diffusion_pytorch_model.safetensors");
    let (mirror, source) = Mirror::of(
        "hf-internal-testing/tiny-flux-pipe",
        "a98bc4ec6a80c47c477ed22d7e81c79872d9c723",
        vec![
            ("variants/fp8/model.safetensors".into(), variant),
            ("variants/fp8/config.json".into(), tensors("transformer/config.json")),
        ],
    );
    let root = temporary("diffusers-extra");
    let store = Store::init(&root).unwrap();
    let progress = |_: &str, _: u64, _: u64| {};
    let access = mirror.access();
    let made = make(
        &store,
        &Request {
            source: &source,
            profiles: &[],
            name: "tiny-flux",
            access: &access,
            registry: None,
            progress: &progress,
            cancellation: None,
        },
    )
    .unwrap();
    let header = made_header(&store, &made);
    assert_eq!(
        names(&header.components),
        ["text_encoder", "text_encoder_2", "transformer", "vae"]
    );
    let configs = names(&header.configs);
    for name in ["model_index", "scheduler", "text_encoder", "text_encoder_2", "transformer", "vae"] {
        assert!(configs.contains(&name), "{name} not in {configs:?}");
    }
    assert!(!configs.iter().any(|name| name.contains("variants")), "{configs:?}");
    let assets = names(&header.assets);
    assert!(assets.iter().any(|name| name.starts_with("tokenizer/")), "{assets:?}");
    assert!(assets.iter().any(|name| name.starts_with("tokenizer_2/")), "{assets:?}");
    assert!(!assets.iter().any(|name| name.starts_with("variants/")), "{assets:?}");
    let _ = std::fs::remove_dir_all(root);
}

/// A Transformers repository is its root model: a native `consolidated.safetensors` beside
/// it and an `original/` folder are neither components nor downloaded.
#[test]
#[ignore = "real network: huggingface.co"]
fn a_transformers_repository_is_its_root_model_and_nothing_else() {
    let model = std::process::Command::new("curl")
        .args(["-sSfL", "https://huggingface.co/trl-internal-testing/tiny-Qwen3ForCausalLM/resolve/main/model.safetensors"])
        .output()
        .unwrap()
        .stdout;
    let revision = String::from_utf8(
        std::process::Command::new("curl")
            .args(["-sSfL", "https://huggingface.co/api/models/trl-internal-testing/tiny-Qwen3ForCausalLM/revision/main"])
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap();
    let revision: serde_json::Value = serde_json::from_str(&revision).unwrap();
    let (mirror, source) = Mirror::of(
        "trl-internal-testing/tiny-Qwen3ForCausalLM",
        revision["sha"].as_str().unwrap(),
        vec![
            ("consolidated.safetensors".into(), model.clone()),
            ("original/model.safetensors".into(), model),
        ],
    );
    let root = temporary("transformers-extra");
    let store = Store::init(&root).unwrap();
    let progress = |_: &str, _: u64, _: u64| {};
    let access = mirror.access();
    let made = make(
        &store,
        &Request {
            source: &source,
            profiles: &[],
            name: "tiny-qwen3",
            access: &access,
            registry: None,
            progress: &progress,
            cancellation: None,
        },
    )
    .unwrap();
    let header = made_header(&store, &made);
    assert_eq!(names(&header.components), ["model"]);
    assert!(names(&header.configs).contains(&"model"));
    let assets = names(&header.assets);
    for file in ["chat_template.jinja", "generation_config.json", "tokenizer.json", "tokenizer_config.json"] {
        assert!(assets.contains(&file), "{file} not in {assets:?}");
    }
    let _ = std::fs::remove_dir_all(root);
}

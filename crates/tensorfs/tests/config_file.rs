//! A component's config file name survives the write: `materialize` and `export_files` render
//! the config under the name the Writer gave (`scheduler_config.json`), not `config.json`, and a
//! rewrite through `base` keeps it. `export_files(None)` renders into a cache-managed folder.
#![cfg(feature = "store")]

use std::fs;
use std::path::PathBuf;

use serde_json::{json, Value};
use tensorfs::format::Dtype;
use tensorfs::store::Store;
use tensorfs::{Client, Config, Format, WriterOptions};

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_written_config_file_name_survives_materialize_export_files_and_rewrite() {
    let dir = Scratch(std::env::temp_dir().join(format!("tfs-config-file-{}", std::process::id())));
    let _ = fs::remove_dir_all(&dir.0);
    let client = Client::new(Config { cache_dir: dir.0.join("cache"), ..Config::from_env() }).unwrap();
    let scheduler = json!({"num_train_timesteps": 1000, "beta_schedule": "scaled_linear"});
    let config = serde_json::from_value(scheduler.clone()).unwrap();

    let mut writer = client.writer(WriterOptions::default()).unwrap();
    writer.component("scheduler", "diffusers.EulerDiscreteScheduler", config, Some("scheduler_config.json")).unwrap();
    let tokenizer = serde_json::from_value(json!({"model_max_length": 77})).unwrap();
    writer.component("tokenizer", "transformers.CLIPTokenizer", tokenizer, Some("tokenizer_config.json")).unwrap();
    let sample = serde_json::from_value(json!({"sample_size": 8})).unwrap();
    let mut unet = writer.component("unet", "diffusers.UNet2DConditionModel", sample, None).unwrap();
    unet.put("w", Dtype::U8, &[4], &mut &[1u8, 2, 3, 4][..]).unwrap();
    let saved = writer.commit().await.unwrap();
    assert_eq!(saved.component("scheduler").unwrap().config_file(), "scheduler_config.json");
    assert_eq!(saved.component("unet").unwrap().config_file(), "config.json");

    let folder = saved.materialize(&dir.0.join("st"), Format::Safetensors, None).await.unwrap();
    let rendered: Value = serde_json::from_slice(&fs::read(folder.join("scheduler/scheduler_config.json")).unwrap()).unwrap();
    assert_eq!(rendered, scheduler);
    assert!(!folder.join("scheduler/config.json").exists());
    assert!(folder.join("unet/config.json").exists());

    let files = saved.component("scheduler").unwrap().export_files(Some(&dir.0.join("files"))).unwrap();
    assert!(files.join("scheduler_config.json").is_file());
    assert!(!files.join("config.json").exists());

    // A rewrite keeps both a redeclared and an inherited component's config file name.
    let mut rewrite = client.writer(WriterOptions { base: Some(&saved), ..WriterOptions::default() }).unwrap();
    let linear = serde_json::from_value(json!({"beta_schedule": "linear"})).unwrap();
    rewrite.component("scheduler", "diffusers.EulerDiscreteScheduler", linear, None).unwrap();
    let rewritten = rewrite.commit().await.unwrap();
    let scheduler = rewritten.component("scheduler").unwrap();
    assert_eq!((scheduler.config_file(), scheduler.config()["beta_schedule"].as_str()), ("scheduler_config.json", Some("linear")));
    assert_eq!(rewritten.component("tokenizer").unwrap().config_file(), "tokenizer_config.json");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn export_files_without_a_directory_renders_a_cache_folder_that_expires() {
    let dir = Scratch(std::env::temp_dir().join(format!("tfs-export-cache-{}", std::process::id())));
    let _ = fs::remove_dir_all(&dir.0);
    let cache = dir.0.join("cache");
    let client = Client::new(Config { cache_dir: cache.clone(), ..Config::from_env() }).unwrap();
    let mut writer = client.writer(WriterOptions::default()).unwrap();
    let config = serde_json::from_value(json!({"model_max_length": 77})).unwrap();
    let mut tokenizer = writer.component("tokenizer", "transformers.CLIPTokenizer", config, Some("tokenizer_config.json")).unwrap();
    tokenizer.asset("vocab.json", br#"{"a":0}"#).unwrap();
    let saved = writer.commit().await.unwrap();
    let component = saved.component("tokenizer").unwrap();

    let managed = component.export_files(None).unwrap();
    assert!(managed.starts_with(cache.join("snapshots")), "{}", managed.display());
    assert_eq!(fs::read(managed.join("vocab.json")).unwrap(), br#"{"a":0}"#);
    assert!(managed.join("tokenizer_config.json").is_file());
    assert_eq!(component.export_files(None).unwrap(), managed, "the folder is reused");
    let explicit = component.export_files(Some(&dir.0.join("mine"))).unwrap();
    drop(saved);
    drop(client);

    // Age the folder's use past the TTL; a collection by the TTL evicts it, never the caller's copy.
    let then = std::time::SystemTime::now() - tensorfs::store::TTL * 2;
    let name = managed.file_name().unwrap().to_str().unwrap();
    fs::File::open(cache.join("uses").join(name)).unwrap().set_modified(then).unwrap();
    platform::durable::open_dir(&managed)
        .unwrap()
        .set_times(fs::FileTimes::new().set_modified(then))
        .unwrap();
    Store::open(&cache).unwrap().collect(u64::MAX, tensorfs::store::TTL).unwrap();
    assert!(!managed.exists(), "an expired export folder outlived the cache TTL");
    assert!(explicit.join("vocab.json").is_file());
}

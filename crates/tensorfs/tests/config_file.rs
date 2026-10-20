//! A component's config file name survives the write: `materialize` and `export_files` render
//! the config under the name the Writer gave (`scheduler_config.json`), not `config.json`.
#![cfg(feature = "store")]

use std::fs;
use std::path::PathBuf;

use serde_json::{json, Value};
use tensorfs::format::Dtype;
use tensorfs::{Client, Config, Format, WriterOptions};

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_written_config_file_name_survives_materialize_and_export_files() {
    let dir = Scratch(std::env::temp_dir().join(format!("tfs-config-file-{}", std::process::id())));
    let _ = fs::remove_dir_all(&dir.0);
    let client = Client::new(Config { cache_dir: dir.0.join("cache"), ..Config::from_env() }).unwrap();
    let scheduler = json!({"num_train_timesteps": 1000, "beta_schedule": "scaled_linear"});
    let config = serde_json::from_value(scheduler.clone()).unwrap();

    let mut writer = client.writer(WriterOptions::default()).unwrap();
    writer.component("scheduler", "diffusers.EulerDiscreteScheduler", config, Some("scheduler_config.json")).unwrap();
    let sample = serde_json::from_value(json!({"sample_size": 8})).unwrap();
    let mut unet = writer.component("unet", "diffusers.UNet2DConditionModel", sample, None).unwrap();
    unet.put("w", Dtype::U8, &[4], &mut &[1u8, 2, 3, 4][..]).unwrap();
    let saved = writer.commit().await.unwrap();

    let folder = saved.materialize(&dir.0.join("st"), Format::Safetensors, None).await.unwrap();
    let rendered: Value = serde_json::from_slice(&fs::read(folder.join("scheduler/scheduler_config.json")).unwrap()).unwrap();
    assert_eq!(rendered, scheduler);
    assert!(!folder.join("scheduler/config.json").exists());
    assert!(folder.join("unet/config.json").exists());

    let files = saved.component("scheduler").unwrap().export_files(&dir.0.join("files")).unwrap();
    assert!(files.join("scheduler_config.json").is_file());
    assert!(!files.join("config.json").exists());
}

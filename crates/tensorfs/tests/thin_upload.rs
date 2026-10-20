//! `upload_folder` of a thin folder (headers and repo files, no `chunks/`): the bytes come
//! from the store, or are already at the endpoint; only an object in neither place refuses.
#![cfg(feature = "store")]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use tensorfs::format::{Dtype, CHUNKS_DIR};
use tensorfs::{Client, Config, Format, Kind, Open, Ref, WriterOptions};

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn scratch() -> Scratch {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("tfs-thin-upload-{}-{n}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    Scratch(dir)
}

fn client(cache: &Path, endpoint: &Path) -> Client {
    Client::new(Config {
        endpoint: format!("file://{}", endpoint.display()),
        cache_dir: cache.to_path_buf(),
        ..Config::from_env()
    })
    .unwrap()
}

/// A copy of `folder` without its `chunks/`.
fn thin(folder: &Path, into: &Path) {
    fs::create_dir_all(into).unwrap();
    for entry in fs::read_dir(folder).unwrap().flatten() {
        if entry.file_name() != CHUNKS_DIR && entry.file_type().unwrap().is_file() {
            fs::copy(entry.path(), into.join(entry.file_name())).unwrap();
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_thin_folder_uploads_from_the_store_or_what_the_endpoint_holds() {
    let dir = scratch();
    let (origin, other) = (dir.0.join("origin"), dir.0.join("other"));
    let weight: Vec<u8> = (0..1 << 20).map(|i| (i * 7 % 251) as u8).collect();

    // A checkpoint in store `a`, its folder made thin: the header and a README only.
    let a = client(&dir.0.join("a"), &origin);
    let mut writer = a.writer(WriterOptions::default()).unwrap();
    writer.component("model", "", Default::default(), None).unwrap().put("w", Dtype::U8, &[weight.len() as u64], &mut &weight[..]).unwrap();
    let saved = writer.commit().await.unwrap();
    let full = saved.materialize(&dir.0.join("full"), Format::Cozytensors, false).await.unwrap();
    let folder = dir.0.join("thin");
    thin(&full, &folder);
    fs::write(folder.join("README.md"), "thin\n").unwrap();
    assert!(!folder.join(CHUNKS_DIR).exists());

    let model: Ref = "acme/thin@1".parse().unwrap();
    let tree = a.upload_folder(&model, &folder, None).await.unwrap();
    let b = client(&dir.0.join("b"), &origin);
    let pulled = b.open("acme/thin@1", &Open { lazy: false, ..Default::default() }).await.unwrap();
    assert_eq!((pulled.id(), pulled.tree()), (saved.id(), Some(&tree)));
    assert_eq!(pulled.component("model").unwrap().read_decoded("w").unwrap(), weight);

    // Store `c` holds none of it; the endpoint does.
    let c = client(&dir.0.join("c"), &origin);
    let again: Ref = "acme/again@1".parse().unwrap();
    assert_eq!(c.upload_folder(&again, &folder, None).await.unwrap(), tree);

    // Neither store `d` nor a fresh endpoint holds the chunk; a local repo needs it here too.
    let d = client(&dir.0.join("d"), &other);
    let refused = d.upload_folder(&model, &folder, None).await.unwrap_err();
    assert_eq!(refused.kind, Kind::NotFound, "{}", refused.detail);
    let local: Ref = "local/thin@1".parse().unwrap();
    let refused = d.upload_folder(&local, &folder, None).await.unwrap_err();
    assert_eq!(refused.kind, Kind::NotFound, "{}", refused.detail);
}

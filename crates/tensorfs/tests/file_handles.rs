//! Saved native handles survive their producer process and do not own store lifetime.
#![cfg(feature = "store")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tensorfs::format::{Config as ComponentConfig, Dtype, ObjectRef};
use tensorfs::{Client, Config, Fetch, Open, WriterOptions};

const DATA: &[u8] = b"native-file-process-proof";

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn client(cache: &Path) -> Client {
    Client::new(Config {
        endpoint: "http://127.0.0.1:9".into(),
        cache_dir: cache.to_path_buf(),
        mirror: None,
        token: None,
        ca_der: None,
    })
    .unwrap()
}

fn read_in_child(cache: &Path, source: &str, expected: &str) {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "native_handle_reader_child",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("TENSORFS_HANDLE_PROOF_CACHE", cache)
        .env("TENSORFS_HANDLE_PROOF_SOURCE", source)
        .env("TENSORFS_HANDLE_PROOF_ID", expected)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if child.try_wait().unwrap().is_some() {
            break;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            let output = child.wait_with_output().unwrap();
            panic!(
                "owned native reader timed out and was reaped: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "fresh reader failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("FILE_HANDLE_READ_OK"));
}

#[test]
#[ignore = "subprocess helper invoked twice by the real saved-handle process test"]
fn native_handle_reader_child() {
    let cache = PathBuf::from(std::env::var_os("TENSORFS_HANDLE_PROOF_CACHE").unwrap());
    let source = std::env::var("TENSORFS_HANDLE_PROOF_SOURCE").unwrap();
    let expected = std::env::var("TENSORFS_HANDLE_PROOF_ID").unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let checkpoint = client(&cache)
            .open(
                &source,
                &Open {
                    local_files_only: true,
                    ..Open::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(checkpoint.id().id(), expected);
        checkpoint.fetch(&Fetch::default()).await.unwrap();
        assert_eq!(
            checkpoint
                .component("model")
                .unwrap()
                .read("value")
                .unwrap(),
            DATA
        );
    });
    println!("FILE_HANDLE_READ_OK {expected}");
}

#[tokio::test(flavor = "current_thread")]
async fn saved_handle_reopens_after_all_producers_drop_and_deletion_keeps_content() {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let scratch = Scratch(Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "native-file-process-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )));
    fs::create_dir(&scratch.0).unwrap();
    let cache = scratch.0.join("cache");
    let path = scratch.0.join("model.cozytensors");
    let owner = client(&cache);
    let mut writer = owner
        .writer(WriterOptions {
            dest: Some(&path),
            ..WriterOptions::default()
        })
        .unwrap();
    let mut input = DATA;
    writer
        .component("model", "test.File", ComponentConfig::new())
        .unwrap()
        .put("value", Dtype::U8, &[DATA.len() as u64], &mut input)
        .unwrap();
    let checkpoint = writer.commit().await.unwrap();
    let expected = checkpoint.id().id();
    assert_eq!(ObjectRef::of(&fs::read(&path).unwrap()).id(), expected);
    assert!(!scratch.0.join("chunks").exists());
    drop(checkpoint);
    drop(writer);
    drop(owner);

    let moved = scratch.0.join("renamed-literal-file");
    fs::rename(&path, &moved).unwrap();
    read_in_child(&cache, moved.to_str().unwrap(), &expected);
    fs::remove_file(&moved).unwrap();
    read_in_child(&cache, &expected, &expected);
}

//! External consumers of the managed public write/convert/export doors on tiny real files.
#![cfg(feature = "store")]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use tensorfs::format::{Body, Component, Dtype, Header, Location, ObjectRef, Tensor};
use tensorfs::{export, Client, Config, Fetch, Format, Kind, Open, Ref, Selection};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
                "public-writer-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(e) => panic!("create scratch: {e}"),
            }
        }
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

const WEIGHT: &[u8] = &[3, 7, 11, 19];
const README: &[u8] = b"\x85repository bytes are not a cozytensors header";
const ASSET: &[u8] = br#"{"vocabulary":["one","two"]}"#;

fn fixture(root: &Path) -> PathBuf {
    let folder = root.join("plain");
    fs::create_dir_all(folder.join("chunks")).unwrap();
    let chunk = ObjectRef::of(WEIGHT);
    let asset = ObjectRef::of(ASSET);
    fs::write(folder.join("chunks").join(&chunk.sha256), WEIGHT).unwrap();
    fs::write(folder.join("chunks").join(&asset.sha256), ASSET).unwrap();
    let header = Header::new(
        vec![chunk],
        None,
        vec![Component {
            name: "model".into(),
            class: "transformers.TestModel".into(),
            config: serde_json::from_str(r#"{"width":4,"optional":null}"#).unwrap(),
            tensors: vec![Tensor {
                key: "w".into(),
                dtype: Dtype::U8,
                shape: vec![4],
                body: Body::Data(Location {
                    chunk: 0,
                    offset: 0,
                    length: 4,
                }),
            }],
            assets: BTreeMap::from([("tokenizer.json".into(), asset)]),
            ..Default::default()
        }],
        Default::default(),
    )
    .unwrap();
    fs::write(folder.join("model.cozytensors"), header.to_cbor().unwrap()).unwrap();
    fs::write(folder.join("README.bin"), README).unwrap();
    folder
}
fn client(cache: &Path, endpoint: &Path) -> Client {
    let mut config = Config::from_env();
    config.cache_dir = cache.to_path_buf();
    config.endpoint = endpoint.to_str().unwrap().to_owned();
    config.mirror = None;
    config.token = None;
    Client::new(config).unwrap()
}
fn check_export(folder: &Path) {
    assert_eq!(fs::read(folder.join("README.bin")).unwrap(), README);
    assert_eq!(fs::read(folder.join("tokenizer.json")).unwrap(), ASSET);
    let config: serde_json::Value =
        serde_json::from_slice(&fs::read(folder.join("config.json")).unwrap()).unwrap();
    assert_eq!(config, serde_json::json!({"width":4,"optional":null}));
    let bytes = fs::read(folder.join("model.safetensors")).unwrap();
    let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + len]).unwrap();
    assert_eq!(header["w"]["dtype"], "U8");
    assert_eq!(header["w"]["shape"], serde_json::json!([4]));
    let offsets = header["w"]["data_offsets"].as_array().unwrap();
    assert_eq!(
        &bytes[8 + len + offsets[0].as_u64().unwrap() as usize
            ..8 + len + offsets[1].as_u64().unwrap() as usize],
        WEIGHT
    );
}

#[tokio::test]
async fn public_export_keeps_plain_folder_repository_bytes_without_admitting_payload_on_open() {
    let scratch = Scratch::new();
    let folder = fixture(&scratch.0);
    let client = client(&scratch.0.join("cache"), &scratch.0.join("endpoint"));
    let checkpoint = client
        .open(folder.to_str().unwrap(), &Open::default())
        .await
        .unwrap();
    assert!(
        !scratch
            .0
            .join("cache")
            .join(ObjectRef::of(WEIGHT).key())
            .exists(),
        "metadata-only open admitted weight payload"
    );
    assert!(
        !scratch
            .0
            .join("cache")
            .join(ObjectRef::of(ASSET).key())
            .exists(),
        "metadata-only open admitted asset payload"
    );
    drop(client);
    checkpoint.fetch(&Fetch::default()).await.unwrap();
    let destination = scratch.0.join("export");
    let paths = export(&checkpoint, &destination, Format::Safetensors, None).unwrap();
    assert!(paths.iter().any(|p| p == "README.bin"));
    check_export(&destination);
}

#[tokio::test]
async fn public_export_requires_selected_repo_bytes_and_owns_them_after_client_drop() {
    let scratch = Scratch::new();
    let folder = fixture(&scratch.0);
    let endpoint = scratch.0.join("endpoint");
    let writer = client(&scratch.0.join("source-cache"), &endpoint);
    let reference: Ref = "acme/tiny".parse().unwrap();
    writer
        .upload_folder(&reference, &folder, None)
        .await
        .unwrap();
    let reader = client(&scratch.0.join("reader-cache"), &endpoint);
    let checkpoint = reader.open("acme/tiny", &Open::default()).await.unwrap();
    let destination = scratch.0.join("export");
    assert_eq!(
        export(&checkpoint, &destination, Format::Safetensors, None)
            .unwrap_err()
            .kind,
        Kind::NotFound
    );
    assert!(
        !destination.exists(),
        "failed availability check created an export"
    );
    checkpoint
        .fetch(&Fetch {
            selection: Selection::Assets {
                component: "model".into(),
                names: None,
            },
            ..Fetch::default()
        })
        .await
        .unwrap();
    assert!(checkpoint
        .missing(&Selection::All)
        .unwrap()
        .contains(&ObjectRef::of(README)));
    assert!(
        checkpoint
            .missing(&Selection::All)
            .unwrap()
            .contains(&ObjectRef::of(WEIGHT)),
        "asset selection fetched weights"
    );
    assert_eq!(
        checkpoint
            .component("model")
            .unwrap()
            .asset("tokenizer.json")
            .unwrap(),
        ASSET
    );
    checkpoint.fetch(&Fetch::default()).await.unwrap();
    assert!(checkpoint.missing(&Selection::All).unwrap().is_empty());
    drop(reader);
    export(&checkpoint, &destination, Format::Safetensors, None).unwrap();
    check_export(&destination);
}

#[tokio::test]
async fn same_header_repository_trees_keep_distinct_folders_through_public_upload_and_export() {
    let scratch = Scratch::new();
    fs::create_dir_all(scratch.0.join("first")).unwrap();
    fs::create_dir_all(scratch.0.join("second")).unwrap();
    let first = fixture(&scratch.0.join("first"));
    let second = fixture(&scratch.0.join("second"));
    let second_readme = b"\x85another repository using exactly the same weights";
    fs::write(second.join("README.bin"), second_readme).unwrap();
    let endpoint = scratch.0.join("endpoint");
    let source = client(&scratch.0.join("source-cache"), &endpoint);
    source
        .upload_folder(&"acme/first".parse().unwrap(), &first, None)
        .await
        .unwrap();
    source
        .upload_folder(&"acme/second".parse().unwrap(), &second, None)
        .await
        .unwrap();
    let reader = client(&scratch.0.join("reader-cache"), &endpoint);
    let a = reader.open("acme/first", &Open::default()).await.unwrap();
    let b = reader.open("acme/second", &Open::default()).await.unwrap();
    assert_eq!(a.id(), b.id());
    assert_ne!(a.tree(), b.tree());
    a.fetch(&Fetch::default()).await.unwrap();
    b.fetch(&Fetch::default()).await.unwrap();
    let a_folder = a.folder().unwrap();
    let b_folder = b.folder().unwrap();
    assert_ne!(a_folder, b_folder, "same-header repository files collided");
    assert_eq!(
        a_folder.file_name().unwrap(),
        a.tree().unwrap().sha256.as_str()
    );
    assert_eq!(
        b_folder.file_name().unwrap(),
        b.tree().unwrap().sha256.as_str()
    );
    assert_eq!(fs::read(a_folder.join("README.bin")).unwrap(), README);
    assert_eq!(
        fs::read(b_folder.join("README.bin")).unwrap(),
        second_readme
    );
    assert_eq!(
        reader
            .upload_folder(&"local/roundtrip".parse().unwrap(), &a_folder, None)
            .await
            .unwrap(),
        *a.tree().unwrap()
    );
    let roundtrip = reader
        .open(
            "local/roundtrip",
            &Open {
                local_files_only: true,
                ..Open::default()
            },
        )
        .await
        .unwrap();
    drop(reader);
    drop(a);
    drop(b);
    let destination = scratch.0.join("export-roundtrip");
    export(&roundtrip, &destination, Format::Safetensors, None).unwrap();
    check_export(&destination);
}

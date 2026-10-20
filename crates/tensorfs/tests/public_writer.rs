//! External consumers of the managed public write/convert/export doors on tiny real files.
#![cfg(feature = "store")]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use tensorfs::format::{Body, Component, Dtype, Header, Location, ObjectRef, Tensor};
use tensorfs::{export, Client, Config, Fetch, Format, Kind, Open, Ref, Selection, WriterOptions};

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
    check_export_readme(folder, "README.bin");
}
fn check_export_readme(folder: &Path, readme: &str) {
    assert_eq!(fs::read(folder.join(readme)).unwrap(), README);
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
    let tree = writer
        .upload_folder(&reference, &folder, None)
        .await
        .unwrap();
    let reader = client(&scratch.0.join("reader-cache"), &endpoint);
    let checkpoint = reader
        .open(&format!("acme/tiny@{}", tree.id()), &Open::default())
        .await
        .unwrap();
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
    let first_tree = source
        .upload_folder(&"acme/first".parse().unwrap(), &first, None)
        .await
        .unwrap();
    let second_tree = source
        .upload_folder(&"acme/second".parse().unwrap(), &second, None)
        .await
        .unwrap();
    let reader = client(&scratch.0.join("reader-cache"), &endpoint);
    let a = reader
        .open(&format!("acme/first@{}", first_tree.id()), &Open::default())
        .await
        .unwrap();
    let b = reader
        .open(
            &format!("acme/second@{}", second_tree.id()),
            &Open::default(),
        )
        .await
        .unwrap();
    assert_eq!(a.id(), b.id());
    assert_ne!(a.tree(), b.tree());
    a.fetch(&Fetch::default()).await.unwrap();
    b.fetch(&Fetch::default()).await.unwrap();
    let a_folder = scratch.0.join("portable-first");
    let b_folder = scratch.0.join("portable-second");
    export(&a, &a_folder, Format::Cozytensors, None).unwrap();
    export(&b, &b_folder, Format::Cozytensors, None).unwrap();
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
            &format!("local/roundtrip@{}", first_tree.id()),
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

#[tokio::test(flavor = "current_thread")]
async fn public_writer_resumes_reopens_components_and_keeps_encoded_bytes_and_order() {
    let scratch = Scratch::new();
    let client = client(&scratch.0.join("cache"), &scratch.0.join("endpoint"));
    let config = serde_json::from_str(r#"{"width":4}"#).unwrap();
    let mut first = client
        .writer(WriterOptions {
            resume: Some("tiny-resume"),
            ..WriterOptions::default()
        })
        .unwrap();
    first
        .component("model", "test.Model", config)
        .unwrap()
        .put("first", Dtype::U8, &[4], &mut std::io::Cursor::new(WEIGHT))
        .unwrap();
    drop(first);
    let mut writer = client
        .writer(WriterOptions {
            resume: Some("tiny-resume"),
            ..WriterOptions::default()
        })
        .unwrap();
    assert!(writer.has("model", "first").unwrap());
    // A resumed producer declares its logical metadata; reopening a declared handle never resets it.
    writer
        .component(
            "model",
            "test.Model",
            serde_json::from_str(r#"{"width":8}"#).unwrap(),
        )
        .unwrap();
    assert_eq!(
        writer.component_writer("undeclared").err().unwrap().kind,
        Kind::NotFound
    );
    let mut a: &[u8] = &[5, 6, 7, 8];
    let mut b: &[u8] = &[9, 10, 11, 12];
    {
        let mut component = writer.component_writer("model").unwrap();
        component
            .put_encoded(
                "packed",
                Dtype::F32,
                &[2],
                tensorfs::format::Encoding {
                    scheme: "acme-q".into(),
                    params: Default::default(),
                },
                vec![
                    tensorfs::raw::PartData {
                        role: "weight".into(),
                        dtype: Dtype::U8,
                        shape: vec![4],
                        data: &mut a,
                    },
                    tensorfs::raw::PartData {
                        role: "scale".into(),
                        dtype: Dtype::U8,
                        shape: vec![4],
                        data: &mut b,
                    },
                ],
            )
            .unwrap();
        component.asset("tokenizer.json", ASSET).unwrap();
        component.order(&["packed".into(), "first".into()]).unwrap();
    }
    let checkpoint = writer.commit().await.unwrap();
    assert_eq!(writer.committed().unwrap().id(), checkpoint.id());
    assert_eq!(
        checkpoint
            .header()
            .components()
            .iter()
            .find(|c| c.name == "model")
            .unwrap()
            .tensors
            .iter()
            .map(|t| t.key.as_str())
            .collect::<Vec<_>>(),
        ["packed", "first"]
    );
    assert_eq!(checkpoint.component("model").unwrap().config()["width"], 8);
    let mut derived = client
        .writer(WriterOptions {
            base: Some(&checkpoint),
            ..WriterOptions::default()
        })
        .unwrap();
    derived
        .component_writer("model")
        .unwrap()
        .copy(
            "first",
            checkpoint.component("model").unwrap(),
            Some("copied"),
        )
        .unwrap();
    let copied = derived.commit().await.unwrap();
    let child = copied.component("model").unwrap().clone();
    drop(derived);
    drop(copied);
    drop(checkpoint);
    drop(writer);
    drop(client);
    assert_eq!(child.read("copied").unwrap(), WEIGHT);
    assert_eq!(child.asset("tokenizer.json").unwrap(), ASSET);
    let mut bytes = [0; 4];
    child
        .read_part_into("packed", "scale", 0, &mut bytes)
        .unwrap();
    assert_eq!(bytes, [9, 10, 11, 12]);
}

#[tokio::test(flavor = "current_thread")]
async fn file_writer_publishes_only_exact_header_and_keeps_content_after_drop_move_and_gc() {
    let scratch = Scratch::new();
    let cache = scratch.0.join("cache");
    let client = client(&cache, &scratch.0.join("endpoint"));
    let dest = scratch
        .0
        .join("published")
        .join("arbitrary-name.cozytensors");
    assert_eq!(
        client
            .writer(WriterOptions {
                dest: Some(&dest),
                variant: Some("bf16"),
                ..WriterOptions::default()
            })
            .err()
            .unwrap()
            .kind,
        Kind::Invalid
    );
    let mut first = client
        .writer(WriterOptions {
            dest: Some(&dest),
            resume: Some("reused-key"),
            ..WriterOptions::default()
        })
        .unwrap();
    first
        .component("model", "test.Model", Default::default())
        .unwrap()
        .put("w", Dtype::U8, &[4], &mut std::io::Cursor::new(WEIGHT))
        .unwrap();
    assert!(!dest.exists(), "header was published before commit");
    let old = first.commit().await.unwrap();
    assert!(old.tree().is_none() && old.named().is_none() && old.variant().is_none());
    let original_header = fs::read(&dest).unwrap();
    assert_eq!(ObjectRef::of(&original_header), *old.id());
    assert!(!dest.parent().unwrap().join("chunks").exists());
    assert_eq!(fs::read_dir(dest.parent().unwrap()).unwrap().count(), 1);
    let pin: serde_json::Value = serde_json::from_slice(
        &fs::read(cache.join("pins").join(format!("{}.json", old.id().sha256))).unwrap(),
    )
    .unwrap();
    assert!(
        pin.get("owner").is_none(),
        "a persistent header got a disposable cache pin"
    );
    // The content digest is an ordinary operator pin id, not a reserved namespace.
    let operator = tensorfs::store2::Store::open(&cache).unwrap();
    let manual = operator
        .put_bytes(b"operator-owned unrelated content")
        .unwrap();
    let mut existing = operator.get_pin(&old.id().sha256).unwrap().unwrap();
    existing.owner = Some("ordinary operator".into());
    existing.roots.push(manual.clone());
    operator.pin(&old.id().sha256, &existing).unwrap();
    let another_path = scratch.0.join("another-header.cozytensors");
    let mut identical = client
        .writer(WriterOptions {
            dest: Some(&another_path),
            ..Default::default()
        })
        .unwrap();
    identical
        .component("model", "test.Model", Default::default())
        .unwrap()
        .put("w", Dtype::U8, &[4], &mut std::io::Cursor::new(WEIGHT))
        .unwrap();
    let same = identical.commit().await.unwrap();
    assert_eq!(same.id(), old.id());
    let preserved = operator.get_pin(&old.id().sha256).unwrap().unwrap();
    assert_eq!(preserved.owner.as_deref(), Some("ordinary operator"));
    assert!(preserved.roots.contains(&manual) && preserved.roots.contains(old.id()));
    drop(identical);
    drop(same);
    drop(operator);
    let mut next = client
        .writer(WriterOptions {
            dest: Some(&dest),
            resume: Some("reused-key"),
            ..WriterOptions::default()
        })
        .unwrap();
    let replacement = [2, 4, 6, 8];
    next.component("model", "test.Model", Default::default())
        .unwrap()
        .put("w", Dtype::U8, &[4], &mut &replacement[..])
        .unwrap();
    let new = next.commit().await.unwrap();
    assert_ne!(new.id(), old.id());
    assert_eq!(
        fs::read(&dest).unwrap().len(),
        original_header.len(),
        "fixture must exercise same-length replacement"
    );
    assert_eq!(old.component("model").unwrap().read("w").unwrap(), WEIGHT);
    let old_id = old.id().clone();
    let new_id = new.id().clone();
    let moved = scratch.0.join("moved-header.cozytensors");
    fs::rename(&dest, &moved).unwrap();
    drop(first);
    drop(next);
    drop(old);
    drop(new);
    drop(client);
    // Operator maintenance is used solely to exercise the real collector, not to create output.
    let operator = tensorfs::store2::Store::open(&cache)
        .unwrap()
        .with_grace(std::time::Duration::ZERO);
    let disposable = operator
        .put_bytes(b"unreferenced positive eviction control")
        .unwrap();
    operator.collect(0, std::time::Duration::ZERO).unwrap();
    assert!(
        !cache.join(disposable.key()).exists(),
        "collector positive control did not run"
    );
    assert!(cache.join(old_id.key()).exists() && cache.join(new_id.key()).exists());
    assert!(
        cache.join(manual.key()).exists(),
        "durable promotion lost an unrelated operator root"
    );
    drop(operator);
    let reopened = self::client(&cache, &scratch.0.join("endpoint"))
        .open(
            moved.to_str().unwrap(),
            &Open {
                local_files_only: true,
                ..Open::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(reopened.id(), &new_id);
    assert_eq!(
        reopened.component("model").unwrap().read("w").unwrap(),
        replacement
    );
}

#[tokio::test(flavor = "current_thread")]
async fn failed_file_publication_still_exposes_the_committed_local_checkpoint() {
    let scratch = Scratch::new();
    let client = client(&scratch.0.join("cache"), &scratch.0.join("endpoint"));
    let dest = scratch.0.join("a-directory");
    fs::create_dir(&dest).unwrap();
    let mut writer = client
        .writer(WriterOptions {
            dest: Some(&dest),
            ..WriterOptions::default()
        })
        .unwrap();
    writer
        .component("model", "test.Model", Default::default())
        .unwrap()
        .put("w", Dtype::U8, &[4], &mut std::io::Cursor::new(WEIGHT))
        .unwrap();
    assert_eq!(writer.commit().await.unwrap_err().kind, Kind::Io);
    let retained = writer
        .committed()
        .expect("sealed local result lost on file publication failure");
    drop(writer);
    drop(client);
    assert_eq!(
        retained.component("model").unwrap().read("w").unwrap(),
        WEIGHT
    );
    assert!(dest.is_dir());
}

#[tokio::test(flavor = "current_thread")]
async fn portable_export_preflights_full_closure_and_atomically_replaces_shared_inodes() {
    let scratch = Scratch::new();
    let folder = fixture(&scratch.0);
    let client = client(&scratch.0.join("cache"), &scratch.0.join("endpoint"));
    let checkpoint = client
        .open(folder.to_str().unwrap(), &Open::default())
        .await
        .unwrap();
    let header = fs::read(folder.join("model.cozytensors")).unwrap();
    let portable = scratch.0.join("portable");
    fs::create_dir_all(portable.join("chunks")).unwrap();
    fs::hard_link(folder.join("README.bin"), portable.join("README.bin")).unwrap();
    fs::hard_link(
        folder.join("chunks").join(&ObjectRef::of(WEIGHT).sha256),
        portable.join("chunks").join(&ObjectRef::of(WEIGHT).sha256),
    )
    .unwrap();
    fs::hard_link(
        folder.join("model.cozytensors"),
        portable.join("model.cozytensors"),
    )
    .unwrap();
    export(&checkpoint, &portable, Format::Cozytensors, None).unwrap();
    export(&checkpoint, &folder, Format::Cozytensors, None).unwrap();
    assert_eq!(fs::read(folder.join("README.bin")).unwrap(), README);
    assert_eq!(
        fs::read(folder.join("chunks").join(&ObjectRef::of(WEIGHT).sha256)).unwrap(),
        WEIGHT
    );
    assert_eq!(
        fs::read(portable.join("model.cozytensors")).unwrap(),
        header
    );
    drop(checkpoint);
    drop(client);
    let reopened = self::client(&scratch.0.join("empty-cache"), &scratch.0.join("endpoint"))
        .open(portable.to_str().unwrap(), &Open::default())
        .await
        .unwrap();
    assert_eq!(
        reopened.component("model").unwrap().read("w").unwrap(),
        WEIGHT
    );
    assert_eq!(
        reopened
            .component("model")
            .unwrap()
            .asset("tokenizer.json")
            .unwrap(),
        ASSET
    );
    assert!(!scratch
        .0
        .join("empty-cache")
        .join(ObjectRef::of(WEIGHT).key())
        .exists());
    fs::remove_file(portable.join("chunks").join(&ObjectRef::of(WEIGHT).sha256)).unwrap();
    let incomplete = scratch.0.join("refused");
    assert_eq!(
        export(&reopened, &incomplete, Format::Cozytensors, None)
            .unwrap_err()
            .kind,
        Kind::NotFound
    );
    assert!(!incomplete.join("model.cozytensors").exists());
    assert_eq!(
        export(
            &reopened,
            &incomplete,
            Format::Cozytensors,
            Some(Dtype::F32)
        )
        .unwrap_err()
        .kind,
        Kind::Invalid
    );
}

#[tokio::test(flavor = "current_thread")]
async fn literal_header_fetches_selected_missing_payload_online_but_not_offline_or_neighbor_repo_files(
) {
    let scratch = Scratch::new();
    let folder = fixture(&scratch.0);
    let endpoint = scratch.0.join("endpoint");
    let producer = client(&scratch.0.join("source-cache"), &endpoint);
    producer
        .upload_folder(&"acme/source".parse().unwrap(), &folder, None)
        .await
        .unwrap();
    let header = scratch.0.join("only-header.cozytensors");
    fs::copy(folder.join("model.cozytensors"), &header).unwrap();
    fs::write(
        scratch.0.join("README.bin"),
        b"neighbor must not be inferred",
    )
    .unwrap();
    let offline = client(&scratch.0.join("offline-cache"), &endpoint)
        .open(
            header.to_str().unwrap(),
            &Open {
                local_files_only: true,
                ..Open::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(
        offline.fetch(&Fetch::default()).await.unwrap_err().kind,
        Kind::NotFound
    );
    let cache = scratch.0.join("online-cache");
    let online = client(&cache, &endpoint)
        .open(header.to_str().unwrap(), &Open::default())
        .await
        .unwrap();
    online
        .fetch(&Fetch {
            selection: Selection::Assets {
                component: "model".into(),
                names: None,
            },
            ..Fetch::default()
        })
        .await
        .unwrap();
    assert_eq!(
        online
            .component("model")
            .unwrap()
            .asset("tokenizer.json")
            .unwrap(),
        ASSET
    );
    assert!(!cache.join(ObjectRef::of(WEIGHT).key()).exists());
    online.fetch(&Fetch::default()).await.unwrap();
    let output = scratch.0.join("literal-portable");
    export(&online, &output, Format::Cozytensors, None).unwrap();
    assert!(!output.join("README.bin").exists());
    assert_eq!(
        online.component("model").unwrap().read("w").unwrap(),
        WEIGHT
    );
}

#[tokio::test(flavor = "current_thread")]
async fn public_streaming_writer_uploads_sealed_asset_before_commit_and_keeps_failed_local_result()
{
    let scratch = Scratch::new();
    let endpoint = scratch.0.join("endpoint");
    let client = client(&scratch.0.join("cache"), &endpoint);
    let mut writer = client
        .writer(WriterOptions {
            repo_id: Some("acme/streamed"),
            ..WriterOptions::default()
        })
        .unwrap();
    writer
        .component("model", "test.Model", Default::default())
        .unwrap()
        .asset("tokenizer.json", ASSET)
        .unwrap();
    let uploaded = endpoint.join(ObjectRef::of(ASSET).key());
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while !uploaded.exists() {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(fs::read(&uploaded).unwrap(), ASSET);
    assert!(
        !endpoint
            .join(tensorfs::format::repo_key("acme/streamed"))
            .exists(),
        "repo was published before commit"
    );
    writer
        .component_writer("model")
        .unwrap()
        .put("w", Dtype::U8, &[4], &mut std::io::Cursor::new(WEIGHT))
        .unwrap();
    let checkpoint = writer.commit().await.unwrap();
    let tree = checkpoint.tree().unwrap().clone();
    let reader = self::client(&scratch.0.join("reader"), &endpoint);
    let remote = reader
        .open(&format!("acme/streamed@{}", tree.id()), &Open::default())
        .await
        .unwrap();
    remote.fetch(&Fetch::default()).await.unwrap();
    assert_eq!(
        remote.component("model").unwrap().read("w").unwrap(),
        WEIGHT
    );
    let broken = scratch.0.join("not-a-directory");
    fs::write(&broken, b"broken endpoint").unwrap();
    let bad = self::client(&scratch.0.join("failed-cache"), &broken);
    let mut failed = bad
        .writer(WriterOptions {
            repo_id: Some("acme/failed"),
            ..WriterOptions::default()
        })
        .unwrap();
    failed
        .component("model", "test.Model", Default::default())
        .unwrap()
        .put("w", Dtype::U8, &[4], &mut std::io::Cursor::new(WEIGHT))
        .unwrap();
    assert!(failed.commit().await.is_err());
    let local = failed
        .committed()
        .expect("stream failure lost sealed local checkpoint");
    let failed_tree = local.tree().unwrap().clone();
    let failed_cache = scratch.0.join("failed-cache");
    let operator = tensorfs::store2::Store::open(&failed_cache)
        .unwrap()
        .with_grace(std::time::Duration::ZERO);
    let durable = operator
        .get_pin(&failed_tree.sha256)
        .unwrap()
        .expect("failed upload has no durable retry root");
    assert!(durable.owner.is_none() && durable.roots.contains(&failed_tree));
    failed.abandon().unwrap();
    drop(bad);
    assert_eq!(local.component("model").unwrap().read("w").unwrap(), WEIGHT);
    drop(local);
    // Remove only the disposable roots: a failed result must survive independently of TTL
    // and of the dropped Checkpoint/Writer native owners.
    for (name, pin) in operator.pins().unwrap() {
        if pin.owner.as_deref() == Some("tensorfs.cache/1") {
            assert!(operator.unpin(&name).unwrap());
        }
    }
    let disposable = operator
        .put_bytes(b"failed-upload collector positive control")
        .unwrap();
    operator.collect(0, std::time::Duration::ZERO).unwrap();
    assert!(!failed_cache.join(disposable.key()).exists());
    drop(operator);
    let recovered = self::client(&failed_cache, &broken)
        .open(
            &failed_tree.id(),
            &Open {
                local_files_only: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(
        recovered.component("model").unwrap().read("w").unwrap(),
        WEIGHT
    );
}

struct WriterRefusal(tensorfs::Failure);
#[async_trait::async_trait]
impl tensorfs::TokenProvider for WriterRefusal {
    async fn headers(
        &self,
        _request: &tensorfs::Request<'_>,
    ) -> tensorfs::Result<Vec<(String, String)>> {
        Err(self.0.clone())
    }
}
#[tokio::test(flavor = "current_thread")]
async fn writer_provider_error_keeps_the_exact_family_and_detail_including_empty_and_literal_prefix(
) {
    let scratch = Scratch::new();
    for kind in [Kind::Denied, Kind::Conflict, Kind::Invalid] {
        for detail in ["", "CREDENTIAL_REQUIRED: literal caller detail"] {
            let mut config = Config::from_env();
            config.cache_dir = scratch
                .0
                .join(format!("cache-{}-{}", kind.code(), detail.len()));
            config.endpoint = "http://127.0.0.1:1".into();
            config.mirror = None;
            config.token = Some(std::sync::Arc::new(WriterRefusal(tensorfs::Failure {
                kind,
                detail: detail.into(),
            })));
            let client = Client::new(config).unwrap();
            let mut writer = client
                .writer(WriterOptions {
                    repo_id: Some("acme/refusal"),
                    ..WriterOptions::default()
                })
                .unwrap();
            writer
                .component("model", "test.Model", Default::default())
                .unwrap()
                .put("w", Dtype::U8, &[4], &mut std::io::Cursor::new(WEIGHT))
                .unwrap();
            let error = tokio::time::timeout(std::time::Duration::from_secs(5), writer.commit())
                .await
                .unwrap()
                .unwrap_err();
            assert_eq!(error.kind, kind);
            assert_eq!(error.detail, detail);
            let local = writer.committed().unwrap();
            writer.abandon().unwrap();
            assert_eq!(local.component("model").unwrap().read("w").unwrap(), WEIGHT);
        }
    }
}

fn source_safetensors() -> Vec<u8> {
    let mut header = serde_json::to_vec(
        &serde_json::json!({"w":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}),
    )
    .unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(WEIGHT);
    bytes
}
fn conversion_source(root: &Path) -> PathBuf {
    let folder = root.join("hf-source");
    fs::create_dir_all(&folder).unwrap();
    fs::write(folder.join("model.safetensors"), source_safetensors()).unwrap();
    fs::write(
        folder.join("config.json"),
        br#"{"width":4,"optional":null}"#,
    )
    .unwrap();
    fs::write(folder.join("tokenizer.json"), ASSET).unwrap();
    fs::write(folder.join("README.md"), README).unwrap();
    folder
}
fn actual_headers(root: &Path) -> Vec<PathBuf> {
    let mut headers = Vec::new();
    let Ok(entries) = fs::read_dir(root) else {
        return headers;
    };
    for entry in entries {
        let path = entry.unwrap().path();
        if path.is_dir() {
            headers.extend(actual_headers(&path));
        } else if tensorfs::format::Header::from_cbor(&fs::read(&path).unwrap()).is_ok() {
            headers.push(path);
        }
    }
    headers
}

#[tokio::test(flavor = "current_thread")]
async fn public_convert_resolves_metadata_then_owns_repo_files_through_portable_upload_and_export()
{
    let scratch = Scratch::new();
    let source = conversion_source(&scratch.0);
    let cache = scratch.0.join("cache");
    let endpoint = scratch.0.join("endpoint");
    let client = client(&cache, &endpoint);
    assert_eq!(
        client
            .resolve_source(source.to_str().unwrap())
            .await
            .unwrap(),
        source.canonicalize().unwrap().display().to_string()
    );
    assert!(
        actual_headers(&cache.join("blobs")).is_empty(),
        "resolve_source admitted model bytes"
    );
    let stages = std::sync::Mutex::new(Vec::new());
    let progress = |stage: &str, held, total| {
        stages.lock().unwrap().push((stage.to_owned(), held, total));
        Ok(())
    };
    let checkpoint = client
        .convert(
            &[source.to_str().unwrap().into()],
            &tensorfs::Convert {
                progress: Some(&progress),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(stages.lock().unwrap().last().unwrap().0, "imported");
    assert_eq!(checkpoint.variant(), Some("u8"));
    let child = checkpoint.component("model").unwrap().clone();
    let portable = scratch.0.join("converted-portable");
    export(&checkpoint, &portable, Format::Cozytensors, None).unwrap();
    assert_eq!(fs::read(portable.join("README.md")).unwrap(), README);
    let tree = client
        .upload_folder(&"local/converted".parse().unwrap(), &portable, None)
        .await
        .unwrap();
    assert_eq!(checkpoint.tree(), Some(&tree));
    let reopened = client
        .open(
            &format!("local/converted@{}", tree.id()),
            &Open {
                local_files_only: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    drop(checkpoint);
    drop(client);
    let result = scratch.0.join("converted-hf");
    export(&reopened, &result, Format::Safetensors, None).unwrap();
    check_export_readme(&result, "README.md");
    drop(reopened);
    assert_eq!(child.read("w").unwrap(), WEIGHT);
    assert_eq!(child.asset("tokenizer.json").unwrap(), ASSET);
}

#[tokio::test(flavor = "current_thread")]
async fn public_convert_result_callback_can_refuse_the_final_stage_and_returns_after_worker_drain()
{
    let scratch = Scratch::new();
    let source = conversion_source(&scratch.0);
    for (index, detail) in ["", "CREDENTIAL_REQUIRED: exact final callback text"]
        .iter()
        .enumerate()
    {
        let cache = scratch.0.join(format!("cache-{index}"));
        let client = client(&cache, &scratch.0.join("endpoint"));
        let stages = std::sync::Mutex::new(Vec::new());
        let progress = |stage: &str, _held, _total| {
            stages.lock().unwrap().push(stage.to_owned());
            if stage == "imported" {
                Err(tensorfs::Failure {
                    kind: Kind::Conflict,
                    detail: (*detail).into(),
                })
            } else {
                Ok(())
            }
        };
        let error = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            client.convert(
                &[source.to_str().unwrap().into()],
                &tensorfs::Convert {
                    progress: Some(&progress),
                    ..Default::default()
                },
            ),
        )
        .await
        .unwrap()
        .unwrap_err();
        assert_eq!(error.kind, Kind::Conflict);
        assert_eq!(error.detail, *detail);
        assert_eq!(stages.lock().unwrap().last().unwrap(), "imported");
        assert!(
            actual_headers(&cache.join("blobs")).is_empty(),
            "callback refusal committed a header"
        );
        assert_eq!(
            fs::read_dir(cache.join("refs")).unwrap().count(),
            0,
            "callback refusal published a repo"
        );
        // The returned error means the worker has drained, not merely that its await was dropped.
        let operator = tensorfs::store2::Store::open(&cache).unwrap();
        assert!(
            operator.sweep_writers(std::time::Duration::ZERO).unwrap() > 0,
            "writer fence still active after callback error returned"
        );
        drop(operator);
        let checkpoint = client
            .convert(
                &[source.to_str().unwrap().into()],
                &tensorfs::Convert::default(),
            )
            .await
            .unwrap();
        assert_eq!(
            checkpoint.component("model").unwrap().read("w").unwrap(),
            WEIGHT
        );
    }
}

struct ConversionProvider;
#[async_trait::async_trait]
impl tensorfs::TokenProvider for ConversionProvider {
    async fn headers(
        &self,
        _request: &tensorfs::Request<'_>,
    ) -> tensorfs::Result<Vec<(String, String)>> {
        Ok(Vec::new())
    }
}
struct StalledConversion {
    source: String,
    waiting: std::sync::Arc<std::sync::atomic::AtomicBool>,
    release: std::sync::Arc<std::sync::atomic::AtomicBool>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
}
impl StalledConversion {
    fn new() -> Self {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let source = format!(
            "http://{}/model.safetensors",
            listener.local_addr().unwrap()
        );
        let waiting = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let release = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (seen, go, done) = (waiting.clone(), release.clone(), stop.clone());
        let worker = std::thread::spawn(move || {
            while !done.load(Ordering::Acquire) {
                let mut socket = match listener.accept() {
                    Ok((s, _)) => s,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                        continue;
                    }
                    Err(e) => panic!("accept: {e}"),
                };
                socket
                    .set_read_timeout(Some(std::time::Duration::from_secs(2)))
                    .unwrap();
                let mut request = Vec::new();
                let mut byte = [0];
                while !request.ends_with(b"\r\n\r\n") {
                    if socket.read(&mut byte).unwrap_or(0) == 0 {
                        break;
                    }
                    request.push(byte[0]);
                }
                let body = source_safetensors();
                let _ = write!(
                    socket,
                    "HTTP/1.1 200 OK\r\ncontent-length:{}\r\nconnection:close\r\n\r\n",
                    body.len()
                );
                let _ = socket.write_all(&body[..16]);
                let _ = socket.flush();
                seen.store(true, Ordering::Release);
                while !go.load(Ordering::Acquire) && !done.load(Ordering::Acquire) {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                let _ = socket.write_all(&body[16..]);
            }
        });
        Self {
            source,
            waiting,
            release,
            stop,
            worker: Some(worker),
        }
    }
}
impl Drop for StalledConversion {
    fn drop(&mut self) {
        self.release.store(true, Ordering::Release);
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker.join().unwrap();
        }
    }
}

#[test]
fn dropped_public_convert_wakes_stalled_io_without_restarting_the_callers_current_thread_runtime() {
    let scratch = Scratch::new();
    let origin = StalledConversion::new();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let provider = std::sync::Arc::new(ConversionProvider);
    let weak = std::sync::Arc::downgrade(&provider);
    let mut config = Config::from_env();
    config.cache_dir = scratch.0.join("cache");
    config.endpoint = origin.source.clone();
    config.token = Some(provider);
    config.mirror = None;
    let client = Client::new(config).unwrap();
    let sources = [origin.source.clone()];
    let options = tensorfs::Convert::default();
    let mut future = Box::pin(client.convert(&sources, &options));
    runtime.block_on(async {
        tokio::select! {
            result = &mut future => panic!("conversion finished before the HTTP body stalled: {result:?}"),
            () = async {tokio::time::timeout(std::time::Duration::from_secs(3), async {while !origin.waiting.load(Ordering::Acquire){tokio::time::sleep(std::time::Duration::from_millis(1)).await;}}).await.unwrap();} => {},
        }
    });
    drop(future);
    drop(client);
    // No runtime is driven here. Native Notify must wake the blocking worker on its own.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
    while weak.upgrade().is_some() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let released_without_driver = weak.upgrade().is_none();
    if !released_without_driver {
        // Rescue and drain before reporting a failure, so a red oracle cannot strand its worker.
        origin.release.store(true, Ordering::Release);
        runtime.block_on(async {
            tokio::time::timeout(std::time::Duration::from_secs(3), async {
                while weak.upgrade().is_some() {
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                }
            })
            .await
            .unwrap();
        });
    }
    assert!(
        released_without_driver,
        "future Drop needed the caller I/O driver to restart before provider/worker release"
    );
    assert!(actual_headers(&scratch.0.join("cache/blobs")).is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn native_export_refuses_same_length_corrupt_chunks_and_known_repository_objects_before_header(
) {
    let scratch = Scratch::new();
    let folder = fixture(&scratch.0);
    let endpoint = scratch.0.join("endpoint");
    let producer = client(&scratch.0.join("source-cache"), &endpoint);
    let plain = producer
        .open(folder.to_str().unwrap(), &Open::default())
        .await
        .unwrap();
    let chunk = folder.join("chunks").join(&ObjectRef::of(WEIGHT).sha256);
    fs::write(&chunk, [0, 0, 0, 0]).unwrap();
    let bad_chunk = scratch.0.join("bad-chunk-export");
    assert_eq!(
        export(&plain, &bad_chunk, Format::Cozytensors, None)
            .unwrap_err()
            .kind,
        Kind::Integrity
    );
    assert!(
        !bad_chunk.join("model.cozytensors").exists(),
        "corrupt source published a portable header"
    );
    assert!(
        !bad_chunk
            .join("chunks")
            .join(&ObjectRef::of(WEIGHT).sha256)
            .exists(),
        "corrupt object was atomically named as valid"
    );
    fs::write(&chunk, WEIGHT).unwrap();
    export(&plain, &bad_chunk, Format::Cozytensors, None).unwrap();
    let tree = producer
        .upload_folder(&"acme/integrity".parse().unwrap(), &folder, None)
        .await
        .unwrap();
    let cache = scratch.0.join("reader-cache");
    let reader = client(&cache, &endpoint);
    let managed = reader.open(&tree.id(), &Open::default()).await.unwrap();
    managed.fetch(&Fetch::default()).await.unwrap();
    let readme = cache.join(ObjectRef::of(README).key());
    // CAS files are intentionally read-only and may be hard-linked. Replace only this
    // fixture inode, preserving the source origin while injecting same-length corruption.
    platform::durable::replace(&readme, &vec![b'!'; README.len()], 0o444).unwrap();
    let bad_repo = scratch.0.join("bad-repo-export");
    assert_eq!(
        export(&managed, &bad_repo, Format::Cozytensors, None)
            .unwrap_err()
            .kind,
        Kind::Integrity
    );
    assert!(!bad_repo.join("model.cozytensors").exists());
    assert!(!bad_repo.join("README.bin").exists());
    platform::durable::replace(&readme, README, 0o444).unwrap();
    export(&managed, &bad_repo, Format::Cozytensors, None).unwrap();
    assert_eq!(fs::read(bad_repo.join("README.bin")).unwrap(), README);
    assert_eq!(
        ObjectRef::of(&fs::read(bad_repo.join("model.cozytensors")).unwrap()),
        *managed.id()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn streaming_writer_resumes_its_keyed_journal_after_drop_then_publishes_the_complete_result()
{
    let scratch = Scratch::new();
    let cache = scratch.0.join("cache");
    let endpoint = scratch.0.join("endpoint");
    let client = client(&cache, &endpoint);
    let mut interrupted = client
        .writer(WriterOptions {
            repo_id: Some("acme/resumed"),
            resume: Some("remote-resume"),
            ..Default::default()
        })
        .unwrap();
    interrupted
        .component(
            "model",
            "test.Model",
            serde_json::from_str(r#"{"width":4}"#).unwrap(),
        )
        .unwrap()
        .put("before", Dtype::U8, &[4], &mut std::io::Cursor::new(WEIGHT))
        .unwrap();
    interrupted
        .component_writer("model")
        .unwrap()
        .asset("tokenizer.json", ASSET)
        .unwrap();
    drop(interrupted);
    assert!(cache.join("tmp/writers/remote-resume/journal").exists());
    assert!(!endpoint
        .join(tensorfs::format::repo_key("acme/resumed"))
        .exists());
    let mut resumed = client
        .writer(WriterOptions {
            repo_id: Some("acme/resumed"),
            resume: Some("remote-resume"),
            ..Default::default()
        })
        .unwrap();
    assert!(resumed.has("model", "before").unwrap());
    resumed
        .component(
            "model",
            "test.Model",
            serde_json::from_str(r#"{"width":4}"#).unwrap(),
        )
        .unwrap();
    resumed
        .component_writer("model")
        .unwrap()
        .put(
            "after",
            Dtype::U8,
            &[4],
            &mut std::io::Cursor::new([13, 14, 15, 16]),
        )
        .unwrap();
    let checkpoint = resumed.commit().await.unwrap();
    assert!(!cache.join("tmp/writers/remote-resume/journal").exists());
    let reader = self::client(&scratch.0.join("reader"), &endpoint);
    let remote = reader
        .open(&checkpoint.tree().unwrap().id(), &Open::default())
        .await
        .unwrap();
    remote.fetch(&Fetch::default()).await.unwrap();
    drop(client);
    drop(resumed);
    drop(checkpoint);
    drop(reader);
    let model = remote.component("model").unwrap();
    assert_eq!(model.read("before").unwrap(), WEIGHT);
    assert_eq!(model.read("after").unwrap(), [13, 14, 15, 16]);
    assert_eq!(model.asset("tokenizer.json").unwrap(), ASSET);
    assert_eq!(model.config()["width"], 4);
}

#[test]
fn public_conversion_without_a_tokio_caller_refuses_instead_of_starting_a_runtime_or_panicking() {
    let scratch = Scratch::new();
    let source = conversion_source(&scratch.0);
    let cache = scratch.0.join("cache");
    let client = client(&cache, &scratch.0.join("endpoint"));
    let sources = [source.to_str().unwrap().to_owned()];
    assert_eq!(
        futures::executor::block_on(client.convert(&sources, &tensorfs::Convert::default()))
            .unwrap_err()
            .kind,
        Kind::Unsupported
    );
    assert_eq!(
        futures::executor::block_on(client.resolve_source(&sources[0]))
            .unwrap_err()
            .kind,
        Kind::Unsupported
    );
    assert!(actual_headers(&cache.join("blobs")).is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn literal_writer_inside_classified_snapshot_preserves_the_callers_header_from_ttl_prune() {
    let scratch = Scratch::new();
    let cache = scratch.0.join("cache");
    let client = client(&cache, &scratch.0.join("endpoint"));
    let operator = tensorfs::store2::Store::open(&cache).unwrap().with_cache();
    let chosen = ObjectRef::of(b"caller chooses an internally classified snapshot root").sha256;
    let control = ObjectRef::of(b"expired internal snapshot positive control").sha256;
    for digest in [&chosen, &control] {
        let folder = cache.join("snapshots").join(digest);
        operator.cache_snapshot(&folder).unwrap();
        fs::create_dir_all(&folder).unwrap();
        fs::write(folder.join("old-cached-file"), b"cache fixture").unwrap();
    }
    let dest = cache
        .join("snapshots")
        .join(&chosen)
        .join("literal.cozytensors");
    let mut writer = client
        .writer(WriterOptions {
            dest: Some(&dest),
            ..Default::default()
        })
        .unwrap();
    writer
        .component("model", "test.Model", Default::default())
        .unwrap()
        .put("w", Dtype::U8, &[4], &mut std::io::Cursor::new(WEIGHT))
        .unwrap();
    let checkpoint = writer.commit().await.unwrap();
    let chosen_record = cache
        .join("tmp/cache/snapshots")
        .join(format!("snapshots-{chosen}.json"));
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&fs::read(&chosen_record).unwrap()).unwrap()
            ["owned"],
        true
    );
    drop(checkpoint);
    drop(writer);
    drop(client);
    for digest in [&chosen, &control] {
        let record = cache
            .join("tmp/cache/snapshots")
            .join(format!("snapshots-{digest}.json"));
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&record).unwrap()).unwrap();
        value["last_use"] = 0.into();
        fs::write(record, serde_json::to_vec(&value).unwrap()).unwrap();
        // Effective recency is max(record, directory mtime); age both real inputs to TTL.
        let folder = cache.join("snapshots").join(digest);
        platform::durable::open_dir(&folder)
            .unwrap()
            .set_times(fs::FileTimes::new().set_modified(std::time::UNIX_EPOCH))
            .unwrap();
    }
    fs::write(cache.join("tmp/cache/sweep.json"), br#"{"last_use":0}"#).unwrap();
    operator.cache_maintain(0).unwrap();
    assert!(
        !cache.join("snapshots").join(control).exists(),
        "TTL prune positive control did not run"
    );
    assert!(
        dest.exists(),
        "TTL pruned an explicitly chosen literal header"
    );
    drop(operator);
    let reopened = self::client(&cache, &scratch.0.join("endpoint"))
        .open(
            dest.to_str().unwrap(),
            &Open {
                local_files_only: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(
        reopened.component("model").unwrap().read("w").unwrap(),
        WEIGHT
    );
}

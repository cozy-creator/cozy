//! Public verification checks local bytes without repairing or fetching them.
#![cfg(feature = "store")]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use tensorfs::format::{self, Body, Dtype, Header, Location, ObjectRef, Tensor};
use tensorfs::{Client, Config, Kind, Open, Selection};

struct Fixture {
    root: PathBuf,
    folder: PathBuf,
    header: ObjectRef,
    chunks: Vec<ObjectRef>,
    asset: ObjectRef,
    unused: ObjectRef,
}

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "verification-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let folder = root.join("source");
        fs::create_dir_all(folder.join("chunks")).unwrap();
        let alignment = tensorfs::format::layout::ALIGN as usize;
        let mut shared_chunk = vec![0; alignment * 2];
        shared_chunk[..4].copy_from_slice(b"abcd");
        shared_chunk[alignment..alignment + 4].copy_from_slice(b"wxyz");
        let chunks: Vec<_> = [shared_chunk, b"ijklmnop".to_vec()]
            .iter()
            .map(|bytes| {
                let object = ObjectRef::of(bytes);
                fs::write(folder.join("chunks").join(&object.sha256), bytes).unwrap();
                object
            })
            .collect();
        let asset = ObjectRef::of(b"vocabulary");
        fs::write(folder.join("chunks").join(&asset.sha256), b"vocabulary").unwrap();
        let unused = ObjectRef::of(b"unreferenced");
        fs::write(folder.join("chunks").join(&unused.sha256), b"unreferenced").unwrap();
        let component = |name: &str, chunk, offset, length| format::Component {
            name: name.into(),
            tensors: vec![Tensor {
                key: "weight".into(),
                dtype: Dtype::U8,
                shape: vec![length],
                body: Body::Data(Location {
                    chunk,
                    offset,
                    length,
                }),
            }],
            assets: if name == "model" {
                BTreeMap::from([("vocab.txt".into(), asset.clone())])
            } else {
                BTreeMap::new()
            },
            ..format::Component::default()
        };
        let bytes = Header::new(
            chunks.iter().cloned().chain([unused.clone()]).collect(),
            None,
            vec![
                component("model", 0, 0, 4),
                component("shared", 0, alignment as u64, 4),
                component("other", 1, 0, 8),
            ],
            BTreeMap::new(),
        )
        .unwrap()
        .to_cbor()
        .unwrap();
        let header = ObjectRef::of(&bytes);
        fs::write(folder.join("model.cozytensors"), bytes).unwrap();
        Self {
            root,
            folder,
            header,
            chunks,
            asset,
            unused,
        }
    }

    async fn open(&self) -> tensorfs::Checkpoint {
        let mut config = Config::from_env();
        config.cache_dir = self.root.join("cache");
        // There is deliberately no server. Verification must never contact this origin.
        config.endpoint = "http://127.0.0.1:1".into();
        config.mirror = None;
        config.token = None;
        Client::new(config)
            .unwrap()
            .open(self.folder.to_str().unwrap(), &Open::default())
            .await
            .unwrap()
    }

    fn object(&self, object: &ObjectRef) -> PathBuf {
        self.folder.join("chunks").join(&object.sha256)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn components(names: &[&str]) -> Selection {
    Selection::Components(names.iter().map(|name| (*name).into()).collect())
}

#[tokio::test]
async fn verification_separates_missing_and_corrupt_without_touching_unselected_objects() {
    let f = Fixture::new();
    let ck = f.open().await;
    assert!(ck.verify(&Selection::All).unwrap().is_ok());
    fs::remove_file(f.object(&f.chunks[0])).unwrap();
    fs::write(f.object(&f.chunks[1]), b"bad-hash").unwrap();
    let report = ck.verify(&Selection::All).unwrap();
    assert_eq!(report.missing, [f.chunks[0].clone()]);
    assert_eq!(report.corrupt, [f.chunks[1].clone()]);
    assert!(!report.is_ok());
    let model = ck.verify(&components(&["model"])).unwrap();
    assert_eq!(model.missing, [f.chunks[0].clone()]);
    assert!(model.corrupt.is_empty());
    let other = ck.verify(&components(&["other"])).unwrap();
    assert!(other.missing.is_empty());
    assert_eq!(other.corrupt, [f.chunks[1].clone()]);
    assert!(
        !f.object(&f.chunks[0]).exists(),
        "verification fetched missing data"
    );
    assert_eq!(
        fs::read(f.object(&f.chunks[1])).unwrap(),
        b"bad-hash",
        "verification repaired corrupt data"
    );
}

#[tokio::test]
async fn verification_includes_selected_assets_and_classifies_wrong_lengths_as_corrupt() {
    let f = Fixture::new();
    let ck = f.open().await;
    fs::write(f.object(&f.asset), b"short").unwrap();
    let report = ck.verify(&components(&["model", "model"])).unwrap();
    assert!(report.missing.is_empty());
    assert_eq!(report.corrupt.as_slice(), std::slice::from_ref(&f.asset));
    assert!(ck.verify(&components(&["other"])).unwrap().is_ok());
    fs::remove_file(f.object(&f.asset)).unwrap();
    let report = ck.verify(&components(&["model"])).unwrap();
    assert_eq!(report.missing.as_slice(), std::slice::from_ref(&f.asset));
    assert!(report.corrupt.is_empty());
}

#[tokio::test]
async fn verification_hashes_the_entire_shared_chunk_for_a_selected_tensor_range() {
    let f = Fixture::new();
    let ck = f.open().await;
    let mut bytes = fs::read(f.object(&f.chunks[0])).unwrap();
    bytes[tensorfs::format::layout::ALIGN as usize] ^= 1;
    fs::write(f.object(&f.chunks[0]), bytes).unwrap();
    let report = ck
        .verify(&Selection::Part {
            component: "model".into(),
            key: "weight".into(),
            role: None,
            range: Some(0..2),
        })
        .unwrap();
    assert_eq!(report.corrupt, [f.chunks[0].clone()]);
    assert!(ck.verify(&components(&["other"])).unwrap().is_ok());
}

#[tokio::test]
async fn verification_always_checks_the_actual_header_even_for_an_empty_selection() {
    let f = Fixture::new();
    let ck = f.open().await;
    fs::remove_file(f.object(&f.chunks[0])).unwrap();
    assert!(ck.verify(&components(&[])).unwrap().is_ok());
    let path = f.folder.join("model.cozytensors");
    let mut bytes = fs::read(&path).unwrap();
    bytes[0] ^= 1;
    fs::write(&path, &bytes).unwrap();
    let report = ck.verify(&components(&[])).unwrap();
    assert!(report.missing.is_empty());
    assert_eq!(report.corrupt.as_slice(), std::slice::from_ref(&f.header));
    assert_eq!(fs::read(&path).unwrap(), bytes);
    fs::remove_file(path).unwrap();
    assert_eq!(
        ck.verify(&components(&[])).unwrap().missing.as_slice(),
        std::slice::from_ref(&f.header)
    );
}

#[tokio::test]
async fn verification_rejects_unknown_selections_as_typed_errors() {
    let f = Fixture::new();
    let ck = f.open().await;
    assert_eq!(
        ck.verify(&components(&["absent"])).unwrap_err().kind,
        Kind::NotFound
    );
    assert_eq!(
        ck.verify(&Selection::Part {
            component: "model".into(),
            key: "weight".into(),
            role: None,
            range: Some(0..5),
        })
        .unwrap_err()
        .kind,
        Kind::Invalid
    );
}

#[tokio::test]
async fn verification_all_includes_declared_chunks_unused_by_components() {
    let f = Fixture::new();
    let ck = f.open().await;
    fs::remove_file(f.object(&f.unused)).unwrap();
    assert_eq!(
        ck.verify(&Selection::All).unwrap().missing.as_slice(),
        std::slice::from_ref(&f.unused)
    );
    assert!(ck
        .verify(&components(&["model", "other", "shared"]))
        .unwrap()
        .is_ok());
}

#[tokio::test]
async fn verification_all_checks_only_the_selected_checkpoint_not_its_repository() {
    let f = Fixture::new();
    let readme = ObjectRef::of(b"repository notes");
    fs::write(f.folder.join("README.md"), b"repository notes").unwrap();
    let alternate = ObjectRef::of(b"alternate weights");
    fs::write(f.object(&alternate), b"alternate weights").unwrap();
    let alternate_header = Header::new(
        vec![alternate.clone()],
        None,
        vec![format::Component {
            name: "alternate".into(),
            tensors: vec![Tensor {
                key: "weight".into(),
                dtype: Dtype::U8,
                shape: vec![alternate.length],
                body: Body::Data(Location {
                    chunk: 0,
                    offset: 0,
                    length: alternate.length,
                }),
            }],
            ..format::Component::default()
        }],
        BTreeMap::new(),
    )
    .unwrap()
    .to_cbor()
    .unwrap();
    fs::write(f.folder.join("model.alt.cozytensors"), alternate_header).unwrap();
    let mut config = Config::from_env();
    config.cache_dir = f.root.join("repo-cache");
    config.mirror = None;
    let client = Client::new(config.clone()).unwrap();
    let reference = "local/verify".parse().unwrap();
    let tree = client
        .upload_folder(&reference, &f.folder, &[], None)
        .await
        .unwrap();
    let ck = client
        .open(
            &tree.id(),
            &Open {
                variant: Some(String::new()),
                ..Open::default()
            },
        )
        .await
        .unwrap();
    fs::remove_file(config.cache_dir.join(alternate.key())).unwrap();
    fs::remove_file(config.cache_dir.join(readme.key())).unwrap();
    fs::remove_file(config.cache_dir.join(tree.key())).unwrap();
    assert!(ck.verify(&Selection::All).unwrap().is_ok());
    fs::remove_file(config.cache_dir.join(f.chunks[0].key())).unwrap();
    let report = ck.verify(&Selection::All).unwrap();
    assert_eq!(report.missing, [f.chunks[0].clone()]);
    assert!(report.corrupt.is_empty());
    assert!(ck.verify(&components(&["other"])).unwrap().is_ok());
}

#[cfg(unix)]
#[tokio::test]
async fn verification_preserves_non_missing_io_errors() {
    let f = Fixture::new();
    let ck = f.open().await;
    let path = f.object(&f.chunks[0]);
    fs::remove_file(&path).unwrap();
    std::os::unix::fs::symlink(&f.chunks[0].sha256, path).unwrap();
    assert_eq!(
        ck.verify(&components(&["model"])).unwrap_err().kind,
        Kind::Io
    );
    assert!(ck.verify(&components(&["other"])).unwrap().is_ok());
}

#[cfg(unix)]
#[tokio::test]
async fn verification_uses_gc_trash_only_after_not_found_and_never_restores_it() {
    let f = Fixture::new();
    let mut config = Config::from_env();
    config.cache_dir = f.root.join("repo-cache");
    config.mirror = None;
    let client = Client::new(config.clone()).unwrap();
    let reference = "local/trash-check".parse().unwrap();
    let tree = client
        .upload_folder(&reference, &f.folder, &[], None)
        .await
        .unwrap();
    let ck = client.open(&tree.id(), &Open::default()).await.unwrap();
    let blob = config.cache_dir.join(f.chunks[0].key());
    let trash = config.cache_dir.join("tmp/gc").join(&f.chunks[0].sha256);
    fs::create_dir_all(trash.parent().unwrap()).unwrap();
    fs::rename(&blob, &trash).unwrap();
    std::os::unix::fs::symlink(blob.file_name().unwrap(), &blob).unwrap();
    assert_eq!(
        ck.verify(&components(&["model"])).unwrap_err().kind,
        Kind::Io
    );
    fs::remove_file(&blob).unwrap();
    assert!(ck.verify(&components(&["model"])).unwrap().is_ok());
    assert!(
        trash.exists() && !blob.exists(),
        "verification moved the GC copy"
    );
}

#[test]
fn python_verification_report_is_selected_and_local() {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let Ok(python) = std::env::var("TENSORFS_TEST_PYTHON") else {
        eprintln!("skipped: TENSORFS_TEST_PYTHON is required for the Python binding proof");
        return;
    };
    let fixture = Fixture::new();
    let pair = |object: &ObjectRef| (object.sha256.clone(), object.length);
    let request = serde_json::json!({
        "folder": fixture.folder,
        "header": pair(&fixture.header),
        "chunks": fixture.chunks.iter().map(pair).collect::<Vec<_>>(),
        "asset": pair(&fixture.asset),
        "unused": pair(&fixture.unused),
    });
    let script =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../tensorfs-py/scripts/verification_report.py");
    let mut child = Command::new(python)
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&serde_json::to_vec(&request).unwrap())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "Python verification proof: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    println!("{}", String::from_utf8_lossy(&output.stdout));
}

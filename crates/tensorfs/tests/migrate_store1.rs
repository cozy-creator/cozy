//! The transition is offline and filesystem-authoritative. Ordinary tests use the committed
//! real-v1 metadata corpus; the explicit hosted gate also writes stores with the published
//! 0.10.0 CLI, and asks that same binary to reopen/init the converted store.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
#[cfg(feature = "faults")]
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;
#[cfg(feature = "faults")]
use std::process::{Child, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde_json::{json, Value};
use tensorfs::format::{Body, Header, ObjectRef};
use tensorfs::migrate::store1;
#[cfg(feature = "faults")]
use tensorfs::store::fault;
use tensorfs::store::{Durability, Store};
use tensorfs::Kind;

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "store1-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        Self(root)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn files(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for entry in fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            found.extend(files(&path));
        } else {
            found.push(path);
        }
    }
    found.sort();
    found
}

fn snapshot(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    files(root)
        .into_iter()
        .map(|p| {
            (
                p.strip_prefix(root).unwrap().to_owned(),
                fs::read(&p).unwrap(),
            )
        })
        .collect()
}

fn write(path: &Path, bytes: &[u8]) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
}

fn old_repo(root: &Path, name: &str, manifest: &ObjectRef, local: bool) -> Vec<u8> {
    let (org, name_part) = name.split_once('/').unwrap();
    let mut checkpoint =
        json!({"manifest": manifest, "published": !local, "future_checkpoint": {"kept": true}});
    let releases = if local {
        checkpoint["source_selection"] = json!("a".repeat(64));
        json!([])
    } else {
        json!([
            {"version":"0.9.0", "revision":3, "yanked":true,"lanes":[{"lane":"plain","manifest":manifest,"future_lane":7}]},
            {"version":"1.0.0", "revision":1, "yanked":false,"lanes":[{"lane":"plain","manifest":manifest}]}
        ])
    };
    let bytes = serde_json::to_vec(&json!({"repo":{"org":org,"name":name_part},"checkpoints":[checkpoint],"releases":releases,"future_repository":{"kept":"verbatim"}})).unwrap();
    write(&root.join(format!("repos/{name}.json")), &bytes);
    bytes
}

/// Actual v1 object/inline tensor vectors, put into the actual store1 namespaces.
fn corpus(root: &Path) -> ObjectRef {
    for dir in ["repos", "manifests", "blobs", "checkouts", "tmp"] {
        fs::create_dir_all(root.join(dir)).unwrap();
    }
    write(&root.join("format"), b"store-format 1\nclaim-protocol 1\n");
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v1");
    let lanes: Vec<Value> =
        serde_json::from_slice(&fs::read(fixture.join("lanes.json")).unwrap()).unwrap();
    let lane = lanes
        .iter()
        .find(|l| l["name"] == "acme/v1-object")
        .unwrap();
    let manifest: ObjectRef = serde_json::from_value(lane["manifest"].clone()).unwrap();
    for entry in fs::read_dir(&fixture).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap();
        if name.len() != 64 {
            continue;
        }
        let bytes = fs::read(&path).unwrap();
        let destination = if name == manifest.sha256 {
            root.join(format!(
                "manifests/{}/{}/{}.json",
                &name[..2],
                &name[2..4],
                name
            ))
        } else {
            root.join(format!("blobs/{}/{}/{}", &name[..2], &name[2..4], name))
        };
        // Copy just the selected manifest; other manifests in the corpus are not blobs.
        if name == manifest.sha256 || tensorfs::migrate::v1::manifest(&bytes).unwrap().is_none() {
            write(&destination, &bytes);
        }
    }
    old_repo(root, "local/mine", &manifest, true);
    old_repo(root, "acme/model", &manifest, false);
    // SQL and private incomplete-root metadata are never parsed/repaired or thrown away.
    write(
        &root.join("tensorfs.sqlite"),
        b"SQLite format 3\0opaque catalog bytes",
    );
    write(
        &root.join("roots/derived/kept.json"),
        br#"{"transaction_id":"private-unfinished","restore_epoch":3}"#,
    );
    write(
        &root.join("roots/derived/kept.parts"),
        b"accepted part journal\n",
    );
    write(
        &root.join("checkouts/mine/note.txt"),
        b"user checkpoint companion",
    );
    manifest
}

fn assert_preserved(root: &Path, before: &BTreeMap<PathBuf, Vec<u8>>) {
    for (path, bytes) in before {
        let after = if path.starts_with("manifests") {
            root.join(".store1-migration/legacy-manifests")
                .join(path.strip_prefix("manifests").unwrap())
        } else if path.starts_with("blobs") {
            root.join(".store1-migration/store3").join(path)
        } else if path == Path::new("format") {
            let journal: Value = serde_json::from_slice(
                &fs::read(root.join(".store1-migration/prepared.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(
                serde_json::from_value::<Vec<u8>>(journal["original_format"].clone()).unwrap(),
                *bytes
            );
            continue;
        } else {
            root.join(path)
        };
        assert_eq!(
            fs::read(&after).unwrap(),
            *bytes,
            "lost original {}",
            path.display()
        );
    }
}

#[test]
fn local_and_public_roots_preserve_bytes_alias_privacy_and_old_metadata() {
    let scratch = Scratch::new();
    let root = scratch.0.join("store");
    let old = corpus(&root);
    let before = snapshot(&root);
    let upgrade = store1::migrate(&root).unwrap().unwrap();
    assert_eq!(upgrade.report.repos, 2);
    assert_eq!(upgrade.report.checkpoints.len(), 1);
    assert_eq!(upgrade.report.checkpoints[0].manifest, old);
    assert_eq!(upgrade.report.aliases["local/mine"], "a".repeat(64));
    assert!(store1::cutover_started(&root).unwrap());
    assert!(root.join("manifests").is_file());
    let store = Store::open(&root)
        .unwrap()
        .with_durability(Durability::Durable)
        .with_grace(Duration::ZERO);
    let local = store.get_ref("local/mine").unwrap().unwrap();
    assert!(local.tags.iter().all(|t| !t.public && !t.yanked));
    assert_eq!(
        local.tag("main").unwrap().commit,
        local.tag(&"a".repeat(64)).unwrap().commit
    );
    let public = store.get_ref("acme/model").unwrap().unwrap();
    assert!(public.tag("1.0.0").unwrap().public);
    assert!(public.tag("0.9.0").unwrap().yanked);
    assert_eq!(public.default_tag().unwrap().name, "1.0.0");
    let current = upgrade.report.checkpoints[0].tree.clone();
    let tree = store.tree(&current).unwrap();
    let header = Header::from_cbor(&store.read(tree.header("").unwrap()).unwrap()).unwrap();
    assert!(!header.components().is_empty());
    for t in header.components().iter().flat_map(|c| &c.tensors) {
        if let Body::Data(at) = &t.body {
            if at.length > 0 {
                assert!(
                    at.length
                        <= store.read(&header.chunks[at.chunk as usize]).unwrap().len() as u64
                );
            }
        }
    }
    let Body::Data(at) = &header.tensor("model", "weight").unwrap().body else {
        panic!("plain vector changed scheme")
    };
    let bytes = tensor_bytes(&store, &header, at.chunk, at.offset, at.length);
    assert_eq!(
        ObjectRef::of(&bytes).sha256,
        "b9e8b63a06482ba7c2ab3193a931c8ef41e1a656028b068f0a747b1a4417a972"
    );
    assert_preserved(&root, &before);
    // A collection can remove old unrooted blob names, but cannot destroy retained legacy bytes.
    store.gc().unwrap();
    assert_preserved(&root, &before);
    drop(upgrade);
    let current_files = snapshot(&root);
    assert!(store1::migrate(&root).unwrap().is_none());
    assert_eq!(
        snapshot(&root),
        current_files,
        "second migration changed the store"
    );
}

#[test]
fn old_owner_writer_and_read_lease_exclude_conversion() {
    let scratch = Scratch::new();
    let root = scratch.0.join("store");
    corpus(&root);
    for path in ["owner.lock", "tmp/writers/recovery.lock"] {
        let path = root.join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .unwrap();
        // Real store1 writers and external readers share recovery.lock; old owner is exclusive.
        if path.file_name().unwrap() == "owner.lock" {
            file.lock().unwrap();
        } else {
            file.lock_shared().unwrap();
        }
        let before = snapshot(&root);
        let error = store1::migrate(&root).err().unwrap();
        assert_eq!(error.kind, Kind::Conflict);
        assert!(!store1::cutover_started(&root).unwrap());
        assert!(!root.join(".store1-migration").exists());
        assert_eq!(snapshot(&root), before);
        drop(file);
    }
    let upgrade = store1::migrate(&root).unwrap().unwrap();
    let owner = OpenOptions::new()
        .read(true)
        .write(true)
        .open(root.join("owner.lock"))
        .unwrap();
    assert!(owner.try_lock().is_err(), "owner handoff lost the claim");
    drop(upgrade);
    owner.try_lock().unwrap();
}

#[test]
fn invalid_public_version_and_ambiguous_alias_refuse_before_cutover() {
    for invalid_version in [true, false] {
        let scratch = Scratch::new();
        let root = scratch.0.join("store");
        corpus(&root);
        let path = root.join(if invalid_version {
            "repos/acme/model.json"
        } else {
            "repos/local/mine.json"
        });
        let mut doc: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        if invalid_version {
            doc["releases"][0]["version"] = json!("not a public version");
        } else {
            let duplicate = doc["checkpoints"][0].clone();
            doc["checkpoints"].as_array_mut().unwrap().push(duplicate);
        }
        fs::write(&path, serde_json::to_vec(&doc).unwrap()).unwrap();
        let before = snapshot(&root);
        let failure = store1::migrate(&root).err().unwrap();
        assert!(!failure.detail.is_empty());
        assert!(!store1::cutover_started(&root).unwrap());
        assert!(root.join("manifests").is_dir());
        for (path, bytes) in before {
            assert_eq!(fs::read(root.join(path)).unwrap(), bytes);
        }
    }
}

#[test]
fn unknown_encoding_keeps_the_original_checkpoint_and_never_cuts_over() {
    let scratch = Scratch::new();
    let root = scratch.0.join("store");
    let old = corpus(&root);
    let path = root.join(format!(
        "manifests/{}/{}/{}.json",
        &old.sha256[..2],
        &old.sha256[2..4],
        old.sha256
    ));
    let mut manifest: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    let header = manifest["entries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["kind"] == "cozytensors")
        .unwrap()["blob"]
        .clone();
    let sha = header["sha256"].as_str().unwrap();
    let header_path = root.join(format!("blobs/{}/{}/{sha}", &sha[..2], &sha[2..4]));
    let mut bytes = fs::read(&header_path).unwrap();
    let mut reader = minicbor::Decoder::new(&bytes);
    reader.array().unwrap();
    reader.str().unwrap();
    reader.skip().unwrap(); // configs
    reader.skip().unwrap(); // assets
    assert!(reader.array().unwrap().unwrap() > 0);
    reader.skip().unwrap();
    let last = reader.position() - 1;
    // Change a spec's bytecode while retaining its CBOR structure and all tensor bytes.
    // It has a different exact spec identity; the converter must not assume plain/1.
    bytes[last] ^= 1;
    let new_header = ObjectRef::of(&bytes);
    write(
        &root.join(format!(
            "blobs/{}/{}/{}",
            &new_header.sha256[..2],
            &new_header.sha256[2..4],
            new_header.sha256
        )),
        &bytes,
    );
    for entry in manifest["entries"].as_array_mut().unwrap() {
        if entry["kind"] == "cozytensors" {
            entry["blob"] = serde_json::to_value(&new_header).unwrap();
        }
    }
    let bytes = serde_json::to_vec(&manifest).unwrap();
    let new_manifest = ObjectRef::of(&bytes);
    write(
        &root.join(format!(
            "manifests/{}/{}/{}.json",
            &new_manifest.sha256[..2],
            &new_manifest.sha256[2..4],
            new_manifest.sha256
        )),
        &bytes,
    );
    old_repo(&root, "local/mine", &new_manifest, true);
    let before = snapshot(&root);
    let error = store1::migrate(&root).err().unwrap();
    assert_eq!(error.kind, Kind::Unsupported);
    assert!(error.detail.contains("spec unknown"), "{error}");
    assert!(!store1::cutover_started(&root).unwrap());
    for (path, bytes) in before {
        assert_eq!(fs::read(root.join(path)).unwrap(), bytes);
    }
}

#[test]
fn absent_empty_and_foreign_roots_are_not_reformatted() {
    let scratch = Scratch::new();
    assert!(store1::migrate(&scratch.0.join("absent"))
        .unwrap()
        .is_none());
    assert!(store1::migrate(&scratch.0).unwrap().is_none());
    fs::write(scratch.0.join("mine.txt"), "foreign").unwrap();
    let before = snapshot(&scratch.0);
    assert!(store1::migrate(&scratch.0).is_err());
    assert_eq!(snapshot(&scratch.0), before);
}

#[cfg(feature = "faults")]
struct ChildGuard(Child);
#[cfg(feature = "faults")]
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
#[cfg(feature = "faults")]
fn migration_child() {
    let Some(root) = std::env::var_os("STORE1_MIGRATION_CHILD") else {
        return;
    };
    fault::arm(&std::env::var("STORE1_MIGRATION_FAULT").unwrap());
    let _upgrade = store1::migrate(Path::new(&root)).unwrap();
}

#[test]
#[cfg(feature = "faults")]
fn every_durable_boundary_recovers_after_process_death() {
    for point in ["prepared", "published", "retired", "fenced", "committed"] {
        let scratch = Scratch::new();
        let root = scratch.0.join("store");
        corpus(&root);
        let before = snapshot(&root);
        let fault = format!("store1.{point}");
        let mut child = ChildGuard(
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "migration_child", "--nocapture"])
                .env("STORE1_MIGRATION_CHILD", &root)
                .env("STORE1_MIGRATION_FAULT", &fault)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .unwrap(),
        );
        let out = child.0.stdout.take().unwrap();
        let (send, receive) = std::sync::mpsc::sync_channel(1);
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines() {
                if line.as_ref().is_ok_and(|l| l == &format!("FAULT {fault}")) {
                    let _ = send.send(());
                    return;
                }
            }
        });
        receive
            .recv_timeout(Duration::from_secs(30))
            .expect("child never reached the boundary");
        child.0.kill().unwrap();
        child.0.wait().unwrap();
        let _completed = store1::migrate(&root).unwrap();
        assert_preserved(&root, &before);
        assert!(Store::open(&root).is_ok());
        assert!(store1::migrate(&root).unwrap().is_none());
    }
}

fn tensor_bytes(
    store: &Store,
    header: &Header,
    chunk: u64,
    mut offset: u64,
    mut length: u64,
) -> Vec<u8> {
    let mut result = Vec::new();
    for object in &header.chunks[chunk as usize..] {
        if length == 0 {
            break;
        }
        if offset >= object.length {
            offset -= object.length;
            continue;
        }
        let bytes = store.read(object).unwrap();
        let take = length.min(object.length - offset);
        result.extend_from_slice(&bytes[offset as usize..(offset + take) as usize]);
        length -= take;
        offset = 0;
    }
    assert_eq!(length, 0, "tensor range ended before its declared length");
    result
}

/// Independent deterministic byte oracle copied from the published CLI's documented
/// splitmix64 fixture, not from the converter or its CBOR parser.
fn old_noise(seed: &str, length: usize) -> Vec<u8> {
    let hash = tensorfs::sha256::digest(seed.as_bytes());
    let mut state = u64::from_le_bytes(hash[..8].try_into().unwrap());
    let mut bytes = Vec::new();
    while bytes.len() < length {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        bytes.extend_from_slice(&(value ^ (value >> 31)).to_le_bytes());
    }
    bytes.truncate(length);
    bytes
}

fn published_tensor_oracle(store: &Store, tree: &ObjectRef) {
    let tree = store.tree(tree).unwrap();
    let header = Header::from_cbor(&store.read(tree.header("").unwrap()).unwrap()).unwrap();
    let mut count = 0;
    for c in header.components() {
        for t in &c.tensors {
            let name = if c.name == "vae" && t.key == "decoder.conv_out.weight" {
                "vae/decoder.conv_in.weight".to_owned()
            } else {
                format!("{}/{}", c.name, t.key)
            };
            match &t.body {
                Body::Data(at) => {
                    let bytes = tensor_bytes(store, &header, at.chunk, at.offset, at.length);
                    assert_eq!(
                        bytes,
                        old_noise(&format!("{name}#value"), bytes.len()),
                        "{name}"
                    );
                    count += 1;
                }
                Body::Encoded { encoding, parts } => {
                    for p in parts {
                        let role = match (encoding.scheme.as_str(), p.role.as_str()) {
                            ("fp8.rowwise", "weight") => "data",
                            ("nvfp4", "scale") => "weight_scale",
                            ("nvfp4", "scale_2") => "weight_scale_2",
                            (_, role) => role,
                        };
                        let bytes = tensor_bytes(store, &header, p.chunk, p.offset, p.length);
                        assert_eq!(
                            bytes,
                            old_noise(&format!("{name}#{role}"), bytes.len()),
                            "{name}/{role}"
                        );
                        count += 1;
                    }
                }
            }
        }
    }
    assert!(count >= 14, "fixture lost tensor coverage");
}

/// Run in the hosted migration gate with the actual published wheel's CLI installed in a
/// throwaway venv. It runs no GPU code and never touches any real default store.
#[test]
#[ignore = "requires the explicitly supplied published TensorFS 0.10.0 CLI"]
fn published_0_10_store_converts_and_old_clients_cannot_recreate_manifests() {
    let old = PathBuf::from(
        std::env::var_os("STORE1_TEST_LEGACY_TFS").expect("supply the actual 0.10.0 tfs"),
    );
    let version = Command::new(&old).arg("version").output().unwrap();
    assert!(version.status.success());
    assert!(String::from_utf8_lossy(&version.stdout).contains("0.10.0"));
    for plain in [true, false] {
        let scratch = Scratch::new();
        let root = scratch.0.join("old");
        let mut command = Command::new(&old);
        command
            .arg("checkpoint")
            .arg("write")
            .arg(&root)
            .args(["--blocks", "2", "--scale", "64"]);
        if plain {
            command.arg("--plain");
        }
        let result = command.output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let manifests = files(&root.join("manifests"));
        let manifest = ObjectRef::of(&fs::read(&manifests[0]).unwrap());
        let bytes = old_repo(&root, "local/mine", &manifest, true);
        let inspected = Command::new(&old)
            .args(["repo", "inspect"])
            .arg(root.join("repos/local/mine.json"))
            .arg("--rows")
            .arg(scratch.0.join("rows.jsonl"))
            .output()
            .unwrap();
        assert!(
            inspected.status.success(),
            "old native codec refused our alias: {}",
            String::from_utf8_lossy(&inspected.stderr)
        );
        assert!(
            !root.join("format").exists(),
            "real published0.10 fixture is unstamped"
        );
        let before = snapshot(&root);
        let upgrade = store1::migrate(&root).unwrap().unwrap();
        assert_eq!(upgrade.report.checkpoints.len(), 1);
        assert_eq!(fs::read(root.join("repos/local/mine.json")).unwrap(), bytes);
        assert_preserved(&root, &before);
        let store = Store::open(&root).unwrap();
        published_tensor_oracle(&store, &upgrade.report.checkpoints[0].tree);
        assert!(store
            .get_ref("local/mine")
            .unwrap()
            .unwrap()
            .tags
            .iter()
            .all(|t| !t.public));
        // Real old Store::open (get), Store::init and ensure-equivalent write must all refuse.
        let header = fs::read(&manifests[0]).err();
        assert!(header.is_some(), "old namespace still readable");
        for mut old_command in [
            {
                let mut c = Command::new(&old);
                c.arg("get").arg(&root).arg(&manifest.sha256);
                c
            },
            {
                let mut c = Command::new(&old);
                c.args(["store", "init"]).arg(&root);
                c
            },
            {
                let mut c = Command::new(&old);
                c.args(["checkpoint", "write"])
                    .arg(&root)
                    .args(["--plain", "--blocks", "1", "--scale", "64"]);
                c
            },
        ] {
            let output = old_command.output().unwrap();
            assert!(!output.status.success(), "old CLI reopened converted store");
            assert!(root.join("manifests").is_file());
        }
        assert_preserved(&root, &before);
        drop(upgrade);
    }
}

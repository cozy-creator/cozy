//! The `tfs` binary on v2 stores: a synthetic checkpoint with every scheme exports to
//! safetensors bit-identical to its stored parts (and to the reference decode with
//! `--dequantize`), exports as a static origin that pulls into a second store byte-identical,
//! and `verify` removes a corrupted object that the next pull fetches again.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value as Json;
use tensorfs::client::regions;
use tensorfs::format::{decode, Body, Dtype, PartBytes};
use tensorfs::store2::{Checkpoint, Store};

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Scratch {
        let path = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("tfs-cli-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// `tfs --json <args>`: its exit success and its document.
fn tfs(store: &Path, args: &[&str]) -> (bool, Json) {
    let out = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .arg("--store")
        .arg(store)
        .arg("--json")
        .args(args)
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    let doc = serde_json::from_str(text.trim()).unwrap_or_else(|e| {
        panic!(
            "tfs {args:?}: {e}\n{text}\n{}",
            String::from_utf8_lossy(&out.stderr)
        )
    });
    (out.status.success(), doc)
}

fn ok(store: &Path, args: &[&str]) -> Json {
    let (success, doc) = tfs(store, args);
    assert!(success, "tfs {args:?}: {doc}");
    doc
}

/// Name -> dtype, shape, bytes.
type Tensors = BTreeMap<String, (String, Vec<u64>, Vec<u8>)>;

/// Each tensor of a safetensors file, and its `__metadata__`.
fn safetensors(path: &Path) -> (Tensors, Json) {
    let bytes = fs::read(path).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    assert_eq!(n % 8, 0, "the header is padded to 8 bytes");
    let header: serde_json::Map<String, Json> = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let data = &bytes[8 + n..];
    let mut out = BTreeMap::new();
    let mut end = 0;
    for (name, t) in &header {
        if name == "__metadata__" {
            continue;
        }
        let at: Vec<usize> = serde_json::from_value(t["data_offsets"].clone()).unwrap();
        end = end.max(at[1]);
        let shape = serde_json::from_value(t["shape"].clone()).unwrap();
        out.insert(
            name.clone(),
            (
                t["dtype"].as_str().unwrap().to_owned(),
                shape,
                data[at[0]..at[1]].to_vec(),
            ),
        );
    }
    assert_eq!(end, data.len(), "no bytes past the last tensor");
    (out, header["__metadata__"].clone())
}

fn raw(ck: &Checkpoint, at: &tensorfs::format::Location) -> Vec<u8> {
    let mut buf = vec![0; at.length as usize];
    ck.bytes_into(at, &mut buf).unwrap();
    buf
}

fn blobs(root: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    for top in fs::read_dir(root.join("blobs")).unwrap().flatten() {
        for shard in fs::read_dir(top.path()).unwrap().flatten() {
            for blob in fs::read_dir(shard.path()).unwrap().flatten() {
                out.insert(
                    blob.file_name().to_string_lossy().into_owned(),
                    fs::read(blob.path()).unwrap(),
                );
            }
        }
    }
    out
}

#[test]
fn a_synthetic_checkpoint_exports_pulls_and_verifies() {
    let dir = Scratch::new("round-trip");
    let (a, b) = (dir.0.join("a"), dir.0.join("b"));
    let written = ok(
        &a,
        &[
            "dev", "ckpt", "--blocks", "2", "--scale", "2", "--name", "acme/syn",
        ],
    );
    assert_eq!(
        (
            written["name"].as_str(),
            written["release"].as_str(),
            written["variant"].as_str()
        ),
        (Some("acme/syn"), Some("1"), Some("mixed"))
    );

    let inspected = ok(&a, &["inspect", "acme/syn"]);
    assert_eq!(inspected["header"], written["header"]);
    let unet = &inspected["components"][0];
    assert_eq!(unet["name"], "unet");
    // One tensor of each scheme: its encoding, complete, at a glance.
    let first = |scheme: &str| {
        let tensors = unet["tensors"].as_array().unwrap();
        let t = tensors.iter().find(|t| {
            (t["encoding"].as_str()).is_some_and(|e| e.split(' ').next() == Some(scheme))
        });
        t.unwrap().clone()
    };
    let k = |s: &str| format!("down_blocks.0.{s}");
    let want = [
        ("fp8.tensorwise", k("attn1.to_v.weight"), [1024, 1024], r#"{scale_dtype: "F32"}"#),
        ("fp8.rowwise", k("attn1.to_q.weight"), [1024, 1024], r#"{axis: 0, scale_dtype: "F32"}"#),
        ("int8.tensorwise", k("attn2.to_v.weight"), [1024, 1024], r#"{scale_dtype: "F32"}"#),
        ("int8.rowwise", k("attn2.to_k.weight"), [1024, 1024], r#"{axis: 0, scale_dtype: "F32"}"#),
        ("int8.convrot", k("attn1.to_k.weight"), [1024, 1024], r#"{rotation: "hadamard", rotation_size: 256, scale_dtype: "F32"}"#),
        ("mxfp8", k("attn2.to_q.weight"), [1024, 1024], r#"{block_size: 32, scale_layout: "row_major"}"#),
        ("nvfp4", k("ff.net.0.proj.weight"), [1024, 4096], r#"{block_size: 16, global_scale: "multiply", nibble_order: "low_first", scale_layout: "row_major"}"#),
    ];
    for (scheme, key, shape, params) in &want {
        let encoding = format!("{scheme} {params}");
        let doc = serde_json::json!({"key": key, "dtype": "BF16", "shape": shape, "encoding": encoding});
        assert_eq!(first(scheme), doc);
    }
    let out = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .arg("--store")
        .arg(&a)
        .args(["inspect", "acme/syn"])
        .output()
        .unwrap();
    let human = String::from_utf8(out.stdout).unwrap();
    for (scheme, _, _, params) in &want {
        assert!(human.contains(&format!("    2 × {scheme} {params}\n")), "{human}");
    }
    assert!(human.contains("    6 × plain\n"), "{human}");

    let store = Store::open(&a).unwrap();
    let header = serde_json::from_value(written["header"].clone()).unwrap();
    let ck = store.checkpoint(&header).unwrap();
    // A standalone part (the 32 MiB embedding) and packs, never across components.
    assert!(ck.header().chunks.iter().any(|c| c.length == 32 << 20));
    let grouped = regions(ck.header(), 4 << 20, &[], &[]);
    let mut seen: Vec<&String> = grouped.iter().flatten().collect();
    assert_eq!(seen.len(), ck.tensors().count());
    seen.sort();
    seen.dedup();
    assert_eq!(
        seen.len(),
        ck.tensors().count(),
        "every tensor in exactly one region"
    );
    assert!(grouped.iter().all(|r| r
        .iter()
        .all(|k| k.split('/').next() == r[0].split('/').next())));

    // Safetensors: plain tensors and encoded parts bit-identical, schemes in the metadata.
    let out = dir.0.join("out");
    let exported = ok(
        &a,
        &["export", "acme/syn", out.to_str().unwrap(), "--safetensors"],
    );
    assert!(exported["files"]
        .as_array()
        .unwrap()
        .iter()
        .any(|f| f == "unet/diffusion_pytorch_model.safetensors"));
    let config: Json = serde_json::from_slice(&fs::read(out.join("unet/config.json")).unwrap()).unwrap();
    assert_eq!(config, Json::Object(ck.component("unet").unwrap().config.clone()));
    let files = [
        ("unet", "unet/diffusion_pytorch_model.safetensors"),
        ("text_encoder", "text_encoder/model.safetensors"),
        ("vae", "vae.safetensors"),
    ];
    let mut checked = 0;
    for (component, file) in files {
        let (tensors, meta) = safetensors(&out.join(file));
        assert_eq!(meta["format"], "pt");
        let encodings: Json = meta.get("cozytensors.encodings").map_or(Json::Null, |t| {
            serde_json::from_str(t.as_str().unwrap()).unwrap()
        });
        for t in &ck.header().component(component).unwrap().tensors {
            let key = &t.key;
            match &t.body {
                Body::Data(at) => {
                    assert_eq!(
                        tensors[key],
                        (t.dtype.name().to_owned(), t.shape.clone(), raw(&ck, at)),
                        "{key}"
                    );
                    checked += 1;
                }
                Body::Encoded { encoding, parts } => {
                    assert_eq!(encodings[key]["scheme"], encoding.scheme.as_str());
                    for p in parts {
                        let role = &p.role;
                        let name = encodings[key]["parts"][role].as_str().unwrap();
                        assert_eq!(
                            tensors[name],
                            (
                                p.dtype.name().to_owned(),
                                p.shape.clone(),
                                raw(&ck, &p.at())
                            ),
                            "{key} {role}"
                        );
                        checked += 1;
                    }
                    assert_eq!(
                        tensors.contains_key(&format!("{key}_scale")),
                        t.part("scale").is_some()
                    );
                }
            }
        }
    }
    assert!(checked > 20);

    // Dequantized: each encoded tensor is the reference decode, rounded once to BF16.
    let deq = dir.0.join("deq");
    ok(
        &a,
        &[
            "export",
            "acme/syn",
            deq.to_str().unwrap(),
            "--safetensors",
            "--dequantize",
            "bf16",
        ],
    );
    let (tensors, meta) = safetensors(&deq.join("unet/diffusion_pytorch_model.safetensors"));
    assert!(meta.get("cozytensors.encodings").is_none());
    for t in &ck.header().component("unet").unwrap().tensors {
        let key = &t.key;
        if let Body::Encoded { encoding, parts } = &t.body {
            let bytes: BTreeMap<&str, Vec<u8>> = parts
                .iter()
                .map(|p| (p.role.as_str(), raw(&ck, &p.at())))
                .collect();
            let views = parts
                .iter()
                .map(|p| {
                    (
                        p.role.as_str(),
                        PartBytes {
                            dtype: &p.dtype,
                            shape: &p.shape,
                            bytes: &bytes[p.role.as_str()],
                        },
                    )
                })
                .collect();
            let want = decode(&Dtype::BF16, &t.shape, encoding, &views).unwrap();
            assert_eq!(
                tensors[key],
                ("BF16".to_owned(), t.shape.clone(), want),
                "{key}"
            );
        }
    }

    // A static origin, pulled into a second store byte-identical.
    let origin = dir.0.join("origin");
    ok(
        &a,
        &["export", "acme/syn", origin.to_str().unwrap(), "--origin"],
    );
    let pulled = ok(&b, &["pull", origin.to_str().unwrap(), "acme/syn@1"]);
    assert_eq!((&pulled["header"], &pulled["tree"]), (&written["header"], &written["tree"]));
    assert_eq!(blobs(&b), blobs(&a));
    assert_eq!(
        fs::read(b.join("refs/acme/syn.json")).unwrap(),
        fs::read(a.join("refs/acme/syn.json")).unwrap()
    );
    assert_eq!(ok(&b, &["verify"])["corrupt"], serde_json::json!([]));

    // A corrupted object is found, removed, and fetched again.
    let chunk = &ck.header().chunks[0];
    let path = Store::open(&b).unwrap().path(chunk).unwrap();
    let mut bytes = fs::read(&path).unwrap();
    bytes[100] ^= 0xff;
    fs::remove_file(&path).unwrap();
    fs::write(&path, &bytes).unwrap();
    let (success, verified) = tfs(&b, &["verify", "acme/syn"]);
    assert!(!success);
    assert_eq!(verified["error"]["code"], "integrity");
    assert_eq!(verified["corrupt"], serde_json::json!([chunk.sha256]));
    assert!(!path.exists());
    let again = ok(&b, &["pull", origin.to_str().unwrap(), "acme/syn"]);
    assert_eq!(again["fetched"], chunk.length);
    assert_eq!(blobs(&b), blobs(&a));

    // Typed failures.
    let (success, missing) = tfs(&b, &["inspect", "acme/none"]);
    assert!(!success);
    assert_eq!(missing["error"]["code"], "not_found");
    assert_eq!(ok(&a, &["gc"])["objects"], 0, "everything is referenced");
}

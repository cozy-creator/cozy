//! The cozytensors/2 conformance vectors (`crates/tensorfs/conformance/vectors`): the Rust writer
//! reproduces the store byte for byte from `cases.json`, the Rust reader decodes every case to
//! its expected bytes or refuses it with its code, and green/red documents read or refuse.
//! `cargo test ... -- --ignored regenerate` rewrites the store after a deliberate change.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use base64::Engine;
use serde::Deserialize;
use tensorfs_core::format::layout::{self, Item, Order};
use tensorfs_core::format::{
    repo_key, Body, Component, Dtype, Encoding, Error, Header, Location, Manifest, ObjectRef, Part,
    Release, Repo, Tensor, HEADER_PATH,
};

#[derive(Deserialize)]
struct Cases {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    #[serde(default = "all")]
    lane: String,
    component: String,
    key: String,
    dtype: Dtype,
    shape: Vec<u64>,
    bits: Option<String>,
    encoding: Option<Encoding>,
    #[serde(default)]
    parts: BTreeMap<String, CasePart>,
    expect: Option<String>,
    refuse: Option<String>,
}

#[derive(Deserialize)]
struct CasePart {
    dtype: Dtype,
    shape: Vec<u64>,
    bits: String,
}

fn all() -> String {
    "all".into()
}

fn b64(s: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD.decode(s).unwrap()
}

fn vectors() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance/vectors")
}

fn cases() -> Vec<Case> {
    let c: Cases =
        serde_json::from_slice(&fs::read(vectors().join("cases.json")).unwrap()).unwrap();
    c.cases
}

const REPO: &str = "Conformance/Schemes";
const LICENSE: &[u8] = b"MIT\n";

/// Every file of the store, by path, as the Rust writer produces it from `cases.json`.
fn write_store() -> BTreeMap<String, Vec<u8>> {
    let cases = cases();
    let mut files = BTreeMap::new();
    let mut put = |bytes: Vec<u8>| {
        let r = ObjectRef::of(&bytes);
        files.insert(r.key(), bytes);
        r
    };
    let mut lanes = BTreeMap::new();

    // Lane "all": every case, laid out by the chunk rule.
    let mine: Vec<&Case> = cases.iter().filter(|c| c.lane == "all").collect();
    let mut items = Vec::new();
    let mut data: Vec<Vec<u8>> = Vec::new();
    for c in &mine {
        match &c.bits {
            Some(bits) => {
                data.push(b64(bits));
                items.push(Item {
                    component: &c.component,
                    key: &c.key,
                    role: "",
                    length: data.last().unwrap().len() as u64,
                });
            }
            None => {
                for (role, p) in &c.parts {
                    data.push(b64(&p.bits));
                    items.push(Item {
                        component: &c.component,
                        key: &c.key,
                        role,
                        length: data.last().unwrap().len() as u64,
                    });
                }
            }
        }
    }
    let plan = layout::plan(&items, Order::Natural);
    let slices: Vec<&[u8]> = data.iter().map(Vec::as_slice).collect();
    let chunks: Vec<ObjectRef> = plan.assemble(&slices).into_iter().map(&mut put).collect();
    let mut header = Header {
        chunks,
        ..Default::default()
    };
    let mut places = plan.places.iter();
    for c in &mine {
        let body = match &c.encoding {
            None => Body::Data(*places.next().unwrap()),
            Some(encoding) => {
                let parts = c.parts.iter().map(|(role, p)| {
                    let at = places.next().unwrap();
                    let part = Part {
                        dtype: p.dtype.clone(),
                        shape: p.shape.clone(),
                        chunk: at.chunk,
                        offset: at.offset,
                        length: at.length,
                    };
                    (role.clone(), part)
                });
                Body::Encoded {
                    encoding: encoding.clone(),
                    parts: parts.collect(),
                }
            }
        };
        let tensor = Tensor {
            dtype: c.dtype.clone(),
            shape: c.shape.clone(),
            body,
        };
        let component = header.components.entry(c.component.clone()).or_default();
        component.tensors.insert(c.key.clone(), tensor);
    }
    let manifest = Manifest {
        files: BTreeMap::from([
            (HEADER_PATH.into(), put(header.to_json().unwrap())),
            ("LICENSE".into(), put(LICENSE.to_vec())),
        ]),
        metadata: Default::default(),
    };
    lanes.insert("all".to_string(), put(manifest.to_json().unwrap()));

    // Lane "spans": one plain tensor at offset 100 of chunk 0, continuing through chunk 2.
    let span = cases.iter().find(|c| c.lane == "spans").unwrap();
    let bytes = b64(span.bits.as_ref().unwrap());
    let first = [vec![0u8; 100], bytes[..3996].to_vec()].concat();
    let chunks = vec![
        put(first),
        put(bytes[3996..8092].to_vec()),
        put(bytes[8092..].to_vec()),
    ];
    let tensor = Tensor {
        dtype: span.dtype.clone(),
        shape: span.shape.clone(),
        body: Body::Data(Location {
            chunk: 0,
            offset: 100,
            length: bytes.len() as u64,
        }),
    };
    let tensors = BTreeMap::from([(span.key.clone(), tensor)]);
    let header = Header {
        chunks,
        components: BTreeMap::from([(span.component.clone(), Component { tensors })]),
        metadata: Default::default(),
    };
    let manifest = Manifest {
        files: BTreeMap::from([(HEADER_PATH.into(), put(header.to_json().unwrap()))]),
        metadata: Default::default(),
    };
    lanes.insert("spans".to_string(), put(manifest.to_json().unwrap()));

    let repo = Repo {
        name: REPO.into(),
        releases: vec![Release {
            version: "1".into(),
            lanes,
        }],
    };
    files.insert(repo.key(), repo.to_json().unwrap());
    files
}

fn read_tree(root: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let rel = path
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                out.insert(rel, fs::read(&path).unwrap());
            }
        }
    }
    out
}

#[test]
fn writer_reproduces_the_store() {
    let want = write_store();
    let have = read_tree(&vectors().join("store"));
    assert_eq!(
        have.keys().collect::<Vec<_>>(),
        want.keys().collect::<Vec<_>>()
    );
    for (path, bytes) in &want {
        assert!(
            have[path] == *bytes,
            "{path} differs from the writer's bytes"
        );
    }
}

#[test]
#[ignore = "rewrites conformance/vectors/store"]
fn regenerate() {
    let root = vectors().join("store");
    let _ = fs::remove_dir_all(&root);
    for (path, bytes) in write_store() {
        let path = root.join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }
}

/// A reader over the committed store, verifying every object it loads.
fn blob(root: &Path, r: &ObjectRef) -> Vec<u8> {
    let bytes = fs::read(root.join(r.key())).unwrap();
    assert_eq!(&ObjectRef::of(&bytes), r);
    bytes
}

#[test]
fn reader_decodes_every_case() {
    let root = vectors().join("store");
    assert_eq!(repo_key(REPO), "refs/conformance/schemes.json");
    let repo = Repo::from_json(&fs::read(root.join(repo_key(REPO))).unwrap()).unwrap();
    assert_eq!(repo.name, REPO);
    let mut checked = 0;
    for c in cases() {
        let manifest =
            Manifest::from_json(&blob(&root, repo.lane(None, &c.lane).unwrap())).unwrap();
        let header = Header::from_json(&blob(&root, manifest.header().unwrap())).unwrap();
        let got = header.read(&c.component, &c.key, |i| {
            Ok::<_, Error>(blob(&root, &header.chunks[i]))
        });
        let name = format!("{}/{}", c.component, c.key);
        match (&c.expect, &c.refuse, got) {
            (Some(want), None, Ok(bytes)) => {
                assert!(bytes == b64(want), "{name}: decoded bytes differ")
            }
            (None, Some(code), Err(e)) => assert_eq!(e.code(), code, "{name}: {e}"),
            (_, _, got) => panic!("{name}: {got:?}"),
        }
        checked += 1;
    }
    assert!(checked >= 46, "{checked}");
}

#[test]
fn green_documents_read_and_red_documents_refuse() {
    for (dir, green) in [("green", true), ("red", false)] {
        let mut n = 0;
        for entry in fs::read_dir(vectors().join(dir)).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            let bytes = fs::read(&path).unwrap();
            let read = match name.split('-').next().unwrap() {
                "header" => Header::from_json(&bytes).map(drop),
                "manifest" => Manifest::from_json(&bytes).map(drop),
                "repo" => Repo::from_json(&bytes).map(drop),
                other => panic!("{name}: unknown kind {other}"),
            };
            assert_eq!(read.is_ok(), green, "{dir}/{name}: {read:?}");
            n += 1;
        }
        assert!(n > 3, "{dir} has {n} documents");
    }
}

#[test]
fn chunk_rule() {
    const MIB: u64 = 1 << 20;
    let item = |component, key, length| Item {
        component,
        key,
        role: "",
        length,
    };
    // Natural order: block 10 after block 9; small parts pack at 4 KiB; a large part splits
    // on the 64 MiB grid without closing the open pack.
    let items = [
        item("unet", "blocks.10.w", 100),
        item("unet", "blocks.9.w", 5000),
        item("unet", "blocks.9.big", 150 * MIB),
        item("unet", "blocks.9.x", 0),
        item("vae", "conv.w", 10),
    ];
    let p = layout::plan(&items, Order::Natural);
    assert_eq!(p.chunks, [64 * MIB, 64 * MIB, 22 * MIB, 8192 + 100, 10]);
    let at = |i: usize| (p.places[i].chunk, p.places[i].offset, p.places[i].length);
    let want = [
        (3, 8192, 100),
        (3, 0, 5000),
        (0, 0, 150 * MIB),
        (0, 0, 0),
        (4, 0, 10),
    ];
    assert_eq!([at(0), at(1), at(2), at(3), at(4)], want);

    // A pack closes at a block boundary once it holds 16 MiB, and at 64 MiB regardless.
    let small = 6 * MIB;
    let keys: Vec<String> = (0..12).map(|i| format!("b.{}.w", i / 4)).collect();
    let items: Vec<Item> = keys.iter().map(|k| item("t", k, small)).collect();
    let p = layout::plan(&items, Order::Given);
    assert_eq!(p.chunks, [24 * MIB, 24 * MIB, 24 * MIB]);
    let same_block: Vec<Item> = (0..12).map(|_| item("t", "b.0.w", small)).collect();
    assert_eq!(
        layout::plan(&same_block, Order::Given).chunks,
        [60 * MIB, 12 * MIB]
    );

    assert_eq!(
        layout::block_of("down_blocks.1.resnets.0.conv1.weight"),
        "down_blocks.1.resnets.0"
    );
    assert_eq!(layout::block_of("conv_in.weight"), "");
    let mut keys = ["a10", "a9", "a", "a09b", "b1", "a9b", "1a"];
    keys.sort_by(|a, b| layout::natural_cmp(a, b));
    assert_eq!(keys, ["1a", "a", "a9", "a09b", "a9b", "a10", "b1"]);
}

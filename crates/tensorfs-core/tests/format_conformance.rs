//! The cozytensors/1 conformance vectors (`crates/tensorfs/conformance/vectors`): the Rust writer
//! reproduces the store byte for byte from `cases/<family>.json`, the Rust reader decodes every
//! case to its expected bytes or refuses it with its code, and green/red documents read or
//! refuse. `cargo test ... -- --ignored regenerate` rewrites the store after a deliberate change.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use base64::Engine;
use serde::Deserialize;
use serde_json::Value as Json;
use tensorfs_core::format::layout::{self, Item};
use tensorfs_core::format::{
    repo_key, Body, Component, Config, Dtype, Encoding, Error, Header, Location, ObjectRef, Part,
    Pipeline, Release, Repo, Tensor, Tree, HEADER_FORMAT,
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
    parts: Vec<CasePart>,
    expect: Option<String>,
    refuse: Option<String>,
}

#[derive(Deserialize)]
struct CasePart {
    role: String,
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
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../tensorfs/conformance/vectors")
}

/// Each `cases/<family>.json`, by family name, in name order.
fn families() -> Vec<(String, Vec<Case>)> {
    let mut out = Vec::new();
    for entry in fs::read_dir(vectors().join("cases")).unwrap() {
        let path = entry.unwrap().path();
        let family = path.file_stem().unwrap().to_string_lossy().to_string();
        let c: Cases = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        out.push((family, c.cases));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// A family's repo: `Conformance/<family>`, at `refs/conformance/<family>.json`.
fn repo_name(family: &str) -> String {
    format!("Conformance/{family}")
}

const LICENSE: &[u8] = b"MIT\n";

/// Every file of the store, by path, as the Rust writer produces it from the cases: one repo per
/// family, so families add files without touching each other's.
fn write_store() -> BTreeMap<String, Vec<u8>> {
    let mut files = BTreeMap::new();
    for (family, cases) in families() {
        write_family(&mut files, &family, &cases);
    }
    files
}

fn write_family(files: &mut BTreeMap<String, Vec<u8>>, family: &str, cases: &[Case]) {
    let mut put = |bytes: Vec<u8>| {
        let r = ObjectRef::of(&bytes);
        files.insert(r.key(), bytes);
        r
    };
    // Each variant's header, by name; the tree of a set of them carries the LICENSE.
    let mut variants: BTreeMap<String, ObjectRef> = BTreeMap::new();
    let tree = |put: &mut dyn FnMut(Vec<u8>) -> ObjectRef, of: &[(&str, &ObjectRef)]| {
        let mut t = Tree {
            files: BTreeMap::from([("LICENSE".to_string(), put(LICENSE.to_vec()))]),
        };
        for (variant, header) in of {
            t.set_variant(variant, (*header).clone());
        }
        put(t.to_json().unwrap())
    };

    // Variant "all": every case in file order (components by first appearance), laid out by the
    // chunk rule.
    let mut grouped: Vec<(&str, Vec<&Case>)> = Vec::new();
    for c in cases.iter().filter(|c| c.lane == "all") {
        match grouped.iter_mut().find(|(name, _)| *name == c.component) {
            Some((_, list)) => list.push(c),
            None => grouped.push((&c.component, vec![c])),
        }
    }
    let mut items = Vec::new();
    let mut data: Vec<Vec<u8>> = Vec::new();
    for c in grouped.iter().flat_map(|(_, list)| list) {
        let parts: Vec<(&str, &str)> = match &c.bits {
            Some(bits) => vec![("", bits.as_str())],
            None => c
                .parts
                .iter()
                .map(|p| (p.role.as_str(), p.bits.as_str()))
                .collect(),
        };
        for (role, bits) in parts {
            data.push(b64(bits));
            items.push(Item {
                component: &c.component,
                key: &c.key,
                role,
                length: data.last().unwrap().len() as u64,
            });
        }
    }
    let plan = layout::plan(&items);
    let slices: Vec<&[u8]> = data.iter().map(Vec::as_slice).collect();
    let chunks: Vec<ObjectRef> = plan.assemble(&slices).into_iter().map(&mut put).collect();
    let mut places = plan.places.iter();
    let mut tensor = |c: &Case| {
        let body = match &c.encoding {
            None => Body::Data(*places.next().unwrap()),
            Some(encoding) => Body::Encoded {
                encoding: encoding.clone(),
                parts: (c.parts.iter())
                    .map(|p| {
                        let at = places.next().unwrap();
                        Part {
                            role: p.role.clone(),
                            dtype: p.dtype.clone(),
                            shape: p.shape.clone(),
                            chunk: at.chunk,
                            offset: at.offset,
                            length: at.length,
                        }
                    })
                    .collect(),
            },
        };
        Tensor {
            key: c.key.clone(),
            dtype: c.dtype.clone(),
            shape: c.shape.clone(),
            body,
        }
    };
    let components = (grouped.iter())
        .map(|(name, list)| Component {
            name: name.to_string(),
            tensors: list.iter().map(|c| tensor(c)).collect(),
            ..Default::default()
        })
        .collect();
    let header = Header::new(chunks, None, components, Default::default()).unwrap();
    variants.insert("all".to_string(), put(header.to_cbor().unwrap()));

    // Variant "spans" (core): one plain tensor at offset 4096 of chunk 0, continuing through
    // chunk 2; release 2 swaps the variants and is yanked, so unversioned lookups land on 1.
    let Some(span) = cases.iter().find(|c| c.lane == "spans") else {
        let repo = Repo {
            name: repo_name(family),
            releases: vec![Release {
                version: "1".into(),
                tree: tree(&mut put, &[("all", &variants["all"])]),
                yanked: false,
                revision: 0,
            }],
            checkpoints: vec![],
        };
        files.insert(repo.key(), repo.to_json().unwrap());
        return;
    };
    let bytes = b64(span.bits.as_ref().unwrap());
    let first = [vec![0u8; 4096], bytes[..3996].to_vec()].concat();
    let chunks = vec![
        put(first),
        put(bytes[3996..8092].to_vec()),
        put(bytes[8092..].to_vec()),
    ];
    let tensor = Tensor {
        key: span.key.clone(),
        dtype: span.dtype.clone(),
        shape: span.shape.clone(),
        body: Body::Data(Location {
            chunk: 0,
            offset: 4096,
            length: bytes.len() as u64,
        }),
    };
    let component = Component {
        name: span.component.clone(),
        tensors: vec![tensor],
        ..Default::default()
    };
    let header = Header::new(chunks, None, vec![component], Default::default()).unwrap();
    variants.insert("spans".to_string(), put(header.to_cbor().unwrap()));
    let (all, spans) = (&variants["all"], &variants["spans"]);
    let first = tree(&mut put, &[("all", all), ("spans", spans)]);
    let swapped = tree(&mut put, &[("all", spans), ("spans", all)]);
    let checkpoints = vec![tree(&mut put, &[("spans", spans)])];
    let release = |version: &str, tree, yanked, revision| Release {
        version: version.into(),
        tree,
        yanked,
        revision,
    };
    let repo = Repo {
        name: repo_name(family),
        releases: vec![
            release("1", first, false, 3),
            release("2", swapped, true, 0),
        ],
        checkpoints,
    };
    files.insert(repo.key(), repo.to_json().unwrap());
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
    let mut checked = 0;
    for (family, cases) in families() {
        let name = repo_name(&family);
        assert_eq!(repo_key(&name), format!("refs/conformance/{family}.json"));
        let repo = Repo::from_json(&fs::read(root.join(repo_key(&name))).unwrap()).unwrap();
        assert_eq!(repo.name, name);
        let tree = |version| Tree::from_json(&blob(&root, &repo.release(version).unwrap().tree)).unwrap();
        if family == "core" {
            assert_eq!(repo.release(None).unwrap().version, "1", "release 2 is yanked");
            assert_eq!(
                tree(Some("2")).header("all"),
                tree(None).header("spans"),
                "an explicit version still resolves"
            );
            assert_eq!(repo.roots().count(), 3);
        }
        for c in cases {
            let latest = tree(None);
            let header = Header::from_cbor(&blob(&root, latest.variant(Some(&c.lane)).unwrap().1)).unwrap();
            let got = header.read(&c.component, &c.key, |i| {
                Ok::<_, Error>(blob(&root, &header.chunks[i]))
            });
            let name = format!("{family}:{}/{}", c.component, c.key);
            match (&c.expect, &c.refuse, got) {
                (Some(want), None, Ok(bytes)) => {
                    assert!(bytes == b64(want), "{name}: decoded bytes differ")
                }
                (None, Some(code), Err(e)) => assert_eq!(e.code(), code, "{name}: {e}"),
                (_, _, got) => panic!("{name}: {got:?}"),
            }
            checked += 1;
        }
    }
    assert!(checked >= 80, "{checked}");
}

/// Real prebuilt execution, not a mock driver or a Triton runtime. Run only on an allocated
/// CUDA worker after building with the required AOT catalog.
#[cfg(feature = "store")]
#[test]
#[ignore = "requires a CUDA GPU and the prebuilt AOT catalog"]
fn cuda_decodes_every_encoded_green_vector_and_logical_windows() {
    use tensorfs_core::format::PartBytes;
    use tensorfs_core::io::{self, Copy, Dst, ReadOpts, Src};
    let mut checked = 0;
    for (family, cases) in families() {
        for case in cases {
            let (Some(encoding), Some(expected)) = (&case.encoding, &case.expect) else { continue };
            let storage: Vec<_> = case.parts.iter().map(|p| b64(&p.bits)).collect();
            let parts = case.parts.iter().zip(&storage).map(|(p, bytes)| (
                p.role.as_str(), PartBytes { dtype: &p.dtype, shape: &p.shape, bytes },
            )).collect();
            let total: u64 = case.shape.iter().product();
            let expected = b64(expected);
            let mut windows = vec![(0, total)];
            if total > 3 { windows.push((1, total - 2)); }
            for (begin, count) in windows {
                let gpu = io::decode::read(&case.dtype, &case.shape, encoding, &parts, 0, begin, count)
                    .unwrap_or_else(|e| panic!("{family}/{} {begin}+{count}: {e}", case.key));
                let mut bytes = vec![0; gpu.len()];
                let copy = Copy { src: Src::Device { device: 0, ptr: gpu.ptr(), stream: 0 }, dst: Dst::host(&mut bytes) };
                // SAFETY: both owned buffers remain live until the engine drains.
                unsafe { io::read(&[copy], &ReadOpts::default()) }.unwrap();
                let width = case.dtype.byte_len(&[1]).unwrap() as usize;
                assert_eq!(bytes, expected[begin as usize * width..(begin + count) as usize * width], "{family}/{} {begin}+{count}", case.key);
                checked += 1;
            }
        }
    }
    assert!(checked > 50, "only {checked} encoded vector windows were checked");
}

/// A green header's projection (its CBOR tree as JSON, byte strings as `{"hex": ...}`) as a
/// header, and whether it holds what this writer does not write (a minor tag, later fields).
fn from_projection(p: &Json) -> (Header, bool) {
    let list = |v: &Json| v.as_array().unwrap().clone();
    let uint = |v: &Json| v.as_u64().unwrap();
    let text = |v: &Json| v.as_str().unwrap().to_owned();
    let config = |v: &Json| serde_json::from_value::<Config>(v.clone()).unwrap();
    let root = list(p);
    let mut later = root.len() > 6 || root[0] != HEADER_FORMAT;
    let mut object = |v: &Json| {
        let c = list(v);
        later |= c.len() > 2;
        ObjectRef {
            sha256: text(&c[0]["hex"]),
            length: uint(&c[1]),
        }
    };
    let chunks: Vec<ObjectRef> = list(&root[1]).iter().map(&mut object).collect();
    let mut components = Vec::new();
    for c in list(&root[4]) {
        let c = list(&c);
        let assets = (c[4].as_object().unwrap().iter())
            .map(|(k, v)| (k.clone(), object(v)))
            .collect();
        components.push((c, assets));
    }
    let pipeline = list(&root[3]);
    later |= pipeline.len() > 2;
    let pipeline = (!pipeline.is_empty()).then(|| Pipeline {
        class: text(&pipeline[0]),
        config: config(&pipeline[1]),
    });
    let mut encodings = Vec::new();
    for e in list(&root[2]) {
        let e = list(&e);
        later |= e.len() > 2;
        encodings.push(Encoding {
            scheme: text(&e[0]),
            params: serde_json::from_value(e[1].clone()).unwrap(),
        });
    }
    // A location, and whether it has later fields.
    let at = |v: &Json| {
        let v = list(v);
        let l = Location {
            chunk: uint(&v[0]),
            offset: uint(&v[1]),
            length: uint(&v[2]),
        };
        (l, v.len() > 3)
    };
    let components = components.into_iter().map(|(c, assets)| {
        later |= c.len() > 6;
        let mut tensors = Vec::new();
        for t in list(&c[3]) {
            let t = list(&t);
            later |= t.len() > 5;
            let shape = |v: &Json| list(v).iter().map(uint).collect::<Vec<u64>>();
            let body = match uint(&t[3]) {
                0 => {
                    let (l, more) = at(&t[4]);
                    later |= more;
                    Body::Data(l)
                }
                i => Body::Encoded {
                    encoding: encodings[i as usize - 1].clone(),
                    parts: (list(&t[4]).iter())
                        .map(|p| {
                            let p = list(p);
                            let (l, more) = at(&p[3]);
                            later |= p.len() > 4 || more;
                            Part {
                                role: text(&p[0]),
                                dtype: Dtype::parse(&text(&p[1])),
                                shape: shape(&p[2]),
                                chunk: l.chunk,
                                offset: l.offset,
                                length: l.length,
                            }
                        })
                        .collect(),
                },
            };
            tensors.push(Tensor {
                key: text(&t[0]),
                dtype: Dtype::parse(&text(&t[1])),
                shape: shape(&t[2]),
                body,
            });
        }
        Component {
            name: text(&c[0]),
            class: text(&c[1]),
            config: config(&c[2]),
            tensors,
            assets,
            config_file: c.get(5).map(text).filter(|f| c.len() == 6 || f != "config.json"),
        }
    });
    let components = components.collect();
    let metadata = serde_json::from_value(root[5].clone()).unwrap();
    (Header::new(chunks, pipeline, components, metadata).unwrap(), later)
}

#[test]
fn green_documents_read_and_red_documents_refuse() {
    for (dir, green) in [("green", true), ("red", false)] {
        let mut n = 0;
        for entry in fs::read_dir(vectors().join(dir)).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if name.starts_with("header-") && name.ends_with(".json") {
                continue; // a green header's projection, checked with its bytes
            }
            let bytes = fs::read(&path).unwrap();
            let read = match name.split('-').next().unwrap() {
                "header" if name.starts_with("header-pre-cut") => {
                    let err = Header::from_cbor(&bytes).unwrap_err();
                    assert_eq!(err.code(), "pre_cut", "{name}: {err}");
                    Err(err)
                }
                "header" if green => {
                    // Both ways: the bytes read as the projection says, and (unless they hold
                    // what this writer does not write) the projection writes these bytes.
                    let p: Json = serde_json::from_slice(&fs::read(path.with_extension("json")).unwrap()).unwrap();
                    let (want, later) = from_projection(&p);
                    let got = Header::from_cbor(&bytes);
                    assert_eq!(got.as_ref().ok(), Some(&want), "{name}");
                    if !later {
                        assert!(want.to_cbor().unwrap() == bytes, "{name}: the writer's bytes differ");
                    }
                    got.map(drop)
                }
                "header" => Header::from_cbor(&bytes).map(drop),
                "tree" => Tree::from_json(&bytes).map(drop),
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
    // The given (header) order stands: block 10 before block 9 here. Small parts pack at
    // 4 KiB; a large part splits on the 64 MiB grid without closing the open pack.
    let items = [
        item("unet", "blocks.10.w", 100),
        item("unet", "blocks.9.w", 5000),
        item("unet", "blocks.9.big", 150 * MIB),
        item("unet", "blocks.9.x", 0),
        item("vae", "conv.w", 10),
    ];
    let p = layout::plan(&items);
    assert_eq!(p.chunks, [4096 + 5000, 64 * MIB, 64 * MIB, 22 * MIB, 10]);
    let at = |i: usize| (p.places[i].chunk, p.places[i].offset, p.places[i].length);
    let want = [
        (0, 0, 100),
        (0, 4096, 5000),
        (1, 0, 150 * MIB),
        (0, 0, 0),
        (4, 0, 10),
    ];
    assert_eq!([at(0), at(1), at(2), at(3), at(4)], want);

    // A pack closes at a block boundary once it holds 16 MiB, and at 64 MiB regardless.
    let small = 6 * MIB;
    let keys: Vec<String> = (0..12).map(|i| format!("b.{}.w", i / 4)).collect();
    let items: Vec<Item> = keys.iter().map(|k| item("t", k, small)).collect();
    let p = layout::plan(&items);
    assert_eq!(p.chunks, [24 * MIB, 24 * MIB, 24 * MIB]);
    let same_block: Vec<Item> = (0..12).map(|_| item("t", "b.0.w", small)).collect();
    assert_eq!(layout::plan(&same_block).chunks, [60 * MIB, 12 * MIB]);

    assert_eq!(
        layout::block_of("down_blocks.1.resnets.0.conv1.weight"),
        "down_blocks.1.resnets.0"
    );
    assert_eq!(layout::block_of("conv_in.weight"), "");
}

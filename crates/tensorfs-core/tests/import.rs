//! Import through the real engine, tables and pipeline: no mocks. Tests marked `network` reach
//! Hugging Face or GitHub.
use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use tensorfs_core::err::{Code, Result};
use tensorfs_core::format::{Dtype, Encoding, Metadata, Value};
use tensorfs_core::store2::Checkpoint;
use tensorfs_core::import::engine::{self, Ops, Reading, TensorReader};
use tensorfs_core::import::source::{Group, SourceSet};
use tensorfs_core::import::{self, Access, Imported, Layout, Part, Request, Sink};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tfs-import-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn vectors() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../tensorfs/vectors")
}

/// A safetensors file of `(key, dtype, shape, bytes)`.
fn safetensors(path: &Path, tensors: &[(&str, &str, Vec<u64>, Vec<u8>)]) {
    let mut head = serde_json::Map::new();
    let mut data = Vec::new();
    for (key, dtype, shape, bytes) in tensors {
        let start = data.len();
        data.extend_from_slice(bytes);
        head.insert(key.to_string(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}));
    }
    let head = serde_json::to_vec(&head).unwrap();
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut out = (head.len() as u64).to_le_bytes().to_vec();
    out.extend(head);
    out.extend(data);
    fs::write(path, out).unwrap();
}

/// A banked header as a sparse file of its full length: geometry only, no payload.
fn sparse(from: &Path, to: &Path) {
    let raw = fs::read(from).unwrap();
    let n = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let doc: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&raw[8..8 + n]).unwrap();
    let end = doc
        .values()
        .filter_map(|v| v["data_offsets"][1].as_u64())
        .max()
        .unwrap_or(0);
    fs::create_dir_all(to.parent().unwrap()).unwrap();
    fs::write(to, &raw[..8 + n]).unwrap();
    fs::File::options()
        .write(true)
        .open(to)
        .unwrap()
        .set_len((8 + n) as u64 + end)
        .unwrap();
}

type Tensors = BTreeMap<(String, String), (Dtype, Vec<u64>, Vec<u8>)>;

#[derive(Default)]
struct Collect {
    tensors: Tensors,
    /// (component, key) in the order the import put them.
    order: Vec<(String, String)>,
    layout: Layout,
    metadata: Metadata,
    imported: Option<Imported>,
}

impl Sink for Collect {
    fn begin(&mut self, _: &str, layout: &Layout) -> Result<()> {
        self.layout = layout.clone();
        Ok(())
    }
    fn put(
        &mut self,
        component: &str,
        key: &str,
        dtype: &Dtype,
        shape: &[u64],
        encoding: Option<&Encoding>,
        mut parts: Vec<Part>,
    ) -> Result<()> {
        assert!(
            encoding.is_none() && parts.len() == 1,
            "{component}/{key}: plain"
        );
        let mut bytes = Vec::new();
        parts[0]
            .data
            .read_to_end(&mut bytes)
            .map_err(|e| tensorfs_core::err::Refusal {
                code: Code::IO_FAILED,
                detail: e.to_string(),
            })?;
        assert_eq!(
            Some(bytes.len() as u64),
            dtype.byte_len(shape),
            "{component}/{key}"
        );
        self.tensors.insert(
            (component.into(), key.into()),
            (dtype.clone(), shape.to_vec(), bytes),
        );
        self.order.push((component.into(), key.into()));
        Ok(())
    }
    fn commit(&mut self, metadata: Metadata) -> Result<()> {
        self.metadata = metadata;
        Ok(())
    }
}

fn run(tmp: &Path, sources: &[String], converter: Option<&str>) -> Result<Collect> {
    let mut sink = Collect::default();
    let access = Access::default();
    let req = Request {
        sources,
        converter,
        variant: None,
        access: &access,
        tmp,
        progress: &|_, _, _| {},
        stop: None,
    };
    sink.imported = Some(import::convert(&req, &mut sink)?);
    // Sources never outlive the import, and never land anywhere but its tmp area.
    assert_eq!(areas(tmp), 0);
    Ok(sink)
}

/// Import areas left under `tmp` (their lock files stay).
fn areas(tmp: &Path) -> usize {
    fs::read_dir(tmp.join("import"))
        .map(|d| d.flatten().filter(|e| e.path().is_dir()).count())
        .unwrap_or(0)
}

fn identity(g: &Group) -> String {
    g.label()
}

// ------------------------------------------------------------------------------ the engine

/// numpy semantics, element by element: the source flat index of every output element.
fn naive(shape: &[u64], ops: &Ops) -> Vec<u64> {
    let numel: u64 = shape.iter().product();
    let resolve = |dims: &[i64], n: u64| -> Vec<u64> {
        let known: u64 = dims.iter().filter(|d| **d > 0).map(|d| *d as u64).product();
        dims.iter()
            .map(|d| if *d == -1 { n / known } else { *d as u64 })
            .collect()
    };
    let view = ops
        .view
        .as_ref()
        .map_or(shape.to_vec(), |v| resolve(v, numel));
    let strides: Vec<u64> = (0..view.len())
        .map(|i| view[i + 1..].iter().product())
        .collect();
    // Axis lists after take: (size, map from index to the view's index on that axis).
    let mut axes: Vec<(usize, Vec<u64>)> = (0..view.len())
        .map(|a| (a, (0..view[a]).collect()))
        .collect();
    if let Some((axis, idx)) = &ops.take {
        axes[*axis].1 = idx.clone();
    }
    if let Some(p) = &ops.permute {
        axes = p.iter().map(|i| axes[*i].clone()).collect();
    }
    let sizes: Vec<u64> = axes.iter().map(|(_, m)| m.len() as u64).collect();
    let total: u64 = sizes.iter().product();
    (0..total)
        .map(|mut n| {
            let mut at = 0;
            for (k, (axis, map)) in axes.iter().enumerate().rev() {
                let i = n % sizes[k];
                n /= sizes[k];
                at += map[i as usize] * strides[*axis];
            }
            at
        })
        .collect()
}

/// The engine's bytes for `ops` over a `shape` of u32 element indices.
fn through_engine(shape: &[u64], ops: &Ops) -> Vec<u64> {
    let numel: u64 = shape.iter().product();
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("tfs-import-ops-{}-{n}", std::process::id()));
    fs::write(
        &path,
        (0..numel as u32)
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
    .unwrap();
    let (out, runs) = engine::layout("ops", shape, ops).unwrap();
    let read = Reading {
        file: 0,
        start: 0,
        len: numel * 4,
        unit: 4,
        runs,
        source: Dtype::U32,
        cast: None,
        inline: None,
    };
    let file = fs::File::open(&path).unwrap();
    let mut bytes = Vec::new();
    TensorReader::new(&file, &read, "c/k".into())
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    fs::remove_file(&path).unwrap();
    assert_eq!(bytes.len() as u64, out.iter().product::<u64>() * 4);
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c) as u64)
        .collect()
}

fn ops(json: serde_json::Value) -> Ops {
    serde_json::from_value(json).unwrap()
}

#[test]
fn every_table_parses_with_unique_targets() {
    let tables = engine::tables();
    let names: Vec<&str> = tables.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "sdxl.single_file",
            "anima.single_file",
            "h3.native",
            "h3.lora",
            "sdxl.lora",
            "seedvr2"
        ]
    );
    for t in tables {
        for c in &t.components {
            let mut seen = std::collections::BTreeSet::new();
            for r in &c.rows {
                let source = match &r.source {
                    engine::Src::Key(k) => k.clone(),
                    engine::Src::Raw(raw) => raw.sha256.clone(),
                };
                assert!(
                    seen.insert((r.target.clone(), source)),
                    "{}/{}: {} twice",
                    t.name,
                    c.name,
                    r.target
                );
                assert_eq!(
                    t.lora.is_some(),
                    r.ops.io.is_some(),
                    "{}/{}",
                    t.name,
                    r.target
                );
            }
        }
    }
}

#[test]
fn ops_follow_numpy_semantics() {
    let cases = [
        (vec![6, 4], serde_json::json!({})),
        (
            vec![6, 4],
            serde_json::json!({"view": [3, -1], "take": [0, [1]], "shape": [2, 4]}),
        ),
        (
            vec![12, 2],
            serde_json::json!({"view": [2, 3, -1], "take": [1, [2]], "shape": [4, -1]}),
        ),
        (
            vec![8, 3],
            serde_json::json!({"view": [2, -1], "take": [0, [1, 0]], "shape": [8, -1]}),
        ),
        (vec![3, 5], serde_json::json!({"permute": [1, 0]})),
        (vec![4, 4, 1, 1], serde_json::json!({"shape": [4, 4]})),
        (
            vec![2, 3, 4],
            serde_json::json!({"take": [2, [3, 0, 2]], "permute": [2, 0, 1]}),
        ),
        (
            vec![24],
            serde_json::json!({"view": [2, 3, 4], "take": [1, [0, 2]], "permute": [1, 2, 0], "shape": [-1]}),
        ),
        (vec![5000, 3], serde_json::json!({"permute": [1, 0]})),
    ];
    for (shape, json) in cases {
        let o = ops(json.clone());
        assert_eq!(
            through_engine(&shape, &o),
            naive(&shape, &o),
            "{shape:?} {json}"
        );
    }
    // Every op the built-in tables use, over its real source rows (trailing dims shrunk where
    // the op leaves them free).
    let mut forms = BTreeMap::new();
    for t in engine::tables() {
        for c in &t.components {
            for r in &c.rows {
                for o in [Some(&r.ops), r.ops.b.as_deref()].into_iter().flatten() {
                    if o.view.is_some() || o.permute.is_some() || o.shape.is_some() {
                        forms.insert(format!("{o:?}"), (t.name.clone(), r.target.clone()));
                    }
                }
            }
        }
    }
    assert!(forms.len() >= 8, "{forms:?}");
    let sdxl =
        header_shapes(&vectors().join("civitai-sdxl-headers/128078-92696.safetensors.header"));
    for t in engine::tables() {
        for c in &t.components {
            for r in &c.rows {
                for o in [Some(&r.ops), r.ops.b.as_deref()].into_iter().flatten() {
                    if forms.remove(&format!("{o:?}")).is_none() {
                        continue;
                    }
                    let key = match &r.source {
                        engine::Src::Key(k) => format!("{}{k}", c.prefix),
                        engine::Src::Raw(_) => continue,
                    };
                    let shape = match (
                        c.expect.get(&key[c.prefix.len()..]),
                        sdxl.get(&key),
                        r.ops.io,
                    ) {
                        (Some(s), _, _) => vec![s[0], 2],
                        (_, Some(s), _) => s.clone(),
                        (_, _, Some((_, out))) => vec![out, 2],
                        _ => continue,
                    };
                    assert_eq!(
                        through_engine(&shape, o),
                        naive(&shape, o),
                        "{}: {} {o:?}",
                        t.name,
                        r.target
                    );
                }
            }
        }
    }
}

fn header_shapes(path: &Path) -> BTreeMap<String, Vec<u64>> {
    let raw = fs::read(path).unwrap();
    let n = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let doc: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&raw[8..8 + n]).unwrap();
    doc.into_iter()
        .filter(|(k, _)| k != "__metadata__")
        .map(|(k, v)| {
            (
                k,
                v["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_u64().unwrap())
                    .collect(),
            )
        })
        .collect()
}

#[test]
fn narrowing_is_numpys_and_refuses_overflow() {
    // (f32 bits, numpy 2.5.2 `astype('<f2')` bits): specials, rounding both ways, ties,
    // subnormals, the overflow edge and NaN payloads.
    let numpy: [(u32, u16); 16] = [
        (0x7f80_0001, 0x7c01),
        (0x7fc0_0000, 0x7e00),
        (0xffc0_0001, 0xfe00),
        (0x7f80_2001, 0x7c01),
        (0x7fa0_0000, 0x7d00),
        (0x477f_efff, 0x7bff),
        (0x477f_f000, 0x7c00),
        (0x3300_0001, 0x0001),
        (0x3300_0000, 0x0000),
        (0x387f_e000, 0x0400),
        (0x3800_0000, 0x0200),
        (0x8000_0000, 0x8000),
        (0x3f80_0000, 0x3c00),
        (0x3f80_1000, 0x3c00),
        (0x3f80_3000, 0x3c02),
        (0xc000_0000, 0xc000),
    ];
    for (input, want) in numpy {
        assert_eq!(engine::f32_to_f16(input), want, "{input:#010x}");
    }
    let dir = scratch("narrow");
    let path = dir.join("x");
    for (values, ok) in [
        (vec![1.0f32, 65504.0, f32::INFINITY, -2.0], true),
        (vec![1.0f32, 70000.0], false),
    ] {
        fs::write(
            &path,
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();
        let n = values.len() as u64;
        let (_, runs) = engine::layout("x", &[n], &Ops::default()).unwrap();
        let read = Reading {
            file: 0,
            start: 0,
            len: n * 4,
            unit: 4,
            runs,
            source: Dtype::F32,
            cast: Some(Dtype::F16),
            inline: None,
        };
        let file = fs::File::open(&path).unwrap();
        let mut out = Vec::new();
        let got = TensorReader::new(&file, &read, "c/k".into())
            .unwrap()
            .read_to_end(&mut out);
        assert_eq!(got.is_ok(), ok);
        if ok {
            assert_eq!(out, [0x00, 0x3c, 0xff, 0x7b, 0x00, 0x7c, 0x00, 0xc0]);
        } else {
            assert!(got
                .unwrap_err()
                .to_string()
                .contains("element 1 (70000) overflows F16"));
        }
    }
    fs::remove_dir_all(&dir).unwrap();
}

// ------------------------------------------------------------------------------ recognition

/// The banked corpus as sparse files under `root`: `(source path, banked file)`.
fn corpus(root: &Path, rows: &[(&str, &str)]) -> SourceSet {
    let mut members = Vec::new();
    for (path, banked) in rows {
        let to = root.join(path);
        if path.ends_with(".json") {
            fs::create_dir_all(to.parent().unwrap()).unwrap();
            fs::copy(vectors().join(banked), &to).unwrap();
        } else {
            sparse(&vectors().join(banked), &to);
        }
        members.push((path.to_string(), to));
    }
    let mut set = SourceSet::scan(&members).unwrap();
    // No real plain header reads as a quantized carrier.
    import::carriers::fold(&mut set).unwrap();
    set
}

fn recognized(set: &SourceSet, named: Option<&str>) -> Result<Vec<(String, String)>> {
    let tables = engine::tables();
    let r = engine::recognize(set, tables, named)?;
    let mut claims: Vec<(String, String)> = r
        .claims
        .iter()
        .map(|c| (tables[c.table].name.clone(), c.name.clone()))
        .collect();
    claims.extend(
        r.unclaimed
            .iter()
            .map(|g| ("as-is".into(), set.groups[*g].label())),
    );
    claims.sort();
    engine::plan(set, tables, &r, &identity)?;
    Ok(claims)
}

fn pairs(rows: &[(&str, &str)]) -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = rows
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
    v.sort();
    v
}

#[test]
fn the_real_header_corpus_recognizes_every_family() {
    let root = scratch("corpus");
    let sdxl = corpus(
        &root.join("sdxl"),
        &[(
            "sd_xl.safetensors",
            "civitai-sdxl-headers/128078-92696.safetensors.header",
        )],
    );
    assert_eq!(
        recognized(&sdxl, None).unwrap(),
        pairs(&[
            ("sdxl.single_file", "text_encoder"),
            ("sdxl.single_file", "text_encoder_2"),
            ("sdxl.single_file", "unet"),
            ("sdxl.single_file", "vae")
        ])
    );

    let anima = corpus(
        &root.join("anima"),
        &[
            (
                "anima.safetensors",
                "anima-civitai-headers/2945208-2824391.safetensors.header",
            ),
            (
                "qwen.safetensors",
                "anima-civitai-headers/2945208-2824387.safetensors.header",
            ),
            (
                "vae.safetensors",
                "anima-civitai-headers/2945208-2004692.safetensors.header",
            ),
        ],
    );
    let names = ["text_conditioner", "text_encoder", "transformer", "vae"];
    assert_eq!(
        recognized(&anima, None).unwrap(),
        pairs(&names.map(|n| ("anima.single_file", n)))
    );

    let mut h3 = Vec::new();
    for entry in fs::read_dir(vectors().join("h3-headers")).unwrap() {
        let name = entry.unwrap().file_name().into_string().unwrap();
        if name.ends_with(".safetensors") || name.ends_with(".index.json") {
            h3.push((name.replace("__", "/"), format!("h3-headers/{name}")));
        }
    }
    let h3_rows: Vec<(&str, &str)> = h3.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let h3 = corpus(&root.join("h3"), &h3_rows);
    assert_eq!(
        recognized(&h3, None).unwrap(),
        pairs(&[
            ("as-is", "audio_vae/diffusion_pytorch_model"),
            ("as-is", "text_encoder/model"),
            ("as-is", "vae/diffusion_pytorch_model"),
            ("h3.native", "transformer"),
            ("h3.native", "transformer_ref"),
        ])
    );

    for (file, components) in [
        (
            "Super_Eye_Detailer_By_Stable_Yogi_SDPD0.safetensors",
            vec!["text_encoder", "text_encoder_2", "unet"],
        ),
        ("DetailedEyes_V3.safetensors", vec!["unet"]),
    ] {
        let set = corpus(
            &root.join(file),
            &[(file, &format!("sdxl-lora-headers/{file}"))],
        );
        assert_eq!(
            recognized(&set, None).unwrap(),
            pairs(
                &components
                    .iter()
                    .map(|c| ("sdxl.lora", *c))
                    .collect::<Vec<_>>()
            ),
            "{file}"
        );
    }

    for entry in fs::read_dir(vectors().join("h3-lora-headers")).unwrap() {
        let file = entry.unwrap().file_name().into_string().unwrap();
        if !file.ends_with(".safetensors") {
            continue;
        }
        let set = corpus(
            &root.join(&file),
            &[(&file, &format!("h3-lora-headers/{file}"))],
        );
        if file.starts_with("minimax_h3_turbo") {
            // Bare native names and AdaLN factors: no table's layout, so stored as it is.
            assert_eq!(
                recognized(&set, None).unwrap(),
                pairs(&[("as-is", file.trim_end_matches(".safetensors"))])
            );
            assert_eq!(
                recognized(&set, Some("h3.lora")).unwrap_err().code,
                Code::MISSING_TENSOR
            );
            continue;
        }
        // Contiguous and head-interleaved QKV factors have identical headers: only by name.
        assert_eq!(
            recognized(&set, None).unwrap_err().code,
            Code::AMBIGUOUS_CLASSIFICATION,
            "{file}"
        );
        assert_eq!(
            recognized(&set, Some("h3.lora")).unwrap(),
            pairs(&[("h3.lora", "transformer")]),
            "{file}"
        );
    }

    // Diffusers, Transformers and adapter layouts no table converts: stored as they are.
    let sdxl_hf = corpus(
        &root.join("sdxl-hf"),
        &[
            (
                "unet/diffusion_pytorch_model.fp16.safetensors",
                "sdxl-headers/unet__diffusion_pytorch_model.fp16.safetensors.header",
            ),
            (
                "text_encoder/model.fp16.safetensors",
                "sdxl-headers/text_encoder__model.fp16.safetensors.header",
            ),
        ],
    );
    assert!(recognized(&sdxl_hf, None)
        .unwrap()
        .iter()
        .all(|(t, _)| t == "as-is"));
    let pdd = corpus(
        &root.join("pdd"),
        &[(
            "acc.safetensors",
            "h3-pdd-headers/MiniMax-H3-FL2VA-Acc-8Step.safetensors",
        )],
    );
    assert_eq!(recognized(&pdd, None).unwrap(), pairs(&[("as-is", "acc")]));
    // A named table that does not fit names what differs.
    let refusal = recognized(&pdd, Some("sdxl.single_file")).unwrap_err();
    assert_eq!(refusal.code, Code::MISSING_TENSOR);
    fs::remove_dir_all(&root).unwrap();
}

// ------------------------------------------------------------------------------ the pipeline

fn f32s(n: u64, seed: u64) -> Vec<u8> {
    (0..n)
        .flat_map(|i| ((seed * 1_000_003 + i) as f32).to_le_bytes())
        .collect()
}

#[test]
fn an_unknown_layout_is_stored_as_is() {
    let dir = scratch("as-is");
    let file = dir.join("in/mystery.safetensors");
    safetensors(
        &file,
        &[
            ("a.weight", "BF16", vec![2, 3], vec![7; 12]),
            ("b", "F4", vec![4], vec![0x21, 0x43]),
        ],
    );
    let out = run(&dir.join("tmp"), &[file.display().to_string()], None).unwrap();
    assert!(!out.imported.as_ref().unwrap().normalized);
    assert!(out.imported.unwrap().converters.is_empty());
    assert_eq!(
        out.tensors[&("model".into(), "a.weight".into())],
        (Dtype::BF16, vec![2, 3], vec![7; 12])
    );
    assert_eq!(
        out.tensors[&("model".into(), "b".into())],
        (Dtype::F4, vec![4], vec![0x21, 0x43])
    );
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn h3_lora_converts_only_when_named() {
    let dir = scratch("h3-lora");
    let file = dir.join("in/wushu.safetensors");
    let rank = 2;
    let mut tensors = Vec::new();
    for (flat, input, output) in [
        ("attn_qkv_proj", 5376, 21504),
        ("attn_out_proj", 7168, 5376),
        ("mlp_fc1", 5376, 28672),
    ] {
        let stem = format!("lora_unet_blocks_7_{flat}");
        tensors.push((
            format!("{stem}.lora_down.weight"),
            vec![rank, input],
            f32s(rank * input, 1),
        ));
        tensors.push((
            format!("{stem}.lora_up.weight"),
            vec![output, rank],
            f32s(output * rank, 2),
        ));
        tensors.push((format!("{stem}.alpha"), vec![], 8f32.to_le_bytes().to_vec()));
    }
    let views: Vec<(&str, &str, Vec<u64>, Vec<u8>)> = tensors
        .iter()
        .map(|(k, s, b)| (k.as_str(), "F32", s.clone(), b.clone()))
        .collect();
    safetensors(&file, &views);
    let sources = [file.display().to_string()];
    assert_eq!(
        run(&dir.join("tmp"), &sources, None).err().unwrap().code,
        Code::AMBIGUOUS_CLASSIFICATION
    );
    let out = run(&dir.join("tmp"), &sources, Some("h3.lora")).unwrap();
    assert_eq!(out.imported.as_ref().unwrap().converters, ["h3.lora"]);
    let get = |k: &str| &out.tensors[&("transformer".into(), k.into())].2;
    let b = f32s(21504 * rank, 2);
    let third = (7168 * rank * 4) as usize;
    for (i, leaf) in ["to_q", "to_k", "to_v"].iter().enumerate() {
        assert_eq!(
            get(&format!("transformer_blocks.7.attn.{leaf}.lora_A.weight")),
            &f32s(rank * 5376, 1)
        );
        assert_eq!(
            get(&format!("transformer_blocks.7.attn.{leaf}.lora_B.weight")),
            &b[i * third..(i + 1) * third].to_vec()
        );
        assert_eq!(
            get(&format!("transformer_blocks.7.attn.{leaf}.alpha")),
            &8f32.to_le_bytes().to_vec()
        );
    }
    let fc1 = f32s(28672 * rank, 2);
    let half = fc1.len() / 2;
    assert_eq!(
        get("transformer_blocks.7.ff.net.0.proj.lora_B.weight"),
        &[&fc1[half..], &fc1[..half]].concat()
    );
    assert_eq!(
        get("transformer_blocks.7.attn.to_out.0.lora_B.weight"),
        &f32s(5376 * rank, 2)
    );
    assert_eq!(out.tensors.len(), 5 * 3);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn sdxl_lora_lands_on_the_canonical_modules() {
    let dir = scratch("sdxl-lora");
    let file = dir.join("in/eyes.safetensors");
    let up = f32s(640 * 4, 3);
    safetensors(
        &file,
        &[
            (
                "lora_unet_input_blocks_4_1_transformer_blocks_0_attn2_to_k.lora_down.weight",
                "F16",
                vec![4, 2048],
                vec![1; 4 * 2048 * 2],
            ),
            (
                "lora_unet_input_blocks_4_1_transformer_blocks_0_attn2_to_k.lora_up.weight",
                "F32",
                vec![640, 4],
                up.clone(),
            ),
            (
                "lora_te1_text_model_encoder_layers_3_mlp_fc1.lora_down.weight",
                "F16",
                vec![4, 768],
                vec![2; 4 * 768 * 2],
            ),
            (
                "lora_te1_text_model_encoder_layers_3_mlp_fc1.lora_up.weight",
                "F16",
                vec![3072, 4],
                vec![3; 3072 * 4 * 2],
            ),
        ],
    );
    // A and B disagree in dtype: refused, never coerced.
    assert_eq!(
        run(&dir.join("tmp"), &[file.display().to_string()], None)
            .err()
            .unwrap()
            .code,
        Code::SHAPE_MISMATCH
    );
    safetensors(
        &file,
        &[
            (
                "lora_unet_input_blocks_4_1_transformer_blocks_0_attn2_to_k.lora_down.weight",
                "F32",
                vec![4, 2048],
                f32s(4 * 2048, 4),
            ),
            (
                "lora_unet_input_blocks_4_1_transformer_blocks_0_attn2_to_k.lora_up.weight",
                "F32",
                vec![640, 4],
                up.clone(),
            ),
            (
                "lora_unet_input_blocks_4_1_transformer_blocks_0_attn2_to_k.alpha",
                "F32",
                vec![],
                4f32.to_le_bytes().to_vec(),
            ),
            (
                "lora_te1_text_model_encoder_layers_3_mlp_fc1.lora_down.weight",
                "F16",
                vec![4, 768],
                vec![2; 4 * 768 * 2],
            ),
            (
                "lora_te1_text_model_encoder_layers_3_mlp_fc1.lora_up.weight",
                "F16",
                vec![3072, 4],
                vec![3; 3072 * 4 * 2],
            ),
        ],
    );
    let out = run(&dir.join("tmp"), &[file.display().to_string()], None).unwrap();
    let keys: Vec<(String, String)> = out.tensors.keys().cloned().collect();
    assert_eq!(
        keys,
        pairs(&[
            (
                "text_encoder",
                "text_model.encoder.layers.3.mlp.fc1.lora_A.weight"
            ),
            (
                "text_encoder",
                "text_model.encoder.layers.3.mlp.fc1.lora_B.weight"
            ),
            (
                "unet",
                "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.alpha"
            ),
            (
                "unet",
                "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_A.weight"
            ),
            (
                "unet",
                "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_B.weight"
            ),
        ])
    );
    assert_eq!(
        out.tensors[&(
            "unet".into(),
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_B.weight".into()
        )]
            .2,
        up
    );
    assert!(out.imported.unwrap().normalized);
    fs::remove_dir_all(&dir).unwrap();
}

/// network: Hugging Face.
#[test]
fn a_tiny_hf_sdxl_imports_end_to_end() {
    let dir = scratch("tiny-sdxl");
    let source = "hf://hf-internal-testing/tiny-sdxl-pipe".to_string();
    let out = run(&dir.join("tmp"), std::slice::from_ref(&source), None).unwrap();
    let imported = out.imported.as_ref().unwrap();
    assert!(imported.normalized && imported.converters.is_empty());
    assert!(imported.sources[0].starts_with("hf://hf-internal-testing/tiny-sdxl-pipe@"));
    let components: std::collections::BTreeSet<&str> =
        out.tensors.keys().map(|(c, _)| c.as_str()).collect();
    assert_eq!(
        components.into_iter().collect::<Vec<_>>(),
        ["text_encoder", "text_encoder_2", "unet", "vae"]
    );
    // D10: the pipeline, every component with its class, config and run files; README and
    // .gitattributes are repo files.
    let layout = &out.layout;
    let pipeline = layout.pipeline.as_ref().unwrap();
    assert_eq!(pipeline.class, "diffusers.StableDiffusionXLPipeline");
    let classes: Vec<(&str, &str, Option<&str>)> = (layout.components.iter())
        .map(|c| (c.name.as_str(), c.class.as_str(), c.config_file.as_deref()))
        .collect();
    assert_eq!(
        classes,
        [
            ("text_encoder", "transformers.CLIPTextModel", None),
            ("text_encoder_2", "transformers.CLIPTextModelWithProjection", None),
            ("unet", "diffusers.UNet2DConditionModel", None),
            ("vae", "diffusers.AutoencoderKL", None),
            ("scheduler", "diffusers.DDIMScheduler", Some("scheduler_config.json")),
            ("tokenizer", "transformers.CLIPTokenizer", Some("tokenizer_config.json")),
            ("tokenizer_2", "transformers.CLIPTokenizer", Some("tokenizer_config.json")),
        ]
    );
    let tokenizer = &layout.components[5];
    assert_eq!(
        tokenizer.assets.keys().collect::<Vec<_>>(),
        ["merges.txt", "special_tokens_map.json", "vocab.json"]
    );
    assert_eq!(
        layout.files.keys().collect::<Vec<_>>(),
        [".gitattributes", "README.md"]
    );
    assert_eq!(
        out.metadata["source"],
        tensorfs_core::format::Value::Str(imported.sources[0].clone())
    );
    // Bit-identical to the repository's own files, landed separately.
    let access = Access::default();
    let listing = import::remote::list(
        &import::remote::parse(&imported.sources[0]).unwrap(),
        &access,
    )
    .unwrap();
    let members = import::remote::select(listing, None, import::remote::Files::All, &|m| {
        import::remote::read_small(m, &access)
    })
    .unwrap();
    let landed: Vec<(String, PathBuf)> = members
        .iter()
        .map(|m| {
            (
                m.path.clone(),
                import::remote::land(m, &dir.join("check").join(&m.path), &access).unwrap(),
            )
        })
        .collect();
    let set = SourceSet::scan(&landed).unwrap();
    let mut checked = 0;
    for g in &set.groups {
        for (key, t) in &g.tensors {
            let (dtype, shape, bytes) = &out.tensors[&(g.folder.clone(), key.clone())];
            let mut want = vec![0u8; bytes.len()];
            platform::durable::FileAt::read_exact_at(
                &fs::File::open(&set.files[t.file]).unwrap(),
                &mut want,
                t.start,
            )
            .unwrap();
            assert_eq!((dtype, shape, bytes), (&t.dtype, &t.shape, &want), "{key}");
            checked += 1;
        }
    }
    assert_eq!(checked, out.tensors.len());

    // Construction order, not the files' sorted order: each component equals the order
    // diffusers/transformers build it in (generated by scripts/orders.py, never a reference).
    let expected: serde_json::Value = serde_json::from_slice(
        &fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("vectors/orders/tiny-sdxl-pipe.json"))
            .unwrap(),
    )
    .unwrap();
    for (component, keys) in expected["components"].as_object().unwrap() {
        let built: Vec<&str> = keys
            .as_array()
            .unwrap()
            .iter()
            .map(|k| k.as_str().unwrap())
            .collect();
        let put: Vec<&str> = (out.order.iter())
            .filter(|(c, _)| c == component)
            .map(|(_, k)| k.as_str())
            .collect();
        assert!(
            put.iter().all(|k| built.contains(k)),
            "{component}: a key the model lacks"
        );
        let want: Vec<&str> = built.iter().copied().filter(|k| put.contains(k)).collect();
        assert_eq!(put, want, "{component} is in construction order");
    }
    fs::remove_dir_all(&dir).unwrap();
}

/// The generated references order a source's sorted keys into the constructor order that the
/// converter tables (generated from the same classes' state_dict) list, for every SDXL and
/// Anima component: two independent generators agree.
#[test]
fn references_reproduce_the_tables_constructor_order() {
    let class = |table: &str, component: &str| match (table, component) {
        (_, "unet") => "UNet2DConditionModel",
        ("sdxl.single_file", "vae") => "AutoencoderKL",
        ("sdxl.single_file", "text_encoder") => "CLIPTextModel",
        (_, "text_encoder_2") => "CLIPTextModelWithProjection",
        ("anima.single_file", "vae") => "AutoencoderKLQwenImage",
        ("anima.single_file", "text_encoder") => "Qwen3Model",
        (_, "text_conditioner") => "AnimaTextConditioner",
        (_, "transformer") => "CosmosTransformer3DModel",
        other => panic!("{other:?}"),
    };
    let mut checked = 0;
    for table in import::tables()
        .iter()
        .filter(|t| t.name.ends_with(".single_file"))
    {
        for c in &table.components {
            let built: Vec<&str> = c.rows.iter().map(|r| r.target.as_str()).collect();
            // A safetensors file lists its keys sorted, and its bytes in that order.
            let mut stored = built.clone();
            stored.sort_unstable();
            let keys: Vec<(&str, usize)> =
                stored.iter().enumerate().map(|(i, k)| (*k, i)).collect();
            let ordered = import::constructor_order(Some(class(&table.name, &c.name)), &keys);
            let unknown: Vec<&str> = ordered.unknown.iter().map(|i| keys[*i].0).collect();
            assert!(unknown.is_empty(), "{}/{}: {unknown:?}", table.name, c.name);
            let got: Vec<&str> = ordered.order.iter().map(|i| keys[*i].0).collect();
            assert_eq!(got, built, "{}/{}", table.name, c.name);
            checked += 1;
        }
    }
    assert_eq!(checked, 8, "SDXL's and Anima's four components each");

    // H3's transformer, from the h3.native table (its rows are the transformer's census).
    let h3 = import::tables()
        .iter()
        .find(|t| t.name == "h3.native")
        .unwrap();
    let built: Vec<&str> = h3.components[0]
        .rows
        .iter()
        .map(|r| r.target.as_str())
        .collect();
    let mut stored = built.clone();
    stored.sort_unstable();
    let keys: Vec<(&str, usize)> = stored.iter().enumerate().map(|(i, k)| (*k, i)).collect();
    let ordered = import::constructor_order(Some("MiniMaxH3Transformer3DModel"), &keys);
    assert!(ordered.unknown.is_empty());
    let got: Vec<&str> = ordered.order.iter().map(|i| keys[*i].0).collect();
    assert_eq!(got, built, "h3.native");
}

/// A layout no reference covers keeps its source's data order.
#[test]
fn an_unknown_layout_keeps_its_data_order() {
    let keys = [("zeta.weight", 2), ("alpha.weight", 0), ("mid.0.bias", 1)];
    let unnamed = import::constructor_order(None, &keys);
    assert_eq!(unnamed.order, [1, 2, 0]);
    assert_eq!(unnamed.unknown, [0, 1, 2], "every key is reported");
    // A class with a reference: what it knows first, in its order; the rest in data order.
    let keys = [
        ("zeta.weight", 0),
        ("conv_in.bias", 1),
        ("conv_in.weight", 2),
    ];
    let unet = import::constructor_order(Some("UNet2DConditionModel"), &keys);
    assert_eq!(
        unet.order,
        [2, 1, 0],
        "weight before bias, then the unknown key"
    );
    assert_eq!(unet.unknown, [0]);
}

/// network: GitHub (SeedVR's conditioning constants, 1.2 MB).
#[test]
fn seedvr2_composes_its_conditioning_from_the_pinned_files() {
    let dir = scratch("seedvr2");
    let base = "https://raw.githubusercontent.com/ByteDance-Seed/SeedVR/e4de8c24441a67e1b7df56abea10645059bb1185";
    let access = Access::default();
    let mut members = Vec::new();
    for name in ["pos_emb.pt", "neg_emb.pt"] {
        let listing = import::remote::list(
            &import::remote::parse(&format!("{base}/{name}")).unwrap(),
            &access,
        )
        .unwrap();
        let path = import::remote::land(&listing.members[0], &dir.join(name), &access).unwrap();
        members.push((name.to_string(), path));
    }
    for (name, banked) in [
        ("seedvr2_ema_7b_fp16.safetensors", "dit"),
        ("ema_vae_fp16.safetensors", "vae"),
    ] {
        sparse(
            &vectors().join(format!("seedvr2-headers/{name}.header")),
            &dir.join(name),
        );
        members.push((name.to_string(), dir.join(name)));
        let _ = banked;
    }
    let set = SourceSet::scan(&members).unwrap();
    let tables = engine::tables();
    let r = engine::recognize(&set, tables, None).unwrap();
    assert!(r.unclaimed.is_empty());
    let planned = engine::plan(&set, tables, &r, &identity).unwrap();
    let positive = planned
        .iter()
        .find(|p| p.component == "conditioning" && p.key == "positive")
        .unwrap();
    assert_eq!(
        (positive.dtype.clone(), positive.shape.clone()),
        (Dtype::BF16, vec![58, 5120])
    );
    let mut bytes = Vec::new();
    let read = &positive.parts[0].read;
    let file = fs::File::open(&set.files[read.file]).unwrap();
    TensorReader::new(&file, read, "conditioning/positive".into())
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    let pt = fs::read(dir.join("pos_emb.pt")).unwrap();
    assert_eq!(bytes, pt[448..448 + 58 * 5120 * 2]);
    assert_eq!(
        planned.iter().filter(|p| p.component == "dit").count(),
        1128
    );
    assert_eq!(planned.iter().filter(|p| p.component == "vae").count(), 250);
    fs::remove_dir_all(&dir).unwrap();
}

fn sha256(bytes: &[u8]) -> String {
    tensorfs_core::sha256::hex_digest(bytes)
}

/// network, about 1.1 GB: block 0's fused QKV and SwiGLU fc1 of the real native FL2VA
/// transformer, converted by h3.native, equal the Diffusers packaging's `to_q`/`to_k`/`to_v`
/// and `ff.net.0.proj` at the same revision byte for byte (the bit-identity tfs-094 measured).
#[test]
#[ignore]
fn h3_native_real_bytes_equal_the_diffusers_packaging() {
    let dir = scratch("h3-real");
    let access = Access::default();
    let repo = "hf://MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe";
    let listing = import::remote::list(&import::remote::parse(repo).unwrap(), &access).unwrap();
    let member = |path: &str| {
        listing
            .members
            .iter()
            .find(|m| m.path == path)
            .unwrap_or_else(|| panic!("{path}"))
            .clone()
    };
    let mut members = Vec::new();
    for entry in fs::read_dir(vectors().join("h3-headers")).unwrap() {
        let name = entry.unwrap().file_name().into_string().unwrap();
        if let Some(rest) = name.strip_prefix("FL2VA__transformer__") {
            let path = format!("FL2VA/transformer/{rest}");
            let to = dir.join(&path);
            if rest.ends_with(".json") {
                fs::create_dir_all(to.parent().unwrap()).unwrap();
                fs::copy(vectors().join("h3-headers").join(&name), &to).unwrap();
            } else {
                sparse(&vectors().join("h3-headers").join(&name), &to);
            }
            members.push((path, to));
        }
    }
    let set = SourceSet::scan(&members).unwrap();
    for key in ["blocks.0.attn.qkv_proj.weight", "blocks.0.mlp.fc1.weight"] {
        let t = &set.groups[0].tensors[key];
        let len = t.dtype.byte_len(&t.shape).unwrap();
        let path = &set.files[t.file];
        let shard = format!(
            "FL2VA/transformer/{}",
            path.file_name().unwrap().to_str().unwrap()
        );
        let bytes =
            import::remote::read_range(&member(&shard), t.start, t.start + len, &access).unwrap();
        platform::durable::FileAt::write_all_at(
            &fs::File::options().write(true).open(path).unwrap(),
            &bytes,
            t.start,
        )
        .unwrap();
    }
    let tables = engine::tables();
    let r = engine::recognize(&set, tables, None).unwrap();
    let planned = engine::plan(&set, tables, &r, &identity).unwrap();
    let index: BTreeMap<String, serde_json::Value> = serde_json::from_slice(
        &import::remote::read_small(
            &member("transformer/diffusion_pytorch_model.safetensors.index.json"),
            &access,
        )
        .unwrap(),
    )
    .unwrap();
    let targets: std::collections::BTreeSet<&str> =
        planned.iter().map(|p| p.key.as_str()).collect();
    assert!(index["weight_map"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .eq(targets.iter().copied()));

    let shard = member("transformer/diffusion_pytorch_model-00001-of-00014.safetensors");
    let n = u64::from_le_bytes(
        import::remote::read_range(&shard, 0, 8, &access)
            .unwrap()
            .try_into()
            .unwrap(),
    );
    let head: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&import::remote::read_range(&shard, 8, 8 + n, &access).unwrap())
            .unwrap();
    for key in [
        "attn.to_q.weight",
        "attn.to_k.weight",
        "attn.to_v.weight",
        "ff.net.0.proj.weight",
    ] {
        let key = format!("transformer_blocks.0.{key}");
        let p = planned.iter().find(|p| p.key == key).unwrap();
        let mut ours = Vec::new();
        let read = &p.parts[0].read;
        TensorReader::new(
            &fs::File::open(&set.files[read.file]).unwrap(),
            read,
            key.clone(),
        )
        .unwrap()
        .read_to_end(&mut ours)
        .unwrap();
        let at = &head[&key]["data_offsets"];
        let (begin, end) = (
            8 + n + at[0].as_u64().unwrap(),
            8 + n + at[1].as_u64().unwrap(),
        );
        let theirs = import::remote::read_range(&shard, begin, end, &access).unwrap();
        assert_eq!(sha256(&ours), sha256(&theirs), "{key}");
    }
    fs::remove_dir_all(&dir).unwrap();
}

/// Keys, shapes and byte counts only: for sources too large to hold.
#[derive(Default)]
struct Count {
    tensors: BTreeMap<(String, String), (Dtype, Vec<u64>)>,
    layout: Layout,
}

impl Sink for Count {
    fn begin(&mut self, _: &str, layout: &Layout) -> Result<()> {
        self.layout = layout.clone();
        Ok(())
    }
    fn put(
        &mut self,
        component: &str,
        key: &str,
        dtype: &Dtype,
        shape: &[u64],
        _: Option<&Encoding>,
        parts: Vec<Part>,
    ) -> Result<()> {
        for p in parts {
            let n = std::io::copy(p.data, &mut std::io::sink()).unwrap();
            assert_eq!(Some(n), p.dtype.byte_len(p.shape), "{component}/{key}");
        }
        self.tensors.insert(
            (component.into(), key.into()),
            (dtype.clone(), shape.to_vec()),
        );
        Ok(())
    }
    fn commit(&mut self, _: Metadata) -> Result<()> {
        Ok(())
    }
}

/// network: Hugging Face (the reference's documents). A Civitai SDXL single file's geometry
/// becomes exactly the upstream Diffusers repository's tensors, with its configs.
#[test]
fn a_single_file_sdxl_becomes_the_upstream_layout() {
    let dir = scratch("sdxl-single");
    let file = dir.join("in/juggernaut.safetensors");
    sparse(
        &vectors().join("civitai-sdxl-headers/128078-92696.safetensors.header"),
        &file,
    );
    let mut sink = Count::default();
    let access = Access::default();
    let sources = [file.display().to_string()];
    let tmp = dir.join("tmp");
    let req = Request {
        sources: &sources,
        converter: None,
        variant: None,
        access: &access,
        tmp: &tmp,
        progress: &|_, _, _| {},
        stop: None,
    };
    let imported = import::convert(&req, &mut sink).unwrap();
    assert!(imported.normalized);
    assert_eq!(imported.converters, ["sdxl.single_file"]);
    for (component, banked) in [
        (
            "text_encoder",
            "text_encoder__model.fp16.safetensors.header",
        ),
        (
            "text_encoder_2",
            "text_encoder_2__model.fp16.safetensors.header",
        ),
        (
            "unet",
            "unet__diffusion_pytorch_model.fp16.safetensors.header",
        ),
        (
            "vae",
            "vae__diffusion_pytorch_model.fp16.safetensors.header",
        ),
    ] {
        let want = header_shapes(&vectors().join("sdxl-headers").join(banked));
        let got: BTreeMap<String, Vec<u64>> = sink
            .tensors
            .iter()
            .filter(|((c, _), _)| c == component)
            .map(|((_, k), (d, s))| {
                assert_eq!(*d, Dtype::F16);
                (k.clone(), s.clone())
            })
            .collect();
        assert_eq!(got, want, "{component}");
    }
    // The reference fills the pipeline, the configs and the tokenizers, and says so.
    let layout = &sink.layout;
    assert_eq!(
        layout.pipeline.as_ref().unwrap().class,
        "diffusers.StableDiffusionXLPipeline"
    );
    let folders: BTreeMap<&str, &import::Folder> =
        layout.components.iter().map(|c| (c.name.as_str(), c)).collect();
    assert_eq!(folders["scheduler"].config_file.as_deref(), Some("scheduler_config.json"));
    assert!(folders["tokenizer_2"].assets.contains_key("vocab.json"));
    for c in ["text_encoder", "unet", "vae"] {
        assert!(!folders[c].config.is_empty() && folders[c].config_file.is_none(), "{c}");
    }
    let (from, files) = &imported.filled[0];
    assert!(from.starts_with("hf://") && files.iter().any(|f| f == "unet/config.json"));
    assert_eq!(areas(&tmp), 0);
    fs::remove_dir_all(&dir).unwrap();
}

/// network: Civitai's public API (a listing, no download).
#[test]
fn sources_parse_and_pin() {
    use import::remote::{parse, Spec};
    let hf = |org: &str, repo: &str, revision: &str, path: &str| Spec::Hf {
        org: org.into(),
        repo: repo.into(),
        revision: revision.into(),
        path: path.into(),
    };
    for (raw, want) in [
        ("hf://a/b", hf("a", "b", "main", "")),
        (
            "hf://a/b@v1/unet/x.safetensors",
            hf("a", "b", "v1", "unet/x.safetensors"),
        ),
        (
            "https://huggingface.co/a/b/blob/c0ffee/x.safetensors",
            hf("a", "b", "c0ffee", "x.safetensors"),
        ),
        (
            "https://huggingface.co/a/b/tree/main/unet",
            hf("a", "b", "main", "unet"),
        ),
        (
            "civitai://128078",
            Spec::Civitai {
                version: 128078,
                file: None,
            },
        ),
        (
            "https://civitai.com/models/101055?modelVersionId=128078",
            Spec::Civitai {
                version: 128078,
                file: None,
            },
        ),
        (
            "https://civitai.com/api/download/models/128078",
            Spec::Civitai {
                version: 128078,
                file: None,
            },
        ),
        (
            "https://example.com/w/pos_emb.pt",
            Spec::Url("https://example.com/w/pos_emb.pt".into()),
        ),
        (
            "./model.safetensors",
            Spec::Local("./model.safetensors".into()),
        ),
    ] {
        assert_eq!(parse(raw).unwrap(), want, "{raw}");
    }
    let listing =
        import::remote::list(&parse("civitai://128078").unwrap(), &Access::default()).unwrap();
    assert!(listing.exact && listing.pinned.starts_with("civitai://128078/"));
    assert_eq!(
        listing.members[0].sha256.as_deref(),
        Some("e6bb9ea85bbf7bf6478a7c6d18b71246f22e95d41bcdd80ed40aa212c33cfeff")
    );
}

/// Each reference against the owner's real v1 lanes (read-only), whose headers keep the
/// construction order their producer wrote: from a lane's keys sorted (as a safetensors file
/// lists them), the class's reference restores the lane's order (weight before bias, blocks by
/// number) for every key it holds. Keys the reference lacks (v1 names that differ from
/// upstream's) are counted, not ordered.
#[test]
#[ignore = "reads the owner's ~/.tensorfs v1 lanes"]
fn references_match_real_lanes() {
    let home = PathBuf::from(std::env::var_os("HOME").unwrap()).join(".tensorfs");
    let blob = |sha: &str| {
        fs::read(
            home.join("blobs")
                .join(&sha[..2])
                .join(&sha[2..4])
                .join(sha),
        )
        .unwrap()
    };
    let mut held = 0;
    // (lane, manifest prefix, the class of components whose config does not name one)
    for (lane, prefix, unnamed) in [
        ("sdxl", "77cac5a0f4a5", None),
        ("anima bf16", "5d6a05188427", None),
        ("qwen-image-2.1 int8", "e7c79ee942b8", None),
        (
            "qwen3.5-9b int8",
            "e0f61ce1ab03",
            Some("Qwen3_5ForConditionalGeneration"),
        ),
    ] {
        let dir = home
            .join("manifests")
            .join(&prefix[..2])
            .join(&prefix[2..4]);
        let manifest = (fs::read_dir(&dir).unwrap().flatten())
            .find(|e| e.file_name().to_string_lossy().starts_with(prefix))
            .unwrap()
            .path();
        let doc: serde_json::Value = serde_json::from_slice(&fs::read(manifest).unwrap()).unwrap();
        let entry = (doc["entries"].as_array().unwrap().iter())
            .find(|e| e["kind"] == "cozytensors")
            .unwrap();
        let v1 =
            tensorfs_core::header::Header::parse(&blob(entry["blob"]["sha256"].as_str().unwrap()))
                .unwrap();
        for (component, tensors) in &v1.components {
            let class = (v1.configs.iter())
                .find(|(name, _)| name == component)
                .and_then(|(_, bytes)| {
                    let config: serde_json::Value = serde_json::from_slice(bytes).ok()?;
                    let class = config
                        .get("_class_name")
                        .or_else(|| config.get("architectures")?.get(0));
                    class?.as_str().map(str::to_owned)
                })
                .or(unnamed.map(str::to_owned));
            let Some(class) = class
                .clone()
                .filter(|c| import::order::classes().any(|k| k == c))
            else {
                println!("{lane}/{component}: {class:?} has no reference");
                continue;
            };
            let built: Vec<&str> = tensors.iter().map(|(k, _)| k.as_str()).collect();
            let mut stored = built.clone();
            stored.sort_unstable();
            let keys: Vec<(&str, usize)> =
                stored.iter().enumerate().map(|(i, k)| (*k, i)).collect();
            let ordered = import::constructor_order(Some(&class), &keys);
            let lacking: Vec<&str> = ordered.unknown.iter().map(|i| keys[*i].0).collect();
            let got: Vec<&str> = (ordered.order.iter())
                .map(|i| keys[*i].0)
                .filter(|k| !lacking.contains(k))
                .collect();
            let want: Vec<&str> = built
                .iter()
                .copied()
                .filter(|k| !lacking.contains(k))
                .collect();
            println!(
                "{lane}/{component} ({class}): {} of {} keys held, order {}; lacking e.g. {:?}",
                want.len(),
                built.len(),
                if got == want { "matches" } else { "DIFFERS" },
                &lacking[..lacking.len().min(2)]
            );
            assert_eq!(got, want, "{lane}/{component}");
            held += want.len();
        }
    }
    assert!(held > 5000, "{held}");
}

/// Every file under `root`, by its path relative to `root`.
fn tree(root: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        for entry in fs::read_dir(dir).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(root, &path, out);
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
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

/// Every tensor of a safetensors file: key → (dtype, shape, bytes).
fn safetensors_tensors(bytes: &[u8]) -> BTreeMap<String, (String, Vec<u64>, Vec<u8>)> {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let head: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    (head.into_iter().filter(|(k, _)| k != "__metadata__"))
        .map(|(k, t)| {
            let at: Vec<usize> = serde_json::from_value(t["data_offsets"].clone()).unwrap();
            let shape = serde_json::from_value(t["shape"].clone()).unwrap();
            let data = bytes[8 + n + at[0]..8 + n + at[1]].to_vec();
            (k, (t["dtype"].as_str().unwrap().to_owned(), shape, data))
        })
        .collect()
}

/// Every tensor of a folder's safetensors files: (folder, key) → (dtype, shape, bytes).
fn folder_tensors(files: &BTreeMap<String, Vec<u8>>) -> BTreeMap<(String, String), (String, Vec<u64>, Vec<u8>)> {
    let mut out = BTreeMap::new();
    for (path, bytes) in files.iter().filter(|(p, _)| p.ends_with(".safetensors")) {
        let folder = path.rsplit_once('/').map_or("", |(f, _)| f).to_owned();
        for (key, t) in safetensors_tensors(bytes) {
            assert!(out.insert((folder.clone(), key), t).is_none(), "{path}");
        }
    }
    out
}

/// network: Hugging Face. Convert keeps everything an export can carry, in the D10 layout: four
/// real repositories at pinned revisions (a Diffusers pipeline, a Transformers model with its
/// tokenizer, a sharded model, a sharded fp16 variant) convert into a plain folder whose
/// cozytensors holds the pipeline, configs and run files, and whose README and .gitattributes
/// are repo files beside it. Exported, every repository file comes back: weights and assets and
/// repo files byte-exact, configs equal as parsed JSON. Converting the export, and again its
/// export, is a fixed point.
#[test]
fn a_convert_survives_export() {
    let access = Access::default();
    // (repo, weights variant, files a current safetensors wrote: reproduced byte for byte; an
    // older writer padded its header otherwise).
    let cases = [
        ("hf-internal-testing/tiny-sdxl-pipe@20594cbc343cfcfe447af5c87cdaf6c436b453f2", None, true),
        ("hf-internal-testing/tiny-random-LlamaForCausalLM@9fb191250dd56d0ba7ec9785a025ed29c03d5998", None, true),
        ("hf-internal-testing/tiny-random-bert-sharded-safetensors@3157eb8d4addc30b6e95a8000f2e3f0f6114bf5a", None, false),
        (
            "hf-internal-testing/tiny-stable-diffusion-pipe-variants-right-format@5017a45b0389efbe1710fe4cf6df15ec8237e49b",
            Some("fp16"),
            false,
        ),
    ];
    for (repo, variant, current) in cases {
        let dir = scratch(&repo[..repo.find('@').unwrap()].replace('/', "-"));
        let tmp = dir.join("tmp");
        let convert = |source: &str, variant: Option<&str>, into: &Path| {
            let sources = [source.to_owned()];
            let req = Request {
                sources: &sources,
                converter: None,
                variant,
                access: &access,
                tmp: &tmp,
                progress: &|_, _, _| {},
                stop: None,
            };
            let _ = fs::remove_dir_all(into);
            let header = import::convert_folder(&req, into).unwrap();
            let ck = Checkpoint::folder(into, None, None).unwrap();
            assert_eq!(ck.id(), &header);
            ck
        };
        let export = |ck: &Checkpoint, folder: &Path, out: &Path| {
            let repo: BTreeMap<String, PathBuf> = (tree(folder).into_keys())
                .filter(|p| !p.starts_with("chunks/") && !p.ends_with(".cozytensors"))
                .map(|p| (p.clone(), folder.join(p)))
                .collect();
            let _ = fs::remove_dir_all(out);
            tensorfs_core::export::safetensors(ck, &repo, out, None).unwrap();
            tree(out)
        };
        let source = format!("hf://{repo}");
        let first = convert(&source, variant, &dir.join("first"));
        let header = first.header();
        assert!(header.metadata["source"] == Value::Str(source.clone()), "{repo}");
        // Repo files are the folder's, never the cozytensors'.
        let folder = tree(&dir.join("first"));
        assert!(folder.contains_key(".gitattributes"), "{repo}");
        assert!(header.assets().all(|(_, name, _)| !name.contains("README") && name != ".gitattributes"));
        let exported = export(&first, &dir.join("first"), &dir.join("out"));

        // Upstream as selected: every non-weight file, and the weights convert takes.
        let listing = import::remote::list(&import::remote::parse(&source).unwrap(), &access).unwrap();
        let selected = import::remote::select(listing, variant, import::remote::Files::All, &|m| {
            import::remote::read_small(m, &access)
        })
        .unwrap();
        let mut upstream = BTreeMap::new();
        for m in &selected {
            let local = import::remote::land(m, &dir.join("upstream").join(&m.path), &access).unwrap();
            upstream.insert(m.path.clone(), fs::read(local).unwrap());
        }
        let configs: Vec<String> = (header.pipeline().map(|_| "model_index.json".to_owned()))
            .into_iter()
            .chain(header.components().iter().filter(|c| !c.config.is_empty()).map(|c| {
                match (header.pipeline(), c.name.as_str()) {
                    (None, "model") => c.config_name().to_owned(),
                    _ => format!("{}/{}", c.name, c.config_name()),
                }
            }))
            .collect();
        let documents = |files: &BTreeMap<String, Vec<u8>>| -> BTreeMap<String, Vec<u8>> {
            (files.iter().filter(|(p, _)| !p.contains(".safetensors")))
                .map(|(p, b)| (p.clone(), b.clone()))
                .collect()
        };
        let (theirs, ours) = (documents(&upstream), documents(&exported));
        assert_eq!(ours.keys().collect::<Vec<_>>(), theirs.keys().collect::<Vec<_>>(), "{repo}");
        for (path, bytes) in &theirs {
            match configs.contains(path) {
                true => {
                    let parsed = |b: &[u8]| serde_json::from_slice::<serde_json::Value>(b).unwrap();
                    assert_eq!(parsed(&ours[path]), parsed(bytes), "{repo}: {path}");
                }
                false => assert!(ours[path] == *bytes, "{repo}: {path} differs from upstream's"),
            }
        }
        // Weights: every upstream tensor, and a file a current safetensors wrote, byte for byte.
        assert!(folder_tensors(&upstream) == folder_tensors(&exported), "{repo}: the tensors");
        let weights = |files: &BTreeMap<String, Vec<u8>>, folder: &str| -> Vec<Vec<u8>> {
            (files.iter())
                .filter(|(p, _)| p.ends_with(".safetensors") && p.rsplit_once('/').map_or("", |(f, _)| f) == folder)
                .map(|(_, b)| b.clone())
                .collect()
        };
        for folder in header.components().iter().map(|c| match (header.pipeline(), c.name.as_str()) {
            (None, "model") => "",
            (_, name) => name,
        }) {
            if let ([theirs], [ours], true) = (&weights(&upstream, folder)[..], &weights(&exported, folder)[..], current) {
                assert!(theirs == ours, "{repo}: {folder} weights are not byte for byte");
            }
        }
        // An index names only the shards written.
        for (path, bytes) in exported.iter().filter(|(p, _)| p.contains(".safetensors.index")) {
            let index: serde_json::Value = serde_json::from_slice(bytes).unwrap();
            let folder = path.rsplit_once('/').map_or(String::new(), |(f, _)| format!("{f}/"));
            for shard in index["weight_map"].as_object().unwrap().values() {
                assert!(exported.contains_key(&format!("{folder}{}", shard.as_str().unwrap())), "{repo}: {path}");
            }
        }

        // The export converts to the same model (its tensors in any order: a class without an
        // order reference keeps its source's data order); its export, again, to the same
        // cozytensors.
        let out = dir.join("out");
        let second = convert(out.to_str().unwrap(), None, &dir.join("second"));
        let model = |ck: &Checkpoint| {
            let tensors: BTreeMap<_, _> = (ck.tensors())
                .map(|(c, t)| ((c.to_owned(), t.key.clone()), (t.dtype.clone(), t.shape.clone(), ck.read(c, &t.key).unwrap())))
                .collect();
            let h = ck.header();
            let assets: Vec<_> = h.assets().map(|(c, n, r)| (c.to_owned(), n.to_owned(), r.clone())).collect();
            let specs: Vec<_> = (h.components().iter())
                .map(|c| (c.name.clone(), c.class.clone(), c.config.clone(), c.config_file.clone()))
                .collect();
            (tensors, assets, specs, h.pipeline().cloned(), h.metadata.get(import::FILE_METADATA).cloned())
        };
        let (a, b) = (model(&first), model(&second));
        assert!(a.0 == b.0, "{repo}: the tensors");
        assert_eq!((&a.1, &a.2, &a.3, &a.4), (&b.1, &b.2, &b.3, &b.4), "{repo}: the model");
        export(&second, &dir.join("second"), &out);
        let third = convert(out.to_str().unwrap(), None, &dir.join("third"));
        assert_eq!(third.id(), second.id(), "{repo}: convert of an export is a fixed point");
        fs::remove_dir_all(&dir).unwrap();
    }
}

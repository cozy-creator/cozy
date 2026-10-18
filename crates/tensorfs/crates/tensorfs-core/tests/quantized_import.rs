//! Quantized checkpoints as ModelOpt, compressed-tensors and ComfyUI write them
//! (`fixtures/quantized/make.py`) import through the real path into a store. Each quantized weight
//! becomes one encoded tensor whose parts are the producer's bytes; the reference decoder turns it
//! into the producer's own dequantization, within a stated error of the BF16 weight it came from;
//! names and order are the plain path's. No mocks.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use tensorfs_core::err::Code;
use tensorfs_core::format::{decode, Body, Dtype, Encoding, PartBytes};
use tensorfs_core::import::{Access, Request};
use tensorfs_core::store2::{Checkpoint, Store};

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Scratch {
        let path = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("quantized-{name}-{}", std::process::id()));
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

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/quantized")
}

/// A safetensors tensor: key, dtype, shape, bytes.
type Entry = (String, String, Vec<u64>, Vec<u8>);
/// A safetensors file's tensors in file order.
type Tensors = Vec<Entry>;

fn read_safetensors(path: &Path) -> Tensors {
    let raw = fs::read(path).unwrap();
    let n = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let head: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&raw[8..8 + n]).unwrap();
    let mut out: Vec<(u64, Entry)> = Vec::new();
    for (key, t) in head.into_iter().filter(|(k, _)| k != "__metadata__") {
        let at: Vec<u64> = serde_json::from_value(t["data_offsets"].clone()).unwrap();
        let bytes = raw[8 + n + at[0] as usize..8 + n + at[1] as usize].to_vec();
        let shape = serde_json::from_value(t["shape"].clone()).unwrap();
        let dtype = t["dtype"].as_str().unwrap().to_owned();
        out.push((at[0], (key, dtype, shape, bytes)));
    }
    out.sort_by_key(|(at, (key, ..))| (*at, key.clone()));
    out.into_iter().map(|(_, t)| t).collect()
}

fn write_safetensors(path: &Path, tensors: &Tensors) {
    let mut head = serde_json::Map::new();
    let mut data = Vec::new();
    for (key, dtype, shape, bytes) in tensors {
        let at = [data.len(), data.len() + bytes.len()];
        data.extend_from_slice(bytes);
        head.insert(
            key.clone(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": at}),
        );
    }
    let head = serde_json::to_vec(&head).unwrap();
    let mut out = (head.len() as u64).to_le_bytes().to_vec();
    out.extend(head);
    out.extend(data);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, out).unwrap();
}

fn import(store: &Store, source: &Path) -> tensorfs_core::err::Result<Checkpoint> {
    let access = Access::default();
    let tmp = store.root().join("tmp");
    let req = Request {
        sources: &[source.display().to_string()],
        converter: None,
        variant: None,
        access: &access,
        tmp: &tmp,
        progress: &|_, _, _| {},
    };
    let manifest = store.import(&req, None)?;
    Ok(store.checkpoint(&manifest).unwrap())
}

fn bf16s(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect()
}

fn value(bits: u16) -> f64 {
    f32::from_bits(u32::from(bits) << 16) as f64
}

/// How the reference decode meets the producer's own dequantization.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Meets {
    /// Bit for bit: the spec's arithmetic is the producer's.
    Exactly,
    /// Within one BF16 step (or 2^-24 of the tensor's largest value): the producer rounds in
    /// another order (`counts` below names each).
    OneStep,
}

/// One quantized weight: its encoding at a glance, its parts, and the error bound against the BF16
/// source as `‖decode − source‖ / ‖source‖`.
struct Want {
    key: &'static str,
    encoding: &'static str,
    parts: &'static [(&'static str, &'static str, &'static [u64], &'static str)],
    meets: Meets,
    error: f64,
}

const NVFP4_MODELOPT: &str = r#"nvfp4 {block_size: 16, global_scale: "multiply", nibble_order: "low_first", scale_layout: "row_major"}"#;
const NVFP4_CT: &str = r#"nvfp4 {block_size: 16, global_scale: "divide", nibble_order: "low_first", scale_layout: "row_major"}"#;
const NVFP4_COMFY: &str = r#"nvfp4 {block_size: 16, global_scale: "multiply", nibble_order: "high_first", scale_layout: "cublas_128x4"}"#;
const MXFP8_ROWS: &str = r#"mxfp8 {block_size: 32, scale_layout: "row_major"}"#;
const MXFP8_TILES: &str = r#"mxfp8 {block_size: 32, scale_layout: "cublas_128x4"}"#;

/// Imports `case`, then checks every quantized weight in `wants` and that every other tensor is
/// the source's, in the plain path's names and order.
fn check(case: &str, wants: &[Want]) -> BTreeMap<String, usize> {
    let dir = Scratch::new(case);
    let store = Store::open(dir.0.join("store")).unwrap();
    let source = fixtures().join(case);
    let ck = import(&store, &source).unwrap();
    let expect: BTreeMap<String, Vec<u8>> =
        read_safetensors(&fixtures().join(format!("{case}.expect.safetensors")))
            .into_iter()
            .map(|(k, _, _, b)| (k, b))
            .collect();
    let producer: BTreeMap<String, (String, Vec<u64>, Vec<u8>)> =
        read_safetensors(&source.join("model.safetensors"))
            .into_iter()
            .map(|(k, d, s, b)| (k, (d, s, b)))
            .collect();
    let raw = |at: &tensorfs_core::format::Location| {
        let mut buf = vec![0; at.length as usize];
        ck.bytes_into(at, &mut buf).unwrap();
        buf
    };
    let mut steps = BTreeMap::new();
    for want in wants {
        let t = ck.tensor("model", want.key).unwrap();
        let Body::Encoded { encoding, parts } = &t.body else {
            panic!("{}: plain", want.key)
        };
        assert_eq!(encoding.to_string(), want.encoding, "{}", want.key);
        assert_eq!(t.dtype, Dtype::BF16, "{}", want.key);
        let module = want.key.strip_suffix(".weight").unwrap();
        let got: Vec<(&str, &str, &[u64])> = parts
            .iter()
            .map(|p| (p.role.as_str(), p.dtype.name(), p.shape.as_slice()))
            .collect();
        let wanted: Vec<(&str, &str, &[u64])> =
            want.parts.iter().map(|(r, d, s, _)| (*r, *d, *s)).collect();
        assert_eq!(got, wanted, "{}", want.key);
        // Each part is the producer's tensor, byte for byte.
        for (p, (_, _, _, suffix)) in parts.iter().zip(want.parts) {
            let (_, _, bytes) = &producer[&format!("{module}.{suffix}")];
            assert_eq!(&raw(&p.at()), bytes, "{} part {}", want.key, p.role);
        }
        let decoded = bf16s(&ck.read("model", want.key).unwrap());
        let dequant = bf16s(&expect[&format!("dequant.{}", want.key)]);
        let source = bf16s(&expect[&format!("source.{}", want.key)]);
        assert_eq!(t.shape.iter().product::<u64>() as usize, source.len());
        let largest = dequant.iter().map(|b| value(*b).abs()).fold(0.0, f64::max);
        let mut differ = 0;
        for (i, (a, b)) in decoded.iter().zip(&dequant).enumerate() {
            if a == b {
                continue;
            }
            differ += 1;
            let near = (a >> 15 == b >> 15 && a.abs_diff(*b) <= 1)
                || (value(*a) - value(*b)).abs() <= largest * 2f64.powi(-24);
            assert!(
                want.meets == Meets::OneStep && near,
                "{} element {i}: decoded {:#06x}, the producer {:#06x}",
                want.key,
                a,
                b
            );
        }
        steps.insert(want.key.to_owned(), differ);
        let (mut err, mut norm) = (0.0, 0.0);
        for (d, s) in decoded.iter().zip(&source) {
            err += (value(*d) - value(*s)).powi(2);
            norm += value(*s).powi(2);
        }
        let error = (err / norm).sqrt();
        assert!(
            error <= want.error,
            "{}: error {error:.4} over {}",
            want.key,
            want.error
        );
    }

    // Everything else is the source's plain tensor; markers and carrier parts are gone; names
    // and order are those of the same model imported plain.
    let encoded: Vec<&str> = wants.iter().map(|w| w.key).collect();
    for (key, t) in ck.tensors().map(|(c, t)| (format!("{c}/{}", t.key), t)) {
        let key = key.strip_prefix("model/").unwrap();
        if encoded.contains(&key) {
            continue;
        }
        let Body::Data(at) = &t.body else {
            panic!("{key}: encoded")
        };
        assert_eq!(raw(at), producer[key].2, "{key}");
    }
    let twin = dir.0.join("twin");
    let mut plain = Tensors::new();
    // The same model in BF16: each quantized weight where its data part sits, the rest as they are.
    for (key, dtype, shape, bytes) in read_safetensors(&source.join("model.safetensors")) {
        let module = key.rsplit_once('.').map_or("", |(m, _)| m);
        let logical = format!("{module}.weight");
        let data = [logical.clone(), format!("{module}.weight_packed")];
        if encoded.contains(&logical.as_str()) && data.contains(&key) {
            let bytes = expect[&format!("source.{logical}")].clone();
            let shape = ck.tensor("model", &logical).unwrap().shape.clone();
            plain.push((logical, "BF16".into(), shape, bytes));
        } else if !encoded.contains(&key.as_str()) && ck.tensor("model", &key).is_ok() {
            plain.push((key, dtype, shape, bytes));
        }
    }
    write_safetensors(&twin.join("model.safetensors"), &plain);
    let twin = import(&store, &twin).unwrap();
    let order = |ck: &Checkpoint| -> Vec<(String, String)> {
        ck.tensors()
            .map(|(c, t)| (c.to_owned(), t.key.clone()))
            .collect()
    };
    assert_eq!(order(&ck), order(&twin), "{case}: names and order");
    steps
}

#[test]
fn modelopt_nvfp4_and_mxfp8() {
    check(
        "modelopt-nvfp4",
        &[
            Want {
                key: "proj.weight",
                encoding: NVFP4_MODELOPT,
                parts: &[
                    ("weight", "F4", &[32, 64], "weight"),
                    ("scale", "F8_E4M3", &[32, 4], "weight_scale"),
                    ("scale_2", "F32", &[], "weight_scale_2"),
                    ("input_scale", "F32", &[], "input_scale"),
                ],
                meets: Meets::Exactly,
                error: 0.15,
            },
            // An expert stack: block scales act on the last axis, every leading axis is rows.
            Want {
                key: "experts.weight",
                encoding: NVFP4_MODELOPT,
                parts: &[
                    ("weight", "F4", &[2, 16, 64], "weight"),
                    ("scale", "F8_E4M3", &[2, 16, 4], "weight_scale"),
                    ("scale_2", "F32", &[], "weight_scale_2"),
                ],
                meets: Meets::Exactly,
                error: 0.15,
            },
        ],
    );
    check(
        "modelopt-mxfp8",
        &[Want {
            key: "proj.weight",
            encoding: MXFP8_ROWS,
            parts: &[
                ("weight", "F8_E4M3", &[32, 64], "weight"),
                ("scale", "F8_E8M0", &[32, 2], "weight_scale"),
            ],
            meets: Meets::Exactly,
            error: 0.04,
        }],
    );
}

#[test]
fn compressed_tensors_nvfp4_and_mxfp8() {
    check(
        "ct-nvfp4",
        &[Want {
            key: "proj.weight",
            encoding: NVFP4_CT,
            parts: &[
                ("weight", "F4", &[32, 64], "weight_packed"),
                ("scale", "F8_E4M3", &[32, 4], "weight_scale"),
                ("scale_2", "F32", &[1], "weight_global_scale"),
                ("input_scale", "F32", &[1], "input_global_scale"),
            ],
            meets: Meets::Exactly,
            error: 0.15,
        }],
    );
    check(
        "ct-mxfp8",
        &[Want {
            key: "proj.weight",
            encoding: MXFP8_ROWS,
            parts: &[
                ("weight", "F8_E4M3", &[32, 64], "weight"),
                ("scale", "F8_E8M0", &[32, 2], "weight_scale"),
            ],
            meets: Meets::Exactly,
            error: 0.04,
        }],
    );
}

#[test]
fn comfy_nvfp4_keeps_its_nibbles_and_tiles() {
    // comfy-kitchen's eager dequantize multiplies in BF16 (`scale_2.to(bf16) * scale.to(bf16)`,
    // then `value * that`); the spec evaluates in f32 and rounds once.
    let steps = check(
        "comfy-nvfp4",
        &[
            Want {
                key: "blocks.0.attn.qkv.weight",
                encoding: NVFP4_COMFY,
                parts: &[
                    ("weight", "F4", &[144, 80], "weight"),
                    ("scale", "F8_E4M3", &[256, 8], "weight_scale"),
                    ("scale_2", "F32", &[], "weight_scale_2"),
                    ("input_scale", "F32", &[], "input_scale"),
                ],
                meets: Meets::OneStep,
                error: 0.15,
            },
            // Experts tiled one by one are one tiling of all their rows (128 per expert).
            Want {
                key: "blocks.0.experts.weight",
                encoding: NVFP4_COMFY,
                parts: &[
                    ("weight", "F4", &[2, 128, 16], "weight"),
                    ("scale", "F8_E4M3", &[256, 4], "weight_scale"),
                    ("scale_2", "F32", &[], "weight_scale_2"),
                ],
                meets: Meets::OneStep,
                error: 0.15,
            },
        ],
    );
    println!("comfy nvfp4 elements one BF16 step from kitchen's: {steps:?}");

    // The same bytes read in ModelOpt's layout decode wrong without an error: the bug a
    // name-and-shape mapping had.
    let dir = Scratch::new("comfy-nvfp4-as-modelopt");
    let store = Store::open(dir.0.join("store")).unwrap();
    let ck = import(&store, &fixtures().join("comfy-nvfp4")).unwrap();
    let key = "blocks.0.attn.qkv.weight";
    let t = ck.tensor("model", key).unwrap();
    let Body::Encoded { encoding, parts } = &t.body else {
        panic!()
    };
    let bytes: Vec<Vec<u8>> = parts
        .iter()
        .map(|p| {
            let mut b = vec![0; p.length as usize];
            ck.bytes_into(&p.at(), &mut b).unwrap();
            b
        })
        .collect();
    let views: BTreeMap<&str, PartBytes> = parts
        .iter()
        .zip(&bytes)
        .map(|(p, b)| {
            (
                p.role.as_str(),
                PartBytes {
                    dtype: &p.dtype,
                    shape: &p.shape,
                    bytes: b,
                },
            )
        })
        .collect();
    let right = bf16s(&decode(&t.dtype, &t.shape, encoding, &views).unwrap());
    let mut low_first: Encoding = encoding.clone();
    low_first
        .params
        .insert("nibble_order".into(), "low_first".into());
    let wrong = bf16s(&decode(&t.dtype, &t.shape, &low_first, &views).unwrap());
    let differ = right.iter().zip(&wrong).filter(|(a, b)| a != b).count();
    assert!(
        differ * 2 > right.len(),
        "{differ} of {} differ",
        right.len()
    );
}

#[test]
fn comfy_mxfp8_declared_by_metadata() {
    check(
        "comfy-mxfp8",
        &[Want {
            key: "proj.weight",
            encoding: MXFP8_TILES,
            parts: &[
                ("weight", "F8_E4M3", &[32, 160], "weight"),
                ("scale", "F8_E8M0", &[128, 8], "weight_scale"),
            ],
            meets: Meets::Exactly,
            error: 0.04,
        }],
    );
}

#[test]
fn comfy_fp8_tensorwise_rowwise_and_legacy() {
    // kitchen's `dequantize_per_tensor_fp8` rounds a per-row scale to BF16 first, as the spec
    // does only for a tensorwise one.
    let steps = check(
        "comfy-fp8",
        &[
            Want {
                key: "attn.to_q.weight",
                encoding: "fp8.tensorwise {}",
                parts: &[
                    ("weight", "F8_E4M3", &[32, 64], "weight"),
                    ("scale", "F32", &[], "weight_scale"),
                    ("input_scale", "F32", &[], "input_scale"),
                ],
                meets: Meets::Exactly,
                error: 0.04,
            },
            Want {
                key: "attn.to_k.weight",
                encoding: "fp8.rowwise {axis: 0}",
                parts: &[
                    ("weight", "F8_E4M3", &[32, 64], "weight"),
                    ("scale", "F32", &[32, 1], "weight_scale"),
                ],
                meets: Meets::OneStep,
                error: 0.04,
            },
        ],
    );
    println!("comfy fp8 elements one BF16 step from kitchen's: {steps:?}");
    check(
        "comfy-scaled-fp8",
        &[Want {
            key: "attn.to_v.weight",
            encoding: "fp8.tensorwise {}",
            parts: &[
                ("weight", "F8_E4M3", &[32, 64], "weight"),
                ("scale", "F32", &[1], "scale_weight"),
                ("input_scale", "F32", &[1], "scale_input"),
            ],
            meets: Meets::Exactly,
            error: 0.04,
        }],
    );
}

#[test]
fn comfy_int8_tensorwise_rowwise_and_convrot() {
    // kitchen's int8 dequantize is `q.float() * scale` (no BF16 scale, unlike the spec's
    // int8.tensorwise), and ConvRot rotates the rounded `q * scale` back where the spec rotates the
    // exact integers.
    let steps = check(
        "comfy-int8",
        &[
            Want {
                key: "ff.up.weight",
                encoding: "int8.tensorwise {}",
                parts: &[
                    ("weight", "I8", &[32, 64], "weight"),
                    ("scale", "F32", &[], "weight_scale"),
                ],
                meets: Meets::OneStep,
                error: 0.05,
            },
            Want {
                key: "ff.gate.weight",
                encoding: "int8.rowwise {axis: 0}",
                parts: &[
                    ("weight", "I8", &[32, 64], "weight"),
                    ("scale", "F32", &[32, 1], "weight_scale"),
                ],
                meets: Meets::Exactly,
                error: 0.02,
            },
            Want {
                key: "ff.down.weight",
                encoding: r#"int8.convrot {rotation: "hadamard", rotation_size: 64}"#,
                parts: &[
                    ("weight", "I8", &[16, 128], "weight"),
                    ("scale", "F32", &[16, 1], "weight_scale"),
                ],
                meets: Meets::OneStep,
                error: 0.02,
            },
        ],
    );
    println!("comfy int8 elements one BF16 step from kitchen's: {steps:?}");
}

/// `case`'s source copied to a scratch folder, `edit`ed (key, dtype, shape, bytes) and imported.
fn refused(case: &str, edit: &dyn Fn(&mut Tensors, &Path)) -> tensorfs_core::err::Refusal {
    let dir = Scratch::new(&format!("refused-{case}"));
    let src = dir.0.join("src");
    fs::create_dir_all(&src).unwrap();
    for entry in fs::read_dir(fixtures().join(case)).unwrap().flatten() {
        fs::copy(entry.path(), src.join(entry.file_name())).unwrap();
    }
    let mut tensors = read_safetensors(&src.join("model.safetensors"));
    edit(&mut tensors, &src);
    write_safetensors(&src.join("model.safetensors"), &tensors);
    let store = Store::open(dir.0.join("store")).unwrap();
    import(&store, &src).map(drop).unwrap_err()
}

fn marker(module: &str, conf: serde_json::Value) -> Entry {
    let bytes = serde_json::to_vec(&conf).unwrap();
    (
        format!("{module}.comfy_quant"),
        "U8".into(),
        vec![bytes.len() as u64],
        bytes,
    )
}

#[test]
fn unknown_undeclared_and_ambiguous_carriers_are_refused() {
    // Comfy's carrier without its markers: nothing declares what the scales mean.
    let e = refused("comfy-fp8", &|t, _| {
        t.retain(|(k, ..)| !k.ends_with(".comfy_quant"))
    });
    assert_eq!(e.code, Code::UNREGISTERED_ENCODING, "{}", e.detail);
    assert!(e.detail.contains("attn.to_k.weight_scale"), "{}", e.detail);

    // Two producers declare one file.
    let e = refused("comfy-fp8", &|_, dir| {
        let doc = serde_json::json!({"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "FP8"}});
        fs::write(dir.join("hf_quant_config.json"), doc.to_string()).unwrap();
    });
    assert_eq!(e.code, Code::AMBIGUOUS_CLASSIFICATION, "{}", e.detail);
    assert!(
        e.detail.contains("Modelopt") && e.detail.contains("Comfy"),
        "{}",
        e.detail
    );

    // A format this import does not read is named, never stored as plain bytes.
    let e = refused("comfy-int8", &|t, _| {
        t.retain(|(k, ..)| k != "ff.up.comfy_quant");
        t.push(marker(
            "ff.up",
            serde_json::json!({"format": "convrot_w4a4"}),
        ));
    });
    assert_eq!(e.code, Code::UNREGISTERED_ENCODING, "{}", e.detail);
    assert!(e.detail.contains("convrot_w4a4"), "{}", e.detail);

    // ModelOpt's AWQ smoothing has no part in cozytensors/1's nvfp4 yet.
    let e = refused("modelopt-nvfp4", &|t, _| {
        t.push((
            "proj.pre_quant_scale".into(),
            "BF16".into(),
            vec![64],
            vec![0; 128],
        ));
    });
    assert_eq!(e.code, Code::UNREGISTERED_ENCODING, "{}", e.detail);
    assert!(e.detail.contains("pre_quant_scale"), "{}", e.detail);

    // A per-expert global scale is not one nvfp4 scale_2.
    let e = refused("comfy-nvfp4", &|t, _| {
        let g = t
            .iter_mut()
            .find(|(k, ..)| k == "blocks.0.experts.weight_scale_2")
            .unwrap();
        g.2 = vec![2];
        g.3 = [g.3.clone(), g.3.clone()].concat();
    });
    assert_eq!(e.code, Code::MISSING_COMPANION_ROLE, "{}", e.detail);
    assert!(e.detail.contains("weight_scale_2"), "{}", e.detail);
}

#[test]
fn every_carrier_names_a_complete_encoding() {
    use tensorfs_core::format::{GlobalScale::*, NibbleOrder::*, ScaleLayout::*};
    use tensorfs_core::import::carriers::{carriers, Producer};
    for c in carriers() {
        let want = match (c.encoding.scheme.as_str(), c.producer) {
            ("nvfp4", Producer::Modelopt) => Encoding::nvfp4(Multiply, LowFirst, RowMajor),
            ("nvfp4", Producer::CompressedTensors) => Encoding::nvfp4(Divide, LowFirst, RowMajor),
            ("nvfp4", Producer::Comfy) => Encoding::nvfp4(Multiply, HighFirst, Cublas128x4),
            ("mxfp8", Producer::Comfy) => Encoding::mxfp8(Cublas128x4),
            ("mxfp8", _) => Encoding::mxfp8(RowMajor),
            ("fp8.tensorwise", _) => Encoding::fp8_tensorwise(),
            ("fp8.rowwise", _) => Encoding::fp8_rowwise(0),
            ("int8.tensorwise", _) => Encoding::int8_tensorwise(),
            ("int8.rowwise", _) => Encoding::int8_rowwise(0),
            ("int8.convrot", _) => Encoding::int8_convrot(256),
            other => panic!("{other:?}"),
        };
        let mut encoding = c.encoding.clone();
        for (param, (_, default)) in &c.marker_params {
            encoding.params.insert(param.clone(), (*default).into());
        }
        assert_eq!(encoding, want, "{:?} {:?}", c.producer, c.formats);
    }
}

#[test]
fn tfs_inspect_shows_comfy_nvfp4_at_a_glance() {
    let dir = Scratch::new("inspect");
    let store = dir.0.join("store");
    let tfs = |args: &[&str]| {
        let out = Command::new(env!("CARGO_BIN_EXE_tfs"))
            .arg("--store")
            .arg(&store)
            .args(args)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "tfs {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap()
    };
    let source = fixtures().join("comfy-nvfp4");
    tfs(&[
        "import",
        source.to_str().unwrap(),
        "--name",
        "acme/comfy-nvfp4",
    ]);
    let doc: serde_json::Value =
        serde_json::from_str(tfs(&["--json", "inspect", "acme/comfy-nvfp4"]).trim()).unwrap();
    let tensors = doc["components"][0]["tensors"].as_array().unwrap();
    let qkv = tensors
        .iter()
        .find(|t| t["key"] == "blocks.0.attn.qkv.weight")
        .unwrap();
    assert_eq!(
        qkv,
        &serde_json::json!({"key": "blocks.0.attn.qkv.weight", "dtype": "BF16", "shape": [144, 80], "encoding": NVFP4_COMFY})
    );
    let human = tfs(&["inspect", "acme/comfy-nvfp4"]);
    assert!(
        human.contains(&format!("    2 × {NVFP4_COMFY}\n")),
        "{human}"
    );
}

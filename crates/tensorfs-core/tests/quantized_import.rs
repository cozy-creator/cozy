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

/// Safetensors `__metadata__`.
type Metadata = serde_json::Map<String, serde_json::Value>;

fn read_metadata(path: &Path) -> Metadata {
    let raw = fs::read(path).unwrap();
    let n = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let head: Metadata = serde_json::from_slice(&raw[8..8 + n]).unwrap();
    head.get("__metadata__")
        .and_then(|m| m.as_object().cloned())
        .unwrap_or_default()
}

fn write_safetensors(path: &Path, tensors: &Tensors, metadata: &Metadata) {
    let mut head = serde_json::Map::new();
    if !metadata.is_empty() {
        head.insert("__metadata__".into(), metadata.clone().into());
    }
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
        stop: None,
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
const NVFP4_COMFY_AWQ: &str = r#"nvfp4 {block_size: 16, global_scale: "multiply", nibble_order: "high_first", pre_quant_scale_dtype: "BF16", scale_layout: "cublas_128x4"}"#;
const NVFP4_MODELOPT_AWQ: &str = r#"nvfp4 {block_size: 16, global_scale: "multiply", nibble_order: "low_first", pre_quant_scale_dtype: "BF16", scale_layout: "row_major"}"#;
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
    write_safetensors(&twin.join("model.safetensors"), &plain, &Metadata::new());
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
fn modelopt_nvfp4_awq_keeps_pre_quant_scale() {
    // ModelOpt 0.47.0's own NVFP4_AWQ export (`conformance/mine/nvfp4.py`). Only o_proj keeps its
    // pre_quant_scale; ModelOpt folds q/k/v's into the norm and down's into up_proj, so those decode
    // to the smoothed weight it quantized. Its table has no −0: signed zeros are its one difference.
    macro_rules! nvfp4 {
        ($m:literal, $n:literal, $k:literal) => {
            Want {
                key: concat!("model.layers.0.", $m, ".weight"),
                encoding: NVFP4_MODELOPT,
                parts: &[
                    ("weight", "F4", &[$n, $k], "weight"),
                    ("scale", "F8_E4M3", &[$n, $k / 16], "weight_scale"),
                    ("scale_2", "F32", &[], "weight_scale_2"),
                    ("input_scale", "F32", &[], "input_scale"),
                ],
                meets: Meets::OneStep,
                error: 0.12,
            }
        };
    }
    let mut wants = [
        nvfp4!("self_attn.q_proj", 64, 64),
        nvfp4!("self_attn.k_proj", 32, 64),
        nvfp4!("self_attn.v_proj", 32, 64),
        nvfp4!("self_attn.o_proj", 64, 64),
        nvfp4!("mlp.gate_proj", 96, 64),
        nvfp4!("mlp.up_proj", 96, 64),
        nvfp4!("mlp.down_proj", 64, 96),
    ];
    wants[3].encoding = NVFP4_MODELOPT_AWQ;
    wants[3].parts = &[
        ("weight", "F4", &[64, 64], "weight"),
        ("scale", "F8_E4M3", &[64, 4], "weight_scale"),
        ("scale_2", "F32", &[], "weight_scale_2"),
        ("input_scale", "F32", &[], "input_scale"),
        ("pre_quant_scale", "BF16", &[64], "pre_quant_scale"),
    ];
    check("modelopt-nvfp4-awq", &wants);
}

#[test]
fn comfy_fused_experts_keep_a_scale_2_each() {
    // Comfy-Org/Lens's gpt-oss experts (a slice): weight_scale_2 [E] and tiles per expert decode
    // bit for bit to OpenAI's MXFP4 original, and to comfy-kitchen's own dequantize.
    check(
        "comfy-nvfp4-experts",
        &[Want {
            key: "layers.1.mlp.experts.down_proj.weight",
            encoding: NVFP4_COMFY,
            parts: &[
                ("weight", "F4", &[3, 64, 64], "weight"),
                ("scale", "F8_E4M3", &[3, 128, 4], "weight_scale"),
                ("scale_2", "F32", &[3], "weight_scale_2"),
            ],
            meets: Meets::Exactly,
            error: 0.0,
        }],
    );
}

#[test]
fn compressed_tensors_nvfp4_and_mxfp8() {
    // Both global scales are reciprocals: `global_scale: divide` covers `scale_2` and `input_scale`.
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
            // Experts are tiled one by one, as the spec's cublas_128x4 tiles a stack.
            Want {
                key: "blocks.0.experts.weight",
                encoding: NVFP4_COMFY,
                parts: &[
                    ("weight", "F4", &[2, 128, 16], "weight"),
                    ("scale", "F8_E4M3", &[2, 128, 4], "weight_scale"),
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
    // kitchen's `dequantize_per_tensor_fp8` rounds the stored f32 scale to BF16 first
    // (`scale.to(bf16)`); the spec uses it exactly.
    let steps = check(
        "comfy-fp8",
        &[
            Want {
                key: "attn.to_q.weight",
                encoding: r#"fp8.tensorwise {scale_dtype: "F32"}"#,
                parts: &[
                    ("weight", "F8_E4M3", &[32, 64], "weight"),
                    ("scale", "F32", &[], "weight_scale"),
                    ("input_scale", "F32", &[], "input_scale"),
                ],
                meets: Meets::OneStep,
                error: 0.04,
            },
            Want {
                key: "attn.to_k.weight",
                encoding: r#"fp8.rowwise {axis: 0, scale_dtype: "F32"}"#,
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
    let steps = check(
        "comfy-scaled-fp8",
        &[Want {
            key: "attn.to_v.weight",
            encoding: r#"fp8.tensorwise {scale_dtype: "F32"}"#,
            parts: &[
                ("weight", "F8_E4M3", &[32, 64], "weight"),
                ("scale", "F32", &[1], "scale_weight"),
                ("input_scale", "F32", &[1], "scale_input"),
            ],
            meets: Meets::OneStep,
            error: 0.04,
        }],
    );
    println!("comfy scaled_fp8 elements one BF16 step from kitchen's: {steps:?}");
}

#[test]
fn comfy_int8_tensorwise_rowwise_and_convrot() {
    // kitchen's int8 dequantize is `q.float() * scale`, the spec's; ConvRot rotates the rounded
    // `q * scale` back where the spec rotates the exact integers.
    let steps = check(
        "comfy-int8",
        &[
            Want {
                key: "ff.up.weight",
                encoding: r#"int8.tensorwise {scale_dtype: "F32"}"#,
                parts: &[
                    ("weight", "I8", &[32, 64], "weight"),
                    ("scale", "F32", &[], "weight_scale"),
                ],
                meets: Meets::Exactly,
                error: 0.05,
            },
            Want {
                key: "ff.gate.weight",
                encoding: r#"int8.rowwise {axis: 0, scale_dtype: "F32"}"#,
                parts: &[
                    ("weight", "I8", &[32, 64], "weight"),
                    ("scale", "F32", &[32, 1], "weight_scale"),
                ],
                meets: Meets::Exactly,
                error: 0.02,
            },
            Want {
                key: "ff.down.weight",
                encoding: r#"int8.convrot {rotation: "hadamard", rotation_size: 64, scale_dtype: "F32"}"#,
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

/// `case`'s source copied to a scratch folder, `edit`ed (its tensors, metadata and folder) and
/// imported; the scratch folder holds the store.
fn edited(case: &str, edit: &dyn Fn(&mut Tensors, &mut Metadata, &Path)) -> (Scratch, Checkpoint) {
    let (dir, result) = try_edited(case, edit);
    (dir, result.unwrap())
}

fn try_edited(
    case: &str,
    edit: &dyn Fn(&mut Tensors, &mut Metadata, &Path),
) -> (Scratch, tensorfs_core::err::Result<Checkpoint>) {
    let dir = Scratch::new(&format!("edited-{case}"));
    let src = dir.0.join("src");
    fs::create_dir_all(&src).unwrap();
    if fixtures().join(case).is_dir() {
        for entry in fs::read_dir(fixtures().join(case)).unwrap().flatten() {
            fs::copy(entry.path(), src.join(entry.file_name())).unwrap();
        }
    }
    let file = src.join("model.safetensors");
    let (mut tensors, mut metadata) = match file.exists() {
        true => (read_safetensors(&file), read_metadata(&file)),
        false => (Tensors::new(), Metadata::new()),
    };
    edit(&mut tensors, &mut metadata, &src);
    write_safetensors(&file, &tensors, &metadata);
    let store = Store::open(dir.0.join("store")).unwrap();
    let result = import(&store, &src);
    (dir, result)
}

fn refused(
    case: &str,
    edit: &dyn Fn(&mut Tensors, &mut Metadata, &Path),
) -> tensorfs_core::err::Refusal {
    try_edited(case, edit).1.map(drop).unwrap_err()
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
    let e = refused("comfy-fp8", &|t, _, _| {
        t.retain(|(k, ..)| !k.ends_with(".comfy_quant"))
    });
    assert_eq!(e.code, Code::UNREGISTERED_ENCODING, "{}", e.detail);
    assert!(e.detail.contains("attn.to_k.weight_scale"), "{}", e.detail);

    // Two producers declare one file.
    let e = refused("comfy-fp8", &|_, _, dir| {
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
    let e = refused("comfy-int8", &|t, _, _| {
        t.retain(|(k, ..)| k != "ff.up.comfy_quant");
        t.push(marker(
            "ff.up",
            serde_json::json!({"format": "convrot_w4a4"}),
        ));
    });
    assert_eq!(e.code, Code::UNREGISTERED_ENCODING, "{}", e.detail);
    assert!(e.detail.contains("convrot_w4a4"), "{}", e.detail);

    // AWQ's pre_quant_scale has one element per input column.
    let e = refused("modelopt-nvfp4", &|t, _, _| {
        t.push((
            "proj.pre_quant_scale".into(),
            "BF16".into(),
            vec![32],
            vec![0; 64],
        ));
    });
    assert_eq!(e.code, Code::MISSING_COMPANION_ROLE, "{}", e.detail);
    assert!(e.detail.contains("pre_quant_scale"), "{}", e.detail);

    // A global scale per expert is one per matrix, or one: not three for two experts.
    let e = refused("comfy-nvfp4", &|t, _, _| {
        let g = t
            .iter_mut()
            .find(|(k, ..)| k == "blocks.0.experts.weight_scale_2")
            .unwrap();
        g.2 = vec![3];
        g.3 = [g.3.clone(), g.3.clone(), g.3.clone()].concat();
    });
    assert_eq!(e.code, Code::MISSING_COMPANION_ROLE, "{}", e.detail);
    assert!(e.detail.contains("weight_scale_2"), "{}", e.detail);
}

#[test]
fn every_carrier_names_a_complete_encoding() {
    use tensorfs_core::format::{GlobalScale::*, NibbleOrder::*, ScaleLayout::*};
    use tensorfs_core::import::carriers::{carriers, Producer};
    for c in carriers() {
        let awq = |e: Encoding| e.with_pre_quant_scale(&Dtype::F32);
        let want = match (c.encoding.scheme.as_str(), c.producer) {
            ("nvfp4", Producer::Modelopt) => awq(Encoding::nvfp4(Multiply, LowFirst, RowMajor)),
            ("nvfp4", Producer::CompressedTensors) => Encoding::nvfp4(Divide, LowFirst, RowMajor),
            ("nvfp4", Producer::Comfy) => awq(Encoding::nvfp4(Multiply, HighFirst, Cublas128x4)),
            ("mxfp8", Producer::Comfy) => Encoding::mxfp8(Cublas128x4),
            ("mxfp8", _) => Encoding::mxfp8(RowMajor),
            ("fp8.tensorwise", _) => Encoding::fp8_tensorwise(&Dtype::F32),
            ("fp8.rowwise", _) => Encoding::fp8_rowwise(0, &Dtype::F32),
            ("int8.tensorwise", _) => Encoding::int8_tensorwise(&Dtype::F32),
            ("int8.rowwise", _) => Encoding::int8_rowwise(0, &Dtype::F32),
            ("int8.convrot", _) => Encoding::int8_convrot(256, &Dtype::F32),
            ("fp8.blockwise", _) => Encoding::fp8_blockwise(128, 128, &Dtype::F32),
            ("mxfp4", Producer::GptOss) => Encoding::mxfp4(true),
            ("mxfp4", _) => Encoding::mxfp4(false),
            other => panic!("{other:?}"),
        };
        let mut encoding = c.encoding.clone();
        for param in c.config_params.keys() {
            encoding.params.insert(param.clone(), tensorfs_core::format::Value::List(vec![128u64.into(), 128u64.into()]));
        }
        for (param, (_, default)) in &c.marker_params {
            encoding.params.insert(param.clone(), (*default).into());
        }
        for param in c.dtype_params.keys() {
            encoding.params.insert(param.clone(), "F32".into());
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

fn tensor(key: &str, dtype: &str, shape: &[u64], bytes: Vec<u8>) -> Entry {
    (key.into(), dtype.into(), shape.to_vec(), bytes)
}

fn zeros(dtype: &str, shape: &[u64]) -> Entry {
    let n = Dtype::parse(dtype).byte_len(shape).unwrap();
    (
        String::new(),
        dtype.into(),
        shape.to_vec(),
        vec![0; n as usize],
    )
}

fn named(key: &str, (_, d, s, b): Entry) -> Entry {
    (key.into(), d, s, b)
}

/// lllyasviel/flux1-dev-bnb-nf4@32d13814 (`flux1-dev-bnb-nf4-v2.safetensors`) is one file with no
/// config: each NF4 weight is `U8 [n, 1]` beside `.absmax`, `.quant_map` and
/// `.quant_state.bitsandbytes__nf4` (JSON bytes). Its keys and dtypes, at a small size, are refused,
/// not stored as plain tensors.
#[test]
fn a_bnb_nf4_single_file_is_refused() {
    let module = "model.diffusion_model.double_blocks.0.img_attn.proj";
    let state =
        br#"{"quant_type": "nf4", "blocksize": 64, "dtype": "bfloat16", "shape": [64, 64]}"#;
    let bnb = |t: &mut Tensors| {
        t.extend([
            named(&format!("{module}.bias"), zeros("BF16", &[64])),
            named(&format!("{module}.weight"), zeros("U8", &[2048, 1])),
            named(&format!("{module}.weight.absmax"), zeros("F32", &[64])),
            named(&format!("{module}.weight.quant_map"), zeros("F32", &[16])),
            tensor(
                &format!("{module}.weight.quant_state.bitsandbytes__nf4"),
                "U8",
                &[state.len() as u64],
                state.to_vec(),
            ),
        ])
    };
    let e = refused("bnb-nf4", &|t, _, _| bnb(t));
    assert_eq!(e.code, Code::UNREGISTERED_ENCODING, "{}", e.detail);
    assert!(
        e.detail.contains("bitsandbytes") && e.detail.contains(".weight.absmax"),
        "{}",
        e.detail
    );

    // Beside a declared ComfyUI carrier too.
    let e = refused("comfy-fp8", &|t, _, _| bnb(t));
    assert_eq!(e.code, Code::UNREGISTERED_ENCODING, "{}", e.detail);
    assert!(e.detail.contains("bitsandbytes"), "{}", e.detail);
}

/// GPTQ and AWQ declare themselves in `quantize_config.json`, `quant_config.json` or `gptq_*`
/// header metadata, not in `config.json`; each is refused with its producer named.
#[test]
fn gptq_and_awq_repos_are_refused_by_their_declaration() {
    let module = "model.layers.0.self_attn.q_proj";
    let gptq = |t: &mut Tensors| {
        t.extend([
            named(&format!("{module}.qweight"), zeros("I32", &[8, 64])),
            named(&format!("{module}.qzeros"), zeros("I32", &[1, 8])),
            named(&format!("{module}.scales"), zeros("F16", &[1, 64])),
            named(&format!("{module}.g_idx"), zeros("I32", &[64])),
            named(&format!("{module}.bias"), zeros("F16", &[64])),
        ])
    };
    let config =
        serde_json::json!({"architectures": ["LlamaForCausalLM"], "torch_dtype": "float16"});
    let e = refused("gptq", &|t, _, dir| {
        gptq(t);
        fs::write(dir.join("config.json"), config.to_string()).unwrap();
        let q = serde_json::json!({"bits": 4, "group_size": 128, "desc_act": false, "sym": true,
            "quant_method": "gptq", "checkpoint_format": "gptq"});
        fs::write(dir.join("quantize_config.json"), q.to_string()).unwrap();
    });
    assert_eq!(e.code, Code::UNREGISTERED_ENCODING, "{}", e.detail);
    assert!(
        e.detail.contains("quantize_config.json") && e.detail.contains("GPTQ"),
        "{}",
        e.detail
    );

    let e = refused("awq", &|t, _, dir| {
        gptq(t);
        let q = serde_json::json!({"zero_point": true, "q_group_size": 128, "w_bit": 4, "version": "GEMM"});
        fs::write(dir.join("quant_config.json"), q.to_string()).unwrap();
    });
    assert_eq!(e.code, Code::UNREGISTERED_ENCODING, "{}", e.detail);
    assert!(
        e.detail.contains("quant_config.json") && e.detail.contains("AWQ"),
        "{}",
        e.detail
    );

    let e = refused("gptq-metadata", &|t, m, _| {
        gptq(t);
        m.insert("gptq_bits".into(), "4".into());
    });
    assert_eq!(e.code, Code::UNREGISTERED_ENCODING, "{}", e.detail);
    assert!(
        e.detail.contains("gptq_bits") && e.detail.contains("GPTQ"),
        "{}",
        e.detail
    );

    // With no declaration at all, the parts name the producer.
    let e = refused("gptq-bare", &|t, _, _| gptq(t));
    assert_eq!(e.code, Code::UNREGISTERED_ENCODING, "{}", e.detail);
    assert!(
        e.detail.contains("GPTQ/AWQ") && e.detail.contains(".qweight"),
        "{}",
        e.detail
    );
}

/// Every key of a ComfyUI marker is read, decoded at the value it states, or one that only changes
/// how the module runs; any other is refused by name.
#[test]
fn comfy_marker_params_are_read_or_refused() {
    let layers = |conf: serde_json::Value| {
        let doc = serde_json::json!({"format_version": "1.0", "layers": {"proj": conf}});
        move |_: &mut Tensors, m: &mut Metadata, _: &Path| {
            m.insert("_quantization_metadata".into(), doc.to_string().into());
        }
    };
    let (_dir, ck) = edited(
        "comfy-mxfp8",
        &layers(serde_json::json!({"format": "mxfp8", "block_size": 32})),
    );
    let Body::Encoded { encoding, .. } = &ck.tensor("model", "proj.weight").unwrap().body else {
        panic!()
    };
    assert_eq!(encoding.to_string(), MXFP8_TILES);
    let e = refused(
        "comfy-mxfp8",
        &layers(serde_json::json!({"format": "mxfp8", "block_size": 64})),
    );
    assert_eq!(e.code, Code::MISSING_COMPANION_ROLE, "{}", e.detail);
    assert!(e.detail.contains("marker block_size: 64"), "{}", e.detail);

    let reconf = |module: &'static str, conf: serde_json::Value| {
        move |t: &mut Tensors, _: &mut Metadata, _: &Path| {
            let key = format!("{module}.comfy_quant");
            t.retain(|(k, ..)| *k != key);
            t.push(marker(module, conf.clone()));
        }
    };
    let fp8 = serde_json::json!({"format": "float8_e4m3fn", "full_precision_matrix_mult": true});
    edited("comfy-fp8", &reconf("attn.to_q", fp8));
    let e = refused(
        "comfy-fp8",
        &reconf(
            "attn.to_q",
            serde_json::json!({"format": "float8_e4m3fn", "weight_dtype": "bf16"}),
        ),
    );
    assert!(
        e.detail
            .contains("marker weight_dtype: \"bf16\" is not read"),
        "{}",
        e.detail
    );

    // ComfyUI reads convrot keys under `params` too.
    let conf = serde_json::json!({"format": "int8_tensorwise", "params": {"convrot": true, "convrot_groupsize": 64}});
    let (_dir, ck) = edited("comfy-int8", &reconf("ff.down", conf));
    let Body::Encoded { encoding, .. } = &ck.tensor("model", "ff.down.weight").unwrap().body else {
        panic!()
    };
    assert_eq!(
        encoding.to_string(),
        r#"int8.convrot {rotation: "hadamard", rotation_size: 64, scale_dtype: "F32"}"#
    );
}

/// xs-049's real bytes (`real-provenance.json`: 128 rows × 512 columns of each file, whole tiles,
/// with BF16 references). ComfyUI's file decodes within NVFP4's error in its own layout and far
/// outside it in any other; two files are refused until tier-1 work reads them.
#[test]
fn real_bytes_from_comfyui_and_modelopt() {
    let dir = Scratch::new("real");
    let store = Store::open(dir.0.join("store")).unwrap();
    let ck = import(&store, &fixtures().join("real-comfy-nvfp4")).unwrap();
    let key = "transformer_blocks.0.attn.to_q.weight";
    let t = ck.tensor("model", key).unwrap();
    let Body::Encoded { encoding, parts } = &t.body else {
        panic!()
    };
    assert_eq!(encoding.to_string(), NVFP4_COMFY);
    let source = read_safetensors(&fixtures().join("real-comfy-nvfp4.expect.safetensors"));
    let source = bf16s(&source[0].3);
    let error = |decoded: &[u16]| {
        let (mut err, mut norm) = (0.0, 0.0);
        for (d, s) in decoded.iter().zip(&source) {
            err += (value(*d) - value(*s)).powi(2);
            norm += value(*s).powi(2);
        }
        (err / norm).sqrt()
    };
    let right = error(&bf16s(&ck.read("model", key).unwrap()));
    assert!(right < 0.12, "{right}");
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
    // 128 rows and 32 blocks: the tiles have the row-major shape, so every reading decodes.
    for (nibbles, layout) in [
        ("low_first", "row_major"),
        ("high_first", "row_major"),
        ("low_first", "cublas_128x4"),
    ] {
        let mut other = encoding.clone();
        other.params.insert("nibble_order".into(), nibbles.into());
        other.params.insert("scale_layout".into(), layout.into());
        let wrong = error(&bf16s(&decode(&t.dtype, &t.shape, &other, &views).unwrap()));
        assert!(wrong > 0.30, "{nibbles}/{layout}: {wrong}");
    }

    // ModelOpt AWQ in ComfyUI's bytes (MiniMax-H3's text encoder): `pre_quant_scale` multiplies
    // the columns back; without it the same bytes are far off (xs-049 U8).
    let ck = import(&store, &fixtures().join("real-comfy-nvfp4-awq")).unwrap();
    let key = "model.layers.0.self_attn.o_proj.weight";
    let t = ck.tensor("model", key).unwrap();
    let Body::Encoded { encoding, parts } = &t.body else {
        panic!()
    };
    assert_eq!(encoding.to_string(), NVFP4_COMFY_AWQ);
    let roles: Vec<&str> = parts.iter().map(|p| p.role.as_str()).collect();
    assert_eq!(roles, ["weight", "scale", "scale_2", "pre_quant_scale"]);
    let source = read_safetensors(&fixtures().join("real-comfy-nvfp4-awq.expect.safetensors"));
    let source = bf16s(&source[0].3);
    let error = |decoded: &[u16]| {
        let (mut err, mut norm) = (0.0, 0.0);
        for (d, s) in decoded.iter().zip(&source) {
            err += (value(*d) - value(*s)).powi(2);
            norm += value(*s).powi(2);
        }
        (err / norm).sqrt()
    };
    let right = error(&bf16s(&ck.read("model", key).unwrap()));
    assert!(right < 0.12, "{right}");
    let bytes: Vec<Vec<u8>> = parts
        .iter()
        .map(|p| {
            let mut b = vec![0; p.length as usize];
            ck.bytes_into(&p.at(), &mut b).unwrap();
            b
        })
        .collect();
    let mut views: BTreeMap<&str, PartBytes> = parts
        .iter()
        .zip(&bytes)
        .map(|(p, b)| {
            let view = PartBytes {
                dtype: &p.dtype,
                shape: &p.shape,
                bytes: b,
            };
            (p.role.as_str(), view)
        })
        .collect();
    views.remove("pre_quant_scale");
    let mut without = encoding.clone();
    without.params.remove("pre_quant_scale_dtype");
    let wrong = error(&bf16s(&decode(&t.dtype, &t.shape, &without, &views).unwrap()));
    assert!(wrong > 0.30, "without pre_quant_scale: {wrong}");
    // ModelOpt's post-processed export carries both declarations: refused until the rule that
    // reads it (xs-049) lands.
    let e = import(&store, &fixtures().join("real-modelopt-nvfp4-comfy-marked"))
        .map(drop)
        .unwrap_err();
    assert_eq!(e.code, Code::AMBIGUOUS_CLASSIFICATION, "{}", e.detail);
}

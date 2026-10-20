//! Quantized checkpoints as ModelOpt, compressed-tensors and ComfyUI write them
//! (`fixtures/quantized/make.py`) import through the real path into a store. Each quantized weight
//! becomes one encoded tensor whose parts are the producer's bytes; the reference decoder turns it
//! into the producer's own dequantization, within a stated error of the BF16 weight it came from;
//! names and order are the plain path's. No mocks.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use tensorfs::err::Code;
use tensorfs::format::{decode, Body, Dtype, Encoding, PartBytes, SCHEMES};
use tensorfs::import::{Access, Request};
use tensorfs::store::{Checkpoint, Store};

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
/// An edit of a fixture's tensors, metadata and folder.
type Edit<'a> = dyn Fn(&mut Tensors, &mut Metadata, &Path) + 'a;

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

fn import(store: &Store, source: &Path) -> tensorfs::err::Result<Checkpoint> {
    let access = Access::default();
    let tmp = store.root().join("tmp");
    let req = Request {
        sources: &[source.display().to_string()],
        converter: None,
        source_variant: None,
        access: &access,
        tmp: &tmp,
        progress: &|_, _, _| {},
        stop: None,
    };
    let (header, _) = store.import(&req, None)?;
    Ok(store.checkpoint(&header).unwrap())
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
    let raw = |at: &tensorfs::format::Location| {
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
            if suffix.is_empty() {
                continue; // a part held in the producer's document (bnb's nested_offset)
            }
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
                encoding: r#"fp8.tensorwise {}"#,
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
                encoding: r#"fp8.rowwise {}"#,
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
            encoding: r#"fp8.tensorwise {}"#,
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
                encoding: r#"int8.tensorwise {}"#,
                parts: &[
                    ("weight", "I8", &[32, 64], "weight"),
                    ("scale", "F32", &[], "weight_scale"),
                ],
                meets: Meets::Exactly,
                error: 0.05,
            },
            Want {
                key: "ff.gate.weight",
                encoding: r#"int8.rowwise {}"#,
                parts: &[
                    ("weight", "I8", &[32, 64], "weight"),
                    ("scale", "F32", &[32, 1], "weight_scale"),
                ],
                meets: Meets::Exactly,
                error: 0.02,
            },
            Want {
                key: "ff.down.weight",
                encoding: r#"int8.convrot {rotation: "kron_regular_h4", rotation_size: 64}"#,
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
) -> (Scratch, tensorfs::err::Result<Checkpoint>) {
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
) -> tensorfs::err::Refusal {
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
            serde_json::json!({"format": "svdquant_w4a4"}),
        ));
    });
    assert_eq!(e.code, Code::UNREGISTERED_ENCODING, "{}", e.detail);
    assert!(e.detail.contains("svdquant_w4a4"), "{}", e.detail);

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
    use tensorfs::format::{GlobalScale::*, NibbleOrder::*, ScaleLayout::*};
    use tensorfs::import::carriers::{carriers, Producer};
    for c in carriers() {
        if c.from_config {
            // The producer's config names the width: every width it may name is a scheme here.
            let widths: &[u32] = match c.producer {
                Producer::Gptq => &[2, 3, 4, 8],
                Producer::Awq => &[4],
                _ => &[1, 2, 3, 4, 5, 6, 7, 8],
            };
            for bits in widths {
                let scheme = c.encoding.scheme.replace("{bits}", &bits.to_string());
                assert!(SCHEMES.iter().any(|s| s.name == scheme), "{scheme}");
            }
            continue;
        }
        let want = match (c.encoding.scheme.as_str(), c.producer) {
            ("nvfp4", Producer::Modelopt) => Encoding::nvfp4(Multiply, LowFirst, RowMajor),
            ("nvfp4", Producer::CompressedTensors) => Encoding::nvfp4(Divide, LowFirst, RowMajor),
            ("nvfp4", Producer::Comfy) => Encoding::nvfp4(Multiply, HighFirst, Cublas128x4),
            ("mxfp8", Producer::Comfy) => Encoding::mxfp8(Cublas128x4),
            ("mxfp8", _) => Encoding::mxfp8(RowMajor),
            ("fp8.tensorwise", _) => Encoding::fp8_tensorwise(),
            ("fp8.rowwise", _) => Encoding::fp8_rowwise(),
            ("int8.tensorwise", _) => Encoding::int8_tensorwise(),
            ("int8.rowwise", _) => Encoding::int8_rowwise(),
            ("int8.convrot", _) => Encoding::int8_convrot(256),
            ("fp8.blockwise", _) => Encoding::fp8_blockwise(128, 128),
            ("mxfp4", Producer::GptOss) => Encoding::mxfp4(true),
            ("mxfp4", _) => Encoding::mxfp4(false),
            ("comfy.asym_w4a8_int8", _) => Encoding::comfy_asym_w4a8_int8(16, 256),
            ("comfy.w6a8_int8", _) => Encoding::comfy_w6a8_int8(16, 256),
            ("int4.convrot", _) => Encoding::int4_convrot(256),
            other => panic!("{other:?}"),
        };
        let mut encoding = c.encoding.clone();
        for param in c.config_params.keys() {
            encoding.params.insert(param.clone(), tensorfs::format::Value::List(vec![128u64.into(), 128u64.into()]));
        }
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
        "convert",
        source.to_str().unwrap(),
        "--name",
        "local/comfy-nvfp4@1",
    ]);
    let doc: serde_json::Value =
        serde_json::from_str(tfs(&["--json", "inspect", "local/comfy-nvfp4"]).trim()).unwrap();
    let tensors = doc["components"][0]["tensors"].as_array().unwrap();
    let qkv = tensors
        .iter()
        .find(|t| t["key"] == "blocks.0.attn.qkv.weight")
        .unwrap();
    assert_eq!(
        qkv,
        &serde_json::json!({"key": "blocks.0.attn.qkv.weight", "dtype": "BF16", "shape": [144, 80], "encoding": NVFP4_COMFY})
    );
    let human = tfs(&["inspect", "local/comfy-nvfp4"]);
    assert!(
        human.contains(&format!("    2 × {NVFP4_COMFY}\n")),
        "{human}"
    );
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

const NF4_DQ: &str = "bnb.nf4 {block_size: 64, double_quant: true}";
type Parts = &'static [(&'static str, &'static str, &'static [u64], &'static str)];
const NF4_DQ_PARTS: Parts = &[
    ("weight", "U8", &[8192, 1], "weight"),
    ("absmax", "U8", &[256], "weight.absmax"),
    ("quant_map", "F32", &[16], "weight.quant_map"),
    ("nested_absmax", "F32", &[1], "weight.nested_absmax"),
    ("nested_quant_map", "F32", &[256], "weight.nested_quant_map"),
    ("nested_offset", "F32", &[], ""),
];

/// bitsandbytes reads from its `quantization_config` or, with no config, from each weight's own
/// quant_state document (`make_bnb.py`): real unsloth and Forge (lllyasviel) bytes decode bit for
/// bit as bitsandbytes dequantizes them, at NF4's ~9.5% from the BF16 originals.
#[test]
fn bitsandbytes_reads_by_its_declaration_or_its_quant_state() {
    let nf4 = |key, parts| Want {
        key,
        encoding: NF4_DQ,
        parts,
        meets: Meets::Exactly,
        error: 0.1,
    };
    check(
        "real-bnb-unsloth",
        &[
            nf4("model.layers.0.self_attn.k_proj.weight", NF4_DQ_PARTS),
            nf4("model.layers.15.mlp.down_proj.weight", NF4_DQ_PARTS),
        ],
    );
    check(
        "real-bnb-forge",
        &[Want {
            key: "model.diffusion_model.double_blocks.0.img_attn.proj.weight",
            encoding: "bnb.nf4 {block_size: 64, double_quant: false}",
            parts: &[
                ("weight", "U8", &[12288, 1], "weight"),
                ("absmax", "F32", &[384], "weight.absmax"),
                ("quant_map", "F32", &[16], "weight.quant_map"),
            ],
            meets: Meets::Exactly,
            error: 0.1,
        }],
    );
    check(
        "bnb-fp4-dq",
        &[Want {
            key: "proj.weight",
            encoding: "bnb.fp4 {block_size: 64, double_quant: true}",
            parts: &[
                ("weight", "U8", &[512, 1], "weight"),
                ("absmax", "U8", &[16], "weight.absmax"),
                ("quant_map", "F32", &[16], "weight.quant_map"),
                ("nested_absmax", "F32", &[1], "weight.nested_absmax"),
                ("nested_quant_map", "F32", &[256], "weight.nested_quant_map"),
                ("nested_offset", "F32", &[], ""),
            ],
            meets: Meets::Exactly,
            error: 0.2,
        }],
    );
    let int8 = [Want {
        key: "proj.weight",
        encoding: "bnb.int8 {}",
        parts: &[
            ("weight", "I8", &[16, 64], "weight"),
            ("SCB", "F32", &[16], "SCB"),
            ("weight_format", "U8", &[], "weight_format"),
        ],
        meets: Meets::Exactly,
        error: 0.01,
    }];
    check("bnb-int8", &int8);
    // With no config: `weight_format` beside `SCB` declares it, and the plain bias gives the dtype.
    let (_dir, ck) = try_edited("bnb-int8", &|_, _, dir| {
        fs::remove_file(dir.join("config.json")).unwrap()
    });
    let t = ck.unwrap();
    let t = t.tensor("model", "proj.weight").unwrap();
    assert!(
        matches!(&t.body, Body::Encoded { encoding, .. } if encoding.to_string() == int8[0].encoding)
    );
    assert_eq!(t.dtype, Dtype::BF16);
}

/// What bitsandbytes' documents do not state the way its scheme reads them is refused, never
/// stored as plain bytes.
#[test]
fn bitsandbytes_carriers_it_cannot_read_are_refused() {
    let w = "model.diffusion_model.double_blocks.0.img_attn.proj.weight";
    let state = format!("{w}.quant_state.bitsandbytes__nf4");
    type Edit = fn(&mut serde_json::Value);
    let cases: [(&str, Edit, Code, &str); 3] = [
        (
            "an unknown key",
            |d| d["quant_storage"] = "bfloat16".into(),
            Code::MALFORMED_JSON,
            "quant_storage",
        ),
        (
            "a nested block of 128",
            |d| {
                d["nested_blocksize"] = 128.into();
                d["nested_dtype"] = "float32".into();
                d["nested_offset"] = 0.5.into();
            },
            Code::UNREGISTERED_ENCODING,
            "nested_blocksize",
        ),
        (
            "a shape the bytes do not hold",
            |d| d["shape"] = serde_json::json!([9, 3072]),
            Code::CARRIER_GEOMETRY,
            "U8 [13824, 1]",
        ),
    ];
    for (what, edit, code, names) in cases {
        let e = refused("real-bnb-forge", &|t, _, _| {
            let e = t.iter_mut().find(|(k, ..)| *k == state).unwrap();
            let mut doc: serde_json::Value = serde_json::from_slice(&e.3).unwrap();
            edit(&mut doc);
            e.3 = doc.to_string().into_bytes();
            e.2 = vec![e.3.len() as u64];
        });
        assert_eq!(e.code, code, "{what}: {}", e.detail);
        assert!(e.detail.contains(names), "{what}: {}", e.detail);
    }
    // A table other than bnb's NF4.
    let e = refused("real-bnb-forge", &|t, _, _| {
        let e = t
            .iter_mut()
            .find(|(k, ..)| *k == format!("{w}.quant_map"))
            .unwrap();
        e.3[..4].copy_from_slice(&0.5f32.to_le_bytes());
    });
    assert_eq!(e.code, Code::UNREGISTERED_ENCODING, "{}", e.detail);
    assert!(e.detail.contains("nf4 table"), "{}", e.detail);
    // Its parts with no quant_state and no config declare nothing.
    let e = refused("real-bnb-forge", &|t, _, _| t.retain(|(k, ..)| *k != state));
    assert_eq!(e.code, Code::UNREGISTERED_ENCODING, "{}", e.detail);
    assert!(
        e.detail.contains("bitsandbytes") && e.detail.contains(".weight.absmax"),
        "{}",
        e.detail
    );
    // Beside a declared ComfyUI carrier: two producers.
    let e = refused("comfy-fp8", &|t, _, _| {
        for (k, d, s, b) in read_safetensors(&fixtures().join("real-bnb-forge/model.safetensors")) {
            t.push((k, d, s, b));
        }
    });
    assert_eq!(e.code, Code::AMBIGUOUS_CLASSIFICATION, "{}", e.detail);
    assert!(
        e.detail.contains("Bitsandbytes") && e.detail.contains("Comfy"),
        "{}",
        e.detail
    );
}

/// GPTQ and AWQ declare themselves in `quantize_config.json`, `quant_config.json` or `gptq_*`
/// header metadata, not in `config.json`; each is refused with its producer named.
/// GPTQ and AWQ declare themselves in `config.json`, `quantize_config.json`, AutoAWQ's
/// `quant_config.json` or GPTQ's `gptq_*` header metadata; each is read for the scheme's params, and
/// what no scheme here reads is refused by name.
#[test]
fn gptq_and_awq_declarations_are_read_or_refused_by_name() {
    let module = "model.layers.0.self_attn.q_proj";
    // K = 64 inputs, N = 64 outputs, 4-bit, one group of 128.
    let gptq = |t: &mut Tensors| {
        t.extend([
            named(&format!("{module}.qweight"), zeros("I32", &[8, 64])),
            named(&format!("{module}.qzeros"), zeros("I32", &[1, 8])),
            named(&format!("{module}.scales"), zeros("F16", &[1, 64])),
            named(&format!("{module}.g_idx"), zeros("I32", &[64])),
            named(&format!("{module}.bias"), zeros("F16", &[64])),
        ])
    };
    let awq = |t: &mut Tensors| {
        t.extend([
            named(&format!("{module}.qweight"), zeros("I32", &[64, 8])),
            named(&format!("{module}.qzeros"), zeros("I32", &[1, 8])),
            named(&format!("{module}.scales"), zeros("F16", &[1, 64])),
        ])
    };
    let config =
        serde_json::json!({"architectures": ["LlamaForCausalLM"], "torch_dtype": "float16"});
    let write = |dir: &Path, name: &str, doc: serde_json::Value| {
        fs::write(dir.join(name), doc.to_string()).unwrap()
    };
    let encoding = |ck: &Checkpoint| {
        let t = ck.tensor("model", &format!("{module}.weight")).unwrap();
        assert_eq!((&t.dtype, &t.shape[..]), (&Dtype::F16, &[64, 64][..]));
        match &t.body {
            Body::Encoded { encoding, .. } => encoding.to_string(),
            _ => panic!("plain"),
        }
    };
    let gptq_config = serde_json::json!({"bits": 4, "group_size": 128, "desc_act": false, "sym": true,
        "quant_method": "gptq", "checkpoint_format": "gptq"});
    let (_d, ck) = edited("gptq", &|t, _, dir| {
        gptq(t);
        write(dir, "config.json", config.clone());
        write(dir, "quantize_config.json", gptq_config.clone());
    });
    let v1 = "gptq.int4 {act_order: false, group_size: 128, symmetric: true, zero_offset: 1}";
    assert_eq!(encoding(&ck), v1);
    // AutoGPTQ's header metadata and GPTQModel's gptq_v2, in agreement with config.json.
    let (_d, ck) = edited("gptq-v2", &|t, m, dir| {
        gptq(t);
        m.insert("gptq_bits".into(), "4".into());
        m.insert("gptq_desc_act".into(), "False".into());
        let mut q = config.clone();
        q["quantization_config"] = serde_json::json!({"bits": 4, "group_size": 128, "desc_act": false,
            "sym": false, "quant_method": "gptq", "format": "gptq_v2", "checkpoint_format": "gptq_v2"});
        write(dir, "config.json", q);
    });
    let v2 = "gptq.int4 {act_order: false, group_size: 128, symmetric: false, zero_offset: 0}";
    assert_eq!(encoding(&ck), v2);
    let (_d, ck) = edited("awq", &|t, _, dir| {
        awq(t);
        write(dir, "config.json", config.clone());
        let q = serde_json::json!({"zero_point": true, "q_group_size": 128, "w_bit": 4, "version": "GEMM"});
        write(dir, "quant_config.json", q);
    });
    assert_eq!(encoding(&ck), "awq.int4 {group_size: 128}");

    let refusals: [(&str, &Edit<'_>, &str); 6] = [
        // AutoGPTQ's header metadata states no sym.
        (
            "gptq-metadata-only",
            &|t, m, _| {
                gptq(t);
                for (k, v) in [
                    ("gptq_bits", "4"),
                    ("gptq_group_size", "128"),
                    ("gptq_desc_act", "False"),
                ] {
                    m.insert(k.into(), v.into());
                }
            },
            "sym is not stated",
        ),
        (
            "gptq-marlin",
            &|t, _, dir| {
                gptq(t);
                write(dir, "config.json", config.clone());
                let mut q = gptq_config.clone();
                q["checkpoint_format"] = "marlin".into();
                write(dir, "quantize_config.json", q);
            },
            "GPTQ format \"marlin\" is not read",
        ),
        (
            "gptq-disagree",
            &|t, m, dir| {
                gptq(t);
                write(dir, "config.json", config.clone());
                write(dir, "quantize_config.json", gptq_config.clone());
                m.insert("gptq_bits".into(), "8".into());
            },
            "states bits",
        ),
        (
            "gptq-dynamic",
            &|t, _, dir| {
                gptq(t);
                write(dir, "config.json", config.clone());
                let mut q = gptq_config.clone();
                q["dynamic"] = serde_json::json!({"-:.*down_proj": {}});
                write(dir, "quantize_config.json", q);
            },
            "dynamic",
        ),
        (
            "awq-gemv",
            &|t, _, dir| {
                awq(t);
                let q = serde_json::json!({"zero_point": true, "q_group_size": 128, "w_bit": 4, "version": "GEMV"});
                write(dir, "quant_config.json", q);
            },
            "AWQ version Some(\"GEMV\") is not read",
        ),
        ("gptq-bare", &|t, _, _| gptq(t), "no declaration of their producer"),
    ];
    for (name, edit, why) in refusals {
        let e = refused(name, edit);
        assert_eq!(e.code, Code::UNREGISTERED_ENCODING, "{name}: {}", e.detail);
        assert!(e.detail.contains(why), "{name}: {}", e.detail);
    }
}

/// Bands of real GPTQ, AWQ and compressed-tensors int4, W8A8 and FP8_DYNAMIC checkpoints
/// (`make_int.py`) import under their declared encodings and decode to their bf16 originals at
/// their format's error (AWQ's in its folded space: the bf16 columns times its input scale,
/// `norm_bf16 / norm_awq`).
#[test]
fn real_int_checkpoints_import_and_decode() {
    let cases: [(&str, &str, &str, f64); 6] = [
        (
            "real-gptq-int4",
            "F16",
            "gptq.int4 {act_order: false, group_size: 128, symmetric: true, zero_offset: 1}",
            0.19,
        ),
        ("real-awq-int4", "F16", "awq.int4 {group_size: 128}", 0.17),
        (
            "real-ct-int4-asym",
            "BF16",
            r#"ct.int4 {group_size: 64, packing: "dense", strategy: "group", symmetric: false}"#,
            0.13,
        ),
        (
            "real-ct-int4-act-order",
            "BF16",
            r#"ct.int4 {group_size: 64, packing: "dense", strategy: "group", symmetric: true}"#,
            0.12,
        ),
        (
            "real-ct-int8-channel",
            "BF16",
            r#"int8.rowwise {}"#,
            0.02,
        ),
        (
            "real-ct-fp8-channel",
            "BF16",
            r#"fp8.rowwise {}"#,
            0.03,
        ),
    ];
    for (case, dtype, want, bound) in cases {
        let dir = Scratch::new(case);
        let store = Store::open(dir.0.join("store")).unwrap();
        let ck = import(&store, &fixtures().join(case)).unwrap();
        let source: BTreeMap<String, Vec<u8>> =
            read_safetensors(&fixtures().join(format!("{case}.expect.safetensors")))
                .into_iter()
                .map(|(k, _, _, b)| (k, b))
                .collect();
        let model: BTreeMap<String, Vec<u8>> =
            read_safetensors(&fixtures().join(case).join("model.safetensors"))
                .into_iter()
                .map(|(k, _, _, b)| (k, b))
                .collect();
        let weights: Vec<&String> = source
            .keys()
            .filter(|k| k.ends_with("proj.weight"))
            .collect();
        assert_eq!(weights.len(), 3, "{case}");
        for key in weights {
            let key = key.strip_prefix("source.").unwrap();
            let t = ck.tensor("model", key).unwrap();
            let Body::Encoded { encoding, .. } = &t.body else {
                panic!("{case} {key} is plain")
            };
            assert_eq!(
                (t.dtype.name(), encoding.to_string().as_str()),
                (dtype, want),
                "{case} {key}"
            );
            let decoded: Vec<f64> = match &t.dtype {
                Dtype::F16 => (ck.read("model", key).unwrap().as_chunks::<2>().0.iter())
                    .map(|b| half(u16::from_le_bytes(*b)))
                    .collect(),
                _ => bf16s(&ck.read("model", key).unwrap())
                    .into_iter()
                    .map(value)
                    .collect(),
            };
            // AWQ divided the norm before the module by its input scale.
            let layer = key.split('.').take(3).collect::<Vec<_>>().join(".");
            let norm = ["input_layernorm", "post_attention_layernorm"]
                .iter()
                .map(|n| format!("{layer}.{n}.weight"))
                .find(|n| source.contains_key(&format!("source.{n}")));
            let fold: Vec<f64> = match norm {
                Some(n) => {
                    let awq: Vec<f64> = (model[&n].as_chunks::<2>().0.iter())
                        .map(|b| half(u16::from_le_bytes(*b)))
                        .collect();
                    let bf: Vec<f64> = bf16s(&source[&format!("source.{n}")])
                        .into_iter()
                        .map(value)
                        .collect();
                    bf.iter().zip(&awq).map(|(b, a)| b / a).collect()
                }
                None => vec![1.0],
            };
            let reference: Vec<f64> = bf16s(&source[&format!("source.{key}")])
                .into_iter()
                .map(value)
                .collect();
            let (mut err, mut sum) = (0.0, 0.0);
            for (n, (d, r)) in decoded.iter().zip(&reference).enumerate() {
                let r = r * fold[n % t.shape[1] as usize % fold.len()];
                err += (d - r).powi(2);
                sum += r * r;
            }
            let distance = (err / sum).sqrt();
            assert!(distance < bound, "{case} {key}: {distance}");
        }
    }
}

fn half(h: u16) -> f64 {
    let (e, m) = (((h >> 10) & 31) as i32, (h & 0x3ff) as f64);
    let v = match e {
        0 => m * 2f64.powi(-24),
        _ => (1024.0 + m) * 2f64.powi(e - 25),
    };
    if h & 0x8000 != 0 {
        -v
    } else {
        v
    }
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
        r#"int8.convrot {rotation: "kron_regular_h4", rotation_size: 64}"#
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
    assert_eq!(encoding.to_string(), NVFP4_COMFY);
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
    let wrong = error(&bf16s(&decode(&t.dtype, &t.shape, encoding, &views).unwrap()));
    assert!(wrong > 0.30, "without pre_quant_scale: {wrong}");
    // ModelOpt's post-processed export carries both declarations: refused until the rule that
    // reads it (xs-049) lands.
    let e = import(&store, &fixtures().join("real-modelopt-nvfp4-comfy-marked"))
        .map(drop)
        .unwrap_err();
    assert_eq!(e.code, Code::AMBIGUOUS_CLASSIFICATION, "{}", e.detail);
}

const W4A8: &str = r#"comfy.asym_w4a8_int8 {group_size: 16, rotation: "kron_regular_h4", rotation_size: 256}"#;
const W6A8: &str = r#"comfy.w6a8_int8 {group_size: 32, rotation: "kron_regular_h4", rotation_size: 256}"#;
const W4A4: &str = r#"int4.convrot {rotation: "kron_regular_h4", rotation_size: 256}"#;

/// xs-049 U7's real bytes (`comfy-provenance.json`: 64 rows × 512 columns of MiniMax-H3 linears,
/// with Comfy-Org's bf16 weights): each ComfyUI integer format imports at its marker's params and
/// decodes to the spec's rotation of kitchen's own int8 values, within its error of the bf16 weight
/// (7.3% W4A8, 2.3% W6A8, 16–22% W4A4 over whole layers).
#[test]
fn comfy_w4a8_w6a8_w4a4_from_real_files() {
    check(
        "real-comfy-w4a8",
        &[Want {
            key: "blocks.20.attn.qkv_proj.weight",
            encoding: W4A8,
            parts: &[
                ("weight", "I8", &[64, 256], "weight"),
                ("group_scale", "F8_E4M3", &[64, 32], "weight_s_rel"),
                ("scale", "F32", &[64], "weight_s_channel"),
                ("codebook", "F32", &[16], "weight_codebook"),
            ],
            meets: Meets::Exactly,
            error: 0.08,
        }],
    );
    check(
        "real-comfy-w6a8",
        &[Want {
            key: "blocks.20.mlp.fc1.weight",
            encoding: W6A8,
            parts: &[
                ("weight", "I8", &[64, 384], "weight"),
                ("group_scale", "F8_E4M3", &[64, 16], "weight_s_rel"),
                ("scale", "F32", &[64], "weight_s_channel"),
            ],
            meets: Meets::Exactly,
            error: 0.03,
        }],
    );
    check(
        "real-comfy-w4a4",
        &[Want {
            key: "blocks.20.attn.out_proj.weight",
            encoding: W4A4,
            parts: &[
                ("weight", "I8", &[64, 256], "weight"),
                ("scale", "F32", &[64], "weight_scale"),
            ],
            meets: Meets::Exactly,
            error: 0.25,
        }],
    );
}

/// The markers the real files write (with `convrot`, without it, in `_quantization_metadata` with
/// toxicdog's `quant_group_size: 64`) read; `U8` group scales (JoaoZaokk's LTX-2.5) read as
/// `F8_E4M3`. What ComfyUI would decode differently or kitchen refuses is refused by name.
#[test]
fn comfy_int_markers_and_shapes_are_read_or_refused() {
    let w4a8 = "blocks.20.attn.qkv_proj";
    let w4a4 = "blocks.20.attn.out_proj";
    let reconf = |module: &'static str, conf: serde_json::Value| {
        move |t: &mut Tensors, _: &mut Metadata, _: &Path| {
            let key = format!("{module}.comfy_quant");
            t.retain(|(k, ..)| *k != key);
            t.push(marker(module, conf.clone()));
        }
    };
    let encoding = |ck: &Checkpoint, module: &str| match &ck
        .tensor("model", &format!("{module}.weight"))
        .unwrap()
        .body
    {
        Body::Encoded { encoding, parts } => {
            let dtypes: Vec<&str> = parts.iter().map(|p| p.dtype.name()).collect();
            format!("{encoding} {dtypes:?}")
        }
        Body::Data(_) => panic!("plain"),
    };
    let conf =
        serde_json::json!({"format": "asym_w4a8_int8", "group_size": 16, "convrot_groupsize": 256});
    let (_d, ck) = edited("real-comfy-w4a8", &reconf(w4a8, conf));
    assert_eq!(
        encoding(&ck, w4a8),
        format!(r#"{W4A8} ["I8", "F8_E4M3", "F32", "F32"]"#)
    );
    let u8_scales = |t: &mut Tensors, _: &mut Metadata, _: &Path| {
        let s = t
            .iter_mut()
            .find(|(k, ..)| k.ends_with(".weight_s_rel"))
            .unwrap();
        s.1 = "U8".into();
    };
    let (_d, ck) = edited("real-comfy-w4a8", &u8_scales);
    assert_eq!(
        encoding(&ck, w4a8),
        format!(r#"{W4A8} ["I8", "F8_E4M3", "F32", "F32"]"#)
    );
    let metadata = |t: &mut Tensors, m: &mut Metadata, _: &Path| {
        t.retain(|(k, ..)| !k.ends_with(".comfy_quant"));
        let layers = serde_json::json!({"format_version": "1.0", "layers": {w4a4: {
            "format": "convrot_w4a4", "convrot_groupsize": 256, "quant_group_size": 64, "linear_dtype": "int8"}}});
        m.insert("_quantization_metadata".into(), layers.to_string().into());
    };
    let (_d, ck) = edited("real-comfy-w4a4", &metadata);
    assert_eq!(encoding(&ck, w4a4), format!(r#"{W4A4} ["I8", "F32"]"#));

    let refusal = |case, edit: &dyn Fn(&mut Tensors, &mut Metadata, &Path), want: &str| {
        let e = refused(case, edit);
        assert!(e.detail.contains(want), "{want}: {}", e.detail);
    };
    let conf = serde_json::json!({"format": "asym_w4a8_int8", "group_size": 16, "convrot": false});
    refusal(
        "real-comfy-w4a8",
        &reconf(w4a8, conf),
        "with convrot: false",
    );
    let conf = serde_json::json!({"format": "asym_w4a8_int8", "group_size": 32, "convrot": true});
    refusal("real-comfy-w4a8", &reconf(w4a8, conf), "not [64, 16]");
    let conf = serde_json::json!({"format": "asym_w4a8_int8", "mma_packed": true});
    refusal(
        "real-comfy-w4a8",
        &reconf(w4a8, conf),
        "marker mma_packed: true is not read",
    );
    let six = |t: &mut Tensors, m: &mut Metadata, p: &Path| {
        t.retain(|(k, ..)| !k.ends_with(".weight_codebook"));
        reconf(
            w4a8,
            serde_json::json!({"format": "w6a8_int8", "group_size": 16}),
        )(t, m, p);
    };
    refusal("real-comfy-w4a8", &six, "not 6-bit rows");
    let conf = serde_json::json!({"format": "convrot_w4a4", "quant_group_size": 32});
    refusal(
        "real-comfy-w4a4",
        &reconf(w4a4, conf),
        "marker quant_group_size: 32",
    );
    let group_scales = |t: &mut Tensors, _: &mut Metadata, _: &Path| {
        let s = t
            .iter_mut()
            .find(|(k, ..)| k.ends_with(".weight_scale"))
            .unwrap();
        s.2 = vec![8, 8];
    };
    refusal("real-comfy-w4a4", &group_scales, "not [64] or [64]+[1]");
}

#[test]
fn comfy_empty_packed_rows_recover_large_widths_without_overflow() {
    // These headers contain no weight bytes. Multiplying their physical widths by eight
    // overflows even though the logical widths and every part's zero byte count fit.
    let columns = 1u64 << 63;
    for (format, bits, group) in [
        ("convrot_w4a4", 4, None),
        ("asym_w4a8_int8", 4, Some(16)),
        ("w6a8_int8", 6, Some(32)),
    ] {
        let width = if bits == 4 {
            columns / 2
        } else {
            (columns / 4) * 3
        };
        let (_dir, ck) = edited("empty-comfy-width", &|t, _, _| {
            let conf = match group {
                Some(g) => serde_json::json!({
                    "format": format, "convrot_groupsize": 256, "group_size": g
                }),
                None => serde_json::json!({
                    "format": format, "convrot_groupsize": 256
                }),
            };
            t.push(marker("proj", conf));
            t.push(("norm.weight".into(), "BF16".into(), vec![1], vec![0; 2]));
            t.push(("proj.weight".into(), "I8".into(), vec![0, width], vec![]));
            match group {
                None => t.push(("proj.weight_scale".into(), "F32".into(), vec![0], vec![])),
                Some(g) => {
                    t.push((
                        "proj.weight_s_rel".into(),
                        "F32".into(),
                        vec![0, columns / g],
                        vec![],
                    ));
                    t.push((
                        "proj.weight_s_channel".into(),
                        "F32".into(),
                        vec![0],
                        vec![],
                    ));
                }
            }
        });
        assert_eq!(
            ck.tensor("model", "proj.weight").unwrap().shape,
            [0, columns]
        );
    }
    for (format, width, why) in [
        ("convrot_w4a4", 1u64 << 63, "packed logical width overflow"),
        (
            "w6a8_int8",
            (1u64 << 62) * 3,
            "packed logical width overflow",
        ),
        ("w6a8_int8", 1u64 << 61, "not 6-bit rows"),
    ] {
        let e = refused("invalid-comfy-width", &|t, _, _| {
            t.push(marker(
                "proj",
                serde_json::json!({"format": format, "convrot_groupsize": 256}),
            ));
            t.push(("norm.weight".into(), "BF16".into(), vec![1], vec![0; 2]));
            t.push(("proj.weight".into(), "I8".into(), vec![0, width], vec![]));
            if format == "convrot_w4a4" {
                t.push(("proj.weight_scale".into(), "F32".into(), vec![0], vec![]));
            } else {
                t.push(("proj.weight_s_rel".into(), "F32".into(), vec![0, 1], vec![]));
                t.push((
                    "proj.weight_s_channel".into(),
                    "F32".into(),
                    vec![0],
                    vec![],
                ));
            }
        });
        assert_eq!(e.code, Code::MISSING_COMPANION_ROLE, "{}", e.detail);
        assert!(e.detail.contains(why), "{format}: {}", e.detail);
    }
}

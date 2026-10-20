//! Real `fp8.blockwise` and `mxfp4` checkpoints (`fixtures/quantized/make_real.py`: rows of DeepSeek-V3.1 and
//! V4, Qwen3-FP8, RedHatAI FP8-BLOCK, gpt-oss-20b, AMD Quark and compressed-tensors files, read by HTTP range at
//! pinned revisions) import through the real path. Each quantized weight becomes one encoded tensor whose parts
//! are the producer's bytes, decodes to exactly what the producer's own reader gives, and lands within a stated
//! error of the BF16 original where one exists. No mocks.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use tensorfs::err::Code;
use tensorfs::format::{decode, Body, Dtype, Encoding, PartBytes};
use tensorfs::import::{Access, Request};
use tensorfs::store::{Checkpoint, Store};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/quantized")
}

/// A safetensors file's tensors: key -> (dtype, shape, bytes).
fn tensors(path: &Path) -> BTreeMap<String, (String, Vec<u64>, Vec<u8>)> {
    let raw = fs::read(path).unwrap();
    let n = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let head: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&raw[8..8 + n]).unwrap();
    (head.into_iter().filter(|(k, _)| k != "__metadata__"))
        .map(|(k, t)| {
            let at: Vec<usize> = serde_json::from_value(t["data_offsets"].clone()).unwrap();
            let shape = serde_json::from_value(t["shape"].clone()).unwrap();
            let bytes = raw[8 + n + at[0]..8 + n + at[1]].to_vec();
            (k, (t["dtype"].as_str().unwrap().to_owned(), shape, bytes))
        })
        .collect()
}

fn import(case: &str, store: &Path) -> tensorfs::err::Result<Checkpoint> {
    let store = Store::open(store).unwrap();
    let (access, tmp) = (Access::default(), store.root().join("tmp"));
    let req = Request {
        sources: &[fixtures().join(case).display().to_string()],
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

fn values(bf16: &[u8]) -> Vec<f64> {
    (bf16.as_chunks::<2>().0.iter())
        .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16) as f64)
        .collect()
}

/// One quantized weight: its encoding as `tfs inspect` shows it, its parts (role, dtype, shape, the producer's key)
/// and, where a BF16 original exists, the bound on `‖decode − original‖ / ‖original‖`.
struct Want {
    key: &'static str,
    encoding: &'static str,
    parts: &'static [(&'static str, &'static str, &'static [u64], &'static str)],
    error: Option<f64>,
}

fn check(case: &str, wants: &[Want]) {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("{case}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let ck = import(case, &dir).unwrap();
    let file = tensors(&fixtures().join(case).join("model.safetensors"));
    let expect = tensors(&fixtures().join(format!("{case}.expect.safetensors")));
    let raw = |at: &tensorfs::format::Location| {
        let mut buf = vec![0; at.length as usize];
        ck.bytes_into(at, &mut buf).unwrap();
        buf
    };
    let mut consumed = Vec::new();
    for want in wants {
        let t = ck.tensor("model", want.key).unwrap();
        let Body::Encoded { encoding, parts } = &t.body else {
            panic!("{}: plain", want.key)
        };
        assert_eq!(encoding.to_string(), want.encoding, "{}", want.key);
        let dequant = &expect[&format!("dequant.{}", want.key)];
        assert_eq!(t.shape, dequant.1, "{}: the logical shape", want.key);
        assert_eq!(t.dtype, Dtype::BF16, "{}", want.key);
        for (p, (role, dtype, shape, key)) in parts.iter().zip(want.parts) {
            assert_eq!(
                (p.role.as_str(), p.dtype.name(), p.shape.as_slice()),
                (*role, *dtype, *shape),
                "{}",
                want.key
            );
            assert_eq!(
                raw(&p.at()),
                file[*key].2,
                "{} part {role}: the producer's bytes",
                want.key
            );
            consumed.push(*key);
        }
        assert_eq!(parts.len(), want.parts.len(), "{}", want.key);
        let decoded = ck.read("model", want.key).unwrap();
        assert!(
            decoded == expect[&format!("dequant.{}", want.key)].2,
            "{}: the producer's reader",
            want.key
        );
        if let Some(bound) = want.error {
            let (d, s) = (
                values(&decoded),
                values(&expect[&format!("source.{}", want.key)].2),
            );
            let err: f64 = d.iter().zip(&s).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
            let error = (err / s.iter().map(|b| b * b).sum::<f64>()).sqrt();
            assert!(
                error <= bound,
                "{}: error {error:.4} over {bound}",
                want.key
            );
        }
    }
    // Everything else is the file's plain tensor, unchanged.
    for (key, (_, _, bytes)) in &file {
        if consumed.contains(&key.as_str()) {
            continue;
        }
        let Body::Data(at) = &ck.tensor("model", key).unwrap().body else {
            panic!("{key}: encoded")
        };
        assert_eq!(&raw(at), bytes, "{key}");
    }
    let _ = fs::remove_dir_all(&dir);
}

const TILES: &str = r#"fp8.blockwise {block_size: [128, 128]}"#;
const MXFP4: &str = "mxfp4 {block_size: 32, transposed: false}";
const MXFP4_TRANSPOSED: &str = "mxfp4 {block_size: 32, transposed: true}";

#[test]
fn deepseek_dialect_fp8_tiles() {
    // Qwen3's FP8: `quant_method: fp8`, `weight_scale_inv` F16, multiplied; a whole tile and a short one.
    check(
        "real-qwen3-fp8",
        &[
            Want {
                key: "model.layers.0.self_attn.q_proj.weight",
                encoding: TILES,
                parts: &[
                    (
                        "weight",
                        "F8_E4M3",
                        &[128, 128],
                        "model.layers.0.self_attn.q_proj.weight",
                    ),
                    (
                        "scale",
                        "F16",
                        &[1, 1],
                        "model.layers.0.self_attn.q_proj.weight_scale_inv",
                    ),
                ],
                error: Some(0.04),
            },
            Want {
                key: "model.layers.0.mlp.down_proj.weight",
                encoding: TILES,
                parts: &[
                    (
                        "weight",
                        "F8_E4M3",
                        &[64, 128],
                        "model.layers.0.mlp.down_proj.weight",
                    ),
                    (
                        "scale",
                        "F16",
                        &[1, 1],
                        "model.layers.0.mlp.down_proj.weight_scale_inv",
                    ),
                ],
                error: Some(0.04),
            },
        ],
    );
    // DeepSeek-V3.1 under `scale_fmt: ue8m0` still stores F32 scales: the part's dtype says which.
    check(
        "real-deepseek-v3.1",
        &[Want {
            key: "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            encoding: TILES,
            parts: &[
                (
                    "weight",
                    "F8_E4M3",
                    &[64, 128],
                    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                ),
                (
                    "scale",
                    "F32",
                    &[1, 1],
                    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_inv",
                ),
            ],
            error: None,
        }],
    );
    // DeepSeek-V4: F8_E8M0 tiles under `.scale`, and FP4 routed experts (`expert_dtype: fp4`) as mxfp4.
    check(
        "real-deepseek-v4",
        &[
            Want {
                key: "layers.1.attn.wkv.weight",
                encoding: TILES,
                parts: &[
                    ("weight", "F8_E4M3", &[16, 256], "layers.1.attn.wkv.weight"),
                    ("scale", "F8_E8M0", &[1, 2], "layers.1.attn.wkv.scale"),
                ],
                error: None,
            },
            Want {
                key: "layers.1.ffn.experts.0.w2.weight",
                encoding: MXFP4,
                parts: &[
                    (
                        "weight",
                        "F4",
                        &[2, 2048],
                        "layers.1.ffn.experts.0.w2.weight",
                    ),
                    (
                        "scale",
                        "F8_E8M0",
                        &[2, 64],
                        "layers.1.ffn.experts.0.w2.scale",
                    ),
                ],
                error: None,
            },
        ],
    );
}

#[test]
fn compressed_tensors_block_and_mxfp4() {
    check(
        "real-ct-fp8-block",
        &[Want {
            key: "model.layers.0.self_attn.q_proj.weight",
            encoding: TILES,
            parts: &[
                (
                    "weight",
                    "F8_E4M3",
                    &[128, 128],
                    "model.layers.0.self_attn.q_proj.weight",
                ),
                (
                    "scale",
                    "BF16",
                    &[1, 1],
                    "model.layers.0.self_attn.q_proj.weight_scale",
                ),
            ],
            error: Some(0.04),
        }],
    );
    check(
        "real-ct-mxfp4",
        &[Want {
            key: "model.layers.0.self_attn.k_proj.weight",
            encoding: MXFP4,
            parts: &[
                (
                    "weight",
                    "F4",
                    &[4, 2048],
                    "model.layers.0.self_attn.k_proj.weight_packed",
                ),
                (
                    "scale",
                    "F8_E8M0",
                    &[4, 64],
                    "model.layers.0.self_attn.k_proj.weight_scale",
                ),
            ],
            error: Some(0.16),
        }],
    );
}

#[test]
fn gpt_oss_and_quark_mxfp4() {
    // gpt-oss's `{module}_blocks [E, N, K/32, 16]` are transformers' `{module}` `[E, K, N]`, stored transposed:
    // exactly unsloth/gpt-oss-20b-BF16's tensors (the square `down_proj` included).
    check(
        "real-gpt-oss",
        &[
            Want {
                key: "model.layers.0.mlp.experts.down_proj",
                encoding: MXFP4_TRANSPOSED,
                parts: &[
                    (
                        "weight",
                        "F4",
                        &[2, 64, 64],
                        "model.layers.0.mlp.experts.down_proj_blocks",
                    ),
                    (
                        "scale",
                        "F8_E8M0",
                        &[2, 64, 2],
                        "model.layers.0.mlp.experts.down_proj_scales",
                    ),
                ],
                error: Some(0.0),
            },
            Want {
                key: "model.layers.0.mlp.experts.gate_up_proj",
                encoding: MXFP4_TRANSPOSED,
                parts: &[
                    (
                        "weight",
                        "F4",
                        &[2, 64, 64],
                        "model.layers.0.mlp.experts.gate_up_proj_blocks",
                    ),
                    (
                        "scale",
                        "F8_E8M0",
                        &[2, 64, 2],
                        "model.layers.0.mlp.experts.gate_up_proj_scales",
                    ),
                ],
                error: Some(0.0),
            },
        ],
    );
    // The hazard `transposed` removes: read untransposed, the square down_proj has the same name and shape as
    // unsloth's tensor and different values.
    let file = tensors(&fixtures().join("real-gpt-oss/model.safetensors"));
    let expect = tensors(&fixtures().join("real-gpt-oss.expect.safetensors"));
    let (blocks, scales) = (
        &file["model.layers.0.mlp.experts.down_proj_blocks"].2,
        &file["model.layers.0.mlp.experts.down_proj_scales"].2,
    );
    let parts = BTreeMap::from([
        (
            "weight",
            PartBytes {
                dtype: &Dtype::F4,
                shape: &[2, 64, 64],
                bytes: blocks,
            },
        ),
        (
            "scale",
            PartBytes {
                dtype: &Dtype::F8_E8M0,
                shape: &[2, 64, 2],
                bytes: scales,
            },
        ),
    ]);
    let read = |transposed: bool| {
        decode(
            &Dtype::BF16,
            &[2, 64, 64],
            &Encoding::mxfp4(transposed),
            &parts,
        )
        .unwrap()
    };
    let source = &expect["source.model.layers.0.mlp.experts.down_proj"].2;
    assert_eq!(&read(true), source);
    assert_ne!(&read(false), source);
    // AMD Quark (`pack_method: reorder`) re-quantized the same expert: it decodes to OpenAI's values exactly.
    check(
        "real-quark-mxfp4",
        &[Want {
            key: "model.layers.0.mlp.experts.0.down_proj.weight",
            encoding: MXFP4,
            parts: &[
                (
                    "weight",
                    "F4",
                    &[4, 2880],
                    "model.layers.0.mlp.experts.0.down_proj.weight",
                ),
                (
                    "scale",
                    "F8_E8M0",
                    &[4, 90],
                    "model.layers.0.mlp.experts.0.down_proj.weight_scale",
                ),
            ],
            error: Some(0.0),
        }],
    );
}

/// The case's files in a scratch dir, `config.json` edited.
fn edited(case: &str, edit: impl FnOnce(&mut serde_json::Value)) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("edited-{case}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join(case)).unwrap();
    fs::copy(
        fixtures().join(case).join("model.safetensors"),
        dir.join(case).join("model.safetensors"),
    )
    .unwrap();
    let mut config: serde_json::Value =
        serde_json::from_slice(&fs::read(fixtures().join(case).join("config.json")).unwrap())
            .unwrap();
    edit(&mut config);
    fs::write(dir.join(case).join("config.json"), config.to_string()).unwrap();
    dir
}

#[test]
fn unproven_or_undeclared_layouts_are_refused() {
    let refused = |case: &str, edit: &dyn Fn(&mut serde_json::Value)| {
        let dir = edited(case, edit);
        let store = Store::open(dir.join("store")).unwrap();
        let (access, tmp) = (Access::default(), store.root().join("tmp"));
        let source = dir.join(case).display().to_string();
        let req = Request {
            sources: &[source],
            converter: None,
            source_variant: None,
            access: &access,
            tmp: &tmp,
            progress: &|_, _, _| {},
            stop: None,
        };
        let e = store.import(&req, None).map(drop).unwrap_err();
        let _ = fs::remove_dir_all(&dir);
        e
    };
    // Quark's nibble order is proven for `pack_method: reorder` only.
    let e = refused("real-quark-mxfp4", &|c| {
        c["quantization_config"]["export"]["pack_method"] = "order".into()
    });
    assert_eq!(e.code, Code::UNREGISTERED_ENCODING, "{}", e.detail);
    assert!(e.detail.contains("pack_method order"), "{}", e.detail);
    // Per-tensor FineGrainedFP8 (no weight_block_size) is not fp8.blockwise.
    let e = refused("real-qwen3-fp8", &|c| {
        c["quantization_config"]["weight_block_size"] = serde_json::Value::Null
    });
    assert_eq!(e.code, Code::UNREGISTERED_ENCODING, "{}", e.detail);
    // A block size the scales do not tile is refused, never read on another grid.
    let e = refused("real-qwen3-fp8", &|c| {
        c["quantization_config"]["weight_block_size"] = serde_json::json!([64, 64])
    });
    assert_eq!(e.code, Code::MISSING_COMPANION_ROLE, "{}", e.detail);
    assert!(e.detail.contains("not tiles"), "{}", e.detail);
    // V4's FP4 experts without `expert_dtype: fp4`: I8 bytes with E8M0 scales are no carrier the declaration names.
    let e = refused("real-deepseek-v4", &|c| {
        c.as_object_mut().unwrap().remove("expert_dtype");
    });
    assert_eq!(e.code, Code::MISSING_COMPANION_ROLE, "{}", e.detail);
    // gpt-oss's blocks with no declaration at all are refused, never stored as plain U8.
    let e = refused("real-gpt-oss", &|c| {
        c.as_object_mut().unwrap().remove("quantization_config");
    });
    assert_eq!(e.code, Code::UNREGISTERED_ENCODING, "{}", e.detail);
    assert!(e.detail.contains("down_proj_scales"), "{}", e.detail);
}

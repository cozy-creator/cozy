//! GGUF files as gguf-py writes them (`fixtures/gguf/make.py`) import through the real path into
//! a store and export back byte for byte, with no mocks. An LLM's tensors take their upstream
//! names: llama's Q/K rows un-permuted and expert stacks split, checked against the bytes the
//! converter started from. A diffusion file keeps its keys.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use tensorfs::err::Code;
use tensorfs::export;
use tensorfs::format::decode::{elem::f16, f16 as f16_bits};
use tensorfs::format::{encode, Body, Dtype, PartBytes, Value};
use tensorfs::import::gguf::{self, At};
use tensorfs::import::{remote, Access, Request};
use tensorfs::store::{Checkpoint, Store};

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Scratch {
        let path = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("gguf-{name}-{}", std::process::id()));
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

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/gguf")
        .join(name)
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

/// A safetensors file of U8 tensors: name -> bytes.
fn expected(path: &Path) -> BTreeMap<String, Vec<u8>> {
    let raw = fs::read(path).unwrap();
    let n = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let head: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&raw[8..8 + n]).unwrap();
    head.into_iter()
        .map(|(k, t)| {
            let at: Vec<usize> = serde_json::from_value(t["data_offsets"].clone()).unwrap();
            (k, raw[8 + n + at[0]..8 + n + at[1]].to_vec())
        })
        .collect()
}

/// A tensor at a glance: dtype, shape, scheme with params, parts (role, dtype, shape).
fn glance(ck: &Checkpoint, key: &str) -> String {
    let t = ck.tensor("model", key).unwrap();
    match &t.body {
        Body::Data(_) => format!("{} {:?}", t.dtype, t.shape),
        Body::Encoded { encoding, parts } => {
            let params: Vec<String> = (encoding.params.iter())
                .map(|(k, v)| format!("{k}: {v}"))
                .collect();
            let parts: Vec<String> = (parts.iter())
                .map(|p| format!("{} {} {:?}", p.role, p.dtype, p.shape))
                .collect();
            format!(
                "{} {:?} {} {{{}}} {}",
                t.dtype,
                t.shape,
                encoding.scheme,
                params.join(","),
                parts.join(" · ")
            )
        }
    }
}

fn stored(ck: &Checkpoint, key: &str, role: &str) -> Vec<u8> {
    let t = ck.tensor("model", key).unwrap();
    let (at, length) = match &t.body {
        Body::Data(at) => (*at, at.length),
        Body::Encoded { parts, .. } => {
            let p = parts.iter().find(|p| p.role == role).unwrap();
            (p.at(), p.length)
        }
    };
    let mut buf = vec![0; length as usize];
    ck.bytes_into(&at, &mut buf).unwrap();
    buf
}

#[test]
fn llama_takes_upstream_names_and_exports_byte_identical() {
    let scratch = Scratch::new("llama");
    let store = Store::open(scratch.0.join("store")).unwrap();
    let source = fixture("llama-moe.gguf");
    let ck = import(&store, &source).unwrap();
    let h = ck.header();
    assert_eq!(h.components().len(), 1);
    let keys: Vec<&str> = h.components()[0]
        .tensors
        .iter()
        .map(|t| t.key.as_str())
        .collect();
    assert_eq!(
        keys,
        [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
            "rope_freqs.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.block_sparse_moe.gate.weight",
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            "model.layers.0.block_sparse_moe.experts.1.w1.weight",
            "model.layers.0.block_sparse_moe.experts.0.w2.weight",
            "model.layers.0.block_sparse_moe.experts.1.w2.weight",
            "model.layers.0.block_sparse_moe.experts.0.w3.weight",
            "model.layers.0.block_sparse_moe.experts.1.w3.weight",
        ],
        "GGUF order, upstream names, experts split in place"
    );
    for (key, want) in [
        (
            "model.embed_tokens.weight",
            "F32 [32, 64] gguf.q8_0 {block_size: 32} blocks U8 [32, 2, 34]",
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            "F32 [32, 64] gguf.q4_1 {block_size: 32} blocks U8 [32, 2, 20]",
        ),
        ("model.layers.0.self_attn.v_proj.weight", "F16 [32, 64]"),
        ("model.layers.0.self_attn.o_proj.weight", "BF16 [64, 64]"),
        ("model.layers.0.block_sparse_moe.gate.weight", "F32 [2, 64]"),
        (
            "model.layers.0.block_sparse_moe.experts.1.w2.weight",
            "F32 [64, 32] gguf.q5_1 {block_size: 32} blocks U8 [64, 1, 24]",
        ),
        (
            "model.layers.0.block_sparse_moe.experts.0.w3.weight",
            "F32 [32, 64] gguf.nvfp4 {block_size: 64} blocks U8 [32, 1, 36] · scale F32 [1] · \
             input_scale F32 [1]",
        ),
    ] {
        assert_eq!(glance(&ck, key), want, "{key}");
    }
    // Q/K un-permuted, experts and their scales split: the converter's upstream bytes.
    for (name, bytes) in expected(&fixture("llama-moe.expect.safetensors")) {
        let (key, role) = name.split_once(':').unwrap_or((&name, ""));
        assert_eq!(stored(&ck, key, role), bytes, "{name}");
    }
    let meta = &h.metadata;
    assert_eq!(meta.get("source"), Some(&Value::Str(source.canonicalize().unwrap().display().to_string())));
    let record: serde_json::Value =
        serde_json::from_slice(&ck.asset("model", "llama-moe.gguf.json").unwrap()).unwrap();
    let kv: BTreeMap<String, (String, serde_json::Value)> =
        serde_json::from_value::<Vec<(String, String, serde_json::Value)>>(record["kv"].clone())
            .unwrap()
            .into_iter()
            .map(|(k, t, v)| (k, (t, v)))
            .collect();
    assert_eq!(kv["test.f32"], ("f32".into(), 0.1.into()));
    assert_eq!(kv["test.nan"], ("f32".into(), "0x7fc00000".into()));
    assert_eq!(
        kv["test.u64"],
        ("u64".into(), (9223372036854775813u64).into())
    );
    assert_eq!(
        kv["test.nested"].1,
        serde_json::json!(["array", [["i32", [1, 2]], ["i32", [3]]]])
    );

    let out = scratch.0.join("out");
    assert_eq!(export::gguf(&ck, &out).unwrap(), ["llama-moe.gguf"]);
    assert_eq!(
        fs::read(out.join("llama-moe.gguf")).unwrap(),
        fs::read(&source).unwrap()
    );
}

#[test]
fn flux_keeps_its_keys_and_exports_byte_identical() {
    let scratch = Scratch::new("flux");
    let store = Store::open(scratch.0.join("store")).unwrap();
    let source = fixture("flux.gguf");
    let ck = import(&store, &source).unwrap();
    for (key, want) in [
        (
            "double_blocks.0.q4_k.weight",
            "F32 [3, 256] gguf.q4_k {block_size: 256} blocks U8 [3, 1, 144]",
        ),
        (
            "double_blocks.0.mxfp4.weight",
            "F32 [3, 64] gguf.mxfp4 {block_size: 32} blocks U8 [3, 2, 17]",
        ),
        (
            "double_blocks.0.q2_0.weight",
            "F32 [3, 64] gguf.q2_0 {block_size: 64} blocks U8 [3, 1, 18]",
        ),
        (
            "double_blocks.0.iq4_nl.weight",
            "F32 [3, 32] gguf.iq4_nl {block_size: 32} blocks U8 [3, 1, 18]",
        ),
        ("double_blocks.0.img_attn.qkv.bias", "F32 [6]"),
        ("img_in.weight", "F16 [4, 8]"),
        ("plain.int64", "I64 [2, 3]"),
        ("plain.float64", "F64 [2, 3]"),
        (
            "input_blocks.0.0.weight",
            "F32 [8, 4, 4, 4] gguf.q8_0 {block_size: 32} blocks U8 [16, 34]",
        ),
    ] {
        assert_eq!(glance(&ck, key), want, "{key}");
    }
    let out = scratch.0.join("out");
    assert_eq!(export::gguf(&ck, &out).unwrap(), ["flux.gguf"]);
    assert_eq!(
        fs::read(out.join("flux.gguf")).unwrap(),
        fs::read(&source).unwrap()
    );
}

/// A GGUF of one f32 tensor `a` `[32]`, its version and offset as given.
fn tiny(version: [u8; 4], offset: u64) -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend(version);
    b.extend(1u64.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    b.extend(1u64.to_le_bytes());
    b.push(b'a');
    b.extend(1u32.to_le_bytes());
    b.extend(32u64.to_le_bytes());
    b.extend(0u32.to_le_bytes());
    b.extend(offset.to_le_bytes());
    b.resize(b.len().next_multiple_of(32) + 128 + offset as usize, 0);
    b
}

#[test]
fn malformed_files_are_refused() {
    let scratch = Scratch::new("refused");
    let store = Store::open(scratch.0.join("store")).unwrap();
    let cases: [(&str, Vec<u8>, Option<Code>); 4] = [
        ("ok", tiny(3u32.to_le_bytes(), 0), None),
        (
            "offset",
            tiny(3u32.to_le_bytes(), 32),
            Some(Code::CARRIER_GEOMETRY),
        ),
        (
            "big-endian",
            tiny(3u32.to_be_bytes(), 0),
            Some(Code::UNKNOWN_FORMAT),
        ),
        (
            "truncated",
            tiny(3u32.to_le_bytes(), 0)[..60].to_vec(),
            Some(Code::CARRIER_TRUNCATED),
        ),
    ];
    for (name, bytes, code) in cases {
        let dir = scratch.0.join(name);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}.gguf"));
        fs::write(&path, bytes).unwrap();
        match (import(&store, &path), code) {
            (Ok(ck), None) => assert_eq!(glance(&ck, "a"), "F32 [32]"),
            (Err(e), Some(code)) => assert_eq!(e.code, code, "{name}: {}", e.detail),
            (got, _) => panic!("{name}: {:?}", got.map(|_| ())),
        }
    }
    // A folder of GGUFs is one file per quantization: the import names one.
    let both = scratch.0.join("both");
    fs::create_dir_all(&both).unwrap();
    fs::copy(fixture("flux.gguf"), both.join("a.gguf")).unwrap();
    fs::copy(fixture("flux.gguf"), both.join("b.gguf")).unwrap();
    let e = import(&store, &both).err().unwrap();
    assert_eq!(e.code, Code::AMBIGUOUS_CLASSIFICATION, "{}", e.detail);
}

#[test]
fn tfs_imports_and_exports_a_gguf() {
    let scratch = Scratch::new("cli");
    let source = fixture("llama-moe.gguf");
    let tfs = |args: &[&str]| {
        let out = Command::new(env!("CARGO_BIN_EXE_tfs"))
            .arg("--store")
            .arg(scratch.0.join("store"))
            .args(args)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    };
    tfs(&["convert", source.to_str().unwrap(), "--name", "local/llama@1"]);
    let out = scratch.0.join("out");
    tfs(&["export", "local/llama@1", out.to_str().unwrap(), "--gguf"]);
    assert_eq!(
        fs::read(out.join("llama-moe.gguf")).unwrap(),
        fs::read(&source).unwrap()
    );
    tfs(&[
        "export",
        "local/llama@1",
        scratch.0.join("st").to_str().unwrap(),
        "--safetensors",
    ]);
    assert!(scratch.0.join("st/model.safetensors").exists());
}

/// The first member a source lists.
fn member(src: &str, access: &Access) -> remote::Member {
    let listing = remote::list(&remote::parse(src).unwrap(), access).unwrap();
    listing.members.into_iter().next().unwrap()
}

#[test]
#[ignore = "reads city96's FLUX.1-dev and Qwen's Qwen3-30B-A3B GGUFs by HTTP range"]
fn real_files_read_by_range() {
    let access = Access::default();
    let m = member("hf://city96/FLUX.1-dev-gguf/flux1-dev-Q4_K_S.gguf", &access);
    let src = gguf::Remote {
        member: &m,
        access: &access,
    };
    let g = gguf::read(&src).unwrap();
    assert_eq!(g.tensors.len(), 780);
    let p = gguf::plan(&g, "model").unwrap();
    assert!(p.arch.is_none() && !p.normalized);
    let proj = (p.tensors.iter())
        .find(|o| o.key == "double_blocks.0.img_attn.proj.weight")
        .unwrap();
    assert_eq!(proj.shape, [3072, 3072]);
    assert_eq!(proj.encoding.as_ref().unwrap().scheme, "gguf.q4_k");
    assert_eq!(proj.parts[0].shape, [3072, 12, 144]);
    let (start, len) = proj.parts[0].ranges[0];
    let mut bytes = vec![0; len as usize];
    src.read_at(start, &mut bytes).unwrap();
    // Every block's d and dmin are finite, non-negative f16: the offsets are the structs'.
    for b in bytes.chunks(144) {
        for h in [&b[0..2], &b[2..4]] {
            let h = u16::from_le_bytes([h[0], h[1]]);
            assert!(h & 0x7c00 != 0x7c00 && h & 0x8000 == 0, "{h:#06x}");
        }
    }

    let m = member(
        "hf://Qwen/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf",
        &access,
    );
    let g = gguf::read(&gguf::Remote {
        member: &m,
        access: &access,
    })
    .unwrap();
    let p = gguf::plan(&g, "model").unwrap();
    assert_eq!(p.class.as_deref(), Some("Qwen3MoeForCausalLM"));
    assert!(p.normalized);
    let keys: std::collections::HashSet<&str> = p.tensors.iter().map(|o| o.key.as_str()).collect();
    for key in [
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.47.mlp.gate.weight",
        "model.layers.47.mlp.experts.127.down_proj.weight",
    ] {
        assert!(keys.contains(key), "{key}");
    }
    let experts = keys.iter().filter(|k| k.contains(".mlp.experts.")).count();
    assert_eq!(experts, 48 * 128 * 3);
}

/// Bytes `[from, from + n)` of a ranged source.
fn range(src: &dyn At, from: u64, n: u64) -> Vec<u8> {
    let mut buf = vec![0; n as usize];
    src.read_at(from, &mut buf).unwrap();
    buf
}

/// The encoder against llama-quantize on a real file: mradermacher's static Q4_K_M of
/// Xenova/llama2.c-stories15M (q5_0, q8_0, q4_k, q6_k), quantized from the F16 conversion of
/// the upstream F32 weights. Encoding the upstream rows, rounded to F16, gives the imported
/// (un-permuted) blocks exactly.
#[test]
#[ignore = "reads mradermacher's and Xenova's stories15M by HTTP range"]
fn the_encoder_reproduces_llama_quantize() {
    let access = Access::default();
    let m = member(
        "hf://mradermacher/llama2.c-stories15M-GGUF/llama2.c-stories15M.Q4_K_M.gguf",
        &access,
    );
    let src = gguf::Remote {
        member: &m,
        access: &access,
    };
    let p = gguf::plan(&gguf::read(&src).unwrap(), "model").unwrap();
    let up = member("hf://Xenova/llama2.c-stories15M/model.safetensors", &access);
    let up = gguf::Remote {
        member: &up,
        access: &access,
    };
    let n = u64::from_le_bytes(range(&up, 0, 8).try_into().unwrap());
    let head: serde_json::Value = serde_json::from_slice(&range(&up, 8, n)).unwrap();
    let mut seen = std::collections::BTreeSet::new();
    for o in p.tensors.iter().filter(|o| o.encoding.is_some()) {
        let scheme = &o.encoding.as_ref().unwrap().scheme;
        // Tied embeddings: the upstream file names the tensor it shares instead.
        let t = match &head[&o.key] {
            serde_json::Value::Null => &head[head["__metadata__"][&o.key].as_str().unwrap()],
            t => t,
        };
        let at: Vec<u64> = serde_json::from_value(t["data_offsets"].clone()).unwrap();
        assert_eq!(t["dtype"], "F32");
        let values: Vec<u8> = range(&up, 8 + n + at[0], at[1] - at[0])
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|b| f16(f16_bits(f32::from_le_bytes(*b))).to_le_bytes())
            .collect();
        let view = PartBytes {
            dtype: &Dtype::F32,
            shape: &o.shape,
            bytes: &values,
        };
        let input = encode::Input {
            dtype: &o.dtype,
            shape: &o.shape,
            encoding: o.encoding.as_ref().unwrap(),
            arrays: &BTreeMap::from([("values", view)]),
        };
        let ours = encode::quantize(&input).unwrap();
        // One read of the tensor's span, then its rows in the planned (upstream) order.
        let ranges = &o.parts[0].ranges;
        let lo = ranges.iter().map(|r| r.0).min().unwrap();
        let hi = ranges.iter().map(|r| r.0 + r.1).max().unwrap();
        let span = range(&src, lo, hi - lo);
        let stored: Vec<u8> = (ranges.iter())
            .flat_map(|(from, len)| &span[(from - lo) as usize..(from - lo + len) as usize])
            .copied()
            .collect();
        assert!(ours[0].bytes == stored, "{} {scheme}", o.key);
        seen.insert(scheme.clone());
    }
    let want = ["gguf.q4_k", "gguf.q5_0", "gguf.q6_k", "gguf.q8_0"];
    assert_eq!(seen.into_iter().collect::<Vec<_>>(), want);
}

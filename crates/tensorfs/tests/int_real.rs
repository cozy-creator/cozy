//! Real int4 checkpoints decode to their bf16 originals: bands of three layers of Qwen's GPTQ-Int4 and
//! AWQ and RedHatAI's compressed-tensors w4a16 (`tests/fixtures/quantized/real-*`, made by
//! `make_int.py`), each read under the encoding its producer declares, land at int4's error from
//! the bf16 rows (AWQ's after its norm fold is undone); the readings that ignore the zero offset or
//! the act order do not.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde_json::Value as Json;
use tensorfs::format::decode::elem;
use tensorfs::format::{decode, Dtype, Encoding, PartBytes, Value};

type Tensors = BTreeMap<String, (Dtype, Vec<u64>, Vec<u8>)>;

/// A safetensors file's tensors.
fn read(path: &Path) -> Tensors {
    let raw = fs::read(path).unwrap();
    let n = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let hdr: BTreeMap<String, Json> = serde_json::from_slice(&raw[8..8 + n]).unwrap();
    let data = &raw[8 + n..];
    (hdr.into_iter())
        .filter(|(k, _)| k != "__metadata__")
        .map(|(k, t)| {
            let dtype = Dtype::parse(t["dtype"].as_str().unwrap());
            let shape = t["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|e| e.as_u64().unwrap())
                .collect();
            let at = t["data_offsets"].as_array().unwrap();
            let (a, b) = (
                at[0].as_u64().unwrap() as usize,
                at[1].as_u64().unwrap() as usize,
            );
            (k, (dtype, shape, data[a..b].to_vec()))
        })
        .collect()
}

fn encoding(scheme: &str, params: &[(&str, Value)]) -> Encoding {
    Encoding {
        scheme: scheme.into(),
        params: params
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    }
}

/// A tensor's elements (`F32`, `BF16` or `F16`).
fn floats((dtype, _, bytes): &(Dtype, Vec<u64>, Vec<u8>)) -> Vec<f64> {
    elem::floats(dtype, bytes)
        .into_iter()
        .map(f64::from)
        .collect()
}

/// Relative L2 distance of `module` read under `encoding` from its bf16 rows; `roles` maps each
/// producer suffix to its part. With `norm`, the bf16 column `j` is first multiplied by AWQ's
/// input scale, `norm_bf16[j] / norm_quantized[j]`: the weight AWQ quantized.
fn distance(
    case: &str,
    module: &str,
    encoding: &Encoding,
    roles: &[(&str, &str)],
    norm: Option<&str>,
) -> f64 {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/quantized");
    let model = read(&dir.join(case).join("model.safetensors"));
    let expect = read(&dir.join(format!("{case}.expect.safetensors")));
    let (_, shape, want) = &expect[&format!("source.{module}.weight")];
    let parts: BTreeMap<&str, PartBytes> = (roles.iter())
        .filter_map(|(suffix, role)| {
            let (dtype, shape, bytes) = model.get(&format!("{module}.{suffix}"))?;
            Some((
                *role,
                PartBytes {
                    dtype,
                    shape,
                    bytes,
                },
            ))
        })
        .collect();
    let got = decode(&Dtype::F32, shape, encoding, &parts)
        .unwrap_or_else(|e| panic!("{case} {module}: {e}"));
    let got = floats(&(Dtype::F32, vec![], got));
    let fold: Vec<f64> = match norm {
        Some(n) => {
            let key = format!("{n}.weight");
            let (q, o) = (
                floats(&model[&key]),
                floats(&expect[&format!("source.{key}")]),
            );
            q.iter().zip(&o).map(|(q, o)| o / q).collect()
        }
        None => vec![1.0; shape[1] as usize],
    };
    let want = floats(&(Dtype::BF16, vec![], want.clone()));
    let want = want
        .iter()
        .enumerate()
        .map(|(t, w)| w * fold[t % fold.len()]);
    let (err, sum) = got.iter().zip(want).fold((0.0, 0.0), |(e, s), (g, w)| {
        (e + (g - w).powi(2), s + w * w)
    });
    (err / sum).sqrt()
}

const MODULES: [&str; 3] = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.12.mlp.down_proj",
    "model.layers.23.self_attn.o_proj",
];
const GPTQ: [(&str, &str); 4] = [
    ("qweight", "qweight"),
    ("qzeros", "qzeros"),
    ("scales", "scales"),
    ("g_idx", "g_idx"),
];
const CT: [(&str, &str); 5] = [
    ("weight_packed", "weight_packed"),
    ("weight_scale", "weight_scale"),
    ("weight_zero_point", "weight_zero_point"),
    ("weight_shape", "weight_shape"),
    ("weight_g_idx", "weight_g_idx"),
];

/// Int4 round to nearest lands at 11–14% of these rows' norm (12–14% symmetric with groups of
/// 128); GPTQ's error compensation trades a few points of weight error for output error.
const INT4: f64 = 0.2;

#[test]
fn gptq_reads_qwen_gptq_int4() {
    let v1 = |sym: bool, offset: u64| {
        let p = [
            ("group_size", 128u64.into()),
            ("symmetric", sym.into()),
            ("act_order", false.into()),
            ("zero_offset", offset.into()),
        ];
        encoding("gptq.int4", &p)
    };
    for m in MODULES {
        let d = distance("real-gptq-int4", m, &v1(true, 1), &GPTQ, None);
        let v2 = distance("real-gptq-int4", m, &v1(false, 0), &GPTQ, None);
        assert!(d < INT4 && v2 > 2.0 * d, "{m}: {d} (as v2: {v2})");
    }
}

#[test]
fn awq_reads_qwen_awq() {
    let e = encoding("awq.int4", &[("group_size", 128u64.into())]);
    let roles = [
        ("qweight", "qweight"),
        ("qzeros", "qzeros"),
        ("scales", "scales"),
    ];
    let modules = [
        (
            "model.layers.0.self_attn.q_proj",
            Some("model.layers.0.input_layernorm"),
        ),
        (
            "model.layers.12.mlp.gate_proj",
            Some("model.layers.12.post_attention_layernorm"),
        ),
        ("model.layers.23.self_attn.o_proj", None),
    ];
    for (m, norm) in modules {
        let d = distance("real-awq-int4", m, &e, &roles, norm);
        let unfolded = distance("real-awq-int4", m, &e, &roles, None);
        assert!(
            d < INT4 && (norm.is_none() || unfolded > 2.0 * d),
            "{m}: {d} (fold kept: {unfolded})"
        );
    }
}

#[test]
fn ct_reads_redhat_w4a16() {
    let ct = |sym: bool| {
        let p = [
            ("group_size", 64u64.into()),
            ("strategy", "group".into()),
            ("symmetric", sym.into()),
            ("packing", "dense".into()),
        ];
        encoding("ct.int4", &p)
    };
    for m in MODULES {
        let d = distance("real-ct-int4-asym", m, &ct(false), &CT, None);
        assert!(d < INT4, "asymmetric {m}: {d}");
        let d = distance("real-ct-int4-act-order", m, &ct(true), &CT, None);
        let column_order = distance("real-ct-int4-act-order", m, &ct(true), &CT[..4], None);
        assert!(
            d < INT4 && column_order > 2.0 * d,
            "act order {m}: {d} (in column order: {column_order})"
        );
    }
}

//! nvfp4's encoders on real producer output (`conformance/mine/nvfp4.py`): round to nearest is
//! ModelOpt 0.47.0's byte for byte, and `pack` writes ModelOpt's and ComfyUI's own bytes, ComfyUI's
//! fused experts with their per-expert `scale_2` and tiles included.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use tensorfs_core::format::encode::{self, Input, OwnedPart};
use tensorfs_core::format::{
    Dtype, Encoding, GlobalScale, NibbleOrder, PartBytes, ScaleLayout, decode,
};

/// A safetensors file: key -> (dtype, shape, bytes).
type Tensors = BTreeMap<String, (Dtype, Vec<u64>, Vec<u8>)>;

fn read(path: &str) -> Tensors {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/quantized")
        .join(path);
    let raw = fs::read(&path).unwrap();
    let n = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let head: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&raw[8..8 + n]).unwrap();
    let data = &raw[8 + n..];
    head.into_iter()
        .filter(|(k, _)| k != "__metadata__")
        .map(|(k, t)| {
            let at: [usize; 2] = serde_json::from_value(t["data_offsets"].clone()).unwrap();
            let dtype: Dtype = serde_json::from_value(t["dtype"].clone()).unwrap();
            let shape = serde_json::from_value(t["shape"].clone()).unwrap();
            (k, (dtype, shape, data[at[0]..at[1]].to_vec()))
        })
        .collect()
}

fn view<'a>(dtype: &'a Dtype, shape: &'a [u64], bytes: &'a [u8]) -> PartBytes<'a> {
    PartBytes {
        dtype,
        shape,
        bytes,
    }
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    (bytes.as_chunks::<2>().0.iter())
        .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16))
        .collect()
}

fn le(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn run(
    f: fn(&Input) -> tensorfs_core::format::Result<Vec<OwnedPart>>,
    encoding: &Encoding,
    shape: &[u64],
    arrays: &BTreeMap<&str, PartBytes>,
) -> BTreeMap<String, OwnedPart> {
    let input = Input {
        dtype: &Dtype::BF16,
        shape,
        encoding,
        arrays,
    };
    f(&input)
        .unwrap()
        .into_iter()
        .map(|p| (p.role.clone(), p))
        .collect()
}

fn decoded(encoding: &Encoding, shape: &[u64], parts: &BTreeMap<String, OwnedPart>) -> Vec<u8> {
    let views = parts
        .iter()
        .map(|(r, p)| (r.as_str(), view(&p.dtype, &p.shape, &p.bytes)))
        .collect();
    decode(&Dtype::BF16, shape, encoding, &views).unwrap()
}

#[test]
fn round_to_nearest_is_modelopts() {
    let file = read("modelopt-nvfp4-awq/model.safetensors");
    let expect = read("modelopt-nvfp4-awq.expect.safetensors");
    let modelopt = Encoding::nvfp4(
        GlobalScale::Multiply,
        NibbleOrder::LowFirst,
        ScaleLayout::RowMajor,
    );
    let modules = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ];
    for module in modules {
        let m = format!("model.layers.0.{module}");
        // What ModelOpt's RTN was given: the weight after its AWQ smoothing moved into it.
        let given = (expect.get(&format!("quantized.{m}.weight")))
            .unwrap_or_else(|| &expect[&format!("source.{m}.weight")]);
        let (_, shape, bf16) = given;
        let values = le(&f32s(bf16));
        let (_, _, scale_2) = &file[&format!("{m}.weight_scale_2")];
        for fused in [true, false] {
            let mut arrays = BTreeMap::from([("values", view(&Dtype::F32, shape, &values))]);
            // q/k/v and gate/up share one scale_2 (ModelOpt fuses them); o and down have their own.
            if fused {
                arrays.insert("scale_2", view(&Dtype::F32, &[], scale_2));
            } else if !["self_attn.o_proj", "mlp.down_proj"].contains(&module) {
                continue;
            }
            let parts = run(encode::quantize, &modelopt, shape, &arrays);
            for (role, suffix) in [
                ("weight", "weight"),
                ("scale", "weight_scale"),
                ("scale_2", "weight_scale_2"),
            ] {
                assert_eq!(
                    parts[role].bytes,
                    file[&format!("{m}.{suffix}")].2,
                    "{m} {role} (fused {fused})"
                );
            }
        }
    }
}

#[test]
fn pre_quant_scale_smooths_and_multiplies_back() {
    // RTN given AWQ's factor: the parts carry it, and the decode is the source weight within NVFP4's error.
    let file = read("modelopt-nvfp4-awq/model.safetensors");
    let expect = read("modelopt-nvfp4-awq.expect.safetensors");
    let m = "model.layers.0.self_attn.o_proj";
    let (_, shape, source) = &expect[&format!("source.{m}.weight")];
    let (pd, ps, p) = &file[&format!("{m}.pre_quant_scale")];
    let values = le(&f32s(source));
    let arrays = BTreeMap::from([
        ("values", view(&Dtype::F32, shape, &values)),
        ("pre_quant_scale", view(pd, ps, p)),
    ]);
    let awq = Encoding::nvfp4(
        GlobalScale::Multiply,
        NibbleOrder::LowFirst,
        ScaleLayout::RowMajor,
    )
    .with_pre_quant_scale(pd);
    let parts = run(encode::quantize, &awq, shape, &arrays);
    assert_eq!(&parts["pre_quant_scale"].bytes, p);
    let out = f32s(&decoded(&awq, shape, &parts));
    let src = f32s(source);
    let err: f32 = out
        .iter()
        .zip(&src)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    let norm: f32 = src.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(err / norm < 0.1, "{}", err / norm);
    // Without it the same encoding refuses: the param and the part go together.
    let plain = Encoding::nvfp4(
        GlobalScale::Multiply,
        NibbleOrder::LowFirst,
        ScaleLayout::RowMajor,
    );
    let input = Input {
        dtype: &Dtype::BF16,
        shape,
        encoding: &plain,
        arrays: &arrays,
    };
    assert!(encode::quantize(&input).is_err());
}

/// The producer's codes (one per element) and row-major scales back out of decoded parts.
fn codes_of(weight: &[u8], high_first: bool) -> Vec<u8> {
    let (even, odd) = if high_first { (4, 0) } else { (0, 4) };
    weight
        .iter()
        .flat_map(|b| [(b >> even) & 15, (b >> odd) & 15])
        .collect()
}

#[test]
fn pack_writes_the_producers_bytes() {
    // ModelOpt: codes, scales, scale_2 and pre_quant_scale give its export's parts.
    let file = read("modelopt-nvfp4-awq/model.safetensors");
    let m = "model.layers.0.self_attn.o_proj";
    let t = |s: &str| &file[&format!("{m}.{s}")];
    let (_, ws, w) = t("weight");
    let shape = [ws[0], ws[1] * 2];
    let codes = codes_of(w, false);
    let arrays = BTreeMap::from([
        ("codes", view(&Dtype::U8, &shape, &codes)),
        (
            "scale",
            view(
                &t("weight_scale").0,
                &t("weight_scale").1,
                &t("weight_scale").2,
            ),
        ),
        (
            "scale_2",
            view(
                &t("weight_scale_2").0,
                &t("weight_scale_2").1,
                &t("weight_scale_2").2,
            ),
        ),
        (
            "input_scale",
            view(
                &t("input_scale").0,
                &t("input_scale").1,
                &t("input_scale").2,
            ),
        ),
        (
            "pre_quant_scale",
            view(
                &t("pre_quant_scale").0,
                &t("pre_quant_scale").1,
                &t("pre_quant_scale").2,
            ),
        ),
    ]);
    let awq = Encoding::nvfp4(
        GlobalScale::Multiply,
        NibbleOrder::LowFirst,
        ScaleLayout::RowMajor,
    )
    .with_pre_quant_scale(&Dtype::BF16);
    let parts = run(encode::pack, &awq, &shape, &arrays);
    for (role, suffix) in [
        ("weight", "weight"),
        ("scale", "weight_scale"),
        ("scale_2", "weight_scale_2"),
        ("input_scale", "input_scale"),
        ("pre_quant_scale", "pre_quant_scale"),
    ] {
        assert_eq!(&parts[role].bytes, &t(suffix).2, "{role}");
    }

    // ComfyUI's fused experts (Comfy-Org/Lens): the same codes and row-major scales packed in its
    // layout are its file's bytes, tiles per expert and weight_scale_2 [E]; both layouts decode to
    // OpenAI's MXFP4 original.
    let lens = read("comfy-nvfp4-experts/model.safetensors");
    let expect = read("comfy-nvfp4-experts.expect.safetensors");
    let m = "layers.1.mlp.experts.down_proj";
    let t = |s: &str| &lens[&format!("{m}.{s}")];
    let (_, ws, w) = t("weight");
    let shape = [ws[0], ws[1], ws[2] * 2];
    let codes = codes_of(w, true);
    let comfy = Encoding::nvfp4(
        GlobalScale::Multiply,
        NibbleOrder::HighFirst,
        ScaleLayout::Cublas128x4,
    );
    let tiled = BTreeMap::from([
        ("weight", view(&Dtype::F4, &shape, w)),
        (
            "scale",
            view(
                &t("weight_scale").0,
                &t("weight_scale").1,
                &t("weight_scale").2,
            ),
        ),
        (
            "scale_2",
            view(
                &t("weight_scale_2").0,
                &t("weight_scale_2").1,
                &t("weight_scale_2").2,
            ),
        ),
    ]);
    let source = &expect[&format!("source.{m}.weight")].2;
    assert_eq!(
        &decode(&Dtype::BF16, &shape, &comfy, &tiled).unwrap(),
        source
    );
    // Row-major scales, read back through the tiles.
    let (rows, w16) = (shape[1] as usize, shape[2] as usize / 16);
    let tiles = &t("weight_scale").2;
    let row_major: Vec<u8> = (0..shape[0] as usize)
        .flat_map(|e| (0..rows).flat_map(move |i| (0..w16).map(move |b| (e, i, b))))
        .map(|(e, i, b)| tiles[e * 512 + 16 * (i % 32) + 4 * (i % 128 / 32) + b])
        .collect();
    let row_shape = [shape[0], shape[1], w16 as u64];
    let arrays = BTreeMap::from([
        ("codes", view(&Dtype::U8, &shape, &codes)),
        ("scale", view(&Dtype::F8_E4M3, &row_shape, &row_major)),
        (
            "scale_2",
            view(
                &t("weight_scale_2").0,
                &t("weight_scale_2").1,
                &t("weight_scale_2").2,
            ),
        ),
    ]);
    let parts = run(encode::pack, &comfy, &shape, &arrays);
    assert_eq!(&parts["weight"].bytes, w);
    assert_eq!(&parts["scale"].bytes, tiles);
    assert_eq!(
        (&parts["scale"].shape, &parts["scale_2"].shape),
        (&t("weight_scale").1, &t("weight_scale_2").1)
    );
    let modelopt = Encoding::nvfp4(
        GlobalScale::Multiply,
        NibbleOrder::LowFirst,
        ScaleLayout::RowMajor,
    );
    let parts = run(encode::pack, &modelopt, &shape, &arrays);
    assert_eq!(parts["scale"].shape, row_shape);
    assert_eq!(&decoded(&modelopt, &shape, &parts), source);
}

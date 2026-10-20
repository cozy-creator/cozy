//! `comfy.*` encoders against comfy-kitchen's other options: every case of
//! `conformance/vectors/cases/comfy.json` with `kitchen` (source and options) comes out of
//! `quantize_with` as exactly its parts, and decodes to its expected bytes. Kitchen's defaults
//! (`source`) are `format_encode.rs`'s.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use base64::Engine;
use serde::Deserialize;
use tensorfs::format::encode::{quantize, quantize_with, Input, Kitchen};
use tensorfs::format::{decode, Dtype, Encoding, PartBytes};

#[derive(Deserialize)]
struct Cases {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    key: String,
    dtype: Dtype,
    shape: Vec<u64>,
    encoding: Encoding,
    parts: Vec<CasePart>,
    expect: Option<String>,
    kitchen: Option<Options>,
}

#[derive(Deserialize)]
struct CasePart {
    role: String,
    dtype: Dtype,
    shape: Vec<u64>,
    bits: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Options {
    source: String,
    codebook: Option<bool>,
    symmetric: Option<bool>,
    f32_scales: Option<bool>,
    scale_search: Option<bool>,
}

fn b64(s: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD.decode(s).unwrap()
}

#[test]
fn encoders_reproduce_kitchen() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("conformance/vectors/cases/comfy.json");
    let cases: Cases = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    let mut checked = 0;
    for c in cases.cases {
        let Some(o) = &c.kitchen else {
            continue;
        };
        let values = b64(&o.source);
        let arrays = BTreeMap::from([(
            "values",
            PartBytes {
                dtype: &Dtype::F32,
                shape: &c.shape,
                bytes: &values,
            },
        )]);
        let input = Input {
            dtype: &c.dtype,
            shape: &c.shape,
            encoding: &c.encoding,
            arrays: &arrays,
        };
        let d = Kitchen::default();
        let k = Kitchen {
            codebook: o.codebook.unwrap_or(d.codebook),
            symmetric: o.symmetric.unwrap_or(d.symmetric),
            f32_scales: o.f32_scales.unwrap_or(d.f32_scales),
            scale_search: o.scale_search.unwrap_or(d.scale_search),
        };
        let parts = quantize_with(&input, &k).unwrap_or_else(|e| panic!("{}: {e}", c.key));
        let roles: Vec<&str> = parts.iter().map(|p| p.role.as_str()).collect();
        assert_eq!(parts.len(), c.parts.len(), "{}: {roles:?}", c.key);
        for (got, want) in parts.iter().zip(&c.parts) {
            let what = format!("{} {}", c.key, want.role);
            assert_eq!(
                (&got.role, &got.dtype, &got.shape),
                (&want.role, &want.dtype, &want.shape),
                "{what}"
            );
            let want = b64(&want.bits);
            let differ = got.bytes.iter().zip(&want).filter(|(a, b)| a != b).count();
            assert!(
                got.bytes == want,
                "{what}: {differ} of {} bytes differ",
                want.len()
            );
        }
        let view = parts
            .iter()
            .map(|p| {
                let part = PartBytes {
                    dtype: &p.dtype,
                    shape: &p.shape,
                    bytes: &p.bytes,
                };
                (p.role.as_str(), part)
            })
            .collect();
        let out = decode(&c.dtype, &c.shape, &c.encoding, &view).unwrap();
        assert_eq!(out, b64(c.expect.as_deref().unwrap()), "{} decodes", c.key);
        checked += 1;
    }
    assert_eq!(checked, 6, "kitchen cases");
}

#[test]
fn encoders_refuse_what_kitchen_refuses() {
    let values = vec![0u8; 4 * 8 * 48];
    let shape = [8u64, 48];
    let arrays = BTreeMap::from([(
        "values",
        PartBytes {
            dtype: &Dtype::F32,
            shape: &shape,
            bytes: &values,
        },
    )]);
    for encoding in [
        Encoding::comfy_asym_w4a8_int8(32, 16),
        Encoding::comfy_asym_w4a8_int8(12, 16),
        Encoding::comfy_w6a8_int8(16, 16),
        Encoding::comfy_convrot_w4a4(16),
        Encoding::comfy_convrot_w4a4(8),
    ] {
        let input = Input {
            dtype: &Dtype::BF16,
            shape: &shape,
            encoding: &encoding,
            arrays: &arrays,
        };
        assert!(quantize(&input).is_err(), "{encoding:?}");
    }
}

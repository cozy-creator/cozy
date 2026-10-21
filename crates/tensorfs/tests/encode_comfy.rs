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
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance/vectors/cases/comfy.json");
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
        Encoding::int4_convrot(16),
        Encoding::int4_convrot(8),
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

fn with_values(
    dtype: &Dtype,
    shape: &[u64],
    encoding: &Encoding,
    values: &[u8],
    options: &Kitchen,
) -> tensorfs::format::Result<Vec<tensorfs::format::encode::OwnedPart>> {
    let arrays = BTreeMap::from([(
        "values",
        PartBytes {
            dtype: &Dtype::F32,
            shape,
            bytes: values,
        },
    )]);
    quantize_with(
        &Input {
            dtype,
            shape,
            encoding,
            arrays: &arrays,
        },
        options,
    )
}

fn encodings() -> [Encoding; 3] {
    [
        Encoding::comfy_asym_w4a8_int8(16, 256),
        Encoding::comfy_w6a8_int8(16, 256),
        Encoding::int4_convrot(256),
    ]
}

fn ordinary_values() -> Vec<u8> {
    (0..256)
        .flat_map(|i| (((i % 17) as f32 - 8.0) / 32.0).to_le_bytes())
        .collect()
}

#[test]
fn malformed_and_empty_values_are_refused_before_rotation() {
    let valid = ordinary_values();
    for encoding in encodings() {
        let shape = [1, 256];
        assert!(with_values(&Dtype::F32, &shape, &encoding, &valid, &Kitchen::default()).is_ok());
        for bytes in [vec![], vec![0], vec![0; 4], vec![0; 1023], vec![0; 1028]] {
            let e = with_values(&Dtype::F32, &shape, &encoding, &bytes, &Kitchen::default())
                .unwrap_err();
            assert!(e.to_string().contains("values have"), "{encoding}: {e}");
        }
        for shape in [[0, 256], [1, 0], [0, 1 << 63]] {
            let e =
                with_values(&Dtype::F32, &shape, &encoding, &[], &Kitchen::default()).unwrap_err();
            assert!(e.to_string().contains("empty values"), "{encoding}: {e}");
        }
        let e = with_values(
            &Dtype::F32,
            &[u64::MAX, 256],
            &encoding,
            &[],
            &Kitchen::default(),
        )
        .unwrap_err();
        assert!(e.to_string().contains("byte count"), "{encoding}: {e}");
    }
}

#[test]
fn public_kitchen_dispatch_refuses_unknown_schemes_and_params() {
    let bytes = ordinary_values();
    let shape = [1, 256];
    for mut encoding in encodings() {
        encoding.params.insert("unread".into(), 1u64.into());
        let e =
            with_values(&Dtype::F32, &shape, &encoding, &bytes, &Kitchen::default()).unwrap_err();
        assert!(e.to_string().contains("unknown param"), "{encoding}: {e}");
    }
    // This is the unchanged corpus's w4a4-quant-group-size refusal: an import-only marker,
    // never an Encoding param.
    let mut encoding = Encoding::int4_convrot(256);
    encoding
        .params
        .insert("quant_group_size".into(), 64u64.into());
    assert!(with_values(&Dtype::F32, &shape, &encoding, &bytes, &Kitchen::default()).is_err());
    for scheme in ["comfy.typo", "fp8.tensorwise"] {
        let mut encoding = Encoding::comfy_asym_w4a8_int8(16, 256);
        encoding.scheme = scheme.into();
        assert!(matches!(
            with_values(&Dtype::F32, &shape, &encoding, &bytes, &Kitchen::default()),
            Err(tensorfs::format::Error::UnknownScheme(_))
        ));
    }
    for mut encoding in encodings() {
        encoding.params.remove("rotation_size");
        assert!(with_values(&Dtype::F32, &shape, &encoding, &bytes, &Kitchen::default()).is_err());
    }
}

#[test]
fn w6_cannot_emit_the_four_bit_correction_role() {
    let bytes = ordinary_values();
    let shape = [1, 256];
    let encoding = Encoding::comfy_w6a8_int8(16, 256);
    let parts = with_values(&Dtype::F32, &shape, &encoding, &bytes, &Kitchen::default()).unwrap();
    assert_eq!(
        parts.iter().map(|p| p.role.as_str()).collect::<Vec<_>>(),
        ["weight", "group_scale", "scale"]
    );
    let options = Kitchen {
        symmetric: false,
        ..Kitchen::default()
    };
    let e = with_values(&Dtype::F32, &shape, &encoding, &bytes, &options).unwrap_err();
    assert!(e.to_string().contains("no asymmetric correction"), "{e}");
}

#[test]
fn finite_values_that_overflow_rotation_are_refused() {
    let bytes = f32::MAX.to_le_bytes().repeat(256);
    for dtype in [Dtype::F32, Dtype::BF16, Dtype::F16] {
        for encoding in encodings() {
            let e =
                with_values(&dtype, &[1, 256], &encoding, &bytes, &Kitchen::default()).unwrap_err();
            assert!(
                e.to_string().contains("not finite"),
                "{dtype} {encoding}: {e}"
            );
        }
    }
}

//! `encode::pack` against the producers: every case in `conformance/vectors/cases` that carries
//! `pack` (the codes and zeros a real producer packed, `I16`) comes out as exactly the case's
//! parts. And the int schemes' round to nearest decodes back within its grid.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use base64::Engine;
use serde::Deserialize;
use tensorfs_core::format::encode::{pack, quantize, Input, OwnedPart};
use tensorfs_core::format::{decode, Dtype, Encoding, PartBytes};

#[derive(Deserialize)]
struct Cases {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    key: String,
    dtype: Dtype,
    shape: Vec<u64>,
    encoding: Option<Encoding>,
    #[serde(default)]
    parts: Vec<CasePart>,
    pack: Option<BTreeMap<String, String>>,
}

#[derive(Deserialize)]
struct CasePart {
    role: String,
    dtype: Dtype,
    shape: Vec<u64>,
    bits: String,
}

fn b64(s: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD.decode(s).unwrap()
}

/// The producers' role for each `pack` input that is a part as is.
fn input_name(role: &str) -> Option<&'static str> {
    match role {
        "scales" | "weight_scale" => Some("scales"),
        "g_idx" | "weight_g_idx" => Some("g_idx"),
        _ => None,
    }
}

#[test]
fn pack_reproduces_the_producers_parts() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tensorfs/conformance/vectors/cases");
    let mut checked = 0;
    for entry in fs::read_dir(dir).unwrap() {
        let cases: Cases =
            serde_json::from_slice(&fs::read(entry.unwrap().path()).unwrap()).unwrap();
        for c in cases.cases {
            let (Some(arrays), Some(encoding)) = (&c.pack, &c.encoding) else {
                continue;
            };
            let raw: Vec<(&str, Vec<u8>)> = (arrays.iter())
                .map(|(k, v)| (k.as_str(), b64(v)))
                .chain(
                    c.parts
                        .iter()
                        .filter_map(|p| Some((input_name(&p.role)?, b64(&p.bits)))),
                )
                .collect();
            let shapes: Vec<Vec<u64>> = (raw.iter())
                .map(|(name, bytes)| {
                    match c.parts.iter().find(|p| input_name(&p.role) == Some(name)) {
                        Some(p) => p.shape.clone(),
                        None => vec![bytes.len() as u64 / 2],
                    }
                })
                .collect();
            let dtypes: Vec<Dtype> = (raw.iter())
                .map(
                    |(name, _)| match c.parts.iter().find(|p| input_name(&p.role) == Some(name)) {
                        Some(p) => p.dtype.clone(),
                        None => Dtype::I16,
                    },
                )
                .collect();
            let view: BTreeMap<&str, PartBytes> = (raw.iter().zip(&shapes).zip(&dtypes))
                .map(|(((name, bytes), shape), dtype)| {
                    (
                        *name,
                        PartBytes {
                            dtype,
                            shape,
                            bytes,
                        },
                    )
                })
                .collect();
            let input = Input {
                dtype: &c.dtype,
                shape: &c.shape,
                encoding,
                arrays: &view,
            };
            let parts = pack(&input).unwrap_or_else(|e| panic!("{}: {e}", c.key));
            let want: Vec<OwnedPart> = (c.parts.iter())
                .map(|p| OwnedPart {
                    role: p.role.clone(),
                    dtype: p.dtype.clone(),
                    shape: p.shape.clone(),
                    bytes: b64(&p.bits),
                })
                .collect();
            assert_eq!(parts, want, "{}", c.key);
            checked += 1;
        }
    }
    assert!(checked >= 30, "{checked} producer cases");
}

/// A deterministic spread of values: a few magnitudes per row, one outlier, one all-zero group.
fn values(n: usize, k: usize) -> Vec<f32> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut x: Vec<f32> = (0..n * k)
        .map(|t| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            u * 0.05 * (1.0 + (t % k) as f32 / k as f32)
        })
        .collect();
    x[3] = 0.4;
    x[k..k + 32].fill(0.0);
    x
}

/// Relative L2 error of `quantize` then `decode`, in F32.
fn round_trip(encoding: &Encoding, n: usize, k: usize) -> Result<f64, String> {
    let x = values(n, k);
    let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let shape = vec![n as u64, k as u64];
    let arrays = BTreeMap::from([(
        "values",
        PartBytes {
            dtype: &Dtype::F32,
            shape: &shape,
            bytes: &bytes,
        },
    )]);
    let input = Input {
        dtype: &Dtype::F32,
        shape: &shape,
        encoding,
        arrays: &arrays,
    };
    let parts = quantize(&input).map_err(|e| e.to_string())?;
    let view = (parts.iter())
        .map(|p| {
            let b = PartBytes {
                dtype: &p.dtype,
                shape: &p.shape,
                bytes: &p.bytes,
            };
            (p.role.as_str(), b)
        })
        .collect();
    let out = decode(&Dtype::F32, &shape, encoding, &view).map_err(|e| e.to_string())?;
    let y = out
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b));
    let (err, norm) = y.zip(&x).fold((0f64, 0f64), |(e, s), (y, x)| {
        (e + ((y - x) as f64).powi(2), s + (*x as f64).powi(2))
    });
    Ok((err / norm).sqrt())
}

/// Uniform rounding errs by about `step/√12` against data of `range/√12`: a relative error near
/// `1/(2^b − 1)`; this allows 2.5 times that.
fn limit(bits: u32) -> f64 {
    2.5 / ((1u32 << bits) - 1) as f64
}

#[test]
fn round_to_nearest_decodes_within_its_grid() {
    for bits in [2, 3, 4, 8] {
        for (sym, offset) in [(false, 1), (true, 0)] {
            let e = Encoding::gptq(bits, 64, sym, false, offset);
            let err = round_trip(&e, 16, 192).unwrap();
            assert!(err < limit(bits), "{e}: {err}");
        }
        let whole = Encoding::gptq(bits, -1, false, false, 0);
        assert!(round_trip(&whole, 8, 128).unwrap() < limit(bits), "{whole}");
    }
    let awq = Encoding::awq_int4(64);
    assert!(round_trip(&awq, 16, 192).unwrap() < limit(4));
    for bits in 1..=8 {
        for (sym, packing) in [(true, "dense"), (false, "word")] {
            let e = Encoding::ct(bits, 32, "group", sym, packing);
            let err = round_trip(&e, 8, 96).unwrap();
            assert!(err < limit(bits), "{e}: {err}");
        }
    }
    let channel = Encoding::ct(4, 96, "channel", false, "dense");
    assert!(round_trip(&channel, 8, 96).unwrap() < limit(4), "{channel}");
    let act_order = Encoding::gptq(4, 64, false, true, 1);
    assert!(round_trip(&act_order, 4, 64)
        .unwrap_err()
        .contains("calibration"));
}

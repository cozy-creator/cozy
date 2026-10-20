//! Closed-form encoders against their producers: every case in `conformance/vectors/cases` that
//! carries `source` (the F32 values a real producer quantized) must come out of
//! `encode::quantize` as exactly the case's parts, and decode back to its expected bytes.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use base64::Engine;
use serde::Deserialize;
use tensorfs::format::encode::{quantize, quantize_fp8_blockwise, Input};
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
    encoding: Option<Encoding>,
    #[serde(default)]
    parts: Vec<CasePart>,
    expect: Option<String>,
    source: Option<String>,
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

#[test]
fn encoders_reproduce_their_producers() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance/vectors/cases");
    let mut checked = 0;
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let cases: Cases = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        for c in cases.cases {
            let (Some(source), Some(encoding)) = (&c.source, &c.encoding) else {
                continue;
            };
            let values = b64(source);
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
                encoding,
                arrays: &arrays,
            };
            // fp8.blockwise's producer is the scale dtype's: the case's scale part names it.
            let parts = match encoding.scheme.as_str() {
                "fp8.blockwise" => quantize_fp8_blockwise(&input, &c.parts[1].dtype),
                _ => quantize(&input),
            };
            let parts = parts.unwrap_or_else(|e| panic!("{}: {e}", c.key));
            assert_eq!(parts.len(), c.parts.len(), "{}", c.key);
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
            let bytes: Vec<Vec<u8>> = parts.iter().map(|p| p.bytes.clone()).collect();
            let view = (parts.iter().zip(&bytes))
                .map(|(p, b)| {
                    let part = PartBytes {
                        dtype: &p.dtype,
                        shape: &p.shape,
                        bytes: b,
                    };
                    (p.role.as_str(), part)
                })
                .collect();
            let out = decode(&c.dtype, &c.shape, encoding, &view).unwrap();
            assert_eq!(out, b64(c.expect.as_deref().unwrap()), "{} decodes", c.key);
            checked += 1;
        }
    }
    assert!(checked >= 8, "{checked} producer cases");
}

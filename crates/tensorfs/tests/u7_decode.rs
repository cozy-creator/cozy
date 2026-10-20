//! Frozen U7 producer/spec values and refusals, without derived store-fixture regeneration.

use std::collections::BTreeMap;
use std::{fs, path::Path};

use base64::Engine;
use serde::Deserialize;
use tensorfs::format::{decode, Dtype, Encoding, Error, PartBytes};

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
    parts: Vec<Part>,
    expect: Option<String>,
    refuse: Option<String>,
}

#[derive(Deserialize)]
struct Part {
    role: String,
    dtype: Dtype,
    shape: Vec<u64>,
    bits: String,
}

fn b64(s: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD.decode(s).unwrap()
}

#[test]
fn u7_golden_values_and_refusals_match_the_frozen_corpus() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance/vectors/cases/comfy.json");
    let cases: Cases = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    let (mut values, mut refusals) = (0, 0);
    for c in cases.cases {
        let bytes: Vec<_> = c.parts.iter().map(|p| b64(&p.bits)).collect();
        let parts: BTreeMap<_, _> = c
            .parts
            .iter()
            .zip(&bytes)
            .map(|(p, bytes)| {
                (
                    p.role.as_str(),
                    PartBytes {
                        dtype: &p.dtype,
                        shape: &p.shape,
                        bytes,
                    },
                )
            })
            .collect();
        let result = decode(&c.dtype, &c.shape, &c.encoding, &parts);
        match (c.expect, c.refuse) {
            (Some(expected), None) => {
                assert_eq!(
                    result.unwrap_or_else(|e| panic!("{}: {e}", c.key)),
                    b64(&expected),
                    "{}",
                    c.key
                );
                values += 1;
            }
            (None, Some(code)) => {
                assert_eq!(code, "decode", "{}", c.key);
                assert!(
                    matches!(result, Err(Error::Decode(_))),
                    "{}: {result:?}",
                    c.key
                );
                refusals += 1;
            }
            _ => panic!("{} has no unambiguous golden result", c.key),
        }
    }
    assert_eq!((values, refusals), (18, 14));
}

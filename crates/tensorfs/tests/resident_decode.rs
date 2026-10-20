//! Existing frozen scheme bytes through the store-free resident-parts entrypoint.

use base64::Engine;
use serde::Deserialize;
use tensorfs::Kind;
use tensorfs::format::{Dtype, Encoding, decode::PartMetadata};
use tensorfs::io::{
    Dst, Src,
    decode::{self, Output, Part},
};

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
    parts: Vec<Stored>,
    expect: Option<String>,
    refuse: Option<String>,
}
#[derive(Deserialize)]
struct Stored {
    role: String,
    dtype: Dtype,
    shape: Vec<u64>,
    bits: String,
}
fn b64(s: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD.decode(s).unwrap()
}
fn cases() -> Vec<Case> {
    serde_json::from_slice::<Cases>(include_bytes!("../conformance/vectors/cases/comfy.json"))
        .unwrap()
        .cases
}

#[test]
fn resident_host_matches_all_frozen_u7_values_and_refuses_without_writing() {
    let (mut passed, mut refused) = (0, 0);
    for case in cases() {
        let data: Vec<_> = case.parts.iter().map(|p| b64(&p.bits)).collect();
        let parts: Vec<_> = case
            .parts
            .iter()
            .zip(&data)
            .map(|(p, data)| Part {
                role: &p.role,
                dtype: &p.dtype,
                shape: &p.shape,
                src: Src::Host(data.as_ptr()),
                len: data.len(),
            })
            .collect();
        let len = case.dtype.byte_len(&case.shape).unwrap() as usize;
        let mut output = vec![0xA5; len];
        let result = unsafe {
            decode::into(
                &case.encoding,
                &case.dtype,
                &case.shape,
                &parts,
                Output {
                    dtype: &case.dtype,
                    shape: &case.shape,
                    dst: Dst::host(&mut output),
                },
            )
        };
        if let Some(expected) = case.expect {
            result.unwrap_or_else(|error| panic!("{}: {error}", case.key));
            assert_eq!(output, b64(&expected), "{}", case.key);
            passed += 1;
        } else {
            assert_eq!(case.refuse.as_deref(), Some("decode"), "{}", case.key);
            assert_eq!(result.unwrap_err().kind, Kind::Decode, "{}", case.key);
            assert!(
                output.iter().all(|b| *b == 0xA5),
                "{} wrote output before refusal",
                case.key
            );
            refused += 1;
        }
    }
    assert_eq!((passed, refused), (18, 14));
}

#[test]
fn metadata_lengths_duplicates_output_and_overlap_are_checked_before_write() {
    let case = cases().into_iter().find(|c| c.expect.is_some()).unwrap();
    let data: Vec<_> = case.parts.iter().map(|p| b64(&p.bits)).collect();
    let metadata: Vec<_> = case
        .parts
        .iter()
        .zip(&data)
        .map(|(p, data)| PartMetadata {
            role: &p.role,
            dtype: &p.dtype,
            shape: &p.shape,
            len: data.len(),
        })
        .collect();
    decode::check(&case.encoding, &case.dtype, &case.shape, &metadata).unwrap();
    let mut short = metadata.clone();
    short[0].len -= 1;
    assert_eq!(
        decode::check(&case.encoding, &case.dtype, &case.shape, &short)
            .unwrap_err()
            .kind,
        Kind::Decode
    );
    let mut repeated = metadata;
    repeated.push(repeated[0]);
    assert_eq!(
        decode::check(&case.encoding, &case.dtype, &case.shape, &repeated)
            .unwrap_err()
            .kind,
        Kind::Decode
    );

    let parts: Vec<_> = case
        .parts
        .iter()
        .zip(&data)
        .map(|(p, data)| Part {
            role: &p.role,
            dtype: &p.dtype,
            shape: &p.shape,
            src: Src::Host(data.as_ptr()),
            len: data.len(),
        })
        .collect();
    let mut output = vec![0xA5; case.dtype.byte_len(&case.shape).unwrap() as usize];
    let other = if case.dtype == Dtype::F32 {
        Dtype::F16
    } else {
        Dtype::F32
    };
    let error = unsafe {
        decode::into(
            &case.encoding,
            &case.dtype,
            &case.shape,
            &parts,
            Output {
                dtype: &other,
                shape: &case.shape,
                dst: Dst::host(&mut output),
            },
        )
    }
    .unwrap_err();
    assert_eq!(error.kind, Kind::Invalid);
    assert!(output.iter().all(|b| *b == 0xA5));

    // The declared input span is a real allocation larger than the output. Aliasing is
    // rejected before any part is read, even though these intentionally arbitrary values
    // would not match the golden tensor. Input bytes must remain exactly unchanged.
    let mut storage = vec![0x5A; output.len().max(data[0].len())];
    let mut aliases = parts;
    aliases[0].src = Src::Host(storage.as_ptr());
    let before = storage.clone();
    let error = unsafe {
        decode::into(
            &case.encoding,
            &case.dtype,
            &case.shape,
            &aliases,
            Output {
                dtype: &case.dtype,
                shape: &case.shape,
                dst: Dst::Host {
                    ptr: storage.as_mut_ptr(),
                    len: output.len(),
                },
            },
        )
    }
    .unwrap_err();
    assert_eq!(error.kind, Kind::Invalid);
    assert_eq!(storage, before);
}

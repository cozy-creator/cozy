//! cozytensors/1 documents through the public reader and writer.

use std::collections::BTreeMap;

use serde_json::json;
use tensorfs_core::format::{
    Body, Component, Dtype, Encoding, Header, Location, Manifest, ObjectRef, Part, Release, Repo,
    Tensor, Value, HEADER_PATH,
};

fn tensors() -> Vec<Tensor> {
    let plain = Tensor {
        key: "a.weight".into(),
        dtype: Dtype::BF16,
        shape: vec![2, 3],
        body: Body::Data(Location {
            chunk: 0,
            offset: 0,
            length: 12,
        }),
    };
    let part = |role: &str, dtype, shape: Vec<u64>, chunk, offset, length| Part {
        role: role.into(),
        dtype,
        shape,
        chunk,
        offset,
        length,
    };
    let encoded = Tensor {
        key: "b.weight".into(),
        dtype: Dtype::F32,
        shape: vec![4, 1024],
        body: Body::Encoded {
            encoding: Encoding::fp8_rowwise(0),
            parts: vec![
                part("weight", Dtype::F8_E4M3, vec![4, 1024], 0, 4096, 4096),
                part("scale", Dtype::F32, vec![4], 1, 0, 16),
            ],
        },
    };
    vec![plain, encoded]
}

fn header_with(tensors: Vec<Tensor>) -> Header {
    let chunks = vec![ObjectRef::of(&[7u8; 8192]), ObjectRef::of(&[1u8; 100])];
    let unet = Component {
        name: "unet".into(),
        tensors,
    };
    let prompt = Value::Map(
        [
            ("rows".into(), Value::List(vec!["a cat".into()])),
            ("n".into(), 1u64.into()),
        ]
        .into(),
    );
    Header::new(chunks, vec![unet], [("prompt".into(), prompt)].into()).unwrap()
}

fn header() -> Header {
    header_with(tensors())
}

#[test]
fn writer_is_deterministic_and_reader_round_trips() {
    let h = header();
    let bytes = h.to_cbor().unwrap();
    // [format, chunks, encodings, components, metadata]: a 5-array, then the format text.
    assert_eq!(&bytes[..15], b"\x85\x6dcozytensors/1");
    // The first chunk: [bstr .size 32, 8192].
    assert_eq!(&bytes[15..19], b"\x82\x82\x58\x20");
    assert_eq!(&bytes[51..54], b"\x19\x20\x00");
    assert!(bytes.ends_with(b"\xa1\x66prompt\xa2\x61n\x01\x64rows\x81\x65a cat"));
    assert_eq!(Header::from_cbor(&bytes).unwrap(), h);
    assert_eq!(Header::from_cbor(&bytes).unwrap().to_cbor().unwrap(), bytes);

    let mut wide = h.clone();
    wide.metadata.insert("max".into(), Value::from(u64::MAX));
    wide.metadata.insert("min".into(), Value::from(i64::MIN));
    let wb = wide.to_cbor().unwrap();
    let tail = [b"\x63max\x1b".as_slice(), &[0xff; 8], b"\x63min\x3b\x7f", &[0xff; 7]].concat();
    assert!(wb.windows(tail.len()).any(|w| w == tail));
    assert_eq!(Header::from_cbor(&wb).unwrap(), wide);
    wide.metadata.insert("huge".into(), Value::Int(1 << 64));
    assert_eq!(wide.to_cbor().unwrap_err().code(), "invalid");

    let m = Manifest {
        files: BTreeMap::from([(HEADER_PATH.into(), ObjectRef::of(&bytes))]),
        metadata: Default::default(),
    };
    let mb = m.to_json().unwrap();
    assert!(mb.ends_with(br#"}},"format":"tensorfs.manifest/1"}"#));
    assert_eq!(Manifest::from_json(&mb).unwrap().header(), m.header());

    let r = Repo {
        name: "Acme/Tiny".into(),
        releases: vec![Release {
            version: "1.0".into(),
            lanes: BTreeMap::from([("bf16".into(), ObjectRef::of(&mb))]),
            ..Default::default()
        }],
        ..Default::default()
    };
    let rb = r.to_json().unwrap();
    assert_eq!(Repo::from_json(&rb).unwrap(), r);
    assert_eq!(r.key(), "refs/acme/tiny.json");
    assert_eq!(r.lane(None, "bf16"), Some(&ObjectRef::of(&mb)));
}

/// Later versions append fields to the root and to records, and add dtypes and schemes: this
/// reader skips the fields and reads the tensors raw.
#[test]
fn appended_fields_newer_dtypes_and_schemes_read() {
    let h = header();
    let good = h.to_cbor().unwrap();
    let digest = &good[19..51];
    let chunk = [b"\x82\x58\x20".as_slice(), digest, b"\x19\x20\x00"].concat();
    let longer = [b"\x84\x58\x20".as_slice(), digest, b"\x19\x20\x00\x41\x00\xa1\x61k\x80"].concat();
    let mut newer = patch(&good, b"\x85\x6dcozytensors/1", b"\x86\x6fcozytensors/1.9");
    newer = patch(&newer, &chunk, &longer);
    newer = patch(&newer, b"\x64BF16", b"\x67F3_E1M1");
    newer = patch(&newer, b"\x6bfp8.rowwise", b"\x6bacme.int3x2");
    newer.extend(b"\x82\x01\x62hi");
    let read = Header::from_cbor(&newer).unwrap();
    let a = read.tensor("unet", "a.weight").unwrap();
    assert_eq!(a.dtype, Dtype::Other("F3_E1M1".into()));
    assert_eq!(a.body, Body::Data(Location { chunk: 0, offset: 0, length: 12 }));
    assert_eq!(read.chunks, h.chunks);
    let b = read.tensor("unet", "b.weight").unwrap();
    let Body::Encoded { encoding, .. } = &b.body else {
        panic!()
    };
    assert_eq!(encoding.scheme, "acme.int3x2");
    let chunk = |i: usize| Ok(vec![i as u8; read.chunks[i].length as usize]);
    assert_eq!(read.read("unet", "b.weight", chunk).unwrap_err().code(), "unknown_scheme");
    let again = Header::from_cbor(&read.to_cbor().unwrap()).unwrap();
    assert_eq!(again, read, "this writer drops the fields it does not know");
}

/// `bytes` with `from` (found once) replaced by `to`.
fn patch(bytes: &[u8], from: &[u8], to: &[u8]) -> Vec<u8> {
    let at = (bytes.windows(from.len()).position(|w| w == from)).expect("pattern present");
    [&bytes[..at], to, &bytes[at + from.len()..]].concat()
}

#[test]
fn red_documents_are_refused() {
    let good = header().to_cbor().unwrap();
    let cases: &[(&str, Vec<u8>, &str)] = &[
        ("major 2", patch(&good, b"cozytensors/1", b"cozytensors/2"), "format"),
        ("minor empty", patch(&good, b"\x6dcozytensors/1", b"\x6ecozytensors/1."), "format"),
        ("trailing byte", [good.as_slice(), &[0]].concat(), "invalid"),
        ("truncated", good[..good.len() - 1].to_vec(), "invalid"),
        ("non-preferred length", patch(&good, b"\x19\x20\x00", b"\x1a\x00\x00\x20\x00"), "invalid"),
        ("non-preferred small int", patch(&good, b"\x61n\x01", b"\x61n\x18\x01"), "invalid"),
        ("indefinite root", patch(&good, b"\x85\x6d", b"\x9f\x6d"), "invalid"),
        ("tag", patch(&good, b"\x61n\x01", b"\x61n\xc1\x01"), "invalid"),
        ("float", patch(&good, b"\x61n\x01", b"\x61n\xf9\x3c\x00"), "invalid"),
        ("null", patch(&good, b"\x61n\x01", b"\x61n\xf6"), "invalid"),
        ("undefined", patch(&good, b"\x61n\x01", b"\x61n\xf7"), "invalid"),
        ("byte string value", patch(&good, b"\x61n\x01", b"\x61n\x41\x01"), "invalid"),
        ("keys out of order", patch(&good, b"\x61n\x01\x64rows\x81\x65a cat", b"\x64rows\x81\x65a cat\x61n\x01"), "invalid"),
        ("repeated key", patch(&good, b"\xa2\x61n\x01\x64rows\x81\x65a cat", b"\xa2\x61n\x01\x61n\x02"), "invalid"),
        ("31-byte digest", patch(&good, b"\x58\x20", b"\x58\x1f"), "invalid"),
        ("empty dtype", patch(&good, b"\x64BF16", b"\x60"), "invalid"),
        ("short record", patch(&good, b"\x82\x82\x58\x20", b"\x82\x81\x58\x20"), "invalid"),
        ("bad trailing field", [patch(&good, b"\x85\x6d", b"\x86\x6d"), vec![0xf6]].concat(), "invalid"),
        ("component name", patch(&good, b"\x64unet", b"\x64un/t"), "invalid"),
        ("repeated tensor key", patch(&good, b"b.weight", b"a.weight"), "invalid"),
    ];
    for (name, bad, code) in cases {
        let err = Header::from_cbor(bad).expect_err(name);
        assert_eq!(err.code(), *code, "{name}: {err}");
    }

    for (name, tensors) in [
        ("length", tweak(|t| t[0].body = Body::Data(Location { chunk: 0, offset: 0, length: 14 }))),
        ("bounds", tweak(|t| t[0].body = Body::Data(Location { chunk: 1, offset: 4096, length: 12 }))),
        ("chunk index", tweak(|t| t[0].body = Body::Data(Location { chunk: 2, offset: 0, length: 12 }))),
        ("unaligned", tweak(|t| t[0].body = Body::Data(Location { chunk: 0, offset: 12, length: 12 }))),
        ("repeated role", tweak(|t| set_role(&mut t[1], 1, "weight"))),
        ("empty role", tweak(|t| set_role(&mut t[1], 1, ""))),
    ] {
        let chunks = header().chunks;
        let unet = Component { name: "unet".into(), tensors };
        let err = Header::new(chunks, vec![unet], Default::default()).expect_err(name);
        assert_eq!(err.code(), "invalid", "{name}: {err}");
    }

    for path in ["../x", "/x", "a//b", "a/./b", ""] {
        let doc = json!({"format": "tensorfs.manifest/1", "files": {path: {"sha256": "0".repeat(64), "length": 0}}});
        assert!(
            Manifest::from_json(doc.to_string().as_bytes()).is_err(),
            "{path:?}"
        );
    }
    let repo = json!({"format": "tensorfs.repo/1", "name": "acme/tiny", "releases": [
        {"version": "1", "lanes": {}}, {"version": "1", "lanes": {}}]});
    assert!(Repo::from_json(repo.to_string().as_bytes()).is_err());
}

fn tweak(f: impl FnOnce(&mut Vec<Tensor>)) -> Vec<Tensor> {
    let mut t = tensors();
    f(&mut t);
    t
}

fn set_role(t: &mut Tensor, i: usize, role: &str) {
    let Body::Encoded { parts, .. } = &mut t.body else {
        unreachable!()
    };
    parts[i].role = role.into();
}

#[test]
fn spans_continue_into_following_chunks() {
    let chunks = vec![
        ObjectRef::of(&[0; 10]),
        ObjectRef::of(&[1; 10]),
        ObjectRef::of(&[2; 10]),
    ];
    let h = Header::new(chunks, vec![], Default::default()).unwrap();
    let at = Location {
        chunk: 0,
        offset: 6,
        length: 20,
    };
    let spans: Vec<_> = h
        .spans(&at)
        .into_iter()
        .map(|s| (s.chunk, s.start, s.length))
        .collect();
    assert_eq!(spans, [(0, 6, 4), (1, 0, 10), (2, 0, 6)]);
    let chunk = |i: usize| Ok(vec![i as u8; 10]);
    assert_eq!(
        h.bytes(&at, chunk).unwrap(),
        [[0; 4].as_slice(), &[1; 10], &[2; 6]].concat()
    );
    assert_eq!(Dtype::F4.byte_len(&[3]), None);
    assert_eq!(Dtype::F4.byte_len(&[16, 64]), Some(512));
    assert_eq!(Dtype::known().count(), 22);
    assert!(Dtype::known().all(|d| Dtype::parse(d.name()) == d));
}

/// Construction order is data: it survives a round trip and the digest covers it.
#[test]
fn order_is_kept_and_a_reorder_changes_the_digest() {
    let h = header();
    let keys = |h: &Header| h.tensors().map(|(_, t)| t.key.clone()).collect::<Vec<_>>();
    assert_eq!(keys(&h), ["a.weight", "b.weight"]);
    let back = Header::from_cbor(&h.to_cbor().unwrap()).unwrap();
    assert_eq!(keys(&back), ["a.weight", "b.weight"]);

    let mut reordered = tensors();
    reordered.reverse();
    let r = header_with(reordered);
    assert_eq!(
        keys(&Header::from_cbor(&r.to_cbor().unwrap()).unwrap()),
        ["b.weight", "a.weight"]
    );
    assert_ne!(
        ObjectRef::of(&r.to_cbor().unwrap()),
        ObjectRef::of(&h.to_cbor().unwrap()),
        "reordering two tensors changes the header digest"
    );
    assert_ne!(r, h);

    let mut swapped = tensors();
    let Body::Encoded { parts, .. } = &mut swapped[1].body else {
        unreachable!()
    };
    parts.reverse();
    assert_ne!(
        ObjectRef::of(&header_with(swapped).to_cbor().unwrap()),
        ObjectRef::of(&h.to_cbor().unwrap()),
        "reordering two parts changes it too"
    );
}

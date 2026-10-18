//! cozytensors/2 documents through the public reader and writer.

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
            encoding: Encoding {
                scheme: "fp8-rowwise".into(),
                params: [("axis".into(), Value::from(0u64))].into(),
            },
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
    let bytes = h.to_json().unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    assert!(
        text.starts_with(r#"{"chunks":[{"length":8192,"sha256":""#),
        "{text}"
    );
    assert!(text
        .contains(r#""format":"cozytensors/2","metadata":{"prompt":{"n":1,"rows":["a cat"]}}}"#));
    assert!(!text.contains(' ') || text.contains("a cat") && text.matches(' ').count() == 1);
    assert_eq!(Header::from_json(&bytes).unwrap(), h);

    let mut wide = h.clone();
    wide.metadata.insert("max".into(), Value::from(u64::MAX));
    wide.metadata.insert("min".into(), Value::from(i64::MIN));
    wide.metadata.insert("none".into(), Value::Null);
    let wb = wide.to_json().unwrap();
    assert!(std::str::from_utf8(&wb)
        .unwrap()
        .contains(r#""max":18446744073709551615,"min":-9223372036854775808,"none":null"#));
    assert_eq!(Header::from_json(&wb).unwrap(), wide);

    let m = Manifest {
        files: BTreeMap::from([(HEADER_PATH.into(), ObjectRef::of(&bytes))]),
        metadata: Default::default(),
    };
    let mb = m.to_json().unwrap();
    assert!(mb.ends_with(br#"}},"format":"tensorfs.manifest/2"}"#));
    assert_eq!(Manifest::from_json(&mb).unwrap().header(), m.header());

    let r = Repo {
        name: "Acme/Tiny".into(),
        releases: vec![Release {
            version: "1.0".into(),
            lanes: BTreeMap::from([("bf16".into(), ObjectRef::of(&mb))]),
        }],
    };
    let rb = r.to_json().unwrap();
    assert_eq!(Repo::from_json(&rb).unwrap(), r);
    assert_eq!(r.key(), "refs/acme/tiny.json");
    assert_eq!(r.lane(None, "bf16"), Some(&ObjectRef::of(&mb)));
}

#[test]
fn unknown_fields_and_minor_versions_read() {
    let mut v: serde_json::Value = serde_json::from_slice(&header().to_json().unwrap()).unwrap();
    v["format"] = json!("cozytensors/2.9");
    v["future"] = json!({"x": [1, 2]});
    let unet = &mut v["components"][0];
    unet["future"] = json!("y");
    unet["tensors"][0]["future"] = json!(true);
    unet["tensors"][0]["dtype"] = json!("F3_E1M1");
    unet["tensors"][1]["encoding"]["scheme"] = json!("acme-int3");
    unet["tensors"][1]["parts"][0]["future"] = json!([]);
    let pretty = serde_json::to_vec_pretty(&v).unwrap();
    let h = Header::from_json(&pretty).unwrap();
    let Body::Encoded { encoding, .. } = &h.tensor("unet", "b.weight").unwrap().body else {
        panic!()
    };
    assert_eq!(encoding.scheme, "acme-int3");
    let a = h.tensor("unet", "a.weight").unwrap();
    assert_eq!(a.dtype, Dtype::Other("F3_E1M1".into()));
    let again = Header::from_json(&h.to_json().unwrap()).unwrap();
    assert_eq!(again.tensor("unet", "a.weight"), Some(a));
}

#[test]
fn red_documents_are_refused() {
    let good = String::from_utf8(header().to_json().unwrap()).unwrap();
    let cases: &[(&str, &str, &str, &str)] = &[
        (
            "major 3",
            r#""format":"cozytensors/2""#,
            r#""format":"cozytensors/3""#,
            "format",
        ),
        (
            "other kind",
            r#""format":"cozytensors/2""#,
            r#""format":"tensorfs.manifest/2""#,
            "format",
        ),
        ("float", r#""length":12"#, r#""length":12.0"#, "invalid"),
        (
            "duplicate",
            r#""length":12"#,
            r#""length":12,"length":12"#,
            "invalid",
        ),
        ("nested duplicate", r#""n":1"#, r#""n":1,"n":2"#, "invalid"),
        ("length", r#""length":12"#, r#""length":14"#, "invalid"),
        (
            "empty dtype",
            r#""dtype":"BF16""#,
            r#""dtype":"""#,
            "invalid",
        ),
        (
            "bounds",
            r#""chunk":1,"dtype":"F32","length":16,"offset":0"#,
            r#""chunk":1,"dtype":"F32","length":16,"offset":90"#,
            "invalid",
        ),
        (
            "chunk index",
            r#""chunk":1,"dtype":"F32""#,
            r#""chunk":2,"dtype":"F32""#,
            "invalid",
        ),
        (
            "component",
            r#""name":"unet""#,
            r#""name":"un/et""#,
            "invalid",
        ),
        (
            "repeated key",
            r#""key":"b.weight""#,
            r#""key":"a.weight""#,
            "invalid",
        ),
        (
            "repeated role",
            r#""role":"scale""#,
            r#""role":"weight""#,
            "invalid",
        ),
        (
            "components as a map",
            r#""components":[{"name":"unet","tensors":"#,
            r#""components":{"unet":{"tensors":"#,
            "invalid",
        ),
        (
            "huge",
            r#""length":12"#,
            r#""length":18446744073709551616"#,
            "invalid",
        ),
    ];
    for (name, from, to, code) in cases {
        assert!(good.contains(from), "{name}: {from} not in {good}");
        let bad = good.replacen(from, to, 1);
        let err = Header::from_json(bad.as_bytes()).expect_err(name);
        assert_eq!(err.code(), *code, "{name}: {err}");
    }
    let digest = good.replacen(
        &header().chunks[0].sha256,
        &header().chunks[0].sha256.to_uppercase(),
        1,
    );
    assert_eq!(
        Header::from_json(digest.as_bytes()).unwrap_err().code(),
        "invalid"
    );

    for path in ["../x", "/x", "a//b", "a/./b", ""] {
        let doc = json!({"format": "tensorfs.manifest/2", "files": {path: {"sha256": "0".repeat(64), "length": 0}}});
        assert!(
            Manifest::from_json(doc.to_string().as_bytes()).is_err(),
            "{path:?}"
        );
    }
    let repo = json!({"format": "tensorfs.repo/2", "name": "acme/tiny", "releases": [
        {"version": "1", "lanes": {}}, {"version": "1", "lanes": {}}]});
    assert!(Repo::from_json(repo.to_string().as_bytes()).is_err());
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
    let future = Dtype::parse("F3_E1M1");
    assert_eq!(
        (future.name(), future.bits(), future.byte_len(&[8])),
        ("F3_E1M1", None, None)
    );
}

/// Construction order is data: it survives a round trip and the digest covers it.
#[test]
fn order_is_kept_and_a_reorder_changes_the_digest() {
    let h = header();
    let keys = |h: &Header| h.tensors().map(|(_, t)| t.key.clone()).collect::<Vec<_>>();
    assert_eq!(keys(&h), ["a.weight", "b.weight"]);
    let back = Header::from_json(&h.to_json().unwrap()).unwrap();
    assert_eq!(keys(&back), ["a.weight", "b.weight"]);

    let mut reordered = tensors();
    reordered.reverse();
    let r = header_with(reordered);
    assert_eq!(
        keys(&Header::from_json(&r.to_json().unwrap()).unwrap()),
        ["b.weight", "a.weight"]
    );
    assert_ne!(
        ObjectRef::of(&r.to_json().unwrap()),
        ObjectRef::of(&h.to_json().unwrap()),
        "reordering two tensors changes the header digest"
    );
    assert_ne!(r, h);

    let mut swapped = tensors();
    let Body::Encoded { parts, .. } = &mut swapped[1].body else {
        unreachable!()
    };
    parts.reverse();
    assert_ne!(
        ObjectRef::of(&header_with(swapped).to_json().unwrap()),
        ObjectRef::of(&h.to_json().unwrap()),
        "reordering two parts changes it too"
    );
}

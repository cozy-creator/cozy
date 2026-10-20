//! Validation at the public document boundary, including fields the typed reader ignores.

use serde_json::json;
use tensorfs::format::{
    Body, Component, Config, Dtype, Encoding, Header, Location, ObjectRef, Part, Pipeline, Repo,
    Result, Tensor, Tree,
};

#[test]
fn tree_and_repo_reject_null_in_ignored_fields() {
    for value in [
        json!(null),
        json!([null]),
        json!({"nested": [{"value": null}]}),
    ] {
        let tree = json!({"format": "cozytensors.tree/1", "files": {}, "future": value});
        let err = Tree::from_json(&serde_json::to_vec(&tree).unwrap()).unwrap_err();
        assert_eq!(err.code(), "invalid", "{tree}: {err}");
        let repo = json!({
            "format": "cozytensors.repo/1", "name": "acme/model", "releases": [], "future": value,
        });
        let err = Repo::from_json(&serde_json::to_vec(&repo).unwrap()).unwrap_err();
        assert_eq!(err.code(), "invalid", "{repo}: {err}");
    }
    // An extension nested inside a known record still participates in the strict walk.
    let mut repo = json!({
        "format": "cozytensors.repo/1", "name": "acme/model",
        "releases": [{"version": "1", "tree": ObjectRef::of(b"tree"), "future": {"value": null}}],
    });
    assert_eq!(
        Repo::from_json(&serde_json::to_vec(&repo).unwrap())
            .unwrap_err()
            .code(),
        "invalid"
    );
    repo["releases"][0]["future"] = json!({"value": [1, true, "ok"]});
    assert!(Repo::from_json(&serde_json::to_vec(&repo).unwrap()).is_ok());
}

#[test]
fn header_configs_preserve_nested_json_null() {
    let value = json!({"optional": null, "nested": [1, null, {"value": null}]});
    let config: Config = serde_json::from_value(value.clone()).unwrap();
    let header = Header::new(
        vec![],
        Some(Pipeline {
            class: String::new(),
            config: config.clone(),
        }),
        vec![Component {
            name: "model".into(),
            config,
            ..Default::default()
        }],
        Default::default(),
    )
    .unwrap();
    let bytes = header.to_cbor().unwrap();
    let read = Header::from_cbor(&bytes).unwrap();
    assert_eq!(
        serde_json::to_value(&read.pipeline().unwrap().config).unwrap(),
        value
    );
    assert_eq!(
        serde_json::to_value(&read.components()[0].config).unwrap(),
        value
    );
    assert_eq!(read.to_cbor().unwrap(), bytes);
}

fn encoded_header(dtype: Dtype, shape: Vec<u64>) -> Result<Header> {
    Header::new(
        vec![ObjectRef::of(&[7])],
        None,
        vec![Component {
            name: "model".into(),
            tensors: vec![Tensor {
                key: "weight".into(),
                dtype,
                shape,
                body: Body::Encoded {
                    encoding: Encoding {
                        scheme: "future.codec".into(),
                        params: Default::default(),
                    },
                    parts: vec![Part {
                        role: "payload".into(),
                        dtype: Dtype::U8,
                        shape: vec![1],
                        chunk: 0,
                        offset: 0,
                        length: 1,
                    }],
                },
            }],
            ..Default::default()
        }],
        Default::default(),
    )
}

#[test]
fn encoded_logical_dtype_sizes_are_checked_before_decoding() {
    for (dtype, shape) in [
        (Dtype::F32, vec![1 << 63]),
        (Dtype::F32, vec![u64::MAX, 0]),
        (Dtype::F4, vec![1]),
        (Dtype::F6_E2M3, vec![1]),
    ] {
        let err = encoded_header(dtype.clone(), shape.clone()).unwrap_err();
        assert_eq!(err.code(), "invalid", "{dtype} {shape:?}: {err}");

        // Construct a valid future-dtype document, then change only its logical dtype.
        // This exercises the CBOR parser without relying on the rejecting constructor.
        let bytes = encoded_header(Dtype::Other("NEW".into()), shape.clone())
            .unwrap()
            .to_cbor()
            .unwrap();
        let old = b"\x66weight\x63NEW";
        let at = bytes.windows(old.len()).position(|w| w == old).unwrap();
        let mut replacement = Vec::new();
        minicbor::Encoder::new(&mut replacement)
            .str("weight")
            .unwrap()
            .str(dtype.name())
            .unwrap();
        let invalid = [&bytes[..at], &replacement, &bytes[at + old.len()..]].concat();
        let err = Header::from_cbor(&invalid).unwrap_err();
        assert_eq!(err.code(), "invalid", "CBOR {dtype} {shape:?}: {err}");
    }
}

#[test]
fn valid_logical_sizes_and_unknown_dtypes_keep_raw_access() {
    for (dtype, shape) in [
        (Dtype::F32, vec![2]),
        (Dtype::F32, vec![0]),
        (Dtype::F4, vec![2]),
        (Dtype::F6_E2M3, vec![4]),
        (Dtype::Other("NEW".into()), vec![u64::MAX, u64::MAX]),
    ] {
        let h = encoded_header(dtype, shape).unwrap();
        let read = Header::from_cbor(&h.to_cbor().unwrap()).unwrap();
        assert_eq!(read, h);
        let part = read
            .tensor("model", "weight")
            .unwrap()
            .part("payload")
            .unwrap();
        assert_eq!(read.bytes(&part.at(), |_| Ok(vec![7])).unwrap(), [7]);
        assert_eq!(
            read.read("model", "weight", |_| Ok(vec![7]))
                .unwrap_err()
                .code(),
            "unknown_scheme"
        );
    }

    let plain = Header::new(
        vec![ObjectRef::of(&[7])],
        None,
        vec![Component {
            name: "model".into(),
            tensors: vec![Tensor {
                key: "weight".into(),
                dtype: Dtype::Other("NEW".into()),
                shape: vec![u64::MAX, u64::MAX],
                body: Body::Data(Location {
                    chunk: 0,
                    offset: 0,
                    length: 1,
                }),
            }],
            ..Default::default()
        }],
        Default::default(),
    )
    .unwrap();
    let read = Header::from_cbor(&plain.to_cbor().unwrap()).unwrap();
    assert_eq!(read.read("model", "weight", |_| Ok(vec![7])).unwrap(), [7]);
}

//! cozytensors/1 documents through the public reader and writer.

use std::collections::BTreeMap;

use serde_json::json;
use tensorfs::format::{
    variant_of, variant_path, Body, Component, Config, ConfigValue, Dtype, Encoding, Header,
    Location, ObjectRef, Part, Pipeline, Release, Repo, Tensor, Tree, Value, HEADER_PATH,
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
            encoding: Encoding::fp8_rowwise(0, &Dtype::F32),
            parts: vec![
                part("weight", Dtype::F8_E4M3, vec![4, 1024], 0, 4096, 4096),
                part("scale", Dtype::F32, vec![4], 1, 0, 16),
            ],
        },
    };
    vec![plain, encoded]
}

/// A config as `config.json` parses.
fn config(json: serde_json::Value) -> Config {
    serde_json::from_value(json).unwrap()
}

fn components(tensors: Vec<Tensor>) -> Vec<Component> {
    let unet = Component {
        name: "unet".into(),
        class: "diffusers.UNet2DConditionModel".into(),
        config: config(json!({"_class_name": "UNet2DConditionModel", "sample_size": 128,
            "norm_eps": 1e-5, "addition_embed_type": null, "upcast": 1.0})),
        tensors,
        ..Default::default()
    };
    let tokenizer = Component {
        name: "tokenizer".into(),
        class: "transformers.CLIPTokenizer".into(),
        config: config(json!({"model_max_length": 77})),
        assets: [
            ("vocab.json".into(), ObjectRef::of(b"{}")),
            ("merges.txt".into(), ObjectRef::of(b"#")),
        ]
        .into(),
        config_file: Some("tokenizer_config.json".into()),
        ..Default::default()
    };
    vec![unet, tokenizer]
}

fn header_with(tensors: Vec<Tensor>) -> Header {
    let chunks = vec![ObjectRef::of(&[7u8; 8192]), ObjectRef::of(&[1u8; 100])];
    let pipeline = Pipeline {
        class: "diffusers.StableDiffusionXLPipeline".into(),
        config: config(json!({"_class_name": "StableDiffusionXLPipeline",
            "unet": ["diffusers", "UNet2DConditionModel"], "tokenizer": ["transformers", "CLIPTokenizer"]})),
    };
    let prompt = Value::Map(
        [
            ("rows".into(), Value::List(vec!["a cat".into()])),
            ("n".into(), 1u64.into()),
        ]
        .into(),
    );
    let metadata = [("prompt".into(), prompt)].into();
    Header::new(chunks, Some(pipeline), components(tensors), metadata).unwrap()
}

fn header() -> Header {
    header_with(tensors())
}

#[test]
fn writer_is_deterministic_and_reader_round_trips() {
    let h = header();
    let bytes = h.to_cbor().unwrap();
    // [format, chunks, encodings, pipeline, components, metadata]: a 6-array, then the format.
    assert_eq!(&bytes[..15], b"\x86\x6dcozytensors/1");
    // The first chunk: [bstr .size 32, 8192].
    assert_eq!(&bytes[15..19], b"\x82\x82\x58\x20");
    assert_eq!(&bytes[51..54], b"\x19\x20\x00");
    assert!(bytes.ends_with(b"\xa1\x66prompt\xa2\x61n\x01\x64rows\x81\x65a cat"));
    assert_eq!(Header::from_cbor(&bytes).unwrap(), h);
    assert_eq!(Header::from_cbor(&bytes).unwrap().to_cbor().unwrap(), bytes);
    // The tokenizer: [name, class, config, tensors, assets, config_file], assets keyed by path.
    let tokenizer = [b"\x86\x69tokenizer\x78\x1atransformers.CLIPTokenizer".as_slice(), b"\xa1"].concat();
    assert!(bytes.windows(tokenizer.len()).any(|w| w == tokenizer));
    assert!(bytes.windows(22).any(|w| w == b"\x75tokenizer_config.json"));
    let read = Header::from_cbor(&bytes).unwrap();
    assert_eq!(read.pipeline().unwrap().class, "diffusers.StableDiffusionXLPipeline");
    let unet = read.component("unet").unwrap();
    assert_eq!(unet.config["norm_eps"], json!(1e-5));
    assert_eq!(unet.config["addition_embed_type"], ConfigValue::Null);
    assert_eq!(unet.config_name(), "config.json");
    let assets: Vec<_> = read.assets().map(|(c, p, _)| (c, p)).collect();
    assert_eq!(assets, [("tokenizer", "merges.txt"), ("tokenizer", "vocab.json")]);
    assert_eq!(read.objects().count(), 4, "two chunks, two assets");

    let mut wide = h.clone();
    wide.metadata.insert("max".into(), Value::from(u64::MAX));
    wide.metadata.insert("min".into(), Value::from(i64::MIN));
    let wb = wide.to_cbor().unwrap();
    let tail = [b"\x63max\x1b".as_slice(), &[0xff; 8], b"\x63min\x3b\x7f", &[0xff; 7]].concat();
    assert!(wb.windows(tail.len()).any(|w| w == tail));
    assert_eq!(Header::from_cbor(&wb).unwrap(), wide);
    wide.metadata.insert("huge".into(), Value::Int(1 << 64));
    assert_eq!(wide.to_cbor().unwrap_err().code(), "invalid");

    let mut t = Tree {
        files: BTreeMap::from([("README.md".into(), ObjectRef::of(b"# tiny"))]),
    };
    t.set_variant("bf16", ObjectRef::of(&bytes));
    t.set_variant("", ObjectRef::of(&wb));
    let tb = t.to_json().unwrap();
    assert!(tb.starts_with(br#"{"files":{"README.md":"#));
    assert!(tb.ends_with(br#"}},"format":"tensorfs.tree/1"}"#));
    let back = Tree::from_json(&tb).unwrap();
    assert_eq!(back, t);
    assert_eq!(back.variant(Some("bf16")).unwrap(), ("bf16", &ObjectRef::of(&bytes)));
    assert_eq!(back.header(""), Some(&ObjectRef::of(&wb)));
    assert!(back.variant(None).is_err(), "two variants: name one");
    let files: Vec<&str> = back.repo_files().map(|(p, _)| p).collect();
    assert_eq!(files, ["README.md"]);
    assert_eq!(HEADER_PATH, "model.cozytensors");

    let r = Repo {
        name: "Acme/Tiny".into(),
        releases: vec![Release {
            version: "1.0".into(),
            tree: ObjectRef::of(&tb),
            yanked: false,
            revision: 0,
        }],
        ..Default::default()
    };
    let rb = r.to_json().unwrap();
    assert_eq!(Repo::from_json(&rb).unwrap(), r);
    assert_eq!(r.key(), "refs/acme/tiny.json");
    assert_eq!(r.release(None).map(|r| &r.tree), Some(&ObjectRef::of(&tb)));
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
    let mut newer = patch(&good, b"\x86\x6dcozytensors/1", b"\x87\x6fcozytensors/1.9");
    newer = patch(&newer, &chunk, &longer);
    // A later field in the pipeline, in a component after its config_file, and in an asset.
    newer = patch(&newer, b"\x82\x78\x23diffusers.StableDiffusionXLPipeline", b"\x83\x78\x23diffusers.StableDiffusionXLPipeline");
    newer = patch(&newer, b"\x82\x85\x64unet", b"\xf5\x82\x85\x64unet");
    newer = patch(&newer, b"\x86\x69tokenizer", b"\x87\x69tokenizer");
    newer = patch(&newer, b"\x6avocab.json\x82", b"\x6avocab.json\x83");
    newer = patch(&newer, b"\x02\x75tokenizer_config.json", b"\x02\x60\x75tokenizer_config.json\x80");
    // With later fields, a default config_file is written out.
    newer = patch(&newer, b"\x85\x64unet", b"\x87\x64unet");
    newer = patch(&newer, b"\xa0\x87\x69tokenizer", b"\xa0\x6bconfig.json\xf4\x87\x69tokenizer");
    newer = patch(&newer, b"\x64BF16", b"\x67F3_E1M1");
    newer = patch(&newer, b"\x6bfp8.rowwise", b"\x6bacme.int3x2");
    newer.extend(b"\x82\x01\x62hi");
    let read = Header::from_cbor(&newer).unwrap();
    let a = read.tensor("unet", "a.weight").unwrap();
    assert_eq!(a.dtype, Dtype::Other("F3_E1M1".into()));
    assert_eq!(a.body, Body::Data(Location { chunk: 0, offset: 0, length: 12 }));
    assert_eq!(read.chunks, h.chunks);
    assert_eq!((read.pipeline(), &read.components()[1]), (h.pipeline(), &h.components()[1]));
    let unet = read.component("unet").unwrap();
    assert_eq!((&unet.config, &unet.config_file), (&h.components()[0].config, &None));
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
        ("indefinite root", patch(&good, b"\x86\x6d", b"\x9f\x6d"), "invalid"),
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
        ("null in a later field", [patch(&good, b"\x86\x6d", b"\x87\x6d"), vec![0xf6]].concat(), "invalid"),
        ("float in a later field", [patch(&good, b"\x86\x6d", b"\x87\x6d"), vec![0xf9, 0x3c, 0]].concat(), "invalid"),
        ("component name", patch(&good, b"\x85\x64unet", b"\x85\x64un/t"), "invalid"),
        ("float not shortest", patch(&good, b"\xf9\x3c\x00", b"\xfa\x3f\x80\x00\x00"), "invalid"),
        ("double not shortest", patch(&good, b"\xf9\x3c\x00", b"\xfb\x3f\xf0\0\0\0\0\0\0"), "invalid"),
        ("NaN payload", patch(&good, b"\xf9\x3c\x00", b"\xf9\x7e\x01"), "invalid"),
        ("float as a class", patch(&good, b"\x78\x1atransformers.CLIPTokenizer", b"\xf9\x3c\x00"), "invalid"),
        ("pipeline of one field", patch(&good, b"\x82\x78\x23diffusers", b"\x81\x78\x23diffusers"), "invalid"),
        ("short component", patch(&good, b"\x85\x64unet", b"\x84\x64unet"), "invalid"),
        ("config_file not text", patch(&good, b"\x75tokenizer_config.json", b"\x15"), "invalid"),
        ("config_file default", patch(&good, b"\x75tokenizer_config.json", b"\x6bconfig.json"), "invalid"),
        ("asset path", patch(&good, b"\x6avocab.json", b"\x6av/../.json"), "invalid"),
        ("asset digest", patch(&good, b"\x6avocab.json\x82\x58\x20", b"\x6avocab.json\x82\x58\x1f"), "invalid"),
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
        let unet = Component { name: "unet".into(), tensors, ..Default::default() };
        let err = Header::new(chunks, None, vec![unet], Default::default()).expect_err(name);
        assert_eq!(err.code(), "invalid", "{name}: {err}");
    }

    for (name, edit) in [
        ("config_file default", (|c| c.config_file = Some("config.json".into())) as fn(&mut Component)),
        ("config_file path", |c| c.config_file = Some("../config.json".into())),
        ("asset path", |c| {
            c.assets.insert("a//b".into(), ObjectRef::of(b""));
        }),
        ("asset is the config", |c| {
            c.assets.insert("tokenizer_config.json".into(), ObjectRef::of(b""));
        }),
    ] {
        let mut c = components(tensors());
        edit(&mut c[1]);
        let err = Header::new(header().chunks, None, c, Default::default()).expect_err(name);
        assert_eq!(err.code(), "invalid", "{name}: {err}");
    }

    // The layouts before the pipeline: a 5-element root is pre-cut, and so is JSON.
    let pre_cut = b"\x85\x6dcozytensors/1\x80\x80\x80\xa0";
    assert_eq!(Header::from_cbor(pre_cut).unwrap_err().code(), "pre_cut");
    assert_eq!(Header::from_cbor(br#"{"format":"cozytensors/1"}"#).unwrap_err().code(), "pre_cut");
    let other = b"\x85\x6dcozytensors/2\x80\x80\x80\xa0";
    assert_eq!(Header::from_cbor(other).unwrap_err().code(), "format");

    for path in ["../x", "/x", "a//b", "a/./b", "", "chunks/x"] {
        let doc = json!({"format": "tensorfs.tree/1", "files": {path: {"sha256": "0".repeat(64), "length": 0}}});
        assert!(Tree::from_json(doc.to_string().as_bytes()).is_err(), "{path:?}");
    }
    let tree = json!({"sha256": "0".repeat(64), "length": 0});
    let repo = json!({"format": "tensorfs.repo/1", "name": "acme/tiny", "releases": [
        {"version": "1", "tree": tree}, {"version": "1", "tree": tree}]});
    assert!(Repo::from_json(repo.to_string().as_bytes()).is_err());
    let lanes = json!({"format": "tensorfs.repo/1", "name": "acme/tiny", "releases": [
        {"version": "1", "lanes": {}}]});
    assert!(Repo::from_json(lanes.to_string().as_bytes()).is_err(), "a release names a tree");
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
    let h = Header::new(chunks, None, vec![], Default::default()).unwrap();
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

/// Floats and null live only in configs; each float takes the shortest of half, single and
/// double that holds it exactly, and reads back bit for bit.
#[test]
fn config_floats_take_their_shortest_exact_form() {
    let floats: &[(f64, &[u8])] = &[
        (1.0, b"\xf9\x3c\x00"),
        (-0.0, b"\xf9\x80\x00"),
        (65504.0, b"\xf9\x7b\xff"),
        (2f64.powi(-24), b"\xf9\x00\x01"),
        (2f64.powi(-14), b"\xf9\x04\x00"),
        (65520.0, b"\xfa\x47\x7f\xf0\x00"),
        (100000.0, b"\xfa\x47\xc3\x50\x00"),
        (2f64.powi(-25), b"\xfa\x33\x00\x00\x00"),
        (0.1, b"\xfb\x3f\xb9\x99\x99\x99\x99\x99\x9a"),
        (1e-5, b"\xfb\x3e\xe4\xf8\xb5\x88\xe3\x68\xf1"),
        (5e-324, b"\xfb\0\0\0\0\0\0\0\x01"),
    ];
    for (f, want) in floats {
        let mut c = components(vec![]);
        c[0].config = config(json!({"v": f}));
        let h = Header::new(vec![], None, c, Default::default()).unwrap();
        let bytes = h.to_cbor().unwrap();
        let entry = [b"\xa1\x61v".as_slice(), want].concat();
        assert!(bytes.windows(entry.len()).any(|w| w == entry), "{f}");
        let back = Header::from_cbor(&bytes).unwrap();
        let got = back.components()[0].config["v"].as_f64().unwrap();
        assert_eq!(got.to_bits(), f.to_bits(), "{f}");
        assert_eq!(back, h);
        // A wider form of the same value is refused.
        if want.len() < 9 {
            let wide = [b"\xa1\x61v\xfb".as_slice(), &f.to_bits().to_be_bytes()].concat();
            let err = Header::from_cbor(&patch(&bytes, &entry, &wide)).unwrap_err();
            assert_eq!(err.code(), "invalid", "{f}");
        }
    }
    // So are infinities and NaN: a config is parsed JSON.
    let good = header().to_cbor().unwrap();
    for bad in [b"\xf9\x7c\x00", b"\xf9\xfc\x00", b"\xf9\x7e\x00"] {
        let err = Header::from_cbor(&patch(&good, b"\xf9\x3c\x00", bad)).unwrap_err();
        assert_eq!(err.code(), "invalid", "{bad:?}: {err}");
    }
    let nested = config(json!({"a": [1, 1.5, null, {"b": -2.5e-8}], "c": -3, "d": "x", "e": true}));
    let mut c = components(vec![]);
    c[0].config = nested.clone();
    let pipeline = Pipeline { class: String::new(), config: nested };
    let h = Header::new(vec![], Some(pipeline), c, Default::default()).unwrap();
    assert_eq!(Header::from_cbor(&h.to_cbor().unwrap()).unwrap(), h);
    let json = serde_json::to_value(&h.components()[0].config).unwrap();
    assert_eq!(json, json!({"a": [1, 1.5, null, {"b": -2.5e-8}], "c": -3, "d": "x", "e": true}));
    let one = config(json!({"v": 1.0}));
    let bytes = Header::new(vec![], Some(Pipeline { class: String::new(), config: one }), vec![], Default::default());
    assert!(bytes.unwrap().to_cbor().unwrap().windows(3).any(|w| w == b"\xf9\x3c\x00"), "1.0 stays a float");
}

#[test]
fn variants_name_their_files() {
    for (path, variant) in [("model.cozytensors", Some("")), ("model.fp8.cozytensors", Some("fp8"))] {
        assert_eq!(variant_of(path), variant);
        assert_eq!(variant_path(variant.unwrap()), path);
    }
    for path in ["model..cozytensors", "models.cozytensors", "a/model.cozytensors", "model.a/b.cozytensors", "README.md"] {
        assert_eq!(variant_of(path), None, "{path}");
    }
}

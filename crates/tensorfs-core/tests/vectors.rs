//! The frozen vector corpus (`vectors/cases`, `vectors/closure`, tfs-008): this file builds
//! it, every committed case answers its frozen verdict, and the committed corpus is exactly
//! what the builder writes. After an intended change, regenerate it with
//!
//!     cargo test -p tensorfs-core --test vectors -- --ignored regenerate

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use tensorfs_core::canon;
use tensorfs_core::dtype::Dtype;
use tensorfs_core::err::Refusal;
use tensorfs_core::header::{Asset, Body, Closure, Header, Part, Tensor};
use tensorfs_core::ids::{Doc, ObjectRef, Plain};
use tensorfs_core::ingest::stamp::Stamp;
use tensorfs_core::ingest::{Evidence, IngestProfile, IngestSubject, IngestVerificationReceipt};
use tensorfs_core::manifest::{Draft, Entry, Manifest};
use tensorfs_core::registry;
use tensorfs_core::spec::EncodingSpec;

type TfsResult<T> = tensorfs_core::err::Result<T>;

macro_rules! ok {
    ($e:expr) => {
        $e.unwrap()
    };
}

fn default_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vectors")
}

// ------------------------------------------------------------------ corpus construction

fn seed_spec(alias: &str, nth: usize) -> EncodingSpec {
    registry::seeds()
        .into_iter()
        .filter(|s| s.alias == alias)
        .nth(nth)
        .expect("seed")
        .spec
}

/// A legal spec whose geometry overflows u64 on a modest logical shape: the corpus's
/// checked-arithmetic probe. Never a platform alias.
fn probe_spec() -> EncodingSpec {
    use tensorfs_core::relation::{Carrier, Dim, Relation};
    EncodingSpec {
        logical_dtypes: vec![Dtype::Bf16],
        logical_rank: Some(2),
        roles: vec![(
            "value".to_string(),
            tensorfs_core::spec::Role {
                carrier: Carrier::Set(vec![Dtype::U8]),
                shape: Relation::Dims(vec![
                    Dim::CeilBlock {
                        axis: 0,
                        block: 1,
                        mul: 9007199254740991,
                    },
                    Dim::Axis { axis: 1 },
                ]),
            },
        )],
        vectors: None,
    }
}

fn pattern(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i * 31 + 7) as u8).collect()
}

fn part_of(dt: Dtype, shape: Vec<u64>) -> Part {
    let bytes = pattern(shape.iter().product::<u64>() as usize * dt.size() as usize);
    Part::plan(dt, shape, &bytes)
}

fn config_bytes() -> Vec<u8> {
    br#"{"hidden_size":1536}"#.to_vec()
}

fn encodings_of(specs: &[&EncodingSpec]) -> Vec<EncodingSpec> {
    let mut v: Vec<EncodingSpec> = specs.iter().map(|spec| (*spec).clone()).collect();
    v.sort_by_key(EncodingSpec::object_id);
    v
}

/// Positive header: plain inline tensor + an fp8-rowwise tensor (segments AND inline).
fn header_a(config: Option<Vec<u8>>) -> (Header, Closure) {
    let plain = seed_spec("plain/1", 0);
    let fp8 = seed_spec("fp8-rowwise/1", 0);
    let mut closure = Closure::default();
    let h = Header {
        configs: config
            .map(|c| vec![("transformer".to_string(), c)])
            .unwrap_or_default(),
        assets: vec![(
            "tokenizer/vocab.txt".to_string(),
            Asset {
                logical_sha256: tensorfs_core::sha256::hex_digest(b"one\ntwo\n"),
                logical_length: 8,
                media_type: "text/plain".to_string(),
                segments: vec![ObjectRef::of(b"one\ntwo\n")],
            },
        )],
        encodings: encodings_of(&[&plain, &fp8]),
        components: vec![(
            "transformer".to_string(),
            vec![
                (
                    "blocks.0.attn.to_q.weight".to_string(),
                    Tensor {
                        dtype: Dtype::Bf16,
                        shape: vec![64, 128],
                        encoding: fp8.object_id(),
                        parts: vec![
                            ("data".to_string(), part_of(Dtype::F8E4M3FN, vec![64, 128])),
                            ("scale".to_string(), part_of(Dtype::F32, vec![64])),
                        ],
                    },
                ),
                (
                    "blocks.0.norm.weight".to_string(),
                    Tensor {
                        dtype: Dtype::Bf16,
                        shape: vec![128],
                        encoding: plain.object_id(),
                        parts: vec![("value".to_string(), part_of(Dtype::Bf16, vec![128]))],
                    },
                ),
            ],
        )],
    };
    closure.insert(plain);
    closure.insert(fp8);
    (h, closure)
}

/// Positive header: nvfp4 W4A4 base variant (packed nibble pairs + block scales + scalar).
fn header_nvfp4() -> (Header, EncodingSpec) {
    let spec = seed_spec("nvfp4-w4a4/1", 0);
    let h = Header {
        configs: vec![],
        assets: vec![],
        encodings: encodings_of(&[&spec]),
        components: vec![(
            "transformer".to_string(),
            vec![(
                "blocks.0.mlp.fc1.weight".to_string(),
                Tensor {
                    dtype: Dtype::Bf16,
                    shape: vec![256, 512],
                    encoding: spec.object_id(),
                    parts: vec![
                        ("weight".to_string(), part_of(Dtype::U8, vec![256, 256])),
                        (
                            "weight_scale".to_string(),
                            part_of(Dtype::F8E4M3FN, vec![256, 32]),
                        ),
                        ("weight_scale_2".to_string(), part_of(Dtype::F32, vec![])),
                    ],
                },
            )],
        )],
    };
    (h, spec)
}

fn manifest_checkpoint(header: &Header, extra_file: bool) -> TfsResult<Manifest> {
    let mut entries = vec![(
        "model.cozytensors".to_string(),
        Entry::CozyTensors(header.object_ref()?),
    )];
    if extra_file {
        entries.push((
            "README.txt".to_string(),
            Entry::File(ObjectRef::of(b"sample model\n")),
        ));
    }
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    Draft { entries }.seal()
}

fn profile_normal(allowed: Vec<ObjectRef>, dialect: &str) -> IngestProfile {
    IngestProfile {
        class: "normal".to_string(),
        source_dialect: dialect.to_string(),
        converter_build: tensorfs_core::ids::object_id(b"converter-build-v1"),
        converter_recipe: ObjectRef::of(b"recipe"),
        expected_tensor_schema_digest: tensorfs_core::ids::object_id(b"tensor-schema"),
        allowed_encodings: allowed,
        max_source_bytes: 1 << 40,
        max_tensors: 100_000,
    }
}

fn subject(profile: &IngestProfile, header: &Header, tenant: &str) -> TfsResult<IngestSubject> {
    Ok(IngestSubject {
        profile: profile.object_ref(),
        tenant: tenant.to_string(),
        sources: vec![(tensorfs_core::ids::object_id(b"hf://acme/model@rev1"), 1)],
        proposed_header: header.object_ref()?,
        candidates: vec![ObjectRef::of(b"candidate")],
    })
}

/// The corpus twin of what `ingest::stamp::stamp` mints. The vector exists so the LEDGER's
/// canonical bytes are frozen alongside the receipt that names them.
fn stamp_doc(
    prof: &IngestProfile,
    sub: &IngestSubject,
    snap: &Manifest,
    header: &Header,
) -> TfsResult<Stamp> {
    Ok(Stamp {
        subject: sub.object_ref(),
        profile: prof.object_ref(),
        header: header.object_ref()?,
        manifest: snap.object_ref(),
        tensor_schema_digest: header.tensor_schema_digest(),
        converter_build: prof.converter_build.clone(),
        converter_recipe: prof.converter_recipe.clone(),
        candidate_set: tensorfs_core::ingest::stamp::candidate_set_digest(
            &sub.candidates.iter().map(|c| c.id()).collect::<Vec<_>>(),
        ),
        golden: tensorfs_core::ingest::stamp::GOLDEN_PASS.to_string(),
        isolation_tier: 1,
        sources: sub.sources.clone(),
        accounting: vec![
            ("bytes_written".to_string(), 1 << 20),
            ("tensors".to_string(), 2),
        ],
    })
}

fn receipt(
    sub: &IngestSubject,
    snap: &Manifest,
    header: &Header,
    stamp: &Stamp,
    signer: &str,
) -> TfsResult<IngestVerificationReceipt> {
    Ok(IngestVerificationReceipt {
        subject: sub.object_ref(),
        manifest: snap.object_ref(),
        header: header.object_ref()?,
        stamp: stamp.object_ref(),
        evidence: Evidence::ApprovedProducer {
            implementation: tensorfs_core::ids::object_id(b"converter-impl"),
            recipe: ObjectRef::of(b"recipe"),
        },
        signer: signer.to_string(),
        signature: "ed25519:AAAA".to_string(),
        observations: vec![
            ("source_bytes".to_string(), 1 << 20),
            ("tensors".to_string(), 2),
        ],
    })
}

// ------------------------------------------------------------------ vector files

struct Case {
    group: &'static str,
    name: String,
    doc: &'static str,
    bytes: Vec<u8>,
    expect: &'static str,
    code: Option<String>,
    note: String,
}

fn ok_case(group: &'static str, name: &str, doc: &'static str, bytes: Vec<u8>, note: &str) -> Case {
    Case {
        group,
        name: name.to_string(),
        doc,
        bytes,
        expect: "ok",
        code: None,
        note: note.to_string(),
    }
}

fn red_case(
    group: &'static str,
    name: &str,
    doc: &'static str,
    bytes: Vec<u8>,
    code: &str,
    note: &str,
) -> Case {
    Case {
        group,
        name: name.to_string(),
        doc,
        bytes,
        expect: "refuse",
        code: Some(code.to_string()),
        note: note.to_string(),
    }
}

fn sub(bytes: &[u8], from: &str, to: &str) -> Vec<u8> {
    let hay = String::from_utf8(bytes.to_vec()).expect("utf8");
    assert!(hay.contains(from), "surgery anchor {from:?} absent");
    hay.replacen(from, to, 1).into_bytes()
}

fn binary_sub(bytes: &[u8], from: &[u8], to: &[u8]) -> Vec<u8> {
    let at = bytes
        .windows(from.len())
        .position(|window| window == from)
        .unwrap_or_else(|| panic!("binary surgery anchor {from:?} absent"));
    let mut out = Vec::with_capacity(bytes.len() + to.len() - from.len());
    out.extend_from_slice(&bytes[..at]);
    out.extend_from_slice(to);
    out.extend_from_slice(&bytes[at + from.len()..]);
    out
}

fn build_corpus() -> TfsResult<(Vec<Case>, Vec<EncodingSpec>)> {
    let mut cases: Vec<Case> = Vec::new();
    let (ha, closure) = header_a(Some(config_bytes()));
    let ha_bytes = ha.conformance_bytes()?;
    let (hn, _) = header_nvfp4();
    let plain = seed_spec("plain/1", 0);
    let fp8 = seed_spec("fp8-rowwise/1", 0);
    let mxfp8 = seed_spec("mxfp8/1", 0);
    let svdq = seed_spec("svdq-microscale/1", 0);
    // the corpus closure ships the WHOLE platform namespace, not only the cited specs
    let mut specs: Vec<EncodingSpec> = registry::seeds().into_iter().map(|s| s.spec).collect();
    for s in closure
        .specs
        .iter()
        .map(|(_, s)| s.clone())
        .chain([probe_spec()])
    {
        if !specs.iter().any(|x| x.object_id() == s.object_id()) {
            specs.push(s);
        }
    }

    // --- positives
    cases.push(ok_case(
        "header",
        "plain-fp8",
        "header",
        ha_bytes.clone(),
        "inline at the 256 B cap + segmented data",
    ));
    cases.push(ok_case(
        "header",
        "nvfp4-base",
        "header",
        hn.conformance_bytes()?,
        "nvfp4 W4A4 base variant, rank-0 scalar role",
    ));
    let (ha2, _) = header_a(None);
    cases.push(ok_case(
        "header",
        "plain-fp8-noconfig",
        "header",
        ha2.conformance_bytes()?,
        "same tensors, no configs: same tensor schema, different header id",
    ));
    let mut reordered = ha.clone();
    reordered.components[0].1.reverse();
    cases.push(ok_case(
        "header",
        "plain-fp8-reordered",
        "header",
        reordered.conformance_bytes()?,
        "same tensor schema and objects, opposite identity-bearing tensor order",
    ));
    cases.push(ok_case(
        "encoding",
        "plain",
        "encoding",
        plain.canonical_bytes(),
        "entry zero; carries vectors",
    ));
    cases.push(ok_case(
        "encoding",
        "mxfp8",
        "encoding",
        mxfp8.canonical_bytes(),
        "ceil-block scale relation; vectorless",
    ));
    cases.push(ok_case(
        "encoding",
        "svdq-microscale",
        "encoding",
        svdq.canonical_bytes(),
        "rank-7 permuted scale grid mined from v1",
    ));
    cases.push(ok_case(
        "encoding_vectors",
        "plain",
        "vectors",
        registry::plain_vectors().fixture_bytes(),
        "raw bit patterns: signed zero, NaN payload, infinity, rank-0",
    ));

    let snap = manifest_checkpoint(&ha, false)?;
    let snap2 = manifest_checkpoint(&ha, true)?;
    cases.push(ok_case(
        "manifest",
        "checkpoint",
        "manifest",
        snap.canonical_bytes(),
        "manifest identity is its exact virtual-tree bytes",
    ));
    cases.push(ok_case(
        "manifest",
        "checkpoint-extra-file",
        "manifest",
        snap2.canonical_bytes(),
        "one more sibling file: new manifest_id, same header id",
    ));

    let prof = profile_normal(
        vec![plain.object_ref(), fp8.object_ref()],
        "safetensors.diffusers",
    );
    let sub_doc = subject(&prof, &ha, "acme")?;
    let stmp = stamp_doc(&prof, &sub_doc, &snap, &ha)?;
    let rcpt = receipt(&sub_doc, &snap, &ha, &stmp, "platform-key-1")?;
    cases.push(ok_case(
        "ingest",
        "profile-normal",
        "profile",
        prof.canonical_bytes(),
        "normal profile over platform-aliased digests",
    ));
    cases.push(ok_case(
        "ingest",
        "subject",
        "subject",
        sub_doc.canonical_bytes(),
        "tenant + exact source generations",
    ));
    cases.push(ok_case(
        "ingest",
        "stamp",
        "stamp",
        stmp.canonical_bytes(),
        "the verified ledger: tensor schema, candidate set, converter, golden, accounting",
    ));
    cases.push(ok_case(
        "ingest",
        "receipt",
        "receipt",
        rcpt.canonical_bytes(),
        "one evidence arm; signer in the envelope; binds its stamp",
    ));

    // --- the two request-time machine results retained by TensorFS.
    {
        let records = registry::capability_records();
        let plain_id = seed_spec("plain/1", 0).object_id();
        let admitted = records
            .admit(&plain_id, "cpu")
            .expect("plain/1 is qualified on the reference CPU");
        cases.push(ok_case(
            "machine",
            "capability-admitted",
            "machine_capability",
            tensorfs_core::machine::capability_json(Some(admitted), &plain_id, "cpu", "", &["cpu"]),
            "qualified encoding/device pair",
        ));
        let refusal = records
            .admit(&plain_id, "cuda.sm89")
            .expect_err("no accelerator class is compiled in");
        cases.push(ok_case(
            "machine",
            "capability-unqualified",
            "machine_capability",
            tensorfs_core::machine::capability_json(
                None,
                &plain_id,
                "cuda.sm89",
                &refusal.detail,
                &["cpu"],
            ),
            "unqualified encoding/device pair names where it is qualified",
        ));
    }

    // --- restricted deterministic CBOR representations (all red)
    let g = "canonical";
    cases.push(red_case(
        g,
        "old-json-header",
        "header",
        canon::write(&ha.diagnostic_value()?),
        "MALFORMED_CBOR",
        "the pre-hardcut JSON header has no reader",
    ));
    cases.push(red_case(
        g,
        "nonpreferred-root-length",
        "header",
        binary_sub(&ha_bytes, &[0x85], &[0x98, 0x05]),
        "NONCANONICAL_ENCODING",
        "root arity five encoded with an unnecessary width byte",
    ));
    cases.push(red_case(
        g,
        "indefinite-root",
        "header",
        binary_sub(&ha_bytes, &[0x85], &[0x9f]),
        "NONCANONICAL_ENCODING",
        "indefinite containers are outside the profile",
    ));
    cases.push(red_case(
        g,
        "forbidden-map",
        "header",
        binary_sub(&ha_bytes, &[0x85], &[0xa5]),
        "MALFORMED_CBOR",
        "maps are outside the all-array profile",
    ));
    cases.push(red_case(
        g,
        "forbidden-tag",
        "header",
        [vec![0xc0], ha_bytes.clone()].concat(),
        "MALFORMED_CBOR",
        "tags are outside the profile",
    ));
    cases.push(red_case(
        g,
        "forbidden-float",
        "header",
        [vec![0xfb], 1f64.to_be_bytes().to_vec()].concat(),
        "MALFORMED_CBOR",
        "floats are outside the profile",
    ));
    cases.push(red_case(
        g,
        "forbidden-null",
        "header",
        vec![0xf6],
        "MALFORMED_CBOR",
        "null is not admitted",
    ));
    cases.push(red_case(
        g,
        "truncated",
        "header",
        ha_bytes[..ha_bytes.len() - 1].to_vec(),
        "MALFORMED_CBOR",
        "the final byte is absent",
    ));
    cases.push(red_case(
        g,
        "trailing-bytes",
        "header",
        [ha_bytes.clone(), vec![0x80]].concat(),
        "TRAILING_BYTES",
        "a second value after the document",
    ));

    // --- header schema/geometry reds
    let g = "header-red";
    cases.push(red_case(
        g,
        "unknown-field",
        "header",
        binary_sub(&ha_bytes, &[0x85], &[0x86]),
        "WRONG_TYPE",
        "the fixed root has no sixth field",
    ));
    cases.push(red_case(
        g,
        "unknown-major",
        "header",
        binary_sub(&ha_bytes, b"cozytensors/1", b"cozytensors/2"),
        "UNKNOWN_FORMAT",
        "unknown format major",
    ));
    let mut h = ha.clone();
    if let Body::Segments(segments) = &mut h.components[0].1[0].1.parts[0].1.body {
        segments[0].length -= 1;
    }
    cases.push(red_case(
        g,
        "anchor-equation",
        "header",
        h.conformance_bytes()?,
        "BYTE_LENGTH_MISMATCH",
        "segment total off by one byte",
    ));
    let mut unknown_dtype = ha_bytes.clone();
    let dtype_at = unknown_dtype
        .windows(b"blocks.0.attn.to_q.weight".len())
        .position(|window| window == b"blocks.0.attn.to_q.weight")
        .unwrap()
        + b"blocks.0.attn.to_q.weight".len();
    unknown_dtype[dtype_at] = 12;
    cases.push(red_case(
        g,
        "dtype-outside-enum",
        "header",
        unknown_dtype,
        "DTYPE_UNKNOWN",
        "carrier dtype outside the closed enum",
    ));

    let mut h = ha.clone();
    h.components[0].1[0].1.parts.retain(|(n, _)| n != "scale");
    cases.push(red_case(
        g,
        "role-missing",
        "header",
        h.conformance_bytes()?,
        "ROLE_SET_MISMATCH",
        "spec role set is EXACT",
    ));
    let mut h = ha.clone();
    h.components[0].1[0]
        .1
        .parts
        .push(("input_scale".to_string(), part_of(Dtype::F32, vec![])));
    cases.push(red_case(
        g,
        "role-extra",
        "header",
        h.conformance_bytes()?,
        "ROLE_SET_MISMATCH",
        "an extra role is a different spec digest",
    ));
    let mut h = ha.clone();
    h.components[0].1[0].1.parts[1].1.shape = vec![32];
    h.components[0].1[0].1.parts[1].1.body = Body::Inline(pattern(128));
    cases.push(red_case(
        g,
        "relation-shape",
        "header",
        h.conformance_bytes()?,
        "SHAPE_MISMATCH",
        "scale shape != relation result",
    ));
    let mut h = ha.clone();
    h.components[0].1[1].1.shape = vec![0];
    cases.push(red_case(
        g,
        "zero-element",
        "header",
        h.conformance_bytes()?,
        "ZERO_ELEMENT",
        "zero-element tensor",
    ));
    let mut h = ha.clone();
    h.components[0].1[1].1.parts[0].1 = Part {
        dtype: Dtype::Bf16,
        shape: vec![128],
        body: Body::Segments(vec![ObjectRef::of(&pattern(256))]),
    };
    cases.push(red_case(
        g,
        "inline-threshold-under",
        "header",
        h.conformance_bytes()?,
        "INLINE_THRESHOLD",
        "256 B stored as a segment",
    ));
    let mut h = ha.clone();
    h.components[0].1[1].1.shape = vec![130];
    h.components[0].1[1].1.parts[0].1 = Part {
        dtype: Dtype::Bf16,
        shape: vec![130],
        body: Body::Inline(pattern(260)),
    };
    cases.push(red_case(
        g,
        "inline-threshold-over",
        "header",
        h.conformance_bytes()?,
        "INLINE_THRESHOLD",
        "260 B stored inline",
    ));
    let mut h = ha.clone();
    h.components[0].1[1].1.parts[0].1.body = Body::Segments(vec![]);
    cases.push(red_case(
        g,
        "empty-segments",
        "header",
        h.conformance_bytes()?,
        "COUNT_CAP",
        "segmented body is a non-empty ObjectRef array",
    ));
    let mut h = ha.clone();
    h.encodings.push(probe_spec());
    h.encodings.sort_by_key(EncodingSpec::object_id);
    cases.push(red_case(
        g,
        "uncited-encoding",
        "header",
        h.conformance_bytes()?,
        "UNCITED_ENCODING",
        "derivable-fact law",
    ));
    let mut h = ha_bytes.clone();
    let key = b"blocks.0.norm.weight";
    let at = h
        .windows(key.len())
        .position(|window| window == key)
        .unwrap()
        + key.len();
    h[at + 4] = 2;
    cases.push(red_case(
        g,
        "encoding-not-listed",
        "header",
        h,
        "ENCODING_MISMATCH",
        "cited digest missing from `encodings`",
    ));
    let mut h = ha.clone();
    h.components.clear();
    cases.push(red_case(
        g,
        "empty-components",
        "header",
        h.conformance_bytes()?,
        "EMPTY_COMPONENTS",
        "no components",
    ));

    // ---- tfs-015: hostile native bytes, keys, totals, capability ----
    cases.push(red_case(
        g,
        "digest-31-bytes",
        "header",
        binary_sub(&ha_bytes, &[0x58, 0x20], &[0x58, 0x1f]),
        "MALFORMED_DIGEST",
        "a digest is exactly 32 native bytes",
    ));
    cases.push(red_case(
        g,
        "digest-33-bytes",
        "header",
        binary_sub(&ha_bytes, &[0x58, 0x20], &[0x58, 0x21]),
        "MALFORMED_DIGEST",
        "a digest is exactly 32 native bytes",
    ));
    let mut h = ha.clone();
    h.configs.push(("aaa".to_string(), config_bytes()));
    cases.push(red_case(
        g,
        "configs-unsorted",
        "header",
        h.conformance_bytes()?,
        "SORT_ORDER",
        "configs are a sorted set, not construction order",
    ));
    for (name, bytes, code, note) in [
        (
            "config-not-canonical",
            br#"{ "hidden_size": 1536 }"#.as_slice(),
            "NONCANONICAL_ENCODING",
            "inline configs store exactly their RFC 8785 bytes",
        ),
        (
            "config-duplicate-key",
            br#"{"hidden_size":1536,"hidden_size":2048}"#.as_slice(),
            "DUPLICATE_KEY",
            "duplicate JSON keys never acquire a value",
        ),
        (
            "config-trailing-bytes",
            br#"{"hidden_size":1536}x"#.as_slice(),
            "TRAILING_BYTES",
            "a config is one complete JSON value",
        ),
        (
            "config-nonfinite",
            br#"{"scale":NaN}"#.as_slice(),
            "MALFORMED_JSON",
            "non-finite values are outside JSON and JCS",
        ),
    ] {
        let mut header = ha.clone();
        header.configs[0].1 = bytes.to_vec();
        cases.push(red_case(
            g,
            name,
            "header",
            header.conformance_bytes()?,
            code,
            note,
        ));
    }
    let mut h = ha.clone();
    h.encodings.reverse();
    cases.push(red_case(
        g,
        "encodings-unsorted",
        "header",
        h.conformance_bytes()?,
        "SORT_ORDER",
        "EncodingSpec references sort by raw digest",
    ));

    // Tensor keys are data-shaped and therefore the hostile-input surface.
    for (name, key, code, note) in [
        ("key-empty", "", "NON_ASCII_FIELD", "the empty key"),
        (
            "key-non-ascii",
            "blocks.0.attn.tö_q.weight",
            "NON_ASCII_FIELD",
            "a non-ASCII tensor key",
        ),
    ] {
        let mut header = ha.clone();
        header.components[0].1[0].0 = key.to_string();
        cases.push(red_case(
            g,
            name,
            "header",
            header.conformance_bytes()?,
            code,
            note,
        ));
    }

    // A zero-length segment is a fetch that can never satisfy the anchor equation.
    let mut h = ha.clone();
    if let Body::Segments(segments) = &mut h.components[0].1[0].1.parts[0].1.body {
        segments[0].length = 0;
    }
    cases.push(red_case(
        g,
        "segment-zero-length",
        "header",
        h.conformance_bytes()?,
        "ZERO_ELEMENT",
        "an empty segment record",
    ));

    // Whole-DOCUMENT byte total, not just this part's.
    let huge = Header {
        configs: vec![],
        assets: vec![],
        encodings: encodings_of(&[&plain]),
        components: vec![(
            "transformer".to_string(),
            vec![(
                "huge.weight".to_string(),
                Tensor {
                    dtype: Dtype::F64,
                    shape: vec![1 << 48],
                    encoding: plain.object_id(),
                    parts: vec![(
                        "value".to_string(),
                        Part {
                            dtype: Dtype::F64,
                            shape: vec![1 << 48],
                            body: Body::Segments(vec![ObjectRef {
                                sha256: "0".repeat(64),
                                length: 1u64 << 51,
                            }]),
                        },
                    )],
                },
            )],
        )],
    };
    cases.push(red_case(
        g,
        "total-bytes-cap",
        "header",
        huge.conformance_bytes()?,
        "TOTAL_BYTES_CAP",
        "one legal tensor over the whole-document byte total",
    ));

    // A spec object present in the closure that this runtime's pin does not alias: valid
    // DATA, unsupported EXECUTION. `header` accepts the same bytes; only serving refuses.
    let mut foreign = seed_spec("plain/1", 0);
    foreign.vectors = None;
    let hf = Header {
        configs: vec![],
        assets: vec![],
        encodings: encodings_of(&[&foreign]),
        components: vec![(
            "transformer".to_string(),
            vec![(
                "blocks.0.norm.weight".to_string(),
                Tensor {
                    dtype: Dtype::Bf16,
                    shape: vec![128],
                    encoding: foreign.object_id(),
                    parts: vec![("value".to_string(), part_of(Dtype::Bf16, vec![128]))],
                },
            )],
        )],
    };
    cases.push(ok_case(
        g,
        "unaliased-spec-stores",
        "header",
        hf.conformance_bytes()?,
        "a pin-absent spec object validates as data",
    ));
    cases.push(red_case(
        g,
        "unaliased-spec-serving",
        "header-serving",
        hf.conformance_bytes()?,
        "UNKNOWN_TO_LOCAL_CAPABILITY",
        "the same bytes refused for SERVING against the local pin",
    ));
    specs.push(foreign);
    let mut h = hn.clone();
    h.components[0].1[0].1.shape = vec![256, 511];
    cases.push(red_case(
        g,
        "division-remainder",
        "header",
        h.conformance_bytes()?,
        "DIVISION_REMAINDER",
        "in/2 with an odd in",
    ));
    let mut h = ha.clone();
    h.components[0].1[1].1.shape = vec![67108864, 67108864];
    cases.push(red_case(
        g,
        "element-count-cap",
        "header",
        h.conformance_bytes()?,
        "COUNT_CAP",
        "aggregate element limit",
    ));
    // the element cap fires before any product can overflow, so the checked-arithmetic arm
    // is reached through a legal-but-extreme relation instead (probe spec, in the closure)
    let probe = probe_spec();
    let h = Header {
        configs: vec![],
        assets: vec![],
        encodings: encodings_of(&[&probe]),
        components: vec![(
            "transformer".to_string(),
            vec![(
                "probe.weight".to_string(),
                Tensor {
                    dtype: Dtype::Bf16,
                    shape: vec![4096, 8],
                    encoding: probe.object_id(),
                    parts: vec![(
                        "value".to_string(),
                        Part {
                            dtype: Dtype::U8,
                            shape: vec![1],
                            body: Body::Inline(vec![0u8]),
                        },
                    )],
                },
            )],
        )],
    };
    cases.push(red_case(
        g,
        "arith-overflow",
        "header",
        h.conformance_bytes()?,
        "ARITH_OVERFLOW",
        "checked 64-bit geometry arithmetic in the relation evaluator",
    ));
    let mut h = ha.clone();
    h.components[0].1[1].1.dtype = Dtype::I8;
    cases.push(red_case(
        g,
        "logical-dtype-carrier",
        "header",
        h.conformance_bytes()?,
        "DTYPE_MISMATCH",
        "carrier no longer equals the logical dtype",
    ));

    // A kind a newer TensorFS wrote is opaque here: it parses, and only materializing refuses.
    let manifest_bytes = snap2.canonical_bytes();
    cases.push(ok_case(
        "manifest",
        "entry-kind-future",
        "manifest",
        sub(&manifest_bytes, "\"kind\":\"file\"", "\"kind\":\"symlink\""),
        "an entry kind from a newer TensorFS is carried opaquely",
    ));

    // --- location-typed Manifest reds
    let mut draft = Draft::of(&snap);
    draft.entries.clear();
    cases.push(red_case(
        "manifest-red",
        "entries-empty",
        "manifest",
        draft.canonical_bytes(),
        "MISSING_FIELD",
        "a manifest lists at least one path",
    ));
    let mut draft = Draft::of(&snap);
    draft.entries.push((
        "../escape".to_string(),
        Entry::File(ObjectRef::of(b"escape")),
    ));
    draft.entries.sort_by(|left, right| left.0.cmp(&right.0));
    cases.push(red_case(
        "manifest-red",
        "path-dotdot",
        "manifest",
        draft.canonical_bytes(),
        "PATH_ILLEGAL",
        "dot components are unrepresentable",
    ));
    let mut draft = Draft::of(&snap);
    draft.entries.push((
        "model2.cozytensors".to_string(),
        Entry::CozyTensors(ha.object_ref()?),
    ));
    draft.entries.sort_by(|left, right| left.0.cmp(&right.0));
    cases.push(red_case(
        "manifest-red",
        "cozytensors-twice",
        "manifest",
        draft.canonical_bytes(),
        "ATTACHMENT_CARDINALITY",
        "one manifest carries at most one CozyTensors header",
    ));

    // --- the border's own documents (tfs-003)
    let g = "ingest";
    let session = tensorfs_core::ingest::transaction::IngestSession {
        tensorfs: None,
        session: "e3b0c44298fc1c14".to_string(),
        tenant: "acme".to_string(),
        candidates: vec![ObjectRef::of(b"a candidate header")],
    };
    cases.push(ok_case(
        g,
        "ingest-session",
        "session",
        session.canonical_bytes(),
        "a temporary session: a candidate is never a release",
    ));

    // --- ingest reds
    let g = "ingest-red";
    let mut future = session.clone();
    future.session = "not a session name".to_string();
    cases.push(red_case(
        g,
        "session-grammar",
        "session",
        future.canonical_bytes(),
        "KEY_GRAMMAR",
        "a session id is an ASCII name, never free text",
    ));
    let refused_dialect = profile_normal(vec![plain.object_ref()], "pickle");
    cases.push(red_case(
        g,
        "pickle-normal-profile",
        "profile",
        refused_dialect.canonical_bytes(),
        "PICKLE_REFUSED",
        "pickle always refuses normal ingest",
    ));
    let foreign = profile_normal(
        vec![ObjectRef::of(b"a spec the platform never aliased")],
        "safetensors.diffusers",
    );
    cases.push(red_case(
        g,
        "unregistered-encoding",
        "profile",
        foreign.canonical_bytes(),
        "UNREGISTERED_ENCODING",
        "launch border admits platform-aliased digests only",
    ));
    let two_arms = sub(
        &rcpt.canonical_bytes(),
        "\"evidence\":[",
        "\"evidence\":[{\"report\":{\"length\":6,\"sha256\":\"0000000000000000000000000000000000000000000000000000000000000000\"},\"t\":\"independent_numerical\",\"verifier_build\":\"sha256:0000000000000000000000000000000000000000000000000000000000000000\"},",
    );
    cases.push(red_case(
        g,
        "two-evidence-arms",
        "receipt",
        two_arms,
        "EVIDENCE_ARM_AMBIGUOUS",
        "exactly one evidence arm",
    ));

    // --- one-byte mutants of every positive
    let positives: Vec<(String, &'static str, Vec<u8>, String)> = cases
        .iter()
        .filter(|c| c.expect == "ok")
        .map(|c| {
            (
                format!("{}-{}", c.group, c.name),
                c.doc,
                c.bytes.clone(),
                tensorfs_core::ids::object_id(&c.bytes),
            )
        })
        .collect();
    for (name, doc, bytes, id) in positives {
        let mut m = bytes.clone();
        let off = m.len() / 3;
        m[off] ^= 0x01;
        cases.push(Case {
            group: "onebyte",
            name,
            doc,
            bytes: m,
            expect: "mutant",
            code: None,
            note: format!("one byte flipped at offset {off} of {id}"),
        });
    }

    Ok((cases, specs))
}

// ------------------------------------------------------------------ gen

fn generate(dir: &Path) {
    let (cases, specs) = build_corpus().unwrap();
    // Regenerate only what the corpus OWNS. `mined/` (producer-mined vectors) and
    // `structure/` (banked fixtures) are evidence with their own provenance and their own
    // banking commands; a corpus rebuild must never be able to delete them.
    let _ = fs::remove_dir_all(dir.join("cases"));
    let _ = fs::remove_dir_all(dir.join("closure"));
    fs::create_dir_all(dir.join("closure")).unwrap();
    for s in &specs {
        let id = s.object_id();
        fs::write(
            dir.join("closure").join(format!("{}.cbor", &id[7..19])),
            s.canonical_bytes(),
        )
        .unwrap();
    }
    let mut n = 0;
    for c in &cases {
        let d = dir.join("cases").join(c.group);
        fs::create_dir_all(&d).unwrap();
        let extension = if matches!(c.doc, "header" | "header-serving" | "encoding") {
            "cbor"
        } else {
            "json"
        };
        fs::write(d.join(format!("{}.{}", c.name, extension)), &c.bytes).unwrap();
        let mut v = format!("doc: {}\nexpect: {}\n", c.doc, c.expect);
        if let Some(code) = &c.code {
            v += &format!("code: {code}\n");
        }
        if c.expect == "ok" {
            v += &format!("id: {}\n", tensorfs_core::ids::object_id(&c.bytes));
            if c.doc == "header" {
                let h = Header::parse(&c.bytes).unwrap();
                v += &format!("tensor_schema: {}\n", h.tensor_schema_digest());
            }
            if c.doc == "manifest" {
                let m = Manifest::parse(&c.bytes).unwrap();
                v += &format!("manifest: {}\n", m.manifest_id());
            }
        }
        v += &format!("note: {}\n", c.note);
        fs::write(d.join(format!("{}.verdict", c.name)), v).unwrap();
        n += 1;
    }
    println!(
        "wrote {n} vectors and {} closure spec objects to {}",
        specs.len(),
        dir.display()
    );
}

// ------------------------------------------------------------------ verify

enum Outcome {
    Ok {
        id: String,
        tensor_schema: Option<String>,
        manifest: Option<String>,
    },
    Refused(Refusal),
}

fn run_doc(doc: &str, bytes: &[u8], closure: &Closure, platform: &[String]) -> Outcome {
    // ONE dispatch, shared with the Python facade's harness (tensorfs_core::corpus).
    match tensorfs_core::corpus::run_doc(doc, bytes, closure, platform) {
        Ok(o) => Outcome::Ok {
            id: o.id,
            tensor_schema: o.tensor_schema,
            manifest: o.manifest,
        },
        Err(e) => Outcome::Refused(e),
    }
}

fn read_verdict(p: &Path) -> Vec<(String, String)> {
    fs::read_to_string(p)
        .unwrap()
        .lines()
        .filter_map(|l| {
            l.split_once(": ")
                .map(|(a, b)| (a.to_string(), b.to_string()))
        })
        .collect()
}

fn field(v: &[(String, String)], k: &str) -> Option<String> {
    v.iter().find(|(a, _)| a == k).map(|(_, b)| b.clone())
}

fn verify(dir: &Path) -> bool {
    let started = Instant::now();
    let mut closure = Closure::default();
    let mut closure_ok = true;
    for e in fs::read_dir(dir.join("closure")).expect("closure dir") {
        let p = e.unwrap().path();
        let bytes = fs::read(&p).unwrap();
        match EncodingSpec::decode_nested(&bytes) {
            Ok(s) => closure.insert(s),
            Err(err) => {
                println!("FAIL closure/{}: {err}", p.display());
                closure_ok = false;
            }
        }
    }
    let platform = registry::platform_digests();

    let mut groups: Vec<PathBuf> = fs::read_dir(dir.join("cases"))
        .expect("cases dir")
        .map(|e| e.unwrap().path())
        .collect();
    groups.sort();
    let (mut pass, mut fail) = (0, 0);
    let mut ids: Vec<(String, Outcome)> = Vec::new();

    for g in groups {
        let mut files: Vec<PathBuf> = fs::read_dir(&g)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| {
                p.extension()
                    .map(|x| x == "json" || x == "cbor")
                    .unwrap_or(false)
            })
            .collect();
        files.sort();
        for f in files {
            let name = format!(
                "{}/{}",
                g.file_name().unwrap().to_string_lossy(),
                f.file_stem().unwrap().to_string_lossy()
            );
            let v = read_verdict(&f.with_extension("verdict"));
            let bytes = fs::read(&f).unwrap();
            let doc = field(&v, "doc").expect("doc kind");
            let expect = field(&v, "expect").expect("expect");
            let outcome = run_doc(&doc, &bytes, &closure, &platform);
            let verdict = match (expect.as_str(), &outcome) {
                (
                    "ok",
                    Outcome::Ok {
                        id,
                        tensor_schema,
                        manifest,
                    },
                ) => {
                    let mut bad = Vec::new();
                    if field(&v, "id").as_deref() != Some(id.as_str()) {
                        bad.push(format!("id {id} != frozen {:?}", field(&v, "id")));
                    }
                    if let Some(expected) = field(&v, "tensor_schema") {
                        if tensor_schema.as_deref() != Some(expected.as_str()) {
                            bad.push(format!(
                                "tensor schema {tensor_schema:?} != frozen {expected}"
                            ));
                        }
                    }
                    if let Some(expected) = field(&v, "manifest") {
                        if manifest.as_deref() != Some(expected.as_str()) {
                            bad.push(format!("manifest {manifest:?} != frozen {expected}"));
                        }
                    }
                    if bad.is_empty() {
                        Ok(format!("id={}", &id[7..19]))
                    } else {
                        Err(bad.join("; "))
                    }
                }
                ("ok", Outcome::Refused(e)) => Err(format!("expected ok, refused {e}")),
                ("refuse", Outcome::Refused(e)) => {
                    let want = field(&v, "code").unwrap_or_default();
                    if e.code.as_str() == want {
                        Ok(format!("{} ({})", e.code.as_str(), truncate(&e.detail, 72)))
                    } else {
                        Err(format!(
                            "refused {} but the frozen code is {want}",
                            e.code.as_str()
                        ))
                    }
                }
                ("refuse", Outcome::Ok { .. }) => Err(format!(
                    "expected {} but it passed",
                    field(&v, "code").unwrap_or_default()
                )),
                ("mutant", Outcome::Refused(e)) => Ok(format!("refused {}", e.code.as_str())),
                ("mutant", Outcome::Ok { id, .. }) => {
                    let from = field(&v, "note").unwrap_or_default();
                    if from.contains(id.as_str()) {
                        Err("a flipped byte kept the same object id".to_string())
                    } else {
                        Ok(format!("distinct id={}", &id[7..19]))
                    }
                }
                (e, _) => Err(format!("unknown expectation {e:?}")),
            };
            match verdict {
                Ok(d) => {
                    pass += 1;
                    println!("PASS {name:<44} {expect:<7} {d}");
                }
                Err(d) => {
                    fail += 1;
                    println!("FAIL {name:<44} {expect:<7} {d}");
                }
            }
            ids.push((name, outcome));
        }
    }

    // cross-vector identity assertions
    let get = |n: &str| -> Option<&Outcome> { ids.iter().find(|(k, _)| k == n).map(|(_, o)| o) };
    let idof = |n: &str| -> String {
        match get(n) {
            Some(Outcome::Ok { id, .. }) => id.clone(),
            _ => String::new(),
        }
    };
    let schemaof = |n: &str| -> String {
        match get(n) {
            Some(Outcome::Ok {
                tensor_schema: Some(schema),
                ..
            }) => schema.clone(),
            _ => String::new(),
        }
    };
    let cpof = |n: &str| -> String {
        match get(n) {
            Some(Outcome::Ok {
                manifest: Some(c), ..
            }) => c.clone(),
            _ => String::new(),
        }
    };
    let short = |value: String| value.chars().take(19).collect::<String>();
    let mut cross: Vec<(String, bool, String)> = Vec::new();
    cross.push((
        "manifest_id == manifest_id (unconditional)".into(),
        cpof("manifest/checkpoint") == idof("manifest/checkpoint")
            && !cpof("manifest/checkpoint").is_empty(),
        cpof("manifest/checkpoint"),
    ));
    cross.push((
        "header ObjectId != manifest_id".into(),
        idof("header/plain-fp8") != cpof("manifest/checkpoint"),
        format!(
            "{} vs {}",
            short(idof("header/plain-fp8")),
            short(cpof("manifest/checkpoint"))
        ),
    ));
    cross.push((
        "a config change flips header id, not tensor_schema_digest".into(),
        idof("header/plain-fp8") != idof("header/plain-fp8-noconfig")
            && schemaof("header/plain-fp8") == schemaof("header/plain-fp8-noconfig"),
        schemaof("header/plain-fp8"),
    ));
    cross.push((
        "tensor reorder flips header id, not tensor_schema_digest".into(),
        idof("header/plain-fp8") != idof("header/plain-fp8-reordered")
            && schemaof("header/plain-fp8") == schemaof("header/plain-fp8-reordered"),
        schemaof("header/plain-fp8-reordered"),
    ));
    cross.push((
        "one more sibling file flips manifest_id only".into(),
        cpof("manifest/checkpoint") != cpof("manifest/checkpoint-extra-file"),
        cpof("manifest/checkpoint-extra-file"),
    ));
    let (cha, _) = header_a(Some(config_bytes()));
    let csnap = ok!(manifest_checkpoint(&cha, false));
    let cprof = profile_normal(
        vec![seed_spec("plain/1", 0).object_ref()],
        "safetensors.diffusers",
    );
    let csub = ok!(subject(&cprof, &cha, "acme"));
    let cstamp = ok!(stamp_doc(&cprof, &csub, &csnap, &cha));
    let crcpt = ok!(receipt(&csub, &csnap, &cha, &cstamp, "platform-key-1"));
    cross.push((
        "receipt binds its exact subject/manifest/header/stamp".into(),
        crcpt.binds(&csub, &csnap, &cha, &cstamp).is_ok(),
        String::new(),
    ));
    let other = ok!(subject(&cprof, &cha, "other-tenant"));
    let replay = crcpt.binds(&other, &csnap, &cha, &cstamp);
    cross.push((
        "cross-subject replay refuses".into(),
        replay.as_ref().err().map(|e| e.code)
            == Some(tensorfs_core::err::Code::CROSS_SUBJECT_REPLAY),
        replay.err().map(|e| e.detail).unwrap_or_default(),
    ));
    let seeds_match = registry::seeds()
        .iter()
        .all(|s| closure.get(&s.spec.object_id()).is_some());
    cross.push((
        "every seed digest resolves in the corpus closure".into(),
        seeds_match,
        String::new(),
    ));

    let case_bytes = |rel: &str| fs::read(dir.join("cases").join(rel)).unwrap_or_default();
    let records = registry::capability_records();
    let plain_id = seed_spec("plain/1", 0).object_id();
    let admitted = records
        .admit(&plain_id, "cpu")
        .expect("plain/1 is qualified on the reference CPU");
    let refused = records
        .admit(&plain_id, "cuda.sm89")
        .expect_err("no accelerator class is compiled in");
    for (name, live) in [
        (
            "capability-admitted",
            tensorfs_core::machine::capability_json(Some(admitted), &plain_id, "cpu", "", &["cpu"]),
        ),
        (
            "capability-unqualified",
            tensorfs_core::machine::capability_json(
                None,
                &plain_id,
                "cuda.sm89",
                &refused.detail,
                &["cpu"],
            ),
        ),
    ] {
        let frozen = case_bytes(&format!("machine/{name}.json"));
        if live == frozen {
            pass += 1;
            println!("PASS machine/{name:<36} exact result");
        } else {
            fail += 1;
            println!("FAIL machine/{name:<36} result moved");
        }
    }

    for (what, ok, detail) in &cross {
        if *ok {
            pass += 1;
            println!("PASS cross: {what:<52} {detail}");
        } else {
            fail += 1;
            println!("FAIL cross: {what:<52} {detail}");
        }
    }

    println!(
        "\n{pass} pass, {fail} fail, {} closure specs, {:.3}s",
        closure.specs.len(),
        started.elapsed().as_secs_f64()
    );
    fail == 0 && closure_ok
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n])
    }
}

fn files(root: &Path) -> Vec<PathBuf> {
    let mut out = vec![];
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push(path.strip_prefix(root).unwrap().to_path_buf());
            }
        }
    }
    out.sort();
    out
}

#[test]
fn every_committed_vector_answers_its_verdict() {
    assert!(verify(&default_dir()));
}

#[test]
fn the_committed_corpus_is_what_the_builder_writes() {
    let built = std::env::temp_dir().join(format!("tensorfs-vectors-{}", std::process::id()));
    let _ = fs::remove_dir_all(&built);
    generate(&built);
    for owned in ["cases", "closure"] {
        let (fresh, committed) = (built.join(owned), default_dir().join(owned));
        assert_eq!(files(&fresh), files(&committed), "{owned}/ file set");
        for file in files(&fresh) {
            assert!(
                fs::read(fresh.join(&file)).unwrap() == fs::read(committed.join(&file)).unwrap(),
                "{owned}/{} differs from what the builder writes",
                file.display()
            );
        }
    }
    fs::remove_dir_all(built).unwrap();
}

/// Rewrites `vectors/cases` and `vectors/closure`; run explicitly after an intended change.
#[test]
#[ignore = "rewrites the committed corpus"]
fn regenerate() {
    generate(&default_dir());
}

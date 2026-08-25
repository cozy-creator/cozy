//! tfs — the runnable conformance/benchmark tool for tfs-013.
//!
//!   tfs registry                 print the seeded platform namespace (alias -> digest)
//!   tfs vectors gen [dir]        (re)write the frozen golden-vector corpus
//!   tfs vectors verify [dir]     run every vector live, print per-vector verdicts
//!   tfs bench [tensors] [reps]   canonical-writer throughput + peak RSS
//!   tfs flip                     identity-flip matrix over live documents
//!   tfs id <file>                object id (+ derived ids) of a stored document

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use tensorfs_core::canon::{self, Value};
use tensorfs_core::dtype::Dtype;
use tensorfs_core::err::Refusal;
use tensorfs_core::header::{Body, Closure, Header, Part, Tensor};
use tensorfs_core::ids::{ConfigRef, Doc, ObjectRef};
use tensorfs_core::ingest::{Evidence, IngestProfile, IngestSubject, IngestVerificationReceipt};
use tensorfs_core::manifest::{BlobDescriptor, Entry, SnapshotManifest};
use tensorfs_core::registry;
use tensorfs_core::spec::EncodingSpec;
use tensorfs_core::vectors::EncodingVectors;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let a: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    match a.as_slice() {
        ["registry"] => cmd_registry(),
        ["vectors", "gen"] => cmd_gen(&default_dir()),
        ["vectors", "gen", d] => cmd_gen(Path::new(d)),
        ["vectors", "verify"] => cmd_verify(&default_dir()),
        ["vectors", "verify", d] => cmd_verify(Path::new(d)),
        ["bench"] => cmd_bench(2000, 20),
        ["bench", n] => cmd_bench(n.parse().unwrap_or(2000), 20),
        ["bench", n, r] => cmd_bench(n.parse().unwrap_or(2000), r.parse().unwrap_or(20)),
        ["flip"] => cmd_flip(),
        ["id", f] => cmd_id(Path::new(f)),
        _ => {
            eprintln!("usage: tfs registry | vectors gen|verify [dir] | bench [n] [reps] | flip | id <file>");
            ExitCode::from(2)
        }
    }
}

fn default_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vectors")
}

// ------------------------------------------------------------------ registry

fn cmd_registry() -> ExitCode {
    let seeds = registry::seeds();
    println!(
        "platform namespace: {} seed specs (identity = digest; names are aliases)",
        seeds.len()
    );
    for s in &seeds {
        let roles: Vec<&str> = s.spec.roles.iter().map(|(n, _)| n.as_str()).collect();
        println!(
            "  {:<28} {}  roles={:?} {}",
            s.alias,
            s.spec.object_id(),
            roles,
            match s.spec.require_executable() {
                Ok(v) => format!("vectors={}", v.id()),
                Err(e) => format!("VECTORLESS ({} — store/inspect only)", e.code.as_str()),
            }
        );
    }
    ExitCode::SUCCESS
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

fn pattern(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i * 31 + 7) as u8).collect()
}

fn part_of(dt: Dtype, shape: Vec<u64>) -> Part {
    let bytes = pattern(shape.iter().product::<u64>() as usize * dt.size() as usize);
    Part::plan(dt, shape, &bytes)
}

fn config_ref() -> ConfigRef {
    let bytes = br#"{"hidden_size":1536}"#;
    ConfigRef {
        sha256: tensorfs_core::sha256::hex_digest(bytes),
        length: bytes.len() as u64,
        media_type: "application/json".to_string(),
    }
}

fn encodings_of(specs: &[&EncodingSpec]) -> Vec<ObjectRef> {
    let mut v: Vec<ObjectRef> = specs.iter().map(|s| s.object_ref()).collect();
    v.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    v
}

/// Positive header: plain inline tensor + an fp8-rowwise tensor (segments AND inline).
fn header_a(config: Option<ConfigRef>) -> (Header, Closure) {
    let plain = seed_spec("plain/1", 0);
    let fp8 = seed_spec("fp8-rowwise/1", 0);
    let mut closure = Closure::default();
    let h = Header {
        configs: config
            .map(|c| vec![("transformer".to_string(), c)])
            .unwrap_or_default(),
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

fn manifest_checkpoint(header: &Header, extra_file: bool) -> SnapshotManifest {
    let blob = BlobDescriptor::of(&pattern(4096));
    let mut entries = vec![
        ("tokenizer.model".to_string(), Entry::Blob(blob)),
        ("transformer".to_string(), Entry::Dir),
        (
            "transformer/config.json".to_string(),
            Entry::File(ObjectRef::of(br#"{"hidden_size":1536}"#)),
        ),
    ];
    if extra_file {
        entries.push((
            "transformer/generation.json".to_string(),
            Entry::File(ObjectRef::of(br#"{"steps":30}"#)),
        ));
    }
    SnapshotManifest {
        header: Some(header.object_ref()),
        entries,
    }
}

fn profile_normal(allowed: Vec<ObjectRef>, dialect: &str) -> IngestProfile {
    IngestProfile {
        class: "normal".to_string(),
        source_dialect: dialect.to_string(),
        converter_build: tensorfs_core::ids::object_id(b"converter-build-v1"),
        converter_recipe: ObjectRef::of(b"recipe"),
        expected_topology_digest: tensorfs_core::ids::object_id(b"topology"),
        allowed_encodings: allowed,
        max_source_bytes: 1 << 40,
        max_tensors: 100_000,
    }
}

fn subject(profile: &IngestProfile, header: &Header, tenant: &str) -> IngestSubject {
    IngestSubject {
        profile: profile.object_ref(),
        tenant: tenant.to_string(),
        sources: vec![(tensorfs_core::ids::object_id(b"hf://acme/model@rev1"), 1)],
        proposed_header: header.object_ref(),
        candidates: vec![ObjectRef::of(b"candidate")],
    }
}

fn receipt(
    sub: &IngestSubject,
    snap: &SnapshotManifest,
    header: &Header,
    signer: &str,
) -> IngestVerificationReceipt {
    IngestVerificationReceipt {
        subject: sub.object_ref(),
        snapshot: snap.object_ref(),
        header: header.object_ref(),
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
    }
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

/// A deliberately non-canonical twin: same content, reversed object key order.
fn reversed(v: &Value) -> Vec<u8> {
    fn emit(v: &Value, out: &mut Vec<u8>) {
        match v {
            Value::Obj(m) => {
                out.push(b'{');
                for (i, (k, x)) in m.iter().rev().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    out.extend_from_slice(&canon::write(&Value::Str(k.clone())));
                    out.push(b':');
                    emit(x, out);
                }
                out.push(b'}');
            }
            Value::Arr(a) => {
                out.push(b'[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    emit(x, out);
                }
                out.push(b']');
            }
            other => out.extend_from_slice(&canon::write(other)),
        }
    }
    let mut out = Vec::new();
    emit(v, &mut out);
    out
}

fn build_corpus() -> (Vec<Case>, Vec<EncodingSpec>) {
    let mut cases: Vec<Case> = Vec::new();
    let (ha, closure) = header_a(Some(config_ref()));
    let ha_bytes = ha.canonical_bytes();
    let (hn, _) = header_nvfp4();
    let plain = seed_spec("plain/1", 0);
    let fp8 = seed_spec("fp8-rowwise/1", 0);
    let mxfp8 = seed_spec("mxfp8/1", 0);
    let svdq = seed_spec("svdq-microscale/1", 0);
    // the corpus closure ships the WHOLE platform namespace, not only the cited specs
    let mut specs: Vec<EncodingSpec> = registry::seeds().into_iter().map(|s| s.spec).collect();
    for (_, s) in closure.specs {
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
        hn.canonical_bytes(),
        "nvfp4 W4A4 base variant, rank-0 scalar role",
    ));
    let (ha2, _) = header_a(None);
    cases.push(ok_case(
        "header",
        "plain-fp8-noconfig",
        "header",
        ha2.canonical_bytes(),
        "same tensors, no configs: same topology, different header id",
    ));
    cases.push(ok_case(
        "spec",
        "plain",
        "spec",
        plain.canonical_bytes(),
        "entry zero; carries vectors",
    ));
    cases.push(ok_case(
        "spec",
        "mxfp8",
        "spec",
        mxfp8.canonical_bytes(),
        "ceil-block scale relation; vectorless",
    ));
    cases.push(ok_case(
        "spec",
        "svdq-microscale",
        "spec",
        svdq.canonical_bytes(),
        "rank-7 permuted scale grid mined from v1",
    ));
    cases.push(ok_case(
        "encoding_vectors",
        "plain",
        "vectors",
        registry::plain_vectors().canonical_bytes(),
        "raw bit patterns: signed zero, NaN payload, infinity, rank-0",
    ));

    let snap = manifest_checkpoint(&ha, false);
    let snap2 = manifest_checkpoint(&ha, true);
    cases.push(ok_case(
        "manifest",
        "checkpoint",
        "manifest",
        snap.canonical_bytes(),
        "checkpoint_id = snapshot_id, header attached structurally",
    ));
    cases.push(ok_case(
        "manifest",
        "checkpoint-extra-file",
        "manifest",
        snap2.canonical_bytes(),
        "one more sibling file: new checkpoint_id, same header id",
    ));

    let prof = profile_normal(
        vec![plain.object_ref(), fp8.object_ref()],
        "safetensors.diffusers",
    );
    let sub_doc = subject(&prof, &ha, "acme");
    let rcpt = receipt(&sub_doc, &snap, &ha, "platform-key-1");
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
        "receipt",
        "receipt",
        rcpt.canonical_bytes(),
        "one evidence arm; signer in the envelope",
    ));

    // --- canonical-byte semantic twins (all red)
    let g = "canonical";
    cases.push(red_case(
        g,
        "whitespace",
        "header",
        sub(&ha_bytes, "{\"components\"", "{ \"components\""),
        "NONCANONICAL_ENCODING",
        "one inserted space",
    ));
    cases.push(red_case(
        g,
        "key-order",
        "header",
        reversed(&ha.to_value()),
        "NONCANONICAL_ENCODING",
        "reversed object key order",
    ));
    cases.push(red_case(
        g,
        "duplicate-key",
        "header",
        sub(
            &ha_bytes,
            "\"format\":\"cozytensors/1\"",
            "\"format\":\"cozytensors/1\",\"format\":\"cozytensors/1\"",
        ),
        "DUPLICATE_KEY",
        "same key twice",
    ));
    cases.push(red_case(
        g,
        "number-float",
        "header",
        sub(&ha_bytes, "\"length\":8192", "\"length\":8192.0"),
        "NON_INTEGER_NUMBER",
        "float spelling of an integer",
    ));
    cases.push(red_case(
        g,
        "number-exponent",
        "header",
        sub(&ha_bytes, "\"length\":8192", "\"length\":8.192e3"),
        "NON_INTEGER_NUMBER",
        "exponent spelling",
    ));
    cases.push(red_case(
        g,
        "number-leading-zero",
        "header",
        sub(&ha_bytes, "\"length\":8192", "\"length\":08192"),
        "NON_INTEGER_NUMBER",
        "non-canonical integer spelling",
    ));
    cases.push(red_case(
        g,
        "null-value",
        "header",
        sub(&ha_bytes, "\"length\":8192", "\"length\":null"),
        "WRONG_TYPE",
        "null is not admitted",
    ));
    cases.push(red_case(
        g,
        "escaped-ascii",
        "header",
        sub(&ha_bytes, "\"format\"", "\"form\\u0061t\""),
        "NONCANONICAL_ENCODING",
        "\\u escape of a plain letter",
    ));
    cases.push(red_case(
        g,
        "unicode-nfc",
        "header",
        sub(&ha_bytes, "transformer", "transf\u{00f6}rmer"),
        "NON_ASCII_FIELD",
        "NFC o-diaeresis in a component name",
    ));
    cases.push(red_case(
        g,
        "unicode-nfd",
        "header",
        sub(&ha_bytes, "transformer", "transfo\u{0308}rmer"),
        "NON_ASCII_FIELD",
        "NFD twin of the same name",
    ));
    cases.push(red_case(
        g,
        "digest-uppercase",
        "header",
        uppercase_digest(&ha_bytes),
        "MALFORMED_DIGEST",
        "uppercase hex in a digest",
    ));
    cases.push(red_case(
        g,
        "trailing-whitespace",
        "header",
        [ha_bytes.clone(), b"\n".to_vec()].concat(),
        "NONCANONICAL_ENCODING",
        "trailing newline: parses, then fails the re-emit law",
    ));
    cases.push(red_case(
        g,
        "trailing-bytes",
        "header",
        [ha_bytes.clone(), b"{}".to_vec()].concat(),
        "TRAILING_BYTES",
        "a second value after the document",
    ));

    // --- header schema/geometry reds
    let g = "header-red";
    cases.push(red_case(
        g,
        "unknown-field",
        "header",
        sub(&ha_bytes, ",\"format\"", ",\"extra\":1,\"format\""),
        "UNKNOWN_FIELD",
        "unknown root field",
    ));
    cases.push(red_case(
        g,
        "unknown-major",
        "header",
        sub(&ha_bytes, "cozytensors/1", "cozytensors/2"),
        "UNKNOWN_FORMAT",
        "unknown format major",
    ));
    cases.push(red_case(
        g,
        "anchor-equation",
        "header",
        sub(&ha_bytes, "\"length\":8192", "\"length\":8191"),
        "BYTE_LENGTH_MISMATCH",
        "segment total off by one byte",
    ));
    cases.push(red_case(
        g,
        "dtype-outside-enum",
        "header",
        sub(
            &ha_bytes,
            "\"dtype\":\"f8_e4m3fn\"",
            "\"dtype\":\"f4_e2m1\"",
        ),
        "DTYPE_UNKNOWN",
        "carrier dtype outside the closed enum",
    ));

    let mut h = ha.clone();
    h.components[0].1[0].1.parts.retain(|(n, _)| n != "scale");
    cases.push(red_case(
        g,
        "role-missing",
        "header",
        h.canonical_bytes(),
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
        h.canonical_bytes(),
        "ROLE_SET_MISMATCH",
        "an extra role is a different spec digest",
    ));
    let mut h = ha.clone();
    h.components[0].1[0].1.parts[1].1.shape = vec![32];
    cases.push(red_case(
        g,
        "relation-shape",
        "header",
        h.canonical_bytes(),
        "SHAPE_MISMATCH",
        "scale shape != relation result",
    ));
    let mut h = ha.clone();
    h.components[0].1[1].1.shape = vec![0];
    cases.push(red_case(
        g,
        "zero-element",
        "header",
        h.canonical_bytes(),
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
        h.canonical_bytes(),
        "INLINE_THRESHOLD",
        "256 B stored as a segment",
    ));
    let mut h = ha.clone();
    h.components[0].1[1].1.shape = vec![130];
    h.components[0].1[1].1.parts[0].1 = Part {
        dtype: Dtype::Bf16,
        shape: vec![130],
        body: Body::Inline(tensorfs_core::b64::encode(&pattern(260))),
    };
    cases.push(red_case(
        g,
        "inline-threshold-over",
        "header",
        h.canonical_bytes(),
        "INLINE_THRESHOLD",
        "260 B stored inline",
    ));
    let inline_anchor = format!(
        "\"inline\":\"{}\"",
        tensorfs_core::b64::encode(&pattern(256))
    );
    cases.push(red_case(
        g,
        "inline-and-segments",
        "header",
        sub(
            &ha_bytes,
            &inline_anchor,
            &format!(
                "{inline_anchor},\"segments\":[{{\"length\":256,\"sha256\":\"{}\"}}]",
                "0".repeat(64)
            ),
        ),
        "INLINE_EXCLUSIVE",
        "both bodies present",
    ));
    let mut h = ha.clone();
    h.encodings.push(ObjectRef::of(b"an uncited spec object"));
    h.encodings.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    cases.push(red_case(
        g,
        "uncited-encoding",
        "header",
        h.canonical_bytes(),
        "UNCITED_ENCODING",
        "derivable-fact law",
    ));
    let mut h = ha.clone();
    h.components[0].1[1].1.encoding = tensorfs_core::ids::object_id(b"a spec nobody shipped");
    h.encodings.push(ObjectRef::of(b"a spec nobody shipped"));
    h.encodings.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    cases.push(red_case(
        g,
        "dangling-spec",
        "header",
        h.canonical_bytes(),
        "UNKNOWN_ENCODING",
        "cited digest absent from the closure",
    ));
    let mut h = ha.clone();
    h.components[0].1[1].1.encoding = tensorfs_core::ids::object_id(b"not listed either");
    cases.push(red_case(
        g,
        "encoding-not-listed",
        "header",
        h.canonical_bytes(),
        "ENCODING_MISMATCH",
        "cited digest missing from `encodings`",
    ));
    let mut h = ha.clone();
    h.components.clear();
    cases.push(red_case(
        g,
        "empty-components",
        "header",
        h.canonical_bytes(),
        "EMPTY_COMPONENTS",
        "no components",
    ));
    let mut h = hn.clone();
    h.components[0].1[0].1.shape = vec![256, 511];
    cases.push(red_case(
        g,
        "division-remainder",
        "header",
        h.canonical_bytes(),
        "DIVISION_REMAINDER",
        "in/2 with an odd in",
    ));
    let mut h = ha.clone();
    h.components[0].1[1].1.dtype = Dtype::I8;
    cases.push(red_case(
        g,
        "logical-dtype-carrier",
        "header",
        h.canonical_bytes(),
        "DTYPE_MISMATCH",
        "carrier no longer equals the logical dtype",
    ));

    // --- spec reds
    let g = "spec-red";
    let plain_bytes = plain.canonical_bytes();
    cases.push(red_case(
        g,
        "optional-role",
        "spec",
        sub(
            &plain_bytes,
            "\"shape\":{\"t\":\"same\"}",
            "\"optional\":true,\"shape\":{\"t\":\"same\"}",
        ),
        "OPTIONAL_ROLE_FORBIDDEN",
        "no optional roles: one digest, one byte contract",
    ));
    cases.push(red_case(
        g,
        "unknown-relation",
        "spec",
        sub(&plain_bytes, "\"t\":\"same\"", "\"t\":\"stride\""),
        "UNKNOWN_RELATION",
        "tag outside the closed grammar",
    ));
    let mxfp8_bytes = mxfp8.canonical_bytes();
    cases.push(red_case(
        g,
        "missing-logical-rank",
        "spec",
        sub(&mxfp8_bytes, ",\"logical_rank\":2", ""),
        "MISSING_FIELD",
        "dims relations require a pinned logical rank",
    ));
    cases.push(red_case(
        g,
        "unsorted-logical-dtypes",
        "spec",
        sub(
            &mxfp8_bytes,
            "[\"bf16\",\"f16\",\"f32\"]",
            "[\"f32\",\"bf16\",\"f16\"]",
        ),
        "SORT_ORDER",
        "closed set must be sorted unique",
    ));

    // --- manifest reds
    let g = "manifest-red";
    let mut m = snap.clone();
    m.entries.swap(0, 1);
    cases.push(red_case(
        g,
        "path-order",
        "manifest",
        m.canonical_bytes(),
        "PATH_ORDER",
        "entries must strictly ascend by bytes",
    ));
    let mut m = snap.clone();
    m.entries.retain(|(p, _)| p != "transformer");
    cases.push(red_case(
        g,
        "parent-missing",
        "manifest",
        m.canonical_bytes(),
        "PATH_PARENT_MISSING",
        "parents-before-children",
    ));
    let mut m = snap.clone();
    m.entries.push((
        "transformer/CONFIG.json".to_string(),
        Entry::File(ObjectRef::of(b"x")),
    ));
    m.entries.sort_by(|a, b| a.0.cmp(&b.0));
    cases.push(red_case(
        g,
        "case-collision",
        "manifest",
        m.canonical_bytes(),
        "PATH_CASE_COLLISION",
        "case-fold collision inside one tree",
    ));
    let mut m = snap.clone();
    m.entries[0].0 = "../escape.bin".to_string();
    m.entries.sort_by(|a, b| a.0.cmp(&b.0));
    cases.push(red_case(
        g,
        "path-dotdot",
        "manifest",
        m.canonical_bytes(),
        "PATH_ILLEGAL",
        "dot component",
    ));
    let mut m = snap.clone();
    m.entries[0].0 = "nul.bin".to_string();
    m.entries.sort_by(|a, b| a.0.cmp(&b.0));
    cases.push(red_case(
        g,
        "windows-reserved",
        "manifest",
        m.canonical_bytes(),
        "PATH_RESERVED",
        "reserved stem",
    ));
    let mut m = snap.clone();
    if let Entry::Blob(b) = &mut m.entries[0].1 {
        b.logical_size += 1;
    }
    cases.push(red_case(
        g,
        "blob-length",
        "manifest",
        m.canonical_bytes(),
        "BYTE_LENGTH_MISMATCH",
        "segment total != logical size",
    ));

    // --- ingest reds
    let g = "ingest-red";
    let pickle = profile_normal(vec![plain.object_ref()], "pickle");
    cases.push(red_case(
        g,
        "pickle-normal-profile",
        "profile",
        pickle.canonical_bytes(),
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

    (cases, specs)
}

fn uppercase_digest(bytes: &[u8]) -> Vec<u8> {
    let s = String::from_utf8(bytes.to_vec()).unwrap();
    let i = s.find("\"sha256\":\"").unwrap() + 10;
    let mut out = s.into_bytes();
    for b in out[i..i + 8].iter_mut() {
        b.make_ascii_uppercase();
    }
    out
}

// ------------------------------------------------------------------ gen

fn cmd_gen(dir: &Path) -> ExitCode {
    let (cases, specs) = build_corpus();
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir.join("closure")).unwrap();
    for s in &specs {
        let id = s.object_id();
        fs::write(
            dir.join("closure").join(format!("{}.json", &id[7..19])),
            s.canonical_bytes(),
        )
        .unwrap();
    }
    let mut n = 0;
    for c in &cases {
        let d = dir.join("cases").join(c.group);
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join(format!("{}.json", c.name)), &c.bytes).unwrap();
        let mut v = format!("doc: {}\nexpect: {}\n", c.doc, c.expect);
        if let Some(code) = &c.code {
            v += &format!("code: {code}\n");
        }
        if c.expect == "ok" {
            v += &format!("id: {}\n", tensorfs_core::ids::object_id(&c.bytes));
            if c.doc == "header" {
                let h = Header::parse(&c.bytes).unwrap();
                v += &format!("topology: {}\n", h.topology_digest());
            }
            if c.doc == "manifest" {
                let m = SnapshotManifest::parse(&c.bytes).unwrap();
                if let Ok(cid) = m.checkpoint_id() {
                    v += &format!("checkpoint: {cid}\n");
                }
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
    ExitCode::SUCCESS
}

// ------------------------------------------------------------------ verify

enum Outcome {
    Ok {
        id: String,
        topology: Option<String>,
        checkpoint: Option<String>,
    },
    Refused(Refusal),
}

fn run_doc(doc: &str, bytes: &[u8], closure: &Closure, platform: &[String]) -> Outcome {
    let id = tensorfs_core::ids::object_id(bytes);
    let r: Result<(Option<String>, Option<String>), Refusal> = match doc {
        "header" => Header::parse(bytes).and_then(|h| {
            h.validate(closure)?;
            Ok((Some(h.topology_digest()), None))
        }),
        "spec" => EncodingSpec::parse(bytes).map(|_| (None, None)),
        "vectors" => EncodingVectors::parse(bytes).and_then(|v| {
            let plain = seed_spec("plain/1", 0);
            v.check(&plain)?;
            Ok((None, None))
        }),
        "manifest" => SnapshotManifest::parse(bytes).and_then(|m| {
            m.validate()?;
            Ok((None, m.checkpoint_id().ok()))
        }),
        "profile" => IngestProfile::parse(bytes).and_then(|p| {
            p.validate(platform)?;
            Ok((None, None))
        }),
        "subject" => IngestSubject::parse(bytes).map(|_| (None, None)),
        "receipt" => IngestVerificationReceipt::parse(bytes).map(|_| (None, None)),
        other => panic!("unknown doc kind {other}"),
    };
    match r {
        Ok((topology, checkpoint)) => Outcome::Ok {
            id,
            topology,
            checkpoint,
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

fn cmd_verify(dir: &Path) -> ExitCode {
    let started = Instant::now();
    let mut closure = Closure::default();
    let mut closure_ok = true;
    for e in fs::read_dir(dir.join("closure")).expect("closure dir") {
        let p = e.unwrap().path();
        let bytes = fs::read(&p).unwrap();
        match EncodingSpec::parse(&bytes) {
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
            .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
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
                        topology,
                        checkpoint,
                    },
                ) => {
                    let mut bad = Vec::new();
                    if field(&v, "id").as_deref() != Some(id.as_str()) {
                        bad.push(format!("id {id} != frozen {:?}", field(&v, "id")));
                    }
                    if let Some(t) = field(&v, "topology") {
                        if topology.as_deref() != Some(t.as_str()) {
                            bad.push(format!("topology {topology:?} != frozen {t}"));
                        }
                    }
                    if let Some(c) = field(&v, "checkpoint") {
                        if checkpoint.as_deref() != Some(c.as_str()) {
                            bad.push(format!("checkpoint {checkpoint:?} != frozen {c}"));
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
    let topof = |n: &str| -> String {
        match get(n) {
            Some(Outcome::Ok {
                topology: Some(t), ..
            }) => t.clone(),
            _ => String::new(),
        }
    };
    let cpof = |n: &str| -> String {
        match get(n) {
            Some(Outcome::Ok {
                checkpoint: Some(c),
                ..
            }) => c.clone(),
            _ => String::new(),
        }
    };
    let mut cross: Vec<(String, bool, String)> = Vec::new();
    cross.push((
        "checkpoint_id == snapshot_id (unconditional)".into(),
        cpof("manifest/checkpoint") == idof("manifest/checkpoint")
            && !cpof("manifest/checkpoint").is_empty(),
        cpof("manifest/checkpoint"),
    ));
    cross.push((
        "header ObjectId != checkpoint_id".into(),
        idof("header/plain-fp8") != cpof("manifest/checkpoint"),
        format!(
            "{} vs {}",
            &idof("header/plain-fp8")[..19],
            &cpof("manifest/checkpoint")[..19]
        ),
    ));
    cross.push((
        "a config change flips header id, not topology_digest".into(),
        idof("header/plain-fp8") != idof("header/plain-fp8-noconfig")
            && topof("header/plain-fp8") == topof("header/plain-fp8-noconfig"),
        topof("header/plain-fp8"),
    ));
    cross.push((
        "one more sibling file flips checkpoint_id only".into(),
        cpof("manifest/checkpoint") != cpof("manifest/checkpoint-extra-file"),
        cpof("manifest/checkpoint-extra-file"),
    ));
    let seeds_match = registry::seeds()
        .iter()
        .all(|s| closure.get(&s.spec.object_id()).is_some());
    cross.push((
        "every seed digest resolves in the corpus closure".into(),
        seeds_match,
        String::new(),
    ));

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
    if fail == 0 && closure_ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n])
    }
}

// ------------------------------------------------------------------ bench

fn rss_kb(field: &str) -> u64 {
    fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with(field))
        .and_then(|l| l.split_whitespace().nth(1).and_then(|n| n.parse().ok()))
        .unwrap_or(0)
}

fn cmd_bench(tensors: usize, reps: usize) -> ExitCode {
    let plain = seed_spec("plain/1", 0);
    let fp8 = seed_spec("fp8-rowwise/1", 0);
    let mut closure = Closure::default();
    closure.insert(plain.clone());
    closure.insert(fp8.clone());

    let build = Instant::now();
    let mut ts = Vec::with_capacity(tensors);
    for i in 0..tensors {
        let key = format!(
            "blocks.{}.attn.to_{}.weight",
            i / 4,
            "qkvo".as_bytes()[i % 4] as char
        );
        ts.push((
            key,
            Tensor {
                dtype: Dtype::Bf16,
                shape: vec![4096, 4096],
                encoding: fp8.object_id(),
                parts: vec![
                    (
                        "data".to_string(),
                        Part {
                            dtype: Dtype::F8E4M3FN,
                            shape: vec![4096, 4096],
                            body: Body::Segments(
                                (0..4)
                                    .map(|s| ObjectRef {
                                        sha256: tensorfs_core::sha256::hex_digest(
                                            format!("{i}:{s}").as_bytes(),
                                        ),
                                        length: 4 * 1024 * 1024,
                                    })
                                    .collect(),
                            ),
                        },
                    ),
                    ("scale".to_string(), part_of(Dtype::F32, vec![4096])),
                ],
            },
        ));
    }
    ts.sort_by(|a, b| a.0.cmp(&b.0));
    let h = Header {
        configs: vec![("transformer".to_string(), config_ref())],
        encodings: encodings_of(&[&fp8]),
        components: vec![("transformer".to_string(), ts)],
    };
    let build_s = build.elapsed().as_secs_f64();

    let t = Instant::now();
    let mut bytes = Vec::new();
    for _ in 0..reps {
        bytes = h.canonical_bytes();
    }
    let write_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let mut id = String::new();
    for _ in 0..reps {
        id = tensorfs_core::ids::object_id(&bytes);
    }
    let hash_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    for _ in 0..reps {
        Header::parse(&bytes).expect("reparse");
    }
    let parse_s = t.elapsed().as_secs_f64();

    let parsed = Header::parse(&bytes).unwrap();
    let t = Instant::now();
    for _ in 0..reps {
        parsed.validate(&closure).expect("validate");
    }
    let val_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let mut topo = String::new();
    for _ in 0..reps {
        topo = parsed.topology_digest();
    }
    let topo_s = t.elapsed().as_secs_f64();

    let mb = bytes.len() as f64 / (1024.0 * 1024.0);
    println!(
        "header: {tensors} tensors, {} B canonical ({mb:.2} MiB), built in {build_s:.3}s",
        bytes.len()
    );
    println!("id: {id}");
    println!("topology_digest: {topo}");
    println!("reps: {reps}");
    let row = |what: &str, s: f64| {
        println!(
            "  {what:<22} {:>8.1} ms/doc   {:>7.1} docs/s   {:>8.1} MiB/s",
            s / reps as f64 * 1000.0,
            reps as f64 / s,
            mb * reps as f64 / s
        )
    };
    row("canonical write", write_s);
    row("sha256 id", hash_s);
    row("strict parse", parse_s);
    row("validate", val_s);
    row("topology projection", topo_s);
    println!(
        "  peak RSS {:.1} MiB (current {:.1} MiB)",
        rss_kb("VmHWM") as f64 / 1024.0,
        rss_kb("VmRSS") as f64 / 1024.0
    );
    ExitCode::SUCCESS
}

// ------------------------------------------------------------------ flip matrix

fn cmd_flip() -> ExitCode {
    let (ha, _) = header_a(Some(config_ref()));
    let snap = manifest_checkpoint(&ha, false);
    let prof = profile_normal(
        vec![seed_spec("plain/1", 0).object_ref()],
        "safetensors.diffusers",
    );
    let sub0 = subject(&prof, &ha, "acme");
    let r0 = receipt(&sub0, &snap, &ha, "platform-key-1");

    let base = (
        ha.object_id(),
        ha.topology_digest(),
        snap.checkpoint_id().unwrap(),
        sub0.object_id(),
        r0.object_id(),
    );

    let mut rows: Vec<(&str, (String, String, String, String, String), [bool; 5])> = Vec::new();

    let mut h = ha.clone();
    if let Body::Segments(s) = &mut h.components[0].1[0].1.parts[0].1.body {
        s[0].sha256 = tensorfs_core::sha256::hex_digest(b"different weight bytes");
    }
    let s = manifest_checkpoint(&h, false);
    rows.push((
        "artifact bytes change",
        (
            h.object_id(),
            h.topology_digest(),
            s.checkpoint_id().unwrap(),
            sub0.object_id(),
            r0.object_id(),
        ),
        [true, false, true, false, false],
    ));

    let mut h = ha.clone();
    h.components[0].1[1].1.dtype = Dtype::F16;
    h.components[0].1[1].1.parts[0].1.dtype = Dtype::F16;
    let s = manifest_checkpoint(&h, false);
    rows.push((
        "logical dtype change",
        (
            h.object_id(),
            h.topology_digest(),
            s.checkpoint_id().unwrap(),
            sub0.object_id(),
            r0.object_id(),
        ),
        [true, true, true, false, false],
    ));

    let (h, _) = header_a(None);
    let s = manifest_checkpoint(&h, false);
    rows.push((
        "config change",
        (
            h.object_id(),
            h.topology_digest(),
            s.checkpoint_id().unwrap(),
            sub0.object_id(),
            r0.object_id(),
        ),
        [true, false, true, false, false],
    ));

    let s = manifest_checkpoint(&ha, true);
    rows.push((
        "snapshot gains a file",
        (
            ha.object_id(),
            ha.topology_digest(),
            s.checkpoint_id().unwrap(),
            sub0.object_id(),
            r0.object_id(),
        ),
        [false, false, true, false, false],
    ));

    let sub1 = subject(&prof, &ha, "other-tenant");
    rows.push((
        "tenant change",
        (
            ha.object_id(),
            ha.topology_digest(),
            snap.checkpoint_id().unwrap(),
            sub1.object_id(),
            r0.object_id(),
        ),
        [false, false, false, true, false],
    ));

    let r1 = receipt(&sub0, &snap, &ha, "platform-key-2-rotated");
    rows.push((
        "signer rotation",
        (
            ha.object_id(),
            ha.topology_digest(),
            snap.checkpoint_id().unwrap(),
            sub0.object_id(),
            r1.object_id(),
        ),
        [false, false, false, false, true],
    ));

    println!("identity-flip matrix (expected flips vs observed)");
    println!(
        "{:<24} {:>10} {:>10} {:>12} {:>10} {:>10}",
        "change", "header", "topology", "checkpoint", "subject", "receipt"
    );
    let mut ok = true;
    for (what, got, want) in &rows {
        let obs = [
            got.0 != base.0,
            got.1 != base.1,
            got.2 != base.2,
            got.3 != base.3,
            got.4 != base.4,
        ];
        let cell = |i: usize| {
            let mark = if obs[i] == want[i] { "" } else { " <-WRONG" };
            ok &= obs[i] == want[i];
            format!("{}{}", if obs[i] { "FLIP" } else { "same" }, mark)
        };
        let cells: Vec<String> = (0..5).map(cell).collect();
        println!(
            "{what:<24} {:>10} {:>10} {:>12} {:>10} {:>10}",
            cells[0], cells[1], cells[2], cells[3], cells[4]
        );
    }
    // alias identity: the alias lives outside the spec object, so renaming flips nothing.
    let plain = seed_spec("plain/1", 0);
    let mut renamed = plain.clone();
    renamed.doc = plain.doc.clone();
    let alias_stable = renamed.object_id() == plain.object_id();
    println!(
        "{:<24} {:>10}",
        "alias rename/promotion",
        if alias_stable { "same" } else { "FLIP <-WRONG" }
    );
    ok &= alias_stable;
    if ok {
        println!("\nall flips as designed");
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

// ------------------------------------------------------------------ id

fn cmd_id(p: &Path) -> ExitCode {
    let bytes = match fs::read(p) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{}: {e}", p.display());
            return ExitCode::FAILURE;
        }
    };
    println!("object_id: {}", tensorfs_core::ids::object_id(&bytes));
    println!("length:    {}", bytes.len());
    if let Ok(h) = Header::parse(&bytes) {
        println!("kind:      cozytensors/1 header");
        println!("topology:  {}", h.topology_digest());
        println!("tensors:   {}", h.tensors().count());
        println!("pretty:\n{}", canon::pretty(&h.to_value(), 0));
    } else if let Ok(m) = SnapshotManifest::parse(&bytes) {
        println!("kind:      snapshot manifest");
        match m.checkpoint_id() {
            Ok(c) => println!("checkpoint_id: {c}"),
            Err(e) => println!("checkpoint_id: n/a ({})", e.code.as_str()),
        }
    } else if let Ok(s) = EncodingSpec::parse(&bytes) {
        println!("kind:      encoding spec ({} roles)", s.roles.len());
    }
    ExitCode::SUCCESS
}

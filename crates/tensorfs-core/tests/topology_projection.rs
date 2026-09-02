//! tfs-038 — the shape-only topology projection, and the proof it cannot drift from the
//! exact one.
//!
//! `topology_digest` is an EXACT index: it finds the pairs that already agree. The half a
//! catalog search actually wants is the pairs that differ only by dtype or encoding, and
//! exact equality cannot see them. The shape-only projection buckets exactly those, so a
//! search is three cheap steps — an exact hit, a shape hit worth grading, and everything
//! else refused without grading at all.
//!
//! Both digests come out of ONE function body (`header::topology_projection`) with one
//! enum argument. `the_shape_projection_is_the_exact_one_with_dtype_elided` is that claim
//! as an executable derivation over every frozen document in the corpus: it takes the
//! EXACT projection's value, structurally removes the dtype leaf, restamps the domain, and
//! requires the canonical bytes to equal the shape projection's exactly. Any edit that
//! moves one projection and not the other fails here.

use std::path::PathBuf;

use tensorfs_core::canon::{self, Value};
use tensorfs_core::dtype::Dtype;
use tensorfs_core::fit::{
    self as compat, Accepts, CodeTopology, CompatibilityReport, Custody, Depth, Requirement,
};
use tensorfs_core::header::{
    self, Closure, Header, Projection, TOPOLOGY_IDENTITY_DOMAIN, TOPOLOGY_SHAPE_DOMAIN,
};
use tensorfs_core::ids::Plain;
use tensorfs_core::spec::EncodingSpec;

fn vectors() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vectors")
}

fn corpus_closure() -> Closure {
    let mut closure = Closure::default();
    for e in std::fs::read_dir(vectors().join("closure")).expect("closure dir") {
        let bytes = std::fs::read(e.unwrap().path()).unwrap();
        closure.insert(EncodingSpec::decode_nested(&bytes).expect("a corpus closure spec"));
    }
    closure
}

/// The EXACT projection's value, rewritten into what a shape-only projection must be:
/// every leaf's `dtype` removed, the domain separator swapped, nothing else touched.
/// Deliberately independent of `topology_projection` — this is the check, not the code
/// under test — and deliberately total: an unexpected key panics rather than being ignored,
/// because a projection that grew a field the shape side does not know about is exactly the
/// drift this test exists to catch.
fn elide_dtype(v: &Value) -> Value {
    let Value::Obj(fields) = v else {
        panic!("a topology projection is an object")
    };
    let mut out: Vec<(String, Value)> = Vec::new();
    for (name, field) in fields {
        match name.as_str() {
            "projection" => {
                assert_eq!(field, &Value::Str(TOPOLOGY_IDENTITY_DOMAIN.to_string()));
                out.push((name.clone(), Value::Str(TOPOLOGY_SHAPE_DOMAIN.to_string())));
            }
            "components" => {
                let Value::Obj(components) = field else {
                    panic!("components is a map")
                };
                let mut cs = Vec::new();
                for (component, keys) in components {
                    let Value::Obj(keys) = keys else {
                        panic!("a component is a map of keys")
                    };
                    let mut ks = Vec::new();
                    for (key, leaf) in keys {
                        let Value::Obj(leaf) = leaf else {
                            panic!("a leaf is an object")
                        };
                        assert!(
                            leaf.iter().any(|(n, _)| n == "dtype"),
                            "the exact projection's leaf carries a dtype"
                        );
                        let stripped: Vec<(String, Value)> =
                            leaf.iter().filter(|(n, _)| n != "dtype").cloned().collect();
                        assert_eq!(
                            stripped.len(),
                            leaf.len() - 1,
                            "exactly the dtype comes off"
                        );
                        ks.push((key.clone(), Value::Obj(stripped)));
                    }
                    cs.push((component.clone(), Value::Obj(ks)));
                }
                out.push((name.clone(), Value::Obj(cs)));
            }
            other => panic!("a topology projection has no {other:?} field"),
        }
    }
    Value::Obj(out)
}

fn tensors(spec: &[(&str, &str, Dtype, Vec<u64>)]) -> Vec<(String, String, Dtype, Vec<u64>)> {
    spec.iter()
        .map(|(c, k, d, s)| (c.to_string(), k.to_string(), *d, s.clone()))
        .collect()
}

fn project(rows: &[(String, String, Dtype, Vec<u64>)], p: Projection) -> Value {
    header::topology_projection(
        p,
        rows.iter()
            .map(|(c, k, d, s)| (c.as_str(), k.as_str(), *d, s.as_slice())),
    )
}

fn digest(rows: &[(String, String, Dtype, Vec<u64>)], p: Projection) -> String {
    header::topology_digest_of(&project(rows, p))
}

// ------------------------------------------------------------------ the no-drift proof

#[test]
fn the_shape_projection_is_the_exact_one_with_dtype_elided() {
    let mut checked = 0usize;

    fn prove(exact: &Value, shape: &Value, what: &str, checked: &mut usize) {
        assert_eq!(
            canon::write(&elide_dtype(exact)),
            canon::write(shape),
            "{what}: the shape projection is not the exact one with its dtype elided"
        );
        *checked += 1;
    }

    // Every frozen header and code topology in the corpus, through the shipped methods.
    let closure = corpus_closure();
    for group in ["header", "onebyte", "layout", "structure"] {
        let dir = vectors().join("cases").join(group);
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut paths: Vec<PathBuf> = entries.map(|e| e.unwrap().path()).collect();
        paths.sort();
        for path in paths {
            let bytes = std::fs::read(&path).unwrap();
            let name = path.display().to_string();
            if let Ok(h) = Header::parse(&bytes) {
                if h.validate(&closure).is_ok() {
                    prove(
                        &h.projected(Projection::Identity),
                        &h.projected(Projection::ShapeOnly),
                        &name,
                        &mut checked,
                    );
                }
            }
            if let Ok(code) = CodeTopology::parse(&bytes) {
                prove(
                    &code.projected(Projection::Identity),
                    &code.projected(Projection::ShapeOnly),
                    &name,
                    &mut checked,
                );
            }
        }
    }
    assert!(checked >= 4, "the corpus supplied {checked} topologies");

    // Plus synthetic shapes the corpus does not carry: every dtype, several ranks, more
    // than one component, and keys whose canonical order is not their insertion order.
    for dtype in [
        Dtype::Bf16,
        Dtype::F16,
        Dtype::F32,
        Dtype::F64,
        Dtype::F8E4M3FN,
        Dtype::F8E5M2,
        Dtype::I8,
        Dtype::U8,
        Dtype::Bool,
    ] {
        let rows = tensors(&[
            ("vae", "zz.last", dtype, vec![3]),
            ("transformer", "b.1.w", dtype, vec![64, 128]),
            ("transformer", "a.0.w", dtype, vec![]),
            ("vae", "aa.first", dtype, vec![1, 2, 3, 4]),
        ]);
        prove(
            &project(&rows, Projection::Identity),
            &project(&rows, Projection::ShapeOnly),
            "synthetic",
            &mut checked,
        );
    }
}

// ------------------------------------------------------------------ the red arm

#[test]
fn two_topologies_differing_only_in_dtype_share_a_shape_digest_and_differ_in_the_exact_one() {
    let shape = vec![64u64, 128];
    let bf16 = tensors(&[(
        "transformer",
        "blocks.0.attn.to_q.weight",
        Dtype::Bf16,
        shape.clone(),
    )]);
    let fp8 = tensors(&[(
        "transformer",
        "blocks.0.attn.to_q.weight",
        Dtype::F8E4M3FN,
        shape.clone(),
    )]);

    assert_ne!(
        digest(&bf16, Projection::Identity),
        digest(&fp8, Projection::Identity),
        "the exact index must SEE a dtype difference — it is the whole reason it is exact"
    );
    assert_eq!(
        digest(&bf16, Projection::ShapeOnly),
        digest(&fp8, Projection::ShapeOnly),
        "a dtype difference is exactly what the shape bucket exists to look past"
    );

    // A SHAPE difference moves both: the shape projection is not a wildcard, it elides one
    // named field. An INCOMPATIBLE pair must not land in the candidate bucket.
    let narrow = tensors(&[(
        "transformer",
        "blocks.0.attn.to_q.weight",
        Dtype::Bf16,
        vec![7],
    )]);
    assert_ne!(
        digest(&bf16, Projection::ShapeOnly),
        digest(&narrow, Projection::ShapeOnly)
    );
    assert_ne!(
        digest(&bf16, Projection::Identity),
        digest(&narrow, Projection::Identity)
    );

    // A KEY difference moves both too.
    let renamed = tensors(&[(
        "transformer",
        "blocks.0.attn.to_k.weight",
        Dtype::Bf16,
        shape,
    )]);
    assert_ne!(
        digest(&bf16, Projection::ShapeOnly),
        digest(&renamed, Projection::ShapeOnly)
    );

    // The two projections are domain-separated, so a shape digest can never be mistaken for
    // an exact one in a bucket key.
    assert_ne!(
        digest(&bf16, Projection::Identity),
        digest(&bf16, Projection::ShapeOnly)
    );
    assert_ne!(TOPOLOGY_IDENTITY_DOMAIN, TOPOLOGY_SHAPE_DOMAIN);
}

// ------------------------------------------------------------------ on the report

/// The report carries all four digests, so ONE `compat check` answers the bucketing
/// question for both sides — an exact miss that is a shape hit is the candidate set.
#[test]
fn the_report_carries_both_projections_for_both_sides() {
    let closure = corpus_closure();
    let header = Header::parse(
        &std::fs::read(vectors().join("cases/header/plain-fp8.cbor")).expect("the corpus header"),
    )
    .expect("it parses");
    header.validate(&closure).expect("it validates");

    let identity = CodeTopology {
        contract: tensorfs_core::ids::object_id(b"contract"),
        components: header
            .components
            .iter()
            .map(|(c, ts)| {
                (
                    c.clone(),
                    ts.iter()
                        .map(|(k, t)| {
                            (
                                k.clone(),
                                Requirement {
                                    dtype: t.dtype,
                                    shape: t.shape.clone(),
                                    accepts: Accepts::Any,
                                    pinned: false,
                                },
                            )
                        })
                        .collect(),
                )
            })
            .collect(),
        constructed_components: header.components.iter().map(|(c, _)| c.clone()).collect(),
    };
    let mut recast = identity.clone();
    for (_, keys) in recast.components.iter_mut() {
        for (_, r) in keys.iter_mut() {
            r.dtype = Dtype::F8E4M3FN;
        }
    }

    let pin: Vec<String> = header
        .encodings
        .iter()
        .map(EncodingSpec::object_id)
        .collect();
    let report = |code: &CodeTopology| -> CompatibilityReport {
        let out = compat::check_compatibility(
            code,
            &header,
            &closure,
            &pin,
            Custody::Canonical,
            Depth::Full,
        )
        .expect("the check runs");
        CompatibilityReport::of(code, &header, Custody::Canonical, &out).expect("the report builds")
    };

    // The exact-index HIT: both projections agree on both sides.
    let hit = report(&identity);
    assert!(hit.is_satisfied());
    assert_eq!(hit.topology_required, hit.topology_present);
    assert_eq!(hit.shape_required, hit.shape_present);

    // The interesting half: an exact MISS that is a shape HIT.
    let candidate = report(&recast);
    assert!(!candidate.is_satisfied());
    assert_ne!(candidate.topology_required, candidate.topology_present);
    assert_eq!(
        candidate.shape_required, candidate.shape_present,
        "a dtype-only recast is the same shape bucket"
    );

    // And they survive the round trip through the canonical document, which is what
    // actually crosses the boundary.
    let bytes = candidate.canonical_bytes();
    let back = CompatibilityReport::parse(&bytes).expect("it re-parses");
    assert_eq!(back, candidate);
    let text = String::from_utf8(bytes).expect("canonical bytes are ASCII");
    assert!(text.contains("\"shape_present\":"));
    assert!(text.contains("\"shape_required\":"));

    // The shape digest is the header's own, by the same code path.
    assert_eq!(candidate.shape_present, header.shape_digest());
    assert_eq!(candidate.shape_required, recast.shape_digest());
}

/// A report missing either shape digest is refused, not defaulted. An absent field that
/// quietly became an empty string is how a bucket key silently starts matching everything.
#[test]
fn a_report_without_its_shape_digests_is_refused() {
    let bytes = std::fs::read(vectors().join("cases/layout/report-recast.json"))
        .expect("the frozen recast report");
    let text = String::from_utf8(bytes).unwrap();
    // The plant renames the field's LAST character, so the key set stays sorted and the
    // refusal is about the missing field rather than about canonical ordering.
    for (field, planted) in [
        ("shape_present", "shape_presenz"),
        ("shape_required", "shape_requirez"),
    ] {
        let cut = text.replace(&format!("\"{field}\":"), &format!("\"{planted}\":"));
        assert_ne!(cut, text, "the plant applied");
        let err = CompatibilityReport::parse(cut.as_bytes())
            .expect_err("a report without its shape digest is not a report");
        assert!(
            matches!(
                err.code,
                tensorfs_core::err::Code::MISSING_FIELD | tensorfs_core::err::Code::UNKNOWN_FIELD
            ),
            "{field}: refused {} instead",
            err.code.as_str()
        );
    }
}

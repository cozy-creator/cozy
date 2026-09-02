//! The MACHINE SURFACE: the `--json` documents `tfs` prints for a program to read
//! (proto-020).
//!
//! These are a producer/consumer contract across two repositories and two languages —
//! tensorfs emits them, tensorhub's `internal/compatibilityproof` decodes them — and for a
//! long time nothing held it. tensorfs added `depth` to `compat check --json` and
//! `depth`+`mismatches` to `grade --json`; tensorhub could then not decode a single real
//! answer, and both repositories' CI stayed green throughout. It happened a second time
//! with the four `topology_*` keys. A recorded sample on the consumer side is a snapshot,
//! not a contract: it goes stale the moment the producer grows a key, and says nothing.
//!
//! So the emitters live HERE, in the library, next to every other document this workspace
//! freezes — not in the `tfs` binary where the corpus generator could not reach them. Each
//! surface is a pure function of library values, its key list is compiled in, and one
//! representative document per verdict is frozen in `vectors/cases/machine/` and re-emitted
//! by `tfs vectors verify`. Adding a key to any of them fails HERE, on the producer side,
//! naming the consumer — instead of silently breaking it at runtime.
//!
//! Every surface is TOTAL: every key is present on every verdict, empty where that verdict
//! does not carry it. A consumer decodes ONE shape and reads the answer field; it never has
//! to discover which keys this particular verdict happened to emit.

use crate::canon::{write, Value};
use crate::capability::CapabilityRecord;
use crate::fit::{CompatibilityReport, Fit, Side};
use crate::ids::Plain;

#[cfg(feature = "grade")]
use crate::grade::GradeDecision;

/// The remedy line every diagnostic in this module ends with. It names the consumer,
/// because the whole failure mode is a producer change that looks local and is not.
pub const CONSUMER_REMEDY: &str =
    "this document is decoded by tensorhub (internal/compatibilityproof). Regenerate the \
     frozen vectors with `tfs vectors gen`, then re-vendor them into tensorhub \
     (internal/compatibilityproof/testdata/tfs-machine/) and teach its decoder the change \
     IN THE SAME LANDING ORDER: tensorfs first, tensorhub second.";

/// `tfs compat check --json`.
pub const CHECK_KEYS: &[&str] = &[
    "depth",
    "details",
    "encodings",
    "report",
    "satisfied",
    "topology_present",
    "topology_required",
    "topology_shape_present",
    "topology_shape_required",
];

/// One entry of `check --json`'s `details` array.
pub const CHECK_DETAIL_KEYS: &[&str] = &["code", "component", "key", "present", "required"];

/// `tfs grade --json`, and every line of `tfs grade batch`.
pub const GRADE_KEYS: &[&str] = &[
    "decision",
    "depth",
    "grade",
    "mismatches",
    "reason",
    "remedy_mapping",
    "remedy_registration",
    "remedy_source",
    "remedy_target",
    "report",
];

/// `tfs capability <encoding> <device> --json`.
pub const CAPABILITY_KEYS: &[&str] = &[
    "admitted",
    "code",
    "device",
    "encoding",
    "implementation",
    "note",
    "qualified_devices",
    "reason",
    "vectors",
];

/// `tfs fit ... --json` (tfs-042). TOTAL like the rest: `component`, `key`, `encoding` and
/// `device` are empty where the code does not name one, and `routes` and `ignored` carry
/// zeros and an empty list off every refusal.
pub const FIT_KEYS: &[&str] = &[
    "code",
    "component",
    "custody",
    "detail",
    "device",
    "encoding",
    "ignored",
    "key",
    "routes",
];

/// The surfaces, by the corpus doc kind that verifies them. `corpus::run_doc` reads this,
/// so a new surface is one row here and one frozen vector — never a second dispatch.
pub const SURFACES: &[(&str, &[&str])] = &[
    ("machine_check", CHECK_KEYS),
    ("machine_grade", GRADE_KEYS),
    ("machine_capability", CAPABILITY_KEYS),
    ("machine_fit", FIT_KEYS),
];

/// `tfs fit --json`: the one Fit code, its §3 text, and the fields the text was built
/// from, so a caller branches on `code` and prints `detail` without re-deriving either.
pub fn fit_json(fit: &Fit, custody: crate::fit::Custody) -> Vec<u8> {
    let (component, key, encoding, device) = match fit {
        Fit::Ok { device, .. } => ("", "", "", device.as_deref().unwrap_or("")),
        Fit::ComponentMissing { component, .. } => (component.as_str(), "", "", ""),
        Fit::TensorMissing { component, key }
        | Fit::ShapeMismatch { component, key, .. }
        | Fit::DtypePinnedMismatch { component, key, .. }
        | Fit::ExtraTensor { component, key } => (component.as_str(), key.as_str(), "", ""),
        Fit::EncodingUnsupported {
            component,
            key,
            encoding,
            ..
        } => (component.as_str(), key.as_str(), encoding.as_str(), ""),
        Fit::EncodingUnqualified {
            encoding, device, ..
        } => ("", "", encoding.as_str(), device.as_str()),
    };
    let (routes, ignored) = match fit {
        Fit::Ok {
            routes, ignored, ..
        } => (routes.clone(), ignored.clone()),
        _ => (crate::fit::Routes::default(), Vec::new()),
    };
    write(&Value::obj(vec![
        ("code", Value::str(fit.code())),
        ("component", Value::str(component)),
        ("custody", Value::str(custody.name())),
        ("detail", Value::str(fit.text())),
        ("device", Value::str(device)),
        ("encoding", Value::str(encoding)),
        (
            "ignored",
            Value::arr(ignored.into_iter().map(Value::str).collect()),
        ),
        ("key", Value::str(key)),
        (
            "routes",
            Value::obj(vec![
                ("decoded_float", Value::uint(routes.decoded_float)),
                ("encoded_gemm", Value::uint(routes.encoded_gemm)),
                ("verbatim", Value::uint(routes.verbatim)),
            ]),
        ),
    ]))
}

/// One side of a disagreement, rendered for a person reading the machine document. The
/// consumer treats it as opaque text; it is here so the two callers cannot render it two
/// ways.
fn side(s: &Side) -> String {
    let mut v = Vec::new();
    if let Some(d) = s.dtype {
        v.push(d.name().to_string());
    }
    if let Some(sh) = &s.shape {
        v.push(format!("{sh:?}"));
    }
    if let Some(e) = &s.encoding {
        v.push(e[7..19].to_string());
    }
    if let Some(a) = &s.accepts {
        v.push(format!(
            "accepts {}",
            if a.len() > 20 { &a[..20] } else { a }
        ));
    }
    if v.is_empty() {
        "ABSENT".to_string()
    } else {
        v.join(" ")
    }
}

/// `tfs compat check --json`, as a pure function of the report the check produced.
///
/// `details` is the WHOLE mismatch set at the depth the walk was taken; `depth` says which,
/// so a consumer can tell a set of one from a walk that stopped after one. The four
/// `topology_*` keys are a search INDEX, never the decision: exact digests equal is an
/// exact-index hit, shape digests equal alone means the pair differs by dtype or encoding
/// and is worth grading. `encodings` is what the CHECKPOINT carries — the input to the
/// independent DEVICE question (`tfs capability`), which the consumer cannot ask without
/// it, and must not parse a header to learn.
pub fn check_json(report: &CompatibilityReport) -> Vec<u8> {
    let details: Vec<Value> = report
        .mismatches
        .iter()
        .map(|d| {
            Value::obj(vec![
                ("code", Value::str(d.code.as_str())),
                ("component", Value::str(d.component.clone())),
                ("key", Value::str(d.key.clone())),
                ("required", Value::str(side(&d.required))),
                ("present", Value::str(side(&d.present))),
            ])
        })
        .collect();
    let mut encodings: Vec<String> = Vec::new();
    for f in &report.facts {
        for e in &f.present_encodings {
            if !encodings.contains(&e.encoding) {
                encodings.push(e.encoding.clone());
            }
        }
    }
    encodings.sort();
    write(&Value::obj(vec![
        ("satisfied", Value::Bool(report.mismatches.is_empty())),
        (
            "encodings",
            Value::arr(encodings.into_iter().map(Value::str).collect()),
        ),
        ("depth", Value::str(report.depth.name())),
        ("details", Value::arr(details)),
        ("report", Value::str(report.object_id())),
        (
            "topology_required",
            Value::str(report.topology_required.clone()),
        ),
        (
            "topology_present",
            Value::str(report.topology_present.clone()),
        ),
        (
            "topology_shape_required",
            Value::str(report.shape_required.clone()),
        ),
        (
            "topology_shape_present",
            Value::str(report.shape_present.clone()),
        ),
    ]))
}

/// `tfs grade --json`, and every line of `tfs grade batch` — literally this function, so
/// the batch is not a second surface that happens to agree today.
///
/// `remedy_*` is the whole point of the four levels: PRODUCIBLE names the encoding the
/// CONTRACT requires, so "we could run this if you convert it" is distinguishable from
/// "no". `depth` and `mismatches` say WHICH question the grade answers — a `full` grade is
/// the worst fact of the whole set and the only one a catalog may rank on; a
/// `first-mismatch` grade is the preflight answer off one fact.
#[cfg(feature = "grade")]
pub fn grade_json(decision: &GradeDecision, report: &CompatibilityReport) -> Vec<u8> {
    use crate::grade::Grade;
    let (registration, mapping, target, source, reason) = match &decision.grade {
        Grade::Compatible => ("", "", "", "", ""),
        Grade::Convertible {
            registration,
            mapping,
        } => (registration.as_str(), mapping.as_str(), "", "", ""),
        Grade::Producible { target, source } => ("", "", target.as_str(), source.as_str(), ""),
        Grade::Incompatible { reason } => ("", "", "", "", reason.as_str()),
    };
    write(&Value::obj(vec![
        (
            "decision",
            Value::str(crate::ids::object_id(&decision.canonical_bytes())),
        ),
        ("grade", Value::str(decision.grade.name())),
        ("report", Value::str(decision.report.clone())),
        ("remedy_registration", Value::str(registration)),
        ("remedy_mapping", Value::str(mapping)),
        ("remedy_target", Value::str(target)),
        ("remedy_source", Value::str(source)),
        ("reason", Value::str(reason)),
        ("depth", Value::str(report.depth.name())),
        ("mismatches", Value::uint(report.mismatches.len() as u64)),
    ]))
}

/// `tfs capability <encoding> <device> --json`.
///
/// `qualified_devices` is the remedy half, and it is why this is not a bare boolean: "this
/// encoding is qualified, but somewhere else" is a routable answer and "nothing anywhere
/// runs this" is not, and a caller that cannot tell them apart can act on neither.
pub fn capability_json(
    admitted: Option<&CapabilityRecord>,
    encoding: &str,
    device: &str,
    refusal: &str,
    qualified: &[&str],
) -> Vec<u8> {
    let (implementation, vectors, note) = match admitted {
        Some(r) => (r.implementation.as_str(), r.vectors.id(), r.note.as_str()),
        None => ("", String::new(), ""),
    };
    write(&Value::obj(vec![
        ("admitted", Value::Bool(admitted.is_some())),
        (
            "code",
            Value::str(if admitted.is_some() {
                ""
            } else {
                crate::err::Code::CAPABILITY_UNQUALIFIED.as_str()
            }),
        ),
        ("device", Value::str(device)),
        ("encoding", Value::str(encoding)),
        ("implementation", Value::str(implementation)),
        ("note", Value::str(note)),
        (
            "qualified_devices",
            Value::arr(qualified.iter().map(|d| Value::str(*d)).collect()),
        ),
        ("reason", Value::str(refusal)),
        ("vectors", Value::str(vectors)),
    ]))
}

/// The keys of one machine document, read back through the strict canonical reader — the
/// same reader the consumer's strictness used to stand in for.
pub fn document_keys(bytes: &[u8]) -> crate::err::Result<Vec<String>> {
    match crate::canon::parse_canonical(bytes, crate::limits::DOC_MAX_BYTES)? {
        Value::Obj(pairs) => Ok(pairs.into_iter().map(|(k, _)| k).collect()),
        other => crate::err::refuse(
            crate::err::Code::UNKNOWN_FORMAT,
            format!("a machine document is a JSON object, got {}", other.kind()),
        ),
    }
}

/// The contract assertion over ONE frozen machine document: its key set is exactly the key
/// set this build emits, sorted (canonical JSON sorts its keys, so the frozen order IS the
/// sorted order).
///
/// This is the arm that fires on an ADDITIVE producer change. The consumer is named in the
/// message because a key added here is not a local edit: it is a cross-repository landing.
pub fn check_surface(doc: &str, bytes: &[u8]) -> crate::err::Result<()> {
    let Some((_, expected)) = SURFACES.iter().find(|(name, _)| *name == doc) else {
        return crate::err::refuse(
            crate::err::Code::UNKNOWN_FORMAT,
            format!("{doc:?} is not a machine surface: {SURFACES:?}"),
        );
    };
    let found = document_keys(bytes)?;
    if found
        .iter()
        .map(String::as_str)
        .eq(expected.iter().copied())
    {
        return Ok(());
    }
    let missing: Vec<&str> = expected
        .iter()
        .copied()
        .filter(|k| !found.iter().any(|f| f == k))
        .collect();
    let added: Vec<&str> = found
        .iter()
        .map(String::as_str)
        .filter(|k| !expected.contains(k))
        .collect();
    crate::err::refuse(
        crate::err::Code::UNKNOWN_FIELD,
        format!(
            "the frozen {doc} document and this build's {doc} surface disagree: \
             {added:?} added, {missing:?} missing. {CONSUMER_REMEDY}"
        ),
    )
}

#[cfg(test)]
mod surface {
    use super::*;
    use crate::registry;

    fn keys(bytes: &[u8]) -> Vec<String> {
        document_keys(bytes).expect("a machine document is canonical JSON")
    }

    fn plain() -> String {
        registry::seeds()
            .iter()
            .find(|s| s.alias == "plain/1")
            .expect("plain/1 is entry zero of the registry")
            .spec
            .object_id()
    }

    /// A report with one mismatch, built by hand: enough to make every emitter produce its
    /// whole key set, and nothing that has to stay in step with a fixture.
    fn report() -> CompatibilityReport {
        CompatibilityReport {
            contract: "example.Contract".to_string(),
            header: crate::ids::object_id(b"header"),
            topology_required: crate::ids::object_id(b"required"),
            topology_present: crate::ids::object_id(b"present"),
            shape_required: crate::ids::object_id(b"shape-required"),
            shape_present: crate::ids::object_id(b"shape-present"),
            custody: crate::fit::Custody::Canonical,
            depth: crate::fit::Depth::Full,
            mismatches: vec![crate::fit::Details {
                code: crate::err::Code::DTYPE_MISMATCH,
                component: "transformer".to_string(),
                key: "blocks.0.attn.to_q.weight".to_string(),
                required: Side {
                    dtype: Some(crate::dtype::Dtype::F32),
                    ..Side::default()
                },
                present: Side {
                    dtype: Some(crate::dtype::Dtype::Bf16),
                    ..Side::default()
                },
            }],
            facts: Vec::new(),
            unconsumed: Vec::new(),
        }
    }

    /// The compiled-in key lists ARE what the emitters emit. Without this, `CHECK_KEYS` and
    /// friends are a comment: an added key would change the document and leave the list
    /// describing the old one, and the frozen-vector arm would then be checking the wrong
    /// contract.
    #[test]
    fn every_key_list_matches_its_emitter() {
        let r = report();
        assert_eq!(keys(&check_json(&r)), CHECK_KEYS);
        let details = match crate::canon::parse_canonical(&check_json(&r), 1 << 20)
            .expect("check --json is canonical")
        {
            Value::Obj(pairs) => {
                pairs
                    .into_iter()
                    .find(|(k, _)| k == "details")
                    .expect("details")
                    .1
            }
            other => panic!("check --json is an object, got {}", other.kind()),
        };
        match details {
            Value::Arr(items) => match &items[0] {
                Value::Obj(pairs) => assert_eq!(
                    pairs.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
                    CHECK_DETAIL_KEYS
                ),
                other => panic!("a detail is an object, got {}", other.kind()),
            },
            other => panic!("details is an array, got {}", other.kind()),
        }
        let recs = registry::capability_records();
        let enc = plain();
        let admitted = recs
            .admit(&enc, "cpu")
            .expect("plain/1 is qualified on cpu");
        let yes = capability_json(Some(admitted), &enc, "cpu", "", &["cpu"]);
        let refusal = recs
            .admit(&enc, "cuda.sm89")
            .expect_err("no accelerator class is qualified yet");
        let no = capability_json(None, &enc, "cuda.sm89", &refusal.detail, &["cpu"]);
        assert_eq!(keys(&yes), CAPABILITY_KEYS);
        assert_eq!(keys(&no), CAPABILITY_KEYS);
        let ok = Fit::Ok {
            routes: crate::fit::Routes::default(),
            ignored: vec!["unet/extra".to_string()],
            device: Some("sm89".to_string()),
        };
        let no = Fit::ShapeMismatch {
            component: "unet".to_string(),
            key: "conv_in.weight".to_string(),
            code: vec![320, 4, 3, 3],
            checkpoint: vec![320, 9, 3, 3],
        };
        assert_eq!(keys(&fit_json(&ok, crate::fit::Custody::Local)), FIT_KEYS);
        assert_eq!(
            keys(&fit_json(&no, crate::fit::Custody::Canonical)),
            FIT_KEYS
        );
    }

    /// The same assertion for the grading surface, on every one of the four grades — the
    /// point of a TOTAL document is that they all carry the same keys. Feature-gated
    /// because the worker wheel builds without `grade`, and the facade still verifies the
    /// frozen grade documents there: `check_surface` reads the key LIST, not the emitter.
    #[cfg(feature = "grade")]
    #[test]
    fn the_grade_key_list_matches_its_emitter_on_every_grade() {
        let r = report();
        for grade in [
            crate::grade::Grade::Compatible,
            crate::grade::Grade::Convertible {
                registration: "bf16-to-f32".to_string(),
                mapping: "widen bf16 -> f32".to_string(),
            },
            crate::grade::Grade::Producible {
                target: "f32".to_string(),
                source: "bf16".to_string(),
            },
            crate::grade::Grade::Incompatible {
                reason: "shapes disagree".to_string(),
            },
        ] {
            let decision = GradeDecision {
                report: r.object_id(),
                grade,
            };
            assert_eq!(keys(&grade_json(&decision, &r)), GRADE_KEYS);
        }
    }

    /// The device axis answering NO is a NAMED refusal with its own code, never a bare
    /// false: a caller that cannot read `CAPABILITY_UNQUALIFIED` off the answer cannot tell
    /// "this card cannot run it" from "the weights do not fit the code".
    #[test]
    fn a_refusal_names_its_code_and_the_devices_that_would_work() {
        let recs = registry::capability_records();
        let enc = plain();
        let refusal = recs.admit(&enc, "cuda.sm89").expect_err("unqualified");
        let no = capability_json(None, &enc, "cuda.sm89", &refusal.detail, &["cpu"]);
        let text = String::from_utf8(no).expect("canonical JSON is UTF-8");
        assert!(text.contains(r#""admitted":false"#), "{text}");
        assert!(
            text.contains(r#""code":"CAPABILITY_UNQUALIFIED""#),
            "{text}"
        );
        assert!(text.contains(r#""qualified_devices":["cpu"]"#), "{text}");
        assert!(text.contains(r#""implementation":"""#), "{text}");
    }

    /// A vectorless spec is structurally unqualifiable, so it refuses on the very device the
    /// reference decoders run on, and says so with an EMPTY qualified set.
    #[test]
    fn a_vectorless_spec_is_qualified_nowhere() {
        let recs = registry::capability_records();
        let enc = registry::seeds()
            .iter()
            .find(|s| s.spec.vectors.is_none())
            .expect("the registry seeds at least one vectorless spec")
            .spec
            .object_id();
        let refusal = recs
            .admit(&enc, "cpu")
            .expect_err("vectorless never qualifies");
        let no = capability_json(None, &enc, "cpu", &refusal.detail, &[]);
        let text = String::from_utf8(no).expect("canonical JSON is UTF-8");
        assert!(text.contains(r#""qualified_devices":[]"#), "{text}");
    }

    /// The refusal detail rides into a canonical document, and this layer is
    /// printable-ASCII only. A stray em dash would make the answer unreadable by the strict
    /// reader on the other side.
    #[test]
    fn the_refusal_detail_stays_printable_ascii() {
        let recs = registry::capability_records();
        let enc = plain();
        let refusal = recs.admit(&enc, "cuda.sm89").expect_err("unqualified");
        let no = capability_json(None, &enc, "cuda.sm89", &refusal.detail, &["cpu"]);
        crate::canon::parse_canonical(&no, 1 << 20)
            .expect("the strict reader accepts its own output");
    }

    /// The red arm of `check_surface` itself: a document that grew a key is refused, and the
    /// refusal names both the key and the consumer.
    #[test]
    fn an_additive_key_is_refused_and_names_the_consumer() {
        let recs = registry::capability_records();
        let enc = plain();
        let admitted = recs.admit(&enc, "cpu").expect("qualified");
        let good = capability_json(Some(admitted), &enc, "cpu", "", &["cpu"]);
        check_surface("machine_capability", &good).expect("its own output is its own contract");
        let grown = String::from_utf8(good)
            .expect("UTF-8")
            .replace(r#""code":"#, r#""admitted_at":"","code":"#);
        let refusal =
            check_surface("machine_capability", grown.as_bytes()).expect_err("a grown key");
        assert_eq!(refusal.code, crate::err::Code::UNKNOWN_FIELD);
        assert!(refusal.detail.contains("admitted_at"), "{}", refusal.detail);
        assert!(refusal.detail.contains("tensorhub"), "{}", refusal.detail);
    }

    /// The DANGEROUS direction, which the consumer's `DisallowUnknownFields` never covered:
    /// Go zero-values a key that stopped being emitted, so a removed or renamed key silently
    /// turns a verdict into `false`/`""` rather than failing. Here it refuses.
    #[test]
    fn a_removed_key_is_refused_too() {
        let good = check_json(&report());
        let shrunk = String::from_utf8(good)
            .expect("UTF-8")
            .replace(r#""depth":"full","#, "");
        let refusal = check_surface("machine_check", shrunk.as_bytes()).expect_err("a lost key");
        assert_eq!(refusal.code, crate::err::Code::UNKNOWN_FIELD);
        assert!(refusal.detail.contains("depth"), "{}", refusal.detail);
        assert!(refusal.detail.contains("tensorhub"), "{}", refusal.detail);
    }
}

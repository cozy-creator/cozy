//! The grading plane, end to end, over the real code paths.
//!
//! This suite exists because the plane had ZERO automated coverage — which is how an
//! audit came to read it as dead and delete it. Every test here drives the shipped
//! function or the shipped binary; nothing is mocked, and the three laws grade.rs was
//! written to hold each have an arm that fails if the law is weakened:
//!
//!   1. a registration RUNS its round trip (and a re-quantization cannot register),
//!   2. the remedy derives from the TARGET, never the observation,
//!   3. a cast-only mismatch a registration covers can never grade INCOMPATIBLE.
//!
//! The CLI arms drive `tfs` itself over the frozen corpus, so the `--json` surface
//! tensorhub decodes is a tested contract rather than a print that happens to parse.

use std::path::{Path, PathBuf};
use std::process::Command;

use tensorfs_core::compat::{self, CodeTopology, CompatibilityReport, Lane};
use tensorfs_core::dtype::Dtype;
use tensorfs_core::err::Code;
use tensorfs_core::grade::{
    self, complete_domain_16, ConversionRegistration, Grade, GradeDecision, Mapping, Registry,
};
use tensorfs_core::header::{Closure, Header};
use tensorfs_core::ids::Plain;
use tensorfs_core::spec::EncodingSpec;

fn vectors() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vectors")
}

fn refusal_code(e: tensorfs_core::err::Refusal) -> &'static str {
    e.code.as_str()
}

// ------------------------------------------------------------------ law 1: the round trip

/// The registration ACT over the COMPLETE 16-bit domain. Not a sample: all 65,536
/// representable bf16 patterns are widened to f32 and narrowed back, and the registration
/// only exists because that returned byte-identical.
#[test]
fn a_widening_registration_proves_itself_over_every_representable_input() {
    let corpus = complete_domain_16();
    assert_eq!(corpus.len(), 1);
    assert_eq!(corpus[0].len(), 1 << 17, "65,536 patterns at 2 bytes each");

    let reg = ConversionRegistration::register(
        "bf16-to-f32",
        Mapping::Widen {
            from: Dtype::Bf16,
            to: Dtype::F32,
        },
        &corpus,
    )
    .expect("bf16 -> f32 -> bf16 is byte-identical over the whole domain");

    assert_eq!(reg.round_trip.coverage, "complete-domain");
    assert_eq!(reg.round_trip.bytes, 1 << 17);
    assert!(reg.covers_cast(Dtype::Bf16, Dtype::F32));
    // A registration is directional: it does not licence the reverse cast.
    assert!(!reg.covers_cast(Dtype::F32, Dtype::Bf16));

    // The document survives its own canonical encoding, which is what a hub stores.
    let parsed = ConversionRegistration::parse(&reg.canonical_bytes()).expect("reparse");
    assert_eq!(parsed, reg);
}

/// A re-quantization structurally CANNOT register. f32 -> bf16 discards mantissa bits, so
/// the round trip is not byte-identical and `register` refuses. There is no flag that
/// skips this, which is the whole reason CONVERTIBLE can be trusted.
#[test]
fn a_narrowing_cannot_register_no_matter_what_it_asserts() {
    // Real f32 bit patterns with mantissa bits below the bf16 cut.
    let mut corpus = Vec::new();
    for i in 0u32..4096 {
        corpus.extend_from_slice(&(0x3f80_0001u32 + i).to_le_bytes());
    }
    let err = ConversionRegistration::register(
        "f32-to-bf16",
        Mapping::Widen {
            from: Dtype::F32,
            to: Dtype::Bf16,
        },
        &[corpus],
    )
    .expect_err("a lossy cast must not become a registration");
    assert_eq!(refusal_code(err), Code::ROUND_TRIP_LOSSY.as_str());
}

/// An empty corpus round-trips trivially and proves nothing, so it is refused rather than
/// accepted as a zero-case pass.
#[test]
fn an_empty_corpus_case_is_not_a_proof() {
    let err = ConversionRegistration::register(
        "empty",
        Mapping::Widen {
            from: Dtype::Bf16,
            to: Dtype::F32,
        },
        &[Vec::new()],
    )
    .expect_err("zero bytes prove nothing");
    assert_eq!(refusal_code(err), Code::MISSING_FIELD.as_str());
}

// ------------------------------------------------------------------ the frozen pair

/// The corpus pair the layout vectors were mined from: a code topology asking for exactly
/// what the header carries. Editing one side is how every grading arm below is built.
fn frozen_pair() -> (CodeTopology, Header, Closure, Vec<String>) {
    let code = CodeTopology::parse(
        &std::fs::read(vectors().join("cases/layout/code-topology.json")).expect("code topology"),
    )
    .expect("parse code topology");
    let header = Header::parse(
        &std::fs::read(vectors().join("cases/header/plain-fp8.cbor")).expect("header"),
    )
    .expect("parse header");
    let mut closure = Closure::default();
    for entry in std::fs::read_dir(vectors().join("closure")).expect("closure dir") {
        let bytes = std::fs::read(entry.expect("entry").path()).expect("spec");
        closure.insert(EncodingSpec::decode_nested(&bytes).expect("parse spec"));
    }
    let pin: Vec<String> = header
        .encodings
        .iter()
        .map(EncodingSpec::object_id)
        .collect();
    (code, header, closure, pin)
}

fn report_for(
    code: &CodeTopology,
    pair: &(CodeTopology, Header, Closure, Vec<String>),
) -> CompatibilityReport {
    let (_, header, closure, pin) = pair;
    let out = compat::check_compatibility(code, header, closure, pin, Lane::Canonical, &[])
        .expect("check");
    CompatibilityReport::of(code, header, Lane::Canonical, &out).expect("report")
}

fn retype(code: &CodeTopology, to: Dtype) -> CodeTopology {
    let mut out = code.clone();
    for (_, keys) in out.components.iter_mut() {
        for (_, requirement) in keys.iter_mut() {
            requirement.dtype = to;
        }
    }
    out
}

fn bf16_to_f32() -> Registry {
    Registry {
        registrations: vec![ConversionRegistration::register(
            "bf16-to-f32",
            Mapping::Widen {
                from: Dtype::Bf16,
                to: Dtype::F32,
            },
            &complete_domain_16(),
        )
        .expect("register")],
    }
}

/// A contract that asks for exactly what is there is the Satisfied case, and Satisfied is
/// COMPATIBLE with no remedy to name.
#[test]
fn an_exact_pair_grades_compatible() {
    let pair = frozen_pair();
    let report = report_for(&pair.0, &pair);
    assert_eq!(
        grade::grade(&report, &Registry::default()),
        Grade::Compatible
    );
}

/// Cast-only, lossless, REGISTERED: CONVERTIBLE, naming the registration that closes it.
/// The same pair with an empty registry is INCOMPATIBLE — the difference between the two
/// verdicts is the registration, not the grader's opinion.
#[test]
fn a_registered_lossless_cast_grades_convertible_and_names_the_registration() {
    let pair = frozen_pair();
    let wider = retype(&pair.0, Dtype::F32);
    let report = report_for(&wider, &pair);

    match grade::grade(&report, &bf16_to_f32()) {
        Grade::Convertible {
            registration,
            mapping,
        } => {
            assert_eq!(registration, "bf16-to-f32");
            assert_eq!(mapping, "widen bf16 -> f32");
        }
        other => panic!("expected CONVERTIBLE, got {other:?}"),
    }
    assert!(matches!(
        grade::grade(&report, &Registry::default()),
        Grade::Incompatible { .. }
    ));
}

/// `resolve` is the FULL grading: it applies the registration to the code side and checks
/// AGAIN, so CONVERTIBLE means the whole topology is closed rather than that one key's
/// name happened to match.
#[test]
fn resolve_proves_the_registration_closes_the_whole_topology() {
    let pair = frozen_pair();
    let wider = retype(&pair.0, Dtype::F32);
    let (_, graded) = grade::resolve(
        &wider,
        &pair.1,
        &pair.2,
        &pair.3,
        Lane::Canonical,
        &[],
        &bf16_to_f32(),
    )
    .expect("resolve");
    assert!(matches!(graded, Grade::Convertible { .. }));
}

// ------------------------------------------------------ law 2: the remedy from the target

/// The v1 fp8-offered-as-plain-cast class. The checkpoint holds bf16 and the contract
/// wants f8_e4m3fn; a grader that read what it SAW would propose bf16 back and be wrong in
/// every number while right in every name. The remedy must name f8_e4m3fn — the contract's
/// dtype — as the target, and bf16 only as the source a job would produce it from.
#[test]
fn the_remedy_names_the_contracts_target_not_the_observed_encoding() {
    let pair = frozen_pair();
    let narrower = retype(&pair.0, Dtype::F8E4M3FN);
    let report = report_for(&narrower, &pair);

    match grade::grade(&report, &bf16_to_f32()) {
        Grade::Producible { target, source } => {
            assert!(
                target.contains("f8_e4m3fn"),
                "target from the CONTRACT: {target}"
            );
            assert!(
                !target.contains("bf16"),
                "the observation must not be the target: {target}"
            );
            assert!(
                source.contains("bf16"),
                "source is the observation: {source}"
            );
        }
        other => panic!("expected PRODUCIBLE, got {other:?}"),
    }
}

/// PRODUCIBLE is a claim about a job being POSSIBLE, so it is reachable only downward in
/// precision. A contract wanting more precision than the checkpoint holds is not a job we
/// could price — it is INCOMPATIBLE, and stays so unless a registration proves otherwise.
#[test]
fn an_unregistered_widening_is_incompatible_not_producible() {
    let pair = frozen_pair();
    let wider = retype(&pair.0, Dtype::F32);
    let report = report_for(&wider, &pair);
    match grade::grade(&report, &Registry::default()) {
        Grade::Incompatible { reason } => assert!(reason.contains("f32")),
        other => panic!("expected INCOMPATIBLE, got {other:?}"),
    }
}

// -------------------------------------------------- law 3: the over-constraint guard

/// The red arm: plant the forbidden shortcut — grade a cast-only mismatch INCOMPATIBLE
/// while a registration covers it — and require the guard to fire with an exit code. This
/// is the v1 recovery as executable law; if `guard_over_constraint` is ever softened, this
/// test is what notices.
#[test]
fn grading_a_covered_cast_incompatible_is_refused_as_over_constraint() {
    let pair = frozen_pair();
    let wider = retype(&pair.0, Dtype::F32);
    let report = report_for(&wider, &pair);
    let registry = bf16_to_f32();

    let planted = Grade::Incompatible {
        reason: "the shortcut a red arm plants".into(),
    };
    let err = grade::guard_over_constraint(&report, &registry, &planted)
        .expect_err("forbidden over-constraint must refuse");
    assert_eq!(refusal_code(err), Code::UNREGISTERED_CONVERSION.as_str());

    // The honest verdict over the same inputs passes the guard.
    let honest = grade::grade(&report, &registry);
    grade::guard_over_constraint(&report, &registry, &honest)
        .expect("CONVERTIBLE is not over-constraint");
}

/// A shape mismatch is a different model, not a different packaging, and no registration
/// makes it go away — so the guard must not fire and the grade must stay INCOMPATIBLE.
#[test]
fn a_shape_mismatch_stays_incompatible_under_a_full_registry() {
    let pair = frozen_pair();
    let mut reshaped = pair.0.clone();
    for (_, keys) in reshaped.components.iter_mut() {
        for (_, requirement) in keys.iter_mut() {
            requirement.shape = vec![7];
        }
    }
    let report = report_for(&reshaped, &pair);
    let registry = bf16_to_f32();
    let graded = grade::grade(&report, &registry);
    assert!(matches!(graded, Grade::Incompatible { .. }), "{graded:?}");
    grade::guard_over_constraint(&report, &registry, &graded).expect("not a cast-only mismatch");
}

/// The decision document is what a hub stores and a consumer reads back. Each grade must
/// survive its canonical encoding intact — a remedy that did not round-trip would be a
/// remedy the consumer never sees.
#[test]
fn every_grade_survives_the_decision_document() {
    let report = "sha256:".to_string() + &"ab".repeat(32);
    for graded in [
        Grade::Compatible,
        Grade::Convertible {
            registration: "bf16-to-f32".into(),
            mapping: "widen bf16 -> f32".into(),
        },
        Grade::Producible {
            target: "f8_e4m3fn at transformer/w".into(),
            source: "this checkpoint's bf16 tensors".into(),
        },
        Grade::Incompatible {
            reason: "a different model".into(),
        },
    ] {
        let decision = GradeDecision {
            report: report.clone(),
            grade: graded,
        };
        let parsed = GradeDecision::parse(&decision.canonical_bytes()).expect("reparse");
        assert_eq!(parsed, decision);
    }
}

// ------------------------------------------------------------------ the `--json` seam

/// tensorhub shells out to this binary and decodes the object below with unknown fields
/// REFUSED, so the surface is total: every key on every grade, empty where that grade does
/// not carry it. These arms are the contract test for that decoder.
fn tfs(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(args)
        .output()
        .expect("run tfs");
    assert!(
        out.status.success(),
        "tfs {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf8")
}

fn field(json: &str, key: &str) -> String {
    let needle = format!("\"{key}\":\"");
    let start = json
        .find(&needle)
        .unwrap_or_else(|| panic!("{key} absent from {json}"))
        + needle.len();
    let rest = &json[start..];
    rest[..rest.find('"').expect("closing quote")].to_string()
}

fn graded_json(dir: &Path, name: &str, code: &CodeTopology, registry: Option<&Path>) -> String {
    let pair = frozen_pair();
    let report = report_for(code, &pair);
    let report_path = dir.join(format!("{name}.report"));
    std::fs::write(&report_path, report.canonical_bytes()).expect("write report");
    let report_arg = report_path.to_string_lossy().to_string();
    let mut args = vec![report_arg.as_str(), "--json"];
    let registry_arg;
    if let Some(r) = registry {
        registry_arg = r.to_string_lossy().to_string();
        args.push("--registry");
        args.push(&registry_arg);
    }
    let mut full = vec!["grade"];
    full.extend_from_slice(&args);
    tfs(&full)
}

#[test]
fn the_json_surface_is_total_and_carries_the_remedy() {
    let dir = std::env::temp_dir().join(format!("tfs-grade-json-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch");
    let pair = frozen_pair();

    // The registration document the CLI consumes, produced by the CLI's own registration
    // act over the complete 16-bit domain.
    let registry_path = dir.join("bf16-to-f32.cbor");
    let registered = tfs(&[
        "grade",
        "register",
        "bf16-to-f32",
        "widen",
        "bf16:f32",
        "--out",
        &registry_path.to_string_lossy(),
    ]);
    assert!(
        registered.contains("complete-domain"),
        "the CLI default corpus for a 16-bit source is the whole domain: {registered}"
    );

    let every_key = [
        "decision",
        "grade",
        "reason",
        "remedy_mapping",
        "remedy_registration",
        "remedy_source",
        "remedy_target",
        "report",
    ];

    let compatible = graded_json(&dir, "exact", &pair.0, None);
    for key in every_key {
        assert!(
            compatible.contains(&format!("\"{key}\":")),
            "{key} absent: {compatible}"
        );
    }
    assert_eq!(field(&compatible, "grade"), "COMPATIBLE");
    assert_eq!(field(&compatible, "remedy_target"), "");

    let convertible = graded_json(
        &dir,
        "convertible",
        &retype(&pair.0, Dtype::F32),
        Some(&registry_path),
    );
    assert_eq!(field(&convertible, "grade"), "CONVERTIBLE");
    assert_eq!(field(&convertible, "remedy_registration"), "bf16-to-f32");
    assert_eq!(field(&convertible, "remedy_mapping"), "widen bf16 -> f32");

    // The distinction that is the product value: "we could run this if you convert it to
    // X" must not read the same as "no".
    let producible = graded_json(
        &dir,
        "producible",
        &retype(&pair.0, Dtype::F8E4M3FN),
        Some(&registry_path),
    );
    assert_eq!(field(&producible, "grade"), "PRODUCIBLE");
    assert!(field(&producible, "remedy_target").contains("f8_e4m3fn"));

    let incompatible = graded_json(&dir, "incompatible", &retype(&pair.0, Dtype::F32), None);
    assert_eq!(field(&incompatible, "grade"), "INCOMPATIBLE");
    assert!(!field(&incompatible, "reason").is_empty());

    for key in every_key {
        for json in [&convertible, &producible, &incompatible] {
            assert!(
                json.contains(&format!("\"{key}\":")),
                "{key} absent: {json}"
            );
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

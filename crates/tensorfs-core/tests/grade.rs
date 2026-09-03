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

use tensorfs_core::dtype::Dtype;
use tensorfs_core::err::Code;
use tensorfs_core::fit::{self, CompatibilityReport, Custody, Depth, TensorRequirements};
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

/// The corpus pair the layout vectors were mined from: requirements asking for exactly
/// what the header carries. Editing one side is how every grading arm below is built.
fn frozen_pair() -> (TensorRequirements, Header, Closure, Vec<String>) {
    let code = TensorRequirements::parse(
        &std::fs::read(vectors().join("cases/layout/tensor-requirements.json"))
            .expect("tensor requirements"),
    )
    .expect("parse tensor requirements");
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

fn report_at(
    code: &TensorRequirements,
    pair: &(TensorRequirements, Header, Closure, Vec<String>),
    depth: Depth,
) -> CompatibilityReport {
    let (_, header, closure, pin) = pair;
    let out = fit::check_compatibility(code, header, closure, pin, Custody::Canonical, depth)
        .expect("check");
    CompatibilityReport::of(code, header, Custody::Canonical, &out).expect("report")
}

fn report_for(
    code: &TensorRequirements,
    pair: &(TensorRequirements, Header, Closure, Vec<String>),
) -> CompatibilityReport {
    report_at(code, pair, Depth::FirstMismatch)
}

/// The contract of the red arm: the FIRST key asks for a dtype the checkpoint can only be
/// converted down to (PRODUCIBLE on its own) and the LAST asks for a shape the checkpoint
/// does not have (INCOMPATIBLE on its own). Keys are compared in canonical order, so the
/// producible fact is genuinely the one a first-mismatch walk stops on.
fn cast_first_then_shape(code: &TensorRequirements) -> TensorRequirements {
    let mut out = code.clone();
    let keys: usize = out.components.iter().map(|(_, k)| k.len()).sum();
    assert!(keys >= 2, "the arm needs two keys to have a second fact");
    out.components[0].1[0].1.dtype = Dtype::F8E4M3FN;
    let last = out.components.len() - 1;
    let tail = out.components[last].1.len() - 1;
    out.components[last].1[tail].1.shape = vec![7];
    out
}

fn retype(code: &TensorRequirements, to: Dtype) -> TensorRequirements {
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
        Custody::Canonical,
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

// ------------------------------------------- tfs-037: the worst fact, not the first

/// THE RED ARM. A pair whose FIRST mismatch is a dtype cast and whose SECOND is a shape
/// mismatch. Graded on the first fact it reads PRODUCIBLE — "we could run this if you
/// convert it" — and it is a lie: the shape mismatch is a different model and no conversion
/// reaches it. A catalog search built on the first fact would systematically over-promise,
/// which is why ranking walks the whole set.
#[test]
fn the_worst_fact_decides_a_ranking_not_the_first() {
    let pair = frozen_pair();
    let contract = cast_first_then_shape(&pair.0);
    let registry = bf16_to_f32();

    // The preflight walk stops on the cast, and the cast alone is PRODUCIBLE.
    let preflight = report_at(&contract, &pair, Depth::FirstMismatch);
    assert_eq!(preflight.mismatches.len(), 1, "one fact answers preflight");
    assert_eq!(preflight.mismatches[0].code, Code::DTYPE_MISMATCH);
    assert!(
        matches!(
            grade::grade(&preflight, &registry),
            Grade::Producible { .. }
        ),
        "the first fact really is a producible cast — that is what makes this a trap"
    );

    // The full walk finds the shape mismatch behind it, and the WORST fact decides.
    let full = report_at(&contract, &pair, Depth::Full);
    assert_eq!(
        full.mismatches.len(),
        2,
        "the second fact is the one that decides"
    );
    assert_eq!(full.mismatches[0].code, Code::DTYPE_MISMATCH);
    assert_eq!(full.mismatches[1].code, Code::SHAPE_MISMATCH);
    match grade::grade_all(&full, &registry).expect("a full report may be ranked") {
        Grade::Incompatible { reason } => assert!(
            reason.contains("shapes disagree"),
            "the verdict names the fact that blocked it: {reason}"
        ),
        other => panic!("expected INCOMPATIBLE from the whole set, got {other:?}"),
    }

    // And the first-fact grade is still available and still PRODUCIBLE: the two answers are
    // different questions, not a bug fixed by deleting one.
    assert!(matches!(
        grade::grade(&full, &registry),
        Grade::Producible { .. }
    ));
}

/// Ranking a report that STOPPED at the first fact is refused, not guessed. Once the depth
/// is dropped a stopped walk and a complete one are indistinguishable, and "the worst fact
/// of a set that holds one" is the over-promise above wearing the right function's name.
#[test]
fn a_stopped_walk_cannot_be_ranked() {
    let pair = frozen_pair();
    let contract = cast_first_then_shape(&pair.0);
    let preflight = report_at(&contract, &pair, Depth::FirstMismatch);
    let err =
        grade::grade_all(&preflight, &bf16_to_f32()).expect_err("a partial set must not be ranked");
    assert_eq!(refusal_code(err), Code::PARTIAL_MISMATCH_SET.as_str());

    // `grade_report` asks the document which question it answers, so it never has to guess.
    assert!(matches!(
        grade::grade_report(&preflight, &bf16_to_f32()).expect("preflight grades"),
        Grade::Producible { .. }
    ));
}

/// Law 2 THROUGH the selection. The worst fact here is the second one, and its remedy must
/// come from the CONTRACT side of THAT fact — not from the first fact's contract, and not
/// from anything the walk observed. Selecting a level from one fact and a target from
/// another is exactly the v1 bug class with a new way in.
#[test]
fn the_worst_grades_remedy_comes_from_its_own_facts_contract() {
    let pair = frozen_pair();
    // First key: a registered lossless widening — CONVERTIBLE, the mildest non-compatible
    // grade. Second key: a downward cast to the contract's f8_e4m3fn — PRODUCIBLE, worse.
    let mut contract = pair.0.clone();
    contract.components[0].1[0].1.dtype = Dtype::F32;
    contract.components[0].1[1].1.dtype = Dtype::F8E4M3FN;
    let full = report_at(&contract, &pair, Depth::Full);
    assert_eq!(full.mismatches.len(), 2);

    match grade::grade_all(&full, &bf16_to_f32()).expect("rank") {
        Grade::Producible { target, source } => {
            assert!(
                target.contains("f8_e4m3fn"),
                "the target is the SECOND fact's contract dtype: {target}"
            );
            assert!(
                target.contains("blocks.0.norm.weight"),
                "and the second fact's key, not the first's: {target}"
            );
            assert!(
                !target.contains("f32"),
                "the first fact's contract must not leak into the worst fact's remedy: {target}"
            );
            assert!(
                source.contains("bf16"),
                "source is the observation: {source}"
            );
        }
        other => panic!("expected PRODUCIBLE from the worst fact, got {other:?}"),
    }
}

/// The over-constraint guard, read over the WHOLE set. A covered cast sorted ahead of a
/// shape mismatch does not make the INCOMPATIBLE verdict over-constraint: the shape
/// mismatch is real and no registration touches it. A guard that read only the first fact
/// would refuse this correct verdict outright.
#[test]
fn the_guard_reads_the_whole_set_before_it_fires() {
    let pair = frozen_pair();
    let mut contract = pair.0.clone();
    contract.components[0].1[0].1.dtype = Dtype::F32; // covered by bf16-to-f32
    contract.components[0].1[1].1.shape = vec![7]; // covered by nothing at all
    let full = report_at(&contract, &pair, Depth::Full);
    let registry = bf16_to_f32();

    let graded = grade::grade_all(&full, &registry).expect(
        "a covered cast beside a shape \
                                                            mismatch is honestly INCOMPATIBLE",
    );
    assert!(matches!(graded, Grade::Incompatible { .. }), "{graded:?}");

    // And it still fires when EVERY fact is a covered cast: that is the v1 shortcut.
    let all_casts = retype(&pair.0, Dtype::F32);
    let covered = report_at(&all_casts, &pair, Depth::Full);
    let planted = Grade::Incompatible {
        reason: "the shortcut a red arm plants".into(),
    };
    let err = grade::guard_over_constraint(&covered, &registry, &planted)
        .expect_err("forbidden over-constraint must refuse");
    assert_eq!(refusal_code(err), Code::UNREGISTERED_CONVERSION.as_str());
}

/// The cheap path is CHEAP, and both paths are the same code. Preflight stops at the first
/// blocking fact and pays for nothing after it; the full walk pays for the whole topology
/// and is the only one that may be ranked. The numbers are printed rather than asserted —
/// a wall-clock threshold in CI is a flake, and what must hold is the work done, which the
/// mismatch counts state exactly.
#[test]
fn preflight_stops_and_the_full_walk_does_not() {
    use std::time::Instant;
    use tensorfs_core::fit::{Accepts, TensorRequirement};
    use tensorfs_core::header::{Part, Tensor};

    const KEYS: usize = 4096;
    let spec = tensorfs_core::registry::seeds()
        .into_iter()
        .find(|s| s.alias == "plain/1")
        .expect("plain/1")
        .spec;
    let encoding = spec.object_id();
    let tensors: Vec<(String, Tensor)> = (0..KEYS)
        .map(|i| {
            (
                format!("blocks.{i:05}.weight"),
                Tensor {
                    dtype: Dtype::Bf16,
                    shape: vec![1],
                    encoding: encoding.clone(),
                    parts: vec![("value".into(), Part::plan(Dtype::Bf16, vec![1], &[0, 0]))],
                },
            )
        })
        .collect();
    let header = Header {
        configs: vec![],
        assets: vec![],
        encodings: vec![spec],
        components: vec![("core".into(), tensors.clone())],
    };
    // EVERY key disagrees, so the full walk records KEYS facts and preflight records one.
    let code = TensorRequirements {
        contract: tensorfs_core::ids::object_id(b"depth-cost"),
        components: vec![(
            "core".into(),
            tensors
                .iter()
                .map(|(k, _)| {
                    (
                        k.clone(),
                        TensorRequirement {
                            dtype: Dtype::F8E4M3FN,
                            shape: vec![1],
                            accepts: Accepts::Any,
                            pinned: false,
                        },
                    )
                })
                .collect(),
        )],
        constructed_components: vec!["core".into()],
    };
    let pin = vec![encoding];

    let mut timing = Vec::new();
    for depth in [Depth::FirstMismatch, Depth::Full] {
        let t = Instant::now();
        let mut out = None;
        for _ in 0..50 {
            out = Some(
                fit::check_compatibility(
                    &code,
                    &header,
                    &Closure::default(),
                    &pin,
                    Custody::Canonical,
                    depth,
                )
                .expect("check"),
            );
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / 50.0;
        let out = out.unwrap();
        timing.push((depth, us, out.mismatches.len()));
    }
    for (depth, us, n) in &timing {
        eprintln!(
            "{KEYS} keys, depth {:<14} {us:9.1} us   {n} fact(s)",
            depth.name()
        );
    }
    assert_eq!(timing[0].2, 1, "preflight records ONE fact");
    assert_eq!(timing[1].2, KEYS, "the full walk records every one");
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

fn graded_json(
    dir: &Path,
    name: &str,
    code: &TensorRequirements,
    registry: Option<&Path>,
) -> String {
    graded_json_at(dir, name, code, registry, Depth::FirstMismatch)
}

fn graded_json_at(
    dir: &Path,
    name: &str,
    code: &TensorRequirements,
    registry: Option<&Path>,
    depth: Depth,
) -> String {
    let pair = frozen_pair();
    let report = report_at(code, &pair, depth);
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

    // tfs-037 over the seam tensorhub reads. The SAME contract, graded from a report that
    // stopped at the first fact and from one that walked the whole topology: PRODUCIBLE
    // against INCOMPATIBLE. A catalog search that read the first would offer a conversion
    // that cannot work, and `depth` is how a consumer tells which answer it holds.
    let contract = cast_first_then_shape(&pair.0);
    let stopped = graded_json_at(
        &dir,
        "worst-first-fact",
        &contract,
        Some(&registry_path),
        Depth::FirstMismatch,
    );
    assert_eq!(field(&stopped, "grade"), "PRODUCIBLE");
    assert_eq!(field(&stopped, "depth"), "first-mismatch");
    assert!(stopped.contains("\"mismatches\":1"), "{stopped}");

    let ranked = graded_json_at(
        &dir,
        "worst-full",
        &contract,
        Some(&registry_path),
        Depth::Full,
    );
    assert_eq!(field(&ranked, "grade"), "INCOMPATIBLE");
    assert_eq!(field(&ranked, "depth"), "full");
    assert!(ranked.contains("\"mismatches\":2"), "{ranked}");
    assert!(field(&ranked, "reason").contains("shapes disagree"));
    assert_eq!(
        field(&ranked, "remedy_target"),
        "",
        "totality holds on the worst grade too"
    );

    for key in every_key {
        for json in [&convertible, &producible, &incompatible, &stopped, &ranked] {
            assert!(
                json.contains(&format!("\"{key}\":")),
                "{key} absent: {json}"
            );
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

// ------------------------------------------------------- tfs-039: the batch grading surface

/// Every frozen `compat_report` vector, once per line, as a JSONL batch document.
fn frozen_reports() -> Vec<Vec<u8>> {
    let dir = vectors().join("cases/layout");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("the layout vectors")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
        .collect();
    paths.sort();
    let reports: Vec<Vec<u8>> = paths
        .iter()
        .map(|p| std::fs::read(p).unwrap())
        .filter(|b| CompatibilityReport::parse(b).is_ok())
        .collect();
    assert!(
        reports.len() >= 4,
        "the corpus carries {} frozen reports",
        reports.len()
    );
    reports
}

fn tfs_status(args: &[&str], stdin: Option<&[u8]>) -> (bool, String, String) {
    use std::io::Write;
    let mut child = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn tfs");
    {
        let mut pipe = child.stdin.take().expect("stdin");
        if let Some(bytes) = stdin {
            if let Err(error) = pipe.write_all(bytes) {
                assert_eq!(
                    error.kind(),
                    std::io::ErrorKind::BrokenPipe,
                    "write stdin: {error}"
                );
            }
        }
    }
    let out = child.wait_with_output().expect("wait");
    (
        out.status.success(),
        String::from_utf8(out.stdout).expect("utf8"),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

/// The batch line is the SAME bytes the single-pair `--json` prints. Not "the same fields"
/// — the same `grade_json`, so there is one machine surface with two ways in and a
/// preflight caller is never made to batch to keep getting the answer it already decodes.
#[test]
fn a_batch_line_is_byte_identical_to_the_single_pair_json() {
    let dir = std::env::temp_dir().join(format!("tfs-batch-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let reports = frozen_reports();

    let mut jsonl = Vec::new();
    let mut singles = Vec::new();
    for (i, bytes) in reports.iter().enumerate() {
        let p = dir.join(format!("{i}.json"));
        std::fs::write(&p, bytes).unwrap();
        singles.push(tfs(&[
            "grade",
            p.to_str().unwrap(),
            "--json",
            "--registry",
            "",
        ]));
        jsonl.extend_from_slice(bytes);
        jsonl.push(b'\n');
    }
    let batch_path = dir.join("batch.jsonl");
    std::fs::write(&batch_path, &jsonl).unwrap();

    let batched = tfs(&[
        "grade",
        "batch",
        batch_path.to_str().unwrap(),
        "--registry",
        "",
    ]);
    let lines: Vec<&str> = batched.lines().collect();
    assert_eq!(
        lines.len(),
        reports.len(),
        "one answer per document, in order"
    );
    for (line, single) in lines.iter().zip(singles.iter()) {
        assert_eq!(
            *line,
            single.trim_end(),
            "the batch line is the --json line"
        );
    }

    // `serve` reads the same framing from stdin and answers identically.
    let (ok, served, _) = tfs_status(&["grade", "serve", "--registry", ""], Some(&jsonl));
    assert!(ok);
    assert_eq!(served, batched);

    // Blank lines are framing, not documents.
    let mut padded = Vec::new();
    for bytes in &reports {
        padded.push(b'\n');
        padded.extend_from_slice(bytes);
        padded.push(b'\n');
    }
    let (ok, spaced, _) = tfs_status(&["grade", "serve", "--registry", ""], Some(&padded));
    assert!(ok);
    assert_eq!(spaced, batched);

    std::fs::remove_dir_all(&dir).ok();
}

/// A partial answer is not an answer. A line that does not decode stops the batch, names
/// the line, and exits failure — it never becomes an output line in the grade shape, which
/// would be a fifth grade with a different spelling.
#[test]
fn a_batch_refuses_at_the_line_it_cannot_grade() {
    let reports = frozen_reports();
    let mut jsonl = Vec::new();
    for bytes in reports.iter().take(2) {
        jsonl.extend_from_slice(bytes);
        jsonl.push(b'\n');
    }
    jsonl.extend_from_slice(br#"{"contract":"nonsense"}"#);
    jsonl.push(b'\n');
    jsonl.extend_from_slice(&reports[0]);
    jsonl.push(b'\n');

    let (ok, out, err) = tfs_status(&["grade", "serve", "--registry", ""], Some(&jsonl));
    assert!(!ok, "a batch with an undecodable line fails");
    assert_eq!(out.lines().count(), 2, "the verdicts before it still stand");
    assert!(err.contains("line 3"), "the refusal names the line: {err}");
    assert!(err.starts_with("REFUSED "), "{err}");
    for line in out.lines() {
        assert!(line.contains("\"grade\":\""), "{line}");
    }
}

/// The ranking law survives the batch: a report taken at `first-mismatch` carrying a full
/// set is `PARTIAL_MISMATCH_SET`, and it refuses in a stream exactly as it does alone.
#[test]
fn the_depth_law_holds_inside_a_batch() {
    let red = std::fs::read(vectors().join("cases/layout-red/worst-graded-partial.json"))
        .expect("the frozen red arm");
    let mut jsonl = frozen_reports()[0].clone();
    jsonl.push(b'\n');
    jsonl.extend_from_slice(&red);
    jsonl.push(b'\n');

    let (ok, out, err) = tfs_status(&["grade", "serve", "--registry", ""], Some(&jsonl));
    assert!(!ok);
    assert_eq!(out.lines().count(), 1);
    assert!(
        err.contains("PARTIAL_MISMATCH_SET"),
        "a stopped walk relabelled full is refused in a batch too: {err}"
    );
}

/// `--prove-with` holds ONE pair's two documents. A batch carries many, so it is refused
/// rather than silently applied to the first or to none.
#[test]
fn prove_with_is_refused_on_a_batch() {
    let reports = frozen_reports();
    let mut jsonl = reports[0].clone();
    jsonl.push(b'\n');
    let (ok, out, err) = tfs_status(
        &[
            "grade",
            "serve",
            "--registry",
            "",
            "--requirements",
            "/nonexistent",
            "--header",
            "/nonexistent",
        ],
        Some(&jsonl),
    );
    assert!(!ok);
    assert!(out.is_empty());
    assert!(err.contains("single-pair only"), "{err}");
}

//! Platform registry seeds. Identity is the spec object's digest; these NAMES are aliases
//! (display/discovery only). Promotion or an alias rename changes no artifact id.

use crate::dtype::Dtype;
use crate::ids::ObjectRef;
use crate::relation::{Carrier, Dim, Relation};
use crate::spec::{EncodingSpec, Role};
use crate::vectors::{Case, EncodingVectors, RoleBits};

fn set(d: &[Dtype]) -> Carrier {
    let mut v = d.to_vec();
    v.sort_by_key(|x| x.name());
    Carrier::Set(v)
}

fn role(carrier: Carrier, shape: Relation) -> Role {
    Role { carrier, shape }
}

fn dims(d: Vec<Dim>) -> Relation {
    Relation::Dims(d)
}

fn logical(names: &[Dtype]) -> Vec<Dtype> {
    let mut v = names.to_vec();
    v.sort_by_key(|x| x.name());
    v
}

/// plain/1 conformance vectors: little-endian carrier law, signed zero, NaN payload,
/// infinity and a rank-0 scalar. Raw bit patterns, never decimal spellings.
pub fn plain_vectors() -> EncodingVectors {
    let bf16 = vec![
        0x80, 0x3f, // 1.0
        0x00, 0x80, // -0.0
        0x80, 0x7f, // +inf
        0x80, 0xff, // -inf
        0xc1, 0x7f, // NaN, payload 0x41
        0x00, 0x00, // +0.0
    ];
    let f32v = vec![0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0xc0]; // 1.0, -2.0
    let scalar = vec![0x2a];
    let mk = |name: &str, dt: Dtype, shape: Vec<u64>, bits: Vec<u8>| Case {
        name: name.to_string(),
        logical_dtype: dt,
        logical_shape: shape.clone(),
        roles: vec![(
            "value".to_string(),
            RoleBits {
                dtype: dt,
                shape,
                bits: bits.clone(),
            },
        )],
        expect_logical_bits: bits,
    };
    EncodingVectors {
        cases: vec![
            mk("bf16-specials-2x3", Dtype::Bf16, vec![2, 3], bf16),
            mk("f32-le-1x2", Dtype::F32, vec![1, 2], f32v),
            mk("u8-rank0-scalar", Dtype::U8, vec![], scalar),
        ],
    }
}

/// A registry entry: the canonical spec object plus, when it has been PRODUCER-MINED, the
/// exact vector document whose digest the spec references. Carrying both together is what
/// makes it impossible to ship a spec pointing at vector bytes nobody stores.
#[derive(Clone)]
pub struct Seed {
    pub alias: &'static str,
    pub spec: EncodingSpec,
    pub vectors: Option<EncodingVectors>,
}

fn spec(
    logical_dtypes: Vec<Dtype>,
    logical_rank: Option<u64>,
    roles: Vec<(&str, Role)>,
    vectors: Option<ObjectRef>,
) -> EncodingSpec {
    let mut r: Vec<(String, Role)> = roles.into_iter().map(|(n, x)| (n.to_string(), x)).collect();
    r.sort_by(|a, b| a.0.cmp(&b.0));
    EncodingSpec {
        logical_dtypes,
        logical_rank,
        roles: r,
        vectors,
    }
}

/// One seed, with the spec's `vectors` reference DERIVED from the vector document rather
/// than transcribed beside it.
fn seed(
    alias: &'static str,
    logical_dtypes: Vec<Dtype>,
    logical_rank: Option<u64>,
    roles: Vec<(&str, Role)>,
    vectors: Option<EncodingVectors>,
) -> Seed {
    let vref = vectors.as_ref().map(|v| v.fixture_ref());
    Seed {
        alias,
        spec: spec(logical_dtypes, logical_rank, roles, vref),
        vectors,
    }
}

/// PRODUCER-MINED vector documents, frozen as data and included at build time. They are
/// parsed rather than transcribed, so a spec's `vectors` digest is DERIVED from the exact
/// bytes under `vectors/mined/` and cannot drift from them. See scripts/mine-vectors.py for
/// which real producer emitted each one.
fn mined(name: &str) -> EncodingVectors {
    let bytes: &[u8] = match name {
        "fp8-rowwise" => include_bytes!("../../../vectors/mined/fp8-rowwise.json"),
        "fp8-rowwise-keepdim" => include_bytes!("../../../vectors/mined/fp8-rowwise-keepdim.json"),
        "fp8-scaled-scalar" => include_bytes!("../../../vectors/mined/fp8-scaled-scalar.json"),
        "fp8-scaled-scalar-weight-only" => {
            include_bytes!("../../../vectors/mined/fp8-scaled-scalar-weight-only.json")
        }
        "nvfp4-w4a4" => include_bytes!("../../../vectors/mined/nvfp4-w4a4.json"),
        "mxfp8" => include_bytes!("../../../vectors/mined/mxfp8.json"),
        other => panic!("no mined vector set {other:?}"),
    };
    EncodingVectors::parse_fixture(bytes).expect("mined vectors are frozen canonical data")
}

/// The seeded platform namespace.
pub fn seeds() -> Vec<Seed> {
    let all = logical(&[
        Dtype::F64,
        Dtype::F32,
        Dtype::F16,
        Dtype::Bf16,
        Dtype::F8E4M3FN,
        Dtype::F8E5M2,
        Dtype::I64,
        Dtype::I32,
        Dtype::I16,
        Dtype::I8,
        Dtype::U8,
        Dtype::Bool,
    ]);
    let float = logical(&[Dtype::Bf16, Dtype::F16, Dtype::F32]);

    let mut out = vec![
        seed(
            "plain/1",
            all,
            None,
            vec![("value", role(Carrier::SameAsLogical, Relation::Same))],
            Some(plain_vectors()),
        ),
        seed(
            "fp8-rowwise/1",
            float.clone(),
            Some(2),
            vec![
                ("data", role(set(&[Dtype::F8E4M3FN]), Relation::Same)),
                (
                    "scale",
                    role(set(&[Dtype::F32]), dims(vec![Dim::Axis { axis: 0 }])),
                ),
            ],
            Some(mined("fp8-rowwise")),
        ),
        // The SAME alias, a DIFFERENT physical variant: rank-2 `[out, 1]` rather than
        // rank-1 `[out]`. Roles are EXACT, so this is its own spec digest rather than an
        // `optional`/loose shape on the one above; the alias groups the two for display,
        // exactly as it does nvfp4's four.
        //
        // PROVENANCE CORRECTED 2026-08-25 (tfs-009, decisions #243/#244). This doc used to
        // cite `examples/safetensors-headers/minimax_h3_fl2va_fp8_scaled` and its "362
        // `.weight_scale` siblings at F32 [out, 1]" as the shipped H3 DiT's geometry. That
        // header was measured WRONG and retired: no publisher ever shipped it. The rank-2
        // `[out, 1]` geometry is real and stays -- its vectors are producer-mined from v1's
        // own quantizer (decisions #187, the Cozy-produced hub tree) -- but it is NOT what
        // ComfyUI's fp8_scaled ships. The real per-producer facts, measured off pinned
        // bytes in h3-evidence: Comfy `fp8_scaled` = rank-0 SCALAR weight_scale + rank-0
        // input_scale (the entry below); Comfy `int8_convrot` = rank-2 [out, 1]; Comfy
        // `nvfp4_awq` = rank-2 F8_E4M3 block scales. Scale rank is a per-producer fact.
        seed(
            "fp8-rowwise/1",
            float.clone(),
            Some(2),
            vec![
                ("data", role(set(&[Dtype::F8E4M3FN]), Relation::Same)),
                (
                    "scale",
                    role(
                        set(&[Dtype::F32]),
                        dims(vec![Dim::Axis { axis: 0 }, Dim::Lit { value: 1 }]),
                    ),
                ),
            ],
            Some(mined("fp8-rowwise-keepdim")),
        ),
        // A THIRD physical variant of the same alias, and the one the largest H3
        // distribution by download count actually ships. MEASURED, not inferred, from
        // `Comfy-Org/MiniMax-H3@4cc1d817` `minimax_h3_fl2va_pruned_fp8_scaled.safetensors`
        // (header bytes sha256 f2331000434d1e54..., pinned in h3-evidence): 200 F8_E4M3
        // weights, 200 rank-0 SCALAR F32 `.weight_scale`, 150 rank-0 scalar F32
        // `.input_scale`. A reader keyed on "fp8 implies rank-2" mis-strides every one of
        // them -- which is the whole reason scale rank is a spec fact and never a dtype
        // inference (decisions #244, narrowing #187 rather than contradicting it).
        //
        // VECTORED 2026-08-25 (job-001's open conformance clause). The old note said
        // qualifying this needed ComfyUI's quantizer over weights that cannot transit this
        // machine. Half of that was true and the wrong half was load-bearing: the QUANTIZER
        // is not the weights. ComfyUI v0.33.0 (`2f35f4a0`) and comfy_kitchen's compiled
        // `quantize_fp8`/`dequantize_fp8` run on CPU, so `scripts/mine-vectors.py` IMPORTS
        // the real producer class and quantizes a 16x32 synthetic tensor with the planted
        // values that break decoders. Not a second implementation and not a transcription —
        // the producer itself, on bytes that were never a model.
        seed(
            "fp8-scaled-scalar/1",
            float.clone(),
            Some(2),
            vec![
                ("data", role(set(&[Dtype::F8E4M3FN]), Relation::Same)),
                ("input_scale", role(set(&[Dtype::F32]), dims(vec![]))),
                ("weight_scale", role(set(&[Dtype::F32]), dims(vec![]))),
            ],
            Some(mined("fp8-scaled-scalar")),
        ),
        // A FOURTH variant, and the one that vindicates "no `optional: true`" on real bytes.
        // The SAME file carries both: measured on the pinned pruned fp8_scaled DiT, 200
        // tensors have `.weight_scale` and only 150 have `.input_scale` -- `out_proj`,
        // `qkv_proj` and `mlp.fc1` carry both, `mlp.fc2` carries the weight scale alone
        // (150 both / 50 weight-only / 0 input-only, no overlap ambiguity). One artifact,
        // one alias, TWO physical role sets. An `optional` flag would have collapsed them
        // into one digest that describes neither tensor exactly; two digests describe both.
        seed(
            "fp8-scaled-scalar/1",
            float.clone(),
            Some(2),
            vec![
                ("data", role(set(&[Dtype::F8E4M3FN]), Relation::Same)),
                ("weight_scale", role(set(&[Dtype::F32]), dims(vec![]))),
            ],
            Some(mined("fp8-scaled-scalar-weight-only")),
        ),
        // DELETED (tfs-008 deletion pass): the int8 `w8a8-perchannel/1` seed. It had no
        // producer, no consumer, and a NAME that actively misleads -- everything in this
        // ecosystem that calls itself w8a8 is fp8-e4m3 per output row (v1's own constant is
        // `W8A8_FLAVOR = "fp8-w8a8"`), which is the `fp8-rowwise/1` entry above. The only
        // real int8 producer in the quarry is svdq's low-rank factor quantizer, which is
        // per-block-32 over rank-32 factors, not per-channel -- a different geometry that
        // belongs to `svdq-microscale/1`. Schema with no path is what the conventions forbid.
        seed(
            "mxfp8/1",
            float.clone(),
            Some(2),
            vec![
                ("data", role(set(&[Dtype::F8E4M3FN]), Relation::Same)),
                (
                    "scale",
                    role(
                        set(&[Dtype::U8]),
                        dims(vec![
                            Dim::Axis { axis: 0 },
                            Dim::CeilDiv { axis: 1, by: 32 },
                        ]),
                    ),
                ),
            ],
            Some(mined("mxfp8")),
        ),
        Seed {
            alias: "svdq-microscale/1",
            vectors: None,
            spec: spec(
                float.clone(),
                Some(2),
                vec![
                    (
                        "weight",
                        role(
                            set(&[Dtype::U8]),
                            dims(vec![Dim::Axis { axis: 0 }, Dim::Div { axis: 1, by: 2 }]),
                        ),
                    ),
                    (
                        "wscale",
                        role(
                            set(&[Dtype::F8E4M3FN]),
                            dims(vec![
                                Dim::Div { axis: 0, by: 128 },
                                Dim::Div { axis: 1, by: 64 },
                                Dim::Lit { value: 1 },
                                Dim::Lit { value: 8 },
                                Dim::Lit { value: 4 },
                                Dim::Lit { value: 4 },
                                Dim::Lit { value: 4 },
                            ]),
                        ),
                    ),
                    (
                        "proj_down",
                        role(
                            set(&[Dtype::Bf16]),
                            dims(vec![Dim::Lit { value: 32 }, Dim::Axis { axis: 1 }]),
                        ),
                    ),
                    (
                        "proj_up",
                        role(
                            set(&[Dtype::Bf16]),
                            dims(vec![Dim::Axis { axis: 0 }, Dim::Lit { value: 32 }]),
                        ),
                    ),
                    (
                        "smooth",
                        role(set(&[Dtype::Bf16]), dims(vec![Dim::Axis { axis: 1 }])),
                    ),
                ],
                None,
            ),
        },
    ];

    // nvfp4 W4A4 -- roles are EXACT, so each producer variation is its own spec digest;
    // the alias groups the four for display. Geometry mined from v1 gen_worker w4a4.py.
    let base = vec![
        (
            "weight",
            role(
                set(&[Dtype::U8]),
                dims(vec![Dim::Axis { axis: 0 }, Dim::Div { axis: 1, by: 2 }]),
            ),
        ),
        (
            "weight_scale",
            role(
                set(&[Dtype::F8E4M3FN]),
                dims(vec![Dim::Axis { axis: 0 }, Dim::Div { axis: 1, by: 16 }]),
            ),
        ),
        ("weight_scale_2", role(set(&[Dtype::F32]), dims(vec![]))),
    ];
    for (suffix, extra) in [
        ("", vec![]),
        (
            "+input_scale",
            vec![("input_scale", role(set(&[Dtype::F32]), dims(vec![])))],
        ),
        (
            "+pre_quant_scale",
            vec![(
                "pre_quant_scale",
                role(set(&[Dtype::F32]), dims(vec![Dim::Axis { axis: 1 }])),
            )],
        ),
        (
            "+input_scale+pre_quant_scale",
            vec![
                ("input_scale", role(set(&[Dtype::F32]), dims(vec![]))),
                (
                    "pre_quant_scale",
                    role(set(&[Dtype::F32]), dims(vec![Dim::Axis { axis: 1 }])),
                ),
            ],
        ),
    ] {
        let mut roles = base.clone();
        roles.extend(extra);
        // Only the BASE variant has producer-mined vectors: v1's quantizer emits exactly
        // the (weight, weight_scale, weight_scale_2) triple. The three variants that add an
        // input_scale / pre_quant_scale are real physical contracts with no mined bytes yet,
        // so they store and inspect and refuse executable qualification -- which is the
        // vectorless rule doing its job rather than an oversight.
        out.push(seed(
            "nvfp4-w4a4/1",
            logical(&[Dtype::Bf16, Dtype::F16]),
            Some(2),
            roles,
            if suffix.is_empty() {
                Some(mined("nvfp4-w4a4"))
            } else {
                None
            },
        ));
    }
    out
}

/// Every VECTORED seed, as (spec digest, the vector set it pins). This is the evidence
/// index a capability record is checked against: a record names the exact vector set its
/// implementation passed, and the only way that name can be true is if the spec pins it.
pub fn vector_refs() -> Vec<(String, crate::ids::ObjectRef)> {
    seeds()
        .iter()
        .filter_map(|s| s.spec.vectors.clone().map(|v| (s.spec.object_id(), v)))
        .collect()
}

/// The qualified (encoding, device) pairs. FAIL-CLOSED: anything absent here refuses, and
/// a record cannot exist without naming the vector set its implementation passed.
pub fn capability_records() -> crate::capability::CapabilityRecords {
    let mut records = Vec::new();
    for s in seeds() {
        // Only a VECTORED spec can be qualified at all. The reference decoders in the `tfs`
        // conformance harness are what these records qualify, on the CPU they run on;
        // every accelerator class is absent until something is qualified there.
        if let Some(v) = s.spec.vectors.clone() {
            records.push(crate::capability::CapabilityRecord {
                encoding: s.spec.object_id(),
                device: "cpu".to_string(),
                implementation: "tfs-conform-reference".to_string(),
                vectors: v,
                note: "reference decoder in the tfs conformance harness, qualified bit-for-bit against producer-mined vectors".to_string(),
            });
        }
    }
    crate::capability::CapabilityRecords { records }
}

/// Launch border policy admits platform-aliased spec digests only.
pub fn platform_digests() -> Vec<String> {
    seeds().iter().map(|s| s.spec.object_id()).collect()
}

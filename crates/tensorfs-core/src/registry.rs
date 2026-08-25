//! Platform registry seeds. Identity is the spec object's digest; these NAMES are aliases
//! (display/discovery only). Promotion or an alias rename changes no artifact id.

use crate::dtype::Dtype;
use crate::ids::{Doc, ObjectRef};
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

pub struct Seed {
    pub alias: &'static str,
    pub spec: EncodingSpec,
}

fn spec(
    logical_dtypes: Vec<Dtype>,
    logical_rank: Option<u64>,
    roles: Vec<(&str, Role)>,
    doc: &str,
    vectors: Option<ObjectRef>,
) -> EncodingSpec {
    let mut r: Vec<(String, Role)> = roles.into_iter().map(|(n, x)| (n.to_string(), x)).collect();
    r.sort_by(|a, b| a.0.cmp(&b.0));
    EncodingSpec {
        logical_dtypes,
        logical_rank,
        roles: r,
        doc: Some(doc.to_string()),
        vectors,
    }
}

/// The seeded platform namespace. Every encoded seed is deliberately VECTORLESS:
/// vectors must be producer-mined (tfs-008), so these store and inspect but can never
/// qualify an executable implementation until real producer bytes are transcribed.
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
    let vec_ref = ObjectRef::of(&plain_vectors().canonical_bytes());

    let mut out = vec![
        Seed {
            alias: "plain/1",
            spec: spec(
                all,
                None,
                vec![("value", role(Carrier::SameAsLogical, Relation::Same))],
                "Plain little-endian row-major storage: one `value` role carrying the logical bytes verbatim. Entry zero of the registry; not a special absent form.",
                Some(vec_ref),
            ),
        },
        Seed {
            alias: "fp8-rowwise/1",
            spec: spec(
                float.clone(),
                Some(2),
                vec![
                    ("data", role(set(&[Dtype::F8E4M3FN]), Relation::Same)),
                    ("scale", role(set(&[Dtype::F32]), dims(vec![Dim::Axis { axis: 0 }]))),
                ],
                "value[i,j] = data[i,j] * scale[i]. Per-output-row fp8 e4m3 weights with one f32 scale per row.",
                None,
            ),
        },
        Seed {
            alias: "w8a8-perchannel/1",
            spec: spec(
                float.clone(),
                Some(2),
                vec![
                    ("data", role(set(&[Dtype::I8]), Relation::Same)),
                    ("scale", role(set(&[Dtype::F32]), dims(vec![Dim::Axis { axis: 0 }]))),
                ],
                "Transitional int8 W8A8: value[i,j] = data[i,j] * scale[i], per-output-channel f32 scales.",
                None,
            ),
        },
        Seed {
            alias: "mxfp8/1",
            spec: spec(
                float.clone(),
                Some(2),
                vec![
                    ("data", role(set(&[Dtype::F8E4M3FN]), Relation::Same)),
                    (
                        "scale",
                        role(set(&[Dtype::U8]), dims(vec![Dim::Axis { axis: 0 }, Dim::CeilDiv { axis: 1, by: 32 }])),
                    ),
                ],
                "OCP Microscaling MXFP8: E4M3 elements with one E8M0 power-of-two scale per 32-element block along the last axis; the E8M0 exponent byte rides a u8 carrier. Execution floors are capability records, never spec fields (native SM100; software-scaled sm89-class; decoded-float below).",
                None,
            ),
        },
        Seed {
            alias: "svdq-microscale/1",
            spec: spec(
                float.clone(),
                Some(2),
                vec![
                    ("weight", role(set(&[Dtype::U8]), dims(vec![Dim::Axis { axis: 0 }, Dim::Div { axis: 1, by: 2 }]))),
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
                    ("proj_down", role(set(&[Dtype::Bf16]), dims(vec![Dim::Lit { value: 32 }, Dim::Axis { axis: 1 }]))),
                    ("proj_up", role(set(&[Dtype::Bf16]), dims(vec![Dim::Axis { axis: 0 }, Dim::Lit { value: 32 }]))),
                    ("smooth", role(set(&[Dtype::Bf16]), dims(vec![Dim::Axis { axis: 1 }]))),
                ],
                "svdq micro-scale. The `wscale` geometry is TRANSCRIBED from v1 nunchaku.micro-scale.v1.json: logical [out, in/16] e4m3 scales written under the warp-lane permutation (0,5,1,4,3,2,6) of extents (out/128, 1, 4, 4, 8, (in/16)/4, 4), so the stored grid is [out/128, in/64, 1, 8, 4, 4, 4]; exact division enforces out%128==0 and (in/16)%4==0. PROVISIONAL: the rank-32 low-rank branch and smoothing-factor geometry await producer-mined confirmation (tfs-008).",
                None,
            ),
        },
    ];

    // nvfp4 W4A4 — roles are EXACT, so each producer variation is its own spec digest;
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
    let doc = "nvfp4 W4A4 (modelopt export shapes): weight u8 [out, in/2] packed e2m1 nibble pairs, element 2j in the LOW nibble; weight_scale e4m3 [out, in/16] per-16-block scales, flat row-major; weight_scale_2 f32 scalar. Dequant: W ~= e2m1(weight) * weight_scale.float() * weight_scale_2. Detection is the (u8 weight + e4m3 weight_scale + f32 weight_scale_2) triple, never a name list.";
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
        let full = format!("{doc} Variant roles: base{suffix}.");
        out.push(Seed {
            alias: "nvfp4-w4a4/1",
            spec: spec(
                logical(&[Dtype::Bf16, Dtype::F16]),
                Some(2),
                roles,
                &full,
                None,
            ),
        });
    }
    out
}

/// Launch border policy admits platform-aliased spec digests only.
pub fn platform_digests() -> Vec<String> {
    seeds().iter().map(|s| s.spec.object_id()).collect()
}

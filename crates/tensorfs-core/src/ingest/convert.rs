//! Converters and the PLAN.
//!
//! The plan is a pure function of `(headers, target)`. It reads zero tensor bytes, which is
//! the whole point: a converts-nothing verdict, a missing companion role, a double-encode,
//! and an unfit placement are all decidable for the cost of a header read — cents, not a
//! rented GPU. `Plan::bytes_read()` is the claim, and the CLI prints it.
//!
//! Three converter CLASSES, and the class decides where the work may run:
//!   Identity     — a re-container. Bytes stream through untouched.
//!   Permutation  — bytes are reordered, never recomputed. CPU-streaming, either border site.
//!   Value        — numbers change (quantize, truncate). REFUSES here, naming the priced job.
//!
//! Untouched tensors carry by OBJECT REFERENCE. Re-ingesting an already-canonical component
//! copies its segment refs into the new header: zero re-hash, zero re-upload, CAS-asserted.

use crate::dtype::{checked_bytes, Dtype};
use crate::err::{refuse, Code, Result};
use crate::header::{Header, Part};
use crate::ids::{Doc, ObjectRef};
use crate::limits;
use crate::spec::EncodingSpec;

use super::carrier::SourceHeader;
use super::fingerprint::{is_marker, module_of, role_of};

// ---------------------------------------------------------------- converter identity

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Identity,
    Permutation,
    Value,
}

impl Class {
    pub fn name(self) -> &'static str {
        match self {
            Class::Identity => "identity",
            Class::Permutation => "permutation",
            Class::Value => "value",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Converter {
    pub name: &'static str,
    pub dialect: &'static str,
    pub class: Class,
    pub doc: &'static str,
    /// Quarry-derived geometry that has NOT yet met the real bytes it describes (tfs-010).
    /// Stated in the type so a receipt cannot quietly claim more than the evidence supports.
    pub validated_on_real_bytes: bool,
}

pub const CONVERTERS: [Converter; 4] = [
    Converter {
        name: "diffusers.identity/1",
        dialect: "safetensors.diffusers",
        class: Class::Identity,
        doc: "Diffusers component spelling IS the canonical logical key spelling. Each key \
              becomes one logical tensor; `.weight_scale` siblings fold into their base as \
              encoding ROLES; marker tensors are dropped. Bytes stream through untouched.",
        validated_on_real_bytes: true,
    },
    Converter {
        name: "single_file.identity/1",
        dialect: "safetensors.single_file",
        class: Class::Identity,
        doc: "One file carrying several components, split on the reviewed component prefix \
              map. Same byte path as the diffusers converter — packaging is not identity.",
        validated_on_real_bytes: true,
    },
    Converter {
        name: "cozytensors.inherit/1",
        dialect: "cozytensors",
        class: Class::Identity,
        doc: "An already-canonical source. Every untouched tensor carries by OBJECT \
              REFERENCE; nothing is re-hashed and nothing is re-uploaded.",
        validated_on_real_bytes: true,
    },
    Converter {
        name: "h3.native/1",
        dialect: "safetensors.single_file",
        class: Class::Permutation,
        doc: "H3 native spelling: head-interleaved fused QKV split into canonical per-head \
              contiguous Q|K|V, SwiGLU fused MLP halves put in canonical (gate, up) order, \
              video-VAE fused QKV and `w1` mapped. Reordering only — never arithmetic.",
        validated_on_real_bytes: false,
    },
];

pub fn converter(name: &str) -> Result<&'static Converter> {
    match CONVERTERS.iter().find(|c| c.name == name) {
        Some(c) => Ok(c),
        None => refuse(
            Code::UNKNOWN_FIELD,
            format!(
                "converter {name:?} is not one of the reviewed set {:?}",
                CONVERTERS.iter().map(|c| c.name).collect::<Vec<_>>()
            ),
        ),
    }
}

// ---------------------------------------------------------------- the target

/// Component-granular, never one rule over the tree: the real bindable H3 tree is an
/// fp8-blockwise text encoder BESIDE a BF16 transformer, and a single tree-wide target
/// cannot say that.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Target {
    pub components: Vec<(String, String)>,
}

impl Target {
    /// `unet=plain/1,text_encoder=plain/1`
    pub fn parse(s: &str) -> Result<Target> {
        let mut components = Vec::new();
        for part in s.split(',').filter(|p| !p.is_empty()) {
            match part.split_once('=') {
                Some((c, e)) => components.push((c.to_string(), e.to_string())),
                None => {
                    return refuse(
                        Code::MISSING_FIELD,
                        format!("target {part:?}: expected component=encoding_alias"),
                    )
                }
            }
        }
        if components.is_empty() {
            return refuse(Code::MISSING_FIELD, "target names no component");
        }
        Ok(Target { components })
    }
    pub fn get(&self, component: &str) -> Result<&str> {
        match self.components.iter().find(|(c, _)| c == component) {
            Some((_, e)) => Ok(e.as_str()),
            None => refuse(
                Code::MISSING_FIELD,
                format!(
                    "the target names no encoding for component {component:?} — targets are \
                     component-granular and a missing component is never defaulted"
                ),
            ),
        }
    }
}

// ---------------------------------------------------------------- planned work

/// Where one output ROLE's bytes come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Bytes {
    /// Stream the carrier's declared run straight through: no buffer, no transform.
    Stream { file: usize, key: String },
    /// Carry the already-admitted objects by reference: zero re-hash, zero re-upload.
    Inherit { part: Box<Part> },
    /// Reorder a resident buffer. Bounded by `CONVERT_TENSOR_MAX_BYTES` and never arithmetic.
    Permute {
        file: usize,
        key: String,
        xform: Xform,
    },
}

/// The ONE fill-transform implementation lives in `crate::fill` and is invoked from exactly
/// two places: this staging pass and the serve-side fill path. Re-exported, never redefined —
/// two producers of layout semantics already disagreed in v1.
pub use crate::fill::Xform;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RolePlan {
    pub role: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub bytes: Bytes,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Op {
    pub component: String,
    pub out_key: String,
    pub encoding: String,
    pub logical_dtype: Dtype,
    pub logical_shape: Vec<u64>,
    pub roles: Vec<RolePlan>,
    /// What this op does to the tensor it came from — the converts-nothing evidence.
    pub effect: Effect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    /// A new canonical container over untouched bytes: a foreign carrier becomes a snapshot.
    Recontain,
    /// Key changes, bytes do not.
    Rekey,
    /// Bytes are reordered.
    Permute,
    /// Nothing at all changed — same key, same encoding, same objects.
    None,
}

#[derive(Debug, Clone, Default)]
pub struct Plan {
    pub converter: String,
    pub class_name: String,
    pub ops: Vec<Op>,
    pub dropped_markers: Vec<String>,
    pub folded_roles: Vec<String>,
    /// Header bytes the planner read. Tensor bytes read: zero, structurally.
    pub header_bytes: u64,
}

impl Plan {
    pub fn tensor_bytes_read(&self) -> u64 {
        0
    }
    pub fn count(&self, e: Effect) -> usize {
        self.ops.iter().filter(|o| o.effect == e).count()
    }
    pub fn inherited(&self) -> usize {
        self.ops
            .iter()
            .filter(|o| {
                o.roles
                    .iter()
                    .all(|r| matches!(r.bytes, Bytes::Inherit { .. }))
            })
            .count()
    }
    pub fn declared_bytes(&self) -> Result<u64> {
        let mut n = 0u64;
        for o in &self.ops {
            for r in &o.roles {
                n += checked_bytes(&o.out_key, &r.shape, r.dtype)?;
            }
        }
        Ok(n)
    }

    /// A converter that converts NOTHING refuses: a bit-identical copy must never stamp as a
    /// new encoding. `Recontain` is real work (a foreign carrier is not a snapshot); an
    /// all-`None` plan over an already-canonical source is the refused class.
    pub fn check_converts_something(&self) -> Result<()> {
        if self.ops.iter().all(|o| o.effect == Effect::None) {
            return refuse(
                Code::CONVERTS_NOTHING,
                format!(
                    "{}: all {} tensors keep their key, encoding, geometry and objects — a \
                     bit-identical copy must never stamp as a new encoding. The existing \
                     checkpoint already IS this result.",
                    self.converter,
                    self.ops.len()
                ),
            );
        }
        Ok(())
    }
}

// ---------------------------------------------------------------- source views

pub enum Source<'a> {
    /// A foreign carrier: `file` indexes the caller's ordered source list.
    Carrier {
        component: String,
        file: usize,
        header: &'a SourceHeader,
    },
    /// An already-canonical checkpoint component.
    Canonical {
        component: String,
        header: &'a Header,
    },
}

impl Source<'_> {
    pub fn component(&self) -> &str {
        match self {
            Source::Carrier { component, .. } => component,
            Source::Canonical { component, .. } => component,
        }
    }
}

// ---------------------------------------------------------------- planning

/// Plan one conversion. Headers in, plan out — no path is opened and no tensor byte is read.
pub fn plan(
    conv: &Converter,
    sources: &[Source<'_>],
    target: &Target,
    specs: &[(String, EncodingSpec)],
    rekey: &Rekey,
) -> Result<Plan> {
    if conv.class == Class::Value {
        return refuse(
            Code::PLACEMENT_UNFIT,
            format!(
                "{}: value-moving converters do not run at the ingest border. The priced \
                 PRODUCIBLE job surface owns quantization and encoder truncation — one job \
                 surface at both sites, never an ingest-only scheduler.",
                conv.name
            ),
        );
    }
    let mut p = Plan {
        converter: conv.name.to_string(),
        class_name: conv.class.name().to_string(),
        ..Default::default()
    };

    for s in sources {
        let component = s.component().to_string();
        let alias = target.get(&component)?;
        let spec = match specs.iter().find(|(a, _)| a == alias) {
            Some((_, sp)) => sp,
            None => {
                return refuse(
                    Code::UNKNOWN_ENCODING,
                    format!("{component}: target encoding {alias:?} is not a platform alias"),
                )
            }
        };
        match s {
            Source::Carrier { file, header, .. } => {
                let plain = match specs.iter().find(|(a, _)| a == "plain/1") {
                    Some((_, sp)) => sp,
                    None => {
                        return refuse(
                            Code::UNKNOWN_ENCODING,
                            "plain/1 must be in the resolved spec set: a real component is \
                             mixed, and its unencoded tensors are plain storage",
                        )
                    }
                };
                plan_carrier(conv, &component, *file, header, alias, spec, plain, &mut p)?
            }
            Source::Canonical { header, .. } => {
                plan_canonical(&component, header, alias, spec, rekey, &mut p)?
            }
        }
    }
    if p.ops.is_empty() {
        return refuse(Code::MISSING_TENSOR, "the plan produces no tensor");
    }
    p.ops
        .sort_by(|a, b| (&a.component, &a.out_key).cmp(&(&b.component, &b.out_key)));
    Ok(p)
}

/// The dtype this artifact's UNENCODED float weights are in — the evidence a quantized
/// tensor's logical type is read from. Ties refuse by returning None rather than picking.
fn base_float(h: &SourceHeader) -> Option<Dtype> {
    let mut counts: Vec<(Dtype, usize)> = Vec::new();
    for t in &h.tensors {
        if matches!(t.dtype, Dtype::Bf16 | Dtype::F16 | Dtype::F32 | Dtype::F64)
            && module_of(&t.key)
                .and_then(|m| h.get(&format!("{m}.weight_scale")))
                .is_none()
        {
            match counts.iter_mut().find(|(d, _)| *d == t.dtype) {
                Some((_, n)) => *n += 1,
                None => counts.push((t.dtype, 1)),
            }
        }
    }
    counts.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    match counts.as_slice() {
        [] => None,
        [(d, _)] => Some(*d),
        [(d, a), (_, b), ..] if a > b => Some(*d),
        _ => None,
    }
}

/// How a CARRIER's vocabulary lands on a SPEC's. The two do not agree and never will: a
/// carrier says `.weight_scale`, `fp8-rowwise/1` calls that role `scale`, and
/// `nvfp4-w4a4/1` happens to call it `weight_scale`. That correspondence is a reviewed
/// dialect fact, so it is DATA here — inferring it from a name similarity is exactly the
/// class of guess this border does not make.
pub struct RoleMap {
    pub alias: &'static str,
    /// Which spec role carries the tensor's own bytes.
    pub data: &'static str,
    /// carrier suffix -> spec role name.
    pub companions: &'static [(&'static str, &'static str)],
}

pub const ROLE_MAPS: [RoleMap; 4] = [
    RoleMap {
        alias: "plain/1",
        data: "value",
        companions: &[],
    },
    RoleMap {
        alias: "fp8-rowwise/1",
        data: "data",
        companions: &[(".weight_scale", "scale")],
    },
    RoleMap {
        alias: "mxfp8/1",
        data: "data",
        companions: &[(".weight_scale", "scale")],
    },
    RoleMap {
        alias: "nvfp4-w4a4/1",
        data: "weight",
        companions: &[
            (".weight_scale", "weight_scale"),
            (".weight_scale_2", "weight_scale_2"),
            (".input_scale", "input_scale"),
            (".pre_quant_scale", "pre_quant_scale"),
        ],
    },
];

pub fn role_map(alias: &str) -> Result<&'static RoleMap> {
    match ROLE_MAPS.iter().find(|m| m.alias == alias) {
        Some(m) => Ok(m),
        None => refuse(
            Code::UNREGISTERED_ENCODING,
            format!(
                "{alias}: no reviewed carrier-role mapping. The border will not guess which \
                 sibling key is which role of an encoding — that correspondence is reviewed \
                 data, and its absence is a refusal, not a default."
            ),
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn plan_carrier(
    conv: &Converter,
    component: &str,
    file: usize,
    h: &SourceHeader,
    alias: &str,
    spec: &EncodingSpec,
    plain: &EncodingSpec,
    p: &mut Plan,
) -> Result<()> {
    p.header_bytes += h.header_bytes;
    let map = role_map(alias)?;
    let plain_map = role_map("plain/1")?;
    let companions: Vec<&str> = spec
        .roles
        .iter()
        .map(|(n, _)| n.as_str())
        .filter(|n| *n != map.data)
        .collect();
    let mut encoded_tensors = 0usize;
    // Does ANYTHING in this component carry the target's roles? The answer separates two
    // refusals that look identical per-tensor and mean opposite things: a component where
    // nothing carries scales is scale-free STORAGE (and converting it is arithmetic), while
    // one where 361 tensors carry scales and the 362nd does not is a TORN source.
    let component_is_encoded = h.tensors.iter().any(|t| {
        module_of(&t.key)
            .map(|m| {
                map.companions
                    .iter()
                    .any(|(s, _)| h.get(&format!("{m}{s}")).is_some())
            })
            .unwrap_or(false)
    });

    for t in &h.tensors {
        if is_marker(&t.key) {
            p.dropped_markers.push(t.key.clone());
            continue;
        }
        if let Some((module, role)) = role_of(&t.key) {
            // A companion role whose module has no weight is a torn source, not a tensor.
            if h.get(&format!("{module}.weight")).is_none() {
                return refuse(
                    Code::MISSING_COMPANION_ROLE,
                    format!(
                        "{}: a {role} role whose base tensor {module:?}.weight is absent",
                        t.key
                    ),
                );
            }
            p.folded_roles.push(t.key.clone());
            continue;
        }

        // What roles does this tensor actually HAVE in the carrier? This is the observation,
        // and it is what decides whether the tensor is encoded — never the target, never a
        // name list, and never the dtype spelling alone (the v1 quarry's lane table was
        // keyed on the dtype and that WAS a real defect: scale-free fp8 storage and scaled
        // fp8 rowwise declare the same element type and execute in different lanes).
        let present: Vec<(String, &super::carrier::SourceTensor)> = match module_of(&t.key) {
            None => Vec::new(),
            Some(m) => map
                .companions
                .iter()
                .filter_map(|(sfx, role)| {
                    h.get(&format!("{m}{sfx}")).map(|x| (role.to_string(), x))
                })
                .collect(),
        };
        // A sibling the target has no role for is an encoding fact on the wrong side of the
        // boundary — counted, never absorbed.
        if let Some(m) = module_of(&t.key) {
            for sfx in super::fingerprint::ROLE_SUFFIXES.iter() {
                if h.get(&format!("{m}{sfx}")).is_some()
                    && !map.companions.iter().any(|(s, _)| s == sfx)
                {
                    return refuse(
                        Code::DOUBLE_ENCODE,
                        format!(
                            "{}: the carrier supplies a {sfx} sibling that target {alias} has \
                             no role for",
                            t.key
                        ),
                    );
                }
            }
        }

        // A real component is MIXED: a quantized weight sits beside its own unquantized
        // bias and the norms the producer's eligibility predicate skipped. The component's
        // target names what its ENCODED tensors carry; everything else is plain storage.
        // One rule over every tensor in a tree is the shape this design explicitly rejects.
        let (use_spec, use_alias, use_map) = if present.is_empty() {
            (plain, "plain/1", plain_map)
        } else {
            encoded_tensors += 1;
            (spec, alias, map)
        };

        if use_alias == "plain/1" && !present.is_empty() {
            unreachable!("present is empty on this arm");
        }
        if use_alias != "plain/1" {
            // The observed role set must be EXACTLY the target's. Roles are exact by
            // construction — a spec with a different role set is a different digest.
            let mut got: Vec<&str> = present.iter().map(|(n, _)| n.as_str()).collect();
            let mut want = companions.clone();
            got.sort_unstable();
            want.sort_unstable();
            if got != want {
                let missing: Vec<&&str> = want.iter().filter(|c| !got.contains(c)).collect();
                return refuse(
                    if missing.is_empty() {
                        Code::DOUBLE_ENCODE
                    } else {
                        Code::MISSING_COMPANION_ROLE
                    },
                    format!(
                        "{}: target {alias} declares roles {want:?}; the carrier supplies \
                         {got:?}. A partial role set is a TORN source, not a convertible one \
                         — the missing bytes are arithmetic, and the remedy derives from the \
                         target's contract, never from what happened to be observed.",
                        t.key
                    ),
                );
            }
            // Companion geometry must satisfy the spec's own relation, checked here rather
            // than after 5 GB have moved.
            let elements = crate::dtype::checked_elements(&t.key, &t.shape)?;
            for (name, st) in &present {
                let role = match use_spec.role(name) {
                    Some(r) => r,
                    None => unreachable!("role set was just compared"),
                };
                let want_shape = role.shape.eval(&t.shape, elements)?;
                if want_shape != st.shape {
                    return refuse(
                        Code::SHAPE_MISMATCH,
                        format!(
                            "{}#{name}: carrier shape {:?} != the target spec's relation \
                             result {want_shape:?}. Two scale geometries under one alias are \
                             two DIGESTS: a reader that assumes the other mis-reads every \
                             row and yields plausible, wrong numbers.",
                            t.key, st.shape
                        ),
                    );
                }
            }
        } else if matches!(t.dtype, Dtype::F8E4M3FN | Dtype::F8E5M2)
            && !companions.is_empty()
            && component_is_encoded
        {
            return refuse(
                Code::MISSING_COMPANION_ROLE,
                format!(
                    "{}: an encoded {} weight with no {companions:?} sibling, in a component \
                     where other tensors have theirs. That is a TORN source, not scale-free \
                     storage — the bytes exist and their scale does not.",
                    t.key,
                    t.dtype.name()
                ),
            );
        } else if matches!(t.dtype, Dtype::F8E4M3FN | Dtype::F8E5M2) && !companions.is_empty() {
            // Scale-FREE fp8 offered under a scaled target. The absence of scales IS an
            // identity: the two have the same element type and the same tensor names, and a
            // reader that takes one for the other either divides by scales that are not
            // there or fails to divide by scales that are. Both are silent.
            return refuse(
                Code::REMEDY_FROM_TARGET,
                format!(
                    "{}: scale-free {} storage offered under target {alias}, which requires \
                     {companions:?}. The absence of scales is the identity, not an omission \
                     — producing them is the priced PRODUCIBLE quantize job.",
                    t.key,
                    t.dtype.name()
                ),
            );
        }

        let data = use_map.data.to_string();
        let data_pos = match use_spec.roles.iter().position(|(n, _)| *n == data) {
            Some(i) => i,
            None => {
                return refuse(
                    Code::ROLE_SET_MISMATCH,
                    format!(
                    "{use_alias}: the reviewed map names data role {data:?}, absent from the spec"
                ),
                )
            }
        };
        let logical_dtype = match &use_spec.roles[data_pos].1.carrier {
            crate::relation::Carrier::SameAsLogical => t.dtype,
            crate::relation::Carrier::Set(_) => match base_float(h) {
                Some(d) => d,
                None => {
                    return refuse(
                        Code::DTYPE_UNKNOWN,
                        format!(
                            "{}: target {use_alias} stores {} carrier bytes, so the logical \
                             dtype is not in the file — and this carrier holds no unencoded \
                             float weight to read it from. The logical type is homeless; an \
                             explicit profile must state it.",
                            t.key,
                            t.dtype.name()
                        ),
                    )
                }
            },
        };
        if !use_spec.admits_logical(logical_dtype) {
            return refuse(
                Code::DTYPE_MISMATCH,
                format!(
                    "{}: logical dtype {} is outside target {use_alias}'s logical set",
                    t.key,
                    logical_dtype.name()
                ),
            );
        }

        let mut roles = vec![RolePlan {
            role: data.clone(),
            dtype: t.dtype,
            shape: t.shape.clone(),
            bytes: Bytes::Stream {
                file,
                key: t.key.clone(),
            },
        }];
        for (name, st) in &present {
            roles.push(RolePlan {
                role: name.clone(),
                dtype: st.dtype,
                shape: st.shape.clone(),
                bytes: Bytes::Stream {
                    file,
                    key: st.key.clone(),
                },
            });
        }
        // The spec's declared role ORDER is the header's role order.
        roles.sort_by_key(|r| {
            use_spec
                .roles
                .iter()
                .position(|(n, _)| *n == r.role)
                .unwrap_or(usize::MAX)
        });

        let out_key = rekey(conv, &t.key);
        let effect = if out_key == t.key {
            Effect::Recontain
        } else {
            Effect::Rekey
        };
        p.ops.push(Op {
            component: component.to_string(),
            out_key,
            encoding: use_alias.to_string(),
            logical_dtype,
            logical_shape: t.shape.clone(),
            roles,
            effect,
        });
    }

    // The component's target named an encoding and NOTHING in the component carries it.
    // Producing it is arithmetic; the verdict's remedy derives from the TARGET's contract.
    if alias != "plain/1" && encoded_tensors == 0 {
        return refuse(
            Code::REMEDY_FROM_TARGET,
            format!(
                "{component}: target {alias} requires roles {companions:?} and not one of the \
                 {} tensors in this component carries them. Nothing here is convertible INTO \
                 {alias} — that is a quantization, and it runs as the priced PRODUCIBLE job \
                 over the plain source. A border cast would give every name, dtype and shape \
                 correct and every number wrong, with nothing raised.",
                h.tensors.len()
            ),
        );
    }
    Ok(())
}

/// A reviewed tier-1 REKEY: a prefix substitution that moves keys and no bytes.
///
/// This is the v1 quarry's `anima.net-to-prefixed@1` shape — "a new manifest over the same
/// chunks, zero new bytes" — and it is the operation that makes inherit-by-reference
/// observable: every tensor's key changes, every tensor's OBJECTS are the ones already
/// admitted, and the run hashes nothing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Rekey {
    pub rules: Vec<(String, String, String)>,
}

impl Rekey {
    /// `component=from:to`, repeated.
    pub fn parse(s: &str) -> Result<(String, String, String)> {
        let (component, rest) = match s.split_once('=') {
            Some(x) => x,
            None => {
                return refuse(
                    Code::MISSING_FIELD,
                    format!("rekey {s:?}: expected component=from:to"),
                )
            }
        };
        match rest.split_once(':') {
            Some((from, to)) => Ok((component.to_string(), from.to_string(), to.to_string())),
            None => refuse(
                Code::MISSING_FIELD,
                format!("rekey {s:?}: expected component=from:to"),
            ),
        }
    }
    fn apply(&self, component: &str, key: &str) -> String {
        for (c, from, to) in &self.rules {
            if c == component {
                if let Some(rest) = key.strip_prefix(from.as_str()) {
                    return format!("{to}{rest}");
                }
            }
        }
        key.to_string()
    }
}

/// Re-ingest of an already-canonical component: every part carries by OBJECT REFERENCE.
fn plan_canonical(
    component: &str,
    h: &Header,
    alias: &str,
    spec: &EncodingSpec,
    rekey: &Rekey,
    p: &mut Plan,
) -> Result<()> {
    let want = spec.object_id();
    for (comp, key, t) in h.tensors() {
        if comp != component {
            continue;
        }
        if t.encoding != want {
            return refuse(
                Code::DOUBLE_ENCODE,
                format!(
                    "{comp}/{key}: already encoded under {} and the target asks for {alias} \
                     ({want}) — re-encoding already-encoded bytes needs the priced job, not \
                     the border",
                    t.encoding
                ),
            );
        }
        let out_key = rekey.apply(component, key);
        let effect = if out_key == *key {
            Effect::None
        } else {
            Effect::Rekey
        };
        p.ops.push(Op {
            component: component.to_string(),
            out_key,
            encoding: alias.to_string(),
            logical_dtype: t.dtype,
            logical_shape: t.shape.clone(),
            roles: t
                .parts
                .iter()
                .map(|(role, part)| RolePlan {
                    role: role.clone(),
                    dtype: part.dtype,
                    shape: part.shape.clone(),
                    bytes: Bytes::Inherit {
                        part: Box::new(part.clone()),
                    },
                })
                .collect(),
            effect,
        });
    }
    Ok(())
}

/// The dialect's key mapping. Identity converters keep the spelling (diffusers spelling IS
/// the canonical spelling); a renaming dialect declares its map here and nowhere else.
fn rekey(conv: &Converter, key: &str) -> String {
    match conv.name {
        "single_file.identity/1" => SINGLE_FILE_PREFIXES
            .iter()
            .find(|(from, _)| key.starts_with(*from))
            .map(|(from, to)| format!("{to}{}", &key[from.len()..]))
            .unwrap_or_else(|| key.to_string()),
        _ => key.to_string(),
    }
}

/// The reviewed single-file → canonical component prefix map. Data, not inference: a prefix
/// that is not here does not get guessed at, it stays as it is and the fingerprint decides.
pub const SINGLE_FILE_PREFIXES: [(&str, &str); 4] = [
    ("model.diffusion_model.", ""),
    ("first_stage_model.", ""),
    ("conditioner.embedders.0.transformer.", ""),
    ("conditioner.embedders.1.model.", ""),
];

/// The bounded-buffer guard for a permutation op.
pub fn check_permute_size(what: &str, nbytes: u64) -> Result<()> {
    if nbytes > limits::CONVERT_TENSOR_MAX_BYTES {
        return refuse(
            Code::CONVERT_SIZE_CAP,
            format!(
                "{what}: a permutation needs {nbytes} B resident, over the declared \
                 {} B converter bound",
                limits::CONVERT_TENSOR_MAX_BYTES
            ),
        );
    }
    Ok(())
}

/// Object references a plan inherits, for the CAS assertion that nothing was re-hashed.
pub fn inherited_objects(p: &Plan) -> Vec<ObjectRef> {
    let mut v: Vec<ObjectRef> = Vec::new();
    for o in &p.ops {
        for r in &o.roles {
            if let Bytes::Inherit { part } = &r.bytes {
                v.extend(part.segments().iter().cloned());
            }
        }
    }
    v.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    v.dedup_by(|a, b| a.sha256 == b.sha256);
    v
}

// ---------------------------------------------------------------- banked geometry

/// The H3 attention geometry, transcribed from the v1 quarry
/// (`jobs/conversion/src/conversion/quant/h3_native_layout.py`) and from the ratified seam
/// registry (`tensorfs/spec/v2/morphisms/minimax-h3.split-to-fused-qkv.v1.json`).
///
/// Measured there, not assumed here: the DiT's `to_q/to_k/to_v` are each `[7168, 5376]` and
/// `7168 = 56 x 128`, so the fused `qkv_proj` axis is `21504 = 56 groups x (q+k+v) x 128`.
/// The head count is 56 and it is the load-bearing fact.
///
/// **Never a constant at runtime.** The quarry derives the head dim from the checkpoint's
/// own `q_norm.weight` width and the head count by division, precisely so a future H3
/// geometry cannot be silently mis-fused; `h3_geometry` below does the same in reverse.
pub const H3_HEADS: u64 = 56;
pub const H3_HEAD_DIM: u64 = 128;

/// Read the fused-attention geometry off THIS carrier: `head_dim` is the width of the
/// sibling `q_norm`, and the group count is what is left after dividing out three members.
/// Returns `(groups, unit_bytes)` for a fused key, or a typed refusal naming what is missing.
pub fn h3_geometry(
    h: &SourceHeader,
    fused_key: &str,
    norm_key: &str,
    rows: u64,
    row_bytes: u64,
) -> Result<(u64, u64)> {
    let head_dim = match h.get(norm_key) {
        Some(t) => match t.shape.first() {
            Some(d) => *d,
            None => {
                return refuse(
                    Code::SHAPE_MISMATCH,
                    format!("{norm_key}: rank-0 norm carries no head dim"),
                )
            }
        },
        None => {
            return refuse(
                Code::MISSING_COMPANION_ROLE,
                format!(
                    "{fused_key}: no {norm_key} to read the head dim from. The head count is \
                     never a constant here — a hardcoded {H3_HEADS} would silently mis-fuse \
                     any future geometry, which is a ~90% numerical error that does not crash."
                ),
            )
        }
    };
    if head_dim == 0 || !rows.is_multiple_of(3 * head_dim) {
        return refuse(
            Code::DIVISION_REMAINDER,
            format!("{fused_key}: {rows} rows do not divide into 3 x {head_dim}"),
        );
    }
    Ok((rows / (3 * head_dim), head_dim * row_bytes))
}

/// The frozen golden cases for one converter: geometries whose reassembly is checked live,
/// on real byte patterns, before the converter is allowed to touch an artifact.
///
/// The RED companion is structural rather than a separate fixture: `Xform::reassembles`
/// slices the source on the transform's own indexing and demands the three members put the
/// source back bit-for-bit. A flat three-way split of a 56-group source passes every
/// name/dtype/shape check and FAILS this, which is the whole point — the quarry pinned that
/// error at ~90% numerically while nothing raised.
pub fn golden_cases(c: &Converter) -> Vec<(&'static str, Xform)> {
    match c.name {
        "h3.native/1" => vec![
            (
                "h3-dit-qkv 56 groups x 128 x bf16",
                Xform::QkvSplit {
                    groups: H3_HEADS,
                    shares: [1, 1, 1],
                    take: 0,
                    unit_bytes: H3_HEAD_DIM * 2,
                },
            ),
            (
                "video-vae 1x1-conv qkv, contiguous thirds",
                Xform::QkvSplit {
                    groups: 1,
                    shares: [1, 1, 1],
                    take: 1,
                    unit_bytes: 384 * 2,
                },
            ),
            (
                "unequal shares 1:1:2 (q 2048, k 2048, v 4096)",
                Xform::QkvSplit {
                    groups: 1,
                    shares: [1, 1, 2],
                    take: 2,
                    unit_bytes: 2048 * 2,
                },
            ),
        ],
        _ => vec![],
    }
}

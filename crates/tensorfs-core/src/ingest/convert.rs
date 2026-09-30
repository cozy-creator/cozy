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
use crate::ids::ObjectRef;
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
    /// `(state, dialect of the stored keys)` for the output's `normalization` config:
    /// `raw` keys need a consumer that reads that dialect, `normalized` ones are canonical.
    /// `None` records nothing (reviewed outputs that predate the record).
    pub normalization: Option<(&'static str, &'static str)>,
    /// The one declared value change a converter may make: floating sources narrow to this
    /// dtype, IEEE round-to-nearest-even, and a finite value it cannot hold refuses
    /// NUMBER_RANGE. `None` moves bytes only.
    pub lane: Option<Dtype>,
    /// The pinned repository (`hf://org/repo@<commit>`, a Diffusers pipeline) whose index,
    /// configs, scheduler and tokenizers the output needs when its source carries none.
    pub reference: Option<&'static str>,
}

/// The header config an output's [`Converter::normalization`] is recorded under.
pub const NORMALIZATION: &str = "normalization";
/// The converter for a source no reviewed profile recognizes: its keys, stored as they are.
pub const IDENTITY: &str = "identity/1";

pub const CONVERTERS: [Converter; 8] = [
    Converter {
        name: "diffusers.identity/1",
        dialect: "safetensors.diffusers",
        class: Class::Identity,
        doc: "Diffusers component spelling IS the canonical logical key spelling. Each key \
              becomes one logical tensor; `.weight_scale` siblings fold into their base as \
              encoding ROLES; marker tensors are dropped. Bytes stream through untouched.",
        validated_on_real_bytes: true,
        normalization: None,
        lane: None,
        reference: None,
    },
    Converter {
        name: "single_file.identity/1",
        dialect: "safetensors.single_file",
        class: Class::Identity,
        doc: "One file carrying several components, split on the reviewed component prefix \
              map. Same byte path as the diffusers converter — packaging is not identity.",
        validated_on_real_bytes: true,
        normalization: None,
        lane: None,
        reference: None,
    },
    Converter {
        name: "cozytensors.inherit/1",
        dialect: "cozytensors",
        class: Class::Identity,
        doc: "An already-canonical source. Every untouched tensor carries by OBJECT \
              REFERENCE; nothing is re-hashed and nothing is re-uploaded.",
        validated_on_real_bytes: true,
        normalization: None,
        lane: None,
        reference: None,
    },
    Converter {
        name: "h3.lora/2",
        dialect: "safetensors.h3_lora",
        class: Class::Permutation,
        doc: "Trainer H3 LoRA A/B and Kohya down/up factors: split contiguous Q/K/V B thirds, swap SwiGLU B halves, preserve A and alpha without arithmetic. Original base checkpoint head interleaving belongs to h3.native/2, not this adapter state-dict dialect.",
        validated_on_real_bytes: false,
        normalization: None,
        lane: None,
        reference: None,
    },
    Converter {
        name: "h3.native/2",
        dialect: "safetensors.h3_native",
        class: Class::Permutation,
        doc: "H3 native spelling: every head-interleaved fused `attn.qkv_proj` split into the \
              canonical per-head contiguous to_q|to_k|to_v, and the ratified rekey rows moved \
              with it. Fused MLP [gate; value] rows become [value; gate] for Diffusers \
              SwiGLU. Reordering only — never arithmetic. A fused attention key this \
              converter has NO banked seam for refuses rather than carrying through.",
        // FLIPPED 2026-08-25 (tfs-003 convergence run, MiniMaxAI/MiniMax-H3@42ed227e): the
        // full 66.28 GB native tree ingested through this converter (639 ops, exit 0), and
        // the produced to_q/to_k/to_v are BIT-IDENTICAL to the diffusers packaging's own
        // bytes on real 77 MB fused tensors from BOTH attention families
        // (transformer_blocks.0 and token_refiner.blocks.0; groups=56, head_dim=128 read
        // from q_norm). The 17 edge rewrites are byte-digest-proven bijections; the sole
        // native-only key is rope.inv_freq, carried.
        // 2026-09-07: h3.native/2 also swaps all main/refiner fc1 row halves, as the
        // pinned official converter requires. The complete trained Ref2VA block0
        // fc1 (308281344 B) converted to e7cf97ae...3b19; fc2 stayed byte-identical.
        validated_on_real_bytes: true,
        normalization: None,
        lane: None,
        reference: None,
    },
    Converter {
        name: IDENTITY,
        dialect: "safetensors",
        class: Class::Identity,
        doc: "An unrecognized source, stored as it is: every key becomes one plain tensor \
              under its own name and dtype. Nothing is folded, dropped or renamed, so a \
              consumer that needs another spelling normalizes it.",
        validated_on_real_bytes: true,
        normalization: Some(("raw", "safetensors")),
        lane: None,
        reference: None,
    },
    Converter {
        name: super::routes::SDXL,
        dialect: "safetensors.single_file",
        class: Class::Permutation,
        doc: "An SDXL single-file checkpoint becomes the Diffusers constructor's checkpoint: \
              the generated route table rekeys every tensor into constructor order, splits \
              OpenCLIP's fused in_proj into q|k|v runs, transposes its text_projection, \
              squeezes 1x1-conv attention to linear and drops CLIP position IDs and \
              logit_scale. bf16/f32 sources narrow to f16. Pipeline index, configs and \
              tokenizers come from the pinned SDXL base reference.",
        // The generated table equals packages-v2 sdxl normalization.json route for route, the
        // mapping the sdxl normalize job served real Civitai checkpoints with.
        validated_on_real_bytes: true,
        normalization: Some(("normalized", "diffusers")),
        lane: Some(Dtype::F16),
        reference: Some(
            "hf://stabilityai/stable-diffusion-xl-base-1.0@462165984030d82259a11f4367a4eed129e94a7b",
        ),
    },
    Converter {
        name: super::routes::ANIMA,
        dialect: "safetensors.single_file",
        class: Class::Identity,
        doc: "Anima's ComfyUI files (the `net.` DiT carrying its `net.llm_adapter.` text \
              conditioner, the Qwen3 text encoder under `model.`, the Qwen-Image VAE) become \
              the Diffusers constructors' checkpoints: the generated route table rekeys every \
              tensor into constructor order and moves no byte. Sources stay bf16. Pipeline \
              index, configs and tokenizers come from the pinned Anima Base v1.0 Diffusers \
              reference.",
        // Every route's dtype, shape and leading 16 KiB equal the reference repository's
        // tensor on Civitai 2945208's real bytes (1,189 routes, 409 whole), and the table
        // equals diffusers v0.40.0 scripts/convert_anima_to_diffusers.py route for route.
        validated_on_real_bytes: true,
        normalization: Some(("normalized", "diffusers")),
        lane: Some(Dtype::Bf16),
        reference: Some(
            "hf://circlestone-labs/Anima-Base-v1.0-Diffusers@073c3a9db359c31ad0e8aa268d15775473c2176c",
        ),
    },
];

/// The `normalization` config the output of `converter` carries, if it records one.
pub fn normalization_config(converter: &str) -> Option<(String, Vec<u8>)> {
    let converter_row = CONVERTERS.iter().find(|c| c.name == converter)?;
    let (state, dialect) = converter_row.normalization?;
    let mut fields = vec![
        ("converter", crate::canon::Value::str(converter)),
        ("dialect", crate::canon::Value::str(dialect)),
        ("state", crate::canon::Value::str(state)),
    ];
    if let Some(reference) = converter_row.reference {
        fields.push(("reference", crate::canon::Value::str(reference)));
    }
    let record = crate::canon::Value::obj(fields);
    Some((NORMALIZATION.to_string(), crate::canon::write(&record)))
}

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
///
/// A component may be named MORE THAN ONCE, and that is the vocabulary a mixed-allocation
/// carrier needs: `transformer=mxfp8/1,transformer=fp8-rowwise/1` says this one component's
/// tensors are drawn from those two encodings, and the carrier's own per-tensor roles pick
/// which is which. The widening stops at the component on purpose — a set that spanned the
/// SNAPSHOT would let the transformer's encoding admit a text-encoder tensor, which is the
/// one-rule-over-the-tree this type exists to refuse.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Target {
    pub components: Vec<(String, String)>,
}

impl Target {
    /// `unet=plain/1,text_encoder=plain/1`, a component repeatable for a mixed one.
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
    /// Every alias this component is allowed to draw from, in declaration order. A set of
    /// one is the ordinary case and behaves exactly as the single alias always did.
    pub fn allowed(&self, component: &str) -> Result<Vec<&str>> {
        let mut v: Vec<&str> = Vec::new();
        for (c, e) in &self.components {
            if c == component && !v.contains(&e.as_str()) {
                v.push(e.as_str());
            }
        }
        if v.is_empty() {
            return refuse(
                Code::MISSING_FIELD,
                format!(
                    "the target names no encoding for component {component:?} — targets are \
                     component-granular and a missing component is never defaulted"
                ),
            );
        }
        Ok(v)
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
    /// Reorder source runs through the ordinary bounded streaming writer; never arithmetic.
    /// Input role size is bounded by `CONVERT_TENSOR_MAX_BYTES`.
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
    /// The alias, for display only. An alias may group several physically distinct specs
    /// (the two `fp8-scaled-scalar/1` role sets one artifact carries), so it is NOT an
    /// identity and nothing resolves a spec through it.
    pub encoding: String,
    /// The selected spec's object digest — the encoding's real identity, chosen PER TENSOR.
    pub encoding_id: String,
    pub logical_dtype: Dtype,
    pub logical_shape: Vec<u64>,
    pub roles: Vec<RolePlan>,
    /// What this op does to the tensor it came from — the converts-nothing evidence.
    pub effect: Effect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    /// A new canonical container over untouched bytes: a foreign carrier becomes a manifest.
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
    construction_order_bound: bool,
}

impl Plan {
    /// Apply the artifact recipe's exact construction traversal. The converter produces a
    /// set of typed operations; it has no authority to invent their execution order.
    pub fn apply_order(&mut self, order: &[(String, String)]) -> Result<()> {
        if order.is_empty() {
            return refuse(
                Code::CONSTRUCTION_ORDER_REQUIRED,
                "artifact production requires a non-empty construction traversal",
            );
        }
        let mut by_key = std::collections::HashMap::with_capacity(self.ops.len());
        for op in std::mem::take(&mut self.ops) {
            let key = (op.component.clone(), op.out_key.clone());
            if by_key.insert(key.clone(), op).is_some() {
                return refuse(
                    Code::DUPLICATE_KEY,
                    format!("converter produced {}/{} twice", key.0, key.1),
                );
            }
        }
        let mut ordered = Vec::with_capacity(order.len());
        for (component, key) in order {
            match by_key.remove(&(component.clone(), key.clone())) {
                Some(op) => ordered.push(op),
                None => {
                    return refuse(
                        Code::TRAVERSAL_INCOMPLETE,
                        format!("construction traversal repeats or invents {component}/{key}"),
                    )
                }
            }
        }
        if !by_key.is_empty() {
            let mut missing: Vec<String> = by_key
                .keys()
                .map(|(component, key)| format!("{component}/{key}"))
                .collect();
            missing.sort();
            return refuse(
                Code::TRAVERSAL_INCOMPLETE,
                format!(
                    "construction traversal omits {} tensor(s), first {:?}",
                    missing.len(),
                    &missing[..missing.len().min(3)]
                ),
            );
        }
        self.ops = ordered;
        self.construction_order_bound = true;
        Ok(())
    }

    pub(crate) fn require_order(&self) -> Result<()> {
        if self.construction_order_bound {
            Ok(())
        } else {
            refuse(
                Code::CONSTRUCTION_ORDER_REQUIRED,
                "artifact execution received a plan with no bound construction traversal",
            )
        }
    }
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
    /// The encodings this plan actually selected, in first-appearance order:
    /// `(alias, spec digest, role names, tensor count)`. With per-tensor selection one
    /// component routinely cites several — the real fp8_scaled DiT cites three.
    pub fn encodings(&self) -> Vec<(String, String, Vec<String>, usize)> {
        let mut out: Vec<(String, String, Vec<String>, usize)> = Vec::new();
        for o in &self.ops {
            match out.iter_mut().find(|(_, id, _, _)| *id == o.encoding_id) {
                Some((_, _, _, n)) => *n += 1,
                None => out.push((
                    o.encoding.clone(),
                    o.encoding_id.clone(),
                    o.roles.iter().map(|r| r.role.clone()).collect(),
                    1,
                )),
            }
        }
        out
    }

    /// The decomposition this plan's permutes actually buy — the number #322 says is the
    /// whole question: `(ops, source bytes, runs, mean run, class)`. Computed from the
    /// declared geometry in O(ops), and it is what tells a reader whether the seam rides a
    /// copy at ~zero marginal cost or is real work with its own budget.
    pub fn permutes(&self) -> Option<(usize, u64, usize, u64, &'static str)> {
        let (mut ops, mut bytes, mut runs, mut gather) = (0usize, 0u64, 0usize, false);
        for o in &self.ops {
            for r in &o.roles {
                if let Bytes::Permute { xform, .. } = &r.bytes {
                    let (moved, count, class) = xform.census();
                    ops += 1;
                    bytes += moved;
                    runs += count;
                    gather |= class == crate::fill::Class::Gather;
                }
            }
        }
        match runs {
            0 => None,
            n => Some((
                ops,
                bytes,
                n,
                bytes / n as u64,
                if gather { "gather" } else { "long-run" },
            )),
        }
    }

    pub fn declared_bytes(&self) -> Result<u64> {
        let mut n = 0u64;
        for o in &self.ops {
            for r in &o.roles {
                n = n
                    .checked_add(checked_bytes(&o.out_key, &r.shape, r.dtype)?)
                    .ok_or_else(|| crate::err::Refusal {
                        code: Code::ARITH_OVERFLOW,
                        detail: "conversion output byte count overflow".into(),
                    })?;
            }
        }
        Ok(n)
    }

    /// A converter that converts NOTHING refuses: a bit-identical copy must never stamp as a
    /// new encoding. `Recontain` is real work (a foreign carrier is not a manifest); an
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

/// The candidate encodings one component may resolve to. An alias that groups several
/// PHYSICALLY distinct specs contributes one candidate per variant, and `plain/1` is always
/// in the closure because a real component is mixed. Nothing here picks: the carrier's own
/// sibling roles pick, per tensor.
struct Candidates<'a> {
    /// `(alias, spec)`, alias repeated once per physical variant it groups.
    encoded: Vec<(&'a str, &'a EncodingSpec)>,
    plain: &'a EncodingSpec,
}

impl Candidates<'_> {
    /// Every carrier suffix ANY candidate names — the component-level "is this encoded at
    /// all" question, which separates scale-free storage from a torn source.
    fn suffixes(&self) -> Result<Vec<&'static str>> {
        let mut v: Vec<&'static str> = Vec::new();
        for (alias, _) in &self.encoded {
            for (sfx, _) in role_map(alias)?.companions {
                if !v.contains(sfx) {
                    v.push(sfx);
                }
            }
        }
        Ok(v)
    }
    /// Every companion ROLE any candidate declares, for the refusal messages.
    fn companion_roles(&self) -> Result<Vec<String>> {
        let mut v: Vec<String> = Vec::new();
        for (alias, spec) in &self.encoded {
            let data = role_map(alias)?.data;
            for (n, _) in &spec.roles {
                if n != data && !v.contains(n) {
                    v.push(n.clone());
                }
            }
        }
        v.sort();
        Ok(v)
    }
}

fn short(id: &str) -> String {
    id.chars().skip(7).take(12).collect()
}

/// One component's ALLOWED ALIAS SET, for the verdicts. A set of one prints bare, so the
/// ordinary single-encoding refusal reads exactly as it always did.
struct Allowed<'a>(&'a [&'a str]);

impl Allowed<'_> {
    /// Nothing but plain storage allowed — then a component with no encoded tensor is the
    /// ordinary case rather than an unperformed conversion.
    fn is_plain_only(&self) -> bool {
        self.0.iter().all(|a| *a == "plain/1")
    }
}

impl std::fmt::Display for Allowed<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            [one] => write!(f, "{one}"),
            many => write!(f, "{many:?}"),
        }
    }
}

/// Plan one conversion. Headers in, plan out — no path is opened and no tensor byte is read.
///
/// `specs` is the CANDIDATE SET: `(alias, spec)` pairs, and one alias may appear more than
/// once when it groups several physical variants. Selection happens per TENSOR.
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

    let plain = match specs.iter().find(|(a, _)| a == "plain/1") {
        Some((_, sp)) => sp,
        None => {
            return refuse(
                Code::UNKNOWN_ENCODING,
                "plain/1 must be in the resolved spec set: a real component is mixed, and \
                 its unencoded tensors are plain storage",
            )
        }
    };
    for s in sources {
        let component = s.component().to_string();
        // The component's ALLOWED SET, not its one alias: a component whose producer
        // allocated two encodings across its tensors is one component, and the candidate
        // closure has to be able to say so or that carrier is unreachable by construction.
        let allowed = target.allowed(&component)?;
        let encoded: Vec<(&str, &EncodingSpec)> = specs
            .iter()
            .filter(|(a, _)| a != "plain/1" && allowed.contains(&a.as_str()))
            .map(|(a, sp)| (a.as_str(), sp))
            .collect();
        for a in &allowed {
            if *a != "plain/1" && !encoded.iter().any(|(e, _)| e == a) {
                return refuse(
                    Code::UNKNOWN_ENCODING,
                    format!("{component}: target encoding {a:?} is not a platform alias"),
                );
            }
        }
        let cands = Candidates { encoded, plain };
        match s {
            Source::Carrier { file, header, .. } => {
                plan_carrier(conv, &component, *file, header, &allowed, &cands, &mut p)?
            }
            Source::Canonical { header, .. } => {
                plan_canonical(&component, header, &allowed, &cands, rekey, &mut p)?
            }
        }
    }
    if p.ops.is_empty() {
        return refuse(Code::MISSING_TENSOR, "the plan produces no tensor");
    }
    Ok(p)
}

/// The dtype this artifact's UNENCODED float weights are in — the evidence a quantized
/// tensor's logical type is read from. Ties refuse by returning None rather than picking.
///
/// Role siblings and markers are NOT weights and are excluded. They were not, and on the
/// real fp8_scaled DiT that was load-bearing: its 350 F32 scale siblings outnumber its 220
/// BF16 weights, so the majority vote returned F32 and every quantized tensor would have
/// been stamped with a logical dtype the artifact does not have — every name, shape and
/// dtype spelling plausible, the numbers silently mis-typed.
fn base_float(h: &SourceHeader) -> Option<Dtype> {
    let mut counts: Vec<(Dtype, usize)> = Vec::new();
    for t in &h.tensors {
        if is_marker(&t.key) || role_of(&t.key).is_some() {
            continue;
        }
        if matches!(t.dtype, Dtype::Bf16 | Dtype::F16 | Dtype::F32 | Dtype::F64)
            && !module_of(&t.key)
                .map(|m| {
                    super::fingerprint::ROLE_SUFFIXES
                        .iter()
                        .any(|s| h.get(&format!("{m}{s}")).is_some())
                })
                .unwrap_or(false)
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

pub const ROLE_MAPS: [RoleMap; 5] = [
    RoleMap {
        alias: "plain/1",
        data: "value",
        companions: &[],
    },
    // ComfyUI's `scaled_fp8` convention, MEASURED on `Comfy-Org/MiniMax-H3@4cc1d817`. The
    // alias groups two physical role sets — 150 modules carry both scales, the 50 `mlp.fc2`
    // ones carry the weight scale alone — so this map names the alias's FULL carrier
    // vocabulary and each spec's exact role set decides which tensor is which.
    RoleMap {
        alias: "fp8-scaled-scalar/1",
        data: "data",
        companions: &[
            (".weight_scale", "weight_scale"),
            (".input_scale", "input_scale"),
        ],
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

// ---------------------------------------------------------------- per-tensor selection

/// Why ONE candidate did not fit ONE tensor. Kept rather than collapsed to a bool, because
/// the refusal has to name the roles found AND every candidate tried with its reason — a
/// bare "no encoding matched" would send a reader back to the bytes this planner just read.
#[derive(Debug, Clone)]
enum Reject {
    /// The carrier supplies a sibling this candidate's dialect has no role for at all.
    Surplus(String),
    /// Role names present vs the candidate's EXACT set.
    RoleSet {
        got: Vec<String>,
        want: Vec<String>,
    },
    Shape {
        role: String,
        got: Vec<u64>,
        want: Vec<u64>,
    },
    Carrier {
        role: String,
        got: &'static str,
    },
    Rank {
        want: u64,
        got: usize,
    },
    Logical(&'static str),
}

impl std::fmt::Display for Reject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Reject::Surplus(s) => write!(f, "no role for a {s} sibling"),
            Reject::RoleSet { got, want } => {
                write!(f, "declares roles {want:?}, carrier supplies {got:?}")
            }
            Reject::Shape { role, got, want } => {
                write!(
                    f,
                    "role {role}: carrier shape {got:?} != the spec's relation {want:?}"
                )
            }
            Reject::Carrier { role, got } => {
                write!(f, "role {role}: carrier dtype {got} outside the spec's set")
            }
            Reject::Rank { want, got } => write!(f, "logical rank {got} != the spec's {want}"),
            Reject::Logical(d) => write!(f, "logical dtype {d} outside the spec's logical set"),
        }
    }
}

/// The refusal CODE for a tensor no candidate fits. Ordered so the most structural
/// disagreement wins: a missing companion (a torn source) outranks a surplus one (an
/// encoding fact the target has no room for), which outranks geometry, which outranks
/// dtype. Every one of these was a separate refusal class before selection became
/// per-tensor and every one is still reachable.
fn code_for(rejects: &[(String, Reject)]) -> Code {
    // The target declares no encoded variant at all and the carrier supplies roles anyway:
    // encoded bytes offered as plain storage, which is the double-encode class exactly.
    if rejects.is_empty() {
        return Code::DOUBLE_ENCODE;
    }
    let (mut extra, mut missing) = (false, false);
    for (_, r) in rejects {
        match r {
            Reject::Surplus(_) => extra = true,
            Reject::RoleSet { got, want } => {
                if want.iter().all(|w| got.contains(w)) {
                    extra = true;
                } else {
                    missing = true;
                }
            }
            _ => {}
        }
    }
    if missing {
        return Code::MISSING_COMPANION_ROLE;
    }
    if extra {
        return Code::DOUBLE_ENCODE;
    }
    if rejects
        .iter()
        .any(|(_, r)| matches!(r, Reject::Shape { .. }))
    {
        return Code::SHAPE_MISMATCH;
    }
    if rejects.iter().any(|(_, r)| {
        matches!(
            r,
            Reject::Carrier { .. } | Reject::Rank { .. } | Reject::Logical(_)
        )
    }) {
        return Code::DTYPE_MISMATCH;
    }
    Code::UNKNOWN_ENCODING
}

/// One tensor's chosen encoding and the carrier tensors its companion roles come from.
struct Chosen<'a> {
    alias: &'a str,
    spec: &'a EncodingSpec,
    data_role: &'static str,
    /// The dtype the tensor STANDS FOR, resolved from the chosen spec's data-role carrier.
    logical: Dtype,
    /// The shape the tensor STANDS FOR, recovered by running the data role's own relation
    /// backwards. Equal to the stored shape for every `Relation::Same` data role; for a
    /// PACKED one (`nvfp4-w4a4/1`'s `[out, in/2]`) it is the `[out, in]` the packing came
    /// from, which is the only shape the companion relations mean anything against.
    logical_shape: Vec<u64>,
    present: Vec<(String, &'a super::carrier::SourceTensor)>,
}

/// Try ONE candidate against ONE tensor's observed siblings. Returns the fit or the reason.
#[allow(clippy::type_complexity)]
fn fit<'a>(
    alias: &'a str,
    spec: &'a EncodingSpec,
    t: &super::carrier::SourceTensor,
    siblings: &[(&'static str, &'a super::carrier::SourceTensor)],
    base: Option<Dtype>,
) -> Result<std::result::Result<Chosen<'a>, Reject>> {
    let map = role_map(alias)?;
    let mut present: Vec<(String, &super::carrier::SourceTensor)> = Vec::new();
    for (sfx, st) in siblings {
        match map.companions.iter().find(|(s, _)| s == sfx) {
            Some((_, role)) => present.push((role.to_string(), st)),
            None => return Ok(Err(Reject::Surplus(sfx.to_string()))),
        }
    }
    // Roles are EXACT: a spec with a different role set is a different digest, so the
    // observed set must BE the candidate's, never a subset and never a superset.
    let want: Vec<String> = spec
        .roles
        .iter()
        .map(|(n, _)| n.clone())
        .filter(|n| n != map.data)
        .collect();
    let mut got: Vec<String> = present.iter().map(|(n, _)| n.clone()).collect();
    let (mut a, mut b) = (got.clone(), want.clone());
    a.sort();
    b.sort();
    if a != b {
        got.sort();
        return Ok(Err(Reject::RoleSet { got, want: b }));
    }
    // The data role's own carrier type. Keyed on the SPEC, never on the dtype spelling: fp8
    // storage with scales and fp8 storage without declare the same element type.
    let data_role = match spec.role(map.data) {
        Some(r) => r,
        None => {
            return refuse(
                Code::ROLE_SET_MISMATCH,
                format!(
                    "{alias}: the reviewed map names data role {:?}, absent from the spec",
                    map.data
                ),
            )
        }
    };
    // THE LOGICAL SHAPE, recovered before anything is evaluated. Every relation in a spec is
    // written against the tensor's LOGICAL shape — the store's own validator evaluates them
    // that way (`header.rs`) — and for a `Relation::Same` data role the stored shape IS the
    // logical one, which is why every encoding registered before nvfp4 was right by accident.
    // `nvfp4-w4a4/1` packs two fp4 elements per byte, so its data role stores `[out, in/2]`;
    // evaluating `weight_scale`'s `in/16` against THAT asks for `in/32` scales, refuses every
    // spec-correct carrier and admits one with half the scales it needs. The data role knows
    // what it packed, so it is asked.
    let stored_rank = match &data_role.shape {
        crate::relation::Relation::Same => t.shape.len(),
        crate::relation::Relation::Dims(d) => d.len(),
    };
    if stored_rank != t.shape.len() {
        return Ok(Err(Reject::Rank {
            want: stored_rank as u64,
            got: t.shape.len(),
        }));
    }
    let logical_shape = match data_role.shape.invert(&t.shape)? {
        Some(s) => s,
        None => {
            return refuse(
                Code::UNKNOWN_RELATION,
                format!(
                    "{alias}: the data role {:?} declares relation {:?}, which cannot be run \
                     backwards — so the tensor's LOGICAL shape is not recoverable from what it \
                     stores, and every companion relation in the spec is written against that \
                     logical shape. A data role that packs must pack invertibly (`axis`/`div` \
                     over the logical axes); this is a registry defect, not a carrier one.",
                    map.data, data_role.shape
                ),
            )
        }
    };
    if let Some(r) = spec.logical_rank {
        if r as usize != logical_shape.len() {
            return Ok(Err(Reject::Rank {
                want: r,
                got: logical_shape.len(),
            }));
        }
    }
    // A spec that stores its OWN carrier type keeps the logical dtype out of the file, so it
    // is read off this artifact's unencoded float weights — or refused. Never assumed.
    let logical = match &data_role.carrier {
        crate::relation::Carrier::SameAsLogical => t.dtype,
        crate::relation::Carrier::Set(_) => match base {
            Some(d) => d,
            None => {
                return refuse(
                    Code::DTYPE_UNKNOWN,
                    format!(
                        "{}: target {alias} stores {} carrier bytes, so the logical dtype is                          not in the file — and this carrier holds no unencoded float weight                          to read it from. The logical type is homeless; an explicit profile                          must state it.",
                        t.key,
                        t.dtype.name()
                    ),
                )
            }
        },
    };
    if !spec.admits_logical(logical) {
        return Ok(Err(Reject::Logical(logical.name())));
    }
    if !data_role.carrier.admits(logical, t.dtype) {
        return Ok(Err(Reject::Carrier {
            role: map.data.to_string(),
            got: t.dtype.name(),
        }));
    }
    // Companion geometry against the spec's own relation, checked here rather than after
    // 5 GB have moved. Two scale geometries under one alias are two DIGESTS: a reader that
    // assumes the other mis-strides every row and yields plausible, wrong numbers.
    let elements = crate::dtype::checked_elements(&t.key, &logical_shape)?;
    for (name, st) in &present {
        let role = match spec.role(name) {
            Some(r) => r,
            None => unreachable!("the role set was just compared"),
        };
        let want_shape = role.shape.eval(&logical_shape, elements)?;
        if want_shape != st.shape {
            return Ok(Err(Reject::Shape {
                role: name.clone(),
                got: st.shape.clone(),
                want: want_shape,
            }));
        }
        if !role.carrier.admits(logical, st.dtype) {
            return Ok(Err(Reject::Carrier {
                role: name.clone(),
                got: st.dtype.name(),
            }));
        }
    }
    Ok(Ok(Chosen {
        alias,
        spec,
        data_role: map.data,
        logical,
        logical_shape,
        present,
    }))
}

#[allow(clippy::too_many_arguments)]
fn plan_carrier(
    conv: &Converter,
    component: &str,
    file: usize,
    h: &SourceHeader,
    allowed: &[&str],
    cands: &Candidates<'_>,
    p: &mut Plan,
) -> Result<()> {
    if conv.name == "h3.lora/2" {
        if allowed != ["plain/1"] {
            return refuse(
                Code::UNKNOWN_ENCODING,
                "H3 LoRA ingestion preserves floating point factors in plain/1",
            );
        }
        return super::h3_lora::plan(component, file, h, cands.plain, p);
    }
    if conv.name == IDENTITY {
        return plan_as_is(component, file, h, allowed, cands.plain, p);
    }
    if let Some(table) = super::routes::table(conv.name) {
        return super::routes::plan(conv, table, component, file, h, allowed, cands.plain, p);
    }
    let alias = Allowed(allowed);
    p.header_bytes += h.header_bytes;
    let plain_map = role_map("plain/1")?;
    let cand_suffixes = cands.suffixes()?;
    let companions = cands.companion_roles()?;
    let mut encoded_tensors = 0usize;
    // Read ONCE per component: it is a whole-header majority vote, and re-running it per
    // tensor per candidate makes the planner quadratic in the key count for no new fact.
    let base = base_float(h);
    // Does ANYTHING in this component carry a candidate's roles? The answer separates two
    // refusals that look identical per-tensor and mean opposite things: a component where
    // nothing carries scales is scale-free STORAGE (and converting it is arithmetic), while
    // one where 199 tensors carry scales and the 200th does not is a TORN source.
    let component_is_encoded = h.tensors.iter().any(|t| {
        module_of(&t.key)
            .map(|m| {
                cand_suffixes
                    .iter()
                    .any(|s| h.get(&format!("{m}{s}")).is_some())
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

        // What roles does THIS TENSOR actually have in the carrier? This is the whole
        // observation, and it is what decides the encoding — never the component, never the
        // target, never a `.comfy_quant` marker (whose payload is tensor BYTES this planner
        // structurally does not read), and never the dtype spelling alone (v1's lane table
        // was keyed on the dtype and that WAS a real defect: scale-free fp8 storage and
        // scaled fp8 rowwise declare the same element type and execute in different lanes).
        let siblings: Vec<(&'static str, &super::carrier::SourceTensor)> = match module_of(&t.key) {
            None => Vec::new(),
            Some(m) => super::fingerprint::ROLE_SUFFIXES
                .iter()
                .filter_map(|s| h.get(&format!("{m}{s}")).map(|st| (*s, st)))
                .collect(),
        };
        let present_sfx: Vec<&'static str> = siblings.iter().map(|(s, _)| *s).collect();

        // A reviewed SEAM: this fused key is not ONE output tensor, it is several, and the
        // geometry that cuts it is read off the carrier's own sibling norm.
        if let Some(seam) = seams(conv).iter().find(|s| t.key.ends_with(s.fused)) {
            plan_seam(conv, component, file, h, seam, t, &siblings, cands.plain, p)?;
            continue;
        }
        // A permutation converter that carried a fused attention key THROUGH would emit a
        // canonical tensor with the right name, dtype and shape and a packing no destination
        // expects — the same silent class as a flat three-way split, one level up.
        if conv.class == Class::Permutation && is_fused_attention(&t.key) {
            return refuse(
                Code::SEAM_UNBANKED,
                format!(
                    "{}: a fused attention projection, and {} banks no seam for it. Its \
                     reviewed seams are {:?}. A permutation converter does not carry a fused \
                     key through — the head geometry of THIS module family has never been \
                     measured, and guessing it is the ~90% numerical error that does not crash.",
                    t.key,
                    conv.name,
                    seams(conv).iter().map(|s| s.fused).collect::<Vec<_>>()
                ),
            );
        }

        // A real component is MIXED: a quantized weight sits beside its own unquantized
        // bias and the norms the producer's eligibility predicate skipped. A tensor with NO
        // companion roles is plain storage. One rule over every tensor in a tree is the
        // shape this design explicitly rejects.
        let chosen = if siblings.is_empty() {
            if matches!(t.dtype, Dtype::F8E4M3FN | Dtype::F8E5M2) && !companions.is_empty() {
                if component_is_encoded {
                    return refuse(
                        Code::MISSING_COMPANION_ROLE,
                        format!(
                            "{}: an encoded {} weight with no {companions:?} sibling, in a \
                             component where other tensors have theirs. That is a TORN \
                             source, not scale-free storage — the bytes exist and their \
                             scale does not.",
                            t.key,
                            t.dtype.name()
                        ),
                    );
                }
                // Scale-FREE fp8 offered under a scaled target. The absence of scales IS an
                // identity: the two have the same element type and the same tensor names,
                // and a reader that takes one for the other either divides by scales that
                // are not there or fails to divide by scales that are. Both are silent.
                return refuse(
                    Code::REMEDY_FROM_TARGET,
                    format!(
                        "{}: scale-free {} storage offered under target {alias}, which \
                         requires {companions:?}. The absence of scales is the identity, not \
                         an omission — producing them is the priced PRODUCIBLE quantize job.",
                        t.key,
                        t.dtype.name()
                    ),
                );
            }
            Chosen {
                alias: "plain/1",
                spec: cands.plain,
                data_role: plain_map.data,
                logical: t.dtype,
                logical_shape: t.shape.clone(),
                present: Vec::new(),
            }
        } else {
            let mut fits: Vec<Chosen<'_>> = Vec::new();
            let mut rejects: Vec<(String, Reject)> = Vec::new();
            for (a, sp) in &cands.encoded {
                match fit(a, sp, t, &siblings, base)? {
                    Ok(c) => fits.push(c),
                    Err(r) => rejects.push((format!("{a} {}", short(&sp.object_id())), r)),
                }
            }
            match fits.len() {
                1 => {
                    encoded_tensors += 1;
                    fits.pop().unwrap()
                }
                0 => {
                    let tried: Vec<String> =
                        rejects.iter().map(|(n, r)| format!("{n} ({r})")).collect();
                    return refuse(
                        code_for(&rejects),
                        format!(
                            "{}: the carrier supplies roles {present_sfx:?} for this tensor \
                             and no candidate of target {alias} takes exactly those. Tried: \
                             {}. Selection is per TENSOR and reads the siblings PRESENT — a \
                             partial role set is a TORN source and a surplus one is an \
                             encoding fact the target has no room for; neither is guessed \
                             past, and the remedy derives from the target's contract, never \
                             from what happened to be observed.",
                            t.key,
                            if tried.is_empty() {
                                format!("nothing — {alias} declares no encoded variant")
                            } else {
                                tried.join("; ")
                            }
                        ),
                    );
                }
                n => {
                    return refuse(
                        Code::AMBIGUOUS_CLASSIFICATION,
                        format!(
                            "{}: roles {present_sfx:?} fit {n} physically distinct specs of \
                             {alias} — {:?}. Two specs that admit the same bytes is a REGISTRY \
                             defect, not a tiebreak.",
                            t.key,
                            fits.iter()
                                .map(|c| short(&c.spec.object_id()))
                                .collect::<Vec<_>>()
                        ),
                    )
                }
            }
        };

        let logical_dtype = chosen.logical;
        let data = chosen.data_role.to_string();
        let mut roles = vec![RolePlan {
            role: data.clone(),
            dtype: t.dtype,
            shape: t.shape.clone(),
            bytes: Bytes::Stream {
                file,
                key: t.key.clone(),
            },
        }];
        for (name, st) in &chosen.present {
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
            chosen
                .spec
                .roles
                .iter()
                .position(|(n, _)| *n == r.role)
                .unwrap_or(usize::MAX)
        });

        let out_key = rekey(conv, &t.key);
        let effect = if conv.name == "h3.native/2"
            && (t.key.ends_with(".mlp.fc1.weight") || t.key.ends_with(".mlp.fc1.bias"))
        {
            if chosen.alias != "plain/1" {
                return refuse(
                    Code::SEAM_UNBANKED,
                    format!(
                        "{}: H3 fused MLP companion roles require a reviewed encoded permutation",
                        t.key
                    ),
                );
            }
            let xform = h3_swiglu(h, t)?;
            roles[0].bytes = Bytes::Permute {
                file,
                key: t.key.clone(),
                xform,
            };
            Effect::Permute
        } else if out_key == t.key {
            Effect::Recontain
        } else {
            Effect::Rekey
        };
        p.ops.push(Op {
            component: component.to_string(),
            out_key,
            encoding: chosen.alias.to_string(),
            encoding_id: chosen.spec.object_id(),
            logical_dtype,
            logical_shape: chosen.logical_shape.clone(),
            roles,
            effect,
        });
    }

    // The component's target named an encoding and NOTHING in the component carries it.
    // Producing it is arithmetic; the verdict's remedy derives from the TARGET's contract.
    if !alias.is_plain_only() && encoded_tensors == 0 {
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

/// Every key, its own plain tensor: no role folds, no marker drops, no rename.
fn plan_as_is(
    component: &str,
    file: usize,
    h: &SourceHeader,
    allowed: &[&str],
    plain: &EncodingSpec,
    p: &mut Plan,
) -> Result<()> {
    if allowed != ["plain/1"] {
        return refuse(
            Code::UNKNOWN_ENCODING,
            format!("{IDENTITY} stores an unrecognized source in plain/1 only"),
        );
    }
    p.header_bytes += h.header_bytes;
    let (data, id) = (role_map("plain/1")?.data, plain.object_id());
    for t in &h.tensors {
        p.ops.push(Op {
            component: component.to_string(),
            out_key: t.key.clone(),
            encoding: "plain/1".into(),
            encoding_id: id.clone(),
            logical_dtype: t.dtype,
            logical_shape: t.shape.clone(),
            roles: vec![RolePlan {
                role: data.into(),
                dtype: t.dtype,
                shape: t.shape.clone(),
                bytes: Bytes::Stream {
                    file,
                    key: t.key.clone(),
                },
            }],
            effect: Effect::Recontain,
        });
    }
    Ok(())
}

/// The original H3 MLP consumes [gate; value], while Diffusers SwiGLU consumes
/// [value; gate]. The sibling fc2 projection proves the gate width, for main and
/// token-refiner blocks alike. Only source runs move; no tensor value changes.
fn h3_swiglu(h: &SourceHeader, t: &super::carrier::SourceTensor) -> Result<Xform> {
    let (base, bias) = if let Some(base) = t.key.strip_suffix(".fc1.weight") {
        (base, false)
    } else {
        (
            t.key.strip_suffix(".fc1.bias").expect("selected fc1 role"),
            true,
        )
    };
    let first = h
        .get(&format!("{base}.fc1.weight"))
        .ok_or_else(|| crate::err::Refusal {
            code: Code::MISSING_COMPANION_ROLE,
            detail: format!("{}: fused MLP weight is absent", t.key),
        })?;
    let second = h
        .get(&format!("{base}.fc2.weight"))
        .ok_or_else(|| crate::err::Refusal {
            code: Code::MISSING_COMPANION_ROLE,
            detail: format!("{}: output MLP projection is absent", t.key),
        })?;
    if first.shape.len() != 2
        || second.shape.len() != 2
        || first.shape[0] == 0
        || first.shape[1] == 0
        || first.shape[0] % 2 != 0
        || first.shape[0] / 2 != second.shape[1]
        || first.shape[1] != second.shape[0]
        || (bias && (t.shape.len() != 1 || t.shape[0] != first.shape[0]))
    {
        return refuse(
            Code::SHAPE_MISMATCH,
            format!(
                "{}: fused MLP must contain equal gate/value row halves matching fc2",
                t.key
            ),
        );
    }
    check_permute_size(&t.key, t.nbytes())?;
    Ok(Xform::SwapHalves {
        half_bytes: t.nbytes() / 2,
    })
}

/// Plan ONE fused key into its members. Nothing is read: the cut comes from the declared
/// geometry plus the sibling norm's width, both header facts.
#[allow(clippy::too_many_arguments)]
fn plan_seam(
    conv: &Converter,
    component: &str,
    file: usize,
    h: &SourceHeader,
    seam: &Seam,
    t: &super::carrier::SourceTensor,
    siblings: &[(&'static str, &super::carrier::SourceTensor)],
    plain: &EncodingSpec,
    p: &mut Plan,
) -> Result<()> {
    if !siblings.is_empty() {
        return refuse(
            Code::SEAM_UNBANKED,
            format!(
                "{}: a fused attention key carrying encoding roles {:?}. The banked seam cuts \
                 the DATA role on whole head groups; where the same cut lands on a companion \
                 scale is banked for no artifact that exists, and splitting a weight while \
                 leaving its scale whole is the silent-wrong-numbers class exactly. The \
                 remedy derives from the target: the priced job over the plain source.",
                t.key,
                siblings.iter().map(|(s, _)| *s).collect::<Vec<_>>()
            ),
        );
    }
    if !plain.admits_logical(t.dtype) {
        return refuse(
            Code::DTYPE_MISMATCH,
            format!(
                "{}: plain/1 does not admit logical {}",
                t.key,
                t.dtype.name()
            ),
        );
    }
    let rows = match t.shape.first() {
        Some(r) => *r,
        None => {
            return refuse(
                Code::SHAPE_MISMATCH,
                format!("{}: a rank-0 tensor has no fused axis", t.key),
            )
        }
    };
    let base = &t.key[..t.key.len() - seam.fused.len()];
    let row_bytes = checked_bytes(&t.key, &t.shape[1..], t.dtype)?;
    let (groups, unit_bytes) =
        h3_geometry(h, &t.key, &format!("{base}{}", seam.norm), rows, row_bytes)?;
    check_permute_size(&t.key, t.nbytes())?;

    let mut shape = t.shape.clone();
    shape[0] = rows / 3;
    for (i, leaf) in seam.members.iter().enumerate() {
        p.ops.push(Op {
            component: component.to_string(),
            out_key: rekey(conv, &format!("{base}{leaf}")),
            encoding: "plain/1".to_string(),
            encoding_id: plain.object_id(),
            logical_dtype: t.dtype,
            logical_shape: shape.clone(),
            roles: vec![RolePlan {
                role: role_map("plain/1")?.data.to_string(),
                dtype: t.dtype,
                shape: shape.clone(),
                bytes: Bytes::Permute {
                    file,
                    key: t.key.clone(),
                    xform: Xform::QkvSplit {
                        groups,
                        shares: [1, 1, 1],
                        take: i as u8,
                        unit_bytes,
                    },
                },
            }],
            effect: Effect::Permute,
        });
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
    allowed: &[&str],
    cands: &Candidates<'_>,
    rekey: &Rekey,
    p: &mut Plan,
) -> Result<()> {
    let alias = Allowed(allowed);
    // The candidate set, by DIGEST — an already-canonical component is mixed exactly as its
    // carrier was, so a re-ingest matches each tensor against every candidate, not one.
    let mut ids: Vec<(String, &str)> = vec![(cands.plain.object_id(), "plain/1")];
    for (a, sp) in &cands.encoded {
        ids.push((sp.object_id(), a));
    }
    for (comp, key, t) in h.tensors() {
        if comp != component {
            continue;
        }
        let chosen = match ids.iter().find(|(id, _)| *id == t.encoding) {
            Some((_, a)) => *a,
            None => {
                return refuse(
                    Code::DOUBLE_ENCODE,
                    format!(
                        "{comp}/{key}: already encoded under {} and the target asks for \
                         {alias} ({:?}) — re-encoding already-encoded bytes needs the priced \
                         job, not the border",
                        t.encoding,
                        ids.iter().map(|(i, _)| short(i)).collect::<Vec<_>>()
                    ),
                );
            }
        };
        let out_key = rekey.apply(component, key);
        let effect = if out_key == *key {
            Effect::None
        } else {
            Effect::Rekey
        };
        p.ops.push(Op {
            component: component.to_string(),
            out_key,
            encoding: chosen.to_string(),
            encoding_id: t.encoding.clone(),
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
            .find(|(from, _, _)| key.starts_with(*from))
            .map(|(from, _, to)| format!("{to}{}", &key[from.len()..]))
            .unwrap_or_else(|| key.to_string()),
        // Leaf first, then the block prefix — the two halves of the ratified map's own
        // `rekey` block, applied in the order it lists them.
        "h3.native/2" => {
            if let Some((_, to)) = H3_NATIVE_EXACT.iter().find(|(from, _)| *from == key) {
                return to.to_string();
            }
            let leafed = H3_NATIVE_LEAVES
                .iter()
                .find(|(from, _)| key.ends_with(*from))
                .map(|(from, to)| format!("{}{to}", &key[..key.len() - from.len()]))
                .unwrap_or_else(|| key.to_string());
            H3_NATIVE_PREFIXES
                .iter()
                .find(|(from, _)| leafed.starts_with(*from))
                .map(|(from, to)| format!("{to}{}", &leafed[from.len()..]))
                .unwrap_or(leafed)
        }
        _ => key.to_string(),
    }
}

/// The ratified H3 native → canonical (Diffusers) key map, INVERTED from the v1 seam
/// registry `minimax-h3.split-to-fused-qkv.v1.json` (`digest 18ed116c…`, banked in
/// h3-evidence's reconciliation row). Its `rekey` block is these five leaf rewrites plus the
/// block prefix; nothing here is inferred from a name similarity.
///
/// The 17 edge-module rewrites, each PROVEN by byte-digest equality on the real artifacts
/// (2026-08-25, tfs-003 convergence run: MiniMaxAI/MiniMax-H3@42ed227e, native FL2VA vs
/// diffusers packaging, same revision — every pair's sha256 over the tensor's exact data
/// run is identical, and the pairing is a bijection). The earlier map left these unclaimed
/// because v1 evidence could not prove them; real bytes now can, so leaving them unmapped
/// would make the two packagings' canonical projections diverge over a correspondence that
/// is measured fact. `rope.inv_freq` is the ONE native key that byte-matches nothing in
/// the diffusers packaging (a [16] f32 rope table; values follow 1/10000^(2i/32) but a
/// bit-exact derivation was not reproduced, so it is CARRIED unrenamed — never invented,
/// never dropped).
const H3_NATIVE_EXACT: [(&str, &str); 17] = [
    ("audio_patch_proj.bias", "audio_proj_in.bias"),
    ("audio_patch_proj.weight", "audio_proj_in.weight"),
    ("condition_proj.bias", "context_embedder.bias"),
    ("condition_proj.weight", "context_embedder.weight"),
    ("final_layer.adaln_proj.linear.bias", "norm_out.linear.bias"),
    (
        "final_layer.adaln_proj.linear.weight",
        "norm_out.linear.weight",
    ),
    ("final_layer.audio_out.bias", "audio_proj_out.bias"),
    ("final_layer.audio_out.weight", "audio_proj_out.weight"),
    ("final_layer.norm.weight", "norm_out.norm.weight"),
    ("final_layer.video_out.bias", "proj_out.bias"),
    ("final_layer.video_out.weight", "proj_out.weight"),
    ("time_embedder.proj_in.bias", "time_embedder.linear_1.bias"),
    (
        "time_embedder.proj_in.weight",
        "time_embedder.linear_1.weight",
    ),
    ("time_embedder.proj_out.bias", "time_embedder.linear_2.bias"),
    (
        "time_embedder.proj_out.weight",
        "time_embedder.linear_2.weight",
    ),
    ("video_patch_proj.bias", "proj_in.bias"),
    ("video_patch_proj.weight", "proj_in.weight"),
];

const H3_NATIVE_LEAVES: [(&str, &str); 6] = [
    (".attn.out_proj.weight", ".attn.to_out.0.weight"),
    (".attn.q_norm.weight", ".attn.norm_q.weight"),
    (".attn.k_norm.weight", ".attn.norm_k.weight"),
    (".mlp.fc1.weight", ".ff.net.0.proj.weight"),
    (".mlp.fc1.bias", ".ff.net.0.proj.bias"),
    (".mlp.fc2.weight", ".ff.net.2.weight"),
];

const H3_NATIVE_PREFIXES: [(&str, &str); 2] = [
    ("token_refiner.blocks.", "token_refiner.refiner_blocks."),
    ("blocks.", "transformer_blocks."),
];

/// A reviewed SEAM: one fused carrier key becomes several canonical keys under a
/// permutation rule, and the geometry is READ from a named sibling rather than assumed.
///
/// Keyed on the module LEAF, not on a block family. v1's seam enumerated
/// `blocks.{i}.attn.qkv_proj.weight` and therefore covered 50 of the real artifact's 52
/// fused modules — the two `token_refiner` attentions fell out, and a converter that fuses
/// 50 produces a key set the real native artifact does not have (h3-evidence reconciliation,
/// `action_for_tfs_003`). A leaf-keyed seam covers every attention module family there is,
/// and each one's head dim comes from its OWN `q_norm`.
pub struct Seam {
    pub fused: &'static str,
    /// The sibling whose width IS the head dim. Never a constant (see `h3_geometry`).
    pub norm: &'static str,
    /// Canonical leaf per member, in SOURCE order: the fused axis is (q, k, v) per head.
    pub members: [&'static str; 3],
}

pub const H3_ATTENTION_SEAM: [Seam; 1] = [Seam {
    fused: ".attn.qkv_proj.weight",
    norm: ".attn.q_norm.weight",
    members: [
        ".attn.to_q.weight",
        ".attn.to_k.weight",
        ".attn.to_v.weight",
    ],
}];

pub fn seams(c: &Converter) -> &'static [Seam] {
    match c.name {
        "h3.native/2" => &H3_ATTENTION_SEAM,
        _ => &[],
    }
}

/// Does this key name a FUSED attention projection at all? The spellings are the ones the
/// pinned H3 tree actually ships: the DiT's `attn.qkv_proj`, the video VAE's 1x1-conv
/// `attn.to_qkv`, the text encoder's and audio VAE's `attn.qkv`.
///
/// A PERMUTATION-class converter that leaves one of these unsplit would emit a canonical
/// tensor whose every name, dtype and shape is right and whose contents are a packing no
/// destination expects. Identity converters are not governed: for them "bytes as arrived"
/// IS the contract, and the fused key stays fused and says so.
pub fn is_fused_attention(key: &str) -> bool {
    let base = key
        .strip_suffix(".weight")
        .or_else(|| key.strip_suffix(".bias"))
        .unwrap_or(key);
    base.ends_with(".attn.qkv_proj")
        || base.ends_with(".attn.to_qkv")
        || base.ends_with(".attn.qkv")
}

/// The reviewed single-file → canonical component prefix map. Data, not inference: a prefix
/// that is not here does not get guessed at, it stays as it is and the fingerprint decides.
pub const SINGLE_FILE_PREFIXES: [(&str, &str, &str); 4] = [
    ("model.diffusion_model.", "unet", ""),
    ("first_stage_model.", "vae", ""),
    ("conditioner.embedders.0.transformer.", "text_encoder", ""),
    ("conditioner.embedders.1.model.", "text_encoder_2", ""),
];

/// Project one reviewed component out of a multi-component single-file carrier. The
/// source offsets remain those of the original file; only the key set presented to
/// fingerprinting and planning narrows. The converter's prefixes are the sole authority (a
/// routed converter's table, else the closed map above): a key belongs to the component
/// whose prefix is the longest it starts with, and an absent component refuses.
pub fn single_file_component(
    conv: &Converter,
    component: &str,
    source: &SourceHeader,
) -> Result<SourceHeader> {
    let prefixes: Vec<(&str, &str)> = match super::routes::table(conv.name) {
        Some(table) => table
            .components
            .iter()
            .map(|c| (c.name.as_str(), c.prefix.as_str()))
            .collect(),
        None => SINGLE_FILE_PREFIXES
            .iter()
            .map(|(prefix, name, _)| (*name, *prefix))
            .collect(),
    };
    let Some(&(_, prefix)) = prefixes.iter().find(|(name, _)| *name == component) else {
        return refuse(
            Code::MISSING_TENSOR,
            format!("{} defines no reviewed component {component:?}", conv.name),
        );
    };
    let owner = |key: &str| {
        prefixes
            .iter()
            .map(|(_, p)| *p)
            .filter(|p| key.starts_with(p))
            .max_by_key(|p| p.len())
    };
    let tensors: Vec<_> = source
        .tensors
        .iter()
        .filter(|tensor| owner(&tensor.key) == Some(prefix))
        .cloned()
        .collect();
    if tensors.is_empty() {
        return refuse(
            Code::MISSING_TENSOR,
            format!("single-file carrier contains no {component:?} tensors under {prefix:?}"),
        );
    }
    let claimed: u64 = tensors.iter().map(|tensor| tensor.nbytes()).sum();
    let data_bytes = source.file_len.saturating_sub(source.data_start);
    Ok(SourceHeader {
        tensors,
        metadata: source.metadata.clone(),
        header_bytes: source.header_bytes,
        data_start: source.data_start,
        file_len: source.file_len,
        gap_bytes: data_bytes.saturating_sub(claimed),
        shards: source.shards.clone(),
    })
}

/// The declared input-size guard for a permutation op; execution streams its source runs.
pub fn check_permute_size(what: &str, nbytes: u64) -> Result<()> {
    if nbytes > limits::CONVERT_TENSOR_MAX_BYTES {
        return refuse(
            Code::CONVERT_SIZE_CAP,
            format!(
                "{what}: permutation source is {nbytes} B, over the declared \
                 {} B converter input bound",
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
/// `Xform::reassembles` checks that each declared transform preserves the source bytes.
/// Both a contiguous split and a head-interleaved split can round-trip their own indexing;
/// the producer's layout needs an independent projected-output test (see `h3_lora`).
pub fn golden_cases(c: &Converter) -> Vec<(&'static str, Xform)> {
    match c.name {
        "h3.lora/2" => vec![
            (
                "H3 LoRA SwiGLU B rows",
                Xform::SwapHalves {
                    half_bytes: 14336 * 16 * 2,
                },
            ),
            (
                "H3 trainer LoRA contiguous QKV B thirds",
                Xform::QkvSplit {
                    groups: 1,
                    shares: [1, 1, 1],
                    take: 0,
                    unit_bytes: H3_HEADS * H3_HEAD_DIM * 16 * 2,
                },
            ),
        ],
        name if name == super::routes::SDXL => vec![
            (
                "OpenCLIP in_proj_bias, row block 2 of 3 (1280 x f16)",
                Xform::QkvSplit {
                    groups: 1,
                    shares: [2, 1, 0],
                    take: 1,
                    unit_bytes: 1280 * 2,
                },
            ),
            (
                "OpenCLIP text_projection 3 x 5 x f32 transpose",
                Xform::Transpose {
                    rows: 3,
                    cols: 5,
                    elem_bytes: 4,
                },
            ),
        ],
        "h3.native/2" => vec![
            (
                "h3-swiglu gate/value rows",
                Xform::SwapHalves { half_bytes: 64 * 2 },
            ),
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

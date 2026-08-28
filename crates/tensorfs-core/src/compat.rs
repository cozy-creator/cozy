//! `check_compatibility` — the layout plane's SECOND operation (Stamp at the border is the
//! first). It compares a bound model's CODE TOPOLOGY with a checkpoint's topology and
//! validates encoding assignments against the pinned registry.
//!
//! It returns the ONE runtime-satisfaction vocabulary: `Satisfied | Mismatch(details)`.
//! It plans nothing, executes nothing, and names no remedy. **Grading a mismatch into
//! COMPATIBLE/CONVERTIBLE/PRODUCIBLE/INCOMPATIBLE is the hub's job** — a second grading
//! layer below the hub would be a third vocabulary for one decision chain, and the fence in
//! `scripts/fence.py` keeps that vocabulary out of this file.
//!
//! What crosses the boundary is a DOCUMENT, not a foreign schema. cozy-runtime owns
//! `ModelConstructionContract` (cr-004) and TensorFS must never grow a second parser for it
//! (tfs-013's zero-foreign-schemas fence), so the contract PROJECTS to `CodeTopology` — the
//! same `component -> key -> {dtype, shape}` projection the header already computes, plus
//! the per-key encoding requirement. `contract` records the digest of the foreign document
//! it was projected from; TensorFS records that digest and never interprets it.
//!
//! The containment ladder is the authority on what matching may look at:
//!   structure projection [keys+shapes] ⊂ checkpoint topology [+encoding ids] ⊂ header
//!   [+parts, objects, configs]
//! Parts and objects are the header's fill and transfer layers, BELOW the topology and
//! invisible to matching. `topology_digest` is the fast-path index, never the decision.

use crate::canon::{as_arr, as_obj, as_str, Fields, Value};
use crate::dtype::Dtype;
use crate::err::{refuse, Code, Result};
use crate::header::{Closure, Header};
use crate::ids::{ascii_name, check_format, prefixed, Doc};
use crate::limits;

// ---------------------------------------------------------------- the required side

/// Which encodings a code destination will accept for one tensor. Closed and REQUIRED —
/// an absent field defaulting to "anything" is how a lane silently widens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Accepts {
    /// Any encoding the pinned registry knows. Plain-only destinations do NOT spell this.
    Any,
    /// Exactly these spec digests, sorted and unique.
    OneOf(Vec<String>),
}

impl Accepts {
    pub fn admits(&self, encoding: &str) -> bool {
        match self {
            Accepts::Any => true,
            Accepts::OneOf(v) => v.iter().any(|e| e == encoding),
        }
    }
    fn to_value(&self) -> Value {
        match self {
            Accepts::Any => Value::str("any"),
            Accepts::OneOf(v) => Value::arr(v.iter().map(|e| Value::str(e.clone())).collect()),
        }
    }
    fn from_value(what: &str, v: &Value) -> Result<Accepts> {
        match v {
            Value::Str(s) if s == "any" => Ok(Accepts::Any),
            Value::Str(s) => refuse(
                Code::UNKNOWN_FIELD,
                format!("{what}: accepts {s:?} is neither \"any\" nor a digest list"),
            ),
            Value::Arr(a) => {
                if a.is_empty() {
                    return refuse(
                        Code::MISSING_FIELD,
                        format!("{what}: an empty accepts list admits nothing and says nothing"),
                    );
                }
                let mut out = Vec::new();
                for x in a {
                    out.push(prefixed(what, as_str(what, "accepts", x)?)?);
                }
                let mut sorted = out.clone();
                sorted.sort();
                sorted.dedup();
                if sorted != out {
                    return refuse(
                        Code::SORT_ORDER,
                        format!("{what}: accepts must be sorted and unique"),
                    );
                }
                Ok(Accepts::OneOf(out))
            }
            o => refuse(
                Code::WRONG_TYPE,
                format!("{what}: accepts is a string or an array, got {}", o.kind()),
            ),
        }
    }
}

/// One code destination: what `load()` actually constructs, as a byte contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Requirement {
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub accepts: Accepts,
}

/// The bounded projection of a ModelConstructionContract. TensorFS's whole view of the
/// code side, and the only thing that crosses into the verifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeTopology {
    /// Digest of the FOREIGN document this was projected from. Recorded, never parsed.
    pub contract: String,
    pub components: Vec<(String, Vec<(String, Requirement)>)>,
}

impl CodeTopology {
    pub fn tensors(&self) -> impl Iterator<Item = (&str, &str, &Requirement)> {
        self.components
            .iter()
            .flat_map(|(c, ks)| ks.iter().map(move |(k, r)| (c.as_str(), k.as_str(), r)))
    }

    pub fn count(&self) -> usize {
        self.components.iter().map(|(_, k)| k.len()).sum()
    }

    /// The SAME projection function a header computes, so the fast-path index is comparable
    /// on both sides. A digest hit means the logical {dtype, shape} sets agree — it never
    /// means the encodings do, which is why it can only ever be an index.
    pub fn topology_digest(&self) -> String {
        crate::header::topology_digest_of(&crate::header::topology_value(
            self.tensors()
                .map(|(c, k, r)| (c, k, r.dtype, r.shape.as_slice())),
        ))
    }
}

impl Doc for CodeTopology {
    const FORMAT: &'static str = "cozytensors.code_topology/1";
    const MAX_BYTES: usize = limits::DOC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("CodeTopology", v)?;
        check_format(
            "CodeTopology",
            as_str("CodeTopology", "format", f.req("format")?)?,
            Self::FORMAT,
        )?;
        let contract = prefixed("CodeTopology.contract", f.req_str("contract")?)?;
        let mut components = Vec::new();
        let mut total = 0usize;
        for (name, cv) in as_obj("CodeTopology", "components", f.req("components")?)? {
            ascii_name("CodeTopology.component", name, limits::MAX_NAME_BYTES)?;
            let mut keys = Vec::new();
            for (key, rv) in as_obj("CodeTopology.components", name, cv)? {
                if key.is_empty() || key.len() > limits::MAX_KEY_BYTES {
                    return refuse(
                        Code::KEY_GRAMMAR,
                        format!("{name}/{key}: key length out of bounds"),
                    );
                }
                total += 1;
                if total > limits::MAX_TENSORS {
                    return refuse(Code::COUNT_CAP, "code topology key count over cap");
                }
                let mut rf = Fields::new("Requirement", rv)?;
                let accepts = Accepts::from_value("Requirement", rf.req("accepts")?)?;
                let dtype = Dtype::parse(as_str("Requirement", "dtype", rf.req("dtype")?)?)?;
                let sa = as_arr("Requirement", "shape", rf.req("shape")?)?;
                if sa.len() > limits::MAX_RANK {
                    return refuse(Code::RANK_CAP, format!("{name}/{key}: rank over cap"));
                }
                let shape = sa
                    .iter()
                    .map(|x| crate::canon::as_uint("Requirement", "shape", x))
                    .collect::<Result<Vec<u64>>>()?;
                rf.done()?;
                crate::dtype::checked_elements(key, &shape)?;
                keys.push((
                    key.clone(),
                    Requirement {
                        dtype,
                        shape,
                        accepts,
                    },
                ));
            }
            if keys.is_empty() {
                return refuse(
                    Code::EMPTY_COMPONENTS,
                    format!("component {name:?} requires no tensor"),
                );
            }
            components.push((name.clone(), keys));
        }
        f.done()?;
        if components.is_empty() || components.len() > limits::MAX_COMPONENTS {
            return refuse(Code::EMPTY_COMPONENTS, "component count out of bounds");
        }
        Ok(CodeTopology {
            contract,
            components,
        })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            (
                "components",
                Value::map(
                    self.components
                        .iter()
                        .map(|(c, ks)| {
                            (
                                c.clone(),
                                Value::map(
                                    ks.iter()
                                        .map(|(k, r)| {
                                            (
                                                k.clone(),
                                                Value::obj(vec![
                                                    ("accepts", r.accepts.to_value()),
                                                    ("dtype", Value::str(r.dtype.name())),
                                                    (
                                                        "shape",
                                                        Value::arr(
                                                            r.shape
                                                                .iter()
                                                                .map(|d| Value::uint(*d))
                                                                .collect(),
                                                        ),
                                                    ),
                                                ]),
                                            )
                                        })
                                        .collect(),
                                ),
                            )
                        })
                        .collect(),
                ),
            ),
            ("contract", Value::str(self.contract.clone())),
            ("format", Value::str(Self::FORMAT)),
        ])
    }
}

// ---------------------------------------------------------------- the verdict

/// Where the checkpoint came from, which is the ONLY thing that changes the extra-key rule.
/// Missing keys refuse on both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lane {
    /// The border classified every key. A surplus at serve time is drift or corruption.
    Canonical,
    /// A local snapshot that never crossed the border: unknown extras are confessed.
    LocalUnbordered,
}

impl Lane {
    pub fn name(self) -> &'static str {
        match self {
            Lane::Canonical => "canonical",
            Lane::LocalUnbordered => "local_unbordered",
        }
    }
    pub fn parse(s: &str) -> Result<Lane> {
        match s {
            "canonical" => Ok(Lane::Canonical),
            "local_unbordered" => Ok(Lane::LocalUnbordered),
            o => refuse(
                Code::UNKNOWN_FIELD,
                format!("lane {o:?} is outside {{canonical, local_unbordered}}"),
            ),
        }
    }
}

/// One side of a disagreement, printed and stored verbatim. A refusal that names only the
/// key teaches nothing; a refusal that names both sides is a diff.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Side {
    pub dtype: Option<Dtype>,
    pub shape: Option<Vec<u64>>,
    pub encoding: Option<String>,
    /// `any`, a digest list, or absent when the side is a checkpoint.
    pub accepts: Option<String>,
}

impl Side {
    fn to_value(&self) -> Value {
        let mut f: Vec<(&str, Value)> = Vec::new();
        if let Some(a) = &self.accepts {
            f.push(("accepts", Value::str(a.clone())));
        }
        if let Some(d) = self.dtype {
            f.push(("dtype", Value::str(d.name())));
        }
        if let Some(e) = &self.encoding {
            f.push(("encoding", Value::str(e.clone())));
        }
        if let Some(s) = &self.shape {
            f.push((
                "shape",
                Value::arr(s.iter().map(|d| Value::uint(*d)).collect()),
            ));
        }
        if f.is_empty() {
            f.push(("absent", Value::Bool(true)));
        }
        Value::obj(f)
    }

    fn from_value(v: &Value) -> Result<Side> {
        let mut f = Fields::new("Side", v)?;
        let accepts = f.opt("accepts").map(|x| as_str("Side", "accepts", x));
        let _ = f.opt("absent");
        let dtype = match f.opt("dtype") {
            Some(x) => Some(Dtype::parse(as_str("Side", "dtype", x)?)?),
            None => None,
        };
        let encoding = match f.opt("encoding") {
            Some(x) => Some(prefixed("Side.encoding", as_str("Side", "encoding", x)?)?),
            None => None,
        };
        let shape = match f.opt("shape") {
            Some(x) => Some(
                as_arr("Side", "shape", x)?
                    .iter()
                    .map(|d| crate::canon::as_uint("Side", "shape", d))
                    .collect::<Result<Vec<u64>>>()?,
            ),
            None => None,
        };
        f.done()?;
        Ok(Side {
            dtype,
            shape,
            encoding,
            accepts: match accepts {
                Some(r) => Some(r?.to_string()),
                None => None,
            },
        })
    }
}

/// The FIRST disagreeing tensor, both sides, plus the component-granular encoding facts a
/// grader needs. Nothing here proposes a remedy or a verdict: the whole point of the split
/// is that this layer reports and the hub decides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Details {
    pub code: Code,
    pub component: String,
    pub key: String,
    pub required: Side,
    pub present: Side,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Satisfaction {
    Satisfied,
    Mismatch(Box<Details>),
}

impl Satisfaction {
    pub fn is_satisfied(&self) -> bool {
        matches!(self, Satisfaction::Satisfied)
    }
}

/// The MEASURED geometry of one role, read out of the header. Never inferred from a dtype
/// spelling: h3-evidence measures ComfyUI shipping rank-0 scalar `weight_scale` for
/// `fp8_scaled`, rank-2 `[out, 1]` for `int8_convrot` and rank-2 F8_E4M3 block scales for
/// `nvfp4_awq` — in ONE repository (decisions #244). A grader keyed on "fp8 implies rank-2"
/// misreads the largest H3 distribution by download count, so the facts carry what was read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleFact {
    pub role: String,
    pub carrier: Dtype,
    pub rank: u64,
    pub shape: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodingFact {
    pub encoding: String,
    pub keys: u64,
    pub logical_dtypes: Vec<Dtype>,
    /// The role geometry of the FIRST tensor citing this encoding in this component.
    pub roles: Vec<RoleFact>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentFacts {
    pub component: String,
    pub required_keys: u64,
    pub present_keys: u64,
    pub required_dtypes: Vec<(Dtype, u64)>,
    pub present_dtypes: Vec<(Dtype, u64)>,
    pub required_encodings: Accepts,
    pub present_encodings: Vec<EncodingFact>,
}

/// Everything one call produced: the two-valued satisfaction, the component-granular facts
/// grading consumes, and the UNCONSUMED set — extra keys a local un-bordered snapshot
/// carried that nothing classified. The unconsumed set is a recorded fact, never a third
/// verdict: `release_cut` is what refuses on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    pub satisfaction: Satisfaction,
    pub facts: Vec<ComponentFacts>,
    pub unconsumed: Vec<String>,
}

fn hist(mut v: Vec<Dtype>) -> Vec<(Dtype, u64)> {
    v.sort();
    let mut out: Vec<(Dtype, u64)> = Vec::new();
    for d in v {
        match out.last_mut() {
            Some((x, n)) if *x == d => *n += 1,
            _ => out.push((d, 1)),
        }
    }
    out
}

fn mismatch(code: Code, component: &str, key: &str, required: Side, present: Side) -> Satisfaction {
    Satisfaction::Mismatch(Box::new(Details {
        code,
        component: component.to_string(),
        key: key.to_string(),
        required,
        present,
    }))
}

/// The one deterministic comparison. Iteration is over the CODE side first, in canonical
/// (sorted) order, so "first disagreeing tensor" means the same tensor on every machine.
///
/// `known_markers` is DATA supplied by the caller, not a vocabulary compiled in here: the
/// moment this file carries a marker taxonomy it becomes a second classifier on the serve
/// path, which is the defect the border exists to prevent. The border owns the list
/// (`ingest::fingerprint::MARKER_SUFFIXES`); this applies a rule to what it is handed.
pub fn check_compatibility(
    code: &CodeTopology,
    header: &Header,
    closure: &Closure,
    pin: &[String],
    lane: Lane,
    known_markers: &[String],
) -> Result<Outcome> {
    let mut facts: Vec<ComponentFacts> = Vec::new();
    let mut unconsumed: Vec<String> = Vec::new();
    let mut verdict = Satisfaction::Satisfied;

    // -- the required side, in canonical order.
    for (comp, reqs) in &code.components {
        let present = header.components.iter().find(|(c, _)| c == comp);
        let tensors: &[(String, crate::header::Tensor)] = match present {
            Some((_, t)) => t.as_slice(),
            None => {
                if verdict.is_satisfied() {
                    verdict = mismatch(
                        Code::MISSING_COMPONENT,
                        comp,
                        &reqs[0].0,
                        Side {
                            dtype: Some(reqs[0].1.dtype),
                            shape: Some(reqs[0].1.shape.clone()),
                            accepts: Some(accepts_text(&reqs[0].1.accepts)),
                            ..Side::default()
                        },
                        Side::default(),
                    );
                }
                &[]
            }
        };

        let mut present_encodings: Vec<EncodingFact> = Vec::new();
        for (key, t) in tensors {
            match present_encodings
                .iter_mut()
                .find(|e| e.encoding == t.encoding)
            {
                Some(e) => {
                    e.keys += 1;
                    if !e.logical_dtypes.contains(&t.dtype) {
                        e.logical_dtypes.push(t.dtype);
                        e.logical_dtypes.sort();
                    }
                }
                None => {
                    let _ = key;
                    present_encodings.push(EncodingFact {
                        encoding: t.encoding.clone(),
                        keys: 1,
                        logical_dtypes: vec![t.dtype],
                        roles: t
                            .parts
                            .iter()
                            .map(|(n, p)| RoleFact {
                                role: n.clone(),
                                carrier: p.dtype,
                                rank: p.shape.len() as u64,
                                shape: p.shape.clone(),
                            })
                            .collect(),
                    });
                }
            }
        }
        present_encodings.sort_by(|a, b| a.encoding.cmp(&b.encoding));

        let mut required_accepts: Vec<String> = Vec::new();
        let mut any = false;
        for (_, r) in reqs {
            match &r.accepts {
                Accepts::Any => any = true,
                Accepts::OneOf(v) => required_accepts.extend(v.iter().cloned()),
            }
        }
        required_accepts.sort();
        required_accepts.dedup();

        facts.push(ComponentFacts {
            component: comp.clone(),
            required_keys: reqs.len() as u64,
            present_keys: tensors.len() as u64,
            required_dtypes: hist(reqs.iter().map(|(_, r)| r.dtype).collect()),
            present_dtypes: hist(tensors.iter().map(|(_, t)| t.dtype).collect()),
            required_encodings: if any || required_accepts.is_empty() {
                Accepts::Any
            } else {
                Accepts::OneOf(required_accepts)
            },
            present_encodings,
        });

        if tensors.is_empty() {
            continue;
        }

        for (key, req) in reqs {
            let want = Side {
                dtype: Some(req.dtype),
                shape: Some(req.shape.clone()),
                accepts: Some(accepts_text(&req.accepts)),
                encoding: None,
            };
            let t = match tensors.iter().find(|(k, _)| k == key) {
                Some((_, t)) => t,
                None => {
                    // Missing keys ALWAYS refuse, on every lane. Nothing downstream can
                    // invent a tensor the checkpoint does not carry.
                    if verdict.is_satisfied() {
                        verdict = mismatch(Code::MISSING_TENSOR, comp, key, want, Side::default());
                    }
                    continue;
                }
            };
            let got = Side {
                dtype: Some(t.dtype),
                shape: Some(t.shape.clone()),
                encoding: Some(t.encoding.clone()),
                accepts: None,
            };
            if !verdict.is_satisfied() {
                continue;
            }
            if t.shape != req.shape {
                verdict = mismatch(Code::SHAPE_MISMATCH, comp, key, want, got);
                continue;
            }
            if t.dtype != req.dtype {
                verdict = mismatch(Code::DTYPE_MISMATCH, comp, key, want, got);
                continue;
            }
            if !req.accepts.admits(&t.encoding) {
                verdict = mismatch(Code::ENCODING_MISMATCH, comp, key, want, got);
                continue;
            }
            // Encoding assignments against the PINNED registry. A spec absent from the
            // closure is not a checkpoint at all; a spec the local pin does not alias is
            // valid data whose execution is unsupported here, and offline the two reasons
            // for that ("unregistered" vs "newer than my pin") are indistinguishable.
            if closure.get(&t.encoding).is_none() {
                verdict = mismatch(Code::UNKNOWN_ENCODING, comp, key, want, got);
                continue;
            }
            if !pin.contains(&t.encoding) {
                verdict = mismatch(Code::UNKNOWN_TO_LOCAL_CAPABILITY, comp, key, want, got);
                continue;
            }
            let spec = closure.get(&t.encoding).unwrap();
            let want_roles: Vec<&str> = spec.roles.iter().map(|(n, _)| n.as_str()).collect();
            let got_roles: Vec<&str> = t.parts.iter().map(|(n, _)| n.as_str()).collect();
            if want_roles != got_roles {
                verdict = mismatch(Code::ROLE_SET_MISMATCH, comp, key, want, got);
            }
        }
    }

    // -- the present side: components and keys the code never asked for.
    for (comp, tensors) in &header.components {
        let reqs = code.components.iter().find(|(c, _)| c == comp);
        let known: &[(String, Requirement)] = match reqs {
            Some((_, r)) => r.as_slice(),
            None => {
                surplus(
                    &mut verdict,
                    &mut unconsumed,
                    lane,
                    known_markers,
                    Code::EXTRA_COMPONENT,
                    comp,
                    &tensors[0].0,
                    Side {
                        dtype: Some(tensors[0].1.dtype),
                        shape: Some(tensors[0].1.shape.clone()),
                        encoding: Some(tensors[0].1.encoding.clone()),
                        accepts: None,
                    },
                );
                continue;
            }
        };
        for (key, t) in tensors {
            if known.iter().any(|(k, _)| k == key) {
                continue;
            }
            surplus(
                &mut verdict,
                &mut unconsumed,
                lane,
                known_markers,
                Code::EXTRA_TENSOR,
                comp,
                key,
                Side {
                    dtype: Some(t.dtype),
                    shape: Some(t.shape.clone()),
                    encoding: Some(t.encoding.clone()),
                    accepts: None,
                },
            );
        }
    }
    unconsumed.sort();

    Ok(Outcome {
        satisfaction: verdict,
        facts,
        unconsumed,
    })
}

/// The direction asymmetry, in one place.
///
/// Canonical lane: the border classified every key, so a surplus is drift or corruption
/// evidence and refuses naming it. Local un-bordered lane: a KNOWN marker is a fact on the
/// wrong side of the boundary — it should have been transcribed into an encoding role at
/// the border and refuses here; an UNKNOWN extra is warn-ignore-confess, recorded in the
/// unconsumed set so a release cut can refuse on it later.
#[allow(clippy::too_many_arguments)]
fn surplus(
    verdict: &mut Satisfaction,
    unconsumed: &mut Vec<String>,
    lane: Lane,
    known_markers: &[String],
    code: Code,
    comp: &str,
    key: &str,
    got: Side,
) {
    let is_marker = known_markers.iter().any(|m| key.ends_with(m.as_str()));
    if lane == Lane::LocalUnbordered && !is_marker {
        unconsumed.push(format!("{comp}/{key}"));
        return;
    }
    if verdict.is_satisfied() {
        *verdict = mismatch(code, comp, key, Side::default(), got);
    }
}

fn accepts_text(a: &Accepts) -> String {
    match a {
        Accepts::Any => "any".to_string(),
        Accepts::OneOf(v) => v.join(","),
    }
}

// ---------------------------------------------------------------- the report document

/// The canonical, digest-fenced result. This is what crosses the hermetic verifier's
/// bounded document interface, and what the hub grades. It carries facts and a
/// two-valued satisfaction; a `verdict`, `grade` or `remedy` field here would be grading
/// below the hub, which the fence forbids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompatibilityReport {
    pub contract: String,
    pub header: String,
    pub topology_required: String,
    pub topology_present: String,
    pub lane: Lane,
    pub satisfaction: Satisfaction,
    pub facts: Vec<ComponentFacts>,
    pub unconsumed: Vec<String>,
}

impl CompatibilityReport {
    pub fn of(
        code: &CodeTopology,
        header: &Header,
        lane: Lane,
        o: &Outcome,
    ) -> Result<CompatibilityReport> {
        Ok(CompatibilityReport {
            contract: code.contract.clone(),
            header: header.object_id()?,
            topology_required: code.topology_digest(),
            topology_present: header.topology_digest(),
            lane,
            satisfaction: o.satisfaction.clone(),
            facts: o.facts.clone(),
            unconsumed: o.unconsumed.clone(),
        })
    }

    /// A release cut takes only a snapshot whose extras were all classified. An
    /// un-bordered snapshot that quietly carried unknown keys never becomes a catalog
    /// release: warn-ignore-confess is a LOCAL affordance, not a publication policy.
    pub fn release_cut(&self) -> Result<()> {
        if !self.unconsumed.is_empty() {
            return refuse(
                Code::UNCLASSIFIED_EXTRAS,
                format!(
                    "{} unclassified extra key(s), first {:?}: an un-bordered snapshot with \
                     unconsumed keys never cuts into a catalog release",
                    self.unconsumed.len(),
                    self.unconsumed[0]
                ),
            );
        }
        match &self.satisfaction {
            Satisfaction::Satisfied => Ok(()),
            Satisfaction::Mismatch(d) => refuse(
                d.code,
                format!(
                    "{}/{}: the cut candidate does not satisfy its contract",
                    d.component, d.key
                ),
            ),
        }
    }
}

fn dtype_hist_value(h: &[(Dtype, u64)]) -> Value {
    Value::map(
        h.iter()
            .map(|(d, n)| (d.name().to_string(), Value::uint(*n)))
            .collect(),
    )
}

fn dtype_hist_from(what: &str, v: &Value) -> Result<Vec<(Dtype, u64)>> {
    let mut out = Vec::new();
    for (k, x) in as_obj("ComponentFacts", what, v)? {
        out.push((Dtype::parse(k)?, crate::canon::as_uint(what, k, x)?));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

impl Doc for CompatibilityReport {
    const FORMAT: &'static str = "tensorfs.compatibility_report/1";
    const MAX_BYTES: usize = limits::DOC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("CompatibilityReport", v)?;
        check_format(
            "CompatibilityReport",
            as_str("CompatibilityReport", "format", f.req("format")?)?,
            Self::FORMAT,
        )?;
        let contract = prefixed("report.contract", f.req_str("contract")?)?;
        let mut facts = Vec::new();
        for fv in as_arr("CompatibilityReport", "facts", f.req("facts")?)? {
            let mut cf = Fields::new("ComponentFacts", fv)?;
            let component = cf.req_str("component")?.to_string();
            let present_dtypes = dtype_hist_from("present_dtypes", cf.req("present_dtypes")?)?;
            let mut present_encodings = Vec::new();
            for ev in as_arr(
                "ComponentFacts",
                "present_encodings",
                cf.req("present_encodings")?,
            )? {
                let mut ef = Fields::new("EncodingFact", ev)?;
                let encoding = prefixed("EncodingFact.encoding", ef.req_str("encoding")?)?;
                let keys = ef.req_uint("keys")?;
                let mut logical_dtypes = Vec::new();
                for d in as_arr("EncodingFact", "logical_dtypes", ef.req("logical_dtypes")?)? {
                    logical_dtypes.push(Dtype::parse(as_str("EncodingFact", "dtype", d)?)?);
                }
                let mut roles = Vec::new();
                for rv in as_arr("EncodingFact", "roles", ef.req("roles")?)? {
                    let mut rf = Fields::new("RoleFact", rv)?;
                    let carrier = Dtype::parse(as_str("RoleFact", "carrier", rf.req("carrier")?)?)?;
                    let rank = rf.req_uint("rank")?;
                    let role = rf.req_str("role")?.to_string();
                    let shape = as_arr("RoleFact", "shape", rf.req("shape")?)?
                        .iter()
                        .map(|d| crate::canon::as_uint("RoleFact", "shape", d))
                        .collect::<Result<Vec<u64>>>()?;
                    rf.done()?;
                    roles.push(RoleFact {
                        role,
                        carrier,
                        rank,
                        shape,
                    });
                }
                ef.done()?;
                present_encodings.push(EncodingFact {
                    encoding,
                    keys,
                    logical_dtypes,
                    roles,
                });
            }
            let present_keys = cf.req_uint("present_keys")?;
            let required_dtypes = dtype_hist_from("required_dtypes", cf.req("required_dtypes")?)?;
            let required_encodings =
                Accepts::from_value("ComponentFacts", cf.req("required_encodings")?)?;
            let required_keys = cf.req_uint("required_keys")?;
            cf.done()?;
            facts.push(ComponentFacts {
                component,
                required_keys,
                present_keys,
                required_dtypes,
                present_dtypes,
                required_encodings,
                present_encodings,
            });
        }
        let header = prefixed("report.header", f.req_str("header")?)?;
        let lane = Lane::parse(f.req_str("lane")?)?;
        let satisfaction = {
            let mut sf = Fields::new("satisfaction", f.req("satisfaction")?)?;
            let t = as_str("satisfaction", "t", sf.req("t")?)?.to_string();
            match t.as_str() {
                "satisfied" => {
                    sf.done()?;
                    Satisfaction::Satisfied
                }
                "mismatch" => {
                    let code = as_str("satisfaction", "code", sf.req("code")?)?.to_string();
                    let component = sf.req_str("component")?.to_string();
                    let key = sf.req_str("key")?.to_string();
                    let present = Side::from_value(sf.req("present")?)?;
                    let required = Side::from_value(sf.req("required")?)?;
                    sf.done()?;
                    let code = match Code::ALL.iter().find(|c| c.as_str() == code) {
                        Some(c) => *c,
                        None => {
                            return refuse(
                                Code::UNKNOWN_FIELD,
                                format!("mismatch code {code:?} is outside the closed code set"),
                            )
                        }
                    };
                    Satisfaction::Mismatch(Box::new(Details {
                        code,
                        component,
                        key,
                        required,
                        present,
                    }))
                }
                o => {
                    return refuse(
                        Code::UNKNOWN_FIELD,
                        format!("satisfaction {o:?} is outside {{satisfied, mismatch}} — the ONE runtime vocabulary"),
                    )
                }
            }
        };
        let topology_present = prefixed("report.topology_present", f.req_str("topology_present")?)?;
        let topology_required =
            prefixed("report.topology_required", f.req_str("topology_required")?)?;
        let mut unconsumed = Vec::new();
        for u in as_arr("CompatibilityReport", "unconsumed", f.req("unconsumed")?)? {
            unconsumed.push(as_str("unconsumed", "key", u)?.to_string());
        }
        f.done()?;
        Ok(CompatibilityReport {
            contract,
            header,
            topology_required,
            topology_present,
            lane,
            satisfaction,
            facts,
            unconsumed,
        })
    }

    fn to_value(&self) -> Value {
        let satisfaction = match &self.satisfaction {
            Satisfaction::Satisfied => Value::obj(vec![("t", Value::str("satisfied"))]),
            Satisfaction::Mismatch(d) => Value::obj(vec![
                ("code", Value::str(d.code.as_str())),
                ("component", Value::str(d.component.clone())),
                ("key", Value::str(d.key.clone())),
                ("present", d.present.to_value()),
                ("required", d.required.to_value()),
                ("t", Value::str("mismatch")),
            ]),
        };
        Value::obj(vec![
            ("contract", Value::str(self.contract.clone())),
            (
                "facts",
                Value::arr(
                    self.facts
                        .iter()
                        .map(|c| {
                            Value::obj(vec![
                                ("component", Value::str(c.component.clone())),
                                ("present_dtypes", dtype_hist_value(&c.present_dtypes)),
                                (
                                    "present_encodings",
                                    Value::arr(
                                        c.present_encodings
                                            .iter()
                                            .map(|e| {
                                                Value::obj(vec![
                                                    ("encoding", Value::str(e.encoding.clone())),
                                                    ("keys", Value::uint(e.keys)),
                                                    (
                                                        "logical_dtypes",
                                                        Value::arr(
                                                            e.logical_dtypes
                                                                .iter()
                                                                .map(|d| Value::str(d.name()))
                                                                .collect(),
                                                        ),
                                                    ),
                                                    (
                                                        "roles",
                                                        Value::arr(
                                                            e.roles
                                                                .iter()
                                                                .map(|r| {
                                                                    Value::obj(vec![
                                                                        (
                                                                            "carrier",
                                                                            Value::str(
                                                                                r.carrier.name(),
                                                                            ),
                                                                        ),
                                                                        (
                                                                            "rank",
                                                                            Value::uint(r.rank),
                                                                        ),
                                                                        (
                                                                            "role",
                                                                            Value::str(
                                                                                r.role.clone(),
                                                                            ),
                                                                        ),
                                                                        (
                                                                            "shape",
                                                                            Value::arr(
                                                                                r.shape
                                                                                    .iter()
                                                                                    .map(|d| {
                                                                                        Value::uint(
                                                                                            *d,
                                                                                        )
                                                                                    })
                                                                                    .collect(),
                                                                            ),
                                                                        ),
                                                                    ])
                                                                })
                                                                .collect(),
                                                        ),
                                                    ),
                                                ])
                                            })
                                            .collect(),
                                    ),
                                ),
                                ("present_keys", Value::uint(c.present_keys)),
                                ("required_dtypes", dtype_hist_value(&c.required_dtypes)),
                                ("required_encodings", c.required_encodings.to_value()),
                                ("required_keys", Value::uint(c.required_keys)),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("format", Value::str(Self::FORMAT)),
            ("header", Value::str(self.header.clone())),
            ("lane", Value::str(self.lane.name())),
            ("satisfaction", satisfaction),
            (
                "topology_present",
                Value::str(self.topology_present.clone()),
            ),
            (
                "topology_required",
                Value::str(self.topology_required.clone()),
            ),
            (
                "unconsumed",
                Value::arr(
                    self.unconsumed
                        .iter()
                        .map(|u| Value::str(u.clone()))
                        .collect(),
                ),
            ),
        ])
    }
}

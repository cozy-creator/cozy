//! `fit` — the ONE code <-> checkpoint matching function (tfs-042, design
//! `tracker/cross-cutting/model-code-fit.md` §3).
//!
//! Two inputs, one answer. The code side is a `CodeTopology`: what the package's factory
//! constructs from the checkpoint's own config document, censused on the meta device —
//! `component -> key -> {dtype, shape, accepts, pinned}` plus the constructed component
//! list. The checkpoint side is its `Header`. `fit` returns exactly one `Fit` code with the
//! text a human reads; it plans nothing, grades nothing and names no remedy.
//!
//! Rules, in order (§3): (1) every constructed component exists; components the checkpoint
//! carries that the code does not construct are ignored (D6); (2) every constructed key
//! exists at the same logical shape; (3) a key whose destination PINS a dtype must be
//! supplied at that dtype or through a lossless widening — otherwise dtype is delivery's
//! business (D4); (4) on `canonical` custody a stored key inside a constructed component
//! that the code does not consume refuses, on `local` it is listed and ignored (D5); (5)
//! every stored encoding must be admitted by the row's `accepts`; (6) with a device, every
//! encoded leaf must have a qualification observation for that device class.
//!
//! The walk itself (`check_compatibility`) records EVERY fact at `Depth::Full`; `fit` reads
//! the first fact in RULE order off it, and the grade plane (`grade.rs`, search only — D11)
//! ranks the whole set. One walk, two readers, so the two cannot disagree about a key.

use crate::canon::{as_arr, as_obj, as_str, Fields, Value};
use crate::capability::CapabilityRecords;
use crate::dtype::Dtype;
use crate::err::{refuse, Code, Result};
use crate::header::{Closure, Header};
use crate::ids::{ascii_name, prefixed, Plain};
use crate::limits;

// ---------------------------------------------------------------- the required side

/// Which encodings a code destination will accept for one tensor. Closed and REQUIRED —
/// an absent field defaulting to "anything" is how a lane silently widens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Accepts {
    /// Any encoding the pinned registry knows (`encoded_leaves="accept"`).
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
///
/// `pinned` says the destination fixes its dtype (an integer buffer, an fp32 island): the
/// checkpoint must supply that dtype or one that widens into it losslessly. An unpinned
/// dtype is the constructed default and is delivery's business, not fit's (D4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Requirement {
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub accepts: Accepts,
    pub pinned: bool,
}

/// The code side: the meta-device census of `factory(config)`, projected to TensorFS's
/// vocabulary. `contract` is the digest of the census this was projected from — recorded,
/// never parsed. `constructed_components` is the same set as `components`' keys, in the
/// order the factory constructed them (D9); the map is canonical (sorted) and the list is
/// where the order survives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeTopology {
    pub contract: String,
    pub components: Vec<(String, Vec<(String, Requirement)>)>,
    pub constructed_components: Vec<String>,
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

    pub fn requirement(&self, component: &str, key: &str) -> Option<&Requirement> {
        self.components
            .iter()
            .find(|(c, _)| c == component)
            .and_then(|(_, ks)| ks.iter().find(|(k, _)| k == key))
            .map(|(_, r)| r)
    }

    pub fn tensor_schema_digest(&self) -> String {
        crate::header::tensor_schema_digest_of(&crate::header::tensor_schema_value(
            self.tensors().map(|(component, key, requirement)| {
                (
                    component,
                    key,
                    requirement.dtype,
                    requirement.shape.as_slice(),
                )
            }),
        ))
    }
}

impl Plain for CodeTopology {
    const MAX_BYTES: usize = limits::DOC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("CodeTopology", v)?;
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
                let pinned = match rf.opt("pinned") {
                    None => false,
                    Some(Value::Bool(true)) => true,
                    Some(_) => {
                        return refuse(
                            Code::WRONG_TYPE,
                            format!("{name}/{key}: pinned is `true` or absent"),
                        )
                    }
                };
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
                        pinned,
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
        let mut constructed_components = Vec::new();
        for x in as_arr(
            "CodeTopology",
            "constructed_components",
            f.req("constructed_components")?,
        )? {
            constructed_components
                .push(as_str("CodeTopology", "constructed_components", x)?.to_string());
        }
        f.done()?;
        if components.is_empty() || components.len() > limits::MAX_COMPONENTS {
            return refuse(Code::EMPTY_COMPONENTS, "component count out of bounds");
        }
        let mut listed = constructed_components.clone();
        listed.sort();
        listed.dedup();
        let named: Vec<&String> = components.iter().map(|(c, _)| c).collect();
        if listed.len() != constructed_components.len() || listed.iter().ne(named.iter().copied()) {
            return refuse(
                Code::MISSING_COMPONENT,
                format!(
                    "constructed_components {constructed_components:?} must name exactly the \
                     components the topology carries ({named:?}), once each"
                ),
            );
        }
        Ok(CodeTopology {
            contract,
            components,
            constructed_components,
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
                                            let mut row = vec![
                                                ("accepts", r.accepts.to_value()),
                                                ("dtype", Value::str(r.dtype.name())),
                                            ];
                                            if r.pinned {
                                                row.push(("pinned", Value::Bool(true)));
                                            }
                                            row.push((
                                                "shape",
                                                Value::arr(
                                                    r.shape
                                                        .iter()
                                                        .map(|d| Value::uint(*d))
                                                        .collect(),
                                                ),
                                            ));
                                            (k.clone(), Value::obj(row))
                                        })
                                        .collect(),
                                ),
                            )
                        })
                        .collect(),
                ),
            ),
            (
                "constructed_components",
                Value::arr(
                    self.constructed_components
                        .iter()
                        .map(|c| Value::str(c.clone()))
                        .collect(),
                ),
            ),
            ("contract", Value::str(self.contract.clone())),
        ])
    }
}

// ---------------------------------------------------------------- custody

/// Where the checkpoint came from, which is the ONLY thing that changes the extra-key rule
/// (D5). Missing keys refuse on both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Custody {
    /// The border classified every key. A stored key the code does not consume is drift.
    Canonical,
    /// A local manifest that never crossed the border: extras are listed and ignored.
    Local,
}

impl Custody {
    pub fn name(self) -> &'static str {
        match self {
            Custody::Canonical => "canonical",
            Custody::Local => "local",
        }
    }
    pub fn parse(s: &str) -> Result<Custody> {
        match s {
            "canonical" => Ok(Custody::Canonical),
            "local" => Ok(Custody::Local),
            o => refuse(
                Code::UNKNOWN_FIELD,
                format!("custody {o:?} is outside {{canonical, local}}"),
            ),
        }
    }
}

// ---------------------------------------------------------------- the walk

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

/// ONE disagreeing tensor, both sides. A fact, never a verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Details {
    pub code: Code,
    pub component: String,
    pub key: String,
    pub required: Side,
    pub present: Side,
}

/// How far a walk goes. `fit` always walks `Full`; `FirstMismatch` is the grade plane's
/// cheap re-check, and the depth travels with a report so a ranking can never mistake a
/// stopped walk for a complete one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Depth {
    FirstMismatch,
    Full,
}

impl Depth {
    pub fn name(self) -> &'static str {
        match self {
            Depth::FirstMismatch => "first-mismatch",
            Depth::Full => "full",
        }
    }

    pub fn parse(s: &str) -> Result<Depth> {
        match s {
            "first-mismatch" => Ok(Depth::FirstMismatch),
            "full" => Ok(Depth::Full),
            o => refuse(
                Code::UNKNOWN_FIELD,
                format!("depth {o:?} is outside {{first-mismatch, full}}"),
            ),
        }
    }
}

/// The MEASURED geometry of one role, read out of the header. Never inferred from a dtype
/// spelling: scale rank is a per-producer fact (decisions #244).
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

/// Everything one walk produced: the fact set, the component-granular facts grading
/// consumes, and the UNCONSUMED set — stored keys inside constructed components that the
/// code does not consume, on local custody (listed, ignored).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    pub mismatches: Vec<Details>,
    pub depth: Depth,
    pub facts: Vec<ComponentFacts>,
    pub unconsumed: Vec<String>,
}

impl Outcome {
    pub fn is_satisfied(&self) -> bool {
        self.mismatches.is_empty()
    }
}

/// A name -> position index over one component's keys, sorted once and binary-searched.
fn index_of<T>(keys: &[(String, T)]) -> Vec<(&str, usize)> {
    let mut ix: Vec<(&str, usize)> = keys
        .iter()
        .enumerate()
        .map(|(i, (k, _))| (k.as_str(), i))
        .collect();
    ix.sort_unstable_by_key(|(k, _)| *k);
    ix
}

fn lookup(ix: &[(&str, usize)], key: &str) -> Option<usize> {
    ix.binary_search_by(|(k, _)| (*k).cmp(key))
        .ok()
        .map(|i| ix[i].1)
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

fn mismatch(code: Code, component: &str, key: &str, required: Side, present: Side) -> Details {
    Details {
        code,
        component: component.to_string(),
        key: key.to_string(),
        required,
        present,
    }
}

/// The one deterministic walk. Iteration is over the CODE side first, in canonical (sorted)
/// order, so "first disagreeing tensor" means the same tensor on every machine. Components
/// the checkpoint carries that the code does not construct are never visited (D6).
pub fn check_compatibility(
    code: &CodeTopology,
    header: &Header,
    closure: &Closure,
    pin: &[String],
    custody: Custody,
    depth: Depth,
) -> Result<Outcome> {
    let mut facts: Vec<ComponentFacts> = Vec::new();
    let mut unconsumed: Vec<String> = Vec::new();
    let mut mismatches: Vec<Details> = Vec::new();
    macro_rules! stop {
        () => {
            depth == Depth::FirstMismatch && !mismatches.is_empty()
        };
    }

    for (comp, reqs) in &code.components {
        let present = header.components.iter().find(|(c, _)| c == comp);
        let tensors: &[(String, crate::header::Tensor)] = match present {
            Some((_, t)) => t.as_slice(),
            None => {
                if !stop!() {
                    mismatches.push(mismatch(
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
                    ));
                }
                &[]
            }
        };

        let mut present_encodings: Vec<EncodingFact> = Vec::new();
        for (_, t) in tensors {
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
                None => present_encodings.push(EncodingFact {
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
                }),
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

        let present_index = index_of(tensors);
        for (key, req) in reqs {
            if stop!() {
                break;
            }
            let want = || Side {
                dtype: Some(req.dtype),
                shape: Some(req.shape.clone()),
                accepts: Some(accepts_text(&req.accepts)),
                encoding: None,
            };
            let t = match lookup(&present_index, key).map(|i| &tensors[i].1) {
                Some(t) => t,
                None => {
                    mismatches.push(mismatch(
                        Code::MISSING_TENSOR,
                        comp,
                        key,
                        want(),
                        Side::default(),
                    ));
                    continue;
                }
            };
            let got = || Side {
                dtype: Some(t.dtype),
                shape: Some(t.shape.clone()),
                encoding: Some(t.encoding.clone()),
                accepts: None,
            };
            if t.shape != req.shape {
                mismatches.push(mismatch(Code::SHAPE_MISMATCH, comp, key, want(), got()));
                continue;
            }
            if t.dtype != req.dtype {
                mismatches.push(mismatch(Code::DTYPE_MISMATCH, comp, key, want(), got()));
                continue;
            }
            if !req.accepts.admits(&t.encoding) {
                mismatches.push(mismatch(Code::ENCODING_MISMATCH, comp, key, want(), got()));
                continue;
            }
            // Encoding assignments against the PINNED registry. cozytensors/1 carries its
            // specs inline; the optional closure remains current for standalone layout
            // inputs. A spec absent from both is not a checkpoint at all; a spec the local
            // pin does not alias is valid data whose execution is unsupported here.
            let spec = closure.get(&t.encoding).or_else(|| {
                header
                    .encodings
                    .iter()
                    .find(|candidate| candidate.object_id() == t.encoding)
            });
            let Some(spec) = spec else {
                mismatches.push(mismatch(Code::UNKNOWN_ENCODING, comp, key, want(), got()));
                continue;
            };
            if !pin.contains(&t.encoding) {
                mismatches.push(mismatch(
                    Code::UNKNOWN_TO_LOCAL_CAPABILITY,
                    comp,
                    key,
                    want(),
                    got(),
                ));
                continue;
            }
            let want_roles: Vec<&str> = spec.roles.iter().map(|(n, _)| n.as_str()).collect();
            let got_roles: Vec<&str> = t.parts.iter().map(|(n, _)| n.as_str()).collect();
            if want_roles != got_roles {
                mismatches.push(mismatch(Code::ROLE_SET_MISMATCH, comp, key, want(), got()));
            }
        }

        // The present side of THIS component: stored keys the code never asked for. On
        // canonical custody a surplus is drift and refuses by name; on local custody it is
        // listed and ignored, at every depth.
        let known_index = index_of(reqs);
        for (key, t) in tensors {
            if lookup(&known_index, key).is_some() {
                continue;
            }
            if custody == Custody::Local {
                unconsumed.push(format!("{comp}/{key}"));
                continue;
            }
            if stop!() {
                continue;
            }
            mismatches.push(mismatch(
                Code::EXTRA_TENSOR,
                comp,
                key,
                Side::default(),
                Side {
                    dtype: Some(t.dtype),
                    shape: Some(t.shape.clone()),
                    encoding: Some(t.encoding.clone()),
                    accepts: None,
                },
            ));
        }
    }
    unconsumed.sort();

    Ok(Outcome {
        mismatches,
        depth,
        facts,
        unconsumed,
    })
}

fn accepts_text(a: &Accepts) -> String {
    match a {
        Accepts::Any => "any".to_string(),
        Accepts::OneOf(v) => v.join(","),
    }
}

// ---------------------------------------------------------------- fit

/// A dtype that widens into another with every value preserved: format containment, the
/// same fact `grade::ConversionRegistration` proves over the complete 16-bit domain.
pub fn lossless_widening(from: Dtype, to: Dtype) -> bool {
    use Dtype::*;
    if from == to {
        return true;
    }
    matches!(
        (from, to),
        (F8E4M3FN | F8E5M2, F16 | Bf16 | F32 | F64)
            | (F16 | Bf16, F32 | F64)
            | (F32, F64)
            | (I8, I16 | I32 | I64)
            | (U8, I16 | I32 | I64)
            | (I16, I32 | I64)
            | (I32, I64)
    )
}

/// How many constructed keys take each delivery route on an `ok` fit. Informational: the
/// route itself is the plan's business.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Routes {
    /// Stored plain: the bytes are the logical tensor.
    pub verbatim: u64,
    /// Stored encoded, admitted by an explicit accepts list: decoded to float on the way in.
    pub decoded_float: u64,
    /// Stored encoded under `accepts: any`: an encoded linear op is swapped in under the code.
    pub encoded_gemm: u64,
}

/// Exactly one of the eight codes of design §3. `text()` is what a human reads, prefixed
/// by the caller with `<package>@<release> <slot> does not fit <org/model>@<release>/<lane>:`.
/// There is no grade (D11).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fit {
    Ok {
        routes: Routes,
        /// `component/key` the checkpoint stores and the code does not consume, on local
        /// custody. Empty on canonical custody by construction.
        ignored: Vec<String>,
        /// The device class the encoded leaves were qualified on, when one was asked.
        device: Option<String>,
    },
    ComponentMissing {
        component: String,
        present: Vec<String>,
    },
    TensorMissing {
        component: String,
        key: String,
    },
    ShapeMismatch {
        component: String,
        key: String,
        code: Vec<u64>,
        checkpoint: Vec<u64>,
    },
    DtypePinnedMismatch {
        component: String,
        key: String,
        pinned: Dtype,
        stored: Dtype,
    },
    ExtraTensor {
        component: String,
        key: String,
    },
    EncodingUnsupported {
        component: String,
        key: String,
        encoding: String,
        accepts: Accepts,
    },
    EncodingUnqualified {
        encoding: String,
        device: String,
        qualified_on: Vec<String>,
    },
}

impl Fit {
    pub fn code(&self) -> &'static str {
        match self {
            Fit::Ok { .. } => "ok",
            Fit::ComponentMissing { .. } => "component_missing",
            Fit::TensorMissing { .. } => "tensor_missing",
            Fit::ShapeMismatch { .. } => "shape_mismatch",
            Fit::DtypePinnedMismatch { .. } => "dtype_pinned_mismatch",
            Fit::ExtraTensor { .. } => "extra_tensor",
            Fit::EncodingUnsupported { .. } => "encoding_unsupported",
            Fit::EncodingUnqualified { .. } => "encoding_unqualified",
        }
    }

    pub const CODES: [&'static str; 8] = [
        "ok",
        "component_missing",
        "tensor_missing",
        "shape_mismatch",
        "dtype_pinned_mismatch",
        "extra_tensor",
        "encoding_unsupported",
        "encoding_unqualified",
    ];

    pub fn is_ok(&self) -> bool {
        matches!(self, Fit::Ok { .. })
    }

    /// Process exit status for `tfs fit`: 0 on `ok`, and one distinct code per refusal so a
    /// caller can branch without parsing.
    pub fn exit_code(&self) -> u8 {
        match self {
            Fit::Ok { .. } => 0,
            Fit::ComponentMissing { .. } => 10,
            Fit::TensorMissing { .. } => 11,
            Fit::ShapeMismatch { .. } => 12,
            Fit::DtypePinnedMismatch { .. } => 13,
            Fit::ExtraTensor { .. } => 14,
            Fit::EncodingUnsupported { .. } => 15,
            Fit::EncodingUnqualified { .. } => 16,
        }
    }

    /// The detail text of design §3, ASCII so it can ride a canonical document.
    pub fn text(&self) -> String {
        match self {
            Fit::Ok {
                routes,
                ignored,
                device,
            } => {
                let mut s = format!(
                    "fits (routes: verbatim x{}, decoded_float x{}, encoded_gemm x{}",
                    routes.verbatim, routes.decoded_float, routes.encoded_gemm
                );
                if let Some(d) = device {
                    let encoded = routes.decoded_float + routes.encoded_gemm;
                    if encoded == 0 {
                        s.push_str(&format!("; nothing to qualify on {d}"));
                    } else {
                        s.push_str(&format!(
                            "; {encoded} encoded {} qualified on {d}",
                            if encoded == 1 { "leaf" } else { "leaves" }
                        ));
                    }
                }
                if !ignored.is_empty() {
                    s.push_str(&format!(
                        "; {} stored key(s) not constructed, ignored on local custody",
                        ignored.len()
                    ));
                }
                s.push(')');
                s
            }
            Fit::ComponentMissing { component, present } => format!(
                "the code constructs {component}; the checkpoint has {}",
                present.join(", ")
            ),
            Fit::TensorMissing { component, key } => format!(
                "{component}.{key}: the code constructs it; the checkpoint does not supply it"
            ),
            Fit::ShapeMismatch {
                component,
                key,
                code,
                checkpoint,
            } => format!(
                "{component}.{key}: code {}, checkpoint {}",
                dims(code),
                dims(checkpoint)
            ),
            Fit::DtypePinnedMismatch {
                component,
                key,
                pinned,
                stored,
            } => format!(
                "{component}.{key} pins {}; stored {}, no lossless mapping registered",
                pinned.name(),
                stored.name()
            ),
            Fit::ExtraTensor { component, key } => format!(
                "{component}.{key}: stored but not constructed, on a canonical checkpoint \
                 (conversion drift)"
            ),
            Fit::EncodingUnsupported {
                component,
                key,
                encoding,
                accepts,
            } => {
                let admitted = match accepts {
                    Accepts::Any => "any registered encoding".to_string(),
                    Accepts::OneOf(v) => {
                        v.iter().map(|d| alias_of(d)).collect::<Vec<_>>().join(", ")
                    }
                };
                let leaves = match accepts {
                    Accepts::OneOf(v) if v.iter().all(|d| d == &plain_digest()) => {
                        " (encoded_leaves=refuse)"
                    }
                    _ => "",
                };
                format!(
                    "{component}.{key} is stored {}; this package accepts {admitted} only{leaves}",
                    alias_of(encoding)
                )
            }
            Fit::EncodingUnqualified {
                encoding,
                device,
                qualified_on,
            } => format!(
                "{} has no qualification observation on {device} (qualified on: {})",
                alias_of(encoding),
                if qualified_on.is_empty() {
                    "nothing".to_string()
                } else {
                    qualified_on.join(", ")
                }
            ),
        }
    }
}

fn dims(shape: &[u64]) -> String {
    format!(
        "[{}]",
        shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// The registry alias of a spec digest (display), or the digest when nothing aliases it.
pub fn alias_of(digest: &str) -> String {
    crate::registry::seeds()
        .iter()
        .find(|s| s.spec.object_id() == digest)
        .map(|s| s.alias.to_string())
        .unwrap_or_else(|| digest.to_string())
}

fn plain_digest() -> String {
    crate::registry::seeds()
        .into_iter()
        .find(|s| s.alias == "plain/1")
        .expect("plain/1 is entry zero of the registry")
        .spec
        .object_id()
}

/// `fit(code, checkpoint, custody, device, observations) -> Fit` — design §3.
///
/// `observations` widens the compiled-in qualification records (the reference decoders on
/// cpu) with what a component holding a card measured; `device` is the class the caller
/// will execute on. Without a device rule (6) is not asked. Plain storage is verbatim and
/// has nothing to qualify.
pub fn fit(
    code: &CodeTopology,
    header: &Header,
    custody: Custody,
    device: Option<&str>,
    observations: Option<&CapabilityRecords>,
) -> Result<Fit> {
    // One registry walk per fit: `seeds()` re-parses the mined vector fixtures each call.
    let seeds = crate::registry::seeds();
    let plain = seeds
        .iter()
        .find(|s| s.alias == "plain/1")
        .expect("plain/1 is entry zero of the registry")
        .spec
        .object_id();
    let mut pin: Vec<String> = seeds.iter().map(|s| s.spec.object_id()).collect();
    pin.extend(header.encodings.iter().map(|s| s.object_id()));
    let out = check_compatibility(
        code,
        header,
        &Closure::default(),
        &pin,
        custody,
        Depth::Full,
    )?;

    // Rule (1).
    if let Some(d) = out
        .mismatches
        .iter()
        .find(|d| d.code == Code::MISSING_COMPONENT)
    {
        return Ok(Fit::ComponentMissing {
            component: d.component.clone(),
            present: header.components.iter().map(|(c, _)| c.clone()).collect(),
        });
    }
    // Rule (2).
    if let Some(d) = out
        .mismatches
        .iter()
        .find(|d| matches!(d.code, Code::MISSING_TENSOR | Code::SHAPE_MISMATCH))
    {
        return Ok(match d.code {
            Code::MISSING_TENSOR => Fit::TensorMissing {
                component: d.component.clone(),
                key: d.key.clone(),
            },
            _ => Fit::ShapeMismatch {
                component: d.component.clone(),
                key: d.key.clone(),
                code: d.required.shape.clone().unwrap_or_default(),
                checkpoint: d.present.shape.clone().unwrap_or_default(),
            },
        });
    }
    // Rule (3): dtype only through pinned destinations.
    for d in out
        .mismatches
        .iter()
        .filter(|d| d.code == Code::DTYPE_MISMATCH)
    {
        let (Some(pinned), Some(stored)) = (d.required.dtype, d.present.dtype) else {
            continue;
        };
        let is_pinned = code
            .requirement(&d.component, &d.key)
            .is_some_and(|r| r.pinned);
        if is_pinned && !lossless_widening(stored, pinned) {
            return Ok(Fit::DtypePinnedMismatch {
                component: d.component.clone(),
                key: d.key.clone(),
                pinned,
                stored,
            });
        }
    }
    // Rule (4).
    if let Some(d) = out.mismatches.iter().find(|d| d.code == Code::EXTRA_TENSOR) {
        return Ok(Fit::ExtraTensor {
            component: d.component.clone(),
            key: d.key.clone(),
        });
    }
    // Rule (5). A spec this build does not register is unsupported here for the same
    // reason a refused one is: nothing on this side can execute it.
    if let Some(d) = out.mismatches.iter().find(|d| {
        matches!(
            d.code,
            Code::ENCODING_MISMATCH | Code::UNKNOWN_TO_LOCAL_CAPABILITY
        )
    }) {
        let accepts = code
            .requirement(&d.component, &d.key)
            .map(|r| r.accepts.clone())
            .unwrap_or(Accepts::Any);
        return Ok(Fit::EncodingUnsupported {
            component: d.component.clone(),
            key: d.key.clone(),
            encoding: d.present.encoding.clone().unwrap_or_default(),
            accepts,
        });
    }
    // Anything else the walk recorded is a malformed checkpoint, not a fit answer.
    if let Some(d) = out.mismatches.first() {
        return refuse(
            d.code,
            format!(
                "{}/{}: the checkpoint header is not a fit input ({})",
                d.component,
                d.key,
                d.code.as_str()
            ),
        );
    }

    // Routes, and rule (6) over the encoded leaves the code actually consumes.
    let mut routes = Routes::default();
    let mut encoded: Vec<String> = Vec::new();
    for (comp, key, req) in code.tensors() {
        let stored = header
            .components
            .iter()
            .find(|(c, _)| c == comp)
            .and_then(|(_, ts)| ts.iter().find(|(k, _)| k == key))
            .map(|(_, t)| &t.encoding)
            .expect("rules (1) and (2) proved every constructed key present");
        if *stored == plain {
            routes.verbatim += 1;
            continue;
        }
        match req.accepts {
            Accepts::Any => routes.encoded_gemm += 1,
            Accepts::OneOf(_) => routes.decoded_float += 1,
        }
        if !encoded.contains(stored) {
            encoded.push(stored.clone());
        }
    }
    if let Some(device) = device {
        let mut records = crate::registry::capability_records();
        if let Some(observed) = observations {
            records = records.with_observed(observed.clone());
        }
        encoded.sort();
        for e in &encoded {
            if records.admit(e, device).is_err() {
                let mut qualified_on: Vec<String> = records
                    .records
                    .iter()
                    .filter(|r| &r.encoding == e)
                    .map(|r| r.device.clone())
                    .collect();
                qualified_on.sort();
                qualified_on.dedup();
                return Ok(Fit::EncodingUnqualified {
                    encoding: e.clone(),
                    device: device.to_string(),
                    qualified_on,
                });
            }
        }
    }
    Ok(Fit::Ok {
        routes,
        ignored: out.unconsumed,
        device: device.map(str::to_string),
    })
}

// ---------------------------------------------------------------- the report document

/// The canonical, digest-fenced walk result the grade plane ranks (search only, D11). It
/// carries facts and the whole fact set; a verdict lives in `Fit`, never here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompatibilityReport {
    pub contract: String,
    pub header: String,
    pub custody: Custody,
    pub depth: Depth,
    pub mismatches: Vec<Details>,
    pub facts: Vec<ComponentFacts>,
    pub unconsumed: Vec<String>,
}

impl CompatibilityReport {
    pub fn of(
        code: &CodeTopology,
        header: &Header,
        custody: Custody,
        o: &Outcome,
    ) -> Result<CompatibilityReport> {
        Ok(CompatibilityReport {
            contract: code.contract.clone(),
            header: header.object_id()?,
            custody,
            depth: o.depth,
            mismatches: o.mismatches.clone(),
            facts: o.facts.clone(),
            unconsumed: o.unconsumed.clone(),
        })
    }

    pub fn is_satisfied(&self) -> bool {
        self.mismatches.is_empty()
    }

    /// A release cut takes only a manifest whose extras were all consumed: listing and
    /// ignoring is a LOCAL affordance, not a publication policy.
    pub fn release_cut(&self) -> Result<()> {
        if !self.unconsumed.is_empty() {
            return refuse(
                Code::UNCLASSIFIED_EXTRAS,
                format!(
                    "{} unconsumed extra key(s), first {:?}: a local manifest with \
                     unconsumed keys never cuts into a catalog release",
                    self.unconsumed.len(),
                    self.unconsumed[0]
                ),
            );
        }
        match self.mismatches.first() {
            None => Ok(()),
            Some(d) => refuse(
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

fn details_value(d: &Details) -> Value {
    Value::obj(vec![
        ("code", Value::str(d.code.as_str())),
        ("component", Value::str(d.component.clone())),
        ("key", Value::str(d.key.clone())),
        ("present", d.present.to_value()),
        ("required", d.required.to_value()),
    ])
}

fn details_from(v: &Value) -> Result<Details> {
    let mut df = Fields::new("details", v)?;
    let code = as_str("details", "code", df.req("code")?)?.to_string();
    let component = df.req_str("component")?.to_string();
    let key = df.req_str("key")?.to_string();
    let present = Side::from_value(df.req("present")?)?;
    let required = Side::from_value(df.req("required")?)?;
    df.done()?;
    let code = match Code::ALL.iter().find(|c| c.as_str() == code) {
        Some(c) => *c,
        None => {
            return refuse(
                Code::UNKNOWN_FIELD,
                format!("mismatch code {code:?} is outside the closed code set"),
            )
        }
    };
    Ok(Details {
        code,
        component,
        key,
        required,
        present,
    })
}

impl Plain for CompatibilityReport {
    const MAX_BYTES: usize = limits::DOC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("CompatibilityReport", v)?;
        let contract = prefixed("report.contract", f.req_str("contract")?)?;
        let custody = Custody::parse(f.req_str("custody")?)?;
        let depth = Depth::parse(f.req_str("depth")?)?;
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
        let mut mismatches = Vec::new();
        for dv in as_arr("CompatibilityReport", "mismatches", f.req("mismatches")?)? {
            mismatches.push(details_from(dv)?);
        }
        let mut unconsumed = Vec::new();
        for u in as_arr("CompatibilityReport", "unconsumed", f.req("unconsumed")?)? {
            unconsumed.push(as_str("unconsumed", "key", u)?.to_string());
        }
        f.done()?;
        // A walk that stopped at the first fact cannot have recorded a second one.
        if depth == Depth::FirstMismatch && mismatches.len() > 1 {
            return refuse(
                Code::PARTIAL_MISMATCH_SET,
                format!(
                    "depth {:?} with {} facts: a stopped walk records one, so this document \
                     mislabels either its depth or its set",
                    depth.name(),
                    mismatches.len()
                ),
            );
        }
        Ok(CompatibilityReport {
            contract,
            header,
            custody,
            depth,
            mismatches,
            facts,
            unconsumed,
        })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            ("contract", Value::str(self.contract.clone())),
            ("custody", Value::str(self.custody.name())),
            ("depth", Value::str(self.depth.name())),
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
            ("header", Value::str(self.header.clone())),
            (
                "mismatches",
                Value::arr(self.mismatches.iter().map(details_value).collect()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::{Closure, Header, Part, Tensor};

    fn pair() -> (CodeTopology, Header, String) {
        let spec = crate::registry::seeds()
            .into_iter()
            .find(|seed| seed.alias == "plain/1")
            .unwrap()
            .spec;
        let encoding = spec.object_id();
        let header = Header {
            configs: vec![],
            assets: vec![],
            encodings: vec![spec],
            components: vec![(
                "core".into(),
                vec![(
                    "weight".into(),
                    Tensor {
                        dtype: Dtype::Bf16,
                        shape: vec![1],
                        encoding: encoding.clone(),
                        parts: vec![("value".into(), Part::plan(Dtype::Bf16, vec![1], &[0, 0]))],
                    },
                )],
            )],
        };
        let code = CodeTopology {
            contract: crate::ids::object_id(b"inline-spec-test"),
            components: vec![(
                "core".into(),
                vec![(
                    "weight".into(),
                    Requirement {
                        dtype: Dtype::Bf16,
                        shape: vec![1],
                        accepts: Accepts::Any,
                        pinned: false,
                    },
                )],
            )],
            constructed_components: vec!["core".into()],
        };
        (code, header, encoding)
    }

    #[test]
    fn inline_header_spec_is_the_current_checkpoint_closure() {
        let (code, header, encoding) = pair();
        let outcome = check_compatibility(
            &code,
            &header,
            &Closure::default(),
            std::slice::from_ref(&encoding),
            Custody::Canonical,
            Depth::FirstMismatch,
        )
        .unwrap();
        assert!(outcome.is_satisfied());

        let mut missing = header;
        missing.encodings.clear();
        let outcome = check_compatibility(
            &code,
            &missing,
            &Closure::default(),
            &[encoding],
            Custody::Canonical,
            Depth::FirstMismatch,
        )
        .unwrap();
        assert_eq!(outcome.mismatches[0].code, Code::UNKNOWN_ENCODING);
    }
}

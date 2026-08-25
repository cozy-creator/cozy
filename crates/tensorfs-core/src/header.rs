//! The `cozytensors/1` header: one uniform per-tensor form (logical + spec-digest encoding
//! + exact parts), configs by reference, and the frozen topology projection.

use crate::b64;
use crate::canon::{as_arr, as_obj, as_str, Fields, Value};
use crate::dtype::{checked_bytes, checked_elements, Dtype};
use crate::err::{refuse, Code, Result};
use crate::ids::{ascii_name, check_format, prefixed, ConfigRef, Doc, ObjectRef};
use crate::limits;
use crate::spec::EncodingSpec;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Body {
    Segments(Vec<ObjectRef>),
    Inline(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Part {
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub body: Body,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub encoding: String,
    pub parts: Vec<(String, Part)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Header {
    pub configs: Vec<(String, ConfigRef)>,
    pub encodings: Vec<ObjectRef>,
    pub components: Vec<(String, Vec<(String, Tensor)>)>,
}

/// The canonical spec objects a header's closure contains.
#[derive(Debug, Default)]
pub struct Closure {
    pub specs: Vec<(String, EncodingSpec)>,
}

impl Closure {
    pub fn get(&self, id: &str) -> Option<&EncodingSpec> {
        self.specs.iter().find(|(k, _)| k == id).map(|(_, s)| s)
    }
    pub fn insert(&mut self, spec: EncodingSpec) {
        self.specs.push((spec.object_id(), spec));
    }
}

fn shape_from(what: &str, v: &Value) -> Result<Vec<u64>> {
    let a = as_arr(what, "shape", v)?;
    if a.len() > limits::MAX_RANK {
        return refuse(Code::RANK_CAP, format!("{what}: rank {} over cap", a.len()));
    }
    a.iter()
        .map(|x| crate::canon::as_uint(what, "shape", x))
        .collect()
}

fn shape_value(s: &[u64]) -> Value {
    Value::arr(s.iter().map(|d| Value::uint(*d)).collect())
}

impl Part {
    /// Deterministic writer law: at or under the inline cap → inline; above → grid segments.
    pub fn plan(dtype: Dtype, shape: Vec<u64>, bytes: &[u8]) -> Part {
        let body = if bytes.len() as u64 <= limits::INLINE_MAX_BYTES {
            Body::Inline(b64::encode(bytes))
        } else {
            Body::Segments(
                bytes
                    .chunks(limits::GRID_BYTES as usize)
                    .map(ObjectRef::of)
                    .collect(),
            )
        };
        Part { dtype, shape, body }
    }

    fn from_value(v: &Value) -> Result<Part> {
        let mut f = Fields::new("part", v)?;
        let dtype = Dtype::parse(as_str("part", "dtype", f.req("dtype")?)?)?;
        let shape = shape_from("part", f.req("shape")?)?;
        let seg = f.opt("segments");
        let inl = f.opt("inline");
        f.done()?;
        let body = match (seg, inl) {
            (Some(_), Some(_)) => {
                return refuse(
                    Code::INLINE_EXCLUSIVE,
                    "a part carries `segments` or `inline`, never both",
                )
            }
            (None, None) => {
                return refuse(
                    Code::MISSING_FIELD,
                    "part carries neither `segments` nor `inline`",
                )
            }
            (Some(s), None) => {
                let a = as_arr("part", "segments", s)?;
                if a.is_empty() || a.len() > limits::MAX_SEGMENTS {
                    return refuse(
                        Code::COUNT_CAP,
                        format!("part.segments count {} out of bounds", a.len()),
                    );
                }
                let mut refs = Vec::new();
                for x in a {
                    let r = ObjectRef::from_value("part.segment", x)?;
                    if r.length == 0 {
                        return refuse(Code::ZERO_ELEMENT, "zero-length segment");
                    }
                    refs.push(r);
                }
                Body::Segments(refs)
            }
            (None, Some(i)) => Body::Inline(as_str("part", "inline", i)?.to_string()),
        };
        Ok(Part { dtype, shape, body })
    }

    fn to_value(&self) -> Value {
        let mut fields = vec![
            ("dtype", Value::str(self.dtype.name())),
            ("shape", shape_value(&self.shape)),
        ];
        match &self.body {
            Body::Segments(s) => fields.push((
                "segments",
                Value::arr(s.iter().map(|r| r.to_value()).collect()),
            )),
            Body::Inline(b) => fields.push(("inline", Value::str(b.clone()))),
        }
        Value::obj(fields)
    }

    /// Anchor equation + the deterministic inline/segment split.
    fn check_bytes(&self, what: &str) -> Result<()> {
        let want = checked_bytes(what, &self.shape, self.dtype)?;
        match &self.body {
            Body::Inline(b) => {
                let got = b64::decode(b)?.len() as u64;
                if got != want {
                    return refuse(
                        Code::BYTE_LENGTH_MISMATCH,
                        format!(
                            "{what}: inline decodes to {got} B, anchor equation wants {want} B"
                        ),
                    );
                }
                if want > limits::INLINE_MAX_BYTES {
                    return refuse(
                        Code::INLINE_THRESHOLD,
                        format!(
                            "{what}: {want} B is above the {} B inline cap — must be segmented",
                            limits::INLINE_MAX_BYTES
                        ),
                    );
                }
            }
            Body::Segments(s) => {
                let mut got: u64 = 0;
                for r in s {
                    got = got
                        .checked_add(r.length)
                        .ok_or_else(|| crate::err::Refusal {
                            code: Code::ARITH_OVERFLOW,
                            detail: format!("{what}: segment length sum overflows"),
                        })?;
                }
                if got != want {
                    return refuse(
                        Code::BYTE_LENGTH_MISMATCH,
                        format!("{what}: segments total {got} B, anchor equation wants {want} B"),
                    );
                }
                if want <= limits::INLINE_MAX_BYTES {
                    return refuse(
                        Code::INLINE_THRESHOLD,
                        format!("{what}: {want} B is at or under the inline cap — must be inline"),
                    );
                }
            }
        }
        Ok(())
    }
}

impl Tensor {
    fn from_value(v: &Value) -> Result<Tensor> {
        let mut f = Fields::new("tensor", v)?;
        let mut lf = Fields::new("tensor.logical", f.req("logical")?)?;
        let dtype = Dtype::parse(as_str("tensor.logical", "dtype", lf.req("dtype")?)?)?;
        let shape = shape_from("tensor.logical", lf.req("shape")?)?;
        lf.done()?;
        let encoding = prefixed(
            "tensor.encoding",
            as_str("tensor", "encoding", f.req("encoding")?)?,
        )?;
        let ps = as_obj("tensor", "parts", f.req("parts")?)?;
        f.done()?;
        if ps.is_empty() || ps.len() > limits::MAX_PARTS {
            return refuse(
                Code::COUNT_CAP,
                format!("tensor.parts count {} out of bounds", ps.len()),
            );
        }
        let mut parts = Vec::new();
        for (name, pv) in ps {
            ascii_name("tensor.part", name, limits::MAX_NAME_BYTES)?;
            parts.push((name.clone(), Part::from_value(pv)?));
        }
        Ok(Tensor {
            dtype,
            shape,
            encoding,
            parts,
        })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            ("encoding", Value::str(self.encoding.clone())),
            (
                "logical",
                Value::obj(vec![
                    ("dtype", Value::str(self.dtype.name())),
                    ("shape", shape_value(&self.shape)),
                ]),
            ),
            (
                "parts",
                Value::Obj({
                    let mut v: Vec<(String, Value)> = self
                        .parts
                        .iter()
                        .map(|(n, p)| (n.clone(), p.to_value()))
                        .collect();
                    v.sort_by(|a, b| a.0.cmp(&b.0));
                    v
                }),
            ),
        ])
    }
}

impl Header {
    pub fn tensors(&self) -> impl Iterator<Item = (&String, &String, &Tensor)> {
        self.components
            .iter()
            .flat_map(|(c, ts)| ts.iter().map(move |(k, t)| (c, k, t)))
    }

    /// The frozen topology projection: component → key → logical {dtype, shape}.
    pub fn topology_value(&self) -> Value {
        Value::obj(vec![
            ("projection", Value::str("cozytensors.topology/1")),
            (
                "components",
                Value::Obj(
                    self.components
                        .iter()
                        .map(|(c, ts)| {
                            (
                                c.clone(),
                                Value::Obj(
                                    ts.iter()
                                        .map(|(k, t)| {
                                            (
                                                k.clone(),
                                                Value::obj(vec![
                                                    ("dtype", Value::str(t.dtype.name())),
                                                    ("shape", shape_value(&t.shape)),
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
        ])
    }

    pub fn topology_digest(&self) -> String {
        crate::ids::object_id(&crate::canon::write(&self.topology_value()))
    }

    /// Structural validation against the checkpoint's spec closure.
    pub fn validate(&self, closure: &Closure) -> Result<()> {
        if self.components.is_empty() || self.components.len() > limits::MAX_COMPONENTS {
            return refuse(
                Code::EMPTY_COMPONENTS,
                format!("components count {} out of bounds", self.components.len()),
            );
        }
        let listed: Vec<String> = self.encodings.iter().map(|r| r.id()).collect();
        let mut sorted = self
            .encodings
            .iter()
            .map(|r| r.sha256.clone())
            .collect::<Vec<_>>();
        let given = sorted.clone();
        sorted.sort();
        sorted.dedup();
        if sorted != given {
            return refuse(Code::SORT_ORDER, "encodings must be a sorted unique list");
        }
        let mut cited: Vec<String> = Vec::new();
        let mut count = 0usize;

        for (comp, tensors) in &self.components {
            ascii_name("component", comp, limits::MAX_NAME_BYTES)?;
            if tensors.is_empty() {
                return refuse(
                    Code::EMPTY_COMPONENTS,
                    format!("component {comp:?} has no tensors"),
                );
            }
            for (key, t) in tensors {
                count += 1;
                if count > limits::MAX_TENSORS {
                    return refuse(Code::COUNT_CAP, "tensor count over cap");
                }
                let what = format!("{comp}/{key}");
                if key.is_empty() || key.len() > limits::MAX_KEY_BYTES {
                    return refuse(
                        Code::KEY_GRAMMAR,
                        format!("{what}: key length out of bounds"),
                    );
                }
                let elements = checked_elements(&what, &t.shape)?;
                if !listed.contains(&t.encoding) {
                    return refuse(
                        Code::ENCODING_MISMATCH,
                        format!(
                            "{what}: cites {} which is not in the header's `encodings` list",
                            t.encoding
                        ),
                    );
                }
                if !cited.contains(&t.encoding) {
                    cited.push(t.encoding.clone());
                }
                let spec = match closure.get(&t.encoding) {
                    Some(s) => s,
                    None => {
                        return refuse(
                            Code::UNKNOWN_ENCODING,
                            format!(
                                "{what}: spec object {} is absent from the closure",
                                t.encoding
                            ),
                        )
                    }
                };
                if !spec.admits_logical(t.dtype) {
                    return refuse(
                        Code::DTYPE_MISMATCH,
                        format!(
                            "{what}: logical dtype {} is outside the spec's logical_dtypes",
                            t.dtype.name()
                        ),
                    );
                }
                if let Some(r) = spec.logical_rank {
                    if r as usize != t.shape.len() {
                        return refuse(
                            Code::SHAPE_MISMATCH,
                            format!(
                                "{what}: logical rank {} != spec logical_rank {r}",
                                t.shape.len()
                            ),
                        );
                    }
                }
                let want: Vec<&str> = spec.roles.iter().map(|(n, _)| n.as_str()).collect();
                let got: Vec<&str> = t.parts.iter().map(|(n, _)| n.as_str()).collect();
                if want != got {
                    return refuse(
                        Code::ROLE_SET_MISMATCH,
                        format!("{what}: parts {got:?} != the spec's exact role set {want:?}"),
                    );
                }
                for (name, part) in &t.parts {
                    let role = spec.role(name).unwrap();
                    let pw = format!("{what}#{name}");
                    if !role.carrier.admits(t.dtype, part.dtype) {
                        return refuse(
                            Code::DTYPE_MISMATCH,
                            format!(
                                "{pw}: carrier {} is outside the role's carrier set",
                                part.dtype.name()
                            ),
                        );
                    }
                    let want_shape = role.shape.eval(&t.shape, elements)?;
                    if want_shape != part.shape {
                        return refuse(
                            Code::SHAPE_MISMATCH,
                            format!(
                                "{pw}: stored shape {:?} != relation result {want_shape:?}",
                                part.shape
                            ),
                        );
                    }
                    part.check_bytes(&pw)?;
                }
            }
        }
        for r in &self.encodings {
            if !cited.contains(&r.id()) {
                return refuse(
                    Code::UNCITED_ENCODING,
                    format!(
                        "`encodings` lists {} which no tensor cites (derivable-fact law)",
                        r.id()
                    ),
                );
            }
        }
        Ok(())
    }
}

impl Doc for Header {
    const FORMAT: &'static str = "cozytensors/1";
    const MAX_BYTES: usize = limits::DOC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("Header", v)?;
        check_format(
            "Header",
            as_str("Header", "format", f.req("format")?)?,
            Self::FORMAT,
        )?;
        let mut configs = Vec::new();
        if let Some(c) = f.opt("configs") {
            for (name, cv) in as_obj("Header", "configs", c)? {
                ascii_name("Header.configs", name, limits::MAX_NAME_BYTES)?;
                configs.push((name.clone(), ConfigRef::from_value(cv)?));
            }
        }
        let mut encodings = Vec::new();
        for e in as_arr("Header", "encodings", f.req("encodings")?)? {
            encodings.push(ObjectRef::from_value("Header.encodings", e)?);
        }
        let mut components = Vec::new();
        for (name, cv) in as_obj("Header", "components", f.req("components")?)? {
            let mut tensors = Vec::new();
            for (key, tv) in as_obj("Header.components", name, cv)? {
                tensors.push((key.clone(), Tensor::from_value(tv)?));
            }
            components.push((name.clone(), tensors));
        }
        f.done()?;
        Ok(Header {
            configs,
            encodings,
            components,
        })
    }

    fn to_value(&self) -> Value {
        let mut fields = vec![
            ("format", Value::str(Self::FORMAT)),
            (
                "encodings",
                Value::arr(self.encodings.iter().map(|r| r.to_value()).collect()),
            ),
            (
                "components",
                Value::Obj({
                    let mut v: Vec<(String, Value)> = self
                        .components
                        .iter()
                        .map(|(c, ts)| {
                            let mut inner: Vec<(String, Value)> =
                                ts.iter().map(|(k, t)| (k.clone(), t.to_value())).collect();
                            inner.sort_by(|a, b| a.0.cmp(&b.0));
                            (c.clone(), Value::Obj(inner))
                        })
                        .collect();
                    v.sort_by(|a, b| a.0.cmp(&b.0));
                    v
                }),
            ),
        ];
        if !self.configs.is_empty() {
            fields.push((
                "configs",
                Value::Obj({
                    let mut v: Vec<(String, Value)> = self
                        .configs
                        .iter()
                        .map(|(n, c)| (n.clone(), c.to_value()))
                        .collect();
                    v.sort_by(|a, b| a.0.cmp(&b.0));
                    v
                }),
            ));
        }
        Value::obj(fields)
    }
}

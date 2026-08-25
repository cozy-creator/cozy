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

/// One segment slice covering part of a requested byte range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    pub index: usize,
    pub obj: ObjectRef,
    /// offset INSIDE that segment
    pub off: u64,
    pub len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Located {
    Inline { off: u64, len: u64 },
    Segments(Vec<Span>),
}

/// Derived from the length list alone — never from a path, a grid assumption, or a fetch.
/// `stride` is set when the run is uniform (what the writer's grid produces), which makes
/// the start index one division; otherwise the start is a binary search over `starts`.
#[derive(Debug, Clone)]
pub struct SegIndex {
    starts: Vec<u64>,
    total: u64,
    stride: Option<u64>,
}

impl SegIndex {
    fn of(segs: &[ObjectRef]) -> SegIndex {
        let mut starts = Vec::with_capacity(segs.len());
        let mut total = 0u64;
        for s in segs {
            starts.push(total);
            total = total.saturating_add(s.length);
        }
        let first = segs[0].length;
        let uniform = segs[..segs.len() - 1].iter().all(|s| s.length == first)
            && segs[segs.len() - 1].length <= first;
        SegIndex {
            starts,
            total,
            stride: if uniform { Some(first) } else { None },
        }
    }

    pub fn total(&self) -> u64 {
        self.total
    }

    /// O(1) on a uniform run, O(log n) otherwise. Zero fetches either way.
    pub fn at(&self, off: u64) -> usize {
        match self.stride {
            Some(s) => ((off / s) as usize).min(self.starts.len() - 1),
            None => match self.starts.binary_search(&off) {
                Ok(i) => i,
                Err(i) => i - 1,
            },
        }
    }

    pub fn start(&self, i: usize) -> u64 {
        self.starts[i]
    }
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
    /// The ONE deterministic inline/segment decision. Never author-selectable and never
    /// spelled twice: the streaming writer (`checkpoint::objectize`), this in-memory twin,
    /// the validator, and any size model all ask this same function.
    pub fn is_inline(nbytes: u64) -> bool {
        nbytes <= limits::INLINE_MAX_BYTES
    }

    /// In-memory twin of `checkpoint::objectize`, for building frozen vectors where no store
    /// exists. Same law, same grid; it just cannot admit the bytes anywhere.
    pub fn plan(dtype: Dtype, shape: Vec<u64>, bytes: &[u8]) -> Part {
        let body = if Part::is_inline(bytes.len() as u64) {
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

    /// The anchor equation's left side: `checked_prod(shape) × sizeof(dtype)`.
    pub fn nbytes(&self, what: &str) -> Result<u64> {
        checked_bytes(what, &self.shape, self.dtype)
    }

    pub fn segments(&self) -> &[ObjectRef] {
        match &self.body {
            Body::Segments(s) => s,
            Body::Inline(_) => &[],
        }
    }

    /// Build the offset index. Only meaningful for a segmented part.
    pub fn index(&self) -> Option<SegIndex> {
        match &self.body {
            Body::Segments(s) => Some(SegIndex::of(s)),
            Body::Inline(_) => None,
        }
    }

    /// Byte range → the exact ordered slices that cover it, with no body fetched. The anchor
    /// equation is checked FIRST, so a range is bounded by the declaration, not by a file.
    pub fn locate(&self, what: &str, off: u64, len: u64) -> Result<Located> {
        let total = self.nbytes(what)?;
        let end = off.checked_add(len).ok_or_else(|| crate::err::Refusal {
            code: Code::RANGE_BOUNDS,
            detail: format!("{what}: offset {off} + length {len} overflows"),
        })?;
        if end > total {
            return refuse(
                Code::RANGE_BOUNDS,
                format!("{what}: range [{off}, {end}) leaves the part's {total} declared bytes"),
            );
        }
        let segs = match &self.body {
            Body::Inline(_) => return Ok(Located::Inline { off, len }),
            Body::Segments(s) => s,
        };
        let idx = SegIndex::of(segs);
        let mut out = Vec::new();
        let mut cur = off;
        let mut i = idx.at(off);
        while cur < end {
            let s = &segs[i];
            let base = idx.start(i);
            let take = (base + s.length).min(end) - cur;
            out.push(Span {
                index: i,
                obj: s.clone(),
                off: cur - base,
                len: take,
            });
            cur += take;
            i += 1;
        }
        Ok(Located::Segments(out))
    }

    /// Anchor equation + the deterministic inline/segment split.
    pub fn check_bytes(&self, what: &str) -> Result<()> {
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
                if !Part::is_inline(want) {
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
                if Part::is_inline(want) {
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
                Value::map(
                    self.parts
                        .iter()
                        .map(|(n, p)| (n.clone(), p.to_value()))
                        .collect(),
                ),
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
                Value::map(
                    self.components
                        .iter()
                        .map(|(c, ts)| {
                            (
                                c.clone(),
                                Value::map(
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
        // Whole-DOCUMENT totals, not only per-part counts.
        let mut refs = self.configs.len() + self.encodings.len();
        let mut total_bytes: u64 = self
            .configs
            .iter()
            .map(|(_, c)| c.length)
            .chain(self.encodings.iter().map(|r| r.length))
            .sum();

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
                    refs += part.segments().len();
                    if refs > limits::MAX_TOTAL_REFS {
                        return refuse(
                            Code::TOTAL_REFS_CAP,
                            format!(
                                "{pw}: whole-document reference total passes {}",
                                limits::MAX_TOTAL_REFS
                            ),
                        );
                    }
                    total_bytes = total_bytes.checked_add(part.nbytes(&pw)?).ok_or_else(|| {
                        crate::err::Refusal {
                            code: Code::ARITH_OVERFLOW,
                            detail: format!("{pw}: whole-document byte total overflows"),
                        }
                    })?;
                    if total_bytes > limits::MAX_TOTAL_BYTES {
                        return refuse(
                            Code::TOTAL_BYTES_CAP,
                            format!(
                                "{pw}: whole-document byte total {total_bytes} passes {}",
                                limits::MAX_TOTAL_BYTES
                            ),
                        );
                    }
                }
            }
        }
        let _ = refs;
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

    /// SERVING admission, which is not validation. cozytensors.md §4 keeps three states
    /// apart: a MISSING spec object never installs (`UNKNOWN_ENCODING`, in `validate`); a
    /// closure that CONTAINS a spec the local pin does not alias is valid DATA and
    /// unsupported EXECUTION — it stores and validates, and only serving refuses, naming
    /// the digest. Offline honesty: a local pin cannot tell "unregistered" from
    /// "newer than my pin", so the refusal says exactly that and nothing more.
    pub fn admit_for_serving(&self, closure: &Closure, pin: &[String]) -> Result<()> {
        self.validate(closure)?;
        for r in &self.encodings {
            if !pin.contains(&r.id()) {
                return refuse(
                    Code::UNKNOWN_TO_LOCAL_CAPABILITY,
                    format!(
                        "spec {} carries no alias in this runtime's pin — unregistered and \
                         newer-than-pin are indistinguishable offline; the checkpoint is \
                         valid data, its execution is unsupported here",
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
                Value::map(
                    self.components
                        .iter()
                        .map(|(c, ts)| {
                            (
                                c.clone(),
                                Value::map(
                                    ts.iter().map(|(k, t)| (k.clone(), t.to_value())).collect(),
                                ),
                            )
                        })
                        .collect(),
                ),
            ),
        ];
        if !self.configs.is_empty() {
            fields.push((
                "configs",
                Value::map(
                    self.configs
                        .iter()
                        .map(|(n, c)| (n.clone(), c.to_value()))
                        .collect(),
                ),
            ));
        }
        Value::obj(fields)
    }
}

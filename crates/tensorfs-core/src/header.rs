//! The `cozytensors/1` header: one uniform per-tensor form (logical + spec-digest encoding
//! + exact parts), configs by reference, and the frozen topology projection.

use crate::canon::Value;
use crate::dtype::{checked_bytes, checked_elements, Dtype};
use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::{ascii_name, ConfigRef, ObjectRef};
use crate::limits;
use crate::spec::EncodingSpec;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Body {
    Segments(Vec<ObjectRef>),
    Inline(Vec<u8>),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Asset {
    pub logical_sha256: String,
    pub logical_length: u64,
    pub media_type: String,
    pub segments: Vec<ObjectRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Header {
    pub configs: Vec<(String, ConfigRef)>,
    pub assets: Vec<(String, Asset)>,
    pub encodings: Vec<EncodingSpec>,
    pub components: Vec<(String, Vec<(String, Tensor)>)>,
}

/// The canonical spec objects a header's closure contains.
#[derive(Debug, Default)]
pub struct Closure {
    pub specs: Vec<(String, EncodingSpec)>,
    index: std::collections::HashMap<String, usize>,
}

impl Closure {
    pub fn get(&self, id: &str) -> Option<&EncodingSpec> {
        self.index
            .get(id)
            .and_then(|index| self.specs.get(*index))
            .map(|(_, spec)| spec)
    }
    pub fn insert(&mut self, spec: EncodingSpec) {
        let id = spec.object_id();
        if let Some(index) = self.index.get(&id).copied() {
            self.specs[index] = (id, spec);
        } else {
            self.index.insert(id.clone(), self.specs.len());
            self.specs.push((id, spec));
        }
    }
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
            Body::Inline(bytes.to_vec())
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
                let got = b.len() as u64;
                if self.dtype == Dtype::Bool && b.iter().any(|byte| *byte > 1) {
                    return refuse(
                        Code::DTYPE_MISMATCH,
                        format!("{what}: bool inline body contains a byte outside {{0,1}}"),
                    );
                }
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
                    crate::ids::hex64("part segment", &r.sha256)?;
                    if r.length == 0 {
                        return refuse(Code::ZERO_ELEMENT, format!("{what}: zero-length segment"));
                    }
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

/// Domain separator for topology identity. This exact literal remains in the hash preimage
/// so existing topology digests do not move; it is not a stored document format.
pub const TOPOLOGY_IDENTITY_DOMAIN: &str = "cozytensors.topology/1";

/// Domain separator for the SHAPE-ONLY projection. A distinct literal on purpose: the two
/// projections are different questions, and a shared domain would let a shape digest be
/// mistaken for an exact one in a bucket key.
pub const TOPOLOGY_SHAPE_DOMAIN: &str = "cozytensors.topology.shape/1";

/// What `tfs ingest` derives from a header and stores on the checkpoint row: identity,
/// the fit index, and the construction document's identity (design §2). `config_digest`
/// is absent exactly when the header names no config: there is no construction document
/// to digest, and the hub refuses to serve one (`checkpoint.config_absent`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Digests {
    pub topology_digest: String,
    pub shape_digest: String,
    pub config_digest: Option<String>,
}

/// Which topology question a digest answers.
///
/// `Identity` is the exact index: a hit means the logical {dtype, shape} sets agree.
/// `ShapeOnly` elides the dtype leaf, so everything that differs ONLY by dtype or encoding
/// lands in one bucket — which is exactly the CONVERTIBLE/PRODUCIBLE candidate set. Catalog
/// search is then three cheap steps: an exact-digest hit is COMPATIBLE, a shape-digest hit
/// is worth grading, and everything else is INCOMPATIBLE without grading at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Projection {
    Identity,
    ShapeOnly,
}

impl Projection {
    pub fn domain(self) -> &'static str {
        match self {
            Projection::Identity => TOPOLOGY_IDENTITY_DOMAIN,
            Projection::ShapeOnly => TOPOLOGY_SHAPE_DOMAIN,
        }
    }
}

/// The ONE topology projection: component → key → logical {dtype, shape}. A real header and
/// reviewed structure data must produce the same bytes for the same structure, so they call
/// the same function over the same iterator shape rather than each building a map.
///
/// Both digests come out of THIS body. The only thing `projection` changes is whether the
/// leaf carries its dtype and which domain separator is stamped; grouping, ordering and
/// canonical form are shared code, so the exact digest and the shape digest cannot drift
/// apart under an edit — a change to how a topology is projected moves both or neither.
pub fn topology_projection<'a>(
    projection: Projection,
    items: impl Iterator<Item = (&'a str, &'a str, Dtype, &'a [u64])>,
) -> Value {
    let mut comps: Vec<(String, Vec<(String, Value)>)> = Vec::new();
    for (c, k, dtype, shape) in items {
        let mut leaf = Vec::with_capacity(2);
        if projection == Projection::Identity {
            leaf.push(("dtype", Value::str(dtype.name())));
        }
        leaf.push(("shape", shape_value(shape)));
        let leaf = Value::obj(leaf);
        match comps.iter_mut().find(|(n, _)| n == c) {
            Some((_, ks)) => ks.push((k.to_string(), leaf)),
            None => comps.push((c.to_string(), vec![(k.to_string(), leaf)])),
        }
    }
    Value::obj(vec![
        (
            "components",
            Value::map(
                comps
                    .into_iter()
                    .map(|(c, ks)| (c, Value::map(ks)))
                    .collect(),
            ),
        ),
        ("projection", Value::str(projection.domain())),
    ])
}

/// The exact projection. Kept as its own name because it is the identity index every
/// existing document already froze.
pub fn topology_value<'a>(
    items: impl Iterator<Item = (&'a str, &'a str, Dtype, &'a [u64])>,
) -> Value {
    topology_projection(Projection::Identity, items)
}

/// The shape-only projection — the same body with the dtype leaf elided.
pub fn topology_shape_value<'a>(
    items: impl Iterator<Item = (&'a str, &'a str, Dtype, &'a [u64])>,
) -> Value {
    topology_projection(Projection::ShapeOnly, items)
}

pub fn topology_digest_of(v: &Value) -> String {
    crate::ids::object_id(&crate::canon::write(v))
}

impl Header {
    pub fn tensors(&self) -> impl Iterator<Item = (&String, &String, &Tensor)> {
        self.components
            .iter()
            .flat_map(|(c, ts)| ts.iter().map(move |(k, t)| (c, k, t)))
    }

    pub fn projected(&self, projection: Projection) -> Value {
        topology_projection(
            projection,
            self.tensors()
                .map(|(c, k, t)| (c.as_str(), k.as_str(), t.dtype, t.shape.as_slice())),
        )
    }

    pub fn topology_value(&self) -> Value {
        self.projected(Projection::Identity)
    }

    pub fn topology_digest(&self) -> String {
        topology_digest_of(&self.topology_value())
    }

    /// The FIT index (D4): the same projection with the dtype elided. Two checkpoints with
    /// equal `shape_digest` fit the same code; `topology_digest` is identity.
    pub fn shape_digest(&self) -> String {
        topology_digest_of(&self.projected(Projection::ShapeOnly))
    }

    /// The construction document (design §2): what the factory consumes, assembled from
    /// the exact bytes of every config the header names — one config is the document
    /// itself; several become the name-keyed map — in the interoperable JCS profile
    /// (`jcs.rs`). Byte-identical to the hub's `ConstructionDocument` and cozy-runtime's
    /// `model_config.construction_document`, so its sha256 is the one `config_digest` a
    /// derive is keyed by. `None` when the header names no config. The supplied bytes must
    /// be exactly the objects the header references, by name, sha256 and length.
    pub fn construction_document(&self, configs: &[(String, Vec<u8>)]) -> Result<Option<Vec<u8>>> {
        if configs.len() != self.configs.len() {
            return refuse(
                Code::MISSING_FIELD,
                format!(
                    "header names {} configs, {} supplied",
                    self.configs.len(),
                    configs.len()
                ),
            );
        }
        let mut members = Vec::with_capacity(self.configs.len());
        for (name, c) in &self.configs {
            let Some((_, bytes)) = configs.iter().find(|(n, _)| n == name) else {
                return refuse(Code::MISSING_FIELD, format!("config {name:?} not supplied"));
            };
            if bytes.len() as u64 != c.length {
                return refuse(
                    Code::LENGTH_MISMATCH,
                    format!(
                        "config {name:?}: {} bytes supplied, header says {}",
                        bytes.len(),
                        c.length
                    ),
                );
            }
            let got = crate::sha256::hex_digest(bytes);
            if got != c.sha256 {
                return refuse(
                    Code::OBJECT_ID_MISMATCH,
                    format!(
                        "config {name:?}: sha256:{got} supplied, header says sha256:{}",
                        c.sha256
                    ),
                );
            }
            let document =
                crate::jcs::parse(bytes, limits::DOC_MAX_BYTES).map_err(|e| Refusal {
                    code: e.code,
                    detail: format!("config {name:?}: {}", e.detail),
                })?;
            if !matches!(document, crate::jcs::Json::Obj(_)) {
                return refuse(
                    Code::WRONG_TYPE,
                    format!("config {name:?} is not an object"),
                );
            }
            members.push((name.clone(), document));
        }
        let document = match members.len() {
            0 => return Ok(None),
            1 => members.pop().expect("one member").1,
            _ => crate::jcs::Json::object(members)?,
        };
        Ok(Some(crate::jcs::write(&document)))
    }

    /// The three catalog digests a checkpoint row carries (design §2), from the header and
    /// the exact bytes of the configs it names.
    pub fn digests(&self, configs: &[(String, Vec<u8>)]) -> Result<Digests> {
        Ok(Digests {
            topology_digest: self.topology_digest(),
            shape_digest: self.shape_digest(),
            config_digest: self
                .construction_document(configs)?
                .map(|document| crate::ids::object_id(&document)),
        })
    }

    /// Closure-independent schema, geometry, ordering and aggregate-limit validation.
    pub fn validate_structure(&self) -> Result<()> {
        if self.components.is_empty() || self.components.len() > limits::MAX_COMPONENTS {
            return refuse(
                Code::EMPTY_COMPONENTS,
                format!("components count {} out of bounds", self.components.len()),
            );
        }
        let sorted_names = |what: &str, names: Vec<&str>| -> Result<()> {
            let mut sorted = names.clone();
            sorted.sort();
            sorted.dedup();
            if sorted != names {
                return refuse(
                    Code::SORT_ORDER,
                    format!("{what} must be sorted and unique"),
                );
            }
            Ok(())
        };
        sorted_names(
            "configs",
            self.configs.iter().map(|(name, _)| name.as_str()).collect(),
        )?;
        sorted_names(
            "assets",
            self.assets.iter().map(|(name, _)| name.as_str()).collect(),
        )?;
        for (name, config) in &self.configs {
            ascii_name("config", name, limits::MAX_NAME_BYTES)?;
            crate::ids::hex64("config digest", &config.sha256)?;
            if config.length == 0 {
                return refuse(Code::ZERO_ELEMENT, format!("config {name:?} is empty"));
            }
            if config.media_type.is_empty()
                || config.media_type.len() > 128
                || !config
                    .media_type
                    .bytes()
                    .all(|byte| (0x20..=0x7e).contains(&byte))
            {
                return refuse(
                    Code::KEY_GRAMMAR,
                    format!("config {name:?}: bad media type"),
                );
            }
        }
        let mut asset_refs = 0usize;
        let mut asset_bytes = 0u64;
        for (name, asset) in &self.assets {
            crate::manifest::check_path(name)?;
            crate::ids::hex64("asset logical digest", &asset.logical_sha256)?;
            if asset.logical_length == 0 || asset.segments.is_empty() {
                return refuse(Code::ZERO_ELEMENT, format!("asset {name:?} is empty"));
            }
            if asset.media_type.is_empty()
                || asset.media_type.len() > 128
                || !asset
                    .media_type
                    .bytes()
                    .all(|byte| (0x20..=0x7e).contains(&byte))
            {
                return refuse(Code::KEY_GRAMMAR, format!("asset {name:?}: bad media type"));
            }
            let mut total = 0u64;
            for segment in &asset.segments {
                crate::ids::hex64("asset segment", &segment.sha256)?;
                if segment.length == 0 {
                    return refuse(Code::ZERO_ELEMENT, format!("asset {name:?}: empty segment"));
                }
                total = total.checked_add(segment.length).ok_or_else(|| Refusal {
                    code: Code::ARITH_OVERFLOW,
                    detail: format!("asset {name:?}: segment length sum overflows"),
                })?;
            }
            if total != asset.logical_length {
                return refuse(
                    Code::BYTE_LENGTH_MISMATCH,
                    format!(
                        "asset {name:?}: segments total {total} B, logical length {} B",
                        asset.logical_length
                    ),
                );
            }
            asset_refs = asset_refs
                .checked_add(asset.segments.len())
                .ok_or_else(|| Refusal {
                    code: Code::ARITH_OVERFLOW,
                    detail: "asset reference total overflows".into(),
                })?;
            asset_bytes = asset_bytes.checked_add(total).ok_or_else(|| Refusal {
                code: Code::ARITH_OVERFLOW,
                detail: "asset byte total overflows".into(),
            })?;
        }
        if self.encodings.is_empty() {
            return refuse(Code::COUNT_CAP, "encodings is empty");
        }
        let mut sorted = self
            .encodings
            .iter()
            .map(EncodingSpec::object_id)
            .collect::<Vec<_>>();
        let given = sorted.clone();
        sorted.sort();
        sorted.dedup();
        if sorted != given {
            return refuse(Code::SORT_ORDER, "encodings must be a sorted unique list");
        }
        let listed: std::collections::HashMap<String, usize> = self
            .encodings
            .iter()
            .enumerate()
            .map(|(index, encoding)| (encoding.object_id(), index))
            .collect();
        let mut cited = vec![false; self.encodings.len()];
        let mut count = 0usize;
        // Whole-DOCUMENT totals, not only per-part counts.
        let mut refs = self
            .configs
            .len()
            .checked_add(asset_refs)
            .ok_or_else(|| Refusal {
                code: Code::ARITH_OVERFLOW,
                detail: "header reference total overflows".into(),
            })?;
        let mut total_bytes = asset_bytes;
        for length in self.configs.iter().map(|(_, config)| config.length) {
            total_bytes = total_bytes.checked_add(length).ok_or_else(|| Refusal {
                code: Code::ARITH_OVERFLOW,
                detail: "header referenced-byte total overflows".into(),
            })?;
        }
        if refs > limits::MAX_TOTAL_REFS {
            return refuse(Code::TOTAL_REFS_CAP, "header reference total over cap");
        }
        if total_bytes > limits::MAX_TOTAL_BYTES {
            return refuse(Code::TOTAL_BYTES_CAP, "header referenced bytes over cap");
        }

        let mut component_names = std::collections::HashSet::new();

        for (comp, tensors) in &self.components {
            ascii_name("component", comp, limits::MAX_NAME_BYTES)?;
            if !component_names.insert(comp.as_str()) {
                return refuse(
                    Code::DUPLICATE_KEY,
                    format!("component {comp:?} appears twice"),
                );
            }
            if tensors.is_empty() {
                return refuse(
                    Code::EMPTY_COMPONENTS,
                    format!("component {comp:?} has no tensors"),
                );
            }
            let mut tensor_names = std::collections::HashSet::new();
            for (key, t) in tensors {
                count = count.checked_add(1).ok_or_else(|| Refusal {
                    code: Code::ARITH_OVERFLOW,
                    detail: "tensor count overflows".into(),
                })?;
                if count > limits::MAX_TENSORS {
                    return refuse(Code::COUNT_CAP, "tensor count over cap");
                }
                let what = format!("{comp}/{key}");
                ascii_name("tensor key", key, limits::MAX_KEY_BYTES)?;
                if !tensor_names.insert(key.as_str()) {
                    return refuse(Code::DUPLICATE_KEY, format!("{what}: key appears twice"));
                }
                checked_elements(&what, &t.shape)?;
                let encoding_index = match listed.get(&t.encoding).copied() {
                    Some(index) => index,
                    None => {
                        return refuse(
                            Code::ENCODING_MISMATCH,
                            format!(
                                "{what}: cites {} which is not in the header's `encodings` list",
                                t.encoding
                            ),
                        )
                    }
                };
                cited[encoding_index] = true;
                if t.parts.is_empty() || t.parts.len() > limits::MAX_PARTS {
                    return refuse(
                        Code::COUNT_CAP,
                        format!("{what}: part count {} out of bounds", t.parts.len()),
                    );
                }
                let mut roles = std::collections::HashSet::new();
                for (name, part) in &t.parts {
                    ascii_name("part role", name, limits::MAX_NAME_BYTES)?;
                    if !roles.insert(name.as_str()) {
                        return refuse(
                            Code::DUPLICATE_KEY,
                            format!("{what}: part role {name:?} appears twice"),
                        );
                    }
                    let pw = format!("{what}#{name}");
                    part.check_bytes(&pw)?;
                    refs = refs
                        .checked_add(part.segments().len())
                        .ok_or_else(|| Refusal {
                            code: Code::ARITH_OVERFLOW,
                            detail: format!("{pw}: reference total overflows"),
                        })?;
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
        for (index, r) in self.encodings.iter().enumerate() {
            if !cited[index] {
                return refuse(
                    Code::UNCITED_ENCODING,
                    format!(
                        "`encodings` lists {} which no tensor cites (derivable-fact law)",
                        r.object_id()
                    ),
                );
            }
        }
        Ok(())
    }

    /// Structural validation against the checkpoint's exact EncodingSpec closure.
    pub fn validate(&self, _closure: &Closure) -> Result<()> {
        self.validate_structure()?;
        let specs: std::collections::HashMap<String, &EncodingSpec> = self
            .encodings
            .iter()
            .map(|spec| (spec.object_id(), spec))
            .collect();
        for (comp, key, tensor) in self.tensors() {
            let what = format!("{comp}/{key}");
            let elements = checked_elements(&what, &tensor.shape)?;
            let spec = specs
                .get(&tensor.encoding)
                .copied()
                .ok_or_else(|| Refusal {
                    code: Code::UNKNOWN_ENCODING,
                    detail: format!(
                        "{what}: encoding {} is absent from the header",
                        tensor.encoding
                    ),
                })?;
            if !spec.admits_logical(tensor.dtype) {
                return refuse(
                    Code::DTYPE_MISMATCH,
                    format!(
                        "{what}: logical dtype {} is outside the spec's logical_dtypes",
                        tensor.dtype.name()
                    ),
                );
            }
            if let Some(rank) = spec.logical_rank {
                if rank as usize != tensor.shape.len() {
                    return refuse(
                        Code::SHAPE_MISMATCH,
                        format!(
                            "{what}: logical rank {} != spec logical_rank {rank}",
                            tensor.shape.len()
                        ),
                    );
                }
            }
            let want: Vec<&str> = spec.roles.iter().map(|(name, _)| name.as_str()).collect();
            let got: Vec<&str> = tensor.parts.iter().map(|(name, _)| name.as_str()).collect();
            if want != got {
                return refuse(
                    Code::ROLE_SET_MISMATCH,
                    format!("{what}: parts {got:?} != the spec's exact role order {want:?}"),
                );
            }
            for (name, part) in &tensor.parts {
                let role = spec.role(name).unwrap();
                let part_what = format!("{what}#{name}");
                if !role.carrier.admits(tensor.dtype, part.dtype) {
                    return refuse(
                        Code::DTYPE_MISMATCH,
                        format!(
                            "{part_what}: carrier {} is outside the role's carrier set",
                            part.dtype.name()
                        ),
                    );
                }
                let want_shape = role.shape.eval(&tensor.shape, elements)?;
                if want_shape != part.shape {
                    return refuse(
                        Code::SHAPE_MISMATCH,
                        format!(
                            "{part_what}: stored shape {:?} != relation result {want_shape:?}",
                            part.shape
                        ),
                    );
                }
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
            if !pin.contains(&r.object_id()) {
                return refuse(
                    Code::UNKNOWN_TO_LOCAL_CAPABILITY,
                    format!(
                        "spec {} carries no alias in this runtime's pin — unregistered and \
                         newer-than-pin are indistinguishable offline; the checkpoint is \
                         valid data, its execution is unsupported here",
                        r.object_id()
                    ),
                );
            }
        }
        Ok(())
    }
}

impl Header {
    pub const FORMAT: &'static str = "cozytensors/1";
    pub const MAX_BYTES: usize = limits::DOC_MAX_BYTES;

    pub fn parse(bytes: &[u8]) -> Result<Self> {
        <Self as crate::ids::StoredDoc>::parse(bytes)
    }
    pub fn canonical_bytes(&self) -> Result<Vec<u8>> {
        <Self as crate::ids::StoredDoc>::canonical_bytes(self)
    }
    pub fn object_id(&self) -> Result<String> {
        <Self as crate::ids::StoredDoc>::object_id(self)
    }
    pub fn object_ref(&self) -> Result<ObjectRef> {
        <Self as crate::ids::StoredDoc>::object_ref(self)
    }

    /// Binary red-vector authoring only. This never computes identity or admits storage;
    /// production callers use the validating fallible methods above.
    #[doc(hidden)]
    pub fn conformance_bytes(&self) -> Result<Vec<u8>> {
        crate::cbor::encode_header_unvalidated(self)
    }

    /// Parse the ordered diagnostic JSON schema used only by conformance tooling. The
    /// resulting Header is encoded by the production CBOR writer; these JSON bytes have no
    /// identity and no stored-reader path.
    pub fn from_diagnostic_value(value: &Value) -> Result<Self> {
        use crate::canon::{as_arr, as_str, Fields};

        fn row<'a>(what: &str, value: &'a Value, n: usize) -> Result<&'a [Value]> {
            let values = crate::canon::as_arr(what, "row", value)?;
            if values.len() != n {
                return refuse(
                    Code::WRONG_TYPE,
                    format!("{what}: row arity {}, expected {n}", values.len()),
                );
            }
            Ok(values)
        }
        fn uint(what: &str, value: &Value) -> Result<u64> {
            crate::canon::as_uint(what, "value", value)
        }
        fn digest(what: &str, value: &Value) -> Result<String> {
            let value = as_str(what, "digest", value)?;
            Ok(crate::ids::prefixed(what, value)?[7..].to_string())
        }
        fn object_ref(what: &str, value: &Value) -> Result<ObjectRef> {
            let values = row(what, value, 2)?;
            Ok(ObjectRef {
                sha256: digest(what, &values[0])?,
                length: uint(what, &values[1])?,
            })
        }
        fn shape(what: &str, value: &Value) -> Result<Vec<u64>> {
            let values = as_arr(what, "shape", value)?;
            if values.len() > limits::MAX_RANK {
                return refuse(Code::RANK_CAP, format!("{what}: rank over cap"));
            }
            values.iter().map(|value| uint(what, value)).collect()
        }
        fn hex_bytes(what: &str, value: &Value) -> Result<Vec<u8>> {
            let text = as_str(what, "inline_hex", value)?;
            if text.len() % 2 != 0 || !text.bytes().all(|b| b.is_ascii_hexdigit()) {
                return refuse(
                    Code::MALFORMED_DIGEST,
                    format!("{what}: malformed hex bytes"),
                );
            }
            text.as_bytes()
                .chunks_exact(2)
                .map(|pair| {
                    std::str::from_utf8(pair)
                        .ok()
                        .and_then(|p| u8::from_str_radix(p, 16).ok())
                        .ok_or_else(|| Refusal {
                            code: Code::MALFORMED_DIGEST,
                            detail: format!("{what}: malformed hex bytes"),
                        })
                })
                .collect()
        }

        let mut fields = Fields::new("Header diagnostic", value)?;
        let format = fields.req_str("format")?;
        if format != Self::FORMAT {
            return refuse(
                Code::UNKNOWN_FORMAT,
                format!("diagnostic format {format:?} != {:?}", Self::FORMAT),
            );
        }
        let mut configs = Vec::new();
        for (i, value) in as_arr("Header diagnostic", "configs", fields.req("configs")?)?
            .iter()
            .enumerate()
        {
            let what = format!("configs[{i}]");
            let values = row(&what, value, 4)?;
            configs.push((
                as_str(&what, "name", &values[0])?.to_string(),
                ConfigRef {
                    sha256: digest(&what, &values[1])?,
                    length: uint(&what, &values[2])?,
                    media_type: as_str(&what, "media_type", &values[3])?.to_string(),
                },
            ));
        }
        let mut assets = Vec::new();
        for (i, value) in as_arr("Header diagnostic", "assets", fields.req("assets")?)?
            .iter()
            .enumerate()
        {
            let what = format!("assets[{i}]");
            let values = row(&what, value, 5)?;
            let segments = as_arr(&what, "segments", &values[4])?
                .iter()
                .enumerate()
                .map(|(j, value)| object_ref(&format!("{what}.segments[{j}]"), value))
                .collect::<Result<Vec<_>>>()?;
            assets.push((
                as_str(&what, "name", &values[0])?.to_string(),
                Asset {
                    logical_sha256: digest(&what, &values[1])?,
                    logical_length: uint(&what, &values[2])?,
                    media_type: as_str(&what, "media_type", &values[3])?.to_string(),
                    segments,
                },
            ));
        }
        let encodings = as_arr("Header diagnostic", "encodings", fields.req("encodings")?)?
            .iter()
            .map(EncodingSpec::from_value)
            .collect::<Result<Vec<_>>>()?;
        let mut components = Vec::new();
        for (i, value) in as_arr("Header diagnostic", "components", fields.req("components")?)?
            .iter()
            .enumerate()
        {
            let what = format!("components[{i}]");
            let values = row(&what, value, 2)?;
            let name = as_str(&what, "name", &values[0])?.to_string();
            let mut tensors = Vec::new();
            for (j, value) in as_arr(&what, "tensors", &values[1])?.iter().enumerate() {
                let tensor_what = format!("{what}.tensors[{j}]");
                let values = row(&tensor_what, value, 5)?;
                let encoding_index =
                    usize::try_from(uint(&tensor_what, &values[3])?).map_err(|_| Refusal {
                        code: Code::NUMBER_RANGE,
                        detail: format!(
                            "{tensor_what}: encoding index is outside this platform's usize range"
                        ),
                    })?;
                let encoding = encodings.get(encoding_index).ok_or_else(|| Refusal {
                    code: Code::ENCODING_MISMATCH,
                    detail: format!("{tensor_what}: encoding index {encoding_index} out of range"),
                })?;
                let mut parts = Vec::new();
                for (k, value) in as_arr(&tensor_what, "parts", &values[4])?
                    .iter()
                    .enumerate()
                {
                    let part_what = format!("{tensor_what}.parts[{k}]");
                    let values = row(&part_what, value, 4)?;
                    let mut body_fields = Fields::new("diagnostic body", &values[3])?;
                    let inline = body_fields.opt("inline_hex");
                    let segments = body_fields.opt("segments");
                    body_fields.done()?;
                    let body = match (inline, segments) {
                        (Some(value), None) => Body::Inline(hex_bytes(&part_what, value)?),
                        (None, Some(value)) => Body::Segments(
                            as_arr(&part_what, "segments", value)?
                                .iter()
                                .enumerate()
                                .map(|(n, value)| {
                                    object_ref(&format!("{part_what}.segments[{n}]"), value)
                                })
                                .collect::<Result<Vec<_>>>()?,
                        ),
                        _ => {
                            return refuse(
                                Code::INLINE_EXCLUSIVE,
                                format!("{part_what}: body has exactly one representation"),
                            )
                        }
                    };
                    parts.push((
                        as_str(&part_what, "role", &values[0])?.to_string(),
                        Part {
                            dtype: Dtype::parse(as_str(&part_what, "dtype", &values[1])?)?,
                            shape: shape(&part_what, &values[2])?,
                            body,
                        },
                    ));
                }
                tensors.push((
                    as_str(&tensor_what, "key", &values[0])?.to_string(),
                    Tensor {
                        dtype: Dtype::parse(as_str(&tensor_what, "dtype", &values[1])?)?,
                        shape: shape(&tensor_what, &values[2])?,
                        encoding: encoding.object_id(),
                        parts,
                    },
                ));
            }
            components.push((name, tensors));
        }
        fields.done()?;
        Ok(Header {
            configs,
            assets,
            encodings,
            components,
        })
    }

    /// Readable JSON-shaped projection. It is never accepted as a header or hashed.
    pub fn diagnostic_value(&self) -> Result<Value> {
        self.validate_structure()?;
        let object_ref =
            |r: &ObjectRef| Value::arr(vec![Value::str(r.id()), Value::uint(r.length)]);
        let encoding_index: std::collections::HashMap<String, usize> = self
            .encodings
            .iter()
            .enumerate()
            .map(|(index, encoding)| (encoding.object_id(), index))
            .collect();
        let mut component_values = Vec::with_capacity(self.components.len());
        for (component, rows) in &self.components {
            let mut tensor_values = Vec::with_capacity(rows.len());
            for (key, tensor) in rows {
                let encoding = encoding_index
                    .get(&tensor.encoding)
                    .copied()
                    .ok_or_else(|| Refusal {
                        code: Code::ENCODING_MISMATCH,
                        detail: format!("{component}/{key}: encoding is not listed"),
                    })?;
                let parts = tensor
                    .parts
                    .iter()
                    .map(|(role, part)| {
                        let body = match &part.body {
                            Body::Inline(bytes) => Value::obj(vec![(
                                "inline_hex",
                                Value::str(crate::sha256::hex(bytes)),
                            )]),
                            Body::Segments(segments) => Value::obj(vec![(
                                "segments",
                                Value::arr(segments.iter().map(&object_ref).collect()),
                            )]),
                        };
                        Value::arr(vec![
                            Value::str(role),
                            Value::str(part.dtype.name()),
                            shape_value(&part.shape),
                            body,
                        ])
                    })
                    .collect();
                tensor_values.push(Value::arr(vec![
                    Value::str(key),
                    Value::str(tensor.dtype.name()),
                    shape_value(&tensor.shape),
                    Value::uint(encoding as u64),
                    Value::arr(parts),
                ]));
            }
            component_values.push(Value::arr(vec![
                Value::str(component),
                Value::arr(tensor_values),
            ]));
        }
        Ok(Value::obj(vec![
            ("format", Value::str(Self::FORMAT)),
            (
                "configs",
                Value::arr(
                    self.configs
                        .iter()
                        .map(|(name, config)| {
                            Value::arr(vec![
                                Value::str(name),
                                Value::str(format!("sha256:{}", config.sha256)),
                                Value::uint(config.length),
                                Value::str(&config.media_type),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "assets",
                Value::arr(
                    self.assets
                        .iter()
                        .map(|(name, asset)| {
                            Value::arr(vec![
                                Value::str(name),
                                Value::str(format!("sha256:{}", asset.logical_sha256)),
                                Value::uint(asset.logical_length),
                                Value::str(&asset.media_type),
                                Value::arr(asset.segments.iter().map(&object_ref).collect()),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "encodings",
                Value::arr(self.encodings.iter().map(EncodingSpec::to_value).collect()),
            ),
            ("components", Value::arr(component_values)),
        ]))
    }
}

impl crate::ids::StoredDoc for Header {
    const FORMAT: &'static str = Header::FORMAT;
    const MAX_BYTES: usize = Header::MAX_BYTES;

    fn canonical_bytes(&self) -> Result<Vec<u8>> {
        crate::cbor::encode_header(self)
    }
    fn parse(bytes: &[u8]) -> Result<Self> {
        crate::cbor::decode_header(bytes)
    }
}

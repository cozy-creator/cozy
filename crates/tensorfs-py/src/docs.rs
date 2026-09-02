//! Document marshalling: canonical bytes in, plain Python data out.
//!
//! Nothing here decides anything. Every function hands bytes to the ONE reader in
//! `tensorfs-core` and turns its result into dicts, lists, ints and strings. Python never
//! sees a partially-validated document: a refusal is raised before any data is built.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use tensorfs_core::canon::Value;
use tensorfs_core::capability::CapabilityRecords;
use tensorfs_core::corpus;
use tensorfs_core::dtype::Dtype;
use tensorfs_core::fit::{Accepts, CodeTopology, Custody, Requirement};
use tensorfs_core::header::{Closure as CoreClosure, Header};
use tensorfs_core::ids::{self, Doc, Plain};
use tensorfs_core::manifest::Manifest;
use tensorfs_core::registry;
use tensorfs_core::spec::EncodingSpec;

use crate::errors::IntoPy;

type CodeTensorRow = (String, String, String, Vec<u64>, Option<Vec<String>>, bool);

/// Author one canonical CodeTopology from the meta-device census rows. TensorFS receives
/// exact component/key fields; it never runs package code or guesses prefixes.
#[pyfunction(name = "_code_topology")]
pub fn code_topology<'p>(
    py: Python<'p>,
    contract_digest: &str,
    tensors: Vec<CodeTensorRow>,
    constructed_components: Vec<String>,
) -> PyResult<Bound<'p, PyBytes>> {
    if tensors.is_empty() {
        return tensorfs_core::err::refuse(
            tensorfs_core::err::Code::EMPTY_COMPONENTS,
            "CodeTopology requires at least one tensor",
        )
        .or_refuse(py);
    }
    let mut components: Vec<(String, Vec<(String, Requirement)>)> = Vec::new();
    let mut prior: Option<(String, String)> = None;
    for (component, key, dtype, shape, accepts, pinned) in tensors {
        let identity = (component.clone(), key.clone());
        if prior.as_ref().is_some_and(|held| held >= &identity) {
            return tensorfs_core::err::refuse(
                tensorfs_core::err::Code::SORT_ORDER,
                "CodeTensor rows must be sorted and unique by (component,key)",
            )
            .or_refuse(py);
        }
        prior = Some(identity);
        let requirement = Requirement {
            dtype: Dtype::parse(&dtype).or_refuse(py)?,
            shape,
            accepts: accepts.map(Accepts::OneOf).unwrap_or(Accepts::Any),
            pinned,
        };
        match components.last_mut() {
            Some((held, keys)) if held == &component => keys.push((key, requirement)),
            _ => components.push((component, vec![(key, requirement)])),
        }
    }
    let topology = CodeTopology {
        contract: contract_digest.to_string(),
        components,
        constructed_components,
    };
    let raw = topology.canonical_bytes();
    CodeTopology::parse(&raw).or_refuse(py)?;
    Ok(PyBytes::new(py, &raw))
}

/// `tensorfs.fit(code_topology, header, *, custody, device=None, observations=None)`:
/// the one code <-> checkpoint answer (design §3), as the same total document `tfs fit
/// --json` prints, plus `ok`.
#[pyfunction(name = "fit")]
#[pyo3(signature = (code_topology, header, *, custody, device=None, observations=None))]
pub fn py_fit<'p>(
    py: Python<'p>,
    code_topology: &[u8],
    header: &[u8],
    custody: &str,
    device: Option<&str>,
    observations: Option<&[u8]>,
) -> PyResult<Bound<'p, PyDict>> {
    let code = CodeTopology::parse(code_topology).or_refuse(py)?;
    let header = Header::parse(header).or_refuse(py)?;
    let custody = Custody::parse(custody).or_refuse(py)?;
    let observed = match observations {
        Some(bytes) => Some(CapabilityRecords::parse_observed(bytes).or_refuse(py)?),
        None => None,
    };
    let verdict = tensorfs_core::fit::fit(&code, &header, custody, device, observed.as_ref())
        .or_refuse(py)?;
    let doc = tensorfs_core::machine::fit_json(&verdict, custody);
    let value = tensorfs_core::canon::parse_canonical(&doc, tensorfs_core::limits::DOC_MAX_BYTES)
        .or_refuse(py)?;
    let out = marshal(py, &value)?.cast_into::<PyDict>()?;
    out.set_item("ok", verdict.is_ok())?;
    Ok(out)
}

/// `tensorfs.capability(encoding, device, observations=None)`: is this (encoding, device)
/// pair qualified — the compiled-in reference records widened by what a card observed.
#[pyfunction]
#[pyo3(signature = (encoding, device, observations=None))]
pub fn capability<'p>(
    py: Python<'p>,
    encoding: &str,
    device: &str,
    observations: Option<&[u8]>,
) -> PyResult<Bound<'p, PyDict>> {
    let mut records = registry::capability_records();
    if let Some(bytes) = observations {
        records = records.with_observed(CapabilityRecords::parse_observed(bytes).or_refuse(py)?);
    }
    let encoding = registry::seeds()
        .iter()
        .find(|s| s.alias == encoding)
        .map(|s| s.spec.object_id())
        .unwrap_or_else(|| encoding.to_string());
    let decision = records.admit(&encoding, device);
    let qualified: Vec<&str> = records
        .records
        .iter()
        .filter(|r| r.encoding == encoding)
        .map(|r| r.device.as_str())
        .collect();
    let (record, refusal) = match &decision {
        Ok(r) => (Some(*r), String::new()),
        Err(e) => (None, e.detail.clone()),
    };
    let doc =
        tensorfs_core::machine::capability_json(record, &encoding, device, &refusal, &qualified);
    let value = tensorfs_core::canon::parse_canonical(&doc, tensorfs_core::limits::DOC_MAX_BYTES)
        .or_refuse(py)?;
    Ok(marshal(py, &value)?.cast_into::<PyDict>()?)
}

/// Canonical `Value` -> plain Python data. The document's shape survives; its Rust types do
/// not cross, so no Python-side schema knowledge is needed or possible.
pub fn marshal<'p>(py: Python<'p>, v: &Value) -> PyResult<Bound<'p, PyAny>> {
    Ok(match v {
        Value::Bool(b) => b.into_pyobject(py)?.to_owned().into_any(),
        Value::Int(i) => i.into_pyobject(py)?.into_any(),
        Value::Str(s) => s.into_pyobject(py)?.into_any(),
        Value::Arr(a) => {
            let l = PyList::empty(py);
            for x in a {
                l.append(marshal(py, x)?)?;
            }
            l.into_any()
        }
        Value::Obj(m) => {
            let d = PyDict::new(py);
            for (k, x) in m {
                d.set_item(k, marshal(py, x)?)?;
            }
            d.into_any()
        }
    })
}

/// Header -> ergonomic lookup dictionaries. Stored order survives insertion order, while
/// CBOR bytes and integer dtype/index codes stay private to TensorFS.
fn marshal_header<'p>(py: Python<'p>, header: &Header) -> PyResult<Bound<'p, PyAny>> {
    let root = PyDict::new(py);
    root.set_item("format", Header::FORMAT)?;

    let configs = PyDict::new(py);
    for (name, config) in &header.configs {
        let row = PyDict::new(py);
        row.set_item("sha256", &config.sha256)?;
        row.set_item("length", config.length)?;
        row.set_item("media_type", &config.media_type)?;
        configs.set_item(name, row)?;
    }
    root.set_item("configs", configs)?;

    let assets = PyDict::new(py);
    for (name, asset) in &header.assets {
        let row = PyDict::new(py);
        row.set_item("logical_sha256", &asset.logical_sha256)?;
        row.set_item("logical_length", asset.logical_length)?;
        row.set_item("media_type", &asset.media_type)?;
        let segments = PyList::empty(py);
        for segment in &asset.segments {
            let item = PyDict::new(py);
            item.set_item("sha256", &segment.sha256)?;
            item.set_item("length", segment.length)?;
            segments.append(item)?;
        }
        row.set_item("segments", segments)?;
        assets.set_item(name, row)?;
    }
    root.set_item("assets", assets)?;

    let encodings = PyList::empty(py);
    for encoding in &header.encodings {
        let row = PyDict::new(py);
        row.set_item("id", encoding.object_id())?;
        row.set_item("definition", marshal(py, &encoding.to_value())?)?;
        encodings.append(row)?;
    }
    root.set_item("encodings", encodings)?;

    let components = PyDict::new(py);
    for (component, tensors) in &header.components {
        let table = PyDict::new(py);
        for (key, tensor) in tensors {
            let row = PyDict::new(py);
            let logical = PyDict::new(py);
            logical.set_item("dtype", tensor.dtype.name())?;
            logical.set_item("shape", &tensor.shape)?;
            row.set_item("logical", logical)?;
            row.set_item("encoding", &tensor.encoding)?;
            let parts = PyDict::new(py);
            for (role, part) in &tensor.parts {
                let item = PyDict::new(py);
                item.set_item("dtype", part.dtype.name())?;
                item.set_item("shape", &part.shape)?;
                match &part.body {
                    tensorfs_core::header::Body::Inline(bytes) => {
                        item.set_item("inline", PyBytes::new(py, bytes))?;
                    }
                    tensorfs_core::header::Body::Segments(values) => {
                        let segments = PyList::empty(py);
                        for segment in values {
                            let value = PyDict::new(py);
                            value.set_item("sha256", &segment.sha256)?;
                            value.set_item("length", segment.length)?;
                            segments.append(value)?;
                        }
                        item.set_item("segments", segments)?;
                    }
                }
                parts.set_item(role, item)?;
            }
            row.set_item("parts", parts)?;
            table.set_item(key, row)?;
        }
        components.set_item(component, table)?;
    }
    root.set_item("components", components)?;
    Ok(root.into_any())
}

/// The encoding closure a header validates against: spec objects by digest.
#[pyclass(name = "Closure")]
pub struct PyClosure {
    pub inner: CoreClosure,
}

#[pymethods]
impl PyClosure {
    #[new]
    fn new() -> Self {
        PyClosure {
            inner: CoreClosure::default(),
        }
    }

    /// Add one exact nested encoding value from a qualification fixture.
    fn insert(&mut self, py: Python<'_>, data: &[u8]) -> PyResult<String> {
        let s = EncodingSpec::decode_nested(data).or_refuse(py)?;
        let id = s.object_id();
        self.inner.insert(s);
        Ok(id)
    }

    fn __len__(&self) -> usize {
        self.inner.specs.len()
    }

    fn digests(&self) -> Vec<String> {
        self.inner.specs.iter().map(|(k, _)| k.clone()).collect()
    }
}

/// `sha256:<hex>` of exactly these bytes — the universal object identity.
#[pyfunction]
pub fn object_id(data: &[u8]) -> String {
    ids::object_id(data)
}

/// Validate canonical CozyTensorsHeader bytes; return them as plain Python data.
#[pyfunction]
#[pyo3(signature = (data, closure=None))]
pub fn parse_header<'p>(
    py: Python<'p>,
    data: &[u8],
    closure: Option<&PyClosure>,
) -> PyResult<Bound<'p, PyAny>> {
    let h = Header::parse(data).or_refuse(py)?;
    if let Some(c) = closure {
        h.validate(&c.inner).or_refuse(py)?;
    }
    marshal_header(py, &h)
}

/// The frozen topology projection of a header — logical dtype included, encodings not.
#[pyfunction]
pub fn topology_digest(py: Python<'_>, data: &[u8]) -> PyResult<String> {
    Ok(Header::parse(data).or_refuse(py)?.topology_digest())
}

/// The FIT index (D4): the same projection with the dtype leaf elided. Two checkpoints with
/// equal `shape_digest` fit the same code.
#[pyfunction]
pub fn shape_digest(py: Python<'_>, data: &[u8]) -> PyResult<String> {
    Ok(Header::parse(data).or_refuse(py)?.shape_digest())
}

fn config_bytes(configs: &Bound<'_, PyDict>) -> PyResult<Vec<(String, Vec<u8>)>> {
    configs
        .iter()
        .map(|(name, bytes)| Ok((name.extract::<String>()?, bytes.extract::<Vec<u8>>()?)))
        .collect()
}

/// The construction document (model-code-fit §2) of a header, from the exact bytes of every
/// config it names (`{name: bytes}`): the RFC 8785 bytes whose sha256 is `config_digest`,
/// or `None` when the header names no config.
#[pyfunction]
pub fn construction_document<'p>(
    py: Python<'p>,
    data: &[u8],
    configs: &Bound<'p, PyDict>,
) -> PyResult<Option<Bound<'p, PyBytes>>> {
    let header = Header::parse(data).or_refuse(py)?;
    let document = header
        .construction_document(&config_bytes(configs)?)
        .or_refuse(py)?;
    Ok(document.map(|bytes| PyBytes::new(py, &bytes)))
}

/// The construction document's identity, or `None` when the header names no config.
#[pyfunction]
pub fn config_digest(
    py: Python<'_>,
    data: &[u8],
    configs: &Bound<'_, PyDict>,
) -> PyResult<Option<String>> {
    let header = Header::parse(data).or_refuse(py)?;
    Ok(header
        .digests(&config_bytes(configs)?)
        .or_refuse(py)?
        .config_digest)
}

/// All three catalog digests of a header: `{"topology_digest", "shape_digest",
/// "config_digest"}` — what `tfs ingest` stores on the checkpoint row. `configs` maps
/// each config name the header declares to its exact bytes.
#[pyfunction]
pub fn digests<'p>(
    py: Python<'p>,
    data: &[u8],
    configs: &Bound<'p, PyDict>,
) -> PyResult<Bound<'p, PyDict>> {
    let header = Header::parse(data).or_refuse(py)?;
    let d = header.digests(&config_bytes(configs)?).or_refuse(py)?;
    let out = PyDict::new(py);
    out.set_item("topology_digest", d.topology_digest)?;
    out.set_item("shape_digest", d.shape_digest)?;
    out.set_item("config_digest", d.config_digest)?;
    Ok(out)
}

/// Validate canonical Manifest bytes.
#[pyfunction]
pub fn parse_manifest<'p>(py: Python<'p>, data: &[u8]) -> PyResult<Bound<'p, PyAny>> {
    let m = Manifest::parse(data).or_refuse(py)?;
    marshal(py, &m.to_value())
}

#[pyfunction]
#[pyo3(signature = (doc, data, closure=None, platform=None))]
pub fn run_doc<'p>(
    py: Python<'p>,
    doc: &str,
    data: &[u8],
    closure: Option<&PyClosure>,
    platform: Option<Vec<String>>,
) -> PyResult<Bound<'p, PyDict>> {
    let empty = CoreClosure::default();
    let closure = closure.map(|closure| &closure.inner).unwrap_or(&empty);
    let platform = platform.unwrap_or_else(registry::platform_digests);
    let outcome = corpus::run_doc(doc, data, closure, &platform).or_refuse(py)?;
    let result = PyDict::new(py);
    result.set_item("id", outcome.id)?;
    result.set_item("topology", outcome.topology)?;
    result.set_item("shape", outcome.shape)?;
    result.set_item("manifest", outcome.manifest)?;
    result.set_item("grade", outcome.grade)?;
    Ok(result)
}

#[pyfunction]
pub fn receipt_binds(
    py: Python<'_>,
    receipt: &[u8],
    subject: &[u8],
    manifest: &[u8],
    header: &[u8],
    stamp: &[u8],
) -> PyResult<()> {
    corpus::receipt_binds(receipt, subject, manifest, header, stamp).or_refuse(py)
}

#[pyfunction]
pub fn doc_kinds() -> Vec<String> {
    corpus::DOC_KINDS
        .iter()
        .map(|kind| kind.to_string())
        .collect()
}

/// The platform-aliased encoding digests this build admits at the border.
#[pyfunction]
pub fn platform_digests() -> Vec<String> {
    registry::platform_digests()
}

/// (alias, spec digest) for every seeded encoding. Aliases are display; digests are identity.
#[pyfunction]
pub fn seed_digests() -> Vec<(String, String)> {
    registry::seeds()
        .into_iter()
        .map(|s| (s.alias.to_string(), s.spec.object_id()))
        .collect()
}

/// The closed dtype enum and its element sizes, exported EXACTLY ONCE. A consumer that
/// needs `torch.frombuffer` gets its item size from here, never from a table of its own.
#[pyfunction]
pub fn dtypes(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    for name in [
        "f64",
        "f32",
        "f16",
        "bf16",
        "f8_e4m3fn",
        "f8_e5m2",
        "i64",
        "i32",
        "i16",
        "i8",
        "u8",
        "bool",
    ] {
        let dt = Dtype::parse(name).or_refuse(py)?;
        d.set_item(dt.name(), dt.size())?;
    }
    Ok(d)
}

#[pyfunction]
pub fn manifest_max_bytes() -> usize {
    Manifest::MAX_BYTES
}

/// Bytes -> the pretty PROJECTION of a canonical document. Projection-only by construction:
/// what comes back is a string, and no reader in this system accepts it.
#[pyfunction]
pub fn render(py: Python<'_>, data: &[u8], max_bytes: usize) -> PyResult<String> {
    tensorfs_core::project::render(data, max_bytes).or_refuse(py)
}

/// Canonical re-emission: parse and write back. `sha256(stored) == object_id` universally,
/// so a document that does not survive this was never canonical.
#[pyfunction]
pub fn recanonicalize<'p>(py: Python<'p>, doc: &str, data: &[u8]) -> PyResult<Bound<'p, PyBytes>> {
    let out = match doc {
        "header" => Header::parse(data)
            .or_refuse(py)?
            .canonical_bytes()
            .or_refuse(py)?,
        "manifest" => Manifest::parse(data).or_refuse(py)?.canonical_bytes(),
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "recanonicalize does not carry {other:?}"
            )))
        }
    };
    Ok(PyBytes::new(py, &out))
}

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
use tensorfs_core::fit::{Custody, TensorRequirements};
use tensorfs_core::header::{Closure as CoreClosure, Header};
use tensorfs_core::ids::{self, Doc};
use tensorfs_core::manifest::Manifest;
use tensorfs_core::registry;
use tensorfs_core::spec::EncodingSpec;

use crate::errors::IntoPy;

type TensorRequirementRow = (String, String, Vec<u64>, Option<String>);

/// Compare a request-time meta-device census with a CozyTensors header. Requirements never
/// become bytes or a digest; the result is the only value crossing this boundary.
#[pyfunction(name = "_fit")]
#[pyo3(signature = (tensors, header, *, custody, encoded_leaves, device=None, observations=None))]
pub fn py_fit<'p>(
    py: Python<'p>,
    tensors: Vec<TensorRequirementRow>,
    header: &[u8],
    custody: &str,
    encoded_leaves: bool,
    device: Option<&str>,
    observations: Option<&[u8]>,
) -> PyResult<Bound<'p, PyDict>> {
    let requirements = TensorRequirements::new(
        tensors
            .into_iter()
            .map(|(component, key, shape, logical_dtype)| {
                Ok((
                    component,
                    key,
                    shape,
                    match logical_dtype {
                        Some(value) => Some(Dtype::parse(&value)?),
                        None => None,
                    },
                ))
            })
            .collect::<tensorfs_core::err::Result<Vec<_>>>()
            .or_refuse(py)?,
    )
    .or_refuse(py)?;
    let header = Header::parse(header).or_refuse(py)?;
    let custody = Custody::parse(custody).or_refuse(py)?;
    let observed = match observations {
        Some(bytes) => Some(CapabilityRecords::parse_observed(bytes).or_refuse(py)?),
        None => None,
    };
    let verdict = tensorfs_core::fit::fit(
        &requirements,
        &header,
        custody,
        encoded_leaves,
        device,
        observed.as_ref(),
    )
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
        configs.set_item(name, PyBytes::new(py, config))?;
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
            logical.set_item("logical_dtype", tensor.dtype.name())?;
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
    result.set_item("tensor_schema", outcome.tensor_schema)?;
    result.set_item("manifest", outcome.manifest)?;
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

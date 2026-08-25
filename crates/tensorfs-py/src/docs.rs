//! Document marshalling: canonical bytes in, plain Python data out.
//!
//! Nothing here decides anything. Every function hands bytes to the ONE reader in
//! `tensorfs-core` and turns its result into dicts, lists, ints and strings. Python never
//! sees a partially-validated document: a refusal is raised before any data is built.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use tensorfs_core::canon::Value;
use tensorfs_core::corpus;
use tensorfs_core::dtype::Dtype;
use tensorfs_core::header::{Closure as CoreClosure, Header};
use tensorfs_core::ids::{self, Doc};
use tensorfs_core::manifest::SnapshotManifest;
use tensorfs_core::registry;
use tensorfs_core::spec::EncodingSpec;

use crate::errors::IntoPy;

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

    /// Add one canonical EncodingSpec object. Its digest is its identity — computed here,
    /// never supplied.
    fn insert(&mut self, py: Python<'_>, data: &[u8]) -> PyResult<String> {
        let s = EncodingSpec::parse(data).or_refuse(py)?;
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
    marshal(py, &h.to_value())
}

/// The frozen topology projection of a header — logical dtype included, encodings not.
#[pyfunction]
pub fn topology_digest(py: Python<'_>, data: &[u8]) -> PyResult<String> {
    Ok(Header::parse(data).or_refuse(py)?.topology_digest())
}

/// Validate canonical SnapshotManifest bytes; `checkpoint_id == snapshot_id`, always.
#[pyfunction]
pub fn parse_manifest<'p>(py: Python<'p>, data: &[u8]) -> PyResult<Bound<'p, PyAny>> {
    let m = SnapshotManifest::parse(data).or_refuse(py)?;
    marshal(py, &m.to_value())
}

/// The corpus dispatch (tfs-013), shared with the Rust harness: one document kind, one
/// reader. Returns the identities the frozen verdict names; raises the typed refusal.
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
    let c = closure.map(|c| &c.inner).unwrap_or(&empty);
    let p = platform.unwrap_or_else(registry::platform_digests);
    let o = corpus::run_doc(doc, data, c, &p).or_refuse(py)?;
    let d = PyDict::new(py);
    d.set_item("id", o.id)?;
    d.set_item("topology", o.topology)?;
    d.set_item("checkpoint", o.checkpoint)?;
    Ok(d)
}

/// The receipt binds its exact subject/snapshot/header/stamp — the cross-vector assertion.
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
    corpus::DOC_KINDS.iter().map(|s| s.to_string()).collect()
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
        "header" => Header::parse(data).or_refuse(py)?.canonical_bytes(),
        "manifest" => SnapshotManifest::parse(data)
            .or_refuse(py)?
            .canonical_bytes(),
        "spec" => EncodingSpec::parse(data).or_refuse(py)?.canonical_bytes(),
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "recanonicalize does not carry {other:?}"
            )))
        }
    };
    Ok(PyBytes::new(py, &out))
}

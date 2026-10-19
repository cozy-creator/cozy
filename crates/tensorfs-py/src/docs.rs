//! Document marshalling: canonical bytes in, plain Python data out.
//!
//! Nothing here decides anything. Every function hands bytes to the ONE reader in
//! `tensorfs` and turns its result into dicts, lists, ints and strings. Python never
//! sees a partially-validated document: a refusal is raised before any data is built.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use tensorfs::canon::Value;
use tensorfs::header::{Closure as CoreClosure, Header};
use tensorfs::spec::EncodingSpec;

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
                    tensorfs::header::Body::Inline(bytes) => {
                        item.set_item("inline", PyBytes::new(py, bytes))?;
                    }
                    tensorfs::header::Body::Segments(values) => {
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

/// Decode admitted CozyTensorsHeader bytes; return them as plain Python data.
#[pyfunction]
#[pyo3(signature = (data, closure=None))]
pub fn parse_header<'p>(
    py: Python<'p>,
    data: &[u8],
    closure: Option<&PyClosure>,
) -> PyResult<Bound<'p, PyAny>> {
    let h = Header::from_stored(data).or_refuse(py)?;
    if let Some(c) = closure {
        h.validate(&c.inner).or_refuse(py)?;
    }
    marshal_header(py, &h)
}

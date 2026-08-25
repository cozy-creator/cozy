//! tensorfs-py — the marshal boundary (tensorfs.md §10). The compiled half owns every
//! byte-level fact; Python owns none. Deliberately tiny: tfs-007 owns the full facade.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tensorfs_core::canon::Value;
use tensorfs_core::err::Refusal;
use tensorfs_core::header::Header;
use tensorfs_core::ids::{self, Doc};

fn refused(e: Refusal) -> PyErr {
    PyValueError::new_err(format!("{}: {}", e.code.as_str(), e.detail))
}

/// Canonical Value -> plain Python data. No Python-side schema knowledge required.
fn marshal<'p>(py: Python<'p>, v: &Value) -> PyResult<Bound<'p, PyAny>> {
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

/// `sha256:<hex>` of exactly these bytes — the universal object identity.
#[pyfunction]
fn object_id(data: &[u8]) -> String {
    ids::object_id(data)
}

/// Validate canonical CozyTensorsHeader bytes; return them as plain Python data.
#[pyfunction]
fn parse_header<'p>(py: Python<'p>, data: &[u8]) -> PyResult<Bound<'p, PyAny>> {
    let h = Header::parse(data).map_err(refused)?;
    marshal(py, &h.to_value())
}

#[pymodule]
fn _ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(object_id, m)?)?;
    m.add_function(wrap_pyfunction!(parse_header, m)?)?;
    Ok(())
}

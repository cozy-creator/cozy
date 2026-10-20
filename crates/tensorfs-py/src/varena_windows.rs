//! CPU-only wheels keep the placement import surface, without pretending to provide a GPU
//! ledger on Windows. TensorFS reading remains available independently.
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

create_exception!(
    tensorfs.varena,
    VarenaError,
    PyException,
    "A placement error; code and sticky describe recovery."
);
create_exception!(tensorfs.varena, Unsupported, VarenaError);
fn unavailable(py: Python<'_>) -> PyErr {
    let error = Unsupported::new_err("GPU placement has no Windows tier yet");
    let _ = error.value(py).setattr("code", "UNSUPPORTED");
    let _ = error.value(py).setattr("sticky", false);
    error
}
macro_rules! unavailable_type {
    ($($rust:ident=$name:literal),* $(,)?)=>{$(
        #[pyclass(name=$name,module="tensorfs.varena",frozen)]
        struct $rust;
        #[pymethods]
        impl $rust {
            #[new]
            #[pyo3(signature=(*_args,**_kwargs))]
            fn new(py:Python<'_>,_args:&Bound<'_,PyTuple>,_kwargs:Option<&Bound<'_,PyDict>>)->PyResult<Self>{Err(unavailable(py))}
        }
    )*};
}
unavailable_type!(
    Plane = "Plane",
    WeightSet = "WeightSet",
    Pinned = "Pinned",
    Part = "Part",
    Copy = "Copy",
    Ceiling = "Ceiling"
);
#[pyfunction]
fn copies() -> Vec<Py<PyAny>> {
    Vec::new()
}
#[pyfunction]
#[pyo3(signature = (device=None, path=None))]
fn ceiling(py: Python<'_>, device: Option<i32>, path: Option<std::path::PathBuf>) -> PyResult<()> {
    let _ = (device, path);
    Err(unavailable(py))
}
pub fn install(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "varena")?;
    m.add_class::<Plane>()?;
    m.add_class::<WeightSet>()?;
    m.add_class::<Pinned>()?;
    m.add_class::<Part>()?;
    m.add_class::<Copy>()?;
    m.add_class::<Ceiling>()?;
    m.add_function(wrap_pyfunction!(copies, &m)?)?;
    m.add_function(wrap_pyfunction!(ceiling, &m)?)?;
    m.add("VarenaError", py.get_type::<VarenaError>())?;
    m.add("Unsupported", py.get_type::<Unsupported>())?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("tensorfs._ext.varena", &m)
}

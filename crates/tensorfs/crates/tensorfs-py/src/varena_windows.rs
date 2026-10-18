//! `tensorfs.varena` on Windows: varena is a Unix mechanism (weight sets cross processes as
//! descriptors) and its Windows tier does not exist yet (xs-043 unit WM). Every name imports;
//! constructing or calling one raises `Unsupported`.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

create_exception!(tensorfs.varena, VarenaError, PyException, "A varena error; `.code` names its class.");
create_exception!(tensorfs.varena, DeviceFull, VarenaError);
create_exception!(tensorfs.varena, CudaError, VarenaError);
create_exception!(tensorfs.varena, CudaUnavailable, VarenaError);
create_exception!(tensorfs.varena, Invalid, VarenaError);
create_exception!(tensorfs.varena, IoError, VarenaError);
create_exception!(tensorfs.varena, Closed, VarenaError);
create_exception!(tensorfs.varena, Unsupported, VarenaError);

fn unsupported(py: Python<'_>) -> PyErr {
    let error = Unsupported::new_err("varena has no Windows tier yet; read objects through Store leases");
    let _ = error.value(py).setattr("code", "UNSUPPORTED");
    error
}

macro_rules! unconstructible {
    ($($rust:ident = $name:literal),*) => {$(
        #[pyclass(name = $name, module = "tensorfs.varena", frozen)]
        struct $rust;

        #[pymethods]
        impl $rust {
            #[new]
            #[pyo3(signature = (*_args, **_kwargs))]
            fn new(py: Python<'_>, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
                Err(unsupported(py))
            }
        }
    )*};
}

unconstructible!(Gpu = "Gpu", Host = "Host", WeightSet = "WeightSet", Pinned = "Pinned", Part = "Part",
    Counters = "Counters", SetStats = "SetStats", GpuStats = "GpuStats");

macro_rules! refused {
    ($($name:ident),*) => {$(
        #[pyfunction]
        #[pyo3(signature = (*_args, **_kwargs))]
        fn $name(py: Python<'_>, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
            Err(unsupported(py))
        }
    )*};
}

refused!(hold_tier, release_tier, reserve, fill_sealed, seal);

pub fn install(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "varena")?;
    m.add_class::<Gpu>()?;
    m.add_class::<Host>()?;
    m.add_class::<WeightSet>()?;
    m.add_class::<Pinned>()?;
    m.add_class::<Part>()?;
    m.add_class::<Counters>()?;
    m.add_class::<SetStats>()?;
    m.add_class::<GpuStats>()?;
    m.add_function(wrap_pyfunction!(hold_tier, &m)?)?;
    m.add_function(wrap_pyfunction!(release_tier, &m)?)?;
    m.add_function(wrap_pyfunction!(reserve, &m)?)?;
    m.add_function(wrap_pyfunction!(fill_sealed, &m)?)?;
    m.add_function(wrap_pyfunction!(seal, &m)?)?;
    macro_rules! add_errors {
        ($($e:ident),*) => { $(m.add(stringify!($e), py.get_type::<$e>())?;)* };
    }
    add_errors!(VarenaError, DeviceFull, CudaError, CudaUnavailable, Invalid, IoError, Closed, Unsupported);
    parent.add_submodule(&m)?;
    py.import("sys")?.getattr("modules")?.set_item("tensorfs._ext.varena", &m)
}

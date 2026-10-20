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
create_exception!(tensorfs.varena, LedgerError, PyException, "A ledger error; `.code` names its class.");

fn unsupported(py: Python<'_>) -> PyErr {
    let error = Unsupported::new_err("varena has no Windows tier yet; read tensors through tensorfs.open");
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

unconstructible!(PyGpu = "Gpu", PyHost = "Host", PyWeightSet = "WeightSet", PyPinned = "Pinned", PyPart = "Part",
    PyCounters = "Counters", PySetStats = "SetStats", PyGpuStats = "GpuStats", PyCopy = "Copy", PyLedger = "Ledger",
    PyRow = "Row", PyVictim = "Victim", PyAdmit = "Admit", PyRowFacts = "RowFacts");

macro_rules! refused {
    ($($name:ident),*) => {$(
        #[pyfunction]
        #[pyo3(signature = (*_args, **_kwargs))]
        fn $name(py: Python<'_>, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
            Err(unsupported(py))
        }
    )*};
}

refused!(copies, hold_tier, release_tier, seal);

pub fn install(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "varena")?;
    macro_rules! add {
        ($($c:ty),*; $($f:ident),*; $($e:ident),*) => {
            $(m.add_class::<$c>()?;)*
            $(m.add_function(wrap_pyfunction!($f, &m)?)?;)*
            $(m.add(stringify!($e), py.get_type::<$e>())?;)*
        };
    }
    add!(PyGpu, PyHost, PyWeightSet, PyPinned, PyPart, PyCounters, PySetStats, PyGpuStats, PyCopy, PyLedger, PyRow, PyVictim, PyAdmit, PyRowFacts;
        copies, hold_tier, release_tier, seal;
        VarenaError, DeviceFull, CudaError, CudaUnavailable, Invalid, IoError, Closed, Unsupported, LedgerError);
    #[cfg(feature = "store")]
    m.add_function(wrap_pyfunction!(crate::store::regions, &m)?)?;
    parent.add_submodule(&m)?;
    py.import("sys")?.getattr("modules")?.set_item("tensorfs._ext.varena", &m)
}

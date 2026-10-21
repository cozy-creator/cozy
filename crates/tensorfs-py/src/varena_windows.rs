//! `tensorfs.varena`'s GPU core, ledger and host tier off Unix, where they do not exist yet
//! (xs-061 W8): every name imports; constructing or calling one raises `Unsupported`. The
//! host-only core (`Host`) is real everywhere.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::varena::Unsupported;

fn unsupported(py: Python<'_>) -> PyErr {
    let error = Unsupported::new_err("varena has no GPU tier, ledger or host tier on this OS yet");
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

unconstructible!(Gpu = "Gpu", Counters = "Counters", SetStats = "SetStats", GpuStats = "GpuStats", Copy = "Copy",
    Row = "Row", Victim = "Victim", Admit = "Admit", RowFacts = "RowFacts");

#[pyclass(name = "Ledger", module = "tensorfs.varena", frozen)]
struct Ledger;

#[pymethods]
impl Ledger {
    #[staticmethod]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn create(py: Python<'_>, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        Err(unsupported(py))
    }

    #[staticmethod]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn adopt(py: Python<'_>, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        Err(unsupported(py))
    }
}

macro_rules! refused {
    ($($name:ident),*) => {$(
        #[pyfunction]
        #[pyo3(signature = (*_args, **_kwargs))]
        fn $name(py: Python<'_>, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
            Err(unsupported(py))
        }
    )*};
}

refused!(hold_tier, release_tier, seal);

/// No GPU tier, so no weight copies to report.
#[pyfunction]
fn copies(py: Python<'_>) -> Bound<'_, pyo3::types::PyList> {
    pyo3::types::PyList::empty(py)
}

pub fn install(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Gpu>()?;
    m.add_class::<Counters>()?;
    m.add_class::<SetStats>()?;
    m.add_class::<GpuStats>()?;
    m.add_class::<Copy>()?;
    m.add_class::<Ledger>()?;
    m.add_class::<Row>()?;
    m.add_class::<Victim>()?;
    m.add_class::<Admit>()?;
    m.add_class::<RowFacts>()?;
    for f in [wrap_pyfunction!(hold_tier, m)?, wrap_pyfunction!(release_tier, m)?, wrap_pyfunction!(seal, m)?, wrap_pyfunction!(copies, m)?] {
        m.add_function(f)?;
    }
    Ok(())
}

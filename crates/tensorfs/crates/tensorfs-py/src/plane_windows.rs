//! `tensorfs.plane` on Windows: the plane is a Unix mechanism (weight sets cross processes as
//! descriptors) and its Windows tier does not exist yet (xs-043 unit WM). Every name imports;
//! constructing or calling one raises `Unsupported`.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

create_exception!(tensorfs.plane, PlaneError, PyException, "A weight-plane error; `.code` names its class.");
create_exception!(tensorfs.plane, BudgetExceeded, PlaneError);
create_exception!(tensorfs.plane, Shortfall, PlaneError);
create_exception!(tensorfs.plane, BelowFloor, PlaneError);
create_exception!(tensorfs.plane, LeaseViolation, PlaneError);
create_exception!(tensorfs.plane, Poisoned, PlaneError);
create_exception!(tensorfs.plane, CudaError, PlaneError);
create_exception!(tensorfs.plane, CudaUnavailable, PlaneError);
create_exception!(tensorfs.plane, Invalid, PlaneError);
create_exception!(tensorfs.plane, IoError, PlaneError);
create_exception!(tensorfs.plane, Closed, PlaneError);
create_exception!(tensorfs.plane, Unsupported, PlaneError);

fn unsupported(py: Python<'_>) -> PyErr {
    let error = Unsupported::new_err("the weight plane has no Windows tier yet; read objects through Store leases");
    let _ = error.value(py).setattr("code", "UNSUPPORTED");
    error
}

macro_rules! unconstructible {
    ($($rust:ident = $name:literal),*) => {$(
        #[pyclass(name = $name, module = "tensorfs.plane", frozen)]
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

unconstructible!(Plane = "Plane", Source = "Source", WeightSet = "WeightSet", Ticket = "Ticket",
    Lease = "Lease", Cursor = "Cursor", DeviceView = "DeviceView");

macro_rules! refused {
    ($($name:ident),*) => {$(
        #[pyfunction]
        #[pyo3(signature = (*_args, **_kwargs))]
        fn $name(py: Python<'_>, _args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
            Err(unsupported(py))
        }
    )*};
}

refused!(hold_tier, release_tier, reserve, fill_sealed);

pub fn install(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "plane")?;
    m.add_class::<Plane>()?;
    m.add_class::<Source>()?;
    m.add_class::<WeightSet>()?;
    m.add_class::<Ticket>()?;
    m.add_class::<Lease>()?;
    m.add_class::<Cursor>()?;
    m.add_class::<DeviceView>()?;
    m.add_function(wrap_pyfunction!(hold_tier, &m)?)?;
    m.add_function(wrap_pyfunction!(release_tier, &m)?)?;
    m.add_function(wrap_pyfunction!(reserve, &m)?)?;
    m.add_function(wrap_pyfunction!(fill_sealed, &m)?)?;
    m.add("SEALED_WHILE_FILLING", false)?;
    m.add("SEALED_HOLES", false)?;
    macro_rules! add_errors {
        ($($e:ident),*) => { $(m.add(stringify!($e), py.get_type::<$e>())?;)* };
    }
    add_errors!(PlaneError, BudgetExceeded, Shortfall, BelowFloor, LeaseViolation, Poisoned, CudaError,
        CudaUnavailable, Invalid, IoError, Closed, Unsupported);
    parent.add_submodule(&m)?;
    py.import("sys")?.getattr("modules")?.set_item("tensorfs._ext.plane", &m)
}

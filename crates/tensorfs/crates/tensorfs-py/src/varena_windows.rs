//! `tensorfs.varena` on Windows: the plane is a Unix mechanism (a sealed host layout crosses
//! processes as a memfd) and its Windows tier does not exist yet (xs-043 unit WM). Every name
//! imports; constructing one raises `Invalid`.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

create_exception!(tensorfs.varena, PlaneError, PyException, "A weight-plane error; `.code` names its class.");
create_exception!(tensorfs.varena, DeviceFull, PlaneError);
create_exception!(tensorfs.varena, Floor, PlaneError);
create_exception!(tensorfs.varena, CudaError, PlaneError);
create_exception!(tensorfs.varena, Closed, PlaneError);
create_exception!(tensorfs.varena, Invalid, PlaneError);
create_exception!(tensorfs.varena, Poisoned, PlaneError);
create_exception!(tensorfs.varena, IoError, PlaneError);

fn unsupported(py: Python<'_>) -> PyErr {
    let error = Invalid::new_err("the weight plane has no Windows tier yet; read tensors through tensorfs.v2");
    let _ = error.value(py).setattr("code", "INVALID");
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

unconstructible!(Plane = "Plane", PageCache = "PageCache", SealedLayout = "SealedLayout", WeightSet = "WeightSet",
    Ticket = "Ticket", Lease = "Lease", Cursor = "Cursor", DeviceView = "DeviceView");

pub fn install(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "varena")?;
    m.add_class::<Plane>()?;
    m.add_class::<PageCache>()?;
    m.add_class::<SealedLayout>()?;
    m.add_class::<WeightSet>()?;
    m.add_class::<Ticket>()?;
    m.add_class::<Lease>()?;
    m.add_class::<Cursor>()?;
    m.add_class::<DeviceView>()?;
    macro_rules! add_errors {
        ($($e:ident),*) => { $(m.add(stringify!($e), py.get_type::<$e>())?;)* };
    }
    add_errors!(PlaneError, DeviceFull, Floor, CudaError, Closed, Invalid, Poisoned, IoError);
    parent.add_submodule(&m)?;
    py.import("sys")?.getattr("modules")?.set_item("tensorfs._ext.varena", &m)
}

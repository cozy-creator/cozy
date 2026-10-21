//! `tensorfs.ClassRef`: a header's class record, `[library, version, name]`.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use tensorfs::format::ClassRef;

/// A class as `(library, version, name)`, for any implementation and never an import path:
/// `ClassRef("diffusers", "0.35.1", "SD3Transformer2DModel")`. `ClassRef("", "", "")` is an
/// unknown model; `version` is `""` when nothing records it.
#[pyclass(name = "ClassRef", module = "tensorfs", frozen, eq, hash, get_all, skip_from_py_object)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PyClassRef {
    library: String,
    version: String,
    name: String,
}

impl From<&ClassRef> for PyClassRef {
    fn from(c: &ClassRef) -> PyClassRef {
        PyClassRef { library: c.library.clone(), version: c.version.clone(), name: c.name.clone() }
    }
}

/// A `ClassRef` or a `(library, version, name)` tuple of strings.
pub(crate) fn class(obj: &Bound<'_, PyAny>) -> PyResult<ClassRef> {
    let (library, version, name) = match obj.cast::<PyClassRef>() {
        Ok(c) => {
            let c = c.get();
            (c.library.clone(), c.version.clone(), c.name.clone())
        }
        Err(_) => obj.extract()?,
    };
    Ok(ClassRef { library, version, name })
}

#[pymethods]
impl PyClassRef {
    #[new]
    fn new(library: String, version: String, name: String) -> Self {
        PyClassRef { library, version, name }
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let fields = [&self.library, &self.version, &self.name];
        PyTuple::new(py, fields)?.try_iter().map(Bound::into_any)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let fields = PyTuple::new(py, [&self.library, &self.version, &self.name])?;
        Ok(format!("ClassRef{}", fields.repr()?))
    }
}

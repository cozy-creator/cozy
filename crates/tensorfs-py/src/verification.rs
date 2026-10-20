use pyo3::prelude::*;

/// The local missing/corrupt objects from a checkpoint verification.
#[pyclass(name = "VerificationReport", module = "tensorfs", frozen, get_all)]
pub(crate) struct PyVerificationReport {
    missing: Vec<(String, u64)>,
    corrupt: Vec<(String, u64)>,
}

impl From<tensorfs::VerificationReport> for PyVerificationReport {
    fn from(report: tensorfs::VerificationReport) -> Self {
        let refs = |objects: Vec<tensorfs::format::ObjectRef>| {
            objects
                .into_iter()
                .map(|object| (object.sha256, object.length))
                .collect()
        };
        Self {
            missing: refs(report.missing),
            corrupt: refs(report.corrupt),
        }
    }
}

#[pymethods]
impl PyVerificationReport {
    #[getter]
    fn ok(&self) -> bool {
        self.missing.is_empty() && self.corrupt.is_empty()
    }
}

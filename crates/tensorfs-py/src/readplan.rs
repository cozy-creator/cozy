//! Read plans: pure planning over a header (no store), for the plane and the store alike.
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tensorfs_core::header::Header;
use tensorfs_core::read::{self, Source as ReadSource};

use crate::errors::IntoPy;

#[pyclass(name = "ReadPlan")]
pub struct PyPlan {
    pub plan: read::ReadPlan,
}

#[pymethods]
impl PyPlan {
    /// Select exact existing part `what` identifiers, retaining all their native windows.
    fn select(&self, py: Python<'_>, whats: Vec<String>) -> PyResult<PyPlan> {
        Ok(PyPlan {
            plan: self.plan.select(&whats).or_refuse(py)?,
        })
    }

    #[getter]
    fn bytes(&self) -> u64 {
        self.plan.bytes
    }
    #[getter]
    fn order(&self) -> &'static str {
        self.plan.order.as_str()
    }
    fn __len__(&self) -> usize {
        self.plan.items.len()
    }

    /// The plan rows, in destination order: what, where it lands, how long, and where the
    /// bytes come from.
    fn items<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyList>> {
        let l = PyList::empty(py);
        for it in &self.plan.items {
            let d = PyDict::new(py);
            d.set_item("what", &it.what)?;
            d.set_item("dest_off", it.dest_off)?;
            d.set_item("len", it.len)?;
            match &it.source {
                ReadSource::Object(r) => {
                    d.set_item("source", "object")?;
                    d.set_item("object", r.obj.id())?;
                    d.set_item("object_length", r.obj.length)?;
                    d.set_item("off", r.off)?;
                }
                ReadSource::Inline(_) => {
                    d.set_item("source", "inline")?;
                }
            }
            l.append(d)?;
        }
        Ok(l)
    }
}

/// Build a read plan from canonical header bytes and the CALLER's traversal. Order is the
/// caller's tensor requirements.
///
/// `components` DECLARES the scope completeness is checked against (#570b): the traversal
/// must name every tensor the header carries for those components and none outside them.
/// Absent, it defaults to every component the header carries — the whole-artifact reading,
/// which is what a caller with no tensor requirements to narrow by means.
#[pyfunction]
#[pyo3(signature = (header, traversal, components=None, window=1<<24))]
pub fn plan(
    py: Python<'_>,
    header: &[u8],
    traversal: Vec<(String, String)>,
    components: Option<Vec<String>>,
    window: u64,
) -> PyResult<PyPlan> {
    let h = Header::from_stored(header).or_refuse(py)?;
    let scope = components.unwrap_or_else(|| h.components.iter().map(|(c, _)| c.clone()).collect());
    let p = read::plan_for_traversal(&h, &traversal, &scope, window).or_refuse(py)?;
    Ok(PyPlan { plan: p })
}


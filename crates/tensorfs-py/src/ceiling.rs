//! `tensorfs.ceiling`: what reads can reach on this machine, measured unless kept this boot.

use std::path::PathBuf;

use pyo3::prelude::*;
use tensorfs::failure::{Failure, Kind};

#[pyclass(name = "Ceiling", module = "tensorfs", frozen, get_all)]
pub struct PyCeiling {
    /// Sequential read GB/s of the volume.
    disk_gbps: f64,
    /// Measured past the page cache (O_DIRECT); else the filesystem refused it.
    disk_direct: bool,
    /// Pinned host-to-device GB/s, when a CUDA device was asked.
    h2d_gbps: Option<f64>,
    /// Streams of the fastest measured H2D probe.
    streams: Option<u32>,
}

/// `path`'s volume (default: the TensorFS home) and, with `device`, pinned H2D to that GPU.
#[pyfunction]
#[pyo3(signature = (device = None, path = None))]
fn ceiling(py: Python<'_>, device: Option<i32>, path: Option<PathBuf>) -> PyResult<PyCeiling> {
    let path = match path {
        Some(path) => path,
        None => tensorfs::home::resolve(None).map_err(|r| crate::errors::raise(py, r.into()))?,
    };
    let c = py.detach(|| tensorfs::io::ceiling(device, &path)).map_err(|e| {
        let kind = if e.kind() == std::io::ErrorKind::Unsupported { Kind::Unsupported } else { Kind::Io };
        crate::errors::raise(py, Failure { kind, detail: e.to_string() })
    })?;
    Ok(PyCeiling { disk_gbps: c.disk_gbps, disk_direct: c.disk_direct, h2d_gbps: c.h2d_gbps, streams: c.streams })
}

pub fn install(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCeiling>()?;
    m.add_function(wrap_pyfunction!(ceiling, m)?)
}

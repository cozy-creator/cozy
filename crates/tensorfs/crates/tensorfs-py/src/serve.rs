//! The owner's half of the executor channels, for tests and tools: what TensorD serves per
//! attempt.

use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use tensorfs_core::err::{Code, Refusal};
use tensorfs_core::transport;

use crate::errors::IntoPy;
use crate::store::PyStore;

/// Serve one fetch channel on `fd` (a connected `cozy_platform::ipc` stream; ownership passes here) from
/// `store`'s fetcher until its peer closes it: what TensorD does for each attempt, for tests
/// and tools. `origins` are the checkpoints the channel may fetch from:
/// `{"hub", "ref", "lane", "manifest", "manifest_length", "credential"?, "allowed_hosts"?,
/// "allow_local"?}`.
#[pyfunction]
#[pyo3(signature = (store, fd, *, origins, run=0))]
pub fn serve_fetch(
    py: Python<'_>,
    store: &PyStore,
    fd: i64,
    origins: Vec<Bound<'_, PyDict>>,
    run: u64,
) -> PyResult<()> {
    use tensorfs_core::fetch_v1::{self as fetch, Access, Origin, Plain};
    let mut granted: Vec<(Origin, std::sync::Arc<dyn Access>)> = vec![];
    for row in origins {
        let text = |key: &str| -> PyResult<String> {
            row.get_item(key)?.map_or(Ok(String::new()), |value| value.extract())
        };
        let hub = text("hub")?;
        let manifest = text("manifest")?;
        let length: u64 = row.get_item("manifest_length")?.map_or(Ok(0), |v| v.extract())?;
        let allowed_hosts: Vec<String> = row.get_item("allowed_hosts")?.map_or(Ok(vec![]), |v| v.extract())?;
        let allow_local: bool = row.get_item("allow_local")?.map_or(Ok(false), |v| v.extract())?;
        let host = transport::base_host(&hub).or_refuse(py)?;
        let credential = transport::credential_from_spec(&text("credential")?, vec![host]).or_refuse(py)?;
        let policy = transport::SourcePolicy { allowed_hosts, allow_local, ..Default::default() };
        let sha256 = manifest.strip_prefix("sha256:").unwrap_or(&manifest).to_string();
        let origin = Origin {
            hub,
            refspec: text("ref")?,
            lane: text("lane")?,
            manifest: tensorfs_core::ids::ObjectRef { sha256, length },
        };
        granted.push((origin, std::sync::Arc::new(Plain { credential, policy })));
    }
    // SAFETY: the caller hands over a connected descriptor it no longer uses.
    let stream = unsafe { cozy_platform::ipc::adopt(fd) };
    let fetcher = fetch::Fetcher::of(&store.store).or_refuse(py)?;
    let grant = fetch::Grant { run, origins: granted, stop: Default::default() };
    py.detach(|| fetch::serve(&fetcher, grant, stream)).or_refuse(py)
}

/// Serve one exact source's channel on `fd` (a connected `cozy_platform::ipc` stream; ownership passes here)
/// until its peer closes it: the owner's half of `SourceCapability.from_fd`, for tests and
/// tools. `check_current` is called before each operation; what it raises ends the channel and
/// is raised here.
#[pyfunction]
#[pyo3(signature = (store, fd, manifest, length, *, check_current=None))]
pub fn serve_source(
    py: Python<'_>,
    store: &PyStore,
    fd: i64,
    manifest: &str,
    length: u64,
    check_current: Option<Py<PyAny>>,
) -> PyResult<()> {
    let sha256 = manifest.strip_prefix("sha256:").unwrap_or(manifest).to_string();
    let source = tensorfs_core::ids::ObjectRef { sha256, length };
    // SAFETY: the caller hands over a connected descriptor it no longer uses.
    let stream = unsafe { cozy_platform::ipc::adopt(fd) };
    let raised: Mutex<Option<PyErr>> = Mutex::new(None);
    let check = || -> Result<(), Refusal> {
        let Some(check) = &check_current else { return Ok(()) };
        Python::attach(|py| match check.bind(py).call0() {
            Ok(_) => Ok(()),
            Err(error) => {
                let detail = error.to_string();
                raised.lock().unwrap().get_or_insert(error);
                Err(Refusal { code: Code::LEASE_REVOKED, detail })
            }
        })
    };
    let served = py.detach(|| {
        tensorfs_core::derived::channel::serve_source(&store.store, &store.meta, &source, stream, &check, None)
    });
    if let Some(error) = raised.lock().unwrap().take() {
        return Err(error);
    }
    served.or_refuse(py)
}

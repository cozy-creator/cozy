//! `tensorfs.ensure`: make one model resident (tensorfs_core::ensure).

use std::path::PathBuf;
use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use tensorfs_core::ensure::{Ensured, Event, Request};
use tensorfs_core::err::{Code, Refusal};
use tensorfs_core::transport;

use crate::errors::IntoPy;
use crate::store::{PyPullCancellation, PyStore};

fn event_dict<'p>(py: Python<'p>, event: &Event) -> PyResult<Bound<'p, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("model", &event.model)?;
    d.set_item("step", &event.step)?;
    d.set_item("phase", event.phase.as_str())?;
    d.set_item("bytes_done", event.bytes_done)?;
    d.set_item("bytes_total", event.bytes_total)?;
    d.set_item("rate", event.rate)?;
    Ok(d)
}

fn result_dict<'p>(py: Python<'p>, r: &Ensured) -> PyResult<Bound<'p, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("model", &r.model)?;
    d.set_item("release", &r.release)?;
    d.set_item("lane", &r.lane)?;
    d.set_item("scope", &r.scope)?;
    d.set_item("manifest", r.manifest.id())?;
    d.set_item("manifest_length", r.manifest.length)?;
    d.set_item("bytes_total", r.bytes_total)?;
    d.set_item("bytes_held", r.bytes_held)?;
    d.set_item("bytes_fetched", r.bytes_fetched)?;
    d.set_item("bytes_cached", r.bytes_cached)?;
    d.set_item("collected_bytes", r.collected_bytes)?;
    d.set_item("joined", r.joined)?;
    d.set_item("seconds", r.seconds)?;
    Ok(d)
}

/// Make one model resident: resolve `ref` at `hub`, share one flight per manifest with
/// every other process on this Store, admit it against the disk (GC when short), and pull
/// with retries earned by measured progress. `progress(event)` sees samples; a raising
/// callback cancels and its exception is re-raised. Refusals raise typed `errors.*`.
#[pyfunction]
#[pyo3(signature = (store, r#ref, *, hub, lane=String::new(), step=String::new(),
       credential=String::new(), ca_file=None, allowed_hosts=Vec::new(), allow_local=false,
       keep=Vec::new(), sample_seconds=None, cancellation=None, progress=None))]
#[allow(clippy::too_many_arguments)]
pub fn ensure<'p>(
    py: Python<'p>,
    store: &PyStore,
    r#ref: &str,
    hub: &str,
    lane: String,
    step: String,
    credential: String,
    ca_file: Option<PathBuf>,
    allowed_hosts: Vec<String>,
    allow_local: bool,
    keep: Vec<String>,
    sample_seconds: Option<f64>,
    cancellation: Option<PyRef<'_, PyPullCancellation>>,
    progress: Option<Py<PyAny>>,
) -> PyResult<Bound<'p, PyDict>> {
    if let Some(path) = &ca_file {
        let pem = std::fs::read(path).map_err(|e| Refusal {
            code: Code::IO_FAILED,
            detail: format!("read {}: {e}", path.display()),
        });
        pem.and_then(|pem| transport::trust_roots(&pem))
            .or_refuse(py)?;
    }
    let host = transport::base_host(hub).or_refuse(py)?;
    let scoped = transport::credential_from_spec(&credential, vec![host]).or_refuse(py)?;
    let policy = transport::SourcePolicy {
        allowed_hosts,
        allow_local,
        max_redirects: 0,
        ..Default::default()
    };
    let token = cancellation.map(|c| c.token.clone()).unwrap_or_default();
    let raised: Mutex<Option<PyErr>> = Mutex::new(None);
    let on_event = |event: &Event| {
        // Python only around the callback, never around transfer I/O.
        Python::attach(|py| {
            let Some(callback) = &progress else { return };
            let outcome = event_dict(py, event).and_then(|d| callback.bind(py).call1((d,)));
            if let Err(error) = outcome {
                raised.lock().unwrap().get_or_insert(error);
                token.cancel();
            }
        })
    };
    let access = std::sync::Arc::new(tensorfs_core::fetch::Plain { credential: scoped, policy });
    let ensured = py.detach(|| {
        let mut request = Request::new(&store.store, hub, r#ref, access);
        request.lane = &lane;
        request.step = &step;
        request.keep = &keep;
        request.cancellation = Some(token.clone());
        if let Some(sample) = sample_seconds {
            request.sample_seconds = sample;
        }
        if progress.is_some() {
            request.on_event = Some(&on_event);
        }
        tensorfs_core::ensure::ensure(&request)
    });
    if let Some(error) = raised.lock().unwrap().take() {
        return Err(error);
    }
    result_dict(py, &ensured.or_refuse(py)?)
}

/// One pressure pass on `store`: with no more than the reserve free, the GC policy frees
/// what lifts it above; `keep` is never evicted.
#[pyfunction]
#[pyo3(signature = (store, *, keep=Vec::new()))]
pub fn relieve<'p>(
    py: Python<'p>,
    store: &PyStore,
    keep: Vec<String>,
) -> PyResult<Bound<'p, PyDict>> {
    let relief = py
        .detach(|| tensorfs_core::ensure::relieve(&store.store, &keep))
        .or_refuse(py)?;
    let d = PyDict::new(py);
    d.set_item("pressure", relief.pressure)?;
    d.set_item("collected_bytes", relief.collected_bytes)?;
    d.set_item("capacity_bytes", relief.capacity_bytes)?;
    d.set_item("available_bytes", relief.available_bytes)?;
    d.set_item("unable", relief.unable)?;
    Ok(d)
}

/// Serve one fetch channel on `fd` (a connected Unix stream; ownership passes here) from
/// `store`'s fetcher until its peer closes it: what TensorD does for each attempt, for tests
/// and tools. `origins` are the checkpoints the channel may fetch from:
/// `{"hub", "ref", "lane", "manifest", "manifest_length", "credential"?, "allowed_hosts"?,
/// "allow_local"?}`.
#[pyfunction]
#[pyo3(signature = (store, fd, *, origins, run=0))]
pub fn serve_fetch(
    py: Python<'_>,
    store: &PyStore,
    fd: i32,
    origins: Vec<Bound<'_, PyDict>>,
    run: u64,
) -> PyResult<()> {
    use std::os::fd::FromRawFd;
    use tensorfs_core::fetch::{self, Access, Origin, Plain};
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
    let stream = unsafe { std::os::unix::net::UnixStream::from_raw_fd(fd) };
    let fetcher = fetch::Fetcher::of(&store.store).or_refuse(py)?;
    let grant = fetch::Grant { run, origins: granted, stop: Default::default() };
    py.detach(|| fetch::serve(&fetcher, grant, stream)).or_refuse(py)
}

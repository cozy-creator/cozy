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
    d.set_item("attempt", event.attempt)?;
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
    d.set_item("cache_written_bytes", r.cache_written_bytes)?;
    d.set_item("collected_bytes", r.collected_bytes)?;
    d.set_item("attempts", r.attempts)?;
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
       keep=Vec::new(), streams=transport::PULL_STREAMS, streams_start=transport::STREAMS,
       sample_seconds=None, cancellation=None, progress=None))]
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
    streams: usize,
    streams_start: usize,
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
    let ensured = py.detach(|| {
        let mut request = Request::new(&store.store, hub, r#ref, &scoped, &policy);
        request.lane = &lane;
        request.step = &step;
        request.keep = &keep;
        request.cancellation = Some(token.clone());
        request.streams = streams;
        // Accepted for callers written before the chunked downloader; it has no ramp.
        let _ = streams_start;
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

/// One pressure pass on `store`: at most 1/10 of its filesystem free (or under the
/// reserve), run the GC policy until more than 1/5 is free; `keep` is never evicted.
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

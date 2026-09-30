//! tensorfs-py — the marshal boundary (tensorfs.md §10, tfs-007).
//!
//! The compiled half owns every byte-level fact: hashing, canonical bytes, geometry,
//! verification, leases and the reader pool. Python owns typing and ergonomics and nothing
//! else, which `scripts/fence.py` enforces rather than merely asking for.
//!
//! Torch stays OUTSIDE this boundary. Nothing here imports it, links it or knows it exists:
//! a consumer gets a filled buffer plus dtype/shape/geometry as plain data and calls
//! `torch.frombuffer` itself. The dtype table crosses exactly once, from `dtypes()`.

mod docs;
mod ensure;
mod errors;
mod ingest;
mod plane;
mod store;

use pyo3::prelude::*;
use tensorfs_core::err::{refuse, Code, Result as TResult};

fn python_distribution_version() -> String {
    let cargo_version = env!("CARGO_PKG_VERSION");
    let Some((release, development)) = cargo_version.split_once("-dev.") else {
        return cargo_version.to_string();
    };
    format!("{release}.dev{development}")
}

/// The supported platforms, checked against the platform the interpreter reports.
///
/// Windows has no `fcntl`, and this store's hold liveness IS an OS fact — an exclusive lock
/// the kernel drops when the owner dies, SIGKILL included (tfs-006). A Windows build without
/// that replacement would have to fall back to a wall clock, which is exactly the v1 defect
/// that stranded roots forever. So the import refuses and names the supported set, rather
/// than importing into a store whose `Taken | Absent | Deferred` contract it cannot honour.
///
/// It is a FUNCTION taking the platform name so the refusal is observable from any platform:
/// a gate nobody can fire is a gate nobody has checked.
fn platform_gate(platform: &str) -> TResult<()> {
    const SUPPORTED: &[&str] = &["linux", "darwin", "freebsd"];
    if SUPPORTED.contains(&platform) {
        return Ok(());
    }
    refuse(
        Code::PLATFORM_UNSUPPORTED,
        format!(
            "{platform:?} is not supported: hold liveness is an advisory-lock fact the kernel \
             releases on process death, and the POSIX-lock replacement has not landed. \
             Supported: {SUPPORTED:?}"
        ),
    )
}

#[pyfunction(name = "platform_gate")]
fn py_platform_gate(py: Python<'_>, platform: &str) -> PyResult<()> {
    platform_gate(platform).map_err(|e| errors::to_py(py, &e))
}

/// Process-wide verify-once counters (proto-061); diff two snapshots around the work.
#[pyfunction]
fn stats(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    let now = tensorfs_core::stats::snapshot();
    let result = pyo3::types::PyDict::new(py);
    result.set_item("catalog_opens", now.catalog_opens)?;
    result.set_item("presence_passes", now.presence_passes)?;
    result.set_item("repo_cache_verified_reads", now.repo_cache_verified_reads)?;
    result.set_item("admission_batches", now.admission_batches)?;
    result.set_item("admitted_objects", now.admitted_objects)?;
    result.set_item("admission_nanos", now.admission_nanos)?;
    result.set_item("syncs", now.syncs)?;
    result.set_item("sync_nanos", now.sync_nanos)?;
    result.set_item("recovery_rehashed", now.recovery_rehashed)?;
    result.set_item("recovery_removed", now.recovery_removed)?;
    result.set_item("recovery_nanos", now.recovery_nanos)?;
    Ok(result)
}

#[pyfunction(name = "home", signature = (root = None))]
fn py_home(py: Python<'_>, root: Option<std::path::PathBuf>) -> PyResult<String> {
    tensorfs_core::home::resolve(root.as_deref())
        .map(|path| path.display().to_string())
        .map_err(|error| errors::to_py(py, &error))
}

#[pymodule]
fn _ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    errors::install(m)?;

    // The gate runs at import, against the real platform, through the same function the
    // arm calls.
    let plat: String = py
        .import("sys")?
        .getattr("platform")?
        .extract()
        .unwrap_or_else(|_| "unknown".to_string());
    let plat = plat
        .trim_end_matches(|c: char| c.is_ascii_digit())
        .to_string();
    platform_gate(&plat).map_err(|e| errors::to_py(py, &e))?;

    m.add("__version__", python_distribution_version())?;
    m.add_function(wrap_pyfunction!(py_platform_gate, m)?)?;
    m.add_function(wrap_pyfunction!(py_home, m)?)?;
    m.add_function(wrap_pyfunction!(stats, m)?)?;

    m.add_function(wrap_pyfunction!(docs::object_id, m)?)?;
    m.add_function(wrap_pyfunction!(docs::parse_header, m)?)?;
    m.add_function(wrap_pyfunction!(docs::parse_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(docs::py_fit, m)?)?;
    m.add_function(wrap_pyfunction!(docs::capability, m)?)?;
    m.add_function(wrap_pyfunction!(docs::run_doc, m)?)?;
    m.add_function(wrap_pyfunction!(docs::receipt_binds, m)?)?;
    m.add_function(wrap_pyfunction!(docs::doc_kinds, m)?)?;
    m.add_function(wrap_pyfunction!(docs::platform_digests, m)?)?;
    m.add_function(wrap_pyfunction!(docs::seed_digests, m)?)?;
    m.add_function(wrap_pyfunction!(docs::dtypes, m)?)?;
    m.add_function(wrap_pyfunction!(docs::manifest_max_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(docs::render, m)?)?;
    m.add_function(wrap_pyfunction!(docs::recanonicalize, m)?)?;
    m.add_class::<docs::PyClosure>()?;

    m.add_function(wrap_pyfunction!(ingest::read_source_heads, m)?)?;
    m.add_function(wrap_pyfunction!(ingest::select_source_profile, m)?)?;
    m.add_function(wrap_pyfunction!(ingest::source_profile_converters, m)?)?;
    m.add_function(wrap_pyfunction!(ingest::prepare_selected_source, m)?)?;
    m.add_function(wrap_pyfunction!(ingest::record_model_source_custody, m)?)?;
    m.add_function(wrap_pyfunction!(ingest::model_source_objects, m)?)?;
    m.add_function(wrap_pyfunction!(store::plan, m)?)?;
    m.add_function(wrap_pyfunction!(store::gc, m)?)?;
    m.add_function(wrap_pyfunction!(store::transfer_object, m)?)?;
    m.add_function(wrap_pyfunction!(store::transfer_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(store::fetch_plan, m)?)?;
    m.add_function(wrap_pyfunction!(store::fetch_admit, m)?)?;
    m.add_function(wrap_pyfunction!(store::fetch_complete, m)?)?;
    m.add_function(wrap_pyfunction!(store::fetch_complete_cozytensors, m)?)?;
    m.add_class::<store::PyPullCancellation>()?;
    m.add_function(wrap_pyfunction!(store::pull, m)?)?;
    m.add_function(wrap_pyfunction!(ensure::ensure, m)?)?;
    m.add_function(wrap_pyfunction!(ensure::relieve, m)?)?;
    m.add("CAPABILITIES", tensorfs_core::CAPABILITIES.to_vec())?;
    m.add_function(wrap_pyfunction!(store::transfer_streams, m)?)?;
    m.add("STREAM_MEMORY", tensorfs_core::transport::STREAM_MEMORY)?;
    m.add("AS_IS_PROFILE", tensorfs_core::ingest::source::AS_IS)?;
    m.add_class::<store::PyRepoObjectCache>()?;
    m.add_class::<store::PyStore>()?;
    m.add_class::<store::PyOperation>()?;
    m.add_class::<store::PyDerivedWriter>()?;
    m.add_class::<store::PyLease>()?;
    m.add_class::<store::PyPlan>()?;
    m.add_class::<store::PyBatch>()?;
    plane::install(m)?;
    Ok(())
}

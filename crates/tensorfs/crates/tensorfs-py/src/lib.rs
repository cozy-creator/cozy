//! tensorfs-py — the marshal boundary (tensorfs.md §10, tfs-007).
//!
//! The compiled half owns every byte-level fact: hashing, canonical bytes, geometry,
//! verification and leases. Python owns typing and ergonomics and nothing else, which
//! `scripts/fence.py` enforces rather than merely asking for.
//!
//! Torch stays OUTSIDE this boundary. Nothing here imports it, links it or knows it exists:
//! a consumer gets a filled buffer plus dtype/shape/geometry as plain data and calls
//! `torch.frombuffer` itself. The dtype table crosses exactly once, from `dtypes()`.

mod docs;
mod errors;
#[cfg_attr(windows, path = "plane_windows.rs")]
mod plane;
mod readplan;
#[cfg(feature = "store")]
mod serve;
#[cfg(feature = "store")]
mod store;
#[cfg(feature = "store")]
mod v2;

use pyo3::prelude::*;

fn python_distribution_version() -> String {
    let cargo_version = env!("CARGO_PKG_VERSION");
    let Some((release, development)) = cargo_version.split_once("-dev.") else {
        return cargo_version.to_string();
    };
    format!("{release}.dev{development}")
}

#[pymodule]
fn _ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // The store's warnings reach the executor's stderr, which its machine keeps.
    let _ = tracing_subscriber::fmt().with_writer(std::io::stderr).try_init();
    errors::install(m)?;

    m.add("__version__", python_distribution_version())?;
    // A plane-only build (no `store` feature) carries documents, headers, read plans, the fit
    // function and the weight plane: it cannot open, read or write a store.
    m.add("STORE", cfg!(feature = "store"))?;
    // The Store contract this extension shares with the `cozy` binary's TensorD.
    m.add("STORE_FORMAT", tensorfs_core::STORE_FORMAT)?;
    m.add("CLAIM_PROTOCOL", tensorfs_core::CLAIM_PROTOCOL)?;

    m.add_function(wrap_pyfunction!(docs::object_id, m)?)?;
    m.add_function(wrap_pyfunction!(docs::parse_header, m)?)?;
    m.add_function(wrap_pyfunction!(docs::py_fit, m)?)?;
    m.add_function(wrap_pyfunction!(docs::seed_digests, m)?)?;
    m.add_function(wrap_pyfunction!(docs::dtypes, m)?)?;
    m.add_function(wrap_pyfunction!(docs::manifest_max_bytes, m)?)?;
    m.add_class::<docs::PyClosure>()?;
    m.add_function(wrap_pyfunction!(readplan::plan, m)?)?;
    m.add_class::<readplan::PyPlan>()?;
    #[cfg(feature = "store")]
    store_api(m)?;
    plane::install(m)?;
    Ok(())
}

/// The store, derived-writer and GC bindings.
#[cfg(feature = "store")]
fn store_api(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(store::gc, m)?)?;
    m.add_function(wrap_pyfunction!(serve::serve_fetch, m)?)?;
    m.add_function(wrap_pyfunction!(serve::serve_source, m)?)?;
    m.add_class::<store::PyStore>()?;
    m.add_class::<store::PyOperation>()?;
    m.add_class::<store::PyDerivedWriter>()?;
    m.add_class::<store::PyLease>()?;
    v2::install(m)
}

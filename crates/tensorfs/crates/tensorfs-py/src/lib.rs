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
mod plane;
mod readplan;
#[cfg(feature = "store")]
mod serve;
#[cfg(feature = "store")]
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
/// It is a FUNCTION taking the platform name so the refusal is observable from any platform
/// (the test below): a gate nobody can fire is a gate nobody has checked.
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

#[pymodule]
fn _ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    errors::install(m)?;

    // The gate runs at import, against the real platform.
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
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_platform_gate_refuses_an_unsupported_platform_by_name() {
        assert!(super::platform_gate("linux").is_ok());
        let refusal = super::platform_gate("win32").unwrap_err();
        assert_eq!(refusal.code, tensorfs_core::err::Code::PLATFORM_UNSUPPORTED);
        assert!(refusal.detail.contains("\"win32\""), "{}", refusal.detail);
    }
}

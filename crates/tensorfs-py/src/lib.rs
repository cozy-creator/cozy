//! tensorfs-py: the `tensorfs` wheel's compiled half. It marshals; every byte-level fact comes
//! from `tensorfs-core` and `tensorfs-plane`. Torch stays outside: tensors cross as DLPack.

mod docs;
mod errors;
#[cfg_attr(windows, path = "varena_windows.rs")]
mod varena;
mod readplan;
#[cfg(feature = "store")]
mod store;
#[cfg(feature = "store")]
mod tensors;

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
    // The library's warnings reach the process's stderr.
    let _ = tracing_subscriber::fmt().with_writer(std::io::stderr).try_init();
    errors::install(m)?;
    m.add("__version__", python_distribution_version())?;
    // The store format and plane claim protocol the `cozy` binary must share (format_contract.py).
    m.add("STORE_FORMAT", tensorfs_core::STORE_FORMAT)?;
    m.add("CLAIM_PROTOCOL", tensorfs_core::CLAIM_PROTOCOL)?;
    // The CBOR header codec's Python projection (owner ruling, 2026-10-18).
    m.add_function(wrap_pyfunction!(docs::parse_header, m)?)?;
    m.add_class::<docs::PyClosure>()?;
    // The v1 read plan the plane registers from, until varena registers checkpoints (U8).
    m.add_function(wrap_pyfunction!(readplan::plan, m)?)?;
    m.add_class::<readplan::PyPlan>()?;
    #[cfg(feature = "store")]
    store::install(m)?;
    varena::install(m)
}

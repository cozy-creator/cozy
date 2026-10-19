//! tensorfs-py: the `tensorfs` wheel's compiled half. It marshals; every byte-level fact comes
//! from `tensorfs` and `tensorfs-plane`. Torch stays outside: tensors cross as DLPack.

mod cleanup;
mod errors;
#[cfg_attr(windows, path = "varena_windows.rs")]
mod varena;
#[cfg(feature = "store")]
mod store;
#[cfg(feature = "store")]
mod tensors;
#[cfg(feature = "store")]
mod credentials;
#[cfg(feature = "store")]
mod dlpack_source;

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
    cleanup::init().map_err(|error| errors::raise(m.py(), error.into()))?;
    m.add("__version__", python_distribution_version())?;
    // The store format and plane claim protocol the `cozy` binary must share (format_contract.py).
    m.add("STORE_FORMAT", tensorfs::STORE_FORMAT)?;
    m.add("CLAIM_PROTOCOL", tensorfs::CLAIM_PROTOCOL)?;
    #[cfg(feature = "store")]
    store::install(m)?;
    #[cfg(feature = "store")]
    credentials::install(m)?;
    varena::install(m)
}
